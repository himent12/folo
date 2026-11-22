//! ACP v1 protocol helpers, tool-isolation setup, model/effort/mode resolution, usage merging, and error extraction.

use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine;
use folo_api::{LabelEffort, LabelUsage};
use serde_json::{Value, json};

use super::split_effort_suffix;
use crate::label::agent::{AgentError, AgentRequest, Failure};

/// Custom workspace agent name written into each labeling call's scratch directory.
pub(super) const LABEL_AGENT_NAME: &str = "folo-labeler";
/// Isolated CLI settings subdirectory under `<home>/.gemini/antigravity-cli/`.
pub(super) const LABEL_CLI_DIR: &str = "folo-label";

const LABEL_AGENT_MD: &str = "---\n\
name: folo-labeler\n\
description: FOLO single-turn image labeler with native tools and customizations disabled.\n\
mainAgent: true\n\
subagent: false\n\
tools:\n\
  - send_message\n\
commandExecutionPolicy: \"off\"\n\
inheritMcp: false\n\
inheritCustomizations: false\n\
---\n\
Do not call any tools or read any files; every image is already attached inline. Reply with one JSON object and \
nothing else.\n";

const STRICT_SETTINGS_JSON: &str = r#"{
  "agentMode": "",
  "toolPermission": "strict",
  "allowNonWorkspaceAccess": false,
  "enableTerminalSandbox": true,
  "permissions": {
    "allow": [],
    "ask": [],
    "deny": [
      "read_file(*)",
      "write_file(*)",
      "command(*)",
      "unsandboxed(*)",
      "read_url(*)",
      "execute_url(*)",
      "download_url(*)",
      "mcp(*)"
    ]
  }
}
"#;

/// Serializes creation/refresh of `<home>/.gemini/antigravity-cli/folo-label/settings.json` across parallel calls so
/// concurrent Windows rewrites never hit sharing violations.
static SETTINGS_LOCK: Mutex<()> = Mutex::new(());
static SETTINGS_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Prepared on-disk isolation for one `agy acp` labeling call.
pub(super) struct Isolation {
    /// Absolute empty scratch workspace containing `.agents/agents/folo-labeler/agent.md`.
    pub work: PathBuf,
    /// Absolute path to the per-call ACP log file inside the scratch directory.
    pub log_file: PathBuf,
}

/// Resolves a relative executable path containing directory separators (such as `./bin/agy`) against the current
/// directory before `Command::current_dir` switches into the scratch workspace. Bare names (`agy`) stay untouched so
/// OS `PATH` lookup still applies.
pub(super) fn resolve_executable(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let has_dir = path.components().count() > 1
        || matches!(path.components().next(), Some(Component::CurDir | Component::ParentDir));
    if !has_dir {
        return path.to_path_buf();
    }
    let joined = match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => return path.to_path_buf(),
    };
    strip_verbatim_prefix(std::fs::canonicalize(&joined).unwrap_or(joined))
}

/// Returns a clean absolute path for an existing directory, stripping Windows' `\\?\` verbatim prefix.
pub(super) fn abs_dir(path: &Path) -> std::io::Result<PathBuf> {
    let abs = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    Ok(strip_verbatim_prefix(std::fs::canonicalize(&abs).unwrap_or(abs)))
}

fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    if let Some(s) = path.to_str()
        && let Some(stripped) = s.strip_prefix(r"\\?\")
        && !stripped.starts_with("UNC\\")
    {
        return PathBuf::from(stripped);
    }
    path
}

/// Resolves the user home directory (`USERPROFILE` on Windows, `HOME` on Unix) where `agy` looks for `.gemini/`.
pub(super) fn user_home() -> Option<PathBuf> {
    #[cfg(windows)]
    let raw = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    #[cfg(not(windows))]
    let raw = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    raw.filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Prepares the scratch workspace custom agent (`folo-labeler`) and the isolated strict deny-all CLI settings file
/// (`<home>/.gemini/antigravity-cli/folo-label/settings.json`). Fails closed if either restriction cannot be written.
pub(super) fn prepare_isolation(scratch: &Path, home: Option<&Path>) -> Result<Isolation, AgentError> {
    let work_raw = scratch.join("work");
    std::fs::create_dir_all(&work_raw)
        .map_err(|e| AgentError::Unavailable(format!("preparing Antigravity scratch workspace: {e}")))?;
    let work = abs_dir(&work_raw)
        .map_err(|e| AgentError::Unavailable(format!("resolving Antigravity scratch workspace: {e}")))?;
    let scratch_abs = abs_dir(scratch)
        .map_err(|e| AgentError::Unavailable(format!("resolving Antigravity scratch directory: {e}")))?;

    let agent_dir = work.join(".agents").join("agents").join(LABEL_AGENT_NAME);
    std::fs::create_dir_all(&agent_dir)
        .map_err(|e| AgentError::Unavailable(format!("preparing Antigravity restricted agent: {e}")))?;
    std::fs::write(agent_dir.join("agent.md"), LABEL_AGENT_MD)
        .map_err(|e| AgentError::Unavailable(format!("writing Antigravity restricted agent: {e}")))?;

    let owned_home;
    let home_dir = match home {
        Some(h) => h,
        None => {
            owned_home = user_home().ok_or_else(|| {
                AgentError::Unavailable(
                    "cannot establish isolated Antigravity permissions: user home directory is unset".into(),
                )
            })?;
            &owned_home
        }
    };
    write_strict_settings(home_dir)?;

    Ok(Isolation { work, log_file: scratch_abs.join("agy-acp.log") })
}

fn write_strict_settings(home: &Path) -> Result<(), AgentError> {
    let _guard = SETTINGS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = home.join(".gemini").join("antigravity-cli").join(LABEL_CLI_DIR);
    std::fs::create_dir_all(&dir)
        .map_err(|e| AgentError::Unavailable(format!("creating isolated Antigravity settings dir: {e}")))?;
    let path = dir.join("settings.json");
    if std::fs::read_to_string(&path).is_ok_and(|existing| existing == STRICT_SETTINGS_JSON) {
        return Ok(());
    }
    let tmp = dir.join(format!(
        "settings.{}.{}.tmp",
        std::process::id(),
        SETTINGS_TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, STRICT_SETTINGS_JSON)
        .map_err(|e| AgentError::Unavailable(format!("writing isolated Antigravity settings: {e}")))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        // Windows `rename` can fail if the destination already exists: write directly and remove the temp file.
        let _ = std::fs::remove_file(&tmp);
        std::fs::write(&path, STRICT_SETTINGS_JSON).map_err(|w| {
            AgentError::Unavailable(format!("saving isolated Antigravity settings ({e}; fallback: {w})"))
        })?;
    }
    Ok(())
}

/// A parsed ACP `SessionConfigOption` with flattened select values (`(value, name)`).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ConfigOption {
    pub id: String,
    pub category: Option<String>,
    pub current_value: Option<String>,
    pub values: Vec<(String, String)>,
}

pub(super) fn parse_config_options(list: Option<&Value>) -> Vec<ConfigOption> {
    let Some(items) = list.and_then(Value::as_array) else { return Vec::new() };
    let mut out = Vec::new();
    for item in items {
        let Some(id) = item["id"].as_str() else { continue };
        let category = item["category"].as_str().map(str::to_string);
        let current_value = item["currentValue"].as_str().map(str::to_string);
        let mut values = Vec::new();
        if let Some(opts) = item["options"].as_array() {
            for opt in opts {
                if let Some(group_opts) = opt["options"].as_array() {
                    for g in group_opts {
                        if let Some(v) = g["value"].as_str() {
                            values.push((v.to_string(), g["name"].as_str().unwrap_or(v).to_string()));
                        }
                    }
                } else if let Some(v) = opt["value"].as_str() {
                    values.push((v.to_string(), opt["name"].as_str().unwrap_or(v).to_string()));
                }
            }
        }
        out.push(ConfigOption { id: id.to_string(), category, current_value, values });
    }
    out
}

pub(super) fn find_model_option(options: &[ConfigOption]) -> Option<&ConfigOption> {
    options
        .iter()
        .find(|o| o.category.as_deref() == Some("model"))
        .or_else(|| options.iter().find(|o| o.id.eq_ignore_ascii_case("model")))
}

pub(super) fn find_effort_option(options: &[ConfigOption]) -> Option<&ConfigOption> {
    options.iter().find(|o| o.category.as_deref() == Some("thought_level")).or_else(|| {
        options.iter().find(|o| {
            matches!(o.id.to_ascii_lowercase().as_str(), "thought_level" | "thinking_level" | "effort")
        })
    })
}

pub(super) fn find_mode_option(options: &[ConfigOption]) -> Option<&ConfigOption> {
    options
        .iter()
        .find(|o| o.category.as_deref() == Some("mode"))
        .or_else(|| options.iter().find(|o| o.id.eq_ignore_ascii_case("mode")))
}

/// ACP session mode ids explicitly known to keep tool execution restricted / gated.
const RESTRICTED_MODES: &[&str] =
    &["strict", "default", "request-review", "request_review", "ask", "read-only", "read_only", "plan"];

/// Returns true only for explicitly allowlisted restricted modes; unknown or permissive modes fail closed.
pub(super) fn is_restricted_mode(mode: &str) -> bool {
    let norm = mode.trim().to_ascii_lowercase();
    RESTRICTED_MODES.contains(&norm.as_str())
}

/// Picks the strictest allowlisted restricted execution mode from `values`, or `None` if none is offered.
pub(super) fn pick_safe_mode(values: &[(String, String)]) -> Option<String> {
    for preferred in RESTRICTED_MODES {
        if let Some((v, _)) = values.iter().find(|(v, _)| v.trim().eq_ignore_ascii_case(preferred)) {
            return Some(v.clone());
        }
    }
    None
}

pub(super) fn match_option_value<'a>(values: &'a [(String, String)], query: &str) -> Option<&'a str> {
    if let Some((v, _)) = values.iter().find(|(v, _)| v == query) {
        return Some(v.as_str());
    }
    if let Some((v, _)) = values.iter().find(|(v, _)| v.eq_ignore_ascii_case(query)) {
        return Some(v.as_str());
    }
    if let Some((v, _)) = values.iter().find(|(_, name)| name.eq_ignore_ascii_case(query)) {
        return Some(v.as_str());
    }
    None
}

/// Result of resolving a caller's `(model, effort)` against the session's advertised `model` config option.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ModelResolution {
    pub model: Option<String>,
    pub desired_effort: String,
    /// An effort suffix was stripped from the requested model slug so it must be applied via `thought_level`.
    pub stripped_suffix: bool,
    /// The requested model slug matched an option directly and already encodes the effort suffix.
    pub suffix_on_model: bool,
}

/// Resolves the requested model string and effort against the session's `model` config option.
/// Supports both base ids (`gemini-4-neon`) and effort-suffixed slugs (`gemini-4-neon-xhigh`), preserving suffix
/// effort even when the suffixed slug itself appears verbatim as a model option.
pub(super) fn resolve_model_and_effort(
    requested_model: Option<&str>,
    req_effort: LabelEffort,
    model_opt: Option<&ConfigOption>,
) -> ModelResolution {
    let default_effort = req_effort.as_str().to_string();
    let Some(raw) = requested_model.map(str::trim).filter(|m| !m.is_empty()) else {
        return ModelResolution {
            model: None,
            desired_effort: default_effort,
            stripped_suffix: false,
            suffix_on_model: false,
        };
    };
    let values = model_opt.map(|o| o.values.as_slice()).unwrap_or(&[]);
    if let Some(exact) = match_option_value(values, raw) {
        let (desired_effort, suffix_on_model) = match split_effort_suffix(raw) {
            Some((_, suffix)) => (suffix.to_string(), true),
            None => (default_effort, false),
        };
        return ModelResolution {
            model: Some(exact.to_string()),
            desired_effort,
            stripped_suffix: false,
            suffix_on_model,
        };
    }
    if let Some((base, suffix)) = split_effort_suffix(raw) {
        if let Some(matched) = match_option_value(values, base) {
            return ModelResolution {
                model: Some(matched.to_string()),
                desired_effort: suffix.to_string(),
                stripped_suffix: true,
                suffix_on_model: false,
            };
        }
        if values.is_empty() {
            return ModelResolution {
                model: Some(base.to_string()),
                desired_effort: suffix.to_string(),
                stripped_suffix: true,
                suffix_on_model: false,
            };
        }
    }
    ModelResolution {
        model: Some(raw.to_string()),
        desired_effort: default_effort,
        stripped_suffix: false,
        suffix_on_model: false,
    }
}

/// Extracts legacy `session/new.models.availableModels` entries as `(modelId, name)` pairs.
pub(super) fn parse_legacy_models(models: &Value) -> Vec<(String, String)> {
    models["availableModels"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m["modelId"].as_str()?;
                    let name = m["name"].as_str().unwrap_or(id);
                    Some((id.to_string(), name.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Maps a desired effort string strictly onto an exact (or documented same-tier equivalent) `thought_level` value.
/// Never crosses reasoning tiers or falls back to the first option so a run never spends tokens at a different
/// effort than its record claims.
pub(super) fn resolve_effort_value(desired: &str, opt: &ConfigOption) -> Result<String, String> {
    if opt.values.is_empty() {
        return Ok(desired.to_string());
    }
    if let Some(exact) = match_option_value(&opt.values, desired) {
        return Ok(exact.to_string());
    }
    // Only `low` and `minimal` share the `LabelEffort::Low` tier; all other tiers require an exact match.
    let equivalent: &[&str] = match desired.to_ascii_lowercase().as_str() {
        "low" => &["minimal"],
        "minimal" => &["low"],
        _ => &[],
    };
    for cand in equivalent {
        if let Some(v) = match_option_value(&opt.values, cand) {
            return Ok(v.to_string());
        }
    }
    let offered: Vec<&str> = opt.values.iter().map(|(v, _)| v.as_str()).collect();
    Err(format!(
        "`agy acp` does not offer reasoning effort `{desired}` for this model (available: {})",
        offered.join(", ")
    ))
}

/// Accumulates billed token and USD cost readings across `usage_update` notifications and the final
/// `PromptResponse`. Context-window occupancy (`usage_update.used`) is never treated as billed input, and partial
/// updates merge without erasing previously reported fields.
#[derive(Default)]
pub(super) struct UsageTracker {
    input_tokens: Option<u64>,
    cached_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    thought_tokens: Option<u64>,
    total_tokens: Option<u64>,
    cost_usd: Option<f64>,
}

impl UsageTracker {
    pub(super) fn seen(&self) -> bool {
        self.input_tokens.is_some()
            || self.cached_input_tokens.is_some()
            || self.output_tokens.is_some()
            || self.thought_tokens.is_some()
            || self.cost_usd.is_some()
    }

    pub(super) fn observe_usage_update(&mut self, update: &Value) {
        self.observe_cost(&update["cost"]);
        self.observe_usage_obj(update);
        if let Some(meta) = update.get("_meta") {
            self.observe_usage_obj(meta);
            if let Some(u) = meta.get("usage") {
                self.observe_usage_obj(u);
            }
            if let Some(u) = meta.get("antigravity").and_then(|a| a.get("usage")) {
                self.observe_usage_obj(u);
            }
        }
    }

    pub(super) fn observe_prompt_result(&mut self, result: &Value) {
        if let Some(u) = result.get("usage") {
            self.observe_usage_obj(u);
        }
        self.observe_cost(&result["cost"]);
        if let Some(meta) = result.get("_meta") {
            if let Some(u) = meta.get("usage") {
                self.observe_usage_obj(u);
            }
            if let Some(u) = meta.get("antigravity").and_then(|a| a.get("usage")) {
                self.observe_usage_obj(u);
            }
            if let Some(c) = meta.get("cost") {
                self.observe_cost(c);
            }
        }
    }

    fn observe_cost(&mut self, cost: &Value) {
        if cost["currency"].as_str().is_some_and(|c| c.eq_ignore_ascii_case("USD"))
            && let Some(amount) = cost["amount"].as_f64()
            && amount.is_finite()
            && amount >= 0.0
        {
            self.cost_usd = Some(amount);
        }
    }

    fn observe_usage_obj(&mut self, obj: &Value) {
        let read = |a: &str, b: &str| obj[a].as_u64().or_else(|| obj[b].as_u64());
        if let Some(v) = read("inputTokens", "input_tokens") {
            self.input_tokens = Some(v);
        }
        if let Some(v) = read("outputTokens", "output_tokens") {
            self.output_tokens = Some(v);
        }
        if let Some(v) = read("cachedReadTokens", "cached_read_tokens")
            .or_else(|| read("cachedInputTokens", "cached_input_tokens"))
        {
            self.cached_input_tokens = Some(v);
        }
        if let Some(v) = read("thoughtTokens", "thought_tokens")
            .or_else(|| read("thinkingTokens", "thinking_tokens"))
            .or_else(|| read("reasoningOutputTokens", "reasoning_output_tokens"))
        {
            self.thought_tokens = Some(v);
        }
        if let Some(v) = read("totalTokens", "total_tokens") {
            self.total_tokens = Some(v);
        }
    }

    pub(super) fn to_usage(&self, calls: u64) -> LabelUsage {
        let seen = self.seen();
        if !seen && calls == 0 {
            return LabelUsage::default();
        }
        let raw_in = self.input_tokens.unwrap_or(0);
        let raw_cached = self.cached_input_tokens.unwrap_or(0);
        let raw_out = self.output_tokens.unwrap_or(0);
        let raw_thought = self.thought_tokens.unwrap_or(0);
        // `LabelUsage::input_tokens` includes cached reads, and `output_tokens` includes reasoning/thinking tokens.
        let total_in = if raw_cached > raw_in { raw_in + raw_cached } else { raw_in };
        let total_out = match self.total_tokens {
            Some(t) if raw_thought > 0 && raw_in + raw_out + raw_thought == t => raw_out + raw_thought,
            Some(t) if raw_in + raw_out == t => raw_out,
            _ => raw_out + raw_thought,
        };
        LabelUsage {
            calls: if seen { calls.max(1) } else { calls },
            input_tokens: total_in,
            cached_input_tokens: raw_cached.min(total_in),
            output_tokens: total_out,
            cost_usd: self.cost_usd,
        }
    }
}

/// Extracts a [`Failure`] (full message including structured provider status/details + optional HTTP status) from a
/// JSON-RPC `error` object.
pub(super) fn rpc_failure(err: &Value, used: LabelUsage) -> Failure {
    let mut message = err["message"].as_str().unwrap_or("ACP request failed").trim().to_string();
    let mut append = |part: &str| {
        let part = part.trim();
        if !part.is_empty() && !message.contains(part) {
            if !message.is_empty() {
                message.push_str(": ");
            }
            message.push_str(part);
        }
    };
    let mut status: Option<u16> = None;
    if let Some(data) = err.get("data") {
        for text in [
            data["error"]["status"].as_str(),
            data["status"].as_str(),
            data["error"]["reason"].as_str(),
            data["reason"].as_str(),
            data.as_str(),
            data["message"].as_str(),
            data["details"].as_str(),
            data["detail"].as_str(),
            data["error"]["message"].as_str(),
        ]
        .into_iter()
        .flatten()
        {
            append(text);
        }
        // Google RPC error `details` arrays can carry `reason`, `message`, or `retryDelay` (e.g. `15s`).
        for arr in [data["error"]["details"].as_array(), data["details"].as_array()].into_iter().flatten() {
            for item in arr {
                for key in ["reason", "message"] {
                    if let Some(s) = item[key].as_str() {
                        append(s);
                    }
                }
                if let Some(delay) = item["retryDelay"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
                    append(&format!("retry after {delay}"));
                }
            }
        }
        for cand in [
            data["status"].as_u64(),
            data["httpStatus"].as_u64(),
            data["http_status"].as_u64(),
            data["statusCode"].as_u64(),
            data["error"]["code"].as_u64(),
            data["code"].as_u64(),
        ] {
            if let Some(code) = cand
                && (100..600).contains(&code)
            {
                status = Some(code as u16);
                break;
            }
        }
    }
    Failure { message, status, used }
}

/// Builds the rejection response for an incoming ACP `session/request_permission` request so no tool ever runs.
pub(super) fn reject_permission(id: &Value, params: &Value) -> Value {
    let reject_id = params["options"].as_array().and_then(|opts| {
        opts.iter()
            .find(|o| matches!(o["kind"].as_str(), Some("reject_once" | "reject_always")))
            .and_then(|o| o["optionId"].as_str())
    });
    match reject_id {
        Some(option_id) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "outcome": { "outcome": "selected", "optionId": option_id } }
        }),
        None => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": { "outcome": { "outcome": "cancelled" } }
        }),
    }
}

/// Builds the ACP `session/prompt` content blocks: instructions + user prompt + numbered inline base64 JPEG images.
pub(super) fn prompt_blocks(req: &AgentRequest<'_>, system_appended: bool) -> Vec<Value> {
    let lead = if system_appended {
        req.prompt.to_string()
    } else {
        format!(
            "{}\n\nDo not call any tools or read any files; every image is already attached inline. Reply with one \
             JSON object and nothing else.\n\n{}",
            req.system, req.prompt
        )
    };
    let mut blocks = Vec::with_capacity(1 + req.images.len() * 2);
    blocks.push(json!({ "type": "text", "text": lead }));
    for (i, jpeg) in req.images.iter().enumerate() {
        blocks.push(json!({ "type": "text", "text": format!("Image {}:", i + 1) }));
        blocks.push(json!({
            "type": "image",
            "mimeType": "image/jpeg",
            "data": base64::engine::general_purpose::STANDARD.encode(jpeg),
        }));
    }
    blocks
}
