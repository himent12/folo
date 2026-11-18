//! Codex as a labeling [`Agent`]: `codex exec`, one turn per call.
//!
//! Codex is a coding agent first: by default every request carries its coding instructions and the definitions of
//! its tools. A labeling call needs none of that, so FOLO's instructions replace Codex's, the user's config
//! (plugins, MCP servers) is not loaded, and everything that adds a tool or instructions is switched off:
//!
//! - the features that add tools ([`FEATURES_OFF`]) and the config keys of further text ([`CONFIG`]: skills,
//!   subagents, `request_user_input`, the environment and permission notes);
//! - the tools a model's own catalog entry brings: since codex-cli 0.16x every model runs in "code mode", whose
//!   JavaScript `exec` tool (and its helpers) is in every request whatever the features say. A call hands Codex its
//!   own catalog (`codex debug models`, read for the provider a call uses) back as `model_catalog_json` with those
//!   entries' tools taken off ([`lean_catalog`]): the model, its instructions and everything else stay as Codex
//!   has them. When the catalog cannot be read, the user's config points Codex at another catalog or server
//!   ([`catalog_conflict`]), or Codex refuses the file, calls go without it, and the run's log says so once
//!   ([`Agent::notice`]).
//!
//! On codex-cli 0.169.0 that takes the fixed cost of a call from about 20000 input tokens (Codex's defaults) and
//! 7800 (the switches before November 2026) to about 50 of Codex's own; with FOLO's instructions and reply schema
//! a call's fixed cost is about 700, so a call costs about what its images do. The images are attached with
//! `--image`, at the size FOLO sends them (Codex does not scale them down: an image's input tokens grow with its
//! area), and `--output-schema` holds the reply to FOLO's JSON shape. [`AgentRequest::fast`] asks for Codex's
//! priority service tier (`service_tier`, what Codex calls Fast), which the catalog says which models offer.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelModel, LabelUsage};
use serde_json::Value;

use super::agent::{
    Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Failure, Memo, Output, PROBE_TIMEOUT,
    fault_hint, last_line, limit_hint, probe_version, run, run_kept,
};

const INSTALL: &str = "Install Codex (https://developers.openai.com/codex/cli) on the server's machine and sign in with `codex login`";

/// Features that put tools (or their instructions) into every request. Switched off through `-c
/// features.<name>=false`, which a Codex version without the feature ignores (`--disable` would refuse to start).
const FEATURES_OFF: [&str; 14] = [
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "multi_agent",
    "image_generation",
    "view_image",
    "goals",
    "browser_use",
    "computer_use",
    "skill_search",
    "sleep_tool",
    "tool_suggest",
    "hooks",
];

/// Config keys that drop further text and tools a labeling call has no use for. A key this Codex version does
/// not know is ignored with a warning. Measured on codex-cli 0.169.0: the skills list is about 3500 tokens, the
/// subagent tools and their instructions about 1700, `request_user_input` about 400.
const CONFIG: [&str; 7] = [
    "web_search='disabled'",
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "include_collaboration_mode_instructions=false",
    "skills.include_instructions=false",
    "agents.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
];

/// The service tier [`AgentRequest::fast`] asks for: the catalog's id of what Codex calls Fast (priority
/// processing: about twice the speed, more of the plan's usage).
const FAST_TIER: &str = "priority";

/// The model provider a call uses: `codex exec --ignore-user-config` does not read a provider from the user's
/// config, so it is Codex's built-in one. `codex debug models` (which has no `--ignore-user-config`) is told the
/// same, so the catalog it prints is the one a call's model comes from.
const PROVIDER: &str = "model_provider='openai'";

/// Top-level keys of Codex's `config.toml` that can make `codex debug models` print another catalog than the one
/// `codex exec --ignore-user-config` uses (another catalog file, a profile, another server); see
/// [`catalog_conflict`]. With one set, FOLO cannot tell which models a call has, nor which offer Fast.
const CATALOG_KEYS: [&str; 4] = ["model_catalog_json", "profile", "openai_base_url", "chatgpt_base_url"];

/// What a call without the lean catalog costs more, in the words of a run's log.
const COSTS_MORE: &str =
    "calls go without FOLO's lean model catalog and cost about 7800 more input tokens each";

/// Codex (`codex`, or the executable `FOLO_CODEX_BIN` names).
pub struct Codex {
    cli: Cli,
    catalog: Memo<Catalog>,
    /// The versions of Codex that refused the lean catalog ([`with_lean_fallback`]): their calls go without it.
    /// Each refusal is its catalog snapshot's version, so a late one from a call that held an older snapshot
    /// neither turns off nor clears a newer version's.
    lean_refused: Mutex<BTreeSet<String>>,
}

/// What Codex's model catalog (`codex debug models`) gives FOLO.
#[derive(Clone, Debug, Default, PartialEq)]
struct Catalog {
    /// The version of Codex that printed it: the snapshot's generation.
    version: String,
    /// The executable that printed it, which a call with it runs (`None`: Codex is not installed).
    path: Option<PathBuf>,
    /// The models that take images ([`catalog_models`]).
    models: Vec<LabelModel>,
    /// Codex's default model: the catalog's first listed model.
    default: Option<String>,
    /// The models that offer the [`FAST_TIER`].
    fast: Vec<String>,
    /// The catalog with its models' tools taken off ([`lean_catalog`]); `None` when there is none to use.
    lean: Option<Arc<String>>,
    /// Why there is no lean catalog, for a run's log; `None` when there is one, or Codex is not installed.
    no_lean: Option<String>,
    /// Why no model is offered Fast although the catalog lists it ([`catalog_conflict`]).
    no_fast: Option<String>,
}

impl Catalog {
    /// The catalog `codex debug models` printed, `conflict` saying how the user's config may have made it another
    /// catalog than a call's ([`catalog_conflict`]): the lean catalog is then left out and no model is offered
    /// Fast, and the models are only suggestions.
    fn read(json: &str, conflict: Option<&str>) -> Catalog {
        let Ok(catalog) = serde_json::from_str::<Value>(json) else {
            return Catalog::unread("`codex debug models` printed no catalog FOLO can read");
        };
        let entries = catalog["models"].as_array().cloned().unwrap_or_default();
        let slug = |m: &Value| m["slug"].as_str().map(str::to_string);
        let mut listed: Vec<&Value> = entries.iter().filter(|m| m["visibility"] == "list").collect();
        listed.sort_by_key(|m| m["priority"].as_u64().unwrap_or(u64::MAX));
        let fast: Vec<String> = if conflict.is_some() {
            Vec::new()
        } else {
            entries
                .iter()
                .filter(|m| {
                    m["service_tiers"].as_array().is_some_and(|t| t.iter().any(|t| t["id"] == FAST_TIER))
                })
                .filter_map(slug)
                .collect()
        };
        let no_fast = conflict.map(|how| {
            format!("Codex's config.toml {how}, so FOLO cannot tell which models offer Fast to its calls")
        });
        let (lean, no_lean) = match conflict {
            Some(how) => (
                None,
                Some(format!(
                    "Codex's config.toml {how}, so the model catalog FOLO reads may not be the one its calls use: \
                     {COSTS_MORE}"
                )),
            ),
            None => match lean_catalog(json) {
                Some(lean) => (Some(Arc::new(lean)), None),
                None => (None, Some(format!("`codex debug models` printed no model catalog: {COSTS_MORE}"))),
            },
        };
        Catalog {
            version: String::new(),
            path: None,
            models: catalog_models(&entries, &fast),
            default: listed.first().and_then(|m| slug(m)),
            fast,
            lean,
            no_lean,
            no_fast,
        }
    }

    /// No catalog, for `why`.
    fn unread(why: &str) -> Catalog {
        Catalog { no_lean: Some(format!("{why}: {COSTS_MORE}")), ..Catalog::default() }
    }

    /// [`Agent::check_fast`] for this catalog.
    fn check_fast(&self, model: Option<&str>) -> Result<(), String> {
        if let Some(why) = &self.no_fast {
            return Err(why.clone());
        }
        match model.or(self.default.as_deref()) {
            Some(model) if self.fast.iter().any(|m| m == model) => Ok(()),
            Some(model) => Err(format!("Codex's model catalog lists no faster service tier for {model}")),
            None => Err(
                "Codex's model catalog could not be read, so FOLO cannot tell whether the model offers a \
                         faster service tier"
                    .into(),
            ),
        }
    }
}

impl Codex {
    /// The agent, looking for `codex` when it is first asked.
    pub const fn new() -> Codex {
        Codex {
            cli: Cli::new("codex", "FOLO_CODEX_BIN"),
            catalog: Memo::new(),
            lean_refused: Mutex::new(BTreeSet::new()),
        }
    }

    /// Codex's own model catalog (`codex debug models`, for the provider a call uses), printed by the executable
    /// and version the CLI's probe found now (read again when that changed); empty when Codex is not installed,
    /// and without models when it cannot be read.
    fn catalog(&self) -> Catalog {
        let Some((path, version)) = self.cli.found_pair() else { return Catalog::default() };
        let current = |c: &Catalog| c.path.as_ref() == Some(&path) && c.version == version;
        self.catalog.get_where(current, || {
            let mut cmd = Command::new(&path);
            cmd.args(["debug", "models", "-c", PROVIDER]);
            let catalog = match run(cmd, None, PROBE_TIMEOUT, &AtomicBool::new(false)) {
                Ok(out) if out.success => {
                    let conflict = codex_home().and_then(|home| config_conflict(&home.join("config.toml")));
                    Catalog::read(&out.stdout, conflict.as_deref())
                }
                Ok(out) => {
                    let why = last_line(&out.stderr).or_else(|| last_line(&out.stdout)).unwrap_or_default();
                    Catalog::unread(&format!("`codex debug models` failed ({why})"))
                }
                Err(_) => Catalog::unread("`codex debug models` did not run"),
            };
            Catalog { version: version.clone(), path: Some(path.clone()), ..catalog }
        })
    }

    /// Whether the version of Codex that printed `catalog` refused its lean catalog.
    fn refused(&self, catalog: &Catalog) -> bool {
        self.lean_refused.lock().unwrap().contains(&catalog.version)
    }

    /// Notes that `version` of Codex refused the lean catalog.
    fn refuse(&self, version: &str) {
        self.lean_refused.lock().unwrap().insert(version.to_string());
    }

    /// One `codex exec`, with `catalog` as its `model_catalog_json` when there is one: what it printed.
    fn call(&self, path: &Path, req: &AgentRequest<'_>, catalog: Option<&str>) -> Result<Output, AgentError> {
        let files = Files::write(req, catalog)
            .map_err(|e| AgentError::Unavailable(format!("preparing the call: {e}")))?;
        let cmd = command(path, req, &files);
        // A call killed after its turn completed (it hung on its way out) still used what that event says.
        run_kept(cmd, Some(req.prompt.as_bytes().to_vec()), CALL_TIMEOUT, req.cancel).map_err(|killed| {
            let (e, out) = *killed;
            e.with_used(usage_of(&out.stdout).unwrap_or_default())
        })
    }
}

/// Codex's home folder: `CODEX_HOME`, else `.codex` in the user's home.
fn codex_home() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("CODEX_HOME").filter(|h| !h.is_empty()) {
        return Some(PathBuf::from(home));
    }
    let home =
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".codex"))
}

/// How Codex's `config.toml` at `path` may point Codex at another catalog or server than a call's
/// ([`catalog_conflict`]). Only a file that is not there is no config: one that cannot be read (no permission, an
/// I/O error, not UTF-8) may set anything, so it counts as a conflict.
fn config_conflict(path: &Path) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(config) => catalog_conflict(&config),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => Some(format!("cannot be read ({e})")),
    }
}

/// How Codex's `config.toml` (its text) may point Codex at another catalog or server than a call's: "sets `key`"
/// for the first of [`CATALOG_KEYS`] it sets at the top level (a bare or a quoted key alike), or that it cannot be
/// read as TOML; `None` when it does neither.
fn catalog_conflict(config: &str) -> Option<String> {
    match config.parse::<toml::Table>() {
        Ok(table) => CATALOG_KEYS.iter().find(|k| table.contains_key(**k)).map(|key| format!("sets `{key}`")),
        Err(_) => Some("cannot be read as TOML".into()),
    }
}

/// Whether a call's `out` is Codex refusing the lean catalog as it started: it failed before any event (so before
/// it asked the model, and with no tokens used), naming `model_catalog_json`.
fn refused_catalog(out: &Output) -> bool {
    let started =
        out.stdout.lines().any(|l| serde_json::from_str::<Value>(l).is_ok_and(|v| v["type"].is_string()));
    !out.success && !started && format!("{}{}", out.stderr, out.stdout).contains("model_catalog_json")
}

/// A call with the lean `catalog`, made once more without it when Codex refused the file as it started
/// ([`refused_catalog`]: no tokens spent, so this is no retry of the batch's); `refuse` is told, so later calls go
/// without it.
fn with_lean_fallback(
    catalog: Option<&str>,
    refuse: impl FnOnce(),
    mut call: impl FnMut(Option<&str>) -> Result<Output, AgentError>,
) -> Result<Output, AgentError> {
    let out = call(catalog)?;
    if catalog.is_some() && refused_catalog(&out) {
        refuse();
        return call(None);
    }
    Ok(out)
}

impl Default for Codex {
    fn default() -> Self {
        Codex::new()
    }
}

impl Agent for Codex {
    fn id(&self) -> LabelAgentId {
        LabelAgentId::Codex
    }

    fn info(&self) -> LabelAgent {
        let version = self.cli.version();
        let catalog = if version.is_some() { self.catalog() } else { Catalog::default() };
        LabelAgent {
            id: self.id(),
            name: self.id().display_name().into(),
            available: version.is_some(),
            hint: version.is_none().then(|| self.cli.hint(INSTALL)),
            fast: catalog.check_fast(None).is_ok(),
            models: catalog.models,
            version,
            // Codex picks its own default; its config file, where a user would set one, is not loaded.
            default_model: None,
            // Measured November 2026 (gpt-6.2-sol): medium effort found 86% of the objects, as low did, with twice
            // the output tokens and twice the time.
            default_effort: LabelEffort::Low,
        }
    }

    fn default_batch(&self) -> usize {
        // Measured November 2026 (codex-cli 0.169.0, gpt-6.2-sol): sixteen photos a call cost 9% less input than
        // eight but found 84-85% of the objects against 86% (two passes each), and a failed call loses twice
        // as many; four would pay a call's fixed cost (about 700 tokens) twice as often.
        8
    }

    fn image_side(&self) -> u32 {
        // An image costs about 1050 input tokens at 1024 px and 650 at 768, and the labels are as good: photos
        // scaled to 768 px found 87% of the objects against 86% at 1024 (and 640 px ones sent at 480, 85% against
        // 86%).
        768
    }

    fn check_fast(&self, model: Option<&str>) -> Result<(), String> {
        self.catalog().check_fast(model)
    }

    fn notice(&self) -> Option<String> {
        let catalog = self.catalog();
        if self.refused(&catalog) {
            return Some(format!(
                "Codex {} refused FOLO's lean model catalog: {COSTS_MORE}",
                catalog.version
            ));
        }
        catalog.no_lean
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        // The executable the catalog came from: a call and its catalog are of one version.
        let catalog = self.catalog();
        let path = match &catalog.path {
            Some(path) => path.clone(),
            None => self.cli.path()?,
        };
        // Fast as this call's catalog offers it (the one the run was started with may be older).
        if req.fast {
            catalog.check_fast(req.model).map_err(AgentError::Unavailable)?;
        }
        let lean = catalog.lean.as_deref().map(String::as_str).filter(|_| !self.refused(&catalog));
        // The refusal is the version that refused: Codex may have been updated in place since the catalog was read.
        let refuse = || self.refuse(&probe_version(&path).unwrap_or_else(|| catalog.version.clone()));
        parse(&with_lean_fallback(lean, refuse, |lean| self.call(&path, req, lean))?)
    }

    fn hint(&self, message: &str) -> Option<String> {
        let m = message.to_lowercase();
        let signed_out = ["login", "log in", "logged in", "api key", "authenticat", "unauthorized", "401"];
        if signed_out.iter().any(|w| m.contains(w)) {
            Some("run `codex login` in a terminal on the server's machine, then start again".into())
        } else if let Some(hint) = fault_hint(message, None) {
            Some(hint)
        } else if ["limit", "quota", "credits", "spend cap"].iter().any(|w| m.contains(w)) {
            Some(limit_hint(message, None))
        } else if m.contains("model") {
            Some("pick another model: this account may not have access to that one".into())
        } else if m.contains("printed no reply") {
            Some("run `codex exec hi` in a terminal on the server's machine to see why (a sign-in, a limit or an update it asks for)".into())
        } else {
            None
        }
    }
}

/// The files of one call, inside its scratch folder.
struct Files {
    /// The empty folder Codex works in, so it finds no `AGENTS.md` and no repository.
    work: std::path::PathBuf,
    instructions: std::path::PathBuf,
    schema: std::path::PathBuf,
    /// The lean model catalog, when there is one.
    catalog: Option<std::path::PathBuf>,
    images: Vec<std::path::PathBuf>,
}

impl Files {
    fn write(req: &AgentRequest<'_>, catalog: Option<&str>) -> std::io::Result<Files> {
        let files = Files {
            work: req.scratch.join("work"),
            instructions: req.scratch.join("instructions.md"),
            schema: req.scratch.join("schema.json"),
            catalog: catalog.map(|_| req.scratch.join("models.json")),
            images: (1..=req.images.len()).map(|i| req.scratch.join(format!("{i}.jpg"))).collect(),
        };
        std::fs::create_dir_all(&files.work)?;
        std::fs::write(&files.instructions, req.system)?;
        std::fs::write(&files.schema, serde_json::to_vec(req.schema).expect("JSON values serialize"))?;
        if let (Some(path), Some(catalog)) = (&files.catalog, catalog) {
            std::fs::write(path, catalog)?;
        }
        for (path, jpeg) in files.images.iter().zip(req.images) {
            std::fs::write(path, jpeg)?;
        }
        Ok(files)
    }
}

/// `codex exec` for one turn: FOLO's instructions, every tool off, the prompt read from stdin.
fn command(path: &Path, req: &AgentRequest<'_>, files: &Files) -> Command {
    let mut cmd = Command::new(path);
    cmd.args(["exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config"])
        .args(["--sandbox", "read-only"])
        .arg("-C")
        .arg(&files.work);
    let instructions = toml_string(&folo_api::json::path(&files.instructions));
    cmd.args(["-c", &format!("model_instructions_file={instructions}")])
        .args(["-c", &format!("model_reasoning_effort='{}'", req.effort.as_str())]);
    for key in CONFIG {
        cmd.args(["-c", key]);
    }
    for feature in FEATURES_OFF {
        cmd.args(["-c", &format!("features.{feature}=false")]);
    }
    if let Some(catalog) = &files.catalog {
        cmd.args(["-c", &format!("model_catalog_json={}", toml_string(&folo_api::json::path(catalog)))]);
    }
    if req.fast {
        cmd.args(["-c", &format!("service_tier='{FAST_TIER}'")]);
    }
    if let Some(model) = req.model {
        cmd.args(["--model", model]);
    }
    cmd.arg("--output-schema").arg(&files.schema);
    for image in &files.images {
        cmd.arg("--image").arg(image);
    }
    // `-`: the prompt comes on stdin.
    cmd.arg("-");
    cmd
}

/// `s` as a TOML string for a `-c key=value` argument: a literal string (`'...'`, nothing to escape, and no
/// double quote for a Windows batch shim to trip over) unless `s` itself has a single quote.
fn toml_string(s: &str) -> String {
    if s.contains('\'') {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        format!("'{s}'")
    }
}

/// Reads the events `codex exec --json` prints: the last agent message is the reply and `turn.completed` has
/// the usage; without them, `turn.failed` / `error` say why. (An `error` before a completed turn was a retry, and
/// `item.completed` items of type `error` are warnings, e.g. about a config key this version does not know.)
///
/// What a completed turn used is counted whether or not there is a reply: a failure carries it.
fn parse(out: &Output) -> Result<AgentReply, AgentError> {
    let (mut text, mut failure) = (None, None);
    for event in out.stdout.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        match event["type"].as_str() {
            Some("item.completed") if event["item"]["type"] == "agent_message" => {
                text = event["item"]["text"].as_str().map(str::to_string);
            }
            Some("turn.failed") => failure = event["error"]["message"].as_str().map(str::to_string),
            Some("error") => failure = event["message"].as_str().map(str::to_string),
            _ => {}
        }
    }
    let usage = usage_of(&out.stdout);
    let (Some(text), Some(usage)) = (text, usage) else {
        let used = usage.unwrap_or_default();
        let why = failure
            .or_else(|| last_line(&out.stderr).filter(|_| !out.success))
            .or_else(|| last_line(&out.stdout).filter(|_| !out.success));
        return Err(match why {
            Some(why) => AgentError::Failed(Failure { used, ..Failure::new(why) }),
            // A turn that completed without a message still used its tokens.
            None if usage.is_some() => {
                AgentError::Failed(Failure { used, ..Failure::new("`codex` printed no reply") })
            }
            None => AgentError::Unavailable("`codex` printed no reply".into()),
        });
    };
    Ok(AgentReply { text, usage })
}

/// What the `turn.completed` event among the events `stdout` has says the call used; `None` without one.
fn usage_of(stdout: &str) -> Option<LabelUsage> {
    let event = stdout
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|event| event["type"] == "turn.completed")?;
    let tokens = |key: &str| event["usage"][key].as_u64().unwrap_or(0);
    Some(LabelUsage {
        calls: 1,
        input_tokens: tokens("input_tokens"),
        cached_input_tokens: tokens("cached_input_tokens"),
        output_tokens: tokens("output_tokens"),
        cost_usd: None,
    })
}

/// Codex's model catalog (`codex debug models`) as `model_catalog_json` for a labeling call: every model as Codex
/// describes it, but without the tools its entry brings into every request. `tool_mode` "direct" instead of
/// "code_mode_only" drops the JavaScript `exec` tool and its helpers, an empty `experimental_supported_tools` the
/// `clock`, `send_message_to_user_async` and `request_user_input_async` tools, and no `apply_patch_tool_type` the
/// patch tool. Only fields the catalog has are changed, so the file stays in the shape this version reads; `None`
/// when the text is not a catalog.
fn lean_catalog(json: &str) -> Option<String> {
    let mut catalog: Value = serde_json::from_str(json).ok()?;
    let models = catalog.get_mut("models")?.as_array_mut()?;
    for model in models.iter_mut().filter_map(Value::as_object_mut) {
        for (key, lean) in [
            ("tool_mode", Value::from("direct")),
            ("experimental_supported_tools", Value::Array(Vec::new())),
            ("apply_patch_tool_type", Value::Null),
        ] {
            if let Some(v) = model.get_mut(key) {
                *v = lean;
            }
        }
    }
    serde_json::to_string(&catalog).ok()
}

/// The listed models of a catalog's `entries` that accept images, in the catalog's order of preference; those in
/// `fast` offer the [`FAST_TIER`].
fn catalog_models(entries: &[Value], fast: &[String]) -> Vec<LabelModel> {
    let mut models: Vec<(u64, LabelModel)> = entries
        .iter()
        .filter(|m| m["visibility"] == "list")
        .filter(|m| m["input_modalities"].as_array().is_some_and(|a| a.iter().any(|x| x == "image")))
        .filter_map(|m| {
            let id = m["slug"].as_str()?.to_string();
            let label = m["display_name"].as_str().unwrap_or(&id).to_string();
            let fast = fast.contains(&id);
            Some((m["priority"].as_u64().unwrap_or(u64::MAX), LabelModel { id, label, fast }))
        })
        .collect();
    models.sort_by_key(|(priority, _)| *priority);
    models.into_iter().map(|(_, m)| m).collect()
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    fn output(stdout: &str, stderr: &str, success: bool) -> Output {
        Output { success, stdout: stdout.into(), stderr: stderr.into() }
    }

    /// The events codex-cli 0.167.0 prints for a successful call, with a warning about an unknown config key.
    const SUCCESS: &str = r#"{"type":"thread.started","thread_id":"01a1c20c"}
{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Codex is ignoring 1 unrecognized configuration setting."}}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"{\"images\":[{\"image\":1,\"boxes\":[]}]}"}}
{"type":"turn.completed","usage":{"input_tokens":8993,"cached_input_tokens":4096,"cache_write_input_tokens":0,"output_tokens":287,"reasoning_output_tokens":150}}"#;

    #[test]
    fn the_last_agent_message_is_the_reply_and_warnings_are_not_failures() {
        let reply = parse(&output(SUCCESS, "Reading prompt from stdin...\n", true)).unwrap();
        assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[]}]}"#);
        assert_eq!(
            reply.usage,
            LabelUsage {
                calls: 1,
                input_tokens: 8993,
                cached_input_tokens: 4096,
                output_tokens: 287,
                cost_usd: None
            }
        );
    }

    #[test]
    fn failed_turns_and_silent_exits_are_failures() {
        let stdout = r#"{"type":"thread.started","thread_id":"t"}
{"type":"error","message":"unexpected status 401 Unauthorized"}
{"type":"turn.failed","error":{"message":"unexpected status 401 Unauthorized: sign in again"}}"#;
        let err = parse(&output(stdout, "", false)).unwrap_err();
        assert_eq!(err, AgentError::failed("unexpected status 401 Unauthorized: sign in again"));
        assert!(Codex::new().hint("unexpected status 401 Unauthorized").unwrap().contains("codex login"));
        let err = parse(&output("", "Error: Unknown feature flag: nope\n", false)).unwrap_err();
        assert_eq!(err, AgentError::failed("Error: Unknown feature flag: nope"));
        let err = parse(&output("", "", true)).unwrap_err();
        let AgentError::Unavailable(message) = &err else { panic!("{err:?}") };
        assert!(Codex::new().hint(message).unwrap().contains("`codex exec hi`"));
        // A turn that completed without a message, on a normal exit: no reply, but its tokens were used.
        let no_message =
            SUCCESS.lines().filter(|l| !l.contains("agent_message")).collect::<Vec<_>>().join("\n");
        let err = parse(&output(&no_message, "", true)).unwrap_err();
        let AgentError::Failed(failure) = &err else { panic!("{err:?}") };
        assert_eq!((failure.message.as_str(), failure.used.input_tokens), ("`codex` printed no reply", 8993));
        assert!(Codex::new().hint(&failure.message).unwrap().contains("`codex exec hi`"));
        // ... and a call killed after it printed that event (it hung on its way out) used them too.
        assert_eq!(usage_of(&no_message).map(|u| u.output_tokens), Some(287));
        assert_eq!(usage_of("{\"type\":\"turn.started\"}"), None);
        // A usage limit says when it resets; a server error that outlasted Codex's own retries is not a limit.
        let limit = "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again \
                     at 3:04 PM.";
        let hint = Codex::new().hint(limit).unwrap();
        assert!(hint.contains("usage limit was reached (try again at 3:04 PM): start again then"), "{hint}");
        assert_eq!(
            Codex::new().hint("exceeded retry limit, last status: 503 Service Unavailable").unwrap(),
            crate::label::agent::BUSY
        );
        // A rate limit is the provider's, in its own words; one that asks for a long wait is that wait.
        let hint = Codex::new().hint("Rate limit reached for gpt-6.2-sol. Please try again in 30s.").unwrap();
        assert!(hint.starts_with("the provider's rate limit was reached (try again in 30s)"), "{hint}");
        let hint =
            Codex::new().hint("Rate limit reached for gpt-6.2-sol. Please try again in 20m0s.").unwrap();
        assert!(hint.starts_with("the provider asked for a long wait (try again in 20m0s)"), "{hint}");
    }

    #[test]
    fn the_command_attaches_the_images_and_switches_the_tools_off() {
        let dir = TestDir::new("label-codex-command");
        let (schema, cancel) = (serde_json::json!({ "type": "object" }), AtomicBool::new(false));
        let req = AgentRequest {
            system: "label things",
            prompt: "Classes: car",
            images: &[vec![1], vec![2, 2]],
            schema: &schema,
            model: Some("gpt-6-luna"),
            effort: LabelEffort::Low,
            fast: true,
            scratch: dir.path(),
            cancel: &cancel,
        };
        let files = Files::write(&req, Some(r#"{"models":[]}"#)).unwrap();
        assert_eq!(std::fs::read_to_string(&files.instructions).unwrap(), "label things");
        assert_eq!(std::fs::read(&files.images[1]).unwrap(), vec![2, 2]);
        assert_eq!(std::fs::read_to_string(files.catalog.as_ref().unwrap()).unwrap(), r#"{"models":[]}"#);
        assert_eq!(std::fs::read_dir(&files.work).unwrap().count(), 0);

        let cmd = command(Path::new("codex"), &req, &files);
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(args[0], "exec");
        assert_eq!(args.last().map(String::as_str), Some("-"));
        for flag in ["--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config"] {
            assert!(args.iter().any(|a| a == flag), "{flag} missing");
        }
        let values = |flag: &str| -> Vec<&str> {
            args.windows(2).filter(|w| w[0] == flag).map(|w| w[1].as_str()).collect()
        };
        assert_eq!(values("--sandbox"), ["read-only"]);
        assert_eq!(values("--model"), ["gpt-6-luna"]);
        assert_eq!(values("--image").len(), 2);
        assert!(values("--image")[0].ends_with("1.jpg"));
        let config = values("-c");
        assert!(config.contains(&"model_reasoning_effort='low'"));
        assert!(config.contains(&"features.shell_tool=false") && config.contains(&"web_search='disabled'"));
        for key in ["skills.include_instructions=false", "agents.enabled=false", "service_tier='priority'"] {
            assert!(config.contains(&key), "{key} missing from {config:?}");
        }
        let instructions = config.iter().find(|c| c.starts_with("model_instructions_file='")).unwrap();
        assert!(instructions.ends_with("instructions.md'") && !instructions.contains('\\'), "{instructions}");
        let catalog = config.iter().find(|c| c.starts_with("model_catalog_json='")).unwrap();
        assert!(catalog.ends_with("models.json'"), "{catalog}");
        // No argument needs quoting a batch shim could get wrong.
        assert!(args.iter().all(|a| !a.contains('"') && !a.contains('\n')));
        assert_eq!(toml_string("D:/data/x.md"), "'D:/data/x.md'");
        assert_eq!(toml_string(r#"C:/O'Brien/a "b".md"#), r#""C:/O'Brien/a \"b\".md""#);

        // Without a catalog (it could not be read) and without the fast tier, neither is set.
        let dir = TestDir::new("label-codex-command-plain");
        let req = AgentRequest { fast: false, scratch: dir.path(), ..req };
        let files = Files::write(&req, None).unwrap();
        let cmd = command(Path::new("codex"), &req, &files);
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(!args.iter().any(|a| a.starts_with("model_catalog_json") || a.starts_with("service_tier")));
    }

    #[test]
    fn the_lean_catalog_takes_off_the_models_tools_and_nothing_else() {
        let json = r#"{"models":[
            {"slug":"a","tool_mode":"code_mode_only","experimental_supported_tools":["clock"],
             "apply_patch_tool_type":"freeform","base_instructions":"x","priority":1},
            {"slug":"old","priority":2}],"etag":"e"}"#;
        let lean: Value = serde_json::from_str(&lean_catalog(json).unwrap()).unwrap();
        let expected = serde_json::json!({"models":[
            {"slug":"a","tool_mode":"direct","experimental_supported_tools":[],
             "apply_patch_tool_type":null,"base_instructions":"x","priority":1},
            {"slug":"old","priority":2}],"etag":"e"});
        assert_eq!(lean, expected);
        assert_eq!(lean_catalog("not json"), None);
        assert_eq!(lean_catalog(r#"{"data":[]}"#), None);
    }

    #[test]
    fn an_error_before_a_completed_turn_was_a_retry() {
        let retried = "{\"type\":\"turn.started\"}\n{\"type\":\"error\",\"message\":\"stream disconnected; retrying 1/5\"}";
        let stdout = SUCCESS.replacen("{\"type\":\"turn.started\"}", retried, 1);
        assert!(parse(&output(&stdout, "", true)).is_ok());
    }

    const FAST: &str = r#"[{"id":"priority","name":"Fast","description":"2x speed"}]"#;

    /// A catalog as `codex debug models` prints it: `b` and `hidden` offer Fast, `text` takes no images.
    fn catalog_json() -> String {
        format!(
            r#"{{"models":[
            {{"slug":"b","display_name":"B","visibility":"list","input_modalities":["text","image"],"priority":2,"service_tiers":{FAST}}},
            {{"slug":"hidden","display_name":"H","visibility":"hide","input_modalities":["text","image"],"priority":0,"service_tiers":{FAST}}},
            {{"slug":"text","display_name":"T","visibility":"list","input_modalities":["text"],"priority":1,"service_tiers":[]}},
            {{"slug":"a","visibility":"list","input_modalities":["image"],"priority":1,"tool_mode":"code_mode_only"}}]}}"#
        )
    }

    #[test]
    fn catalog_lists_visible_image_models_by_priority() {
        let catalog = Catalog::read(&catalog_json(), None);
        assert_eq!(
            catalog.models,
            [
                LabelModel { id: "a".into(), label: "a".into(), fast: false },
                LabelModel { id: "b".into(), label: "B".into(), fast: true }
            ]
        );
        assert_eq!(catalog.default.as_deref(), Some("text"), "the first listed model, images or not");
        assert!(catalog.lean.is_some() && catalog.no_lean.is_none());
        let unread = Catalog::read("not json", None);
        assert!(unread.models.is_empty() && unread.lean.is_none());
        assert!(unread.no_lean.unwrap().contains("7800 more input tokens"));
        assert!(Catalog::read("{}", None).models.is_empty());
    }

    #[test]
    fn fast_is_offered_for_the_models_whose_entry_lists_the_priority_tier() {
        let catalog = Catalog::read(&catalog_json(), None);
        assert_eq!(catalog.check_fast(Some("b")), Ok(()));
        assert_eq!(catalog.check_fast(Some("hidden")), Ok(()), "a model the list hides can still be named");
        assert_eq!(
            catalog.check_fast(Some("text")).unwrap_err(),
            "Codex's model catalog lists no faster service tier for text"
        );
        assert!(catalog.check_fast(Some("not-in-the-catalog")).is_err());
        // The default model (`text`) does not offer it, so neither does a run that names no model.
        assert!(catalog.check_fast(None).is_err());
        let b_first = catalog_json().replace(r#""priority":2"#, r#""priority":0"#);
        assert_eq!(Catalog::read(&b_first, None).check_fast(None), Ok(()));
        assert!(Catalog::default().check_fast(None).unwrap_err().contains("could not be read"));
    }

    #[test]
    fn a_config_that_may_point_codex_elsewhere_keeps_the_lean_catalog_and_fast_out() {
        let config = "model_auto_compact_token_limit = 400000\n# profile = \"x\"\n\n[profiles.fast]\nmodel_catalog_json = \"x\"\n";
        assert_eq!(catalog_conflict(config), None, "comments and keys in tables do not count");
        assert_eq!(
            catalog_conflict("openai_base_url = \"https://proxy\"\n[x]\n").as_deref(),
            Some("sets `openai_base_url`")
        );
        assert_eq!(catalog_conflict("  profile=\"work\"").as_deref(), Some("sets `profile`"));
        // Quoted keys are the same keys.
        assert_eq!(
            catalog_conflict("'model_catalog_json' = 'x.json'").as_deref(),
            Some("sets `model_catalog_json`")
        );
        assert_eq!(
            catalog_conflict("\"chatgpt_base_url\" = \"x\"").as_deref(),
            Some("sets `chatgpt_base_url`")
        );
        assert_eq!(catalog_conflict("profile = ").as_deref(), Some("cannot be read as TOML"));
        assert_eq!(catalog_conflict(""), None);
        let catalog = Catalog::read(&catalog_json(), Some("sets `profile`"));
        assert_eq!(catalog.lean, None);
        assert!(catalog.no_lean.as_deref().unwrap().starts_with("Codex's config.toml sets `profile`"));
        assert_eq!(catalog.models.len(), 2, "the models are still offered");
        // ... but no model is offered Fast, and asking for it says why.
        assert!(catalog.models.iter().all(|m| !m.fast));
        let why = catalog.check_fast(Some("b")).unwrap_err();
        assert!(
            why.contains("sets `profile`") && why.contains("cannot tell which models offer Fast"),
            "{why}"
        );
    }

    #[test]
    fn a_refused_lean_catalog_is_left_out_of_that_call_and_by_that_version_only() {
        // What codex-cli prints when it cannot read the file: no event, nothing asked of the model.
        let refusal = || {
            output(
                "",
                "Error: failed to parse model_catalog_json path `x` as JSON: missing field `slug`\n",
                false,
            )
        };
        let mut seen = Vec::new();
        let mut refused = false;
        let out = with_lean_fallback(
            Some("{}"),
            || refused = true,
            |catalog| {
                seen.push(catalog.is_some());
                Ok(if catalog.is_some() { refusal() } else { output(SUCCESS, "", true) })
            },
        )
        .unwrap();
        assert!(parse(&out).is_ok() && seen == [true, false] && refused);
        // A call that started (it printed events) and failed is the call's failure, whatever its message names:
        // no second call.
        let started = output(
            "{\"type\":\"thread.started\",\"thread_id\":\"t\"}\n{\"type\":\"turn.failed\",\"error\":{\"message\":\"model_catalog_json says no\"}}",
            "",
            false,
        );
        let mut calls = 0;
        let mut refused = false;
        let out = with_lean_fallback(
            Some("{}"),
            || refused = true,
            |_| {
                calls += 1;
                Ok(started.clone())
            },
        )
        .unwrap();
        assert!(parse(&out).is_err() && calls == 1 && !refused);
        // Without the lean catalog nothing is retried.
        let mut calls = 0;
        let _ = with_lean_fallback(
            None,
            || {},
            |_| {
                calls += 1;
                Ok(refusal())
            },
        );
        assert_eq!(calls, 1);

        // The refusal holds for the version that refused, across reads of its catalog, and not for another.
        let codex = Codex::new();
        let snapshot =
            |version: &str| Catalog { version: version.into(), ..Catalog::read(&catalog_json(), None) };
        codex.refuse("0.169.0");
        assert!(codex.refused(&snapshot("0.169.0")));
        assert!(!codex.refused(&snapshot("0.170.0")), "a newer Codex is given the lean catalog again");
        // Refusals interleave: the newer version refuses too, then a late refusal from a call that still held
        // the older snapshot comes in. Neither clears the other.
        codex.refuse("0.170.0");
        codex.refuse("0.169.0");
        assert!(codex.refused(&snapshot("0.170.0")) && codex.refused(&snapshot("0.169.0")));
        assert!(!codex.refused(&snapshot("0.171.0")));
    }

    #[test]
    fn a_config_that_cannot_be_read_counts_as_one_that_may_point_codex_elsewhere() {
        let dir = TestDir::new("label-codex-config");
        let path = dir.path().join("config.toml");
        assert_eq!(config_conflict(&path), None, "no config at all");
        std::fs::write(&path, "model = \"x\"\n").unwrap();
        assert_eq!(config_conflict(&path), None);
        std::fs::write(&path, "profile = \"work\"\n").unwrap();
        assert_eq!(config_conflict(&path).as_deref(), Some("sets `profile`"));
        // Not UTF-8: not read, so it may set anything.
        std::fs::write(&path, b"profile = \"\xff\xfe\"\n").unwrap();
        assert!(config_conflict(&path).unwrap().starts_with("cannot be read ("));
        // A folder where the file should be: an error other than its absence.
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(config_conflict(&path).unwrap().starts_with("cannot be read ("));
        let catalog = Catalog::read(&catalog_json(), config_conflict(&path).as_deref());
        assert!(catalog.lean.is_none() && catalog.check_fast(Some("b")).is_err());
    }
}
