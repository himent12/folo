//! Antigravity CLI (`agy`) as a labeling [`Agent`]: `agy acp`, one turn per call over ACP v1 JSON-RPC on stdio.
//!
//! `agy -p` has no `--image` flag, rejects non-text content blocks on `--input-format stream-json`, and blocks for
//! interactive OAuth when signed out. `agy acp` instead advertises native inline image prompts
//! (`promptCapabilities.image`), dynamic `model` and `thought_level` configuration (`configOptions`), system-prompt
//! appending (`_meta.antigravity.systemPrompt.append`), and immediate JSON-RPC errors when signed out.
//!
//! Each labeling call runs `agy acp` in an isolated scratch workspace containing a dedicated `folo-labeler` custom
//! agent (`tools: [send_message]`, `commandExecutionPolicy: "off"`, `inheritMcp: false`,
//! `inheritCustomizations: false`), paired with an isolated CLI settings profile (`--cli_dir_name folo-label`) whose
//! `settings.json` sets `toolPermission: "strict"` and denies all file, command, URL, and MCP actions, plus
//! `--sandbox`, `--include-default-customizations=false`, `--disable-telemetry`, and `mcpServers: []`. The ACP
//! session also rejects auto-approving modes, rejects any `session/request_permission` request, answers `-32601` to
//! client tool methods, and aborts closed if a `tool_call` update ever appears. Model suggestions come from
//! `agy models` (ANSI-stripped, effort suffixes collapsed into base model ids) with a curated fallback list
//! including `gemini-4-neon` when `agy models` cannot be read.

mod acp;
mod process;
mod protocol;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelModel};

use super::agent::{
    Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Memo, PROBE_TIMEOUT, fault_hint,
    limit_hint, run,
};

const INSTALL: &str = "Install Antigravity CLI (https://antigravity.google/docs/getting-started) on the server's \
                       machine and sign in by running `agy` once";

/// Curated Antigravity models offered when `agy` is not installed or `agy models` cannot be read (e.g. before sign
/// in). Any model name the CLI accepts (including effort-suffixed slugs such as `gemini-4-neon-xhigh`) also works.
const FALLBACK_MODELS: [(&str, &str); 5] = [
    ("gemini-4-neon", "Gemini 4 Neon"),
    ("gemini-4-hydrogen", "Gemini 4 Hydrogen"),
    ("gemini-4-carbon", "Gemini 4 Carbon"),
    ("gemini-4-argon", "Gemini 4 Argon"),
    ("gemini-3.8-flash", "Gemini 3.8 Flash"),
];

/// Effort suffixes that `agy models` (and users) attach to a base model slug.
const EFFORT_SUFFIXES: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/// Antigravity CLI (`agy`, or the executable `FOLO_ANTIGRAVITY_BIN` names).
pub struct Antigravity {
    cli: Cli,
    catalog: Memo<Catalog>,
}

/// Cached model list read from `agy models` for one executable and version.
#[derive(Clone, Debug, Default, PartialEq)]
struct Catalog {
    path: Option<PathBuf>,
    version: String,
    models: Vec<LabelModel>,
}

impl Antigravity {
    /// The agent, looking for `agy` when it is first asked.
    pub const fn new() -> Antigravity {
        Antigravity { cli: Cli::new("agy", "FOLO_ANTIGRAVITY_BIN"), catalog: Memo::new() }
    }

    /// The models `agy models` lists for the probed executable and version (cached); empty when `agy` is missing or
    /// `agy models` failed (for instance before sign in).
    fn catalog(&self) -> Catalog {
        let Some((path, version)) = self.cli.found_pair() else { return Catalog::default() };
        let current = |c: &Catalog| c.path.as_ref() == Some(&path) && c.version == version;
        self.catalog.get_where(current, || {
            let mut cmd = Command::new(&path);
            cmd.arg("models");
            let models = match run(cmd, None, PROBE_TIMEOUT, &AtomicBool::new(false)) {
                Ok(out) if out.success => parse_models(&out.stdout),
                _ => Vec::new(),
            };
            Catalog { path: Some(path.clone()), version: version.clone(), models }
        })
    }

    fn call_with_timeout(&self, req: &AgentRequest<'_>, timeout: Duration) -> Result<AgentReply, AgentError> {
        let path = self.cli.path()?;
        process::run_acp(&path, req, timeout)
    }
}

impl Default for Antigravity {
    fn default() -> Self {
        Antigravity::new()
    }
}

impl Agent for Antigravity {
    fn id(&self) -> LabelAgentId {
        LabelAgentId::Antigravity
    }

    fn info(&self) -> LabelAgent {
        let version = self.cli.version();
        let models = if version.is_some() {
            let discovered = self.catalog().models;
            if discovered.is_empty() { fallback_models() } else { discovered }
        } else {
            fallback_models()
        };
        LabelAgent {
            id: self.id(),
            name: self.id().display_name().into(),
            available: version.is_some(),
            hint: version.is_none().then(|| self.cli.hint(INSTALL)),
            version,
            models,
            // Antigravity picks the account's default (Gemini 4 Neon on Pro/Ultra/Enterprise, Gemini 3.8 Flash on
            // Free/Plus); users can also pin any listed or unlisted model explicitly.
            default_model: None,
            default_effort: LabelEffort::Low,
            fast: false,
        }
    }

    fn default_batch(&self) -> usize {
        // Amortises ACP session startup over 8 images per turn, matching Codex.
        8
    }

    fn resolve_request(
        &self,
        model: Option<String>,
        effort: Option<LabelEffort>,
    ) -> Result<(Option<String>, LabelEffort), String> {
        if let Some(m) = model.as_deref()
            && let Some((base, suffix)) = split_effort_suffix(m)
        {
            let mapped = effort_for_suffix(suffix);
            if let Some(explicit) = effort
                && explicit != mapped
            {
                return Err(format!(
                    "model '{m}' already specifies effort '{suffix}', which conflicts with effort '{}': omit effort \
                     or pass the base model '{base}'",
                    explicit.as_str()
                ));
            }
            return Ok((model, mapped));
        }
        Ok((model, effort.unwrap_or(LabelEffort::Low)))
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        self.call_with_timeout(req, CALL_TIMEOUT)
    }

    fn hint(&self, message: &str) -> Option<String> {
        let m = message.to_lowercase();
        let signed_out = [
            "authentication required",
            "authenticat",
            "please sign in",
            "sign in",
            "login",
            "log in",
            "logged in",
            "unauthorized",
            "401",
            "api key",
            "oauth",
            "credentials",
        ];
        if signed_out.iter().any(|w| m.contains(w)) {
            Some(
                "run `agy` (or `agy acp --login`) in a terminal on the server's machine to sign in, then start again"
                    .into(),
            )
        } else if let Some(hint) = fault_hint(message, None) {
            Some(hint)
        } else if ["limit", "quota", "credits", "resource_exhausted", "spend cap"]
            .iter()
            .any(|w| m.contains(w))
        {
            Some(limit_hint(message, None))
        } else if m.contains("model") {
            Some("pick another model: this account may not have access to that one".into())
        } else if m.contains("printed no reply") || m.contains("exited before replying") {
            Some(
                "run `agy -p hi` in a terminal on the server's machine to see why (a sign-in, a quota limit or an \
                 update it asks for)"
                    .into(),
            )
        } else {
            None
        }
    }
}

fn fallback_models() -> Vec<LabelModel> {
    FALLBACK_MODELS.map(|(id, label)| LabelModel { id: id.into(), label: label.into(), fast: false }).to_vec()
}

/// Maps an Antigravity effort suffix onto the closest [`LabelEffort`] tier for run/benchmark metadata.
fn effort_for_suffix(suffix: &str) -> LabelEffort {
    match suffix {
        "minimal" | "low" => LabelEffort::Low,
        "medium" => LabelEffort::Medium,
        "high" | "xhigh" | "max" => LabelEffort::High,
        _ => LabelEffort::Low,
    }
}

/// Splits an effort suffix (`-low`, `-medium`, `-high`, `-xhigh`, `-max`, `-minimal`) off a model slug.
pub(super) fn split_effort_suffix(slug: &str) -> Option<(&str, &'static str)> {
    let lower = slug.to_ascii_lowercase();
    for suffix in EFFORT_SUFFIXES {
        let pat_len = suffix.len() + 1;
        if lower.len() > pat_len && lower.ends_with(suffix) && lower.as_bytes()[lower.len() - pat_len] == b'-'
        {
            return Some((&slug[..slug.len() - pat_len], suffix));
        }
    }
    None
}

/// Strips a trailing effort parenthetical such as `" (XHigh)"` or `" (Low)"` from a display label.
fn strip_effort_label(label: &str) -> &str {
    let trimmed = label.trim();
    if let Some(open) = trimmed.rfind(" (")
        && let Some(inner) = trimmed[open + 2..].strip_suffix(')')
        && EFFORT_SUFFIXES.iter().any(|e| inner.eq_ignore_ascii_case(e))
    {
        return trimmed[..open].trim_end();
    }
    trimmed
}

/// Strips ANSI CSI (`ESC [ ... final`) and OSC (`ESC ] ... BEL/ST`) escape sequences from one line of CLI output.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match chars.peek().copied() {
            Some('[') => {
                chars.next();
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
            Some(']') => {
                chars.next();
                while let Some(ch) = chars.next() {
                    if ch == '\x07' || (ch == '\x1b' && chars.peek() == Some(&'\\')) {
                        if ch == '\x1b' {
                            chars.next();
                        }
                        break;
                    }
                }
            }
            Some(_) => {
                chars.next();
            }
            None => break,
        }
    }
    out
}

/// Parses `agy models` plain-text output (`<slug>   <Display Name>`) into base models, stripping ANSI formatting,
/// ignoring diagnostics/timestamps/prose, deduplicating effort-suffixed variants, and skipping multi-model
/// orchestrators.
fn parse_models(stdout: &str) -> Vec<LabelModel> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for raw_line in stdout.lines() {
        let cleaned = strip_ansi(raw_line);
        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            continue;
        }
        let slug_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        let raw_slug = &trimmed[..slug_end];
        let rest = &trimmed[slug_end..];
        let tabular = rest.starts_with("  ") || rest.starts_with('\t');
        let raw_label = rest.trim();
        if !looks_like_model_slug(raw_slug, tabular, raw_label) {
            continue;
        }
        let lower = raw_slug.to_ascii_lowercase();
        if lower.starts_with("gemini-pro-orchestrator") {
            continue;
        }
        let (base, label) = match split_effort_suffix(raw_slug) {
            Some((base, _)) => (base, strip_effort_label(raw_label)),
            None => (raw_slug, raw_label),
        };
        let id = base.to_string();
        if !seen.insert(id.to_ascii_lowercase()) {
            continue;
        }
        let label = if label.is_empty() { id.clone() } else { label.to_string() };
        out.push(LabelModel { id, label, fast: false });
    }
    out
}

fn looks_like_model_slug(s: &str, tabular: bool, label: &str) -> bool {
    if s.is_empty() || s.len() > 100 {
        return false;
    }
    let bytes = s.as_bytes();
    // Model slugs start with an ASCII letter (rules out ISO timestamps like `2026-11-22T...` and numbers) and end
    // with an alphanumeric character (rules out `Error:` / `Warning:`).
    if !bytes[0].is_ascii_alphabetic() || !bytes[bytes.len() - 1].is_ascii_alphanumeric() {
        return false;
    }
    if !bytes.iter().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':' | b'/')) {
        return false;
    }
    let has_sep = bytes.iter().any(|b| matches!(b, b'-' | b'_' | b'/'));
    let has_digit = bytes.iter().any(|b| b.is_ascii_digit());
    if !has_sep && !has_digit {
        return false;
    }
    // Without a digit in the slug, require tabular `<slug>   <Display Name>` formatting so hyphenated prose lines
    // like `sign-in required` never become bogus models.
    if !has_digit && (!tabular || label.is_empty()) {
        return false;
    }
    if let Some(first_word) = label.split_whitespace().next()
        && matches!(
            first_word.to_ascii_uppercase().trim_end_matches(':'),
            "WARNING" | "WARN" | "ERROR" | "ERR" | "INFO" | "DEBUG" | "TRACE" | "FATAL"
        )
    {
        return false;
    }
    let lower = s.to_ascii_lowercase();
    !matches!(
        lower.as_str(),
        "sign-in"
            | "log-in"
            | "opt-in"
            | "re-run"
            | "rate-limit"
            | "not-found"
            | "non-zero"
            | "no-auth"
            | "pre-release"
            | "model-id"
    )
}
