//! Codex as a labeling [`Agent`]: `codex exec`, one turn per call.
//!
//! Files: this one holds [`Codex`], its [`Agent`] implementation, scratch file layout, and `codex exec` command
//! builder; `catalog` reads `codex debug models`, checks `config.toml` conflicts, and strips tool entries into the
//! lean catalog; `parse` reads `codex exec --json` events and token usage; `tests` holds unit tests.
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
//!   entries' tools taken off (`catalog::lean_catalog`): the model, its instructions and everything else stay as
//!   Codex has them. When the catalog cannot be read, the user's config points Codex at another catalog or server
//!   (`catalog::catalog_conflict`), or Codex refuses the file, calls go without it, and the run's log says so once
//!   ([`Agent::notice`]).
//!
//! On codex-cli 0.169.0 that takes the fixed cost of a call from about 20000 input tokens (Codex's defaults) and
//! 7800 (the switches before November 2026) to about 50 of Codex's own; with FOLO's instructions and reply schema
//! a call's fixed cost is about 700, so a call costs about what its images do. The images are attached with
//! `--image`, at the size FOLO sends them (Codex does not scale them down: an image's input tokens grow with its
//! area), and `--output-schema` holds the reply to FOLO's JSON shape. [`AgentRequest::fast`] asks for Codex's
//! priority service tier (`service_tier`, what Codex calls Fast), which the catalog says which models offer.

mod catalog;
mod parse;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use folo_api::{LabelAgent, LabelAgentId, LabelEffort};

use self::catalog::{Catalog, codex_home, config_conflict, refused_catalog};
use self::parse::{parse, usage_of};
use super::agent::{
    Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Memo, Output, PROBE_TIMEOUT, fault_hint,
    last_line, limit_hint, run, run_kept,
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
pub(super) const FAST_TIER: &str = "priority";

/// The model provider a call uses: `codex exec --ignore-user-config` does not read a provider from the user's
/// config, so it is Codex's built-in one. `codex debug models` (which has no `--ignore-user-config`) is told the
/// same, so the catalog it prints is the one a call's model comes from.
const PROVIDER: &str = "model_provider='openai'";

/// What a call without the lean catalog costs more, in the words of a run's log.
pub(super) const COSTS_MORE: &str =
    "calls go without FOLO's lean model catalog and cost about 7800 more input tokens each";

/// Codex (`codex`, or the executable `FOLO_CODEX_BIN` names).
pub struct Codex {
    cli: Cli,
    catalog: Memo<Catalog>,
    /// The versions of Codex that refused the lean catalog ([`with_lean_fallback`]): their calls go without it.
    /// A refusal is recorded against the version of the catalog snapshot the call held ([`Codex::refuse`]), so a
    /// late one from an older snapshot neither turns off nor clears a newer version's; the probe and the catalog
    /// are then read again, so a Codex updated in place gets its own snapshot (and is tried with its own lean
    /// catalog).
    lean_refused: Mutex<BTreeSet<String>>,
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

    /// Notes that the version of Codex that printed `catalog` refused its lean catalog, and forgets the probe and
    /// the catalog while they are still that snapshot's: the next call reads the executable and its catalog again
    /// (Codex may have been updated in place since). A late refusal does not evict a newer snapshot.
    fn refuse(&self, catalog: &Catalog) {
        self.lean_refused.lock().unwrap().insert(catalog.version.clone());
        self.cli.forget_version(&catalog.version);
        self.catalog.forget_if(|held| held.version == catalog.version && held.path == catalog.path);
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
        // Fast as this call's catalog offers it (the one the run was started with may be older): a catalog that
        // was read and lists no fast tier for the model refuses it; one that could not be read leaves it out.
        let fast = req.fast && catalog.read;
        if fast {
            catalog.check_fast(req.model).map_err(AgentError::Unavailable)?;
        }
        let req = &AgentRequest { fast, ..*req };
        let lean = catalog.lean.as_deref().map(String::as_str).filter(|_| !self.refused(&catalog));
        let refuse = || self.refuse(&catalog);
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
