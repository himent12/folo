//! Codex as a labeling [`Agent`]: `codex exec`, one turn per call.
//!
//! Codex is a coding agent first: by default every request carries its coding instructions and the definitions of
//! its tools. A labeling call needs none of that, so FOLO's instructions replace Codex's, the features that add
//! tools are switched off, and the user's config (plugins, MCP servers) is not loaded. That takes the fixed cost
//! of a call from about 20000 input tokens to about 8000; the rest is Codex's own and stays, which is why
//! [`Agent::default_batch`] is larger here. The images are attached with `--image`, and `--output-schema` holds
//! the reply to FOLO's JSON shape.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use folo_api::{LabelAgent, LabelAgentId, LabelModel, LabelUsage};
use serde_json::Value;

use super::agent::{
    Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Memo, Output, last_line, run,
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

/// Config keys that drop further text a labeling call has no use for.
const CONFIG: [&str; 4] = [
    "web_search='disabled'",
    "include_environment_context=false",
    "include_permissions_instructions=false",
    "include_collaboration_mode_instructions=false",
];

/// Codex (`codex`, or the executable `FOLO_CODEX_BIN` names).
pub struct Codex {
    cli: Cli,
    models: Memo<Vec<LabelModel>>,
}

impl Codex {
    /// The agent, looking for `codex` when it is first asked.
    pub const fn new() -> Codex {
        Codex { cli: Cli::new("codex", "FOLO_CODEX_BIN"), models: Memo::new() }
    }

    /// The models of Codex's own catalog that take images (`codex debug models`); none when it cannot be read.
    fn models(&self) -> Vec<LabelModel> {
        self.models.get(|| {
            let Ok(path) = self.cli.path() else { return Vec::new() };
            let mut cmd = Command::new(path);
            cmd.args(["debug", "models"]);
            match run(cmd, None, CALL_TIMEOUT, &AtomicBool::new(false)) {
                Ok(out) if out.success => catalog_models(&out.stdout),
                _ => Vec::new(),
            }
        })
    }
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
        LabelAgent {
            id: self.id(),
            name: self.id().display_name().into(),
            available: version.is_some(),
            hint: version.is_none().then(|| self.cli.hint(INSTALL)),
            models: if version.is_some() { self.models() } else { Vec::new() },
            version,
            // Codex picks its own default; its config file, where a user would set one, is not loaded.
            default_model: None,
        }
    }

    fn default_batch(&self) -> usize {
        // Enough images to spread Codex's fixed cost per call thin.
        8
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        let path = self.cli.path()?;
        let files =
            Files::write(req).map_err(|e| AgentError::Unavailable(format!("preparing the call: {e}")))?;
        let cmd = command(&path, req, &files);
        parse(&run(cmd, Some(req.prompt.as_bytes().to_vec()), CALL_TIMEOUT, req.cancel)?)
    }

    fn hint(&self, message: &str) -> Option<String> {
        let m = message.to_lowercase();
        let signed_out = ["login", "log in", "logged in", "api key", "authenticat", "unauthorized", "401"];
        if signed_out.iter().any(|w| m.contains(w)) {
            Some("run `codex login` in a terminal on the server's machine, then start again".into())
        } else if m.contains("model") {
            Some("pick another model: this account may not have access to that one".into())
        } else if m.contains("limit") || m.contains("quota") {
            Some(
                "the account's usage limit was reached; start again once it resets, or use another agent"
                    .into(),
            )
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
    images: Vec<std::path::PathBuf>,
}

impl Files {
    fn write(req: &AgentRequest<'_>) -> std::io::Result<Files> {
        let files = Files {
            work: req.scratch.join("work"),
            instructions: req.scratch.join("instructions.md"),
            schema: req.scratch.join("schema.json"),
            images: (1..=req.images.len()).map(|i| req.scratch.join(format!("{i}.jpg"))).collect(),
        };
        std::fs::create_dir_all(&files.work)?;
        std::fs::write(&files.instructions, req.system)?;
        std::fs::write(&files.schema, serde_json::to_vec(req.schema).expect("JSON values serialize"))?;
        for (path, jpeg) in files.images.iter().zip(req.images) {
            std::fs::write(path, jpeg)?;
        }
        Ok(files)
    }
}

/// `codex exec` for one turn: FOLO's instructions, the tool features off, the prompt read from stdin.
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
fn parse(out: &Output) -> Result<AgentReply, AgentError> {
    let (mut text, mut usage, mut failure) = (None, None, None);
    for event in out.stdout.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        match event["type"].as_str() {
            Some("item.completed") if event["item"]["type"] == "agent_message" => {
                text = event["item"]["text"].as_str().map(str::to_string);
            }
            Some("turn.completed") => usage = Some(event["usage"].clone()),
            Some("turn.failed") => failure = event["error"]["message"].as_str().map(str::to_string),
            Some("error") => failure = event["message"].as_str().map(str::to_string),
            _ => {}
        }
    }
    let (Some(text), Some(usage)) = (text, usage) else {
        if let Some(failure) = failure {
            return Err(AgentError::Failed(folo_api::codes::cap(
                &failure,
                folo_api::codes::MAX_MESSAGE_CHARS,
            )));
        }
        let why = last_line(&out.stderr).or_else(|| last_line(&out.stdout));
        return Err(match why {
            Some(why) if !out.success => AgentError::Failed(why),
            _ => AgentError::Unavailable("`codex` printed no reply".into()),
        });
    };
    let tokens = |key: &str| usage[key].as_u64().unwrap_or(0);
    Ok(AgentReply {
        text,
        usage: LabelUsage {
            calls: 1,
            input_tokens: tokens("input_tokens"),
            cached_input_tokens: tokens("cached_input_tokens"),
            output_tokens: tokens("output_tokens"),
            cost_usd: None,
        },
    })
}

/// The listed models of `codex debug models` that accept images, in the catalog's order of preference.
fn catalog_models(json: &str) -> Vec<LabelModel> {
    let Ok(catalog) = serde_json::from_str::<Value>(json) else { return Vec::new() };
    let mut models: Vec<(u64, LabelModel)> = catalog["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|m| m["visibility"] == "list")
        .filter(|m| m["input_modalities"].as_array().is_some_and(|a| a.iter().any(|x| x == "image")))
        .filter_map(|m| {
            let id = m["slug"].as_str()?.to_string();
            let label = m["display_name"].as_str().unwrap_or(&id).to_string();
            Some((m["priority"].as_u64().unwrap_or(u64::MAX), LabelModel { id, label }))
        })
        .collect();
    models.sort_by_key(|(priority, _)| *priority);
    models.into_iter().map(|(_, m)| m).collect()
}

#[cfg(test)]
mod tests {
    use folo_api::LabelEffort;
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
        assert_eq!(err, AgentError::Failed("unexpected status 401 Unauthorized: sign in again".into()));
        assert!(Codex::new().hint("unexpected status 401 Unauthorized").unwrap().contains("codex login"));
        let err = parse(&output("", "Error: Unknown feature flag: nope\n", false)).unwrap_err();
        assert_eq!(err, AgentError::Failed("Error: Unknown feature flag: nope".into()));
        assert!(matches!(parse(&output("", "", true)).unwrap_err(), AgentError::Unavailable(_)));
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
            scratch: dir.path(),
            cancel: &cancel,
        };
        let files = Files::write(&req).unwrap();
        assert_eq!(std::fs::read_to_string(&files.instructions).unwrap(), "label things");
        assert_eq!(std::fs::read(&files.images[1]).unwrap(), vec![2, 2]);
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
        let instructions = config.iter().find(|c| c.starts_with("model_instructions_file='")).unwrap();
        assert!(instructions.ends_with("instructions.md'") && !instructions.contains('\\'), "{instructions}");
        // No argument needs quoting a batch shim could get wrong.
        assert!(args.iter().all(|a| !a.contains('"') && !a.contains('\n')));
        assert_eq!(toml_string("D:/data/x.md"), "'D:/data/x.md'");
        assert_eq!(toml_string(r#"C:/O'Brien/a "b".md"#), r#""C:/O'Brien/a \"b\".md""#);
    }

    #[test]
    fn an_error_before_a_completed_turn_was_a_retry() {
        let retried = "{\"type\":\"turn.started\"}\n{\"type\":\"error\",\"message\":\"stream disconnected; retrying 1/5\"}";
        let stdout = SUCCESS.replacen("{\"type\":\"turn.started\"}", retried, 1);
        assert!(parse(&output(&stdout, "", true)).is_ok());
    }

    #[test]
    fn catalog_lists_visible_image_models_by_priority() {
        let json = r#"{"models":[
            {"slug":"b","display_name":"B","visibility":"list","input_modalities":["text","image"],"priority":2},
            {"slug":"hidden","display_name":"H","visibility":"hide","input_modalities":["text","image"],"priority":0},
            {"slug":"text","display_name":"T","visibility":"list","input_modalities":["text"],"priority":1},
            {"slug":"a","visibility":"list","input_modalities":["image"],"priority":1}]}"#;
        let models = catalog_models(json);
        assert_eq!(
            models,
            [
                LabelModel { id: "a".into(), label: "a".into() },
                LabelModel { id: "b".into(), label: "B".into() }
            ]
        );
        assert!(catalog_models("not json").is_empty() && catalog_models("{}").is_empty());
    }
}
