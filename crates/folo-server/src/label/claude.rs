//! Claude Code as a labeling [`Agent`]: `claude -p`, one request per call.
//!
//! The call is stripped to what labeling needs, because everything else is paid for in tokens on every call:
//! FOLO's instructions replace Claude Code's system prompt, there are no tools, MCP servers, skills, hooks,
//! settings or `CLAUDE.md`, and no session is written. The images travel on stdin inside a `stream-json` user
//! message, so the model sees them in its first (and only) request instead of reading files through a tool.

use std::path::Path;
use std::process::Command;

use base64::Engine;
use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelModel, LabelUsage};
use serde_json::{Value, json};

use super::agent::{Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Output, last_line, run};

/// The model a run uses when it names none: on photos, Haiku at medium effort finds about as many objects as
/// the larger models for a small fraction of their cost (docs/LABELING.md has the numbers).
const DEFAULT_MODEL: &str = "haiku";
/// Claude Code's own aliases, which always name the latest model of each tier.
const MODELS: [(&str, &str); 3] = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")];
const INSTALL: &str = "Install Claude Code (https://claude.com/claude-code) on the server's machine and sign in by \
                       running `claude` once";

/// Claude Code (`claude`, or the executable `FOLO_CLAUDE_BIN` names).
pub struct ClaudeCode {
    cli: Cli,
}

impl ClaudeCode {
    /// The agent, looking for `claude` when it is first asked.
    pub const fn new() -> ClaudeCode {
        ClaudeCode { cli: Cli::new("claude", "FOLO_CLAUDE_BIN") }
    }
}

impl Default for ClaudeCode {
    fn default() -> Self {
        ClaudeCode::new()
    }
}

impl Agent for ClaudeCode {
    fn id(&self) -> LabelAgentId {
        LabelAgentId::Claude
    }

    fn info(&self) -> LabelAgent {
        let version = self.cli.version();
        LabelAgent {
            id: self.id(),
            name: self.id().display_name().into(),
            available: version.is_some(),
            hint: version.is_none().then(|| self.cli.hint(INSTALL)),
            version,
            models: MODELS.map(|(id, label)| LabelModel { id: id.into(), label: label.into() }).to_vec(),
            default_model: Some(DEFAULT_MODEL.into()),
            // Haiku labels markedly better once it reasons a little; more than that buys little.
            default_effort: LabelEffort::Medium,
        }
    }

    fn default_batch(&self) -> usize {
        // A call's fixed cost is a few hundred tokens, so small batches cost nothing extra and fail small.
        4
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        let cmd = command(&self.cli.path()?, req);
        parse(&run(cmd, Some(stdin(req)), CALL_TIMEOUT, req.cancel)?)
    }

    fn hint(&self, message: &str) -> Option<String> {
        let m = message.to_lowercase();
        let signed_out = ["/login", "log in", "api key", "authenticat", "unauthorized", "oauth"];
        if signed_out.iter().any(|w| m.contains(w)) {
            Some("run `claude` in a terminal on the server's machine and sign in, then start again".into())
        } else if m.contains("unknown option") {
            Some("update Claude Code (`claude update`): FOLO needs version 2.1 or newer".into())
        } else if m.contains("model") {
            Some("pick another model: this account may not have access to that one".into())
        } else if m.contains("limit") {
            Some(
                "the account's usage limit was reached; start again once it resets, or use another agent"
                    .into(),
            )
        } else {
            None
        }
    }
}

/// `claude -p` for one request: FOLO's system prompt, nothing else loaded, the request read from stdin.
fn command(path: &Path, req: &AgentRequest<'_>) -> Command {
    let mut cmd = Command::new(path);
    cmd.current_dir(req.scratch)
        .args(["-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"])
        .args(["--system-prompt", req.system])
        // No tools at all, no MCP servers (none are configured, and the user's are ignored), and none of the
        // user's or the project's customisations.
        .args(["--tools", "", "--strict-mcp-config", "--safe-mode", "--disable-slash-commands"])
        .arg("--no-session-persistence")
        .args(["--effort", req.effort.as_str()]);
    if let Some(model) = req.model {
        cmd.args(["--model", model]);
    }
    cmd
}

/// The request as one `stream-json` user message: the prompt, then every image under its number.
fn stdin(req: &AgentRequest<'_>) -> Vec<u8> {
    let mut content = vec![json!({ "type": "text", "text": req.prompt })];
    for (i, jpeg) in req.images.iter().enumerate() {
        content.push(json!({ "type": "text", "text": format!("Image {}:", i + 1) }));
        content.push(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64::engine::general_purpose::STANDARD.encode(jpeg),
            },
        }));
    }
    let message = json!({ "type": "user", "message": { "role": "user", "content": content } });
    let mut line = serde_json::to_vec(&message).expect("JSON values serialize");
    line.push(b'\n');
    line
}

/// Reads the `result` event of Claude Code's `stream-json` output.
fn parse(out: &Output) -> Result<AgentReply, AgentError> {
    let result = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["type"] == "result");
    let Some(result) = result else {
        // No result event: the CLI stopped before it asked the model (a flag this version does not know, ...).
        let why = last_line(&out.stderr).or_else(|| last_line(&out.stdout));
        return Err(match why {
            Some(why) if !out.success => AgentError::Failed(why),
            _ => AgentError::Unavailable("`claude` printed no result".into()),
        });
    };
    let text = result["result"].as_str().unwrap_or_default().to_string();
    if result["is_error"].as_bool().unwrap_or(!out.success) {
        let why = if text.is_empty() { "Claude Code reported an error".to_string() } else { text };
        return Err(AgentError::Failed(folo_api::codes::cap(&why, folo_api::codes::MAX_MESSAGE_CHARS)));
    }
    let tokens = |key: &str| result["usage"][key].as_u64().unwrap_or(0);
    let cached = tokens("cache_read_input_tokens");
    Ok(AgentReply {
        text,
        usage: LabelUsage {
            calls: 1,
            input_tokens: tokens("input_tokens") + tokens("cache_creation_input_tokens") + cached,
            cached_input_tokens: cached,
            output_tokens: tokens("output_tokens"),
            cost_usd: result["total_cost_usd"].as_f64(),
        },
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    fn output(stdout: &str, stderr: &str, success: bool) -> Output {
        Output { success, stdout: stdout.into(), stderr: stderr.into() }
    }

    /// The lines Claude Code 2.1.302 prints for a successful call (trimmed to the fields that matter).
    const SUCCESS: &str = r#"{"type":"system","subtype":"init","tools":[],"mcp_servers":[],"model":"claude-haiku-5-5"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"{\"images\":[]}"}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":1,"result":"{\"images\":[{\"image\":1,\"boxes\":[]}]}","total_cost_usd":0.001055,"usage":{"input_tokens":2,"cache_creation_input_tokens":826,"cache_read_input_tokens":1084,"output_tokens":827}}"#;

    #[test]
    fn a_result_event_becomes_a_reply_with_usage() {
        let reply = parse(&output(SUCCESS, "", true)).unwrap();
        assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[]}]}"#);
        assert_eq!(
            reply.usage,
            LabelUsage {
                calls: 1,
                input_tokens: 1912,
                cached_input_tokens: 1084,
                output_tokens: 827,
                cost_usd: Some(0.001055)
            }
        );
    }

    #[test]
    fn an_error_result_is_a_failure_with_the_cli_message() {
        let stdout = r#"{"type":"result","subtype":"success","is_error":true,"result":"There's an issue with the selected model (nope). It may not exist or you may not have access to it.","usage":{}}"#;
        let err = parse(&output(stdout, "", false)).unwrap_err();
        let AgentError::Failed(message) = &err else { panic!("{err:?}") };
        assert!(message.starts_with("There's an issue with the selected model (nope)"));
        assert!(ClaudeCode::new().hint(message).unwrap().contains("another model"));
        let hint = ClaudeCode::new().hint("Invalid API key · Please run /login").unwrap();
        assert!(hint.contains("sign in"), "{hint}");
    }

    #[test]
    fn no_result_event_reports_what_the_cli_printed() {
        let err = parse(&output("", "error: unknown option '--safe-mode'\n", false)).unwrap_err();
        assert_eq!(err, AgentError::Failed("error: unknown option '--safe-mode'".into()));
        assert!(ClaudeCode::new().hint("error: unknown option '--safe-mode'").unwrap().contains("update"));
        let err = parse(&output("not json\n", "", true)).unwrap_err();
        assert!(matches!(err, AgentError::Unavailable(_)), "{err:?}");
    }

    #[test]
    fn the_command_loads_nothing_but_the_request() {
        let (schema, cancel) = (json!({}), AtomicBool::new(false));
        let req = AgentRequest {
            system: "label things",
            prompt: "Classes: car",
            images: &[vec![0xff, 0xd8, 0xff], vec![1, 2, 3]],
            schema: &schema,
            model: Some("haiku"),
            effort: LabelEffort::Medium,
            scratch: Path::new("."),
            cancel: &cancel,
        };
        let cmd = command(Path::new("claude"), &req);
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let joined = args.join(" ");
        for flag in [
            "-p",
            "--strict-mcp-config",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
        ] {
            assert!(args.iter().any(|a| a == flag), "{flag} missing from {joined}");
        }
        let after = |flag: &str| args[args.iter().position(|a| a == flag).unwrap() + 1].as_str();
        assert_eq!(after("--tools"), "");
        assert_eq!(after("--system-prompt"), "label things");
        assert_eq!((after("--model"), after("--effort")), ("haiku", "medium"));
        assert_eq!((after("--input-format"), after("--output-format")), ("stream-json", "stream-json"));
        // No argument has a line break (a Windows batch shim cannot take one).
        assert!(args.iter().all(|a| !a.contains('\n')));

        let line = stdin(&req);
        assert_eq!(line.last(), Some(&b'\n'));
        let message: Value = serde_json::from_slice(&line).unwrap();
        let content = message["message"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 5);
        assert_eq!(content[0]["text"], "Classes: car");
        assert_eq!(content[3]["text"], "Image 2:");
        assert_eq!(content[4]["source"]["data"], "AQID");
        assert_eq!(content[2]["source"]["media_type"], "image/jpeg");
    }
}
