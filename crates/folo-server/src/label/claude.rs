//! Claude Code as a labeling [`Agent`]: `claude -p`, one request per call.
//!
//! The call is stripped to what labeling needs, because everything else is paid for in tokens on every call:
//! FOLO's instructions replace Claude Code's system prompt, there are no tools, MCP servers, skills, hooks,
//! settings or `CLAUDE.md`, and no session is written. The images travel on stdin inside a `stream-json` user
//! message, so the model sees them in its first (and only) request instead of reading files through a tool.
//!
//! Three environment variables take off what the flags leave ([`ENV`]; measured on Claude Code 2.1.326): prompt
//! caching, which on a subscription writes the whole request (the images too) to a one-hour cache at twice the
//! input price although nothing of it but a few hundred tokens of preamble is ever read back; the second request
//! Claude Code makes to title the session; and a server-side advisor tool. A call's fixed cost, FOLO's instructions
//! included, is then about 900 input tokens (measured as the difference between 4 and 8 photos a call). `--bare`
//! would take off more, but it reads only an API key, never a subscription's sign-in.

use std::path::Path;
use std::process::Command;

use base64::Engine;
use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelModel, LabelUsage};
use serde_json::{Value, json};

use super::agent::{
    Agent, AgentError, AgentReply, AgentRequest, CALL_TIMEOUT, Cli, Output, fault_hint, last_line,
    limit_hint, run,
};

/// The model a run uses when it names none: on photos Haiku finds about as many objects as Sonnet, with fewer
/// wrong boxes, for a twelfth of its cost; Opus finds more but also draws more wrong boxes, for 25 times the cost
/// (docs/LABELING.md has the numbers).
const DEFAULT_MODEL: &str = "haiku";
/// Claude Code's own aliases, which always name the latest model of each tier.
const MODELS: [(&str, &str); 3] = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")];
const INSTALL: &str = "Install Claude Code (https://claude.com/claude-code) on the server's machine and sign in by \
                       running `claude` once";

/// The environment of every call (see the module docs): no prompt caching (a cache write costs more than the
/// input it caches, and a call's images are never asked about again), no requests besides the call's own (the
/// session title, telemetry), no advisor tool.
const ENV: [(&str, &str); 3] = [
    ("DISABLE_PROMPT_CACHING", "1"),
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
    ("CLAUDE_CODE_DISABLE_ADVISOR_TOOL", "1"),
];

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
            models: MODELS
                .map(|(id, label)| LabelModel { id: id.into(), label: label.into(), fast: false })
                .to_vec(),
            default_model: Some(DEFAULT_MODEL.into()),
            // Measured November 2026 (Claude Code 2.1.326): Haiku at low effort labels as well as at medium
            // (found 77-78% against 77%, on 60 COCO photos in two passes) for 20% less and in less time.
            default_effort: LabelEffort::Low,
            // Claude Code's fast mode is not offered in `-p` ("not available in the Agent SDK").
            fast: false,
        }
    }

    fn default_batch(&self) -> usize {
        // Eight photos a call cost 18% less input than four (a call's fixed cost is about 900 tokens), but Haiku
        // then found 72-74% of the objects instead of 77-78% (two passes each): it labels best a few at a time.
        4
    }

    fn image_side(&self) -> u32 {
        // Haiku labels worse on smaller images (a 640 px photo sent at 480 px: found 75% against 79%, IoU 0.80
        // against 0.84) and better on larger ones (scaled up to 1024 px: 83%), so photos go at 1024 px.
        super::AGENT_IMAGE_SIDE
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
        } else if let Some(hint) = fault_hint(message) {
            Some(hint)
        } else if m.contains("limit") {
            Some(limit_hint(message))
        } else if m.contains("model") {
            Some("pick another model: this account may not have access to that one".into())
        } else if m.contains("printed no result") {
            Some("run `claude -p hi` in a terminal on the server's machine to see why (a sign-in, a limit or an update it asks for)".into())
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
        .args(["--effort", req.effort.as_str()])
        .envs(ENV);
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

/// Reads the `result` event of Claude Code's `stream-json` output. An error result says what went wrong in its
/// structured fields ([`error_message`]) and still reports what the call used, which the failure carries.
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
            Some(why) if !out.success => AgentError::failed(why),
            _ => AgentError::Unavailable("`claude` printed no result".into()),
        });
    };
    let usage = usage_of(&result);
    if result["is_error"].as_bool().unwrap_or(!out.success) {
        return Err(AgentError::Failed(error_message(&result), usage));
    }
    Ok(AgentReply { text: result["result"].as_str().unwrap_or_default().to_string(), usage })
}

/// What a result event says the call used. A call that never reached the model (no tokens, no cost) is no call.
fn usage_of(result: &Value) -> LabelUsage {
    let tokens = |key: &str| result["usage"][key].as_u64().unwrap_or(0);
    let cached = tokens("cache_read_input_tokens");
    let mut usage = LabelUsage {
        calls: 1,
        input_tokens: tokens("input_tokens") + tokens("cache_creation_input_tokens") + cached,
        cached_input_tokens: cached,
        output_tokens: tokens("output_tokens"),
        cost_usd: result["total_cost_usd"].as_f64(),
    };
    let paid = usage.input_tokens + usage.output_tokens > 0 || usage.cost_usd.is_some_and(|c| c > 0.0);
    if result["is_error"].as_bool() == Some(true) && !paid {
        usage = LabelUsage::default();
    }
    usage
}

/// The message of an error result: its text (`result`, else the `errors` of an `error_during_execution`), with
/// the API's status and error type when the result has them (`api_error_status`: "API Error: 503 api_error"), so
/// that a passing fault or a limit reads as one ([`super::agent::is_transient`]); the subtype when there is
/// nothing else.
fn error_message(result: &Value) -> String {
    /// Adds `part` unless it is empty or already said.
    fn add(parts: &mut Vec<String>, part: &str) {
        let part = part.trim();
        if !part.is_empty() && !parts.iter().any(|p| p.contains(part)) {
            parts.push(part.to_string());
        }
    }
    let mut parts: Vec<String> = Vec::new();
    add(&mut parts, result["result"].as_str().unwrap_or_default());
    for error in result["errors"].as_array().into_iter().flatten() {
        match error {
            Value::String(text) => add(&mut parts, text),
            other => {
                add(&mut parts, &other["message"].as_str().map_or_else(|| other.to_string(), str::to_string))
            }
        }
    }
    match &result["api_error"] {
        Value::String(text) => add(&mut parts, text),
        Value::Object(_) => add(&mut parts, result["api_error"]["message"].as_str().unwrap_or_default()),
        _ => {}
    }
    if let Some(status) = result["api_error_status"].as_u64()
        && !parts.iter().any(|p| p.contains(&status.to_string()))
    {
        let kind =
            result["api_error_type"].as_str().or(result["api_error_code"].as_str()).unwrap_or_default();
        add(&mut parts, &format!("API Error: {status} {kind}"));
    }
    if parts.is_empty() {
        let why = match result["subtype"].as_str() {
            Some(subtype) if subtype != "success" => format!("Claude Code reported an error ({subtype})"),
            _ => "Claude Code reported an error".to_string(),
        };
        add(&mut parts, &why);
    }
    folo_api::codes::cap(&parts.join(" · "), folo_api::codes::MAX_MESSAGE_CHARS)
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
        let AgentError::Failed(message, used) = &err else { panic!("{err:?}") };
        assert!(message.starts_with("There's an issue with the selected model (nope)"));
        assert_eq!(*used, LabelUsage::default(), "it never reached the model");
        assert!(ClaudeCode::new().hint(message).unwrap().contains("another model"));
        let hint = ClaudeCode::new().hint("Invalid API key · Please run /login").unwrap();
        assert!(hint.contains("sign in"), "{hint}");
        // A limit says when it resets; an overloaded provider is not "pick another model".
        let hint =
            ClaudeCode::new().hint("You've hit your session limit · resets 3pm (Europe/Berlin)").unwrap();
        assert!(
            hint.contains("usage limit was reached (resets 3pm (Europe/Berlin)): start again then"),
            "{hint}"
        );
        let hint = ClaudeCode::new()
            .hint("Opus is experiencing high load, please use /model to switch to Sonnet")
            .unwrap();
        assert_eq!(hint, crate::label::agent::BUSY);
    }

    /// A whole `stream-json` output whose call failed: the init event, the API error as the assistant's message,
    /// and `result` (`result_fields` are its fields after `type`).
    fn failed_stream(api_error: &str, result_fields: &str) -> String {
        let assistant = json!({
            "type": "assistant",
            "message": { "role": "assistant", "content": [{ "type": "text", "text": api_error }] },
            "isApiErrorMessage": true,
        });
        format!(
            "{{\"type\":\"system\",\"subtype\":\"init\",\"tools\":[],\"model\":\"claude-haiku-5-5\"}}\n{assistant}\n\
             {{\"type\":\"result\",{result_fields}}}\n"
        )
    }

    #[test]
    fn an_error_during_execution_is_read_from_its_errors_with_what_it_used() {
        // A server error that outlived Claude Code's own retries, reported in `errors`: a passing fault, with
        // what the attempts used.
        let api_error = r#"API Error: 503 {"type":"error","error":{"type":"api_error","message":"upstream connect error"}}"#;
        let fields = format!(
            r#""subtype":"error_during_execution","is_error":true,"num_turns":1,"total_cost_usd":0.0004,"usage":{{"input_tokens":1200,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}},"errors":[{}]"#,
            serde_json::to_string(api_error).unwrap()
        );
        let err = parse(&output(&failed_stream(api_error, &fields), "", false)).unwrap_err();
        let AgentError::Failed(message, used) = &err else { panic!("{err:?}") };
        assert!(message.starts_with("API Error: 503"), "{message}");
        assert!(crate::label::agent::is_transient(message));
        assert_eq!(ClaudeCode::new().hint(message).as_deref(), Some(crate::label::agent::BUSY));
        assert_eq!((used.calls, used.input_tokens, used.cost_usd), (1, 1200, Some(0.0004)));
        // The status alone, in the structured fields of a result without text: still a passing fault.
        let fields = r#""subtype":"success","is_error":true,"result":"","api_error_status":529,"api_error_type":"overloaded_error","usage":{"input_tokens":0,"output_tokens":0}"#;
        let err = parse(&output(&failed_stream("API Error", fields), "", false)).unwrap_err();
        let AgentError::Failed(message, used) = &err else { panic!("{err:?}") };
        assert_eq!(message, "API Error: 529 overloaded_error");
        assert!(crate::label::agent::is_transient(message));
        assert_eq!(*used, LabelUsage::default());
        // A limit in `errors`: not asked again, and the hint keeps when it resets.
        let limit = "You've hit your session limit · resets 3pm (Europe/Berlin)";
        let fields = format!(
            r#""subtype":"error_during_execution","is_error":true,"api_error_status":429,"usage":{{}},"errors":["{limit}"]"#
        );
        let err = parse(&output(&failed_stream(limit, &fields), "", false)).unwrap_err();
        let AgentError::Failed(message, _) = &err else { panic!("{err:?}") };
        assert!(!crate::label::agent::is_transient(message), "{message}");
        let hint = ClaudeCode::new().hint(message).unwrap();
        assert!(hint.contains("usage limit was reached (resets 3pm (Europe/Berlin))"), "{hint}");
        // Nothing but the subtype.
        let fields = r#""subtype":"error_max_turns","is_error":true"#;
        let err = parse(&output(&failed_stream("", fields), "", false)).unwrap_err();
        assert_eq!(err, AgentError::failed("Claude Code reported an error (error_max_turns)"));
    }

    #[test]
    fn no_result_event_reports_what_the_cli_printed() {
        let err = parse(&output("", "error: unknown option '--safe-mode'\n", false)).unwrap_err();
        assert_eq!(err, AgentError::failed("error: unknown option '--safe-mode'"));
        assert!(ClaudeCode::new().hint("error: unknown option '--safe-mode'").unwrap().contains("update"));
        let err = parse(&output("not json\n", "", true)).unwrap_err();
        let AgentError::Unavailable(message) = &err else { panic!("{err:?}") };
        // Nothing to go on: the hint is how to see what the CLI says.
        assert!(ClaudeCode::new().hint(message).unwrap().contains("`claude -p hi`"));
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
            fast: false,
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
        // No prompt caching, no title request, no advisor.
        let env: Vec<(String, String)> = cmd
            .get_envs()
            .filter_map(|(k, v)| Some((k.to_string_lossy().into_owned(), v?.to_string_lossy().into_owned())))
            .collect();
        for var in [
            "DISABLE_PROMPT_CACHING",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL",
        ] {
            assert!(env.contains(&(var.to_string(), "1".to_string())), "{var} missing from {env:?}");
        }

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
