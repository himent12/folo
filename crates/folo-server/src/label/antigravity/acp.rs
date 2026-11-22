//! ACP v1 JSON-RPC transport and single-turn driver for `agy acp`.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use base64::Engine;
use folo_api::{LabelEffort, LabelUsage};
use serde_json::{Value, json};

use super::split_effort_suffix;
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, POLL, last_line};
use crate::label::tree::Tree;

/// How long to wait for `agy acp` to exit cleanly after closing stdin before killing its process tree.
const CLOSE_GRACE: Duration = Duration::from_millis(250);

/// Process-tree guard for `agy acp`: ensures the child and everything it spawned are killed on every exit path.
struct AcpProcess {
    child: Child,
    tree: Tree,
    reaped: bool,
}

impl AcpProcess {
    fn spawn(path: &Path, work: &Path) -> Result<AcpProcess, AgentError> {
        let mut cmd = acp_command(path, work);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        Tree::prepare(&mut cmd);
        let program = cmd.get_program().to_string_lossy().into_owned();
        let unavailable = |e: std::io::Error| AgentError::Unavailable(format!("starting `{program}`: {e}"));
        let mut child = cmd.spawn().map_err(unavailable)?;
        let tree = Tree::start(&child).map_err(|e| {
            let _ = child.kill();
            let _ = child.wait();
            unavailable(e)
        })?;
        Ok(AcpProcess { child, tree, reaped: false })
    }

    fn finish_gracefully(&mut self) {
        if self.reaped {
            return;
        }
        let deadline = Instant::now() + CLOSE_GRACE;
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                self.reaped = true;
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        self.kill();
    }

    fn kill(&mut self) {
        if !self.reaped {
            self.tree.kill(&mut self.child);
            self.reaped = true;
        }
    }
}

impl Drop for AcpProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Builds the `agy acp` command in an empty working directory with default customizations and telemetry off.
fn acp_command(path: &Path, work: &Path) -> Command {
    let mut cmd = Command::new(path);
    cmd.args(["acp", "--include-default-customizations=false", "--disable-telemetry"]).current_dir(work);
    cmd
}

/// Spawns `agy acp`, drives one ACP v1 labeling turn, and cleans up the child tree.
pub(super) fn run_acp(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
) -> Result<AgentReply, AgentError> {
    if req.cancel.load(Ordering::Relaxed) {
        return Err(AgentError::Cancelled(LabelUsage::default()));
    }
    let work = req.scratch.join("work");
    std::fs::create_dir_all(&work)
        .map_err(|e| AgentError::Unavailable(format!("preparing the call: {e}")))?;

    let mut proc = AcpProcess::spawn(path, &work)?;
    let stdin = proc.child.stdin.take().expect("stdin is piped");
    let stdout = proc.child.stdout.take().expect("stdout is piped");
    let stderr = proc.child.stderr.take().expect("stderr is piped");

    // Dedicated stdin writer thread so a wedged child never blocks deadline/cancel checks on large image batches.
    let (in_tx, in_rx) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut stdin = stdin;
        while let Ok(bytes) = in_rx.recv() {
            if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                break;
            }
        }
    });

    // Dedicated stdout line reader thread.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if out_tx.send(line).is_err() {
                break;
            }
        }
    });

    // Dedicated stderr drain thread so stderr never blocks and the tail is available on unexpected exit.
    let stderr_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stderr_out = stderr_buf.clone();
    std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut chunk = [0u8; 8 * 1024];
        while let Ok(n @ 1..) = stderr.read(&mut chunk) {
            stderr_out.lock().unwrap().extend_from_slice(&chunk[..n]);
        }
    });

    let mut send = |msg: &Value| -> bool {
        let mut bytes = serde_json::to_vec(msg).expect("JSON values serialize");
        bytes.push(b'\n');
        in_tx.send(bytes).is_ok()
    };
    let read_stderr = || String::from_utf8_lossy(&stderr_buf.lock().unwrap()).into_owned();

    let result = drive_acp(req, &work, timeout, &mut send, &out_rx, &read_stderr);

    // Close stdin so the child can exit cleanly; if it fails to exit promptly (or on cancel/timeout), kill the tree.
    drop(in_tx);
    match &result {
        Ok(_) => proc.finish_gracefully(),
        Err(AgentError::Cancelled(_) | AgentError::Timeout(_)) => proc.kill(),
        Err(_) => proc.finish_gracefully(),
    }
    result
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

fn match_option_value<'a>(values: &'a [(String, String)], query: &str) -> Option<&'a str> {
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

/// Resolves the requested model string and effort against the session's `model` config option.
/// Supports both base ids (`gemini-4-neon`) and effort-suffixed slugs (`gemini-4-neon-xhigh`).
pub(super) fn resolve_model_and_effort(
    requested_model: Option<&str>,
    req_effort: LabelEffort,
    model_opt: Option<&ConfigOption>,
) -> (Option<String>, String) {
    let default_effort = req_effort.as_str().to_string();
    let Some(raw) = requested_model.map(str::trim).filter(|m| !m.is_empty()) else {
        return (None, default_effort);
    };
    let values = model_opt.map(|o| o.values.as_slice()).unwrap_or(&[]);
    if let Some(exact) = match_option_value(values, raw) {
        return (Some(exact.to_string()), default_effort);
    }
    if let Some((base, suffix)) = split_effort_suffix(raw) {
        if let Some(matched) = match_option_value(values, base) {
            return (Some(matched.to_string()), suffix.to_string());
        }
        if values.is_empty() {
            return (Some(base.to_string()), suffix.to_string());
        }
    }
    (Some(raw.to_string()), default_effort)
}

/// Maps a desired effort string onto one of the session's available `thought_level` values.
pub(super) fn resolve_effort_value(desired: &str, opt: &ConfigOption) -> Option<String> {
    if opt.values.is_empty() {
        return Some(desired.to_string());
    }
    if let Some(exact) = match_option_value(&opt.values, desired) {
        return Some(exact.to_string());
    }
    let fallbacks: &[&str] = match desired.to_ascii_lowercase().as_str() {
        "minimal" => &["low", "medium"],
        "low" => &["minimal", "medium"],
        "medium" => &["low", "high"],
        "high" => &["xhigh", "max", "medium"],
        "xhigh" => &["max", "high", "medium"],
        "max" => &["xhigh", "high", "medium"],
        _ => &[],
    };
    for cand in fallbacks {
        if let Some(v) = match_option_value(&opt.values, cand) {
            return Some(v.to_string());
        }
    }
    opt.values.first().map(|(v, _)| v.clone())
}

/// Accumulates token and USD cost readings across `usage_update` notifications and the final `PromptResponse`.
#[derive(Default)]
struct UsageTracker {
    has_detailed: bool,
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
    fallback_used: u64,
    cost_usd: Option<f64>,
    seen: bool,
}

impl UsageTracker {
    fn observe_usage_update(&mut self, update: &Value) {
        if let Some(used) = update["used"].as_u64() {
            self.fallback_used = self.fallback_used.max(used);
            self.seen = true;
        }
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

    fn observe_prompt_result(&mut self, result: &Value) {
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
            self.seen = true;
        }
    }

    fn observe_usage_obj(&mut self, obj: &Value) {
        let read = |a: &str, b: &str| obj[a].as_u64().or_else(|| obj[b].as_u64());
        let input = read("inputTokens", "input_tokens");
        let output = read("outputTokens", "output_tokens");
        let cached = read("cachedReadTokens", "cached_read_tokens")
            .or_else(|| read("cachedInputTokens", "cached_input_tokens"));
        let thought = read("thoughtTokens", "thought_tokens")
            .or_else(|| read("thinkingTokens", "thinking_tokens"))
            .or_else(|| read("reasoningOutputTokens", "reasoning_output_tokens"));
        let total = read("totalTokens", "total_tokens");

        if input.is_none() && output.is_none() && cached.is_none() && thought.is_none() {
            return;
        }
        self.has_detailed = true;
        self.seen = true;
        let raw_in = input.unwrap_or(0);
        let raw_cached = cached.unwrap_or(0);
        let raw_out = output.unwrap_or(0);
        let raw_thought = thought.unwrap_or(0);
        // `LabelUsage::input_tokens` includes cached reads, and `output_tokens` includes reasoning/thinking tokens.
        let total_in = if raw_cached > raw_in { raw_in + raw_cached } else { raw_in };
        let total_out = match total {
            Some(t) if raw_thought > 0 && raw_in + raw_out + raw_thought == t => raw_out + raw_thought,
            Some(t) if raw_in + raw_out == t => raw_out,
            _ => raw_out + raw_thought,
        };
        self.input_tokens = total_in;
        self.cached_input_tokens = raw_cached.min(total_in);
        self.output_tokens = total_out;
    }

    fn to_usage(&self, calls: u64) -> LabelUsage {
        if !self.seen && calls == 0 {
            return LabelUsage::default();
        }
        let (input_tokens, cached_input_tokens, output_tokens) = if self.has_detailed {
            (self.input_tokens, self.cached_input_tokens, self.output_tokens)
        } else {
            (self.fallback_used, 0, 0)
        };
        LabelUsage {
            calls: if self.seen { calls.max(1) } else { calls },
            input_tokens,
            cached_input_tokens,
            output_tokens,
            cost_usd: self.cost_usd,
        }
    }
}

/// Extracts a [`Failure`] (full message + optional HTTP status) from a JSON-RPC `error` object.
fn rpc_failure(err: &Value, used: LabelUsage) -> Failure {
    let mut message = err["message"].as_str().unwrap_or("ACP request failed").trim().to_string();
    let mut status: Option<u16> = None;
    if let Some(data) = err.get("data") {
        let extra = data
            .as_str()
            .or_else(|| data["message"].as_str())
            .or_else(|| data["details"].as_str())
            .or_else(|| data["detail"].as_str())
            .or_else(|| data["error"]["message"].as_str());
        if let Some(extra) = extra.map(str::trim).filter(|s| !s.is_empty())
            && !message.contains(extra)
        {
            if !message.is_empty() {
                message.push_str(": ");
            }
            message.push_str(extra);
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
fn reject_permission(id: &Value, params: &Value) -> Value {
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
fn prompt_blocks(req: &AgentRequest<'_>, system_appended: bool) -> Vec<Value> {
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

/// The ACP v1 JSON-RPC conversation for one labeling call, decoupled from OS processes so unit tests can drive it
/// directly over in-memory channels.
pub(super) fn drive_acp(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<String>,
    stderr_text: &dyn Fn() -> String,
) -> Result<AgentReply, AgentError> {
    let deadline = Instant::now() + timeout;
    let mut next_id: u64 = 1;
    let mut usage = UsageTracker::default();
    let mut reply_text = String::new();
    let mut session_id: Option<String> = None;

    let mut rpc_call = |method: &str,
                        params: Value,
                        collect_reply: bool,
                        session_id: &Option<String>,
                        usage: &mut UsageTracker,
                        reply_text: &mut String|
     -> Result<Value, AgentError> {
        let id = next_id;
        next_id += 1;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if !send(&msg) {
            let err = last_line(&stderr_text());
            return Err(match err {
                Some(why) => AgentError::Failed(Failure { used: usage.to_usage(0), ..Failure::new(why) }),
                None => AgentError::Unavailable("`agy` exited before replying".into()),
            });
        }
        loop {
            if req.cancel.load(Ordering::Relaxed) {
                if let Some(sid) = session_id {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                }
                return Err(AgentError::Cancelled(usage.to_usage(0)));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if let Some(sid) = session_id {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                }
                return Err(AgentError::Timeout(usage.to_usage(0)));
            }
            match recv.recv_timeout(remaining.min(POLL)) {
                Ok(line) => {
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                    // Server -> client request: answer immediately so the agent never deadlocks waiting for tools.
                    if let (Some(req_id), Some(req_method)) = (msg.get("id"), msg["method"].as_str()) {
                        let response = if req_method == "session/request_permission" {
                            reject_permission(req_id, &msg["params"])
                        } else {
                            json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": { "code": -32601, "message": "Method not supported in labeling mode" }
                            })
                        };
                        let _ = send(&response);
                        continue;
                    }
                    // Notification (`session/update`).
                    if msg["method"].as_str() == Some("session/update") {
                        let update = &msg["params"]["update"];
                        match update["sessionUpdate"].as_str() {
                            Some("agent_message_chunk") if collect_reply => {
                                if update["content"]["type"] == "text"
                                    && let Some(t) = update["content"]["text"].as_str()
                                {
                                    reply_text.push_str(t);
                                }
                            }
                            Some("usage_update") => usage.observe_usage_update(update),
                            _ => {}
                        }
                        continue;
                    }
                    // Response to our pending call.
                    if msg.get("id").and_then(Value::as_u64) == Some(id) {
                        if let Some(err) = msg.get("error") {
                            return Err(AgentError::Failed(rpc_failure(err, usage.to_usage(0))));
                        }
                        return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let err = last_line(&stderr_text());
                    return Err(match err {
                        Some(why) => {
                            AgentError::Failed(Failure { used: usage.to_usage(0), ..Failure::new(why) })
                        }
                        None => AgentError::Unavailable("`agy` exited before replying".into()),
                    });
                }
            }
        }
    };

    // 1. Handshake (`initialize`).
    let init = rpc_call(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": { "name": "folo", "version": env!("CARGO_PKG_VERSION") },
        }),
        false,
        &session_id,
        &mut usage,
        &mut reply_text,
    )?;
    let caps = &init["agentCapabilities"];
    if !req.images.is_empty() && caps["promptCapabilities"]["image"].as_bool() == Some(false) {
        return Err(AgentError::Unavailable("`agy acp` does not support inline image prompts".into()));
    }
    let system_append_supported =
        caps["_meta"]["antigravity"]["systemPrompt"]["append"].as_bool() == Some(true);
    let close_supported = caps["sessionCapabilities"].get("close").is_some_and(|v| !v.is_null());

    // 2. Open an isolated session in the empty scratch folder (`session/new`).
    let mut new_params = json!({
        "cwd": folo_api::json::path(work),
        "mcpServers": [],
    });
    if system_append_supported {
        let append = format!(
            "{}\nDo not call any tools or read any files; every image is already attached inline. Reply with one \
             JSON object and nothing else.",
            req.system
        );
        new_params["_meta"] = json!({ "antigravity": { "systemPrompt": { "append": append } } });
    }
    let session = rpc_call("session/new", new_params, false, &session_id, &mut usage, &mut reply_text)?;
    let sid = session["sessionId"]
        .as_str()
        .ok_or_else(|| AgentError::Unavailable("`agy acp` returned no sessionId".into()))?
        .to_string();
    session_id = Some(sid.clone());
    let mut config_options = parse_config_options(session.get("configOptions"));

    // 3. Configure model and reasoning effort (`session/set_config_option`, with legacy `session/set_model` fallback).
    let (resolved_model, desired_effort) =
        resolve_model_and_effort(req.model, req.effort, find_model_option(&config_options));
    if let Some(model) = resolved_model {
        if let Some(opt) = find_model_option(&config_options).cloned() {
            if opt.current_value.as_deref() != Some(model.as_str()) {
                let updated = rpc_call(
                    "session/set_config_option",
                    json!({ "sessionId": sid, "configId": opt.id, "value": model }),
                    false,
                    &session_id,
                    &mut usage,
                    &mut reply_text,
                )?;
                let refreshed = parse_config_options(updated.get("configOptions"));
                if !refreshed.is_empty() {
                    config_options = refreshed;
                }
            }
        } else if session.get("models").is_some() {
            let _ = rpc_call(
                "session/set_model",
                json!({ "sessionId": sid, "modelId": model }),
                false,
                &session_id,
                &mut usage,
                &mut reply_text,
            )?;
        }
    }
    if let Some(opt) = find_effort_option(&config_options).cloned()
        && let Some(effort_value) = resolve_effort_value(&desired_effort, &opt)
        && opt.current_value.as_deref() != Some(effort_value.as_str())
    {
        let _ = rpc_call(
            "session/set_config_option",
            json!({ "sessionId": sid, "configId": opt.id, "value": effort_value }),
            false,
            &session_id,
            &mut usage,
            &mut reply_text,
        )?;
    }

    // 4. Send the single-turn prompt with inline base64 JPEG images (`session/prompt`).
    let prompt_result = rpc_call(
        "session/prompt",
        json!({ "sessionId": sid, "prompt": prompt_blocks(req, system_append_supported) }),
        true,
        &session_id,
        &mut usage,
        &mut reply_text,
    )?;
    usage.observe_prompt_result(&prompt_result);

    // 5. Best-effort clean session close.
    if close_supported {
        let close_id = next_id;
        let _ = send(&json!({
            "jsonrpc": "2.0",
            "id": close_id,
            "method": "session/close",
            "params": { "sessionId": sid }
        }));
    }

    let stop_reason = prompt_result["stopReason"].as_str().unwrap_or("end_turn");
    if stop_reason == "cancelled" {
        return Err(AgentError::Cancelled(usage.to_usage(0)));
    }
    if stop_reason == "refusal" {
        return Err(AgentError::Failed(Failure {
            used: usage.to_usage(1),
            ..Failure::new("Antigravity refused the prompt")
        }));
    }
    if reply_text.trim().is_empty() {
        let used = usage.to_usage(0);
        return Err(if usage.seen {
            AgentError::Failed(Failure { used, ..Failure::new("`agy` printed no reply") })
        } else {
            AgentError::Unavailable("`agy` printed no reply".into())
        });
    }
    Ok(AgentReply { text: reply_text, usage: usage.to_usage(1) })
}
