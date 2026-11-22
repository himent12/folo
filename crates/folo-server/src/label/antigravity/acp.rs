//! ACP v1 subprocess lifecycle and single-turn JSON-RPC conversation driver for `agy acp`.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::{LabelEffort, LabelUsage};
use serde_json::{Value, json};

use super::protocol::{
    Isolation, LABEL_AGENT_NAME, LABEL_CLI_DIR, UsageTracker, find_effort_option, find_mode_option,
    find_model_option, is_permissive_mode, match_option_value, parse_config_options, parse_legacy_models,
    pick_safe_mode, prepare_isolation, prompt_blocks, reject_permission, resolve_effort_value,
    resolve_executable, resolve_model_and_effort, rpc_failure,
};
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, POLL, last_line};
use crate::label::tree::Tree;

/// How long to wait for `agy acp` to exit cleanly after closing stdin before killing its process tree.
const CLOSE_GRACE: Duration = Duration::from_millis(250);
/// How long to wait for a responsive `agy acp` child to acknowledge `session/cancel` and emit final usage.
pub(super) const CANCEL_GRACE: Duration = Duration::from_millis(500);
/// How long to wait for stderr to reach EOF after stdout closes before classifying an unexpected exit.
const STDERR_DRAIN: Duration = Duration::from_millis(250);
/// Maximum bytes retained from child stderr (most recent tail).
pub(super) const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Bounded stderr tail with EOF tracking so unexpected exits wait briefly for diagnostics without unbounded memory.
pub(super) struct StderrTail {
    bytes: Mutex<VecDeque<u8>>,
    ended: AtomicBool,
}

impl StderrTail {
    pub(super) fn new() -> Arc<StderrTail> {
        Arc::new(StderrTail { bytes: Mutex::new(VecDeque::new()), ended: AtomicBool::new(false) })
    }

    pub(super) fn push(&self, chunk: &[u8]) {
        let mut buf = self.bytes.lock().unwrap();
        if chunk.len() >= MAX_STDERR_BYTES {
            buf.clear();
            buf.extend(&chunk[chunk.len() - MAX_STDERR_BYTES..]);
            return;
        }
        let excess = (buf.len() + chunk.len()).saturating_sub(MAX_STDERR_BYTES);
        if excess > 0 {
            buf.drain(..excess);
        }
        buf.extend(chunk);
    }

    pub(super) fn mark_ended(&self) {
        self.ended.store(true, Ordering::Release);
    }

    pub(super) fn ended(&self) -> bool {
        self.ended.load(Ordering::Acquire)
    }

    pub(super) fn drain_text(&self, wait: Duration) -> String {
        let deadline = Instant::now() + wait;
        while !self.ended() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let buf = self.bytes.lock().unwrap();
        let (a, b) = buf.as_slices();
        let mut all = Vec::with_capacity(a.len() + b.len());
        all.extend_from_slice(a);
        all.extend_from_slice(b);
        String::from_utf8_lossy(&all).into_owned()
    }
}

/// Process-tree guard for `agy acp`: tracks child exit separately from process-group termination so descendants on
/// Unix (or Windows job members) are always killed even when the immediate child exits cleanly.
struct AcpProcess {
    child: Child,
    tree: Tree,
    child_exited: bool,
    tree_killed: bool,
}

impl AcpProcess {
    fn spawn(path: &Path, iso: &Isolation) -> Result<AcpProcess, AgentError> {
        let mut cmd = acp_command(path, iso);
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
        Ok(AcpProcess { child, tree, child_exited: false, tree_killed: false })
    }

    fn finish_gracefully(&mut self, stdout_ended: &AtomicBool, stderr_ended: &AtomicBool) {
        let deadline = Instant::now() + CLOSE_GRACE;
        while Instant::now() < deadline {
            if !self.child_exited && matches!(self.child.try_wait(), Ok(Some(_))) {
                self.child_exited = true;
            }
            if self.child_exited
                && stdout_ended.load(Ordering::Acquire)
                && stderr_ended.load(Ordering::Acquire)
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // Always terminate the process group / job even after a clean parent exit so no helper outlives the call.
        self.kill();
    }

    fn kill(&mut self) {
        if !self.tree_killed {
            self.tree.kill(&mut self.child);
            self.tree_killed = true;
            self.child_exited = true;
        }
    }
}

impl Drop for AcpProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Builds the isolated `agy acp` command in the scratch workspace with the restricted `folo-labeler` agent, strict
/// `folo-label` CLI settings, terminal sandbox, default customizations off, and telemetry off.
pub(super) fn acp_command(path: &Path, iso: &Isolation) -> Command {
    let resolved = resolve_executable(path);
    let mut cmd = Command::new(resolved);
    cmd.args(["acp", "--agent", LABEL_AGENT_NAME, "--cli_dir_name", LABEL_CLI_DIR, "--log-file"])
        .arg(&iso.log_file)
        .args(["--sandbox", "--include-default-customizations=false", "--disable-telemetry"])
        .current_dir(&iso.work);
    cmd
}

/// Spawns `agy acp`, drives one ACP v1 labeling turn, and cleans up the entire child tree and reader threads.
pub(super) fn run_acp(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
) -> Result<AgentReply, AgentError> {
    run_acp_with_home(path, req, timeout, None)
}

pub(super) fn run_acp_with_home(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
    home: Option<&Path>,
) -> Result<AgentReply, AgentError> {
    if req.cancel.load(Ordering::Relaxed) {
        return Err(AgentError::Cancelled(LabelUsage::default()));
    }
    let iso = prepare_isolation(req.scratch, home)?;

    let mut proc = AcpProcess::spawn(path, &iso)?;
    let stdin = proc.child.stdin.take().expect("stdin is piped");
    let stdout = proc.child.stdout.take().expect("stdout is piped");
    let stderr = proc.child.stderr.take().expect("stderr is piped");

    // Dedicated stdin writer thread so a wedged child never blocks deadline/cancel checks on large image batches.
    let (in_tx, in_rx) = mpsc::channel::<Vec<u8>>();
    let _in_thread = std::thread::spawn(move || {
        let mut stdin = stdin;
        while let Ok(bytes) = in_rx.recv() {
            if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                break;
            }
        }
    });

    // Dedicated stdout line reader thread.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let stdout_ended = Arc::new(AtomicBool::new(false));
    let stdout_done = stdout_ended.clone();
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if out_tx.send(line).is_err() {
                break;
            }
        }
        stdout_done.store(true, Ordering::Release);
    });

    // Dedicated bounded stderr tail reader thread.
    let stderr_tail = StderrTail::new();
    let stderr_out = stderr_tail.clone();
    let err_thread = std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut chunk = [0u8; 8 * 1024];
        while let Ok(n @ 1..) = stderr.read(&mut chunk) {
            stderr_out.push(&chunk[..n]);
        }
        stderr_out.mark_ended();
    });

    let mut send = |msg: &Value| -> bool {
        let mut bytes = serde_json::to_vec(msg).expect("JSON values serialize");
        bytes.push(b'\n');
        in_tx.send(bytes).is_ok()
    };
    let read_stderr = || stderr_tail.drain_text(STDERR_DRAIN);

    let result = drive_acp(req, &iso.work, timeout, &mut send, &out_rx, &read_stderr);

    // Close stdin and wait briefly for a clean exit; always kill the process group / job afterwards so no
    // descendant survives holding pipes open.
    drop(in_tx);
    match &result {
        Err(AgentError::Cancelled(_) | AgentError::Timeout(_)) => proc.kill(),
        Ok(_) | Err(_) => proc.finish_gracefully(&stdout_ended, &stderr_tail.ended),
    }
    // Wait briefly for the reader threads to observe EOF once the tree is dead so they do not outlive the call.
    join_bounded(out_thread, &stdout_ended, STDERR_DRAIN);
    join_bounded(err_thread, &stderr_tail.ended, STDERR_DRAIN);
    result
}

fn join_bounded(handle: JoinHandle<()>, ended: &AtomicBool, wait: Duration) {
    let deadline = Instant::now() + wait;
    while !ended.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if ended.load(Ordering::Acquire) {
        let _ = handle.join();
    }
}

/// What stopped a single ACP RPC call early.
fn unexpected_exit(usage: &UsageTracker, stderr_text: &dyn Fn() -> String) -> AgentError {
    if let Some(why) = last_line(&stderr_text()) {
        return AgentError::Failed(Failure { used: usage.to_usage(0), ..Failure::new(why) });
    }
    if usage.seen() {
        return AgentError::Failed(Failure {
            used: usage.to_usage(0),
            ..Failure::new("`agy` exited before replying")
        });
    }
    AgentError::Unavailable("`agy` exited before replying".into())
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
    drive_acp_with_grace(req, work, timeout, CANCEL_GRACE, send, recv, stderr_text)
}

pub(super) fn drive_acp_with_grace(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    cancel_grace: Duration,
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
            return Err(unexpected_exit(usage, stderr_text));
        }
        let mut cancel_deadline: Option<Instant> = None;
        loop {
            if cancel_deadline.is_none() && req.cancel.load(Ordering::Relaxed) {
                if let Some(sid) = session_id {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                    // Give a responsive child a short bounded grace period to acknowledge cancel and emit final usage.
                    if collect_reply && !cancel_grace.is_zero() {
                        cancel_deadline = Some(Instant::now() + cancel_grace);
                    } else {
                        return Err(AgentError::Cancelled(usage.to_usage(0)));
                    }
                } else {
                    return Err(AgentError::Cancelled(usage.to_usage(0)));
                }
            }
            if let Some(end) = cancel_deadline
                && Instant::now() >= end
            {
                return Err(AgentError::Cancelled(usage.to_usage(0)));
            }
            let wait_until = match cancel_deadline {
                Some(end) => end.min(deadline),
                None => deadline,
            };
            let remaining = wait_until.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if cancel_deadline.is_some() {
                    return Err(AgentError::Cancelled(usage.to_usage(0)));
                }
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
                            Some("agent_message_chunk") if collect_reply && cancel_deadline.is_none() => {
                                if update["content"]["type"] == "text"
                                    && let Some(t) = update["content"]["text"].as_str()
                                {
                                    reply_text.push_str(t);
                                }
                            }
                            Some("usage_update") => usage.observe_usage_update(update),
                            Some("tool_call" | "tool_call_update") => {
                                // Fail closed if a tool ever executes despite the restricted agent & strict settings.
                                if let Some(sid) = session_id {
                                    let _ = send(&json!({
                                        "jsonrpc": "2.0",
                                        "method": "session/cancel",
                                        "params": { "sessionId": sid }
                                    }));
                                }
                                return Err(AgentError::Unavailable(
                                    "Antigravity attempted to invoke a tool in labeling mode".into(),
                                ));
                            }
                            _ => {}
                        }
                        continue;
                    }
                    // Response to our pending call.
                    if msg.get("id").and_then(Value::as_u64) == Some(id) {
                        if let Some(res) = msg.get("result")
                            && collect_reply
                        {
                            usage.observe_prompt_result(res);
                        }
                        if cancel_deadline.is_some() {
                            return Err(AgentError::Cancelled(usage.to_usage(0)));
                        }
                        if let Some(err) = msg.get("error") {
                            return Err(AgentError::Failed(rpc_failure(err, usage.to_usage(0))));
                        }
                        return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if cancel_deadline.is_some() {
                        return Err(AgentError::Cancelled(usage.to_usage(0)));
                    }
                    return Err(unexpected_exit(usage, stderr_text));
                }
            }
        }
    };

    // 1. Handshake (`initialize`): require protocol v1 and explicit image support for image-bearing calls.
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
    if init["protocolVersion"].as_u64() != Some(1) {
        let got = &init["protocolVersion"];
        return Err(AgentError::Unavailable(format!(
            "`agy acp` negotiated unsupported ACP protocol version ({got}); expected 1"
        )));
    }
    let caps = &init["agentCapabilities"];
    if !req.images.is_empty() && caps["promptCapabilities"]["image"].as_bool() != Some(true) {
        return Err(AgentError::Unavailable(
            "`agy acp` does not advertise inline image prompt support".into(),
        ));
    }
    let system_append_supported =
        caps["_meta"]["antigravity"]["systemPrompt"]["append"].as_bool() == Some(true);
    let close_supported = caps["sessionCapabilities"].get("close").is_some_and(|v| !v.is_null());

    // 2. Open an isolated session in the scratch folder (`session/new`) with an absolute `cwd`.
    let cwd = super::protocol::abs_dir(work)
        .map_err(|e| AgentError::Unavailable(format!("resolving session cwd: {e}")))?;
    let mut new_params = json!({
        "cwd": folo_api::json::path(&cwd),
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

    // 2b. Ensure the session is not in an auto-approving execution mode (fail closed if stuck in one).
    if let Some(mode_opt) = find_mode_option(&config_options).cloned()
        && let Some(cur) = mode_opt.current_value.as_deref()
        && is_permissive_mode(cur)
    {
        let safe = pick_safe_mode(&mode_opt.values).ok_or_else(|| {
            AgentError::Unavailable(format!(
                "`agy acp` started in auto-approving mode `{cur}` and offers no restricted mode"
            ))
        })?;
        let updated = rpc_call(
            "session/set_config_option",
            json!({ "sessionId": sid, "configId": mode_opt.id, "value": safe }),
            false,
            &session_id,
            &mut usage,
            &mut reply_text,
        )?;
        let refreshed = parse_config_options(updated.get("configOptions"));
        if !refreshed.is_empty() {
            config_options = refreshed;
            if let Some(after) = find_mode_option(&config_options).and_then(|o| o.current_value.as_deref())
                && is_permissive_mode(after)
            {
                return Err(AgentError::Unavailable(format!(
                    "`agy acp` remained in auto-approving mode `{after}` after requesting `{safe}`"
                )));
            }
        }
    } else if let Some(modes) = session.get("modes")
        && let Some(cur) = modes["currentModeId"].as_str()
        && is_permissive_mode(cur)
    {
        let avail: Vec<(String, String)> = modes["availableModes"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m["id"].as_str()?;
                        Some((id.to_string(), m["name"].as_str().unwrap_or(id).to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let safe = pick_safe_mode(&avail).ok_or_else(|| {
            AgentError::Unavailable(format!(
                "`agy acp` started in auto-approving mode `{cur}` and offers no restricted mode"
            ))
        })?;
        let _ = rpc_call(
            "session/set_mode",
            json!({ "sessionId": sid, "modeId": safe }),
            false,
            &session_id,
            &mut usage,
            &mut reply_text,
        )?;
    }

    // 3. Configure model and reasoning effort (`session/set_config_option`, with legacy `session/set_model` fallback).
    // Fail closed if an explicitly requested model or non-default effort cannot be configured in this session.
    let res = resolve_model_and_effort(req.model, req.effort, find_model_option(&config_options));
    if let Some(model) = res.model {
        if let Some(opt) = find_model_option(&config_options).cloned() {
            if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&model)) {
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
                    if let Some(actual) =
                        find_model_option(&config_options).and_then(|o| o.current_value.as_deref())
                        && !actual.eq_ignore_ascii_case(&model)
                    {
                        return Err(AgentError::Failed(Failure::new(format!(
                            "`agy acp` selected model `{actual}` instead of requested `{model}`"
                        ))));
                    }
                }
            }
        } else if let Some(models) = session.get("models").filter(|m| !m.is_null()) {
            let avail = parse_legacy_models(models);
            let target = match_option_value(&avail, &model).unwrap_or(&model).to_string();
            let current = models["currentModelId"].as_str();
            if !current.is_some_and(|cur| cur.eq_ignore_ascii_case(&target)) {
                let _ = rpc_call(
                    "session/set_model",
                    json!({ "sessionId": sid, "modelId": target }),
                    false,
                    &session_id,
                    &mut usage,
                    &mut reply_text,
                )?;
            }
        } else {
            return Err(AgentError::Failed(Failure::new(format!(
                "`agy acp` does not allow selecting model `{model}` in this session"
            ))));
        }
    }
    if let Some(opt) = find_effort_option(&config_options).cloned() {
        if let Some(effort_value) = resolve_effort_value(&res.desired_effort, &opt)
            && !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&effort_value))
        {
            let updated = rpc_call(
                "session/set_config_option",
                json!({ "sessionId": sid, "configId": opt.id, "value": effort_value }),
                false,
                &session_id,
                &mut usage,
                &mut reply_text,
            )?;
            let refreshed = parse_config_options(updated.get("configOptions"));
            if !refreshed.is_empty()
                && let Some(actual) = find_effort_option(&refreshed).and_then(|o| o.current_value.as_deref())
                && !actual.eq_ignore_ascii_case(&effort_value)
            {
                return Err(AgentError::Failed(Failure::new(format!(
                    "`agy acp` selected effort `{actual}` instead of requested `{effort_value}`"
                ))));
            }
        }
    } else if res.stripped_suffix || (!res.suffix_on_model && req.effort != LabelEffort::Low) {
        return Err(AgentError::Failed(Failure::new(format!(
            "`agy acp` does not allow setting reasoning effort `{}` in this session",
            res.desired_effort
        ))));
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
        return Err(if usage.seen() {
            AgentError::Failed(Failure { used, ..Failure::new("`agy` printed no reply") })
        } else {
            AgentError::Unavailable("`agy` printed no reply".into())
        });
    }
    Ok(AgentReply { text: reply_text, usage: usage.to_usage(1) })
}
