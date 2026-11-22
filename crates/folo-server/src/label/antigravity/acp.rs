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
    ConfigOption, Isolation, LABEL_AGENT_NAME, LABEL_CLI_DIR, UsageTracker, find_effort_option,
    find_mode_option, find_model_option, is_restricted_mode, match_option_value, parse_config_options,
    parse_legacy_models, pick_safe_mode, prepare_isolation, prompt_blocks, reject_permission,
    resolve_effort_value, resolve_executable, resolve_model_and_effort, rpc_failure,
};
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, POLL, last_line};
use crate::label::tree::Tree;

/// How long to wait for `agy acp` to exit cleanly after closing stdin before killing its process tree.
const CLOSE_GRACE: Duration = Duration::from_millis(250);
/// How long to wait for a responsive `agy acp` child to acknowledge `session/cancel` and emit final usage.
pub(super) const CANCEL_GRACE: Duration = Duration::from_millis(500);
/// How long to drain buffered stdout/stderr after the parent process has exited before reporting early exit.
const EXIT_DRAIN: Duration = Duration::from_millis(250);
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
/// Unix (and Windows job members, required via [`Tree::start_strict`]) are always killed even when the immediate
/// child exits cleanly.
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
        let tree = Tree::start_strict(&child).map_err(|e| {
            let _ = child.kill();
            let _ = child.wait();
            unavailable(e)
        })?;
        Ok(AcpProcess { child, tree, child_exited: false, tree_killed: false })
    }

    fn poll_exited(&mut self) -> bool {
        if !self.child_exited && matches!(self.child.try_wait(), Ok(Some(_))) {
            self.child_exited = true;
        }
        self.child_exited
    }

    fn finish_gracefully(&mut self, stdout_ended: &AtomicBool, stderr_ended: &AtomicBool) {
        let deadline = Instant::now() + CLOSE_GRACE;
        while Instant::now() < deadline {
            if self.poll_exited()
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

/// Outcome of driving one ACP conversation, plus whether the caller must kill the child tree immediately (skipping
/// any graceful exit window, e.g. on cancellation, timeout, or a tool/mode policy violation).
pub(super) struct DriveOutcome {
    pub result: Result<AgentReply, AgentError>,
    pub kill_immediately: bool,
}

/// Spawns `agy acp`, drives one ACP v1 labeling turn, and cleans up the entire child tree and reader threads.
pub(super) fn run_acp(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
) -> Result<AgentReply, AgentError> {
    run_acp_checked(path, req, timeout, None).0
}

/// Same as [`run_acp`], returning `(result, pipes_closed)` for deterministic descendant-cleanup tests.
pub(super) fn run_acp_checked(
    path: &Path,
    req: &AgentRequest<'_>,
    timeout: Duration,
    home: Option<&Path>,
) -> (Result<AgentReply, AgentError>, bool) {
    if req.cancel.load(Ordering::Relaxed) {
        return (Err(AgentError::Cancelled(LabelUsage::default())), true);
    }
    let iso = match prepare_isolation(req.scratch, home) {
        Ok(iso) => iso,
        Err(e) => return (Err(e), true),
    };

    let mut proc = match AcpProcess::spawn(path, &iso) {
        Ok(p) => p,
        Err(e) => return (Err(e), true),
    };
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
    let read_stderr = || stderr_tail.drain_text(EXIT_DRAIN);
    let mut parent_exited = || proc.poll_exited();

    let outcome = drive_acp_inner(
        req,
        &iso.work,
        timeout,
        CANCEL_GRACE,
        &mut send,
        &out_rx,
        &read_stderr,
        &mut parent_exited,
    );

    // Close stdin; kill immediately on cancel, timeout, or policy violation, otherwise allow a brief clean exit
    // before always killing the process group / job so no descendant outlives the call.
    drop(in_tx);
    if outcome.kill_immediately {
        proc.kill();
    } else {
        proc.finish_gracefully(&stdout_ended, &stderr_tail.ended);
    }
    let out_closed = join_bounded(out_thread, &stdout_ended, EXIT_DRAIN);
    let err_closed = join_bounded(err_thread, &stderr_tail.ended, EXIT_DRAIN);
    (outcome.result, out_closed && err_closed)
}

fn join_bounded(handle: JoinHandle<()>, ended: &AtomicBool, wait: Duration) -> bool {
    let deadline = Instant::now() + wait;
    while !ended.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if ended.load(Ordering::Acquire) {
        let _ = handle.join();
        true
    } else {
        false
    }
}

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

/// Checks that any advertised mode selector (config option or legacy mode state) is on an explicitly allowed
/// restricted mode. Fails closed on missing, dropped, or unrecognized mode ids.
fn verify_restricted_mode(
    options: &[ConfigOption],
    legacy_mode: Option<&str>,
    require_mode_option: bool,
) -> Result<(), AgentError> {
    if let Some(opt) = find_mode_option(options) {
        match opt.current_value.as_deref() {
            Some(cur) if is_restricted_mode(cur) => Ok(()),
            Some(cur) => Err(AgentError::Unavailable(format!(
                "`agy acp` is in non-restricted mode `{cur}`; refusing to run in labeling mode"
            ))),
            None => Err(AgentError::Unavailable(
                "`agy acp` advertised a mode selector without a current restricted mode".into(),
            )),
        }
    } else if require_mode_option {
        Err(AgentError::Unavailable(
            "`agy acp` omitted `mode` from its updated `configOptions`; refusing to run in labeling mode"
                .into(),
        ))
    } else if let Some(cur) = legacy_mode {
        if is_restricted_mode(cur) {
            Ok(())
        } else {
            Err(AgentError::Unavailable(format!(
                "`agy acp` is in non-restricted mode `{cur}`; refusing to run in labeling mode"
            )))
        }
    } else {
        Ok(())
    }
}

/// The ACP v1 JSON-RPC conversation for one labeling call, decoupled from OS processes so unit tests can drive it
/// directly over in-memory channels.
#[cfg(test)]
pub(super) fn drive_acp(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<String>,
    stderr_text: &dyn Fn() -> String,
) -> Result<AgentReply, AgentError> {
    let mut never_exited = || false;
    drive_acp_inner(req, work, timeout, CANCEL_GRACE, send, recv, stderr_text, &mut never_exited).result
}

#[cfg(test)]
pub(super) fn drive_acp_with_grace(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    cancel_grace: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<String>,
    stderr_text: &dyn Fn() -> String,
) -> DriveOutcome {
    let mut never_exited = || false;
    drive_acp_inner(req, work, timeout, cancel_grace, send, recv, stderr_text, &mut never_exited)
}

#[allow(clippy::too_many_arguments)]
fn drive_acp_inner(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    cancel_grace: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<String>,
    stderr_text: &dyn Fn() -> String,
    parent_exited: &mut dyn FnMut() -> bool,
) -> DriveOutcome {
    let deadline = Instant::now() + timeout;
    let mut next_id: u64 = 1;
    let mut usage = UsageTracker::default();
    let mut reply_text = String::new();
    let mut session_id: Option<String> = None;
    let mut kill_immediately = false;
    let mut enforce_mode = false;
    let mut mode_selector_advertised = false;
    let mut notified_mode: Option<String> = None;

    let mut rpc_call = |method: &str,
                        params: Value,
                        collect_reply: bool,
                        session_id: &Option<String>,
                        enforce_mode: bool,
                        mode_selector_advertised: bool,
                        notified_mode: &mut Option<String>,
                        usage: &mut UsageTracker,
                        reply_text: &mut String,
                        kill_immediately: &mut bool|
     -> Result<Value, AgentError> {
        let id = next_id;
        next_id += 1;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if !send(&msg) {
            return Err(unexpected_exit(usage, stderr_text));
        }
        let mut cancel_deadline: Option<Instant> = None;
        let mut exit_drain_deadline: Option<Instant> = None;
        loop {
            if cancel_deadline.is_none() && req.cancel.load(Ordering::Relaxed) {
                *kill_immediately = true;
                if let Some(sid) = session_id {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                    if collect_reply && !cancel_grace.is_zero() {
                        cancel_deadline = Some(Instant::now() + cancel_grace);
                    } else {
                        return Err(AgentError::Cancelled(usage.to_usage(0)));
                    }
                } else {
                    return Err(AgentError::Cancelled(usage.to_usage(0)));
                }
            }
            let now = Instant::now();
            if let Some(end) = cancel_deadline
                && now >= end
            {
                return Err(AgentError::Cancelled(usage.to_usage(0)));
            }
            if let Some(end) = exit_drain_deadline
                && now >= end
            {
                *kill_immediately = true;
                return Err(unexpected_exit(usage, stderr_text));
            }
            let mut wait_until = deadline;
            if let Some(end) = cancel_deadline {
                wait_until = wait_until.min(end);
            }
            if let Some(end) = exit_drain_deadline {
                wait_until = wait_until.min(end);
            }
            let remaining = wait_until.saturating_duration_since(now);
            if remaining.is_zero() {
                if cancel_deadline.is_some() {
                    return Err(AgentError::Cancelled(usage.to_usage(0)));
                }
                if exit_drain_deadline.is_some() {
                    *kill_immediately = true;
                    return Err(unexpected_exit(usage, stderr_text));
                }
                *kill_immediately = true;
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
                    // Server -> client request: deny immediately so the agent never deadlocks waiting for tools.
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
                                *kill_immediately = true;
                                if let Some(sid) = session_id {
                                    let _ = send(&json!({
                                        "jsonrpc": "2.0",
                                        "method": "session/cancel",
                                        "params": { "sessionId": sid }
                                    }));
                                }
                                return Err(AgentError::Failed(Failure {
                                    used: usage.to_usage(0),
                                    ..Failure::new("Antigravity attempted to invoke a tool in labeling mode")
                                }));
                            }
                            Some("current_mode_update") => {
                                let mode = update["currentModeId"].as_str().unwrap_or("");
                                *notified_mode = Some(mode.to_string());
                                if enforce_mode && !is_restricted_mode(mode) {
                                    *kill_immediately = true;
                                    if let Some(sid) = session_id {
                                        let _ = send(&json!({
                                            "jsonrpc": "2.0",
                                            "method": "session/cancel",
                                            "params": { "sessionId": sid }
                                        }));
                                    }
                                    return Err(AgentError::Failed(Failure {
                                        used: usage.to_usage(0),
                                        ..Failure::new(format!(
                                            "Antigravity switched to non-restricted mode `{mode}` in labeling mode"
                                        ))
                                    }));
                                }
                            }
                            Some("config_option_update") => {
                                let updated_opts = parse_config_options(update.get("configOptions"));
                                if enforce_mode
                                    && let Err(AgentError::Unavailable(why)) = verify_restricted_mode(
                                        &updated_opts,
                                        None,
                                        mode_selector_advertised && !updated_opts.is_empty(),
                                    )
                                {
                                    *kill_immediately = true;
                                    if let Some(sid) = session_id {
                                        let _ = send(&json!({
                                            "jsonrpc": "2.0",
                                            "method": "session/cancel",
                                            "params": { "sessionId": sid }
                                        }));
                                    }
                                    return Err(AgentError::Failed(Failure {
                                        used: usage.to_usage(0),
                                        ..Failure::new(why)
                                    }));
                                }
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
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // If the immediate `agy` child already exited while a descendant holds stdout open, drain
                    // briefly for buffered output instead of stalling until the full 5-minute call deadline.
                    if exit_drain_deadline.is_none() && parent_exited() {
                        exit_drain_deadline = Some(Instant::now() + EXIT_DRAIN);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if cancel_deadline.is_some() {
                        return Err(AgentError::Cancelled(usage.to_usage(0)));
                    }
                    return Err(unexpected_exit(usage, stderr_text));
                }
            }
        }
    };

    macro_rules! try_drive {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return DriveOutcome { result: Err(e), kill_immediately },
            }
        };
    }

    // 1. Handshake (`initialize`): require protocol v1 and explicit image support for image-bearing calls.
    let init = try_drive!(rpc_call(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": { "name": "folo", "version": env!("CARGO_PKG_VERSION") },
        }),
        false,
        &session_id,
        enforce_mode,
        mode_selector_advertised,
        &mut notified_mode,
        &mut usage,
        &mut reply_text,
        &mut kill_immediately,
    ));
    if init["protocolVersion"].as_u64() != Some(1) {
        let got = &init["protocolVersion"];
        return DriveOutcome {
            result: Err(AgentError::Unavailable(format!(
                "`agy acp` negotiated unsupported ACP protocol version ({got}); expected 1"
            ))),
            kill_immediately,
        };
    }
    let caps = &init["agentCapabilities"];
    if !req.images.is_empty() && caps["promptCapabilities"]["image"].as_bool() != Some(true) {
        return DriveOutcome {
            result: Err(AgentError::Unavailable(
                "`agy acp` does not advertise inline image prompt support".into(),
            )),
            kill_immediately,
        };
    }
    let system_append_supported =
        caps["_meta"]["antigravity"]["systemPrompt"]["append"].as_bool() == Some(true);
    let close_supported = caps["sessionCapabilities"].get("close").is_some_and(|v| !v.is_null());

    // 2. Open an isolated session in the scratch folder (`session/new`) with an absolute `cwd`.
    let cwd = try_drive!(
        super::protocol::abs_dir(work)
            .map_err(|e| AgentError::Unavailable(format!("resolving session cwd: {e}")))
    );
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
    let session = try_drive!(rpc_call(
        "session/new",
        new_params,
        false,
        &session_id,
        enforce_mode,
        mode_selector_advertised,
        &mut notified_mode,
        &mut usage,
        &mut reply_text,
        &mut kill_immediately,
    ));
    let sid = try_drive!(
        session["sessionId"]
            .as_str()
            .ok_or_else(|| AgentError::Unavailable("`agy acp` returned no sessionId".into()))
    )
    .to_string();
    session_id = Some(sid.clone());
    let mut config_options = parse_config_options(session.get("configOptions"));
    mode_selector_advertised = find_mode_option(&config_options).is_some();
    let mut legacy_mode = session
        .get("modes")
        .and_then(|m| m["currentModeId"].as_str())
        .map(str::to_string)
        .or_else(|| notified_mode.clone());

    // 2b. Enforce a known restricted execution mode whenever a mode selector is advertised.
    if let Some(mode_opt) = find_mode_option(&config_options).cloned() {
        if !mode_opt.current_value.as_deref().is_some_and(is_restricted_mode) {
            let cur = mode_opt.current_value.as_deref().unwrap_or("<unset>");
            let safe = try_drive!(pick_safe_mode(&mode_opt.values).ok_or_else(|| {
                AgentError::Unavailable(format!(
                    "`agy acp` started in non-restricted mode `{cur}` and offers no restricted mode"
                ))
            }));
            let updated = try_drive!(rpc_call(
                "session/set_config_option",
                json!({ "sessionId": sid, "configId": mode_opt.id, "value": safe }),
                false,
                &session_id,
                enforce_mode,
                mode_selector_advertised,
                &mut notified_mode,
                &mut usage,
                &mut reply_text,
                &mut kill_immediately,
            ));
            let refreshed = parse_config_options(updated.get("configOptions"));
            if refreshed.is_empty() {
                return DriveOutcome {
                    result: Err(AgentError::Unavailable(
                        "`agy acp` did not return updated `configOptions` after setting mode".into(),
                    )),
                    kill_immediately,
                };
            }
            config_options = refreshed;
        }
    } else if let Some(modes) = session.get("modes").filter(|m| !m.is_null()) {
        let cur = legacy_mode.as_deref().unwrap_or("<unset>");
        if !is_restricted_mode(cur) {
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
            let safe = try_drive!(pick_safe_mode(&avail).ok_or_else(|| {
                AgentError::Unavailable(format!(
                    "`agy acp` started in non-restricted mode `{cur}` and offers no restricted mode"
                ))
            }));
            let _ = try_drive!(rpc_call(
                "session/set_mode",
                json!({ "sessionId": sid, "modeId": safe }),
                false,
                &session_id,
                enforce_mode,
                mode_selector_advertised,
                &mut notified_mode,
                &mut usage,
                &mut reply_text,
                &mut kill_immediately,
            ));
            legacy_mode = Some(safe);
        }
    }
    try_drive!(verify_restricted_mode(&config_options, legacy_mode.as_deref(), mode_selector_advertised));
    enforce_mode = true;

    // 3. Configure model and reasoning effort (`session/set_config_option`, requiring the complete updated
    // `configOptions` snapshot per ACP v1, or legacy `session/set_model`), then verify all settings before prompting.
    let legacy_model_opt = if find_model_option(&config_options).is_none() {
        session.get("models").filter(|m| !m.is_null()).map(|models| ConfigOption {
            id: "model".into(),
            category: Some("model".into()),
            current_value: models["currentModelId"].as_str().map(str::to_string),
            values: parse_legacy_models(models),
        })
    } else {
        None
    };
    let res = resolve_model_and_effort(
        req.model,
        req.effort,
        find_model_option(&config_options).or(legacy_model_opt.as_ref()),
    );
    let mut expected_model: Option<String> = None;
    let mut expected_effort: Option<String> = None;

    if let Some(model) = res.model {
        if let Some(opt) = find_model_option(&config_options).cloned() {
            expected_model = Some(model.clone());
            if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&model)) {
                let updated = try_drive!(rpc_call(
                    "session/set_config_option",
                    json!({ "sessionId": sid, "configId": opt.id, "value": model }),
                    false,
                    &session_id,
                    enforce_mode,
                    mode_selector_advertised,
                    &mut notified_mode,
                    &mut usage,
                    &mut reply_text,
                    &mut kill_immediately,
                ));
                let refreshed = parse_config_options(updated.get("configOptions"));
                if refreshed.is_empty() {
                    return DriveOutcome {
                        result: Err(AgentError::Failed(Failure::new(format!(
                            "`agy acp` did not return updated `configOptions` confirming model `{model}`"
                        )))),
                        kill_immediately,
                    };
                }
                config_options = refreshed;
            }
        } else if let Some(legacy_opt) = legacy_model_opt.as_ref() {
            let target = match match_option_value(&legacy_opt.values, &model) {
                Some(matched) => matched.to_string(),
                None if legacy_opt.values.is_empty() => model.clone(),
                None => {
                    return DriveOutcome {
                        result: Err(AgentError::Failed(Failure::new(format!(
                            "`agy acp` does not offer model `{model}` in this session"
                        )))),
                        kill_immediately,
                    };
                }
            };
            if !legacy_opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&target)) {
                let _ = try_drive!(rpc_call(
                    "session/set_model",
                    json!({ "sessionId": sid, "modelId": target }),
                    false,
                    &session_id,
                    enforce_mode,
                    mode_selector_advertised,
                    &mut notified_mode,
                    &mut usage,
                    &mut reply_text,
                    &mut kill_immediately,
                ));
            }
        } else {
            return DriveOutcome {
                result: Err(AgentError::Failed(Failure::new(format!(
                    "`agy acp` does not allow selecting model `{model}` in this session"
                )))),
                kill_immediately,
            };
        }
    }

    if let Some(opt) = find_effort_option(&config_options).cloned() {
        let effort_value = try_drive!(
            resolve_effort_value(&res.desired_effort, &opt)
                .map_err(|why| AgentError::Failed(Failure::new(why)))
        );
        expected_effort = Some(effort_value.clone());
        if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&effort_value)) {
            let updated = try_drive!(rpc_call(
                "session/set_config_option",
                json!({ "sessionId": sid, "configId": opt.id, "value": effort_value }),
                false,
                &session_id,
                enforce_mode,
                mode_selector_advertised,
                &mut notified_mode,
                &mut usage,
                &mut reply_text,
                &mut kill_immediately,
            ));
            let refreshed = parse_config_options(updated.get("configOptions"));
            if refreshed.is_empty() {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` did not return updated `configOptions` confirming effort `{effort_value}`"
                    )))),
                    kill_immediately,
                };
            }
            config_options = refreshed;
        }
    } else if res.stripped_suffix || (!res.suffix_on_model && req.effort != LabelEffort::Low) {
        return DriveOutcome {
            result: Err(AgentError::Failed(Failure::new(format!(
                "`agy acp` does not allow setting reasoning effort `{}` in this session",
                res.desired_effort
            )))),
            kill_immediately,
        };
    }

    // Final pre-prompt invariant check against the latest configuration snapshot: mode must still be restricted,
    // and both requested model and effort must still match.
    try_drive!(verify_restricted_mode(&config_options, legacy_mode.as_deref(), mode_selector_advertised));
    if let Some(want) = expected_model.as_deref() {
        match find_model_option(&config_options).and_then(|o| o.current_value.as_deref()) {
            Some(actual) if actual.eq_ignore_ascii_case(want) => {}
            Some(actual) => {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` selected model `{actual}` instead of requested `{want}`"
                    )))),
                    kill_immediately,
                };
            }
            None => {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` omitted model `{want}` from its latest `configOptions`"
                    )))),
                    kill_immediately,
                };
            }
        }
    }
    if let Some(want) = expected_effort.as_deref() {
        match find_effort_option(&config_options).and_then(|o| o.current_value.as_deref()) {
            Some(actual) if actual.eq_ignore_ascii_case(want) => {}
            Some(actual) => {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` selected effort `{actual}` instead of requested `{want}`"
                    )))),
                    kill_immediately,
                };
            }
            None => {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` omitted effort `{want}` from its latest `configOptions`"
                    )))),
                    kill_immediately,
                };
            }
        }
    }

    // 4. Send the single-turn prompt with inline base64 JPEG images (`session/prompt`).
    let prompt_result = try_drive!(rpc_call(
        "session/prompt",
        json!({ "sessionId": sid, "prompt": prompt_blocks(req, system_append_supported) }),
        true,
        &session_id,
        enforce_mode,
        mode_selector_advertised,
        &mut notified_mode,
        &mut usage,
        &mut reply_text,
        &mut kill_immediately,
    ));

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
        return DriveOutcome {
            result: Err(AgentError::Cancelled(usage.to_usage(0))),
            kill_immediately: true,
        };
    }
    if stop_reason == "refusal" {
        return DriveOutcome {
            result: Err(AgentError::Failed(Failure {
                used: usage.to_usage(1),
                ..Failure::new("Antigravity refused the prompt")
            })),
            kill_immediately,
        };
    }
    if reply_text.trim().is_empty() {
        let used = usage.to_usage(0);
        return DriveOutcome {
            result: Err(if usage.seen() {
                AgentError::Failed(Failure { used, ..Failure::new("`agy` printed no reply") })
            } else {
                AgentError::Unavailable("`agy` printed no reply".into())
            }),
            kill_immediately,
        };
    }
    DriveOutcome { result: Ok(AgentReply { text: reply_text, usage: usage.to_usage(1) }), kill_immediately }
}
