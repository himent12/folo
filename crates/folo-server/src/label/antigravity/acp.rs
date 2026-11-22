//! Single-turn ACP v1 JSON-RPC session state machine and mode/model/effort invariant enforcement.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::process::unexpected_exit;
use super::protocol::{
    ConfigOption, UsageTracker, find_effort_option, find_mode_option, find_model_option, is_restricted_mode,
    match_option_value, parse_config_options, parse_legacy_models, pick_safe_mode, prompt_blocks,
    reject_permission, resolve_effort_value, resolve_model_and_effort, rpc_failure, suffix_matches_effort,
};
use super::split_effort_suffix;
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure, POLL};

/// How long to wait for a responsive `agy acp` child to acknowledge `session/cancel` and emit final usage.
pub(super) const CANCEL_GRACE: Duration = Duration::from_millis(500);
/// How long to drain buffered stdout/stderr after the parent process has exited before reporting early exit.
pub(super) const EXIT_DRAIN: Duration = Duration::from_millis(250);
/// Maximum queued lines drained before dispatching an RPC; bounds work if a noisy child/helper floods stdout.
const MAX_DRAIN_LINES: usize = 256;
/// Maximum wall-clock budget for draining queued stdout lines before dispatching an RPC.
const DRAIN_BUDGET: Duration = Duration::from_millis(250);

/// Outcome of driving one ACP conversation, plus whether the caller must kill the child tree immediately (skipping
/// any graceful exit window, e.g. on cancellation, timeout, or a tool/mode/config policy violation).
pub(super) struct DriveOutcome {
    pub result: Result<AgentReply, AgentError>,
    pub kill_immediately: bool,
}

/// Live ACP session state tracked across responses and `session/update` notifications in receive order.
#[derive(Default)]
struct SessionState {
    session_id: Option<String>,
    config_options: Vec<ConfigOption>,
    /// True once `configOptions` has advertised a `mode` selector.
    mode_option_seen: bool,
    /// True once `configOptions` has advertised a `model` selector.
    model_option_seen: bool,
    /// True once `configOptions` has advertised a `thought_level`/effort selector.
    effort_option_seen: bool,
    /// Latest legacy mode id from `session/new.modes` or `current_mode_update` notifications.
    legacy_mode: Option<String>,
    /// Set by `current_mode_update` during an RPC call so `session/set_mode` never overwrites a live update.
    mode_notified_in_call: bool,
    /// Set by `config_option_update` during an RPC call so legacy `session/set_model` knows whether effort was refreshed.
    config_updated_in_call: bool,
    /// Set once step 2b verifies restricted mode; any later mode drift fails closed immediately.
    enforce_mode: bool,
    /// Set once step 3 verifies model and effort; any later `config_option_update` drift fails closed immediately.
    enforce_config: bool,
    expected_model: Option<String>,
    expected_effort: Option<String>,
    /// True when no `thought_level` selector exists and the verified model ID itself encodes `expected_effort`.
    effort_encoded_on_model: bool,
    usage: UsageTracker,
    reply_text: String,
    kill_immediately: bool,
}

impl SessionState {
    fn apply_config_snapshot(&mut self, opts: Vec<ConfigOption>) {
        if find_mode_option(&opts).is_some() {
            self.mode_option_seen = true;
        }
        if find_model_option(&opts).is_some() {
            self.model_option_seen = true;
        }
        if find_effort_option(&opts).is_some() {
            self.effort_option_seen = true;
        }
        self.config_options = opts;
    }

    fn verify_mode(&self) -> Result<(), AgentError> {
        if let Some(opt) = find_mode_option(&self.config_options) {
            match opt.current_value.as_deref() {
                Some(cur) if is_restricted_mode(cur) => {}
                Some(cur) => {
                    return Err(AgentError::Unavailable(format!(
                        "`agy acp` is in non-restricted mode `{cur}`; refusing to run in labeling mode"
                    )));
                }
                None => {
                    return Err(AgentError::Unavailable(
                        "`agy acp` advertised a mode selector without a current restricted mode".into(),
                    ));
                }
            }
        } else if self.mode_option_seen {
            return Err(AgentError::Unavailable(
                "`agy acp` omitted `mode` from its updated `configOptions`; refusing to run in labeling mode"
                    .into(),
            ));
        }
        if let Some(cur) = self.legacy_mode.as_deref()
            && !is_restricted_mode(cur)
        {
            return Err(AgentError::Unavailable(format!(
                "`agy acp` is in non-restricted mode `{cur}`; refusing to run in labeling mode"
            )));
        }
        Ok(())
    }

    fn verify_expected_config(&self) -> Result<(), String> {
        if let Some(want) = self.expected_model.as_deref() {
            if let Some(opt) = find_model_option(&self.config_options) {
                match opt.current_value.as_deref() {
                    Some(actual) if actual.eq_ignore_ascii_case(want) => {}
                    Some(actual) => {
                        return Err(format!(
                            "`agy acp` selected model `{actual}` instead of requested `{want}`"
                        ));
                    }
                    None => {
                        return Err(format!("`agy acp` omitted current model value for requested `{want}`"));
                    }
                }
            } else if self.model_option_seen {
                return Err(format!("`agy acp` omitted model `{want}` from its latest `configOptions`"));
            }
        }
        if let Some(want) = self.expected_effort.as_deref() {
            if let Some(opt) = find_effort_option(&self.config_options) {
                match opt.current_value.as_deref() {
                    Some(actual) if actual.eq_ignore_ascii_case(want) => {}
                    Some(actual) => {
                        return Err(format!(
                            "`agy acp` selected effort `{actual}` instead of requested `{want}`"
                        ));
                    }
                    None => {
                        return Err(format!("`agy acp` omitted current effort value for requested `{want}`"));
                    }
                }
            } else if self.effort_option_seen {
                return Err(format!("`agy acp` omitted effort `{want}` from its latest `configOptions`"));
            } else if !self.effort_encoded_on_model {
                return Err(format!(
                    "`agy acp` does not report or allow setting reasoning effort `{want}` in this session"
                ));
            }
        }
        Ok(())
    }

    fn abort_policy(
        &mut self,
        send: &mut dyn FnMut(&Value) -> bool,
        message: impl Into<String>,
    ) -> AgentError {
        self.kill_immediately = true;
        if let Some(sid) = self.session_id.as_deref() {
            let _ = send(&json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": { "sessionId": sid }
            }));
        }
        AgentError::Failed(Failure { used: self.usage.to_usage(0), ..Failure::new(message) })
    }
}

/// Processes one incoming ACP line (request, notification, or pending response) in receive order.
fn handle_incoming_line(
    line: &str,
    expected_id: Option<u64>,
    collect_reply: bool,
    cancelling: bool,
    state: &mut SessionState,
    send: &mut dyn FnMut(&Value) -> bool,
) -> Result<Option<Value>, AgentError> {
    let Ok(msg) = serde_json::from_str::<Value>(line) else { return Ok(None) };
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
        return Ok(None);
    }
    // Notification (`session/update`).
    if msg["method"].as_str() == Some("session/update") {
        let update = &msg["params"]["update"];
        match update["sessionUpdate"].as_str() {
            Some("agent_message_chunk") if collect_reply && !cancelling => {
                if update["content"]["type"] == "text"
                    && let Some(t) = update["content"]["text"].as_str()
                {
                    state.reply_text.push_str(t);
                }
            }
            Some("usage_update") => state.usage.observe_usage_update(update),
            Some("tool_call" | "tool_call_update") => {
                return Err(
                    state.abort_policy(send, "Antigravity attempted to invoke a tool in labeling mode")
                );
            }
            Some("current_mode_update") => {
                let mode = update["currentModeId"].as_str().unwrap_or("");
                state.legacy_mode = Some(mode.to_string());
                state.mode_notified_in_call = true;
                if state.enforce_mode && !is_restricted_mode(mode) {
                    return Err(state.abort_policy(
                        send,
                        format!("Antigravity switched to non-restricted mode `{mode}` in labeling mode"),
                    ));
                }
            }
            Some("config_option_update") => {
                state.config_updated_in_call = true;
                state.apply_config_snapshot(parse_config_options(update.get("configOptions")));
                if state.enforce_mode
                    && let Err(AgentError::Unavailable(why)) = state.verify_mode()
                {
                    return Err(state.abort_policy(send, why));
                }
                if state.enforce_config
                    && let Err(why) = state.verify_expected_config()
                {
                    return Err(state.abort_policy(send, why));
                }
            }
            _ => {}
        }
        return Ok(None);
    }
    // Response to our pending call.
    if expected_id.is_some() && msg.get("id").and_then(Value::as_u64) == expected_id {
        return Ok(Some(msg));
    }
    Ok(None)
}

/// Drains already-queued notifications before dispatching the next RPC (e.g. `session/prompt`), while checking
/// cancellation, call deadline, parent exit, and a strict line/time budget so sustained stdout can never starve Stop.
#[allow(clippy::too_many_arguments)]
fn drain_pending(
    recv: &mpsc::Receiver<String>,
    state: &mut SessionState,
    send: &mut dyn FnMut(&Value) -> bool,
    stderr_text: &dyn Fn() -> String,
    cancel: &AtomicBool,
    deadline: Instant,
    parent_exited: &mut dyn FnMut() -> bool,
) -> Result<(), AgentError> {
    let drain_deadline = (Instant::now() + DRAIN_BUDGET).min(deadline);
    let mut drained = 0usize;
    loop {
        if cancel.load(Ordering::Relaxed) {
            state.kill_immediately = true;
            return Err(AgentError::Cancelled(state.usage.to_usage(0)));
        }
        let now = Instant::now();
        if now >= deadline {
            state.kill_immediately = true;
            return Err(AgentError::Timeout(state.usage.to_usage(0)));
        }
        if parent_exited() {
            state.kill_immediately = true;
            return Err(unexpected_exit(&state.usage, stderr_text));
        }
        match recv.try_recv() {
            Ok(line) => {
                if drained >= MAX_DRAIN_LINES || now >= drain_deadline {
                    state.kill_immediately = true;
                    return Err(AgentError::Failed(Failure {
                        used: state.usage.to_usage(0),
                        ..Failure::new("`agy acp` flooded stdout before the next request")
                    }));
                }
                drained += 1;
                let _ = handle_incoming_line(&line, None, false, false, state, send)?;
            }
            Err(mpsc::TryRecvError::Empty) => return Ok(()),
            Err(mpsc::TryRecvError::Disconnected) => return Err(unexpected_exit(&state.usage, stderr_text)),
        }
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn drive_acp_with_exit(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<String>,
    stderr_text: &dyn Fn() -> String,
    parent_exited: &mut dyn FnMut() -> bool,
) -> DriveOutcome {
    drive_acp_inner(req, work, timeout, CANCEL_GRACE, send, recv, stderr_text, parent_exited)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn drive_acp_inner(
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
    let mut state = SessionState::default();

    let mut rpc_call = |method: &str,
                        params: Value,
                        collect_reply: bool,
                        state: &mut SessionState|
     -> Result<Value, AgentError> {
        let id = next_id;
        next_id += 1;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        // Drain any already-queued notifications, verify active invariants, and check cancellation/deadline right
        // before dispatching so a late mode/config drift or stop never sends a billable prompt.
        drain_pending(recv, state, send, stderr_text, req.cancel, deadline, parent_exited)?;
        if state.enforce_mode {
            state.verify_mode()?;
        }
        if state.enforce_config {
            state.verify_expected_config().map_err(|why| AgentError::Failed(Failure::new(why)))?;
        }
        if req.cancel.load(Ordering::Relaxed) {
            state.kill_immediately = true;
            return Err(AgentError::Cancelled(state.usage.to_usage(0)));
        }
        if Instant::now() >= deadline {
            state.kill_immediately = true;
            return Err(AgentError::Timeout(state.usage.to_usage(0)));
        }
        state.mode_notified_in_call = false;
        state.config_updated_in_call = false;
        if !send(&msg) {
            return Err(unexpected_exit(&state.usage, stderr_text));
        }
        let mut cancel_deadline: Option<Instant> = None;
        let mut exit_drain_deadline: Option<Instant> = None;
        let mut next_exit_poll = Instant::now();
        loop {
            if cancel_deadline.is_none() && req.cancel.load(Ordering::Relaxed) {
                state.kill_immediately = true;
                if let Some(sid) = state.session_id.as_deref() {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                    if collect_reply && !cancel_grace.is_zero() {
                        cancel_deadline = Some(Instant::now() + cancel_grace);
                    } else {
                        return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                    }
                } else {
                    return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                }
            }
            let now = Instant::now();
            // Poll immediate parent exit on an independent timer so a chatty helper holding stdout open
            // cannot starve exit detection until the full 5-minute call deadline.
            if exit_drain_deadline.is_none() && now >= next_exit_poll {
                if parent_exited() {
                    exit_drain_deadline = Some(now + EXIT_DRAIN);
                } else {
                    next_exit_poll = now + POLL;
                }
            }
            if let Some(end) = cancel_deadline
                && now >= end
            {
                return Err(AgentError::Cancelled(state.usage.to_usage(0)));
            }
            if let Some(end) = exit_drain_deadline
                && now >= end
            {
                state.kill_immediately = true;
                return Err(unexpected_exit(&state.usage, stderr_text));
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
                    return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                }
                if exit_drain_deadline.is_some() {
                    state.kill_immediately = true;
                    return Err(unexpected_exit(&state.usage, stderr_text));
                }
                state.kill_immediately = true;
                if let Some(sid) = state.session_id.as_deref() {
                    let _ = send(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                }
                return Err(AgentError::Timeout(state.usage.to_usage(0)));
            }
            let poll_wait = if exit_drain_deadline.is_none() {
                remaining.min(POLL).min(next_exit_poll.saturating_duration_since(now))
            } else {
                remaining.min(POLL)
            };
            match recv.recv_timeout(poll_wait) {
                Ok(line) => {
                    if let Some(resp) = handle_incoming_line(
                        &line,
                        Some(id),
                        collect_reply,
                        cancel_deadline.is_some(),
                        state,
                        send,
                    )? {
                        if let Some(res) = resp.get("result")
                            && collect_reply
                        {
                            state.usage.observe_prompt_result(res);
                        }
                        if cancel_deadline.is_some() {
                            return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                        }
                        if let Some(err) = resp.get("error") {
                            return Err(AgentError::Failed(rpc_failure(err, state.usage.to_usage(0))));
                        }
                        return Ok(resp.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if cancel_deadline.is_some() {
                        return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                    }
                    return Err(unexpected_exit(&state.usage, stderr_text));
                }
            }
        }
    };

    macro_rules! try_drive {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return DriveOutcome { result: Err(e), kill_immediately: state.kill_immediately },
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
        &mut state,
    ));
    if init["protocolVersion"].as_u64() != Some(1) {
        let got = &init["protocolVersion"];
        return DriveOutcome {
            result: Err(AgentError::Unavailable(format!(
                "`agy acp` negotiated unsupported ACP protocol version ({got}); expected 1"
            ))),
            kill_immediately: state.kill_immediately,
        };
    }
    let caps = &init["agentCapabilities"];
    if !req.images.is_empty() && caps["promptCapabilities"]["image"].as_bool() != Some(true) {
        return DriveOutcome {
            result: Err(AgentError::Unavailable(
                "`agy acp` does not advertise inline image prompt support".into(),
            )),
            kill_immediately: state.kill_immediately,
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
    let session = try_drive!(rpc_call("session/new", new_params, false, &mut state));
    let sid = try_drive!(
        session["sessionId"]
            .as_str()
            .ok_or_else(|| AgentError::Unavailable("`agy acp` returned no sessionId".into()))
    )
    .to_string();
    state.session_id = Some(sid.clone());
    state.apply_config_snapshot(parse_config_options(session.get("configOptions")));
    if !state.mode_notified_in_call && !state.mode_option_seen {
        state.legacy_mode =
            session.get("modes").and_then(|m| m["currentModeId"].as_str()).map(str::to_string);
    }

    // 2b. Enforce an allowlisted restricted execution mode whenever a mode selector is advertised.
    if let Some(mode_opt) = find_mode_option(&state.config_options).cloned() {
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
                &mut state,
            ));
            let refreshed = parse_config_options(updated.get("configOptions"));
            if refreshed.is_empty() {
                return DriveOutcome {
                    result: Err(AgentError::Unavailable(
                        "`agy acp` did not return updated `configOptions` after setting mode".into(),
                    )),
                    kill_immediately: state.kill_immediately,
                };
            }
            state.apply_config_snapshot(refreshed);
        }
    } else if let Some(modes) = session.get("modes").filter(|m| !m.is_null()) {
        let cur = state.legacy_mode.as_deref().unwrap_or("<unset>");
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
                &mut state,
            ));
            // Only assume `safe` took effect if no live `current_mode_update` arrived during the call.
            if !state.mode_notified_in_call {
                state.legacy_mode = Some(safe);
            }
        }
    }
    try_drive!(state.verify_mode());
    state.enforce_mode = true;

    // 3. Configure model and reasoning effort (`session/set_config_option`, requiring the complete updated
    // `configOptions` snapshot per ACP v1, or legacy `session/set_model`), then verify all settings before prompting.
    let legacy_model_opt = if find_model_option(&state.config_options).is_none() {
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
        find_model_option(&state.config_options).or(legacy_model_opt.as_ref()),
    );
    let mut verified_legacy_model: Option<String> =
        legacy_model_opt.as_ref().and_then(|o| o.current_value.clone());

    if let Some(model) = res.model {
        if let Some(opt) = find_model_option(&state.config_options).cloned() {
            state.expected_model = Some(model.clone());
            if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&model)) {
                let updated = try_drive!(rpc_call(
                    "session/set_config_option",
                    json!({ "sessionId": sid, "configId": opt.id, "value": model }),
                    false,
                    &mut state,
                ));
                let refreshed = parse_config_options(updated.get("configOptions"));
                if refreshed.is_empty() {
                    return DriveOutcome {
                        result: Err(AgentError::Failed(Failure::new(format!(
                            "`agy acp` did not return updated `configOptions` confirming model `{model}`"
                        )))),
                        kill_immediately: state.kill_immediately,
                    };
                }
                state.apply_config_snapshot(refreshed);
                try_drive!(state.verify_mode());
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
                        kill_immediately: state.kill_immediately,
                    };
                }
            };
            state.expected_model = Some(target.clone());
            if !legacy_opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&target)) {
                let _ = try_drive!(rpc_call(
                    "session/set_model",
                    json!({ "sessionId": sid, "modelId": target }),
                    false,
                    &mut state,
                ));
                // Legacy `session/set_model` returns `{}` without refreshed `configOptions`: invalidate any prior
                // effort reading from the old model unless a live `config_option_update` already refreshed it.
                if !state.config_updated_in_call {
                    for opt in &mut state.config_options {
                        let is_effort = opt.category.as_deref() == Some("thought_level")
                            || matches!(
                                opt.id.to_ascii_lowercase().as_str(),
                                "thought_level" | "thinking_level" | "effort"
                            );
                        if is_effort {
                            opt.current_value = None;
                        }
                    }
                }
            }
            verified_legacy_model = Some(target);
        } else {
            return DriveOutcome {
                result: Err(AgentError::Failed(Failure::new(format!(
                    "`agy acp` does not allow selecting model `{model}` in this session"
                )))),
                kill_immediately: state.kill_immediately,
            };
        }
    } else if let Some(cur) = find_model_option(&state.config_options)
        .and_then(|o| o.current_value.as_deref())
        .or(verified_legacy_model.as_deref())
    {
        // Pin the session's reported default model so a later setter or mid-prompt update cannot drift away.
        state.expected_model = Some(cur.to_string());
    }

    if let Some(opt) = find_effort_option(&state.config_options).cloned() {
        let effort_value = try_drive!(
            resolve_effort_value(&res.desired_effort, &opt)
                .map_err(|why| AgentError::Failed(Failure::new(why)))
        );
        state.expected_effort = Some(effort_value.clone());
        if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&effort_value)) {
            let updated = try_drive!(rpc_call(
                "session/set_config_option",
                json!({ "sessionId": sid, "configId": opt.id, "value": effort_value }),
                false,
                &mut state,
            ));
            let refreshed = parse_config_options(updated.get("configOptions"));
            if refreshed.is_empty() {
                return DriveOutcome {
                    result: Err(AgentError::Failed(Failure::new(format!(
                        "`agy acp` did not return updated `configOptions` confirming effort `{effort_value}`"
                    )))),
                    kill_immediately: state.kill_immediately,
                };
            }
            state.apply_config_snapshot(refreshed);
        }
    } else if state.effort_option_seen {
        return DriveOutcome {
            result: Err(AgentError::Failed(Failure::new(format!(
                "`agy acp` omitted reasoning effort `{}` from its updated `configOptions`",
                res.desired_effort
            )))),
            kill_immediately: state.kill_immediately,
        };
    } else {
        // Without a `thought_level` selector, require positive evidence from the verified model ID itself (e.g.
        // `gemini-4-neon-low`) so a reasoning model cannot silently prompt at its provider default (`high`).
        let current_model = find_model_option(&state.config_options)
            .and_then(|o| o.current_value.as_deref())
            .or(verified_legacy_model.as_deref());
        let encoded = current_model.and_then(|m| {
            let (_, suffix) = split_effort_suffix(m)?;
            suffix_matches_effort(suffix, &res.desired_effort).then(|| m.to_string())
        });
        let Some(encoded_model) = encoded else {
            return DriveOutcome {
                result: Err(AgentError::Failed(Failure::new(format!(
                    "`agy acp` does not report or allow setting reasoning effort `{}` in this session",
                    res.desired_effort
                )))),
                kill_immediately: state.kill_immediately,
            };
        };
        if state.expected_model.is_none() {
            state.expected_model = Some(encoded_model);
        }
        state.expected_effort = Some(res.desired_effort.clone());
        state.effort_encoded_on_model = true;
    }

    // Lock in configuration enforcement and verify all invariants before building the prompt; `rpc_call` also
    // drains any queued notifications and rechecks invariants/cancel/deadline right before writing `session/prompt`.
    state.enforce_config = true;
    try_drive!(state.verify_mode());
    try_drive!(state.verify_expected_config().map_err(|why| AgentError::Failed(Failure::new(why))));

    if req.cancel.load(Ordering::Relaxed) {
        state.kill_immediately = true;
        return DriveOutcome {
            result: Err(AgentError::Cancelled(state.usage.to_usage(0))),
            kill_immediately: true,
        };
    }
    if Instant::now() >= deadline {
        state.kill_immediately = true;
        return DriveOutcome {
            result: Err(AgentError::Timeout(state.usage.to_usage(0))),
            kill_immediately: true,
        };
    }

    // 4. Send the single-turn prompt with inline base64 JPEG images (`session/prompt`).
    let blocks = prompt_blocks(req, system_append_supported);
    let prompt_result = try_drive!(rpc_call(
        "session/prompt",
        json!({ "sessionId": sid, "prompt": blocks }),
        true,
        &mut state,
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
            result: Err(AgentError::Cancelled(state.usage.to_usage(0))),
            kill_immediately: true,
        };
    }
    if stop_reason == "refusal" {
        return DriveOutcome {
            result: Err(AgentError::Failed(Failure {
                used: state.usage.to_usage(1),
                ..Failure::new("Antigravity refused the prompt")
            })),
            kill_immediately: state.kill_immediately,
        };
    }
    if state.reply_text.trim().is_empty() {
        let used = state.usage.to_usage(0);
        return DriveOutcome {
            result: Err(if state.usage.seen() {
                AgentError::Failed(Failure { used, ..Failure::new("`agy` printed no reply") })
            } else {
                AgentError::Unavailable("`agy` printed no reply".into())
            }),
            kill_immediately: state.kill_immediately,
        };
    }
    DriveOutcome {
        result: Ok(AgentReply { text: state.reply_text, usage: state.usage.to_usage(1) }),
        kill_immediately: state.kill_immediately,
    }
}
