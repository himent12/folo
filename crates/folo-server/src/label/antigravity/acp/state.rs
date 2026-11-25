//! Live ACP session state, incoming notification/request handling, and bounded pre-dispatch stdout draining.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::super::process::unexpected_exit;
use super::super::protocol::{
    ConfigOption, UsageTracker, find_effort_option, find_mode_option, find_model_option, is_restricted_mode,
    parse_config_options, reject_permission,
};
use super::StdoutLine;
use crate::label::agent::{AgentError, Failure};

/// Maximum queued lines drained before dispatching an RPC; bounds work if a noisy child/helper floods stdout.
const MAX_DRAIN_LINES: usize = 256;
/// Maximum wall-clock budget for draining queued stdout lines before dispatching an RPC.
const DRAIN_BUDGET: Duration = Duration::from_millis(250);

/// Live ACP session state tracked across responses and `session/update` notifications in receive order.
#[derive(Default)]
pub(super) struct SessionState {
    pub(super) session_id: Option<String>,
    pub(super) config_options: Vec<ConfigOption>,
    /// True once `configOptions` has advertised a `mode` selector.
    pub(super) mode_option_seen: bool,
    /// True once `configOptions` has advertised a `model` selector.
    pub(super) model_option_seen: bool,
    /// True once `configOptions` has advertised a `thought_level`/effort selector.
    pub(super) effort_option_seen: bool,
    /// Latest legacy mode id from `session/new.modes` or `current_mode_update` notifications.
    pub(super) legacy_mode: Option<String>,
    /// Set by `current_mode_update` during an RPC call so `session/set_mode` never overwrites a live update.
    pub(super) mode_notified_in_call: bool,
    /// Set by `config_option_update` during an RPC call so legacy `session/set_model` knows whether effort was refreshed.
    pub(super) config_updated_in_call: bool,
    /// Set once step 2b verifies restricted mode; any later mode drift fails closed immediately.
    pub(super) enforce_mode: bool,
    /// Set once step 3 verifies model and effort; any later `config_option_update` drift fails closed immediately.
    pub(super) enforce_config: bool,
    /// True when the caller left `model` unset so the first reported default model is pinned against later drift.
    pub(super) pin_default_model: bool,
    pub(super) expected_model: Option<String>,
    pub(super) expected_effort: Option<String>,
    /// True when no `thought_level` selector exists and the verified model ID itself encodes `expected_effort`.
    pub(super) effort_encoded_on_model: bool,
    pub(super) usage: UsageTracker,
    pub(super) reply_text: String,
    pub(super) kill_immediately: bool,
}

impl SessionState {
    pub(super) fn apply_config_snapshot(&mut self, opts: Vec<ConfigOption>) {
        if find_mode_option(&opts).is_some() {
            self.mode_option_seen = true;
        }
        if let Some(model_opt) = find_model_option(&opts) {
            self.model_option_seen = true;
            if self.pin_default_model
                && self.expected_model.is_none()
                && let Some(cur) = model_opt.current_value.as_deref()
            {
                self.expected_model = Some(cur.to_string());
            }
        }
        if find_effort_option(&opts).is_some() {
            self.effort_option_seen = true;
        }
        self.config_options = opts;
    }

    pub(super) fn verify_mode(&self) -> Result<(), AgentError> {
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

    pub(super) fn verify_expected_config(&self) -> Result<(), String> {
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

    pub(super) fn abort_policy(
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
pub(super) fn handle_incoming_line(
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
pub(super) fn drain_pending<T: Into<StdoutLine>>(
    recv: &mpsc::Receiver<T>,
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
            Ok(item) => match item.into() {
                StdoutLine::Error(why) => {
                    state.kill_immediately = true;
                    return Err(AgentError::Failed(Failure {
                        used: state.usage.to_usage(0),
                        ..Failure::new(why)
                    }));
                }
                StdoutLine::Line(line) => {
                    if drained >= MAX_DRAIN_LINES || now >= drain_deadline {
                        state.kill_immediately = true;
                        return Err(AgentError::Failed(Failure {
                            used: state.usage.to_usage(0),
                            ..Failure::new("`agy acp` flooded stdout before the next request")
                        }));
                    }
                    drained += 1;
                    let _ = handle_incoming_line(&line, None, false, false, state, send)?;
                    let after = Instant::now();
                    if after >= deadline {
                        state.kill_immediately = true;
                        return Err(AgentError::Timeout(state.usage.to_usage(0)));
                    }
                    if after >= drain_deadline {
                        state.kill_immediately = true;
                        return Err(AgentError::Failed(Failure {
                            used: state.usage.to_usage(0),
                            ..Failure::new("`agy acp` flooded stdout before the next request")
                        }));
                    }
                }
            },
            Err(mpsc::TryRecvError::Empty) => return Ok(()),
            Err(mpsc::TryRecvError::Disconnected) => return Err(unexpected_exit(&state.usage, stderr_text)),
        }
    }
}
