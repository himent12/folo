//! Stateful JSON-RPC request/response driver for ACP v1 sessions.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::super::process::unexpected_exit;
use super::super::protocol::rpc_failure;
use super::state::{SessionState, drain_pending, handle_incoming_line};
use super::{EXIT_DRAIN, StdoutLine};
use crate::label::agent::{AgentError, AgentRequest, Failure, POLL};

/// Stateful JSON-RPC transport driver so `drain` and `call` share `send`, `recv`, and `parent_exited` cleanly.
pub(super) struct RpcDriver<'a, T> {
    pub(super) req: &'a AgentRequest<'a>,
    pub(super) deadline: Instant,
    pub(super) cancel_grace: Duration,
    pub(super) next_id: u64,
    pub(super) send: &'a mut dyn FnMut(&Value) -> bool,
    pub(super) recv: &'a mpsc::Receiver<T>,
    pub(super) stderr_text: &'a dyn Fn() -> String,
    pub(super) parent_exited: &'a mut dyn FnMut() -> bool,
}

impl<T: Into<StdoutLine>> RpcDriver<'_, T> {
    pub(super) fn drain(&mut self, state: &mut SessionState) -> Result<(), AgentError> {
        drain_pending(
            self.recv,
            state,
            self.send,
            self.stderr_text,
            self.req.cancel,
            self.deadline,
            self.parent_exited,
        )
    }

    pub(super) fn call(
        &mut self,
        method: &str,
        params: Value,
        collect_reply: bool,
        state: &mut SessionState,
    ) -> Result<Value, AgentError> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        // Drain any already-queued notifications, verify active invariants, and check cancellation/deadline right
        // before dispatching so a late mode/config drift or stop never sends a billable prompt.
        self.drain(state)?;
        if state.enforce_mode {
            state.verify_mode()?;
        }
        if state.enforce_config {
            state.verify_expected_config().map_err(|why| AgentError::Failed(Failure::new(why)))?;
        }
        if self.req.cancel.load(Ordering::Relaxed) {
            state.kill_immediately = true;
            return Err(AgentError::Cancelled(state.usage.to_usage(0)));
        }
        if Instant::now() >= self.deadline {
            state.kill_immediately = true;
            return Err(AgentError::Timeout(state.usage.to_usage(0)));
        }
        state.mode_notified_in_call = false;
        state.config_updated_in_call = false;
        if !(self.send)(&msg) {
            return Err(unexpected_exit(&state.usage, self.stderr_text));
        }
        let mut cancel_deadline: Option<Instant> = None;
        let mut exit_drain_deadline: Option<Instant> = None;
        let mut next_exit_poll = Instant::now();
        loop {
            if cancel_deadline.is_none() && self.req.cancel.load(Ordering::Relaxed) {
                state.kill_immediately = true;
                if let Some(sid) = state.session_id.as_deref() {
                    let _ = (self.send)(&json!({
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": { "sessionId": sid }
                    }));
                    if collect_reply && !self.cancel_grace.is_zero() {
                        cancel_deadline = Some(Instant::now() + self.cancel_grace);
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
                if (self.parent_exited)() {
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
                return Err(unexpected_exit(&state.usage, self.stderr_text));
            }
            let mut wait_until = self.deadline;
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
                    return Err(unexpected_exit(&state.usage, self.stderr_text));
                }
                state.kill_immediately = true;
                if let Some(sid) = state.session_id.as_deref() {
                    let _ = (self.send)(&json!({
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
            match self.recv.recv_timeout(poll_wait) {
                Ok(item) => match item.into() {
                    StdoutLine::Error(why) => {
                        state.kill_immediately = true;
                        if cancel_deadline.is_some() {
                            return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                        }
                        return Err(AgentError::Failed(Failure {
                            used: state.usage.to_usage(0),
                            ..Failure::new(why)
                        }));
                    }
                    StdoutLine::Line(line) => {
                        if let Some(resp) = handle_incoming_line(
                            &line,
                            Some(id),
                            collect_reply,
                            cancel_deadline.is_some(),
                            state,
                            self.send,
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
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if cancel_deadline.is_some() {
                        return Err(AgentError::Cancelled(state.usage.to_usage(0)));
                    }
                    return Err(unexpected_exit(&state.usage, self.stderr_text));
                }
            }
        }
    }
}
