//! Single-turn ACP v1 JSON-RPC session orchestration and mode/model/effort invariant enforcement.
//!
//! Files: this one holds [`drive_acp_inner`], [`DriveOutcome`], and the handshake/config/prompt flow; `state` holds
//! [`SessionState`], incoming notification/request handling, and bounded pre-dispatch draining; `driver` holds
//! [`RpcDriver`] (the JSON-RPC request/response wait loop).

mod driver;
mod state;

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use self::driver::RpcDriver;
use self::state::SessionState;
use super::protocol::{
    ConfigOption, find_effort_option, find_mode_option, find_model_option, is_restricted_mode,
    match_option_value, parse_config_options, parse_legacy_models, pick_safe_mode, prompt_blocks,
    resolve_effort_value, resolve_model_and_effort, suffix_matches_effort,
};
use super::split_effort_suffix;
use crate::label::agent::{AgentError, AgentReply, AgentRequest, Failure};

/// How long to wait for a responsive `agy acp` child to acknowledge `session/cancel` and emit final usage.
pub(super) const CANCEL_GRACE: Duration = Duration::from_millis(500);
/// How long to drain buffered stdout/stderr after the parent process has exited before reporting early exit.
pub(super) const EXIT_DRAIN: Duration = Duration::from_millis(250);

/// Outcome of driving one ACP conversation, plus whether the caller must kill the child tree immediately (skipping
/// any graceful exit window, e.g. on cancellation, timeout, or a tool/mode/config policy violation).
pub(super) struct DriveOutcome {
    pub result: Result<AgentReply, AgentError>,
    pub kill_immediately: bool,
}

/// One item delivered by the stdout reader thread to the ACP driver, preserving framing/read errors in stream order.
pub(super) enum StdoutLine {
    Line(String),
    Error(String),
}

impl From<String> for StdoutLine {
    fn from(line: String) -> Self {
        StdoutLine::Line(line)
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
pub(super) fn drive_acp_inner<T: Into<StdoutLine>>(
    req: &AgentRequest<'_>,
    work: &Path,
    timeout: Duration,
    cancel_grace: Duration,
    send: &mut dyn FnMut(&Value) -> bool,
    recv: &mpsc::Receiver<T>,
    stderr_text: &dyn Fn() -> String,
    parent_exited: &mut dyn FnMut() -> bool,
) -> DriveOutcome {
    let deadline = Instant::now() + timeout;
    let mut state = SessionState {
        pin_default_model: req.model.map(str::trim).is_none_or(str::is_empty),
        ..SessionState::default()
    };
    let mut driver =
        RpcDriver { req, deadline, cancel_grace, next_id: 1, send, recv, stderr_text, parent_exited };

    macro_rules! try_drive {
        ($expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return DriveOutcome { result: Err(e), kill_immediately: state.kill_immediately },
            }
        };
    }

    // 1. Handshake (`initialize`): require protocol v1 and explicit image support for image-bearing calls.
    let init = try_drive!(driver.call(
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
    let session = try_drive!(driver.call("session/new", new_params, false, &mut state));
    let sid = try_drive!(
        session["sessionId"]
            .as_str()
            .ok_or_else(|| AgentError::Unavailable("`agy acp` returned no sessionId".into()))
    )
    .to_string();
    state.session_id = Some(sid.clone());
    state.apply_config_snapshot(parse_config_options(session.get("configOptions")));
    if state.pin_default_model
        && state.expected_model.is_none()
        && let Some(cur) = session.get("models").and_then(|m| m["currentModelId"].as_str())
    {
        state.expected_model = Some(cur.to_string());
    }
    if !state.mode_notified_in_call && !state.mode_option_seen {
        state.legacy_mode =
            session.get("modes").and_then(|m| m["currentModeId"].as_str()).map(str::to_string);
    }
    // Drain any updates queued right after `session/new` before enforcing mode or resolving model/effort.
    try_drive!(driver.drain(&mut state));

    // 2b. Enforce an allowlisted restricted execution mode whenever a mode selector is advertised.
    if let Some(mode_opt) = find_mode_option(&state.config_options).cloned() {
        if !mode_opt.current_value.as_deref().is_some_and(is_restricted_mode) {
            let cur = mode_opt.current_value.as_deref().unwrap_or("<unset>");
            let safe = try_drive!(pick_safe_mode(&mode_opt.values).ok_or_else(|| {
                AgentError::Unavailable(format!(
                    "`agy acp` started in non-restricted mode `{cur}` and offers no restricted mode"
                ))
            }));
            let updated = try_drive!(driver.call(
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
            let _ = try_drive!(driver.call(
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
    try_drive!(driver.drain(&mut state));

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
        find_effort_option(&state.config_options).is_some(),
    );
    let mut verified_legacy_model: Option<String> =
        legacy_model_opt.as_ref().and_then(|o| o.current_value.clone());
    let mut legacy_switched_to: Option<String> = None;

    if let Some(model) = res.model {
        if let Some(opt) = find_model_option(&state.config_options).cloned() {
            state.expected_model = Some(model.clone());
            if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&model)) {
                let updated = try_drive!(driver.call(
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
                let _ = try_drive!(driver.call(
                    "session/set_model",
                    json!({ "sessionId": sid, "modelId": target }),
                    false,
                    &mut state,
                ));
                legacy_switched_to = Some(target.clone());
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
    } else if state.expected_model.is_none()
        && let Some(cur) = find_model_option(&state.config_options)
            .and_then(|o| o.current_value.as_deref())
            .or(verified_legacy_model.as_deref())
    {
        // Pin the session's reported default model so a later setter or mid-prompt update cannot drift away.
        state.expected_model = Some(cur.to_string());
    }
    try_drive!(driver.drain(&mut state));
    if let Some(target) = legacy_switched_to.as_deref() {
        // Legacy `session/set_model` returns `{}` without refreshed `configOptions`: invalidate any prior or
        // queued effort reading unless a `config_option_update` explicitly confirmed BOTH `model == target` and
        // its effort option.
        let confirmed_target = state.config_updated_in_call
            && find_model_option(&state.config_options)
                .and_then(|o| o.current_value.as_deref())
                .is_some_and(|m| m.eq_ignore_ascii_case(target));
        if !confirmed_target {
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

    if let Some(opt) = find_effort_option(&state.config_options).cloned() {
        let effort_value = try_drive!(
            resolve_effort_value(&res.desired_effort, &opt)
                .map_err(|why| AgentError::Failed(Failure::new(why)))
        );
        state.expected_effort = Some(effort_value.clone());
        if !opt.current_value.as_deref().is_some_and(|cur| cur.eq_ignore_ascii_case(&effort_value)) {
            let updated = try_drive!(driver.call(
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

    // Lock in configuration enforcement and verify all invariants before building the prompt; `driver.call` also
    // drains any queued notifications and rechecks invariants/cancel/deadline right before writing `session/prompt`.
    state.enforce_config = true;
    try_drive!(driver.drain(&mut state));
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
    let prompt_result = try_drive!(driver.call(
        "session/prompt",
        json!({ "sessionId": sid, "prompt": blocks }),
        true,
        &mut state,
    ));

    // 5. Best-effort clean session close.
    if close_supported {
        let close_id = driver.next_id;
        let _ = (driver.send)(&json!({
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
