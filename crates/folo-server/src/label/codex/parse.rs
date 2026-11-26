//! Parsing `codex exec --json` stdout events and token usage.

use folo_api::LabelUsage;
use serde_json::Value;

use super::super::agent::{AgentError, AgentReply, Failure, Output, last_line};

/// Reads the events `codex exec --json` prints: the last agent message is the reply and `turn.completed` has
/// the usage; without them, `turn.failed` / `error` say why. (An `error` before a completed turn was a retry, and
/// `item.completed` items of type `error` are warnings, e.g. about a config key this version does not know.)
///
/// What a completed turn used is counted whether or not there is a reply: a failure carries it.
pub(in crate::label) fn parse(out: &Output) -> Result<AgentReply, AgentError> {
    let (mut text, mut failure) = (None, None);
    for event in out.stdout.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
        match event["type"].as_str() {
            Some("item.completed") if event["item"]["type"] == "agent_message" => {
                text = event["item"]["text"].as_str().map(str::to_string);
            }
            Some("turn.failed") => failure = event["error"]["message"].as_str().map(str::to_string),
            Some("error") => failure = event["message"].as_str().map(str::to_string),
            _ => {}
        }
    }
    let usage = usage_of(&out.stdout);
    let (Some(text), Some(usage)) = (text, usage) else {
        let used = usage.unwrap_or_default();
        let why = failure
            .or_else(|| last_line(&out.stderr).filter(|_| !out.success))
            .or_else(|| last_line(&out.stdout).filter(|_| !out.success));
        return Err(match why {
            Some(why) => AgentError::Failed(Failure { used, ..Failure::new(why) }),
            // A turn that completed without a message still used its tokens.
            None if usage.is_some() => {
                AgentError::Failed(Failure { used, ..Failure::new("`codex` printed no reply") })
            }
            None => AgentError::Unavailable("`codex` printed no reply".into()),
        });
    };
    Ok(AgentReply { text, usage })
}

/// What the `turn.completed` event among the events `stdout` has says the call used; `None` without one.
pub(super) fn usage_of(stdout: &str) -> Option<LabelUsage> {
    let event = stdout
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|event| event["type"] == "turn.completed")?;
    let tokens = |key: &str| event["usage"][key].as_u64().unwrap_or(0);
    Some(LabelUsage {
        calls: 1,
        input_tokens: tokens("input_tokens"),
        cached_input_tokens: tokens("cached_input_tokens"),
        output_tokens: tokens("output_tokens"),
        cost_usd: None,
    })
}
