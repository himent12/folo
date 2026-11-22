//! Unit and fake-subprocess tests for the Antigravity ACP labeling driver.

mod session;
mod subprocess;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use base64::Engine;
use folo_api::{LabelEffort, LabelModel, LabelUsage};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::acp::{drive_acp, drive_acp_with_grace};
use super::process::{MAX_STDERR_BYTES, StderrTail, acp_command};
use super::protocol::{
    LABEL_AGENT_NAME, LABEL_CLI_DIR, find_effort_option, find_model_option, parse_config_options,
    prepare_isolation, resolve_effort_value, resolve_executable, resolve_model_and_effort, rpc_failure,
};
use super::{Antigravity, fallback_models, parse_models};
use crate::label::agent::{Agent, AgentError, AgentRequest};

fn test_req<'a>(
    dir: &'a Path,
    images: &'a [Vec<u8>],
    model: Option<&'a str>,
    effort: LabelEffort,
    schema: &'a Value,
    cancel: &'a AtomicBool,
) -> AgentRequest<'a> {
    AgentRequest {
        system: "label objects",
        prompt: "Classes: car\n2 images follow.",
        images,
        schema,
        model,
        effort,
        fast: false,
        scratch: dir,
        cancel,
    }
}

fn init_ok(id: Option<u64>, image: bool, append: bool, close: bool) -> String {
    let mut caps = json!({ "promptCapabilities": { "image": image } });
    if close {
        caps["sessionCapabilities"] = json!({ "close": {} });
    }
    if append {
        caps["_meta"] = json!({ "antigravity": { "systemPrompt": { "append": true } } });
    }
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "protocolVersion": 1, "agentCapabilities": caps }
    })
    .to_string()
}

#[test]
fn agy_models_output_strips_ansi_ignores_noise_and_collapses_suffixes() {
    let out = "\n\
        2026-11-22T10:20:30Z WARNING catalog is stale\n\
        sign-in required\n\
        \x1b[36mgemini-4-neon-xhigh\x1b[0m          Gemini 4 Neon (XHigh)\n\
        gemini-4-neon-high           Gemini 4 Neon (High)\n\
        gemini-4-neon-low            Gemini 4 Neon (Low)\n\
        gemini-4-hydrogen-high       Gemini 4 Hydrogen (High)\n\
        gemini-3.8-flash-medium      Gemini 3.8 Flash (Medium)\n\
        gemini-pro-orchestrator      Gemini Pro Orchestrator\n\
        gemini-4-carbon              Gemini 4 Carbon\n";
    let models = parse_models(out);
    assert_eq!(
        models,
        vec![
            LabelModel { id: "gemini-4-neon".into(), label: "Gemini 4 Neon".into(), fast: false },
            LabelModel { id: "gemini-4-hydrogen".into(), label: "Gemini 4 Hydrogen".into(), fast: false },
            LabelModel { id: "gemini-3.8-flash".into(), label: "Gemini 3.8 Flash".into(), fast: false },
            LabelModel { id: "gemini-4-carbon".into(), label: "Gemini 4 Carbon".into(), fast: false },
        ]
    );
    assert!(
        parse_models(
            "Error: Authentication required\nPlease sign in.\nRun `agy` to authenticate.\n\
             2026-11-22T10:20:30Z WARNING catalog is stale\nsign-in required\n"
        )
        .is_empty()
    );
    assert!(fallback_models().iter().any(|m| m.id == "gemini-4-neon" && m.label == "Gemini 4 Neon"));
}

#[test]
fn resolve_request_maps_effort_suffixes_and_rejects_conflicts() {
    let agy = Antigravity::new();
    assert_eq!(
        agy.resolve_request(Some("gemini-4-neon".into()), None),
        Ok((Some("gemini-4-neon".into()), LabelEffort::Low))
    );
    assert_eq!(
        agy.resolve_request(Some("gemini-4-neon-xhigh".into()), None),
        Ok((Some("gemini-4-neon-xhigh".into()), LabelEffort::High))
    );
    assert_eq!(
        agy.resolve_request(Some("gemini-4-neon-medium".into()), Some(LabelEffort::Medium)),
        Ok((Some("gemini-4-neon-medium".into()), LabelEffort::Medium))
    );
    let conflict =
        agy.resolve_request(Some("gemini-4-neon-xhigh".into()), Some(LabelEffort::Low)).unwrap_err();
    assert!(conflict.contains("conflicts") && conflict.contains("gemini-4-neon"), "{conflict}");
}

#[test]
fn model_and_effort_resolution_preserves_suffix_and_never_crosses_tiers() {
    let config = parse_config_options(Some(&json!([
        {
            "id": "model",
            "category": "model",
            "currentValue": "gemini-3.8-flash",
            "options": [
                {
                    "group": "gemini-4",
                    "name": "Gemini 4",
                    "options": [
                        { "value": "gemini-4-neon", "name": "Gemini 4 Neon" },
                        { "value": "gemini-4-neon-xhigh", "name": "Gemini 4 Neon (XHigh)" },
                        { "value": "gemini-4-hydrogen", "name": "Gemini 4 Hydrogen" }
                    ]
                },
                { "value": "gemini-3.8-flash", "name": "Gemini 3.8 Flash" }
            ]
        },
        {
            "id": "thought_level",
            "category": "thought_level",
            "currentValue": "high",
            "options": [
                { "value": "medium", "name": "Medium" },
                { "value": "high", "name": "High" },
                { "value": "xhigh", "name": "XHigh" }
            ]
        }
    ])));
    let model_opt = find_model_option(&config).unwrap();
    let effort_opt = find_effort_option(&config).unwrap();

    let plain = resolve_model_and_effort(Some("gemini-4-neon"), LabelEffort::Low, Some(model_opt), true);
    assert_eq!(
        (plain.model.as_deref(), plain.desired_effort.as_str(), plain.stripped_suffix),
        (Some("gemini-4-neon"), "low", false)
    );
    let exact =
        resolve_model_and_effort(Some("gemini-4-neon-xhigh"), LabelEffort::Low, Some(model_opt), true);
    assert_eq!(
        (exact.model.as_deref(), exact.desired_effort.as_str(), exact.stripped_suffix, exact.suffix_on_model),
        (Some("gemini-4-neon-xhigh"), "xhigh", false, true)
    );
    let stripped =
        resolve_model_and_effort(Some("gemini-4-hydrogen-xhigh"), LabelEffort::Low, Some(model_opt), true);
    assert_eq!(
        (stripped.model.as_deref(), stripped.desired_effort.as_str(), stripped.stripped_suffix),
        (Some("gemini-4-hydrogen"), "xhigh", true)
    );
    // Without an effort selector, prefer an advertised suffixed variant even when the bare base ID also exists.
    let both = parse_config_options(Some(&json!([{
        "id": "model",
        "category": "model",
        "currentValue": "gemini-4-neon",
        "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-4-neon-high" }]
    }])));
    let variant = resolve_model_and_effort(Some("gemini-4-neon"), LabelEffort::High, Some(&both[0]), false);
    assert_eq!(
        (variant.model.as_deref(), variant.desired_effort.as_str(), variant.suffix_on_model),
        (Some("gemini-4-neon-high"), "high", true)
    );

    // Unsupported tiers (`low` and `max` here) fail closed instead of silently running `medium` or `xhigh`.
    assert!(resolve_effort_value("low", effort_opt).is_err());
    assert!(resolve_effort_value("max", effort_opt).is_err());
    assert_eq!(resolve_effort_value("xhigh", effort_opt).as_deref(), Ok("xhigh"));
    // `low` and `minimal` are documented equivalents within `LabelEffort::Low`.
    let minimal_only = parse_config_options(Some(&json!([{
        "id": "thought_level",
        "category": "thought_level",
        "currentValue": "minimal",
        "options": [{ "value": "minimal" }, { "value": "medium" }]
    }])));
    assert_eq!(resolve_effort_value("low", &minimal_only[0]).as_deref(), Ok("minimal"));
}

#[test]
fn acp_round_trip_merges_usage_denies_tools_and_verifies_full_config_snapshots() {
    let dir = TestDir::new("label-agy-acp");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 1, 2], vec![0xff, 0xd8, 3, 4]];
    let req = test_req(dir.path(), &imgs, Some("gemini-4-neon-xhigh"), LabelEffort::Low, &schema, &cancel);

    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut sent = Vec::<Value>::new();
    let mut mode = "auto";
    let mut model = "gemini-3.8-flash";
    let mut effort = "low";
    let mut send = |msg: &Value| -> bool {
        sent.push(msg.clone());
        let id = msg.get("id").and_then(Value::as_u64);
        let snapshot = |mo: &str, m: &str, e: &str| {
            json!([
                { "id": "mode", "category": "mode", "currentValue": mo, "options": [{ "value": "auto" }, { "value": "default" }] },
                { "id": "model", "category": "model", "currentValue": m, "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-4-neon-xhigh" }, { "value": "gemini-3.8-flash" }] },
                { "id": "thought_level", "category": "thought_level", "currentValue": e, "options": [{ "value": "low" }, { "value": "xhigh" }] }
            ])
        };
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, true, true));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "sess-42", "configOptions": snapshot(mode, model, effort) } })
                        .to_string(),
                );
            }
            Some("session/set_config_option") => {
                match (msg["params"]["configId"].as_str(), msg["params"]["value"].as_str()) {
                    (Some("mode"), Some("default")) => mode = "default",
                    (Some("model"), Some("gemini-4-neon-xhigh")) => {
                        model = "gemini-4-neon-xhigh";
                        effort = "xhigh";
                    }
                    _ => {}
                }
                let _ = out_tx.send(
                    json!({ "jsonrpc": "2.0", "id": id, "result": { "configOptions": snapshot(mode, model, effort) } })
                        .to_string(),
                );
            }
            Some("session/prompt") => {
                for line in [
                    json!({
                        "jsonrpc": "2.0",
                        "id": 99,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": "sess-42",
                            "options": [
                                { "optionId": "allow", "kind": "allow_once", "name": "Allow" },
                                { "optionId": "deny", "kind": "reject_once", "name": "Reject" }
                            ]
                        }
                    }),
                    json!({ "jsonrpc": "2.0", "id": 100, "method": "fs/read_text_file", "params": { "path": "/etc/passwd" } }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "sessionId": "sess-42", "update": { "sessionUpdate": "agent_thought_chunk", "content": { "type": "text", "text": "thinking..." } } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "sessionId": "sess-42", "update": { "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "{\"images\":[" } } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "sessionId": "sess-42", "update": { "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "{\"image\":1,\"boxes\":[]}]}" } } }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "sess-42",
                            "update": {
                                "sessionUpdate": "usage_update",
                                "used": 9999,
                                "size": 1000000,
                                "inputTokens": 1200,
                                "outputTokens": 85,
                                "thoughtTokens": 40,
                                "totalTokens": 1325,
                                "cost": { "amount": 0.0042, "currency": "USD" }
                            }
                        }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": { "stopReason": "end_turn", "usage": { "cachedReadTokens": 300 } }
                    }),
                ] {
                    let _ = out_tx.send(line.to_string());
                }
            }
            _ => {}
        }
        true
    };

    let reply = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap();
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[]}]}"#);
    assert_eq!(
        reply.usage,
        LabelUsage {
            calls: 1,
            input_tokens: 1200,
            cached_input_tokens: 300,
            output_tokens: 125,
            cost_usd: Some(0.0042),
        }
    );

    let configs: Vec<(&str, &str)> = sent
        .iter()
        .filter(|m| m["method"] == "session/set_config_option")
        .map(|m| (m["params"]["configId"].as_str().unwrap(), m["params"]["value"].as_str().unwrap()))
        .collect();
    assert_eq!(configs, vec![("mode", "default"), ("model", "gemini-4-neon-xhigh")]);

    let new_msg = sent.iter().find(|m| m["method"] == "session/new").unwrap();
    assert!(Path::new(new_msg["params"]["cwd"].as_str().unwrap()).is_absolute());
    let prompt_msg = sent.iter().find(|m| m["method"] == "session/prompt").unwrap();
    let blocks = prompt_msg["params"]["prompt"].as_array().unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(
        blocks[2]["data"].as_str().unwrap(),
        base64::engine::general_purpose::STANDARD.encode(&imgs[0])
    );
    assert_eq!(
        sent.iter().find(|m| m.get("id") == Some(&json!(99))).unwrap()["result"]["outcome"],
        json!({ "outcome": "selected", "optionId": "deny" })
    );
    assert_eq!(sent.iter().find(|m| m.get("id") == Some(&json!(100))).unwrap()["error"]["code"], -32601);
}

#[test]
fn handshake_config_verification_mode_recheck_and_policy_aborts_fail_closed() {
    let dir = TestDir::new("label-agy-fail-closed");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 0xff, 0xd9]];

    // 1. Unsupported protocol version and omitted image capability fail closed.
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let _ = out_tx.send(
            json!({ "jsonrpc": "2.0", "id": msg.get("id"), "result": { "protocolVersion": 2 } }).to_string(),
        );
        true
    };
    assert!(matches!(
        drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Unavailable(m)) if m.contains("protocol version")
    ));

    let img_req = test_req(dir.path(), &imgs, None, LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let _ = out_tx.send(
            json!({ "jsonrpc": "2.0", "id": msg.get("id"), "result": { "protocolVersion": 1 } }).to_string(),
        );
        true
    };
    assert!(matches!(
        drive_acp(&img_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Unavailable(m)) if m.contains("inline image")
    ));

    // 2. Empty `{}` response from `session/set_config_option` fails closed (ACP requires updated `configOptions`).
    let model_req = test_req(dir.path(), &[], Some("gemini-4-neon"), LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string());
            }
            _ => {}
        }
        true
    };
    assert!(matches!(
        drive_acp(&model_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Failed(f)) if f.message.contains("did not return updated `configOptions`")
    ));

    // 3. Model setter that silently resets `mode` back to `auto` fails closed before prompting.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut saw_prompt = false;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [
                                { "id": "mode", "category": "mode", "currentValue": "default", "options": [{ "value": "default" }, { "value": "auto" }] },
                                { "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }] }
                            ]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "configOptions": [
                                { "id": "mode", "category": "mode", "currentValue": "auto", "options": [{ "value": "default" }, { "value": "auto" }] },
                                { "id": "model", "category": "model", "currentValue": "gemini-4-neon", "options": [{ "value": "gemini-4-neon" }] }
                            ]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => saw_prompt = true,
            _ => {}
        }
        true
    };
    assert!(matches!(
        drive_acp(&model_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Unavailable(m)) if m.contains("non-restricted mode `auto`")
    ));
    assert!(!saw_prompt);

    // 4. Effort setter that silently switches `model` back to another model fails closed before prompting.
    let high_req = test_req(dir.path(), &[], Some("gemini-4-neon"), LabelEffort::High, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [
                                { "id": "model", "category": "model", "currentValue": "gemini-4-neon", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }, { "value": "high" }] }
                            ]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "configOptions": [
                                { "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                { "id": "thought_level", "category": "thought_level", "currentValue": "high", "options": [{ "value": "low" }, { "value": "high" }] }
                            ]
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    assert!(matches!(
        drive_acp(&high_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Failed(f)) if f.message.contains("selected model `gemini-3.8-flash` instead of requested `gemini-4-neon`")
    ));

    // 5. Unrecognized mode id fails closed (allowlist, not denylist).
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "mode", "category": "mode", "currentValue": "untrusted-custom", "options": [{ "value": "untrusted-custom" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    assert!(matches!(
        drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Unavailable(m)) if m.contains("offers no restricted mode")
    ));

    // 6. `tool_call`, permissive `current_mode_update`, or permissive `config_option_update` mid-prompt aborts
    // immediately (`kill_immediately`), preserves already-reported usage, and returns a non-retryable failure.
    for (notif, expect_msg) in [
        (json!({ "sessionUpdate": "tool_call", "title": "read_file" }), "invoke a tool"),
        (
            json!({ "sessionUpdate": "current_mode_update", "currentModeId": "auto" }),
            "non-restricted mode `auto`",
        ),
        (
            json!({
                "sessionUpdate": "config_option_update",
                "configOptions": [{ "id": "mode", "category": "mode", "currentValue": "auto", "options": [{ "value": "auto" }] }]
            }),
            "non-restricted mode `auto`",
        ),
    ] {
        let (out_tx, out_rx) = mpsc::channel::<String>();
        let mut saw_cancel = false;
        let mut send = |msg: &Value| -> bool {
            let id = msg.get("id").and_then(Value::as_u64);
            match msg["method"].as_str() {
                Some("initialize") => {
                    let _ = out_tx.send(init_ok(id, true, false, false));
                }
                Some("session/new") => {
                    let _ = out_tx.send(
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "sessionId": "s1",
                                "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                            }
                        })
                        .to_string(),
                    );
                }
                Some("session/prompt") => {
                    let _ = out_tx.send(
                        json!({
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": { "update": { "sessionUpdate": "usage_update", "inputTokens": 321, "outputTokens": 11 } }
                        })
                        .to_string(),
                    );
                    let _ = out_tx.send(
                        json!({ "jsonrpc": "2.0", "method": "session/update", "params": { "update": notif } })
                            .to_string(),
                    );
                }
                Some("session/cancel") => saw_cancel = true,
                _ => {}
            }
            true
        };
        let out = drive_acp_with_grace(
            &req,
            &work,
            Duration::from_secs(5),
            Duration::from_millis(200),
            &mut send,
            &out_rx,
            &String::new,
        );
        assert!(out.kill_immediately && saw_cancel);
        let Err(AgentError::Failed(f)) = out.result else { panic!("{:?}", out.result) };
        assert!(f.message.contains(expect_msg) && !f.is_transient(), "{f:?}");
        assert_eq!((f.used.calls, f.used.input_tokens, f.used.output_tokens), (1, 321, 11));
    }
}

#[test]
fn usage_does_not_invent_context_occupancy_and_preserves_quiet_exit_and_cancel_usage() {
    let dir = TestDir::new("label-agy-usage");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);

    // 1. Context-only `usage_update` never invents billed input tokens.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "{\"images\":[]}" } } }
                    })
                    .to_string(),
                );
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 1400, "size": 1000000 } }
                    })
                    .to_string(),
                );
                let _ = out_tx.send(
                    json!({ "jsonrpc": "2.0", "id": id, "result": { "stopReason": "end_turn" } }).to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let reply = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap();
    assert_eq!(reply.usage, LabelUsage { calls: 1, ..LabelUsage::default() });

    // 2. Quiet stdout EOF after billed usage returns Failed preserving `used`.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut out_opt = Some(out_tx);
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        let Some(tx) = out_opt.as_ref() else { return false };
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => {
                let _ = tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "usage_update", "inputTokens": 777, "outputTokens": 12 } }
                    })
                    .to_string(),
                );
                out_opt.take();
            }
            _ => {}
        }
        true
    };
    let err = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert_eq!(
        err,
        AgentError::Failed(crate::label::agent::Failure {
            used: LabelUsage { calls: 1, input_tokens: 777, output_tokens: 12, ..LabelUsage::default() },
            ..crate::label::agent::Failure::new("`agy` exited before replying")
        })
    );

    // 3. Responsive cancellation grace captures final usage reported on `stopReason: "cancelled"`.
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let req_cancel = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel_flag);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut prompt_id = 0u64;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => {
                prompt_id = id.unwrap();
                cancel_flag.store(true, Ordering::Relaxed);
            }
            Some("session/cancel") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": prompt_id,
                        "result": { "stopReason": "cancelled", "usage": { "inputTokens": 450, "outputTokens": 25 } }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let out = drive_acp_with_grace(
        &req_cancel,
        &work,
        Duration::from_secs(5),
        Duration::from_millis(200),
        &mut send,
        &out_rx,
        &String::new,
    );
    assert!(out.kill_immediately);
    assert_eq!(
        out.result,
        Err(AgentError::Cancelled(LabelUsage {
            calls: 1,
            input_tokens: 450,
            output_tokens: 25,
            ..LabelUsage::default()
        }))
    );
}

#[test]
fn structured_errors_hints_stderr_cap_and_isolation_work_cleanly() {
    let f = rpc_failure(
        &json!({
            "code": -32000,
            "message": "Request failed",
            "data": {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota exceeded for requests per minute",
                    "details": [{ "reason": "RATE_LIMIT_EXCEEDED", "retryDelay": "15s" }]
                }
            }
        }),
        LabelUsage::default(),
    );
    assert_eq!(f.status, Some(429));
    assert!(f.is_transient(), "structured per-minute RATE_LIMIT_EXCEEDED with 15s delay must retry: {f:?}");

    let agy = Antigravity::new();
    assert!(agy.hint("Authentication required").unwrap().contains("agy acp --login"));
    assert!(agy.hint(&f.message).unwrap().contains("provider's rate limit"));
    assert!(
        agy.hint("RESOURCE_EXHAUSTED: You exceeded your current quota.").unwrap().contains("usage limit")
    );
    assert_eq!(agy.hint("API Error: 503 Service Unavailable").as_deref(), Some(crate::label::agent::BUSY));
    assert!(agy.hint("unknown model foo").unwrap().contains("pick another model"));
    assert!(agy.hint("`agy` printed no reply").unwrap().contains("`agy -p hi`"));

    // Bounded stderr tail keeps the latest 64 KiB and drains deterministically (no fixed sleep).
    let tail = StderrTail::new();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let t2 = tail.clone();
    let writer = std::thread::spawn(move || {
        go_rx.recv().unwrap();
        t2.push(&vec![b'a'; MAX_STDERR_BYTES + 100]);
        t2.push(b"\nAuthentication required\n");
        t2.mark_ended();
    });
    go_tx.send(()).unwrap();
    let text = tail.drain_text(Duration::from_secs(5));
    writer.join().unwrap();
    assert_eq!(text.len(), MAX_STDERR_BYTES);
    assert!(text.ends_with("Authentication required\n"));

    assert_eq!(resolve_executable(Path::new("agy")), PathBuf::from("agy"));
    assert!(resolve_executable(Path::new("./bin/agy")).is_absolute());

    let dir = TestDir::new("label-agy-iso");
    let home = dir.join("user-home");
    let iso = prepare_isolation(dir.path(), Some(&home)).unwrap();
    let agent_md =
        std::fs::read_to_string(iso.work.join(".agents/agents").join(LABEL_AGENT_NAME).join("agent.md"))
            .unwrap();
    assert!(agent_md.contains("inheritMcp: false") && agent_md.contains("inheritCustomizations: false"));
    let settings: Value = serde_json::from_str(
        &std::fs::read_to_string(
            home.join(".gemini/antigravity-cli").join(LABEL_CLI_DIR).join("settings.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(settings["toolPermission"], "strict");
    assert_eq!(settings["allowNonWorkspaceAccess"], false);
    assert!(settings["permissions"]["deny"].as_array().unwrap().contains(&json!("read_file(*)")));
    let cmd = acp_command(Path::new("./bin/agy"), &iso);
    let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
    assert!(args.windows(2).any(|w| w == ["--agent", LABEL_AGENT_NAME]));
    assert!(args.windows(2).any(|w| w == ["--cli_dir_name", LABEL_CLI_DIR]));
    assert!(args.iter().any(|a| a == "--sandbox"));
    assert!(Path::new(cmd.get_program()).is_absolute());
}
