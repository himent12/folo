//! Unit and fake-subprocess tests for the Antigravity ACP labeling driver.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use base64::Engine;
use folo_api::{LabelEffort, LabelModel, LabelUsage};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::acp::{
    MAX_STDERR_BYTES, StderrTail, acp_command, drive_acp, drive_acp_with_grace, run_acp_with_home,
};
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
fn model_and_effort_resolution_preserves_exact_suffix_and_supports_grouped_options() {
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

    // Plain base model keeps requested effort.
    let plain = resolve_model_and_effort(Some("gemini-4-neon"), LabelEffort::Low, Some(model_opt));
    assert_eq!(
        (plain.model.as_deref(), plain.desired_effort.as_str(), plain.stripped_suffix),
        (Some("gemini-4-neon"), "low", false)
    );
    // Exact suffixed model keeps its suffix effort (`xhigh`) so `thought_level` is never downgraded to `low`.
    let exact = resolve_model_and_effort(Some("gemini-4-neon-xhigh"), LabelEffort::Low, Some(model_opt));
    assert_eq!(
        (exact.model.as_deref(), exact.desired_effort.as_str(), exact.stripped_suffix, exact.suffix_on_model),
        (Some("gemini-4-neon-xhigh"), "xhigh", false, true)
    );
    // Suffixed model whose base (not suffixed slug) is listed strips the suffix and sets `stripped_suffix`.
    let stripped =
        resolve_model_and_effort(Some("gemini-4-hydrogen-xhigh"), LabelEffort::Low, Some(model_opt));
    assert_eq!(
        (stripped.model.as_deref(), stripped.desired_effort.as_str(), stripped.stripped_suffix),
        (Some("gemini-4-hydrogen"), "xhigh", true)
    );
    // Unknown future model passes straight through.
    let future = resolve_model_and_effort(Some("gemini-5-alpha"), LabelEffort::Medium, Some(model_opt));
    assert_eq!((future.model.as_deref(), future.desired_effort.as_str()), (Some("gemini-5-alpha"), "medium"));

    assert_eq!(resolve_effort_value("low", effort_opt).as_deref(), Some("medium"));
    assert_eq!(resolve_effort_value("xhigh", effort_opt).as_deref(), Some("xhigh"));
    assert_eq!(resolve_effort_value("max", effort_opt).as_deref(), Some("xhigh"));
}

#[test]
fn acp_round_trip_merges_usage_denies_tools_and_keeps_exact_suffix_effort() {
    let dir = TestDir::new("label-agy-acp");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 1, 2], vec![0xff, 0xd8, 3, 4]];
    let req = test_req(dir.path(), &imgs, Some("gemini-4-neon-xhigh"), LabelEffort::Low, &schema, &cancel);

    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut sent = Vec::<Value>::new();
    let mut send = |msg: &Value| -> bool {
        sent.push(msg.clone());
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, true, true));
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "sess-42",
                            "configOptions": [
                                {
                                    "id": "mode",
                                    "category": "mode",
                                    "currentValue": "auto",
                                    "options": [
                                        { "value": "auto", "name": "Auto" },
                                        { "value": "default", "name": "Default" }
                                    ]
                                },
                                {
                                    "id": "model",
                                    "category": "model",
                                    "currentValue": "gemini-3.8-flash",
                                    "options": [
                                        { "value": "gemini-4-neon", "name": "Gemini 4 Neon" },
                                        { "value": "gemini-4-neon-xhigh", "name": "Gemini 4 Neon (XHigh)" },
                                        { "value": "gemini-3.8-flash", "name": "Gemini 3.8 Flash" }
                                    ]
                                },
                                {
                                    "id": "thought_level",
                                    "category": "thought_level",
                                    "currentValue": "low",
                                    "options": [
                                        { "value": "low", "name": "Low" },
                                        { "value": "xhigh", "name": "XHigh" }
                                    ]
                                }
                            ]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                let cid = msg["params"]["configId"].as_str().unwrap_or("");
                let val = msg["params"]["value"].as_str().unwrap_or("");
                let result = if cid == "model" && val == "gemini-4-neon-xhigh" {
                    // Selecting the suffixed model already updates thought_level to `xhigh`; FOLO must NOT
                    // downgrade it back to `low`.
                    json!({
                        "configOptions": [
                            { "id": "mode", "category": "mode", "currentValue": "default", "options": [{ "value": "default" }] },
                            { "id": "model", "category": "model", "currentValue": "gemini-4-neon-xhigh", "options": [{ "value": "gemini-4-neon-xhigh" }] },
                            { "id": "thought_level", "category": "thought_level", "currentValue": "xhigh", "options": [{ "value": "low" }, { "value": "xhigh" }] }
                        ]
                    })
                } else {
                    json!({})
                };
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string());
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
                    json!({
                        "jsonrpc": "2.0",
                        "id": 100,
                        "method": "fs/read_text_file",
                        "params": { "path": "/etc/passwd" }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "sess-42",
                            "update": {
                                "sessionUpdate": "agent_thought_chunk",
                                "content": { "type": "text", "text": "thinking privately..." }
                            }
                        }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "sess-42",
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": { "type": "text", "text": "{\"images\":[" }
                            }
                        }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "sess-42",
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": { "type": "text", "text": "{\"image\":1,\"boxes\":[]}]}" }
                            }
                        }
                    }),
                    // Earlier update reports input/output/thought tokens + context occupancy `used` + USD cost.
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
                    // Final result has only a partial usage object (`cachedReadTokens`): must merge without wiping
                    // the earlier `inputTokens` / `outputTokens` / `thoughtTokens`.
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "stopReason": "end_turn",
                            "usage": { "cachedReadTokens": 300 }
                        }
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

    // Auto mode was switched to `default`, suffixed model selected, and `thought_level` was NOT downgraded to `low`.
    let configs: Vec<(&str, &str)> = sent
        .iter()
        .filter(|m| m["method"] == "session/set_config_option")
        .map(|m| (m["params"]["configId"].as_str().unwrap(), m["params"]["value"].as_str().unwrap()))
        .collect();
    assert_eq!(configs, vec![("mode", "default"), ("model", "gemini-4-neon-xhigh")]);

    let new_msg = sent.iter().find(|m| m["method"] == "session/new").unwrap();
    assert!(Path::new(new_msg["params"]["cwd"].as_str().unwrap()).is_absolute());
    assert!(
        new_msg["params"]["_meta"]["antigravity"]["systemPrompt"]["append"]
            .as_str()
            .unwrap()
            .contains("label objects")
    );
    assert_eq!(new_msg["params"]["mcpServers"], json!([]));

    let prompt_msg = sent.iter().find(|m| m["method"] == "session/prompt").unwrap();
    let blocks = prompt_msg["params"]["prompt"].as_array().unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(blocks[2]["type"], "image");
    assert_eq!(blocks[2]["mimeType"], "image/jpeg");
    assert_eq!(
        blocks[2]["data"].as_str().unwrap(),
        base64::engine::general_purpose::STANDARD.encode(&imgs[0])
    );

    let perm_reply = sent.iter().find(|m| m.get("id") == Some(&json!(99))).unwrap();
    assert_eq!(perm_reply["result"]["outcome"], json!({ "outcome": "selected", "optionId": "deny" }));
    let method_reply = sent.iter().find(|m| m.get("id") == Some(&json!(100))).unwrap();
    assert_eq!(method_reply["error"]["code"], -32601);
    assert!(sent.iter().any(|m| m["method"] == "session/close"));
}

#[test]
fn handshake_mode_tool_call_and_unconfigurable_model_fail_closed() {
    let dir = TestDir::new("label-agy-fail-closed");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 0xff, 0xd9]];

    // 1. Unsupported protocol version is rejected.
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        let _ = out_tx
            .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "protocolVersion": 2 } }).to_string());
        true
    };
    let err = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert!(matches!(err, AgentError::Unavailable(m) if m.contains("protocol version")));

    // 2. Omitted image capability is rejected when images are present.
    let img_req = test_req(dir.path(), &imgs, None, LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        let _ = out_tx
            .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "protocolVersion": 1 } }).to_string());
        true
    };
    let err =
        drive_acp(&img_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert!(matches!(err, AgentError::Unavailable(m) if m.contains("inline image")));

    // 3. Session without model selector fails closed when a model was explicitly requested.
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
                    json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1", "configOptions": [] } })
                        .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let err =
        drive_acp(&model_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert!(matches!(err, AgentError::Failed(f) if f.message.contains("selecting model `gemini-4-neon`")));

    // 4. Auto-approving mode with no safe alternative fails closed.
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
                            "configOptions": [{
                                "id": "mode",
                                "category": "mode",
                                "currentValue": "auto",
                                "options": [{ "value": "auto" }, { "value": "bypass" }]
                            }]
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let err = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert!(matches!(err, AgentError::Unavailable(m) if m.contains("auto-approving mode")));

    // 5. Unexpected `tool_call` update during prompt immediately aborts closed and sends `session/cancel`.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut saw_cancel = false;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx
                    .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1" } }).to_string());
            }
            Some("session/prompt") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": { "update": { "sessionUpdate": "tool_call", "title": "read_file" } }
                    })
                    .to_string(),
                );
            }
            Some("session/cancel") => saw_cancel = true,
            _ => {}
        }
        true
    };
    let err = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert!(matches!(err, AgentError::Unavailable(m) if m.contains("invoke a tool")));
    assert!(saw_cancel);
}

#[test]
fn usage_does_not_invent_context_occupancy_and_preserves_quiet_exit_and_cancel_usage() {
    let dir = TestDir::new("label-agy-usage");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);

    // 1. Context-only `usage_update` (`used: 1400, size: 1000000`) never invents billed input tokens.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx
                    .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1" } }).to_string());
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
    assert_eq!(
        reply.usage,
        LabelUsage { calls: 1, input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, cost_usd: None }
    );

    // 2. Quiet stdout EOF after billed usage returns Failed preserving `used` (never drops tokens via Unavailable).
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
                let _ = tx
                    .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1" } }).to_string());
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
                // Drop sender (quiet EOF without stderr).
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
                let _ = out_tx
                    .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1" } }).to_string());
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
                        "result": {
                            "stopReason": "cancelled",
                            "usage": { "inputTokens": 450, "outputTokens": 25 }
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let err = drive_acp_with_grace(
        &req_cancel,
        &work,
        Duration::from_secs(5),
        Duration::from_millis(200),
        &mut send,
        &out_rx,
        &String::new,
    )
    .unwrap_err();
    assert_eq!(
        err,
        AgentError::Cancelled(LabelUsage {
            calls: 1,
            input_tokens: 450,
            output_tokens: 25,
            ..LabelUsage::default()
        })
    );
}

#[test]
fn structured_errors_hints_stderr_cap_and_paths_work_cleanly() {
    // Structured Google RPC error preserves `RESOURCE_EXHAUSTED`, HTTP 429, and retryDelay.
    let f = rpc_failure(
        &json!({
            "code": -32000,
            "message": "Request failed",
            "data": {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{ "retryDelay": "15s" }]
                }
            }
        }),
        LabelUsage::default(),
    );
    assert_eq!(f.status, Some(429));
    assert!(f.message.contains("RESOURCE_EXHAUSTED") && f.message.contains("15s"), "{}", f.message);

    let agy = Antigravity::new();
    assert!(agy.hint("Authentication required").unwrap().contains("agy acp --login"));
    assert!(agy.hint(&f.message).unwrap().contains("provider's rate limit"));
    assert!(
        agy.hint("RESOURCE_EXHAUSTED: You exceeded your current quota.").unwrap().contains("usage limit")
    );
    assert_eq!(agy.hint("API Error: 503 Service Unavailable").as_deref(), Some(crate::label::agent::BUSY));
    assert!(agy.hint("unknown model foo").unwrap().contains("pick another model"));
    assert!(agy.hint("`agy` printed no reply").unwrap().contains("`agy -p hi`"));

    // Bounded stderr tail keeps the latest 64 KiB and waits for delayed EOF.
    let tail = StderrTail::new();
    tail.push(&vec![b'a'; MAX_STDERR_BYTES + 100]);
    tail.push(b"\nAuthentication required\n");
    let t2 = tail.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        t2.mark_ended();
    });
    let text = tail.drain_text(Duration::from_millis(500));
    assert_eq!(text.len(), MAX_STDERR_BYTES);
    assert!(text.ends_with("Authentication required\n"));

    // Relative executable with directory component is absolutized; bare name stays bare.
    assert_eq!(resolve_executable(Path::new("agy")), PathBuf::from("agy"));
    assert!(resolve_executable(Path::new("./bin/agy")).is_absolute());

    // Isolation creates restricted `folo-labeler` agent, strict deny-all `settings.json`, and safe CLI flags.
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

#[test]
fn subprocess_acp_script_labels_kills_descendants_and_handles_wedged_cancel() {
    let dir = TestDir::new("label-agy-subproc");
    let home = dir.join("user-home");
    let bin = write_fake_agy(dir.path(), false);
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 0xff, 0xd9]];
    let req = test_req(dir.path(), &imgs, Some("gemini-4-neon"), LabelEffort::Low, &schema, &cancel);

    let agy = Antigravity::new();
    agy.cli.remember(bin.to_str().unwrap(), "1.5.13");
    let info = agy.info();
    assert!(info.available && info.version.as_deref() == Some("1.5.13"));
    assert_eq!(info.models[0].id, "gemini-4-neon");
    assert_eq!(info.models[0].label, "Gemini 4 Neon");

    // 1. Subprocess spawns a long-sleeping helper holding pipes open and exits cleanly: must finish quickly and
    // terminate the helper process group / job.
    let t0 = Instant::now();
    let reply = run_acp_with_home(&bin, &req, Duration::from_secs(30), Some(&home)).unwrap();
    assert!(t0.elapsed() < Duration::from_secs(20), "call hung waiting on descendant: {:?}", t0.elapsed());
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[["car",1,2,30,40]]}]}"#);
    assert_eq!(reply.usage.calls, 1);
    assert_eq!(reply.usage.input_tokens, 512);
    assert_eq!(reply.usage.output_tokens, 32);

    #[cfg(unix)]
    if let Ok(pid_txt) = std::fs::read_to_string(dir.join("work/helper.pid")) {
        let pid = pid_txt.trim();
        let alive = std::process::Command::new("kill").args(["-0", pid]).status().is_ok_and(|s| s.success());
        assert!(!alive, "descendant helper {pid} survived clean ACP exit");
    }

    // 2. Wedged subprocess that ignores stdin on prompt still aborts promptly when cancelled.
    let wedged_bin = write_fake_agy(dir.path(), true);
    let cancel_wedged = Arc::new(AtomicBool::new(false));
    let trigger = cancel_wedged.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        trigger.store(true, Ordering::Relaxed);
    });
    let req_wedged = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel_wedged);
    let t1 = Instant::now();
    let err = run_acp_with_home(&wedged_bin, &req_wedged, Duration::from_secs(30), Some(&home)).unwrap_err();
    assert!(matches!(err, AgentError::Cancelled(_)), "{err:?}");
    assert!(t1.elapsed() < Duration::from_secs(15), "wedged cancel took {:?}", t1.elapsed());
}

/// Writes a cross-platform fake `agy` executable handling `--version`, `models` and `acp` (optionally wedging on
/// `session/prompt` or spawning a long-lived helper holding stdout/stderr open on exit).
fn write_fake_agy(dir: &Path, wedge: bool) -> PathBuf {
    let stem = if wedge { "agy_wedge" } else { "agy" };
    if cfg!(windows) {
        let ps1 = dir.join(format!("fake_{stem}.ps1"));
        let prompt_body = if wedge {
            "Start-Sleep -Seconds 60"
        } else {
            r#"Write-Output '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\"images\":[{\"image\":1,\"boxes\":[[\"car\",1,2,30,40]]}]}"}}}}'
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{ stopReason='end_turn'; usage=@{ inputTokens=512; outputTokens=32 } } } | ConvertTo-Json -Depth 5 -Compress)"#
        };
        let close_body = if wedge {
            "exit 0"
        } else {
            r#"Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{} } | ConvertTo-Json -Compress)
            Start-Process -NoNewWindow -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' | Out-Null
            exit 0"#
        };
        let ps_script = format!(
            r#"[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
while ($null -ne ($line = [Console]::In.ReadLine())) {{
    if ([string]::IsNullOrWhiteSpace($line)) {{ continue }}
    $msg = $line | ConvertFrom-Json
    switch ($msg.method) {{
        'initialize' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ protocolVersion=1; agentCapabilities=@{{ promptCapabilities=@{{ image=$true }}; sessionCapabilities=@{{ close=@{{}} }}; _meta=@{{ antigravity=@{{ systemPrompt=@{{ append=$true }} }} }} }} }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/new' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ sessionId='s1'; configOptions=@(@{{ id='model'; category='model'; currentValue='gemini-3.8-flash'; options=@(@{{ value='gemini-4-neon'; name='Gemini 4 Neon' }}) }}) }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/set_config_option' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{}} }} | ConvertTo-Json -Compress)
        }}
        'session/prompt' {{
            {prompt_body}
        }}
        'session/close' {{
            {close_body}
        }}
    }}
}}
"#
        );
        std::fs::write(&ps1, ps_script).unwrap();
        let cmd = dir.join(format!("{stem}.cmd"));
        let wrapper = format!(
            "@echo off\r\n\
             if \"%~1\"==\"--version\" goto ver\r\n\
             if \"%~1\"==\"models\" goto models\r\n\
             powershell -NoProfile -ExecutionPolicy Bypass -File \"{}\" %*\r\n\
             exit /b %ERRORLEVEL%\r\n\
             :ver\r\n\
             echo 1.5.13\r\n\
             exit /b 0\r\n\
             :models\r\n\
             echo gemini-4-neon-xhigh   Gemini 4 Neon (XHigh)\r\n\
             echo gemini-3.8-flash-high Gemini 3.8 Flash (High)\r\n\
             exit /b 0\r\n",
            ps1.display()
        );
        std::fs::write(&cmd, wrapper).unwrap();
        cmd
    } else {
        let sh = dir.join(stem);
        let prompt_body = if wedge {
            "sleep 60"
        } else {
            r#"printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\\"images\\":[{\\"image\\":1,\\"boxes\\":[[\\"car\\",1,2,30,40]]}]}"}}}}\n'
      printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn","usage":{"inputTokens":512,"outputTokens":32}}}\n' "$id""#
        };
        let close_body = if wedge {
            "exit 0"
        } else {
            r#"printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      sleep 60 &
      printf '%s\n' "$!" > helper.pid
      exit 0"#
        };
        let script = format!(
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "1.5.13"; exit 0; fi
if [ "$1" = "models" ]; then
  printf 'gemini-4-neon-xhigh   Gemini 4 Neon (XHigh)\n'
  printf 'gemini-3.8-flash-high Gemini 3.8 Flash (High)\n'
  exit 0
fi
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"protocolVersion":1,"agentCapabilities":{{"promptCapabilities":{{"image":true}},"sessionCapabilities":{{"close":{{}}}},"_meta":{{"antigravity":{{"systemPrompt":{{"append":true}}}}}}}}}}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"sessionId":"s1","configOptions":[{{"id":"model","category":"model","currentValue":"gemini-3.8-flash","options":[{{"value":"gemini-4-neon","name":"Gemini 4 Neon"}}]}}]}}}}\n' "$id"
      ;;
    *'"method":"session/set_config_option"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{}}}}\n' "$id"
      ;;
    *'"method":"session/prompt"'*)
      {prompt_body}
      ;;
    *'"method":"session/close"'*)
      {close_body}
      ;;
  esac
done
"#
        );
        std::fs::write(&sh, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sh, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        sh
    }
}
