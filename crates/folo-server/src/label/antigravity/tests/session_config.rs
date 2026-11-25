//! ACP v1 session invariant tests for legacy model switching, default-model pinning, variant resolution,
//! and pre-dispatch stdout flood bounding.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Duration;

use folo_api::LabelEffort;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::{init_ok, test_req};
use crate::label::agent::AgentError;
use crate::label::antigravity::acp::{drive_acp, drive_acp_with_grace};

#[test]
fn legacy_model_switch_invalidates_effort_default_model_pins_base_resolves_variant_and_drain_bounds_flood() {
    let dir = TestDir::new("label-agy-r4");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);

    // 1. Legacy `session/set_model` returning `{}` invalidates Flash's stale `thought_level: low`, so FOLO must
    // explicitly re-set `thought_level: low` and verify the refreshed snapshot before prompting.
    let low_neon = test_req(dir.path(), &[], Some("gemini-4-neon"), LabelEffort::Low, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut saw_effort_set = false;
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
                            "models": { "currentModelId": "gemini-3.8-flash", "availableModels": [{ "modelId": "gemini-3.8-flash" }, { "modelId": "gemini-4-neon" }] },
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }, { "value": "high" }] }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_model") => {
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string());
            }
            Some("session/set_config_option") => {
                saw_effort_set = true;
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "high", "options": [{ "value": "low" }, { "value": "high" }] }]
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
        drive_acp(&low_neon, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Failed(f)) if f.message.contains("selected effort `high` instead of requested `low`")
    ));
    assert!(saw_effort_set, "legacy model switch must re-confirm effort instead of reusing stale low");

    // 2. Default-model run pins reported default (`gemini-4-neon`), so mid-prompt drift to Flash fails closed.
    let default_req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);
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
                                { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
                            ]
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
                        "params": {
                            "update": {
                                "sessionUpdate": "config_option_update",
                                "configOptions": [
                                    { "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                    { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
                                ]
                            }
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let out = drive_acp_with_grace(
        &default_req,
        &work,
        Duration::from_secs(5),
        Duration::from_millis(200),
        &mut send,
        &out_rx,
        &String::new,
    );
    assert!(out.kill_immediately);
    assert!(matches!(
        out.result,
        Err(AgentError::Failed(f)) if f.message.contains("selected model `gemini-3.8-flash` instead of requested `gemini-4-neon`")
    ));

    // 3. Collapsed catalog base ID (`gemini-4-neon` + `High`) resolves to advertised suffixed variant
    // (`gemini-4-neon-high`) when only suffixed models are advertised and no `thought_level` selector exists.
    let high_base = test_req(dir.path(), &[], Some("gemini-4-neon"), LabelEffort::High, &schema, &cancel);
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut selected_val = None;
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
                                "id": "model",
                                "category": "model",
                                "currentValue": "gemini-4-neon-low",
                                "options": [{ "value": "gemini-4-neon-low" }, { "value": "gemini-4-neon-high" }]
                            }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                selected_val = msg["params"]["value"].as_str().map(str::to_string);
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "configOptions": [{
                                "id": "model",
                                "category": "model",
                                "currentValue": selected_val.as_deref().unwrap_or(""),
                                "options": [{ "value": "gemini-4-neon-low" }, { "value": "gemini-4-neon-high" }]
                            }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => {
                let _ = out_tx.send(
                    json!({ "jsonrpc": "2.0", "method": "session/update", "params": { "update": { "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "{\"images\":[]}" } } } })
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
    assert_eq!(
        drive_acp(&high_base, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap().text,
        r#"{"images":[]}"#
    );
    assert_eq!(selected_val.as_deref(), Some("gemini-4-neon-high"));

    // 4. Flooded stdout between RPCs hits `drain_pending` budget and fails closed without dispatching `session/prompt`.
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
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                );
                for _ in 0..300 {
                    let _ = out_tx.send("noise-line".into());
                }
            }
            Some("session/prompt") => saw_prompt = true,
            _ => {}
        }
        true
    };
    let out = drive_acp_with_grace(
        &default_req,
        &work,
        Duration::from_secs(5),
        Duration::from_millis(200),
        &mut send,
        &out_rx,
        &String::new,
    );
    assert!(out.kill_immediately && !saw_prompt);
    assert!(matches!(out.result, Err(AgentError::Failed(f)) if f.message.contains("flooded stdout")));

    // 5. Legacy `session/set_model` with an effort-only `config_option_update` (either BEFORE or IMMEDIATELY
    // AFTER `{}` ack, without confirming `model == gemini-4-neon`) invalidates effort and forces re-setting.
    for after_ack in [false, true] {
        let (out_tx, out_rx) = mpsc::channel::<String>();
        let mut saw_effort_set_unrelated = false;
        let mut send = |msg: &Value| -> bool {
            let id = msg.get("id").and_then(Value::as_u64);
            match msg["method"].as_str() {
                Some("initialize") => {
                    let _ = out_tx.send(init_ok(id, true, false, false));
                }
                Some("session/new") => {
                    let _ = out_tx.send(json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": {
                            "sessionId": "s1",
                            "models": { "currentModelId": "gemini-3.8-flash", "availableModels": [{ "modelId": "gemini-3.8-flash" }, { "modelId": "gemini-4-neon" }] },
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }, { "value": "high" }] }]
                        }
                    }).to_string());
                }
                Some("session/set_model") => {
                    let upd = json!({
                        "jsonrpc": "2.0", "method": "session/update",
                        "params": { "update": { "sessionUpdate": "config_option_update", "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }, { "value": "high" }] }] } }
                    }).to_string();
                    let ack = json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string();
                    if after_ack {
                        let _ = out_tx.send(ack);
                        let _ = out_tx.send(upd);
                    } else {
                        let _ = out_tx.send(upd);
                        let _ = out_tx.send(ack);
                    }
                }
                Some("session/set_config_option") => {
                    saw_effort_set_unrelated = true;
                    let _ = out_tx.send(json!({
                        "jsonrpc": "2.0", "id": id,
                        "result": { "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "high", "options": [{ "value": "low" }, { "value": "high" }] }] }
                    }).to_string());
                }
                _ => {}
            }
            true
        };
        assert!(matches!(
            drive_acp(&low_neon, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
            Err(AgentError::Failed(f)) if f.message.contains("selected effort `high` instead of requested `low`")
        ));
        assert!(saw_effort_set_unrelated, "after_ack={after_ack}");
    }

    // 6. Default-model run where `model` is first reported in a pre-dispatch `config_option_update` pins Neon and
    // rejects a mid-prompt switch to Flash.
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
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "config_option_update",
                                "configOptions": [
                                    { "id": "model", "category": "model", "currentValue": "gemini-4-neon", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                    { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
                                ]
                            }
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
                        "params": {
                            "update": {
                                "sessionUpdate": "config_option_update",
                                "configOptions": [
                                    { "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                    { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
                                ]
                            }
                        }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let out = drive_acp_with_grace(
        &default_req,
        &work,
        Duration::from_secs(5),
        Duration::from_millis(200),
        &mut send,
        &out_rx,
        &String::new,
    );
    assert!(out.kill_immediately);
    assert!(matches!(
        out.result,
        Err(AgentError::Failed(f)) if f.message.contains("selected model `gemini-3.8-flash` instead of requested `gemini-4-neon`")
    ));

    // 7. Default-model run pins Neon reported in `session/new` even before mode negotiation, so switching mode
    // from `auto` to `default` cannot silently swap the model to Flash.
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
                                { "id": "mode", "category": "mode", "currentValue": "auto", "options": [{ "value": "auto" }, { "value": "default" }] },
                                { "id": "model", "category": "model", "currentValue": "gemini-4-neon", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
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
                                { "id": "mode", "category": "mode", "currentValue": "default", "options": [{ "value": "auto" }, { "value": "default" }] },
                                { "id": "model", "category": "model", "currentValue": "gemini-3.8-flash", "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-3.8-flash" }] },
                                { "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }
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
        drive_acp(&default_req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new),
        Err(AgentError::Failed(f)) if f.message.contains("selected model `gemini-3.8-flash` instead of requested `gemini-4-neon`")
    ));

    // 8. Queued `config_option_update` right after `session/new` advertising `[gemini-4-neon, gemini-4-neon-high]`
    // with no effort selector is drained before model resolution so `gemini-4-neon` at High picks `gemini-4-neon-high`.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut picked = None;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(init_ok(id, true, false, false));
            }
            Some("session/new") => {
                let _ = out_tx
                    .send(json!({ "jsonrpc": "2.0", "id": id, "result": { "sessionId": "s1" } }).to_string());
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "config_option_update",
                                "configOptions": [{
                                    "id": "model",
                                    "category": "model",
                                    "currentValue": "gemini-4-neon",
                                    "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-4-neon-high" }]
                                }]
                            }
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/set_config_option") => {
                picked = msg["params"]["value"].as_str().map(str::to_string);
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "configOptions": [{
                                "id": "model",
                                "category": "model",
                                "currentValue": picked.as_deref().unwrap_or(""),
                                "options": [{ "value": "gemini-4-neon" }, { "value": "gemini-4-neon-high" }]
                            }]
                        }
                    })
                    .to_string(),
                );
            }
            Some("session/prompt") => {
                let _ = out_tx.send(
                    json!({ "jsonrpc": "2.0", "method": "session/update", "params": { "update": { "sessionUpdate": "agent_message_chunk", "content": { "type": "text", "text": "{\"images\":[]}" } } } })
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
    assert_eq!(
        drive_acp(&high_base, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap().text,
        r#"{"images":[]}"#
    );
    assert_eq!(picked.as_deref(), Some("gemini-4-neon-high"));
}
