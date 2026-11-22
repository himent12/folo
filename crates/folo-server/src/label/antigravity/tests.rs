//! Unit and fake-subprocess tests for the Antigravity ACP labeling driver.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Duration;

use base64::Engine;
use folo_api::{LabelEffort, LabelModel, LabelUsage};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::acp::{
    drive_acp, find_effort_option, find_model_option, parse_config_options, resolve_effort_value,
    resolve_model_and_effort, run_acp,
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

#[test]
fn agy_models_output_collapses_effort_suffixes_and_skips_orchestrator() {
    let out = "\n\
        gemini-4-neon-xhigh          Gemini 4 Neon (XHigh)\n\
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
    // Error lines or empty output produce no bogus models (info() then falls back to curated defaults).
    assert!(
        parse_models("Error: Authentication required\nPlease sign in.\nRun `agy` to authenticate.\n")
            .is_empty()
    );
    assert!(fallback_models().iter().any(|m| m.id == "gemini-4-neon" && m.label == "Gemini 4 Neon"));
}

#[test]
fn model_and_effort_resolution_supports_base_ids_xhigh_slugs_and_grouped_options() {
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

    // Plain base model keeps the requested effort.
    assert_eq!(
        resolve_model_and_effort(Some("gemini-4-neon"), LabelEffort::Low, Some(model_opt)),
        (Some("gemini-4-neon".into()), "low".into())
    );
    // Effort-suffixed slug selects the base model and overrides the effort.
    assert_eq!(
        resolve_model_and_effort(Some("gemini-4-neon-xhigh"), LabelEffort::Low, Some(model_opt)),
        (Some("gemini-4-neon".into()), "xhigh".into())
    );
    // Unknown future model passes straight through.
    assert_eq!(
        resolve_model_and_effort(Some("gemini-5-alpha"), LabelEffort::Medium, Some(model_opt)),
        (Some("gemini-5-alpha".into()), "medium".into())
    );
    // When a model starts at `medium`, default `low` falls back cleanly instead of failing.
    assert_eq!(resolve_effort_value("low", effort_opt).as_deref(), Some("medium"));
    assert_eq!(resolve_effort_value("xhigh", effort_opt).as_deref(), Some("xhigh"));
    assert_eq!(resolve_effort_value("max", effort_opt).as_deref(), Some("xhigh"));
}

#[test]
fn acp_round_trip_sends_inline_jpegs_denies_tools_and_collects_reply_and_usage() {
    let dir = TestDir::new("label-agy-acp");
    let work = dir.join("work");
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
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": 1,
                            "agentCapabilities": {
                                "promptCapabilities": { "image": true },
                                "sessionCapabilities": { "close": {} },
                                "_meta": { "antigravity": { "systemPrompt": { "append": true } } }
                            }
                        }
                    })
                    .to_string(),
                );
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
                                    "id": "model",
                                    "category": "model",
                                    "currentValue": "gemini-3.8-flash",
                                    "options": [
                                        { "value": "gemini-4-neon", "name": "Gemini 4 Neon" },
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
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string());
            }
            Some("session/prompt") => {
                // Simulate a tool permission request, an unsupported client method, thought text (ignored),
                // streamed reply chunks, usage update, and completion with detailed usage.
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
                    json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "sess-42",
                            "update": {
                                "sessionUpdate": "usage_update",
                                "used": 1400,
                                "size": 1000000,
                                "cost": { "amount": 0.0042, "currency": "USD" }
                            }
                        }
                    }),
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "stopReason": "end_turn",
                            "usage": {
                                "inputTokens": 1200,
                                "cachedReadTokens": 300,
                                "outputTokens": 85,
                                "thoughtTokens": 40,
                                "totalTokens": 1325
                            }
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

    // Model and thought_level were configured from `gemini-4-neon-xhigh`.
    let configs: Vec<(&str, &str)> = sent
        .iter()
        .filter(|m| m["method"] == "session/set_config_option")
        .map(|m| (m["params"]["configId"].as_str().unwrap(), m["params"]["value"].as_str().unwrap()))
        .collect();
    assert_eq!(configs, vec![("model", "gemini-4-neon"), ("thought_level", "xhigh")]);

    // Inline base64 JPEGs were sent in `session/prompt`, and system prompt went into `session/new` `_meta`.
    let new_msg = sent.iter().find(|m| m["method"] == "session/new").unwrap();
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

    // Permission request was rejected and unsupported tool method answered -32601.
    let perm_reply = sent.iter().find(|m| m.get("id") == Some(&json!(99))).unwrap();
    assert_eq!(perm_reply["result"]["outcome"], json!({ "outcome": "selected", "optionId": "deny" }));
    let method_reply = sent.iter().find(|m| m.get("id") == Some(&json!(100))).unwrap();
    assert_eq!(method_reply["error"]["code"], -32601);
    assert!(sent.iter().any(|m| m["method"] == "session/close"));
}

#[test]
fn auth_error_cancel_timeout_and_hints_are_reported_cleanly() {
    let dir = TestDir::new("label-agy-errors");
    let work = dir.join("work");
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);

    // Signed-out `session/new` returns -32000 Authentication required immediately.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string());
            }
            Some("session/new") => {
                let _ = out_tx.send(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32000, "message": "Authentication required" }
                    })
                    .to_string(),
                );
            }
            _ => {}
        }
        true
    };
    let err = drive_acp(&req, &work, Duration::from_secs(5), &mut send, &out_rx, &String::new).unwrap_err();
    assert_eq!(err, AgentError::failed("Authentication required"));

    // Hints cover sign-in, Google RESOURCE_EXHAUSTED limits, passing faults, bad models and silent exits.
    let agy = Antigravity::new();
    assert!(agy.hint("Authentication required").unwrap().contains("agy acp --login"));
    assert!(
        agy.hint("RESOURCE_EXHAUSTED: You exceeded your current quota.").unwrap().contains("usage limit")
    );
    assert!(agy.hint("RESOURCE_EXHAUSTED: Try again in 15s.").unwrap().contains("provider's rate limit"));
    assert_eq!(agy.hint("API Error: 503 Service Unavailable").as_deref(), Some(crate::label::agent::BUSY));
    assert!(agy.hint("unknown model foo").unwrap().contains("pick another model"));
    assert!(agy.hint("`agy` printed no reply").unwrap().contains("`agy -p hi`"));

    // Timeout preserves usage reported before the hang and sends `session/cancel`.
    let (out_tx, out_rx) = mpsc::channel::<String>();
    let mut saw_cancel = false;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(json!({ "jsonrpc": "2.0", "id": id, "result": {} }).to_string());
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
                        "params": { "update": { "sessionUpdate": "usage_update", "used": 777 } }
                    })
                    .to_string(),
                );
            }
            Some("session/cancel") => saw_cancel = true,
            _ => {}
        }
        true
    };
    let err =
        drive_acp(&req, &work, Duration::from_millis(80), &mut send, &out_rx, &String::new).unwrap_err();
    assert_eq!(err, AgentError::Timeout(LabelUsage { calls: 1, input_tokens: 777, ..LabelUsage::default() }));
    assert!(saw_cancel);
}

#[test]
fn subprocess_acp_script_labels_and_exits_cleanly() {
    let dir = TestDir::new("label-agy-subproc");
    let bin = write_fake_agy(dir.path());
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 0xff, 0xd9]];
    let req = test_req(dir.path(), &imgs, Some("gemini-4-neon"), LabelEffort::Low, &schema, &cancel);

    // Probe version + models through the fake executable.
    let agy = Antigravity::new();
    agy.cli.remember(bin.to_str().unwrap(), "1.5.13");
    let info = agy.info();
    assert!(info.available && info.version.as_deref() == Some("1.5.13"));
    assert_eq!(info.models[0].id, "gemini-4-neon");
    assert_eq!(info.models[0].label, "Gemini 4 Neon");

    // Drive the real subprocess over stdio pipes and verify Tree cleanup.
    let reply = run_acp(&bin, &req, Duration::from_secs(30)).unwrap();
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[["car",1,2,30,40]]}]}"#);
    assert_eq!(reply.usage.calls, 1);
    assert_eq!(reply.usage.input_tokens, 512);
}

/// Writes a tiny cross-platform fake `agy` executable handling `--version`, `models` and `acp`.
fn write_fake_agy(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        let ps1 = dir.join("fake_agy.ps1");
        let ps_script = r#"[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
while ($null -ne ($line = [Console]::In.ReadLine())) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $msg = $line | ConvertFrom-Json
    switch ($msg.method) {
        'initialize' {
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{ protocolVersion=1; agentCapabilities=@{ promptCapabilities=@{ image=$true }; sessionCapabilities=@{ close=@{} }; _meta=@{ antigravity=@{ systemPrompt=@{ append=$true } } } } } } | ConvertTo-Json -Depth 10 -Compress)
        }
        'session/new' {
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{ sessionId='s1'; configOptions=@(@{ id='model'; category='model'; currentValue='gemini-3.8-flash'; options=@(@{ value='gemini-4-neon'; name='Gemini 4 Neon' }) }) } } | ConvertTo-Json -Depth 10 -Compress)
        }
        'session/set_config_option' {
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{} } | ConvertTo-Json -Compress)
        }
        'session/prompt' {
            Write-Output '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\"images\":[{\"image\":1,\"boxes\":[[\"car\",1,2,30,40]]}]}"}}}}'
            Write-Output '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"usage_update","used":512,"size":1000000}}}'
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{ stopReason='end_turn' } } | ConvertTo-Json -Compress)
        }
        'session/close' {
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{} } | ConvertTo-Json -Compress)
            exit 0
        }
    }
}
"#;
        std::fs::write(&ps1, ps_script).unwrap();
        let cmd = dir.join("agy.cmd");
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
        let sh = dir.join("agy");
        let script = r#"#!/bin/sh
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
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1,"agentCapabilities":{"promptCapabilities":{"image":true},"sessionCapabilities":{"close":{}},"_meta":{"antigravity":{"systemPrompt":{"append":true}}}}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"s1","configOptions":[{"id":"model","category":"model","currentValue":"gemini-3.8-flash","options":[{"value":"gemini-4-neon","name":"Gemini 4 Neon"}]}]}}\n' "$id"
      ;;
    *'"method":"session/set_config_option"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
    *'"method":"session/prompt"'*)
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\\"images\\":[{\\"image\\":1,\\"boxes\\":[[\\"car\\",1,2,30,40]]}]}"}}}}\n'
      printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"usage_update","used":512,"size":1000000}}}\n'
      printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}\n' "$id"
      ;;
    *'"method":"session/close"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      break
      ;;
  esac
done
"#;
        std::fs::write(&sh, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sh, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        sh
    }
}
