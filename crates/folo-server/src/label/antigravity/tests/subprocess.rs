//! Subprocess lifecycle and descendant-cleanup tests for `agy acp` using cross-platform fake executables.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use folo_api::LabelEffort;
use folo_test_support::TestDir;
use serde_json::json;

use super::test_req;
use crate::label::agent::{Agent, AgentError, cancel_when};
use crate::label::antigravity::Antigravity;
use crate::label::antigravity::acp::run_acp_checked;

#[derive(Copy, Clone)]
enum FakeMode {
    CleanWithHelper,
    EarlyExitWithHelper,
    WedgedOnPrompt,
}

#[test]
fn subprocess_acp_handles_clean_exit_early_exit_with_helper_and_wedged_cancel() {
    let dir = TestDir::new("label-agy-subproc");
    let home = dir.join("user-home");
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let imgs = vec![vec![0xff, 0xd8, 0xff, 0xd9]];

    // 1. Clean exit while a sleeping helper inherits stdio: returns reply promptly and kills the helper
    // (proven deterministically by `pipes_closed == true`: dead processes close inherited fds before reaping).
    let bin = write_fake_agy(&dir.join("clean"), FakeMode::CleanWithHelper);
    let agy = Antigravity::new();
    agy.cli.remember(bin.to_str().unwrap(), "1.5.13");
    let info = agy.info();
    assert!(info.available && info.version.as_deref() == Some("1.5.13"));
    assert_eq!(
        (info.models[0].id.as_str(), info.models[0].label.as_str()),
        ("gemini-4-neon", "Gemini 4 Neon")
    );

    let clean_scratch = dir.join("s-clean");
    let req = test_req(&clean_scratch, &imgs, Some("gemini-4-neon"), LabelEffort::Low, &schema, &cancel);
    let t0 = Instant::now();
    let (res, pipes_closed) = run_acp_checked(&bin, &req, Duration::from_secs(30), Some(&home));
    let reply = res.unwrap();
    assert!(pipes_closed, "helper holding stdout/stderr survived clean ACP exit");
    assert!(t0.elapsed() < Duration::from_secs(20), "call hung waiting on descendant: {:?}", t0.elapsed());
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[["car",1,2,30,40]]}]}"#);
    assert_eq!((reply.usage.calls, reply.usage.input_tokens, reply.usage.output_tokens), (1, 512, 32));

    // 2. Parent exits BEFORE responding to `session/prompt` while a sleeping helper inherits stdout: must detect
    // parent exit promptly (never stall until the 30s timeout), kill the helper, and close all pipes.
    let early_bin = write_fake_agy(&dir.join("early"), FakeMode::EarlyExitWithHelper);
    let early_scratch = dir.join("s-early");
    let early_req = test_req(&early_scratch, &[], None, LabelEffort::Low, &schema, &cancel);
    let t1 = Instant::now();
    let (res, pipes_closed) = run_acp_checked(&early_bin, &early_req, Duration::from_secs(30), Some(&home));
    assert!(matches!(res, Err(AgentError::Failed(ref f)) if f.message.contains("early crash")), "{res:?}");
    assert!(pipes_closed, "helper survived early parent exit");
    assert!(t1.elapsed() < Duration::from_secs(15), "early exit with helper stalled: {:?}", t1.elapsed());

    // 3. Wedged subprocess on `session/prompt` signaled deterministically by marker file (no fixed sleep).
    let wedged_bin = write_fake_agy(&dir.join("wedged"), FakeMode::WedgedOnPrompt);
    let wedged_scratch = dir.join("s-wedged");
    let cancel_wedged = cancel_when(wedged_scratch.join("work/wedged"));
    let wedged_req = test_req(&wedged_scratch, &[], None, LabelEffort::Low, &schema, &cancel_wedged);
    let t2 = Instant::now();
    let (res, pipes_closed) = run_acp_checked(&wedged_bin, &wedged_req, Duration::from_secs(30), Some(&home));
    assert!(matches!(res, Err(AgentError::Cancelled(_))), "{res:?}");
    assert!(pipes_closed, "wedged child survived cancel");
    assert!(t2.elapsed() < Duration::from_secs(15), "wedged cancel took {:?}", t2.elapsed());
}

/// Writes a cross-platform fake `agy` executable handling `--version`, `models` and `acp`.
fn write_fake_agy(dir: &Path, mode: FakeMode) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    if cfg!(windows) {
        let ps1 = dir.join("fake_agy.ps1");
        let prompt_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"Write-Output '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\"images\":[{\"image\":1,\"boxes\":[[\"car\",1,2,30,40]]}]}"}}}}'
            Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{ stopReason='end_turn'; usage=@{ inputTokens=512; outputTokens=32 } } } | ConvertTo-Json -Depth 5 -Compress)"#
            }
            FakeMode::EarlyExitWithHelper => {
                r#"[Console]::Error.WriteLine('early crash')
            Start-Process -NoNewWindow -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' | Out-Null
            exit 1"#
            }
            FakeMode::WedgedOnPrompt => {
                r#"[IO.File]::WriteAllText('wedged', '1')
            Start-Sleep -Seconds 60"#
            }
        };
        let close_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{} } | ConvertTo-Json -Compress)
            Start-Process -NoNewWindow -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' | Out-Null
            exit 0"#
            }
            FakeMode::EarlyExitWithHelper | FakeMode::WedgedOnPrompt => "exit 0",
        };
        let ps_script = format!(
            r#"[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$curModel = 'gemini-3.8-flash'
while ($null -ne ($line = [Console]::In.ReadLine())) {{
    if ([string]::IsNullOrWhiteSpace($line)) {{ continue }}
    $msg = $line | ConvertFrom-Json
    switch ($msg.method) {{
        'initialize' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ protocolVersion=1; agentCapabilities=@{{ promptCapabilities=@{{ image=$true }}; sessionCapabilities=@{{ close=@{{}} }}; _meta=@{{ antigravity=@{{ systemPrompt=@{{ append=$true }} }} }} }} }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/new' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ sessionId='s1'; configOptions=@(@{{ id='model'; category='model'; currentValue=$curModel; options=@(@{{ value='gemini-4-neon'; name='Gemini 4 Neon' }}) }}) }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/set_config_option' {{
            if ($msg.params.configId -eq 'model') {{ $curModel = $msg.params.value }}
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ configOptions=@(@{{ id='model'; category='model'; currentValue=$curModel; options=@(@{{ value='gemini-4-neon'; name='Gemini 4 Neon' }}) }}) }} }} | ConvertTo-Json -Depth 10 -Compress)
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
        let prompt_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"{\\"images\\":[{\\"image\\":1,\\"boxes\\":[[\\"car\\",1,2,30,40]]}]}"}}}}\n'
      printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn","usage":{"inputTokens":512,"outputTokens":32}}}\n' "$id""#
            }
            FakeMode::EarlyExitWithHelper => {
                r#"printf 'early crash\n' >&2
      sleep 60 &
      exit 1"#
            }
            FakeMode::WedgedOnPrompt => {
                r#"touch wedged
      sleep 60"#
            }
        };
        let close_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      sleep 60 &
      exit 0"#
            }
            FakeMode::EarlyExitWithHelper | FakeMode::WedgedOnPrompt => "exit 0",
        };
        let script = format!(
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then echo "1.5.13"; exit 0; fi
if [ "$1" = "models" ]; then
  printf 'gemini-4-neon-xhigh   Gemini 4 Neon (XHigh)\n'
  printf 'gemini-3.8-flash-high Gemini 3.8 Flash (High)\n'
  exit 0
fi
cur_model="gemini-3.8-flash"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"protocolVersion":1,"agentCapabilities":{{"promptCapabilities":{{"image":true}},"sessionCapabilities":{{"close":{{}}}},"_meta":{{"antigravity":{{"systemPrompt":{{"append":true}}}}}}}}}}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"sessionId":"s1","configOptions":[{{"id":"model","category":"model","currentValue":"%s","options":[{{"value":"gemini-4-neon","name":"Gemini 4 Neon"}}]}}]}}}}\n' "$id" "$cur_model"
      ;;
    *'"method":"session/set_config_option"'*)
      case "$line" in *'"value":"gemini-4-neon"'*) cur_model="gemini-4-neon" ;; esac
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"configOptions":[{{"id":"model","category":"model","currentValue":"%s","options":[{{"value":"gemini-4-neon","name":"Gemini 4 Neon"}}]}}]}}}}\n' "$id" "$cur_model"
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
