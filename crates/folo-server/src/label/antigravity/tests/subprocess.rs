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
use crate::label::antigravity::process::run_acp_checked;

const HELPER_STDOUT_MARKER: &str = "helper-stdout-acquired";
const HELPER_STDERR_MARKER: &str = "helper-stderr-acquired";

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

    // 1. Clean exit while a sleeping helper has ALREADY written directly onto BOTH inherited stdio pipes (proven
    // by `saw_stdout_marker` and `stderr.contains(HELPER_STDERR_MARKER)`): returns reply promptly and kills the
    // helper (proven deterministically by `pipes_closed == true`: dead processes close inherited fds before reaping).
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
    let check = run_acp_checked(&bin, &req, Duration::from_secs(30), Some(&home), Some(HELPER_STDOUT_MARKER));
    let reply = check.result.unwrap();
    assert!(check.saw_stdout_marker, "helper did not inherit captured stdout pipe before parent exit");
    assert!(
        check.stderr.contains(HELPER_STDERR_MARKER),
        "helper did not inherit captured stderr pipe before parent exit: {:?}",
        check.stderr
    );
    assert!(check.pipes_closed, "helper holding stdout/stderr survived clean ACP exit");
    assert!(t0.elapsed() < Duration::from_secs(20), "call hung waiting on descendant: {:?}", t0.elapsed());
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[["car",1,2,30,40]]}]}"#);
    assert_eq!((reply.usage.calls, reply.usage.input_tokens, reply.usage.output_tokens), (1, 512, 32));

    // 2. Parent exits BEFORE responding to `session/prompt` after its helper has written onto BOTH captured stdio
    // pipes: must detect parent exit promptly (never stall until the 30s timeout), kill the helper, and close pipes.
    let early_bin = write_fake_agy(&dir.join("early"), FakeMode::EarlyExitWithHelper);
    let early_scratch = dir.join("s-early");
    let early_req = test_req(&early_scratch, &[], None, LabelEffort::Low, &schema, &cancel);
    let t1 = Instant::now();
    let check = run_acp_checked(
        &early_bin,
        &early_req,
        Duration::from_secs(30),
        Some(&home),
        Some(HELPER_STDOUT_MARKER),
    );
    assert!(
        matches!(check.result, Err(AgentError::Failed(ref f)) if f.message.contains("early crash")),
        "{:?}",
        check.result
    );
    assert!(check.saw_stdout_marker, "helper did not inherit captured stdout pipe before crash");
    assert!(
        check.stderr.contains(HELPER_STDERR_MARKER),
        "helper did not inherit captured stderr pipe before crash: {:?}",
        check.stderr
    );
    assert!(check.pipes_closed, "helper survived early parent exit");
    assert!(t1.elapsed() < Duration::from_secs(15), "early exit with helper stalled: {:?}", t1.elapsed());

    // 3. Wedged subprocess on `session/prompt` signaled deterministically by marker file (no fixed sleep).
    let wedged_bin = write_fake_agy(&dir.join("wedged"), FakeMode::WedgedOnPrompt);
    let wedged_scratch = dir.join("s-wedged");
    let cancel_wedged = cancel_when(wedged_scratch.join("work/wedged"));
    let wedged_req = test_req(&wedged_scratch, &[], None, LabelEffort::Low, &schema, &cancel_wedged);
    let t2 = Instant::now();
    let check = run_acp_checked(&wedged_bin, &wedged_req, Duration::from_secs(30), Some(&home), None);
    assert!(matches!(check.result, Err(AgentError::Cancelled(_))), "{:?}", check.result);
    assert!(check.pipes_closed, "wedged child survived cancel");
    assert!(t2.elapsed() < Duration::from_secs(15), "wedged cancel took {:?}", t2.elapsed());
}

/// Writes a cross-platform fake `agy` executable handling `--version`, `models` and `acp`.
fn write_fake_agy(dir: &Path, mode: FakeMode) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    if cfg!(windows) {
        let ps1 = dir.join("fake_agy.ps1");
        let spawn_helper_ps = format!(
            r#"$evName = 'Local\folo-helper-' + [Guid]::NewGuid().ToString('N')
            $ev = [Threading.EventWaitHandle]::new($false, [Threading.EventResetMode]::ManualReset, $evName)
            $psi = [Diagnostics.ProcessStartInfo]::new('powershell.exe', "-NoProfile -Command `"[Console]::Out.WriteLine('{HELPER_STDOUT_MARKER}'); [Console]::Out.Flush(); [Console]::Error.WriteLine('{HELPER_STDERR_MARKER}'); [Console]::Error.Flush(); [void][Threading.EventWaitHandle]::OpenExisting('$evName').Set(); Start-Sleep -Seconds 60`"")
            $psi.UseShellExecute = $false
            [Diagnostics.Process]::Start($psi) | Out-Null
            $ready = $ev.WaitOne(15000)
            $ev.Dispose()
            if (-not $ready) {{ [Console]::Error.WriteLine('helper failed to signal ready'); [Console]::Error.Flush(); exit 2 }}"#
        );
        let prompt_body = match mode {
            FakeMode::CleanWithHelper => format!(
                r#"{spawn_helper_ps}
            Write-Output '{{"jsonrpc":"2.0","method":"session/update","params":{{"sessionId":"s1","update":{{"sessionUpdate":"agent_message_chunk","content":{{"type":"text","text":"{{\"images\":[{{\"image\":1,\"boxes\":[[\"car\",1,2,30,40]]}}]}}"}}}}}}}}'
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ stopReason='end_turn'; usage=@{{ inputTokens=512; outputTokens=32 }} }} }} | ConvertTo-Json -Depth 5 -Compress)"#
            ),
            FakeMode::EarlyExitWithHelper => format!(
                r#"{spawn_helper_ps}
            [Console]::Error.WriteLine('early crash')
            [Console]::Error.Flush()
            exit 1"#
            ),
            FakeMode::WedgedOnPrompt => r#"[IO.File]::WriteAllText('wedged', '1')
            Start-Sleep -Seconds 60"#
                .to_string(),
        };
        let close_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"Write-Output (@{ jsonrpc='2.0'; id=$msg.id; result=@{} } | ConvertTo-Json -Compress)
            exit 0"#
            }
            FakeMode::EarlyExitWithHelper | FakeMode::WedgedOnPrompt => "exit 0",
        };
        let ps_script = format!(
            r#"[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$curModel = 'gemini-3.8-flash'
$curThought = 'high'
function ConfigOptions {{
    @(
        @{{ id='model'; category='model'; currentValue=$curModel; options=@(@{{ value='gemini-4-neon'; name='Gemini 4 Neon' }}, @{{ value='gemini-3.8-flash'; name='Gemini 3.8 Flash' }}) }},
        @{{ id='thought_level'; category='thought_level'; currentValue=$curThought; options=@(@{{ value='low'; name='Low' }}, @{{ value='high'; name='High' }}) }}
    )
}}
while ($null -ne ($line = [Console]::In.ReadLine())) {{
    if ([string]::IsNullOrWhiteSpace($line)) {{ continue }}
    $msg = $line | ConvertFrom-Json
    switch ($msg.method) {{
        'initialize' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ protocolVersion=1; agentCapabilities=@{{ promptCapabilities=@{{ image=$true }}; sessionCapabilities=@{{ close=@{{}} }}; _meta=@{{ antigravity=@{{ systemPrompt=@{{ append=$true }} }} }} }} }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/new' {{
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ sessionId='s1'; configOptions=(ConfigOptions) }} }} | ConvertTo-Json -Depth 10 -Compress)
        }}
        'session/set_config_option' {{
            if ($msg.params.configId -eq 'model') {{ $curModel = $msg.params.value }}
            elseif ($msg.params.configId -eq 'thought_level') {{ $curThought = $msg.params.value }}
            Write-Output (@{{ jsonrpc='2.0'; id=$msg.id; result=@{{ configOptions=(ConfigOptions) }} }} | ConvertTo-Json -Depth 10 -Compress)
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
        let spawn_helper_sh = format!(
            r#"fifo="$PWD/helper_ready.fifo"
      rm -f "$fifo"
      mkfifo "$fifo"
      ( printf '{HELPER_STDOUT_MARKER}\n'; printf '{HELPER_STDERR_MARKER}\n' >&2; printf 'ready\n' > "$fifo"; exec sleep 60 ) &
      read -r _ < "$fifo""#
        );
        let prompt_body = match mode {
            FakeMode::CleanWithHelper => format!(
                r#"{spawn_helper_sh}
      printf '{{"jsonrpc":"2.0","method":"session/update","params":{{"sessionId":"s1","update":{{"sessionUpdate":"agent_message_chunk","content":{{"type":"text","text":"{{\\"images\\":[{{\\"image\\":1,\\"boxes\\":[[\\"car\\",1,2,30,40]]}}]}}"}}}}}}}}\n'
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"stopReason":"end_turn","usage":{{"inputTokens":512,"outputTokens":32}}}}}}\n' "$id""#
            ),
            FakeMode::EarlyExitWithHelper => format!(
                r#"{spawn_helper_sh}
      printf 'early crash\n' >&2
      exit 1"#
            ),
            FakeMode::WedgedOnPrompt => r#"touch wedged
      sleep 60"#
                .to_string(),
        };
        let close_body = match mode {
            FakeMode::CleanWithHelper => {
                r#"printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
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
cur_thought="high"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"protocolVersion":1,"agentCapabilities":{{"promptCapabilities":{{"image":true}},"sessionCapabilities":{{"close":{{}}}},"_meta":{{"antigravity":{{"systemPrompt":{{"append":true}}}}}}}}}}}}\n' "$id"
      ;;
    *'"method":"session/new"'*)
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"sessionId":"s1","configOptions":[{{"id":"model","category":"model","currentValue":"%s","options":[{{"value":"gemini-4-neon","name":"Gemini 4 Neon"}},{{"value":"gemini-3.8-flash","name":"Gemini 3.8 Flash"}}]}},{{"id":"thought_level","category":"thought_level","currentValue":"%s","options":[{{"value":"low","name":"Low"}},{{"value":"high","name":"High"}}]}}]}}}}\n' "$id" "$cur_model" "$cur_thought"
      ;;
    *'"method":"session/set_config_option"'*)
      case "$line" in
        *'"configId":"model"'*'"value":"gemini-4-neon"'*) cur_model="gemini-4-neon" ;;
      esac
      case "$line" in
        *'"configId":"thought_level"'*'"value":"low"'*) cur_thought="low" ;;
        *'"configId":"thought_level"'*'"value":"high"'*) cur_thought="high" ;;
      esac
      printf '{{"jsonrpc":"2.0","id":%s,"result":{{"configOptions":[{{"id":"model","category":"model","currentValue":"%s","options":[{{"value":"gemini-4-neon","name":"Gemini 4 Neon"}},{{"value":"gemini-3.8-flash","name":"Gemini 3.8 Flash"}}]}},{{"id":"thought_level","category":"thought_level","currentValue":"%s","options":[{{"value":"low","name":"Low"}},{{"value":"high","name":"High"}}]}}]}}}}\n' "$id" "$cur_model" "$cur_thought"
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
