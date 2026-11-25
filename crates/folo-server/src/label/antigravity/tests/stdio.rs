//! Unit tests for bounded stdout line reading, Condvar completion signals, and ACP stdout framing error propagation.

use std::io::{BufReader, Cursor, ErrorKind};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use folo_api::LabelEffort;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::{init_ok, test_req};
use crate::label::agent::{Agent, AgentError};
use crate::label::antigravity::Antigravity;
use crate::label::antigravity::acp::{CANCEL_GRACE, StdoutLine, drive_acp_inner};
use crate::label::antigravity::process::{DoneSignal, join_bounded, read_bounded_line, read_stdout_loop};

#[test]
fn bounded_stdout_line_reader_handles_endings_chunk_boundaries_and_oversize_lines() {
    let mut reader = BufReader::with_capacity(3, Cursor::new(b"ab\ncd\r\n\n\r\nef".as_slice()));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap().as_deref(), Some("ab"));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap().as_deref(), Some("cd"));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap().as_deref(), Some(""));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap().as_deref(), Some(""));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap().as_deref(), Some("ef"));
    assert_eq!(read_bounded_line(&mut reader, 4).unwrap(), None);

    // Exact boundary with CRLF split across buffer refills (`abcd\r` then `\n`) succeeds.
    let mut exact = BufReader::with_capacity(5, Cursor::new(b"abcd\r\nok".as_slice()));
    assert_eq!(read_bounded_line(&mut exact, 4).unwrap().as_deref(), Some("abcd"));
    assert_eq!(read_bounded_line(&mut exact, 4).unwrap().as_deref(), Some("ok"));
    assert_eq!(read_bounded_line(&mut exact, 4).unwrap(), None);

    // Trailing `\r` at EOF without `\n` counts toward line length and is rejected when exceeding the cap.
    let mut trailing_cr_eof = BufReader::with_capacity(5, Cursor::new(b"abcd\r".as_slice()));
    let err = read_bounded_line(&mut trailing_cr_eof, 4).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // Oversize line without newline fails immediately with InvalidData instead of buffering indefinitely.
    let mut oversize = BufReader::with_capacity(3, Cursor::new(b"abcdefgh\n".as_slice()));
    let err = read_bounded_line(&mut oversize, 4).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // Oversize line with newline inside one buffer fails with InvalidData.
    let mut oversize_one_chunk = BufReader::with_capacity(32, Cursor::new(b"abcde\n".as_slice()));
    let err = read_bounded_line(&mut oversize_one_chunk, 4).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    // Invalid UTF-8 fails with InvalidData.
    let mut bad_utf8 = BufReader::new(Cursor::new(b"\xff\xfe\n".as_slice()));
    let err = read_bounded_line(&mut bad_utf8, 16).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
}

#[test]
fn done_signal_and_join_bounded_wake_promptly_and_respect_timeout() {
    // Immediate wake when worker signals completion.
    let done = Arc::new(DoneSignal::new());
    let done_bg = done.clone();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        go_rx.recv().unwrap();
        done_bg.mark();
    });
    let t0 = Instant::now();
    go_tx.send(()).unwrap();
    assert!(join_bounded(handle, &done, Duration::from_secs(5)));
    assert!(t0.elapsed() < Duration::from_secs(2), "condvar join took too long: {:?}", t0.elapsed());

    // Timeout returns false without blocking past the bound; unblocking afterwards finishes cleanly.
    let stuck_done = Arc::new(DoneSignal::new());
    let stuck_done_bg = stuck_done.clone();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let stuck_handle = std::thread::spawn(move || {
        let _ = release_rx.recv();
        stuck_done_bg.mark();
    });
    assert!(!stuck_done.wait_timeout(Duration::from_millis(15)));
    release_tx.send(()).unwrap();
    assert!(join_bounded(stuck_handle, &stuck_done, Duration::from_secs(5)));
}

#[test]
fn oversize_stdout_framing_error_propagates_with_usage_and_immediate_kill() {
    let dir = TestDir::new("label-agy-framing-err");
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let schema = json!({ "type": "object" });
    let cancel = AtomicBool::new(false);
    let req = test_req(dir.path(), &[], None, LabelEffort::Low, &schema, &cancel);

    // 1. Mid-prompt framing error produced by `read_stdout_loop` after a billed `usage_update` preserves usage,
    // sets `kill_immediately`, does NOT blame stderr or report `Unavailable`, and does not trigger the sign-in hint.
    let (out_tx, out_rx) = mpsc::channel::<StdoutLine>();
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(StdoutLine::Line(init_ok(id, true, false, false)));
            }
            Some("session/new") => {
                let _ = out_tx.send(StdoutLine::Line(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                ));
            }
            Some("session/prompt") => {
                let usage_line = json!({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": { "update": { "sessionUpdate": "usage_update", "inputTokens": 410, "outputTokens": 19 } }
                })
                .to_string();
                let stream = format!("{usage_line}\n{}\n", "x".repeat(usage_line.len() + 64));
                let saw = AtomicBool::new(false);
                read_stdout_loop(
                    BufReader::new(Cursor::new(stream.into_bytes())),
                    usage_line.len() + 16,
                    None,
                    &saw,
                    &out_tx,
                );
            }
            _ => {}
        }
        true
    };
    let mut never_exited = || false;
    let out = drive_acp_inner(
        &req,
        &work,
        Duration::from_secs(5),
        CANCEL_GRACE,
        &mut send,
        &out_rx,
        &|| "unrelated stderr warning".into(),
        &mut never_exited,
    );
    assert!(out.kill_immediately);
    let Err(AgentError::Failed(f)) = out.result else { panic!("{:?}", out.result) };
    assert!(f.message.contains("stdout framing error") && f.message.contains("exceeded"), "{f:?}");
    assert!(!f.is_transient(), "{f:?}");
    assert_eq!((f.used.calls, f.used.input_tokens, f.used.output_tokens), (1, 410, 19));
    assert_eq!(Antigravity::new().hint(&f.message), None);

    // 2. Framing error queued before `session/prompt` aborts inside `drain_pending` without dispatching the prompt.
    let (out_tx, out_rx) = mpsc::channel::<StdoutLine>();
    let mut saw_prompt = false;
    let mut send = |msg: &Value| -> bool {
        let id = msg.get("id").and_then(Value::as_u64);
        match msg["method"].as_str() {
            Some("initialize") => {
                let _ = out_tx.send(StdoutLine::Line(init_ok(id, true, false, false)));
            }
            Some("session/new") => {
                let _ = out_tx.send(StdoutLine::Line(
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "sessionId": "s1",
                            "configOptions": [{ "id": "thought_level", "category": "thought_level", "currentValue": "low", "options": [{ "value": "low" }] }]
                        }
                    })
                    .to_string(),
                ));
                let _ = out_tx.send(StdoutLine::Error("`agy acp` stdout framing error: too big".into()));
            }
            Some("session/prompt") => saw_prompt = true,
            _ => {}
        }
        true
    };
    let mut never_exited = || false;
    let out = drive_acp_inner(
        &req,
        &work,
        Duration::from_secs(5),
        CANCEL_GRACE,
        &mut send,
        &out_rx,
        &String::new,
        &mut never_exited,
    );
    assert!(out.kill_immediately && !saw_prompt);
    assert!(matches!(out.result, Err(AgentError::Failed(f)) if f.message.contains("stdout framing error")));
}
