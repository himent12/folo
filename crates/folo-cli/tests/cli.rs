//! Runs the `folo` binary and checks the global conventions: JSON lines on stdout, stable exit codes.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use folo_test_support::{ChildGuard, TestDir};
use serde_json::Value;

fn folo(args: &[&str]) -> (i32, String, String) {
    let out = folo_cmd().args(args).output().unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn json_lines(stdout: &str) -> Vec<Value> {
    stdout.lines().map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}"))).collect()
}

#[test]
fn describe_json_is_one_object() {
    let (code, stdout, _) = folo(&["describe", "--json"]);
    assert_eq!(code, 0);
    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["name"], "folo");
    assert!(lines[0]["commands"].as_array().unwrap().len() >= 10);
}

#[test]
fn doctor_json() {
    let (code, stdout, stderr) = folo(&["doctor", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let v = &json_lines(&stdout)[0];
    assert!(v["cpu"]["threads"].as_u64().unwrap() >= 1);
    assert!(v["memory"]["total_bytes"].as_u64().unwrap() > 0);
    assert!(v["web_dist"]["exists"].is_boolean());
    assert_eq!(v["gpu_detection"], "wgpu");
    for g in v["gpus"].as_array().unwrap() {
        assert!(g["name"].is_string() && g["backend"].is_string() && g["shader_f16"].is_boolean(), "{g}");
        assert!(g["memory_bytes"].is_u64(), "{g}");
    }
    assert!(v["gpu_selected"].is_object());
    // The same `capabilities` as `/api/system`: available iff no reason.
    let c = &v["capabilities"];
    assert!(c["device"].is_string() && c["shader_f16"].is_boolean(), "{c}");
    for k in ["tensor_cores", "amp"] {
        assert!(c[k]["available"].is_boolean(), "{c}");
    }
    assert_eq!(c["amp"]["available"].as_bool().unwrap(), c["amp"]["reason"].is_null(), "{c}");
    assert_eq!(c["tensor_cores"]["enabled"].as_bool().unwrap(), c["tensor_cores"]["reason"].is_null(), "{c}");
    if let Some(tc) = v["gpu_selected"]["tensor_cores"].as_str().filter(|s| *s != "on") {
        assert_eq!(c["tensor_cores"]["reason"], tc, "the status doctor prints");
    }
    assert!(stderr.contains("amp "), "{stderr}");
    assert!(stderr.contains("cpu"), "human text goes to stderr in --json mode");
    assert!(stderr.contains("gpu"), "{stderr}");

    // `--version` names the same build: folo 0.1.0 (1a2b3c4d5e6f 2026-11-07).
    let (code, stdout, _) = folo(&["--version"]);
    assert_eq!(code, 0);
    let line = stdout.trim();
    let rest = line.strip_prefix(&format!("folo {} (", env!("CARGO_PKG_VERSION"))).expect(line);
    let (sha, date) = rest.strip_suffix(')').and_then(|r| r.split_once(' ')).expect(line);
    assert!(sha == "unknown" || (sha.len() == 12 && sha.chars().all(|c| c.is_ascii_hexdigit())), "{line}");
    let d: Vec<&str> = date.split('-').collect();
    assert_eq!(d.iter().map(|p| p.len()).collect::<Vec<_>>(), [4, 2, 2], "{line}");
    assert!(d.iter().all(|p| p.parse::<u32>().is_ok()), "{line}");
    assert_eq!(v["git_sha"], sha);
    assert_eq!(v["build_date"], date);
}

#[test]
fn doctor_reports_folo_wasm_in_the_web_dir() {
    let dir = TestDir::new("doctor-wasm");
    std::fs::create_dir_all(dir.join("folo-wasm")).unwrap();
    let report = || {
        // No GPU probe (`webgpu` is the browser's backend): only the web folder matters here.
        let out = folo_cmd()
            .args(["doctor", "--json", "--web-dir", dir.to_str().unwrap()])
            .env("FOLO_GPU_BACKEND", "webgpu")
            .output()
            .unwrap();
        let (stdout, stderr) =
            (String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap());
        assert_eq!(out.status.code(), Some(0), "{stderr}");
        (json_lines(&stdout).remove(0), stderr)
    };
    let (v, stderr) = report();
    assert_eq!(v["web_source"], "dir", "--web-dir wins over an embedded Studio");
    // The folder as given holds `\` on Windows; its record does not.
    assert!(!v["web_dist"]["path"].as_str().unwrap().contains('\\'), "{v}");
    assert_eq!(v["web_dist"]["folo_wasm"], false);
    assert_eq!(v["web_folo_wasm"], false);
    assert!(v["web_embedded"]["folo_wasm"].is_boolean());
    assert!(stderr.contains("no folo-wasm"), "{stderr}");

    std::fs::write(dir.join("folo-wasm/folo_wasm.js"), "").unwrap();
    std::fs::write(dir.join("folo-wasm/folo_wasm_bg.wasm"), "").unwrap();
    let (v, _) = report();
    assert_eq!(v["web_dist"]["folo_wasm"], true);
    assert_eq!(v["web_folo_wasm"], false, "folo serve does not serve a folder without index.html");

    std::fs::write(dir.join("index.html"), "<!doctype html>").unwrap();
    let (v, stderr) = report();
    assert_eq!(v["web_dist"]["folo_wasm"], true);
    assert_eq!(v["web_folo_wasm"], true);
    assert!(stderr.contains("can run models in the browser"), "{stderr}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_command_group_without_its_command_is_usage() {
    // `folo bench` was a stub answering `not_implemented`; it is a group of commands now (tests/bench.rs).
    let args = ["bench", "--json"];
    let (code, stdout, stderr) = folo(&args);
    assert_eq!(code, 2, "{args:?}: {stderr}");
    let v = &json_lines(&stdout)[0];
    assert_eq!(v["error"]["code"], "usage", "{args:?}");
    assert!(v["error"]["message"].as_str().unwrap().contains("run, status"), "{v}");
}

#[test]
fn missing_files_exit_3() {
    for (args, code, run) in [
        (&["train", "--data", "no/such/dataset", "--json"][..], "dataset_not_found"),
        (&["val", "--model", "no/such.folo", "--data", "d", "--json"], "model_not_found"),
        (&["predict", "--model", "no/such.folo", "a.jpg", "--json"], "model_not_found"),
        (&["export", "--model", "no/such.folo", "--json"], "model_not_found"),
        (&["train", "--resume", "no/such/last.folo", "--json"], "path_not_found"),
    ]
    .map(|(args, code)| (args, code, std::thread::spawn(move || folo(args))))
    {
        let (exit, stdout, stderr) = run.join().unwrap();
        assert_eq!(exit, 3, "{args:?}: {stderr}");
        assert_eq!(json_lines(&stdout).last().unwrap()["error"]["code"], code, "{args:?}");
    }
}

#[test]
fn usage_errors_exit_2() {
    let (code, stdout, _) = folo(&["no-such-command", "--json"]);
    assert_eq!(code, 2);
    assert_eq!(json_lines(&stdout)[0]["error"]["code"], "usage");
    let (code, _, _) = folo(&["serve", "--port", "nope"]);
    assert_eq!(code, 2);
    let (code, _, _) = folo(&["train"]);
    assert_eq!(code, 2, "missing required --data");
    let (code, stdout, _) = folo(&["--help"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("serve"));
}

/// `--fusion-iou` outside `0..=1` (or NaN) is a usage error before any model is opened.
#[test]
fn fusion_iou_out_of_range_is_a_usage_error() {
    for args in [
        &[
            "val",
            "--json",
            "--model",
            "none.folo",
            "--data",
            "none",
            "--fusion",
            "dfl",
            "--fusion-iou",
            "NaN",
        ][..],
        &["val", "--json", "--model", "none.folo", "--data", "none", "--fusion-iou", "1.5"],
        &["predict", "--json", "--model", "none.folo", "a.jpg", "--fusion", "vote", "--fusion-iou=-0.1"],
    ] {
        let (code, stdout, _) = folo(args);
        assert_eq!(code, 2, "{args:?}");
        let e = &json_lines(&stdout)[0]["error"];
        assert_eq!(e["code"], "usage", "{args:?}");
        assert!(e["message"].as_str().unwrap().contains("--fusion-iou"), "{e}");
    }
}

/// The `--json` error of a missing argument names it: clap puts the names on the lines after "the following
/// required arguments were not provided:", which the message used to cut off.
#[test]
fn usage_errors_name_the_missing_argument() {
    for (args, want) in [
        (&["dataset", "stats", "--json"][..], "<PATH>"),
        (&["train", "--json"], "--data"),
        (&["predict", "--json", "a.jpg"], "--model"),
    ] {
        let (code, stdout, _) = folo(args);
        assert_eq!(code, 2, "{args:?}");
        let e = &json_lines(&stdout)[0]["error"];
        assert_eq!(e["code"], "usage");
        let message = e["message"].as_str().unwrap();
        assert!(message.contains(want), "{args:?}: {message}");
        assert!(!message.contains('\n') && !message.contains("Usage:"), "{args:?}: {message}");
    }
}

/// `predict --save` overlapping a source (inside it at any depth, before or after it exists, or holding one) is a
/// usage error before the model is opened, whatever the spelling, and writes nothing (issue #313: the second run
/// read the first run's copies as images).
#[test]
fn predict_save_overlapping_a_source_is_a_usage_error() {
    let dir = TestDir::new("cli-predict-save");
    std::fs::create_dir_all(dir.join("photos/out")).unwrap();
    std::fs::create_dir_all(dir.join("photos/sub")).unwrap();
    std::fs::write(dir.join("photos/a.jpg"), b"x").unwrap();
    let listing = |p: &str| -> Vec<_> {
        std::fs::read_dir(dir.join(p)).unwrap().map(|e| e.unwrap().file_name()).collect()
    };
    let before = (listing("."), listing("photos"));
    let absolute = dir.join("photos/annotated");
    let mut refused = vec![
        ("photos", "photos/annotated"),
        ("photos", "photos/out"),
        ("photos", "photos/sub/deep/new"),
        ("photos", "photos"),
        ("photos", "./photos/annotated"),
        ("photos", "photos/sub/../annotated"),
        // `..` after a folder that does not exist (`scratch`) is folded before the existing part is resolved.
        ("photos", "scratch/../photos/out"),
        ("photos", absolute.to_str().unwrap()),
        ("photos/sub", "photos"),
        ("photos/a.jpg", "photos"),
    ];
    if cfg!(windows) {
        refused.push(("photos", "PHOTOS/annotated"));
    }
    let predict = |src: &str, save: &str| {
        let out = folo_cmd()
            .current_dir(&dir)
            .args(["predict", "--json", "--model", "none.folo", src, "--save", save])
            .output()
            .unwrap();
        (out.status.code().unwrap(), json_lines(&String::from_utf8(out.stdout).unwrap()).pop().unwrap())
    };
    for (src, save) in refused {
        let (code, last) = predict(src, save);
        assert_eq!(code, 2, "{src} --save {save}: {last}");
        assert_eq!(last["error"]["code"], "usage", "{src} --save {save}: {last}");
        assert!(last["error"]["message"].as_str().unwrap().contains("--save"), "{last}");
    }
    assert_eq!((listing("."), listing("photos")), before, "a refused run creates nothing");
    // Outside the sources (also a sibling sharing the name's start) this check passes: the missing model stops it.
    for save in ["annotated", "photos-annotated", "photos/new/../../out"] {
        let (code, last) = predict("photos", save);
        assert_eq!((code, &last["error"]["code"]), (3, &"model_not_found".into()), "--save {save}: {last}");
    }
    // A source that does not exist gets the answer it gets without `--save` (the missing model, which main checks
    // first; with a model, `path_not_found`): the overlap with `photos` does not hide it.
    let run = |save: &[&str]| {
        let args = ["predict", "--json", "--model", "none.folo", "missing", "photos"];
        let out = folo_cmd().current_dir(&dir).args(args).args(save).output().unwrap();
        let last = json_lines(&String::from_utf8(out.stdout).unwrap()).pop().unwrap();
        (out.status.code().unwrap(), last["error"]["code"].clone())
    };
    assert_eq!(run(&[]), (3, "model_not_found".into()));
    assert_eq!(run(&["--save", "photos/out"]), run(&[]));
}

/// A `--data-dir` that is a file is a usage error up front: `serve` reported `listening` and `mcp` and `cloud` ran,
/// and every write then failed with `internal` ("creating a folder in <file>/datasets").
#[test]
fn data_dir_that_is_a_file_is_a_usage_error() {
    let dir = TestDir::new("cli-data-dir-file");
    let file = dir.join("not-a-dir");
    std::fs::write(&file, b"x").unwrap();
    let file = file.to_str().unwrap();
    // `folo mcp` keeps stdout for the protocol: its error is on stderr.
    let mcp = folo_cmd().args(["mcp", "--data-dir", file]).stdin(Stdio::null()).output().unwrap();
    let stderr = String::from_utf8_lossy(&mcp.stderr);
    assert_eq!(mcp.status.code(), Some(2), "mcp: {stderr}");
    assert!(mcp.stdout.is_empty() && stderr.contains("--data-dir"), "mcp: {stderr}");
    for args in [
        &["cloud", "instances", "--json", "--data-dir", file][..],
        &["serve", "--json", "--port", "0", "--data-dir", file],
    ] {
        let mut child: ChildGuard = folo_cmd()
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
            .into();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        // A server that started anyway is still running: the ChildGuard stops it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let status = loop {
            match child.try_wait().unwrap() {
                Some(s) => break Some(s),
                None if std::time::Instant::now() > deadline => break None,
                None => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        };
        let v: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{args:?}: {e}: {line:?}"));
        assert_eq!(v["error"]["code"], "usage", "{args:?}: {line}");
        assert!(v["error"]["message"].as_str().unwrap().contains("--data-dir"), "{args:?}: {line}");
        assert_eq!(status.and_then(|s| s.code()), Some(2), "{args:?}");
    }
}

/// One process owns a data dir, and a relative `--data-dir` names the same folder from anywhere: a server started
/// with `--data-dir data` reports it absolute, so a second `folo serve` points at it and an embedded `folo mcp`
/// started from another working directory goes through it (its owner check compares the folders). No GPU (as the
/// other tests): a slow adapter start could outlast that check's timeout.
#[test]
fn a_data_dir_has_one_owner_whatever_the_working_directory() {
    use std::io::Write;
    let (a, b) = (TestDir::new("cli-owner-a"), TestDir::new("cli-owner-b"));
    let data = a.join("data");
    let mut server: ChildGuard = folo_cmd()
        .args(["--json", "serve", "--port", "0", "--data-dir", "data"])
        .current_dir(&*a)
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut line = String::new();
    BufReader::new(server.stdout.take().unwrap()).read_line(&mut line).unwrap();
    let listening: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(listening["event"], "listening", "{line}");
    assert!(!listening["data_dir"].as_str().unwrap().contains('\\'), "{line}");
    let reported = std::path::PathBuf::from(listening["data_dir"].as_str().unwrap());
    assert!(reported.is_absolute(), "{line}");
    assert_eq!(std::fs::canonicalize(&reported).unwrap(), std::fs::canonicalize(&data).unwrap());

    let second = folo_cmd()
        .args(["--json", "serve", "--port", "0", "--data-dir", data.to_str().unwrap()])
        .current_dir(&*b)
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(second.status.code(), Some(1));
    let first_line = String::from_utf8_lossy(&second.stdout).lines().next().unwrap_or_default().to_string();
    let in_use: Value = serde_json::from_str(&first_line).unwrap();
    assert_eq!((&in_use["event"], &in_use["url"]), (&Value::from("data_dir_in_use"), &listening["url"]));

    let mut mcp: ChildGuard = folo_cmd()
        .args(["mcp", "--data-dir", data.to_str().unwrap()])
        .current_dir(&*b)
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut stdin = mcp.stdin.take().unwrap();
    for msg in [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"folo_info","arguments":{}}}"#,
    ] {
        writeln!(stdin, "{msg}").unwrap();
    }
    let info = BufReader::new(mcp.stdout.take().unwrap())
        .lines()
        .map_while(Result::ok)
        .find(|l| l.contains(r#""id":2"#))
        .expect("folo_info answered");
    drop(stdin);
    assert!(info.contains(r#"\"mode\":\"proxy\""#) || info.contains(r#""mode":"proxy""#), "{info}");
}

/// Reads one line from `r` within `timeout`, failing cleanly instead of hanging if the child never answers.
fn read_line_within(r: impl std::io::Read + Send + 'static, timeout: std::time::Duration) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = BufReader::new(r).read_line(&mut line);
        let _ = tx.send(line);
    });
    rx.recv_timeout(timeout).expect("child wrote a line before the deadline")
}

/// Reads lines from `r` until one matches `pred`, within `timeout`.
fn find_line_within(
    r: impl std::io::Read + Send + 'static,
    timeout: std::time::Duration,
    pred: fn(&str) -> bool,
) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            if pred(&line) {
                let _ = tx.send(line);
                break;
            }
        }
    });
    rx.recv_timeout(timeout).expect("child answered before the deadline")
}

/// A `serve.json` written by an older server bound to `0.0.0.0` is normalized to loopback when read, so a second
/// `folo serve` reports a connectable URL and `folo mcp` auto-proxies to the owner without binding a wildcard
/// listener in the test.
#[test]
fn mcp_auto_proxies_through_a_wildcard_serve_json_record() {
    use std::io::Write;
    use std::time::Duration;
    let dir = TestDir::new("serve-wildcard-record");
    let mut server: ChildGuard = folo_cmd()
        .args(["--json", "serve", "--port", "0", "--data-dir", dir.to_str().unwrap()])
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let line = read_line_within(server.stdout.take().unwrap(), Duration::from_secs(30));
    let listening: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    assert_eq!(listening["event"], "listening", "{line}");
    let url = listening["url"].as_str().unwrap().to_string();
    let port = url.rsplit(':').next().unwrap();

    // Warm `/api/system` before `folo mcp` checks the owner so the first hardware sample never races `serves()`.
    folo_client::Client::with_timeouts(&url, Duration::from_secs(5), Duration::from_secs(60))
        .get::<Value>("/system")
        .unwrap();

    let serve_json: Value = serde_json::from_slice(&std::fs::read(dir.join("serve.json")).unwrap()).unwrap();
    assert_eq!(serve_json["url"], format!("http://127.0.0.1:{port}"), "{serve_json}");

    // Simulate a `serve.json` written by an older server bound to `0.0.0.0`.
    std::fs::write(
        dir.join("serve.json"),
        serde_json::to_vec(&serde_json::json!({ "url": format!("http://0.0.0.0:{port}"), "pid": 1 }))
            .unwrap(),
    )
    .unwrap();

    let second = folo_cmd()
        .args(["--json", "serve", "--port", "0", "--data-dir", dir.to_str().unwrap()])
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(second.status.code(), Some(1));
    let first_line = String::from_utf8_lossy(&second.stdout).lines().next().unwrap_or_default().to_string();
    let in_use: Value = serde_json::from_str(&first_line).unwrap_or_else(|e| panic!("{e}: {first_line:?}"));
    assert_eq!(in_use["event"], "data_dir_in_use", "{in_use}");
    assert_eq!(in_use["url"], format!("http://127.0.0.1:{port}"), "{in_use}");

    let mut mcp: ChildGuard = folo_cmd()
        .args(["mcp", "--data-dir", dir.to_str().unwrap()])
        .env("FOLO_GPU_BACKEND", "webgpu")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut stdin = mcp.stdin.take().unwrap();
    for msg in [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"folo_info","arguments":{}}}"#,
    ] {
        writeln!(stdin, "{msg}").unwrap();
    }
    let info =
        find_line_within(mcp.stdout.take().unwrap(), Duration::from_secs(30), |l| l.contains(r#""id":2"#));
    drop(stdin);
    assert!(info.contains(r#"\"mode\":\"proxy\""#) || info.contains(r#""mode":"proxy""#), "{info}");
}

/// `folo mcp`'s stdout carries protocol messages only, also with `--json` (docs/CLI.md): a startup error is on
/// stderr, not an error envelope on stdout. So is a usage error clap reports, wherever `--json` is; other commands
/// keep their JSON usage error.
#[test]
fn mcp_errors_stay_off_stdout_with_json() {
    for (args, want) in [
        (&["mcp", "--json", "--server", "ftp://example.com"][..], "--server"),
        (&["mcp", "--json", "--extra"], "--extra"),
        (&["--json", "mcp", "--extra"], "--extra"),
        (&["mcp", "--extra", "--json"], "--extra"),
        (&["mcp", "--json", "--server"], "--server"),
        // A root-level error before `mcp`.
        (&["--json", "--json", "mcp"], "--json"),
        (&["--json=true", "mcp", "--json"], "--json"),
    ] {
        let out = folo_cmd().args(args).stdin(Stdio::null()).output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {stderr}");
        assert!(out.stdout.is_empty(), "{args:?}: stdout: {}", String::from_utf8_lossy(&out.stdout));
        assert!(stderr.contains(want), "{args:?}: {stderr}");
    }
    // A root flag that is not Unicode, before `mcp`.
    #[cfg(windows)]
    let flag = {
        use std::os::windows::ffi::OsStringExt;
        std::ffi::OsString::from_wide(&"--x".encode_utf16().chain([0xD800]).collect::<Vec<_>>())
    };
    #[cfg(unix)]
    let flag = {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(b"--x\xFF".to_vec())
    };
    let out = folo_cmd().arg(flag).args(["mcp", "--json"]).stdin(Stdio::null()).output().unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.stdout.is_empty(), "stdout: {}", String::from_utf8_lossy(&out.stdout));
    for args in [
        &["--json", "serve", "--extra"][..],
        &["--json", "--", "mcp"],
        &["predict", "mcp", "--json", "--bad"],
        &["mcpp", "--json"],
        &["--json"],
    ] {
        let (code, stdout, _) = folo(args);
        assert_eq!(code, 2, "{args:?}");
        assert_eq!(json_lines(&stdout)[0]["error"]["code"], "usage", "{args:?}");
    }
}

#[test]
fn serve_mock_prints_the_url_as_json() {
    let mut child: ChildGuard = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--json", "--web-dir", "does/not/exist"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    child.kill().unwrap();
    let _ = child.wait();
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["event"], "listening");
    assert_eq!(v["mock"], true);
    assert_eq!(v["web_ready"], false);
    assert!(v["url"].as_str().unwrap().starts_with("http://127.0.0.1:"));
}

/// `folo serve --mock` on loopback refuses a foreign `Host` (DNS rebinding) and answers the URL it prints (FOLO
/// Desktop connects to that `http://127.0.0.1:<port>`) and an `--allow-host` name; `describe` lists the flag.
#[test]
fn serve_refuses_foreign_hosts_on_loopback() {
    use std::io::{Read, Write};
    let mut child: ChildGuard = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--json", "--allow-host", "folo.tunnel.example"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    let v: Value = serde_json::from_str(&line).unwrap();
    let addr = v["url"].as_str().unwrap().trim_start_matches("http://").to_owned();
    let status = |host: &str| {
        let mut s = std::net::TcpStream::connect(&addr).unwrap();
        write!(s, "GET /api/health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").unwrap();
        let mut res = String::new();
        s.read_to_string(&mut res).unwrap();
        res.split(' ').nth(1).unwrap_or("").to_owned()
    };
    assert_eq!(status(&addr), "200");
    assert_eq!(status("folo.tunnel.example"), "200");
    assert_eq!(status("evil.example"), "403");
    child.kill().unwrap();
    let _ = child.wait();

    let (code, stdout, _) = folo(&["describe", "serve", "--json"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("\"allow-host\"") || stdout.contains("--allow-host"), "{stdout}");
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c
}
