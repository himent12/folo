//! Runs the `folo` binary and checks the global conventions: JSON lines on stdout, stable exit codes.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use serde_json::Value;

fn folo(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_folo")).args(args).output().unwrap();
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
    assert!(stderr.contains("cpu"), "human text goes to stderr in --json mode");
}

#[test]
fn stubs_exit_4_with_error_envelope() {
    for args in [
        &["train", "--data", "d", "--json"][..],
        &["val", "--model", "m.folo", "--data", "d", "--json"],
        &["predict", "--model", "m.folo", "a.jpg", "--json"],
        &["export", "--model", "m.folo", "--json"],
        &["bench", "--json"],
        &["dataset", "check", "d", "--json"],
        &["dataset", "stats", "d", "--json"],
        &["dataset", "synth", "s", "--json"],
        &["mcp", "--json"],
    ] {
        let (code, stdout, stderr) = folo(args);
        assert_eq!(code, 4, "{args:?}: {stderr}");
        let v = &json_lines(&stdout)[0];
        assert_eq!(v["error"]["code"], "not_implemented", "{args:?}");
        assert!(v["error"]["hint"].is_string());
    }
    let (code, stdout, stderr) = folo(&["train", "--data", "d"]);
    assert_eq!(code, 4);
    assert!(stdout.is_empty());
    assert!(stderr.contains("not implemented"));
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

#[test]
fn serve_mock_prints_the_url_as_json() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["serve", "--mock", "--port", "0", "--json", "--web-dir", "does/not/exist"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
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
