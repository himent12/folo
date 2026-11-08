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
    assert_eq!(v["gpu_detection"], "wgpu");
    for g in v["gpus"].as_array().unwrap() {
        assert!(g["name"].is_string() && g["backend"].is_string() && g["shader_f16"].is_boolean(), "{g}");
    }
    assert!(v["gpu_selected"].is_object());
    assert!(stderr.contains("cpu"), "human text goes to stderr in --json mode");
    assert!(stderr.contains("gpu"), "{stderr}");
}

#[test]
fn stubs_exit_4_with_error_envelope() {
    let args = ["bench", "--json"];
    let (code, stdout, stderr) = folo(&args);
    assert_eq!(code, 4, "{args:?}: {stderr}");
    let v = &json_lines(&stdout)[0];
    assert_eq!(v["error"]["code"], "not_implemented", "{args:?}");
    assert!(v["error"]["hint"].is_string());
    let (code, stdout, stderr) = folo(&["bench"]);
    assert_eq!(code, 4);
    assert!(stdout.is_empty());
    assert!(stderr.contains("not implemented"));
}

#[test]
fn missing_files_exit_3() {
    for (args, code) in [
        (&["train", "--data", "no/such/dataset", "--json"][..], "dataset_not_found"),
        (&["val", "--model", "no/such.folo", "--data", "d", "--json"], "model_not_found"),
        (&["predict", "--model", "no/such.folo", "a.jpg", "--json"], "model_not_found"),
        (&["export", "--model", "no/such.folo", "--json"], "model_not_found"),
        (&["train", "--resume", "no/such/last.folo", "--json"], "path_not_found"),
    ] {
        let (exit, stdout, stderr) = folo(args);
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

#[test]
fn version_has_sha_and_build_date() {
    let (code, stdout, _) = folo(&["--version"]);
    assert_eq!(code, 0);
    // folo 0.1.0 (1a2b3c4d5e6f 2026-11-07)
    let line = stdout.trim();
    let rest = line.strip_prefix(&format!("folo {} (", env!("CARGO_PKG_VERSION"))).expect(line);
    let (sha, date) = rest.strip_suffix(')').and_then(|r| r.split_once(' ')).expect(line);
    assert!(sha == "unknown" || (sha.len() == 12 && sha.chars().all(|c| c.is_ascii_hexdigit())), "{line}");
    let d: Vec<&str> = date.split('-').collect();
    assert_eq!(d.iter().map(|p| p.len()).collect::<Vec<_>>(), [4, 2, 2], "{line}");
    assert!(d.iter().all(|p| p.parse::<u32>().is_ok()), "{line}");
    let (_, stdout, _) = folo(&["doctor", "--json"]);
    let v = &json_lines(&stdout)[0];
    assert_eq!(v["git_sha"], sha);
    assert_eq!(v["build_date"], date);
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
