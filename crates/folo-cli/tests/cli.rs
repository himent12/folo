//! Runs the `folo` binary and checks the global conventions: JSON lines on stdout, stable exit codes.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use folo_test_support::{ChildGuard, TestDir};
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
}

#[test]
fn doctor_reports_folo_wasm_in_the_web_dir() {
    let dir = TestDir::new("doctor-wasm");
    std::fs::create_dir_all(dir.join("folo-wasm")).unwrap();
    let report = || {
        let (code, stdout, stderr) = folo(&["doctor", "--json", "--web-dir", dir.to_str().unwrap()]);
        assert_eq!(code, 0, "{stderr}");
        (json_lines(&stdout).remove(0), stderr)
    };
    let (v, stderr) = report();
    assert_eq!(v["web_source"], "dir", "--web-dir wins over an embedded Studio");
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
    let mut child: ChildGuard = Command::new(env!("CARGO_BIN_EXE_folo"))
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
