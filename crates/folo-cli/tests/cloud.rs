//! `folo cloud` with `--json` on a data dir and no server of its own: providers, connecting a `remote` server (a
//! `folo serve --token` child) with the token from the environment and from stdin, the connection's check and
//! offers, instances from the ledger, terminating one that is gone, and the usage errors that keep secrets off the
//! command line.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use folo_test_support::{ChildGuard, TestDir};
use serde_json::{Value, json};

const TOKEN: &str = "cli-remote-token-0123456789";

/// Runs `folo cloud <args> --json --data-dir <home>` with `env` and `stdin`: exit code, stdout records, stderr.
fn cloud(home: &Path, args: &[&str], env: &[(&str, &str)], stdin: &str) -> (i32, Vec<Value>, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_folo"))
        .arg("cloud")
        .args(args)
        .args(["--json", "--data-dir"])
        .arg(home)
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let records =
        stdout.lines().map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}"))).collect();
    (out.status.code().unwrap(), records, String::from_utf8(out.stderr).unwrap())
}

/// The error code of a failed command, checking its exit code.
fn failure(home: &Path, args: &[&str], exit: i32) -> String {
    let (code, records, _) = cloud(home, args, &[], "");
    assert_eq!(code, exit, "{args:?}: {records:?}");
    records.last().unwrap()["error"]["code"].as_str().unwrap().to_string()
}

#[test]
fn connections_offers_and_instances_from_the_command_line() {
    let root = TestDir::new("cli-cloud");
    let home = root.join("home");
    let (code, providers, _) = cloud(&home, &["providers"], &[], "");
    assert_eq!(code, 0);
    let ids: Vec<&str> = providers.iter().map(|p| p["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["runpod", "remote"]);

    // Secrets come from the environment or stdin, never from the command line.
    assert_eq!(failure(&home, &["connect", "runpod", "--secret", "api_key=rpa_LITERAL"], 2), "usage");
    assert_eq!(failure(&home, &["connect", "runpod", "--set", "api_key=rpa_LITERAL"], 2), "usage");
    assert_eq!(
        failure(&home, &["connect", "runpod", "--secret", "api_key=env:FOLO_CLI_TEST_UNSET"], 2),
        "usage"
    );
    assert_eq!(failure(&home, &["connect", "runpod"], 2), "invalid_body");
    assert_eq!(failure(&home, &["connect", "nope"], 3), "provider_not_found");
    assert!(!home.join("cloud").join("connections.json").exists(), "nothing stored");

    // A `folo serve --token` to connect to (no GPU needed: it is only checked and asked for its offer).
    let mut serve: ChildGuard = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["serve", "--port", "0", "--json", "--token", TOKEN, "--data-dir"])
        .arg(root.join("server"))
        .env("RUST_LOG", "warn")
        .env("FOLO_GPU_ADAPTER", "folo-cli-tests-no-such-adapter")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut first = String::new();
    BufReader::new(serve.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let url = serde_json::from_str::<Value>(&first).unwrap()["url"].as_str().unwrap().to_string();

    let set_url = format!("url={url}");
    let (code, made, err) = cloud(
        &home,
        &[
            "connect",
            "remote",
            "--set",
            &set_url,
            "--secret",
            "token=env:FOLO_CLI_TEST_TOKEN",
            "--name",
            "box",
        ],
        &[("FOLO_CLI_TEST_TOKEN", TOKEN)],
        "",
    );
    assert_eq!(code, 0, "{err}");
    let c = &made[0];
    assert_eq!((c["name"].as_str(), c["status"].as_str()), (Some("box"), Some("ok")));
    assert_eq!(
        (c["values"]["url"].as_str(), c["values"]["token"].as_str()),
        (Some(url.as_str()), Some("••••6789"))
    );
    let id = c["id"].as_str().unwrap().to_string();
    // The same from stdin; a wrong token is the provider's answer.
    let (code, records, _) =
        cloud(&home, &["connect", "remote", "--set", &set_url, "--secret", "token=-"], &[], "nope\n");
    assert_eq!((code, records[0]["error"]["code"].as_str()), (1, Some("cloud_auth_failed")));
    let (code, records, _) = cloud(
        &home,
        &["connect", "remote", "--set", &set_url, "--secret", "token=-"],
        &[],
        &format!("{TOKEN}\n"),
    );
    assert_eq!((code, records[0]["name"].as_str()), (0, Some("Remote FOLO server")));
    let second = records[0]["id"].as_str().unwrap().to_string();

    let (_, list, human) = cloud(&home, &["connections"], &[], "");
    assert_eq!(list.len(), 2);
    assert!(!human.contains(TOKEN));
    let (code, checked, _) = cloud(&home, &["check", &id], &[], "");
    assert_eq!((code, checked[0]["status"].as_str()), (0, Some("ok")));
    let (code, offers, _) = cloud(&home, &["offers", &id], &[], "");
    assert_eq!((code, offers.len(), offers[0]["id"].as_str()), (0, 1, Some("cpu")));
    assert_eq!(failure(&home, &["offers", "cc_none"], 3), "connection_not_found");
    let (code, gone, _) = cloud(&home, &["disconnect", &second], &[], "");
    assert_eq!((code, &gone[0]), (0, &json!({ "event": "disconnected", "id": second })));

    // Instances come from the ledger; terminating one that is gone answers its record, without the provider.
    let (code, none, _) = cloud(&home, &["instances"], &[], "");
    assert_eq!((code, none.len()), (0, 0));
    assert_eq!(failure(&home, &["terminate", "ci_none"], 3), "instance_not_found");
    let record = json!({ "id": "ci_0001", "connection_id": id, "provider": "runpod", "provider_ref": "pod1",
                         "name": "folo-run_1-abcd", "run_id": "run_1",
                         "offer": { "id": "o", "gpu": "RTX 4090", "price_per_hour": 0.69 },
                         "status": "terminated", "created": "2026-11-09T00:00:00Z", "ended": "2026-11-09T01:00:00Z" });
    std::fs::create_dir_all(home.join("cloud").join("instances")).unwrap();
    std::fs::write(home.join("cloud").join("instances").join("ci_0001.json"), record.to_string()).unwrap();
    let (_, listed, _) = cloud(&home, &["instances"], &[], "");
    assert_eq!((listed[0]["id"].as_str(), listed[0]["cost_usd"].as_f64()), (Some("ci_0001"), Some(0.69)));
    let (code, ended, _) = cloud(&home, &["terminate", "ci_0001"], &[], "");
    assert_eq!((code, ended[0]["status"].as_str()), (0, Some("terminated")));

    // The token is in connections.json and nowhere else.
    let text = std::fs::read_to_string(home.join("cloud").join("connections.json")).unwrap();
    assert!(text.contains(TOKEN));
    let human = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["cloud", "connections", "--data-dir"])
        .arg(&home)
        .output()
        .unwrap();
    let shown = String::from_utf8(human.stdout).unwrap();
    assert!(shown.starts_with("ID ") && shown.contains("box") && !shown.contains(TOKEN), "{shown}");
}
