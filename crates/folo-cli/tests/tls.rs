//! End-to-end CLI tests for `folo serve --tls-cert / --tls-key` (and `FOLO_TLS_CERT` / `FOLO_TLS_KEY`) and
//! `folo mcp --tls-ca` (and `FOLO_TLS_CA`): HTTPS startup banners & JSON records, authenticated API & MCP proxy
//! flows, plaintext rejection, and upfront validation of missing, unpaired, malformed, or mismatched credentials.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::Duration;

use folo_test_support::{ChildGuard, TestDir};
use serde_json::Value;

const SECRET: &str = "cli-tls-token-0123456789";

fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1")
        .env("FOLO_GPU_BACKEND", "webgpu")
        .env_remove("FOLO_TLS_CERT")
        .env_remove("FOLO_TLS_KEY")
        .env_remove("FOLO_TLS_CA")
        .env_remove("FOLO_TOKEN")
        .env_remove("FOLO_SERVER");
    c
}

fn make_cert(sans: &[&str]) -> (String, String) {
    let params = rcgen::CertificateParams::new(sans.iter().map(|s| (*s).to_string()).collect::<Vec<_>>())
        .expect("cert params");
    let key = rcgen::KeyPair::generate().expect("key pair");
    let cert = params.self_signed(&key).expect("self-signed cert");
    (cert.pem(), key.serialize_pem())
}

fn read_line_within(r: impl std::io::Read + Send + 'static, timeout: Duration) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(r);
        let mut line = String::new();
        let _ = reader.read_line(&mut line);
        let _ = tx.send(line);
        // Keep draining until EOF so the child never hits a broken pipe on subsequent writes.
        let mut sink = String::new();
        while reader.read_line(&mut sink).unwrap_or(0) > 0 {
            sink.clear();
        }
    });
    rx.recv_timeout(timeout).expect("child wrote a line before the deadline")
}

fn find_line_within(
    r: impl std::io::Read + Send + 'static,
    timeout: Duration,
    pred: fn(&str) -> bool,
) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut sent = false;
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            if !sent && pred(&line) {
                let _ = tx.send(line);
                sent = true;
            }
        }
    });
    rx.recv_timeout(timeout).expect("child answered before the deadline")
}

#[test]
fn serve_tls_serves_https_with_token_and_mcp_proxy() {
    let dir = TestDir::new("cli-serve-tls");
    let (cert_pem, key_pem) = make_cert(&["localhost", "127.0.0.1"]);
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    std::fs::write(&cert_path, &cert_pem).unwrap();
    std::fs::write(&key_path, &key_pem).unwrap();

    // Start `folo serve --mock` with TLS + token via env vars (`FOLO_TLS_CERT`, `FOLO_TLS_KEY`, `FOLO_TOKEN`).
    let mut server: ChildGuard = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--json"])
        .env("FOLO_TLS_CERT", &cert_path)
        .env("FOLO_TLS_KEY", &key_path)
        .env("FOLO_TOKEN", SECRET)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .into();

    let line = read_line_within(server.stdout.take().unwrap(), Duration::from_secs(30));
    let listening: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    assert_eq!(listening["event"], "listening", "{listening}");
    assert_eq!(listening["tls"], true, "{listening}");
    assert_eq!(listening["token"], true, "{listening}");
    let url = listening["url"].as_str().unwrap().to_string();
    assert!(url.starts_with("https://127.0.0.1:"), "{listening}");
    assert_eq!(listening["api"], format!("{url}/api"), "{listening}");

    // Human banner on stderr carries https:// and TLS enabled.
    let banner = find_line_within(server.stderr.take().unwrap(), Duration::from_secs(10), |l| {
        l.contains("TLS:") && l.contains("enabled")
    });
    assert!(banner.contains("enabled"), "{banner}");

    // Authenticated HTTPS requests succeed through `folo-client` with the custom CA.
    let client = folo_client::Client::with_tls_ca_file(
        &url,
        Duration::from_secs(5),
        Duration::from_secs(15),
        &cert_path,
    )
    .unwrap();
    let health: Value = client.get("/health").unwrap();
    assert_eq!(health["status"], "ok");
    let unauth = client.get::<Value>("/system").unwrap_err();
    assert_eq!(unauth.status.map(|s| s.as_u16()), Some(401), "{unauth:?}");
    assert_eq!(unauth.code(), "unauthorized", "{unauth:?}");
    let authed = client.token(SECRET);
    let sys: Value = authed.get("/system").unwrap();
    assert!(sys["cpu"]["threads"].as_u64().unwrap() >= 1);

    // Plaintext HTTP hitting the HTTPS port fails without crashing `folo serve`.
    let addr = url.trim_start_matches("https://");
    let mut plain = TcpStream::connect(addr).unwrap();
    plain.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let _ = plain.write_all(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    let mut buf = Vec::new();
    let _ = plain.read_to_end(&mut buf);
    assert!(!buf.starts_with(b"HTTP/1.1 200"));
    assert_eq!(authed.get::<Value>("/health").unwrap()["status"], "ok");

    // `folo mcp --server https://... --tls-ca <cert>` connects over HTTPS and answers `folo_info`.
    let mut mcp: ChildGuard = folo_cmd()
        .args(["mcp", "--server", &url, "--token", SECRET, "--tls-ca", cert_path.to_str().unwrap()])
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
    assert!(info.contains("https://127.0.0.1:"), "{info}");
}

#[test]
fn serve_tls_rejects_unpaired_missing_malformed_and_mismatched_files_upfront() {
    let dir = TestDir::new("cli-serve-tls-errors");
    let (cert_a, key_a) = make_cert(&["localhost"]);
    let (_cert_b, key_b) = make_cert(&["localhost"]);
    let cert_a_path = dir.join("cert-a.pem");
    let key_a_path = dir.join("key-a.pem");
    let key_b_path = dir.join("key-b.pem");
    let bad_pem_path = dir.join("bad.pem");
    std::fs::write(&cert_a_path, &cert_a).unwrap();
    std::fs::write(&key_a_path, &key_a).unwrap();
    std::fs::write(&key_b_path, &key_b).unwrap();
    std::fs::write(&bad_pem_path, "not valid pem data").unwrap();

    let run = |args: &[&str], env: &[(&str, &str)]| -> (i32, Value) {
        let out = folo_cmd().args(args).envs(env.iter().copied()).stdin(Stdio::null()).output().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let last: Value =
            stdout.lines().last().and_then(|l| serde_json::from_str(l).ok()).unwrap_or_else(|| {
                panic!(
                    "{args:?}: expected JSON error on stdout, got stdout={stdout:?} stderr={}",
                    String::from_utf8_lossy(&out.stderr)
                )
            });
        (out.status.code().unwrap(), last)
    };

    let cert_s = cert_a_path.to_str().unwrap();
    let key_a_s = key_a_path.to_str().unwrap();
    let key_b_s = key_b_path.to_str().unwrap();
    let bad_s = bad_pem_path.to_str().unwrap();
    let missing_s = dir.join("does-not-exist.pem");
    let missing_s = missing_s.to_str().unwrap();

    // 1. Unpaired flags (CLI and env) -> usage (exit 2).
    let (code, v) = run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", cert_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("--tls-key"), "{v}");

    let (code, v) = run(&["serve", "--mock", "--json", "--port", "0", "--tls-key", key_a_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("--tls-cert"), "{v}");

    let (code, v) = run(&["serve", "--mock", "--json", "--port", "0"], &[("FOLO_TLS_CERT", cert_s)]);
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{v}");

    // Empty paths -> usage (exit 2).
    let (code, v) =
        run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", "", "--tls-key", key_a_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{v}");

    // 2. Nonexistent cert or key -> path_not_found (exit 3).
    let (code, v) = run(
        &["serve", "--mock", "--json", "--port", "0", "--tls-cert", missing_s, "--tls-key", key_a_s],
        &[],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (3, Some("path_not_found")), "{v}");

    let (code, v) =
        run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", cert_s, "--tls-key", missing_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (3, Some("path_not_found")), "{v}");

    // 3. Malformed / empty PEM -> serve_failed (exit 1).
    let (code, v) =
        run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", bad_s, "--tls-key", key_a_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("serve_failed")), "{v}");

    let (code, v) =
        run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", cert_s, "--tls-key", bad_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("serve_failed")), "{v}");

    // 4. Mismatched cert and key -> serve_failed (exit 1) with clear message.
    let (code, v) =
        run(&["serve", "--mock", "--json", "--port", "0", "--tls-cert", cert_s, "--tls-key", key_b_s], &[]);
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("serve_failed")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("do not match"), "{v}");

    // 5. `folo mcp --tls-ca` rejects missing file (exit 3) and malformed PEM (exit 2), keeping stdout clean.
    let mcp_missing = folo_cmd()
        .args(["mcp", "--server", "https://127.0.0.1:1", "--tls-ca", missing_s])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(mcp_missing.status.code(), Some(3));
    assert!(mcp_missing.stdout.is_empty());

    let mcp_bad = folo_cmd()
        .args(["mcp", "--server", "https://127.0.0.1:1", "--tls-ca", bad_s])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(mcp_bad.status.code(), Some(2));
    assert!(mcp_bad.stdout.is_empty());

    // 6. `folo describe serve --json` advertises `--tls-cert` / `--tls-key` pairing (`requires`) and `--port` bounds.
    let desc = folo_cmd().args(["describe", "serve", "--json"]).output().unwrap();
    assert_eq!(desc.status.code(), Some(0));
    let v: Value = serde_json::from_slice(&desc.stdout).unwrap();
    let args = v["args"].as_array().unwrap();
    let cert_arg = args.iter().find(|a| a["name"] == "tls_cert").unwrap();
    let key_arg = args.iter().find(|a| a["name"] == "tls_key").unwrap();
    let port_arg = args.iter().find(|a| a["name"] == "port").unwrap();
    assert_eq!(
        (cert_arg["long"].as_str(), cert_arg["env"].as_str(), cert_arg["type"].as_str()),
        (Some("--tls-cert"), Some("FOLO_TLS_CERT"), Some("path"))
    );
    assert_eq!(cert_arg["requires"], serde_json::json!(["--tls-key"]));
    assert_eq!(
        (key_arg["long"].as_str(), key_arg["env"].as_str(), key_arg["type"].as_str()),
        (Some("--tls-key"), Some("FOLO_TLS_KEY"), Some("path"))
    );
    assert_eq!(key_arg["requires"], serde_json::json!(["--tls-cert"]));
    assert_eq!(port_arg["bounds"], serde_json::json!({ "minimum": 0, "maximum": 65535 }));

    // 7. Single-line / mangled private key PEM never leaks key bytes into JSON or stderr.
    let sentinel = "CLISECRETKEYBYTES987654321";
    let mangled_key_path = dir.join("mangled-key.pem");
    std::fs::write(
        &mangled_key_path,
        format!("-----BEGIN PRIVATE KEY-----{sentinel}-----END PRIVATE KEY-----"),
    )
    .unwrap();
    let out = folo_cmd()
        .args([
            "serve",
            "--mock",
            "--json",
            "--port",
            "0",
            "--tls-cert",
            cert_s,
            "--tls-key",
            mangled_key_path.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout_s = String::from_utf8_lossy(&out.stdout);
    let stderr_s = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stdout={stdout_s} stderr={stderr_s}");
    assert!(!stdout_s.contains(sentinel) && !stderr_s.contains(sentinel), "secret leaked!");
    assert!(
        !stdout_s.contains("67, 76, 73, 83") && !stderr_s.contains("67, 76, 73, 83"),
        "secret bytes leaked!"
    );

    // 8. Corrupt second certificate (`AA==`) in the chain is rejected upfront before binding.
    let bad_chain_path = dir.join("bad-chain.pem");
    std::fs::write(
        &bad_chain_path,
        format!("{cert_a}\n-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n"),
    )
    .unwrap();
    let (code, v) = run(
        &[
            "serve",
            "--mock",
            "--json",
            "--port",
            "0",
            "--tls-cert",
            bad_chain_path.to_str().unwrap(),
            "--tls-key",
            key_a_s,
        ],
        &[],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("serve_failed")), "{v}");

    // 9. `folo mcp --tls-ca` rejects invalid X.509 DER (`AA==`) and non-PEM BEFORE touching `--data-dir` / `serve.lock`.
    let bad_ca_der = dir.join("bad-ca-der.pem");
    std::fs::write(&bad_ca_der, "-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n").unwrap();
    let fresh_data = dir.join("fresh-mcp-data");
    for bad_ca in [bad_s, bad_ca_der.to_str().unwrap()] {
        let out = folo_cmd()
            .args(["mcp", "--data-dir", fresh_data.to_str().unwrap(), "--tls-ca", bad_ca])
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "stderr={}", String::from_utf8_lossy(&out.stderr));
        assert!(out.stdout.is_empty());
        assert!(!fresh_data.join("serve.lock").exists(), "bad --tls-ca must not acquire serve.lock");
    }
}

#[test]
fn serve_tls_preserves_dns_host_for_dns_only_cert_and_mcp_auto_proxy() {
    let dir = TestDir::new("cli-serve-tls-dns");
    let data = dir.join("data");
    // DNS-only certificate (`DNS:localhost`, no IP SAN): requires `https://localhost:<port>` to verify.
    let (cert_pem, key_pem) = make_cert(&["localhost"]);
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    std::fs::write(&cert_path, &cert_pem).unwrap();
    std::fs::write(&key_path, &key_pem).unwrap();

    let mut server: ChildGuard = folo_cmd()
        .args([
            "serve",
            "--host",
            "localhost",
            "--port",
            "0",
            "--json",
            "--data-dir",
            data.to_str().unwrap(),
            "--tls-cert",
            cert_path.to_str().unwrap(),
            "--tls-key",
            key_path.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();

    let line = read_line_within(server.stdout.take().unwrap(), Duration::from_secs(30));
    let listening: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    assert_eq!(listening["event"], "listening", "{listening}");
    let url = listening["url"].as_str().unwrap().to_string();
    assert!(url.starts_with("https://localhost:"), "expected DNS hostname in URL: {listening}");

    // `serve.json` records `https://localhost:<port>` and a verified client succeeds against it.
    let serve_json: Value = serde_json::from_slice(&std::fs::read(data.join("serve.json")).unwrap()).unwrap();
    assert_eq!(serve_json["url"], url, "{serve_json}");
    let client = folo_client::Client::with_tls_ca_file(
        &url,
        Duration::from_secs(5),
        Duration::from_secs(30),
        &cert_path,
    )
    .unwrap();
    let sys: Value = client.get("/system").unwrap();
    assert!(sys["data_dir"].is_string(), "{sys}");

    // `folo mcp --data-dir <data> --tls-ca <cert>` auto-discovers via `serve.json` and proxies cleanly over HTTPS.
    let mut mcp: ChildGuard = folo_cmd()
        .args(["mcp", "--data-dir", data.to_str().unwrap(), "--tls-ca", cert_path.to_str().unwrap()])
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
    assert!(info.contains("https://localhost:"), "{info}");

    // An invalid `--tls-ca` against the running TLS server reports a usage error (exit 2), NOT `data_dir_in_use`.
    let bad_ca = dir.join("not-a-ca.pem");
    std::fs::write(&bad_ca, "not pem").unwrap();
    let bad_mcp = folo_cmd()
        .args(["mcp", "--data-dir", data.to_str().unwrap(), "--tls-ca", bad_ca.to_str().unwrap()])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(bad_mcp.status.code(), Some(2), "stderr={}", String::from_utf8_lossy(&bad_mcp.stderr));
    assert!(bad_mcp.stdout.is_empty());
}

#[test]
fn serve_tls_works_with_bench_label_and_cloud_remote() {
    let dir = TestDir::new("cli-serve-tls-clients");
    let worker_data = dir.join("worker-data");
    let control_data = dir.join("control-data");
    let (cert_pem, key_pem) = make_cert(&["127.0.0.1"]);
    let (other_ca_pem, _) = make_cert(&["127.0.0.1"]);
    let cert_path = dir.join("self.pem");
    let key_path = dir.join("self.key");
    let other_ca_path = dir.join("other-ca.pem");
    let bad_ca_path = dir.join("bad-ca.pem");
    std::fs::write(&cert_path, &cert_pem).unwrap();
    std::fs::write(&key_path, &key_pem).unwrap();
    std::fs::write(&other_ca_path, &other_ca_pem).unwrap();
    std::fs::write(&bad_ca_path, "not valid pem").unwrap();

    let cert_s = cert_path.to_str().unwrap();
    let key_s = key_path.to_str().unwrap();
    let other_ca_s = other_ca_path.to_str().unwrap();
    let bad_ca_s = bad_ca_path.to_str().unwrap();
    let missing_ca = dir.join("missing-ca.pem");
    let missing_ca_s = missing_ca.to_str().unwrap();
    let worker_s = worker_data.to_str().unwrap();
    let control_s = control_data.to_str().unwrap();

    let mut server: ChildGuard = folo_cmd()
        .args([
            "serve",
            "--data-dir",
            worker_s,
            "--port",
            "0",
            "--device",
            "cpu",
            "--tls-cert",
            cert_s,
            "--tls-key",
            key_s,
            "--token",
            SECRET,
            "--json",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();

    let line = read_line_within(server.stdout.take().unwrap(), Duration::from_secs(30));
    let listening: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    assert_eq!(listening["event"], "listening", "{listening}");
    let https_url = listening["url"].as_str().unwrap().to_string();
    assert!(https_url.starts_with("https://127.0.0.1:"), "{listening}");
    let http_url = https_url.replacen("https://", "http://", 1);

    let run = |args: &[&str], env: &[(&str, &str)]| -> (i32, Value) {
        let out = folo_cmd().args(args).envs(env.iter().copied()).stdin(Stdio::null()).output().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let last: Value =
            stdout.lines().last().and_then(|l| serde_json::from_str(l).ok()).unwrap_or_else(|| {
                panic!(
                    "{args:?}: expected JSON output, got stdout={stdout:?} stderr={}",
                    String::from_utf8_lossy(&out.stderr)
                )
            });
        (out.status.code().unwrap(), last)
    };

    // 1. `folo bench list` succeeds over HTTPS via `--server` + `--tls-ca` and via `--data-dir` + `FOLO_TLS_CA`.
    let (code, v) =
        run(&["bench", "list", "--server", &https_url, "--token", SECRET, "--tls-ca", cert_s, "--json"], &[]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["benchmarks"], serde_json::json!([]), "{v}");

    let (code, v) = run(
        &["bench", "list", "--data-dir", worker_s, "--json"],
        &[("FOLO_TOKEN", SECRET), ("FOLO_TLS_CA", cert_s)],
    );
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["benchmarks"], serde_json::json!([]), "{v}");

    // 2. `folo label jobs` succeeds over HTTPS via `--server` + `--tls-ca` and via `--data-dir` + `FOLO_TLS_CA`.
    let (code, v) =
        run(&["label", "jobs", "--server", &https_url, "--token", SECRET, "--tls-ca", cert_s, "--json"], &[]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["jobs"], serde_json::json!([]), "{v}");

    let (code, v) = run(
        &["label", "jobs", "--data-dir", worker_s, "--json"],
        &[("FOLO_TOKEN", SECRET), ("FOLO_TLS_CA", cert_s)],
    );
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["jobs"], serde_json::json!([]), "{v}");

    // 3. Invalid or missing `--tls-ca` on `bench` / `label` fails upfront (exit 2 / 3), NOT `data_dir_in_use`.
    for cmd in [&["bench", "list"][..], &["label", "jobs"]] {
        let (code, v) = run(&[cmd[0], cmd[1], "--data-dir", worker_s, "--tls-ca", bad_ca_s, "--json"], &[]);
        assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{cmd:?}: {v}");
        let (code, v) =
            run(&[cmd[0], cmd[1], "--data-dir", worker_s, "--tls-ca", missing_ca_s, "--json"], &[]);
        assert_eq!((code, v["error"]["code"].as_str()), (3, Some("path_not_found")), "{cmd:?}: {v}");
    }

    // 4. `folo cloud connect remote` against `https://` without `FOLO_TLS_CA` hints at `FOLO_TLS_CA`.
    let (code, v) = run(
        &[
            "cloud",
            "connect",
            "remote",
            "--data-dir",
            control_s,
            "--set",
            &format!("url={https_url}"),
            "--secret",
            "token=env:FOLO_TOKEN",
            "--json",
        ],
        &[("FOLO_TOKEN", SECRET)],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("cloud_unavailable")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("FOLO_TLS_CA=<ca.pem>"), "{v}");

    // 5. Wrong CA in `FOLO_TLS_CA` reports that the cert is not trusted by `FOLO_TLS_CA`.
    let (code, v) = run(
        &[
            "cloud",
            "connect",
            "remote",
            "--data-dir",
            control_s,
            "--set",
            &format!("url={https_url}"),
            "--secret",
            "token=env:FOLO_TOKEN",
            "--json",
        ],
        &[("FOLO_TOKEN", SECRET), ("FOLO_TLS_CA", other_ca_s)],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("cloud_unavailable")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("not trusted by FOLO_TLS_CA"), "{v}");

    // 6. Missing `FOLO_TLS_CA` file fails upfront as `invalid_body` (exit 2) naming `FOLO_TLS_CA`.
    let (code, v) = run(
        &[
            "cloud",
            "connect",
            "remote",
            "--data-dir",
            control_s,
            "--set",
            &format!("url={https_url}"),
            "--secret",
            "token=env:FOLO_TOKEN",
            "--json",
        ],
        &[("FOLO_TOKEN", SECRET), ("FOLO_TLS_CA", missing_ca_s)],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("invalid_body")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("FOLO_TLS_CA"), "{v}");

    // 7. `http://` against the TLS worker tells the user to use `https://`.
    let (code, v) = run(
        &[
            "cloud",
            "connect",
            "remote",
            "--data-dir",
            control_s,
            "--set",
            &format!("url={http_url}"),
            "--secret",
            "token=env:FOLO_TOKEN",
            "--json",
        ],
        &[("FOLO_TOKEN", SECRET)],
    );
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("cloud_unavailable")), "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("use https:// instead of http://"), "{v}");

    // 8. `folo cloud connect remote`, `check`, and `offers` succeed over HTTPS when `FOLO_TLS_CA` trusts the cert.
    let (code, conn) = run(
        &[
            "cloud",
            "connect",
            "remote",
            "--data-dir",
            control_s,
            "--name",
            "tls-worker",
            "--set",
            &format!("url={https_url}"),
            "--secret",
            "token=env:FOLO_TOKEN",
            "--json",
        ],
        &[("FOLO_TOKEN", SECRET), ("FOLO_TLS_CA", cert_s)],
    );
    assert_eq!(code, 0, "{conn}");
    assert_eq!(conn["status"], "ok", "{conn}");
    let conn_id = conn["id"].as_str().unwrap().to_string();

    let (code, checked) =
        run(&["cloud", "check", &conn_id, "--data-dir", control_s, "--json"], &[("FOLO_TLS_CA", cert_s)]);
    assert_eq!(code, 0, "{checked}");
    assert_eq!(checked["status"], "ok", "{checked}");

    let (code, offer) =
        run(&["cloud", "offers", &conn_id, "--data-dir", control_s, "--json"], &[("FOLO_TLS_CA", cert_s)]);
    assert_eq!(code, 0, "{offer}");
    assert_eq!(offer["id"], "cpu", "{offer}");
}
