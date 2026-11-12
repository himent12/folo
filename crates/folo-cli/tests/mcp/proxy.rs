//! Proxy mode: forwarding to `folo serve`, and its token.

use std::io::{BufRead, BufReader};
use std::process::Stdio;

use folo_test_support::ChildGuard;
use serde_json::{Value, json};

use super::harness::{ALL_TOOLS, OLD, check_tools, checked, folo_cmd, init_old, spawn};

#[test]
fn proxy_mode_forwards_to_folo_serve() {
    let mut serve: ChildGuard = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--json"])
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut first = String::new();
    BufReader::new(serve.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let listening: Value = serde_json::from_str(&first).unwrap();
    let url = listening["url"].as_str().unwrap().to_string();

    let mut c = spawn(&["--server", &url], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let schemas = check_tools(&mut c, &ALL_TOOLS);
    let info = checked(&mut c, &schemas, "folo_info", json!({}));
    assert_eq!(info["mode"], "proxy");
    assert_eq!(info["server"], url);
    let ds = checked(&mut c, &schemas, "list_datasets", json!({ "limit": 1 }));
    assert_eq!(ds["total"], 3);
    assert_eq!(ds["next_cursor"], "o1");
    checked(&mut c, &schemas, "get_dataset", json!({ "dataset_id": "ds_7c2f" }));
    let imgs = checked(
        &mut c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": "ds_7c2f", "split": "test", "limit": 2 }),
    );
    assert_eq!(imgs["total"], 100);
    let image_id = imgs["items"][0]["image_id"].as_str().unwrap().to_string();
    let img =
        c.result("resources/read", json!({ "uri": format!("folo://datasets/ds_7c2f/images/{image_id}") }));
    assert_eq!(img["contents"][0]["mimeType"], "image/jpeg");
    let pred = checked(
        &mut c,
        &schemas,
        "predict",
        json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e", "image_id": "000001" }),
    );
    assert_eq!(pred["model_id"], "m_31aa");
    // The default read is the server's 512 px thumbnail (`?thumb=512` through the proxy); the original is 1280 wide.
    let b64 = img["contents"][0]["blob"].as_str().unwrap();
    let up = checked(&mut c, &schemas, "predict", json!({ "model_id": "m_31aa", "image_base64": b64 }));
    assert_eq!(up["image"]["width"], 512);
    let full = c.result(
        "resources/read",
        json!({ "uri": format!("folo://datasets/ds_7c2f/images/{image_id}?size=original") }),
    );
    let b64 = full["contents"][0]["blob"].as_str().unwrap();
    let up = checked(&mut c, &schemas, "predict", json!({ "model_id": "m_31aa", "image_base64": b64 }));
    assert_eq!(up["image"]["width"], 1280);
    checked(&mut c, &schemas, "val", json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e" }));
    let st = checked(&mut c, &schemas, "run_status", json!({ "run_id": "run_2b02" }));
    assert_eq!(st["status"], "stopped");
    assert!(!st["log"].as_array().unwrap().is_empty());
    checked(&mut c, &schemas, "compare_runs", json!({ "run_ids": ["run_2b02", "run_2b03"] }));
    // The cloud tools go to /api/cloud of the server.
    let offers =
        checked(&mut c, &schemas, "cloud_offers", json!({ "connection_id": "cc_4d2a", "refresh": true }));
    assert!(offers["offers"].as_array().unwrap().len() > 1);
    let instances = checked(&mut c, &schemas, "cloud_instances", json!({}));
    let seeded = instances["instances"][0]["id"].clone();
    checked(&mut c, &schemas, "cloud_terminate", json!({ "instance_id": seeded }));
    c.tool_error("cloud_terminate", json!({ "instance_id": "ci_none" }), "instance_not_found");
    let t = checked(&mut c, &schemas, "train", json!({ "dataset_id": "ds_5a1e", "epochs": 1 }));
    let stopped = checked(&mut c, &schemas, "stop_run", json!({ "run_id": t["run_id"] }));
    assert_eq!(stopped["status"], "stopping");
    // reorder_run → POST /api/runs/{id}/queue: runs started through the API share one training slot.
    let q1 = checked(&mut c, &schemas, "train", json!({ "dataset_id": "ds_5a1e", "epochs": 100 }));
    let q2 = checked(&mut c, &schemas, "train", json!({ "dataset_id": "ds_5a1e", "epochs": 100 }));
    let moved = checked(&mut c, &schemas, "reorder_run", json!({ "run_id": q2["run_id"], "action": "top" }));
    assert_eq!(moved, json!({ "run_id": q2["run_id"], "status": "queued", "queue_position": 1 }));
    let st = checked(&mut c, &schemas, "run_status", json!({ "run_id": q1["run_id"] }));
    assert_eq!(st["queue_position"], 2);
    let back = checked(&mut c, &schemas, "reorder_run", json!({ "run_id": q2["run_id"], "position": 2 }));
    assert_eq!(back["queue_position"], 2);
    c.tool_error("reorder_run", json!({ "run_id": "run_2b02", "action": "top" }), "run_not_queued");
    c.tool_error("reorder_run", json!({ "run_id": q1["run_id"] }), "invalid_arguments");
    for q in [&q1, &q2] {
        checked(&mut c, &schemas, "stop_run", json!({ "run_id": q["run_id"] }));
    }
    checked(&mut c, &schemas, "export", json!({ "model_id": "m_31aa", "format": "folo-f16" }));
    // The proxy forwards the ONNX switches to a server that announces them (`/api/system` features).
    let h =
        checked(&mut c, &schemas, "export", json!({ "model_id": "m_31aa", "format": "onnx", "half": true }));
    assert!(h["path"].as_str().unwrap().ends_with("f16.onnx"), "{h}");
    c.tool_error(
        "export",
        json!({ "model_id": "m_31aa", "format": "folo-f16", "decode": true }),
        "invalid_arguments",
    );
    c.tool_error("get_dataset", json!({ "dataset_id": "nope" }), "dataset_not_found");
    // A run started through the proxy belongs to `folo serve` and outlives this MCP session.
    let kept = checked(&mut c, &schemas, "train", json!({ "dataset_id": "ds_5a1e", "epochs": 100 }));
    c.finish();
    let mut c = spawn(&["--server", &url], None);
    init_old(&mut c);
    let st = c.ok("run_status", json!({ "run_id": kept["run_id"] }));
    assert!(st["status"] == "queued" || st["status"] == "running", "{st}");
    c.finish();
    let _ = serve.kill();
    let _ = serve.wait();
}

/// `folo mcp --server <url> --token <secret>` against `folo serve` with `FOLO_TOKEN`: the tools work with the token
/// and are `unauthorized` without it, and neither process prints it. An empty token is a usage error.
#[test]
fn proxy_mode_sends_the_token() {
    const SECRET: &str = "mcp-s3cret";
    let mut serve: ChildGuard = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--json"])
        .env("FOLO_TOKEN", SECRET)
        .env("RUST_LOG", "folo_server=debug,tower_http=trace")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .into();
    // The server's log, read as it is written (a full pipe would block the server).
    let mut stderr = serve.stderr.take().unwrap();
    let log = std::thread::spawn(move || {
        let mut log = String::new();
        std::io::Read::read_to_string(&mut stderr, &mut log).unwrap();
        log
    });
    let mut first = String::new();
    BufReader::new(serve.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let listening: Value = serde_json::from_str(&first).unwrap();
    assert_eq!(listening["token"], true);
    assert!(!first.contains(SECRET), "{first}");
    let url = listening["url"].as_str().unwrap().to_string();

    let mut c = spawn(&["--server", &url, "--token", SECRET], None);
    init_old(&mut c);
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 3);
    c.finish();
    let mut c = spawn(&["--server", &url], None);
    init_old(&mut c);
    let e = c.tool_error("list_datasets", json!({}), "unauthorized");
    assert!(e["hint"].as_str().unwrap().contains("--token"), "{e}");
    c.finish();

    let _ = serve.kill();
    let _ = serve.wait();
    let log = log.join().unwrap();
    assert!(log.contains("/api/datasets") && !log.contains(SECRET), "{log}");

    let empty = folo_cmd()
        .args(["serve", "--mock", "--port", "0", "--token", ""])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(empty.code(), Some(2));
}
