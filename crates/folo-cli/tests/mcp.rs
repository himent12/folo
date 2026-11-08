//! `folo mcp` end to end (docs/MCP.md "Tests"): spawns the binary and speaks raw JSON-RPC over stdio, for each
//! supported revision. Every line the server writes to stdout must be a JSON-RPC 2.0 message.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};

const NEW: &str = "2026-07-28";
const OLD: &str = "2025-11-25";

struct Client {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    lines: Arc<Mutex<Vec<String>>>,
    /// Notifications received while waiting for responses, oldest first.
    notes: Vec<Value>,
    next_id: u64,
    /// `Some(version)` for 2026-07-28 clients, which send their context in every request's `_meta`.
    inline: Option<&'static str>,
}

fn spawn(args: &[&str], inline: Option<&'static str>) -> Client {
    let mut child = Command::new(env!("CARGO_BIN_EXE_folo"))
        .arg("mcp")
        .args(args)
        .env("RUST_LOG", "warn")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let stdin = child.stdin.take().unwrap();
    let lines = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = channel();
    let seen = lines.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            seen.lock().unwrap().push(line.clone());
            if let Ok(v) = serde_json::from_str::<Value>(&line)
                && tx.send(v).is_err()
            {
                break;
            }
        }
    });
    Client { child, stdin, rx, lines, notes: Vec::new(), next_id: 1, inline }
}

impl Client {
    fn send(&mut self, v: &Value) {
        let line = serde_json::to_string(v).unwrap();
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn meta(&self, extra: Value) -> Value {
        let mut meta = extra;
        if let Some(v) = self.inline {
            if meta.get("io.modelcontextprotocol/protocolVersion").is_none() {
                meta["io.modelcontextprotocol/protocolVersion"] = json!(v);
            }
            meta["io.modelcontextprotocol/clientCapabilities"] = json!({});
            meta["io.modelcontextprotocol/clientInfo"] = json!({ "name": "folo-test", "version": "1" });
        }
        meta
    }

    /// Sends a request and returns its id.
    fn start(&mut self, method: &str, params: Value, meta: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut params = params;
        let meta = self.meta(meta);
        if meta.as_object().is_some_and(|m| !m.is_empty()) {
            params["_meta"] = meta;
        }
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        id
    }

    fn recv(&mut self, deadline: Instant) -> Value {
        let left = deadline.saturating_duration_since(Instant::now());
        match self.rx.recv_timeout(left) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => {
                panic!("timed out; stdout so far: {:#?}", self.lines.lock().unwrap())
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("server exited; stdout: {:#?}", self.lines.lock().unwrap())
            }
        }
    }

    /// The full response (`result` or `error`) to request `id`; notifications on the way are kept in `notes`.
    fn response(&mut self, id: u64) -> Value {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let v = self.recv(deadline);
            if v.get("id") == Some(&json!(id)) && v.get("method").is_none() {
                return v;
            }
            if v.get("method").is_some() && v.get("id").is_none() {
                self.notes.push(v);
            }
        }
    }

    fn request_with(&mut self, method: &str, params: Value, meta: Value) -> Value {
        let id = self.start(method, params, meta);
        self.response(id)
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.request_with(method, params, json!({}))
    }

    fn result(&mut self, method: &str, params: Value) -> Value {
        let r = self.request(method, params);
        assert!(r.get("error").is_none(), "{method} failed: {r}");
        r["result"].clone()
    }

    /// Calls a tool and returns its result (not asserting success).
    fn call(&mut self, name: &str, args: Value) -> Value {
        self.result("tools/call", json!({ "name": name, "arguments": args }))
    }

    /// Calls a tool that must succeed and returns its `structuredContent`.
    fn ok(&mut self, name: &str, args: Value) -> Value {
        let r = self.call(name, args.clone());
        assert_ne!(r["isError"], json!(true), "{name} {args} failed: {r}");
        let content = r["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].as_str().unwrap().len() < 400, "summary is short");
        assert!(
            serde_json::to_vec(&r["structuredContent"]).unwrap().len() <= 32 * 1024,
            "{name} result over 32 KB"
        );
        r["structuredContent"].clone()
    }

    /// Calls a tool that must fail with `code` and returns the error object.
    fn tool_error(&mut self, name: &str, args: Value, code: &str) -> Value {
        let r = self.call(name, args.clone());
        assert_eq!(r["isError"], json!(true), "{name} {args} should fail: {r}");
        assert!(r.get("structuredContent").is_none());
        let content = r["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        let e: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(e["code"], code, "{e}");
        assert!(e["message"].is_string() && e.get("hint").is_some());
        e
    }

    /// Waits for a notification matching `pred` (already received ones first).
    fn notification(&mut self, timeout: Duration, pred: impl Fn(&Value) -> bool) -> Value {
        if let Some(i) = self.notes.iter().position(&pred) {
            return self.notes.remove(i);
        }
        let deadline = Instant::now() + timeout;
        loop {
            let v = self.recv(deadline);
            if v.get("id").is_none() && pred(&v) {
                return v;
            }
            if v.get("id").is_none() {
                self.notes.push(v);
            }
        }
    }

    /// Closes stdin, waits for the process and checks that stdout carried only JSON-RPC 2.0 messages.
    fn finish(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.child.try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline, "folo mcp did not exit after stdin closed");
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(100));
        for line in self.lines.lock().unwrap().iter() {
            let v: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("stdout line is not JSON ({e}): {line}"));
            assert_eq!(v["jsonrpc"], "2.0", "{line}");
            let response = v.get("id").is_some() && (v.get("result").is_some() || v.get("error").is_some());
            let message = v.get("method").is_some();
            assert!(response || message, "not a JSON-RPC message: {line}");
            assert!(!line.contains('\n'));
        }
    }
}

fn validate(schema: &Value, instance: &Value, what: &str) {
    let v = jsonschema::validator_for(schema).unwrap_or_else(|e| panic!("{what}: bad schema: {e}"));
    if let Err(e) = v.validate(instance) {
        panic!("{what}: result does not match the outputSchema: {e}\n{instance}");
    }
}

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("folo-mcp-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

const ALL_TOOLS: [&str; 16] = [
    "folo_info",
    "list_datasets",
    "get_dataset",
    "list_dataset_images",
    "register_dataset",
    "create_synthetic_dataset",
    "list_models",
    "train",
    "run_status",
    "list_runs",
    "stop_run",
    "val",
    "predict",
    "predict_batch",
    "export",
    "compare_runs",
];

/// `tools/list` with schema checks; returns `name -> outputSchema`.
fn check_tools(c: &mut Client, expected: &[&str]) -> std::collections::HashMap<String, Value> {
    let tools = c.result("tools/list", json!({}))["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, expected);
    let mut out = std::collections::HashMap::new();
    for t in &tools {
        assert!(t["description"].as_str().unwrap().len() > 20);
        for key in ["inputSchema", "outputSchema"] {
            assert_eq!(t[key]["type"], "object", "{} {key}", t["name"]);
            jsonschema::validator_for(&t[key]).unwrap_or_else(|e| panic!("{} {key}: {e}", t["name"]));
        }
        out.insert(t["name"].as_str().unwrap().to_string(), t["outputSchema"].clone());
    }
    if let Some(train) = tools.iter().find(|t| t["name"] == "train") {
        assert_eq!(train["inputSchema"]["required"], json!(["dataset_id"]));
        assert!(train["inputSchema"]["properties"]["resume_run_id"].is_object());
        assert_eq!(train["annotations"]["readOnlyHint"], false);
    }
    out
}

/// Calls a tool and validates its structured result against the tool's outputSchema.
fn checked(
    c: &mut Client,
    schemas: &std::collections::HashMap<String, Value>,
    name: &str,
    args: Value,
) -> Value {
    let v = c.ok(name, args);
    validate(&schemas[name], &v, name);
    v
}

/// Everything after the handshake / discovery, for one revision.
fn full_flow(c: &mut Client, data_dir: &std::path::Path) {
    let schemas = check_tools(c, &ALL_TOOLS);
    let info = checked(c, &schemas, "folo_info", json!({}));
    assert_eq!(info["mode"], "mock");
    assert_eq!(info["scales"].as_array().unwrap().len(), 5);

    // Resources: listing, templates, docs and an image blob.
    let res = c.result("resources/list", json!({}))["resources"].as_array().unwrap().clone();
    let uris: Vec<&str> = res.iter().map(|r| r["uri"].as_str().unwrap()).collect();
    for want in
        ["folo://datasets/ds_5a1e", "folo://runs/run_3c02", "folo://runs/run_3c02/metrics", "folo://docs/MCP"]
    {
        assert!(uris.contains(&want), "{want} missing from {uris:?}");
    }
    let templates =
        c.result("resources/templates/list", json!({}))["resourceTemplates"].as_array().unwrap().clone();
    assert!(templates.iter().any(|t| t["uriTemplate"] == "folo://datasets/{id}/images/{image_id}"));
    let doc = c.result("resources/read", json!({ "uri": "folo://docs/API" }));
    assert_eq!(doc["contents"][0]["mimeType"], "text/markdown");
    assert!(doc["contents"][0]["text"].as_str().unwrap().contains("/api/datasets"));
    let page = checked(
        c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": "ds_5a1e", "split": "val", "limit": 3, "include_boxes": true }),
    );
    assert_eq!(page["items"].as_array().unwrap().len(), 3);
    assert_eq!(page["total"], 200);
    assert_eq!(page["next_cursor"], "o3");
    let next = checked(
        c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": "ds_5a1e", "cursor": "o3", "limit": 1 }),
    );
    assert!(next["items"][0].get("boxes").is_none());
    let image_id = page["items"][0]["image_id"].as_str().unwrap().to_string();
    let img =
        c.result("resources/read", json!({ "uri": format!("folo://datasets/ds_5a1e/images/{image_id}") }));
    let blob = &img["contents"][0];
    assert_eq!(blob["mimeType"], "image/jpeg");
    let jpeg = base64::engine::general_purpose::STANDARD.decode(blob["blob"].as_str().unwrap()).unwrap();
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    let missing = c.request("resources/read", json!({ "uri": "folo://datasets/nope" }));
    assert!(missing["error"]["code"].is_i64(), "{missing}");
    let samples = c.result("resources/read", json!({ "uri": "folo://runs/run_2b02/samples/1" }));
    assert_eq!(samples["contents"][0]["mimeType"], "image/jpeg");

    // Prompts: retrieval returns messages only.
    let prompts = c.result("prompts/list", json!({}))["prompts"].as_array().unwrap().clone();
    let names: Vec<&str> = prompts.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["train_on_folder", "diagnose_run"]);
    let p = c.result("prompts/get", json!({ "name": "diagnose_run", "arguments": { "run_id": "run_2b02" } }));
    let text = p["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(text.contains("run_2b02") && text.contains("TrainConfig"));
    let p = c.result(
        "prompts/get",
        json!({ "name": "train_on_folder", "arguments": { "folder_path": "D:/data/cats" } }),
    );
    assert!(p["messages"][0]["content"]["text"].as_str().unwrap().contains("register_dataset"));
    let bad = c.request("prompts/get", json!({ "name": "train_on_folder", "arguments": {} }));
    assert_eq!(bad["error"]["code"], -32602);

    // create_synthetic_dataset with progress → train → subscribe / run_status → predict.
    let id = c.start(
        "tools/call",
        json!({ "name": "create_synthetic_dataset", "arguments": { "name": "tiny", "images": 20, "imgsz": 64 } }),
        json!({ "progressToken": "synth-1" }),
    );
    let r = c.response(id);
    let progress: Vec<Value> = c
        .notes
        .iter()
        .filter(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == "synth-1")
        .cloned()
        .collect();
    assert!(!progress.is_empty(), "no progress for a request with a progressToken");
    let values: Vec<f64> = progress.iter().map(|n| n["params"]["progress"].as_f64().unwrap()).collect();
    assert!(values.windows(2).all(|w| w[0] < w[1]), "progress must increase: {values:?}");
    assert_eq!(*values.last().unwrap(), 20.0);
    c.notes.retain(|n| n["method"] != "notifications/progress");
    let ds = r["result"]["structuredContent"].clone();
    validate(&schemas["create_synthetic_dataset"], &ds, "create_synthetic_dataset");
    let ds_id = ds["id"].as_str().unwrap().to_string();
    assert_eq!(ds["splits"]["train"], 16);
    // No progressToken, no progress.
    c.ok("create_synthetic_dataset", json!({ "name": "tiny2", "images": 5, "imgsz": 64 }));
    assert!(c.notes.iter().all(|n| n["method"] != "notifications/progress"));

    let detail = checked(c, &schemas, "get_dataset", json!({ "dataset_id": ds_id }));
    assert!(detail["stats"]["class_counts"].is_array());
    assert_eq!(detail["issues"], json!([]));
    // traffic-cams has broken labels, which get_dataset lists as issues.
    let cams = checked(c, &schemas, "get_dataset", json!({ "dataset_id": "ds_7c2f" }));
    assert!(cams["issues_total"].as_u64().unwrap() >= 1, "{cams}");
    assert!(cams["issues"][0]["kind"].is_string());

    let t = checked(
        c,
        &schemas,
        "train",
        json!({ "dataset_id": ds_id, "epochs": 2, "batch": 4, "imgsz": 64, "name": "tiny-n" }),
    );
    let run_id = t["run_id"].as_str().unwrap().to_string();
    assert_eq!(t["status"], "queued");
    let run_uri = format!("folo://runs/{run_id}");
    let subscribed = if c.inline.is_some() {
        let listen = c.start(
            "subscriptions/listen",
            json!({ "notifications": { "resourceSubscriptions": [run_uri, "folo://nope"] } }),
            json!({}),
        );
        let ack = c.notification(Duration::from_secs(10), |n| {
            n["method"] == "notifications/subscriptions/acknowledged"
        });
        assert_eq!(ack["params"]["notifications"]["resourceSubscriptions"], json!([run_uri]), "{ack}");
        Some(listen)
    } else {
        let r = c.result("resources/subscribe", json!({ "uri": run_uri }));
        assert!(r.is_object());
        let bad = c.request("resources/subscribe", json!({ "uri": "folo://runs/nope" }));
        assert!(bad.get("error").is_some(), "{bad}");
        None
    };
    let update =
        c.notification(Duration::from_secs(20), |n| n["method"] == "notifications/resources/updated");
    assert_eq!(update["params"]["uri"], run_uri);
    if let Some(listen) = subscribed {
        assert_eq!(
            update["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"],
            json!(listen),
            "{update}"
        );
        c.send(&json!({ "jsonrpc": "2.0", "method": "notifications/cancelled", "params": { "requestId": listen } }));
    } else {
        c.result("resources/unsubscribe", json!({ "uri": run_uri }));
    }
    let snap = c.result("resources/read", json!({ "uri": run_uri }));
    let snap: Value = serde_json::from_str(snap["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(snap["run"]["id"], run_id);
    assert!(snap["revision"].as_u64().unwrap() >= 1);

    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        let s = checked(c, &schemas, "run_status", json!({ "run_id": run_id, "log_lines": 5 }));
        if s["status"] == "completed" {
            break s;
        }
        assert!(Instant::now() < deadline, "run did not complete: {s}");
        std::thread::sleep(Duration::from_millis(500));
    };
    assert_eq!(status["last_epoch"]["epoch"], 2);
    assert!(status["log"].as_array().unwrap().len() <= 5);
    let model_id = status["model_id"].as_str().unwrap().to_string();

    let pred = checked(
        c,
        &schemas,
        "predict",
        json!({ "model_id": model_id, "dataset_id": ds_id, "image_id": "000017", "conf": 0.1 }),
    );
    assert!(pred["boxes"].is_array());
    assert_eq!(pred["image"]["width"], 64);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    checked(c, &schemas, "predict", json!({ "model_id": model_id, "image_base64": b64, "max_det": 3 }));

    // The rest of the table.
    std::fs::write(data_dir.join("a.jpg"), &jpeg).unwrap();
    std::fs::write(data_dir.join("b.jpg"), &jpeg).unwrap();
    std::fs::write(data_dir.join("c.jpg"), b"not an image").unwrap();
    checked(c, &schemas, "predict", json!({ "model_id": model_id, "image_path": "a.jpg" }));
    let id = c.start(
        "tools/call",
        json!({ "name": "predict_batch", "arguments": { "model_id": model_id, "glob": "*.jpg" } }),
        json!({ "progressToken": 7 }),
    );
    let r = c.response(id)["result"].clone();
    assert_ne!(r["isError"], json!(true), "{r}");
    let batch = r["structuredContent"].clone();
    validate(&schemas["predict_batch"], &batch, "predict_batch");
    assert_eq!(batch["processed"], 2);
    assert_eq!(batch["failed"][0]["code"], "image_decode_failed");
    let link =
        r["content"].as_array().unwrap().iter().find(|c| c["type"] == "resource_link").unwrap().clone();
    assert_eq!(link["uri"], batch["details_uri"]);
    assert!(
        c.notes.iter().any(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == 7)
    );
    c.notes.retain(|n| n["method"] != "notifications/progress");
    let details = c.result("resources/read", json!({ "uri": batch["details_uri"] }));
    let details: Value = serde_json::from_str(details["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(details["images"].as_array().unwrap().len(), 3);

    let val = checked(c, &schemas, "val", json!({ "model_id": model_id, "dataset_id": ds_id }));
    assert_eq!(val["images"], 4);
    checked(c, &schemas, "export", json!({ "model_id": model_id, "format": "onnx" }));
    let models = checked(c, &schemas, "list_models", json!({ "limit": 2 }));
    assert_eq!(models["items"].as_array().unwrap().len(), 2);
    let runs = checked(c, &schemas, "list_runs", json!({ "status": "completed" }));
    assert!(runs["items"].as_array().unwrap().iter().all(|r| r["status"] == "completed"));
    let cmp = checked(c, &schemas, "compare_runs", json!({ "run_ids": [run_id, "run_2b02"] }));
    assert!(cmp["config_diff"]["dataset_id"].is_array());
    checked(c, &schemas, "list_datasets", json!({}));
    let live = checked(c, &schemas, "list_runs", json!({ "status": "running", "limit": 1 }));
    if let Some(r) = live["items"].as_array().unwrap().first() {
        checked(c, &schemas, "stop_run", json!({ "run_id": r["id"] }));
    }

    // Tool errors (isError) vs protocol errors (JSON-RPC).
    c.tool_error("get_dataset", json!({ "dataset_id": "nope" }), "dataset_not_found");
    c.tool_error("list_datasets", json!({ "limit": "many" }), "invalid_arguments");
    c.tool_error("get_dataset", json!({}), "invalid_arguments");
    c.tool_error("list_datasets", json!({ "limit": 500 }), "invalid_arguments");
    c.tool_error("predict", json!({ "model_id": model_id }), "invalid_arguments");
    c.tool_error("compare_runs", json!({ "run_ids": [run_id] }), "invalid_arguments");
    let e = c.tool_error("train", json!({ "dataset_id": ds_id, "resume_run_id": run_id }), "not_implemented");
    assert!(e["hint"].as_str().unwrap().contains("--mock"));
    let unknown = c.request("tools/call", json!({ "name": "no_such_tool", "arguments": {} }));
    assert_eq!(unknown["error"]["code"], -32602, "{unknown}");
    let method = c.request("no/such/method", json!({}));
    assert_eq!(method["error"]["code"], -32601, "{method}");
    // Unparsable input is ignored (as by every official SDK); the server keeps working.
    writeln!(c.stdin, "this is not json").unwrap();
    c.ok("folo_info", json!({}));
}

#[test]
fn revision_2025_11_25() {
    let dir = tmp_dir("old");
    let mut c = spawn(&["--mock", "--data-dir", dir.to_str().unwrap()], None);
    let init = c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "folo-test", "version": "1" } }),
    );
    assert_eq!(init["protocolVersion"], OLD);
    assert_eq!(init["serverInfo"]["name"], "folo");
    let caps = &init["capabilities"];
    assert_eq!(caps["tools"]["listChanged"], false);
    assert_eq!(caps["resources"]["subscribe"], true);
    assert!(caps["prompts"].is_object() && caps["logging"].is_object());
    c.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
    // A 2026-only method is not part of this revision.
    let listen = c.request("subscriptions/listen", json!({ "notifications": {} }));
    assert_eq!(listen["error"]["code"], -32601, "{listen}");
    full_flow(&mut c, &dir);
    c.finish();
}

#[test]
fn revision_2026_07_28() {
    let dir = tmp_dir("new");
    let mut c = spawn(&["--mock", "--data-dir", dir.to_str().unwrap()], Some(NEW));
    // A direct tool call, without discovery or a handshake.
    let info = c.ok("folo_info", json!({}));
    assert_eq!(info["mode"], "mock");
    let disc = c.result("server/discover", json!({}));
    assert!(disc["supportedVersions"].as_array().unwrap().contains(&json!(NEW)), "{disc}");
    assert!(disc["supportedVersions"].as_array().unwrap().contains(&json!(OLD)), "{disc}");
    assert_eq!(disc["capabilities"]["resources"]["subscribe"], true);
    // Unsupported version in a request's metadata.
    let id = c.start(
        "tools/call",
        json!({ "name": "folo_info", "arguments": {} }),
        json!({ "io.modelcontextprotocol/protocolVersion": "2099-01-01" }),
    );
    let bad = c.response(id);
    assert!(bad["error"]["code"].is_i64(), "{bad}");
    // The legacy subscription method is not part of this revision.
    let legacy = c.request("resources/subscribe", json!({ "uri": "folo://runs/run_3c02" }));
    assert_eq!(legacy["error"]["code"], -32601, "{legacy}");
    full_flow(&mut c, &dir);
    c.finish();
}

#[test]
fn unsupported_version_before_anything_else() {
    let mut c = spawn(&["--mock"], Some("2099-01-01"));
    let r = c.request("tools/call", json!({ "name": "folo_info", "arguments": {} }));
    assert!(r["error"]["code"].is_i64(), "{r}");
    assert!(r["error"]["data"].to_string().contains(NEW), "the error lists the supported versions: {r}");
    c.finish();
}

#[test]
fn read_only_hides_and_rejects_mutating_tools() {
    let mut c = spawn(&["--mock", "--read-only"], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let listed = [
        "folo_info",
        "list_datasets",
        "get_dataset",
        "list_dataset_images",
        "list_models",
        "run_status",
        "list_runs",
        "val",
        "predict",
        "predict_batch",
        "compare_runs",
    ];
    check_tools(&mut c, &listed);
    for (tool, args) in [
        ("train", json!({ "dataset_id": "ds_5a1e" })),
        ("create_synthetic_dataset", json!({ "name": "x" })),
        ("register_dataset", json!({ "path": "." })),
        ("stop_run", json!({ "run_id": "run_3c02" })),
        ("export", json!({ "model_id": "m_31aa", "format": "onnx" })),
    ] {
        c.tool_error(tool, args, "read_only");
    }
    assert_eq!(c.ok("folo_info", json!({}))["read_only"], true);
    let prompts = c.result("prompts/list", json!({}))["prompts"].as_array().unwrap().clone();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0]["name"], "diagnose_run");
    c.finish();
}

#[test]
fn confine_rejects_paths_outside_the_data_dir() {
    let root = tmp_dir("confine");
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(root.join("outside.jpg"), b"x").unwrap();
    let mut c = spawn(&["--mock", "--confine", "--data-dir", data.to_str().unwrap()], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let outside = root.join("outside.jpg");
    c.tool_error("predict", json!({ "model_id": "m_31aa", "image_path": outside }), "path_outside_data_dir");
    c.tool_error(
        "predict",
        json!({ "model_id": "m_31aa", "image_path": "../outside.jpg" }),
        "path_outside_data_dir",
    );
    c.tool_error(
        "predict_batch",
        json!({ "model_id": "m_31aa", "glob": "../*.jpg" }),
        "path_outside_data_dir",
    );
    c.tool_error("register_dataset", json!({ "path": root }), "path_outside_data_dir");
    c.finish();
}

fn init_old(c: &mut Client) {
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
}

/// A one-image YOLO dataset: `<dir>/images/train/a.jpg` + `<dir>/labels/train/a.txt`.
fn yolo_files(dir: &std::path::Path) {
    for sub in ["images/train", "labels/train"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    image::RgbImage::from_pixel(32, 24, image::Rgb([200, 40, 40]))
        .save(dir.join("images/train/a.jpg"))
        .unwrap();
    std::fs::write(dir.join("labels/train/a.txt"), "0 0.5 0.5 0.25 0.25\n").unwrap();
}

#[test]
fn confine_checks_what_a_dataset_reads() {
    let root = tmp_dir("confine-ds");
    let data = root.join("data");
    let outside = root.join("outside");
    yolo_files(&outside);
    // Inside the data dir, but its data.yaml reads the images from outside.
    let sneaky = data.join("sneaky");
    std::fs::create_dir_all(&sneaky).unwrap();
    let yaml =
        format!("path: {}\ntrain: images/train\nval: images/train\nnames: [thing]\n", outside.display());
    std::fs::write(sneaky.join("data.yaml"), yaml.replace('\\', "/")).unwrap();
    let honest = data.join("honest");
    yolo_files(&honest);

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error("register_dataset", json!({ "path": "sneaky" }), "path_outside_data_dir");
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0, "the rejected dataset was unregistered again");
    // An image list inside the root naming outside images, and a list file outside the root.
    let listed = data.join("listed");
    std::fs::create_dir_all(&listed).unwrap();
    let outside_img = outside.join("images/train/a.jpg").to_string_lossy().replace('\\', "/");
    std::fs::write(listed.join("train.txt"), format!("{outside_img}\n")).unwrap();
    std::fs::write(listed.join("data.yaml"), "train: train.txt\nval: train.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "listed" }), "path_outside_data_dir");
    let far = data.join("far");
    std::fs::create_dir_all(&far).unwrap();
    std::fs::write(root.join("list.txt"), "images/train/a.jpg\n").unwrap();
    let list = root.join("list.txt").to_string_lossy().replace('\\', "/");
    std::fs::write(far.join("data.yaml"), format!("train: {list}\nnames: [thing]\n")).unwrap();
    c.tool_error("register_dataset", json!({ "path": "far" }), "path_outside_data_dir");
    // A list outside whose name contains a comma, naming an image inside: the whole name must be checked.
    let comma = data.join("comma");
    std::fs::create_dir_all(&comma).unwrap();
    std::fs::write(root.join("secret,list.txt"), "../honest/images/train/a.jpg\n").unwrap();
    std::fs::write(comma.join("data.yaml"), "train: ../../secret,list.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "comma" }), "path_outside_data_dir");
    // An indented key (a YAML form the precheck's parser need not understand) naming an outside list.
    let indented = data.join("indented");
    std::fs::create_dir_all(&indented).unwrap();
    std::fs::write(indented.join("data.yaml"), "  train: ../../list.txt\nnames: [thing]\n").unwrap();
    // (For the reader this one is not valid YAML at all; either way nothing outside is read.)
    c.tool_error("register_dataset", json!({ "path": "indented" }), "invalid_dataset");
    // A folder inside the root that is a link (a junction on Windows) to an outside folder.
    let linked = data.join("linked");
    std::fs::create_dir_all(linked.join("labels/train")).unwrap();
    assert!(
        make_dir_link(&outside.join("images"), &linked.join("images")),
        "could not create a directory link"
    );
    c.tool_error("register_dataset", json!({ "path": "linked" }), "path_outside_data_dir");
    remove_dir_link(&linked.join("images"));
    // YAML forms a hand-written parser might miss: a tab after the colon, an escaped scalar, another manifest name.
    // The reader itself refuses the outside list, so the exact syntax does not matter.
    for (name, yaml) in [
        ("tabbed", "train:\t../../list.txt\nnames: [thing]\n"),
        ("escaped", "train: \"../../l\\u0069st.txt\"\nnames: [thing]\n"),
    ] {
        let d = data.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("data.yaml"), yaml).unwrap();
        c.tool_error("register_dataset", json!({ "path": name }), "path_outside_data_dir");
    }
    let alt = data.join("alt");
    std::fs::create_dir_all(&alt).unwrap();
    std::fs::write(alt.join("dataset.yml"), "train: ../../list.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "alt" }), "path_outside_data_dir");
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0);
    // Class names as a flow map are not paths (and are not valid file names on Windows).
    let mapped = data.join("mapped");
    yolo_files(&mapped);
    std::fs::write(
        mapped.join("data.yaml"),
        "train: images/train\nval: images/train\nnames: {0: cat, 1: dog}\n",
    )
    .unwrap();
    let m = c.ok("register_dataset", json!({ "path": "mapped" }));
    assert_eq!(m["classes"], json!(["cat", "dog"]));
    // Edited after registration to read outside: the re-index (on the next request after ~5 s) cannot read it.
    let mapped_id = m["id"].as_str().unwrap().to_string();
    let outside_dir = outside.join("images/train").to_string_lossy().replace('\\', "/");
    std::fs::write(
        mapped.join("data.yaml"),
        format!("train: {outside_dir}\nval: {outside_dir}\nnames: [cat]\n"),
    )
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5500));
    let after = c.call("list_dataset_images", json!({ "dataset_id": mapped_id }));
    if after["isError"] == true {
        // The re-index refused the outside folder and failed.
        let e: Value = serde_json::from_str(after["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(e["code"], "path_outside_data_dir", "{e}");
    } else {
        let outside_bytes = std::fs::read(outside.join("images/train/a.jpg")).unwrap();
        for im in after["structuredContent"]["items"].as_array().unwrap() {
            let uri = format!("folo://datasets/{mapped_id}/images/{}", im["image_id"].as_str().unwrap());
            let r = c.request("resources/read", json!({ "uri": uri }));
            if let Some(blob) = r["result"]["contents"][0]["blob"].as_str() {
                let bytes = base64::engine::general_purpose::STANDARD.decode(blob).unwrap();
                assert_ne!(bytes, outside_bytes, "an outside image was served: {im}");
            }
        }
    }
    let ds = c.ok("register_dataset", json!({ "path": "honest" }));
    let id = ds["id"].as_str().unwrap().to_string();
    let imgs = c.ok("list_dataset_images", json!({ "dataset_id": id }));
    let image_id = imgs["items"][0]["image_id"].as_str().unwrap().to_string();
    let img = c.result("resources/read", json!({ "uri": format!("folo://datasets/{id}/images/{image_id}") }));
    assert_eq!(img["contents"][0]["mimeType"], "image/jpeg");
    c.finish();

    // A dataset registered without --confine (persisted by the real store) is invisible with it.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let out_ds = c.ok("register_dataset", json!({ "path": outside }));
    let out_id = out_ds["id"].as_str().unwrap().to_string();
    c.finish();
    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let list = c.ok("list_datasets", json!({}));
    let ids: Vec<&str> =
        list["items"].as_array().unwrap().iter().map(|d| d["id"].as_str().unwrap()).collect();
    // `mapped` now reads outside (edited above) and is hidden too.
    assert_eq!(ids, [id.as_str()]);
    c.tool_error("get_dataset", json!({ "dataset_id": out_id }), "path_outside_data_dir");
    c.tool_error("list_dataset_images", json!({ "dataset_id": out_id }), "path_outside_data_dir");
    let read = c.request("resources/read", json!({ "uri": format!("folo://datasets/{out_id}/images/0") }));
    assert!(read.get("error").is_some(), "{read}");
    let res = c.result("resources/list", json!({}))["resources"].clone();
    assert!(!res.to_string().contains(&out_id));
    c.finish();
}

#[test]
fn confine_rebuilds_persisted_indexes() {
    // A COCO dataset whose images are inside but whose annotations folder links outside, registered without
    // --confine (so its index is cached), must not be trusted from that cache once --confine is on.
    let root = tmp_dir("confine-coco");
    let data = root.join("data");
    let ds = data.join("coco");
    std::fs::create_dir_all(ds.join("images")).unwrap();
    image::RgbImage::from_pixel(32, 24, image::Rgb([9, 9, 9])).save(ds.join("images/a.jpg")).unwrap();
    let ann = root.join("outside-ann");
    std::fs::create_dir_all(&ann).unwrap();
    let coco = json!({
        "images": [{ "id": 1, "file_name": "a.jpg", "width": 32, "height": 24 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "secretcat" }],
    });
    std::fs::write(ann.join("instances_train.json"), coco.to_string()).unwrap();
    assert!(make_dir_link(&ann, &ds.join("annotations")), "could not create a directory link");
    // And a YOLO dataset whose labels folder links outside (it still indexes under the guard, minus the labels).
    let y = data.join("yolo");
    std::fs::create_dir_all(y.join("images/train")).unwrap();
    image::RgbImage::from_pixel(32, 24, image::Rgb([9, 9, 9])).save(y.join("images/train/a.jpg")).unwrap();
    let out_labels = root.join("outside-labels");
    std::fs::create_dir_all(out_labels.join("train")).unwrap();
    std::fs::write(
        out_labels.join("train/a.txt"),
        "0 0.5 0.5 0.25 0.25
",
    )
    .unwrap();
    assert!(make_dir_link(&out_labels, &y.join("labels")));
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let reg = c.ok("register_dataset", json!({ "path": "coco" }));
    assert_eq!(reg["classes"], json!(["secretcat"]));
    let id = reg["id"].as_str().unwrap().to_string();
    let yreg = c.ok("register_dataset", json!({ "path": "yolo" }));
    assert_eq!(yreg["stats"]["boxes"], 1);
    let yid = yreg["id"].as_str().unwrap().to_string();
    c.finish();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let list = c.ok("list_datasets", json!({}));
    assert!(!list.to_string().contains("secretcat"), "{list}");
    assert_eq!(list["total"], 0, "{list}");
    c.tool_error("get_dataset", json!({ "dataset_id": yid }), "path_outside_data_dir");
    let r = c.call("get_dataset", json!({ "dataset_id": id }));
    assert_eq!(r["isError"], true);
    assert!(!r.to_string().contains("secretcat"), "{r}");
    c.finish();

    // The confined session's partial index was not cached: without --confine the dataset is whole again.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let d = c.ok("get_dataset", json!({ "dataset_id": yid }));
    assert_eq!(d["stats"]["boxes"], 1, "{d}");
    let d = c.ok("get_dataset", json!({ "dataset_id": id }));
    assert_eq!(d["classes"], json!(["secretcat"]), "{d}");
    assert_eq!(d["splits"]["train"].as_u64().unwrap() + d["splits"]["val"].as_u64().unwrap(), 1, "{d}");
    c.finish();
}

#[test]
fn confine_keeps_yolo_labels_inside() {
    // With the data dir at <tmp>/images, the label of <tmp>/images/ds/a.jpg would be <tmp>/labels/ds/a.txt.
    let root = tmp_dir("confine-labels");
    let data = root.join("images");
    let ds = data.join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).save(ds.join("a.jpg")).unwrap();
    std::fs::write(ds.join("list.txt"), "a.jpg\n").unwrap();
    std::fs::write(ds.join("data.yaml"), "train: list.txt\nnames: [secret]\n").unwrap();
    std::fs::create_dir_all(root.join("labels/ds")).unwrap();
    std::fs::write(root.join("labels/ds/a.txt"), "0 0.5 0.5 0.9 0.9\n").unwrap();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error("register_dataset", json!({ "path": "ds" }), "path_outside_data_dir");
    c.finish();

    // Registered without --confine (persisted), the dataset is rejected with it: its labels are outside.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let id = c.ok("register_dataset", json!({ "path": "ds" }))["id"].as_str().unwrap().to_string();
    c.finish();
    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error(
        "list_dataset_images",
        json!({ "dataset_id": id, "include_boxes": true }),
        "path_outside_data_dir",
    );
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0);
    c.finish();
}

/// A directory link at `link` to `target`: a junction on Windows (no privileges needed), a symlink elsewhere.
fn make_dir_link(target: &std::path::Path, link: &std::path::Path) -> bool {
    #[cfg(windows)]
    {
        Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .stdout(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
}

fn remove_dir_link(link: &std::path::Path) {
    // A junction (Windows) is removed like an empty directory, a symlink like a file.
    if std::fs::remove_dir(link).is_err() {
        std::fs::remove_file(link).unwrap();
    }
}

#[test]
fn results_stay_under_the_size_limit() {
    let mut c = spawn(&["--mock"], None);
    init_old(&mut c);
    let long = "x".repeat(200);
    c.tool_error("create_synthetic_dataset", json!({ "name": long, "images": 5 }), "invalid_arguments");
    let classes: Vec<String> = (0..81).map(|i| format!("c{i}")).collect();
    c.tool_error(
        "create_synthetic_dataset",
        json!({ "name": "n", "images": 5, "classes": classes }),
        "invalid_arguments",
    );
    c.tool_error("train", json!({ "dataset_id": "ds_5a1e", "name": long }), "invalid_arguments");
    // Error results are bounded too, even when the arguments are huge.
    let huge = "q".repeat(40_000);
    let r = c.call("list_datasets", json!({ "cursor": huge }));
    assert_eq!(r["isError"], true);
    assert!(serde_json::to_vec(&r).unwrap().len() < 4096, "{}", serde_json::to_vec(&r).unwrap().len());
    let r = c.call("get_dataset", json!({ "dataset_id": huge }));
    assert!(serde_json::to_vec(&r).unwrap().len() < 4096);
    // The largest allowed dataset still fits, inline and in a page of 200.
    let classes: Vec<String> = (0..80).map(|i| format!("{i:0>64}")).collect();
    let name = "y".repeat(128);
    for _ in 0..3 {
        c.ok(
            "create_synthetic_dataset",
            json!({ "name": name, "images": 5, "imgsz": 64, "classes": classes }),
        );
    }
    let page = c.ok("list_datasets", json!({ "limit": 200 }));
    assert_eq!(page["total"], 6);
    let mut seen = page["items"].as_array().unwrap().len();
    let mut cursor = page["next_cursor"].clone();
    while let Some(cur) = cursor.as_str().map(str::to_string) {
        let p = c.ok("list_datasets", json!({ "cursor": cur, "limit": 200 }));
        seen += p["items"].as_array().unwrap().len();
        cursor = p["next_cursor"].clone();
    }
    assert_eq!(seen, 6);
    c.finish();
}

#[test]
fn real_store_reports_not_implemented() {
    let dir = tmp_dir("real");
    let mut c = spawn(&["--data-dir", dir.to_str().unwrap()], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let info = c.ok("folo_info", json!({}));
    assert_eq!(info["mode"], "embedded");
    let e = c.tool_error("val", json!({ "model_id": "m_1", "dataset_id": "ds_1" }), "not_implemented");
    assert!(e["hint"].as_str().unwrap().contains("folo mcp --mock"));
    assert_eq!(c.ok("list_runs", json!({}))["total"], 0);
    c.finish();
}

#[test]
fn proxy_mode_forwards_to_folo_serve() {
    let mut serve = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["serve", "--mock", "--port", "0", "--json"])
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
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
    let b64 = img["contents"][0]["blob"].as_str().unwrap();
    let up = checked(&mut c, &schemas, "predict", json!({ "model_id": "m_31aa", "image_base64": b64 }));
    assert_eq!(up["image"]["width"], 1280);
    checked(&mut c, &schemas, "val", json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e" }));
    let st = checked(&mut c, &schemas, "run_status", json!({ "run_id": "run_2b02" }));
    assert_eq!(st["status"], "stopped");
    assert!(!st["log"].as_array().unwrap().is_empty());
    checked(&mut c, &schemas, "compare_runs", json!({ "run_ids": ["run_2b02", "run_2b03"] }));
    let t = checked(&mut c, &schemas, "train", json!({ "dataset_id": "ds_5a1e", "epochs": 1 }));
    let stopped = checked(&mut c, &schemas, "stop_run", json!({ "run_id": t["run_id"] }));
    assert_eq!(stopped["status"], "stopping");
    checked(&mut c, &schemas, "export", json!({ "model_id": "m_31aa", "format": "folo-f16" }));
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

    // A server that is not there is a tool error, not a crash.
    let mut c = spawn(&["--server", "http://127.0.0.1:9"], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    c.tool_error("list_datasets", json!({}), "server_unreachable");
    c.finish();
}
