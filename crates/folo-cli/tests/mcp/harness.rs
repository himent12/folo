//! The harness: a JSON-RPC client over a spawned `folo mcp`, schema checks, and helpers for test
//! folders, YOLO files and directory links.

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_test_support::{ChildGuard, TestDir};
use serde_json::{Value, json};

pub(super) const NEW: &str = "2026-07-28";
pub(super) const OLD: &str = "2025-11-25";

/// How long [`Client::finish`] lets `folo mcp` take to exit once stdin closes: only a hang gets here. The shutdown
/// waits (at most a minute, `folo_server::real::SHUTDOWN_WAIT`) for a training run to stop after its current batch
/// with its last checkpoint on disk, which took 17 s on a machine whose disk other builds kept busy.
const EXIT_DEADLINE: Duration = Duration::from_secs(300);

pub(super) struct Client {
    child: ChildGuard,
    pub(super) stdin: ChildStdin,
    rx: Receiver<Value>,
    lines: Arc<Mutex<Vec<String>>>,
    /// The thread that reads stdout; it ends when the server closes it.
    reader: std::thread::JoinHandle<()>,
    /// Notifications received while waiting for responses, oldest first.
    pub(super) notes: Vec<Value>,
    next_id: u64,
    /// `Some(version)` for 2026-07-28 clients, which send their context in every request's `_meta`.
    pub(super) inline: Option<&'static str>,
}

pub(super) fn spawn(args: &[&str], inline: Option<&'static str>) -> Client {
    // No published weights unless a test puts some there (the machine's own ~/.folo/weights would list as models).
    spawn_with_weights(args, inline, &folo_test_support::root().join("folo-mcp-test-no-weights"))
}

/// [`spawn`] with `FOLO_WEIGHTS_DIR` (the `folo weights` cache, listed as published models) set to `weights`.
/// The server sees no GPU (the browser's `webgpu` backend only): these tests run on the mock or `--device cpu`,
/// and creating a GPU device for `folo_info`'s capabilities takes about a second. It looks at a dataset's files
/// again on every request (`FOLO_DATASET_RECHECK_MS`), not after the 5 s a test would have to wait.
pub(super) fn spawn_with_weights(
    args: &[&str],
    inline: Option<&'static str>,
    weights: &std::path::Path,
) -> Client {
    let mut child: ChildGuard = folo_cmd()
        .arg("mcp")
        .args(args)
        .env("RUST_LOG", "warn")
        .env("FOLO_WEIGHTS_DIR", weights)
        .env("FOLO_GPU_BACKEND", "webgpu")
        .env("FOLO_DATASET_RECHECK_MS", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let stdout = child.stdout.take().unwrap();
    let stdin = child.stdin.take().unwrap();
    let lines = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = channel();
    let seen = lines.clone();
    let reader = std::thread::spawn(move || {
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
    Client { child, stdin, rx, lines, reader, notes: Vec::new(), next_id: 1, inline }
}

impl Client {
    pub(super) fn send(&mut self, v: &Value) {
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
    pub(super) fn start(&mut self, method: &str, params: Value, meta: Value) -> u64 {
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
    pub(super) fn response(&mut self, id: u64) -> Value {
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

    /// The next error response with no request id or `"id": null` (e.g. `-32600 Invalid request`).
    pub(super) fn invalid_request_error(&mut self) -> Value {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let v = self.recv(deadline);
            if v.get("error").is_some() && v.get("id").is_none_or(Value::is_null) {
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

    pub(super) fn request(&mut self, method: &str, params: Value) -> Value {
        self.request_with(method, params, json!({}))
    }

    pub(super) fn result(&mut self, method: &str, params: Value) -> Value {
        let r = self.request(method, params);
        assert!(r.get("error").is_none(), "{method} failed: {r}");
        r["result"].clone()
    }

    /// Calls a tool and returns its result (not asserting success).
    pub(super) fn call(&mut self, name: &str, args: Value) -> Value {
        self.result("tools/call", json!({ "name": name, "arguments": args }))
    }

    /// Calls a tool that must succeed and returns its `structuredContent`.
    pub(super) fn ok(&mut self, name: &str, args: Value) -> Value {
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
    pub(super) fn tool_error(&mut self, name: &str, args: Value, code: &str) -> Value {
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
    pub(super) fn notification(&mut self, timeout: Duration, pred: impl Fn(&Value) -> bool) -> Value {
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
    pub(super) fn finish(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + EXIT_DEADLINE;
        while self.child.try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline, "folo mcp did not exit after stdin closed");
            std::thread::sleep(Duration::from_millis(5));
        }
        // Every line it wrote has been read once the reader sees the end of stdout.
        self.reader.join().unwrap();
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

pub(super) fn validate(schema: &Value, instance: &Value, what: &str) {
    let v = jsonschema::validator_for(schema).unwrap_or_else(|e| panic!("{what}: bad schema: {e}"));
    if let Err(e) = v.validate(instance) {
        panic!("{what}: result does not match the outputSchema: {e}\n{instance}");
    }
}

pub(super) fn tmp_dir(name: &str) -> TestDir {
    TestDir::new(&format!("mcp-test-{name}"))
}

pub(super) const ALL_TOOLS: [&str; 44] = [
    "folo_info",
    "list_datasets",
    "get_dataset",
    "list_dataset_issues",
    "list_dataset_images",
    "register_dataset",
    "create_synthetic_dataset",
    "audit_dataset",
    "merge_datasets",
    "filter_dataset",
    "dataset_export",
    "list_models",
    "train",
    "run_status",
    "list_runs",
    "stop_run",
    "reorder_run",
    "val",
    "predict",
    "predict_batch",
    "export",
    "compare_runs",
    "cloud_connections",
    "cloud_offers",
    "cloud_instances",
    "cloud_terminate",
    "list_label_jobs",
    "create_label_job",
    "start_labeling",
    "stop_labeling",
    "label_status",
    "list_label_images",
    "view_label_image",
    "set_label_boxes",
    "review_label_images",
    "export_label_dataset",
    "bench_catalog",
    "list_benchmarks",
    "get_benchmark",
    "create_benchmark",
    "delete_benchmark",
    "start_bench_entry",
    "bench_entry",
    "stop_bench_entry",
];

/// The most `tools/list` may take, and the benchmark tools of it, in bytes (docs/MCP.md states what they take).
const TOTAL_BYTES: usize = 108 * 1024;
const BENCH_BYTES: usize = 11 * 1024;

/// `tools/list` with schema checks; returns `name -> outputSchema`.
pub(super) fn check_tools(c: &mut Client, expected: &[&str]) -> std::collections::HashMap<String, Value> {
    let tools = c.result("tools/list", json!({}))["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, expected);
    // Shared types are compacted (docs/MCP.md "Schemas"): the whole listing stays small. The labeling and the
    // benchmark tools are measured on their own too (an agent that never labels or benchmarks still pays for them).
    let share = |part: &str| -> usize {
        tools
            .iter()
            .filter(|t| t["name"].as_str().unwrap().contains(part))
            .map(|t| serde_json::to_vec(t).unwrap().len())
            .sum()
    };
    let (bytes, label, bench) = (serde_json::to_vec(&tools).unwrap().len(), share("label"), share("bench"));
    let sizes = format!(
        "tools/list is {bytes} bytes, {label} of them the labeling tools, {bench} the benchmark tools"
    );
    assert!(bytes < TOTAL_BYTES && label < 14 * 1024 && bench < BENCH_BYTES, "{sizes}");
    let mut out = std::collections::HashMap::new();
    for t in &tools {
        assert!(t["description"].as_str().unwrap().len() > 20);
        assert_eq!(
            t["inputSchema"]["additionalProperties"], false,
            "{} rejects unknown arguments",
            t["name"]
        );
        assert!(!t.to_string().contains("0.009999999776482582"), "f32 noise in {}", t["name"]);
        for key in ["inputSchema", "outputSchema"] {
            assert_eq!(t[key]["type"], "object", "{} {key}", t["name"]);
            jsonschema::validator_for(&t[key]).unwrap_or_else(|e| panic!("{} {key}: {e}", t["name"]));
        }
        out.insert(t["name"].as_str().unwrap().to_string(), t["outputSchema"].clone());
    }
    if let Some(train) = tools.iter().find(|t| t["name"] == "train") {
        // `dataset_id` is required unless `resume_run_id` is given.
        let input = jsonschema::validator_for(&train["inputSchema"]).unwrap();
        assert!(input.is_valid(&json!({ "dataset_id": "ds_1", "epochs": 2 })));
        assert!(input.is_valid(&json!({ "resume_run_id": "run_1" })));
        assert!(!input.is_valid(&json!({ "epochs": 2 })));
        assert!(!input.is_valid(&json!({ "resume_run_id": null })));
        assert!(train["inputSchema"]["properties"]["resume_run_id"].is_object());
        assert_eq!(train["annotations"]["readOnlyHint"], false);
    }
    out
}

/// Calls a tool and validates its structured result against the tool's outputSchema.
pub(super) fn checked(
    c: &mut Client,
    schemas: &std::collections::HashMap<String, Value>,
    name: &str,
    args: Value,
) -> Value {
    let v = c.ok(name, args);
    validate(&schemas[name], &v, name);
    v
}

pub(super) fn init_old(c: &mut Client) {
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
}

/// A one-image YOLO dataset: `<dir>/images/train/a.jpg` + `<dir>/labels/train/a.txt`.
pub(super) fn yolo_files(dir: &std::path::Path) {
    for sub in ["images/train", "labels/train"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    image::RgbImage::from_pixel(32, 24, image::Rgb([200, 40, 40]))
        .save(dir.join("images/train/a.jpg"))
        .unwrap();
    std::fs::write(dir.join("labels/train/a.txt"), "0 0.5 0.5 0.25 0.25\n").unwrap();
}

/// A directory link at `link` to `target`: a junction on Windows (no privileges needed), a symlink elsewhere.
pub(super) fn make_dir_link(target: &std::path::Path, link: &std::path::Path) -> bool {
    #[cfg(windows)]
    {
        // mklink reads a `/` as the start of a switch: pass `\` separators (FOLO_TEST_TMP may use `/`).
        let native = |p: &std::path::Path| p.to_string_lossy().replace('/', "\\");
        Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(native(link))
            .arg(native(target))
            .stdout(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
}

pub(super) fn remove_dir_link(link: &std::path::Path) {
    // A junction (Windows) is removed like an empty directory, a symlink like a file.
    if std::fs::remove_dir(link).is_err() {
        std::fs::remove_file(link).unwrap();
    }
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
pub fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1").env_remove("FOLO_SERVER");
    c
}
