use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use folo_api::{ExportFormat, OnnxSwitches, Split, ValRequest};
use folo_server::store::{DynamicAxes, FEATURE_EXPORT_DYNAMIC_AXES, FEATURE_VAL_DIAGNOSTICS, ValDiagConfig};

use super::*;

/// Loopback mock API server that drains request bodies (preventing TCP RST on close) and shuts down cleanly
/// on drop.
struct MockApi {
    addr: SocketAddr,
    base: String,
    log: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MockApi {
    fn start(handler: impl Fn(&str, usize) -> (&'static str, String) + Send + 'static) -> MockApi {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{addr}");
        let log = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let recorded = Arc::clone(&log);
        let done = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let mut idx = 0usize;
            for stream in listener.incoming() {
                if done.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut s) = stream else { continue };
                let Some(req) = read_request(&mut s) else { continue };
                recorded.lock().unwrap().push(req.clone());
                let (status, body) = handler(&req, idx);
                idx += 1;
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
                     close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.flush();
            }
        });
        MockApi { addr, base, log, stop, handle: Some(handle) }
    }

    fn requests(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

impl Drop for MockApi {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Reads the HTTP request head, drains any `Content-Length` body so closing the socket never triggers a TCP
/// `RST`, and returns `"METHOD /path"`.
fn read_request(s: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut b = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        match s.read(&mut b) {
            Ok(1) => buf.push(b[0]),
            _ => break,
        }
    }
    if buf.is_empty() {
        return None;
    }
    let head = String::from_utf8_lossy(&buf);
    let content_length = head
        .lines()
        .skip(1)
        .find_map(|line| {
            let (k, v) = line.split_once(':')?;
            k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        let _ = s.read_exact(&mut body);
    }
    let first = head.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    Some(format!("{} {}", parts.next().unwrap_or(""), parts.next().unwrap_or("")))
}

fn system_json(features: &[&str]) -> String {
    serde_json::to_string(&SystemResponse {
        system: SystemInfo {
            cpu: folo_api::CpuInfo { name: "test".into(), threads: 4, util: 0.1 },
            memory: folo_api::MemoryInfo { used_bytes: 512, total_bytes: 1024 },
            gpus: vec![],
        },
        data_dir: None,
        features: features.iter().map(|s| (*s).to_string()).collect(),
        capabilities: None,
        gpu_pool: None,
    })
    .unwrap()
}

#[test]
fn unreachable_server_is_a_tool_error() {
    // A server that hangs up without an answer. (A closed port would do too, but on Windows a refused
    // loopback connection takes 2 s.)
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            drop(conn);
        }
    });
    let s = HttpStore::new(&url, Some("s3cret"));
    let e = s.list_datasets().unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::BAD_GATEWAY, "server_unreachable"));
    assert!(e.body.hint.unwrap().contains("folo serve"));
    assert!(!e.body.message.contains("s3cret"));
}

#[test]
fn feature_and_mock_probes_share_cache_across_until_and_export() {
    let server = MockApi::start(|req, _| match req {
        "GET /api/health" => (
            "200 OK",
            serde_json::json!({
                "status": "ok",
                "version": "0.1.0",
                "mock": true,
                "features": [FEATURE_EXPORT_ONNX_SWITCHES, folo_api::FEATURE_VISION_BENCH],
            })
            .to_string(),
        ),
        "POST /api/models/m1/export" => {
            ("200 OK", serde_json::json!({ "path": "/tmp/m1-f16.onnx", "bytes": 1024 }).to_string())
        }
        other => panic!("unexpected request: {other}"),
    });

    let store = HttpStore::new(&server.base, None);
    // `is_mock` probes `/api/health` once and populates both `mock` and `features`.
    assert!(store.is_mock());
    assert!(store.has_feature(FEATURE_EXPORT_ONNX_SWITCHES).unwrap());
    // `until` shares the cache with its parent store.
    let deadline = Instant::now() + Duration::from_secs(5);
    assert!(store.until(deadline).has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    // ONNX-switch export reuses the cached features without hitting `/api/system`.
    let exported = store
        .export_model(
            "m1",
            ExportSpec {
                format: ExportFormat::Onnx,
                onnx: OnnxSwitches { half: true, ..OnnxSwitches::default() },
            },
        )
        .unwrap();
    assert_eq!(exported.path, "/tmp/m1-f16.onnx");

    // Clearing the cache forces a fresh probe, and probing through `until` populates the parent cache too.
    store.clear_probe_cache();
    assert!(store.until(deadline).has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    assert!(store.is_mock());

    assert_eq!(server.requests(), vec!["GET /api/health", "POST /api/models/m1/export", "GET /api/health"]);
}

#[test]
fn legacy_server_falls_back_to_system_once_and_caches_missing_export_feature() {
    let server = MockApi::start(|req, _| match req {
        "GET /api/health" => (
            "200 OK",
            serde_json::json!({ "status": "ok", "version": "0.1.0", "mock": false, "features": [] })
                .to_string(),
        ),
        "GET /api/system" => ("200 OK", system_json(&[folo_api::FEATURE_VISION_BENCH])),
        other => panic!("unexpected request: {other}"),
    });

    let store = HttpStore::new(&server.base, None);
    assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    assert!(!store.is_mock());
    let err = store
        .export_model(
            "m1",
            ExportSpec {
                format: ExportFormat::Onnx,
                onnx: OnnxSwitches { decode: true, ..OnnxSwitches::default() },
            },
        )
        .unwrap_err();
    assert_eq!(err.status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/system"]);
}

#[test]
fn older_server_rejects_val_diagnostics_and_split_dynamic_axes() {
    let store = HttpStore::new("http://127.0.0.1:1", None).assume_features(vec![]);
    let err = store
        .val_with_diagnostics(
            ValRequest {
                model_id: "m1".into(),
                dataset_id: "ds1".into(),
                split: Split::Val,
                imgsz: None,
                conf: 0.001,
                iou: 0.7,
            },
            Some(ValDiagConfig::default()),
        )
        .unwrap_err();
    assert_eq!((err.status, err.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));

    let err = store
        .export_model_with_axes(
            "m1",
            ExportSpec { format: ExportFormat::Onnx, onnx: OnnxSwitches::default() },
            DynamicAxes::BATCH,
        )
        .unwrap_err();
    assert_eq!((err.status, err.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));

    // Confirm capability names match constants.
    assert_eq!(FEATURE_VAL_DIAGNOSTICS, "val_diagnostics");
    assert_eq!(FEATURE_EXPORT_DYNAMIC_AXES, "export_dynamic_axes");
}

#[test]
fn transient_health_error_is_not_cached() {
    let server = MockApi::start(|req, idx| match (req, idx) {
        ("GET /api/health", 0) => (
            "503 Service Unavailable",
            serde_json::json!({ "error": { "code": "internal", "message": "starting up" } }).to_string(),
        ),
        ("GET /api/health", _) => (
            "200 OK",
            serde_json::json!({
                "status": "ok",
                "version": "0.1.0",
                "mock": true,
                "features": [folo_api::FEATURE_VISION_BENCH],
            })
            .to_string(),
        ),
        other => panic!("unexpected request: {other:?}"),
    });

    let store = HttpStore::new(&server.base, None);
    assert!(!store.is_mock(), "transient 503 returns false without poisoning the cache");
    assert!(store.is_mock(), "next probe succeeds once the server recovers");
    assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/health"]);
}

#[test]
fn concurrent_legacy_feature_checks_probe_system_only_once() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let server = MockApi::start(move |req, _| match req {
        "GET /api/health" => (
            "200 OK",
            serde_json::json!({ "status": "ok", "version": "0.1.0", "mock": false, "features": [] })
                .to_string(),
        ),
        "GET /api/system" => {
            let _ = entered_tx.send(());
            let _ = release_rx.lock().unwrap().recv();
            ("200 OK", system_json(&[folo_api::FEATURE_VISION_BENCH]))
        }
        other => panic!("unexpected request: {other}"),
    });

    let store = Arc::new(HttpStore::new(&server.base, None));
    let parent = Arc::clone(&store);
    let child = store.until(Instant::now() + Duration::from_secs(5));

    let t1 = std::thread::spawn(move || parent.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    // Wait until t1 is inside `GET /api/system`, then launch a concurrent `until` caller.
    entered_rx.recv().unwrap();
    let t2 = std::thread::spawn(move || child.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    // Wait deterministically until t2 is blocked on the single-flight condvar before releasing the response.
    while store.probe.state.lock().unwrap().waiters == 0 {
        std::thread::yield_now();
    }
    release_tx.send(()).unwrap();

    assert!(t1.join().unwrap());
    assert!(t2.join().unwrap());
    assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/system"]);
}

#[test]
fn in_flight_probe_does_not_repopulate_after_clear_probe_cache() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let server = MockApi::start(move |req, idx| match (req, idx) {
        ("GET /api/health", 0) => {
            let _ = entered_tx.send(());
            let _ = release_rx.lock().unwrap().recv();
            (
                "200 OK",
                serde_json::json!({
                    "status": "ok",
                    "version": "0.1.0",
                    "mock": false,
                    "features": ["stale_feature"],
                })
                .to_string(),
            )
        }
        ("GET /api/health", _) => (
            "200 OK",
            serde_json::json!({
                "status": "ok",
                "version": "0.1.0",
                "mock": true,
                "features": [folo_api::FEATURE_VISION_BENCH],
            })
            .to_string(),
        ),
        other => panic!("unexpected request: {other:?}"),
    });

    let store = Arc::new(HttpStore::new(&server.base, None));
    let probe_store = Arc::clone(&store);
    let t = std::thread::spawn(move || probe_store.has_feature("stale_feature").unwrap());
    entered_rx.recv().unwrap();
    // Clear the cache while the first probe is in flight, then let it finish.
    store.clear_probe_cache();
    release_tx.send(()).unwrap();
    assert!(t.join().unwrap());

    // The stale probe must not have populated the cache: the next check makes a fresh request.
    assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
    assert!(store.is_mock());
    assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/health"]);
}

#[test]
fn multipart_rejects_header_injection_and_avoids_boundary_collision() {
    let bad = [("model_id", "m_1\r\n--fake".to_string())];
    assert_eq!(multipart(&bad, b"img").unwrap_err().body.code, "invalid_arguments");
    let bad_nul = [("model_id", "m_1\0x".to_string())];
    assert_eq!(multipart(&bad_nul, b"img").unwrap_err().body.code, "invalid_arguments");

    let fields = [("model_id", "m_1".to_string())];
    let (ct, body) = multipart(&fields, b"hello--folo-mcp-world").unwrap();
    let boundary = ct.strip_prefix("multipart/form-data; boundary=").unwrap();
    assert!(body.windows(boundary.len()).filter(|w| *w == boundary.as_bytes()).count() == 3);
}

#[test]
fn older_server_rejects_folo2_arch_and_proxy_reads_arch_header() {
    use folo_server::store::{FEATURE_FOLO2_ARCH, RunOptions};

    let older = HttpStore::new("http://127.0.0.1:1", None).assume_features(vec![]);
    let cfg: folo_api::TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
    let v2_opts = RunOptions { auto_batch: Default::default(), arch: Some("folo2-n".into()) };

    let e = older.create_run_with_options(cfg.clone(), v2_opts.clone()).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));

    let e = older.estimate_run_with_options(cfg, v2_opts).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));

    let resume_fields = serde_json::json!({ "arch": "folo2-n" }).as_object().unwrap().clone();
    let e = older.resume_run("run_1", &resume_fields).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));
    // Invalid arch on resume fails with `400 invalid_config`, not `501 not_implemented`.
    let bad_resume = serde_json::json!({ "arch": "folo2-q" }).as_object().unwrap().clone();
    let e = older.resume_run("run_1", &bad_resume).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::BAD_REQUEST, "invalid_config"));
    assert_eq!(FEATURE_FOLO2_ARCH, "folo2_arch");

    // Passing a v1 `RunOptions.arch` (e.g. `Some("folo-s")`) to an older server without `folo2_arch` normalizes
    // to `config.arch = Arch::S` and delegates to `POST /api/runs` instead of returning `501`.
    let v1_server = MockApi::start(|req, _| match req {
        "POST /api/runs" => (
            "201 Created",
            serde_json::json!({
                "id": "run_v1",
                "name": "v1-run",
                "status": "queued",
                "config": { "dataset_id": "ds_1", "arch": "folo-s" },
                "created": "2026-11-27T00:00:00Z",
                "started": null,
                "finished": null,
                "epoch": 0,
                "epochs": 100,
                "step": 0,
                "steps_per_epoch": 0,
                "eta_s": null,
                "best": null,
                "device": "cpu",
                "model_id": null,
                "error": null
            })
            .to_string(),
        ),
        other => panic!("unexpected request: {other}"),
    });
    let v1_proxy = HttpStore::new(&v1_server.base, None).assume_features(vec![]);
    let v1_cfg: folo_api::TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
    let v1_run = v1_proxy
        .create_run_with_options(
            v1_cfg,
            RunOptions { auto_batch: Default::default(), arch: Some("folo-s".into()) },
        )
        .unwrap();
    assert_eq!(v1_run.config.arch, folo_api::Arch::S);

    // A server exposing `x-folo-arch: folo2-n` on `GET /api/models/{id}` surfaces the exact arch to the proxy.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            let _ = read_request(&mut s);
            let body = serde_json::json!({
                "id": "m_v2",
                "name": "v2-best",
                "arch": "folo-n",
                "imgsz": 64,
                "classes": ["circle", "square"],
                "params": 1000,
                "flops_g": 1.0,
                "bytes": 1024,
                "created": "2026-11-27T00:00:00Z",
                "run_id": null,
                "metrics": null,
                "head": "o2m"
            })
            .to_string();
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nx-folo-arch: folo2-n\r\nContent-Length: \
                 {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.flush();
        }
    });
    let proxy = HttpStore::new(&format!("http://{addr}"), None);
    assert_eq!(proxy.model_arch("m_v2").unwrap(), "folo2-n");
    handle.join().unwrap();

    // `GET /api/models` carrying additive `arch_name` preserves `folo2-n` through `list_models_with_arch`
    // (and falls back to `arch` for older servers that omit `arch_name`).
    let list_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let list_addr = list_listener.local_addr().unwrap();
    let list_handle = std::thread::spawn(move || {
        if let Ok((mut s, _)) = list_listener.accept() {
            let _ = read_request(&mut s);
            let body = serde_json::json!({
                "models": [
                    {
                        "id": "m_v2",
                        "name": "v2-best",
                        "arch": "folo-n",
                        "arch_name": "folo2-n",
                        "imgsz": 64,
                        "classes": ["circle"],
                        "params": 1000,
                        "flops_g": 1.0,
                        "bytes": 1024,
                        "created": "2026-11-27T00:00:00Z",
                        "run_id": null,
                        "metrics": null,
                        "head": "o2m"
                    },
                    {
                        "id": "m_legacy",
                        "name": "legacy-best",
                        "arch": "folo-s",
                        "imgsz": 64,
                        "classes": ["circle"],
                        "params": 1000,
                        "flops_g": 1.0,
                        "bytes": 1024,
                        "created": "2026-11-26T00:00:00Z",
                        "run_id": null,
                        "metrics": null,
                        "head": "o2m"
                    }
                ]
            })
            .to_string();
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                 {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = s.flush();
        }
    });
    let list_proxy = HttpStore::new(&format!("http://{list_addr}"), None);
    let listed = list_proxy.list_models_with_arch().unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!((listed[0].0.id.as_str(), listed[0].1.as_str()), ("m_v2", "folo2-n"));
    assert_eq!((listed[1].0.id.as_str(), listed[1].1.as_str()), ("m_legacy", "folo-s"));
    list_handle.join().unwrap();

    // Older real server (no `folo2_arch`, no `x-folo-arch` header): probing `/models/{id}/weights` detects a
    // `folo2-n` safetensors header so a weights-only MCP `train` call is rejected with `501 not_implemented`
    // rather than silently fine-tuning as v1.
    let old_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let old_addr = old_listener.local_addr().unwrap();
    let old_stop = Arc::new(AtomicBool::new(false));
    let old_done = Arc::clone(&old_stop);
    let old_handle = std::thread::spawn(move || {
        let mut st_hdr = serde_json::json!({
            "__metadata__": {
                "folo.arch": "folo-n",
                "folo.config": serde_json::json!({ "family": "folo2", "scale": "n" }).to_string()
            }
        })
        .to_string()
        .into_bytes();
        while !st_hdr.len().is_multiple_of(8) {
            st_hdr.push(b' ');
        }
        let mut weights_bytes = (st_hdr.len() as u64).to_le_bytes().to_vec();
        weights_bytes.extend_from_slice(&st_hdr);
        for stream in old_listener.incoming() {
            if old_done.load(Ordering::SeqCst) {
                break;
            }
            let Ok(mut s) = stream else { continue };
            let Some(req) = read_request(&mut s) else { continue };
            match req.as_str() {
                "GET /api/health" => {
                    let body = serde_json::json!({
                        "status": "ok",
                        "version": "0.1.0",
                        "mock": false,
                        "features": []
                    })
                    .to_string();
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                         {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                "GET /api/system" => {
                    let body = system_json(&[]);
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                         {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                "GET /api/datasets/ds_1" => {
                    let body = serde_json::json!({
                        "id": "ds_1",
                        "name": "tiny",
                        "path": "/tmp/tiny",
                        "format": "yolo",
                        "classes": ["circle"],
                        "splits": { "train": 4, "val": 2, "test": 0 },
                        "total_images": 6,
                        "total_boxes": 6,
                        "size_bytes": 100,
                        "created": "2026-11-27T00:00:00Z"
                    })
                    .to_string();
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                         {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                "GET /api/models/m_old_v2" => {
                    let body = serde_json::json!({
                        "id": "m_old_v2",
                        "name": "old-v2",
                        "arch": "folo-n",
                        "imgsz": 64,
                        "classes": ["circle"],
                        "params": 1000,
                        "flops_g": 1.0,
                        "bytes": weights_bytes.len(),
                        "created": "2026-11-27T00:00:00Z",
                        "run_id": null,
                        "metrics": null,
                        "head": "o2m"
                    })
                    .to_string();
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: \
                         {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                "GET /api/models/m_old_v2/weights" => {
                    let len = weights_bytes.len();
                    let _ = write!(
                        s,
                        "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Range: \
                         bytes 0-{}/{len}\r\nETag: \"v2etag\"\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
                        len - 1
                    );
                    let _ = s.write_all(&weights_bytes);
                }
                other => panic!(
                    "unexpected request to older server (weights-only folo2 must be refused before POST /api/runs): {other}"
                ),
            }
            let _ = s.flush();
        }
    });
    let old_proxy = Arc::new(HttpStore::new(&format!("http://{old_addr}"), None));
    assert_eq!(old_proxy.model_arch("m_old_v2").unwrap(), "folo2-n");
    let home = folo_test_support::TestDir::new("mcp-old-proxy-v2");
    let opts = crate::Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };
    let mcp = crate::server::FoloMcp::new(old_proxy.clone(), Some(old_proxy), opts);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let args = serde_json::json!({ "dataset_id": "ds_1", "weights": "m_old_v2", "epochs": 1 })
        .as_object()
        .unwrap()
        .clone();
    let res = rt.block_on(mcp.inner.train(args));
    let err = res.err().expect("weights-only folo2 fine-tune against older server must fail");
    assert_eq!((err.status, err.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_implemented"));
    old_stop.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect(old_addr);
    old_handle.join().unwrap();
}
