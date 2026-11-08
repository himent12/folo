//! Integration tests: every endpoint of docs/API.md in mock mode through `tower::ServiceExt::oneshot`, plus SSE,
//! the real store, static files, CORS and compression.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Cors, MockStore, Options, RealStore, Server, Store};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

fn mock() -> (Server, Arc<MockStore>) {
    let store = Arc::new(MockStore::new());
    let server = Server::new(store.clone() as Arc<dyn Store>, Options::default());
    (server, store)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let body = res.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, headers, body)
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn post_json(uri: &str, v: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().clone();
    let (status, _, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().clone();
    let (got, headers, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().starts_with("application/json"));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}");
    assert!(!e.error.message.is_empty());
    e.error
}

#[tokio::test]
async fn system_endpoints() {
    let (server, _) = mock();
    let app = server.router();
    let h: Health = ok(&app, get("/api/health")).await;
    assert_eq!((h.status.as_str(), h.mock, h.api.as_str()), ("ok", true, "v1"));
    let s: SystemInfo = ok(&app, get("/api/system")).await;
    assert!(s.cpu.threads >= 1);
    assert!(s.memory.total_bytes > 0);
    assert_eq!(s.gpus.len(), 1);
    err(&app, get("/api/nope"), StatusCode::NOT_FOUND, "route_not_found").await;
    err(
        &app,
        Request::put("/api/health").body(Body::empty()).unwrap(),
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
    )
    .await;
}

#[tokio::test]
async fn datasets_and_images() {
    let (server, _) = mock();
    let app = server.router();
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    let names: Vec<_> = list.datasets.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["shapes-1k", "traffic-cams", "r6-operators"]);
    assert!(list.datasets.iter().all(|d| d.stats.is_none()));
    let r6 = list.datasets.iter().find(|d| d.name == "r6-operators").unwrap();
    assert_eq!(r6.classes, ["body", "Head"]);

    let full: Dataset = ok(&app, get(&format!("/api/datasets/{}", r6.id))).await;
    let stats = full.stats.unwrap();
    assert_eq!(stats.class_counts.iter().sum::<u64>(), stats.boxes);
    assert_eq!(stats.box_size_hist.counts.iter().sum::<u64>(), stats.boxes);
    let n = full.splits.train + full.splits.val + full.splits.test;
    assert_eq!(stats.boxes_per_image_hist.counts.iter().sum::<u64>(), n);

    let page: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?split=val&offset=10&limit=5", r6.id))).await;
    assert_eq!(page.total, full.splits.val);
    assert_eq!(page.images.len(), 5);
    assert!(page.images.iter().all(|i| i.split == Split::Val && i.width == 1920 && i.height == 1080));
    let heads: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?class=1&limit=1000", r6.id))).await;
    assert!(heads.images.iter().all(|i| i.boxes.iter().any(|b| b.class == 1)));
    assert!(heads.total < n);
    let again: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?split=val&offset=10&limit=5", r6.id))).await;
    assert_eq!(again, page, "mock data is deterministic");
    err(
        &app,
        get(&format!("/api/datasets/{}/images?split=nope", r6.id)),
        StatusCode::BAD_REQUEST,
        "invalid_query",
    )
    .await;

    let img = &page.images[0];
    let (status, headers, body) =
        send(&app, get(&format!("/api/datasets/{}/images/{}", r6.id, img.id))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
    let decoded = image::load_from_memory(&body).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (1920, 1080));
    let (_, _, thumb) =
        send(&app, get(&format!("/api/datasets/{}/images/{}?thumb=256", r6.id, img.id))).await;
    let t = image::load_from_memory(&thumb).unwrap();
    assert_eq!((t.width(), t.height()), (256, 144));
    err(&app, get(&format!("/api/datasets/{}/images/zzz", r6.id)), StatusCode::NOT_FOUND, "image_not_found")
        .await;
    err(&app, get("/api/datasets/ds_none"), StatusCode::NOT_FOUND, "dataset_not_found").await;

    // Synthetic datasets.
    let body =
        json!({ "name": "tiny", "images": 20, "classes": ["circle", "star"], "imgsz": 320, "seed": 4 });
    let (status, _, b) = send(&app, post_json("/api/datasets/synthetic", body)).await;
    assert_eq!(status, StatusCode::CREATED);
    let syn: Dataset = serde_json::from_slice(&b).unwrap();
    assert_eq!(syn.splits, Splits { train: 16, val: 4, test: 0 });
    assert_eq!(syn.stats.as_ref().unwrap().class_counts.len(), 2);
    let tp: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images", syn.id))).await;
    assert_eq!(tp.total, 20);
    err(
        &app,
        post_json("/api/datasets/synthetic", json!({ "name": "" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(
        &app,
        post_json("/api/datasets/synthetic", json!({ "images": 3 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;

    // Register from disk: a missing path is a clean 404.
    err(
        &app,
        post_json("/api/datasets", json!({ "path": "Z:/definitely/not/here" })),
        StatusCode::NOT_FOUND,
        "path_not_found",
    )
    .await;

    // Delete.
    let (status, _, _) =
        send(&app, Request::delete(format!("/api/datasets/{}", syn.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 3);
    err(
        &app,
        Request::delete(format!("/api/datasets/{}", syn.id)).body(Body::empty()).unwrap(),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
}

#[tokio::test]
async fn models() {
    let (server, _) = mock();
    let app = server.router();
    let list: ModelList = ok(&app, get("/api/models")).await;
    assert!((5..=6).contains(&list.models.len()));
    let created: Vec<_> = list.models.iter().map(|m| m.created.clone()).collect();
    let mut sorted = created.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(created, sorted, "newest first");
    let m = &list.models[0];
    let one: Model = ok(&app, get(&format!("/api/models/{}", m.id))).await;
    assert_eq!(&one, m);
    let ex: ExportResult =
        ok(&app, post_json(&format!("/api/models/{}/export", m.id), json!({ "format": "onnx" }))).await;
    assert!(ex.path.ends_with(".onnx") && ex.bytes > 0);
    err(
        &app,
        post_json(&format!("/api/models/{}/export", m.id), json!({ "format": "tflite" })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    err(&app, get("/api/models/m_none"), StatusCode::NOT_FOUND, "model_not_found").await;
}

#[tokio::test]
async fn runs_lifecycle() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let statuses: Vec<_> = list.runs.iter().map(|r| r.status).collect();
    assert!(statuses.contains(&RunStatus::Running));
    assert!(statuses.contains(&RunStatus::Completed));
    assert!(statuses.contains(&RunStatus::Failed));
    assert!(statuses.contains(&RunStatus::Stopped));
    let failed = list.runs.iter().find(|r| r.status == RunStatus::Failed).unwrap();
    assert!(failed.error.as_deref().unwrap().contains("out of memory"));
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap();
    assert!(live.epoch > 10 && live.eta_s.is_some() && live.best.is_some());

    // Finished run: full history.
    let done = list.runs.iter().find(|r| r.status == RunStatus::Completed).unwrap();
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", done.id))).await;
    assert_eq!(m.steps.len() as u64, done.step);
    assert_eq!(m.epochs.len() as u32, done.epochs);
    let first = &m.epochs[0];
    let last = m.epochs.last().unwrap();
    assert!(last.train_loss.total < first.train_loss.total);
    assert!(last.val.map50_95 > first.val.map50_95);
    assert_eq!(done.best.unwrap().map50_95, m.epochs.iter().map(|e| e.val.map50_95).fold(0.0, f32::max));
    let thin: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics?every=100", done.id))).await;
    assert_eq!(thin.steps.last().unwrap().step, done.step);
    assert!(thin.steps.len() as u64 <= done.step / 100 + 1);

    let s: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples?epoch=5", done.id))).await;
    assert_eq!((s.epoch, s.samples.len()), (5, 16));
    assert!(s.samples.iter().all(|x| x.pred.iter().all(|p| p.score >= 0.25)));
    let latest: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples", done.id))).await;
    assert_eq!(latest.epoch, done.epochs);
    err(
        &app,
        get(&format!("/api/runs/{}/samples?epoch=9999", done.id)),
        StatusCode::NOT_FOUND,
        "epoch_not_available",
    )
    .await;

    // Create, stop, delete.
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let cfg = json!({ "dataset_id": ds.datasets[0].id, "arch": "folo-s", "epochs": 5 });
    let (status, _, b) = send(&app, post_json("/api/runs", cfg)).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let run: Run = serde_json::from_slice(&b).unwrap();
    assert_eq!(run.status, RunStatus::Queued);
    assert_eq!(run.name, "shapes-1k-s");
    assert_eq!(run.config.batch, 16);
    let list: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(list.runs[0].id, run.id, "newest first");
    err(
        &app,
        Request::delete(format!("/api/runs/{}", run.id)).body(Body::empty()).unwrap(),
        StatusCode::CONFLICT,
        "run_active",
    )
    .await;
    let stopped: Run =
        ok(&app, Request::post(format!("/api/runs/{}/stop", run.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(stopped.status, RunStatus::Stopping);
    err(
        &app,
        Request::post(format!("/api/runs/{}/stop", done.id)).body(Body::empty()).unwrap(),
        StatusCode::CONFLICT,
        "run_not_active",
    )
    .await;
    tokio::time::sleep(Duration::from_millis(2000)).await;
    let r: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(r.status, RunStatus::Stopped);
    let (status, _, _) =
        send(&app, Request::delete(format!("/api/runs/{}", run.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(&app, get(&format!("/api/runs/{}", run.id)), StatusCode::NOT_FOUND, "run_not_found").await;

    // Validation.
    let bad = json!({ "dataset_id": ds.datasets[0].id, "imgsz": 100 });
    let e = err(&app, post_json("/api/runs", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert!(e.message.contains("imgsz"));
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": "ds_none" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
    err(
        &app,
        post_json("/api/runs", json!({ "epochs": 3 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    let (status, _, _) =
        send(&app, Request::post("/api/runs").body(Body::from(r#"{"dataset_id":"x"}"#)).unwrap()).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn predict_json_and_multipart() {
    let (server, _) = mock();
    let app = server.router();
    let models: ModelList = ok(&app, get("/api/models")).await;
    let model = models.models.iter().find(|m| m.name == "traffic-s-best").unwrap();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let traffic = ds.datasets.iter().find(|d| d.name == "traffic-cams").unwrap();

    let req = json!({ "model_id": model.id, "dataset_id": traffic.id, "image_id": "000010", "conf": 0.3 });
    let p: PredictResponse = ok(&app, post_json("/api/predict", req.clone())).await;
    assert_eq!((p.image.width, p.image.height), (1280, 720));
    assert!(p.boxes.iter().all(|b| b.score >= 0.3 && b.x1 <= b.x2 && b.y1 <= b.y2));
    assert!(p.boxes.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(p.timing.total_ms > 0.0);
    let again: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
    assert_eq!(again, p, "deterministic");

    // Multipart upload of a JPEG we fetched from the server.
    let (_, _, jpg) = send(&app, get(&format!("/api/datasets/{}/images/000003", traffic.id))).await;
    let multipart = |bytes: &[u8], max_det: &str| {
        let boundary = "XfoloBoundary";
        let mut body = Vec::new();
        for (name, value) in [("model_id", model.id.as_str()), ("conf", "0.1"), ("max_det", max_det)] {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                    .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        Request::post("/api/predict")
            .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
            .body(Body::from(body))
            .unwrap()
    };
    let up: PredictResponse = ok(&app, multipart(&jpg, "300")).await;
    assert_eq!((up.image.width, up.image.height), (1280, 720));
    assert!(!up.boxes.is_empty());
    let up2: PredictResponse = ok(&app, multipart(&jpg, "1")).await;
    assert_eq!(up2.boxes.len(), 1);
    assert_eq!(up2.boxes[0], up.boxes[0], "same bytes, same answer");
    err(&app, multipart(b"not an image", "300"), StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed")
        .await;

    err(
        &app,
        post_json(
            "/api/predict",
            json!({ "model_id": "m_none", "dataset_id": traffic.id, "image_id": "000001" }),
        ),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;
    err(
        &app,
        post_json(
            "/api/predict",
            json!({ "model_id": model.id, "dataset_id": traffic.id, "image_id": "000001", "conf": 2 }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(&app, get("/api/predict/stream"), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
}

/// Reads SSE frames until `n` events arrived (or the timeout hits). Returns `(event, data)` pairs and whether a
/// keep-alive comment was seen.
async fn read_events(body: &mut Body, n: usize, timeout: Duration) -> Vec<(String, String)> {
    let mut buf = String::new();
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    while out.len() < n {
        let frame = match tokio::time::timeout_at(deadline, body.frame()).await {
            Ok(Some(f)) => f.unwrap(),
            Ok(None) | Err(_) => break,
        };
        if let Ok(data) = frame.into_data() {
            buf.push_str(std::str::from_utf8(&data).unwrap());
        }
        while let Some(end) = buf.find("\n\n") {
            let block: String = buf.drain(..end + 2).collect();
            let mut ev = None;
            let mut data = String::new();
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    ev = Some(v.to_string());
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data.push_str(v);
                }
            }
            if let Some(ev) = ev {
                out.push((ev, data));
            }
        }
    }
    out
}

#[tokio::test]
async fn sse_live_run_streams_and_cleans_up() {
    let (server, store) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap().id.clone();

    // Two subscribers at once.
    let uri = format!("/api/runs/{live}/events");
    let res_a = app.clone().oneshot(get(&uri)).await.unwrap();
    let res_b = app.clone().oneshot(get(&uri)).await.unwrap();
    assert_eq!(res_a.status(), StatusCode::OK);
    assert_eq!(res_a.headers()[header::CONTENT_TYPE], "text/event-stream");
    assert_eq!(store.hub_state(&live), (2, true));

    let mut body_a = res_a.into_body();
    let mut body_b = res_b.into_body();
    let a = read_events(&mut body_a, 4, Duration::from_secs(5)).await;
    let b = read_events(&mut body_b, 2, Duration::from_secs(5)).await;
    assert_eq!(a.len(), 4, "{a:?}");
    assert_eq!(a[0].0, "status");
    let st: StatusEvent = serde_json::from_str(&a[0].1).unwrap();
    assert_eq!(st.status, RunStatus::Running);
    for (name, data) in &a[1..] {
        let ev = RunEvent::from_sse(name, data).expect("known event").expect("valid payload");
        if let RunEvent::Step(s) = ev {
            assert!(s.step > 0 && s.lr > 0.0 && s.loss.total > 0.0);
        }
    }
    assert!(a.iter().any(|(n, _)| n == "step"));
    assert_eq!(b[0].0, "status");

    // Disconnect one client: the other keeps streaming, the hub notices.
    drop(body_a);
    let more = read_events(&mut body_b, 2, Duration::from_secs(5)).await;
    assert_eq!(more.len(), 2);
    assert_eq!(store.hub_state(&live).0, 1);

    // Disconnect the last one: the ticker exits on its next tick.
    drop(body_b);
    tokio::time::sleep(folo_server::mock::TICK * 4).await;
    assert_eq!(store.hub_state(&live), (0, false));

    // A new subscriber restarts it; shutting the server down ends the stream.
    let res = app.clone().oneshot(get(&uri)).await.unwrap();
    let mut body = res.into_body();
    assert_eq!(read_events(&mut body, 1, Duration::from_secs(5)).await.len(), 1);
    server.close_streams();
    let end = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match body.frame().await {
                None => break,
                Some(_) => continue,
            }
        }
    })
    .await;
    assert!(end.is_ok(), "stream ended after close_streams");
}

#[tokio::test]
async fn sse_finished_run_and_errors() {
    let (server, store) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let failed = list.runs.iter().find(|r| r.status == RunStatus::Failed).unwrap();
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", failed.id))).await.unwrap();
    let mut body = res.into_body();
    let ev = read_events(&mut body, 1, Duration::from_secs(2)).await;
    let st: StatusEvent = serde_json::from_str(&ev[0].1).unwrap();
    assert_eq!(st.status, RunStatus::Failed);
    assert!(st.error.is_some());
    assert_eq!(store.hub_state(&failed.id), (1, false), "finished runs need no ticker");
    err(&app, get("/api/runs/run_none/events"), StatusCode::NOT_FOUND, "run_not_found").await;
}

#[tokio::test]
async fn real_store_disk_dataset_and_not_implemented() {
    let root = std::env::temp_dir().join(format!("folo-real-store-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let ds_dir = root.join("ds");
    std::fs::create_dir_all(ds_dir.join("images/train")).unwrap();
    std::fs::create_dir_all(ds_dir.join("labels/train")).unwrap();
    std::fs::write(ds_dir.join("data.yaml"), "names:\n  0: body\n  1: Head\n").unwrap();
    image::RgbImage::from_pixel(64, 32, image::Rgb([1, 2, 3]))
        .save(ds_dir.join("images/train/x.png"))
        .unwrap();
    std::fs::write(ds_dir.join("labels/train/x.txt"), "0 0.5 0.5 0.5 0.5\n1 0.5 0.25 0.1 0.1\n").unwrap();

    let data_dir = root.join("home");
    let server = Server::new(Arc::new(RealStore::open(data_dir.clone())), Options::default());
    let app = server.router();
    let h: Health = ok(&app, get("/api/health")).await;
    assert!(!h.mock);
    let (status, _, b) =
        send(&app, post_json("/api/datasets", json!({ "path": ds_dir.to_str().unwrap() }))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let ds: Dataset = serde_json::from_slice(&b).unwrap();
    assert_eq!((ds.name.as_str(), ds.format), ("ds", DatasetFormat::Yolo));
    assert_eq!(ds.classes, ["body", "Head"]);
    assert_eq!(ds.stats.unwrap().class_counts, [1, 1]);
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    assert_eq!(page.images[0].boxes[0].x1, 16.0);
    let (status, headers, _) = send(&app, get(&format!("/api/datasets/{}/images/0", ds.id))).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "image/png"));

    // Registrations persist across restarts.
    let reopened = Server::new(Arc::new(RealStore::open(data_dir.clone())), Options::default()).router();
    let list: DatasetList = ok(&reopened, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 1);

    let cfg = json!({ "dataset_id": ds.id });
    err(&app, post_json("/api/runs", cfg), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
    let p = json!({ "model_id": "m_1", "dataset_id": ds.id, "image_id": "0" });
    let e = err(&app, post_json("/api/predict", p), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
    assert!(e.hint.is_some());
    err(
        &app,
        post_json("/api/datasets/synthetic", json!({ "name": "s" })),
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented",
    )
    .await;
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert!(runs.runs.is_empty());
    let models: ModelList = ok(&app, get("/api/models")).await;
    assert!(models.models.is_empty());
    let s: SystemInfo = ok(&app, get("/api/system")).await;
    assert!(s.gpus.is_empty());

    let (status, _, _) =
        send(&app, Request::delete(format!("/api/datasets/{}", ds.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    assert!(list.datasets.is_empty());
    assert!(ds_dir.join("images/train/x.png").exists(), "unregistering never deletes files");
    std::fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn static_files_spa_fallback() {
    let dir = std::env::temp_dir().join(format!("folo-web-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::write(dir.join("index.html"), "<!doctype html><title>Studio</title>").unwrap();
    std::fs::write(dir.join("assets/app.js"), "console.log(1)").unwrap();
    let server =
        Server::new(Arc::new(MockStore::new()), Options { web_dir: Some(dir.clone()), cors: Cors::Loopback });
    let app = server.router();
    let (s, _, b) = send(&app, get("/")).await;
    assert_eq!(s, StatusCode::OK);
    assert!(String::from_utf8_lossy(&b).contains("Studio"));
    let (s, h, b) = send(&app, get("/assets/app.js")).await;
    assert_eq!((s, b.as_slice()), (StatusCode::OK, b"console.log(1)".as_slice()));
    assert!(h[header::CONTENT_TYPE].to_str().unwrap().contains("javascript"));
    let (s, _, b) = send(&app, get("/runs/run_1a01")).await;
    assert_eq!(s, StatusCode::OK);
    assert!(String::from_utf8_lossy(&b).contains("Studio"), "SPA fallback");
    err(&app, get("/api/not-a-route"), StatusCode::NOT_FOUND, "route_not_found").await;

    // Without a built web app: a plain 404 with a hint, API unaffected.
    let bare = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let (s, _, b) = send(&bare, get("/")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(String::from_utf8_lossy(&b).contains("not built"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn cors_and_gzip() {
    let (server, _) = mock();
    let app = server.router();
    let preflight = |origin: &str| {
        Request::builder()
            .method("OPTIONS")
            .uri("/api/runs")
            .header(header::ORIGIN, origin)
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type")
            .body(Body::empty())
            .unwrap()
    };
    let (_, h, _) = send(&app, preflight("http://localhost:5173")).await;
    assert_eq!(h[header::ACCESS_CONTROL_ALLOW_ORIGIN], "http://localhost:5173");
    let (_, h, _) = send(&app, preflight("https://evil.example")).await;
    assert!(h.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none());

    let any = Server::new(Arc::new(MockStore::new()), Options { web_dir: None, cors: Cors::Any }).router();
    let (_, h, _) = send(&any, preflight("https://elsewhere.example")).await;
    assert_eq!(h[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");

    let req = Request::get("/api/runs").header(header::ACCEPT_ENCODING, "gzip").body(Body::empty()).unwrap();
    let (s, h, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h[header::CONTENT_ENCODING], "gzip");

    // SSE is never compressed (it would buffer events).
    let list: RunList = ok(&app, get("/api/runs")).await;
    let req = Request::get(format!("/api/runs/{}/events", list.runs[0].id))
        .header(header::ACCEPT_ENCODING, "gzip")
        .body(Body::empty())
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert!(res.headers().get(header::CONTENT_ENCODING).is_none());
}

#[tokio::test]
async fn cross_origin_mutations_are_refused() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap().id.clone();
    // A "simple" form POST needs no preflight, so CORS alone would let it through.
    let form_post = |origin: &str, host: &str| {
        Request::post(format!("/api/runs/{live}/stop"))
            .header(header::ORIGIN, origin)
            .header(header::HOST, host)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from("a=b"))
            .unwrap()
    };
    err(&app, form_post("https://evil.example", "127.0.0.1:7878"), StatusCode::FORBIDDEN, "forbidden_origin")
        .await;
    err(&app, form_post("null", "127.0.0.1:7878"), StatusCode::FORBIDDEN, "forbidden_origin").await;
    let r: Run = ok(&app, get(&format!("/api/runs/{live}"))).await;
    assert_eq!(r.status, RunStatus::Running, "the refused request had no effect");
    // Reads are fine from anywhere (CORS hides the response).
    let req = Request::get("/api/health")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK);
    // Same origin (Studio served by this server on a LAN address) and loopback origins may mutate.
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let create = |origin: &str, host: &str| {
        Request::post("/api/runs")
            .header(header::ORIGIN, origin)
            .header(header::HOST, host)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "dataset_id": ds.datasets[0].id }).to_string()))
            .unwrap()
    };
    let (s, _, _) = send(&app, create("http://192.168.1.5:7878", "192.168.1.5:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
    let (s, _, _) = send(&app, create("http://localhost:5173", "127.0.0.1:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
    // --cors-any turns the guard off.
    let any = Server::new(Arc::new(MockStore::new()), Options { web_dir: None, cors: Cors::Any }).router();
    let (s, _, _) = send(&any, create("https://elsewhere.example", "127.0.0.1:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
}

#[tokio::test]
async fn malformed_paths_and_oversized_uploads_use_the_envelope() {
    let (server, _) = mock();
    let app = server.router();
    err(&app, get("/api/datasets/%FF"), StatusCode::BAD_REQUEST, "invalid_path").await;
    err(&app, get("/api/runs/%FF/metrics"), StatusCode::BAD_REQUEST, "invalid_path").await;

    let boundary = "XfoloBoundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\nm_31aa\r\n--{boundary}\r\n\
         Content-Disposition: form-data; name=\"image\"; filename=\"big.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
    )
    .into_bytes();
    body.resize(body.len() + 65 * 1024 * 1024, 0);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::post("/api/predict")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    err(&app, req, StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large").await;
}

#[tokio::test]
async fn predictions_stay_within_the_model_classes() {
    let (server, _) = mock();
    let app = server.router();
    // A shapes model (circle / square / triangle) on traffic images: it may only answer with its own classes.
    let model: Model = ok(&app, get("/api/models/m_31aa")).await;
    assert_eq!(model.classes.len(), 3);
    for i in 0..20 {
        let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_7c2f", "image_id": format!("{i:06}"), "conf": 0.0 });
        let p: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
        for b in &p.boxes {
            assert!((b.class as usize) < model.classes.len(), "{b:?}");
            assert_eq!(b.name, model.classes[b.class as usize]);
        }
    }
    // A matching model keeps the dataset's classes.
    let req = json!({ "model_id": "m_53d1", "dataset_id": "ds_3b9d", "image_id": "000700", "conf": 0.0 });
    let p: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
    assert!(p.boxes.iter().any(|b| b.name == "body"));
}

#[tokio::test]
async fn sse_first_subscriber_sees_queued_to_running() {
    let (server, _) = mock();
    let app = server.router();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let run: Run =
        ok(&app, post_json("/api/runs", json!({ "dataset_id": ds.datasets[0].id, "epochs": 3 }))).await;
    assert_eq!(run.status, RunStatus::Queued);
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", run.id))).await.unwrap();
    let mut body = res.into_body();
    let mut statuses = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
    'outer: loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let evs = read_events(&mut body, 1, left).await;
        assert!(!evs.is_empty(), "no step within 8 s; statuses {statuses:?}");
        for (name, data) in evs {
            match name.as_str() {
                "status" => statuses.push(serde_json::from_str::<StatusEvent>(&data).unwrap().status),
                "step" => break 'outer,
                _ => {}
            }
        }
    }
    assert_eq!(
        statuses,
        [RunStatus::Queued, RunStatus::Running],
        "every transition is reported before steps"
    );
}

#[tokio::test]
async fn datasets_without_classes_cannot_train() {
    let dir = std::env::temp_dir().join(format!("folo-empty-ds-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("images")).unwrap();
    let (server, _) = mock();
    let app = server.router();
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": dir.to_str().unwrap() }))).await;
    assert!(ds.classes.is_empty());
    let e = err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("no classes"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn malformed_registry_is_never_overwritten() {
    let root = std::env::temp_dir().join(format!("folo-bad-registry-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("ds/images")).unwrap();
    std::fs::write(root.join("datasets.json"), "{ not json").unwrap();
    let app = Server::new(Arc::new(RealStore::open(root.clone())), Options::default()).router();
    let body = json!({ "path": root.join("ds").to_str().unwrap() });
    err(&app, post_json("/api/datasets", body), StatusCode::INTERNAL_SERVER_ERROR, "internal").await;
    assert_eq!(std::fs::read_to_string(root.join("datasets.json")).unwrap(), "{ not json");
    std::fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn model_eval_curves_and_confusion() {
    let (server, _) = mock();
    let app = server.router();
    let models: ModelList = ok(&app, get("/api/models")).await;
    for m in &models.models {
        let ev: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        let n = m.classes.len();
        assert!(ev.images > 0, "{}", m.id);
        assert_eq!((ev.pr.len(), ev.per_class.len()), (n, n));
        for (c, curve) in ev.pr.iter().enumerate() {
            assert_eq!(curve.class as usize, c);
            assert_eq!((curve.recall.len(), curve.precision.len()), (101, 101));
            assert_eq!((curve.recall[0], curve.recall[100]), (0.0, 1.0));
            assert!(curve.precision.iter().all(|p| (0.0..=1.0).contains(p)));
            assert!(curve.precision.windows(2).all(|w| w[0] >= w[1]), "interpolated precision never rises");
            let mean = curve.precision.iter().sum::<f32>() / 101.0;
            assert!((curve.ap50 - mean).abs() < 1e-3);
            assert_eq!(ev.per_class[c].map50, curve.ap50);
            assert!(curve.ap50 > 0.2 && curve.ap50 < 1.0, "{} class {c}: {}", m.id, curve.ap50);
        }
        assert_eq!(ev.confusion.len(), n + 1);
        assert!(ev.confusion.iter().all(|row| row.len() == n + 1));
        assert_eq!(ev.confusion[n][n], 0);
        let truth: u32 = ev.confusion[..n].iter().flatten().sum();
        let diag: u32 = (0..n).map(|c| ev.confusion[c][c]).sum();
        assert!(truth > 0 && diag as f32 > 0.5 * truth as f32, "{}: mostly diagonal", m.id);
        for c in 0..n {
            let row = &ev.confusion[c];
            assert!((0..n).filter(|&j| j != c).all(|j| row[c] >= row[j]), "{} row {c}: {row:?}", m.id);
        }
        let again: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        assert_eq!(again, ev, "deterministic");
    }
    err(&app, get("/api/models/m_none/eval"), StatusCode::NOT_FOUND, "model_not_found").await;
}

#[tokio::test]
async fn run_logs_history() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let by = |s: RunStatus| list.runs.iter().find(|r| r.status == s).unwrap().clone();

    let done = by(RunStatus::Completed);
    let all: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=5000", done.id))).await;
    let msgs: Vec<&str> = all.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(msgs[0].starts_with("training "));
    assert!(msgs.last().unwrap().starts_with("training complete"));
    let epochs = msgs.iter().filter(|m| m.starts_with("epoch ")).count();
    assert_eq!(epochs as u32, done.epochs);
    assert!(all.lines.windows(2).all(|w| w[0].t <= w[1].t), "chronological");
    let tail: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=5", done.id))).await;
    assert_eq!(tail.lines.as_slice(), &all.lines[all.lines.len() - 5..]);
    let default: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", done.id))).await;
    assert_eq!(default.lines.len(), all.lines.len().min(500));

    let failed: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", by(RunStatus::Failed).id))).await;
    let last = failed.lines.last().unwrap();
    assert_eq!(last.level, LogLevel::Error);
    assert!(last.message.contains("out of memory"));
    let stopped: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", by(RunStatus::Stopped).id))).await;
    assert!(stopped.lines.last().unwrap().message.starts_with("stopped at step"));

    // The live run: history up to now, in the same format as the SSE `log` events.
    let live = by(RunStatus::Running);
    let hist: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", live.id))).await;
    let n = hist.lines.iter().filter(|l| l.message.starts_with("epoch ")).count() as u32;
    assert!(
        n + 1 >= live.epoch.saturating_sub(1) && n <= live.epoch,
        "{n} epochs logged, run at {}",
        live.epoch
    );

    err(&app, get(&format!("/api/runs/{}/logs?limit=x", done.id)), StatusCode::BAD_REQUEST, "invalid_query")
        .await;
    err(&app, get("/api/runs/run_none/logs"), StatusCode::NOT_FOUND, "run_not_found").await;
}

#[tokio::test]
async fn logs_of_a_run_stopped_while_queued_match_sse() {
    let (server, _) = mock();
    let app = server.router();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let run: Run = ok(&app, post_json("/api/runs", json!({ "dataset_id": ds.datasets[0].id }))).await;
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", run.id))).await.unwrap();
    let mut body = res.into_body();
    let stop = Request::post(format!("/api/runs/{}/stop", run.id)).body(Body::empty()).unwrap();
    let r: Run = ok(&app, stop).await;
    assert_eq!(r.status, RunStatus::Stopping);
    // Collect SSE log lines until the final `stopped` status.
    let mut sse_logs = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(6);
    'outer: loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let evs = read_events(&mut body, 1, left).await;
        assert!(!evs.is_empty(), "no stopped status; logs so far {sse_logs:?}");
        for (name, data) in evs {
            match name.as_str() {
                "log" => sse_logs.push(serde_json::from_str::<LogLine>(&data).unwrap().message),
                "status"
                    if serde_json::from_str::<StatusEvent>(&data).unwrap().status == RunStatus::Stopped =>
                {
                    break 'outer;
                }
                _ => {}
            }
        }
    }
    let hist: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    let hist: Vec<String> = hist.lines.into_iter().map(|l| l.message).collect();
    assert_eq!(hist.len(), 2, "{hist:?}");
    assert!(hist[0].starts_with("stop requested") && hist[1].starts_with("stopped at step 0"));
    assert_eq!(hist, sse_logs, "history matches what the stream sent");
}
