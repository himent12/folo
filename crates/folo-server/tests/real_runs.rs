//! The real store's runs, models and inference end to end on a tiny synthetic dataset (CPU, 64 px): create the
//! dataset, train over HTTP while reading the SSE stream, find the run's model, evaluate, export and predict with
//! it, stop / resume / delete runs, and reload everything after a store restart.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Options, RealStore, Server, Store};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Longest wait for a run to reach a state (a 2-epoch run here takes a few seconds).
const TIMEOUT: Duration = Duration::from_secs(300);

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("folo-runs-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn store(home: &Path) -> Arc<RealStore> {
    Arc::new(RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO).device("cpu"))
}

fn app(store: &Arc<RealStore>) -> Router {
    Server::new(store.clone(), Options::default()).router()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, body.collect().await.unwrap().to_bytes().to_vec())
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

fn post(uri: &str) -> Request<Body> {
    Request::post(uri).body(Body::empty()).unwrap()
}

fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().to_string();
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().to_string();
    let (got, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A synthetic dataset of `images` 64 px images (80 / 20 train / val) with two classes.
async fn dataset(app: &Router, name: &str, images: u32) -> Dataset {
    let body =
        json!({ "name": name, "images": images, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 });
    ok(app, post_json("/api/datasets/synthetic", body)).await
}

/// A tiny CPU run's config.
fn tiny(ds: &Dataset, epochs: u32) -> Value {
    json!({ "dataset_id": ds.id, "epochs": epochs, "batch": 4, "imgsz": 64, "device": "cpu",
            "augment": { "close_mosaic": 1 } })
}

/// An SSE stream of `GET /api/runs/{id}/events`.
struct Events {
    body: Body,
    buf: String,
}

impl Events {
    async fn open(app: &Router, run: &str) -> Events {
        let res = app.clone().oneshot(get(&format!("/api/runs/{run}/events"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        Events { body: res.into_body(), buf: String::new() }
    }

    /// The next event (name, data), skipping comments.
    async fn next(&mut self) -> (String, Value) {
        loop {
            if let Some(end) = self.buf.find("\n\n") {
                let block: String = self.buf.drain(..end + 2).collect();
                let (mut name, mut data) = (None, String::new());
                for line in block.lines() {
                    if let Some(n) = line.strip_prefix("event:") {
                        name = Some(n.trim().to_string());
                    } else if let Some(d) = line.strip_prefix("data:") {
                        data.push_str(d.trim_start());
                    }
                }
                if let Some(name) = name {
                    return (name, serde_json::from_str(&data).unwrap());
                }
                continue;
            }
            let frame = tokio::time::timeout(TIMEOUT, self.body.frame())
                .await
                .expect("SSE event in time")
                .expect("SSE stream open")
                .unwrap();
            if let Ok(data) = frame.into_data() {
                self.buf.push_str(&String::from_utf8_lossy(&data).replace("\r\n", "\n"));
            }
        }
    }

    /// Reads events until a `status` event with one of `statuses`; returns every event read.
    async fn until_status(&mut self, statuses: &[&str]) -> Vec<(String, Value)> {
        let mut seen = Vec::new();
        loop {
            let ev = self.next().await;
            let done = ev.0 == "status" && statuses.contains(&ev.1["status"].as_str().unwrap());
            seen.push(ev);
            if done {
                return seen;
            }
        }
    }

    /// Reads events until the first `epoch` event.
    async fn until_epoch(&mut self) {
        while self.next().await.0 != "epoch" {}
    }
}

fn count(events: &[(String, Value)], name: &str) -> usize {
    events.iter().filter(|(n, _)| n == name).count()
}

#[tokio::test(flavor = "multi_thread")]
async fn train_evaluate_predict_and_reload() {
    let t = TempDir::new("train");
    let home = t.0.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "tiny", 12).await;
    assert_eq!(ds.splits, Splits { train: 10, val: 2, test: 0 });

    // Validation against the dataset and the engine's limits.
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": "ds_none" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
    let e = err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id, "imgsz": 48 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("imgsz"), "{}", e.message);
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id, "epochs": 0 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    let w = json!({ "dataset_id": ds.id, "weights": "m_none" });
    err(&app, post_json("/api/runs", w), StatusCode::NOT_FOUND, "model_not_found").await;
    err(
        &app,
        post_json("/api/runs", json!({ "epochs": 2 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    assert!(ok::<RunList>(&app, get("/api/runs")).await.runs.is_empty(), "rejected configs leave no run");

    // A 2-epoch run, watched over SSE from the start.
    let (status, b) = send(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let run: Run = serde_json::from_slice(&b).unwrap();
    assert_eq!(run.status, RunStatus::Queued);
    assert_eq!(run.name, "tiny-n");
    assert_eq!((run.epochs, run.config.imgsz), (2, 64));
    assert!(run.device.contains("cpu"), "{}", run.device);
    let mut sse = Events::open(&app, &run.id).await;
    let first = sse.next().await;
    assert_eq!(first.0, "status");
    let events = sse.until_status(&["completed", "failed"]).await;
    let last = &events.last().unwrap().1;
    assert_eq!(last["status"], "completed", "{last}: {events:?}");
    assert_eq!(count(&events, "epoch"), 2);
    assert!(count(&events, "step") >= 2 && count(&events, "log") >= 1);
    // Steps and the trainer's statuses carry the run's length (3 batches of 4 out of 10 images, 2 epochs).
    for (name, v) in events.iter().filter(|(n, v)| n == "step" || (n == "status" && v["status"] != "queued"))
    {
        assert_eq!((v["steps_per_epoch"].as_u64(), v["epochs"].as_u64()), (Some(3), Some(2)), "{name} {v}");
    }

    let done: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!((done.epoch, done.steps_per_epoch), (2, 3));
    assert_eq!(done.step, 6);
    assert!(done.started.is_some() && done.finished.is_some() && done.eta_s.is_none());
    let best = done.best.unwrap();
    let model_id = done.model_id.clone().unwrap();
    assert_eq!(model_id, format!("m_{}", run.id.trim_start_matches("run_")));
    for f in
        ["run.json", "config.json", "metrics.jsonl", "log.jsonl", "weights/best.folo", "weights/last.folo"]
    {
        assert!(home.join("runs").join(&run.id).join(f).is_file(), "{f}");
    }

    // History: every step, both epochs, the log.
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", run.id))).await;
    assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2]);
    let best_epoch = &m.epochs[best.epoch as usize - 1];
    assert_eq!(best_epoch.val.map50_95, best.map50_95);
    let thin: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics?every=4", run.id))).await;
    assert_eq!(thin.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [4, 6]);
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    assert_eq!(logs.lines[0].message, "queued");
    assert!(logs.lines.iter().any(|l| l.message.contains("finished 2 epochs")), "{:?}", logs.lines);
    let two: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=2", run.id))).await;
    assert_eq!(two.lines.len(), 2);
    assert_eq!(two.lines[1], *logs.lines.last().unwrap());

    // Validation samples: the validation images, with dataset image ids.
    let samples: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples", run.id))).await;
    assert_eq!(samples.epoch, 2);
    assert_eq!(samples.samples.len(), 2);
    let val: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    for (s, im) in samples.samples.iter().zip(&val.images) {
        assert_eq!((s.image_id.as_str(), s.dataset_id.as_str()), (im.id.as_str(), ds.id.as_str()));
        assert_eq!((s.width, s.height), (64, 64));
        assert_eq!(s.truth.len(), im.boxes.len());
        assert!(s.pred.iter().all(|p| p.score >= 0.25));
    }
    let e1: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples?epoch=1", run.id))).await;
    assert_eq!(e1.epoch, 1);
    let uri = format!("/api/runs/{}/samples?epoch=3", run.id);
    err(&app, get(&uri), StatusCode::NOT_FOUND, "epoch_not_available").await;

    // Its models: best and last.
    let models: ModelList = ok(&app, get("/api/models")).await;
    let ids: Vec<&str> = models.models.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&model_id.as_str()) && ids.contains(&format!("{model_id}-last").as_str()),
        "{ids:?}"
    );
    let model: Model = ok(&app, get(&format!("/api/models/{model_id}"))).await;
    assert_eq!((model.arch, model.imgsz, model.name.as_str()), (Arch::N, 64, "tiny-n-best"));
    assert_eq!(model.classes, ["circle", "square"]);
    assert_eq!(model.run_id.as_deref(), Some(run.id.as_str()));
    assert!(model.params > 2_000_000 && model.bytes > 0 && model.flops_g > 0.0);
    assert_eq!(model.metrics.unwrap().map50_95, best.map50_95);
    let last: Model = ok(&app, get(&format!("/api/models/{model_id}-last"))).await;
    assert_eq!(last.metrics.unwrap().map50_95, m.epochs[1].val.map50_95);
    err(&app, get("/api/models/m_none"), StatusCode::NOT_FOUND, "model_not_found").await;

    // Its weights: the checkpoint as stored, and the f16 export (cached under cache/weights/).
    let (status, body) = send(&app, get(&format!("/api/models/{model_id}/weights"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, std::fs::read(home.join("runs").join(&run.id).join("weights/best.folo")).unwrap());
    let (status, f16) = send(&app, get(&format!("/api/models/{model_id}/weights?format=f16"))).await;
    assert_eq!(status, StatusCode::OK);
    let file = folo_nn::io::ModelFile::from_bytes(f16.clone()).unwrap();
    assert_eq!(file.metadata().dtype, folo_nn::io::FloatDType::F16);
    assert!(file.metadata().fused && f16.len() < body.len() / 2, "{} vs {}", f16.len(), body.len());
    let cached = || {
        std::fs::read_dir(home.join("cache/weights"))
            .map(|d| {
                d.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{model_id}.")))
                    .count()
            })
            .unwrap_or(0)
    };
    assert_eq!(cached(), 1);

    // Eval on the validation split (cached), export.
    let eval: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(eval.images, 2);
    assert_eq!(eval.confusion.len(), 3);
    assert!(eval.confusion.iter().all(|r| r.len() == 3) && eval.confusion[2][2] == 0);
    for c in &eval.pr {
        assert_eq!((c.recall.len(), c.precision.len()), (101, 101));
        assert!(c.precision.windows(2).all(|w| w[0] >= w[1]), "the envelope never rises");
    }
    let again: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(again, eval);
    assert!(home.join("cache/eval").join(format!("{model_id}.json")).is_file());
    // POST /api/val: the same split and thresholds as /eval give the same numbers; other splits and errors.
    let v: ValReport =
        ok(&app, post_json("/api/val", json!({ "model_id": model_id, "dataset_id": ds.id }))).await;
    assert_eq!((v.split, v.images, v.imgsz), (Split::Val, 2, 64));
    assert_eq!(v.val.per_class, eval.per_class);
    assert!(v.timing.total_s > 0.0 && v.timing.per_image_ms > 0.0);
    let body =
        json!({ "model_id": model_id, "dataset_id": ds.id, "split": "train", "imgsz": 96, "conf": 0.25 });
    let t: ValReport = ok(&app, post_json("/api/val", body)).await;
    assert_eq!((t.split, t.images, t.imgsz), (Split::Train, 10, 96));
    for (body, status, code) in [
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "split": "test" }),
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
        ),
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "imgsz": 100 }),
            StatusCode::BAD_REQUEST,
            "invalid_body",
        ),
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "iou": 1.5 }),
            StatusCode::BAD_REQUEST,
            "invalid_body",
        ),
        (json!({ "model_id": "m_none", "dataset_id": ds.id }), StatusCode::NOT_FOUND, "model_not_found"),
        (
            json!({ "model_id": model_id, "dataset_id": "ds_none" }),
            StatusCode::NOT_FOUND,
            "dataset_not_found",
        ),
    ] {
        err(&app, post_json("/api/val", body), status, code).await;
    }

    let x: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "folo-f16" }))).await;
    assert_eq!(Path::new(&x.path), home.join("exports").join("tiny-n-best-f16.folo"));
    assert_eq!(std::fs::metadata(&x.path).unwrap().len(), x.bytes);
    let x2: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "folo-f16" }))).await;
    assert_ne!(x2.path, x.path, "exports never overwrite");
    let onnx: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "onnx" }))).await;
    assert_eq!(Path::new(&onnx.path), home.join("exports").join("tiny-n-best.onnx"));
    let bytes = std::fs::read(&onnx.path).unwrap();
    assert_eq!(bytes.len() as u64, onnx.bytes);
    // A ModelProto: field 1 (ir_version) = 8 first.
    assert_eq!(&bytes[..2], &[0x08, 8]);

    // Predict: a dataset image (JSON) and an upload (multipart).
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r.model_id, model_id);
    assert_eq!((r.image.width, r.image.height, r.device.as_str()), (64, 64, "cpu"));
    assert!(!r.boxes.is_empty(), "conf 0.001 keeps low-score boxes");
    assert!(r.boxes.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(r.boxes.iter().all(|b| b.score >= 0.001 && ["circle", "square"].contains(&b.name.as_str())));
    let ti = r.timing;
    assert!(ti.inference_ms > 0.0 && ti.total_ms >= ti.preprocess_ms + ti.inference_ms);
    let png = std::fs::read(home.join("datasets/tiny").join(&val.images[0].file)).unwrap();
    let r2: PredictResponse =
        ok(&app, multipart(&model_id, Some(("image.jpg", png)), &[("conf", "0.001")])).await;
    assert_eq!((r2.image.width, r2.image.height), (64, 64));
    assert!(!r2.boxes.is_empty());
    let bad = multipart(&model_id, Some(("x.jpg", b"not an image".to_vec())), &[]);
    err(&app, bad, StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed").await;
    let p = json!({ "model_id": "m_none", "dataset_id": ds.id, "image_id": "0" });
    err(&app, post_json("/api/predict", p), StatusCode::NOT_FOUND, "model_not_found").await;

    // Fine-tuning from the run's model.
    let mut ft = tiny(&ds, 1);
    ft["weights"] = json!(model_id);
    ft["name"] = json!("tuned");
    let tuned: Run = ok(&app, post_json("/api/runs", ft)).await;
    let mut sse = Events::open(&app, &tuned.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed");
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", tuned.id))).await;
    assert!(logs.lines.iter().any(|l| l.message.starts_with("transferred")), "{:?}", logs.lines);
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(
        runs.runs.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        [&tuned.id, &run.id],
        "newest first"
    );

    // Everything is back after a restart.
    let system: SystemInfo = ok(&app, get("/api/system")).await;
    drop(sse);
    drop(app);
    s.shutdown();
    drop(s);
    let s = store(&home);
    let app = self::app(&s);
    let again: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(again, done);
    let m2: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", run.id))).await;
    assert_eq!(m2, m);
    let models2: ModelList = ok(&app, get("/api/models")).await;
    assert_eq!(models2.models.len(), models.models.len() + 2, "the fine-tuned run's two models too");
    let eval2: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(eval2, eval);
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r3: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r3.boxes, r.boxes, "same weights, same image, same boxes");
    assert_eq!(ok::<SystemInfo>(&app, get("/api/system")).await.gpus, system.gpus);

    // Deleting a finished run removes it, its folder and its models (their cached f16 weights too).
    let tuned_model = format!("m_{}", tuned.id.trim_start_matches("run_"));
    let (status, _) = send(&app, get(&format!("/api/models/{tuned_model}/weights?format=f16"))).await;
    assert_eq!(status, StatusCode::OK);
    let tuned_cache = || {
        std::fs::read_dir(home.join("cache/weights"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{tuned_model}.")))
            .count()
    };
    assert_eq!(tuned_cache(), 1);
    let (status, _) = send(&app, delete(&format!("/api/runs/{}", tuned.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!home.join("runs").join(&tuned.id).exists());
    err(&app, get(&format!("/api/runs/{}", tuned.id)), StatusCode::NOT_FOUND, "run_not_found").await;
    let r = json!({ "resume_run_id": tuned.id });
    err(&app, post_json("/api/runs", r), StatusCode::NOT_FOUND, "run_not_found").await;
    assert_eq!(ok::<ModelList>(&app, get("/api/models")).await.models.len(), models.models.len());
    assert_eq!(tuned_cache(), 0);
    let uri = format!("/api/models/{tuned_model}/weights");
    err(&app, get(&uri), StatusCode::NOT_FOUND, "model_not_found").await;

    // A dataset whose classes changed order since training is not evaluated against the old labels.
    let yaml = home.join("datasets/tiny/data.yaml");
    let text = std::fs::read_to_string(&yaml).unwrap();
    std::fs::write(
        &yaml,
        text.replace("\"circle\"", "\"tmp\"")
            .replace("\"square\"", "\"circle\"")
            .replace("\"tmp\"", "\"square\""),
    )
    .unwrap();
    let e =
        err(&app, get(&format!("/api/models/{model_id}/eval")), StatusCode::BAD_REQUEST, "invalid_dataset")
            .await;
    assert!(e.message.contains("same order"), "{}", e.message);
}

/// A `multipart/form-data` predict request.
fn multipart(model_id: &str, file: Option<(&str, Vec<u8>)>, fields: &[(&str, &str)]) -> Request<Body> {
    let boundary = "folo-test-boundary";
    let mut body = Vec::new();
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    };
    field("model_id", model_id);
    for (k, v) in fields {
        field(k, v);
    }
    if let Some((name, bytes)) = file {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: \
                 application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::post("/api/predict")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn queue_stop_resume_and_restart() {
    let t = TempDir::new("stop");
    let home = t.0.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "stoppy", 12).await;

    // One run at a time: the second waits in the queue, and stopping it there is immediate.
    let a: Run = ok(&app, post_json("/api/runs", tiny(&ds, 3))).await;
    let b: Run = ok(&app, post_json("/api/runs", tiny(&ds, 3))).await;
    let mut sse = Events::open(&app, &a.id).await;
    sse.until_epoch().await;
    assert_eq!(ok::<Run>(&app, get(&format!("/api/runs/{}", b.id))).await.status, RunStatus::Queued);
    let stopped_b: Run = ok(&app, post(&format!("/api/runs/{}/stop", b.id))).await;
    assert_eq!(stopped_b.status, RunStatus::Stopped);
    assert!(stopped_b.model_id.is_none() && stopped_b.started.is_none());
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", b.id))).await;
    assert!(logs.lines[0].message.starts_with("queued behind 1"), "{:?}", logs.lines);

    // A running run cannot be deleted; a stop takes effect after the current batch and keeps the checkpoint.
    err(&app, delete(&format!("/api/runs/{}", a.id)), StatusCode::CONFLICT, "run_active").await;
    let stopping: Run = ok(&app, post(&format!("/api/runs/{}/stop", a.id))).await;
    assert!(matches!(stopping.status, RunStatus::Stopping | RunStatus::Stopped), "{:?}", stopping.status);
    // Stopping again while it stops answers at once (with the run, or 409 once it has stopped).
    let again =
        tokio::time::timeout(Duration::from_secs(10), send(&app, post(&format!("/api/runs/{}/stop", a.id))))
            .await
            .expect("a repeated stop answers");
    assert!(matches!(again.0, StatusCode::OK | StatusCode::CONFLICT), "{}", again.0);
    let events = sse.until_status(&["stopped", "completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "stopped", "{events:?}");
    let a1: Run = ok(&app, get(&format!("/api/runs/{}", a.id))).await;
    assert_eq!(a1.status, RunStatus::Stopped);
    assert!((1..3).contains(&a1.epoch), "{}", a1.epoch);
    assert!(a1.model_id.is_some(), "stopped after a validated epoch");
    err(&app, post(&format!("/api/runs/{}/stop", a.id)), StatusCode::CONFLICT, "run_not_active").await;

    // Resume: same id, the stored config (only `device` may change), on to the last epoch.
    let resume = |fields: Value| post_json("/api/runs", fields);
    let e = err(
        &app,
        resume(json!({ "resume_run_id": a.id, "epochs": 5 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("epochs"), "{}", e.message);
    err(&app, resume(json!({ "resume_run_id": "run_none" })), StatusCode::NOT_FOUND, "run_not_found").await;
    let (status, body) =
        send(&app, resume(json!({ "resume_run_id": a.id, "dataset_id": ds.id, "device": "cpu" }))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let resumed: Run = serde_json::from_slice(&body).unwrap();
    assert_eq!((resumed.id.as_str(), resumed.status), (a.id.as_str(), RunStatus::Queued));
    let events = sse.until_status(&["completed", "failed", "stopped"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let a2: Run = ok(&app, get(&format!("/api/runs/{}", a.id))).await;
    assert_eq!((a2.status, a2.epoch, a2.step), (RunStatus::Completed, 3, 9));
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", a.id))).await;
    assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2, 3]);
    assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), (1..=9).collect::<Vec<_>>());
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", a.id))).await;
    assert!(logs.lines.iter().any(|l| l.message == "resuming from weights/last.folo"));
    assert!(logs.lines.iter().any(|l| l.message.starts_with("resumed from")));
    err(&app, resume(json!({ "resume_run_id": a.id })), StatusCode::CONFLICT, "run_not_resumable").await;

    // A run that was never trained resumes from scratch.
    let (status, body) = send(&app, resume(json!({ "resume_run_id": b.id }))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let mut sse_b = Events::open(&app, &b.id).await;
    sse_b.until_epoch().await;

    // Shutting the store down stops the active run gracefully.
    drop(sse);
    drop(sse_b);
    drop(app);
    s.shutdown();
    let b1 = s.get_run(&b.id).unwrap();
    assert_eq!(b1.status, RunStatus::Stopped);
    drop(s);

    // A run a crashed server left `running` comes back `stopped` (and resumable).
    let path = home.join("runs").join(&b.id).join("run.json");
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["run"]["status"] = json!("running");
    file["run"]["finished"] = Value::Null;
    // ... and that crashed after writing its checkpoint, before the epoch reached metrics.jsonl and run.json.
    let ckpt_epochs = file["epochs_done"].as_u64().unwrap() as u32;
    assert!(ckpt_epochs >= 1);
    file["epochs_done"] = json!(0);
    file["run"]["best"] = Value::Null;
    file["last_metrics"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let metrics = home.join("runs").join(&b.id).join("metrics.jsonl");
    let text = std::fs::read_to_string(&metrics).unwrap();
    let cut = text.rfind("{\"event\":\"epoch\"").unwrap();
    let tail = &text[cut..];
    let rest = tail.find('\n').map_or("", |n| &tail[n + 1..]);
    std::fs::write(&metrics, format!("{}{rest}", &text[..cut])).unwrap();
    let s = store(&home);
    let app = self::app(&s);
    let b2: Run = ok(&app, get(&format!("/api/runs/{}", b.id))).await;
    assert_eq!(b2.status, RunStatus::Stopped);
    assert!(b2.finished.is_some());
    assert_eq!(b2.epoch, ckpt_epochs, "progress comes back from the checkpoint");
    assert!(b2.best.is_some());
    let b_last = format!("{}-last", b2.model_id.clone().unwrap().trim_end_matches("-last"));
    let last: Model = ok(&app, get(&format!("/api/models/{b_last}"))).await;
    assert_eq!(last.metrics, None, "the last epoch's metrics are unknown, not the best epoch's");
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", b.id))).await;
    assert!(logs.lines.last().unwrap().message.contains("server stopped while the run was running"));
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(runs.runs.len(), 2);
    let r: Run = ok(&app, resume(json!({ "resume_run_id": b.id }))).await;
    assert_eq!(r.status, RunStatus::Queued);
    let mut sse = Events::open(&app, &b.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let done: Run = ok(&app, get(&format!("/api/runs/{}", b.id))).await;
    assert_eq!(done.epoch, 3);
    let last: Model = ok(&app, get(&format!("/api/models/{b_last}"))).await;
    assert!(last.metrics.is_some(), "known again once an epoch was validated");
}

/// `device: auto` on a machine with a GPU: the run trains there and predictions run there. Skips (with a note)
/// without a hardware adapter, like folo-vision's GPU tests.
#[tokio::test(flavor = "multi_thread")]
async fn auto_trains_and_predicts_on_the_gpu() {
    match folo_gpu::GpuContext::shared() {
        Ok(ctx) if ctx.summary().device_type != "Cpu" => {}
        Ok(ctx) => return eprintln!("note: skipping, software adapter {}", ctx.summary().name),
        Err(e) => return eprintln!("note: skipping, no GPU adapter: {e}"),
    }
    let t = TempDir::new("gpu");
    let home = t.0.join("home");
    let s = Arc::new(RealStore::open(home.clone()).recheck_every(Duration::ZERO));
    let app = app(&s);
    let ds = dataset(&app, "gpu", 12).await;
    let mut cfg = tiny(&ds, 1);
    cfg["device"] = json!("auto");
    let run: Run = ok(&app, post_json("/api/runs", cfg)).await;
    let gpu = folo_gpu::GpuContext::shared().unwrap().summary().name.clone();
    assert!(run.device.starts_with(&gpu), "{} on {gpu}", run.device);
    let mut sse = Events::open(&app, &run.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let model_id = ok::<Run>(&app, get(&format!("/api/runs/{}", run.id))).await.model_id.unwrap();
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r.device, "gpu:0");
    assert!(!r.boxes.is_empty());
    let system: SystemInfo = ok(&app, get("/api/system")).await;
    assert_eq!(system.gpus.first().map(|g| g.name.as_str()), Some(gpu.as_str()));
}

#[tokio::test(flavor = "multi_thread")]
async fn no_runs_start_after_shutdown() {
    let t = TempDir::new("shutdown");
    let s = store(&t.0.join("home"));
    let app = app(&s);
    let ds = dataset(&app, "tiny", 4).await;
    let stopped: Run = ok(&app, post_json("/api/runs", tiny(&ds, 1))).await;
    ok::<Run>(&app, post(&format!("/api/runs/{}/stop", stopped.id))).await;
    let s2 = s.clone();
    tokio::task::spawn_blocking(move || s2.shutdown()).await.unwrap();
    let e = err(&app, post_json("/api/runs", tiny(&ds, 1)), StatusCode::SERVICE_UNAVAILABLE, "shutting_down")
        .await;
    assert!(e.hint.is_some());
    let resume = json!({ "resume_run_id": stopped.id });
    err(&app, post_json("/api/runs", resume), StatusCode::SERVICE_UNAVAILABLE, "shutting_down").await;
    assert_eq!(s.list_runs().unwrap().len(), 1, "a refused run is not created");
}

/// A run the trainer refuses asynchronously keeps the engine's error code: an `amp` run on the CPU fails with
/// `not_supported` in its final `status` event and in `Run.error_code`, not as a bare message.
#[tokio::test(flavor = "multi_thread")]
async fn amp_run_without_tensor_cores_fails_with_not_supported() {
    let t = TempDir::new("amp");
    let s = store(&t.0.join("home"));
    let app = app(&s);
    let ds = dataset(&app, "amp", 4).await;
    let mut cfg = tiny(&ds, 1);
    cfg["amp"] = json!(true);
    let run: Run = ok(&app, post_json("/api/runs", cfg)).await;
    assert!(run.config.amp && run.error_code.is_none());
    let mut sse = Events::open(&app, &run.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    let last = &events.last().unwrap().1;
    assert_eq!(
        (last["status"].as_str(), last["error_code"].as_str()),
        (Some("failed"), Some("not_supported"))
    );
    assert!(last["error"].as_str().unwrap().contains("tensor cores"), "{last}");
    let got: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!((got.status, got.error_code.as_deref()), (RunStatus::Failed, Some("not_supported")));
    // The code survives a store restart (it is part of the persisted run).
    drop(app);
    tokio::task::spawn_blocking(move || s.shutdown()).await.unwrap();
    let app = self::app(&store(&t.0.join("home")));
    let again: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(again.error_code.as_deref(), Some("not_supported"));
}
