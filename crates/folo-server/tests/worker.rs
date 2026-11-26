//! What a cloud worker serves its control plane, on real runs (CPU, 64 px): a run's history through cursors
//! (`after_step`, `after_epoch`, `after`, `total`), and the cloud resume path in miniature: a run stopped after an
//! epoch on one server, its `last.folo` uploaded to a second one with the dataset, continued there with
//! `checkpoint` to the end. The mock's cursors and its refusal of `checkpoint` too.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{MockStore, Options, RealStore, Server};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Longest wait for a run to reach a state.
const TIMEOUT: Duration = Duration::from_secs(300);

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

/// Polls run `id` until `done` holds.
async fn wait_run(app: &Router, id: &str, done: impl Fn(&Run) -> bool) -> Run {
    let t0 = Instant::now();
    loop {
        let run: Run = ok(app, get(&format!("/api/runs/{id}"))).await;
        if done(&run) {
            return run;
        }
        assert!(t0.elapsed() < TIMEOUT, "run {id} stuck: {run:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A tiny CPU run's config on dataset `ds` (3 steps an epoch on the test's 5 training images).
fn tiny(ds: &str, epochs: u32) -> Value {
    json!({ "dataset_id": ds, "epochs": epochs, "batch": 2, "imgsz": 64, "device": "cpu",
            "augment": { "close_mosaic": 1 } })
}

/// Uploads `bytes` as `kind` named `name`, in parts of the largest size the server takes.
async fn upload(app: &Router, kind: &str, name: &str, bytes: &[u8]) -> Uploaded {
    let created: UploadCreated =
        ok(app, post_json("/api/uploads", json!({ "kind": kind, "name": name }))).await;
    let u = created.upload.unwrap();
    for (n, chunk) in bytes.chunks(u.max_part_bytes as usize).enumerate() {
        let part = Request::put(format!("/api/uploads/{}/parts/{n}", u.id))
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(chunk.to_vec()))
            .unwrap();
        ok::<Upload>(app, part).await;
    }
    ok(app, post(&format!("/api/uploads/{}/complete", u.id))).await
}

#[tokio::test(flavor = "multi_thread")]
async fn a_checkpoint_continues_on_another_server_and_history_has_cursors() {
    let t = TestDir::new("worker");
    let (a, b) = (store(&t.join("a")), store(&t.join("b")));
    let (app_a, app_b) = (app(&a), app(&b));
    let body =
        json!({ "name": "tiny", "images": 6, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 });
    let ds: Dataset = ok(&app_a, post_json("/api/datasets/synthetic", body)).await;

    // On A: a 2-epoch run stopped after its first epoch.
    a.hold_runs_after_epochs(true);
    let run: Run = ok(&app_a, post_json("/api/runs", tiny(&ds.id, 2))).await;
    wait_run(&app_a, &run.id, |r| r.best.is_some()).await;
    ok::<Run>(&app_a, post(&format!("/api/runs/{}/stop", run.id))).await;
    let stopped = wait_run(&app_a, &run.id, |r| r.status == RunStatus::Stopped).await;
    a.hold_runs_after_epochs(false);
    assert_eq!((stopped.epoch, stopped.steps_per_epoch), (1, 3));

    // Cursors: steps after a step, epochs after an epoch, applied before `every`.
    let metrics = |q: &str| get(&format!("/api/runs/{}/metrics{q}", run.id));
    let all: RunMetrics = ok(&app_a, metrics("")).await;
    let steps = |m: &RunMetrics| m.steps.iter().map(|s| s.step).collect::<Vec<_>>();
    let epochs = |m: &RunMetrics| m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>();
    assert_eq!((steps(&all), epochs(&all)), (vec![1, 2, 3], vec![1]));
    assert_eq!(steps(&ok(&app_a, metrics("?after_step=1")).await), [2, 3]);
    assert_eq!(steps(&ok(&app_a, metrics("?after_step=3")).await), Vec::<u64>::new());
    let thin: RunMetrics = ok(&app_a, metrics("?after_step=0&every=2&after_epoch=1")).await;
    assert_eq!((steps(&thin), epochs(&thin)), (vec![2, 3], vec![]));
    assert_eq!(epochs(&ok(&app_a, metrics("?after_epoch=0")).await), [1]);
    err(&app_a, metrics("?after_step=x"), StatusCode::BAD_REQUEST, "invalid_query").await;
    // Logs: `total` counts every line; `after` answers the lines after that many.
    let logs = |q: &str| get(&format!("/api/runs/{}/logs{q}", run.id));
    let full: RunLogs = ok(&app_a, logs("?limit=5000")).await;
    let total = full.total;
    assert!(total >= 4 && total == full.lines.len() as u64, "{full:?}");
    let none: RunLogs = ok(&app_a, logs(&format!("?after={total}"))).await;
    assert_eq!((none.lines.len(), none.total), (0, total));
    let two: RunLogs = ok(&app_a, logs(&format!("?after={}", total - 2))).await;
    assert_eq!(two.lines, full.lines[full.lines.len() - 2..]);
    let first: RunLogs = ok(&app_a, logs("?after=0&limit=3")).await;
    assert_eq!(first.lines, full.lines[..3]);
    assert_eq!(first.total, total);
    // Paginated cursor polling advances by `after + lines.len()` (not `total`, which stays the full log length).
    let mut cursor = 0u64;
    let mut paged = Vec::new();
    while cursor < total {
        let page: RunLogs = ok(&app_a, logs(&format!("?after={cursor}&limit=1"))).await;
        assert_eq!(page.total, total);
        assert_eq!(page.lines.len(), 1);
        cursor += page.lines.len() as u64;
        paged.extend(page.lines);
    }
    assert_eq!(paged, full.lines);
    assert!(ok::<RunLogs>(&app_a, logs(&format!("?after={}", total + 10))).await.lines.is_empty());
    let tail: RunLogs = ok(&app_a, logs("?limit=2")).await;
    assert_eq!((tail.lines.as_slice(), tail.total), (two.lines.as_slice(), total));

    // To B: the dataset (packed, as a control plane sends it) and the checkpoint.
    let src = folo_data::open(&ds.path).unwrap();
    let mut packed = Vec::new();
    folo_data::pack::pack(&src, &mut packed).unwrap();
    let ds_b = upload(&app_b, "dataset", "tiny", &packed).await.dataset.unwrap();
    let last = format!("/api/models/m_{}-last/weights", run.id.trim_start_matches("run_"));
    let (status, ckpt) = send(&app_a, get(&last)).await;
    assert_eq!(status, StatusCode::OK);
    let model = upload(&app_b, "model", "tiny-last", &ckpt).await.model.unwrap();

    // What `checkpoint` refuses.
    let with = |extra: Value| {
        let mut body = tiny(&ds_b.id, 2);
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        post_json("/api/runs", body)
    };
    for extra in [
        json!({ "checkpoint": model.id, "resume_run_id": run.id }),
        json!({ "checkpoint": model.id, "weights": model.id }),
    ] {
        let e = err(&app_b, with(extra), StatusCode::BAD_REQUEST, "invalid_body").await;
        assert!(e.message.contains("cannot be combined"), "{}", e.message);
    }
    err(&app_b, with(json!({ "checkpoint": "m_none" })), StatusCode::NOT_FOUND, "model_not_found").await;
    err(&app_b, with(json!({ "checkpoint": 7 })), StatusCode::UNPROCESSABLE_ENTITY, "invalid_body").await;
    let plain = t.join("b").join("models").join("plain.folo");
    let config = folo_nn::io::ModelConfig::new("n", 2).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(config, &mut folo_core::rng::Rng::seed(1)).unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.save(&plain).unwrap();
    let plain_id = ok::<ModelList>(&app_b, get("/api/models"))
        .await
        .models
        .into_iter()
        .find(|x| x.name == "plain")
        .unwrap()
        .id;
    let e = err(
        &app_b,
        with(json!({ "checkpoint": plain_id })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_weights",
    )
    .await;
    assert!(e.message.contains("no training state"), "{}", e.message);
    assert!(ok::<RunList>(&app_b, get("/api/runs")).await.runs.is_empty(), "refused requests leave no run");

    // On B: a new run that continues at epoch 2 (the step numbers too) and completes.
    // Even if the request asks for a different epoch count (5), the queued response immediately reports the
    // checkpoint's real schedule (2 epochs).
    let (status, body) = send(&app_b, with(json!({ "checkpoint": model.id, "epochs": 5 }))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&body));
    let cont: Run = serde_json::from_slice(&body).unwrap();
    assert_eq!((cont.status, cont.epochs, cont.best), (RunStatus::Queued, 2, stopped.best));
    let dir = t.join("b").join("runs").join(&cont.id);
    let done = wait_run(&app_b, &cont.id, |r| !r.status.is_active()).await;
    assert_eq!(
        (done.status, done.epoch, done.epochs, done.step),
        (RunStatus::Completed, 2, 2, 6),
        "{done:?}"
    );
    let m: RunMetrics = ok(&app_b, get(&format!("/api/runs/{}/metrics", cont.id))).await;
    assert_eq!((steps(&m), epochs(&m)), ((4..=6).collect::<Vec<_>>(), vec![2]));
    // `Run.best` must stay consistent with the checkpoint's recorded best epoch (a tying epoch 2 must not
    // overwrite epoch 1 as best).
    let final_ckpt = folo_nn::io::ModelFile::open(dir.join("weights").join("last.folo")).unwrap();
    let ckpt_best = final_ckpt.checkpoint().unwrap().best;
    let done_best = done.best.expect("completed run has best");
    assert_eq!(done_best.epoch, (ckpt_best.epoch + 1) as u32, "API best epoch must match checkpoint");
    if let Some(prev) = stopped.best
        && m.epochs[0].val.map50_95 <= prev.map50_95
    {
        assert_eq!(done_best.epoch, prev.epoch, "tying epoch 2 must not overwrite imported best epoch");
    }
    let log: RunLogs = ok(&app_b, get(&format!("/api/runs/{}/logs", cont.id))).await;
    assert!(log.lines.iter().any(|l| l.message.starts_with("continuing the training state of")), "{log:?}");
    assert!(done.model_id.is_some());
    assert!(std::fs::read(dir.join("weights").join("last.folo")).unwrap() != ckpt, "trained on");

    // Continuing a checkpoint that already completed its schedule is refused with 409 run_not_resumable
    // (whether the request asks for the same or more epochs), and leaves no bogus run behind.
    let completed_last = format!("m_{}-last", cont.id.trim_start_matches("run_"));
    for epochs in [2, 3] {
        let mut body = tiny(&ds_b.id, epochs);
        body.as_object_mut().unwrap().insert("checkpoint".into(), json!(completed_last));
        let e = err(&app_b, post_json("/api/runs", body), StatusCode::CONFLICT, "run_not_resumable").await;
        assert!(e.message.contains("already completed"), "{}", e.message);
        assert!(e.hint.as_deref().is_some_and(|h| h.contains("weights")), "{e:?}");
    }
    assert_eq!(ok::<RunList>(&app_b, get("/api/runs")).await.runs.len(), 1);
}

#[tokio::test]
async fn the_mock_has_cursors_but_no_checkpoints() {
    let app = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let all: RunMetrics = ok(&app, get("/api/runs/run_1a01/metrics")).await;
    let after_step = all.steps[all.steps.len() - 5].step;
    let some: RunMetrics =
        ok(&app, get(&format!("/api/runs/run_1a01/metrics?after_step={after_step}&after_epoch=2"))).await;
    assert_eq!(some.steps, all.steps[all.steps.len() - 4..]);
    assert_eq!(some.epochs, all.epochs[2..]);
    let full: RunLogs = ok(&app, get("/api/runs/run_1a01/logs?limit=5000")).await;
    assert!(full.total > 2 && full.total == full.lines.len() as u64);
    let rest: RunLogs = ok(&app, get(&format!("/api/runs/run_1a01/logs?after={}", full.total - 1))).await;
    assert_eq!((rest.lines.as_slice(), rest.total), (&full.lines[full.lines.len() - 1..], full.total));

    let body = json!({ "dataset_id": "ds_5a1e", "checkpoint": "m_31aa" });
    err(&app, post_json("/api/runs", body), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
}
