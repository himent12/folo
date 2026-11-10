//! The hardware: `device: auto` on a GPU, `amp` without tensor cores, and the throughput table.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{Events, app, dataset, err, get, ok, post, post_json, store, tiny};

/// `device: auto` on a machine with a GPU: the run trains there and predictions run there. Skips (with a note)
/// without a hardware adapter, like folo-vision's GPU tests.
#[tokio::test(flavor = "multi_thread")]
async fn auto_trains_and_predicts_on_the_gpu() {
    match folo_gpu::GpuContext::shared() {
        Ok(ctx) if ctx.summary().device_type != "Cpu" => {}
        Ok(ctx) => return eprintln!("note: skipping, software adapter {}", ctx.summary().name),
        Err(e) => return eprintln!("note: skipping, no GPU adapter: {e}"),
    }
    let t = TestDir::new("runs-gpu");
    let home = t.join("home");
    // Inference memory is released after 300 ms idle, the predictors after 600 ms (30 s and 5 min by default).
    let s = Arc::new(
        RealStore::open(home.clone())
            .recheck_every(Duration::ZERO)
            .gpu_idle_release(Duration::from_millis(300), Duration::from_millis(600)),
    );
    let app = app(&s);
    let ds = dataset(&app, "gpu", 6).await;
    // A run stopped after its first epoch (so it has a model): held there until the stop.
    s.hold_runs_after_epochs(true);
    let mut cfg = tiny(&ds, 3);
    cfg["device"] = json!("auto");
    let run: Run = ok(&app, post_json("/api/runs", cfg)).await;
    let gpu = folo_gpu::GpuContext::shared().unwrap().summary().name.clone();
    assert!(run.device.starts_with(&gpu), "{} on {gpu}", run.device);
    let mut sse = Events::open(&app, &run.id).await;
    sse.until_epoch().await;
    // Queued behind it while it trains: a fine-tune whose weights file is gone by the time it starts, so its
    // trainer fails after placing the model on the GPU. Its buffers are released too.
    let zoo = home.join("models");
    std::fs::create_dir_all(&zoo).unwrap();
    let mut seed = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(4),
    )
    .unwrap();
    seed.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    seed.save(zoo.join("seed.folo")).unwrap();
    let models = ok::<ModelList>(&app, get("/api/models")).await.models;
    let seed_id = models.iter().find(|m| m.source == Some(ModelSource::File)).unwrap().id.clone();
    let mut doomed = tiny(&ds, 1);
    doomed["device"] = json!("auto");
    doomed["weights"] = json!(seed_id);
    let doomed: Run = ok(&app, post_json("/api/runs", doomed)).await;
    assert_eq!(doomed.status, RunStatus::Queued);
    std::fs::remove_file(zoo.join("seed.folo")).unwrap();
    ok::<Run>(&app, post(&format!("/api/runs/{}/stop", run.id))).await;
    let events = sse.until_status(&["completed", "failed", "stopped"]).await;
    assert_eq!(events.last().unwrap().1["status"], "stopped", "{events:?}");
    let mut sse = Events::open(&app, &doomed.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "failed", "{events:?}");
    // Both runs gave their pooled buffers back before they were reported finished (the other tests of this binary
    // train on the CPU, so nothing else fills the pool meanwhile).
    let system: SystemResponse = ok(&app, get("/api/system")).await;
    let pool = system.gpu_pool.expect("the run used the GPU");
    eprintln!("after the run: {pool:?}");
    assert_eq!((pool.device.as_str(), pool.cached_bytes), ("gpu:0", 0), "{pool:?}");
    assert!(pool.peak_live_bytes > pool.live_bytes, "{pool:?}");
    let model_id = ok::<Run>(&app, get(&format!("/api/runs/{}", run.id))).await.model_id.unwrap();
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r.device, "gpu:0");
    assert!(!r.boxes.is_empty());
    let busy = ok::<SystemResponse>(&app, get("/api/system")).await.gpu_pool.unwrap();
    assert!(busy.cached_bytes > 0 && busy.live_bytes > 0, "a prediction caches its buffers: {busy:?}");
    // Idle: the cache goes back to the driver, then the predictor (its weights) too.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let idle = loop {
        let now = ok::<SystemResponse>(&app, get("/api/system")).await.gpu_pool.unwrap();
        if now.cached_bytes == 0 && now.live_bytes < busy.live_bytes {
            break now;
        }
        assert!(std::time::Instant::now() < deadline, "never released: {now:?} (busy {busy:?})");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    eprintln!("idle after a prediction: {idle:?} (busy {busy:?})");
    let system: SystemInfo = ok(&app, get("/api/system")).await;
    assert_eq!(system.gpus.first().map(|g| g.name.as_str()), Some(gpu.as_str()));
    // A one-second probe there measures the peak GPU memory too.
    let probe = json!({ "imgsz": 64, "batch": 4, "budget_s": 1.0 });
    let b: BenchmarkReport = ok(&app, post_json("/api/system/benchmark", probe)).await;
    assert_eq!((b.device.as_str(), b.device_name.as_str()), ("gpu:0", gpu.as_str()));
    assert!(b.training.batch == 4 && b.training.peak_vram_bytes.is_some_and(|v| v > 0), "{b:?}");
    assert!(b.train_imgs_per_s > 0.0 && b.latency_ms > 0.0, "{b:?}");
}

/// An `amp` run this server cannot train (here: on the CPU) is refused when it is submitted, with `501
/// not_supported` and the same reason `/api/system`'s `capabilities.amp` gives, before anything is queued or written;
/// resuming a mixed-precision run is refused the same way.
#[tokio::test(flavor = "multi_thread")]
async fn amp_run_without_tensor_cores_is_refused_up_front() {
    let t = TestDir::new("runs-amp");
    let home = t.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "amp", 4).await;
    let mut cfg = tiny(&ds, 1);
    cfg["amp"] = json!(true);
    let e =
        err(&app, post_json("/api/runs", cfg.clone()), StatusCode::NOT_IMPLEMENTED, "not_supported").await;
    assert!(e.message.contains("tensor cores") && e.message.contains("on the CPU"), "{}", e.message);
    let hint = e.hint.unwrap();
    assert!(hint.contains("`amp: false`") && hint.contains("--tensor-cores"), "{hint}");
    assert!(s.list_runs().unwrap().is_empty(), "nothing was queued");
    assert!(!home.join("runs").exists() || std::fs::read_dir(home.join("runs")).unwrap().next().is_none());
    // The estimate is not refused (it answers what such a run would take where it can run).
    let _: RunEstimate = ok(&app, post_json("/api/runs/estimate", cfg)).await;

    // Mixed-precision runs from a server that had tensor cores (their `amp` flag set in run.json): a stopped one is
    // refused on resume with resume advice (`amp` must match), but only once it would otherwise resume; a completed
    // one stays `run_not_resumable`.
    // `done` is held after its epoch, so `stopped` waits in the queue behind it and stops there.
    s.hold_runs_after_epochs(true);
    let done: Run = ok(&app, post_json("/api/runs", tiny(&ds, 1))).await;
    let stopped: Run = ok(&app, post_json("/api/runs", tiny(&ds, 1))).await;
    let stopped_now: Run = ok(&app, post(&format!("/api/runs/{}/stop", stopped.id))).await;
    assert_eq!(stopped_now.status, RunStatus::Stopped, "stopped while queued");
    s.hold_runs_after_epochs(false);
    let mut sse = Events::open(&app, &done.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    drop(sse);
    drop(app);
    tokio::task::spawn_blocking(move || s.shutdown()).await.unwrap();
    for id in [&stopped.id, &done.id] {
        let p = home.join("runs").join(id).join("run.json");
        let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        v["run"]["config"]["amp"] = json!(true);
        std::fs::write(&p, v.to_string()).unwrap();
    }
    let s = store(&home);
    let app = self::app(&s);
    let resume = |id: &str| post_json("/api/runs", json!({ "resume_run_id": id }));
    let e = err(&app, resume(&stopped.id), StatusCode::NOT_IMPLEMENTED, "not_supported").await;
    let hint = e.hint.unwrap();
    assert!(hint.contains("resumes in it") && !hint.contains("`amp: false`"), "{hint}");
    assert_eq!(ok::<Run>(&app, get(&format!("/api/runs/{}", stopped.id))).await.status, RunStatus::Stopped);
    err(&app, resume(&done.id), StatusCode::CONFLICT, "run_not_resumable").await;
    // A field that differs is still `invalid_config`, checked before amp.
    let bad = json!({ "resume_run_id": stopped.id, "epochs": 7 });
    err(&app, post_json("/api/runs", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
}

/// The throughput table from real activity: a 3-epoch run's epochs 2 (mosaic) and 3 (close mosaic) are measured
/// (the session's first is warm-up), a model's second prediction is (its first loads it), the table and the
/// estimates say so, and everything survives a restart. A probe is refused while a run is queued and then
/// measures on the CPU in about a second.
#[tokio::test(flavor = "multi_thread")]
async fn throughput_from_runs_predictions_and_a_probe() {
    let t = TestDir::new("runs-throughput");
    let home = t.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "speedy", 6).await;

    // Before any run: the reference, labelled.
    let e: RunEstimate = ok(&app, post_json("/api/runs/estimate", tiny(&ds, 3))).await;
    assert_eq!(
        (e.source, e.confidence, e.device.as_str()),
        (ThroughputSource::Reference, Confidence::Low, "cpu")
    );
    assert_eq!((e.train_images, e.val_images, e.batches_per_epoch), (5, 1, 2));
    assert_eq!(
        e.phases.iter().map(|p| (p.phase, p.epochs)).collect::<Vec<_>>(),
        [(PhaseKind::Mosaic, 2), (PhaseKind::CloseMosaic, 1)]
    );
    let bad = json!({ "dataset_id": ds.id, "imgsz": 48, "device": "cpu" });
    err(&app, post_json("/api/runs/estimate", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "dataset_id": "ds_none" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;

    let run: Run = ok(&app, post_json("/api/runs", tiny(&ds, 3))).await;
    let probe = json!({ "imgsz": 64, "batch": 4, "budget_s": 1.0 });
    err(&app, post_json("/api/system/benchmark", probe.clone()), StatusCode::CONFLICT, "run_active").await;
    let mut sse = Events::open(&app, &run.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let done: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    let model = done.model_id.unwrap();
    for _ in 0..2 {
        let req = json!({ "model_id": model, "dataset_id": ds.id, "image_id": "0" });
        ok::<PredictResponse>(&app, post_json("/api/predict", req)).await;
    }

    let table: ThroughputTable = ok(&app, get("/api/system/throughput")).await;
    let row = table.training.iter().find(|r| r.source == ThroughputSource::Measured).expect("a measured row");
    assert_eq!((row.device.as_str(), row.arch, row.imgsz, row.amp), ("cpu", Arch::N, 64, false));
    assert_eq!(
        (row.mosaic.unwrap().samples, row.close_mosaic.unwrap().samples, row.samples),
        (1, 1, 2),
        "{row:?}"
    );
    assert!(row.updated.is_some() && row.note.is_none() && row.peak_vram_bytes.is_none());
    let infer =
        table.inference.iter().find(|r| r.source == ThroughputSource::Measured).expect("a predict row");
    assert_eq!(
        (infer.arch, infer.imgsz, infer.samples),
        (Arch::N, 64, 1),
        "the first prediction loads the model"
    );

    let e: RunEstimate = ok(&app, post_json("/api/runs/estimate", tiny(&ds, 3))).await;
    assert_eq!((e.source, e.confidence), (ThroughputSource::Measured, Confidence::Medium), "{e:?}");
    assert!(e.basis.exact && e.basis.samples == 2);
    let other = json!({ "dataset_id": ds.id, "imgsz": 96, "device": "cpu", "epochs": 3 });
    let e: RunEstimate = ok(&app, post_json("/api/runs/estimate", other)).await;
    assert_eq!(e.source, ThroughputSource::Measured);
    assert!(!e.basis.exact && e.basis.imgsz == 64, "{:?}", e.basis);

    // A probe on the idle server: a measured row of its own, marked as a probe.
    let r: BenchmarkReport = ok(&app, post_json("/api/system/benchmark", probe)).await;
    assert_eq!((r.device.as_str(), r.training.imgsz, r.training.probes), ("cpu", 64, 1));
    assert_eq!(r.training.samples, 3, "the run's 2 epochs and the probe");
    assert!(
        r.train_imgs_per_s > 0.0 && r.train_batches >= 1 && r.predictions >= 5 && r.latency_ms > 0.0,
        "{r:?}"
    );
    assert!(r.elapsed_s < 60.0, "{r:?}");

    // Persisted: a new store on the same data dir reads it back.
    s.shutdown();
    assert!(home.join("throughput.json").is_file());
    let app2 = super::harness::app(&store(&home));
    let back: ThroughputTable = ok(&app2, get("/api/system/throughput")).await;
    let measured = |t: &ThroughputTable| {
        (
            t.training.iter().filter(|r| r.source == ThroughputSource::Measured).cloned().collect::<Vec<_>>(),
            t.inference
                .iter()
                .filter(|r| r.source == ThroughputSource::Measured)
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    let now: ThroughputTable = ok(&app, get("/api/system/throughput")).await;
    assert_eq!(measured(&back), measured(&now));
}
