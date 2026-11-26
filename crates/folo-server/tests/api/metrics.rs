//! Mock metrics and their operating point, throughput, run estimates and the benchmark probe.

use std::sync::Arc;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_server::{MockStore, Options, Server};
use serde_json::{Value, json};

use super::harness::{err, get, mock, ok, post_json};

/// Every precision / recall in mock mode says which confidence it is taken at (`best_f1_conf`).
#[tokio::test]
async fn mock_metrics_carry_the_operating_point() {
    let (server, _) = mock();
    let app = server.router();
    let conf_ok = |c: Option<f32>| {
        c.is_some_and(|c| (0.001..1.0).contains(&c) && (c * 1000.0 - (c * 1000.0).round()).abs() < 1e-3)
    };
    let models: ModelList = ok(&app, get("/api/models")).await;
    for m in &models.models {
        let metrics = m.metrics.unwrap();
        assert!(conf_ok(metrics.best_f1_conf), "{}: {metrics:?}", m.id);
        let ev: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        assert_eq!(ev.best_f1_conf, metrics.best_f1_conf, "{}: /eval and the model agree", m.id);
        // What was evaluated: the run's dataset (every mock dataset has a validation split) and the confusion
        // matrix's thresholds, the real evaluator's.
        let run: Run = ok(&app, get(&format!("/api/runs/{}", m.run_id.as_deref().unwrap()))).await;
        let ds: Dataset = ok(&app, get(&format!("/api/datasets/{}", run.config.dataset_id))).await;
        assert_eq!((ev.dataset_id.as_deref(), ev.dataset_name.as_deref()), (Some(&*ds.id), Some(&*ds.name)));
        assert_eq!((ev.split, ev.split_fallback.as_ref()), (Some(Split::Val), None), "{}", m.id);
        assert_eq!((ev.confusion_conf, ev.confusion_iou), (Some(0.25), Some(0.45)), "{}", m.id);
    }
    // Every epoch of every run, finished or live, carries its operating point (not only the best one).
    let runs: RunList = ok(&app, get("/api/runs")).await;
    for r in &runs.runs {
        let hist: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", r.id))).await;
        assert!(!hist.epochs.is_empty() || r.epoch == 0, "{}", r.id);
        for e in &hist.epochs {
            assert!(conf_ok(e.val.best_f1_conf), "{} epoch {}: {:?}", r.id, e.epoch, e.val);
        }
    }
    let hist: RunMetrics = ok(&app, get("/api/runs/run_1a01/metrics")).await;
    let v = serde_json::to_value(&hist.epochs[0]).unwrap();
    assert!(v["val"]["best_f1_conf"].is_number(), "{v}");
    // POST /api/val: never below the request's NMS threshold; names the dataset.
    let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e", "conf": 0.6 });
    let r: ValReport = ok(&app, post_json("/api/val", req)).await;
    assert!(r.val.best_f1_conf.unwrap() >= 0.6, "{:?}", r.val.best_f1_conf);
    assert_eq!(r.dataset_name.as_deref(), Some("shapes-1k"));
}

/// A run model's stored metrics are the metrics of the epoch its weights were saved from (the run's best epoch),
/// operating point included, and that epoch's overall precision / recall / mAP are its class means (as the engine
/// reports them), so the run page and the model page show the same numbers. `/eval` re-evaluates the model: its
/// per-class numbers stay within rounding of the epoch's (exactly equal operating point).
#[tokio::test]
async fn mock_model_metrics_match_their_run_epoch() {
    let (server, _) = mock();
    let app = server.router();
    let models: ModelList = ok(&app, get("/api/models")).await;
    let mut bad = Vec::new();
    let mut checked = 0;
    for m in models.models.iter().filter(|m| m.run_id.is_some()) {
        let run_id = m.run_id.as_deref().unwrap();
        let run: Run = ok(&app, get(&format!("/api/runs/{run_id}"))).await;
        let hist: RunMetrics = ok(&app, get(&format!("/api/runs/{run_id}/metrics"))).await;
        let best = run.best.unwrap().epoch;
        let v = &hist.epochs.iter().find(|e| e.epoch == best).unwrap().val;
        let from_run = ModelMetrics {
            map50: v.map50,
            map50_95: v.map50_95,
            precision: v.precision,
            recall: v.recall,
            best_f1_conf: v.best_f1_conf,
        };
        if m.metrics != Some(from_run) {
            bad.push(format!("{}: stored {:?} vs {run_id} epoch {best} {from_run:?}", m.id, m.metrics));
        }
        let mean = |pc: &[ClassMetrics], f: fn(&ClassMetrics) -> f32| {
            pc.iter().map(f).sum::<f32>() / pc.len() as f32
        };
        type Field = (&'static str, fn(&ClassMetrics) -> f32, f32);
        let fields: [Field; 4] = [
            ("precision", |c| c.precision, v.precision),
            ("recall", |c| c.recall, v.recall),
            ("map50", |c| c.map50, v.map50),
            ("map50_95", |c| c.map50_95, v.map50_95),
        ];
        for (name, f, overall) in fields {
            if (mean(&v.per_class, f) - overall).abs() > 1e-3 {
                bad.push(format!(
                    "{run_id} epoch {best}: {name} {overall} vs class mean {}",
                    mean(&v.per_class, f)
                ));
            }
        }
        let ev: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        if ev.best_f1_conf != from_run.best_f1_conf {
            bad.push(format!("{}: /eval conf {:?} vs {:?}", m.id, ev.best_f1_conf, from_run.best_f1_conf));
        }
        for (e, r) in ev.per_class.iter().zip(&v.per_class) {
            let close = |a: f32, b: f32| (a - b).abs() <= 0.01;
            if !(close(e.precision, r.precision) && close(e.recall, r.recall) && close(e.map50, r.map50)) {
                bad.push(format!("{} class {}: /eval {e:?} vs epoch {r:?}", m.id, e.class));
            }
        }
        checked += 1;
    }
    assert!(
        bad.is_empty(),
        "{}",
        bad.join(
            "
"
        )
    );
    assert!(checked >= 5, "{checked}");
}

/// `GET /api/system/throughput` in mock mode: the measured rows of the mock's runs first, labelled, then a
/// reference row (with its docs/BENCHMARKS.md note) for every size and precision nothing measured.
#[tokio::test]
async fn mock_throughput_table_labels_both_sources() {
    let (server, _) = mock();
    let app = server.router();
    let s: SystemResponse = ok(&app, get("/api/system")).await;
    for f in [FEATURE_THROUGHPUT, FEATURE_RUN_ESTIMATE, FEATURE_BENCHMARK] {
        assert!(s.features.iter().any(|x| x == f), "{f}: {:?}", s.features);
    }
    let raw: Value = ok(&app, get("/api/system/throughput")).await;
    let t: ThroughputTable = serde_json::from_value(raw.clone()).unwrap();
    assert_eq!((t.device.as_str(), t.device_name.as_str()), ("gpu:0", "NVIDIA GeForce RTX 4080 SUPER"));
    assert_eq!(raw["training"][0]["source"], "measured", "{}", raw["training"][0]);
    let n640 =
        t.training.iter().find(|r| r.device == "gpu:0" && r.arch == Arch::N && r.imgsz == 640 && !r.amp);
    let n640 = n640.expect("a folo-n 640 row");
    assert_eq!((n640.source, n640.samples, n640.note.as_ref()), (ThroughputSource::Measured, 99, None));
    let mosaic = n640.mosaic.unwrap();
    assert!(
        mosaic.low <= mosaic.imgs_per_s && mosaic.imgs_per_s <= mosaic.high && mosaic.samples == 81,
        "{mosaic:?}"
    );
    assert!(n640.close_mosaic.is_some() && n640.val.is_some() && n640.updated.is_some());
    assert!(n640.peak_vram_bytes.is_some_and(|b| b > 3_000_000_000) && n640.batch == 16);
    assert_eq!(
        (n640.imgs_per_s, n640.measured_on.as_str()),
        (mosaic.imgs_per_s, "NVIDIA GeForce RTX 4080 SUPER")
    );
    // No row twice; references carry no samples and say where they come from.
    let mut keys = std::collections::HashSet::new();
    for r in &t.training {
        assert!(keys.insert((r.device.clone(), r.arch, r.imgsz, r.amp)), "{r:?} twice");
        if r.source == ThroughputSource::Reference {
            assert_eq!((r.samples, r.updated.as_ref()), (0, None));
            assert!(r.note.as_ref().is_some_and(|n| n.contains("docs/BENCHMARKS.md")), "{r:?}");
        }
    }
    let x = t.training.iter().find(|r| r.arch == Arch::X && r.imgsz == 640 && r.device == "gpu:0" && !r.amp);
    assert_eq!(x.unwrap().source, ThroughputSource::Reference);
    assert!(
        t.training.iter().any(|r| r.device == "cpu" && !r.amp)
            && !t.training.iter().any(|r| r.device == "cpu" && r.amp)
    );
    let infer =
        t.inference.iter().find(|r| r.source == ThroughputSource::Measured).expect("a measured predict row");
    assert_eq!(
        (infer.arch, infer.imgsz, infer.precision, infer.samples),
        (Arch::N, 640, InferPrecision::F32, 214)
    );
    assert!((infer.imgs_per_s - 1000.0 / infer.latency_ms).abs() < 0.1);
    assert!(
        t.inference
            .iter()
            .any(|r| r.precision == InferPrecision::F16 && r.source == ThroughputSource::Reference)
    );
    // Deterministic.
    let again: Value = ok(&app, get("/api/system/throughput")).await;
    assert_eq!(again, raw);
}

/// `POST /api/runs/estimate` in mock mode: measured where the mock measured, carried to nearby sizes, the reference
/// elsewhere, and the same checks as `POST /api/runs`.
#[tokio::test]
async fn mock_run_estimates() {
    let (server, _) = mock();
    let app = server.router();
    // shapes-1k: 800 train / 200 val images; folo-n 640 is measured.
    let e: RunEstimate =
        ok(&app, post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "epochs": 100 }))).await;
    assert_eq!(
        (e.source, e.confidence, e.val_source),
        (ThroughputSource::Measured, Confidence::High, ThroughputSource::Measured)
    );
    assert!(e.basis.exact && e.basis.samples == 99 && e.basis.updated.is_some(), "{:?}", e.basis);
    assert_eq!((e.train_images, e.val_images, e.batches_per_epoch, e.epochs), (800, 200, 50, 100));
    assert_eq!(
        e.phases.iter().map(|p| (p.phase, p.epochs)).collect::<Vec<_>>(),
        [(PhaseKind::Mosaic, 90), (PhaseKind::CloseMosaic, 10)]
    );
    let sum: f64 = e.phases.iter().map(|p| p.epochs as f64 * p.epoch_s).sum::<f64>() + e.startup_s;
    assert!((sum - e.est_total_s).abs() < 1.0, "{sum} vs {}", e.est_total_s);
    assert!(e.est_total_s_low < e.est_total_s && e.est_total_s < e.est_total_s_high);
    assert_eq!(
        (e.vram_source, e.est_peak_vram_bytes),
        (Some(ThroughputSource::Measured), Some(3_652_000_000))
    );
    // A size nothing measured: carried from the nearest measured row (folo-s 640).
    let e: RunEstimate =
        ok(&app, post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "arch": "folo-m" }))).await;
    assert_eq!((e.source, e.confidence), (ThroughputSource::Measured, Confidence::Medium));
    assert!(!e.basis.exact && e.basis.arch == Arch::S, "{:?}", e.basis);
    // The CPU: nothing measured, so the labelled reference.
    let body = json!({ "dataset_id": "ds_5a1e", "device": "cpu", "imgsz": 320, "epochs": 10 });
    let e: RunEstimate = ok(&app, post_json("/api/runs/estimate", body)).await;
    assert_eq!(
        (e.source, e.confidence, e.device.as_str()),
        (ThroughputSource::Reference, Confidence::Low, "cpu")
    );
    assert!(e.basis.note.contains("docs/BENCHMARKS.md"), "{}", e.basis.note);
    assert_eq!((e.est_peak_vram_bytes, e.vram_source), (None, None));
    assert_eq!(e.phases.iter().map(|p| p.epochs).collect::<Vec<_>>(), [5, 5], "close_mosaic capped at half");
    // The capped value, and the CLI command of the same config (defaults left out).
    assert_eq!(e.close_mosaic, 5);
    let cli = e.cli.unwrap();
    assert!(
        cli.starts_with("folo train --data ") && cli.ends_with("--epochs 10 --imgsz 320 --device cpu"),
        "{cli}"
    );
    // Validation as for POST /api/runs.
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "dataset_id": "ds_nope" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
    let bad = json!({ "dataset_id": "ds_5a1e", "epochs": 0 });
    err(&app, post_json("/api/runs/estimate", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "epochs": 3 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    // The body's `distill` key, and `amp` on a device without it, answer as on POST /api/runs.
    let distill = json!({ "dataset_id": "ds_5a1e", "distill": { "teacher": "folo-n-coco" } });
    err(&app, post_json("/api/runs/estimate", distill), StatusCode::BAD_REQUEST, "invalid_config").await;
    let amp = json!({ "dataset_id": "ds_5a1e", "device": "cpu", "amp": true });
    err(&app, post_json("/api/runs/estimate", amp), StatusCode::NOT_IMPLEMENTED, "not_supported").await;
    // Fine-tuning with weights of another architecture adopts the weights' arch for the estimate.
    let models: ModelList = ok(&app, get("/api/models")).await;
    let non_nano =
        models.models.iter().find(|m| m.arch != Arch::N).expect("mock catalog has non-nano models");
    let nano: RunEstimate =
        ok(&app, post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "arch": "folo-n" }))).await;
    let expected: RunEstimate =
        ok(&app, post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "arch": non_nano.arch })))
            .await;
    assert_ne!(nano.est_total_s, expected.est_total_s, "fixture architectures must have distinct estimates");
    let from_weights: RunEstimate = ok(
        &app,
        post_json(
            "/api/runs/estimate",
            json!({ "dataset_id": "ds_5a1e", "arch": "folo-n", "weights": non_nano.id }),
        ),
    )
    .await;
    assert_eq!(from_weights.est_total_s, expected.est_total_s);
    assert_eq!(from_weights.basis.arch, expected.basis.arch);
    let weights_cli = from_weights.cli.unwrap();
    assert!(
        weights_cli.contains("--weights ") && !weights_cli.contains("--arch "),
        "CLI command with --weights omits --arch: {weights_cli}"
    );
    // Mock rejects unavailable local GPU ordinals and invalid augmentation probabilities like the real server.
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "device": "gpu:1" })),
        StatusCode::NOT_FOUND,
        "device_not_found",
    )
    .await;
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "augment": { "mosaic": -1.0 } })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    // Misspelled config keys fail loudly with 422 invalid_body on both /api/runs and /api/runs/estimate.
    for path in ["/api/runs/estimate", "/api/runs"] {
        for typo in [
            json!({ "dataset_id": "ds_5a1e", "batch_size": 4 }),
            json!({ "dataset_id": "ds_5a1e", "augment": { "mosiac": 0.0 } }),
            json!({ "dataset_id": "ds_5a1e", "compute": { "connection_id": "cc_1", "offer_id": "o_1", "max_hour": 1.0 } }),
        ] {
            err(&app, post_json(path, typo), StatusCode::UNPROCESSABLE_ENTITY, "invalid_body").await;
        }
    }
    err(
        &app,
        post_json("/api/runs/estimate", json!({ "dataset_id": "ds_5a1e", "resume_run_id": "run_1" })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    // A name too long is the run's error only: New run keeps its estimate while the name is being typed.
    let long = json!({ "dataset_id": "ds_5a1e", "name": "n".repeat(65) });
    let _: RunEstimate = ok(&app, post_json("/api/runs/estimate", long.clone())).await;
    err(&app, post_json("/api/runs", long), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert_eq!(ok::<RunList>(&app, get("/api/runs")).await.runs.len(), 7, "an estimate creates no run");
    // A run's command: GET /api/runs/{id}/cli, the same mapping, also as argv.
    let run: Run =
        ok(&app, post_json("/api/runs", json!({ "dataset_id": "ds_5a1e", "epochs": 3, "name": "a b" })))
            .await;
    let c: CliCommand = ok(&app, get(&format!("/api/runs/{}/cli", run.id))).await;
    assert_eq!(&c.argv[..2], ["folo", "train"]);
    assert!(c.cli.ends_with("--name 'a b' --epochs 3"), "{}", c.cli);
    err(&app, get("/api/runs/run_nope/cli"), StatusCode::NOT_FOUND, "run_not_found").await;
    let s: SystemResponse = ok(&app, get("/api/system")).await;
    assert!(s.features.iter().any(|f| f == FEATURE_RUN_CLI), "{:?}", s.features);
}

/// `POST /api/system/benchmark` in mock mode: it updates the table (as a probe), checks its settings and is
/// refused while a run created through the API is active.
#[tokio::test]
async fn mock_benchmark_probe() {
    let (server, _) = mock();
    let app = server.router();
    let r: BenchmarkReport = ok(&app, post_json("/api/system/benchmark", json!({}))).await;
    assert_eq!((r.training.arch, r.training.imgsz, r.training.amp), (Arch::N, 320, false));
    assert_eq!(
        (r.training.source, r.training.samples, r.training.probes),
        (ThroughputSource::Measured, 1, 1)
    );
    assert!(r.train_imgs_per_s > 300.0 && r.latency_ms > 0.0 && r.predictions >= 5, "{r:?}");
    let t: ThroughputTable = ok(&app, get("/api/system/throughput")).await;
    let row =
        t.training.iter().find(|x| x.arch == Arch::N && x.imgsz == 320 && !x.amp && x.device == "gpu:0");
    assert_eq!(row.unwrap().source, ThroughputSource::Measured, "the probe's row replaces the reference");
    let bad = json!({ "budget_s": 0.5 });
    err(&app, post_json("/api/system/benchmark", bad), StatusCode::BAD_REQUEST, "invalid_body").await;
    err(
        &app,
        post_json("/api/system/benchmark", json!({ "imgsz": 100 })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    let none = Server::new(
        Arc::new(MockStore::new().with_tensor_cores(folo_server::mock::MockTensorCores::None)),
        Options::default(),
    )
    .router();
    err(
        &none,
        post_json("/api/system/benchmark", json!({ "amp": true })),
        StatusCode::NOT_IMPLEMENTED,
        "not_supported",
    )
    .await;
    let run: Run = ok(&app, post_json("/api/runs", json!({ "dataset_id": "ds_5a1e", "epochs": 2 }))).await;
    err(&app, post_json("/api/system/benchmark", json!({})), StatusCode::CONFLICT, "run_active").await;
    ok::<Run>(&app, post_json(&format!("/api/runs/{}/stop", run.id), json!({}))).await;
}
