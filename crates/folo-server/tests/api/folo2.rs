//! End-to-end `folo2` (`folo2-n`..`folo2-x`) HTTP API coverage on both `MockStore` and `RealStore`:
//! capability advertisement, run creation & estimation, CLI command emission, `x-folo-arch` header on
//! `GET /api/models/{id}`, `val`, `predict` (`tile`, `tta`), export (`folo-f16`, `onnx`), weight downloads,
//! fine-tune arch validation, and cloud refusal.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_server::api::*;
use folo_server::store::{FEATURE_FOLO2_ARCH, TileConfig, TtaConfig};
use folo_server::{Options, RealStore, Server};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{err, get, mock, ok, post_json, send};

#[tokio::test]
async fn folo2_capability_estimate_cli_and_cloud_refusal_on_mock() {
    let (server, _store) = mock();
    let app = server.router();

    let health: Health = ok(&app, get("/api/health")).await;
    assert!(health.features.iter().any(|f| f == FEATURE_FOLO2_ARCH), "{:?}", health.features);
    let sys: SystemResponse = ok(&app, get("/api/system")).await;
    assert!(sys.features.iter().any(|f| f == FEATURE_FOLO2_ARCH), "{:?}", sys.features);

    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let ds_id = ds.datasets[0].id.clone();

    let est: Value = ok(
        &app,
        post_json(
            "/api/runs/estimate",
            json!({ "dataset_id": ds_id, "arch": "folo2-s", "epochs": 2, "batch": 4, "imgsz": 64 }),
        ),
    )
    .await;
    assert!(est.is_object(), "{est}");

    // Cloud runs on `folo2-*` are refused upfront so cloud workers never silently train `folo-n`.
    err(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds_id,
                "arch": "folo2-n",
                "epochs": 1,
                "compute": { "connection_id": "cc_4d2a", "offer_id": "gpu_rtx4090" }
            }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;

    let run: Run = ok(
        &app,
        post_json(
            "/api/runs",
            json!({ "dataset_id": ds_id, "name": "mock-v2", "arch": "folo2-n", "epochs": 1, "batch": 4 }),
        ),
    )
    .await;
    let cli: Value = ok(&app, get(&format!("/api/runs/{}/cli", run.id))).await;
    assert!(cli["cli"].as_str().unwrap().contains("--model folo2-n"), "{cli}");

    // Advance mock time until the `folo2-n` run completes and produces a model, then verify `x-folo-arch`,
    // fine-tune arch preservation, and cloud refusal on both estimate and create.
    let (_, store) = (server, _store);
    store.pass_time(Duration::from_secs(60));
    let done: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(done.status, RunStatus::Completed, "{done:?}");
    let model_id = done.model_id.expect("mock folo2 model_id");
    let (st, hdrs, _) = send(&app, get(&format!("/api/models/{model_id}"))).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(hdrs.get("x-folo-arch").and_then(|v| v.to_str().ok()), Some("folo2-n"));

    // Fine-tuning from a mock `folo2-n` model preserves `folo2-n` and refuses cloud on both estimate and create.
    for path in ["/api/runs/estimate", "/api/runs"] {
        err(
            &app,
            post_json(
                path,
                json!({
                    "dataset_id": ds_id,
                    "weights": model_id,
                    "epochs": 1,
                    "compute": { "connection_id": "cc_4d2a", "offer_id": "gpu_rtx4090" }
                }),
            ),
            StatusCode::BAD_REQUEST,
            "invalid_config",
        )
        .await;
    }
    let ft: Run = ok(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds_id, "weights": model_id, "epochs": 1, "batch": 4 })),
    )
    .await;
    assert!(ft.name.contains("folo2-n"), "mock fine-tune default name preserves folo2-n: {}", ft.name);
}

#[tokio::test]
async fn folo2_real_store_train_val_predict_and_export_end_to_end() {
    let dir = TestDir::new("api-folo2-real");
    let store = Arc::new(RealStore::open(dir.to_path_buf()));
    let app = Server::new(store.clone(), Options::default()).router();

    let ds: Dataset = ok(
        &app,
        post_json(
            "/api/datasets/synthetic",
            json!({ "name": "v2-shapes", "images": 8, "imgsz": 64, "classes": ["circle", "square"], "seed": 14 }),
        ),
    )
    .await;

    // Start a 1-epoch CPU `folo2-n` run.
    let created: Run = ok(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds.id,
                "name": "real-v2",
                "arch": "folo2-n",
                "epochs": 1,
                "batch": 4,
                "imgsz": 64,
                "device": "cpu",
                "augment": { "close_mosaic": 1 }
            }),
        ),
    )
    .await;

    let cli: Value = ok(&app, get(&format!("/api/runs/{}/cli", created.id))).await;
    assert!(cli["cli"].as_str().unwrap().contains("--model folo2-n"), "{cli}");

    // Wait for completion.
    let deadline = Instant::now() + Duration::from_secs(120);
    let done: Run = loop {
        let r: Run = ok(&app, get(&format!("/api/runs/{}", created.id))).await;
        if !matches!(r.status, RunStatus::Queued | RunStatus::Running) {
            break r;
        }
        assert!(Instant::now() < deadline, "folo2 run timed out: {r:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert_eq!(done.status, RunStatus::Completed, "{done:?}");
    let model_id = done.model_id.expect("model_id on completed run");

    // `GET /api/models/{id}` exposes the exact `folo2-n` architecture in `x-folo-arch`.
    let (status, headers, _) = send(&app, get(&format!("/api/models/{model_id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers.get("x-folo-arch").and_then(|v| v.to_str().ok()), Some("folo2-n"));

    // Cloud runs on `folo2-n` are refused upfront with `invalid_config`.
    let e = err(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds.id,
                "arch": "folo2-n",
                "epochs": 1,
                "batch": 4,
                "imgsz": 64,
                "compute": { "connection_id": "cc_missing", "offer_id": "off_missing" }
            }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("folo2"), "{e:?}");

    // `POST /api/val` with diagnostics works on the `folo2-n` model.
    let val: Value = ok(
        &app,
        post_json(
            "/api/val",
            json!({ "model_id": model_id, "dataset_id": ds.id, "diagnostics": true, "worst": 2 }),
        ),
    )
    .await;
    assert_eq!(val["imgsz"], 64);
    assert!(val["diagnostics"]["images_diagnosed"].as_u64().unwrap() >= 1, "{val}");

    // `POST /api/predict` works in plain, SAHI (`tile`), and `tta` modes.
    let imgs: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?split=val&limit=1", ds.id))).await;
    let img_id = &imgs.images[0].id;
    let p_plain: PredictResponse = ok(
        &app,
        post_json(
            "/api/predict",
            json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": img_id, "conf": 0.01 }),
        ),
    )
    .await;
    assert_eq!((p_plain.image.width, p_plain.image.height), (64, 64));

    let tile_cfg = TileConfig {
        tile_width: 32,
        tile_height: 32,
        overlap_x: 0.2,
        overlap_y: 0.2,
        ..TileConfig::default()
    };
    let p_tile: PredictResponse = ok(
        &app,
        post_json(
            "/api/predict",
            json!({
                "model_id": model_id,
                "dataset_id": ds.id,
                "image_id": img_id,
                "conf": 0.01,
                "tile": tile_cfg
            }),
        ),
    )
    .await;
    assert_eq!((p_tile.image.width, p_tile.image.height), (64, 64));

    let tta_cfg = TtaConfig { scales: vec![1.0], ..TtaConfig::default() };
    let p_tta: PredictResponse = ok(
        &app,
        post_json(
            "/api/predict",
            json!({
                "model_id": model_id,
                "dataset_id": ds.id,
                "image_id": img_id,
                "conf": 0.01,
                "tta": tta_cfg
            }),
        ),
    )
    .await;
    assert_eq!((p_tta.image.width, p_tta.image.height), (64, 64));

    // `POST /api/models/{id}/export` works for both `folo-f16` and `onnx` (`dynamic_batch`) and preserves real
    // class names in the ONNX `names` metadata.
    let ex_f16: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "folo-f16" }))).await;
    assert!(std::path::Path::new(&ex_f16.path).is_file() && ex_f16.bytes > 0);

    let ex_onnx: ExportResult = ok(
        &app,
        post_json(
            &format!("/api/models/{model_id}/export"),
            json!({ "format": "onnx", "dynamic_batch": true }),
        ),
    )
    .await;
    assert!(ex_onnx.path.ends_with(".onnx") && ex_onnx.bytes > 0);
    let onnx_file_lossy = String::from_utf8_lossy(&std::fs::read(&ex_onnx.path).unwrap()).into_owned();
    assert!(
        onnx_file_lossy.contains("\"0\":\"circle\"") && onnx_file_lossy.contains("\"1\":\"square\""),
        "server ONNX export must preserve class names"
    );

    // `GET /api/models/{id}/weights?format=f16` and `format=onnx` stream non-empty fused exports (with `names`).
    for fmt in ["f16", "onnx"] {
        let (st, _, bytes) = send(&app, get(&format!("/api/models/{model_id}/weights?format={fmt}"))).await;
        assert_eq!(st, StatusCode::OK, "weights?format={fmt}");
        assert!(!bytes.is_empty(), "weights?format={fmt} empty");
        if fmt == "onnx" {
            let lossy = String::from_utf8_lossy(&bytes);
            assert!(
                lossy.contains("\"0\":\"circle\"") && lossy.contains("\"1\":\"square\""),
                "weights?format=onnx must preserve class names"
            );
        }
    }

    // `GET /api/models/{id}/eval` evaluates the `folo2-n` model cleanly.
    let eval: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert!(eval.images >= 1, "{eval:?}");

    // `folo2-n` training and prediction samples must not contaminate the v1 throughput table.
    let tp: ThroughputTable = ok(&app, get("/api/system/throughput")).await;
    assert!(
        tp.training.iter().all(|r| r.source != ThroughputSource::Measured),
        "folo2 train samples must not pollute v1 measured throughput: {:?}",
        tp.training
    );
    assert!(
        tp.inference.iter().all(|r| r.source != ThroughputSource::Measured),
        "folo2 infer samples must not pollute v1 measured throughput: {:?}",
        tp.inference
    );

    // Fine-tuning from a `folo2-n` model refuses cloud compute upfront.
    err(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds.id,
                "weights": model_id,
                "epochs": 1,
                "batch": 4,
                "imgsz": 64,
                "compute": { "connection_id": "cc_missing", "offer_id": "off_missing" }
            }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;

    // A `folo2-n` weights file whose informational `folo.arch` was tampered to `"folo-n"` still resolves from its
    // validated `folo.config` as `folo2-n` (both in `x-folo-arch` and cloud refusal).
    let tampered_path = dir.join("models/tampered-v2.folo");
    std::fs::create_dir_all(tampered_path.parent().unwrap()).unwrap();
    let src_bytes = std::fs::read(&ex_f16.path).unwrap();
    let hlen = u64::from_le_bytes(src_bytes[..8].try_into().unwrap()) as usize;
    let mut hdr: Value = serde_json::from_slice(&src_bytes[8..8 + hlen]).unwrap();
    hdr["__metadata__"]["folo.arch"] = json!("folo-n");
    let mut h_bytes = serde_json::to_vec(&hdr).unwrap();
    while !h_bytes.len().is_multiple_of(8) {
        h_bytes.push(b' ');
    }
    let mut out_bytes = (h_bytes.len() as u64).to_le_bytes().to_vec();
    out_bytes.extend_from_slice(&h_bytes);
    out_bytes.extend_from_slice(&src_bytes[8 + hlen..]);
    std::fs::write(&tampered_path, out_bytes).unwrap();

    let models_list: ModelList = ok(&app, get("/api/models")).await;
    let tampered =
        models_list.models.iter().find(|m| m.name == "tampered-v2").expect("tampered-v2 model listed");
    let (st, hdrs, _) = send(&app, get(&format!("/api/models/{}", tampered.id))).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(hdrs.get("x-folo-arch").and_then(|v| v.to_str().ok()), Some("folo2-n"));
    err(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds.id,
                "weights": tampered.id,
                "epochs": 1,
                "batch": 4,
                "imgsz": 64,
                "compute": { "connection_id": "cc_missing", "offer_id": "off_missing" }
            }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;

    // HTTP `resume_run_id` on a stopped `folo2-n` run: refuses mismatched `arch`, refuses cloud resume, and
    // accepts matching `arch: "folo2-n"` while keeping the override persisted.
    let long_run: Run = ok(
        &app,
        post_json(
            "/api/runs",
            json!({
                "dataset_id": ds.id,
                "name": "real-v2-resume",
                "arch": "folo2-n",
                "epochs": 200,
                "batch": 4,
                "imgsz": 64,
                "device": "cpu",
                "augment": { "close_mosaic": 1 }
            }),
        ),
    )
    .await;
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let r: Run = ok(&app, get(&format!("/api/runs/{}", long_run.id))).await;
        if r.epoch >= 1 {
            let _: Run = ok(
                &app,
                axum::http::Request::post(format!("/api/runs/{}/stop", long_run.id))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await;
            break;
        }
        assert!(Instant::now() < deadline, "long folo2 run did not finish epoch 1: {r:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let stopped_epoch = loop {
        let r: Run = ok(&app, get(&format!("/api/runs/{}", long_run.id))).await;
        if r.status == RunStatus::Stopped {
            break r.epoch;
        }
        assert!(Instant::now() < deadline, "long folo2 run did not stop: {r:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    err(
        &app,
        post_json("/api/runs", json!({ "resume_run_id": long_run.id, "arch": "folo-n" })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    err(
        &app,
        post_json(
            "/api/runs",
            json!({
                "resume_run_id": long_run.id,
                "compute": { "connection_id": "cc_missing", "offer_id": "off_missing" }
            }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    let resumed: Run =
        ok(&app, post_json("/api/runs", json!({ "resume_run_id": long_run.id, "arch": "folo2-n" }))).await;
    assert_eq!(resumed.id, long_run.id);
    loop {
        let r: Run = ok(&app, get(&format!("/api/runs/{}", long_run.id))).await;
        if r.epoch > stopped_epoch {
            let _: Run = ok(
                &app,
                axum::http::Request::post(format!("/api/runs/{}/stop", long_run.id))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await;
            break;
        }
        assert!(Instant::now() < deadline, "resumed folo2 run did not advance: {r:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    loop {
        let r: Run = ok(&app, get(&format!("/api/runs/{}", long_run.id))).await;
        if !matches!(r.status, RunStatus::Queued | RunStatus::Running) {
            break;
        }
        assert!(Instant::now() < deadline, "resumed folo2 run did not stop cleanly: {r:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
