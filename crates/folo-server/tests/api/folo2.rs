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
    let (server, _) = mock();
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

    // `POST /api/models/{id}/export` works for both `folo-f16` and `onnx` (`dynamic_batch`).
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

    // `GET /api/models/{id}/weights?format=f16` and `format=onnx` stream non-empty fused exports.
    for fmt in ["f16", "onnx"] {
        let (st, _, bytes) = send(&app, get(&format!("/api/models/{model_id}/weights?format={fmt}"))).await;
        assert_eq!(st, StatusCode::OK, "weights?format={fmt}");
        assert!(!bytes.is_empty(), "weights?format={fmt} empty");
    }
}
