//! Tests for calibrated per-class F1 thresholds across MCP `val`, `predict`, and `predict_batch` (embedded,
//! proxied, `--confine`, older-server gating, and 32 KB response budget handling).

use std::sync::Arc;

use super::*;
use crate::Options;
use crate::proxy::HttpStore;
use crate::server::FoloMcp;
use folo_api::{ClassMetrics, Split, ValMetrics, ValReport, ValTiming};
use folo_server::Store;

/// `val` returns calibrated per-class `thresholds` on both embedded and proxied servers; `predict` and
/// `predict_batch` apply `thresholds_path` (enforcing `--confine`, including nonexistent outside paths) and
/// `per_class_conf`, reject out-of-range class IDs and mutually exclusive arguments upfront, and gate older
/// servers on `threshold_calibration`.
#[tokio::test]
async fn val_and_predict_wire_calibrated_thresholds_embedded_and_proxied() {
    let home = folo_test_support::TestDir::new("mcp-threshold-calibration");
    let outside = folo_test_support::TestDir::new("mcp-threshold-outside");
    let mock_store: Arc<dyn Store> = Arc::new(folo_server::MockStore::new());
    let datasets = mock_store.list_datasets().unwrap();
    let models = mock_store.list_models().unwrap();
    let (model, ds_id) = models
        .iter()
        .find_map(|m| {
            datasets
                .iter()
                .find(|d| d.classes.iter().any(|c| m.classes.contains(c)))
                .map(|d| (m.clone(), d.id.clone()))
        })
        .expect("matching mock model and dataset");
    let image_id = mock_store
        .list_images(&ds_id, &folo_server::store::ImageQuery { limit: 1, ..Default::default() })
        .unwrap()
        .images[0]
        .id
        .clone();

    let mock_server = folo_server::Server::new(Arc::clone(&mock_store), Default::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(mock_server.serve(listener, async move {
        let _ = stop_rx.await;
    }));
    let opts = Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: true,
        device: "cpu".into(),
        published_weights: None,
    };
    let embedded = FoloMcp::new(Arc::clone(&mock_store), None, opts.clone());
    let live_http = Arc::new(HttpStore::new(&url, None));
    let proxied = FoloMcp::new(live_http.clone(), Some(live_http), opts.clone());
    let old_http = Arc::new(HttpStore::new(&url, None).assume_features(vec![]));
    let old_mcp = FoloMcp::new(old_http.clone(), Some(old_http), opts);

    // 1. `val` returns calibrated `thresholds` on both embedded and proxied servers.
    for backend in [&embedded, &proxied] {
        let v = backend
            .inner
            .val(
                ValArgs {
                    model_id: model.id.clone(),
                    dataset_id: ds_id.clone(),
                    split: Split::Val,
                    imgsz: None,
                    conf: 0.001,
                    iou: 0.7,
                    diagnostics: false,
                    worst: None,
                    diag_conf: None,
                    diag_tp_iou: None,
                    diag_bg_iou: None,
                },
                &mut Progress::none(),
            )
            .await
            .unwrap();
        let thr = &v.value["thresholds"];
        assert!(thr.is_object(), "val must include thresholds: {}", v.value);
        assert!(thr["global"]["best_conf"].is_f64(), "{thr}");
        assert!(thr["macro_f1"].is_f64(), "{thr}");
        assert!(!thr["per_class"].as_array().unwrap().is_empty(), "{thr}");
    }

    // 2. Write calibrated thresholds files inside `data_dir` (1.0 suppresses all boxes, 0.0 keeps detections).
    let suppress_path = home.join("suppress.json");
    let keep_path = home.join("keep.json");
    let per_class_one: Vec<_> = (0..model.classes.len())
        .map(|c| serde_json::json!({"class": c, "best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4}))
        .collect();
    let per_class_zero: Vec<_> = (0..model.classes.len())
        .map(|c| serde_json::json!({"class": c, "best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4}))
        .collect();
    std::fs::write(
        &suppress_path,
        serde_json::to_vec(&serde_json::json!({
            "global": {"best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": per_class_one,
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &keep_path,
        serde_json::to_vec(&serde_json::json!({
            "global": {"best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": per_class_zero,
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();

    let img_path = home.join("sample.png");
    let png = folo_data::encode(&folo_data::Image::new(32, 32, [100, 100, 100]), folo_data::ImageFormat::Png)
        .unwrap();
    std::fs::write(&img_path, &png).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let all_one: BTreeMap<u32, f32> = (0..model.classes.len() as u32).map(|c| (c, 1.0)).collect();

    for backend in [&embedded, &proxied] {
        let kept = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: None,
                dataset_id: Some(ds_id.clone()),
                image_id: Some(image_id.clone()),
                conf: 0.25,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: Some("keep.json".into()),
                per_class_conf: None,
            })
            .await
            .unwrap();
        let suppressed = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: None,
                dataset_id: Some(ds_id.clone()),
                image_id: Some(image_id.clone()),
                conf: 0.25,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: Some("suppress.json".into()),
                per_class_conf: None,
            })
            .await
            .unwrap();
        assert!(!kept.value["boxes"].as_array().unwrap().is_empty(), "keep.json should keep boxes");
        assert!(suppressed.value["boxes"].as_array().unwrap().is_empty(), "suppress.json should drop boxes");

        // Direct `per_class_conf` map over multipart (upload) also filters deterministically.
        let mp = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: Some(b64.clone()),
                dataset_id: None,
                image_id: None,
                conf: 0.0,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: Some(all_one.clone()),
            })
            .await
            .unwrap();
        assert!(mp.value["boxes"].as_array().unwrap().is_empty(), "per_class_conf=1.0 should drop boxes");

        // `predict_batch` applies both `thresholds_path` and `per_class_conf` cleanly across embedded & proxy.
        let batch_keep = backend
            .inner
            .predict_batch(
                PredictBatchArgs {
                    model_id: model.id.clone(),
                    paths: Some(vec!["sample.png".into()]),
                    glob: None,
                    conf: 0.25,
                    iou: 0.7,
                    tile: None,
                    tile_overlap: None,
                    tile_full_image: None,
                    tta: false,
                    tta_scales: None,
                    tta_hflip: None,
                    thresholds_path: Some("keep.json".into()),
                    per_class_conf: None,
                },
                &mut Progress::none(),
            )
            .await
            .unwrap();
        let batch_drop = backend
            .inner
            .predict_batch(
                PredictBatchArgs {
                    model_id: model.id.clone(),
                    paths: Some(vec!["sample.png".into()]),
                    glob: None,
                    conf: 0.0,
                    iou: 0.7,
                    tile: None,
                    tile_overlap: None,
                    tile_full_image: None,
                    tta: false,
                    tta_scales: None,
                    tta_hflip: None,
                    thresholds_path: None,
                    per_class_conf: Some(all_one.clone()),
                },
                &mut Progress::none(),
            )
            .await
            .unwrap();
        assert!(batch_keep.value["boxes_total"].as_u64().unwrap() > 0, "{}", batch_keep.value);
        assert_eq!(batch_drop.value["boxes_total"], 0, "{}", batch_drop.value);
    }

    // 3. Mutual exclusion, `--confine` (existing AND nonexistent outside paths), out-of-range class IDs, and
    //    proxy `not_implemented` gate (for both `predict` and `predict_batch`) all fail with the expected codes.
    let both = embedded
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: Some("keep.json".into()),
            per_class_conf: Some(BTreeMap::from([(0, 0.5)])),
        })
        .await
        .err()
        .unwrap();
    assert_eq!(both.body.code, "invalid_arguments");

    let escaped_file = outside.join("outside.json");
    std::fs::copy(&keep_path, &escaped_file).unwrap();
    for escaped in [
        escaped_file.to_string_lossy().into_owned(),
        outside.join("does-not-exist.json").to_string_lossy().into_owned(),
        "../nonexistent-outside.json".to_string(),
    ] {
        let confined = embedded
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: None,
                dataset_id: Some(ds_id.clone()),
                image_id: Some(image_id.clone()),
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: Some(escaped.clone()),
                per_class_conf: None,
            })
            .await
            .err()
            .unwrap();
        assert_eq!(confined.body.code, "path_outside_data_dir", "{escaped}");
    }

    // Out-of-range class IDs rejected cleanly on both `predict` (`invalid_config`) and `predict_batch`
    // (`invalid_arguments`, checked upfront against model metadata before iterating images).
    let oob_class = model.classes.len() as u32 + 3;
    let oob_pred = embedded
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: Some(BTreeMap::from([(oob_class, 0.5)])),
        })
        .await
        .err()
        .unwrap();
    assert_eq!(oob_pred.body.code, "invalid_config");

    let oob_batch = embedded
        .inner
        .predict_batch(
            PredictBatchArgs {
                model_id: model.id.clone(),
                paths: Some(vec!["sample.png".into()]),
                glob: None,
                conf: 0.25,
                iou: 0.7,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: Some(BTreeMap::from([(oob_class, 0.5)])),
            },
            &mut Progress::none(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(oob_batch.body.code, "invalid_arguments");

    let old_err = old_mcp
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id),
            image_id: Some(image_id),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: Some(BTreeMap::from([(0, 0.5)])),
        })
        .await
        .err()
        .unwrap();
    assert_eq!(old_err.body.code, "not_implemented");

    let old_batch_err = old_mcp
        .inner
        .predict_batch(
            PredictBatchArgs {
                model_id: model.id,
                paths: Some(vec!["sample.png".into()]),
                glob: None,
                conf: 0.25,
                iou: 0.7,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: Some(BTreeMap::from([(0, 0.5)])),
            },
            &mut Progress::none(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(old_batch_err.body.code, "not_implemented");

    let _ = stop_tx.send(());
    let _ = task.await;
}

/// When a `ValToolResult` exceeds the 32 KiB MCP budget, `fit_val_result` omits `thresholds` whole (never
/// returning a silently truncated partial calibration).
#[test]
fn oversized_val_result_omits_thresholds_whole_instead_of_partial() {
    let per_class_thr: Vec<ClassThresholdOut> = (0..400)
        .map(|c| ClassThresholdOut {
            class: c,
            best_conf: 0.35,
            best_f1: 0.8,
            precision: 0.8,
            recall: 0.8,
            truths: 10,
            preds: 10,
        })
        .collect();
    let mut result = ValToolResult {
        report: ValReport {
            model_id: "m_big".into(),
            dataset_id: "ds_big".into(),
            dataset_name: Some("big".into()),
            split: Split::Val,
            imgsz: 640,
            images: 100,
            val: ValMetrics {
                map50: 0.7,
                map50_95: 0.5,
                precision: 0.8,
                recall: 0.7,
                best_f1_conf: Some(0.35),
                per_class: vec![ClassMetrics {
                    class: 0,
                    map50: 0.7,
                    map50_95: 0.5,
                    precision: 0.8,
                    recall: 0.7,
                }],
            },
            timing: ValTiming { total_s: 1.0, per_image_ms: 10.0 },
        },
        diagnostics: Some(ValDiagnosticsOut {
            conf: 0.25,
            tp_iou: 0.5,
            bg_iou: 0.1,
            images_diagnosed: 1,
            failing_images: 1,
            summary: ValDiagnosticsSummaryOut {
                num_truths: 2,
                num_preds: 2,
                tp: 1,
                fp: 1,
                fn_count: 1,
                localization_errors: 1,
                class_confusions: 0,
                duplicate_errors: 0,
                background_fp: 0,
                mean_f1: 0.5,
                mean_tp_iou: 0.7,
            },
            worst_images: vec![ImageDiagnosticsOut {
                image_index: 0,
                image_id: Some("images/val/0.jpg".into()),
                num_truths: 2,
                num_preds: 2,
                tp: 1,
                fp: 1,
                fn_count: 1,
                localization_errors: 1,
                class_confusions: 0,
                duplicate_errors: 0,
                background_fp: 0,
                f1: 0.5,
                mean_tp_iou: 0.7,
            }],
        }),
        thresholds: Some(ThresholdCalibrationOut {
            global: OperatingPointOut { best_conf: 0.35, best_f1: 0.75, precision: 0.8, recall: 0.7 },
            per_class: per_class_thr,
            macro_f1: 0.8,
        }),
    };
    assert!(json_len(&result) > MAX_RESULT_BYTES);
    assert!(fit_val_result(&mut result), "must report that thresholds were omitted");
    assert!(result.thresholds.is_none(), "must omit whole calibration, never partial");
    assert_eq!(
        result.diagnostics.as_ref().unwrap().worst_images.len(),
        1,
        "diagnostics that fit after omitting oversized calibration must be preserved"
    );
    assert!(json_len(&result) <= MAX_RESULT_BYTES);
}
