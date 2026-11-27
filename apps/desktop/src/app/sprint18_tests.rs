//! Sprint 18 end-to-end tests:
//! 1. Playground per-class confidence thresholds (UI controls, JSON import/export, model-switch
//!    class remapping, Compare A/B overlays/diffs/export, Batch QA triage & batch exports).
//! 2. Models calibrated thresholds CLI (`folo val --save-thresholds`, `folo predict --thresholds`),
//!    on-demand `POST /api/val` calibration, and "Open in Playground" handoff.
//! 3. Cloud spot preemption auto-resume status across `run-cloud+preempted`, `Recovering`,
//!    `Recovered`, `Exhausted`, `Armed`, and `Unsupported` (`remote`) states.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::api::mock::MockBackend;
use crate::api::{Backend, CloudPhase, RunStatus, ScoredBox, Split};
use crate::app::Route;
use crate::app::tests::{drive, open, open_with};
use crate::screens::models::export_builder::{ModelCliMode, build_model_cli_with_arch};
use crate::screens::playground::{
    BatchItemResult, BatchItemStatus, BatchQaFilter, CompareClassFilter, PlayBatchItem, PlayBatchSource,
    PlayBatchState, filter_detections_with_per_class,
};
use crate::screens::runs::{SpotRecoveryState, spot_recovery_status};

#[test]
fn playground_per_class_thresholds_drive_overlays_compare_batch_and_remap_on_model_switch() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        // Pick shapes-hard-s-best (classes: circle, square, triangle, star, hex).
        this.pick_play_model("m_44c0".into(), cx);
        this.play_conf = 0.40;
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // 1. Expanding per-class panel renders per-class rows and actions cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.play_thresholds.expanded = true;
        this.set_play_class_conf(0, 0.85, cx);
        this.set_play_class_conf(1, 0.20, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_thresholds.per_class.get(&0), Some(&0.85));
        assert_eq!(this.play_thresholds.per_class.get(&1), Some(&0.20));
        assert_eq!(this.play_fetched_per_class.get(&0), Some(&0.85));
        assert_eq!(this.play_fetched_per_class.get(&1), Some(&0.20));
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "play-per-class-toggle",
            "play-thr-import",
            "play-thr-copy",
            "play-thr-export",
            "play-thr-reset-all",
            "play-cls-thr-dec-0",
            "play-cls-thr-inc-0",
            "play-cls-thr-rst-0",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
    });

    // 2. Effective per-class thresholds drive export JSON and Compare A/B JSON export.
    drive(&mut app, window, |this, _, cx| {
        let exported = this.build_play_thresholds_export().unwrap().0;
        let cal: folo_vision::ThresholdCalibration = serde_json::from_str(&exported).unwrap();
        cal.validate().unwrap();
        assert_eq!(cal.per_class.len(), 2);
        assert!((cal.global.best_conf - 0.40).abs() < 1e-4);
        assert!((cal.threshold_for(0) - 0.85).abs() < 1e-4);
        assert!((cal.threshold_for(1) - 0.20).abs() < 1e-4);
        assert!((cal.threshold_for(2) - 0.40).abs() < 1e-4);

        // Single-mode JSON export carries `per_class_conf` and filters detections accordingly.
        let det_json = this.build_play_export_json().unwrap().unwrap().0;
        let val: serde_json::Value = serde_json::from_str(&det_json).unwrap();
        assert!(val.get("per_class_conf").is_some(), "single export includes per_class_conf");

        // Compare A/B maps per-class thresholds by class name and emits `per_class_conf`.
        this.set_play_compare(true, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, cx| {
        let cmp_json = this.build_play_export_json().unwrap().unwrap().0;
        let val: serde_json::Value = serde_json::from_str(&cmp_json).unwrap();
        assert_eq!(val["mode"], "compare");
        assert!(val["a"].get("per_class_conf").is_some());
        this.play_compare.class_filter = CompareClassFilter::OnlyA;
        cx.notify();
    });
    app.frame(window.into());

    // 3. Import JSON overrides global + per-class values cleanly and surfaces validation error on bad JSON.
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.apply_play_thresholds_json("{not valid", cx));
        assert!(this.play_thresholds.import_error.is_some());

        let payload = r#"{"default_conf": 0.30, "per_class": {"circle": 0.70, "star": 0.15}}"#;
        assert!(this.apply_play_thresholds_json(payload, cx));
        assert!(this.play_thresholds.import_error.is_none());
        assert!((this.play_conf - 0.30).abs() < 1e-6);
        assert_eq!(this.play_thresholds.per_class.get(&0), Some(&0.70));
        assert_eq!(this.play_thresholds.per_class.get(&3), Some(&0.15));
        assert_eq!(this.play_thresholds.per_class.len(), 2);
    });

    // 4. Batch QA filtering and batch JSON/CSV/summary exports honor per-class thresholds.
    let mut per_class = BTreeMap::new();
    per_class.insert(0u32, 0.90f32);
    per_class.insert(1u32, 0.15f32);
    let boxes = vec![
        ScoredBox {
            class: 0,
            name: "circle".into(),
            score: 0.75, // blocked by class 0 override (0.90)
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        },
        ScoredBox {
            class: 1,
            name: "square".into(),
            score: 0.20, // admitted by class 1 override (0.15 < 0.40 global)
            x1: 20.0,
            y1: 20.0,
            x2: 40.0,
            y2: 40.0,
        },
    ];
    let kept = filter_detections_with_per_class(&boxes, 0.40, &per_class, None);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].class, 1);

    let mut batch = PlayBatchState::default();
    batch.items.push(PlayBatchItem {
        name: "img_1.png".into(),
        source: PlayBatchSource::File(PathBuf::from("img_1.png")),
        status: BatchItemStatus::Done(BatchItemResult {
            pred: crate::api::PredictResponse {
                model_id: "m_44c0".into(),
                image: crate::api::ImageSize { width: 100, height: 100 },
                boxes,
                timing: crate::api::Timing {
                    preprocess_ms: 1.0,
                    inference_ms: 2.0,
                    postprocess_ms: 0.5,
                    total_ms: 3.5,
                },
                device: "gpu:0".into(),
                cold: false,
            },
            width: 100,
            height: 100,
            model_id: "m_44c0".into(),
            iou_bits: 0.70f32.to_bits(),
            target: None,
            floor: 0.40,
            per_class_conf: per_class.clone(),
            content_hash: None,
        }),
    });
    let empty_pred = crate::api::PredictResponse {
        model_id: "m_44c0".into(),
        image: crate::api::ImageSize { width: 100, height: 100 },
        boxes: Vec::new(),
        timing: crate::api::Timing {
            preprocess_ms: 1.0,
            inference_ms: 2.0,
            postprocess_ms: 0.5,
            total_ms: 3.5,
        },
        device: "gpu:0".into(),
        cold: false,
    };
    batch.items.push(PlayBatchItem {
        name: "img_2.png".into(),
        source: PlayBatchSource::File(PathBuf::from("img_2.png")),
        status: BatchItemStatus::Done(BatchItemResult {
            pred: empty_pred,
            width: 100,
            height: 100,
            model_id: "m_44c0".into(),
            iou_bits: 0.70f32.to_bits(),
            target: None,
            floor: 0.40,
            per_class_conf: per_class.clone(),
            content_hash: None,
        }),
    });
    batch.qa_filter = BatchQaFilter::WithDetections;
    assert_eq!(batch.filtered_indices(Some("m_44c0"), 0.70, None, 0.40, &per_class, None, true), vec![0]);
    batch.qa_filter = BatchQaFilter::Empty;
    assert_eq!(batch.filtered_indices(Some("m_44c0"), 0.70, None, 0.40, &per_class, None, true), vec![1]);
    batch.qa_filter = BatchQaFilter::ByClass(0);
    assert_eq!(
        batch.filtered_indices(Some("m_44c0"), 0.70, None, 0.40, &per_class, None, true),
        Vec::<usize>::new()
    );
    batch.qa_filter = BatchQaFilter::ByClass(1);
    assert_eq!(batch.filtered_indices(Some("m_44c0"), 0.70, None, 0.40, &per_class, None, true), vec![0]);

    // Wire the batch into FoloApp to verify end-to-end batch JSON & Summary CSV exports.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_compare(false, cx);
        this.play_conf = 0.40;
        this.play_thresholds.per_class = per_class.clone();
        this.play_class = None;
        this.play_batch = batch;
        this.play_batch.qa_filter = BatchQaFilter::All;
        let batch_json = this.build_play_batch_export_json().unwrap().unwrap().0;
        let parsed_bj: serde_json::Value = serde_json::from_str(&batch_json).unwrap();
        assert!(parsed_bj.get("per_class_conf").is_some(), "{batch_json}");
        assert_eq!(parsed_bj["total_detections"], 1);
        assert_eq!(parsed_bj["images"][0]["detections"][0]["name"], "square");
        let summary_csv = this.build_play_batch_summary_csv().unwrap().0;
        assert!(summary_csv.contains("square") && !summary_csv.contains("circle"), "{summary_csv}");
    });

    // 4b. Global-only export stays active for `folo predict --thresholds` and round-trips cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.play_conf = 0.70;
        this.reset_all_play_class_conf(cx);
        let exported = this.build_play_thresholds_export().unwrap().0;
        let cal: folo_vision::ThresholdCalibration = serde_json::from_str(&exported).unwrap();
        cal.validate().unwrap();
        assert!(!cal.is_empty(), "global-only export must remain active for CLI --thresholds");
        assert!((cal.global.best_conf - 0.70).abs() < 1e-4);
        assert!((cal.threshold_for(0) - 0.70).abs() < 1e-4);
        assert!(this.apply_play_thresholds_json(&exported, cx));
        assert!((this.play_conf - 0.70).abs() < 1e-4);
        assert!(this.play_thresholds.per_class.is_empty());
    });

    // 5. Switching to a model with an incompatible class list clears stale overrides (no silent leak!).
    drive(&mut app, window, |this, _, cx| {
        let incompatible = this
            .store
            .models
            .ready()
            .unwrap()
            .iter()
            .find(|m| !m.classes.iter().any(|c| c == "circle" || c == "square"))
            .map(|m| m.id.clone());
        if let Some(other_id) = incompatible {
            this.pick_play_model(other_id, cx);
            assert!(
                this.play_thresholds.per_class.is_empty(),
                "incompatible class list must not inherit stale per-class overrides"
            );
        }
    });
}

#[test]
fn models_calibration_cli_and_playground_handoff_carry_calibrated_thresholds() {
    let (mut app, window) = open();
    let (mid, dsid) = drive(&mut app, window, |this, w, cx| {
        let m = this.store.models.ready().unwrap().iter().find(|m| m.run_id.is_some()).unwrap().clone();
        let rid = m.run_id.clone().unwrap();
        let dsid = this.store.run(&rid).unwrap().config.dataset_id.clone();
        this.navigate(Route::Model(m.id.clone()), w, cx);
        (m.id, dsid)
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // 1. Validate & Predict CLI builder modes render and switch cleanly.
    drive(&mut app, window, |this, _, cx| {
        assert!(this.store.supports_threshold_calibration());
        this.model_cli.mode = ModelCliMode::Validate;
        this.model_cli.val_save_thresholds = true;
        let m = this.store.model(&mid).unwrap();
        let built =
            build_model_cli_with_arch(m, this.store.model_arch(m), Some("D:/data/shapes"), &this.model_cli);
        assert!(built.command.starts_with("folo val "), "{}", built.command);
        assert!(built.command.contains("--save-thresholds "), "{}", built.command);
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "pr-calibrate-thresholds",
            "pr-open-playground",
            "model-cli-val-split",
            "model-cli-save-thresholds",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
        // Trigger on-demand calibration via `POST /api/val`.
        this.calibrate_model_thresholds(mid.clone(), dsid.clone(), Split::Val, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // 2. Calibration lands in `store.calibrations` and renders Copy JSON + Open calibrated in Playground.
    let expected_global = drive(&mut app, window, |this, _, _| {
        let cal =
            this.store.calibrations.get(&mid).and_then(|l| l.ready()).expect("calibrated thresholds loaded");
        cal.validate().unwrap();
        assert!(!cal.per_class.is_empty());
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "pr-copy-thresholds"), "missing pr-copy-thresholds in {keys:?}");
        cal.global.best_conf
    });

    // 3. Handoff into Playground carries calibrated global + per-class overrides.
    drive(&mut app, window, |this, w, cx| {
        this.open_model_in_playground(mid.clone(), true, w, cx);
    });
    for _ in 0..2 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, Route::Playground);
        assert_eq!(this.play_model.as_deref(), Some(mid.as_str()));
        assert!((this.play_conf - crate::screens::playground::round_conf(expected_global)).abs() < 1e-4);
        assert!(
            !this.play_thresholds.per_class.is_empty(),
            "calibrated per-class overrides carried into Playground"
        );
    });

    // 4. An empty calibration (no GT boxes on split) must never set `play_conf` to 0.0.
    drive(&mut app, window, |this, w, cx| {
        this.play_conf = 0.35;
        this.play_thresholds.per_class.clear();
        this.store
            .calibrations
            .insert(mid.clone(), crate::store::Load::Ready(folo_vision::ThresholdCalibration::default()));
        this.open_model_in_playground(mid.clone(), true, w, cx);
        assert!(this.play_conf > 0.05, "empty calibration must not set play_conf to 0: {}", this.play_conf);
        assert!(this.play_thresholds.per_class.is_empty());
    });
}

#[test]
fn cloud_spot_preemption_auto_resume_renders_cleanly_in_run_detail() {
    let api = Arc::new(MockBackend::with_cloud_runs(4180, true).cloud_preempted());
    let (mut app, window) = open_with(api.clone());
    drive(&mut app, window, |this, w, cx| {
        this.cloud.providers = api.cloud_providers().into();
        this.cloud.connections = api.cloud_connections().into();
        this.cloud.instances = api.cloud_instances().into();
        this.navigate(Route::Run("run_c2e1".into()), w, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, _| {
        let r = this.store.run("run_c2e1").unwrap();
        let c = r.cloud.as_ref().unwrap();
        let logs = this.store.logs.get("run_c2e1").and_then(|l| l.ready()).map(|h| h.lines.as_slice());
        let instances = this.cloud.instances.ready().map(|v| v.as_slice());
        let spot = spot_recovery_status(r, c, true, logs, instances);
        assert_eq!(spot.state, SpotRecoveryState::Recovered);
        assert_eq!(spot.retries, 1);
        assert_eq!(spot.max_retries, 3);
        assert_eq!(spot.replaced_instances, 1);
        assert!(spot.resume_epoch.is_some());
        assert!(spot.banner().is_some());
        assert!(spot.header_badge().is_some());
    });

    // Also render Recovering (queued/syncing after preemption) and Exhausted (failed) without panic.
    drive(&mut app, window, |this, _, cx| {
        if let Some(runs) = this.store.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == "run_c2e1")
        {
            r.status = RunStatus::Queued;
            if let Some(c) = r.cloud.as_mut() {
                c.phase = CloudPhase::Syncing;
            }
        }
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        if let Some(runs) = this.store.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == "run_c2e1")
        {
            r.status = RunStatus::Failed;
            r.error_code = Some("cloud_instance_lost".into());
            r.error = Some(
                "spot instance reclaimed; automatic checkpoint resume exhausted all 3 attempt(s) — the weights copied so far stay"
                    .into(),
            );
            if let Some(c) = r.cloud.as_mut() {
                c.phase = CloudPhase::Released;
            }
        }
        cx.notify();
    });
    app.frame(window.into());
}
