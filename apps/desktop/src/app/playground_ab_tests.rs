//! Headless UI tests for Sprint 12 Desktop features:
//! 1. Playground Model A/B Visual Compare (toggle, default Model B, swap, stale async cancellation,
//!    model deletion reconciliation, backend reset, narrow & wide rendering).
//! 2. Playground Annotated PNG & formatted JSON file export (Single and Compare modes via `TestDir`).
//! 3. Benchmark Entry Image error-analysis filter chips, sort orders, GT/Pred/Errors-only toggles,
//!    and per-image TP/FP/FN badges.

use std::sync::Arc;

use super::Route;
use super::tests::{drive, open, open_with};
use crate::api::mock::MockBackend;
use crate::api::{BenchEntryStatus, ImageSize, PredictResponse, ScoredBox, Timing};
use crate::screens::bench::{EntryImageFilter, EntryImageOrder};
use crate::store::Load;

fn fake_pred(model_id: &str, total_ms: f64, boxes: Vec<ScoredBox>) -> PredictResponse {
    PredictResponse {
        model_id: model_id.into(),
        image: ImageSize { width: 640, height: 480 },
        boxes,
        timing: Timing {
            preprocess_ms: 1.0,
            inference_ms: (total_ms - 1.5).max(0.2),
            postprocess_ms: 0.5,
            total_ms,
        },
        device: "cpu".into(),
        cold: false,
    }
}

#[test]
fn playground_compare_ab_lifecycle_swap_cancellation_and_reconcile() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());

    // Stage an image on Playground and enable Compare A/B.
    let (model_a, model_b) = drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        let a = this.play_model.clone().expect("default model A");
        let b = this
            .store
            .models
            .ready()
            .unwrap()
            .iter()
            .find(|m| m.id != a)
            .expect("second demo model")
            .id
            .clone();
        (a, b)
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_pred.ready().is_some());
        this.set_play_compare(true, cx);
    });
    app.frame(window.into());

    // Model B auto-selected (distinct from A) and predicted cleanly.
    drive(&mut app, window, |this, _, _| {
        assert!(this.play_compare.enabled);
        let chosen_b = this.play_compare.model_b.clone().expect("model B picked");
        assert_ne!(
            chosen_b, model_a,
            "default Model B should differ from Model A when multiple models exist"
        );
        assert!(this.play_compare.pred_b.ready().is_some(), "Model B prediction finished");
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "play-mode",
            "play-model-b",
            "play-swap-ab",
            "play-export-png",
            "play-export-json",
            "stage-canvas-b",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing compare control {expected} in {keys:?}");
        }
    });

    // Pick explicit model_b, then swap A and B and confirm slots + predictions swap cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.pick_play_model_b(model_b.clone(), cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_model.as_deref(), Some(model_a.as_str()));
        assert_eq!(this.play_compare.model_b.as_deref(), Some(model_b.as_str()));
        this.swap_play_models(cx);
        assert_eq!(this.play_model.as_deref(), Some(model_b.as_str()));
        assert_eq!(this.play_compare.model_b.as_deref(), Some(model_a.as_str()));
        assert!(this.play_pred.ready().is_some() && this.play_compare.pred_b.ready().is_some());
    });

    // Stale async cancellation: a late Model B response with an older sequence is ignored.
    drive(&mut app, window, |this, _, _| {
        let stale_seq = this.play_compare.pred_b_seq;
        this.play_compare.pred_b_seq += 1;
        let current_boxes = this.play_compare.pred_b.ready().unwrap().boxes.len();
        let bogus = fake_pred(
            "stale",
            999.0,
            vec![ScoredBox { class: 9, name: "bogus".into(), score: 0.99, x1: 0., y1: 0., x2: 10., y2: 10. }],
        );
        this.play_pred_b_answered(stale_seq, (this.play_image_gen, "stale".into()), Ok(bogus));
        assert_eq!(
            this.play_compare.pred_b.ready().unwrap().boxes.len(),
            current_boxes,
            "stale async Model B result must not overwrite current prediction"
        );
    });

    // Disabling Compare mode cancels in-flight Model B responses.
    drive(&mut app, window, |this, _, cx| {
        let seq_before = this.play_compare.pred_b_seq;
        this.set_play_compare(false, cx);
        assert!(!this.play_compare.enabled);
        assert!(this.play_compare.pred_b_seq > seq_before, "toggling off bumps pred_b_seq");
    });

    // Re-enabling and deleting Model B reconciles cleanly to a surviving model.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_compare(true, cx);
        let gone_id = this.play_compare.model_b.clone().unwrap();
        this.store.models.ready_mut().unwrap().retain(|m| m.id != gone_id);
        this.reconcile_play_model(cx);
        assert_ne!(this.play_compare.model_b.as_deref(), Some(gone_id.as_str()));
        assert!(this.play_compare.model_b.is_some());
    });

    // Switching backends resets Compare mode cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.set_backend(Arc::new(MockBackend::new(4200, true)), super::ServerMode::Local, cx);
        assert!(!this.play_compare.enabled);
        assert!(this.play_compare.model_b.is_none());
        assert!(matches!(this.play_compare.pred_b, Load::Idle));
    });
}

#[test]
fn playground_exports_annotated_png_and_json_in_single_and_compare_modes() {
    let (mut app, window) = open();
    let dir = folo_test_support::TestDir::new("desktop-play-export-ui");

    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.play_image.is_some() && this.play_pred.ready().is_some());
    });

    // Single mode PNG export.
    let single_png = dir.path().join("single-out.png");
    app.platform().answer_prompts(Some(vec![single_png.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_png(cx));
    assert!(single_png.exists(), "Single PNG export file created");
    let single_img = image::open(&single_png).expect("valid PNG").to_rgba8();
    let (w, h) = drive(&mut app, window, |this, _, _| {
        let im = this.play_image.as_ref().unwrap();
        (im.width, im.height)
    });
    assert_eq!(single_img.dimensions(), (w, h));

    // Single mode JSON export.
    let single_json = dir.path().join("single-out.json");
    app.platform().answer_prompts(Some(vec![single_json.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_json(cx));
    let parsed_single: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&single_json).unwrap()).expect("valid JSON");
    assert!(parsed_single.get("detections").is_some());

    // Compare mode PNG + JSON export.
    drive(&mut app, window, |this, _, cx| this.set_play_compare(true, cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.play_compare.pred_b.ready().is_some());
    });

    let cmp_png = dir.path().join("compare-out.png");
    app.platform().answer_prompts(Some(vec![cmp_png.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_png(cx));
    let cmp_img = image::open(&cmp_png).expect("valid Compare PNG").to_rgba8();
    assert!(cmp_img.width() > w * 2, "Compare PNG puts A and B side-by-side");
    assert!(cmp_img.height() > h, "Compare PNG includes header strip");

    let cmp_json = dir.path().join("compare-out.json");
    app.platform().answer_prompts(Some(vec![cmp_json.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_json(cx));
    let parsed_cmp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cmp_json).unwrap()).expect("valid Compare JSON");
    assert_eq!(parsed_cmp["mode"], "compare");
    assert!(parsed_cmp["comparison"]["count_a"].is_number());
    assert!(parsed_cmp["comparison"]["count_b"].is_number());
    assert!(parsed_cmp["a"]["detections"].is_array());
    assert!(parsed_cmp["b"]["detections"].is_array());
}

#[test]
fn benchmark_entry_images_error_analysis_filters_and_overlays_render_cleanly() {
    let (mut app, window) = open();
    let bid = "bn_5e12";
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Bench(bid.into()), w, cx));
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    let eid = drive(&mut app, window, |this, w, cx| {
        let eid = this
            .bench
            .entries
            .get(bid)
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|e| e.status == BenchEntryStatus::Done)
            .unwrap()
            .id
            .clone();
        this.navigate(Route::BenchEntry(bid.into(), eid.clone()), w, cx);
        eid
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Verify filter chips, overlay toggles, and sort control are wired and focusable.
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "entry-img-sort",
            "eimg-filter-all",
            "eimg-filter-fn",
            "eimg-filter-fp",
            "eimg-filter-err",
            "eimg-grid-gt",
            "eimg-grid-preds",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing bench error-analysis control {expected}");
        }
    });

    // Cycle filters, sort modes, and open an entry image with Errors-only toggle without panic.
    drive(&mut app, window, |this, _, cx| {
        let first_img = this
            .bench
            .entry_images
            .get(&(bid.into(), eid.clone()))
            .and_then(|x| x.page.ready())
            .unwrap()
            .images
            .first()
            .unwrap()
            .image
            .id
            .clone();
        this.bench.selected_entry_image = Some((bid.into(), eid.clone(), first_img));
        this.bench.entry_image_filter = EntryImageFilter::HasErrors;
        this.bench.entry_image_order = EntryImageOrder::LowestIou;
        this.bench.entry_inspect_errors_only = true;
        cx.notify();
    });
    app.frame(window.into());

    drive(&mut app, window, |this, _, cx| {
        this.bench.entry_image_filter = EntryImageFilter::MissedGt;
        this.bench.entry_image_order = EntryImageOrder::MostFn;
        this.bench.entry_inspect_hide_truth = true;
        cx.notify();
    });
    app.frame(window.into());

    drive(&mut app, window, |this, _, cx| {
        this.bench.entry_image_filter = EntryImageFilter::FalsePositives;
        this.bench.entry_image_order = EntryImageOrder::MostFp;
        this.bench.entry_inspect_hide_truth = false;
        this.bench.entry_inspect_hide_preds = true;
        cx.notify();
    });
    app.frame(window.into());
}
