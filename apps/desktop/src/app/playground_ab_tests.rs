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

    // Regression (R1): clear_bench() keeps entry_image_order and image_sort synchronized, and
    // dismiss_bench_form() ignores a preview hidden by the active error filter.
    drive(&mut app, window, |this, w, cx| {
        this.bench.entry_image_order = EntryImageOrder::InOrder;
        this.bench.image_sort = crate::bench::ImageSort::Order;
        this.clear_bench();
        assert_eq!(this.bench.entry_image_order, EntryImageOrder::InOrder);
        assert_eq!(this.bench.image_sort, crate::bench::ImageSort::Order);
        // Simulate an entry with 1 clean image selected while filter is MissedGt -> dismiss must NOT claim it.
        let clean = crate::api::BenchEntryImage {
            image: crate::api::BenchImage {
                id: "clean_only".into(),
                file: "clean.jpg".into(),
                width: 64,
                height: 64,
                boxes: vec![crate::api::BBox { class: 0, x1: 0., y1: 0., x2: 10., y2: 10. }],
            },
            detections: vec![crate::api::BenchDetection {
                class: 0,
                score: 0.9,
                x1: 0.,
                y1: 0.,
                x2: 10.,
                y2: 10.,
            }],
            missed: 0,
            wrong: 0,
        };
        this.bench.entry_image_filter = EntryImageFilter::MissedGt;
        this.bench.selected_entry_image = Some((bid.into(), eid.clone(), "clean_only".into()));
        this.bench.entry_images.insert(
            (bid.into(), eid.clone()),
            crate::bench::EntryImages {
                sort: crate::bench::ImageSort::Worst,
                page: Load::Ready(crate::api::BenchEntryImagePage {
                    total: 1,
                    conf: Some(0.25),
                    images: vec![clean],
                }),
                more: false,
                more_error: None,
                seq: 1,
            },
        );
        assert!(!this.dismiss_bench_form(w, cx), "Esc must not target a filtered-out preview");
        assert!(this.bench.selected_entry_image.is_none(), "stale hidden selection is cleared");
    });
}

#[test]
fn playground_compare_freshness_guards_reenable_exports_and_class_filter() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        // Pick an NMS (O2m) model for A and another NMS model for B so IoU changes affect both.
        this.play_model = Some("m_31aa".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_pred.ready().is_some());
        // Setting a Single-mode class filter must be cleared when entering Compare mode.
        this.play_class = Some(0);
        this.set_play_compare(true, cx);
        assert_eq!(this.play_class, None, "entering Compare clears Single-mode class filter");
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.pred_a_is_fresh() && this.pred_b_is_fresh());
        // Switch back to Single, change NMS IoU, and re-enable Compare: Model B must re-run!
        let seq_before = this.play_compare.pred_b_seq;
        this.set_play_compare(false, cx);
        this.play_iou = 0.45;
        // Make sure Model B is O2m so IoU matters.
        if let Some(mb_id) = this.play_compare.model_b.clone()
            && let Some(m) = this.store.models.ready_mut().unwrap().iter_mut().find(|m| m.id == mb_id)
        {
            m.head = crate::api::Head::O2m;
        }
        assert!(!this.pred_b_is_fresh(), "Model B from old IoU must not count as fresh");
        this.set_play_compare(true, cx);
        assert!(
            this.play_compare.pred_b_seq > seq_before,
            "re-enabling Compare after IoU change triggers a fresh Model B prediction"
        );
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.pred_b_is_fresh(), "new Model B prediction completed");
        // While Model A is pending an IoU rerun, A must not be exported as fresh under new IoU.
        this.play_pred_pending = true;
        assert!(!this.pred_a_is_fresh());
        let (json, _, _) = this.build_play_export_json().expect("B is still fresh").expect("valid json");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["a"].is_null(), "stale pending Model A must be omitted from Compare export");
        assert!(v["b"].is_object(), "fresh Model B remains exported");
        this.play_pred_pending = false;

        // Regression (R2): cancelling a lowered-confidence Model B rerun via set_play_compare(false)
        // restores fetched_floor_b to the last completed floor so a later low-threshold step refetches B.
        this.play_compare.fetched_floor_b = 0.01;
        this.set_play_compare(false, cx);
        assert!(
            (this.play_compare.fetched_floor_b - super::PLAY_CONF_FLOOR).abs() < 1e-6,
            "disabling Compare restores fetched_floor_b to completed floor"
        );
        this.set_play_compare(true, cx);
        let seq_before = this.play_compare.pred_b_seq;
        this.set_play_conf(0.02, cx);
        assert!(
            this.play_compare.pred_b_seq > seq_before,
            "lowering confidence below completed floor triggers a fresh Model B prediction"
        );
    });
}

#[test]
fn playground_drag_pan_reverses_immediately_past_clamp_limit() {
    use crate::viz::{Fit, clamp_pan};
    use folo_ui::{Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent, PlatformInput, point, px};

    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    drive(&mut app, window, |this, _, cx| {
        this.set_play_compare(true, cx);
        this.play_zoom = 2.0;
        this.play_pan = (0.0, 0.0);
        cx.notify();
    });
    app.frame(window.into());

    // Position 20 px inside the right clamp edge so a +100 px drag inside Pane A overshoots by 80 px.
    let (start_pt, overshoot_pt, reverse_pt, max_x) = drive(&mut app, window, |this, _, cx| {
        let b = this.play_view.get().expect("compare Pane A bounds recorded");
        let img = this.play_image.as_ref().expect("staged image");
        let dims = (img.width as f32, img.height as f32);
        let max_x = clamp_pan(b, dims.0, dims.1, Fit::Contain, 2.0, (1e6, 0.0)).0;
        assert!(max_x > 30.0, "expected positive horizontal pan headroom, got {max_x}");
        this.play_pan = (max_x - 20.0, 0.0);
        cx.notify();
        let cy = b.origin.y + b.size.height / 2.0;
        let x0 = b.origin.x + px(20.0);
        (point(x0, cy), point(x0 + px(100.0), cy), point(x0 + px(85.0), cy), max_x)
    });
    app.frame(window.into());

    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: start_pt,
            modifiers: Modifiers::none(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.frame(window.into());

    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: overshoot_pt,
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::none(),
        }),
    );
    app.frame(window.into());
    let clamped_x = drive(&mut app, window, |this, _, _| this.play_pan.0);
    assert!((clamped_x - max_x).abs() < 1.0, "expected pan clamped at right edge {max_x}, got {clamped_x}");

    // Reverse by 15 px while still holding Left: must move immediately without an 80 px dead zone.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: reverse_pt,
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::none(),
        }),
    );
    app.frame(window.into());
    let after_reverse_x = drive(&mut app, window, |this, _, _| this.play_pan.0);
    assert!(
        (after_reverse_x - (clamped_x - 15.0)).abs() < 1.0,
        "reversing drag past clamp limit must move immediately (clamped={clamped_x}, after={after_reverse_x})"
    );
}

#[test]
fn bench_filter_switch_at_bottom_keeps_scroll_offset_anchored() {
    use crate::theme::{Mode, Theme};
    use folo_ui::{
        AppContext, Bounds, HeadlessApp, HeadlessOptions, WindowBounds, WindowOptions, point, px, size,
    };

    let api = Arc::new(MockBackend::new(4200, true));
    let win_size = size(px(1680.), px(1050.));
    let mut app = HeadlessApp::new(HeadlessOptions { size: win_size, ..Default::default() });
    let window = app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), win_size);
        let opts =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(opts, |w, cx| cx.new(|cx| super::FoloApp::new(api, true, w, cx))).expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));

    // Open a scored benchmark entry with enough images to span multiple tile rows.
    let bid = "bn_5e12";
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench(bid.into()), w, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, w, cx| {
        let eid = this
            .bench
            .entries
            .get(bid)
            .and_then(|l| l.ready())
            .and_then(|v| v.iter().find(|e| e.status == BenchEntryStatus::Done))
            .expect("done entry")
            .id
            .clone();
        this.navigate(Route::BenchEntry(bid.into(), eid), w, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Scroll to the bottom under `All`, then switch to `FalsePositives` (fewer tiles).
    let (scroll_before, grid_h_before) = drive(&mut app, window, |this, _, cx| {
        let max_y = this.page_scroll.max_offset().height;
        assert!(f32::from(max_y) > 100.0, "expected scrollable entry page, max_offset={max_y:?}");
        this.page_scroll.set_offset(point(px(0.), -max_y));
        cx.notify();
        (f32::from(-max_y), this.bench.entry_grid_height.get())
    });
    app.frame(window.into());
    assert!(grid_h_before > 100.0, "expected measured grid height, got {grid_h_before}");

    // Activate the real `eimg-filter-fp` chip via focus + Enter so the chip's own handler runs.
    drive(&mut app, window, |this, w, cx| {
        this.focus_key("eimg-filter-fp", w, cx);
    });
    app.simulate_input(
        window.into(),
        folo_ui::PlatformInput::KeyDown(folo_ui::KeyDownEvent {
            keystroke: folo_ui::Keystroke::parse("enter").unwrap(),
            is_held: false,
        }),
    );
    app.frame(window.into());
    app.frame(window.into());

    let (active_filter, scroll_after) = drive(&mut app, window, |this, _, _| {
        (this.bench.entry_image_filter, f32::from(this.page_scroll.offset().y))
    });
    assert_eq!(active_filter, EntryImageFilter::FalsePositives, "FP chip handler should activate filter");
    assert!(
        (scroll_after - scroll_before).abs() < 2.0,
        "page_scroll must stay anchored on filter shrink (before={scroll_before}, after={scroll_after})"
    );
}
