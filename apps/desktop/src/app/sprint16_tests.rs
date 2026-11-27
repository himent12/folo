//! End-to-end UI tests for Sprint 16 Desktop capabilities:
//! 1. Labeling sequence accelerators (`U` next unlabeled, `Ctrl+D` duplicate box, `P` copy prev frame)
//!    with undo/redo, cross-resolution scaling, edge clamping, empty-previous handling, and toolbar controls.
//! 2. Dataset Detail Health Report CSV/JSON export & verified Curate/Convert CLI Builder.
//! 3. Playground Batch QA filter (`All` / `With detections` / `Empty` / `By class`) with filtered
//!    stepping and filtered Batch Summary CSV export.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput,
    WindowBounds, WindowHandle, WindowOptions, point, px, size,
};

use crate::api::mock::MockBackend;
use crate::api::{BBox, Backend, ImageSize, LabelImageStatus, PredictResponse, ScoredBox, Timing};
use crate::app::tests::drive;
use crate::app::{DsTab, FoloApp, Route};
use crate::label::{LabelImageFilter, PendingImageSwitch, SelectedBox};
use crate::screens::datasets::{DatasetCliMode, DatasetTargetFormat};
use crate::screens::playground::{
    BatchItemResult, BatchItemStatus, BatchQaFilter, PlayBatchItem, PlayBatchSource,
};
use crate::theme::{Mode, Theme};

fn press(app: &mut HeadlessApp, w: WindowHandle<FoloApp>, key: &str) {
    let ks = Keystroke::parse(key).unwrap();
    app.simulate_input(
        w.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(w.into());
    app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
    app.run_until_idle();
    app.frame(w.into());
}

fn open_with(api: Arc<dyn Backend>, w: f32, h: f32, mode: Mode) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let mut app = HeadlessApp::new(HeadlessOptions { size: size(px(w), px(h)), ..Default::default() });
    let window = app.update(|cx| {
        cx.set_global(Theme::new(mode));
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(w), px(h)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |win, cx| cx.new(|cx| FoloApp::new(api, true, win, cx)))
            .expect("open test window")
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    (app, window)
}

fn settle(app: &mut HeadlessApp, w: WindowHandle<FoloApp>) {
    for _ in 0..4 {
        app.frame(w.into());
        app.run_until_idle();
    }
}

#[test]
fn labeling_sequence_accelerators_end_to_end() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    // Seed deterministic 3-image sequence:
    // [0] labeled 640x480 with 2 boxes (one near corner for edge clamping)
    // [1] pending 1280x960 (2x resolution for cross-resolution copy prev test)
    // [2] labeled 640x480 with 0 boxes
    let (id0, id1, _id2) = drive(&mut app, window, |this, _, cx| {
        let list = this.label.images.get_mut("lb_4f10").and_then(|l| l.ready_mut()).unwrap();
        assert!(list.len() >= 3);
        list[0].status = LabelImageStatus::Labeled;
        list[0].width = 640;
        list[0].height = 480;
        list[0].boxes = vec![
            BBox { class: 0, x1: 50.0, y1: 60.0, x2: 150.0, y2: 160.0 },
            BBox { class: 1, x1: 540.0, y1: 390.0, x2: 635.0, y2: 475.0 },
        ];
        list[1].status = LabelImageStatus::Pending;
        list[1].width = 1280;
        list[1].height = 960;
        list[1].boxes.clear();
        for im in &mut list[2..] {
            im.status = LabelImageStatus::Labeled;
            im.boxes.clear();
        }
        let ids = (list[0].id.clone(), list[1].id.clone(), list[2].id.clone());
        this.select_label_image(ids.0.clone(), cx);
        ids
    });
    settle(&mut app, window);

    // 1. Pressing 'P' on the first frame gives clear feedback and does not dirty draft.
    press(&mut app, window, "p");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.label.editor.is_dirty());
        assert_eq!(this.toasts.last().map(|t| t.title.as_ref()), Some("No previous frame"));
    });

    // 2. Select corner box 1 and duplicate via Ctrl+D -> flips inward, updates selection, marks dirty,
    //    and leaves 'D' (Draw tool) intact and conflict-free.
    drive(&mut app, window, |this, _, cx| {
        let rev = this.label.editor.draft_revision();
        this.label.selected_box = Some(SelectedBox { image_id: id0.clone(), index: 1, revision: rev });
        cx.notify();
    });
    settle(&mut app, window);
    press(&mut app, window, "ctrl-d");
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.editor.is_dirty_for("lb_4f10", &id0));
        assert_eq!(this.label.editor.draft.len(), 3);
        let dup = this.label.editor.draft[2];
        assert!(dup.x1 >= 0.0 && dup.x2 <= 640.0 && dup.y1 >= 0.0 && dup.y2 <= 480.0);
        assert!(dup.x1 < 540.0 && dup.y1 < 390.0, "near-corner duplicate flipped inward: {dup:?}");
        assert_eq!(this.label.selected_box.as_ref().map(|s| s.index), Some(2));
        // Visible toolbar buttons mounted.
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "label-next-unlabeled"));
        assert!(keys.iter().any(|k| k == "label-dup-box"));
    });

    // 3. Pressing 'U' while dirty guards unsaved edits with PendingImageSwitch::JumpToImage.
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.selected_image.as_deref(), Some(id0.as_str()));
        assert_eq!(
            this.label.editor.pending_switch,
            Some(PendingImageSwitch::JumpToImage { image_id: id1.clone(), clear_filters: false })
        );
    });

    // Undo duplicate -> clean again; switch filter to Failed (which matches 0 images, so
    // active_label_image_id is None and the empty-filter view renders), verify the empty-filter
    // Next unlabeled button mounts, and press 'U': jumps to Pending id1 and resets filter to All!
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.label.editor.is_dirty());
        this.label.img_filter = LabelImageFilter::Failed;
        this.reconcile_selected_label_image("lb_4f10");
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.selected_image.is_none(), "empty Failed filter has no active image");
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(
            keys.iter().any(|k| k == "label-next-unlabeled"),
            "Next unlabeled button must be available on empty-filter view"
        );
    });
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.selected_image.as_deref(), Some(id1.as_str()));
        assert_eq!(this.label.img_filter, LabelImageFilter::All);
    });

    // Pressing 'U' again while on the only unlabeled image gives clear status feedback.
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.toasts.last().map(|t| t.title.as_ref()), Some("Already on the only unlabeled image"));
    });

    // 4. Pressing 'P' on id1 (1280x960) copies both boxes from id0 (640x480) scaled 2x as one atomic undo step.
    press(&mut app, window, "p");
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.editor.is_dirty_for("lb_4f10", &id1));
        assert_eq!(this.label.editor.draft.len(), 2);
        assert_eq!(this.label.editor.draft[0], BBox { class: 0, x1: 100.0, y1: 120.0, x2: 300.0, y2: 320.0 });
        assert_eq!(this.label.editor.undo.len(), 1, "copied boxes push one atomic undo step");
    });

    // Pressing 'P' a second time skips exact duplicates.
    press(&mut app, window, "p");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), 2);
        assert_eq!(this.toasts.last().map(|t| t.title.as_ref()), Some("Boxes already present"));
    });

    // Single Ctrl+Z reverts all copied boxes atomically.
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.editor.draft.is_empty());
        assert!(!this.label.editor.is_dirty());
    });

    // 5. When all images are labeled, 'U' reports "All images are labeled".
    drive(&mut app, window, |this, _, cx| {
        if let Some(list) = this.label.images.get_mut("lb_4f10").and_then(|l| l.ready_mut()) {
            list[1].status = LabelImageStatus::Labeled;
        }
        cx.notify();
    });
    settle(&mut app, window);
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.toasts.last().map(|t| t.title.as_ref()), Some("All images are labeled"));
    });
}

#[test]
fn dataset_health_report_export_and_curate_cli_builder_render_and_copy() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);

    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Dataset("ds_3b91".into()), w, cx);
        this.ds_tab = DsTab::Statistics;
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "ds-export-health-csv"), "Health report CSV mounted");
        assert!(keys.iter().any(|k| k == "ds-export-health-json"), "Health report JSON mounted");
        assert!(keys.iter().any(|k| k == "ds-cli-mode"), "Curate CLI mode switcher mounted");
        assert!(keys.iter().any(|k| k == "ds-cli-copy"), "Curate CLI copy button mounted");
    });

    // Switch CLI builder to Convert -> VOC and copy via Enter on ds-cli-copy.
    drive(&mut app, window, |this, w, cx| {
        this.ds_cli.mode = DatasetCliMode::Convert;
        this.ds_cli.convert_to = DatasetTargetFormat::Voc;
        this.focus_key("ds-cli-copy", w, cx);
        cx.notify();
    });
    settle(&mut app, window);
    press(&mut app, window, "enter");

    let copied = drive(&mut app, window, |_, _, cx| cx.read_from_clipboard().and_then(|c| c.text()));
    let cmd = copied.expect("copied dataset CLI command");
    assert!(cmd.starts_with("folo dataset convert "), "unexpected CLI command: {cmd}");
    assert!(cmd.contains(" --from coco "), "expected registered source format --from coco in: {cmd}");
    assert!(cmd.ends_with(" --to voc"), "expected --to voc in: {cmd}");
}

#[test]
fn playground_batch_qa_filter_stepping_and_summary_csv_respect_active_filter() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Light);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::Playground, w, cx));
    settle(&mut app, window);

    // Seed a 3-item batch: [0] circle, [1] empty, [2] square.
    let model_id = drive(&mut app, window, |this, _, _| this.play_model.clone().expect("default model"));
    let mk_res = |boxes: Vec<ScoredBox>| {
        BatchItemStatus::Done(BatchItemResult {
            pred: PredictResponse {
                model_id: model_id.clone(),
                image: ImageSize { width: 640, height: 480 },
                boxes,
                timing: Timing { preprocess_ms: 1.0, inference_ms: 5.0, postprocess_ms: 0.5, total_ms: 6.5 },
                device: "cpu".into(),
                cold: false,
            },
            width: 640,
            height: 480,
            model_id: model_id.clone(),
            iou_bits: 0.7_f32.to_bits(),
            target: None,
            floor: 0.05,
            content_hash: None,
        })
    };
    drive(&mut app, window, |this, _, cx| {
        this.play_batch.clear();
        this.play_batch.items = vec![
            PlayBatchItem {
                name: "shapes / a.png".into(),
                source: PlayBatchSource::Dataset {
                    dataset_id: "ds_3b91".into(),
                    image_id: "img_00801".into(),
                },
                status: mk_res(vec![ScoredBox {
                    class: 0,
                    name: "circle".into(),
                    score: 0.91,
                    x1: 10.,
                    y1: 10.,
                    x2: 60.,
                    y2: 60.,
                }]),
            },
            PlayBatchItem {
                name: "shapes / b.png".into(),
                source: PlayBatchSource::Dataset {
                    dataset_id: "ds_3b91".into(),
                    image_id: "img_00802".into(),
                },
                status: mk_res(vec![]),
            },
            PlayBatchItem {
                name: "shapes / c.png".into(),
                source: PlayBatchSource::Dataset {
                    dataset_id: "ds_3b91".into(),
                    image_id: "img_00803".into(),
                },
                status: mk_res(vec![ScoredBox {
                    class: 1,
                    name: "square".into(),
                    score: 0.84,
                    x1: 20.,
                    y1: 20.,
                    x2: 80.,
                    y2: 80.,
                }]),
            },
        ];
        this.play_batch.active = 0;
        cx.notify();
    });
    settle(&mut app, window);

    // Filter to WithDetections -> [0, 2]; stepping forward skips empty index 1!
    drive(&mut app, window, |this, _, cx| {
        this.set_play_batch_qa_filter(BatchQaFilter::WithDetections, cx);
        let (csv, _, _) = this.build_play_batch_summary_csv().expect("filtered summary csv");
        let lines: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "header + 2 matching images (excluding empty b.png):\n{csv}");
        assert!(lines[1].starts_with("a.png,done,1,circle,0.9100,6.50,640,480,"));
        assert!(lines[2].starts_with("c.png,done,1,square,0.8400,6.50,640,480,"));
    });

    drive(&mut app, window, |this, _, cx| this.step_play_batch(1, cx));
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 2);

    // Conflicting Single-mode class filter (play_class = Circle) + ByClass(Square):
    // set_play_batch_qa_filter clears conflicting play_class, and Summary CSV still reports Square!
    drive(&mut app, window, |this, _, cx| {
        this.play_class = Some(0);
        this.set_play_batch_qa_filter(BatchQaFilter::ByClass(1), cx);
        assert_eq!(this.play_class, None, "conflicting play_class cleared on ByClass select");
        // Even if play_class were set conflictingly again, effective_class guarantees non-zero Square export.
        this.play_class = Some(0);
        let (csv, _, _) = this.build_play_batch_summary_csv().expect("by-class summary csv");
        let lines: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("c.png,done,1,square,0.8400,6.50,640,480,"), "{csv}");
        this.play_class = None;
    });

    // Empty-filter "Show all" button restores keyboard focus to play-batch-qa-all when activated.
    drive(&mut app, window, |this, w, cx| {
        this.set_play_batch_qa_filter(BatchQaFilter::ByClass(99), cx);
        cx.notify();
        let _ = w;
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        this.focus_key("play-batch-qa-reset", w, cx);
        assert!(this.fh("play-batch-qa-reset", cx).is_focused(w));
    });
    press(&mut app, window, "enter");
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.play_batch.qa_filter, BatchQaFilter::All);
        assert!(
            this.fh("play-batch-qa-all", cx).is_focused(w),
            "focus must transfer from unmounting Show all button to All chip"
        );
    });

    // Model switch invalidates batch and resets ByClass back to All so stale class IDs never linger.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_batch_qa_filter(BatchQaFilter::ByClass(1), cx);
        assert_eq!(this.play_batch.qa_filter, BatchQaFilter::ByClass(1));
        let other_model =
            this.store.models.ready().unwrap().iter().find(|m| m.id != model_id).map(|m| m.id.clone());
        if let Some(other) = other_model {
            this.pick_play_model(other, cx);
        } else {
            this.invalidate_play_batch();
        }
        assert_eq!(this.play_batch.qa_filter, BatchQaFilter::All, "model switch resets QA filter to All");
    });

    // Restore original model + fresh results and verify Empty filter jumps to b.png.
    drive(&mut app, window, |this, _, cx| {
        this.play_model = Some(model_id.clone());
        this.play_batch.items[0].status = mk_res(vec![ScoredBox {
            class: 0,
            name: "circle".into(),
            score: 0.91,
            x1: 10.,
            y1: 10.,
            x2: 60.,
            y2: 60.,
        }]);
        this.play_batch.items[1].status = mk_res(vec![]);
        this.play_batch.items[2].status = mk_res(vec![ScoredBox {
            class: 1,
            name: "square".into(),
            score: 0.84,
            x1: 20.,
            y1: 20.,
            x2: 80.,
            y2: 80.,
        }]);
        this.set_play_batch_qa_filter(BatchQaFilter::Empty, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.active, 1);
        let (csv, _, _) = this.build_play_batch_summary_csv().expect("empty summary csv");
        let lines: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("b.png,done,0,,"));
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "play-batch-qa-empty"));
        assert!(keys.iter().any(|k| k == "play-batch-qa-cls-0"));
    });
}

#[test]
fn playtest_r16_playground_byclass_syncs_stage_rail_compare_confidence_and_focus() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 960., 620., Mode::Light);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::Playground, w, cx));
    settle(&mut app, window);

    let model_id = drive(&mut app, window, |this, _, _| this.play_model.clone().expect("default model"));
    let mk_pred = |mid: &str, boxes: Vec<ScoredBox>| PredictResponse {
        model_id: mid.into(),
        image: ImageSize { width: 640, height: 480 },
        boxes,
        timing: Timing { preprocess_ms: 1.0, inference_ms: 5.0, postprocess_ms: 0.5, total_ms: 6.5 },
        device: "cpu".into(),
        cold: false,
    };
    let mk_res = |boxes: Vec<ScoredBox>| {
        BatchItemStatus::Done(BatchItemResult {
            pred: mk_pred(&model_id, boxes),
            width: 640,
            height: 480,
            model_id: model_id.clone(),
            iou_bits: 0.7_f32.to_bits(),
            target: None,
            floor: 0.05,
            content_hash: None,
        })
    };

    // Batch with 2 images:
    // [0] weak circle (0.50) + strong square (0.92)
    // [1] strong circle (0.90) + square (0.85) + star (0.88)
    let boxes_0 = vec![
        ScoredBox { class: 0, name: "circle".into(), score: 0.50, x1: 10., y1: 10., x2: 50., y2: 50. },
        ScoredBox { class: 1, name: "square".into(), score: 0.92, x1: 60., y1: 10., x2: 110., y2: 60. },
    ];
    let boxes_1 = vec![
        ScoredBox { class: 0, name: "circle".into(), score: 0.90, x1: 15., y1: 15., x2: 65., y2: 65. },
        ScoredBox { class: 1, name: "square".into(), score: 0.85, x1: 70., y1: 15., x2: 120., y2: 65. },
        ScoredBox { class: 2, name: "star".into(), score: 0.88, x1: 130., y1: 15., x2: 180., y2: 65. },
    ];
    drive(&mut app, window, |this, _, cx| {
        this.play_conf = 0.35;
        this.play_batch.clear();
        this.play_batch.items = vec![
            PlayBatchItem {
                name: "shapes / img_0.jpg".into(),
                source: PlayBatchSource::Dataset {
                    dataset_id: "ds_3b91".into(),
                    image_id: "img_00801".into(),
                },
                status: mk_res(boxes_0.clone()),
            },
            PlayBatchItem {
                name: "shapes / img_1.jpg".into(),
                source: PlayBatchSource::Dataset {
                    dataset_id: "ds_3b91".into(),
                    image_id: "img_00802".into(),
                },
                status: mk_res(boxes_1.clone()),
            },
        ];
        this.play_batch.active = 0;
        this.select_play_batch_item(0, cx);
    });
    settle(&mut app, window);

    // 1. Select ByClass(0 = circle) and step to img_1: stage visible_boxes, hover ceiling,
    //    and JSON export all show ONLY circle (1 box), never square or star!
    drive(&mut app, window, |this, _, cx| {
        this.set_play_batch_qa_filter(BatchQaFilter::ByClass(0), cx);
        this.step_play_batch(1, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.active, 1);
        let p = this.play_pred.ready().expect("pred_a ready");
        let vis = this.visible_boxes(p, None);
        assert_eq!(vis.len(), 1, "stage overlay must match ByClass(circle) filter");
        assert_eq!(vis[0].class, 0);
        let (json, _, _) = this.build_play_export_json().expect("single json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["detections"].as_array().unwrap().len(), 1);
        assert_eq!(v["detections"][0]["name"], "circle");
    });

    // 2. Compare A/B mode with ByClass(circle) filters Model A by class ID and Model B by name
    //    (even when Model B uses a different numeric class index for "circle").
    drive(&mut app, window, |this, _, cx| {
        this.set_play_compare(true, cx);
        this.play_compare.pred_b = crate::store::Load::Ready(mk_pred(
            this.play_compare.model_b.as_deref().unwrap_or("m_b"),
            vec![
                ScoredBox {
                    class: 7,
                    name: "circle".into(),
                    score: 0.87,
                    x1: 20.,
                    y1: 20.,
                    x2: 70.,
                    y2: 70.,
                },
                ScoredBox {
                    class: 0,
                    name: "square".into(),
                    score: 0.91,
                    x1: 80.,
                    y1: 20.,
                    x2: 130.,
                    y2: 70.,
                },
            ],
        ));
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let pb = this.play_compare.pred_b.ready().expect("pred_b ready");
        let vis_b = this.visible_boxes_b(pb, None);
        assert_eq!(vis_b.len(), 1, "Model B must filter by matched class name");
        assert_eq!(vis_b[0].label.as_deref(), Some("circle"));
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "play-cmp-class-clear"), "Compare scope bar mounts Clear action");
    });

    // 3. Raising confidence when staged img_0 drops out of ByClass(circle) reconciles to img_1;
    //    raising past all matches (0.95) enters zero-match state cleanly without staging excluded boxes.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_compare(false, cx);
        this.select_play_batch_item(0, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_batch.active, 0);
        this.set_play_conf(0.82, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_batch.active, 1, "raising conf must reconcile off excluded img_0 onto img_1");
        assert!(!this.play_batch_qa_zero_match());
        this.set_play_conf(0.95, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.play_batch_qa_zero_match(), "0 matches at 0.95 enters zero-match state");
        assert!(!this.has_play_export_json(), "excluded stage frame is not exported during zero-match");
    });

    // 4. Focus continuity when a focused inactive class chip drops out of the top-8 during live Run All.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_conf(0.25, cx);
        this.set_play_batch_qa_filter(BatchQaFilter::All, cx);
        // Item 0 has class 9 (initially visible since only 1 class exists).
        this.play_batch.items[0].status = mk_res(vec![ScoredBox {
            class: 9,
            name: "rare".into(),
            score: 0.9,
            x1: 10.,
            y1: 10.,
            x2: 40.,
            y2: 40.,
        }]);
        this.play_batch.items[1].status = BatchItemStatus::Pending;
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        this.focus_key("play-batch-qa-cls-9", w, cx);
        assert!(this.fh("play-batch-qa-cls-9", cx).is_focused(w));
        // Now 8 higher-priority classes (0..8) appear in item 0 + item 1 (count=2 each), pushing
        // inactive class 9 (count=1) out of the top 8!
        let top8 = |extra: Option<ScoredBox>| {
            let mut v: Vec<ScoredBox> = (0..8u32)
                .map(|c| ScoredBox {
                    class: c,
                    name: format!("c{c}"),
                    score: 0.9,
                    x1: 10.,
                    y1: 10.,
                    x2: 40.,
                    y2: 40.,
                })
                .collect();
            if let Some(b) = extra {
                v.push(b);
            }
            v
        };
        this.play_batch.items[0].status = mk_res(top8(Some(ScoredBox {
            class: 9,
            name: "rare".into(),
            score: 0.9,
            x1: 10.,
            y1: 10.,
            x2: 40.,
            y2: 40.,
        })));
        this.play_batch.items[1].status = mk_res(top8(None));
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(
            this.fh("play-batch-qa-all", cx).is_focused(w),
            "focus must transfer to All chip when an inactive class chip drops out of the top 8"
        );
    });
}

#[test]
fn playtest_r16_labeling_next_unlabeled_disabled_and_guarded_during_save() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 960., 620., Mode::Light);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    let (id1, id2) = drive(&mut app, window, |this, _, cx| {
        let list = this.label.images.get_mut("lb_4f10").and_then(|l| l.ready_mut()).unwrap();
        list[0].status = LabelImageStatus::Labeled;
        list[0].boxes = vec![BBox { class: 0, x1: 20., y1: 20., x2: 120., y2: 120. }];
        list[1].status = LabelImageStatus::Pending;
        list[1].boxes.clear();
        list[2].status = LabelImageStatus::Pending;
        list[2].boxes.clear();
        let ids = (list[1].id.clone(), list[2].id.clone());
        this.select_label_image(ids.0.clone(), cx);
        ids
    });
    settle(&mut app, window);

    // Dirty frame 1 via P, simulate an in-flight save (`label.mutating = true`), and focus Next unlabeled.
    press(&mut app, window, "p");
    drive(&mut app, window, |this, w, cx| {
        assert!(this.label.editor.is_dirty_for("lb_4f10", &id1));
        this.focus_key("label-next-unlabeled", w, cx);
        assert!(this.fh("label-next-unlabeled", cx).is_focused(w));
        this.label.mutating = true;
        cx.notify();
    });
    settle(&mut app, window);

    // Toolbar button swaps to disabled with focus continuity, and pressing 'U' is ignored cleanly
    // (no stuck unsaved-edits PendingImageSwitch banner).
    drive(&mut app, window, |this, w, cx| {
        assert!(
            this.fh("label-next-unlabeled-disabled", cx).is_focused(w),
            "Next unlabeled must disable during save while preserving keyboard focus"
        );
    });
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.selected_image.as_deref(), Some(id1.as_str()));
        assert!(
            this.label.editor.pending_switch.is_none(),
            "U during save must not open an unsaved-edits confirmation banner"
        );
    });

    // Once save completes (`mutating = false` and draft clean), focus restores and 'U' jumps to id2.
    drive(&mut app, window, |this, _, cx| {
        this.label.mutating = false;
        this.label.editor.discard();
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(this.fh("label-next-unlabeled", cx).is_focused(w));
    });
    press(&mut app, window, "u");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.selected_image.as_deref(), Some(id2.as_str()));
    });
}
