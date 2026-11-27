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

    // Undo duplicate -> clean again; switch filter to Reviewed (which hides Pending id1), then press 'U':
    // jumps to id1 and automatically switches filter back to All.
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.label.editor.is_dirty());
        this.label.img_filter = LabelImageFilter::Reviewed;
        cx.notify();
    });
    settle(&mut app, window);
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
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 matching images (excluding empty b.png):\n{csv}");
        assert!(lines[1].starts_with("a.png,done,1,circle,0.9100,6.50,640,480,"));
        assert!(lines[2].starts_with("c.png,done,1,square,0.8400,6.50,640,480,"));
    });

    drive(&mut app, window, |this, _, cx| this.step_play_batch(1, cx));
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 2);

    // Switch filter to Empty -> jumps automatically to index 1 and Summary CSV contains only b.png.
    drive(&mut app, window, |this, _, cx| {
        this.set_play_batch_qa_filter(BatchQaFilter::Empty, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.active, 1);
        let (csv, _, _) = this.build_play_batch_summary_csv().expect("empty summary csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("b.png,done,0,,"));
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "play-batch-qa-empty"));
        assert!(keys.iter().any(|k| k == "play-batch-qa-cls-0"));
    });
}
