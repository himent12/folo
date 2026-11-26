//! Deterministic state, navigation, mutation, conflict, and render tests for Agent Labeling.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput,
    WindowBounds, WindowHandle, WindowOptions, point, px, size,
};

use crate::api::mock::MockBackend;
use crate::api::{
    Backend, FEATURE_AGENT_LABELING, ImportLabelImages, LabelImageStatus, LabelOutcome, LabelScope,
    LabelSource, UpdateLabelImage, label_box_revision,
};
use crate::app::tests::{drive, open};
use crate::app::{FoloApp, Route, Section};
use crate::label::{
    LabelImageFilter, LabelJobFilter, SelectedBox, filter_label_images, filter_label_jobs, parse_class_list,
    route_label, validate_new_job,
};
use crate::store::Load;
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
fn class_list_parsing_and_new_job_validation() {
    assert_eq!(parse_class_list(" car, person , , bike\ntruck "), vec!["car", "person", "bike", "truck"]);
    assert!(validate_new_job("", "car", "").is_err());
    assert!(validate_new_job("job", "   ,  ", "").is_err());
    assert!(validate_new_job("job", "car, Car", "").is_err());
    assert!(validate_new_job("job", "bad:class", "").is_err());
    assert_eq!(
        validate_new_job("line-check", "scratch, dent", "Tight boxes").unwrap(),
        vec!["scratch", "dent"]
    );
}

#[test]
fn navigation_ctrl7_palette_and_route_helpers() {
    assert_eq!(Route::parse("labeling"), Some(Route::LabelJobs));
    assert_eq!(Route::parse("label"), Some(Route::LabelJobs));
    assert_eq!(Route::parse("label:lb_4f10"), Some(Route::LabelJob("lb_4f10".into())));
    assert_eq!(Route::parse("label:"), None);
    assert_eq!(Route::LabelJobs.section(), Section::Labeling);
    assert_eq!(Route::LabelJob("lb_4f10".into()).section(), Section::Labeling);
    assert_eq!(route_label(&Route::LabelJob("lb_4f10".into())), Some("lb_4f10"));
    assert_eq!(route_label(&Route::LabelJobs), None);

    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    settle(&mut app, window);

    // Ctrl+7 navigates directly to the Agent Labeling jobs list.
    press(&mut app, window, "ctrl-7");
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::LabelJobs);

    // Command palette exposes Labeling section, New labeling job action, and seeded job.
    drive(&mut app, window, |this, w, cx| this.open_palette(w, cx));
    settle(&mut app, window);
    let shown = drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().shown.clone());
    assert!(shown.iter().any(|(g, t)| *g == "Go to" && t == "Labeling"), "{shown:?}");
    assert!(shown.iter().any(|(g, t)| *g == "Actions" && t == "New labeling job"), "{shown:?}");
    assert!(shown.iter().any(|(g, t)| *g == "Labeling" && t == "line-rejects"), "{shown:?}");
    drive(&mut app, window, |this, w, cx| this.close_palette(w, cx));

    // Navigating to a job sets parent_route to LabelJobs and Esc steps back cleanly.
    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.parent_route()), Some(Route::LabelJobs));
    press(&mut app, window, "escape");
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::LabelJobs);
}

#[test]
fn job_list_creation_folder_import_filtering_and_deletion() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1120., 720., Mode::Dark);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJobs, w, cx));
    settle(&mut app, window);

    let initial_jobs =
        drive(&mut app, window, |this, _, _| this.label.jobs.ready().expect("label jobs loaded").clone());
    assert!(!initial_jobs.is_empty(), "mock seeds a default labeling job");
    assert_eq!(filter_label_jobs(&initial_jobs, LabelJobFilter::All, "reject").len(), 1);
    assert_eq!(filter_label_jobs(&initial_jobs, LabelJobFilter::All, "circle").len(), 1);
    assert_eq!(filter_label_jobs(&initial_jobs, LabelJobFilter::Active, "").len(), 0);

    // Open New labeling job form, test validation error, then submit a valid job with source folder.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_label_job(w, cx);
        this.submit_new_label_job(cx);
    });
    settle(&mut app, window);
    assert!(
        drive(&mut app, window, |this, _, _| {
            this.label.new_job.as_ref().and_then(|f| f.error.as_ref()).is_some()
        }),
        "empty form shows validation error"
    );

    drive(&mut app, window, |this, _, cx| {
        let f = this.label.new_job.as_ref().unwrap();
        f.name.clone().update(cx, |i, cx| i.set_text("pcb-defects", cx));
        f.classes.clone().update(cx, |i, cx| i.set_text("solder_bridge, missing_pin", cx));
        f.path.clone().update(cx, |i, cx| i.set_text("D:/images/pcb-batch-1", cx));
        f.instructions.clone().update(cx, |i, cx| i.set_text("Tight boxes around defects.", cx));
        this.submit_new_label_job(cx);
    });
    settle(&mut app, window);

    let new_id = drive(&mut app, window, |this, _, _| match &this.route {
        Route::LabelJob(id) => id.clone(),
        other => panic!("expected navigation to new Route::LabelJob, got {other:?}"),
    });
    assert!(drive(&mut app, window, |this, _, _| this.label.new_job.is_none()));
    let imgs_before = drive(&mut app, window, |this, _, _| {
        this.label.images.get(&new_id).and_then(|l| l.ready()).expect("imported images").len()
    });
    assert!(imgs_before > 0, "source folder imported images on job creation");

    // Add a second folder to the new job.
    drive(&mut app, window, |this, w, cx| {
        this.open_add_label_folder(&new_id, w, cx);
        let f = this.label.add_folder.as_ref().unwrap();
        f.path.clone().update(cx, |i, cx| i.set_text("D:/images/pcb-batch-2", cx));
        this.submit_add_label_folder(cx);
    });
    settle(&mut app, window);
    let imgs_after = drive(&mut app, window, |this, _, _| {
        this.label.images.get(&new_id).and_then(|l| l.ready()).expect("reloaded images").len()
    });
    assert!(imgs_after > imgs_before, "Add folder grew image count ({imgs_after} <= {imgs_before})");

    // Delete the new job: navigates back to LabelJobs and marks it removed so history skips it.
    drive(&mut app, window, |this, _, cx| this.delete_label_job(&new_id, cx));
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::LabelJobs);
    assert!(drive(&mut app, window, |this, _, _| this.label.removed.contains(&new_id)));
}

#[test]
fn review_workspace_keyboard_mutations_conflicts_agent_and_export() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone(), 1280., 800., Mode::Light);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    // Filter to unreviewed labeled images and verify stepping + toggles.
    drive(&mut app, window, |this, _, cx| {
        this.label.img_filter = LabelImageFilter::ToReview;
        this.reconcile_selected_label_image("lb_4f10");
        cx.notify();
    });
    settle(&mut app, window);

    let (first_id, second_id) = drive(&mut app, window, |this, _, _| {
        let shown = this.current_filtered_label_images("lb_4f10");
        assert!(shown.len() >= 2, "need at least 2 unreviewed images");
        (shown[0].id.clone(), shown[1].id.clone())
    });
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.selected_image.clone()),
        Some(first_id.clone())
    );

    // Right / Left step through filtered images, B / L toggle overlays, +/-/0 zoom.
    press(&mut app, window, "right");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.selected_image.clone()),
        Some(second_id.clone())
    );
    press(&mut app, window, "left");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.selected_image.clone()),
        Some(first_id.clone())
    );
    press(&mut app, window, "b");
    assert!(!drive(&mut app, window, |this, _, _| this.label.show_boxes));
    press(&mut app, window, "b");
    press(&mut app, window, "l");
    assert!(!drive(&mut app, window, |this, _, _| this.label.show_labels));
    press(&mut app, window, "+");
    assert!(drive(&mut app, window, |this, _, _| this.label.zoom > 1.0));
    press(&mut app, window, "0");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.zoom), 1.0);

    // Select a box on an image with >= 1 box, reclassify it with key '2', then delete it with Delete.
    let (target_img, orig_boxes) = drive(&mut app, window, |this, _, _| {
        let all = this.label.images.get("lb_4f10").and_then(|l| l.ready()).unwrap();
        let im = all
            .iter()
            .find(|i| i.status == LabelImageStatus::Labeled && !i.reviewed && !i.boxes.is_empty())
            .expect("unreviewed image with boxes");
        (im.id.clone(), im.boxes.clone())
    });
    drive(&mut app, window, |this, _, cx| {
        this.select_label_image(target_img.clone(), cx);
        let rev = label_box_revision(&orig_boxes);
        this.label.selected_box = Some(SelectedBox { image_id: target_img.clone(), index: 0, revision: rev });
        cx.notify();
    });
    settle(&mut app, window);

    // Reclassify box 0 to class index 1 (key '2', or '1' if already 1).
    let next_cls = if orig_boxes[0].class == 1 { 0u32 } else { 1u32 };
    let key = (next_cls + 1).to_string();
    press(&mut app, window, &key);
    settle(&mut app, window);
    let after_recls = drive(&mut app, window, |this, _, _| {
        this.label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == target_img)
            .unwrap()
            .clone()
    });
    assert_eq!(after_recls.boxes[0].class, next_cls, "reclassify updated box class");
    assert_eq!(after_recls.source, Some(LabelSource::Manual));

    // Delete box 0 with Delete key (revision updated automatically after reclassify).
    press(&mut app, window, "delete");
    settle(&mut app, window);
    let after_del = drive(&mut app, window, |this, _, _| {
        this.label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == target_img)
            .unwrap()
            .clone()
    });
    assert_eq!(after_del.boxes.len(), orig_boxes.len() - 1, "Delete removed the selected box");

    // Bare Enter does NOT silently approve an image when focus is on a segmented filter or root.
    press(&mut app, window, "enter");
    settle(&mut app, window);
    let still_unreviewed = drive(&mut app, window, |this, _, _| {
        !this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == target_img)
            .unwrap()
            .reviewed
    });
    assert!(still_unreviewed, "bare Enter must not approve/mutate the active image");

    // Approve via 'a' marks reviewed and advances selection.
    press(&mut app, window, "a");
    settle(&mut app, window);
    let approved = drive(&mut app, window, |this, _, _| {
        this.label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == target_img)
            .unwrap()
            .reviewed
    });
    assert!(approved, "pressing 'a' marked image reviewed");

    // Stale revision conflict returns 409 label_boxes_changed on the mock backend.
    let conflict = api
        .update_label_image(
            "lb_4f10",
            &target_img,
            &UpdateLabelImage { boxes: Some(Vec::new()), reviewed: Some(true), expect: Some("stale".into()) },
        )
        .unwrap_err();
    assert_eq!(conflict.code, "label_boxes_changed");
    assert_eq!(conflict.status, Some(409));

    // Class filter narrows the image list to images carrying that class.
    let all_imgs = drive(&mut app, window, |this, _, _| {
        this.label.images.get("lb_4f10").and_then(|l| l.ready()).unwrap().clone()
    });
    let cls0 = filter_label_images(&all_imgs, LabelImageFilter::All, Some(0));
    assert!(cls0.iter().all(|im| im.boxes.iter().any(|b| b.class == 0)));

    // Start an agent labeling run on the unlabeled images; mock deterministically labels them all.
    let pending_before = drive(&mut app, window, |this, _, _| {
        let j = this.label_job_by_id("lb_4f10").unwrap();
        j.counts.pending + j.counts.failed
    });
    assert!(pending_before > 0, "seeded job has pending/failed images to label");
    drive(&mut app, window, |this, _, cx| this.start_label_run("lb_4f10", LabelScope::Unlabeled, cx));
    settle(&mut app, window);
    let after_run = drive(&mut app, window, |this, _, _| this.label_job_by_id("lb_4f10").unwrap().clone());
    assert_eq!(after_run.counts.pending, 0);
    assert_eq!(after_run.counts.failed, 0);
    assert_eq!(after_run.counts.labeled, after_run.counts.images);
    assert_eq!(after_run.run.as_ref().and_then(|r| r.outcome), Some(LabelOutcome::Completed));

    // Export / Register as Dataset updates store.datasets and attaches dataset_id.
    let ds_before = drive(&mut app, window, |this, _, _| this.store.datasets.ready().map_or(0, Vec::len));
    drive(&mut app, window, |this, w, cx| {
        let job = this.label_job_by_id("lb_4f10").unwrap().clone();
        this.open_export_label_dataset(&job, w, cx);
        let f = this.label.export_ds.as_ref().unwrap();
        f.name.clone().update(cx, |i, cx| i.set_text("line-rejects-v1", cx));
        this.submit_export_label_dataset(cx);
    });
    settle(&mut app, window);
    let (ds_after, linked_ds) = drive(&mut app, window, |this, _, _| {
        (
            this.store.datasets.ready().map_or(0, Vec::len),
            this.label_job_by_id("lb_4f10").and_then(|j| j.dataset_id.clone()),
        )
    });
    assert_eq!(ds_after, ds_before + 1, "exported dataset registered in store.datasets");
    assert!(linked_ds.is_some(), "job records its exported dataset_id");
}

#[test]
fn unsupported_and_failing_backends_render_cleanly() {
    // 1. Server without `agent_labeling` renders unsupported panel without fetching jobs.
    let api = Arc::new(MockBackend::new(4200, true).without_bench());
    let (mut app, window) = open_with(api, 1120., 720., Mode::Dark);
    drive(&mut app, window, |this, w, cx| {
        assert!(!this.store.features.iter().any(|f| f == FEATURE_AGENT_LABELING));
        this.navigate(Route::LabelJobs, w, cx);
    });
    settle(&mut app, window);
    assert!(drive(&mut app, window, |this, _, _| this.label_unsupported()));
    assert!(drive(&mut app, window, |this, _, _| matches!(this.label.jobs, Load::Idle)));

    // 2. Failing backend renders error state without panicking on both list and detail routes.
    let failing = Arc::new(MockBackend::new(4200, true).failing());
    let (mut fail_app, fail_win) = open_with(failing, 1120., 720., Mode::Light);
    drive(&mut fail_app, fail_win, |this, w, cx| this.navigate(Route::LabelJobs, w, cx));
    settle(&mut fail_app, fail_win);
    drive(&mut fail_app, fail_win, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut fail_app, fail_win);
    assert!(drive(&mut fail_app, fail_win, |this, _, _| this.store.system.error().is_some()));
}

#[test]
fn review_round1_regressions_revision_binding_ctrl_enter_filmstrip_and_monotonic_ids() {
    // 1. Filmstrip tile count never clips inside the review column at 1280×800 or 1120×720.
    for review_w in [668.0_f32, 708.0, 822.0, 852.0, 1000.0] {
        let n = crate::screens::label::filmstrip_max_tiles(review_w);
        let needed = n as f32 * 72.0 + n.saturating_sub(1) as f32 * 6.0;
        assert!(needed <= review_w - 30.0, "review_w={review_w}: {n} tiles need {needed}px");
    }

    // 2. Mock folder import after deleting a non-last image never produces duplicate image IDs.
    let api = Arc::new(MockBackend::new(4200, true));
    let before = api.label_images("lb_4f10", 0, 500).unwrap().images;
    assert!(before.len() >= 3);
    api.delete_label_image("lb_4f10", &before[0].id).unwrap();
    let added = api.import_label_images("lb_4f10", &ImportLabelImages { path: "D:/more".into() }).unwrap();
    assert!(!added.added.is_empty());
    let after = api.label_images("lb_4f10", 0, 500).unwrap().images;
    let unique: std::collections::HashSet<&str> = after.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(unique.len(), after.len(), "imported images must have strictly unique IDs");

    // 3. Stale box-revision selection refuses to delete or reclassify a shifted box array.
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);
    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);
    let target = drive(&mut app, window, |this, _, _| {
        this.label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| !i.boxes.is_empty())
            .unwrap()
            .clone()
    });
    let orig_len = target.boxes.len();
    drive(&mut app, window, |this, _, cx| {
        this.select_label_image(target.id.clone(), cx);
        this.label.selected_box =
            Some(SelectedBox { image_id: target.id.clone(), index: 0, revision: "stale-rev".into() });
        this.delete_selected_label_box("lb_4f10", &target.id, cx);
    });
    settle(&mut app, window);
    let (after_len, sel_cleared, err_code) = drive(&mut app, window, |this, _, _| {
        let len = this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == target.id)
            .unwrap()
            .boxes
            .len();
        (len, this.label.selected_box.is_none(), this.label.review_error.as_ref().map(|e| e.code.clone()))
    });
    assert_eq!(after_len, orig_len, "stale revision selection must not delete a box");
    assert!(sel_cleared, "stale box selection is invalidated");
    assert_eq!(err_code.as_deref(), Some("label_boxes_changed"));

    // 4. Ctrl+Enter closes an open dropdown menu without submitting an open labeling form behind it.
    drive(&mut app, window, |this, w, cx| {
        this.open_add_label_folder("lb_4f10", w, cx);
        this.label.add_folder.as_ref().unwrap().path.clone().update(cx, |i, cx| i.set_text("D:/extra", cx));
        this.menu_open = Some("label-agent");
        cx.notify();
    });
    app.frame(window.into());
    press(&mut app, window, "ctrl-enter");
    drive(&mut app, window, |this, _, _| {
        assert!(this.menu_open.is_none(), "Ctrl+Enter closed the open menu");
        let f = this.label.add_folder.as_ref().expect("Add folder form stays open, not submitted");
        assert!(!f.submitting, "Add folder form was not submitted behind the menu");
    });

    // 5. clear_label() advances generation so Retry isolates prior in-flight callbacks, and
    // merge_log_lines handles both overlapping tails and snapshots ahead of the SSE buffer.
    let line = |m: &str| crate::api::LogLine {
        t: "2026-01-01T00:00:00Z".into(),
        level: crate::api::LogLevel::Info,
        message: m.into(),
    };
    let mut snap = vec![line("a"), line("b"), line("c")];
    crate::label::merge_log_lines(&mut snap, vec![line("b"), line("c"), line("d")]);
    assert_eq!(snap.iter().map(|l| l.message.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c", "d"]);
    // Snapshot already ahead of SSE buffer: buffered `[b, c]` must not be duplicated at the end.
    crate::label::merge_log_lines(&mut snap, vec![line("b"), line("c")]);
    assert_eq!(snap.iter().map(|l| l.message.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c", "d"]);
    let (g0, g1) = drive(&mut app, window, |this, _, _| {
        let g0 = this.label.epoch;
        this.clear_label();
        (g0, this.label.epoch)
    });
    assert_eq!(g1, g0 + 1);
}
