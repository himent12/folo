//! Deterministic tests for interactive bounding-box editing in Agent Labeling:
//! click-drag drawing, coordinate mapping, relabeling (hotkeys + >9-class picker), delete,
//! undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y`), dirty-state indicators, unsaved switch guard,
//! 409 conflict recovery, `Esc` precedence, and narrow/standard viewport rendering.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput,
    ScrollDelta, ScrollWheelEvent, WindowBounds, WindowHandle, WindowOptions, point, px, size,
};

use crate::api::mock::MockBackend;
use crate::api::{BBox, Backend, LabelImageStatus, LabelSource, UpdateLabelImage, label_box_revision};
use crate::app::tests::drive;
use crate::app::{FoloApp, Route};
use crate::label::{EditTool, LabelImageFilter, PendingImageSwitch, SelectedBox};
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
fn interactive_draw_relabel_delete_undo_redo_discard_and_save_end_to_end() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone(), 1280., 800., Mode::Dark);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    let (img_id, orig_boxes) = drive(&mut app, window, |this, _, cx| {
        let all = this.label.images.get("lb_4f10").and_then(|l| l.ready()).unwrap();
        let im = all
            .iter()
            .find(|i| i.status == LabelImageStatus::Labeled && !i.boxes.is_empty())
            .expect("labeled image with boxes")
            .clone();
        this.select_label_image(im.id.clone(), cx);
        (im.id, im.boxes)
    });
    settle(&mut app, window);

    // 1. Toggle Draw tool with 'd', pick class 2 ('2'), and simulate click-drag box creation
    //    through the stage coordinate mapper.
    press(&mut app, window, "d");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.tool), EditTool::Draw);
    press(&mut app, window, "2");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.draw_class), 1);

    drive(&mut app, window, |this, _, cx| {
        let b = this.label.stage_view.get().expect("stage bounds recorded during paint");
        let p0 = point(b.origin.x + px(80.), b.origin.y + px(60.));
        let p1 = point(b.origin.x + px(210.), b.origin.y + px(175.));
        let im = this.ensure_label_editor_synced("lb_4f10", &img_id).unwrap();
        let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
        let img_p0 =
            crate::viz::to_image(b, dims, crate::viz::Fit::Contain, this.label.zoom, this.label.pan, p0);
        this.label.editor.begin_draw(img_p0, (f32::from(p0.x), f32::from(p0.y)), 1);
        this.update_label_box_draw(p1, cx);
        assert!(
            this.label.editor.preview_box(dims.0, dims.1).is_some(),
            "active drag produces live preview box"
        );
        this.finish_label_box_draw(cx);
    });
    settle(&mut app, window);

    // Newly drawn box is appended, selected, and marks the editor dirty.
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.editor.is_dirty_for("lb_4f10", &img_id));
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len() + 1);
        let drawn = this.label.editor.draft.last().unwrap();
        assert_eq!(drawn.class, 1);
        assert!(drawn.x2 > drawn.x1 && drawn.y2 > drawn.y1);
        let sel = this.label.selected_box.as_ref().expect("drawn box auto-selected");
        assert_eq!(sel.index, orig_boxes.len());
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "label-save"), "Save button enabled when dirty: {keys:?}");
        assert!(keys.iter().any(|k| k == "label-discard"), "Discard button enabled when dirty");
    });

    // 2. Relabel the selected drawn box to class 0 using '1'.
    press(&mut app, window, "1");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.last().unwrap().class, 0);
    });

    // 3. Delete the selected box via Backspace.
    press(&mut app, window, "backspace");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len());
        assert!(!this.label.editor.is_dirty_for("lb_4f10", &img_id), "back to matching base boxes");
        assert!(this.label.editor.can_undo());
    });

    // 4. Ctrl+Z undoes the delete (restoring the drawn class-0 box), Ctrl+Z again undoes relabel
    //    (back to class 1), Ctrl+Shift+Z redoes relabel (back to class 0), Ctrl+Y redoes delete.
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len() + 1);
        assert_eq!(this.label.editor.draft.last().unwrap().class, 0);
        assert!(this.label.editor.is_dirty_for("lb_4f10", &img_id));
    });
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.last().unwrap().class, 1);
    });
    press(&mut app, window, "ctrl-shift-z");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.last().unwrap().class, 0);
    });
    press(&mut app, window, "ctrl-y");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len());
    });
    // Undo back to having the drawn box, test Discard + undo Discard.
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len() + 1);
        this.discard_label_edits("lb_4f10", &img_id, cx);
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len());
        assert!(!this.label.editor.is_dirty_for("lb_4f10", &img_id));
    });
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), orig_boxes.len() + 1, "Discard itself is undoable");
    });

    // 5. Save via Ctrl+S persists the newly drawn box through PUT /api/label/jobs/{id}/images/{img}
    //    and preserves the undo stack so the user can still undo after saving.
    press(&mut app, window, "ctrl-s");
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.label.editor.is_dirty_for("lb_4f10", &img_id));
        assert!(this.label.editor.can_undo(), "undo stack preserved after save");
        let saved = this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == img_id)
            .unwrap();
        assert_eq!(saved.boxes.len(), orig_boxes.len() + 1);
        assert_eq!(saved.source, Some(LabelSource::Manual));
    });

    // Verify Esc precedence: exit Draw tool -> stays on LabelJob instead of navigating back.
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.tool), EditTool::Draw);
    press(&mut app, window, "escape");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.tool), EditTool::Select);
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::LabelJob("lb_4f10".into()));
}

#[test]
fn unsaved_switch_guard_conflict_recovery_and_viewports() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone(), 1120., 780., Mode::Light);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    let (first_id, second_id, orig_len) = drive(&mut app, window, |this, _, cx| {
        let all = this.label.images.get("lb_4f10").and_then(|l| l.ready()).unwrap();
        let im0 = all.iter().find(|i| !i.boxes.is_empty()).unwrap();
        let im1 = all.iter().find(|i| i.id != im0.id).unwrap();
        let (a, b, len) = (im0.id.clone(), im1.id.clone(), im0.boxes.len());
        this.select_label_image(a.clone(), cx);
        (a, b, len)
    });
    settle(&mut app, window);

    // Dirty the draft on first_id by deleting box 0.
    drive(&mut app, window, |this, _, cx| {
        let im = this.ensure_label_editor_synced("lb_4f10", &first_id).unwrap();
        let rev = label_box_revision(&im.boxes);
        this.label.selected_box = Some(SelectedBox { image_id: first_id.clone(), index: 0, revision: rev });
        this.delete_selected_label_box("lb_4f10", &first_id, cx);
    });
    settle(&mut app, window);

    // 1. Stepping right while dirty sets a pending_switch banner instead of losing edits.
    press(&mut app, window, "right");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.selected_image.as_deref(), Some(first_id.as_str()));
        assert_eq!(this.label.editor.pending_switch, Some(PendingImageSwitch::Step(1)));
    });
    // Esc dismisses the pending switch banner without leaving the page.
    press(&mut app, window, "escape");
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.editor.pending_switch.is_none());
        assert_eq!(this.route, Route::LabelJob("lb_4f10".into()));
    });

    // Requesting a filter switch while dirty guards it too; resolving with Discard switches cleanly.
    drive(&mut app, window, |this, _, cx| {
        this.request_label_img_filter("lb_4f10", LabelImageFilter::ToReview, cx);
        assert_eq!(
            this.label.editor.pending_switch,
            Some(PendingImageSwitch::SetFilter(LabelImageFilter::ToReview))
        );
        this.resolve_pending_label_switch("lb_4f10", false, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.label.img_filter, LabelImageFilter::ToReview);
        this.label.img_filter = LabelImageFilter::All;
        this.select_label_image(first_id.clone(), cx);
    });
    settle(&mut app, window);

    // 2. Simulate 409 label_boxes_changed conflict: mutate first_id directly on the backend while
    //    the user has unsaved local edits in Desktop, then Save.
    drive(&mut app, window, |this, _, cx| {
        let im = this.ensure_label_editor_synced("lb_4f10", &first_id).unwrap();
        let rev = label_box_revision(&im.boxes);
        this.label.selected_box = Some(SelectedBox { image_id: first_id.clone(), index: 0, revision: rev });
        this.delete_selected_label_box("lb_4f10", &first_id, cx);
        // External edit on the server changes the boxes under our feet.
        let mut server_boxes = im.boxes.clone();
        server_boxes.push(BBox { class: 0, x1: 5., y1: 5., x2: 45., y2: 45. });
        api.update_label_image(
            "lb_4f10",
            &first_id,
            &UpdateLabelImage { boxes: Some(server_boxes), reviewed: Some(false), expect: None },
        )
        .unwrap();
        // Now attempt to save our local edit (expecting the old base revision -> 409).
        this.save_label_edits("lb_4f10", &first_id, false, cx);
    });
    settle(&mut app, window);

    // User's draft must be preserved intact, conflict error shown, and an ordinary Save must STILL
    // refuse with 409 instead of silently overwriting remote boxes — only explicit Overwrite succeeds!
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.label.review_error.as_ref().map(|e| e.code.as_str()), Some("label_boxes_changed"));
        assert_eq!(
            this.label.editor.draft.len(),
            orig_len - 1,
            "409 conflict must preserve the user's draft boxes"
        );
        assert!(this.label.editor.changed_elsewhere, "server refresh flags changed_elsewhere");
        // Plain Ctrl+S / Save still sends the original expect_rev -> rejected with 409.
        this.save_label_edits("lb_4f10", &first_id, false, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(
            this.label.review_error.as_ref().map(|e| e.code.as_str()),
            Some("label_boxes_changed"),
            "plain Save must not silently overwrite concurrent remote edits"
        );
        // Explicit "Overwrite server boxes" uses refreshed base_revision and succeeds!
        this.save_label_edits_overwrite("lb_4f10", &first_id, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.label.review_error.is_none());
        assert!(!this.label.editor.is_dirty_for("lb_4f10", &first_id));
        let saved = this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == first_id)
            .unwrap();
        assert_eq!(saved.boxes.len(), orig_len - 1, "retry save persisted user's draft");
    });

    // 3. Save & continue flow via pending switch banner.
    drive(&mut app, window, |this, _, cx| {
        this.label.editor.add_box(BBox { class: 0, x1: 12., y1: 12., x2: 60., y2: 60. });
        this.request_select_label_image(second_id.clone(), cx);
        assert_eq!(
            this.label.editor.pending_switch,
            Some(PendingImageSwitch::SelectImage(second_id.clone()))
        );
        this.resolve_pending_label_switch("lb_4f10", true, cx);
    });
    settle(&mut app, window);
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.selected_image.clone()),
        Some(second_id),
        "Save & continue saved edits and completed the pending switch"
    );

    // 4. Layout smoke render at 1280x800 and 1120x780 in both Dark and Light modes with active edits.
    for (w, h, mode) in [(1280.0_f32, 800.0_f32, Mode::Dark), (1120.0, 780.0, Mode::Light)] {
        let (mut a, win) = open_with(Arc::new(MockBackend::new(4200, true)), w, h, mode);
        drive(&mut a, win, |this, win, cx| this.navigate(Route::LabelJob("lb_4f10".into()), win, cx));
        settle(&mut a, win);
        drive(&mut a, win, |this, _, cx| {
            this.label.editor.tool = EditTool::Draw;
            this.label.editor.add_box(BBox { class: 0, x1: 10., y1: 10., x2: 50., y2: 50. });
            cx.notify();
        });
        settle(&mut a, win);
    }
}

#[test]
fn route_exit_guard_sse_filter_eviction_dirty_delete_and_many_classes() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    // 1. Dirty draft guards navigate(), back(), and Esc; Discard executes deferred navigation on next frame.
    drive(&mut app, window, |this, w, cx| {
        this.label.editor.add_box(BBox { class: 0, x1: 8., y1: 8., x2: 48., y2: 48. });
        assert!(this.label.editor.is_dirty());
        this.navigate(Route::Overview, w, cx);
        assert_eq!(this.route, Route::LabelJob("lb_4f10".into()), "navigate blocked while dirty");
        assert_eq!(this.label.editor.pending_switch, Some(PendingImageSwitch::Navigate(Route::Overview)));
        this.back(w, cx);
        assert_eq!(this.route, Route::LabelJob("lb_4f10".into()), "back blocked while dirty");
        assert_eq!(this.label.editor.pending_switch, Some(PendingImageSwitch::Back));
        this.resolve_pending_label_switch("lb_4f10", false, cx);
    });
    settle(&mut app, window);
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.route.clone()),
        Route::Overview,
        "Discard executed deferred Back navigation cleanly"
    );

    // 2. Concurrent SSE update marking a dirty image reviewed under ToReview keeps the dirty draft open.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::LabelJob("lb_4f10".into()), w, cx);
    });
    settle(&mut app, window);
    let dirty_id = drive(&mut app, window, |this, _, cx| {
        this.label.img_filter = LabelImageFilter::ToReview;
        this.reconcile_selected_label_image("lb_4f10");
        let id = this.label.selected_image.clone().expect("unreviewed image");
        let _ = this.ensure_label_editor_synced("lb_4f10", &id);
        this.label.editor.add_box(BBox { class: 0, x1: 15., y1: 15., x2: 55., y2: 55. });
        // Simulate SSE marking this image reviewed from another client.
        if let Some(im) = this
            .label
            .images
            .get_mut("lb_4f10")
            .and_then(|l| l.ready_mut())
            .and_then(|list| list.iter_mut().find(|i| i.id == id))
        {
            im.reviewed = true;
        }
        this.reconcile_selected_label_image("lb_4f10");
        cx.notify();
        id
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(
            this.label.selected_image.as_deref(),
            Some(dirty_id.as_str()),
            "dirty image stays selected even when SSE moves it out of ToReview"
        );
        assert!(this.label.editor.is_dirty_for("lb_4f10", &dirty_id));
    });

    // 3. Removing a dirty image resets label.editor so filters never deadlock.
    drive(&mut app, window, |this, _, cx| {
        this.delete_label_image("lb_4f10", &dirty_id, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.label.editor.is_dirty(), "deleting a dirty image clears editor dirty state");
        assert!(this.label.editor.image_id.as_deref() != Some(dirty_id.as_str()));
    });

    // 4. Jobs with > 9 classes render the Select class picker dropdown instead of 9 wrapped chips.
    drive(&mut app, window, |this, _, cx| {
        this.label.img_filter = LabelImageFilter::All;
        if let Some(job) = this.label.jobs.ready_mut().and_then(|l| l.iter_mut().find(|j| j.id == "lb_4f10"))
        {
            job.classes = (0..12).map(|i| format!("defect_class_{i}")).collect();
        }
        this.reconcile_selected_label_image("lb_4f10");
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "label-class-picker"), "Select picker rendered for >9 classes");
        assert!(keys.iter().any(|k| k == "label-rm-img"), "Remove image footer remains mounted");
    });

    // 5. Conflict Load server boxes clears pre-conflict undo history, export blocks while dirty,
    //    should_close guards dirty draft, and externally deleted job never traps navigation.
    drive(&mut app, window, |this, w, cx| {
        let img_id = this.label.selected_image.clone().unwrap();
        let _ = this.ensure_label_editor_synced("lb_4f10", &img_id);
        this.label.editor.add_box(BBox { class: 0, x1: 6., y1: 6., x2: 40., y2: 40. });
        // Simulate remote edit -> changed_elsewhere = true, then Load server boxes clears undo.
        this.label.editor.changed_elsewhere = true;
        this.discard_label_edits("lb_4f10", &img_id, cx);
        assert!(!this.label.editor.can_undo(), "conflict discard clears incompatible undo history");

        // Re-dirty to check export guard, should_close guard, and external job deletion escape.
        this.label.editor.add_box(BBox { class: 0, x1: 6., y1: 6., x2: 40., y2: 40. });
        assert!(!this.should_close(w, cx), "should_close guards renderable dirty draft");
        assert_eq!(this.label.editor.pending_switch, Some(PendingImageSwitch::CloseWindow));
        this.label.editor.pending_switch = None;

        let job = this.label_job_by_id("lb_4f10").unwrap().clone();
        this.open_export_label_dataset(&job, w, cx);
        this.submit_export_label_dataset(cx);
        assert_eq!(
            this.label.export_ds.as_ref().and_then(|f| f.error.as_ref()).map(|e| e.code.as_str()),
            Some("unsaved_edits"),
            "Create dataset blocks while unsaved box edits exist"
        );
        this.close_export_label_dataset(w, cx);

        // Externally deleted job (removed from label.jobs without touching removed/editor) must not trap navigate.
        if let Some(jobs) = this.label.jobs.ready_mut() {
            jobs.retain(|j| j.id != "lb_4f10");
        }
        this.navigate(Route::LabelJobs, w, cx);
        assert_eq!(this.route, Route::LabelJobs, "missing job never traps user on Not found screen");
    });
}

#[test]
fn playtest_r11_wheel_scroll_stage_stability_and_conflict_banner() {
    // 1. Drawing the first box (clean -> dirty) must NOT shift the stage canvas vertically across
    //    standard, side-by-side breakpoint, and narrow viewports (1280, 1200, 1120, 960).
    for (w, h) in [(1280.0_f32, 800.0_f32), (1200.0, 800.0), (1120.0, 780.0), (960.0, 620.0)] {
        let (mut app, win) = open_with(Arc::new(MockBackend::new(4200, true)), w, h, Mode::Dark);
        drive(&mut app, win, |this, win, cx| this.navigate(Route::LabelJob("lb_4f10".into()), win, cx));
        settle(&mut app, win);

        let y_clean = drive(&mut app, win, |this, _, _| {
            assert!(!this.label.editor.is_dirty());
            f32::from(this.label.stage_view.get().expect("clean stage bounds").origin.y)
        });
        drive(&mut app, win, |this, _, cx| {
            this.label.editor.add_box(BBox { class: 0, x1: 12., y1: 12., x2: 52., y2: 52. });
            cx.notify();
        });
        settle(&mut app, win);
        let y_dirty = drive(&mut app, win, |this, _, _| {
            assert!(this.label.editor.is_dirty());
            f32::from(this.label.stage_view.get().expect("dirty stage bounds").origin.y)
        });
        assert!(
            (y_clean - y_dirty).abs() < 0.5,
            "stage jumped vertically at {w}x{h} on clean->dirty transition: {y_clean} -> {y_dirty}"
        );
    }

    // 2. Keyboard focus transfers cleanly between disabled and enabled Save / Discard handles when
    //    undo/redo flips dirty state, keeping Tab order intact.
    {
        let (mut app, win) = open_with(Arc::new(MockBackend::new(4200, true)), 1280., 800., Mode::Dark);
        drive(&mut app, win, |this, win, cx| this.navigate(Route::LabelJob("lb_4f10".into()), win, cx));
        settle(&mut app, win);
        drive(&mut app, win, |this, _, cx| {
            this.label.editor.add_box(BBox { class: 0, x1: 10., y1: 10., x2: 50., y2: 50. });
            cx.notify();
        });
        settle(&mut app, win);
        press(&mut app, win, "ctrl-z");
        drive(&mut app, win, |this, win, cx| {
            assert!(!this.label.editor.is_dirty());
            this.focus_key("label-save-disabled", win, cx);
            assert!(this.fh("label-save-disabled", cx).is_focused(win));
        });
        press(&mut app, win, "ctrl-shift-z");
        drive(&mut app, win, |this, win, cx| {
            assert!(this.label.editor.is_dirty());
            assert!(
                this.fh("label-save", cx).is_focused(win),
                "focus must move from label-save-disabled to label-save on redo"
            );
        });
        press(&mut app, win, "ctrl-z");
        drive(&mut app, win, |this, win, cx| {
            assert!(!this.label.editor.is_dirty());
            assert!(
                this.fh("label-save-disabled", cx).is_focused(win),
                "focus must move back to label-save-disabled on undo"
            );
        });
    }

    // 3. At 960x620 and 1x zoom:
    //    - NaN delta is ignored safely (zoom stays 1x)
    //    - Wheel down at 1x bubbles to scroll the page down
    //    - Wheel up zooms in without scrolling the page
    //    - Wheel down while zoomed returns zoom to 1x without scrolling the page
    let (mut app, win) = open_with(Arc::new(MockBackend::new(4200, true)), 960., 620., Mode::Dark);
    drive(&mut app, win, |this, win, cx| this.navigate(Route::LabelJob("lb_4f10".into()), win, cx));
    settle(&mut app, win);

    let (stage_center, y_before) = drive(&mut app, win, |this, _, _| {
        let b = this.label.stage_view.get().expect("stage bounds at 960x620");
        assert!((this.label.zoom - 1.0).abs() < 1e-4);
        (point(b.origin.x + b.size.width * 0.5, b.origin.y + b.size.height * 0.5), f32::from(b.origin.y))
    });
    app.simulate_input(
        win.into(),
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: stage_center,
            delta: ScrollDelta::Pixels(point(px(0.), px(f32::NAN))),
            ..Default::default()
        }),
    );
    settle(&mut app, win);
    assert!(
        drive(&mut app, win, |this, _, _| (this.label.zoom - 1.0).abs() < 1e-4),
        "NaN wheel delta leaves zoom at 1x"
    );

    app.simulate_input(
        win.into(),
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: stage_center,
            delta: ScrollDelta::Pixels(point(px(0.), px(-120.))),
            ..Default::default()
        }),
    );
    settle(&mut app, win);
    let y_after_down = drive(&mut app, win, |this, _, _| {
        assert!((this.label.zoom - 1.0).abs() < 1e-4, "wheel down at 1x keeps zoom at 1x");
        f32::from(this.label.stage_view.get().expect("stage bounds after wheel down").origin.y)
    });
    assert!(
        y_after_down < y_before - 1.0,
        "wheel down at 1x must bubble to scroll page (stage y {y_before} -> {y_after_down})"
    );

    // Wheel up zooms into the stage and consumes the event (page offset stays unchanged).
    let (stage_center_now, y_before_zoom) = drive(&mut app, win, |this, _, _| {
        let b = this.label.stage_view.get().unwrap();
        (point(b.origin.x + b.size.width * 0.5, b.origin.y + b.size.height * 0.5), f32::from(b.origin.y))
    });
    app.simulate_input(
        win.into(),
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: stage_center_now,
            delta: ScrollDelta::Pixels(point(px(0.), px(120.))),
            ..Default::default()
        }),
    );
    settle(&mut app, win);
    let (zoomed, y_after_zoom_in) = drive(&mut app, win, |this, _, _| {
        (this.label.zoom, f32::from(this.label.stage_view.get().unwrap().origin.y))
    });
    assert!(zoomed > 1.05, "wheel up over stage zooms in: {zoomed}");
    assert!(
        (y_after_zoom_in - y_before_zoom).abs() < 0.5,
        "wheel up zoom must not scroll the page: {y_before_zoom} -> {y_after_zoom_in}"
    );

    // Wheel down while zoomed returns to 1x and consumes the event (page offset stays unchanged).
    app.simulate_input(
        win.into(),
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: stage_center_now,
            delta: ScrollDelta::Pixels(point(px(0.), px(-240.))),
            ..Default::default()
        }),
    );
    settle(&mut app, win);
    let (zoomed_back, y_after_zoom_out) = drive(&mut app, win, |this, _, _| {
        (this.label.zoom, f32::from(this.label.stage_view.get().unwrap().origin.y))
    });
    assert!((zoomed_back - 1.0).abs() < 1e-4, "wheel down while zoomed returns to 1x: {zoomed_back}");
    assert!(
        (y_after_zoom_out - y_before_zoom).abs() < 0.5,
        "zoom-out wheel must not scroll the page: {y_before_zoom} -> {y_after_zoom_out}"
    );

    // 4. Conflict recovery banner wraps cleanly inside viewport bounds and its actions are
    //    focusable and actionable via Enter at 1280x800 and 960x620.
    for (w, h) in [(1280.0_f32, 800.0_f32), (960.0, 620.0)] {
        let (mut a, win) = open_with(Arc::new(MockBackend::new(4200, true)), w, h, Mode::Dark);
        drive(&mut a, win, |this, win, cx| this.navigate(Route::LabelJob("lb_4f10".into()), win, cx));
        settle(&mut a, win);
        let y_no_banner =
            drive(&mut a, win, |this, _, _| f32::from(this.label.stage_view.get().unwrap().origin.y));
        drive(&mut a, win, |this, _, cx| {
            this.label.editor.add_box(BBox { class: 0, x1: 8., y1: 8., x2: 40., y2: 40. });
            this.label.editor.changed_elsewhere = true;
            cx.notify();
        });
        settle(&mut a, win);
        let y_with_banner = drive(&mut a, win, |this, win, cx| {
            this.focus_key("label-load-server", win, cx);
            assert!(
                this.fh("label-load-server", cx).is_focused(win),
                "Load server button mounted at {w}x{h}"
            );
            assert!(
                this.focus.borrow().contains_key("label-overwrite-edits"),
                "Overwrite server button mounted at {w}x{h}"
            );
            f32::from(this.label.stage_view.get().unwrap().origin.y)
        });
        assert!(
            y_with_banner > y_no_banner + 30.0,
            "conflict banner renders cleanly above stage at {w}x{h}: {y_no_banner} -> {y_with_banner}"
        );
        let scene = a.scene(win.into());
        assert!(
            scene.primitives.iter().all(|p| match p {
                folo_ui::Primitive::Quad(q) => f64::from(q.bounds.right()) <= f64::from(w) + 2.0,
                _ => true,
            }),
            "conflict banner quads must not overflow viewport width {w}"
        );
        press(&mut a, win, "enter");
        settle(&mut a, win);
        drive(&mut a, win, |this, _, _| {
            assert!(!this.label.editor.changed_elsewhere, "Enter on Load server boxes clears conflict");
            assert!(!this.label.editor.is_dirty(), "Load server boxes reverts dirty draft");
        });
    }
}

#[test]
fn labeling_box_move_resize_undo_redo_and_hotkey_input_guard() {
    use crate::label::{ResizeHandle, TransformKind};
    use folo_ui::Focusable;

    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    let (img_id, orig_box) = drive(&mut app, window, |this, _, cx| {
        let all = this.label.images.get("lb_4f10").and_then(|l| l.ready()).unwrap();
        let im = all
            .iter()
            .find(|i| i.status == LabelImageStatus::Labeled && !i.boxes.is_empty())
            .expect("labeled image with boxes")
            .clone();
        this.select_label_image(im.id.clone(), cx);
        (im.id, im.boxes[0])
    });
    settle(&mut app, window);

    // 1. Move box 0 via stage transform helper -> marks dirty, updates selection revision, undoable!
    let moved_box = drive(&mut app, window, |this, _, cx| {
        let b = this.label.stage_view.get().expect("stage bounds");
        let im = this.ensure_label_editor_synced("lb_4f10", &img_id).unwrap();
        let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
        let p0 = point(b.origin.x + px(100.), b.origin.y + px(100.));
        let p1 = point(b.origin.x + px(130.), b.origin.y + px(120.));
        let img_p0 =
            crate::viz::to_image(b, dims, crate::viz::Fit::Contain, this.label.zoom, this.label.pan, p0);
        assert!(this.label.editor.begin_transform(
            0,
            img_p0,
            (f32::from(p0.x), f32::from(p0.y)),
            TransformKind::Move
        ));
        this.update_label_box_transform(p1, cx);
        assert!(this.label.editor.preview_transform(dims.0, dims.1).is_some());
        this.finish_label_box_transform(cx);
        assert!(this.label.editor.is_dirty_for("lb_4f10", &img_id));
        let sel = this.label.selected_box.as_ref().expect("moved box stays selected");
        assert_eq!(sel.revision, this.label.editor.draft_revision());
        this.label.editor.draft[0]
    });
    assert_ne!(moved_box, orig_box);

    // 2. Resize box 0 from SE handle -> second undo entry; Ctrl+Z x2 returns to clean base!
    let resized_box = drive(&mut app, window, |this, _, cx| {
        let b = this.label.stage_view.get().expect("stage bounds");
        let im = this.ensure_label_editor_synced("lb_4f10", &img_id).unwrap();
        let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
        let p0 = point(b.origin.x + px(130.), b.origin.y + px(120.));
        let p1 = point(b.origin.x + px(180.), b.origin.y + px(170.));
        let img_p0 =
            crate::viz::to_image(b, dims, crate::viz::Fit::Contain, this.label.zoom, this.label.pan, p0);
        assert!(this.label.editor.begin_transform(
            0,
            img_p0,
            (f32::from(p0.x), f32::from(p0.y)),
            TransformKind::Resize(ResizeHandle::Se),
        ));
        this.update_label_box_transform(p1, cx);
        this.finish_label_box_transform(cx);
        this.label.editor.draft[0]
    });
    assert_ne!(resized_box, moved_box);

    press(&mut app, window, "ctrl-z");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.draft[0]), moved_box);
    press(&mut app, window, "ctrl-z");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft[0], orig_box);
        assert!(!this.label.editor.is_dirty());
    });
    press(&mut app, window, "ctrl-y");
    assert_eq!(drive(&mut app, window, |this, _, _| this.label.editor.draft[0]), moved_box);

    // 3. When a TextInput is focused (e.g. inline Export form name input), numeric '1'-'9' hotkeys
    //    do NOT change draw_class or reclassify boxes.
    drive(&mut app, window, |this, w, cx| {
        this.label.editor.draw_class = 0;
        let job = this.label_job_by_id("lb_4f10").unwrap().clone();
        this.open_export_label_dataset(&job, w, cx);
        let fh = this.label.export_ds.as_ref().unwrap().name.read(cx).focus_handle(cx);
        w.focus(&fh);
    });
    settle(&mut app, window);
    press(&mut app, window, "2");
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.label.editor.draw_class, 0, "1-9 hotkeys ignored while TextInput is focused");
        this.close_export_label_dataset(w, cx);
    });
    settle(&mut app, window);
    press(&mut app, window, "2");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.editor.draw_class),
        1,
        "1-9 hotkeys work once TextInput is closed"
    );
}

#[test]
fn labeling_pointer_drag_delete_mid_drag_and_hidden_boxes_guard() {
    use folo_ui::{MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent};

    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api, 1280., 800., Mode::Dark);
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });

    drive(&mut app, window, |this, w, cx| this.navigate(Route::LabelJob("lb_4f10".into()), w, cx));
    settle(&mut app, window);

    // Pick a labeled image and seed it with 2 distinct boxes for mid-drag Delete verification.
    let (img_id, box_a, box_b) = drive(&mut app, window, |this, _, cx| {
        let ba = BBox { class: 0, x1: 60.0, y1: 60.0, x2: 180.0, y2: 180.0 };
        let bb = BBox { class: 1, x1: 300.0, y1: 120.0, x2: 440.0, y2: 260.0 };
        let im_id = match this.label.images.get_mut("lb_4f10") {
            Some(crate::store::Load::Ready(all)) => {
                let im =
                    all.iter_mut().find(|i| i.status == LabelImageStatus::Labeled).expect("labeled image");
                im.boxes = vec![ba, bb];
                im.id.clone()
            }
            _ => panic!("label images not loaded"),
        };
        this.select_label_image(im_id.clone(), cx);
        this.ensure_label_editor_synced("lb_4f10", &im_id);
        this.label.selected_box = None;
        cx.notify();
        (im_id, ba, bb)
    });
    settle(&mut app, window);

    // Compute screen-space center of box_a and SE handle of box_b inside label-stage.
    let (stage_b, pt_a, pt_a_drag, pt_b_se) = drive(&mut app, window, |this, _, _| {
        let b = this.label.stage_view.get().expect("stage bounds");
        let im = this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == img_id)
            .unwrap();
        let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
        let rect = crate::viz::image_rect(
            b,
            dims.0,
            dims.1,
            crate::viz::Fit::Contain,
            this.label.zoom,
            this.label.pan,
        );
        let sx = f32::from(rect.size.width) / dims.0;
        let sy = f32::from(rect.size.height) / dims.1;
        let a_center = point(
            rect.left() + px(((box_a.x1 + box_a.x2) * 0.5) * sx),
            rect.top() + px(((box_a.y1 + box_a.y2) * 0.5) * sy),
        );
        let a_drag = point(a_center.x + px(40.), a_center.y + px(30.));
        let b_se = point(rect.left() + px(box_b.x2 * sx), rect.top() + px(box_b.y2 * sy));
        (b, a_center, a_drag, b_se)
    });
    assert!(stage_b.contains(&pt_a));

    // 1. Real pointer sequence: MouseDown on box 0 -> MouseMove past drag threshold -> Delete -> MouseUp.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: pt_a,
            modifiers: folo_ui::Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: pt_a_drag,
            pressed_button: Some(MouseButton::Left),
            modifiers: folo_ui::Modifiers::default(),
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    assert!(
        drive(&mut app, window, |this, _, _| this.label.editor.transforming.is_some()),
        "dragging box 0 starts Move transform"
    );

    // Press Delete while mouse is still held: deletes box 0, cancels transform, shifts box_b to index 0.
    press(&mut app, window, "delete");
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: pt_a_drag,
            modifiers: folo_ui::Modifiers::default(),
            click_count: 1,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.label.editor.draft.len(), 1, "only box_b survives");
        assert_eq!(
            this.label.editor.draft[0], box_b,
            "mouse-up after mid-drag Delete must NOT overwrite surviving box_b"
        );
        assert!(this.label.editor.transforming.is_none());
    });

    // 2. Hidden boxes (show_boxes == false via 'B') must not expose resize handles or mutate invisible boxes.
    drive(&mut app, window, |this, _, cx| {
        this.label.selected_box = Some(crate::label::SelectedBox {
            image_id: img_id.clone(),
            index: 0,
            revision: this.label.editor.draft_revision(),
        });
        cx.notify();
    });
    settle(&mut app, window);
    press(&mut app, window, "b");
    assert!(!drive(&mut app, window, |this, _, _| this.label.show_boxes));

    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: pt_b_se,
            modifiers: folo_ui::Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    assert!(
        drive(&mut app, window, |this, _, _| this.label.editor.transforming.is_none()),
        "clicking where a hidden handle was must not start Resize while show_boxes is false"
    );
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: point(pt_b_se.x + px(30.), pt_b_se.y + px(30.)),
            modifiers: folo_ui::Modifiers::default(),
            click_count: 1,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.label.editor.draft[0]),
        box_b,
        "hidden box must remain unchanged"
    );

    // 3. Clean SSE sync_image cancels any in-flight transform.
    drive(&mut app, window, |this, _, _| {
        this.label.editor.base = vec![box_b];
        this.label.editor.draft = vec![box_b];
        assert!(this.label.editor.begin_transform(
            0,
            (350., 180.),
            (350., 180.),
            crate::label::TransformKind::Move
        ));
        let mut sse_img = this
            .label
            .images
            .get("lb_4f10")
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|i| i.id == img_id)
            .unwrap()
            .clone();
        sse_img.boxes = vec![box_a];
        this.label.editor.sync_image("lb_4f10", &sse_img, 3, false);
        assert!(this.label.editor.transforming.is_none(), "clean SSE sync cancels active transform");
        assert_eq!(this.label.editor.draft, vec![box_a]);
    });
}
