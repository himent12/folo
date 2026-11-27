//! Deterministic tests for interactive bounding-box editing in Agent Labeling:
//! click-drag drawing, coordinate mapping, relabeling (hotkeys + >9-class picker), delete,
//! undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y`), dirty-state indicators, unsaved switch guard,
//! 409 conflict recovery, `Esc` precedence, and narrow/standard viewport rendering.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput,
    WindowBounds, WindowHandle, WindowOptions, point, px, size,
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

    // User's draft must be preserved intact, conflict error shown, and server base refreshed so a
    // retry Save succeeds cleanly.
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.label.review_error.as_ref().map(|e| e.code.as_str()), Some("label_boxes_changed"));
        assert_eq!(
            this.label.editor.draft.len(),
            orig_len - 1,
            "409 conflict must preserve the user's draft boxes"
        );
        assert!(this.label.editor.changed_elsewhere, "server refresh flags changed_elsewhere");
        // Retrying Save now sends the refreshed base revision and succeeds!
        this.save_label_edits("lb_4f10", &first_id, false, cx);
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
