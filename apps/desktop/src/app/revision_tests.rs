//! Issue #346: a box moved in a label file (the dataset's stats as they were, its `revision` not)
//! shows where it is now once the dataset is read again: on a visit, and when the window is active
//! again. Never under an open inspector, and the grid keeps its place.

use std::sync::Arc;

use folo_ui::{HeadlessApp, WindowHandle, point, px};

use super::tests::{drive, open_with};
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::store::Load;

const ID: &str = "ds_8f2c";

/// Draws, letting every call answer between frames.
fn frames(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
}

/// The dataset's page, drawn, its list in.
fn on_its_page(api: Arc<MockBackend>) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open_with(api);
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Dataset(ID.into()), w, cx));
    frames(&mut app, window);
    (app, window)
}

/// Makes what the app holds from before an edit: another revision, the first box 5 px further left.
/// Gives where that box is now.
fn held_from_before(this: &mut FoloApp) -> f32 {
    let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
    old.revision = Some("before".into());
    this.store.dataset.insert(ID.into(), Load::Ready(old));
    let list = this.store.images.get_mut(ID).and_then(|l| l.ready_mut()).expect("listed");
    let b = list.iter_mut().find_map(|im| im.boxes.first_mut()).expect("a box");
    b.x1 -= 5.;
    b.x2 -= 5.;
    b.x1 + 5.
}

/// The first box's left edge in the list the app shows, and that list's length.
fn shown(this: &FoloApp) -> (Option<f32>, usize) {
    let list = this.store.images.get(ID).and_then(|l| l.ready()).expect("a list shown");
    (list.iter().find_map(|im| im.boxes.first()).map(|b| b.x1), list.len())
}

/// The user switches away and back between two frames: the app still sees it.
fn away_and_back(app: &mut HeadlessApp) {
    app.platform().set_active(false);
    app.platform().set_active(true);
    app.run_until_idle();
}

#[test]
fn back_to_the_window_reads_the_dataset_again_even_between_two_frames() {
    let (mut app, window) = on_its_page(Arc::new(MockBackend::new(4200, true)));
    let now = drive(&mut app, window, |this, _, _| held_from_before(this));
    away_and_back(&mut app);
    frames(&mut app, window);
    let (x1, n) = drive(&mut app, window, |this, _, _| shown(this));
    assert_eq!(x1, Some(now), "the grid draws the box where it is now");
    assert_eq!(n, 1000, "every image");
}

#[test]
fn a_read_again_waits_for_the_open_inspector_to_close() {
    let (mut app, window) = on_its_page(Arc::new(MockBackend::new(4200, true)));
    drive(&mut app, window, |this, w, cx| this.open_inspector(ID.into(), 700, w, cx));
    let now = drive(&mut app, window, |this, _, _| held_from_before(this));
    away_and_back(&mut app);
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(shown(this), (Some(now - 5.), 1000), "the list under the inspector is as it was");
        assert_eq!(this.inspector.as_ref().map(|i| i.index), Some(700), "on its image");
        assert!(this.ds_recheck.contains(ID), "the read waits");
    });
    drive(&mut app, window, |this, w, cx| this.close_inspector(w, cx));
    frames(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| shown(this)).0, Some(now), "read once it closed");
}

#[test]
fn a_read_on_its_way_when_the_inspector_opens_leaves_its_list_alone() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = on_its_page(api.clone());
    let now = drive(&mut app, window, |this, _, _| held_from_before(this));
    let read = api.hold("dataset");
    away_and_back(&mut app);
    app.frame(window.into());
    app.run_until_held(1);
    // (Plain updates while it is held: `drive` waits for every call.)
    app.update(|cx| window.update(cx, |this, w, cx| this.open_inspector(ID.into(), 700, w, cx)).unwrap());
    drop(read);
    app.run_until_idle();
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(shown(this), (Some(now - 5.), 1000), "the answer did not touch the list");
        assert_eq!(this.inspector.as_ref().map(|i| i.index), Some(700));
    });
    drive(&mut app, window, |this, w, cx| this.close_inspector(w, cx));
    frames(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| shown(this)).0, Some(now), "read again once it closed");
}

/// Review of #379: a run's sample inspector stayed on an empty scrim when a read of its dataset
/// dropped the list (nothing on the run page asks for it again).
#[test]
fn a_run_sample_inspector_keeps_its_image_through_a_read_again() {
    let (mut app, window) = on_its_page(Arc::new(MockBackend::new(4200, true)));
    let (now, image) = drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Run("run_52e8".into()), w, cx);
        // A read latched on an earlier page (a card's Retry, say), taken when the sample opens.
        this.ds_recheck.insert(ID.into());
        let image = this.store.images.get(ID).and_then(|l| l.ready()).expect("listed")[5].id.clone();
        (held_from_before(this), image)
    });
    drive(&mut app, window, |this, w, cx| this.open_sample(ID.into(), image, "run_52e8".into(), w, cx));
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let i = this.inspector.as_ref().expect("still open");
        assert_eq!((i.index, i.want.as_deref()), (5, None), "on its image, not a scrim");
        assert_eq!(shown(this), (Some(now - 5.), 1000), "its list kept");
    });
    drive(&mut app, window, |this, w, cx| this.close_inspector(w, cx));
    frames(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| shown(this)).0, Some(now), "read once it closed");
}

/// A list of more than one page read again after a revision-only change: the old one stays on screen
/// while the new one arrives, then takes its place, the grid scrolled where it was (dropped and
/// read again, the grid landed at the end of the first page).
#[test]
fn a_revision_only_change_keeps_the_grid_where_it_was() {
    let api = Arc::new(MockBackend::new(4200, true).with_more_val_images(ID, 25_000));
    let (mut app, window) = on_its_page(api.clone());
    let offset = |this: &FoloApp| this.grid_scroll.0.borrow().base_handle.offset();
    drive(&mut app, window, |this, _, _| {
        this.grid_scroll.0.borrow().base_handle.set_offset(point(px(0.), px(-800_000.)));
    });
    frames(&mut app, window);
    let now = drive(&mut app, window, |this, _, _| held_from_before(this));
    let (first, second) = (api.hold("images"), api.hold("images"));
    away_and_back(&mut app);
    app.frame(window.into());
    // (Plain updates while a call is held: `drive` waits for every call.)
    let asked = app.update(|cx| window.update(cx, |this, _, _| this.ds_reads.contains_key(ID)).unwrap());
    assert!(asked, "back to the window, the dataset is read again");
    app.run_until_held(1);
    drop(first);
    // Until the first page is in and the second asked for (bounded by frames, not a time).
    for _ in 0..20 {
        if second.is_held() {
            break;
        }
        app.frame(window.into());
        app.run_until_held(1);
    }
    assert!(second.is_held(), "the rest of the list is asked for");
    for _ in 0..3 {
        app.frame(window.into());
    }
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                assert_eq!(
                    shown(this),
                    (Some(now - 5.), 26_000),
                    "the old list stays while the new one loads"
                );
                assert!(!this.images_pending(ID), "complete as shown: no \"Loading…\"");
                assert_eq!(offset(this).y, px(-800_000.));
            })
            .unwrap()
    });
    drop(second);
    app.run_until_idle();
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(shown(this), (Some(now), 26_000), "the new list in");
        assert_eq!(offset(this).y, px(-800_000.), "the grid where it was");
    });
}
