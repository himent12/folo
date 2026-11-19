//! Issue #346: a box moved in a label file (the dataset's stats as they were, its `revision` not)
//! shows where it is now once the dataset is read again: on a visit, and when the window is active
//! again. Never under an open inspector: its dataset is read once it closes, whichever way it closes.

use std::sync::Arc;

use folo_ui::{HeadlessApp, WindowHandle};

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

/// Draws a few frames with at most one call held back (one more held one fails the test, as a stuck
/// call does).
fn frames_holding_one(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_held(1);
    }
}

/// Whether a read of the dataset is out (its answer not in yet).
fn reading(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> bool {
    app.update(|cx| window.update(cx, |this, _, _| this.ds_reads.contains_key(ID)).unwrap())
}

#[test]
fn a_read_again_waits_for_the_open_inspector_to_close() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = on_its_page(api.clone());
    drive(&mut app, window, |this, w, cx| this.open_inspector(ID.into(), 700, w, cx));
    let now = drive(&mut app, window, |this, _, _| held_from_before(this));
    // Every read of the dataset from here on is held: its ticket stays out until released.
    let (first, second) = (api.hold("dataset"), api.hold("dataset"));
    away_and_back(&mut app);
    frames_holding_one(&mut app, window);
    assert!(!reading(&mut app, window) && !first.is_held(), "no read while the inspector is open");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(shown(this), (Some(now - 5.), 1000), "the list under the inspector is as it was");
        assert_eq!(this.inspector.as_ref().map(|i| i.index), Some(700), "on its image");
        assert!(this.ds_recheck.contains(ID), "the read waits");
    });
    // (Plain updates while a call is held: `drive` waits for every call.)
    app.update(|cx| window.update(cx, |this, w, cx| this.close_inspector(w, cx)).unwrap());
    app.run_until_held(1);
    first.wait_held();
    frames_holding_one(&mut app, window);
    assert!(reading(&mut app, window) && !second.is_held(), "one read once it closed");
    drop((first, second));
    app.run_until_idle();
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
        // A read latched on an earlier page (a card's Retry, say): it waits for the inspector.
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
    assert!(drive(&mut app, window, |this, _, _| read_again(this)), "read once it closed");
}

/// A run's sample open on the dataset, its list from before an edit, and the window activated: the
/// read waits for the inspector. Gives where the edited box is now and the sample's image id.
fn sample_open_when_back(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (f32, String) {
    let image = drive(app, window, |this, w, cx| {
        this.navigate(Route::Run("run_52e8".into()), w, cx);
        this.store.images.get(ID).and_then(|l| l.ready()).expect("listed")[5].id.clone()
    });
    let again = image.clone();
    drive(app, window, |this, w, cx| this.open_sample(ID.into(), again, "run_52e8".into(), w, cx));
    frames(app, window);
    let now = drive(app, window, |this, _, _| held_from_before(this));
    away_and_back(app);
    frames(app, window);
    drive(app, window, |this, _, _| {
        assert!(this.ds_recheck.contains(ID), "back to the window, the inspected dataset is marked");
        assert_eq!(shown(this), (Some(now - 5.), 1000), "its list kept while open");
    });
    (now, image)
}

/// The record read again (the mock's has no revision; the one held from before had one).
fn read_again(this: &FoloApp) -> bool {
    !this.ds_recheck.contains(ID)
        && this.store.dataset.get(ID).and_then(|l| l.ready()).is_some_and(|d| d.revision.is_none())
}

/// Review of #379: on a run page nothing asked for the dataset again once a sample inspector closed,
/// and a sample opened again showed the box where it was.
#[test]
fn a_sample_inspector_closed_on_a_run_page_reads_its_dataset_again() {
    let (mut app, window) = on_its_page(Arc::new(MockBackend::new(4200, true)));
    let (now, image) = sample_open_when_back(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.close_inspector(w, cx));
    frames(&mut app, window);
    assert!(drive(&mut app, window, |this, _, _| read_again(this)), "read once it closed");
    drive(&mut app, window, |this, w, cx| this.open_sample(ID.into(), image, "run_52e8".into(), w, cx));
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        let i = this.inspector.as_ref().expect("open again");
        assert_eq!((i.index, i.want.as_deref()), (5, None), "on its image");
        assert_eq!(shown(this), (Some(now), 1000), "the box where it is now");
    });
}

/// Review of #379: the close question took a sample inspector's place without its held read.
#[test]
fn the_close_question_in_a_sample_inspector_s_place_reads_its_dataset_again() {
    let (mut app, window) = on_its_page(Arc::new(MockBackend::new(4200, true)));
    sample_open_when_back(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.should_close_with(None, w, cx));
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.inspector.is_none() && this.confirm_close.is_some(), "the question in its place");
        assert!(read_again(this), "its dataset read again");
    });
}
