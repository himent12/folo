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

/// Review of #379: the folder moved away while a read was out, an inspector opened before its
/// answer: the error replaced the record under it and nothing read the dataset once it closed.
#[test]
fn an_error_answer_under_an_inspector_waits_for_it_to_close() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = on_its_page(api.clone());
    api.break_dataset(ID, "path_not_found", "moved away");
    let read = api.hold("dataset");
    away_and_back(&mut app);
    app.frame(window.into());
    app.run_until_held(1);
    // (Plain updates while it is held: `drive` waits for every call.)
    app.update(|cx| window.update(cx, |this, w, cx| this.open_inspector(ID.into(), 700, w, cx)).unwrap());
    drop(read);
    app.run_until_idle();
    let (first, second) = (api.hold("dataset"), api.hold("dataset"));
    frames_holding_one(&mut app, window);
    assert!(!reading(&mut app, window) && !first.is_held(), "no read while the inspector is open");
    app.update(|cx| {
        window.update(cx, |this, _, _| {
            assert!(this.store.dataset.get(ID).is_some_and(|l| l.ready().is_some()), "the record stays");
            assert!(this.ds_recheck.contains(ID), "the read waits");
        })
    })
    .unwrap();
    app.update(|cx| window.update(cx, |this, w, cx| this.close_inspector(w, cx)).unwrap());
    app.run_until_held(1);
    first.wait_held();
    frames_holding_one(&mut app, window);
    assert!(reading(&mut app, window) && !second.is_held(), "one read once it closed");
    drop((first, second));
    app.run_until_idle();
    let failed =
        drive(&mut app, window, |this, _, _| this.store.dataset.get(ID).and_then(|l| l.error()).cloned());
    assert_eq!(failed.map(|e| e.code), Some("path_not_found".into()), "its answer shown once closed");
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
    drive(&mut app, window, |this, w, cx| {
        this.open_sample(ID.into(), image, "run_52e8".into(), w, cx);
        // Installed before it asks for the dataset: the latched read still waits, none went out.
        assert!(this.ds_recheck.contains(ID) && !this.ds_reads.contains_key(ID), "no read under it");
    });
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

/// #382: returning to the window while on Overview or Datasets refreshes loaded dataset cards
/// (updating their cached record and invalidating stale preview boxes) without duplicating reads
/// that are already in flight.
#[test]
fn back_to_the_window_on_overview_and_datasets_rechecks_loaded_cards() {
    for route in [Route::Overview, Route::Datasets] {
        let api = Arc::new(MockBackend::new(4200, true));
        let (mut app, window) = open_with(api.clone());
        drive(&mut app, window, |this, w, cx| this.navigate(route.clone(), w, cx));
        frames(&mut app, window);

        // Simulate an external edit by marking the cached dataset record with an older revision and
        // mutating a cached preview box.
        let now = drive(&mut app, window, |this, _, _| {
            let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("card read");
            old.revision = Some("before".into());
            this.store.dataset.insert(ID.into(), Load::Ready(old));
            let previews =
                this.store.previews.get_mut(ID).and_then(|l| l.ready_mut()).expect("card previews");
            let b = previews.iter_mut().find_map(|im| im.boxes.first_mut()).expect("preview box");
            b.x1 -= 5.;
            b.x2 -= 5.;
            b.x1 + 5.
        });

        away_and_back(&mut app);
        frames(&mut app, window);

        drive(&mut app, window, |this, _, _| {
            assert!(read_again(this), "{route:?}: dataset record refreshed on window activation");
            let previews =
                this.store.previews.get(ID).and_then(|l| l.ready()).expect("card previews reloaded");
            let x1 = previews.iter().find_map(|im| im.boxes.first()).map(|b| b.x1);
            assert_eq!(x1, Some(now), "{route:?}: card preview boxes refreshed on revision change");
        });
    }
}

/// Pumps frames until the expected `images` call count reaches `want_image_calls`, then blocks
/// cleanly via `wait_held()` (which has its own bounded timeout and does not depend on worker
/// thread scheduling speed).
fn wait_for_image_page_hold(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    api: &MockBackend,
    want_image_calls: usize,
    wait_held: impl FnOnce(),
    label: &str,
) {
    for _ in 0..64 {
        if api.image_lists().len() >= want_image_calls {
            wait_held();
            return;
        }
        app.frame(window.into());
        app.run_until_held(1);
    }
    panic!(
        "expected held request '{label}' to be dispatched (image_lists len={}, wanted {want_image_calls})",
        api.image_lists().len()
    );
}

/// #382: refreshing a large multi-page dataset after an external edit preserves the user's scroll
/// place (anchored by `(file, split)`) across paged reloads, including while the later batch is
/// delayed, when triggered right after closing an open inspector on a non-first-column tile, and
/// when a non-default sort reorders rows as later pages arrive.
#[test]
fn large_dataset_grid_preserves_scroll_position_across_paged_revision_refresh() {
    let api = Arc::new(MockBackend::new(4200, true).with_paged_splits(ID, 10_000, 60));
    let (mut app, window) = on_its_page(api.clone());

    // Switch to the Val split (which arrives entirely on the second image page, offset 10_000+)
    // and scroll partway down the grid.
    drive(&mut app, window, |this, _, cx| {
        this.ds_split = Some(crate::api::Split::Val);
        cx.notify();
    });
    frames(&mut app, window);

    let (saved_y, expected_file, expected_row) = drive(&mut app, window, |this, _, cx| {
        let cols = this.grid_cols();
        let item_h = f32::from(this.grid_scroll.0.borrow().last_item_height.expect("measured row height"));
        // Scroll to row 6 + 37 px intra-row offset.
        let want_row = 6usize;
        let want_y = want_row as f32 * item_h + 37.;
        this.grid_scroll
            .0
            .borrow()
            .base_handle
            .set_offset(folo_ui::point(folo_ui::px(0.), -folo_ui::px(want_y)));
        cx.notify();
        let indices = this.filtered_indices(ID);
        let all = this.store.images.get(ID).and_then(|l| l.ready()).unwrap();
        let meta = &all[indices[want_row * cols]];
        (want_y, meta.file.clone(), want_row)
    });
    app.frame(window.into());

    // Simulate external label edit on disk (older cached revision), delay the second image batch,
    // and reactivate the window.
    drive(&mut app, window, |this, _, _| {
        let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
        old.revision = Some("before".into());
        this.store.dataset.insert(ID.into(), Load::Ready(old));
    });
    // Hold page 0 and page 1 in order so we can observe the intermediate state after page 0 lands
    // and before page 1 (containing all Val images) arrives.
    let base_calls = api.image_lists().len();
    let page0 = api.hold("images");
    let page1 = api.hold("images");
    away_and_back(&mut app);
    wait_for_image_page_hold(&mut app, window, &api, base_calls + 1, || page0.wait_held(), "page 0");
    drop(page0);
    wait_for_image_page_hold(&mut app, window, &api, base_calls + 2, || page1.wait_held(), "page 1");
    app.frame(window.into());

    // A second window activation while page 1 is still pending returns an unchanged dataset record
    // and must NOT discard the active reload's scroll anchor.
    app.platform().set_active(false);
    app.platform().set_active(true);
    app.run_until_held(1);
    frames_holding_one(&mut app, window);

    // Anchor must still be waiting while page 1 (containing Val images) is in flight.
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                let anchor =
                    this.ds_scroll_anchor.as_ref().expect("scroll anchor survives second activation");
                assert_eq!(anchor.file, expected_file);
                assert_eq!(anchor.split, crate::api::Split::Val);
                assert!(!anchor.pending_recheck, "anchor is committed to active reload");
            })
            .unwrap()
    });

    // Release page 1 and let the grid repaint.
    drop(page1);
    app.run_until_idle();
    frames(&mut app, window);

    drive(&mut app, window, |this, _, _| {
        assert!(this.ds_scroll_anchor.is_none(), "scroll anchor consumed once target batch arrived");
        let got_y = f32::from(-this.grid_scroll.0.borrow().base_handle.offset().y);
        let item_h = f32::from(this.grid_scroll.0.borrow().last_item_height.expect("row height"));
        let got_row = (got_y / item_h).floor() as usize;
        assert_eq!(got_row, expected_row, "grid restored to the same row (got_y={got_y}, saved_y={saved_y})");
        assert!(
            (got_y - saved_y).abs() <= 2.0,
            "grid preserved intra-row scroll offset (got_y={got_y}, saved_y={saved_y})"
        );
    });

    // Repeat through an open inspector on a non-first-column Val tile (index 37, column 2):
    // closing the inspector anchors that exact file and focuses its refreshed tile once page 1 lands.
    let inspected_idx = 37usize;
    let (inspected_file, expected_tile_id) = drive(&mut app, window, |this, w, cx| {
        this.open_inspector(ID.into(), inspected_idx, w, cx);
        let meta = this.filtered_image(ID, inspected_idx).expect("inspected tile").clone();
        (meta.file, meta.id)
    });
    drive(&mut app, window, |this, _, _| {
        let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
        old.revision = Some("before-inspector".into());
        this.store.dataset.insert(ID.into(), Load::Ready(old));
    });
    away_and_back(&mut app);
    frames(&mut app, window);
    let insp_base = api.image_lists().len();
    let p0 = api.hold("images");
    let p1 = api.hold("images");
    app.update(|cx| window.update(cx, |this, w, cx| this.close_inspector(w, cx)).unwrap());
    wait_for_image_page_hold(&mut app, window, &api, insp_base + 1, || p0.wait_held(), "inspector page 0");
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                let anchor = this.ds_scroll_anchor.as_ref().expect("exact inspected file anchored");
                assert_eq!(anchor.file, inspected_file, "anchors exact inspected column, not col 0");
                assert!(anchor.focus_from.is_some(), "records focus handle for conditional restoration");
            })
            .unwrap()
    });
    drop(p0);
    wait_for_image_page_hold(&mut app, window, &api, insp_base + 2, || p1.wait_held(), "inspector page 1");
    drop(p1);
    app.run_until_idle();
    frames(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        assert!(this.ds_scroll_anchor.is_none(), "inspector-close scroll anchor consumed");
        let got_y = f32::from(-this.grid_scroll.0.borrow().base_handle.offset().y);
        assert!(got_y > 100., "grid stayed scrolled near the inspected val image (got_y={got_y})");
        let tile_fh = this.fh(format!("tile-{expected_tile_id}"), cx);
        assert!(tile_fh.is_focused(w), "focus restored to refreshed inspected tile");
    });

    // Opening the command palette while an inspector-close reload is pending must NOT lose focus
    // when the target page arrives.
    drive(&mut app, window, |this, w, cx| this.open_inspector(ID.into(), inspected_idx, w, cx));
    drive(&mut app, window, |this, _, _| {
        let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
        old.revision = Some("before-palette".into());
        this.store.dataset.insert(ID.into(), Load::Ready(old));
    });
    away_and_back(&mut app);
    frames(&mut app, window);
    let pal_base = api.image_lists().len();
    let pal_p0 = api.hold("images");
    let pal_p1 = api.hold("images");
    app.update(|cx| window.update(cx, |this, w, cx| this.close_inspector(w, cx)).unwrap());
    wait_for_image_page_hold(&mut app, window, &api, pal_base + 1, || pal_p0.wait_held(), "palette page 0");
    drop(pal_p0);
    wait_for_image_page_hold(&mut app, window, &api, pal_base + 2, || pal_p1.wait_held(), "palette page 1");
    app.update(|cx| window.update(cx, |this, w, cx| this.open_palette(w, cx)).unwrap());
    drop(pal_p1);
    app.run_until_idle();
    frames(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        use folo_ui::Focusable;
        let palette = this.palette.as_ref().expect("palette remains open");
        assert!(
            palette.input.read(cx).focus_handle(cx).is_focused(w),
            "arriving page must not steal focus from open command palette"
        );
    });
}

/// #382 review: typing a filename query while a later page is held cancels any pending scroll anchor
/// so arriving pages do not yank the grid away from row 0.
#[test]
fn typing_filename_search_during_paged_reload_clears_scroll_anchor() {
    let api = Arc::new(MockBackend::new(4200, true).with_paged_splits(ID, 10_000, 60));
    let (mut app, window) = on_its_page(api.clone());
    drive(&mut app, window, |this, _, cx| {
        this.ds_split = Some(crate::api::Split::Val);
        cx.notify();
    });
    frames(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        let item_h = f32::from(this.grid_scroll.0.borrow().last_item_height.expect("row height"));
        this.grid_scroll
            .0
            .borrow()
            .base_handle
            .set_offset(folo_ui::point(folo_ui::px(0.), -folo_ui::px(6. * item_h)));
        let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
        old.revision = Some("before".into());
        this.store.dataset.insert(ID.into(), Load::Ready(old));
        cx.notify();
    });
    app.frame(window.into());

    let base_calls = api.image_lists().len();
    let page0 = api.hold("images");
    let page1 = api.hold("images");
    away_and_back(&mut app);
    wait_for_image_page_hold(&mut app, window, &api, base_calls + 1, || page0.wait_held(), "search page 0");
    drop(page0);
    wait_for_image_page_hold(&mut app, window, &api, base_calls + 2, || page1.wait_held(), "search page 1");
    app.frame(window.into());

    // Simulate user typing into `ds_img_search` (matching the old Val anchor) while page 1 is held.
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                assert!(this.ds_scroll_anchor.is_some(), "anchor armed before search edit");
                this.ds_img_search.update(cx, |inp, cx| inp.set_text("val/", cx));
            })
            .unwrap()
    });
    app.run_until_held(1);
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                assert!(this.ds_scroll_anchor.is_none(), "typing a filename query cancels scroll anchor");
            })
            .unwrap()
    });

    drop(page1);
    app.run_until_idle();
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(f32::from(-this.grid_scroll.0.borrow().base_handle.offset().y), 0., "stays at row 0");
    });
}

/// #382 review: under a non-default sort (NameDesc, where later pages sort ahead of page 0), the
/// scroll anchor stays active through the final batch so the grid stays on the same file after the
/// final re-sort; and a retryable later-page failure keeps the anchor until Retry succeeds.
#[test]
fn non_default_sort_and_retryable_later_page_preserve_scroll_anchor() {
    let api = Arc::new(MockBackend::new(4200, true).with_paged_splits(ID, 10_000, 60));
    let (mut app, window) = on_its_page(api.clone());
    // Sort Z->A on All images: all 60 `images/val/*` files sort BEFORE `images/train/*`.
    drive(&mut app, window, |this, _, cx| {
        this.ds_img_dataset = Some(ID.into());
        this.ds_img_sort = crate::screens::datasets::ImageSort::NameDesc;
        cx.notify();
    });
    frames(&mut app, window);

    // Scroll to row 20 (which lands on a `train` image that ALSO appears in page 0, but shifts
    // down by 12 rows once the 60 `val` images on page 1 arrive and sort ahead of it!).
    let (anchor_file, expected_row) = drive(&mut app, window, |this, _, cx| {
        let cols = this.grid_cols();
        let item_h = f32::from(this.grid_scroll.0.borrow().last_item_height.expect("row height"));
        let want_row = 20usize;
        this.grid_scroll
            .0
            .borrow()
            .base_handle
            .set_offset(folo_ui::point(folo_ui::px(0.), -folo_ui::px(want_row as f32 * item_h)));
        cx.notify();
        let indices = this.filtered_indices(ID);
        let all = this.store.images.get(ID).and_then(|l| l.ready()).unwrap();
        let meta = &all[indices[want_row * cols]];
        assert_eq!(meta.split, crate::api::Split::Train, "anchor is in page 0");
        (meta.file.clone(), want_row)
    });
    app.frame(window.into());

    // Trigger revision reload where page 1 fails permanently first (inline Retry), then succeeds.
    api.fail_images_from_with(
        super::data::IMAGE_PAGE,
        crate::api::ApiError::new("permission_denied", "temporary lock"),
        false,
    );
    drive(&mut app, window, |this, _, _| {
        let mut old = this.store.dataset.get(ID).and_then(|l| l.ready()).cloned().expect("read");
        old.revision = Some("before".into());
        this.store.dataset.insert(ID.into(), Load::Ready(old));
    });
    away_and_back(&mut app);
    frames(&mut app, window);

    // Page 0 loaded 10_000 train images and page 1 failed: under NameDesc + failed_page, anchor
    // must NOT be dropped yet.
    drive(&mut app, window, |this, _, _| {
        assert!(this.failed_image_page(ID).is_some(), "page 1 recorded as failed");
        let a = this.ds_scroll_anchor.as_ref().expect("anchor kept across failed later page");
        assert_eq!(a.file, anchor_file);
    });

    // Clear the error and click Retry: page 1 lands, val images sort ahead of train, and the grid
    // re-anchors back to `expected_row` (row 20) where `anchor_file` now lives.
    api.clear_fail_images();
    drive(&mut app, window, |this, _, cx| this.retry_failed_image_page(ID, cx));
    frames(&mut app, window);

    drive(&mut app, window, |this, _, _| {
        assert!(this.ds_scroll_anchor.is_none(), "anchor consumed once final sorted page landed");
        let cols = this.grid_cols();
        let item_h = f32::from(this.grid_scroll.0.borrow().last_item_height.expect("row height"));
        let got_row =
            (f32::from(-this.grid_scroll.0.borrow().base_handle.offset().y) / item_h).round() as usize;
        assert_eq!(got_row, expected_row, "re-anchored to exact sorted row after page 1 arrived");
        let indices = this.filtered_indices(ID);
        let all = this.store.images.get(ID).and_then(|l| l.ready()).unwrap();
        assert_eq!(all[indices[got_row * cols]].file, anchor_file);
    });

    // Completed reload with zero matching images must clear any stale anchor during render (no leak).
    // Settle the empty-matching query first so the subscription fires before we arm the anchor.
    drive(&mut app, window, |this, _, cx| {
        this.ds_img_search.update(cx, |inp, cx| inp.set_text("no-such-file-xyz", cx));
    });
    frames(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.filtered_indices(ID).len(), 0, "zero matches settled before arming anchor");
        this.ds_scroll_anchor = Some(super::data::DatasetScrollAnchor {
            dataset: ID.into(),
            file: "deleted.jpg".into(),
            split: crate::api::Split::Val,
            intra_row: 0.,
            strategy: None,
            pending_recheck: false,
            focus_from: None,
        });
        cx.notify();
    });
    frames(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.ds_scroll_anchor.is_none(), "empty completed grid clears stale scroll anchor");
    });
}
