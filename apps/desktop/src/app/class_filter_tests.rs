//! Datasets › class filter (issue #301): the filter belongs to the dataset it was set on and
//! applies there only while its class is one of that dataset's classes, whichever way the page was
//! reached (a launch straight onto it, the list and back) and whatever else reads the filtered list
//! (the image inspector opened from another dataset's run sample).

use super::select_driving::click_option;
use super::tests::{drive, open};
use super::{FoloApp, Route};
use folo_ui::{HeadlessApp, WindowHandle};

/// shapes-hard: five classes (hexagon is the fifth), and shapes: three.
const HARD: &str = "ds_3b91";
const SHAPES: &str = "ds_8f2c";

/// How many of dataset `id`'s images the page (and the inspector) shows, and how many it has.
fn shown(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, id: &'static str) -> (usize, usize) {
    drive(app, window, move |this, _, _| {
        let all = this.store.images.get(id).and_then(|l| l.ready()).map_or(0, |l| l.len());
        assert!(all > 0, "{id}'s images are in");
        (this.filtered_indices(id).len(), all)
    })
}

/// Draws the window until what its page asked for is in.
fn settle(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
}

fn go(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, route: Route) {
    drive(app, window, move |this, w, cx| this.navigate(route, w, cx));
    settle(app, window);
}

/// `--open dataset/<id>` sets the route before the first frame, without a navigation: a class
/// picked there stays through the list and back (it was cleared, as if another dataset had opened).
#[test]
fn a_class_filter_set_after_launching_on_its_dataset_stays_through_the_list() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| this.route = Route::Dataset(HARD.into()));
    settle(&mut app, window);
    click_option(&mut app, window, "ds-class", 5);
    let (filtered, all) = shown(&mut app, window, HARD);
    assert!(filtered < all, "hexagon picked: {filtered} of {all}");
    go(&mut app, window, Route::Datasets);
    go(&mut app, window, Route::Dataset(HARD.into()));
    assert_eq!(shown(&mut app, window, HARD), (filtered, all), "the same dataset keeps its filter");
}

/// A run's sample from another dataset opens the inspector on that dataset: the class filter set
/// on shapes-hard neither filters it (class 0 is a circle there too, but was picked on the other
/// dataset) nor is cleared by it (hexagon, past shapes' three classes, hid every image of it).
#[test]
fn a_run_sample_from_another_dataset_neither_takes_nor_clears_the_class_filter() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    drive(&mut app, window, |this, _, cx| this.ensure_dataset(SHAPES, cx));
    settle(&mut app, window);
    let (with_circle, without_circle) = drive(&mut app, window, |this, _, _| {
        let all = this.store.images.get(SHAPES).and_then(|l| l.ready()).unwrap();
        let has = |m: &&crate::api::ImageInfo| m.boxes.iter().any(|b| b.class == 0);
        (all.iter().find(has).map(|m| m.id.clone()), all.iter().any(|m| !has(&m)))
    });
    assert!(without_circle, "some image of shapes has no circle, so a circle filter would show");
    let with_circle = with_circle.expect("an image of shapes with a circle");
    for (option, class) in [(1, "circle"), (5, "hexagon")] {
        click_option(&mut app, window, "ds-class", option);
        let (hard_filtered, hard_all) = shown(&mut app, window, HARD);
        assert!(hard_filtered < hard_all, "{class} picked on shapes-hard");
        let image = with_circle.clone();
        drive(&mut app, window, move |this, w, cx| {
            this.open_sample(SHAPES.into(), image, "run_x".into(), w, cx)
        });
        settle(&mut app, window);
        let (filtered, all) = shown(&mut app, window, SHAPES);
        assert_eq!(filtered, all, "{class}: the inspector walks every image of shapes");
        drive(&mut app, window, |this, _, _| {
            let i = this.inspector.as_ref().expect("the inspector is open");
            assert!(i.want.is_none(), "{class}: landed");
            let at = this.filtered_image(SHAPES, i.index).map(|m| m.id.clone());
            assert_eq!(at.as_deref(), Some(with_circle.as_str()), "{class}: on its sample");
            this.inspector = None;
        });
        settle(&mut app, window);
        assert_eq!(shown(&mut app, window, HARD), (hard_filtered, hard_all), "{class}: still on shapes-hard");
    }
}

/// A class list read again with fewer classes (or another server's dataset under the same id)
/// leaves the stored class past them: the select shows "All classes", and so does the grid.
#[test]
fn a_class_past_a_refreshed_class_list_filters_nothing() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    click_option(&mut app, window, "ds-class", 5);
    let (filtered, all) = shown(&mut app, window, HARD);
    assert!(filtered < all, "hexagon picked: {filtered} of {all}");
    drive(&mut app, window, |this, _, cx| {
        let ds = this.store.dataset.get_mut(HARD).and_then(|l| l.ready_mut()).unwrap();
        ds.classes.truncate(3);
        let list = this.store.datasets.ready_mut().unwrap();
        list.iter_mut().find(|d| d.id == HARD).unwrap().classes.truncate(3);
        cx.notify();
    });
    settle(&mut app, window);
    assert_eq!(shown(&mut app, window, HARD), (all, all), "no hexagon any more: every image");
}
