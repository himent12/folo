//! Datasets › class filter (issue #301): the filter belongs to the dataset it was set on and
//! applies there only while its class is one of that dataset's classes, whichever way the page was
//! reached (a launch straight onto it, the list and back) and whatever else reads the filtered list
//! (the image inspector opened from another dataset's run sample).

use super::select_driving::{click_option, press};
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

/// Review round 1: Statistics' "Show them" (Empty images) and "All classes" picked by pointer on
/// another dataset each cleared shapes-hard's class filter; and the empty-images filter, kept for
/// every dataset, met the class filter back on shapes-hard. Each stays with its own dataset.
#[test]
fn another_datasets_empty_images_and_all_classes_leave_the_class_filter_alone() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    click_option(&mut app, window, "ds-class", 5);
    let hard = shown(&mut app, window, HARD);
    assert!(hard.0 < hard.1, "hexagon picked: {hard:?}");
    let ids: Vec<String> = drive(&mut app, window, |this, _, cx| {
        let ids: Vec<String> = this.store.datasets.ready().unwrap().iter().map(|d| d.id.clone()).collect();
        ids.iter().for_each(|id| this.ensure_dataset(id, cx));
        ids
    });
    settle(&mut app, window);
    let other: &'static str = drive(&mut app, window, move |this, _, _| {
        let has_empty = |id: &&String| {
            let images = this.store.images.get(id.as_str()).and_then(|l| l.ready());
            *id != HARD && images.is_some_and(|l| l.iter().any(|m| m.boxes.is_empty()))
        };
        Box::leak(
            ids.iter().find(has_empty).expect("a demo dataset with empty images").clone().into_boxed_str(),
        )
    });
    go(&mut app, window, Route::Dataset(other.into()));
    click_option(&mut app, window, "ds-class", 0);
    drive(&mut app, window, |this, _, cx| {
        this.ds_tab = super::DsTab::Statistics;
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("ds-empty-tile", cx)));
    app.frame(window.into());
    press(&mut app, window, "enter");
    settle(&mut app, window);
    let (empty, all) = shown(&mut app, window, other);
    assert!(0 < empty && empty < all, "Show them: only {other}'s empty images ({empty} of {all})");
    go(&mut app, window, Route::Dataset(HARD.into()));
    assert_eq!(shown(&mut app, window, HARD), hard, "shapes-hard keeps hexagon, and only that");
    drive(&mut app, window, |this, _, _| assert!(!this.empty_only(HARD)));
}

fn found_local(_: std::time::Duration) -> Result<crate::api::http::Spawned, String> {
    Ok(crate::api::http::Spawned::Running("http://127.0.0.1:4100".into()))
}

fn demo(_: &str) -> std::sync::Arc<dyn crate::api::Backend> {
    std::sync::Arc::new(crate::api::mock::MockBackend::new(4200, true))
}

/// Review round 1: dataset ids are unique per data dir, so the same id on another server is
/// another dataset (`circle` there may be `person`): a switch drops the filters. A recovery
/// reaches the lost server's data dir again, so its datasets keep theirs.
#[test]
fn another_server_drops_the_dataset_filters_and_a_recovery_keeps_them() {
    use super::{Connector, ServerMode};
    let (mut app, window) = open();
    let filters = |this: &FoloApp| (this.ds_class.clone(), this.ds_empty_of.clone());
    let set = (Some((HARD.to_string(), 4)), Some(SHAPES.to_string()));
    let kept = set.clone();
    // The local server stops answering; the data dir's server answers at another address.
    drive(&mut app, window, move |this, _, cx| {
        (this.ds_class, this.ds_empty_of) = kept;
        this.connector = Connector { probe: |_| false, start: found_local, backend: demo };
        this.server_mode = ServerMode::Local;
        this.local_url = Some("http://127.0.0.1:4000".into());
        this.store.api = std::sync::Arc::new(crate::api::mock::MockBackend::new(4200, true).failing());
        this.store.system = crate::store::Load::Idle;
        this.ensure_core(cx);
    });
    let kept = set.clone();
    drive(&mut app, window, move |this, _, _| {
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"), "recovered");
        assert!(this.backend_gen > 0, "onto a new backend");
        assert_eq!(filters(this), kept, "the same data dir: the filters stay");
    });
    // Another server, typed in Settings.
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector { probe: |_| true, start: found_local, backend: demo };
        this.server_url.update(cx, |i, cx| i.set_text("http://192.168.9.9:7878", cx));
        this.switch_source(ServerMode::Local, cx);
    });
    drive(&mut app, window, move |this, _, _| {
        assert_eq!(this.connected_url.as_deref(), Some("http://192.168.9.9:7878"));
        assert_eq!(filters(this), (None, None), "another server's datasets");
    });
}

/// Review round 1: the class filter can turn on (its dataset's class list read after its images)
/// or off (read again with fewer classes) under the open inspector, which keeps a place in the
/// filtered list: it moved to another image. It stays on the image it showed.
#[test]
fn the_inspector_keeps_its_image_when_the_class_filter_changes_under_it() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    click_option(&mut app, window, "ds-class", 5);
    // A hexagon image whose place in the filtered list is not its place among all images.
    let (image, at) = drive(&mut app, window, |this, _, _| {
        let all = this.store.images.get(HARD).and_then(|l| l.ready()).unwrap();
        let f = this.filtered_indices(HARD);
        let (at, &i) = f.iter().enumerate().find(|(k, i)| **i != *k).expect("a hexagon image after another");
        (all[i].id.clone(), at)
    });
    drive(&mut app, window, move |this, w, cx| this.open_inspector(HARD.into(), at, w, cx));
    settle(&mut app, window);
    let on = |app: &mut HeadlessApp| {
        drive(app, window, |this, _, _| {
            let i = this.inspector.as_ref().expect("the inspector is open");
            this.filtered_image(HARD, i.index).map(|m| m.id.clone())
        })
    };
    assert_eq!(on(&mut app).as_deref(), Some(image.as_str()));
    drive(&mut app, window, |this, _, cx| {
        let ds = this.store.dataset.get_mut(HARD).and_then(|l| l.ready_mut()).unwrap();
        ds.classes.truncate(3);
        let list = this.store.datasets.ready_mut().unwrap();
        list.iter_mut().find(|d| d.id == HARD).unwrap().classes.truncate(3);
        cx.notify();
    });
    settle(&mut app, window);
    let (filtered, all) = shown(&mut app, window, HARD);
    assert_eq!(filtered, all, "the filter is off");
    assert_eq!(on(&mut app).as_deref(), Some(image.as_str()), "still on its image");
}

/// Review round 1: a run's sample the split filter hides clears the split (the inspector walks the
/// filtered list), and the class filter only if it is the sample's dataset's.
#[test]
fn a_sample_the_split_hides_clears_the_split_but_not_another_datasets_class() {
    use crate::api::Split;
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    click_option(&mut app, window, "ds-class", 5);
    drive(&mut app, window, |this, _, cx| this.ensure_dataset(SHAPES, cx));
    settle(&mut app, window);
    let image = drive(&mut app, window, |this, _, _| {
        let m = this.store.images.get(SHAPES).and_then(|l| l.ready()).unwrap()[0].clone();
        this.ds_split = Some(if m.split == Split::Train { Split::Val } else { Split::Train });
        m.id
    });
    let wanted = image.clone();
    drive(&mut app, window, move |this, w, cx| {
        this.open_sample(SHAPES.into(), wanted, "run_x".into(), w, cx)
    });
    settle(&mut app, window);
    drive(&mut app, window, move |this, _, _| {
        let i = this.inspector.as_ref().expect("the inspector is open");
        assert!(i.want.is_none(), "landed");
        assert_eq!(this.filtered_image(SHAPES, i.index).map(|m| m.id.clone()), Some(image));
        assert_eq!(this.ds_split, None, "the split hid it: cleared");
        assert_eq!(this.ds_class, Some((HARD.to_string(), 4)), "shapes-hard's class stays");
    });
}
