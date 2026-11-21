//! Datasets › class filter (issue #301): the filter belongs to the dataset it was set on and
//! applies there only while its class is one of that dataset's classes, whichever way the page was
//! reached (a launch straight onto it, the list and back) and whatever else reads the filtered list
//! (the image inspector opened from another dataset's run sample). A change of data source forgets
//! it.

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

/// A class list read again with fewer classes leaves the stored class past them: the select shows "All classes", and so does the grid.
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

/// Final check: a change of data source (a switch, a reconnect, a restarted server recovered) may
/// be another data dir, where the ids and class indices mean other things: it forgets the filters,
/// nothing comes back, and what is picked on the new source stays as picked, before and after its
/// `/system` answers (once, "All classes" picked after a class came back as the old class).
#[test]
fn a_change_of_data_source_forgets_the_filters_and_keeps_new_picks() {
    use crate::api::mock::MockBackend;
    use crate::store::Load;
    // A local `folo serve` names its data dir; restarted, it names the same one.
    let serve = || MockBackend::new(4200, true).with_data_dir("D:/data");
    let (mut app, window) = super::tests::open_with(std::sync::Arc::new(serve()));
    go(&mut app, window, Route::Dataset(HARD.into()));
    click_option(&mut app, window, "ds-class", 5);
    let (filtered, all) = shown(&mut app, window, HARD);
    assert!(filtered < all, "hexagon picked: {filtered} of {all}");
    // The same server, restarted: a new backend whose `/system` is still preparing (it is asked
    // once the test lets it).
    let switch = |app: &mut HeadlessApp| {
        drive(app, window, move |this, _, cx| {
            this.ds_empty_of = Some(SHAPES.into());
            this.set_backend(std::sync::Arc::new(serve()), super::ServerMode::Local, cx);
            this.store.system = Load::Loading;
        });
        settle(app, window);
        drive(app, window, |this, _, _| {
            assert!(this.store.system.ready().is_none(), "`/system` has not answered");
            assert!(!this.empty_only(SHAPES), "the empty-images filter went");
        });
    };
    let answer = |app: &mut HeadlessApp| {
        drive(app, window, |this, _, cx| {
            this.store.system = Load::Idle;
            this.ensure_core(cx);
        });
        settle(app, window);
        assert!(drive(app, window, |this, _, _| this.store.system.ready().is_some()), "answered");
    };
    switch(&mut app);
    assert_eq!(shown(&mut app, window, HARD), (all, all), "the old hexagon filter went");
    click_option(&mut app, window, "ds-class", 1);
    let circle = shown(&mut app, window, HARD);
    assert!(circle.0 < all, "circle picked before the answer applies at once: {circle:?}");
    click_option(&mut app, window, "ds-class", 0);
    assert_eq!(shown(&mut app, window, HARD), (all, all), "All classes");
    answer(&mut app);
    assert_eq!(shown(&mut app, window, HARD), (all, all), "All classes stays after the answer");
    // A class picked before the answer stays after it.
    switch(&mut app);
    click_option(&mut app, window, "ds-class", 1);
    assert_eq!(shown(&mut app, window, HARD), circle, "circle, before the answer");
    answer(&mut app);
    assert_eq!(shown(&mut app, window, HARD), circle, "circle stays after the answer");
}

/// Review round 2: the class filter turned on after the inspector's image was drawn, before the
/// next frame; Next stepped from the old place in the new list, past another image. It steps from
/// the image on screen.
#[test]
fn next_steps_from_the_image_on_screen_when_the_class_filter_changed_before_a_frame() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    // Hexagon images (class 4) by their place among all images.
    let (at, from, next) = drive(&mut app, window, |this, _, _| {
        let all = this.store.images.get(HARD).and_then(|l| l.ready()).unwrap();
        let hex: Vec<usize> = (0..all.len()).filter(|&i| all[i].boxes.iter().any(|b| b.class == 4)).collect();
        let k = (1..hex.len() - 1).find(|&k| hex[k] != k).expect("a hexagon image after others");
        (hex[k], all[hex[k]].id.clone(), all[hex[k + 1]].id.clone())
    });
    drive(&mut app, window, move |this, w, cx| this.open_inspector(HARD.into(), at, w, cx));
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        let i = this.inspector.as_ref().unwrap();
        assert_eq!(this.filtered_image(HARD, i.index).map(|m| m.id.clone()), Some(from));
        this.ds_class = Some((HARD.into(), 4));
        this.step_inspector(1, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, move |this, _, _| {
        let i = this.inspector.as_ref().unwrap();
        assert_eq!(this.filtered_image(HARD, i.index).map(|m| m.id.clone()), Some(next), "the next hexagon");
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

/// Review round 3: the empty-images filter cleared under the open inspector (a server switch)
/// moves every place, as the class filter does: it stays on the image it showed.
#[test]
fn the_inspector_keeps_its_image_when_the_empty_images_filter_goes() {
    let (mut app, window) = open();
    go(&mut app, window, Route::Dataset(HARD.into()));
    drive(&mut app, window, |this, _, cx| {
        this.ds_empty_of = Some(HARD.into());
        cx.notify();
    });
    settle(&mut app, window);
    let (image, at) = drive(&mut app, window, |this, _, _| {
        let all = this.store.images.get(HARD).and_then(|l| l.ready()).unwrap();
        let f = this.filtered_indices(HARD);
        let (at, &i) = f.iter().enumerate().find(|(k, i)| **i != *k).expect("an empty image after others");
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
        this.ds_empty_of = None;
        cx.notify();
    });
    settle(&mut app, window);
    assert_eq!(on(&mut app).as_deref(), Some(image.as_str()), "still on its image");
}
