//! Selects whose chosen option matters (review r5): a click on the chosen option calls back, as on
//! every select, so New run can ask about its dataset again; a select with nothing chosen (no
//! teacher yet, a class filter past the dataset's classes) points at no option, so each option can
//! be picked by pointer and by keys.

use super::select_driving::{click_option, press};
use super::tests::{drive, open};
use super::{FoloApp, Route};
use folo_ui::{HeadlessApp, WindowHandle};

/// The list counts no training image in the form's dataset (read before images were added, its
/// first check already answered); the server counts them now. Picking that dataset again asks
/// once more, so Start is no longer held back.
#[test]
fn re_picking_new_runs_dataset_asks_the_server_again() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.open_new_run(Some("ds_8f2c"), w, cx));
    app.frame(window.into());
    let at = drive(&mut app, window, |this, _, _| {
        let list = this.store.datasets.ready_mut().unwrap();
        list.iter_mut().find(|d| d.id == "ds_8f2c").unwrap().splits.train = 0;
        assert!(this.dataset_blocked().is_some(), "blocked by the old count");
        this.store.datasets.ready().unwrap().iter().position(|d| d.id == "ds_8f2c").unwrap()
    });
    app.frame(window.into());
    click_option(&mut app, window, "nr-dataset", at);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.new_run.as_ref().unwrap().dataset, "ds_8f2c", "still the same dataset");
        assert_eq!(this.dataset_blocked(), None, "asked again: the server's count lifts the block");
    });
}

/// New loop on the demo benchmark with no teacher set (none could teach when it opened), drawn.
/// Returns the first teacher listed.
fn new_loop_without_teacher(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> String {
    drive(app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_5e12".into()), w, cx);
        this.ensure_bench("bn_5e12", cx);
        this.ensure_agents(cx);
    });
    app.frame(window.into());
    let first = drive(app, window, |this, _, cx| {
        this.bench.tab = crate::bench::BenchTab::Loops;
        this.open_new_loop(cx);
        let b = this.bench.list.ready().unwrap().iter().find(|b| b.id == "bn_5e12").unwrap().clone();
        this.bench_teachers(&b).first().expect("the demo has a teacher").id.clone()
    });
    app.frame(window.into());
    set_teacher(app, window, None);
    first
}

fn set_teacher(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, to: Option<String>) {
    drive(app, window, move |this, _, cx| {
        this.bench.new_loop.as_mut().unwrap().teacher = to;
        cx.notify();
    });
    app.frame(window.into());
}

fn teacher(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> Option<String> {
    drive(app, window, |this, _, _| this.bench.new_loop.as_ref().unwrap().teacher.clone())
}

/// A teacher that appears while New loop is open (no teacher set) is the first option, and no
/// option is the choice: it can be picked by pointer and by keys (review r5: the select was told
/// option 0 was chosen, so a pick of it did nothing).
#[test]
fn new_loops_first_teacher_can_be_picked_when_none_is_set() {
    let (mut app, window) = open();
    let first = new_loop_without_teacher(&mut app, window);
    click_option(&mut app, window, "bench-loop-teacher", 0);
    assert_eq!(teacher(&mut app, window), Some(first.clone()), "by pointer");
    set_teacher(&mut app, window, None);
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("bench-loop-teacher", cx)));
    press(&mut app, window, "enter");
    press(&mut app, window, "home");
    press(&mut app, window, "enter");
    assert_eq!(teacher(&mut app, window), Some(first), "by keys");
}

/// The class stored for the open dataset (even past its classes).
fn class_filter(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> Option<u32> {
    drive(app, window, |this, _, _| {
        let Route::Dataset(open) = &this.route else { return None };
        this.ds_class.as_ref().filter(|(of, _)| of == open).map(|(_, c)| *c)
    })
}

fn open_dataset(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, route: Route) {
    drive(app, window, move |this, w, cx| this.navigate(route, w, cx));
    app.frame(window.into());
}

/// Datasets › class filter: the filter belongs to the dataset it was set on. Another dataset opens
/// unfiltered (review r5: shapes-hard's 5th class carried over to 3-class shapes, which showed
/// "All classes" over an empty grid), the same one keeps it, and a stored class past the dataset's
/// classes is no option, so "All classes" clears it by pointer and by keys.
#[test]
fn a_class_filter_stays_with_its_dataset() {
    let (mut app, window) = open();
    let (hard, shapes) = (Route::Dataset("ds_3b91".into()), Route::Dataset("ds_8f2c".into()));
    open_dataset(&mut app, window, hard);
    click_option(&mut app, window, "ds-class", 5);
    assert_eq!(class_filter(&mut app, window), Some(4), "hexagon, picked through the control");
    open_dataset(&mut app, window, shapes.clone());
    assert_eq!(class_filter(&mut app, window), None, "another dataset opens unfiltered");
    drive(&mut app, window, |this, _, _| {
        let all = this.store.images.get("ds_8f2c").and_then(|l| l.ready()).map_or(0, |l| l.len());
        assert!(all > 0, "its images are in");
        assert_eq!(this.filtered_indices("ds_8f2c").len(), all, "all of them shown");
    });
    // Set here, it stays through the list and back.
    click_option(&mut app, window, "ds-class", 2);
    assert_eq!(class_filter(&mut app, window), Some(1));
    open_dataset(&mut app, window, Route::Datasets);
    open_dataset(&mut app, window, shapes);
    assert_eq!(class_filter(&mut app, window), Some(1), "the same dataset keeps its filter");
    // A class past this dataset's three (its classes changed on a re-read): "All classes" clears it.
    drive(&mut app, window, |this, _, cx| {
        this.ds_class = Some(("ds_8f2c".into(), 4));
        cx.notify();
    });
    app.frame(window.into());
    click_option(&mut app, window, "ds-class", 0);
    assert_eq!(class_filter(&mut app, window), None, "by pointer");
    drive(&mut app, window, |this, _, cx| {
        this.ds_class = Some(("ds_8f2c".into(), 4));
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("ds-class", cx)));
    press(&mut app, window, "enter");
    press(&mut app, window, "home");
    press(&mut app, window, "enter");
    assert_eq!(class_filter(&mut app, window), None, "by keys");
}

/// Opening a long select whose active option sits below the popup's max height scrolls that option
/// into view on the very first frame it mounts (via `ScrollHandle::scroll_to_item` resolving after
/// layout in the same prepaint pass).
#[test]
fn opening_a_long_select_scrolls_chosen_option_into_view_on_first_frame() {
    let (mut app, window) = open();
    open_dataset(&mut app, window, Route::Dataset("ds_8f2c".into()));
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        let ds = this.store.dataset.get_mut("ds_8f2c").and_then(|l| l.ready_mut()).unwrap();
        ds.classes = (0..25).map(|i| format!("cls_{i:02}")).collect();
        // Option 0 is "All classes", so class 20 is option index 21 (far below the 260 px menu).
        this.ds_class = Some(("ds_8f2c".into(), 20));
        w.focus(&this.fh("ds-class", cx));
        cx.notify();
    });
    app.frame(window.into());
    press(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.menu_open, Some("ds-class"));
        assert_eq!(this.menu_hl, 21);
        assert!(
            f32::from(this.menu_scroll.offset().y) < 0.,
            "chosen option 21 should scroll into view on the first open frame, got offset {:?}",
            this.menu_scroll.offset()
        );
    });
}
