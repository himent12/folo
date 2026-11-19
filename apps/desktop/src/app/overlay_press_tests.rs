//! Presses on an overlay that a click opened (#350): the click that opens a dialog is often the
//! first half of a double-click, and the second press lands on the dialog, not on the button. An
//! overlay that a key opened takes every press, whatever click it continues.

use folo_ui::{HeadlessApp, WindowHandle};

use super::select_driving::{click_n_at, press};
use super::tests::{drive, open};
use super::{FoloApp, Route, ServerMode};

/// A spot on the page clear of every centred dialog and the palette: the backdrop once one is up,
/// blank page (below the Runs table, beside the Datasets cards) before.
const BACKDROP: (f32, f32) = (1200., 700.);
/// Generate synthetic, in the Datasets header (1280 × 800): under the title bar, so the backdrop
/// once the form is up.
const SYNTH_BUTTON: (f32, f32) = (1050., 73.);
/// Delete, in a finished run's header.
const DELETE_BUTTON: (f32, f32) = (930., 73.);

fn click(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, (x, y): (f32, f32), count: usize) {
    click_n_at(app, window, x, y, count);
}

/// The Datasets page of a local server (Generate synthetic is offered).
fn datasets() -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = ServerMode::Local;
        this.navigate(Route::Datasets, w, cx);
    });
    app.frame(window.into());
    (app, window)
}

fn synth_open(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> bool {
    drive(app, window, |this, _, _| this.synth.is_some())
}

#[test]
fn the_second_press_of_the_double_click_that_opened_a_dialog_leaves_it_open() {
    let (mut app, window) = datasets();
    click(&mut app, window, SYNTH_BUTTON, 1);
    assert!(synth_open(&mut app, window), "the click on Generate synthetic opens the form");
    // The second press of the double-click lands on the backdrop, and the triple click's third.
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(synth_open(&mut app, window), "the double-click keeps the form");
    click(&mut app, window, SYNTH_BUTTON, 3);
    assert!(synth_open(&mut app, window), "and a triple click");
    // A deliberate press on the backdrop still dismisses it.
    click(&mut app, window, BACKDROP, 1);
    assert!(!synth_open(&mut app, window), "a later press dismisses");
    // Opened again by a double-click: the new one keeps out the rest of its own click too.
    click(&mut app, window, SYNTH_BUTTON, 1);
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(synth_open(&mut app, window), "a reopened dialog stays too");
}

#[test]
fn a_dialog_opened_by_a_key_takes_the_next_press_of_an_earlier_click() {
    for key in ["enter", "space"] {
        let (mut app, window) = datasets();
        // A click on the page, then the focused button activated by its key within the
        // double-click time: the next press on the backdrop is the platform's second of that click.
        click(&mut app, window, BACKDROP, 1);
        drive(&mut app, window, |this, w, cx| w.focus(&this.fh("ds-synth", cx)));
        app.frame(window.into());
        press(&mut app, window, key);
        assert!(synth_open(&mut app, window), "{key} on Generate synthetic opens the form");
        click(&mut app, window, BACKDROP, 2);
        assert!(!synth_open(&mut app, window), "a press on the backdrop dismisses what {key} opened");
    }
}

#[test]
fn the_palette_opened_by_its_shortcut_takes_the_next_press_of_an_earlier_click() {
    let (mut app, window) = open();
    app.update(|cx| {
        super::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    app.frame(window.into());
    // A click on the blank page under the table, Ctrl+K, then the same spot clicked again within
    // the double-click time (the platform reports it as the click's second press).
    click(&mut app, window, BACKDROP, 1);
    press(&mut app, window, "ctrl-k");
    assert!(drive(&mut app, window, |this, _, _| this.palette.is_some()), "Ctrl+K opens the palette");
    click(&mut app, window, BACKDROP, 2);
    assert!(drive(&mut app, window, |this, _, _| this.palette.is_none()), "the press dismisses it");
}

#[test]
fn a_double_click_that_opens_a_confirmation_cannot_confirm_it() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run("run_52e8".into()), w, cx));
    app.frame(window.into());
    click(&mut app, window, DELETE_BUTTON, 1);
    assert!(drive(&mut app, window, |this, _, _| this.confirm_delete.is_some()), "Delete asks");
    // The confirm button: rightmost in the footer under the body (20 px in, 14 px down).
    let body = drive(&mut app, window, |this, _, _| this.dialog_scroll.bounds());
    let confirm = (f32::from(body.right()) - 20. - 8., f32::from(body.bottom()) + 14. + 8.);
    // The second press of the double-click that opened it lands on Delete run: nothing happens.
    click(&mut app, window, confirm, 2);
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_some(), "still asking");
        assert!(this.store.run("run_52e8").is_some(), "nothing deleted");
    });
    // A click of its own there confirms (the spot is the button).
    click(&mut app, window, confirm, 1);
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_none());
        assert!(this.store.run("run_52e8").is_none(), "deleted");
    });
}
