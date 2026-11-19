//! Presses on an overlay that a double-click opened (#350): the click that opens a dialog is often
//! the first half of a double-click, and the second press lands on the dialog, not on the button.

use folo_ui::{HeadlessApp, WindowHandle};

use super::select_driving::click_n_at;
use super::tests::{drive, open};
use super::{DeleteTarget, FoloApp, Route, ServerMode};

/// A spot on the page clear of every centred dialog: the backdrop once one is up.
const BACKDROP: (f32, f32) = (1200., 700.);

fn click(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, (x, y): (f32, f32), count: usize) {
    click_n_at(app, window, x, y, count);
}

#[test]
fn the_second_press_of_the_double_click_that_opened_a_dialog_leaves_it_open() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = ServerMode::Local;
        this.navigate(Route::Datasets, w, cx);
    });
    app.frame(window.into());
    // The first press opens Generate synthetic (as the button does on that click).
    click(&mut app, window, BACKDROP, 1);
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    // The second press of the double-click lands on the backdrop, and the triple click's third.
    click(&mut app, window, BACKDROP, 2);
    assert!(drive(&mut app, window, |this, _, _| this.synth.is_some()), "the double-click keeps the form");
    click(&mut app, window, BACKDROP, 3);
    assert!(drive(&mut app, window, |this, _, _| this.synth.is_some()), "and a triple click");
    // A deliberate press on the backdrop still dismisses it.
    click(&mut app, window, BACKDROP, 1);
    assert!(drive(&mut app, window, |this, _, _| this.synth.is_none()), "a later press dismisses");

    // Opened from the keyboard: the first press on the backdrop dismisses, as before.
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    click(&mut app, window, BACKDROP, 1);
    assert!(drive(&mut app, window, |this, _, _| this.synth.is_none()));
    // Closed and opened again by a double-click: the new one has seen no press yet.
    click(&mut app, window, BACKDROP, 1);
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    click(&mut app, window, BACKDROP, 2);
    assert!(drive(&mut app, window, |this, _, _| this.synth.is_some()), "a reopened dialog stays too");
}

#[test]
fn a_double_click_that_opens_a_confirmation_cannot_confirm_it() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.ask_delete(DeleteTarget::Run("run_52e8".into()), w, cx));
    app.frame(window.into());
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
