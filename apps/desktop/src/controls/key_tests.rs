//! Enter and Space on a focused control, pressed and released as a keyboard does: one press, one
//! action. (folo-ui, as GPUI, also turns the release into a click on the focused element.)

use folo_ui::{HeadlessApp, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, WindowHandle};

use crate::app::tests::{drive, open};
use crate::app::{FoloApp, Route};

/// One press of `key`: down, then up, each answered, a frame drawn after each as on screen.
fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
    let keystroke = Keystroke::parse(key).unwrap();
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    app.frame(window.into());
}

/// The Playground with the demo's model on one of its images, `key`'s control focused.
fn playground_focused(key: &'static str) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.pick_play_model("m_44c0".into(), cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh(key, cx)));
    app.frame(window.into());
    (app, window)
}

#[test]
fn enter_opens_a_select_and_enter_again_picks_once() {
    let (mut app, window) = playground_focused("play-model");
    press(&mut app, window, "enter");
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("play-model"), "Enter opens it");
    let before = drive(&mut app, window, |this, _, _| this.play_model.clone());
    press(&mut app, window, "down");
    press(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.menu_open, None, "Enter picks and closes it");
        assert_ne!(this.play_model, before, "the highlighted model is picked");
    });
    press(&mut app, window, "space");
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("play-model"), "so does Space");
}

#[test]
fn enter_on_a_button_acts_once() {
    let (mut app, window) = playground_focused("play-picker-toggle");
    let open_before = drive(&mut app, window, |this, _, _| this.play_picker_open);
    press(&mut app, window, "enter");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_picker_open),
        !open_before,
        "one press, one toggle"
    );
}
