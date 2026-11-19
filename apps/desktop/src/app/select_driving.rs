//! Driving a select as a person does, for the app's tests: real key presses, and pointer clicks on
//! a menu row at the place it is drawn.

use folo_ui::{HeadlessApp, WindowHandle};

use super::FoloApp;
use super::tests::drive;

/// One press of `key` on the focused control: down, then up, a frame after each (as on screen).
pub(crate) fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
    use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};
    let keystroke = Keystroke::parse(key).unwrap();
    let down = KeyDownEvent { keystroke: keystroke.clone(), is_held: false };
    app.simulate_input(window.into(), PlatformInput::KeyDown(down));
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    app.frame(window.into());
}

/// A left click at `(x, y)`: the pointer moves there (a frame drawn, so it hovers), then presses and
/// releases.
pub(crate) fn click_at(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, x: f32, y: f32) {
    click_n_at(app, window, x, y, 1);
}

/// [`click_at`] as the platform reports the `click_count`th click of a quick series there (2: the
/// second half of a double-click).
pub(crate) fn click_n_at(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    x: f32,
    y: f32,
    click_count: usize,
) {
    use folo_ui::{
        Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PlatformInput, point, px,
    };
    let position = point(px(x), px(y));
    let modifiers = Modifiers::default();
    let moved = MouseMoveEvent { position, pressed_button: None, modifiers };
    app.simulate_input(window.into(), PlatformInput::MouseMove(moved));
    app.run_until_idle();
    app.frame(window.into());
    let button = MouseButton::Left;
    let down = MouseDownEvent { button, position, modifiers, click_count, first_mouse: false };
    app.simulate_input(window.into(), PlatformInput::MouseDown(down));
    app.run_until_idle();
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent { button, position, modifiers, click_count }),
    );
    app.run_until_idle();
    app.frame(window.into());
}

/// The open select's trigger, after Enter on the focused select `key`.
pub(crate) fn open_select(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    key: &'static str,
) -> folo_ui::Bounds<folo_ui::Pixels> {
    drive(app, window, move |this, w, cx| w.focus(&this.fh(key, cx)));
    app.frame(window.into());
    press(app, window, "enter");
    drive(app, window, move |this, _, _| {
        assert_eq!(this.menu_open, Some(key), "Enter opens {key}");
        this.menu_trigger.get().expect("the open menu's trigger was measured")
    })
}

/// Opens the select `key` (focused, Enter) and clicks its option `i`: the menu hangs 4 px below the
/// trigger, behind a 1 px border and 4 px of padding, its rows 32 px tall, 1 px apart. When that
/// row would not show below the trigger, the page is scrolled first to put the trigger 200 px from
/// the top (a select in a dialog has to fit as it is).
pub(crate) fn click_option(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    key: &'static str,
    i: usize,
) {
    let viewport_h = drive(app, window, |_, w, _| f32::from(w.viewport_size().height));
    // Room below the trigger for the menu down to row `i` (select.rs: 12 px kept from the edge).
    let fits = |trigger: folo_ui::Bounds<folo_ui::Pixels>| {
        let below = viewport_h - f32::from(trigger.bottom()) - 12.;
        5. + 33. * (i as f32 + 1.) <= below.min(288.)
    };
    let mut trigger = open_select(app, window, key);
    if !fits(trigger) {
        let top = f32::from(trigger.top());
        press(app, window, "escape");
        drive(app, window, move |this, _, _| {
            let at = this.page_scroll.offset();
            this.page_scroll.set_offset(folo_ui::point(at.x, at.y - folo_ui::px(top - 200.)));
        });
        app.frame(window.into());
        trigger = open_select(app, window, key);
    }
    assert!(fits(trigger), "{key}'s menu opens below it, row {i} shown: {trigger:?}");
    let menu_top = f32::from(trigger.bottom()) + 4.;
    click_at(app, window, f32::from(trigger.left()) + 40., menu_top + 1. + 4. + 33. * i as f32 + 16.);
    assert_eq!(drive(app, window, |this, _, _| this.menu_open), None, "the click closes {key}'s menu");
}
