//! The palette's text field driven by keys, as a person edits a query: the selection and word keys
//! every single-line field answers (stress d5-window-input-2).

use folo_ui::{HeadlessApp, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, WindowHandle};

use super::FoloApp;
use super::tests::{drive, open};

/// One press of `key` (down, then up), a frame drawn after it.
fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
    let keystroke = Keystroke::parse(key).unwrap();
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
    );
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.frame(window.into());
}

/// The palette open with `text` in its field, the caret at the end.
fn palette_with(text: &str) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    app.update(crate::input::bind_keys);
    let text = text.to_string();
    drive(&mut app, window, move |this, w, cx| {
        this.open_palette(w, cx);
        let input = this.palette.as_ref().unwrap().input.clone();
        input.update(cx, |i, cx| i.set_text(text, cx));
    });
    app.frame(window.into());
    press(&mut app, window, "end");
    (app, window)
}

/// The field's text and selected byte range.
fn field(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (String, std::ops::Range<usize>) {
    drive(app, window, |this, _, cx| {
        let input = this.palette.as_ref().unwrap().input.read(cx);
        (input.text().to_string(), input.selection())
    })
}

#[test]
fn shift_home_and_shift_end_select_to_the_ends() {
    let (mut app, window) = palette_with("shapes hard");
    press(&mut app, window, "shift-home");
    assert_eq!(field(&mut app, window), ("shapes hard".into(), 0..11), "Shift+Home from the end selects all");
    press(&mut app, window, "home");
    press(&mut app, window, "shift-end");
    assert_eq!(field(&mut app, window).1, 0..11, "Shift+End from the start selects all");
    press(&mut app, window, "backspace");
    assert_eq!(field(&mut app, window).0, "", "and typing over it replaces it");
}

#[test]
fn ctrl_shift_arrows_select_by_word() {
    let (mut app, window) = palette_with("shapes hard run");
    press(&mut app, window, "ctrl-shift-left");
    assert_eq!(field(&mut app, window).1, 12..15, "the last word");
    press(&mut app, window, "ctrl-shift-left");
    assert_eq!(field(&mut app, window).1, 7..15, "and the one before");
    press(&mut app, window, "home");
    press(&mut app, window, "ctrl-shift-right");
    assert_eq!(field(&mut app, window).1, 0..6, "from the start, the first word");
}

#[test]
fn ctrl_delete_deletes_the_next_word() {
    let (mut app, window) = palette_with("shapes hard");
    press(&mut app, window, "home");
    press(&mut app, window, "ctrl-delete");
    assert_eq!(field(&mut app, window), (" hard".into(), 0..0));
    // At the end there is nothing after the caret: nothing changes.
    press(&mut app, window, "end");
    press(&mut app, window, "ctrl-delete");
    assert_eq!(field(&mut app, window).0, " hard");
}
