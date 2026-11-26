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

#[test]
fn character_index_for_point_stuck_drag_and_ime_selection_direction() {
    use folo_ui::{EntityInputHandler, MouseButton, MouseDownEvent, MouseMoveEvent, point, px};

    // 1. Empty field maps any point inside its bounds to index 0 (not into placeholder glyphs).
    let (mut app, window) = palette_with("");
    drive(&mut app, window, |this, w, cx| {
        let input = this.palette.as_ref().unwrap().input.clone();
        let b = input.read(cx).last_bounds().expect("input laid out");
        let mid = point(b.left() + px(24.), b.top() + b.size.height * 0.5);
        let idx = input.update(cx, |i, cx| i.character_index_for_point(mid, w, cx));
        assert_eq!(idx, Some(0), "empty input must hit-test to offset 0");
    });

    // 2. Non-empty (and horizontally scrolled) field maps distinct X coordinates to increasing
    //    UTF-16 offsets and respects `scroll_x`.
    let long = "abcdefghijklmnopqrstuvwxyz_0123456789_abcdefghijklmnopqrstuvwxyz_0123456789_end";
    let (mut app, window) = palette_with(long);
    let b = drive(&mut app, window, |this, w, cx| {
        let input = this.palette.as_ref().unwrap().input.clone();
        let b = input.read(cx).last_bounds().expect("input laid out");
        let y = b.top() + b.size.height * 0.5;
        let i_left =
            input.update(cx, |i, cx| i.character_index_for_point(point(b.left() + px(4.), y), w, cx));
        let i_mid =
            input.update(cx, |i, cx| i.character_index_for_point(point(b.left() + px(80.), y), w, cx));
        assert!(i_left.is_some() && i_mid.is_some(), "hit-test inside text returns offsets");
        assert!(
            i_mid.unwrap() > i_left.unwrap(),
            "rightward point must yield a larger UTF-16 offset: left={i_left:?}, mid={i_mid:?}"
        );
        b
    });

    // 3. Releasing mouse outside the window (next MouseMove has pressed_button == None) must clear
    //    `is_selecting` instead of dragging the selection forever.
    let y = b.top() + b.size.height * 0.5;
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(b.left() + px(80.), y),
            click_count: 1,
            ..Default::default()
        }),
    );
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(b.left() + px(20.), y),
            pressed_button: Some(MouseButton::Left),
            ..Default::default()
        }),
    );
    app.frame(window.into());
    let dragged_sel = field(&mut app, window).1;
    assert!(!dragged_sel.is_empty(), "active drag created a selection");
    // Move with no button pressed (missed MouseUp): must stop selecting and preserve selection.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(b.left() + px(140.), y),
            pressed_button: None,
            ..Default::default()
        }),
    );
    app.frame(window.into());
    assert_eq!(field(&mut app, window).1, dragged_sel, "released drag must not alter selection");

    // 4. IME composition after a reversed selection resets `selection_reversed` so subsequent
    //    Shift+Right extends forward cleanly.
    let (mut app, window) = palette_with("hello world");
    press(&mut app, window, "shift-left");
    press(&mut app, window, "shift-left");
    assert_eq!(field(&mut app, window).1, 9..11);
    drive(&mut app, window, |this, w, cx| {
        let input = this.palette.as_ref().unwrap().input.clone();
        input.update(cx, |i, cx| {
            i.replace_and_mark_text_in_range(None, "XY", Some(0..1), w, cx);
        });
    });
    app.frame(window.into());
    assert_eq!(field(&mut app, window), ("hello worXY".into(), 9..10));
    press(&mut app, window, "shift-right");
    assert_eq!(
        field(&mut app, window).1,
        9..11,
        "Shift+Right after IME composition extends forward instead of reversing"
    );
}
