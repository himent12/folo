//! Hit testing over the caption areas and the resize frame, mouse, wheel, keys and text.

use crate::common::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use folo_ui::*;
use std::cell::RefCell;
use std::rc::Rc;

pub fn hit_tests(checks: &mut Checks, hwnd: HWND, scale: f32, log: &Rc<RefCell<Log>>) {
    let o = client_origin(hwnd);
    let r = rect(hwnd);
    let at = |x: f32, y: f32| (o.x + (x * scale) as i32, o.y + (y * scale) as i32);
    let cases: [(&str, (i32, i32), u32); 9] = [
        ("title bar (Drag)", at(400., 20.), HTCAPTION),
        ("Minimize", at(830., 20.), HTMINBUTTON),
        ("Maximize", at(890., 20.), HTMAXBUTTON),
        ("Close", at(940., 20.), HTCLOSE),
        ("the page", at(400., 300.), HTCLIENT),
        ("the top edge strip", (o.x + (400. * scale) as i32, o.y + 1), HTTOP),
        ("the left edge", (r.left + 3, (r.top + r.bottom) / 2), HTLEFT),
        ("the right edge", (r.right - 3, (r.top + r.bottom) / 2), HTRIGHT),
        ("the bottom edge", ((r.left + r.right) / 2, r.bottom - 3), HTBOTTOM),
    ];
    for (name, (x, y), want) in cases {
        post(hwnd, WM_NCMOUSEMOVE, HTCLIENT as usize, pt(x, y));
        pump();
        let got = send(hwnd, WM_NCHITTEST, 0, pt(x, y)) as u32;
        checks.check(&format!("WM_NCHITTEST over {name}"), got == want, format!("got {got}, want {want}"));
    }
    take_inputs(log);
}

pub fn mouse(checks: &mut Checks, hwnd: HWND, scale: f32, log: &Rc<RefCell<Log>>) {
    let lp = |x: f32, y: f32| pt((x * scale) as i32, (y * scale) as i32);
    let near =
        |p: Point<Pixels>, x: f32, y: f32| (p.x / px(1.) - x).abs() < 1. && (p.y / px(1.) - y).abs() < 1.;
    set_keys(&[]);
    post(hwnd, WM_MOUSEMOVE, 0, lp(150., 90.));
    pump();
    let got = take_inputs(log);
    let ok = matches!(got.as_slice(), [PlatformInput::MouseMove(m)] if near(m.position, 150., 90.) && m.pressed_button.is_none());
    checks.check("mouse move in logical px", ok, format!("{got:?}"));
    checks.check(
        "hover reported",
        log.borrow().hover.first() == Some(&true),
        format!("{:?}", log.borrow().hover),
    );

    // Two quick left clicks: counts 1 then 2, with capture while pressed.
    post(hwnd, WM_LBUTTONDOWN, 1, lp(200., 100.));
    pump();
    // SAFETY: a plain query.
    let captured = unsafe { GetCapture() } == hwnd;
    post(hwnd, WM_MOUSEMOVE, 1, lp(201., 100.));
    post(hwnd, WM_LBUTTONUP, 0, lp(201., 100.));
    post(hwnd, WM_LBUTTONDOWN, 1, lp(201., 101.));
    post(hwnd, WM_LBUTTONUP, 0, lp(201., 101.));
    pump();
    // SAFETY: a plain query.
    let released = unsafe { GetCapture() }.is_invalid();
    let got = take_inputs(log);
    let counts: Vec<(String, usize)> = got
        .iter()
        .filter_map(|i| match i {
            PlatformInput::MouseDown(d) => Some((format!("down {:?}", d.button), d.click_count)),
            PlatformInput::MouseUp(u) => Some((format!("up {:?}", u.button), u.click_count)),
            _ => None,
        })
        .collect();
    let want =
        vec![("down Left".into(), 1), ("up Left".into(), 1), ("down Left".into(), 2), ("up Left".into(), 2)];
    checks.check("left double click counts 1, 2", counts == want, format!("{counts:?}"));
    let drag = got.iter().any(|i| matches!(i, PlatformInput::MouseMove(m) if m.pressed_button == Some(MouseButton::Left) && near(m.position, 201., 100.)));
    checks.check("move while pressed reports the button", drag, "");
    checks.check("capture while pressed, released after", captured && released, "");

    // Right, middle and the navigation buttons; Shift held.
    set_keys(&[VK_SHIFT, VK_LSHIFT]);
    post(hwnd, WM_RBUTTONDOWN, 2, lp(10., 20.));
    post(hwnd, WM_RBUTTONUP, 0, lp(10., 20.));
    post(hwnd, WM_MBUTTONDOWN, 0x10, lp(11., 20.));
    post(hwnd, WM_MBUTTONUP, 0, lp(11., 20.));
    post(hwnd, WM_XBUTTONDOWN, 1 << 16, lp(12., 20.));
    post(hwnd, WM_XBUTTONUP, 1 << 16, lp(12., 20.));
    post(hwnd, WM_XBUTTONDOWN, 2 << 16, lp(13., 20.));
    post(hwnd, WM_XBUTTONUP, 2 << 16, lp(13., 20.));
    pump();
    let got = take_inputs(log);
    let downs: Vec<(MouseButton, bool, usize)> = got
        .iter()
        .filter_map(|i| match i {
            PlatformInput::MouseDown(d) => Some((d.button, d.modifiers.shift, d.click_count)),
            _ => None,
        })
        .collect();
    let want = vec![
        (MouseButton::Right, true, 1),
        (MouseButton::Middle, true, 1),
        (MouseButton::Navigate(NavigationDirection::Back), true, 1),
        (MouseButton::Navigate(NavigationDirection::Forward), true, 1),
    ];
    checks.check("right, middle, back, forward with shift", downs == want, format!("{downs:?}"));

    // Wheel: lines, by the system setting; shift makes it horizontal; WM_MOUSEHWHEEL is horizontal.
    let mut lines = 3u32;
    let mut chars = 3u32;
    // SAFETY: u32 parameters.
    unsafe {
        let _ = SystemParametersInfoW(
            SPI_GETWHEELSCROLLLINES,
            0,
            Some(&mut lines as *mut u32 as *mut _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
        let _ = SystemParametersInfoW(
            SPI_GETWHEELSCROLLCHARS,
            0,
            Some(&mut chars as *mut u32 as *mut _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
    }
    let o = client_origin(hwnd);
    let screen = pt(o.x + (300. * scale) as i32, o.y + (200. * scale) as i32);
    set_keys(&[]);
    post(hwnd, WM_MOUSEWHEEL, (120usize) << 16, screen);
    pump();
    set_keys(&[VK_SHIFT, VK_LSHIFT]);
    post(hwnd, WM_MOUSEWHEEL, ((-120i16 as u16) as usize) << 16, screen);
    pump();
    set_keys(&[]);
    post(hwnd, WM_MOUSEHWHEEL, (120usize) << 16, screen);
    pump();
    let got = take_inputs(log);
    let deltas: Vec<(f32, f32, bool)> = got
        .iter()
        .filter_map(|i| match i {
            PlatformInput::ScrollWheel(w) if near(w.position, 300., 200.) => match w.delta {
                ScrollDelta::Lines(d) => Some((d.x, d.y, w.modifiers.shift)),
                ScrollDelta::Pixels(_) => None,
            },
            _ => None,
        })
        .collect();
    let want = vec![(0., lines as f32, false), (-(chars as f32), 0., true), (-(chars as f32), 0., false)];
    checks.check(
        "wheel lines (vertical, shift horizontal, horizontal)",
        deltas == want,
        format!("{deltas:?}, want {want:?}"),
    );
}

/// One key check: name, keys held, virtual key, then the expected key, shift, ctrl, key_char, text.
type KeyCase = (
    &'static str,
    &'static [VIRTUAL_KEY],
    VIRTUAL_KEY,
    &'static str,
    bool,
    bool,
    Option<&'static str>,
    &'static [&'static str],
);

pub fn keys(checks: &mut Checks, hwnd: HWND, log: &Rc<RefCell<Log>>) {
    take_inputs(log);
    take_text(log);
    // (name, keys held, virtual key, key, shift, ctrl, key_char, text inserted)
    let cases: [KeyCase; 10] = [
        ("a", &[], VK_A, "a", false, false, Some("a"), &["replace:a"]),
        ("A", &[VK_SHIFT, VK_LSHIFT], VK_A, "a", true, false, Some("A"), &["replace:A"]),
        ("Enter", &[], VK_RETURN, "enter", false, false, None, &[]),
        ("Ctrl+K (handled: no text)", &[VK_CONTROL, VK_LCONTROL], VK_K, "k", false, true, None, &[]),
        ("Shift+Tab", &[VK_SHIFT, VK_LSHIFT], VK_TAB, "tab", true, false, None, &[]),
        ("F11", &[], VK_F11, "f11", false, false, None, &[]),
        ("Left", &[], VK_LEFT, "left", false, false, None, &[]),
        ("Up", &[], VK_UP, "up", false, false, None, &[]),
        ("Right", &[], VK_RIGHT, "right", false, false, None, &[]),
        ("Down", &[], VK_DOWN, "down", false, false, None, &[]),
    ];
    for (name, held, vk, key, shift, ctrl, key_char, text) in cases {
        set_keys(held);
        post(hwnd, WM_KEYDOWN, vk.0 as usize, key_lparam(vk, false, false));
        pump();
        post(hwnd, WM_KEYUP, vk.0 as usize, key_lparam(vk, false, true));
        pump();
        let got = take_inputs(log);
        let inserted = take_text(log);
        let down = got.iter().find_map(|i| match i {
            PlatformInput::KeyDown(k) => Some(k.clone()),
            _ => None,
        });
        let up = got.iter().any(|i| matches!(i, PlatformInput::KeyUp(k) if k.keystroke.key == key));
        let ok = down.as_ref().is_some_and(|d| {
            d.keystroke.key == key
                && d.keystroke.modifiers.shift == shift
                && d.keystroke.modifiers.control == ctrl
                && d.keystroke.key_char.as_deref() == key_char
                && !d.is_held
        }) && up
            && inserted == text;
        checks.check(
            &format!("key {name}"),
            ok,
            format!("{:?}, text {inserted:?}", down.map(|d| d.keystroke)),
        );
    }
    // Auto-repeat: the previous-state bit makes is_held.
    set_keys(&[]);
    post(hwnd, WM_KEYDOWN, VK_LEFT.0 as usize, key_lparam(VK_LEFT, true, false));
    pump();
    let got = take_inputs(log);
    checks.check(
        "key repeat is_held",
        matches!(got.as_slice(), [PlatformInput::KeyDown(k)] if k.is_held),
        format!("{got:?}"),
    );
    // A modifier press reports the change once.
    set_keys(&[VK_SHIFT, VK_LSHIFT]);
    post(hwnd, WM_KEYDOWN, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, false, false));
    post(hwnd, WM_KEYDOWN, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, true, false));
    pump();
    set_keys(&[]);
    post(hwnd, WM_KEYUP, VK_SHIFT.0 as usize, key_lparam(VK_SHIFT, false, true));
    pump();
    let got = take_inputs(log);
    let changes: Vec<bool> = got
        .iter()
        .filter_map(|i| match i {
            PlatformInput::ModifiersChanged(m) => Some(m.modifiers.shift),
            _ => None,
        })
        .collect();
    checks.check("modifiers changed: shift on (once), off", changes == [true, false], format!("{changes:?}"));
    // A surrogate pair in two WM_CHARs is one character.
    post(hwnd, WM_CHAR, 0xD83D, 1);
    post(hwnd, WM_CHAR, 0xDE00, 1);
    pump();
    let text = take_text(log);
    checks.check("WM_CHAR surrogate pair", text == ["replace:\u{1F600}"], format!("{text:?}"));
    set_keys(&[]);
}
