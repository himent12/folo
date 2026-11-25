//! The Win32 backend on real windows that are never shown: created hidden (no `WS_VISIBLE`, never
//! activated, nothing on any display) and driven by sending their messages on the test's own thread.
//! Key state is this thread's own (`SetKeyboardState`): no real input is read or injected, and the
//! Caps Lock light is not touched.

use super::WindowsPlatform;
use super::window::WindowsWindow;
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::w;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

crate::actions!(win32_tests, [Handled]);

fn platform() -> Rc<WindowsPlatform> {
    Rc::new(WindowsPlatform::new(Arc::new(FixedTextSystem::new())).expect("the Win32 platform"))
}

fn bounds() -> Bounds<Pixels> {
    Bounds::new(point(px(0.), px(0.)), size(px(300.), px(200.)))
}

fn hidden_params() -> WindowParams {
    WindowParams {
        bounds: bounds(),
        titlebar: None,
        kind: WindowKind::Normal,
        is_movable: true,
        is_resizable: true,
        is_minimizable: true,
        focus: false,
        show: false,
        display_id: None,
        window_min_size: None,
        app_id: None,
    }
}

/// Sets this thread's key state: `down` held, `toggled` on (Caps Lock).
fn set_keys(down: &[VIRTUAL_KEY], toggled: &[VIRTUAL_KEY]) {
    let mut state = [0u8; 256];
    for k in down {
        state[k.0 as usize] |= 0x80;
    }
    for k in toggled {
        state[k.0 as usize] |= 0x01;
    }
    // SAFETY: a 256-byte key state for this thread.
    unsafe { SetKeyboardState(&state).expect("SetKeyboardState") };
}

fn key_lparam(vk: VIRTUAL_KEY, up: bool) -> isize {
    // SAFETY: a plain layout query.
    let scan = unsafe { MapVirtualKeyW(vk.0 as u32, MAPVK_VK_TO_VSC) } as isize;
    let l = 1 | (scan << 16);
    if up { l | (1 << 30) | (1 << 31) } else { l }
}

fn send(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) {
    // SAFETY: a message to our own window, on its thread.
    unsafe { SendMessageW(hwnd, msg, Some(WPARAM(wparam)), Some(LPARAM(lparam))) };
}

fn post(hwnd: HWND, msg: u32, wparam: usize, lparam: isize) {
    // SAFETY: as above, through the queue.
    unsafe { PostMessageW(Some(hwnd), msg, WPARAM(wparam), LPARAM(lparam)).expect("PostMessageW") };
}

/// Runs what is queued for this thread (the WM_CHAR a TranslateMessage posted, main-thread tasks), as
/// the platform's loop does: dispatched, never translated.
fn pump() {
    let mut msg = MSG::default();
    // SAFETY: the standard non-blocking loop over this thread's queue.
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            DispatchMessageW(&msg);
        }
    }
}

/// A key press and release with `down` held and `toggled` on.
fn press(hwnd: HWND, vk: VIRTUAL_KEY, down: &[VIRTUAL_KEY], toggled: &[VIRTUAL_KEY]) {
    set_keys(down, toggled);
    send(hwnd, WM_KEYDOWN, vk.0 as usize, key_lparam(vk, false));
    pump();
    send(hwnd, WM_KEYUP, vk.0 as usize, key_lparam(vk, true));
    pump();
    set_keys(&[], &[]);
}

fn destroy(hwnd: HWND) {
    // SAFETY: destroying our own window on its thread.
    let _ = unsafe { DestroyWindow(hwnd) };
    pump();
}

/// `WM_DPICHANGED` asked whether the window was maximized while it held the state mutably, which
/// panicked ("already mutably borrowed") inside the window procedure and aborted the process on every
/// move to a monitor with another scale.
#[test]
fn a_dpi_change_rescales_the_window() {
    let platform = platform();
    let window = platform.open_windows_window(hidden_params()).unwrap();
    let hwnd = window.hwnd();
    let resizes: Rc<RefCell<Vec<f32>>> = Rc::default();
    window.on_resize(Box::new({
        let resizes = resizes.clone();
        move |_, scale| resizes.borrow_mut().push(scale)
    }));
    let new_dpi = if window.scale_factor() == 2.0 { 144 } else { 192 };
    let mut rect = RECT::default();
    // SAFETY: a valid out pointer.
    unsafe { GetWindowRect(hwnd, &mut rect).unwrap() };
    // Called directly (not sent): a panic is then this test's failure, not an abort.
    window.0.handle_dpi_changed(
        hwnd,
        WPARAM(new_dpi | (new_dpi << 16)),
        LPARAM(&rect as *const RECT as isize),
    );
    let scale = new_dpi as f32 / 96.;
    assert_eq!(window.scale_factor(), scale);
    assert_eq!(resizes.borrow().last(), Some(&scale), "the core heard of the new scale");
    destroy(hwnd);
}

/// A window opened hidden gets no WM_SIZE until it is shown (one with the custom title bar gets one
/// from its frame change, one with the system's does not) and kept `CW_USEDEFAULT` (`i32::MIN`) as its
/// size: the core made a 0x0 frame of it, and the first frame panicked in the renderer inside the
/// window procedure.
#[test]
fn a_window_opened_hidden_has_its_real_size() {
    let platform = platform();
    for appears_transparent in [false, true] {
        let titlebar = TitlebarOptions { title: None, appears_transparent, traffic_light_position: None };
        let window = platform
            .open_windows_window(WindowParams { titlebar: Some(titlebar), ..hidden_params() })
            .unwrap();
        let hwnd = window.hwnd();
        let mut client = RECT::default();
        // SAFETY: a valid out pointer.
        unsafe { GetClientRect(hwnd, &mut client).unwrap() };
        let scale = window.scale_factor();
        let device = |p: Pixels| (p.0 * scale).round() as i32;
        let content = window.content_size();
        assert!(client.right > 0 && client.bottom > 0, "{client:?}");
        assert_eq!(
            (device(content.width), device(content.height)),
            (client.right, client.bottom),
            "custom title bar: {appears_transparent}"
        );
        destroy(hwnd);
    }
}

/// What the window's input callback saw of the mouse buttons.
fn record_buttons(window: &WindowsWindow) -> Rc<RefCell<Vec<String>>> {
    let seen: Rc<RefCell<Vec<String>>> = Rc::default();
    window.on_input(Box::new({
        let seen = seen.clone();
        move |input| {
            match input {
                PlatformInput::MouseDown(e) => seen.borrow_mut().push(format!("down {:?}", e.button)),
                PlatformInput::MouseUp(e) => seen.borrow_mut().push(format!("up {:?}", e.button)),
                _ => {}
            }
            DispatchEventResult::default()
        }
    }));
    seen
}

fn captured(hwnd: HWND) -> bool {
    // SAFETY: a plain query.
    unsafe { GetCapture() == hwnd }
}

/// A press whose release the window never sees (the capture was taken or cancelled, the user switched
/// to another app) used to leave the app's selection or drag going: it now gets its mouse up.
#[test]
fn losing_the_capture_or_activation_releases_the_pressed_buttons() {
    let platform = platform();
    let window = platform.open_windows_window(hidden_params()).unwrap();
    let hwnd = window.hwnd();
    let seen = record_buttons(&window);
    let take = || std::mem::take(&mut *seen.borrow_mut());
    let at = (10 << 16) | 10;

    // A plain click: one down, one up (releasing the capture then finds nothing pressed).
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    assert!(captured(hwnd));
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert!(!captured(hwnd));
    assert_eq!(take(), ["down Left", "up Left"]);

    // The capture goes away (Windows' WM_CANCELMODE, another window's SetCapture).
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    // SAFETY: ending this thread's capture, as Windows does.
    unsafe { ReleaseCapture().unwrap() };
    assert_eq!(take(), ["down Left", "up Left"]);
    // A release that arrives later is still reported (as one outside any press is).
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert_eq!(take(), ["up Left"]);

    // Two buttons: the capture stays until the last is released, and losing it releases what is left.
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_RBUTTONDOWN, 0, at);
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert!(captured(hwnd), "the right button is still down");
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert!(!captured(hwnd));
    assert_eq!(take(), ["down Left", "down Right", "up Left", "up Right"]);

    // Deactivation with a button down (Alt+Tab while selecting).
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_ACTIVATE, WA_INACTIVE as usize, 0);
    pump();
    assert_eq!(take(), ["down Left", "up Left"]);
    destroy(hwnd);
}

/// A view filling the window that counts its clicks and outside mouse-up releases.
struct Clickable {
    clicks: Rc<RefCell<usize>>,
    up_out: Rc<RefCell<usize>>,
}

impl Render for Clickable {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let clicks = self.clicks.clone();
        let up_out = self.up_out.clone();
        div()
            .id("clickable")
            .size_full()
            .on_click(move |_, _, _| *clicks.borrow_mut() += 1)
            .on_mouse_up_out(MouseButton::Left, move |_, _, _| *up_out.borrow_mut() += 1)
    }
}

/// A press cancelled by the capture going away or by deactivation clicks nothing, even with the cursor
/// over the element (the press is reported released outside the window so `on_mouse_up_out` still runs);
/// a press released in the window clicks. Works identically under `FOLO_UI_OFFSCREEN=1` without moving
/// the window onto a display or touching the real pointer.
#[test]
fn a_cancelled_press_does_not_click() {
    let platform = platform();
    let app = App::new_app(platform.clone(), Arc::new(()));
    let clicks: Rc<RefCell<usize>> = Rc::default();
    let up_out: Rc<RefCell<usize>> = Rc::default();
    let handle = app
        .update(|cx| {
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds())),
                focus: false,
                show: false,
                ..Default::default()
            };
            let clicks = clicks.clone();
            let up_out = up_out.clone();
            cx.open_window(options, move |_, cx| cx.new(|_| Clickable { clicks, up_out }))
        })
        .unwrap();
    let draw = || app.update(|cx| cx.update_window(handle.into(), |_, window, cx| window.draw(cx))).unwrap();
    draw();
    let hwnd = platform.window_handles()[0];
    let mut client = RECT::default();
    // SAFETY: valid out pointer for our own window.
    unsafe { GetClientRect(hwnd, &mut client).unwrap() };
    let inside = POINT { x: client.right / 2, y: client.bottom / 2 };
    assert!(
        inside.x > client.left
            && inside.y > client.top
            && inside.x < client.right
            && inside.y < client.bottom,
        "{inside:?} in {client:?}"
    );
    // Point the injected cursor directly inside the client area: if `cancel_presses` did not clamp
    // the release outside the client rect, it would land on `Clickable` and fire `on_click`.
    platform.set_test_client_cursor(Some(inside));
    let at = ((inside.y as isize) << 16) | (inside.x as isize & 0xFFFF);
    let take = || (std::mem::take(&mut *clicks.borrow_mut()), std::mem::take(&mut *up_out.borrow_mut()));

    send(hwnd, WM_MOUSEMOVE, 0, at);
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert_eq!(take(), (1, 0), "a press released in the window clicks");

    draw();
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    // SAFETY: ending this thread's capture, as Windows does.
    unsafe { ReleaseCapture().unwrap() };
    assert_eq!(take(), (0, 1), "the capture went away");

    draw();
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(take(), (0, 1), "WM_CANCELMODE");

    draw();
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_ACTIVATE, WA_INACTIVE as usize, 0);
    pump();
    assert_eq!(take(), (0, 1), "the window was deactivated");
    // The cancelled press is over: a release arriving later clicks nothing either.
    draw();
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert_eq!(take(), (0, 0), "a stray release");

    // With the cursor already outside the client area, cancellation still clears the press without
    // clicking.
    platform.set_test_client_cursor(Some(POINT { x: client.right + 20, y: client.bottom + 20 }));
    draw();
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(take(), (0, 1), "cancelled with cursor outside client rect");

    // Ordinary click still succeeds after cancelled presses.
    platform.set_test_client_cursor(Some(inside));
    draw();
    send(hwnd, WM_MOUSEMOVE, 0, at);
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_LBUTTONUP, 0, at);
    assert_eq!(take(), (1, 0), "ordinary click still succeeds after cancellations");

    destroy(hwnd);
    drop(app);
}

/// `cancel_presses` clamps an in-client cursor just outside the client rect (`(-1, -1)` in device
/// pixels) and keeps an already-outside cursor coordinate as reported.
#[test]
fn cancelled_press_reports_release_outside_client_rect() {
    let platform = platform();
    let window = platform.open_windows_window(hidden_params()).unwrap();
    let hwnd = window.hwnd();
    let scale = window.scale_factor();
    let mut client = RECT::default();
    // SAFETY: valid out pointer for our own window.
    unsafe { GetClientRect(hwnd, &mut client).unwrap() };
    let inside = POINT { x: client.right / 2, y: client.bottom / 2 };
    let outside = POINT { x: client.right + 25, y: client.bottom + 15 };
    let at = ((inside.y as isize) << 16) | (inside.x as isize & 0xFFFF);

    let ups: Rc<RefCell<Vec<Point<Pixels>>>> = Rc::default();
    window.on_input(Box::new({
        let ups = ups.clone();
        move |input| {
            if let PlatformInput::MouseUp(e) = input {
                ups.borrow_mut().push(e.position);
            }
            DispatchEventResult::default()
        }
    }));

    platform.set_test_client_cursor(Some(inside));
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(
        std::mem::take(&mut *ups.borrow_mut()),
        vec![super::util::logical_point((client.left - 1) as f32, (client.top - 1) as f32, scale)],
        "cursor inside client rect is clamped just outside it"
    );

    // The right/bottom boundary is inclusive in Bounds::contains, so it must be clamped too.
    platform.set_test_client_cursor(Some(POINT { x: client.right, y: client.bottom }));
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(
        std::mem::take(&mut *ups.borrow_mut()),
        vec![super::util::logical_point((client.left - 1) as f32, (client.top - 1) as f32, scale)],
        "cursor on right/bottom client edge is clamped just outside it"
    );

    platform.set_test_client_cursor(Some(outside));
    send(hwnd, WM_LBUTTONDOWN, 0, at);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(
        std::mem::take(&mut *ups.borrow_mut()),
        vec![super::util::logical_point(outside.x as f32, outside.y as f32, scale)],
        "cursor outside client rect is preserved"
    );

    destroy(hwnd);
}

/// A focusable field that records what the platform types into it; `ctrl-k` is bound and handled.
/// `marked` is the length of the marked (composed) text, 0 for none.
struct Field {
    focus: FocusHandle,
    typed: Rc<RefCell<Vec<String>>>,
    marked: Rc<std::cell::Cell<usize>>,
}

impl Render for Field {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let (focus, typed, marked) = (self.focus.clone(), self.typed.clone(), self.marked.clone());
        div()
            .key_context("Field")
            .track_focus(&self.focus)
            .on_action(|_: &Handled, _, _| {})
            .size_full()
            .child(
                canvas(
                    |_, _, _| (),
                    move |_, _, window, cx| window.handle_input(&focus, Recorder(typed, marked), cx),
                )
                .size_full(),
            )
    }
}

struct Recorder(Rc<RefCell<Vec<String>>>, Rc<std::cell::Cell<usize>>);

impl InputHandler for Recorder {
    fn selected_text_range(&mut self, _: bool, _: &mut Window, _: &mut App) -> Option<UTF16Selection> {
        None
    }
    fn marked_text_range(&mut self, _: &mut Window, _: &mut App) -> Option<std::ops::Range<usize>> {
        let n = self.1.get();
        (n > 0).then_some(0..n)
    }
    fn text_for_range(
        &mut self,
        _: std::ops::Range<usize>,
        _: &mut Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<String> {
        None
    }
    fn replace_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        text: &str,
        _: &mut Window,
        _: &mut App,
    ) {
        self.1.set(0);
        self.0.borrow_mut().push(text.to_string());
    }
    fn replace_and_mark_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        text: &str,
        _: Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) {
        self.1.set(text.encode_utf16().count());
        self.0.borrow_mut().push(format!("mark {text}"));
    }
    fn unmark_text(&mut self, _: &mut Window, _: &mut App) {}
    fn bounds_for_range(
        &mut self,
        _: std::ops::Range<usize>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<Bounds<Pixels>> {
        None
    }
    fn character_index_for_point(&mut self, _: Point<Pixels>, _: &mut Window, _: &mut App) -> Option<usize> {
        None
    }
}

/// Typed text comes from Windows' own translation (`WM_CHAR`, dead keys), which knows Caps Lock, AltGr
/// and dead keys, and never from the keystroke's `key_char`. The core used to insert `key_char` for an
/// unhandled key and stop it there, so Caps Lock + A typed "a" (and AltGr+Q on a German layout "q").
#[test]
fn typed_text_is_windows_translation() {
    let platform = platform();
    let app = App::new_app(platform.clone(), Arc::new(()));
    let typed: Rc<RefCell<Vec<String>>> = Rc::default();
    let handle = app
        .update(|cx| {
            cx.bind_keys([KeyBinding::new("ctrl-k", Handled, Some("Field"))]);
            let focus = cx.focus_handle();
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds())),
                focus: false,
                show: false,
                ..Default::default()
            };
            let typed = typed.clone();
            cx.open_window(options, move |window, cx| {
                window.focus(&focus);
                cx.new(|_| Field { focus, typed, marked: Rc::default() })
            })
        })
        .unwrap();
    app.update(|cx| cx.update_window(handle.into(), |_, window, cx| window.draw(cx))).unwrap();
    let hwnd = platform.window_handles()[0];
    let take = || std::mem::take(&mut *typed.borrow_mut());

    press(hwnd, VK_A, &[], &[]);
    assert_eq!(take(), ["a"]);
    press(hwnd, VK_A, &[VK_SHIFT, VK_LSHIFT], &[]);
    assert_eq!(take(), ["A"]);
    press(hwnd, VK_A, &[], &[VK_CAPITAL]);
    assert_eq!(take(), ["A"], "Caps Lock");
    press(hwnd, VK_A, &[VK_SHIFT, VK_LSHIFT], &[VK_CAPITAL]);
    assert_eq!(take(), ["a"], "Caps Lock and Shift");
    press(hwnd, VK_K, &[VK_CONTROL, VK_LCONTROL], &[]);
    assert_eq!(take(), Vec::<String>::new(), "a handled shortcut types nothing");
    for vk in [VK_RETURN, VK_TAB, VK_BACK, VK_ESCAPE] {
        press(hwnd, vk, &[], &[]);
    }
    assert_eq!(take(), Vec::<String>::new(), "no control characters");

    // AltGr (right Alt, which Windows reports with left Ctrl): whatever the layout makes of Q, it is
    // never the plain "q"; the "@" a German layout's WM_CHAR carries is typed.
    press(hwnd, VK_Q, &[VK_CONTROL, VK_LCONTROL, VK_MENU, VK_RMENU], &[]);
    post(hwnd, WM_CHAR, '@' as usize, 1);
    pump();
    let text = take();
    assert!(!text.iter().any(|t| t == "q") && text.last().map(String::as_str) == Some("@"), "{text:?}");

    // A dead key composes: the accent is marked, then the composed character replaces it.
    post(hwnd, WM_DEADCHAR, '\u{B4}' as usize, 1);
    post(hwnd, WM_CHAR, '\u{E9}' as usize, 1);
    pump();
    assert_eq!(take(), ["mark \u{B4}", "\u{E9}"]);

    // A flagless WM_IME_COMPOSITION (the composition is gone) drops only composed text: with nothing
    // marked it replaces nothing (it used to replace the selection with "", stress d5-window-input-1);
    // with an accent marked, that goes.
    send(hwnd, WM_IME_COMPOSITION, 0, 0);
    assert_eq!(take(), Vec::<String>::new(), "nothing composed: the text stays");
    post(hwnd, WM_DEADCHAR, '\u{B4}' as usize, 1);
    pump();
    send(hwnd, WM_IME_COMPOSITION, 0, 0);
    assert_eq!(take(), ["mark \u{B4}", ""], "the composed accent is dropped");

    // A character outside the BMP arrives as a surrogate pair in two WM_CHARs.
    post(hwnd, WM_CHAR, 0xD83D, 1);
    post(hwnd, WM_CHAR, 0xDE00, 1);
    pump();
    assert_eq!(take(), ["\u{1F600}"]);

    destroy(hwnd);
    drop(app);
}

/// Off screen, a window of the process that reaches a display is moved off every display: the
/// file dialog that had no owner opened centred on the primary display, on the screen of whoever was
/// using the PC. Checked on a hidden window (never shown), where the parking hook's own move happens.
#[test]
fn park_moves_a_window_on_a_display_off_every_display() {
    use ::windows::Win32::UI::HiDpi::{
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetThreadDpiAwarenessContext,
    };
    // Every rectangle in device pixels, in one awareness for the whole test: this thread's own, whatever
    // the process's. Without it the size read back scaled about one run in 40 (450 x 300 for 300 x 200
    // at 150 %), most likely because a test beside this one makes a `WindowsPlatform`, which turns the
    // whole process per-monitor aware meanwhile.
    // SAFETY: this thread's own setting, put back at the end.
    let before = unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    let make = |x: i32, y: i32, style: WINDOW_STYLE, parent: Option<HWND>| {
        // SAFETY: a plain hidden STATIC window (no WS_VISIBLE), destroyed below.
        unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!(""),
                style,
                x,
                y,
                300,
                200,
                parent,
                None,
                None,
                None,
            )
        }
        .expect("a hidden window")
    };
    // The primary display's top left is (0, 0).
    let (x, y) = (100, 100);
    let on_screen = make(x, y, WS_POPUP, None);
    assert!(super::offscreen::on_a_display(&super::offscreen::window_rect(on_screen)), "starts on a display");
    assert!(super::offscreen::park(on_screen), "parked");
    let r = super::offscreen::window_rect(on_screen);
    assert!(!super::offscreen::on_a_display(&r), "now off every display: {r:?}");
    assert_eq!((r.right - r.left, r.bottom - r.top), (300, 200), "same size");
    // SAFETY: a plain query.
    assert!(!unsafe { IsWindowVisible(on_screen) }.as_bool(), "still hidden");
    assert!(!super::offscreen::park(on_screen), "a parked window stays where it is");
    // A child moves with its parent: left alone.
    let parent = make(x, y, WS_POPUP, None);
    let child = make(10, 10, WS_CHILD, Some(parent));
    assert!(!super::offscreen::park(child));
    for hwnd in [child, parent, on_screen] {
        destroy(hwnd);
    }
    // SAFETY: as above.
    unsafe { SetThreadDpiAwarenessContext(before) };
}

/// The backend's file dialogs, Explorer and browser go through the shared off-screen gate, built from
/// `FOLO_UI_OFFSCREEN`: closed, nothing reaches the shell (a recorder stands in for it, so nothing real
/// is launched either way).
#[test]
fn off_screen_the_backend_launches_no_dialog_explorer_or_browser() {
    use crate::platform::services::{Recorder, call_each};
    let mut platform = WindowsPlatform::new(Arc::new(FixedTextSystem::new())).expect("the Win32 platform");
    let recorder = Recorder::default();
    let built = platform.launch_with(true, Box::new(recorder.clone()));
    assert_eq!(built, super::offscreen(), "the gate follows FOLO_UI_OFFSCREEN");
    let p: &dyn Platform = &platform;
    let refused = call_each(
        |u| p.open_url(u),
        |path| p.reveal_path(path),
        |o| p.prompt_for_paths(o),
        |d, n| p.prompt_for_new_path(d, n),
    );
    assert_eq!(refused, (true, true), "both dialogs answer with the refusal");
    assert!(recorder.0.borrow().is_empty(), "launched: {:?}", recorder.0.borrow());
    // Open, the same calls reach it.
    platform.launch_with(false, Box::new(recorder.clone()));
    let p: &dyn Platform = &platform;
    call_each(
        |u| p.open_url(u),
        |path| p.reveal_path(path),
        |o| p.prompt_for_paths(o),
        |d, n| p.prompt_for_new_path(d, n),
    );
    assert_eq!(recorder.0.borrow().len(), 4);
}

/// Deactivating the window or cancelling mode disarms a held caption button (`HTCLOSE`) so a later
/// non-client release does not close the window.
#[test]
fn deactivation_and_cancel_mode_disarm_caption_button_press() {
    let platform = platform();
    let window = platform.open_windows_window(hidden_params()).unwrap();
    let hwnd = window.hwnd();
    let close_asks = Rc::new(std::cell::Cell::new(0usize));
    let cancelled_ups = Rc::new(std::cell::Cell::new(0usize));
    window.on_should_close(Box::new({
        let close_asks = close_asks.clone();
        move || {
            close_asks.set(close_asks.get() + 1);
            false
        }
    }));
    window.on_input(Box::new({
        let cancelled_ups = cancelled_ups.clone();
        move |input| {
            if let PlatformInput::MouseUp(e) = input
                && e.button == MouseButton::Left
                && e.position.x < px(0.)
            {
                cancelled_ups.set(cancelled_ups.get() + 1);
            }
            DispatchEventResult { propagate: true, default_prevented: false }
        }
    }));

    // Baseline: pressing and releasing HTCLOSE triggers WM_CLOSE -> should_close.
    send(hwnd, WM_NCLBUTTONDOWN, HTCLOSE as usize, 0);
    send(hwnd, WM_NCLBUTTONUP, HTCLOSE as usize, 0);
    pump();
    assert_eq!(close_asks.replace(0), 1, "press+release on HTCLOSE asks to close");

    // Deactivating while HTCLOSE is held disarms it and emits an off-viewport release for active styles.
    let mut client = RECT::default();
    // SAFETY: valid out pointer for our own window.
    unsafe { GetClientRect(hwnd, &mut client).unwrap() };
    platform.set_test_client_cursor(Some(POINT { x: client.right / 2, y: client.bottom / 2 }));
    cancelled_ups.set(0);
    send(hwnd, WM_NCLBUTTONDOWN, HTCLOSE as usize, 0);
    send(hwnd, WM_ACTIVATE, WA_INACTIVE as usize, 0);
    assert_eq!(cancelled_ups.replace(0), 1, "deactivation emits off-viewport release");
    send(hwnd, WM_NCLBUTTONUP, HTCLOSE as usize, 0);
    pump();
    assert_eq!(close_asks.replace(0), 0, "deactivation disarms HTCLOSE");

    // WM_CANCELMODE while HTCLOSE is held also disarms it and emits an off-viewport release.
    cancelled_ups.set(0);
    send(hwnd, WM_NCLBUTTONDOWN, HTCLOSE as usize, 0);
    send(hwnd, WM_CANCELMODE, 0, 0);
    assert_eq!(cancelled_ups.replace(0), 1, "WM_CANCELMODE emits off-viewport release");
    send(hwnd, WM_NCLBUTTONUP, HTCLOSE as usize, 0);
    pump();
    assert_eq!(close_asks.replace(0), 0, "WM_CANCELMODE disarms HTCLOSE");

    destroy(hwnd);
}
