//! A real-window check of the Win32 backend that the user never sees: `FOLO_UI_OFFSCREEN=1` is forced
//! on, so the 960x620 window opens outside every display, never activated, with no taskbar button.
//! It presents a test pattern and drives the window with posted messages (never real input), checking
//! what the callbacks receive. Prints PASS / FAIL per step and exits 0 only if all pass.
//!
//!   cargo run -p folo-ui --release --example win_smoke            # all checks and the timings
//!   cargo run -p folo-ui --release --example win_smoke -- --quick # skip the 5 s idle-CPU check

#[cfg(not(windows))]
fn main() {
    eprintln!("win_smoke runs on Windows only");
}

#[cfg(windows)]
mod common;
#[cfg(windows)]
mod dialog;
#[cfg(windows)]
mod input;
#[cfg(windows)]
mod system;
#[cfg(windows)]
mod window;

#[cfg(windows)]
use ::windows::Win32::UI::WindowsAndMessaging::*;
#[cfg(windows)]
use common::*;
#[cfg(windows)]
use dialog::*;
#[cfg(windows)]
use folo_ui::windows::{WindowsPlatform, WindowsWindow, ms_since_process_start};
#[cfg(windows)]
use folo_ui::*;
#[cfg(windows)]
use input::*;
#[cfg(windows)]
use std::cell::{Cell, RefCell};
#[cfg(windows)]
use std::rc::Rc;
#[cfg(windows)]
use std::sync::Arc;
#[cfg(windows)]
use std::time::{Duration, Instant};
#[cfg(windows)]
use system::*;
#[cfg(windows)]
use window::*;

#[cfg(windows)]
fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    // `--onscreen-timings`: only the present timings, on a window at a normal place, which is the
    // only way the window DC is not clipped away. Refused unless the caller says it runs this on a
    // private desktop nobody looks at (FOLO_SMOKE_SANDBOX=1).
    let onscreen = std::env::args().any(|a| a == "--onscreen-timings");
    // `--dialog`: the modal file dialog check, which shows a dialog: same rule.
    let dialog = std::env::args().any(|a| a == "--dialog");
    if (onscreen || dialog) && std::env::var_os("FOLO_SMOKE_SANDBOX").is_none_or(|v| v != "1") {
        eprintln!(
            "--onscreen-timings and --dialog show windows: run them on a private desktop with FOLO_SMOKE_SANDBOX=1"
        );
        std::process::exit(2);
    }
    if !onscreen {
        // SAFETY: set before any other thread exists; the backend reads it once.
        unsafe { std::env::set_var("FOLO_UI_OFFSCREEN", "1") };
    }
    let started_ms = ms_since_process_start();
    let platform = Rc::new(WindowsPlatform::new(Arc::new(NoText)).expect("platform"));
    let checks = Rc::new(RefCell::new(Checks { fails: 0 }));
    let quit_ran = Rc::new(Cell::new(false));
    platform.on_quit(Box::new({
        let quit_ran = quit_ran.clone();
        move || quit_ran.set(true)
    }));
    let p = platform.clone();
    let c = checks.clone();
    platform.run(Box::new(move || {
        run_checks(&p, &mut c.borrow_mut(), Mode { quick, onscreen, dialog }, started_ms)
    }));
    let mut checks = checks.borrow_mut();
    checks.check("run returned after the last window closed; on_quit ran", quit_ran.get(), "");
    let fails = checks.fails;
    drop(checks);
    drop(platform);
    println!(
        "{}",
        if fails == 0 { "all checks passed".to_string() } else { format!("{fails} check(s) failed") }
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}

#[cfg(windows)]
/// What to run.
#[cfg(windows)]
struct Mode {
    quick: bool,
    onscreen: bool,
    dialog: bool,
}

#[cfg(windows)]
fn run_checks(platform: &Rc<WindowsPlatform>, checks: &mut Checks, mode: Mode, started_ms: f64) {
    let Mode { quick, onscreen, dialog: dialog_only } = mode;
    let fg0 = foreground();
    let log = Rc::new(RefCell::new(Log::default()));
    let primary = platform.primary_display().expect("display");
    let bounds = Bounds::new(primary.bounds().origin + point(px(100.), px(100.)), size(px(960.), px(620.)));
    let params = WindowParams {
        bounds,
        titlebar: Some(TitlebarOptions {
            title: Some("win_smoke".into()),
            appears_transparent: true,
            traffic_light_position: None,
        }),
        kind: WindowKind::Normal,
        is_movable: true,
        is_resizable: true,
        is_minimizable: true,
        focus: false,
        show: true,
        display_id: Some(primary.id()),
        window_min_size: Some(size(px(320.), px(200.))),
        app_id: None,
    };
    let window: Rc<RefCell<WindowsWindow>> =
        Rc::new(RefCell::new(platform.open_windows_window(params).expect("window")));
    let hwnd = window.borrow().hwnd();
    let scale = window.borrow().scale_factor();
    let frame =
        Rc::new(RefCell::new(Frame { buffer: Framebuffer::default(), seed: 1, animate: 0, damage: None }));
    let first_present_ms = Rc::new(Cell::new(f64::NAN));
    // The core's part, faked: draw a pattern of the client size and present it.
    {
        let w = window.borrow();
        let (log2, frame2, window2, first) =
            (log.clone(), frame.clone(), Rc::downgrade(&window), first_present_ms.clone());
        w.on_request_frame(Box::new(move |options| {
            let mut l = log2.borrow_mut();
            l.frames.push((Instant::now(), options));
            if l.in_resize {
                l.frame_inside_resize += 1;
            }
            drop(l);
            let Some(window) = window2.upgrade() else { return };
            let mut f = frame2.borrow_mut();
            let (cw, ch) = client_size(hwnd);
            if f.buffer.width != cw as u32 || f.buffer.height != ch as u32 {
                f.buffer.resize(cw as u32, ch as u32);
                let seed = f.seed;
                pattern(&mut f.buffer, seed);
            }
            let full = [Bounds::new(
                point(DevicePixels(0), DevicePixels(0)),
                size(DevicePixels(cw), DevicePixels(ch)),
            )];
            let damage = f.damage.take().map(|d| [d]).unwrap_or(full);
            window.borrow_mut().present(&f.buffer, &damage);
            if first.get().is_nan() {
                first.set(ms_since_process_start());
            }
            if f.animate > 0 {
                f.animate -= 1;
                window.borrow().request_frame();
            }
        }));
        let l = log.clone();
        w.on_input(Box::new(move |input| {
            // Ctrl+K is "handled by the app"; everything else propagates.
            let handled = matches!(&input, PlatformInput::KeyDown(k) if k.keystroke.key == "k" && k.keystroke.modifiers.control);
            l.borrow_mut().inputs.push(input);
            DispatchEventResult { propagate: !handled, default_prevented: false }
        }));
        let l = log.clone();
        w.on_resize(Box::new(move |s, f| {
            l.borrow_mut().resizes.push((s, f));
            l.borrow_mut().in_resize = true;
        }));
        let l = log.clone();
        w.on_hover_status_change(Box::new(move |h| l.borrow_mut().hover.push(h)));
        // The app's caption areas: the top 4 px resize, then a 40 px title bar with three buttons.
        let l = log.clone();
        w.on_hit_test_window_control(Box::new(move || {
            let l = l.borrow();
            let p = l.inputs.iter().rev().find_map(|i| match i {
                PlatformInput::MouseMove(m) => Some(m.position),
                _ => None,
            })?;
            let (x, y) = (p.x / px(1.), p.y / px(1.));
            match (x, y) {
                (_, y) if !(4. ..40.).contains(&y) => None,
                (x, _) if (800. ..860.).contains(&x) => Some(WindowControlArea::Min),
                (x, _) if (860. ..920.).contains(&x) => Some(WindowControlArea::Max),
                (x, _) if x >= 920. => Some(WindowControlArea::Close),
                _ => Some(WindowControlArea::Drag),
            }
        }));
        w.on_should_close(Box::new(|| true));
    }
    window.borrow_mut().set_input_handler(Box::new(Handler(log.clone())));

    // ---- First frame.
    let presented = pump_until(Duration::from_secs(30), || !first_present_ms.get().is_nan());
    let since_open = first_present_ms.get() - started_ms;
    checks.check(
        "first frame presented",
        presented,
        format!("{:.1} ms after process start ({since_open:.1} ms after main)", first_present_ms.get()),
    );
    if dialog_only {
        dialog(checks, platform, &window, &log, &frame);
        post(hwnd, WM_CLOSE, 0, 0);
        return;
    }
    if onscreen {
        timings(checks, hwnd, &window, &frame, true);
        post(hwnd, WM_CLOSE, 0, 0);
        return;
    }
    let r = rect(hwnd);
    checks.check(
        "window is outside every monitor",
        off_every_monitor(r),
        format!("({}, {}) - ({}, {})", r.left, r.top, r.right, r.bottom),
    );
    let (cw, ch) = client_size(hwnd);
    let want = (960. * scale).round() as i32;
    checks.check(
        "client area is 960x620 logical",
        cw == want && ch == (620. * scale).round() as i32,
        format!("{cw}x{ch} device px at scale {scale}"),
    );
    checks.check(
        "not activated, foreground unchanged",
        foreground() == fg0 && !window.borrow().is_active(),
        "",
    );
    checks.check(
        "window_bounds is the requested place",
        window.borrow().window_bounds()
            == WindowBounds::Windowed(window.borrow().window_bounds().get_bounds())
            && (window.borrow().window_bounds().get_bounds().size.width / px(1.) - 960.).abs() < 0.01,
        format!("{:?}", window.borrow().window_bounds()),
    );

    // ---- PrintWindow equals what was presented, after a full and a partial present.
    let shot = print_client(hwnd, cw, ch);
    let same = shot == frame.borrow().buffer.pixels;
    let diff = mismatch(&shot, &frame.borrow().buffer);
    checks.check(
        "PrintWindow(3) of the client equals the presented frame",
        same,
        format!("{} px{diff}", shot.len()),
    );
    {
        let mut f = frame.borrow_mut();
        f.seed = 99;
        let mut next = f.buffer.clone();
        pattern(&mut next, 99);
        // Only the 64x64 square at (128, 64) changes and is presented.
        for y in 64..128 {
            for x in 128..192 {
                let v = next.pixel(x, y);
                f.buffer.set_pixel(x, y, v);
            }
        }
        f.damage = Some(Bounds::new(
            point(DevicePixels(128), DevicePixels(64)),
            size(DevicePixels(64), DevicePixels(64)),
        ));
    }
    window.borrow().request_frame();
    let n = log.borrow().frames.len();
    pump_until(Duration::from_secs(30), || log.borrow().frames.len() > n);
    let shot = print_client(hwnd, cw, ch);
    checks.check(
        "a 64x64 damage present shows up, the rest is kept",
        shot == frame.borrow().buffer.pixels,
        "",
    );

    hit_tests(checks, hwnd, scale, &log);
    mouse(checks, hwnd, scale, &log);
    keys(checks, hwnd, &log);
    resize(checks, hwnd, scale, &log, &window);
    modes(checks, hwnd, &window, &log, fg0);
    pacing(checks, &window, &log, &frame);
    dispatcher(checks, platform);
    if !quick {
        idle(checks, &log);
    }
    timings(checks, hwnd, &window, &frame, false);
    checks.check(
        "still not activated, foreground unchanged",
        foreground() == fg0,
        format!("{:?} (was {fg0:?}, window {hwnd:?})", foreground()),
    );

    // ---- Close: WM_CLOSE asks should_close, destroys the window, and the loop ends.
    post(hwnd, WM_CLOSE, 0, 0);
}
