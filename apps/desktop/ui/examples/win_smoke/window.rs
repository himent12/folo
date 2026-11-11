//! Resize, maximize / restore / full screen, and frame pacing.

use crate::common::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use folo_ui::windows::WindowsWindow;
use folo_ui::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

pub fn resize(
    checks: &mut Checks,
    hwnd: HWND,
    scale: f32,
    log: &Rc<RefCell<Log>>,
    window: &Rc<RefCell<WindowsWindow>>,
) {
    let r = rect(hwnd);
    log.borrow_mut().resizes.clear();
    log.borrow_mut().frame_inside_resize = 0;
    let grow = (100. * scale) as i32;
    // SAFETY: resizing our own window without activating it.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            r.right - r.left + grow,
            r.bottom - r.top + grow,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
        .unwrap();
    }
    // No pumping: both must have happened inside SetWindowPos.
    let l = log.borrow();
    let ok = l.resizes.last().is_some_and(|(s, f)| {
        (s.width / px(1.) - 1060.).abs() < 1. && (s.height / px(1.) - 720.).abs() < 1. && *f == scale
    });
    checks.check("resize reports the new logical size and scale", ok, format!("{:?}", l.resizes));
    let expose = l.frames.last().is_some_and(|(_, o)| o.require_presentation);
    checks.check(
        "resize draws synchronously (require_presentation)",
        l.frame_inside_resize >= 1 && expose,
        format!("{} frame(s) inside", l.frame_inside_resize),
    );
    drop(l);
    log.borrow_mut().in_resize = false;
    let (cw, ch) = client_size(hwnd);
    checks.check(
        "content_size follows",
        window.borrow().content_size() == size(px(cw as f32 / scale), px(ch as f32 / scale)),
        "",
    );
    // SAFETY: back to the original size.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            r.right - r.left,
            r.bottom - r.top,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
        .unwrap()
    };
    log.borrow_mut().in_resize = false;
    pump();
}

pub fn modes(checks: &mut Checks, hwnd: HWND, window: &Rc<RefCell<WindowsWindow>>, fg0: HWND) {
    let restored = window.borrow().window_bounds();
    let b = restored.get_bounds();
    window.borrow().zoom();
    let maxed = pump_until(Duration::from_secs(3), || window.borrow().is_maximized());
    let wb = window.borrow().window_bounds();
    checks.check("zoom maximizes", maxed, "");
    checks.check(
        "maximized window_bounds keeps the restored bounds",
        wb == WindowBounds::Maximized(b),
        format!("{wb:?}"),
    );
    let (r, fg) = (rect(hwnd), foreground());
    checks.check(
        "maximized stays off screen, foreground unchanged",
        off_every_monitor(r) && fg == fg0,
        format!(
            "({}, {}) - ({}, {}), foreground {fg:?} (was {fg0:?}, window {hwnd:?})",
            r.left, r.top, r.right, r.bottom
        ),
    );
    window.borrow().zoom();
    let back = pump_until(Duration::from_secs(3), || !window.borrow().is_maximized());
    pump_until(Duration::from_millis(200), || false);
    let wb = window.borrow().window_bounds();
    checks.check(
        "zoom again restores, window_bounds windowed as before",
        back && wb == restored,
        format!("{wb:?}"),
    );

    window.borrow().toggle_fullscreen();
    let full = pump_until(Duration::from_secs(3), || window.borrow().is_fullscreen());
    pump_until(Duration::from_millis(200), || false);
    let display = window.borrow().display().unwrap().bounds();
    let wb = window.borrow().window_bounds();
    let content = window.borrow().content_size();
    checks.check(
        "full screen covers the display's size",
        full && (content.width / px(1.) - display.size.width / px(1.)).abs() < 1.,
        format!("{content:?} vs {:?}", display.size),
    );
    checks.check(
        "full screen window_bounds keeps the restored bounds",
        wb == WindowBounds::Fullscreen(b),
        format!("{wb:?}"),
    );
    let (r, fg) = (rect(hwnd), foreground());
    checks.check(
        "full screen stays off screen, foreground unchanged",
        off_every_monitor(r) && fg == fg0,
        format!(
            "({}, {}) - ({}, {}), foreground {fg:?} (was {fg0:?}, window {hwnd:?})",
            r.left, r.top, r.right, r.bottom
        ),
    );
    window.borrow().toggle_fullscreen();
    let left = pump_until(Duration::from_secs(3), || !window.borrow().is_fullscreen());
    pump_until(Duration::from_millis(200), || false);
    let wb = window.borrow().window_bounds();
    checks.check(
        "leaving full screen restores style and place",
        left && wb == restored && window.borrow().content_size() == b.size,
        format!("{wb:?}"),
    );
}

pub fn pacing(
    checks: &mut Checks,
    window: &Rc<RefCell<WindowsWindow>>,
    log: &Rc<RefCell<Log>>,
    frame: &Rc<RefCell<Frame>>,
) {
    // Coalescing: many requests before the next refresh are one frame.
    pump_until(Duration::from_millis(100), || false);
    let n = log.borrow().frames.len();
    for _ in 0..50 {
        window.borrow().request_frame();
    }
    pump_until(Duration::from_millis(150), || false);
    let got = log.borrow().frames.len() - n;
    checks.check("50 requests before a refresh draw once", got == 1, format!("{got} frame(s)"));

    // An animation: each frame asks for the next; ticks follow the compositor.
    let n = log.borrow().frames.len();
    frame.borrow_mut().animate = 240;
    window.borrow().request_frame();
    pump_until(Duration::from_secs(5), || frame.borrow().animate == 0);
    pump_until(Duration::from_millis(50), || false);
    let l = log.borrow();
    let times: Vec<Instant> = l.frames[n..].iter().map(|(t, _)| *t).collect();
    let intervals: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1000.).collect();
    let mean = intervals.iter().sum::<f64>() / intervals.len().max(1) as f64;
    let mut sorted = intervals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.);
    let min = sorted.first().copied().unwrap_or(0.);
    let period = refresh_ms();
    checks.check(
        "animation ticks at the display refresh",
        (median - period).abs() < period * 0.25 && min > period * 0.4,
        format!("{} frames, median {median:.2} ms, mean {mean:.2} ms, min {min:.2} ms; refresh {period:.2} ms ({:.1} Hz)", times.len(), 1000. / period),
    );
}

fn refresh_ms() -> f64 {
    use ::windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmGetCompositionTimingInfo};
    let mut info =
        DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
    // SAFETY: a sized DWM_TIMING_INFO.
    let _ = unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut info) };
    let (n, d) = (info.rateRefresh.uiNumerator as f64, info.rateRefresh.uiDenominator as f64);
    if n > 0. { 1000. * d / n } else { 16.67 }
}
