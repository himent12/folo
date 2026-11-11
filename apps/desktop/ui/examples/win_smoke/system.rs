//! The dispatcher, idleness, and present timings.

use crate::common::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::BOOL;
use folo_ui::windows::{WindowsPlatform, WindowsWindow, process_cpu_time};
use folo_ui::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryProcessCycleTime(process: HANDLE, cycles: *mut u64) -> BOOL;
    fn GetCurrentProcess() -> HANDLE;
}

/// CPU cycles all threads of this process have used so far (exact, unlike GetProcessTimes' ticks).
fn cycles() -> u64 {
    let mut cycles = 0u64;
    // SAFETY: the current-process pseudo handle and a valid out pointer.
    let _ = unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut cycles) };
    cycles
}

pub fn dispatcher(checks: &mut Checks, platform: &Rc<WindowsPlatform>) {
    let d = platform.dispatcher();
    // 1000 background jobs.
    let done = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    for _ in 0..1000 {
        let (done, dd) = (done.clone(), d.clone());
        let (runnable, task) = async_task::spawn(
            async move {
                done.fetch_add(1, Ordering::SeqCst);
            },
            move |r| dd.dispatch(r),
        );
        task.detach();
        runnable.schedule();
    }
    let ok = pump_until(Duration::from_secs(5), || done.load(Ordering::SeqCst) == 1000);
    checks.check(
        "1000 background jobs run",
        ok,
        format!(
            "{:.1} ms, {} pool thread(s)",
            start.elapsed().as_secs_f64() * 1000.,
            platform.windows_dispatcher().pool_threads()
        ),
    );

    // Main-thread jobs, queued from the main thread and from a background thread, run in order.
    let order = Rc::new(RefCell::new(Vec::new()));
    for i in 0..100 {
        let (order, dd) = (order.clone(), d.clone());
        let (runnable, task) = async_task::spawn_local(async move { order.borrow_mut().push(i) }, move |r| {
            dd.dispatch_on_main_thread(r)
        });
        task.detach();
        runnable.schedule();
    }
    let ok = pump_until(Duration::from_secs(2), || order.borrow().len() == 100);
    checks.check(
        "main-thread jobs run in order on the main thread",
        ok && *order.borrow() == (0..100).collect::<Vec<_>>() && d.is_main_thread(),
        "",
    );
    let from_bg = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let dd = d.clone();
    let fb = from_bg.clone();
    std::thread::spawn(move || {
        for i in 0..100 {
            let (fb, d2) = (fb.clone(), dd.clone());
            let main = dd.clone();
            let (runnable, task) =
                async_task::spawn(async move { fb.lock().push((i, main.is_main_thread())) }, move |r| {
                    d2.dispatch_on_main_thread(r)
                });
            task.detach();
            runnable.schedule();
        }
    })
    .join()
    .unwrap();
    let ok = pump_until(Duration::from_secs(2), || from_bg.lock().len() == 100);
    let v = from_bg.lock().clone();
    checks.check(
        "main-thread jobs from a background thread run in order there",
        ok && v.iter().enumerate().all(|(i, &(j, main))| i == j && main),
        "",
    );

    // Timers: one thread, deadline order.
    let fired = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let start = Instant::now();
    for (i, ms) in [(0, 45u64), (1, 15), (2, 30), (3, 60)] {
        let (fired, dd) = (fired.clone(), d.clone());
        let (runnable, task) =
            async_task::spawn(async move { fired.lock().push((i, start.elapsed())) }, move |r| {
                dd.dispatch_after(Duration::from_millis(ms), r)
            });
        task.detach();
        runnable.schedule();
    }
    let ok = pump_until(Duration::from_secs(2), || fired.lock().len() == 4);
    let v = fired.lock().clone();
    let order: Vec<i32> = v.iter().map(|(i, _)| *i).collect();
    let late = v.iter().zip([15u64, 30, 45, 60]).all(|((_, t), ms)| *t >= Duration::from_millis(ms));
    checks.check(
        "timers fire in deadline order, not early",
        ok && order == [1, 2, 0, 3] && late,
        format!("{v:?}"),
    );
}

pub fn idle(checks: &mut Checks, log: &Rc<RefCell<Log>>) {
    pump_until(Duration::from_millis(300), || false);
    let frames = log.borrow().frames.len();
    let cpu0 = process_cpu_time();
    let cycles0 = cycles();
    let t0 = Instant::now();
    let mut messages = 0usize;
    // Block in the message wait for 5 s, dispatching whatever arrives (nothing should).
    while t0.elapsed() < Duration::from_secs(5) {
        let left = Duration::from_secs(5).saturating_sub(t0.elapsed());
        // SAFETY: waiting on this thread's queue.
        unsafe { MsgWaitForMultipleObjects(None, false, left.as_millis() as u32, QS_ALLINPUT) };
        let mut msg = MSG::default();
        // SAFETY: as in `pump`.
        unsafe {
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                messages += 1;
                DispatchMessageW(&msg);
            }
        }
    }
    let cpu = process_cpu_time() - cpu0;
    let cycles = cycles() - cycles0;
    let frames = log.borrow().frames.len() - frames;
    checks.check(
        "idle window is idle (5 s, no input)",
        cpu < Duration::from_millis(20) && frames == 0,
        format!(
            "process CPU {:.2} ms (GetProcessTimes), {:.2} M cycles (QueryProcessCycleTime) over {:.1} s, {messages} message(s), {frames} frame(s)",
            cpu.as_secs_f64() * 1000.,
            cycles as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        ),
    );
}

/// The alternative the backend does not use: `SetDIBitsToDevice` of the whole frame to the window.
fn set_dibits_ms(hwnd: HWND, fb: &Framebuffer, n: usize) -> f64 {
    let info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: fb.width as i32,
            biHeight: -(fb.height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    // SAFETY: the window's own DC; the header describes `fb` exactly.
    let mut v: Vec<f64> = (0..n)
        .map(|_| unsafe {
            let dc = GetDC(Some(hwnd));
            let t = Instant::now();
            SetDIBitsToDevice(
                dc,
                0,
                0,
                fb.width,
                fb.height,
                0,
                0,
                0,
                fb.height,
                fb.pixels.as_ptr() as *const _,
                &info,
                DIB_RGB_COLORS,
            );
            let _ = GdiFlush();
            let ms = t.elapsed().as_secs_f64() * 1000.;
            ReleaseDC(Some(hwnd), dc);
            ms
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[n / 2]
}

pub fn timings(
    checks: &mut Checks,
    hwnd: HWND,
    window: &Rc<RefCell<WindowsWindow>>,
    frame: &Rc<RefCell<Frame>>,
    onscreen: bool,
) {
    let r0 = rect(hwnd);
    let (bw, bh) = {
        let (cw, ch) = client_size(hwnd);
        (r0.right - r0.left - cw, r0.bottom - r0.top - ch)
    };
    let mut lines = Vec::new();
    for (w, h) in [(1920, 1200), (3840, 2160)] {
        // SAFETY: resizing our own window without activating it.
        unsafe {
            SetWindowPos(hwnd, None, 0, 0, w + bw, h + bh, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE)
                .unwrap()
        };
        pump();
        let (cw, ch) = client_size(hwnd);
        let mut fb = Framebuffer::new(cw as u32, ch as u32);
        pattern(&mut fb, 7);
        let full =
            [Bounds::new(point(DevicePixels(0), DevicePixels(0)), size(DevicePixels(cw), DevicePixels(ch)))];
        let small = [Bounds::new(
            point(DevicePixels(200), DevicePixels(200)),
            size(DevicePixels(64), DevicePixels(64)),
        )];
        let time = |damage: &[Bounds<DevicePixels>], n: usize| {
            let mut v: Vec<f64> = (0..n)
                .map(|_| {
                    let t = Instant::now();
                    window.borrow_mut().present(&fb, damage);
                    // SAFETY: no arguments.
                    let _ = unsafe { GdiFlush() };
                    t.elapsed().as_secs_f64() * 1000.
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[n / 2]
        };
        time(&full, 3);
        // Interleaved rounds, so a busy machine slows both methods alike.
        let (mut full_ms, mut direct_ms) = (Vec::new(), Vec::new());
        for _ in 0..10 {
            full_ms.push(time(&full, 5));
            direct_ms.push(set_dibits_ms(hwnd, &fb, 5));
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let (full_ms, direct_ms) = (median(&mut full_ms), median(&mut direct_ms));
        let small_ms = time(&small, 200);
        // A capture makes the core present again: give it this frame.
        frame.borrow_mut().buffer = fb.clone();
        let mut shot = print_client(hwnd, cw, ch);
        let mut diff = mismatch(&shot, &fb);
        if !diff.is_empty() {
            let first = diff.clone();
            shot = print_client(hwnd, cw, ch);
            diff = format!(
                " (first capture{first}; second{})",
                if shot == fb.pixels { " equal".into() } else { mismatch(&shot, &fb) }
            );
        }
        // On screen, the part of a window bigger than the display is not drawn: only off screen,
        // where the capture's paint draws it all, must the whole window match.
        let whole = shot == fb.pixels || (onscreen && w > 1920);
        checks.check(
            &format!("present {cw}x{ch} matches PrintWindow"),
            whole && cw == w && ch == h,
            format!("client {cw}x{ch}{diff}"),
        );
        lines.push(format!(
            "present {cw}x{ch} full: {full_ms:.3} ms, 64x64: {small_ms:.4} ms (median); the same SetDIBitsToDevice from the example: {direct_ms:.3} ms"
        ));
    }
    for l in lines {
        println!("      {l}");
    }
    // SAFETY: back to the original size; the frame callback repaints.
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            r0.right - r0.left,
            r0.bottom - r0.top,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
        .unwrap()
    };
    frame.borrow_mut().buffer = Framebuffer::default();
    pump();
}
