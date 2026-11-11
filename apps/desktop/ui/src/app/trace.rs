//! Startup trace: with `FOLO_UI_TRACE` set (any value but `0`), the app prints `first-render <ms>` (the root
//! view's first `render`) and `first-frame <ms>` (the first frame presented) to stderr, in milliseconds since
//! the process was created. Owner tag: `"runtime"`.

use std::{
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// Whether the startup lines are wanted. Read once.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FOLO_UI_TRACE").is_ok_and(|value| !value.is_empty() && value != "0"))
}

/// Prints `first-render` the first time it is called.
pub(crate) fn first_render() {
    static DONE: AtomicBool = AtomicBool::new(false);
    once(&DONE, "first-render");
}

/// Prints `first-frame` the first time it is called (the window calls it after presenting).
pub fn first_frame() {
    static DONE: AtomicBool = AtomicBool::new(false);
    once(&DONE, "first-frame");
}

fn once(done: &AtomicBool, label: &str) {
    if enabled() && !done.swap(true, Ordering::Relaxed) {
        eprintln!("{label} {:.1}", since_process_start_ms());
    }
}

/// When the runtime first looked at the clock: the fallback origin where the OS does not say when the
/// process started. `Application::new` touches it.
pub(crate) fn mark_start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Milliseconds since the process was created (Windows: `GetProcessTimes`; elsewhere: since
/// `Application::new`, which runs within a millisecond or two of `main`).
fn since_process_start_ms() -> f64 {
    #[cfg(windows)]
    if let Some(ms) = windows_process_age_ms() {
        return ms;
    }
    mark_start().elapsed().as_secs_f64() * 1000.0
}

#[cfg(windows)]
fn windows_process_age_ms() -> Option<f64> {
    use ::windows::Win32::{
        Foundation::FILETIME,
        System::{
            SystemInformation::GetSystemTimeAsFileTime,
            Threading::{GetCurrentProcess, GetProcessTimes},
        },
    };
    let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    let (mut created, mut exited, mut kernel, mut user) = Default::default();
    // SAFETY: the pseudo-handle of the current process is always valid; the out-pointers are live locals.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) }
        .ok()?;
    // SAFETY: no preconditions.
    let now = unsafe { GetSystemTimeAsFileTime() };
    // FILETIME counts 100 ns intervals.
    Some(ticks(now).saturating_sub(ticks(created)) as f64 / 10_000.0)
}
