// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/vsync.rs
//! Frame pacing: one helper thread that, while any window asked for a frame, waits for the compositor
//! (`DwmFlush`, so it follows the display's real refresh rate) and posts each such window one
//! `WM_FOLO_FRAME`. With no request pending it blocks on a condition variable: an idle app does nothing.
//! `shut_down` (when the platform's loop ends) stops the thread and joins it.

use super::WM_FOLO_FRAME;
use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmFlush, DwmGetCompositionTimingInfo};
use ::windows::Win32::UI::WindowsAndMessaging::PostMessageW;
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A wait shorter than this means `DwmFlush` returned without a composition (display asleep, nothing
/// to compose): sleep one refresh period instead, as GPUI's vsync provider does.
const EARLY_RETURN: Duration = Duration::from_millis(1);
/// The period assumed when DWM cannot say (60 Hz).
const DEFAULT_PERIOD: Duration = Duration::from_micros(16_666);

/// One window's frame-request state, shared with the pacer thread.
pub(crate) struct FrameSlot {
    pub(crate) hwnd: isize,
    /// The core asked for a frame that has not been drawn yet.
    pub(crate) requested: AtomicBool,
    /// A `WM_FOLO_FRAME` is queued and not handled yet (so a slow main thread never gets two).
    pub(crate) in_flight: AtomicBool,
}

impl FrameSlot {
    pub(crate) fn new(hwnd: HWND) -> Arc<Self> {
        Arc::new(Self {
            hwnd: hwnd.0 as isize,
            requested: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
        })
    }
}

#[derive(Default)]
struct State {
    waiting: Vec<Arc<FrameSlot>>,
    /// The thread, once started (taken by `shut_down` to join it).
    thread: Option<JoinHandle<()>>,
    /// Set by `shut_down`: the thread exits and no other starts.
    stopped: bool,
}

/// The platform's frame pacer.
#[derive(Default)]
pub(crate) struct Pacer {
    state: Mutex<State>,
    wake: Condvar,
}

impl Pacer {
    /// Asks for one frame for `slot` at the next refresh. Coalesces: a pending request is kept.
    pub(crate) fn request(self: &Arc<Self>, slot: &Arc<FrameSlot>) {
        if slot.requested.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut state = self.state.lock();
        if state.stopped {
            return;
        }
        state.waiting.push(slot.clone());
        if state.thread.is_none() {
            let pacer = self.clone();
            state.thread =
                std::thread::Builder::new().name("folo-ui-vsync".into()).spawn(move || pacer.run()).ok();
        }
        drop(state);
        self.wake.notify_one();
    }

    /// Stops the thread and waits for it (at most one refresh). Later requests are ignored.
    pub(crate) fn shut_down(&self) {
        let thread = {
            let mut state = self.state.lock();
            state.stopped = true;
            state.waiting.clear();
            state.thread.take()
        };
        self.wake.notify_all();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }

    fn run(&self) {
        let period = refresh_period();
        loop {
            {
                let mut state = self.state.lock();
                while state.waiting.is_empty() && !state.stopped {
                    self.wake.wait(&mut state);
                }
                if state.stopped {
                    return;
                }
            }
            let start = Instant::now();
            // SAFETY: no arguments; blocks until the next composition pass.
            let ok = unsafe { DwmFlush() }.is_ok();
            if !ok || start.elapsed() < EARLY_RETURN {
                std::thread::sleep(period.saturating_sub(start.elapsed()));
            }
            let mut state = self.state.lock();
            state.waiting.retain(|slot| {
                if slot.in_flight.swap(true, Ordering::AcqRel) {
                    return true; // the last frame message is still queued: try again next refresh
                }
                // SAFETY: posting to a window handle; a destroyed window only makes the call fail.
                let posted =
                    unsafe { PostMessageW(Some(HWND(slot.hwnd as _)), WM_FOLO_FRAME, WPARAM(0), LPARAM(0)) };
                if posted.is_err() {
                    slot.in_flight.store(false, Ordering::Release);
                    slot.requested.store(false, Ordering::Release);
                }
                false
            });
        }
    }
}

/// The compositor's refresh period (from DWM's refresh rate), used only when `DwmFlush` returns early.
pub(crate) fn refresh_period() -> Duration {
    let mut info =
        DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
    // SAFETY: a sized DWM_TIMING_INFO; a null window asks for the desktop's composition timing.
    if unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut info) }.is_err() {
        return DEFAULT_PERIOD;
    }
    let (num, den) = (info.rateRefresh.uiNumerator as u64, info.rateRefresh.uiDenominator as u64);
    let period = (den * 1_000_000).checked_div(num).map_or(DEFAULT_PERIOD, Duration::from_micros);
    if period < EARLY_RETURN { DEFAULT_PERIOD } else { period }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shut_down_stops_and_joins_the_thread() {
        let pacer = Arc::new(Pacer::default());
        // A null window: the frame message goes to the pacer thread's own queue, which nobody reads.
        let slot = FrameSlot::new(HWND::default());
        pacer.request(&slot);
        assert!(pacer.state.lock().thread.is_some(), "the thread started");
        pacer.shut_down();
        assert!(pacer.state.lock().thread.is_none());
        assert_eq!(Arc::strong_count(&pacer), 1, "the thread has ended and dropped its pacer");
        // Requests after the shutdown start nothing.
        let slot = FrameSlot::new(HWND::default());
        pacer.request(&slot);
        {
            let state = pacer.state.lock();
            assert!(state.thread.is_none() && state.waiting.is_empty());
        }
        assert_eq!(Arc::strong_count(&pacer), 1);
        pacer.shut_down();
    }

    #[test]
    fn the_refresh_period_is_sane() {
        let p = refresh_period();
        assert!(p >= Duration::from_micros(2_000) && p <= Duration::from_millis(50), "{p:?}");
    }
}
