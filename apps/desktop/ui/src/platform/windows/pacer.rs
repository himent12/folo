//! Frame pacing: one helper thread that, while any window asked for a frame, waits for the compositor
//! (`DwmFlush`, so it follows the display's real refresh rate) and posts each such window one
//! `WM_FOLO_FRAME`. With no request pending it blocks on a condition variable: an idle app does nothing.

use super::WM_FOLO_FRAME;
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::Graphics::Dwm::{DWM_TIMING_INFO, DwmFlush, DwmGetCompositionTimingInfo};
use ::windows::Win32::UI::WindowsAndMessaging::PostMessageW;

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
        Arc::new(Self { hwnd: hwnd.0 as isize, requested: AtomicBool::new(false), in_flight: AtomicBool::new(false) })
    }
}

#[derive(Default)]
struct State {
    waiting: Vec<Arc<FrameSlot>>,
    running: bool,
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
        state.waiting.push(slot.clone());
        if !state.running {
            state.running = true;
            let pacer = self.clone();
            let _ = std::thread::Builder::new().name("folo-ui-vsync".into()).spawn(move || pacer.run());
        }
        drop(state);
        self.wake.notify_one();
    }

    fn run(&self) {
        let period = refresh_period();
        loop {
            {
                let mut state = self.state.lock();
                while state.waiting.is_empty() {
                    self.wake.wait(&mut state);
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
                let posted = unsafe { PostMessageW(Some(HWND(slot.hwnd as _)), WM_FOLO_FRAME, WPARAM(0), LPARAM(0)) };
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
    let mut info = DWM_TIMING_INFO { cbSize: std::mem::size_of::<DWM_TIMING_INFO>() as u32, ..Default::default() };
    // SAFETY: a sized DWM_TIMING_INFO; a null window asks for the desktop's composition timing.
    if unsafe { DwmGetCompositionTimingInfo(HWND::default(), &mut info) }.is_err() {
        return DEFAULT_PERIOD;
    }
    let (num, den) = (info.rateRefresh.uiNumerator as u64, info.rateRefresh.uiDenominator as u64);
    let period = if num == 0 { DEFAULT_PERIOD } else { Duration::from_micros(den * 1_000_000 / num) };
    if period < EARLY_RETURN { DEFAULT_PERIOD } else { period }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refresh_period_is_sane() {
        let p = refresh_period();
        assert!(p >= Duration::from_micros(2_000) && p <= Duration::from_millis(50), "{p:?}");
    }
}
