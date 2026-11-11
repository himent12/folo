//! Settings › Reduce motion, and the fade-in that honours it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_ui::{Animation, AnimationExt, IntoElement, SharedString, Styled};

static REDUCE_MOTION: AtomicBool = AtomicBool::new(false);

/// Settings › Reduce motion: stops pulses and shimmers and makes transitions instant.
/// Whether the OS asks for less motion (Windows: Settings › Accessibility › Animation effects off).
pub fn os_prefers_reduced_motion() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            SPI_GETCLIENTAREAANIMATION, SystemParametersInfoW,
        };
        let mut on: i32 = 1;
        // SAFETY: SPI_GETCLIENTAREAANIMATION writes one BOOL into the pointed-to value.
        let ok =
            unsafe { SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, (&mut on as *mut i32).cast(), 0) };
        ok != 0 && on == 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

pub fn set_reduce_motion(on: bool) {
    REDUCE_MOTION.store(on, Ordering::Relaxed);
}

pub fn reduce_motion() -> bool {
    REDUCE_MOTION.load(Ordering::Relaxed)
}

/// Fade an overlay in over 160 ms when it first appears (instant under Reduce motion).
pub fn appear<E: IntoElement + Styled + 'static>(el: E, id: &str) -> folo_ui::AnyElement {
    if reduce_motion() {
        return el.into_any_element();
    }
    el.with_opacity_animation(
        SharedString::from(id.to_string()),
        Animation::new(Duration::from_millis(160)).with_easing(folo_ui::ease_out_quint()),
        |p| p,
    )
    .into_any_element()
}
