//! The Win32 platform backend (`ui/DESIGN.md`, Platform contract): our own window class and message
//! loop, GPUI's custom title bar, input and IME, OLE file drops, clipboard, file dialogs, per-monitor
//! DPI, frames paced by the compositor and presented with GDI. No GPU anywhere.
//!
//! Developer switches: `FOLO_UI_OFFSCREEN=1` opens windows outside every display, never activated,
//! with no taskbar button; `FOLO_UI_TRACE=1` prints `first-present <ms>` at the first present.

mod chrome;
mod clipboard;
mod dispatcher;
mod display;
mod drop_target;
mod events;
mod keyboard;
mod offscreen;
mod pacer;
mod platform;
mod platform_window;
mod present;
mod shell;
#[cfg(test)]
mod tests;
mod util;
mod window;

pub use dispatcher::{IDLE_EXIT, MAX_POOL_THREADS, WindowsDispatcher};
pub use display::WindowsDisplay;
pub use platform::WindowsPlatform;
pub use util::{ms_since_process_start, offscreen, process_cpu_time};
pub use window::WindowsWindow;

pub(crate) use platform::Shared;

use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_USER};

/// Main-thread tasks are waiting (to the message window; posted once until handled).
pub(crate) const WM_FOLO_MAIN_TASKS: u32 = WM_USER + 1;
/// Run the boxed closure in lparam (to the message window; see `post_call`).
pub(crate) const WM_FOLO_CALL: u32 = WM_USER + 2;
/// End the message loop (to the message window).
pub(crate) const WM_FOLO_QUIT: u32 = WM_USER + 3;
/// The pacer's per-refresh tick (to a window that asked for a frame).
pub(crate) const WM_FOLO_FRAME: u32 = WM_USER + 4;
/// Draw everything now (posted to a window when it is shown: an off-screen window gets no WM_PAINT,
/// and this comes before the low-priority WM_PAINT of one on screen).
pub(crate) const WM_FOLO_EXPOSE: u32 = WM_USER + 5;

/// Runs `f` on the main thread from the message loop, after the current message: for work that must
/// not run inside a callback (destroying a window, a modal dialog, a style swap). Main thread only.
pub(crate) fn post_call(message_window: HWND, f: Box<dyn FnOnce()>) {
    let call = Box::into_raw(Box::new(f));
    // SAFETY: posting to our message window, which frees the box when it handles the message.
    let posted =
        unsafe { PostMessageW(Some(message_window), WM_FOLO_CALL, WPARAM(0), LPARAM(call as isize)) };
    if posted.is_err() {
        // SAFETY: the message was not queued, so the box is still ours.
        drop(unsafe { Box::from_raw(call) });
    }
}
