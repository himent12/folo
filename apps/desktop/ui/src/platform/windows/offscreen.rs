//! `FOLO_UI_OFFSCREEN=1` window modes. Windows' own maximize moves a window onto its display and
//! activates it (and snaps any `WS_MAXIMIZE` window back onto its monitor), and its restore activates
//! it too, so off screen they are emulated: maximized is our own flag with the work area's size (plus
//! the frame Windows lets overhang) at the off-screen origin, restore puts the saved rectangle back,
//! and minimize / un-minimize use the
//! non-activating show commands. Caption commands (`WM_SYSCOMMAND`) are routed here, and a move or
//! size loop driven by the real mouse never starts.
//!
//! No other window of the process may reach a display either. The file dialogs and the shell's
//! launches are refused (`REFUSED`): a dialog's owner is the active window, and an off-screen window is
//! never active, so the dialog had none and Windows centred it on the primary display at its
//! remembered size (1200 x 720), on the screen of whoever uses the PC; Explorer and the browser open
//! windows of other processes there. Anything else that shows a window (a library's dialog, another
//! thread's) is moved off the displays as soon as it shows or moves (`Parker`), and named on stderr.

use super::window::{OFFSCREEN_ORIGIN, WindowInner};
use ::windows::Win32::Foundation::{HWND, RECT};
use ::windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONULL, MONITORINFO, MonitorFromRect};
use ::windows::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId};
use ::windows::Win32::UI::Accessibility::{HWINEVENTHOOK, SetWinEventHook, UnhookWinEvent};
use ::windows::Win32::UI::HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi};
use ::windows::Win32::UI::WindowsAndMessaging::*;

/// Why a file dialog, `open_url` or `reveal_path` did nothing off screen.
pub(crate) const REFUSED: &str =
    "FOLO_UI_OFFSCREEN=1: file dialogs, Explorer and the browser stay closed (they would open on a display)";

/// Moves `hwnd` to the off-screen origin, keeping its size, when it is a top-level window that any
/// display shows a part of; true if it moved it. Without activating it or changing the z-order, and
/// without waiting on another thread's window.
pub(crate) fn park(hwnd: HWND) -> bool {
    // SAFETY: plain queries.
    let top = unsafe { GetAncestor(hwnd, GA_ROOT) } == hwnd;
    let r = window_rect(hwnd);
    if !top || r.right <= r.left || r.bottom <= r.top || !on_a_display(&r) {
        return false;
    }
    // SAFETY: plain queries.
    let other_thread = unsafe { GetWindowThreadProcessId(hwnd, None) != GetCurrentThreadId() };
    let mut flags = SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE;
    if other_thread {
        flags |= SWP_ASYNCWINDOWPOS;
    }
    // SAFETY: repositioning a window of this process.
    unsafe { SetWindowPos(hwnd, None, OFFSCREEN_ORIGIN, OFFSCREEN_ORIGIN, 0, 0, flags) }.is_ok()
}

/// While alive, every window of this process that shows or moves onto a display is parked off
/// screen (`park`). Events come through the message loop of the thread that installed it.
pub(crate) struct Parker([HWINEVENTHOOK; 2]);

impl Parker {
    /// Hooks this process's windows showing and moving; `None` if Windows refused.
    pub(crate) fn install() -> Option<Self> {
        // SAFETY: out-of-context hooks for our own process; the callback is a plain function.
        let hook = |event| unsafe {
            SetWinEventHook(
                event,
                event,
                None,
                Some(on_event),
                GetCurrentProcessId(),
                0,
                WINEVENT_OUTOFCONTEXT,
            )
        };
        let hooks = [hook(EVENT_OBJECT_SHOW), hook(EVENT_OBJECT_LOCATIONCHANGE)];
        if hooks.iter().any(|h| h.is_invalid()) {
            drop(Parker(hooks));
            return None;
        }
        Some(Parker(hooks))
    }
}

impl Drop for Parker {
    fn drop(&mut self) {
        for hook in self.0 {
            if !hook.is_invalid() {
                // SAFETY: a hook this thread installed.
                let _ = unsafe { UnhookWinEvent(hook) };
            }
        }
    }
}

unsafe extern "system" fn on_event(
    _: HWINEVENTHOOK,
    _: u32,
    hwnd: HWND,
    object: i32,
    child: i32,
    _: u32,
    _: u32,
) {
    // The window itself (not its caret, cursor or a child object), visible.
    // SAFETY: a plain query.
    if object != OBJID_WINDOW.0 || child != CHILDID_SELF as i32 || !unsafe { IsWindowVisible(hwnd) }.as_bool()
    {
        return;
    }
    let before = window_rect(hwnd);
    if park(hwnd) {
        let mut class = [0u16; 128];
        let mut title = [0u16; 128];
        // SAFETY: valid buffers.
        let (c, t) = unsafe { (GetClassNameW(hwnd, &mut class), GetWindowTextW(hwnd, &mut title)) };
        eprintln!(
            "folo-ui: off screen: moved a {} window '{}' at ({}, {})-({}, {}) off the displays",
            String::from_utf16_lossy(&class[..c.max(0) as usize]),
            String::from_utf16_lossy(&title[..t.max(0) as usize]),
            before.left,
            before.top,
            before.right,
            before.bottom,
        );
    }
}

/// The window's rectangle in screen pixels.
pub(crate) fn window_rect(hwnd: HWND) -> RECT {
    let mut r = RECT::default();
    // SAFETY: a valid out pointer.
    let _ = unsafe { GetWindowRect(hwnd, &mut r) };
    r
}

/// Captures the window once into a 1x1 bitmap. Off screen the window DC shows nothing, and the
/// compositor builds the window's picture only when it is captured (`PrintWindow` with
/// `PW_RENDERFULLCONTENT`, which repaints it); the first capture after the window appears or changes
/// size sometimes comes out torn, later ones do not. Taking that one here keeps the tools' whole.
pub(crate) fn warm_capture(hwnd: HWND) {
    use ::windows::Win32::Graphics::Gdi::{
        CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
    };
    use ::windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
    // SAFETY: a 1x1 memory bitmap made, used and freed here; PrintWindow repaints our own window.
    unsafe {
        let screen = GetDC(None);
        let dc = CreateCompatibleDC(Some(screen));
        let bitmap = CreateCompatibleBitmap(screen, 1, 1);
        let old = SelectObject(dc, bitmap.into());
        // PW_CLIENTONLY | PW_RENDERFULLCONTENT, as the capture tools use.
        let _ = PrintWindow(hwnd, dc, PRINT_WINDOW_FLAGS(3));
        SelectObject(dc, old);
        let _ = DeleteObject(bitmap.into());
        let _ = DeleteDC(dc);
        ReleaseDC(None, screen);
    }
}

/// Whether any part of `rect` is on a display.
pub(crate) fn on_a_display(rect: &RECT) -> bool {
    // SAFETY: a plain query.
    !unsafe { MonitorFromRect(rect, MONITOR_DEFAULTTONULL) }.is_invalid()
}

impl WindowInner {
    /// Maximizes off screen (see the module docs).
    pub(crate) fn offscreen_maximize(&self) {
        if self.is_maximized() {
            return;
        }
        let normal = window_rect(self.hwnd);
        let monitor = self.state.borrow().display.handle;
        let mut info =
            MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..Default::default() };
        // SAFETY: a sized MONITORINFO.
        let _ = unsafe { GetMonitorInfoW(monitor, &mut info) };
        let work = info.rcWork;
        // SAFETY: plain queries.
        let frame = unsafe {
            let dpi = GetDpiForWindow(self.hwnd);
            GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)
        };
        {
            let mut state = self.state.borrow_mut();
            state.offscreen_restore = Some(normal);
            state.offscreen_maximized = true;
        }
        let (cx, cy) = (work.right - work.left + 2 * frame, work.bottom - work.top + 2 * frame);
        // SAFETY: repositioning our own window, without activating it.
        let _ = unsafe {
            SetWindowPos(
                self.hwnd,
                None,
                OFFSCREEN_ORIGIN,
                OFFSCREEN_ORIGIN,
                cx,
                cy,
                SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER,
            )
        };
    }

    /// Restores an off-screen maximized window to its saved rectangle; un-minimizes a minimized one.
    pub(crate) fn offscreen_restore(&self) {
        // SAFETY: a plain query and a non-activating show.
        if unsafe { IsIconic(self.hwnd) }.as_bool() {
            let _ = unsafe { ShowWindow(self.hwnd, SW_SHOWNOACTIVATE) };
            return;
        }
        if !self.is_maximized() {
            return;
        }
        let saved = {
            let mut state = self.state.borrow_mut();
            state.offscreen_maximized = false;
            state.offscreen_restore.take()
        };
        let r = saved.unwrap_or(RECT {
            left: OFFSCREEN_ORIGIN,
            top: OFFSCREEN_ORIGIN,
            right: OFFSCREEN_ORIGIN + 960,
            bottom: OFFSCREEN_ORIGIN + 620,
        });
        // SAFETY: repositioning our own window, without activating it.
        let _ = unsafe {
            SetWindowPos(
                self.hwnd,
                None,
                r.left,
                r.top,
                r.right - r.left,
                r.bottom - r.top,
                SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER,
            )
        };
    }

    /// Maximizes, or restores a maximized window.
    pub(crate) fn offscreen_toggle_zoom(&self) {
        if self.is_maximized() { self.offscreen_restore() } else { self.offscreen_maximize() }
    }

    /// Minimizes without activating anything.
    pub(crate) fn offscreen_minimize(&self) {
        // SAFETY: a non-activating show of our own window.
        let _ = unsafe { ShowWindow(self.hwnd, SW_SHOWMINNOACTIVE) };
    }

    /// Off screen, the caption's system commands: `Some` when handled here.
    pub(crate) fn offscreen_system_command(&self, command: u32) -> Option<isize> {
        if !self.offscreen {
            return None;
        }
        match command & 0xFFF0 {
            SC_MAXIMIZE => self.offscreen_maximize(),
            SC_RESTORE => self.offscreen_restore(),
            SC_MINIMIZE => self.offscreen_minimize(),
            // A move or size loop follows the real mouse: never off screen.
            SC_MOVE | SC_SIZE => {}
            _ => return None,
        }
        Some(0)
    }
}
