//! `FOLO_UI_OFFSCREEN=1` window modes. Windows' own maximize moves a window onto its display and
//! activates it (and snaps any `WS_MAXIMIZE` window back onto its monitor), and its restore activates
//! it too, so off screen they are emulated: maximized is our own flag with the work area's size (plus
//! the frame Windows lets overhang) at the off-screen origin, restore puts the saved rectangle back,
//! and minimize / un-minimize use the
//! non-activating show commands. Caption commands (`WM_SYSCOMMAND`) are routed here, and a move or
//! size loop driven by the real mouse never starts.

use super::window::{OFFSCREEN_ORIGIN, WindowInner};
use ::windows::Win32::Foundation::{HWND, RECT};
use ::windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTONULL, MONITORINFO, MonitorFromRect};
use ::windows::Win32::UI::HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi};
use ::windows::Win32::UI::WindowsAndMessaging::*;

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
