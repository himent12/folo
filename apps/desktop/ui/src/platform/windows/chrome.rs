// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/events.rs, system_settings.rs
//! The custom title bar, as GPUI does it: `WM_NCCALCSIZE` makes the client area cover the window
//! (less the invisible resize border on the left, right and bottom, and the maximized insets), hit
//! testing asks the app's caption areas first, a caption press goes to the app before Windows, and
//! the caption buttons act on release. Also DPI and display changes.

use super::display::WindowsDisplay;
use super::events::screen_to_logical;
use super::util::*;
use super::window::{TaskbarEdge, WindowInner};
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::UI::HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi};
use ::windows::Win32::UI::Shell::{
    ABM_GETSTATE, ABM_GETTASKBARPOS, ABS_AUTOHIDE, APPBARDATA, SHAppBarMessage,
};
use ::windows::Win32::UI::WindowsAndMessaging::*;

/// How far a maximized window leaves an auto-hide taskbar (so it can still slide out).
const AUTO_HIDE_TASKBAR_THICKNESS_PX: i32 = 1;

impl WindowInner {
    pub(crate) fn handle_calc_client_size(
        &self,
        hwnd: HWND,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Option<isize> {
        if !self.hide_title_bar || self.state.borrow().is_fullscreen() || wparam.0 == 0 {
            return None;
        }
        let is_maximized = self.is_maximized();
        let insets = client_area_insets(hwnd, is_maximized, self.version);
        // SAFETY: with wparam TRUE, lparam points to an NCCALCSIZE_PARAMS.
        let rect = unsafe { &mut (*(lparam.0 as *mut NCCALCSIZE_PARAMS)).rgrc[0] };
        rect.left += insets.left;
        rect.top += insets.top;
        rect.right -= insets.right;
        rect.bottom -= insets.bottom;
        if is_maximized && let Some(edge) = self.state.borrow().auto_hide_taskbar {
            match edge {
                TaskbarEdge::Left => rect.left += AUTO_HIDE_TASKBAR_THICKNESS_PX,
                TaskbarEdge::Top => rect.top += AUTO_HIDE_TASKBAR_THICKNESS_PX,
                TaskbarEdge::Right => rect.right -= AUTO_HIDE_TASKBAR_THICKNESS_PX,
                TaskbarEdge::Bottom => rect.bottom -= AUTO_HIDE_TASKBAR_THICKNESS_PX,
            }
        }
        Some(0)
    }

    pub(crate) fn handle_dpi_changed(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let new_scale_factor = wparam.loword() as f32 / USER_DEFAULT_SCREEN_DPI as f32;
        let (was_maximized, size_events) = {
            let mut state = self.state.borrow_mut();
            state.scale_factor = new_scale_factor;
            state.border_offset.update(hwnd);
            state.full_present = true;
            (self.is_maximized(), state.size_events)
        };
        // SAFETY: for WM_DPICHANGED lparam points to the suggested window RECT.
        let rect = unsafe { *(lparam.0 as *const RECT) };
        // This sends WM_SIZE and WM_MOVE before it returns; WM_SIZE redraws at the new scale.
        // SAFETY: repositioning our own window.
        let _ = unsafe {
            SetWindowPos(
                hwnd,
                None,
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
                SWP_NOZORDER | SWP_NOACTIVATE,
            )
        };
        // Maximized (or the same device size), no WM_SIZE came: tell the core about the new scale anyway.
        if was_maximized || self.state.borrow().size_events == size_events {
            let mut client = RECT::default();
            // SAFETY: a valid out pointer.
            let _ = unsafe { GetClientRect(hwnd, &mut client) };
            self.handle_size_change(
                size(DevicePixels(client.right.max(1)), DevicePixels(client.bottom.max(1))),
                new_scale_factor,
            );
        }
        Some(0)
    }

    /// A monitor was connected, disconnected or changed. If ours is gone Windows moves the window
    /// and minimizes it: show it again and take the new monitor (GPUI's handling).
    pub(crate) fn handle_display_change(&self, hwnd: HWND) -> Option<isize> {
        let previous = self.state.borrow().display.handle;
        if WindowsDisplay::is_connected(previous) {
            return None;
        }
        // SAFETY: showing our own window and a plain query.
        let monitor = unsafe {
            let _ = ShowWindow(hwnd, if self.offscreen { SW_SHOWNOACTIVATE } else { SW_SHOWNORMAL });
            MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL)
        };
        if let Ok(display) = WindowsDisplay::with_handle(monitor) {
            self.state.borrow_mut().display = display;
            Some(0)
        } else {
            None
        }
    }

    pub(crate) fn handle_hit_test(
        &self,
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Option<isize> {
        if !self.is_movable || self.state.borrow().is_fullscreen() {
            return None;
        }
        if let Some(Some(area)) = self.with_callback(|c| &mut c.hit_test_window_control, |f| f()) {
            return Some(match area {
                WindowControlArea::Drag => HTCAPTION,
                WindowControlArea::Close => HTCLOSE,
                WindowControlArea::Max => HTMAXBUTTON,
                WindowControlArea::Min => HTMINBUTTON,
            } as isize);
        }
        if !self.hide_title_bar {
            return None;
        }
        // The resize border Windows computes (left, right, bottom, corners).
        // SAFETY: default hit testing of our window.
        let hit = unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
        if matches!(
            hit.0 as u32,
            HTNOWHERE
                | HTRIGHT
                | HTLEFT
                | HTTOPLEFT
                | HTTOP
                | HTTOPRIGHT
                | HTBOTTOMRIGHT
                | HTBOTTOM
                | HTBOTTOMLEFT
        ) {
            return Some(hit.0);
        }
        // The top edge is inside the client area: the frame's thickness of it resizes.
        // SAFETY: plain queries.
        let frame_y = unsafe { GetSystemMetricsForDpi(SM_CYFRAME, GetDpiForWindow(hwnd)) };
        let mut p = POINT { x: lparam.signed_loword() as i32, y: lparam.signed_hiword() as i32 };
        // SAFETY: a valid in/out pointer.
        let _ = unsafe { ScreenToClient(hwnd, &mut p) };
        if !self.is_maximized() && p.y >= 0 && p.y <= frame_y {
            return Some(HTTOP as isize);
        }
        Some(HTCLIENT as isize)
    }

    pub(crate) fn handle_nc_mouse_move(&self, hwnd: HWND, lparam: LPARAM) -> Option<isize> {
        self.start_tracking_mouse(
            hwnd,
            ::windows::Win32::UI::Input::KeyboardAndMouse::TME_LEAVE
                | ::windows::Win32::UI::Input::KeyboardAndMouse::TME_NONCLIENT,
        );
        let scale_factor = self.state.borrow().scale_factor;
        let input = PlatformInput::MouseMove(MouseMoveEvent {
            position: screen_to_logical(hwnd, lparam, scale_factor),
            pressed_button: None,
            modifiers: super::keyboard::current_modifiers(),
        });
        match self.dispatch_input(input)? {
            r if !r.propagate => Some(0),
            _ => None,
        }
    }

    /// A press on a caption area goes to the app first; only if nothing handled it (or prevented its
    /// default) does Windows get it. Caption buttons are remembered and act on release.
    pub(crate) fn handle_nc_mouse_down(
        &self,
        hwnd: HWND,
        button: MouseButton,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Option<isize> {
        let mut p = POINT { x: lparam.signed_loword() as i32, y: lparam.signed_hiword() as i32 };
        // SAFETY: a valid in/out pointer.
        let _ = unsafe { ScreenToClient(hwnd, &mut p) };
        if self.state.borrow().callbacks.input.is_some() {
            let (click_count, scale_factor) = {
                let mut state = self.state.borrow_mut();
                (
                    state.click_state.update(button, point(DevicePixels(p.x), DevicePixels(p.y))),
                    state.scale_factor,
                )
            };
            let input = PlatformInput::MouseDown(MouseDownEvent {
                button,
                position: logical_point(p.x as f32, p.y as f32, scale_factor),
                modifiers: super::keyboard::current_modifiers(),
                click_count,
                first_mouse: false,
            });
            if let Some(result) = self.dispatch_input(input)
                && (!result.propagate || result.default_prevented)
            {
                return Some(0);
            }
        }
        if button != MouseButton::Left {
            return None;
        }
        match wparam.0 as u32 {
            code @ (HTMINBUTTON | HTMAXBUTTON | HTCLOSE) => {
                self.state.borrow_mut().nc_button_pressed = Some(code);
                Some(0)
            }
            _ => None,
        }
    }

    pub(crate) fn handle_nc_mouse_up(
        &self,
        hwnd: HWND,
        button: MouseButton,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Option<isize> {
        if self.state.borrow().callbacks.input.is_some() {
            let scale_factor = self.state.borrow().scale_factor;
            let input = PlatformInput::MouseUp(MouseUpEvent {
                button,
                position: screen_to_logical(hwnd, lparam, scale_factor),
                modifiers: super::keyboard::current_modifiers(),
                click_count: 1,
            });
            if self.dispatch_input(input).is_some_and(|r| !r.propagate) {
                return Some(0);
            }
        }
        let pressed = self.state.borrow_mut().nc_button_pressed.take();
        if button != MouseButton::Left {
            return None;
        }
        let code = wparam.0 as u32;
        if pressed != Some(code) {
            return None;
        }
        if self.offscreen && code != HTCLOSE {
            if code == HTMINBUTTON {
                self.offscreen_minimize()
            } else {
                self.offscreen_toggle_zoom()
            }
            return Some(0);
        }
        // SAFETY: showing, maximizing or closing our own window.
        unsafe {
            match code {
                HTMINBUTTON => {
                    let _ = ShowWindowAsync(hwnd, SW_MINIMIZE);
                }
                HTMAXBUTTON => {
                    let _ = ShowWindowAsync(hwnd, if self.is_maximized() { SW_NORMAL } else { SW_MAXIMIZE });
                }
                HTCLOSE => {
                    let _ = PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
                }
                _ => return None,
            }
        }
        Some(0)
    }
}

/// GPUI's client-area insets: the frame thickness left, right and bottom (all four sides when
/// maximized, so the client area does not spill onto a neighbouring monitor); on top nothing on
/// Windows 10 and one device pixel per 96 DPI on Windows 11.
pub(crate) fn client_area_insets(hwnd: HWND, is_maximized: bool, version: WindowsVersion) -> RECT {
    // SAFETY: a plain query.
    let dpi = unsafe { GetDpiForWindow(hwnd) };
    let frame = frame_thickness(dpi);
    let top = if is_maximized {
        frame
    } else {
        match version {
            WindowsVersion::Win10 => 0,
            WindowsVersion::Win11 => (dpi as f32 / USER_DEFAULT_SCREEN_DPI as f32).round() as i32,
        }
    };
    RECT { left: frame, top, right: frame, bottom: frame }
}

/// The resize handle plus the padding outside it.
fn frame_thickness(dpi: u32) -> i32 {
    // SAFETY: plain queries.
    unsafe { GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi) }
}

/// Makes Windows send WM_NCCALCSIZE again.
pub(crate) fn notify_frame_changed(hwnd: HWND) {
    // SAFETY: a frame-changed notification for our window; nothing moves or activates.
    let _ = unsafe {
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED
                | SWP_NOACTIVATE
                | SWP_NOCOPYBITS
                | SWP_NOMOVE
                | SWP_NOOWNERZORDER
                | SWP_NOREPOSITION
                | SWP_NOSENDCHANGING
                | SWP_NOSIZE
                | SWP_NOZORDER,
        )
    };
}

/// The edge of an auto-hide taskbar on `display`, if there is one there (GPUI's rules).
pub(crate) fn auto_hide_taskbar_edge(display: &WindowsDisplay) -> Option<TaskbarEdge> {
    let mut info = APPBARDATA { cbSize: std::mem::size_of::<APPBARDATA>() as u32, ..Default::default() };
    // SAFETY: a sized APPBARDATA.
    if unsafe { SHAppBarMessage(ABM_GETSTATE, &mut info) } as u32 != ABS_AUTOHIDE {
        return None;
    }
    // SAFETY: as above.
    if unsafe { SHAppBarMessage(ABM_GETTASKBARPOS, &mut info) } == 0 {
        return None;
    }
    let bar = Bounds::new(
        point(DevicePixels(info.rc.left), DevicePixels(info.rc.top)),
        size(DevicePixels(info.rc.right - info.rc.left), DevicePixels(info.rc.bottom - info.rc.top)),
    );
    taskbar_edge(bar, display.physical_bounds())
}

fn taskbar_edge(bar: Bounds<DevicePixels>, screen: Bounds<DevicePixels>) -> Option<TaskbarEdge> {
    if screen.intersect(&bar) != bar {
        return None;
    }
    let full_width = bar.size.height < screen.size.height && bar.size.width == screen.size.width;
    let full_height = bar.size.width < screen.size.width && bar.size.height == screen.size.height;
    if bar.bottom() == screen.bottom() && bar.right() == screen.right() {
        if full_width {
            return Some(TaskbarEdge::Bottom);
        }
        if full_height {
            return Some(TaskbarEdge::Right);
        }
    } else if bar.top() == screen.top() && bar.left() == screen.left() {
        if full_width {
            return Some(TaskbarEdge::Top);
        }
        if full_height {
            return Some(TaskbarEdge::Left);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: i32, y: i32, w: i32, h: i32) -> Bounds<DevicePixels> {
        Bounds::new(point(DevicePixels(x), DevicePixels(y)), size(DevicePixels(w), DevicePixels(h)))
    }

    #[test]
    fn taskbar_edges() {
        let screen = r(0, 0, 3840, 2160);
        assert_eq!(taskbar_edge(r(0, 2088, 3840, 72), screen), Some(TaskbarEdge::Bottom));
        assert_eq!(taskbar_edge(r(0, 0, 3840, 72), screen), Some(TaskbarEdge::Top));
        assert_eq!(taskbar_edge(r(0, 0, 72, 2160), screen), Some(TaskbarEdge::Left));
        assert_eq!(taskbar_edge(r(3768, 0, 72, 2160), screen), Some(TaskbarEdge::Right));
        assert_eq!(taskbar_edge(r(3840, 0, 72, 2160), screen), None);
    }
}
