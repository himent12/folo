// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/events.rs
//! The window procedure's dispatch and the window, paint, frame and mouse messages, with GPUI's
//! semantics (GPUI 0.2.2's `src/platform/windows/events.rs`). Keys and IME are in `keyboard.rs`, the
//! non-client area (title bar, hit testing, resize frame) in `chrome.rs`.

use super::WM_FOLO_FRAME;
use super::display::WindowsDisplay;
use super::util::*;
use super::window::{WheelSettings, WindowInner};
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::System::SystemServices::{
    MK_LBUTTON, MK_MBUTTON, MK_RBUTTON, MK_XBUTTON1, MK_XBUTTON2, MODIFIERKEYS_FLAGS,
};
use ::windows::Win32::UI::Controls::{HOVER_DEFAULT, WM_MOUSELEAVE};
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::PCWSTR;
use std::rc::Rc;
use std::sync::atomic::Ordering;

impl WindowInner {
    pub(crate) fn handle_msg(
        self: &Rc<Self>,
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        let handled = match msg {
            WM_ACTIVATE => self.handle_activate(hwnd, wparam),
            WM_CREATE => self.handle_create(hwnd),
            WM_MOVE => self.handle_move(hwnd, lparam),
            WM_SIZE => self.handle_size(wparam, lparam),
            WM_GETMINMAXINFO => self.handle_get_min_max_info(lparam),
            WM_WINDOWPOSCHANGING => self.handle_window_pos_changing(hwnd, msg, wparam, lparam),
            WM_NCCALCSIZE => self.handle_calc_client_size(hwnd, wparam, lparam),
            WM_DPICHANGED => self.handle_dpi_changed(hwnd, wparam, lparam),
            WM_DISPLAYCHANGE => self.handle_display_change(hwnd),
            WM_NCHITTEST => self.handle_hit_test(hwnd, msg, wparam, lparam),
            WM_ERASEBKGND => Some(1),
            WM_PAINT => self.handle_paint(hwnd),
            WM_PRINTCLIENT => self.handle_print_client(wparam),
            WM_FOLO_FRAME => self.handle_frame(),
            super::WM_FOLO_EXPOSE => self.handle_expose(),
            WM_CLOSE => self.handle_close(),
            WM_DESTROY => self.handle_destroy(hwnd),
            WM_MOUSEMOVE => self.handle_mouse_move(hwnd, lparam, wparam),
            WM_MOUSELEAVE | WM_NCMOUSELEAVE => self.handle_mouse_leave(),
            WM_NCMOUSEMOVE => self.handle_nc_mouse_move(hwnd, lparam),
            // A double click is a second single click: we count clicks ourselves. Unhandled, it falls
            // through to Windows, which maximizes or restores on a caption double click.
            WM_NCLBUTTONDBLCLK | WM_NCLBUTTONDOWN => {
                self.handle_nc_mouse_down(hwnd, MouseButton::Left, wparam, lparam)
            }
            WM_NCRBUTTONDOWN => self.handle_nc_mouse_down(hwnd, MouseButton::Right, wparam, lparam),
            WM_NCMBUTTONDOWN => self.handle_nc_mouse_down(hwnd, MouseButton::Middle, wparam, lparam),
            WM_NCLBUTTONUP => self.handle_nc_mouse_up(hwnd, MouseButton::Left, wparam, lparam),
            WM_NCRBUTTONUP => self.handle_nc_mouse_up(hwnd, MouseButton::Right, wparam, lparam),
            WM_NCMBUTTONUP => self.handle_nc_mouse_up(hwnd, MouseButton::Middle, wparam, lparam),
            WM_LBUTTONDOWN => self.handle_mouse_down(hwnd, MouseButton::Left, lparam),
            WM_RBUTTONDOWN => self.handle_mouse_down(hwnd, MouseButton::Right, lparam),
            WM_MBUTTONDOWN => self.handle_mouse_down(hwnd, MouseButton::Middle, lparam),
            WM_XBUTTONDOWN => {
                xbutton(wparam).and_then(|b| self.handle_mouse_down(hwnd, b, lparam)).or(Some(1))
            }
            WM_LBUTTONUP => self.handle_mouse_up(MouseButton::Left, lparam),
            WM_RBUTTONUP => self.handle_mouse_up(MouseButton::Right, lparam),
            WM_MBUTTONUP => self.handle_mouse_up(MouseButton::Middle, lparam),
            WM_XBUTTONUP => xbutton(wparam).and_then(|b| self.handle_mouse_up(b, lparam)).or(Some(1)),
            WM_MOUSEWHEEL => self.handle_mouse_wheel(hwnd, wparam, lparam),
            WM_MOUSEHWHEEL => self.handle_mouse_horizontal_wheel(hwnd, wparam, lparam),
            WM_CAPTURECHANGED => self.handle_capture_changed(hwnd, lparam),
            WM_CANCELMODE => {
                self.end_presses(hwnd);
                None
            }
            WM_SYSKEYDOWN => self.handle_syskeydown(hwnd, wparam, lparam),
            WM_SYSKEYUP => self.handle_syskeyup(hwnd, wparam, lparam),
            WM_SYSCOMMAND => self.handle_system_command(wparam),
            WM_KEYDOWN => self.handle_keydown(hwnd, wparam, lparam),
            WM_KEYUP => self.handle_keyup(hwnd, wparam, lparam),
            WM_CHAR => self.handle_char(wparam),
            WM_DEADCHAR => self.handle_dead_char(wparam),
            WM_IME_STARTCOMPOSITION => self.handle_ime_position(hwnd),
            WM_IME_COMPOSITION => self.handle_ime_composition(hwnd, lparam),
            WM_SETCURSOR => self.handle_set_cursor(hwnd, lparam),
            WM_SETTINGCHANGE => self.handle_settings_changed(hwnd, wparam, lparam),
            _ => None,
        };
        match handled {
            Some(n) => LRESULT(n),
            // SAFETY: default processing of a message we did not handle.
            None => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
        }
    }

    fn handle_create(&self, hwnd: HWND) -> Option<isize> {
        if self.hide_title_bar {
            super::chrome::notify_frame_changed(hwnd);
            Some(0)
        } else {
            None
        }
    }

    fn handle_activate(self: &Rc<Self>, hwnd: HWND, wparam: WPARAM) -> Option<isize> {
        let activated = wparam.loword() > 0;
        if !activated {
            // A button held while the user switches away is released somewhere else.
            self.end_presses(hwnd);
        }
        // GPUI reports activation from the executor, after the message: do the same through the queue.
        let this = self.clone();
        super::post_call(
            self.shared.message_window(),
            Box::new(move || {
                this.with_callback(|c| &mut c.active_status_change, |f| f(activated));
            }),
        );
        None
    }

    fn handle_move(&self, hwnd: HWND, lparam: LPARAM) -> Option<isize> {
        {
            let mut state = self.state.borrow_mut();
            let origin = logical_point(
                lparam.signed_loword() as f32,
                lparam.signed_hiword() as f32,
                state.scale_factor,
            );
            state.origin = origin;
            let center_x = origin.x.0 + state.logical_size.width.0 / 2.;
            let center_y = origin.y.0 + state.logical_size.height.0 / 2.;
            let b = state.display.bounds();
            if center_x < b.left().0
                || center_x > b.right().0
                || center_y < b.top().0
                || center_y > b.bottom().0
            {
                // The centre may be on another monitor now (a minimized window gives none).
                // SAFETY: a plain query.
                let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL) };
                if !monitor.is_invalid()
                    && state.display.handle != monitor
                    && let Ok(display) = WindowsDisplay::with_handle(monitor)
                {
                    state.display = display;
                }
            }
        }
        self.with_callback(|c| &mut c.moved, |f| f());
        Some(0)
    }

    fn handle_get_min_max_info(&self, lparam: LPARAM) -> Option<isize> {
        let state = self.state.borrow();
        // SAFETY: for WM_GETMINMAXINFO lparam points to the MINMAXINFO to fill in.
        let info = unsafe { &mut *(lparam.0 as *mut MINMAXINFO) };
        if self.offscreen {
            // An off-screen window may be bigger than the display (the present timings use 4K).
            info.ptMaxTrackSize = POINT { x: 8192, y: 8192 };
        }
        let min = state.min_size?;
        info.ptMinTrackSize.x =
            min.width.scale(state.scale_factor).0 as i32 + state.border_offset.width_offset;
        info.ptMinTrackSize.y =
            min.height.scale(state.scale_factor).0 as i32 + state.border_offset.height_offset;
        Some(0)
    }

    fn handle_window_pos_changing(
        &self,
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Option<isize> {
        if !self.offscreen {
            return None;
        }
        // Off screen, whatever moves the window, it never lands on a display and is never raised.
        // Windows' own handling (which snaps a maximized window to its monitor) runs first.
        // SAFETY: default processing, then lparam points to the WINDOWPOS being applied.
        let pos = unsafe {
            DefWindowProcW(hwnd, msg, wparam, lparam);
            &mut *(lparam.0 as *mut WINDOWPOS)
        };
        pos.flags |= SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER;
        if !pos.flags.contains(SWP_NOMOVE) || !pos.flags.contains(SWP_NOSIZE) {
            let current = super::offscreen::window_rect(hwnd);
            let (x, y) =
                if pos.flags.contains(SWP_NOMOVE) { (current.left, current.top) } else { (pos.x, pos.y) };
            let (cx, cy) = if pos.flags.contains(SWP_NOSIZE) {
                (current.right - current.left, current.bottom - current.top)
            } else {
                (pos.cx, pos.cy)
            };
            if super::offscreen::on_a_display(&RECT { left: x, top: y, right: x + cx, bottom: y + cy }) {
                pos.x = super::window::OFFSCREEN_ORIGIN;
                pos.y = super::window::OFFSCREEN_ORIGIN;
                pos.flags &= !SWP_NOMOVE;
            }
        }
        Some(0)
    }

    fn handle_size(&self, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        if wparam.0 == SIZE_MINIMIZED as usize {
            self.state.borrow_mut().minimized = true;
            return Some(0);
        }
        let width = lparam.loword().max(1) as i32;
        let height = lparam.hiword().max(1) as i32;
        let scale_factor = {
            let mut state = self.state.borrow_mut();
            state.minimized = false;
            state.size_events += 1;
            state.scale_factor
        };
        self.handle_size_change(size(DevicePixels(width), DevicePixels(height)), scale_factor);
        Some(0)
    }

    /// The client area changed size: tell the core, then draw now (inside a live resize too, so the
    /// new edges are never black and never lag).
    pub(crate) fn handle_size_change(&self, device_size: Size<DevicePixels>, scale_factor: f32) {
        let logical = device_size.to_pixels(scale_factor);
        self.state.borrow_mut().logical_size = logical;
        self.with_callback(|c| &mut c.resize, |f| f(logical, scale_factor));
        if self.draw(true, false) {
            // SAFETY: validating our own window: what WM_SIZE drew covers the whole client area.
            let _ = unsafe { ValidateRect(Some(self.hwnd), None) };
            if self.offscreen {
                super::offscreen::warm_capture(self.hwnd);
            }
        }
    }

    /// The core presents again, inside BeginPaint / EndPaint: for a window wholly off screen the
    /// window DC shows nothing, and this paint (which `PrintWindow` sends) is the only way its pixels
    /// are ever seen.
    fn handle_paint(&self, hwnd: HWND) -> Option<isize> {
        let mut ps = PAINTSTRUCT::default();
        // SAFETY: the BeginPaint / EndPaint pair of our window.
        let dc = unsafe { BeginPaint(hwnd, &mut ps) };
        self.paint_dc.set(Some(dc));
        self.draw(true, false);
        self.paint_dc.set(None);
        // SAFETY: as above.
        let _ = unsafe { EndPaint(hwnd, &ps) };
        if self.offscreen {
            // Off screen this paint is a capture's: let the compositor take the whole of it first.
            // SAFETY: no arguments.
            let _ = unsafe { ::windows::Win32::Graphics::Dwm::DwmFlush() };
        }
        Some(0)
    }

    /// The core presents into the DC in wparam (`WM_PRINTCLIENT`).
    fn handle_print_client(&self, wparam: WPARAM) -> Option<isize> {
        self.paint_dc.set(Some(HDC(wparam.0 as _)));
        self.draw(true, false);
        self.paint_dc.set(None);
        // That present went to the print DC: the window's next one copies everything again.
        self.state.borrow_mut().full_present = true;
        Some(0)
    }

    /// The window was just shown: draw it whole now, before any WM_PAINT.
    fn handle_expose(&self) -> Option<isize> {
        if self.draw(true, false) {
            // The whole client area is drawn: the WM_PAINT that would follow has nothing to do.
            // SAFETY: validating our own window.
            let _ = unsafe { ValidateRect(Some(self.hwnd), None) };
            if self.offscreen {
                super::offscreen::warm_capture(self.hwnd);
            }
        }
        Some(0)
    }

    /// The pacer's tick: at most one frame per refresh while the core asks for frames.
    fn handle_frame(&self) -> Option<isize> {
        self.frame.in_flight.store(false, Ordering::Release);
        self.frame.requested.store(false, Ordering::Release);
        if self.state.borrow().minimized {
            // Nothing to show; the restore's WM_SIZE draws.
            return Some(0);
        }
        self.draw(false, false);
        Some(0)
    }

    fn handle_close(&self) -> Option<isize> {
        let should_close = self.with_callback(|c| &mut c.should_close, |f| f())?;
        if should_close { None } else { Some(0) }
    }

    fn handle_destroy(&self, hwnd: HWND) -> Option<isize> {
        self.destroyed.set(true);
        super::drop_target::revoke(self);
        let close = self.state.borrow_mut().callbacks.close.take();
        if let Some(close) = close {
            close();
        }
        if let Some(owner) = self.owner.take() {
            // SAFETY: the hidden owner window is ours.
            let _ = unsafe { DestroyWindow(owner) };
        }
        self.shared.window_closed(hwnd);
        Some(0)
    }

    pub(crate) fn start_tracking_mouse(&self, hwnd: HWND, flags: TRACKMOUSEEVENT_FLAGS) {
        if std::mem::replace(&mut self.state.borrow_mut().hovered, true) {
            return;
        }
        if self.offscreen {
            // The real cursor is never over an off-screen window, so Windows would post WM_MOUSELEAVE
            // at once and undo every hover a posted mouse move made: there, only a posted
            // WM_MOUSELEAVE ends the hover.
            self.with_callback(|c| &mut c.hovered_status_change, |f| f(true));
            return;
        }
        let mut track = TRACKMOUSEEVENT {
            cbSize: std::mem::size_of::<TRACKMOUSEEVENT>() as u32,
            dwFlags: flags,
            hwndTrack: hwnd,
            dwHoverTime: HOVER_DEFAULT,
        };
        // SAFETY: a sized TRACKMOUSEEVENT for our window.
        let _ = unsafe { TrackMouseEvent(&mut track) };
        self.with_callback(|c| &mut c.hovered_status_change, |f| f(true));
    }

    fn handle_mouse_move(&self, hwnd: HWND, lparam: LPARAM, wparam: WPARAM) -> Option<isize> {
        self.start_tracking_mouse(hwnd, TME_LEAVE);
        let scale_factor = self.state.borrow().scale_factor;
        let flags = MODIFIERKEYS_FLAGS(wparam.loword() as u32);
        let pressed_button = if flags.contains(MK_LBUTTON) {
            Some(MouseButton::Left)
        } else if flags.contains(MK_RBUTTON) {
            Some(MouseButton::Right)
        } else if flags.contains(MK_MBUTTON) {
            Some(MouseButton::Middle)
        } else if flags.contains(MK_XBUTTON1) {
            Some(MouseButton::Navigate(NavigationDirection::Back))
        } else if flags.contains(MK_XBUTTON2) {
            Some(MouseButton::Navigate(NavigationDirection::Forward))
        } else {
            None
        };
        let input = PlatformInput::MouseMove(MouseMoveEvent {
            position: logical_point(
                lparam.signed_loword() as f32,
                lparam.signed_hiword() as f32,
                scale_factor,
            ),
            pressed_button,
            modifiers: super::keyboard::current_modifiers(),
        });
        handled_result(self.dispatch_input(input))
    }

    fn handle_mouse_leave(&self) -> Option<isize> {
        self.state.borrow_mut().hovered = false;
        self.with_callback(|c| &mut c.hovered_status_change, |f| f(false));
        Some(0)
    }

    fn handle_mouse_down(&self, hwnd: HWND, button: MouseButton, lparam: LPARAM) -> Option<isize> {
        // SAFETY: capturing the mouse for our window so a drag continues outside it.
        unsafe { SetCapture(hwnd) };
        let (x, y) = (lparam.signed_loword() as i32, lparam.signed_hiword() as i32);
        let (click_count, scale_factor) = {
            let mut state = self.state.borrow_mut();
            if !state.pressed_buttons.contains(&button) {
                state.pressed_buttons.push(button);
            }
            (state.click_state.update(button, point(DevicePixels(x), DevicePixels(y))), state.scale_factor)
        };
        let input = PlatformInput::MouseDown(MouseDownEvent {
            button,
            position: logical_point(x as f32, y as f32, scale_factor),
            modifiers: super::keyboard::current_modifiers(),
            click_count,
            first_mouse: false,
        });
        handled_result(self.dispatch_input(input))
    }

    fn handle_mouse_up(&self, button: MouseButton, lparam: LPARAM) -> Option<isize> {
        let (click_count, scale_factor, still_pressed) = {
            let mut state = self.state.borrow_mut();
            state.pressed_buttons.retain(|b| *b != button);
            (state.click_state.current_count, state.scale_factor, !state.pressed_buttons.is_empty())
        };
        if !still_pressed {
            // SAFETY: releasing our capture (its WM_CAPTURECHANGED finds nothing pressed).
            let _ = unsafe { ReleaseCapture() };
        }
        let input = PlatformInput::MouseUp(MouseUpEvent {
            button,
            position: logical_point(
                lparam.signed_loword() as f32,
                lparam.signed_hiword() as f32,
                scale_factor,
            ),
            modifiers: super::keyboard::current_modifiers(),
            click_count,
        });
        handled_result(self.dispatch_input(input))
    }

    /// The mouse capture went elsewhere: another window took it, or Windows ended it (`WM_CANCELMODE`
    /// for a menu or a dialog, a switch to another app). A button pressed in the window is then
    /// released where we never hear of it, so each gets its mouse up now and the app's drags, slider
    /// drags and selections end. (GPUI 0.2.2 does nothing here, and a press outlived the capture.)
    fn handle_capture_changed(&self, hwnd: HWND, lparam: LPARAM) -> Option<isize> {
        if lparam.0 != hwnd.0 as isize {
            self.cancel_presses(hwnd);
        }
        Some(0)
    }

    /// Ends the buttons pressed in the window: clears any armed non-client caption button, lets go of
    /// the capture if we hold it (whose `WM_CAPTURECHANGED` releases them), and releases any still
    /// pressed.
    fn end_presses(&self, hwnd: HWND) {
        let empty = {
            let mut state = self.state.borrow_mut();
            state.nc_button_pressed = None;
            state.pressed_buttons.is_empty()
        };
        if empty {
            return;
        }
        // SAFETY: plain capture calls for our own window.
        unsafe {
            if GetCapture() == hwnd {
                let _ = ReleaseCapture();
            }
        }
        self.cancel_presses(hwnd);
    }

    /// Cursor position in `hwnd`'s client device pixels.
    pub(crate) fn client_cursor_pos(&self, hwnd: HWND) -> POINT {
        #[cfg(test)]
        if let Some(p) = self.shared.test_client_cursor.get() {
            return p;
        }
        let mut p = POINT::default();
        // SAFETY: valid out pointers; reading the cursor position does not move it.
        unsafe {
            let _ = GetCursorPos(&mut p);
            let _ = ScreenToClient(hwnd, &mut p);
        }
        p
    }

    /// A mouse up for each button pressed in the window, outside the window: where a real release lands
    /// when the button comes up elsewhere. The press was cancelled, not completed, so it must click and
    /// drop nothing, even with the cursor still over the element pressed; what ends a press wherever it
    /// is released (`on_mouse_up_out`, the pressed style) still runs. At the cursor if it is outside the
    /// client area, else just above and left of it.
    fn cancel_presses(&self, hwnd: HWND) {
        let (buttons, click_count, scale_factor) = {
            let mut state = self.state.borrow_mut();
            state.nc_button_pressed = None;
            (std::mem::take(&mut state.pressed_buttons), state.click_state.current_count, state.scale_factor)
        };
        if buttons.is_empty() {
            return;
        }
        let mut p = self.client_cursor_pos(hwnd);
        let mut client = RECT::default();
        // SAFETY: valid out pointer.
        unsafe {
            let _ = GetClientRect(hwnd, &mut client);
        }
        if p.x >= client.left && p.x < client.right && p.y >= client.top && p.y < client.bottom {
            p = POINT { x: client.left - 1, y: client.top - 1 };
        }
        let position = logical_point(p.x as f32, p.y as f32, scale_factor);
        for button in buttons {
            self.dispatch_input(PlatformInput::MouseUp(MouseUpEvent {
                button,
                position,
                modifiers: super::keyboard::current_modifiers(),
                click_count,
            }));
        }
    }

    fn handle_mouse_wheel(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let modifiers = super::keyboard::current_modifiers();
        let (scale_factor, wheel) = {
            let state = self.state.borrow();
            (state.scale_factor, state.wheel)
        };
        // Shift turns the vertical wheel horizontal, by the horizontal setting (GPUI's rule).
        let amount = if modifiers.shift { wheel.scroll_chars } else { wheel.scroll_lines };
        let distance = (wparam.signed_hiword() as f32 / WHEEL_DELTA as f32) * amount as f32;
        let position = screen_to_logical(hwnd, lparam, scale_factor);
        let input = PlatformInput::ScrollWheel(ScrollWheelEvent {
            position,
            delta: ScrollDelta::Lines(if modifiers.shift {
                point(distance, 0.)
            } else {
                point(0., distance)
            }),
            modifiers,
            touch_phase: TouchPhase::Moved,
        });
        handled_result(self.dispatch_input(input))
    }

    fn handle_mouse_horizontal_wheel(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        let (scale_factor, wheel) = {
            let state = self.state.borrow();
            (state.scale_factor, state.wheel)
        };
        let distance = (-wparam.signed_hiword() as f32 / WHEEL_DELTA as f32) * wheel.scroll_chars as f32;
        let input = PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: screen_to_logical(hwnd, lparam, scale_factor),
            delta: ScrollDelta::Lines(point(distance, 0.)),
            modifiers: super::keyboard::current_modifiers(),
            touch_phase: TouchPhase::Moved,
        });
        handled_result(self.dispatch_input(input))
    }

    fn handle_set_cursor(&self, hwnd: HWND, lparam: LPARAM) -> Option<isize> {
        // SAFETY: a plain query.
        let enabled = unsafe { IsWindowEnabled(hwnd) }.as_bool();
        if !enabled
            || matches!(
                lparam.loword() as u32,
                HTLEFT | HTRIGHT | HTTOP | HTTOPLEFT | HTTOPRIGHT | HTBOTTOM | HTBOTTOMLEFT | HTBOTTOMRIGHT
            )
        {
            return None;
        }
        // SAFETY: setting a shared system cursor (or none).
        unsafe { SetCursor(self.shared.cursor.get()) };
        Some(1)
    }

    fn handle_settings_changed(&self, hwnd: HWND, wparam: WPARAM, lparam: LPARAM) -> Option<isize> {
        if wparam.0 != 0 {
            let display = {
                let mut state = self.state.borrow_mut();
                state.click_state.system_update(wparam.0);
                state.border_offset.update(hwnd);
                state.display
            };
            match wparam.0 as u32 {
                // SPI_SETWORKAREA
                47 => {
                    self.state.borrow_mut().auto_hide_taskbar =
                        super::chrome::auto_hide_taskbar_edge(&display)
                }
                // SPI_SETWHEELSCROLLLINES, SPI_SETWHEELSCROLLCHARS
                105 | 109 => self.state.borrow_mut().wheel = WheelSettings::read(),
                _ => {}
            }
        } else {
            self.handle_theme_changed(hwnd, lparam);
        }
        // Recompute the client area (an auto-hide taskbar may have appeared).
        super::chrome::notify_frame_changed(hwnd);
        Some(0)
    }

    fn handle_theme_changed(&self, hwnd: HWND, lparam: LPARAM) {
        let parameter = PCWSTR::from_raw(lparam.0 as *const u16);
        // SAFETY: for WM_SETTINGCHANGE with wparam 0, lparam is null or a NUL-terminated string.
        if parameter.is_null()
            || unsafe { parameter.to_string() }.ok().as_deref() != Some("ImmersiveColorSet")
        {
            return;
        }
        let appearance = system_appearance();
        let (old, frame) = {
            let mut state = self.state.borrow_mut();
            (std::mem::replace(&mut state.appearance, appearance), state.frame_appearance)
        };
        if old != appearance {
            // The frame follows the system unless the app picked its appearance; the app's
            // callback may pick it anew (`set_frame_appearance`), so set the frame first.
            configure_dwm_dark_mode(hwnd, frame.unwrap_or(appearance));
            self.with_callback(|c| &mut c.appearance_changed, |f| f());
        }
    }
}

/// `Some(0)` if the core handled the event (stopped its propagation), `Some(1)` otherwise.
fn handled_result(result: Option<DispatchEventResult>) -> Option<isize> {
    match result {
        Some(r) if !r.propagate => Some(0),
        _ => Some(1),
    }
}

fn xbutton(wparam: WPARAM) -> Option<MouseButton> {
    match wparam.hiword() {
        XBUTTON1 => Some(MouseButton::Navigate(NavigationDirection::Back)),
        XBUTTON2 => Some(MouseButton::Navigate(NavigationDirection::Forward)),
        _ => None,
    }
}

/// A screen position in `lparam` as window logical pixels.
pub(crate) fn screen_to_logical(hwnd: HWND, lparam: LPARAM, scale_factor: f32) -> Point<Pixels> {
    let mut p = POINT { x: lparam.signed_loword() as i32, y: lparam.signed_hiword() as i32 };
    // SAFETY: a valid in/out pointer.
    let _ = unsafe { ScreenToClient(hwnd, &mut p) };
    logical_point(p.x as f32, p.y as f32, scale_factor)
}
