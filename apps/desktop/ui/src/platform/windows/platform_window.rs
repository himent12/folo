// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/window.rs (PlatformWindow for WindowsWindow)
//! The `PlatformWindow` methods of `WindowsWindow` (the core's view of the window) and its `Drop`.

use super::drop_target;
use super::util::*;
use super::window::{WindowOpenState, WindowsWindow, calculate_window_rect};
use crate::*;
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::HSTRING;
use std::rc::Rc;

impl Drop for WindowsWindow {
    fn drop(&mut self) {
        // Destroying from inside a callback would re-enter the window procedure: do it from the queue.
        let inner = self.0.clone();
        super::post_call(
            self.0.shared.message_window(),
            Box::new(move || {
                if !inner.destroyed.get() {
                    drop_target::revoke(&inner);
                    // SAFETY: destroying our own window on its thread.
                    let _ = unsafe { DestroyWindow(inner.hwnd) };
                }
            }),
        );
    }
}

impl PlatformWindow for WindowsWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        let state = self.0.state.borrow();
        Bounds { origin: state.origin, size: state.logical_size }
    }

    fn is_maximized(&self) -> bool {
        self.0.is_maximized()
    }

    fn window_bounds(&self) -> WindowBounds {
        self.0.window_bounds()
    }

    fn content_size(&self) -> Size<Pixels> {
        self.0.state.borrow().logical_size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        let (rect, origin) = {
            let state = self.0.state.borrow();
            let bounds = crate::bounds(state.origin, size).to_device_pixels(state.scale_factor);
            (calculate_window_rect(bounds, state.border_offset), bounds.origin)
        };
        let hwnd = self.0.hwnd;
        super::post_call(
            self.0.shared.message_window(),
            Box::new(move || {
                // SAFETY: resizing our own window.
                let _ = unsafe {
                    SetWindowPos(
                        hwnd,
                        None,
                        origin.x.0,
                        origin.y.0,
                        rect.right - rect.left,
                        rect.bottom - rect.top,
                        SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
                    )
                };
            }),
        );
    }

    fn scale_factor(&self) -> f32 {
        self.0.state.borrow().scale_factor
    }

    fn appearance(&self) -> WindowAppearance {
        self.0.state.borrow().appearance
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(Rc::new(self.0.state.borrow().display))
    }

    fn mouse_position(&self) -> Point<Pixels> {
        let point = self.0.client_cursor_pos(self.0.hwnd);
        logical_point(point.x as f32, point.y as f32, self.scale_factor())
    }

    fn modifiers(&self) -> Modifiers {
        super::keyboard::current_modifiers()
    }

    fn capslock(&self) -> Capslock {
        super::keyboard::current_capslock()
    }

    fn set_input_handler(&mut self, input_handler: Box<dyn PlatformInputHandler>) {
        self.0.state.borrow_mut().input_handler = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<Box<dyn PlatformInputHandler>> {
        self.0.state.borrow_mut().input_handler.take()
    }

    fn activate(&self) {
        // Off screen, nothing may take the foreground (and no Alt key may be injected).
        if self.0.offscreen {
            self.0.apply_initial_placement();
            return;
        }
        let inner = self.0.clone();
        super::post_call(
            self.0.shared.message_window(),
            Box::new(move || {
                inner.apply_initial_placement();
                let hwnd = inner.hwnd;
                // SAFETY: activating our own window; the Alt press and release is GPUI's way of being
                // allowed to take the foreground.
                unsafe {
                    if IsIconic(hwnd).as_bool() {
                        let _ = ShowWindowAsync(hwnd, SW_RESTORE);
                    }
                    let _ = SetActiveWindow(hwnd);
                    let _ = SetFocus(Some(hwnd));
                    let key = |flags| INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT { wVk: VK_MENU, dwFlags: flags, ..Default::default() },
                        },
                    };
                    let inputs = [key(KEYBD_EVENT_FLAGS(0)), key(KEYEVENTF_KEYUP)];
                    SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
                    let _ = SetForegroundWindow(hwnd);
                }
            }),
        );
    }

    fn is_active(&self) -> bool {
        // SAFETY: a plain query.
        self.0.hwnd == unsafe { GetActiveWindow() }
    }

    fn is_hovered(&self) -> bool {
        self.0.state.borrow().hovered
    }

    fn set_title(&mut self, title: &str) {
        // SAFETY: the string outlives the call.
        let _ = unsafe { SetWindowTextW(self.0.hwnd, &HSTRING::from(title)) };
    }

    fn minimize(&self) {
        if self.0.offscreen {
            let inner = self.0.clone();
            super::post_call(self.0.shared.message_window(), Box::new(move || inner.offscreen_minimize()));
            return;
        }
        // SAFETY: showing our own window.
        let _ = unsafe { ShowWindowAsync(self.0.hwnd, SW_MINIMIZE) };
    }

    fn zoom(&self) {
        // SAFETY: plain queries and showing our own window.
        unsafe {
            if self.0.offscreen && IsWindowVisible(self.0.hwnd).as_bool() {
                let inner = self.0.clone();
                super::post_call(
                    self.0.shared.message_window(),
                    Box::new(move || inner.offscreen_toggle_zoom()),
                );
            } else if IsWindowVisible(self.0.hwnd).as_bool() {
                let cmd = if self.0.is_maximized() { SW_RESTORE } else { SW_MAXIMIZE };
                let _ = ShowWindowAsync(self.0.hwnd, cmd);
            } else if let Some(status) = self.0.state.borrow_mut().initial_placement.as_mut() {
                status.state = WindowOpenState::Maximized;
            }
        }
    }

    fn toggle_fullscreen(&self) {
        // SAFETY: a plain query.
        if unsafe { IsWindowVisible(self.0.hwnd) }.as_bool() {
            let inner = self.0.clone();
            super::post_call(self.0.shared.message_window(), Box::new(move || inner.toggle_fullscreen_now()));
        } else if let Some(status) = self.0.state.borrow_mut().initial_placement.as_mut() {
            status.state = WindowOpenState::Fullscreen;
        }
    }

    fn is_fullscreen(&self) -> bool {
        self.0.state.borrow().is_fullscreen()
    }

    fn request_frame(&self) {
        self.0.pacer.request(&self.0.frame);
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.0.state.borrow_mut().callbacks.request_frame = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.0.state.borrow_mut().callbacks.input = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.state.borrow_mut().callbacks.active_status_change = Some(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.state.borrow_mut().callbacks.hovered_status_change = Some(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.0.state.borrow_mut().callbacks.resize = Some(callback);
    }

    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        self.0.state.borrow_mut().callbacks.moved = Some(callback);
    }

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        self.0.state.borrow_mut().callbacks.should_close = Some(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        self.0.state.borrow_mut().callbacks.hit_test_window_control = Some(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.0.state.borrow_mut().callbacks.close = Some(callback);
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        self.0.state.borrow_mut().callbacks.appearance_changed = Some(callback);
    }

    fn set_frame_appearance(&self, appearance: Option<WindowAppearance>) {
        let frame = {
            let mut state = self.0.state.borrow_mut();
            state.frame_appearance = appearance;
            appearance.unwrap_or(state.appearance)
        };
        configure_dwm_dark_mode(self.0.hwnd, frame);
    }

    fn present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]) {
        self.0.present(frame, damage);
    }

    fn update_ime_position(&self, bounds: Bounds<Pixels>) {
        self.0.state.borrow_mut().ime_position = Some(bounds);
        super::keyboard::position_ime(&self.0);
    }
}
