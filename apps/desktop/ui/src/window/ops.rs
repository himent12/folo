//! Window queries and operations the app calls (size, scale, rem, line height, mouse, modifiers, bounds,
//! display, maximize / full screen / minimize / zoom, title, close) and focus. Owner tag: `"window"`.
//! Most forward to the platform window or read the window's state.
use std::{rc::Rc, sync::Arc};

use crate::{
    App, Bounds, FocusHandle, FocusId, Modifiers, Pixels, PlatformDisplay, Point, Size, Window, WindowBounds,
    WindowTextSystem,
};

impl Window {
    /// The client area's size (logical px).
    pub fn viewport_size(&self) -> Size<Pixels> {
        self.viewport_size
    }

    /// Device pixels per logical pixel.
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// The root font size (16 px).
    pub fn rem_size(&self) -> Pixels {
        self.rem_size
    }

    /// The text style's line height at the current font size.
    pub fn line_height(&self) -> Pixels {
        self.text_style().line_height_in_pixels(self.rem_size())
    }

    /// The per-window text system (shaping, line layout cache).
    pub fn text_system(&self) -> &Arc<WindowTextSystem> {
        &self.text_system
    }

    /// The mouse position (window logical px).
    pub fn mouse_position(&self) -> Point<Pixels> {
        self.mouse_position
    }

    /// Modifier keys held.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// The window's outer bounds (screen logical px).
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.platform_window.bounds()
    }

    /// The restored bounds and mode (what the app saves).
    pub fn window_bounds(&self) -> WindowBounds {
        self.platform_window.window_bounds()
    }

    /// The display the window is on.
    pub fn display(&self, _cx: &App) -> Option<Rc<dyn PlatformDisplay>> {
        self.platform_window.display()
    }

    /// Whether the window is maximized.
    pub fn is_maximized(&self) -> bool {
        self.platform_window.is_maximized()
    }

    /// Whether the window is full screen.
    pub fn is_fullscreen(&self) -> bool {
        self.platform_window.is_fullscreen()
    }

    /// Whether the window is active.
    pub fn is_window_active(&self) -> bool {
        self.active.get()
    }

    /// How many times the window became active since it opened (each return counts).
    pub fn window_activations(&self) -> u64 {
        self.activations
    }

    /// Enters or leaves full screen.
    pub fn toggle_fullscreen(&self) {
        self.platform_window.toggle_fullscreen()
    }

    /// Minimizes the window.
    pub fn minimize_window(&self) {
        self.platform_window.minimize()
    }

    /// Maximizes or restores the window.
    pub fn zoom_window(&self) {
        self.platform_window.zoom()
    }

    /// Sets the title.
    pub fn set_window_title(&mut self, title: &str) {
        self.platform_window.set_title(title)
    }

    /// Closes the window after the current update.
    pub fn remove_window(&mut self) {
        self.removed = true;
    }

    /// Focuses `handle`; the window redraws (GPUI's `Window::focus`: nothing happens if it already is).
    pub fn focus(&mut self, handle: &FocusHandle) {
        self.focus_id(handle.id);
    }

    /// Focuses the element with `id` (what `focus` and Tab navigation do).
    pub(crate) fn focus_id(&mut self, id: FocusId) {
        if self.focus == Some(id) {
            return;
        }
        self.focus = Some(id);
        self.refresh();
    }

    /// Clears focus.
    pub fn blur(&mut self) {
        self.focus = None;
        self.refresh();
    }

    /// The focused handle, if any (and still alive).
    pub fn focused(&self, cx: &App) -> Option<FocusHandle> {
        self.focus.and_then(|id| FocusHandle::for_id(id, &cx.focus_handles))
    }

    /// The focused id, if its handle is still alive (`focused` without making a handle).
    pub(crate) fn focused_id(&self, cx: &App) -> Option<FocusId> {
        self.focus.filter(|id| cx.focus_handles.handles.read().contains_key(id))
    }

    /// Moves focus to the next tab stop of the rendered frame (wrapping).
    pub fn focus_next(&mut self) {
        if let Some(id) = self.rendered_frame.dispatch_tree.next_tab_stop(self.focus, false) {
            self.focus_id(id);
        }
    }

    /// Moves focus to the previous tab stop (wrapping).
    pub fn focus_prev(&mut self) {
        if let Some(id) = self.rendered_frame.dispatch_tree.next_tab_stop(self.focus, true) {
            self.focus_id(id);
        }
    }
}
