//! The seam between the focused element's text input handler and the platform: `InputHandlerSlot` (what
//! `Window::handle_input` records in the frame), `SharedInputHandler` (the frame's handler, shared by the
//! window and the platform) and `WindowInputHandler`, which implements the foundation's
//! `PlatformInputHandler` by leasing the window out of the app for each IME / text call. Owner tag:
//! `"window"`.
//!
//! The window keeps a clone of the shared handler so it can insert typed text itself, inside a window
//! update, when no listener or binding took a key on a platform with no text input of its own (the
//! headless one: `Window::insert_text_for_key`); the platform's calls come from outside any update and
//! lease the window first.

use std::{cell::RefCell, ops::Range, rc::Rc, rc::Weak};

use crate::{
    AnyWindowHandle, AppCell, Bounds, FocusId, InputHandler, Pixels, PlatformInputHandler, Point,
    UTF16Selection, window::update_window,
};

/// An input handler painted this frame, for the element with `focus_id`.
pub(crate) struct InputHandlerSlot {
    /// The element's focus handle id (only the focused element's handler is recorded).
    pub(crate) focus_id: FocusId,
    /// The handler.
    pub(crate) handler: Box<dyn InputHandler>,
}

/// The rendered frame's input handler, shared by the window and the platform's `WindowInputHandler`.
#[derive(Clone)]
pub(crate) struct SharedInputHandler(pub(crate) Rc<RefCell<Box<dyn InputHandler>>>);

impl SharedInputHandler {
    /// Shares `handler`.
    pub(crate) fn new(handler: Box<dyn InputHandler>) -> Self {
        SharedInputHandler(Rc::new(RefCell::new(handler)))
    }
}

/// The platform's view of the focused element's input handler: each call borrows the app, leases the
/// window, runs the handler with `(&mut Window, &mut App)`, and flushes effects. A call made while the app
/// or the window is gone (or already leased) answers `None` / does nothing.
pub(crate) struct WindowInputHandler {
    /// The app.
    pub(crate) app: Weak<AppCell>,
    /// The window it belongs to.
    pub(crate) window: AnyWindowHandle,
    /// The handler.
    pub(crate) handler: SharedInputHandler,
}

impl WindowInputHandler {
    fn with<R>(
        &mut self,
        f: impl FnOnce(&mut dyn InputHandler, &mut crate::Window, &mut crate::App) -> R,
    ) -> Option<R> {
        let handler = self.handler.0.clone();
        update_window(&self.app, self.window, move |window, cx| f(handler.borrow_mut().as_mut(), window, cx))
    }
}

impl PlatformInputHandler for WindowInputHandler {
    fn selected_text_range(&mut self, ignore_disabled_input: bool) -> Option<UTF16Selection> {
        self.with(|handler, window, cx| handler.selected_text_range(ignore_disabled_input, window, cx))
            .flatten()
    }

    fn marked_text_range(&mut self) -> Option<Range<usize>> {
        self.with(|handler, window, cx| handler.marked_text_range(window, cx)).flatten()
    }

    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
    ) -> Option<String> {
        self.with(|handler, window, cx| handler.text_for_range(range_utf16, adjusted_range, window, cx))
            .flatten()
    }

    fn replace_text_in_range(&mut self, replacement_range: Option<Range<usize>>, text: &str) {
        self.with(|handler, window, cx| handler.replace_text_in_range(replacement_range, text, window, cx));
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
    ) {
        self.with(|handler, window, cx| {
            handler.replace_and_mark_text_in_range(range_utf16, new_text, new_selected_range, window, cx)
        });
    }

    fn unmark_text(&mut self) {
        self.with(|handler, window, cx| handler.unmark_text(window, cx));
    }

    fn bounds_for_range(&mut self, range_utf16: Range<usize>) -> Option<Bounds<Pixels>> {
        self.with(|handler, window, cx| handler.bounds_for_range(range_utf16, window, cx)).flatten()
    }

    fn character_index_for_point(&mut self, point: Point<Pixels>) -> Option<usize> {
        self.with(|handler, window, cx| handler.character_index_for_point(point, window, cx)).flatten()
    }
}
