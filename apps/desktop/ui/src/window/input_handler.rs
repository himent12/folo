//! The seam between the focused element's text input handler and the platform: `InputHandlerSlot` (what
//! `Window::handle_input` records in the frame) and `WindowInputHandler`, which implements the foundation's
//! `PlatformInputHandler` by leasing the window out of the app for each IME / text call. Owner tag:
//! `"window"`.
#![allow(dead_code)] // skeleton

use std::{ops::Range, rc::Weak};

use crate::{AppCell, Bounds, FocusId, InputHandler, Pixels, PlatformInputHandler, Point, UTF16Selection, WindowId};

/// An input handler painted this frame, for the element with `focus_id`.
pub(crate) struct InputHandlerSlot {
    /// The element's focus handle id (only the focused element's handler is given to the platform).
    pub(crate) focus_id: FocusId,
    /// The handler.
    pub(crate) handler: Box<dyn InputHandler>,
}

/// The platform's view of the focused element's input handler: each call borrows the app, leases the
/// window, runs the handler with `(&mut Window, &mut App)`, and flushes effects.
pub(crate) struct WindowInputHandler {
    /// The app.
    pub(crate) app: Weak<AppCell>,
    /// The window it belongs to.
    pub(crate) window: WindowId,
    /// The handler (taken back by the window when the next frame paints a new one).
    pub(crate) handler: Box<dyn InputHandler>,
}

impl PlatformInputHandler for WindowInputHandler {
    fn selected_text_range(&mut self, _ignore_disabled_input: bool) -> Option<UTF16Selection> {
        todo!("window")
    }

    fn marked_text_range(&mut self) -> Option<Range<usize>> {
        todo!("window")
    }

    fn text_for_range(&mut self, _range_utf16: Range<usize>, _adjusted_range: &mut Option<Range<usize>>) -> Option<String> {
        todo!("window")
    }

    fn replace_text_in_range(&mut self, _replacement_range: Option<Range<usize>>, _text: &str) {
        todo!("window")
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        _range_utf16: Option<Range<usize>>,
        _new_text: &str,
        _new_selected_range: Option<Range<usize>>,
    ) {
        todo!("window")
    }

    fn unmark_text(&mut self) {
        todo!("window")
    }

    fn bounds_for_range(&mut self, _range_utf16: Range<usize>) -> Option<Bounds<Pixels>> {
        todo!("window")
    }

    fn character_index_for_point(&mut self, _point: Point<Pixels>) -> Option<usize> {
        todo!("window")
    }
}
