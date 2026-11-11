//! Input dispatch, in the order of GPUI's `window.rs`. Owner tag: `"window"`.
//!
//! Mouse events are hit-tested against the rendered frame and dispatched to its mouse listeners in capture
//! (paint order) then bubble (reverse paint order) phases, honouring `stop_propagation`, `prevent_default`,
//! blocking hitboxes, hover recomputation (a hover change refreshes the window), pressed-button capture, and
//! file drags as an active drag carrying `ExternalPaths`. Key events walk the dispatch tree along the focus
//! path: key listeners (capture, then bubble), then keymap matching with the context stack and action
//! dispatch (capture, bubble, then the app's global handlers), then text insertion through the platform
//! input handler when nothing handled the key. Window control areas answer the platform's hit test.
#![allow(dead_code)] // skeleton

use crate::{
    Action, App, DispatchEventResult, FileDropEvent, KeyDownEvent, Keystroke, PlatformInput, Window,
    WindowControlArea,
};

impl Window {
    /// The `on_input` callback: updates mouse position and modifiers, then dispatches.
    pub(crate) fn dispatch_event(&mut self, _event: PlatformInput, _cx: &mut App) -> DispatchEventResult {
        todo!("window")
    }

    /// Dispatches a mouse event to the rendered frame's mouse listeners.
    pub(crate) fn dispatch_mouse_event(&mut self, _event: &dyn std::any::Any, _cx: &mut App) {
        todo!("window")
    }

    /// Turns a file drop into an active drag of `ExternalPaths` and the matching mouse events.
    pub(crate) fn dispatch_file_drop(&mut self, _event: &FileDropEvent, _cx: &mut App) {
        todo!("window")
    }

    /// Dispatches a key event along the focus path, then the keymap, then text input.
    pub(crate) fn dispatch_key_event(&mut self, _event: &dyn std::any::Any, _cx: &mut App) {
        todo!("window")
    }

    /// Matches `keystroke` against the keymap under the focus path's contexts and dispatches the first
    /// action someone handles. Returns whether one was handled.
    pub(crate) fn dispatch_keystroke_bindings(&mut self, _keystroke: &Keystroke, _cx: &mut App) -> bool {
        todo!("window")
    }

    /// Sends the key's text to the platform input handler (`replace_text_in_range(None, text)`) when no
    /// listener or binding handled it.
    pub(crate) fn insert_text_for_key(&mut self, _event: &KeyDownEvent, _cx: &mut App) {
        todo!("window")
    }

    /// Dispatches `action` from the focused element (or the root): capture, bubble, then global handlers.
    pub fn dispatch_action(&mut self, _action: Box<dyn Action>, _cx: &mut App) {
        todo!("window")
    }

    /// Recomputes the hit test for the current mouse position; refreshes when hover changed.
    pub(crate) fn reset_hover(&mut self, _cx: &mut App) {
        todo!("window")
    }

    /// The `on_hit_test_window_control` callback: the control area under the mouse in the rendered frame.
    pub(crate) fn hit_test_window_control(&self) -> Option<WindowControlArea> {
        todo!("window")
    }

    /// Whether the event dispatched now should keep propagating (cleared by `cx.stop_propagation()`).
    pub(crate) fn propagation_stopped(&self, cx: &App) -> bool {
        !cx.propagate_event
    }
}
