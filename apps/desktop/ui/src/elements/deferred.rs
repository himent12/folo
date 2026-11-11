//! `deferred(child)`: lays the child out in place but paints it after the whole tree (popovers, menus),
//! ordered by priority. Owner tag: `"elements"`.
#![allow(dead_code)] // skeleton

use crate::{AnyElement, IntoElement, elements::skeleton_element};

/// Defers `child`'s paint.
pub fn deferred(child: impl IntoElement) -> Deferred {
    Deferred { child: Some(child.into_any_element()), priority: 0 }
}

/// The deferred element.
pub struct Deferred {
    /// The child (handed to `Window::defer_draw` in prepaint).
    pub(crate) child: Option<AnyElement>,
    /// Higher paints later.
    pub(crate) priority: usize,
}

impl Deferred {
    /// Sets the priority (higher paints on top).
    pub fn with_priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }
}

skeleton_element!([] Deferred, (), (), "elements");
