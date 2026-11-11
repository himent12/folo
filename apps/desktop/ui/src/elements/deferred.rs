//! `deferred(child)`: lays the child out in place but paints it after the whole tree (popovers, menus),
//! ordered by priority. Owner tag: `"elements"`.

use std::panic;

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId, Pixels,
    Window,
};

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

impl Element for Deferred {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.as_mut().expect("deferred lays out once").request_layout(window, cx), ())
    }

    /// Hands the child to the window, which paints it after the tree at the current element offset.
    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        window: &mut Window,
        _cx: &mut App,
    ) {
        let child = self.child.take().expect("deferred prepaints once");
        let element_offset = window.element_offset();
        window.defer_draw(child, element_offset, self.priority);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _prepaint: &mut (),
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

impl IntoElement for Deferred {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
