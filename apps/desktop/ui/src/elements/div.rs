//! `Div` and `Stateful<E>`: the flex box everything is built from. Owner tag: `"window"` (with
//! `interactive/`). A `Div` is an `Interactivity` plus children; its phases run the interactivity around
//! the children's (GPUI 0.2.2's `div.rs`).
use smallvec::SmallVec;

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, Hitbox, InspectorElementId, InteractiveElement,
    Interactivity, IntoElement, LayoutId, ParentElement, Pixels, Point, StatefulInteractiveElement, StyleRefinement,
    Styled, Window, interactive::skips_children, point,
};

/// A new `Div`.
pub fn div() -> Div {
    Div { interactivity: Interactivity::default(), children: SmallVec::new() }
}

/// The flex box element.
pub struct Div {
    /// Listeners, styles, state.
    pub(crate) interactivity: Interactivity,
    /// The children.
    pub(crate) children: SmallVec<[AnyElement; 2]>,
}

/// What `Div::request_layout` keeps: the children's layout ids.
pub struct DivFrameState {
    /// The children's layout ids (from `request_layout`).
    pub(crate) child_layout_ids: SmallVec<[LayoutId; 2]>,
}

/// What `Div::prepaint` keeps for paint.
pub struct DivPrepaintState {
    /// The hitbox, when the div has one.
    pub(crate) hitbox: Option<Hitbox>,
}

impl Styled for Div {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for Div {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

impl ParentElement for Div {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements)
    }
}

impl Element for Div {
    type RequestLayoutState = DivFrameState;
    type PrepaintState = DivPrepaintState;

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut child_layout_ids = SmallVec::new();
        let children = &mut self.children;
        let layout_id = self.interactivity.request_layout(global_id, inspector_id, window, cx, |style, window, cx| {
            window.with_text_style(style.text_style().cloned(), |window| {
                child_layout_ids.extend(children.iter_mut().map(|child| child.request_layout(window, cx)));
                window.request_layout(style, child_layout_ids.iter().copied(), cx)
            })
        });
        (layout_id, DivFrameState { child_layout_ids })
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        // The children's bounding box: the scrollable content size (and the handle's child bounds).
        let content_size = if request_layout.child_layout_ids.is_empty() {
            bounds.size
        } else {
            let mut child_min = point(Pixels::MAX, Pixels::MAX);
            let mut child_max = Point::default();
            let mut handle_state = self.interactivity.tracked_scroll_handle.as_ref().map(|handle| handle.0.borrow_mut());
            if let Some(state) = handle_state.as_deref_mut() {
                state.child_bounds.clear();
            }
            for child_layout_id in &request_layout.child_layout_ids {
                let child_bounds = window.layout_bounds(*child_layout_id);
                child_min = child_min.min(&child_bounds.origin);
                child_max = child_max.max(&child_bounds.bottom_right());
                if let Some(state) = handle_state.as_deref_mut() {
                    state.child_bounds.push(child_bounds);
                }
            }
            (child_max - child_min).into()
        };
        if let Some(handle) = self.interactivity.tracked_scroll_handle.as_ref() {
            handle.scroll_to_active_item();
        }

        let children = &mut self.children;
        let hitbox = self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            content_size,
            window,
            cx,
            |style, scroll_offset, hitbox, window, cx| {
                if !skips_children(style) {
                    window.with_element_offset(scroll_offset, |window| {
                        for child in children.iter_mut() {
                            child.prepaint(window, cx);
                        }
                    });
                }
                hitbox
            },
        );
        DivPrepaintState { hitbox }
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let children = &mut self.children;
        self.interactivity.paint(global_id, inspector_id, bounds, prepaint.hitbox.as_ref(), window, cx, |style, window, cx| {
            if !skips_children(style) {
                for child in children.iter_mut() {
                    child.paint(window, cx);
                }
            }
        });
    }
}

impl IntoElement for Div {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// An element with an id (`div().id("x")`): enables the stateful listeners.
pub struct Stateful<E> {
    /// The element.
    pub(crate) element: E,
}

impl<E: InteractiveElement> InteractiveElement for Stateful<E> {
    fn interactivity(&mut self) -> &mut Interactivity {
        self.element.interactivity()
    }
}

impl<E: InteractiveElement> StatefulInteractiveElement for Stateful<E> {}

impl<E: Styled> Styled for Stateful<E> {
    fn style(&mut self) -> &mut StyleRefinement {
        self.element.style()
    }
}

impl<E: ParentElement> ParentElement for Stateful<E> {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.element.extend(elements)
    }
}

impl<E: Element> Element for Stateful<E> {
    type RequestLayoutState = E::RequestLayoutState;
    type PrepaintState = E::PrepaintState;

    fn id(&self) -> Option<crate::ElementId> {
        self.element.id()
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        id: Option<&crate::GlobalElementId>,
        inspector_id: Option<&crate::InspectorElementId>,
        window: &mut crate::Window,
        cx: &mut crate::App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        self.element.request_layout(id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&crate::GlobalElementId>,
        inspector_id: Option<&crate::InspectorElementId>,
        bounds: crate::Bounds<crate::Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut crate::Window,
        cx: &mut crate::App,
    ) -> Self::PrepaintState {
        self.element.prepaint(id, inspector_id, bounds, request_layout, window, cx)
    }

    fn paint(
        &mut self,
        id: Option<&crate::GlobalElementId>,
        inspector_id: Option<&crate::InspectorElementId>,
        bounds: crate::Bounds<crate::Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut crate::Window,
        cx: &mut crate::App,
    ) {
        self.element.paint(id, inspector_id, bounds, request_layout, prepaint, window, cx)
    }
}

impl<E: Element> IntoElement for Stateful<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
