//! `Div` and `Stateful<E>`: the flex box everything is built from. Owner tag: `"window"` (with
//! `interactive/`). A `Div` is an `Interactivity` plus children; its phases run the interactivity around
//! the children's (GPUI 0.2.2's `div.rs`).
#![allow(dead_code)] // skeleton

use smallvec::SmallVec;

use crate::{
    AnyElement, Element, Hitbox, InteractiveElement, Interactivity, IntoElement, LayoutId, ParentElement,
    StatefulInteractiveElement, StyleRefinement, Styled, elements::skeleton_element,
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

skeleton_element!([] Div, DivFrameState, DivPrepaintState, "window");

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
