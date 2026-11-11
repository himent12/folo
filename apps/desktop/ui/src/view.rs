//! Views as elements: `AnyView` (an entity plus its render function) and `Entity<V: Render>` as a child.
//! Owner tag: `"elements"`.
//!
//! There is no per-view caching: any `notify` redraws the whole window (DESIGN.md, the runtime's rule), so a
//! view element just renders its entity every frame under `ElementId::View(entity_id)`.

use std::panic;

use crate::{
    AnyElement, AnyEntity, App, Bounds, Element, ElementId, Entity, EntityId, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, Pixels, Render, Window,
};

/// A type-erased view: the entity and the function that renders it.
#[derive(Clone)]
pub struct AnyView {
    /// The view's entity.
    pub(crate) entity: AnyEntity,
    /// Leases the entity, calls `Render::render` and erases the result.
    pub(crate) render: fn(&AnyView, &mut Window, &mut App) -> AnyElement,
}

impl AnyView {
    /// The view's entity id.
    pub fn entity_id(&self) -> EntityId {
        self.entity.entity_id()
    }

    /// The typed entity, if the view is a `T`.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, Self> {
        match self.entity.downcast::<T>() {
            Ok(entity) => Ok(entity),
            Err(entity) => Err(AnyView { entity, render: self.render }),
        }
    }
}

impl<V: Render> From<Entity<V>> for AnyView {
    fn from(value: Entity<V>) -> Self {
        AnyView { entity: value.into_any(), render: render_view::<V> }
    }
}

/// `AnyView::render` for views of type `V`.
fn render_view<V: Render>(view: &AnyView, window: &mut Window, cx: &mut App) -> AnyElement {
    let view = view.clone().downcast::<V>().unwrap_or_else(|_| panic!("AnyView of the wrong type"));
    let element = view.update(cx, |view, cx| view.render(window, cx).into_any_element());
    crate::app::trace::first_render();
    element
}

impl Element for AnyView {
    /// The rendered tree of this frame.
    type RequestLayoutState = Option<AnyElement>;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(ElementId::View(self.entity_id()))
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
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut element = (self.render)(self, window, cx);
        let layout_id = element.request_layout(window, cx);
        (layout_id, Some(element))
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if let Some(element) = request_layout {
            element.prepaint(window, cx);
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if let Some(element) = request_layout {
            element.paint(window, cx);
        }
    }
}

impl IntoElement for AnyView {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
