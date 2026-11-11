// Adapted from GPUI (Apache-2.0): crates/gpui/src/element.rs (the `Element` trait), crates/gpui/src/util.rs (`FluentBuilder`)
//! The element protocol: `Element`, `IntoElement`, `AnyElement`, `Render`, `RenderOnce`, `ParentElement`,
//! element ids and layout ids. Owner tag: `"elements"`.
//!
//! An element lives for one frame: built by `render`, then `request_layout` -> `prepaint` -> `paint`, then
//! dropped. Anything that must survive to the next frame goes in element state keyed by `GlobalElementId`
//! (`Window::with_element_state`).
#![allow(dead_code)] // skeleton

use std::{any::Any, fmt, panic, sync::Arc};

use smallvec::SmallVec;

use crate::{
    App, AvailableSpace, Bounds, Context, Entity, EntityId, FluentBuilder, FocusHandle, FocusId, Pixels,
    Point, SharedString, Size, Window,
};

/// An element: GPUI 0.2.2's trait, unchanged (the app implements it by hand in `src/input.rs`).
pub trait Element: 'static + IntoElement {
    /// State kept from `request_layout` to `prepaint` and `paint`.
    type RequestLayoutState: 'static;
    /// State kept from `prepaint` to `paint`.
    type PrepaintState: 'static;

    /// The id that keys this element's state across frames, if it has one.
    fn id(&self) -> Option<ElementId>;

    /// Where the element was built (inspector only; we return `None` everywhere).
    fn source_location(&self) -> Option<&'static panic::Location<'static>>;

    /// Asks the window for a layout node and returns its id.
    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState);

    /// Runs once the layout is known: hitboxes, child prepaint, anything paint needs.
    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState;

    /// Paints into the window's next frame (scene, listeners, cursor styles).
    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    );

    /// Type-erases the element.
    fn into_any(self) -> AnyElement {
        AnyElement::new(self)
    }
}

/// Something that can become an element (every builder the app returns from `render`).
pub trait IntoElement: Sized {
    /// The element it becomes.
    type Element: Element;

    /// Converts to the element.
    fn into_element(self) -> Self::Element;

    /// Converts to a type-erased element.
    fn into_any_element(self) -> AnyElement {
        self.into_element().into_any()
    }
}

/// A view's state renders its element tree each frame.
pub trait Render: 'static + Sized {
    /// Builds this frame's tree.
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement;
}

/// A component consumed by rendering (`#[derive(IntoElement)]` in GPUI; here implemented by hand).
pub trait RenderOnce: 'static {
    /// Builds the element tree, consuming the component.
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

/// Glue that makes a `RenderOnce` an element.
pub struct Component<C: RenderOnce> {
    component: Option<C>,
}

impl<C: RenderOnce> Component<C> {
    /// Wraps a component.
    pub fn new(component: C) -> Self {
        Self { component: Some(component) }
    }
}

impl<C: RenderOnce> Element for Component<C> {
    type RequestLayoutState = AnyElement;
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
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        todo!("elements")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        todo!("elements")
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        todo!("elements")
    }
}

impl<C: RenderOnce> IntoElement for Component<C> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// An element with children (`child`, `children`).
pub trait ParentElement {
    /// Appends children.
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>);

    /// Adds one child.
    fn child(mut self, child: impl IntoElement) -> Self
    where
        Self: Sized,
    {
        self.extend(std::iter::once(child.into_element().into_any()));
        self
    }

    /// Adds several children.
    fn children(mut self, children: impl IntoIterator<Item = impl IntoElement>) -> Self
    where
        Self: Sized,
    {
        self.extend(children.into_iter().map(|child| child.into_any_element()));
        self
    }
}

/// The element-erasing seam: what `AnyElement` drives, implemented by `Drawable<E>`.
pub(crate) trait ElementObject {
    /// The element's id.
    fn inner_element(&mut self) -> &mut dyn Any;
    /// Phase 1.
    fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId;
    /// Phase 2.
    fn prepaint(&mut self, window: &mut Window, cx: &mut App);
    /// Phase 3.
    fn paint(&mut self, window: &mut Window, cx: &mut App);
    /// Lays the element out as a root of its own taffy tree in `available_space`.
    fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels>;
}

/// Where an element is in its three phases; the states it carries between them.
pub(crate) enum ElementDrawPhase<RequestLayoutState, PrepaintState> {
    /// Built, not laid out.
    Start,
    /// `request_layout` ran.
    RequestLayout {
        /// Its layout node.
        layout_id: LayoutId,
        /// Its global id, when it has an `id()`.
        global_id: Option<GlobalElementId>,
        /// The element's state.
        request_layout: RequestLayoutState,
    },
    /// Layout computed but not prepainted (`layout_as_root`).
    LayoutComputed {
        /// Its layout node.
        layout_id: LayoutId,
        /// Its global id.
        global_id: Option<GlobalElementId>,
        /// The available space it was laid out in.
        available_space: Size<AvailableSpace>,
        /// The element's state.
        request_layout: RequestLayoutState,
    },
    /// `prepaint` ran.
    Prepaint {
        /// Its global id.
        global_id: Option<GlobalElementId>,
        /// Its bounds.
        bounds: Bounds<Pixels>,
        /// The element's states.
        request_layout: RequestLayoutState,
        /// The prepaint state.
        prepaint: PrepaintState,
    },
    /// Painted.
    Painted,
}

/// An element plus its phase: the concrete `ElementObject`.
pub(crate) struct Drawable<E: Element> {
    /// The element.
    pub(crate) element: E,
    /// Its phase and phase states.
    pub(crate) phase: ElementDrawPhase<E::RequestLayoutState, E::PrepaintState>,
}

impl<E: Element> ElementObject for Drawable<E> {
    fn inner_element(&mut self) -> &mut dyn Any {
        &mut self.element
    }

    fn request_layout(&mut self, _window: &mut Window, _cx: &mut App) -> LayoutId {
        todo!("elements")
    }

    fn prepaint(&mut self, _window: &mut Window, _cx: &mut App) {
        todo!("elements")
    }

    fn paint(&mut self, _window: &mut Window, _cx: &mut App) {
        todo!("elements")
    }

    fn layout_as_root(
        &mut self,
        _available_space: Size<AvailableSpace>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Size<Pixels> {
        todo!("elements")
    }
}

/// A type-erased element. Boxed once when built; the frame drops it after paint.
pub struct AnyElement(Box<dyn ElementObject>);

impl AnyElement {
    /// Erases `element`.
    pub fn new<E: Element>(element: E) -> Self {
        AnyElement(Box::new(Drawable { element, phase: ElementDrawPhase::Start }))
    }

    /// The element as `T`, if it is one.
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.0.inner_element().downcast_mut::<T>()
    }

    /// Phase 1: requests the layout of the element and its children.
    pub fn request_layout(&mut self, window: &mut Window, cx: &mut App) -> LayoutId {
        self.0.request_layout(window, cx)
    }

    /// Phase 2. Returns the focus handle a child asked to be focused during prepaint, as GPUI does.
    pub fn prepaint(&mut self, _window: &mut Window, _cx: &mut App) -> Option<FocusHandle> {
        todo!("elements")
    }

    /// Phase 3.
    pub fn paint(&mut self, window: &mut Window, cx: &mut App) {
        self.0.paint(window, cx)
    }

    /// Lays the element out as its own root in `available_space` and returns its size.
    pub fn layout_as_root(
        &mut self,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        self.0.layout_as_root(available_space, window, cx)
    }

    /// Prepaints at `origin` (after a layout), offset from the current element offset.
    pub fn prepaint_at(
        &mut self,
        _origin: Point<Pixels>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<FocusHandle> {
        todo!("elements")
    }

    /// `layout_as_root` then `prepaint_at(origin)`.
    pub fn prepaint_as_root(
        &mut self,
        _origin: Point<Pixels>,
        _available_space: Size<AvailableSpace>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Option<FocusHandle> {
        todo!("elements")
    }
}

impl Element for AnyElement {
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
    ) -> (LayoutId, Self::RequestLayoutState) {
        (self.request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.paint(window, cx);
    }
}

impl IntoElement for AnyElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }

    fn into_any_element(self) -> AnyElement {
        self
    }
}

/// An element that draws nothing and takes no space.
#[derive(Clone, Copy, Debug, Default)]
pub struct Empty;

impl Element for Empty {
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
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        todo!("elements")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

impl IntoElement for Empty {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl<T: IntoElement> IntoElement for Option<T> {
    type Element = Option<T::Element>;

    fn into_element(self) -> Self::Element {
        self.map(IntoElement::into_element)
    }
}

impl<E: Element> Element for Option<E> {
    type RequestLayoutState = Option<E::RequestLayoutState>;
    type PrepaintState = Option<E::PrepaintState>;

    fn id(&self) -> Option<ElementId> {
        self.as_ref().and_then(Element::id)
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        todo!("elements")
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        todo!("elements")
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        todo!("elements")
    }
}

/// A layout node in the window's taffy tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutId(pub(crate) taffy::NodeId);

/// An element's identity among its siblings; a path of these keys element state across frames.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementId {
    /// A view (the entity that renders it).
    View(EntityId),
    /// An integer.
    Integer(u64),
    /// A name.
    Name(SharedString),
    /// The element tracks this focus handle and has no other id.
    FocusHandle(FocusId),
    /// A name and an integer (`("row", i)`).
    NamedInteger(SharedString, u64),
}

impl fmt::Display for ElementId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElementId::View(id) => write!(f, "view-{id:?}"),
            ElementId::Integer(i) => write!(f, "{i}"),
            ElementId::Name(name) => write!(f, "{name}"),
            ElementId::FocusHandle(id) => write!(f, "focus-{id:?}"),
            ElementId::NamedInteger(name, i) => write!(f, "{name}-{i}"),
        }
    }
}

impl From<usize> for ElementId {
    fn from(id: usize) -> Self {
        ElementId::Integer(id as u64)
    }
}

impl From<SharedString> for ElementId {
    fn from(name: SharedString) -> Self {
        ElementId::Name(name)
    }
}

impl From<&'static str> for ElementId {
    fn from(name: &'static str) -> Self {
        ElementId::Name(name.into())
    }
}

impl From<(&'static str, usize)> for ElementId {
    fn from((name, id): (&'static str, usize)) -> Self {
        ElementId::NamedInteger(name.into(), id as u64)
    }
}

impl From<(&'static str, u64)> for ElementId {
    fn from((name, id): (&'static str, u64)) -> Self {
        ElementId::NamedInteger(name.into(), id)
    }
}

impl From<(SharedString, usize)> for ElementId {
    fn from((name, id): (SharedString, usize)) -> Self {
        ElementId::NamedInteger(name, id as u64)
    }
}

/// The path of element ids from the root view to an element: the key of its state across frames. One hash
/// of a short inline path per lookup.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GlobalElementId(pub(crate) Arc<[ElementId]>);

/// GPUI's inspector id. We have no inspector: a unit type the `Element` trait still passes (as `None`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InspectorElementId;

/// The id path the window builds while walking the tree (pushed by `with_global_id`).
pub(crate) type ElementIdStack = SmallVec<[ElementId; 32]>;

/// An `Entity<V: Render>` used as a child element renders the view in place.
impl<V: Render> IntoElement for Entity<V> {
    type Element = crate::AnyView;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

/// Conditional builder steps (`.when(..)`, `.when_some(..)`) for every element.
impl<T: IntoElement> FluentBuilder for T {}
