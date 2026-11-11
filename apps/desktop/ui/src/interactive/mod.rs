//! Interactivity: the listeners, styles and state an interactive element (`Div`, `Stateful<Div>`) carries,
//! and the drag / tooltip types. Owner tag: `"window"`. Behaviour is GPUI 0.2.2's `elements/div.rs`.
//!
//! Files: `mod.rs` (the `Interactivity` struct, listener types, `DragMoveEvent`, `AnyDrag`, `AnyTooltip`),
//! `traits.rs` (`InteractiveElement`, `StatefulInteractiveElement`: the builder methods the app calls),
//! `phases.rs` (`Interactivity::request_layout` / `prepaint` / `paint` and the style computation),
//! `listeners.rs` (the mouse, click, hover, scroll and keyboard listeners registered at paint),
//! `tooltip.rs` (the tooltip state machine), `scroll.rs` (`ScrollHandle`).
//!
//! Per-element state that must survive frames (pressed, click start, hover, scroll offset, tooltip) is
//! `InteractiveElementState`, kept by `GlobalElementId` through `Window::with_element_state`.

mod listeners;
mod phases;
mod scroll;
mod tooltip;
mod traits;

pub(crate) use phases::{GroupHitboxes, skips_children};
pub use scroll::*;
pub(crate) use tooltip::ActiveTooltip;
pub use traits::*;

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use crate::{
    AnyView, App, Bounds, ClickEvent, DispatchPhase, ElementId, FocusHandle, HitboxBehavior, KeyContext,
    KeyDownEvent, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, ScrollWheelEvent,
    SharedString, Size, StyleRefinement, TooltipId, Window, WindowControlArea, window::DispatchNodeId,
};

/// A mouse-down listener.
pub(crate) type MouseDownListener =
    Box<dyn Fn(&MouseDownEvent, DispatchPhase, &crate::Hitbox, &mut Window, &mut App)>;
/// A mouse-up listener.
pub(crate) type MouseUpListener =
    Box<dyn Fn(&MouseUpEvent, DispatchPhase, &crate::Hitbox, &mut Window, &mut App)>;
/// A mouse-move listener.
pub(crate) type MouseMoveListener =
    Box<dyn Fn(&MouseMoveEvent, DispatchPhase, &crate::Hitbox, &mut Window, &mut App)>;
/// A scroll-wheel listener.
pub(crate) type ScrollWheelListener =
    Box<dyn Fn(&ScrollWheelEvent, DispatchPhase, &crate::Hitbox, &mut Window, &mut App)>;
/// A click listener.
pub(crate) type ClickListener = Rc<dyn Fn(&ClickEvent, &mut Window, &mut App)>;
/// A hover listener (`true` on enter, `false` on leave).
pub(crate) type HoverListener = Box<dyn Fn(&bool, &mut Window, &mut App)>;
/// A key-down listener.
pub(crate) type KeyDownListener = Box<dyn Fn(&KeyDownEvent, DispatchPhase, &mut Window, &mut App)>;
/// An action listener (the action as `Any`).
pub(crate) type ActionListener = Box<dyn Fn(&dyn Any, DispatchPhase, &mut Window, &mut App)>;
/// A drop listener (the dragged value as `Any`).
pub(crate) type DropListener = Box<dyn Fn(&dyn Any, &mut Window, &mut App)>;
/// Whether a shown tooltip is still wanted, given its bounds.
pub(crate) type TooltipVisibility = Rc<dyn Fn(Bounds<Pixels>, &mut Window, &mut App) -> bool>;
/// Builds a tooltip view.
pub(crate) type TooltipBuilder = Rc<dyn Fn(&mut Window, &mut App) -> AnyView>;

/// A style applied while a named group (an ancestor with `.group(name)`) is hovered or active.
pub(crate) struct GroupStyle {
    /// The group name.
    pub(crate) group: SharedString,
    /// The style refinement.
    pub(crate) style: Box<StyleRefinement>,
}

/// The drag in progress: what is dragged (`ExternalPaths` for a file drag) and where it started.
pub struct AnyDrag {
    /// The dragged value.
    pub(crate) value: Arc<dyn Any>,
    /// Its type.
    pub(crate) value_type: TypeId,
}

/// A mouse move while something of type `T` is dragged over the window (`on_drag_move`).
pub struct DragMoveEvent<T> {
    /// The mouse move.
    pub event: MouseMoveEvent,
    /// The element's bounds.
    pub bounds: Bounds<Pixels>,
    drag: PhantomData<T>,
    dragged_item: Arc<dyn Any>,
}

impl<T: 'static> DragMoveEvent<T> {
    /// The event for a move over an element with `bounds` while `dragged_item` is dragged.
    pub(crate) fn new(event: MouseMoveEvent, bounds: Bounds<Pixels>, dragged_item: Arc<dyn Any>) -> Self {
        DragMoveEvent { event, bounds, drag: PhantomData, dragged_item }
    }

    /// The dragged value (the event holds it, so the app is not needed; the signature is GPUI's).
    pub fn drag<'b>(&'b self, _cx: &'b App) -> &'b T {
        self.dragged_item.downcast_ref::<T>().expect("a drag-move listener only sees drags of its type")
    }

    /// The dragged value without the app.
    pub fn dragged_item(&self) -> &dyn Any {
        self.dragged_item.as_ref()
    }
}

/// A tooltip request: the view and how to tell whether it is still wanted.
#[derive(Clone)]
pub struct AnyTooltip {
    /// The tooltip view.
    pub(crate) view: AnyView,
    /// Where the mouse was when it appeared.
    pub(crate) mouse_position: Point<Pixels>,
    /// Whether the tooltip is still wanted, given its bounds (it hides when the mouse leaves its element).
    pub(crate) check_visible_and_update: TooltipVisibility,
}

/// Whether the element (or its group) is pressed (GPUI's `ElementClickedState`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ElementClickedState {
    /// The element's group was pressed.
    pub(crate) group: bool,
    /// The element itself was pressed.
    pub(crate) element: bool,
}

impl ElementClickedState {
    /// Either.
    pub(crate) fn is_clicked(&self) -> bool {
        self.group || self.element
    }
}

/// State an interactive element keeps across frames (GPUI's `InteractiveElementState`).
#[derive(Default)]
pub(crate) struct InteractiveElementState {
    /// The focus handle it made for itself (`focusable` without a handle).
    pub(crate) focus_handle: Option<FocusHandle>,
    /// Whether the mouse went down on it or its group (active styles).
    pub(crate) clicked_state: Option<Rc<RefCell<ElementClickedState>>>,
    /// Whether it is hovered (hover listeners fire on change).
    pub(crate) hover_state: Option<Rc<RefCell<bool>>>,
    /// The mouse-down that may become a click.
    pub(crate) pending_mouse_down: Option<Rc<RefCell<Option<MouseDownEvent>>>>,
    /// Scroll offset of a scroll container without a `ScrollHandle`.
    pub(crate) scroll_offset: Option<Rc<RefCell<Point<Pixels>>>>,
    /// The tooltip timer / shown tooltip.
    pub(crate) active_tooltip: Option<Rc<RefCell<Option<ActiveTooltip>>>>,
}

/// Everything an interactive element carries. Built by the builder methods, consumed by its phases.
#[derive(Default)]
pub struct Interactivity {
    /// The element id (`Stateful`), if any.
    pub(crate) element_id: Option<ElementId>,
    /// Pressed this frame (set at prepaint from the element state).
    pub(crate) active: Option<bool>,
    /// Hovered this frame (set at paint).
    pub(crate) hovered: Option<bool>,
    /// The tooltip it shows this frame, if one is visible.
    pub(crate) tooltip_id: Option<TooltipId>,
    /// The size of its children's bounding box (scroll containers).
    pub(crate) content_size: Size<Pixels>,
    /// The key context it adds to the dispatch tree.
    pub(crate) key_context: Option<KeyContext>,
    /// It takes focus (a tracked handle, or one kept in its element state).
    pub(crate) focusable: bool,
    /// The focus handle it tracks (`track_focus`).
    pub(crate) tracked_focus_handle: Option<FocusHandle>,
    /// The scroll handle it reports to (`track_scroll`).
    pub(crate) tracked_scroll_handle: Option<ScrollHandle>,
    /// The scroll offset cell (from the scroll handle or the element state) of a scroll container.
    pub(crate) scroll_offset: Option<Rc<RefCell<Point<Pixels>>>>,
    /// The base style.
    pub(crate) base_style: Box<StyleRefinement>,
    /// Applied while hovered.
    pub(crate) hover_style: Option<Box<StyleRefinement>>,
    /// Applied while a group is hovered.
    pub(crate) group_hover_style: Option<GroupStyle>,
    /// Applied while pressed.
    pub(crate) active_style: Option<Box<StyleRefinement>>,
    /// Applied while a group is pressed.
    pub(crate) group_active_style: Option<GroupStyle>,
    /// The group name it defines (`.group(name)`).
    pub(crate) group: Option<SharedString>,
    /// Mouse-down listeners.
    pub(crate) mouse_down_listeners: Vec<MouseDownListener>,
    /// Mouse-up listeners.
    pub(crate) mouse_up_listeners: Vec<MouseUpListener>,
    /// Mouse-move listeners.
    pub(crate) mouse_move_listeners: Vec<MouseMoveListener>,
    /// Scroll-wheel listeners.
    pub(crate) scroll_wheel_listeners: Vec<ScrollWheelListener>,
    /// Key-down listeners (capture and bubble).
    pub(crate) key_down_listeners: Vec<KeyDownListener>,
    /// Action listeners by action type.
    pub(crate) action_listeners: Vec<(TypeId, ActionListener)>,
    /// Drop listeners by dragged type.
    pub(crate) drop_listeners: Vec<(TypeId, DropListener)>,
    /// Click listeners.
    pub(crate) click_listeners: Vec<ClickListener>,
    /// The hover listener.
    pub(crate) hover_listener: Option<HoverListener>,
    /// The tooltip builder.
    pub(crate) tooltip_builder: Option<TooltipBuilder>,
    /// The window control area it is (title bar caption and buttons).
    pub(crate) window_control: Option<WindowControlArea>,
    /// How its hitbox affects the ones under it (`occlude`).
    pub(crate) hitbox_behavior: HitboxBehavior,
    /// Its position in Tab order, when set on the element.
    pub(crate) tab_index: Option<isize>,
    /// Whether the handle it makes for itself is a tab stop.
    pub(crate) tab_stop: bool,
    /// Its dispatch node this frame (pushed at prepaint, made current again at paint).
    pub(crate) dispatch_node: Option<DispatchNodeId>,
}
