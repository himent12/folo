//! Interactivity: the listeners, styles and state an interactive element (`Div`, `Stateful<Div>`, `Svg`)
//! carries, and the drag / tooltip types. Owner tag: `"window"`.
//!
//! Files: `mod.rs` (the `Interactivity` struct, listener types, `DragMoveEvent`, `AnyDrag`, `AnyTooltip`),
//! `traits.rs` (`InteractiveElement`, `StatefulInteractiveElement`: the builder methods the app calls),
//! `phases.rs` (`Interactivity::request_layout` / `prepaint` / `paint`: hitboxes, listeners, hover /
//! active / focus / group styles, click detection, scrolling, tooltips), `scroll.rs` (`ScrollHandle`).
//!
//! Per-element state that must survive frames (hover, pressed, click start, scroll offset, tooltip timer)
//! is `InteractiveElementState`, kept by `GlobalElementId` through `Window::with_element_state`.
#![allow(dead_code)] // skeleton

mod phases;
mod scroll;
mod traits;

pub use scroll::*;
pub use traits::*;

use std::{
    any::{Any, TypeId},
    cell::{Cell, RefCell},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use smallvec::SmallVec;

use crate::{
    AnyView, App, Bounds, ClickEvent, CursorStyle, DispatchPhase, ElementId, FocusHandle, KeyContext,
    KeyDownEvent, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, ScrollWheelEvent,
    SharedString, StyleRefinement, Window, WindowControlArea,
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
/// A drag-move listener (the `DragMoveEvent<T>` erased).
pub(crate) type DragMoveListener =
    Box<dyn Fn(&MouseMoveEvent, &Bounds<Pixels>, &AnyDrag, &mut Window, &mut App)>;
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
    /// Where the cursor grabbed it, relative to the drag view (unused for file drags).
    pub(crate) cursor_offset: Point<Pixels>,
}

/// A mouse move while something of type `T` is dragged over an element (`on_drag_move`).
pub struct DragMoveEvent<T> {
    /// The mouse move.
    pub event: MouseMoveEvent,
    /// The element's bounds.
    pub bounds: Bounds<Pixels>,
    drag: PhantomData<T>,
    dragged_item: Arc<dyn Any>,
}

impl<T: 'static> DragMoveEvent<T> {
    /// The dragged value.
    pub fn drag<'b>(&self, _cx: &'b App) -> &'b T {
        todo!("window")
    }

    /// The dragged value without the app (the event owns a reference to it).
    pub fn dragged_item(&self) -> &dyn Any {
        self.dragged_item.as_ref()
    }
}

/// A tooltip request: the view builder and the hitbox it belongs to.
#[derive(Clone)]
pub struct AnyTooltip {
    /// The tooltip view.
    pub(crate) view: AnyView,
    /// Where the mouse was when it was asked for.
    pub(crate) mouse_position: Point<Pixels>,
    /// Whether the tooltip is still wanted, given its bounds (it hides when the mouse leaves its element).
    pub(crate) check_visible_and_update: TooltipVisibility,
}

/// State an interactive element keeps across frames.
#[derive(Default)]
pub(crate) struct InteractiveElementState {
    /// The focus handle it made for itself (`focusable` without a handle).
    pub(crate) focus_handle: Option<FocusHandle>,
    /// Whether the mouse is down on it (active style, click detection).
    pub(crate) pressed: Option<Rc<Cell<bool>>>,
    /// Whether it is hovered (hover listeners fire on change).
    pub(crate) hover_state: Option<Rc<Cell<bool>>>,
    /// The mouse-down that may become a click.
    pub(crate) pending_mouse_down: Option<Rc<RefCell<Option<MouseDownEvent>>>>,
    /// Scroll offset of a scroll container without a `ScrollHandle`.
    pub(crate) scroll_offset: Option<Rc<RefCell<Point<Pixels>>>>,
    /// The tooltip timer / shown tooltip.
    pub(crate) active_tooltip: Option<Rc<RefCell<Option<crate::Task<()>>>>>,
}

/// Everything an interactive element carries. Built by the builder methods, consumed by its phases.
#[derive(Default)]
pub struct Interactivity {
    /// The element id (`Stateful`), if any.
    pub(crate) element_id: Option<ElementId>,
    /// The key context it adds to the dispatch tree.
    pub(crate) key_context: Option<KeyContext>,
    /// The focus handle it tracks (`track_focus`).
    pub(crate) tracked_focus_handle: Option<FocusHandle>,
    /// The scroll handle it reports to (`track_scroll`).
    pub(crate) tracked_scroll_handle: Option<ScrollHandle>,
    /// The base style.
    pub(crate) base_style: Box<StyleRefinement>,
    /// Applied while focused.
    pub(crate) focus_style: Option<Box<StyleRefinement>>,
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
    /// Drag-move listeners by dragged type.
    pub(crate) drag_move_listeners: SmallVec<[(TypeId, DragMoveListener); 1]>,
    /// Click listeners.
    pub(crate) click_listeners: Vec<ClickListener>,
    /// The hover listener.
    pub(crate) hover_listener: Option<HoverListener>,
    /// The tooltip builder.
    pub(crate) tooltip_builder: Option<TooltipBuilder>,
    /// The window control area it is (title bar caption and buttons).
    pub(crate) window_control: Option<WindowControlArea>,
    /// The hitbox blocks mouse events below it (`occlude`).
    pub(crate) occlude_mouse: bool,
    /// Cursor style (from the style refinement's `mouse_cursor`, resolved at paint).
    pub(crate) cursor_style: Option<CursorStyle>,
    /// Can take focus by tab (`tab_index` on an element).
    pub(crate) tab_index: Option<isize>,
}
