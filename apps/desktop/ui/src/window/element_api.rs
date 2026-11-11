//! The element-author API on `Window` besides painting: content masks, text style, opacity, element ids and
//! element state, hitboxes, mouse / key / action listeners, the input handler, cursor styles, tooltips,
//! deferred draws, element offsets. Owner tag: `"window"`.
//!
//! Performance: element state is looked up by one hash of `(GlobalElementId, TypeId)`; touching it moves it
//! from the rendered frame's map to the next frame's (no clone). Listeners are boxed once per registration
//! into lists that keep capacity across frames.
#![allow(dead_code)] // skeleton

use std::any::{Any, TypeId};

use crate::{
    AnyElement, AnyTooltip, App, Bounds, ContentMask, CursorStyle, DispatchPhase, ElementId, FocusHandle,
    GlobalElementId, Hitbox, HitboxBehavior, InputHandler, KeyEvent, MouseEvent, Pixels, Point, TextStyle,
    TextStyleRefinement, TooltipId, Window, WindowControlArea,
};

impl Window {
    /// Runs `f` with `mask` intersected into the content mask (`None` leaves it as is).
    pub fn with_content_mask<R>(&mut self, _mask: Option<ContentMask<Pixels>>, _f: impl FnOnce(&mut Self) -> R) -> R {
        todo!("window")
    }

    /// The current content mask (the whole viewport at the root).
    pub fn content_mask(&self) -> ContentMask<Pixels> {
        todo!("window")
    }

    /// Runs `f` with `style` refining the text style (`None` leaves it as is).
    pub fn with_text_style<F, R>(&mut self, _style: Option<TextStyleRefinement>, _f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        todo!("window")
    }

    /// The text style in effect (the default refined by every pushed refinement).
    pub fn text_style(&self) -> TextStyle {
        todo!("window")
    }

    /// Runs `f` with colours painted at `opacity` times the current element opacity.
    pub fn with_element_opacity<R>(&mut self, _opacity: Option<f32>, _f: impl FnOnce(&mut Self) -> R) -> R {
        todo!("window")
    }

    /// Runs `f` with `offset` added to the element offset (scroll containers).
    pub fn with_element_offset<R>(&mut self, _offset: Point<Pixels>, _f: impl FnOnce(&mut Self) -> R) -> R {
        todo!("window")
    }

    /// Runs `f` with the element offset set to `offset`.
    pub fn with_absolute_element_offset<R>(&mut self, _offset: Point<Pixels>, _f: impl FnOnce(&mut Self) -> R) -> R {
        todo!("window")
    }

    /// Runs `f` with `element_id` pushed on the id path, giving it the element's global id.
    pub fn with_global_id<R>(&mut self, _element_id: ElementId, _f: impl FnOnce(&GlobalElementId, &mut Self) -> R) -> R {
        todo!("window")
    }

    /// Runs `f` with `id` pushed on the id path.
    pub fn with_id<R>(&mut self, id: impl Into<ElementId>, f: impl FnOnce(&mut Self) -> R) -> R {
        self.with_global_id(id.into(), |_, window| f(window))
    }

    /// Runs `f` with the element's state of type `S` from the previous frame (`None` the first time) and
    /// keeps the state it returns for the next frame.
    pub fn with_element_state<S, R>(&mut self, _global_id: &GlobalElementId, _f: impl FnOnce(Option<S>, &mut Self) -> (R, S)) -> R
    where
        S: 'static,
    {
        todo!("window")
    }

    /// `with_element_state` for an element that may have no id (`f` gets `None` then and keeps nothing).
    pub fn with_optional_element_state<S, R>(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _f: impl FnOnce(Option<Option<S>>, &mut Self) -> (R, Option<S>),
    ) -> R
    where
        S: 'static,
    {
        todo!("window")
    }

    /// Adds a hitbox (clipped by the content mask) to this frame.
    pub fn insert_hitbox(&mut self, _bounds: Bounds<Pixels>, _behavior: HitboxBehavior) -> Hitbox {
        todo!("window")
    }

    /// Adds a window control area (caption, minimize, maximize, close) for the platform's hit test.
    pub fn insert_window_control_hitbox(&mut self, _area: WindowControlArea, _hitbox: Hitbox) {
        todo!("window")
    }

    /// Sets the cursor over `hitbox`.
    pub fn set_cursor_style(&mut self, _style: CursorStyle, _hitbox: &Hitbox) {
        todo!("window")
    }

    /// Sets the cursor for the whole window this frame.
    pub fn set_window_cursor_style(&mut self, _style: CursorStyle) {
        todo!("window")
    }

    /// Asks for a tooltip this frame.
    pub fn set_tooltip(&mut self, _tooltip: AnyTooltip) -> TooltipId {
        todo!("window")
    }

    /// Adds a mouse listener of event type `Event` for this frame.
    pub fn on_mouse_event<Event: MouseEvent>(
        &mut self,
        _handler: impl FnMut(&Event, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        todo!("window")
    }

    /// Adds a key listener of event type `Event` to the current dispatch node.
    pub fn on_key_event<Event: KeyEvent>(&mut self, _listener: impl Fn(&Event, DispatchPhase, &mut Window, &mut App) + 'static) {
        todo!("window")
    }

    /// Adds an action listener to the current dispatch node.
    pub fn on_action(&mut self, _action_type: TypeId, _listener: impl Fn(&dyn Any, DispatchPhase, &mut Window, &mut App) + 'static) {
        todo!("window")
    }

    /// Records `input_handler` as the text input handler of the element with `focus_handle`; the focused
    /// element's handler is given to the platform after the frame.
    pub fn handle_input(&mut self, _focus_handle: &FocusHandle, _input_handler: impl InputHandler, _cx: &App) {
        todo!("window")
    }

    /// Draws `element` after the main tree, at `absolute_offset`, above lower priorities.
    pub fn defer_draw(&mut self, _element: AnyElement, _absolute_offset: Point<Pixels>, _priority: usize) {
        todo!("window")
    }

    /// Starts a dispatch node for the element being painted (Interactivity calls it; ended by
    /// `pop_dispatch_node`).
    pub(crate) fn push_dispatch_node(&mut self) {
        todo!("window")
    }

    /// Ends the current dispatch node.
    pub(crate) fn pop_dispatch_node(&mut self) {
        todo!("window")
    }

    /// Sets the current dispatch node's key context.
    pub fn set_key_context(&mut self, _context: crate::KeyContext) {
        todo!("window")
    }

    /// Makes the current dispatch node track `focus_handle` (focus path, tab stop).
    pub fn set_focus_handle(&mut self, _focus_handle: &FocusHandle, _cx: &App) {
        todo!("window")
    }
}
