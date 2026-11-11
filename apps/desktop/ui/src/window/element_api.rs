//! The element-author API on `Window` besides painting: content masks, text style, opacity, element ids and
//! element state, hitboxes, mouse / key / action listeners, the input handler, cursor styles, tooltips,
//! deferred draws, element offsets, dispatch nodes. Owner tag: `"window"`. Semantics are GPUI 0.2.2's
//! `window.rs` (each stack pushes for the call and pops after; `None` leaves the state as it is).
//!
//! Performance: element state is looked up by one hash of `(GlobalElementId, TypeId)`; touching it moves it
//! from the rendered frame's map to the next frame's (no clone, and its box is reused). Listeners are
//! boxed once per registration into lists that keep capacity across frames.

use std::{
    any::{Any, TypeId},
    rc::Rc,
};

use crate::{
    AnyElement, AnyTooltip, App, Bounds, ContentMask, CursorStyle, CursorStyleRequest, DeferredDraw,
    DispatchPhase, ElementId, FocusHandle, GlobalElementId, Hitbox, HitboxBehavior, HitboxId, InputHandler,
    KeyEvent, MouseEvent, Pixels, Point, TextStyle, TextStyleRefinement, TooltipId, TooltipRequest, Window,
    WindowControlArea, WindowControlHitbox,
    style::Refineable,
    window::{DispatchNodeId, input_handler::InputHandlerSlot},
};

impl Window {
    /// Runs `f` with `mask` intersected into the content mask (`None` leaves it as is).
    pub fn with_content_mask<R>(
        &mut self,
        mask: Option<ContentMask<Pixels>>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(mask) = mask else { return f(self) };
        let mask = mask.intersect(&self.content_mask());
        self.content_mask_stack.push(mask);
        let result = f(self);
        self.content_mask_stack.pop();
        result
    }

    /// The current content mask (the whole viewport at the root).
    pub fn content_mask(&self) -> ContentMask<Pixels> {
        self.content_mask_stack
            .last()
            .cloned()
            .unwrap_or_else(|| ContentMask { bounds: Bounds { origin: Point::default(), size: self.viewport_size } })
    }

    /// Runs `f` with `style` refining the text style (`None` leaves it as is).
    pub fn with_text_style<F, R>(&mut self, style: Option<TextStyleRefinement>, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let Some(style) = style else { return f(self) };
        self.text_style_stack.push(style);
        let result = f(self);
        self.text_style_stack.pop();
        result
    }

    /// The text style in effect (the default refined by every pushed refinement).
    pub fn text_style(&self) -> TextStyle {
        let mut style = TextStyle::default();
        for refinement in &self.text_style_stack {
            style.refine(refinement);
        }
        style
    }

    /// Runs `f` with colours painted at `opacity` times the current element opacity.
    pub fn with_element_opacity<R>(&mut self, opacity: Option<f32>, f: impl FnOnce(&mut Self) -> R) -> R {
        let Some(opacity) = opacity else { return f(self) };
        let previous = self.element_opacity;
        self.element_opacity = previous * opacity;
        let result = f(self);
        self.element_opacity = previous;
        result
    }

    /// The opacity colours are painted at now (the product of every `with_element_opacity`).
    pub fn element_opacity(&self) -> f32 {
        self.element_opacity
    }

    /// Runs `f` with `offset` added to the element offset (scroll containers).
    pub fn with_element_offset<R>(&mut self, offset: Point<Pixels>, f: impl FnOnce(&mut Self) -> R) -> R {
        if offset.x == Pixels::ZERO && offset.y == Pixels::ZERO {
            return f(self);
        }
        let absolute = self.element_offset() + offset;
        self.with_absolute_element_offset(absolute, f)
    }

    /// Runs `f` with the element offset set to `offset`.
    pub fn with_absolute_element_offset<R>(
        &mut self,
        offset: Point<Pixels>,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.element_offset_stack.push(offset);
        let result = f(self);
        self.element_offset_stack.pop();
        result
    }

    /// Runs `f` with `element_id` pushed on the id path, giving it the element's global id.
    pub fn with_global_id<R>(
        &mut self,
        element_id: ElementId,
        f: impl FnOnce(&GlobalElementId, &mut Self) -> R,
    ) -> R {
        self.element_id_stack.push(element_id);
        let global_id = GlobalElementId(self.element_id_stack.as_slice().into());
        let result = f(&global_id, self);
        self.element_id_stack.pop();
        result
    }

    /// Runs `f` with `id` pushed on the id path.
    pub fn with_id<R>(&mut self, id: impl Into<ElementId>, f: impl FnOnce(&mut Self) -> R) -> R {
        self.with_global_id(id.into(), |_, window| f(window))
    }

    /// Runs `f` with the element's state of type `S` from the previous frame (`None` the first time) and
    /// keeps the state it returns for the next frame.
    pub fn with_element_state<S, R>(
        &mut self,
        global_id: &GlobalElementId,
        f: impl FnOnce(Option<S>, &mut Self) -> (R, S),
    ) -> R
    where
        S: 'static,
    {
        let key = (global_id.clone(), TypeId::of::<S>());
        let existing =
            self.next_frame.element_states.remove(&key).or_else(|| self.rendered_frame.element_states.remove(&key));
        if let Some(mut boxed) = existing {
            // The box holds an `Option<S>` so it can be reused instead of reallocated.
            let slot = boxed.downcast_mut::<Option<S>>().expect("element state of the requested type");
            let state = slot.take().expect("re-entrant with_element_state for one element and type");
            let (result, state) = f(Some(state), self);
            *boxed.downcast_mut::<Option<S>>().expect("checked above") = Some(state);
            self.next_frame.element_states.insert(key, boxed);
            result
        } else {
            let (result, state) = f(None, self);
            self.next_frame.element_states.insert(key, Box::new(Some(state)));
            result
        }
    }

    /// `with_element_state` for an element that may have no id (`f` gets `None` then and keeps nothing).
    pub fn with_optional_element_state<S, R>(
        &mut self,
        global_id: Option<&GlobalElementId>,
        f: impl FnOnce(Option<Option<S>>, &mut Self) -> (R, Option<S>),
    ) -> R
    where
        S: 'static,
    {
        if let Some(global_id) = global_id {
            self.with_element_state(global_id, |state, window| {
                let (result, state) = f(Some(state), window);
                (result, state.expect("an element with an id returns its state"))
            })
        } else {
            f(None, self).0
        }
    }

    /// Adds a hitbox (clipped by the content mask) to this frame. Ids grow across frames, as GPUI's, so a
    /// hitbox of the frame being built is never mistaken for one of the frame on screen.
    pub fn insert_hitbox(&mut self, bounds: Bounds<Pixels>, behavior: HitboxBehavior) -> Hitbox {
        let id = HitboxId(self.next_hitbox_id);
        self.next_hitbox_id = self.next_hitbox_id.wrapping_add(1);
        let hitbox = Hitbox { id, bounds, content_mask: self.content_mask(), behavior };
        self.next_frame.hitboxes.push(hitbox.clone());
        hitbox
    }

    /// Adds a window control area (caption, minimize, maximize, close) for the platform's hit test.
    pub fn insert_window_control_hitbox(&mut self, area: WindowControlArea, hitbox: Hitbox) {
        self.next_frame.window_control_hitboxes.push(WindowControlHitbox { area, hitbox });
    }

    /// Sets the cursor over `hitbox`.
    pub fn set_cursor_style(&mut self, style: CursorStyle, hitbox: &Hitbox) {
        self.next_frame.cursor_styles.push(CursorStyleRequest { hitbox_id: Some(hitbox.id), style });
    }

    /// Sets the cursor for the whole window this frame.
    pub fn set_window_cursor_style(&mut self, style: CursorStyle) {
        self.next_frame.cursor_styles.push(CursorStyleRequest { hitbox_id: None, style });
    }

    /// Asks for a tooltip this frame.
    pub fn set_tooltip(&mut self, tooltip: AnyTooltip) -> TooltipId {
        let id = TooltipId(self.next_tooltip_id);
        self.next_tooltip_id = self.next_tooltip_id.wrapping_add(1);
        self.next_frame.tooltip_requests.push(Some(TooltipRequest { id, tooltip }));
        id
    }

    /// Adds a mouse listener of event type `Event` for this frame.
    pub fn on_mouse_event<Event: MouseEvent>(
        &mut self,
        mut handler: impl FnMut(&Event, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.next_frame.mouse_listeners.push(Some(Box::new(
            move |event: &dyn Any, phase: DispatchPhase, window: &mut Window, cx: &mut App| {
                if let Some(event) = event.downcast_ref::<Event>() {
                    handler(event, phase, window, cx)
                }
            },
        )));
    }

    /// Adds a key listener of event type `Event` to the current dispatch node.
    pub fn on_key_event<Event: KeyEvent>(
        &mut self,
        listener: impl Fn(&Event, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.next_frame.dispatch_tree.on_key_event(Rc::new(
            move |event: &dyn Any, phase, window: &mut Window, cx: &mut App| {
                if let Some(event) = event.downcast_ref::<Event>() {
                    listener(event, phase, window, cx)
                }
            },
        ));
    }

    /// Adds an action listener to the current dispatch node.
    pub fn on_action(
        &mut self,
        action_type: TypeId,
        listener: impl Fn(&dyn Any, DispatchPhase, &mut Window, &mut App) + 'static,
    ) {
        self.next_frame.dispatch_tree.on_action(action_type, Rc::new(listener));
    }

    /// Records `input_handler` as the text input handler of the element with `focus_handle` if that
    /// element is focused; it is given to the platform after the frame.
    pub fn handle_input(&mut self, focus_handle: &FocusHandle, input_handler: impl InputHandler, _cx: &App) {
        if focus_handle.is_focused(self) {
            self.next_frame
                .input_handlers
                .push(Some(InputHandlerSlot { focus_id: focus_handle.id, handler: Box::new(input_handler) }));
        }
    }

    /// Draws `element` after the main tree, at `absolute_offset`, above lower priorities.
    pub fn defer_draw(&mut self, element: AnyElement, absolute_offset: Point<Pixels>, priority: usize) {
        let parent_node = self.next_frame.dispatch_tree.active_node_id().unwrap_or_default();
        self.next_frame.deferred_draws.push(DeferredDraw {
            priority,
            parent_node,
            element_id_stack: self.element_id_stack.clone(),
            text_style_stack: self.text_style_stack.clone(),
            element: Some(element),
            absolute_offset,
        });
    }

    /// Starts a dispatch node for the element being prepainted (Interactivity calls it; ended by
    /// `pop_dispatch_node`, made current again at paint by `set_active_dispatch_node`).
    pub(crate) fn push_dispatch_node(&mut self) -> DispatchNodeId {
        self.next_frame.dispatch_tree.push_node()
    }

    /// Ends the current node.
    pub(crate) fn pop_dispatch_node(&mut self) {
        self.next_frame.dispatch_tree.pop_node()
    }

    /// Makes `node` current for paint (GPUI's `set_active_node`).
    pub(crate) fn set_active_dispatch_node(&mut self, node: DispatchNodeId) {
        self.next_frame.dispatch_tree.set_active_node(node)
    }

    /// Sets the current dispatch node's key context.
    pub fn set_key_context(&mut self, context: crate::KeyContext) {
        self.next_frame.dispatch_tree.set_key_context(context);
    }

    /// Makes the current dispatch node track `focus_handle` (prepaint: the focus path).
    pub fn set_focus_handle(&mut self, focus_handle: &FocusHandle, _cx: &App) {
        if focus_handle.is_focused(self) {
            self.next_frame.focus = Some(focus_handle.id);
        }
        self.next_frame.dispatch_tree.set_focus_id(focus_handle.id);
    }

    /// Records `focus_handle` in Tab order (paint, as GPUI's `tab_stops.insert`).
    pub(crate) fn insert_tab_stop(&mut self, focus_handle: &FocusHandle) {
        self.next_frame.dispatch_tree.insert_tab_stop(focus_handle.id, focus_handle.tab_index, focus_handle.tab_stop);
    }

    /// Prevents the default of the event being dispatched (a focusable parent then does not take focus on
    /// mouse down, and the platform keeps a caption press).
    pub fn prevent_default(&mut self) {
        self.default_prevented = true;
    }

    /// Whether `prevent_default` was called for the event being dispatched.
    pub fn default_prevented(&self) -> bool {
        self.default_prevented
    }
}
