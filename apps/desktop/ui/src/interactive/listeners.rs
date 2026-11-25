//! The listeners an interactive element registers on the window at paint, as GPUI 0.2.2's `div.rs`
//! (`paint_mouse_listeners`, `paint_keyboard_listeners`, `paint_hover_group_handler`,
//! `paint_scroll_listener`): focus on mouse down, the element's own mouse listeners, hover redraws, drops,
//! click detection (down and up on the same element; Enter / Space key-up while focused), hover
//! listeners, tooltips and the pressed (active) state. Owner tag: `"window"`.

use std::rc::Rc;

use crate::{
    ClickEvent, DispatchPhase, Hitbox, HitboxId, InteractiveElementState, Interactivity, KeyUpEvent,
    KeyboardButton, KeyboardClickEvent, MouseButton, MouseClickEvent, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, Overflow, Pixels, ScrollWheelEvent, Style, Window,
    interactive::{ElementClickedState, tooltip},
};

impl Interactivity {
    /// Registers the mouse listeners for this frame (the element has a hitbox).
    pub(crate) fn paint_mouse_listeners(
        &mut self,
        hitbox: &Hitbox,
        element_state: Option<&mut InteractiveElementState>,
        window: &mut Window,
    ) {
        let is_focused = self.tracked_focus_handle.as_ref().is_some_and(|handle| handle.is_focused(window));

        // A focusable element takes focus on mouse down, unless a child already did (`prevent_default`).
        if let Some(focus_handle) = self.tracked_focus_handle.clone() {
            window.on_mouse_down_focus(hitbox.id, focus_handle);
        }

        for listener in self.mouse_down_listeners.drain(..) {
            let hitbox = hitbox.clone();
            window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
                listener(event, phase, &hitbox, window, cx)
            });
        }
        for listener in self.mouse_up_listeners.drain(..) {
            let hitbox = hitbox.clone();
            window.on_mouse_event(move |event: &MouseUpEvent, phase, window, cx| {
                listener(event, phase, &hitbox, window, cx)
            });
        }
        for listener in self.mouse_move_listeners.drain(..) {
            let hitbox = hitbox.clone();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, window, cx| {
                listener(event, phase, &hitbox, window, cx)
            });
        }
        for listener in self.scroll_wheel_listeners.drain(..) {
            let hitbox = hitbox.clone();
            window.on_mouse_event(move |event: &ScrollWheelEvent, phase, window, cx| {
                listener(event, phase, &hitbox, window, cx)
            });
        }

        // Hover styles (and a cursor) redraw when the hover state changes, and only then.
        if self.hover_style.is_some() || self.base_style.mouse_cursor.is_some() {
            window.on_hover_refresh(hitbox.id, hitbox.is_hovered(window));
        }

        let drop_listeners = std::mem::take(&mut self.drop_listeners);
        if !drop_listeners.is_empty() {
            let hitbox = hitbox.clone();
            window.on_mouse_event(move |_: &MouseUpEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble || !hitbox.is_hovered(window) {
                    return;
                }
                let Some(drag_type) = window.active_drag.as_ref().map(|drag| drag.value_type) else { return };
                for (drop_type, listener) in &drop_listeners {
                    if *drop_type == drag_type
                        && let Some(drag) = window.active_drag.take()
                    {
                        listener(drag.value.as_ref(), window, cx);
                        window.refresh();
                        cx.stop_propagation();
                    }
                }
            });
        }

        let Some(element_state) = element_state else { return };
        let click_listeners = std::mem::take(&mut self.click_listeners);
        if !click_listeners.is_empty() {
            self.paint_click_listeners(hitbox, element_state, click_listeners, is_focused, window);
        }

        if let Some(hover_listener) = self.hover_listener.take() {
            let hitbox = hitbox.clone();
            let was_hovered = element_state.hover_state.get_or_insert_with(Default::default).clone();
            let has_mouse_down =
                element_state.pending_mouse_down.get_or_insert_with(Default::default).clone();
            window.on_mouse_event(move |_: &MouseMoveEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble {
                    return;
                }
                let is_hovered = has_mouse_down.borrow().is_none()
                    && window.active_drag.is_none()
                    && hitbox.is_hovered(window);
                let changed = {
                    let mut was_hovered = was_hovered.borrow_mut();
                    let changed = is_hovered != *was_hovered;
                    *was_hovered = is_hovered;
                    changed
                };
                if changed {
                    hover_listener(&is_hovered, window, cx);
                }
            });
        }

        if let Some(build_tooltip) = self.tooltip_builder.take() {
            let active_tooltip = element_state.active_tooltip.get_or_insert_with(Default::default).clone();
            let pending_mouse_down =
                element_state.pending_mouse_down.get_or_insert_with(Default::default).clone();
            let check_is_hovered_during_prepaint = Rc::new({
                let pending_mouse_down = pending_mouse_down.clone();
                let source_bounds = hitbox.bounds;
                move |window: &Window| {
                    window.hovered.get()
                        && pending_mouse_down.borrow().is_none()
                        && source_bounds.contains(&window.mouse_position())
                }
            });
            let check_is_hovered = Rc::new({
                let hitbox = hitbox.clone();
                move |window: &Window| pending_mouse_down.borrow().is_none() && hitbox.is_hovered(window)
            });
            tooltip::register_tooltip_mouse_handlers(
                &active_tooltip,
                self.tooltip_id,
                build_tooltip,
                check_is_hovered,
                check_is_hovered_during_prepaint,
                window,
            );
        }

        // Track pressed state only when an active style can observe it (or to clear a press already in
        // flight if the style was removed while held).
        let is_currently_clicked =
            element_state.clicked_state.as_ref().is_some_and(|state| state.borrow().is_clicked());
        if self.active_style.is_some() || self.group_active_style.is_some() || is_currently_clicked {
            let active_state = element_state.clicked_state.get_or_insert_with(Default::default).clone();
            if is_currently_clicked {
                window.on_mouse_event(move |_: &MouseUpEvent, phase, window, _| {
                    if phase == DispatchPhase::Capture {
                        *active_state.borrow_mut() = ElementClickedState::default();
                        window.refresh();
                    }
                });
            } else {
                let group_hitbox = self
                    .group_active_style
                    .as_ref()
                    .and_then(|group| window.group_hitboxes.get(&group.group));
                let hitbox = hitbox.clone();
                window.on_mouse_event(move |_: &MouseDownEvent, phase, window, _| {
                    if phase == DispatchPhase::Bubble && !window.default_prevented() {
                        let group = group_hitbox.is_some_and(|id: HitboxId| id.is_hovered(window));
                        let element = hitbox.is_hovered(window);
                        if group || element {
                            *active_state.borrow_mut() = ElementClickedState { group, element };
                            window.refresh();
                        }
                    }
                });
            }
        }
    }

    /// Click detection: a left mouse down on the element is remembered; a mouse up over it fires the
    /// click (the capture phase clears the pending down even if a handler stops propagation); a focused
    /// element is clicked by Enter or Space (key up, no modifiers).
    fn paint_click_listeners(
        &mut self,
        hitbox: &Hitbox,
        element_state: &mut InteractiveElementState,
        click_listeners: Vec<crate::interactive::ClickListener>,
        is_focused: bool,
        window: &mut Window,
    ) {
        let pending_mouse_down =
            element_state.pending_mouse_down.get_or_insert_with(Default::default).clone();
        let click_listeners: Rc<[crate::interactive::ClickListener]> = click_listeners.into();

        window.on_mouse_event({
            let pending_mouse_down = pending_mouse_down.clone();
            let hitbox = hitbox.clone();
            move |event: &MouseDownEvent, phase, window, _| {
                if phase == DispatchPhase::Bubble
                    && event.button == MouseButton::Left
                    && hitbox.is_hovered(window)
                {
                    *pending_mouse_down.borrow_mut() = Some(event.clone());
                    window.refresh();
                }
            }
        });

        if is_focused {
            let click_listeners = click_listeners.clone();
            let bounds = hitbox.bounds;
            window.on_key_event(move |event: &KeyUpEvent, phase, window, cx| {
                if !phase.bubble() || window.default_prevented() {
                    return;
                }
                let stroke = &event.keystroke;
                let button = match stroke.key.as_str() {
                    "enter" => KeyboardButton::Enter,
                    "space" => KeyboardButton::Space,
                    _ => return,
                };
                if stroke.modifiers.modified() {
                    return;
                }
                let click = ClickEvent::Keyboard(KeyboardClickEvent { button, bounds });
                for listener in click_listeners.iter() {
                    listener(&click, window, cx);
                }
            });
        }

        window.on_mouse_event({
            let mut captured_mouse_down = None;
            let hitbox = hitbox.clone();
            move |event: &MouseUpEvent, phase, window, cx| match phase {
                DispatchPhase::Capture => {
                    let mut pending = pending_mouse_down.borrow_mut();
                    if pending.is_some() {
                        // Kept only when released over the element (GPUI: a moved element is not clicked).
                        captured_mouse_down = pending.take().filter(|_| hitbox.is_hovered(window));
                        window.refresh();
                    }
                }
                DispatchPhase::Bubble => {
                    if let Some(down) = captured_mouse_down.take() {
                        let click = ClickEvent::Mouse(MouseClickEvent { down, up: event.clone() });
                        for listener in click_listeners.iter() {
                            listener(&click, window, cx);
                        }
                    }
                }
            }
        });
    }

    /// Registers the key context, key and action listeners on the current dispatch node.
    pub(crate) fn paint_keyboard_listeners(&mut self, window: &mut Window) {
        if let Some(context) = self.key_context.clone() {
            window.set_key_context(context);
        }
        for listener in self.key_down_listeners.drain(..) {
            window.on_key_event(move |event: &crate::KeyDownEvent, phase, window, cx| {
                listener(event, phase, window, cx)
            });
        }
        for (action_type, listener) in self.action_listeners.drain(..) {
            window.on_action(action_type, listener);
        }
    }

    /// A `group_hover` style redraws when its group's hover state changes.
    pub(crate) fn paint_hover_group_handler(&self, window: &mut Window) {
        let group_hitbox = self
            .group_hover_style
            .as_ref()
            .and_then(|group_hover| window.group_hitboxes.get(&group_hover.group));
        if let Some(group_hitbox) = group_hitbox {
            window.on_hover_refresh(group_hitbox, group_hitbox.is_hovered(window));
        }
    }

    /// A scroll container moves its offset by the wheel delta (line deltas at the current line height),
    /// clamped to `-scroll_max..=0` on the axes that scroll, and redraws only when the clamped offset
    /// actually changes.
    pub(crate) fn paint_scroll_listener(&self, hitbox: &Hitbox, style: &Style, window: &mut Window) {
        let Some(scroll_offset) = self.scroll_offset.clone() else { return };
        let scroll_max = self.scroll_max;
        let overflow = style.overflow;
        let allow_concurrent_scroll = style.allow_concurrent_scroll;
        let restrict_scroll_to_axis = style.restrict_scroll_to_axis;
        let line_height = window.line_height();
        let hitbox = hitbox.clone();
        window.on_mouse_event(move |event: &ScrollWheelEvent, phase, window, _| {
            if phase != DispatchPhase::Bubble || !hitbox.should_handle_scroll(window) {
                return;
            }
            let delta = event.delta.pixel_delta(line_height);
            let mut delta_x = Pixels::ZERO;
            if overflow.x == Overflow::Scroll {
                if delta.x != Pixels::ZERO {
                    delta_x = delta.x;
                } else if !restrict_scroll_to_axis && overflow.y != Overflow::Scroll {
                    delta_x = delta.y;
                }
            }
            let mut delta_y = Pixels::ZERO;
            if overflow.y == Overflow::Scroll {
                if delta.y != Pixels::ZERO {
                    delta_y = delta.y;
                } else if !restrict_scroll_to_axis && overflow.x != Overflow::Scroll {
                    delta_y = delta.x;
                }
            }
            if !allow_concurrent_scroll && delta_x != Pixels::ZERO && delta_y != Pixels::ZERO {
                if delta_x.abs() > delta_y.abs() {
                    delta_y = Pixels::ZERO;
                } else {
                    delta_x = Pixels::ZERO;
                }
            }
            let clamp_axis = |current: Pixels, delta: Pixels, max: Pixels| -> Pixels {
                if delta == Pixels::ZERO {
                    return current;
                }
                let next = current + delta;
                if max.0.is_finite() && max.0 >= 0.0 && next.0.is_finite() {
                    let min = if max.0 == 0.0 { Pixels::ZERO } else { -max };
                    next.clamp(min, Pixels::ZERO)
                } else {
                    next
                }
            };
            let moved = {
                let mut offset = scroll_offset.borrow_mut();
                let old = *offset;
                offset.x = clamp_axis(offset.x, delta_x, scroll_max.width);
                offset.y = clamp_axis(offset.y, delta_y, scroll_max.height);
                *offset != old
            };
            if moved {
                window.refresh();
            }
        });
    }
}
