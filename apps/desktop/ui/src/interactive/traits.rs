//! `InteractiveElement` and `StatefulInteractiveElement`: the builder methods the app calls on `div()` and
//! `div().id(..)`, with GPUI 0.2.2's signatures. Each only records into `Interactivity`. Owner tag:
//! `"window"`.
use std::{any::TypeId, fmt::Debug, rc::Rc};

use crate::{
    Action, AnyView, App, ClickEvent, DispatchPhase, DragMoveEvent, ElementId, FocusHandle, HitboxBehavior,
    Interactivity, KeyContext, KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    Overflow, ScrollHandle, ScrollWheelEvent, SharedString, Stateful, StyleRefinement, Window,
    WindowControlArea,
};

/// An element with event listeners and state styles.
pub trait InteractiveElement: Sized {
    /// The element's interactivity.
    fn interactivity(&mut self) -> &mut Interactivity;

    /// Names a group, for `group_hover` / `group_active` on descendants.
    fn group(mut self, group: impl Into<SharedString>) -> Self {
        self.interactivity().group = Some(group.into());
        self
    }

    /// Gives the element an id, making it stateful (clicks, scrolling, tooltips, hover listeners).
    fn id(mut self, id: impl Into<ElementId>) -> Stateful<Self> {
        self.interactivity().element_id = Some(id.into());
        Stateful { element: self }
    }

    /// Tracks a focus handle (focus styles, key dispatch, tab stops).
    fn track_focus(mut self, focus_handle: &FocusHandle) -> Self {
        self.interactivity().focusable = true;
        self.interactivity().tracked_focus_handle = Some(focus_handle.clone());
        self
    }

    /// Adds a key context (`"TextInput"`); one that does not parse is ignored, as GPUI does.
    fn key_context<C, E>(mut self, key_context: C) -> Self
    where
        C: TryInto<KeyContext, Error = E>,
        E: Debug,
    {
        if let Ok(key_context) = key_context.try_into() {
            self.interactivity().key_context = Some(key_context);
        }
        self
    }

    /// Style while hovered.
    fn hover(mut self, f: impl FnOnce(StyleRefinement) -> StyleRefinement) -> Self {
        self.interactivity().hover_style = Some(Box::new(f(StyleRefinement::default())));
        self
    }

    /// Style while the named group is hovered.
    fn group_hover(
        mut self,
        group_name: impl Into<SharedString>,
        f: impl FnOnce(StyleRefinement) -> StyleRefinement,
    ) -> Self {
        self.interactivity().group_hover_style = Some(crate::interactive::GroupStyle {
            group: group_name.into(),
            style: Box::new(f(StyleRefinement::default())),
        });
        self
    }

    /// Bubble-phase left/right/middle mouse down on the element.
    fn on_mouse_down(
        mut self,
        button: MouseButton,
        listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.interactivity().mouse_down_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Bubble && event.button == button && hitbox.is_hovered(window) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Capture-phase mouse down of any button on the element.
    fn capture_any_mouse_down(
        mut self,
        listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.interactivity().mouse_down_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Capture && hitbox.is_hovered(window) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Mouse up on the element.
    fn on_mouse_up(
        mut self,
        button: MouseButton,
        listener: impl Fn(&MouseUpEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.interactivity().mouse_up_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Bubble && event.button == button && hitbox.is_hovered(window) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Mouse down outside the element (capture phase).
    fn on_mouse_down_out(
        mut self,
        listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        // GPUI tests the raw bounds here (not hover), in the capture phase.
        self.interactivity().mouse_down_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Capture && !hitbox.contains(&window.mouse_position()) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Mouse up of `button` outside the element (capture phase).
    fn on_mouse_up_out(
        mut self,
        button: MouseButton,
        listener: impl Fn(&MouseUpEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.interactivity().mouse_up_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Capture && event.button == button && !hitbox.is_hovered(window) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// A mouse move over the element (bubble phase, only while its hitbox is hovered, as GPUI).
    fn on_mouse_move(mut self, listener: impl Fn(&MouseMoveEvent, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().mouse_move_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Mouse moves while a `T` is dragged over the element.
    fn on_drag_move<T: 'static>(
        mut self,
        listener: impl Fn(&DragMoveEvent<T>, &mut Window, &mut App) + 'static,
    ) -> Self {
        // GPUI: capture phase, for any position, while a `T` is dragged.
        self.interactivity().mouse_move_listeners.push(Box::new(move |event, phase, hitbox, window, cx| {
            if phase == DispatchPhase::Capture
                && let Some(drag) = window.active_drag.as_ref()
                && drag.value_type == TypeId::of::<T>()
            {
                let event = DragMoveEvent::new(event.clone(), hitbox.bounds, drag.value.clone());
                listener(&event, window, cx)
            }
        }));
        self
    }

    /// Scroll wheel over the element.
    fn on_scroll_wheel(
        mut self,
        listener: impl Fn(&ScrollWheelEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.interactivity().scroll_wheel_listeners.push(Box::new(
            move |event, phase, hitbox, window, cx| {
                if phase == DispatchPhase::Bubble && hitbox.should_handle_scroll(window) {
                    listener(event, window, cx)
                }
            },
        ));
        self
    }

    /// Bubble-phase handler of action `A` when the element is on the focus path.
    fn on_action<A: Action>(mut self, listener: impl Fn(&A, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().action_listeners.push((
            TypeId::of::<A>(),
            Box::new(move |action, phase, window, cx| {
                if phase == DispatchPhase::Bubble
                    && let Some(action) = action.downcast_ref::<A>()
                {
                    listener(action, window, cx)
                }
            }),
        ));
        self
    }

    /// Bubble-phase key down when the element is on the focus path.
    fn on_key_down(mut self, listener: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().key_down_listeners.push(Box::new(move |event, phase, window, cx| {
            if phase == DispatchPhase::Bubble {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// Capture-phase key down when the element is on the focus path.
    fn capture_key_down(mut self, listener: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().key_down_listeners.push(Box::new(move |event, phase, window, cx| {
            if phase == DispatchPhase::Capture {
                listener(event, window, cx)
            }
        }));
        self
    }

    /// A `T` (`ExternalPaths` for files) dropped on the element.
    fn on_drop<T: 'static>(mut self, listener: impl Fn(&T, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().drop_listeners.push((
            TypeId::of::<T>(),
            Box::new(move |value, window, cx| {
                if let Some(value) = value.downcast_ref::<T>() {
                    listener(value, window, cx)
                }
            }),
        ));
        self
    }

    /// Blocks mouse events and hover for whatever is under the element.
    fn occlude(mut self) -> Self {
        self.interactivity().hitbox_behavior = HitboxBehavior::BlockMouse;
        self
    }

    /// Makes the element a window control area (caption, minimize, maximize, close) for the platform.
    fn window_control_area(mut self, area: WindowControlArea) -> Self {
        self.interactivity().window_control = Some(area);
        self
    }
}

/// An interactive element with an id: clicks, active styles, scrolling, tooltips, hover listeners.
pub trait StatefulInteractiveElement: InteractiveElement {
    /// Scrolls horizontally on overflow.
    fn overflow_x_scroll(mut self) -> Self {
        self.interactivity().base_style.overflow.x = Some(Overflow::Scroll);
        self
    }

    /// Scrolls vertically on overflow.
    fn overflow_y_scroll(mut self) -> Self {
        self.interactivity().base_style.overflow.y = Some(Overflow::Scroll);
        self
    }

    /// Reports scroll state to (and takes offsets from) `scroll_handle`.
    fn track_scroll(mut self, scroll_handle: &ScrollHandle) -> Self {
        self.interactivity().tracked_scroll_handle = Some(scroll_handle.clone());
        self
    }

    /// Style while pressed.
    fn active(mut self, f: impl FnOnce(StyleRefinement) -> StyleRefinement) -> Self {
        self.interactivity().active_style = Some(Box::new(f(StyleRefinement::default())));
        self
    }

    /// Style while the named group is pressed.
    fn group_active(
        mut self,
        group_name: impl Into<SharedString>,
        f: impl FnOnce(StyleRefinement) -> StyleRefinement,
    ) -> Self {
        self.interactivity().group_active_style = Some(crate::interactive::GroupStyle {
            group: group_name.into(),
            style: Box::new(f(StyleRefinement::default())),
        });
        self
    }

    /// A click (mouse down and up on the element, or Enter / Space while focused, as GPUI 0.2.2).
    fn on_click(mut self, listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().click_listeners.push(Rc::new(listener));
        self
    }

    /// Hover enter (`true`) and leave (`false`).
    fn on_hover(mut self, listener: impl Fn(&bool, &mut Window, &mut App) + 'static) -> Self {
        self.interactivity().hover_listener = Some(Box::new(listener));
        self
    }

    /// A tooltip shown after the hover delay.
    fn tooltip(mut self, build_tooltip: impl Fn(&mut Window, &mut App) -> AnyView + 'static) -> Self {
        self.interactivity().tooltip_builder = Some(Rc::new(build_tooltip));
        self
    }
}
