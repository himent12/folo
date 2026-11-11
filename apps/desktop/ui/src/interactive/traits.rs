//! `InteractiveElement` and `StatefulInteractiveElement`: the builder methods the app calls on `div()` and
//! `div().id(..)`, with GPUI 0.2.2's signatures. Each only records into `Interactivity`. Owner tag:
//! `"window"`.
#![allow(dead_code)] // skeleton

use std::fmt::Debug;

use crate::{
    Action, AnyView, App, ClickEvent, DragMoveEvent, ElementId, FocusHandle, Interactivity, KeyContext,
    KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ScrollHandle, ScrollWheelEvent,
    SharedString, Stateful, StyleRefinement, Window, WindowControlArea,
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
        _button: MouseButton,
        _listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Capture-phase mouse down of any button on the element.
    fn capture_any_mouse_down(
        mut self,
        _listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Mouse up on the element.
    fn on_mouse_up(
        mut self,
        _button: MouseButton,
        _listener: impl Fn(&MouseUpEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Mouse down outside the element (capture phase).
    fn on_mouse_down_out(
        mut self,
        _listener: impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Mouse up of `button` outside the element (capture phase).
    fn on_mouse_up_out(
        mut self,
        _button: MouseButton,
        _listener: impl Fn(&MouseUpEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// A mouse move over the element (bubble phase, only while its hitbox is hovered, as GPUI).
    fn on_mouse_move(mut self, _listener: impl Fn(&MouseMoveEvent, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Mouse moves while a `T` is dragged over the element.
    fn on_drag_move<T: 'static>(
        mut self,
        _listener: impl Fn(&DragMoveEvent<T>, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Scroll wheel over the element.
    fn on_scroll_wheel(
        mut self,
        _listener: impl Fn(&ScrollWheelEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Bubble-phase handler of action `A` when the element is on the focus path.
    fn on_action<A: Action>(mut self, _listener: impl Fn(&A, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Bubble-phase key down when the element is on the focus path.
    fn on_key_down(mut self, _listener: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Capture-phase key down when the element is on the focus path.
    fn capture_key_down(
        mut self,
        _listener: impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// A `T` (`ExternalPaths` for files) dropped on the element.
    fn on_drop<T: 'static>(mut self, _listener: impl Fn(&T, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Blocks mouse events and hover for whatever is under the element.
    fn occlude(mut self) -> Self {
        self.interactivity().occlude_mouse = true;
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
        let _ = self.interactivity();
        todo!("window")
    }

    /// Scrolls vertically on overflow.
    fn overflow_y_scroll(mut self) -> Self {
        let _ = self.interactivity();
        todo!("window")
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
    fn on_click(mut self, _listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// Hover enter (`true`) and leave (`false`).
    fn on_hover(mut self, _listener: impl Fn(&bool, &mut Window, &mut App) + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }

    /// A tooltip shown after the hover delay.
    fn tooltip(mut self, _build_tooltip: impl Fn(&mut Window, &mut App) -> AnyView + 'static) -> Self {
        let _ = self.interactivity();
        todo!("window")
    }
}
