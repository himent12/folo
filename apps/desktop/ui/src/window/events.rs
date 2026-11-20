//! Input dispatch, in the order of GPUI 0.2.2's `window.rs`. Owner tag: `"window"`.
//!
//! Mouse events are hit-tested against the rendered frame (the cursor is reset when the hovered set
//! changes) and dispatched to its mouse listeners in capture (paint order) then bubble (reverse paint
//! order) phases, honouring `stop_propagation` and `prevent_default`; file drags become an active drag
//! carrying `ExternalPaths` and the matching mouse events. Key events walk the dispatch tree along the
//! focus path: first the keymap (bindings under the path's contexts, each action dispatched capture,
//! bubble, then to the app's global handlers until one is handled), then the key listeners (capture root
//! to leaf, bubble leaf to root), then, when nothing handled a key that types text, the text goes to the
//! focused input handler. Window control areas answer the platform's hit test.

use std::any::Any;

use crate::{
    Action, AnyDrag, App, DispatchEventResult, DispatchPhase, ExternalPaths, FileDropEvent, KeyDownEvent,
    Keystroke, Modifiers, MouseButton, MouseMoveEvent, MouseUpEvent, PlatformInput, Window,
    WindowControlArea, window::DispatchNodeId,
};

impl Window {
    /// The `on_input` callback: updates mouse position and modifiers, then dispatches. The result tells the
    /// platform whether the event is still unhandled (`propagate`) and whether a handler prevented its
    /// default (a caption press then stays with the app).
    pub(crate) fn dispatch_event(&mut self, event: PlatformInput, cx: &mut App) -> DispatchEventResult {
        cx.propagate_event = true;
        self.default_prevented = false;

        let event = match event {
            PlatformInput::MouseMove(event) => {
                self.mouse_position = event.position;
                self.modifiers = event.modifiers;
                PlatformInput::MouseMove(event)
            }
            PlatformInput::MouseDown(event) => {
                self.mouse_position = event.position;
                self.modifiers = event.modifiers;
                PlatformInput::MouseDown(event)
            }
            PlatformInput::MouseUp(event) => {
                self.mouse_position = event.position;
                self.modifiers = event.modifiers;
                PlatformInput::MouseUp(event)
            }
            PlatformInput::MouseExited(event) => {
                self.modifiers = event.modifiers;
                PlatformInput::MouseExited(event)
            }
            PlatformInput::ModifiersChanged(event) => {
                self.modifiers = event.modifiers;
                PlatformInput::ModifiersChanged(event)
            }
            PlatformInput::ScrollWheel(event) => {
                self.mouse_position = event.position;
                self.modifiers = event.modifiers;
                PlatformInput::ScrollWheel(event)
            }
            PlatformInput::FileDrop(event) => self.translate_file_drop(event, cx),
            event @ (PlatformInput::KeyDown(_) | PlatformInput::KeyUp(_)) => event,
        };
        // A pointer event means the mouse is over the window (Windows sends no `MouseExited`, and the
        // hover callback may lag the first move). Checked after translation: a file dragged in from
        // outside reaches the window only as file-drop events, which became moves and an up above.
        if matches!(
            event,
            PlatformInput::MouseMove(_)
                | PlatformInput::MouseDown(_)
                | PlatformInput::MouseUp(_)
                | PlatformInput::ScrollWheel(_)
        ) {
            self.hovered.set(true);
        }

        if let Some(mouse_event) = event.mouse_event() {
            self.dispatch_mouse_event(mouse_event, cx);
        } else if let Some(key_event) = event.keyboard_event() {
            self.dispatch_key_event(key_event, cx);
            if let PlatformInput::KeyDown(key_down) = &event
                && cx.propagate_event
                && self.platform_window.inserts_key_text()
            {
                self.insert_text_for_key(key_down, cx);
            }
        }

        DispatchEventResult { propagate: cx.propagate_event, default_prevented: self.default_prevented }
    }

    /// GPUI's translation of a file drop: entering starts an active drag of `ExternalPaths` (a left-button
    /// mouse move), hovering moves, dropping is a left mouse up (the drop listeners take the drag), leaving
    /// cancels the drag. Leaving (or cancelling) also ends the hover the drag's moves set, as the mouse
    /// leaving the window does: the platforms report no hover change during a drag, and a real mouse move
    /// over the window hovers it again.
    fn translate_file_drop(&mut self, event: FileDropEvent, cx: &mut App) -> PlatformInput {
        let mouse_move = |position| {
            PlatformInput::MouseMove(MouseMoveEvent {
                position,
                pressed_button: Some(MouseButton::Left),
                modifiers: Modifiers::default(),
            })
        };
        match event {
            FileDropEvent::Entered { position, paths } => {
                self.mouse_position = position;
                if self.active_drag.is_none() {
                    self.active_drag = Some(AnyDrag::new(paths));
                }
                mouse_move(position)
            }
            FileDropEvent::Pending { position } => {
                self.mouse_position = position;
                mouse_move(position)
            }
            FileDropEvent::Submit { position } => {
                cx.activate(true);
                self.mouse_position = position;
                PlatformInput::MouseUp(MouseUpEvent {
                    button: MouseButton::Left,
                    position,
                    modifiers: Modifiers::default(),
                    click_count: 1,
                })
            }
            FileDropEvent::Exited => {
                self.active_drag = None;
                if self.hovered.get() {
                    self.on_hover_status_change(false, cx);
                    cx.propagate_event = true;
                }
                PlatformInput::FileDrop(FileDropEvent::Exited)
            }
        }
    }

    /// Dispatches a mouse event to the rendered frame's mouse listeners.
    pub(crate) fn dispatch_mouse_event(&mut self, event: &dyn Any, cx: &mut App) {
        let hit_test = self.hit_test_mouse(&self.rendered_frame);
        if hit_test != self.mouse_hit_test {
            self.mouse_hit_test = hit_test;
            self.reset_cursor_style(cx);
        }

        // The listeners are taken out for the dispatch (they get the window) and put back after.
        let mut listeners = std::mem::take(&mut self.rendered_frame.mouse_listeners);
        for listener in listeners.iter_mut().flatten() {
            listener(event, DispatchPhase::Capture, self, cx);
            if !cx.propagate_event {
                break;
            }
        }
        if cx.propagate_event {
            for listener in listeners.iter_mut().rev().flatten() {
                listener(event, DispatchPhase::Bubble, self, cx);
                if !cx.propagate_event {
                    break;
                }
            }
        }
        // A draw cannot happen during dispatch (the window is leased), so these are this frame's.
        self.rendered_frame.mouse_listeners = listeners;

        if self.active_drag.is_some() {
            if event.is::<MouseMoveEvent>() {
                self.refresh();
            } else if event.is::<MouseUpEvent>() {
                self.active_drag = None;
                self.refresh();
            }
        }
    }

    /// The dispatch node of the focused element in the rendered frame (the root when nothing is focused).
    fn focus_node(&self) -> Option<DispatchNodeId> {
        let tree = &self.rendered_frame.dispatch_tree;
        self.focus.and_then(|id| tree.focusable_node_id(id)).or_else(|| tree.root_node_id())
    }

    /// Dispatches a key event: bindings first, then key listeners (GPUI 0.2.2's `dispatch_key_event`). A
    /// dirty window is drawn first so the dispatch tree matches the state.
    pub(crate) fn dispatch_key_event(&mut self, event: &dyn Any, cx: &mut App) {
        if self.dirty.get() {
            self.draw(cx);
        }
        let Some(node) = self.focus_node() else { return };
        if let Some(key_down) = event.downcast_ref::<KeyDownEvent>() {
            cx.propagate_event = true;
            if self.dispatch_keystroke_bindings(&key_down.keystroke, cx) {
                return;
            }
        }
        self.dispatch_key_listeners(event, node, cx);
    }

    /// Key listeners along the focus path: capture root to leaf, then bubble leaf to root.
    fn dispatch_key_listeners(&mut self, event: &dyn Any, node: DispatchNodeId, cx: &mut App) {
        let path = self.rendered_frame.dispatch_tree.dispatch_path(node);
        for &id in &path {
            let listeners = self.rendered_frame.dispatch_tree.node(id).key_listeners.clone();
            for listener in listeners {
                listener(event, DispatchPhase::Capture, self, cx);
                if !cx.propagate_event {
                    return;
                }
            }
        }
        for &id in path.iter().rev() {
            let listeners = self.rendered_frame.dispatch_tree.node(id).key_listeners.clone();
            for listener in listeners {
                listener(event, DispatchPhase::Bubble, self, cx);
                if !cx.propagate_event {
                    return;
                }
            }
        }
    }

    /// Matches `keystroke` against the keymap under the focus path's contexts and dispatches each bound
    /// action, best match first, until one is handled. Returns whether one was handled.
    pub(crate) fn dispatch_keystroke_bindings(&mut self, keystroke: &Keystroke, cx: &mut App) -> bool {
        let Some(node) = self.focus_node() else { return false };
        let path = self.rendered_frame.dispatch_tree.dispatch_path(node);
        let matched = self.rendered_frame.dispatch_tree.bindings_for_input(keystroke, &path);
        for action in matched.bindings {
            self.dispatch_action_on_node(node, action.as_ref(), cx);
            if !cx.propagate_event {
                return true;
            }
        }
        false
    }

    /// Sends the key's text to the focused input handler (`replace_text_in_range(None, text)`) when no
    /// listener or binding handled the key, the key types text (`key_char`) and no modifier other than
    /// shift is held. Only on a platform without text input of its own
    /// (`PlatformWindow::inserts_key_text`, the headless one): Win32 and winit type through the OS
    /// (`WM_CHAR` and the IME, winit's text and IME events), as GPUI's platforms do. The key then
    /// counts as handled.
    pub(crate) fn insert_text_for_key(&mut self, event: &KeyDownEvent, cx: &mut App) {
        let keystroke = &event.keystroke;
        let modifiers = keystroke.modifiers;
        if modifiers.control || modifiers.alt || modifiers.platform || modifiers.function {
            return;
        }
        let Some(text) = keystroke.key_char.as_deref() else { return };
        let Some(handler) = self.input_handler.clone() else { return };
        handler.0.borrow_mut().replace_text_in_range(None, text, self, cx);
        cx.propagate_event = false;
    }

    /// Dispatches `action` from the focused element (or the root) after the current effects: capture,
    /// bubble, then global handlers.
    pub fn dispatch_action(&mut self, action: Box<dyn Action>, cx: &mut App) {
        let focus = self.focused_id(cx);
        let handle = self.handle;
        cx.defer(move |cx| {
            let _ = handle.update(cx, |_, window, cx| {
                let tree = &window.rendered_frame.dispatch_tree;
                let node = focus.and_then(|id| tree.focusable_node_id(id)).or_else(|| tree.root_node_id());
                if let Some(node) = node {
                    window.dispatch_action_on_node(node, action.as_ref(), cx);
                }
            });
        });
    }

    /// GPUI's `dispatch_action_on_node`: global capture, window capture (root to leaf), window bubble (leaf
    /// to root; a handler stops propagation unless it calls `cx.propagate()`), then global bubble.
    pub(crate) fn dispatch_action_on_node(
        &mut self,
        node: DispatchNodeId,
        action: &dyn Action,
        cx: &mut App,
    ) {
        let path = self.rendered_frame.dispatch_tree.dispatch_path(node);
        let action_type = action.as_any().type_id();

        cx.propagate_event = true;
        if let Some(listeners) = cx.global_action_listeners.get(&action_type).cloned() {
            for listener in &listeners {
                listener(action.as_any(), DispatchPhase::Capture, cx);
                if !cx.propagate_event {
                    break;
                }
            }
        }
        if !cx.propagate_event {
            return;
        }

        for &id in &path {
            let listeners = self.rendered_frame.dispatch_tree.node(id).action_listeners.clone();
            for listener in listeners {
                if listener.action_type == action_type {
                    (listener.listener)(action.as_any(), DispatchPhase::Capture, self, cx);
                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }
        for &id in path.iter().rev() {
            let listeners = self.rendered_frame.dispatch_tree.node(id).action_listeners.clone();
            for listener in listeners {
                if listener.action_type == action_type {
                    cx.propagate_event = false;
                    (listener.listener)(action.as_any(), DispatchPhase::Bubble, self, cx);
                    if !cx.propagate_event {
                        return;
                    }
                }
            }
        }

        if let Some(listeners) = cx.global_action_listeners.get(&action_type).cloned() {
            for listener in listeners.iter().rev() {
                cx.propagate_event = false;
                listener(action.as_any(), DispatchPhase::Bubble, cx);
                if !cx.propagate_event {
                    break;
                }
            }
        }
    }

    /// The `on_hover_status_change` callback. Leaving the window clears hover: an empty hit test, and the
    /// rendered frame's mouse-move listeners run once with it, so hover styles redraw, `on_hover(false)`
    /// fires and pending tooltips cancel (the platforms send no `MouseExited` for this).
    pub(crate) fn on_hover_status_change(&mut self, hovered: bool, cx: &mut App) {
        self.hovered.set(hovered);
        if !hovered {
            let event = MouseMoveEvent {
                position: self.mouse_position,
                pressed_button: None,
                modifiers: self.modifiers,
            };
            cx.propagate_event = true;
            self.dispatch_mouse_event(&event, cx);
        }
        self.refresh();
    }

    /// The `on_hit_test_window_control` callback: the control area under the mouse in the rendered frame
    /// (an area under an occluding element is not reported). Public so that a listener that stops a press
    /// can leave one on a caption area to the platform.
    pub fn hit_test_window_control(&self) -> Option<WindowControlArea> {
        let hit_test = self.rendered_frame.hit_test(self.mouse_position);
        self.rendered_frame
            .window_control_hitboxes
            .iter()
            .find(|control| hit_test.ids.contains(&control.hitbox.id))
            .map(|control| control.area)
    }
}

impl AnyDrag {
    /// A drag of external files (GPUI also keeps where it started, for a drag view; file drags have none).
    pub(crate) fn new(paths: ExternalPaths) -> Self {
        AnyDrag { value: std::sync::Arc::new(paths), value_type: std::any::TypeId::of::<ExternalPaths>() }
    }
}
