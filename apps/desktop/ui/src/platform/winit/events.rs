//! A window's winit events, turned into the core's callbacks with GPUI's semantics.

use super::chrome::ChromeAction;
use super::keys::{self, KeyInput, TextRules};
use super::mouse;
use super::window::{WindowInner, call};
use crate::{
    KeyDownEvent, KeyUpEvent, ModifiersChangedEvent, MouseDownEvent, MouseExitEvent, MouseMoveEvent,
    MouseUpEvent, PlatformInput, PlatformInputHandler, RequestFrameOptions, ScrollWheelEvent, point, px,
};
use std::ops::Range;
use std::time::Instant;
use winit::event::{ElementState, Ime, KeyEvent, MouseButton as WinitButton, WindowEvent};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

impl WindowInner {
    /// Handles one event for this window.
    pub fn handle_event(&self, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                let options = {
                    let mut state = self.state.borrow_mut();
                    state.redraw_pending = false;
                    let requested = std::mem::take(&mut state.frame_requested);
                    // A redraw the system asked for (expose) must show a whole frame.
                    RequestFrameOptions {
                        require_presentation: !requested || state.full_present,
                        force_render: false,
                    }
                };
                self.draw(options);
            }
            WindowEvent::Resized(_) => {
                self.state.borrow_mut().full_present = true;
                self.track_restore_bounds();
                self.notify_resize();
                // Synchronously, so a live resize never shows an unpainted edge.
                self.state.borrow_mut().frame_requested = false;
                self.draw(RequestFrameOptions { require_presentation: true, force_render: false });
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                {
                    let mut state = self.state.borrow_mut();
                    state.scale_factor = scale_factor as f32;
                    state.full_present = true;
                }
                self.notify_resize();
            }
            WindowEvent::Moved(_) => {
                self.track_restore_bounds();
                self.update_frame_interval();
                call(&self.callbacks.moved, |callback| callback());
            }
            WindowEvent::CloseRequested => self.request_close(),
            WindowEvent::Focused(active) => {
                self.state.borrow_mut().active = active;
                call(&self.callbacks.active_status_change, |callback| callback(active));
            }
            WindowEvent::ThemeChanged(_) => {
                call(&self.callbacks.appearance_changed, |callback| callback());
            }
            WindowEvent::CursorEntered { .. } => {
                self.state.borrow_mut().hovered = true;
                call(&self.callbacks.hover_status_change, |callback| callback(true));
            }
            WindowEvent::CursorLeft { .. } => {
                let (position, pressed_button, modifiers) = {
                    let mut state = self.state.borrow_mut();
                    state.hovered = false;
                    (state.mouse_position, state.pressed_buttons.first().copied(), state.modifiers)
                };
                self.dispatch(PlatformInput::MouseExited(MouseExitEvent {
                    position,
                    pressed_button,
                    modifiers,
                }));
                call(&self.callbacks.hover_status_change, |callback| callback(false));
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = self.scale_factor();
                let position = point(px(position.x as f32 / scale), px(position.y as f32 / scale));
                let (dragging_files, pressed_button, modifiers) = {
                    let mut state = self.state.borrow_mut();
                    state.mouse_position = position;
                    (state.file_drops.is_active(), state.pressed_buttons.first().copied(), state.modifiers)
                };
                let input = if dragging_files {
                    PlatformInput::FileDrop(crate::FileDropEvent::Pending { position })
                } else {
                    PlatformInput::MouseMove(MouseMoveEvent { position, pressed_button, modifiers })
                };
                self.dispatch(input);
            }
            WindowEvent::MouseInput { state: element, button, .. } => self.mouse_button(element, button),
            WindowEvent::MouseWheel { delta, phase, .. } => {
                let (position, modifiers, scale) = {
                    let state = self.state.borrow();
                    (state.mouse_position, state.modifiers, state.scale_factor)
                };
                self.dispatch(PlatformInput::ScrollWheel(ScrollWheelEvent {
                    position,
                    delta: mouse::scroll_delta(delta, scale),
                    modifiers,
                    touch_phase: mouse::touch_phase(phase),
                }));
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                let (modifiers, capslock) = {
                    let mut state = self.state.borrow_mut();
                    state.modifiers = keys::modifiers(modifiers.state());
                    (state.modifiers, state.capslock)
                };
                self.dispatch(PlatformInput::ModifiersChanged(ModifiersChangedEvent { modifiers, capslock }));
            }
            WindowEvent::KeyboardInput { event, is_synthetic, .. } => {
                // Synthetic presses (keys held while focus arrived) are not typing.
                if !is_synthetic {
                    self.key(&event);
                }
            }
            WindowEvent::Ime(ime) => self.ime(ime),
            WindowEvent::HoveredFile(path) => self.state.borrow_mut().file_drops.hovered(path),
            WindowEvent::DroppedFile(path) => self.state.borrow_mut().file_drops.dropped(path),
            WindowEvent::HoveredFileCancelled => self.state.borrow_mut().file_drops.cancelled(),
            _ => {}
        }
    }

    /// End of an event batch: report the file drops it added up to, forget the batch's key text.
    pub fn end_batch(&self) {
        let events = {
            let mut state = self.state.borrow_mut();
            state.key_text = None;
            state.commit_text = None;
            let position = state.mouse_position;
            state.file_drops.flush(position)
        };
        for event in events {
            self.dispatch(PlatformInput::FileDrop(event));
        }
    }

    fn notify_resize(&self) {
        let (size, scale) = (self.content_size(), self.scale_factor());
        call(&self.callbacks.resize, |callback| callback(size, scale));
    }

    /// The user (or a caption Close button) asked to close; the core may refuse.
    pub fn request_close(&self) {
        if call(&self.callbacks.should_close, |callback| callback()).unwrap_or(true) {
            self.close();
        }
    }

    fn mouse_button(&self, element: ElementState, button: WinitButton) {
        let Some(button) = mouse::button(button) else { return };
        let left = button == crate::MouseButton::Left;
        let area = if left {
            call(&self.callbacks.hit_test_window_control, |callback| callback()).flatten()
        } else {
            None
        };
        match element {
            ElementState::Pressed => {
                let (position, modifiers, click_count) = {
                    let mut state = self.state.borrow_mut();
                    let position = state.mouse_position;
                    let count = state.clicks.press(button, position, Instant::now());
                    if !state.pressed_buttons.contains(&button) {
                        state.pressed_buttons.push(button);
                    }
                    (position, state.modifiers, count)
                };
                let result = self.dispatch(PlatformInput::MouseDown(MouseDownEvent {
                    button,
                    position,
                    modifiers,
                    click_count,
                    first_mouse: false,
                }));
                if left {
                    let handled = !result.propagate || result.default_prevented;
                    let action = self.state.borrow_mut().chrome.press(area, handled, click_count);
                    self.act(action);
                }
            }
            ElementState::Released => {
                let (position, modifiers, click_count) = {
                    let mut state = self.state.borrow_mut();
                    state.pressed_buttons.retain(|b| *b != button);
                    (state.mouse_position, state.modifiers, state.clicks.count())
                };
                let result = self.dispatch(PlatformInput::MouseUp(MouseUpEvent {
                    button,
                    position,
                    modifiers,
                    click_count,
                }));
                if left {
                    let action = self.state.borrow_mut().chrome.release(area, !result.propagate);
                    self.act(action);
                }
            }
        }
    }

    fn act(&self, action: ChromeAction) {
        match action {
            ChromeAction::None => {}
            ChromeAction::Drag => {
                if self.window.drag_window().is_ok() {
                    // The system owns the pointer until the move ends; its release never reaches us.
                    self.state.borrow_mut().pressed_buttons.retain(|b| *b != crate::MouseButton::Left);
                }
            }
            ChromeAction::ToggleMaximize => self.window.set_maximized(!self.window.is_maximized()),
            ChromeAction::Minimize => self.window.set_minimized(true),
            ChromeAction::Close => self.request_close(),
        }
    }

    fn key(&self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        let unmodified = event.key_without_modifiers();
        let modifiers = {
            let mut state = self.state.borrow_mut();
            if pressed && keys::is_capslock(&event.logical_key) {
                state.capslock = keys::toggle(state.capslock);
            }
            state.modifiers
        };
        let input = KeyInput {
            logical: &event.logical_key,
            unmodified: &unmodified,
            physical: event.physical_key,
            text: event.text.as_deref(),
        };
        let keystroke = keys::keystroke(input, modifiers);
        if !pressed {
            self.dispatch(PlatformInput::KeyUp(KeyUpEvent { keystroke }));
            return;
        }

        let already_committed = {
            let mut state = self.state.borrow_mut();
            state.key_text = keystroke.key_char.clone();
            keystroke.key_char.is_some() && state.commit_text == keystroke.key_char
        };
        let result = self.dispatch(PlatformInput::KeyDown(KeyDownEvent {
            keystroke: keystroke.clone(),
            is_held: event.repeat,
        }));
        // GPUI's rule: a key the app did not handle types its text, if the system's rules say the
        // modifiers held type characters (Shift; Option on macOS; AltGr).
        let composing = self.state.borrow().composing;
        if result.propagate
            && !composing
            && !already_committed
            && TextRules::CURRENT.types_text(&keystroke.modifiers)
            && let Some(text) = &keystroke.key_char
        {
            self.with_input_handler(|handler| handler.replace_text_in_range(None, text));
        }
    }

    fn ime(&self, ime: Ime) {
        match ime {
            Ime::Enabled => {}
            Ime::Preedit(text, cursor) => {
                if text.is_empty() {
                    // The composition was cleared (before a commit, or cancelled).
                    if std::mem::take(&mut self.state.borrow_mut().composing) {
                        self.with_input_handler(cancel_composition);
                    }
                    return;
                }
                self.state.borrow_mut().composing = true;
                let selected = cursor.map(|(start, end)| utf16_range(&text, start..end));
                let area = self.with_input_handler(|handler| {
                    handler.replace_and_mark_text_in_range(None, &text, selected);
                    let range = handler.selected_text_range(false)?.range;
                    handler.bounds_for_range(range)
                });
                if let Some(Some(bounds)) = area {
                    self.set_ime_area(bounds);
                }
            }
            Ime::Commit(text) => {
                let typed_by_key = {
                    let mut state = self.state.borrow_mut();
                    let was_composing = std::mem::take(&mut state.composing);
                    state.commit_text = Some(text.clone());
                    !was_composing && state.key_text.as_deref() == Some(text.as_str())
                };
                // A plain key some systems also report as a commit was handled (or typed) as a key.
                if !typed_by_key {
                    self.with_input_handler(|handler| handler.replace_text_in_range(None, &text));
                }
            }
            Ime::Disabled => {
                // winit's contract: pending preedit is cleared, not kept as typed text.
                if std::mem::take(&mut self.state.borrow_mut().composing) {
                    self.with_input_handler(cancel_composition);
                }
            }
        }
    }
}

/// Removes the composition's text and ends it (a preedit that was cleared or abandoned types nothing).
pub(crate) fn cancel_composition(handler: &mut dyn PlatformInputHandler) {
    if let Some(marked) = handler.marked_text_range() {
        handler.replace_text_in_range(Some(marked), "");
    }
    handler.unmark_text();
}

/// A byte range of `text` as a UTF-16 range (what the input handler takes).
pub(crate) fn utf16_range(text: &str, bytes: Range<usize>) -> Range<usize> {
    let units = |end: usize| text.get(..end.min(text.len())).map_or(0, |s| s.encode_utf16().count());
    units(bytes.start)..units(bytes.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preedit_cursors_convert_to_utf16() {
        assert_eq!(utf16_range("abc", 1..2), 1..2);
        // "日" is 3 bytes, 1 unit; "😀" 4 bytes, 2 units.
        assert_eq!(utf16_range("日本😀x", 3..6), 1..2);
        assert_eq!(utf16_range("日本😀x", 6..10), 2..4);
        assert_eq!(utf16_range("ab", 5..9), 2..2);
    }

    /// A text field as the app's `TextInput` behaves: `unmark_text` only forgets the marked range.
    #[derive(Default)]
    struct Field {
        text: String,
        selection: Range<usize>,
        marked: Option<Range<usize>>,
    }

    impl PlatformInputHandler for Field {
        fn selected_text_range(&mut self, _: bool) -> Option<crate::UTF16Selection> {
            Some(crate::UTF16Selection { range: self.selection.clone(), reversed: false })
        }
        fn marked_text_range(&mut self) -> Option<Range<usize>> {
            self.marked.clone()
        }
        fn text_for_range(&mut self, range: Range<usize>, _: &mut Option<Range<usize>>) -> Option<String> {
            self.text.get(range).map(str::to_owned)
        }
        fn replace_text_in_range(&mut self, range: Option<Range<usize>>, text: &str) {
            let range = range.or(self.marked.take()).unwrap_or(self.selection.clone());
            self.text.replace_range(range.clone(), text);
            self.selection = range.start + text.len()..range.start + text.len();
            self.marked = None;
        }
        fn replace_and_mark_text_in_range(
            &mut self,
            range: Option<Range<usize>>,
            text: &str,
            _: Option<Range<usize>>,
        ) {
            let range = range.or(self.marked.clone()).unwrap_or(self.selection.clone());
            self.text.replace_range(range.clone(), text);
            self.marked = Some(range.start..range.start + text.len());
            self.selection = range.start + text.len()..range.start + text.len();
        }
        fn unmark_text(&mut self) {
            self.marked = None;
        }
        fn bounds_for_range(&mut self, _: Range<usize>) -> Option<crate::Bounds<crate::Pixels>> {
            None
        }
        fn character_index_for_point(&mut self, _: crate::Point<crate::Pixels>) -> Option<usize> {
            None
        }
    }

    #[test]
    fn a_cancelled_composition_types_nothing() {
        // "ab", then a composition "ni" the IME abandons (`Ime::Disabled` after switching input
        // sources, or an empty `Ime::Preedit`): the field is "ab" again, with nothing marked.
        let mut field = Field { text: "ab".into(), selection: 2..2, marked: None };
        field.replace_and_mark_text_in_range(None, "ni", Some(2..2));
        assert_eq!((field.text.as_str(), field.marked.clone()), ("abni", Some(2..4)));
        cancel_composition(&mut field);
        assert_eq!((field.text.as_str(), field.marked.clone(), field.selection.clone()), ("ab", None, 2..2));
        // With no composition it changes nothing.
        cancel_composition(&mut field);
        assert_eq!((field.text.as_str(), field.selection.clone()), ("ab", 2..2));
    }
}
