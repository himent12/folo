//! Single-line text input with IME support, selection, clipboard and placeholder.
//!
//! Adapted from GPUI's `examples/input.rs`; styled by its parent (font, size and colour come
//! from the surrounding text style). Emits [`InputEvent`]s so owners can react to edits.

use std::ops::Range;

use folo_ui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, ElementId, ElementInputHandler, Entity,
    EntityInputHandler, EventEmitter, FocusHandle, Focusable, GlobalElementId, KeyBinding, LayoutId,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PaintQuad, Pixels, Point, ShapedLine,
    SharedString, Style, TextRun, UTF16Selection, UnderlineStyle, Window, actions, div, fill, point,
    prelude::*, px, relative, size,
};

use crate::theme::{alpha, theme};

actions!(
    text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        Paste,
        Cut,
        Copy,
        WordLeft,
        WordRight,
        DeleteWordLeft,
        DeleteWordRight,
        SelectWordLeft,
        SelectWordRight,
        SelectHome,
        SelectEnd
    ]
);

pub fn bind_keys(cx: &mut App) {
    let ctx = Some("TextInput");
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, ctx),
        KeyBinding::new("delete", Delete, ctx),
        KeyBinding::new("left", Left, ctx),
        KeyBinding::new("right", Right, ctx),
        KeyBinding::new("shift-left", SelectLeft, ctx),
        KeyBinding::new("shift-right", SelectRight, ctx),
        KeyBinding::new("ctrl-left", WordLeft, ctx),
        KeyBinding::new("ctrl-right", WordRight, ctx),
        KeyBinding::new("ctrl-backspace", DeleteWordLeft, ctx),
        KeyBinding::new("ctrl-delete", DeleteWordRight, ctx),
        KeyBinding::new("ctrl-shift-left", SelectWordLeft, ctx),
        KeyBinding::new("ctrl-shift-right", SelectWordRight, ctx),
        KeyBinding::new("shift-home", SelectHome, ctx),
        KeyBinding::new("shift-end", SelectEnd, ctx),
        KeyBinding::new("ctrl-a", SelectAll, ctx),
        KeyBinding::new("ctrl-v", Paste, ctx),
        KeyBinding::new("ctrl-c", Copy, ctx),
        KeyBinding::new("ctrl-x", Cut, ctx),
        KeyBinding::new("home", Home, ctx),
        KeyBinding::new("end", End, ctx),
    ]);
}

#[derive(Clone, Debug)]
pub enum InputEvent {
    Changed(SharedString),
}

pub struct TextInput {
    focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<ShapedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    is_selecting: bool,
    /// Horizontal scroll of the text, so the caret stays visible in long values.
    scroll_x: Pixels,
    /// A secret: shown as one bullet per character, never copied or cut, and moved through as a
    /// single word.
    masked: bool,
}

/// What a masked field shows for each character.
const BULLET: char = '•';

/// The text a masked field shows for `s`: one bullet per character.
fn mask(s: &str) -> String {
    std::iter::repeat_n(BULLET, s.chars().count()).collect()
}

/// A byte offset in `s` (a character boundary) as the same boundary in `mask(s)`.
fn masked_offset(s: &str, offset: usize) -> usize {
    s.get(..offset).map_or(s.chars().count(), |head| head.chars().count()) * BULLET.len_utf8()
}

/// A byte offset in `mask(s)` back in `s`, at the start of the bullet it falls in.
fn unmasked_offset(s: &str, offset: usize) -> usize {
    s.char_indices().nth(offset / BULLET.len_utf8()).map_or(s.len(), |(i, _)| i)
}

impl EventEmitter<InputEvent> for TextInput {}

impl TextInput {
    pub fn new(placeholder: impl Into<SharedString>, cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle().tab_stop(true),
            content: SharedString::default(),
            placeholder: placeholder.into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            is_selecting: false,
            scroll_x: px(0.),
            masked: false,
        }
    }

    /// Makes this a field for a secret (an API key, a token): bullets on screen, no copy or cut,
    /// and word moves go to its ends.
    pub fn masked(mut self) -> Self {
        self.masked = true;
        self
    }

    pub fn text(&self) -> SharedString {
        self.content.clone()
    }

    #[cfg(test)]
    pub(crate) fn last_bounds(&self) -> Option<Bounds<Pixels>> {
        self.last_bounds
    }

    /// An offset in the content as an offset in the text on screen (the bullets of a masked field).
    fn shown(&self, offset: usize) -> usize {
        if self.masked { masked_offset(&self.content, offset) } else { offset }
    }

    /// An offset in the text on screen back in the content.
    fn unshown(&self, offset: usize) -> usize {
        if self.masked { unmasked_offset(&self.content, offset) } else { offset }
    }

    /// Replace the content, select all of it (so typing replaces it) and notify.
    pub fn set_text(&mut self, text: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.content = text.into();
        self.selected_range = 0..self.content.len();
        self.selection_reversed = false;
        self.marked_range = None;
        cx.emit(InputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
    }

    fn word_left(&mut self, _: &WordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.previous_word(self.cursor_offset()), cx);
    }

    fn word_right(&mut self, _: &WordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.next_word(self.cursor_offset()), cx);
    }

    fn delete_word_left(&mut self, _: &DeleteWordLeft, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_word(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_word_right(&mut self, _: &DeleteWordRight, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_word(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn select_word_left(&mut self, _: &SelectWordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_word(self.cursor_offset()), cx);
    }

    fn select_word_right(&mut self, _: &SelectWordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_word(self.cursor_offset()), cx);
    }

    fn select_home(&mut self, _: &SelectHome, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(0, cx);
    }

    fn select_end(&mut self, _: &SelectEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.content.len(), cx);
    }

    /// The selected byte range (tests read it).
    #[cfg(test)]
    pub(crate) fn selection(&self) -> Range<usize> {
        self.selected_range.clone()
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx)
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        if self.content.is_empty() {
            cx.propagate();
            return;
        }
        self.move_to(0, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        if self.content.is_empty() {
            cx.propagate();
            return;
        }
        self.move_to(self.content.len(), cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        window.focus(&self.focus_handle);
        if event.modifiers.shift {
            self.is_selecting = true;
            self.select_to(self.index_for_mouse_position(event.position), cx);
        } else if event.click_count >= 2 {
            self.is_selecting = false;
            self.move_to(0, cx);
            self.select_to(self.content.len(), cx);
        } else {
            self.is_selecting = true;
            self.move_to(self.index_for_mouse_position(event.position), cx)
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            if event.pressed_button == Some(MouseButton::Left) {
                self.select_to(self.index_for_mouse_position(event.position), cx);
            } else {
                self.is_selecting = false;
            }
        }
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            self.replace_text_in_range(None, &one_line(&text), window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        // A secret never reaches the clipboard.
        if !self.selected_range.is_empty() && !self.masked {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() && !self.masked {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx)
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify()
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed { self.selected_range.start } else { self.selected_range.end }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        if self.content.is_empty() {
            return 0;
        }
        let (Some(bounds), Some(line)) = (self.last_bounds.as_ref(), self.last_layout.as_ref()) else {
            return 0;
        };
        if position.y < bounds.top() {
            return 0;
        }
        if position.y > bounds.bottom() {
            return self.content.len();
        }
        self.unshown(line.closest_index_for_x(position.x - bounds.left() + self.scroll_x))
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset
        } else {
            self.selected_range.end = offset
        };
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify()
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        utf16_to_utf8(&self.content, offset)
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;
        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }
        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range_utf16: &Range<usize>) -> Range<usize> {
        let a = self.offset_from_utf16(range_utf16.start);
        let b = self.offset_from_utf16(range_utf16.end);
        a.min(b)..a.max(b)
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        self.content[..offset].char_indices().next_back().map_or(0, |(i, _)| i)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        self.content[offset..].chars().next().map_or(self.content.len(), |c| offset + c.len_utf8())
    }

    fn previous_word(&self, offset: usize) -> usize {
        // A secret has no words to show: it is one.
        if self.masked {
            return 0;
        }
        let s = &self.content[..offset];
        let trimmed = s.trim_end_matches(|c: char| !c.is_alphanumeric());
        trimmed
            .rfind(|c: char| !c.is_alphanumeric())
            .map_or(0, |i| i + trimmed[i..].chars().next().map_or(1, |c| c.len_utf8()))
    }

    fn next_word(&self, offset: usize) -> usize {
        if self.masked {
            return self.content.len();
        }
        let s = &self.content[offset..];
        let skip = s.len() - s.trim_start_matches(|c: char| !c.is_alphanumeric()).len();
        let rest = &s[skip..];
        offset + skip + rest.find(|c: char| !c.is_alphanumeric()).unwrap_or(rest.len())
    }

    fn changed(&mut self, cx: &mut Context<Self>) {
        cx.emit(InputEvent::Changed(self.content.clone()));
        cx.notify();
    }
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        // The platform's text services see a secret as its bullets.
        let text = &self.content[range];
        Some(if self.masked { mask(text) } else { text.to_string() })
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range.as_ref().map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|r| self.range_from_utf16(r))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let clean = one_line(new_text);
        if range.is_empty() && clean.is_empty() && self.marked_range.is_none() {
            return;
        }
        self.content = (self.content[0..range.start].to_owned() + &clean + &self.content[range.end..]).into();
        self.selected_range = range.start + clean.len()..range.start + clean.len();
        self.selection_reversed = false;
        self.marked_range.take();
        self.changed(cx);
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|r| self.range_from_utf16(r))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let clean = one_line(new_text);
        self.content = (self.content[0..range.start].to_owned() + &clean + &self.content[range.end..]).into();
        self.marked_range = (!clean.is_empty()).then(|| range.start..range.start + clean.len());
        // The IME's selection is relative to the raw composed text; map UTF-16 offsets through
        // `one_line` sanitization so collapsed `\r\n` or dropped control chars keep the caret right.
        self.selected_range = new_selected_range_utf16
            .as_ref()
            .map(|r| {
                let a = one_line_utf16_to_utf8(new_text, r.start);
                let b = one_line_utf16_to_utf8(new_text, r.end);
                range.start + a.min(b)..range.start + a.max(b)
            })
            .unwrap_or_else(|| range.start + clean.len()..range.start + clean.len());
        self.selection_reversed = false;
        self.changed(cx);
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        bounds: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let last_layout = self.last_layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        let x = |o: usize| bounds.left() + last_layout.x_for_index(self.shown(o)) - self.scroll_x;
        let (x1, x2) = (x(range.start), x(range.end));
        Some(Bounds::from_corners(point(x1.min(x2), bounds.top()), point(x1.max(x2), bounds.bottom())))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        let line_point = self.last_bounds?.localize(&point)?;
        if self.content.is_empty() {
            return Some(0);
        }
        let last_layout = self.last_layout.as_ref()?;
        let utf8_index = self.unshown(last_layout.index_for_x(line_point.x + self.scroll_x)?);
        Some(self.offset_to_utf16(utf8_index))
    }
}

struct TextElement {
    input: Entity<TextInput>,
}

struct PrepaintState {
    line: Option<ShapedLine>,
    cursor: Option<PaintQuad>,
    selection: Vec<PaintQuad>,
    scroll_x: Pixels,
    /// Placeholder text starts just right of the caret.
    text_dx: Pixels,
}

/// Keep the caret at least `margin` inside a field of `width`, scrolling no further than the text.
fn follow_caret(scroll: Pixels, caret: Pixels, text_w: Pixels, width: Pixels, margin: Pixels) -> Pixels {
    let mut s = scroll;
    if caret - s > width - margin {
        s = caret - width + margin;
    }
    if caret - s < px(0.) {
        s = caret;
    }
    // Don't leave empty space after the text once it fits again.
    let max = (text_w - width + margin).max(px(0.));
    s.min(max).max(px(0.))
}

impl IntoElement for TextElement {
    type Element = Self;
    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextElement {
    type RequestLayoutState = ();
    type PrepaintState = PrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&folo_ui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = window.line_height().into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&folo_ui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> PrepaintState {
        let t = theme(cx);
        let input = self.input.read(cx);
        let content = input.content.clone();
        // Offsets below are on screen: a masked field's bullets are longer than its characters.
        let selected_range = input.shown(input.selected_range.start)..input.shown(input.selected_range.end);
        let cursor = input.shown(input.cursor_offset());
        let marked_range = input.marked_range.as_ref().map(|r| input.shown(r.start)..input.shown(r.end));
        let style = window.text_style();
        let (display_text, text_color) = if content.is_empty() {
            (input.placeholder.clone(), t.text_tertiary)
        } else if input.masked {
            (mask(&content).into(), style.color)
        } else {
            (content, style.color)
        };
        let run = TextRun {
            len: display_text.len(),
            font: style.font(),
            color: text_color,
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let runs = if let Some(marked) = marked_range.as_ref() {
            vec![
                TextRun { len: marked.start, ..run.clone() },
                TextRun {
                    len: marked.end - marked.start,
                    underline: Some(UnderlineStyle {
                        color: Some(run.color),
                        thickness: px(1.0),
                        wavy: false,
                    }),
                    ..run.clone()
                },
                TextRun { len: display_text.len() - marked.end, ..run },
            ]
            .into_iter()
            .filter(|r| r.len > 0)
            .collect()
        } else {
            vec![run]
        };
        let font_size = style.font_size.to_pixels(window.rem_size());
        let line = window.text_system().shape_line(display_text, font_size, &runs, None);
        let empty = input.content.is_empty();
        let caret_x = if empty { px(0.) } else { line.x_for_index(cursor) };
        let scroll_x = if empty {
            px(0.)
        } else {
            follow_caret(input.scroll_x, caret_x, line.width, bounds.size.width, px(8.))
        };
        let cursor_pos = caret_x - scroll_x;
        let line_h = window.line_height();
        let caret_h = (font_size * 1.25).min(line_h);
        let caret_top = bounds.top() + (line_h - caret_h) / 2.;
        // Selection only shows while the field has focus.
        let focused = input.focus_handle.is_focused(window);
        let (selection, cursor) = if selected_range.is_empty() || empty || !focused {
            (
                Vec::new(),
                Some(fill(
                    Bounds::new(point(bounds.left() + cursor_pos, caret_top), size(px(1.5), caret_h)),
                    t.accent_fill(),
                )),
            )
        } else {
            let quads = line
                .selection_spans(selected_range)
                .into_iter()
                .map(|(x1, x2)| {
                    fill(
                        Bounds::from_corners(
                            point(bounds.left() + x1.min(x2) - scroll_x, caret_top),
                            point(bounds.left() + x1.max(x2) - scroll_x, caret_top + caret_h),
                        ),
                        alpha(t.accent_fill(), 0.28),
                    )
                })
                .collect();
            (quads, None)
        };
        let text_dx = if empty { px(4.) } else { px(0.) };
        PrepaintState { line: Some(line), cursor, selection, scroll_x, text_dx }
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&folo_ui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        prepaint: &mut PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.input.read(cx).focus_handle.clone();
        window.handle_input(&focus_handle, ElementInputHandler::new(bounds, self.input.clone()), cx);
        let line = prepaint.line.take().unwrap();
        let scroll_x = prepaint.scroll_x;
        let selection = std::mem::take(&mut prepaint.selection);
        let cursor = focus_handle.is_focused(window).then(|| prepaint.cursor.take()).flatten();
        window.with_content_mask(Some(folo_ui::ContentMask { bounds }), |window| {
            for quad in selection {
                window.paint_quad(quad);
            }
            line.paint(
                point(bounds.origin.x - scroll_x + prepaint.text_dx, bounds.origin.y),
                window.line_height(),
                window,
                cx,
            )
            .unwrap();
            if let Some(cursor) = cursor {
                window.paint_quad(cursor);
            }
        });
        self.input.update(cx, |input, _| {
            input.last_layout = Some(line);
            input.last_bounds = Some(bounds);
            input.scroll_x = scroll_x;
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .key_context("TextInput")
            .track_focus(&self.focus_handle(cx))
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::word_left))
            .on_action(cx.listener(Self::word_right))
            .on_action(cx.listener(Self::delete_word_left))
            .on_action(cx.listener(Self::delete_word_right))
            .on_action(cx.listener(Self::select_word_left))
            .on_action(cx.listener(Self::select_word_right))
            .on_action(cx.listener(Self::select_home))
            .on_action(cx.listener(Self::select_end))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .child(TextElement { input: cx.entity() })
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

/// Pasted text for a single-line field: each line break (CRLF, CR or LF) and tab a space, other
/// control characters dropped. (Only LF was replaced: a Windows clipboard's CRLF left a CR in the
/// value, invisible on screen and sent along with it.)
fn one_line(s: &str) -> String {
    s.replace("\r\n", " ")
        .chars()
        .filter_map(|c| match c {
            '\r' | '\n' | '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

/// Byte offset in `one_line(raw)` corresponding to UTF-16 offset `offset` in `raw`.
fn one_line_utf16_to_utf8(raw: &str, offset: usize) -> usize {
    let mut raw_utf16 = 0;
    let mut clean_utf8 = 0;
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if raw_utf16 >= offset {
            break;
        }
        if ch == '\r' && chars.peek() == Some(&'\n') {
            chars.next();
            raw_utf16 += 2;
            clean_utf8 += 1;
        } else if matches!(ch, '\r' | '\n' | '\t') {
            raw_utf16 += 1;
            clean_utf8 += 1;
        } else if ch.is_control() {
            raw_utf16 += ch.len_utf16();
        } else {
            raw_utf16 += ch.len_utf16();
            clean_utf8 += ch.len_utf8();
        }
    }
    clean_utf8
}

/// Byte offset in `s` of UTF-16 offset `offset`, clamped to the end and to char boundaries.
fn utf16_to_utf8(s: &str, offset: usize) -> usize {
    let mut utf8 = 0;
    let mut utf16 = 0;
    for ch in s.chars() {
        if utf16 >= offset {
            break;
        }
        utf16 += ch.len_utf16();
        utf8 += ch.len_utf8();
    }
    utf8
}

#[cfg(test)]
mod tests {
    use super::{mask, masked_offset, one_line, unmasked_offset, utf16_to_utf8};

    #[test]
    fn pasted_text_becomes_one_line_without_control_characters() {
        assert_eq!(one_line("http://127.0.0.1:18101\r\n"), "http://127.0.0.1:18101 ");
        assert_eq!(one_line("a\r\nb\nc\rd\te"), "a b c d e");
        assert_eq!(one_line("x\u{0}\u{7f}\u{1b}y"), "xy");
        assert_eq!(one_line("名前 🐟"), "名前 🐟");
    }

    #[test]
    fn a_masked_field_maps_offsets_between_text_and_bullets() {
        let s = "ab中d😀";
        assert_eq!(mask(s), "•••••");
        assert_eq!(mask(""), "");
        // Every character boundary maps to a bullet boundary and back.
        let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).chain([s.len()]).collect();
        for (n, &b) in bounds.iter().enumerate() {
            let shown = masked_offset(s, b);
            assert_eq!(shown, n * '•'.len_utf8(), "boundary {b}");
            assert!(mask(s).is_char_boundary(shown));
            assert_eq!(unmasked_offset(s, shown), b);
        }
        // Inside a bullet (a pointer between its bytes): its start.
        assert_eq!(unmasked_offset(s, 7), 2);
        // Past the end: the end.
        assert_eq!(unmasked_offset(s, 99), s.len());
    }

    #[test]
    fn caret_stays_in_view() {
        use super::follow_caret;
        use folo_ui::px;
        // Typing past the right edge scrolls so the caret sits 8 px inside.
        assert_eq!(follow_caret(px(0.), px(300.), px(300.), px(200.), px(8.)), px(108.));
        // Moving home scrolls back to the start.
        assert_eq!(follow_caret(px(108.), px(0.), px(300.), px(200.), px(8.)), px(0.));
        // Short text never scrolls.
        assert_eq!(follow_caret(px(40.), px(50.), px(60.), px(200.), px(8.)), px(0.));
    }

    #[test]
    fn utf16_offsets_land_on_char_boundaries() {
        let s = "ab中d";
        assert_eq!(utf16_to_utf8(s, 0), 0);
        assert_eq!(utf16_to_utf8(s, 2), 2);
        assert_eq!(utf16_to_utf8(s, 3), 5);
        assert_eq!(utf16_to_utf8(s, 99), s.len());
        // An IME selection of 1..1 inside "中" composed after "ab" maps to the end of "中".
        let composed = "中";
        let start = 2 + utf16_to_utf8(composed, 1);
        assert!("ab中".is_char_boundary(start));
        assert_eq!(start, 5);
        // Surrogate pairs count as two UTF-16 units.
        assert_eq!(utf16_to_utf8("😀x", 2), 4);
    }
}
