//! Text input for custom elements: the `InputHandler` trait the window calls, `EntityInputHandler` (what
//! `src/input.rs` implements on its view) and `ElementInputHandler` (adapts the one to the other).
//! Owner tag: `"elements"`. Ranges are UTF-16 code units, as GPUI's.

use std::ops::Range;

use crate::{App, Bounds, Context, Entity, Pixels, Point, UTF16Selection, Window};

/// The window-side text input protocol (GPUI's `InputHandler`): `PlatformInputHandler` with the window and
/// the app passed in.
pub trait InputHandler: 'static {
    /// The selection (`None`: no editable text).
    fn selected_text_range(
        &mut self,
        ignore_disabled_input: bool,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UTF16Selection>;
    /// The IME composition range.
    fn marked_text_range(&mut self, window: &mut Window, cx: &mut App) -> Option<Range<usize>>;
    /// The text in `range_utf16`.
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<String>;
    /// Replaces a range (or the marked text, or the selection) with `text`.
    fn replace_text_in_range(
        &mut self,
        replacement_range: Option<Range<usize>>,
        text: &str,
        window: &mut Window,
        cx: &mut App,
    );
    /// Replaces a range and marks it as the IME composition.
    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    );
    /// Ends the composition.
    fn unmark_text(&mut self, window: &mut Window, cx: &mut App);
    /// Window bounds of the text in `range_utf16`.
    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Bounds<Pixels>>;
    /// The character index at `point`.
    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<usize>;
}

/// A view that edits text (GPUI 0.2.2's trait, as `src/input.rs` implements it).
pub trait EntityInputHandler: 'static + Sized {
    /// The text in `range`.
    fn text_for_range(
        &mut self,
        range: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<String>;

    /// The selection.
    fn selected_text_range(
        &mut self,
        ignore_disabled_input: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<UTF16Selection>;

    /// The IME composition range.
    fn marked_text_range(&self, window: &mut Window, cx: &mut Context<Self>) -> Option<Range<usize>>;

    /// Ends the composition.
    fn unmark_text(&mut self, window: &mut Window, cx: &mut Context<Self>);

    /// Replaces a range with `text`.
    fn replace_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        text: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    );

    /// Replaces a range and marks it.
    fn replace_and_mark_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    );

    /// Window bounds of the text in `range_utf16`, given the element's bounds.
    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        element_bounds: Bounds<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>>;

    /// The character index at `point`.
    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<usize>;
}

/// The `InputHandler` an element paints for its view (`window.handle_input(&fh, ElementInputHandler::new(..))`).
pub struct ElementInputHandler<V> {
    /// The view.
    view: Entity<V>,
    /// The element's bounds this frame.
    element_bounds: Bounds<Pixels>,
}

impl<V: 'static> ElementInputHandler<V> {
    /// A handler for `view`, painted at `element_bounds`.
    pub fn new(element_bounds: Bounds<Pixels>, view: Entity<V>) -> Self {
        ElementInputHandler { view, element_bounds }
    }
}

impl<V: EntityInputHandler> InputHandler for ElementInputHandler<V> {
    fn selected_text_range(
        &mut self,
        ignore_disabled_input: bool,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<UTF16Selection> {
        self.view.update(cx, |view, cx| view.selected_text_range(ignore_disabled_input, window, cx))
    }

    fn marked_text_range(&mut self, window: &mut Window, cx: &mut App) -> Option<Range<usize>> {
        self.view.update(cx, |view, cx| view.marked_text_range(window, cx))
    }

    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<String> {
        self.view.update(cx, |view, cx| view.text_for_range(range_utf16, adjusted_range, window, cx))
    }

    fn replace_text_in_range(
        &mut self,
        replacement_range: Option<Range<usize>>,
        text: &str,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.view.update(cx, |view, cx| view.replace_text_in_range(replacement_range, text, window, cx))
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.view.update(cx, |view, cx| {
            view.replace_and_mark_text_in_range(range_utf16, new_text, new_selected_range, window, cx)
        })
    }

    fn unmark_text(&mut self, window: &mut Window, cx: &mut App) {
        self.view.update(cx, |view, cx| view.unmark_text(window, cx))
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Bounds<Pixels>> {
        self.view.update(cx, |view, cx| view.bounds_for_range(range_utf16, self.element_bounds, window, cx))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<usize> {
        self.view.update(cx, |view, cx| view.character_index_for_point(point, window, cx))
    }
}
