//! Text as elements: `&'static str`, `String`, `SharedString`, and `StyledText` (text with runs). Owner tag:
//! `"elements"`. Shaping, wrapping and truncation are the text package's (`WindowTextSystem::shape_text`,
//! `LineWrapper`); this element measures through `request_measured_layout`, keeps the shaped lines from
//! prepaint to paint, and paints them through the window's `GlyphPainter`.
//!
//! Performance: text is `SharedString` (never copied per frame); shaped layouts come from the window's line
//! layout cache, which keeps what the previous frame used.
#![allow(dead_code)] // skeleton

use std::{cell::RefCell, rc::Rc};

use crate::{SharedString, TextRun, elements::skeleton_element};

/// Text with explicit runs (fonts, colours, underlines per byte range).
pub struct StyledText {
    /// The text.
    pub(crate) text: SharedString,
    /// The runs, if given (else one run in the inherited text style).
    pub(crate) runs: Option<Vec<TextRun>>,
    /// The layout measured in `request_layout`, shared with prepaint / paint.
    pub(crate) layout: TextLayout,
}

impl StyledText {
    /// Text in the inherited style.
    pub fn new(text: impl Into<SharedString>) -> Self {
        StyledText { text: text.into(), runs: None, layout: TextLayout::default() }
    }

    /// Sets the runs (their lengths must sum to the text's byte length).
    pub fn with_runs(mut self, runs: Vec<TextRun>) -> Self {
        self.runs = Some(runs);
        self
    }
}

/// The shaped, wrapped lines of a text element this frame, shared between its phases.
#[derive(Clone, Default)]
pub struct TextLayout(pub(crate) Rc<RefCell<Option<TextLayoutInner>>>);

/// What `TextLayout` holds once measured.
pub(crate) struct TextLayoutInner {
    /// The wrapped lines (the text package's type).
    pub(crate) lines: smallvec::SmallVec<[crate::WrappedLine; 1]>,
    /// The line height used.
    pub(crate) line_height: crate::Pixels,
    /// The element's bounds once prepainted.
    pub(crate) bounds: Option<crate::Bounds<crate::Pixels>>,
}

skeleton_element!([] StyledText, (), (), "elements");
skeleton_element!([] SharedString, TextLayout, (), "elements");
skeleton_element!([] &'static str, TextLayout, (), "elements");

impl crate::IntoElement for String {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}
