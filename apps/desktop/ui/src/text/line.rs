// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system/line.rs, crates/gpui/src/style.rs (TextAlign)
//! Shaped lines ready to paint: `ShapedLine` (one line), `WrappedLine` (one line with wrap boundaries),
//! their decoration runs, and `TextAlign`.

use crate::{
    Bounds, GlyphPainter, Hsla, LineLayout, Pixels, Point, SharedString, StrikethroughStyle, TextSystem,
    UnderlineStyle, WrappedLineLayout, paint_line, paint_line_background,
};
use anyhow::Result;
use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// How text is aligned horizontally inside its box.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextAlign {
    /// Align to the left edge.
    #[default]
    Left,
    /// Center the text.
    Center,
    /// Align to the right edge.
    Right,
}

/// The decoration (colours, underline, strikethrough) of `len` bytes of a line.
#[derive(Debug, Clone)]
pub struct DecorationRun {
    /// The length of the run in utf-8 bytes.
    pub len: u32,
    /// The color for this run.
    pub color: Hsla,
    /// The background color for this run.
    pub background_color: Option<Hsla>,
    /// The underline style for this run.
    pub underline: Option<UnderlineStyle>,
    /// The strikethrough style for this run.
    pub strikethrough: Option<StrikethroughStyle>,
}

/// A line of text that has been shaped and decorated (derefs to its `LineLayout`).
#[derive(Clone, Default, Debug)]
pub struct ShapedLine {
    pub(crate) layout: Arc<LineLayout>,
    /// The text that was shaped for this line.
    pub text: SharedString,
    pub(crate) decoration_runs: SmallVec<[DecorationRun; 2]>,
}

impl Deref for ShapedLine {
    type Target = Arc<LineLayout>;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl DerefMut for ShapedLine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.layout
    }
}

impl ShapedLine {
    /// The length of the line in utf-8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.layout.len
    }

    /// Overrides the length (to render text a as text b, e.g. invisibles).
    pub fn with_len(mut self, len: usize) -> Self {
        let layout = self.layout.as_ref();
        self.layout = Arc::new(LineLayout {
            font_size: layout.font_size,
            width: layout.width,
            ascent: layout.ascent,
            descent: layout.descent,
            runs: layout.runs.clone(),
            len,
        });
        self
    }

    /// The decoration runs of the line.
    pub fn decoration_runs(&self) -> &[DecorationRun] {
        &self.decoration_runs
    }

    /// Paints the line's glyphs and decorations, left-aligned, with its top-left at `origin`
    /// (GPUI's `ShapedLine::paint` is this with the window as `painter`).
    pub fn paint_with(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        text_system: &TextSystem,
        content_mask: Bounds<Pixels>,
        painter: &mut dyn GlyphPainter,
    ) -> Result<()> {
        paint_line(
            origin,
            &self.layout,
            line_height,
            TextAlign::default(),
            None,
            &self.decoration_runs,
            &[],
            text_system,
            content_mask,
            painter,
        )
    }

    /// Paints the line's background colours (GPUI's `ShapedLine::paint_background`).
    pub fn paint_background_with(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        text_system: &TextSystem,
        painter: &mut dyn GlyphPainter,
    ) -> Result<()> {
        paint_line_background(
            origin,
            &self.layout,
            line_height,
            TextAlign::default(),
            None,
            &self.decoration_runs,
            &[],
            text_system,
            painter,
        )
    }
}

/// A line of text that has been shaped, decorated and wrapped (derefs to its `WrappedLineLayout`).
#[derive(Clone, Default, Debug)]
pub struct WrappedLine {
    pub(crate) layout: Arc<WrappedLineLayout>,
    /// The text that was shaped for this line.
    pub text: SharedString,
    pub(crate) decoration_runs: SmallVec<[DecorationRun; 2]>,
}

impl Deref for WrappedLine {
    type Target = Arc<WrappedLineLayout>;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl DerefMut for WrappedLine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.layout
    }
}

impl WrappedLine {
    /// The length of the underlying, unwrapped layout, in utf-8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// The decoration runs of the line.
    pub fn decoration_runs(&self) -> &[DecorationRun] {
        &self.decoration_runs
    }

    /// Paints the wrapped line with its top-left at `origin`, aligned within `bounds`' width (or the
    /// wrap width) (GPUI's `WrappedLine::paint` is this with the window as `painter`).
    #[allow(clippy::too_many_arguments)]
    pub fn paint_with(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        text_system: &TextSystem,
        content_mask: Bounds<Pixels>,
        painter: &mut dyn GlyphPainter,
    ) -> Result<()> {
        let align_width = match bounds {
            Some(bounds) => Some(bounds.size.width),
            None => self.layout.wrap_width,
        };
        paint_line(
            origin,
            &self.layout.unwrapped_layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &self.wrap_boundaries,
            text_system,
            content_mask,
            painter,
        )
    }

    /// Paints the wrapped line's background colours (GPUI's `WrappedLine::paint_background`).
    pub fn paint_background_with(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        text_system: &TextSystem,
        painter: &mut dyn GlyphPainter,
    ) -> Result<()> {
        let align_width = match bounds {
            Some(bounds) => Some(bounds.size.width),
            None => self.layout.wrap_width,
        };
        paint_line_background(
            origin,
            &self.layout.unwrapped_layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &self.wrap_boundaries,
            text_system,
            painter,
        )
    }
}
