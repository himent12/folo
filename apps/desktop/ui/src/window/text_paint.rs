//! Text painting through the window: `impl GlyphPainter for Window` (the text package's seam) and GPUI's
//! `ShapedLine::paint` / `WrappedLine::paint` signatures (window and app passed in), which hand the line
//! to the text package's painters with the window as the painter. Owner tag: `"window"`.

use anyhow::Result;

use crate::{
    App, Bounds, FontId, GlyphId, GlyphPainter, Hsla, Pixels, Point, ShapedLine, StrikethroughStyle,
    TextAlign, UnderlineStyle, Window, WrappedLine, fill,
};

impl GlyphPainter for Window {
    fn paint_glyph(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: Hsla,
    ) -> Result<()> {
        Window::paint_glyph(self, origin, font_id, glyph_id, font_size, color)
    }

    fn paint_emoji(
        &mut self,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
    ) -> Result<()> {
        Window::paint_emoji(self, origin, font_id, glyph_id, font_size)
    }

    fn paint_underline(&mut self, origin: Point<Pixels>, width: Pixels, style: &UnderlineStyle) {
        Window::paint_underline(self, origin, width, style)
    }

    fn paint_strikethrough(&mut self, origin: Point<Pixels>, width: Pixels, style: &StrikethroughStyle) {
        Window::paint_strikethrough(self, origin, width, style)
    }

    fn paint_background(&mut self, bounds: Bounds<Pixels>, color: Hsla) {
        self.paint_quad(fill(bounds, color))
    }
}

impl ShapedLine {
    /// Paints the line with its top-left at `origin` (GPUI's `ShapedLine::paint`).
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        let content_mask = window.content_mask().bounds;
        self.paint_with(origin, line_height, cx.text_system(), content_mask, window)
    }

    /// Paints the line's run backgrounds (GPUI's `ShapedLine::paint_background`).
    pub fn paint_background(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        self.paint_background_with(origin, line_height, cx.text_system(), window)
    }
}

impl WrappedLine {
    /// Paints the wrapped line, each visual line aligned within `bounds` (GPUI's `WrappedLine::paint`).
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        let content_mask = window.content_mask().bounds;
        self.paint_with(origin, line_height, align, bounds, cx.text_system(), content_mask, window)
    }

    /// Paints the run backgrounds of the wrapped line (GPUI's `WrappedLine::paint_background`).
    pub fn paint_background(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        self.paint_background_with(origin, line_height, align, bounds, cx.text_system(), window)
    }
}
