//! The element-author paint API on `Window` (GPUI's signatures): quads, paths, images, SVG icons, glyphs,
//! emoji, shadows, underlines, plus `PaintQuad`, `ContentMask` and the `quad` / `fill` / `outline` helpers.
//! Owner tag: `"window"`.
//!
//! Every paint call scales to device pixels, intersects with the current content mask and pushes one
//! primitive onto `next_frame.scene` (painter's order). Element opacity multiplies colour alpha here.
//! Glyph and icon masks come from the window's `MaskCache`s (keyed by glyph params / SVG path and size), so
//! steady frames rasterize nothing; images are shared as `Arc<RenderImage>`.
#![allow(dead_code)] // skeleton

use std::sync::Arc;

use crate::{
    App, Background, BorderStyle, Bounds, BoxShadow, Corners, Edges, FontId, GlyphId, Hsla, Pixels, Point,
    RenderImage, SharedString, StrikethroughStyle, UnderlineStyle, Window, transparent_black,
};

/// A rectangular clip: what may be painted (window logical px).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ContentMask<P: Clone + Default + std::fmt::Debug + PartialEq> {
    /// The visible region.
    pub bounds: Bounds<P>,
}

/// A quad to paint: bounds, corner radii, background, borders.
#[derive(Clone, Debug)]
pub struct PaintQuad {
    /// The bounds (window logical px).
    pub bounds: Bounds<Pixels>,
    /// Corner radii.
    pub corner_radii: Corners<Pixels>,
    /// The fill.
    pub background: Background,
    /// Border widths.
    pub border_widths: Edges<Pixels>,
    /// Border colour.
    pub border_color: Hsla,
    /// Solid or dashed.
    pub border_style: BorderStyle,
}

impl PaintQuad {
    /// Sets the corner radii.
    pub fn corner_radii(self, corner_radii: impl Into<Corners<Pixels>>) -> Self {
        PaintQuad { corner_radii: corner_radii.into(), ..self }
    }

    /// Sets the border widths.
    pub fn border_widths(self, border_widths: impl Into<Edges<Pixels>>) -> Self {
        PaintQuad { border_widths: border_widths.into(), ..self }
    }

    /// Sets the border colour.
    pub fn border_color(self, border_color: impl Into<Hsla>) -> Self {
        PaintQuad { border_color: border_color.into(), ..self }
    }

    /// Sets the background.
    pub fn background(self, background: impl Into<Background>) -> Self {
        PaintQuad { background: background.into(), ..self }
    }
}

/// A quad with everything given.
pub fn quad(
    bounds: Bounds<Pixels>,
    corner_radii: impl Into<Corners<Pixels>>,
    background: impl Into<Background>,
    border_widths: impl Into<Edges<Pixels>>,
    border_color: impl Into<Hsla>,
    border_style: BorderStyle,
) -> PaintQuad {
    PaintQuad {
        bounds,
        corner_radii: corner_radii.into(),
        background: background.into(),
        border_widths: border_widths.into(),
        border_color: border_color.into(),
        border_style,
    }
}

/// A filled rectangle.
pub fn fill(bounds: impl Into<Bounds<Pixels>>, background: impl Into<Background>) -> PaintQuad {
    PaintQuad {
        bounds: bounds.into(),
        corner_radii: (0.).into(),
        background: background.into(),
        border_widths: (0.).into(),
        border_color: transparent_black(),
        border_style: BorderStyle::default(),
    }
}

/// A 1 px outline.
pub fn outline(
    bounds: impl Into<Bounds<Pixels>>,
    border_color: impl Into<Hsla>,
    border_style: BorderStyle,
) -> PaintQuad {
    PaintQuad {
        bounds: bounds.into(),
        corner_radii: (0.).into(),
        background: transparent_black().into(),
        border_widths: (1.).into(),
        border_color: border_color.into(),
        border_style,
    }
}

impl Window {
    /// Paints a quad.
    pub fn paint_quad(&mut self, _quad: PaintQuad) {
        todo!("window")
    }

    /// Paints a path built by `PathBuilder` (fill or stroke) with a colour or gradient.
    pub fn paint_path(&mut self, _path: crate::BuiltPath, _color: impl Into<Background>) {
        todo!("window")
    }

    /// Paints frame `frame_index` of an image.
    pub fn paint_image(
        &mut self,
        _bounds: Bounds<Pixels>,
        _corner_radii: Corners<Pixels>,
        _data: Arc<RenderImage>,
        _frame_index: usize,
        _grayscale: bool,
    ) -> anyhow::Result<()> {
        todo!("window")
    }

    /// Paints an SVG asset as a single-colour icon (mask from the icon cache).
    pub fn paint_svg(
        &mut self,
        _bounds: Bounds<Pixels>,
        _path: SharedString,
        _color: Hsla,
        _cx: &App,
    ) -> anyhow::Result<()> {
        todo!("window")
    }

    /// Paints one glyph (mask from the glyph cache; the renderer applies the text backend's blend).
    pub fn paint_glyph(
        &mut self,
        _origin: Point<Pixels>,
        _font_id: FontId,
        _glyph_id: GlyphId,
        _font_size: Pixels,
        _color: Hsla,
    ) -> anyhow::Result<()> {
        todo!("window")
    }

    /// Paints a colour emoji glyph.
    pub fn paint_emoji(
        &mut self,
        _origin: Point<Pixels>,
        _font_id: FontId,
        _glyph_id: GlyphId,
        _font_size: Pixels,
    ) -> anyhow::Result<()> {
        todo!("window")
    }

    /// Paints box shadows under a rounded rectangle.
    pub fn paint_shadows(
        &mut self,
        _bounds: Bounds<Pixels>,
        _corner_radii: Corners<Pixels>,
        _shadows: &[BoxShadow],
    ) {
        todo!("window")
    }

    /// Paints an underline (straight or wavy).
    pub fn paint_underline(&mut self, _origin: Point<Pixels>, _width: Pixels, _style: &UnderlineStyle) {
        todo!("window")
    }

    /// Paints a strikethrough.
    pub fn paint_strikethrough(
        &mut self,
        _origin: Point<Pixels>,
        _width: Pixels,
        _style: &StrikethroughStyle,
    ) {
        todo!("window")
    }

    /// Runs `f` as one layer. GPUI batches primitives by layer for the GPU; the CPU renderer has nothing to
    /// batch, so this only runs `f`.
    pub fn paint_layer<R>(&mut self, _bounds: Bounds<Pixels>, f: impl FnOnce(&mut Self) -> R) -> R {
        f(self)
    }
}
