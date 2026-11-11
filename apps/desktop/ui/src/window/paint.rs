// Adapted from GPUI (Apache-2.0): crates/gpui/src/window.rs (`PaintQuad`, `quad`, `fill`, `outline`)
//! `PaintQuad` and the `quad` / `fill` / `outline` helpers. The `Window` paint methods that take them are in
//! `primitives.rs`. Owner tag: `"window"`.

use crate::{Background, BorderStyle, Bounds, Corners, Edges, Hsla, Pixels, transparent_black};

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
