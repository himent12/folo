// Adapted from GPUI (Apache-2.0): crates/gpui/src/style.rs (Style::paint, overflow_mask), crates/gpui/src/window.rs (paint_quad, paint_shadows, ContentMask)
//! Painting a `Style` as pure functions into a [`Scene`]: what GPUI's `Style::paint`, `Window::paint_quad`
//! and `Window::paint_shadows` do, with the window's state passed in as a [`StylePaint`].
//!
//! Order (GPUI's): [`Style::paint_before_children`] pushes the shadows, then the background quad; the
//! caller paints the children; [`Style::paint_after_children`] pushes the border as four quads, each
//! clipped to one side's band, on top of the children.

use super::{BoxShadow, Overflow, Style};
use crate::{
    Background, BackgroundTag, BorderStyle, Bounds, Corners, Edges, Fill, Hsla, Pixels, Point, Quad,
    ScaledPixels, Scene, Shadow, point,
};
use std::fmt::Debug;

/// A rectangle that clips what is painted (GPUI's `ContentMask`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContentMask<P: Clone + Debug + Default + PartialEq> {
    /// The visible area.
    pub bounds: Bounds<P>,
}

impl ContentMask<Pixels> {
    /// The mask in device pixels.
    pub fn scale(&self, factor: f32) -> ContentMask<ScaledPixels> {
        ContentMask { bounds: self.bounds.scale(factor) }
    }

    /// The area both masks show.
    pub fn intersect(&self, other: &Self) -> Self {
        ContentMask { bounds: self.bounds.intersect(&other.bounds) }
    }
}

/// The window state painting reads (GPUI's `window.rem_size()`, `scale_factor()`, `element_opacity()`
/// and `content_mask()` at paint time).
#[derive(Clone, Debug, PartialEq)]
pub struct StylePaint {
    /// The rem size.
    pub rem_size: Pixels,
    /// The window's scale factor.
    pub scale_factor: f32,
    /// The element opacity: the product of the opacity stack, this element's `Style::opacity` included
    /// (GPUI's `with_element_opacity(style.opacity, ..)` around `Style::paint`).
    pub opacity: f32,
    /// The current content mask, in logical pixels.
    pub content_mask: ContentMask<Pixels>,
}

impl StylePaint {
    /// The same state with `mask` intersected into the content mask (GPUI's `with_content_mask(Some(mask))`).
    pub fn with_content_mask(&self, mask: &ContentMask<Pixels>) -> Self {
        Self { content_mask: mask.intersect(&self.content_mask), ..self.clone() }
    }
}

/// Pushes a quad as GPUI's `Window::paint_quad` does: scaled to device pixels (no rounding), colors faded
/// by the element opacity, clipped by the content mask, and skipped when nothing of it is inside the mask.
#[allow(clippy::too_many_arguments)]
pub fn paint_quad(
    scene: &mut Scene,
    cx: &StylePaint,
    bounds: Bounds<Pixels>,
    corner_radii: Corners<Pixels>,
    background: Background,
    border_widths: Edges<Pixels>,
    border_color: Hsla,
    border_style: BorderStyle,
) {
    let sf = cx.scale_factor;
    let bounds = bounds.scale(sf);
    let clip = cx.content_mask.scale(sf).bounds;
    if bounds.intersect(&clip).is_empty() {
        return;
    }
    scene.push_quad(Quad {
        bounds,
        clip,
        background: background.opacity(cx.opacity),
        border_color: border_color.opacity(cx.opacity),
        corner_radii: corner_radii.scale(sf),
        border_widths: border_widths.scale(sf),
        border_style,
    });
}

/// Pushes box shadows as GPUI's `Window::paint_shadows` does: each layer's box is `bounds` offset and
/// dilated by its spread, scaled to device pixels with its blur radius, color faded by the opacity. A
/// layer whose box (before the blur) misses the content mask is skipped, as GPUI's
/// `Scene::insert_primitive` skips it, even where its blur would reach into the mask.
pub fn paint_shadows(
    scene: &mut Scene,
    cx: &StylePaint,
    bounds: Bounds<Pixels>,
    corner_radii: Corners<Pixels>,
    shadows: &[BoxShadow],
) {
    let sf = cx.scale_factor;
    let clip = cx.content_mask.scale(sf).bounds;
    for shadow in shadows {
        let shadow_bounds = (bounds + shadow.offset).dilate(shadow.spread_radius).scale(sf);
        if shadow_bounds.intersect(&clip).is_empty() {
            continue;
        }
        scene.push_shadow(Shadow {
            bounds: shadow_bounds,
            clip,
            corner_radii: corner_radii.scale(sf),
            color: shadow.color.opacity(cx.opacity),
            blur_radius: shadow.blur_radius.scale(sf),
        });
    }
}

impl Style {
    /// The corner radii in pixels, clamped to half the shorter side of `bounds` (GPUI clamps before painting).
    pub fn corner_radii_for(&self, bounds: &Bounds<Pixels>, rem_size: Pixels) -> Corners<Pixels> {
        self.corner_radii.to_pixels(rem_size).clamp_radii_for_quad_size(bounds.size)
    }

    /// The first half of GPUI's `Style::paint`: the shadows, then the background quad (when the background
    /// is set and not transparent; its border color is the background's color with zero alpha).
    pub fn paint_before_children(&self, bounds: Bounds<Pixels>, cx: &StylePaint, scene: &mut Scene) {
        let has_shadows = !self.box_shadow.is_empty();
        let background =
            self.background.as_ref().and_then(Fill::color).filter(|color| !color.is_transparent());
        if !has_shadows && background.is_none() {
            return;
        }
        let sf = cx.scale_factor;
        let scaled_bounds = bounds.scale(sf);
        let scaled_clip = cx.content_mask.scale(sf).bounds;
        let paint_bg = background.is_some() && !scaled_bounds.intersect(&scaled_clip).is_empty();
        if !has_shadows && !paint_bg {
            return;
        }

        let corner_radii = self.corner_radii_for(&bounds, cx.rem_size);
        if has_shadows {
            paint_shadows(scene, cx, bounds, corner_radii, &self.box_shadow);
        }

        if paint_bg && let Some(background) = background {
            let mut border_color = match background.tag {
                BackgroundTag::Solid | BackgroundTag::PatternSlash => background.solid,
                BackgroundTag::LinearGradient => background.colors[0].color,
            };
            border_color.a = 0.;
            scene.push_quad(Quad {
                bounds: scaled_bounds,
                clip: scaled_clip,
                background: background.opacity(cx.opacity),
                border_color: border_color.opacity(cx.opacity),
                corner_radii: corner_radii.scale(sf),
                border_widths: Edges::default(),
                border_style: self.border_style,
            });
        }
    }

    /// The second half of GPUI's `Style::paint`, after the children: when the border is visible, one quad
    /// with the border widths and a transparent background, pushed four times, each clipped to the top,
    /// right, bottom or left band (bands as deep as the larger of the widest border and largest radius).
    pub fn paint_after_children(&self, bounds: Bounds<Pixels>, cx: &StylePaint, scene: &mut Scene) {
        if !self.is_border_visible() {
            return;
        }
        let sf = cx.scale_factor;
        let scaled_bounds = bounds.scale(sf);
        let scaled_mask = cx.content_mask.scale(sf).bounds;
        if scaled_bounds.intersect(&scaled_mask).is_empty() {
            return;
        }

        let corner_radii = self.corner_radii_for(&bounds, cx.rem_size);
        let border_widths = self.border_widths.to_pixels(cx.rem_size);
        let max_border_width = border_widths.max();
        let max_corner_radius = corner_radii.max();
        let band = max_border_width.max(max_corner_radius);

        let top_bounds = Bounds::from_corners(bounds.origin, bounds.top_right() + point(Pixels::ZERO, band));
        let bottom_bounds =
            Bounds::from_corners(bounds.bottom_left() - point(Pixels::ZERO, band), bounds.bottom_right());
        let left_bounds = Bounds::from_corners(
            top_bounds.bottom_left(),
            bottom_bounds.origin + point(max_border_width, Pixels::ZERO),
        );
        let right_bounds = Bounds::from_corners(
            top_bounds.bottom_right() - point(max_border_width, Pixels::ZERO),
            bottom_bounds.top_right(),
        );

        let border_color = self.border_color.unwrap_or_default();
        let mut background = border_color;
        background.a = 0.;
        let scaled_radii = corner_radii.scale(sf);
        let scaled_widths = border_widths.scale(sf);
        let faded_bg = Background::from(background).opacity(cx.opacity);
        let faded_border = border_color.opacity(cx.opacity);
        for side in [top_bounds, right_bounds, bottom_bounds, left_bounds] {
            let clip = side.intersect(&cx.content_mask.bounds).scale(sf);
            if scaled_bounds.intersect(&clip).is_empty() {
                continue;
            }
            scene.push_quad(Quad {
                bounds: scaled_bounds,
                clip,
                background: faded_bg,
                border_color: faded_border,
                corner_radii: scaled_radii,
                border_widths: scaled_widths,
                border_style: self.border_style,
            });
        }
    }

    /// The content mask for this element's children (GPUI's `overflow_mask`): `None` when both axes overflow
    /// visibly. Otherwise, with `inner` being `bounds` inset by the border widths when the border color is
    /// visible: both axes hidden give `inner`; one axis hidden gives `inner`'s range on the *visible* axis and
    /// the full `bounds` range on the hidden one (GPUI's arithmetic, kept as is).
    pub fn overflow_mask(&self, bounds: Bounds<Pixels>, rem_size: Pixels) -> Option<ContentMask<Pixels>> {
        let x_visible = self.overflow.x == Overflow::Visible;
        let y_visible = self.overflow.y == Overflow::Visible;
        if x_visible && y_visible {
            return None;
        }
        let mut min = bounds.origin;
        let mut max = bounds.bottom_right();
        if self.border_color.is_some_and(|color| !color.is_transparent()) {
            min.x += self.border_widths.left.to_pixels(rem_size);
            max.x -= self.border_widths.right.to_pixels(rem_size);
            min.y += self.border_widths.top.to_pixels(rem_size);
            max.y -= self.border_widths.bottom.to_pixels(rem_size);
        }
        let bounds = match (x_visible, y_visible) {
            (true, false) => {
                Bounds::from_corners(point(min.x, bounds.origin.y), point(max.x, bounds.bottom_right().y))
            }
            (false, true) => {
                Bounds::from_corners(point(bounds.origin.x, min.y), point(bounds.bottom_right().x, max.y))
            }
            _ => Bounds::from_corners(min, max),
        };
        Some(ContentMask { bounds })
    }
}

/// Where an SVG mask rendered at [`SMOOTH_SVG_SCALE_FACTOR`](crate::SMOOTH_SVG_SCALE_FACTOR) lands (GPUI's
/// `paint_svg`): centred on `bounds` (device pixels), at half the mask's size, origin rounded, size ceiled.
pub fn svg_sprite_bounds(
    bounds: Bounds<ScaledPixels>,
    mask_size: crate::Size<crate::DevicePixels>,
) -> Bounds<ScaledPixels> {
    let factor = crate::SMOOTH_SVG_SCALE_FACTOR;
    let half = Point::new(
        ScaledPixels::from(mask_size.width.0 as f32 / factor / 2.),
        ScaledPixels::from(mask_size.height.0 as f32 / factor / 2.),
    );
    let svg_bounds = Bounds {
        origin: bounds.center() - half,
        size: mask_size.map(|value| ScaledPixels::from(value.0 as f32 / factor)),
    };
    svg_bounds.map_origin(|origin| origin.round()).map_size(|size| size.ceil())
}
