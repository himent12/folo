//! Resolving a scene: each visible primitive becomes an `Item` with the pixel rectangle it may write,
//! the rectangle it certainly covers with one opaque colour (for occlusion), and a signature hashing
//! every field that affects its pixels (ids for masks, images and paths, never their bytes).

use super::geom::{IRect, clip_rect, raster_span};
use super::hash::{finish, mix, mix_f2};
use super::quad::opaque_core;
use super::tiles::TILE;
use crate::{Background, BackgroundTag, Bounds, Corners, Hsla, Primitive, ScaledPixels};

/// One primitive, resolved for binning (items are indexed like the scene's primitives; one that
/// draws nothing has an empty rectangle and tile span).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Item {
    /// The pixels it may write (bounds, clip and frame intersected).
    pub rect: IRect,
    /// The pixels it certainly paints with an opaque colour, hiding everything under them.
    pub opaque: IRect,
    /// Everything that affects its pixels, hashed.
    pub sig: u64,
    /// The tiles it touches: columns `tiles[0]..tiles[2]`, rows `tiles[1]..tiles[3]`.
    pub tiles: [u16; 4],
}

#[inline]
fn mix_bounds(h: u64, b: &Bounds<ScaledPixels>) -> u64 {
    mix_f2(mix_f2(h, b.origin.x.0, b.origin.y.0), b.size.width.0, b.size.height.0)
}

#[inline]
fn mix_rect(h: u64, r: &IRect) -> u64 {
    mix(
        mix(h, (r.x0 as u32 as u64) << 32 | r.y0 as u32 as u64),
        (r.x1 as u32 as u64) << 32 | r.y1 as u32 as u64,
    )
}

#[inline]
fn mix_hsla(h: u64, c: &Hsla) -> u64 {
    mix_f2(mix_f2(h, c.h, c.s), c.l, c.a)
}

#[inline]
fn mix_corners(h: u64, c: &Corners<ScaledPixels>) -> u64 {
    mix_f2(mix_f2(h, c.top_left.0, c.top_right.0), c.bottom_right.0, c.bottom_left.0)
}

fn mix_background(mut h: u64, b: &Background) -> u64 {
    h = mix(h, b.tag as u64 | (b.color_space as u64) << 8);
    h = mix_hsla(h, &b.solid);
    if b.tag != BackgroundTag::Solid {
        h = mix(h, b.gradient_angle_or_pattern_height.to_bits() as u64);
        for s in &b.colors {
            h = mix(mix_hsla(h, &s.color), s.percentage.to_bits() as u64);
        }
    }
    h
}

/// Whether a background paints nothing.
fn invisible(b: &Background) -> bool {
    b.is_transparent()
}

/// The shadow's drawn rectangle: its bounds grown by three blur radii (the shader's margin).
pub(crate) fn shadow_bounds(b: &Bounds<ScaledPixels>, blur: f32) -> Bounds<ScaledPixels> {
    let margin = 3.0 * blur;
    let mut out = *b;
    out.origin.x.0 -= margin;
    out.origin.y.0 -= margin;
    out.size.width.0 += 2.0 * margin;
    out.size.height.0 += 2.0 * margin;
    out
}

/// The pixels primitives may write: their bounds' pixels inside their clip's and the frame's. Runs of
/// primitives share a clip, so the last clip's pixel rectangle is kept.
struct Clips {
    frame: IRect,
    last: Option<([u32; 4], IRect)>,
}

impl Clips {
    /// `pixel_rect(bounds, clip)` inside the frame.
    #[inline]
    fn rect(&mut self, bounds: &Bounds<ScaledPixels>, clip: &Bounds<ScaledPixels>) -> IRect {
        let key = [clip.origin.x.0, clip.origin.y.0, clip.size.width.0, clip.size.height.0].map(f32::to_bits);
        let clip = match self.last {
            Some((k, r)) if k == key => r,
            _ => {
                let r = clip_rect(clip).intersect(&self.frame);
                self.last = Some((key, r));
                r
            }
        };
        let (x0, x1) = raster_span(bounds.origin.x.0, bounds.origin.x.0 + bounds.size.width.0);
        let (y0, y1) = raster_span(bounds.origin.y.0, bounds.origin.y.0 + bounds.size.height.0);
        IRect::new(x0, y0, x1, y1).intersect(&clip)
    }
}

/// Resolves `prims` into `out` (one item each); primitives that draw nothing inside `frame` get
/// `Item::default()`.
pub(crate) fn resolve(prims: &[Primitive], frame: IRect, out: &mut [Item]) {
    debug_assert_eq!(prims.len(), out.len());
    let mut clips = Clips { frame, last: None };
    for (p, out) in prims.iter().zip(out.iter_mut()) {
        *out = Item::default();
        let (rect, opaque, sig) = match p {
            Primitive::Quad(q) => {
                let border = q.border_color.a > 0. && q.border_widths.any(|w| w.0 > 0.);
                if invisible(&q.background) && !border {
                    continue;
                }
                let rect = clips.rect(&q.bounds, &q.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(1, &rect);
                h = mix_bounds(h, &q.bounds);
                h = mix_background(h, &q.background);
                h = mix_hsla(h, &q.border_color);
                h = mix_corners(h, &q.corner_radii);
                let w = &q.border_widths;
                h = mix_f2(mix_f2(h, w.top.0, w.right.0), w.bottom.0, w.left.0);
                h = mix(h, q.border_style as u64);
                (rect, opaque_core(q, rect), h)
            }
            Primitive::Shadow(s) => {
                if s.color.a <= 0. || s.blur_radius.0.is_nan() || s.blur_radius.0 <= 0. {
                    // A zero blur divides by zero in the shader: NaN alpha, which draws nothing.
                    continue;
                }
                let rect = clips.rect(&shadow_bounds(&s.bounds, s.blur_radius.0), &s.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(2, &rect);
                h = mix_bounds(h, &s.bounds);
                h = mix_corners(h, &s.corner_radii);
                h = mix_hsla(h, &s.color);
                h = mix(h, s.blur_radius.0.to_bits() as u64);
                (rect, IRect::EMPTY, h)
            }
            Primitive::MonoSprite(s) => {
                if s.color.a <= 0. {
                    continue;
                }
                let rect = clips.rect(&s.bounds, &s.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(3, &rect);
                h = mix_bounds(h, &s.bounds);
                h = mix_hsla(h, &s.color);
                h = mix(h, s.mask.id << 1 | (s.kind == crate::SpriteKind::Svg) as u64);
                (rect, IRect::EMPTY, h)
            }
            Primitive::ColorSprite(s) => {
                if s.opacity.is_nan() || s.opacity <= 0. {
                    continue;
                }
                let rect = clips.rect(&s.bounds, &s.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(4, &rect);
                h = mix_bounds(h, &s.bounds);
                h = mix(mix(h, s.mask.id), s.opacity.to_bits() as u64);
                (rect, IRect::EMPTY, h)
            }
            Primitive::Image(m) => {
                if m.opacity.is_nan() || m.opacity <= 0. {
                    continue;
                }
                let rect = clips.rect(&m.bounds, &m.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(5, &rect);
                h = mix_bounds(h, &m.bounds);
                h = mix_corners(h, &m.corner_radii);
                h = mix(h, m.image.id.0);
                h = mix(h, m.frame as u64 | (m.grayscale as u64) << 63);
                h = mix(h, m.opacity.to_bits() as u64);
                (rect, IRect::EMPTY, h)
            }
            Primitive::Path(p) => {
                if invisible(&p.paint) {
                    continue;
                }
                let rect = clips.rect(&p.bounds, &p.clip);
                if rect.is_empty() {
                    continue;
                }
                let geometry = p.path.geometry_hash();
                let mut h = mix_rect(6, &rect);
                h = mix_bounds(h, &p.bounds);
                // A gradient spans the clipped bounds, so the clip matters beyond `rect`.
                h = mix_rect(h, &clip_rect(&p.clip));
                h = mix_bounds(h, &p.clip);
                h = mix(h, geometry);
                h = mix_background(h, &p.paint);
                (rect, IRect::EMPTY, h)
            }
            Primitive::Underline(u) => {
                if u.color.a <= 0. {
                    continue;
                }
                let rect = clips.rect(&u.bounds, &u.clip);
                if rect.is_empty() {
                    continue;
                }
                let mut h = mix_rect(7, &rect);
                h = mix_bounds(h, &u.bounds);
                h = mix_hsla(h, &u.color);
                h = mix(h, u.thickness.0.to_bits() as u64 | (u.wavy as u64) << 32);
                (rect, IRect::EMPTY, h)
            }
        };
        // Rectangles are inside the frame: non-negative.
        let t = TILE - 1;
        let tiles = [rect.x0 >> 6, rect.y0 >> 6, (rect.x1 + t) >> 6, (rect.y1 + t) >> 6].map(|v| v as u16);
        *out = Item { rect, opaque, sig: finish(sig), tiles };
    }
}
