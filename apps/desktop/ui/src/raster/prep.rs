//! Preparing the items that dirty tiles draw (colours converted, tables and masks fetched from the
//! caches or made), and drawing one prepared item into a tile.

use super::blend::{Src, Tint, alpha16, fill_span, rgba};
use super::cache::ByteCache;
use super::geom::{IRect, raster_span};
use super::gradient::{GradientSpan, Paint};
use super::hash::{finish, mix, mix_f2};
use super::image::{ImageCopy, ImageDraw, ImageKey};
use super::items::{Item, shadow_bounds};
use super::path::{PathMask, PathPaint, draw as draw_path, draw_underline};
use super::quad::QuadPrep;
use super::quad_eval::QuadShader;
use super::shadow::{self, ShadowMask, ShadowShape};
use super::sprite::{AlphaLut, Axis, draw_color, draw_glyph, draw_svg};
use super::target::Target;
use crate::{Background, BackgroundTag, GlyphBlend, Hsla, Primitive, SpriteKind};
use std::sync::Arc;

/// An item ready to draw.
pub(crate) enum Prep {
    /// An unrounded, unbordered quad with a solid background (or a straight underline).
    Fill(Src),
    Quad(Box<QuadPrep>),
    Shadow { mask: Arc<ShadowMask>, origin: (i32, i32), tint: Tint, alpha: u32 },
    Glyph { lut: Arc<AlphaLut>, tint: Tint },
    Svg { ax: Axis, ay: Axis, alpha: u32, tint: Tint },
    Color { opacity: u32 },
    Image(Box<ImageDraw>),
    Path { mask: Arc<PathMask>, paint: Box<PathPaint> },
    Underline([f32; 4]),
}

/// Byte budgets of the renderer's caches.
pub(crate) const SHADOW_BUDGET: usize = 24 << 20;
pub(crate) const PATH_BUDGET: usize = 16 << 20;
pub(crate) const IMAGE_BUDGET: usize = 64 << 20;
pub(crate) const TABLE_BUDGET: usize = 2 << 20;
/// Frames an unused entry survives even under budget (about a minute at 60 Hz).
const MAX_IDLE: u64 = 3600;

type LutKey = [u32; 9];

/// The renderer's caches.
pub(crate) struct Caches {
    pub shadows: ByteCache<shadow::ShadowKey, Arc<ShadowMask>>,
    pub paths: ByteCache<(u64, IRect), Arc<PathMask>>,
    pub images: ByteCache<ImageKey, Arc<ImageCopy>>,
    pub luts: ByteCache<LutKey, Arc<AlphaLut>>,
    pub gradients: ByteCache<u64, Arc<[Src]>>,
}

impl Caches {
    pub fn new() -> Self {
        Caches {
            shadows: ByteCache::new(SHADOW_BUDGET, MAX_IDLE),
            paths: ByteCache::new(PATH_BUDGET, MAX_IDLE),
            images: ByteCache::new(IMAGE_BUDGET, MAX_IDLE),
            luts: ByteCache::new(TABLE_BUDGET, MAX_IDLE),
            gradients: ByteCache::new(TABLE_BUDGET, MAX_IDLE),
        }
    }

    pub fn end_frame(&mut self) {
        self.shadows.end_frame();
        self.paths.end_frame();
        self.images.end_frame();
        self.luts.end_frame();
        self.gradients.end_frame();
    }

    /// Splits `total` bytes over the caches in the default proportions.
    pub fn set_budget(&mut self, total: usize) {
        let sum = SHADOW_BUDGET + PATH_BUDGET + IMAGE_BUDGET + 2 * TABLE_BUDGET;
        let part = |b: usize| (total as u128 * b as u128 / sum as u128) as usize;
        self.shadows.set_budget(part(SHADOW_BUDGET));
        self.paths.set_budget(part(PATH_BUDGET));
        self.images.set_budget(part(IMAGE_BUDGET));
        self.luts.set_budget(part(TABLE_BUDGET));
        self.gradients.set_budget(part(TABLE_BUDGET));
    }

    /// The bytes every cache holds.
    pub fn bytes(&self) -> usize {
        self.shadows.bytes() + self.paths.bytes() + self.images.bytes() + self.luts.bytes() + self.gradients.bytes()
    }

    fn gradient(&mut self, bg: &Background, p: &Paint) -> GradientSpan {
        let mut h = mix(0x6ad1, bg.color_space as u64);
        for s in &bg.colors {
            h = mix_f2(mix_f2(h, s.color.h, s.color.s), s.color.l, s.color.a);
        }
        let lut = self.gradients.get_or_insert_with(&finish(h), || {
            let t: Arc<[Src]> = GradientSpan::table(p).into();
            let bytes = t.len() * std::mem::size_of::<Src>() + 64;
            (t, bytes)
        });
        GradientSpan::new(p, lut)
    }
}

/// Glyph tables keyed by (colour, blend); consecutive glyphs mostly share one.
pub(crate) struct LutMemo {
    last: Option<(Hsla, Arc<AlphaLut>, Tint)>,
}

impl LutMemo {
    pub fn new() -> Self {
        LutMemo { last: None }
    }

    fn get(&mut self, caches: &mut Caches, color: Hsla, blend: &GlyphBlend) -> (Arc<AlphaLut>, Tint) {
        if let Some((c, lut, tint)) = &self.last
            && *c == color
        {
            return (lut.clone(), *tint);
        }
        let c = rgba(color);
        let b = |v: f32| v.to_bits();
        let g = blend.gamma_ratios;
        let key = [
            b(c[0]),
            b(c[1]),
            b(c[2]),
            b(c[3]),
            b(g[0]) ^ blend.apply as u32,
            b(g[1]),
            b(g[2]),
            b(g[3]),
            b(blend.grayscale_enhanced_contrast),
        ];
        let lut = caches.luts.get_or_insert_with(&key, || (Arc::new(AlphaLut::glyph(c, blend)), 1024 + 64));
        let tint = Tint::new(c);
        self.last = Some((color, lut.clone(), tint));
        (lut, tint)
    }
}

fn origin_of(b: &crate::Bounds<crate::ScaledPixels>) -> (i32, i32) {
    (raster_span(b.origin.x.0, 0.).0, raster_span(b.origin.y.0, 0.).0)
}

/// Prepares `item` of `prims`. `frame` is the frame rectangle (masks never exceed it).
pub(crate) fn prepare(
    item: &Item,
    prims: &[Primitive],
    caches: &mut Caches,
    memo: &mut LutMemo,
    blend: &GlyphBlend,
    frame: IRect,
) -> Prep {
    match &prims[item.prim as usize] {
        Primitive::Quad(q) => {
            let r = &q.corner_radii;
            let w = &q.border_widths;
            let simple = [r.top_left, r.top_right, r.bottom_right, r.bottom_left, w.top, w.right, w.bottom, w.left]
                .iter()
                .all(|v| v.0 == 0.);
            if simple && q.background.tag == BackgroundTag::Solid {
                return Prep::Fill(Src::new(rgba(q.background.solid)));
            }
            let sh = QuadShader::new(q);
            let grad = (q.background.tag == BackgroundTag::LinearGradient).then(|| caches.gradient(&q.background, &sh.paint));
            Prep::Quad(Box::new(QuadPrep::new(sh, grad)))
        }
        Primitive::Shadow(s) => {
            let drawn = shadow_bounds(&s.bounds, s.blur_radius.0);
            let (x0, x1) = raster_span(drawn.origin.x.0, drawn.origin.x.0 + drawn.size.width.0);
            let (y0, y1) = raster_span(drawn.origin.y.0, drawn.origin.y.0 + drawn.size.height.0);
            // The mask covers the drawn pixels inside the frame.
            let m = IRect::new(x0, y0, x1, y1).intersect(&frame);
            let half = [s.bounds.size.width.0 / 2., s.bounds.size.height.0 / 2.];
            let center = [s.bounds.origin.x.0 + half[0], s.bounds.origin.y.0 + half[1]];
            let r = &s.corner_radii;
            let shape = ShadowShape {
                half,
                radii: [r.top_left.0, r.top_right.0, r.bottom_right.0, r.bottom_left.0],
                blur: s.blur_radius.0,
                offset: [m.x0 as f32 - center[0], m.y0 as f32 - center[1]],
                w: m.width(),
                h: m.height(),
            };
            let mask = caches.shadows.get_or_insert_with(&shape.key(), || {
                let mask = Arc::new(shape.render());
                let bytes = mask.bytes();
                (mask, bytes)
            });
            let c = rgba(s.color);
            Prep::Shadow { mask, origin: (m.x0, m.y0), tint: Tint::new(c), alpha: alpha16(c[3]) }
        }
        Primitive::MonoSprite(s) => match s.kind {
            SpriteKind::Glyph => {
                let (lut, tint) = memo.get(caches, s.color, blend);
                Prep::Glyph { lut, tint }
            }
            SpriteKind::Svg => {
                let c = rgba(s.color);
                let b = &s.bounds;
                Prep::Svg {
                    ax: Axis::new(b.origin.x.0, b.size.width.0, s.mask.size.width.0),
                    ay: Axis::new(b.origin.y.0, b.size.height.0, s.mask.size.height.0),
                    alpha: alpha16(c[3]),
                    tint: Tint::new(c),
                }
            }
        },
        Primitive::ColorSprite(s) => Prep::Color { opacity: alpha16(s.opacity) },
        Primitive::Image(m) => {
            let b = &m.bounds;
            let (x0, x1) = raster_span(b.origin.x.0, b.origin.x.0 + b.size.width.0);
            let (y0, y1) = raster_span(b.origin.y.0, b.origin.y.0 + b.size.height.0);
            let dst = [b.origin.x.0 - x0 as f32, b.origin.y.0 - y0 as f32, b.size.width.0, b.size.height.0];
            let key = ImageKey {
                id: m.image.id.0,
                frame: m.frame,
                grayscale: m.grayscale,
                w: x1 - x0,
                h: y1 - y0,
                bits: dst.map(f32::to_bits),
            };
            let copy = caches.images.get_or_insert_with(&key, || {
                let c = Arc::new(ImageCopy::new(&m.image, m.frame, m.grayscale, x1 - x0, y1 - y0, dst));
                let bytes = c.bytes();
                (c, bytes)
            });
            Prep::Image(Box::new(ImageDraw::new(copy, (x0, y0), m.opacity, m)))
        }
        Primitive::Path(p) => {
            let b = &p.bounds;
            let (x0, x1) = raster_span(b.origin.x.0, b.origin.x.0 + b.size.width.0);
            let (y0, y1) = raster_span(b.origin.y.0, b.origin.y.0 + b.size.height.0);
            let rect = IRect::new(x0, y0, x1, y1).intersect(&frame);
            let mask = caches.paths.get_or_insert_with(&(item.aux, rect), || {
                let m = Arc::new(PathMask::new(&p.path, rect));
                let bytes = m.bytes();
                (m, bytes)
            });
            let paint = if p.paint.tag == BackgroundTag::LinearGradient {
                let cb = p.bounds.intersect(&p.clip);
                let paint = Paint::new(
                    &p.paint,
                    [cb.origin.x.0, cb.origin.y.0],
                    [cb.size.width.0, cb.size.height.0],
                );
                PathPaint::Gradient(caches.gradient(&p.paint, &paint))
            } else {
                let c = rgba(p.paint.solid);
                PathPaint::Solid(AlphaLut::coverage(c[3]), Tint::new(c))
            };
            Prep::Path { mask, paint: Box::new(paint) }
        }
        Primitive::Underline(u) => {
            let c = rgba(u.color);
            if u.wavy { Prep::Underline(c) } else { Prep::Fill(Src::new(c)) }
        }
    }
}

/// Draws a prepared item's pixels inside `r` (its rectangle cut to the tile).
pub(crate) fn draw(t: &mut Target, prep: &Prep, prim: &Primitive, r: IRect) {
    match (prep, prim) {
        (Prep::Fill(s), _) => {
            if s.inv == 0 {
                for y in r.y0..r.y1 {
                    t.row(y, r.x0, r.x1).fill(s.packed);
                }
            } else {
                for y in r.y0..r.y1 {
                    fill_span(t.row(y, r.x0, r.x1), s);
                }
            }
        }
        (Prep::Quad(q), _) => q.draw(t, r),
        (Prep::Shadow { mask, origin, tint, alpha }, _) => shadow::draw(t, mask, *origin, tint, *alpha, r),
        (Prep::Glyph { lut, tint }, Primitive::MonoSprite(s)) => draw_glyph(t, &s.mask, origin_of(&s.bounds), lut, tint, r),
        (Prep::Svg { ax, ay, alpha, tint }, Primitive::MonoSprite(s)) => draw_svg(t, &s.mask, ax, ay, *alpha, tint, r),
        (Prep::Color { opacity }, Primitive::ColorSprite(s)) => draw_color(t, &s.mask, origin_of(&s.bounds), *opacity, r),
        (Prep::Image(d), _) => d.draw(t, r),
        (Prep::Path { mask, paint }, _) => {
            let r = r.intersect(&mask.rect);
            if !r.is_empty() {
                draw_path(t, mask, paint, r);
            }
        }
        (Prep::Underline(c), Primitive::Underline(u)) => draw_underline(t, u, *c, r),
        _ => {}
    }
}
