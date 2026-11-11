//! Preparing the items that dirty tiles draw (masks and tables fetched from the caches or made), and
//! drawing one item into a tile.

use super::blend::{Src, Tint, alpha16, fill_span, rgba};
use super::cache::ByteCache;
use super::geom::{IRect, raster_span};
use super::gradient::{GradientSpan, Paint};
use super::hash::{finish, mix, mix_f2};
use super::image::{ImageCopy, ImageDraw, ImageKey};
use super::items::shadow_bounds;
use super::path::{PathMask, PathPaint, draw as draw_path, draw_underline, geometry_hash};
use super::quad::QuadPrep;
use super::quad_eval::QuadShader;
use super::shadow::{self, ShadowKey, ShadowMask, ShadowShape};
use super::sprite::{AlphaLut, Axis, draw_color, draw_glyph, draw_svg};
use super::target::Target;
use crate::{
    Background, BackgroundTag, GlyphBlend, Hsla, Image, Path, PathData, Primitive, Quad, RenderImage, Shadow,
    SpriteKind,
};
use std::sync::{Arc, OnceLock};

/// An item whose preparation needed the caches.
pub(crate) enum Prep {
    /// A gradient quad (its colour table is cached).
    Quad(Box<QuadPrep>),
    Shadow {
        mask: Arc<ShadowMask>,
        origin: (i32, i32),
        tint: Tint,
        alpha: u32,
    },
    Image(Box<ImageDraw>),
    Path {
        mask: Arc<PathMask>,
        paint: Box<PathPaint>,
    },
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
        self.shadows.bytes()
            + self.paths.bytes()
            + self.images.bytes()
            + self.luts.bytes()
            + self.gradients.bytes()
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

/// The glyph tables a frame uses, one per (colour, blend); glyph items point into it. Consecutive
/// glyphs mostly share a colour, so lookups compare with the last one first.
#[derive(Default)]
pub(crate) struct GlyphTables {
    pub tables: Vec<(Arc<AlphaLut>, Tint)>,
    colors: Vec<Hsla>,
    last: Option<(Hsla, u32)>,
}

impl GlyphTables {
    pub fn clear(&mut self) {
        self.tables.clear();
        self.colors.clear();
        self.last = None;
    }

    /// The index of the table for `color`.
    fn index(&mut self, caches: &mut Caches, color: Hsla, blend: &GlyphBlend) -> u32 {
        if let Some((c, i)) = self.last
            && c == color
        {
            return i;
        }
        let i = match self.colors.iter().position(|c| *c == color) {
            Some(i) => i as u32,
            None => {
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
                let lut =
                    caches.luts.get_or_insert_with(&key, || (Arc::new(AlphaLut::glyph(c, blend)), 1024 + 64));
                self.tables.push((lut, Tint::new(c)));
                self.colors.push(color);
                self.tables.len() as u32 - 1
            }
        };
        self.last = Some((color, i));
        i
    }
}

fn origin_of(b: &crate::Bounds<crate::ScaledPixels>) -> (i32, i32) {
    (raster_span(b.origin.x.0, 0.).0, raster_span(b.origin.y.0, 0.).0)
}

/// What `prepare` made for an item: nothing (prepared in the tile), a glyph table index, a prepared
/// item, or a job for a cache miss (`Job::run` on any thread, then `Job::finish`).
pub(crate) enum Prepared {
    Lazy,
    Glyph(u32),
    /// A rounded or bordered quad: its `QuadPrep` is made by the first tile that draws it.
    Quad,
    Item(Prep),
    Pending(Job),
}

/// How a tile draws an item: from the tile alone, with a glyph table, with what `prepare` made, or
/// with the quad's shared preparation.
#[derive(Clone, Copy)]
pub(crate) enum Use<'a> {
    Lazy,
    Glyph(&'a (Arc<AlphaLut>, Tint)),
    Prep(&'a Prep),
    Quad(&'a OnceLock<QuadPrep>),
}

/// An unrounded, unbordered quad.
fn is_plain(q: &Quad) -> bool {
    let (r, w) = (&q.corner_radii, &q.border_widths);
    [r.top_left, r.top_right, r.bottom_right, r.bottom_left, w.top, w.right, w.bottom, w.left]
        .iter()
        .all(|v| v.0 == 0.)
}

/// A cache miss to fill: the expensive part of preparing a shadow, an image or a path.
pub(crate) enum Job {
    Shadow(ShadowKey, ShadowShape),
    Image(ImageKey, Arc<RenderImage>, usize, bool, [f32; 4]),
    Path((u64, IRect), Arc<PathData>),
}

/// What a job made.
pub(crate) enum Made {
    Shadow(Arc<ShadowMask>),
    Image(Arc<ImageCopy>),
    Path(Arc<PathMask>),
}

impl Job {
    /// The expensive part: shadow mask, resampled image or path coverage. Needs no cache.
    pub fn run(&self) -> Made {
        match self {
            Job::Shadow(_, shape) => Made::Shadow(Arc::new(shape.render())),
            Job::Image(key, image, frame, grayscale, dst) => {
                Made::Image(Arc::new(ImageCopy::new(image, *frame, *grayscale, key.w, key.h, *dst)))
            }
            Job::Path(key, path) => Made::Path(Arc::new(PathMask::new(path, key.1))),
        }
    }

    /// Caches what `run` made and prepares `prim` with it.
    pub fn finish(self, made: Made, prim: &Primitive, caches: &mut Caches, frame: IRect) -> Prep {
        match (self, made, prim) {
            (Job::Shadow(key, _), Made::Shadow(mask), Primitive::Shadow(s)) => {
                caches.shadows.insert(key, mask.clone(), mask.bytes());
                shadow_prep(s, mask, frame)
            }
            (Job::Image(key, ..), Made::Image(copy), Primitive::Image(m)) => {
                caches.images.insert(key, copy.clone(), copy.bytes());
                image_prep(m, copy, frame)
            }
            (Job::Path(key, _), Made::Path(mask), Primitive::Path(p)) => {
                caches.paths.insert(key, mask.clone(), mask.bytes());
                path_prep(p, mask, caches)
            }
            _ => unreachable!("a job finishes with what it made, for its own primitive"),
        }
    }
}

/// The shadow's mask rectangle (its drawn pixels inside the frame) and shape.
fn shadow_shape(s: &Shadow, frame: IRect) -> (IRect, ShadowShape) {
    let drawn = shadow_bounds(&s.bounds, s.blur_radius.0);
    let (x0, x1) = raster_span(drawn.origin.x.0, drawn.origin.x.0 + drawn.size.width.0);
    let (y0, y1) = raster_span(drawn.origin.y.0, drawn.origin.y.0 + drawn.size.height.0);
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
    (m, shape)
}

fn shadow_prep(s: &Shadow, mask: Arc<ShadowMask>, frame: IRect) -> Prep {
    let m = shadow_shape(s, frame).0;
    let c = rgba(s.color);
    Prep::Shadow { mask, origin: (m.x0, m.y0), tint: Tint::new(c), alpha: alpha16(c[3]) }
}

/// Destinations up to this many pixels (16 MB of copy) are resampled whole.
const MAX_FULL_COPY: u64 = 4 << 20;

/// Where the image's copy lands (see `MAX_FULL_COPY`), its cache key, and the bounds
/// relative to the copy (`x`, `y`, `width`, `height`).
fn image_key(m: &Image, frame: IRect) -> ((i32, i32), ImageKey, [f32; 4]) {
    let b = &m.bounds;
    let (x0, x1) = raster_span(b.origin.x.0, b.origin.x.0 + b.size.width.0);
    let (y0, y1) = raster_span(b.origin.y.0, b.origin.y.0 + b.size.height.0);
    // The whole destination is resampled, so the copy survives scrolling; a destination larger
    // than `MAX_FULL_COPY` pixels (a zoomed image) is resampled only where the frame shows it.
    let full = IRect::new(x0, y0, x1, y1);
    let area = full.width().max(0) as u64 * full.height().max(0) as u64;
    let c = if area <= MAX_FULL_COPY { full } else { full.intersect(&frame) };
    let dst = [b.origin.x.0 - c.x0 as f32, b.origin.y.0 - c.y0 as f32, b.size.width.0, b.size.height.0];
    let key = ImageKey {
        id: m.image.id.0,
        frame: m.frame,
        grayscale: m.grayscale,
        w: c.width().max(0),
        h: c.height().max(0),
        bits: dst.map(f32::to_bits),
    };
    ((c.x0, c.y0), key, dst)
}

fn image_prep(m: &Image, copy: Arc<ImageCopy>, frame: IRect) -> Prep {
    let (origin, ..) = image_key(m, frame);
    Prep::Image(Box::new(ImageDraw::new(copy, origin, m.opacity, m)))
}

/// The path's coverage key: its geometry and its mask rectangle (bounds inside the frame).
fn path_key(p: &Path, frame: IRect) -> (u64, IRect) {
    let b = &p.bounds;
    let (x0, x1) = raster_span(b.origin.x.0, b.origin.x.0 + b.size.width.0);
    let (y0, y1) = raster_span(b.origin.y.0, b.origin.y.0 + b.size.height.0);
    (geometry_hash(&p.path), IRect::new(x0, y0, x1, y1).intersect(&frame))
}

fn path_prep(p: &Path, mask: Arc<PathMask>, caches: &mut Caches) -> Prep {
    let paint = if p.paint.tag == BackgroundTag::LinearGradient {
        let cb = p.bounds.intersect(&p.clip);
        let paint = Paint::new(&p.paint, [cb.origin.x.0, cb.origin.y.0], [cb.size.width.0, cb.size.height.0]);
        PathPaint::Gradient(caches.gradient(&p.paint, &paint))
    } else {
        let c = rgba(p.paint.solid);
        PathPaint::Solid(Box::new(AlphaLut::coverage(c[3])), Tint::new(c))
    };
    Prep::Path { mask, paint: Box::new(paint) }
}

/// Prepares `prim` for drawing: the work that needs the caches (shadow masks, image copies, path
/// coverage, gradient and glyph tables). Everything else is cheap and done by each tile itself.
/// A cache miss comes back as a `Job`. `frame` is the frame rectangle (masks never exceed it).
pub(crate) fn prepare(
    prim: &Primitive,
    caches: &mut Caches,
    glyphs: &mut GlyphTables,
    blend: &GlyphBlend,
    frame: IRect,
) -> Prepared {
    match prim {
        Primitive::Quad(q) => {
            if q.background.tag != BackgroundTag::LinearGradient {
                // Plain solid quads are a fill; others share one `QuadPrep` across their tiles.
                return if is_plain(q) && q.background.tag == BackgroundTag::Solid {
                    Prepared::Lazy
                } else {
                    Prepared::Quad
                };
            }
            let sh = QuadShader::new(q);
            let grad = caches.gradient(&q.background, &sh.paint);
            Prepared::Item(Prep::Quad(Box::new(QuadPrep::new(sh, Some(grad)))))
        }
        Primitive::Shadow(s) => {
            let (_, shape) = shadow_shape(s, frame);
            let key = shape.key();
            match caches.shadows.get(&key) {
                Some(mask) => Prepared::Item(shadow_prep(s, mask, frame)),
                None => Prepared::Pending(Job::Shadow(key, shape)),
            }
        }
        // Icons take the glyph correction too (GPUI's one monochrome sprite shader).
        Primitive::MonoSprite(s) => Prepared::Glyph(glyphs.index(caches, s.color, blend)),
        Primitive::ColorSprite(_) | Primitive::Underline(_) => Prepared::Lazy,
        Primitive::Image(m) => {
            let (_, key, dst) = image_key(m, frame);
            match caches.images.get(&key) {
                Some(copy) => Prepared::Item(image_prep(m, copy, frame)),
                None => Prepared::Pending(Job::Image(key, m.image.clone(), m.frame, m.grayscale, dst)),
            }
        }
        Primitive::Path(p) => {
            let key = path_key(p, frame);
            match caches.paths.get(&key) {
                Some(mask) => Prepared::Item(path_prep(p, mask, caches)),
                None => Prepared::Pending(Job::Path(key, p.path.clone())),
            }
        }
    }
}

/// Draws `prim`'s pixels inside `r` (its rectangle cut to the tile). `prep` is what `prepare` made
/// for it (`None` for lazy items), `glyphs` the frame's glyph tables.
pub(crate) fn draw(t: &mut Target, prim: &Primitive, used: Use, r: IRect) {
    match used {
        Use::Prep(p) => {
            match p {
                Prep::Quad(q) => q.draw(t, r),
                Prep::Shadow { mask, origin, tint, alpha } => shadow::draw(t, mask, *origin, tint, *alpha, r),
                Prep::Image(d) => d.draw(t, r),
                Prep::Path { mask, paint } => {
                    let r = r.intersect(&mask.rect);
                    if !r.is_empty() {
                        draw_path(t, mask, paint, r);
                    }
                }
            }
            return;
        }
        Use::Quad(cell) => {
            if let Primitive::Quad(q) = prim {
                cell.get_or_init(|| QuadPrep::new(QuadShader::new(q), None)).draw(t, r);
            }
            return;
        }
        Use::Glyph(..) | Use::Lazy => {}
    }
    match prim {
        Primitive::Quad(q) => {
            if is_plain(q) && q.background.tag == BackgroundTag::Solid {
                fill(t, &Src::new(rgba(q.background.solid)), r);
            } else {
                QuadPrep::new(QuadShader::new(q), None).draw(t, r);
            }
        }
        Primitive::MonoSprite(s) => match s.kind {
            SpriteKind::Glyph => {
                if let Use::Glyph((lut, tint)) = used {
                    draw_glyph(t, &s.mask, origin_of(&s.bounds), lut, tint, r);
                }
            }
            SpriteKind::Svg => {
                if let Use::Glyph((lut, tint)) = used {
                    let b = &s.bounds;
                    let ax = Axis::new(b.origin.x.0, b.size.width.0, s.mask.size.width.0);
                    let ay = Axis::new(b.origin.y.0, b.size.height.0, s.mask.size.height.0);
                    draw_svg(t, &s.mask, &ax, &ay, lut, tint, r);
                }
            }
        },
        Primitive::ColorSprite(s) => draw_color(t, &s.mask, origin_of(&s.bounds), alpha16(s.opacity), r),
        Primitive::Underline(u) => {
            let c = rgba(u.color);
            if u.wavy {
                draw_underline(t, u, c, r);
            } else {
                fill(t, &Src::new(c), r);
            }
        }
        // Prepared kinds always have a `Prep`.
        Primitive::Shadow(_) | Primitive::Image(_) | Primitive::Path(_) => {}
    }
}

fn fill(t: &mut Target, s: &Src, r: IRect) {
    if s.inv == 0 {
        for y in r.y0..r.y1 {
            t.row(y, r.x0, r.x1).fill(s.packed);
        }
    } else if !s.is_noop() {
        for y in r.y0..r.y1 {
            fill_span(t.row(y, r.x0, r.x1), s);
        }
    }
}
