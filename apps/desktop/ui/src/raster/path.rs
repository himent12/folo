// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (underline_fragment)
//! Paths (coverage from tiny-skia, cached by geometry) and underlines (straight and wavy).

use super::blend::{ONE, Src, Tint, sat};
use super::geom::IRect;
use super::gradient::GradientSpan;
use super::hash::{finish, mix, mix_f2};
use super::sprite::{AlphaLut, draw_mask};
use super::target::Target;
use crate::{FillRule, LineCap, LineJoin, PathData, PathKind, Underline};

/// A hash of everything about a path's geometry that affects its coverage.
pub(crate) fn geometry_hash(p: &PathData) -> u64 {
    let mut h = 0x5157_4a5e_u64;
    for c in &p.contours {
        h = mix(h, c.points.len() as u64 | (c.closed as u64) << 63);
        for pt in &c.points {
            h = mix_f2(h, pt.x.0, pt.y.0);
        }
    }
    match &p.kind {
        PathKind::Fill(rule) => h = mix(h, *rule as u64),
        PathKind::Stroke(s) => {
            h = mix_f2(h, s.width.0, s.miter_limit);
            h = mix(h, 16 | (s.cap as u64) << 8 | (s.join as u64) << 4);
            if let Some(d) = &s.dash {
                for v in d {
                    h = mix(h, v.0.to_bits() as u64);
                }
            }
        }
    }
    finish(h)
}

/// A path's coverage over `rect` (absolute device px), one byte per pixel.
pub(crate) struct PathMask {
    pub rect: IRect,
    pub data: Vec<u8>,
}

impl PathMask {
    pub fn bytes(&self) -> usize {
        self.data.len() + 64
    }

    /// Rasterizes `p` over `rect`.
    pub fn new(p: &PathData, rect: IRect) -> Self {
        let (w, h) = (rect.width().max(0) as u32, rect.height().max(0) as u32);
        let empty = || PathMask { rect, data: vec![0; (w * h) as usize] };
        let Some(mut mask) = tiny_skia::Mask::new(w, h) else {
            return empty();
        };
        let (ox, oy) = (rect.x0 as f32, rect.y0 as f32);
        let mut b = tiny_skia::PathBuilder::new();
        for c in &p.contours {
            let mut pts = c.points.iter();
            let Some(first) = pts.next() else { continue };
            b.move_to(first.x.0 - ox, first.y.0 - oy);
            for pt in pts {
                b.line_to(pt.x.0 - ox, pt.y.0 - oy);
            }
            if c.closed {
                b.close();
            }
        }
        let Some(path) = b.finish() else {
            return empty();
        };
        let (path, rule) = match &p.kind {
            PathKind::Fill(rule) => (
                Some(path),
                match rule {
                    FillRule::NonZero => tiny_skia::FillRule::Winding,
                    FillRule::EvenOdd => tiny_skia::FillRule::EvenOdd,
                },
            ),
            PathKind::Stroke(s) => {
                let stroke = tiny_skia::Stroke {
                    width: s.width.0,
                    miter_limit: s.miter_limit,
                    line_cap: match s.cap {
                        LineCap::Butt => tiny_skia::LineCap::Butt,
                        LineCap::Square => tiny_skia::LineCap::Square,
                        LineCap::Round => tiny_skia::LineCap::Round,
                    },
                    line_join: match s.join {
                        LineJoin::Miter => tiny_skia::LineJoin::Miter,
                        LineJoin::Bevel => tiny_skia::LineJoin::Bevel,
                        LineJoin::Round => tiny_skia::LineJoin::Round,
                    },
                    dash: s
                        .dash
                        .as_ref()
                        .and_then(|d| tiny_skia::StrokeDash::new(d.iter().map(|v| v.0).collect(), 0.)),
                };
                (path.stroke(&stroke, 1.0), tiny_skia::FillRule::Winding)
            }
        };
        if let Some(path) = path {
            mask.fill_path(&path, rule, true, tiny_skia::Transform::identity());
        }
        PathMask { rect, data: mask.data().to_vec() }
    }
}

/// How a path is painted over its coverage.
pub(crate) enum PathPaint {
    Solid(Box<AlphaLut>, Tint),
    Gradient(GradientSpan),
}

/// Draws the part of `mask` inside `r`.
pub(crate) fn draw(t: &mut Target, mask: &PathMask, paint: &PathPaint, r: IRect) {
    let w = mask.rect.width() as usize;
    let origin = (mask.rect.x0, mask.rect.y0);
    match paint {
        PathPaint::Solid(lut, tint) => draw_mask(t, &mask.data, w, origin, lut, tint, r),
        PathPaint::Gradient(g) => {
            for y in r.y0..r.y1 {
                let start = (y - origin.1) as usize * w + (r.x0 - origin.0) as usize;
                let src = &mask.data[start..start + (r.x1 - r.x0) as usize];
                let row = t.row(y, r.x0, r.x1);
                for ((x, p), &m) in (r.x0..r.x1).zip(row.iter_mut()).zip(src) {
                    if m != 0 {
                        *p = scaled(g.at(x, y), m as u32).over(*p);
                    }
                }
            }
        }
    }
}

/// `s` with its alpha multiplied by `m / 255`.
#[inline]
fn scaled(s: &Src, m: u32) -> Src {
    if m == 255 {
        return *s;
    }
    let k = |v: u32| ((v as u64 * m as u64 + 127) / 255) as u32;
    let inv = ONE - k(ONE - s.inv);
    Src { r: k(s.r), g: k(s.g), b: k(s.b), inv, packed: s.packed }
}

/// Draws an underline inside `r`: straight ones are a solid span, wavy ones run GPUI's shader.
pub(crate) fn draw_underline(t: &mut Target, u: &Underline, color: [f32; 4], r: IRect) {
    if !u.wavy {
        let s = Src::new(color);
        for y in r.y0..r.y1 {
            super::blend::fill_span(t.row(y, r.x0, r.x1), &s);
        }
        return;
    }
    const WAVE_FREQUENCY: f32 = 2.0;
    const WAVE_HEIGHT_RATIO: f32 = 0.8;
    let half_thickness = u.thickness.0 * 0.5;
    let (ox, oy) = (u.bounds.origin.x.0, u.bounds.origin.y.0);
    let h = u.bounds.size.height.0;
    let frequency = (std::f32::consts::PI * WAVE_FREQUENCY * u.thickness.0) / h;
    let amplitude = (u.thickness.0 * WAVE_HEIGHT_RATIO) / h;
    for y in r.y0..r.y1 {
        let row = t.row(y, r.x0, r.x1);
        let st_y = (y as f32 + 0.5 - oy) / h - 0.5;
        for (x, p) in (r.x0..r.x1).zip(row.iter_mut()) {
            let st_x = (x as f32 + 0.5 - ox) / h;
            let sine = (st_x * frequency).sin() * amplitude;
            let d_sine = (st_x * frequency).cos() * amplitude * frequency;
            let distance = (st_y - sine) / (1. + d_sine * d_sine).sqrt();
            let px = distance * h;
            let alpha = sat(0.5 - (-(px + half_thickness)).max(px - half_thickness));
            let s = Src::new([color[0], color[1], color[2], color[3] * alpha]);
            *p = s.over(*p);
        }
    }
}
