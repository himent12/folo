// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (underline_fragment)
//! Paths (4x multisampled coverage, as GPUI's path target, from tiny-skia's aliased fills, cached by
//! geometry) and underlines (straight and wavy).

use super::blend::{ONE, Src, Tint, alpha16, sat};
use super::geom::IRect;
use super::gradient::GradientSpan;
use super::hash::{finish, mix, mix_f2};
use super::sprite::{AlphaLut, draw_mask};
use super::target::Target;
use super::triangles::COUNT_MAX;
use crate::{FillRule, LineCap, LineJoin, PathData, PathKind, Underline};

/// A hash of everything about a path's geometry that affects its coverage.
pub(crate) fn compute_geometry_hash(p: &PathData) -> u64 {
    let mut h = 0x5157_4a5e_u64;
    for c in &p.contours {
        h = mix(h, c.points.len() as u64 | (c.closed as u64) << 63);
        for pt in &c.points {
            h = mix_f2(h, pt.x.0, pt.y.0);
        }
    }
    match &p.kind {
        PathKind::Fill(rule) => h = mix(h, *rule as u64),
        PathKind::Triangles => {
            h = mix(h, 0x7e1 | (p.triangles.len() as u64) << 12);
            for pt in &p.triangles {
                h = mix_f2(h, pt.x.0, pt.y.0);
            }
        }
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

/// The SVG miter limit (tiny-skia's: bevel when `1 / sin(angle / 2)` exceeds it) that lyon's
/// `miter_limit` amounts to. lyon's normal at a join has length `1 / sin(angle / 2)` and it bevels
/// when `normal.square_length() > miter_limit^2 * 4` (`miter_limit_is_exceeded`, lyon_tessellation
/// 1.0.23, GPUI's), so its default 4 keeps miters up to a ratio of 8.
pub(crate) fn svg_miter_limit(lyon_limit: f32) -> f32 {
    lyon_limit * 2.
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

    /// Rasterizes `p` over `rect`; triangles keep only the samples inside `clip` (`[x0, y0, x1, y1]`).
    pub fn new(p: &PathData, rect: IRect, clip: [f32; 4]) -> Self {
        if p.kind == PathKind::Triangles {
            return PathMask { rect, data: super::triangles::coverage(&p.triangles, clip, rect) };
        }
        let (w, h) = (rect.width().max(0) as u32, rect.height().max(0) as u32);
        let len = w as usize * h as usize;
        let empty = || PathMask { rect, data: vec![0; len] };
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
            PathKind::Triangles => unreachable!("triangles are rasterized above"),
            PathKind::Stroke(s) => {
                let stroke = tiny_skia::Stroke {
                    width: s.width.0,
                    miter_limit: svg_miter_limit(s.miter_limit),
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
        let Some(path) = path else {
            return empty();
        };
        // GPUI draws paths into a 4x MSAA target (D3D11's standard pattern) and resolves it: a
        // pixel's coverage is the share of its four samples inside the path. Each sample is one
        // aliased fill (centre sampling) of the path moved by minus the sample's offset.
        let mut samples = vec![0u8; len];
        for (k, (dx, dy)) in MSAA_4X.into_iter().enumerate() {
            mask.clear();
            let t = tiny_skia::Transform::from_translate(-dx, -dy);
            mask.fill_path(&path, rule, false, t);
            for (s, &m) in samples.iter_mut().zip(mask.data()) {
                *s |= ((m != 0) as u8) << (2 * k);
            }
        }
        PathMask { rect, data: samples }
    }
}

/// D3D11's standard 4x sample positions, in pixels from the pixel's centre.
const MSAA_4X: [(f32, f32); 4] =
    [(-2. / 16., -6. / 16.), (6. / 16., -2. / 16.), (-6. / 16., 2. / 16.), (2. / 16., 6. / 16.)];

/// The resolved coverage (`0..=255`) of each mask byte: the share of its samples covered, as the
/// resolve's 8-bit UNORM rounding of `n / 4`.
pub(crate) fn coverage_of(b: u8) -> u8 {
    const QUARTERS: [u8; 5] = [0, 64, 128, 191, 255];
    QUARTERS[(0..4).filter(|k| (b >> (2 * k)) & COUNT_MAX != 0).count()]
}

/// The alpha (`0..=ONE`) a solid paint of alpha `a` leaves for each mask byte, as GPUI's path target
/// computes it: each triangle over a sample blends `a` over it (premultiplied, ONE / INV_SRC_ALPHA),
/// the sample stored as 8-bit UNORM after each, and the resolve averages the four samples and
/// rounds to 8 bits again.
pub(crate) fn sample_lut(a: f32) -> AlphaLut {
    let a = sat(a);
    // The sample's 8-bit alpha after 0..=3 blends.
    let mut level = [0u32; 4];
    for n in 1..4 {
        level[n] = ((a + level[n - 1] as f32 / 255. * (1. - a)) * 255. + 0.5) as u32;
    }
    AlphaLut(std::array::from_fn(|b| {
        let sum: u32 = (0..4).map(|k| level[(b >> (2 * k)) & COUNT_MAX as usize]).sum();
        alpha16(((sum + 2) / 4) as f32 / 255.)
    }))
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
                        *p = scaled(g.at(x, y), coverage_of(m) as u32).over(*p);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Contour, LineCap, LineJoin, PathData, PathKind, ScaledPixels, StrokeStyle, point};

    const OPEN: [f32; 4] = [-1e6, -1e6, 1e6, 1e6];

    fn pt(x: f32, y: f32) -> crate::Point<ScaledPixels> {
        point(ScaledPixels(x), ScaledPixels(y))
    }

    /// Coverage counts GPUI's four MSAA samples: an edge through a column covers the samples left
    /// of it, in quarters (D3D11's standard pattern, x offsets -6, -2, 2, 6 sixteenths).
    #[test]
    fn coverage_is_four_samples() {
        let rect = |x1: f32| {
            let c = Contour { points: vec![pt(0., 0.), pt(x1, 0.), pt(x1, 4.), pt(0., 4.)], closed: true };
            PathMask::new(
                &PathData::new(vec![c], PathKind::Fill(FillRule::NonZero)),
                IRect::new(0, 0, 4, 4),
                OPEN,
            )
        };
        // Samples of column 2 sit at x = 2.125, 2.375, 2.625, 2.875.
        for (x1, want) in [(2.1, 0), (2.2, 64), (2.5, 128), (2.7, 191), (2.9, 255)] {
            let m = rect(x1);
            assert_eq!(coverage_of(m.data[4 + 1]), 255, "{x1}");
            assert_eq!(coverage_of(m.data[4 + 2]), want, "{x1}");
        }
    }

    /// A sample under two triangles of a translucent paint blends twice; the resolve rounds.
    #[test]
    fn overlaps_blend_per_sample() {
        let lut = sample_lut(0.4);
        let a = |b: u8| (lut.0[b as usize] as f32 / ONE as f32 * 255.).round() as u32;
        assert_eq!(a(0), 0);
        assert_eq!(a(0b01_01_01_01), 102);
        // 0.4 + 0.4 * 0.6 = 0.64 (163/255 after rounding).
        assert_eq!(a(0b10_10_10_10), 163);
        // Two samples once, two twice: (102 + 102 + 163 + 163) / 4 = 132.5 -> 133.
        assert_eq!(a(0b10_10_01_01), 133);
        let opaque = sample_lut(1.);
        assert_eq!(opaque.0[0b00_01_00_11], alpha16(128. / 255.));
    }

    /// lyon's miter limit 4 keeps a miter up to a ratio `1 / sin(angle / 2)` of 8: a 10-degree
    /// spike (ratio 11.5) is beveled, a 20-degree one (ratio 5.8, beyond SVG's 4) keeps its point.
    #[test]
    fn miter_limit_is_lyons() {
        assert_eq!(svg_miter_limit(4.), 8.);
        let spike = |deg: f32| {
            let h = 20. / (deg.to_radians() / 2.).tan();
            let c = Contour { points: vec![pt(10., 4.), pt(30., 4. + h), pt(50., 4.)], closed: false };
            let kind = PathKind::Stroke(StrokeStyle {
                width: ScaledPixels(2.),
                cap: LineCap::Butt,
                join: LineJoin::Miter,
                miter_limit: 4.,
                dash: None,
            });
            let m = PathMask::new(&PathData::new(vec![c], kind), IRect::new(0, 0, 60, 400), OPEN);
            // The lowest covered row in the spike's column.
            (0..400).rev().find(|&y| m.data[y * 60 + 30] != 0).unwrap() as f32
        };
        // Tip of the centre line at 4 + h; a miter reaches about one ratio of half widths past it.
        let h20 = 20. / 10f32.to_radians().tan();
        assert!(spike(20.) > 4. + h20 + 2.5, "{} vs {}", spike(20.), h20);
        let h10 = 20. / 5f32.to_radians().tan();
        assert!(spike(10.) < 4. + h10 + 1.5, "{} vs {}", spike(10.), h10);
    }
}
