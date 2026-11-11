//! Triangles rasterized as GPUI's path pipeline does on the GPU: into a 4x MSAA target (D3D11's
//! standard sample pattern), each sample covered when it lies inside a triangle by the top-left fill
//! rule, with vertices snapped to the rasterizer's 1/256 px grid, then resolved (the share of a
//! pixel's four samples covered). The path's clip clips the triangles themselves (`SV_ClipDistance`),
//! so it applies per sample too. Each sample counts the triangles over it (up to 3): a translucent
//! paint blends once per triangle on the GPU (`raster::path::SampleLut` resolves the counts).

use super::geom::IRect;
use crate::{Point, ScaledPixels};

/// D3D11's standard 4x sample positions, in 1/256 px from the pixel's centre.
const SAMPLES: [(i64, i64); 4] = [(-32, -96), (96, -32), (-96, 32), (32, 96)];

/// A mask byte holds four 2-bit counts, sample `k` in bits `2k..2k + 2`.
pub(crate) const COUNT_MAX: u8 = 3;

/// Coordinates beyond this many pixels are clamped (far outside any frame; keeps the fixed-point
/// products within `i64`).
const LIMIT: f32 = (1 << 22) as f32;

/// `v` in 1/256 px, rounded to nearest (ties up), as the rasterizer converts positions.
#[inline]
fn fixed(v: f32) -> i64 {
    let s = v.clamp(-LIMIT, LIMIT) * 256. + 0.5;
    let i = s as i64;
    if (i as f32) > s { i - 1 } else { i }
}

/// One edge `a -> b` as an edge function `E(p) = dx * (p.y - a.y) - dy * (p.x - a.x)`, positive inside;
/// `min` is the smallest value that counts as inside (0 on a top or left edge, 1 elsewhere).
#[derive(Clone, Copy)]
struct Edge {
    ax: i64,
    ay: i64,
    dx: i64,
    dy: i64,
    min: i64,
}

impl Edge {
    fn new(a: (i64, i64), b: (i64, i64)) -> Self {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        // With the interior on the positive side (y down), the edge is a left edge when it goes up,
        // a top edge when it is horizontal and goes right.
        let top_left = dy < 0 || (dy == 0 && dx > 0);
        Edge { ax: a.0, ay: a.1, dx, dy, min: if top_left { 0 } else { 1 } }
    }

    #[inline(always)]
    fn inside(&self, x: i64, y: i64) -> bool {
        self.dx * (y - self.ay) - self.dy * (x - self.ax) >= self.min
    }

    /// The x where the edge's line crosses row `y` (float), when it is not horizontal.
    fn x_at(&self, y: i64) -> Option<f64> {
        (self.dy != 0).then(|| self.ax as f64 + (y - self.ay) as f64 * self.dx as f64 / self.dy as f64)
    }
}

/// How many of `triangles` (three points each, device px) cover each sample of each pixel of `rect`,
/// one byte per pixel (2 bits per sample, saturating), keeping only samples inside `clip`
/// (`[x0, y0, x1, y1]`, device px).
pub(crate) fn coverage(triangles: &[Point<ScaledPixels>], clip: [f32; 4], rect: IRect) -> Vec<u8> {
    let (w, h) = (rect.width().max(0) as usize, rect.height().max(0) as usize);
    let mut bits = vec![0u8; w * h];
    let clip = clip.map(fixed);
    for t in triangles.chunks_exact(3) {
        let mut v = [0; 3].map(|_| (0i64, 0i64));
        for (v, p) in v.iter_mut().zip(t) {
            *v = (fixed(p.x.0), fixed(p.y.0));
        }
        let area = (v[1].0 - v[0].0) * (v[2].1 - v[0].1) - (v[1].1 - v[0].1) * (v[2].0 - v[0].0);
        if area == 0 {
            continue;
        }
        if area < 0 {
            v.swap(1, 2);
        }
        let edges = [Edge::new(v[0], v[1]), Edge::new(v[1], v[2]), Edge::new(v[2], v[0])];
        let ymin = v.iter().map(|p| p.1).min().unwrap_or(0).max(clip[1]);
        let ymax = v.iter().map(|p| p.1).max().unwrap_or(0).min(clip[3]);
        let xmin = v.iter().map(|p| p.0).min().unwrap_or(0).max(clip[0]);
        let xmax = v.iter().map(|p| p.0).max().unwrap_or(0).min(clip[2]);
        if ymin > ymax || xmin > xmax {
            continue;
        }
        // Pixel rows whose samples can fall in [ymin, ymax].
        let j0 = (((ymin - 128 - 96) >> 8) as i32).max(rect.y0);
        let j1 = ((((ymax - 128 + 96) >> 8) + 1) as i32).min(rect.y1);
        for j in j0..j1 {
            let row = (j - rect.y0) as usize * w;
            for (k, &(ox, oy)) in SAMPLES.iter().enumerate() {
                let sy = j as i64 * 256 + 128 + oy;
                if sy < clip[1] || sy >= clip[3] {
                    continue;
                }
                // The interval the edges' crossings allow on this sample row, widened by a pixel;
                // every sample in it is tested exactly.
                let (mut lo, mut hi) = (xmin as f64, xmax as f64);
                for e in &edges {
                    if let Some(x) = e.x_at(sy) {
                        // Inside is x >= crossing when the edge goes up (dy < 0), else x <= crossing.
                        if e.dy < 0 { lo = lo.max(x) } else { hi = hi.min(x) }
                    }
                }
                if lo > hi + 256. {
                    continue;
                }
                let i0 = ((((lo as i64) - 128 - ox) >> 8) as i32 - 1).max(rect.x0);
                let i1 = ((((hi as i64) - 128 - ox) >> 8) as i32 + 2).min(rect.x1);
                for i in i0..i1 {
                    let sx = i as i64 * 256 + 128 + ox;
                    if sx >= clip[0] && sx < clip[2] && edges.iter().all(|e| e.inside(sx, sy)) {
                        let b = &mut bits[row + (i - rect.x0) as usize];
                        if (*b >> (2 * k)) & COUNT_MAX != COUNT_MAX {
                            *b += 1 << (2 * k);
                        }
                    }
                }
            }
        }
    }
    bits
}

/// The covered samples of a mask byte (any count).
#[cfg(test)]
pub(crate) fn covered(b: u8) -> u32 {
    (0..4).filter(|k| (b >> (2 * k)) & COUNT_MAX != 0).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point;

    fn tri(p: [(f32, f32); 3]) -> Vec<Point<ScaledPixels>> {
        p.iter().map(|&(x, y)| point(ScaledPixels(x), ScaledPixels(y))).collect()
    }

    const OPEN: [f32; 4] = [-1e6, -1e6, 1e6, 1e6];

    /// Two triangles sharing a diagonal cover each sample on it once: a square is exactly solid, and
    /// an edge through a column covers the samples left of it (x offsets -6, -2, 2, 6 sixteenths).
    #[test]
    fn samples_follow_the_top_left_rule() {
        let mut sq = tri([(0., 0.), (4., 0.), (4., 4.)]);
        sq.extend(tri([(0., 0.), (4., 4.), (0., 4.)]));
        let c = coverage(&sq, OPEN, IRect::new(-1, -1, 5, 5));
        for y in 0..6 {
            for x in 0..6 {
                let inside = (1..5).contains(&x) && (1..5).contains(&y);
                // Every sample once: the diagonal's samples belong to one triangle.
                assert_eq!(c[y * 6 + x], if inside { 0b01_01_01_01 } else { 0 }, "{x} {y}");
            }
        }
        for (x1, want) in [(2.1, 0), (2.2, 1), (2.5, 2), (2.7, 3), (2.9, 4)] {
            let mut r = tri([(0., 0.), (x1, 0.), (x1, 4.)]);
            r.extend(tri([(0., 0.), (x1, 4.), (0., 4.)]));
            assert_eq!(covered(coverage(&r, OPEN, IRect::new(0, 0, 4, 4))[4 + 2]), want, "{x1}");
        }
        // Overlapping triangles count per sample, up to 3.
        let one = tri([(0., 0.), (4., 0.), (0., 4.)]);
        let four: Vec<_> = (0..4).flat_map(|_| one.clone()).collect();
        assert_eq!(coverage(&four, OPEN, IRect::new(0, 0, 4, 4))[0], 0xff);
        let two: Vec<_> = (0..2).flat_map(|_| one.clone()).collect();
        assert_eq!(coverage(&two, OPEN, IRect::new(0, 0, 4, 4))[0], 0b10_10_10_10);
        // A sample exactly on a right edge is out, on a left edge in: x1 = 2 + 6/16 drops the
        // sample at 2.375, x0 = 2 + 6/16 keeps it.
        let r = tri([(0., 0.), (2.375, 0.), (2.375, 4.)]);
        let q = tri([(0., 4.), (2.375, 4.), (0., 0.)]);
        let both: Vec<_> = r.into_iter().chain(q).collect();
        assert_eq!(covered(coverage(&both, OPEN, IRect::new(0, 0, 4, 4))[4 + 2]), 1);
        let l = tri([(2.375, 0.), (9., 0.), (9., 4.)]);
        let m = tri([(2.375, 0.), (9., 4.), (2.375, 4.)]);
        let both: Vec<_> = l.into_iter().chain(m).collect();
        assert_eq!(covered(coverage(&both, OPEN, IRect::new(0, 0, 4, 4))[4 + 2]), 3);
    }

    /// The clip cuts per sample, keeping samples on its left / top edge.
    #[test]
    fn clip_applies_per_sample() {
        let mut sq = tri([(0., 0.), (8., 0.), (8., 8.)]);
        sq.extend(tri([(0., 0.), (8., 8.), (0., 8.)]));
        let c = coverage(&sq, [2.5, 0., 8., 8.], IRect::new(0, 0, 8, 8));
        assert_eq!(c[8..13].iter().map(|&b| covered(b)).collect::<Vec<_>>(), [0, 0, 2, 4, 4]);
        let c = coverage(&sq, [2.375, 0., 8., 8.], IRect::new(0, 0, 8, 8));
        assert_eq!(covered(c[8 + 2]), 3);
    }
}
