//! Anti-aliased signed-distance rasterisation and procedural backgrounds for the synthetic shapes dataset.

use crate::image::Image;
use crate::rng::{Rng, mix_seed};

/// A drawing primitive in pixel space. A shape is the union of its primitives.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Prim {
    /// A simple polygon (convex or not), vertices in order.
    Poly(Vec<[f32; 2]>),
    /// A filled disc.
    Disc { c: [f32; 2], r: f32 },
    /// An annulus: points within `half` of the circle of radius `mid`.
    Annulus { c: [f32; 2], mid: f32, half: f32 },
}

impl Prim {
    /// Signed distance in pixels (negative inside).
    #[inline]
    fn sdf(&self, x: f32, y: f32) -> f32 {
        match self {
            Prim::Disc { c, r } => ((x - c[0]).hypot(y - c[1])) - r,
            Prim::Annulus { c, mid, half } => ((x - c[0]).hypot(y - c[1]) - mid).abs() - half,
            Prim::Poly(v) => sd_polygon(v, x, y),
        }
    }

    /// Tight pixel bounds `[x1, y1, x2, y2]`.
    pub(crate) fn bounds(&self) -> [f32; 4] {
        match self {
            Prim::Disc { c, r } => [c[0] - r, c[1] - r, c[0] + r, c[1] + r],
            Prim::Annulus { c, mid, half } => {
                let r = mid + half;
                [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
            }
            Prim::Poly(v) => v.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| {
                [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]
            }),
        }
    }
}

/// Exact signed distance to a simple polygon (Inigo Quilez's formulation).
fn sd_polygon(v: &[[f32; 2]], x: f32, y: f32) -> f32 {
    let n = v.len();
    let mut d = (x - v[0][0]).powi(2) + (y - v[0][1]).powi(2);
    let mut s = 1.0f32;
    let mut j = n - 1;
    for i in 0..n {
        let (ex, ey) = (v[j][0] - v[i][0], v[j][1] - v[i][1]);
        let (wx, wy) = (x - v[i][0], y - v[i][1]);
        let t = ((wx * ex + wy * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
        let (bx, by) = (wx - ex * t, wy - ey * t);
        d = d.min(bx * bx + by * by);
        let c1 = y >= v[i][1];
        let c2 = y < v[j][1];
        let c3 = ex * wy > ey * wx;
        if (c1 && c2 && c3) || (!c1 && !c2 && !c3) {
            s = -s;
        }
        j = i;
    }
    s * d.sqrt()
}

/// Union bounds of a list of primitives.
pub(crate) fn bounds_of(prims: &[Prim]) -> [f32; 4] {
    prims.iter().map(Prim::bounds).fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |a, b| {
        [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
    })
}

#[inline]
fn sdf_union(prims: &[Prim], x: f32, y: f32) -> f32 {
    prims.iter().fold(f32::MAX, |d, p| d.min(p.sdf(x, y)))
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A floating-point RGB canvas (linear 0..1 values) that is quantised once at the end.
pub(crate) struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 3]>,
}

/// How one shape is painted.
pub(crate) struct Paint {
    pub color: [f32; 3],
    /// Direction-of-light shading strength (0 = flat).
    pub shade: f32,
    /// Drop shadow offset in pixels and opacity.
    pub shadow: ([f32; 2], f32),
    /// Darker rim strength.
    pub rim: f32,
}

impl Canvas {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        Canvas { w, h, px: vec![[0.0; 3]; w * h] }
    }

    fn window(&self, b: [f32; 4], margin: f32) -> (usize, usize, usize, usize) {
        let x0 = (b[0] - margin).floor().max(0.0) as usize;
        let y0 = (b[1] - margin).floor().max(0.0) as usize;
        let x1 = ((b[2] + margin).ceil().max(0.0) as usize).min(self.w);
        let y1 = ((b[3] + margin).ceil().max(0.0) as usize).min(self.h);
        (x0, y0, x1.max(x0), y1.max(y0))
    }

    /// Mean colour under the bounds (sparse sampling).
    pub(crate) fn mean_under(&self, b: [f32; 4]) -> [f32; 3] {
        let (x0, y0, x1, y1) = self.window(b, 0.0);
        let mut acc = [0.0f32; 3];
        let mut n = 0.0;
        let step = ((x1 - x0).max(y1 - y0) / 12).max(1);
        for y in (y0..y1).step_by(step) {
            for x in (x0..x1).step_by(step) {
                let p = self.px[y * self.w + x];
                for c in 0..3 {
                    acc[c] += p[c];
                }
                n += 1.0;
            }
        }
        if n == 0.0 { [0.5; 3] } else { acc.map(|v| v / n) }
    }

    /// Paints the union of `prims` with anti-aliased edges, a soft drop shadow, light shading and a rim.
    pub(crate) fn draw(&mut self, prims: &[Prim], paint: &Paint) {
        let b = bounds_of(prims);
        let size = (b[2] - b[0]).max(b[3] - b[1]).max(1.0);
        let centre = [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5];
        let ([sx, sy], alpha) = paint.shadow;
        if alpha > 0.0 {
            let blur = (size * 0.05).clamp(1.0, 8.0);
            let sb = [b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy];
            let (x0, y0, x1, y1) = self.window(sb, blur + 1.0);
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = sdf_union(prims, x as f32 + 0.5 - sx, y as f32 + 0.5 - sy);
                    let a = alpha * smoothstep(blur, -blur, d);
                    if a > 0.0 {
                        let p = &mut self.px[y * self.w + x];
                        for v in p.iter_mut() {
                            *v *= 1.0 - a;
                        }
                    }
                }
            }
        }
        let (x0, y0, x1, y1) = self.window(b, 1.0);
        let rim_w = (size * 0.04).clamp(0.8, 3.0);
        let inv = 2.0 / size;
        for y in y0..y1 {
            for x in x0..x1 {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let d = sdf_union(prims, fx, fy);
                let cov = (0.5 - d).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                // Light from the top left: brighter on that side.
                let t = (-(fx - centre[0]) - (fy - centre[1])) * inv * std::f32::consts::FRAC_1_SQRT_2;
                let light = 1.0 + paint.shade * t.clamp(-1.0, 1.0);
                let rim = 1.0 - paint.rim * smoothstep(-rim_w, 0.0, d);
                let k = light * rim;
                let p = &mut self.px[y * self.w + x];
                for (pc, &col) in p.iter_mut().zip(&paint.color) {
                    let v = (col * k).clamp(0.0, 1.0);
                    *pc += (v - *pc) * cov;
                }
            }
        }
    }

    /// Quantises to 8-bit with per-pixel grain of amplitude `grain` (in 0..255 levels) seeded by `seed`.
    pub(crate) fn to_image(&self, grain: f32, seed: u64) -> Image {
        let mut data = vec![0u8; self.w * self.h * 3];
        let mut rng = Rng::new(seed);
        for (o, p) in data.as_chunks_mut::<3>().0.iter_mut().zip(&self.px) {
            // Triangular-ish noise from two uniforms, shared across channels (luma grain) plus a little chroma.
            let n = (rng.f32() + rng.f32() - 1.0) * grain;
            for c in 0..3 {
                let chroma = (rng.f32() - 0.5) * grain * 0.3;
                o[c] = (p[c] * 255.0 + n + chroma).round().clamp(0.0, 255.0) as u8;
            }
        }
        Image { width: self.w as u32, height: self.h as u32, data }
    }
}

/// Parameters of a procedural background.
#[derive(Clone, Debug)]
pub(crate) struct Background {
    pub kind: u8,
    pub colors: [[f32; 3]; 3],
    pub angle: f32,
    pub scale: f32,
    pub strength: f32,
    pub stripe_freq: f32,
    pub vignette: f32,
    pub seed: u64,
}

/// Value noise lattice for one octave.
struct Lattice {
    gw: usize,
    vals: Vec<f32>,
    cell: f32,
}

impl Lattice {
    fn new(w: usize, h: usize, cell: f32, seed: u64) -> Self {
        let gw = (w as f32 / cell).ceil() as usize + 2;
        let gh = (h as f32 / cell).ceil() as usize + 2;
        let mut rng = Rng::new(seed);
        Lattice { gw, vals: (0..gw * gh).map(|_| rng.f32()).collect(), cell }
    }

    #[inline]
    fn at(&self, x: f32, y: f32) -> f32 {
        let (gx, gy) = (x / self.cell, y / self.cell);
        let (ix, iy) = (gx as usize, gy as usize);
        let (fx, fy) = (gx - ix as f32, gy - iy as f32);
        let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
        let i = iy * self.gw + ix;
        let (a, b, c, d) =
            (self.vals[i], self.vals[i + 1], self.vals[i + self.gw], self.vals[i + self.gw + 1]);
        let top = a + (b - a) * sx;
        let bot = c + (d - c) * sx;
        top + (bot - top) * sy
    }
}

/// Fractal value noise in `0..1` sampled on the pixel grid.
fn fbm(w: usize, h: usize, base_cell: f32, octaves: usize, seed: u64) -> Vec<f32> {
    let lattices: Vec<Lattice> = (0..octaves)
        .map(|o| Lattice::new(w, h, (base_cell / (1 << o) as f32).max(2.0), mix_seed(&[seed, o as u64])))
        .collect();
    let mut norm = 0.0;
    let mut amp = 1.0;
    let amps: Vec<f32> = (0..octaves)
        .map(|_| {
            let a = amp;
            norm += a;
            amp *= 0.5;
            a
        })
        .collect();
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32, y as f32);
            let mut v = 0.0;
            for (l, a) in lattices.iter().zip(&amps) {
                v += l.at(fx, fy) * a;
            }
            out[y * w + x] = v / norm;
        }
    }
    out
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

impl Background {
    /// Paints the background into a canvas.
    pub(crate) fn paint(&self, canvas: &mut Canvas) {
        let (w, h) = (canvas.w, canvas.h);
        let diag = ((w * w + h * h) as f32).sqrt();
        let (sa, ca) = self.angle.sin_cos();
        let (octaves, cell) = match self.kind {
            3 => (3, self.scale * 0.15),
            _ => (4, self.scale),
        };
        let noise = fbm(w, h, cell.max(3.0), octaves, self.seed);
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - cx, y as f32 - cy);
                let g = ((fx * ca + fy * sa) / diag + 0.5).clamp(0.0, 1.0);
                let n = noise[y * w + x];
                let mut c = lerp3(self.colors[0], self.colors[1], g);
                match self.kind {
                    // Smooth gradient with a light cloud texture.
                    0 => c = lerp3(c, self.colors[2], (n - 0.5) * self.strength + 0.5 * self.strength),
                    // Marbled: noise drives a strong mix of three colours.
                    1 => {
                        let t = smoothstep(0.25, 0.75, n);
                        c = lerp3(lerp3(self.colors[0], self.colors[2], t), self.colors[1], g * 0.5);
                    }
                    // Soft stripes warped by noise.
                    2 => {
                        let phase = (fx * -sa + fy * ca) * self.stripe_freq + n * 6.0;
                        let s = 0.5 + 0.5 * phase.sin();
                        c = lerp3(c, self.colors[2], s * self.strength);
                    }
                    // Fine fabric / paper grain.
                    _ => {
                        let t = (n - 0.5) * 2.0 * self.strength;
                        c = c.map(|v| v * (1.0 + t));
                    }
                }
                if self.vignette > 0.0 {
                    let r = (fx * fx + fy * fy).sqrt() / (diag * 0.5);
                    let k = 1.0 - self.vignette * r * r;
                    c = c.map(|v| v * k);
                }
                canvas.px[y * w + x] = c.map(|v| v.clamp(0.0, 1.0));
            }
        }
    }
}

/// HSV (all in 0..1) to RGB.
pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Relative luminance of a colour (Rec. 709 weights).
pub(crate) fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_sdf_signs() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!((sd_polygon(&sq, 5.0, 5.0) + 5.0).abs() < 1e-5);
        assert!((sd_polygon(&sq, 15.0, 5.0) - 5.0).abs() < 1e-5);
        assert!((sd_polygon(&sq, 13.0, 14.0) - 5.0).abs() < 1e-5);
        let p = Prim::Annulus { c: [0.0, 0.0], mid: 5.0, half: 1.0 };
        assert!(p.sdf(5.0, 0.0) < 0.0 && p.sdf(0.0, 0.0) > 0.0 && p.sdf(7.0, 0.0) > 0.0);
    }

    #[test]
    fn coverage_is_antialiased_and_exact() {
        let mut c = Canvas::new(40, 40);
        let disc = [Prim::Disc { c: [20.0, 20.0], r: 10.0 }];
        c.draw(&disc, &Paint { color: [1.0; 3], shade: 0.0, shadow: ([0.0; 2], 0.0), rim: 0.0 });
        let total: f32 = c.px.iter().map(|p| p[0]).sum();
        let area = std::f32::consts::PI * 100.0;
        assert!((total - area).abs() / area < 0.01, "{total} vs {area}");
        assert!(c.px.iter().any(|p| p[0] > 0.05 && p[0] < 0.95), "edges are blended");
    }
}
