//! Anti-aliased signed-distance rasterisation and procedural backgrounds for the synthetic shapes dataset.

use super::math;
use crate::buffers;
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

/// Precomputed polygon edge data for bit-exact SDF evaluation without per-pixel edge recomputation.
#[derive(Clone, Copy)]
struct PolyEdge {
    vx: f32,
    vy: f32,
    vj_y: f32,
    ex: f32,
    ey: f32,
    ee: f32,
}

enum PreparedPrim {
    Disc { cx: f32, cy: f32, r: f32 },
    Annulus { cx: f32, cy: f32, mid: f32, half: f32 },
    Poly { v0: [f32; 2], edges: Vec<PolyEdge> },
}

impl PreparedPrim {
    fn from_prim(p: &Prim) -> Self {
        match p {
            Prim::Disc { c, r } => PreparedPrim::Disc { cx: c[0], cy: c[1], r: *r },
            Prim::Annulus { c, mid, half } => {
                PreparedPrim::Annulus { cx: c[0], cy: c[1], mid: *mid, half: *half }
            }
            Prim::Poly(v) => {
                let n = v.len();
                let mut edges = Vec::with_capacity(n);
                let mut j = n - 1;
                for i in 0..n {
                    let (ex, ey) = (v[j][0] - v[i][0], v[j][1] - v[i][1]);
                    edges.push(PolyEdge {
                        vx: v[i][0],
                        vy: v[i][1],
                        vj_y: v[j][1],
                        ex,
                        ey,
                        ee: ex * ex + ey * ey,
                    });
                    j = i;
                }
                PreparedPrim::Poly { v0: v[0], edges }
            }
        }
    }

    #[inline(always)]
    fn sdf(&self, x: f32, y: f32) -> f32 {
        match self {
            PreparedPrim::Disc { cx, cy, r } => dist(x - cx, y - cy) - r,
            PreparedPrim::Annulus { cx, cy, mid, half } => (dist(x - cx, y - cy) - mid).abs() - half,
            PreparedPrim::Poly { v0, edges } => {
                let (dx, dy) = (x - v0[0], y - v0[1]);
                let mut d = dx * dx + dy * dy;
                let mut s = 1.0f32;
                for e in edges {
                    let (wx, wy) = (x - e.vx, y - e.vy);
                    let t = ((wx * e.ex + wy * e.ey) / e.ee).clamp(0.0, 1.0);
                    let (bx, by) = (wx - e.ex * t, wy - e.ey * t);
                    d = d.min(bx * bx + by * by);
                    let c1 = y >= e.vy;
                    let c2 = y < e.vj_y;
                    let c3 = e.ex * wy > e.ey * wx;
                    if (c1 && c2 && c3) || (!c1 && !c2 && !c3) {
                        s = -s;
                    }
                }
                s * d.sqrt()
            }
        }
    }
}

/// Length of `(dx, dy)` from basic operations (`hypot` is libm's, and differs between systems in its last bit).
#[inline(always)]
fn dist(dx: f32, dy: f32) -> f32 {
    (dx * dx + dy * dy).sqrt()
}

/// Union bounds of a list of primitives.
pub(crate) fn bounds_of(prims: &[Prim]) -> [f32; 4] {
    prims.iter().map(Prim::bounds).fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |a, b| {
        [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
    })
}

#[inline(always)]
fn sdf_union_prepared(prims: &[PreparedPrim], x: f32, y: f32) -> f32 {
    if prims.len() == 1 { prims[0].sdf(x, y) } else { prims.iter().fold(f32::MAX, |d, p| d.min(p.sdf(x, y))) }
}

#[inline(always)]
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

    /// Reuses the canvas backing allocation for a new `(w, h)` image (caller paints every pixel via
    /// [`Background::paint`]).
    pub(crate) fn resize_for_paint(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.px.resize(w * h, [0.0; 3]);
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
            let row = &self.px[y * self.w..];
            for x in (x0..x1).step_by(step) {
                let p = row[x];
                acc[0] += p[0];
                acc[1] += p[1];
                acc[2] += p[2];
                n += 1.0;
            }
        }
        if n == 0.0 { [0.5; 3] } else { acc.map(|v| v / n) }
    }

    /// Paints the union of `prims` with anti-aliased edges, a soft drop shadow, light shading and a rim.
    pub(crate) fn draw(&mut self, prims: &[Prim], paint: &Paint) {
        let prepared: Vec<PreparedPrim> = prims.iter().map(PreparedPrim::from_prim).collect();
        let b = bounds_of(prims);
        let size = (b[2] - b[0]).max(b[3] - b[1]).max(1.0);
        let centre = [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5];
        let ([sx, sy], alpha) = paint.shadow;
        if alpha > 0.0 {
            let blur = (size * 0.05).clamp(1.0, 8.0);
            let sb = [b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy];
            let (x0, y0, x1, y1) = self.window(sb, blur + 1.0);
            for y in y0..y1 {
                let fy = y as f32 + 0.5 - sy;
                let row = &mut self.px[y * self.w + x0..y * self.w + x1];
                for (dx, p) in row.iter_mut().enumerate() {
                    let d = sdf_union_prepared(&prepared, (x0 + dx) as f32 + 0.5 - sx, fy);
                    let a = alpha * smoothstep(blur, -blur, d);
                    if a > 0.0 {
                        let k = 1.0 - a;
                        p[0] *= k;
                        p[1] *= k;
                        p[2] *= k;
                    }
                }
            }
        }
        let (x0, y0, x1, y1) = self.window(b, 1.0);
        let rim_w = (size * 0.04).clamp(0.8, 3.0);
        let inv = 2.0 / size;
        let col = paint.color;
        for y in y0..y1 {
            let fy = y as f32 + 0.5;
            let dy_c = fy - centre[1];
            let row = &mut self.px[y * self.w + x0..y * self.w + x1];
            for (dx, p) in row.iter_mut().enumerate() {
                let fx = (x0 + dx) as f32 + 0.5;
                let d = sdf_union_prepared(&prepared, fx, fy);
                let cov = (0.5 - d).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                // Light from the top left: brighter on that side.
                let t = (-(fx - centre[0]) - dy_c) * inv * std::f32::consts::FRAC_1_SQRT_2;
                let light = 1.0 + paint.shade * t.clamp(-1.0, 1.0);
                let rim = 1.0 - paint.rim * smoothstep(-rim_w, 0.0, d);
                let k = light * rim;
                let v0 = (col[0] * k).clamp(0.0, 1.0);
                let v1 = (col[1] * k).clamp(0.0, 1.0);
                let v2 = (col[2] * k).clamp(0.0, 1.0);
                p[0] += (v0 - p[0]) * cov;
                p[1] += (v1 - p[1]) * cov;
                p[2] += (v2 - p[2]) * cov;
            }
        }
    }

    /// Quantises to 8-bit with per-pixel grain of amplitude `grain` (in 0..255 levels) seeded by `seed`.
    pub(crate) fn to_image(&self, grain: f32, seed: u64) -> Image {
        let len =
            self.w.checked_mul(self.h).and_then(|n| n.checked_mul(3)).expect("synthetic image size overflow");
        let mut data = buffers::scratch(len);
        let mut rng = Rng::new(seed);
        for (o, p) in data.as_chunks_mut::<3>().0.iter_mut().zip(&self.px) {
            // Triangular-ish noise from two uniforms, shared across channels (luma grain) plus a little chroma.
            let n = (rng.f32() + rng.f32() - 1.0) * grain;
            let c0 = (rng.f32() - 0.5) * grain * 0.3;
            let c1 = (rng.f32() - 0.5) * grain * 0.3;
            let c2 = (rng.f32() - 0.5) * grain * 0.3;
            o[0] = (p[0] * 255.0 + n + c0).round().clamp(0.0, 255.0) as u8;
            o[1] = (p[1] * 255.0 + n + c1).round().clamp(0.0, 255.0) as u8;
            o[2] = (p[2] * 255.0 + n + c2).round().clamp(0.0, 255.0) as u8;
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

/// Value noise lattice for one octave, with precomputed horizontal `(ix, sx)` taps.
struct Lattice {
    gw: usize,
    vals: Vec<f32>,
    cell: f32,
    x_taps: Vec<(usize, f32)>,
}

impl Lattice {
    fn new(w: usize, h: usize, cell: f32, seed: u64) -> Self {
        let gw = (w as f32 / cell).ceil() as usize + 2;
        let gh = (h as f32 / cell).ceil() as usize + 2;
        let mut rng = Rng::new(seed);
        let vals = (0..gw * gh).map(|_| rng.f32()).collect();
        let x_taps = (0..w)
            .map(|x| {
                let gx = x as f32 / cell;
                let ix = gx as usize;
                let fx = gx - ix as f32;
                let sx = fx * fx * (3.0 - 2.0 * fx);
                (ix, sx)
            })
            .collect();
        Lattice { gw, vals, cell, x_taps }
    }

    /// Accumulates `amp * sample(x, y)` into `row_out[x]` for every column `x` of row `y`, using the exact same
    /// IEEE-754 operations as per-pixel bilinear value noise.
    #[inline]
    fn add_row(&self, y: usize, amp: f32, row_out: &mut [f32]) {
        let gy = y as f32 / self.cell;
        let iy = gy as usize;
        let fy = gy - iy as f32;
        let sy = fy * fy * (3.0 - 2.0 * fy);
        let base = iy * self.gw;
        let top_row = &self.vals[base..base + self.gw];
        let bot_row = &self.vals[base + self.gw..base + 2 * self.gw];
        for (dst, &(ix, sx)) in row_out.iter_mut().zip(&self.x_taps) {
            let a = top_row[ix];
            let b = top_row[ix + 1];
            let c = bot_row[ix];
            let d = bot_row[ix + 1];
            let top = a + (b - a) * sx;
            let bot = c + (d - c) * sx;
            *dst += (top + (bot - top) * sy) * amp;
        }
    }
}

#[inline(always)]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

impl Background {
    /// Paints the background into a canvas, streaming fBm noise one row at a time with zero full-frame noise
    /// allocation.
    pub(crate) fn paint(&self, canvas: &mut Canvas) {
        let (w, h) = (canvas.w, canvas.h);
        if w == 0 || h == 0 {
            return;
        }
        let diag = ((w * w + h * h) as f32).sqrt();
        let half_diag = diag * 0.5;
        let (sa, ca) = math::sin_cos(self.angle);
        let (octaves, cell) = match self.kind {
            3 => (3, self.scale * 0.15),
            _ => (4, self.scale),
        };
        let base_cell = cell.max(3.0);
        let lattices: Vec<Lattice> = (0..octaves)
            .map(|o| {
                Lattice::new(w, h, (base_cell / (1 << o) as f32).max(2.0), mix_seed(&[self.seed, o as u64]))
            })
            .collect();
        let mut norm = 0.0f32;
        let mut amp = 1.0f32;
        let mut amps = [0.0f32; 4];
        for a in amps.iter_mut().take(octaves) {
            *a = amp;
            norm += amp;
            amp *= 0.5;
        }
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let mut row_noise = vec![0.0f32; w];
        let c0 = self.colors[0];
        let c1 = self.colors[1];
        let c2 = self.colors[2];
        let strength = self.strength;
        let half_strength = 0.5 * strength;
        let stripe_freq = self.stripe_freq;
        let vignette = self.vignette;

        for y in 0..h {
            row_noise.fill(0.0);
            for (l, &a) in lattices.iter().zip(&amps[..octaves]) {
                l.add_row(y, a, &mut row_noise);
            }
            let fy = y as f32 - cy;
            let fy_sa = fy * sa;
            let fy_ca = fy * ca;
            let fy_sq = fy * fy;
            let row = &mut canvas.px[y * w..(y + 1) * w];
            for (x, (dst, &v)) in row.iter_mut().zip(&row_noise).enumerate() {
                let fx = x as f32 - cx;
                let g = ((fx * ca + fy_sa) / diag + 0.5).clamp(0.0, 1.0);
                let n = v / norm;
                let mut c = lerp3(c0, c1, g);
                match self.kind {
                    // Smooth gradient with a light cloud texture.
                    0 => c = lerp3(c, c2, (n - 0.5) * strength + half_strength),
                    // Marbled: noise drives a strong mix of three colours.
                    1 => {
                        let t = smoothstep(0.25, 0.75, n);
                        c = lerp3(lerp3(c0, c2, t), c1, g * 0.5);
                    }
                    // Soft stripes warped by noise.
                    2 => {
                        let phase = (fx * -sa + fy_ca) * stripe_freq + n * 6.0;
                        let s = 0.5 + 0.5 * math::sin_cos(phase).0;
                        c = lerp3(c, c2, s * strength);
                    }
                    // Fine fabric / paper grain.
                    _ => {
                        let t = (n - 0.5) * 2.0 * strength;
                        let scale = 1.0 + t;
                        c = [c[0] * scale, c[1] * scale, c[2] * scale];
                    }
                }
                if vignette > 0.0 {
                    let r = (fx * fx + fy_sq).sqrt() / half_diag;
                    let k = 1.0 - vignette * r * r;
                    c = [c[0] * k, c[1] * k, c[2] * k];
                }
                *dst = [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)];
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

    fn sd_polygon_ref(v: &[[f32; 2]], x: f32, y: f32) -> f32 {
        let n = v.len();
        let (dx, dy) = (x - v[0][0], y - v[0][1]);
        let mut d = dx * dx + dy * dy;
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

    #[test]
    fn polygon_sdf_signs() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let prep = PreparedPrim::from_prim(&Prim::Poly(sq.clone()));
        for (x, y) in [(5.0, 5.0), (15.0, 5.0), (13.0, 14.0), (0.0, 10.0), (3.25, 7.75)] {
            assert_eq!(prep.sdf(x, y).to_bits(), sd_polygon_ref(&sq, x, y).to_bits());
        }
        assert!((prep.sdf(5.0, 5.0) + 5.0).abs() < 1e-5);
        assert!((prep.sdf(15.0, 5.0) - 5.0).abs() < 1e-5);
        assert!((prep.sdf(13.0, 14.0) - 5.0).abs() < 1e-5);
        let p = PreparedPrim::from_prim(&Prim::Annulus { c: [0.0, 0.0], mid: 5.0, half: 1.0 });
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

    #[test]
    fn streaming_lattice_row_matches_direct_sample_bit_exact() {
        let (w, h, cell) = (53, 37, 7.5f32);
        let lat = Lattice::new(w, h, cell, 42);
        let mut row = vec![0.0f32; w];
        for y in 0..h {
            row.fill(0.0);
            lat.add_row(y, 0.5, &mut row);
            for (x, &got) in row.iter().enumerate() {
                let (gx, gy) = (x as f32 / cell, y as f32 / cell);
                let (ix, iy) = (gx as usize, gy as usize);
                let (fx, fy) = (gx - ix as f32, gy - iy as f32);
                let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
                let i = iy * lat.gw + ix;
                let (a, b, c, d) =
                    (lat.vals[i], lat.vals[i + 1], lat.vals[i + lat.gw], lat.vals[i + lat.gw + 1]);
                let top = a + (b - a) * sx;
                let bot = c + (d - c) * sx;
                let want = (top + (bot - top) * sy) * 0.5;
                assert_eq!(got.to_bits(), want.to_bits(), "mismatch at ({x}, {y})");
            }
        }
    }
}
