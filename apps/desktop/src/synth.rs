//! Procedural "shapes" scenes: the images behind the in-app mock backend.
//!
//! Mirrors the engine's synthetic datasets (circles / squares / triangles / stars / hexagons on
//! textured backgrounds) so the UI can show real pixels without a server. A scene is described in
//! normalised coordinates and rendered at any size, so thumbnails and full-size views agree.

use crate::api::types::LabelBox as BBox;

/// SplitMix64: tiny, fast, good enough for deterministic fixtures.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    pub fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n.max(1) as u64) as u32
    }

    /// Approximately normal (sum of uniforms), mean 0, sd 1.
    pub fn normal(&mut self) -> f32 {
        (0..4).map(|_| self.f32()).sum::<f32>() * 1.732 - 3.464
    }
}

/// Stable 64-bit hash for string-keyed seeds.
pub fn hash(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01B3))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Circle,
    Square,
    Triangle,
    Star,
    Hexagon,
}

impl Kind {
    pub fn for_class(class: u32) -> Kind {
        match class % 5 {
            0 => Kind::Circle,
            1 => Kind::Square,
            2 => Kind::Triangle,
            3 => Kind::Star,
            _ => Kind::Hexagon,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub class: u32,
    pub kind: Kind,
    /// Centre and half-size, normalised to the image (0..1).
    pub cx: f32,
    pub cy: f32,
    pub r: f32,
    pub rot: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub seed: u64,
    pub bg_a: [f32; 3],
    pub bg_b: [f32; 3],
    pub grain: f32,
    pub shapes: Vec<Shape>,
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as i32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

impl Scene {
    /// A scene with up to `max_objects` shapes of `classes` classes. `small` biases sizes down.
    pub fn generate(seed: u64, classes: u32, max_objects: u32, small: bool) -> Scene {
        let mut rng = Rng::new(seed);
        let hue = rng.f32();
        let bg_a = hsv(hue, rng.range(0.15, 0.45), rng.range(0.18, 0.42));
        let bg_b = hsv(hue + rng.range(-0.08, 0.08), rng.range(0.1, 0.4), rng.range(0.30, 0.62));
        let n = if rng.f32() < 0.03 { 0 } else { 1 + rng.below(max_objects) };
        let mut shapes = Vec::new();
        for _ in 0..n {
            let class = rng.below(classes);
            let r = if small { rng.range(0.03, 0.13) } else { rng.range(0.05, 0.2) };
            shapes.push(Shape {
                class,
                kind: Kind::for_class(class),
                cx: rng.range(r * 0.6, 1.0 - r * 0.6),
                cy: rng.range(r * 0.6, 1.0 - r * 0.6),
                r,
                rot: rng.range(0.0, std::f32::consts::TAU),
                color: hsv(rng.f32(), rng.range(0.45, 0.9), rng.range(0.65, 0.98)),
            });
        }
        Scene { seed, bg_a, bg_b, grain: rng.range(0.02, 0.07), shapes }
    }

    /// Pixel `xyxy` boxes for an image of `w` × `h` (tight around each shape, clipped).
    pub fn boxes(&self, w: u32, h: u32) -> Vec<BBox> {
        let s = w.min(h) as f32;
        self.shapes
            .iter()
            .map(|sh| {
                let (ex, ey) = extent(sh);
                let (cx, cy) = (sh.cx * w as f32, sh.cy * h as f32);
                BBox {
                    class: sh.class,
                    x1: (cx - ex * sh.r * s).max(0.0).round(),
                    y1: (cy - ey * sh.r * s).max(0.0).round(),
                    x2: (cx + ex * sh.r * s).min(w as f32).round(),
                    y2: (cy + ey * sh.r * s).min(h as f32).round(),
                }
            })
            .collect()
    }

    /// Render to tightly packed RGBA8.
    pub fn render(&self, w: u32, h: u32) -> Vec<u8> {
        let (wf, hf) = (w as f32, h as f32);
        let s = w.min(h) as f32;
        let mut px = vec![[0f32; 3]; (w * h) as usize];
        // Background: diagonal gradient + soft value-noise texture.
        let mut noise_rng = Rng::new(self.seed ^ 0xA5A5);
        let cells = 9usize;
        let lattice: Vec<f32> = (0..(cells + 1) * (cells + 1)).map(|_| noise_rng.f32()).collect();
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / wf, y as f32 / hf);
                let t = (u * 0.6 + v * 0.4).clamp(0.0, 1.0);
                let (gx, gy) = (u * cells as f32, v * cells as f32);
                let (ix, iy) = ((gx as usize).min(cells - 1), (gy as usize).min(cells - 1));
                let (fx, fy) = (gx - ix as f32, gy - iy as f32);
                let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
                let l = |i: usize, j: usize| lattice[j * (cells + 1) + i];
                let n = (l(ix, iy) * (1.0 - sx) + l(ix + 1, iy) * sx) * (1.0 - sy)
                    + (l(ix, iy + 1) * (1.0 - sx) + l(ix + 1, iy + 1) * sx) * sy;
                let g = (n - 0.5) * 0.16;
                let p = &mut px[(y * w + x) as usize];
                for c in 0..3 {
                    p[c] = self.bg_a[c] + (self.bg_b[c] - self.bg_a[c]) * t + g;
                }
            }
        }
        // Shapes, painter's order, 2×2 supersampled edges with a soft drop shadow.
        for sh in &self.shapes {
            let (ex, ey) = extent(sh);
            let (cx, cy, r) = (sh.cx * wf, sh.cy * hf, sh.r * s);
            let x0 = ((cx - ex * r - 3.0).floor().max(0.0)) as u32;
            let x1 = ((cx + ex * r + 5.0).ceil().min(wf)) as u32;
            let y0 = ((cy - ey * r - 3.0).floor().max(0.0)) as u32;
            let y1 = ((cy + ey * r + 5.0).ceil().min(hf)) as u32;
            let shadow = (r * 0.06).max(1.0);
            let pts = outline(sh);
            for y in y0..y1 {
                for x in x0..x1 {
                    let mut cover = 0.0;
                    let mut shade = 0.0;
                    for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                        let (lx, ly) = ((x as f32 + ox - cx) / r, (y as f32 + oy - cy) / r);
                        if inside(sh.kind, &pts, lx, ly) {
                            cover += 0.25;
                        }
                        if inside(sh.kind, &pts, lx - shadow / r, ly - shadow / r) {
                            shade += 0.25;
                        }
                    }
                    let p = &mut px[(y * w + x) as usize];
                    if shade > 0.0 {
                        for v in p.iter_mut() {
                            *v *= 1.0 - 0.35 * shade;
                        }
                    }
                    if cover > 0.0 {
                        // Light from the top-left for a little depth.
                        let lit = 1.0 + 0.12 * ((cx - x as f32) + (cy - y as f32)) / (2.0 * r);
                        for c in 0..3 {
                            p[c] = p[c] * (1.0 - cover) + (sh.color[c] * lit) * cover;
                        }
                    }
                }
            }
        }
        // Grain, then quantise.
        let mut grain = Rng::new(self.seed ^ 0x5EED);
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for p in px {
            let g = (grain.f32() - 0.5) * self.grain;
            for v in p {
                out.push(((v + g).clamp(0.0, 1.0) * 255.0) as u8);
            }
            out.push(255);
        }
        out
    }
}

/// Half-extent of a shape's rotated outline in units of `r`, per axis.
fn extent(sh: &Shape) -> (f32, f32) {
    match sh.kind {
        Kind::Circle => (1.0, 1.0),
        _ => {
            let pts = outline(sh);
            let ex = pts.iter().fold(0f32, |m, p| m.max(p.0.abs()));
            let ey = pts.iter().fold(0f32, |m, p| m.max(p.1.abs()));
            (ex, ey)
        }
    }
}

fn outline(sh: &Shape) -> Vec<(f32, f32)> {
    let poly = |n: usize, radius: &dyn Fn(usize) -> f32, phase: f32| -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let a = sh.rot + phase + i as f32 * std::f32::consts::TAU / n as f32;
                let rr = radius(i);
                (a.cos() * rr, a.sin() * rr)
            })
            .collect()
    };
    match sh.kind {
        Kind::Circle => Vec::new(),
        Kind::Square => poly(4, &|_| 1.0, std::f32::consts::FRAC_PI_4),
        Kind::Triangle => poly(3, &|_| 1.0, -std::f32::consts::FRAC_PI_2),
        Kind::Star => poly(10, &|i| if i % 2 == 0 { 1.0 } else { 0.45 }, -std::f32::consts::FRAC_PI_2),
        Kind::Hexagon => poly(6, &|_| 1.0, 0.0),
    }
}

/// Point-in-shape test in shape-local units; `pts` is the shape's precomputed `outline`.
fn inside(kind: Kind, pts: &[(f32, f32)], x: f32, y: f32) -> bool {
    match kind {
        Kind::Circle => x * x + y * y <= 1.0,
        _ => {
            // Even-odd rule against the rotated outline.
            let mut odd = false;
            let mut j = pts.len() - 1;
            for i in 0..pts.len() {
                let (xi, yi) = pts[i];
                let (xj, yj) = pts[j];
                if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                    odd = !odd;
                }
                j = i;
            }
            odd
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_are_deterministic() {
        let a = Scene::generate(42, 3, 6, false);
        let b = Scene::generate(42, 3, 6, false);
        assert_eq!(a.render(32, 24), b.render(32, 24));
        assert_eq!(a.boxes(640, 480), b.boxes(640, 480));
    }

    #[test]
    fn boxes_stay_inside_the_image() {
        for seed in 0..50 {
            let s = Scene::generate(seed, 5, 8, seed % 2 == 0);
            for b in s.boxes(640, 480) {
                assert!(b.x1 >= 0.0 && b.y1 >= 0.0 && b.x2 <= 640.0 && b.y2 <= 480.0);
                assert!(b.x2 > b.x1 && b.y2 > b.y1);
            }
        }
    }

    #[test]
    fn render_has_expected_size() {
        let s = Scene::generate(7, 3, 4, false);
        assert_eq!(s.render(17, 9).len(), 17 * 9 * 4);
    }
}
