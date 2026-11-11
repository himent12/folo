// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (shadow_fragment, erf,
// blur_along_x, gaussian)
//! Box shadows: GPUI's analytic blurred rounded rectangle, computed once into a cached alpha mask.
//! Rows in the middle of the shadow are identical, so the mask stores them once.

use super::blend::{ONE, Tint};
use super::geom::IRect;
use super::quad_eval::pick_corner_radius;
use super::target::Target;

/// HLSL's `M_PI_F` (3.141592653f) is this `f32`.
const M_PI_F: f32 = std::f32::consts::PI;

/// What a shadow mask depends on (all as `f32` bits): size, radii, blur, and the offset of the mask's
/// first pixel from the box centre; plus the mask's size in pixels.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ShadowKey {
    pub bits: [u32; 9],
    pub w: i32,
    pub h: i32,
}

/// A shadow's alpha (`0..=65535`) over a pixel rectangle, rows deduplicated.
pub(crate) struct ShadowMask {
    w: usize,
    /// Unique rows, `w` values each.
    rows: Vec<u16>,
    /// For each pixel row, its index in `rows`.
    row_of: Vec<u32>,
}

impl ShadowMask {
    pub fn bytes(&self) -> usize {
        self.rows.len() * 2 + self.row_of.len() * 4 + 64
    }

    fn row(&self, j: usize) -> &[u16] {
        let r = self.row_of[j] as usize;
        &self.rows[r * self.w..(r + 1) * self.w]
    }
}

/// HLSL `sign`.
#[inline(always)]
fn sign(v: f32) -> f32 {
    if v > 0. {
        1.
    } else if v < 0. {
        -1.
    } else {
        0.
    }
}

/// The shader's erf approximation.
#[inline(always)]
fn erf(v: f32) -> f32 {
    let s = sign(v);
    let a = v.abs();
    let mut x = 1. + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    x *= x;
    s - s / (x * x)
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    (-(x * x) / (2. * sigma * sigma)).exp() / ((2. * M_PI_F).sqrt() * sigma)
}

/// The inputs of one mask, in the shader's terms.
pub(crate) struct ShadowShape {
    pub half: [f32; 2],
    /// top-left, top-right, bottom-right, bottom-left.
    pub radii: [f32; 4],
    pub blur: f32,
    /// Pixel centre of the mask's (0, 0) relative to the box centre, minus 0.5.
    pub offset: [f32; 2],
    pub w: i32,
    pub h: i32,
}

impl ShadowShape {
    pub fn key(&self) -> ShadowKey {
        let b = |v: f32| v.to_bits();
        ShadowKey {
            bits: [
                b(self.half[0]),
                b(self.half[1]),
                b(self.radii[0]),
                b(self.radii[1]),
                b(self.radii[2]),
                b(self.radii[3]),
                b(self.blur),
                b(self.offset[0]),
                b(self.offset[1]),
            ],
            w: self.w,
            h: self.h,
        }
    }

    /// Computes the mask (`shadow_fragment` at every pixel centre).
    pub fn render(&self) -> ShadowMask {
        let (w, h) = (self.w.max(0) as usize, self.h.max(0) as usize);
        let sigma = self.blur;
        let k = 0.5f32.sqrt() / sigma;
        let hs = self.half;
        let xs: Vec<f32> = (0..w).map(|i| i as f32 + 0.5 + self.offset[0]).collect();
        let mut rows: Vec<u16> = Vec::new();
        let mut row_of = Vec::with_capacity(h);
        let mut interior_row: Option<u32> = None;
        let mut alpha = vec![0f32; w];
        let max_r = self.radii.iter().fold(0f32, |m, &r| m.max(r));
        for j in 0..h {
            let py = j as f32 + 0.5 + self.offset[1];
            let low = py - hs[1];
            let high = py + hs[1];
            let start = (-3. * sigma).max(low).min(high);
            let end = (3. * sigma).max(low).min(high);
            let step = (end - start) / 4.;
            // Rows whose samples all sit on the straight sides: identical, so stored once.
            let interior = low <= -3. * sigma && high >= 3. * sigma && py.abs() + 3. * sigma <= hs[1] - max_r;
            if interior && let Some(r) = interior_row {
                row_of.push(r);
                continue;
            }
            let mut ys = [0f32; 4];
            let mut wt = [0f32; 4];
            let mut y = start + step * 0.5;
            for i in 0..4 {
                ys[i] = y;
                wt[i] = gaussian(y, sigma) * step;
                y += step;
            }
            // `curved` per sample, for each horizontal half's corner (the corner depends on the quadrant).
            let curved = |corner: f32| -> [f32; 4] {
                std::array::from_fn(|i| {
                    let delta = (hs[1] - corner - (py - ys[i]).abs()).min(0.);
                    hs[0] - corner + (corner * corner - delta * delta).max(0.).sqrt()
                })
            };
            let cl = curved(pick_corner_radius([-1., py], &self.radii));
            let cr = curved(pick_corner_radius([1., py], &self.radii));
            for (a, &x) in alpha.iter_mut().zip(&xs) {
                let c = if x < 0. { &cl } else { &cr };
                let mut sum = 0.;
                for i in 0..4 {
                    let lo = 0.5 + 0.5 * erf((x - c[i]) * k);
                    let hi = 0.5 + 0.5 * erf((x + c[i]) * k);
                    sum += (hi - lo) * wt[i];
                }
                *a = sum;
            }
            let idx = (rows.len() / w.max(1)) as u32;
            rows.extend(alpha.iter().map(|&a| (a.clamp(0., 1.) * 65535. + 0.5) as u16));
            if interior {
                interior_row = Some(idx);
            }
            row_of.push(idx);
        }
        ShadowMask { w, rows, row_of }
    }
}

/// Draws the part of `mask` (placed with its (0, 0) at `origin`) inside `r` in `tint` at `alpha`.
pub(crate) fn draw(t: &mut Target, mask: &ShadowMask, origin: (i32, i32), tint: &Tint, alpha: u32, r: IRect) {
    let x0 = (r.x0 - origin.0) as usize;
    let n = (r.x1 - r.x0) as usize;
    for y in r.y0..r.y1 {
        let src = &mask.row((y - origin.1) as usize)[x0..x0 + n];
        let row = t.row(y, r.x0, r.x1);
        for (p, &m) in row.iter_mut().zip(src) {
            if m != 0 {
                let a = (m as u32 * alpha + 32767) / 65535;
                if a != 0 {
                    *p = tint.over(*p, a.min(ONE));
                }
            }
        }
    }
}
