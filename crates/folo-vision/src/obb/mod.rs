//! Exact Oriented Bounding Box (OBB) geometry, Sutherland–Hodgman convex polygon intersection,
//! exact rotated IoU, and rotated Non-Maximum Suppression.
//!
//! Each [`Obb`] is parameterised as `(cx, cy, w, h, theta_rad)` where `theta_rad` is the
//! counter-clockwise rotation angle in radians around `(cx, cy)`.
//!
//! **Exactness & numerical stability.** Intersection area and IoU execute entirely in `f64` with
//! zero heap allocations (stack-allocated 16-vertex clipping buffers; two intersecting rectangles
//! have at most 8 vertices). Before clipping:
//! 1. Non-finite or zero/negative-extent boxes immediately return `0.0` (never `NaN`).
//! 2. A canonical `(a, b)` ordering guarantees bit-for-bit symmetry `obb_iou(a, b) == obb_iou(b, a)`.
//! 3. Disjoint bounding circles immediately short-circuit to `0.0`.
//! 4. Coordinates are translated into `a`'s rotated local frame and normalized by `max(w1, h1, w2, h2)`,
//!    putting every vertex in `O(1)` range across arbitrary scales (`1e-25` to `1e30`) and turning
//!    Sutherland–Hodgman clipping against `a` into four exact axis-aligned half-plane clips immune to
//!    collinear-edge and `pi/2`-rotation degeneracy.

mod nms;
#[cfg(test)]
mod tests;

pub use nms::{ObbDetection, RotatedNmsConfig, batched_rotated_nms, rotated_nms, rotated_nms_detections};

use crate::boxes::Xyxy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Element count above which pairwise OBB IoU matrix computation switches to rayon.
const PAR_THRESHOLD: usize = 1 << 12;

/// An oriented bounding box parameterised by centre `(cx, cy)`, size `(w, h)`, and counter-clockwise
/// rotation angle `theta_rad` (radians).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Obb {
    /// Centre x.
    pub cx: f32,
    /// Centre y.
    pub cy: f32,
    /// Box width along its local x-axis.
    pub w: f32,
    /// Box height along its local y-axis.
    pub h: f32,
    /// Counter-clockwise rotation angle in radians.
    pub theta_rad: f32,
}

impl Obb {
    /// Construct an oriented bounding box `(cx, cy, w, h, theta_rad)`.
    #[inline]
    pub const fn new(cx: f32, cy: f32, w: f32, h: f32, theta_rad: f32) -> Self {
        Self { cx, cy, w, h, theta_rad }
    }

    /// Construct from `[cx, cy, w, h, theta_rad]`.
    #[inline]
    pub const fn from_array(a: [f32; 5]) -> Self {
        Self::new(a[0], a[1], a[2], a[3], a[4])
    }

    /// Convert to `[cx, cy, w, h, theta_rad]`.
    #[inline]
    pub const fn to_array(self) -> [f32; 5] {
        [self.cx, self.cy, self.w, self.h, self.theta_rad]
    }

    /// Construct an axis-aligned (`theta_rad == 0`) [`Obb`] from an [`Xyxy`] box.
    #[inline]
    pub fn from_xyxy(b: Xyxy) -> Self {
        let c = b.to_cxcywh();
        Self::new(c.cx, c.cy, c.w.max(0.0), c.h.max(0.0), 0.0)
    }

    /// Whether every parameter is finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.cx.is_finite()
            && self.cy.is_finite()
            && self.w.is_finite()
            && self.h.is_finite()
            && self.theta_rad.is_finite()
    }

    /// Area in `f64` (`0.0` for non-finite or non-positive extent).
    #[inline]
    pub fn area_f64(&self) -> f64 {
        if !self.is_finite() || self.w <= 0.0 || self.h <= 0.0 {
            return 0.0;
        }
        (self.w as f64) * (self.h as f64)
    }

    /// Area in `f32` (`0.0` for non-finite or non-positive extent).
    #[inline]
    pub fn area(&self) -> f32 {
        self.area_f64() as f32
    }

    /// Four corners in counter-clockwise order in `f64`:
    /// `(-w/2, -h/2)`, `(+w/2, -h/2)`, `(+w/2, +h/2)`, `(-w/2, +h/2)` rotated by `theta_rad` around `(cx, cy)`.
    #[inline]
    pub fn corners_f64(&self) -> [[f64; 2]; 4] {
        let cx = self.cx as f64;
        let cy = self.cy as f64;
        let hw = 0.5 * (self.w as f64).max(0.0);
        let hh = 0.5 * (self.h as f64).max(0.0);
        let (s, c) = (self.theta_rad as f64).sin_cos();
        let dx_w = hw * c;
        let dy_w = hw * s;
        let dx_h = -hh * s;
        let dy_h = hh * c;
        [
            [cx - dx_w - dx_h, cy - dy_w - dy_h],
            [cx + dx_w - dx_h, cy + dy_w - dy_h],
            [cx + dx_w + dx_h, cy + dy_w + dy_h],
            [cx - dx_w + dx_h, cy - dy_w + dy_h],
        ]
    }

    /// Four corners in counter-clockwise order in `f32`.
    #[inline]
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let c = self.corners_f64();
        [
            [c[0][0] as f32, c[0][1] as f32],
            [c[1][0] as f32, c[1][1] as f32],
            [c[2][0] as f32, c[2][1] as f32],
            [c[3][0] as f32, c[3][1] as f32],
        ]
    }

    /// Tight axis-aligned bounding box enclosing this [`Obb`] in `f64` as `[min_x, min_y, max_x, max_y]`.
    #[inline]
    pub(crate) fn aabb_f64(&self) -> [f64; 4] {
        let cx = self.cx as f64;
        let cy = self.cy as f64;
        let hw = 0.5 * (self.w as f64).max(0.0);
        let hh = 0.5 * (self.h as f64).max(0.0);
        let (s, c) = (self.theta_rad as f64).sin_cos();
        let ex = hw * c.abs() + hh * s.abs();
        let ey = hw * s.abs() + hh * c.abs();
        [cx - ex, cy - ey, cx + ex, cy + ey]
    }

    /// Tight axis-aligned bounding box enclosing this [`Obb`].
    #[inline]
    pub fn bounding_xyxy(&self) -> Xyxy {
        let [x1, y1, x2, y2] = self.aabb_f64();
        Xyxy::new(x1 as f32, y1 as f32, x2 as f32, y2 as f32)
    }
}

/// Unsigned area of a simple polygon given as vertices in order, via the shoelace formula in `f64`.
#[inline]
pub fn convex_polygon_area_f64(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut prev = poly[n - 1];
    for &cur in poly {
        sum += prev[0] * cur[1] - cur[0] * prev[1];
        prev = cur;
    }
    0.5 * sum.abs()
}

/// General Sutherland–Hodgman convex polygon intersection in `f64`: clips `subject` (CCW) against
/// every directed edge of `clipper` (CCW), returning the intersection polygon vertices (empty if
/// disjoint or degenerate).
pub fn clip_convex_polygon_f64(subject: &[[f64; 2]], clipper: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if subject.len() < 3 || clipper.len() < 3 {
        return Vec::new();
    }
    let mut poly = subject.to_vec();
    let mut next = Vec::with_capacity(subject.len() + clipper.len());
    let m = clipper.len();

    for i in 0..m {
        if poly.is_empty() {
            break;
        }
        let a = clipper[i];
        let b = clipper[(i + 1) % m];
        let ex = b[0] - a[0];
        let ey = b[1] - a[1];
        next.clear();

        let mut prev = *poly.last().expect("poly is non-empty");
        let mut prev_side = ex * (prev[1] - a[1]) - ey * (prev[0] - a[0]);

        for &cur in &poly {
            let cur_side = ex * (cur[1] - a[1]) - ey * (cur[0] - a[0]);
            let prev_in = prev_side >= 0.0;
            let cur_in = cur_side >= 0.0;

            if cur_in {
                if !prev_in {
                    let denom = prev_side - cur_side;
                    if denom.abs() > 0.0 {
                        let t = (prev_side / denom).clamp(0.0, 1.0);
                        next.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
                    }
                }
                next.push(cur);
            } else if prev_in {
                let denom = prev_side - cur_side;
                if denom.abs() > 0.0 {
                    let t = (prev_side / denom).clamp(0.0, 1.0);
                    next.push([prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])]);
                }
            }
            prev = cur;
            prev_side = cur_side;
        }
        std::mem::swap(&mut poly, &mut next);
    }
    poly
}

/// Stack-allocated Sutherland–Hodgman half-plane clip for a single axis-aligned boundary:
/// keeps points with `sign * p[axis] <= sign * bound` (for `sign = +1` keeps `<= bound`; for `sign = -1` keeps `>= bound`).
#[inline]
fn clip_axis_halfplane(
    src: &[[f64; 2]; 16],
    src_len: usize,
    dst: &mut [[f64; 2]; 16],
    axis: usize,
    bound: f64,
    keep_le: bool,
) -> usize {
    if src_len == 0 {
        return 0;
    }
    let other = 1 - axis;
    let signed_dist = |p: [f64; 2]| if keep_le { bound - p[axis] } else { p[axis] - bound };

    let mut out_len = 0usize;
    let mut prev = src[src_len - 1];
    let mut prev_d = signed_dist(prev);

    for &cur in &src[..src_len] {
        let cur_d = signed_dist(cur);
        let prev_in = prev_d >= 0.0;
        let cur_in = cur_d >= 0.0;

        if cur_in {
            if !prev_in {
                let denom = prev_d - cur_d;
                if denom != 0.0 {
                    let t = (prev_d / denom).clamp(0.0, 1.0);
                    let mut pt = [0.0; 2];
                    pt[axis] = bound;
                    pt[other] = prev[other] + t * (cur[other] - prev[other]);
                    dst[out_len] = pt;
                    out_len += 1;
                }
            }
            dst[out_len] = cur;
            out_len += 1;
        } else if prev_in {
            let denom = prev_d - cur_d;
            if denom != 0.0 {
                let t = (prev_d / denom).clamp(0.0, 1.0);
                let mut pt = [0.0; 2];
                pt[axis] = bound;
                pt[other] = prev[other] + t * (cur[other] - prev[other]);
                dst[out_len] = pt;
                out_len += 1;
            }
        }
        prev = cur;
        prev_d = cur_d;
    }
    out_len
}

/// Canonical total ordering on `Obb` so `(a, b)` and `(b, a)` produce bitwise identical results.
#[inline]
fn canonical_pair<'a>(a: &'a Obb, b: &'a Obb) -> (&'a Obb, &'a Obb) {
    let ord =
        a.w.total_cmp(&b.w)
            .then_with(|| a.h.total_cmp(&b.h))
            .then_with(|| a.cx.total_cmp(&b.cx))
            .then_with(|| a.cy.total_cmp(&b.cy))
            .then_with(|| a.theta_rad.total_cmp(&b.theta_rad));
    if ord.is_gt() { (b, a) } else { (a, b) }
}

/// Core normalized intersection & union areas in `f64` (divided by `scale^2`, where `scale = max(w1, h1, w2, h2)`).
/// Returns `(inter_norm, area_a_norm, area_b_norm, scale)`.
#[inline]
fn obb_overlap_normalized_f64(a: &Obb, b: &Obb) -> (f64, f64, f64, f64) {
    if !a.is_finite() || !b.is_finite() || a.w <= 0.0 || a.h <= 0.0 || b.w <= 0.0 || b.h <= 0.0 {
        return (0.0, 0.0, 0.0, 1.0);
    }
    let (a, b) = canonical_pair(a, b);
    let w1 = a.w as f64;
    let h1 = a.h as f64;
    let w2 = b.w as f64;
    let h2 = b.h as f64;
    let scale = w1.max(h1).max(w2).max(h2);
    let inv_s = 1.0 / scale;

    let sw1 = w1 * inv_s;
    let sh1 = h1 * inv_s;
    let sw2 = w2 * inv_s;
    let sh2 = h2 * inv_s;
    let area_a = sw1 * sh1;
    let area_b = sw2 * sh2;

    // Bounding-circle fast reject in normalized coordinates.
    let dx = (b.cx as f64 - a.cx as f64) * inv_s;
    let dy = (b.cy as f64 - a.cy as f64) * inv_s;
    let r1 = 0.5 * sw1.hypot(sh1);
    let r2 = 0.5 * sw2.hypot(sh2);
    let r_sum = r1 + r2;
    if dx * dx + dy * dy >= r_sum * r_sum {
        return (0.0, area_a, area_b, scale);
    }

    // Transform `b` into `a`'s local orthonormal frame where `a` is `[-hw1, hw1] x [-hh1, hh1]`.
    let (sin_a, cos_a) = (a.theta_rad as f64).sin_cos();
    let lcx = dx * cos_a + dy * sin_a;
    let lcy = -dx * sin_a + dy * cos_a;
    let dtheta = (b.theta_rad as f64) - (a.theta_rad as f64);
    let (sin_d, cos_d) = dtheta.sin_cos();

    let hw1 = 0.5 * sw1;
    let hh1 = 0.5 * sh1;
    let hw2 = 0.5 * sw2;
    let hh2 = 0.5 * sh2;

    // Fast AABB reject in `a`'s local frame.
    let ex2 = hw2 * cos_d.abs() + hh2 * sin_d.abs();
    let ey2 = hw2 * sin_d.abs() + hh2 * cos_d.abs();
    if lcx + ex2 <= -hw1 || lcx - ex2 >= hw1 || lcy + ey2 <= -hh1 || lcy - ey2 >= hh1 {
        return (0.0, area_a, area_b, scale);
    }

    let dx_w = hw2 * cos_d;
    let dy_w = hw2 * sin_d;
    let dx_h = -hh2 * sin_d;
    let dy_h = hh2 * cos_d;

    let mut buf_a = [[0.0f64; 2]; 16];
    let mut buf_b = [[0.0f64; 2]; 16];
    buf_a[0] = [lcx - dx_w - dx_h, lcy - dy_w - dy_h];
    buf_a[1] = [lcx + dx_w - dx_h, lcy + dy_w - dy_h];
    buf_a[2] = [lcx + dx_w + dx_h, lcy + dy_w + dy_h];
    buf_a[3] = [lcx - dx_w + dx_h, lcy - dy_w + dy_h];

    let n1 = clip_axis_halfplane(&buf_a, 4, &mut buf_b, 0, -hw1, false);
    let n2 = clip_axis_halfplane(&buf_b, n1, &mut buf_a, 0, hw1, true);
    let n3 = clip_axis_halfplane(&buf_a, n2, &mut buf_b, 1, -hh1, false);
    let n4 = clip_axis_halfplane(&buf_b, n3, &mut buf_a, 1, hh1, true);

    let inter = convex_polygon_area_f64(&buf_a[..n4]).clamp(0.0, area_a.min(area_b));
    (inter, area_a, area_b, scale)
}

/// Exact intersection area of two oriented bounding boxes in `f64` (`0.0` for disjoint, degenerate,
/// or non-finite boxes). Bit-for-bit symmetric in `a` and `b`.
#[inline]
pub fn obb_intersection_area_f64(a: &Obb, b: &Obb) -> f64 {
    let (inter_norm, _, _, scale) = obb_overlap_normalized_f64(a, b);
    inter_norm * (scale * scale)
}

/// Exact intersection area of two oriented bounding boxes in `f32`.
#[inline]
pub fn obb_intersection_area(a: &Obb, b: &Obb) -> f32 {
    obb_intersection_area_f64(a, b) as f32
}

/// Exact rotated Intersection-over-Union (IoU) in `[0.0, 1.0]` computed in `f64` via
/// Sutherland–Hodgman convex polygon clipping.
///
/// Always finite (never `NaN`), bit-for-bit symmetric in `a` and `b`, and returns `0.0` when either
/// box has non-finite fields or non-positive width/height.
#[inline]
pub fn obb_iou_f64(a: &Obb, b: &Obb) -> f64 {
    let (inter, area_a, area_b, _) = obb_overlap_normalized_f64(a, b);
    if inter <= 0.0 {
        return 0.0;
    }
    let union = area_a + area_b - inter;
    if union > 0.0 { (inter / union).clamp(0.0, 1.0) } else { 0.0 }
}

/// Exact rotated IoU in `[0.0, 1.0]` as `f32`.
#[inline]
pub fn obb_iou(a: &Obb, b: &Obb) -> f32 {
    obb_iou_f64(a, b) as f32
}

/// Pairwise rotated IoU matrix, row-major `[a.len(), b.len()]`: `out[i * b.len() + j] = obb_iou(&a[i], &b[j])`.
pub fn obb_iou_matrix(a: &[Obb], b: &[Obb]) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len() * b.len()];
    if b.is_empty() {
        return out;
    }
    let aabbs_b: Vec<[f64; 4]> = b.iter().map(Obb::aabb_f64).collect();
    let compute_row = |(ra, orow): (&Obb, &mut [f32])| {
        if !ra.is_finite() || ra.w <= 0.0 || ra.h <= 0.0 {
            return;
        }
        let ab_a = ra.aabb_f64();
        for ((o, rb), ab_b) in orow.iter_mut().zip(b).zip(&aabbs_b) {
            if ab_a[2] <= ab_b[0] || ab_b[2] <= ab_a[0] || ab_a[3] <= ab_b[1] || ab_b[3] <= ab_a[1] {
                *o = 0.0;
            } else {
                *o = obb_iou(ra, rb);
            }
        }
    };
    if out.len() >= PAR_THRESHOLD {
        a.par_iter().zip(out.par_chunks_mut(b.len())).for_each(compute_row);
    } else {
        a.iter().zip(out.chunks_mut(b.len())).for_each(compute_row);
    }
    out
}
