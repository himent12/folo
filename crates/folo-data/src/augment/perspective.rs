//! Projective (homography) image warp and box-aware [`random_perspective`] augmentation.
//!
//! Uses a 3×3 [`Homography`] with exact 11-bit bilinear interpolation matching [`crate::warp::warp_affine`],
//! behind-horizon guards (`w <= 0`), and stack-allocated Sutherland-Hodgman polygon clipping of each projected
//! bounding-box quadrilateral against `[0, width] × [0, height]` so partially clipped boxes get tight bounds and
//! off-screen boxes whose unclipped AABB overlaps the canvas never produce phantom detections.

use super::BoxFilter;
use crate::bbox::BBox;
use crate::buffers;
use crate::image::{Image, Sample};
use crate::letterbox::PAD_VALUE;
use crate::par::for_rows;
use crate::rng::Rng;
use crate::simd::assert_image_data;
use crate::warp::Affine;

const WBITS: u32 = 11;
const WONE: u32 = 1 << WBITS;
const WSHIFT: u32 = 2 * WBITS;
const WHALF: u32 = 1 << (WSHIFT - 1);
const HORIZON_EPS: f64 = 1e-12;

/// A 3×3 projective matrix in row-major order `[h00, h01, h02, h10, h11, h12, h20, h21, h22]` mapping `(x, y)`
/// to `(u / w, v / w)` with `u = h00*x + h01*y + h02`, `v = h10*x + h11*y + h12`, `w = h20*x + h21*y + h22`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography(pub [f64; 9]);

impl Default for Homography {
    fn default() -> Self {
        Homography::IDENTITY
    }
}

impl Homography {
    /// The identity projective transform.
    pub const IDENTITY: Homography = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Embeds a 2×3 [`Affine`] transform as a 3×3 [`Homography`] with bottom row `[0, 0, 1]`.
    pub fn from_affine(a: &Affine) -> Self {
        let [m00, m01, m02, m10, m11, m12] = a.0;
        Homography([m00, m01, m02, m10, m11, m12, 0.0, 0.0, 1.0])
    }

    /// `other ∘ self`: apply `self` first, then `other`.
    pub fn then(&self, other: &Homography) -> Homography {
        let a = &other.0;
        let b = &self.0;
        let mut out = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
            }
        }
        Homography(out)
    }

    /// Returns this matrix divided by its positive max-entry norm (`max_i |h_i|`), preserving both the exact
    /// projective transform and the sign of `w`, or `None` if all entries are zero or any entry is non-finite.
    pub fn normalized(&self) -> Option<Homography> {
        let mut scale = 0.0f64;
        for &v in &self.0 {
            if !v.is_finite() {
                return None;
            }
            scale = scale.max(v.abs());
        }
        if scale == 0.0 {
            return None;
        }
        let out = self.0.map(|v| v / scale);
        out.iter().all(|v| v.is_finite()).then_some(Homography(out))
    }

    /// Determinant of the 3×3 matrix.
    pub fn det(&self) -> f64 {
        let [a, b, c, d, e, f, g, h, i] = self.0;
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// The exact matrix inverse (scale-invariant singularity check on the max-norm normalized matrix), or `None`
    /// when singular or non-finite.
    pub fn inverse(&self) -> Option<Homography> {
        let mut scale = 0.0f64;
        for &v in &self.0 {
            if !v.is_finite() {
                return None;
            }
            scale = scale.max(v.abs());
        }
        if scale == 0.0 {
            return None;
        }
        let hn = Homography(self.0.map(|v| v / scale));
        let [a, b, c, d, e, f, g, h, i] = hn.0;
        let det = hn.det();
        let det_scale = (a * e * i)
            .abs()
            .max((a * f * h).abs())
            .max((b * d * i).abs())
            .max((b * f * g).abs())
            .max((c * d * h).abs())
            .max((c * e * g).abs());
        if !det.is_finite() || det_scale == 0.0 || det.abs() <= 1e-12 * det_scale {
            return None;
        }
        let inv_det = 1.0 / det;
        let out = [
            (e * i - f * h) * inv_det / scale,
            (c * h - b * i) * inv_det / scale,
            (b * f - c * e) * inv_det / scale,
            (f * g - d * i) * inv_det / scale,
            (a * i - c * g) * inv_det / scale,
            (c * d - a * f) * inv_det / scale,
            (d * h - e * g) * inv_det / scale,
            (b * g - a * h) * inv_det / scale,
            (a * e - b * d) * inv_det / scale,
        ];
        out.iter().all(|v| v.is_finite()).then_some(Homography(out))
    }

    /// Homogeneous evaluation `(u, v, w)` for point `(x, y)`.
    #[inline]
    pub fn apply_homogeneous(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let [h00, h01, h02, h10, h11, h12, h20, h21, h22] = self.0;
        (h00 * x + h01 * y + h02, h10 * x + h11 * y + h12, h20 * x + h21 * y + h22)
    }

    /// Projects `(x, y)` to `(u / w, v / w)`, returning `None` if `w` is on or behind the horizon (relative to
    /// the positive max-entry scale of `self`) or non-finite.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let hn = self.normalized()?;
        let (u, v, w) = hn.apply_homogeneous(x, y);
        if !w.is_finite() || w <= HORIZON_EPS {
            return None;
        }
        let (px, py) = (u / w, v / w);
        (px.is_finite() && py.is_finite()).then_some((px, py))
    }

    /// Whether this is numerically the identity (up to uniform positive scaling).
    pub fn is_identity(&self) -> bool {
        let Some(hn) = self.normalized() else {
            return false;
        };
        let s = hn.0[8];
        if s <= HORIZON_EPS {
            return false;
        }
        let inv = 1.0 / s;
        hn.0.iter().zip(Homography::IDENTITY.0).all(|(a, b)| (a * inv - b).abs() < 1e-9)
    }

    /// Solves for the unique homography (`h22 = 1`) mapping the 4 source points `src` to `dst`, or `None` if
    /// any three source or destination points are collinear / the system is ill-conditioned.
    pub fn from_quad_correspondence(src: [(f64, f64); 4], dst: [(f64, f64); 4]) -> Option<Homography> {
        let mut a = [[0.0f64; 9]; 8];
        for i in 0..4 {
            let (x, y) = src[i];
            let (u, v) = dst[i];
            if !(x.is_finite() && y.is_finite() && u.is_finite() && v.is_finite()) {
                return None;
            }
            a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        // Gaussian elimination with partial pivoting on the 8×8 linear system.
        for col in 0..8 {
            let mut pivot_row = col;
            let mut pivot_abs = a[col][col].abs();
            for (r, row) in a.iter().enumerate().skip(col + 1) {
                let v = row[col].abs();
                if v > pivot_abs {
                    pivot_abs = v;
                    pivot_row = r;
                }
            }
            if !pivot_abs.is_finite() || pivot_abs < 1e-12 {
                return None;
            }
            if pivot_row != col {
                a.swap(col, pivot_row);
            }
            let piv = a[col][col];
            for r in (col + 1)..8 {
                let factor = a[r][col] / piv;
                a[r][col] = 0.0;
                for c in (col + 1)..9 {
                    a[r][c] -= factor * a[col][c];
                }
            }
        }
        let mut sol = [0.0f64; 8];
        for r in (0..8).rev() {
            let mut sum = a[r][8];
            for c in (r + 1)..8 {
                sum -= a[r][c] * sol[c];
            }
            sol[r] = sum / a[r][r];
            if !sol[r].is_finite() {
                return None;
            }
        }
        let h = Homography([sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], 1.0]);
        // Reject singular homographies or solutions where any source corner falls on/behind the horizon.
        if h.inverse().is_none() || src.iter().any(|&(x, y)| h.apply(x, y).is_none()) {
            return None;
        }
        Some(h)
    }
}

/// Parameters for [`random_perspective`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveConfig {
    /// Probability of applying a random perspective warp (default `0.5`).
    pub prob: f32,
    /// Maximum corner perturbation as a fraction of the output width / height, in `[0, 0.45]` (default `0.15`).
    pub distortion: f32,
}

impl Default for PerspectiveConfig {
    fn default() -> Self {
        PerspectiveConfig { prob: 0.5, distortion: 0.15 }
    }
}

impl PerspectiveConfig {
    /// Checks parameter ranges; returns a message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("perspective.prob must be in [0, 1], got {}", self.prob));
        }
        if !(self.distortion.is_finite() && (0.0..=0.45).contains(&self.distortion)) {
            return Err(format!("perspective.distortion must be in [0, 0.45], got {}", self.distortion));
        }
        Ok(())
    }
}

/// Warps `src` by `h` (source → destination) into a `width × height` RGB image using bilinear sampling; pixels
/// outside the source or behind the projective horizon (`w <= 1e-12`) get `border`.
pub fn warp_perspective(src: &Image, h: &Homography, width: u32, height: u32, border: [u8; 3]) -> Image {
    assert_image_data(src);
    if width == 0 || height == 0 {
        return Image { width, height, data: Vec::new() };
    }
    let w = width as usize;
    let h_out = height as usize;
    let stride = w.checked_mul(3).expect("warp_perspective: image dimensions overflow");
    let total = h_out.checked_mul(stride).expect("warp_perspective: image dimensions overflow");
    let mut out = buffers::scratch(total);
    let Some(inv) = h.normalized().and_then(|hn| hn.inverse()).and_then(|im| im.normalized()) else {
        let mut img = Image { width, height, data: out };
        img.fill_rect(0, 0, width as i64, height as i64, border);
        return img;
    };
    let (sw, sh) = (src.width as i64, src.height as i64);
    if sw == 0 || sh == 0 {
        let mut img = Image { width, height, data: out };
        img.fill_rect(0, 0, width as i64, height as i64, border);
        return img;
    }
    let sstride = src.stride();
    let sdata = &src.data;
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = inv.0;

    for_rows(&mut out, stride, |y, row| {
        let yc = y as f64 + 0.5;
        let mut u = m00 * 0.5 + m01 * yc + m02;
        let mut v = m10 * 0.5 + m11 * yc + m12;
        let mut ww = m20 * 0.5 + m21 * yc + m22;
        for px in row.as_chunks_mut::<3>().0.iter_mut() {
            if !ww.is_finite() || ww <= HORIZON_EPS {
                *px = border;
            } else {
                let inv_w = 1.0 / ww;
                let sx = u * inv_w - 0.5;
                let sy = v * inv_w - 0.5;
                if !sx.is_finite()
                    || !sy.is_finite()
                    || sx < -1.0
                    || sy < -1.0
                    || sx >= sw as f64
                    || sy >= sh as f64
                {
                    *px = border;
                } else {
                    let fx = sx.floor();
                    let fy = sy.floor();
                    let ix = fx as i64;
                    let iy = fy as i64;
                    let wx = (((sx - fx) * WONE as f64).floor() as i64).clamp(0, WONE as i64 - 1) as u32;
                    let wy = (((sy - fy) * WONE as f64).floor() as i64).clamp(0, WONE as i64 - 1) as u32;
                    let (iwx, iwy) = (WONE - wx, WONE - wy);
                    if ix >= 0 && iy >= 0 && ix + 1 < sw && iy + 1 < sh {
                        let o = iy as usize * sstride + ix as usize * 3;
                        let r0 = &sdata[o..o + 6];
                        let r1 = &sdata[o + sstride..o + sstride + 6];
                        for ch in 0..3 {
                            let t = r0[ch] as u32 * iwx + r0[ch + 3] as u32 * wx;
                            let b = r1[ch] as u32 * iwx + r1[ch + 3] as u32 * wx;
                            px[ch] = ((t * iwy + b * wy + WHALF) >> WSHIFT) as u8;
                        }
                    } else {
                        let fetch = |x: i64, y: i64, ch: usize| -> u32 {
                            if x >= 0 && y >= 0 && x < sw && y < sh {
                                sdata[y as usize * sstride + x as usize * 3 + ch] as u32
                            } else {
                                border[ch] as u32
                            }
                        };
                        for (ch, slot) in px.iter_mut().enumerate() {
                            let t = fetch(ix, iy, ch) * iwx + fetch(ix + 1, iy, ch) * wx;
                            let b = fetch(ix, iy + 1, ch) * iwx + fetch(ix + 1, iy + 1, ch) * wx;
                            *slot = ((t * iwy + b * wy + WHALF) >> WSHIFT) as u8;
                        }
                    }
                }
            }
            u += m00;
            v += m10;
            ww += m20;
        }
    });
    Image { width, height, data: out }
}

/// Unsigned area of a simple polygon via the shoelace formula.
fn polygon_area(pts: &[(f64, f64)]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..pts.len() {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % pts.len()];
        sum += x0 * y1 - x1 * y0;
    }
    0.5 * sum.abs()
}

/// Clips a convex polygon (up to 8 vertices) against one axis-aligned half-plane in stack memory.
fn clip_halfplane(
    src: &[(f64, f64); 8],
    src_len: usize,
    dst: &mut [(f64, f64); 8],
    signed_dist: impl Fn((f64, f64)) -> f64,
) -> usize {
    if src_len == 0 {
        return 0;
    }
    let mut out_len = 0usize;
    let mut prev = src[src_len - 1];
    let mut prev_d = signed_dist(prev);
    for &cur in &src[..src_len] {
        let cur_d = signed_dist(cur);
        let prev_in = prev_d >= 0.0;
        let cur_in = cur_d >= 0.0;
        if prev_in != cur_in {
            let denom = prev_d - cur_d;
            if denom.abs() > 0.0 {
                let t = (prev_d / denom).clamp(0.0, 1.0);
                let inter = (prev.0 + t * (cur.0 - prev.0), prev.1 + t * (cur.1 - prev.1));
                if out_len < 8 {
                    dst[out_len] = inter;
                    out_len += 1;
                }
            }
        }
        if cur_in && out_len < 8 {
            dst[out_len] = cur;
            out_len += 1;
        }
        prev = cur;
        prev_d = cur_d;
    }
    out_len
}

/// Clips a projected 4-corner quadrilateral against `[0, fw] × [0, fh]` via Sutherland-Hodgman, returning the
/// exact clipped polygon area and its tight `[x1, y1, x2, y2]` bounding box (or `None` if outside).
fn clip_quad_to_canvas(quad: [(f64, f64); 4], fw: f64, fh: f64) -> Option<(f64, [f32; 4])> {
    let mut buf_a = [(0.0f64, 0.0f64); 8];
    let mut buf_b = [(0.0f64, 0.0f64); 8];
    buf_a[..4].copy_from_slice(&quad);
    let n = clip_halfplane(&buf_a, 4, &mut buf_b, |p| p.0);
    let n = clip_halfplane(&buf_b, n, &mut buf_a, |p| fw - p.0);
    let n = clip_halfplane(&buf_a, n, &mut buf_b, |p| p.1);
    let n = clip_halfplane(&buf_b, n, &mut buf_a, |p| fh - p.1);
    if n < 3 {
        return None;
    }
    let area = polygon_area(&buf_a[..n]);
    if !(area.is_finite() && area > 0.0) {
        return None;
    }
    let (mut lx, mut ly, mut hx, mut hy) =
        (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in &buf_a[..n] {
        lx = lx.min(x);
        ly = ly.min(y);
        hx = hx.max(x);
        hy = hy.max(y);
    }
    let rect = [
        (lx as f32).clamp(0.0, fw as f32),
        (ly as f32).clamp(0.0, fh as f32),
        (hx as f32).clamp(0.0, fw as f32),
        (hy as f32).clamp(0.0, fh as f32),
    ];
    Some((area, rect))
}

/// Transforms normalised boxes of a `src_w × src_h` image through `h` into normalised boxes of `width × height`.
///
/// Boxes with any corner on/behind the horizon (`w <= 1e-12` on the max-norm normalized homography), degenerate
/// area, or clipped visibility failing `filter` are dropped.
pub fn transform_boxes_perspective(
    boxes: &[BBox],
    src_w: u32,
    src_h: u32,
    h: &Homography,
    width: u32,
    height: u32,
    filter: &BoxFilter,
) -> Vec<BBox> {
    if boxes.is_empty() || src_w == 0 || src_h == 0 || width == 0 || height == 0 {
        return Vec::new();
    }
    let Some(hn) = h.normalized() else {
        return Vec::new();
    };
    if hn.inverse().and_then(|im| im.normalized()).is_none() {
        return Vec::new();
    }
    let (fw, fh) = (width as f64, height as f64);
    let mut out = Vec::with_capacity(boxes.len());
    for bx in boxes {
        let [x1, y1, x2, y2] = bx.pixel_xyxy(src_w, src_h);
        if !(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite()) {
            continue;
        }
        if !(x2 > x1 && y2 > y1) {
            continue;
        }
        let corners =
            [(x1 as f64, y1 as f64), (x2 as f64, y1 as f64), (x2 as f64, y2 as f64), (x1 as f64, y2 as f64)];
        let mut quad = [(0.0f64, 0.0f64); 4];
        let mut ok = true;
        for (slot, (cx, cy)) in quad.iter_mut().zip(corners) {
            let (u, v, w) = hn.apply_homogeneous(cx, cy);
            if !w.is_finite() || w <= HORIZON_EPS {
                ok = false;
                break;
            }
            let (px, py) = (u / w, v / w);
            if !(px.is_finite() && py.is_finite()) {
                ok = false;
                break;
            }
            *slot = (px, py);
        }
        if !ok {
            continue;
        }
        let unclipped_area = polygon_area(&quad);
        if !(unclipped_area.is_finite() && unclipped_area > 0.0) {
            continue;
        }
        let Some((visible_area, [cx1, cy1, cx2, cy2])) = clip_quad_to_canvas(quad, fw, fh) else {
            continue;
        };
        if visible_area < filter.min_area_ratio as f64 * unclipped_area {
            continue;
        }
        if filter.keep(cx2 - cx1, cy2 - cy1, unclipped_area as f32) {
            out.push(BBox::from_pixel_xyxy(bx.class, [cx1, cy1, cx2, cy2], width, height));
        }
    }
    out
}

/// Returns `true` when the ordered 4-vertex quadrilateral is strictly convex (all 4 consecutive edge cross
/// products share the same positive orientation).
fn quad_is_strictly_convex(q: [(f64, f64); 4]) -> bool {
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        let c = q[(i + 2) % 4];
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if !(cross.is_finite() && cross > 1e-9) {
            return false;
        }
    }
    true
}

/// Samples a deterministic well-conditioned corner-jitter homography mapping `[0, src_w] × [0, src_h]` into
/// `[0, out_w] × [0, out_h]`. Retries rejected / concave corner draws up to 8 times before falling back to
/// the exact source-to-output scale homography so the source image is never silently cropped.
pub fn random_perspective_homography(
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    distortion: f32,
    rng: &mut Rng,
) -> Homography {
    if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return Homography::IDENTITY;
    }
    let d = distortion.clamp(0.0, 0.45) as f64;
    let (sw, sh) = (src_w as f64, src_h as f64);
    let (ow, oh) = (out_w as f64, out_h as f64);
    let dx = d * ow;
    let dy = d * oh;
    let src = [(0.0, 0.0), (sw, 0.0), (sw, sh), (0.0, sh)];
    for _ in 0..8 {
        let dst = [
            (rng.uniform(0.0, dx as f32) as f64, rng.uniform(0.0, dy as f32) as f64),
            (ow - rng.uniform(0.0, dx as f32) as f64, rng.uniform(0.0, dy as f32) as f64),
            (ow - rng.uniform(0.0, dx as f32) as f64, oh - rng.uniform(0.0, dy as f32) as f64),
            (rng.uniform(0.0, dx as f32) as f64, oh - rng.uniform(0.0, dy as f32) as f64),
        ];
        if !quad_is_strictly_convex(dst) {
            continue;
        }
        if let Some(h) = Homography::from_quad_correspondence(src, dst) {
            return h;
        }
    }
    Homography::from_affine(&Affine::scale(ow / sw, oh / sh))
}

/// Warps `sample` by `h` into `width × height`, projecting and clipping boxes via [`transform_boxes_perspective`].
pub fn perspective_sample(s: &Sample, h: &Homography, width: u32, height: u32, filter: &BoxFilter) -> Sample {
    if h.is_identity() && s.image.width == width && s.image.height == height {
        return s.clone();
    }
    Sample {
        image: warp_perspective(&s.image, h, width, height, [PAD_VALUE; 3]),
        boxes: transform_boxes_perspective(&s.boxes, s.image.width, s.image.height, h, width, height, filter),
    }
}

/// Randomly applies a perspective warp according to `cfg` into `out_w × out_h`. When skipped (or `distortion == 0`
/// at matching size), returns the sample unchanged (or resized by exact scale homography when output size differs).
pub fn random_perspective(
    s: &Sample,
    out_w: u32,
    out_h: u32,
    cfg: &PerspectiveConfig,
    filter: &BoxFilter,
    rng: &mut Rng,
) -> Sample {
    if cfg.validate().is_err() || !rng.chance(cfg.prob) || cfg.distortion == 0.0 {
        if s.image.width == out_w && s.image.height == out_h {
            return s.clone();
        }
        let scale = Homography::from_affine(&Affine::scale(
            out_w as f64 / s.image.width.max(1) as f64,
            out_h as f64 / s.image.height.max(1) as f64,
        ));
        return perspective_sample(s, &scale, out_w, out_h, filter);
    }
    let h = random_perspective_homography(s.image.width, s.image.height, out_w, out_h, cfg.distortion, rng);
    perspective_sample(s, &h, out_w, out_h, filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warp::warp_affine;

    #[test]
    fn homography_algebra_and_quad_solver() {
        let src = [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)];
        let dst = [(12.0, 8.0), (92.0, 15.0), (88.0, 74.0), (5.0, 68.0)];
        let h = Homography::from_quad_correspondence(src, dst).expect("invertible quad");
        let inv = h.inverse().expect("inverse exists");
        for (s, d) in src.iter().zip(&dst) {
            let (u, v) = h.apply(s.0, s.1).unwrap();
            assert!((u - d.0).abs() < 1e-9 && (v - d.1).abs() < 1e-9, "fwd ({u},{v}) vs {d:?}");
            let (rx, ry) = inv.apply(u, v).unwrap();
            assert!((rx - s.0).abs() < 1e-9 && (ry - s.1).abs() < 1e-9, "inv ({rx},{ry}) vs {s:?}");
        }
        // Collinear destination points must be rejected.
        let bad = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)];
        assert!(Homography::from_quad_correspondence(src, bad).is_none());
    }

    #[test]
    fn warp_perspective_matches_affine_and_tracks_painted_box_pixels() {
        let mut img = Image::new(64, 48, [10, 10, 10]);
        img.fill_rect(16, 12, 48, 36, [240, 20, 30]);
        let bx = BBox::from_pixel_xyxy(0, [16.0, 12.0, 48.0, 36.0], 64, 48);

        // Affine equivalence: warp_perspective matches warp_affine within <= 1 LSB.
        let aff = Affine::translate(-32.0, -24.0)
            .then(&Affine::rotate(14.0))
            .then(&Affine::scale(0.9, 1.05))
            .then(&Affine::translate(34.0, 26.0));
        let wa = warp_affine(&img, &aff, 64, 48, [PAD_VALUE; 3]);
        let wp = warp_perspective(&img, &Homography::from_affine(&aff), 64, 48, [PAD_VALUE; 3]);
        for (a, b) in wa.data.iter().zip(&wp.data) {
            assert!((*a as i32 - *b as i32).abs() <= 1, "affine parity {a} vs {b}");
        }

        // True projective warp: every bright red pixel in the warped image must lie inside the projected box.
        let sample = Sample { image: img, boxes: vec![bx] };
        let cfg = PerspectiveConfig { prob: 1.0, distortion: 0.25 };
        let out = random_perspective(&sample, 64, 48, &cfg, &BoxFilter::default(), &mut Rng::new(123));
        assert_eq!(out.boxes.len(), 1);
        let [bx1, by1, bx2, by2] = out.boxes[0].pixel_xyxy(64, 48);
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (64u32, 48u32, 0u32, 0u32);
        for y in 0..48 {
            for x in 0..64 {
                if out.image.pixel(x, y)[0] > 180 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x + 1);
                    max_y = max_y.max(y + 1);
                }
            }
        }
        assert!(
            bx1 <= min_x as f32 + 1.5 && by1 <= min_y as f32 + 1.5,
            "box [{bx1},{by1}] vs px [{min_x},{min_y}]"
        );
        assert!(
            bx2 >= max_x as f32 - 1.5 && by2 >= max_y as f32 - 1.5,
            "box [{bx2},{by2}] vs px [{max_x},{max_y}]"
        );
    }

    #[test]
    fn perspective_drops_phantom_horizon_singular_and_degenerate_boxes() {
        let filter = BoxFilter { min_px: 1.0, min_area_ratio: 0.1, max_aspect: 100.0 };

        // 1. Phantom-AABB trap: rotate a square by 45 degrees and place it just outside the top-left corner so
        // its unclipped AABB overlaps [0, 64]x[0, 64], yet the rotated quad itself has zero intersection with
        // [0, 64]x[0, 64]. Sutherland-Hodgman clipping must drop it cleanly!
        let m = Affine::rotate(45.0).then(&Affine::translate(-25.0, -25.0));
        let h_rot = Homography::from_affine(&m);
        let boxes = vec![BBox::from_pixel_xyxy(0, [0.0, 0.0, 20.0, 20.0], 64, 64)];
        assert!(transform_boxes_perspective(&boxes, 64, 64, &h_rot, 64, 64, &filter).is_empty());

        // 2. Horizon-crossing box: `w = 1 - 0.05 * x` crosses `w = 0` at `x = 20`.
        let h_horizon = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.05, 0.0, 1.0]);
        let crossing = vec![
            BBox::from_pixel_xyxy(0, [2.0, 2.0, 10.0, 10.0], 64, 64), // w in [0.5, 0.9] -> kept
            BBox::from_pixel_xyxy(1, [10.0, 2.0, 30.0, 10.0], 64, 64), // w crosses 0 -> dropped
            BBox::from_pixel_xyxy(2, [25.0, 2.0, 40.0, 10.0], 64, 64), // w < 0 -> dropped
        ];
        let kept = transform_boxes_perspective(&crossing, 64, 64, &h_horizon, 64, 64, &filter);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].class, 0);

        // 3. Singular homography, NaN/zero-area boxes, and empty output images.
        let singular = Homography([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(transform_boxes_perspective(&crossing, 64, 64, &singular, 64, 64, &filter).is_empty());
        let bad_boxes = vec![
            BBox::new(0, f32::NAN, 0.5, 0.2, 0.2),
            BBox::new(1, 0.5, 0.5, 0.0, 0.2),
            BBox::new(2, 0.5, 0.5, -0.1, 0.2),
        ];
        assert!(
            transform_boxes_perspective(&bad_boxes, 64, 64, &Homography::IDENTITY, 64, 64, &filter)
                .is_empty()
        );
        assert!(
            transform_boxes_perspective(&crossing, 64, 64, &Homography::IDENTITY, 0, 64, &filter).is_empty()
        );

        // 4. Determinism & config validation.
        let s = Sample {
            image: Image::new(32, 32, [50, 100, 150]),
            boxes: vec![BBox::new(0, 0.5, 0.5, 0.4, 0.4)],
        };
        let cfg = PerspectiveConfig { prob: 1.0, distortion: 0.2 };
        assert!(cfg.validate().is_ok());
        let a = random_perspective(&s, 32, 32, &cfg, &filter, &mut Rng::new(55));
        let b = random_perspective(&s, 32, 32, &cfg, &filter, &mut Rng::new(55));
        assert_eq!(a.image.data, b.image.data);
        assert_eq!(a.boxes, b.boxes);
        assert!(PerspectiveConfig { distortion: 0.5, ..Default::default() }.validate().is_err());
        assert!(PerspectiveConfig { prob: -0.1, ..Default::default() }.validate().is_err());

        // 5. High distortion (0.45) with downscaling (128x96 -> 64x48, seed 11) must retry concave draws /
        // scale cleanly rather than falling back to IDENTITY and cropping out boxes in the right half.
        let mut rect_img = Image::new(128, 96, [10, 10, 10]);
        rect_img.fill_rect(80, 40, 100, 60, [240, 20, 30]);
        let rect_sample = Sample {
            image: rect_img,
            boxes: vec![BBox::from_pixel_xyxy(0, [80.0, 40.0, 100.0, 60.0], 128, 96)],
        };
        let high_cfg = PerspectiveConfig { prob: 1.0, distortion: 0.45 };
        for seed in 0..32u64 {
            let warped = random_perspective(&rect_sample, 64, 48, &high_cfg, &filter, &mut Rng::new(seed));
            assert_eq!(warped.boxes.len(), 1, "seed={seed} dropped valid box");
        }

        // 6. Positively rescaled homographies (1e13 * H and 1e-13 * H) must agree identically with H for
        // both pixels and boxes.
        let base_h = Homography([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let ref_out = perspective_sample(&s, &base_h, 32, 32, &filter);
        for scale in [1e13f64, 1e-13] {
            let scaled = Homography(base_h.0.map(|v| v * scale));
            let got_out = perspective_sample(&s, &scaled, 32, 32, &filter);
            assert_eq!(got_out.image.data, ref_out.image.data, "scale={scale} pixels diverged");
            assert_eq!(got_out.boxes, ref_out.boxes, "scale={scale} boxes diverged");
        }

        // 7. Small-scale + large-translation homographies (where det(H / max|H|) < 1e-12 even though H is
        // well-conditioned) must invert cleanly and match affine warping for both pixels and boxes.
        let solid = Sample {
            image: Image::new(256, 256, [220, 40, 80]),
            boxes: vec![BBox::from_pixel_xyxy(0, [0.0, 0.0, 256.0, 256.0], 256, 256)],
        };
        let tile_aff = Affine::scale(1.0 / 32.0, 1.0 / 32.0).then(&Affine::translate(1024.0, 0.0));
        let tile_h = Homography::from_affine(&tile_aff);
        assert!(tile_h.inverse().is_some());
        let tile_out = perspective_sample(&solid, &tile_h, 1056, 32, &filter);
        assert_eq!(tile_out.image.pixel(1028, 4), [220, 40, 80]);
        assert_eq!(tile_out.boxes.len(), 1);
        let [tx1, ty1, tx2, ty2] = tile_out.boxes[0].pixel_xyxy(1056, 32);
        assert!((tx1 - 1024.0).abs() < 1e-3 && (tx2 - 1032.0).abs() < 1e-3);
        assert!(ty1.abs() < 1e-3 && (ty2 - 8.0).abs() < 1e-3);

        // Additional well-conditioned large-translation / small-scale inverses and quad correspondences.
        let aff_thumb = Affine::scale(0.05, 0.05).then(&Affine::translate(1500.0, 1500.0));
        assert!(Homography::from_affine(&aff_thumb).inverse().is_some());
        assert!(Homography::from_affine(&Affine::translate(10_001.0, 0.0)).inverse().is_some());
        let quad_src = [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)];
        let quad_dst = [(660.0, 660.0), (692.0, 660.0), (692.0, 678.0), (660.0, 678.0)];
        assert!(Homography::from_quad_correspondence(quad_src, quad_dst).is_some());
    }
}
