//! Axis-aligned box types and the IoU family.
//!
//! Two plain `f32` box representations:
//! - [`Xyxy`]: corners `(x1, y1)` top-left and `(x2, y2)` bottom-right. This is the canonical form used by NMS,
//!   metrics and the assigner.
//! - [`Cxcywh`]: centre plus width and height (the YOLO label format, in whatever units the caller uses).
//!
//! Coordinates are in pixels of whatever image the caller is working in (input image, letterboxed network input or
//! the original image); nothing here assumes normalised coordinates except where stated.
//!
//! **Degenerate boxes.** A box with `x2 < x1` or `y2 < y1` is treated as having zero width or height: [`Xyxy::area`]
//! is never negative, and every overlap function returns a finite value (no NaN, no division by zero) for empty,
//! inverted or identical boxes. IoU of two empty boxes is `0`.
//!
//! **Overlap measures.** [`IouKind`] selects plain IoU, GIoU (Rezatofighi et al. 2019), DIoU or CIoU (Zheng et al.
//! 2020). They are available per pair ([`iou`]), elementwise over two equal-length slices ([`iou_elementwise`]) and as
//! a dense row-major matrix ([`iou_matrix`]). The formulas follow Ultralytics' `bbox_iou` except that instead of
//! adding `eps = 1e-7` to every denominator they test the denominator explicitly, so identical boxes give exactly
//! `1.0`.
//!
//! **Letterbox.** [`Letterbox`] describes the resize + pad that maps an original image into the square network input
//! (and back), with the same rounding as Ultralytics' `LetterBox` / `scale_boxes`.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Element count above which the pairwise and elementwise helpers switch to rayon.
const PAR_THRESHOLD: usize = 1 << 15;

/// A box as corners: `(x1, y1)` top-left, `(x2, y2)` bottom-right.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Xyxy {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
}

/// A box as centre and size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cxcywh {
    /// Centre x.
    pub cx: f32,
    /// Centre y.
    pub cy: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Xyxy {
    /// A box from its corners.
    #[inline]
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// From a `[x1, y1, x2, y2]` array.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// As a `[x1, y1, x2, y2]` array.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }

    /// Width, clamped at zero for inverted boxes.
    #[inline]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Height, clamped at zero for inverted boxes.
    #[inline]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Area; zero for empty or inverted boxes.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Centre point `(cx, cy)`.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Convert to centre + size. Inverted boxes keep their (negative) extent so the conversion round-trips.
    #[inline]
    pub fn to_cxcywh(self) -> Cxcywh {
        Cxcywh {
            cx: (self.x1 + self.x2) * 0.5,
            cy: (self.y1 + self.y2) * 0.5,
            w: self.x2 - self.x1,
            h: self.y2 - self.y1,
        }
    }

    /// Clip to the image rectangle `[0, width] x [0, height]`.
    #[inline]
    pub fn clip(self, width: f32, height: f32) -> Self {
        Self {
            x1: self.x1.clamp(0.0, width),
            y1: self.y1.clamp(0.0, height),
            x2: self.x2.clamp(0.0, width),
            y2: self.y2.clamp(0.0, height),
        }
    }

    /// Multiply x coordinates by `sx` and y coordinates by `sy` (e.g. normalised → pixels).
    #[inline]
    pub fn scale(self, sx: f32, sy: f32) -> Self {
        Self { x1: self.x1 * sx, y1: self.y1 * sy, x2: self.x2 * sx, y2: self.y2 * sy }
    }

    /// Shift by `(dx, dy)`.
    #[inline]
    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self { x1: self.x1 + dx, y1: self.y1 + dy, x2: self.x2 + dx, y2: self.y2 + dy }
    }

    /// Area of the intersection with `other` (zero when disjoint).
    #[inline]
    pub fn intersection(&self, other: &Xyxy) -> f32 {
        let w = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        w * h
    }

    /// Whether the point lies strictly inside the box by more than `eps` on every side.
    #[inline]
    pub fn contains_point(&self, x: f32, y: f32, eps: f32) -> bool {
        (x - self.x1).min(y - self.y1).min(self.x2 - x).min(self.y2 - y) > eps
    }
}

impl Cxcywh {
    /// A box from its centre and size.
    #[inline]
    pub const fn new(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self { cx, cy, w, h }
    }

    /// Convert to corners.
    #[inline]
    pub fn to_xyxy(self) -> Xyxy {
        let (hw, hh) = (self.w * 0.5, self.h * 0.5);
        Xyxy { x1: self.cx - hw, y1: self.cy - hh, x2: self.cx + hw, y2: self.cy + hh }
    }

    /// Area; zero for negative sizes.
    #[inline]
    pub fn area(&self) -> f32 {
        self.w.max(0.0) * self.h.max(0.0)
    }
}

impl From<Cxcywh> for Xyxy {
    fn from(b: Cxcywh) -> Self {
        b.to_xyxy()
    }
}

impl From<Xyxy> for Cxcywh {
    fn from(b: Xyxy) -> Self {
        b.to_cxcywh()
    }
}

/// Which overlap measure to compute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IouKind {
    /// Intersection over union, in `[0, 1]`.
    #[default]
    Iou,
    /// Generalised IoU: `IoU - (C - U) / C` with `C` the enclosing box area; in `[-1, 1]`.
    Giou,
    /// Distance IoU: `IoU - rho² / c²` (centre distance over enclosing diagonal, squared); in `[-1, 1]`.
    Diou,
    /// Complete IoU: DIoU minus the aspect-ratio term `alpha * v`; in `[-1.5, 1]`.
    Ciou,
}

/// Plain IoU of two boxes. `0` when the union is empty.
#[inline]
pub fn box_iou(a: &Xyxy, b: &Xyxy) -> f32 {
    let inter = a.intersection(b);
    let union = a.area() + b.area() - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The overlap measure `kind` between two boxes. Symmetric in `a` and `b`, always finite and within the range
/// documented on [`IouKind`], including on degenerate, tiny (`1e-30`) and huge (`1e25`) finite boxes. Inverted
/// boxes are first collapsed to zero extent at their `(x1, y1)` corner.
#[inline]
pub fn iou(a: &Xyxy, b: &Xyxy, kind: IouKind) -> f32 {
    let collapse = |b: &Xyxy| Xyxy { x1: b.x1, y1: b.y1, x2: b.x2.max(b.x1), y2: b.y2.max(b.y1) };
    let (a, b) = (&collapse(a), &collapse(b));
    let (ax1, ay1, ax2, ay2) = (a.x1 as f64, a.y1 as f64, a.x2 as f64, a.y2 as f64);
    let (bx1, by1, bx2, by2) = (b.x1 as f64, b.y1 as f64, b.x2 as f64, b.y2 as f64);
    let iw = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let ih = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = iw * ih;
    let wa = (ax2 - ax1).max(0.0);
    let ha = (ay2 - ay1).max(0.0);
    let wb = (bx2 - bx1).max(0.0);
    let hb = (by2 - by1).max(0.0);
    let union = wa * ha + wb * hb - inter;
    let iou = if union > 0.0 { (inter / union) as f32 } else { 0.0 };
    if kind == IouKind::Iou {
        return iou;
    }
    // Enclosing box in f64: avoids f32 overflow on huge boxes and underflow on tiny boxes.
    let cw = (ax2.max(bx2) - ax1.min(bx1)).max(0.0);
    let ch = (ay2.max(by2) - ay1.min(by1)).max(0.0);
    match kind {
        IouKind::Iou => iou,
        IouKind::Giou => {
            let c_area = cw * ch;
            if c_area > 0.0 { (iou as f64 - (c_area - union) / c_area) as f32 } else { iou }
        }
        IouKind::Diou | IouKind::Ciou => {
            let c2 = cw * cw + ch * ch;
            // Pairwise differences first: each term negates exactly when a and b swap, so the result is symmetric.
            let dx = ((bx1 - ax1) + (bx2 - ax2)) * 0.5;
            let dy = ((by1 - ay1) + (by2 - ay2)) * 0.5;
            let rho2 = dx * dx + dy * dy;
            let dist = if c2 > 0.0 { (rho2 / c2).min(1.0) as f32 } else { 0.0 };
            if kind == IouKind::Diou {
                return iou - dist;
            }
            // atan2(w, h) == atan(w / h) for h > 0 and stays finite for h == 0.
            let d = (wb.atan2(hb) - wa.atan2(ha)) as f32;
            let v = (4.0 / (std::f32::consts::PI * std::f32::consts::PI)) * d * d;
            let denom = v - iou + 1.0;
            let alpha = if denom > 0.0 { v / denom } else { 0.0 };
            iou - (dist + v * alpha)
        }
    }
}

/// CIoU in the exact form of `docs/MODEL.md` §7.2 (Ultralytics' `bbox_iou(..., CIoU=True)`), with its `eps = 1e-7`
/// terms added to both heights, the union and the squared enclosing diagonal, and inside the `alpha` denominator.
/// The assigner and the box loss use this form. Unlike [`iou`] with [`IouKind::Ciou`], identical boxes score slightly
/// below 1 (noticeably so for boxes of a few 1e-3 pixels). Symmetric in `a` and `b`, and guarded against negative
/// extents or `f32` `c²` overflow on degenerate / extreme boxes.
#[inline]
pub fn ciou_eps(a: &Xyxy, b: &Xyxy) -> f32 {
    const EPS: f64 = 1e-7;
    let collapse = |b: &Xyxy| Xyxy { x1: b.x1, y1: b.y1, x2: b.x2.max(b.x1), y2: b.y2.max(b.y1) };
    let (a, b) = (collapse(a), collapse(b));
    // Compute geometry in f64 so huge boxes (e.g. 1e25) never overflow areas or c² to inf -> inf/inf = NaN.
    let (ax1, ay1, ax2, ay2) = (a.x1 as f64, a.y1 as f64, a.x2 as f64, a.y2 as f64);
    let (bx1, by1, bx2, by2) = (b.x1 as f64, b.y1 as f64, b.x2 as f64, b.y2 as f64);
    let (w1, h1) = ((ax2 - ax1).max(0.0), (ay2 - ay1).max(0.0) + EPS);
    let (w2, h2) = ((bx2 - bx1).max(0.0), (by2 - by1).max(0.0) + EPS);
    let inter = (ax2.min(bx2) - ax1.max(bx1)).max(0.0) * (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let union = (w1 * h1 + w2 * h2 - inter).max(0.0) + EPS;
    let iou = (inter / union).clamp(0.0, 1.0);
    let cw = (ax2.max(bx2) - ax1.min(bx1)).max(0.0);
    let ch = (ay2.max(by2) - ay1.min(by1)).max(0.0);
    let c2 = cw * cw + ch * ch + EPS;
    let dx = ((bx1 - ax1) + (bx2 - ax2)) * 0.5;
    let dy = ((by1 - ay1) + (by2 - ay2)) * 0.5;
    let rho2 = dx * dx + dy * dy;
    let dist = (rho2 / c2).min(1.0);
    let d = w2.atan2(h2) - w1.atan2(h1);
    let v = (4.0 / (std::f64::consts::PI * std::f64::consts::PI)) * d * d;
    // Ultralytics' evaluation order of §7.2's `1 − iou + v + eps`.
    let alpha = v / (v - iou + (1.0 + EPS)).max(EPS);
    (iou - (dist + v * alpha)) as f32
}

/// Elementwise overlap: `out[i] = iou(a[i], b[i], kind)`.
///
/// # Panics
/// If the three slices differ in length.
pub fn iou_elementwise_into(a: &[Xyxy], b: &[Xyxy], kind: IouKind, out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "iou_elementwise: length mismatch");
    assert_eq!(a.len(), out.len(), "iou_elementwise: output length mismatch");
    if a.len() >= PAR_THRESHOLD {
        out.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).for_each(|(o, (x, y))| *o = iou(x, y, kind));
    } else {
        for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
            *o = iou(x, y, kind);
        }
    }
}

/// Elementwise overlap of two equal-length slices.
pub fn iou_elementwise(a: &[Xyxy], b: &[Xyxy], kind: IouKind) -> Vec<f32> {
    let mut out = vec![0.0; a.len()];
    iou_elementwise_into(a, b, kind, &mut out);
    out
}

/// Pairwise overlap matrix, row-major `[a.len(), b.len()]`: `out[i * b.len() + j] = iou(a[i], b[j], kind)`.
///
/// # Panics
/// If `out.len() != a.len() * b.len()`.
pub fn iou_matrix_into(a: &[Xyxy], b: &[Xyxy], kind: IouKind, out: &mut [f32]) {
    assert_eq!(out.len(), a.len() * b.len(), "iou_matrix: output length mismatch");
    if b.is_empty() {
        return;
    }
    let row = |(ra, orow): (&Xyxy, &mut [f32])| {
        if kind == IouKind::Iou {
            // Hoist collapsed f64 corners and area of `ra`; stays branch-light while matching `iou()` on extreme boxes.
            let (ax1, ay1) = (ra.x1 as f64, ra.y1 as f64);
            let (ax2, ay2) = (ra.x2.max(ra.x1) as f64, ra.y2.max(ra.y1) as f64);
            let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
            for (o, rb) in orow.iter_mut().zip(b) {
                let (bx1, by1) = (rb.x1 as f64, rb.y1 as f64);
                let (bx2, by2) = (rb.x2.max(rb.x1) as f64, rb.y2.max(rb.y1) as f64);
                let iw = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
                let ih = (ay2.min(by2) - ay1.max(by1)).max(0.0);
                let inter = iw * ih;
                let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
                let union = area_a + area_b - inter;
                *o = if union > 0.0 { (inter / union) as f32 } else { 0.0 };
            }
        } else {
            for (o, rb) in orow.iter_mut().zip(b) {
                *o = iou(ra, rb, kind);
            }
        }
    };
    if out.len() >= PAR_THRESHOLD {
        a.par_iter().zip(out.par_chunks_mut(b.len())).for_each(row);
    } else {
        a.iter().zip(out.chunks_mut(b.len())).for_each(row);
    }
}

/// Pairwise overlap matrix, row-major `[a.len(), b.len()]`.
pub fn iou_matrix(a: &[Xyxy], b: &[Xyxy], kind: IouKind) -> Vec<f32> {
    let mut out = vec![0.0; a.len() * b.len()];
    iou_matrix_into(a, b, kind, &mut out);
    out
}

/// The resize + pad that maps an original image into a letterboxed network input.
///
/// Forward: `letterboxed = original * gain + pad`. Inverse: `original = (letterboxed - pad) / gain`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Letterbox {
    /// Uniform scale factor from original to letterboxed pixels.
    pub gain: f32,
    /// Left padding in letterboxed pixels.
    pub pad_x: f32,
    /// Top padding in letterboxed pixels.
    pub pad_y: f32,
    /// Width of the resized (unpadded) image inside the letterbox.
    pub resized_w: u32,
    /// Height of the resized (unpadded) image inside the letterbox.
    pub resized_h: u32,
}

impl Letterbox {
    /// The letterbox that fits a `src_w x src_h` image into `dst_w x dst_h`, keeping the aspect ratio and centring
    /// it, with the same rounding as Ultralytics' `LetterBox` (`scaleup = true`, `center = true`): the resized image
    /// is `round(src * gain)` pixels and the left/top pads are `round(d / 2 - 0.1)` where `d` is the leftover. Like
    /// Python, the arithmetic is in `f64` and `round` is ties-to-even.
    ///
    /// # Panics
    /// If any size is zero.
    pub fn fit(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0, "Letterbox::fit: zero-sized image");
        let gain = (dst_h as f64 / src_h as f64).min(dst_w as f64 / src_w as f64);
        let new_w = (src_w as f64 * gain).round_ties_even();
        let new_h = (src_h as f64 * gain).round_ties_even();
        let dw = (dst_w as f64 - new_w) / 2.0;
        let dh = (dst_h as f64 - new_h) / 2.0;
        Self {
            gain: gain as f32,
            pad_x: (dw - 0.1).round_ties_even().max(0.0) as f32,
            pad_y: (dh - 0.1).round_ties_even().max(0.0) as f32,
            resized_w: new_w as u32,
            resized_h: new_h as u32,
        }
    }

    /// Map a box from original-image pixels into letterboxed pixels.
    #[inline]
    pub fn forward(&self, b: Xyxy) -> Xyxy {
        b.scale(self.gain, self.gain).translate(self.pad_x, self.pad_y)
    }

    /// Map a box from letterboxed pixels back to original-image pixels (no clipping; see [`Xyxy::clip`]).
    #[inline]
    pub fn inverse(&self, b: Xyxy) -> Xyxy {
        let inv = 1.0 / self.gain;
        b.translate(-self.pad_x, -self.pad_y).scale(inv, inv)
    }

    /// Map a box back to original pixels and clip it to the original `src_w x src_h` image (Ultralytics'
    /// `scale_boxes`).
    #[inline]
    pub fn inverse_clipped(&self, b: Xyxy, src_w: u32, src_h: u32) -> Xyxy {
        self.inverse(b).clip(src_w as f32, src_h as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn conversions_round_trip() {
        let b = Xyxy::new(10.0, 20.0, 50.0, 80.0);
        let c = b.to_cxcywh();
        assert_eq!(c, Cxcywh::new(30.0, 50.0, 40.0, 60.0));
        assert_eq!(c.to_xyxy(), b);
        assert_eq!(Xyxy::from(c), b);
        assert_eq!(b.area(), 2400.0);
        assert_eq!(c.area(), 2400.0);
        assert_eq!(Xyxy::from_array(b.to_array()), b);
    }

    #[test]
    fn degenerate_area_and_clip() {
        let inverted = Xyxy::new(10.0, 10.0, 5.0, 20.0);
        assert_eq!(inverted.area(), 0.0);
        assert_eq!(Xyxy::new(-5.0, -5.0, 700.0, 300.0).clip(640.0, 480.0), Xyxy::new(0.0, 0.0, 640.0, 300.0));
    }

    #[test]
    fn iou_hand_computed() {
        // Two 2x2 squares overlapping in a 1x2 strip: inter 2, union 6.
        let a = Xyxy::new(0.0, 0.0, 2.0, 2.0);
        let b = Xyxy::new(1.0, 0.0, 3.0, 2.0);
        assert!(close(box_iou(&a, &b), 1.0 / 3.0, 1e-7));
        // GIoU: enclosing 3x2 = 6 = union, so GIoU == IoU.
        assert!(close(iou(&a, &b, IouKind::Giou), 1.0 / 3.0, 1e-7));
        // DIoU: centres (1,1),(2,1): rho² = 1; enclosing diagonal² = 9 + 4 = 13.
        assert!(close(iou(&a, &b, IouKind::Diou), 1.0 / 3.0 - 1.0 / 13.0, 1e-7));
        // CIoU: same aspect ratio, v = 0, so CIoU == DIoU.
        assert!(close(iou(&a, &b, IouKind::Ciou), 1.0 / 3.0 - 1.0 / 13.0, 1e-7));
    }

    #[test]
    fn giou_disjoint_hand_computed() {
        // Unit squares at (0,0) and (2,0): IoU 0, enclosing 3x1, union 2 => GIoU = -(3-2)/3.
        let a = Xyxy::new(0.0, 0.0, 1.0, 1.0);
        let b = Xyxy::new(2.0, 0.0, 3.0, 1.0);
        assert_eq!(box_iou(&a, &b), 0.0);
        assert!(close(iou(&a, &b, IouKind::Giou), -1.0 / 3.0, 1e-7));
        // DIoU: rho² = 4, c² = 9 + 1 = 10.
        assert!(close(iou(&a, &b, IouKind::Diou), -0.4, 1e-7));
    }

    #[test]
    fn ciou_aspect_term_hand_computed() {
        // a = 2x2 at origin, b = 2x4 sharing the same top-left corner.
        // inter = 4, union = 4 + 8 - 4 = 8, IoU = 0.5.
        // Centres (1,1) and (1,2): rho² = 1; enclosing 2x4: c² = 20.
        // v = 4/pi² * (atan(2/4) - atan(1))², alpha = v / (v - 0.5 + 1).
        let a = Xyxy::new(0.0, 0.0, 2.0, 2.0);
        let b = Xyxy::new(0.0, 0.0, 2.0, 4.0);
        let d = (0.5f64).atan() - std::f64::consts::FRAC_PI_4;
        let v = 4.0 / (std::f64::consts::PI * std::f64::consts::PI) * d * d;
        let alpha = v / (v + 0.5);
        let want = 0.5 - (1.0 / 20.0 + v * alpha);
        assert!(close(iou(&a, &b, IouKind::Ciou), want as f32, 1e-6));
        assert!(close(iou(&b, &a, IouKind::Ciou), want as f32, 1e-6), "CIoU must be symmetric");
    }

    #[test]
    fn identical_and_degenerate_are_finite() {
        let a = Xyxy::new(3.0, 4.0, 9.0, 12.0);
        for kind in [IouKind::Iou, IouKind::Giou, IouKind::Diou, IouKind::Ciou] {
            assert_eq!(iou(&a, &a, kind), 1.0, "{kind:?} of identical boxes");
        }
        let point = Xyxy::new(5.0, 5.0, 5.0, 5.0);
        let line = Xyxy::new(0.0, 5.0, 10.0, 5.0);
        let inverted = Xyxy::new(10.0, 10.0, 0.0, 0.0);
        let cases = [
            (point, point),
            (point, line),
            (line, a),
            (inverted, a),
            (inverted, inverted),
            (point, a),
            // Asymmetric inverted boxes far from each other used to give unbounded distance terms.
            (Xyxy::new(100.0, 0.0, 0.0, 10.0), Xyxy::new(0.0, 0.0, 1.0, 10.0)),
            (Xyxy::new(0.0, 0.0, 1.0, 10.0), Xyxy::new(5.0, 500.0, -300.0, 0.0)),
            // Nearly coincident points: the centre delta must not depend on argument order through rounding.
            (Xyxy::new(100.0, 0.0, 100.0, 0.0), Xyxy::new(100.0, 0.0, 100.000_01, 0.0)),
            // Tiny sub-pixel boxes (f32 c² would underflow to 0).
            (Xyxy::new(0.0, 0.0, 1e-25, 1e-25), Xyxy::new(2e-25, 0.0, 3e-25, 1e-25)),
            // Extreme huge boxes (f32 c² would overflow to inf -> inf/inf = NaN).
            (Xyxy::new(0.0, 0.0, 1e25, 1e25), Xyxy::new(0.5e25, 0.0, 1.5e25, 1e25)),
        ];
        for (x, y) in cases {
            for kind in [IouKind::Iou, IouKind::Giou, IouKind::Diou, IouKind::Ciou] {
                let v = iou(&x, &y, kind);
                assert!(v.is_finite(), "{kind:?}({x:?}, {y:?}) = {v}");
                let lo = match kind {
                    IouKind::Iou => 0.0,
                    IouKind::Giou | IouKind::Diou => -1.0,
                    IouKind::Ciou => -1.5,
                };
                assert!((lo..=1.0).contains(&v), "{kind:?}({x:?}, {y:?}) = {v} out of range");
                assert_eq!(v, iou(&y, &x, kind), "{kind:?} must be symmetric");
            }
            let ce = ciou_eps(&x, &y);
            assert!(ce.is_finite(), "ciou_eps({x:?}, {y:?}) = {ce}");
            assert!((-1.5..=1.0).contains(&ce), "ciou_eps({x:?}, {y:?}) = {ce} out of range");
            assert_eq!(ce, ciou_eps(&y, &x), "ciou_eps must be symmetric");
        }
        // Check exact box_iou == 0 on the degenerate/disjoint cases (all except the last overlapping huge pair).
        for (x, y) in &cases[..cases.len() - 1] {
            assert_eq!(box_iou(x, y), 0.0);
        }
        // Matrix path must agree with scalar `iou` even on extreme huge boxes (no f32 overflow).
        let huge = Xyxy::new(0.0, 0.0, 1e25, 1e25);
        assert_eq!(iou(&huge, &huge, IouKind::Iou), 1.0);
        assert_eq!(iou_matrix(&[huge], &[huge], IouKind::Iou), vec![1.0]);
    }

    #[test]
    fn matrix_and_elementwise_agree_with_scalar() {
        let a: Vec<Xyxy> = (0..7)
            .map(|i| {
                let f = i as f32;
                Xyxy::new(f, f * 0.5, f + 4.0 + f * 0.3, f * 0.5 + 3.0)
            })
            .collect();
        let b: Vec<Xyxy> = (0..5)
            .map(|i| {
                let f = i as f32;
                Xyxy::new(2.0 * f, 1.0, 2.0 * f + 3.0, 4.0 + f)
            })
            .collect();
        for kind in [IouKind::Iou, IouKind::Giou, IouKind::Diou, IouKind::Ciou] {
            let m = iou_matrix(&a, &b, kind);
            for (i, x) in a.iter().enumerate() {
                for (j, y) in b.iter().enumerate() {
                    assert!(close(m[i * b.len() + j], iou(x, y, kind), 1e-6));
                }
            }
            let e = iou_elementwise(&a[..5], &b, kind);
            for i in 0..5 {
                assert_eq!(e[i], iou(&a[i], &b[i], kind));
            }
        }
        assert!(iou_matrix(&a, &[], IouKind::Iou).is_empty());
    }

    #[test]
    fn letterbox_round_trip() {
        // 1280x720 into 640x640: gain 0.5, resized 640x360, vertical pad (640-360)/2 = 140.
        let lb = Letterbox::fit(1280, 720, 640, 640);
        assert_eq!(lb.gain, 0.5);
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 140.0));
        assert_eq!((lb.resized_w, lb.resized_h), (640, 360));
        let b = Xyxy::new(100.0, 200.0, 300.0, 400.0);
        let f = lb.forward(b);
        assert_eq!(f, Xyxy::new(50.0, 240.0, 150.0, 340.0));
        assert_eq!(lb.inverse(f), b);
        // Odd leftover: 1280x854 into 640x640 → resized 640x427, leftover 213 → top pad round(106.5 - 0.1) = 106.
        let lb = Letterbox::fit(1280, 854, 640, 640);
        assert_eq!(lb.pad_y, 106.0);
        assert_eq!(lb.inverse_clipped(Xyxy::new(-10.0, 0.0, 700.0, 640.0), 1280, 854).x2, 1280.0);
        // Exact half: 853 * 0.5 = 426.5 rounds to even (426, as Python does), leftover 214 → pad 107.
        let lb = Letterbox::fit(1280, 853, 640, 640);
        assert_eq!((lb.resized_w, lb.resized_h, lb.pad_y), (640, 426, 107.0));
    }
}
