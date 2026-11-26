//! Bounding boxes in normalised centre format.

use serde::{Deserialize, Serialize};

/// An axis-aligned box with a class id, in normalised `cx, cy, w, h` (fractions of image width / height).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// Class index into the dataset's class names.
    pub class: u32,
    /// Centre x in `0..1`.
    pub cx: f32,
    /// Centre y in `0..1`.
    pub cy: f32,
    /// Width in `0..1`.
    pub w: f32,
    /// Height in `0..1`.
    pub h: f32,
}

impl BBox {
    /// Creates a box from normalised centre coordinates.
    pub const fn new(class: u32, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        BBox { class, cx, cy, w, h }
    }

    /// Creates a box from normalised corners `x1, y1, x2, y2`.
    pub fn from_xyxy(class: u32, [x1, y1, x2, y2]: [f32; 4]) -> Self {
        BBox { class, cx: 0.5 * (x1 + x2), cy: 0.5 * (y1 + y2), w: x2 - x1, h: y2 - y1 }
    }

    /// Creates a box from pixel corners in an image of `width x height`.
    /// Returns an empty box at the origin if `width == 0` or `height == 0`.
    pub fn from_pixel_xyxy(class: u32, [x1, y1, x2, y2]: [f32; 4], width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            return BBox::new(class, 0.0, 0.0, 0.0, 0.0);
        }
        let (iw, ih) = (1.0 / width as f32, 1.0 / height as f32);
        BBox::from_xyxy(class, [x1 * iw, y1 * ih, x2 * iw, y2 * ih])
    }

    /// Normalised corners `[x1, y1, x2, y2]`.
    pub fn xyxy(&self) -> [f32; 4] {
        let (hw, hh) = (0.5 * self.w, 0.5 * self.h);
        [self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh]
    }

    /// Pixel corners `[x1, y1, x2, y2]` in an image of `width x height`.
    pub fn pixel_xyxy(&self, width: u32, height: u32) -> [f32; 4] {
        let [x1, y1, x2, y2] = self.xyxy();
        let (w, h) = (width as f32, height as f32);
        [x1 * w, y1 * h, x2 * w, y2 * h]
    }

    /// Normalised area `w * h`.
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// The box clipped to the unit square (may become empty; never has negative width or height).
    pub fn clipped(&self) -> BBox {
        let [x1, y1, x2, y2] = self.xyxy();
        let c = |v: f32| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        let cx1 = c(x1);
        let cy1 = c(y1);
        let cx2 = c(x2).max(cx1);
        let cy2 = c(y2).max(cy1);
        BBox::from_xyxy(self.class, [cx1, cy1, cx2, cy2])
    }

    /// Intersection over union with another box (class is ignored).
    pub fn iou(&self, other: &BBox) -> f32 {
        if !(self.cx.is_finite()
            && self.cy.is_finite()
            && self.w.is_finite()
            && self.h.is_finite()
            && other.cx.is_finite()
            && other.cy.is_finite()
            && other.w.is_finite()
            && other.h.is_finite()
            && self.w > 0.0
            && self.h > 0.0
            && other.w > 0.0
            && other.h > 0.0)
        {
            return 0.0;
        }
        let [a1, b1, a2, b2] = self.xyxy();
        let [c1, d1, c2, d2] = other.xyxy();
        let iw = (a2.min(c2) - a1.max(c1)).max(0.0);
        let ih = (b2.min(d2) - b1.max(d1)).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xyxy_round_trip() {
        let b = BBox::new(3, 0.5, 0.4, 0.2, 0.1);
        let r = BBox::from_xyxy(3, b.xyxy());
        assert!((r.cx - b.cx).abs() < 1e-6 && (r.w - b.w).abs() < 1e-6 && (r.h - b.h).abs() < 1e-6);
        let p = b.pixel_xyxy(200, 100);
        for (got, want) in p.iter().zip([80.0, 35.0, 120.0, 45.0]) {
            assert!((got - want).abs() < 1e-4);
        }
        let q = BBox::from_pixel_xyxy(3, p, 200, 100);
        assert!((q.cy - 0.4).abs() < 1e-6);
        assert!((b.iou(&b) - 1.0).abs() < 1e-6);
        assert!((BBox::new(0, 1.0, 0.5, 0.4, 0.2).clipped().w - 0.2).abs() < 1e-6);
    }

    #[test]
    fn clipped_iou_and_zero_dim_edge_cases() {
        // Out-of-bounds box is clamped to [0, 1].
        let oob = BBox::new(1, -0.5, 1.2, 0.4, 0.6).clipped();
        assert_eq!(oob.w, 0.0);
        assert!((oob.h - 0.1).abs() < 1e-6, "got {}", oob.h);
        // Inverted (negative width/height) and NaN boxes never yield negative dimensions or NaN IoU.
        let inv = BBox::new(2, 0.5, 0.5, -0.3, -0.2);
        let c = inv.clipped();
        assert!(c.w >= 0.0 && c.h >= 0.0 && c.cx.is_finite() && c.cy.is_finite());
        assert_eq!(inv.iou(&inv), 0.0);
        let nan_box = BBox::new(0, f32::NAN, 0.5, 0.2, 0.2);
        let cn = nan_box.clipped();
        assert_eq!((cn.cx, cn.w), (0.0, 0.0));
        assert_eq!(nan_box.iou(&BBox::new(0, 0.5, 0.5, 0.2, 0.2)), 0.0);
        // Zero-dimension image returns finite zero-size box rather than inf/NaN.
        for (w, h) in [(0, 100), (100, 0), (0, 0)] {
            let z = BBox::from_pixel_xyxy(4, [10.0, 20.0, 30.0, 40.0], w, h);
            assert_eq!(z, BBox::new(4, 0.0, 0.0, 0.0, 0.0));
        }
    }
}
