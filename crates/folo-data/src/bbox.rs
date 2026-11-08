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
    pub fn from_pixel_xyxy(class: u32, [x1, y1, x2, y2]: [f32; 4], width: u32, height: u32) -> Self {
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

    /// The box clipped to the unit square (may become empty).
    pub fn clipped(&self) -> BBox {
        let [x1, y1, x2, y2] = self.xyxy();
        let c = |v: f32| v.clamp(0.0, 1.0);
        BBox::from_xyxy(self.class, [c(x1), c(y1), c(x2), c(y2)])
    }

    /// Intersection over union with another box (class is ignored).
    pub fn iou(&self, other: &BBox) -> f32 {
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
}
