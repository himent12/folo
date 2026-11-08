//! Letterbox resize: keep the aspect ratio, pad to the target size, and map boxes both ways exactly.

use crate::bbox::BBox;
use crate::image::{Image, Sample};
use crate::resize::resize_into;
use serde::{Deserialize, Serialize};

/// The grey FOLO (and YOLO) pads with.
pub const PAD_VALUE: u8 = 114;

/// How [`letterbox`] fits an image into the target.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LetterboxOptions {
    /// Allow enlarging images smaller than the target (default `true`).
    pub scale_up: bool,
    /// When set, the output is the smallest multiple of this stride that holds the resized image (minimal padding,
    /// "rect" inference) instead of exactly the target size.
    pub stride: Option<u32>,
    /// Centre the image in the padded area (default `true`); otherwise it sits in the top-left corner.
    pub center: bool,
    /// Padding grey level (default 114).
    pub pad_value: u8,
}

impl Default for LetterboxOptions {
    fn default() -> Self {
        LetterboxOptions { scale_up: true, stride: None, center: true, pad_value: PAD_VALUE }
    }
}

/// The geometry of one letterbox operation, enough to map boxes forward and predictions back.
///
/// A pixel coordinate `x` in the original image maps to `x * ratio[0] + pad[0]` in the output (and likewise for
/// `y`). `ratio` is per axis because the resized size is rounded to whole pixels; it is exact for that rounding.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Letterbox {
    /// Original `[width, height]`.
    pub orig: [u32; 2],
    /// Size of the resized image inside the output, `[width, height]`.
    pub resized: [u32; 2],
    /// Output `[width, height]`.
    pub out: [u32; 2],
    /// Scale `[x, y]` from original to output pixels.
    pub ratio: [f32; 2],
    /// Offset `[left, top]` of the resized image in the output, in pixels.
    pub pad: [f32; 2],
}

impl Letterbox {
    /// Computes the geometry for fitting `orig_w x orig_h` into `target_w x target_h`.
    pub fn compute(orig_w: u32, orig_h: u32, target_w: u32, target_h: u32, opts: &LetterboxOptions) -> Self {
        assert!(orig_w > 0 && orig_h > 0 && target_w > 0 && target_h > 0, "letterbox of an empty size");
        let mut r = (target_w as f64 / orig_w as f64).min(target_h as f64 / orig_h as f64);
        if !opts.scale_up {
            r = r.min(1.0);
        }
        let rw = ((orig_w as f64 * r).round() as u32).clamp(1, target_w);
        let rh = ((orig_h as f64 * r).round() as u32).clamp(1, target_h);
        let (ow, oh) = match opts.stride {
            Some(s) if s > 0 => (rw.div_ceil(s) * s, rh.div_ceil(s) * s),
            _ => (target_w, target_h),
        };
        let (px, py) = if opts.center { ((ow - rw) / 2, (oh - rh) / 2) } else { (0, 0) };
        Letterbox {
            orig: [orig_w, orig_h],
            resized: [rw, rh],
            out: [ow, oh],
            ratio: [rw as f32 / orig_w as f32, rh as f32 / orig_h as f32],
            pad: [px as f32, py as f32],
        }
    }

    /// The identity geometry for an image that is already `width x height`.
    pub fn identity(width: u32, height: u32) -> Self {
        Letterbox {
            orig: [width, height],
            resized: [width, height],
            out: [width, height],
            ratio: [1.0, 1.0],
            pad: [0.0, 0.0],
        }
    }

    /// Resizes and pads `img` according to this geometry.
    pub fn apply(&self, img: &Image, pad_value: u8) -> Image {
        assert_eq!([img.width, img.height], self.orig, "letterbox geometry computed for another size");
        let [ow, oh] = self.out;
        let mut out = Image::new(ow, oh, [pad_value; 3]);
        let stride = out.stride();
        let top = self.pad[1] as usize;
        resize_into(
            img,
            &mut out.data[top * stride..],
            stride,
            self.pad[0] as usize,
            self.resized[0],
            self.resized[1],
        );
        out
    }

    /// Maps original pixel corners `[x1, y1, x2, y2]` to output pixels.
    pub fn map_xyxy(&self, [x1, y1, x2, y2]: [f32; 4]) -> [f32; 4] {
        let ([rx, ry], [px, py]) = (self.ratio, self.pad);
        [x1 * rx + px, y1 * ry + py, x2 * rx + px, y2 * ry + py]
    }

    /// Maps output pixel corners (e.g. predictions) back to original pixels, clipped to the original image.
    pub fn unmap_xyxy(&self, [x1, y1, x2, y2]: [f32; 4]) -> [f32; 4] {
        let ([rx, ry], [px, py]) = (self.ratio, self.pad);
        let (w, h) = (self.orig[0] as f32, self.orig[1] as f32);
        [
            ((x1 - px) / rx).clamp(0.0, w),
            ((y1 - py) / ry).clamp(0.0, h),
            ((x2 - px) / rx).clamp(0.0, w),
            ((y2 - py) / ry).clamp(0.0, h),
        ]
    }

    /// Maps a normalised box of the original image to a normalised box of the output.
    pub fn map_box(&self, b: &BBox) -> BBox {
        let p = self.map_xyxy(b.pixel_xyxy(self.orig[0], self.orig[1]));
        BBox::from_pixel_xyxy(b.class, p, self.out[0], self.out[1])
    }

    /// Maps a normalised box of the output back to a normalised box of the original image (clipped).
    pub fn unmap_box(&self, b: &BBox) -> BBox {
        let p = self.unmap_xyxy(b.pixel_xyxy(self.out[0], self.out[1]));
        BBox::from_pixel_xyxy(b.class, p, self.orig[0], self.orig[1])
    }
}

/// Letterboxes an image into `target_w x target_h`.
pub fn letterbox(img: &Image, target_w: u32, target_h: u32, opts: &LetterboxOptions) -> (Image, Letterbox) {
    let lb = Letterbox::compute(img.width, img.height, target_w, target_h, opts);
    (lb.apply(img, opts.pad_value), lb)
}

/// Letterboxes a sample, remapping its boxes.
pub fn letterbox_sample(
    s: &Sample,
    target_w: u32,
    target_h: u32,
    opts: &LetterboxOptions,
) -> (Sample, Letterbox) {
    let (image, lb) = letterbox(&s.image, target_w, target_h, opts);
    let boxes = s.boxes.iter().map(|b| lb.map_box(b)).collect();
    (Sample { image, boxes }, lb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry() {
        let lb = Letterbox::compute(1280, 720, 640, 640, &LetterboxOptions::default());
        assert_eq!(lb.resized, [640, 360]);
        assert_eq!(lb.out, [640, 640]);
        assert_eq!(lb.pad, [0.0, 140.0]);
        let rect = Letterbox::compute(
            1280,
            720,
            640,
            640,
            &LetterboxOptions { stride: Some(32), ..Default::default() },
        );
        assert_eq!(rect.out, [640, 384]);
        assert_eq!(rect.pad, [0.0, 12.0]);
        let small = Letterbox::compute(
            100,
            50,
            640,
            640,
            &LetterboxOptions { scale_up: false, ..Default::default() },
        );
        assert_eq!(small.resized, [100, 50]);
        assert_eq!(small.pad, [270.0, 295.0]);
    }

    #[test]
    fn box_round_trip() {
        for &(w, h) in &[(1280u32, 720u32), (333, 999), (640, 640), (17, 3)] {
            let lb = Letterbox::compute(w, h, 640, 640, &LetterboxOptions::default());
            let b = BBox::new(2, 0.31, 0.62, 0.2, 0.33);
            let back = lb.unmap_box(&lb.map_box(&b));
            assert!((back.cx - b.cx).abs() < 1e-5 && (back.cy - b.cy).abs() < 1e-5, "{w}x{h}");
            assert!((back.w - b.w).abs() < 1e-5 && (back.h - b.h).abs() < 1e-5, "{w}x{h}");
        }
    }

    #[test]
    fn pixels_land_in_the_mapped_box() {
        let mut img = Image::new(300, 200, [0, 0, 0]);
        img.fill_rect(60, 40, 150, 120, [255, 0, 0]);
        let s = Sample {
            image: img,
            boxes: vec![BBox::from_pixel_xyxy(0, [60.0, 40.0, 150.0, 120.0], 300, 200)],
        };
        let (out, lb) = letterbox_sample(&s, 256, 256, &LetterboxOptions::default());
        assert_eq!(out.image.pixel(0, 0), [114, 114, 114]);
        let [x1, y1, x2, y2] = out.boxes[0].pixel_xyxy(256, 256);
        let red = |p: [u8; 3]| p[0] > 128 && p[1] < 64;
        for y in 0..256 {
            for x in 0..256 {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let inside = fx > x1 + 1.0 && fx < x2 - 1.0 && fy > y1 + 1.0 && fy < y2 - 1.0;
                let outside = fx < x1 - 1.0 || fx > x2 + 1.0 || fy < y1 - 1.0 || fy > y2 + 1.0;
                if inside {
                    assert!(red(out.image.pixel(x, y)), "({x},{y}) should be red");
                }
                if outside {
                    assert!(!red(out.image.pixel(x, y)), "({x},{y}) should not be red");
                }
            }
        }
        let _ = lb;
    }
}
