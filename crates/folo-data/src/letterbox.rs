//! Letterbox resize: keep the aspect ratio, pad to the target size, and map boxes both ways exactly.

use crate::bbox::BBox;
use crate::image::{Image, Sample};
use crate::resize::{RowResizer, resize_into};
use rayon::prelude::*;
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
        let mut out =
            Image { width: ow, height: oh, data: crate::buffers::scratch(ow as usize * oh as usize * 3) };
        out.data.fill(pad_value);
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

    /// [`Letterbox::apply`] straight into planar `f32` (the network input of one image), with no intermediate
    /// image: `dst` is `[3, out_h, out_w]` (R, G, B planes), and every byte `v` of the letterboxed RGB canvas,
    /// padding included, is written as `v as f32 * scale`. Bit for bit the planes of `apply(img, pad_value)`
    /// scaled that way (same resize taps and rounding). Rows run in parallel unless the thread is in
    /// [`crate::par::serial`] mode.
    ///
    /// # Panics
    /// If the geometry was computed for another image size, or `dst.len() != 3 * out_w * out_h`.
    pub fn apply_planar_f32(&self, img: &Image, pad_value: u8, scale: f32, dst: &mut [f32]) {
        assert_eq!([img.width, img.height], self.orig, "letterbox geometry computed for another size");
        let [ow, oh] = self.out.map(|v| v as usize);
        let plane = ow * oh;
        assert_eq!(
            dst.len(),
            3 * plane,
            "planar letterbox: dst holds {} floats, not 3 x {ow} x {oh}",
            dst.len()
        );
        if plane == 0 {
            return;
        }
        let [rw, rh] = self.resized.map(|v| v as usize);
        let [left, top] = self.pad.map(|v| v as usize);
        let lut: [f32; 256] = std::array::from_fn(|v| v as f32 * scale);
        let pad = lut[pad_value as usize];
        let rows = RowResizer::new(img, rw as u32, rh as u32);
        let (r, rest) = dst.split_at_mut(plane);
        let (g, b) = rest.split_at_mut(plane);
        let band = |y0: usize, r: &mut [f32], g: &mut [f32], b: &mut [f32]| {
            let mut px = vec![0u8; rw * 3];
            for (i, ((r, g), b)) in
                r.chunks_exact_mut(ow).zip(g.chunks_exact_mut(ow)).zip(b.chunks_exact_mut(ow)).enumerate()
            {
                let y = y0 + i;
                if y < top || y >= top + rh {
                    for p in [&mut *r, &mut *g, &mut *b] {
                        p.fill(pad);
                    }
                    continue;
                }
                rows.row(y - top, &mut px);
                for p in [&mut *r, &mut *g, &mut *b] {
                    p[..left].fill(pad);
                    p[left + rw..].fill(pad);
                }
                let (r, g, b) = (&mut r[left..left + rw], &mut g[left..left + rw], &mut b[left..left + rw]);
                for (x, p) in px.as_chunks::<3>().0.iter().enumerate() {
                    r[x] = lut[p[0] as usize];
                    g[x] = lut[p[1] as usize];
                    b[x] = lut[p[2] as usize];
                }
            }
        };
        let band_rows = (64 * 1024 / (12 * ow)).max(1);
        if crate::par::is_serial() || 12 * plane < 64 * 1024 {
            band(0, r, g, b);
        } else {
            let n = band_rows * ow;
            r.par_chunks_mut(n)
                .zip(g.par_chunks_mut(n))
                .zip(b.par_chunks_mut(n))
                .enumerate()
                .for_each(|(t, ((r, g), b))| band(t * band_rows, r, g, b));
        }
    }

    /// [`Letterbox::apply`] straight into packed `[out_h, out_w]` `u32` words `[R, G, B, 0]` (`u32::from_le_bytes([r,
    /// g, b, 0])`), with no intermediate image and 3× less write traffic than [`Letterbox::apply_planar_f32`] —
    /// feeds `folo_core::cpu::q8::stem_u8` directly. Bit for bit the bytes of `apply(img, pad_value)`.
    ///
    /// # Panics
    /// If the geometry was computed for another image size, or `dst.len() != out_w * out_h`.
    pub fn apply_rgb0_u8(&self, img: &Image, pad_value: u8, dst: &mut [u32]) {
        assert_eq!([img.width, img.height], self.orig, "letterbox geometry computed for another size");
        let [ow, oh] = self.out.map(|v| v as usize);
        let plane = ow * oh;
        assert_eq!(dst.len(), plane, "rgb0 letterbox: dst holds {} words, not {ow} x {oh}", dst.len());
        if plane == 0 {
            return;
        }
        let [rw, rh] = self.resized.map(|v| v as usize);
        let [left, top] = self.pad.map(|v| v as usize);
        let pad_word = u32::from_le_bytes([pad_value, pad_value, pad_value, 0]);
        let rows = RowResizer::new(img, rw as u32, rh as u32);
        let band = |y0: usize, out: &mut [u32]| {
            let mut px = vec![0u8; rw * 3];
            for (i, drow) in out.chunks_exact_mut(ow).enumerate() {
                let y = y0 + i;
                if y < top || y >= top + rh {
                    drow.fill(pad_word);
                    continue;
                }
                rows.row(y - top, &mut px);
                drow[..left].fill(pad_word);
                drow[left + rw..].fill(pad_word);
                for (d, p) in drow[left..left + rw].iter_mut().zip(px.as_chunks::<3>().0) {
                    *d = u32::from_le_bytes([p[0], p[1], p[2], 0]);
                }
            }
        };
        let band_rows = (64 * 1024 / (4 * ow)).max(1);
        if crate::par::is_serial() || 4 * plane < 64 * 1024 {
            band(0, dst);
        } else {
            let n = band_rows * ow;
            dst.par_chunks_mut(n).enumerate().for_each(|(t, o)| band(t * band_rows, o));
        }
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

    /// `apply` then a plain planar conversion: the reference of [`Letterbox::apply_planar_f32`].
    fn planar_reference(lb: &Letterbox, img: &Image, pad: u8, scale: f32) -> Vec<f32> {
        let canvas = lb.apply(img, pad);
        let plane = canvas.data.len() / 3;
        let mut v = vec![0f32; 3 * plane];
        for (i, p) in canvas.data.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                v[c * plane + i] = p[c] as f32 * scale;
            }
        }
        v
    }

    #[test]
    fn planar_f32_is_bit_exact_with_apply() {
        let rect = LetterboxOptions { stride: Some(32), ..Default::default() };
        let corner = LetterboxOptions { center: false, ..Default::default() };
        let no_up = LetterboxOptions { scale_up: false, ..Default::default() };
        let cases: [(u32, u32, u32, &LetterboxOptions); 9] = [
            (640, 640, 640, &rect),
            (1280, 720, 640, &rect),
            (1280, 720, 640, &LetterboxOptions::default()),
            (333, 999, 640, &rect),
            (100, 50, 640, &no_up),
            (3000, 1700, 640, &rect),
            (2001, 4099, 320, &corner),
            (7, 3, 64, &LetterboxOptions::default()),
            (1, 1, 32, &rect),
        ];
        for (w, h, s, opts) in cases {
            let mut img = Image::new(w, h, [0, 0, 0]);
            for (i, v) in img.data.iter_mut().enumerate() {
                *v = (i.wrapping_mul(2654435761) >> 7) as u8;
            }
            let lb = Letterbox::compute(w, h, s, s, opts);
            let want = planar_reference(&lb, &img, PAD_VALUE, 1.0 / 255.0);
            let mut got = vec![f32::NAN; want.len()];
            lb.apply_planar_f32(&img, PAD_VALUE, 1.0 / 255.0, &mut got);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{w}x{h} -> {lb:?}");
            let mut serial = vec![f32::NAN; want.len()];
            crate::par::serial(|| lb.apply_planar_f32(&img, 7, 0.5, &mut serial));
            assert_eq!(serial, planar_reference(&lb, &img, 7, 0.5), "{w}x{h} serial");
        }
    }

    #[test]
    fn rgb0_u8_is_bit_exact_with_apply() {
        let rect = LetterboxOptions { stride: Some(32), ..Default::default() };
        let corner = LetterboxOptions { center: false, ..Default::default() };
        let no_up = LetterboxOptions { scale_up: false, ..Default::default() };
        let cases: [(u32, u32, u32, &LetterboxOptions); 7] = [
            (640, 640, 640, &rect),
            (1280, 720, 640, &rect),
            (1280, 720, 640, &LetterboxOptions::default()),
            (333, 999, 640, &rect),
            (100, 50, 640, &no_up),
            (7, 3, 64, &corner),
            (1, 1, 32, &rect),
        ];
        for (w, h, s, opts) in cases {
            let mut img = Image::new(w, h, [0, 0, 0]);
            for (i, v) in img.data.iter_mut().enumerate() {
                *v = (i.wrapping_mul(2654435761) >> 7) as u8;
            }
            let lb = Letterbox::compute(w, h, s, s, opts);
            let canvas = lb.apply(&img, PAD_VALUE);
            let want: Vec<u32> = canvas
                .data
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| u32::from_le_bytes([p[0], p[1], p[2], 0]))
                .collect();
            let mut got = vec![u32::MAX; want.len()];
            lb.apply_rgb0_u8(&img, PAD_VALUE, &mut got);
            assert_eq!(got, want, "{w}x{h} parallel");
            let mut serial = vec![u32::MAX; want.len()];
            crate::par::serial(|| lb.apply_rgb0_u8(&img, PAD_VALUE, &mut serial));
            assert_eq!(serial, want, "{w}x{h} serial");
        }
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
