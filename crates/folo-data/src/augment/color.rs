//! Deterministic RGB colour jitter (brightness, contrast, saturation) and shared box clamping/sanitisation.
//!
//! Photometric transforms do not move object geometry, so valid in-bounds boxes are preserved bit-for-bit while
//! out-of-bounds boxes are clamped to `[0, 1]` and degenerate or non-finite boxes are dropped.

use super::BoxFilter;
use crate::bbox::BBox;
use crate::image::{Image, Sample};
use crate::par::for_bands;
use crate::rng::Rng;
use crate::simd::assert_image_data;

/// Parameters for [`color_jitter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorJitterConfig {
    /// Probability of applying colour jitter (default `0.5`).
    pub prob: f32,
    /// Symmetric brightness jitter radius; factor is sampled uniformly in `[(1 - brightness).max(0), 1 + brightness]`
    /// (default `0.2`). `0.0` disables brightness jitter.
    pub brightness: f32,
    /// Symmetric contrast jitter radius; factor is sampled uniformly in `[(1 - contrast).max(0), 1 + contrast]`
    /// (default `0.2`). `0.0` disables contrast jitter.
    pub contrast: f32,
    /// Symmetric saturation jitter radius; factor is sampled uniformly in `[(1 - saturation).max(0), 1 + saturation]`
    /// (default `0.2`). `0.0` disables saturation jitter.
    pub saturation: f32,
}

impl Default for ColorJitterConfig {
    fn default() -> Self {
        Self { prob: 0.5, brightness: 0.2, contrast: 0.2, saturation: 0.2 }
    }
}

impl ColorJitterConfig {
    /// Validates parameter ranges; returns an error message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("color_jitter.prob must be in [0, 1], got {}", self.prob));
        }
        for (name, v) in
            [("brightness", self.brightness), ("contrast", self.contrast), ("saturation", self.saturation)]
        {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("color_jitter.{name} must be finite and >= 0, got {v}"));
            }
        }
        Ok(())
    }
}

/// Clamps bounding boxes to the unit square `[0, 1]`, drops non-finite or zero-area boxes, and optionally
/// filters surviving boxes through `filter` using the image pixel dimensions `width × height`.
///
/// Valid boxes already strictly inside `[0, 1]` keep their exact floating-point coordinates (when retained).
pub fn clamp_and_filter_boxes(boxes: &mut Vec<BBox>, width: u32, height: u32, filter: Option<&BoxFilter>) {
    if boxes.is_empty() {
        return;
    }
    if width == 0 || height == 0 {
        boxes.clear();
        return;
    }
    let (fw, fh) = (width as f32, height as f32);
    let mut write = 0usize;
    for read in 0..boxes.len() {
        let b = boxes[read];
        if !(b.cx.is_finite()
            && b.cy.is_finite()
            && b.w.is_finite()
            && b.h.is_finite()
            && b.w > 0.0
            && b.h > 0.0)
        {
            continue;
        }
        let [x1, y1, x2, y2] = b.xyxy();
        let orig_area = (b.w * fw) * (b.h * fh);
        let in_bounds = x1 >= 0.0 && y1 >= 0.0 && x2 <= 1.0 && y2 <= 1.0;
        let clamped = if in_bounds {
            b
        } else {
            let cx1 = x1.clamp(0.0, 1.0);
            let cy1 = y1.clamp(0.0, 1.0);
            let cx2 = x2.clamp(0.0, 1.0);
            let cy2 = y2.clamp(0.0, 1.0);
            if !(cx2 > cx1 && cy2 > cy1) {
                continue;
            }
            BBox::from_xyxy(b.class, [cx1, cy1, cx2, cy2])
        };
        if let Some(f) = filter
            && !f.keep(clamped.w * fw, clamped.h * fh, orig_area)
        {
            continue;
        }
        boxes[write] = clamped;
        write += 1;
    }
    boxes.truncate(write);
}

/// Integer BT.601 luminance in `0..=255` (`round((77*R + 150*G + 29*B) / 256)`).
#[inline]
pub(crate) fn luma_bt601(r: u8, g: u8, b: u8) -> u8 {
    ((77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32 + 128) >> 8) as u8
}

fn apply_single_lut(img: &mut Image, lut: &[u8; 256]) {
    let stride = img.stride();
    let h = img.height as usize;
    if stride == 0 || h == 0 {
        return;
    }
    for_bands(&mut img.data[..h * stride], stride, |_, band| {
        for byte in band {
            *byte = lut[*byte as usize];
        }
    });
}

/// Scales RGB brightness in place by `factor` (`0.0` -> black, `1.0` -> identity, `> 1.0` -> brighter).
///
/// Non-finite or negative `factor` and empty images are safe no-ops.
pub fn adjust_brightness(img: &mut Image, factor: f32) {
    assert_image_data(img);
    if img.width == 0 || img.height == 0 || !factor.is_finite() || factor < 0.0 || factor == 1.0 {
        return;
    }
    let f = factor as f64;
    let mut lut = [0u8; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = (v as f64 * f).round().clamp(0.0, 255.0) as u8;
    }
    apply_single_lut(img, &lut);
}

/// Scales RGB contrast in place around the image's mean BT.601 luminance by `factor`
/// (`0.0` -> uniform mean grey, `1.0` -> identity).
///
/// Constant greyscale images stay bit-exact for any valid `factor`. Non-finite or negative `factor` and empty
/// images are safe no-ops.
pub fn adjust_contrast(img: &mut Image, factor: f32) {
    assert_image_data(img);
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || !factor.is_finite() || factor < 0.0 || factor == 1.0 {
        return;
    }
    let total = h * w * 3;
    let mut sum_luma = 0u64;
    for px in img.data[..total].as_chunks::<3>().0 {
        sum_luma += luma_bt601(px[0], px[1], px[2]) as u64;
    }
    let mean = sum_luma as f64 / (w as u64 * h as u64) as f64;
    let f = factor as f64;
    let mut lut = [0u8; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = (mean + (v as f64 - mean) * f).round().clamp(0.0, 255.0) as u8;
    }
    apply_single_lut(img, &lut);
}

/// Scales RGB saturation in place by blending each pixel with its BT.601 luminance
/// (`0.0` -> greyscale, `1.0` -> identity).
///
/// Greyscale pixels (`R == G == B`) stay bit-exact for any valid `factor`. Non-finite or negative `factor`
/// and empty images are safe no-ops.
pub fn adjust_saturation(img: &mut Image, factor: f32) {
    assert_image_data(img);
    let stride = img.stride();
    let h = img.height as usize;
    if stride == 0 || h == 0 || !factor.is_finite() || factor < 0.0 || factor == 1.0 {
        return;
    }
    for_bands(&mut img.data[..h * stride], stride, |_, band| {
        for px in band.as_chunks_mut::<3>().0 {
            let y = luma_bt601(px[0], px[1], px[2]) as f32;
            px[0] = (y + (px[0] as f32 - y) * factor).round().clamp(0.0, 255.0) as u8;
            px[1] = (y + (px[1] as f32 - y) * factor).round().clamp(0.0, 255.0) as u8;
            px[2] = (y + (px[2] as f32 - y) * factor).round().clamp(0.0, 255.0) as u8;
        }
    });
}

/// Randomly applies brightness, contrast, and saturation jitter in a deterministic random order using `rng`,
/// returning `true` when jitter was applied.
///
/// Bounding boxes are preserved and clamped to `[0, 1]` (dropping non-finite or degenerate boxes). Invalid
/// configs, all-zero jitter radii, or `rng.chance(cfg.prob) == false` are no-ops returning `false`.
pub fn color_jitter(sample: &mut Sample, cfg: &ColorJitterConfig, rng: &mut Rng) -> bool {
    if cfg.validate().is_err()
        || (cfg.brightness == 0.0 && cfg.contrast == 0.0 && cfg.saturation == 0.0)
        || !rng.chance(cfg.prob)
    {
        return false;
    }
    clamp_and_filter_boxes(&mut sample.boxes, sample.image.width, sample.image.height, None);
    if sample.image.width == 0 || sample.image.height == 0 {
        return false;
    }
    let mut order = [0u8, 1, 2];
    rng.shuffle(&mut order);
    let mut changed = false;
    for op in order {
        match op {
            0 if cfg.brightness > 0.0 => {
                let lo = (1.0 - cfg.brightness).max(0.0);
                let hi = 1.0 + cfg.brightness;
                let f = rng.uniform(lo, hi);
                adjust_brightness(&mut sample.image, f);
                changed = true;
            }
            1 if cfg.contrast > 0.0 => {
                let lo = (1.0 - cfg.contrast).max(0.0);
                let hi = 1.0 + cfg.contrast;
                let f = rng.uniform(lo, hi);
                adjust_contrast(&mut sample.image, f);
                changed = true;
            }
            2 if cfg.saturation > 0.0 => {
                let lo = (1.0 - cfg.saturation).max(0.0);
                let hi = 1.0 + cfg.saturation;
                let f = rng.uniform(lo, hi);
                adjust_saturation(&mut sample.image, f);
                changed = true;
            }
            _ => {}
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern_image(w: u32, h: u32) -> Image {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 3) as usize;
                data[o] = ((x * 37 + y * 13 + 20) % 256) as u8;
                data[o + 1] = ((x * 19 + y * 53 + 40) % 256) as u8;
                data[o + 2] = (((x * 71) ^ (y * 29)) % 256) as u8;
            }
        }
        Image { width: w, height: h, data }
    }

    #[test]
    fn brightness_contrast_saturation_identities_and_extremes() {
        let orig = pattern_image(16, 12);

        // Factor 1.0 is bit-exact identity for all three ops.
        let mut id = orig.clone();
        adjust_brightness(&mut id, 1.0);
        adjust_contrast(&mut id, 1.0);
        adjust_saturation(&mut id, 1.0);
        assert_eq!(id.data, orig.data);

        // Brightness 0.0 -> all zeros.
        let mut dark = orig.clone();
        adjust_brightness(&mut dark, 0.0);
        assert!(dark.data.iter().all(|&b| b == 0));

        // Saturation 0.0 -> exact greyscale (R == G == B).
        let mut gray = orig.clone();
        adjust_saturation(&mut gray, 0.0);
        assert!(gray.data.as_chunks::<3>().0.iter().all(|px| px[0] == px[1] && px[1] == px[2]));

        // Contrast 0.0 -> uniform mean luminance.
        let mut flat = orig.clone();
        adjust_contrast(&mut flat, 0.0);
        let first = flat.pixel(0, 0);
        assert!(first[0] == first[1] && first[1] == first[2]);
        assert!(flat.data.as_chunks::<3>().0.iter().all(|&px| px == first));

        // Uniform grey stays bit-exact under contrast and saturation changes, even at huge factors.
        let mut solid = Image::new(7, 5, [111, 111, 111]);
        adjust_contrast(&mut solid, 1.8);
        adjust_contrast(&mut solid, 1e18);
        adjust_saturation(&mut solid, 2.5);
        adjust_saturation(&mut solid, 1e8);
        assert!(solid.data.as_chunks::<3>().0.iter().all(|&px| px == [111, 111, 111]));

        // 1x1 and 0x0 and invalid factors never panic.
        let mut one = Image::new(1, 1, [100, 150, 200]);
        adjust_brightness(&mut one, -1.0);
        adjust_contrast(&mut one, f32::NAN);
        adjust_saturation(&mut one, f32::INFINITY);
        assert_eq!(one.pixel(0, 0), [100, 150, 200]);
        adjust_brightness(&mut one, 0.5);
        assert_eq!(one.pixel(0, 0), [50, 75, 100]);

        let mut empty = Image { width: 0, height: 0, data: Vec::new() };
        adjust_brightness(&mut empty, 1.5);
        adjust_contrast(&mut empty, 1.5);
        adjust_saturation(&mut empty, 1.5);
    }

    #[test]
    fn color_jitter_is_deterministic_clamps_boxes_and_handles_edge_cases() {
        let base = Sample {
            image: pattern_image(24, 18),
            boxes: vec![
                // Valid in-bounds box -> preserved bit-exact.
                BBox::new(0, 0.5, 0.5, 0.4, 0.3),
                // Partially out-of-bounds box -> clamped to [0, 1].
                BBox::from_xyxy(1, [-0.2, 0.1, 0.4, 1.3]),
                // Completely out-of-bounds -> dropped.
                BBox::from_xyxy(2, [1.1, 0.2, 1.5, 0.6]),
                // Degenerate zero-width & NaN -> dropped.
                BBox::new(3, 0.5, 0.5, 0.0, 0.2),
                BBox::new(4, f32::NAN, 0.5, 0.2, 0.2),
            ],
        };
        let cfg = ColorJitterConfig { prob: 1.0, brightness: 0.35, contrast: 0.35, saturation: 0.35 };
        assert!(cfg.validate().is_ok());

        let mut a = base.clone();
        let mut b = base.clone();
        assert!(color_jitter(&mut a, &cfg, &mut Rng::new(42)));
        assert!(color_jitter(&mut b, &cfg, &mut Rng::new(42)));
        assert_eq!(a, b);
        assert_ne!(a.image.data, base.image.data);
        assert_eq!(a.boxes.len(), 2);
        assert_eq!(a.boxes[0], base.boxes[0]);
        assert_eq!(a.boxes[1].class, 1);
        let [x1, y1, x2, y2] = a.boxes[1].xyxy();
        assert!(
            (x1 - 0.0).abs() < 1e-6
                && (y1 - 0.1).abs() < 1e-6
                && (x2 - 0.4).abs() < 1e-6
                && (y2 - 1.0).abs() < 1e-6
        );

        // Empty boxes and 1x1 image.
        let mut tiny = Sample { image: Image::new(1, 1, [80, 120, 160]), boxes: vec![] };
        assert!(color_jitter(&mut tiny, &cfg, &mut Rng::new(7)));
        assert!(tiny.boxes.is_empty());

        // Invalid configs rejected without modifying sample.
        for bad in [
            ColorJitterConfig { prob: 1.2, ..Default::default() },
            ColorJitterConfig { brightness: -0.1, ..Default::default() },
            ColorJitterConfig { contrast: f32::NAN, ..Default::default() },
            ColorJitterConfig { saturation: f32::INFINITY, ..Default::default() },
        ] {
            assert!(bad.validate().is_err());
            let mut probe = base.clone();
            assert!(!color_jitter(&mut probe, &bad, &mut Rng::new(1)));
            assert_eq!(probe.image.data, base.image.data);
            assert_eq!(probe.boxes.len(), base.boxes.len());
            assert_eq!(probe.boxes[..4], base.boxes[..4]);
            assert!(probe.boxes[4].cx.is_nan());
        }
    }
}
