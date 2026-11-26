//! Separable box and Gaussian blur augmentations on 8-bit RGB images.
//!
//! Blur modifies pixels without moving object geometry, so bounding boxes are preserved unchanged. Both
//! [`box_blur`] and [`gaussian_blur`] use edge-clamped (`replicate`) borders, preserve uniform images bit for
//! bit, run row-parallel via [`crate::par`], and are deterministic under [`Rng`](crate::Rng).

use crate::buffers;
use crate::image::{Image, Sample};
use crate::par::{for_bands, for_rows};
use crate::rng::Rng;
use crate::simd::assert_image_data;

/// Filter family used by [`random_blur`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlurKind {
    /// Uniform `(2 * radius + 1) × (2 * radius + 1)` box average.
    Box,
    /// Separable 2D Gaussian filter with standard deviation `sigma`.
    Gaussian,
}

/// Parameters for [`random_blur`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurConfig {
    /// Probability of applying blur (default `0.5`).
    pub prob: f32,
    /// Minimum filter radius in pixels (`kernel_size = 2 * radius + 1`, default `1`).
    pub min_radius: u32,
    /// Maximum filter radius in pixels (inclusive, default `2`).
    pub max_radius: u32,
    /// Minimum Gaussian standard deviation when `kind == BlurKind::Gaussian` (default `0.5`).
    pub min_sigma: f32,
    /// Maximum Gaussian standard deviation when `kind == BlurKind::Gaussian` (default `1.5`).
    pub max_sigma: f32,
    /// Filter family (default [`BlurKind::Gaussian`]).
    pub kind: BlurKind,
}

impl Default for BlurConfig {
    fn default() -> Self {
        BlurConfig {
            prob: 0.5,
            min_radius: 1,
            max_radius: 2,
            min_sigma: 0.5,
            max_sigma: 1.5,
            kind: BlurKind::Gaussian,
        }
    }
}

impl BlurConfig {
    /// Checks parameter ranges; returns a message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("blur.prob must be in [0, 1], got {}", self.prob));
        }
        if self.min_radius > self.max_radius {
            return Err(format!("blur.min_radius ({}) > max_radius ({})", self.min_radius, self.max_radius));
        }
        if self.kind == BlurKind::Gaussian {
            if !(self.min_sigma.is_finite() && self.min_sigma > 0.0) {
                return Err(format!("blur.min_sigma must be finite and > 0, got {}", self.min_sigma));
            }
            if !(self.max_sigma.is_finite() && self.max_sigma >= self.min_sigma) {
                return Err(format!(
                    "blur.max_sigma ({}) must be finite and >= min_sigma ({})",
                    self.max_sigma, self.min_sigma
                ));
            }
        }
        Ok(())
    }
}

/// Applies an in-place separable `(2 * radius + 1) × (2 * radius + 1)` box blur with replicate borders.
///
/// Uses an O(1)-per-pixel sliding window in `u64` with half-up integer rounding `(sum + k / 2) / k`, so
/// constant images stay bit-exact and large radii never overflow. `radius == 0` or empty images are no-ops.
pub fn box_blur(img: &mut Image, radius: u32) {
    assert_image_data(img);
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || radius == 0 {
        return;
    }
    let stride = w * 3;
    let total = h * stride;
    let r = radius as usize;
    let k = (r as u64) * 2 + 1;
    let half = k / 2;

    // Horizontal pass: `img.data` -> `tmp`.
    let mut tmp = buffers::scratch(total);
    let src_data = &img.data[..total];
    for_rows(&mut tmp, stride, |y, dst_row| {
        let src_row = &src_data[y * stride..(y + 1) * stride];
        let mut sum = [0u64; 3];
        // Initialize window centered at x = 0: `r + 1` copies of pixel 0, plus pixels `1..=r` clamped.
        let r0 = r as u64 + 1;
        sum[0] = src_row[0] as u64 * r0;
        sum[1] = src_row[1] as u64 * r0;
        sum[2] = src_row[2] as u64 * r0;
        for dx in 1..=r {
            let ix = dx.min(w - 1) * 3;
            sum[0] += src_row[ix] as u64;
            sum[1] += src_row[ix + 1] as u64;
            sum[2] += src_row[ix + 2] as u64;
        }
        for x in 0..w {
            let d = x * 3;
            dst_row[d] = ((sum[0] + half) / k) as u8;
            dst_row[d + 1] = ((sum[1] + half) / k) as u8;
            dst_row[d + 2] = ((sum[2] + half) / k) as u8;
            if x + 1 < w {
                let rem_x = x.saturating_sub(r) * 3;
                let add_x = x.saturating_add(1).saturating_add(r).min(w - 1) * 3;
                sum[0] = sum[0] + src_row[add_x] as u64 - src_row[rem_x] as u64;
                sum[1] = sum[1] + src_row[add_x + 1] as u64 - src_row[rem_x + 1] as u64;
                sum[2] = sum[2] + src_row[add_x + 2] as u64 - src_row[rem_x + 2] as u64;
            }
        }
    });

    // Vertical pass: `tmp` -> `img.data` in parallel row bands with a per-band sliding `u64` column sum.
    let tmp_ref = &tmp[..total];
    for_bands(&mut img.data[..total], stride, |y0, band| {
        let mut col_sum = vec![0u64; stride];
        // Initialize vertical window centered at `y0`: rows `y0 - r ..= y0 + r` clamped to `[0, h - 1]`.
        let top_clamped = r.saturating_sub(y0) as u64;
        if top_clamped > 0 {
            let row0 = &tmp_ref[..stride];
            for (acc, &v) in col_sum.iter_mut().zip(row0) {
                *acc += v as u64 * top_clamped;
            }
        }
        let first_in = y0.saturating_sub(r);
        let last_in = y0.saturating_add(r).min(h - 1);
        for sy in first_in..=last_in {
            let srow = &tmp_ref[sy * stride..(sy + 1) * stride];
            for (acc, &v) in col_sum.iter_mut().zip(srow) {
                *acc += v as u64;
            }
        }
        let bot_clamped = y0.saturating_add(r).saturating_sub(h - 1) as u64;
        if bot_clamped > 0 {
            let row_last = &tmp_ref[(h - 1) * stride..h * stride];
            for (acc, &v) in col_sum.iter_mut().zip(row_last) {
                *acc += v as u64 * bot_clamped;
            }
        }

        let band_rows = band.len() / stride;
        for (bi, dst_row) in band.chunks_exact_mut(stride).enumerate() {
            for (d, &s) in dst_row.iter_mut().zip(&col_sum) {
                *d = ((s + half) / k) as u8;
            }
            if bi + 1 < band_rows {
                let y = y0 + bi;
                let rem_y = y.saturating_sub(r);
                let add_y = y.saturating_add(1).saturating_add(r).min(h - 1);
                let rem_row = &tmp_ref[rem_y * stride..(rem_y + 1) * stride];
                let add_row = &tmp_ref[add_y * stride..(add_y + 1) * stride];
                for ((acc, &a), &r_v) in col_sum.iter_mut().zip(add_row).zip(rem_row) {
                    *acc = *acc + a as u64 - r_v as u64;
                }
            }
        }
    });
    buffers::recycle(tmp);
}

/// Builds a normalised symmetric 1D Gaussian kernel of effective radius `eff_r <= radius`, clamping beyond
/// `39 * sigma` where `exp(-d^2 / (2 * sigma^2)) == 0.0` in `f64` and converting offsets to `f64` before
/// squaring so large radii never overflow `isize` or `usize` on 32-bit targets.
fn gaussian_kernel_1d(radius: usize, sigma: f64) -> (usize, Vec<f64>) {
    // Beyond 39 * sigma, `d^2 / (2 * sigma^2) > 760 > -ln(f64::MIN_POSITIVE_SUBNORMAL)`, so weights are exact 0.
    let support = (sigma * 39.0).ceil();
    let eff_r =
        if support.is_finite() && support < radius as f64 { support.max(0.0) as usize } else { radius };
    let len =
        eff_r.checked_mul(2).and_then(|v| v.checked_add(1)).expect("gaussian_blur: kernel length overflow");
    let denom = 2.0 * sigma * sigma;
    let inv_2s2 = if denom.is_finite() && denom > 0.0 { 1.0 / denom } else { 0.0 };
    let mut k = vec![0.0f64; len];
    let mut sum = 0.0f64;
    for (i, slot) in k.iter_mut().enumerate() {
        let d = i as f64 - eff_r as f64;
        let w = (-(d * d) * inv_2s2).exp();
        *slot = w;
        sum += w;
    }
    let inv = 1.0 / sum;
    for w in &mut k {
        *w *= inv;
    }
    (eff_r, k)
}

/// Applies an in-place separable 2D Gaussian blur with replicate borders.
///
/// Uses an `f32` intermediate horizontal buffer (avoiding intermediate `u8` quantization) and `f64` normalized
/// weights, so uniform images stay bit-exact. Empty images, `radius == 0`, or non-positive/non-finite `sigma`
/// are no-ops.
pub fn gaussian_blur(img: &mut Image, radius: u32, sigma: f32) {
    assert_image_data(img);
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || radius == 0 || !sigma.is_finite() || sigma <= 0.0 {
        return;
    }
    let (r, kernel) = gaussian_kernel_1d(radius as usize, sigma as f64);
    if r == 0 {
        return;
    }
    let stride = w * 3;
    let total = h * stride;

    // Horizontal pass: `img.data` (`u8`) -> `tmp` (`f32`).
    let mut tmp = vec![0.0f32; total];
    let src_data = &img.data[..total];
    for_rows(&mut tmp, stride, |y, dst_row| {
        let src_row = &src_data[y * stride..(y + 1) * stride];
        for x in 0..w {
            let (mut sr, mut sg, mut sb) = (0.0f64, 0.0f64, 0.0f64);
            for (ki, &wt) in kernel.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - r as i64).clamp(0, w as i64 - 1) as usize * 3;
                sr += src_row[sx] as f64 * wt;
                sg += src_row[sx + 1] as f64 * wt;
                sb += src_row[sx + 2] as f64 * wt;
            }
            let d = x * 3;
            dst_row[d] = sr as f32;
            dst_row[d + 1] = sg as f32;
            dst_row[d + 2] = sb as f32;
        }
    });

    // Vertical pass: `tmp` (`f32`) -> `img.data` (`u8`).
    let tmp_ref = &tmp;
    for_rows(&mut img.data[..total], stride, |y, dst_row| {
        let mut acc = vec![0.0f64; stride];
        for (ki, &wt) in kernel.iter().enumerate() {
            let sy = (y as i64 + ki as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
            let srow = &tmp_ref[sy * stride..(sy + 1) * stride];
            for (a, &v) in acc.iter_mut().zip(srow) {
                *a += v as f64 * wt;
            }
        }
        for (d, a) in dst_row.iter_mut().zip(acc) {
            *d = a.round().clamp(0.0, 255.0) as u8;
        }
    });
}

/// Randomly applies [`box_blur`] or [`gaussian_blur`] to `sample` according to `cfg`, returning `true` when blur
/// was applied. Boxes are preserved unchanged.
pub fn random_blur(sample: &mut Sample, cfg: &BlurConfig, rng: &mut Rng) -> bool {
    if cfg.validate().is_err() || !rng.chance(cfg.prob) {
        return false;
    }
    if sample.image.width == 0 || sample.image.height == 0 {
        return false;
    }
    let span = (cfg.max_radius - cfg.min_radius) as usize + 1;
    let radius = cfg.min_radius + rng.below(span) as u32;
    if radius == 0 {
        return false;
    }
    match cfg.kind {
        BlurKind::Box => box_blur(&mut sample.image, radius),
        BlurKind::Gaussian => {
            let sigma = rng.uniform(cfg.min_sigma, cfg.max_sigma);
            gaussian_blur(&mut sample.image, radius, sigma);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BBox;

    fn pattern_image(w: u32, h: u32) -> Image {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 3) as usize;
                data[o] = ((x * 37 + y * 13) % 256) as u8;
                data[o + 1] = ((x * 19 + y * 53 + 7) % 256) as u8;
                data[o + 2] = (((x * 71) ^ (y * 29)) % 256) as u8;
            }
        }
        Image { width: w, height: h, data }
    }

    fn box_blur_ref(img: &Image, radius: u32) -> Image {
        let (w, h) = (img.width as usize, img.height as usize);
        let r = radius as isize;
        let k = (radius as u64) * 2 + 1;
        let half = k / 2;
        let mut tmp = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let mut s = 0u64;
                    for dx in -r..=r {
                        let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        s += img.data[(y * w + sx) * 3 + ch] as u64;
                    }
                    tmp[(y * w + x) * 3 + ch] = ((s + half) / k) as u8;
                }
            }
        }
        let mut out = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let mut s = 0u64;
                    for dy in -r..=r {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        s += tmp[(sy * w + x) * 3 + ch] as u64;
                    }
                    out[(y * w + x) * 3 + ch] = ((s + half) / k) as u8;
                }
            }
        }
        Image { width: img.width, height: img.height, data: out }
    }

    #[test]
    fn box_blur_matches_naive_reference_and_preserves_constants() {
        for (w, h, r) in [(1u32, 1u32, 2u32), (3, 5, 4), (17, 13, 1), (32, 24, 3), (65, 49, 5)] {
            let orig = pattern_image(w, h);
            let want = box_blur_ref(&orig, r);
            let mut got = orig.clone();
            box_blur(&mut got, r);
            assert_eq!(got.data, want.data, "box_blur mismatch {w}x{h} r={r}");

            // Uniform image stays bit-exact.
            let mut solid = Image::new(w, h, [17, 128, 251]);
            box_blur(&mut solid, r);
            assert!(solid.data.as_chunks::<3>().0.iter().all(|&p| p == [17, 128, 251]));
        }
        // Radius 0 and empty images are no-ops.
        let orig = pattern_image(9, 7);
        let mut copy = orig.clone();
        box_blur(&mut copy, 0);
        assert_eq!(copy.data, orig.data);
        let mut empty = Image { width: 0, height: 0, data: Vec::new() };
        box_blur(&mut empty, 3);
        assert!(empty.data.is_empty());
    }

    #[test]
    fn gaussian_blur_symmetry_constant_preservation_and_smoothing() {
        // Constant image stays bit-exact across radii and sigmas.
        for &(r, sigma) in &[(1u32, 0.5f32), (2, 1.2), (5, 3.0), (2, 1e-6)] {
            let mut solid = Image::new(15, 11, [42, 199, 255]);
            gaussian_blur(&mut solid, r, sigma);
            assert!(solid.data.as_chunks::<3>().0.iter().all(|&p| p == [42, 199, 255]));
        }
        // Impulse at center produces a symmetric peak strictly smaller than 255 with monotone radial decay.
        let mut impulse = Image::new(11, 11, [0, 0, 0]);
        impulse.data[(5 * 11 + 5) * 3] = 255;
        gaussian_blur(&mut impulse, 2, 1.0);
        let c = impulse.pixel(5, 5)[0];
        let n1 = impulse.pixel(4, 5)[0];
        let n2 = impulse.pixel(3, 5)[0];
        assert!(c > n1 && n1 > n2 && n2 > 0, "expected radial decay: c={c} n1={n1} n2={n2}");
        assert_eq!(impulse.pixel(4, 5), impulse.pixel(6, 5));
        assert_eq!(impulse.pixel(5, 4), impulse.pixel(5, 6));
        assert_eq!(impulse.pixel(4, 4), impulse.pixel(6, 6));

        // Invalid sigma or radius 0 or empty image are no-ops.
        let orig = pattern_image(8, 6);
        for &(r, s) in &[(0u32, 1.0f32), (2, 0.0), (2, -1.0), (2, f32::NAN), (2, f32::INFINITY)] {
            let mut c = orig.clone();
            gaussian_blur(&mut c, r, s);
            assert_eq!(c.data, orig.data);
        }
        // Large radius (exceeding sqrt(i32::MAX) = 46_340) must not overflow on 32-bit targets and must
        // preserve uniform pixels cleanly.
        let mut tiny = Image::new(1, 1, [77, 133, 211]);
        gaussian_blur(&mut tiny, 46_341, 1.0);
        assert_eq!(tiny.pixel(0, 0), [77, 133, 211]);
        let mut tiny_box = Image::new(1, 1, [77, 133, 211]);
        box_blur(&mut tiny_box, 46_341);
        assert_eq!(tiny_box.pixel(0, 0), [77, 133, 211]);
    }

    #[test]
    fn random_blur_is_deterministic_preserves_boxes_and_validates_config() {
        let sample = Sample {
            image: pattern_image(32, 24),
            boxes: vec![BBox::new(0, 0.5, 0.5, 0.3, 0.4), BBox::new(2, 0.2, 0.8, 0.1, 0.1)],
        };
        for kind in [BlurKind::Box, BlurKind::Gaussian] {
            let cfg = BlurConfig { prob: 1.0, min_radius: 1, max_radius: 3, kind, ..Default::default() };
            assert!(cfg.validate().is_ok());
            let mut a = sample.clone();
            let mut b = sample.clone();
            assert!(random_blur(&mut a, &cfg, &mut Rng::new(99)));
            assert!(random_blur(&mut b, &cfg, &mut Rng::new(99)));
            assert_eq!(a.image.data, b.image.data);
            assert_eq!(a.boxes, sample.boxes);
            assert_ne!(a.image.data, sample.image.data);
        }
        // Invalid configs rejected cleanly.
        assert!(BlurConfig { prob: 1.5, ..Default::default() }.validate().is_err());
        assert!(BlurConfig { min_radius: 3, max_radius: 1, ..Default::default() }.validate().is_err());
        assert!(BlurConfig { min_sigma: 0.0, ..Default::default() }.validate().is_err());
        assert!(BlurConfig { min_sigma: 2.0, max_sigma: 1.0, ..Default::default() }.validate().is_err());
        let mut unchanged = sample.clone();
        assert!(!random_blur(
            &mut unchanged,
            &BlurConfig { prob: f32::NAN, ..Default::default() },
            &mut Rng::new(1)
        ));
        assert_eq!(unchanged.image.data, sample.image.data);
    }
}
