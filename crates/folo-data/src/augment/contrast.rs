//! Global [`auto_contrast`] and CLAHE-style local contrast ([`clahe`]) augmentations on 8-bit RGB images.
//!
//! Both operations preserve object geometry, clamp out-of-bounds bounding boxes to `[0, 1]`, drop degenerate
//! or non-finite boxes, handle `1×1` and non-divisible image sizes gracefully, and are deterministic under
//! [`Rng`](crate::Rng).

use super::color::{clamp_and_filter_boxes, luma_bt601};
use crate::image::{Image, Sample};
use crate::par::for_bands;
use crate::rng::Rng;
use crate::simd::assert_image_data;

/// Parameters for [`random_auto_contrast`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoContrastConfig {
    /// Probability of applying auto-contrast (default `0.5`).
    pub prob: f32,
    /// Percentage of darkest and lightest pixels clipped from each channel's histogram in `[0, 50)`
    /// (default `0.0`, full min–max stretch).
    pub cutoff: f32,
}

impl Default for AutoContrastConfig {
    fn default() -> Self {
        Self { prob: 0.5, cutoff: 0.0 }
    }
}

impl AutoContrastConfig {
    /// Validates parameter ranges; returns an error message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("auto_contrast.prob must be in [0, 1], got {}", self.prob));
        }
        if !(self.cutoff.is_finite() && (0.0..50.0).contains(&self.cutoff)) {
            return Err(format!("auto_contrast.cutoff must be finite and in [0, 50), got {}", self.cutoff));
        }
        Ok(())
    }
}

/// Colour space for [`clahe`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClaheSpace {
    /// Equalize BT.601 luminance only and shift RGB channels by `ΔY`, preserving chromaticity and greyscale.
    #[default]
    Luminance,
    /// Equalize R, G, and B channels independently.
    PerChannel,
}

/// Parameters for [`clahe`] and [`random_clahe`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClaheConfig {
    /// Probability of applying CLAHE in [`random_clahe`] (default `0.5`).
    pub prob: f32,
    /// Number of tiles `[grid_x, grid_y]` along width and height (each must be `>= 1`, default `[8, 8]`).
    /// Automatically clamped to the image dimensions when an image is smaller than the grid.
    pub tile_grid: [u32; 2],
    /// Histogram clip limit multiplier relative to a uniform tile histogram (`> 0.0`, default `2.0`).
    pub clip_limit: f32,
    /// Channel mode (default [`ClaheSpace::Luminance`]).
    pub space: ClaheSpace,
}

impl Default for ClaheConfig {
    fn default() -> Self {
        Self { prob: 0.5, tile_grid: [8, 8], clip_limit: 2.0, space: ClaheSpace::Luminance }
    }
}

impl ClaheConfig {
    /// Validates parameter ranges; returns an error message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("clahe.prob must be in [0, 1], got {}", self.prob));
        }
        if self.tile_grid[0] == 0 || self.tile_grid[1] == 0 {
            return Err(format!("clahe.tile_grid entries must be >= 1, got {:?}", self.tile_grid));
        }
        if !(self.clip_limit.is_finite() && self.clip_limit > 0.0) {
            return Err(format!("clahe.clip_limit must be finite and > 0, got {}", self.clip_limit));
        }
        Ok(())
    }
}

/// Stretches each RGB channel's histogram independently to `[0, 255]`, optionally clipping `cutoff_pct` percent
/// of pixels from the low and high tails (`cutoff_pct ∈ [0, 50)`).
///
/// Constant channels stay bit-exact. Empty images or invalid `cutoff_pct` values are safe no-ops.
pub fn auto_contrast(img: &mut Image, cutoff_pct: f32) {
    assert_image_data(img);
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0 || h == 0 || !cutoff_pct.is_finite() || !(0.0..50.0).contains(&cutoff_pct) {
        return;
    }
    let stride = w * 3;
    let total = h * stride;
    let n_pixels = (w as u64) * (h as u64);

    let mut hists = [[0u64; 256]; 3];
    for px in img.data[..total].as_chunks::<3>().0 {
        hists[0][px[0] as usize] += 1;
        hists[1][px[1] as usize] += 1;
        hists[2][px[2] as usize] += 1;
    }

    let cut = ((n_pixels as f64) * (cutoff_pct as f64) / 100.0).floor() as u64;
    let mut luts = [[0u8; 256]; 3];
    for ch in 0..3 {
        let hist = &hists[ch];
        let mut lo = 0usize;
        let mut acc = 0u64;
        for (i, &c) in hist.iter().enumerate() {
            acc += c;
            if acc > cut {
                lo = i;
                break;
            }
        }
        let mut hi = 255usize;
        let mut racc = 0u64;
        for (i, &c) in hist.iter().enumerate().rev() {
            racc += c;
            if racc > cut {
                hi = i;
                break;
            }
        }
        if hi <= lo {
            for (v, slot) in luts[ch].iter_mut().enumerate() {
                *slot = v as u8;
            }
        } else {
            let span = (hi - lo) as u32;
            let half = span / 2;
            for (v, slot) in luts[ch].iter_mut().enumerate() {
                *slot = if v <= lo {
                    0
                } else if v >= hi {
                    255
                } else {
                    (((v - lo) as u32 * 255 + half) / span) as u8
                };
            }
        }
    }

    for_bands(&mut img.data[..total], stride, |_, band| {
        for px in band.as_chunks_mut::<3>().0 {
            px[0] = luts[0][px[0] as usize];
            px[1] = luts[1][px[1] as usize];
            px[2] = luts[2][px[2] as usize];
        }
    });
}

/// Randomly applies [`auto_contrast`] to `sample`, clamping and sanitising bounding boxes.
pub fn random_auto_contrast(sample: &mut Sample, cfg: &AutoContrastConfig, rng: &mut Rng) -> bool {
    if cfg.validate().is_err() || !rng.chance(cfg.prob) {
        return false;
    }
    clamp_and_filter_boxes(&mut sample.boxes, sample.image.width, sample.image.height, None);
    if sample.image.width == 0 || sample.image.height == 0 {
        return false;
    }
    auto_contrast(&mut sample.image, cfg.cutoff);
    true
}

/// Builds a contrast-limited cumulative-distribution lookup table for one tile histogram.
fn build_clahe_lut(hist: &[u64; 256], tile_pixels: u64, clip_limit: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if tile_pixels == 0 || hist.contains(&tile_pixels) {
        // Uniform (single-value) tile: keep identity so flat backgrounds do not jump to 255.
        for (v, slot) in lut.iter_mut().enumerate() {
            *slot = v as u8;
        }
        return lut;
    }

    let limit = ((tile_pixels as f64 / 256.0) * clip_limit as f64).ceil().max(1.0) as u64;
    let mut clipped = *hist;
    let mut excess = 0u64;
    for c in &mut clipped {
        if *c > limit {
            excess += *c - limit;
            *c = limit;
        }
    }

    // Redistribute clipped counts uniformly across all 256 bins (with evenly spaced remainder).
    let base = excess / 256;
    let rem = (excess % 256) as usize;
    if base > 0 {
        for c in &mut clipped {
            *c += base;
        }
    }
    if rem > 0 {
        for k in 0..rem {
            let idx = (k * 256) / rem;
            clipped[idx] += 1;
        }
    }

    let mut cdf = 0u64;
    let mut min_cdf = None;
    let mut cdfs = [0u64; 256];
    for (i, &c) in clipped.iter().enumerate() {
        cdf += c;
        cdfs[i] = cdf;
        if min_cdf.is_none() && c > 0 {
            min_cdf = Some(cdf);
        }
    }
    let min_c = min_cdf.unwrap_or(0);
    let denom = tile_pixels.saturating_sub(min_c);
    if denom == 0 {
        for (v, slot) in lut.iter_mut().enumerate() {
            *slot = v as u8;
        }
        return lut;
    }
    let half = denom / 2;
    for (i, slot) in lut.iter_mut().enumerate() {
        let num = cdfs[i].saturating_sub(min_c);
        *slot = ((num.saturating_mul(255).saturating_add(half)) / denom).min(255) as u8;
    }
    lut
}

/// Precomputed 1D tile partition and interpolation taps along one image axis.
struct TileAxis {
    grid: usize,
    bounds: Vec<usize>,
    t0: Vec<usize>,
    t1: Vec<usize>,
    frac: Vec<f32>,
}

fn tile_interp_axis(len: usize, grid: usize) -> TileAxis {
    let mut bounds = Vec::with_capacity(grid + 1);
    for t in 0..=grid {
        bounds.push(((t as u128 * len as u128) / grid as u128) as usize);
    }
    let centers: Vec<f64> = (0..grid).map(|t| 0.5 * (bounds[t] + bounds[t + 1]) as f64).collect();
    let mut t0 = Vec::with_capacity(len);
    let mut t1 = Vec::with_capacity(len);
    let mut frac = Vec::with_capacity(len);
    let mut hi = 1usize;
    for p in 0..len {
        let pos = p as f64 + 0.5;
        if grid == 1 || pos <= centers[0] {
            t0.push(0);
            t1.push(0);
            frac.push(0.0);
        } else if pos >= centers[grid - 1] {
            t0.push(grid - 1);
            t1.push(grid - 1);
            frac.push(0.0);
        } else {
            while hi + 1 < grid && centers[hi] < pos {
                hi += 1;
            }
            let lo = hi - 1;
            let span = (centers[hi] - centers[lo]).max(1e-9);
            let a = ((pos - centers[lo]) / span).clamp(0.0, 1.0) as f32;
            t0.push(lo);
            t1.push(hi);
            frac.push(a);
        }
    }
    TileAxis { grid, bounds, t0, t1, frac }
}

/// Equalizes one single-channel `[h, w]` plane using CLAHE tile LUTs and bilinear tile interpolation.
fn clahe_plane(
    plane: &[u8],
    w: usize,
    h: usize,
    ax: &TileAxis,
    ay: &TileAxis,
    clip_limit: f32,
    out: &mut [u8],
) {
    let (gx, gy) = (ax.grid, ay.grid);
    let mut luts = vec![[0u8; 256]; gy * gx];
    for ty in 0..gy {
        let y0 = ay.bounds[ty];
        let y1 = ay.bounds[ty + 1];
        for tx in 0..gx {
            let x0 = ax.bounds[tx];
            let x1 = ax.bounds[tx + 1];
            let mut hist = [0u64; 256];
            for y in y0..y1 {
                let row = &plane[y * w..(y + 1) * w];
                for &v in &row[x0..x1] {
                    hist[v as usize] += 1;
                }
            }
            let tile_pixels = ((y1 - y0) as u64) * ((x1 - x0) as u64);
            luts[ty * gx + tx] = build_clahe_lut(&hist, tile_pixels, clip_limit);
        }
    }

    for y in 0..h {
        let (r0, r1, wy) = (ay.t0[y] * gx, ay.t1[y] * gx, ay.frac[y]);
        let src_row = &plane[y * w..(y + 1) * w];
        let dst_row = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            let v = src_row[x] as usize;
            let (c0, c1, wx) = (ax.t0[x], ax.t1[x], ax.frac[x]);
            let v00 = luts[r0 + c0][v] as f32;
            let v10 = luts[r0 + c1][v] as f32;
            let v01 = luts[r1 + c0][v] as f32;
            let v11 = luts[r1 + c1][v] as f32;
            let top = v00 + (v10 - v00) * wx;
            let bot = v01 + (v11 - v01) * wx;
            let interp = top + (bot - top) * wy;
            dst_row[x] = interp.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in place to `img`.
///
/// Empty images or invalid grid/clip parameters (`tile_grid` containing `0`, non-finite or `<= 0` `clip_limit`)
/// are safe no-ops. Constant images and greyscale chromaticity (in [`ClaheSpace::Luminance`] mode) are
/// preserved accurately.
pub fn clahe(img: &mut Image, cfg: &ClaheConfig) {
    assert_image_data(img);
    let (w, h) = (img.width as usize, img.height as usize);
    if w == 0
        || h == 0
        || cfg.tile_grid[0] == 0
        || cfg.tile_grid[1] == 0
        || !cfg.clip_limit.is_finite()
        || cfg.clip_limit <= 0.0
    {
        return;
    }
    let gx = (cfg.tile_grid[0] as usize).min(w).max(1);
    let gy = (cfg.tile_grid[1] as usize).min(h).max(1);
    let ax = tile_interp_axis(w, gx);
    let ay = tile_interp_axis(h, gy);
    let n = w * h;
    let total = n * 3;

    match cfg.space {
        ClaheSpace::Luminance => {
            let mut luma = vec![0u8; n];
            for (dst, px) in luma.iter_mut().zip(img.data[..total].as_chunks::<3>().0) {
                *dst = luma_bt601(px[0], px[1], px[2]);
            }
            let mut eq = vec![0u8; n];
            clahe_plane(&luma, w, h, &ax, &ay, cfg.clip_limit, &mut eq);
            for ((px, &y_old), &y_new) in
                img.data[..total].as_chunks_mut::<3>().0.iter_mut().zip(&luma).zip(&eq)
            {
                let dy = y_new as i32 - y_old as i32;
                px[0] = (px[0] as i32 + dy).clamp(0, 255) as u8;
                px[1] = (px[1] as i32 + dy).clamp(0, 255) as u8;
                px[2] = (px[2] as i32 + dy).clamp(0, 255) as u8;
            }
        }
        ClaheSpace::PerChannel => {
            let mut ch_in = vec![0u8; n];
            let mut ch_out = vec![0u8; n];
            for ch in 0..3 {
                for (dst, px) in ch_in.iter_mut().zip(img.data[..total].as_chunks::<3>().0) {
                    *dst = px[ch];
                }
                clahe_plane(&ch_in, w, h, &ax, &ay, cfg.clip_limit, &mut ch_out);
                for (px, &v) in img.data[..total].as_chunks_mut::<3>().0.iter_mut().zip(&ch_out) {
                    px[ch] = v;
                }
            }
        }
    }
}

/// Randomly applies [`clahe`] to `sample`, clamping and sanitising bounding boxes.
pub fn random_clahe(sample: &mut Sample, cfg: &ClaheConfig, rng: &mut Rng) -> bool {
    if cfg.validate().is_err() || !rng.chance(cfg.prob) {
        return false;
    }
    clamp_and_filter_boxes(&mut sample.boxes, sample.image.width, sample.image.height, None);
    if sample.image.width == 0 || sample.image.height == 0 {
        return false;
    }
    clahe(&mut sample.image, cfg);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BBox;

    #[test]
    fn auto_contrast_stretches_range_respects_cutoff_and_keeps_constants() {
        // Low-contrast gradient [60..140] expands to full [0..255].
        let mut img = Image::new(17, 5, [0, 0, 0]);
        for y in 0..5 {
            for x in 0..17 {
                let v = (60 + x * 5) as u8;
                img.set_pixel(x, y, [v, v, v]);
            }
        }
        auto_contrast(&mut img, 0.0);
        assert_eq!(img.pixel(0, 0), [0, 0, 0]);
        assert_eq!(img.pixel(16, 4), [255, 255, 255]);
        assert_eq!(img.pixel(8, 2), [128, 128, 128]);

        // Outlier pixels (0 and 255) ignored when cutoff >= 2% on a 100-pixel image.
        let mut with_outliers = Image::new(10, 10, [100, 100, 100]);
        for x in 0..10 {
            for y in 0..10 {
                let v = if x < 5 { 90 } else { 150 };
                with_outliers.set_pixel(x, y, [v, v, v]);
            }
        }
        with_outliers.set_pixel(0, 0, [0, 0, 0]);
        with_outliers.set_pixel(9, 9, [255, 255, 255]);
        auto_contrast(&mut with_outliers, 2.0);
        assert_eq!(with_outliers.pixel(1, 0), [0, 0, 0]);
        assert_eq!(with_outliers.pixel(8, 9), [255, 255, 255]);

        // Constant image and 1x1 image stay bit-exact.
        let mut solid = Image::new(8, 8, [77, 123, 211]);
        auto_contrast(&mut solid, 5.0);
        assert!(solid.data.as_chunks::<3>().0.iter().all(|&p| p == [77, 123, 211]));

        let mut one = Image::new(1, 1, [42, 84, 126]);
        auto_contrast(&mut one, 0.0);
        assert_eq!(one.pixel(0, 0), [42, 84, 126]);
    }

    #[test]
    fn clahe_boosts_local_contrast_preserves_constants_and_handles_small_images() {
        // Two-tone image: left half dark gradient [10..26), right half bright gradient [210..226).
        // 64x32 with a 4x2 grid gives 16x16 = 256 pixels per tile.
        let mut img = Image::new(64, 32, [0, 0, 0]);
        for y in 0..32 {
            for x in 0..64 {
                let v = if x < 32 { (10 + (x % 16)) as u8 } else { (210 + (x % 16)) as u8 };
                img.set_pixel(x, y, [v, v, v]);
            }
        }
        let cfg = ClaheConfig { prob: 1.0, tile_grid: [4, 2], clip_limit: 4.0, space: ClaheSpace::Luminance };
        let mut eq = img.clone();
        clahe(&mut eq, &cfg);
        // Local spread inside dark left tile increases significantly beyond the original 15-level span.
        let left_vals: Vec<u8> = (0..16).map(|x| eq.pixel(x, 8)[0]).collect();
        let spread = left_vals.iter().max().unwrap() - left_vals.iter().min().unwrap();
        assert!(
            spread > 40,
            "expected local contrast boost in dark tile, got spread={spread} ({left_vals:?})"
        );
        // Luminance mode preserves greyscale (R == G == B) everywhere.
        assert!(eq.data.as_chunks::<3>().0.iter().all(|p| p[0] == p[1] && p[1] == p[2]));

        // PerChannel mode also boosts local contrast and preserves constant images.
        let cfg_pc = ClaheConfig { space: ClaheSpace::PerChannel, ..cfg };
        let mut eq_pc = img.clone();
        clahe(&mut eq_pc, &cfg_pc);
        assert_ne!(eq_pc.data, img.data);

        let mut solid = Image::new(13, 11, [60, 120, 180]);
        clahe(&mut solid, &cfg);
        assert!(solid.data.as_chunks::<3>().0.iter().all(|&p| p == [60, 120, 180]));
        clahe(&mut solid, &cfg_pc);
        assert!(solid.data.as_chunks::<3>().0.iter().all(|&p| p == [60, 120, 180]));

        // 1x1, smaller-than-grid (3x5 with 8x8 grid), and 0x0 never panic.
        let mut one = Image::new(1, 1, [90, 100, 110]);
        clahe(&mut one, &cfg);
        assert_eq!(one.pixel(0, 0), [90, 100, 110]);

        let mut small = Image::new(3, 5, [0, 0, 0]);
        for y in 0..5 {
            for x in 0..3 {
                small.set_pixel(x, y, [(20 + x * 30) as u8, (40 + y * 20) as u8, 128]);
            }
        }
        clahe(&mut small, &ClaheConfig { tile_grid: [8, 8], ..cfg });
        // Wide-strip tile boundaries must not overflow 32-bit `usize` (`t * len / grid`).
        let wide_ax = tile_interp_axis(70_000, 62_000);
        assert_eq!(wide_ax.bounds.first().copied(), Some(0));
        assert_eq!(wide_ax.bounds.last().copied(), Some(70_000));
        assert!(wide_ax.bounds.windows(2).all(|w| w[0] <= w[1]));
        let mut empty = Image { width: 0, height: 0, data: Vec::new() };
        clahe(&mut empty, &cfg);
    }

    #[test]
    fn random_auto_contrast_and_clahe_are_deterministic_and_clamp_boxes() {
        let mut img = Image::new(16, 16, [0, 0, 0]);
        for y in 0..16 {
            for x in 0..16 {
                let v = (50 + ((x * 7 + y * 5) % 80)) as u8;
                img.set_pixel(x, y, [v, v.saturating_add(10), v.saturating_sub(10)]);
            }
        }
        let base = Sample {
            image: img,
            boxes: vec![
                BBox::new(0, 0.5, 0.5, 0.4, 0.4),
                BBox::from_xyxy(1, [0.7, 0.7, 1.4, 1.2]),
                BBox::new(2, 0.5, 0.5, -0.1, 0.2),
            ],
        };

        let ac_cfg = AutoContrastConfig { prob: 1.0, cutoff: 1.0 };
        let (mut a1, mut a2) = (base.clone(), base.clone());
        assert!(random_auto_contrast(&mut a1, &ac_cfg, &mut Rng::new(11)));
        assert!(random_auto_contrast(&mut a2, &ac_cfg, &mut Rng::new(11)));
        assert_eq!(a1, a2);
        assert_eq!(a1.boxes.len(), 2);
        assert_eq!(a1.boxes[0], base.boxes[0]);
        assert!((a1.boxes[1].xyxy()[2] - 1.0).abs() < 1e-6);

        let cl_cfg =
            ClaheConfig { prob: 1.0, tile_grid: [4, 4], clip_limit: 2.5, space: ClaheSpace::Luminance };
        let (mut c1, mut c2) = (base.clone(), base.clone());
        assert!(random_clahe(&mut c1, &cl_cfg, &mut Rng::new(19)));
        assert!(random_clahe(&mut c2, &cl_cfg, &mut Rng::new(19)));
        assert_eq!(c1, c2);
        assert_eq!(c1.boxes, a1.boxes);

        // Invalid configs rejected cleanly.
        assert!(AutoContrastConfig { cutoff: 50.0, ..Default::default() }.validate().is_err());
        assert!(AutoContrastConfig { cutoff: -1.0, ..Default::default() }.validate().is_err());
        assert!(ClaheConfig { tile_grid: [0, 4], ..Default::default() }.validate().is_err());
        assert!(ClaheConfig { clip_limit: 0.0, ..Default::default() }.validate().is_err());
    }
}
