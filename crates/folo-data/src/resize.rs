//! Fast bilinear resize for 8-bit RGB images.
//!
//! Pixel centres are aligned (`src = (dst + 0.5) * scale - 0.5`), so pixel *edges* map linearly and box coordinates
//! scale exactly by `dst / src`. Weights are 11-bit fixed point; rows are processed in parallel. When shrinking by 2x
//! or more an integer box filter runs first so large downscales do not alias.

use crate::image::Image;
use crate::par::{for_bands, for_rows};
use crate::simd::assert_image_data;
use crate::simd::bilinear::{
    hinterp_taps_to_planar_f32_div255, hinterp_taps_to_planar_f32_scale, hinterp_taps_to_planar_u8,
    hinterp_taps_to_rgb, hinterp_taps_to_rgb0_u32, vblend_rgb_to_rgba32, vshift_rgb_to_rgba32,
};
use crate::simd::box_filter::{box_hreduce_u32_to_u8, sum_rows_u8_to_u32};
use crate::simd::planar::{
    rgb_to_planar_f32_div255, rgb_to_planar_f32_scale, rgb_to_planar_u8, rgb_to_rgb0_u32,
};

/// Fixed-point bits of the bilinear weights (`1 << RESIZE_WEIGHT_BITS` is a weight of 1).
pub const RESIZE_WEIGHT_BITS: u32 = 11;
const BITS: u32 = RESIZE_WEIGHT_BITS;
const ONE: u32 = 1 << BITS;

/// Per destination coordinate: the two source indices and the weight of the second one.
struct Taps {
    i0: Vec<u32>,
    i1: Vec<u32>,
    w: Vec<u32>,
}

/// Builds taps for `dst` outputs over `src` inputs, where the source has `src_extent` real pixels of size `cell`
/// (`cell > 1` after a box prefilter whose last cell may be partial).
fn taps(dst: u32, src: u32, src_extent: f64) -> Taps {
    let scale = src_extent / dst as f64;
    let mut t = Taps { i0: Vec::with_capacity(dst as usize), i1: Vec::new(), w: Vec::new() };
    t.i1.reserve(dst as usize);
    t.w.reserve(dst as usize);
    let max = (src - 1) as f64;
    for d in 0..dst {
        let s = ((d as f64 + 0.5) * scale - 0.5).clamp(0.0, max);
        let i0 = s.floor() as u32;
        let i1 = (i0 + 1).min(src - 1);
        let w = (((s - i0 as f64) * ONE as f64).round() as u32).min(ONE);
        t.i0.push(i0);
        t.i1.push(i1);
        t.w.push(w);
    }
    t
}

/// The exact sampling of [`resize_into`] for a `src_w x src_h` → `width x height` resize, so that another
/// implementation (the GPU letterbox) can reproduce it bit for bit:
///
/// 1. When shrinking by 2x or more on an axis, an integer box prefilter: the source is reduced to
///    `ceil(src_w / fx) x ceil(src_h / fy)` pixels, each channel of a pixel the rounded mean
///    `(sum + n / 2) / n` of its `fx x fy` block (the last block on an axis may be partial; `n` counts what is
///    there).
/// 2. Bilinear taps on that image: destination pixel `(x, y)` mixes columns `x0[x]`, `x1[x]` with weight
///    `wx[x]` on `x1` and rows `y0[y]`, `y1[y]` with `wy[y]` (weights in `1 / 2^RESIZE_WEIGHT_BITS`):
///    `t = top[x0]·(1 − wx) + top[x1]·wx`, `u` likewise on the bottom row, then
///    `(t·(1 − wy) + u·wy + 2^(2·BITS − 1)) >> (2·BITS)`, all in integers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResizePlan {
    /// Box-prefilter factors `[fx, fy]`; `[1, 1]` means no prefilter.
    pub prefilter: [u32; 2],
    /// Left source column of each destination column (prefiltered pixels).
    pub x0: Vec<u32>,
    /// Right source column of each destination column.
    pub x1: Vec<u32>,
    /// Weight of `x1`, in `1 / 2^RESIZE_WEIGHT_BITS`.
    pub wx: Vec<u32>,
    /// Top source row of each destination row.
    pub y0: Vec<u32>,
    /// Bottom source row of each destination row.
    pub y1: Vec<u32>,
    /// Weight of `y1`, in `1 / 2^RESIZE_WEIGHT_BITS`.
    pub wy: Vec<u32>,
}

/// The [`ResizePlan`] of a `src_w x src_h` → `width x height` resize.
///
/// # Panics
/// If any size is zero.
pub fn resize_plan(src_w: u32, src_h: u32, width: u32, height: u32) -> ResizePlan {
    assert!(width > 0 && height > 0 && src_w > 0 && src_h > 0, "resize to or from an empty image");
    let fx = (src_w / width).max(1);
    let fy = (src_h / height).max(1);
    let (pw, ph, ext_w, ext_h) = if fx >= 2 || fy >= 2 {
        (src_w.div_ceil(fx), src_h.div_ceil(fy), src_w as f64 / fx as f64, src_h as f64 / fy as f64)
    } else {
        (src_w, src_h, src_w as f64, src_h as f64)
    };
    let xt = taps(width, pw, ext_w);
    let yt = taps(height, ph, ext_h);
    ResizePlan { prefilter: [fx, fy], x0: xt.i0, x1: xt.i1, wx: xt.w, y0: yt.i0, y1: yt.i1, wy: yt.w }
}

/// Resizes `src` to `width x height` with bilinear filtering.
pub fn resize(src: &Image, width: u32, height: u32) -> Image {
    let mut dst = crate::buffers::scratch(width as usize * height as usize * 3);
    resize_into(src, &mut dst, width as usize * 3, 0, width, height);
    Image { width, height, data: dst }
}

/// Resizes `src` into a `width x height` window of a larger RGB buffer: row `y` of the window starts at
/// `dst[y * dst_stride + x_offset * 3]`. `dst` must hold at least `height` rows of `dst_stride` bytes.
pub fn resize_into(src: &Image, dst: &mut [u8], dst_stride: usize, x_offset: usize, width: u32, height: u32) {
    assert!(width > 0 && height > 0 && src.width > 0 && src.height > 0, "resize to or from an empty image");
    let x_end = x_offset
        .checked_add(width as usize)
        .and_then(|n| n.checked_mul(3))
        .expect("resize window overflows usize");
    assert!(x_end <= dst_stride, "resize window exceeds the row stride");
    let total = dst_stride.checked_mul(height as usize).expect("resize destination size overflow");
    let dst = &mut dst[..total];
    let rows = RowResizer::new(src, width, height);
    let row_bytes = width as usize * 3;
    for_bands(dst, dst_stride, |y0, band| {
        let mut scratch = rows.band_scratch();
        for (i, out) in band.chunks_exact_mut(dst_stride).enumerate() {
            rows.row_with_scratch(y0 + i, &mut out[x_offset * 3..x_offset * 3 + row_bytes], &mut scratch);
        }
    });
}

/// One `width x height` resize of an image, produced a row at a time, so callers can write each row wherever (and in
/// whatever format) they need without an intermediate image. The box prefilter, when the plan needs one, runs once in
/// [`RowResizer::new`].
pub(crate) struct RowResizer<'a> {
    src: &'a Image,
    /// The box-prefiltered source (`None`: no prefilter).
    filtered: Option<Image>,
    /// `None` for an identity resize (rows are copied).
    plan: Option<ResizePlan>,
    width: usize,
}

/// Per-band scratch holding the vertically blended `[R, G, B, 0]` `u32` row (plus one duplicated trailing column so
/// clamped `x1 == src_w - 1` reads `v[x0 + 1]`).
pub(crate) struct RowScratch {
    v: Vec<u32>,
    cached: Option<(u32, u32, u32)>,
}

impl<'a> RowResizer<'a> {
    pub(crate) fn new(src: &'a Image, width: u32, height: u32) -> Self {
        assert!(
            width > 0 && height > 0 && src.width > 0 && src.height > 0,
            "resize to or from an empty image"
        );
        assert_image_data(src);
        if src.width == width && src.height == height {
            return RowResizer { src, filtered: None, plan: None, width: width as usize };
        }
        let plan = resize_plan(src.width, src.height, width, height);
        let [fx, fy] = plan.prefilter;
        // Integer box prefilter for large reductions.
        let filtered = (fx >= 2 || fy >= 2).then(|| box_downsample(src, fx, fy));
        let sw = filtered.as_ref().unwrap_or(src).width as usize;
        assert!(plan.x0.iter().all(|&k| (k as usize) < sw), "resize tap out of source bounds");
        RowResizer { src, filtered, plan: Some(plan), width: width as usize }
    }

    /// Allocates reusable vertical-blend scratch for a band of rows.
    pub(crate) fn band_scratch(&self) -> RowScratch {
        let len = match &self.plan {
            None => 0,
            Some(_) => (self.filtered.as_ref().unwrap_or(self.src).width as usize + 1) * 4,
        };
        RowScratch { v: vec![0u32; len], cached: None }
    }

    /// Populates `scratch.v` with the vertically blended source row for destination row `y`.
    #[inline]
    fn fill_v<'b>(&self, y: usize, plan: &ResizePlan, scratch: &'b mut RowScratch) -> &'b [u32] {
        let (y0, y1, wy) = (plan.y0[y], plan.y1[y], plan.wy[y]);
        let key = (y0, y1, wy);
        let s = self.filtered.as_ref().unwrap_or(self.src);
        let sw = s.width as usize;
        let need = (sw + 1) * 4;
        if scratch.v.len() < need {
            scratch.v.resize(need, 0);
            scratch.cached = None;
        }
        if scratch.cached != Some(key) {
            let sstride = s.stride();
            let top = &s.data[y0 as usize * sstride..][..sstride];
            if y0 == y1 || wy == 0 {
                vshift_rgb_to_rgba32(top, &mut scratch.v[..sw * 4]);
            } else {
                let bot = &s.data[y1 as usize * sstride..][..sstride];
                vblend_rgb_to_rgba32(top, bot, wy, &mut scratch.v[..sw * 4]);
            }
            // Duplicate the last column so `v[x0 + 1]` equals `v[x1]` even when `x0 == sw - 1`.
            let last = (sw - 1) * 4;
            scratch.v.copy_within(last..last + 4, sw * 4);
            scratch.cached = Some(key);
        }
        &scratch.v[..need]
    }

    /// Writes destination row `y` (`width` interleaved RGB pixels) into `out[..width * 3]`.
    #[inline]
    pub(crate) fn row_with_scratch(&self, y: usize, out: &mut [u8], scratch: &mut RowScratch) {
        let out = &mut out[..self.width * 3];
        let Some(plan) = &self.plan else {
            out.copy_from_slice(&self.src.data[y * out.len()..(y + 1) * out.len()]);
            return;
        };
        let v = self.fill_v(y, plan, scratch);
        // SAFETY: `RowResizer::new` asserted `plan.x0[x] < sw` and `fill_v` returns `(sw + 1) * 4` elements.
        unsafe { hinterp_taps_to_rgb(v, &plan.x0, &plan.wx, out) };
    }

    /// Writes destination row `y` straight into planar `u8` (`r`, `g`, `b` of length `width`).
    #[inline]
    pub(crate) fn row_planar_u8(
        &self,
        y: usize,
        r: &mut [u8],
        g: &mut [u8],
        b: &mut [u8],
        scratch: &mut RowScratch,
    ) {
        let Some(plan) = &self.plan else {
            let stride = self.width * 3;
            rgb_to_planar_u8(&self.src.data[y * stride..(y + 1) * stride], r, g, b, false);
            return;
        };
        let v = self.fill_v(y, plan, scratch);
        // SAFETY: `RowResizer::new` asserted `plan.x0[x] < sw` and `fill_v` returns `(sw + 1) * 4` elements.
        unsafe { hinterp_taps_to_planar_u8(v, &plan.x0, &plan.wx, r, g, b) };
    }

    /// Writes destination row `y` straight into planar `f32` (`v as f32 * scale`).
    #[inline]
    pub(crate) fn row_planar_f32_scale(
        &self,
        y: usize,
        scale: f32,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        scratch: &mut RowScratch,
    ) {
        let Some(plan) = &self.plan else {
            let stride = self.width * 3;
            rgb_to_planar_f32_scale(&self.src.data[y * stride..(y + 1) * stride], r, g, b, scale);
            return;
        };
        let v = self.fill_v(y, plan, scratch);
        // SAFETY: `RowResizer::new` asserted `plan.x0[x] < sw` and `fill_v` returns `(sw + 1) * 4` elements.
        unsafe { hinterp_taps_to_planar_f32_scale(v, &plan.x0, &plan.wx, scale, r, g, b) };
    }

    /// Writes destination row `y` straight into planar `f32` (`v as f32 / 255.0`).
    #[inline]
    pub(crate) fn row_planar_f32_div255(
        &self,
        y: usize,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
        scratch: &mut RowScratch,
    ) {
        let Some(plan) = &self.plan else {
            let stride = self.width * 3;
            rgb_to_planar_f32_div255(&self.src.data[y * stride..(y + 1) * stride], r, g, b, false);
            return;
        };
        let v = self.fill_v(y, plan, scratch);
        // SAFETY: `RowResizer::new` asserted `plan.x0[x] < sw` and `fill_v` returns `(sw + 1) * 4` elements.
        unsafe { hinterp_taps_to_planar_f32_div255(v, &plan.x0, &plan.wx, r, g, b) };
    }

    /// Writes destination row `y` straight into packed `[R, G, B, 0]` `u32` words.
    #[inline]
    pub(crate) fn row_rgb0_u32(&self, y: usize, dst: &mut [u32], scratch: &mut RowScratch) {
        let Some(plan) = &self.plan else {
            let stride = self.width * 3;
            rgb_to_rgb0_u32(&self.src.data[y * stride..(y + 1) * stride], dst);
            return;
        };
        let v = self.fill_v(y, plan, scratch);
        // SAFETY: `RowResizer::new` asserted `plan.x0[x] < sw` and `fill_v` returns `(sw + 1) * 4` elements.
        unsafe { hinterp_taps_to_rgb0_u32(v, &plan.x0, &plan.wx, dst) };
    }
}

impl Drop for RowResizer<'_> {
    fn drop(&mut self) {
        if let Some(img) = self.filtered.take() {
            crate::buffers::recycle(img.data);
        }
    }
}

/// Averages `fx x fy` blocks. The last block in each direction may be partial; it averages what is there.
///
/// Each output band reuses one `u32` row accumulator (summed with AVX2), then reduces each pixel's `fx` columns
/// and divides with rounding, `(sum + n / 2) / n`, using AVX2 horizontal reduction (or a scalar multiply-shift
/// [`RoundDiv`]). Blocks too large for `u32` sums take the `u64` path.
fn box_downsample(src: &Image, fx: u32, fy: u32) -> Image {
    assert_image_data(src);
    if 255 * fx as u64 * fy as u64 > u32::MAX as u64 / 2 {
        return box_downsample_wide(src, fx, fy);
    }
    let (w, h) = (src.width as usize, src.height as usize);
    let (fx, fy) = (fx as usize, fy as usize);
    let ow = w.div_ceil(fx);
    let oh = h.div_ceil(fy);
    let mut out = crate::buffers::scratch(ow * oh * 3);
    let stride = w * 3;
    // Columns in the last (maybe partial) block.
    let last_w = w - (ow - 1) * fx;
    for_bands(&mut out, ow * 3, |oy0, band| {
        let mut col = vec![0u32; stride];
        for (bi, row) in band.chunks_exact_mut(ow * 3).enumerate() {
            let oy = oy0 + bi;
            let y0 = oy * fy;
            let y1 = (y0 + fy).min(h);
            sum_rows_u8_to_u32(&mut col, &src.data[y0 * stride..y1 * stride], stride);
            let ny = y1 - y0;
            let full_n = (fx * ny) as u32;
            let last = RoundDiv::new((last_w * ny) as u32);
            let (body, tail) = col.split_at((ow - 1) * fx * 3);
            let (row_body, row_tail) = row.split_at_mut((ow - 1) * 3);
            if !row_body.is_empty() && !box_hreduce_u32_to_u8(body, row_body, fx, full_n) {
                let full = RoundDiv::new(full_n);
                if fx == 2 && full_n.is_power_of_two() {
                    let shift = full_n.trailing_zeros();
                    let half = full_n >> 1;
                    for (px, block) in row_body.as_chunks_mut::<3>().0.iter_mut().zip(body.as_chunks::<6>().0)
                    {
                        px[0] = ((block[0] + block[3] + half) >> shift) as u8;
                        px[1] = ((block[1] + block[4] + half) >> shift) as u8;
                        px[2] = ((block[2] + block[5] + half) >> shift) as u8;
                    }
                } else if fx == 2 {
                    for (px, block) in row_body.as_chunks_mut::<3>().0.iter_mut().zip(body.as_chunks::<6>().0)
                    {
                        px[0] = full.div(block[0] + block[3]) as u8;
                        px[1] = full.div(block[1] + block[4]) as u8;
                        px[2] = full.div(block[2] + block[5]) as u8;
                    }
                } else if fx == 3 {
                    for (px, block) in row_body.as_chunks_mut::<3>().0.iter_mut().zip(body.as_chunks::<9>().0)
                    {
                        px[0] = full.div(block[0] + block[3] + block[6]) as u8;
                        px[1] = full.div(block[1] + block[4] + block[7]) as u8;
                        px[2] = full.div(block[2] + block[5] + block[8]) as u8;
                    }
                } else {
                    for (px, block) in
                        row_body.as_chunks_mut::<3>().0.iter_mut().zip(body.chunks_exact(fx * 3))
                    {
                        let mut s = [0u32; 3];
                        for p in block.as_chunks::<3>().0 {
                            s[0] += p[0];
                            s[1] += p[1];
                            s[2] += p[2];
                        }
                        px[0] = full.div(s[0]) as u8;
                        px[1] = full.div(s[1]) as u8;
                        px[2] = full.div(s[2]) as u8;
                    }
                }
            }
            let mut s = [0u32; 3];
            for p in tail.as_chunks::<3>().0 {
                s[0] += p[0];
                s[1] += p[1];
                s[2] += p[2];
            }
            for c in 0..3 {
                row_tail[c] = last.div(s[c]) as u8;
            }
        }
    });
    Image { width: ow as u32, height: oh as u32, data: out }
}

/// `(x + n / 2) / n` for the block sums of [`box_downsample`] (`x <= 255 n`), exactly, as
/// `((x + n / 2) · m) >> 40` with `m = ceil(2^40 / n)`. With `q = x + n / 2 < 256 n`, the product over `2^40` is
/// `q / n + q·e / 2^40` for some `0 <= e < 1`, and the error `q / 2^40 < 256 n / 2^40` stays below `1 / n` (the
/// smallest gap from a non-integer `q / n` to the next integer) while `n <= 2^16`. Larger `n` divide.
#[derive(Clone, Copy)]
struct RoundDiv {
    n: u32,
    m: u64,
}

impl RoundDiv {
    const SHIFT: u32 = 40;

    fn new(n: u32) -> Self {
        debug_assert!(n > 0);
        RoundDiv { n, m: (1u64 << Self::SHIFT).div_ceil(n as u64) }
    }

    #[inline]
    fn div(self, x: u32) -> u32 {
        let q = x as u64 + (self.n / 2) as u64;
        if self.n <= 1 << 16 { ((q * self.m) >> Self::SHIFT) as u32 } else { (q / self.n as u64) as u32 }
    }
}

/// [`box_downsample`] with `u64` sums, for blocks whose sums could overflow `u32`.
fn box_downsample_wide(src: &Image, fx: u32, fy: u32) -> Image {
    assert_image_data(src);
    let (w, h) = (src.width as usize, src.height as usize);
    let (fx, fy) = (fx as usize, fy as usize);
    let ow = w.div_ceil(fx);
    let oh = h.div_ceil(fy);
    let mut out = crate::buffers::scratch(ow * oh * 3);
    let stride = w * 3;
    for_rows(&mut out, ow * 3, |oy, row| {
        let y0 = oy * fy;
        let y1 = (y0 + fy).min(h);
        let mut acc = vec![0u64; ow * 3];
        for y in y0..y1 {
            let srow = &src.data[y * stride..(y + 1) * stride];
            for ox in 0..ow {
                let x0 = ox * fx;
                let x1 = (x0 + fx).min(w);
                let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
                for p in srow[x0 * 3..x1 * 3].as_chunks::<3>().0.iter() {
                    r += p[0] as u64;
                    g += p[1] as u64;
                    b += p[2] as u64;
                }
                acc[ox * 3] += r;
                acc[ox * 3 + 1] += g;
                acc[ox * 3 + 2] += b;
            }
        }
        for ox in 0..ow {
            let n = (((ox * fx + fx).min(w) - ox * fx) * (y1 - y0)) as u64;
            for c in 0..3 {
                row[ox * 3 + c] = ((acc[ox * 3 + c] + n / 2) / n) as u8;
            }
        }
    });
    Image { width: ow as u32, height: oh as u32, data: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant() {
        let mut img = Image::new(7, 5, [10, 20, 30]);
        img.set_pixel(3, 2, [200, 100, 0]);
        assert_eq!(resize(&img, 7, 5), img);
        let flat = Image::new(37, 23, [9, 99, 199]);
        for (w, h) in [(80, 40), (5, 3), (36, 23), (1, 1), (200, 7)] {
            let r = resize(&flat, w, h);
            assert!(r.data.chunks(3).all(|p| p == [9, 99, 199]), "{w}x{h}");
        }
    }

    #[test]
    fn upscale_2x_interpolates() {
        // Two pixels 0 and 200 -> 4 pixels: centres at -0.25, 0.25, 0.75, 1.25 in source space.
        let img = Image::from_raw(2, 1, vec![0, 0, 0, 200, 200, 200]).unwrap();
        let r = resize(&img, 4, 1);
        let v: Vec<u8> = r.data.chunks(3).map(|p| p[0]).collect();
        assert_eq!(v, vec![0, 50, 150, 200]);
    }

    #[test]
    fn large_downscale_averages() {
        // A fine checkerboard averages to mid grey instead of aliasing to black or white.
        let mut img = Image::new(64, 64, [0, 0, 0]);
        for y in 0..64 {
            for x in 0..64 {
                if (x + y) % 2 == 0 {
                    img.set_pixel(x, y, [255, 255, 255]);
                }
            }
        }
        let r = resize(&img, 8, 8);
        assert!(r.data.iter().all(|&v| (120..=135).contains(&v)), "{:?}", &r.data[..12]);
    }

    #[test]
    fn round_div_is_exact() {
        for n in (1..=4096).chain((4097..=1 << 16).step_by(97)).chain([1 << 16, (1 << 16) + 1, 1 << 20]) {
            let d = RoundDiv::new(n);
            for k in 0..=255u32 {
                for x in [k * n, k * n + n / 2, k * n + n - 1] {
                    if x <= 255 * n {
                        assert_eq!(d.div(x), (x + n / 2) / n, "{x} / {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn box_downsample_matches_the_u64_reference() {
        for (w, h, fx, fy) in [
            (1280, 720, 2, 2),
            (1921, 1081, 3, 3),
            (37, 23, 5, 2),
            (8, 9, 1, 3),
            (300, 7, 7, 7),
            (5, 5, 5, 5),
        ] {
            let mut img = Image::new(w, h, [0, 0, 0]);
            for (i, v) in img.data.iter_mut().enumerate() {
                *v = (i.wrapping_mul(2654435761) >> 9) as u8;
            }
            assert_eq!(
                box_downsample(&img, fx, fy),
                box_downsample_wide(&img, fx, fy),
                "{w}x{h} / {fx}x{fy}"
            );
        }
        let white = Image::new(600, 500, [255, 255, 255]);
        assert!(box_downsample(&white, 300, 250).data.iter().all(|&v| v == 255));
    }

    #[test]
    fn huge_reduction_does_not_overflow() {
        let img = Image::new(4096, 4113, [255, 255, 255]);
        assert_eq!(resize(&img, 1, 1).data, vec![255, 255, 255]);
        assert!(resize(&img, 3, 2).data.iter().all(|&v| v == 255));
    }

    #[test]
    fn parallel_matches_serial() {
        let mut img = Image::new(301, 211, [0, 0, 0]);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        let a = resize(&img, 640, 480);
        let b = crate::par::serial(|| resize(&img, 640, 480));
        assert_eq!(a, b);
    }

    #[test]
    fn prefilter_buffer_is_recycled_in_pool() {
        let _guard = crate::buffers::TEST_POOL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let img = Image::new(640, 480, [42, 84, 126]);
        crate::buffers::pooled(|| {
            let p1 = {
                let r = RowResizer::new(&img, 200, 150);
                r.filtered.as_ref().expect("2x+ downscale uses prefilter").data.as_ptr()
            };
            let r2 = RowResizer::new(&img, 200, 150);
            let p2 = r2.filtered.as_ref().expect("2x+ downscale uses prefilter").data.as_ptr();
            assert_eq!(p1, p2, "RowResizer drop should recycle the prefilter buffer into the pool");
        });
    }
}
