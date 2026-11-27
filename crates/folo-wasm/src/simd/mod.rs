//! RGBA preprocessing kernels for the browser: SIMD128-accelerated RGBA→RGB stripping and direct
//! RGBA→planar-`f32` letterboxing (bit-exact with `folo_data::Letterbox::apply_planar_f32`).
//!
//! When compiled for `wasm32` with `-C target-feature=+simd128` (as `scripts/build-wasm.sh` does), the hot
//! loops run on 128-bit WASM SIMD instructions (`core::arch::wasm32`). Elsewhere a portable scalar fallback
//! computes the exact same bytes and floats.

use folo_data::Letterbox;
use folo_data::resize::{RESIZE_WEIGHT_BITS, resize_plan};
use folo_vision::TileRect;

const BITS: u32 = RESIZE_WEIGHT_BITS;
const ONE: u32 = 1 << BITS;
const SHIFT: u32 = 2 * BITS;
const HALF: u32 = 1 << (SHIFT - 1);

/// Reusable scratch buffers for zero-allocation repeated RGBA frame preprocessing, tile cropping, and
/// horizontal flipping on [`crate::Detector`].
#[derive(Default)]
pub(crate) struct RgbaScratch {
    v_row: Vec<u32>,
    box_col_sum: Vec<u32>,
    box_out: Vec<u8>,
    pub(crate) rgb: Vec<u8>,
    pub(crate) rgba_work: Vec<u8>,
}

/// Crop `rect` from an RGBA image (`width * height * 4` bytes) into reusable `dst` (`rect.width * rect.height * 4`
/// bytes) using contiguous row copies.
///
/// # Panics
/// If `rgba.len() != width * height * 4` or `rect` is empty / out of bounds.
pub(crate) fn crop_rgba_tile(rgba: &[u8], width: u32, height: u32, rect: TileRect, dst: &mut Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    let expected = w.checked_mul(h).and_then(|n| n.checked_mul(4)).expect("RGBA image size overflow");
    assert_eq!(rgba.len(), expected, "crop_rgba_tile: RGBA buffer length mismatch");
    assert!(
        rect.width > 0
            && rect.height > 0
            && rect.x.checked_add(rect.width).is_some_and(|e| e <= width)
            && rect.y.checked_add(rect.height).is_some_and(|e| e <= height),
        "crop_rgba_tile: tile {rect:?} out of bounds for {width}x{height}"
    );
    let dst_row = rect.width as usize * 4;
    let src_stride = w * 4;
    let x_off = rect.x as usize * 4;
    let total = dst_row.checked_mul(rect.height as usize).expect("tile RGBA size overflow");
    dst.resize(total, 0);
    for row in 0..rect.height as usize {
        let src_start = (rect.y as usize + row) * src_stride + x_off;
        let dst_start = row * dst_row;
        dst[dst_start..dst_start + dst_row].copy_from_slice(&rgba[src_start..src_start + dst_row]);
    }
}

/// Horizontally mirror an RGBA image (`width * height * 4` bytes) into reusable `dst`, matching
/// `folo_data::Image::oriented(2)` on the RGB channels.
///
/// # Panics
/// If `rgba.len() != width * height * 4`.
pub(crate) fn flip_rgba_h(rgba: &[u8], width: u32, height: u32, dst: &mut Vec<u8>) {
    let (w, h) = (width as usize, height as usize);
    let expected = w.checked_mul(h).and_then(|n| n.checked_mul(4)).expect("RGBA image size overflow");
    assert_eq!(rgba.len(), expected, "flip_rgba_h: RGBA buffer length mismatch");
    dst.resize(expected, 0);
    let stride = w * 4;
    for y in 0..h {
        let src_row = &rgba[y * stride..(y + 1) * stride];
        let dst_row = &mut dst[y * stride..(y + 1) * stride];
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        // SAFETY: `src_row.len() == dst_row.len() == w * 4`.
        unsafe {
            wasm_simd::flip_rgba_row_simd128(src_row, dst_row, w);
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        for (dst_px, src_px) in
            dst_row.as_chunks_mut::<4>().0.iter_mut().zip(src_row.as_chunks::<4>().0.iter().rev())
        {
            *dst_px = *src_px;
        }
    }
}

/// Strip alpha from `rgba` (`4 * n` bytes) into `out` (`3 * n` bytes).
///
/// # Panics
/// If `rgba.len()` is not a multiple of 4 or `out.len() != rgba.len() / 4 * 3`.
pub(crate) fn rgba_to_rgb(rgba: &[u8], out: &mut [u8]) {
    assert_eq!(rgba.len() % 4, 0, "rgba_to_rgb: input length must be a multiple of 4");
    let n = rgba.len() / 4;
    assert_eq!(out.len(), n * 3, "rgba_to_rgb: output length must be 3 * pixels");
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `rgba.len() == n * 4` and `out.len() == n * 3`.
    unsafe {
        wasm_simd::rgba_to_rgb_simd128(rgba, out, n);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    rgba_to_rgb_scalar(rgba, out);
}

#[cfg_attr(all(target_arch = "wasm32", target_feature = "simd128"), allow(dead_code))]
fn rgba_to_rgb_scalar(rgba: &[u8], out: &mut [u8]) {
    for (dst, src) in out.as_chunks_mut::<3>().0.iter_mut().zip(rgba.as_chunks::<4>().0) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
    }
}

/// Letterbox an RGBA image (`width * height * 4` bytes, alpha ignored) directly into planar `[3, out_h, out_w]`
/// `f32` scaled by `scale`, with zero intermediate RGB image allocation.
///
/// Bit-for-bit identical to stripping alpha with `rgba_to_image` and running
/// `folo_data::Letterbox::apply_planar_f32`.
///
/// # Panics
/// If `lb.orig != [width, height]`, `rgba.len() != width * height * 4`, `dst.len() != 3 * out_w * out_h`, or the
/// letterbox window lies outside `lb.out`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn letterbox_rgba_planar_f32(
    rgba: &[u8],
    width: u32,
    height: u32,
    lb: &Letterbox,
    pad_value: u8,
    scale: f32,
    dst: &mut [f32],
) {
    let mut scratch = RgbaScratch::default();
    letterbox_rgba_planar_f32_with_scratch(rgba, width, height, lb, pad_value, scale, dst, &mut scratch);
}

/// [`letterbox_rgba_planar_f32`] reusing `scratch` across repeated frames (zero per-frame heap allocations for
/// vertical blend and box-prefilter work buffers once warmed up).
#[allow(clippy::too_many_arguments)]
pub(crate) fn letterbox_rgba_planar_f32_with_scratch(
    rgba: &[u8],
    width: u32,
    height: u32,
    lb: &Letterbox,
    pad_value: u8,
    scale: f32,
    dst: &mut [f32],
    scratch: &mut RgbaScratch,
) {
    assert_eq!(lb.orig, [width, height], "letterbox geometry computed for another size");
    let (w, h) = (width as usize, height as usize);
    let expected_rgba = w.checked_mul(h).and_then(|n| n.checked_mul(4)).expect("RGBA image size overflow");
    assert_eq!(rgba.len(), expected_rgba, "RGBA buffer length mismatch");
    let [ow, oh] = lb.out.map(|v| v as usize);
    let [rw, rh] = lb.resized.map(|v| v as usize);
    let plane = ow.checked_mul(oh).expect("letterbox output size overflow");
    assert_eq!(dst.len(), plane.checked_mul(3).expect("letterbox planar size overflow"));
    if plane == 0 {
        return;
    }
    assert!(
        lb.pad[0].is_finite() && lb.pad[1].is_finite() && lb.pad[0] >= 0.0 && lb.pad[1] >= 0.0,
        "invalid letterbox pad {:?}",
        lb.pad
    );
    let [left, top] = lb.pad.map(|v| v as usize);
    assert!(
        left.checked_add(rw).is_some_and(|e| e <= ow) && top.checked_add(rh).is_some_and(|e| e <= oh),
        "letterbox window exceeds output"
    );

    let pad = pad_value as f32 * scale;
    let (r_plane, rest) = dst.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);

    // Fill top and bottom padding bands.
    let fill_rows = |y0: usize, y1: usize, r: &mut [f32], g: &mut [f32], b: &mut [f32]| {
        let range = y0 * ow..y1 * ow;
        r[range.clone()].fill(pad);
        g[range.clone()].fill(pad);
        b[range].fill(pad);
    };
    fill_rows(0, top, r_plane, g_plane, b_plane);
    fill_rows(top + rh, oh, r_plane, g_plane, b_plane);

    // Identity resize: convert RGBA rows straight to planar f32.
    if width == lb.resized[0] && height == lb.resized[1] {
        let sstride = w * 4;
        for y in 0..rh {
            let oy = (top + y) * ow;
            let r_row = &mut r_plane[oy..oy + ow];
            let g_row = &mut g_plane[oy..oy + ow];
            let b_row = &mut b_plane[oy..oy + ow];
            r_row[..left].fill(pad);
            r_row[left + rw..].fill(pad);
            g_row[..left].fill(pad);
            g_row[left + rw..].fill(pad);
            b_row[..left].fill(pad);
            b_row[left + rw..].fill(pad);
            rgba_row_to_planar_f32_scale(
                &rgba[y * sstride..(y + 1) * sstride],
                scale,
                &mut r_row[left..left + rw],
                &mut g_row[left..left + rw],
                &mut b_row[left..left + rw],
            );
        }
        return;
    }

    let plan = resize_plan(width, height, lb.resized[0], lb.resized[1]);
    let [fx, fy] = plan.prefilter;
    let (src_bytes, sw): (&[u8], usize) = if fx >= 2 || fy >= 2 {
        let pw = box_downsample_rgba_into(
            rgba,
            w,
            h,
            fx as usize,
            fy as usize,
            &mut scratch.box_col_sum,
            &mut scratch.box_out,
        );
        (scratch.box_out.as_slice(), pw)
    } else {
        (rgba, w)
    };
    assert!(plan.x0.iter().all(|&k| (k as usize) < sw), "resize tap out of source bounds");

    let sstride = sw * 4;
    scratch.v_row.resize((sw + 1) * 4, 0);
    let v_row = scratch.v_row.as_mut_slice();
    let mut cached: Option<(u32, u32, u32)> = None;

    for y in 0..rh {
        let key = (plan.y0[y], plan.y1[y], plan.wy[y]);
        if cached != Some(key) {
            let (y0, y1, wy) = key;
            let top_src = &src_bytes[y0 as usize * sstride..][..sstride];
            if y0 == y1 || wy == 0 {
                vshift_rgba_to_u32(top_src, &mut v_row[..sw * 4]);
            } else {
                let bot_src = &src_bytes[y1 as usize * sstride..][..sstride];
                vblend_rgba_to_u32(top_src, bot_src, wy, &mut v_row[..sw * 4]);
            }
            // Duplicate trailing pixel so `x0 + 1` matches clamped `x1 == sw - 1`.
            let last = (sw - 1) * 4;
            v_row.copy_within(last..last + 4, sw * 4);
            cached = Some(key);
        }
        let oy = (top + y) * ow;
        let r_row = &mut r_plane[oy..oy + ow];
        let g_row = &mut g_plane[oy..oy + ow];
        let b_row = &mut b_plane[oy..oy + ow];
        r_row[..left].fill(pad);
        r_row[left + rw..].fill(pad);
        g_row[..left].fill(pad);
        g_row[left + rw..].fill(pad);
        b_row[..left].fill(pad);
        b_row[left + rw..].fill(pad);
        hinterp_rgba32_to_planar_f32_scale(
            v_row,
            &plan.x0,
            &plan.wx,
            scale,
            &mut r_row[left..left + rw],
            &mut g_row[left..left + rw],
            &mut b_row[left..left + rw],
        );
    }
}

/// Integer box prefilter directly on RGBA (`4 * w * h` bytes), writing `[R, G, B, 0]` (`4 * pw * ph` bytes)
/// into reusable `out` (and reusing `col_sum`). Matches `folo_data`'s `box_downsample` rounded mean
/// `(sum + n / 2) / n` bit for bit.
fn box_downsample_rgba_into(
    rgba: &[u8],
    w: usize,
    h: usize,
    fx: usize,
    fy: usize,
    col_sum: &mut Vec<u32>,
    out: &mut Vec<u8>,
) -> usize {
    let pw = w.div_ceil(fx);
    let ph = h.div_ceil(fy);
    let sstride = w * 4;
    let dstride = pw * 4;
    out.resize(ph * dstride, 0);
    let use_u64 = (fx as u64)
        .checked_mul(fy as u64)
        .and_then(|n| n.checked_mul(255)?.checked_add(n / 2))
        .is_none_or(|v| v > u32::MAX as u64);
    if !use_u64 {
        col_sum.resize(sstride, 0);
        for oy in 0..ph {
            let y0 = oy * fy;
            let y1 = (y0 + fy).min(h);
            let ny = (y1 - y0) as u32;
            col_sum.fill(0);
            for y in y0..y1 {
                let row = &rgba[y * sstride..(y + 1) * sstride];
                vaccum_u8_into_u32(row, col_sum);
            }
            let dst = &mut out[oy * dstride..(oy + 1) * dstride];
            box_hreduce_rgba32(col_sum, dst, w, pw, fx, ny);
        }
    } else {
        for oy in 0..ph {
            let y0 = oy * fy;
            let y1 = (y0 + fy).min(h);
            let ny = (y1 - y0) as u64;
            let dst = &mut out[oy * dstride..(oy + 1) * dstride];
            for ox in 0..pw {
                let x0 = ox * fx;
                let x1 = (x0 + fx).min(w);
                let n = (x1 - x0) as u64 * ny;
                let half = n / 2;
                let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
                for y in y0..y1 {
                    let row = &rgba[y * sstride..(y + 1) * sstride];
                    for x in x0..x1 {
                        let q = x * 4;
                        sr += row[q] as u64;
                        sg += row[q + 1] as u64;
                        sb += row[q + 2] as u64;
                    }
                }
                let d = ox * 4;
                dst[d] = ((sr + half) / n) as u8;
                dst[d + 1] = ((sg + half) / n) as u8;
                dst[d + 2] = ((sb + half) / n) as u8;
            }
        }
    }
    pw
}

fn vaccum_u8_into_u32(row: &[u8], col_sum: &mut [u32]) {
    debug_assert_eq!(row.len(), col_sum.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `row.len() == col_sum.len()`.
    unsafe {
        wasm_simd::vaccum_u8_into_u32_simd128(row, col_sum);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for (acc, &v) in col_sum.iter_mut().zip(row) {
        *acc += v as u32;
    }
}

fn box_hreduce_rgba32(col_sum: &[u32], dst: &mut [u8], w: usize, pw: usize, fx: usize, ny: u32) {
    debug_assert!(fx >= 1 && ny >= 1 && pw == w.div_ceil(fx));
    debug_assert_eq!(col_sum.len(), w * 4);
    debug_assert_eq!(dst.len(), pw * 4);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `fx >= 1`, `ny >= 1`, `pw == w.div_ceil(fx)`, `col_sum.len() == w * 4`, and `dst.len() == pw * 4`
    // (all guaranteed by `box_downsample_rgba`).
    unsafe {
        wasm_simd::box_hreduce_rgba32_simd128(col_sum, dst, w, pw, fx, ny);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for ox in 0..pw {
        let x0 = ox * fx;
        let x1 = (x0 + fx).min(w);
        let n = (x1 - x0) as u32 * ny;
        let half = n / 2;
        let (mut sr, mut sg, mut sb) = (0u32, 0u32, 0u32);
        for x in x0..x1 {
            let q = x * 4;
            sr += col_sum[q];
            sg += col_sum[q + 1];
            sb += col_sum[q + 2];
        }
        let d = ox * 4;
        dst[d] = ((sr + half) / n) as u8;
        dst[d + 1] = ((sg + half) / n) as u8;
        dst[d + 2] = ((sb + half) / n) as u8;
    }
}

fn rgba_row_to_planar_f32_scale(src: &[u8], scale: f32, r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let n = r.len();
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(src.len(), n * 4);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: lengths checked above.
    unsafe {
        wasm_simd::rgba_row_to_planar_f32_scale_simd128(src, scale, r, g, b);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for (i, px) in src.as_chunks::<4>().0.iter().enumerate() {
        r[i] = px[0] as f32 * scale;
        g[i] = px[1] as f32 * scale;
        b[i] = px[2] as f32 * scale;
    }
}

fn vshift_rgba_to_u32(src: &[u8], dst: &mut [u32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `src.len() == dst.len()`.
    unsafe {
        wasm_simd::vshift_u8_to_u32_simd128(src, dst);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = (s as u32) << BITS;
    }
}

fn vblend_rgba_to_u32(top: &[u8], bot: &[u8], wy: u32, dst: &mut [u32]) {
    debug_assert_eq!(top.len(), dst.len());
    debug_assert_eq!(bot.len(), dst.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `top.len() == bot.len() == dst.len()`.
    unsafe {
        wasm_simd::vblend_u8_to_u32_simd128(top, bot, wy, dst);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let w0 = ONE - wy;
        for ((d, &t), &b) in dst.iter_mut().zip(top).zip(bot) {
            *d = t as u32 * w0 + b as u32 * wy;
        }
    }
}

fn hinterp_rgba32_to_planar_f32_scale(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    scale: f32,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let n = r.len();
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    debug_assert_eq!(x0.len(), n);
    debug_assert_eq!(wx.len(), n);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: caller asserted every `x0[i] < sw` and `v.len() == (sw + 1) * 4`, so `x0[i] * 4 + 8 <= v.len()`.
    unsafe {
        wasm_simd::hinterp_rgba32_to_planar_f32_scale_simd128(v, x0, wx, scale, r, g, b);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for i in 0..n {
        let base = x0[i] as usize * 4;
        let w1 = wx[i];
        let w0 = ONE - w1;
        let rv = (v[base] * w0 + v[base + 4] * w1 + HALF) >> SHIFT;
        let gv = (v[base + 1] * w0 + v[base + 5] * w1 + HALF) >> SHIFT;
        let bv = (v[base + 2] * w0 + v[base + 6] * w1 + HALF) >> SHIFT;
        r[i] = rv as f32 * scale;
        g[i] = gv as f32 * scale;
        b[i] = bv as f32 * scale;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;

#[cfg(test)]
mod tests {
    use super::*;
    use folo_data::{Image, LetterboxOptions, PAD_VALUE};

    fn pattern_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                out.push(((x.wrapping_mul(37) ^ y.wrapping_mul(73)) & 0xff) as u8);
                out.push(((x.wrapping_mul(113) + y.wrapping_mul(19)) & 0xff) as u8);
                out.push(((x.wrapping_mul(7) + y.wrapping_mul(211)) & 0xff) as u8);
                out.push(((x + y * 5) & 0xff) as u8);
            }
        }
        out
    }

    #[test]
    fn rgba_to_rgb_matches_scalar_across_lengths() {
        for n in [0usize, 1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 67] {
            let rgba: Vec<u8> = (0..n * 4).map(|i| (i * 29 + 7) as u8).collect();
            let mut got = vec![0u8; n * 3];
            rgba_to_rgb(&rgba, &mut got);
            let want: Vec<u8> = rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
            assert_eq!(got, want, "n={n}");
        }
    }

    #[test]
    fn letterbox_rgba_matches_folo_data_bit_exact() {
        const INV_255: f32 = 1.0 / 255.0;
        let cases = [
            // Identity
            (64u32, 64u32, 64u32, true),
            // Upscaling
            (37, 29, 128, true),
            // Mild downscaling (fx=1, fy=1)
            (199, 131, 128, true),
            // Heavy downscaling with box prefilter (fx = 2, 3, 4, 5) and partial edge cells
            (321, 217, 96, true),
            (321, 217, 96, false),
            (403, 299, 96, true),
            (515, 387, 96, false),
            // Extreme aspect ratios (single-axis prefilter: [2, 1] and [1, 2])
            (192, 9, 96, true),
            (9, 192, 96, true),
            // Tiny 1x1
            (1, 1, 32, true),
        ];
        // Verify the two single-axis cases truly exercise `[2, 1]` and `[1, 2]` box prefiltering.
        let lb_x =
            Letterbox::compute(192, 9, 96, 96, &LetterboxOptions { stride: Some(32), ..Default::default() });
        assert_eq!(resize_plan(192, 9, lb_x.resized[0], lb_x.resized[1]).prefilter, [2, 1]);
        let lb_y =
            Letterbox::compute(9, 192, 96, 96, &LetterboxOptions { stride: Some(32), ..Default::default() });
        assert_eq!(resize_plan(9, 192, lb_y.resized[0], lb_y.resized[1]).prefilter, [1, 2]);
        for (w, h, imgsz, rect) in cases {
            let rgba = pattern_rgba(w, h);
            let mut rgb = vec![0u8; (w * h * 3) as usize];
            rgba_to_rgb(&rgba, &mut rgb);
            let img = Image { width: w, height: h, data: rgb };
            let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
            let lb = Letterbox::compute(w, h, imgsz, imgsz, &opts);
            let plane = lb.out[0] as usize * lb.out[1] as usize;
            let mut want = vec![0.0f32; 3 * plane];
            let mut got = vec![0.0f32; 3 * plane];
            lb.apply_planar_f32(&img, PAD_VALUE, INV_255, &mut want);
            letterbox_rgba_planar_f32(&rgba, w, h, &lb, PAD_VALUE, INV_255, &mut got);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "mismatch for {w}x{h} -> imgsz={imgsz} rect={rect}"
            );
        }
    }

    #[test]
    fn letterbox_scratch_reuses_allocations_and_matches_exact_bits() {
        const INV_255: f32 = 1.0 / 255.0;
        let (w, h) = (321u32, 217u32);
        let rgba = pattern_rgba(w, h);
        let opts = LetterboxOptions { stride: Some(32), ..LetterboxOptions::default() };
        let lb = Letterbox::compute(w, h, 96, 96, &opts);
        let plane = lb.out[0] as usize * lb.out[1] as usize;
        let mut scratch = RgbaScratch::default();
        let mut first = vec![0.0f32; 3 * plane];
        let mut second = vec![0.0f32; 3 * plane];

        letterbox_rgba_planar_f32_with_scratch(
            &rgba,
            w,
            h,
            &lb,
            PAD_VALUE,
            INV_255,
            &mut first,
            &mut scratch,
        );
        let ptrs_before = (scratch.v_row.as_ptr(), scratch.box_col_sum.as_ptr(), scratch.box_out.as_ptr());
        assert!(!scratch.v_row.is_empty() && !scratch.box_col_sum.is_empty() && !scratch.box_out.is_empty());

        letterbox_rgba_planar_f32_with_scratch(
            &rgba,
            w,
            h,
            &lb,
            PAD_VALUE,
            INV_255,
            &mut second,
            &mut scratch,
        );
        let ptrs_after = (scratch.v_row.as_ptr(), scratch.box_col_sum.as_ptr(), scratch.box_out.as_ptr());
        assert_eq!(
            ptrs_before, ptrs_after,
            "repeated frames must reuse scratch buffers without reallocating"
        );
        assert_eq!(first, second);
    }

    #[test]
    fn crop_and_flip_rgba_match_folo_vision_and_data() {
        let (w, h) = (37u32, 23u32);
        let rgba = pattern_rgba(w, h);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        rgba_to_rgb(&rgba, &mut rgb);
        let img = Image { width: w, height: h, data: rgb };

        // Tile crop parity + scratch reuse
        let rect = TileRect { x: 5, y: 3, width: 17, height: 11 };
        let mut tile_buf = Vec::new();
        crop_rgba_tile(&rgba, w, h, rect, &mut tile_buf);
        let ptr_before = tile_buf.as_ptr();
        crop_rgba_tile(&rgba, w, h, rect, &mut tile_buf);
        assert_eq!(tile_buf.as_ptr(), ptr_before, "tile scratch must not reallocate");
        let mut tile_rgb = vec![0u8; (rect.width * rect.height * 3) as usize];
        rgba_to_rgb(&tile_buf, &mut tile_rgb);
        let want_crop = folo_vision::tile::crop_tile(&img, rect).unwrap();
        assert_eq!(tile_rgb, want_crop.data);

        // Horizontal flip parity across non-multiple-of-4 widths + scratch reuse
        for fw in [1u32, 3, 4, 5, 16, 19, 37] {
            let src = pattern_rgba(fw, 7);
            let mut src_rgb = vec![0u8; (fw * 7 * 3) as usize];
            rgba_to_rgb(&src, &mut src_rgb);
            let want_flip = Image { width: fw, height: 7, data: src_rgb }.oriented(2);
            let mut flip_buf = Vec::new();
            flip_rgba_h(&src, fw, 7, &mut flip_buf);
            let mut got_rgb = vec![0u8; (fw * 7 * 3) as usize];
            rgba_to_rgb(&flip_buf, &mut got_rgb);
            assert_eq!(got_rgb, want_flip.data, "flip mismatch at width={fw}");
        }
    }
}
