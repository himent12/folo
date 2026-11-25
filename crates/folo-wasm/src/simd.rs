//! RGBA preprocessing kernels for the browser: SIMD128-accelerated RGBA→RGB stripping and direct
//! RGBA→planar-`f32` letterboxing (bit-exact with `folo_data::Letterbox::apply_planar_f32`).
//!
//! When compiled for `wasm32` with `-C target-feature=+simd128` (as `scripts/build-wasm.sh` does), the hot
//! loops run on 128-bit WASM SIMD instructions (`core::arch::wasm32`). Elsewhere a portable scalar fallback
//! computes the exact same bytes and floats.

use folo_data::Letterbox;
use folo_data::resize::{RESIZE_WEIGHT_BITS, resize_plan};

const BITS: u32 = RESIZE_WEIGHT_BITS;
const ONE: u32 = 1 << BITS;
const SHIFT: u32 = 2 * BITS;
const HALF: u32 = 1 << (SHIFT - 1);

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
pub(crate) fn letterbox_rgba_planar_f32(
    rgba: &[u8],
    width: u32,
    height: u32,
    lb: &Letterbox,
    pad_value: u8,
    scale: f32,
    dst: &mut [f32],
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
    let filtered = (fx >= 2 || fy >= 2).then(|| box_downsample_rgba(rgba, w, h, fx as usize, fy as usize));
    let (src_bytes, sw): (&[u8], usize) = match &filtered {
        Some((buf, pw)) => (buf.as_slice(), *pw),
        None => (rgba, w),
    };
    assert!(plan.x0.iter().all(|&k| (k as usize) < sw), "resize tap out of source bounds");

    let sstride = sw * 4;
    let mut v_row = vec![0u32; (sw + 1) * 4];
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
            &v_row,
            &plan.x0,
            &plan.wx,
            scale,
            &mut r_row[left..left + rw],
            &mut g_row[left..left + rw],
            &mut b_row[left..left + rw],
        );
    }
}

/// Integer box prefilter directly on RGBA (`4 * w * h` bytes), writing `[R, G, B, 0]` (`4 * pw * ph` bytes).
/// Matches `folo_data`'s `box_downsample` rounded mean `(sum + n / 2) / n` bit for bit.
fn box_downsample_rgba(rgba: &[u8], w: usize, h: usize, fx: usize, fy: usize) -> (Vec<u8>, usize) {
    let pw = w.div_ceil(fx);
    let ph = h.div_ceil(fy);
    let sstride = w * 4;
    let dstride = pw * 4;
    let mut out = vec![0u8; ph * dstride];
    let use_u64 = (fx as u64)
        .checked_mul(fy as u64)
        .and_then(|n| n.checked_mul(255)?.checked_add(n / 2))
        .is_none_or(|v| v > u32::MAX as u64);
    if !use_u64 {
        let mut col_sum = vec![0u32; sstride];
        for oy in 0..ph {
            let y0 = oy * fy;
            let y1 = (y0 + fy).min(h);
            let ny = (y1 - y0) as u32;
            col_sum.fill(0);
            for y in y0..y1 {
                let row = &rgba[y * sstride..(y + 1) * sstride];
                for (acc, &v) in col_sum.iter_mut().zip(row) {
                    *acc += v as u32;
                }
            }
            let dst = &mut out[oy * dstride..(oy + 1) * dstride];
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
    (out, pw)
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
mod wasm_simd {
    use core::arch::wasm32::*;

    use super::{BITS, HALF, ONE, SHIFT};

    /// Strips alpha from 16 RGBA pixels (64 bytes) into 48 RGB bytes per iteration via three `u8x16_shuffle`s.
    ///
    /// # Safety
    /// `rgba.len() >= n * 4` and `out.len() >= n * 3`.
    pub(super) unsafe fn rgba_to_rgb_simd128(rgba: &[u8], out: &mut [u8], n: usize) {
        let chunks = n / 16;
        let src_ptr = rgba.as_ptr();
        let dst_ptr = out.as_mut_ptr();
        for i in 0..chunks {
            // SAFETY: `i * 64 + 64 <= chunks * 64 <= n * 4 <= rgba.len()` and
            // `i * 48 + 48 <= chunks * 48 <= n * 3 <= out.len()`.
            unsafe {
                let sp = src_ptr.add(i * 64) as *const v128;
                let dp = dst_ptr.add(i * 48) as *mut v128;
                let v0 = v128_load(sp);
                let v1 = v128_load(sp.add(1));
                let v2 = v128_load(sp.add(2));
                let v3 = v128_load(sp.add(3));
                let o0 = u8x16_shuffle::<0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20>(v0, v1);
                let o1 = u8x16_shuffle::<5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25>(v1, v2);
                let o2 =
                    u8x16_shuffle::<10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30>(v2, v3);
                v128_store(dp, o0);
                v128_store(dp.add(1), o1);
                v128_store(dp.add(2), o2);
            }
        }
        let done = chunks * 16;
        for (dst, src) in
            out[done * 3..].as_chunks_mut::<3>().0.iter_mut().zip(rgba[done * 4..].as_chunks::<4>().0)
        {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    }

    /// Converts an RGBA row to planar `f32 * scale`, 4 pixels per iteration.
    ///
    /// # Safety
    /// `src.len() == r.len() * 4` and `r.len() == g.len() == b.len()`.
    pub(super) unsafe fn rgba_row_to_planar_f32_scale_simd128(
        src: &[u8],
        scale: f32,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let n = r.len();
        let chunks = n / 4;
        let zero = u8x16_splat(0);
        let vscale = f32x4_splat(scale);
        let sp = src.as_ptr();
        let rp = r.as_mut_ptr();
        let gp = g.as_mut_ptr();
        let bp = b.as_mut_ptr();
        for i in 0..chunks {
            // SAFETY: `i * 16 + 16 <= n * 4` and `(i * 4 + 4) <= n`.
            unsafe {
                let v = v128_load(sp.add(i * 16) as *const v128);
                let ru =
                    u8x16_shuffle::<0, 16, 16, 16, 4, 16, 16, 16, 8, 16, 16, 16, 12, 16, 16, 16>(v, zero);
                let gu =
                    u8x16_shuffle::<1, 16, 16, 16, 5, 16, 16, 16, 9, 16, 16, 16, 13, 16, 16, 16>(v, zero);
                let bu =
                    u8x16_shuffle::<2, 16, 16, 16, 6, 16, 16, 16, 10, 16, 16, 16, 14, 16, 16, 16>(v, zero);
                v128_store(rp.add(i * 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(ru), vscale));
                v128_store(gp.add(i * 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(gu), vscale));
                v128_store(bp.add(i * 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(bu), vscale));
            }
        }
        for i in chunks * 4..n {
            let q = i * 4;
            r[i] = src[q] as f32 * scale;
            g[i] = src[q + 1] as f32 * scale;
            b[i] = src[q + 2] as f32 * scale;
        }
    }

    /// Shifts `u8` row elements into `u32` fixed point (`(s as u32) << BITS`), 16 bytes per iteration.
    ///
    /// # Safety
    /// `src.len() == dst.len()`.
    pub(super) unsafe fn vshift_u8_to_u32_simd128(src: &[u8], dst: &mut [u32]) {
        let n = src.len();
        let chunks = n / 16;
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..chunks {
            // SAFETY: `i * 16 + 16 <= n`.
            unsafe {
                let bytes = v128_load(sp.add(i * 16) as *const v128);
                let lo16 = u16x8_extend_low_u8x16(bytes);
                let hi16 = u16x8_extend_high_u8x16(bytes);
                let d0 = u32x4_shl(u32x4_extend_low_u16x8(lo16), BITS);
                let d1 = u32x4_shl(u32x4_extend_high_u16x8(lo16), BITS);
                let d2 = u32x4_shl(u32x4_extend_low_u16x8(hi16), BITS);
                let d3 = u32x4_shl(u32x4_extend_high_u16x8(hi16), BITS);
                let outp = dp.add(i * 16) as *mut v128;
                v128_store(outp, d0);
                v128_store(outp.add(1), d1);
                v128_store(outp.add(2), d2);
                v128_store(outp.add(3), d3);
            }
        }
        for i in chunks * 16..n {
            dst[i] = (src[i] as u32) << BITS;
        }
    }

    /// Blends two `u8` rows in `u32` fixed point (`t * (ONE - wy) + b * wy`), 16 bytes per iteration.
    ///
    /// # Safety
    /// `top.len() == bot.len() == dst.len()`.
    pub(super) unsafe fn vblend_u8_to_u32_simd128(top: &[u8], bot: &[u8], wy: u32, dst: &mut [u32]) {
        let n = top.len();
        let chunks = n / 16;
        let w0 = ONE - wy;
        let vw0 = u32x4_splat(w0);
        let vwy = u32x4_splat(wy);
        let tp = top.as_ptr();
        let bp = bot.as_ptr();
        let dp = dst.as_mut_ptr();
        for i in 0..chunks {
            // SAFETY: `i * 16 + 16 <= n`.
            unsafe {
                let tb = v128_load(tp.add(i * 16) as *const v128);
                let bb = v128_load(bp.add(i * 16) as *const v128);
                let tlo = u16x8_extend_low_u8x16(tb);
                let thi = u16x8_extend_high_u8x16(tb);
                let blo = u16x8_extend_low_u8x16(bb);
                let bhi = u16x8_extend_high_u8x16(bb);
                let d0 = u32x4_add(
                    u32x4_mul(u32x4_extend_low_u16x8(tlo), vw0),
                    u32x4_mul(u32x4_extend_low_u16x8(blo), vwy),
                );
                let d1 = u32x4_add(
                    u32x4_mul(u32x4_extend_high_u16x8(tlo), vw0),
                    u32x4_mul(u32x4_extend_high_u16x8(blo), vwy),
                );
                let d2 = u32x4_add(
                    u32x4_mul(u32x4_extend_low_u16x8(thi), vw0),
                    u32x4_mul(u32x4_extend_low_u16x8(bhi), vwy),
                );
                let d3 = u32x4_add(
                    u32x4_mul(u32x4_extend_high_u16x8(thi), vw0),
                    u32x4_mul(u32x4_extend_high_u16x8(bhi), vwy),
                );
                let outp = dp.add(i * 16) as *mut v128;
                v128_store(outp, d0);
                v128_store(outp.add(1), d1);
                v128_store(outp.add(2), d2);
                v128_store(outp.add(3), d3);
            }
        }
        for i in chunks * 16..n {
            dst[i] = top[i] as u32 * w0 + bot[i] as u32 * wy;
        }
    }

    /// Horizontally interpolates a vertically blended `[R, G, B, _]` `u32` row into planar `f32 * scale`,
    /// 4 destination pixels per iteration with an in-register 4×3 transpose.
    ///
    /// # Safety
    /// Every `x0[i] as usize * 4 + 8 <= v.len()`, and `r.len() == g.len() == b.len() == x0.len() == wx.len()`.
    pub(super) unsafe fn hinterp_rgba32_to_planar_f32_scale_simd128(
        v: &[u32],
        x0: &[u32],
        wx: &[u32],
        scale: f32,
        r: &mut [f32],
        g: &mut [f32],
        b: &mut [f32],
    ) {
        let n = r.len();
        let chunks = n / 4;
        let vhalf = u32x4_splat(HALF);
        let vscale = f32x4_splat(scale);
        let vp = v.as_ptr();
        let rp = r.as_mut_ptr();
        let gp = g.as_mut_ptr();
        let bp = b.as_mut_ptr();

        #[inline(always)]
        unsafe fn interp_one(vp: *const u32, idx: u32, w1: u32, vhalf: v128) -> v128 {
            // SAFETY: caller guarantees `idx as usize * 4 + 8 <= v.len()`.
            unsafe {
                let p = vp.add(idx as usize * 4) as *const v128;
                let a = v128_load(p);
                let b = v128_load(p.add(1));
                let vw0 = u32x4_splat(ONE - w1);
                let vw1 = u32x4_splat(w1);
                u32x4_shr(u32x4_add(u32x4_add(u32x4_mul(a, vw0), u32x4_mul(b, vw1)), vhalf), SHIFT)
            }
        }

        for i in 0..chunks {
            let k = i * 4;
            // SAFETY: `k + 4 <= n` and `x0[k..k+4]` are validated in bounds.
            unsafe {
                let q0 = interp_one(vp, x0[k], wx[k], vhalf);
                let q1 = interp_one(vp, x0[k + 1], wx[k + 1], vhalf);
                let q2 = interp_one(vp, x0[k + 2], wx[k + 2], vhalf);
                let q3 = interp_one(vp, x0[k + 3], wx[k + 3], vhalf);
                let t0 = u32x4_shuffle::<0, 4, 1, 5>(q0, q1);
                let t1 = u32x4_shuffle::<0, 4, 1, 5>(q2, q3);
                let t2 = u32x4_shuffle::<2, 6, 3, 7>(q0, q1);
                let t3 = u32x4_shuffle::<2, 6, 3, 7>(q2, q3);
                let r_u32 = u32x4_shuffle::<0, 1, 4, 5>(t0, t1);
                let g_u32 = u32x4_shuffle::<2, 3, 6, 7>(t0, t1);
                let b_u32 = u32x4_shuffle::<0, 1, 4, 5>(t2, t3);
                v128_store(rp.add(k) as *mut v128, f32x4_mul(f32x4_convert_u32x4(r_u32), vscale));
                v128_store(gp.add(k) as *mut v128, f32x4_mul(f32x4_convert_u32x4(g_u32), vscale));
                v128_store(bp.add(k) as *mut v128, f32x4_mul(f32x4_convert_u32x4(b_u32), vscale));
            }
        }
        for i in chunks * 4..n {
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
}

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
            // Heavy downscaling with box prefilter (fx>=2, fy>=2) and partial edge cells
            (321, 217, 96, true),
            (321, 217, 96, false),
            // Extreme aspect ratio (prefilter on one axis only)
            (257, 31, 96, true),
            // Tiny 1x1
            (1, 1, 32, true),
        ];
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
}
