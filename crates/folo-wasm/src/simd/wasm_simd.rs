//! WASM SIMD128 kernels for [`super`] (`-C target-feature=+simd128`).

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
            let o2 = u8x16_shuffle::<10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30>(v2, v3);
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

/// Converts an RGBA row to planar `f32 * scale`, 2×4-unrolled (8 pixels per iteration) + 4-pixel + tail.
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
    let zero = u8x16_splat(0);
    let vscale = f32x4_splat(scale);
    let sp = src.as_ptr();
    let rp = r.as_mut_ptr();
    let gp = g.as_mut_ptr();
    let bp = b.as_mut_ptr();
    let mut i = 0usize;
    let vec8 = n / 8 * 8;
    while i < vec8 {
        // SAFETY: `(i + 8) * 4 <= n * 4` and `i + 8 <= n`.
        unsafe {
            let v0 = v128_load(sp.add(i * 4) as *const v128);
            let v1 = v128_load(sp.add(i * 4 + 16) as *const v128);
            let r0 = u8x16_shuffle::<0, 16, 16, 16, 4, 16, 16, 16, 8, 16, 16, 16, 12, 16, 16, 16>(v0, zero);
            let g0 = u8x16_shuffle::<1, 16, 16, 16, 5, 16, 16, 16, 9, 16, 16, 16, 13, 16, 16, 16>(v0, zero);
            let b0 = u8x16_shuffle::<2, 16, 16, 16, 6, 16, 16, 16, 10, 16, 16, 16, 14, 16, 16, 16>(v0, zero);
            let r1 = u8x16_shuffle::<0, 16, 16, 16, 4, 16, 16, 16, 8, 16, 16, 16, 12, 16, 16, 16>(v1, zero);
            let g1 = u8x16_shuffle::<1, 16, 16, 16, 5, 16, 16, 16, 9, 16, 16, 16, 13, 16, 16, 16>(v1, zero);
            let b1 = u8x16_shuffle::<2, 16, 16, 16, 6, 16, 16, 16, 10, 16, 16, 16, 14, 16, 16, 16>(v1, zero);
            v128_store(rp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(r0), vscale));
            v128_store(rp.add(i + 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(r1), vscale));
            v128_store(gp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(g0), vscale));
            v128_store(gp.add(i + 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(g1), vscale));
            v128_store(bp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(b0), vscale));
            v128_store(bp.add(i + 4) as *mut v128, f32x4_mul(f32x4_convert_u32x4(b1), vscale));
        }
        i += 8;
    }
    if i + 4 <= n {
        // SAFETY: `(i + 4) * 4 <= n * 4` and `i + 4 <= n`.
        unsafe {
            let v = v128_load(sp.add(i * 4) as *const v128);
            let ru = u8x16_shuffle::<0, 16, 16, 16, 4, 16, 16, 16, 8, 16, 16, 16, 12, 16, 16, 16>(v, zero);
            let gu = u8x16_shuffle::<1, 16, 16, 16, 5, 16, 16, 16, 9, 16, 16, 16, 13, 16, 16, 16>(v, zero);
            let bu = u8x16_shuffle::<2, 16, 16, 16, 6, 16, 16, 16, 10, 16, 16, 16, 14, 16, 16, 16>(v, zero);
            v128_store(rp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(ru), vscale));
            v128_store(gp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(gu), vscale));
            v128_store(bp.add(i) as *mut v128, f32x4_mul(f32x4_convert_u32x4(bu), vscale));
        }
        i += 4;
    }
    for idx in i..n {
        let q = idx * 4;
        r[idx] = src[q] as f32 * scale;
        g[idx] = src[q + 1] as f32 * scale;
        b[idx] = src[q + 2] as f32 * scale;
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

/// Accumulates `col_sum[i] += row[i] as u32`, 16 bytes per iteration.
///
/// # Safety
/// `row.len() == col_sum.len()`.
pub(super) unsafe fn vaccum_u8_into_u32_simd128(row: &[u8], col_sum: &mut [u32]) {
    let n = row.len();
    let chunks = n / 16;
    let sp = row.as_ptr();
    let dp = col_sum.as_mut_ptr();
    for i in 0..chunks {
        // SAFETY: `i * 16 + 16 <= n`.
        unsafe {
            let bytes = v128_load(sp.add(i * 16) as *const v128);
            let lo16 = u16x8_extend_low_u8x16(bytes);
            let hi16 = u16x8_extend_high_u8x16(bytes);
            let outp = dp.add(i * 16) as *mut v128;
            v128_store(outp, u32x4_add(v128_load(outp), u32x4_extend_low_u16x8(lo16)));
            v128_store(outp.add(1), u32x4_add(v128_load(outp.add(1)), u32x4_extend_high_u16x8(lo16)));
            v128_store(outp.add(2), u32x4_add(v128_load(outp.add(2)), u32x4_extend_low_u16x8(hi16)));
            v128_store(outp.add(3), u32x4_add(v128_load(outp.add(3)), u32x4_extend_high_u16x8(hi16)));
        }
    }
    for i in chunks * 16..n {
        col_sum[i] += row[i] as u32;
    }
}

/// Horizontally reduces vertically summed RGBA `u32` columns into `[R, G, B, 0]` `u8` pixels using
/// 4-lane `u32x4_add`, matching `(sum + n / 2) / n` bit for bit.
///
/// # Safety
/// `fx >= 1`, `ny >= 1`, `pw == w.div_ceil(fx)`, `col_sum.len() == w * 4`, and `dst.len() == pw * 4`.
pub(super) unsafe fn box_hreduce_rgba32_simd128(
    col_sum: &[u32],
    dst: &mut [u8],
    w: usize,
    pw: usize,
    fx: usize,
    ny: u32,
) {
    let cp = col_sum.as_ptr() as *const v128;
    let full = w / fx;
    let full_n = fx as u32 * ny;
    let full_half = full_n / 2;
    let write_px = |dst: &mut [u8], ox: usize, sum: v128, n: u32, half: u32| {
        let d = ox * 4;
        dst[d] = ((u32x4_extract_lane::<0>(sum) + half) / n) as u8;
        dst[d + 1] = ((u32x4_extract_lane::<1>(sum) + half) / n) as u8;
        dst[d + 2] = ((u32x4_extract_lane::<2>(sum) + half) / n) as u8;
    };
    match fx {
        2 => {
            for ox in 0..full {
                // SAFETY: `ox * 2 + 1 < w`.
                let sum = unsafe { u32x4_add(v128_load(cp.add(ox * 2)), v128_load(cp.add(ox * 2 + 1))) };
                write_px(dst, ox, sum, full_n, full_half);
            }
        }
        3 => {
            for ox in 0..full {
                // SAFETY: `ox * 3 + 2 < w`.
                let sum = unsafe {
                    let p = cp.add(ox * 3);
                    u32x4_add(u32x4_add(v128_load(p), v128_load(p.add(1))), v128_load(p.add(2)))
                };
                write_px(dst, ox, sum, full_n, full_half);
            }
        }
        4 => {
            for ox in 0..full {
                // SAFETY: `ox * 4 + 3 < w`.
                let sum = unsafe {
                    let p = cp.add(ox * 4);
                    u32x4_add(
                        u32x4_add(v128_load(p), v128_load(p.add(1))),
                        u32x4_add(v128_load(p.add(2)), v128_load(p.add(3))),
                    )
                };
                write_px(dst, ox, sum, full_n, full_half);
            }
        }
        _ => {
            for ox in 0..full {
                let x0 = ox * fx;
                // SAFETY: `x0 + fx <= w`.
                let mut sum = u32x4_splat(0);
                unsafe {
                    for x in x0..x0 + fx {
                        sum = u32x4_add(sum, v128_load(cp.add(x)));
                    }
                }
                write_px(dst, ox, sum, full_n, full_half);
            }
        }
    }
    if full < pw {
        let x0 = full * fx;
        let n = (w - x0) as u32 * ny;
        let half = n / 2;
        let mut sum = u32x4_splat(0);
        // SAFETY: `x0..w` are within `col_sum` (`w * 4` u32s = `w` `v128`s).
        unsafe {
            for x in x0..w {
                sum = u32x4_add(sum, v128_load(cp.add(x)));
            }
        }
        write_px(dst, full, sum, n, half);
    }
}

/// Reverses one RGBA row (`w` 4-byte pixels) using SIMD128 `u32x4_shuffle::<3, 2, 1, 0>` (4 pixels per step).
///
/// # Safety
/// `src.len() == w * 4` and `dst.len() == w * 4`.
pub(super) unsafe fn flip_rgba_row_simd128(src: &[u8], dst: &mut [u8], w: usize) {
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;
    while x + 4 <= w {
        let src_off = (w - x - 4) * 4;
        let dst_off = x * 4;
        // SAFETY: `src_off + 16 <= w * 4` and `dst_off + 16 <= w * 4`.
        unsafe {
            let v = v128_load(sp.add(src_off) as *const v128);
            let rev = u32x4_shuffle::<3, 2, 1, 0>(v, v);
            v128_store(dp.add(dst_off) as *mut v128, rev);
        }
        x += 4;
    }
    while x < w {
        let s = (w - 1 - x) * 4;
        let d = x * 4;
        dst[d..d + 4].copy_from_slice(&src[s..s + 4]);
        x += 1;
    }
}
