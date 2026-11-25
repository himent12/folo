//! Tap-based horizontal bilinear interpolation kernels for `resize.rs` and `letterbox.rs`.

use super::{ROUND, SHIFT, WONE};
#[cfg(target_arch = "x86_64")]
use super::{interp_pair_avx2, pack_4_rgb0_u8_avx2, pack_8_i32_to_u64_avx2, transpose_8_rgb0_i32_avx2};
#[cfg(target_arch = "x86_64")]
use crate::simd::{INV255_HI, INV255_LO, has_avx2, has_avx2_fma};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal interpolation from `v` (`(max_x0 + 2) * 4` `u32`s, where column `src_w` duplicates `src_w - 1` when
/// clamped) using precomputed taps `x0`, `wx` (`len == out_rgb.len() / 3`), writing interleaved RGB `u8`.
///
/// # Safety
/// Every `k` in `x0` must satisfy `(k as usize + 2) * 4 <= v.len()`.
#[inline]
pub(crate) unsafe fn hinterp_taps_to_rgb(v: &[u32], x0: &[u32], wx: &[u32], out_rgb: &mut [u8]) {
    let w = x0.len();
    assert!(wx.len() == w && out_rgb.len() == w * 3);
    debug_assert!(x0.iter().all(|&k| (k as usize + 2) * 4 <= v.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 6 {
        // SAFETY: CPU supports AVX2; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_rgb_avx2(v, x0, wx, out_rgb) };
        return;
    }
    hinterp_taps_to_rgb_scalar(v, x0, wx, out_rgb);
}

#[inline]
pub(crate) fn hinterp_taps_to_rgb_scalar(v: &[u32], x0: &[u32], wx: &[u32], out_rgb: &mut [u8]) {
    for (x, px) in out_rgb.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let k = x0[x] as usize * 4;
        let w = wx[x];
        let iw = WONE - w;
        for ch in 0..3 {
            px[ch] = ((v[k + ch] * iw + v[k + 4 + ch] * w + ROUND) >> SHIFT) as u8;
        }
    }
}

/// Horizontal interpolation from `v` using taps `x0`, `wx` directly into planar `u8` (`r`, `g`, `b`).
///
/// # Safety
/// Every `k` in `x0` must satisfy `(k as usize + 2) * 4 <= v.len()`.
#[inline]
pub(crate) unsafe fn hinterp_taps_to_planar_u8(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
) {
    let w = x0.len();
    assert!(wx.len() == w && r.len() == w && g.len() == w && b.len() == w);
    debug_assert!(x0.iter().all(|&k| (k as usize + 2) * 4 <= v.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_planar_u8_avx2(v, x0, wx, r, g, b) };
        return;
    }
    for x in 0..w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        r[x] = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        g[x] = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        b[x] = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
    }
}

/// Horizontal interpolation from `v` using taps `x0`, `wx` directly into planar `f32` (`val as f32 * scale`,
/// matching `Letterbox::apply_planar_f32` bit for bit).
///
/// # Safety
/// Every `k` in `x0` must satisfy `(k as usize + 2) * 4 <= v.len()`.
#[inline]
pub(crate) unsafe fn hinterp_taps_to_planar_f32_scale(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    scale: f32,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    assert!(wx.len() == w && r.len() == w && g.len() == w && b.len() == w);
    debug_assert!(x0.iter().all(|&k| (k as usize + 2) * 4 <= v.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_planar_f32_avx2(v, x0, wx, Some(scale), r, g, b) };
        return;
    }
    for x in 0..w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        r[x] = (((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8) as f32 * scale;
        g[x] = (((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8) as f32 * scale;
        b[x] = (((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8) as f32 * scale;
    }
}

/// Horizontal interpolation from `v` using taps `x0`, `wx` directly into planar `f32` (`val as f32 / 255.0`,
/// matching `DataLoader` bit for bit).
///
/// # Safety
/// Every `k` in `x0` must satisfy `(k as usize + 2) * 4 <= v.len()`.
#[inline]
pub(crate) unsafe fn hinterp_taps_to_planar_f32_div255(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    assert!(wx.len() == w && r.len() == w && g.len() == w && b.len() == w);
    debug_assert!(x0.iter().all(|&k| (k as usize + 2) * 4 <= v.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && w >= 8 {
        // SAFETY: CPU supports AVX2+FMA; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_planar_f32_div255_avx2_fma(v, x0, wx, r, g, b) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_planar_f32_avx2(v, x0, wx, None, r, g, b) };
        return;
    }
    for x in 0..w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        r[x] = (((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
        g[x] = (((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
        b[x] = (((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
    }
}

/// Horizontal interpolation from `v` using taps `x0`, `wx` directly into packed `[R, G, B, 0]` `u32` words.
///
/// # Safety
/// Every `k` in `x0` must satisfy `(k as usize + 2) * 4 <= v.len()`.
#[inline]
pub(crate) unsafe fn hinterp_taps_to_rgb0_u32(v: &[u32], x0: &[u32], wx: &[u32], dst: &mut [u32]) {
    let w = x0.len();
    assert!(wx.len() == w && dst.len() == w);
    debug_assert!(x0.iter().all(|&k| (k as usize + 2) * 4 <= v.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 4 {
        // SAFETY: CPU supports AVX2; caller guarantees `(x0[x] + 2) * 4 <= v.len()` and lengths are asserted.
        unsafe { hinterp_taps_to_rgb0_u32_avx2(v, x0, wx, dst) };
        return;
    }
    for x in 0..w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        let r = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        let g = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        let b = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
        dst[x] = u32::from_le_bytes([r, g, b, 0]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hinterp_taps_to_rgb_avx2(v: &[u32], x0: &[u32], wx: &[u32], out_rgb: &mut [u8]) {
    let w = x0.len();
    let vp = v.as_ptr();
    let op = out_rgb.as_mut_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rgb0_to_rgb = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let mut x = 0usize;
    // 16-byte overshoot store at `x * 3` needs `x * 3 + 16 <= w * 3` => `x + 6 <= w`.
    while x + 6 <= w {
        // SAFETY: `x + 4 <= w`, every `x0[x] + 1 < v.len() / 4`, and `x * 3 + 16 <= out_rgb.len()`.
        unsafe {
            let p01 = interp_pair_avx2(
                vp,
                x0[x] as usize,
                wx[x] as i32,
                x0[x + 1] as usize,
                wx[x + 1] as i32,
                vround,
            );
            let p23 = interp_pair_avx2(
                vp,
                x0[x + 2] as usize,
                wx[x + 2] as i32,
                x0[x + 3] as usize,
                wx[x + 3] as i32,
                vround,
            );
            let rgba = pack_4_rgb0_u8_avx2(p01, p23);
            let rgb = _mm_shuffle_epi8(rgba, rgb0_to_rgb);
            _mm_storeu_si128(op.add(x * 3) as *mut __m128i, rgb);
        }
        x += 4;
    }
    while x < w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        out_rgb[x * 3] = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        out_rgb[x * 3 + 1] = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        out_rgb[x * 3 + 2] = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hinterp_taps_to_planar_u8_avx2(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
) {
    let w = x0.len();
    let vp = v.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vround = _mm256_set1_epi32(ROUND as i32);
    let mut x = 0usize;
    while x + 8 <= w {
        // SAFETY: `x + 8 <= w` and every `x0[x] + 1 < v.len() / 4`.
        unsafe {
            let v0 = interp_pair_avx2(
                vp,
                x0[x] as usize,
                wx[x] as i32,
                x0[x + 4] as usize,
                wx[x + 4] as i32,
                vround,
            );
            let v1 = interp_pair_avx2(
                vp,
                x0[x + 1] as usize,
                wx[x + 1] as i32,
                x0[x + 5] as usize,
                wx[x + 5] as i32,
                vround,
            );
            let v2 = interp_pair_avx2(
                vp,
                x0[x + 2] as usize,
                wx[x + 2] as i32,
                x0[x + 6] as usize,
                wx[x + 6] as i32,
                vround,
            );
            let v3 = interp_pair_avx2(
                vp,
                x0[x + 3] as usize,
                wx[x + 3] as i32,
                x0[x + 7] as usize,
                wx[x + 7] as i32,
                vround,
            );
            let (vr, vg, vb) = transpose_8_rgb0_i32_avx2(v0, v1, v2, v3);
            (rp.add(x) as *mut u64).write_unaligned(pack_8_i32_to_u64_avx2(vr, false));
            (gp.add(x) as *mut u64).write_unaligned(pack_8_i32_to_u64_avx2(vg, false));
            (bp.add(x) as *mut u64).write_unaligned(pack_8_i32_to_u64_avx2(vb, false));
        }
        x += 8;
    }
    while x < w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        r[x] = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        g[x] = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        b[x] = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hinterp_taps_to_planar_f32_avx2(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    scale: Option<f32>,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    let vp = v.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vround = _mm256_set1_epi32(ROUND as i32);
    let vparam = _mm256_set1_ps(scale.unwrap_or(255.0));
    let mut x = 0usize;
    while x + 8 <= w {
        // SAFETY: `x + 8 <= w` and every `x0[x] + 1 < v.len() / 4`.
        unsafe {
            let v0 = interp_pair_avx2(
                vp,
                x0[x] as usize,
                wx[x] as i32,
                x0[x + 4] as usize,
                wx[x + 4] as i32,
                vround,
            );
            let v1 = interp_pair_avx2(
                vp,
                x0[x + 1] as usize,
                wx[x + 1] as i32,
                x0[x + 5] as usize,
                wx[x + 5] as i32,
                vround,
            );
            let v2 = interp_pair_avx2(
                vp,
                x0[x + 2] as usize,
                wx[x + 2] as i32,
                x0[x + 6] as usize,
                wx[x + 6] as i32,
                vround,
            );
            let v3 = interp_pair_avx2(
                vp,
                x0[x + 3] as usize,
                wx[x + 3] as i32,
                x0[x + 7] as usize,
                wx[x + 7] as i32,
                vround,
            );
            let (vr, vg, vb) = transpose_8_rgb0_i32_avx2(v0, v1, v2, v3);
            let (fr, fg, fb) = if scale.is_some() {
                (
                    _mm256_mul_ps(_mm256_cvtepi32_ps(vr), vparam),
                    _mm256_mul_ps(_mm256_cvtepi32_ps(vg), vparam),
                    _mm256_mul_ps(_mm256_cvtepi32_ps(vb), vparam),
                )
            } else {
                (
                    _mm256_div_ps(_mm256_cvtepi32_ps(vr), vparam),
                    _mm256_div_ps(_mm256_cvtepi32_ps(vg), vparam),
                    _mm256_div_ps(_mm256_cvtepi32_ps(vb), vparam),
                )
            };
            _mm256_storeu_ps(rp.add(x), fr);
            _mm256_storeu_ps(gp.add(x), fg);
            _mm256_storeu_ps(bp.add(x), fb);
        }
        x += 8;
    }
    while x < w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        let rv = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8 as f32;
        let gv = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8 as f32;
        let bv = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8 as f32;
        if let Some(s) = scale {
            r[x] = rv * s;
            g[x] = gv * s;
            b[x] = bv * s;
        } else {
            r[x] = rv / 255.0;
            g[x] = gv / 255.0;
            b[x] = bv / 255.0;
        }
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn hinterp_taps_to_planar_f32_div255_avx2_fma(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    let vp = v.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vround = _mm256_set1_epi32(ROUND as i32);
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let mut x = 0usize;
    while x + 8 <= w {
        // SAFETY: `x + 8 <= w` and every `x0[x] + 1 < v.len() / 4`.
        unsafe {
            let v0 = interp_pair_avx2(
                vp,
                x0[x] as usize,
                wx[x] as i32,
                x0[x + 4] as usize,
                wx[x + 4] as i32,
                vround,
            );
            let v1 = interp_pair_avx2(
                vp,
                x0[x + 1] as usize,
                wx[x + 1] as i32,
                x0[x + 5] as usize,
                wx[x + 5] as i32,
                vround,
            );
            let v2 = interp_pair_avx2(
                vp,
                x0[x + 2] as usize,
                wx[x + 2] as i32,
                x0[x + 6] as usize,
                wx[x + 6] as i32,
                vround,
            );
            let v3 = interp_pair_avx2(
                vp,
                x0[x + 3] as usize,
                wx[x + 3] as i32,
                x0[x + 7] as usize,
                wx[x + 7] as i32,
                vround,
            );
            let (vr, vg, vb) = transpose_8_rgb0_i32_avx2(v0, v1, v2, v3);
            let fr = _mm256_cvtepi32_ps(vr);
            let fg = _mm256_cvtepi32_ps(vg);
            let fb = _mm256_cvtepi32_ps(vb);
            _mm256_storeu_ps(rp.add(x), _mm256_fmadd_ps(fr, vhi, _mm256_mul_ps(fr, vlo)));
            _mm256_storeu_ps(gp.add(x), _mm256_fmadd_ps(fg, vhi, _mm256_mul_ps(fg, vlo)));
            _mm256_storeu_ps(bp.add(x), _mm256_fmadd_ps(fb, vhi, _mm256_mul_ps(fb, vlo)));
        }
        x += 8;
    }
    while x < w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        r[x] = (((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
        g[x] = (((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
        b[x] = (((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8) as f32 / 255.0;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hinterp_taps_to_rgb0_u32_avx2(v: &[u32], x0: &[u32], wx: &[u32], dst: &mut [u32]) {
    let w = x0.len();
    let vp = v.as_ptr();
    let dp = dst.as_mut_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    let mut x = 0usize;
    while x + 4 <= w {
        // SAFETY: `x + 4 <= w` and every `x0[x] + 1 < v.len() / 4`.
        unsafe {
            let p01 = interp_pair_avx2(
                vp,
                x0[x] as usize,
                wx[x] as i32,
                x0[x + 1] as usize,
                wx[x + 1] as i32,
                vround,
            );
            let p23 = interp_pair_avx2(
                vp,
                x0[x + 2] as usize,
                wx[x + 2] as i32,
                x0[x + 3] as usize,
                wx[x + 3] as i32,
                vround,
            );
            let rgba = pack_4_rgb0_u8_avx2(p01, p23);
            _mm_storeu_si128(dp.add(x) as *mut __m128i, rgba);
        }
        x += 4;
    }
    while x < w {
        let k = x0[x] as usize * 4;
        let wx = wx[x];
        let iwx = WONE - wx;
        let r = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        let g = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        let b = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
        dst[x] = u32::from_le_bytes([r, g, b, 0]);
        x += 1;
    }
}
