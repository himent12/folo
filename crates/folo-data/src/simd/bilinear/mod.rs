//! Bit-exact AVX2 fixed-point bilinear vertical row blend and horizontal interpolation kernels, shared by
//! `warp.rs` and `resize.rs` / `letterbox.rs`.
//!
//! All intermediate sums match the scalar 11-bit fixed-point arithmetic bit for bit:
//! - Vertical blend into padded `[R, G, B, 0]` `u32`s (`4 * cols` elements):
//!   `v[c*4 + ch] = top[c*3 + ch] * (2048 - wy) + bot[c*3 + ch] * wy` (max `255 * 2048 = 522_240`).
//! - Horizontal blend of adjacent columns `k` and `k + 1` (loaded together as one 256-bit vector `[v[k], v[k+1]]`):
//!   `((v[k] << 11) + (v[k+1] - v[k]) * wx + (1 << 21)) >> 22`,
//!   algebraically identical to `(v[k] * (2048 - wx) + v[k+1] * wx + (1 << 21)) >> 22` with every intermediate strictly
//!   within `[-1.07e9, +1.07e9] ⊂ [i32::MIN, i32::MAX]`.

mod hinterp;
mod warp;

#[cfg(test)]
pub(crate) use hinterp::hinterp_taps_to_rgb_scalar;
pub(crate) use hinterp::{
    hinterp_taps_to_planar_f32_div255, hinterp_taps_to_planar_f32_scale, hinterp_taps_to_planar_u8,
    hinterp_taps_to_rgb, hinterp_taps_to_rgb0_u32,
};
#[cfg(test)]
pub(crate) use warp::warp_hinterp_interior_to_rgb_scalar;
pub(crate) use warp::{warp_general_interior_to_rgb, warp_hinterp_interior_to_rgb};

#[cfg(target_arch = "x86_64")]
use super::has_avx2;

pub(crate) const WBITS: u32 = 11;
pub(crate) const WONE: u32 = 1 << WBITS;
pub(crate) const ROUND: u32 = 1 << (2 * WBITS - 1);
pub(crate) const SHIFT: u32 = 2 * WBITS;

/// Vertically blends `top` and `bot` (`cols * 3` bytes each) with weight `wy ∈ 0..=2048` into `dst` (`cols * 4`
/// `u32`s, `[R, G, B, 0]` per column).
#[inline]
pub(crate) fn vblend_rgb_to_rgba32(top: &[u8], bot: &[u8], wy: u32, dst: &mut [u32]) {
    let cols = top.len() / 3;
    assert!(top.len() == cols * 3 && bot.len() == cols * 3 && dst.len() >= cols * 4 && wy <= WONE);
    if wy == 0 {
        vshift_rgb_to_rgba32(top, dst);
        return;
    }
    if wy == WONE {
        vshift_rgb_to_rgba32(bot, dst);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && cols >= 6 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to satisfy `cols`.
        unsafe { vblend_rgb_to_rgba32_avx2(top, bot, wy, dst, cols) };
        return;
    }
    vblend_rgb_to_rgba32_scalar(top, bot, wy, dst, cols);
}

/// Scales one RGB row `src` (`cols * 3` bytes) by `WONE` (`<< 11`) into `dst` (`cols * 4` `u32`s, `[R, G, B, 0]`),
/// used when `wy == 0` or `top` and `bot` are the same row.
#[inline]
pub(crate) fn vshift_rgb_to_rgba32(src: &[u8], dst: &mut [u32]) {
    let cols = src.len() / 3;
    assert!(src.len() == cols * 3 && dst.len() >= cols * 4);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && cols >= 6 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to satisfy `cols`.
        unsafe { vshift_rgb_to_rgba32_avx2(src, dst, cols) };
        return;
    }
    vshift_rgb_to_rgba32_scalar(src, dst, cols);
}

#[inline]
pub(crate) fn vblend_rgb_to_rgba32_scalar(top: &[u8], bot: &[u8], wy: u32, dst: &mut [u32], cols: usize) {
    let iwy = WONE - wy;
    for c in 0..cols {
        let s = c * 3;
        let d = c * 4;
        dst[d] = top[s] as u32 * iwy + bot[s] as u32 * wy;
        dst[d + 1] = top[s + 1] as u32 * iwy + bot[s + 1] as u32 * wy;
        dst[d + 2] = top[s + 2] as u32 * iwy + bot[s + 2] as u32 * wy;
        dst[d + 3] = 0;
    }
}

#[inline]
pub(crate) fn vshift_rgb_to_rgba32_scalar(src: &[u8], dst: &mut [u32], cols: usize) {
    for c in 0..cols {
        let s = c * 3;
        let d = c * 4;
        dst[d] = (src[s] as u32) << WBITS;
        dst[d + 1] = (src[s + 1] as u32) << WBITS;
        dst[d + 2] = (src[s + 2] as u32) << WBITS;
        dst[d + 3] = 0;
    }
}

// ------------------------------------------------------------------------------------------------
// Shared AVX2 primitives and vertical blend implementations
// ------------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Interpolates two pixels `(k0, wx0)` and `(k1, wx1)` from `vp` (`[R, G, B, 0]` `u32` per column), returning
/// `[R0, G0, B0, 0 | R1, G1, B1, 0]` as `i32`s in `0..=255`.
///
/// # Safety
/// `vp.add(k0 * 4)` and `vp.add(k1 * 4)` must each point to at least 8 readable `u32`s (columns `k` and `k + 1`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn interp_pair_avx2(
    vp: *const u32,
    k0: usize,
    wx0: i32,
    k1: usize,
    wx1: i32,
    vround: __m256i,
) -> __m256i {
    let (a, b) = unsafe {
        (
            _mm256_loadu_si256(vp.add(k0 * 4) as *const __m256i),
            _mm256_loadu_si256(vp.add(k1 * 4) as *const __m256i),
        )
    };
    let t0 = _mm256_permute2x128_si256::<0x20>(a, b);
    let t1 = _mm256_permute2x128_si256::<0x31>(a, b);
    // Lane 3 of every column in `v` is 0, so `diff[3] == diff[7] == 0` and broadcasting `wx0`/`wx1` across all 4
    // lanes of each 128-bit half gives identical bits with 2 instructions instead of 6 scalar inserts.
    let w = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi32(wx0)), _mm_set1_epi32(wx1));
    let diff = _mm256_sub_epi32(t1, t0);
    let prod = _mm256_mullo_epi32(diff, w);
    let sum = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32::<11>(t0), prod), vround);
    _mm256_srli_epi32::<22>(sum)
}

/// Transposes 8 pixels held as 4 pairs (`v0 = [p0 | p4]`, `v1 = [p1 | p5]`, `v2 = [p2 | p6]`, `v3 = [p3 | p7]`,
/// each `[R, G, B, 0]` `i32`) into three `__m256i` channel vectors `R[0..8], G[0..8], B[0..8]`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn transpose_8_rgb0_i32_avx2(
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
    v3: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let t0 = _mm256_unpacklo_epi32(v0, v1); // [R0 R1 G0 G1 | R4 R5 G4 G5]
    let t1 = _mm256_unpackhi_epi32(v0, v1); // [B0 B1  0  0 | B4 B5  0  0]
    let t2 = _mm256_unpacklo_epi32(v2, v3); // [R2 R3 G2 G3 | R6 R7 G6 G7]
    let t3 = _mm256_unpackhi_epi32(v2, v3); // [B2 B3  0  0 | B6 B7  0  0]
    let r = _mm256_unpacklo_epi64(t0, t2); // [R0 R1 R2 R3 | R4 R5 R6 R7]
    let g = _mm256_unpackhi_epi64(t0, t2); // [G0 G1 G2 G3 | G4 G5 G6 G7]
    let b = _mm256_unpacklo_epi64(t1, t3); // [B0 B1 B2 B3 | B4 B5 B6 B7]
    (r, g, b)
}

/// Packs 4 interpolated `[R, G, B, 0]` `i32` pixels (`p01 = [p0 | p1]`, `p23 = [p2 | p3]`) into one `__m128i`
/// holding 4 `[R, G, B, 0]` `u8` pixels (`u32` words `[R, G, B, 0]`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn pack_4_rgb0_u8_avx2(p01: __m256i, p23: __m256i) -> __m128i {
    let u16s = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(p01, p23));
    _mm_packus_epi16(_mm256_castsi256_si128(u16s), _mm256_extracti128_si256::<1>(u16s))
}

/// Packs 8 `i32`s in `0..=255` into 8 `u8`s in a `u64` (optionally reversed when `flip_lr`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn pack_8_i32_to_u64_avx2(v: __m256i, flip_lr: bool) -> u64 {
    if flip_lr {
        let m = _mm256_setr_epi8(
            12, 8, 4, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1,
        );
        let s = _mm256_shuffle_epi8(v, m);
        let lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(s)) as u32;
        let hi = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(s)) as u32;
        (hi as u64) | ((lo as u64) << 32)
    } else {
        let m = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1,
        );
        let s = _mm256_shuffle_epi8(v, m);
        let lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(s)) as u32;
        let hi = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(s)) as u32;
        (lo as u64) | ((hi as u64) << 32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vblend_rgb_to_rgba32_avx2(top: &[u8], bot: &[u8], wy: u32, dst: &mut [u32], cols: usize) {
    let tp = top.as_ptr();
    let bp = bot.as_ptr();
    let dp = dst.as_mut_ptr();
    let iwy = (WONE - wy) as i16;
    let wy16 = wy as i16;
    let weights = _mm256_set1_epi32((iwy as u16 as u32 | ((wy16 as u16 as u32) << 16)) as i32);
    let rgb_to_rgb0 = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let mut c = 0usize;
    // 2x unrolled (8 pixels/iter): 16-byte load at `(c + 4) * 3` needs `(c + 4) * 3 + 16 <= cols * 3` => `c + 10 <= cols`.
    while c + 10 <= cols {
        // SAFETY: `(c + 4) * 3 + 16 <= top.len()` and `(c + 8) * 4 <= dst.len()`.
        unsafe {
            let t0 = _mm_shuffle_epi8(_mm_loadu_si128(tp.add(c * 3) as *const __m128i), rgb_to_rgb0);
            let b0 = _mm_shuffle_epi8(_mm_loadu_si128(bp.add(c * 3) as *const __m128i), rgb_to_rgb0);
            let t1 = _mm_shuffle_epi8(_mm_loadu_si128(tp.add(c * 3 + 12) as *const __m128i), rgb_to_rgb0);
            let b1 = _mm_shuffle_epi8(_mm_loadu_si128(bp.add(c * 3 + 12) as *const __m128i), rgb_to_rgb0);
            let v01 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_unpacklo_epi8(t0, b0)), weights);
            let v23 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_unpackhi_epi8(t0, b0)), weights);
            let v45 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_unpacklo_epi8(t1, b1)), weights);
            let v67 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_unpackhi_epi8(t1, b1)), weights);
            _mm256_storeu_si256(dp.add(c * 4) as *mut __m256i, v01);
            _mm256_storeu_si256(dp.add(c * 4 + 8) as *mut __m256i, v23);
            _mm256_storeu_si256(dp.add(c * 4 + 16) as *mut __m256i, v45);
            _mm256_storeu_si256(dp.add(c * 4 + 24) as *mut __m256i, v67);
        }
        c += 8;
    }
    // 16-byte load at `c * 3` requires `c * 3 + 16 <= cols * 3`, i.e. `c + 6 <= cols`.
    while c + 6 <= cols {
        // SAFETY: `c * 3 + 16 <= top.len()` and `(c + 4) * 4 <= dst.len()`.
        unsafe {
            let t12 = _mm_loadu_si128(tp.add(c * 3) as *const __m128i);
            let b12 = _mm_loadu_si128(bp.add(c * 3) as *const __m128i);
            let t16 = _mm_shuffle_epi8(t12, rgb_to_rgb0);
            let b16 = _mm_shuffle_epi8(b12, rgb_to_rgb0);
            let lo = _mm_unpacklo_epi8(t16, b16);
            let hi = _mm_unpackhi_epi8(t16, b16);
            let v01 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(lo), weights);
            let v23 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(hi), weights);
            _mm256_storeu_si256(dp.add(c * 4) as *mut __m256i, v01);
            _mm256_storeu_si256(dp.add(c * 4 + 8) as *mut __m256i, v23);
        }
        c += 4;
    }
    let iwy = WONE - wy;
    while c < cols {
        let s = c * 3;
        let d = c * 4;
        dst[d] = top[s] as u32 * iwy + bot[s] as u32 * wy;
        dst[d + 1] = top[s + 1] as u32 * iwy + bot[s + 1] as u32 * wy;
        dst[d + 2] = top[s + 2] as u32 * iwy + bot[s + 2] as u32 * wy;
        dst[d + 3] = 0;
        c += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vshift_rgb_to_rgba32_avx2(src: &[u8], dst: &mut [u32], cols: usize) {
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let rgb_to_rgb0 = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let mut c = 0usize;
    while c + 10 <= cols {
        // SAFETY: `(c + 4) * 3 + 16 <= src.len()` and `(c + 8) * 4 <= dst.len()`.
        unsafe {
            let s0 = _mm_shuffle_epi8(_mm_loadu_si128(sp.add(c * 3) as *const __m128i), rgb_to_rgb0);
            let s1 = _mm_shuffle_epi8(_mm_loadu_si128(sp.add(c * 3 + 12) as *const __m128i), rgb_to_rgb0);
            let v01 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(s0));
            let v23 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(s0)));
            let v45 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(s1));
            let v67 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(s1)));
            _mm256_storeu_si256(dp.add(c * 4) as *mut __m256i, v01);
            _mm256_storeu_si256(dp.add(c * 4 + 8) as *mut __m256i, v23);
            _mm256_storeu_si256(dp.add(c * 4 + 16) as *mut __m256i, v45);
            _mm256_storeu_si256(dp.add(c * 4 + 24) as *mut __m256i, v67);
        }
        c += 8;
    }
    while c + 6 <= cols {
        // SAFETY: `c * 3 + 16 <= src.len()` and `(c + 4) * 4 <= dst.len()`.
        unsafe {
            let s16 = _mm_shuffle_epi8(_mm_loadu_si128(sp.add(c * 3) as *const __m128i), rgb_to_rgb0);
            let v01 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(s16));
            let v23 = _mm256_slli_epi32::<11>(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(s16)));
            _mm256_storeu_si256(dp.add(c * 4) as *mut __m256i, v01);
            _mm256_storeu_si256(dp.add(c * 4 + 8) as *mut __m256i, v23);
        }
        c += 4;
    }
    while c < cols {
        let s = c * 3;
        let d = c * 4;
        dst[d] = (src[s] as u32) << WBITS;
        dst[d + 1] = (src[s + 1] as u32) << WBITS;
        dst[d + 2] = (src[s + 2] as u32) << WBITS;
        dst[d + 3] = 0;
        c += 1;
    }
}
