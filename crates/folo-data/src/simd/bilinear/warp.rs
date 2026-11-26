//! Fixed-point affine warp interior interpolation kernels for `warp.rs`.

use super::{ROUND, SHIFT, WONE};
#[cfg(target_arch = "x86_64")]
use super::{interp_pair_avx2, pack_4_rgb0_u8_avx2};
#[cfg(target_arch = "x86_64")]
use crate::simd::has_avx2;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal fixed-point warp interpolation along an interior span (every `x in 0..count` is guaranteed to satisfy
/// `0 <= ((rel_fu0 + x * du) >> 16) < last`), writing interleaved RGB `u8` into `out_rgb[..count * 3]`.
#[inline]
pub(crate) fn warp_hinterp_interior_to_rgb(v: &[u32], rel_fu0: i64, du: i64, out_rgb: &mut [u8]) {
    let count = out_rgb.len() / 3;
    assert_eq!(out_rgb.len(), count * 3);
    #[cfg(target_arch = "x86_64")]
    {
        let cols = v.len() / 4;
        let hi_excl = (cols as i128 - 1) << 16;
        let last_rel = rel_fu0 as i128 + (count as i128 - 1) * du as i128;
        if has_avx2()
            && count >= 6
            && (2..=32000).contains(&cols)
            && du.abs() <= 4 << 16
            && (0..hi_excl).contains(&(rel_fu0 as i128))
            && (0..hi_excl).contains(&last_rel)
        {
            // SAFETY: CPU supports AVX2; both endpoints of linear `rel_fu` lie in `0..((cols-1)<<16)` with
            // `cols <= 32000` and `|du| <= 4<<16`, so every tap `k < cols - 1` fits in `v` and `rel_fu` fits `i32`.
            unsafe { warp_hinterp_interior_to_rgb_avx2(v, rel_fu0 as i32, du as i32, out_rgb) };
            return;
        }
    }
    warp_hinterp_interior_to_rgb_scalar(v, rel_fu0, du, out_rgb);
}

#[inline]
pub(crate) fn warp_hinterp_interior_to_rgb_scalar(v: &[u32], rel_fu0: i64, du: i64, out_rgb: &mut [u8]) {
    let mut rfu = rel_fu0;
    for px in out_rgb.as_chunks_mut::<3>().0.iter_mut() {
        let k = (rfu >> 16) as usize * 4;
        let wx = ((rfu >> 5) & (WONE as i64 - 1)) as u32;
        let iwx = WONE - wx;
        rfu += du;
        px[0] = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        px[1] = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        px[2] = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
    }
}

/// Interior span of `warp_general` for a `sw x sh` image (`sstride == sw * 3`), writing interleaved RGB into
/// `out_rgb`. Verifies `sdata` length and both endpoints of `(fu, fv)` in O(1) before entering AVX2.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn warp_general_interior_to_rgb(
    sdata: &[u8],
    sw: usize,
    sh: usize,
    sstride: usize,
    fu0: i64,
    fv0: i64,
    du: i64,
    dv: i64,
    out_rgb: &mut [u8],
) {
    let count = out_rgb.len() / 3;
    assert_eq!(out_rgb.len(), count * 3);
    assert!(sstride == sw.checked_mul(3).expect("stride overflow"));
    assert!(sh.checked_mul(sstride).is_some_and(|need| sdata.len() >= need));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && count >= 6 && sw >= 2 && sh >= 3 {
        let fu_last = fu0 as i128 + (count as i128 - 1) * du as i128;
        let fv_last = fv0 as i128 + (count as i128 - 1) * dv as i128;
        let u_hi = (sw as i128 - 1) << 16;
        let v_hi = (sh as i128 - 2) << 16;
        if (0..u_hi).contains(&(fu0 as i128))
            && (0..u_hi).contains(&fu_last)
            && (0..v_hi).contains(&(fv0 as i128))
            && (0..v_hi).contains(&fv_last)
        {
            // SAFETY: CPU supports AVX2; `sdata.len() >= sh * sstride` is asserted above, and by linearity both
            // endpoints prove every tap has `0 <= ix <= sw - 2` and `0 <= iy <= sh - 3`, so
            // `(iy + 1) * sstride + ix * 3 + 8 <= sh * sstride <= sdata.len()`.
            unsafe { warp_general_interior_to_rgb_avx2(sdata, sstride, fu0, fv0, du, dv, out_rgb) };
            return;
        }
    }
    let (mut fu, mut fv) = (fu0, fv0);
    for px in out_rgb.as_chunks_mut::<3>().0.iter_mut() {
        let ix = (fu >> 16) as usize;
        let iy = (fv >> 16) as usize;
        let wx = ((fu >> 5) & (WONE as i64 - 1)) as u32;
        let wy = ((fv >> 5) & (WONE as i64 - 1)) as u32;
        fu += du;
        fv += dv;
        let o = iy * sstride + ix * 3;
        let (r0, r1) = (&sdata[o..o + 6], &sdata[o + sstride..o + sstride + 6]);
        let (iwx, iwy) = (WONE - wx, WONE - wy);
        for ch in 0..3 {
            let t = r0[ch] as u32 * iwx + r0[ch + 3] as u32 * wx;
            let u = r1[ch] as u32 * iwx + r1[ch + 3] as u32 * wx;
            px[ch] = ((t * iwy + u * wy + ROUND) >> SHIFT) as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn warp_hinterp_interior_to_rgb_avx2(v: &[u32], rel_fu0: i32, du: i32, out_rgb: &mut [u8]) {
    let count = out_rgb.len() / 3;
    let vp = v.as_ptr();
    let op = out_rgb.as_mut_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rgb0_to_rgb = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let mut rfu = rel_fu0;
    let mut x = 0usize;
    while x + 6 <= count {
        let r0 = rfu;
        let r1 = r0 + du;
        let r2 = r1 + du;
        let r3 = r2 + du;
        rfu = r3 + du;
        // SAFETY: caller guarantees every `k < v.len() / 4 - 1`, and `x * 3 + 16 <= out_rgb.len()`.
        unsafe {
            let p01 = interp_pair_avx2(
                vp,
                (r0 >> 16) as usize,
                (r0 >> 5) & 2047,
                (r1 >> 16) as usize,
                (r1 >> 5) & 2047,
                vround,
            );
            let p23 = interp_pair_avx2(
                vp,
                (r2 >> 16) as usize,
                (r2 >> 5) & 2047,
                (r3 >> 16) as usize,
                (r3 >> 5) & 2047,
                vround,
            );
            let rgba = pack_4_rgb0_u8_avx2(p01, p23);
            let rgb = _mm_shuffle_epi8(rgba, rgb0_to_rgb);
            _mm_storeu_si128(op.add(x * 3) as *mut __m128i, rgb);
        }
        x += 4;
    }
    while x < count {
        let k = (rfu >> 16) as usize * 4;
        let wx = ((rfu >> 5) & 2047) as u32;
        let iwx = WONE - wx;
        rfu += du;
        out_rgb[x * 3] = ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8;
        out_rgb[x * 3 + 1] = ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8;
        out_rgb[x * 3 + 2] = ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn warp_general_interior_to_rgb_avx2(
    sdata: &[u8],
    sstride: usize,
    fu0: i64,
    fv0: i64,
    du: i64,
    dv: i64,
    out_rgb: &mut [u8],
) {
    let count = out_rgb.len() / 3;
    let sp = sdata.as_ptr();
    let op = out_rgb.as_mut_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    // Pairs `[R0, R1, G0, G1, B0, B1, 0, 0]` for top row (low 8 bytes) and bottom row (high 8 bytes) from a 128-bit
    // register holding `[top_6_bytes .. | bot_6_bytes ..]`.
    let pair_shuf = _mm_setr_epi8(0, 3, 1, 4, 2, 5, -1, -1, 8, 11, 9, 12, 10, 13, -1, -1);
    let rgb0_to_rgb = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let (mut fu, mut fv) = (fu0, fv0);
    let mut x = 0usize;

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    #[inline]
    unsafe fn sample_pair(
        sp: *const u8,
        sstride: usize,
        fu0: i64,
        fv0: i64,
        fu1: i64,
        fv1: i64,
        pair_shuf: __m128i,
        vround: __m256i,
    ) -> __m256i {
        let o0 = (fv0 >> 16) as usize * sstride + (fu0 >> 16) as usize * 3;
        let o1 = (fv1 >> 16) as usize * sstride + (fu1 >> 16) as usize * 3;
        let wx0 = ((fu0 >> 5) & 2047) as u32;
        let wx1 = ((fu1 >> 5) & 2047) as u32;
        let wy0 = ((fv0 >> 5) & 2047) as i32;
        let wy1 = ((fv1 >> 5) & 2047) as i32;
        let w0 = ((WONE - wx0) | (wx0 << 16)) as i32;
        let w1 = ((WONE - wx1) | (wx1 << 16)) as i32;
        // SAFETY: caller guarantees `o + sstride + 8 <= sdata.len()` for both taps.
        let (top01, bot01) = unsafe {
            let t0 = _mm_loadl_epi64(sp.add(o0) as *const __m128i);
            let t1 = _mm_loadl_epi64(sp.add(o1) as *const __m128i);
            let b0 = _mm_loadl_epi64(sp.add(o0 + sstride) as *const __m128i);
            let b1 = _mm_loadl_epi64(sp.add(o1 + sstride) as *const __m128i);
            (_mm_unpacklo_epi64(t0, t1), _mm_unpacklo_epi64(b0, b1))
        };
        let w01 =
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi32(w0)), _mm_set1_epi32(w1));
        let t01 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_shuffle_epi8(top01, pair_shuf)), w01);
        let u01 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_shuffle_epi8(bot01, pair_shuf)), w01);
        let wy01 =
            _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(_mm_set1_epi32(wy0)), _mm_set1_epi32(wy1));
        _mm256_srli_epi32::<22>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_slli_epi32::<11>(t01),
                _mm256_mullo_epi32(_mm256_sub_epi32(u01, t01), wy01),
            ),
            vround,
        ))
    }

    while x + 6 <= count {
        let fu1 = fu + du;
        let fv1 = fv + dv;
        let fu2 = fu1 + du;
        let fv2 = fv1 + dv;
        let fu3 = fu2 + du;
        let fv3 = fv2 + dv;
        // SAFETY: caller guarantees every tap in `0..count` is interior + 8-byte readable; `x * 3 + 16 <= out_rgb.len()`.
        unsafe {
            let p01 = sample_pair(sp, sstride, fu, fv, fu1, fv1, pair_shuf, vround);
            let p23 = sample_pair(sp, sstride, fu2, fv2, fu3, fv3, pair_shuf, vround);
            let rgba = pack_4_rgb0_u8_avx2(p01, p23);
            _mm_storeu_si128(op.add(x * 3) as *mut __m128i, _mm_shuffle_epi8(rgba, rgb0_to_rgb));
        }
        fu = fu3 + du;
        fv = fv3 + dv;
        x += 4;
    }
    while x < count {
        let ix = (fu >> 16) as usize;
        let iy = (fv >> 16) as usize;
        let wx = ((fu >> 5) & 2047) as u32;
        let wy = ((fv >> 5) & 2047) as u32;
        fu += du;
        fv += dv;
        let o = iy * sstride + ix * 3;
        let (r0, r1) = (&sdata[o..o + 6], &sdata[o + sstride..o + sstride + 6]);
        let (iwx, iwy) = (WONE - wx, WONE - wy);
        for ch in 0..3 {
            let t = r0[ch] as u32 * iwx + r0[ch + 3] as u32 * wx;
            let u = r1[ch] as u32 * iwx + r1[ch + 3] as u32 * wx;
            out_rgb[x * 3 + ch] = ((t * iwy + u * wy + ROUND) >> SHIFT) as u8;
        }
        x += 1;
    }
}
