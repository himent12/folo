//! Bit-exact AVX2 RGB deinterleaving (`RGB u8` → planar `u8`, planar `f32`, packed `RGB0 u32`) and horizontal row flip.

#[cfg(target_arch = "x86_64")]
use super::has_avx2;

/// Converts interleaved RGB `src` (`width * 3` bytes) into planar `r`, `g`, `b` (`width` bytes each), optionally
/// mirrored left-to-right (`flip_lr`).
#[inline]
pub(crate) fn rgb_to_planar_u8(src: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8], flip_lr: bool) {
    let w = r.len();
    assert!(src.len() == w * 3 && g.len() == w && b.len() == w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 16 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to match `w`.
        unsafe { rgb_to_planar_u8_avx2(src, r, g, b, flip_lr) };
        return;
    }
    rgb_to_planar_u8_scalar(src, r, g, b, flip_lr);
}

#[inline]
pub(crate) fn rgb_to_planar_u8_scalar(src: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8], flip_lr: bool) {
    let w = r.len();
    if flip_lr {
        for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
            let d = w - 1 - x;
            r[d] = px[0];
            g[d] = px[1];
            b[d] = px[2];
        }
    } else {
        for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
            r[x] = px[0];
            g[x] = px[1];
            b[x] = px[2];
        }
    }
}

/// Converts interleaved RGB `src` (`width * 3` bytes) into planar `f32` using `v as f32 * scale` (bit-exact IEEE-754).
#[inline]
pub(crate) fn rgb_to_planar_f32_scale(src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32], scale: f32) {
    let w = r.len();
    assert!(src.len() == w * 3 && g.len() == w && b.len() == w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 16 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to match `w`.
        unsafe { rgb_to_planar_f32_scale_avx2(src, r, g, b, scale) };
        return;
    }
    rgb_to_planar_f32_scale_scalar(src, r, g, b, scale);
}

#[inline]
pub(crate) fn rgb_to_planar_f32_scale_scalar(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    scale: f32,
) {
    for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
        r[x] = px[0] as f32 * scale;
        g[x] = px[1] as f32 * scale;
        b[x] = px[2] as f32 * scale;
    }
}

/// Converts interleaved RGB `src` (`width * 3` bytes) into planar `f32` using `v as f32 / 255.0` (bit-exact IEEE-754),
/// optionally mirrored left-to-right (`flip_lr`).
#[inline]
pub(crate) fn rgb_to_planar_f32_div255(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    flip_lr: bool,
) {
    let w = r.len();
    assert!(src.len() == w * 3 && g.len() == w && b.len() == w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 16 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to match `w`.
        unsafe { rgb_to_planar_f32_div255_avx2(src, r, g, b, flip_lr) };
        return;
    }
    rgb_to_planar_f32_div255_scalar(src, r, g, b, flip_lr);
}

#[inline]
pub(crate) fn rgb_to_planar_f32_div255_scalar(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    flip_lr: bool,
) {
    let w = r.len();
    if flip_lr {
        for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
            let d = w - 1 - x;
            r[d] = px[0] as f32 / 255.0;
            g[d] = px[1] as f32 / 255.0;
            b[d] = px[2] as f32 / 255.0;
        }
    } else {
        for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
            r[x] = px[0] as f32 / 255.0;
            g[x] = px[1] as f32 / 255.0;
            b[x] = px[2] as f32 / 255.0;
        }
    }
}

/// Converts interleaved RGB `src` (`width * 3` bytes) into packed `[R, G, B, 0]` `u32` words.
#[inline]
pub(crate) fn rgb_to_rgb0_u32(src: &[u8], dst: &mut [u32]) {
    let w = dst.len();
    assert_eq!(src.len(), w * 3);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 6 {
        // SAFETY: CPU supports AVX2 and slice lengths are asserted to match `w`.
        unsafe { rgb_to_rgb0_u32_avx2(src, dst) };
        return;
    }
    rgb_to_rgb0_u32_scalar(src, dst);
}

#[inline]
pub(crate) fn rgb_to_rgb0_u32_scalar(src: &[u8], dst: &mut [u32]) {
    for (d, p) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
        *d = u32::from_le_bytes([p[0], p[1], p[2], 0]);
    }
}

/// Mirrors one row of interleaved RGB pixels (`width * 3` bytes) in place.
#[inline]
pub(crate) fn flip_lr_row(row: &mut [u8]) {
    let w = row.len() / 3;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 12 {
        // SAFETY: CPU supports AVX2 and `row.len() >= w * 3`.
        unsafe { flip_lr_row_avx2(row, w) };
        return;
    }
    flip_lr_row_scalar(row, w);
}

#[inline]
pub(crate) fn flip_lr_row_scalar(row: &mut [u8], w: usize) {
    for x in 0..w / 2 {
        let (a, b) = (x * 3, (w - 1 - x) * 3);
        row.swap(a, b);
        row.swap(a + 1, b + 1);
        row.swap(a + 2, b + 2);
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Deinterleaves 16 RGB pixels (48 bytes starting at `src`) into three 16-byte vectors `(R, G, B)`.
///
/// # Safety
/// `src` must point to at least 48 readable bytes, and the CPU must support AVX2 (and SSSE3).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn deinterleave_rgb16_avx2(src: *const u8) -> (__m128i, __m128i, __m128i) {
    let c0 = unsafe { _mm_loadu_si128(src as *const __m128i) };
    let c1 = unsafe { _mm_loadu_si128(src.add(16) as *const __m128i) };
    let c2 = unsafe { _mm_loadu_si128(src.add(32) as *const __m128i) };
    let mr0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mr1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let mr2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let mg0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mg1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let mg2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let mb0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mb1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let mb2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    let r = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, mr0), _mm_shuffle_epi8(c1, mr1)),
        _mm_shuffle_epi8(c2, mr2),
    );
    let g = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, mg0), _mm_shuffle_epi8(c1, mg1)),
        _mm_shuffle_epi8(c2, mg2),
    );
    let b = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(c0, mb0), _mm_shuffle_epi8(c1, mb1)),
        _mm_shuffle_epi8(c2, mb2),
    );
    (r, g, b)
}

/// Interleaves three 16-byte channel vectors `(R, G, B)` into 48 RGB bytes starting at `dst`.
///
/// # Safety
/// `dst` must point to at least 48 writable bytes, and the CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn interleave_rgb16_avx2(r: __m128i, g: __m128i, b: __m128i, dst: *mut u8) {
    let r0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let r1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let r2 = _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1);
    let g2 = _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1);
    let b2 = _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15);
    let o0 =
        _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, r0), _mm_shuffle_epi8(g, g0)), _mm_shuffle_epi8(b, b0));
    let o1 =
        _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, r1), _mm_shuffle_epi8(g, g1)), _mm_shuffle_epi8(b, b1));
    let o2 =
        _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, r2), _mm_shuffle_epi8(g, g2)), _mm_shuffle_epi8(b, b2));
    unsafe {
        _mm_storeu_si128(dst as *mut __m128i, o0);
        _mm_storeu_si128(dst.add(16) as *mut __m128i, o1);
        _mm_storeu_si128(dst.add(32) as *mut __m128i, o2);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_planar_u8_avx2(src: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8], flip_lr: bool) {
    let w = r.len();
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so `sp.add(x * 3)` has 48 readable bytes and `rp/gp/bp` have 16 writable bytes.
        let (mut vr, mut vg, mut vb) = unsafe { deinterleave_rgb16_avx2(sp.add(x * 3)) };
        let dst_x = if flip_lr {
            vr = _mm_shuffle_epi8(vr, rev16);
            vg = _mm_shuffle_epi8(vg, rev16);
            vb = _mm_shuffle_epi8(vb, rev16);
            w - x - 16
        } else {
            x
        };
        unsafe {
            _mm_storeu_si128(rp.add(dst_x) as *mut __m128i, vr);
            _mm_storeu_si128(gp.add(dst_x) as *mut __m128i, vg);
            _mm_storeu_si128(bp.add(dst_x) as *mut __m128i, vb);
        }
        x += 16;
    }
    while x < w {
        let d = if flip_lr { w - 1 - x } else { x };
        r[d] = src[x * 3];
        g[d] = src[x * 3 + 1];
        b[d] = src[x * 3 + 2];
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_planar_f32_scale_avx2(src: &[u8], r: &mut [f32], g: &mut [f32], b: &mut [f32], scale: f32) {
    let w = r.len();
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vscale = _mm256_set1_ps(scale);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s are writable at `x`.
        let (vr, vg, vb) = unsafe { deinterleave_rgb16_avx2(sp.add(x * 3)) };
        for (v, dp) in [(vr, rp), (vg, gp), (vb, bp)] {
            let lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), vscale);
            let hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v))), vscale);
            unsafe {
                _mm256_storeu_ps(dp.add(x), lo);
                _mm256_storeu_ps(dp.add(x + 8), hi);
            }
        }
        x += 16;
    }
    while x < w {
        r[x] = src[x * 3] as f32 * scale;
        g[x] = src[x * 3 + 1] as f32 * scale;
        b[x] = src[x * 3 + 2] as f32 * scale;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_planar_f32_div255_avx2(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    flip_lr: bool,
) {
    let w = r.len();
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let v255 = _mm256_set1_ps(255.0);
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s are writable at `dst_x`.
        let (mut vr, mut vg, mut vb) = unsafe { deinterleave_rgb16_avx2(sp.add(x * 3)) };
        let dst_x = if flip_lr {
            vr = _mm_shuffle_epi8(vr, rev16);
            vg = _mm_shuffle_epi8(vg, rev16);
            vb = _mm_shuffle_epi8(vb, rev16);
            w - x - 16
        } else {
            x
        };
        for (v, dp) in [(vr, rp), (vg, gp), (vb, bp)] {
            let lo = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), v255);
            let hi = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v))), v255);
            unsafe {
                _mm256_storeu_ps(dp.add(dst_x), lo);
                _mm256_storeu_ps(dp.add(dst_x + 8), hi);
            }
        }
        x += 16;
    }
    while x < w {
        let d = if flip_lr { w - 1 - x } else { x };
        r[d] = src[x * 3] as f32 / 255.0;
        g[d] = src[x * 3 + 1] as f32 / 255.0;
        b[d] = src[x * 3 + 2] as f32 / 255.0;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_rgb0_u32_avx2(src: &[u8], dst: &mut [u32]) {
    let w = dst.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mask = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let mut x = 0usize;
    // Need 16 readable bytes at `x * 3`, i.e. `x * 3 + 16 <= w * 3` => `x + 6 <= w`.
    while x + 6 <= w {
        // SAFETY: `x * 3 + 16 <= src.len()` and `x + 4 <= dst.len()`.
        unsafe {
            let v = _mm_loadu_si128(sp.add(x * 3) as *const __m128i);
            let packed = _mm_shuffle_epi8(v, mask);
            _mm_storeu_si128(dp.add(x) as *mut __m128i, packed);
        }
        x += 4;
    }
    while x < w {
        dst[x] = u32::from_le_bytes([src[x * 3], src[x * 3 + 1], src[x * 3 + 2], 0]);
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flip_lr_row_avx2(row: &mut [u8], w: usize) {
    let p = row.as_mut_ptr();
    let rev4 = _mm_setr_epi8(9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1, -1, -1, -1);
    // Outermost 2 pairs scalar so every 16-byte SIMD load at `right` stays within `row`.
    let edge = 2.min(w / 2);
    for x in 0..edge {
        let (a, b) = (x * 3, (w - 1 - x) * 3);
        row.swap(a, b);
        row.swap(a + 1, b + 1);
        row.swap(a + 2, b + 2);
    }
    let mut x = edge;
    while x + 4 <= w / 2 {
        let left = x * 3;
        let right = (w - 4 - x) * 3;
        // SAFETY: `x >= 2`, so `right + 16 <= w * 3`; `x + 4 <= w / 2`, so `[left..left+12)` and `[right..right+12)`
        // are disjoint. Exact 12-byte stores (`u64` + `u32`) avoid overwriting neighbouring pixels.
        unsafe {
            let vl = _mm_shuffle_epi8(_mm_loadu_si128(p.add(left) as *const __m128i), rev4);
            let vr = _mm_shuffle_epi8(_mm_loadu_si128(p.add(right) as *const __m128i), rev4);
            (p.add(left) as *mut i64).write_unaligned(_mm_cvtsi128_si64(vr));
            (p.add(left + 8) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(vr));
            (p.add(right) as *mut i64).write_unaligned(_mm_cvtsi128_si64(vl));
            (p.add(right + 8) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(vl));
        }
        x += 4;
    }
    while x < w / 2 {
        let (a, b) = (x * 3, (w - 1 - x) * 3);
        row.swap(a, b);
        row.swap(a + 1, b + 1);
        row.swap(a + 2, b + 2);
        x += 1;
    }
}
