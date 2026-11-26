//! Bit-exact AVX2 RGB deinterleaving (`RGB u8` → planar `u8`, planar `f32`, packed `RGB0 u32`), horizontal row flip,
//! mixup blending, and `RGBA`/`L8` → `RGB` decode conversions.

use super::DIV255_LUT;
#[cfg(target_arch = "x86_64")]
use super::{INV255_HI, INV255_LO, has_avx2, has_avx2_fma};

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
    if has_avx2_fma() && w >= 16 {
        // SAFETY: CPU supports AVX2+FMA and slice lengths are asserted to match `w`.
        unsafe { rgb_to_planar_f32_div255_avx2_fma(src, r, g, b, flip_lr) };
        return;
    }
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
            r[d] = DIV255_LUT[px[0] as usize];
            g[d] = DIV255_LUT[px[1] as usize];
            b[d] = DIV255_LUT[px[2] as usize];
        }
    } else {
        for (x, px) in src.as_chunks::<3>().0.iter().enumerate() {
            r[x] = DIV255_LUT[px[0] as usize];
            g[x] = DIV255_LUT[px[1] as usize];
            b[x] = DIV255_LUT[px[2] as usize];
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

/// Copies one interleaved RGB row `src` (`width * 3` bytes) mirrored left-to-right into `dst` (`width * 3` bytes).
#[inline]
pub(crate) fn copy_flip_lr_row(src: &[u8], dst: &mut [u8]) {
    let w = src.len() / 3;
    assert!(src.len() == w * 3 && dst.len() == src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 6 {
        // SAFETY: CPU supports AVX2 and both slices hold `w * 3` bytes.
        unsafe { copy_flip_lr_row_avx2(src, dst, w) };
        return;
    }
    copy_flip_lr_row_scalar(src, dst, w);
}

#[inline]
pub(crate) fn copy_flip_lr_row_scalar(src: &[u8], dst: &mut [u8], w: usize) {
    for x in 0..w {
        let s = x * 3;
        let d = (w - 1 - x) * 3;
        dst[d] = src[s];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
    }
}

/// Blends `other` into `dst` in place: `dst[i] = ((dst[i] * wa + other[i] * (256 - wa) + 128) >> 8) as u8`,
/// bit-exact for every `wa in 0..=256`.
#[inline]
pub(crate) fn mixup_blend_u8(dst: &mut [u8], other: &[u8], wa: u32) {
    assert!(dst.len() == other.len() && wa <= 256);
    if wa == 256 {
        return;
    }
    if wa == 0 {
        dst.copy_from_slice(other);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && dst.len() >= 32 {
        // SAFETY: CPU supports AVX2, slice lengths are equal, and `1 <= wa <= 255`.
        unsafe { mixup_blend_u8_avx2(dst, other, wa) };
        return;
    }
    mixup_blend_u8_scalar(dst, other, wa);
}

#[inline]
pub(crate) fn mixup_blend_u8_scalar(dst: &mut [u8], other: &[u8], wa: u32) {
    let wb = 256 - wa;
    for (a, &b) in dst.iter_mut().zip(other) {
        *a = ((*a as u32 * wa + b as u32 * wb + 128) >> 8) as u8;
    }
}

/// Blends `a` and `b` into `dst` out of place: `dst[i] = ((a[i] * wa + b[i] * (256 - wa) + 128) >> 8) as u8`,
/// bit-exact with [`mixup_blend_u8`] for every `wa in 0..=256`.
#[inline]
pub(crate) fn mixup_blend_u8_into(a: &[u8], b: &[u8], wa: u32, dst: &mut [u8]) {
    assert!(a.len() == b.len() && dst.len() == a.len() && wa <= 256);
    if wa == 256 {
        dst.copy_from_slice(a);
        return;
    }
    if wa == 0 {
        dst.copy_from_slice(b);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && dst.len() >= 32 {
        // SAFETY: CPU supports AVX2, slice lengths are equal, and `1 <= wa <= 255`.
        unsafe { mixup_blend_u8_into_avx2(a, b, wa, dst) };
        return;
    }
    let wb = 256 - wa;
    for ((d, &va), &vb) in dst.iter_mut().zip(a).zip(b) {
        *d = ((va as u32 * wa + vb as u32 * wb + 128) >> 8) as u8;
    }
}

/// Strips alpha from interleaved `RGBA` bytes (`n * 4`) into `RGB` bytes (`n * 3`).
#[inline]
pub(crate) fn rgba_to_rgb(src: &[u8], dst: &mut [u8]) {
    let n = dst.len() / 3;
    assert!(dst.len() == n * 3 && n.checked_mul(4) == Some(src.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && n >= 6 {
        // SAFETY: CPU supports AVX2 and `src` holds `n * 4` bytes, `dst` holds `n * 3` bytes.
        unsafe { rgba_to_rgb_avx2(src, dst, n) };
        return;
    }
    rgba_to_rgb_scalar(src, dst);
}

#[inline]
pub(crate) fn rgba_to_rgb_scalar(src: &[u8], dst: &mut [u8]) {
    for (d, s) in dst.as_chunks_mut::<3>().0.iter_mut().zip(src.as_chunks::<4>().0) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
}

/// Expands 8-bit greyscale `src` (`n` bytes) into interleaved `RGB` `dst` (`n * 3` bytes, `[l, l, l]`).
#[inline]
pub(crate) fn luma_to_rgb(src: &[u8], dst: &mut [u8]) {
    assert!(src.len().checked_mul(3) == Some(dst.len()));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && src.len() >= 16 {
        // SAFETY: CPU supports AVX2 and `dst.len() == src.len() * 3`.
        unsafe { luma_to_rgb_avx2(src, dst) };
        return;
    }
    luma_to_rgb_scalar(src, dst);
}

#[inline]
pub(crate) fn luma_to_rgb_scalar(src: &[u8], dst: &mut [u8]) {
    for (px, &l) in dst.as_chunks_mut::<3>().0.iter_mut().zip(src) {
        *px = [l; 3];
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

/// Deinterleaves 32 RGB pixels (96 bytes starting at `src`) into three 32-byte vectors `(R, G, B)`.
///
/// # Safety
/// `src` must point to at least 96 readable bytes, and the CPU must support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn deinterleave_rgb32_avx2(src: *const u8) -> (__m256i, __m256i, __m256i) {
    let (a, b, c) = unsafe {
        (
            _mm256_loadu_si256(src as *const __m256i),
            _mm256_loadu_si256(src.add(32) as *const __m256i),
            _mm256_loadu_si256(src.add(64) as *const __m256i),
        )
    };
    // Low 128-bit lane holds pixels 0..16 (bytes 0..48); high 128-bit lane holds pixels 16..32 (bytes 48..96).
    let c0 = _mm256_permute2x128_si256::<0x30>(a, b);
    let c1 = _mm256_permute2x128_si256::<0x21>(a, c);
    let c2 = _mm256_permute2x128_si256::<0x30>(b, c);
    let mr0 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ));
    let mr1 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1,
    ));
    let mr2 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13,
    ));
    let mg0 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ));
    let mg1 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1,
    ));
    let mg2 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14,
    ));
    let mb0 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ));
    let mb1 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1,
    ));
    let mb2 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15,
    ));
    let r = _mm256_or_si256(
        _mm256_or_si256(_mm256_shuffle_epi8(c0, mr0), _mm256_shuffle_epi8(c1, mr1)),
        _mm256_shuffle_epi8(c2, mr2),
    );
    let g = _mm256_or_si256(
        _mm256_or_si256(_mm256_shuffle_epi8(c0, mg0), _mm256_shuffle_epi8(c1, mg1)),
        _mm256_shuffle_epi8(c2, mg2),
    );
    let b = _mm256_or_si256(
        _mm256_or_si256(_mm256_shuffle_epi8(c0, mb0), _mm256_shuffle_epi8(c1, mb1)),
        _mm256_shuffle_epi8(c2, mb2),
    );
    (r, g, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_planar_u8_avx2(src: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8], flip_lr: bool) {
    let w = r.len();
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let rev16_256 = _mm256_broadcastsi128_si256(rev16);
    let mut x = 0usize;
    while x + 32 <= w {
        // SAFETY: `x + 32 <= w`, so `sp.add(x * 3)` has 96 readable bytes and `rp/gp/bp` have 32 writable bytes.
        let (mut vr, mut vg, mut vb) = unsafe { deinterleave_rgb32_avx2(sp.add(x * 3)) };
        let dst_x = if flip_lr {
            vr = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vr, vr), rev16_256);
            vg = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vg, vg), rev16_256);
            vb = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vb, vb), rev16_256);
            w - x - 32
        } else {
            x
        };
        unsafe {
            _mm256_storeu_si256(rp.add(dst_x) as *mut __m256i, vr);
            _mm256_storeu_si256(gp.add(dst_x) as *mut __m256i, vg);
            _mm256_storeu_si256(bp.add(dst_x) as *mut __m256i, vb);
        }
        x += 32;
    }
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
    while x + 32 <= w {
        // SAFETY: `x + 32 <= w`, so 96 bytes are readable from `sp` and 32 `f32`s are writable at `x`.
        let (vr, vg, vb) = unsafe { deinterleave_rgb32_avx2(sp.add(x * 3)) };
        for (v, dp) in [(vr, rp), (vg, gp), (vb, bp)] {
            let v0 = _mm256_castsi256_si128(v);
            let v1 = _mm256_extracti128_si256::<1>(v);
            let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0)), vscale);
            let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v0))), vscale);
            let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v1)), vscale);
            let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v1))), vscale);
            unsafe {
                _mm256_storeu_ps(dp.add(x), f0);
                _mm256_storeu_ps(dp.add(x + 8), f1);
                _mm256_storeu_ps(dp.add(x + 16), f2);
                _mm256_storeu_ps(dp.add(x + 24), f3);
            }
        }
        x += 32;
    }
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
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rgb_to_planar_f32_div255_avx2_fma(
    src: &[u8],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    flip_lr: bool,
) {
    let w = r.len();
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let rev16_256 = _mm256_broadcastsi128_si256(rev16);
    let mut x = 0usize;
    while x + 32 <= w {
        // SAFETY: `x + 32 <= w`, so 96 bytes are readable from `sp` and 32 `f32`s are writable at `dst_x`.
        let (mut vr, mut vg, mut vb) = unsafe { deinterleave_rgb32_avx2(sp.add(x * 3)) };
        let dst_x = if flip_lr {
            vr = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vr, vr), rev16_256);
            vg = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vg, vg), rev16_256);
            vb = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vb, vb), rev16_256);
            w - x - 32
        } else {
            x
        };
        for (v, dp) in [(vr, rp), (vg, gp), (vb, bp)] {
            let v0 = _mm256_castsi256_si128(v);
            let v1 = _mm256_extracti128_si256::<1>(v);
            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v0)));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v1));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v1)));
            unsafe {
                _mm256_storeu_ps(dp.add(dst_x), _mm256_fmadd_ps(f0, vhi, _mm256_mul_ps(f0, vlo)));
                _mm256_storeu_ps(dp.add(dst_x + 8), _mm256_fmadd_ps(f1, vhi, _mm256_mul_ps(f1, vlo)));
                _mm256_storeu_ps(dp.add(dst_x + 16), _mm256_fmadd_ps(f2, vhi, _mm256_mul_ps(f2, vlo)));
                _mm256_storeu_ps(dp.add(dst_x + 24), _mm256_fmadd_ps(f3, vhi, _mm256_mul_ps(f3, vlo)));
            }
        }
        x += 32;
    }
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
            let flo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
            let fhi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v)));
            let lo = _mm256_fmadd_ps(flo, vhi, _mm256_mul_ps(flo, vlo));
            let hi = _mm256_fmadd_ps(fhi, vhi, _mm256_mul_ps(fhi, vlo));
            unsafe {
                _mm256_storeu_ps(dp.add(dst_x), lo);
                _mm256_storeu_ps(dp.add(dst_x + 8), hi);
            }
        }
        x += 16;
    }
    while x < w {
        let d = if flip_lr { w - 1 - x } else { x };
        r[d] = DIV255_LUT[src[x * 3] as usize];
        g[d] = DIV255_LUT[src[x * 3 + 1] as usize];
        b[d] = DIV255_LUT[src[x * 3 + 2] as usize];
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb_to_planar_f32_div255_avx2(
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
    let rev16_256 = _mm256_broadcastsi128_si256(rev16);
    let mut x = 0usize;
    while x + 32 <= w {
        // SAFETY: `x + 32 <= w`, so 96 bytes are readable from `sp` and 32 `f32`s are writable at `dst_x`.
        let (mut vr, mut vg, mut vb) = unsafe { deinterleave_rgb32_avx2(sp.add(x * 3)) };
        let dst_x = if flip_lr {
            vr = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vr, vr), rev16_256);
            vg = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vg, vg), rev16_256);
            vb = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x01>(vb, vb), rev16_256);
            w - x - 32
        } else {
            x
        };
        for (v, dp) in [(vr, rp), (vg, gp), (vb, bp)] {
            let v0 = _mm256_castsi256_si128(v);
            let v1 = _mm256_extracti128_si256::<1>(v);
            unsafe {
                _mm256_storeu_ps(
                    dp.add(dst_x),
                    _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v0)), v255),
                );
                _mm256_storeu_ps(
                    dp.add(dst_x + 8),
                    _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v0))), v255),
                );
                _mm256_storeu_ps(
                    dp.add(dst_x + 16),
                    _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v1)), v255),
                );
                _mm256_storeu_ps(
                    dp.add(dst_x + 24),
                    _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v1))), v255),
                );
            }
        }
        x += 32;
    }
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
        r[d] = DIV255_LUT[src[x * 3] as usize];
        g[d] = DIV255_LUT[src[x * 3 + 1] as usize];
        b[d] = DIV255_LUT[src[x * 3 + 2] as usize];
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

/// Reverses 16 interleaved RGB pixels (48 bytes = 3 × 16-byte vectors) in place using pure SIMD shuffles.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn reverse_rgb16_avx2(v0: __m128i, v1: __m128i, v2: __m128i) -> (__m128i, __m128i, __m128i) {
    // v0: p0..p4, p5[0]; v1: p5[1..2], p6..p9, p10[0..1]; v2: p10[2], p11..p15.
    let m0_v2 = _mm_setr_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, -1);
    let m0_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14);
    let m1_v1 = _mm_setr_epi8(15, -1, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4, -1, 0);
    // `alignr::<15>(v2, v0)` puts `v0[15]` (= p5[0]) at lane 0 and `v2[0]` (= p10[2]) at lane 1.
    let m1_v02 = _mm_setr_epi8(-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1);
    let m2_v1 = _mm_setr_epi8(1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let m2_v0 = _mm_setr_epi8(-1, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2);

    let o0 = _mm_or_si128(_mm_shuffle_epi8(v2, m0_v2), _mm_shuffle_epi8(v1, m0_v1));
    let v02 = _mm_alignr_epi8::<15>(v2, v0);
    let o1 = _mm_or_si128(_mm_shuffle_epi8(v1, m1_v1), _mm_shuffle_epi8(v02, m1_v02));
    let o2 = _mm_or_si128(_mm_shuffle_epi8(v1, m2_v1), _mm_shuffle_epi8(v0, m2_v0));
    (o0, o1, o2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn flip_lr_row_avx2(row: &mut [u8], w: usize) {
    let p = row.as_mut_ptr();
    let half = w / 2;
    let mut x = 0usize;
    // Fast 16-pixel (48-byte) symmetric pure-SIMD swaps with exact 16-byte stores (zero overshoot).
    while x + 16 <= half {
        let left = x * 3;
        let right = (w - 16 - x) * 3;
        // SAFETY: `x + 16 <= half`, so `[left..left+48)` and `[right..right+48)` are disjoint and within `w * 3`.
        unsafe {
            let l0 = _mm_loadu_si128(p.add(left) as *const __m128i);
            let l1 = _mm_loadu_si128(p.add(left + 16) as *const __m128i);
            let l2 = _mm_loadu_si128(p.add(left + 32) as *const __m128i);
            let r0 = _mm_loadu_si128(p.add(right) as *const __m128i);
            let r1 = _mm_loadu_si128(p.add(right + 16) as *const __m128i);
            let r2 = _mm_loadu_si128(p.add(right + 32) as *const __m128i);
            let (rl0, rl1, rl2) = reverse_rgb16_avx2(l0, l1, l2);
            let (rr0, rr1, rr2) = reverse_rgb16_avx2(r0, r1, r2);
            _mm_storeu_si128(p.add(left) as *mut __m128i, rr0);
            _mm_storeu_si128(p.add(left + 16) as *mut __m128i, rr1);
            _mm_storeu_si128(p.add(left + 32) as *mut __m128i, rr2);
            _mm_storeu_si128(p.add(right) as *mut __m128i, rl0);
            _mm_storeu_si128(p.add(right + 16) as *mut __m128i, rl1);
            _mm_storeu_si128(p.add(right + 32) as *mut __m128i, rl2);
        }
        x += 16;
    }
    let rev4 = _mm_setr_epi8(9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1, -1, -1, -1);
    // For the 4-pixel tail, ensure at least 2 pixels remain past `right + 12` so 16-byte load stays in bounds.
    while x < 2.min(half) {
        let (a, b) = (x * 3, (w - 1 - x) * 3);
        row.swap(a, b);
        row.swap(a + 1, b + 1);
        row.swap(a + 2, b + 2);
        x += 1;
    }
    // Re-derive `p` after any `row.swap` reborrow above so provenance stays valid under Stacked/Tree Borrows.
    let p = row.as_mut_ptr();
    while x + 4 <= half {
        let left = x * 3;
        let right = (w - 4 - x) * 3;
        // SAFETY: `x >= 2`, so `right + 16 <= w * 3`; `x + 4 <= half`, so `[left..left+12)` and `[right..right+12)`
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
    while x < half {
        let (a, b) = (x * 3, (w - 1 - x) * 3);
        row.swap(a, b);
        row.swap(a + 1, b + 1);
        row.swap(a + 2, b + 2);
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_flip_lr_row_avx2(src: &[u8], dst: &mut [u8], w: usize) {
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut x = 0usize;
    while x + 16 <= w {
        let d = (w - x - 16) * 3;
        // SAFETY: `x + 16 <= w`, so 48 bytes are readable at `sp + x * 3` and writable at `dp + d`.
        unsafe {
            let v0 = _mm_loadu_si128(sp.add(x * 3) as *const __m128i);
            let v1 = _mm_loadu_si128(sp.add(x * 3 + 16) as *const __m128i);
            let v2 = _mm_loadu_si128(sp.add(x * 3 + 32) as *const __m128i);
            let (o0, o1, o2) = reverse_rgb16_avx2(v0, v1, v2);
            _mm_storeu_si128(dp.add(d) as *mut __m128i, o0);
            _mm_storeu_si128(dp.add(d + 16) as *mut __m128i, o1);
            _mm_storeu_si128(dp.add(d + 32) as *mut __m128i, o2);
        }
        x += 16;
    }
    let rev4 = _mm_setr_epi8(9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1, -1, -1, -1);
    // 16-byte load at `x * 3` needs `x * 3 + 16 <= w * 3` => `x + 6 <= w`.
    while x + 6 <= w {
        let d = (w - x - 4) * 3;
        // SAFETY: `x * 3 + 16 <= src.len()` and `d + 12 <= dst.len()`. Exact 12-byte store avoids overshoot.
        unsafe {
            let v = _mm_shuffle_epi8(_mm_loadu_si128(sp.add(x * 3) as *const __m128i), rev4);
            (dp.add(d) as *mut i64).write_unaligned(_mm_cvtsi128_si64(v));
            (dp.add(d + 8) as *mut i32).write_unaligned(_mm_extract_epi32::<2>(v));
        }
        x += 4;
    }
    while x < w {
        let s = x * 3;
        let d = (w - 1 - x) * 3;
        dst[d] = src[s];
        dst[d + 1] = src[s + 1];
        dst[d + 2] = src[s + 2];
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mixup_blend_u8_avx2(dst: &mut [u8], other: &[u8], wa: u32) {
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let op = other.as_ptr();
    let wb = 256 - wa;
    let vwa = _mm256_set1_epi16(wa as i16);
    let vwb = _mm256_set1_epi16(wb as i16);
    let v128 = _mm256_set1_epi16(128);
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n == dst.len() == other.len()`.
        // Every lane satisfies `a * wa + b * wb + 128 <= 255 * 256 + 128 = 65408 <= u16::MAX`, so modular
        // 16-bit multiply/add + logical `>> 8` is exact and produces values in `0..=255`.
        unsafe {
            let a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(dp.add(i) as *const __m128i));
            let b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(op.add(i) as *const __m128i));
            let s0 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(a0, vwa), _mm256_mullo_epi16(b0, vwb)),
                v128,
            );
            let r0 = _mm256_srli_epi16::<8>(s0);

            let a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(dp.add(i + 16) as *const __m128i));
            let b1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(op.add(i + 16) as *const __m128i));
            let s1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(a1, vwa), _mm256_mullo_epi16(b1, vwb)),
                v128,
            );
            let r1 = _mm256_srli_epi16::<8>(s1);

            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r0, r1));
            _mm256_storeu_si256(dp.add(i) as *mut __m256i, packed);
        }
        i += 32;
    }
    while i < n {
        dst[i] = ((dst[i] as u32 * wa + other[i] as u32 * wb + 128) >> 8) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mixup_blend_u8_into_avx2(a: &[u8], b: &[u8], wa: u32, dst: &mut [u8]) {
    let n = dst.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();
    let wb = 256 - wa;
    let vwa = _mm256_set1_epi16(wa as i16);
    let vwb = _mm256_set1_epi16(wb as i16);
    let v128 = _mm256_set1_epi16(128);
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n == a.len() == b.len() == dst.len()`.
        unsafe {
            let a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
            let b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
            let r0 = _mm256_srli_epi16::<8>(_mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(a0, vwa), _mm256_mullo_epi16(b0, vwb)),
                v128,
            ));

            let a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(ap.add(i + 16) as *const __m128i));
            let b1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(bp.add(i + 16) as *const __m128i));
            let r1 = _mm256_srli_epi16::<8>(_mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(a1, vwa), _mm256_mullo_epi16(b1, vwb)),
                v128,
            ));

            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r0, r1));
            _mm256_storeu_si256(dp.add(i) as *mut __m256i, packed);
        }
        i += 32;
    }
    while i < n {
        dst[i] = ((a[i] as u32 * wa + b[i] as u32 * wb + 128) >> 8) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgba_to_rgb_avx2(src: &[u8], dst: &mut [u8], n: usize) {
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    let mut i = 0usize;
    // 16-byte overshoot store at `i * 3` needs `i * 3 + 16 <= n * 3` => `i + 6 <= n`.
    while i + 6 <= n {
        // SAFETY: `i * 4 + 16 <= src.len()` and `i * 3 + 16 <= dst.len()`.
        unsafe {
            let v = _mm_loadu_si128(sp.add(i * 4) as *const __m128i);
            _mm_storeu_si128(dp.add(i * 3) as *mut __m128i, _mm_shuffle_epi8(v, mask));
        }
        i += 4;
    }
    while i < n {
        dst[i * 3] = src[i * 4];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn luma_to_rgb_avx2(src: &[u8], dst: &mut [u8]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let m0 = _mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5);
    let m1 = _mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10);
    let m2 = _mm_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15);
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= src.len()` and `(i + 16) * 3 <= dst.len()`.
        unsafe {
            let v = _mm_loadu_si128(sp.add(i) as *const __m128i);
            _mm_storeu_si128(dp.add(i * 3) as *mut __m128i, _mm_shuffle_epi8(v, m0));
            _mm_storeu_si128(dp.add(i * 3 + 16) as *mut __m128i, _mm_shuffle_epi8(v, m1));
            _mm_storeu_si128(dp.add(i * 3 + 32) as *mut __m128i, _mm_shuffle_epi8(v, m2));
        }
        i += 16;
    }
    while i < n {
        let l = src[i];
        dst[i * 3] = l;
        dst[i * 3 + 1] = l;
        dst[i * 3 + 2] = l;
        i += 1;
    }
}
