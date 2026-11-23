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
    debug_assert_eq!(top.len(), cols * 3);
    debug_assert_eq!(bot.len(), cols * 3);
    debug_assert!(dst.len() >= cols * 4);
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
        // SAFETY: CPU supports AVX2 and slice lengths satisfy `cols`.
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
    debug_assert_eq!(src.len(), cols * 3);
    debug_assert!(dst.len() >= cols * 4);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && cols >= 6 {
        // SAFETY: CPU supports AVX2 and slice lengths satisfy `cols`.
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

/// Horizontal interpolation from `v` (`(max_x0 + 2) * 4` `u32`s, where column `src_w` duplicates `src_w - 1` when
/// clamped) using precomputed taps `x0`, `wx` (`len == out_rgb.len() / 3`), writing interleaved RGB `u8`.
#[inline]
pub(crate) fn hinterp_taps_to_rgb(v: &[u32], x0: &[u32], wx: &[u32], out_rgb: &mut [u8]) {
    let w = x0.len();
    debug_assert_eq!(wx.len(), w);
    debug_assert_eq!(out_rgb.len(), w * 3);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 6 {
        // SAFETY: CPU supports AVX2; `v` has at least `(x0[x] + 2) * 4` elements for every `x < w`.
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
#[inline]
pub(crate) fn hinterp_taps_to_planar_u8(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
) {
    let w = x0.len();
    debug_assert_eq!(wx.len(), w);
    debug_assert_eq!(r.len(), w);
    debug_assert_eq!(g.len(), w);
    debug_assert_eq!(b.len(), w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `v` covers `x0[x] + 1`.
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
#[inline]
pub(crate) fn hinterp_taps_to_planar_f32_scale(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    scale: f32,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    debug_assert_eq!(wx.len(), w);
    debug_assert_eq!(r.len(), w);
    debug_assert_eq!(g.len(), w);
    debug_assert_eq!(b.len(), w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `v` covers `x0[x] + 1`.
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
#[inline]
pub(crate) fn hinterp_taps_to_planar_f32_div255(
    v: &[u32],
    x0: &[u32],
    wx: &[u32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = x0.len();
    debug_assert_eq!(wx.len(), w);
    debug_assert_eq!(r.len(), w);
    debug_assert_eq!(g.len(), w);
    debug_assert_eq!(b.len(), w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 8 {
        // SAFETY: CPU supports AVX2; caller guarantees `v` covers `x0[x] + 1`.
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
#[inline]
pub(crate) fn hinterp_taps_to_rgb0_u32(v: &[u32], x0: &[u32], wx: &[u32], dst: &mut [u32]) {
    let w = x0.len();
    debug_assert_eq!(wx.len(), w);
    debug_assert_eq!(dst.len(), w);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && w >= 4 {
        // SAFETY: CPU supports AVX2; caller guarantees `v` covers `x0[x] + 1`.
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

/// Horizontal fixed-point warp interpolation along an interior span (every `x in 0..count` is guaranteed to satisfy
/// `0 <= ((rel_fu0 + x * du) >> 16) < last`), writing interleaved RGB `u8` into `out_rgb[..count * 3]`.
#[inline]
pub(crate) fn warp_hinterp_interior_to_rgb(v: &[u32], rel_fu0: i64, du: i64, out_rgb: &mut [u8]) {
    let count = out_rgb.len() / 3;
    debug_assert_eq!(out_rgb.len(), count * 3);
    let cols = v.len() / 4;
    #[cfg(target_arch = "x86_64")]
    if has_avx2()
        && count >= 6
        && (2..=32000).contains(&cols)
        && du.abs() <= 4 << 16
        && rel_fu0 >= 0
        && rel_fu0 < ((cols as i64 - 1) << 16)
    {
        // SAFETY: CPU supports AVX2; `cols <= 32000`, `|du| <= 4<<16`, and every `k < cols - 1`, so `rel_fu` fits `i32`.
        unsafe { warp_hinterp_interior_to_rgb_avx2(v, rel_fu0 as i32, du as i32, out_rgb) };
        return;
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

/// Interior span of `warp_general` (every pixel guaranteed to have `0 <= ix < sw - 1`, `0 <= iy < sh - 1`, and
/// `iy * sstride + ix * 3 + sstride + 8 <= sdata.len()`), writing interleaved RGB into `out_rgb`.
#[inline]
pub(crate) fn warp_general_interior_to_rgb(
    sdata: &[u8],
    sstride: usize,
    fu0: i64,
    fv0: i64,
    du: i64,
    dv: i64,
    out_rgb: &mut [u8],
) {
    let count = out_rgb.len() / 3;
    debug_assert_eq!(out_rgb.len(), count * 3);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && count >= 4 {
        // SAFETY: CPU supports AVX2; caller guarantees every interior tap + 8 bytes is within `sdata`.
        unsafe { warp_general_interior_to_rgb_avx2(sdata, sstride, fu0, fv0, du, dv, out_rgb) };
        return;
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

/// Adds `src` bytes into `acc` (`u32`s) — used by `box_downsample` in `resize.rs`.
#[inline]
pub(crate) fn add_u8_to_u32(acc: &mut [u32], src: &[u8]) {
    debug_assert_eq!(acc.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && src.len() >= 16 {
        // SAFETY: CPU supports AVX2 and lengths match.
        unsafe { add_u8_to_u32_avx2(acc, src) };
        return;
    }
    for (a, &v) in acc.iter_mut().zip(src) {
        *a += v as u32;
    }
}

// ------------------------------------------------------------------------------------------------
// AVX2 implementations
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
    let w = _mm256_setr_epi32(wx0, wx0, wx0, 0, wx1, wx1, wx1, 0);
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
unsafe fn pack_4_rgb0_u8_avx2(p01: __m256i, p23: __m256i) -> __m128i {
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
    #[inline]
    unsafe fn sample_one(
        sp: *const u8,
        sstride: usize,
        fu: i64,
        fv: i64,
        pair_shuf: __m128i,
    ) -> (__m128i, __m128i, i32) {
        let ix = (fu >> 16) as usize;
        let iy = (fv >> 16) as usize;
        let wx = ((fu >> 5) & 2047) as i16;
        let wy = ((fv >> 5) & 2047) as i32;
        let iwx = (WONE as i16) - wx;
        let o = iy * sstride + ix * 3;
        // SAFETY: caller guarantees `o + sstride + 8 <= sdata.len()`.
        let (r0, r1) = unsafe {
            (
                _mm_loadl_epi64(sp.add(o) as *const __m128i),
                _mm_loadl_epi64(sp.add(o + sstride) as *const __m128i),
            )
        };
        let both = _mm_unpacklo_epi64(r0, r1);
        let paired = _mm_shuffle_epi8(both, pair_shuf);
        let w = _mm256_set1_epi32((iwx as u16 as u32 | ((wx as u16 as u32) << 16)) as i32);
        let tu = _mm256_madd_epi16(_mm256_cvtepu8_epi16(paired), w);
        (_mm256_castsi256_si128(tu), _mm256_extracti128_si256::<1>(tu), wy)
    }

    while x + 6 <= count {
        // SAFETY: caller guarantees every tap in `0..count` is interior + 8-byte readable; `x * 3 + 16 <= out_rgb.len()`.
        unsafe {
            let (t0, u0, wy0) = sample_one(sp, sstride, fu, fv, pair_shuf);
            fu += du;
            fv += dv;
            let (t1, u1, wy1) = sample_one(sp, sstride, fu, fv, pair_shuf);
            fu += du;
            fv += dv;
            let (t2, u2, wy2) = sample_one(sp, sstride, fu, fv, pair_shuf);
            fu += du;
            fv += dv;
            let (t3, u3, wy3) = sample_one(sp, sstride, fu, fv, pair_shuf);
            fu += du;
            fv += dv;

            let t01 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(t0), t1);
            let u01 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(u0), u1);
            let wy01 = _mm256_setr_epi32(wy0, wy0, wy0, 0, wy1, wy1, wy1, 0);
            let p01 = _mm256_srli_epi32::<22>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_slli_epi32::<11>(t01),
                    _mm256_mullo_epi32(_mm256_sub_epi32(u01, t01), wy01),
                ),
                vround,
            ));

            let t23 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(t2), t3);
            let u23 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(u2), u3);
            let wy23 = _mm256_setr_epi32(wy2, wy2, wy2, 0, wy3, wy3, wy3, 0);
            let p23 = _mm256_srli_epi32::<22>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_slli_epi32::<11>(t23),
                    _mm256_mullo_epi32(_mm256_sub_epi32(u23, t23), wy23),
                ),
                vround,
            ));

            let rgba = pack_4_rgb0_u8_avx2(p01, p23);
            _mm_storeu_si128(op.add(x * 3) as *mut __m128i, _mm_shuffle_epi8(rgba, rgb0_to_rgb));
        }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_u8_to_u32_avx2(acc: &mut [u32], src: &[u8]) {
    let n = src.len();
    let ap = acc.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n == acc.len() == src.len()`.
        unsafe {
            let b = _mm_loadu_si128(sp.add(i) as *const __m128i);
            let lo = _mm256_cvtepu8_epi32(b);
            let hi = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(b));
            let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
            let a1 = _mm256_loadu_si256(ap.add(i + 8) as *const __m256i);
            _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi32(a0, lo));
            _mm256_storeu_si256(ap.add(i + 8) as *mut __m256i, _mm256_add_epi32(a1, hi));
        }
        i += 16;
    }
    while i < n {
        acc[i] += src[i] as u32;
        i += 1;
    }
}
