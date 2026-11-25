//! Bit-exact explicit AVX2 HSV colour jitter and L1-fused `warp / RGB → HSV → flip_lr → planar (u8 / f32)` passes.
//!
//! Mathematical identities verified bit-exact over all 2^24 RGB triples:
//! - Packed `max_lut[m] = vlut[m] | (recip[m] << 8)` halves the `max`-indexed table gathers (3 gathers/8 px).
//! - Exact division by `255` for `0 <= x <= 65025`:
//!   `(x + 127) / 255 == ((x + 128) + ((x + 128) >> 8)) >> 8`.
//! - Exact division by `255 * 256` for `0 <= y <= 16_646_400`:
//!   `let a = (y + 32896) >> 8; (y + 32640) / 65280 == (a + (a >> 8)) >> 8`.

use super::PlanarRow;
use super::bilinear::{ROUND, SHIFT, WONE};
use super::planar::{rgb_to_planar_f32_div255, rgb_to_planar_u8};
#[cfg(target_arch = "x86_64")]
use super::{INV255_HI, INV255_LO, has_avx2, has_avx2_fma};

/// Hue units per full circle (6 sectors of 256).
pub(crate) const HUE_CIRCLE: i32 = 6 * 256;

/// Compile-time rounded fixed-point reciprocals `round(2^16 / d)` for `d in 1..=255` (`RECIP[0] = 0`).
const RECIP: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut d = 1usize;
    while d < 256 {
        t[d] = ((1u32 << 16) + d as u32 / 2) / d as u32;
        d += 1;
    }
    t
};

/// Exact IEEE-754 `i as f32 / 255.0` lookup table for division-free scalar planar `f32` conversion.
const DIV255_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = i as f32 / 255.0;
        i += 1;
    }
    t
};

/// Precomputed integer HSV tables, built once per image.
pub(crate) struct HsvTables {
    pub shift: i32,
    pub slut: [u8; 256],
    pub vlut: [u8; 256],
    /// True when every saturation entry maps to `0` (e.g. `sat_gain == 0.0`): every pixel becomes
    /// `[vlut[max(r, g, b)]; 3]`, bit-exact with the full hexcone kernel.
    pub zero_sat: bool,
    /// `slut[i] as u32` for direct AVX2 lookup without per-row stack rebuilds.
    #[cfg(target_arch = "x86_64")]
    pub slut_u32: [u32; 256],
    /// `(vlut[i] as u32) | (RECIP[i] << 8)`: one lookup delivers both `v` and `RECIP[max]`.
    #[cfg(target_arch = "x86_64")]
    pub max_lut: [u32; 256],
}

impl HsvTables {
    /// Builds HSV lookup tables, or `None` when the transform is the identity.
    pub(crate) fn new(hue_turns: f32, sat_gain: f32, val_gain: f32) -> Option<Self> {
        let shift = ((hue_turns * HUE_CIRCLE as f32).round() as i32).rem_euclid(HUE_CIRCLE);
        if shift == 0 && sat_gain == 1.0 && val_gain == 1.0 {
            return None;
        }
        let lut = |g: f32| -> [u8; 256] {
            let mut t = [0u8; 256];
            for (i, v) in t.iter_mut().enumerate() {
                *v = (i as f32 * g).round().clamp(0.0, 255.0) as u8;
            }
            t
        };
        let (slut, vlut) = (lut(sat_gain), lut(val_gain));
        let zero_sat = slut.iter().all(|&v| v == 0);
        #[cfg(target_arch = "x86_64")]
        let slut_u32 = std::array::from_fn(|i| slut[i] as u32);
        #[cfg(target_arch = "x86_64")]
        let max_lut = std::array::from_fn(|i| (vlut[i] as u32) | (RECIP[i] << 8));
        Some(HsvTables {
            shift,
            slut,
            vlut,
            zero_sat,
            #[cfg(target_arch = "x86_64")]
            slut_u32,
            #[cfg(target_arch = "x86_64")]
            max_lut,
        })
    }

    /// Applies this HSV transform to one pixel (division-free scalar kernel, bit-exact with the integer hexcone
    /// specification across all 2^24 RGB triples).
    #[inline]
    pub(crate) fn apply_pixel(&self, px: [u8; 3]) -> [u8; 3] {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let max = r.max(g).max(b);
        let v = self.vlut[max as usize] as i32;
        if self.zero_sat {
            return [v as u8; 3];
        }
        let min = r.min(g).min(b);
        let delta = max - min;
        if delta == 0 {
            return [v as u8; 3];
        }
        let rd = RECIP[delta as usize] as i32;
        let mut h = if max == r {
            ((g - b) * rd) >> 8
        } else if max == g {
            512 + (((b - r) * rd) >> 8)
        } else {
            1024 + (((r - g) * rd) >> 8)
        };
        h += self.shift;
        if h < 0 {
            h += HUE_CIRCLE;
        } else if h >= HUE_CIRCLE {
            h -= HUE_CIRCLE;
        }
        let s0 = ((delta * 255 * RECIP[max as usize] as i32 + (1 << 15)) >> 16).min(255);
        let s = self.slut[s0 as usize] as i32;
        let sector = h >> 8;
        let frac = h & 255;
        let xp = v * (255 - s) + 128;
        let p = (xp + (xp >> 8)) >> 8;
        let sf = s * frac;
        let aq = (v * (255 * 256 - sf) + (255 * 128 + 256)) >> 8;
        let q = (aq + (aq >> 8)) >> 8;
        let at = (v * (255 * 256 - ((s << 8) - sf)) + (255 * 128 + 256)) >> 8;
        let t = (at + (at >> 8)) >> 8;
        let (r2, g2, b2) = match sector {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };
        [r2 as u8, g2 as u8, b2 as u8]
    }
}

/// Applies HSV in place on one contiguous slice of interleaved RGB pixels (`pixels * 3` bytes).
#[inline]
pub(crate) fn hsv_row_inplace(row: &mut [u8], t: &HsvTables) {
    if t.zero_sat {
        hsv_row_zero_sat_inplace(row, &t.vlut);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && row.len() >= 48 {
        // SAFETY: CPU supports AVX2.
        unsafe { hsv_row_inplace_avx2(row, t) };
        return;
    }
    hsv_row_inplace_scalar(row, t);
}

#[inline]
fn hsv_row_zero_sat_inplace(row: &mut [u8], vlut: &[u8; 256]) {
    for px in row.as_chunks_mut::<3>().0.iter_mut() {
        let v = vlut[px[0].max(px[1]).max(px[2]) as usize];
        *px = [v, v, v];
    }
}

#[inline]
pub(crate) fn hsv_row_inplace_scalar(row: &mut [u8], t: &HsvTables) {
    for px in row.as_chunks_mut::<3>().0.iter_mut() {
        *px = t.apply_pixel(*px);
    }
}

/// Single-pass L1-fused `RGB row → optional HSV → optional flip_lr → planar (u8 or f32/255)`.
#[inline]
pub(crate) fn rgb_row_to_planar_fused(
    src: &[u8],
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let w = src.len() / 3;
    assert_eq!(src.len(), w * 3);
    match dst {
        PlanarRow::U8 { r, g, b } => assert!(r.len() == w && g.len() == w && b.len() == w),
        PlanarRow::F32Div255 { r, g, b } => assert!(r.len() == w && g.len() == w && b.len() == w),
    }
    match hsv {
        None => match dst {
            PlanarRow::U8 { r, g, b } => rgb_to_planar_u8(src, r, g, b, flip_lr),
            PlanarRow::F32Div255 { r, g, b } => rgb_to_planar_f32_div255(src, r, g, b, flip_lr),
        },
        Some(t) => {
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma()
                && w >= 16
                && let PlanarRow::F32Div255 { r, g, b } = dst
            {
                // SAFETY: CPU supports AVX2+FMA and row lengths are asserted to match `w`.
                unsafe { rgb_row_hsv_to_planar_f32_avx2_fma(src, t, flip_lr, r, g, b) };
                return;
            }
            #[cfg(target_arch = "x86_64")]
            if has_avx2() && w >= 16 {
                // SAFETY: CPU supports AVX2 and row lengths are asserted to match `w`.
                unsafe { rgb_row_hsv_to_planar_avx2(src, t, flip_lr, dst) };
                return;
            }
            rgb_row_hsv_to_planar_scalar(src, t, flip_lr, dst);
        }
    }
}

#[inline]
pub(crate) fn rgb_row_hsv_to_planar_scalar(
    src: &[u8],
    t: &HsvTables,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let w = src.len() / 3;
    let pixels = src.as_chunks::<3>().0;
    match dst {
        PlanarRow::U8 { r, g, b } => {
            if flip_lr {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    let d = w - 1 - x;
                    r[d] = o[0];
                    g[d] = o[1];
                    b[d] = o[2];
                }
            } else {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    r[x] = o[0];
                    g[x] = o[1];
                    b[x] = o[2];
                }
            }
        }
        PlanarRow::F32Div255 { r, g, b } => {
            if flip_lr {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    let d = w - 1 - x;
                    r[d] = DIV255_LUT[o[0] as usize];
                    g[d] = DIV255_LUT[o[1] as usize];
                    b[d] = DIV255_LUT[o[2] as usize];
                }
            } else {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    r[x] = DIV255_LUT[o[0] as usize];
                    g[x] = DIV255_LUT[o[1] as usize];
                    b[x] = DIV255_LUT[o[2] as usize];
                }
            }
        }
    }
}

/// Single-pass L1-fused `warp_rows interior horizontal interpolation → optional HSV → optional flip_lr → planar`.
///
/// Writes source columns `x in x_start .. x_end` (where `rel_fu0` is the relative fixed-point position at `x_start`,
/// guaranteed in-bounds for every `x < x_end`) into destination row `dst` of total `width` pixels. Border ranges
/// outside `[x_start, x_end)` are handled by the caller.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn warp_interior_to_planar_fused(
    v: &[u32],
    rel_fu0: i64,
    du: i64,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    assert!(x_start <= x_end && x_end <= width);
    match dst {
        PlanarRow::U8 { r, g, b } => assert!(r.len() == width && g.len() == width && b.len() == width),
        PlanarRow::F32Div255 { r, g, b } => assert!(r.len() == width && g.len() == width && b.len() == width),
    }
    let count = x_end - x_start;
    if count == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cols = v.len() / 4;
        let hi_excl = (cols as i128 - 1) << 16;
        let last_rel = rel_fu0 as i128 + (count as i128 - 1) * du as i128;
        if has_avx2()
            && count >= 8
            && (2..=32000).contains(&cols)
            && du.abs() <= 4 << 16
            && (0..hi_excl).contains(&(rel_fu0 as i128))
            && (0..hi_excl).contains(&last_rel)
        {
            if has_avx2_fma()
                && let PlanarRow::F32Div255 { r, g, b } = dst
            {
                // SAFETY: CPU supports AVX2+FMA; `dst` lengths match `width`, `x_end <= width`, and endpoints prove bounds.
                unsafe {
                    warp_interior_to_planar_f32_fused_avx2_fma(
                        v,
                        rel_fu0 as i32,
                        du as i32,
                        width,
                        x_start,
                        x_end,
                        hsv,
                        flip_lr,
                        r,
                        g,
                        b,
                    )
                };
                return;
            }
            // SAFETY: CPU supports AVX2; `dst` lengths match `width`, `x_end <= width`, and by linearity both
            // endpoints of `rel_fu` lie in `0..((cols-1)<<16)` with `cols <= 32000` and `|du| <= 4<<16`.
            unsafe {
                warp_interior_to_planar_fused_avx2(
                    v,
                    rel_fu0 as i32,
                    du as i32,
                    width,
                    x_start,
                    x_end,
                    hsv,
                    flip_lr,
                    dst,
                )
            };
            return;
        }
    }
    warp_interior_to_planar_fused_scalar(v, rel_fu0, du, width, x_start, x_end, hsv, flip_lr, dst);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn warp_interior_to_planar_fused_scalar(
    v: &[u32],
    rel_fu0: i64,
    du: i64,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let mut rfu = rel_fu0;
    for x in x_start..x_end {
        let k = (rfu >> 16) as usize * 4;
        let wx = ((rfu >> 5) & (WONE as i64 - 1)) as u32;
        let iwx = WONE - wx;
        rfu += du;
        let mut px = [
            ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8,
            ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8,
            ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8,
        ];
        if let Some(t) = hsv {
            px = t.apply_pixel(px);
        }
        let d = if flip_lr { width - 1 - x } else { x };
        match dst {
            PlanarRow::U8 { r, g, b } => {
                r[d] = px[0];
                g[d] = px[1];
                b[d] = px[2];
            }
            PlanarRow::F32Div255 { r, g, b } => {
                r[d] = DIV255_LUT[px[0] as usize];
                g[d] = DIV255_LUT[px[1] as usize];
                b[d] = DIV255_LUT[px[2] as usize];
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// AVX2 explicit HSV & fused kernels
// ------------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::bilinear::{interp_pair_avx2, pack_8_i32_to_u64_avx2, transpose_8_rgb0_i32_avx2};
#[cfg(target_arch = "x86_64")]
use super::planar::{deinterleave_rgb16_avx2, interleave_rgb16_avx2};

/// Bit-exact AVX2 HSV transform on 8 pixels in `i32` (`0..=255`) registers `(r, g, b)`.
///
/// # Safety
/// CPU supports AVX2; every lane of `r`, `g`, `b` is in `0..=255`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn hsv_8_i32_avx2(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    t: &HsvTables,
) -> (__m256i, __m256i, __m256i) {
    let c0 = _mm256_setzero_si256();
    let c1 = _mm256_set1_epi32(1);
    let c2 = _mm256_set1_epi32(2);
    let c3 = _mm256_set1_epi32(3);
    let c4 = _mm256_set1_epi32(4);
    let c128 = _mm256_set1_epi32(128);
    let c255 = _mm256_set1_epi32(255);
    let c512 = _mm256_set1_epi32(512);
    let c1024 = _mm256_set1_epi32(1024);
    let c1535 = _mm256_set1_epi32(HUE_CIRCLE - 1);
    let c1536 = _mm256_set1_epi32(HUE_CIRCLE);
    let c32768 = _mm256_set1_epi32(1 << 15);
    let c32896 = _mm256_set1_epi32(255 * 128 + 256);
    let c65280 = _mm256_set1_epi32(255 * 256);
    let vshift = _mm256_set1_epi32(t.shift);

    let max = _mm256_max_epi32(_mm256_max_epi32(r, g), b);
    // SAFETY: `max` is in `0..=255`; `max_lut` has 256 `u32` entries.
    let packed_max = unsafe { _mm256_i32gather_epi32::<4>(t.max_lut.as_ptr() as *const i32, max) };
    let v = _mm256_and_si256(packed_max, c255);
    if t.zero_sat {
        return (v, v, v);
    }
    let min = _mm256_min_epi32(_mm256_min_epi32(r, g), b);
    let delta = _mm256_sub_epi32(max, min);

    // SAFETY: `delta` is in `0..=255`; `RECIP` has 256 `u32` entries.
    let rd = unsafe { _mm256_i32gather_epi32::<4>(RECIP.as_ptr() as *const i32, delta) };
    let rmax = _mm256_srli_epi32::<8>(packed_max);

    let eq_r = _mm256_cmpeq_epi32(max, r);
    let eq_g = _mm256_cmpeq_epi32(max, g);
    let num = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_sub_epi32(r, g), _mm256_sub_epi32(b, r), eq_g),
        _mm256_sub_epi32(g, b),
        eq_r,
    );
    let off = _mm256_blendv_epi8(_mm256_blendv_epi8(c1024, c512, eq_g), c0, eq_r);

    // `((num * 256 * rd) >> 16)` == `(num * rd) >> 8` (exact, no overflow).
    let mut h =
        _mm256_add_epi32(_mm256_add_epi32(off, _mm256_srai_epi32::<8>(_mm256_mullo_epi32(num, rd))), vshift);
    let neg = _mm256_srai_epi32::<31>(h);
    h = _mm256_add_epi32(h, _mm256_and_si256(c1536, neg));
    let ge = _mm256_cmpgt_epi32(h, c1535);
    h = _mm256_sub_epi32(h, _mm256_and_si256(c1536, ge));

    let delta255 = _mm256_sub_epi32(_mm256_slli_epi32::<8>(delta), delta);
    let s0 = _mm256_min_epi32(
        _mm256_srli_epi32::<16>(_mm256_add_epi32(_mm256_mullo_epi32(delta255, rmax), c32768)),
        c255,
    );
    // SAFETY: `s0` is clamped to `0..=255`; `slut_u32` has 256 `u32` entries.
    let s = unsafe { _mm256_i32gather_epi32::<4>(t.slut_u32.as_ptr() as *const i32, s0) };

    let sector = _mm256_srli_epi32::<8>(h);
    let frac = _mm256_and_si256(h, c255);

    // Exact `(v * (255 - s) + 127) / 255`: `let a = x + 128; (a + (a >> 8)) >> 8`.
    let xp = _mm256_add_epi32(_mm256_mullo_epi32(v, _mm256_sub_epi32(c255, s)), c128);
    let p = _mm256_srli_epi32::<8>(_mm256_add_epi32(xp, _mm256_srli_epi32::<8>(xp)));

    // Exact `(y + 32640) / 65280`: `let a = (y + 32896) >> 8; (a + (a >> 8)) >> 8`.
    let sf = _mm256_mullo_epi32(s, frac);
    let s256_m_sf = _mm256_sub_epi32(_mm256_slli_epi32::<8>(s), sf);
    let yq = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, sf));
    let aq = _mm256_srli_epi32::<8>(_mm256_add_epi32(yq, c32896));
    let q = _mm256_srli_epi32::<8>(_mm256_add_epi32(aq, _mm256_srli_epi32::<8>(aq)));

    let yw = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, s256_m_sf));
    let aw = _mm256_srli_epi32::<8>(_mm256_add_epi32(yw, c32896));
    let w = _mm256_srli_epi32::<8>(_mm256_add_epi32(aw, _mm256_srli_epi32::<8>(aw)));

    // Sectors 0..5: (v, w, p), (q, v, p), (p, v, w), (p, q, v), (w, p, v), (v, p, q).
    // Defaulting to sector 5 eliminates one `cmpeq` and two `or` masks.
    let s0m = _mm256_cmpeq_epi32(sector, c0);
    let s1m = _mm256_cmpeq_epi32(sector, c1);
    let s2m = _mm256_cmpeq_epi32(sector, c2);
    let s3m = _mm256_cmpeq_epi32(sector, c3);
    let s4m = _mm256_cmpeq_epi32(sector, c4);

    let r2 = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_blendv_epi8(v, p, _mm256_or_si256(s2m, s3m)), q, s1m),
        w,
        s4m,
    );
    let g2 = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_blendv_epi8(p, v, _mm256_or_si256(s1m, s2m)), q, s3m),
        w,
        s0m,
    );
    let b2 = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_blendv_epi8(q, p, _mm256_or_si256(s0m, s1m)), v, _mm256_or_si256(s3m, s4m)),
        w,
        s2m,
    );
    (r2, g2, b2)
}

/// Packs 16 `i32`s (`lo` = `0..8`, `hi` = `8..16`, all in `0..=255`) into 16 `u8`s in `__m128i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_16_i32_to_u8_avx2(lo: __m256i, hi: __m256i) -> __m128i {
    let p16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi));
    _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256::<1>(p16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hsv_row_inplace_avx2(row: &mut [u8], t: &HsvTables) {
    let w = row.len() / 3;
    let ptr = row.as_mut_ptr();
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so `ptr.add(x * 3)` has 48 readable & writable bytes.
        unsafe {
            let (vr, vg, vb) = deinterleave_rgb16_avx2(ptr.add(x * 3));
            let (r_lo, g_lo, b_lo) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(vr),
                _mm256_cvtepu8_epi32(vg),
                _mm256_cvtepu8_epi32(vb),
                t,
            );
            let (r_hi, g_hi, b_hi) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                t,
            );
            interleave_rgb16_avx2(
                pack_16_i32_to_u8_avx2(r_lo, r_hi),
                pack_16_i32_to_u8_avx2(g_lo, g_hi),
                pack_16_i32_to_u8_avx2(b_lo, b_hi),
                ptr.add(x * 3),
            );
        }
        x += 16;
    }
    while x < w {
        let off = x * 3;
        let o = t.apply_pixel([row[off], row[off + 1], row[off + 2]]);
        row[off] = o[0];
        row[off + 1] = o[1];
        row[off + 2] = o[2];
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_row_hsv_to_planar_avx2(src: &[u8], t: &HsvTables, flip_lr: bool, dst: &mut PlanarRow<'_>) {
    let w = src.len() / 3;
    let sp = src.as_ptr();
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let v255 = _mm256_set1_ps(255.0);
    let mut x = 0usize;
    match dst {
        PlanarRow::U8 { r, g, b } => {
            let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
            while x + 16 <= w {
                // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 bytes writable at `dst_x`.
                unsafe {
                    let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
                    let (r_lo, g_lo, b_lo) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(vr),
                        _mm256_cvtepu8_epi32(vg),
                        _mm256_cvtepu8_epi32(vb),
                        t,
                    );
                    let (r_hi, g_hi, b_hi) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                        t,
                    );
                    let mut pr = pack_16_i32_to_u8_avx2(r_lo, r_hi);
                    let mut pg = pack_16_i32_to_u8_avx2(g_lo, g_hi);
                    let mut pb = pack_16_i32_to_u8_avx2(b_lo, b_hi);
                    let dst_x = if flip_lr {
                        pr = _mm_shuffle_epi8(pr, rev16);
                        pg = _mm_shuffle_epi8(pg, rev16);
                        pb = _mm_shuffle_epi8(pb, rev16);
                        w - x - 16
                    } else {
                        x
                    };
                    _mm_storeu_si128(rp.add(dst_x) as *mut __m128i, pr);
                    _mm_storeu_si128(gp.add(dst_x) as *mut __m128i, pg);
                    _mm_storeu_si128(bp.add(dst_x) as *mut __m128i, pb);
                }
                x += 16;
            }
            while x < w {
                let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
                let d = if flip_lr { w - 1 - x } else { x };
                r[d] = o[0];
                g[d] = o[1];
                b[d] = o[2];
                x += 1;
            }
        }
        PlanarRow::F32Div255 { r, g, b } => {
            let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
            while x + 16 <= w {
                // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s writable at `dst_x`.
                unsafe {
                    let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
                    let (mut r_lo, mut g_lo, mut b_lo) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(vr),
                        _mm256_cvtepu8_epi32(vg),
                        _mm256_cvtepu8_epi32(vb),
                        t,
                    );
                    let (mut r_hi, mut g_hi, mut b_hi) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                        t,
                    );
                    if flip_lr {
                        r_lo = _mm256_permutevar8x32_epi32(r_lo, rev8);
                        g_lo = _mm256_permutevar8x32_epi32(g_lo, rev8);
                        b_lo = _mm256_permutevar8x32_epi32(b_lo, rev8);
                        r_hi = _mm256_permutevar8x32_epi32(r_hi, rev8);
                        g_hi = _mm256_permutevar8x32_epi32(g_hi, rev8);
                        b_hi = _mm256_permutevar8x32_epi32(b_hi, rev8);
                        let dst_x = w - x - 16;
                        _mm256_storeu_ps(rp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(r_hi), v255));
                        _mm256_storeu_ps(rp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(r_lo), v255));
                        _mm256_storeu_ps(gp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(g_hi), v255));
                        _mm256_storeu_ps(gp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(g_lo), v255));
                        _mm256_storeu_ps(bp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(b_hi), v255));
                        _mm256_storeu_ps(bp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(b_lo), v255));
                    } else {
                        _mm256_storeu_ps(rp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(r_lo), v255));
                        _mm256_storeu_ps(rp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(r_hi), v255));
                        _mm256_storeu_ps(gp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(g_lo), v255));
                        _mm256_storeu_ps(gp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(g_hi), v255));
                        _mm256_storeu_ps(bp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(b_lo), v255));
                        _mm256_storeu_ps(bp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(b_hi), v255));
                    }
                }
                x += 16;
            }
            while x < w {
                let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
                let d = if flip_lr { w - 1 - x } else { x };
                r[d] = o[0] as f32 / 255.0;
                g[d] = o[1] as f32 / 255.0;
                b[d] = o[2] as f32 / 255.0;
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn warp_interior_to_planar_fused_avx2(
    v: &[u32],
    rel_fu0: i32,
    du: i32,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let vp = v.as_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let v255 = _mm256_set1_ps(255.0);
    let du4 = du * 4;
    let mut rfu = rel_fu0;
    let mut x = x_start;

    while x + 8 <= x_end {
        let r0 = rfu;
        let r1 = r0 + du;
        let r2 = r1 + du;
        let r3 = r2 + du;
        let r4 = r0 + du4;
        let r5 = r1 + du4;
        let r6 = r2 + du4;
        let r7 = r3 + du4;
        rfu = r4 + du4;

        // SAFETY: caller guarantees every tap in `x_start..x_end` has `k + 1 < v.len() / 4`, and `x + 8 <= width`.
        unsafe {
            let p04 = interp_pair_avx2(
                vp,
                (r0 >> 16) as usize,
                (r0 >> 5) & 2047,
                (r4 >> 16) as usize,
                (r4 >> 5) & 2047,
                vround,
            );
            let p15 = interp_pair_avx2(
                vp,
                (r1 >> 16) as usize,
                (r1 >> 5) & 2047,
                (r5 >> 16) as usize,
                (r5 >> 5) & 2047,
                vround,
            );
            let p26 = interp_pair_avx2(
                vp,
                (r2 >> 16) as usize,
                (r2 >> 5) & 2047,
                (r6 >> 16) as usize,
                (r6 >> 5) & 2047,
                vround,
            );
            let p37 = interp_pair_avx2(
                vp,
                (r3 >> 16) as usize,
                (r3 >> 5) & 2047,
                (r7 >> 16) as usize,
                (r7 >> 5) & 2047,
                vround,
            );
            let (mut vr, mut vg, mut vb) = transpose_8_rgb0_i32_avx2(p04, p15, p26, p37);
            if let Some(t) = hsv {
                (vr, vg, vb) = hsv_8_i32_avx2(vr, vg, vb, t);
            }
            let dst_x = if flip_lr { width - x - 8 } else { x };
            match dst {
                PlanarRow::U8 { r, g, b } => {
                    (r.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vr, flip_lr));
                    (g.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vg, flip_lr));
                    (b.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vb, flip_lr));
                }
                PlanarRow::F32Div255 { r, g, b } => {
                    if flip_lr {
                        vr = _mm256_permutevar8x32_epi32(vr, rev8);
                        vg = _mm256_permutevar8x32_epi32(vg, rev8);
                        vb = _mm256_permutevar8x32_epi32(vb, rev8);
                    }
                    _mm256_storeu_ps(r.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vr), v255));
                    _mm256_storeu_ps(g.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vg), v255));
                    _mm256_storeu_ps(b.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vb), v255));
                }
            }
        }
        x += 8;
    }

    if x < x_end {
        warp_interior_to_planar_fused_scalar(v, rfu as i64, du as i64, width, x, x_end, hsv, flip_lr, dst);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rgb_row_hsv_to_planar_f32_avx2_fma(
    src: &[u8],
    t: &HsvTables,
    flip_lr: bool,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = src.len() / 3;
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s writable at `dst_x`.
        unsafe {
            let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
            let (mut r_lo, mut g_lo, mut b_lo) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(vr),
                _mm256_cvtepu8_epi32(vg),
                _mm256_cvtepu8_epi32(vb),
                t,
            );
            let (mut r_hi, mut g_hi, mut b_hi) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                t,
            );
            let (dst_x, first, second) = if flip_lr {
                r_lo = _mm256_permutevar8x32_epi32(r_lo, rev8);
                g_lo = _mm256_permutevar8x32_epi32(g_lo, rev8);
                b_lo = _mm256_permutevar8x32_epi32(b_lo, rev8);
                r_hi = _mm256_permutevar8x32_epi32(r_hi, rev8);
                g_hi = _mm256_permutevar8x32_epi32(g_hi, rev8);
                b_hi = _mm256_permutevar8x32_epi32(b_hi, rev8);
                (w - x - 16, [(r_hi, rp), (g_hi, gp), (b_hi, bp)], [(r_lo, rp), (g_lo, gp), (b_lo, bp)])
            } else {
                (x, [(r_lo, rp), (g_lo, gp), (b_lo, bp)], [(r_hi, rp), (g_hi, gp), (b_hi, bp)])
            };
            for (v, dp) in first {
                let f = _mm256_cvtepi32_ps(v);
                _mm256_storeu_ps(dp.add(dst_x), _mm256_fmadd_ps(f, vhi, _mm256_mul_ps(f, vlo)));
            }
            for (v, dp) in second {
                let f = _mm256_cvtepi32_ps(v);
                _mm256_storeu_ps(dp.add(dst_x + 8), _mm256_fmadd_ps(f, vhi, _mm256_mul_ps(f, vlo)));
            }
        }
        x += 16;
    }
    while x < w {
        let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
        let d = if flip_lr { w - 1 - x } else { x };
        r[d] = o[0] as f32 / 255.0;
        g[d] = o[1] as f32 / 255.0;
        b[d] = o[2] as f32 / 255.0;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn warp_interior_to_planar_f32_fused_avx2_fma(
    v: &[u32],
    rel_fu0: i32,
    du: i32,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let vp = v.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let du4 = du * 4;
    let mut rfu = rel_fu0;
    let mut x = x_start;

    while x + 8 <= x_end {
        let r0 = rfu;
        let r1 = r0 + du;
        let r2 = r1 + du;
        let r3 = r2 + du;
        let r4 = r0 + du4;
        let r5 = r1 + du4;
        let r6 = r2 + du4;
        let r7 = r3 + du4;
        rfu = r4 + du4;

        // SAFETY: caller guarantees every tap in `x_start..x_end` has `k + 1 < v.len() / 4`, and `x + 8 <= width`.
        unsafe {
            let p04 = interp_pair_avx2(
                vp,
                (r0 >> 16) as usize,
                (r0 >> 5) & 2047,
                (r4 >> 16) as usize,
                (r4 >> 5) & 2047,
                vround,
            );
            let p15 = interp_pair_avx2(
                vp,
                (r1 >> 16) as usize,
                (r1 >> 5) & 2047,
                (r5 >> 16) as usize,
                (r5 >> 5) & 2047,
                vround,
            );
            let p26 = interp_pair_avx2(
                vp,
                (r2 >> 16) as usize,
                (r2 >> 5) & 2047,
                (r6 >> 16) as usize,
                (r6 >> 5) & 2047,
                vround,
            );
            let p37 = interp_pair_avx2(
                vp,
                (r3 >> 16) as usize,
                (r3 >> 5) & 2047,
                (r7 >> 16) as usize,
                (r7 >> 5) & 2047,
                vround,
            );
            let (mut vr, mut vg, mut vb) = transpose_8_rgb0_i32_avx2(p04, p15, p26, p37);
            if let Some(t) = hsv {
                (vr, vg, vb) = hsv_8_i32_avx2(vr, vg, vb, t);
            }
            let dst_x = if flip_lr {
                vr = _mm256_permutevar8x32_epi32(vr, rev8);
                vg = _mm256_permutevar8x32_epi32(vg, rev8);
                vb = _mm256_permutevar8x32_epi32(vb, rev8);
                width - x - 8
            } else {
                x
            };
            let fr = _mm256_cvtepi32_ps(vr);
            let fg = _mm256_cvtepi32_ps(vg);
            let fb = _mm256_cvtepi32_ps(vb);
            _mm256_storeu_ps(rp.add(dst_x), _mm256_fmadd_ps(fr, vhi, _mm256_mul_ps(fr, vlo)));
            _mm256_storeu_ps(gp.add(dst_x), _mm256_fmadd_ps(fg, vhi, _mm256_mul_ps(fg, vlo)));
            _mm256_storeu_ps(bp.add(dst_x), _mm256_fmadd_ps(fb, vhi, _mm256_mul_ps(fb, vlo)));
        }
        x += 8;
    }

    if x < x_end {
        warp_interior_to_planar_fused_scalar(
            v,
            rfu as i64,
            du as i64,
            width,
            x,
            x_end,
            hsv,
            flip_lr,
            &mut PlanarRow::F32Div255 { r, g, b },
        );
    }
}
