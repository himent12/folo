//! Bit-exact gather-free AVX2 HSV colour jitter and lookup tables.
//!
//! Mathematical identities verified bit-exact over all 2^24 RGB triples:
//! - Zero `vpgatherdd` instructions: `RECIP[d] = round(2^16 / d)` for `d in 1..=255` is exact via IEEE-754
//!   `cvtps_epi32(65536.0 / d)` (with `max(d, 1)` preventing zero-lane FP assists), `vlut`/`slut` use exact
//!   gain scaling, and the 6-sector RGB dispatch uses byte-packed `vpshufb` + `vpsrlvd`.
//! - Exact division by `255` for `0 <= x <= 65025`:
//!   `(x + 127) / 255 == ((x + 128) + ((x + 128) >> 8)) >> 8`.
//! - Exact division by `255 * 256` for `0 <= y <= 16_646_400`:
//!   `let a = (y + 32896) >> 8; (y + 32640) / 65280 == (a + (a >> 8)) >> 8`.

mod fused;

#[cfg(test)]
pub(crate) use fused::{rgb_row_hsv_to_planar_scalar, warp_interior_to_planar_fused_scalar};
pub(crate) use fused::{rgb_row_to_planar_fused, warp_interior_to_planar_fused};

#[cfg(target_arch = "x86_64")]
use super::planar::{deinterleave_rgb16_avx2, interleave_rgb16_avx2};
#[cfg(target_arch = "x86_64")]
use super::{has_avx2, has_avx2_fma};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Hue units per full circle (6 sectors of 256).
pub(crate) const HUE_CIRCLE: i32 = 6 * 256;

/// Compile-time rounded fixed-point reciprocals `round(2^16 / d)` for `d in 1..=255` (`RECIP[0] = 0`).
pub(super) const RECIP: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut d = 1usize;
    while d < 256 {
        t[d] = ((1u32 << 16) + d as u32 / 2) / d as u32;
        d += 1;
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
    /// Linear saturation / value gains and fast-path flags for gather-free AVX2 evaluation.
    #[cfg(target_arch = "x86_64")]
    pub sat_gain: f32,
    #[cfg(target_arch = "x86_64")]
    pub val_gain: f32,
    #[cfg(target_arch = "x86_64")]
    pub fast_s: bool,
    #[cfg(target_arch = "x86_64")]
    pub fast_v: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn gain_fast_fma_exact(g: f32, lut: &[u8; 256]) -> bool {
    let vg = _mm_set_ss(g);
    let vhalf = _mm_set_ss(0.5);
    let v255 = _mm_set_ss(255.0);
    (0..256).all(|i| {
        let fi = _mm_set_ss(i as f32);
        let y = _mm_cvttss_si32(_mm_min_ss(_mm_fmadd_ss(fi, vg, vhalf), v255));
        y as u8 == lut[i]
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn gain_fast_add_exact(g: f32, lut: &[u8; 256]) -> bool {
    if !g.is_finite() || !(0.0..=256.0).contains(&g) {
        return false;
    }
    let add_ok = (0..256).all(|i| (((i as f32) * g + 0.5).min(255.0) as i32) as u8 == lut[i]);
    if !add_ok {
        return false;
    }
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma()` guarantees FMA support.
        unsafe { gain_fast_fma_exact(g, lut) }
    } else {
        true
    }
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
        let fast_s = gain_fast_add_exact(sat_gain, &slut);
        #[cfg(target_arch = "x86_64")]
        let fast_v = gain_fast_add_exact(val_gain, &vlut);
        Some(HsvTables {
            shift,
            slut,
            vlut,
            zero_sat,
            #[cfg(target_arch = "x86_64")]
            sat_gain,
            #[cfg(target_arch = "x86_64")]
            val_gain,
            #[cfg(target_arch = "x86_64")]
            fast_s,
            #[cfg(target_arch = "x86_64")]
            fast_v,
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
        #[cfg(target_arch = "x86_64")]
        if has_avx2() && row.len() >= 48 {
            // SAFETY: CPU supports AVX2.
            unsafe { hsv_row_zero_sat_inplace_avx2(row, t) };
            return;
        }
        hsv_row_zero_sat_inplace(row, &t.vlut);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && row.len() >= 48 {
        // SAFETY: CPU supports AVX2+FMA.
        unsafe { hsv_row_inplace_avx2_fma(row, t) };
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

// ------------------------------------------------------------------------------------------------
// AVX2 explicit HSV primitives and standalone in-place kernels
// ------------------------------------------------------------------------------------------------

/// Evaluates `(fx * gain).round().clamp(0.0, 255.0) as i32` across 8 lanes without table gathers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn eval_gain_lut_8_avx2(fx: __m256, gain: f32, fast: bool) -> __m256i {
    let vg = _mm256_set1_ps(gain);
    let vhalf = _mm256_set1_ps(0.5);
    let v255 = _mm256_set1_ps(255.0);
    if fast {
        _mm256_cvttps_epi32(_mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(fx, vg), vhalf), v255))
    } else {
        // Order matters: `max_ps(p, 0.0)` maps NaN -> 0.0, then `min_ps(_, 255.0)` bounds into `[0.0, 255.0]`.
        let p = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(fx, vg), _mm256_setzero_ps()), v255);
        let fl = _mm256_floor_ps(p);
        let frac = _mm256_sub_ps(p, fl);
        let inc = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(frac, vhalf)),
            _mm256_set1_epi32(1),
        );
        _mm256_add_epi32(_mm256_cvttps_epi32(fl), inc)
    }
}

/// Bit-exact gather-free AVX2 HSV transform on 8 pixels in `i32` (`0..=255`) registers `(r, g, b)`.
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
    let c255 = _mm256_set1_epi32(255);

    let max = _mm256_max_epi32(_mm256_max_epi32(r, g), b);
    let fmax = _mm256_cvtepi32_ps(max);
    let v = unsafe { eval_gain_lut_8_avx2(fmax, t.val_gain, t.fast_v) };
    if t.zero_sat {
        return (v, v, v);
    }
    let min = _mm256_min_epi32(_mm256_min_epi32(r, g), b);
    let delta = _mm256_sub_epi32(max, min);

    // `RECIP[d] = round(2^16 / d)` is bit-exact via `cvtps_epi32(65536.0 / d)` for all `d in 1..=255`
    // (`max(d, 1)` avoids zero-lane FP microcode assists; `andnot(d == 0, _)` yields `RECIP[0] == 0`).
    let v65536 = _mm256_set1_ps(65536.0);
    let safe_d = _mm256_cvtepi32_ps(_mm256_max_epi32(delta, c1));
    let safe_m = _mm256_max_ps(fmax, _mm256_set1_ps(1.0));
    let rd =
        _mm256_andnot_si256(_mm256_cmpeq_epi32(delta, c0), _mm256_cvtps_epi32(_mm256_div_ps(v65536, safe_d)));
    let rmax =
        _mm256_andnot_si256(_mm256_cmpeq_epi32(max, c0), _mm256_cvtps_epi32(_mm256_div_ps(v65536, safe_m)));

    let eq_r = _mm256_cmpeq_epi32(max, r);
    let eq_g = _mm256_cmpeq_epi32(max, g);
    let num = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_sub_epi32(r, g), _mm256_sub_epi32(b, r), eq_g),
        _mm256_sub_epi32(g, b),
        eq_r,
    );
    let off = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_set1_epi32(1024), _mm256_set1_epi32(512), eq_g),
        c0,
        eq_r,
    );

    // `((num * 256 * rd) >> 16)` == `(num * rd) >> 8` (exact, no overflow).
    let c1536 = _mm256_set1_epi32(HUE_CIRCLE);
    let mut h = _mm256_add_epi32(
        _mm256_add_epi32(off, _mm256_srai_epi32::<8>(_mm256_mullo_epi32(num, rd))),
        _mm256_set1_epi32(t.shift),
    );
    let neg = _mm256_srai_epi32::<31>(h);
    h = _mm256_add_epi32(h, _mm256_and_si256(c1536, neg));
    let ge = _mm256_cmpgt_epi32(h, _mm256_set1_epi32(HUE_CIRCLE - 1));
    h = _mm256_sub_epi32(h, _mm256_and_si256(c1536, ge));

    let delta255 = _mm256_sub_epi32(_mm256_slli_epi32::<8>(delta), delta);
    let s0 = _mm256_min_epi32(
        _mm256_srli_epi32::<16>(_mm256_add_epi32(
            _mm256_mullo_epi32(delta255, rmax),
            _mm256_set1_epi32(1 << 15),
        )),
        c255,
    );
    let s = unsafe { eval_gain_lut_8_avx2(_mm256_cvtepi32_ps(s0), t.sat_gain, t.fast_s) };

    let sector = _mm256_srli_epi32::<8>(h);
    let frac = _mm256_and_si256(h, c255);

    // Exact `(v * (255 - s) + 127) / 255`: `let a = x + 128; (a + (a >> 8)) >> 8`.
    // Both `v` and `255 - s` (as well as `s` and `frac`) are in `0..=255`, so high 16 bits of each `i32` lane
    // are 0 and `vpmaddwd` (`_mm256_madd_epi16`) computes exact `a * b` in 1 uop / 5-cycle latency vs `vpmulld`.
    let xp = _mm256_add_epi32(_mm256_madd_epi16(v, _mm256_sub_epi32(c255, s)), _mm256_set1_epi32(128));
    let p = _mm256_srli_epi32::<8>(_mm256_add_epi32(xp, _mm256_srli_epi32::<8>(xp)));

    // Exact `(y + 32640) / 65280`: `let a = (y + 32896) >> 8; (a + (a >> 8)) >> 8`.
    let c32896 = _mm256_set1_epi32(255 * 128 + 256);
    let c65280 = _mm256_set1_epi32(255 * 256);
    let sf = _mm256_madd_epi16(s, frac);
    let s256_m_sf = _mm256_sub_epi32(_mm256_slli_epi32::<8>(s), sf);
    let yq = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, sf));
    let aq = _mm256_srli_epi32::<8>(_mm256_add_epi32(yq, c32896));
    let q = _mm256_srli_epi32::<8>(_mm256_add_epi32(aq, _mm256_srli_epi32::<8>(aq)));

    let yw = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, s256_m_sf));
    let aw = _mm256_srli_epi32::<8>(_mm256_add_epi32(yw, c32896));
    let w = _mm256_srli_epi32::<8>(_mm256_add_epi32(aw, _mm256_srli_epi32::<8>(aw)));

    // Pack `[v, p, q, w]` into the 4 bytes of each `u32` lane and select per-channel via `vpshufb + vpsrlvd`.
    // Sectors 0..5: (v, w, p), (q, v, p), (p, v, w), (p, q, v), (w, p, v), (v, p, q).
    let vpqw = _mm256_or_si256(
        _mm256_or_si256(v, _mm256_slli_epi32::<8>(p)),
        _mm256_or_si256(_mm256_slli_epi32::<16>(q), _mm256_slli_epi32::<24>(w)),
    );
    let sec_idx = _mm256_or_si256(sector, _mm256_set1_epi32(0x8080_8000u32 as i32));
    let r_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 16, 8, 8, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let g_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(24, 0, 0, 16, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let b_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(8, 8, 24, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let r2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, r_sh), c255);
    let g2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, g_sh), c255);
    let b2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, b_sh), c255);
    (r2, g2, b2)
}

/// Evaluates `(fx * gain).round().clamp(0.0, 255.0) as i32` using FMA on the fast path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn eval_gain_lut_8_avx2_fma(fx: __m256, gain: f32, fast: bool) -> __m256i {
    let vg = _mm256_set1_ps(gain);
    let vhalf = _mm256_set1_ps(0.5);
    let v255 = _mm256_set1_ps(255.0);
    if fast {
        _mm256_cvttps_epi32(_mm256_min_ps(_mm256_fmadd_ps(fx, vg, vhalf), v255))
    } else {
        let p = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(fx, vg), _mm256_setzero_ps()), v255);
        let fl = _mm256_floor_ps(p);
        let frac = _mm256_sub_ps(p, fl);
        let inc = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(frac, vhalf)),
            _mm256_set1_epi32(1),
        );
        _mm256_add_epi32(_mm256_cvttps_epi32(fl), inc)
    }
}

/// Computes `RECIP[d] = round(65536.0 / d)` for 8 lanes (`d in 1..=255`) using `vrcpps` + 1 FMA Newton-Raphson
/// step, eliminating `vdivps` latency/throughput bottlenecks.
///
/// Exactness proof for all x86_64 CPUs conforming to the `vrcpps` ISA spec (`|x * r0 - 1| <= 1.5 * 2^-12`):
/// after one FMA NR step `r1 = r0 + r0 * (1 - x * r0)`, `|65536 * r1 - 65536 / d| <= 0.0127 / d`, whereas for
/// every `d in 1..=255` the exact quotient `65536 / d` stays at least `0.5 / d` away from every half-integer
/// `k + 0.5` (a ~40x margin), so `cvtps_epi32(65536.0 * r1)` equals `RECIP[d]` identically.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(super) unsafe fn rcp65536_8_avx2_fma(x: __m256, v1: __m256, v65536: __m256) -> __m256i {
    let r0 = _mm256_rcp_ps(x);
    let r1 = _mm256_fmadd_ps(r0, _mm256_fnmadd_ps(x, r0, v1), r0);
    _mm256_cvtps_epi32(_mm256_mul_ps(v65536, r1))
}

/// Division-free, gather-free AVX2+FMA HSV transform on 8 pixels in `i32` (`0..=255`) registers `(r, g, b)`.
///
/// # Safety
/// CPU supports AVX2 and FMA; every lane of `r`, `g`, `b` is in `0..=255`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(crate) unsafe fn hsv_8_i32_avx2_fma(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    t: &HsvTables,
) -> (__m256i, __m256i, __m256i) {
    let c0 = _mm256_setzero_si256();
    let c1 = _mm256_set1_epi32(1);
    let c255 = _mm256_set1_epi32(255);

    let max = _mm256_max_epi32(_mm256_max_epi32(r, g), b);
    let fmax = _mm256_cvtepi32_ps(max);
    let v = unsafe { eval_gain_lut_8_avx2_fma(fmax, t.val_gain, t.fast_v) };
    if t.zero_sat {
        return (v, v, v);
    }
    let min = _mm256_min_epi32(_mm256_min_epi32(r, g), b);
    let delta = _mm256_sub_epi32(max, min);

    let v1 = _mm256_set1_ps(1.0);
    let v65536 = _mm256_set1_ps(65536.0);
    let safe_d = _mm256_cvtepi32_ps(_mm256_max_epi32(delta, c1));
    let safe_m = _mm256_max_ps(fmax, v1);
    let rd = _mm256_andnot_si256(_mm256_cmpeq_epi32(delta, c0), unsafe {
        rcp65536_8_avx2_fma(safe_d, v1, v65536)
    });
    let rmax =
        _mm256_andnot_si256(_mm256_cmpeq_epi32(max, c0), unsafe { rcp65536_8_avx2_fma(safe_m, v1, v65536) });

    let eq_r = _mm256_cmpeq_epi32(max, r);
    let eq_g = _mm256_cmpeq_epi32(max, g);
    let num = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_sub_epi32(r, g), _mm256_sub_epi32(b, r), eq_g),
        _mm256_sub_epi32(g, b),
        eq_r,
    );
    let off = _mm256_blendv_epi8(
        _mm256_blendv_epi8(_mm256_set1_epi32(1024), _mm256_set1_epi32(512), eq_g),
        c0,
        eq_r,
    );

    let c1536 = _mm256_set1_epi32(HUE_CIRCLE);
    let mut h = _mm256_add_epi32(
        _mm256_add_epi32(off, _mm256_srai_epi32::<8>(_mm256_mullo_epi32(num, rd))),
        _mm256_set1_epi32(t.shift),
    );
    let neg = _mm256_srai_epi32::<31>(h);
    h = _mm256_add_epi32(h, _mm256_and_si256(c1536, neg));
    let ge = _mm256_cmpgt_epi32(h, _mm256_set1_epi32(HUE_CIRCLE - 1));
    h = _mm256_sub_epi32(h, _mm256_and_si256(c1536, ge));

    let delta255 = _mm256_sub_epi32(_mm256_slli_epi32::<8>(delta), delta);
    let s0 = _mm256_min_epi32(
        _mm256_srli_epi32::<16>(_mm256_add_epi32(
            _mm256_mullo_epi32(delta255, rmax),
            _mm256_set1_epi32(1 << 15),
        )),
        c255,
    );
    let s = unsafe { eval_gain_lut_8_avx2_fma(_mm256_cvtepi32_ps(s0), t.sat_gain, t.fast_s) };

    let sector = _mm256_srli_epi32::<8>(h);
    let frac = _mm256_and_si256(h, c255);

    let xp = _mm256_add_epi32(_mm256_madd_epi16(v, _mm256_sub_epi32(c255, s)), _mm256_set1_epi32(128));
    let p = _mm256_srli_epi32::<8>(_mm256_add_epi32(xp, _mm256_srli_epi32::<8>(xp)));

    let c32896 = _mm256_set1_epi32(255 * 128 + 256);
    let c65280 = _mm256_set1_epi32(255 * 256);
    let sf = _mm256_madd_epi16(s, frac);
    let s256_m_sf = _mm256_sub_epi32(_mm256_slli_epi32::<8>(s), sf);
    let yq = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, sf));
    let aq = _mm256_srli_epi32::<8>(_mm256_add_epi32(yq, c32896));
    let q = _mm256_srli_epi32::<8>(_mm256_add_epi32(aq, _mm256_srli_epi32::<8>(aq)));

    let yw = _mm256_mullo_epi32(v, _mm256_sub_epi32(c65280, s256_m_sf));
    let aw = _mm256_srli_epi32::<8>(_mm256_add_epi32(yw, c32896));
    let w = _mm256_srli_epi32::<8>(_mm256_add_epi32(aw, _mm256_srli_epi32::<8>(aw)));

    let vpqw = _mm256_or_si256(
        _mm256_or_si256(v, _mm256_slli_epi32::<8>(p)),
        _mm256_or_si256(_mm256_slli_epi32::<16>(q), _mm256_slli_epi32::<24>(w)),
    );
    let sec_idx = _mm256_or_si256(sector, _mm256_set1_epi32(0x8080_8000u32 as i32));
    let r_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 16, 8, 8, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let g_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(24, 0, 0, 16, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let b_sh = _mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_setr_epi8(8, 8, 24, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        sec_idx,
    );
    let r2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, r_sh), c255);
    let g2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, g_sh), c255);
    let b2 = _mm256_and_si256(_mm256_srlv_epi32(vpqw, b_sh), c255);
    (r2, g2, b2)
}

/// Packs 16 `i32`s (`lo` = `0..8`, `hi` = `8..16`, all in `0..=255`) into 16 `u8`s in `__m128i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn pack_16_i32_to_u8_avx2(lo: __m256i, hi: __m256i) -> __m128i {
    let p16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi));
    _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256::<1>(p16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsv_row_zero_sat_inplace_avx2(row: &mut [u8], t: &HsvTables) {
    let w = row.len() / 3;
    let ptr = row.as_mut_ptr();
    let m0 = _mm_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5);
    let m1 = _mm_setr_epi8(5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10);
    let m2 = _mm_setr_epi8(10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so `ptr.add(x * 3)` has 48 readable & writable bytes.
        unsafe {
            let (vr, vg, vb) = deinterleave_rgb16_avx2(ptr.add(x * 3));
            let vmax = _mm_max_epu8(_mm_max_epu8(vr, vg), vb);
            let v_lo =
                eval_gain_lut_8_avx2(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vmax)), t.val_gain, t.fast_v);
            let v_hi = eval_gain_lut_8_avx2(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vmax))),
                t.val_gain,
                t.fast_v,
            );
            let v16 = pack_16_i32_to_u8_avx2(v_lo, v_hi);
            _mm_storeu_si128(ptr.add(x * 3) as *mut __m128i, _mm_shuffle_epi8(v16, m0));
            _mm_storeu_si128(ptr.add(x * 3 + 16) as *mut __m128i, _mm_shuffle_epi8(v16, m1));
            _mm_storeu_si128(ptr.add(x * 3 + 32) as *mut __m128i, _mm_shuffle_epi8(v16, m2));
        }
        x += 16;
    }
    while x < w {
        let off = x * 3;
        let v = t.vlut[row[off].max(row[off + 1]).max(row[off + 2]) as usize];
        row[off] = v;
        row[off + 1] = v;
        row[off + 2] = v;
        x += 1;
    }
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
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn hsv_row_inplace_avx2_fma(row: &mut [u8], t: &HsvTables) {
    let w = row.len() / 3;
    let ptr = row.as_mut_ptr();
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so `ptr.add(x * 3)` has 48 readable & writable bytes.
        unsafe {
            let (vr, vg, vb) = deinterleave_rgb16_avx2(ptr.add(x * 3));
            let (r_lo, g_lo, b_lo) = hsv_8_i32_avx2_fma(
                _mm256_cvtepu8_epi32(vr),
                _mm256_cvtepu8_epi32(vg),
                _mm256_cvtepu8_epi32(vb),
                t,
            );
            let (r_hi, g_hi, b_hi) = hsv_8_i32_avx2_fma(
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
