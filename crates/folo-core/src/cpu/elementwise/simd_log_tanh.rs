//! AVX2 vectorised `log`, `tanh`, `softplus` and backward kernels with bit-exact scalar twins.
//!
//! All forward kernels (`ln`, `tanh`, `softplus`) and `tanh` backward are completely division-free, non-FMA,
//! and bitwise identical to their scalar twins for every non-NaN input (and preserve the input NaN payload
//! on single-operand ops). `log_bwd` computes exact IEEE-754 `g / x` at 8-wide AVX2 width.

use super::simd::{
    LN2_HI, LN2_LO, define_bias_inplace, define_bwd_slice, define_unary_slice, exp_clamped_scalar,
    exp_neg_abs_scalar, maxps, minps, recip_1_plus_scalar,
};
#[cfg(target_arch = "x86_64")]
use super::simd::{exp_clamped8, exp_neg_abs8, has_avx2, recip_1_plus8};

// Range-reduction constants for `ln(x)`: reduce `m` into `[sqrt(2)/2, sqrt(2)]`.
const SQRT2_HALF_BITS: i32 = 0x3F35_04F3;
const LN_MAGIC: i32 = 0x3F80_0000 - SQRT2_HALF_BITS; // 0x004A_FB0D
const SUBNORMAL_SCALE: f32 = 16_777_216.0; // 2^24
const QNAN_BITS: u32 = 0x7FC0_0000;

// Degree-8 minimax polynomial for `(ln(1 + f) - f) / f^2` on `f ∈ [sqrt(2)/2 - 1, sqrt(2) - 1]`
// (max rel err `1.29e-8` across the whole reduced range).
const LN_C2: f32 = -4.999_999e-1;
const LN_C3: f32 = 3.333_333e-1;
const LN_C4: f32 = -2.500_127e-1;
const LN_C5: f32 = 2.000_130_1e-1;
const LN_C6: f32 = -1.661_659_3e-1;
const LN_C7: f32 = 1.420_033e-1;
const LN_C8: f32 = -1.319_644e-1;
const LN_C9: f32 = 1.276_082_8e-1;
const LN_C10: f32 = -7.559_620_6e-2;

// Cutoff and degree-4 minimax odd polynomial for `tanh(x) = x + x * x^2 * Q(x^2)` on `|x| < 0.625`
// (max rel err `1.15e-8`). Above `0.625`, `exp(-2|x|) <= 0.2865` avoids cancellation in `1 - 2*e/(1 + e)`.
const TANH_SMALL_CUTOFF: f32 = 0.625;
const TANH_C1: f32 = -3.333_329_3e-1;
const TANH_C2: f32 = 1.333_176e-1;
const TANH_C3: f32 = -5.376_721e-2;
const TANH_C4: f32 = 2.073_115_1e-2;
const TANH_C5: f32 = -5.810_007e-3;

// Degree-3 minimax polynomial in `z^2` for `ln(1 + e) = u + u * z^2 * R(z^2)` on `e ∈ [0, 1]`,
// where `u = 2e / (2 + e) = e * recip_1_plus(0.5 * e)` and `z = 0.5 * u` (max rel err `1.03e-8`).
const LN1P_C1: f32 = 3.333_322_4e-1;
const LN1P_C2: f32 = 2.000_965_5e-1;
const LN1P_C3: f32 = 1.402_164e-1;
const LN1P_C4: f32 = 1.389_356_6e-1;

// ---------------------------------------------------------------------------
// Non-FMA scalar twins (bitwise identical to the AVX2 kernels below)
// ---------------------------------------------------------------------------

/// Division-free scalar twin of `ln8`: bitwise identical for all inputs (including subnormals, `±0`, `±∞`,
/// negatives -> canonical quiet NaN, and NaN payload preservation).
#[inline(always)]
pub(crate) fn ln_scalar(x: f32) -> f32 {
    let is_sub = x < f32::MIN_POSITIVE;
    let x_norm = if is_sub { x * SUBNORMAL_SCALE } else { x };
    let exp_adj = if is_sub { -24i32 } else { 0i32 };
    let u = x_norm.to_bits() as i32;
    let u_adj = u.wrapping_add(LN_MAGIC);
    let k = ((u_adj >> 23).wrapping_sub(127)).wrapping_add(exp_adj);
    let m_bits = (u_adj & 0x007F_FFFF).wrapping_add(SQRT2_HALF_BITS);
    let m = f32::from_bits(m_bits as u32);
    let f = m - 1.0;
    let f2 = f * f;
    let mut p = LN_C10;
    p = p * f + LN_C9;
    p = p * f + LN_C8;
    p = p * f + LN_C7;
    p = p * f + LN_C6;
    p = p * f + LN_C5;
    p = p * f + LN_C4;
    p = p * f + LN_C3;
    p = p * f + LN_C2;
    let ln_m = f + f2 * p;
    let kf = k as f32;
    let mut res = (kf * LN2_LO + ln_m) + kf * LN2_HI;
    if x == 0.0 {
        res = f32::NEG_INFINITY;
    }
    if x < 0.0 {
        res = f32::from_bits(QNAN_BITS);
    }
    if x == f32::INFINITY {
        res = f32::INFINITY;
    }
    if x.is_nan() { x } else { res }
}

/// Exact IEEE-754 scalar twin of `log_bwd8`: `g / x`.
#[inline(always)]
pub(crate) fn log_bwd_scalar(x: f32, g: f32) -> f32 {
    g / x
}

/// Division-free scalar twin of `tanh8`: bitwise identical for all inputs (preserves `±0.0`, saturates to
/// `±1.0` for large `|x|` and `±∞`, preserves NaN payload).
#[inline(always)]
pub(crate) fn tanh_scalar(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    let sign = x.to_bits() & 0x8000_0000;
    let x2 = ax * ax;
    let mut q = TANH_C5;
    q = q * x2 + TANH_C4;
    q = q * x2 + TANH_C3;
    q = q * x2 + TANH_C2;
    q = q * x2 + TANH_C1;
    let poly = ax + ax * (x2 * q);
    let t = maxps(-2.0 * ax, -104.0);
    let e = exp_clamped_scalar(t);
    let rec = recip_1_plus_scalar(e);
    let large = 1.0 - 2.0 * (e * rec);
    let mag = if ax < TANH_SMALL_CUTOFF { poly } else { large };
    let signed = f32::from_bits(mag.to_bits() | sign);
    if x.is_nan() { x } else { signed }
}

/// Division-free scalar twin of `tanh_bwd8`'s derivative factor `sech^2(x) = 4e / (1 + e)^2`, avoiding the
/// catastrophic cancellation of `1 - tanh(x)^2` for large `|x|`. Clamped to `<= 1.0` so tiny `|x|` never
/// overshoot the mathematical upper bound of `1.0`.
#[inline(always)]
pub(crate) fn tanh_deriv_scalar(x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    let t = maxps(-2.0 * ax, -104.0);
    let e = exp_clamped_scalar(t);
    let rec = recip_1_plus_scalar(e);
    let ds = minps(((4.0 * e) * rec) * rec, 1.0);
    if x.is_nan() { x } else { ds }
}

/// Division-free scalar twin of `softplus8`: `max(x, 0) + ln(1 + exp(-|x|))`, accurate across all finite
/// inputs including deeply negative `x` (where `1 + e` would lose precision) and `±∞`, preserving NaN payload.
#[inline(always)]
pub(crate) fn softplus_scalar(x: f32) -> f32 {
    let e = exp_neg_abs_scalar(x);
    let r = recip_1_plus_scalar(0.5 * e);
    let u = e * r;
    let z = 0.5 * u;
    let z2 = z * z;
    let mut p = LN1P_C4;
    p = p * z2 + LN1P_C3;
    p = p * z2 + LN1P_C2;
    p = p * z2 + LN1P_C1;
    let ln1p_e = u + u * (z2 * p);
    let sp = maxps(x, 0.0) + ln1p_e;
    if x.is_nan() { x } else { sp }
}

// ---------------------------------------------------------------------------
// AVX2 8-lane kernels (non-FMA, bitwise identical to the scalar twins above)
// ---------------------------------------------------------------------------

/// `ln(x)` on eight lanes, division-free and bitwise identical to [`ln_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn ln8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let is_sub = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(f32::MIN_POSITIVE));
    let x_norm = _mm256_blendv_ps(x, _mm256_mul_ps(x, _mm256_set1_ps(SUBNORMAL_SCALE)), is_sub);
    let exp_adj = _mm256_and_si256(_mm256_castps_si256(is_sub), _mm256_set1_epi32(-24));
    let u = _mm256_castps_si256(x_norm);
    let u_adj = _mm256_add_epi32(u, _mm256_set1_epi32(LN_MAGIC));
    let k_raw = _mm256_sub_epi32(_mm256_srai_epi32::<23>(u_adj), _mm256_set1_epi32(127));
    let k = _mm256_add_epi32(k_raw, exp_adj);
    let m_bits = _mm256_add_epi32(
        _mm256_and_si256(u_adj, _mm256_set1_epi32(0x007F_FFFF)),
        _mm256_set1_epi32(SQRT2_HALF_BITS),
    );
    let m = _mm256_castsi256_ps(m_bits);
    let f = _mm256_sub_ps(m, _mm256_set1_ps(1.0));
    let f2 = _mm256_mul_ps(f, f);
    let mut p = _mm256_set1_ps(LN_C10);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C9));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C8));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C7));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C6));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C5));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C4));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C3));
    p = _mm256_add_ps(_mm256_mul_ps(p, f), _mm256_set1_ps(LN_C2));
    let ln_m = _mm256_add_ps(f, _mm256_mul_ps(f2, p));
    let kf = _mm256_cvtepi32_ps(k);
    let mut res = _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(kf, _mm256_set1_ps(LN2_LO)), ln_m),
        _mm256_mul_ps(kf, _mm256_set1_ps(LN2_HI)),
    );
    let zero = _mm256_setzero_ps();
    res = _mm256_blendv_ps(res, _mm256_set1_ps(f32::NEG_INFINITY), _mm256_cmp_ps::<_CMP_EQ_OQ>(x, zero));
    res = _mm256_blendv_ps(
        res,
        _mm256_set1_ps(f32::from_bits(QNAN_BITS)),
        _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero),
    );
    let pos_inf = _mm256_set1_ps(f32::INFINITY);
    res = _mm256_blendv_ps(res, pos_inf, _mm256_cmp_ps::<_CMP_EQ_OQ>(x, pos_inf));
    _mm256_blendv_ps(res, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// `tanh(x)` on eight lanes, division-free and bitwise identical to [`tanh_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn tanh8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let sign_bit = _mm256_set1_ps(-0.0);
    let ax = _mm256_andnot_ps(sign_bit, x);
    let sign = _mm256_and_ps(sign_bit, x);
    let x2 = _mm256_mul_ps(ax, ax);
    let mut q = _mm256_set1_ps(TANH_C5);
    q = _mm256_add_ps(_mm256_mul_ps(q, x2), _mm256_set1_ps(TANH_C4));
    q = _mm256_add_ps(_mm256_mul_ps(q, x2), _mm256_set1_ps(TANH_C3));
    q = _mm256_add_ps(_mm256_mul_ps(q, x2), _mm256_set1_ps(TANH_C2));
    q = _mm256_add_ps(_mm256_mul_ps(q, x2), _mm256_set1_ps(TANH_C1));
    let poly = _mm256_add_ps(ax, _mm256_mul_ps(ax, _mm256_mul_ps(x2, q)));
    let t = _mm256_max_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0), ax), _mm256_set1_ps(-104.0));
    let e = exp_clamped8(t);
    let rec = recip_1_plus8(e);
    let large = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(_mm256_set1_ps(2.0), _mm256_mul_ps(e, rec)));
    let is_small = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, _mm256_set1_ps(TANH_SMALL_CUTOFF));
    let mag = _mm256_blendv_ps(large, poly, is_small);
    let signed = _mm256_or_ps(mag, sign);
    _mm256_blendv_ps(signed, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// `g * sech^2(x)` on eight lanes, division-free and bitwise identical to `g * tanh_deriv_scalar(x)` on all
/// non-NaN inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn tanh_bwd8(x: std::arch::x86_64::__m256, g: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_max_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0), ax), _mm256_set1_ps(-104.0));
    let e = exp_clamped8(t);
    let rec = recip_1_plus8(e);
    let ds = _mm256_min_ps(
        _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(4.0), e), rec), rec),
        _mm256_set1_ps(1.0),
    );
    let ds = _mm256_blendv_ps(ds, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
    _mm256_mul_ps(g, ds)
}

/// `max(x, 0) + ln(1 + exp(-|x|))` on eight lanes, division-free and bitwise identical to [`softplus_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn softplus8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let half = _mm256_set1_ps(0.5);
    let e = exp_neg_abs8(x);
    let r = recip_1_plus8(_mm256_mul_ps(half, e));
    let u = _mm256_mul_ps(e, r);
    let z = _mm256_mul_ps(half, u);
    let z2 = _mm256_mul_ps(z, z);
    let mut p = _mm256_set1_ps(LN1P_C4);
    p = _mm256_add_ps(_mm256_mul_ps(p, z2), _mm256_set1_ps(LN1P_C3));
    p = _mm256_add_ps(_mm256_mul_ps(p, z2), _mm256_set1_ps(LN1P_C2));
    p = _mm256_add_ps(_mm256_mul_ps(p, z2), _mm256_set1_ps(LN1P_C1));
    let ln1p_e = _mm256_add_ps(u, _mm256_mul_ps(u, _mm256_mul_ps(z2, p)));
    let sp = _mm256_add_ps(_mm256_max_ps(x, _mm256_setzero_ps()), ln1p_e);
    _mm256_blendv_ps(sp, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// Exact IEEE-754 `g / x` on eight lanes for `Log` backward.
pub(crate) fn log_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), g.len());
    debug_assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { log_bwd_avx2(x, g, out) };
    }
    for ((o, &xv), &gv) in out.iter_mut().zip(x).zip(g) {
        *o = gv / xv;
    }
}

/// Exact IEEE-754 `g / x` with scalar broadcast `g` for `Log` backward.
pub(crate) fn log_bwd_broadcast_g(x: &[f32], g: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { log_bwd_bc_avx2(x, g, out) };
    }
    for (o, &xv) in out.iter_mut().zip(x) {
        *o = g / xv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn log_bwd_avx2(x: &[f32], g: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let (xp, xr) = x.as_chunks::<16>();
    let (gp, gr) = g.as_chunks::<16>();
    let (op, or) = out.as_chunks_mut::<16>();
    for ((xc, gc), oc) in xp.iter().zip(gp.iter()).zip(op.iter_mut()) {
        // SAFETY: `xc`, `gc`, `oc` are 16 floats; unaligned loads and stores.
        unsafe {
            let xp = xc.as_ptr();
            let gp = gc.as_ptr();
            let dp = oc.as_mut_ptr();
            let v0 = _mm256_div_ps(_mm256_loadu_ps(gp), _mm256_loadu_ps(xp));
            let v1 = _mm256_div_ps(_mm256_loadu_ps(gp.add(8)), _mm256_loadu_ps(xp.add(8)));
            _mm256_storeu_ps(dp, v0);
            _mm256_storeu_ps(dp.add(8), v1);
        }
    }
    let (x8, xt) = xr.as_chunks::<8>();
    let (g8, gt) = gr.as_chunks::<8>();
    let (o8, ot) = or.as_chunks_mut::<8>();
    for ((xc, gc), oc) in x8.iter().zip(g8.iter()).zip(o8.iter_mut()) {
        // SAFETY: `xc`, `gc`, `oc` are 8 floats; unaligned loads and stores.
        unsafe {
            let v = _mm256_div_ps(_mm256_loadu_ps(gc.as_ptr()), _mm256_loadu_ps(xc.as_ptr()));
            _mm256_storeu_ps(oc.as_mut_ptr(), v);
        }
    }
    for ((o, &xv), &gv) in ot.iter_mut().zip(xt).zip(gt) {
        *o = gv / xv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn log_bwd_bc_avx2(x: &[f32], g: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let gv = _mm256_set1_ps(g);
    let (xp, xr) = x.as_chunks::<16>();
    let (op, or) = out.as_chunks_mut::<16>();
    for (xc, oc) in xp.iter().zip(op.iter_mut()) {
        // SAFETY: `xc` and `oc` are 16 floats; unaligned loads and stores.
        unsafe {
            let xp = xc.as_ptr();
            let dp = oc.as_mut_ptr();
            let v0 = _mm256_div_ps(gv, _mm256_loadu_ps(xp));
            let v1 = _mm256_div_ps(gv, _mm256_loadu_ps(xp.add(8)));
            _mm256_storeu_ps(dp, v0);
            _mm256_storeu_ps(dp.add(8), v1);
        }
    }
    let (x8, xt) = xr.as_chunks::<8>();
    let (o8, ot) = or.as_chunks_mut::<8>();
    for (xc, oc) in x8.iter().zip(o8.iter_mut()) {
        // SAFETY: `xc` and `oc` are 8 floats; unaligned loads and stores.
        unsafe {
            let v = _mm256_div_ps(gv, _mm256_loadu_ps(xc.as_ptr()));
            _mm256_storeu_ps(oc.as_mut_ptr(), v);
        }
    }
    for (o, &xv) in ot.iter_mut().zip(xt) {
        *o = g / xv;
    }
}

define_unary_slice!(ln_slice, ln_avx2, ln8, ln_scalar);
define_unary_slice!(tanh_slice, tanh_avx2, tanh8, tanh_scalar);
define_unary_slice!(softplus_slice, softplus_avx2, softplus8, softplus_scalar);

define_bwd_slice!(
    tanh_bwd_slice,
    tanh_bwd_broadcast_g,
    tanh_bwd_avx2,
    tanh_bwd_bc_avx2,
    tanh_bwd8,
    tanh_deriv_scalar
);

define_bias_inplace!(tanh_bias_inplace, tanh_bias_avx2, has_avx2, "avx2", tanh8, tanh_scalar);
