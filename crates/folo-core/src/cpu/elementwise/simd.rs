//! AVX2 vectorised transcendental kernels and their scalar twins.
//!
//! Every kernel here has an 8-lane AVX2 implementation and a scalar twin that produces bitwise identical
//! results for all non-NaN inputs (including `±0.0`, subnormals, and `±∞`), while any NaN input always
//! yields NaN (and single-operand `exp`/`sigmoid` preserve the input NaN payload). Results therefore depend
//! only on each element's value — never on slice boundaries, rayon thread counts, or whether an operand is
//! strided or contiguous.
//!
//! The standalone elementwise / autograd kernels (`exp`, `sigmoid`, `silu`, and their backwards) use
//! non-FMA IEEE-754 `*`/`+`/`-` so their scalar twins (`folo_core::cpu::{silu, silu_backward}` and strided
//! fallbacks) never emit libm `fmaf` calls and auto-vectorize cleanly inside `avx2`-only callers such as
//! `folo-nn`'s fused `BatchNorm2d`. The fused conv/GEMM epilogue (`silu8` / `silu_bias_inplace`) retains
//! its FMA formulation.

const LOG2E: f32 = std::f32::consts::LOG2_E;
// Cody-Waite split of ln(2) = LN2_HI + LN2_LO (exact high bits with 12 trailing zero mantissa bits + residual,
// so `n * LN2_HI` is exact in f32 for all `|n| <= 150` even without FMA).
pub(super) const LN2_HI: f32 = 0.693_359_4;
pub(super) const LN2_LO: f32 = -2.121_944_4e-4;

// Degree-5 minimax coefficients for `(exp(r) - 1) / r` on `r ∈ [-ln(2)/2, ln(2)/2]` (max rel err `9.54e-8`).
const EXP5_C5: f32 = 8.290_312e-3;
const EXP5_C4: f32 = 4.189_795_3e-2;
const EXP5_C3: f32 = 1.666_763_6e-1;
const EXP5_C2: f32 = 4.999_915e-1;
const EXP5_C1: f32 = 9.999_997e-1;

/// `1.5 * 2^23` (`0x4B40_0000`): adding `u ∈ [-2^22, 2^22)` rounds to nearest-even integer in the low
/// mantissa bits without `vroundps` or `vcvtps2dq`.
const EXP_MAGIC_0: f32 = 12_582_912.0;
const EXP_MAGIC_0_BITS: i32 = 0x4B40_0000;

/// Minimax IEEE-754 reciprocal seed magic (`max |1 - d * r0| ≈ 0.05051`, exact `r(1.0) == 1.0`, `r(2.0) == 0.5`).
const RECIP_MAGIC: u32 = 0x7EF3_11C3;

/// Bound on the relative error of the vectorised SiLU (`|y - silu(x)| <= SILU_MAX_REL_ERR * |silu(x)|`, plus
/// the subnormal spacing once `silu(x)` underflows), checked by `silu_avx2_error_bound`: about 4 ulp.
pub const SILU_MAX_REL_ERR: f64 = 5e-7;

/// `max(a, b)` with x86 `vmaxps` NaN semantics (`b` when either is NaN), for bit-exact scalar == AVX2.
#[inline(always)]
pub(super) fn maxps(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `min(a, b)` with x86 `vminps` NaN semantics (`b` when either is NaN), for bit-exact scalar == AVX2.
#[inline(always)]
pub(super) fn minps(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

// ---------------------------------------------------------------------------
// Non-FMA scalar twins (zero `fmaf` calls; bitwise identical to AVX2 below)
// ---------------------------------------------------------------------------

/// Core non-FMA `exp(xc)` for `xc` already clamped into `[-104.0, 89.0]` (never NaN).
#[inline(always)]
pub(super) fn exp_clamped_scalar(xc: f32) -> f32 {
    let z = xc * LOG2E + EXP_MAGIC_0;
    let n = z - EXP_MAGIC_0;
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    let mut p = EXP5_C5;
    p = p * r + EXP5_C4;
    p = p * r + EXP5_C3;
    p = p * r + EXP5_C2;
    p = p * r + EXP5_C1;
    let y = p * r + 1.0;
    let ni = (z.to_bits() as i32).wrapping_sub(EXP_MAGIC_0_BITS);
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    (y * s1) * s2
}

/// Non-FMA `exp(-|x|)` clamped at `-104.0` (NaN maps to `-104.0`; callers handle NaN propagation).
#[inline(always)]
pub(super) fn exp_neg_abs_scalar(x: f32) -> f32 {
    let t_abs = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    let t = maxps(0.0 - t_abs, -104.0);
    exp_clamped_scalar(t)
}

/// Division-free non-FMA `1 / (1 + e)` for `e ∈ [0, 1]` via bit-hack seed + Householder-6
/// `(1 + e0)(1 + e0^2 + e0^4)` (error `e0^6 <= 1.66e-8`, exact at `e = 0` and `e = 1`).
#[inline(always)]
pub(super) fn recip_1_plus_scalar(e: f32) -> f32 {
    let d = 1.0 + e;
    let r0 = f32::from_bits(RECIP_MAGIC.wrapping_sub(d.to_bits()));
    let e0 = 1.0 - d * r0;
    let u = r0 + r0 * e0;
    let e2 = e0 * e0;
    let t2 = e2 + e2 * e2;
    u + u * t2
}

/// Scalar twin of `exp8` (bitwise identical for all non-NaN inputs; preserves NaN payload).
#[inline(always)]
pub(crate) fn exp_scalar(x: f32) -> f32 {
    let xc = minps(maxps(x, -104.0), 89.0);
    let v = exp_clamped_scalar(xc);
    if x.is_nan() { x } else { v }
}

/// Scalar twin of `sigmoid8` (bitwise identical for all non-NaN inputs; preserves NaN payload).
#[inline(always)]
pub(crate) fn sigmoid_scalar(x: f32) -> f32 {
    let e = exp_neg_abs_scalar(x);
    let rec = recip_1_plus_scalar(e);
    let sig = if x >= 0.0 { rec } else { e * rec };
    if x.is_nan() { x } else { sig }
}

/// Division-free non-FMA scalar twin of `silu_op8` (auto-vectorizes in `avx2`-only loops; bitwise identical
/// to `Tensor::silu` for all non-NaN inputs; NaN stays NaN).
#[inline(always)]
pub(crate) fn silu_scalar(x: f32) -> f32 {
    let e = exp_neg_abs_scalar(x);
    let rec = recip_1_plus_scalar(e);
    let sig = if x >= 0.0 { rec } else { e * rec };
    x * sig
}

/// Derivative factor `sigmoid'(x) = s * (1 - s)` written stably as `e * rec * rec` (non-FMA scalar twin).
#[inline(always)]
pub(crate) fn sigmoid_deriv_scalar(x: f32) -> f32 {
    let e = exp_neg_abs_scalar(x);
    let rec = recip_1_plus_scalar(e);
    let ds = (e * rec) * rec;
    if x.is_nan() { x } else { ds }
}

/// Derivative factor `silu'(x) = s + x * s * (1 - s)` with `s * (1 - s) = e * rec * rec` (non-FMA scalar twin).
#[inline(always)]
pub(crate) fn silu_deriv_scalar(x: f32) -> f32 {
    let e = exp_neg_abs_scalar(x);
    let rec = recip_1_plus_scalar(e);
    let er = e * rec;
    let sig = if x >= 0.0 { rec } else { er };
    sig + x * (er * rec)
}

/// Scalar twin of `silu_bwd8`: `g * silu'(x)`.
#[inline(always)]
pub(crate) fn silu_bwd_scalar(x: f32, g: f32) -> f32 {
    g * silu_deriv_scalar(x)
}

/// FMA scalar twin of the fused conv/GEMM epilogue `silu8`: `e = exp(-|x|)` via Cody-Waite reduction and
/// degree-5 minimax polynomial (FMA), `1 / (1 + e)` via bit-hack seed + 4-FMA Householder-6, `sigmoid = r`
/// for `x >= 0` else `e * r`. Bitwise identical to `silu8` for all non-NaN inputs; NaN stays NaN.
#[inline]
pub(crate) fn silu_epilogue_scalar(x: f32) -> f32 {
    let t_abs = f32::from_bits(x.to_bits() & 0x7FFF_FFFF);
    let t = maxps(0.0 - t_abs, -104.0);
    let z = t.mul_add(LOG2E, EXP_MAGIC_0);
    let n = z - EXP_MAGIC_0;
    let r = (-LN2_HI).mul_add(n, t);
    let r = (-LN2_LO).mul_add(n, r);
    let mut p = EXP5_C5;
    p = p.mul_add(r, EXP5_C4);
    p = p.mul_add(r, EXP5_C3);
    p = p.mul_add(r, EXP5_C2);
    p = p.mul_add(r, EXP5_C1);
    let y = p.mul_add(r, 1.0);
    let ni = (z.to_bits() as i32).wrapping_sub(EXP_MAGIC_0_BITS);
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    let e = (y * s1) * s2;
    let d = 1.0 + e;
    let r0 = f32::from_bits(RECIP_MAGIC.wrapping_sub(d.to_bits()));
    let e0 = (-d).mul_add(r0, 1.0);
    let u = r0.mul_add(e0, r0);
    let e2 = e0 * e0;
    let t2 = e2.mul_add(e2, e2);
    let rec = u.mul_add(t2, u);
    let sig = if x >= 0.0 { rec } else { e * rec };
    x * sig
}

// ---------------------------------------------------------------------------
// AVX2 8-lane kernels (non-FMA for standalone ops; FMA for conv/GEMM `silu8`)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Core non-FMA `exp(xc)` on eight lanes for `xc` already clamped into `[-104.0, 89.0]` (never NaN).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn exp_clamped8(xc: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let magic0 = _mm256_set1_ps(EXP_MAGIC_0);
    let z = _mm256_add_ps(_mm256_mul_ps(xc, _mm256_set1_ps(LOG2E)), magic0);
    let n = _mm256_sub_ps(z, magic0);
    let r = _mm256_sub_ps(xc, _mm256_mul_ps(n, _mm256_set1_ps(LN2_HI)));
    let r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(LN2_LO)));
    let mut p = _mm256_set1_ps(EXP5_C5);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C4));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C2));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C1));
    let y = _mm256_add_ps(_mm256_mul_ps(p, r), one);
    // 2^n as two normal powers of two (n is in -150..=128), so huge results overflow to +inf and tiny ones
    // underflow gradually instead of wrapping the exponent field.
    let ni = _mm256_sub_epi32(_mm256_castps_si256(z), _mm256_set1_epi32(EXP_MAGIC_0_BITS));
    let n1 = _mm256_srai_epi32::<1>(ni);
    let n2 = _mm256_sub_epi32(ni, n1);
    let bias127 = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias127)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias127)));
    _mm256_mul_ps(_mm256_mul_ps(y, s1), s2)
}

/// Non-FMA `exp(-|x|)` on eight lanes, clamped to `-104.0` (NaN maps to `-104.0`; callers propagate NaN).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn exp_neg_abs8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let t_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_max_ps(_mm256_sub_ps(zero, t_abs), _mm256_set1_ps(-104.0));
    exp_clamped8(t)
}

/// Division-free non-FMA `1 / (1 + e)` on eight lanes for `e ∈ [0, 1]`, via bit-hack seed + Householder-6.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn recip_1_plus8(e: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let d = _mm256_add_ps(one, e);
    let r0 =
        _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(RECIP_MAGIC as i32), _mm256_castps_si256(d)));
    let e0 = _mm256_sub_ps(one, _mm256_mul_ps(d, r0));
    let u = _mm256_add_ps(r0, _mm256_mul_ps(r0, e0));
    let e2 = _mm256_mul_ps(e0, e0);
    let t2 = _mm256_add_ps(e2, _mm256_mul_ps(e2, e2));
    _mm256_add_ps(u, _mm256_mul_ps(u, t2))
}

/// `exp(x)` on eight lanes, bitwise identical to [`exp_scalar`]: Cody-Waite reduction, degree-5 minimax
/// polynomial, gradual subnormal underflow and `+inf` overflow, NaN payload preserved.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-104.0)), _mm256_set1_ps(89.0));
    let v = exp_clamped8(xc);
    _mm256_blendv_ps(v, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// `1 / (1 + exp(-x))` on eight lanes, division-free and bitwise identical to [`sigmoid_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn sigmoid8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp_neg_abs8(x);
    let rec = recip_1_plus8(e);
    let zero = _mm256_setzero_ps();
    let sig = _mm256_blendv_ps(_mm256_mul_ps(e, rec), rec, _mm256_cmp_ps::<_CMP_GE_OQ>(x, zero));
    _mm256_blendv_ps(sig, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// `x * sigmoid(x)` on eight lanes (non-FMA), division-free and bitwise identical to [`silu_scalar`] on all
/// non-NaN inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn silu_op8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp_neg_abs8(x);
    let rec = recip_1_plus8(e);
    let zero = _mm256_setzero_ps();
    let sig = _mm256_blendv_ps(_mm256_mul_ps(e, rec), rec, _mm256_cmp_ps::<_CMP_GE_OQ>(x, zero));
    _mm256_mul_ps(x, sig)
}

/// `g * sigmoid'(x)` on eight lanes, division-free and bitwise identical to `g * sigmoid_deriv_scalar(x)` on
/// all non-NaN inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn sigmoid_bwd8(x: std::arch::x86_64::__m256, g: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp_neg_abs8(x);
    let rec = recip_1_plus8(e);
    let ds = _mm256_mul_ps(_mm256_mul_ps(e, rec), rec);
    let ds = _mm256_blendv_ps(ds, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
    _mm256_mul_ps(g, ds)
}

/// `g * silu'(x)` on eight lanes, division-free and bitwise identical to [`silu_bwd_scalar`] on all non-NaN
/// inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn silu_bwd8(x: std::arch::x86_64::__m256, g: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let e = exp_neg_abs8(x);
    let rec = recip_1_plus8(e);
    let zero = _mm256_setzero_ps();
    let er = _mm256_mul_ps(e, rec);
    let sig = _mm256_blendv_ps(er, rec, _mm256_cmp_ps::<_CMP_GE_OQ>(x, zero));
    let ds = _mm256_add_ps(sig, _mm256_mul_ps(x, _mm256_mul_ps(er, rec)));
    _mm256_mul_ps(g, ds)
}

/// `g * exp(x)` on eight lanes, bitwise identical to `g * exp_scalar(x)` on all non-NaN inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn exp_bwd8(x: std::arch::x86_64::__m256, g: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    std::arch::x86_64::_mm256_mul_ps(g, exp8(x))
}

/// `g * sigmoid(x)` on eight lanes, bitwise identical to `g * sigmoid_scalar(x)` on all non-NaN inputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn softplus_bwd8(x: std::arch::x86_64::__m256, g: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    std::arch::x86_64::_mm256_mul_ps(g, sigmoid8(x))
}

/// Fused conv/GEMM epilogue `x * sigmoid(x)` on eight lanes (AVX2 + FMA), division-free and bitwise
/// identical to [`silu_epilogue_scalar`] on all non-NaN inputs: `e = exp(-|x|)` by Cody-Waite reduction and
/// degree-5 minimax polynomial (FMA), `1 / (1 + e)` by bit-hack seed plus 4-FMA Householder-6 refinement
/// `(1 + e0)(1 + e0^2 + e0^4)` (zero `vdivps`, zero `vroundps`, zero `vcvtps2dq`), `sigmoid = r` for
/// `x >= 0` and `e * r` below. NaN stays NaN.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let magic0 = _mm256_set1_ps(EXP_MAGIC_0);
    let t_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_max_ps(_mm256_sub_ps(zero, t_abs), _mm256_set1_ps(-104.0));
    let z = _mm256_fmadd_ps(t, _mm256_set1_ps(LOG2E), magic0);
    let n = _mm256_sub_ps(z, magic0);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), t);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
    let mut p = _mm256_set1_ps(EXP5_C5);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP5_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP5_C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP5_C2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(EXP5_C1));
    let y = _mm256_fmadd_ps(p, r, one);
    let ni = _mm256_sub_epi32(_mm256_castps_si256(z), _mm256_set1_epi32(EXP_MAGIC_0_BITS));
    let n1 = _mm256_srai_epi32::<1>(ni);
    let n2 = _mm256_sub_epi32(ni, n1);
    let bias127 = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias127)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias127)));
    let e = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);
    let d = _mm256_add_ps(one, e);
    let r0 =
        _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(RECIP_MAGIC as i32), _mm256_castps_si256(d)));
    let e0 = _mm256_fnmadd_ps(d, r0, one);
    let u = _mm256_fmadd_ps(r0, e0, r0);
    let e2 = _mm256_mul_ps(e0, e0);
    let t2 = _mm256_fmadd_ps(e2, e2, e2);
    let rec = _mm256_fmadd_ps(u, t2, u);
    let sig = _mm256_blendv_ps(_mm256_mul_ps(e, rec), rec, _mm256_cmp_ps::<_CMP_GE_OQ>(x, zero));
    _mm256_mul_ps(x, sig)
}

/// Generate a 2x8-unrolled + 8-wide + padded-tail AVX2 unary slice loop and its runtime-dispatched wrapper.
macro_rules! define_unary_slice {
    ($pub_fn:ident, $avx_fn:ident, $vec8:ident, $scalar:ident) => {
        pub(crate) fn $pub_fn(x: &[f32], out: &mut [f32]) {
            debug_assert_eq!(x.len(), out.len());
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                // SAFETY: AVX2 was detected above.
                return unsafe { $avx_fn(x, out) };
            }
            for (o, &v) in out.iter_mut().zip(x) {
                *o = $scalar(v);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        fn $avx_fn(x: &[f32], out: &mut [f32]) {
            use std::arch::x86_64::*;
            let (xp, xr) = x.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (xc, oc) in xp.iter().zip(op.iter_mut()) {
                // SAFETY: `xc` and `oc` are 16 floats; unaligned loads and stores.
                unsafe {
                    let sp = xc.as_ptr();
                    let dp = oc.as_mut_ptr();
                    let v0 = $vec8(_mm256_loadu_ps(sp));
                    let v1 = $vec8(_mm256_loadu_ps(sp.add(8)));
                    _mm256_storeu_ps(dp, v0);
                    _mm256_storeu_ps(dp.add(8), v1);
                }
            }
            let (x8, xt) = xr.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (xc, oc) in x8.iter().zip(o8.iter_mut()) {
                // SAFETY: `xc` and `oc` are 8 floats; unaligned loads and stores.
                unsafe {
                    let v = $vec8(_mm256_loadu_ps(xc.as_ptr()));
                    _mm256_storeu_ps(oc.as_mut_ptr(), v);
                }
            }
            if !xt.is_empty() {
                let mut pad = [0f32; 8];
                pad[..xt.len()].copy_from_slice(xt);
                // SAFETY: `pad` is 8 floats; load completes before taking the mutable store pointer.
                unsafe {
                    let v = $vec8(_mm256_loadu_ps(pad.as_ptr()));
                    _mm256_storeu_ps(pad.as_mut_ptr(), v);
                }
                ot.copy_from_slice(&pad[..xt.len()]);
            }
        }
    };
}
pub(super) use define_unary_slice;

/// Generate a 2x8-unrolled + 8-wide + padded-tail AVX2 backward slice loop (both `(x, g)` and `(x, broadcast g)`).
macro_rules! define_bwd_slice {
    ($pub_fn:ident, $pub_bc_fn:ident, $avx_fn:ident, $avx_bc_fn:ident, $vec8:ident, $deriv_scalar:ident) => {
        pub(crate) fn $pub_fn(x: &[f32], g: &[f32], out: &mut [f32]) {
            debug_assert_eq!(x.len(), g.len());
            debug_assert_eq!(x.len(), out.len());
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                // SAFETY: AVX2 was detected above.
                return unsafe { $avx_fn(x, g, out) };
            }
            for ((o, &xv), &gv) in out.iter_mut().zip(x).zip(g) {
                *o = gv * $deriv_scalar(xv);
            }
        }

        pub(crate) fn $pub_bc_fn(x: &[f32], g: f32, out: &mut [f32]) {
            debug_assert_eq!(x.len(), out.len());
            #[cfg(target_arch = "x86_64")]
            if has_avx2() {
                // SAFETY: AVX2 was detected above.
                return unsafe { $avx_bc_fn(x, g, out) };
            }
            for (o, &xv) in out.iter_mut().zip(x) {
                *o = g * $deriv_scalar(xv);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        fn $avx_fn(x: &[f32], g: &[f32], out: &mut [f32]) {
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
                    let v0 = $vec8(_mm256_loadu_ps(xp), _mm256_loadu_ps(gp));
                    let v1 = $vec8(_mm256_loadu_ps(xp.add(8)), _mm256_loadu_ps(gp.add(8)));
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
                    let v = $vec8(_mm256_loadu_ps(xc.as_ptr()), _mm256_loadu_ps(gc.as_ptr()));
                    _mm256_storeu_ps(oc.as_mut_ptr(), v);
                }
            }
            if !xt.is_empty() {
                let mut xpad = [0f32; 8];
                let mut gpad = [0f32; 8];
                xpad[..xt.len()].copy_from_slice(xt);
                gpad[..gt.len()].copy_from_slice(gt);
                // SAFETY: `xpad` and `gpad` are 8 floats; load completes before taking the store pointer.
                unsafe {
                    let v = $vec8(_mm256_loadu_ps(xpad.as_ptr()), _mm256_loadu_ps(gpad.as_ptr()));
                    _mm256_storeu_ps(xpad.as_mut_ptr(), v);
                }
                ot.copy_from_slice(&xpad[..xt.len()]);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        fn $avx_bc_fn(x: &[f32], g: f32, out: &mut [f32]) {
            use std::arch::x86_64::*;
            let gv = _mm256_set1_ps(g);
            let (xp, xr) = x.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (xc, oc) in xp.iter().zip(op.iter_mut()) {
                // SAFETY: `xc` and `oc` are 16 floats; unaligned loads and stores.
                unsafe {
                    let xp = xc.as_ptr();
                    let dp = oc.as_mut_ptr();
                    let v0 = $vec8(_mm256_loadu_ps(xp), gv);
                    let v1 = $vec8(_mm256_loadu_ps(xp.add(8)), gv);
                    _mm256_storeu_ps(dp, v0);
                    _mm256_storeu_ps(dp.add(8), v1);
                }
            }
            let (x8, xt) = xr.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (xc, oc) in x8.iter().zip(o8.iter_mut()) {
                // SAFETY: `xc` and `oc` are 8 floats; unaligned loads and stores.
                unsafe {
                    let v = $vec8(_mm256_loadu_ps(xc.as_ptr()), gv);
                    _mm256_storeu_ps(oc.as_mut_ptr(), v);
                }
            }
            if !xt.is_empty() {
                let mut xpad = [0f32; 8];
                xpad[..xt.len()].copy_from_slice(xt);
                // SAFETY: `xpad` is 8 floats; load completes before taking the mutable store pointer.
                unsafe {
                    let v = $vec8(_mm256_loadu_ps(xpad.as_ptr()), gv);
                    _mm256_storeu_ps(xpad.as_mut_ptr(), v);
                }
                ot.copy_from_slice(&xpad[..xt.len()]);
            }
        }
    };
}
pub(super) use define_bwd_slice;

/// Generate a 2x8-unrolled + 8-wide + padded-tail in-place `v[i] = op(v[i] + bias)` epilogue loop.
macro_rules! define_bias_inplace {
    ($pub_fn:ident, $avx_fn:ident, $detect:ident, $feat:literal, $vec8:ident, $scalar:ident) => {
        pub(crate) fn $pub_fn(v: &mut [f32], bias: f32) {
            #[cfg(target_arch = "x86_64")]
            if $detect() {
                // SAFETY: required CPU features were detected above.
                return unsafe { $avx_fn(v, bias) };
            }
            for x in v.iter_mut() {
                *x = $scalar(*x + bias);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feat)]
        fn $avx_fn(v: &mut [f32], bias: f32) {
            use std::arch::x86_64::*;
            let b = _mm256_set1_ps(bias);
            let (pairs, rem) = v.as_chunks_mut::<16>();
            for c in pairs {
                // SAFETY: `c` is 16 floats; unaligned loads and stores.
                unsafe {
                    let p = c.as_mut_ptr();
                    let x0 = _mm256_add_ps(_mm256_loadu_ps(p), b);
                    let x1 = _mm256_add_ps(_mm256_loadu_ps(p.add(8)), b);
                    _mm256_storeu_ps(p, $vec8(x0));
                    _mm256_storeu_ps(p.add(8), $vec8(x1));
                }
            }
            let (chunks, tail) = rem.as_chunks_mut::<8>();
            for c in chunks {
                // SAFETY: `c` is 8 floats; unaligned loads and stores.
                unsafe {
                    let x = _mm256_add_ps(_mm256_loadu_ps(c.as_ptr()), b);
                    _mm256_storeu_ps(c.as_mut_ptr(), $vec8(x));
                }
            }
            if !tail.is_empty() {
                // Padded lanes keep results independent of segment boundaries (GEMM tiles move with threads).
                let mut pad = [0f32; 8];
                pad[..tail.len()].copy_from_slice(tail);
                // SAFETY: `pad` is 8 floats; load completes before taking the mutable store pointer.
                unsafe {
                    let x = _mm256_add_ps(_mm256_loadu_ps(pad.as_ptr()), b);
                    _mm256_storeu_ps(pad.as_mut_ptr(), $vec8(x));
                }
                let n = tail.len();
                tail.copy_from_slice(&pad[..n]);
            }
        }
    };
}
pub(super) use define_bias_inplace;

define_unary_slice!(exp_slice, exp_avx2, exp8, exp_scalar);
define_unary_slice!(sigmoid_slice, sigmoid_avx2, sigmoid8, sigmoid_scalar);
define_unary_slice!(silu_slice, silu_avx2, silu_op8, silu_scalar);

define_bwd_slice!(exp_bwd_slice, exp_bwd_broadcast_g, exp_bwd_avx2, exp_bwd_bc_avx2, exp_bwd8, exp_scalar);
define_bwd_slice!(
    sigmoid_bwd_slice,
    sigmoid_bwd_broadcast_g,
    sigmoid_bwd_avx2,
    sigmoid_bwd_bc_avx2,
    sigmoid_bwd8,
    sigmoid_deriv_scalar
);
define_bwd_slice!(
    silu_bwd_slice,
    silu_bwd_broadcast_g,
    silu_bwd_avx2,
    silu_bwd_bc_avx2,
    silu_bwd8,
    silu_deriv_scalar
);
define_bwd_slice!(
    softplus_bwd_slice,
    softplus_bwd_broadcast_g,
    softplus_bwd_avx2,
    softplus_bwd_bc_avx2,
    softplus_bwd8,
    sigmoid_scalar
);

define_bias_inplace!(
    silu_bias_inplace,
    silu_bias_avx2,
    has_avx2_fma,
    "avx2,fma",
    silu8,
    silu_epilogue_scalar
);
define_bias_inplace!(sigmoid_bias_inplace, sigmoid_bias_avx2, has_avx2, "avx2", sigmoid8, sigmoid_scalar);
