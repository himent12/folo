//! Bit-exact division-free minimax fast SiLU (`silu_q8`) for the CPU int8 epilogue.
//!
//! Computes `x / (1 + exp(-x))` using:
//! 1. Base-2 range reduction `u = t · log2(e)`, `n = round_ties_even(u)`, `f = u - n` (`f ∈ [-0.5, 0.5]`),
//!    with `t = clamp(-x, -86, 86)` so `2^n` and the reciprocal seed stay strictly normal floats.
//! 2. Economised degree-5 minimax polynomial for `2^f` on `[-0.5, 0.5]` (relative error `< 1e-7`, exact `2^0 = 1`).
//! 3. Deterministic IEEE-754 bit-hack reciprocal seed `0x7EF3_11C3 - bits(1 + e)` followed by 3 exact FMA
//!    Newton-Raphson steps (`r ← r · (2 - d · r)`), achieving `< 2e-7` hybrid error (`|err| / max(1, |ref|)`,
//!    max true relative error `~2.4e-6` near `-79`) with ZERO `vdivps` and ZERO platform-divergent `rcpps`.
//!
//! Scalar and AVX2 paths execute the exact same IEEE-754 FMA/mul/add/sub/integer-sub operations in the same
//! order, so they produce bit-identical results across CPUs.

/// `max(a, b)` as `vmaxps` computes it (`b` when either is NaN).
#[inline(always)]
pub(super) fn maxps(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `min(a, b)` as `vminps` computes it (`b` when either is NaN).
#[inline(always)]
pub(super) fn minps(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

// Chebyshev-economised degree-5 minimax coefficients for 2^f on f ∈ [-0.5, 0.5].
const C5: f32 = 1.340_029e-3;
const C4: f32 = 9.675_892e-3;
const C3: f32 = 5.550_327_5e-2;
const C2: f32 = 2.402_210_9e-1;
const C1: f32 = 6.931_472e-1;
const C0: f32 = 1.0;

/// Bit-hack reciprocal seed constant (`~2 * 127 << 23` tuned for minimax relative error `< 0.0385` on normal `d`).
const RECIP_MAGIC: u32 = 0x7EF3_11C3;

/// The epilogue's SiLU, `x / (1 + e^{−x})`, division-free and bit-exact between scalar and AVX2 (`|err| / max(1, |ref|) < 2e-7`
/// on `[-86, 86]`, max true relative error `~2.4e-6` in the far negative tail, finite at extremes, `silu_q8(0.0) == 0.0`, NaN stays NaN).
#[inline]
pub fn silu_q8(x: f32) -> f32 {
    let x = maxps(-88.0, x);
    let neg = f32::from_bits(x.to_bits() ^ 0x8000_0000);
    let t = maxps(minps(neg, 86.0), -86.0);
    let u = t * std::f32::consts::LOG2_E;
    let n = u.round_ties_even();
    let f = u - n;
    let mut p = C5;
    p = p.mul_add(f, C4);
    p = p.mul_add(f, C3);
    p = p.mul_add(f, C2);
    p = p.mul_add(f, C1);
    p = p.mul_add(f, C0);
    let e = p * f32::from_bits(((n as i32).wrapping_add(127) as u32) << 23);
    let d = 1.0 + e;
    let mut r = f32::from_bits(RECIP_MAGIC.wrapping_sub(d.to_bits()));
    r = r.mul_add((-d).mul_add(r, 1.0), r);
    r = r.mul_add((-d).mul_add(r, 1.0), r);
    r = r.mul_add((-d).mul_add(r, 1.0), r);
    x * r
}

/// [`silu_q8`] on 8 lanes, bit-identical to scalar [`silu_q8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(_mm256_set1_ps(-88.0), x);
    let neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
    let t = _mm256_max_ps(_mm256_min_ps(neg, _mm256_set1_ps(86.0)), _mm256_set1_ps(-86.0));
    let u = _mm256_mul_ps(t, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(u);
    let f = _mm256_sub_ps(u, n);
    let mut p = _mm256_set1_ps(C5);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C4));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C3));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C2));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C1));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C0));
    let s = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
    let e = _mm256_mul_ps(p, _mm256_castsi256_ps(s));
    let one = _mm256_set1_ps(1.0);
    let d = _mm256_add_ps(one, e);
    let mut r =
        _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(RECIP_MAGIC as i32), _mm256_castps_si256(d)));
    r = _mm256_fmadd_ps(r, _mm256_fnmadd_ps(d, r, one), r);
    r = _mm256_fmadd_ps(r, _mm256_fnmadd_ps(d, r, one), r);
    r = _mm256_fmadd_ps(r, _mm256_fnmadd_ps(d, r, one), r);
    _mm256_mul_ps(x, r)
}
