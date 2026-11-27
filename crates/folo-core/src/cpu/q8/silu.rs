//! Bit-exact division-free minimax fast SiLU (`silu_q8`) for the CPU int8 epilogue.
//!
//! Computes `x / (1 + exp(-x))` using:
//! 1. Fused base-2 range reduction `u = clamp(x · (-log2(e)), -124, 124)`, `z = u + (1.5·2^23 + 127)` so
//!    `n = z - magic` rounds to nearest-even and `bits(z) << 23` yields `2^n` in a single shift (zero `vroundps`,
//!    zero `vcvtps2dq`, zero `vpaddd`).
//! 2. Economised degree-5 minimax polynomial for `2^f` on `f ∈ [-0.5, 0.5]` evaluated in Estrin form (3-stage
//!    parallel FMA tree, 40% shorter critical path than Horner, relative error `< 2e-7`, `2^0 = 1`).
//! 3. Deterministic IEEE-754 bit-hack reciprocal seed `0x7EF3_11C3 - bits(1 + e)` followed by a 4-FMA factored
//!    Householder-6 step `r = r0 · (1 + e0) · (1 + e0^2 + e0^4)` (`e0 = 1 - d·r0`, truncation error `e0^6 <= 1.66e-8`,
//!    exact `r(1.0) == 1.0` and `r(2.0) == 0.5`, 33% lower latency than 3 Newton-Raphson steps), achieving
//!    `< 2.5e-7` hybrid error (`|err| / max(1, |ref|)`) with ZERO `vdivps` and ZERO platform-divergent `rcpps`.
//!
//! Scalar and AVX2 paths execute the exact same IEEE-754 FMA/mul/add/sub/integer operations in the same order,
//! producing bit-identical results on all non-NaN inputs across CPUs (and preserving NaN classification on NaN inputs).

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

const NEG_LOG2_E: f32 = -std::f32::consts::LOG2_E;
/// `1.5 * 2^23 + 127` (`0x4B40_007F`): adding `u ∈ [-124, 124]` puts `n + 127 ∈ [3, 251]` in bits 0..8, so
/// `bits(z) << 23` is IEEE-754 `2^n` directly.
const MAGIC_EXP: f32 = 12_583_039.0;
/// Bit-hack reciprocal seed constant (`max |1 - d·r0| ≈ 0.05051`, exact `r(1.0) == 1.0` & `r(2.0) == 0.5`).
const RECIP_MAGIC: u32 = 0x7EF3_11C3;

/// The epilogue's SiLU, `x / (1 + e^{−x})`, division-free and bit-exact between scalar and AVX2 on all non-NaN
/// inputs (`|err| / max(1, |ref|) < 2.5e-7` on `[-86, 86]`, finite at extremes, `silu_q8(0.0) == 0.0`,
/// `silu_q8(-0.0) == -0.0`, `silu_q8(+∞) == +∞`, NaN stays NaN).
#[inline]
pub fn silu_q8(x: f32) -> f32 {
    // Clamping `x >= -85.0` bounds `x * (-log2(e)) <= 122.63 < 124.0` automatically, eliminating an upper clamp
    // on `u` while keeping `|silu(-85)| ≈ 1.03e-35 << 2.5e-7`.
    let x = maxps(-85.0, x);
    let u = maxps(x * NEG_LOG2_E, -124.0);
    let z = u + MAGIC_EXP;
    let n = z - MAGIC_EXP;
    let f = u - n;
    let f2 = f * f;
    let p01 = C1.mul_add(f, C0);
    let p23 = C3.mul_add(f, C2);
    let p45 = C5.mul_add(f, C4);
    let p2345 = p45.mul_add(f2, p23);
    let p = p2345.mul_add(f2, p01);
    let e = p * f32::from_bits(z.to_bits() << 23);
    let d = 1.0 + e;
    let r0 = f32::from_bits(RECIP_MAGIC.wrapping_sub(d.to_bits()));
    let e0 = (-d).mul_add(r0, 1.0);
    let u2 = r0.mul_add(e0, r0);
    let e2 = e0 * e0;
    let t2 = e2.mul_add(e2, e2);
    let r = u2.mul_add(t2, u2);
    x * r
}

/// [`silu_q8`] on 8 lanes, bit-identical to scalar [`silu_q8`].
#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(_mm256_set1_ps(-85.0), x);
    let u = _mm256_max_ps(_mm256_mul_ps(x, _mm256_set1_ps(NEG_LOG2_E)), _mm256_set1_ps(-124.0));
    let magic = _mm256_set1_ps(MAGIC_EXP);
    let z = _mm256_add_ps(u, magic);
    let n = _mm256_sub_ps(z, magic);
    let f = _mm256_sub_ps(u, n);
    let f2 = _mm256_mul_ps(f, f);
    let p01 = _mm256_fmadd_ps(_mm256_set1_ps(C1), f, _mm256_set1_ps(C0));
    let p23 = _mm256_fmadd_ps(_mm256_set1_ps(C3), f, _mm256_set1_ps(C2));
    let p45 = _mm256_fmadd_ps(_mm256_set1_ps(C5), f, _mm256_set1_ps(C4));
    let p2345 = _mm256_fmadd_ps(p45, f2, p23);
    let p = _mm256_fmadd_ps(p2345, f2, p01);
    let s = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
    let e = _mm256_mul_ps(p, _mm256_castsi256_ps(s));
    let one = _mm256_set1_ps(1.0);
    let d = _mm256_add_ps(one, e);
    let r0 =
        _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(RECIP_MAGIC as i32), _mm256_castps_si256(d)));
    let e0 = _mm256_fnmadd_ps(d, r0, one);
    let u2 = _mm256_fmadd_ps(r0, e0, r0);
    let e2 = _mm256_mul_ps(e0, e0);
    let t2 = _mm256_fmadd_ps(e2, e2, e2);
    let r = _mm256_fmadd_ps(u2, t2, u2);
    _mm256_mul_ps(x, r)
}

/// [`silu_q8`] on 2 independent 8-lane vectors (16 floats), using Estrin degree-5 evaluation so both FMA pipes
/// stay saturated while peaking at <= 12 live YMM registers (zero stack spills on x86-64 AVX2); bit-identical
/// to scalar [`silu_q8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2_x2(x: [std::arch::x86_64::__m256; 2]) -> [std::arch::x86_64::__m256; 2] {
    use std::arch::x86_64::*;
    let min_x = _mm256_set1_ps(-85.0);
    let x0 = _mm256_max_ps(min_x, x[0]);
    let x1 = _mm256_max_ps(min_x, x[1]);
    let nlog2e = _mm256_set1_ps(NEG_LOG2_E);
    let u_lo = _mm256_set1_ps(-124.0);
    let u0 = _mm256_max_ps(_mm256_mul_ps(x0, nlog2e), u_lo);
    let u1 = _mm256_max_ps(_mm256_mul_ps(x1, nlog2e), u_lo);
    let magic = _mm256_set1_ps(MAGIC_EXP);
    let z0 = _mm256_add_ps(u0, magic);
    let z1 = _mm256_add_ps(u1, magic);
    let f0 = _mm256_sub_ps(u0, _mm256_sub_ps(z0, magic));
    let f1 = _mm256_sub_ps(u1, _mm256_sub_ps(z1, magic));
    let f2_0 = _mm256_mul_ps(f0, f0);
    let f2_1 = _mm256_mul_ps(f1, f1);
    let c5 = _mm256_set1_ps(C5);
    let c4 = _mm256_set1_ps(C4);
    let p45_0 = _mm256_fmadd_ps(c5, f0, c4);
    let p45_1 = _mm256_fmadd_ps(c5, f1, c4);
    let c3 = _mm256_set1_ps(C3);
    let c2 = _mm256_set1_ps(C2);
    let p23_0 = _mm256_fmadd_ps(c3, f0, c2);
    let p23_1 = _mm256_fmadd_ps(c3, f1, c2);
    let c1 = _mm256_set1_ps(C1);
    let c0 = _mm256_set1_ps(C0);
    let p01_0 = _mm256_fmadd_ps(c1, f0, c0);
    let p01_1 = _mm256_fmadd_ps(c1, f1, c0);
    let p2345_0 = _mm256_fmadd_ps(p45_0, f2_0, p23_0);
    let p2345_1 = _mm256_fmadd_ps(p45_1, f2_1, p23_1);
    let p0 = _mm256_fmadd_ps(p2345_0, f2_0, p01_0);
    let p1 = _mm256_fmadd_ps(p2345_1, f2_1, p01_1);
    let e0_exp = _mm256_mul_ps(p0, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z0))));
    let e1_exp = _mm256_mul_ps(p1, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z1))));
    let one = _mm256_set1_ps(1.0);
    let d0 = _mm256_add_ps(one, e0_exp);
    let d1 = _mm256_add_ps(one, e1_exp);
    let rmagic = _mm256_set1_epi32(RECIP_MAGIC as i32);
    let r0_0 = _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d0)));
    let r0_1 = _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d1)));
    let err0 = _mm256_fnmadd_ps(d0, r0_0, one);
    let err1 = _mm256_fnmadd_ps(d1, r0_1, one);
    let u2_0 = _mm256_fmadd_ps(r0_0, err0, r0_0);
    let u2_1 = _mm256_fmadd_ps(r0_1, err1, r0_1);
    let e2_0 = _mm256_mul_ps(err0, err0);
    let e2_1 = _mm256_mul_ps(err1, err1);
    let t2_0 = _mm256_fmadd_ps(e2_0, e2_0, e2_0);
    let t2_1 = _mm256_fmadd_ps(e2_1, e2_1, e2_1);
    let r0 = _mm256_fmadd_ps(u2_0, t2_0, u2_0);
    let r1 = _mm256_fmadd_ps(u2_1, t2_1, u2_1);
    [_mm256_mul_ps(x0, r0), _mm256_mul_ps(x1, r1)]
}

/// [`silu_q8`] on 4 independent 8-lane vectors (32 floats, one full `MR = 4` C4 channel group), evaluated as
/// two register-spill-free [`silu_q8_avx2_x2`] pairs; bit-identical to scalar [`silu_q8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2_x4(x: [std::arch::x86_64::__m256; 4]) -> [std::arch::x86_64::__m256; 4] {
    let [y0, y1] = silu_q8_avx2_x2([x[0], x[1]]);
    let [y2, y3] = silu_q8_avx2_x2([x[2], x[3]]);
    [y0, y1, y2, y3]
}
