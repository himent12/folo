//! Bit-exact division-free minimax fast SiLU (`silu_q8`) for the CPU int8 epilogue.
//!
//! Computes `x / (1 + exp(-x))` using:
//! 1. Fused base-2 range reduction `u = clamp(x · (-log2(e)), -124, 124)`, `z = u + (1.5·2^23 + 127)` so
//!    `n = z - magic` rounds to nearest-even and `bits(z) << 23` yields `2^n` in a single shift (zero `vroundps`,
//!    zero `vcvtps2dq`, zero `vpaddd`).
//! 2. Economised degree-5 minimax polynomial for `2^f` on `f ∈ [-0.5, 0.5]` (relative error `< 1e-7`, `2^0 = 1`).
//! 3. Deterministic IEEE-754 bit-hack reciprocal seed `0x7EF3_11C3 - bits(1 + e)` followed by a 4-FMA factored
//!    Householder-6 step `r = r0 · (1 + e0) · (1 + e0^2 + e0^4)` (`e0 = 1 - d·r0`, truncation error `e0^6 <= 1.66e-8`,
//!    exact `r(1.0) == 1.0` and `r(2.0) == 0.5`, 33% lower latency than 3 Newton-Raphson steps), achieving
//!    `< 2e-7` hybrid error (`|err| / max(1, |ref|)`) with ZERO `vdivps` and ZERO platform-divergent `rcpps`.
//!
//! Scalar and AVX2 paths execute the exact same IEEE-754 FMA/mul/add/sub/integer operations in the same order,
//! producing bit-identical results across CPUs.

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

/// The epilogue's SiLU, `x / (1 + e^{−x})`, division-free and bit-exact between scalar and AVX2 (`|err| / max(1, |ref|) < 2e-7`
/// on `[-86, 86]`, finite at extremes, `silu_q8(0.0) == 0.0`, `silu_q8(-0.0) == -0.0`, NaN stays NaN).
#[inline]
pub fn silu_q8(x: f32) -> f32 {
    let x = maxps(-88.0, x);
    let u = minps(maxps(x * NEG_LOG2_E, -124.0), 124.0);
    let z = u + MAGIC_EXP;
    let n = z - MAGIC_EXP;
    let f = u - n;
    let mut p = C5;
    p = p.mul_add(f, C4);
    p = p.mul_add(f, C3);
    p = p.mul_add(f, C2);
    p = p.mul_add(f, C1);
    p = p.mul_add(f, C0);
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
    let x = _mm256_max_ps(_mm256_set1_ps(-88.0), x);
    let u = _mm256_min_ps(
        _mm256_max_ps(_mm256_mul_ps(x, _mm256_set1_ps(NEG_LOG2_E)), _mm256_set1_ps(-124.0)),
        _mm256_set1_ps(124.0),
    );
    let magic = _mm256_set1_ps(MAGIC_EXP);
    let z = _mm256_add_ps(u, magic);
    let n = _mm256_sub_ps(z, magic);
    let f = _mm256_sub_ps(u, n);
    let mut p = _mm256_set1_ps(C5);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C4));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C3));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C2));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C1));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(C0));
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

/// [`silu_q8`] on 4 independent 8-lane vectors (32 floats, one full `MR = 4` C4 channel group), interleaved to
/// hide FMA latency and share constant broadcasts; bit-identical to 4 calls of [`silu_q8_avx2`] / scalar [`silu_q8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2_x4(x: [std::arch::x86_64::__m256; 4]) -> [std::arch::x86_64::__m256; 4] {
    use std::arch::x86_64::*;
    let min_x = _mm256_set1_ps(-88.0);
    let x = [
        _mm256_max_ps(min_x, x[0]),
        _mm256_max_ps(min_x, x[1]),
        _mm256_max_ps(min_x, x[2]),
        _mm256_max_ps(min_x, x[3]),
    ];
    let nlog2e = _mm256_set1_ps(NEG_LOG2_E);
    let u_lo = _mm256_set1_ps(-124.0);
    let u_hi = _mm256_set1_ps(124.0);
    let u = [
        _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x[0], nlog2e), u_lo), u_hi),
        _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x[1], nlog2e), u_lo), u_hi),
        _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x[2], nlog2e), u_lo), u_hi),
        _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x[3], nlog2e), u_lo), u_hi),
    ];
    let magic = _mm256_set1_ps(MAGIC_EXP);
    let z = [
        _mm256_add_ps(u[0], magic),
        _mm256_add_ps(u[1], magic),
        _mm256_add_ps(u[2], magic),
        _mm256_add_ps(u[3], magic),
    ];
    let f = [
        _mm256_sub_ps(u[0], _mm256_sub_ps(z[0], magic)),
        _mm256_sub_ps(u[1], _mm256_sub_ps(z[1], magic)),
        _mm256_sub_ps(u[2], _mm256_sub_ps(z[2], magic)),
        _mm256_sub_ps(u[3], _mm256_sub_ps(z[3], magic)),
    ];
    let c5 = _mm256_set1_ps(C5);
    let c4 = _mm256_set1_ps(C4);
    let mut p = [
        _mm256_fmadd_ps(c5, f[0], c4),
        _mm256_fmadd_ps(c5, f[1], c4),
        _mm256_fmadd_ps(c5, f[2], c4),
        _mm256_fmadd_ps(c5, f[3], c4),
    ];
    for c in [C3, C2, C1, C0] {
        let vc = _mm256_set1_ps(c);
        p[0] = _mm256_fmadd_ps(p[0], f[0], vc);
        p[1] = _mm256_fmadd_ps(p[1], f[1], vc);
        p[2] = _mm256_fmadd_ps(p[2], f[2], vc);
        p[3] = _mm256_fmadd_ps(p[3], f[3], vc);
    }
    let e = [
        _mm256_mul_ps(p[0], _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z[0])))),
        _mm256_mul_ps(p[1], _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z[1])))),
        _mm256_mul_ps(p[2], _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z[2])))),
        _mm256_mul_ps(p[3], _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(z[3])))),
    ];
    let one = _mm256_set1_ps(1.0);
    let d = [
        _mm256_add_ps(one, e[0]),
        _mm256_add_ps(one, e[1]),
        _mm256_add_ps(one, e[2]),
        _mm256_add_ps(one, e[3]),
    ];
    let rmagic = _mm256_set1_epi32(RECIP_MAGIC as i32);
    let r0 = [
        _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d[0]))),
        _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d[1]))),
        _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d[2]))),
        _mm256_castsi256_ps(_mm256_sub_epi32(rmagic, _mm256_castps_si256(d[3]))),
    ];
    let e0 = [
        _mm256_fnmadd_ps(d[0], r0[0], one),
        _mm256_fnmadd_ps(d[1], r0[1], one),
        _mm256_fnmadd_ps(d[2], r0[2], one),
        _mm256_fnmadd_ps(d[3], r0[3], one),
    ];
    let u2 = [
        _mm256_fmadd_ps(r0[0], e0[0], r0[0]),
        _mm256_fmadd_ps(r0[1], e0[1], r0[1]),
        _mm256_fmadd_ps(r0[2], e0[2], r0[2]),
        _mm256_fmadd_ps(r0[3], e0[3], r0[3]),
    ];
    let e2 = [
        _mm256_mul_ps(e0[0], e0[0]),
        _mm256_mul_ps(e0[1], e0[1]),
        _mm256_mul_ps(e0[2], e0[2]),
        _mm256_mul_ps(e0[3], e0[3]),
    ];
    let t2 = [
        _mm256_fmadd_ps(e2[0], e2[0], e2[0]),
        _mm256_fmadd_ps(e2[1], e2[1], e2[1]),
        _mm256_fmadd_ps(e2[2], e2[2], e2[2]),
        _mm256_fmadd_ps(e2[3], e2[3], e2[3]),
    ];
    let r = [
        _mm256_fmadd_ps(u2[0], t2[0], u2[0]),
        _mm256_fmadd_ps(u2[1], t2[1], u2[1]),
        _mm256_fmadd_ps(u2[2], t2[2], u2[2]),
        _mm256_fmadd_ps(u2[3], t2[3], u2[3]),
    ];
    [
        _mm256_mul_ps(x[0], r[0]),
        _mm256_mul_ps(x[1], r[1]),
        _mm256_mul_ps(x[2], r[2]),
        _mm256_mul_ps(x[3], r[3]),
    ]
}
