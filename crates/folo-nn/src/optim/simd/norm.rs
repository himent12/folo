//! 1-pass overflow-safe online-rescaled norm accumulators, scaling, and AMP unscale/finiteness CPU kernels.

/// Overflow- and underflow-safe representation of a sum of squares `Σ xᵢ² = scale² · ssq` with
/// `scale >= 0` and `ssq >= 0` (Hammarling / LAPACK `lassq` online rescaling).
///
/// Non-finite propagation matches IEEE-754 `sqrt(Σ xᵢ²)`:
/// - Any `NaN` yields `NaN`.
/// - Any `±Inf` (with no `NaN`) yields `+Inf`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ScaledSumSq {
    pub scale: f64,
    pub ssq: f64,
}

impl ScaledSumSq {
    /// Zero accumulator (`norm() == 0.0`).
    pub const ZERO: Self = Self { scale: 0.0, ssq: 0.0 };

    /// Construct from an unscaled `f64` sum of squares (e.g. from an `f32` slice, where every finite `f32`
    /// squares into `[2^-298, 2^77]` with zero overflow or underflow in `f64`).
    #[inline]
    pub fn from_sq_f64(sq: f64) -> Self {
        if sq.is_nan() {
            Self { scale: f64::NAN, ssq: f64::NAN }
        } else if sq.is_infinite() {
            Self { scale: f64::INFINITY, ssq: 1.0 }
        } else if sq <= 0.0 {
            Self::ZERO
        } else {
            Self { scale: sq.sqrt(), ssq: 1.0 }
        }
    }

    /// Combine two rescaled accumulators without overflow or underflow.
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        if self.scale.is_nan() || other.scale.is_nan() || self.ssq.is_nan() || other.ssq.is_nan() {
            return Self { scale: f64::NAN, ssq: f64::NAN };
        }
        if self.scale.is_infinite()
            || other.scale.is_infinite()
            || self.ssq.is_infinite()
            || other.ssq.is_infinite()
        {
            return Self { scale: f64::INFINITY, ssq: 1.0 };
        }
        if other.scale == 0.0 || other.ssq == 0.0 {
            return self;
        }
        if self.scale == 0.0 || self.ssq == 0.0 {
            return other;
        }
        if self.scale >= other.scale {
            let r = other.scale / self.scale;
            Self { scale: self.scale, ssq: self.ssq + other.ssq * (r * r) }
        } else {
            let r = self.scale / other.scale;
            Self { scale: other.scale, ssq: other.ssq + self.ssq * (r * r) }
        }
    }

    /// Euclidean norm `scale · sqrt(ssq)`.
    #[inline]
    pub fn norm(self) -> f64 {
        if self.scale.is_nan() || self.ssq.is_nan() {
            f64::NAN
        } else if self.scale.is_infinite() || self.ssq.is_infinite() {
            f64::INFINITY
        } else if self.scale == 0.0 || self.ssq == 0.0 {
            0.0
        } else {
            self.scale * self.ssq.sqrt()
        }
    }
}

/// In-place `buf[i] *= factor`.
pub(crate) fn scale_slice(buf: &mut [f32], factor: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        unsafe { scale_avx2(buf, factor) };
        return;
    }
    scale_scalar(buf, factor);
}

#[inline]
fn scale_scalar(buf: &mut [f32], factor: f32) {
    for v in buf {
        *v *= factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(buf: &mut [f32], factor: f32) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let vf = _mm256_set1_ps(factor);
    let ptr = buf.as_mut_ptr();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vf);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), vf);
            let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), vf);
            let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), vf);
            _mm256_storeu_ps(ptr.add(i), v0);
            _mm256_storeu_ps(ptr.add(i + 8), v1);
            _mm256_storeu_ps(ptr.add(i + 16), v2);
            _mm256_storeu_ps(ptr.add(i + 24), v3);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vf);
            _mm256_storeu_ps(ptr.add(i), v0);
        }
        i += 8;
    }
    scale_scalar(&mut buf[i..], factor);
}

/// In-place 1-pass two-factor scaling `buf[i] = (buf[i] * f1) * f2`, used when a single `f32` clip factor
/// would be subnormal or underflow to zero on extreme finite gradients (e.g. near `f32::MAX`).
pub(crate) fn scale2_slice(buf: &mut [f32], f1: f32, f2: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        unsafe { scale2_avx2(buf, f1, f2) };
        return;
    }
    scale2_scalar(buf, f1, f2);
}

#[inline]
pub(super) fn scale2_scalar(buf: &mut [f32], f1: f32, f2: f32) {
    for v in buf {
        *v = (*v * f1) * f2;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale2_avx2(buf: &mut [f32], f1: f32, f2: f32) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let vf1 = _mm256_set1_ps(f1);
    let vf2 = _mm256_set1_ps(f2);
    let ptr = buf.as_mut_ptr();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vf1), vf2);
            let v1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), vf1), vf2);
            let v2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), vf1), vf2);
            let v3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), vf1), vf2);
            _mm256_storeu_ps(ptr.add(i), v0);
            _mm256_storeu_ps(ptr.add(i + 8), v1);
            _mm256_storeu_ps(ptr.add(i + 16), v2);
            _mm256_storeu_ps(ptr.add(i + 24), v3);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vf1), vf2);
            _mm256_storeu_ps(ptr.add(i), v0);
        }
        i += 8;
    }
    scale2_scalar(&mut buf[i..], f1, f2);
}

/// Read-only 1-pass finiteness check over an `f32` slice (avoids write traffic when `inv_scale == 1.0`).
pub(crate) fn check_finite_slice(src: &[f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { check_finite_avx2(src) };
    }
    check_finite_scalar(src)
}

#[inline]
pub(super) fn check_finite_scalar(src: &[f32]) -> bool {
    let mut finite = true;
    for &v in src {
        finite &= v.is_finite();
    }
    finite
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_finite_avx2(src: &[f32]) -> bool {
    use std::arch::x86_64::*;

    let n = src.len();
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut bad0 = _mm256_setzero_si256();
    let mut bad1 = _mm256_setzero_si256();
    let ptr = src.as_ptr();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let v3 = _mm256_loadu_ps(ptr.add(i + 24));
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            let b1 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v1), exp_mask), exp_mask);
            let b2 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v2), exp_mask), exp_mask);
            let b3 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v3), exp_mask), exp_mask);
            bad0 = _mm256_or_si256(bad0, _mm256_or_si256(b0, b1));
            bad1 = _mm256_or_si256(bad1, _mm256_or_si256(b2, b3));
        }
        i += 32;
    }
    let mut bad = _mm256_or_si256(bad0, bad1);
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, b0);
        }
        i += 8;
    }
    let vec_finite = _mm256_testz_si256(bad, bad) != 0;
    vec_finite & check_finite_scalar(&src[i..])
}

/// Read-only 1-pass finiteness check + `f64` sum of squares over an `f32` slice (avoids write traffic when
/// `inv_scale == 1.0`).
pub(crate) fn check_sum_sq_f64_slice(src: &[f32]) -> (bool, f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { check_sum_sq_f64_avx2(src) };
    }
    check_sum_sq_f64_scalar(src)
}

#[inline]
pub(super) fn check_sum_sq_f64_scalar(src: &[f32]) -> (bool, f64) {
    let mut finite = true;
    let mut sq = 0.0f64;
    for &v in src {
        finite &= v.is_finite();
        let d = v as f64;
        sq += d * d;
    }
    (finite, sq)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_sum_sq_f64_avx2(src: &[f32]) -> (bool, f64) {
    use std::arch::x86_64::*;

    let n = src.len();
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut bad = _mm256_setzero_si256();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let ptr = src.as_ptr();
    let mut i = 0;

    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            let b1 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v1), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, _mm256_or_si256(b0, b1));

            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            let d2 = _mm256_cvtps_pd(_mm256_castps256_ps128(v1));
            let d3 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v1));

            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(d2, d2));
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(d3, d3));
        }
        i += 16;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, b0);

            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
        }
        i += 8;
    }

    let sum_v = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` has 4 `f64` elements.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum_v) };
    let vec_sq = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let vec_finite = _mm256_testz_si256(bad, bad) != 0;
    let (tail_finite, tail_sq) = check_sum_sq_f64_scalar(&src[i..]);
    (vec_finite & tail_finite, vec_sq + tail_sq)
}

/// Fused in-place AMP unscale + finiteness check: multiplies every element of `buf` by `inv_scale` and
/// returns `true` iff every resulting value is finite.
pub(crate) fn unscale_check_slice(buf: &mut [f32], inv_scale: f32) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { unscale_check_avx2(buf, inv_scale) };
    }
    unscale_check_scalar(buf, inv_scale)
}

#[inline]
pub(super) fn unscale_check_scalar(buf: &mut [f32], inv_scale: f32) -> bool {
    let mut finite = true;
    for v in buf {
        let nv = *v * inv_scale;
        *v = nv;
        finite &= nv.is_finite();
    }
    finite
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unscale_check_avx2(buf: &mut [f32], inv_scale: f32) -> bool {
    use std::arch::x86_64::*;

    let n = buf.len();
    let v_inv = _mm256_set1_ps(inv_scale);
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut bad0 = _mm256_setzero_si256();
    let mut bad1 = _mm256_setzero_si256();
    let ptr = buf.as_mut_ptr();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_inv);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_inv);
            let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_inv);
            let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_inv);
            _mm256_storeu_ps(ptr.add(i), v0);
            _mm256_storeu_ps(ptr.add(i + 8), v1);
            _mm256_storeu_ps(ptr.add(i + 16), v2);
            _mm256_storeu_ps(ptr.add(i + 24), v3);
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            let b1 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v1), exp_mask), exp_mask);
            let b2 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v2), exp_mask), exp_mask);
            let b3 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v3), exp_mask), exp_mask);
            bad0 = _mm256_or_si256(bad0, _mm256_or_si256(b0, b1));
            bad1 = _mm256_or_si256(bad1, _mm256_or_si256(b2, b3));
        }
        i += 32;
    }
    let mut bad = _mm256_or_si256(bad0, bad1);
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_inv);
            _mm256_storeu_ps(ptr.add(i), v0);
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, b0);
        }
        i += 8;
    }
    let vec_finite = _mm256_testz_si256(bad, bad) != 0;
    vec_finite & unscale_check_scalar(&mut buf[i..], inv_scale)
}

/// Fused in-place AMP unscale + finiteness check + `f64` sum-of-squares over one gradient buffer.
pub(crate) fn unscale_check_sum_sq_slice(buf: &mut [f32], inv_scale: f32) -> (bool, f64) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { unscale_check_sum_sq_avx2(buf, inv_scale) };
    }
    unscale_check_sum_sq_scalar(buf, inv_scale)
}

#[inline]
pub(super) fn unscale_check_sum_sq_scalar(buf: &mut [f32], inv_scale: f32) -> (bool, f64) {
    let mut finite = true;
    let mut sq = 0.0f64;
    for v in buf {
        let nv = *v * inv_scale;
        *v = nv;
        finite &= nv.is_finite();
        let d = nv as f64;
        sq += d * d;
    }
    (finite, sq)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unscale_check_sum_sq_avx2(buf: &mut [f32], inv_scale: f32) -> (bool, f64) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let v_inv = _mm256_set1_ps(inv_scale);
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut bad = _mm256_setzero_si256();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let ptr = buf.as_mut_ptr();
    let mut i = 0;

    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_inv);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_inv);
            _mm256_storeu_ps(ptr.add(i), v0);
            _mm256_storeu_ps(ptr.add(i + 8), v1);

            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            let b1 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v1), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, _mm256_or_si256(b0, b1));

            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            let d2 = _mm256_cvtps_pd(_mm256_castps256_ps128(v1));
            let d3 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v1));

            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(d2, d2));
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(d3, d3));
        }
        i += 16;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_inv);
            _mm256_storeu_ps(ptr.add(i), v0);
            let b0 = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v0), exp_mask), exp_mask);
            bad = _mm256_or_si256(bad, b0);

            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
        }
        i += 8;
    }

    let sum_v = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` has 4 `f64` elements.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum_v) };
    let vec_sq = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let vec_finite = _mm256_testz_si256(bad, bad) != 0;
    let (tail_finite, tail_sq) = unscale_check_sum_sq_scalar(&mut buf[i..], inv_scale);
    (vec_finite & tail_finite, vec_sq + tail_sq)
}

/// Sum of squares `Σ (v as f64)²` over an `f32` slice, vectorized in `f64` with AVX2.
pub(crate) fn sum_sq_f64_slice(src: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { sum_sq_f64_avx2(src) };
    }
    sum_sq_f64_scalar(src)
}

#[inline]
fn sum_sq_f64_scalar(src: &[f32]) -> f64 {
    let mut sq = 0.0f64;
    for &v in src {
        let d = v as f64;
        sq += d * d;
    }
    sq
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_f64_avx2(src: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let n = src.len();
    let ptr = src.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut i = 0;

    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            let d2 = _mm256_cvtps_pd(_mm256_castps256_ps128(v1));
            let d3 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v1));
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(d2, d2));
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(d3, d3));
        }
        i += 16;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let v0 = _mm256_loadu_ps(ptr.add(i));
            let d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v0));
            let d1 = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(v0));
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
        }
        i += 8;
    }

    let sum_v = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` has 4 `f64` elements.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum_v) };
    ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + sum_sq_f64_scalar(&src[i..])
}
