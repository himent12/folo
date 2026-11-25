//! Vectorized linear-combination, AMP unscaling, norm-reduction, and Muon/ortho CPU kernels.

/// In-place `dst[i] = a * dst[i] + b * src[i]` (with exact specialisations for `a == 1.0` and `b == 1.0`).
pub(crate) fn axpby_slice(dst: &mut [f32], src: &[f32], a: f32, b: f32) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above; slice lengths verified equal.
        unsafe {
            match (a == 1.0, b == 1.0) {
                (true, true) => axpby_avx2::<true, true>(dst, src, a, b),
                (true, false) => axpby_avx2::<true, false>(dst, src, a, b),
                (false, true) => axpby_avx2::<false, true>(dst, src, a, b),
                (false, false) => axpby_avx2::<false, false>(dst, src, a, b),
            }
        }
        return;
    }
    axpby_scalar(dst, src, a, b);
}

#[inline]
pub(super) fn axpby_scalar(dst: &mut [f32], src: &[f32], a: f32, b: f32) {
    match (a == 1.0, b == 1.0) {
        (true, true) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }
        (true, false) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s * b;
            }
        }
        (false, true) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = *d * a + s;
            }
        }
        (false, false) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = *d * a + s * b;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn axpby_avx2<const A_ONE: bool, const B_ONE: bool>(dst: &mut [f32], src: &[f32], a: f32, b: f32) {
    use std::arch::x86_64::*;

    let n = dst.len();
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vd = _mm256_loadu_ps(dst_ptr.add($off));
            let vs = _mm256_loadu_ps(src_ptr.add($off));
            let lhs = if A_ONE { vd } else { _mm256_mul_ps(vd, va) };
            let rhs = if B_ONE { vs } else { _mm256_mul_ps(vs, vb) };
            _mm256_storeu_ps(dst_ptr.add($off), _mm256_add_ps(lhs, rhs));
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `dst`, `src` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `dst`, `src` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    axpby_scalar(&mut dst[i..], &src[i..], a, b);
}

/// Fused Muon momentum + Nesterov update on contiguous CPU slices:
/// ```text
/// gm = (1 - mu) * g
/// b  = mu * b + gm
/// u  = gm + mu * b
/// ```
pub(crate) fn muon_momentum_nesterov_slice(b: &mut [f32], g: &[f32], u: &mut [f32], mu: f32) {
    assert_eq!(b.len(), g.len());
    assert_eq!(b.len(), u.len());
    let one_minus_mu = 1.0 - mu;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above; slice lengths verified equal.
        unsafe { muon_momentum_nesterov_avx2(b, g, u, mu, one_minus_mu) };
        return;
    }
    muon_momentum_nesterov_scalar(b, g, u, mu, one_minus_mu);
}

#[inline]
pub(super) fn muon_momentum_nesterov_scalar(
    b: &mut [f32],
    g: &[f32],
    u: &mut [f32],
    mu: f32,
    one_minus_mu: f32,
) {
    for ((b, &g), u) in b.iter_mut().zip(g).zip(u) {
        let gm = one_minus_mu * g;
        let nb = mu * *b + gm;
        *b = nb;
        *u = gm + mu * nb;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn muon_momentum_nesterov_avx2(b: &mut [f32], g: &[f32], u: &mut [f32], mu: f32, one_minus_mu: f32) {
    use std::arch::x86_64::*;

    let n = b.len();
    let v_mu = _mm256_set1_ps(mu);
    let v_1mm = _mm256_set1_ps(one_minus_mu);
    let b_ptr = b.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let u_ptr = u.as_mut_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vb = _mm256_loadu_ps(b_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let gm = _mm256_mul_ps(v_1mm, vg);
            let nb = _mm256_add_ps(_mm256_mul_ps(v_mu, vb), gm);
            let vu = _mm256_add_ps(gm, _mm256_mul_ps(v_mu, nb));
            _mm256_storeu_ps(b_ptr.add($off), nb);
            _mm256_storeu_ps(u_ptr.add($off), vu);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `b`, `g`, `u` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `b`, `g`, `u` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    muon_momentum_nesterov_scalar(&mut b[i..], &g[i..], &mut u[i..], mu, one_minus_mu);
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

/// Sum of squares in `f32` (`Σ (v * scale)²` when `scale != 1.0`, else `Σ v²`).
pub(crate) fn sum_sq_f32_slice(src: &[f32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe {
            if scale == 1.0 {
                sum_sq_f32_avx2::<false>(src, scale)
            } else {
                sum_sq_f32_avx2::<true>(src, scale)
            }
        };
    }
    sum_sq_f32_scalar(src, scale)
}

#[inline]
fn sum_sq_f32_scalar(src: &[f32], scale: f32) -> f32 {
    let mut sq = 0.0f32;
    if scale == 1.0 {
        for &v in src {
            sq += v * v;
        }
    } else {
        for &v in src {
            let vs = v * scale;
            sq += vs * vs;
        }
    }
    sq
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_sq_f32_avx2<const SCALED: bool>(src: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    let n = src.len();
    let ptr = src.as_ptr();
    let vs = _mm256_set1_ps(scale);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let mut v0 = _mm256_loadu_ps(ptr.add(i));
            let mut v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let mut v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let mut v3 = _mm256_loadu_ps(ptr.add(i + 24));
            if SCALED {
                v0 = _mm256_mul_ps(v0, vs);
                v1 = _mm256_mul_ps(v1, vs);
                v2 = _mm256_mul_ps(v2, vs);
                v3 = _mm256_mul_ps(v3, vs);
            }
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(v1, v1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(v2, v2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(v3, v3));
        }
        i += 32;
    }
    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let mut v0 = _mm256_loadu_ps(ptr.add(i));
            if SCALED {
                v0 = _mm256_mul_ps(v0, vs);
            }
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v0, v0));
        }
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` has 8 `f32` elements.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let vec_sum =
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    vec_sum + sum_sq_f32_scalar(&src[i..], scale)
}

/// Maximum absolute value over `src` (`0.0` when empty).
fn max_abs_f32_slice(src: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        return unsafe { max_abs_f32_avx2(src) };
    }
    src.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_abs_f32_avx2(src: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = src.len();
    let ptr = src.as_ptr();
    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
    let mut m0 = _mm256_setzero_ps();
    let mut m1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), abs_mask);
            let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), abs_mask);
            m0 = _mm256_max_ps(a0, m0);
            m1 = _mm256_max_ps(a1, m1);
        }
        i += 16;
    }
    let mut mv = _mm256_max_ps(m0, m1);
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), abs_mask);
            mv = _mm256_max_ps(a0, mv);
        }
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` has 8 `f32` elements.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), mv) };
    let mut out = lanes.into_iter().fold(0.0f32, f32::max);
    for &v in &src[i..] {
        out = out.max(v.abs());
    }
    out
}

/// In-place `buf[i] = (buf[i] * scale) / denom` (or `buf[i] / denom` when `scale == 1.0`).
fn scale_div_slice(buf: &mut [f32], scale: f32, denom: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above.
        unsafe {
            if scale == 1.0 {
                scale_div_avx2::<false>(buf, scale, denom);
            } else {
                scale_div_avx2::<true>(buf, scale, denom);
            }
        }
        return;
    }
    scale_div_scalar(buf, scale, denom);
}

#[inline]
fn scale_div_scalar(buf: &mut [f32], scale: f32, denom: f32) {
    if scale == 1.0 {
        for v in buf {
            *v /= denom;
        }
    } else {
        for v in buf {
            *v = (*v * scale) / denom;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_div_avx2<const SCALED: bool>(buf: &mut [f32], scale: f32, denom: f32) {
    use std::arch::x86_64::*;

    let n = buf.len();
    let ptr = buf.as_mut_ptr();
    let vs = _mm256_set1_ps(scale);
    let vd = _mm256_set1_ps(denom);
    let mut i = 0;

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        unsafe {
            let mut v0 = _mm256_loadu_ps(ptr.add(i));
            let mut v1 = _mm256_loadu_ps(ptr.add(i + 8));
            let mut v2 = _mm256_loadu_ps(ptr.add(i + 16));
            let mut v3 = _mm256_loadu_ps(ptr.add(i + 24));
            if SCALED {
                v0 = _mm256_mul_ps(v0, vs);
                v1 = _mm256_mul_ps(v1, vs);
                v2 = _mm256_mul_ps(v2, vs);
                v3 = _mm256_mul_ps(v3, vs);
            }
            _mm256_storeu_ps(ptr.add(i), _mm256_div_ps(v0, vd));
            _mm256_storeu_ps(ptr.add(i + 8), _mm256_div_ps(v1, vd));
            _mm256_storeu_ps(ptr.add(i + 16), _mm256_div_ps(v2, vd));
            _mm256_storeu_ps(ptr.add(i + 24), _mm256_div_ps(v3, vd));
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let mut v0 = _mm256_loadu_ps(ptr.add(i));
            if SCALED {
                v0 = _mm256_mul_ps(v0, vs);
            }
            _mm256_storeu_ps(ptr.add(i), _mm256_div_ps(v0, vd));
        }
        i += 8;
    }
    scale_div_scalar(&mut buf[i..], scale, denom);
}

/// Zero-allocation CPU per-channel Polar LMO update for depthwise conv weights from precomputed `u`.
pub(crate) fn muon_dw_polar_step_slice(
    p: &mut [f32],
    u: &[f32],
    rows: usize,
    cols: usize,
    lr: f32,
    ns_eps: f32,
) {
    assert_eq!(p.len(), rows * cols);
    assert_eq!(u.len(), rows * cols);
    const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
    const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
    let neg_lr = -lr;
    for r in 0..rows {
        let base = r * cols;
        let row = &u[base..base + cols];
        let max_abs = max_abs_f32_slice(row);
        let sc_big = if max_abs > BIG_THRESH { DOWN_SCALE } else { 1.0 };
        let sq = sum_sq_f32_slice(row, sc_big);
        let denom = sq.sqrt() + ns_eps * sc_big;
        let dst = &mut p[base..base + cols];
        if sc_big == 1.0 {
            for (d, &v) in dst.iter_mut().zip(row) {
                *d += neg_lr * (v / denom);
            }
        } else {
            for (d, &v) in dst.iter_mut().zip(row) {
                *d += neg_lr * ((v * sc_big) / denom);
            }
        }
    }
}

/// Fused zero-allocation CPU depthwise Muon momentum + (optional) Nesterov + Polar LMO parameter update.
#[allow(clippy::too_many_arguments)]
pub(crate) fn muon_dw_polar_fused_slice(
    p: &mut [f32],
    b: &mut [f32],
    g: &[f32],
    rows: usize,
    cols: usize,
    mu: f32,
    nesterov: bool,
    lr: f32,
    ns_eps: f32,
) {
    assert_eq!(p.len(), rows * cols);
    assert_eq!(b.len(), rows * cols);
    assert_eq!(g.len(), rows * cols);
    let one_minus_mu = 1.0 - mu;
    const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
    const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
    let neg_lr = -lr;

    let mut stack_u = [0.0f32; 64];
    let mut heap_u: Vec<f32> = if nesterov && cols > stack_u.len() { vec![0.0f32; cols] } else { Vec::new() };

    for r in 0..rows {
        let base = r * cols;
        let b_row = &mut b[base..base + cols];
        let g_row = &g[base..base + cols];
        let p_row = &mut p[base..base + cols];
        let u_row: &[f32] = if nesterov {
            let u_buf: &mut [f32] = if cols <= stack_u.len() { &mut stack_u[..cols] } else { &mut heap_u };
            muon_momentum_nesterov_slice(b_row, g_row, u_buf, mu);
            u_buf
        } else {
            axpby_slice(b_row, g_row, mu, one_minus_mu);
            b_row
        };
        let max_abs = max_abs_f32_slice(u_row);
        let sc_big = if max_abs > BIG_THRESH { DOWN_SCALE } else { 1.0 };
        let sq = sum_sq_f32_slice(u_row, sc_big);
        let denom = sq.sqrt() + ns_eps * sc_big;
        if sc_big == 1.0 {
            for (d, &v) in p_row.iter_mut().zip(u_row) {
                *d += neg_lr * (v / denom);
            }
        } else {
            for (d, &v) in p_row.iter_mut().zip(u_row) {
                *d += neg_lr * ((v * sc_big) / denom);
            }
        }
    }
}

/// In-place per-batch Frobenius normalization for CPU `orthogonalize_many_geo`.
/// Returns `(norms, eps_scales)` (non-empty when `use_s4` is true).
pub(crate) fn ortho_normalize_batch(
    x: &mut [f32],
    batch: usize,
    stride: usize,
    use_s4: bool,
    ns_eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), batch * stride);
    const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
    const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
    let mut norms = if use_s4 { Vec::with_capacity(batch) } else { Vec::new() };
    let mut eps_ss = if use_s4 { Vec::with_capacity(batch) } else { Vec::new() };
    for bi in 0..batch {
        let slice = &mut x[bi * stride..(bi + 1) * stride];
        if use_s4 {
            let max_abs = max_abs_f32_slice(slice);
            let sc = if max_abs > BIG_THRESH { DOWN_SCALE } else { 1.0 };
            let eps_s = sc * ns_eps;
            let norm = sum_sq_f32_slice(slice, sc).sqrt() + eps_s;
            scale_div_slice(slice, sc, norm);
            norms.push(norm);
            eps_ss.push(eps_s);
        } else {
            let norm = sum_sq_f32_slice(slice, 1.0).sqrt() + ns_eps;
            scale_div_slice(slice, 1.0, norm);
        }
    }
    (norms, eps_ss)
}
