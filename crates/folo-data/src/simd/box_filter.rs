//! Bit-exact AVX2 vertical row accumulation and horizontal box-filter reduction kernels for `resize.rs`.
//!
//! For every `n in 2..=4096` and every block sum `x <= 255 * n`, rounded division `(x + n / 2) / n` is computed
//! bit-exact via `((x + (n >> 1)) * ceil(2^32 / n)) >> 32` using `_mm256_mul_epu32`, or a plain shift when `n` is a
//! power of two.

#[cfg(target_arch = "x86_64")]
use super::bilinear::pack_4_rgb0_u8_avx2;
#[cfg(target_arch = "x86_64")]
use super::has_avx2;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Sums `ny` consecutive `stride`-byte rows from `src_band` (`ny * stride` bytes) directly into `acc` (`stride`
/// `u32`s), overwriting `acc` without a separate zeroing pass.
#[inline]
pub(crate) fn sum_rows_u8_to_u32(acc: &mut [u32], src_band: &[u8], stride: usize) {
    assert_eq!(acc.len(), stride);
    assert!(stride > 0 && src_band.len().is_multiple_of(stride));
    let ny = src_band.len() / stride;
    if ny == 0 {
        acc.fill(0);
        return;
    }
    if ny == 1 {
        set_u8_to_u32(acc, src_band);
        return;
    }
    sum2_u8_to_u32(acc, &src_band[..stride], &src_band[stride..2 * stride]);
    for row in src_band[2 * stride..].chunks_exact(stride) {
        add_u8_to_u32(acc, row);
    }
}

/// Widens `src` bytes into `acc` (`u32`s), overwriting `acc`.
#[inline]
fn set_u8_to_u32(acc: &mut [u32], src: &[u8]) {
    assert_eq!(acc.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && src.len() >= 16 {
        // SAFETY: CPU supports AVX2 and lengths are asserted equal.
        unsafe { set_u8_to_u32_avx2(acc, src) };
        return;
    }
    for (a, &v) in acc.iter_mut().zip(src) {
        *a = v as u32;
    }
}

/// Writes `a[i] as u32 + b[i] as u32` into `acc[i]` in a single pass.
#[inline]
fn sum2_u8_to_u32(acc: &mut [u32], a: &[u8], b: &[u8]) {
    assert!(acc.len() == a.len() && acc.len() == b.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && a.len() >= 16 {
        // SAFETY: CPU supports AVX2 and lengths are asserted equal.
        unsafe { sum2_u8_to_u32_avx2(acc, a, b) };
        return;
    }
    for ((dst, &va), &vb) in acc.iter_mut().zip(a).zip(b) {
        *dst = va as u32 + vb as u32;
    }
}

/// Adds `src` bytes into `acc` (`u32`s).
#[inline]
pub(crate) fn add_u8_to_u32(acc: &mut [u32], src: &[u8]) {
    assert_eq!(acc.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && src.len() >= 16 {
        // SAFETY: CPU supports AVX2 and lengths are asserted equal.
        unsafe { add_u8_to_u32_avx2(acc, src) };
        return;
    }
    for (a, &v) in acc.iter_mut().zip(src) {
        *a += v as u32;
    }
}

/// Horizontally reduces `body` (`n_out * fx * 3` `u32`s of vertically summed RGB pixels) by `fx` columns per output
/// pixel and divides by `full_n` with exact rounding `(sum + full_n / 2) / full_n`, writing `row_body` (`n_out * 3`
/// bytes). Returns `true` if AVX2 handled the full slice, or `false` if the caller should run the scalar path (e.g.
/// `full_n > 4096` or non-AVX2 target).
#[inline]
pub(crate) fn box_hreduce_u32_to_u8(body: &[u32], row_body: &mut [u8], fx: usize, full_n: u32) -> bool {
    let n_out = row_body.len() / 3;
    assert!(
        row_body.len() == n_out * 3
            && row_body.len().checked_mul(fx) == Some(body.len())
            && fx >= 1
            && full_n >= 1
    );
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && full_n <= 4096 && ((fx == 1 && body.len() >= 16) || (fx >= 2 && n_out >= 6)) {
        // SAFETY: CPU supports AVX2, `full_n <= 4096`, and slice lengths match `n_out * fx * 3` and `n_out * 3`.
        unsafe { box_hreduce_u32_to_u8_avx2(body, row_body, fx, full_n) };
        return true;
    }
    false
}

/// Scalar reference for [`box_hreduce_u32_to_u8`], used by tests and fallback paths.
#[cfg(test)]
pub(crate) fn box_hreduce_u32_to_u8_scalar(body: &[u32], row_body: &mut [u8], fx: usize, full_n: u32) {
    let half = full_n / 2;
    for (px, block) in row_body.as_chunks_mut::<3>().0.iter_mut().zip(body.chunks_exact(fx * 3)) {
        let mut s = [0u32; 3];
        for p in block.as_chunks::<3>().0 {
            s[0] += p[0];
            s[1] += p[1];
            s[2] += p[2];
        }
        px[0] = ((s[0] + half) / full_n) as u8;
        px[1] = ((s[1] + half) / full_n) as u8;
        px[2] = ((s[2] + half) / full_n) as u8;
    }
}

// ------------------------------------------------------------------------------------------------
// AVX2 implementations
// ------------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn set_u8_to_u32_avx2(acc: &mut [u32], src: &[u8]) {
    let n = src.len();
    let ap = acc.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n == acc.len() == src.len()`.
        unsafe {
            let b = _mm_loadu_si128(sp.add(i) as *const __m128i);
            _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_cvtepu8_epi32(b));
            _mm256_storeu_si256(ap.add(i + 8) as *mut __m256i, _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(b)));
        }
        i += 16;
    }
    while i < n {
        acc[i] = src[i] as u32;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum2_u8_to_u32_avx2(acc: &mut [u32], a: &[u8], b: &[u8]) {
    let n = a.len();
    let dp = acc.as_mut_ptr();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n == acc.len() == a.len() == b.len()`.
        unsafe {
            let va = _mm_loadu_si128(ap.add(i) as *const __m128i);
            let vb = _mm_loadu_si128(bp.add(i) as *const __m128i);
            let lo = _mm256_add_epi32(_mm256_cvtepu8_epi32(va), _mm256_cvtepu8_epi32(vb));
            let hi = _mm256_add_epi32(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(va)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
            );
            _mm256_storeu_si256(dp.add(i) as *mut __m256i, lo);
            _mm256_storeu_si256(dp.add(i + 8) as *mut __m256i, hi);
        }
        i += 16;
    }
    while i < n {
        acc[i] = a[i] as u32 + b[i] as u32;
        i += 1;
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

/// Exact rounded unsigned division `(x + n / 2) / n` on 8 `u32` lanes for `2 <= n <= 4096` and `x <= 255 * n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_div_mul_u32_avx2(x: __m256i, vhalf: __m256i, vm: __m256i) -> __m256i {
    let q = _mm256_add_epi32(x, vhalf);
    let p_even = _mm256_mul_epu32(q, vm);
    let p_odd = _mm256_mul_epu32(_mm256_srli_epi64::<32>(q), vm);
    _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(p_even), p_odd)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_div_u32_avx2(
    x: __m256i,
    is_pow2: bool,
    vhalf: __m256i,
    vshift: __m128i,
    vm: __m256i,
) -> __m256i {
    if is_pow2 {
        _mm256_srl_epi32(_mm256_add_epi32(x, vhalf), vshift)
    } else {
        unsafe { round_div_mul_u32_avx2(x, vhalf, vm) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn box_hreduce_u32_to_u8_avx2(body: &[u32], row_body: &mut [u8], fx: usize, full_n: u32) {
    let is_pow2 = full_n.is_power_of_two();
    let half = full_n >> 1;
    let shift = full_n.trailing_zeros();
    let m = if is_pow2 { 0u32 } else { (1u64 << 32).div_ceil(full_n as u64) as u32 };
    let vhalf = _mm256_set1_epi32(half as i32);
    let vshift = _mm_cvtsi32_si128(shift as i32);
    let vm = _mm256_set1_epi32(m as i32);

    if fx == 1 {
        let n = body.len();
        let bp = body.as_ptr();
        let op = row_body.as_mut_ptr();
        let mut i = 0usize;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n == body.len() == row_body.len()`.
            unsafe {
                let a = _mm256_loadu_si256(bp.add(i) as *const __m256i);
                let b = _mm256_loadu_si256(bp.add(i + 8) as *const __m256i);
                let (qa, qb) = if full_n == 1 {
                    (a, b)
                } else {
                    (
                        round_div_u32_avx2(a, is_pow2, vhalf, vshift, vm),
                        round_div_u32_avx2(b, is_pow2, vhalf, vshift, vm),
                    )
                };
                let packed = pack_4_rgb0_u8_avx2(qa, qb);
                _mm_storeu_si128(op.add(i) as *mut __m128i, packed);
            }
            i += 16;
        }
        while i < n {
            row_body[i] = ((body[i] + half) / full_n) as u8;
            i += 1;
        }
        return;
    }

    let n_out = row_body.len() / 3;
    let bp = body.as_ptr();
    let op = row_body.as_mut_ptr();
    let stride = fx * 3;
    let mask_rgb0 = _mm256_setr_epi32(-1, -1, -1, 0, -1, -1, -1, 0);
    let rgb0_to_rgb = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    let mut x = 0usize;
    // 16-byte overshoot store at `x * 3` needs `x * 3 + 16 <= n_out * 3` => `x + 6 <= n_out`.
    // Each 128-bit load at `(x + 3) * stride + (fx - 1) * 3` reads 4 `u32`s (ending at `(x + 4) * stride + 1`),
    // which is strictly `<= n_out * stride` whenever `x + 5 <= n_out` and `fx >= 2`.
    while x + 6 <= n_out {
        // SAFETY: `x + 6 <= n_out`, `fx >= 2`, so every 128-bit source load and 16-byte destination store is in bounds.
        unsafe {
            let p0 = bp.add(x * stride);
            let p1 = p0.add(stride);
            let p2 = p1.add(stride);
            let p3 = p2.add(stride);

            let (s0, s1, s2, s3) = match fx {
                2 => (
                    _mm_add_epi32(
                        _mm_loadu_si128(p0 as *const __m128i),
                        _mm_loadu_si128(p0.add(3) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_loadu_si128(p1 as *const __m128i),
                        _mm_loadu_si128(p1.add(3) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_loadu_si128(p2 as *const __m128i),
                        _mm_loadu_si128(p2.add(3) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_loadu_si128(p3 as *const __m128i),
                        _mm_loadu_si128(p3.add(3) as *const __m128i),
                    ),
                ),
                3 => (
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p0 as *const __m128i),
                            _mm_loadu_si128(p0.add(3) as *const __m128i),
                        ),
                        _mm_loadu_si128(p0.add(6) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p1 as *const __m128i),
                            _mm_loadu_si128(p1.add(3) as *const __m128i),
                        ),
                        _mm_loadu_si128(p1.add(6) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p2 as *const __m128i),
                            _mm_loadu_si128(p2.add(3) as *const __m128i),
                        ),
                        _mm_loadu_si128(p2.add(6) as *const __m128i),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p3 as *const __m128i),
                            _mm_loadu_si128(p3.add(3) as *const __m128i),
                        ),
                        _mm_loadu_si128(p3.add(6) as *const __m128i),
                    ),
                ),
                4 => (
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p0 as *const __m128i),
                            _mm_loadu_si128(p0.add(3) as *const __m128i),
                        ),
                        _mm_add_epi32(
                            _mm_loadu_si128(p0.add(6) as *const __m128i),
                            _mm_loadu_si128(p0.add(9) as *const __m128i),
                        ),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p1 as *const __m128i),
                            _mm_loadu_si128(p1.add(3) as *const __m128i),
                        ),
                        _mm_add_epi32(
                            _mm_loadu_si128(p1.add(6) as *const __m128i),
                            _mm_loadu_si128(p1.add(9) as *const __m128i),
                        ),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p2 as *const __m128i),
                            _mm_loadu_si128(p2.add(3) as *const __m128i),
                        ),
                        _mm_add_epi32(
                            _mm_loadu_si128(p2.add(6) as *const __m128i),
                            _mm_loadu_si128(p2.add(9) as *const __m128i),
                        ),
                    ),
                    _mm_add_epi32(
                        _mm_add_epi32(
                            _mm_loadu_si128(p3 as *const __m128i),
                            _mm_loadu_si128(p3.add(3) as *const __m128i),
                        ),
                        _mm_add_epi32(
                            _mm_loadu_si128(p3.add(6) as *const __m128i),
                            _mm_loadu_si128(p3.add(9) as *const __m128i),
                        ),
                    ),
                ),
                _ => {
                    let mut a0 = _mm_loadu_si128(p0 as *const __m128i);
                    let mut a1 = _mm_loadu_si128(p1 as *const __m128i);
                    let mut a2 = _mm_loadu_si128(p2 as *const __m128i);
                    let mut a3 = _mm_loadu_si128(p3 as *const __m128i);
                    for k in 1..fx {
                        let off = k * 3;
                        a0 = _mm_add_epi32(a0, _mm_loadu_si128(p0.add(off) as *const __m128i));
                        a1 = _mm_add_epi32(a1, _mm_loadu_si128(p1.add(off) as *const __m128i));
                        a2 = _mm_add_epi32(a2, _mm_loadu_si128(p2.add(off) as *const __m128i));
                        a3 = _mm_add_epi32(a3, _mm_loadu_si128(p3.add(off) as *const __m128i));
                    }
                    (a0, a1, a2, a3)
                }
            };

            let s01 =
                _mm256_and_si256(_mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s0), s1), mask_rgb0);
            let s23 =
                _mm256_and_si256(_mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s2), s3), mask_rgb0);
            let q01 = round_div_u32_avx2(s01, is_pow2, vhalf, vshift, vm);
            let q23 = round_div_u32_avx2(s23, is_pow2, vhalf, vshift, vm);
            let rgba = pack_4_rgb0_u8_avx2(q01, q23);
            let rgb = _mm_shuffle_epi8(rgba, rgb0_to_rgb);
            _mm_storeu_si128(op.add(x * 3) as *mut __m128i, rgb);
        }
        x += 4;
    }

    while x < n_out {
        let block = &body[x * stride..(x + 1) * stride];
        let mut s = [0u32; 3];
        for p in block.as_chunks::<3>().0 {
            s[0] += p[0];
            s[1] += p[1];
            s[2] += p[2];
        }
        row_body[x * 3] = ((s[0] + half) / full_n) as u8;
        row_body[x * 3 + 1] = ((s[1] + half) / full_n) as u8;
        row_body[x * 3 + 2] = ((s[2] + half) / full_n) as u8;
        x += 1;
    }
}
