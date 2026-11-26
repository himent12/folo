//! AVX2 + FMA kernels for symmetric block-quantized int4 ([`Q4Block`]) quantization, dequantization,
//! `Q4 × f32` dequantizing FMA dot / 4-row GEMV, and exact `Q4 × Q8` integer dot (`vpmaddubsw` + `vpmaddwd`).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::{BLOCK_SIZE, PACKED_BLOCK_BYTES, Q4Block, Q8Block};

/// Horizontal max of 8 `f32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehl_ps(m128, m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_shuffle_ps::<0x55>(m64, m64);
    _mm_cvtss_f32(_mm_max_ss(m64, shuf2))
}

/// Horizontal sum of 8 `f32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(s128, s128);
    let s64 = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_shuffle_ps::<0x55>(s64, s64);
    _mm_cvtss_f32(_mm_add_ss(s64, shuf2))
}

/// Horizontal sum of 8 `i32` lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_epi32(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256::<1>(v);
    let lo = _mm256_castsi256_si128(v);
    let s128 = _mm_add_epi32(lo, hi);
    let s64 = _mm_add_epi32(s128, _mm_srli_si128::<8>(s128));
    let s32 = _mm_add_epi32(s64, _mm_srli_si128::<4>(s64));
    _mm_cvtsi128_si32(s32)
}

/// Replaces non-finite lanes (`NaN` or `±∞`) with `+0.0`.
#[inline]
#[target_feature(enable = "avx2")]
fn zero_non_finite(v: __m256, abs_mask: __m256, inf: __m256) -> __m256 {
    let finite = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v, abs_mask), inf);
    _mm256_and_ps(v, finite)
}

/// Unpacks 16 packed nibble bytes into 32 signed `i8` values in `[-8, 7]` (elements `0..16` in the low
/// 128-bit lane, elements `16..32` in the high 128-bit lane).
#[inline]
#[target_feature(enable = "avx2")]
fn unpack_q4_i8(qs: &[u8; PACKED_BLOCK_BYTES]) -> __m256i {
    // SAFETY: `qs` is 16 bytes.
    let packed = unsafe { _mm_loadu_si128(qs.as_ptr() as *const __m128i) };
    let mask = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(packed, mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
    let u = _mm256_set_m128i(hi, lo);
    _mm256_sub_epi8(u, _mm256_set1_epi8(8))
}

/// AVX2 symmetric `Q4` block quantization, bit-exact with [`super::quantize_block_q4_scalar`].
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn quantize_block_q4_avx2(src: &[f32; BLOCK_SIZE]) -> Q4Block {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let inf = _mm256_set1_ps(f32::INFINITY);
    let p = src.as_ptr();
    // SAFETY: `src` holds 32 floats (`4 × 8`).
    let (v0, v1, v2, v3) = unsafe {
        (
            zero_non_finite(_mm256_loadu_ps(p), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(8)), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(16)), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(24)), abs_mask, inf),
        )
    };
    let m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
    let m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
    let amax = hmax_ps(_mm256_max_ps(m01, m23));
    let scale = amax / 7.0;
    let inv = 1.0 / scale;
    if !(scale > 0.0 && inv.is_finite()) {
        return Q4Block::ZERO;
    }

    let vinv = _mm256_set1_ps(inv);
    let vmin = _mm256_set1_ps(-7.0);
    let vmax = _mm256_set1_ps(7.0);
    let q0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v0, vinv), vmin), vmax));
    let q1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v1, vinv), vmin), vmax));
    let q2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v2, vinv), vmin), vmax));
    let q3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v3, vinv), vmin), vmax));

    // Pack `q0..q1` -> 16 signed `i8` (`0..16`) and `q2..q3` -> 16 signed `i8` (`16..32`).
    let p01_16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q0, q1));
    let p23_16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q2, q3));
    let lo_i8 = _mm_packs_epi16(_mm256_castsi256_si128(p01_16), _mm256_extracti128_si256::<1>(p01_16));
    let hi_i8 = _mm_packs_epi16(_mm256_castsi256_si128(p23_16), _mm256_extracti128_si256::<1>(p23_16));

    let bias = _mm_set1_epi8(8);
    let lo_u = _mm_add_epi8(lo_i8, bias);
    let hi_u = _mm_slli_epi16::<4>(_mm_add_epi8(hi_i8, bias));
    let packed = _mm_or_si128(lo_u, _mm_and_si128(hi_u, _mm_set1_epi8(0xF0u8 as i8)));

    let mut qs = [0u8; PACKED_BLOCK_BYTES];
    // SAFETY: `qs` holds 16 bytes.
    unsafe { _mm_storeu_si128(qs.as_mut_ptr() as *mut __m128i, packed) };
    Q4Block { scale, qs }
}

/// AVX2 `Q4` block dequantization, bit-exact with [`super::dequantize_block_q4_scalar`].
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dequantize_block_q4_avx2(block: &Q4Block, out: &mut [f32; BLOCK_SIZE]) {
    let q_i8 = unpack_q4_i8(&block.qs);
    let lo128 = _mm256_castsi256_si128(q_i8);
    let hi128 = _mm256_extracti128_si256::<1>(q_i8);
    let vs = _mm256_set1_ps(block.scale);

    let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128)), vs);
    let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo128))), vs);
    let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128)), vs);
    let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi128))), vs);

    let dp = out.as_mut_ptr();
    // SAFETY: `out` holds 32 floats.
    unsafe {
        _mm256_storeu_ps(dp, f0);
        _mm256_storeu_ps(dp.add(8), f1);
        _mm256_storeu_ps(dp.add(16), f2);
        _mm256_storeu_ps(dp.add(24), f3);
    }
}

/// AVX2 symmetric `Q8` block quantization, bit-exact with [`super::quantize_block_q8_scalar`].
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn quantize_block_q8_avx2(src: &[f32; BLOCK_SIZE]) -> Q8Block {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let inf = _mm256_set1_ps(f32::INFINITY);
    let p = src.as_ptr();
    // SAFETY: `src` holds 32 floats.
    let (v0, v1, v2, v3) = unsafe {
        (
            zero_non_finite(_mm256_loadu_ps(p), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(8)), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(16)), abs_mask, inf),
            zero_non_finite(_mm256_loadu_ps(p.add(24)), abs_mask, inf),
        )
    };
    let m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
    let m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
    let amax = hmax_ps(_mm256_max_ps(m01, m23));
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    if !(scale > 0.0 && inv.is_finite()) {
        return Q8Block::ZERO;
    }

    let vinv = _mm256_set1_ps(inv);
    let vmin = _mm256_set1_ps(-127.0);
    let vmax = _mm256_set1_ps(127.0);
    let q0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v0, vinv), vmin), vmax));
    let q1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v1, vinv), vmin), vmax));
    let q2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v2, vinv), vmin), vmax));
    let q3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v3, vinv), vmin), vmax));

    let sum_v = _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
    let sum = hsum_epi32(sum_v);

    let p01_16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q0, q1));
    let p23_16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(q2, q3));
    let lo_i8 = _mm_packs_epi16(_mm256_castsi256_si128(p01_16), _mm256_extracti128_si256::<1>(p01_16));
    let hi_i8 = _mm_packs_epi16(_mm256_castsi256_si128(p23_16), _mm256_extracti128_si256::<1>(p23_16));
    let packed = _mm256_set_m128i(hi_i8, lo_i8);

    let mut qs = [0i8; BLOCK_SIZE];
    // SAFETY: `qs` holds 32 bytes.
    unsafe { _mm256_storeu_si256(qs.as_mut_ptr() as *mut __m256i, packed) };
    Q8Block { scale, sum, qs }
}

/// Accumulates one 32-element `Q4 × f32` block dot product into `acc` (`acc += scale * dot(q4, x)`).
#[inline]
#[target_feature(enable = "avx2,fma")]
fn block_dot_q4_f32(blk: &Q4Block, x0: __m256, x1: __m256, x2: __m256, x3: __m256, acc: __m256) -> __m256 {
    let q_i8 = unpack_q4_i8(&blk.qs);
    let lo128 = _mm256_castsi256_si128(q_i8);
    let hi128 = _mm256_extracti128_si256::<1>(q_i8);

    let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo128));
    let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo128)));
    let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi128));
    let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi128)));

    let p0 = _mm256_fmadd_ps(w0, x0, _mm256_mul_ps(w1, x1));
    let p1 = _mm256_fmadd_ps(w2, x2, _mm256_mul_ps(w3, x3));
    let blk_sum = _mm256_add_ps(p0, p1);
    _mm256_fmadd_ps(blk_sum, _mm256_set1_ps(blk.scale), acc)
}

/// Scalar tail dot for `len < BLOCK_SIZE`.
#[inline]
fn tail_dot_q4_f32(blk: &Q4Block, tail: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (idx, &xv) in tail.iter().enumerate() {
        let q = if idx < PACKED_BLOCK_BYTES {
            (blk.qs[idx] & 0x0F) as i32 - 8
        } else {
            (blk.qs[idx - PACKED_BLOCK_BYTES] >> 4) as i32 - 8
        };
        s += q as f32 * xv;
    }
    s * blk.scale
}

/// AVX2 + FMA `Q4 × f32` dot product with 2-block accumulator unrolling.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q4_f32_avx2(w: &[Q4Block], x: &[f32]) -> f32 {
    let (chunks, tail) = x.as_chunks::<BLOCK_SIZE>();
    let nfull = chunks.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 2 <= nfull {
        let p0 = chunks[i].as_ptr();
        let p1 = chunks[i + 1].as_ptr();
        // SAFETY: each chunk has 32 floats and `i + 1 < nfull <= w.len()`.
        unsafe {
            acc0 = block_dot_q4_f32(
                &w[i],
                _mm256_loadu_ps(p0),
                _mm256_loadu_ps(p0.add(8)),
                _mm256_loadu_ps(p0.add(16)),
                _mm256_loadu_ps(p0.add(24)),
                acc0,
            );
            acc1 = block_dot_q4_f32(
                &w[i + 1],
                _mm256_loadu_ps(p1),
                _mm256_loadu_ps(p1.add(8)),
                _mm256_loadu_ps(p1.add(16)),
                _mm256_loadu_ps(p1.add(24)),
                acc1,
            );
        }
        i += 2;
    }
    if i < nfull {
        let p0 = chunks[i].as_ptr();
        // SAFETY: chunk has 32 floats and `i < nfull <= w.len()`.
        unsafe {
            acc0 = block_dot_q4_f32(
                &w[i],
                _mm256_loadu_ps(p0),
                _mm256_loadu_ps(p0.add(8)),
                _mm256_loadu_ps(p0.add(16)),
                _mm256_loadu_ps(p0.add(24)),
                acc0,
            );
        }
    }
    let mut total = hsum_ps(_mm256_add_ps(acc0, acc1));
    if !tail.is_empty() {
        total += tail_dot_q4_f32(&w[nfull], tail);
    }
    total
}

/// 4-row simultaneous AVX2 + FMA `Q4 × f32` dot product, loading `x` once for every 4 weight rows.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q4_f32_x4_avx2(rows: [&[Q4Block]; 4], x: &[f32]) -> [f32; 4] {
    let (chunks, tail) = x.as_chunks::<BLOCK_SIZE>();
    let nfull = chunks.len();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    for (i, chunk) in chunks.iter().enumerate() {
        let p = chunk.as_ptr();
        // SAFETY: `chunk` has 32 floats and every row has at least `nfull` blocks.
        let (x0, x1, x2, x3) = unsafe {
            (
                _mm256_loadu_ps(p),
                _mm256_loadu_ps(p.add(8)),
                _mm256_loadu_ps(p.add(16)),
                _mm256_loadu_ps(p.add(24)),
            )
        };
        a0 = block_dot_q4_f32(&rows[0][i], x0, x1, x2, x3, a0);
        a1 = block_dot_q4_f32(&rows[1][i], x0, x1, x2, x3, a1);
        a2 = block_dot_q4_f32(&rows[2][i], x0, x1, x2, x3, a2);
        a3 = block_dot_q4_f32(&rows[3][i], x0, x1, x2, x3, a3);
    }
    let mut res = [hsum_ps(a0), hsum_ps(a1), hsum_ps(a2), hsum_ps(a3)];
    if !tail.is_empty() {
        for r in 0..4 {
            res[r] += tail_dot_q4_f32(&rows[r][nfull], tail);
        }
    }
    res
}

/// Exact integer AVX2 `Q4 × Q8` dot product using `vpmaddubsw` + `vpmaddwd` and precomputed `Q8Block::sum`.
///
/// For each block, `u_q4 ∈ [0, 15]` and `s_q8 ∈ [-127, 127]`, so pairwise products satisfy
/// `|u0*s0 + u1*s1| <= 15 * 127 * 2 = 3810 << 32767` (zero `i16` saturation) and
/// `Σ (q4 * s_q8) = Σ (u_q4 * s_q8) - 8 * Σ s_q8` is exact in `i32`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_q4_q8_avx2(w: &[Q4Block], x: &[Q8Block]) -> f32 {
    let mask = _mm_set1_epi8(0x0F);
    let ones16 = _mm256_set1_epi16(1);
    let mut acc = 0.0f32;
    for (wb, xb) in w.iter().zip(x) {
        // SAFETY: `wb.qs` is 16 bytes and `xb.qs` is 32 bytes.
        let (u_q4, s_q8) = unsafe {
            let packed = _mm_loadu_si128(wb.qs.as_ptr() as *const __m128i);
            let lo = _mm_and_si128(packed, mask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
            (_mm256_set_m128i(hi, lo), _mm256_loadu_si256(xb.qs.as_ptr() as *const __m256i))
        };
        let prod16 = _mm256_maddubs_epi16(u_q4, s_q8);
        let prod32 = _mm256_madd_epi16(prod16, ones16);
        let biased = hsum_epi32(prod32);
        let exact = biased - 8 * xb.sum;
        acc += (exact as f32) * (wb.scale * xb.scale);
    }
    acc
}
