//! Symmetric per-output-channel s8 weight quantisation (`quantize_weights`), with an AVX2 fast path that is
//! bitwise identical to the scalar reference on all inputs (including ties-away-from-zero, `±0.0`, `±∞`, and
//! `NaN`).

use rayon::prelude::*;

/// Scalar reference for one row of symmetric s8 weight quantisation (`q[i] = clamp(round(w[i] / s), -127, 127)`).
#[inline]
pub(super) fn quantize_row_scalar(row: &[f32], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(row.len(), dst.len());
    let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    for (d, &v) in dst.iter_mut().zip(row) {
        *d = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn quantize8_avx2(
    v: std::arch::x86_64::__m256,
    vs: std::arch::x86_64::__m256,
    sign_bit: std::arch::x86_64::__m256,
    half: std::arch::x86_64::__m256,
    one: std::arch::x86_64::__m256,
    max127: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let x = _mm256_div_ps(v, vs);
    let sign = _mm256_and_ps(x, sign_bit);
    let abs_x = _mm256_andnot_ps(sign_bit, x);
    // Exact round-half-away-from-zero: `floor(|x|) + (|x| - floor(|x|) >= 0.5 ? 1 : 0)`.
    let fl = _mm256_floor_ps(abs_x);
    let frac = _mm256_sub_ps(abs_x, fl);
    let up = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(frac, half), one);
    let r = _mm256_add_ps(fl, up);
    // Clamp magnitude to 127.0, restore sign, and zero out NaN lanes (Rust `NaN as i8 == 0`).
    let clamped = _mm256_min_ps(r, max127);
    let signed = _mm256_or_ps(clamped, sign);
    let clean = _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(x, x), signed);
    _mm256_cvttps_epi32(clean)
}

/// AVX2 symmetric s8 quantisation of one weight row, bitwise identical to [`quantize_row_scalar`].
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn quantize_row_avx2(row: &[f32], dst: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    assert_eq!(row.len(), dst.len());
    let sign_bit = _mm256_set1_ps(-0.0);

    // Pass 1: 16-wide absmax (NaN-ignoring, matching `a.max(v.abs())`).
    let mut m0 = _mm256_setzero_ps();
    let mut m1 = _mm256_setzero_ps();
    let (r16, r_rem) = row.as_chunks::<16>();
    for c in r16 {
        // SAFETY: `c` is `[f32; 16]` (64 bytes) and AVX2 is enabled.
        unsafe {
            let p = c.as_ptr();
            let a0 = _mm256_andnot_ps(sign_bit, _mm256_loadu_ps(p));
            let a1 = _mm256_andnot_ps(sign_bit, _mm256_loadu_ps(p.add(8)));
            // `vmaxps(a, m)` returns `m` when `a` is NaN, matching `m.max(a)`.
            m0 = _mm256_max_ps(a0, m0);
            m1 = _mm256_max_ps(a1, m1);
        }
    }
    let (r8, r_tail) = r_rem.as_chunks::<8>();
    for c in r8 {
        // SAFETY: `c` is `[f32; 8]` (32 bytes) and AVX2 is enabled.
        unsafe {
            let a0 = _mm256_andnot_ps(sign_bit, _mm256_loadu_ps(c.as_ptr()));
            m0 = _mm256_max_ps(a0, m0);
        }
    }
    let m_vec = _mm256_max_ps(m1, m0);
    let mut lanes = [0f32; 8];
    // SAFETY: `lanes` holds 8 floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m_vec) };
    let mut m = lanes.iter().copied().fold(0f32, f32::max);
    for &v in r_tail {
        m = m.max(v.abs());
    }

    let s = if m > 0.0 { m / 127.0 } else { 1.0 };

    // Pass 2: quantise and pack 32 floats -> 32 `i8` per step, then 8-wide chunks, then scalar tail.
    let vs = _mm256_set1_ps(s);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let max127 = _mm256_set1_ps(127.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let zero_i = _mm256_setzero_si256();

    let (w32, w_rem) = row.as_chunks::<32>();
    let (d32, d_rem) = dst.as_chunks_mut::<32>();
    for (wc, dc) in w32.iter().zip(d32.iter_mut()) {
        // SAFETY: `wc` is `[f32; 32]` and `dc` is `[i8; 32]`; AVX2 is enabled.
        unsafe {
            let p = wc.as_ptr();
            let q0 = quantize8_avx2(_mm256_loadu_ps(p), vs, sign_bit, half, one, max127);
            let q1 = quantize8_avx2(_mm256_loadu_ps(p.add(8)), vs, sign_bit, half, one, max127);
            let q2 = quantize8_avx2(_mm256_loadu_ps(p.add(16)), vs, sign_bit, half, one, max127);
            let q3 = quantize8_avx2(_mm256_loadu_ps(p.add(24)), vs, sign_bit, half, one, max127);
            let p01 = _mm256_packs_epi32(q0, q1);
            let p23 = _mm256_packs_epi32(q2, q3);
            let packed = _mm256_packs_epi16(p01, p23);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256(dc.as_mut_ptr() as *mut __m256i, ordered);
        }
    }
    let (w8, w_tail) = w_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (wc, dc) in w8.iter().zip(d8.iter_mut()) {
        // SAFETY: `wc` is `[f32; 8]` and `dc` is `[i8; 8]`; AVX2 is enabled.
        unsafe {
            let q0 = quantize8_avx2(_mm256_loadu_ps(wc.as_ptr()), vs, sign_bit, half, one, max127);
            let p0 = _mm256_packs_epi32(q0, zero_i);
            let packed = _mm256_packs_epi16(p0, zero_i);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm_storel_epi64(dc.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ordered));
        }
    }
    for (d, &v) in d_tail.iter_mut().zip(w_tail) {
        *d = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

#[inline]
fn quantize_row(row: &[f32], dst: &mut [i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if row.len() >= 8 && super::kernel::uses_avx2() {
        // SAFETY: `uses_avx2()` checked AVX2 support above.
        return unsafe { quantize_row_avx2(row, dst) };
    }
    quantize_row_scalar(row, dst)
}

/// Symmetric per-output-channel s8 weights of `w` (`[o, K]` row-major): `w[r, k] ≈ scale[r] · q[r, k]`, `q` in
/// `-127..=127` (`round(w / scale)`, ties away from zero). Returns `(q, scale)`.
pub fn quantize_weights(w: &[f32], o: usize) -> (Vec<i8>, Vec<f32>) {
    let k = w.len() / o.max(1);
    let mut q = vec![0i8; w.len()];
    let mut scales = vec![0f32; o];
    if o == 0 {
        return (q, scales);
    }
    if k == 0 {
        scales.fill(1.0);
        return (q, scales);
    }
    let total = o * k;
    if total >= crate::cpu::PAR_MIN && o > 1 {
        let rows_per_task = (crate::cpu::CHUNK / k).max(1);
        q[..total]
            .par_chunks_exact_mut(k)
            .zip(scales.par_iter_mut())
            .enumerate()
            .with_min_len(rows_per_task)
            .for_each(|(r, (dst, sc))| {
                *sc = quantize_row(&w[r * k..(r + 1) * k], dst);
            });
    } else {
        for (r, (dst, sc)) in q[..total].chunks_exact_mut(k).zip(scales.iter_mut()).enumerate() {
            *sc = quantize_row(&w[r * k..(r + 1) * k], dst);
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn ref_quantize_weights(w: &[f32], o: usize) -> (Vec<i8>, Vec<f32>) {
        let k = w.len() / o.max(1);
        let mut q = vec![0i8; w.len()];
        let mut scales = vec![0f32; o];
        for r in 0..o {
            let row = &w[r * k..(r + 1) * k];
            let m = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if m > 0.0 { m / 127.0 } else { 1.0 };
            scales[r] = s;
            for (d, &v) in q[r * k..(r + 1) * k].iter_mut().zip(row) {
                *d = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (q, scales)
    }

    #[test]
    fn quantize_weights_avx2_matches_scalar_bitwise_across_lengths_ties_and_nonfinite() {
        let mut rng = Rng::seed(2026);
        // Aligned, unaligned, 32-wide + 8-wide + tail, zero-length, and non-divisible edge cases.
        for &(o, k) in &[
            (0usize, 0usize),
            (4, 0),
            (1, 1),
            (1, 7),
            (1, 8),
            (1, 15),
            (1, 16),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 41),
            (3, 65),
            (64, 577), // > PAR_MIN parallel path
        ] {
            let w: Vec<f32> = (0..o * k).map(|_| rng.normal() * 0.7).collect();
            let (gq, gs) = quantize_weights(&w, o);
            let (wq, ws) = ref_quantize_weights(&w, o);
            assert_eq!(gq, wq, "q mismatch for ({o}, {k})");
            assert_eq!(
                gs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ws.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "scale mismatch for ({o}, {k})"
            );
        }

        // Exact halfway ties-away-from-zero, just-below-halfway boundary values, signed zeros, subnormals,
        // infinities, and NaNs (including NaN-only and all-zero rows).
        let mut specials = vec![
            0.0f32,
            -0.0,
            f32::from_bits(0x3eff_ffff), // 0.5 - 2^-25 (must round to 0, NOT 1)
            -f32::from_bits(0x3eff_ffff),
            0.5,
            -0.5,
            1.5 - 1e-7,
            1.5,
            -1.5,
            2.5,
            -2.5,
            64.5 - 2.0f32.powi(-18),
            64.5,
            -64.5,
            126.5,
            -126.5,
            127.0,
            -127.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
        ];
        // Pad to > 32 + 8 + tail = 45 elements so 32-wide, 8-wide, and tail paths all run.
        while specials.len() < 45 {
            specials.push((specials.len() as f32 - 30.0) * 0.5);
        }
        // Build multi-row matrix: special row, all-zero row, NaN-only row, +inf row, -inf row.
        let k = specials.len();
        let mut mat = Vec::new();
        mat.extend_from_slice(&specials);
        mat.extend(std::iter::repeat_n(0.0f32, k));
        mat.extend(std::iter::repeat_n(f32::NAN, k));
        let mut with_inf = specials.clone();
        with_inf[3] = f32::INFINITY;
        with_inf[19] = f32::NEG_INFINITY;
        mat.extend_from_slice(&with_inf);
        // Also test trailing unconsumed elements when `w.len()` is not divisible by `o`.
        let mut ragged = mat.clone();
        ragged.extend_from_slice(&[1.0, 2.0, 3.0]);
        for (input, o) in [(mat.as_slice(), 4), (ragged.as_slice(), 4)] {
            let (gq, gs) = quantize_weights(input, o);
            let (wq, ws) = ref_quantize_weights(input, o);
            assert_eq!(gq, wq, "specials q mismatch (len={})", input.len());
            assert_eq!(
                gs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ws.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "specials scale mismatch (len={})",
                input.len()
            );
        }
    }
}
