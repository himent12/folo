//! Bit-exact AVX2 (and portable scalar) separable max-pooling primitives for `max_pool2d_values` and
//! `max_pool2d` (with argmax indices).
//!
//! Every primitive preserves the exact row-major first-maximum / first-NaN tie-breaking rule of
//! `beats(a, b, bi)`: `a` replaces `b` iff `a > b || bi < 0 || (a.is_nan() && !b.is_nan())`, preserving exact
//! `-0.0` vs `+0.0` sign bits and NaN payloads bit-for-bit across scalar and AVX2 paths.

#[cfg(target_arch = "x86_64")]
use super::has_avx2;

/// One step of value-only max scan: `a` replaces `b` iff `a > b || (a.is_nan() && !b.is_nan())`.
#[inline(always)]
pub(super) fn step_val(b: &mut f32, a: f32) {
    if a > *b || (a.is_nan() && !b.is_nan()) {
        *b = a;
    }
}

/// One step of indexed max scan: valid candidate `(a, ai >= 0)` replaces `(b, bi)` iff `bi < 0` or `a` beats `b`.
#[inline(always)]
pub(super) fn step_idx(b: &mut f32, bi: &mut i32, a: f32, ai: i32) {
    if ai >= 0 && (*bi < 0 || a > *b || (a.is_nan() && !b.is_nan())) {
        *b = a;
        *bi = ai;
    }
}

/// 8-lane AVX2 `step_val`: returns `a` where `a > b || (a.is_nan() && !b.is_nan())`, else `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn max_step8(b: std::arch::x86_64::__m256, a: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let nle = _mm256_cmp_ps::<_CMP_NLE_UQ>(a, b);
    let b_ord = _mm256_cmp_ps::<_CMP_ORD_Q>(b, b);
    _mm256_blendv_ps(b, a, _mm256_and_ps(nle, b_ord))
}

/// 8-lane AVX2 `step_idx`: returns updated `(best, bi)` for valid `ai >= 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn beats_indexed8(
    b: std::arch::x86_64::__m256,
    bi: std::arch::x86_64::__m256i,
    a: std::arch::x86_64::__m256,
    ai: std::arch::x86_64::__m256i,
    neg1: std::arch::x86_64::__m256i,
    zero: std::arch::x86_64::__m256i,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    let a_valid = _mm256_cmpgt_epi32(ai, neg1);
    let b_empty = _mm256_cmpgt_epi32(zero, bi);
    let val_beats = _mm256_and_ps(_mm256_cmp_ps::<_CMP_NLE_UQ>(a, b), _mm256_cmp_ps::<_CMP_ORD_Q>(b, b));
    let take_a = _mm256_and_si256(a_valid, _mm256_or_si256(b_empty, _mm256_castps_si256(val_beats)));
    (_mm256_blendv_ps(b, a, _mm256_castsi256_ps(take_a)), _mm256_blendv_epi8(bi, ai, take_a))
}

// ---------- Value-only (`max_pool2d_values`) kernels ----------

/// Horizontal `sw == 1` max over `kw` taps: `dst[ox] = max_{t in 0..kw} row[ox + t]` (single write per output).
#[inline]
pub(super) fn max_row_s1(row: &[f32], kw: usize, dst: &mut [f32]) {
    let ow = dst.len();
    assert!(kw >= 1 && ow.checked_add(kw - 1).is_some_and(|n| row.len() >= n));
    #[cfg(target_arch = "x86_64")]
    if ow >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_row_s1_avx2(row, kw, dst) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if ow >= 4 {
        return super::wasm_simd::max_row_s1_wasm(row, kw, dst);
    }
    for (ox, d) in dst.iter_mut().enumerate() {
        let mut m = row[ox];
        for t in 1..kw {
            step_val(&mut m, row[ox + t]);
        }
        *d = m;
    }
}

/// # Safety
/// The CPU supports AVX2, `kw >= 1`, and `row.len() >= dst.len() + kw - 1` without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_row_s1_avx2(row: &[f32], kw: usize, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let ow = dst.len();
    let mut ox = 0;
    // SAFETY: `ox + 8 <= ow` and `row.len() >= ow + kw - 1` guarantee `ox + t + 8 <= row.len()`.
    unsafe {
        while ox + 8 <= ow {
            let base = row.as_ptr().add(ox);
            let mut m = _mm256_loadu_ps(base);
            for t in 1..kw {
                m = max_step8(m, _mm256_loadu_ps(base.add(t)));
            }
            _mm256_storeu_ps(dst.as_mut_ptr().add(ox), m);
            ox += 8;
        }
    }
    for (j, d) in dst[ox..].iter_mut().enumerate() {
        let mut m = row[ox + j];
        for t in 1..kw {
            step_val(&mut m, row[ox + j + t]);
        }
        *d = m;
    }
}

/// Horizontal `sw == 2, kw == 2` max: `dst[ox] = max(row[2*ox], row[2*ox + 1])`.
#[inline]
pub(super) fn max_pairs(row: &[f32], dst: &mut [f32]) {
    assert!(dst.len().checked_mul(2).is_some_and(|n| row.len() >= n));
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_pairs_avx2(row, dst) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 {
        return super::wasm_simd::max_pairs_wasm(row, dst);
    }
    for (d, pair) in dst.iter_mut().zip(row.as_chunks::<2>().0) {
        let mut m = pair[0];
        step_val(&mut m, pair[1]);
        *d = m;
    }
}

/// # Safety
/// The CPU supports AVX2 and `row.len() >= 2 * dst.len()` without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_pairs_avx2(row: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let (sc, sr) = row[..2 * n].as_chunks::<16>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `s` is 16 floats and `d` is 8 floats.
        unsafe {
            let v0 = _mm256_loadu_ps(s.as_ptr());
            let v1 = _mm256_loadu_ps(s.as_ptr().add(8));
            let ev = _mm256_shuffle_ps::<0x88>(v0, v1);
            let od = _mm256_shuffle_ps::<0xDD>(v0, v1);
            let m = max_step8(ev, od);
            let ordered = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(m)));
            _mm256_storeu_ps(d.as_mut_ptr(), ordered);
        }
    }
    for (d, pair) in dr.iter_mut().zip(sr.as_chunks::<2>().0) {
        let mut m = pair[0];
        step_val(&mut m, pair[1]);
        *d = m;
    }
}

/// Vertical max over `kh` rows of width `ow` (stride `ow` between rows), writing `dst[0..ow]` once.
#[inline]
pub(super) fn max_cols(rv: &[f32], ow: usize, kh: usize, dst: &mut [f32]) {
    assert!(kh >= 1 && dst.len() == ow && kh.checked_mul(ow).is_some_and(|n| rv.len() >= n));
    #[cfg(target_arch = "x86_64")]
    if ow >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_cols_avx2(rv, ow, kh, dst) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if ow >= 4 {
        return super::wasm_simd::max_cols_wasm(rv, ow, kh, dst);
    }
    dst.copy_from_slice(&rv[..ow]);
    for t in 1..kh {
        let src = &rv[t * ow..(t + 1) * ow];
        for (b, &a) in dst.iter_mut().zip(src) {
            step_val(b, a);
        }
    }
}

/// # Safety
/// The CPU supports AVX2, `kh >= 1`, `dst.len() == ow`, and `rv.len() >= kh * ow` without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_cols_avx2(rv: &[f32], ow: usize, kh: usize, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let mut ox = 0;
    // SAFETY: `ox + 8 <= ow` and `rv.len() >= kh * ow`.
    unsafe {
        while ox + 8 <= ow {
            let mut m = _mm256_loadu_ps(rv.as_ptr().add(ox));
            for t in 1..kh {
                m = max_step8(m, _mm256_loadu_ps(rv.as_ptr().add(t * ow + ox)));
            }
            _mm256_storeu_ps(dst.as_mut_ptr().add(ox), m);
            ox += 8;
        }
    }
    for j in ox..ow {
        let mut m = rv[j];
        for t in 1..kh {
            step_val(&mut m, rv[t * ow + j]);
        }
        dst[j] = m;
    }
}

// ---------- Indexed (`max_pool2d` with argmax) kernels ----------

/// Horizontal `sw == 1` indexed max over `kw` taps; adds `row_base` (`yy * w`) to every valid (`>= 0`) index.
#[inline]
pub(super) fn max_row_idx_s1(
    row: &[f32],
    row_idx: &[i32],
    kw: usize,
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    let ow = dst_v.len();
    assert!(
        kw >= 1
            && dst_i.len() == ow
            && ow.checked_add(kw - 1).is_some_and(|n| row.len() >= n && row_idx.len() >= n)
    );
    #[cfg(target_arch = "x86_64")]
    if ow >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_row_idx_s1_avx2(row, row_idx, kw, row_base, dst_v, dst_i) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if ow >= 4 {
        return super::wasm_simd::max_row_idx_s1_wasm(row, row_idx, kw, row_base, dst_v, dst_i);
    }
    for ox in 0..ow {
        let mut m = row[ox];
        let mut mi = row_idx[ox];
        for t in 1..kw {
            step_idx(&mut m, &mut mi, row[ox + t], row_idx[ox + t]);
        }
        dst_v[ox] = m;
        dst_i[ox] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

/// # Safety
/// The CPU supports AVX2, `kw >= 1`, `dst_i.len() == dst_v.len()`, and `row`/`row_idx` hold at least
/// `dst_v.len() + kw - 1` entries without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_row_idx_s1_avx2(
    row: &[f32],
    row_idx: &[i32],
    kw: usize,
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    use std::arch::x86_64::*;
    let ow = dst_v.len();
    let neg1 = _mm256_set1_epi32(-1);
    let zero = _mm256_setzero_si256();
    let vbase = _mm256_set1_epi32(row_base);
    let mut ox = 0;
    // SAFETY: `ox + 8 <= ow` and `row`/`row_idx` have at least `ow + kw - 1` entries.
    unsafe {
        while ox + 8 <= ow {
            let vp = row.as_ptr().add(ox);
            let ip = row_idx.as_ptr().add(ox);
            let mut m = _mm256_loadu_ps(vp);
            let mut mi = _mm256_loadu_si256(ip as *const __m256i);
            for t in 1..kw {
                let a = _mm256_loadu_ps(vp.add(t));
                let ai = _mm256_loadu_si256(ip.add(t) as *const __m256i);
                (m, mi) = beats_indexed8(m, mi, a, ai, neg1, zero);
            }
            let valid = _mm256_cmpgt_epi32(mi, neg1);
            let flat = _mm256_blendv_epi8(neg1, _mm256_add_epi32(mi, vbase), valid);
            _mm256_storeu_ps(dst_v.as_mut_ptr().add(ox), m);
            _mm256_storeu_si256(dst_i.as_mut_ptr().add(ox) as *mut __m256i, flat);
            ox += 8;
        }
    }
    for j in ox..ow {
        let mut m = row[j];
        let mut mi = row_idx[j];
        for t in 1..kw {
            step_idx(&mut m, &mut mi, row[j + t], row_idx[j + t]);
        }
        dst_v[j] = m;
        dst_i[j] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

/// Horizontal `sw == 2, kw == 2` indexed max; adds `row_base` (`yy * w`) to every valid (`>= 0`) index.
#[inline]
pub(super) fn max_pairs_idx(
    row: &[f32],
    row_idx: &[i32],
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    let ow = dst_v.len();
    assert!(dst_i.len() == ow && ow.checked_mul(2).is_some_and(|n| row.len() >= n && row_idx.len() >= n));
    #[cfg(target_arch = "x86_64")]
    if ow >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_pairs_idx_avx2(row, row_idx, row_base, dst_v, dst_i) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if ow >= 4 {
        return super::wasm_simd::max_pairs_idx_wasm(row, row_idx, row_base, dst_v, dst_i);
    }
    for ox in 0..ow {
        let mut m = row[2 * ox];
        let mut mi = row_idx[2 * ox];
        step_idx(&mut m, &mut mi, row[2 * ox + 1], row_idx[2 * ox + 1]);
        dst_v[ox] = m;
        dst_i[ox] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

/// # Safety
/// The CPU supports AVX2, `dst_i.len() == dst_v.len()`, and `row`/`row_idx` hold at least `2 * dst_v.len()` entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_pairs_idx_avx2(
    row: &[f32],
    row_idx: &[i32],
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    use std::arch::x86_64::*;
    let ow = dst_v.len();
    let neg1 = _mm256_set1_epi32(-1);
    let zero = _mm256_setzero_si256();
    let vbase = _mm256_set1_epi32(row_base);
    let mut ox = 0;
    // SAFETY: `ox + 8 <= ow` implies `2 * ox + 16 <= 2 * ow <= row.len(), row_idx.len()`.
    unsafe {
        while ox + 8 <= ow {
            let vp = row.as_ptr().add(2 * ox);
            let ip = row_idx.as_ptr().add(2 * ox);
            let v0 = _mm256_loadu_ps(vp);
            let v1 = _mm256_loadu_ps(vp.add(8));
            let i0 = _mm256_castsi256_ps(_mm256_loadu_si256(ip as *const __m256i));
            let i1 = _mm256_castsi256_ps(_mm256_loadu_si256(ip.add(8) as *const __m256i));
            let ev = _mm256_shuffle_ps::<0x88>(v0, v1);
            let od = _mm256_shuffle_ps::<0xDD>(v0, v1);
            let ei = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(i0, i1));
            let oi = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(i0, i1));
            let (m, mi) = beats_indexed8(ev, ei, od, oi, neg1, zero);
            let valid = _mm256_cmpgt_epi32(mi, neg1);
            let flat = _mm256_blendv_epi8(neg1, _mm256_add_epi32(mi, vbase), valid);
            let ord_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(m)));
            let ord_i = _mm256_permute4x64_epi64::<0xD8>(flat);
            _mm256_storeu_ps(dst_v.as_mut_ptr().add(ox), ord_v);
            _mm256_storeu_si256(dst_i.as_mut_ptr().add(ox) as *mut __m256i, ord_i);
            ox += 8;
        }
    }
    for j in ox..ow {
        let mut m = row[2 * j];
        let mut mi = row_idx[2 * j];
        step_idx(&mut m, &mut mi, row[2 * j + 1], row_idx[2 * j + 1]);
        dst_v[j] = m;
        dst_i[j] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

/// Vertical indexed max over `kh` rows of `(rv, ri)`, clamping all-padding `-1` indices to `0` and widening to `i64`.
#[inline]
pub(super) fn max_cols_idx(
    rv: &[f32],
    ri: &[i32],
    ow: usize,
    kh: usize,
    dst_v: &mut [f32],
    dst_i: &mut [i64],
) {
    assert!(
        kh >= 1
            && dst_v.len() == ow
            && dst_i.len() == ow
            && kh.checked_mul(ow).is_some_and(|n| rv.len() >= n && ri.len() >= n)
    );
    #[cfg(target_arch = "x86_64")]
    if ow >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above; bounds asserted above.
        return unsafe { max_cols_idx_avx2(rv, ri, ow, kh, dst_v, dst_i) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if ow >= 4 {
        return super::wasm_simd::max_cols_idx_wasm(rv, ri, ow, kh, dst_v, dst_i);
    }
    for ox in 0..ow {
        let mut m = rv[ox];
        let mut mi = ri[ox];
        for t in 1..kh {
            step_idx(&mut m, &mut mi, rv[t * ow + ox], ri[t * ow + ox]);
        }
        dst_v[ox] = m;
        dst_i[ox] = mi.max(0) as i64;
    }
}

/// # Safety
/// The CPU supports AVX2, `kh >= 1`, `dst_v.len() == dst_i.len() == ow`, and `rv`/`ri` hold at least `kh * ow` entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_cols_idx_avx2(
    rv: &[f32],
    ri: &[i32],
    ow: usize,
    kh: usize,
    dst_v: &mut [f32],
    dst_i: &mut [i64],
) {
    use std::arch::x86_64::*;
    let neg1 = _mm256_set1_epi32(-1);
    let zero = _mm256_setzero_si256();
    let mut ox = 0;
    // SAFETY: `ox + 8 <= ow` and `rv`/`ri` hold at least `kh * ow` entries.
    unsafe {
        while ox + 8 <= ow {
            let mut m = _mm256_loadu_ps(rv.as_ptr().add(ox));
            let mut mi = _mm256_loadu_si256(ri.as_ptr().add(ox) as *const __m256i);
            for t in 1..kh {
                let a = _mm256_loadu_ps(rv.as_ptr().add(t * ow + ox));
                let ai = _mm256_loadu_si256(ri.as_ptr().add(t * ow + ox) as *const __m256i);
                (m, mi) = beats_indexed8(m, mi, a, ai, neg1, zero);
            }
            let clamped = _mm256_max_epi32(mi, zero);
            let lo64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(clamped));
            let hi64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(clamped));
            _mm256_storeu_ps(dst_v.as_mut_ptr().add(ox), m);
            _mm256_storeu_si256(dst_i.as_mut_ptr().add(ox) as *mut __m256i, lo64);
            _mm256_storeu_si256(dst_i.as_mut_ptr().add(ox + 4) as *mut __m256i, hi64);
            ox += 8;
        }
    }
    for j in ox..ow {
        let mut m = rv[j];
        let mut mi = ri[j];
        for t in 1..kh {
            step_idx(&mut m, &mut mi, rv[t * ow + j], ri[t * ow + j]);
        }
        dst_v[j] = m;
        dst_i[j] = mi.max(0) as i64;
    }
}
