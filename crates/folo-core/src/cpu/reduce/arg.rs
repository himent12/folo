//! AVX2 row and column-tiled `ArgMax` / `ArgMin` reductions over `[outer, d, inner]`, bitwise identical to
//! the scalar reference on all inputs (first-maximum/minimum tie-breaking, `-0.0 == +0.0`, `±∞`, and
//! first-NaN precedence).

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use super::has_avx2;
use crate::cpu::elementwise::SendPtr;
use crate::cpu::{CHUNK, PAR_MIN, par_for};

/// Column strip width for `inner > 1` arg-reductions (`256 * (4 + 4) = 2 KiB` in L1, zero heap allocs).
pub(super) const ARG_COL_TILE: usize = 256;

#[inline(always)]
pub(super) fn better(max: bool, v: f32, best: f32) -> bool {
    if best.is_nan() {
        false
    } else if v.is_nan() {
        true
    } else if max {
        v > best
    } else {
        v < best
    }
}

#[inline]
pub(super) fn arg_row_scalar(max: bool, row: &[f32]) -> i64 {
    let mut bi = 0usize;
    let mut best = row[0];
    for (r, &v) in row.iter().enumerate().skip(1) {
        if better(max, v, best) {
            best = v;
            bi = r;
        }
    }
    bi as i64
}

/// 16-lane AVX2 row `ArgMax` / `ArgMin` (`inner == 1`), bitwise identical to [`arg_row_scalar`].
///
/// Each lane tracks `(best_val, best_idx)` in strictly increasing index order (`lane, lane+16, ...`). Once a
/// lane encounters a NaN, `b_ord` goes to zero so it locks onto its earliest NaN index. Merging the 16 lanes
/// picks the NaN lane with the minimum index if any lane is NaN, otherwise the best value breaking ties by
/// minimum index — mathematically identical to a left-to-right serial scan.
///
/// # Safety
/// The CPU supports AVX2 and `16 <= row.len() <= i32::MAX as usize`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn arg_row_avx2(max: bool, row: &[f32]) -> i64 {
    use std::arch::x86_64::*;
    debug_assert!(row.len() >= 16 && row.len() <= i32::MAX as usize);
    let (chunks, tail) = row.as_chunks::<16>();
    let mut best_vals = [0f32; 16];
    let mut best_idxs = [0i32; 16];
    // SAFETY: `chunks` is non-empty (`row.len() >= 16`), each chunk is `[f32; 16]`, and AVX2 is enabled.
    unsafe {
        let first = chunks[0].as_ptr();
        let mut b0 = _mm256_loadu_ps(first);
        let mut b1 = _mm256_loadu_ps(first.add(8));
        let mut i0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let mut i1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        let step = _mm256_set1_epi32(16);
        let mut cur0 = _mm256_add_epi32(i0, step);
        let mut cur1 = _mm256_add_epi32(i1, step);
        for c in &chunks[1..] {
            let p = c.as_ptr();
            let v0 = _mm256_loadu_ps(p);
            let v1 = _mm256_loadu_ps(p.add(8));
            let cmp0 =
                if max { _mm256_cmp_ps::<_CMP_GT_OQ>(v0, b0) } else { _mm256_cmp_ps::<_CMP_LT_OQ>(v0, b0) };
            let cmp1 =
                if max { _mm256_cmp_ps::<_CMP_GT_OQ>(v1, b1) } else { _mm256_cmp_ps::<_CMP_LT_OQ>(v1, b1) };
            let take0 = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_ORD_Q>(b0, b0),
                _mm256_or_ps(cmp0, _mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v0)),
            );
            let take1 = _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_ORD_Q>(b1, b1),
                _mm256_or_ps(cmp1, _mm256_cmp_ps::<_CMP_UNORD_Q>(v1, v1)),
            );
            b0 = _mm256_blendv_ps(b0, v0, take0);
            b1 = _mm256_blendv_ps(b1, v1, take1);
            i0 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(i0),
                _mm256_castsi256_ps(cur0),
                take0,
            ));
            i1 = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(i1),
                _mm256_castsi256_ps(cur1),
                take1,
            ));
            cur0 = _mm256_add_epi32(cur0, step);
            cur1 = _mm256_add_epi32(cur1, step);
        }
        _mm256_storeu_ps(best_vals.as_mut_ptr(), b0);
        _mm256_storeu_ps(best_vals.as_mut_ptr().add(8), b1);
        _mm256_storeu_si256(best_idxs.as_mut_ptr() as *mut __m256i, i0);
        _mm256_storeu_si256(best_idxs.as_mut_ptr().add(8) as *mut __m256i, i1);
    }

    let mut best = best_vals[0];
    let mut bi = best_idxs[0] as usize;
    for k in 1..16 {
        let v = best_vals[k];
        let idx = best_idxs[k] as usize;
        let wins = if best.is_nan() {
            v.is_nan() && idx < bi
        } else if v.is_nan() {
            true
        } else if max {
            v > best || (v == best && idx < bi)
        } else {
            v < best || (v == best && idx < bi)
        };
        if wins {
            best = v;
            bi = idx;
        }
    }

    let base = chunks.len() * 16;
    for (off, &v) in tail.iter().enumerate() {
        if better(max, v, best) {
            best = v;
            bi = base + off;
        }
    }
    bi as i64
}

#[inline]
fn arg_row(max: bool, row: &[f32]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if row.len() >= 16 && row.len() <= i32::MAX as usize && has_avx2() {
        // SAFETY: AVX2 detected and `16 <= row.len() <= i32::MAX`.
        return unsafe { arg_row_avx2(max, row) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if row.len() >= 16 && row.len() <= i32::MAX as usize {
        return super::wasm_simd::arg_row_wasm(max, row);
    }
    arg_row_scalar(max, row)
}

fn arg_col_tile_scalar(max: bool, src: &[f32], dst: &mut [i64], d: usize, inner: usize, c0: usize) {
    let tw = dst.len();
    let mut best = [0f32; ARG_COL_TILE];
    best[..tw].copy_from_slice(&src[c0..c0 + tw]);
    dst.fill(0);
    for r in 1..d {
        let row = &src[r * inner + c0..r * inner + c0 + tw];
        for j in 0..tw {
            let v = row[j];
            if better(max, v, best[j]) {
                best[j] = v;
                dst[j] = r as i64;
            }
        }
    }
}

/// 16-wide AVX2 column-tiled `ArgMax` / `ArgMin` for `tw <= ARG_COL_TILE` columns, bitwise identical to
/// [`arg_col_tile_scalar`].
///
/// # Safety
/// The CPU supports AVX2 and `d <= i32::MAX as usize` and `dst.len() <= ARG_COL_TILE`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn arg_col_tile_avx2(max: bool, src: &[f32], dst: &mut [i64], d: usize, inner: usize, c0: usize) {
    use std::arch::x86_64::*;
    let tw = dst.len();
    debug_assert!(tw <= ARG_COL_TILE && d <= i32::MAX as usize);
    let mut best_v = [0f32; ARG_COL_TILE];
    let mut best_i = [0i32; ARG_COL_TILE];
    best_v[..tw].copy_from_slice(&src[c0..c0 + tw]);

    let v16 = tw / 16 * 16;
    let v8 = tw / 8 * 8;
    for r in 1..d {
        let row = &src[r * inner + c0..r * inner + c0 + tw];
        let vr = _mm256_set1_epi32(r as i32);
        let vr_ps = _mm256_castsi256_ps(vr);
        let mut j = 0;
        // SAFETY: `j + 16 <= v16 <= tw <= ARG_COL_TILE` and AVX2 is enabled.
        unsafe {
            while j < v16 {
                let rp = row.as_ptr().add(j);
                let bp = best_v.as_mut_ptr().add(j);
                let ip = best_i.as_mut_ptr().add(j);
                let v0 = _mm256_loadu_ps(rp);
                let v1 = _mm256_loadu_ps(rp.add(8));
                let b0 = _mm256_loadu_ps(bp);
                let b1 = _mm256_loadu_ps(bp.add(8));
                let cmp0 = if max {
                    _mm256_cmp_ps::<_CMP_GT_OQ>(v0, b0)
                } else {
                    _mm256_cmp_ps::<_CMP_LT_OQ>(v0, b0)
                };
                let cmp1 = if max {
                    _mm256_cmp_ps::<_CMP_GT_OQ>(v1, b1)
                } else {
                    _mm256_cmp_ps::<_CMP_LT_OQ>(v1, b1)
                };
                let take0 = _mm256_and_ps(
                    _mm256_cmp_ps::<_CMP_ORD_Q>(b0, b0),
                    _mm256_or_ps(cmp0, _mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v0)),
                );
                let take1 = _mm256_and_ps(
                    _mm256_cmp_ps::<_CMP_ORD_Q>(b1, b1),
                    _mm256_or_ps(cmp1, _mm256_cmp_ps::<_CMP_UNORD_Q>(v1, v1)),
                );
                _mm256_storeu_ps(bp, _mm256_blendv_ps(b0, v0, take0));
                _mm256_storeu_ps(bp.add(8), _mm256_blendv_ps(b1, v1, take1));
                let i0 = _mm256_loadu_si256(ip as *const __m256i);
                let i1 = _mm256_loadu_si256(ip.add(8) as *const __m256i);
                _mm256_storeu_si256(
                    ip as *mut __m256i,
                    _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(i0), vr_ps, take0)),
                );
                _mm256_storeu_si256(
                    ip.add(8) as *mut __m256i,
                    _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(i1), vr_ps, take1)),
                );
                j += 16;
            }
            if j < v8 {
                let v0 = _mm256_loadu_ps(row.as_ptr().add(j));
                let b0 = _mm256_loadu_ps(best_v.as_ptr().add(j));
                let cmp0 = if max {
                    _mm256_cmp_ps::<_CMP_GT_OQ>(v0, b0)
                } else {
                    _mm256_cmp_ps::<_CMP_LT_OQ>(v0, b0)
                };
                let take0 = _mm256_and_ps(
                    _mm256_cmp_ps::<_CMP_ORD_Q>(b0, b0),
                    _mm256_or_ps(cmp0, _mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v0)),
                );
                _mm256_storeu_ps(best_v.as_mut_ptr().add(j), _mm256_blendv_ps(b0, v0, take0));
                let i0 = _mm256_loadu_si256(best_i.as_ptr().add(j) as *const __m256i);
                _mm256_storeu_si256(
                    best_i.as_mut_ptr().add(j) as *mut __m256i,
                    _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(i0), vr_ps, take0)),
                );
            }
        }
        for jj in v8..tw {
            let v = row[jj];
            if better(max, v, best_v[jj]) {
                best_v[jj] = v;
                best_i[jj] = r as i32;
            }
        }
    }

    // Widen `best_i[..tw]` (`i32` -> `i64`) into `dst[..tw]`.
    let mut j = 0;
    // SAFETY: `j + 8 <= v8 <= tw == dst.len()` and `tw <= ARG_COL_TILE`.
    unsafe {
        while j < v8 {
            let idx8 = _mm256_loadu_si256(best_i.as_ptr().add(j) as *const __m256i);
            let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(idx8));
            let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(idx8));
            _mm256_storeu_si256(dst.as_mut_ptr().add(j) as *mut __m256i, lo);
            _mm256_storeu_si256(dst.as_mut_ptr().add(j + 4) as *mut __m256i, hi);
            j += 8;
        }
    }
    for jj in v8..tw {
        dst[jj] = best_i[jj] as i64;
    }
}

fn arg_col_tile(max: bool, src: &[f32], dst: &mut [i64], d: usize, inner: usize, c0: usize) {
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && d <= i32::MAX as usize && has_avx2() {
        // SAFETY: AVX2 detected, `dst.len() <= ARG_COL_TILE`, and `d <= i32::MAX`.
        return unsafe { arg_col_tile_avx2(max, src, dst, d, inner, c0) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 && d <= i32::MAX as usize {
        return super::wasm_simd::arg_col_tile_wasm(max, src, dst, d, inner, c0);
    }
    arg_col_tile_scalar(max, src, dst, d, inner, c0);
}

/// `ArgMax` (`max == true`) or `ArgMin` (`max == false`) over the middle dim of `[outer, d, inner]`.
pub(crate) fn reduce_arg(max: bool, x: &[f32], (outer, d, inner): (usize, usize, usize)) -> Vec<i64> {
    let mut out = vec![0i64; outer * inner];
    if out.is_empty() || d <= 1 {
        return out;
    }
    let total = outer * d * inner;
    if inner == 1 {
        let do_row = |(o, slot): (usize, &mut i64)| *slot = arg_row(max, &x[o * d..(o + 1) * d]);
        if total >= PAR_MIN && outer > 1 {
            out.par_iter_mut().enumerate().with_min_len((CHUNK / d.max(1)).max(1)).for_each(do_row);
        } else {
            out.iter_mut().enumerate().for_each(do_row);
        }
        return out;
    }
    let tiles_per = inner.div_ceil(ARG_COL_TILE);
    let total_tasks = outer * tiles_per;
    let optr = SendPtr(out.as_mut_ptr());
    let run = |t: usize| {
        let optr = &optr;
        let (o, ct) = (t / tiles_per, t % tiles_per);
        let c0 = ct * ARG_COL_TILE;
        let tw = ARG_COL_TILE.min(inner - c0);
        let src = &x[o * d * inner..(o + 1) * d * inner];
        // SAFETY: each `(o, ct)` owns disjoint columns `c0..c0 + tw` of row `o` in `out`.
        let dst = unsafe { std::slice::from_raw_parts_mut(optr.ptr().add(o * inner + c0), tw) };
        arg_col_tile(max, src, dst, d, inner, c0);
    };
    if total >= PAR_MIN && total_tasks > 1 {
        par_for(total_tasks, &run);
    } else {
        (0..total_tasks).for_each(run);
    }
    out
}
