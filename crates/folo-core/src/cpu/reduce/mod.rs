//! Reductions and top-k over one dimension of a contiguous tensor viewed as `[outer, d, inner]`.

mod arg;
#[cfg(test)]
mod tests;
mod topk;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;

pub(crate) use arg::reduce_arg;
pub(crate) use topk::topk;

use rayon::prelude::*;

use super::elementwise::SendPtr;
use super::{CHUNK, PAR_MIN};
use crate::backend::ReduceOp;
use crate::cpu::par_for;

/// Column strip width for `inner > 1` variance reduction (1 KiB per stack accumulator, zero heap allocs).
const VAR_COL_TILE: usize = 256;

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// 16-lane scalar twin of [`sum_slice_avx2`]: independent accumulators lose less precision than a serial sum.
#[inline]
fn sum_slice_scalar(x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for i in 0..16 {
            acc[i] += c[i];
        }
    }
    let mut s = 0f32;
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v;
    }
    for a in acc {
        s += a;
    }
    s
}

/// 16-lane AVX2 sum over a contiguous slice, bitwise identical to [`sum_slice_scalar`].
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_slice_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    // SAFETY: `chunks` yields `&[f32; 16]` (64 bytes), `acc` is `[f32; 16]`, and AVX2 is enabled.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for c in chunks {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(c.as_ptr()));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(c.as_ptr().add(8)));
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), a0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// Sum with 16 independent accumulators: AVX2 on x86-64, bitwise identical to the scalar fallback.
#[inline]
pub(crate) fn sum_slice(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 16 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { sum_slice_avx2(x) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 16 {
        return wasm_simd::sum_slice_wasm(x);
    }
    sum_slice_scalar(x)
}

/// AVX2 column-wise sum `dst[i] += row[i]`, bitwise identical to the scalar loop.
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn col_sum_avx2(dst: &mut [f32], row: &[f32]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), row.len());
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    let (r16, r_rem) = row.as_chunks::<16>();
    for (dc, rc) in d16.iter_mut().zip(r16) {
        // SAFETY: `dc` and `rc` are `[f32; 16]` (64 bytes) and AVX2 is enabled.
        unsafe {
            let a0 = _mm256_loadu_ps(dc.as_ptr());
            let a1 = _mm256_loadu_ps(dc.as_ptr().add(8));
            let v0 = _mm256_loadu_ps(rc.as_ptr());
            let v1 = _mm256_loadu_ps(rc.as_ptr().add(8));
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(a0, v0));
            _mm256_storeu_ps(dc.as_mut_ptr().add(8), _mm256_add_ps(a1, v1));
        }
    }
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    let (r8, r_tail) = r_rem.as_chunks::<8>();
    for (dc, rc) in d8.iter_mut().zip(r8) {
        // SAFETY: `dc` and `rc` are `[f32; 8]` (32 bytes) and AVX2 is enabled.
        unsafe {
            let a = _mm256_loadu_ps(dc.as_ptr());
            let v = _mm256_loadu_ps(rc.as_ptr());
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(a, v));
        }
    }
    for (acc, &v) in d_tail.iter_mut().zip(r_tail) {
        *acc += v;
    }
}

/// Elementwise `dst[i] += src[i]`, AVX2-accelerated and bitwise identical to the scalar loop.
#[inline]
pub(crate) fn col_sum_slice(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { col_sum_avx2(dst, src) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 {
        wasm_simd::col_sum_wasm(dst, src);
        return;
    }
    for (acc, &v) in dst.iter_mut().zip(src) {
        *acc += v;
    }
}

#[inline]
fn sum_sq_slice_scalar(x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for i in 0..16 {
            acc[i] += c[i] * c[i];
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v * v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// 16-lane AVX2+FMA sum of squares `sum(x[i]^2)`.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_slice_avx2_fma(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    // SAFETY: `chunks` yields `&[f32; 16]`, `acc` is `[f32; 16]`, and AVX2+FMA are enabled.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for c in chunks {
            let v0 = _mm256_loadu_ps(c.as_ptr());
            let v1 = _mm256_loadu_ps(c.as_ptr().add(8));
            a0 = _mm256_fmadd_ps(v0, v0, a0);
            a1 = _mm256_fmadd_ps(v1, v1, a1);
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), a0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v * v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// Sum of squares over one block (`<= REDUCE_BLOCK`), AVX2+FMA accelerated.
#[inline]
pub(crate) fn sum_sq_slice(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 16 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        return unsafe { sum_sq_slice_avx2_fma(x) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 16 {
        return wasm_simd::sum_sq_slice_wasm(x);
    }
    sum_sq_slice_scalar(x)
}

#[inline]
fn sum_sq_diff_slice_scalar(x: &[f32], mean: f32) -> f32 {
    let mut acc = [0f32; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for i in 0..16 {
            let d = c[i] - mean;
            acc[i] += d * d;
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        let d = v - mean;
        acc[i] += d * d;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// 16-lane AVX2+FMA sum of squared deviations `sum((x[i] - mean)^2)`.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq_diff_slice_avx2_fma(x: &[f32], mean: f32) -> f32 {
    use std::arch::x86_64::*;
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    // SAFETY: `chunks` yields `&[f32; 16]`, `acc` is `[f32; 16]`, and AVX2+FMA are enabled.
    unsafe {
        let vm = _mm256_set1_ps(mean);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        for c in chunks {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), vm);
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(c.as_ptr().add(8)), vm);
            a0 = _mm256_fmadd_ps(d0, d0, a0);
            a1 = _mm256_fmadd_ps(d1, d1, a1);
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), a0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
    }
    for (i, &v) in rest.iter().enumerate() {
        let d = v - mean;
        acc[i] += d * d;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// Sum of squared deviations `sum((x[i] - mean)^2)` over one block, AVX2+FMA accelerated.
#[inline]
pub(crate) fn sum_sq_diff_slice(x: &[f32], mean: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 16 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        return unsafe { sum_sq_diff_slice_avx2_fma(x, mean) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 16 {
        return wasm_simd::sum_sq_diff_slice_wasm(x, mean);
    }
    sum_sq_diff_slice_scalar(x, mean)
}

/// AVX2+FMA column-wise squared deviation update `dst[i] += (row[i] - mean[i])^2`.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn col_sq_diff_avx2_fma(dst: &mut [f32], row: &[f32], mean: &[f32]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), row.len());
    assert_eq!(dst.len(), mean.len());
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    let (r16, r_rem) = row.as_chunks::<16>();
    let (m16, m_rem) = mean.as_chunks::<16>();
    for ((dc, rc), mc) in d16.iter_mut().zip(r16).zip(m16) {
        // SAFETY: all chunks are `[f32; 16]` (64 bytes) and AVX2+FMA are enabled.
        unsafe {
            let a0 = _mm256_loadu_ps(dc.as_ptr());
            let a1 = _mm256_loadu_ps(dc.as_ptr().add(8));
            let df0 = _mm256_sub_ps(_mm256_loadu_ps(rc.as_ptr()), _mm256_loadu_ps(mc.as_ptr()));
            let df1 = _mm256_sub_ps(_mm256_loadu_ps(rc.as_ptr().add(8)), _mm256_loadu_ps(mc.as_ptr().add(8)));
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_fmadd_ps(df0, df0, a0));
            _mm256_storeu_ps(dc.as_mut_ptr().add(8), _mm256_fmadd_ps(df1, df1, a1));
        }
    }
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    let (r8, r_tail) = r_rem.as_chunks::<8>();
    let (m8, m_tail) = m_rem.as_chunks::<8>();
    for ((dc, rc), mc) in d8.iter_mut().zip(r8).zip(m8) {
        // SAFETY: all chunks are `[f32; 8]` (32 bytes) and AVX2+FMA are enabled.
        unsafe {
            let a = _mm256_loadu_ps(dc.as_ptr());
            let df = _mm256_sub_ps(_mm256_loadu_ps(rc.as_ptr()), _mm256_loadu_ps(mc.as_ptr()));
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_fmadd_ps(df, df, a));
        }
    }
    for ((acc, &v), &m) in d_tail.iter_mut().zip(r_tail).zip(m_tail) {
        let df = v - m;
        *acc += df * df;
    }
}

#[inline]
fn col_sq_diff_slice(dst: &mut [f32], row: &[f32], mean: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        unsafe { col_sq_diff_avx2_fma(dst, row, mean) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 {
        wasm_simd::col_sq_diff_wasm(dst, row, mean);
        return;
    }
    for ((acc, &v), &m) in dst.iter_mut().zip(row).zip(mean) {
        let df = v - m;
        *acc += df * df;
    }
}

/// `(outer, d, inner)` of a shape around `dim`.
pub(crate) fn split3(dims: &[usize], dim: usize) -> (usize, usize, usize) {
    (dims[..dim].iter().product(), dims[dim], dims[dim + 1..].iter().product())
}

/// Rows longer than this are reduced as fixed blocks of this many elements whose partial results are combined
/// by [`tree_combine`]. The blocking depends only on the row length, so the result is bitwise the same for any
/// thread count and from run to run.
pub(crate) const REDUCE_BLOCK: usize = 1 << 13;

/// Combine `v` pairwise in a fixed order: `((v0 ∘ v1) ∘ (v2 ∘ v3)) ∘ ...`. Precision degrades with the log of
/// the length instead of the length.
pub(crate) fn tree_combine(mut v: Vec<f32>, comb: impl Fn(f32, f32) -> f32) -> Option<f32> {
    while v.len() > 1 {
        let half = v.len().div_ceil(2);
        for i in 0..half {
            v[i] = if 2 * i + 1 < v.len() { comb(v[2 * i], v[2 * i + 1]) } else { v[2 * i] };
        }
        v.truncate(half);
    }
    v.pop()
}

pub(super) fn combine(op: ReduceOp, a: f32, b: f32) -> f32 {
    match op {
        ReduceOp::Sum => a + b,
        ReduceOp::Max => {
            if b > a || b.is_nan() {
                b
            } else {
                a
            }
        }
        _ => {
            if b < a || b.is_nan() {
                b
            } else {
                a
            }
        }
    }
}

pub(super) fn identity(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum => 0.0,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
        _ => unreachable!("arg reductions use reduce_arg"),
    }
}

/// 16-lane scalar twin of [`minmax_slice_avx2`]: exact IEEE-754 NaN propagation and signed-zero order.
#[inline]
pub(super) fn minmax_slice_scalar(op: ReduceOp, x: &[f32]) -> f32 {
    let id = identity(op);
    let mut acc = [id; 16];
    let (chunks, rest) = x.as_chunks::<16>();
    for c in chunks {
        for i in 0..16 {
            acc[i] = combine(op, acc[i], c[i]);
        }
    }
    let mut res = id;
    for a in acc {
        res = combine(op, res, a);
    }
    for &v in rest {
        res = combine(op, res, v);
    }
    res
}

/// 16-way AVX2 Max/Min over a contiguous slice, bitwise identical to [`minmax_slice_scalar`].
///
/// `vmaxps(v, acc)` / `vminps(v, acc)` (with `v` first, `acc` second) returns `v > acc ? v : acc` /
/// `v < acc ? v : acc`, matching [`combine`] bitwise on every non-NaN pair (including `-0.0` vs `+0.0`).
/// A single `vcmpunordps(v0, v1)` tracks whether any lane is NaN; if so we fall back to
/// [`minmax_slice_scalar`] to preserve exact NaN propagation and payload.
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn minmax_slice_avx2(op: ReduceOp, x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let max = op == ReduceOp::Max;
    let id = identity(op);
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [id; 16];
    // SAFETY: `chunks` yields `&[f32; 16]` (64 bytes), `acc` is `[f32; 16]` (64 bytes), and AVX2 is enabled.
    let found_nan = unsafe {
        let mut a0 = _mm256_set1_ps(id);
        let mut a1 = _mm256_set1_ps(id);
        let mut any_nan = _mm256_setzero_ps();
        for c in chunks {
            let v0 = _mm256_loadu_ps(c.as_ptr());
            let v1 = _mm256_loadu_ps(c.as_ptr().add(8));
            any_nan = _mm256_or_ps(any_nan, _mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v1));
            if max {
                a0 = _mm256_max_ps(v0, a0);
                a1 = _mm256_max_ps(v1, a1);
            } else {
                a0 = _mm256_min_ps(v0, a0);
                a1 = _mm256_min_ps(v1, a1);
            }
        }
        if _mm256_movemask_ps(any_nan) != 0 {
            true
        } else {
            _mm256_storeu_ps(acc.as_mut_ptr(), a0);
            _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
            false
        }
    };
    if found_nan {
        return minmax_slice_scalar(op, x);
    }
    let mut res = id;
    for a in acc {
        res = combine(op, res, a);
    }
    for &v in rest {
        res = combine(op, res, v);
    }
    res
}

/// Branchless AVX2 column-wise Max/Min update `dst[i] = combine(op, dst[i], row[i])`, bitwise identical to
/// [`combine`] on every `(f32, f32)` pair (including NaN payloads and signed zeros).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn col_minmax_avx2(op: ReduceOp, dst: &mut [f32], row: &[f32]) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), row.len());
    let max = op == ReduceOp::Max;
    let (d_chunks, d_rem) = dst.as_chunks_mut::<8>();
    let (r_chunks, r_rem) = row.as_chunks::<8>();
    for (dc, rc) in d_chunks.iter_mut().zip(r_chunks) {
        // SAFETY: `dc` and `rc` are `[f32; 8]` (32 bytes) and AVX2 is enabled.
        unsafe {
            let a = _mm256_loadu_ps(dc.as_ptr());
            let v = _mm256_loadu_ps(rc.as_ptr());
            let m = if max { _mm256_max_ps(v, a) } else { _mm256_min_ps(v, a) };
            let v_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v);
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_blendv_ps(m, v, v_nan));
        }
    }
    for (acc, &v) in d_rem.iter_mut().zip(r_rem) {
        *acc = combine(op, *acc, v);
    }
}

/// One block (at most [`REDUCE_BLOCK`] long) reduced serially.
fn reduce_block(op: ReduceOp, x: &[f32]) -> f32 {
    match op {
        ReduceOp::Sum => sum_slice(x),
        ReduceOp::Max | ReduceOp::Min => {
            #[cfg(target_arch = "x86_64")]
            if x.len() >= 16 && has_avx2() {
                // SAFETY: AVX2 was detected above.
                return unsafe { minmax_slice_avx2(op, x) };
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if x.len() >= 16 {
                return wasm_simd::minmax_slice_wasm(op, x);
            }
            minmax_slice_scalar(op, x)
        }
        _ => unreachable!("arg reductions use reduce_arg"),
    }
}

/// Reduce a contiguous row with the fixed blocking described at [`REDUCE_BLOCK`].
fn reduce_row(op: ReduceOp, x: &[f32]) -> f32 {
    if x.len() <= REDUCE_BLOCK {
        return reduce_block(op, x);
    }
    let parts: Vec<f32> = if x.len() >= PAR_MIN {
        x.par_chunks(REDUCE_BLOCK).map(|c| reduce_block(op, c)).collect()
    } else {
        x.chunks(REDUCE_BLOCK).map(|c| reduce_block(op, c)).collect()
    };
    tree_combine(parts, |a, b| combine(op, a, b)).unwrap_or(identity(op))
}

/// Deterministic sum of a row using fixed [`REDUCE_BLOCK`] chunks and [`tree_combine`].
pub(crate) fn sum_row(x: &[f32]) -> f32 {
    reduce_row(ReduceOp::Sum, x)
}

/// Deterministic sum of squares of a row using fixed [`REDUCE_BLOCK`] chunks and [`tree_combine`].
pub(crate) fn sum_sq_row(x: &[f32]) -> f32 {
    if x.len() <= REDUCE_BLOCK {
        return sum_sq_slice(x);
    }
    let parts: Vec<f32> = if x.len() >= PAR_MIN {
        x.par_chunks(REDUCE_BLOCK).map(sum_sq_slice).collect()
    } else {
        x.chunks(REDUCE_BLOCK).map(sum_sq_slice).collect()
    };
    tree_combine(parts, |a, b| a + b).unwrap_or(0.0)
}

/// Deterministic sum of squared deviations `sum((x[i] - mean)^2)` of a row using fixed [`REDUCE_BLOCK`] chunks.
pub(crate) fn sum_sq_diff_row(x: &[f32], mean: f32) -> f32 {
    if x.len() <= REDUCE_BLOCK {
        return sum_sq_diff_slice(x, mean);
    }
    let parts: Vec<f32> = if x.len() >= PAR_MIN {
        x.par_chunks(REDUCE_BLOCK).map(|c| sum_sq_diff_slice(c, mean)).collect()
    } else {
        x.chunks(REDUCE_BLOCK).map(|c| sum_sq_diff_slice(c, mean)).collect()
    };
    tree_combine(parts, |a, b| a + b).unwrap_or(0.0)
}

/// Reduce the middle dimension of `[outer, d, inner]`. Every output element is computed by one task in an order
/// that depends only on the shape, so results are bitwise reproducible across runs and thread counts.
pub(crate) fn reduce_f32(op: ReduceOp, x: &[f32], (outer, d, inner): (usize, usize, usize)) -> Vec<f32> {
    let mut out = super::alloc_uninit(outer * inner);
    if outer * inner == 0 {
        return out;
    }
    let par = outer * d * inner >= PAR_MIN;
    if inner == 1 {
        let row = |(o, dst): (usize, &mut f32)| *dst = reduce_row(op, &x[o * d..(o + 1) * d]);
        if par && outer > 1 {
            out.par_iter_mut().enumerate().with_min_len((CHUNK / d.max(1)).max(1)).for_each(row);
        } else {
            out.iter_mut().enumerate().for_each(row);
        }
        return out;
    }
    // Columns: each task owns a run of output columns of one outer block and walks the d rows in order.
    let cols = if par { (CHUNK / d.max(1)).clamp(64.min(inner), inner) } else { inner };
    let per = inner.div_ceil(cols);
    let task = |(t, dst): (usize, &mut [f32])| {
        let (o, c0) = (t / per, (t % per) * cols);
        let src = &x[o * d * inner..(o + 1) * d * inner];
        dst.fill(identity(op));
        for r in 0..d {
            let row = &src[r * inner + c0..r * inner + c0 + dst.len()];
            match op {
                ReduceOp::Sum => col_sum_slice(dst, row),
                _ => {
                    #[cfg(target_arch = "x86_64")]
                    if dst.len() >= 8 && has_avx2() {
                        // SAFETY: AVX2 was detected above.
                        unsafe { col_minmax_avx2(op, dst, row) };
                        continue;
                    }
                    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                    if dst.len() >= 4 {
                        wasm_simd::col_minmax_wasm(op, dst, row);
                        continue;
                    }
                    for (acc, &v) in dst.iter_mut().zip(row) {
                        *acc = combine(op, *acc, v);
                    }
                }
            }
        }
    };
    // Chunks of `cols` within each block of `inner` outputs (the last chunk of a block may be shorter).
    let chunks = out.chunks_mut(inner).flat_map(|b| b.chunks_mut(cols));
    if par {
        chunks.enumerate().collect::<Vec<_>>().into_par_iter().for_each(task);
    } else {
        chunks.enumerate().for_each(task);
    }
    out
}

/// Two-pass variance over the middle dimension of `[outer, d, inner]` (biased or Bessel-corrected).
/// Bitwise reproducible across thread counts: `inner == 1` uses [`REDUCE_BLOCK`] tree reductions, and `inner > 1`
/// tiles columns in fixed [`VAR_COL_TILE`]-wide stack buffers.
pub(crate) fn reduce_var_f32(
    x: &[f32],
    (outer, d, inner): (usize, usize, usize),
    unbiased: bool,
) -> Vec<f32> {
    let mut out = super::alloc_uninit(outer * inner);
    if outer * inner == 0 {
        return out;
    }
    let denom = if unbiased { d.saturating_sub(1) } else { d };
    if denom == 0 {
        out.fill(f32::NAN);
        return out;
    }
    let inv_d = 1.0 / d as f32;
    let inv_denom = 1.0 / denom as f32;
    let par = outer * d * inner >= PAR_MIN;
    if inner == 1 {
        let do_row = |o: usize| {
            let row = &x[o * d..(o + 1) * d];
            let mean = sum_row(row) * inv_d;
            sum_sq_diff_row(row, mean) * inv_denom
        };
        if par && outer > 1 {
            let out_ptr = SendPtr(out.as_mut_ptr());
            let chunk = (CHUNK / d.max(1)).max(1);
            let tasks = outer.div_ceil(chunk);
            par_for(tasks, &|t| {
                let start = t * chunk;
                let end = (start + chunk).min(outer);
                for o in start..end {
                    let v = do_row(o);
                    // SAFETY: tasks cover disjoint ranges `start..end` within `0..outer`.
                    unsafe { *out_ptr.ptr().add(o) = v };
                }
            });
        } else {
            for (o, dst) in out.iter_mut().enumerate() {
                *dst = do_row(o);
            }
        }
        return out;
    }
    let tiles_per_outer = inner.div_ceil(VAR_COL_TILE);
    let total_tiles = outer * tiles_per_outer;
    let out_ptr = SendPtr(out.as_mut_ptr());
    let do_tile = |t: usize| {
        let o = t / tiles_per_outer;
        let c0 = (t % tiles_per_outer) * VAR_COL_TILE;
        let len = (inner - c0).min(VAR_COL_TILE);
        let src = &x[o * d * inner..(o + 1) * d * inner];
        let mut mean = [0.0f32; VAR_COL_TILE];
        let mut sq = [0.0f32; VAR_COL_TILE];
        let m = &mut mean[..len];
        let s = &mut sq[..len];
        for r in 0..d {
            col_sum_slice(m, &src[r * inner + c0..r * inner + c0 + len]);
        }
        for v in m.iter_mut() {
            *v *= inv_d;
        }
        for r in 0..d {
            col_sq_diff_slice(s, &src[r * inner + c0..r * inner + c0 + len], m);
        }
        // SAFETY: each tile `(o, c0..c0+len)` owns a disjoint slice of `out[0..outer * inner]`.
        let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.ptr().add(o * inner + c0), len) };
        for (y, &v) in dst.iter_mut().zip(s.iter()) {
            *y = v * inv_denom;
        }
    };
    if par && total_tiles > 1 {
        par_for(total_tiles, &do_tile);
    } else {
        for t in 0..total_tiles {
            do_tile(t);
        }
    }
    out
}
