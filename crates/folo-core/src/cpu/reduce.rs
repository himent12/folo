//! Reductions and top-k over one dimension of a contiguous tensor viewed as `[outer, d, inner]`.

use std::cmp::Ordering;

use rayon::prelude::*;

use super::{CHUNK, PAR_MIN};
use crate::backend::ReduceOp;
use crate::cpu::par_chunks;

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Sum with 16 independent accumulators: vectorises and loses less precision than a serial sum.
#[inline]
pub(crate) fn sum_slice(x: &[f32]) -> f32 {
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

fn combine(op: ReduceOp, a: f32, b: f32) -> f32 {
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

fn identity(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum => 0.0,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
        _ => unreachable!("arg reductions use reduce_arg"),
    }
}

/// 16-lane scalar twin of [`minmax_slice_avx2`]: exact IEEE-754 NaN propagation and signed-zero order.
#[inline]
fn minmax_slice_scalar(op: ReduceOp, x: &[f32]) -> f32 {
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
                ReduceOp::Sum => {
                    for (acc, &v) in dst.iter_mut().zip(row) {
                        *acc += v;
                    }
                }
                _ => {
                    #[cfg(target_arch = "x86_64")]
                    if dst.len() >= 8 && has_avx2() {
                        // SAFETY: AVX2 was detected above.
                        unsafe { col_minmax_avx2(op, dst, row) };
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

pub(crate) fn reduce_arg(max: bool, x: &[f32], (outer, d, inner): (usize, usize, usize)) -> Vec<i64> {
    let mut out = vec![0i64; outer * inner];
    let better = |v: f32, best: f32| {
        if best.is_nan() {
            false
        } else if v.is_nan() {
            true
        } else if max {
            v > best
        } else {
            v < best
        }
    };
    let row = |(o, dst): (usize, &mut [i64])| {
        let src = &x[o * d * inner..(o + 1) * d * inner];
        for (j, slot) in dst.iter_mut().enumerate() {
            let mut bi = 0usize;
            let mut best = src[j];
            for r in 1..d {
                let v = src[r * inner + j];
                if better(v, best) {
                    best = v;
                    bi = r;
                }
            }
            *slot = bi as i64;
        }
    };
    if outer * d * inner >= PAR_MIN {
        par_chunks(&mut out, inner.max(1), &|i, c| row((i, c)));
    } else {
        out.chunks_mut(inner.max(1)).enumerate().for_each(row);
    }
    out
}

/// Restore max-heap property under `cmp` (so `heap[0]` is the worst of the kept `k` elements).
#[inline]
fn sift_down(
    heap: &mut [(f32, usize)],
    mut i: usize,
    cmp: &impl Fn(&(f32, usize), &(f32, usize)) -> Ordering,
) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let mut worst = left;
        if right < n && cmp(&heap[right], &heap[left]) == Ordering::Greater {
            worst = right;
        }
        if cmp(&heap[worst], &heap[i]) == Ordering::Greater {
            heap.swap(i, worst);
            i = worst;
        } else {
            break;
        }
    }
}

pub(crate) fn topk(x: &[f32], rows: usize, d: usize, k: usize, largest: bool) -> (Vec<f32>, Vec<i64>) {
    if k == 0 || rows == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut vals = super::alloc_uninit(rows * k);
    let mut idx = vec![0i64; rows * k];
    // `total_cmp` orders -0.0 before 0.0; they are equal, so they tie (broken by index) like any other values.
    let key = |v: f32| if v == 0.0 { 0.0 } else { v };
    let cmp = |a: &(f32, usize), b: &(f32, usize)| {
        let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
        o.then(a.1.cmp(&b.1))
    };
    let use_heap = k > 1 && k <= 1024 && k.saturating_mul(4) < d;
    let init_cap = if k == 1 {
        0
    } else if use_heap {
        k
    } else {
        d
    };
    let row = |buf: &mut Vec<(f32, usize)>, r: usize, v: &mut [f32], ix: &mut [i64]| {
        let src = &x[r * d..(r + 1) * d];
        if k == 1 {
            let mut best = (src[0], 0usize);
            for (i, &val) in src.iter().enumerate().skip(1) {
                // Fast reject: since `i > best.1`, any non-NaN `val <= best.0` (or `>=` for smallest)
                // cannot beat `best` even on ties (`-0.0 == +0.0` included). NaN comparisons are false,
                // so NaN falls through to exact `total_cmp`.
                if largest {
                    if val <= best.0 {
                        continue;
                    }
                } else if val >= best.0 {
                    continue;
                }
                if cmp(&(val, i), &best) == Ordering::Less {
                    best = (val, i);
                }
            }
            v[0] = best.0;
            ix[0] = best.1 as i64;
            return;
        }
        buf.clear();
        if use_heap {
            buf.extend(src[..k].iter().copied().zip(0..k));
            for start in (0..k / 2).rev() {
                sift_down(buf, start, &cmp);
            }
            for (i, &val) in src.iter().enumerate().skip(k) {
                if largest {
                    if val <= buf[0].0 {
                        continue;
                    }
                } else if val >= buf[0].0 {
                    continue;
                }
                if cmp(&(val, i), &buf[0]) == Ordering::Less {
                    buf[0] = (val, i);
                    sift_down(buf, 0, &cmp);
                }
            }
            buf.sort_unstable_by(cmp);
        } else {
            buf.extend(src.iter().copied().zip(0..));
            if k < d {
                buf.select_nth_unstable_by(k - 1, cmp);
                buf.truncate(k);
            }
            buf.sort_unstable_by(cmp);
        }
        for (j, &(val, i)) in buf.iter().enumerate() {
            v[j] = val;
            ix[j] = i as i64;
        }
    };
    if rows * d >= PAR_MIN && rows > 1 {
        vals.par_chunks_mut(k)
            .zip(idx.par_chunks_mut(k))
            .enumerate()
            .for_each_init(|| Vec::with_capacity(init_cap), |buf, (r, (v, ix))| row(buf, r, v, ix));
    } else {
        let mut buf = Vec::with_capacity(init_cap);
        for (r, (v, ix)) in vals.chunks_mut(k).zip(idx.chunks_mut(k)).enumerate() {
            row(&mut buf, r, v, ix);
        }
    }
    (vals, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_avx2_matches_scalar_on_nans_signed_zeros_and_tails() {
        let mut xs: Vec<f32> = (0..1037).map(|i| ((i as f32) * 0.37).sin() * 100.0).collect();
        // Exercise clean data (no NaN), signed zeros, infinities, and NaN in early/mid/tail lanes.
        for op in [ReduceOp::Max, ReduceOp::Min] {
            assert_eq!(reduce_block(op, &xs).to_bits(), minmax_slice_scalar(op, &xs).to_bits());
            // Signed zeros only: exact tie behavior.
            let zeros: Vec<f32> = (0..37).map(|i| if i % 2 == 0 { -0.0 } else { 0.0 }).collect();
            assert_eq!(reduce_block(op, &zeros).to_bits(), minmax_slice_scalar(op, &zeros).to_bits());
            let zeros_pos: Vec<f32> = (0..37).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }).collect();
            assert_eq!(reduce_block(op, &zeros_pos).to_bits(), minmax_slice_scalar(op, &zeros_pos).to_bits());
        }
        // NaN in first lane, middle lane, and tail lane.
        for pos in [0, 7, 8, 15, 512, xs.len() - 1] {
            let mut v = xs.clone();
            v[pos] = f32::from_bits(0x7fc0_0000 + pos as u32);
            for op in [ReduceOp::Max, ReduceOp::Min] {
                let got = reduce_block(op, &v);
                let want = minmax_slice_scalar(op, &v);
                assert!(got.is_nan() && want.is_nan());
                assert_eq!(got.to_bits(), want.to_bits(), "{op:?} NaN at {pos}");
            }
        }
        // Column reduction (inner > 1) with NaNs and signed zeros across > 8 columns.
        let (outer, d, inner) = (2, 7, 29);
        let mut grid: Vec<f32> = (0..outer * d * inner).map(|i| ((i as f32) * 0.23).cos() * 5.0).collect();
        grid[inner + 3] = f32::from_bits(0x7fc0_1234);
        grid[3 * inner + 17] = -0.0;
        grid[4 * inner + 17] = 0.0;
        for op in [ReduceOp::Max, ReduceOp::Min] {
            let got = reduce_f32(op, &grid, (outer, d, inner));
            for o in 0..outer {
                for c in 0..inner {
                    let mut want = identity(op);
                    for r in 0..d {
                        want = combine(op, want, grid[(o * d + r) * inner + c]);
                    }
                    assert_eq!(got[o * inner + c].to_bits(), want.to_bits(), "col {op:?} at ({o}, {c})");
                }
            }
        }
        // Keep xs used cleanly.
        xs[0] = f32::INFINITY;
        assert_eq!(reduce_block(ReduceOp::Max, &xs), f32::INFINITY);
    }

    #[test]
    fn topk_bounded_k_matches_full_sort_with_nans_ties_and_signed_zeros() {
        let ref_topk = |x: &[f32], rows: usize, d: usize, k: usize, largest: bool| {
            let key = |v: f32| if v == 0.0 { 0.0 } else { v };
            let cmp = |a: &(f32, usize), b: &(f32, usize)| {
                let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
                o.then(a.1.cmp(&b.1))
            };
            let mut vals = Vec::with_capacity(rows * k);
            let mut idxs = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let mut items: Vec<(f32, usize)> = x[r * d..(r + 1) * d].iter().copied().zip(0..).collect();
                items.sort_unstable_by(cmp);
                for &(v, i) in items.iter().take(k) {
                    vals.push(v);
                    idxs.push(i as i64);
                }
            }
            (vals, idxs)
        };

        // Construct rows containing ties, -0.0/+0.0, ±inf, +NaN, -NaN, and distinct NaN payloads.
        let (rows, d) = (6, 257);
        let mut x: Vec<f32> = (0..rows * d).map(|i| (((i * 37) % 19) as f32) - 9.0).collect();
        let specials = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0x7fc0_0002),
            f32::from_bits(0xffc0_0001), // -NaN
            f32::from_bits(0xffc0_0002), // -NaN
        ];
        for r in 0..rows {
            for (j, &s) in specials.iter().enumerate() {
                x[r * d + (j * 29 + r * 3) % d] = s;
            }
        }
        // Cover k=0, k=1 (zero-alloc scan), k=2/3/10/32 (heap), k=100/257 (select_nth/full).
        for k in [0, 1, 2, 3, 10, 32, 100, d] {
            for largest in [true, false] {
                let (gv, gi) = topk(&x, rows, d, k, largest);
                let (wv, wi) = ref_topk(&x, rows, d, k, largest);
                assert_eq!(gi, wi, "topk idx mismatch (k={k}, largest={largest})");
                assert_eq!(
                    gv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "topk val bits mismatch (k={k}, largest={largest})"
                );
            }
        }
        // Empty rows and parallel rows (> PAR_MIN).
        assert_eq!(topk(&[], 0, 10, 3, true), (vec![], vec![]));
        let (pr, pd, pk) = (64, 600, 12);
        let px: Vec<f32> = (0..pr * pd).map(|i| ((i as f32) * 0.17).sin()).collect();
        let (gv, gi) = topk(&px, pr, pd, pk, true);
        let (wv, wi) = ref_topk(&px, pr, pd, pk, true);
        assert_eq!(gi, wi);
        assert_eq!(
            gv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
