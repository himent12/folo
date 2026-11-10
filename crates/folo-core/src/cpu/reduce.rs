//! Reductions, softmax and top-k over one dimension of a contiguous tensor viewed as `[outer, d, inner]`.

use rayon::prelude::*;

use super::elementwise::fast_exp;
use super::{CHUNK, PAR_MIN};
use crate::backend::ReduceOp;

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

/// One block (at most [`REDUCE_BLOCK`] long) reduced serially.
fn reduce_block(op: ReduceOp, x: &[f32]) -> f32 {
    match op {
        ReduceOp::Sum => sum_slice(x),
        _ => x.iter().fold(identity(op), |a, &b| combine(op, a, b)),
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
        out.par_chunks_mut(inner.max(1)).enumerate().for_each(row);
    } else {
        out.chunks_mut(inner.max(1)).enumerate().for_each(row);
    }
    out
}

pub(crate) fn softmax(x: &[f32], (outer, d, inner): (usize, usize, usize), log: bool) -> Vec<f32> {
    let mut out = super::alloc_uninit(x.len());
    if out.is_empty() {
        return out;
    }
    let block = |(o, dst): (usize, &mut [f32])| {
        let src = &x[o * d * inner..(o + 1) * d * inner];
        if inner == 1 {
            let m = src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let m = if m.is_finite() { m } else { 0.0 };
            for (y, &v) in dst.iter_mut().zip(src) {
                *y = fast_exp(v - m);
            }
            let s = sum_slice(dst);
            if log {
                let ls = s.ln();
                for (y, &v) in dst.iter_mut().zip(src) {
                    *y = (v - m) - ls;
                }
            } else {
                let inv = 1.0 / s;
                for y in dst.iter_mut() {
                    *y *= inv;
                }
            }
        } else {
            let mut m = vec![f32::NEG_INFINITY; inner];
            for r in 0..d {
                for (a, &v) in m.iter_mut().zip(&src[r * inner..(r + 1) * inner]) {
                    *a = a.max(v);
                }
            }
            for a in m.iter_mut() {
                if !a.is_finite() {
                    *a = 0.0;
                }
            }
            let mut s = vec![0f32; inner];
            for r in 0..d {
                let row = &mut dst[r * inner..(r + 1) * inner];
                for (((y, &v), &mx), acc) in
                    row.iter_mut().zip(&src[r * inner..(r + 1) * inner]).zip(&m).zip(&mut s)
                {
                    *y = fast_exp(v - mx);
                    *acc += *y;
                }
            }
            for r in 0..d {
                let row = &mut dst[r * inner..(r + 1) * inner];
                if log {
                    for (((y, &v), &mx), &acc) in
                        row.iter_mut().zip(&src[r * inner..(r + 1) * inner]).zip(&m).zip(&s)
                    {
                        *y = v - mx - acc.ln();
                    }
                } else {
                    for (y, &acc) in row.iter_mut().zip(&s) {
                        *y /= acc;
                    }
                }
            }
        }
    };
    if x.len() >= PAR_MIN && outer > 1 {
        out.par_chunks_mut(d * inner).enumerate().for_each(block);
    } else {
        out.chunks_mut(d * inner).enumerate().for_each(block);
    }
    out
}

pub(crate) fn topk(x: &[f32], rows: usize, d: usize, k: usize, largest: bool) -> (Vec<f32>, Vec<i64>) {
    let mut vals = vec![0f32; rows * k];
    let mut idx = vec![0i64; rows * k];
    if k == 0 {
        return (vals, idx);
    }
    // `total_cmp` orders -0.0 before 0.0; they are equal, so they tie (broken by index) like any other values.
    let key = |v: f32| if v == 0.0 { 0.0 } else { v };
    let cmp = |a: &(f32, usize), b: &(f32, usize)| {
        let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
        o.then(a.1.cmp(&b.1))
    };
    let row = |(r, (v, ix)): (usize, (&mut [f32], &mut [i64]))| {
        let src = &x[r * d..(r + 1) * d];
        let mut items: Vec<(f32, usize)> = src.iter().copied().zip(0..).collect();
        if k < d {
            items.select_nth_unstable_by(k - 1, cmp);
            items.truncate(k);
        }
        items.sort_unstable_by(cmp);
        for (j, &(val, i)) in items.iter().enumerate() {
            v[j] = val;
            ix[j] = i as i64;
        }
    };
    if rows * d >= PAR_MIN && rows > 1 {
        vals.par_chunks_mut(k).zip(idx.par_chunks_mut(k)).enumerate().for_each(row);
    } else {
        vals.chunks_mut(k).zip(idx.chunks_mut(k)).enumerate().for_each(row);
    }
    (vals, idx)
}
