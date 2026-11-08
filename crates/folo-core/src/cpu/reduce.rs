//! Reductions, softmax and top-k over one dimension of a contiguous tensor viewed as `[outer, d, inner]`.

use rayon::prelude::*;

use super::PAR_MIN;
use super::elementwise::fast_exp;
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

pub(crate) fn reduce_f32(op: ReduceOp, x: &[f32], (outer, d, inner): (usize, usize, usize)) -> Vec<f32> {
    let mut out = vec![0f32; outer * inner];
    if outer * inner == 0 {
        return out;
    }
    let par = outer * d * inner >= PAR_MIN;
    let init = match op {
        ReduceOp::Sum => 0.0,
        ReduceOp::Max => f32::NEG_INFINITY,
        ReduceOp::Min => f32::INFINITY,
        _ => unreachable!("arg reductions use reduce_arg"),
    };
    let comb = |a: f32, b: f32| match op {
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
    };
    let row = |(o, dst): (usize, &mut [f32])| {
        let src = &x[o * d * inner..(o + 1) * d * inner];
        if inner == 1 {
            dst[0] = match op {
                ReduceOp::Sum => sum_slice(src),
                _ => src.iter().fold(init, |a, &b| comb(a, b)),
            };
        } else {
            dst.fill(init);
            for r in 0..d {
                for (acc, &v) in dst.iter_mut().zip(&src[r * inner..(r + 1) * inner]) {
                    *acc = comb(*acc, v);
                }
            }
        }
    };
    if par && outer > 1 {
        out.par_chunks_mut(inner).enumerate().for_each(row);
    } else if par && inner == 1 {
        // A single long row: split it, then combine the partial results.
        out[0] = x
            .par_chunks(1 << 14)
            .map(|c| match op {
                ReduceOp::Sum => sum_slice(c),
                _ => c.iter().fold(init, |a, &b| comb(a, b)),
            })
            .reduce(|| init, comb);
    } else {
        out.chunks_mut(inner).enumerate().for_each(row);
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
    let cmp = |a: &(f32, usize), b: &(f32, usize)| {
        let o = if largest { b.0.total_cmp(&a.0) } else { a.0.total_cmp(&b.0) };
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
