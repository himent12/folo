//! Index-based kernels on contiguous data viewed as `[outer, d, inner]`.
//!
//! The accumulating kernels (`index_add`, `scatter_add` and the gradients of `index_select` / `gather`) add the
//! source entries that land on one destination element in increasing source order whatever the thread count,
//! so their results are bitwise identical to a serial loop: rows are partitioned between tasks by destination,
//! never split between them.

use rayon::prelude::*;

use super::elementwise::SendPtr;
use super::{CHUNK, PAR_MIN};
use crate::{Error, Result};

/// Validate and convert index values to `usize`.
pub(crate) fn to_usize<I: Copy + Into<i64>>(op: &'static str, idx: &[I], size: usize) -> Result<Vec<usize>> {
    idx.iter()
        .map(|&i| {
            let v: i64 = i.into();
            if v < 0 || v as u64 >= size as u64 {
                Err(Error::IndexOutOfRange { op, index: v, size })
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// `out[o, i, :] = x[o, idx[i], :]`.
pub(crate) fn index_select<T: Copy + Send + Sync + Default>(
    x: &[T],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
) -> Vec<T> {
    let n = idx.len();
    let mut out = vec![T::default(); outer * n * inner];
    if out.is_empty() {
        return out;
    }
    let row = |(r, dst): (usize, &mut [T])| {
        let (o, i) = (r / n, r % n);
        let s = (o * d + idx[i]) * inner;
        dst.copy_from_slice(&x[s..s + inner]);
    };
    if out.len() >= PAR_MIN {
        out.par_chunks_mut(inner).enumerate().with_min_len((CHUNK / inner).max(1)).for_each(row);
    } else {
        out.chunks_mut(inner).enumerate().for_each(row);
    }
    out
}

/// `out[o, idx[i], :] += src[o, i, :]` in place.
pub(crate) fn index_add_into(
    out: &mut [f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
) {
    let n = idx.len();
    if out.is_empty() || n == 0 || inner == 0 {
        return;
    }
    let add = |dst: &mut [f32], o: usize, i: usize| {
        let s = &src[(o * n + i) * inner..(o * n + i + 1) * inner];
        for (a, &b) in dst.iter_mut().zip(s) {
            *a += b;
        }
    };
    if outer * n * inner < PAR_MIN {
        for (o, block) in out.chunks_mut(d * inner).enumerate() {
            for (i, &j) in idx.iter().enumerate() {
                add(&mut block[j * inner..(j + 1) * inner], o, i);
            }
        }
        return;
    }
    // Group the sources by destination row, each group in increasing source order (a stable sort).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| idx[i]);
    let mut runs: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    for (p, &i) in order.iter().enumerate() {
        match runs.last_mut() {
            Some((j, r)) if *j == idx[i] => r.end = p + 1,
            _ => runs.push((idx[i], p..p + 1)),
        }
    }
    // One task per (outer block, destination row): disjoint slices of `out`, carved in increasing order.
    let mut tasks: Vec<(&mut [f32], usize, std::ops::Range<usize>)> = Vec::with_capacity(outer * runs.len());
    for (o, block) in out.chunks_mut(d * inner).enumerate() {
        let mut rest = block;
        let mut at = 0;
        for (j, r) in &runs {
            let (_, tail) = rest.split_at_mut((j - at) * inner);
            let (row, tail) = tail.split_at_mut(inner);
            tasks.push((row, o, r.clone()));
            rest = tail;
            at = j + 1;
        }
    }
    let per_task = (n * inner / runs.len()).max(1);
    tasks.into_par_iter().with_min_len((CHUNK / per_task).max(1)).for_each(|(row, o, r)| {
        for &i in &order[r] {
            add(row, o, i);
        }
    });
}

/// `out = x; out[o, idx[i], :] += src[o, i, :]`.
pub(crate) fn index_add(x: &[f32], s3: (usize, usize, usize), idx: &[usize], src: &[f32]) -> Vec<f32> {
    let mut out = super::alloc_uninit(x.len());
    copy_par(&mut out, x);
    index_add_into(&mut out, s3, idx, src);
    out
}

/// `out[o, i, j] = x[o, idx[o, i, j], j]`, with `idx` of shape `[outer, di, inner]`.
pub(crate) fn gather<T: Copy + Send + Sync + Default>(
    x: &[T],
    (_outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    di: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); idx.len()];
    if out.is_empty() {
        return out;
    }
    let f = |(c, dst): (usize, &mut [T])| {
        let base = c * CHUNK;
        for (q, (v, &j)) in dst.iter_mut().zip(&idx[base..]).enumerate() {
            let e = base + q;
            let (o, k) = (e / (di * inner), e % inner);
            *v = x[(o * d + j) * inner + k];
        }
    };
    if out.len() >= PAR_MIN {
        out.par_chunks_mut(CHUNK).enumerate().for_each(f);
    } else {
        out.chunks_mut(CHUNK).enumerate().for_each(f);
    }
    out
}

/// `out[o, idx[o, i, j], j] += src[o, i, j]` in place, with `idx`/`src` of shape `[outer, di, inner]`.
pub(crate) fn scatter_add_into(
    out: &mut [f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
    di: usize,
) {
    if out.is_empty() || idx.is_empty() {
        return;
    }
    let block = |(o, dst): (usize, &mut [f32])| {
        let s = &src[o * di * inner..(o + 1) * di * inner];
        let ix = &idx[o * di * inner..(o + 1) * di * inner];
        for (k, (&v, &j)) in s.iter().zip(ix).enumerate() {
            dst[j * inner + k % inner] += v;
        }
    };
    let work = outer * di * inner;
    if work < PAR_MIN {
        out.chunks_mut(d * inner).enumerate().for_each(block);
    } else if outer >= rayon::current_num_threads() {
        out.par_chunks_mut(d * inner).enumerate().for_each(block);
    } else if inner >= 64 {
        // Few outer blocks: split the columns. Column `k` of a block is only written by sources of column `k`.
        let base = SendPtr(out.as_mut_ptr());
        let cols = (CHUNK / di).clamp(16, inner);
        (0..outer * inner.div_ceil(cols)).into_par_iter().for_each(|t| {
            let p = &base;
            let per = inner.div_ceil(cols);
            let (o, c0) = (t / per, (t % per) * cols);
            let c1 = (c0 + cols).min(inner);
            for i in 0..di {
                let r = (o * di + i) * inner;
                for k in c0..c1 {
                    let j = idx[r + k];
                    // SAFETY: `j < d` (validated), so the element lies in block `o` of `out`; it has column `k`,
                    // and each (block, column range) belongs to exactly one task, so writes never race.
                    unsafe { *p.0.add((o * d + j) * inner + k) += src[r + k] };
                }
            }
        });
    } else {
        // Few blocks and narrow rows: group the sources of each block by destination.
        for o in 0..outer {
            let blk = &mut out[o * d * inner..(o + 1) * d * inner];
            let ix = &idx[o * di * inner..(o + 1) * di * inner];
            let s = &src[o * di * inner..(o + 1) * di * inner];
            let dest = |e: usize| ix[e] * inner + e % inner;
            let mut order: Vec<usize> = (0..ix.len()).collect();
            order.sort_by_key(|&e| dest(e));
            let mut runs: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
            for (p, &e) in order.iter().enumerate() {
                match runs.last_mut() {
                    Some((t, r)) if *t == dest(e) => r.end = p + 1,
                    _ => runs.push((dest(e), p..p + 1)),
                }
            }
            let base = SendPtr(blk.as_mut_ptr());
            runs.par_iter().with_min_len(CHUNK / 8).for_each(|(t, r)| {
                let mut acc = 0f32;
                let p = &base;
                // SAFETY: `t < d * inner` and every destination is a distinct run, owned by one task.
                let dst = unsafe { &mut *p.0.add(*t) };
                acc += *dst;
                for &e in &order[r.clone()] {
                    acc += s[e];
                }
                *dst = acc;
            });
        }
    }
}

/// `out = x; out[o, idx[o, i, j], j] += src[o, i, j]`.
pub(crate) fn scatter_add(
    x: &[f32],
    s3: (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
    di: usize,
) -> Vec<f32> {
    let mut out = super::alloc_uninit(x.len());
    copy_par(&mut out, x);
    scatter_add_into(&mut out, s3, idx, src, di);
    out
}

fn copy_par(dst: &mut [f32], src: &[f32]) {
    if dst.len() >= PAR_MIN {
        dst.par_chunks_mut(CHUNK).zip(src.par_chunks(CHUNK)).for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial_index_add(
        out: &mut [f32],
        (outer, d, inner): (usize, usize, usize),
        idx: &[usize],
        src: &[f32],
    ) {
        let n = idx.len();
        for o in 0..outer {
            for (i, &j) in idx.iter().enumerate() {
                for k in 0..inner {
                    out[(o * d + j) * inner + k] += src[(o * n + i) * inner + k];
                }
            }
        }
    }

    fn serial_scatter_add(
        out: &mut [f32],
        (outer, d, inner): (usize, usize, usize),
        idx: &[usize],
        src: &[f32],
        di: usize,
    ) {
        for o in 0..outer {
            for i in 0..di {
                for k in 0..inner {
                    let e = (o * di + i) * inner + k;
                    out[(o * d + idx[e]) * inner + k] += src[e];
                }
            }
        }
    }

    fn vals(n: usize, seed: u64) -> Vec<f32> {
        let mut r = crate::rng::Rng::seed(seed);
        (0..n).map(|_| r.uniform(-1.0, 1.0) * 1e3f32.powf(r.uniform(-1.0, 1.0))).collect()
    }

    #[test]
    fn parallel_accumulation_is_bitwise_serial() {
        let mut r = crate::rng::Rng::seed(7);
        // (outer, d, inner, n): long rows, many duplicates, few rows of a big table, inner == 1.
        for &(outer, d, inner, n) in
            &[(1, 50, 4096, 300), (3, 1000, 64, 2000), (1, 100_000, 8, 16), (2, 64, 1, 70_000)]
        {
            let idx: Vec<usize> = (0..n).map(|_| (r.next_u64() % d as u64) as usize).collect();
            let src = vals(outer * n * inner, 1);
            let base = vals(outer * d * inner, 2);
            let mut want = base.clone();
            serial_index_add(&mut want, (outer, d, inner), &idx, &src);
            let got = index_add(&base, (outer, d, inner), &idx, &src);
            assert!(
                want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{outer} {d} {inner} {n}"
            );
        }
        // scatter_add strategies: many blocks, few wide blocks, few narrow blocks.
        for &(outer, d, inner, di) in
            &[(64, 10, 64, 20), (2, 30, 4096, 40), (1, 500, 1, 100_000), (3, 7, 3, 9000)]
        {
            let idx: Vec<usize> =
                (0..outer * di * inner).map(|_| (r.next_u64() % d as u64) as usize).collect();
            let src = vals(idx.len(), 3);
            let base = vals(outer * d * inner, 4);
            let mut want = base.clone();
            serial_scatter_add(&mut want, (outer, d, inner), &idx, &src, di);
            let got = scatter_add(&base, (outer, d, inner), &idx, &src, di);
            assert!(
                want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{outer} {d} {inner} {di}"
            );
        }
    }
}
