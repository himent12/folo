//! Index-based kernels on contiguous data viewed as `[outer, d, inner]`.
//!
//! The accumulating kernels (`index_add`, `scatter_add` and the gradients of `index_select` / `gather`) add the
//! source entries that land on one destination element in increasing source order whatever the thread count,
//! so their results are bitwise identical to a serial loop: rows are partitioned between tasks by destination,
//! never split between them.

mod simd;
#[cfg(test)]
mod tests;

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::elementwise::SendPtr;
use super::reduce::col_sum_slice;
use super::{CHUNK, PAR_MIN, alloc_uninit, par_chunks, par_for};
use crate::{Error, Result};

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[inline]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Scalar fallback for index validation and conversion to `usize`.
fn to_usize_scalar<I: Copy + Into<i64>>(op: &'static str, idx: &[I], size: usize) -> Result<Vec<usize>> {
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

/// Validate and convert `i64` indices to `usize`, returning the first out-of-range index on error.
pub(crate) fn to_usize_i64(op: &'static str, idx: &[i64], size: usize) -> Result<Vec<usize>> {
    if idx.is_empty() {
        return Ok(Vec::new());
    }
    if size == 0 {
        return Err(Error::IndexOutOfRange { op, index: idx[0], size: 0 });
    }
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    if size <= i64::MAX as usize && has_avx2() {
        let max_valid = (size - 1) as i64;
        let mut out = vec![0usize; idx.len()];
        if idx.len() >= PAR_MIN {
            let first_bad = AtomicUsize::new(usize::MAX);
            par_chunks(&mut out, CHUNK, &|c, chunk| {
                let start = c * CHUNK;
                if start >= first_bad.load(Ordering::Relaxed) {
                    return;
                }
                // SAFETY: AVX2 was detected above and `chunk.len()` matches the `idx` slice.
                if let Some(off) = unsafe {
                    simd::to_usize_i64_chunk_avx2(&idx[start..start + chunk.len()], chunk, max_valid)
                } {
                    first_bad.fetch_min(start + off, Ordering::Relaxed);
                }
            });
            let bad = first_bad.load(Ordering::Relaxed);
            if bad != usize::MAX {
                return Err(Error::IndexOutOfRange { op, index: idx[bad], size });
            }
            return Ok(out);
        }
        // SAFETY: AVX2 was detected above.
        if let Some(bad) = unsafe { simd::to_usize_i64_chunk_avx2(idx, &mut out, max_valid) } {
            return Err(Error::IndexOutOfRange { op, index: idx[bad], size });
        }
        return Ok(out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if idx.len() >= 4 {
        let max_valid = (size - 1) as i64;
        let mut out = vec![0usize; idx.len()];
        if let Some(bad) = simd::to_usize_i64_chunk_wasm(idx, &mut out, max_valid) {
            return Err(Error::IndexOutOfRange { op, index: idx[bad], size });
        }
        return Ok(out);
    }
    to_usize_scalar(op, idx, size)
}

/// Validate and convert `u32` indices to `usize`, returning the first out-of-range index on error.
pub(crate) fn to_usize_u32(op: &'static str, idx: &[u32], size: usize) -> Result<Vec<usize>> {
    if idx.is_empty() {
        return Ok(Vec::new());
    }
    if size == 0 {
        return Err(Error::IndexOutOfRange { op, index: idx[0] as i64, size: 0 });
    }
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    if has_avx2() {
        let mut out = vec![0usize; idx.len()];
        if idx.len() >= PAR_MIN {
            let first_bad = AtomicUsize::new(usize::MAX);
            par_chunks(&mut out, CHUNK, &|c, chunk| {
                let start = c * CHUNK;
                if start >= first_bad.load(Ordering::Relaxed) {
                    return;
                }
                // SAFETY: AVX2 was detected above and `chunk.len()` matches the `idx` slice.
                if let Some(off) =
                    unsafe { simd::to_usize_u32_chunk_avx2(&idx[start..start + chunk.len()], chunk, size) }
                {
                    first_bad.fetch_min(start + off, Ordering::Relaxed);
                }
            });
            let bad = first_bad.load(Ordering::Relaxed);
            if bad != usize::MAX {
                return Err(Error::IndexOutOfRange { op, index: idx[bad] as i64, size });
            }
            return Ok(out);
        }
        // SAFETY: AVX2 was detected above.
        if let Some(bad) = unsafe { simd::to_usize_u32_chunk_avx2(idx, &mut out, size) } {
            return Err(Error::IndexOutOfRange { op, index: idx[bad] as i64, size });
        }
        return Ok(out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if idx.len() >= 4 {
        let mut out = vec![0usize; idx.len()];
        if let Some(bad) = simd::to_usize_u32_chunk_wasm(idx, &mut out, size) {
            return Err(Error::IndexOutOfRange { op, index: idx[bad] as i64, size });
        }
        return Ok(out);
    }
    to_usize_scalar(op, idx, size)
}

/// `out[o, i, :] = x[o, idx[i], :]` for `f32`, using pooled uninitialized storage and AVX2 gather for `inner == 1`.
pub(crate) fn index_select_f32(
    x: &[f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
) -> Vec<f32> {
    let n = idx.len();
    let total = outer * n * inner;
    let mut out = alloc_uninit(total);
    if total == 0 {
        return out;
    }
    if inner == 1 {
        let run_outer = |o: usize, dst: &mut [f32]| {
            let src = &x[o * d..(o + 1) * d];
            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            if n >= 8 && d <= i32::MAX as usize && has_avx2() {
                // SAFETY: AVX2 was detected and `src.len() == d <= i32::MAX`.
                unsafe { simd::gather_1d_f32_avx2(src, idx, dst) };
                return;
            }
            for (v, &j) in dst.iter_mut().zip(idx) {
                *v = src[j];
            }
        };
        if total >= PAR_MIN {
            if outer >= rayon::current_num_threads() {
                par_chunks(&mut out, n, &run_outer);
            } else {
                let step = CHUNK.min(n).max(1);
                let chunks_per_o = n.div_ceil(step);
                let optr = SendPtr(out.as_mut_ptr());
                let optr = &optr;
                par_for(outer * chunks_per_o, &|t| {
                    let (o, c) = (t / chunks_per_o, t % chunks_per_o);
                    let i0 = c * step;
                    let len = (n - i0).min(step);
                    let src = &x[o * d..(o + 1) * d];
                    let ix = &idx[i0..i0 + len];
                    // SAFETY: each `(o, c)` owns disjoint `o * n + i0..+len` in `out`.
                    let dst = unsafe { std::slice::from_raw_parts_mut(optr.0.add(o * n + i0), len) };
                    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                    if len >= 8 && d <= i32::MAX as usize && has_avx2() {
                        // SAFETY: AVX2 was detected and `src.len() == d <= i32::MAX`.
                        unsafe { simd::gather_1d_f32_avx2(src, ix, dst) };
                        return;
                    }
                    for (v, &j) in dst.iter_mut().zip(ix) {
                        *v = src[j];
                    }
                });
            }
        } else {
            for (o, dst) in out.chunks_exact_mut(n).enumerate() {
                run_outer(o, dst);
            }
        }
        return out;
    }
    let copy_rows = |r0: usize, chunk: &mut [f32]| {
        for (off, dst) in chunk.chunks_exact_mut(inner).enumerate() {
            let r = r0 + off;
            let (o, i) = (r / n, r % n);
            let s = (o * d + idx[i]) * inner;
            dst.copy_from_slice(&x[s..s + inner]);
        }
    };
    if total >= PAR_MIN {
        let rows_per_chunk = (CHUNK / inner).max(1);
        par_chunks(&mut out, rows_per_chunk * inner, &|c, chunk| copy_rows(c * rows_per_chunk, chunk));
    } else {
        copy_rows(0, &mut out);
    }
    out
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

/// Group `0..n` by `key(i) < num_keys` in strictly increasing source order per key (stable).
/// Uses O(n + num_keys) counting sort when `num_keys` is bounded, falling back to comparison sort otherwise.
fn group_by_dest(
    n: usize,
    num_keys: usize,
    key: impl Fn(usize) -> usize,
) -> (Vec<usize>, Vec<(usize, std::ops::Range<usize>)>) {
    if num_keys <= n.saturating_mul(4).max(65_536) {
        let mut cursor = vec![0usize; num_keys];
        for i in 0..n {
            cursor[key(i)] += 1;
        }
        let mut runs = Vec::new();
        let mut pos = 0usize;
        for (j, slot) in cursor.iter_mut().enumerate() {
            let c = *slot;
            if c > 0 {
                *slot = pos;
                runs.push((j, pos..pos + c));
                pos += c;
            }
        }
        let mut order = vec![0usize; n];
        for i in 0..n {
            let j = key(i);
            let p = cursor[j];
            order[p] = i;
            cursor[j] = p + 1;
        }
        return (order, runs);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| key(i));
    let mut runs: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
    for (p, &i) in order.iter().enumerate() {
        let k = key(i);
        match runs.last_mut() {
            Some((j, r)) if *j == k => r.end = p + 1,
            _ => runs.push((k, p..p + 1)),
        }
    }
    (order, runs)
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
        col_sum_slice(dst, s);
    };
    if outer * n * inner < PAR_MIN {
        for (o, block) in out.chunks_mut(d * inner).enumerate() {
            for (i, &j) in idx.iter().enumerate() {
                add(&mut block[j * inner..(j + 1) * inner], o, i);
            }
        }
        return;
    }
    // Group the sources by destination row, each group in increasing source order.
    let (order, runs) = group_by_dest(n, d, |i| idx[i]);
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
    let mut out = alloc_uninit(x.len());
    copy_par(&mut out, x);
    index_add_into(&mut out, s3, idx, src);
    out
}

/// `out[o, i, j] = x[o, idx[o, i, j], j]` for `f32`, division-free with pooled storage and AVX2 gather.
pub(crate) fn gather_f32(
    x: &[f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    di: usize,
) -> Vec<f32> {
    let total = idx.len();
    let mut out = alloc_uninit(total);
    if total == 0 {
        return out;
    }
    if inner == 1 {
        if di == 1 {
            let run_chunk = |c: usize, dst: &mut [f32]| {
                let o0 = c * CHUNK;
                let ix = &idx[o0..o0 + dst.len()];
                for (t, (v, &j)) in dst.iter_mut().zip(ix).enumerate() {
                    *v = x[(o0 + t) * d + j];
                }
            };
            if total >= PAR_MIN {
                par_chunks(&mut out, CHUNK, &run_chunk);
            } else {
                run_chunk(0, &mut out);
            }
            return out;
        }
        let run_outer = |o: usize, dst: &mut [f32]| {
            let src = &x[o * d..(o + 1) * d];
            let ix = &idx[o * di..(o + 1) * di];
            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            if di >= 8 && d <= i32::MAX as usize && has_avx2() {
                // SAFETY: AVX2 was detected and `src.len() == d <= i32::MAX`.
                unsafe { simd::gather_1d_f32_avx2(src, ix, dst) };
                return;
            }
            for (v, &j) in dst.iter_mut().zip(ix) {
                *v = src[j];
            }
        };
        if total >= PAR_MIN {
            let rows_per_chunk = (CHUNK / di).max(1);
            if outer >= rayon::current_num_threads() {
                par_chunks(&mut out, rows_per_chunk * di, &|c, chunk| {
                    let o0 = c * rows_per_chunk;
                    for (off, dst) in chunk.chunks_exact_mut(di).enumerate() {
                        run_outer(o0 + off, dst);
                    }
                });
            } else {
                let step = CHUNK.min(di).max(1);
                let chunks_per_o = di.div_ceil(step);
                let optr = SendPtr(out.as_mut_ptr());
                let optr = &optr;
                par_for(outer * chunks_per_o, &|t| {
                    let (o, c) = (t / chunks_per_o, t % chunks_per_o);
                    let i0 = c * step;
                    let len = (di - i0).min(step);
                    let src = &x[o * d..(o + 1) * d];
                    let ix = &idx[o * di + i0..o * di + i0 + len];
                    // SAFETY: each `(o, c)` owns disjoint `o * di + i0..+len` in `out`.
                    let dst = unsafe { std::slice::from_raw_parts_mut(optr.0.add(o * di + i0), len) };
                    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                    if len >= 8 && d <= i32::MAX as usize && has_avx2() {
                        // SAFETY: AVX2 was detected and `src.len() == d <= i32::MAX`.
                        unsafe { simd::gather_1d_f32_avx2(src, ix, dst) };
                        return;
                    }
                    for (v, &j) in dst.iter_mut().zip(ix) {
                        *v = src[j];
                    }
                });
            }
        } else {
            for (o, dst) in out.chunks_exact_mut(di).enumerate() {
                run_outer(o, dst);
            }
        }
        return out;
    }

    let block_len = d * inner;
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    let can_avx2_row = inner >= 8 && block_len <= i32::MAX as usize && has_avx2();
    let run_rows = |r0: usize, chunk: &mut [f32]| {
        for (off, dst) in chunk.chunks_exact_mut(inner).enumerate() {
            let r = r0 + off;
            let o = r / di;
            let src_block = &x[o * block_len..(o + 1) * block_len];
            let ix = &idx[r * inner..(r + 1) * inner];
            #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
            if can_avx2_row {
                // SAFETY: AVX2 was detected and `src_block.len() == d * inner <= i32::MAX`.
                unsafe { simd::gather_row_f32_avx2(src_block, inner, 0, ix, dst) };
                continue;
            }
            for (k, (v, &j)) in dst.iter_mut().zip(ix).enumerate() {
                *v = src_block[j * inner + k];
            }
        }
    };
    if total >= PAR_MIN {
        let total_rows = outer * di;
        if total_rows >= rayon::current_num_threads() {
            let rows_per_chunk = (CHUNK / inner).max(1);
            par_chunks(&mut out, rows_per_chunk * inner, &|c, chunk| {
                run_rows(c * rows_per_chunk, chunk);
            });
        } else {
            let step = CHUNK.min(inner).max(1);
            let tiles = inner.div_ceil(step);
            let optr = SendPtr(out.as_mut_ptr());
            let optr = &optr;
            par_for(total_rows * tiles, &|t| {
                let (r, ti) = (t / tiles, t % tiles);
                let o = r / di;
                let k0 = ti * step;
                let len = (inner - k0).min(step);
                let src_block = &x[o * block_len..(o + 1) * block_len];
                let ix = &idx[r * inner + k0..r * inner + k0 + len];
                // SAFETY: each `(r, ti)` owns disjoint `r * inner + k0..+len` in `out`.
                let dst = unsafe { std::slice::from_raw_parts_mut(optr.0.add(r * inner + k0), len) };
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                if can_avx2_row && len >= 8 {
                    // SAFETY: AVX2 was detected and `src_block.len() == d * inner <= i32::MAX`.
                    unsafe { simd::gather_row_f32_avx2(src_block, inner, k0, ix, dst) };
                    return;
                }
                for (q, (v, &j)) in dst.iter_mut().zip(ix).enumerate() {
                    *v = src_block[j * inner + k0 + q];
                }
            });
        }
    } else {
        run_rows(0, &mut out);
    }
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
    let block_len = d * inner;
    let run_rows = |r0: usize, chunk: &mut [T]| {
        for (off, dst) in chunk.chunks_exact_mut(inner).enumerate() {
            let r = r0 + off;
            let o = r / di;
            let src_block = &x[o * block_len..(o + 1) * block_len];
            let ix = &idx[r * inner..(r + 1) * inner];
            for (k, (v, &j)) in dst.iter_mut().zip(ix).enumerate() {
                *v = src_block[j * inner + k];
            }
        }
    };
    if out.len() >= PAR_MIN {
        let rows_per_chunk = (CHUNK / inner).max(1);
        par_chunks(&mut out, rows_per_chunk * inner, &|c, chunk| run_rows(c * rows_per_chunk, chunk));
    } else {
        run_rows(0, &mut out);
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
    let block_len = d * inner;
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    let can_avx2 = inner >= 8 && block_len <= i32::MAX as usize && has_avx2();
    let block = |(o, dst): (usize, &mut [f32])| {
        let s = &src[o * di * inner..(o + 1) * di * inner];
        let ix = &idx[o * di * inner..(o + 1) * di * inner];
        if inner == 1 {
            for (&v, &j) in s.iter().zip(ix) {
                dst[j] += v;
            }
        } else {
            for i in 0..di {
                let row_s = &s[i * inner..(i + 1) * inner];
                let row_ix = &ix[i * inner..(i + 1) * inner];
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                if can_avx2 {
                    // SAFETY: AVX2 was detected, `dst.len() == d * inner <= i32::MAX`, and every index is `< d`.
                    unsafe { simd::scatter_add_cols_avx2(dst.as_mut_ptr(), inner, 0, row_ix, row_s) };
                    continue;
                }
                for (k, (&v, &j)) in row_s.iter().zip(row_ix).enumerate() {
                    dst[j * inner + k] += v;
                }
            }
        }
    };
    let work = outer * di * inner;
    if work < PAR_MIN {
        out.chunks_mut(block_len).enumerate().for_each(block);
    } else if outer >= rayon::current_num_threads() {
        par_chunks(out, block_len, &|i, c| block((i, c)));
    } else if inner >= 64 {
        // Few outer blocks: split the columns. Column `k` of a block is only written by sources of column `k`.
        let base = SendPtr(out.as_mut_ptr());
        let cols = (CHUNK / di).clamp(16, inner);
        par_for(outer * inner.div_ceil(cols), &|t| {
            let p = &base;
            let per = inner.div_ceil(cols);
            let (o, c0) = (t / per, (t % per) * cols);
            let c1 = (c0 + cols).min(inner);
            let out_o = o * block_len;
            // SAFETY: `out_o` is the start of block `o` inside `out`; this task exclusively owns columns `c0..c1`.
            let blk_ptr = unsafe { p.0.add(out_o) };
            for i in 0..di {
                let r = (o * di + i) * inner;
                #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
                if can_avx2 && c1 - c0 >= 8 {
                    // SAFETY: AVX2 was detected, `block_len <= i32::MAX`, every index is `< d`, and this task
                    // exclusively owns columns `c0..c1` of block `o`.
                    unsafe {
                        simd::scatter_add_cols_avx2(
                            blk_ptr,
                            inner,
                            c0,
                            &idx[r + c0..r + c1],
                            &src[r + c0..r + c1],
                        );
                    }
                    continue;
                }
                for k in c0..c1 {
                    let j = idx[r + k];
                    // SAFETY: `j < d` (validated), so the element lies in block `o` of `out`; it has column `k`,
                    // and each (block, column range) belongs to exactly one task, so writes never race.
                    unsafe { *blk_ptr.add(j * inner + k) += src[r + k] };
                }
            }
        });
    } else {
        // Few blocks and narrow rows: group the sources of each block by destination.
        for o in 0..outer {
            let blk = &mut out[o * block_len..(o + 1) * block_len];
            let ix = &idx[o * di * inner..(o + 1) * di * inner];
            let s = &src[o * di * inner..(o + 1) * di * inner];
            let (order, runs) = if inner == 1 {
                group_by_dest(ix.len(), d, |e| ix[e])
            } else {
                group_by_dest(ix.len(), block_len, |e| ix[e] * inner + e % inner)
            };
            let base = SendPtr(blk.as_mut_ptr());
            runs.par_iter().with_min_len(CHUNK / 8).for_each(|(t, r)| {
                let p = &base;
                // SAFETY: `t < d * inner` and every destination is a distinct run, owned by one task.
                let dst = unsafe { &mut *p.0.add(*t) };
                // Start from the destination itself (not 0.0 + dst, which turns -0.0 into +0.0).
                let mut acc = *dst;
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
    let mut out = alloc_uninit(x.len());
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
