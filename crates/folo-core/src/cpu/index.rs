//! Index-based kernels on contiguous data viewed as `[outer, d, inner]`.
//!
//! The accumulating kernels (`index_add`, `scatter_add` and the gradients of `index_select` / `gather`) add the
//! source entries that land on one destination element in increasing source order whatever the thread count,
//! so their results are bitwise identical to a serial loop: rows are partitioned between tasks by destination,
//! never split between them.

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

/// Validate and copy `src[..]` (`i64`) into `dst[..]` (`usize`) using AVX2 4-wide signed range checks.
/// Returns `Some(bad_offset)` with the smallest out-of-range offset inside `src` on error.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
unsafe fn to_usize_i64_chunk_avx2(src: &[i64], dst: &mut [usize], max_valid: i64) -> Option<usize> {
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi64x(max_valid);
    let (sc, sr) = src.as_chunks::<4>();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[i64; 4]` (32 bytes), `d` is `[usize; 4]` (32 bytes on 64-bit), AVX2 is enabled.
        unsafe {
            let v = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
            let neg = _mm256_cmpgt_epi64(zero, v);
            let hi = _mm256_cmpgt_epi64(v, vmax);
            let bad = _mm256_or_si256(neg, hi);
            if _mm256_testz_si256(bad, bad) == 0 {
                let base = ci * 4;
                for (k, &val) in s.iter().enumerate() {
                    if val < 0 || val > max_valid {
                        return Some(base + k);
                    }
                }
            }
            _mm256_storeu_si256(d.as_mut_ptr() as *mut __m256i, v);
        }
    }
    let tail_base = sc.len() * 4;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val < 0 || val > max_valid {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
}

/// Validate and copy `src[..]` (`u32`) into `dst[..]` (`usize`) using AVX2 8-wide unsigned range checks.
/// Returns `Some(bad_offset)` with the smallest out-of-range offset inside `src` on error.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
unsafe fn to_usize_u32_chunk_avx2(src: &[u32], dst: &mut [usize], size: usize) -> Option<usize> {
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let bounded = size <= u32::MAX as usize;
    let max_u32 = if bounded { (size - 1) as u32 } else { u32::MAX };
    let vmax = _mm256_set1_epi32(max_u32 as i32);
    let (sc, sr) = src.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[u32; 8]` (32 bytes), `d` is `[usize; 8]` (64 bytes on 64-bit), AVX2 is enabled.
        unsafe {
            let v = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
            if bounded {
                let ok = _mm256_cmpeq_epi32(_mm256_max_epu32(v, vmax), vmax);
                if _mm256_movemask_epi8(ok) != -1 {
                    let base = ci * 8;
                    for (k, &val) in s.iter().enumerate() {
                        if val > max_u32 {
                            return Some(base + k);
                        }
                    }
                }
            }
            let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v));
            let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(v));
            _mm256_storeu_si256(d.as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(d.as_mut_ptr().add(4) as *mut __m256i, hi);
        }
    }
    let tail_base = sc.len() * 8;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val as u64 >= size as u64 {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
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
                if let Some(off) =
                    unsafe { to_usize_i64_chunk_avx2(&idx[start..start + chunk.len()], chunk, max_valid) }
                {
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
        if let Some(bad) = unsafe { to_usize_i64_chunk_avx2(idx, &mut out, max_valid) } {
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
                    unsafe { to_usize_u32_chunk_avx2(&idx[start..start + chunk.len()], chunk, size) }
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
        if let Some(bad) = unsafe { to_usize_u32_chunk_avx2(idx, &mut out, size) } {
            return Err(Error::IndexOutOfRange { op, index: idx[bad] as i64, size });
        }
        return Ok(out);
    }
    to_usize_scalar(op, idx, size)
}

/// Pack 8 validated `usize` values (each `<= i32::MAX`) into 8 `i32` lanes in one `__m256i`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_8_usize_to_i32(ix: &[usize; 8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // SAFETY: `ix` is `[usize; 8]` (64 bytes on 64-bit) and AVX2 is enabled.
    unsafe {
        let p = ix.as_ptr() as *const f32;
        let lo = _mm256_loadu_ps(p);
        let hi = _mm256_loadu_ps(p.add(8));
        _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0x88>(lo, hi)))
    }
}

/// Gather `dst[t] = src[ix[t]]` for `t < dst.len()` using AVX2 `_mm256_i32gather_ps` when offsets fit `i32`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
unsafe fn gather_1d_f32_avx2(src: &[f32], ix: &[usize], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ix.len(), dst.len());
    debug_assert!(src.len() <= i32::MAX as usize);
    let (ic, ir) = ix.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let base = src.as_ptr();
    for (i8, d8) in ic.iter().zip(dc.iter_mut()) {
        // SAFETY: every validated index is `< src.len() <= i32::MAX`, `d8` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let idx_v = pack_8_usize_to_i32(i8);
            let vals = _mm256_i32gather_ps::<4>(base, idx_v);
            _mm256_storeu_ps(d8.as_mut_ptr(), vals);
        }
    }
    for (d, &j) in dr.iter_mut().zip(ir) {
        *d = src[j];
    }
}

/// Gather one row `dst[k] = src_block[ix[k] * inner + k]` for `k in k0..k0 + dst.len()` using AVX2 gather.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
unsafe fn gather_row_f32_avx2(src_block: &[f32], inner: usize, k0: usize, ix: &[usize], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ix.len(), dst.len());
    debug_assert!(src_block.len() <= i32::MAX as usize);
    let v_inner = _mm256_set1_epi32(inner as i32);
    let ramp = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut vk = _mm256_add_epi32(_mm256_set1_epi32(k0 as i32), ramp);
    let v8 = _mm256_set1_epi32(8);
    let (ic, ir) = ix.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let base = src_block.as_ptr();
    for (i8, d8) in ic.iter().zip(dc.iter_mut()) {
        // SAFETY: every `ix[t] * inner + k < src_block.len() <= i32::MAX`, `d8` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let idx_v = pack_8_usize_to_i32(i8);
            let off = _mm256_add_epi32(_mm256_mullo_epi32(idx_v, v_inner), vk);
            let vals = _mm256_i32gather_ps::<4>(base, off);
            _mm256_storeu_ps(d8.as_mut_ptr(), vals);
            vk = _mm256_add_epi32(vk, v8);
        }
    }
    let tail_k0 = k0 + ic.len() * 8;
    for (t, (d, &j)) in dr.iter_mut().zip(ir).enumerate() {
        *d = src_block[j * inner + tail_k0 + t];
    }
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
                unsafe { gather_1d_f32_avx2(src, idx, dst) };
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
                        unsafe { gather_1d_f32_avx2(src, ix, dst) };
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
                unsafe { gather_1d_f32_avx2(src, ix, dst) };
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
                        unsafe { gather_1d_f32_avx2(src, ix, dst) };
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
                unsafe { gather_row_f32_avx2(src_block, inner, 0, ix, dst) };
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
                    unsafe { gather_row_f32_avx2(src_block, inner, k0, ix, dst) };
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
                for (k, (&v, &j)) in row_s.iter().zip(row_ix).enumerate() {
                    dst[j * inner + k] += v;
                }
            }
        }
    };
    let work = outer * di * inner;
    if work < PAR_MIN {
        out.chunks_mut(d * inner).enumerate().for_each(block);
    } else if outer >= rayon::current_num_threads() {
        par_chunks(out, d * inner, &|i, c| block((i, c)));
    } else if inner >= 64 {
        // Few outer blocks: split the columns. Column `k` of a block is only written by sources of column `k`.
        let base = SendPtr(out.as_mut_ptr());
        let cols = (CHUNK / di).clamp(16, inner);
        par_for(outer * inner.div_ceil(cols), &|t| {
            let p = &base;
            let per = inner.div_ceil(cols);
            let (o, c0) = (t / per, (t % per) * cols);
            let c1 = (c0 + cols).min(inner);
            let out_o = o * d * inner;
            for i in 0..di {
                let r = (o * di + i) * inner;
                for k in c0..c1 {
                    let j = idx[r + k];
                    // SAFETY: `j < d` (validated), so the element lies in block `o` of `out`; it has column `k`,
                    // and each (block, column range) belongs to exactly one task, so writes never race.
                    unsafe { *p.0.add(out_o + j * inner + k) += src[r + k] };
                }
            }
        });
    } else {
        // Few blocks and narrow rows: group the sources of each block by destination.
        for o in 0..outer {
            let blk = &mut out[o * d * inner..(o + 1) * d * inner];
            let ix = &idx[o * di * inner..(o + 1) * di * inner];
            let s = &src[o * di * inner..(o + 1) * di * inner];
            let (order, runs) = if inner == 1 {
                group_by_dest(ix.len(), d, |e| ix[e])
            } else {
                group_by_dest(ix.len(), d * inner, |e| ix[e] * inner + e % inner)
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
    fn to_usize_avx2_matches_scalar_and_reports_first_oob() {
        for len in [0usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 33, PAR_MIN + 19] {
            let size = 100usize;
            let idx_i64: Vec<i64> = (0..len).map(|i| (i % size) as i64).collect();
            let idx_u32: Vec<u32> = (0..len).map(|i| (i % size) as u32).collect();
            assert_eq!(
                to_usize_i64("t", &idx_i64, size).unwrap(),
                to_usize_scalar("t", &idx_i64, size).unwrap()
            );
            assert_eq!(
                to_usize_u32("t", &idx_u32, size).unwrap(),
                to_usize_scalar("t", &idx_u32, size).unwrap()
            );
            if len > 0 {
                // Inject two OOB elements and verify the FIRST one is reported.
                for bad_pos in [0usize, len / 2, len - 1] {
                    let mut bad_i = idx_i64.clone();
                    bad_i[bad_pos] = if bad_pos % 2 == 0 { -1 } else { size as i64 };
                    bad_i[len - 1] = -99;
                    let got = to_usize_i64("t", &bad_i, size).unwrap_err();
                    let want = to_usize_scalar("t", &bad_i, size).unwrap_err();
                    assert_eq!(got.to_string(), want.to_string(), "i64 len={len} bad_pos={bad_pos}");

                    let mut bad_u = idx_u32.clone();
                    bad_u[bad_pos] = size as u32;
                    bad_u[len - 1] = u32::MAX;
                    let got = to_usize_u32("t", &bad_u, size).unwrap_err();
                    let want = to_usize_scalar("t", &bad_u, size).unwrap_err();
                    assert_eq!(got.to_string(), want.to_string(), "u32 len={len} bad_pos={bad_pos}");
                }
            }
        }
        // Empty valid with size == 0; non-empty fails at index 0.
        assert!(to_usize_i64("t", &[], 0).unwrap().is_empty());
        assert!(to_usize_u32("t", &[], 0).unwrap().is_empty());
        assert!(to_usize_i64("t", &[0], 0).is_err());
        assert!(to_usize_u32("t", &[0], 0).is_err());
    }

    #[test]
    fn gather_and_index_select_f32_match_generic_bitwise() {
        let mut r = crate::rng::Rng::seed(19);
        for &(outer, d, inner, n) in &[
            (1usize, 13, 1, 1),
            (1, 13, 1, 17),
            (3, 19, 1, 33),
            (2, 64, 1, PAR_MIN + 7),
            (40, 17, 1, 2000),
            (3, 19, 7, 11),
            (2, 32, 16, 15),
        ] {
            let x = vals(outer * d * inner, 11);
            let idx: Vec<usize> = (0..n).map(|_| (r.next_u64() % d as u64) as usize).collect();
            let got = index_select_f32(&x, (outer, d, inner), &idx);
            let want = index_select(&x, (outer, d, inner), &idx);
            assert!(
                got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "index_select {outer} {d} {inner} {n}"
            );
        }
        for &(outer, d, inner, di) in &[
            (5usize, 11, 1, 1),
            (PAR_MIN + 5, 13, 1, 1),
            (3, 17, 1, 19),
            (2, 64, 1, PAR_MIN / 2 + 13),
            (64, 32, 1, 1024),
            (3, 11, 7, 5),
            (2, 17, 33, 9),
            (1, 256, 513, 128),
            (2, 16, 4097, 8),
        ] {
            let x = vals(outer * d * inner, 21);
            let idx: Vec<usize> =
                (0..outer * di * inner).map(|_| (r.next_u64() % d as u64) as usize).collect();
            let got = gather_f32(&x, (outer, d, inner), &idx, di);
            let want = gather(&x, (outer, d, inner), &idx, di);
            assert!(
                got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "gather {outer} {d} {inner} {di}"
            );
        }
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
        // Signed zeros survive every strategy: -0.0 + -0.0 + ... stays -0.0.
        for &(outer, d, inner, di) in &[(1, 1, 1, 40_000), (1, 2, 128, 300), (40, 1, 1, 1000)] {
            let idx = vec![0usize; outer * di * inner];
            let src = vec![-0.0f32; idx.len()];
            let got = scatter_add(&vec![-0.0; outer * d * inner], (outer, d, inner), &idx, &src, di);
            assert!(got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()), "{outer} {d} {inner} {di}");
        }
        let got = index_add(&vec![-0.0; 4 * 64], (1, 4, 64), &vec![1; 1000], &vec![-0.0; 1000 * 64]);
        assert!(got.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()));
    }

    #[test]
    fn empty_offset_f32_views_do_not_panic_in_index_select_or_gather() {
        use crate::backend::{Backend, View};
        use crate::{CpuStorage, Layout, Shape, Storage};

        let backend = crate::cpu::CpuBackend;
        let x_empty = Storage::Cpu(CpuStorage::F32(vec![]));
        let l_empty = Layout::new(Shape::new(&[0, 4]), vec![4, 1].into(), 10);
        let idx_s = Storage::Cpu(CpuStorage::I64(vec![0, 2]));
        let l_idx = Layout::contiguous(Shape::new(&[2]));
        let out_sel =
            backend.index_select(View::new(&x_empty, &l_empty), 1, View::new(&idx_s, &l_idx)).unwrap();
        assert_eq!(out_sel.len(), 0);

        let idx_empty = Storage::Cpu(CpuStorage::I64(vec![]));
        let l_gidx = Layout::new(Shape::new(&[0, 2]), vec![2, 1].into(), 7);
        let out_g = backend.gather(View::new(&x_empty, &l_empty), 1, View::new(&idx_empty, &l_gidx)).unwrap();
        assert_eq!(out_g.len(), 0);
    }
}
