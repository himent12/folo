//! Index-based kernels on contiguous data viewed as `[outer, d, inner]`.

use rayon::prelude::*;

use super::PAR_MIN;
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
    let f = |(o, dst): (usize, &mut [T])| {
        let src = &x[o * d * inner..(o + 1) * d * inner];
        for (i, &j) in idx.iter().enumerate() {
            dst[i * inner..(i + 1) * inner].copy_from_slice(&src[j * inner..(j + 1) * inner]);
        }
    };
    if out.len() >= PAR_MIN && outer > 1 {
        out.par_chunks_mut(n * inner).enumerate().for_each(f);
    } else {
        out.chunks_mut(n * inner).enumerate().for_each(f);
    }
    out
}

/// `out = x; out[o, idx[i], :] += src[o, i, :]`.
pub(crate) fn index_add(
    x: &[f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
) -> Vec<f32> {
    let mut out = x.to_vec();
    let n = idx.len();
    if out.is_empty() {
        return out;
    }
    let f = |(o, dst): (usize, &mut [f32])| {
        let s = &src[o * n * inner..(o + 1) * n * inner];
        for (i, &j) in idx.iter().enumerate() {
            for (a, &b) in dst[j * inner..(j + 1) * inner].iter_mut().zip(&s[i * inner..(i + 1) * inner]) {
                *a += b;
            }
        }
    };
    if out.len() >= PAR_MIN && outer > 1 {
        out.par_chunks_mut(d * inner).enumerate().for_each(f);
    } else {
        out.chunks_mut(d * inner).enumerate().for_each(f);
    }
    out
}

/// `out[o, i, j] = x[o, idx[o, i, j], j]`, with `idx` of shape `[outer, di, inner]`.
pub(crate) fn gather<T: Copy + Send + Sync + Default>(
    x: &[T],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    di: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); outer * di * inner];
    if out.is_empty() {
        return out;
    }
    let f = |(o, dst): (usize, &mut [T])| {
        let src = &x[o * d * inner..(o + 1) * d * inner];
        let ix = &idx[o * di * inner..(o + 1) * di * inner];
        for (k, (v, &j)) in dst.iter_mut().zip(ix).enumerate() {
            *v = src[j * inner + k % inner];
        }
    };
    if out.len() >= PAR_MIN && outer > 1 {
        out.par_chunks_mut(di * inner).enumerate().for_each(f);
    } else {
        out.chunks_mut(di * inner).enumerate().for_each(f);
    }
    out
}

/// `out = x; out[o, idx[o, i, j], j] += src[o, i, j]`, with `idx`/`src` of shape `[outer, di, inner]`.
pub(crate) fn scatter_add(
    x: &[f32],
    (outer, d, inner): (usize, usize, usize),
    idx: &[usize],
    src: &[f32],
    di: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    if out.is_empty() {
        return out;
    }
    let f = |(o, dst): (usize, &mut [f32])| {
        let s = &src[o * di * inner..(o + 1) * di * inner];
        let ix = &idx[o * di * inner..(o + 1) * di * inner];
        for (k, (&v, &j)) in s.iter().zip(ix).enumerate() {
            dst[j * inner + k % inner] += v;
        }
    };
    if out.len() >= PAR_MIN && outer > 1 {
        out.par_chunks_mut(d * inner).enumerate().for_each(f);
    } else {
        out.chunks_mut(d * inner).enumerate().for_each(f);
    }
    out
}
