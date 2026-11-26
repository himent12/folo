//! Fused AVX2+FMA `layer_norm` and `rms_norm` forward and backward CPU kernels.
//!
//! Every reduction uses fixed [`REDUCE_BLOCK`] chunking and [`tree_combine`] within rows, and a fixed
//! `(outer, d)`-only block partition with pairwise [`col_sum_slice`] tree reduction for parameter gradients
//! (`dweight`, `dbias`), guaranteeing bitwise reproducibility across rayon thread counts.

mod rms;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;

pub(crate) use rms::{rms_norm, rms_norm_backward};

use super::elementwise::SendPtr;
#[cfg(target_arch = "x86_64")]
use super::reduce::has_avx2_fma;
use super::reduce::{REDUCE_BLOCK, col_sum_slice, sum_row, sum_sq_diff_row, tree_combine};
use super::{CHUNK, PAR_MIN, alloc_uninit, alloc_zeroed, par_for};

/// Deterministic number of row blocks for backward parameter-gradient accumulation (`1..=64`), depending
/// solely on `(outer, d)` and never on the rayon thread count.
#[inline]
pub(super) fn bwd_row_blocks(outer: usize, d: usize) -> (usize, usize) {
    let target = (outer.saturating_mul(d) / CHUNK).clamp(1, 64);
    let rows_per_block = outer.div_ceil(target).max(1);
    let n_blocks = outer.div_ceil(rows_per_block);
    (rows_per_block, n_blocks)
}

/// Pairwise binary-tree sum of `parts` (all of equal length `d`) in a fixed order.
pub(super) fn tree_sum_vecs(mut parts: Vec<Vec<f32>>) -> Vec<f32> {
    while parts.len() > 1 {
        let half = parts.len().div_ceil(2);
        let mut next = Vec::with_capacity(half);
        let mut it = parts.into_iter();
        while let Some(mut a) = it.next() {
            if let Some(b) = it.next() {
                col_sum_slice(&mut a, &b);
            }
            next.push(a);
        }
        parts = next;
    }
    parts.pop().unwrap_or_default()
}

/// Pairwise binary-tree sum of `f64` partials in a fixed order (deterministic cold-path fallback).
pub(super) fn tree_sum_f64(mut v: Vec<f64>) -> f64 {
    while v.len() > 1 {
        let half = v.len().div_ceil(2);
        for i in 0..half {
            v[i] = if 2 * i + 1 < v.len() { v[2 * i] + v[2 * i + 1] } else { v[2 * i] };
        }
        v.truncate(half);
    }
    v.pop().unwrap_or(0.0)
}

/// Cold-path `f64` LayerNorm statistics when `f32` mean/variance overflows or underflows near `eps == 0`.
#[cold]
fn ln_stats_f64(row: &[f32], eps: f32) -> (f32, f32) {
    let inv_d = 1.0 / row.len() as f64;
    let sum_f64 = |f: fn(&[f32], f64) -> f64, m: f64| -> f64 {
        if row.len() <= REDUCE_BLOCK {
            return f(row, m);
        }
        let parts: Vec<f64> = row.chunks(REDUCE_BLOCK).map(|c| f(c, m)).collect();
        tree_sum_f64(parts)
    };
    let m64 = sum_f64(|c, _| c.iter().map(|&v| v as f64).sum(), 0.0) * inv_d;
    let m32 = m64 as f32;
    // Measure squared deviations around the actual rounded `f32` mean used by `ln_apply` and backward.
    let m_use = if m32.is_finite() { m32 as f64 } else { m64 };
    let v64 = sum_f64(
        |c, m| {
            c.iter()
                .map(|&v| {
                    let df = v as f64 - m;
                    df * df
                })
                .sum()
        },
        m_use,
    ) * inv_d;
    let r = (1.0 / (v64 + eps as f64).sqrt()) as f32;
    (m32, r)
}

#[inline]
fn ln_row_stats(row: &[f32], inv_d: f32, eps: f32) -> (f32, f32) {
    let m = sum_row(row) * inv_d;
    let v = sum_sq_diff_row(row, m) * inv_d;
    if !m.is_finite() || !v.is_finite() || (v < 1e-30 && eps < 1e-30) {
        return ln_stats_f64(row, eps);
    }
    (m, 1.0 / (v + eps).sqrt())
}

// ---------- LayerNorm forward ----------

#[inline]
pub(super) fn ln_apply_scalar(
    x: &[f32],
    y: &mut [f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    mean: f32,
    rstd: f32,
) {
    match (w, b) {
        (Some(w), Some(b)) => {
            for (((&xv, yv), &wv), &bv) in x.iter().zip(y).zip(w).zip(b) {
                *yv = ((xv - mean) * rstd) * wv + bv;
            }
        }
        (Some(w), None) => {
            for ((&xv, yv), &wv) in x.iter().zip(y).zip(w) {
                *yv = ((xv - mean) * rstd) * wv;
            }
        }
        (None, Some(b)) => {
            for ((&xv, yv), &bv) in x.iter().zip(y).zip(b) {
                *yv = (xv - mean) * rstd + bv;
            }
        }
        (None, None) => {
            for (&xv, yv) in x.iter().zip(y) {
                *yv = (xv - mean) * rstd;
            }
        }
    }
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln_apply_avx2_fma(
    x: &[f32],
    y: &mut [f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    mean: f32,
    rstd: f32,
) {
    use std::arch::x86_64::*;
    let n = x.len();
    assert_eq!(y.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    if let Some(b) = b {
        assert_eq!(b.len(), n);
    }
    let mut i = 0usize;
    // SAFETY: every vector load/store accesses `i..i+8` (or `i..i+16`) within `0..n`, and AVX2+FMA are enabled.
    unsafe {
        let vm = _mm256_set1_ps(mean);
        let vr = _mm256_set1_ps(rstd);
        match (w, b) {
            (Some(w), Some(b)) => {
                while i + 16 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let n1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    let y0 = _mm256_fmadd_ps(
                        n0,
                        _mm256_loadu_ps(w.as_ptr().add(i)),
                        _mm256_loadu_ps(b.as_ptr().add(i)),
                    );
                    let y1 = _mm256_fmadd_ps(
                        n1,
                        _mm256_loadu_ps(w.as_ptr().add(i + 8)),
                        _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                    );
                    _mm256_storeu_ps(y.as_mut_ptr().add(i), y0);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), y1);
                    i += 16;
                }
                if i + 8 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let y0 = _mm256_fmadd_ps(
                        n0,
                        _mm256_loadu_ps(w.as_ptr().add(i)),
                        _mm256_loadu_ps(b.as_ptr().add(i)),
                    );
                    _mm256_storeu_ps(y.as_mut_ptr().add(i), y0);
                    i += 8;
                }
            }
            (Some(w), None) => {
                while i + 16 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let n1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(n0, _mm256_loadu_ps(w.as_ptr().add(i))),
                    );
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i + 8),
                        _mm256_mul_ps(n1, _mm256_loadu_ps(w.as_ptr().add(i + 8))),
                    );
                    i += 16;
                }
                if i + 8 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(n0, _mm256_loadu_ps(w.as_ptr().add(i))),
                    );
                    i += 8;
                }
            }
            (None, Some(b)) => {
                while i + 16 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let n1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_add_ps(n0, _mm256_loadu_ps(b.as_ptr().add(i))),
                    );
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i + 8),
                        _mm256_add_ps(n1, _mm256_loadu_ps(b.as_ptr().add(i + 8))),
                    );
                    i += 16;
                }
                if i + 8 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_add_ps(n0, _mm256_loadu_ps(b.as_ptr().add(i))),
                    );
                    i += 8;
                }
            }
            (None, None) => {
                while i + 16 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let n1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i), n0);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), n1);
                    i += 16;
                }
                if i + 8 <= n {
                    let n0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i), n0);
                    i += 8;
                }
            }
        }
    }
    if i < n {
        ln_apply_scalar(&x[i..], &mut y[i..], w.map(|s| &s[i..]), b.map(|s| &s[i..]), mean, rstd);
    }
}

#[inline]
fn ln_apply(x: &[f32], y: &mut [f32], w: Option<&[f32]>, b: Option<&[f32]>, mean: f32, rstd: f32) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 8 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        unsafe { ln_apply_avx2_fma(x, y, w, b, mean, rstd) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 4 {
        wasm_simd::ln_apply_wasm(x, y, w, b, mean, rstd);
        return;
    }
    ln_apply_scalar(x, y, w, b, mean, rstd);
}

/// LayerNorm forward over `[outer, d]`, returning `(y [outer*d], mean [outer], rstd [outer])`.
pub(crate) fn layer_norm(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    outer: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), outer.checked_mul(d).expect("layer_norm: outer * d overflow"));
    if let Some(w) = weight {
        assert_eq!(w.len(), d, "layer_norm: weight length");
    }
    if let Some(b) = bias {
        assert_eq!(b.len(), d, "layer_norm: bias length");
    }
    if outer == 0 || d == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut y = alloc_uninit(outer * d);
    let mut mean = alloc_uninit(outer);
    let mut rstd = alloc_uninit(outer);
    let inv_d = 1.0 / d as f32;

    if outer == 1 {
        let (m, r) = ln_row_stats(x, inv_d, eps);
        mean[0] = m;
        rstd[0] = r;
        if d >= PAR_MIN {
            let n_chunks = d.div_ceil(REDUCE_BLOCK);
            let y_ptr = SendPtr(y.as_mut_ptr());
            par_for(n_chunks, &|c| {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                // SAFETY: chunk ranges `s..e` are disjoint within `0..d`.
                let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr.ptr().add(s), e - s) };
                ln_apply(&x[s..e], yd, weight.map(|w| &w[s..e]), bias.map(|b| &b[s..e]), m, r);
            });
        } else {
            ln_apply(x, &mut y, weight, bias, m, r);
        }
        return (y, mean, rstd);
    }

    let do_row = |o: usize, yd: &mut [f32]| -> (f32, f32) {
        let row = &x[o * d..(o + 1) * d];
        let (m, r) = ln_row_stats(row, inv_d, eps);
        ln_apply(row, yd, weight, bias, m, r);
        (m, r)
    };
    if outer * d >= PAR_MIN {
        let y_ptr = SendPtr(y.as_mut_ptr());
        let m_ptr = SendPtr(mean.as_mut_ptr());
        let r_ptr = SendPtr(rstd.as_mut_ptr());
        let chunk = (CHUNK / d.max(1)).max(1);
        let tasks = outer.div_ceil(chunk);
        par_for(tasks, &|t| {
            let start = t * chunk;
            let end = (start + chunk).min(outer);
            for o in start..end {
                // SAFETY: tasks cover disjoint row ranges `start..end` within `0..outer`.
                let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr.ptr().add(o * d), d) };
                let (m, r) = do_row(o, yd);
                unsafe {
                    *m_ptr.ptr().add(o) = m;
                    *r_ptr.ptr().add(o) = r;
                }
            }
        });
    } else {
        for o in 0..outer {
            let (m, r) = do_row(o, &mut y[o * d..(o + 1) * d]);
            mean[o] = m;
            rstd[o] = r;
        }
    }
    (y, mean, rstd)
}

// ---------- LayerNorm backward ----------

/// Compute `(sum(gw), sum(gw * x_hat), sum(x_hat))` over a block. Tracking `sum(x_hat)` preserves exact
/// translation invariance `sum(dx) == 0` even when the forward `mean` rounded in `f32`.
#[inline]
fn ln_bwd_sums_scalar(x: &[f32], g: &[f32], w: Option<&[f32]>, mean: f32, rstd: f32) -> (f32, f32, f32) {
    let mut a1 = [0.0f32; 16];
    let mut a2 = [0.0f32; 16];
    let mut a3 = [0.0f32; 16];
    let n = x.len();
    let mut i = 0usize;
    match w {
        Some(w) => {
            while i + 16 <= n {
                for k in 0..16 {
                    let xh = (x[i + k] - mean) * rstd;
                    let gw = g[i + k] * w[i + k];
                    a1[k] += gw;
                    a2[k] += gw * xh;
                    a3[k] += xh;
                }
                i += 16;
            }
            for k in 0..(n - i) {
                let xh = (x[i + k] - mean) * rstd;
                let gw = g[i + k] * w[i + k];
                a1[k] += gw;
                a2[k] += gw * xh;
                a3[k] += xh;
            }
        }
        None => {
            while i + 16 <= n {
                for k in 0..16 {
                    let xh = (x[i + k] - mean) * rstd;
                    let gw = g[i + k];
                    a1[k] += gw;
                    a2[k] += gw * xh;
                    a3[k] += xh;
                }
                i += 16;
            }
            for k in 0..(n - i) {
                let xh = (x[i + k] - mean) * rstd;
                let gw = g[i + k];
                a1[k] += gw;
                a2[k] += gw * xh;
                a3[k] += xh;
            }
        }
    }
    let (mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..16 {
        c1 += a1[k];
        c2 += a2[k];
        c3 += a3[k];
    }
    (c1, c2, c3)
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln_bwd_sums_avx2_fma(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    mean: f32,
    rstd: f32,
) -> (f32, f32, f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    assert_eq!(g.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    let mut a1 = [0.0f32; 16];
    let mut a2 = [0.0f32; 16];
    let mut a3 = [0.0f32; 16];
    let mut i = 0usize;
    // SAFETY: slice lengths asserted above, loops check `i + 16 <= n`, `a1..a3` hold 16 floats, AVX2+FMA enabled.
    unsafe {
        let vm = _mm256_set1_ps(mean);
        let vr = _mm256_set1_ps(rstd);
        let mut s1_0 = _mm256_setzero_ps();
        let mut s1_1 = _mm256_setzero_ps();
        let mut s2_0 = _mm256_setzero_ps();
        let mut s2_1 = _mm256_setzero_ps();
        let mut s3_0 = _mm256_setzero_ps();
        let mut s3_1 = _mm256_setzero_ps();
        match w {
            Some(w) => {
                while i + 16 <= n {
                    let xh0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let xh1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    let gw0 =
                        _mm256_mul_ps(_mm256_loadu_ps(g.as_ptr().add(i)), _mm256_loadu_ps(w.as_ptr().add(i)));
                    let gw1 = _mm256_mul_ps(
                        _mm256_loadu_ps(g.as_ptr().add(i + 8)),
                        _mm256_loadu_ps(w.as_ptr().add(i + 8)),
                    );
                    s1_0 = _mm256_add_ps(s1_0, gw0);
                    s1_1 = _mm256_add_ps(s1_1, gw1);
                    s2_0 = _mm256_fmadd_ps(gw0, xh0, s2_0);
                    s2_1 = _mm256_fmadd_ps(gw1, xh1, s2_1);
                    s3_0 = _mm256_add_ps(s3_0, xh0);
                    s3_1 = _mm256_add_ps(s3_1, xh1);
                    i += 16;
                }
            }
            None => {
                while i + 16 <= n {
                    let xh0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vm), vr);
                    let xh1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vm), vr);
                    let gw0 = _mm256_loadu_ps(g.as_ptr().add(i));
                    let gw1 = _mm256_loadu_ps(g.as_ptr().add(i + 8));
                    s1_0 = _mm256_add_ps(s1_0, gw0);
                    s1_1 = _mm256_add_ps(s1_1, gw1);
                    s2_0 = _mm256_fmadd_ps(gw0, xh0, s2_0);
                    s2_1 = _mm256_fmadd_ps(gw1, xh1, s2_1);
                    s3_0 = _mm256_add_ps(s3_0, xh0);
                    s3_1 = _mm256_add_ps(s3_1, xh1);
                    i += 16;
                }
            }
        }
        _mm256_storeu_ps(a1.as_mut_ptr(), s1_0);
        _mm256_storeu_ps(a1.as_mut_ptr().add(8), s1_1);
        _mm256_storeu_ps(a2.as_mut_ptr(), s2_0);
        _mm256_storeu_ps(a2.as_mut_ptr().add(8), s2_1);
        _mm256_storeu_ps(a3.as_mut_ptr(), s3_0);
        _mm256_storeu_ps(a3.as_mut_ptr().add(8), s3_1);
    }
    for k in 0..(n - i) {
        let xh = (x[i + k] - mean) * rstd;
        let gw = match w {
            Some(w) => g[i + k] * w[i + k],
            None => g[i + k],
        };
        a1[k] += gw;
        a2[k] += gw * xh;
        a3[k] += xh;
    }
    let (mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..16 {
        c1 += a1[k];
        c2 += a2[k];
        c3 += a3[k];
    }
    (c1, c2, c3)
}

#[inline]
fn ln_bwd_sums_block(x: &[f32], g: &[f32], w: Option<&[f32]>, mean: f32, rstd: f32) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 16 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        return unsafe { ln_bwd_sums_avx2_fma(x, g, w, mean, rstd) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 16 {
        return wasm_simd::ln_bwd_sums_wasm(x, g, w, mean, rstd);
    }
    ln_bwd_sums_scalar(x, g, w, mean, rstd)
}

/// Returns `(c1_adj, c2_d)` where `c1_adj = mean(gw) - mean(x_hat) * mean(gw * x_hat)` and
/// `c2_d = mean(gw * x_hat)`, so `dx = rstd * (gw - (c1_adj + x_hat * c2_d))` preserves exact centering.
fn ln_bwd_coeffs_row(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    mean: f32,
    rstd: f32,
    inv_d: f32,
    par: bool,
) -> (f32, f32) {
    let d = x.len();
    let (c1, c2, c3) = if d <= REDUCE_BLOCK {
        ln_bwd_sums_block(x, g, w, mean, rstd)
    } else {
        let n_chunks = d.div_ceil(REDUCE_BLOCK);
        let mut p1 = vec![0.0f32; n_chunks];
        let mut p2 = vec![0.0f32; n_chunks];
        let mut p3 = vec![0.0f32; n_chunks];
        if par {
            let p1_ptr = SendPtr(p1.as_mut_ptr());
            let p2_ptr = SendPtr(p2.as_mut_ptr());
            let p3_ptr = SendPtr(p3.as_mut_ptr());
            par_for(n_chunks, &|c| {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                let (c1, c2, c3) = ln_bwd_sums_block(&x[s..e], &g[s..e], w.map(|w| &w[s..e]), mean, rstd);
                // SAFETY: each `c` writes a distinct slot in `0..n_chunks`.
                unsafe {
                    *p1_ptr.ptr().add(c) = c1;
                    *p2_ptr.ptr().add(c) = c2;
                    *p3_ptr.ptr().add(c) = c3;
                }
            });
        } else {
            for c in 0..n_chunks {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                let (c1, c2, c3) = ln_bwd_sums_block(&x[s..e], &g[s..e], w.map(|w| &w[s..e]), mean, rstd);
                p1[c] = c1;
                p2[c] = c2;
                p3[c] = c3;
            }
        }
        (
            tree_combine(p1, |a, b| a + b).unwrap_or(0.0),
            tree_combine(p2, |a, b| a + b).unwrap_or(0.0),
            tree_combine(p3, |a, b| a + b).unwrap_or(0.0),
        )
    };
    let c1_d = c1 * inv_d;
    let c2_d = c2 * inv_d;
    let xh_mean = c3 * inv_d;
    (c1_d - xh_mean * c2_d, c2_d)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn ln_bwd_write_scalar(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    mut db: Option<&mut [f32]>,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) {
    for i in 0..x.len() {
        let xh = (x[i] - mean) * rstd;
        let gv = g[i];
        let gw = match w {
            Some(w) => gv * w[i],
            None => gv,
        };
        dx[i] = rstd * (gw - (c1_d + xh * c2_d));
        if let Some(ref mut dw) = dw {
            dw[i] += gv * xh;
        }
        if let Some(ref mut db) = db {
            db[i] += gv;
        }
    }
}

/// Const-specialised, branchless 16-wide AVX2+FMA LayerNorm backward write loop.
///
/// # Safety
/// The CPU supports AVX2 and FMA; `x`, `g`, `dx` hold `n` floats; when `HAS_W`/`HAS_DW`/`HAS_DB` are true,
/// `w`/`dw`/`db` point to `n` valid floats respectively and `dx`/`dw`/`db` do not alias each other.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ln_bwd_write_kernel<const HAS_W: bool, const HAS_DW: bool, const HAS_DB: bool>(
    x: *const f32,
    g: *const f32,
    w: *const f32,
    dx: *mut f32,
    dw: *mut f32,
    db: *mut f32,
    n: usize,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) -> usize {
    use std::arch::x86_64::*;
    let vm = _mm256_set1_ps(mean);
    let vr = _mm256_set1_ps(rstd);
    let vc1 = _mm256_set1_ps(c1_d);
    let vc2 = _mm256_set1_ps(c2_d);
    let mut i = 0usize;
    // SAFETY: loops check `i + 16 <= n` / `i + 8 <= n`, pointer validity is guaranteed by caller, AVX2+FMA enabled.
    unsafe {
        while i + 16 <= n {
            let xh0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), vm), vr);
            let xh1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i + 8)), vm), vr);
            let gv0 = _mm256_loadu_ps(g.add(i));
            let gv1 = _mm256_loadu_ps(g.add(i + 8));
            let (gw0, gw1) = if HAS_W {
                (
                    _mm256_mul_ps(gv0, _mm256_loadu_ps(w.add(i))),
                    _mm256_mul_ps(gv1, _mm256_loadu_ps(w.add(i + 8))),
                )
            } else {
                (gv0, gv1)
            };
            let adj0 = _mm256_fmadd_ps(xh0, vc2, vc1);
            let adj1 = _mm256_fmadd_ps(xh1, vc2, vc1);
            _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(vr, _mm256_sub_ps(gw0, adj0)));
            _mm256_storeu_ps(dx.add(i + 8), _mm256_mul_ps(vr, _mm256_sub_ps(gw1, adj1)));
            if HAS_DW {
                let dwp = dw.add(i);
                _mm256_storeu_ps(dwp, _mm256_fmadd_ps(gv0, xh0, _mm256_loadu_ps(dwp)));
                _mm256_storeu_ps(dwp.add(8), _mm256_fmadd_ps(gv1, xh1, _mm256_loadu_ps(dwp.add(8))));
            }
            if HAS_DB {
                let dbp = db.add(i);
                _mm256_storeu_ps(dbp, _mm256_add_ps(_mm256_loadu_ps(dbp), gv0));
                _mm256_storeu_ps(dbp.add(8), _mm256_add_ps(_mm256_loadu_ps(dbp.add(8)), gv1));
            }
            i += 16;
        }
        if i + 8 <= n {
            let xh = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), vm), vr);
            let gv = _mm256_loadu_ps(g.add(i));
            let gw = if HAS_W { _mm256_mul_ps(gv, _mm256_loadu_ps(w.add(i))) } else { gv };
            let adj = _mm256_fmadd_ps(xh, vc2, vc1);
            _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(vr, _mm256_sub_ps(gw, adj)));
            if HAS_DW {
                let dwp = dw.add(i);
                _mm256_storeu_ps(dwp, _mm256_fmadd_ps(gv, xh, _mm256_loadu_ps(dwp)));
            }
            if HAS_DB {
                let dbp = db.add(i);
                _mm256_storeu_ps(dbp, _mm256_add_ps(_mm256_loadu_ps(dbp), gv));
            }
            i += 8;
        }
    }
    i
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ln_bwd_write_avx2_fma(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    mut db: Option<&mut [f32]>,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) {
    let n = x.len();
    assert_eq!(g.len(), n);
    assert_eq!(dx.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    if let Some(ref dw) = dw {
        assert_eq!(dw.len(), n);
    }
    if let Some(ref db) = db {
        assert_eq!(db.len(), n);
    }
    let xp = x.as_ptr();
    let gp = g.as_ptr();
    let wp = w.map_or(std::ptr::null(), <[f32]>::as_ptr);
    let dxp = dx.as_mut_ptr();
    let dwp = dw.as_deref_mut().map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr);
    let dbp = db.as_deref_mut().map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr);
    // SAFETY: lengths asserted above, mutable slices are disjoint by Rust aliasing rules, AVX2+FMA enabled.
    let i = unsafe {
        match (w.is_some(), dw.is_some(), db.is_some()) {
            (true, true, true) => {
                ln_bwd_write_kernel::<true, true, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, true, false) => {
                ln_bwd_write_kernel::<true, true, false>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, false, true) => {
                ln_bwd_write_kernel::<true, false, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, false, false) => ln_bwd_write_kernel::<true, false, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, true, true) => {
                ln_bwd_write_kernel::<false, true, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (false, true, false) => ln_bwd_write_kernel::<false, true, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, false, true) => ln_bwd_write_kernel::<false, false, true>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, false, false) => ln_bwd_write_kernel::<false, false, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
        }
    };
    if i < n {
        ln_bwd_write_scalar(
            &x[i..],
            &g[i..],
            w.map(|s| &s[i..]),
            &mut dx[i..],
            dw.map(|s| &mut s[i..]),
            db.map(|s| &mut s[i..]),
            mean,
            rstd,
            c1_d,
            c2_d,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn ln_bwd_write(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    dw: Option<&mut [f32]>,
    db: Option<&mut [f32]>,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 8 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        unsafe { ln_bwd_write_avx2_fma(x, g, w, dx, dw, db, mean, rstd, c1_d, c2_d) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len() >= 4 {
        wasm_simd::ln_bwd_write_wasm(x, g, w, dx, dw, db, mean, rstd, c1_d, c2_d);
        return;
    }
    ln_bwd_write_scalar(x, g, w, dx, dw, db, mean, rstd, c1_d, c2_d);
}

/// LayerNorm backward over `[outer, d]`, returning `(dx, dweight, dbias)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_backward(
    x: &[f32],
    grad: &[f32],
    weight: Option<&[f32]>,
    mean: &[f32],
    rstd: &[f32],
    outer: usize,
    d: usize,
    need_weight: bool,
    need_bias: bool,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let total = outer.checked_mul(d).expect("layer_norm_backward: outer * d overflow");
    assert_eq!(x.len(), total, "layer_norm_backward: x length");
    assert_eq!(grad.len(), total, "layer_norm_backward: grad length");
    assert_eq!(mean.len(), outer, "layer_norm_backward: mean length");
    assert_eq!(rstd.len(), outer, "layer_norm_backward: rstd length");
    if let Some(w) = weight {
        assert_eq!(w.len(), d, "layer_norm_backward: weight length");
    }
    if outer == 0 || d == 0 {
        return (Vec::new(), need_weight.then(|| alloc_zeroed(d)), need_bias.then(|| alloc_zeroed(d)));
    }
    let mut dx = alloc_uninit(total);
    let inv_d = 1.0 / d as f32;

    if outer == 1 {
        let m = mean[0];
        let r = rstd[0];
        let par = d >= PAR_MIN;
        let (c1_adj, c2_d) = ln_bwd_coeffs_row(x, grad, weight, m, r, inv_d, par);
        let mut dw = need_weight.then(|| alloc_zeroed(d));
        let mut db = need_bias.then(|| alloc_zeroed(d));
        if par {
            let n_chunks = d.div_ceil(REDUCE_BLOCK);
            let dx_ptr = SendPtr(dx.as_mut_ptr());
            let dw_ptr = dw.as_mut().map(|v| SendPtr(v.as_mut_ptr()));
            let db_ptr = db.as_mut().map(|v| SendPtr(v.as_mut_ptr()));
            par_for(n_chunks, &|c| {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                // SAFETY: chunks `s..e` are disjoint across `0..d`.
                let dxs = unsafe { std::slice::from_raw_parts_mut(dx_ptr.ptr().add(s), e - s) };
                let dws = dw_ptr.map(|p| unsafe { std::slice::from_raw_parts_mut(p.ptr().add(s), e - s) });
                let dbs = db_ptr.map(|p| unsafe { std::slice::from_raw_parts_mut(p.ptr().add(s), e - s) });
                ln_bwd_write(
                    &x[s..e],
                    &grad[s..e],
                    weight.map(|w| &w[s..e]),
                    dxs,
                    dws,
                    dbs,
                    m,
                    r,
                    c1_adj,
                    c2_d,
                );
            });
        } else {
            ln_bwd_write(x, grad, weight, &mut dx, dw.as_deref_mut(), db.as_deref_mut(), m, r, c1_adj, c2_d);
        }
        return (dx, dw, db);
    }

    let (rows_per_block, n_blocks) = bwd_row_blocks(outer, d);
    if n_blocks == 1 {
        let mut dw = need_weight.then(|| alloc_zeroed(d));
        let mut db = need_bias.then(|| alloc_zeroed(d));
        for o in 0..outer {
            let s = o * d;
            let (rx, rg) = (&x[s..s + d], &grad[s..s + d]);
            let (m, r) = (mean[o], rstd[o]);
            let (c1_adj, c2_d) = ln_bwd_coeffs_row(rx, rg, weight, m, r, inv_d, false);
            ln_bwd_write(
                rx,
                rg,
                weight,
                &mut dx[s..s + d],
                dw.as_deref_mut(),
                db.as_deref_mut(),
                m,
                r,
                c1_adj,
                c2_d,
            );
        }
        return (dx, dw, db);
    }

    let mut dw_parts: Vec<Vec<f32>> =
        if need_weight { (0..n_blocks).map(|_| alloc_zeroed(d)).collect() } else { Vec::new() };
    let mut db_parts: Vec<Vec<f32>> =
        if need_bias { (0..n_blocks).map(|_| alloc_zeroed(d)).collect() } else { Vec::new() };

    let dx_ptr = SendPtr(dx.as_mut_ptr());
    let dw_ptrs: Vec<SendPtr<f32>> = dw_parts.iter_mut().map(|v| SendPtr(v.as_mut_ptr())).collect();
    let db_ptrs: Vec<SendPtr<f32>> = db_parts.iter_mut().map(|v| SendPtr(v.as_mut_ptr())).collect();

    let run_block = |b: usize| {
        let start = b * rows_per_block;
        let end = (start + rows_per_block).min(outer);
        // SAFETY: each block `b` owns distinct `dw_parts[b]`, `db_parts[b]`, and rows `start..end` of `dx`.
        let mut dw_b = need_weight.then(|| unsafe { std::slice::from_raw_parts_mut(dw_ptrs[b].ptr(), d) });
        let mut db_b = need_bias.then(|| unsafe { std::slice::from_raw_parts_mut(db_ptrs[b].ptr(), d) });
        for o in start..end {
            let s = o * d;
            let (rx, rg) = (&x[s..s + d], &grad[s..s + d]);
            let (m, r) = (mean[o], rstd[o]);
            let (c1_adj, c2_d) = ln_bwd_coeffs_row(rx, rg, weight, m, r, inv_d, false);
            let dxs = unsafe { std::slice::from_raw_parts_mut(dx_ptr.ptr().add(s), d) };
            ln_bwd_write(rx, rg, weight, dxs, dw_b.as_deref_mut(), db_b.as_deref_mut(), m, r, c1_adj, c2_d);
        }
    };

    if outer * d >= PAR_MIN {
        par_for(n_blocks, &run_block);
    } else {
        for b in 0..n_blocks {
            run_block(b);
        }
    }

    let dw = need_weight.then(|| tree_sum_vecs(dw_parts));
    let db = need_bias.then(|| tree_sum_vecs(db_parts));
    (dx, dw, db)
}
