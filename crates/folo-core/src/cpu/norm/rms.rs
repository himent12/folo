//! Fused AVX2+FMA `rms_norm` forward and backward CPU kernels.

use super::{bwd_row_blocks, tree_sum_vecs};
use crate::cpu::elementwise::SendPtr;
#[cfg(target_arch = "x86_64")]
use crate::cpu::reduce::has_avx2_fma;
use crate::cpu::reduce::{REDUCE_BLOCK, sum_sq_row, tree_combine};
use crate::cpu::{CHUNK, PAR_MIN, alloc_uninit, alloc_zeroed, par_for};

#[inline]
fn rms_apply_scalar(x: &[f32], y: &mut [f32], w: Option<&[f32]>, rstd: f32) {
    match w {
        Some(w) => {
            for ((&xv, yv), &wv) in x.iter().zip(y).zip(w) {
                *yv = (xv * rstd) * wv;
            }
        }
        None => {
            for (&xv, yv) in x.iter().zip(y) {
                *yv = xv * rstd;
            }
        }
    }
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_apply_avx2_fma(x: &[f32], y: &mut [f32], w: Option<&[f32]>, rstd: f32) {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut i = 0usize;
    // SAFETY: loops check `i + 16 <= n` / `i + 8 <= n` within `0..n`, and AVX2+FMA are enabled.
    unsafe {
        let vr = _mm256_set1_ps(rstd);
        match w {
            Some(w) => {
                while i + 16 <= n {
                    let n0 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr);
                    let n1 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vr);
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
                    let n0 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr);
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(n0, _mm256_loadu_ps(w.as_ptr().add(i))),
                    );
                    i += 8;
                }
            }
            None => {
                while i + 16 <= n {
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr),
                    );
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i + 8),
                        _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vr),
                    );
                    i += 16;
                }
                if i + 8 <= n {
                    _mm256_storeu_ps(
                        y.as_mut_ptr().add(i),
                        _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr),
                    );
                    i += 8;
                }
            }
        }
    }
    if i < n {
        rms_apply_scalar(&x[i..], &mut y[i..], w.map(|s| &s[i..]), rstd);
    }
}

#[inline]
fn rms_apply(x: &[f32], y: &mut [f32], w: Option<&[f32]>, rstd: f32) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 8 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        unsafe { rms_apply_avx2_fma(x, y, w, rstd) };
        return;
    }
    rms_apply_scalar(x, y, w, rstd);
}

/// RmsNorm forward over `[outer, d]`, returning `(y [outer*d], rstd [outer])`.
pub(crate) fn rms_norm(
    x: &[f32],
    weight: Option<&[f32]>,
    outer: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    if outer == 0 || d == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut y = alloc_uninit(outer * d);
    let mut rstd = alloc_uninit(outer);
    let inv_d = 1.0 / d as f32;

    if outer == 1 {
        let ms = sum_sq_row(x) * inv_d;
        let r = 1.0 / (ms + eps).sqrt();
        rstd[0] = r;
        if d >= PAR_MIN {
            let n_chunks = d.div_ceil(REDUCE_BLOCK);
            let y_ptr = SendPtr(y.as_mut_ptr());
            par_for(n_chunks, &|c| {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                // SAFETY: chunk ranges `s..e` are disjoint within `0..d`.
                let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr.ptr().add(s), e - s) };
                rms_apply(&x[s..e], yd, weight.map(|w| &w[s..e]), r);
            });
        } else {
            rms_apply(x, &mut y, weight, r);
        }
        return (y, rstd);
    }

    let do_row = |o: usize, yd: &mut [f32]| -> f32 {
        let row = &x[o * d..(o + 1) * d];
        let ms = sum_sq_row(row) * inv_d;
        let r = 1.0 / (ms + eps).sqrt();
        rms_apply(row, yd, weight, r);
        r
    };
    if outer * d >= PAR_MIN {
        let y_ptr = SendPtr(y.as_mut_ptr());
        let r_ptr = SendPtr(rstd.as_mut_ptr());
        let chunk = (CHUNK / d.max(1)).max(1);
        let tasks = outer.div_ceil(chunk);
        par_for(tasks, &|t| {
            let start = t * chunk;
            let end = (start + chunk).min(outer);
            for o in start..end {
                // SAFETY: tasks cover disjoint row ranges `start..end` within `0..outer`.
                let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr.ptr().add(o * d), d) };
                let r = do_row(o, yd);
                unsafe { *r_ptr.ptr().add(o) = r };
            }
        });
    } else {
        for o in 0..outer {
            rstd[o] = do_row(o, &mut y[o * d..(o + 1) * d]);
        }
    }
    (y, rstd)
}

#[inline]
fn rms_bwd_c_scalar(x: &[f32], g: &[f32], w: Option<&[f32]>, rstd: f32) -> f32 {
    let mut acc = [0.0f32; 16];
    let n = x.len();
    let mut i = 0usize;
    match w {
        Some(w) => {
            while i + 16 <= n {
                for k in 0..16 {
                    let xh = x[i + k] * rstd;
                    acc[k] += (g[i + k] * w[i + k]) * xh;
                }
                i += 16;
            }
            for k in 0..(n - i) {
                let xh = x[i + k] * rstd;
                acc[k] += (g[i + k] * w[i + k]) * xh;
            }
        }
        None => {
            while i + 16 <= n {
                for k in 0..16 {
                    acc[k] += g[i + k] * (x[i + k] * rstd);
                }
                i += 16;
            }
            for k in 0..(n - i) {
                acc[k] += g[i + k] * (x[i + k] * rstd);
            }
        }
    }
    let mut s = 0.0f32;
    for a in acc {
        s += a;
    }
    s
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_bwd_c_avx2_fma(x: &[f32], g: &[f32], w: Option<&[f32]>, rstd: f32) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut acc = [0.0f32; 16];
    let mut i = 0usize;
    // SAFETY: loops check `i + 16 <= n`, `acc` holds 16 floats, and AVX2+FMA are enabled.
    unsafe {
        let vr = _mm256_set1_ps(rstd);
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        match w {
            Some(w) => {
                while i + 16 <= n {
                    let xh0 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr);
                    let xh1 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vr);
                    let gw0 =
                        _mm256_mul_ps(_mm256_loadu_ps(g.as_ptr().add(i)), _mm256_loadu_ps(w.as_ptr().add(i)));
                    let gw1 = _mm256_mul_ps(
                        _mm256_loadu_ps(g.as_ptr().add(i + 8)),
                        _mm256_loadu_ps(w.as_ptr().add(i + 8)),
                    );
                    s0 = _mm256_fmadd_ps(gw0, xh0, s0);
                    s1 = _mm256_fmadd_ps(gw1, xh1, s1);
                    i += 16;
                }
            }
            None => {
                while i + 16 <= n {
                    let xh0 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr);
                    let xh1 = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), vr);
                    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(g.as_ptr().add(i)), xh0, s0);
                    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(g.as_ptr().add(i + 8)), xh1, s1);
                    i += 16;
                }
            }
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), s0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), s1);
    }
    for k in 0..(n - i) {
        let xh = x[i + k] * rstd;
        let gw = match w {
            Some(w) => g[i + k] * w[i + k],
            None => g[i + k],
        };
        acc[k] += gw * xh;
    }
    let mut s = 0.0f32;
    for a in acc {
        s += a;
    }
    s
}

#[inline]
fn rms_bwd_c_block(x: &[f32], g: &[f32], w: Option<&[f32]>, rstd: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 16 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        return unsafe { rms_bwd_c_avx2_fma(x, g, w, rstd) };
    }
    rms_bwd_c_scalar(x, g, w, rstd)
}

fn rms_bwd_c_row(x: &[f32], g: &[f32], w: Option<&[f32]>, rstd: f32, par: bool) -> f32 {
    let d = x.len();
    if d <= REDUCE_BLOCK {
        return rms_bwd_c_block(x, g, w, rstd);
    }
    let n_chunks = d.div_ceil(REDUCE_BLOCK);
    let mut parts = vec![0.0f32; n_chunks];
    if par {
        let p_ptr = SendPtr(parts.as_mut_ptr());
        par_for(n_chunks, &|c| {
            let s = c * REDUCE_BLOCK;
            let e = (s + REDUCE_BLOCK).min(d);
            let v = rms_bwd_c_block(&x[s..e], &g[s..e], w.map(|w| &w[s..e]), rstd);
            // SAFETY: each `c` writes a distinct slot in `0..n_chunks`.
            unsafe { *p_ptr.ptr().add(c) = v };
        });
    } else {
        for (c, slot) in parts.iter_mut().enumerate() {
            let s = c * REDUCE_BLOCK;
            let e = (s + REDUCE_BLOCK).min(d);
            *slot = rms_bwd_c_block(&x[s..e], &g[s..e], w.map(|w| &w[s..e]), rstd);
        }
    }
    tree_combine(parts, |a, b| a + b).unwrap_or(0.0)
}

#[inline]
fn rms_bwd_write_scalar(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    rstd: f32,
    c_d: f32,
) {
    for i in 0..x.len() {
        let xh = x[i] * rstd;
        let gv = g[i];
        let gw = match w {
            Some(w) => gv * w[i],
            None => gv,
        };
        dx[i] = rstd * (gw - xh * c_d);
        if let Some(ref mut dw) = dw {
            dw[i] += gv * xh;
        }
    }
}

/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_bwd_write_avx2_fma(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    rstd: f32,
    c_d: f32,
) {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut i = 0usize;
    // SAFETY: loops check `i + 8 <= n`, all slices have length `n`, and AVX2+FMA are enabled.
    unsafe {
        let vr = _mm256_set1_ps(rstd);
        let vc = _mm256_set1_ps(c_d);
        while i + 8 <= n {
            let xh = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vr);
            let gv = _mm256_loadu_ps(g.as_ptr().add(i));
            let gw = match w {
                Some(w) => _mm256_mul_ps(gv, _mm256_loadu_ps(w.as_ptr().add(i))),
                None => gv,
            };
            let dxv = _mm256_mul_ps(vr, _mm256_fnmadd_ps(xh, vc, gw));
            _mm256_storeu_ps(dx.as_mut_ptr().add(i), dxv);
            if let Some(ref mut dw) = dw {
                let cur = _mm256_loadu_ps(dw.as_ptr().add(i));
                _mm256_storeu_ps(dw.as_mut_ptr().add(i), _mm256_fmadd_ps(gv, xh, cur));
            }
            i += 8;
        }
    }
    if i < n {
        rms_bwd_write_scalar(
            &x[i..],
            &g[i..],
            w.map(|s| &s[i..]),
            &mut dx[i..],
            dw.map(|s| &mut s[i..]),
            rstd,
            c_d,
        );
    }
}

#[inline]
fn rms_bwd_write(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    dw: Option<&mut [f32]>,
    rstd: f32,
    c_d: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if x.len() >= 8 && has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above.
        unsafe { rms_bwd_write_avx2_fma(x, g, w, dx, dw, rstd, c_d) };
        return;
    }
    rms_bwd_write_scalar(x, g, w, dx, dw, rstd, c_d);
}

/// RmsNorm backward over `[outer, d]`, returning `(dx, dweight)`.
pub(crate) fn rms_norm_backward(
    x: &[f32],
    grad: &[f32],
    weight: Option<&[f32]>,
    rstd: &[f32],
    outer: usize,
    d: usize,
    need_weight: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    if outer == 0 || d == 0 {
        return (Vec::new(), need_weight.then(|| alloc_zeroed(d)));
    }
    let mut dx = alloc_uninit(outer * d);
    let inv_d = 1.0 / d as f32;

    if outer == 1 {
        let r = rstd[0];
        let par = d >= PAR_MIN;
        let c_d = rms_bwd_c_row(x, grad, weight, r, par) * inv_d;
        let mut dw = need_weight.then(|| alloc_zeroed(d));
        if par {
            let n_chunks = d.div_ceil(REDUCE_BLOCK);
            let dx_ptr = SendPtr(dx.as_mut_ptr());
            let dw_ptr = dw.as_mut().map(|v| SendPtr(v.as_mut_ptr()));
            par_for(n_chunks, &|c| {
                let s = c * REDUCE_BLOCK;
                let e = (s + REDUCE_BLOCK).min(d);
                // SAFETY: chunks `s..e` are disjoint across `0..d`.
                let dxs = unsafe { std::slice::from_raw_parts_mut(dx_ptr.ptr().add(s), e - s) };
                let dws = dw_ptr.map(|p| unsafe { std::slice::from_raw_parts_mut(p.ptr().add(s), e - s) });
                rms_bwd_write(&x[s..e], &grad[s..e], weight.map(|w| &w[s..e]), dxs, dws, r, c_d);
            });
        } else {
            rms_bwd_write(x, grad, weight, &mut dx, dw.as_deref_mut(), r, c_d);
        }
        return (dx, dw);
    }

    let (rows_per_block, n_blocks) = bwd_row_blocks(outer, d);
    if n_blocks == 1 {
        let mut dw = need_weight.then(|| alloc_zeroed(d));
        for (o, &r) in rstd.iter().enumerate().take(outer) {
            let s = o * d;
            let (rx, rg) = (&x[s..s + d], &grad[s..s + d]);
            let c_d = rms_bwd_c_row(rx, rg, weight, r, false) * inv_d;
            rms_bwd_write(rx, rg, weight, &mut dx[s..s + d], dw.as_deref_mut(), r, c_d);
        }
        return (dx, dw);
    }

    let mut dw_parts: Vec<Vec<f32>> =
        if need_weight { (0..n_blocks).map(|_| alloc_zeroed(d)).collect() } else { Vec::new() };
    let dx_ptr = SendPtr(dx.as_mut_ptr());
    let dw_ptrs: Vec<SendPtr<f32>> = dw_parts.iter_mut().map(|v| SendPtr(v.as_mut_ptr())).collect();

    let run_block = |b: usize| {
        let start = b * rows_per_block;
        let end = (start + rows_per_block).min(outer);
        // SAFETY: each block `b` owns distinct `dw_parts[b]` and rows `start..end` of `dx`.
        let mut dw_b = need_weight.then(|| unsafe { std::slice::from_raw_parts_mut(dw_ptrs[b].ptr(), d) });
        for (o, &r) in rstd.iter().enumerate().take(end).skip(start) {
            let s = o * d;
            let (rx, rg) = (&x[s..s + d], &grad[s..s + d]);
            let c_d = rms_bwd_c_row(rx, rg, weight, r, false) * inv_d;
            let dxs = unsafe { std::slice::from_raw_parts_mut(dx_ptr.ptr().add(s), d) };
            rms_bwd_write(rx, rg, weight, dxs, dw_b.as_deref_mut(), r, c_d);
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
    (dx, dw)
}
