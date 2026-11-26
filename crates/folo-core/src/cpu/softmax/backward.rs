//! Fused cache-tiled AVX2 backward for softmax and log_softmax.

use super::COL_TILE;
#[cfg(target_arch = "x86_64")]
use super::{has_avx2, hsum16};
#[cfg(target_arch = "x86_64")]
use crate::cpu::elementwise::exp8;
use crate::cpu::elementwise::{SendPtr, exp_scalar};
use crate::cpu::reduce::{REDUCE_BLOCK, tree_combine};
use crate::cpu::{PAR_MIN, alloc_uninit, par_chunks, par_for};

/// 16-lane sum of one block (at most [`REDUCE_BLOCK`] elements), matching `reduce::sum_slice` bit-for-bit.
fn sum_block(g: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { sum_block_avx2(g) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if g.len() >= 16 {
        return super::wasm_simd::sum_block_wasm(g);
    }
    let mut acc = [0f32; 16];
    let (gc, gr) = g.as_chunks::<16>();
    for c in gc {
        for i in 0..16 {
            acc[i] += c[i];
        }
    }
    for (i, &v) in gr.iter().enumerate() {
        acc[i] += v;
    }
    acc.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sum_block_avx2(g: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (gc, gr) = g.as_chunks::<16>();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    for gs in gc {
        // SAFETY: `gs` is 16 floats.
        unsafe {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(gs.as_ptr()));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(gs.as_ptr().add(8)));
        }
    }
    hsum16(a0, a1, gr)
}

/// 16-lane dot product `sum(g * y)` of one block (at most [`REDUCE_BLOCK`] elements), bit-exact with
/// `reduce::sum_slice` over `g * y`.
fn dot_block(g: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(g.len(), y.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { dot_block_avx2(g, y) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if g.len() >= 16 {
        return super::wasm_simd::dot_block_wasm(g, y);
    }
    let mut acc = [0f32; 16];
    let (gc, gr) = g.as_chunks::<16>();
    let (yc, yr) = y.as_chunks::<16>();
    for (gs, ys) in gc.iter().zip(yc) {
        for i in 0..16 {
            acc[i] += gs[i] * ys[i];
        }
    }
    for (i, (&gv, &yv)) in gr.iter().zip(yr).enumerate() {
        acc[i] += gv * yv;
    }
    acc.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn dot_block_avx2(g: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (gc, gr) = g.as_chunks::<16>();
    let (yc, yr) = y.as_chunks::<16>();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    for (gs, ys) in gc.iter().zip(yc) {
        // SAFETY: `gs` and `ys` are 16 floats.
        unsafe {
            let p0 = _mm256_mul_ps(_mm256_loadu_ps(gs.as_ptr()), _mm256_loadu_ps(ys.as_ptr()));
            let p1 = _mm256_mul_ps(_mm256_loadu_ps(gs.as_ptr().add(8)), _mm256_loadu_ps(ys.as_ptr().add(8)));
            a0 = _mm256_add_ps(a0, p0);
            a1 = _mm256_add_ps(a1, p1);
        }
    }
    let mut rprod = [0f32; 16];
    for (i, (&gv, &yv)) in gr.iter().zip(yr).enumerate() {
        rprod[i] = gv * yv;
    }
    hsum16(a0, a1, &rprod[..gr.len()])
}

/// Blocked row sum matching `reduce::reduce_row(ReduceOp::Sum, g)` bit-for-bit at any length.
fn sum_row_blocked(g: &[f32]) -> f32 {
    if g.len() <= REDUCE_BLOCK {
        return sum_block(g);
    }
    let parts: Vec<f32> = g.chunks(REDUCE_BLOCK).map(sum_block).collect();
    tree_combine(parts, |a, b| a + b).unwrap_or(0.0)
}

/// Blocked row dot product `sum(g * y)` matching `reduce::reduce_row(ReduceOp::Sum, g * y)` bit-for-bit.
fn dot_row_blocked(g: &[f32], y: &[f32]) -> f32 {
    if g.len() <= REDUCE_BLOCK {
        return dot_block(g, y);
    }
    let parts: Vec<f32> =
        g.chunks(REDUCE_BLOCK).zip(y.chunks(REDUCE_BLOCK)).map(|(gc, yc)| dot_block(gc, yc)).collect();
    tree_combine(parts, |a, b| a + b).unwrap_or(0.0)
}

/// `inner == 1` backward row for `softmax` (`y * (g - sum(g * y))`) or `log_softmax` (`g - exp(y) * sum(g)`).
fn softmax_bwd_row(y: &[f32], g: &[f32], out: &mut [f32], log: bool) {
    debug_assert_eq!(y.len(), g.len());
    debug_assert_eq!(y.len(), out.len());
    let red = if log { sum_row_blocked(g) } else { dot_row_blocked(g, y) };
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { softmax_bwd_row_apply_avx2(y, g, out, red, log) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if y.len() >= 8 {
        return super::wasm_simd::softmax_bwd_row_apply_wasm(y, g, out, red, log);
    }
    if log {
        for ((o, &yv), &gv) in out.iter_mut().zip(y).zip(g) {
            *o = gv - exp_scalar(yv) * red;
        }
    } else {
        for ((o, &yv), &gv) in out.iter_mut().zip(y).zip(g) {
            *o = yv * (gv - red);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn softmax_bwd_row_apply_avx2(y: &[f32], g: &[f32], out: &mut [f32], red: f32, log: bool) {
    use std::arch::x86_64::*;
    let (yc, yr) = y.as_chunks::<16>();
    let (gc, gr) = g.as_chunks::<16>();
    let (oc, or) = out.as_chunks_mut::<16>();
    let rv = _mm256_set1_ps(red);
    if log {
        for ((ys, gs), os) in yc.iter().zip(gc).zip(oc.iter_mut()) {
            // SAFETY: `ys`, `gs`, `os` are 16 floats.
            unsafe {
                let e0 = exp8(_mm256_loadu_ps(ys.as_ptr()));
                let e1 = exp8(_mm256_loadu_ps(ys.as_ptr().add(8)));
                let r0 = _mm256_sub_ps(_mm256_loadu_ps(gs.as_ptr()), _mm256_mul_ps(e0, rv));
                let r1 = _mm256_sub_ps(_mm256_loadu_ps(gs.as_ptr().add(8)), _mm256_mul_ps(e1, rv));
                _mm256_storeu_ps(os.as_mut_ptr(), r0);
                _mm256_storeu_ps(os.as_mut_ptr().add(8), r1);
            }
        }
        for ((o, &yv), &gv) in or.iter_mut().zip(yr).zip(gr) {
            *o = gv - exp_scalar(yv) * red;
        }
    } else {
        for ((ys, gs), os) in yc.iter().zip(gc).zip(oc.iter_mut()) {
            // SAFETY: `ys`, `gs`, `os` are 16 floats.
            unsafe {
                let r0 = _mm256_mul_ps(
                    _mm256_loadu_ps(ys.as_ptr()),
                    _mm256_sub_ps(_mm256_loadu_ps(gs.as_ptr()), rv),
                );
                let r1 = _mm256_mul_ps(
                    _mm256_loadu_ps(ys.as_ptr().add(8)),
                    _mm256_sub_ps(_mm256_loadu_ps(gs.as_ptr().add(8)), rv),
                );
                _mm256_storeu_ps(os.as_mut_ptr(), r0);
                _mm256_storeu_ps(os.as_mut_ptr().add(8), r1);
            }
        }
        for ((o, &yv), &gv) in or.iter_mut().zip(yr).zip(gr) {
            *o = yv * (gv - red);
        }
    }
}

/// Column-tiled `inner > 1` backward strip `c0..c0 + tw` (`tw <= COL_TILE`).
///
/// # Safety
/// `dst_base` points to `d * inner` floats and no other task writes columns `c0..c0 + tw` of this plane.
#[allow(clippy::too_many_arguments)]
unsafe fn softmax_bwd_col_tile(
    y: &[f32],
    g: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected; caller guarantees disjoint column strip.
        return unsafe { softmax_bwd_col_tile_avx2(y, g, dst_base, d, inner, c0, tw, log) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if tw >= 4 {
        // SAFETY: caller guarantees disjoint column strip.
        return unsafe { super::wasm_simd::softmax_bwd_col_tile_wasm(y, g, dst_base, d, inner, c0, tw, log) };
    }
    let mut red = [0f32; COL_TILE];
    if log {
        for r in 0..d {
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            for (a, &gv) in red[..tw].iter_mut().zip(grow) {
                *a += gv;
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: disjoint column strip `c0..c0 + tw` of row `r`.
            let drow = unsafe { std::slice::from_raw_parts_mut(dst_base.add(r * inner + c0), tw) };
            for (((o, &yv), &gv), &sv) in drow.iter_mut().zip(yrow).zip(grow).zip(&red[..tw]) {
                *o = gv - exp_scalar(yv) * sv;
            }
        }
    } else {
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            for ((a, &yv), &gv) in red[..tw].iter_mut().zip(yrow).zip(grow) {
                *a += gv * yv;
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: disjoint column strip `c0..c0 + tw` of row `r`.
            let drow = unsafe { std::slice::from_raw_parts_mut(dst_base.add(r * inner + c0), tw) };
            for (((o, &yv), &gv), &dv) in drow.iter_mut().zip(yrow).zip(grow).zip(&red[..tw]) {
                *o = yv * (gv - dv);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn softmax_bwd_col_tile_avx2(
    y: &[f32],
    g: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    use std::arch::x86_64::*;
    let mut red = [0f32; COL_TILE];
    let v16 = tw / 16 * 16;
    let v8 = tw / 8 * 8;
    if log {
        for r in 0..d {
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            // SAFETY: `j + 16 <= v16 <= tw <= COL_TILE` and `j + 8 <= v8 <= tw`.
            unsafe {
                while j < v16 {
                    let rp = red.as_mut_ptr().add(j);
                    let gp = grow.as_ptr().add(j);
                    _mm256_storeu_ps(rp, _mm256_add_ps(_mm256_loadu_ps(rp), _mm256_loadu_ps(gp)));
                    _mm256_storeu_ps(
                        rp.add(8),
                        _mm256_add_ps(_mm256_loadu_ps(rp.add(8)), _mm256_loadu_ps(gp.add(8))),
                    );
                    j += 16;
                }
                if j < v8 {
                    let rp = red.as_mut_ptr().add(j);
                    _mm256_storeu_ps(
                        rp,
                        _mm256_add_ps(_mm256_loadu_ps(rp), _mm256_loadu_ps(grow.as_ptr().add(j))),
                    );
                }
            }
            for jj in v8..tw {
                red[jj] += grow[jj];
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw` in row `r`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 16 <= v16 <= tw <= COL_TILE` and `j + 8 <= v8 <= tw`.
            unsafe {
                while j < v16 {
                    let yp = yrow.as_ptr().add(j);
                    let gp = grow.as_ptr().add(j);
                    let rp = red.as_ptr().add(j);
                    let e0 = exp8(_mm256_loadu_ps(yp));
                    let e1 = exp8(_mm256_loadu_ps(yp.add(8)));
                    let r0 = _mm256_sub_ps(_mm256_loadu_ps(gp), _mm256_mul_ps(e0, _mm256_loadu_ps(rp)));
                    let r1 = _mm256_sub_ps(
                        _mm256_loadu_ps(gp.add(8)),
                        _mm256_mul_ps(e1, _mm256_loadu_ps(rp.add(8))),
                    );
                    _mm256_storeu_ps(dptr.add(j), r0);
                    _mm256_storeu_ps(dptr.add(j + 8), r1);
                    j += 16;
                }
                if j < v8 {
                    let e = exp8(_mm256_loadu_ps(yrow.as_ptr().add(j)));
                    let r = _mm256_sub_ps(
                        _mm256_loadu_ps(grow.as_ptr().add(j)),
                        _mm256_mul_ps(e, _mm256_loadu_ps(red.as_ptr().add(j))),
                    );
                    _mm256_storeu_ps(dptr.add(j), r);
                }
            }
            for jj in v8..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = grow[jj] - exp_scalar(yrow[jj]) * red[jj] };
            }
        }
    } else {
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            // SAFETY: `j + 16 <= v16 <= tw <= COL_TILE` and `j + 8 <= v8 <= tw`.
            unsafe {
                while j < v16 {
                    let yp = yrow.as_ptr().add(j);
                    let gp = grow.as_ptr().add(j);
                    let rp = red.as_mut_ptr().add(j);
                    let p0 = _mm256_mul_ps(_mm256_loadu_ps(gp), _mm256_loadu_ps(yp));
                    let p1 = _mm256_mul_ps(_mm256_loadu_ps(gp.add(8)), _mm256_loadu_ps(yp.add(8)));
                    _mm256_storeu_ps(rp, _mm256_add_ps(_mm256_loadu_ps(rp), p0));
                    _mm256_storeu_ps(rp.add(8), _mm256_add_ps(_mm256_loadu_ps(rp.add(8)), p1));
                    j += 16;
                }
                if j < v8 {
                    let rp = red.as_mut_ptr().add(j);
                    let p = _mm256_mul_ps(
                        _mm256_loadu_ps(grow.as_ptr().add(j)),
                        _mm256_loadu_ps(yrow.as_ptr().add(j)),
                    );
                    _mm256_storeu_ps(rp, _mm256_add_ps(_mm256_loadu_ps(rp), p));
                }
            }
            for jj in v8..tw {
                red[jj] += grow[jj] * yrow[jj];
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw` in row `r`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 16 <= v16 <= tw <= COL_TILE` and `j + 8 <= v8 <= tw`.
            unsafe {
                while j < v16 {
                    let yp = yrow.as_ptr().add(j);
                    let gp = grow.as_ptr().add(j);
                    let rp = red.as_ptr().add(j);
                    let r0 = _mm256_mul_ps(
                        _mm256_loadu_ps(yp),
                        _mm256_sub_ps(_mm256_loadu_ps(gp), _mm256_loadu_ps(rp)),
                    );
                    let r1 = _mm256_mul_ps(
                        _mm256_loadu_ps(yp.add(8)),
                        _mm256_sub_ps(_mm256_loadu_ps(gp.add(8)), _mm256_loadu_ps(rp.add(8))),
                    );
                    _mm256_storeu_ps(dptr.add(j), r0);
                    _mm256_storeu_ps(dptr.add(j + 8), r1);
                    j += 16;
                }
                if j < v8 {
                    let r = _mm256_mul_ps(
                        _mm256_loadu_ps(yrow.as_ptr().add(j)),
                        _mm256_sub_ps(
                            _mm256_loadu_ps(grow.as_ptr().add(j)),
                            _mm256_loadu_ps(red.as_ptr().add(j)),
                        ),
                    );
                    _mm256_storeu_ps(dptr.add(j), r);
                }
            }
            for jj in v8..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = yrow[jj] * (grow[jj] - red[jj]) };
            }
        }
    }
}

/// Fused backward of `softmax` (`!log`) or `log_softmax` (`log`) along the middle dim of `[outer, d, inner]`.
pub(crate) fn softmax_backward(
    y: &[f32],
    g: &[f32],
    (outer, d, inner): (usize, usize, usize),
    log: bool,
) -> Vec<f32> {
    debug_assert_eq!(y.len(), g.len());
    let mut out = alloc_uninit(y.len());
    if out.is_empty() {
        return out;
    }
    if inner == 1 {
        let row = |o: usize, dst: &mut [f32]| {
            softmax_bwd_row(&y[o * d..(o + 1) * d], &g[o * d..(o + 1) * d], dst, log);
        };
        if y.len() >= PAR_MIN && outer > 1 {
            par_chunks(&mut out, d, &row);
        } else {
            out.chunks_exact_mut(d).enumerate().for_each(|(o, dst)| row(o, dst));
        }
        return out;
    }
    let tiles_per = inner.div_ceil(COL_TILE);
    let total_tasks = outer * tiles_per;
    let optr = SendPtr(out.as_mut_ptr());
    let run = |t: usize| {
        let optr = &optr;
        let (o, ct) = (t / tiles_per, t % tiles_per);
        let c0 = ct * COL_TILE;
        let tw = COL_TILE.min(inner - c0);
        let yp = &y[o * d * inner..(o + 1) * d * inner];
        let gp = &g[o * d * inner..(o + 1) * d * inner];
        // SAFETY: `(o, ct)` owns disjoint columns `c0..c0 + tw` of outer plane `o`.
        unsafe { softmax_bwd_col_tile(yp, gp, optr.0.add(o * d * inner), d, inner, c0, tw, log) };
    };
    if y.len() >= PAR_MIN && total_tasks > 1 {
        par_for(total_tasks, &run);
    } else {
        (0..total_tasks).for_each(run);
    }
    out
}
