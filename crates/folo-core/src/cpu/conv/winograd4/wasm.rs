//! WebAssembly SIMD128 4-tile input (`Bᵀ d B`) and output (`Aᵀ M A`) transforms for Winograd F(4x4, 3x3),
//! bitwise identical to `winograd4::transform`'s non-FMA scalar path on `wasm32 +simd128`.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::transform::{at_1d_scalar, bt_1d_scalar, direct_bad_tiles, load_row_seg};
use super::{SEG, Tiles};
use crate::backend::UnaryOp;
use crate::cpu::conv::{ChannelEpilogue, ConvGeom};

#[inline(always)]
fn bt_1d_wasm(d0: v128, d1: v128, d2: v128, d3: v128, d4: v128, d5: v128) -> [v128; 6] {
    let c25 = f32x4_splat(0.25);
    let c50 = f32x4_splat(0.5);
    let m_even = f32x4_sub(d2, d4);
    let m_odd = f32x4_sub(d1, d3);
    let t0 = f32x4_sub(d0, d2);
    let b0 = f32x4_sub(f32x4_mul(c25, t0), m_even);
    let b1 = f32x4_add(f32x4_mul(c50, m_odd), m_even);
    let b2 = f32x4_sub(m_even, f32x4_mul(c50, m_odd));
    let s_even = f32x4_sub(d4, f32x4_mul(c25, d2));
    let s_odd = f32x4_sub(d3, f32x4_mul(c25, d1));
    let b3 = f32x4_add(s_even, s_odd);
    let b4 = f32x4_sub(s_even, s_odd);
    let b5 = f32x4_add(f32x4_mul(c25, m_odd), f32x4_sub(d5, d3));
    [b0, b1, b2, b3, b4, b5]
}

#[inline(always)]
fn at_1d_wasm(m0: v128, m1: v128, m2: v128, m3: v128, m4: v128, m5: v128) -> [v128; 4] {
    let s12 = f32x4_add(m1, m2);
    let d12 = f32x4_sub(m1, m2);
    let s34 = f32x4_add(m3, m4);
    let d34 = f32x4_sub(m3, m4);
    let y0 = f32x4_add(f32x4_add(m0, s12), s34);
    let y1 = f32x4_add(f32x4_mul(f32x4_splat(0.5), d12), d34);
    let y2 = f32x4_add(f32x4_mul(f32x4_splat(0.25), s12), s34);
    let y3 = f32x4_add(f32x4_mul(f32x4_splat(0.125), d12), f32x4_add(d34, m5));
    [y0, y1, y2, y3]
}

/// Explicit Wasm SIMD128 4-tile input transform (`Bᵀ d B`).
///
/// # Safety
/// Same contract as `transform::input_transform`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn input_transform_wasm(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    let (c, h, w) = (g.c, g.h as isize, g.w as isize);
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let tw = g.ow.div_ceil(4);
    let n = t1 - t0;
    let mut tr = [[[0f32; SEG]; 6]; 6];
    let mut seg = [0f32; 4 * SEG + 2];
    for (cl, plane) in planes.iter().enumerate() {
        let ci = ci0 + cl;
        // SAFETY: row `(xi, ci)` belongs exclusively to this task.
        let row = |xi: usize| unsafe { std::slice::from_raw_parts_mut(v.add((xi * c + ci) * tb), tb) };
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            let x0 = 4 * tx0 as isize - pw;
            let (lo, hi) = (x0.max(0), (x0 + 4 * len as isize + 2).min(w));
            let s = &mut seg[..4 * len + 2];
            for (r, tr_r) in tr.iter_mut().enumerate() {
                load_row_seg(plane, 4 * ty as isize + r as isize - ph, (h, w), x0, (lo, hi), s);
                let mut k = 0;
                // SAFETY: `4 * (k + 4) + 2 <= s.len()` and `k + 4 <= SEG`.
                unsafe {
                    while k + 4 <= len {
                        let sp = s.as_ptr().add(4 * k);
                        let v0 = v128_load(sp as *const v128);
                        let v1 = v128_load(sp.add(4) as *const v128);
                        let v2 = v128_load(sp.add(8) as *const v128);
                        let v3 = v128_load(sp.add(12) as *const v128);
                        let t0 = i32x4_shuffle::<0, 4, 1, 5>(v0, v1);
                        let t1 = i32x4_shuffle::<0, 4, 1, 5>(v2, v3);
                        let t2 = i32x4_shuffle::<2, 6, 3, 7>(v0, v1);
                        let t3 = i32x4_shuffle::<2, 6, 3, 7>(v2, v3);
                        let d0 = i32x4_shuffle::<0, 1, 4, 5>(t0, t1);
                        let d1 = i32x4_shuffle::<2, 3, 6, 7>(t0, t1);
                        let d2 = i32x4_shuffle::<0, 1, 4, 5>(t2, t3);
                        let d3 = i32x4_shuffle::<2, 3, 6, 7>(t2, t3);
                        let tail = f32x4(*sp.add(16), *sp.add(17), 0.0, 0.0);
                        let d4 = i32x4_shuffle::<1, 2, 3, 4>(d0, tail);
                        let d5 = i32x4_shuffle::<1, 2, 3, 5>(d1, tail);
                        let cols = bt_1d_wasm(d0, d1, d2, d3, d4, d5);
                        for (dst_j, &col_v) in tr_r.iter_mut().zip(&cols) {
                            v128_store(dst_j.as_mut_ptr().add(k) as *mut v128, col_v);
                        }
                        k += 4;
                    }
                }
                for kk in k..len {
                    let b = 4 * kk;
                    let cols = bt_1d_scalar(s[b], s[b + 1], s[b + 2], s[b + 3], s[b + 4], s[b + 5]);
                    for (dst_j, &val) in tr_r.iter_mut().zip(&cols) {
                        dst_j[kk] = val;
                    }
                }
            }
            for j in 0..6 {
                let dsts: [*mut f32; 6] = std::array::from_fn(|i| row(i * 6 + j)[pos..].as_mut_ptr());
                let mut k = 0;
                // SAFETY: `k + 4 <= len <= SEG`, and `pos + k + 4 <= tb`.
                unsafe {
                    while k + 4 <= len {
                        let ys = bt_1d_wasm(
                            v128_load(tr[0][j].as_ptr().add(k) as *const v128),
                            v128_load(tr[1][j].as_ptr().add(k) as *const v128),
                            v128_load(tr[2][j].as_ptr().add(k) as *const v128),
                            v128_load(tr[3][j].as_ptr().add(k) as *const v128),
                            v128_load(tr[4][j].as_ptr().add(k) as *const v128),
                            v128_load(tr[5][j].as_ptr().add(k) as *const v128),
                        );
                        for (&dp, &yv) in dsts.iter().zip(&ys) {
                            v128_store(dp.add(k) as *mut v128, yv);
                        }
                        k += 4;
                    }
                    for kk in k..len {
                        let ys = bt_1d_scalar(
                            tr[0][j][kk],
                            tr[1][j][kk],
                            tr[2][j][kk],
                            tr[3][j][kk],
                            tr[4][j][kk],
                            tr[5][j][kk],
                        );
                        for (&dp, &yv) in dsts.iter().zip(&ys) {
                            *dp.add(kk) = yv;
                        }
                    }
                }
            }
        }
        if n < tb {
            for xi in 0..36 {
                row(xi)[n..].fill(0.0);
            }
        }
    }
}

/// Explicit Wasm SIMD128 4-tile output transform (`Aᵀ M A`) with fused register epilogue (`None`/`Silu`/`Relu`)
/// and per-tile pre-epilogue overflow detection.
///
/// # Safety
/// Same contract as `transform::output_transform`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn output_transform_wasm(
    m: &[f32],
    m_stride: usize,
    os: std::ops::Range<usize>,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    planes: &[&[f32]],
    wt: &[f32],
    out: *mut f32,
) {
    let (oh, ow) = (g.oh, g.ow);
    let tw = ow.div_ceil(4);
    let oc0 = os.start;
    let mut sm = [[[0f32; SEG]; 6]; 4];
    let mut y = [0f32; 4 * SEG];
    let fast_epi = match epi {
        None => true,
        Some(e) => matches!(e.act, None | Some(UnaryOp::Silu) | Some(UnaryOp::Relu)),
    };
    let zero = f32x4_splat(0.0);
    for oc in os {
        let mrow = |xi: usize| &m[(xi * m_stride + (oc - oc0)) * tb..][..tb];
        let bias_val = epi.and_then(|e| e.bias).map_or(0.0, |b| b[oc]);
        let act = epi.and_then(|e| e.act);
        let add_bias = act == Some(UnaryOp::Silu) || bias_val != 0.0;
        let vb = f32x4_splat(bias_val);
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            for j in 0..6 {
                let ptr = |i: usize| mrow(i * 6 + j)[pos..].as_ptr();
                let (p0, p1, p2, p3, p4, p5) = (ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5));
                let mut k = 0;
                // SAFETY: `k + 4 <= len <= SEG` and `pos + k + 4 <= tb`.
                unsafe {
                    while k + 4 <= len {
                        let s = at_1d_wasm(
                            v128_load(p0.add(k) as *const v128),
                            v128_load(p1.add(k) as *const v128),
                            v128_load(p2.add(k) as *const v128),
                            v128_load(p3.add(k) as *const v128),
                            v128_load(p4.add(k) as *const v128),
                            v128_load(p5.add(k) as *const v128),
                        );
                        for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                            v128_store(sm_i[j].as_mut_ptr().add(k) as *mut v128, sv);
                        }
                        k += 4;
                    }
                    for kk in k..len {
                        let s = at_1d_scalar(
                            *p0.add(kk),
                            *p1.add(kk),
                            *p2.add(kk),
                            *p3.add(kk),
                            *p4.add(kk),
                            *p5.add(kk),
                        );
                        for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                            sm_i[j][kk] = sv;
                        }
                    }
                }
            }
            let ox0 = 4 * tx0;
            let ox1 = (ox0 + 4 * len).min(ow);
            let out_len = ox1 - ox0;
            let mut bad = 0u64;
            for (r, sm_r) in sm.iter().enumerate() {
                let oy = 4 * ty + r;
                if oy >= oh {
                    break;
                }
                // SAFETY: `row_dst` owns `out_len` floats of row `oy` in channel `oc`.
                let row_dst = unsafe { out.add((oc * oh + oy) * ow + ox0) };
                let mut k = 0;
                let mut direct_done = 0usize;
                // SAFETY: `k + 4 <= len <= SEG` and `4 * (k + 4) <= 4 * SEG`.
                unsafe {
                    while k + 4 <= len {
                        let [y0, y1, y2, y3] = at_1d_wasm(
                            v128_load(sm_r[0].as_ptr().add(k) as *const v128),
                            v128_load(sm_r[1].as_ptr().add(k) as *const v128),
                            v128_load(sm_r[2].as_ptr().add(k) as *const v128),
                            v128_load(sm_r[3].as_ptr().add(k) as *const v128),
                            v128_load(sm_r[4].as_ptr().add(k) as *const v128),
                            v128_load(sm_r[5].as_ptr().add(k) as *const v128),
                        );
                        // Pre-epilogue finiteness check per tile: `finite * 0 == 0`, `±inf/NaN * 0 == NaN`.
                        let nf = f32x4_add(
                            f32x4_add(f32x4_mul(y0, zero), f32x4_mul(y1, zero)),
                            f32x4_add(f32x4_mul(y2, zero), f32x4_mul(y3, zero)),
                        );
                        bad |= (i32x4_bitmask(f32x4_ne(nf, nf)) as u64) << k;
                        let t0 = i32x4_shuffle::<0, 4, 1, 5>(y0, y1);
                        let t1 = i32x4_shuffle::<0, 4, 1, 5>(y2, y3);
                        let t2 = i32x4_shuffle::<2, 6, 3, 7>(y0, y1);
                        let t3 = i32x4_shuffle::<2, 6, 3, 7>(y2, y3);
                        let mut q0 = i32x4_shuffle::<0, 1, 4, 5>(t0, t1);
                        let mut q1 = i32x4_shuffle::<2, 3, 6, 7>(t0, t1);
                        let mut q2 = i32x4_shuffle::<0, 1, 4, 5>(t2, t3);
                        let mut q3 = i32x4_shuffle::<2, 3, 6, 7>(t2, t3);
                        if fast_epi && 4 * k + 16 <= out_len {
                            if add_bias {
                                q0 = f32x4_add(q0, vb);
                                q1 = f32x4_add(q1, vb);
                                q2 = f32x4_add(q2, vb);
                                q3 = f32x4_add(q3, vb);
                            }
                            match act {
                                Some(UnaryOp::Silu) => {
                                    q0 = crate::cpu::elementwise::silu4_wasm(q0);
                                    q1 = crate::cpu::elementwise::silu4_wasm(q1);
                                    q2 = crate::cpu::elementwise::silu4_wasm(q2);
                                    q3 = crate::cpu::elementwise::silu4_wasm(q3);
                                }
                                Some(UnaryOp::Relu) => {
                                    q0 = f32x4_pmax(q0, zero);
                                    q1 = f32x4_pmax(q1, zero);
                                    q2 = f32x4_pmax(q2, zero);
                                    q3 = f32x4_pmax(q3, zero);
                                }
                                _ => {}
                            }
                            let dp = row_dst.add(4 * k);
                            v128_store(dp as *mut v128, q0);
                            v128_store(dp.add(4) as *mut v128, q1);
                            v128_store(dp.add(8) as *mut v128, q2);
                            v128_store(dp.add(12) as *mut v128, q3);
                            direct_done = 4 * k + 16;
                        } else {
                            let yp = y.as_mut_ptr().add(4 * k);
                            v128_store(yp as *mut v128, q0);
                            v128_store(yp.add(4) as *mut v128, q1);
                            v128_store(yp.add(8) as *mut v128, q2);
                            v128_store(yp.add(12) as *mut v128, q3);
                        }
                        k += 4;
                    }
                }
                for (kk, yy) in y.as_chunks_mut::<4>().0[k..len].iter_mut().enumerate() {
                    let kk = k + kk;
                    *yy = at_1d_scalar(
                        sm_r[0][kk],
                        sm_r[1][kk],
                        sm_r[2][kk],
                        sm_r[3][kk],
                        sm_r[4][kk],
                        sm_r[5][kk],
                    );
                    if yy.iter().any(|v| !v.is_finite()) {
                        bad |= 1u64 << kk;
                    }
                }
                if direct_done < out_len {
                    // SAFETY: remaining `out_len - direct_done` elements of row `oy` belong to this task.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(row_dst.add(direct_done), out_len - direct_done)
                    };
                    dst.copy_from_slice(&y[direct_done..out_len]);
                    if let Some(e) = epi {
                        e.apply(oc, dst);
                    }
                }
            }
            if bad != 0 {
                // SAFETY: every set bit in `bad` is in `0..len` of this task's owned segment.
                unsafe { direct_bad_tiles(planes, wt, oc, ty, tx0, bad, g, epi, out) };
            }
        }
    }
}
