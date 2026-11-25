//! WebAssembly SIMD128 4-tile input (`Bᵀ d B`) and output (`Aᵀ M A`) transforms for Winograd F(2x2, 3x3),
//! bitwise identical to `winograd::input_transform_scalar` and `winograd::output_transform_scalar`.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::winograd::{SEG, Tiles, winograd_h_scalar};
use super::{ChannelEpilogue, ConvGeom};
use crate::backend::UnaryOp;

/// Explicit Wasm SIMD128 4-tile input transform (`Bᵀ d B`), bitwise identical to `input_transform_scalar`.
///
/// # Safety
/// Same pointer ownership contract as `winograd::input_transform`.
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
    let tw = g.ow.div_ceil(2);
    let n = t1 - t0;
    let mut tr = [[[0f32; SEG]; 4]; 4];
    let mut seg = [0f32; 2 * SEG + 2];
    for (cl, plane) in planes.iter().enumerate() {
        let ci = ci0 + cl;
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            assert!(len <= SEG && pos + len <= tb);
            let x0 = 2 * tx0 as isize - pw;
            let (lo, hi) = (x0.max(0), (x0 + 2 * len as isize + 2).min(w));
            let k_vec = len / 4 * 4;
            for (r, [t0r, t1r, t2r, t3r]) in tr.iter_mut().enumerate() {
                let y = 2 * ty as isize + r as isize - ph;
                let s = &mut seg[..2 * len + 2];
                if (0..h).contains(&y) && lo < hi {
                    let prow = &plane[(y * w) as usize..((y + 1) * w) as usize];
                    let (a, b) = ((lo - x0) as usize, (hi - x0) as usize);
                    s[..a].fill(0.0);
                    s[a..b].copy_from_slice(&prow[lo as usize..hi as usize]);
                    s[b..].fill(0.0);
                } else {
                    s.fill(0.0);
                }
                let mut k = 0;
                while k < k_vec {
                    // SAFETY: `k + 4 <= len <= SEG`, so `2 * k + 10 <= 2 * len + 2 == s.len()` and
                    // `k + 4 <= SEG` for `t0r..t3r`.
                    unsafe {
                        let sp = s.as_ptr().add(2 * k);
                        let x0 = v128_load(sp as *const v128);
                        let x1 = v128_load(sp.add(4) as *const v128);
                        let y0 = v128_load(sp.add(2) as *const v128);
                        let y1 = v128_load(sp.add(6) as *const v128);
                        let e0 = i32x4_shuffle::<0, 2, 4, 6>(x0, x1);
                        let o0 = i32x4_shuffle::<1, 3, 5, 7>(x0, x1);
                        let e1 = i32x4_shuffle::<0, 2, 4, 6>(y0, y1);
                        let o1 = i32x4_shuffle::<1, 3, 5, 7>(y0, y1);
                        v128_store(t0r.as_mut_ptr().add(k) as *mut v128, f32x4_sub(e0, e1));
                        v128_store(t1r.as_mut_ptr().add(k) as *mut v128, f32x4_add(o0, e1));
                        v128_store(t2r.as_mut_ptr().add(k) as *mut v128, f32x4_sub(e1, o0));
                        v128_store(t3r.as_mut_ptr().add(k) as *mut v128, f32x4_sub(o0, o1));
                    }
                    k += 4;
                }
                if k_vec < len {
                    winograd_h_scalar(
                        &s[2 * k_vec..],
                        &mut t0r[k_vec..len],
                        &mut t1r[k_vec..len],
                        &mut t2r[k_vec..len],
                        &mut t3r[k_vec..len],
                    );
                }
            }
            let [tr0, tr1, tr2, tr3] = &tr;
            for j in 0..4 {
                // SAFETY: `j, 4+j, 8+j, 12+j` are distinct `xi` rows of `V` for channel `ci`, owned exclusively
                // by this task (`pos + len <= tb` asserted above).
                let (d0, d1, d2, d3) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(v.add((j * c + ci) * tb + pos), len),
                        std::slice::from_raw_parts_mut(v.add(((4 + j) * c + ci) * tb + pos), len),
                        std::slice::from_raw_parts_mut(v.add(((8 + j) * c + ci) * tb + pos), len),
                        std::slice::from_raw_parts_mut(v.add(((12 + j) * c + ci) * tb + pos), len),
                    )
                };
                let (a0, a1, a2, a3) = (&tr0[j][..len], &tr1[j][..len], &tr2[j][..len], &tr3[j][..len]);
                let mut k = 0;
                while k < k_vec {
                    // SAFETY: `k + 4 <= len` for all eight slices.
                    unsafe {
                        let v0 = v128_load(a0.as_ptr().add(k) as *const v128);
                        let v1 = v128_load(a1.as_ptr().add(k) as *const v128);
                        let v2 = v128_load(a2.as_ptr().add(k) as *const v128);
                        let v3 = v128_load(a3.as_ptr().add(k) as *const v128);
                        v128_store(d0.as_mut_ptr().add(k) as *mut v128, f32x4_sub(v0, v2));
                        v128_store(d1.as_mut_ptr().add(k) as *mut v128, f32x4_add(v1, v2));
                        v128_store(d2.as_mut_ptr().add(k) as *mut v128, f32x4_sub(v2, v1));
                        v128_store(d3.as_mut_ptr().add(k) as *mut v128, f32x4_sub(v1, v3));
                    }
                    k += 4;
                }
                for k in k_vec..len {
                    d0[k] = a0[k] - a2[k];
                    d1[k] = a1[k] + a2[k];
                    d2[k] = a2[k] - a1[k];
                    d3[k] = a1[k] - a3[k];
                }
            }
        }
        if n < tb {
            for xi in 0..16 {
                // SAFETY: row `(xi, ci)` of `V` is `tb` floats inside `v` and owned exclusively by this task.
                unsafe { std::slice::from_raw_parts_mut(v.add((xi * c + ci) * tb + n), tb - n) }.fill(0.0);
            }
        }
    }
}

/// Apply bias + fused activation in Wasm SIMD128 registers, matching `bias_act_inplace` bit for bit.
#[inline(always)]
fn apply_winograd_epi_wasm(v: v128, act: Option<UnaryOp>, bv: v128) -> v128 {
    let x = f32x4_add(v, bv);
    let zero = f32x4_splat(0.0);
    match act {
        None => x,
        Some(UnaryOp::Silu) => crate::cpu::elementwise::silu4_wasm(x),
        // `f32x4_pmax(zero, x)` = `0.0 < x ? x : +0.0`, matching `x.max(0.0)` on `-0.0` and NaN.
        Some(UnaryOp::Relu) => f32x4_pmax(zero, x),
        Some(UnaryOp::LeakyRelu(a)) => {
            let pos = f32x4_gt(x, zero);
            let neg = f32x4_mul(f32x4_splat(a), x);
            v128_bitselect(x, neg, pos)
        }
        // `f32x4_pmin(f32x4_pmax(y, 0.0), 6.0)` preserves NaN and `-0.0`, matching `y.clamp(0.0, 6.0)`.
        Some(UnaryOp::Hardswish) => {
            let y = f32x4_add(x, f32x4_splat(3.0));
            let c = f32x4_pmin(f32x4_pmax(y, zero), f32x4_splat(6.0));
            f32x4_mul(f32x4_mul(x, c), f32x4_splat(1.0 / 6.0))
        }
        _ => unreachable!("apply_winograd_epi_wasm called for unfused activation"),
    }
}

/// Explicit Wasm SIMD128 4-tile output transform (`Aᵀ M A`) with fused register epilogue (`None`/`Silu`/
/// `Relu`/`LeakyRelu`/`Hardswish`), bitwise identical to `output_transform_scalar`.
///
/// # Safety
/// Same pointer ownership contract as `winograd::gemm_output`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn output_transform_wasm(
    m: &[f32],
    mo: usize,
    o0: usize,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    out: *mut f32,
) {
    let (oh, ow) = (g.oh, g.ow);
    let tw = ow.div_ceil(2);
    let mut sm = [[[0f32; SEG]; 4]; 2];
    let mut y_tail = [0f32; 8];
    let can_fuse = epi.is_some_and(|e| {
        matches!(
            e.act,
            None | Some(UnaryOp::Silu | UnaryOp::Relu | UnaryOp::LeakyRelu(_) | UnaryOp::Hardswish)
        )
    });
    for ol in 0..mo {
        let oc = o0 + ol;
        let mrow = |xi: usize| &m[(xi * mo + ol) * tb..][..tb];
        let (fused_act, bv) = if let (true, Some(e)) = (can_fuse, epi) {
            let b = e.bias.map_or(0.0, |b| b[oc]);
            // Without an activation, `bias_act_inplace` skips a zero bias; `x + -0.0` preserves `-0.0`.
            let b = if e.act.is_none() && b == 0.0 { -0.0 } else { b };
            (Some(e.act), f32x4_splat(b))
        } else {
            (None, f32x4_splat(0.0))
        };
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            assert!(len <= SEG && pos + len <= tb);
            let k_vec = len / 4 * 4;
            let [s0, s1] = &mut sm;
            for (j, (s0j, s1j)) in s0.iter_mut().zip(s1.iter_mut()).enumerate() {
                let r = |xi: usize| &mrow(xi)[pos..pos + len];
                let (m0, m1, m2, m3) = (r(j), r(4 + j), r(8 + j), r(12 + j));
                let mut k = 0;
                while k < k_vec {
                    // SAFETY: `k + 4 <= len <= SEG`.
                    unsafe {
                        let v0 = v128_load(m0.as_ptr().add(k) as *const v128);
                        let v1 = v128_load(m1.as_ptr().add(k) as *const v128);
                        let v2 = v128_load(m2.as_ptr().add(k) as *const v128);
                        let v3 = v128_load(m3.as_ptr().add(k) as *const v128);
                        v128_store(s0j.as_mut_ptr().add(k) as *mut v128, f32x4_add(f32x4_add(v0, v1), v2));
                        v128_store(s1j.as_mut_ptr().add(k) as *mut v128, f32x4_sub(f32x4_sub(v1, v2), v3));
                    }
                    k += 4;
                }
                for k in k_vec..len {
                    s0j[k] = m0[k] + m1[k] + m2[k];
                    s1j[k] = m1[k] - m2[k] - m3[k];
                }
            }
            let ox0 = 2 * tx0;
            let ox1 = (ox0 + 2 * len).min(ow);
            let dst_len = ox1 - ox0;
            for (r, [s0, s1, s2, s3]) in sm.iter().enumerate() {
                let oy = 2 * ty + r;
                if oy >= oh {
                    break;
                }
                // SAFETY: row `oy`, columns `ox0..ox1` of output channel `oc` are inside the image and owned
                // exclusively by this task.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.add((oc * oh + oy) * ow + ox0), dst_len) };
                let mut k = 0;
                while k < k_vec {
                    // SAFETY: `k + 4 <= len <= SEG`, `dst` is `dst_len` floats, and `y_tail` is 8 floats.
                    unsafe {
                        let v0 = v128_load(s0.as_ptr().add(k) as *const v128);
                        let v1 = v128_load(s1.as_ptr().add(k) as *const v128);
                        let v2 = v128_load(s2.as_ptr().add(k) as *const v128);
                        let v3 = v128_load(s3.as_ptr().add(k) as *const v128);
                        let ye = f32x4_add(f32x4_add(v0, v1), v2);
                        let yo = f32x4_sub(f32x4_sub(v1, v2), v3);
                        let mut o0v = i32x4_shuffle::<0, 4, 1, 5>(ye, yo);
                        let mut o1v = i32x4_shuffle::<2, 6, 3, 7>(ye, yo);
                        if let Some(act) = fused_act {
                            o0v = apply_winograd_epi_wasm(o0v, act, bv);
                            o1v = apply_winograd_epi_wasm(o1v, act, bv);
                        }
                        let off = 2 * k;
                        if off + 8 <= dst_len {
                            v128_store(dst.as_mut_ptr().add(off) as *mut v128, o0v);
                            v128_store(dst.as_mut_ptr().add(off + 4) as *mut v128, o1v);
                        } else {
                            v128_store(y_tail.as_mut_ptr() as *mut v128, o0v);
                            v128_store(y_tail.as_mut_ptr().add(4) as *mut v128, o1v);
                            dst[off..].copy_from_slice(&y_tail[..dst_len - off]);
                        }
                    }
                    k += 4;
                }
                let tail_off = 2 * k_vec;
                if tail_off < dst_len {
                    let rem_tiles = len - k_vec;
                    for (t, yy) in y_tail.as_chunks_mut::<2>().0[..rem_tiles].iter_mut().enumerate() {
                        let idx = k_vec + t;
                        yy[0] = s0[idx] + s1[idx] + s2[idx];
                        yy[1] = s1[idx] - s2[idx] - s3[idx];
                    }
                    let tail_dst = &mut dst[tail_off..];
                    tail_dst.copy_from_slice(&y_tail[..tail_dst.len()]);
                    if let (true, Some(e)) = (can_fuse, epi) {
                        e.apply(oc, tail_dst);
                    }
                }
                if let (false, Some(e)) = (can_fuse, epi) {
                    e.apply(oc, dst);
                }
            }
        }
    }
}
