//! Depthwise convs (groups = channels): direct kernels per output plane, with AVX2+FMA fast paths for
//! stride-1 (ring-buffered 3×3 and general stride-1), stride-2 (even/odd AVX2 deinterleave), and stride-1/2
//! backward kernels, Wasm SIMD128 at stride 1, and TLS scratch reuse.

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
use std::cell::RefCell;

use smallvec::SmallVec;

use super::{ChannelEpilogue, ConvGeom, ConvInput, valid_cols};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::backend::UnaryOp;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::cpu::gemm::BiasAct;
use crate::cpu::par_chunks;

/// Maximum per-thread TLS scratch retained across depthwise calls (256K floats = 1 MiB).
#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
const DW_TLS_CAP: usize = 1 << 18;

#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
thread_local! {
    /// Reusable padded-plane scratch per worker thread, avoiding per-task heap allocations on every depthwise call.
    static DW_PAD_TLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Borrow the thread-local depthwise scratch buffer, shrinking oversized buffers before releasing; falls back to
/// a local `Vec` on re-entrant use or TLS teardown.
#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
fn with_dw_pad<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut f = Some(f);
    let res = DW_PAD_TLS.try_with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            let out = (f.take().unwrap())(&mut buf);
            if buf.capacity() > DW_TLS_CAP {
                *buf = Vec::new();
            }
            Some(out)
        } else {
            None
        }
    });
    match res {
        Ok(Some(out)) => out,
        _ => {
            let mut local = Vec::new();
            (f.take().unwrap())(&mut local)
        }
    }
}

/// Dot product with independent accumulators so it vectorises.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for v in acc {
        s += v;
    }
    s
}

/// 8-lane Wasm SIMD128 dot product (2x4 accumulators), bitwise identical to [`dot_scalar`].
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dot_wasm(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::wasm32::*;
    assert_eq!(a.len(), b.len());
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    // SAFETY: `as_chunks::<8>()` guarantees 8-float chunks, and `acc` is `[f32; 8]`.
    unsafe {
        let mut s0 = f32x4_splat(0.0);
        let mut s1 = f32x4_splat(0.0);
        for (x, y) in ca.iter().zip(cb) {
            let xp = x.as_ptr();
            let yp = y.as_ptr();
            s0 = f32x4_add(s0, f32x4_mul(v128_load(xp as *const v128), v128_load(yp as *const v128)));
            s1 = f32x4_add(
                s1,
                f32x4_mul(v128_load(xp.add(4) as *const v128), v128_load(yp.add(4) as *const v128)),
            );
        }
        v128_store(acc.as_mut_ptr() as *mut v128, s0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, s1);
    }
    let mut s: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for v in acc {
        s += v;
    }
    s
}

/// 8-wide Wasm SIMD128 row update `dst[i] += wv * src[i]`, bitwise identical to scalar `*d += wv * s`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn mul_add_row_wasm(dst: &mut [f32], src: &[f32], wv: f32) {
    use std::arch::wasm32::*;
    assert_eq!(dst.len(), src.len());
    let w = f32x4_splat(wv);
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    let (s8, s_rem) = src.as_chunks::<8>();
    for (dc, sc) in d8.iter_mut().zip(s8) {
        // SAFETY: `dc` and `sc` are `[f32; 8]`.
        unsafe {
            let dp = dc.as_mut_ptr();
            let sp = sc.as_ptr();
            let d0 = v128_load(dp as *const v128);
            let d1 = v128_load(dp.add(4) as *const v128);
            let s0 = v128_load(sp as *const v128);
            let s1 = v128_load(sp.add(4) as *const v128);
            v128_store(dp as *mut v128, f32x4_add(d0, f32x4_mul(w, s0)));
            v128_store(dp.add(4) as *mut v128, f32x4_add(d1, f32x4_mul(w, s1)));
        }
    }
    let (d4, d_tail) = d_rem.as_chunks_mut::<4>();
    let (s4, s_tail) = s_rem.as_chunks::<4>();
    for (dc, sc) in d4.iter_mut().zip(s4) {
        // SAFETY: `dc` and `sc` are `[f32; 4]`.
        unsafe {
            let d0 = v128_load(dc.as_ptr() as *const v128);
            let s0 = v128_load(sc.as_ptr() as *const v128);
            v128_store(dc.as_mut_ptr() as *mut v128, f32x4_add(d0, f32x4_mul(w, s0)));
        }
    }
    for (d, &s) in d_tail.iter_mut().zip(s_tail) {
        *d += wv * s;
    }
}

/// Precompute `(kx * dw - pw, valid_cols)` once per kernel column (stack-backed for `kw <= 8`).
fn precompute_kx_cols(g: &ConvGeom) -> SmallVec<[(isize, std::ops::Range<usize>); 8]> {
    let (sw, pw, dw) = (g.p.stride[1] as isize, g.p.padding[1] as isize, g.p.dilation[1] as isize);
    (0..g.kw)
        .map(|kx| {
            let d = kx as isize * dw - pw;
            (d, valid_cols(g.ow, sw, d, g.w as isize))
        })
        .collect()
}

/// Output columns `ox` in `0..ow` with `0 <= ox * sw + tap - pw < w`, using overflow-safe unsigned arithmetic.
#[inline]
fn valid_cols_u(ow: usize, sw: usize, tap: usize, pw: usize, w: usize) -> std::ops::Range<usize> {
    if ow == 0 || w == 0 {
        return 0..0;
    }
    let max_ox_end = (usize::MAX / sw).saturating_add(1);
    let (lo, hi) = if tap >= pw {
        let shift = tap - pw;
        if shift >= w {
            return 0..0;
        }
        (0, (w - shift).div_ceil(sw))
    } else {
        let need = pw - tap;
        let lo = need.div_ceil(sw);
        let hi = match need.checked_add(w) {
            Some(lim) => lim.div_ceil(sw),
            None => max_ox_end,
        };
        (lo, hi)
    };
    let hi = hi.min(max_ox_end).min(ow);
    let lo = lo.min(hi);
    lo..hi
}

/// One output plane of a depthwise conv: `op += conv(xp, wk)` (`op` is `oh x ow`, `xp` one `h x w` input plane).
#[inline(always)]
pub(super) fn depthwise_plane(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    let [sh, sw] = g.p.stride;
    let [ph, pw] = g.p.padding;
    let [dh, dw] = g.p.dilation;
    for oy in 0..g.oh {
        let orow = &mut op[oy * g.ow..(oy + 1) * g.ow];
        for ky in 0..g.kh {
            let Some(iy) = oy
                .checked_mul(sh)
                .and_then(|v| ky.checked_mul(dh).and_then(|d| v.checked_add(d)))
                .and_then(|v| v.checked_sub(ph))
            else {
                continue;
            };
            if iy >= g.h {
                continue;
            }
            let xrow = &xp[iy * g.w..(iy + 1) * g.w];
            for kx in 0..g.kw {
                let wv = wk[ky * g.kw + kx];
                let Some(tap) = kx.checked_mul(dw) else {
                    continue;
                };
                let r = valid_cols_u(g.ow, sw, tap, pw, g.w);
                if r.is_empty() {
                    continue;
                }
                if sw == 1 {
                    let s0 = if tap >= pw { r.start + (tap - pw) } else { r.start - (pw - tap) };
                    let len = r.len();
                    for (o, &xv) in orow[r].iter_mut().zip(&xrow[s0..s0 + len]) {
                        *o += wv * xv;
                    }
                } else if tap >= pw {
                    let shift = tap - pw;
                    for ox in r {
                        orow[ox] += wv * xrow[ox * sw + shift];
                    }
                } else {
                    let need = pw - tap;
                    for ox in r {
                        orow[ox] += wv * xrow[ox * sw - need];
                    }
                }
            }
        }
    }
}

/// Output elements per [`depthwise_forward`] task (whole planes, at least one): small planes are grouped so a
/// task is not dominated by scheduling.
const DW_TASK_OUTPUTS: usize = 1 << 14;

/// Look up input channel `ic` of image `img` directly from `x.parts` without building a `Vec<Vec<&[f32]>>`.
#[inline]
fn input_plane<'a>(x: ConvInput<'a>, img: usize, mut ic: usize, hw: usize) -> &'a [f32] {
    for p in x.parts {
        if ic < p.c {
            let start = p.offset + img * p.img_stride + ic * hw;
            return &p.data[start..start + hw];
        }
        ic -= p.c;
    }
    unreachable!("depthwise: input channel out of bounds")
}

pub(in super::super) fn depthwise_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    if g.n == 0 || g.o == 0 || ohw == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(target_arch = "x86_64")]
    let avx2_plan = if avx2 { avx2::plan_avx2(g) } else { avx2::Avx2Plan::Fallback };
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    let wasm_pad = if !accumulate && g.p.stride == [1, 1] { wasm_pad_dims(g) } else { None };
    let total = g.n * g.o;
    let per_task = DW_TASK_OUTPUTS.div_ceil(ohw).min(total.div_ceil(4 * rayon::current_num_threads())).max(1);
    par_chunks(out, per_task * ohw, &|t, ops| {
        let run_ops = |ops: &mut [f32],
                       #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
                       pad: &mut Vec<f32>| {
            for (p, op) in ops.chunks_mut(ohw).enumerate() {
                let plane = t * per_task + p;
                let (img, oc) = (plane / g.o, plane % g.o);
                let xp = input_plane(x, img, oc / mult, hw);
                let wk = &wt[oc * khw..(oc + 1) * khw];
                #[cfg(target_arch = "x86_64")]
                if !matches!(avx2_plan, avx2::Avx2Plan::Fallback) && wk.iter().all(|v| v.is_finite()) {
                    // SAFETY: AVX2 and FMA were detected above and `avx2_plan` checked all padded dimensions.
                    unsafe {
                        avx2::run_avx2_plan(
                            avx2_plan,
                            xp,
                            wk,
                            op,
                            g,
                            accumulate,
                            epi.map(|e| e.rows_from(oc)),
                            pad,
                        );
                    }
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                if let Some((owp, wpd, pad_len)) = wasm_pad
                    && wk.iter().all(|v| v.is_finite())
                {
                    depthwise_s1_wasm(xp, wk, op, g, (owp, wpd, pad_len), epi.map(|e| e.rows_from(oc)), pad);
                    continue;
                }
                if !accumulate {
                    op.fill(0.0);
                }
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 and FMA were detected above.
                    unsafe { avx2::depthwise_plane_avx2(xp, wk, op, g) };
                } else {
                    depthwise_plane(xp, wk, op, g);
                }
                #[cfg(not(target_arch = "x86_64"))]
                depthwise_plane(xp, wk, op, g);
                if let Some(e) = epi {
                    e.apply(oc, op);
                }
            }
        };
        #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
        with_dw_pad(|pad| run_ops(ops, pad));
        #[cfg(not(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128"))))]
        run_ops(ops);
    });
}

/// Maximum padded plane scratch (4M floats = 16 MiB) allowed for [`depthwise_s1_wasm`] before falling back
/// to the zero-scratch scalar path.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
const DW_WASM_MAX_PAD: usize = 1 << 22;

/// Checked `(owp, wpd, h * wpd)` for [`depthwise_s1_wasm`], or `None` if sizing overflows or exceeds
/// [`DW_WASM_MAX_PAD`].
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn wasm_pad_dims(g: &ConvGeom) -> Option<(usize, usize, usize)> {
    let pw = g.p.padding[1];
    let dw = g.p.dilation[1];
    let owp = g.ow.checked_add(3)? / 4 * 4;
    let tap_span = dw.checked_mul(g.kw.saturating_sub(1))?.checked_add(owp)?;
    let row_span = pw.checked_add(g.w)?;
    let wpd = tap_span.max(row_span);
    let pad_len = g.h.checked_mul(wpd)?;
    (pad_len <= DW_WASM_MAX_PAD).then_some((owp, wpd, pad_len))
}

/// One output plane of a stride-1 overwriting depthwise conv with Wasm SIMD128 (`f32x4` × 4 register
/// accumulators): pads the input plane once, computes taps in `(ky, kx)` order, and fuses `bias`,
/// `bias + SiLU`, and `bias + ReLU` directly in `v128` registers.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn depthwise_s1_wasm(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    (owp, wpd, pad_len): (usize, usize, usize),
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    use std::arch::wasm32::*;
    let (ph, pw) = (g.p.padding[0], g.p.padding[1]);
    let (dh, dw) = (g.p.dilation[0], g.p.dilation[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    pad.clear();
    pad.resize(pad_len, 0.0);
    if w > 0 {
        for (prow, xrow) in pad.chunks_exact_mut(wpd).zip(xp.chunks_exact(w)) {
            prow[pw..pw + w].copy_from_slice(xrow);
        }
    }
    let (reg, bias) = match epi {
        Some(e) if matches!(e.act, None | Some(UnaryOp::Silu | UnaryOp::Relu)) => {
            (true, e.bias.map_or(0.0, |b| b[0]))
        }
        _ => (false, 0.0),
    };
    let act = epi.and_then(|e| e.act);
    let bv = f32x4_splat(if act.is_none() && bias == 0.0 { -0.0 } else { bias });
    let zero = f32x4_splat(0.0);
    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 4).min(4);
            let mut acc = [zero; 4];
            for ky in 0..g.kh {
                let Some(iy) =
                    ky.checked_mul(dh).and_then(|v| oy.checked_add(v)).and_then(|v| v.checked_sub(ph))
                else {
                    continue;
                };
                if iy >= h {
                    continue;
                }
                let prow = &pad[iy * wpd..(iy + 1) * wpd];
                for kx in 0..g.kw {
                    let wv = f32x4_splat(wk[ky * g.kw + kx]);
                    let base = x0 + kx * dw;
                    assert!(base + 4 * nv <= wpd);
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        // SAFETY: `base + 4 * nv <= wpd` (asserted), inside this padded row.
                        let xv = unsafe { v128_load(prow.as_ptr().add(base + 4 * v) as *const v128) };
                        *a = f32x4_add(*a, f32x4_mul(wv, xv));
                    }
                }
            }
            for (v, a) in acc.iter().enumerate().take(nv) {
                let c = x0 + 4 * v;
                let r = if !reg {
                    *a
                } else {
                    let x = f32x4_add(*a, bv);
                    match act {
                        None => x,
                        Some(UnaryOp::Silu) => crate::cpu::elementwise::silu4_wasm(x),
                        Some(UnaryOp::Relu) => f32x4_pmax(zero, x),
                        _ => unreachable!(),
                    }
                };
                if c + 4 <= ow {
                    // SAFETY: columns `c..c + 4` are inside `orow`.
                    unsafe { v128_store(orow.as_mut_ptr().add(c) as *mut v128, r) };
                } else if c < ow {
                    let mut tmp = [0f32; 4];
                    // SAFETY: `tmp` holds 4 floats.
                    unsafe { v128_store(tmp.as_mut_ptr() as *mut v128, r) };
                    orow[c..].copy_from_slice(&tmp[..ow - c]);
                }
            }
            x0 += 4 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// Scalar-only depthwise forward for direct SIMD vs scalar parity tests.
#[cfg(test)]
pub(in super::super) fn depthwise_forward_scalar(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    for (plane, op) in out.chunks_mut(ohw).enumerate() {
        let (img, oc) = (plane / g.o, plane % g.o);
        let xp = input_plane(x, img, oc / mult, hw);
        let wk = &wt[oc * khw..(oc + 1) * khw];
        if !accumulate {
            op.fill(0.0);
        }
        depthwise_plane(xp, wk, op, g);
        if let Some(e) = epi {
            e.apply(oc, op);
        }
    }
}

pub(in super::super) fn depthwise_backward_input(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    if g.n == 0 || g.c == 0 || hw == 0 || ohw == 0 {
        return;
    }
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, dh) = (g.p.padding[0] as isize, g.p.dilation[0] as isize);
    let kx_cols = precompute_kx_cols(g);
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2");
    let total = g.n * g.c;
    let per_task = DW_TASK_OUTPUTS.div_ceil(hw).min(total.div_ceil(4 * rayon::current_num_threads())).max(1);
    par_chunks(gx, per_task * hw, &|t, gxps| {
        for (p, gxp) in gxps.chunks_mut(hw).enumerate() {
            let plane = t * per_task + p;
            let (img, ic) = (plane / g.c, plane % g.c);
            for oc in ic * mult..(ic + 1) * mult {
                let gp = &gy[(img * g.o + oc) * ohw..][..ohw];
                let wk = &wt[oc * khw..(oc + 1) * khw];
                for oy in 0..g.oh {
                    let grow = &gp[oy * g.ow..(oy + 1) * g.ow];
                    for ky in 0..g.kh {
                        let iy = oy as isize * sh - ph + ky as isize * dh;
                        if iy < 0 || iy >= g.h as isize {
                            continue;
                        }
                        let xrow = &mut gxp[iy as usize * g.w..(iy as usize + 1) * g.w];
                        for (kx, (d, r)) in kx_cols.iter().enumerate() {
                            if r.is_empty() {
                                continue;
                            }
                            let wv = wk[ky * g.kw + kx];
                            if sw == 1 {
                                let s0 = (r.start as isize + *d) as usize;
                                let len = r.len();
                                let dst = &mut xrow[s0..s0 + len];
                                let src = &grow[r.clone()];
                                #[cfg(target_arch = "x86_64")]
                                if avx2 && len >= 8 {
                                    // SAFETY: AVX2 was detected above.
                                    unsafe { avx2::mul_add_row_avx2(dst, src, wv) };
                                    continue;
                                }
                                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                                if len >= 4 {
                                    mul_add_row_wasm(dst, src, wv);
                                    continue;
                                }
                                for (xv, &gv) in dst.iter_mut().zip(src) {
                                    *xv += wv * gv;
                                }
                            } else if sw == 2 {
                                let base = (r.start as isize * 2 + *d) as usize;
                                let src = &grow[r.clone()];
                                #[cfg(target_arch = "x86_64")]
                                if avx2 && src.len() >= 8 {
                                    // SAFETY: AVX2 was detected above.
                                    unsafe { avx2::mul_add_stride2_avx2(&mut xrow[base..], src, wv) };
                                    continue;
                                }
                                for (k, &gv) in src.iter().enumerate() {
                                    xrow[base + k * 2] += wv * gv;
                                }
                            } else {
                                let base = (r.start as isize * sw + *d) as usize;
                                let step = sw as usize;
                                for (k, &gv) in grow[r.clone()].iter().enumerate() {
                                    xrow[base + k * step] += wv * gv;
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}

pub(in super::super) fn depthwise_backward_weight(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    if g.n == 0 || g.o == 0 || khw == 0 || ohw == 0 || hw == 0 {
        return;
    }
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, dh) = (g.p.padding[0] as isize, g.p.dilation[0] as isize);
    let kx_cols = precompute_kx_cols(g);
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2");
    let per_task =
        DW_TASK_OUTPUTS.div_ceil(g.n * ohw).min(g.o.div_ceil(4 * rayon::current_num_threads())).max(1);
    par_chunks(gw, per_task * khw, &|t, gks| {
        for (p, gk) in gks.chunks_mut(khw).enumerate() {
            let oc = t * per_task + p;
            let ic = oc / mult;
            for img in 0..g.n {
                let xp = &x[(img * g.c + ic) * hw..][..hw];
                let gp = &gy[(img * g.o + oc) * ohw..][..ohw];
                for ky in 0..g.kh {
                    for (kx, (d, cols)) in kx_cols.iter().enumerate() {
                        if cols.is_empty() {
                            continue;
                        }
                        let mut acc = 0f32;
                        for oy in 0..g.oh {
                            let iy = oy as isize * sh - ph + ky as isize * dh;
                            if iy < 0 || iy >= g.h as isize {
                                continue;
                            }
                            let xrow = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
                            let grow = &gp[oy * g.ow..(oy + 1) * g.ow];
                            if sw == 1 {
                                let s0 = (cols.start as isize + *d) as usize;
                                let gs = &grow[cols.clone()];
                                let xs = &xrow[s0..s0 + cols.len()];
                                #[cfg(target_arch = "x86_64")]
                                if avx2 && gs.len() >= 8 {
                                    // SAFETY: AVX2 was detected above.
                                    acc += unsafe { avx2::dot_avx2(gs, xs) };
                                    continue;
                                }
                                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                                if gs.len() >= 8 {
                                    acc += dot_wasm(gs, xs);
                                    continue;
                                }
                                acc += dot_scalar(gs, xs);
                            } else if sw == 2 {
                                let base = (cols.start as isize * 2 + *d) as usize;
                                let gs = &grow[cols.clone()];
                                let xs = &xrow[base..];
                                #[cfg(target_arch = "x86_64")]
                                if avx2 && gs.len() >= 8 {
                                    // SAFETY: AVX2 was detected above.
                                    acc += unsafe { avx2::dot_stride2_avx2(gs, xs) };
                                    continue;
                                }
                                let vec_n = gs.len().min(xs.len() / 2) / 8 * 8;
                                let mut sub = [0f32; 8];
                                for k in (0..vec_n).step_by(8) {
                                    for lane in 0..8 {
                                        sub[lane] += gs[k + lane] * xs[(k + lane) * 2];
                                    }
                                }
                                let mut s: f32 = (vec_n..gs.len()).map(|j| gs[j] * xs[j * 2]).sum();
                                for v in sub {
                                    s += v;
                                }
                                acc += s;
                            } else {
                                let base = (cols.start as isize * sw + *d) as usize;
                                let step = sw as usize;
                                for (k, &gv) in grow[cols.clone()].iter().enumerate() {
                                    acc += gv * xrow[base + k * step];
                                }
                            }
                        }
                        gk[ky * g.kw + kx] += acc;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::backend::{Conv2dParams, UnaryOp};
    use crate::cpu::conv::ConvPart;
    use crate::cpu::gemm::BiasAct;
    use crate::rng::Rng;

    #[test]
    fn depthwise_k3s1_s2_and_relu_silu_match_scalar() {
        let mut rng = Rng::seed(77);
        let cases = [
            // (n, c, h, w, kh, kw, stride, pad, dil)
            (1, 8, 17, 19, 3, 3, 1, 1, 1),
            (2, 16, 40, 40, 3, 3, 1, 1, 1),
            (1, 8, 3, 5, 3, 3, 1, 0, 1),
            (1, 8, 33, 35, 3, 3, 2, 1, 1),
            (2, 12, 41, 39, 5, 5, 2, 2, 1),
            (1, 6, 17, 21, 3, 3, 2, 2, 2),
            (1, 8, 25, 27, 5, 5, 1, 2, 1),
        ];
        for (n, c, h, w, kh, kw, s, pad, dil) in cases {
            let p = Conv2dParams { stride: [s, s], padding: [pad, pad], dilation: [dil, dil], groups: c };
            let [oh, ow] = p.output_hw(h, w, kh, kw).unwrap();
            let g = ConvGeom { n, c, h, w, o: c, kh, kw, oh, ow, p };
            let x: Vec<f32> = (0..n * c * h * w).map(|_| rng.normal()).collect();
            let wt: Vec<f32> = (0..c * kh * kw).map(|_| rng.normal() * 0.3).collect();
            let bias: Vec<f32> = (0..c).map(|_| rng.normal() * 0.2).collect();
            let part = [ConvPart::contiguous(&x, c, h * w)];
            let input = ConvInput { parts: &part };

            for act in [None, Some(UnaryOp::Relu), Some(UnaryOp::Silu)] {
                for &acc in &[false, true] {
                    let init: Vec<f32> = (0..n * c * oh * ow).map(|_| rng.normal() * 0.5).collect();
                    let mut got = init.clone();
                    let mut want = init;
                    let epi = Some(BiasAct { bias: Some(&bias), act });
                    depthwise_forward(input, &wt, &mut got, &g, acc, epi);
                    depthwise_forward_scalar(input, &wt, &mut want, &g, acc, epi);
                    for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                            "dw fwd mismatch ({n},{c},{h},{w}) k={kh} s={s} act={act:?} acc={acc} at {i}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_s2_backward_grad_check_and_nonfinite_weight() {
        let mut rng = Rng::seed(88);
        let x = Tensor::randn([2, 8, 19, 21], 0.0, 1.0, &mut rng).requires_grad_();
        let w = Tensor::randn([8, 1, 3, 3], 0.0, 0.4, &mut rng).requires_grad_();
        let b = Tensor::randn([8], 0.0, 0.2, &mut rng).requires_grad_();
        crate::testing::grad_check(|xs| xs[0].conv2d(&xs[1], Some(&xs[2]), 2, 1, 1, 8), &[x.clone(), w, b])
            .unwrap();

        let w5 = Tensor::randn([8, 1, 5, 5], 0.0, 0.3, &mut rng).requires_grad_();
        crate::testing::grad_check(|xs| xs[0].conv2d(&xs[1], None, 2, 2, 1, 8), &[x, w5]).unwrap();

        // Non-finite weight must fall back cleanly instead of producing 0.0 * Inf = NaN on padded columns.
        let p = Conv2dParams { stride: [2, 2], padding: [1, 1], dilation: [1, 1], groups: 2 };
        let [oh, ow] = p.output_hw(6, 6, 3, 3).unwrap();
        let g = ConvGeom { n: 1, c: 2, h: 6, w: 6, o: 2, kh: 3, kw: 3, oh, ow, p };
        let xdata = vec![1.0f32; 2 * 6 * 6];
        let mut wt = vec![0.5f32; 2 * 9];
        wt[0] = f32::INFINITY;
        let part = [ConvPart::contiguous(&xdata, 2, 36)];
        let mut got = vec![0.0; 2 * oh * ow];
        let mut want = vec![0.0; 2 * oh * ow];
        depthwise_forward(ConvInput { parts: &part }, &wt, &mut got, &g, false, None);
        depthwise_forward_scalar(ConvInput { parts: &part }, &wt, &mut want, &g, false, None);
        for (a, b) in got.iter().zip(&want) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // Extreme vertical padding/dilation with a non-finite weight must not overflow in fallback indexing.
        let p_big = Conv2dParams {
            stride: [1, 1],
            padding: [1usize << (usize::BITS - 2), 0],
            dilation: [1usize << (usize::BITS - 1), 1],
            groups: 2,
        };
        let [oh_b, ow_b] = p_big.output_hw(3, 1, 2, 1).unwrap();
        let g_big = ConvGeom { n: 1, c: 2, h: 3, w: 1, o: 2, kh: 2, kw: 1, oh: oh_b, ow: ow_b, p: p_big };
        let x_big = vec![1.0f32; 2 * 3];
        let wt_big = vec![f32::INFINITY, 1.0, 1.0, 1.0];
        let part_big = [ConvPart::contiguous(&x_big, 2, 3)];
        let mut out_big = vec![0.0; 2 * oh_b * ow_b];
        depthwise_forward(ConvInput { parts: &part_big }, &wt_big, &mut out_big, &g_big, false, None);
        assert_eq!(out_big, vec![0.0; 2 * oh_b * ow_b]);

        // Odd-width stride-2 dot product (16 samples in 31 floats) matches scalar bit-for-bit.
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let mut g_s2 = vec![0.0f32; 16];
            let mut x_s2 = vec![0.0f32; 31];
            g_s2[0] = 1.0;
            x_s2[0] = (1u32 << 24) as f32;
            g_s2[8] = 1.0;
            x_s2[16] = -((1u32 << 24) as f32);
            g_s2[9] = 1.0;
            x_s2[18] = 0.5;
            // SAFETY: AVX2 detected above.
            let got = unsafe { avx2::dot_stride2_avx2(&g_s2, &x_s2) };
            let vec_n = g_s2.len().min(x_s2.len() / 2) / 8 * 8;
            let mut sub = [0f32; 8];
            for k in (0..vec_n).step_by(8) {
                for lane in 0..8 {
                    sub[lane] += g_s2[k + lane] * x_s2[(k + lane) * 2];
                }
            }
            let mut want: f32 = (vec_n..g_s2.len()).map(|j| g_s2[j] * x_s2[j * 2]).sum();
            for v in sub {
                want += v;
            }
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }
}
