//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) depthwise convolution kernels: ring-buffered 3×3 stride-1,
//! general stride-1, even/odd deinterleaved stride-2 forward (with in-register bias + ReLU/SiLU fusion and
//! accumulation), and stride-1 / stride-2 backward helpers.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::super::shifted::deinterleave_f32_even_odd;
use crate::backend::UnaryOp;
use crate::cpu::conv::ConvGeom;
use crate::cpu::gemm::BiasAct;

/// Maximum padded scratch (4M floats = 16 MiB) allowed for Wasm SIMD128 depthwise kernels before falling back
/// to the zero-scratch scalar path.
const DW_WASM_MAX_PAD: usize = 1 << 22;

#[derive(Clone, Copy)]
pub(super) enum WasmPlan {
    /// 3-row sliding ring buffer for `3×3` stride-1 dilation-1 (`3 * wpd` floats).
    K3S1 { owp: usize, wpd: usize },
    /// Full padded plane for general stride-1 (`h * wpd` floats).
    S1 { owp: usize, wpd: usize, pad_len: usize },
    /// Even/odd deinterleaved padded planes for stride-2 (`(h + 1) * 2 * pairs` floats).
    S2 { owp: usize, pairs: usize },
    /// Zero-scratch scalar fallback.
    Fallback,
}

/// Select the best overflow-checked Wasm SIMD128 depthwise plan for `g`.
pub(super) fn plan_wasm(g: &ConvGeom) -> WasmPlan {
    let pw = g.p.padding[1];
    let dw = g.p.dilation[1];
    let Some(owp) = g.ow.checked_add(3).map(|v| v / 4 * 4) else {
        return WasmPlan::Fallback;
    };
    if g.p.stride == [1, 1] {
        if g.kh == 3 && g.kw == 3 && g.p.dilation == [1, 1] {
            let wpd = owp.checked_add(2).and_then(|t| pw.checked_add(g.w).map(|r| t.max(r)));
            if let Some(wpd) = wpd
                && wpd.checked_mul(3).is_some_and(|n| n <= DW_WASM_MAX_PAD)
            {
                return WasmPlan::K3S1 { owp, wpd };
            }
            return WasmPlan::Fallback;
        }
        let wpd = dw
            .checked_mul(g.kw.saturating_sub(1))
            .and_then(|v| v.checked_add(owp))
            .and_then(|t| pw.checked_add(g.w).map(|r| t.max(r)));
        if let Some(wpd) = wpd
            && let Some(pad_len) = g.h.checked_mul(wpd)
            && pad_len <= DW_WASM_MAX_PAD
        {
            return WasmPlan::S1 { owp, wpd, pad_len };
        }
        return WasmPlan::Fallback;
    }
    if g.p.stride[1] == 2 {
        let tap_pairs =
            dw.checked_mul(g.kw.saturating_sub(1)).map(|v| v / 2).and_then(|v| v.checked_add(owp));
        let row_pairs = pw.checked_add(g.w).map(|v| v.div_ceil(2));
        if let (Some(tp), Some(rp)) = (tap_pairs, row_pairs) {
            let pairs = tp.max(rp);
            let total = g.h.checked_add(1).and_then(|r| r.checked_mul(2)).and_then(|r| r.checked_mul(pairs));
            if total.is_some_and(|n| n <= DW_WASM_MAX_PAD) {
                return WasmPlan::S2 { owp, pairs };
            }
        }
    }
    WasmPlan::Fallback
}

/// Dispatch the chosen Wasm SIMD128 depthwise plan.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_wasm_plan(
    plan: WasmPlan,
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    match plan {
        WasmPlan::K3S1 { owp, wpd } => depthwise_k3s1_wasm(xp, wk, op, g, owp, wpd, accumulate, epi, pad),
        WasmPlan::S1 { owp, wpd, pad_len } => {
            depthwise_s1_wasm(xp, wk, op, g, owp, wpd, pad_len, accumulate, epi, pad)
        }
        WasmPlan::S2 { owp, pairs } => depthwise_s2_wasm(xp, wk, op, g, owp, pairs, accumulate, epi, pad),
        WasmPlan::Fallback => unreachable!(),
    }
}

/// 8-lane Wasm SIMD128 dot product (2×4 accumulators), bitwise identical to `dot_scalar`.
pub(super) fn dot_wasm(a: &[f32], b: &[f32]) -> f32 {
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

/// 8-lane Wasm SIMD128 strided dot product `sum_k g[k] * x[2 * k]`, bitwise identical to the 8-lane scalar
/// stride-2 reduction order.
pub(super) fn dot_stride2_wasm(g: &[f32], x: &[f32]) -> f32 {
    let n = g.len();
    assert!(n == 0 || (n - 1) * 2 < x.len(), "dot_stride2_wasm: x slice too short");
    let vec_n = n.min(x.len() / 2) / 8 * 8;
    let mut acc = [0f32; 8];
    // SAFETY: `k + 8 <= vec_n <= n` and `2 * (k + 8) <= x.len()`.
    unsafe {
        let mut s0 = f32x4_splat(0.0);
        let mut s1 = f32x4_splat(0.0);
        for k in (0..vec_n).step_by(8) {
            let gp = g.as_ptr().add(k);
            let xp = x.as_ptr().add(2 * k);
            let g0 = v128_load(gp as *const v128);
            let g1 = v128_load(gp.add(4) as *const v128);
            let x0 = v128_load(xp as *const v128);
            let x1 = v128_load(xp.add(4) as *const v128);
            let x2 = v128_load(xp.add(8) as *const v128);
            let x3 = v128_load(xp.add(12) as *const v128);
            let ev0 = i32x4_shuffle::<0, 2, 4, 6>(x0, x1);
            let ev1 = i32x4_shuffle::<0, 2, 4, 6>(x2, x3);
            s0 = f32x4_add(s0, f32x4_mul(g0, ev0));
            s1 = f32x4_add(s1, f32x4_mul(g1, ev1));
        }
        v128_store(acc.as_mut_ptr() as *mut v128, s0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, s1);
    }
    let mut s: f32 = (vec_n..n).map(|k| g[k] * x[2 * k]).sum();
    for v in acc {
        s += v;
    }
    s
}

/// 8-wide Wasm SIMD128 row update `dst[i] += wv * src[i]`, bitwise identical to scalar `*d += wv * s`.
pub(super) fn mul_add_row_wasm(dst: &mut [f32], src: &[f32], wv: f32) {
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

/// 4-wide Wasm SIMD128 stride-2 scatter update `dst[2 * k] += wv * src[k]`, preserving odd elements of `dst`
/// verbatim bit-for-bit.
pub(super) fn mul_add_stride2_wasm(dst: &mut [f32], src: &[f32], wv: f32) {
    let n = src.len();
    assert!(n == 0 || (n - 1) * 2 < dst.len(), "mul_add_stride2_wasm: dst slice too short");
    let vec_n = n.min(dst.len() / 2) / 4 * 4;
    let w = f32x4_splat(wv);
    // SAFETY: `k + 4 <= vec_n <= n` and `2 * (k + 4) <= dst.len()`.
    unsafe {
        for k in (0..vec_n).step_by(4) {
            let p = f32x4_mul(w, v128_load(src.as_ptr().add(k) as *const v128));
            let dptr = dst.as_mut_ptr().add(2 * k);
            let d0 = v128_load(dptr as *const v128);
            let d1 = v128_load(dptr.add(4) as *const v128);
            let even = i32x4_shuffle::<0, 2, 4, 6>(d0, d1);
            let odd = i32x4_shuffle::<1, 3, 5, 7>(d0, d1);
            let even_up = f32x4_add(even, p);
            let o0 = i32x4_shuffle::<0, 4, 1, 5>(even_up, odd);
            let o1 = i32x4_shuffle::<2, 6, 3, 7>(even_up, odd);
            v128_store(dptr as *mut v128, o0);
            v128_store(dptr.add(4) as *mut v128, o1);
        }
    }
    for (k, &s) in src.iter().enumerate().skip(vec_n) {
        dst[2 * k] += wv * s;
    }
}

/// Initialise up to 4 Wasm SIMD128 accumulators (`0.0`, or loaded from `orow` when `accumulate`).
#[inline]
fn init_acc_wasm(orow: &[f32], x0: usize, nv: usize, accumulate: bool) -> [v128; 4] {
    let mut acc = [f32x4_splat(0.0); 4];
    if accumulate {
        let ow = orow.len();
        for (v, a) in acc.iter_mut().enumerate().take(nv) {
            let c = x0 + 4 * v;
            if c + 4 <= ow {
                // SAFETY: `c + 4 <= ow == orow.len()`.
                *a = unsafe { v128_load(orow.as_ptr().add(c) as *const v128) };
            } else {
                let mut tmp = [0f32; 4];
                let len = ow.saturating_sub(c).min(4);
                tmp[..len].copy_from_slice(&orow[c..c + len]);
                // SAFETY: `tmp` holds 4 floats.
                *a = unsafe { v128_load(tmp.as_ptr() as *const v128) };
            }
        }
    }
    acc
}

/// Apply in-register bias + optional SiLU/ReLU and store `nv` `f32x4` vectors (`x0..x0 + 4 * nv`) into `orow`.
#[inline]
fn store_epi_wasm(
    orow: &mut [f32],
    x0: usize,
    nv: usize,
    acc: &[v128; 4],
    reg: bool,
    act: Option<UnaryOp>,
    bv: v128,
) {
    let ow = orow.len();
    let zero = f32x4_splat(0.0);
    for (v, &a) in acc.iter().enumerate().take(nv) {
        let c = x0 + 4 * v;
        let r = if !reg {
            a
        } else {
            let x = f32x4_add(a, bv);
            match act {
                None => x,
                Some(UnaryOp::Silu) => crate::cpu::elementwise::silu4_wasm(x),
                // `f32x4_pmax(zero, x)` = `0.0 < x ? x : +0.0`, matching `x.max(0.0)` on `-0.0` and NaN.
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
}

/// Specialized 3×3 stride-1 dilation-1 depthwise conv using a 3-row sliding ring buffer (`3 * wpd` floats)
/// with unrolled interior rows and in-register bias + SiLU/ReLU epilogue.
#[allow(clippy::too_many_arguments)]
fn depthwise_k3s1_wasm(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    owp: usize,
    wpd: usize,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    let (ph, pw) = (g.p.padding[0], g.p.padding[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    pad.clear();
    pad.resize(3 * wpd, 0.0);
    let mut loaded = [usize::MAX; 3];

    let (reg, bias) = match epi {
        Some(e) if matches!(e.act, None | Some(UnaryOp::Silu | UnaryOp::Relu)) => {
            (true, e.bias.map_or(0.0, |b| b[0]))
        }
        _ => (false, 0.0),
    };
    let act = epi.and_then(|e| e.act);
    let bv = f32x4_splat(if act.is_none() && bias == 0.0 { -0.0 } else { bias });

    let wv = [
        f32x4_splat(wk[0]),
        f32x4_splat(wk[1]),
        f32x4_splat(wk[2]),
        f32x4_splat(wk[3]),
        f32x4_splat(wk[4]),
        f32x4_splat(wk[5]),
        f32x4_splat(wk[6]),
        f32x4_splat(wk[7]),
        f32x4_splat(wk[8]),
    ];

    assert!(owp + 2 <= wpd && pw + w <= wpd && xp.len() >= h * w && op.len() >= g.oh * ow);

    for oy in 0..g.oh {
        let iys = [
            oy.checked_sub(ph).filter(|&y| y < h),
            oy.checked_add(1).and_then(|v| v.checked_sub(ph)).filter(|&y| y < h),
            oy.checked_add(2).and_then(|v| v.checked_sub(ph)).filter(|&y| y < h),
        ];
        if w > 0 {
            for iy in iys.into_iter().flatten() {
                let slot = iy % 3;
                if loaded[slot] != iy {
                    pad[slot * wpd + pw..slot * wpd + pw + w].copy_from_slice(&xp[iy * w..(iy + 1) * w]);
                    loaded[slot] = iy;
                }
            }
        }
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        if let [Some(y0), Some(y1), Some(y2)] = iys {
            let p0 = &pad[(y0 % 3) * wpd..(y0 % 3 + 1) * wpd];
            let p1 = &pad[(y1 % 3) * wpd..(y1 % 3 + 1) * wpd];
            let p2 = &pad[(y2 % 3) * wpd..(y2 % 3 + 1) * wpd];
            let mut x0 = 0;
            while x0 < owp {
                let nv = ((owp - x0) / 4).min(4);
                let mut acc = init_acc_wasm(orow, x0, nv, accumulate);
                // SAFETY: `x0 + 4 * nv + 2 <= owp + 2 <= wpd` for all three padded ring rows.
                unsafe {
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let r0 = p0.as_ptr().add(x0 + 4 * v);
                        *a = f32x4_add(*a, f32x4_mul(wv[0], v128_load(r0 as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[1], v128_load(r0.add(1) as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[2], v128_load(r0.add(2) as *const v128)));
                    }
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let r1 = p1.as_ptr().add(x0 + 4 * v);
                        *a = f32x4_add(*a, f32x4_mul(wv[3], v128_load(r1 as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[4], v128_load(r1.add(1) as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[5], v128_load(r1.add(2) as *const v128)));
                    }
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let r2 = p2.as_ptr().add(x0 + 4 * v);
                        *a = f32x4_add(*a, f32x4_mul(wv[6], v128_load(r2 as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[7], v128_load(r2.add(1) as *const v128)));
                        *a = f32x4_add(*a, f32x4_mul(wv[8], v128_load(r2.add(2) as *const v128)));
                    }
                }
                store_epi_wasm(orow, x0, nv, &acc, reg, act, bv);
                x0 += 4 * nv;
            }
        } else {
            let mut x0 = 0;
            while x0 < owp {
                let nv = ((owp - x0) / 4).min(4);
                let mut acc = init_acc_wasm(orow, x0, nv, accumulate);
                for (ky, iy) in iys.into_iter().enumerate() {
                    let Some(iy) = iy else { continue };
                    let prow = &pad[(iy % 3) * wpd..(iy % 3 + 1) * wpd];
                    let (w0, w1, w2) = (wv[ky * 3], wv[ky * 3 + 1], wv[ky * 3 + 2]);
                    // SAFETY: `x0 + 4 * nv + 2 <= owp + 2 <= wpd`.
                    unsafe {
                        for (v, a) in acc.iter_mut().enumerate().take(nv) {
                            let r = prow.as_ptr().add(x0 + 4 * v);
                            *a = f32x4_add(*a, f32x4_mul(w0, v128_load(r as *const v128)));
                            *a = f32x4_add(*a, f32x4_mul(w1, v128_load(r.add(1) as *const v128)));
                            *a = f32x4_add(*a, f32x4_mul(w2, v128_load(r.add(2) as *const v128)));
                        }
                    }
                }
                store_epi_wasm(orow, x0, nv, &acc, reg, act, bv);
                x0 += 4 * nv;
            }
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// General stride-1 Wasm SIMD128 depthwise conv with TLS padded-plane reuse and in-register bias + SiLU/ReLU.
#[allow(clippy::too_many_arguments)]
fn depthwise_s1_wasm(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    owp: usize,
    wpd: usize,
    pad_len: usize,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
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
    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 4).min(4);
            let mut acc = init_acc_wasm(orow, x0, nv, accumulate);
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
            store_epi_wasm(orow, x0, nv, &acc, reg, act, bv);
            x0 += 4 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// Stride-2 (`stride[1] == 2`) Wasm SIMD128 depthwise conv: pads each input row and deinterleaves it into
/// contiguous even and odd phase rows so every kernel tap is a contiguous 4-float `v128_load`.
#[allow(clippy::too_many_arguments)]
fn depthwise_s2_wasm(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    owp: usize,
    pairs: usize,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    let (sh, ph, pw) = (g.p.stride[0], g.p.padding[0], g.p.padding[1]);
    let (dh, dw) = (g.p.dilation[0], g.p.dilation[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    let row_stride = 2 * pairs;
    // Layout in `pad`: `h` rows of `[even(pairs), odd(pairs)]` followed by one `2 * pairs` staging row.
    pad.clear();
    pad.resize((h + 1) * row_stride, 0.0);
    let (phases, tmp_row) = pad.split_at_mut(h * row_stride);
    if w > 0 {
        for (y, phase_row) in phases.chunks_exact_mut(row_stride).enumerate() {
            tmp_row[pw..pw + w].copy_from_slice(&xp[y * w..(y + 1) * w]);
            let (even, odd) = phase_row.split_at_mut(pairs);
            deinterleave_f32_even_odd(tmp_row, even, odd);
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

    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 4).min(4);
            let mut acc = init_acc_wasm(orow, x0, nv, accumulate);
            for ky in 0..g.kh {
                let Some(iy) = oy
                    .checked_mul(sh)
                    .and_then(|v| ky.checked_mul(dh).and_then(|d| v.checked_add(d)))
                    .and_then(|v| v.checked_sub(ph))
                else {
                    continue;
                };
                if iy >= h {
                    continue;
                }
                let phase_row = &phases[iy * row_stride..(iy + 1) * row_stride];
                let (even, odd) = phase_row.split_at(pairs);
                for kx in 0..g.kw {
                    let wv = f32x4_splat(wk[ky * g.kw + kx]);
                    let kxd = kx * dw;
                    let prow = if kxd & 1 == 0 { even } else { odd };
                    let base = x0 + (kxd >> 1);
                    assert!(base + 4 * nv <= pairs);
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        // SAFETY: `base + 4 * nv <= pairs` (asserted), inside `prow`.
                        let xv = unsafe { v128_load(prow.as_ptr().add(base + 4 * v) as *const v128) };
                        *a = f32x4_add(*a, f32x4_mul(wv, xv));
                    }
                }
            }
            store_epi_wasm(orow, x0, nv, &acc, reg, act, bv);
            x0 += 4 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}
