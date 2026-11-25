//! Depthwise convs (groups = channels): a direct kernel per output plane, with AVX2 at stride 1 (the plane
//! copied once with its padding, every output vector its taps' FMAs in registers), and their gradients.

use super::{ChannelEpilogue, ConvGeom, ConvInput, valid_cols};
#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
use crate::backend::UnaryOp;
#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
use crate::cpu::gemm::BiasAct;
use crate::cpu::par_chunks;

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

/// 8-wide AVX2 dot product, bitwise identical to [`dot_scalar`] (non-FMA `mul` + `add`, 8 lanes, remainder
/// summed first).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    assert_eq!(a.len(), b.len());
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    // SAFETY: `as_chunks::<8>()` guarantees 8-float chunks, `acc` is `[f32; 8]`, and AVX2 is enabled.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        for (x, y) in ca.iter().zip(cb) {
            let prod = _mm256_mul_ps(_mm256_loadu_ps(x.as_ptr()), _mm256_loadu_ps(y.as_ptr()));
            s0 = _mm256_add_ps(s0, prod);
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), s0);
    }
    let mut s: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for v in acc {
        s += v;
    }
    s
}

/// 16-wide AVX2 row update `dst[i] += wv * src[i]`, bitwise identical to scalar `*d += wv * s` (non-FMA).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_add_row_avx2(dst: &mut [f32], src: &[f32], wv: f32) {
    use std::arch::x86_64::*;
    assert_eq!(dst.len(), src.len());
    let w = _mm256_set1_ps(wv);
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    let (s16, s_rem) = src.as_chunks::<16>();
    for (dc, sc) in d16.iter_mut().zip(s16) {
        // SAFETY: `dc` and `sc` are `[f32; 16]` (64 bytes) and AVX2 is enabled.
        unsafe {
            let d0 = _mm256_loadu_ps(dc.as_ptr());
            let d1 = _mm256_loadu_ps(dc.as_ptr().add(8));
            let s0 = _mm256_loadu_ps(sc.as_ptr());
            let s1 = _mm256_loadu_ps(sc.as_ptr().add(8));
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(d0, _mm256_mul_ps(w, s0)));
            _mm256_storeu_ps(dc.as_mut_ptr().add(8), _mm256_add_ps(d1, _mm256_mul_ps(w, s1)));
        }
    }
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    for (dc, sc) in d8.iter_mut().zip(s8) {
        // SAFETY: `dc` and `sc` are `[f32; 8]` (32 bytes) and AVX2 is enabled.
        unsafe {
            let d0 = _mm256_loadu_ps(dc.as_ptr());
            let s0 = _mm256_loadu_ps(sc.as_ptr());
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(d0, _mm256_mul_ps(w, s0)));
        }
    }
    for (d, &s) in d_tail.iter_mut().zip(s_tail) {
        *d += wv * s;
    }
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

/// Precompute `(kx * dw - pw, valid_cols)` once per kernel column.
fn precompute_kx_cols(g: &ConvGeom) -> Vec<(isize, std::ops::Range<usize>)> {
    let (sw, pw, dw) = (g.p.stride[1] as isize, g.p.padding[1] as isize, g.p.dilation[1] as isize);
    (0..g.kw)
        .map(|kx| {
            let d = kx as isize * dw - pw;
            (d, valid_cols(g.ow, sw, d, g.w as isize))
        })
        .collect()
}

/// One output plane of a depthwise conv: `op += conv(xp, wk)` (`op` is `oh x ow`, `xp` one `h x w` input plane).
#[inline(always)]
fn depthwise_plane(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    for oy in 0..g.oh {
        let orow = &mut op[oy * g.ow..(oy + 1) * g.ow];
        for ky in 0..g.kh {
            let iy = oy as isize * sh - ph + ky as isize * dh;
            if iy < 0 || iy >= g.h as isize {
                continue;
            }
            let xrow = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
            for kx in 0..g.kw {
                let wv = wk[ky * g.kw + kx];
                let d = kx as isize * dw - pw;
                let r = valid_cols(g.ow, sw, d, g.w as isize);
                if r.is_empty() {
                    continue;
                }
                if sw == 1 {
                    let s0 = (r.start as isize + d) as usize;
                    let len = r.len();
                    for (o, &xv) in orow[r].iter_mut().zip(&xrow[s0..s0 + len]) {
                        *o += wv * xv;
                    }
                } else {
                    for ox in r {
                        orow[ox] += wv * xrow[(ox as isize * sw + d) as usize];
                    }
                }
            }
        }
    }
}

/// [`depthwise_plane`] compiled for AVX2 (8-wide rows).
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_plane_avx2(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    depthwise_plane(xp, wk, op, g)
}

/// Output elements per [`depthwise_forward`] task (whole planes, at least one): small planes are grouped so a
/// task is not dominated by scheduling.
const DW_TASK_OUTPUTS: usize = 1 << 14;

pub(super) fn depthwise_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(target_arch = "x86_64")]
    let fast = avx2 && g.p.stride == [1, 1];
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    let wasm_pad = if !accumulate && g.p.stride == [1, 1] { wasm_pad_dims(g) } else { None };
    let total = g.n * g.o;
    let per_task = DW_TASK_OUTPUTS.div_ceil(ohw).min(total.div_ceil(4 * rayon::current_num_threads())).max(1);
    par_chunks(out, per_task * ohw, &|t, ops| {
        #[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]
        let mut pad = Vec::new();
        for (p, op) in ops.chunks_mut(ohw).enumerate() {
            let plane = t * per_task + p;
            let (img, oc) = (plane / g.o, plane % g.o);
            let xp = planes[img][oc / mult];
            let wk = &wt[oc * khw..(oc + 1) * khw];
            #[cfg(target_arch = "x86_64")]
            if fast {
                // SAFETY: AVX2 and FMA were detected above.
                unsafe {
                    depthwise_s1_avx2(xp, wk, op, g, accumulate, epi.map(|e| e.rows_from(oc)), &mut pad)
                };
                continue;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            if let Some((owp, wpd, pad_len)) = wasm_pad
                && wk.iter().all(|v| v.is_finite())
            {
                depthwise_s1_wasm(xp, wk, op, g, (owp, wpd, pad_len), epi.map(|e| e.rows_from(oc)), &mut pad);
                continue;
            }
            if !accumulate {
                op.fill(0.0);
            }
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: AVX2 and FMA were detected above.
                unsafe { depthwise_plane_avx2(xp, wk, op, g) };
            } else {
                depthwise_plane(xp, wk, op, g);
            }
            #[cfg(not(target_arch = "x86_64"))]
            depthwise_plane(xp, wk, op, g);
            if let Some(e) = epi {
                e.apply(oc, op);
            }
        }
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
///
/// Caller ensures `!accumulate` and every weight in `wk` is finite, so starting accumulators at `+0.0`
/// guarantees `acc` never becomes `-0.0` and padded `wv * 0.0` taps leave every accumulator bitwise
/// identical to [`depthwise_plane`].
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
                let iy = (oy + ky * dh) as isize - ph as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                let prow = &pad[iy as usize * wpd..(iy as usize + 1) * wpd];
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
            x0 += 4 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// Scalar-only depthwise forward for direct SIMD vs scalar parity tests.
#[cfg(test)]
pub(super) fn depthwise_forward_scalar(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    for (plane, op) in out.chunks_mut(ohw).enumerate() {
        let (img, oc) = (plane / g.o, plane % g.o);
        let xp = planes[img][oc / mult];
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

/// One output plane of a stride-1 depthwise conv with AVX2: `op = conv(xp, wk)` (plus the previous `op` if
/// `accumulate`), then `epi` (its row 0 is this channel). The input plane is first copied into `pad` with its
/// zero padding (and slack up to a whole number of vectors), so every output vector is the `kh·kw` taps'
/// fused multiply-adds, in `(ky, kx)` order, of plain unaligned loads, held in registers four vectors at a
/// time; the identity and SiLU epilogues are applied there too (the lanes of `silu_bias_inplace`), others as a
/// pass over the finished plane.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_s1_avx2(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    use std::arch::x86_64::*;
    let (ph, pw) = (g.p.padding[0], g.p.padding[1]);
    let (dh, dw) = (g.p.dilation[0], g.p.dilation[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    let owp = ow.div_ceil(8) * 8;
    // Padded row: `prow[X] = x[X - pw]`, for every column a tap of a computed output column reads.
    let wpd = (owp + dw * (g.kw - 1)).max(pw + w);
    pad.clear();
    pad.resize(h * wpd, 0.0);
    // A zero-width input (all padding) has nothing to copy.
    if w > 0 {
        for (prow, xrow) in pad.chunks_exact_mut(wpd).zip(xp.chunks_exact(w)) {
            prow[pw..pw + w].copy_from_slice(xrow);
        }
    }
    // In registers: bias + SiLU / bias; otherwise a pass after the plane.
    let (reg, bias) = match epi {
        Some(e) if matches!(e.act, None | Some(UnaryOp::Silu)) => (true, e.bias.map_or(0.0, |b| b[0])),
        _ => (false, 0.0),
    };
    let silu = epi.is_some_and(|e| e.act == Some(UnaryOp::Silu));
    // Without an activation `bias_act_inplace` skips a zero bias; `x + -0.0` is `x` bit for bit.
    let bv = _mm256_set1_ps(if !silu && bias == 0.0 { -0.0 } else { bias });
    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 8).min(4);
            let mut acc = [_mm256_setzero_ps(); 4];
            if accumulate {
                for (v, a) in acc.iter_mut().enumerate().take(nv) {
                    let c = x0 + 8 * v;
                    let mut tmp = [0f32; 8];
                    let len = ow.saturating_sub(c).min(8);
                    tmp[..len].copy_from_slice(&orow[c..c + len]);
                    // SAFETY: `tmp` holds 8 floats.
                    *a = unsafe { _mm256_loadu_ps(tmp.as_ptr()) };
                }
            }
            for ky in 0..g.kh {
                let iy = (oy + ky * dh) as isize - ph as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                let prow = &pad[iy as usize * wpd..(iy as usize + 1) * wpd];
                for kx in 0..g.kw {
                    let wv = _mm256_set1_ps(wk[ky * g.kw + kx]);
                    let base = x0 + kx * dw;
                    assert!(base + 8 * nv <= wpd);
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        // SAFETY: `base + 8 * nv <= wpd` (asserted), inside this padded row.
                        *a = _mm256_fmadd_ps(
                            wv,
                            unsafe { _mm256_loadu_ps(prow.as_ptr().add(base + 8 * v)) },
                            *a,
                        );
                    }
                }
            }
            for (v, a) in acc.iter().enumerate().take(nv) {
                let c = x0 + 8 * v;
                let r = match (reg, silu) {
                    (false, _) => *a,
                    (true, false) => _mm256_add_ps(*a, bv),
                    (true, true) => crate::cpu::elementwise::silu8(_mm256_add_ps(*a, bv)),
                };
                if c + 8 <= ow {
                    // SAFETY: columns c..c + 8 are inside `orow`.
                    unsafe { _mm256_storeu_ps(orow.as_mut_ptr().add(c), r) };
                } else if c < ow {
                    let mut tmp = [0f32; 8];
                    // SAFETY: `tmp` holds 8 floats.
                    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), r) };
                    orow[c..].copy_from_slice(&tmp[..ow - c]);
                }
            }
            x0 += 8 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

pub(super) fn depthwise_backward_input(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
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
                                    unsafe { mul_add_row_avx2(dst, src, wv) };
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

pub(super) fn depthwise_backward_weight(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom) {
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
                                    acc += unsafe { dot_avx2(gs, xs) };
                                    continue;
                                }
                                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                                if gs.len() >= 8 {
                                    acc += dot_wasm(gs, xs);
                                    continue;
                                }
                                acc += dot_scalar(gs, xs);
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
