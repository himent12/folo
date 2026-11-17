//! Depthwise convs (groups = channels): a direct kernel per output plane, with AVX2 at stride 1 (the plane
//! copied once with its padding, every output vector its taps' FMAs in registers), and their gradients.

use super::{ChannelEpilogue, ConvGeom, ConvInput, valid_cols};
#[cfg(target_arch = "x86_64")]
use crate::backend::UnaryOp;
#[cfg(target_arch = "x86_64")]
use crate::cpu::gemm::BiasAct;
use crate::cpu::par_chunks;

/// Dot product with independent accumulators so it vectorises.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
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
    let total = g.n * g.o;
    let per_task = DW_TASK_OUTPUTS.div_ceil(ohw).min(total.div_ceil(4 * rayon::current_num_threads())).max(1);
    par_chunks(out, per_task * ohw, &|t, ops| {
        #[cfg(target_arch = "x86_64")]
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
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    par_chunks(gx, hw, &|plane, gxp| {
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
                            for (xv, &gv) in xrow[s0..s0 + len].iter_mut().zip(&grow[r]) {
                                *xv += wv * gv;
                            }
                        } else {
                            for ox in r {
                                xrow[(ox as isize * sw + d) as usize] += wv * grow[ox];
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
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    par_chunks(gw, khw, &|oc, gk| {
        let ic = oc / mult;
        for img in 0..g.n {
            let xp = &x[(img * g.c + ic) * hw..][..hw];
            let gp = &gy[(img * g.o + oc) * ohw..][..ohw];
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let d = kx as isize * dw - pw;
                    let cols = valid_cols(g.ow, sw, d, g.w as isize);
                    let mut acc = 0f32;
                    for oy in 0..g.oh {
                        let iy = oy as isize * sh - ph + ky as isize * dh;
                        if iy < 0 || iy >= g.h as isize {
                            continue;
                        }
                        let xrow = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
                        let grow = &gp[oy * g.ow..(oy + 1) * g.ow];
                        if cols.is_empty() {
                            continue;
                        }
                        if sw == 1 {
                            let s0 = (cols.start as isize + d) as usize;
                            acc += dot(&grow[cols.clone()], &xrow[s0..s0 + cols.len()]);
                        } else {
                            for ox in cols.clone() {
                                acc += grow[ox] * xrow[(ox as isize * sw + d) as usize];
                            }
                        }
                    }
                    gk[ky * g.kw + kx] += acc;
                }
            }
        }
    });
}
