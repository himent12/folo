//! AVX2+FMA depthwise convolution kernels: ring-buffered 3×3 stride-1, general stride-1, even/odd deinterleaved
//! stride-2 forward (with in-register bias + ReLU/SiLU fusion), and stride-1/stride-2 backward helpers.

use super::super::shifted::deinterleave_f32_even_odd;
use super::depthwise_plane;
use crate::backend::UnaryOp;
use crate::cpu::conv::ConvGeom;
use crate::cpu::gemm::BiasAct;

/// Maximum padded scratch (4M floats = 16 MiB) allowed for AVX2 depthwise kernels before falling back to the
/// zero-scratch path.
const DW_AVX2_MAX_PAD: usize = 1 << 22;

#[derive(Clone, Copy)]
pub(super) enum Avx2Plan {
    /// 3-row sliding ring buffer for `3×3` stride-1 dilation-1 (`3 * wpd` floats).
    K3S1 { owp: usize, wpd: usize },
    /// Full padded plane for general stride-1 (`h * wpd` floats).
    S1 { owp: usize, wpd: usize, pad_len: usize },
    /// Even/odd deinterleaved padded planes for stride-2 (`h * 2 * pairs + 2 * pairs` floats).
    S2 { owp: usize, pairs: usize },
    /// Zero-scratch fallback (`depthwise_plane_avx2`).
    Fallback,
}

/// Select the best overflow-checked AVX2 plan for `g`.
pub(super) fn plan_avx2(g: &ConvGeom) -> Avx2Plan {
    let pw = g.p.padding[1];
    let dw = g.p.dilation[1];
    let Some(owp) = g.ow.checked_add(7).map(|v| v / 8 * 8) else {
        return Avx2Plan::Fallback;
    };
    if g.p.stride == [1, 1] {
        if g.kh == 3 && g.kw == 3 && g.p.dilation == [1, 1] {
            let wpd = owp.checked_add(2).and_then(|t| pw.checked_add(g.w).map(|r| t.max(r)));
            if let Some(wpd) = wpd
                && wpd.checked_mul(3).is_some_and(|n| n <= DW_AVX2_MAX_PAD)
            {
                return Avx2Plan::K3S1 { owp, wpd };
            }
            return Avx2Plan::Fallback;
        }
        let wpd = dw
            .checked_mul(g.kw.saturating_sub(1))
            .and_then(|v| v.checked_add(owp))
            .and_then(|t| pw.checked_add(g.w).map(|r| t.max(r)));
        if let Some(wpd) = wpd
            && let Some(pad_len) = g.h.checked_mul(wpd)
            && pad_len <= DW_AVX2_MAX_PAD
        {
            return Avx2Plan::S1 { owp, wpd, pad_len };
        }
        return Avx2Plan::Fallback;
    }
    if g.p.stride[1] == 2 {
        let tap_pairs =
            dw.checked_mul(g.kw.saturating_sub(1)).map(|v| v / 2).and_then(|v| v.checked_add(owp));
        let row_pairs = pw.checked_add(g.w).map(|v| v.div_ceil(2));
        if let (Some(tp), Some(rp)) = (tap_pairs, row_pairs) {
            let pairs = tp.max(rp);
            let total = g.h.checked_add(1).and_then(|r| r.checked_mul(2)).and_then(|r| r.checked_mul(pairs));
            if total.is_some_and(|n| n <= DW_AVX2_MAX_PAD) {
                return Avx2Plan::S2 { owp, pairs };
            }
        }
    }
    Avx2Plan::Fallback
}

/// Dispatch the chosen AVX2 depthwise plan.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_avx2_plan(
    plan: Avx2Plan,
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    // SAFETY: AVX2+FMA enabled by target_feature; `plan` validated all bounds.
    unsafe {
        match plan {
            Avx2Plan::K3S1 { owp, wpd } => depthwise_k3s1_avx2(xp, wk, op, g, owp, wpd, accumulate, epi, pad),
            Avx2Plan::S1 { owp, wpd, pad_len } => {
                depthwise_s1_avx2(xp, wk, op, g, owp, wpd, pad_len, accumulate, epi, pad)
            }
            Avx2Plan::S2 { owp, pairs } => depthwise_s2_avx2(xp, wk, op, g, owp, pairs, accumulate, epi, pad),
            Avx2Plan::Fallback => unreachable!(),
        }
    }
}

/// [`depthwise_plane`] compiled for AVX2+FMA (8-wide rows).
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn depthwise_plane_avx2(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    depthwise_plane(xp, wk, op, g)
}

/// 8-wide AVX2 dot product, bitwise identical to `dot_scalar` (non-FMA `mul` + `add`, 8 lanes, remainder
/// summed first).
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
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

/// 8-wide AVX2 strided dot product `sum_k g[k] * x[2 * k]`, using the same 8-lane non-FMA reduction order as
/// the stride-2 scalar fallback.
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_stride2_avx2(g: &[f32], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = g.len();
    assert!(n == 0 || (n - 1) * 2 < x.len(), "dot_stride2_avx2: x slice too short");
    let vec_n = n.min(x.len() / 2) / 8 * 8;
    let mut acc = [0f32; 8];
    // SAFETY: `k + 8 <= vec_n <= n` and `2 * (k + 8) <= x.len()`.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        for k in (0..vec_n).step_by(8) {
            let gv = _mm256_loadu_ps(g.as_ptr().add(k));
            let lo = _mm256_loadu_ps(x.as_ptr().add(2 * k));
            let hi = _mm256_loadu_ps(x.as_ptr().add(2 * k + 8));
            let ev = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0x88>(lo, hi),
            )));
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(gv, ev));
        }
        _mm256_storeu_ps(acc.as_mut_ptr(), s0);
    }
    let mut s: f32 = (vec_n..n).map(|k| g[k] * x[2 * k]).sum();
    for v in acc {
        s += v;
    }
    s
}

/// 16-wide AVX2 row update `dst[i] += wv * src[i]`, bitwise identical to scalar `*d += wv * s` (non-FMA).
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mul_add_row_avx2(dst: &mut [f32], src: &[f32], wv: f32) {
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

/// 8-wide AVX2 stride-2 scatter update `dst[2 * k] += wv * src[k]`, leaving odd elements of `dst` untouched
/// bit-for-bit (non-FMA).
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn mul_add_stride2_avx2(dst: &mut [f32], src: &[f32], wv: f32) {
    use std::arch::x86_64::*;
    let n = src.len();
    assert!(n == 0 || (n - 1) * 2 < dst.len(), "mul_add_stride2_avx2: dst slice too short");
    let vec_n = n.min(dst.len() / 2) / 8 * 8;
    let w = _mm256_set1_ps(wv);
    let zero = _mm256_setzero_ps();
    // SAFETY: `k + 8 <= vec_n <= n` and `2 * (k + 8) <= dst.len()`.
    unsafe {
        for k in (0..vec_n).step_by(8) {
            let p = _mm256_mul_ps(w, _mm256_loadu_ps(src.as_ptr().add(k)));
            let u_lo = _mm256_unpacklo_ps(p, zero);
            let u_hi = _mm256_unpackhi_ps(p, zero);
            let add_lo = _mm256_permute2f128_ps::<0x20>(u_lo, u_hi);
            let add_hi = _mm256_permute2f128_ps::<0x31>(u_lo, u_hi);
            let dptr = dst.as_mut_ptr().add(2 * k);
            let d0 = _mm256_loadu_ps(dptr);
            let d1 = _mm256_loadu_ps(dptr.add(8));
            // Blend mask 0xAA (1010_1010) preserves odd lanes of `d0`/`d1` verbatim.
            _mm256_storeu_ps(dptr, _mm256_blend_ps::<0xAA>(_mm256_add_ps(d0, add_lo), d0));
            _mm256_storeu_ps(dptr.add(8), _mm256_blend_ps::<0xAA>(_mm256_add_ps(d1, add_hi), d1));
        }
    }
    for (k, &s) in src.iter().enumerate().skip(vec_n) {
        dst[2 * k] += wv * s;
    }
}

/// Apply in-register bias + optional SiLU/ReLU and store `nv` AVX2 vectors (`x0..x0 + 8 * nv`) into `orow`.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn store_epi_avx2(
    orow: &mut [f32],
    x0: usize,
    nv: usize,
    acc: &[std::arch::x86_64::__m256; 4],
    reg: bool,
    act: Option<UnaryOp>,
    bv: std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let ow = orow.len();
    let zero = _mm256_setzero_ps();
    for (v, &a) in acc.iter().enumerate().take(nv) {
        let c = x0 + 8 * v;
        let r = if !reg {
            a
        } else {
            let x = _mm256_add_ps(a, bv);
            match act {
                None => x,
                Some(UnaryOp::Silu) => crate::cpu::elementwise::silu8(x),
                // `_mm256_max_ps(x, zero)` returns `zero` (+0.0) on `-0.0` and NaN, matching `x.max(0.0)`.
                Some(UnaryOp::Relu) => _mm256_max_ps(x, zero),
                _ => unreachable!(),
            }
        };
        if c + 8 <= ow {
            // SAFETY: columns `c..c + 8` lie inside `orow`.
            unsafe { _mm256_storeu_ps(orow.as_mut_ptr().add(c), r) };
        } else if c < ow {
            let mut tmp = [0f32; 8];
            // SAFETY: `tmp` holds 8 floats.
            unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), r) };
            orow[c..].copy_from_slice(&tmp[..ow - c]);
        }
    }
}

/// Initialise up to 4 AVX2 accumulators (`0.0`, or loaded from `orow` when `accumulate`).
#[target_feature(enable = "avx2")]
#[inline]
fn init_acc_avx2(orow: &[f32], x0: usize, nv: usize, accumulate: bool) -> [std::arch::x86_64::__m256; 4] {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_setzero_ps(); 4];
    if accumulate {
        let ow = orow.len();
        for (v, a) in acc.iter_mut().enumerate().take(nv) {
            let c = x0 + 8 * v;
            if c + 8 <= ow {
                // SAFETY: `c + 8 <= ow == orow.len()`.
                *a = unsafe { _mm256_loadu_ps(orow.as_ptr().add(c)) };
            } else {
                let mut tmp = [0f32; 8];
                let len = ow.saturating_sub(c).min(8);
                tmp[..len].copy_from_slice(&orow[c..c + len]);
                // SAFETY: `tmp` holds 8 floats.
                *a = unsafe { _mm256_loadu_ps(tmp.as_ptr()) };
            }
        }
    }
    acc
}

/// Specialized 3×3 stride-1 dilation-1 depthwise conv using a 3-row L1-resident sliding ring buffer (`3 * wpd`
/// floats) with unrolled interior rows and in-register bias + SiLU/ReLU epilogue.
///
/// # Safety
/// The CPU supports AVX2 and FMA, and `owp` / `wpd` were validated by [`plan_avx2`].
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_k3s1_avx2(
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
    use std::arch::x86_64::*;
    let (ph, pw) = (g.p.padding[0], g.p.padding[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    let ring_len = 3 * wpd;
    pad.clear();
    pad.resize(ring_len, 0.0);
    let mut loaded = [usize::MAX; 3];

    let (reg, bias) = match epi {
        Some(e) if matches!(e.act, None | Some(UnaryOp::Silu | UnaryOp::Relu)) => {
            (true, e.bias.map_or(0.0, |b| b[0]))
        }
        _ => (false, 0.0),
    };
    let act = epi.and_then(|e| e.act);
    let bv = _mm256_set1_ps(if act.is_none() && bias == 0.0 { -0.0 } else { bias });

    let wv = [
        _mm256_set1_ps(wk[0]),
        _mm256_set1_ps(wk[1]),
        _mm256_set1_ps(wk[2]),
        _mm256_set1_ps(wk[3]),
        _mm256_set1_ps(wk[4]),
        _mm256_set1_ps(wk[5]),
        _mm256_set1_ps(wk[6]),
        _mm256_set1_ps(wk[7]),
        _mm256_set1_ps(wk[8]),
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
                let nv = ((owp - x0) / 8).min(4);
                let mut acc = init_acc_avx2(orow, x0, nv, accumulate);
                // SAFETY: `x0 + 8 * nv + 2 <= owp + 2 <= wpd` for all three padded ring rows.
                unsafe {
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let c = x0 + 8 * v;
                        let r0 = p0.as_ptr().add(c);
                        *a = _mm256_fmadd_ps(wv[0], _mm256_loadu_ps(r0), *a);
                        *a = _mm256_fmadd_ps(wv[1], _mm256_loadu_ps(r0.add(1)), *a);
                        *a = _mm256_fmadd_ps(wv[2], _mm256_loadu_ps(r0.add(2)), *a);
                    }
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let c = x0 + 8 * v;
                        let r1 = p1.as_ptr().add(c);
                        *a = _mm256_fmadd_ps(wv[3], _mm256_loadu_ps(r1), *a);
                        *a = _mm256_fmadd_ps(wv[4], _mm256_loadu_ps(r1.add(1)), *a);
                        *a = _mm256_fmadd_ps(wv[5], _mm256_loadu_ps(r1.add(2)), *a);
                    }
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        let c = x0 + 8 * v;
                        let r2 = p2.as_ptr().add(c);
                        *a = _mm256_fmadd_ps(wv[6], _mm256_loadu_ps(r2), *a);
                        *a = _mm256_fmadd_ps(wv[7], _mm256_loadu_ps(r2.add(1)), *a);
                        *a = _mm256_fmadd_ps(wv[8], _mm256_loadu_ps(r2.add(2)), *a);
                    }
                }
                store_epi_avx2(orow, x0, nv, &acc, reg, act, bv);
                x0 += 8 * nv;
            }
        } else {
            let mut x0 = 0;
            while x0 < owp {
                let nv = ((owp - x0) / 8).min(4);
                let mut acc = init_acc_avx2(orow, x0, nv, accumulate);
                for (ky, iy) in iys.into_iter().enumerate() {
                    let Some(iy) = iy else { continue };
                    let prow = &pad[(iy % 3) * wpd..(iy % 3 + 1) * wpd];
                    let w0 = wv[ky * 3];
                    let w1 = wv[ky * 3 + 1];
                    let w2 = wv[ky * 3 + 2];
                    // SAFETY: `x0 + 8 * nv + 2 <= owp + 2 <= wpd`.
                    unsafe {
                        for (v, a) in acc.iter_mut().enumerate().take(nv) {
                            let r = prow.as_ptr().add(x0 + 8 * v);
                            *a = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r), *a);
                            *a = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r.add(1)), *a);
                            *a = _mm256_fmadd_ps(w2, _mm256_loadu_ps(r.add(2)), *a);
                        }
                    }
                }
                store_epi_avx2(orow, x0, nv, &acc, reg, act, bv);
                x0 += 8 * nv;
            }
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// General stride-1 AVX2+FMA depthwise conv with TLS padded-plane reuse and in-register bias + SiLU/ReLU.
///
/// # Safety
/// The CPU supports AVX2 and FMA, and `(owp, wpd, pad_len)` were validated by [`plan_avx2`].
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_s1_avx2(
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
    use std::arch::x86_64::*;
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
    let bv = _mm256_set1_ps(if act.is_none() && bias == 0.0 { -0.0 } else { bias });
    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 8).min(4);
            let mut acc = init_acc_avx2(orow, x0, nv, accumulate);
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
            store_epi_avx2(orow, x0, nv, &acc, reg, act, bv);
            x0 += 8 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

/// Stride-2 (`stride[1] == 2`) AVX2+FMA depthwise conv: pads each input row and deinterleaves it into contiguous
/// even and odd phase rows so every kernel tap is a contiguous 8-float load.
///
/// # Safety
/// The CPU supports AVX2 and FMA, and `(owp, pairs)` were validated by [`plan_avx2`].
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_s2_avx2(
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
    use std::arch::x86_64::*;
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
    let bv = _mm256_set1_ps(if act.is_none() && bias == 0.0 { -0.0 } else { bias });

    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 8).min(4);
            let mut acc = init_acc_avx2(orow, x0, nv, accumulate);
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
                    let wv = _mm256_set1_ps(wk[ky * g.kw + kx]);
                    let kxd = kx * dw;
                    let prow = if kxd & 1 == 0 { even } else { odd };
                    let base = x0 + (kxd >> 1);
                    assert!(base + 8 * nv <= pairs);
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        // SAFETY: `base + 8 * nv <= pairs` (asserted), inside `prow`.
                        *a = _mm256_fmadd_ps(
                            wv,
                            unsafe { _mm256_loadu_ps(prow.as_ptr().add(base + 8 * v)) },
                            *a,
                        );
                    }
                }
            }
            store_epi_avx2(orow, x0, nv, &acc, reg, act, bv);
            x0 += 8 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}
