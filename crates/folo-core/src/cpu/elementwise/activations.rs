//! Vectorised AVX2 and Wasm SIMD128 `gelu`, `gelu_backward`, `swiglu`, and `swiglu_backward` kernels with
//! bitwise-identical non-FMA scalar twins.
//!
//! Every kernel here composes directly over the division-free non-FMA primitives in [`super::simd`],
//! [`super::simd_log_tanh`], and [`super::wasm_simd`], guaranteeing bitwise identity between AVX2, Wasm SIMD128,
//! and scalar execution on all non-NaN inputs (including `±0.0`, subnormals, huge finite values, and `±∞`),
//! while preserving NaN propagation.

use super::simd::{self, maxps, minps, silu_deriv_scalar, silu_scalar};
use super::simd_log_tanh::{tanh_deriv_scalar, tanh_scalar};

/// `sqrt(2 / pi)` for tanh-approximate GELU.
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// Cubic coefficient `0.044715` for tanh-approximate GELU.
const GELU_COEFF: f32 = 0.044_715;
/// Derivative cubic coefficient `3 * 0.044715` for tanh-approximate GELU backward.
const GELU_COEFF_3: f32 = 0.134_145;
/// Beyond `|x| = 12`, `u = sqrt(2/pi) * (x + 0.044715 * x^3)` has `|u| > 71`, where `tanh(u) = ±1.0` and
/// `sech^2(u) = 0.0` exactly in `f32`. Clamping `x` before forming `x^3` prevents overflow to `±∞` / `0 * ∞ = NaN`.
const GELU_CLAMP: f32 = 12.0;

// ---------------------------------------------------------------------------
// Scalar twins (bitwise identical to AVX2 and Wasm SIMD128 on all non-NaN inputs)
// ---------------------------------------------------------------------------

/// Tanh-approximate GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, bitwise identical to
/// [`gelu_slice`] on all non-NaN inputs and overflow-safe up to `±f32::MAX` / `±∞`.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let xc = minps(maxps(x, -GELU_CLAMP), GELU_CLAMP);
    let x2 = xc * xc;
    let u = GELU_SQRT_2_OVER_PI * (xc + GELU_COEFF * (x2 * xc));
    let t = tanh_scalar(u);
    let y = 0.5 * x * (1.0 + t);
    let y = if x <= -GELU_CLAMP { 0.0 } else { y };
    if x.is_nan() { x } else { y }
}

/// Derivative factor `gelu'(x)` for tanh-approximate GELU, overflow-safe up to `±f32::MAX` / `±∞`.
#[inline(always)]
fn gelu_deriv_scalar(x: f32) -> f32 {
    let xc = minps(maxps(x, -GELU_CLAMP), GELU_CLAMP);
    let x2 = xc * xc;
    let u = GELU_SQRT_2_OVER_PI * (xc + GELU_COEFF * (x2 * xc));
    let t = tanh_scalar(u);
    let sech2 = tanh_deriv_scalar(u);
    let du = GELU_SQRT_2_OVER_PI * (1.0 + GELU_COEFF_3 * x2);
    let d = 0.5 * (1.0 + t) + (0.5 * xc) * (sech2 * du);
    if x.is_nan() { x } else { d }
}

/// `g * gelu'(x)`, bitwise identical to [`gelu_backward_slice`] on all non-NaN inputs.
#[inline(always)]
pub fn gelu_backward(x: f32, g: f32) -> f32 {
    g * gelu_deriv_scalar(x)
}

/// `silu(gate) * up`, bitwise identical to [`swiglu_slice`] on all non-NaN inputs.
#[inline(always)]
pub fn swiglu(gate: f32, up: f32) -> f32 {
    silu_scalar(gate) * up
}

/// Backward of [`swiglu`]: returns `(dgate, dup) = ((g * silu'(gate)) * up, g * silu(gate))`, bitwise identical
/// to [`swiglu_backward_slice`] on all non-NaN inputs. Multiplying `g * silu'(gate)` before `* up` prevents
/// spurious `inf * 0 = NaN` when `g * up` would overflow on saturated negative gates.
#[inline(always)]
pub fn swiglu_backward(gate: f32, up: f32, g: f32) -> (f32, f32) {
    let dgate = (g * silu_deriv_scalar(gate)) * up;
    let dup = g * silu_scalar(gate);
    (dgate, dup)
}

// ---------------------------------------------------------------------------
// AVX2 8-lane kernels (non-FMA, bitwise identical to scalar twins)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{GELU_CLAMP, GELU_COEFF, GELU_COEFF_3, GELU_SQRT_2_OVER_PI};
    use crate::cpu::elementwise::simd::{silu_bwd8, silu_op8};
    use crate::cpu::elementwise::simd_log_tanh::{tanh_bwd8, tanh8};

    #[target_feature(enable = "avx2")]
    #[inline]
    fn gelu8(x: __m256) -> __m256 {
        let neg_c = _mm256_set1_ps(-GELU_CLAMP);
        let pos_c = _mm256_set1_ps(GELU_CLAMP);
        let xc = _mm256_min_ps(_mm256_max_ps(x, neg_c), pos_c);
        let x2 = _mm256_mul_ps(xc, xc);
        let u = _mm256_mul_ps(
            _mm256_set1_ps(GELU_SQRT_2_OVER_PI),
            _mm256_add_ps(xc, _mm256_mul_ps(_mm256_set1_ps(GELU_COEFF), _mm256_mul_ps(x2, xc))),
        );
        let t = tanh8(u);
        let y = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5), x), _mm256_add_ps(_mm256_set1_ps(1.0), t));
        let y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps::<_CMP_LE_OQ>(x, neg_c));
        _mm256_blendv_ps(y, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn gelu_bwd8(x: __m256, g: __m256) -> __m256 {
        let neg_c = _mm256_set1_ps(-GELU_CLAMP);
        let pos_c = _mm256_set1_ps(GELU_CLAMP);
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let xc = _mm256_min_ps(_mm256_max_ps(x, neg_c), pos_c);
        let x2 = _mm256_mul_ps(xc, xc);
        let u = _mm256_mul_ps(
            _mm256_set1_ps(GELU_SQRT_2_OVER_PI),
            _mm256_add_ps(xc, _mm256_mul_ps(_mm256_set1_ps(GELU_COEFF), _mm256_mul_ps(x2, xc))),
        );
        let t = tanh8(u);
        let sech2 = tanh_bwd8(u, one);
        let du = _mm256_mul_ps(
            _mm256_set1_ps(GELU_SQRT_2_OVER_PI),
            _mm256_add_ps(one, _mm256_mul_ps(_mm256_set1_ps(GELU_COEFF_3), x2)),
        );
        let d = _mm256_add_ps(
            _mm256_mul_ps(half, _mm256_add_ps(one, t)),
            _mm256_mul_ps(_mm256_mul_ps(half, xc), _mm256_mul_ps(sech2, du)),
        );
        let d = _mm256_blendv_ps(d, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x));
        _mm256_mul_ps(g, d)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn gelu_slice_avx2(x: &[f32], out: &mut [f32]) {
        let (xp, xr) = x.as_chunks::<16>();
        let (op, or) = out.as_chunks_mut::<16>();
        for (xc, oc) in xp.iter().zip(op.iter_mut()) {
            // SAFETY: `xc` and `oc` are 16 floats; AVX2 enabled.
            unsafe {
                let sp = xc.as_ptr();
                let dp = oc.as_mut_ptr();
                let v0 = gelu8(_mm256_loadu_ps(sp));
                let v1 = gelu8(_mm256_loadu_ps(sp.add(8)));
                _mm256_storeu_ps(dp, v0);
                _mm256_storeu_ps(dp.add(8), v1);
            }
        }
        let (x8, xt) = xr.as_chunks::<8>();
        let (o8, ot) = or.as_chunks_mut::<8>();
        for (xc, oc) in x8.iter().zip(o8.iter_mut()) {
            // SAFETY: `xc` and `oc` are 8 floats; AVX2 enabled.
            unsafe { _mm256_storeu_ps(oc.as_mut_ptr(), gelu8(_mm256_loadu_ps(xc.as_ptr()))) };
        }
        for (o, &v) in ot.iter_mut().zip(xt) {
            *o = super::gelu(v);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn gelu_bwd_slice_avx2(x: &[f32], g: &[f32], out: &mut [f32]) {
        let (xp, xr) = x.as_chunks::<16>();
        let (gp, gr) = g.as_chunks::<16>();
        let (op, or) = out.as_chunks_mut::<16>();
        for ((xc, gc), oc) in xp.iter().zip(gp.iter()).zip(op.iter_mut()) {
            // SAFETY: `xc`, `gc`, `oc` are 16 floats; AVX2 enabled.
            unsafe {
                let (xp, gp, dp) = (xc.as_ptr(), gc.as_ptr(), oc.as_mut_ptr());
                let v0 = gelu_bwd8(_mm256_loadu_ps(xp), _mm256_loadu_ps(gp));
                let v1 = gelu_bwd8(_mm256_loadu_ps(xp.add(8)), _mm256_loadu_ps(gp.add(8)));
                _mm256_storeu_ps(dp, v0);
                _mm256_storeu_ps(dp.add(8), v1);
            }
        }
        let (x8, xt) = xr.as_chunks::<8>();
        let (g8, gt) = gr.as_chunks::<8>();
        let (o8, ot) = or.as_chunks_mut::<8>();
        for ((xc, gc), oc) in x8.iter().zip(g8.iter()).zip(o8.iter_mut()) {
            // SAFETY: `xc`, `gc`, `oc` are 8 floats; AVX2 enabled.
            unsafe {
                _mm256_storeu_ps(
                    oc.as_mut_ptr(),
                    gelu_bwd8(_mm256_loadu_ps(xc.as_ptr()), _mm256_loadu_ps(gc.as_ptr())),
                );
            }
        }
        for ((o, &xv), &gv) in ot.iter_mut().zip(xt).zip(gt) {
            *o = super::gelu_backward(xv, gv);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn swiglu_slice_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let (gp, gr) = gate.as_chunks::<16>();
        let (up16, ur) = up.as_chunks::<16>();
        let (op, or) = out.as_chunks_mut::<16>();
        for ((gc, uc), oc) in gp.iter().zip(up16.iter()).zip(op.iter_mut()) {
            // SAFETY: `gc`, `uc`, `oc` are 16 floats; AVX2 enabled.
            unsafe {
                let (gp, up_p, dp) = (gc.as_ptr(), uc.as_ptr(), oc.as_mut_ptr());
                let v0 = _mm256_mul_ps(silu_op8(_mm256_loadu_ps(gp)), _mm256_loadu_ps(up_p));
                let v1 = _mm256_mul_ps(silu_op8(_mm256_loadu_ps(gp.add(8))), _mm256_loadu_ps(up_p.add(8)));
                _mm256_storeu_ps(dp, v0);
                _mm256_storeu_ps(dp.add(8), v1);
            }
        }
        let (g8, gt) = gr.as_chunks::<8>();
        let (u8c, ut) = ur.as_chunks::<8>();
        let (o8, ot) = or.as_chunks_mut::<8>();
        for ((gc, uc), oc) in g8.iter().zip(u8c.iter()).zip(o8.iter_mut()) {
            // SAFETY: `gc`, `uc`, `oc` are 8 floats; AVX2 enabled.
            unsafe {
                _mm256_storeu_ps(
                    oc.as_mut_ptr(),
                    _mm256_mul_ps(silu_op8(_mm256_loadu_ps(gc.as_ptr())), _mm256_loadu_ps(uc.as_ptr())),
                );
            }
        }
        for ((o, &gv), &uv) in ot.iter_mut().zip(gt).zip(ut) {
            *o = super::swiglu(gv, uv);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn swiglu_bwd_slice_avx2(
        gate: &[f32],
        up: &[f32],
        grad: &[f32],
        dgate: &mut [f32],
        dup: &mut [f32],
    ) {
        let (gp, gr) = gate.as_chunks::<16>();
        let (up16, ur) = up.as_chunks::<16>();
        let (grp, grr) = grad.as_chunks::<16>();
        let (dgp, dgr) = dgate.as_chunks_mut::<16>();
        let (dup16, dur) = dup.as_chunks_mut::<16>();
        for ((((gc, uc), grc), dgc), duc) in
            gp.iter().zip(up16.iter()).zip(grp.iter()).zip(dgp.iter_mut()).zip(dup16.iter_mut())
        {
            // SAFETY: all five slices are 16 floats; AVX2 enabled.
            unsafe {
                let (gp, up_p, grp_p, dgp_p, dup_p) =
                    (gc.as_ptr(), uc.as_ptr(), grc.as_ptr(), dgc.as_mut_ptr(), duc.as_mut_ptr());
                for off in [0usize, 8] {
                    let gv = _mm256_loadu_ps(gp.add(off));
                    let uv = _mm256_loadu_ps(up_p.add(off));
                    let grv = _mm256_loadu_ps(grp_p.add(off));
                    let dg = _mm256_mul_ps(silu_bwd8(gv, grv), uv);
                    let du = _mm256_mul_ps(grv, silu_op8(gv));
                    _mm256_storeu_ps(dgp_p.add(off), dg);
                    _mm256_storeu_ps(dup_p.add(off), du);
                }
            }
        }
        let (g8, gt) = gr.as_chunks::<8>();
        let (u8c, ut) = ur.as_chunks::<8>();
        let (gr8, grt) = grr.as_chunks::<8>();
        let (dg8, dgt) = dgr.as_chunks_mut::<8>();
        let (du8, dut) = dur.as_chunks_mut::<8>();
        for ((((gc, uc), grc), dgc), duc) in
            g8.iter().zip(u8c.iter()).zip(gr8.iter()).zip(dg8.iter_mut()).zip(du8.iter_mut())
        {
            // SAFETY: all five slices are 8 floats; AVX2 enabled.
            unsafe {
                let gv = _mm256_loadu_ps(gc.as_ptr());
                let uv = _mm256_loadu_ps(uc.as_ptr());
                let grv = _mm256_loadu_ps(grc.as_ptr());
                _mm256_storeu_ps(dgc.as_mut_ptr(), _mm256_mul_ps(silu_bwd8(gv, grv), uv));
                _mm256_storeu_ps(duc.as_mut_ptr(), _mm256_mul_ps(grv, silu_op8(gv)));
            }
        }
        for ((((dg, du), &gv), &uv), &grv) in dgt.iter_mut().zip(dut.iter_mut()).zip(gt).zip(ut).zip(grt) {
            (*dg, *du) = super::swiglu_backward(gv, uv, grv);
        }
    }
}

// ---------------------------------------------------------------------------
// Wasm SIMD128 4-lane kernels (non-FMA, bitwise identical to scalar twins)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    use super::{GELU_CLAMP, GELU_COEFF, GELU_COEFF_3, GELU_SQRT_2_OVER_PI};
    use crate::cpu::elementwise::wasm_simd::{silu_bwd4, silu4, tanh_bwd4, tanh4};

    #[inline(always)]
    fn gelu4(x: v128) -> v128 {
        let neg_c = f32x4_splat(-GELU_CLAMP);
        let pos_c = f32x4_splat(GELU_CLAMP);
        // `f32x4_pmax(a, b)` returns `a < b ? b : a`, matching `maxps(x, -12.0)` / `minps(_, 12.0)` on NaN.
        let xc = f32x4_pmin(pos_c, f32x4_pmax(neg_c, x));
        let x2 = f32x4_mul(xc, xc);
        let u = f32x4_mul(
            f32x4_splat(GELU_SQRT_2_OVER_PI),
            f32x4_add(xc, f32x4_mul(f32x4_splat(GELU_COEFF), f32x4_mul(x2, xc))),
        );
        let t = tanh4(u);
        let y = f32x4_mul(f32x4_mul(f32x4_splat(0.5), x), f32x4_add(f32x4_splat(1.0), t));
        let y = v128_bitselect(f32x4_splat(0.0), y, f32x4_le(x, neg_c));
        v128_bitselect(x, y, f32x4_ne(x, x))
    }

    #[inline(always)]
    fn gelu_bwd4(x: v128, g: v128) -> v128 {
        let neg_c = f32x4_splat(-GELU_CLAMP);
        let pos_c = f32x4_splat(GELU_CLAMP);
        let one = f32x4_splat(1.0);
        let half = f32x4_splat(0.5);
        let xc = f32x4_pmin(pos_c, f32x4_pmax(neg_c, x));
        let x2 = f32x4_mul(xc, xc);
        let u = f32x4_mul(
            f32x4_splat(GELU_SQRT_2_OVER_PI),
            f32x4_add(xc, f32x4_mul(f32x4_splat(GELU_COEFF), f32x4_mul(x2, xc))),
        );
        let t = tanh4(u);
        let sech2 = tanh_bwd4(u, one);
        let du = f32x4_mul(
            f32x4_splat(GELU_SQRT_2_OVER_PI),
            f32x4_add(one, f32x4_mul(f32x4_splat(GELU_COEFF_3), x2)),
        );
        let d = f32x4_add(
            f32x4_mul(half, f32x4_add(one, t)),
            f32x4_mul(f32x4_mul(half, xc), f32x4_mul(sech2, du)),
        );
        let d = v128_bitselect(x, d, f32x4_ne(x, x));
        f32x4_mul(g, d)
    }

    pub(super) fn gelu_slice_wasm(x: &[f32], out: &mut [f32]) {
        let (xp, xr) = x.as_chunks::<8>();
        let (op, or) = out.as_chunks_mut::<8>();
        for (xc, oc) in xp.iter().zip(op.iter_mut()) {
            // SAFETY: `xc` and `oc` are 8 floats.
            unsafe {
                let (sp, dp) = (xc.as_ptr(), oc.as_mut_ptr());
                v128_store(dp as *mut v128, gelu4(v128_load(sp as *const v128)));
                v128_store(dp.add(4) as *mut v128, gelu4(v128_load(sp.add(4) as *const v128)));
            }
        }
        let (x4, xt) = xr.as_chunks::<4>();
        let (o4, ot) = or.as_chunks_mut::<4>();
        for (xc, oc) in x4.iter().zip(o4.iter_mut()) {
            // SAFETY: `xc` and `oc` are 4 floats.
            unsafe { v128_store(oc.as_mut_ptr() as *mut v128, gelu4(v128_load(xc.as_ptr() as *const v128))) };
        }
        for (o, &v) in ot.iter_mut().zip(xt) {
            *o = super::gelu(v);
        }
    }

    pub(super) fn gelu_bwd_slice_wasm(x: &[f32], g: &[f32], out: &mut [f32]) {
        let (xp, xr) = x.as_chunks::<8>();
        let (gp, gr) = g.as_chunks::<8>();
        let (op, or) = out.as_chunks_mut::<8>();
        for ((xc, gc), oc) in xp.iter().zip(gp.iter()).zip(op.iter_mut()) {
            // SAFETY: `xc`, `gc`, `oc` are 8 floats.
            unsafe {
                let (xp, gp, dp) = (xc.as_ptr(), gc.as_ptr(), oc.as_mut_ptr());
                v128_store(
                    dp as *mut v128,
                    gelu_bwd4(v128_load(xp as *const v128), v128_load(gp as *const v128)),
                );
                v128_store(
                    dp.add(4) as *mut v128,
                    gelu_bwd4(v128_load(xp.add(4) as *const v128), v128_load(gp.add(4) as *const v128)),
                );
            }
        }
        let (x4, xt) = xr.as_chunks::<4>();
        let (g4, gt) = gr.as_chunks::<4>();
        let (o4, ot) = or.as_chunks_mut::<4>();
        for ((xc, gc), oc) in x4.iter().zip(g4.iter()).zip(o4.iter_mut()) {
            // SAFETY: `xc`, `gc`, `oc` are 4 floats.
            unsafe {
                v128_store(
                    oc.as_mut_ptr() as *mut v128,
                    gelu_bwd4(v128_load(xc.as_ptr() as *const v128), v128_load(gc.as_ptr() as *const v128)),
                );
            }
        }
        for ((o, &xv), &gv) in ot.iter_mut().zip(xt).zip(gt) {
            *o = super::gelu_backward(xv, gv);
        }
    }

    pub(super) fn swiglu_slice_wasm(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let (gp, gr) = gate.as_chunks::<8>();
        let (up8, ur) = up.as_chunks::<8>();
        let (op, or) = out.as_chunks_mut::<8>();
        for ((gc, uc), oc) in gp.iter().zip(up8.iter()).zip(op.iter_mut()) {
            // SAFETY: `gc`, `uc`, `oc` are 8 floats.
            unsafe {
                let (gp, up_p, dp) = (gc.as_ptr(), uc.as_ptr(), oc.as_mut_ptr());
                v128_store(
                    dp as *mut v128,
                    f32x4_mul(silu4(v128_load(gp as *const v128)), v128_load(up_p as *const v128)),
                );
                v128_store(
                    dp.add(4) as *mut v128,
                    f32x4_mul(
                        silu4(v128_load(gp.add(4) as *const v128)),
                        v128_load(up_p.add(4) as *const v128),
                    ),
                );
            }
        }
        let (g4, gt) = gr.as_chunks::<4>();
        let (u4, ut) = ur.as_chunks::<4>();
        let (o4, ot) = or.as_chunks_mut::<4>();
        for ((gc, uc), oc) in g4.iter().zip(u4.iter()).zip(o4.iter_mut()) {
            // SAFETY: `gc`, `uc`, `oc` are 4 floats.
            unsafe {
                v128_store(
                    oc.as_mut_ptr() as *mut v128,
                    f32x4_mul(
                        silu4(v128_load(gc.as_ptr() as *const v128)),
                        v128_load(uc.as_ptr() as *const v128),
                    ),
                );
            }
        }
        for ((o, &gv), &uv) in ot.iter_mut().zip(gt).zip(ut) {
            *o = super::swiglu(gv, uv);
        }
    }

    pub(super) fn swiglu_bwd_slice_wasm(
        gate: &[f32],
        up: &[f32],
        grad: &[f32],
        dgate: &mut [f32],
        dup: &mut [f32],
    ) {
        let (gp, gr) = gate.as_chunks::<8>();
        let (up8, ur) = up.as_chunks::<8>();
        let (grp, grr) = grad.as_chunks::<8>();
        let (dgp, dgr) = dgate.as_chunks_mut::<8>();
        let (dup8, dur) = dup.as_chunks_mut::<8>();
        for ((((gc, uc), grc), dgc), duc) in
            gp.iter().zip(up8.iter()).zip(grp.iter()).zip(dgp.iter_mut()).zip(dup8.iter_mut())
        {
            // SAFETY: all five slices are 8 floats.
            unsafe {
                let (gp, up_p, grp_p, dgp_p, dup_p) =
                    (gc.as_ptr(), uc.as_ptr(), grc.as_ptr(), dgc.as_mut_ptr(), duc.as_mut_ptr());
                for off in [0usize, 4] {
                    let gv = v128_load(gp.add(off) as *const v128);
                    let uv = v128_load(up_p.add(off) as *const v128);
                    let grv = v128_load(grp_p.add(off) as *const v128);
                    v128_store(dgp_p.add(off) as *mut v128, f32x4_mul(silu_bwd4(gv, grv), uv));
                    v128_store(dup_p.add(off) as *mut v128, f32x4_mul(grv, silu4(gv)));
                }
            }
        }
        let (g4, gt) = gr.as_chunks::<4>();
        let (u4, ut) = ur.as_chunks::<4>();
        let (gr4, grt) = grr.as_chunks::<4>();
        let (dg4, dgt) = dgr.as_chunks_mut::<4>();
        let (du4, dut) = dur.as_chunks_mut::<4>();
        for ((((gc, uc), grc), dgc), duc) in
            g4.iter().zip(u4.iter()).zip(gr4.iter()).zip(dg4.iter_mut()).zip(du4.iter_mut())
        {
            // SAFETY: all five slices are 4 floats.
            unsafe {
                let gv = v128_load(gc.as_ptr() as *const v128);
                let uv = v128_load(uc.as_ptr() as *const v128);
                let grv = v128_load(grc.as_ptr() as *const v128);
                v128_store(dgc.as_mut_ptr() as *mut v128, f32x4_mul(silu_bwd4(gv, grv), uv));
                v128_store(duc.as_mut_ptr() as *mut v128, f32x4_mul(grv, silu4(gv)));
            }
        }
        for ((((dg, du), &gv), &uv), &grv) in dgt.iter_mut().zip(dut.iter_mut()).zip(gt).zip(ut).zip(grt) {
            (*dg, *du) = super::swiglu_backward(gv, uv, grv);
        }
    }
}

// ---------------------------------------------------------------------------
// Public slice entry points
// ---------------------------------------------------------------------------

/// Vectorised `out[i] = gelu(x[i])` (AVX2 on x86_64, SIMD128 on wasm32, bitwise-identical scalar fallback).
///
/// # Panics
/// If `x.len() != out.len()`.
pub fn gelu_slice(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len(), "gelu_slice: length mismatch");
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { avx2::gelu_slice_avx2(x, out) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm::gelu_slice_wasm(x, out);
    }
    #[allow(unreachable_code)]
    for (o, &v) in out.iter_mut().zip(x) {
        *o = gelu(v);
    }
}

/// Vectorised `out[i] = g[i] * gelu'(x[i])` (AVX2 on x86_64, SIMD128 on wasm32, bitwise-identical scalar fallback).
///
/// # Panics
/// If `x.len() != g.len()` or `x.len() != out.len()`.
pub fn gelu_backward_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), g.len(), "gelu_backward_slice: x/g length mismatch");
    assert_eq!(x.len(), out.len(), "gelu_backward_slice: out length mismatch");
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { avx2::gelu_bwd_slice_avx2(x, g, out) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm::gelu_bwd_slice_wasm(x, g, out);
    }
    #[allow(unreachable_code)]
    for ((o, &xv), &gv) in out.iter_mut().zip(x).zip(g) {
        *o = gelu_backward(xv, gv);
    }
}

/// Vectorised `out[i] = g[i] * silu'(x[i])` (AVX2 on x86_64, SIMD128 on wasm32, bitwise-identical scalar fallback).
///
/// # Panics
/// If `x.len() != g.len()` or `x.len() != out.len()`.
pub fn silu_backward_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), g.len(), "silu_backward_slice: x/g length mismatch");
    assert_eq!(x.len(), out.len(), "silu_backward_slice: out length mismatch");
    simd::silu_bwd_slice(x, g, out);
}

/// Vectorised `out[i] = silu(gate[i]) * up[i]` (AVX2 on x86_64, SIMD128 on wasm32, bitwise-identical scalar
/// fallback).
///
/// # Panics
/// If `gate.len() != up.len()` or `gate.len() != out.len()`.
pub fn swiglu_slice(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "swiglu_slice: gate/up length mismatch");
    assert_eq!(gate.len(), out.len(), "swiglu_slice: out length mismatch");
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { avx2::swiglu_slice_avx2(gate, up, out) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm::swiglu_slice_wasm(gate, up, out);
    }
    #[allow(unreachable_code)]
    for ((o, &gv), &uv) in out.iter_mut().zip(gate).zip(up) {
        *o = swiglu(gv, uv);
    }
}

/// Vectorised backward of [`swiglu_slice`]: writes `dgate[i] = (grad[i] * silu'(gate[i])) * up[i]` and
/// `dup[i] = grad[i] * silu(gate[i])` in a single pass.
///
/// # Panics
/// If any of the five slices have differing lengths.
pub fn swiglu_backward_slice(gate: &[f32], up: &[f32], grad: &[f32], dgate: &mut [f32], dup: &mut [f32]) {
    let n = gate.len();
    assert_eq!(up.len(), n, "swiglu_backward_slice: up length mismatch");
    assert_eq!(grad.len(), n, "swiglu_backward_slice: grad length mismatch");
    assert_eq!(dgate.len(), n, "swiglu_backward_slice: dgate length mismatch");
    assert_eq!(dup.len(), n, "swiglu_backward_slice: dup length mismatch");
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { avx2::swiglu_bwd_slice_avx2(gate, up, grad, dgate, dup) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return wasm::swiglu_bwd_slice_wasm(gate, up, grad, dgate, dup);
    }
    #[allow(unreachable_code)]
    for ((((dg, du), &gv), &uv), &grv) in dgate.iter_mut().zip(dup.iter_mut()).zip(gate).zip(up).zip(grad) {
        (*dg, *du) = swiglu_backward(gv, uv, grv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gelu_f64_ref(x: f64) -> f64 {
        let s = (2.0f64 / std::f64::consts::PI).sqrt();
        let u = s * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + u.tanh())
    }

    fn gelu_deriv_f64_ref(x: f64) -> f64 {
        let s = (2.0f64 / std::f64::consts::PI).sqrt();
        let u = s * (x + 0.044715 * x * x * x);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let du = s * (1.0 + 3.0 * 0.044715 * x * x);
        0.5 * (1.0 + t) + 0.5 * x * sech2 * du
    }

    fn assert_eq_bits_or_nan(got: f32, want: f32, ctx: &str) {
        if want.is_nan() {
            assert!(got.is_nan(), "{ctx}: expected NaN, got {got}");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{ctx}: got {got} ({:#010x}) vs want {want} ({:#010x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    #[test]
    fn gelu_and_swiglu_slices_match_scalar_bitwise_and_extremes() {
        let mut xs = vec![
            0.0f32,
            -0.0,
            1e-30,
            -1e-30,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            -0.5,
            0.5,
            -1.0,
            1.0,
            -3.0,
            3.0,
            -11.9,
            11.9,
            -12.0,
            12.0,
            -12.1,
            12.1,
            -100.0,
            100.0,
            -1e20,
            1e20,
            f32::MIN,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
        ];
        let mut rng = crate::rng::Rng::seed(314);
        for _ in 0..45 {
            xs.push(rng.uniform(-8.0, 8.0));
        }
        let gs: Vec<f32> = (0..xs.len()).map(|i| ((i as f32) * 0.37 - 5.1).sin() * 2.5).collect();
        let ups: Vec<f32> = (0..xs.len()).map(|i| ((i as f32) * 0.71 + 1.3).cos() * 3.0).collect();

        for len in [0usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, xs.len()] {
            let x = &xs[..len];
            let g = &gs[..len];
            let u = &ups[..len];

            let mut gy = vec![0f32; len];
            gelu_slice(x, &mut gy);
            for (i, (&got, &xv)) in gy.iter().zip(x).enumerate() {
                assert_eq_bits_or_nan(got, gelu(xv), &format!("gelu len={len} i={i} x={xv}"));
            }

            let mut gb = vec![0f32; len];
            gelu_backward_slice(x, g, &mut gb);
            for (i, ((&got, &xv), &gv)) in gb.iter().zip(x).zip(g).enumerate() {
                assert_eq_bits_or_nan(
                    got,
                    gelu_backward(xv, gv),
                    &format!("gelu_bwd len={len} i={i} x={xv}"),
                );
            }

            let mut sb = vec![0f32; len];
            silu_backward_slice(x, g, &mut sb);
            for (i, ((&got, &xv), &gv)) in sb.iter().zip(x).zip(g).enumerate() {
                assert_eq_bits_or_nan(
                    got,
                    crate::cpu::silu_backward(xv, gv),
                    &format!("silu_bwd len={len} i={i} x={xv}"),
                );
            }

            let mut sw = vec![0f32; len];
            swiglu_slice(x, u, &mut sw);
            for (i, ((&got, &xv), &uv)) in sw.iter().zip(x).zip(u).enumerate() {
                assert_eq_bits_or_nan(got, swiglu(xv, uv), &format!("swiglu len={len} i={i} x={xv}"));
            }

            let mut dg = vec![0f32; len];
            let mut du = vec![0f32; len];
            swiglu_backward_slice(x, u, g, &mut dg, &mut du);
            for i in 0..len {
                let (wg, wu) = swiglu_backward(x[i], u[i], g[i]);
                assert_eq_bits_or_nan(dg[i], wg, &format!("swiglu_bwd dgate len={len} i={i}"));
                assert_eq_bits_or_nan(du[i], wu, &format!("swiglu_bwd dup len={len} i={i}"));
            }
        }
        // Extreme overflow-avoidance check for swiglu_backward on saturated negative gate (`exp(-105) == 0.0`,
        // while `exp(-100)` is a non-zero subnormal so `grad * silu'(-100) * up` stays finite).
        let (mut dg, mut du) = ([0f32; 16], [0f32; 16]);
        swiglu_backward_slice(&[-105.0; 16], &[1e20; 16], &[1e20; 16], &mut dg, &mut du);
        assert!(dg.iter().all(|&v| v == 0.0) && du.iter().all(|&v| v == 0.0));
        swiglu_backward_slice(&[-100.0; 16], &[1e20; 16], &[1e20; 16], &mut dg, &mut du);
        assert!(dg.iter().all(|v| v.is_finite()) && du.iter().all(|v| v.is_finite()));
        // Finite large inputs for GELU stay exact & non-NaN.
        assert_eq!(gelu(f32::MAX), f32::MAX);
        assert_eq!(gelu(f32::MIN), 0.0);
        assert_eq!(gelu(f32::INFINITY), f32::INFINITY);
        assert_eq!(gelu(f32::NEG_INFINITY), 0.0);
        assert_eq!(gelu_backward(f32::MAX, 2.5), 2.5);
        assert_eq!(gelu_backward(f32::MIN, 2.5), 0.0);
    }

    #[test]
    fn gelu_and_swiglu_accuracy_and_finite_differences() {
        // Accuracy against f64 exact formulas across [-10, 10].
        for i in -200..=200 {
            let x = i as f32 * 0.05;
            let got = gelu(x) as f64;
            let want = gelu_f64_ref(x as f64);
            assert!((got - want).abs() <= 2e-6 * want.abs().max(1.0), "gelu({x}): got {got}, want {want}");
            let got_d = gelu_backward(x, 1.0) as f64;
            let want_d = gelu_deriv_f64_ref(x as f64);
            assert!(
                (got_d - want_d).abs() <= 5e-6 * want_d.abs().max(1.0),
                "gelu'({x}): got {got_d}, want {want_d}"
            );
        }
        // Central finite-difference check on GELU and SwiGLU.
        let eps = 1e-3f32;
        let xs: Vec<f32> = (-12..=12).map(|k| k as f32 * 0.35).collect();
        let ups: Vec<f32> = (0..xs.len()).map(|k| (k as f32 - 12.0) * 0.25 + 0.1).collect();
        let gs: Vec<f32> = (0..xs.len()).map(|k| ((k as f32) * 0.5).cos()).collect();
        let mut g_bwd = vec![0f32; xs.len()];
        gelu_backward_slice(&xs, &gs, &mut g_bwd);
        for i in 0..xs.len() {
            let fd = gs[i] * (gelu(xs[i] + eps) - gelu(xs[i] - eps)) / (2.0 * eps);
            assert!(
                (g_bwd[i] - fd).abs() <= 1e-3 * fd.abs().max(1.0),
                "gelu fd at {}: {} vs {}",
                xs[i],
                g_bwd[i],
                fd
            );
        }
        let (mut dg, mut du) = (vec![0f32; xs.len()], vec![0f32; xs.len()]);
        swiglu_backward_slice(&xs, &ups, &gs, &mut dg, &mut du);
        for i in 0..xs.len() {
            let fd_g = gs[i] * (swiglu(xs[i] + eps, ups[i]) - swiglu(xs[i] - eps, ups[i])) / (2.0 * eps);
            let fd_u = gs[i] * (swiglu(xs[i], ups[i] + eps) - swiglu(xs[i], ups[i] - eps)) / (2.0 * eps);
            assert!((dg[i] - fd_g).abs() <= 1e-3 * fd_g.abs().max(1.0), "swiglu dgate fd at {i}");
            assert!((du[i] - fd_u).abs() <= 1e-3 * fd_u.abs().max(1.0), "swiglu dup fd at {i}");
        }
    }
}
