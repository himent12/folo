//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) elementwise transcendental and binary kernels.
//!
//! Every kernel here uses non-FMA IEEE-754 operations (`f32x4_mul`, `f32x4_add`, `f32x4_sub`, `f32x4_div`)
//! and is bitwise identical to its scalar twin in `simd.rs` / `simd_log_tanh.rs` / `elementwise/mod.rs` on
//! all non-NaN inputs (including `±0.0`, subnormals, and `±∞`), while preserving NaN semantics (and input
//! NaN payloads on single-operand transcendentals).

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::simd::{
    EXP_MAGIC_0, EXP_MAGIC_0_BITS, EXP5_C1, EXP5_C2, EXP5_C3, EXP5_C4, EXP5_C5, LN2_HI, LN2_LO, LOG2E,
    RECIP_MAGIC,
};
use super::simd_log_tanh::{
    LN_C2, LN_C3, LN_C4, LN_C5, LN_C6, LN_C7, LN_C8, LN_C9, LN_C10, LN_MAGIC, LN1P_C1, LN1P_C2, LN1P_C3,
    LN1P_C4, QNAN_BITS, SQRT2_HALF_BITS, SUBNORMAL_SCALE, TANH_C1, TANH_C2, TANH_C3, TANH_C4, TANH_C5,
    TANH_SMALL_CUTOFF,
};
use crate::backend::BinaryOp;

// ---------------------------------------------------------------------------
// 4-lane transcendental primitives (bitwise identical to scalar twins)
// ---------------------------------------------------------------------------

/// Core non-FMA `exp(xc)` on four lanes for `xc` already clamped into `[-104.0, 89.0]` (never NaN).
#[inline(always)]
pub(super) fn exp_clamped4(xc: v128) -> v128 {
    let one = f32x4_splat(1.0);
    let magic0 = f32x4_splat(EXP_MAGIC_0);
    let z = f32x4_add(f32x4_mul(xc, f32x4_splat(LOG2E)), magic0);
    let n = f32x4_sub(z, magic0);
    let r = f32x4_sub(xc, f32x4_mul(n, f32x4_splat(LN2_HI)));
    let r = f32x4_sub(r, f32x4_mul(n, f32x4_splat(LN2_LO)));
    let mut p = f32x4_splat(EXP5_C5);
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP5_C4));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP5_C3));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP5_C2));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP5_C1));
    let y = f32x4_add(f32x4_mul(p, r), one);
    let ni = i32x4_sub(z, i32x4_splat(EXP_MAGIC_0_BITS));
    let n1 = i32x4_shr(ni, 1);
    let n2 = i32x4_sub(ni, n1);
    let bias127 = i32x4_splat(127);
    let s1 = i32x4_shl(i32x4_add(n1, bias127), 23);
    let s2 = i32x4_shl(i32x4_add(n2, bias127), 23);
    f32x4_mul(f32x4_mul(y, s1), s2)
}

/// Non-FMA `exp(-|x|)` on four lanes, clamped to `-104.0` (NaN maps to `-104.0`; callers propagate NaN).
#[inline(always)]
pub(super) fn exp_neg_abs4(x: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let t_abs = f32x4_abs(x);
    // `f32x4_pmax(a, b)` returns `a < b ? b : a`, so placing `-104.0` first matches `maxps(-|x|, -104.0)` on NaN.
    let t = f32x4_pmax(f32x4_splat(-104.0), f32x4_sub(zero, t_abs));
    exp_clamped4(t)
}

/// Division-free non-FMA `1 / (1 + e)` on four lanes for `e ∈ [0, 1]`, via bit-hack seed + Householder-6.
#[inline(always)]
pub(super) fn recip_1_plus4(e: v128) -> v128 {
    let one = f32x4_splat(1.0);
    let d = f32x4_add(one, e);
    let r0 = i32x4_sub(i32x4_splat(RECIP_MAGIC as i32), d);
    let e0 = f32x4_sub(one, f32x4_mul(d, r0));
    let u = f32x4_add(r0, f32x4_mul(r0, e0));
    let e2 = f32x4_mul(e0, e0);
    let t2 = f32x4_add(e2, f32x4_mul(e2, e2));
    f32x4_add(u, f32x4_mul(u, t2))
}

/// `exp(x)` on four lanes, bitwise identical to `exp_scalar` (preserves NaN payload).
#[inline(always)]
pub(crate) fn exp4(x: v128) -> v128 {
    let xc = f32x4_pmin(f32x4_splat(89.0), f32x4_pmax(f32x4_splat(-104.0), x));
    let v = exp_clamped4(xc);
    v128_bitselect(x, v, f32x4_ne(x, x))
}

/// `1 / (1 + exp(-x))` on four lanes, division-free and bitwise identical to `sigmoid_scalar`.
#[inline(always)]
pub(crate) fn sigmoid4(x: v128) -> v128 {
    let e = exp_neg_abs4(x);
    let rec = recip_1_plus4(e);
    let zero = f32x4_splat(0.0);
    let sig = v128_bitselect(rec, f32x4_mul(e, rec), f32x4_ge(x, zero));
    v128_bitselect(x, sig, f32x4_ne(x, x))
}

/// `x * sigmoid(x)` on four lanes (non-FMA), division-free and bitwise identical to `silu_scalar`.
#[inline(always)]
pub(crate) fn silu4(x: v128) -> v128 {
    let e = exp_neg_abs4(x);
    let rec = recip_1_plus4(e);
    let zero = f32x4_splat(0.0);
    let sig = v128_bitselect(rec, f32x4_mul(e, rec), f32x4_ge(x, zero));
    f32x4_mul(x, sig)
}

/// `g * exp(x)` on four lanes, bitwise identical to `g * exp_scalar(x)`.
#[inline(always)]
fn exp_bwd4(x: v128, g: v128) -> v128 {
    f32x4_mul(g, exp4(x))
}

/// `g * sigmoid'(x)` on four lanes, bitwise identical to `g * sigmoid_deriv_scalar(x)`.
#[inline(always)]
fn sigmoid_bwd4(x: v128, g: v128) -> v128 {
    let e = exp_neg_abs4(x);
    let rec = recip_1_plus4(e);
    let ds = f32x4_mul(f32x4_mul(e, rec), rec);
    let ds = v128_bitselect(x, ds, f32x4_ne(x, x));
    f32x4_mul(g, ds)
}

/// `g * silu'(x)` on four lanes, bitwise identical to `silu_bwd_scalar(x, g)`.
#[inline(always)]
fn silu_bwd4(x: v128, g: v128) -> v128 {
    let e = exp_neg_abs4(x);
    let rec = recip_1_plus4(e);
    let zero = f32x4_splat(0.0);
    let er = f32x4_mul(e, rec);
    let sig = v128_bitselect(rec, er, f32x4_ge(x, zero));
    let ds = f32x4_add(sig, f32x4_mul(x, f32x4_mul(er, rec)));
    f32x4_mul(g, ds)
}

/// `g * sigmoid(x)` on four lanes, bitwise identical to `g * sigmoid_scalar(x)`.
#[inline(always)]
fn softplus_bwd4(x: v128, g: v128) -> v128 {
    f32x4_mul(g, sigmoid4(x))
}

/// `ln(x)` on four lanes, division-free and bitwise identical to `ln_scalar`.
#[inline(always)]
fn ln4(x: v128) -> v128 {
    let is_sub = f32x4_lt(x, f32x4_splat(f32::MIN_POSITIVE));
    let x_norm = v128_bitselect(f32x4_mul(x, f32x4_splat(SUBNORMAL_SCALE)), x, is_sub);
    let exp_adj = v128_and(is_sub, i32x4_splat(-24));
    let u_adj = i32x4_add(x_norm, i32x4_splat(LN_MAGIC));
    let k_raw = i32x4_sub(i32x4_shr(u_adj, 23), i32x4_splat(127));
    let k = i32x4_add(k_raw, exp_adj);
    let m = i32x4_add(v128_and(u_adj, i32x4_splat(0x007F_FFFF)), i32x4_splat(SQRT2_HALF_BITS));
    let f = f32x4_sub(m, f32x4_splat(1.0));
    let f2 = f32x4_mul(f, f);
    let f4 = f32x4_mul(f2, f2);
    let q0 = f32x4_add(f32x4_mul(f32x4_splat(LN_C3), f), f32x4_splat(LN_C2));
    let q1 = f32x4_add(f32x4_mul(f32x4_splat(LN_C5), f), f32x4_splat(LN_C4));
    let q2 = f32x4_add(f32x4_mul(f32x4_splat(LN_C7), f), f32x4_splat(LN_C6));
    let q3 = f32x4_add(f32x4_mul(f32x4_splat(LN_C9), f), f32x4_splat(LN_C8));
    let r0 = f32x4_add(f32x4_mul(q1, f2), q0);
    let t3 = f32x4_add(f32x4_mul(f32x4_splat(LN_C10), f2), q3);
    let r1 = f32x4_add(f32x4_mul(t3, f2), q2);
    let p = f32x4_add(f32x4_mul(r1, f4), r0);
    let ln_m = f32x4_add(f, f32x4_mul(f2, p));
    let kf = f32x4_convert_i32x4(k);
    let mut res =
        f32x4_add(f32x4_add(f32x4_mul(kf, f32x4_splat(LN2_LO)), ln_m), f32x4_mul(kf, f32x4_splat(LN2_HI)));
    let zero = f32x4_splat(0.0);
    res = v128_bitselect(f32x4_splat(f32::NEG_INFINITY), res, f32x4_eq(x, zero));
    res = v128_bitselect(f32x4_splat(f32::from_bits(QNAN_BITS)), res, f32x4_lt(x, zero));
    let pos_inf = f32x4_splat(f32::INFINITY);
    res = v128_bitselect(pos_inf, res, f32x4_eq(x, pos_inf));
    v128_bitselect(x, res, f32x4_ne(x, x))
}

/// `tanh(x)` on four lanes, division-free and bitwise identical to `tanh_scalar`.
#[inline(always)]
fn tanh4(x: v128) -> v128 {
    let sign_bit = f32x4_splat(-0.0);
    let ax = f32x4_abs(x);
    let sign = v128_and(sign_bit, x);
    let x2 = f32x4_mul(ax, ax);
    let mut q = f32x4_splat(TANH_C5);
    q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(TANH_C4));
    q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(TANH_C3));
    q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(TANH_C2));
    q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(TANH_C1));
    let poly = f32x4_add(ax, f32x4_mul(ax, f32x4_mul(x2, q)));
    let t = f32x4_pmax(f32x4_splat(-104.0), f32x4_mul(f32x4_splat(-2.0), ax));
    let e = exp_clamped4(t);
    let rec = recip_1_plus4(e);
    let large = f32x4_sub(f32x4_splat(1.0), f32x4_mul(f32x4_splat(2.0), f32x4_mul(e, rec)));
    let mag = v128_bitselect(poly, large, f32x4_lt(ax, f32x4_splat(TANH_SMALL_CUTOFF)));
    let signed = v128_or(mag, sign);
    v128_bitselect(x, signed, f32x4_ne(x, x))
}

/// `g * sech^2(x)` on four lanes, bitwise identical to `g * tanh_deriv_scalar(x)`.
#[inline(always)]
fn tanh_bwd4(x: v128, g: v128) -> v128 {
    let ax = f32x4_abs(x);
    let t = f32x4_pmax(f32x4_splat(-104.0), f32x4_mul(f32x4_splat(-2.0), ax));
    let e = exp_clamped4(t);
    let rec = recip_1_plus4(e);
    let ds = f32x4_pmin(f32x4_splat(1.0), f32x4_mul(f32x4_mul(f32x4_mul(f32x4_splat(4.0), e), rec), rec));
    let ds = v128_bitselect(x, ds, f32x4_ne(x, x));
    f32x4_mul(g, ds)
}

/// `max(x, 0) + ln(1 + exp(-|x|))` on four lanes, bitwise identical to `softplus_scalar`.
#[inline(always)]
fn softplus4(x: v128) -> v128 {
    let half = f32x4_splat(0.5);
    let e = exp_neg_abs4(x);
    let r = recip_1_plus4(f32x4_mul(half, e));
    let u = f32x4_mul(e, r);
    let z = f32x4_mul(half, u);
    let z2 = f32x4_mul(z, z);
    let mut p = f32x4_splat(LN1P_C4);
    p = f32x4_add(f32x4_mul(p, z2), f32x4_splat(LN1P_C3));
    p = f32x4_add(f32x4_mul(p, z2), f32x4_splat(LN1P_C2));
    p = f32x4_add(f32x4_mul(p, z2), f32x4_splat(LN1P_C1));
    let ln1p_e = f32x4_add(u, f32x4_mul(u, f32x4_mul(z2, p)));
    let sp = f32x4_add(f32x4_pmax(f32x4_splat(0.0), x), ln1p_e);
    v128_bitselect(x, sp, f32x4_ne(x, x))
}

// ---------------------------------------------------------------------------
// 2x4-unrolled + 4-wide + padded-tail slice runners
// ---------------------------------------------------------------------------

#[inline]
fn run_unary_slice(x: &[f32], out: &mut [f32], op: impl Fn(v128) -> v128) {
    debug_assert_eq!(x.len(), out.len());
    let (xp, xr) = x.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for (xc, oc) in xp.iter().zip(op8.iter_mut()) {
        // SAFETY: `xc` and `oc` are 8 floats; Wasm `v128_load`/`v128_store` allow unaligned access.
        unsafe {
            let sp = xc.as_ptr();
            let dp = oc.as_mut_ptr();
            let v0 = op(v128_load(sp as *const v128));
            let v1 = op(v128_load(sp.add(4) as *const v128));
            v128_store(dp as *mut v128, v0);
            v128_store(dp.add(4) as *mut v128, v1);
        }
    }
    let (x4, xt) = xr.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for (xc, oc) in x4.iter().zip(o4.iter_mut()) {
        // SAFETY: `xc` and `oc` are 4 floats.
        unsafe {
            let v = op(v128_load(xc.as_ptr() as *const v128));
            v128_store(oc.as_mut_ptr() as *mut v128, v);
        }
    }
    if !xt.is_empty() {
        let mut pad = [0f32; 4];
        pad[..xt.len()].copy_from_slice(xt);
        // SAFETY: `pad` is 4 floats.
        unsafe {
            let v = op(v128_load(pad.as_ptr() as *const v128));
            v128_store(pad.as_mut_ptr() as *mut v128, v);
        }
        ot.copy_from_slice(&pad[..xt.len()]);
    }
}

#[inline]
fn run_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32], op: impl Fn(v128, v128) -> v128) {
    debug_assert_eq!(x.len(), g.len());
    debug_assert_eq!(x.len(), out.len());
    let (xp, xr) = x.as_chunks::<8>();
    let (gp, gr) = g.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for ((xc, gc), oc) in xp.iter().zip(gp.iter()).zip(op8.iter_mut()) {
        // SAFETY: `xc`, `gc`, `oc` are 8 floats.
        unsafe {
            let xp = xc.as_ptr();
            let gp = gc.as_ptr();
            let dp = oc.as_mut_ptr();
            let v0 = op(v128_load(xp as *const v128), v128_load(gp as *const v128));
            let v1 = op(v128_load(xp.add(4) as *const v128), v128_load(gp.add(4) as *const v128));
            v128_store(dp as *mut v128, v0);
            v128_store(dp.add(4) as *mut v128, v1);
        }
    }
    let (x4, xt) = xr.as_chunks::<4>();
    let (g4, gt) = gr.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for ((xc, gc), oc) in x4.iter().zip(g4.iter()).zip(o4.iter_mut()) {
        // SAFETY: `xc`, `gc`, `oc` are 4 floats.
        unsafe {
            let v = op(v128_load(xc.as_ptr() as *const v128), v128_load(gc.as_ptr() as *const v128));
            v128_store(oc.as_mut_ptr() as *mut v128, v);
        }
    }
    if !xt.is_empty() {
        let mut xpad = [0f32; 4];
        let mut gpad = [0f32; 4];
        xpad[..xt.len()].copy_from_slice(xt);
        gpad[..gt.len()].copy_from_slice(gt);
        // SAFETY: `xpad` and `gpad` are 4 floats.
        unsafe {
            let v = op(v128_load(xpad.as_ptr() as *const v128), v128_load(gpad.as_ptr() as *const v128));
            v128_store(xpad.as_mut_ptr() as *mut v128, v);
        }
        ot.copy_from_slice(&xpad[..xt.len()]);
    }
}

#[inline]
fn run_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32], op: impl Fn(v128, v128) -> v128) {
    debug_assert_eq!(x.len(), out.len());
    let gv = f32x4_splat(g);
    let (xp, xr) = x.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for (xc, oc) in xp.iter().zip(op8.iter_mut()) {
        // SAFETY: `xc` and `oc` are 8 floats.
        unsafe {
            let xp = xc.as_ptr();
            let dp = oc.as_mut_ptr();
            let v0 = op(v128_load(xp as *const v128), gv);
            let v1 = op(v128_load(xp.add(4) as *const v128), gv);
            v128_store(dp as *mut v128, v0);
            v128_store(dp.add(4) as *mut v128, v1);
        }
    }
    let (x4, xt) = xr.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for (xc, oc) in x4.iter().zip(o4.iter_mut()) {
        // SAFETY: `xc` and `oc` are 4 floats.
        unsafe {
            let v = op(v128_load(xc.as_ptr() as *const v128), gv);
            v128_store(oc.as_mut_ptr() as *mut v128, v);
        }
    }
    if !xt.is_empty() {
        let mut xpad = [0f32; 4];
        xpad[..xt.len()].copy_from_slice(xt);
        // SAFETY: `xpad` is 4 floats.
        unsafe {
            let v = op(v128_load(xpad.as_ptr() as *const v128), gv);
            v128_store(xpad.as_mut_ptr() as *mut v128, v);
        }
        ot.copy_from_slice(&xpad[..xt.len()]);
    }
}

#[inline]
fn run_bias_inplace(v: &mut [f32], bias: f32, op: impl Fn(v128) -> v128) {
    let b = f32x4_splat(bias);
    let (pairs, rem) = v.as_chunks_mut::<8>();
    for c in pairs {
        // SAFETY: `c` is 8 floats.
        unsafe {
            let p = c.as_mut_ptr();
            let x0 = f32x4_add(v128_load(p as *const v128), b);
            let x1 = f32x4_add(v128_load(p.add(4) as *const v128), b);
            v128_store(p as *mut v128, op(x0));
            v128_store(p.add(4) as *mut v128, op(x1));
        }
    }
    let (chunks, tail) = rem.as_chunks_mut::<4>();
    for c in chunks {
        // SAFETY: `c` is 4 floats.
        unsafe {
            let x = f32x4_add(v128_load(c.as_ptr() as *const v128), b);
            v128_store(c.as_mut_ptr() as *mut v128, op(x));
        }
    }
    if !tail.is_empty() {
        let mut pad = [0f32; 4];
        pad[..tail.len()].copy_from_slice(tail);
        // SAFETY: `pad` is 4 floats.
        unsafe {
            let x = f32x4_add(v128_load(pad.as_ptr() as *const v128), b);
            v128_store(pad.as_mut_ptr() as *mut v128, op(x));
        }
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

// Exported slice entry points matching `simd.rs` and `simd_log_tanh.rs`.
pub(super) fn exp_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, exp4);
}
pub(super) fn sigmoid_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, sigmoid4);
}
pub(super) fn silu_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, silu4);
}
pub(super) fn ln_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, ln4);
}
pub(super) fn tanh_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, tanh4);
}
pub(super) fn softplus_slice(x: &[f32], out: &mut [f32]) {
    run_unary_slice(x, out, softplus4);
}

pub(super) fn exp_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, exp_bwd4);
}
pub(super) fn exp_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, exp_bwd4);
}
pub(super) fn sigmoid_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, sigmoid_bwd4);
}
pub(super) fn sigmoid_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, sigmoid_bwd4);
}
pub(super) fn silu_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, silu_bwd4);
}
pub(super) fn silu_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, silu_bwd4);
}
pub(super) fn softplus_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, softplus_bwd4);
}
pub(super) fn softplus_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, softplus_bwd4);
}
pub(super) fn tanh_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, tanh_bwd4);
}
pub(super) fn tanh_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, tanh_bwd4);
}
pub(super) fn log_bwd_slice(x: &[f32], g: &[f32], out: &mut [f32]) {
    run_bwd_slice(x, g, out, |xv, gv| f32x4_div(gv, xv));
}
pub(super) fn log_bwd_bc_slice(x: &[f32], g: f32, out: &mut [f32]) {
    run_bwd_bc_slice(x, g, out, |xv, gv| f32x4_div(gv, xv));
}

pub(super) fn silu_bias_inplace(v: &mut [f32], bias: f32) {
    run_bias_inplace(v, bias, silu4);
}
pub(super) fn sigmoid_bias_inplace(v: &mut [f32], bias: f32) {
    run_bias_inplace(v, bias, sigmoid4);
}
pub(super) fn tanh_bias_inplace(v: &mut [f32], bias: f32) {
    run_bias_inplace(v, bias, tanh4);
}

// ---------------------------------------------------------------------------
// 4-lane binary ops (contiguous, broadcast-b, broadcast-a), IEEE-754 exact
// ---------------------------------------------------------------------------

/// IEEE-754 `f32::max` (`maxNum`: non-NaN wins; `if a > b || b.is_nan() { a } else { b }`).
#[inline(always)]
fn max4_exact(a: v128, b: v128) -> v128 {
    let a_wins = v128_or(f32x4_gt(a, b), f32x4_ne(b, b));
    v128_bitselect(a, b, a_wins)
}

/// IEEE-754 `f32::min` (`minNum`: non-NaN wins; `if a < b || b.is_nan() { a } else { b }`).
#[inline(always)]
fn min4_exact(a: v128, b: v128) -> v128 {
    let a_wins = v128_or(f32x4_lt(a, b), f32x4_ne(b, b));
    v128_bitselect(a, b, a_wins)
}

#[inline]
fn run_bin_vv(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    vop: impl Fn(v128, v128) -> v128,
    sop: impl Fn(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let (ap, ar) = a.as_chunks::<8>();
    let (bp, br) = b.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in ap.iter().zip(bp.iter()).zip(op8.iter_mut()) {
        // SAFETY: `ac`, `bc`, `oc` are 8 floats.
        unsafe {
            let ap = ac.as_ptr();
            let bp = bc.as_ptr();
            let dp = oc.as_mut_ptr();
            v128_store(dp as *mut v128, vop(v128_load(ap as *const v128), v128_load(bp as *const v128)));
            v128_store(
                dp.add(4) as *mut v128,
                vop(v128_load(ap.add(4) as *const v128), v128_load(bp.add(4) as *const v128)),
            );
        }
    }
    let (a4, at) = ar.as_chunks::<4>();
    let (b4, bt) = br.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for ((ac, bc), oc) in a4.iter().zip(b4.iter()).zip(o4.iter_mut()) {
        // SAFETY: `ac`, `bc`, `oc` are 4 floats.
        unsafe {
            v128_store(
                oc.as_mut_ptr() as *mut v128,
                vop(v128_load(ac.as_ptr() as *const v128), v128_load(bc.as_ptr() as *const v128)),
            );
        }
    }
    for ((o, &av), &bv) in ot.iter_mut().zip(at).zip(bt) {
        *o = sop(av, bv);
    }
}

#[inline]
fn run_bin_vs(
    a: &[f32],
    b: f32,
    out: &mut [f32],
    vop: impl Fn(v128, v128) -> v128,
    sop: impl Fn(f32, f32) -> f32,
) {
    debug_assert_eq!(a.len(), out.len());
    let bv = f32x4_splat(b);
    let (ap, ar) = a.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for (ac, oc) in ap.iter().zip(op8.iter_mut()) {
        // SAFETY: `ac` and `oc` are 8 floats.
        unsafe {
            let ap = ac.as_ptr();
            let dp = oc.as_mut_ptr();
            v128_store(dp as *mut v128, vop(v128_load(ap as *const v128), bv));
            v128_store(dp.add(4) as *mut v128, vop(v128_load(ap.add(4) as *const v128), bv));
        }
    }
    let (a4, at) = ar.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for (ac, oc) in a4.iter().zip(o4.iter_mut()) {
        // SAFETY: `ac` and `oc` are 4 floats.
        unsafe {
            v128_store(oc.as_mut_ptr() as *mut v128, vop(v128_load(ac.as_ptr() as *const v128), bv));
        }
    }
    for (o, &av) in ot.iter_mut().zip(at) {
        *o = sop(av, b);
    }
}

#[inline]
fn run_bin_sv(
    a: f32,
    b: &[f32],
    out: &mut [f32],
    vop: impl Fn(v128, v128) -> v128,
    sop: impl Fn(f32, f32) -> f32,
) {
    debug_assert_eq!(b.len(), out.len());
    let av = f32x4_splat(a);
    let (bp, br) = b.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for (bc, oc) in bp.iter().zip(op8.iter_mut()) {
        // SAFETY: `bc` and `oc` are 8 floats.
        unsafe {
            let bp = bc.as_ptr();
            let dp = oc.as_mut_ptr();
            v128_store(dp as *mut v128, vop(av, v128_load(bp as *const v128)));
            v128_store(dp.add(4) as *mut v128, vop(av, v128_load(bp.add(4) as *const v128)));
        }
    }
    let (b4, bt) = br.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for (bc, oc) in b4.iter().zip(o4.iter_mut()) {
        // SAFETY: `bc` and `oc` are 4 floats.
        unsafe {
            v128_store(oc.as_mut_ptr() as *mut v128, vop(av, v128_load(bc.as_ptr() as *const v128)));
        }
    }
    for (o, &bv) in ot.iter_mut().zip(bt) {
        *o = sop(a, bv);
    }
}

/// Vectorized contiguous/unit-stride `(1, 1)` slice kernel for hot binary ops (`Add`/`Sub`/`Mul`/`Div`/`Maximum`/`Minimum`).
pub(super) fn binary_vv(op: BinaryOp, a: &[f32], b: &[f32], out: &mut [f32]) {
    match op {
        BinaryOp::Add => run_bin_vv(a, b, out, |x, y| f32x4_add(x, y), |x, y| x + y),
        BinaryOp::Sub => run_bin_vv(a, b, out, |x, y| f32x4_sub(x, y), |x, y| x - y),
        BinaryOp::Mul => run_bin_vv(a, b, out, |x, y| f32x4_mul(x, y), |x, y| x * y),
        BinaryOp::Div => run_bin_vv(a, b, out, |x, y| f32x4_div(x, y), |x, y| x / y),
        BinaryOp::Maximum => run_bin_vv(a, b, out, max4_exact, f32::max),
        BinaryOp::Minimum => run_bin_vv(a, b, out, min4_exact, f32::min),
        _ => unreachable!("binary_vv called for non-SIMD binary op"),
    }
}

/// Vectorized broadcast-`b` `(1, 0)` slice kernel for hot binary ops.
pub(super) fn binary_vs(op: BinaryOp, a: &[f32], b: f32, out: &mut [f32]) {
    match op {
        BinaryOp::Add => run_bin_vs(a, b, out, |x, y| f32x4_add(x, y), |x, y| x + y),
        BinaryOp::Sub => run_bin_vs(a, b, out, |x, y| f32x4_sub(x, y), |x, y| x - y),
        BinaryOp::Mul => run_bin_vs(a, b, out, |x, y| f32x4_mul(x, y), |x, y| x * y),
        BinaryOp::Div => run_bin_vs(a, b, out, |x, y| f32x4_div(x, y), |x, y| x / y),
        BinaryOp::Maximum => run_bin_vs(a, b, out, max4_exact, f32::max),
        BinaryOp::Minimum => run_bin_vs(a, b, out, min4_exact, f32::min),
        _ => unreachable!("binary_vs called for non-SIMD binary op"),
    }
}

/// Vectorized broadcast-`a` `(0, 1)` slice kernel for hot binary ops.
pub(super) fn binary_sv(op: BinaryOp, a: f32, b: &[f32], out: &mut [f32]) {
    match op {
        BinaryOp::Add => run_bin_sv(a, b, out, |x, y| f32x4_add(x, y), |x, y| x + y),
        BinaryOp::Sub => run_bin_sv(a, b, out, |x, y| f32x4_sub(x, y), |x, y| x - y),
        BinaryOp::Mul => run_bin_sv(a, b, out, |x, y| f32x4_mul(x, y), |x, y| x * y),
        BinaryOp::Div => run_bin_sv(a, b, out, |x, y| f32x4_div(x, y), |x, y| x / y),
        BinaryOp::Maximum => run_bin_sv(a, b, out, max4_exact, f32::max),
        BinaryOp::Minimum => run_bin_sv(a, b, out, min4_exact, f32::min),
        _ => unreachable!("binary_sv called for non-SIMD binary op"),
    }
}

/// Dispatch a hot binary op (`Add`/`Sub`/`Mul`/`Div`/`Maximum`/`Minimum`) across contiguous, unit-stride,
/// broadcast-`b`, broadcast-`a`, and general-strided layouts using Wasm SIMD128.
pub(super) fn binary_map2(
    op: BinaryOp,
    a: &[f32],
    la: &crate::Layout,
    b: &[f32],
    lb: &crate::Layout,
    scalar: fn(f32, f32) -> f32,
) -> Vec<f32> {
    let n = la.numel();
    let mut out = crate::cpu::pool_alloc::alloc_uninit(n);
    if let (Some(ra), Some(rb)) = (la.contiguous_range(), lb.contiguous_range()) {
        let (a, b) = (&a[ra], &b[rb]);
        let piece = |c: usize, o: &mut [f32]| {
            let r = c * crate::cpu::CHUNK..c * crate::cpu::CHUNK + o.len();
            binary_vv(op, &a[r.clone()], &b[r], o);
        };
        if n >= crate::cpu::PAR_MIN {
            crate::cpu::par_chunks(&mut out, crate::cpu::CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan =
        crate::layout::RowPlan::<2>::new(la.dims(), [la.strides(), lb.strides()], [la.offset(), lb.offset()]);
    let [sa, sb] = plan.inner;
    super::fill_rows(&mut out, &plan, |row, [oa, ob]| {
        let len = row.len();
        match (sa, sb) {
            (1, 1) => binary_vv(op, &a[oa..oa + len], &b[ob..ob + len], row),
            (1, 0) => binary_vs(op, &a[oa..oa + len], b[ob], row),
            (0, 1) => binary_sv(op, a[oa], &b[ob..ob + len], row),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = scalar(a[oa + i * sa], b[ob + i * sb]);
                }
            }
        }
    });
    out
}
