//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) unary activations, math/scalar ops, unary backwards,
//! fused bias+activation epilogues, and `where_cond` (`F32` / `U8`), bitwise identical to scalar IEEE-754.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::wasm_simd::{max4_exact, min4_exact, run_bin_sv, run_bin_vs, run_bin_vv};
use super::{fill_rows, max_num, min_num, sign};
use crate::Layout;
use crate::backend::UnaryOp;
use crate::cpu::pool_alloc::alloc_uninit;
use crate::cpu::{CHUNK, PAR_MIN, par_chunks};
use crate::layout::RowPlan;

#[inline(always)]
fn hardswish_scalar(v: f32) -> f32 {
    v * (v + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)
}

#[inline(always)]
fn hardswish4(v: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let c3 = f32x4_splat(3.0);
    let c6 = f32x4_splat(6.0);
    let inv6 = f32x4_splat(1.0 / 6.0);
    let clamped = f32x4_pmin(c6, f32x4_pmax(zero, f32x4_add(v, c3)));
    f32x4_mul(f32x4_mul(v, clamped), inv6)
}

#[inline(always)]
fn hardswish_bwd_scalar(x: f32, g: f32) -> f32 {
    if x < -3.0 {
        0.0
    } else if x > 3.0 {
        g
    } else {
        g * (2.0 * x + 3.0) * (1.0 / 6.0)
    }
}

#[inline(always)]
fn hardswish_bwd4(x: v128, g: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let two = f32x4_splat(2.0);
    let c3 = f32x4_splat(3.0);
    let neg3 = f32x4_splat(-3.0);
    let inv6 = f32x4_splat(1.0 / 6.0);
    let mid = f32x4_mul(f32x4_mul(g, f32x4_add(f32x4_mul(two, x), c3)), inv6);
    let r = v128_bitselect(g, mid, f32x4_gt(x, c3));
    v128_bitselect(zero, r, f32x4_lt(x, neg3))
}

/// Exact round-half-away-from-zero (`f32::round`) across all 2^32 binary32 values:
/// `copysign(trunc(|v| + (0.5 - 2^-25)), v)`.
#[inline(always)]
fn round_away4(v: v128) -> v128 {
    let sign_bit = f32x4_splat(-0.0);
    let prev_half = f32x4_splat(f32::from_bits(0x3eff_ffff));
    let s = v128_and(v, sign_bit);
    let r = f32x4_trunc(f32x4_add(f32x4_abs(v), prev_half));
    v128_or(r, s)
}

#[inline(always)]
fn sign4(v: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let pos = v128_and(f32x4_gt(v, zero), f32x4_splat(1.0));
    let neg = v128_and(f32x4_lt(v, zero), f32x4_splat(-1.0));
    v128_or(pos, neg)
}

#[inline]
fn run_unary_exact(x: &[f32], out: &mut [f32], vop: impl Fn(v128) -> v128, sop: impl Fn(f32) -> f32) {
    debug_assert_eq!(x.len(), out.len());
    let (xp, xr) = x.as_chunks::<8>();
    let (op8, or) = out.as_chunks_mut::<8>();
    for (xc, oc) in xp.iter().zip(op8.iter_mut()) {
        // SAFETY: `xc` and `oc` are 8 floats.
        unsafe {
            let sp = xc.as_ptr();
            let dp = oc.as_mut_ptr();
            v128_store(dp as *mut v128, vop(v128_load(sp as *const v128)));
            v128_store(dp.add(4) as *mut v128, vop(v128_load(sp.add(4) as *const v128)));
        }
    }
    let (x4, xt) = xr.as_chunks::<4>();
    let (o4, ot) = or.as_chunks_mut::<4>();
    for (xc, oc) in x4.iter().zip(o4.iter_mut()) {
        // SAFETY: `xc` and `oc` are 4 floats.
        unsafe {
            v128_store(oc.as_mut_ptr() as *mut v128, vop(v128_load(xc.as_ptr() as *const v128)));
        }
    }
    for (o, &s) in ot.iter_mut().zip(xt) {
        *o = sop(s);
    }
}

#[inline]
fn map1_wasm(
    x: &[f32],
    lx: &Layout,
    vop: impl Fn(v128) -> v128 + Sync,
    sop: impl Fn(f32) -> f32 + Sync,
) -> Vec<f32> {
    let n = lx.numel();
    let mut out = alloc_uninit(n);
    if n == 0 {
        return out;
    }
    if let Some(r) = lx.contiguous_range() {
        let x = &x[r];
        let piece =
            |c: usize, o: &mut [f32]| run_unary_exact(&x[c * CHUNK..c * CHUNK + o.len()], o, &vop, &sop);
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan = RowPlan::<1>::new(lx.dims(), [lx.strides()], [lx.offset()]);
    let s = plan.inner[0];
    fill_rows(&mut out, &plan, |row, [o]| {
        if s == 1 {
            run_unary_exact(&x[o..o + row.len()], row, &vop, &sop);
        } else if s == 0 {
            row.fill(sop(x[o]));
        } else {
            for (i, r) in row.iter_mut().enumerate() {
                *r = sop(x[o + i * s]);
            }
        }
    });
    out
}

#[inline]
fn map2_wasm(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    vop: impl Fn(v128, v128) -> v128 + Sync,
    sop: impl Fn(f32, f32) -> f32 + Sync,
) -> Vec<f32> {
    let n = la.numel();
    let mut out = alloc_uninit(n);
    if n == 0 {
        return out;
    }
    if let (Some(ra), Some(rb)) = (la.contiguous_range(), lb.contiguous_range()) {
        let (a, b) = (&a[ra], &b[rb]);
        let piece = |c: usize, o: &mut [f32]| {
            let r = c * CHUNK..c * CHUNK + o.len();
            run_bin_vv(&a[r.clone()], &b[r], o, &vop, &sop);
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan = RowPlan::<2>::new(la.dims(), [la.strides(), lb.strides()], [la.offset(), lb.offset()]);
    let [sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oa, ob]| {
        let len = row.len();
        match (sa, sb) {
            (1, 1) => run_bin_vv(&a[oa..oa + len], &b[ob..ob + len], row, &vop, &sop),
            (1, 0) => run_bin_vs(&a[oa..oa + len], b[ob], row, &vop, &sop),
            (0, 1) => run_bin_sv(a[oa], &b[ob..ob + len], row, &vop, &sop),
            (0, 0) => row.fill(sop(a[oa], b[ob])),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = sop(a[oa + i * sa], b[ob + i * sb]);
                }
            }
        }
    });
    out
}

/// Wasm SIMD128 unary forward dispatch for activations, arithmetic/math, and scalar ops.
pub(super) fn try_unary_wasm(op: UnaryOp, x: &[f32], lx: &Layout) -> Option<Vec<f32>> {
    use UnaryOp::*;
    Some(match op {
        Neg => map1_wasm(x, lx, |v| f32x4_neg(v), |s| -s),
        Abs => map1_wasm(x, lx, |v| f32x4_abs(v), f32::abs),
        Sqrt | PowScalar(0.5) => map1_wasm(x, lx, |v| f32x4_sqrt(v), f32::sqrt),
        Rsqrt => map1_wasm(x, lx, |v| f32x4_div(f32x4_splat(1.0), f32x4_sqrt(v)), |s| 1.0 / s.sqrt()),
        Recip => map1_wasm(x, lx, |v| f32x4_div(f32x4_splat(1.0), v), |s| 1.0 / s),
        Square | PowScalar(2.0) => map1_wasm(x, lx, |v| f32x4_mul(v, v), |s| s * s),
        Relu => map1_wasm(x, lx, |v| f32x4_pmax(f32x4_splat(0.0), v), |s| max_num(s, 0.0)),
        LeakyRelu(a) => {
            let av = f32x4_splat(a);
            let zero = f32x4_splat(0.0);
            map1_wasm(
                x,
                lx,
                move |v| v128_bitselect(v, f32x4_mul(av, v), f32x4_gt(v, zero)),
                move |s| {
                    if s > 0.0 { s } else { a * s }
                },
            )
        }
        Hardswish => map1_wasm(x, lx, hardswish4, hardswish_scalar),
        Clamp(lo, hi) => {
            let (vlo, vhi) = (f32x4_splat(lo), f32x4_splat(hi));
            if !lo.is_nan() && !hi.is_nan() {
                map1_wasm(
                    x,
                    lx,
                    move |v| f32x4_pmin(vhi, f32x4_pmax(vlo, v)),
                    move |s| min_num(max_num(s, lo), hi),
                )
            } else {
                map1_wasm(
                    x,
                    lx,
                    move |v| min4_exact(max4_exact(v, vlo), vhi),
                    move |s| min_num(max_num(s, lo), hi),
                )
            }
        }
        AddScalar(c) => {
            let cv = f32x4_splat(c);
            map1_wasm(x, lx, move |v| f32x4_add(v, cv), move |s| s + c)
        }
        MulScalar(c) => {
            let cv = f32x4_splat(c);
            map1_wasm(x, lx, move |v| f32x4_mul(v, cv), move |s| s * c)
        }
        RSubScalar(c) => {
            let cv = f32x4_splat(c);
            map1_wasm(x, lx, move |v| f32x4_sub(cv, v), move |s| c - s)
        }
        RDivScalar(c) => {
            let cv = f32x4_splat(c);
            map1_wasm(x, lx, move |v| f32x4_div(cv, v), move |s| c / s)
        }
        DivScalar(c) => {
            let cv = f32x4_splat(c);
            map1_wasm(x, lx, move |v| f32x4_div(v, cv), move |s| s / c)
        }
        MaxScalar(c) => {
            let cv = f32x4_splat(c);
            if !c.is_nan() {
                map1_wasm(x, lx, move |v| f32x4_pmax(cv, v), move |s| max_num(s, c))
            } else {
                map1_wasm(x, lx, move |v| max4_exact(v, cv), move |s| max_num(s, c))
            }
        }
        MinScalar(c) => {
            let cv = f32x4_splat(c);
            if !c.is_nan() {
                map1_wasm(x, lx, move |v| f32x4_pmin(cv, v), move |s| min_num(s, c))
            } else {
                map1_wasm(x, lx, move |v| min4_exact(v, cv), move |s| min_num(s, c))
            }
        }
        Floor => map1_wasm(x, lx, |v| f32x4_floor(v), f32::floor),
        Ceil => map1_wasm(x, lx, |v| f32x4_ceil(v), f32::ceil),
        Round => map1_wasm(x, lx, round_away4, f32::round),
        Sign => map1_wasm(x, lx, sign4, sign),
        _ => return None,
    })
}

/// Wasm SIMD128 unary backward dispatch.
pub(super) fn try_unary_backward_wasm(
    op: UnaryOp,
    x: &[f32],
    lx: &Layout,
    g: &[f32],
    lg: &Layout,
) -> Option<Vec<f32>> {
    use UnaryOp::*;
    Some(match op {
        Neg | RSubScalar(_) => map2_wasm(x, lx, g, lg, |_, gv| f32x4_neg(gv), |_, gs| -gs),
        Abs => map2_wasm(x, lx, g, lg, |xv, gv| f32x4_mul(gv, sign4(xv)), |xs, gs| gs * sign(xs)),
        AddScalar(_) => map2_wasm(x, lx, g, lg, |_, gv| gv, |_, gs| gs),
        MulScalar(c) => {
            let cv = f32x4_splat(c);
            map2_wasm(x, lx, g, lg, move |_, gv| f32x4_mul(gv, cv), move |_, gs| gs * c)
        }
        DivScalar(c) => {
            let cv = f32x4_splat(c);
            map2_wasm(x, lx, g, lg, move |_, gv| f32x4_div(gv, cv), move |_, gs| gs / c)
        }
        RDivScalar(c) => {
            let cv = f32x4_splat(c);
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| f32x4_div(f32x4_neg(f32x4_mul(gv, f32x4_div(cv, xv))), xv),
                move |xs, gs| -(gs * (c / xs)) / xs,
            )
        }
        Sqrt => map2_wasm(
            x,
            lx,
            g,
            lg,
            |xv, gv| f32x4_div(f32x4_mul(gv, f32x4_splat(0.5)), f32x4_sqrt(xv)),
            |xs, gs| gs * 0.5 / xs.sqrt(),
        ),
        Rsqrt => map2_wasm(
            x,
            lx,
            g,
            lg,
            |xv, gv| f32x4_div(f32x4_div(f32x4_mul(f32x4_splat(-0.5), gv), f32x4_sqrt(xv)), xv),
            |xs, gs| -0.5 * gs / xs.sqrt() / xs,
        ),
        Recip => map2_wasm(
            x,
            lx,
            g,
            lg,
            |xv, gv| f32x4_div(f32x4_neg(f32x4_div(gv, xv)), xv),
            |xs, gs| -(gs / xs) / xs,
        ),
        Square => map2_wasm(
            x,
            lx,
            g,
            lg,
            |xv, gv| f32x4_mul(f32x4_mul(f32x4_splat(2.0), xv), gv),
            |xs, gs| 2.0 * xs * gs,
        ),
        Relu => {
            let zero = f32x4_splat(0.0);
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| v128_and(gv, f32x4_gt(xv, zero)),
                |xs, gs| {
                    if xs > 0.0 { gs } else { 0.0 }
                },
            )
        }
        LeakyRelu(a) => {
            let av = f32x4_splat(a);
            let zero = f32x4_splat(0.0);
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| v128_bitselect(gv, f32x4_mul(av, gv), f32x4_gt(xv, zero)),
                move |xs, gs| if xs > 0.0 { gs } else { a * gs },
            )
        }
        Hardswish => map2_wasm(x, lx, g, lg, hardswish_bwd4, hardswish_bwd_scalar),
        Clamp(lo, hi) => {
            let (vlo, vhi) = (f32x4_splat(lo), f32x4_splat(hi));
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| v128_and(gv, v128_and(f32x4_ge(xv, vlo), f32x4_le(xv, vhi))),
                move |xs, gs| if xs >= lo && xs <= hi { gs } else { 0.0 },
            )
        }
        MaxScalar(c) => {
            let cv = f32x4_splat(c);
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| v128_and(gv, f32x4_ge(xv, cv)),
                move |xs, gs| {
                    if xs >= c { gs } else { 0.0 }
                },
            )
        }
        MinScalar(c) => {
            let cv = f32x4_splat(c);
            map2_wasm(
                x,
                lx,
                g,
                lg,
                move |xv, gv| v128_and(gv, f32x4_le(xv, cv)),
                move |xs, gs| {
                    if xs <= c { gs } else { 0.0 }
                },
            )
        }
        _ => return None,
    })
}

/// Wasm SIMD128 fused `v[i] = op(v[i] + bias)` for `None`, `Relu`, `LeakyRelu`, and `Hardswish`.
pub(super) fn try_bias_act_inplace_wasm(op: Option<UnaryOp>, bias: f32, v: &mut [f32]) -> bool {
    #[inline]
    fn run(v: &mut [f32], bias: f32, vop: impl Fn(v128) -> v128, sop: impl Fn(f32) -> f32) {
        let bv = f32x4_splat(bias);
        let (pairs, rem) = v.as_chunks_mut::<8>();
        for c in pairs {
            // SAFETY: `c` is 8 floats.
            unsafe {
                let p = c.as_mut_ptr();
                let x0 = f32x4_add(v128_load(p as *const v128), bv);
                let x1 = f32x4_add(v128_load(p.add(4) as *const v128), bv);
                v128_store(p as *mut v128, vop(x0));
                v128_store(p.add(4) as *mut v128, vop(x1));
            }
        }
        let (c4, tail) = rem.as_chunks_mut::<4>();
        for c in c4 {
            // SAFETY: `c` is 4 floats.
            unsafe {
                let p = c.as_mut_ptr();
                v128_store(p as *mut v128, vop(f32x4_add(v128_load(p as *const v128), bv)));
            }
        }
        for slot in tail {
            *slot = sop(*slot + bias);
        }
    }
    match op {
        None => run(v, bias, |x| x, |s| s),
        Some(UnaryOp::Relu) => {
            let zero = f32x4_splat(0.0);
            run(v, bias, move |x| f32x4_pmax(zero, x), |s| max_num(s, 0.0));
        }
        Some(UnaryOp::LeakyRelu(a)) => {
            let zero = f32x4_splat(0.0);
            let av = f32x4_splat(a);
            run(
                v,
                bias,
                move |x| v128_bitselect(x, f32x4_mul(av, x), f32x4_gt(x, zero)),
                move |s| {
                    if s > 0.0 { s } else { a * s }
                },
            );
        }
        Some(UnaryOp::Hardswish) => run(v, bias, hardswish4, hardswish_scalar),
        _ => return false,
    }
    true
}

#[inline]
fn where3_f32_slice(c: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    let zero = f32x4_splat(0.0);
    let (cc, cr) = c.as_chunks::<4>();
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    let (oc, or) = out.as_chunks_mut::<4>();
    for (((cv, av), bv), ov) in cc.iter().zip(ac).zip(bc).zip(oc.iter_mut()) {
        // SAFETY: all four chunks are 4 floats.
        unsafe {
            let m = f32x4_ne(v128_load(cv.as_ptr() as *const v128), zero);
            let r = v128_bitselect(
                v128_load(av.as_ptr() as *const v128),
                v128_load(bv.as_ptr() as *const v128),
                m,
            );
            v128_store(ov.as_mut_ptr() as *mut v128, r);
        }
    }
    for (((o, &cv), &av), &bv) in or.iter_mut().zip(cr).zip(ar).zip(br) {
        *o = if cv != 0.0 { av } else { bv };
    }
}

/// Wasm SIMD128 `cond != 0.0 ? a : b` for `F32` conditions.
pub(super) fn where3_f32_wasm(
    c: &[f32],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
) -> Vec<f32> {
    let n = la.numel();
    if n == 0 {
        return Vec::new();
    }
    if let (Some(rc), Some(ra), Some(rb)) =
        (lc.contiguous_range(), la.contiguous_range(), lb.contiguous_range())
    {
        let mut out = alloc_uninit(n);
        let (c, a, b) = (&c[rc], &a[ra], &b[rb]);
        let piece = |k: usize, o: &mut [f32]| {
            let r = k * CHUNK..k * CHUNK + o.len();
            where3_f32_slice(&c[r.clone()], &a[r.clone()], &b[r], o);
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let mut out = alloc_uninit(n);
    let plan = RowPlan::<3>::new(
        la.dims(),
        [lc.strides(), la.strides(), lb.strides()],
        [lc.offset(), la.offset(), lb.offset()],
    );
    let [sc, sa, sb] = plan.inner;
    let zero = f32x4_splat(0.0);
    fill_rows(&mut out, &plan, |row, [oc, oa, ob]| {
        let len = row.len();
        match (sc, sa, sb) {
            (1, 1, 1) => where3_f32_slice(&c[oc..oc + len], &a[oa..oa + len], &b[ob..ob + len], row),
            (1, 1, 0) => {
                let bv_s = b[ob];
                let bv = f32x4_splat(bv_s);
                let (cc, cr) = c[oc..oc + len].as_chunks::<4>();
                let (ac, ar) = a[oa..oa + len].as_chunks::<4>();
                let (rc, rr) = row.as_chunks_mut::<4>();
                for ((cv, av), ov) in cc.iter().zip(ac).zip(rc.iter_mut()) {
                    // SAFETY: all chunks are 4 floats.
                    unsafe {
                        let m = f32x4_ne(v128_load(cv.as_ptr() as *const v128), zero);
                        v128_store(
                            ov.as_mut_ptr() as *mut v128,
                            v128_bitselect(v128_load(av.as_ptr() as *const v128), bv, m),
                        );
                    }
                }
                for ((o, &cv), &av) in rr.iter_mut().zip(cr).zip(ar) {
                    *o = if cv != 0.0 { av } else { bv_s };
                }
            }
            (1, 0, 1) => {
                let av_s = a[oa];
                let av = f32x4_splat(av_s);
                let (cc, cr) = c[oc..oc + len].as_chunks::<4>();
                let (bc, br) = b[ob..ob + len].as_chunks::<4>();
                let (rc, rr) = row.as_chunks_mut::<4>();
                for ((cv, bv), ov) in cc.iter().zip(bc).zip(rc.iter_mut()) {
                    // SAFETY: all chunks are 4 floats.
                    unsafe {
                        let m = f32x4_ne(v128_load(cv.as_ptr() as *const v128), zero);
                        v128_store(
                            ov.as_mut_ptr() as *mut v128,
                            v128_bitselect(av, v128_load(bv.as_ptr() as *const v128), m),
                        );
                    }
                }
                for ((o, &cv), &bv) in rr.iter_mut().zip(cr).zip(br) {
                    *o = if cv != 0.0 { av_s } else { bv };
                }
            }
            (1, 0, 0) => {
                let (av_s, bv_s) = (a[oa], b[ob]);
                let (av, bv) = (f32x4_splat(av_s), f32x4_splat(bv_s));
                let (cc, cr) = c[oc..oc + len].as_chunks::<4>();
                let (rc, rr) = row.as_chunks_mut::<4>();
                for (cv, ov) in cc.iter().zip(rc.iter_mut()) {
                    // SAFETY: all chunks are 4 floats.
                    unsafe {
                        let m = f32x4_ne(v128_load(cv.as_ptr() as *const v128), zero);
                        v128_store(ov.as_mut_ptr() as *mut v128, v128_bitselect(av, bv, m));
                    }
                }
                for (o, &cv) in rr.iter_mut().zip(cr) {
                    *o = if cv != 0.0 { av_s } else { bv_s };
                }
            }
            (0, 1, 1) => {
                if c[oc] != 0.0 {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 1, 0) => {
                if c[oc] != 0.0 {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.fill(b[ob]);
                }
            }
            (0, 0, 1) => {
                if c[oc] != 0.0 {
                    row.fill(a[oa]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 0, 0) => row.fill(if c[oc] != 0.0 { a[oa] } else { b[ob] }),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = if c[oc + i * sc] != 0.0 { a[oa + i * sa] } else { b[ob + i * sb] };
                }
            }
        }
    });
    out
}

#[inline]
fn where3_u8_slice(c: &[u8], a: &[f32], b: &[f32], out: &mut [f32]) {
    let zero_u = u8x16_splat(0);
    let (c16, cr) = c.as_chunks::<16>();
    let (a16, ar) = a.as_chunks::<16>();
    let (b16, br) = b.as_chunks::<16>();
    let (o16, or) = out.as_chunks_mut::<16>();
    for (((cv, av), bv), ov) in c16.iter().zip(a16).zip(b16).zip(o16.iter_mut()) {
        // SAFETY: `cv` is 16 bytes; `av`, `bv`, `ov` are 16 floats.
        unsafe {
            let mb = u8x16_ne(v128_load(cv.as_ptr() as *const v128), zero_u);
            let mlo16 = i16x8_extend_low_i8x16(mb);
            let mhi16 = i16x8_extend_high_i8x16(mb);
            let m0 = i32x4_extend_low_i16x8(mlo16);
            let m1 = i32x4_extend_high_i16x8(mlo16);
            let m2 = i32x4_extend_low_i16x8(mhi16);
            let m3 = i32x4_extend_high_i16x8(mhi16);
            let (ap, bp, dp) = (av.as_ptr(), bv.as_ptr(), ov.as_mut_ptr());
            v128_store(
                dp as *mut v128,
                v128_bitselect(v128_load(ap as *const v128), v128_load(bp as *const v128), m0),
            );
            v128_store(
                dp.add(4) as *mut v128,
                v128_bitselect(v128_load(ap.add(4) as *const v128), v128_load(bp.add(4) as *const v128), m1),
            );
            v128_store(
                dp.add(8) as *mut v128,
                v128_bitselect(v128_load(ap.add(8) as *const v128), v128_load(bp.add(8) as *const v128), m2),
            );
            v128_store(
                dp.add(12) as *mut v128,
                v128_bitselect(
                    v128_load(ap.add(12) as *const v128),
                    v128_load(bp.add(12) as *const v128),
                    m3,
                ),
            );
        }
    }
    for (((o, &cv), &av), &bv) in or.iter_mut().zip(cr).zip(ar).zip(br) {
        *o = if cv != 0 { av } else { bv };
    }
}

/// Wasm SIMD128 `cond != 0 ? a : b` for `U8` conditions.
pub(super) fn where3_u8_wasm(
    c: &[u8],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
) -> Vec<f32> {
    let n = la.numel();
    if n == 0 {
        return Vec::new();
    }
    if let (Some(rc), Some(ra), Some(rb)) =
        (lc.contiguous_range(), la.contiguous_range(), lb.contiguous_range())
    {
        let mut out = alloc_uninit(n);
        let (c, a, b) = (&c[rc], &a[ra], &b[rb]);
        let piece = |k: usize, o: &mut [f32]| {
            let r = k * CHUNK..k * CHUNK + o.len();
            where3_u8_slice(&c[r.clone()], &a[r.clone()], &b[r], o);
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let mut out = alloc_uninit(n);
    let plan = RowPlan::<3>::new(
        la.dims(),
        [lc.strides(), la.strides(), lb.strides()],
        [lc.offset(), la.offset(), lb.offset()],
    );
    let [sc, sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oc, oa, ob]| {
        let len = row.len();
        match (sc, sa, sb) {
            (1, 1, 1) => where3_u8_slice(&c[oc..oc + len], &a[oa..oa + len], &b[ob..ob + len], row),
            (0, 1, 1) => {
                if c[oc] != 0 {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 1, 0) => {
                if c[oc] != 0 {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.fill(b[ob]);
                }
            }
            (0, 0, 1) => {
                if c[oc] != 0 {
                    row.fill(a[oa]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 0, 0) => row.fill(if c[oc] != 0 { a[oa] } else { b[ob] }),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = if c[oc + i * sc] != 0 { a[oa + i * sa] } else { b[ob + i * sb] };
                }
            }
        }
    });
    out
}
