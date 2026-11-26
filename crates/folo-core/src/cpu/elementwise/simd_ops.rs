//! AVX2 vectorised activations, arithmetic/math unary ops, unary backwards, and fused bias+activation epilogues.
//!
//! Every kernel here is bitwise identical to its IEEE-754 scalar counterpart on all non-NaN inputs (including
//! `±0.0`, subnormals, and `±∞`) and preserves NaN semantics identically across contiguous, strided, broadcast,
//! and parallel chunked layouts.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::simd::has_avx2;
use super::simd_binary::{map2_avx2, max8_exact, min8_exact};
use super::{fill_rows, max_num, min_num, sign};
use crate::Layout;
use crate::backend::UnaryOp;
use crate::cpu::pool_alloc::alloc_uninit;
use crate::cpu::{CHUNK, PAR_MIN, par_chunks};
use crate::layout::RowPlan;

/// True when `_mm256_max_ps(v, _mm256_set1_ps(c))` / `_mm256_min_ps` is bitwise identical to
/// `v.max(c)` / `v.min(c)` for every `v` (i.e. `c` is not NaN).
#[inline(always)]
fn fast_minmax_scalar(c: f32) -> bool {
    !c.is_nan()
}

#[inline(always)]
fn hardswish_scalar(v: f32) -> f32 {
    v * (v + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)
}

#[target_feature(enable = "avx2")]
#[inline]
fn hardswish8(v: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let c3 = _mm256_set1_ps(3.0);
    let c6 = _mm256_set1_ps(6.0);
    let inv6 = _mm256_set1_ps(1.0 / 6.0);
    let clamped = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(v, c3), zero), c6);
    _mm256_mul_ps(_mm256_mul_ps(v, clamped), inv6)
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

#[target_feature(enable = "avx2")]
#[inline]
fn hardswish_bwd8(x: __m256, g: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let two = _mm256_set1_ps(2.0);
    let c3 = _mm256_set1_ps(3.0);
    let neg3 = _mm256_set1_ps(-3.0);
    let inv6 = _mm256_set1_ps(1.0 / 6.0);
    let mid = _mm256_mul_ps(_mm256_mul_ps(g, _mm256_add_ps(_mm256_mul_ps(two, x), c3)), inv6);
    let r = _mm256_blendv_ps(mid, g, _mm256_cmp_ps::<_CMP_GT_OQ>(x, c3));
    _mm256_blendv_ps(r, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(x, neg3))
}

/// Exact round-half-away-from-zero (`f32::round`) across all 2^32 binary32 values:
/// `copysign(trunc(|v| + (0.5 - 2^-25)), v)`.
#[target_feature(enable = "avx2")]
#[inline]
fn round_away8(v: __m256) -> __m256 {
    let sign_bit = _mm256_set1_ps(-0.0);
    let prev_half = _mm256_set1_ps(f32::from_bits(0x3eff_ffff));
    let s = _mm256_and_ps(v, sign_bit);
    let abs_v = _mm256_andnot_ps(sign_bit, v);
    let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm256_add_ps(abs_v, prev_half));
    _mm256_or_ps(r, s)
}

#[target_feature(enable = "avx2")]
#[inline]
fn sign8(v: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let pos = _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(v, zero), _mm256_set1_ps(1.0));
    let neg = _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(v, zero), _mm256_set1_ps(-1.0));
    _mm256_or_ps(pos, neg)
}

macro_rules! avx2_unary_loop {
    ($x:ident, $out:ident, $v:ident => $vexpr:expr, $s:ident => $sexpr:expr) => {{
        debug_assert_eq!($x.len(), $out.len());
        let (xp, xr) = $x.as_chunks::<16>();
        let (op, or) = $out.as_chunks_mut::<16>();
        for (xc, oc) in xp.iter().zip(op.iter_mut()) {
            // SAFETY: `xc` and `oc` are 16 floats; unaligned AVX2 loads/stores.
            unsafe {
                let sp = xc.as_ptr();
                let dp = oc.as_mut_ptr();
                let $v = _mm256_loadu_ps(sp);
                let r0 = $vexpr;
                let $v = _mm256_loadu_ps(sp.add(8));
                let r1 = $vexpr;
                _mm256_storeu_ps(dp, r0);
                _mm256_storeu_ps(dp.add(8), r1);
            }
        }
        let (x8, xt) = xr.as_chunks::<8>();
        let (o8, ot) = or.as_chunks_mut::<8>();
        for (xc, oc) in x8.iter().zip(o8.iter_mut()) {
            // SAFETY: `xc` and `oc` are 8 floats; unaligned AVX2 loads/stores.
            unsafe {
                let $v = _mm256_loadu_ps(xc.as_ptr());
                _mm256_storeu_ps(oc.as_mut_ptr(), $vexpr);
            }
        }
        for (o, &$s) in ot.iter_mut().zip(xt) {
            *o = $sexpr;
        }
    }};
}

macro_rules! avx2_bias_inplace_loop {
    ($v:ident, $bias:ident, $vx:ident => $vexpr:expr, $sx:ident => $sexpr:expr) => {{
        let b = _mm256_set1_ps($bias);
        let (pairs, rem) = $v.as_chunks_mut::<16>();
        for c in pairs {
            // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
            unsafe {
                let p = c.as_mut_ptr();
                let $vx = _mm256_add_ps(_mm256_loadu_ps(p), b);
                let r0 = $vexpr;
                let $vx = _mm256_add_ps(_mm256_loadu_ps(p.add(8)), b);
                let r1 = $vexpr;
                _mm256_storeu_ps(p, r0);
                _mm256_storeu_ps(p.add(8), r1);
            }
        }
        let (chunks, tail) = rem.as_chunks_mut::<8>();
        for c in chunks {
            // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
            unsafe {
                let $vx = _mm256_add_ps(_mm256_loadu_ps(c.as_ptr()), b);
                _mm256_storeu_ps(c.as_mut_ptr(), $vexpr);
            }
        }
        for slot in tail {
            let $sx = *slot + $bias;
            *slot = $sexpr;
        }
    }};
}

#[inline]
fn map1_avx2(
    x: &[f32],
    lx: &Layout,
    slice_fn: impl Fn(&[f32], &mut [f32]) + Sync,
    scalar_fn: impl Fn(f32) -> f32 + Sync,
) -> Vec<f32> {
    let n = lx.numel();
    let mut out = alloc_uninit(n);
    if n == 0 {
        return out;
    }
    if let Some(r) = lx.contiguous_range() {
        let x = &x[r];
        let piece = |c: usize, o: &mut [f32]| slice_fn(&x[c * CHUNK..c * CHUNK + o.len()], o);
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
            slice_fn(&x[o..o + row.len()], row);
        } else if s == 0 {
            row.fill(scalar_fn(x[o]));
        } else {
            for (i, r) in row.iter_mut().enumerate() {
                *r = scalar_fn(x[o + i * s]);
            }
        }
    });
    out
}

/// AVX2 unary forward dispatch. Returns `None` if AVX2 is unavailable or `op` is handled elsewhere.
pub(super) fn try_unary_avx2(op: UnaryOp, x: &[f32], lx: &Layout) -> Option<Vec<f32>> {
    if !has_avx2() {
        return None;
    }
    macro_rules! go {
        ($v:ident => $vexpr:expr, $s:ident => $sexpr:expr) => {{
            #[target_feature(enable = "avx2")]
            unsafe fn slice(src: &[f32], dst: &mut [f32]) {
                avx2_unary_loop!(src, dst, $v => $vexpr, $s => $sexpr);
            }
            // SAFETY: `has_avx2()` checked above.
            Some(map1_avx2(x, lx, |s, o| unsafe { slice(s, o) }, |$s| $sexpr))
        }};
    }
    macro_rules! go_p1 {
        ($p:ident, $vp:ident, $v:ident => $vexpr:expr, $s:ident => $sexpr:expr) => {{
            #[target_feature(enable = "avx2")]
            unsafe fn slice(src: &[f32], dst: &mut [f32], $p: f32) {
                let $vp = _mm256_set1_ps($p);
                avx2_unary_loop!(src, dst, $v => $vexpr, $s => $sexpr);
            }
            // SAFETY: `has_avx2()` checked above.
            Some(map1_avx2(x, lx, |s, o| unsafe { slice(s, o, $p) }, |$s| $sexpr))
        }};
    }
    use UnaryOp::*;
    match op {
        Neg => go!(v => _mm256_xor_ps(v, _mm256_set1_ps(-0.0)), s => -s),
        Abs => go!(v => _mm256_andnot_ps(_mm256_set1_ps(-0.0), v), s => s.abs()),
        Sqrt => go!(v => _mm256_sqrt_ps(v), s => s.sqrt()),
        Rsqrt => go!(v => _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(v)), s => 1.0 / s.sqrt()),
        Recip => go!(v => _mm256_div_ps(_mm256_set1_ps(1.0), v), s => 1.0 / s),
        Square => go!(v => _mm256_mul_ps(v, v), s => s * s),
        Relu => go!(v => _mm256_max_ps(v, _mm256_setzero_ps()), s => max_num(s, 0.0)),
        LeakyRelu(a) => go_p1!(
            a, av,
            v => _mm256_blendv_ps(_mm256_mul_ps(av, v), v, _mm256_cmp_ps::<_CMP_GT_OQ>(v, _mm256_setzero_ps())),
            s => if s > 0.0 { s } else { a * s }
        ),
        Hardswish => go!(v => hardswish8(v), s => hardswish_scalar(s)),
        Clamp(lo, hi) => {
            if fast_minmax_scalar(lo) && fast_minmax_scalar(hi) {
                #[target_feature(enable = "avx2")]
                unsafe fn slice(src: &[f32], dst: &mut [f32], lo: f32, hi: f32) {
                    let (vlo, vhi) = (_mm256_set1_ps(lo), _mm256_set1_ps(hi));
                    avx2_unary_loop!(src, dst, v => _mm256_min_ps(_mm256_max_ps(v, vlo), vhi), s => min_num(max_num(s, lo), hi));
                }
                // SAFETY: `has_avx2()` checked above.
                Some(map1_avx2(x, lx, |s, o| unsafe { slice(s, o, lo, hi) }, |s| min_num(max_num(s, lo), hi)))
            } else {
                #[target_feature(enable = "avx2")]
                unsafe fn slice(src: &[f32], dst: &mut [f32], lo: f32, hi: f32) {
                    let (vlo, vhi) = (_mm256_set1_ps(lo), _mm256_set1_ps(hi));
                    avx2_unary_loop!(src, dst, v => min8_exact(max8_exact(v, vlo), vhi), s => min_num(max_num(s, lo), hi));
                }
                // SAFETY: `has_avx2()` checked above.
                Some(map1_avx2(x, lx, |s, o| unsafe { slice(s, o, lo, hi) }, |s| min_num(max_num(s, lo), hi)))
            }
        }
        PowScalar(2.0) => go!(v => _mm256_mul_ps(v, v), s => s * s),
        PowScalar(0.5) => go!(v => _mm256_sqrt_ps(v), s => s.sqrt()),
        AddScalar(c) => go_p1!(c, cv, v => _mm256_add_ps(v, cv), s => s + c),
        MulScalar(c) => go_p1!(c, cv, v => _mm256_mul_ps(v, cv), s => s * c),
        RSubScalar(c) => go_p1!(c, cv, v => _mm256_sub_ps(cv, v), s => c - s),
        RDivScalar(c) => go_p1!(c, cv, v => _mm256_div_ps(cv, v), s => c / s),
        DivScalar(c) => go_p1!(c, cv, v => _mm256_div_ps(v, cv), s => s / c),
        MaxScalar(c) => {
            if fast_minmax_scalar(c) {
                go_p1!(c, cv, v => _mm256_max_ps(v, cv), s => max_num(s, c))
            } else {
                go_p1!(c, cv, v => max8_exact(v, cv), s => max_num(s, c))
            }
        }
        MinScalar(c) => {
            if fast_minmax_scalar(c) {
                go_p1!(c, cv, v => _mm256_min_ps(v, cv), s => min_num(s, c))
            } else {
                go_p1!(c, cv, v => min8_exact(v, cv), s => min_num(s, c))
            }
        }
        Floor => go!(v => _mm256_floor_ps(v), s => s.floor()),
        Ceil => go!(v => _mm256_ceil_ps(v), s => s.ceil()),
        Round => go!(v => round_away8(v), s => s.round()),
        Sign => go!(v => sign8(v), s => sign(s)),
        _ => None,
    }
}

/// AVX2 unary backward dispatch. Returns `None` if AVX2 is unavailable or `op` is handled elsewhere.
pub(super) fn try_unary_backward_avx2(
    op: UnaryOp,
    x: &[f32],
    lx: &Layout,
    g: &[f32],
    lg: &Layout,
) -> Option<Vec<f32>> {
    if !has_avx2() {
        return None;
    }
    macro_rules! go2 {
        ($vx:ident, $vg:ident => $vexpr:expr; $sx:ident, $sg:ident => $sexpr:expr) => {{
            let (vv, vs, sv) = super::simd_binary::avx2_bin_loops!($vx, $vg => $vexpr; $sx, $sg => $sexpr);
            // SAFETY: `has_avx2()` checked above.
            Some(map2_avx2(
                x, lx, g, lg,
                |a, b, o| unsafe { vv(a, b, o) },
                |a, b, o| unsafe { vs(a, b, o) },
                |a, b, o| unsafe { sv(a, b, o) },
                |$sx, $sg| $sexpr,
            ))
        }};
    }
    macro_rules! go2_p1 {
        ($p:ident, $vp:ident, $vx:ident, $vg:ident => $vexpr:expr; $sx:ident, $sg:ident => $sexpr:expr) => {{
            let (vv, vs, sv) = super::simd_binary::avx2_bin_loops!(
                $vx, $vg => $vexpr; $sx, $sg => $sexpr; param $p, $vp
            );
            // SAFETY: `has_avx2()` checked above.
            Some(map2_avx2(
                x, lx, g, lg,
                |a, b, o| unsafe { vv(a, b, o, $p) },
                |a, b, o| unsafe { vs(a, b, o, $p) },
                |a, b, o| unsafe { sv(a, b, o, $p) },
                |$sx, $sg| $sexpr,
            ))
        }};
    }
    use UnaryOp::*;
    match op {
        Neg | RSubScalar(_) => go2!(_vx, vg => _mm256_xor_ps(vg, _mm256_set1_ps(-0.0)); _sx, sg => -sg),
        Abs => go2!(vx, vg => _mm256_mul_ps(vg, sign8(vx)); sx, sg => sg * sign(sx)),
        AddScalar(_) => go2!(_vx, vg => vg; _sx, sg => sg),
        MulScalar(c) => go2_p1!(c, cv, _vx, vg => _mm256_mul_ps(vg, cv); _sx, sg => sg * c),
        DivScalar(c) => go2_p1!(c, cv, _vx, vg => _mm256_div_ps(vg, cv); _sx, sg => sg / c),
        RDivScalar(c) => go2_p1!(
            c, cv,
            vx, vg => _mm256_div_ps(_mm256_xor_ps(_mm256_mul_ps(vg, _mm256_div_ps(cv, vx)), _mm256_set1_ps(-0.0)), vx);
            sx, sg => -(sg * (c / sx)) / sx
        ),
        Sqrt => go2!(
            vx, vg => _mm256_div_ps(_mm256_mul_ps(vg, _mm256_set1_ps(0.5)), _mm256_sqrt_ps(vx));
            sx, sg => sg * 0.5 / sx.sqrt()
        ),
        Rsqrt => go2!(
            vx, vg => _mm256_div_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(-0.5), vg), _mm256_sqrt_ps(vx)), vx);
            sx, sg => -0.5 * sg / sx.sqrt() / sx
        ),
        Recip => go2!(
            vx, vg => _mm256_div_ps(_mm256_xor_ps(_mm256_div_ps(vg, vx), _mm256_set1_ps(-0.0)), vx);
            sx, sg => -(sg / sx) / sx
        ),
        Square => go2!(
            vx, vg => _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0), vx), vg);
            sx, sg => 2.0 * sx * sg
        ),
        Relu => go2!(
            vx, vg => _mm256_and_ps(vg, _mm256_cmp_ps::<_CMP_GT_OQ>(vx, _mm256_setzero_ps()));
            sx, sg => if sx > 0.0 { sg } else { 0.0 }
        ),
        LeakyRelu(a) => go2_p1!(
            a, av,
            vx, vg => _mm256_blendv_ps(_mm256_mul_ps(av, vg), vg, _mm256_cmp_ps::<_CMP_GT_OQ>(vx, _mm256_setzero_ps()));
            sx, sg => if sx > 0.0 { sg } else { a * sg }
        ),
        Hardswish => go2!(vx, vg => hardswish_bwd8(vx, vg); sx, sg => hardswish_bwd_scalar(sx, sg)),
        // SAFETY: `has_avx2()` checked above.
        Clamp(lo, hi) => Some(unsafe { clamp_bwd_avx2(x, lx, g, lg, lo, hi) }),
        MaxScalar(c) => go2_p1!(
            c, cv,
            vx, vg => _mm256_and_ps(vg, _mm256_cmp_ps::<_CMP_GE_OQ>(vx, cv));
            sx, sg => if sx >= c { sg } else { 0.0 }
        ),
        MinScalar(c) => go2_p1!(
            c, cv,
            vx, vg => _mm256_and_ps(vg, _mm256_cmp_ps::<_CMP_LE_OQ>(vx, cv));
            sx, sg => if sx <= c { sg } else { 0.0 }
        ),
        _ => None,
    }
}

/// # Safety
/// Caller must ensure `has_avx2()` is true.
unsafe fn clamp_bwd_avx2(x: &[f32], lx: &Layout, g: &[f32], lg: &Layout, lo: f32, hi: f32) -> Vec<f32> {
    #[target_feature(enable = "avx2")]
    unsafe fn vv(a: &[f32], b: &[f32], out: &mut [f32], lo: f32, hi: f32) {
        let (vlo, vhi) = (_mm256_set1_ps(lo), _mm256_set1_ps(hi));
        let (ap, ar) = a.as_chunks::<16>();
        let (bp, br) = b.as_chunks::<16>();
        let (op, or) = out.as_chunks_mut::<16>();
        for ((ac, bc), oc) in ap.iter().zip(bp.iter()).zip(op.iter_mut()) {
            // SAFETY: 16-float chunks.
            unsafe {
                let (ap, bp, dp) = (ac.as_ptr(), bc.as_ptr(), oc.as_mut_ptr());
                for off in [0usize, 8] {
                    let xv = _mm256_loadu_ps(ap.add(off));
                    let gv = _mm256_loadu_ps(bp.add(off));
                    let m = _mm256_and_ps(
                        _mm256_cmp_ps::<_CMP_GE_OQ>(xv, vlo),
                        _mm256_cmp_ps::<_CMP_LE_OQ>(xv, vhi),
                    );
                    _mm256_storeu_ps(dp.add(off), _mm256_and_ps(gv, m));
                }
            }
        }
        let (a8, at) = ar.as_chunks::<8>();
        let (b8, bt) = br.as_chunks::<8>();
        let (o8, ot) = or.as_chunks_mut::<8>();
        for ((ac, bc), oc) in a8.iter().zip(b8.iter()).zip(o8.iter_mut()) {
            // SAFETY: 8-float chunks.
            unsafe {
                let xv = _mm256_loadu_ps(ac.as_ptr());
                let gv = _mm256_loadu_ps(bc.as_ptr());
                let m =
                    _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(xv, vlo), _mm256_cmp_ps::<_CMP_LE_OQ>(xv, vhi));
                _mm256_storeu_ps(oc.as_mut_ptr(), _mm256_and_ps(gv, m));
            }
        }
        for ((o, &xv), &gv) in ot.iter_mut().zip(at).zip(bt) {
            *o = if xv >= lo && xv <= hi { gv } else { 0.0 };
        }
    }
    #[target_feature(enable = "avx2")]
    unsafe fn vs(a: &[f32], gv_s: f32, out: &mut [f32], lo: f32, hi: f32) {
        let (vlo, vhi, gv) = (_mm256_set1_ps(lo), _mm256_set1_ps(hi), _mm256_set1_ps(gv_s));
        avx2_unary_loop!(
            a, out,
            xv => _mm256_and_ps(gv, _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(xv, vlo), _mm256_cmp_ps::<_CMP_LE_OQ>(xv, vhi))),
            sx => if sx >= lo && sx <= hi { gv_s } else { 0.0 }
        );
    }
    // SAFETY: callers check `has_avx2()`.
    map2_avx2(
        x,
        lx,
        g,
        lg,
        |a, b, o| unsafe { vv(a, b, o, lo, hi) },
        |a, b, o| unsafe { vs(a, b, o, lo, hi) },
        |av, b, o| {
            if av >= lo && av <= hi {
                o.copy_from_slice(b);
            } else {
                o.fill(0.0);
            }
        },
        |xv, gv| if xv >= lo && xv <= hi { gv } else { 0.0 },
    )
}

/// AVX2 fused `v[i] = op(v[i] + bias)` for `None`, `Relu`, `LeakyRelu`, and `Hardswish`.
pub(super) fn try_bias_act_inplace_avx2(op: Option<UnaryOp>, bias: f32, v: &mut [f32]) -> bool {
    if !has_avx2() {
        return false;
    }
    match op {
        None => {
            #[target_feature(enable = "avx2")]
            unsafe fn go(v: &mut [f32], bias: f32) {
                avx2_bias_inplace_loop!(v, bias, x => x, s => s);
            }
            // SAFETY: `has_avx2()` checked above.
            unsafe { go(v, bias) };
            true
        }
        Some(UnaryOp::Relu) => {
            #[target_feature(enable = "avx2")]
            unsafe fn go(v: &mut [f32], bias: f32) {
                let zero = _mm256_setzero_ps();
                avx2_bias_inplace_loop!(v, bias, x => _mm256_max_ps(x, zero), s => max_num(s, 0.0));
            }
            // SAFETY: `has_avx2()` checked above.
            unsafe { go(v, bias) };
            true
        }
        Some(UnaryOp::LeakyRelu(a)) => {
            #[target_feature(enable = "avx2")]
            unsafe fn go(v: &mut [f32], bias: f32, a: f32) {
                let zero = _mm256_setzero_ps();
                let av = _mm256_set1_ps(a);
                avx2_bias_inplace_loop!(
                    v, bias,
                    x => _mm256_blendv_ps(_mm256_mul_ps(av, x), x, _mm256_cmp_ps::<_CMP_GT_OQ>(x, zero)),
                    s => if s > 0.0 { s } else { a * s }
                );
            }
            // SAFETY: `has_avx2()` checked above.
            unsafe { go(v, bias, a) };
            true
        }
        Some(UnaryOp::Hardswish) => {
            #[target_feature(enable = "avx2")]
            unsafe fn go(v: &mut [f32], bias: f32) {
                avx2_bias_inplace_loop!(v, bias, x => hardswish8(x), s => hardswish_scalar(s));
            }
            // SAFETY: `has_avx2()` checked above.
            unsafe { go(v, bias) };
            true
        }
        _ => false,
    }
}
