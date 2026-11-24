//! Copies, casts and elementwise kernels.

mod simd;

use super::pool_alloc::alloc_uninit;
use super::{CHUNK, PAR_MIN};
use crate::Layout;
use crate::backend::{BinaryOp, UnaryOp};
use crate::cpu::{par_chunks, par_for};
use crate::layout::RowPlan;

pub use simd::SILU_MAX_REL_ERR;
pub(crate) use simd::silu_bias_inplace;
#[cfg(target_arch = "x86_64")]
pub(crate) use simd::silu8;

/// Fill `out` (contiguous, `plan.rows * plan.len` long) row by row, in parallel when large.
#[inline]
pub(crate) fn fill_rows<T: Send, const K: usize>(
    out: &mut [T],
    plan: &RowPlan<K>,
    f: impl Fn(&mut [T], [usize; K]) + Sync,
) {
    let len = plan.len;
    if len == 0 || plan.rows == 0 {
        return;
    }
    // The rows from `start` on into `rows_out` (whole rows), serially. The one walk over the rows every path
    // below shares, so each op compiles it once.
    let run = |start: usize, rows_out: &mut [T]| {
        let end = start + rows_out.len() / len;
        let mut rows = rows_out.chunks_mut(len);
        plan.for_rows(start..end, |_, offs| f(rows.next().unwrap(), offs));
    };
    if out.len() < PAR_MIN {
        return run(0, out);
    }
    if plan.rows == 1 {
        // One long row: split it into chunks with adjusted offsets.
        let (base, inner) = (plan.base, plan.inner);
        return par_chunks(out, CHUNK, &|c, chunk| {
            f(chunk, std::array::from_fn(|k| base[k] + c * CHUNK * inner[k]));
        });
    }
    let rows_per_task = (CHUNK / len).max(1);
    par_chunks(out, rows_per_task * len, &|t, chunk| run(t * rows_per_task, chunk));
}

/// A contiguous copy of the elements of `src` selected by `layout`.
pub(crate) fn copy_strided<T: Copy + Send + Sync + Default>(src: &[T], layout: &Layout) -> Vec<T> {
    if let Some(r) = layout.contiguous_range() {
        return src[r].to_vec();
    }
    copy_strided_into(src, layout, vec![T::default(); layout.numel()])
}

/// [`copy_strided`] into a caller-provided buffer of `layout.numel()` elements.
pub(crate) fn copy_strided_into<T: Copy + Send + Sync>(
    src: &[T],
    layout: &Layout,
    mut out: Vec<T>,
) -> Vec<T> {
    debug_assert_eq!(out.len(), layout.numel());
    if let Some(r) = layout.contiguous_range() {
        out.copy_from_slice(&src[r]);
        return out;
    }
    let plan = RowPlan::<1>::new(layout.dims(), [layout.strides()], [layout.offset()]);
    let s = plan.inner[0];
    fill_rows(&mut out, &plan, |row, [o]| {
        if s == 1 {
            row.copy_from_slice(&src[o..o + row.len()]);
        } else {
            for (i, v) in row.iter_mut().enumerate() {
                *v = src[o + i * s];
            }
        }
    });
    out
}

/// Write `src` (read through `src_layout`) into `dst` through `dst_layout`.
///
/// # Panics
/// If the shapes differ, if either layout reaches outside its buffer, or if `dst_layout` maps two logical
/// elements to the same storage element (the parallel writes below rely on that never happening).
pub(crate) fn copy_into<T: Copy + Send + Sync>(
    dst: &mut [T],
    dst_layout: &Layout,
    src: &[T],
    src_layout: &Layout,
) {
    strided_update(dst, dst_layout, src, src_layout, |_, s| s);
}

/// `dst[i] = f(dst[i], src[i])` through two layouts of the same shape. Same panics as [`copy_into`].
pub(crate) fn fill_rows_inplace(
    dst: &mut [f32],
    dst_layout: &Layout,
    src: &[f32],
    src_layout: &Layout,
    f: impl Fn(f32, f32) -> f32 + Sync,
) {
    strided_update(dst, dst_layout, src, src_layout, f);
}

fn strided_update<T: Copy + Send + Sync>(
    dst: &mut [T],
    dst_layout: &Layout,
    src: &[T],
    src_layout: &Layout,
    f: impl Fn(T, T) -> T + Sync,
) {
    assert_eq!(dst_layout.shape(), src_layout.shape(), "strided write: shape mismatch");
    assert!(dst_layout.fits(dst.len()), "strided write: destination layout out of bounds");
    assert!(src_layout.fits(src.len()), "strided write: source layout out of bounds");
    assert!(dst_layout.is_non_overlapping(), "strided write: destination layout overlaps itself");
    let plan = RowPlan::<2>::new(
        dst_layout.dims(),
        [dst_layout.strides(), src_layout.strides()],
        [dst_layout.offset(), src_layout.offset()],
    );
    let [ds, ss] = plan.inner;
    let len = plan.len;
    let base = SendPtr(dst.as_mut_ptr());
    // Writes `len` elements of one destination row starting at `o[0]`, elements `from..to` of the row.
    let row = |o: [usize; 2], from: usize, to: usize| {
        let p = &base;
        for i in from..to {
            let s = src[o[1] + i * ss];
            // SAFETY: the destination layout lies inside `dst` (asserted above) and is injective, so every
            // (row, i) pair addresses a distinct element; rows and row chunks are handed to exactly one task,
            // so no two tasks write the same element and no reference to `dst` exists meanwhile.
            unsafe {
                let d = p.0.add(o[0] + i * ds);
                *d = f(*d, s);
            }
        }
    };
    // Rows `start..end`, serially: the one walk over the rows every path below shares.
    let run = |start: usize, end: usize| plan.for_rows(start..end, |_, o| row(o, 0, len));
    if plan.rows * len < PAR_MIN {
        run(0, plan.rows);
    } else if plan.rows == 1 {
        let o = plan.base;
        par_for(len.div_ceil(CHUNK), &|t| row(o, t * CHUNK, ((t + 1) * CHUNK).min(len)));
    } else {
        let rows_per_task = (CHUNK / len.max(1)).max(1);
        par_for(plan.rows.div_ceil(rows_per_task), &|t| {
            let start = t * rows_per_task;
            run(start, (start + rows_per_task).min(plan.rows));
        });
    }
}

/// A raw pointer that may be shared across rayon tasks writing disjoint regions.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub *mut T);
// SAFETY: users guarantee disjoint access.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: users guarantee disjoint access.
unsafe impl<T> Sync for SendPtr<T> {}

/// Map one strided `f32` operand to a contiguous output.
#[inline]
pub(crate) fn map1(x: &[f32], lx: &Layout, f: impl Fn(f32) -> f32 + Sync) -> Vec<f32> {
    let n = lx.numel();
    let mut out = alloc_uninit(n);
    if let Some(r) = lx.contiguous_range() {
        let x = &x[r];
        // Piece `c` of CHUNK elements (the whole output when serial).
        let piece = |c: usize, o: &mut [f32]| {
            let x = &x[c * CHUNK..c * CHUNK + o.len()];
            for (o, &x) in o.iter_mut().zip(x) {
                *o = f(x);
            }
        };
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
            let n = row.len();
            for (r, &x) in row.iter_mut().zip(&x[o..o + n]) {
                *r = f(x);
            }
        } else {
            for (i, r) in row.iter_mut().enumerate() {
                *r = f(x[o + i * s]);
            }
        }
    });
    out
}

/// Map one strided `f32` operand using a vectorised contiguous/unit-stride slice kernel and its bit-exact
/// scalar twin for non-unit strides.
#[inline]
fn map1_slice(
    x: &[f32],
    lx: &Layout,
    slice_fn: fn(&[f32], &mut [f32]),
    scalar_fn: fn(f32) -> f32,
) -> Vec<f32> {
    let n = lx.numel();
    let mut out = alloc_uninit(n);
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
        } else {
            for (i, r) in row.iter_mut().enumerate() {
                *r = scalar_fn(x[o + i * s]);
            }
        }
    });
    out
}

/// Map two strided `f32` operands of the same logical shape to a contiguous output.
#[inline]
pub(crate) fn map2(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    f: impl Fn(f32, f32) -> f32 + Sync,
) -> Vec<f32> {
    let n = la.numel();
    let mut out = alloc_uninit(n);
    if let (Some(ra), Some(rb)) = (la.contiguous_range(), lb.contiguous_range()) {
        let (a, b) = (&a[ra], &b[rb]);
        // Piece `c` of CHUNK elements (the whole output when serial).
        let piece = |c: usize, o: &mut [f32]| {
            let r = c * CHUNK..c * CHUNK + o.len();
            for ((o, &a), &b) in o.iter_mut().zip(&a[r.clone()]).zip(&b[r]) {
                *o = f(a, b);
            }
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
            (1, 1) => {
                for ((r, &a), &b) in row.iter_mut().zip(&a[oa..oa + len]).zip(&b[ob..ob + len]) {
                    *r = f(a, b);
                }
            }
            (1, 0) => {
                let b = b[ob];
                for (r, &a) in row.iter_mut().zip(&a[oa..oa + len]) {
                    *r = f(a, b);
                }
            }
            (0, 1) => {
                let a = a[oa];
                for (r, &b) in row.iter_mut().zip(&b[ob..ob + len]) {
                    *r = f(a, b);
                }
            }
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = f(a[oa + i * sa], b[ob + i * sb]);
                }
            }
        }
    });
    out
}

/// Map two strided `f32` operands using vectorised contiguous/unit-stride slice kernels (both `(1, 1)` and
/// `(1, 0)` broadcast-`b`, common in `sum_all`/`mean_all` backwards) and their bit-exact scalar twin.
#[inline]
fn map2_slice(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    slice_fn: fn(&[f32], &[f32], &mut [f32]),
    bc_b_fn: fn(&[f32], f32, &mut [f32]),
    scalar_fn: fn(f32, f32) -> f32,
) -> Vec<f32> {
    let n = la.numel();
    let mut out = alloc_uninit(n);
    if let (Some(ra), Some(rb)) = (la.contiguous_range(), lb.contiguous_range()) {
        let (a, b) = (&a[ra], &b[rb]);
        let piece = |c: usize, o: &mut [f32]| {
            let r = c * CHUNK..c * CHUNK + o.len();
            slice_fn(&a[r.clone()], &b[r], o);
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
            (1, 1) => slice_fn(&a[oa..oa + len], &b[ob..ob + len], row),
            (1, 0) => bc_b_fn(&a[oa..oa + len], b[ob], row),
            (0, 1) => {
                let av = a[oa];
                for (r, &bv) in row.iter_mut().zip(&b[ob..ob + len]) {
                    *r = scalar_fn(av, bv);
                }
            }
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = scalar_fn(a[oa + i * sa], b[ob + i * sb]);
                }
            }
        }
    });
    out
}

/// `cond != 0 ? a : b` with three strided operands of the output shape.
pub(crate) fn where3<C: Copy + Send + Sync>(
    c: &[C],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    truthy: impl Fn(C) -> bool + Sync,
) -> Vec<f32> {
    let mut out = alloc_uninit(la.numel());
    let plan = RowPlan::<3>::new(
        la.dims(),
        [lc.strides(), la.strides(), lb.strides()],
        [lc.offset(), la.offset(), lb.offset()],
    );
    let [sc, sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oc, oa, ob]| {
        for (i, r) in row.iter_mut().enumerate() {
            *r = if truthy(c[oc + i * sc]) { a[oa + i * sa] } else { b[ob + i * sb] };
        }
    });
    out
}

const LOG2E: f32 = std::f32::consts::LOG2_E;

/// `e^x` to about 1 ulp over the whole finite range (including subnormal results), written so loops over it
/// vectorise: no libm call and only branch-free selects.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    // Beyond these bounds the result is +inf / rounds to 0; clamping keeps the integer exponent in range.
    let xc = x.clamp(-104.0, 89.0);
    // Round to nearest via the 1.5 * 2^23 trick (exact for |v| < 2^22).
    let magic = 12_582_912.0f32;
    let n = (xc * LOG2E + magic) - magic;
    let r = xc - n * 0.693_359_4 - n * -2.121_944_4e-4;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let y = p * r * r + r + 1.0;
    // 2^n as two normal powers of two (n is in -150..=129), so huge results overflow to inf and tiny ones
    // underflow gradually instead of wrapping the exponent field.
    let ni = n as i32;
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    let v = y * s1 * s2;
    if x.is_nan() { x } else { v }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    // Stable in both tails: e = exp(-|x|) never overflows, and e / (1 + e) keeps tiny values for x << 0.
    let e = fast_exp(-x.abs());
    let r = 1.0 / (1.0 + e);
    if x >= 0.0 { r } else { e * r }
}

/// `x · sigmoid(x)`: scalar SiLU helper for fused CPU layers (matches `Tensor::silu` within [`SILU_MAX_REL_ERR`]).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `g · silu'(x)`: scalar SiLU backward helper for fused CPU layers.
#[inline(always)]
pub fn silu_backward(x: f32, g: f32) -> f32 {
    let s = sigmoid(x);
    g * (s + x * s * (1.0 - s))
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + fast_exp(-x.abs()).ln_1p()
}

#[inline(always)]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `v[i] = op(v[i] + bias)` in place (`op = None`: just the bias), for fused epilogues. The usual activations
/// have dedicated loops; other ops go through [`unary`].
pub(crate) fn bias_act_inplace(op: Option<UnaryOp>, bias: f32, v: &mut [f32]) {
    #[inline(always)]
    fn go(v: &mut [f32], bias: f32, f: impl Fn(f32) -> f32) {
        for x in v.iter_mut() {
            *x = f(*x + bias);
        }
    }
    match op {
        None if bias == 0.0 => {}
        None => go(v, bias, |x| x),
        Some(UnaryOp::Silu) => simd::silu_bias_inplace(v, bias),
        Some(UnaryOp::Relu) => go(v, bias, |x| x.max(0.0)),
        Some(UnaryOp::Sigmoid) => simd::sigmoid_bias_inplace(v, bias),
        Some(UnaryOp::LeakyRelu(a)) => go(v, bias, |x| if x > 0.0 { x } else { a * x }),
        Some(UnaryOp::Hardswish) => go(v, bias, |x| x * (x + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)),
        Some(op) => {
            go(v, bias, |x| x);
            let y = unary(op, v, &Layout::contiguous(crate::Shape::new(&[v.len()])));
            v.copy_from_slice(&y);
            super::recycle(y);
        }
    }
}

/// The error function, to about 1 ulp in `f32`: a Taylor series near 0 (relative accuracy for tiny `x`), and
/// `1 - erfc(x)` with the Chebyshev-fitted `erfc` of Numerical Recipes (fractional error < 1.2e-7) beyond.
#[inline]
pub(crate) fn erf(x: f32) -> f32 {
    let z = (x as f64).abs();
    let r = if z < 0.5 {
        // 2/sqrt(pi) * sum (-1)^n z^(2n+1) / (n! (2n+1)); the terms after z^17 are below 1e-12 relative.
        let z2 = z * z;
        let mut term = z;
        let mut sum = z;
        for n in 1..9 {
            term *= -z2 / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        let t = 1.0 / (1.0 + 0.5 * z);
        let p = -1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
        1.0 - t * (-z * z + p).exp()
    };
    if x.is_nan() { x } else { (r as f32).copysign(x) }
}

/// Apply a unary op.
pub(crate) fn unary(op: UnaryOp, x: &[f32], lx: &Layout) -> Vec<f32> {
    use UnaryOp::*;
    match op {
        Neg => map1(x, lx, |v| -v),
        Abs => map1(x, lx, f32::abs),
        Exp => map1_slice(x, lx, simd::exp_slice, simd::exp_scalar),
        Log => map1(x, lx, f32::ln),
        Sqrt => map1(x, lx, f32::sqrt),
        Rsqrt => map1(x, lx, |v| 1.0 / v.sqrt()),
        Recip => map1(x, lx, |v| 1.0 / v),
        Square => map1(x, lx, |v| v * v),
        Sin => map1(x, lx, f32::sin),
        Cos => map1(x, lx, f32::cos),
        Tanh => map1(x, lx, f32::tanh),
        Atan => map1(x, lx, f32::atan),
        Asin => map1(x, lx, f32::asin),
        Acos => map1(x, lx, f32::acos),
        Erf => map1(x, lx, erf),
        Sigmoid => map1_slice(x, lx, simd::sigmoid_slice, simd::sigmoid_scalar),
        Silu => map1_slice(x, lx, simd::silu_slice, simd::silu_epilogue_scalar),
        Relu => map1(x, lx, |v| v.max(0.0)),
        LeakyRelu(a) => map1(x, lx, move |v| if v > 0.0 { v } else { a * v }),
        Hardswish => map1(x, lx, |v| v * (v + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)),
        Softplus => map1(x, lx, softplus),
        Clamp(lo, hi) => map1(x, lx, move |v| v.max(lo).min(hi)),
        PowScalar(p) => {
            if p == 2.0 {
                map1(x, lx, |v| v * v)
            } else if p == 0.5 {
                map1(x, lx, f32::sqrt)
            } else {
                map1(x, lx, move |v| v.powf(p))
            }
        }
        AddScalar(c) => map1(x, lx, move |v| v + c),
        MulScalar(c) => map1(x, lx, move |v| v * c),
        RSubScalar(c) => map1(x, lx, move |v| c - v),
        RDivScalar(c) => map1(x, lx, move |v| c / v),
        DivScalar(c) => map1(x, lx, move |v| v / c),
        MaxScalar(c) => map1(x, lx, move |v| v.max(c)),
        MinScalar(c) => map1(x, lx, move |v| v.min(c)),
        Floor => map1(x, lx, f32::floor),
        Ceil => map1(x, lx, f32::ceil),
        Round => map1(x, lx, f32::round),
        Sign => map1(x, lx, sign),
    }
}

/// `grad * op'(x)`.
pub(crate) fn unary_backward(op: UnaryOp, x: &[f32], lx: &Layout, g: &[f32], lg: &Layout) -> Vec<f32> {
    use UnaryOp::*;
    match op {
        Neg => map2(x, lx, g, lg, |_, g| -g),
        Abs => map2(x, lx, g, lg, |x, g| g * sign(x)),
        Exp => map2_slice(x, lx, g, lg, simd::exp_bwd_slice, simd::exp_bwd_broadcast_g, simd::exp_bwd_scalar),
        Log => map2(x, lx, g, lg, |x, g| g / x),
        Sqrt => map2(x, lx, g, lg, |x, g| g * 0.5 / x.sqrt()),
        // Divide in stages rather than squaring x, which under/overflows for representable results.
        Rsqrt => map2(x, lx, g, lg, |x, g| -0.5 * g / x.sqrt() / x),
        Recip => map2(x, lx, g, lg, |x, g| -(g / x) / x),
        Square => map2(x, lx, g, lg, |x, g| 2.0 * x * g),
        Sin => map2(x, lx, g, lg, |x, g| g * x.cos()),
        Cos => map2(x, lx, g, lg, |x, g| -g * x.sin()),
        Tanh => map2(x, lx, g, lg, |x, g| {
            let t = x.tanh();
            g * (1.0 - t * t)
        }),
        // g / (1 + x^2), written as (g / x) / (x + 1 / x) for |x| > 1 so that x^2 never overflows.
        Atan => {
            map2(x, lx, g, lg, |x, g| if x.abs() > 1.0 { (g / x) / (x + 1.0 / x) } else { g / (1.0 + x * x) })
        }
        // (1 - x)(1 + x) keeps precision near |x| = 1, where 1 - x^2 cancels.
        Asin => map2(x, lx, g, lg, |x, g| g / ((1.0 - x) * (1.0 + x)).sqrt()),
        Acos => map2(x, lx, g, lg, |x, g| -g / ((1.0 - x) * (1.0 + x)).sqrt()),
        // In f64, rounded once: exp(-x^2) underflows f32 long before g * exp(-x^2) does (x = 11, g = 3e38).
        Erf => map2(x, lx, g, lg, |x, g| {
            let x = x as f64;
            (g as f64 * std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp()) as f32
        }),
        Sigmoid => map2_slice(
            x,
            lx,
            g,
            lg,
            simd::sigmoid_bwd_slice,
            simd::sigmoid_bwd_broadcast_g,
            simd::sigmoid_bwd_scalar,
        ),
        Silu => {
            map2_slice(x, lx, g, lg, simd::silu_bwd_slice, simd::silu_bwd_broadcast_g, simd::silu_bwd_scalar)
        }
        Relu => map2(x, lx, g, lg, |x, g| if x > 0.0 { g } else { 0.0 }),
        LeakyRelu(a) => map2(x, lx, g, lg, move |x, g| if x > 0.0 { g } else { a * g }),
        Hardswish => map2(x, lx, g, lg, |x, g| {
            if x < -3.0 {
                0.0
            } else if x > 3.0 {
                g
            } else {
                g * (2.0 * x + 3.0) * (1.0 / 6.0)
            }
        }),
        Softplus => map2_slice(
            x,
            lx,
            g,
            lg,
            simd::softplus_bwd_slice,
            simd::softplus_bwd_broadcast_g,
            simd::softplus_bwd_scalar,
        ),
        Clamp(lo, hi) => map2(x, lx, g, lg, move |x, g| if x >= lo && x <= hi { g } else { 0.0 }),
        PowScalar(p) => map2(x, lx, g, lg, move |x, g| if p == 0.0 { 0.0 } else { g * p * x.powf(p - 1.0) }),
        AddScalar(_) => map2(x, lx, g, lg, |_, g| g),
        MulScalar(c) => map2(x, lx, g, lg, move |_, g| g * c),
        RSubScalar(_) => map2(x, lx, g, lg, |_, g| -g),
        RDivScalar(c) => map2(x, lx, g, lg, move |x, g| -(g * (c / x)) / x),
        DivScalar(c) => map2(x, lx, g, lg, move |_, g| g / c),
        MaxScalar(c) => map2(x, lx, g, lg, move |x, g| if x >= c { g } else { 0.0 }),
        MinScalar(c) => map2(x, lx, g, lg, move |x, g| if x <= c { g } else { 0.0 }),
        Floor | Ceil | Round | Sign => map2(x, lx, g, lg, |_, _| 0.0),
    }
}

#[inline(always)]
fn mask(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Apply a binary op to two operands already broadcast to the output shape.
pub(crate) fn binary(op: BinaryOp, a: &[f32], la: &Layout, b: &[f32], lb: &Layout) -> Vec<f32> {
    use BinaryOp::*;
    match op {
        Add => map2(a, la, b, lb, |a, b| a + b),
        Sub => map2(a, la, b, lb, |a, b| a - b),
        Mul => map2(a, la, b, lb, |a, b| a * b),
        Div => map2(a, la, b, lb, |a, b| a / b),
        Pow => map2(a, la, b, lb, f32::powf),
        Maximum => map2(a, la, b, lb, f32::max),
        Minimum => map2(a, la, b, lb, f32::min),
        Atan2 => map2(a, la, b, lb, f32::atan2),
        Eq => map2(a, la, b, lb, |a, b| mask(a == b)),
        Ne => map2(a, la, b, lb, |a, b| mask(a != b)),
        Gt => map2(a, la, b, lb, |a, b| mask(a > b)),
        Ge => map2(a, la, b, lb, |a, b| mask(a >= b)),
        Lt => map2(a, la, b, lb, |a, b| mask(a < b)),
        Le => map2(a, la, b, lb, |a, b| mask(a <= b)),
    }
}

#[cfg(test)]
mod tests {
    use super::simd::silu_epilogue_scalar;
    use super::*;

    #[test]
    fn fast_exp_accuracy() {
        let mut worst = 0f64;
        let mut x = -87.3f32;
        while x < 88.72 {
            let e = fast_exp(x) as f64;
            let t = (x as f64).exp();
            worst = worst.max(((e - t) / t).abs());
            x += 0.0137;
        }
        assert!(worst < 4e-7, "worst relative error {worst}");
        // Near the top of the range and in the subnormal range (absolute error there is limited by spacing).
        for x in [88.0f32, 88.5, 88.7, 88.72] {
            let (e, t) = (fast_exp(x) as f64, (x as f64).exp());
            assert!(((e - t) / t).abs() < 4e-7, "{x}: {e} vs {t}");
        }
        for x in [-88.0f32, -95.0, -100.0, -103.0] {
            let (e, t) = (fast_exp(x) as f64, (x as f64).exp());
            assert!((e - t).abs() <= 2.0 * f32::from_bits(1) as f64 + 1e-6 * t, "{x}: {e} vs {t}");
        }
        assert_eq!(fast_exp(88.8), f32::INFINITY);
        assert_eq!(fast_exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(fast_exp(-110.0), 0.0);
        assert_eq!(fast_exp(f32::NEG_INFINITY), 0.0);
        assert!(fast_exp(f32::NAN).is_nan());
        assert_eq!(fast_exp(0.0), 1.0);
    }

    #[test]
    fn silu_does_not_depend_on_segment_boundaries() {
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.37).sin() * 9.0).collect();
        let mut whole = xs.clone();
        silu_bias_inplace(&mut whole, 0.25);
        let mut pieces = xs.clone();
        let mut at = 0;
        for len in (1..20).cycle() {
            if at >= pieces.len() {
                break;
            }
            let end = (at + len).min(pieces.len());
            silu_bias_inplace(&mut pieces[at..end], 0.25);
            at = end;
        }
        assert!(whole.iter().zip(&pieces).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn silu_avx2_error_bound() {
        let silu = |x: f64| x / (1.0 + (-x).exp());
        let check = |xs: &[f32]| {
            let mut v = xs.to_vec();
            silu_bias_inplace(&mut v, 0.0);
            let mut worst = 0f64;
            for (&x, &y) in xs.iter().zip(&v) {
                let want = silu(x as f64);
                let err = (y as f64 - want).abs();
                // Once exp(-|x|) is subnormal its spacing, times |x|, limits any f32 x * sigmoid(x).
                let slack = (x.abs() as f64 + 1.0) * 2.0 * f32::from_bits(1) as f64;
                assert!(err <= SILU_MAX_REL_ERR * want.abs() + slack, "silu({x}) = {y}, want {want}");
                if want.abs() > 1e-30 {
                    worst = worst.max(err / want.abs());
                }
            }
            worst
        };
        // A dense sweep (every lane position sees every kind of value), the tails, and the subnormal range.
        let xs: Vec<f32> = (0..400_003).map(|i| -100.0 + i as f32 * 5e-4).collect();
        let worst = check(&xs);
        let tails: Vec<f32> =
            [-104.0f32, -103.5, -95.0, -88.0, -87.3, -20.0, -1e-30, 1e-30, 1e-7, 20.0, 88.7, 1e30, f32::MAX]
                .into_iter()
                .chain((0..64).map(|i| -0.5 + i as f32 / 64.0))
                .collect();
        check(&tails);
        println!("worst relative error of the vectorised SiLU: {worst:e}");
        // Special values match the scalar path.
        let mut sp = vec![f32::NAN, f32::INFINITY, -0.0, 0.0, -110.0, f32::MIN, 3.0, -3.0, 1.0];
        silu_bias_inplace(&mut sp, 0.0);
        assert!(sp[0].is_nan());
        assert_eq!(sp[1], f32::INFINITY);
        assert_eq!(sp[2], 0.0);
        assert_eq!(sp[3], 0.0);
        assert!(sp[4] <= 0.0 && sp[4] > -1e-40);
        // The bias is added before the activation.
        let mut b = vec![0.5f32; 19];
        silu_bias_inplace(&mut b, -1.5);
        for y in b {
            assert!(((y as f64) - silu(-1.0)).abs() < 1e-6);
        }
        // Scalar twin matches AVX2 bit for bit on normal, subnormal, tail and special values.
        let mut scalar_check = tails;
        scalar_check.extend([0.0f32, -0.0, f32::INFINITY, -110.0, f32::MIN, 3.0, -3.0, 1.0, -1.0, 0.25]);
        let mut avx_out = scalar_check.clone();
        silu_bias_inplace(&mut avx_out, 0.0);
        for (&x, &y) in scalar_check.iter().zip(&avx_out) {
            let sc = silu_epilogue_scalar(x + 0.0);
            assert_eq!(sc.to_bits(), y.to_bits(), "silu_bias scalar vs AVX2 mismatch at {x}: {sc} vs {y}");
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            for ch in scalar_check.chunks(8) {
                let mut v = [0f32; 8];
                v[..ch.len()].copy_from_slice(ch);
                // SAFETY: AVX2 and FMA were detected.
                let got: [f32; 8] = unsafe {
                    use std::arch::x86_64::*;
                    let mut o = [0f32; 8];
                    _mm256_storeu_ps(o.as_mut_ptr(), silu8(_mm256_loadu_ps(v.as_ptr())));
                    o
                };
                for (&x, &y) in v.iter().zip(&got) {
                    assert_eq!(
                        silu_epilogue_scalar(x).to_bits(),
                        y.to_bits(),
                        "silu8 direct scalar vs AVX2 mismatch at {x}"
                    );
                }
            }
        }
        assert!(silu_epilogue_scalar(f32::NAN).is_nan());
    }

    #[test]
    fn exp_sigmoid_silu_simd_accuracy_and_edges() {
        // Dense sweep across finite, subnormal, and tail inputs (odd length exercises 16 + 8 + tail paths).
        let mut xs: Vec<f32> = (0..100_005).map(|i| -105.0 + i as f32 * 0.002_05).collect();
        xs.extend([
            f32::NEG_INFINITY,
            -110.0,
            -104.0,
            -103.5,
            -95.0,
            -90.0,
            -88.0,
            -87.3,
            -20.0,
            -1.0,
            -1e-30,
            -0.0,
            0.0,
            1e-30,
            1e-7,
            1.0,
            20.0,
            88.0,
            88.72,
            88.8,
            90.0,
            110.0,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
        ]);
        let gs: Vec<f32> = (0..xs.len()).map(|i| ((i as f32 * 0.73).sin()) * 2.5).collect();
        let n = xs.len();

        let (mut y_exp, mut y_sig, mut y_silu) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        let (mut b_exp, mut b_sig, mut b_silu) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
        simd::exp_slice(&xs, &mut y_exp);
        simd::sigmoid_slice(&xs, &mut y_sig);
        simd::silu_slice(&xs, &mut y_silu);
        simd::exp_bwd_slice(&xs, &gs, &mut b_exp);
        simd::sigmoid_bwd_slice(&xs, &gs, &mut b_sig);
        simd::silu_bwd_slice(&xs, &gs, &mut b_silu);

        let sub = f32::from_bits(1) as f64;
        for i in 0..n {
            let (x, g) = (xs[i], gs[i]);
            // Bit-exact parity between AVX2 slice and scalar twin (including NaN payload).
            assert_eq!(y_exp[i].to_bits(), simd::exp_scalar(x).to_bits(), "exp bits at {x}");
            assert_eq!(y_sig[i].to_bits(), simd::sigmoid_scalar(x).to_bits(), "sigmoid bits at {x}");
            assert_eq!(y_silu[i].to_bits(), simd::silu_epilogue_scalar(x).to_bits(), "silu bits at {x}");
            assert_eq!(b_exp[i].to_bits(), simd::exp_bwd_scalar(x, g).to_bits(), "exp_bwd bits at {x}");
            assert_eq!(b_sig[i].to_bits(), simd::sigmoid_bwd_scalar(x, g).to_bits(), "sig_bwd bits at {x}");
            assert_eq!(b_silu[i].to_bits(), simd::silu_bwd_scalar(x, g).to_bits(), "silu_bwd bits at {x}");

            if x.is_nan() {
                assert!(y_exp[i].is_nan() && y_sig[i].is_nan() && y_silu[i].is_nan());
                assert!(b_exp[i].is_nan() && b_sig[i].is_nan() && b_silu[i].is_nan());
                continue;
            }
            // Accuracy vs f64 reference.
            let x64 = x as f64;
            let g64 = g as f64;
            let want_exp = x64.exp();
            if want_exp.is_infinite() || want_exp > f32::MAX as f64 {
                assert_eq!(y_exp[i], f32::INFINITY, "exp overflow at {x}");
            } else {
                let err = (y_exp[i] as f64 - want_exp).abs();
                assert!(err <= 4e-7 * want_exp + 2.0 * sub, "exp({x}) = {}, want {want_exp}", y_exp[i]);
            }
            let want_sig = 1.0 / (1.0 + (-x64).exp());
            let sig_err = (y_sig[i] as f64 - want_sig).abs();
            assert!(sig_err <= 5e-7 * want_sig + 2.0 * sub, "sigmoid({x}) = {}, want {want_sig}", y_sig[i]);

            let want_sig_bwd = g64 * (-x64.abs()).exp() / (1.0 + (-x64.abs()).exp()).powi(2);
            let sb_err = (b_sig[i] as f64 - want_sig_bwd).abs();
            assert!(
                sb_err <= 8e-7 * want_sig_bwd.abs() + (2.0 * g64.abs() + 1.0) * sub,
                "sig_bwd({x}) = {}, want {want_sig_bwd}",
                b_sig[i]
            );

            if x.is_finite() {
                let dsig = (-x64.abs()).exp() / (1.0 + (-x64.abs()).exp()).powi(2);
                let want_silu_bwd = g64 * (want_sig + x64 * dsig);
                let scale = g64.abs() * (want_sig + x64.abs() * dsig);
                let sib_err = (b_silu[i] as f64 - want_silu_bwd).abs();
                let slack = ((x64.abs() + 1.0) * g64.abs() * 2.0 + 1.0) * sub;
                assert!(
                    sib_err <= 8e-7 * scale + slack,
                    "silu_bwd({x}) = {}, want {want_silu_bwd}",
                    b_silu[i]
                );
            }
        }

        // Exact edge-value checks.
        assert_eq!(simd::exp_scalar(0.0), 1.0);
        assert_eq!(simd::exp_scalar(-0.0), 1.0);
        assert_eq!(simd::exp_scalar(90.0), f32::INFINITY);
        assert_eq!(simd::exp_scalar(f32::INFINITY), f32::INFINITY);
        assert_eq!(simd::exp_scalar(f32::NEG_INFINITY), 0.0);
        assert!(simd::exp_scalar(-90.0) > 0.0 && simd::exp_scalar(-90.0) < 1e-38);
        assert_eq!(simd::sigmoid_scalar(0.0), 0.5);
        assert_eq!(simd::sigmoid_scalar(-0.0), 0.5);
        assert_eq!(simd::sigmoid_scalar(90.0), 1.0);
        assert_eq!(simd::sigmoid_scalar(f32::INFINITY), 1.0);
        assert_eq!(simd::sigmoid_scalar(f32::NEG_INFINITY), 0.0);
        assert_eq!(simd::sigmoid_bwd_scalar(0.0, 2.0), 0.5);
        assert_eq!(simd::sigmoid_bwd_scalar(f32::INFINITY, 2.0), 0.0);
        assert_eq!(simd::sigmoid_bwd_scalar(f32::NEG_INFINITY, 2.0), 0.0);
        assert_eq!(simd::silu_bwd_scalar(0.0, 2.0), 1.0);
        assert_eq!(simd::silu_bwd_scalar(90.0, 2.0), 2.0);
        assert!(simd::silu_bwd_scalar(f32::INFINITY, 2.0).is_nan());
        assert!(simd::silu_bwd_scalar(f32::NEG_INFINITY, 2.0).is_nan());

        // Broadcast-g backward matches dense-g backward bit-for-bit.
        let mut bc_sig = vec![0f32; n];
        let mut bc_silu = vec![0f32; n];
        let mut dense_sig = vec![0f32; n];
        let mut dense_silu = vec![0f32; n];
        let g_const = vec![1.25f32; n];
        simd::sigmoid_bwd_broadcast_g(&xs, 1.25, &mut bc_sig);
        simd::sigmoid_bwd_slice(&xs, &g_const, &mut dense_sig);
        simd::silu_bwd_broadcast_g(&xs, 1.25, &mut bc_silu);
        simd::silu_bwd_slice(&xs, &g_const, &mut dense_silu);
        for i in 0..n {
            assert_eq!(bc_sig[i].to_bits(), dense_sig[i].to_bits());
            assert_eq!(bc_silu[i].to_bits(), dense_silu[i].to_bits());
        }
    }

    #[test]
    fn erf_accuracy() {
        // Reference: the Maclaurin series in f64 (converges everywhere; 200 terms suffice for |x| <= 4.5).
        let reference = |x: f64| {
            let (mut term, mut sum) = (x, x);
            for n in 1..200 {
                term *= -x * x / n as f64;
                sum += term / (2 * n + 1) as f64;
            }
            sum * std::f64::consts::FRAC_2_SQRT_PI
        };
        let mut worst = 0f64;
        let mut x = -4.5f32;
        while x <= 4.5 {
            let (got, want) = (erf(x) as f64, reference(x as f64));
            let a = (want.abs() as f32).max(f32::MIN_POSITIVE);
            let ulp = (f32::from_bits(a.to_bits() + 1) - a) as f64;
            worst = worst.max((got - want).abs() / ulp);
            x += 0.00731;
        }
        assert!(worst <= 2.0, "worst error {worst} ulp");
        for x in [1e-30f32, 1e-8, -3e-5] {
            let want = x as f64 * std::f64::consts::FRAC_2_SQRT_PI;
            assert!(((erf(x) as f64) - want).abs() <= 1e-7 * want.abs(), "{x}");
        }
        assert_eq!(erf(0.0), 0.0);
        assert!(erf(-0.0).is_sign_negative());
        assert_eq!(erf(10.0), 1.0);
        assert_eq!(erf(f32::NEG_INFINITY), -1.0);
        assert!(erf(f32::NAN).is_nan());
    }
}
