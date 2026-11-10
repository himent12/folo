//! Copies, casts and elementwise kernels.

use rayon::prelude::*;

use super::pool_alloc::alloc_uninit;
use super::{CHUNK, PAR_MIN};
use crate::Layout;
use crate::backend::{BinaryOp, UnaryOp};
use crate::layout::RowPlan;

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
    if out.len() < PAR_MIN || plan.rows == 1 {
        if plan.rows == 1 && out.len() >= PAR_MIN {
            // One long row: split it into chunks with adjusted offsets.
            let inner = plan.inner;
            plan.for_rows(0..1, |_, offs| {
                out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
                    let o = std::array::from_fn(|k| offs[k] + c * CHUNK * inner[k]);
                    f(chunk, o);
                });
            });
            return;
        }
        let mut rows = out.chunks_mut(len);
        plan.for_rows(0..plan.rows, |_, offs| f(rows.next().unwrap(), offs));
        return;
    }
    let rows_per_task = (CHUNK / len).max(1);
    out.par_chunks_mut(rows_per_task * len).enumerate().for_each(|(t, chunk)| {
        let start = t * rows_per_task;
        let end = (start + rows_per_task).min(plan.rows);
        let mut rows = chunk.chunks_mut(len);
        plan.for_rows(start..end, |_, offs| f(rows.next().unwrap(), offs));
    });
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
    if plan.rows * len < PAR_MIN {
        plan.for_rows(0..plan.rows, |_, o| row(o, 0, len));
    } else if plan.rows == 1 {
        plan.for_rows(0..1, |_, o| {
            (0..len.div_ceil(CHUNK))
                .into_par_iter()
                .for_each(|t| row(o, t * CHUNK, ((t + 1) * CHUNK).min(len)));
        });
    } else {
        let rows_per_task = (CHUNK / len.max(1)).max(1);
        (0..plan.rows.div_ceil(rows_per_task)).into_par_iter().for_each(|t| {
            let start = t * rows_per_task;
            plan.for_rows(start..(start + rows_per_task).min(plan.rows), |_, o| row(o, 0, len));
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
        if n >= PAR_MIN {
            out.par_chunks_mut(CHUNK).zip(x.par_chunks(CHUNK)).for_each(|(o, x)| {
                for (o, &x) in o.iter_mut().zip(x) {
                    *o = f(x);
                }
            });
        } else {
            for (o, &x) in out.iter_mut().zip(x) {
                *o = f(x);
            }
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
        if n >= PAR_MIN {
            out.par_chunks_mut(CHUNK).zip(a.par_chunks(CHUNK).zip(b.par_chunks(CHUNK))).for_each(
                |(o, (a, b))| {
                    for ((o, &a), &b) in o.iter_mut().zip(a).zip(b) {
                        *o = f(a, b);
                    }
                },
            );
        } else {
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                *o = f(a, b);
            }
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

/// `x · sigmoid(x)`: the CPU's SiLU, element by element, exactly as `Tensor::silu` computes it on the CPU.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `g · silu'(x)`, exactly as the CPU backward of `Tensor::silu` computes it.
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

/// `v[i] = silu(v[i] + bias)` in place: the fused-epilogue SiLU. With AVX2 + FMA (detected at runtime) eight
/// lanes at a time through [`silu8`] (a ragged tail too, padded), whose error against the exact SiLU is
/// bounded by [`SILU_MAX_REL_ERR`] (tested); otherwise the scalar `x * sigmoid(x)`. Either way every element's
/// result depends only on its value, never on its position in `v`.
pub(crate) fn silu_bias_inplace(v: &mut [f32], bias: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 and FMA were detected above.
            return unsafe { silu_bias_avx2(v, bias) };
        }
    }
    for x in v.iter_mut() {
        let t = *x + bias;
        *x = t * sigmoid(t);
    }
}

/// Bound on the relative error of the vectorised SiLU (`|y - silu(x)| <= SILU_MAX_REL_ERR * |silu(x)|`, plus
/// the subnormal spacing once `silu(x)` underflows), checked by `silu_avx2_error_bound`: about 4 ulp.
pub const SILU_MAX_REL_ERR: f64 = 5e-7;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn silu_bias_avx2(v: &mut [f32], bias: f32) {
    use std::arch::x86_64::*;
    let b = _mm256_set1_ps(bias);
    let (chunks, tail) = v.as_chunks_mut::<8>();
    for c in chunks {
        // SAFETY: `c` is 8 floats; unaligned loads and stores.
        unsafe {
            let x = _mm256_add_ps(_mm256_loadu_ps(c.as_ptr()), b);
            _mm256_storeu_ps(c.as_mut_ptr(), silu8(x));
        }
    }
    if !tail.is_empty() {
        // The tail goes through the same lanes (padded), so an element's value never depends on where a
        // caller's segment boundaries fall (GEMM tiles move with the thread count).
        let mut pad = [0f32; 8];
        pad[..tail.len()].copy_from_slice(tail);
        // SAFETY: `pad` is 8 floats.
        unsafe {
            let x = _mm256_add_ps(_mm256_loadu_ps(pad.as_ptr()), b);
            _mm256_storeu_ps(pad.as_mut_ptr(), silu8(x));
        }
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// `x * sigmoid(x)` on eight lanes: `e = exp(-|x|)` by Cody-Waite reduction and the degree-6 polynomial of
/// [`fast_exp`] (FMA), `sigmoid = 1 / (1 + e)` for `x >= 0` and `e / (1 + e)` below, so neither tail overflows.
/// NaN stays NaN.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    // t = -|x| clamped to the bottom of the exp range (a NaN lane becomes -104; x * sig is NaN again below).
    let t = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let t = _mm256_max_ps(_mm256_sub_ps(zero, t), _mm256_set1_ps(-104.0));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        t,
        _mm256_set1_ps(LOG2E),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), t);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    let y = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r), one);
    // 2^n as two normal powers of two (n is in -150..=0), so tiny results underflow gradually.
    let ni = _mm256_cvtps_epi32(n);
    let n1 = _mm256_srai_epi32::<1>(ni);
    let n2 = _mm256_sub_epi32(ni, n1);
    let bias127 = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias127)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias127)));
    let e = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);
    let r = _mm256_div_ps(one, _mm256_add_ps(one, e));
    let sig = _mm256_blendv_ps(_mm256_mul_ps(e, r), r, _mm256_cmp_ps::<_CMP_GE_OQ>(x, zero));
    _mm256_mul_ps(x, sig)
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
        Some(UnaryOp::Silu) => silu_bias_inplace(v, bias),
        Some(UnaryOp::Relu) => go(v, bias, |x| x.max(0.0)),
        Some(UnaryOp::Sigmoid) => go(v, bias, sigmoid),
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
        Exp => map1(x, lx, fast_exp),
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
        Sigmoid => map1(x, lx, sigmoid),
        Silu => map1(x, lx, silu),
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
        Exp => map2(x, lx, g, lg, |x, g| g * fast_exp(x)),
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
        Sigmoid => map2(x, lx, g, lg, |x, g| {
            let s = sigmoid(x);
            g * s * (1.0 - s)
        }),
        Silu => map2(x, lx, g, lg, silu_backward),
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
        Softplus => map2(x, lx, g, lg, |x, g| g * sigmoid(x)),
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
