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

/// Write `src` (read through `src_layout`) into `dst` through `dst_layout`. Shapes must match and `dst_layout`
/// must not broadcast.
pub(crate) fn copy_into<T: Copy + Send + Sync>(
    dst: &mut [T],
    dst_layout: &Layout,
    src: &[T],
    src_layout: &Layout,
) {
    debug_assert_eq!(dst_layout.shape(), src_layout.shape());
    debug_assert!(!dst_layout.has_broadcast());
    let plan = RowPlan::<2>::new(
        dst_layout.dims(),
        [dst_layout.strides(), src_layout.strides()],
        [dst_layout.offset(), src_layout.offset()],
    );
    let [ds, ss] = plan.inner;
    let len = plan.len;
    let body = |dst: &mut [T], o: [usize; 2]| {
        if ds == 1 && ss == 1 {
            dst[o[0]..o[0] + len].copy_from_slice(&src[o[1]..o[1] + len]);
        } else {
            for i in 0..len {
                dst[o[0] + i * ds] = src[o[1] + i * ss];
            }
        }
    };
    if plan.rows * len >= PAR_MIN && plan.rows > 1 {
        // Rows of a non-broadcast destination are disjoint, so writing them concurrently is sound.
        let ptr = SendPtr(dst.as_mut_ptr());
        let dlen = dst.len();
        let rows_per_task = (CHUNK / len.max(1)).max(1);
        let tasks = plan.rows.div_ceil(rows_per_task);
        (0..tasks).into_par_iter().for_each(|t| {
            let p = &ptr;
            // SAFETY: every task writes a disjoint set of destination rows (dst_layout does not broadcast),
            // all within `dst` (the caller validated the layout against the storage length).
            let dst = unsafe { std::slice::from_raw_parts_mut(p.0, dlen) };
            let start = t * rows_per_task;
            plan.for_rows(start..(start + rows_per_task).min(plan.rows), |_, o| body(dst, o));
        });
    } else {
        plan.for_rows(0..plan.rows, |_, o| body(dst, o));
    }
}

/// `dst[i] = f(dst[i], src[i])` through two layouts of the same shape (`dst_layout` must not broadcast).
pub(crate) fn fill_rows_inplace(
    dst: &mut [f32],
    dst_layout: &Layout,
    src: &[f32],
    src_layout: &Layout,
    f: impl Fn(f32, f32) -> f32 + Sync,
) {
    debug_assert!(!dst_layout.has_broadcast());
    let plan = RowPlan::<2>::new(
        dst_layout.dims(),
        [dst_layout.strides(), src_layout.strides()],
        [dst_layout.offset(), src_layout.offset()],
    );
    let [ds, ss] = plan.inner;
    let len = plan.len;
    let body = |dst: &mut [f32], o: [usize; 2]| {
        if ds == 1 && ss == 1 {
            for (d, &s) in dst[o[0]..o[0] + len].iter_mut().zip(&src[o[1]..o[1] + len]) {
                *d = f(*d, s);
            }
        } else if ds == 1 && ss == 0 {
            let s = src[o[1]];
            for d in &mut dst[o[0]..o[0] + len] {
                *d = f(*d, s);
            }
        } else {
            for i in 0..len {
                let j = o[0] + i * ds;
                dst[j] = f(dst[j], src[o[1] + i * ss]);
            }
        }
    };
    if plan.rows * len >= PAR_MIN {
        let ptr = SendPtr(dst.as_mut_ptr());
        let dlen = dst.len();
        if plan.rows == 1 {
            // One long row: split it into chunks.
            plan.for_rows(0..1, |_, o| {
                let tasks = len.div_ceil(CHUNK);
                (0..tasks).into_par_iter().for_each(|t| {
                    let p = &ptr;
                    // SAFETY: chunks of one non-broadcast row are disjoint and inside `dst`.
                    let dst = unsafe { std::slice::from_raw_parts_mut(p.0, dlen) };
                    let (a, b) = (t * CHUNK, ((t + 1) * CHUNK).min(len));
                    for i in a..b {
                        let j = o[0] + i * ds;
                        dst[j] = f(dst[j], src[o[1] + i * ss]);
                    }
                });
            });
            return;
        }
        let rows_per_task = (CHUNK / len.max(1)).max(1);
        let tasks = plan.rows.div_ceil(rows_per_task);
        (0..tasks).into_par_iter().for_each(|t| {
            let p = &ptr;
            // SAFETY: rows of a non-broadcast destination are disjoint; every task writes its own rows.
            let dst = unsafe { std::slice::from_raw_parts_mut(p.0, dlen) };
            let start = t * rows_per_task;
            plan.for_rows(start..(start + rows_per_task).min(plan.rows), |_, o| body(dst, o));
        });
    } else {
        plan.for_rows(0..plan.rows, |_, o| body(dst, o));
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

/// `e^x` to about 1 ulp, written so loops over it vectorise (no libm call, branch-free selects).
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    let xc = x.clamp(-87.0, 88.0);
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
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    let v = y * scale;
    if x > 88.72 {
        f32::INFINITY
    } else if x < -87.33 {
        if x.is_nan() { x } else { 0.0 }
    } else if x.is_nan() {
        x
    } else {
        v
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
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
        Sigmoid => map1(x, lx, sigmoid),
        Silu => map1(x, lx, |v| v * sigmoid(v)),
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
        Rsqrt => map2(x, lx, g, lg, |x, g| g * -0.5 / (x * x.sqrt())),
        Recip => map2(x, lx, g, lg, |x, g| -g / (x * x)),
        Square => map2(x, lx, g, lg, |x, g| 2.0 * x * g),
        Sin => map2(x, lx, g, lg, |x, g| g * x.cos()),
        Cos => map2(x, lx, g, lg, |x, g| -g * x.sin()),
        Tanh => map2(x, lx, g, lg, |x, g| {
            let t = x.tanh();
            g * (1.0 - t * t)
        }),
        Sigmoid => map2(x, lx, g, lg, |x, g| {
            let s = sigmoid(x);
            g * s * (1.0 - s)
        }),
        Silu => map2(x, lx, g, lg, |x, g| {
            let s = sigmoid(x);
            g * (s + x * s * (1.0 - s))
        }),
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
        RDivScalar(c) => map2(x, lx, g, lg, move |x, g| -c * g / (x * x)),
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
        let mut x = -87.0f32;
        while x < 88.0 {
            let e = fast_exp(x) as f64;
            let t = (x as f64).exp();
            worst = worst.max(((e - t) / t).abs());
            x += 0.0137;
        }
        assert!(worst < 4e-7, "worst relative error {worst}");
        assert_eq!(fast_exp(100.0), f32::INFINITY);
        assert_eq!(fast_exp(-100.0), 0.0);
        assert!(fast_exp(f32::NAN).is_nan());
        assert_eq!(fast_exp(0.0), 1.0);
    }
}
