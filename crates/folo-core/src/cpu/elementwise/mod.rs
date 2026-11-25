//! Copies, casts and elementwise kernels.

mod simd;
mod simd_log_tanh;
#[cfg(test)]
mod tests;
mod transpose;

use super::pool_alloc::alloc_uninit;
use super::{CHUNK, PAR_MIN};
use crate::Layout;
use crate::backend::{BinaryOp, UnaryOp};
use crate::cpu::{par_chunks, par_for};
use crate::layout::RowPlan;

pub use simd::SILU_MAX_REL_ERR;
pub(crate) use simd::{exp_scalar, silu_bias_inplace};
#[cfg(target_arch = "x86_64")]
pub(crate) use simd::{exp8, silu8};
pub(crate) use transpose::{
    copy_strided_f32, copy_strided_f32_into, copy_strided_f32_into_slice, transpose_f32_2d,
    transpose_f32_2d_serial,
};

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
    if layout.numel() == 0 {
        return Vec::new();
    }
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
    assert_eq!(out.len(), layout.numel(), "copy_strided_into: length mismatch");
    if out.is_empty() {
        return out;
    }
    if let Some(r) = layout.contiguous_range() {
        let s = &src[r];
        if out.len() >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &|c, chunk| {
                let start = c * CHUNK;
                chunk.copy_from_slice(&s[start..start + chunk.len()]);
            });
        } else {
            out.copy_from_slice(s);
        }
        return out;
    }
    let plan = RowPlan::<1>::new(layout.dims(), [layout.strides()], [layout.offset()]);
    let s = plan.inner[0];
    fill_rows(&mut out, &plan, |row, [o]| {
        if s == 1 {
            row.copy_from_slice(&src[o..o + row.len()]);
        } else if s == 0 {
            row.fill(src[o]);
        } else {
            for (i, v) in row.iter_mut().enumerate() {
                *v = src[o + i * s];
            }
        }
    });
    out
}

/// Dispatch a two-operand row kernel over `plan` serially or via rayon, guaranteeing `from < to` on every call.
#[inline]
fn run_strided_rows(plan: &RowPlan<2>, row: impl Fn([usize; 2], usize, usize) + Sync) {
    let len = plan.len;
    if plan.rows == 0 || len == 0 {
        return;
    }
    let run = |start: usize, end: usize| plan.for_rows(start..end, |_, o| row(o, 0, len));
    if plan.rows * len < PAR_MIN {
        run(0, plan.rows);
    } else if plan.rows == 1 {
        let o = plan.base;
        par_for(len.div_ceil(CHUNK), &|t| row(o, t * CHUNK, ((t + 1) * CHUNK).min(len)));
    } else {
        let rows_per_task = (CHUNK / len).max(1);
        par_for(plan.rows.div_ceil(rows_per_task), &|t| {
            let start = t * rows_per_task;
            run(start, (start + rows_per_task).min(plan.rows));
        });
    }
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
    assert_eq!(dst_layout.shape(), src_layout.shape(), "strided write: shape mismatch");
    assert!(dst_layout.fits(dst.len()), "strided write: destination layout out of bounds");
    assert!(src_layout.fits(src.len()), "strided write: source layout out of bounds");
    assert!(dst_layout.is_non_overlapping(), "strided write: destination layout overlaps itself");
    let n = dst_layout.numel();
    if n == 0 {
        return;
    }
    if let (Some(rd), Some(rs)) = (dst_layout.contiguous_range(), src_layout.contiguous_range()) {
        let (d, s) = (&mut dst[rd], &src[rs]);
        if n >= PAR_MIN {
            par_chunks(d, CHUNK, &|c, chunk| {
                let start = c * CHUNK;
                chunk.copy_from_slice(&s[start..start + chunk.len()]);
            });
        } else {
            d.copy_from_slice(s);
        }
        return;
    }
    let plan = RowPlan::<2>::new(
        dst_layout.dims(),
        [dst_layout.strides(), src_layout.strides()],
        [dst_layout.offset(), src_layout.offset()],
    );
    let [ds, ss] = plan.inner;
    let base = SendPtr(dst.as_mut_ptr());
    run_strided_rows(&plan, |o, from, to| {
        let p = &base;
        match (ds, ss) {
            (1, 1) => {
                // SAFETY: `dst_layout` lies inside `dst` and is injective (asserted above), `ds == 1`, and
                // every task owns disjoint `(row, from..to)` elements.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                d.copy_from_slice(&src[o[1] + from..o[1] + to]);
            }
            (1, 0) => {
                // SAFETY: same disjointness and bounds guarantee as `(1, 1)`.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                d.fill(src[o[1]]);
            }
            (_, 0) => {
                let sv = src[o[1]];
                for i in from..to {
                    // SAFETY: `dst_layout` lies inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe { *p.0.add(o[0] + i * ds) = sv };
                }
            }
            _ => {
                for i in from..to {
                    let sv = src[o[1] + i * ss];
                    // SAFETY: `dst_layout` lies inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe { *p.0.add(o[0] + i * ds) = sv };
                }
            }
        }
    });
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
    let n = dst_layout.numel();
    if n == 0 {
        return;
    }
    if let (Some(rd), Some(rs)) = (dst_layout.contiguous_range(), src_layout.contiguous_range()) {
        let (d, s) = (&mut dst[rd], &src[rs]);
        let piece = |c: usize, chunk: &mut [T]| {
            let start = c * CHUNK;
            let len = chunk.len();
            for (dv, &sv) in chunk.iter_mut().zip(&s[start..start + len]) {
                *dv = f(*dv, sv);
            }
        };
        if n >= PAR_MIN {
            par_chunks(d, CHUNK, &piece);
        } else {
            piece(0, d);
        }
        return;
    }
    let plan = RowPlan::<2>::new(
        dst_layout.dims(),
        [dst_layout.strides(), src_layout.strides()],
        [dst_layout.offset(), src_layout.offset()],
    );
    let [ds, ss] = plan.inner;
    let base = SendPtr(dst.as_mut_ptr());
    run_strided_rows(&plan, |o, from, to| {
        let p = &base;
        match (ds, ss) {
            (1, 1) => {
                // SAFETY: `dst_layout` lies inside `dst` and is injective (asserted above), `ds == 1`, and
                // every task owns disjoint `(row, from..to)` elements.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                for (dv, &sv) in d.iter_mut().zip(&src[o[1] + from..o[1] + to]) {
                    *dv = f(*dv, sv);
                }
            }
            (1, 0) => {
                // SAFETY: same disjointness and bounds guarantee as `(1, 1)`.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                let sv = src[o[1]];
                for dv in d.iter_mut() {
                    *dv = f(*dv, sv);
                }
            }
            (_, 0) => {
                let sv = src[o[1]];
                for i in from..to {
                    // SAFETY: `dst_layout` lies inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe {
                        let d = p.0.add(o[0] + i * ds);
                        *d = f(*d, sv);
                    }
                }
            }
            _ => {
                for i in from..to {
                    let sv = src[o[1] + i * ss];
                    // SAFETY: `dst_layout` lies inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe {
                        let d = p.0.add(o[0] + i * ds);
                        *d = f(*d, sv);
                    }
                }
            }
        }
    });
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

/// Map one strided `f32` operand using a vectorised contiguous/unit-stride slice kernel and its scalar twin
/// for non-unit strides.
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

/// Map two strided `f32` operands of the form `b * deriv(a)` using vectorised contiguous/unit-stride slice
/// kernels (both `(1, 1)` and `(1, 0)` broadcast-`b`, common in `sum_all`/`mean_all` backwards), hoisting
/// `deriv(a)` out of `(0, 1)` broadcast-`a` rows, and using the scalar twin on general strides.
#[inline]
fn map2_slice(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    slice_fn: fn(&[f32], &[f32], &mut [f32]),
    bc_b_fn: fn(&[f32], f32, &mut [f32]),
    deriv_scalar: fn(f32) -> f32,
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
                let d = deriv_scalar(a[oa]);
                for (r, &bv) in row.iter_mut().zip(&b[ob..ob + len]) {
                    *r = bv * d;
                }
            }
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = b[ob + i * sb] * deriv_scalar(a[oa + i * sa]);
                }
            }
        }
    });
    out
}

/// Map two strided `f32` operands of the form `bwd_scalar(a, b)` using vectorised `(1, 1)` and `(1, 0)` slice
/// kernels and `bwd_scalar` directly on `(0, 1)` / general strides (so ops like `g / x` avoid `g * (1 / x)`).
#[inline]
fn map2_exact_slice(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    slice_fn: fn(&[f32], &[f32], &mut [f32]),
    bc_b_fn: fn(&[f32], f32, &mut [f32]),
    bwd_scalar: fn(f32, f32) -> f32,
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
                    *r = bwd_scalar(av, bv);
                }
            }
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = bwd_scalar(a[oa + i * sa], b[ob + i * sb]);
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
    let n = la.numel();
    if n == 0 {
        return Vec::new();
    }
    let mut out = alloc_uninit(n);
    if let (Some(rc), Some(ra), Some(rb)) =
        (lc.contiguous_range(), la.contiguous_range(), lb.contiguous_range())
    {
        let (c, a, b) = (&c[rc], &a[ra], &b[rb]);
        let piece = |k: usize, o: &mut [f32]| {
            let r = k * CHUNK..k * CHUNK + o.len();
            for (((o, &cv), &av), &bv) in o.iter_mut().zip(&c[r.clone()]).zip(&a[r.clone()]).zip(&b[r]) {
                *o = if truthy(cv) { av } else { bv };
            }
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan = RowPlan::<3>::new(
        la.dims(),
        [lc.strides(), la.strides(), lb.strides()],
        [lc.offset(), la.offset(), lb.offset()],
    );
    let [sc, sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oc, oa, ob]| {
        let len = row.len();
        match (sc, sa, sb) {
            (1, 1, 1) => {
                for (((r, &cv), &av), &bv) in
                    row.iter_mut().zip(&c[oc..oc + len]).zip(&a[oa..oa + len]).zip(&b[ob..ob + len])
                {
                    *r = if truthy(cv) { av } else { bv };
                }
            }
            (1, 1, 0) => {
                let bv = b[ob];
                for ((r, &cv), &av) in row.iter_mut().zip(&c[oc..oc + len]).zip(&a[oa..oa + len]) {
                    *r = if truthy(cv) { av } else { bv };
                }
            }
            (1, 0, 1) => {
                let av = a[oa];
                for ((r, &cv), &bv) in row.iter_mut().zip(&c[oc..oc + len]).zip(&b[ob..ob + len]) {
                    *r = if truthy(cv) { av } else { bv };
                }
            }
            (1, 0, 0) => {
                let (av, bv) = (a[oa], b[ob]);
                for (r, &cv) in row.iter_mut().zip(&c[oc..oc + len]) {
                    *r = if truthy(cv) { av } else { bv };
                }
            }
            (0, 1, 1) => {
                if truthy(c[oc]) {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 1, 0) => {
                if truthy(c[oc]) {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.fill(b[ob]);
                }
            }
            (0, 0, 1) => {
                if truthy(c[oc]) {
                    row.fill(a[oa]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 0, 0) => {
                row.fill(if truthy(c[oc]) { a[oa] } else { b[ob] });
            }
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = if truthy(c[oc + i * sc]) { a[oa + i * sa] } else { b[ob + i * sb] };
                }
            }
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

/// `x · sigmoid(x)`: the CPU's SiLU, element by element, bitwise identical to `Tensor::silu` on the CPU for
/// all non-NaN inputs (NaN stays NaN).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    simd::silu_scalar(x)
}

/// `g · silu'(x)`: bitwise identical to the CPU backward of `Tensor::silu` for all non-NaN inputs.
#[inline(always)]
pub fn silu_backward(x: f32, g: f32) -> f32 {
    simd::silu_bwd_scalar(x, g)
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
        Some(UnaryOp::Tanh) => simd_log_tanh::tanh_bias_inplace(v, bias),
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
        Log => map1_slice(x, lx, simd_log_tanh::ln_slice, simd_log_tanh::ln_scalar),
        Sqrt => map1(x, lx, f32::sqrt),
        Rsqrt => map1(x, lx, |v| 1.0 / v.sqrt()),
        Recip => map1(x, lx, |v| 1.0 / v),
        Square => map1(x, lx, |v| v * v),
        Sin => map1(x, lx, f32::sin),
        Cos => map1(x, lx, f32::cos),
        Tanh => map1_slice(x, lx, simd_log_tanh::tanh_slice, simd_log_tanh::tanh_scalar),
        Atan => map1(x, lx, f32::atan),
        Asin => map1(x, lx, f32::asin),
        Acos => map1(x, lx, f32::acos),
        Erf => map1(x, lx, erf),
        Sigmoid => map1_slice(x, lx, simd::sigmoid_slice, simd::sigmoid_scalar),
        Silu => map1_slice(x, lx, simd::silu_slice, simd::silu_scalar),
        Relu => map1(x, lx, |v| v.max(0.0)),
        LeakyRelu(a) => map1(x, lx, move |v| if v > 0.0 { v } else { a * v }),
        Hardswish => map1(x, lx, |v| v * (v + 3.0).clamp(0.0, 6.0) * (1.0 / 6.0)),
        Softplus => map1_slice(x, lx, simd_log_tanh::softplus_slice, simd_log_tanh::softplus_scalar),
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
        Exp => map2_slice(x, lx, g, lg, simd::exp_bwd_slice, simd::exp_bwd_broadcast_g, simd::exp_scalar),
        Log => map2_exact_slice(
            x,
            lx,
            g,
            lg,
            simd_log_tanh::log_bwd_slice,
            simd_log_tanh::log_bwd_broadcast_g,
            simd_log_tanh::log_bwd_scalar,
        ),
        Sqrt => map2(x, lx, g, lg, |x, g| g * 0.5 / x.sqrt()),
        // Divide in stages rather than squaring x, which under/overflows for representable results.
        Rsqrt => map2(x, lx, g, lg, |x, g| -0.5 * g / x.sqrt() / x),
        Recip => map2(x, lx, g, lg, |x, g| -(g / x) / x),
        Square => map2(x, lx, g, lg, |x, g| 2.0 * x * g),
        Sin => map2(x, lx, g, lg, |x, g| g * x.cos()),
        Cos => map2(x, lx, g, lg, |x, g| -g * x.sin()),
        Tanh => map2_slice(
            x,
            lx,
            g,
            lg,
            simd_log_tanh::tanh_bwd_slice,
            simd_log_tanh::tanh_bwd_broadcast_g,
            simd_log_tanh::tanh_deriv_scalar,
        ),
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
            simd::sigmoid_deriv_scalar,
        ),
        Silu => map2_slice(
            x,
            lx,
            g,
            lg,
            simd::silu_bwd_slice,
            simd::silu_bwd_broadcast_g,
            simd::silu_deriv_scalar,
        ),
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
            simd::sigmoid_scalar,
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
