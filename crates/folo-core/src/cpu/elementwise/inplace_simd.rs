//! SIMD-accelerated in-place broadcast and contiguous elementwise updates (`dst op= src`, `dst += alpha * src`,
//! `dst *= c`) for x86_64 AVX2 and WebAssembly SIMD128, bitwise identical to scalar IEEE-754 arithmetic.

use super::{SendPtr, run_strided_rows};
use crate::Layout;
use crate::cpu::{CHUNK, PAR_MIN, par_chunks};
use crate::layout::RowPlan;

/// In-place binary update applied by [`binary_inplace`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum InplaceOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `dst[i] += alpha * src[i]` (non-FMA, bitwise identical across AVX2, Wasm SIMD128, and scalar).
    AddScaled(f32),
}

#[inline(always)]
fn apply_scalar(op: InplaceOp, a: f32, b: f32) -> f32 {
    match op {
        InplaceOp::Add => a + b,
        InplaceOp::Sub => a - b,
        InplaceOp::Mul => a * b,
        InplaceOp::Div => a / b,
        InplaceOp::AddScaled(alpha) => a + alpha * b,
    }
}

#[inline]
fn update_slice_vv_scalar(op: InplaceOp, dst: &mut [f32], src: &[f32]) {
    match op {
        InplaceOp::Add => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }
        InplaceOp::Sub => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d -= s;
            }
        }
        InplaceOp::Mul => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d *= s;
            }
        }
        InplaceOp::Div => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d /= s;
            }
        }
        InplaceOp::AddScaled(alpha) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += alpha * s;
            }
        }
    }
}

#[inline]
fn update_slice_vs_scalar(op: InplaceOp, dst: &mut [f32], s: f32) {
    match op {
        InplaceOp::Add => {
            for d in dst.iter_mut() {
                *d += s;
            }
        }
        InplaceOp::Sub => {
            for d in dst.iter_mut() {
                *d -= s;
            }
        }
        InplaceOp::Mul => {
            for d in dst.iter_mut() {
                *d *= s;
            }
        }
        InplaceOp::Div => {
            for d in dst.iter_mut() {
                *d /= s;
            }
        }
        InplaceOp::AddScaled(alpha) => {
            let delta = alpha * s;
            for d in dst.iter_mut() {
                *d += delta;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{InplaceOp, update_slice_vs_scalar, update_slice_vv_scalar};

    macro_rules! avx2_inplace_vv {
        ($dst:ident, $src:ident, $vd:ident, $vs:ident => $vexpr:expr) => {{
            let (dp, dr) = $dst.as_chunks_mut::<16>();
            let (sp, sr) = $src.as_chunks::<16>();
            for (dc, sc) in dp.iter_mut().zip(sp) {
                // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let (dptr, sptr) = (dc.as_mut_ptr(), sc.as_ptr());
                    let ($vd, $vs) = (_mm256_loadu_ps(dptr), _mm256_loadu_ps(sptr));
                    let r0 = $vexpr;
                    let ($vd, $vs) = (_mm256_loadu_ps(dptr.add(8)), _mm256_loadu_ps(sptr.add(8)));
                    let r1 = $vexpr;
                    _mm256_storeu_ps(dptr, r0);
                    _mm256_storeu_ps(dptr.add(8), r1);
                }
            }
            let (d8, dt) = dr.as_chunks_mut::<8>();
            let (s8, st) = sr.as_chunks::<8>();
            for (dc, sc) in d8.iter_mut().zip(s8) {
                // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let ($vd, $vs) = (_mm256_loadu_ps(dc.as_ptr()), _mm256_loadu_ps(sc.as_ptr()));
                    _mm256_storeu_ps(dc.as_mut_ptr(), $vexpr);
                }
            }
            (dt, st)
        }};
    }

    macro_rules! avx2_inplace_vs {
        ($dst:ident, $vd:ident => $vexpr:expr) => {{
            let (dp, dr) = $dst.as_chunks_mut::<16>();
            for dc in dp {
                // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let dptr = dc.as_mut_ptr();
                    let $vd = _mm256_loadu_ps(dptr);
                    let r0 = $vexpr;
                    let $vd = _mm256_loadu_ps(dptr.add(8));
                    let r1 = $vexpr;
                    _mm256_storeu_ps(dptr, r0);
                    _mm256_storeu_ps(dptr.add(8), r1);
                }
            }
            let (d8, dt) = dr.as_chunks_mut::<8>();
            for dc in d8 {
                // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let $vd = _mm256_loadu_ps(dc.as_ptr());
                    _mm256_storeu_ps(dc.as_mut_ptr(), $vexpr);
                }
            }
            dt
        }};
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn update_vv_avx2(op: InplaceOp, dst: &mut [f32], src: &[f32]) {
        debug_assert_eq!(dst.len(), src.len());
        let (dt, st) = match op {
            InplaceOp::Add => avx2_inplace_vv!(dst, src, vd, vs => _mm256_add_ps(vd, vs)),
            InplaceOp::Sub => avx2_inplace_vv!(dst, src, vd, vs => _mm256_sub_ps(vd, vs)),
            InplaceOp::Mul => avx2_inplace_vv!(dst, src, vd, vs => _mm256_mul_ps(vd, vs)),
            InplaceOp::Div => avx2_inplace_vv!(dst, src, vd, vs => _mm256_div_ps(vd, vs)),
            InplaceOp::AddScaled(alpha) => {
                let va = _mm256_set1_ps(alpha);
                avx2_inplace_vv!(dst, src, vd, vs => _mm256_add_ps(vd, _mm256_mul_ps(va, vs)))
            }
        };
        if !dt.is_empty() {
            update_slice_vv_scalar(op, dt, st);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn update_vs_avx2(op: InplaceOp, dst: &mut [f32], s: f32) {
        let dt = match op {
            InplaceOp::Add => {
                let vs = _mm256_set1_ps(s);
                avx2_inplace_vs!(dst, vd => _mm256_add_ps(vd, vs))
            }
            InplaceOp::Sub => {
                let vs = _mm256_set1_ps(s);
                avx2_inplace_vs!(dst, vd => _mm256_sub_ps(vd, vs))
            }
            InplaceOp::Mul => {
                let vs = _mm256_set1_ps(s);
                avx2_inplace_vs!(dst, vd => _mm256_mul_ps(vd, vs))
            }
            InplaceOp::Div => {
                let vs = _mm256_set1_ps(s);
                avx2_inplace_vs!(dst, vd => _mm256_div_ps(vd, vs))
            }
            InplaceOp::AddScaled(alpha) => {
                let vdelta = _mm256_set1_ps(alpha * s);
                avx2_inplace_vs!(dst, vd => _mm256_add_ps(vd, vdelta))
            }
        };
        if !dt.is_empty() {
            update_slice_vs_scalar(op, dt, s);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    use super::{InplaceOp, update_slice_vs_scalar, update_slice_vv_scalar};

    #[inline]
    fn run_vv<'a, 'b>(
        dst: &'a mut [f32],
        src: &'b [f32],
        vop: impl Fn(v128, v128) -> v128,
    ) -> (&'a mut [f32], &'b [f32]) {
        let (dp, dr) = dst.as_chunks_mut::<8>();
        let (sp, sr) = src.as_chunks::<8>();
        for (dc, sc) in dp.iter_mut().zip(sp) {
            // SAFETY: 8-float chunks.
            unsafe {
                let (dptr, sptr) = (dc.as_mut_ptr(), sc.as_ptr());
                let r0 = vop(v128_load(dptr as *const v128), v128_load(sptr as *const v128));
                let r1 = vop(v128_load(dptr.add(4) as *const v128), v128_load(sptr.add(4) as *const v128));
                v128_store(dptr as *mut v128, r0);
                v128_store(dptr.add(4) as *mut v128, r1);
            }
        }
        let (d4, dt) = dr.as_chunks_mut::<4>();
        let (s4, st) = sr.as_chunks::<4>();
        for (dc, sc) in d4.iter_mut().zip(s4) {
            // SAFETY: 4-float chunks.
            unsafe {
                let r = vop(v128_load(dc.as_ptr() as *const v128), v128_load(sc.as_ptr() as *const v128));
                v128_store(dc.as_mut_ptr() as *mut v128, r);
            }
        }
        (dt, st)
    }

    #[inline]
    fn run_vs(dst: &mut [f32], vop: impl Fn(v128) -> v128) -> &mut [f32] {
        let (dp, dr) = dst.as_chunks_mut::<8>();
        for dc in dp {
            // SAFETY: 8-float chunks.
            unsafe {
                let dptr = dc.as_mut_ptr();
                let r0 = vop(v128_load(dptr as *const v128));
                let r1 = vop(v128_load(dptr.add(4) as *const v128));
                v128_store(dptr as *mut v128, r0);
                v128_store(dptr.add(4) as *mut v128, r1);
            }
        }
        let (d4, dt) = dr.as_chunks_mut::<4>();
        for dc in d4 {
            // SAFETY: 4-float chunks.
            unsafe {
                let r = vop(v128_load(dc.as_ptr() as *const v128));
                v128_store(dc.as_mut_ptr() as *mut v128, r);
            }
        }
        dt
    }

    pub(super) fn update_vv_wasm(op: InplaceOp, dst: &mut [f32], src: &[f32]) {
        debug_assert_eq!(dst.len(), src.len());
        let (dt, st) = match op {
            InplaceOp::Add => run_vv(dst, src, |d, s| f32x4_add(d, s)),
            InplaceOp::Sub => run_vv(dst, src, |d, s| f32x4_sub(d, s)),
            InplaceOp::Mul => run_vv(dst, src, |d, s| f32x4_mul(d, s)),
            InplaceOp::Div => run_vv(dst, src, |d, s| f32x4_div(d, s)),
            InplaceOp::AddScaled(alpha) => {
                let va = f32x4_splat(alpha);
                run_vv(dst, src, |d, s| f32x4_add(d, f32x4_mul(va, s)))
            }
        };
        if !dt.is_empty() {
            update_slice_vv_scalar(op, dt, st);
        }
    }

    pub(super) fn update_vs_wasm(op: InplaceOp, dst: &mut [f32], s: f32) {
        let dt = match op {
            InplaceOp::Add => {
                let vs = f32x4_splat(s);
                run_vs(dst, |d| f32x4_add(d, vs))
            }
            InplaceOp::Sub => {
                let vs = f32x4_splat(s);
                run_vs(dst, |d| f32x4_sub(d, vs))
            }
            InplaceOp::Mul => {
                let vs = f32x4_splat(s);
                run_vs(dst, |d| f32x4_mul(d, vs))
            }
            InplaceOp::Div => {
                let vs = f32x4_splat(s);
                run_vs(dst, |d| f32x4_div(d, vs))
            }
            InplaceOp::AddScaled(alpha) => {
                let vdelta = f32x4_splat(alpha * s);
                run_vs(dst, |d| f32x4_add(d, vdelta))
            }
        };
        if !dt.is_empty() {
            update_slice_vs_scalar(op, dt, s);
        }
    }
}

#[inline]
fn update_slice_vv(op: InplaceOp, dst: &mut [f32], src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && super::simd::has_avx2() {
        // SAFETY: AVX2 was detected above and `dst.len() == src.len()`.
        unsafe { avx2::update_vv_avx2(op, dst, src) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 {
        wasm::update_vv_wasm(op, dst, src);
        return;
    }
    update_slice_vv_scalar(op, dst, src);
}

#[inline]
fn update_slice_vs(op: InplaceOp, dst: &mut [f32], s: f32) {
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= 8 && super::simd::has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { avx2::update_vs_avx2(op, dst, s) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if dst.len() >= 4 {
        wasm::update_vs_wasm(op, dst, s);
        return;
    }
    update_slice_vs_scalar(op, dst, s);
}

/// SIMD-accelerated `dst[i] = op(dst[i], src[i])` through two layouts of the same shape.
///
/// # Panics
/// If the shapes differ, either layout reaches outside its buffer, or `dst_layout` overlaps itself.
pub(crate) fn binary_inplace(
    op: InplaceOp,
    dst: &mut [f32],
    dst_layout: &Layout,
    src: &[f32],
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
        let piece = |c: usize, chunk: &mut [f32]| {
            let start = c * CHUNK;
            update_slice_vv(op, chunk, &s[start..start + chunk.len()]);
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
                // SAFETY: `dst_layout` fits inside `dst` and is injective; every task owns disjoint `(row, from..to)`.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                update_slice_vv(op, d, &src[o[1] + from..o[1] + to]);
            }
            (1, 0) => {
                // SAFETY: same disjointness and bounds guarantee as `(1, 1)`.
                let d = unsafe { std::slice::from_raw_parts_mut(p.0.add(o[0] + from), to - from) };
                update_slice_vs(op, d, src[o[1]]);
            }
            (_, 0) => {
                let sv = src[o[1]];
                for i in from..to {
                    // SAFETY: `dst_layout` fits inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe {
                        let d = p.0.add(o[0] + i * ds);
                        *d = apply_scalar(op, *d, sv);
                    }
                }
            }
            _ => {
                for i in from..to {
                    let sv = src[o[1] + i * ss];
                    // SAFETY: `dst_layout` fits inside `dst` and is injective; each `(row, i)` is unique.
                    unsafe {
                        let d = p.0.add(o[0] + i * ds);
                        *d = apply_scalar(op, *d, sv);
                    }
                }
            }
        }
    });
}

/// SIMD-accelerated `dst[i] *= c` in place without allocating a temporary scalar tensor.
///
/// # Panics
/// If `dst_layout` reaches outside `dst` or overlaps itself.
pub(crate) fn mul_scalar_inplace(dst: &mut [f32], dst_layout: &Layout, c: f32) {
    assert!(dst_layout.fits(dst.len()), "strided write: destination layout out of bounds");
    assert!(dst_layout.is_non_overlapping(), "strided write: destination layout overlaps itself");
    let n = dst_layout.numel();
    if n == 0 {
        return;
    }
    if let Some(rd) = dst_layout.contiguous_range() {
        let d = &mut dst[rd];
        let piece = |_c: usize, chunk: &mut [f32]| update_slice_vs(InplaceOp::Mul, chunk, c);
        if n >= PAR_MIN {
            par_chunks(d, CHUNK, &piece);
        } else {
            piece(0, d);
        }
        return;
    }
    let scalar_layout =
        Layout::new(dst_layout.shape().clone(), crate::shape::DimVec::from_elem(0, dst_layout.rank()), 0);
    binary_inplace(InplaceOp::Mul, dst, dst_layout, &[c], &scalar_layout);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::{Shape, Tensor, no_grad};

    fn check_bits(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: len");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "{label}[{i}]: expected NaN, got {g}");
            } else {
                assert_eq!(g.to_bits(), w.to_bits(), "{label}[{i}]: {g} vs {w}");
            }
        }
    }

    #[test]
    fn binary_inplace_matches_scalar_across_lengths_and_layouts() {
        let mut rng = Rng::seed(2026);
        let ops = [
            InplaceOp::Add,
            InplaceOp::Sub,
            InplaceOp::Mul,
            InplaceOp::Div,
            InplaceOp::AddScaled(-0.075),
            InplaceOp::AddScaled(0.0),
        ];
        // Contiguous: odd lengths, vector boundaries, special values, and parallel size (>= PAR_MIN).
        for n in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 33, PAR_MIN + 19] {
            let mut a: Vec<f32> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
            let mut b: Vec<f32> = (0..n).map(|_| rng.uniform(-5.0, 5.0)).collect();
            if n >= 8 {
                a[0] = -0.0;
                b[0] = -0.0;
                a[1] = f32::INFINITY;
                b[1] = 2.0;
                a[2] = f32::NAN;
                b[3] = f32::NEG_INFINITY;
                a[4] = 1e-38;
                b[4] = 1e-38;
            }
            let l = Layout::contiguous(Shape::new(&[n]));
            for op in ops {
                let mut got = a.clone();
                let mut want = a.clone();
                binary_inplace(op, &mut got, &l, &b, &l);
                update_slice_vv_scalar(op, &mut want, &b);
                check_bits(&got, &want, &format!("contig n={n} {op:?}"));
            }
        }

        // Broadcast `(1, 0)` (row broadcast), `(1, 1)` (column broadcast), and non-contiguous dst `(ds > 1)`.
        let base_a: Vec<f32> = (0..12 * 37).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let row_b: Vec<f32> = (0..37).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let col_b: Vec<f32> = (0..12).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let l_full = Layout::contiguous(Shape::new(&[12, 37]));
        let l_row =
            Layout::contiguous(Shape::new(&[1, 37])).broadcast_to(&Shape::new(&[12, 37]), "t").unwrap();
        let l_col =
            Layout::contiguous(Shape::new(&[12, 1])).broadcast_to(&Shape::new(&[12, 37]), "t").unwrap();
        let l_t = l_full.transpose(0, 1);
        let l_t_col =
            Layout::contiguous(Shape::new(&[37, 1])).broadcast_to(&Shape::new(&[37, 12]), "t").unwrap();
        let l_t_row =
            Layout::contiguous(Shape::new(&[1, 12])).broadcast_to(&Shape::new(&[37, 12]), "t").unwrap();

        for op in ops {
            // (1, 1) broadcast rows
            let mut got = base_a.clone();
            let mut want = base_a.clone();
            binary_inplace(op, &mut got, &l_full, &row_b, &l_row);
            super::super::fill_rows_inplace(&mut want, &l_full, &row_b, &l_row, |x, y| {
                apply_scalar(op, x, y)
            });
            check_bits(&got, &want, &format!("bc_row {op:?}"));

            // (1, 0) broadcast col
            let mut got = base_a.clone();
            let mut want = base_a.clone();
            binary_inplace(op, &mut got, &l_full, &col_b, &l_col);
            super::super::fill_rows_inplace(&mut want, &l_full, &col_b, &l_col, |x, y| {
                apply_scalar(op, x, y)
            });
            check_bits(&got, &want, &format!("bc_col {op:?}"));

            // Non-unit dst stride `(ds, 0)` and `(ds, 1)`
            let mut got = base_a.clone();
            let mut want = base_a.clone();
            binary_inplace(op, &mut got, &l_t, &row_b, &l_t_col);
            super::super::fill_rows_inplace(&mut want, &l_t, &row_b, &l_t_col, |x, y| apply_scalar(op, x, y));
            check_bits(&got, &want, &format!("strided_ds0 {op:?}"));

            let mut got = base_a.clone();
            let mut want = base_a.clone();
            binary_inplace(op, &mut got, &l_t, &col_b, &l_t_row);
            super::super::fill_rows_inplace(&mut want, &l_t, &col_b, &l_t_row, |x, y| apply_scalar(op, x, y));
            check_bits(&got, &want, &format!("strided_ds1 {op:?}"));
        }

        // `mul_scalar_inplace` through Tensor API (contiguous + transposed strided view).
        let t = Tensor::from_slice(&base_a, [12, 37]).unwrap();
        let mut want = base_a.clone();
        update_slice_vs_scalar(InplaceOp::Mul, &mut want, -0.375);
        {
            let _ng = no_grad();
            t.mul_scalar_assign(-0.375).unwrap();
        }
        check_bits(&t.to_vec::<f32>().unwrap(), &want, "mul_scalar_assign");
    }
}
