//! Conditional selection (`where_cond`) for `F32` and `U8` masks, with AVX2 (`x86_64`) and Wasm SIMD128
//! (`wasm32 +simd128`) fast paths.

#[cfg(any(not(all(target_arch = "wasm32", target_feature = "simd128")), test))]
use super::fill_rows;
use crate::Layout;
#[cfg(any(not(all(target_arch = "wasm32", target_feature = "simd128")), test))]
use crate::cpu::pool_alloc::alloc_uninit;
#[cfg(any(not(all(target_arch = "wasm32", target_feature = "simd128")), test))]
use crate::cpu::{CHUNK, PAR_MIN, par_chunks};
#[cfg(any(not(all(target_arch = "wasm32", target_feature = "simd128")), test))]
use crate::layout::RowPlan;

/// `cond != 0 ? a : b` with three strided operands of the output shape.
#[cfg(any(not(all(target_arch = "wasm32", target_feature = "simd128")), test))]
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

/// `cond != 0.0 ? a : b` for `F32` conditions, using AVX2 on `x86_64` or Wasm SIMD128 on `wasm32`.
pub(crate) fn where3_f32(c: &[f32], lc: &Layout, a: &[f32], la: &Layout, b: &[f32], lb: &Layout) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        super::simd_binary::where3_f32_avx2(c, lc, a, la, b, lb)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        super::wasm_ops::where3_f32_wasm(c, lc, a, la, b, lb)
    }
    #[cfg(not(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        where3(c, lc, a, la, b, lb, |v| v != 0.0)
    }
}

/// `cond != 0 ? a : b` for `U8` conditions, using AVX2 on `x86_64` or Wasm SIMD128 on `wasm32`.
pub(crate) fn where3_u8(c: &[u8], lc: &Layout, a: &[f32], la: &Layout, b: &[f32], lb: &Layout) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        super::simd_binary::where3_u8_avx2(c, lc, a, la, b, lb)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        super::wasm_ops::where3_u8_wasm(c, lc, a, la, b, lb)
    }
    #[cfg(not(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        where3(c, lc, a, la, b, lb, |v| v != 0)
    }
}
