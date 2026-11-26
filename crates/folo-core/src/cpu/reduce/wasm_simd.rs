//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) reductions and `ArgMax`/`ArgMin` kernels, bitwise identical
//! to their scalar twins in `reduce/mod.rs` and `reduce/arg.rs`.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::arg::{ARG_COL_TILE, better};
use super::{combine, identity, minmax_slice_scalar};
use crate::backend::ReduceOp;

/// 16-lane Wasm SIMD128 sum over a contiguous slice, bitwise identical to `sum_slice_scalar`.
pub(super) fn sum_slice_wasm(x: &[f32]) -> f32 {
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    // SAFETY: each chunk is `[f32; 16]` and `acc` is `[f32; 16]`.
    unsafe {
        for c in chunks {
            let p = c.as_ptr();
            a0 = f32x4_add(a0, v128_load(p as *const v128));
            a1 = f32x4_add(a1, v128_load(p.add(4) as *const v128));
            a2 = f32x4_add(a2, v128_load(p.add(8) as *const v128));
            a3 = f32x4_add(a3, v128_load(p.add(12) as *const v128));
        }
        v128_store(acc.as_mut_ptr() as *mut v128, a0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, a1);
        v128_store(acc.as_mut_ptr().add(8) as *mut v128, a2);
        v128_store(acc.as_mut_ptr().add(12) as *mut v128, a3);
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// Wasm SIMD128 column-wise sum `dst[i] += row[i]`, bitwise identical to the scalar loop.
pub(super) fn col_sum_wasm(dst: &mut [f32], row: &[f32]) {
    assert_eq!(dst.len(), row.len());
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    let (r8, r_rem) = row.as_chunks::<8>();
    for (dc, rc) in d8.iter_mut().zip(r8) {
        // SAFETY: `dc` and `rc` are `[f32; 8]`.
        unsafe {
            let dp = dc.as_mut_ptr();
            let rp = rc.as_ptr();
            v128_store(
                dp as *mut v128,
                f32x4_add(v128_load(dp as *const v128), v128_load(rp as *const v128)),
            );
            v128_store(
                dp.add(4) as *mut v128,
                f32x4_add(v128_load(dp.add(4) as *const v128), v128_load(rp.add(4) as *const v128)),
            );
        }
    }
    let (d4, d_tail) = d_rem.as_chunks_mut::<4>();
    let (r4, r_tail) = r_rem.as_chunks::<4>();
    for (dc, rc) in d4.iter_mut().zip(r4) {
        // SAFETY: `dc` and `rc` are `[f32; 4]`.
        unsafe {
            let dp = dc.as_mut_ptr();
            v128_store(
                dp as *mut v128,
                f32x4_add(v128_load(dp as *const v128), v128_load(rc.as_ptr() as *const v128)),
            );
        }
    }
    for (acc, &v) in d_tail.iter_mut().zip(r_tail) {
        *acc += v;
    }
}

/// 16-lane Wasm SIMD128 sum of squares `sum(x[i]^2)` (non-FMA), bitwise identical to `sum_sq_slice_scalar`.
pub(super) fn sum_sq_slice_wasm(x: &[f32]) -> f32 {
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    // SAFETY: each chunk is `[f32; 16]` and `acc` is `[f32; 16]`.
    unsafe {
        for c in chunks {
            let p = c.as_ptr();
            let v0 = v128_load(p as *const v128);
            let v1 = v128_load(p.add(4) as *const v128);
            let v2 = v128_load(p.add(8) as *const v128);
            let v3 = v128_load(p.add(12) as *const v128);
            a0 = f32x4_add(a0, f32x4_mul(v0, v0));
            a1 = f32x4_add(a1, f32x4_mul(v1, v1));
            a2 = f32x4_add(a2, f32x4_mul(v2, v2));
            a3 = f32x4_add(a3, f32x4_mul(v3, v3));
        }
        v128_store(acc.as_mut_ptr() as *mut v128, a0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, a1);
        v128_store(acc.as_mut_ptr().add(8) as *mut v128, a2);
        v128_store(acc.as_mut_ptr().add(12) as *mut v128, a3);
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += v * v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// 16-lane Wasm SIMD128 `sum((x[i] - mean)^2)` (non-FMA), bitwise identical to `sum_sq_diff_slice_scalar`.
pub(super) fn sum_sq_diff_slice_wasm(x: &[f32], mean: f32) -> f32 {
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [0f32; 16];
    let vm = f32x4_splat(mean);
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    // SAFETY: each chunk is `[f32; 16]` and `acc` is `[f32; 16]`.
    unsafe {
        for c in chunks {
            let p = c.as_ptr();
            let d0 = f32x4_sub(v128_load(p as *const v128), vm);
            let d1 = f32x4_sub(v128_load(p.add(4) as *const v128), vm);
            let d2 = f32x4_sub(v128_load(p.add(8) as *const v128), vm);
            let d3 = f32x4_sub(v128_load(p.add(12) as *const v128), vm);
            a0 = f32x4_add(a0, f32x4_mul(d0, d0));
            a1 = f32x4_add(a1, f32x4_mul(d1, d1));
            a2 = f32x4_add(a2, f32x4_mul(d2, d2));
            a3 = f32x4_add(a3, f32x4_mul(d3, d3));
        }
        v128_store(acc.as_mut_ptr() as *mut v128, a0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, a1);
        v128_store(acc.as_mut_ptr().add(8) as *mut v128, a2);
        v128_store(acc.as_mut_ptr().add(12) as *mut v128, a3);
    }
    for (i, &v) in rest.iter().enumerate() {
        let d = v - mean;
        acc[i] += d * d;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// Wasm SIMD128 column-wise squared deviation update `dst[i] += (row[i] - mean[i])^2`.
pub(super) fn col_sq_diff_wasm(dst: &mut [f32], row: &[f32], mean: &[f32]) {
    assert_eq!(dst.len(), row.len());
    assert_eq!(dst.len(), mean.len());
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    let (r8, r_rem) = row.as_chunks::<8>();
    let (m8, m_rem) = mean.as_chunks::<8>();
    for ((dc, rc), mc) in d8.iter_mut().zip(r8).zip(m8) {
        // SAFETY: all chunks are `[f32; 8]`.
        unsafe {
            let dp = dc.as_mut_ptr();
            let rp = rc.as_ptr();
            let mp = mc.as_ptr();
            let df0 = f32x4_sub(v128_load(rp as *const v128), v128_load(mp as *const v128));
            let df1 = f32x4_sub(v128_load(rp.add(4) as *const v128), v128_load(mp.add(4) as *const v128));
            v128_store(dp as *mut v128, f32x4_add(v128_load(dp as *const v128), f32x4_mul(df0, df0)));
            v128_store(
                dp.add(4) as *mut v128,
                f32x4_add(v128_load(dp.add(4) as *const v128), f32x4_mul(df1, df1)),
            );
        }
    }
    let (d4, d_tail) = d_rem.as_chunks_mut::<4>();
    let (r4, r_tail) = r_rem.as_chunks::<4>();
    let (m4, m_tail) = m_rem.as_chunks::<4>();
    for ((dc, rc), mc) in d4.iter_mut().zip(r4).zip(m4) {
        // SAFETY: all chunks are `[f32; 4]`.
        unsafe {
            let dp = dc.as_mut_ptr();
            let df = f32x4_sub(v128_load(rc.as_ptr() as *const v128), v128_load(mc.as_ptr() as *const v128));
            v128_store(dp as *mut v128, f32x4_add(v128_load(dp as *const v128), f32x4_mul(df, df)));
        }
    }
    for ((acc, &v), &m) in d_tail.iter_mut().zip(r_tail).zip(m_tail) {
        let df = v - m;
        *acc += df * df;
    }
}

/// 16-lane Wasm SIMD128 Max/Min over a contiguous slice, bitwise identical to `minmax_slice_scalar`.
///
/// `f32x4_pmax(acc, v)` (`acc < v ? v : acc`) and `f32x4_pmin(acc, v)` (`v < acc ? v : acc`) match
/// `combine(op, acc, v)` on every non-NaN pair (including `-0.0` vs `+0.0`). Any NaN lane triggers a
/// fallback to `minmax_slice_scalar` so NaN propagation and payload match bit-for-bit.
pub(super) fn minmax_slice_wasm(op: ReduceOp, x: &[f32]) -> f32 {
    let max = op == ReduceOp::Max;
    let id = identity(op);
    let (chunks, rest) = x.as_chunks::<16>();
    let mut acc = [id; 16];
    let vid = f32x4_splat(id);
    let (mut a0, mut a1, mut a2, mut a3) = (vid, vid, vid, vid);
    let mut any_nan = i32x4_splat(0);
    // SAFETY: each chunk is `[f32; 16]` and `acc` is `[f32; 16]`.
    unsafe {
        for c in chunks {
            let p = c.as_ptr();
            let v0 = v128_load(p as *const v128);
            let v1 = v128_load(p.add(4) as *const v128);
            let v2 = v128_load(p.add(8) as *const v128);
            let v3 = v128_load(p.add(12) as *const v128);
            any_nan = v128_or(
                any_nan,
                v128_or(
                    v128_or(f32x4_ne(v0, v0), f32x4_ne(v1, v1)),
                    v128_or(f32x4_ne(v2, v2), f32x4_ne(v3, v3)),
                ),
            );
            if max {
                a0 = f32x4_pmax(a0, v0);
                a1 = f32x4_pmax(a1, v1);
                a2 = f32x4_pmax(a2, v2);
                a3 = f32x4_pmax(a3, v3);
            } else {
                a0 = f32x4_pmin(a0, v0);
                a1 = f32x4_pmin(a1, v1);
                a2 = f32x4_pmin(a2, v2);
                a3 = f32x4_pmin(a3, v3);
            }
        }
        if v128_any_true(any_nan) {
            return minmax_slice_scalar(op, x);
        }
        v128_store(acc.as_mut_ptr() as *mut v128, a0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, a1);
        v128_store(acc.as_mut_ptr().add(8) as *mut v128, a2);
        v128_store(acc.as_mut_ptr().add(12) as *mut v128, a3);
    }
    let mut res = id;
    for a in acc {
        res = combine(op, res, a);
    }
    for &v in rest {
        res = combine(op, res, v);
    }
    res
}

/// Branchless Wasm SIMD128 column-wise Max/Min update `dst[i] = combine(op, dst[i], row[i])`.
pub(super) fn col_minmax_wasm(op: ReduceOp, dst: &mut [f32], row: &[f32]) {
    assert_eq!(dst.len(), row.len());
    let max = op == ReduceOp::Max;
    let (d_chunks, d_rem) = dst.as_chunks_mut::<4>();
    let (r_chunks, r_rem) = row.as_chunks::<4>();
    for (dc, rc) in d_chunks.iter_mut().zip(r_chunks) {
        // SAFETY: `dc` and `rc` are `[f32; 4]`.
        unsafe {
            let a = v128_load(dc.as_ptr() as *const v128);
            let v = v128_load(rc.as_ptr() as *const v128);
            let m = if max { f32x4_pmax(a, v) } else { f32x4_pmin(a, v) };
            let v_nan = f32x4_ne(v, v);
            v128_store(dc.as_mut_ptr() as *mut v128, v128_bitselect(v, m, v_nan));
        }
    }
    for (acc, &v) in d_rem.iter_mut().zip(r_rem) {
        *acc = combine(op, *acc, v);
    }
}

#[inline(always)]
fn arg_take4(max: bool, v: v128, b: v128) -> v128 {
    let cmp = if max { f32x4_gt(v, b) } else { f32x4_lt(v, b) };
    v128_and(f32x4_eq(b, b), v128_or(cmp, f32x4_ne(v, v)))
}

/// 16-lane Wasm SIMD128 row `ArgMax` / `ArgMin` (`inner == 1`), bitwise identical to `arg_row_scalar`.
pub(super) fn arg_row_wasm(max: bool, row: &[f32]) -> i64 {
    debug_assert!(row.len() >= 16 && row.len() <= i32::MAX as usize);
    let (chunks, tail) = row.as_chunks::<16>();
    let mut best_vals = [0f32; 16];
    let mut best_idxs = [0i32; 16];
    // SAFETY: `chunks` is non-empty (`row.len() >= 16`) and each chunk is `[f32; 16]`.
    unsafe {
        let first = chunks[0].as_ptr();
        let mut b0 = v128_load(first as *const v128);
        let mut b1 = v128_load(first.add(4) as *const v128);
        let mut b2 = v128_load(first.add(8) as *const v128);
        let mut b3 = v128_load(first.add(12) as *const v128);
        let mut i0 = i32x4(0, 1, 2, 3);
        let mut i1 = i32x4(4, 5, 6, 7);
        let mut i2 = i32x4(8, 9, 10, 11);
        let mut i3 = i32x4(12, 13, 14, 15);
        let step = i32x4_splat(16);
        let mut c0 = i32x4_add(i0, step);
        let mut c1 = i32x4_add(i1, step);
        let mut c2 = i32x4_add(i2, step);
        let mut c3 = i32x4_add(i3, step);
        for ch in &chunks[1..] {
            let p = ch.as_ptr();
            let v0 = v128_load(p as *const v128);
            let v1 = v128_load(p.add(4) as *const v128);
            let v2 = v128_load(p.add(8) as *const v128);
            let v3 = v128_load(p.add(12) as *const v128);
            let t0 = arg_take4(max, v0, b0);
            let t1 = arg_take4(max, v1, b1);
            let t2 = arg_take4(max, v2, b2);
            let t3 = arg_take4(max, v3, b3);
            b0 = v128_bitselect(v0, b0, t0);
            b1 = v128_bitselect(v1, b1, t1);
            b2 = v128_bitselect(v2, b2, t2);
            b3 = v128_bitselect(v3, b3, t3);
            i0 = v128_bitselect(c0, i0, t0);
            i1 = v128_bitselect(c1, i1, t1);
            i2 = v128_bitselect(c2, i2, t2);
            i3 = v128_bitselect(c3, i3, t3);
            c0 = i32x4_add(c0, step);
            c1 = i32x4_add(c1, step);
            c2 = i32x4_add(c2, step);
            c3 = i32x4_add(c3, step);
        }
        v128_store(best_vals.as_mut_ptr() as *mut v128, b0);
        v128_store(best_vals.as_mut_ptr().add(4) as *mut v128, b1);
        v128_store(best_vals.as_mut_ptr().add(8) as *mut v128, b2);
        v128_store(best_vals.as_mut_ptr().add(12) as *mut v128, b3);
        v128_store(best_idxs.as_mut_ptr() as *mut v128, i0);
        v128_store(best_idxs.as_mut_ptr().add(4) as *mut v128, i1);
        v128_store(best_idxs.as_mut_ptr().add(8) as *mut v128, i2);
        v128_store(best_idxs.as_mut_ptr().add(12) as *mut v128, i3);
    }
    let mut best = best_vals[0];
    let mut bi = best_idxs[0] as usize;
    for k in 1..16 {
        let v = best_vals[k];
        let idx = best_idxs[k] as usize;
        let wins = if best.is_nan() {
            v.is_nan() && idx < bi
        } else if v.is_nan() {
            true
        } else if max {
            v > best || (v == best && idx < bi)
        } else {
            v < best || (v == best && idx < bi)
        };
        if wins {
            best = v;
            bi = idx;
        }
    }
    let base = chunks.len() * 16;
    for (off, &v) in tail.iter().enumerate() {
        if better(max, v, best) {
            best = v;
            bi = base + off;
        }
    }
    bi as i64
}

/// 8-wide Wasm SIMD128 column-tiled `ArgMax` / `ArgMin` for `tw <= ARG_COL_TILE` columns.
pub(super) fn arg_col_tile_wasm(max: bool, src: &[f32], dst: &mut [i64], d: usize, inner: usize, c0: usize) {
    let tw = dst.len();
    debug_assert!(tw <= ARG_COL_TILE && d <= i32::MAX as usize);
    let mut best_v = [0f32; ARG_COL_TILE];
    let mut best_i = [0i32; ARG_COL_TILE];
    best_v[..tw].copy_from_slice(&src[c0..c0 + tw]);

    let v8 = tw / 8 * 8;
    let v4 = tw / 4 * 4;
    for r in 1..d {
        let row = &src[r * inner + c0..r * inner + c0 + tw];
        let vr = i32x4_splat(r as i32);
        let mut j = 0;
        // SAFETY: `j + 8 <= v8 <= tw <= ARG_COL_TILE` and `j + 4 <= v4 <= tw`.
        unsafe {
            while j < v8 {
                let rp = row.as_ptr().add(j);
                let bp = best_v.as_mut_ptr().add(j);
                let ip = best_i.as_mut_ptr().add(j);
                let v0 = v128_load(rp as *const v128);
                let v1 = v128_load(rp.add(4) as *const v128);
                let b0 = v128_load(bp as *const v128);
                let b1 = v128_load(bp.add(4) as *const v128);
                let t0 = arg_take4(max, v0, b0);
                let t1 = arg_take4(max, v1, b1);
                v128_store(bp as *mut v128, v128_bitselect(v0, b0, t0));
                v128_store(bp.add(4) as *mut v128, v128_bitselect(v1, b1, t1));
                let i0 = v128_load(ip as *const v128);
                let i1 = v128_load(ip.add(4) as *const v128);
                v128_store(ip as *mut v128, v128_bitselect(vr, i0, t0));
                v128_store(ip.add(4) as *mut v128, v128_bitselect(vr, i1, t1));
                j += 8;
            }
            if j < v4 {
                let v0 = v128_load(row.as_ptr().add(j) as *const v128);
                let bp = best_v.as_mut_ptr().add(j);
                let ip = best_i.as_mut_ptr().add(j);
                let b0 = v128_load(bp as *const v128);
                let t0 = arg_take4(max, v0, b0);
                v128_store(bp as *mut v128, v128_bitselect(v0, b0, t0));
                v128_store(ip as *mut v128, v128_bitselect(vr, v128_load(ip as *const v128), t0));
            }
        }
        for jj in v4..tw {
            let v = row[jj];
            if better(max, v, best_v[jj]) {
                best_v[jj] = v;
                best_i[jj] = r as i32;
            }
        }
    }
    let mut j = 0;
    // SAFETY: `j + 4 <= v4 <= tw == dst.len()`.
    unsafe {
        while j < v4 {
            let idx4 = v128_load(best_i.as_ptr().add(j) as *const v128);
            let lo = i64x2_extend_low_i32x4(idx4);
            let hi = i64x2_extend_high_i32x4(idx4);
            v128_store(dst.as_mut_ptr().add(j) as *mut v128, lo);
            v128_store(dst.as_mut_ptr().add(j + 2) as *mut v128, hi);
            j += 4;
        }
    }
    for jj in v4..tw {
        dst[jj] = best_i[jj] as i64;
    }
}
