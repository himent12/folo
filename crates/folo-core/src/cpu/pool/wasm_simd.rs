//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) pooling and 2x2 nearest-upsampling primitives, bitwise
//! identical to their scalar and AVX2 counterparts (including row-major first-maximum / first-NaN tie rules
//! and `-0.0` vs `+0.0` sign bits).

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::max_avx2::{step_idx, step_val};

/// 4-lane `step_val`: returns `a` where `a > b || (a.is_nan() && !b.is_nan())`, else `b`.
#[inline(always)]
fn max_step4(b: v128, a: v128) -> v128 {
    let nle = v128_not(f32x4_le(a, b));
    let b_ord = f32x4_eq(b, b);
    v128_bitselect(a, b, v128_and(nle, b_ord))
}

/// 4-lane `step_idx`: returns updated `(best, bi)` for valid `ai >= 0`.
#[inline(always)]
fn beats_indexed4(b: v128, bi: v128, a: v128, ai: v128, neg1: v128, zero: v128) -> (v128, v128) {
    let a_valid = i32x4_gt(ai, neg1);
    let b_empty = i32x4_gt(zero, bi);
    let val_beats = v128_and(v128_not(f32x4_le(a, b)), f32x4_eq(b, b));
    let take_a = v128_and(a_valid, v128_or(b_empty, val_beats));
    (v128_bitselect(a, b, take_a), v128_bitselect(ai, bi, take_a))
}

pub(super) fn row_add_wasm(dst: &mut [f32], src: &[f32]) {
    let (dc, dr) = dst.as_chunks_mut::<4>();
    let (sc, sr) = src.as_chunks::<4>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `d` and `s` are 4 floats.
        unsafe {
            let v = f32x4_add(v128_load(d.as_ptr() as *const v128), v128_load(s.as_ptr() as *const v128));
            v128_store(d.as_mut_ptr() as *mut v128, v);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d += s;
    }
}

pub(super) fn row_add_scaled_wasm(dst: &mut [f32], src: &[f32], scale: f32) {
    let sv = f32x4_splat(scale);
    let (dc, dr) = dst.as_chunks_mut::<4>();
    let (sc, sr) = src.as_chunks::<4>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `d` and `s` are 4 floats.
        unsafe {
            let v = f32x4_add(
                v128_load(d.as_ptr() as *const v128),
                f32x4_mul(v128_load(s.as_ptr() as *const v128), sv),
            );
            v128_store(d.as_mut_ptr() as *mut v128, v);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d += s * scale;
    }
}

pub(super) fn row_scale_wasm(dst: &mut [f32], scale: f32) {
    let sv = f32x4_splat(scale);
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for d in dc {
        // SAFETY: `d` is 4 floats.
        unsafe {
            v128_store(d.as_mut_ptr() as *mut v128, f32x4_mul(v128_load(d.as_ptr() as *const v128), sv));
        }
    }
    for d in dr {
        *d *= scale;
    }
}

pub(super) fn row_add_scalar_wasm(dst: &mut [f32], v: f32) {
    let vv = f32x4_splat(v);
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for d in dc {
        // SAFETY: `d` is 4 floats.
        unsafe {
            v128_store(d.as_mut_ptr() as *mut v128, f32x4_add(v128_load(d.as_ptr() as *const v128), vv));
        }
    }
    for d in dr {
        *d += v;
    }
}

pub(super) fn sum_pairs_wasm(src: &[f32], dst: &mut [f32]) {
    let n = dst.len();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    let (sc, sr) = src[..2 * n].as_chunks::<8>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `s` is 8 floats and `d` is 4 floats.
        unsafe {
            let v0 = v128_load(s.as_ptr() as *const v128);
            let v1 = v128_load(s.as_ptr().add(4) as *const v128);
            let ev = u32x4_shuffle::<0, 2, 4, 6>(v0, v1);
            let od = u32x4_shuffle::<1, 3, 5, 7>(v0, v1);
            v128_store(d.as_mut_ptr() as *mut v128, f32x4_add(ev, od));
        }
    }
    for (d, pair) in dr.iter_mut().zip(sr.as_chunks::<2>().0) {
        *d = pair[0] + pair[1];
    }
}

pub(super) fn max_row_s1_wasm(row: &[f32], kw: usize, dst: &mut [f32]) {
    let ow = dst.len();
    let mut ox = 0;
    // SAFETY: `ox + 4 <= ow` and `row.len() >= ow + kw - 1`.
    unsafe {
        while ox + 4 <= ow {
            let base = row.as_ptr().add(ox);
            let mut m = v128_load(base as *const v128);
            for t in 1..kw {
                m = max_step4(m, v128_load(base.add(t) as *const v128));
            }
            v128_store(dst.as_mut_ptr().add(ox) as *mut v128, m);
            ox += 4;
        }
    }
    for (j, d) in dst[ox..].iter_mut().enumerate() {
        let mut m = row[ox + j];
        for t in 1..kw {
            step_val(&mut m, row[ox + j + t]);
        }
        *d = m;
    }
}

pub(super) fn max_pairs_wasm(row: &[f32], dst: &mut [f32]) {
    let n = dst.len();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    let (sc, sr) = row[..2 * n].as_chunks::<8>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `s` is 8 floats and `d` is 4 floats.
        unsafe {
            let v0 = v128_load(s.as_ptr() as *const v128);
            let v1 = v128_load(s.as_ptr().add(4) as *const v128);
            let ev = u32x4_shuffle::<0, 2, 4, 6>(v0, v1);
            let od = u32x4_shuffle::<1, 3, 5, 7>(v0, v1);
            v128_store(d.as_mut_ptr() as *mut v128, max_step4(ev, od));
        }
    }
    for (d, pair) in dr.iter_mut().zip(sr.as_chunks::<2>().0) {
        let mut m = pair[0];
        step_val(&mut m, pair[1]);
        *d = m;
    }
}

pub(super) fn max_cols_wasm(rv: &[f32], ow: usize, kh: usize, dst: &mut [f32]) {
    let mut ox = 0;
    // SAFETY: `ox + 4 <= ow` and `rv.len() >= kh * ow`.
    unsafe {
        while ox + 4 <= ow {
            let mut m = v128_load(rv.as_ptr().add(ox) as *const v128);
            for t in 1..kh {
                m = max_step4(m, v128_load(rv.as_ptr().add(t * ow + ox) as *const v128));
            }
            v128_store(dst.as_mut_ptr().add(ox) as *mut v128, m);
            ox += 4;
        }
    }
    for j in ox..ow {
        let mut m = rv[j];
        for t in 1..kh {
            step_val(&mut m, rv[t * ow + j]);
        }
        dst[j] = m;
    }
}

pub(super) fn max_row_idx_s1_wasm(
    row: &[f32],
    row_idx: &[i32],
    kw: usize,
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    let ow = dst_v.len();
    let neg1 = i32x4_splat(-1);
    let zero = i32x4_splat(0);
    let vbase = i32x4_splat(row_base);
    let mut ox = 0;
    // SAFETY: `ox + 4 <= ow` and `row`/`row_idx` have at least `ow + kw - 1` entries.
    unsafe {
        while ox + 4 <= ow {
            let vp = row.as_ptr().add(ox);
            let ip = row_idx.as_ptr().add(ox);
            let mut m = v128_load(vp as *const v128);
            let mut mi = v128_load(ip as *const v128);
            for t in 1..kw {
                let a = v128_load(vp.add(t) as *const v128);
                let ai = v128_load(ip.add(t) as *const v128);
                (m, mi) = beats_indexed4(m, mi, a, ai, neg1, zero);
            }
            let valid = i32x4_gt(mi, neg1);
            let flat = v128_bitselect(i32x4_add(mi, vbase), neg1, valid);
            v128_store(dst_v.as_mut_ptr().add(ox) as *mut v128, m);
            v128_store(dst_i.as_mut_ptr().add(ox) as *mut v128, flat);
            ox += 4;
        }
    }
    for j in ox..ow {
        let mut m = row[j];
        let mut mi = row_idx[j];
        for t in 1..kw {
            step_idx(&mut m, &mut mi, row[j + t], row_idx[j + t]);
        }
        dst_v[j] = m;
        dst_i[j] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

pub(super) fn max_pairs_idx_wasm(
    row: &[f32],
    row_idx: &[i32],
    row_base: i32,
    dst_v: &mut [f32],
    dst_i: &mut [i32],
) {
    let ow = dst_v.len();
    let neg1 = i32x4_splat(-1);
    let zero = i32x4_splat(0);
    let vbase = i32x4_splat(row_base);
    let mut ox = 0;
    // SAFETY: `ox + 4 <= ow` implies `2 * ox + 8 <= 2 * ow <= row.len(), row_idx.len()`.
    unsafe {
        while ox + 4 <= ow {
            let vp = row.as_ptr().add(2 * ox);
            let ip = row_idx.as_ptr().add(2 * ox);
            let v0 = v128_load(vp as *const v128);
            let v1 = v128_load(vp.add(4) as *const v128);
            let i0 = v128_load(ip as *const v128);
            let i1 = v128_load(ip.add(4) as *const v128);
            let ev = u32x4_shuffle::<0, 2, 4, 6>(v0, v1);
            let od = u32x4_shuffle::<1, 3, 5, 7>(v0, v1);
            let ei = u32x4_shuffle::<0, 2, 4, 6>(i0, i1);
            let oi = u32x4_shuffle::<1, 3, 5, 7>(i0, i1);
            let (m, mi) = beats_indexed4(ev, ei, od, oi, neg1, zero);
            let valid = i32x4_gt(mi, neg1);
            let flat = v128_bitselect(i32x4_add(mi, vbase), neg1, valid);
            v128_store(dst_v.as_mut_ptr().add(ox) as *mut v128, m);
            v128_store(dst_i.as_mut_ptr().add(ox) as *mut v128, flat);
            ox += 4;
        }
    }
    for j in ox..ow {
        let mut m = row[2 * j];
        let mut mi = row_idx[2 * j];
        step_idx(&mut m, &mut mi, row[2 * j + 1], row_idx[2 * j + 1]);
        dst_v[j] = m;
        dst_i[j] = if mi >= 0 { row_base + mi } else { -1 };
    }
}

pub(super) fn max_cols_idx_wasm(
    rv: &[f32],
    ri: &[i32],
    ow: usize,
    kh: usize,
    dst_v: &mut [f32],
    dst_i: &mut [i64],
) {
    let neg1 = i32x4_splat(-1);
    let zero = i32x4_splat(0);
    let mut ox = 0;
    // SAFETY: `ox + 4 <= ow` and `rv`/`ri` hold at least `kh * ow` entries.
    unsafe {
        while ox + 4 <= ow {
            let mut m = v128_load(rv.as_ptr().add(ox) as *const v128);
            let mut mi = v128_load(ri.as_ptr().add(ox) as *const v128);
            for t in 1..kh {
                let a = v128_load(rv.as_ptr().add(t * ow + ox) as *const v128);
                let ai = v128_load(ri.as_ptr().add(t * ow + ox) as *const v128);
                (m, mi) = beats_indexed4(m, mi, a, ai, neg1, zero);
            }
            let clamped = i32x4_max(mi, zero);
            let lo64 = i64x2_extend_low_i32x4(clamped);
            let hi64 = i64x2_extend_high_i32x4(clamped);
            v128_store(dst_v.as_mut_ptr().add(ox) as *mut v128, m);
            v128_store(dst_i.as_mut_ptr().add(ox) as *mut v128, lo64);
            v128_store(dst_i.as_mut_ptr().add(ox + 2) as *mut v128, hi64);
            ox += 4;
        }
    }
    for j in ox..ow {
        let mut m = rv[j];
        let mut mi = ri[j];
        for t in 1..kh {
            step_idx(&mut m, &mut mi, rv[t * ow + j], ri[t * ow + j]);
        }
        dst_v[j] = m;
        dst_i[j] = mi.max(0) as i64;
    }
}

pub(super) fn duplicate_row2_wasm(src: &[f32], dst: &mut [f32]) {
    let (sc, sr) = src.as_chunks::<4>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` is 4 floats and `d` is 8 floats.
        unsafe {
            let v = v128_load(s.as_ptr() as *const v128);
            let o0 = u32x4_shuffle::<0, 0, 1, 1>(v, v);
            let o1 = u32x4_shuffle::<2, 2, 3, 3>(v, v);
            v128_store(d.as_mut_ptr() as *mut v128, o0);
            v128_store(d.as_mut_ptr().add(4) as *mut v128, o1);
        }
    }
    for (d, &v) in dr.as_chunks_mut::<2>().0.iter_mut().zip(sr) {
        d[0] = v;
        d[1] = v;
    }
}

pub(super) fn upsample2x2_plane_wasm(xp: &[f32], h: usize, w: usize, o: &mut [f32]) {
    let ow = 2 * w;
    for y in 0..h {
        let src = &xp[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        let (sc, sr) = src.as_chunks::<4>();
        let (d0c, d0r) = d0.as_chunks_mut::<8>();
        let (d1c, d1r) = d1.as_chunks_mut::<8>();
        for ((s, p0), p1) in sc.iter().zip(d0c.iter_mut()).zip(d1c.iter_mut()) {
            // SAFETY: `s` is 4 floats, `p0` and `p1` are 8 floats.
            unsafe {
                let v = v128_load(s.as_ptr() as *const v128);
                let o0 = u32x4_shuffle::<0, 0, 1, 1>(v, v);
                let o1 = u32x4_shuffle::<2, 2, 3, 3>(v, v);
                v128_store(p0.as_mut_ptr() as *mut v128, o0);
                v128_store(p0.as_mut_ptr().add(4) as *mut v128, o1);
                v128_store(p1.as_mut_ptr() as *mut v128, o0);
                v128_store(p1.as_mut_ptr().add(4) as *mut v128, o1);
            }
        }
        for ((p0, p1), &v) in d0r.as_chunks_mut::<2>().0.iter_mut().zip(d1r.as_chunks_mut::<2>().0).zip(sr) {
            p0[0] = v;
            p0[1] = v;
            p1[0] = v;
            p1[1] = v;
        }
    }
}

pub(super) fn upsample2x2_bwd_plane_wasm(gs: &[f32], h: usize, w: usize, gp: &mut [f32]) {
    let ow = 2 * w;
    for y in 0..h {
        let r0 = &gs[2 * y * ow..(2 * y + 1) * ow];
        let r1 = &gs[(2 * y + 1) * ow..(2 * y + 2) * ow];
        let dst = &mut gp[y * w..(y + 1) * w];
        let (dc, dr) = dst.as_chunks_mut::<4>();
        let (r0c, r0r) = r0.as_chunks::<8>();
        let (r1c, r1r) = r1.as_chunks::<8>();
        for ((d, p0), p1) in dc.iter_mut().zip(r0c).zip(r1c) {
            // SAFETY: `p0` and `p1` are 8 floats and `d` is 4 floats.
            unsafe {
                let a0 = v128_load(p0.as_ptr() as *const v128);
                let a1 = v128_load(p0.as_ptr().add(4) as *const v128);
                let s0 = f32x4_add(u32x4_shuffle::<0, 2, 4, 6>(a0, a1), u32x4_shuffle::<1, 3, 5, 7>(a0, a1));
                let b0 = v128_load(p1.as_ptr() as *const v128);
                let b1 = v128_load(p1.as_ptr().add(4) as *const v128);
                let s1 = f32x4_add(u32x4_shuffle::<0, 2, 4, 6>(b0, b1), u32x4_shuffle::<1, 3, 5, 7>(b0, b1));
                v128_store(d.as_mut_ptr() as *mut v128, f32x4_add(s0, s1));
            }
        }
        for ((d, p0), p1) in dr.iter_mut().zip(r0r.as_chunks::<2>().0).zip(r1r.as_chunks::<2>().0) {
            *d = (p0[0] + p0[1]) + (p1[0] + p1[1]);
        }
    }
}
