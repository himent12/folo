//! WebAssembly SIMD128 (`v128`, 4×4 `f32x4`) cache-tiled 2D transpose and C=4 interleave/deinterleave kernels.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::transpose::TILE;

/// In-register 4×4 transpose of four `f32x4` rows `[r0, r1, r2, r3]`.
#[inline(always)]
fn transpose_4x4(r0: v128, r1: v128, r2: v128, r3: v128) -> (v128, v128, v128, v128) {
    let t0 = u32x4_shuffle::<0, 4, 1, 5>(r0, r1);
    let t1 = u32x4_shuffle::<2, 6, 3, 7>(r0, r1);
    let t2 = u32x4_shuffle::<0, 4, 1, 5>(r2, r3);
    let t3 = u32x4_shuffle::<2, 6, 3, 7>(r2, r3);
    (
        u32x4_shuffle::<0, 1, 4, 5>(t0, t2),
        u32x4_shuffle::<2, 3, 6, 7>(t0, t2),
        u32x4_shuffle::<0, 1, 4, 5>(t1, t3),
        u32x4_shuffle::<2, 3, 6, 7>(t1, t3),
    )
}

/// Wasm SIMD128 4-column deinterleave (`[N, 4] -> [4, N]`).
///
/// # Safety
/// `dst_ptr` must be valid for exclusive writes at `c * dst_rs + i` (`c < 4`, `i < src_rows`, `dst_rs >= src_rows`).
pub(super) unsafe fn transpose_c4_deinterleave_wasm(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
) {
    assert!(src_rows > 0 && dst_rs >= src_rows);
    assert!(
        (src_rows - 1)
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(4))
            .is_some_and(|need| src.len() >= need)
    );
    let i_vec = src_rows / 4 * 4;
    // SAFETY: caller guarantees 4 destination rows spaced by `dst_rs >= src_rows`.
    let (d0, d1, d2, d3) =
        unsafe { (dst_ptr, dst_ptr.add(dst_rs), dst_ptr.add(2 * dst_rs), dst_ptr.add(3 * dst_rs)) };
    for i in (0..i_vec).step_by(4) {
        // SAFETY: `(i + 3) * src_rs + j0 + 4 <= src.len()` and `i + 4 <= src_rows`.
        unsafe {
            let sptr = src.as_ptr().add(i * src_rs + j0);
            let r0 = v128_load(sptr as *const v128);
            let r1 = v128_load(sptr.add(src_rs) as *const v128);
            let r2 = v128_load(sptr.add(2 * src_rs) as *const v128);
            let r3 = v128_load(sptr.add(3 * src_rs) as *const v128);
            let (c0, c1, c2, c3) = transpose_4x4(r0, r1, r2, r3);
            v128_store(d0.add(i) as *mut v128, c0);
            v128_store(d1.add(i) as *mut v128, c1);
            v128_store(d2.add(i) as *mut v128, c2);
            v128_store(d3.add(i) as *mut v128, c3);
        }
    }
    for i in i_vec..src_rows {
        let base = i * src_rs + j0;
        // SAFETY: `i < src_rows`, exclusively owned.
        unsafe {
            *d0.add(i) = src[base];
            *d1.add(i) = src[base + 1];
            *d2.add(i) = src[base + 2];
            *d3.add(i) = src[base + 3];
        }
    }
}

/// Wasm SIMD128 4-row interleave (`[4, N] -> [N, 4]`).
///
/// # Safety
/// `dst_ptr` must be valid for exclusive writes at `j * dst_rs + c` (`j < bh`, `c < 4`, `dst_rs >= 4`).
pub(super) unsafe fn transpose_c4_interleave_wasm(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    j0: usize,
    bh: usize,
) {
    assert!(bh > 0 && dst_rs >= 4);
    assert!(
        3usize
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(bh))
            .is_some_and(|need| src.len() >= need)
    );
    let j_vec = bh / 4 * 4;
    for j in (0..j_vec).step_by(4) {
        // SAFETY: `3 * src_rs + j0 + j + 4 <= src.len()` and `(j + 3) * dst_rs + 4` is owned by caller.
        unsafe {
            let sptr = src.as_ptr().add(j0 + j);
            let r0 = v128_load(sptr as *const v128);
            let r1 = v128_load(sptr.add(src_rs) as *const v128);
            let r2 = v128_load(sptr.add(2 * src_rs) as *const v128);
            let r3 = v128_load(sptr.add(3 * src_rs) as *const v128);
            let (p0, p1, p2, p3) = transpose_4x4(r0, r1, r2, r3);
            let dptr = dst_ptr.add(j * dst_rs);
            v128_store(dptr as *mut v128, p0);
            v128_store(dptr.add(dst_rs) as *mut v128, p1);
            v128_store(dptr.add(2 * dst_rs) as *mut v128, p2);
            v128_store(dptr.add(3 * dst_rs) as *mut v128, p3);
        }
    }
    for j in j_vec..bh {
        let col = j0 + j;
        // SAFETY: `j < bh`, `dst_rs >= 4`, exclusively owned.
        unsafe {
            let d = dst_ptr.add(j * dst_rs);
            *d = src[col];
            *d.add(1) = src[src_rs + col];
            *d.add(2) = src[2 * src_rs + col];
            *d.add(3) = src[3 * src_rs + col];
        }
    }
}

/// Wasm SIMD128 4×4 register-blocked transpose for destination rows `j0..j0 + bh`.
///
/// # Safety
/// `dst_ptr` must be valid for exclusive writes at `j * dst_rs + i` (`j < bh`, `i < src_rows`, `dst_rs >= src_rows`).
pub(super) unsafe fn transpose_band_wasm(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
    bh: usize,
) {
    assert!(bh > 0 && src_rows > 0 && dst_rs >= src_rows);
    assert!(
        (src_rows - 1)
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(bh))
            .is_some_and(|need| src.len() >= need)
    );
    let j_vec = bh / 4 * 4;
    for i0 in (0..src_rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(src_rows);
        let bw = i1 - i0;
        let i_vec = i0 + bw / 4 * 4;

        for j in (0..j_vec).step_by(4) {
            for i in (i0..i_vec).step_by(4) {
                // SAFETY: `i + 3 < src_rows`, `j0 + j + 3 < j0 + bh`, and `j + 3 < bh`.
                unsafe {
                    let sptr = src.as_ptr().add(i * src_rs + j0 + j);
                    let r0 = v128_load(sptr as *const v128);
                    let r1 = v128_load(sptr.add(src_rs) as *const v128);
                    let r2 = v128_load(sptr.add(2 * src_rs) as *const v128);
                    let r3 = v128_load(sptr.add(3 * src_rs) as *const v128);
                    let (c0, c1, c2, c3) = transpose_4x4(r0, r1, r2, r3);
                    let dptr = dst_ptr.add(j * dst_rs + i);
                    v128_store(dptr as *mut v128, c0);
                    v128_store(dptr.add(dst_rs) as *mut v128, c1);
                    v128_store(dptr.add(2 * dst_rs) as *mut v128, c2);
                    v128_store(dptr.add(3 * dst_rs) as *mut v128, c3);
                }
            }
            if i_vec < i1 {
                for jc in j..j + 4 {
                    // SAFETY: `jc < bh`, `i_vec..i1 <= src_rows <= dst_rs`.
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(dst_ptr.add(jc * dst_rs + i_vec), i1 - i_vec)
                    };
                    for (k, v) in d.iter_mut().enumerate() {
                        *v = src[(i_vec + k) * src_rs + j0 + jc];
                    }
                }
            }
        }
        for j in j_vec..bh {
            // SAFETY: `j < bh`, `i0..i1 <= src_rows <= dst_rs`.
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr.add(j * dst_rs + i0), i1 - i0) };
            for (k, v) in d.iter_mut().enumerate() {
                *v = src[(i0 + k) * src_rs + j0 + j];
            }
        }
    }
}
