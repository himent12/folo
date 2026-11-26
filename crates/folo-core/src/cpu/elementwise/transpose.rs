//! Cache-blocked 2D & 4D AVX2 transpose (8x8, C=3 RGB, C=4 RGBA, block-element swaps) and fast `f32` strided
//! copy helpers.

use super::super::pool_alloc::alloc_uninit;
use super::super::{CHUNK, PAR_MIN, par_chunks, par_for};
use super::{SendPtr, fill_rows};
use crate::Layout;
use crate::layout::RowPlan;

/// Cache tile side (multiple of 8): a 32×32 `f32` block is 4 KiB and fits comfortably in L1d.
pub(super) const TILE: usize = 32;

/// Transpose a `src_rows × src_cols` matrix in `src` (row stride `src_rs`) into a `src_cols × src_rows` matrix
/// in `dst` (row stride `dst_rs`): `dst[j * dst_rs + i] = src[i * src_rs + j]`.
///
/// Parallelises across `TILE`-row destination bands or `src_rows` strips when `src_rows * src_cols >= PAR_MIN`.
pub(crate) fn transpose_f32_2d(
    src: &[f32],
    src_rs: usize,
    dst: &mut [f32],
    dst_rs: usize,
    src_rows: usize,
    src_cols: usize,
) {
    if src_rows == 0 || src_cols == 0 {
        return;
    }
    let src_need = (src_rows - 1)
        .checked_mul(src_rs)
        .and_then(|v| v.checked_add(src_cols))
        .expect("transpose_f32_2d: source extent overflow");
    let dst_need = (src_cols - 1)
        .checked_mul(dst_rs)
        .and_then(|v| v.checked_add(src_rows))
        .expect("transpose_f32_2d: destination extent overflow");
    assert!(dst_rs >= src_rows, "transpose_f32_2d: dst_rs ({dst_rs}) < src_rows ({src_rows})");
    assert!(src.len() >= src_need, "transpose_f32_2d: src too short ({} < {src_need})", src.len());
    assert!(dst.len() >= dst_need, "transpose_f32_2d: dst too short ({} < {dst_need})", dst.len());

    let dst = &mut dst[..dst_need];
    if src_rows.saturating_mul(src_cols) < PAR_MIN {
        transpose_f32_2d_serial(src, src_rs, dst, dst_rs, src_rows, src_cols);
        return;
    }
    let col_bands = src_cols.div_ceil(TILE);
    let row_bands = src_rows.div_ceil(TILE);
    if src_cols > TILE && (col_bands >= 8 || col_bands >= row_bands) {
        par_chunks(dst, TILE * dst_rs, &|b, band_dst| {
            let j0 = b * TILE;
            let bh = (src_cols - j0).min(TILE);
            // SAFETY: `band_dst` holds rows `j0..j0 + bh` (with stride `dst_rs >= src_rows`) exclusively for
            // this task, and `src` bounds were checked above.
            unsafe { transpose_band_ptr(src, src_rs, band_dst.as_mut_ptr(), dst_rs, src_rows, j0, bh) };
        });
    } else if src_rows >= 2 * TILE {
        // Tall-skinny matrix (e.g. NHWC -> NCHW for C = 3, 4, 16, 32, 64): split `src_rows` into cache-line-
        // aligned strips so all CPU cores run concurrently without false sharing.
        let strip = ((CHUNK / src_cols).max(TILE)).div_ceil(TILE) * TILE;
        let tasks = src_rows.div_ceil(strip);
        let dst_ptr = SendPtr(dst.as_mut_ptr());
        par_for(tasks, &|t| {
            let i0 = t * strip;
            let rows_t = (src_rows - i0).min(strip);
            let src_sub = &src[i0 * src_rs..(i0 + rows_t - 1) * src_rs + src_cols];
            for j0 in (0..src_cols).step_by(TILE) {
                let bh = (src_cols - j0).min(TILE);
                // SAFETY: each task `t` owns disjoint column range `i0..i0 + rows_t` of every destination row
                // `j0..j0 + bh` (and `dst_rs >= src_rows`), so raw writes across tasks never alias.
                unsafe {
                    transpose_band_ptr(
                        src_sub,
                        src_rs,
                        dst_ptr.ptr().add(j0 * dst_rs + i0),
                        dst_rs,
                        rows_t,
                        j0,
                        bh,
                    );
                }
            }
        });
    } else {
        transpose_f32_2d_serial(src, src_rs, dst, dst_rs, src_rows, src_cols);
    }
}

/// Serial variant of [`transpose_f32_2d`], used inside outer batch parallelism.
pub(crate) fn transpose_f32_2d_serial(
    src: &[f32],
    src_rs: usize,
    dst: &mut [f32],
    dst_rs: usize,
    src_rows: usize,
    src_cols: usize,
) {
    if src_rows == 0 || src_cols == 0 {
        return;
    }
    let src_need = (src_rows - 1)
        .checked_mul(src_rs)
        .and_then(|v| v.checked_add(src_cols))
        .expect("transpose_f32_2d_serial: source extent overflow");
    let dst_need = (src_cols - 1)
        .checked_mul(dst_rs)
        .and_then(|v| v.checked_add(src_rows))
        .expect("transpose_f32_2d_serial: destination extent overflow");
    assert!(dst_rs >= src_rows, "transpose_f32_2d_serial: dst_rs ({dst_rs}) < src_rows ({src_rows})");
    assert!(src.len() >= src_need, "transpose_f32_2d_serial: src too short ({} < {src_need})", src.len());
    assert!(dst.len() >= dst_need, "transpose_f32_2d_serial: dst too short ({} < {dst_need})", dst.len());

    let dst = &mut dst[..dst_need];
    if src_cols <= TILE {
        // SAFETY: `dst` covers all `src_cols` rows of stride `dst_rs >= src_rows`, and `src` bounds were checked.
        unsafe { transpose_band_ptr(src, src_rs, dst.as_mut_ptr(), dst_rs, src_rows, 0, src_cols) };
        return;
    }
    for (b, band_dst) in dst.chunks_mut(TILE * dst_rs).enumerate() {
        let j0 = b * TILE;
        let bh = (src_cols - j0).min(TILE);
        // SAFETY: `band_dst` holds rows `j0..j0 + bh` exclusively, and `src` bounds were checked above.
        unsafe { transpose_band_ptr(src, src_rs, band_dst.as_mut_ptr(), dst_rs, src_rows, j0, bh) };
    }
}

/// Transpose destination rows `j0..j0 + bh` (across `src_rows` columns) into raw `dst_ptr` (row stride `dst_rs`).
///
/// # Safety
/// - `dst_ptr` must be valid for exclusive writes at `dst_ptr.add(j * dst_rs + i)` for all `0 <= j < bh` and
///   `0 <= i < src_rows`, with `dst_rs >= src_rows`.
/// - `src` must contain at least `(src_rows - 1) * src_rs + j0 + bh` elements.
#[inline]
unsafe fn transpose_band_ptr(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
    bh: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above, and the caller guarantees `src` and `dst_ptr` bounds.
        if bh == 3 && j0 == 0 && src_rs == 3 && src_rows >= 8 {
            unsafe { transpose_c3_deinterleave_avx2(src, dst_ptr, dst_rs, src_rows) };
            return;
        }
        if src_rows == 3 && dst_rs == 3 && bh >= 8 {
            unsafe { transpose_c3_interleave_avx2(src, src_rs, dst_ptr, j0, bh) };
            return;
        }
        if bh == 4 && src_rows >= 8 {
            unsafe { transpose_c4_deinterleave_avx2(src, src_rs, dst_ptr, dst_rs, src_rows, j0) };
            return;
        }
        if src_rows == 4 && bh >= 8 {
            unsafe { transpose_c4_interleave_avx2(src, src_rs, dst_ptr, dst_rs, j0, bh) };
            return;
        }
        if bh >= 8 && src_rows >= 8 {
            unsafe { transpose_band_avx2(src, src_rs, dst_ptr, dst_rs, src_rows, j0, bh) };
            return;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: caller guarantees `src` and `dst_ptr` bounds.
        if bh == 4 && src_rows >= 4 {
            unsafe {
                super::transpose_wasm::transpose_c4_deinterleave_wasm(
                    src, src_rs, dst_ptr, dst_rs, src_rows, j0,
                )
            };
            return;
        }
        if src_rows == 4 && bh >= 4 {
            unsafe {
                super::transpose_wasm::transpose_c4_interleave_wasm(src, src_rs, dst_ptr, dst_rs, j0, bh)
            };
            return;
        }
        if bh >= 4 && src_rows >= 4 {
            unsafe {
                super::transpose_wasm::transpose_band_wasm(src, src_rs, dst_ptr, dst_rs, src_rows, j0, bh)
            };
            return;
        }
    }
    // SAFETY: forwarded from caller.
    unsafe { transpose_band_scalar_ptr(src, src_rs, dst_ptr, dst_rs, src_rows, j0, bh) };
}

/// Scalar cache-tiled transpose into raw `dst_ptr`.
///
/// # Safety
/// Same invariants as [`transpose_band_ptr`].
#[inline]
unsafe fn transpose_band_scalar_ptr(
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
    for i0 in (0..src_rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(src_rows);
        for j in 0..bh {
            // SAFETY: `j < bh` and `i0..i1 <= src_rows <= dst_rs`, exclusively owned by caller.
            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr.add(j * dst_rs + i0), i1 - i0) };
            for (k, v) in d.iter_mut().enumerate() {
                *v = src[(i0 + k) * src_rs + j0 + j];
            }
        }
    }
}

/// AVX2 deinterleave for contiguous 3-channel inputs (`[N, 3] -> [3, N]`, e.g. RGB `NHWC -> NCHW`).
///
/// # Safety
/// The CPU supports AVX2, and `dst_ptr` is valid for writes at `c * dst_rs + i` (`c < 3`, `i < src_rows`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_c3_deinterleave_avx2(src: &[f32], dst_ptr: *mut f32, dst_rs: usize, src_rows: usize) {
    use std::arch::x86_64::*;

    assert!(src_rows > 0 && dst_rs >= src_rows);
    assert!(src_rows.checked_mul(3).is_some_and(|need| src.len() >= need));

    let idx_r = _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5);
    let idx_g = _mm256_setr_epi32(1, 4, 7, 2, 5, 0, 3, 6);
    let idx_b = _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7);

    let i_vec = src_rows / 8 * 8;
    let d0 = dst_ptr;
    // SAFETY: `dst_rs >= src_rows` and caller guarantees 3 rows of `dst_rs`.
    let (d1, d2) = unsafe { (dst_ptr.add(dst_rs), dst_ptr.add(2 * dst_rs)) };
    for i in (0..i_vec).step_by(8) {
        // SAFETY: `i + 8 <= i_vec <= src_rows`, so `i * 3 + 24 <= src_rows * 3 <= src.len()`, and `i + 8 <= src_rows`.
        unsafe {
            let sptr = src.as_ptr().add(i * 3);
            let v0 = _mm256_loadu_ps(sptr);
            let v1 = _mm256_loadu_ps(sptr.add(8));
            let v2 = _mm256_loadu_ps(sptr.add(16));

            let r_mix = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(v0, v1), v2);
            let g_mix = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(v2, v0), v1);
            let b_mix = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(v1, v2), v0);

            _mm256_storeu_ps(d0.add(i), _mm256_permutevar8x32_ps(r_mix, idx_r));
            _mm256_storeu_ps(d1.add(i), _mm256_permutevar8x32_ps(g_mix, idx_g));
            _mm256_storeu_ps(d2.add(i), _mm256_permutevar8x32_ps(b_mix, idx_b));
        }
    }
    for i in i_vec..src_rows {
        let base = i * 3;
        // SAFETY: `i < src_rows`, exclusively owned.
        unsafe {
            *d0.add(i) = src[base];
            *d1.add(i) = src[base + 1];
            *d2.add(i) = src[base + 2];
        }
    }
}

/// AVX2 interleave for 3-row sources into contiguous 3-channel outputs (`[3, N] -> [N, 3]`, e.g. RGB `NCHW -> NHWC`).
///
/// # Safety
/// The CPU supports AVX2, and `dst_ptr` is valid for writes of `bh * 3` contiguous floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_c3_interleave_avx2(src: &[f32], src_rs: usize, dst_ptr: *mut f32, j0: usize, bh: usize) {
    use std::arch::x86_64::*;

    assert!(bh > 0);
    assert!(
        2usize
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(bh))
            .is_some_and(|need| src.len() >= need)
    );

    let idx_r = _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5);
    let idx_g = _mm256_setr_epi32(5, 0, 3, 6, 1, 4, 7, 2);
    let idx_b = _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7);

    let j_vec = bh / 8 * 8;
    for j in (0..j_vec).step_by(8) {
        // SAFETY: `2 * src_rs + j0 + j + 8 <= 2 * src_rs + j0 + bh <= src.len()`, and `j * 3 + 24 <= bh * 3`.
        unsafe {
            let sptr = src.as_ptr().add(j0 + j);
            let r = _mm256_loadu_ps(sptr);
            let g = _mm256_loadu_ps(sptr.add(src_rs));
            let b = _mm256_loadu_ps(sptr.add(2 * src_rs));

            let r_mix = _mm256_permutevar8x32_ps(r, idx_r);
            let g_mix = _mm256_permutevar8x32_ps(g, idx_g);
            let b_mix = _mm256_permutevar8x32_ps(b, idx_b);

            let v0 = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(r_mix, g_mix), b_mix);
            let v1 = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(b_mix, r_mix), g_mix);
            let v2 = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(g_mix, b_mix), r_mix);

            let dptr = dst_ptr.add(j * 3);
            _mm256_storeu_ps(dptr, v0);
            _mm256_storeu_ps(dptr.add(8), v1);
            _mm256_storeu_ps(dptr.add(16), v2);
        }
    }
    for j in j_vec..bh {
        let col = j0 + j;
        // SAFETY: `j * 3 + 2 < bh * 3`, exclusively owned.
        unsafe {
            let d = dst_ptr.add(j * 3);
            *d = src[col];
            *d.add(1) = src[src_rs + col];
            *d.add(2) = src[2 * src_rs + col];
        }
    }
}

/// AVX2 4-column deinterleave (`[N, 4] -> [4, N]`, e.g. RGBA / 4-box `NHWC -> NCHW`).
///
/// # Safety
/// The CPU supports AVX2, and `dst_ptr` is valid for writes at `c * dst_rs + i` (`c < 4`, `i < src_rows`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_c4_deinterleave_avx2(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
) {
    use std::arch::x86_64::*;

    assert!(src_rows > 0 && dst_rs >= src_rows);
    assert!(
        (src_rows - 1)
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(4))
            .is_some_and(|need| src.len() >= need)
    );

    let i_vec = src_rows / 8 * 8;
    // SAFETY: caller guarantees 4 destination rows spaced by `dst_rs >= src_rows`.
    let (d0, d1, d2, d3) =
        unsafe { (dst_ptr, dst_ptr.add(dst_rs), dst_ptr.add(2 * dst_rs), dst_ptr.add(3 * dst_rs)) };
    for i in (0..i_vec).step_by(8) {
        // SAFETY: `(i + 7) * src_rs + j0 + 4 <= src.len()` and `i + 8 <= src_rows`.
        unsafe {
            let sptr = src.as_ptr().add(i * src_rs + j0);
            let (p0, p1, p2, p3) = if src_rs == 4 && j0 == 0 {
                let v0 = _mm256_loadu_ps(sptr);
                let v1 = _mm256_loadu_ps(sptr.add(8));
                let v2 = _mm256_loadu_ps(sptr.add(16));
                let v3 = _mm256_loadu_ps(sptr.add(24));
                (
                    _mm256_permute2f128_ps::<0x20>(v0, v2),
                    _mm256_permute2f128_ps::<0x31>(v0, v2),
                    _mm256_permute2f128_ps::<0x20>(v1, v3),
                    _mm256_permute2f128_ps::<0x31>(v1, v3),
                )
            } else {
                let pair = |a: *const f32, b: *const f32| {
                    _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(_mm_loadu_ps(a)), _mm_loadu_ps(b))
                };
                (
                    pair(sptr, sptr.add(4 * src_rs)),
                    pair(sptr.add(src_rs), sptr.add(5 * src_rs)),
                    pair(sptr.add(2 * src_rs), sptr.add(6 * src_rs)),
                    pair(sptr.add(3 * src_rs), sptr.add(7 * src_rs)),
                )
            };

            let t0 = _mm256_unpacklo_ps(p0, p1);
            let t1 = _mm256_unpackhi_ps(p0, p1);
            let t2 = _mm256_unpacklo_ps(p2, p3);
            let t3 = _mm256_unpackhi_ps(p2, p3);

            _mm256_storeu_ps(d0.add(i), _mm256_shuffle_ps::<0x44>(t0, t2));
            _mm256_storeu_ps(d1.add(i), _mm256_shuffle_ps::<0xEE>(t0, t2));
            _mm256_storeu_ps(d2.add(i), _mm256_shuffle_ps::<0x44>(t1, t3));
            _mm256_storeu_ps(d3.add(i), _mm256_shuffle_ps::<0xEE>(t1, t3));
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

/// AVX2 4-row interleave (`[4, N] -> [N, 4]`, e.g. RGBA / 4-box `NCHW -> NHWC`).
///
/// # Safety
/// The CPU supports AVX2, and `dst_ptr` is valid for writes at `j * dst_rs + c` (`j < bh`, `c < 4`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_c4_interleave_avx2(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    j0: usize,
    bh: usize,
) {
    use std::arch::x86_64::*;

    assert!(bh > 0 && dst_rs >= 4);
    assert!(
        3usize
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(bh))
            .is_some_and(|need| src.len() >= need)
    );

    let j_vec = bh / 8 * 8;
    for j in (0..j_vec).step_by(8) {
        // SAFETY: `3 * src_rs + j0 + j + 8 <= src.len()` and `(j + 7) * dst_rs + 4` is owned by caller.
        unsafe {
            let sptr = src.as_ptr().add(j0 + j);
            let r = _mm256_loadu_ps(sptr);
            let g = _mm256_loadu_ps(sptr.add(src_rs));
            let b = _mm256_loadu_ps(sptr.add(2 * src_rs));
            let a = _mm256_loadu_ps(sptr.add(3 * src_rs));

            let t0 = _mm256_unpacklo_ps(r, g);
            let t1 = _mm256_unpackhi_ps(r, g);
            let t2 = _mm256_unpacklo_ps(b, a);
            let t3 = _mm256_unpackhi_ps(b, a);

            let p0 = _mm256_shuffle_ps::<0x44>(t0, t2);
            let p1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
            let p2 = _mm256_shuffle_ps::<0x44>(t1, t3);
            let p3 = _mm256_shuffle_ps::<0xEE>(t1, t3);

            let dptr = dst_ptr.add(j * dst_rs);
            if dst_rs == 4 {
                _mm256_storeu_ps(dptr, _mm256_permute2f128_ps::<0x20>(p0, p1));
                _mm256_storeu_ps(dptr.add(8), _mm256_permute2f128_ps::<0x20>(p2, p3));
                _mm256_storeu_ps(dptr.add(16), _mm256_permute2f128_ps::<0x31>(p0, p1));
                _mm256_storeu_ps(dptr.add(24), _mm256_permute2f128_ps::<0x31>(p2, p3));
            } else {
                _mm_storeu_ps(dptr, _mm256_castps256_ps128(p0));
                _mm_storeu_ps(dptr.add(dst_rs), _mm256_castps256_ps128(p1));
                _mm_storeu_ps(dptr.add(2 * dst_rs), _mm256_castps256_ps128(p2));
                _mm_storeu_ps(dptr.add(3 * dst_rs), _mm256_castps256_ps128(p3));
                _mm_storeu_ps(dptr.add(4 * dst_rs), _mm256_extractf128_ps::<1>(p0));
                _mm_storeu_ps(dptr.add(5 * dst_rs), _mm256_extractf128_ps::<1>(p1));
                _mm_storeu_ps(dptr.add(6 * dst_rs), _mm256_extractf128_ps::<1>(p2));
                _mm_storeu_ps(dptr.add(7 * dst_rs), _mm256_extractf128_ps::<1>(p3));
            }
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

/// AVX2 8x8 register-blocked transpose for destination rows `j0..j0 + bh`.
///
/// # Safety
/// The CPU supports AVX2, and `dst_ptr` satisfies the invariants of [`transpose_band_ptr`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_band_avx2(
    src: &[f32],
    src_rs: usize,
    dst_ptr: *mut f32,
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
    bh: usize,
) {
    use std::arch::x86_64::*;

    assert!(bh > 0 && src_rows > 0 && dst_rs >= src_rows);
    assert!(
        (src_rows - 1)
            .checked_mul(src_rs)
            .and_then(|v| v.checked_add(j0))
            .and_then(|v| v.checked_add(bh))
            .is_some_and(|need| src.len() >= need)
    );

    let j_vec = bh / 8 * 8;
    for i0 in (0..src_rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(src_rows);
        let bw = i1 - i0;
        let i_vec = i0 + bw / 8 * 8;

        for j in (0..j_vec).step_by(8) {
            for i in (i0..i_vec).step_by(8) {
                // SAFETY: `i + 7 < src_rows`, `j0 + j + 7 < j0 + bh`, and `j + 7 < bh`; the assertions above
                // guarantee every 8-float unaligned load from `src` and store to `dst_ptr` is in bounds.
                unsafe {
                    let sptr = src.as_ptr().add(i * src_rs + j0 + j);
                    let r0 = _mm256_loadu_ps(sptr);
                    let r1 = _mm256_loadu_ps(sptr.add(src_rs));
                    let r2 = _mm256_loadu_ps(sptr.add(2 * src_rs));
                    let r3 = _mm256_loadu_ps(sptr.add(3 * src_rs));
                    let r4 = _mm256_loadu_ps(sptr.add(4 * src_rs));
                    let r5 = _mm256_loadu_ps(sptr.add(5 * src_rs));
                    let r6 = _mm256_loadu_ps(sptr.add(6 * src_rs));
                    let r7 = _mm256_loadu_ps(sptr.add(7 * src_rs));

                    let t0 = _mm256_unpacklo_ps(r0, r1);
                    let t1 = _mm256_unpackhi_ps(r0, r1);
                    let t2 = _mm256_unpacklo_ps(r2, r3);
                    let t3 = _mm256_unpackhi_ps(r2, r3);
                    let t4 = _mm256_unpacklo_ps(r4, r5);
                    let t5 = _mm256_unpackhi_ps(r4, r5);
                    let t6 = _mm256_unpacklo_ps(r6, r7);
                    let t7 = _mm256_unpackhi_ps(r6, r7);

                    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                    let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                    let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);

                    let dptr = dst_ptr.add(j * dst_rs + i);
                    _mm256_storeu_ps(dptr, _mm256_permute2f128_ps::<0x20>(u0, u4));
                    _mm256_storeu_ps(dptr.add(dst_rs), _mm256_permute2f128_ps::<0x20>(u1, u5));
                    _mm256_storeu_ps(dptr.add(2 * dst_rs), _mm256_permute2f128_ps::<0x20>(u2, u6));
                    _mm256_storeu_ps(dptr.add(3 * dst_rs), _mm256_permute2f128_ps::<0x20>(u3, u7));
                    _mm256_storeu_ps(dptr.add(4 * dst_rs), _mm256_permute2f128_ps::<0x31>(u0, u4));
                    _mm256_storeu_ps(dptr.add(5 * dst_rs), _mm256_permute2f128_ps::<0x31>(u1, u5));
                    _mm256_storeu_ps(dptr.add(6 * dst_rs), _mm256_permute2f128_ps::<0x31>(u2, u6));
                    _mm256_storeu_ps(dptr.add(7 * dst_rs), _mm256_permute2f128_ps::<0x31>(u3, u7));
                }
            }
            if i_vec < i1 {
                for jc in j..j + 8 {
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

/// Choose `(tc, tr)` tile counts for a `rows × cols` block-element transpose with `block`-float elements so the
/// active write streams stay in L1d.
#[inline]
fn block_tile_dims(block: usize) -> (usize, usize) {
    let tc = if block <= 16 {
        16
    } else if block <= 64 {
        8
    } else if block <= 256 {
        4
    } else {
        2
    };
    let tr = (2048 / (tc * block).max(1)).clamp(4, 32);
    (tc, tr)
}

/// Cache-blocked transpose of a `rows × cols` grid of `block`-float contiguous strips:
/// `dst[(c * rows + r) * block .. + block] = src[r * src_rs + c * block .. + block]`.
///
/// Accelerates 4D head swaps (`[B, S, H, D] <-> [B, H, S, D]`) and spatial swaps (`[N, C, H, W] <-> [N, H, C, W]`).
fn transpose_blocks_f32(src: &[f32], src_rs: usize, dst: &mut [f32], rows: usize, cols: usize, block: usize) {
    if rows == 0 || cols == 0 || block == 0 {
        return;
    }
    let src_need = (rows - 1)
        .checked_mul(src_rs)
        .and_then(|v| v.checked_add(cols.checked_mul(block)?))
        .expect("transpose_blocks_f32: source extent overflow");
    let dst_need = cols
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(block))
        .expect("transpose_blocks_f32: size overflow");
    assert!(src.len() >= src_need, "transpose_blocks_f32: src too short ({} < {src_need})", src.len());
    assert_eq!(dst.len(), dst_need, "transpose_blocks_f32: dst length mismatch");

    let (tc, tr) = block_tile_dims(block);
    if dst_need < PAR_MIN {
        transpose_blocks_f32_serial(src, src_rs, dst, rows, cols, block);
        return;
    }
    let col_bands = cols.div_ceil(tc);
    let row_bands = rows.div_ceil(tr);
    if cols > tc && (col_bands >= 8 || col_bands >= row_bands) {
        let band_stride = tc * rows * block;
        par_chunks(dst, band_stride, &|b, band_dst| {
            let c0 = b * tc;
            let c1 = (c0 + tc).min(cols);
            let nc = c1 - c0;
            for r0 in (0..rows).step_by(tr) {
                let r1 = (r0 + tr).min(rows);
                for r in r0..r1 {
                    let srow = &src[r * src_rs + c0 * block..r * src_rs + c1 * block];
                    for c in 0..nc {
                        let d0 = (c * rows + r) * block;
                        band_dst[d0..d0 + block].copy_from_slice(&srow[c * block..(c + 1) * block]);
                    }
                }
            }
        });
    } else if rows >= 2 * tr {
        let strip = ((CHUNK / (cols * block).max(1)).max(tr)).div_ceil(tr) * tr;
        let tasks = rows.div_ceil(strip);
        let dst_ptr = SendPtr(dst.as_mut_ptr());
        par_for(tasks, &|t| {
            let r_start = t * strip;
            let r_end = (r_start + strip).min(rows);
            for c0 in (0..cols).step_by(tc) {
                let c1 = (c0 + tc).min(cols);
                for r0 in (r_start..r_end).step_by(tr) {
                    let r1 = (r0 + tr).min(r_end);
                    for r in r0..r1 {
                        let srow = &src[r * src_rs + c0 * block..r * src_rs + c1 * block];
                        for c in c0..c1 {
                            let d0 = (c * rows + r) * block;
                            // SAFETY: each task `t` owns disjoint `r in r_start..r_end` for every `c`, and
                            // `d0 + block <= cols * rows * block == dst.len()`.
                            let d = unsafe { std::slice::from_raw_parts_mut(dst_ptr.ptr().add(d0), block) };
                            let sc = (c - c0) * block;
                            d.copy_from_slice(&srow[sc..sc + block]);
                        }
                    }
                }
            }
        });
    } else {
        transpose_blocks_f32_serial(src, src_rs, dst, rows, cols, block);
    }
}

/// Serial variant of [`transpose_blocks_f32`], used inside outer batch parallelism.
fn transpose_blocks_f32_serial(
    src: &[f32],
    src_rs: usize,
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    block: usize,
) {
    if rows == 0 || cols == 0 || block == 0 {
        return;
    }
    let src_need = (rows - 1)
        .checked_mul(src_rs)
        .and_then(|v| v.checked_add(cols.checked_mul(block)?))
        .expect("transpose_blocks_f32_serial: source extent overflow");
    let dst_need = cols
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(block))
        .expect("transpose_blocks_f32_serial: size overflow");
    assert!(src.len() >= src_need, "transpose_blocks_f32_serial: src too short ({} < {src_need})", src.len());
    assert_eq!(dst.len(), dst_need, "transpose_blocks_f32_serial: dst length mismatch");

    let (tc, tr) = block_tile_dims(block);
    for c0 in (0..cols).step_by(tc) {
        let c1 = (c0 + tc).min(cols);
        for r0 in (0..rows).step_by(tr) {
            let r1 = (r0 + tr).min(rows);
            for r in r0..r1 {
                let srow = &src[r * src_rs + c0 * block..r * src_rs + c1 * block];
                for c in c0..c1 {
                    let d0 = (c * rows + r) * block;
                    let sc = (c - c0) * block;
                    dst[d0..d0 + block].copy_from_slice(&srow[sc..sc + block]);
                }
            }
        }
    }
}

/// Contiguous copy of `src` through `layout` into an existing `f32` slice, using the cache-blocked AVX2
/// transpose whenever the layout's innermost dimensions form a 2D transpose (`.t()`, batched transpose,
/// `NCHW <-> NHWC`) or a 4D block-element transpose (`[B, S, H, D] <-> [B, H, S, D]`, `[N, C, H, W] <-> [N, H, C, W]`).
pub(crate) fn copy_strided_f32_into_slice(src: &[f32], layout: &Layout, out: &mut [f32]) {
    assert_eq!(out.len(), layout.numel(), "copy_strided_f32_into_slice: length mismatch");
    if out.is_empty() {
        return;
    }
    if let Some(r) = layout.contiguous_range() {
        let s = &src[r];
        if out.len() >= PAR_MIN {
            par_chunks(out, CHUNK, &|c, chunk| {
                let start = c * CHUNK;
                chunk.copy_from_slice(&s[start..start + chunk.len()]);
            });
        } else {
            out.copy_from_slice(s);
        }
        return;
    }
    let plan = RowPlan::<1>::new(layout.dims(), [layout.strides()], [layout.offset()]);
    if let Some((batches, cols)) = plan.transpose_plane() {
        let rows = plan.len;
        let src_rs = plan.inner[0];
        let plane = cols * rows;
        if batches == 1 {
            let off = plan.start_offset(0)[0];
            transpose_f32_2d(&src[off..], src_rs, out, rows, rows, cols);
        } else if out.len() < PAR_MIN {
            for b in 0..batches {
                let off = plan.start_offset(b * cols)[0];
                transpose_f32_2d_serial(
                    &src[off..],
                    src_rs,
                    &mut out[b * plane..(b + 1) * plane],
                    rows,
                    rows,
                    cols,
                );
            }
        } else if batches >= 4 || plane < PAR_MIN {
            par_chunks(out, plane, &|b, dst_plane| {
                let off = plan.start_offset(b * cols)[0];
                transpose_f32_2d_serial(&src[off..], src_rs, dst_plane, rows, rows, cols);
            });
        } else {
            for b in 0..batches {
                let off = plan.start_offset(b * cols)[0];
                transpose_f32_2d(&src[off..], src_rs, &mut out[b * plane..(b + 1) * plane], rows, rows, cols);
            }
        }
        return;
    }
    if let Some((batches, cols, rows, src_rs)) = plan.block_transpose_plane() {
        let block = plan.len;
        let plane = cols * rows * block;
        if batches == 1 {
            let off = plan.start_offset(0)[0];
            transpose_blocks_f32(&src[off..], src_rs, out, rows, cols, block);
        } else if out.len() < PAR_MIN {
            for b in 0..batches {
                let off = plan.start_offset(b * cols * rows)[0];
                transpose_blocks_f32_serial(
                    &src[off..],
                    src_rs,
                    &mut out[b * plane..(b + 1) * plane],
                    rows,
                    cols,
                    block,
                );
            }
        } else if batches >= 4 || plane < PAR_MIN {
            par_chunks(out, plane, &|b, dst_plane| {
                let off = plan.start_offset(b * cols * rows)[0];
                transpose_blocks_f32_serial(&src[off..], src_rs, dst_plane, rows, cols, block);
            });
        } else {
            for b in 0..batches {
                let off = plan.start_offset(b * cols * rows)[0];
                transpose_blocks_f32(
                    &src[off..],
                    src_rs,
                    &mut out[b * plane..(b + 1) * plane],
                    rows,
                    cols,
                    block,
                );
            }
        }
        return;
    }
    let s = plan.inner[0];
    fill_rows(out, &plan, |row, [o]| {
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
}

/// [`copy_strided_f32_into_slice`] into a caller-provided `Vec<f32>` of `layout.numel()` elements.
pub(crate) fn copy_strided_f32_into(src: &[f32], layout: &Layout, mut out: Vec<f32>) -> Vec<f32> {
    copy_strided_f32_into_slice(src, layout, &mut out);
    out
}

/// A contiguous `Vec<f32>` copy of `src` through `layout` (allocated from the CPU buffer cache).
pub(crate) fn copy_strided_f32(src: &[f32], layout: &Layout) -> Vec<f32> {
    let out = alloc_uninit(layout.numel());
    copy_strided_f32_into(src, layout, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    fn scalar_copy_reference(src: &[f32], layout: &Layout) -> Vec<f32> {
        let plan = RowPlan::<1>::new(layout.dims(), [layout.strides()], [layout.offset()]);
        let mut out = vec![0.0f32; layout.numel()];
        let s = plan.inner[0];
        let mut row_idx = 0;
        plan.for_rows(0..plan.rows, |_, [o]| {
            let dst = &mut out[row_idx * plan.len..(row_idx + 1) * plan.len];
            for (i, v) in dst.iter_mut().enumerate() {
                *v = src[o + i * s];
            }
            row_idx += 1;
        });
        out
    }

    fn pattern(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 * 0.001 - 500.0).collect()
    }

    #[test]
    fn transpose_2d_c3_c4_and_tall_parallel_bit_exact() {
        for &(r, c) in &[
            (3, 1),
            (3, 7),
            (3, 8),
            (3, 33),
            (3, 64 * 64),
            (1, 3),
            (7, 3),
            (8, 3),
            (33, 3),
            (256 * 256, 3),
            (4, 9),
            (4, 64 * 64),
            (9, 4),
            (256 * 256, 4),
            (16, 4096),
            (4096, 16),
            (48, 2048),
            (2048, 48),
            (13, 37),
            (129, 257),
            (512, 512),
        ] {
            let src = pattern(r * c);
            let mut got = vec![f32::NAN; r * c];
            transpose_f32_2d(&src, c, &mut got, r, r, c);
            let mut want = vec![0.0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    want[j * r + i] = src[i * c + j];
                }
            }
            assert_eq!(got.len(), want.len());
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "mismatch at ({r}x{c}) idx {k}");
            }
        }
    }

    #[test]
    fn transpose_2d_strided_c4_bit_exact() {
        // Non-unit row stride for C=4 deinterleave and C=4 interleave.
        let src_rs = 11;
        let src_rows = 29;
        let src = pattern(src_rows * src_rs);
        let mut got = vec![f32::NAN; 4 * 35];
        transpose_f32_2d(&src[2..], src_rs, &mut got, 35, src_rows, 4);
        for j in 0..4 {
            for i in 0..src_rows {
                assert_eq!(got[j * 35 + i].to_bits(), src[2 + i * src_rs + j].to_bits());
            }
        }

        let dst_rs = 7;
        let src_cols = 27;
        let src2 = pattern(4 * 40);
        let mut got2 = vec![f32::NAN; src_cols * dst_rs];
        transpose_f32_2d(&src2[3..], 40, &mut got2, dst_rs, 4, src_cols);
        for j in 0..src_cols {
            for i in 0..4 {
                assert_eq!(got2[j * dst_rs + i].to_bits(), src2[3 + i * 40 + j].to_bits());
            }
        }
    }

    #[test]
    fn nchw_nhwc_and_4d_block_transposes_bit_exact() {
        let cases: &[(&[usize], &[usize])] = &[
            // RGB / RGBA / multi-channel NCHW <-> NHWC (both small and parallel)
            (&[1, 3, 128, 128], &[0, 2, 3, 1]),
            (&[1, 128, 128, 3], &[0, 3, 1, 2]),
            (&[2, 4, 65, 63], &[0, 2, 3, 1]),
            (&[2, 65, 63, 4], &[0, 3, 1, 2]),
            (&[1, 16, 80, 80], &[0, 2, 3, 1]),
            (&[1, 80, 80, 16], &[0, 3, 1, 2]),
            (&[4, 32, 32, 32], &[0, 2, 3, 1]),
            // 4D attention head swaps [B, S, H, D] <-> [B, H, S, D]
            (&[1, 128, 8, 64], &[0, 2, 1, 3]),
            (&[1, 8, 128, 64], &[0, 2, 1, 3]),
            (&[2, 65, 12, 32], &[0, 2, 1, 3]),
            (&[5, 17, 7, 13], &[0, 2, 1, 3]),
            (&[1, 512, 4, 64], &[0, 2, 1, 3]),
            (&[1, 4, 512, 64], &[0, 2, 1, 3]),
            // 4D spatial swaps [N, C, H, W] <-> [N, H, C, W]
            (&[2, 32, 64, 64], &[0, 2, 1, 3]),
            (&[1, 64, 32, 19], &[0, 2, 1, 3]),
        ];
        for &(shape, perm) in cases {
            let n: usize = shape.iter().product();
            let src = pattern(n);
            let layout = Layout::contiguous(shape).permute(perm);
            let got = copy_strided_f32(&src, &layout);
            let want = scalar_copy_reference(&src, &layout);
            assert_eq!(got.len(), want.len());
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "mismatch on {shape:?} perm {perm:?} at {k}");
            }
        }
        // Non-zero offset & sliced source into block_transpose_plane.
        let x = Tensor::from_vec(pattern(2 * 30 * 12 * 16), [2, 30, 12, 16]).unwrap();
        let view = x.narrow(1, 3, 20).unwrap().narrow(2, 2, 8).unwrap().permute([0, 2, 1, 3]).unwrap();
        let plan = RowPlan::<1>::new(view.dims(), [view.layout().strides()], [view.layout().offset()]);
        assert!(plan.block_transpose_plane().is_some());
        let got = view.contiguous().unwrap().to_vec::<f32>().unwrap();
        let src = x.to_vec::<f32>().unwrap();
        let want = scalar_copy_reference(&src, view.layout());
        assert_eq!(got, want);
    }
}
