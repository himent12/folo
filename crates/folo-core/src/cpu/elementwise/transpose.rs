//! Cache-blocked 8x8 AVX2 2D transpose and fast `f32` strided copy helpers.

use super::super::pool_alloc::alloc_uninit;
use super::super::{CHUNK, PAR_MIN, par_chunks};
use super::fill_rows;
use crate::Layout;
use crate::layout::RowPlan;

/// Cache tile side (multiple of 8): a 32×32 `f32` block is 4 KiB and fits comfortably in L1d.
const TILE: usize = 32;

/// Transpose a `src_rows × src_cols` matrix in `src` (row stride `src_rs`) into a `src_cols × src_rows` matrix
/// in `dst` (row stride `dst_rs`): `dst[j * dst_rs + i] = src[i * src_rs + j]`.
///
/// Parallelises across `TILE`-row destination bands when `src_rows * src_cols >= PAR_MIN`.
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
    if src_rows.saturating_mul(src_cols) >= PAR_MIN && src_cols > TILE {
        par_chunks(dst, TILE * dst_rs, &|b, band_dst| {
            let j0 = b * TILE;
            let bh = (src_cols - j0).min(TILE);
            transpose_band(src, src_rs, band_dst, dst_rs, src_rows, j0, bh);
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
    for (b, band_dst) in dst.chunks_mut(TILE * dst_rs).enumerate() {
        let j0 = b * TILE;
        let bh = (src_cols - j0).min(TILE);
        transpose_band(src, src_rs, band_dst, dst_rs, src_rows, j0, bh);
    }
}

/// Transpose destination rows `j0..j0 + bh` (all `src_rows` columns) into `band_dst`.
#[inline]
fn transpose_band(
    src: &[f32],
    src_rs: usize,
    band_dst: &mut [f32],
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
    bh: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if bh >= 8 && src_rows >= 8 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above; `transpose_band_avx2` validates all slice bounds.
        unsafe { transpose_band_avx2(src, src_rs, band_dst, dst_rs, src_rows, j0, bh) };
        return;
    }
    transpose_band_scalar(src, src_rs, band_dst, dst_rs, src_rows, j0, bh);
}

#[inline]
fn transpose_band_scalar(
    src: &[f32],
    src_rs: usize,
    band_dst: &mut [f32],
    dst_rs: usize,
    src_rows: usize,
    j0: usize,
    bh: usize,
) {
    for i0 in (0..src_rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(src_rows);
        for j in 0..bh {
            let d = &mut band_dst[j * dst_rs + i0..j * dst_rs + i1];
            for (k, v) in d.iter_mut().enumerate() {
                *v = src[(i0 + k) * src_rs + j0 + j];
            }
        }
    }
}

/// AVX2 8x8 register-blocked transpose for destination rows `j0..j0 + bh`.
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_band_avx2(
    src: &[f32],
    src_rs: usize,
    band_dst: &mut [f32],
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
    assert!(
        (bh - 1)
            .checked_mul(dst_rs)
            .and_then(|v| v.checked_add(src_rows))
            .is_some_and(|need| band_dst.len() >= need)
    );

    let j_vec = bh / 8 * 8;
    for i0 in (0..src_rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(src_rows);
        let bw = i1 - i0;
        let i_vec = i0 + bw / 8 * 8;

        for j in (0..j_vec).step_by(8) {
            for i in (i0..i_vec).step_by(8) {
                // SAFETY: `i + 7 < src_rows`, `j0 + j + 7 < j0 + bh`, and `j + 7 < bh`; the assertions above
                // guarantee every 8-float unaligned load from `src` and store to `band_dst` is in bounds.
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

                    let dptr = band_dst.as_mut_ptr().add(j * dst_rs + i);
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
                    let d = &mut band_dst[jc * dst_rs + i_vec..jc * dst_rs + i1];
                    for (k, v) in d.iter_mut().enumerate() {
                        *v = src[(i_vec + k) * src_rs + j0 + jc];
                    }
                }
            }
        }
        for j in j_vec..bh {
            let d = &mut band_dst[j * dst_rs + i0..j * dst_rs + i1];
            for (k, v) in d.iter_mut().enumerate() {
                *v = src[(i0 + k) * src_rs + j0 + j];
            }
        }
    }
}

/// Contiguous copy of `src` through `layout` into an existing `f32` slice, using the cache-blocked AVX2
/// transpose whenever the layout's innermost pair of merged dimensions is a 2D transpose (e.g. `.t()`, batched
/// transpose, or `NCHW <-> NHWC` permute).
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
