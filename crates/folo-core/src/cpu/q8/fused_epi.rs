//! Shared types, non-contiguous slice-copy helpers, and zero-spill AVX2+FMA epilogue helpers for the fused
//! x86-64 `q8` tile drivers (`super::fused`).

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::kernel::{MR, NR, Tile, TileF32, uses_avx_vnni, uses_avx2, uses_avx512_vnni};
use super::silu::silu_q8_avx2_x2;

/// Fast-path B addressing mode for a single-part conv K-loop.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BMode {
    /// General row pointer table (`rows[p] + col`).
    Rows,
    /// Single contiguous 1×1 stride-1 input (`b_base + p * hw + col`, zero pointer-table loads).
    Pointwise1x1 { base: *const u32, hw: usize },
}

/// Active fused x86-64 ISA tier (requires AVX2 + FMA for the fused epilogue).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FusedIsa {
    AvxVnni,
    Avx512Vnni,
    Avx2,
}

/// Returns the active fused x86-64 ISA if AVX2 + FMA are enabled (`FOLO_Q8_SCALAR` not set).
#[inline]
pub(super) fn active_isa() -> Option<FusedIsa> {
    if !uses_avx2() {
        return None;
    }
    if uses_avx_vnni() {
        return Some(FusedIsa::AvxVnni);
    }
    if uses_avx512_vnni() {
        return Some(FusedIsa::Avx512Vnni);
    }
    Some(FusedIsa::Avx2)
}

/// Read-only parameters for one single-part fused task.
pub(super) struct SinglePartTask<'a> {
    pub ks: usize,
    pub a: &'a [i32],
    pub corr: &'a [i32],
    pub scale: &'a [f32],
    pub bias: &'a [f32],
    pub rows: &'a [*const u32],
    pub b_mode: BMode,
    pub r_n: usize,
    pub wq: usize,
    pub oh: usize,
    pub ow: usize,
    pub silu: bool,
    pub is_u8: bool,
    pub group_out: &'a [([f32; MR], [i32; MR])],
    pub group_res: &'a [([f32; MR], [i32; MR])],
    pub res_img_ptr: *const u32,
    pub q_img_ptr: *mut u32,
    pub f_img_ptr: *mut f32,
    pub cb_start: usize,
    pub cb_end: usize,
    pub ob_start: usize,
    pub ob_end: usize,
}

/// Read-only parameters for one part inside a decoupled multi-part fused task.
pub(super) struct PartSlice<'a> {
    pub ks: usize,
    pub row_offset: usize,
    pub a: &'a [i32],
    pub corr: &'a [i32],
    pub scale: &'a [f32],
}

/// Read-only parameters for one decoupled multi-part fused task.
pub(super) struct MultiPartTask<'a> {
    pub parts: &'a [PartSlice<'a>],
    pub bias: &'a [f32],
    pub rows: &'a [*const u32],
    pub r_n: usize,
    pub wq: usize,
    pub oh: usize,
    pub ow: usize,
    pub silu: bool,
    pub is_u8: bool,
    pub group_out: &'a [([f32; MR], [i32; MR])],
    pub group_res: &'a [([f32; MR], [i32; MR])],
    pub res_img_ptr: *const u32,
    pub q_img_ptr: *mut u32,
    pub f_img_ptr: *mut f32,
    pub cb_start: usize,
    pub cb_end: usize,
    pub ob_start: usize,
    pub ob_end: usize,
}

/// Aligned 384-byte local accumulator tile for clean L1 store-forwarding.
#[repr(align(32))]
pub(super) struct AlignedTile(pub Tile);

#[repr(align(32))]
pub(super) struct AlignedTileF32(pub TileF32);

/// Contiguous slice-copy writer for non-contiguous `u8` C4 tiles (e.g. `k=3` chunks spanning padded row ends).
///
/// # Safety
/// Every valid pixel `(oy, ox)` in `col..col+cols` with `oy < oh && ox < ow` is disjointly owned by this task
/// inside `dst_plane[0..oh * ow]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn write_u8_noncontiguous(
    qw: &[u32; NR],
    cols: usize,
    mut oy: usize,
    mut ox: usize,
    oh: usize,
    ow: usize,
    wq: usize,
    dst_plane: *mut u32,
) {
    let mut j = 0usize;
    while j < cols && oy < oh {
        if ox < ow {
            let take = (ow - ox).min(cols - j);
            // SAFETY: guaranteed by caller.
            unsafe {
                std::ptr::copy_nonoverlapping(qw.as_ptr().add(j), dst_plane.add(oy * ow + ox), take);
            }
            j += take;
            ox += take;
        }
        let pad_rem = wq - ox;
        if pad_rem >= cols - j {
            break;
        }
        j += pad_rem;
        oy += 1;
        ox = 0;
    }
}

/// Contiguous slice-copy reader for non-contiguous `u8` C4 residual tiles.
///
/// # Safety
/// Every valid pixel `(oy, ox)` in `col..col+cols` with `oy < oh && ox < ow` is readable inside
/// `src_plane[0..oh * ow]`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn read_u8_noncontiguous(
    resw: &mut [u32; NR],
    cols: usize,
    mut oy: usize,
    mut ox: usize,
    oh: usize,
    ow: usize,
    wq: usize,
    src_plane: *const u32,
) {
    *resw = [0u32; NR];
    let mut j = 0usize;
    while j < cols && oy < oh {
        if ox < ow {
            let take = (ow - ox).min(cols - j);
            // SAFETY: guaranteed by caller.
            unsafe {
                std::ptr::copy_nonoverlapping(src_plane.add(oy * ow + ox), resw.as_mut_ptr().add(j), take);
            }
            j += take;
            ox += take;
        }
        let pad_rem = wq - ox;
        if pad_rem >= cols - j {
            break;
        }
        j += pad_rem;
        oy += 1;
        ox = 0;
    }
}

/// Contiguous slice-copy writer for non-contiguous `f32` tiles (`4` output channels).
///
/// # Safety
/// Every valid pixel `(oy, ox)` in `col..col+cols` for channels `0..4` is disjointly owned by this task.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn write_f32_noncontiguous(
    fv: &[[f32; NR]; MR],
    cols: usize,
    oy0: usize,
    ox0: usize,
    oh: usize,
    ow: usize,
    wq: usize,
    ohw: usize,
    dst_base: *mut f32,
) {
    for (rr, row) in fv.iter().enumerate() {
        let dst_plane = unsafe { dst_base.add(rr * ohw) };
        let (mut j, mut oy, mut ox) = (0usize, oy0, ox0);
        while j < cols && oy < oh {
            if ox < ow {
                let take = (ow - ox).min(cols - j);
                // SAFETY: guaranteed by caller.
                unsafe {
                    std::ptr::copy_nonoverlapping(row.as_ptr().add(j), dst_plane.add(oy * ow + ox), take);
                }
                j += take;
                ox += take;
            }
            let pad_rem = wq - ox;
            if pad_rem >= cols - j {
                break;
            }
            j += pad_rem;
            oy += 1;
            ox = 0;
        }
    }
}

/// Build the 16-lane `i16` zero-point shuffle pairs for channels `(0, 1)` and `(2, 3)`.
#[inline(always)]
pub(super) fn zero_pairs(zero: &[i32; MR]) -> ([i16; 16], [i16; 16]) {
    let (z0, z1, z2, z3) = (zero[0] as i16, zero[1] as i16, zero[2] as i16, zero[3] as i16);
    (
        [z0, z0, z0, z0, z1, z1, z1, z1, z0, z0, z0, z0, z1, z1, z1, z1],
        [z2, z2, z2, z2, z3, z3, z3, z3, z2, z2, z2, z2, z3, z3, z3, z3],
    )
}

/// Epilogue step on one 8-pixel column slice of 4 output channels, directly from 4 `f32` vectors `y0..y3`.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn finish_col8<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
    mut y0: __m256,
    mut y1: __m256,
    mut y2: __m256,
    mut y3: __m256,
    res_ptr: *const u32,
    rs: &[f32; MR],
    rz: &[i32; MR],
    inv: &[f32; MR],
    z01_arr: &[i16; 16],
    z23_arr: &[i16; 16],
    q_dst: *mut u32,
    f_dst: &mut [[f32; NR]; MR],
    k_off: usize,
) {
    if SILU {
        [y0, y1] = silu_q8_avx2_x2([y0, y1]);
        [y2, y3] = silu_q8_avx2_x2([y2, y3]);
    }
    // SAFETY: caller guarantees `res_ptr` readable for 8 words when `HAS_RES`, `q_dst` writable for 8 words when `OUT_U8`.
    unsafe {
        if HAS_RES {
            let mask_ff = _mm256_set1_epi32(0xFF);
            let w = _mm256_loadu_si256(res_ptr as *const __m256i);
            let b0 = _mm256_and_si256(w, mask_ff);
            let b1 = _mm256_and_si256(_mm256_srli_epi32::<8>(w), mask_ff);
            let b2 = _mm256_and_si256(_mm256_srli_epi32::<16>(w), mask_ff);
            let b3 = _mm256_srli_epi32::<24>(w);
            y0 = _mm256_fmadd_ps(
                _mm256_set1_ps(rs[0]),
                _mm256_cvtepi32_ps(_mm256_sub_epi32(b0, _mm256_set1_epi32(rz[0]))),
                y0,
            );
            y1 = _mm256_fmadd_ps(
                _mm256_set1_ps(rs[1]),
                _mm256_cvtepi32_ps(_mm256_sub_epi32(b1, _mm256_set1_epi32(rz[1]))),
                y1,
            );
            y2 = _mm256_fmadd_ps(
                _mm256_set1_ps(rs[2]),
                _mm256_cvtepi32_ps(_mm256_sub_epi32(b2, _mm256_set1_epi32(rz[2]))),
                y2,
            );
            y3 = _mm256_fmadd_ps(
                _mm256_set1_ps(rs[3]),
                _mm256_cvtepi32_ps(_mm256_sub_epi32(b3, _mm256_set1_epi32(rz[3]))),
                y3,
            );
        }
        if OUT_U8 {
            let lo = _mm256_set1_ps(-512.0);
            let hi = _mm256_set1_ps(512.0);
            let transpose = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14,
                3, 7, 11, 15,
            );
            let z01 = _mm256_loadu_si256(z01_arr.as_ptr() as *const __m256i);
            let z23 = _mm256_loadu_si256(z23_arr.as_ptr() as *const __m256i);
            let v0 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(y0, _mm256_set1_ps(inv[0])), lo),
                hi,
            ));
            let v1 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(y1, _mm256_set1_ps(inv[1])), lo),
                hi,
            ));
            let v2 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(y2, _mm256_set1_ps(inv[2])), lo),
                hi,
            ));
            let v3 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(y3, _mm256_set1_ps(inv[3])), lo),
                hi,
            ));
            let p01 = _mm256_adds_epi16(_mm256_packs_epi32(v0, v1), z01);
            let p23 = _mm256_adds_epi16(_mm256_packs_epi32(v2, v3), z23);
            let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
            _mm256_storeu_si256(q_dst as *mut __m256i, bytes);
        } else {
            _mm256_storeu_ps(f_dst[0].as_mut_ptr().add(k_off), y0);
            _mm256_storeu_ps(f_dst[1].as_mut_ptr().add(k_off), y1);
            _mm256_storeu_ps(f_dst[2].as_mut_ptr().add(k_off), y2);
            _mm256_storeu_ps(f_dst[3].as_mut_ptr().add(k_off), y3);
        }
    }
}

/// Apply the single-part epilogue from a 32-byte aligned L1 accumulator tile `acc`.
/// Keeping `acc` as the handoff between the 16-YMM K-loop and the 12-YMM `silu_q8_avx2_x2` epilogue prevents
/// LLVM's register allocator from hoisting epilogue YMM constants across the K-loop and spilling `vpdpbusd`
/// accumulators on long-K `3×3` convs, while avoiding any `vzeroupper` or ABI save/restore.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn apply_epi_aligned<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
    acc: &AlignedTile,
    corr: &[i32; MR],
    scale: &[f32; MR],
    bias: &[f32; MR],
    res_ptr: *const u32,
    rs: &[f32; MR],
    rz: &[i32; MR],
    inv: &[f32; MR],
    z01_arr: &[i16; 16],
    z23_arr: &[i16; 16],
    q_dst: *mut u32,
    f_dst: &mut [[f32; NR]; MR],
) {
    // SAFETY: `acc.0` is 32-byte aligned (`MR x NR = 4 x 24` `i32`s); pointer bounds guaranteed by caller.
    unsafe {
        for k in 0..3usize {
            let off = 8 * k;
            let y0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_load_si256(acc.0[0].as_ptr().add(off) as *const __m256i),
                    _mm256_set1_epi32(corr[0]),
                )),
                _mm256_set1_ps(scale[0]),
                _mm256_set1_ps(bias[0]),
            );
            let y1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_load_si256(acc.0[1].as_ptr().add(off) as *const __m256i),
                    _mm256_set1_epi32(corr[1]),
                )),
                _mm256_set1_ps(scale[1]),
                _mm256_set1_ps(bias[1]),
            );
            let y2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_load_si256(acc.0[2].as_ptr().add(off) as *const __m256i),
                    _mm256_set1_epi32(corr[2]),
                )),
                _mm256_set1_ps(scale[2]),
                _mm256_set1_ps(bias[2]),
            );
            let y3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_load_si256(acc.0[3].as_ptr().add(off) as *const __m256i),
                    _mm256_set1_epi32(corr[3]),
                )),
                _mm256_set1_ps(scale[3]),
                _mm256_set1_ps(bias[3]),
            );
            finish_col8::<SILU, HAS_RES, OUT_U8>(
                y0,
                y1,
                y2,
                y3,
                if HAS_RES { res_ptr.add(off) } else { res_ptr },
                rs,
                rz,
                inv,
                z01_arr,
                z23_arr,
                if OUT_U8 { q_dst.add(off) } else { q_dst },
                f_dst,
                off,
            );
        }
    }
}
