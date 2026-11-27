//! 512-bit AVX-512 VNNI rotating row-strip `3×3` (`stride ∈ {1, 2}`, `pad = 1`) kernel and row driver.

use std::arch::x86_64::*;

use super::super::fused_3x3::RowArgs;
use super::super::fused_epi::{AlignedTile, apply_epi_aligned, zero_pairs};
use super::super::kernel::{MR, NR};
use super::{AlignedTile512, NR512, apply_epi_512, dp512, store_tile512};

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn k_loop_3x3_512<const IS_S2: bool, const ZVECS: usize>(
    groups: usize,
    a: *const i32,
    slots: [*const u32; 3],
    group_stride: usize,
    ev_w: usize,
    col: usize,
    acc: &mut AlignedTile512,
) {
    let z = _mm512_setzero_si512();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: caller guarantees `groups * 9 * MR` weights and `col + ZVECS * 16` readable words per tap.
    unsafe {
        let mut p0 = slots[0].add(col);
        let mut p1 = slots[1].add(col);
        let mut p2 = slots[2].add(col);
        let mut ap = a;
        for _ in 0..groups {
            for base in [p0, p1, p2] {
                macro_rules! tap {
                    ($bp:expr, $aoff:expr) => {
                        let bp = $bp;
                        let b0 = _mm512_loadu_si512(bp as *const __m512i);
                        let b1 =
                            if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
                        let b2 =
                            if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
                        let wp = ap.add(($aoff) * MR);
                        dp512!(
                            ZVECS, b0, b1, b2, wp, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32
                        );
                    };
                }
                if IS_S2 {
                    let od = base.add(ev_w);
                    tap!(od, 0);
                    tap!(base, 1);
                    tap!(od.add(1), 2);
                } else {
                    tap!(base, 0);
                    tap!(base.add(1), 1);
                    tap!(base.add(2), 2);
                }
                ap = ap.wrapping_add(3 * MR);
            }
            p0 = p0.wrapping_add(group_stride);
            p1 = p1.wrapping_add(group_stride);
            p2 = p2.wrapping_add(group_stride);
        }
        store_tile512!(ZVECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn k_loop_3x3_8<const IS_S2: bool>(
    groups: usize,
    a: *const i32,
    slots: [*const u32; 3],
    group_stride: usize,
    ev_w: usize,
    col: usize,
    acc: &mut AlignedTile,
) {
    let z = _mm256_setzero_si256();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    // SAFETY: caller guarantees `groups * 9 * MR` weights and `col + 8` readable words per tap.
    unsafe {
        let mut p0 = slots[0].add(col);
        let mut p1 = slots[1].add(col);
        let mut p2 = slots[2].add(col);
        let mut ap = a;
        for _ in 0..groups {
            for base in [p0, p1, p2] {
                macro_rules! tap8 {
                    ($bp:expr, $aoff:expr) => {
                        let b0 = _mm256_loadu_si256(($bp) as *const __m256i);
                        let wp = ap.add(($aoff) * MR);
                        c0 = _mm256_dpbusd_epi32(c0, b0, _mm256_set1_epi32(*wp));
                        c1 = _mm256_dpbusd_epi32(c1, b0, _mm256_set1_epi32(*wp.add(1)));
                        c2 = _mm256_dpbusd_epi32(c2, b0, _mm256_set1_epi32(*wp.add(2)));
                        c3 = _mm256_dpbusd_epi32(c3, b0, _mm256_set1_epi32(*wp.add(3)));
                    };
                }
                if IS_S2 {
                    let od = base.add(ev_w);
                    tap8!(od, 0);
                    tap8!(base, 1);
                    tap8!(od.add(1), 2);
                } else {
                    tap8!(base, 0);
                    tap8!(base.add(1), 1);
                    tap8!(base.add(2), 2);
                }
                ap = ap.wrapping_add(3 * MR);
            }
            p0 = p0.wrapping_add(group_stride);
            p1 = p1.wrapping_add(group_stride);
            p2 = p2.wrapping_add(group_stride);
        }
        _mm256_store_si256(acc.0[0].as_mut_ptr() as *mut __m256i, c0);
        _mm256_store_si256(acc.0[1].as_mut_ptr() as *mut __m256i, c1);
        _mm256_store_si256(acc.0[2].as_mut_ptr() as *mut __m256i, c2);
        _mm256_store_si256(acc.0[3].as_mut_ptr() as *mut __m256i, c3);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn compute_3x3_row<const IS_S2: bool, const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
    r: &RowArgs<'_>,
) {
    let p = r.p;
    let (img, oy, ob_start, ob_end) = (r.img, r.oy, r.ob_start, r.ob_end);
    let (group_stride, ev_w, slot_words, ord, buf) = (r.group_stride, r.ev_w, r.slot_words, r.ord, r.buf);
    let ohw = p.oh * p.ow;
    let ks = p.groups * 9;
    let res_img_ptr = match &p.res_view {
        Some(rv) => rv.data[rv.offset + img * rv.img_stride..].as_ptr(),
        None => std::ptr::null(),
    };
    let q_img_ptr = if OUT_U8 { p.qptr.at(img * p.og * ohw) } else { std::ptr::null_mut() };
    let f_img_ptr = if !OUT_U8 { p.fptr.at(img * p.og * 4 * ohw) } else { std::ptr::null_mut() };
    let mut acc512 = AlignedTile512([[0; NR512]; MR]);
    let mut acc24 = AlignedTile([[0; NR]; MR]);
    let mut qw512 = [0u32; NR512];
    let mut qw24 = [0u32; NR];
    let mut resw512: [u32; NR512];
    let mut resw24: [u32; NR];
    let mut fv512 = [[0f32; NR512]; MR];
    let mut fv24 = [[0f32; NR]; MR];
    let bptr = buf.as_ptr();
    let row_off = oy * p.ow;
    // SAFETY: caller validated all buffer bounds and disjoint `(img, oy, ob_start..ob_end)` ownership;
    // `ow8 = ow.next_multiple_of(8)` guarantees `col + 16 <= ow8` for `rem >= 9` and `col + 8 <= ow8` for `rem <= 8`
    // whenever `col` is a multiple of 16.
    unsafe {
        let slots =
            [bptr.add(ord[0] * slot_words), bptr.add(ord[1] * slot_words), bptr.add(ord[2] * slot_words)];
        for g in ob_start..ob_end {
            let a_panel = p.a.as_ptr().add(g * ks * MR);
            let corr: [i32; MR] = p.corr[4 * g..4 * g + 4].try_into().unwrap();
            let scale: [f32; MR] = p.scale[4 * g..4 * g + 4].try_into().unwrap();
            let bias: [f32; MR] = p.bias[4 * g..4 * g + 4].try_into().unwrap();
            let (inv, zero, z01_arr, z23_arr) = if OUT_U8 {
                let (inv, zero) = &p.group_out[g];
                let (z01, z23) = zero_pairs(zero);
                (*inv, *zero, z01, z23)
            } else {
                ([0.0; MR], [0; MR], [0i16; 16], [0i16; 16])
            };
            let (rs, rz) = if HAS_RES { p.group_res[g] } else { ([0.0; MR], [0; MR]) };
            let res_plane = if HAS_RES { res_img_ptr.add(g * ohw) } else { std::ptr::null() };
            let q_plane = if OUT_U8 { q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
            let f_base = if !OUT_U8 { f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
            macro_rules! tile_z {
                ($col:expr, $zvecs:literal, $len:expr, $direct:expr) => {{
                    k_loop_3x3_512::<IS_S2, $zvecs>(
                        p.groups,
                        a_panel,
                        slots,
                        group_stride,
                        ev_w,
                        $col,
                        &mut acc512,
                    );
                    let rp = if HAS_RES {
                        if $direct {
                            res_plane.add(row_off + $col)
                        } else {
                            resw512 = [0u32; NR512];
                            std::ptr::copy_nonoverlapping(
                                res_plane.add(row_off + $col),
                                resw512.as_mut_ptr(),
                                $len,
                            );
                            resw512.as_ptr()
                        }
                    } else {
                        std::ptr::null()
                    };
                    apply_epi_512::<SILU, HAS_RES, OUT_U8, $zvecs>(
                        &acc512,
                        &corr,
                        &scale,
                        &bias,
                        rp,
                        &rs,
                        &rz,
                        &inv,
                        &zero,
                        if OUT_U8 {
                            if $direct { q_plane.add(row_off + $col) } else { qw512.as_mut_ptr() }
                        } else {
                            std::ptr::null_mut()
                        },
                        &mut fv512,
                    );
                    if OUT_U8 {
                        if !$direct {
                            std::ptr::copy_nonoverlapping(qw512.as_ptr(), q_plane.add(row_off + $col), $len);
                        }
                    } else {
                        for (rr, row) in fv512.iter().enumerate() {
                            std::ptr::copy_nonoverlapping(
                                row.as_ptr(),
                                f_base.add(rr * ohw + row_off + $col),
                                $len,
                            );
                        }
                    }
                }};
            }
            let mut col = 0;
            while col + NR512 <= p.ow {
                tile_z!(col, 3, NR512, true);
                col += NR512;
            }
            let mut rem = p.ow - col;
            if rem >= 32 {
                tile_z!(col, 2, 32, true);
                col += 32;
                rem -= 32;
            } else if rem >= 16 {
                tile_z!(col, 1, 16, true);
                col += 16;
                rem -= 16;
            }
            if rem > 0 {
                if rem <= 8 {
                    k_loop_3x3_8::<IS_S2>(p.groups, a_panel, slots, group_stride, ev_w, col, &mut acc24);
                    let rp = if HAS_RES {
                        if rem == 8 {
                            res_plane.add(row_off + col)
                        } else {
                            resw24 = [0u32; NR];
                            std::ptr::copy_nonoverlapping(
                                res_plane.add(row_off + col),
                                resw24.as_mut_ptr(),
                                rem,
                            );
                            resw24.as_ptr()
                        }
                    } else {
                        std::ptr::null()
                    };
                    if OUT_U8 {
                        if rem == 8 {
                            apply_epi_aligned::<SILU, HAS_RES, true, 1>(
                                &acc24,
                                &corr,
                                &scale,
                                &bias,
                                rp,
                                &rs,
                                &rz,
                                &inv,
                                &z01_arr,
                                &z23_arr,
                                q_plane.add(row_off + col),
                                &mut fv24,
                            );
                        } else {
                            apply_epi_aligned::<SILU, HAS_RES, true, 1>(
                                &acc24,
                                &corr,
                                &scale,
                                &bias,
                                rp,
                                &rs,
                                &rz,
                                &inv,
                                &z01_arr,
                                &z23_arr,
                                qw24.as_mut_ptr(),
                                &mut fv24,
                            );
                            std::ptr::copy_nonoverlapping(qw24.as_ptr(), q_plane.add(row_off + col), rem);
                        }
                    } else {
                        apply_epi_aligned::<SILU, HAS_RES, false, 1>(
                            &acc24,
                            &corr,
                            &scale,
                            &bias,
                            rp,
                            &rs,
                            &rz,
                            &inv,
                            &z01_arr,
                            &z23_arr,
                            std::ptr::null_mut(),
                            &mut fv24,
                        );
                        for (rr, row) in fv24.iter().enumerate() {
                            std::ptr::copy_nonoverlapping(
                                row.as_ptr(),
                                f_base.add(rr * ohw + row_off + col),
                                rem,
                            );
                        }
                    }
                } else {
                    tile_z!(col, 1, rem, false);
                }
            }
        }
    }
}

/// Execute one `3×3` rotating row-strip output row on 512-bit AVX-512 VNNI.
///
/// # Safety
/// CPU supports `avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni`; row ownership and extents are valid.
pub(in super::super) unsafe fn run_3x3_row<const IS_S2: bool>(r: &RowArgs<'_>) {
    let has_res = r.p.res_view.is_some();
    // SAFETY: forwarded from caller.
    unsafe {
        match (r.p.silu, has_res, r.p.is_u8) {
            (true, false, true) => compute_3x3_row::<IS_S2, true, false, true>(r),
            (false, false, true) => compute_3x3_row::<IS_S2, false, false, true>(r),
            (true, true, true) => compute_3x3_row::<IS_S2, true, true, true>(r),
            (false, true, true) => compute_3x3_row::<IS_S2, false, true, true>(r),
            (true, false, false) => compute_3x3_row::<IS_S2, true, false, false>(r),
            (false, false, false) => compute_3x3_row::<IS_S2, false, false, false>(r),
            (true, true, false) => compute_3x3_row::<IS_S2, true, true, false>(r),
            (false, true, false) => compute_3x3_row::<IS_S2, false, true, false>(r),
        }
    }
}
