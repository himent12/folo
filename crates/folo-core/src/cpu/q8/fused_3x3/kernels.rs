//! 256-bit x86-64 microkernels (AVX-VNNI, AVX-512 VNNI VL, AVX2) for the fused 3×3 row-strip driver.

use std::arch::x86_64::*;

use super::super::fused_epi::{AlignedTile, apply_epi_aligned_fstride, zero_pairs};
use super::super::kernel::{MR, NR};
use super::BatchRowArgs;

macro_rules! define_3x3_strip_isa {
    (
        mod $mod_name:ident,
        feature = $feat:literal,
        dp_mode = $dp_mode:ident
    ) => {
        pub(super) mod $mod_name {
            use super::*;

            #[target_feature(enable = $feat)]
            #[inline]
            #[allow(clippy::too_many_arguments)]
            unsafe fn k_loop_s2<const VECS: usize>(
                groups: usize,
                a: *const i32,
                slots: [*const u32; 3],
                ch_stride: usize,
                ev_w: usize,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `groups * 9 * MR` weights and valid `(ev, od)` extents per slot
                // (`col + VECS * 8 <= ev_w`). Trailing `wrapping_add` pointers are never dereferenced.
                unsafe {
                    let mut p0 = slots[0].add(col);
                    let mut p1 = slots[1].add(col);
                    let mut p2 = slots[2].add(col);
                    let mut ap = a;
                    for _ in 0..groups {
                        macro_rules! tap {
                                    ($bp:expr, $aoff:expr) => {
                                        let bp = ($bp) as *const __m256i;
                                        let b0 = _mm256_loadu_si256(bp);
                                        let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(1)) } else { z };
                                        let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(2)) } else { z };
                                        let wp = ap.add(($aoff) * MR);
                                        define_3x3_strip_isa!(@dp $dp_mode, VECS, b0, b1, b2, wp,
                                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                                    };
                                }
                        macro_rules! row_s2 {
                            ($ev:expr, $base:expr) => {
                                let od = ($ev).add(ev_w);
                                tap!(od, $base);
                                tap!($ev, $base + 1);
                                tap!(od.add(1), $base + 2);
                            };
                        }
                        row_s2!(p0, 0);
                        row_s2!(p1, 3);
                        row_s2!(p2, 6);
                        ap = ap.wrapping_add(9 * MR);
                        p0 = p0.wrapping_add(ch_stride);
                        p1 = p1.wrapping_add(ch_stride);
                        p2 = p2.wrapping_add(ch_stride);
                    }
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(1), c01);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(2), c02);
                    }
                    _mm256_store_si256(out.add(3), c10);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(4), c11);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(5), c12);
                    }
                    _mm256_store_si256(out.add(6), c20);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(7), c21);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(8), c22);
                    }
                    _mm256_store_si256(out.add(9), c30);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(10), c31);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(11), c32);
                    }
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            #[allow(clippy::too_many_arguments)]
            unsafe fn k_loop_s1<const VECS: usize>(
                groups: usize,
                a: *const i32,
                slots: [*const u32; 3],
                slot_w: usize,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `groups * 9 * MR` weights and `col + VECS * 8 + 2 <= slot_w` per group row.
                // Trailing `wrapping_add` pointers are never dereferenced.
                unsafe {
                    let mut p0 = slots[0].add(col);
                    let mut p1 = slots[1].add(col);
                    let mut p2 = slots[2].add(col);
                    let mut ap = a;
                    for _ in 0..groups {
                        macro_rules! tap {
                                    ($bp:expr, $aoff:expr) => {
                                        let bp = ($bp) as *const __m256i;
                                        let b0 = _mm256_loadu_si256(bp);
                                        let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(1)) } else { z };
                                        let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(2)) } else { z };
                                        let wp = ap.add(($aoff) * MR);
                                        define_3x3_strip_isa!(@dp $dp_mode, VECS, b0, b1, b2, wp,
                                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                                    };
                                }
                        macro_rules! row_s1 {
                            ($rp:expr, $base:expr) => {
                                tap!($rp, $base);
                                tap!(($rp).add(1), $base + 1);
                                tap!(($rp).add(2), $base + 2);
                            };
                        }
                        row_s1!(p0, 0);
                        row_s1!(p1, 3);
                        row_s1!(p2, 6);
                        ap = ap.wrapping_add(9 * MR);
                        p0 = p0.wrapping_add(slot_w);
                        p1 = p1.wrapping_add(slot_w);
                        p2 = p2.wrapping_add(slot_w);
                    }
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(1), c01);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(2), c02);
                    }
                    _mm256_store_si256(out.add(3), c10);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(4), c11);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(5), c12);
                    }
                    _mm256_store_si256(out.add(6), c20);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(7), c21);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(8), c22);
                    }
                    _mm256_store_si256(out.add(9), c30);
                    if VECS >= 2 {
                        _mm256_store_si256(out.add(10), c31);
                    }
                    if VECS >= 3 {
                        _mm256_store_si256(out.add(11), c32);
                    }
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn compute_rows<
                const IS_S2: bool,
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
            >(
                r: &BatchRowArgs<'_>,
            ) {
                let p = r.p;
                let (img, oy_start, rows_len, ob_start, ob_end) =
                    (r.img, r.oy_start, r.rows_len, r.ob_start, r.ob_end);
                let (group_stride, ev_w, slot_words, ord, buf) =
                    (r.group_stride, r.ev_w, r.slot_words, r.ord, r.buf);
                let stride = if IS_S2 { 2usize } else { 1usize };
                let ohw = p.oh * p.ow;
                let full_chunks = p.ow / NR;
                let tail = p.ow % NR;
                let ks = p.groups * 9;
                let res_img_ptr = match &p.res_view {
                    Some(rv) => rv.data[rv.offset + img * rv.img_stride..].as_ptr(),
                    None => std::ptr::null(),
                };
                let q_img_ptr = if OUT_U8 { p.qptr.at(img * p.og * ohw) } else { std::ptr::null_mut() };
                let f_img_ptr = if !OUT_U8 { p.fptr.at(img * p.og * 4 * ohw) } else { std::ptr::null_mut() };
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut resw: [u32; NR];
                let mut fv = [[0f32; NR]; MR];
                let bptr = buf.as_ptr();
                // SAFETY: caller validated all buffer bounds, `rows_len <= 4`, slot indices, and disjoint
                // `(img, oy_start..oy_start + rows_len, ob_start..ob_end)` ownership.
                unsafe {
                    for g in ob_start..ob_end {
                        let a_panel = p.a.as_ptr().add(g * ks * MR);
                        let corr: [i32; MR] = p.corr[4 * g..4 * g + 4].try_into().unwrap();
                        let scale: [f32; MR] = p.scale[4 * g..4 * g + 4].try_into().unwrap();
                        let bias: [f32; MR] = p.bias[4 * g..4 * g + 4].try_into().unwrap();
                        let (inv, z01_arr, z23_arr) = if OUT_U8 {
                            let (inv, zero) = &p.group_out[g];
                            let (z01, z23) = zero_pairs(zero);
                            (*inv, z01, z23)
                        } else {
                            ([0.0; MR], [0i16; 16], [0i16; 16])
                        };
                        let (rs, rz) = if HAS_RES { p.group_res[g] } else { ([0.0; MR], [0; MR]) };
                        let res_plane = if HAS_RES { res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                        let q_plane = if OUT_U8 { q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                        let f_base = if !OUT_U8 { f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                        for r_idx in 0..rows_len {
                            let row_off = (oy_start + r_idx) * p.ow;
                            let s0 = r_idx * stride;
                            let slots = [
                                bptr.add(ord[s0] * slot_words),
                                bptr.add(ord[s0 + 1] * slot_words),
                                bptr.add(ord[s0 + 2] * slot_words),
                            ];
                            for chunk in 0..full_chunks {
                                let col = chunk * NR;
                                if IS_S2 {
                                    k_loop_s2::<3>(
                                        p.groups,
                                        a_panel,
                                        slots,
                                        group_stride,
                                        ev_w,
                                        col,
                                        &mut acc,
                                    );
                                } else {
                                    k_loop_s1::<3>(p.groups, a_panel, slots, group_stride, col, &mut acc);
                                }
                                let res_ptr =
                                    if HAS_RES { res_plane.add(row_off + col) } else { std::ptr::null() };
                                apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, 3>(
                                    &acc,
                                    &corr,
                                    &scale,
                                    &bias,
                                    res_ptr,
                                    &rs,
                                    &rz,
                                    &inv,
                                    &z01_arr,
                                    &z23_arr,
                                    if OUT_U8 { q_plane.add(row_off + col) } else { std::ptr::null_mut() },
                                    if !OUT_U8 { f_base.add(row_off + col) } else { std::ptr::null_mut() },
                                    ohw,
                                );
                            }
                            if tail > 0 {
                                let col = full_chunks * NR;
                                macro_rules! run_tail {
                                    ($vecs:literal) => {{
                                        if IS_S2 {
                                            k_loop_s2::<$vecs>(
                                                p.groups,
                                                a_panel,
                                                slots,
                                                group_stride,
                                                ev_w,
                                                col,
                                                &mut acc,
                                            );
                                        } else {
                                            k_loop_s1::<$vecs>(
                                                p.groups,
                                                a_panel,
                                                slots,
                                                group_stride,
                                                col,
                                                &mut acc,
                                            );
                                        }
                                        let direct = tail == $vecs * 8;
                                        let res_ptr = if HAS_RES {
                                            if direct {
                                                res_plane.add(row_off + col)
                                            } else {
                                                resw = [0u32; NR];
                                                std::ptr::copy_nonoverlapping(
                                                    res_plane.add(row_off + col),
                                                    resw.as_mut_ptr(),
                                                    tail,
                                                );
                                                resw.as_ptr()
                                            }
                                        } else {
                                            std::ptr::null()
                                        };
                                        let q_dst = if OUT_U8 {
                                            if direct { q_plane.add(row_off + col) } else { qw.as_mut_ptr() }
                                        } else {
                                            std::ptr::null_mut()
                                        };
                                        let (f_dst, f_stride) = if !OUT_U8 {
                                            if direct {
                                                (f_base.add(row_off + col), ohw)
                                            } else {
                                                (fv.as_mut_ptr() as *mut f32, NR)
                                            }
                                        } else {
                                            (std::ptr::null_mut(), 0)
                                        };
                                        apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, $vecs>(
                                            &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv, &z01_arr,
                                            &z23_arr, q_dst, f_dst, f_stride,
                                        );
                                        if OUT_U8 {
                                            if !direct {
                                                std::ptr::copy_nonoverlapping(
                                                    qw.as_ptr(),
                                                    q_plane.add(row_off + col),
                                                    tail,
                                                );
                                            }
                                        } else if !direct {
                                            for (rr, row) in fv.iter().enumerate() {
                                                std::ptr::copy_nonoverlapping(
                                                    row.as_ptr(),
                                                    f_base.add(rr * ohw + row_off + col),
                                                    tail,
                                                );
                                            }
                                        }
                                    }};
                                }
                                if tail <= 8 {
                                    run_tail!(1);
                                } else if tail <= 16 {
                                    run_tail!(2);
                                } else {
                                    run_tail!(3);
                                }
                            }
                        }
                    }
                }
            }

            pub(in super::super) unsafe fn run_rows<const IS_S2: bool>(r: &BatchRowArgs<'_>) {
                let has_res = r.p.res_view.is_some();
                // SAFETY: ISA checked by caller.
                unsafe {
                    match (r.p.silu, has_res, r.p.is_u8) {
                        (true, false, true) => compute_rows::<IS_S2, true, false, true>(r),
                        (false, false, true) => compute_rows::<IS_S2, false, false, true>(r),
                        (true, true, true) => compute_rows::<IS_S2, true, true, true>(r),
                        (false, true, true) => compute_rows::<IS_S2, false, true, true>(r),
                        (true, false, false) => compute_rows::<IS_S2, true, false, false>(r),
                        (false, false, false) => compute_rows::<IS_S2, false, false, false>(r),
                        (true, true, false) => compute_rows::<IS_S2, true, true, false>(r),
                        (false, true, false) => compute_rows::<IS_S2, false, true, false>(r),
                    }
                }
            }
        }
    };

    (@dp avxvnni, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_avx_epi32($c00, $b0, w0);
        if $vecs >= 2 {
            $c01 = _mm256_dpbusd_avx_epi32($c01, $b1, w0);
        }
        if $vecs >= 3 {
            $c02 = _mm256_dpbusd_avx_epi32($c02, $b2, w0);
        }
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_avx_epi32($c10, $b0, w1);
        if $vecs >= 2 {
            $c11 = _mm256_dpbusd_avx_epi32($c11, $b1, w1);
        }
        if $vecs >= 3 {
            $c12 = _mm256_dpbusd_avx_epi32($c12, $b2, w1);
        }
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_avx_epi32($c20, $b0, w2);
        if $vecs >= 2 {
            $c21 = _mm256_dpbusd_avx_epi32($c21, $b1, w2);
        }
        if $vecs >= 3 {
            $c22 = _mm256_dpbusd_avx_epi32($c22, $b2, w2);
        }
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_avx_epi32($c30, $b0, w3);
        if $vecs >= 2 {
            $c31 = _mm256_dpbusd_avx_epi32($c31, $b1, w3);
        }
        if $vecs >= 3 {
            $c32 = _mm256_dpbusd_avx_epi32($c32, $b2, w3);
        }
    };

    (@dp avx512vnni, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_epi32($c00, $b0, w0);
        if $vecs >= 2 {
            $c01 = _mm256_dpbusd_epi32($c01, $b1, w0);
        }
        if $vecs >= 3 {
            $c02 = _mm256_dpbusd_epi32($c02, $b2, w0);
        }
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_epi32($c10, $b0, w1);
        if $vecs >= 2 {
            $c11 = _mm256_dpbusd_epi32($c11, $b1, w1);
        }
        if $vecs >= 3 {
            $c12 = _mm256_dpbusd_epi32($c12, $b2, w1);
        }
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_epi32($c20, $b0, w2);
        if $vecs >= 2 {
            $c21 = _mm256_dpbusd_epi32($c21, $b1, w2);
        }
        if $vecs >= 3 {
            $c22 = _mm256_dpbusd_epi32($c22, $b2, w2);
        }
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_epi32($c30, $b0, w3);
        if $vecs >= 2 {
            $c31 = _mm256_dpbusd_epi32($c31, $b1, w3);
        }
        if $vecs >= 3 {
            $c32 = _mm256_dpbusd_epi32($c32, $b2, w3);
        }
    };

    (@dp avx2, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {{
        let m_ff = _mm256_set1_epi16(0x00FF);
        let (w0e, w0o) = split_w_avx2(*$ap);
        let (w1e, w1o) = split_w_avx2(*$ap.add(1));
        let (w2e, w2o) = split_w_avx2(*$ap.add(2));
        let (w3e, w3o) = split_w_avx2(*$ap.add(3));
        macro_rules! col_step {
            ($b:expr, $r0:ident, $r1:ident, $r2:ident, $r3:ident) => {
                let b_ev = _mm256_and_si256($b, m_ff);
                let b_od = _mm256_srli_epi16::<8>($b);
                $r0 = _mm256_add_epi32($r0, _mm256_madd_epi16(b_ev, _mm256_set1_epi32(w0e)));
                $r0 = _mm256_add_epi32($r0, _mm256_madd_epi16(b_od, _mm256_set1_epi32(w0o)));
                $r1 = _mm256_add_epi32($r1, _mm256_madd_epi16(b_ev, _mm256_set1_epi32(w1e)));
                $r1 = _mm256_add_epi32($r1, _mm256_madd_epi16(b_od, _mm256_set1_epi32(w1o)));
                $r2 = _mm256_add_epi32($r2, _mm256_madd_epi16(b_ev, _mm256_set1_epi32(w2e)));
                $r2 = _mm256_add_epi32($r2, _mm256_madd_epi16(b_od, _mm256_set1_epi32(w2o)));
                $r3 = _mm256_add_epi32($r3, _mm256_madd_epi16(b_ev, _mm256_set1_epi32(w3e)));
                $r3 = _mm256_add_epi32($r3, _mm256_madd_epi16(b_od, _mm256_set1_epi32(w3o)));
            };
        }
        col_step!($b0, $c00, $c10, $c20, $c30);
        if $vecs >= 2 {
            col_step!($b1, $c01, $c11, $c21, $c31);
        }
        if $vecs >= 3 {
            col_step!($b2, $c02, $c12, $c22, $c32);
        }
    }};
}

#[inline(always)]
fn split_w_avx2(w: i32) -> (i32, i32) {
    let b = w.to_le_bytes();
    let ev = (b[0] as i8 as i16 as u16 as u32) | ((b[2] as i8 as i16 as u16 as u32) << 16);
    let od = (b[1] as i8 as i16 as u16 as u32) | ((b[3] as i8 as i16 as u16 as u32) << 16);
    (ev as i32, od as i32)
}

define_3x3_strip_isa!(mod vnni_3x3, feature = "avx2,fma,avxvnni", dp_mode = avxvnni);
define_3x3_strip_isa!(mod avx512vnni_3x3, feature = "avx2,fma,avx512vnni,avx512vl", dp_mode = avx512vnni);
define_3x3_strip_isa!(mod avx2_3x3, feature = "avx2,fma", dp_mode = avx2);
