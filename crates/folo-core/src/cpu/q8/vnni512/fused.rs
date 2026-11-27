//! 512-bit AVX-512 VNNI fused tile-block drivers for single-part convs, decoupled multi-part convs, and the
//! direct `u8` RGB0 stem.

use std::arch::x86_64::*;

use super::super::fused_epi::{
    AlignedTile, BMode, MultiPartTask, SinglePartTask, apply_epi_aligned, finish_col8, read_u8_noncontiguous,
    write_f32_noncontiguous, write_u8_noncontiguous, zero_pairs,
};
use super::super::kernel::{MR, NR};
use super::{
    AlignedTile512, AlignedTileF32_512, NR512, apply_epi_512, dp512, finish_col16, k_loop_1x1_512,
    k_loop_bmode_8, k_loop_bmode_512, k_loop_rows_512, kernel_avx512vnni_zmm, store_tile512,
};

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn run_single_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(t: &SinglePartTask<'_>) {
    let ohw = t.oh * t.ow;
    let ks = t.ks;
    let mut acc512 = AlignedTile512([[0; NR512]; MR]);
    let mut acc24 = AlignedTile([[0; NR]; MR]);
    let mut qw512 = [0u32; NR512];
    let mut qw24 = [0u32; NR];
    let mut resw512: [u32; NR512];
    let mut resw24 = [0u32; NR];
    let mut fv512 = [[0f32; NR512]; MR];
    let mut fv24 = [[0f32; NR]; MR];
    // SAFETY: caller validated all slice extents and disjoint task ownership.
    unsafe {
        if t.wq == t.ow {
            let col_start = t.cb_start * NR;
            let col_end = (t.cb_end * NR).min(t.r_n);
            let col_tile = ((24 * 1024 / (ks.max(1) * NR512 * 4)).clamp(1, 8)) * NR512;
            let mut c0 = col_start;
            while c0 < col_end {
                let c1 = (c0 + col_tile).min(col_end);
                for g in t.ob_start..t.ob_end {
                    let a_panel = t.a.as_ptr().add(g * ks * MR);
                    let corr: [i32; MR] = t.corr[4 * g..4 * g + 4].try_into().unwrap();
                    let scale: [f32; MR] = t.scale[4 * g..4 * g + 4].try_into().unwrap();
                    let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                    let (inv, zero, z01_arr, z23_arr) = if OUT_U8 {
                        let (inv, zero) = &t.group_out[g];
                        let (z01, z23) = zero_pairs(zero);
                        (*inv, *zero, z01, z23)
                    } else {
                        ([0.0; MR], [0; MR], [0i16; 16], [0i16; 16])
                    };
                    let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                    let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                    let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                    let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                    macro_rules! full_z {
                        ($col:expr, $zvecs:literal) => {{
                            let w = $zvecs * 16;
                            k_loop_bmode_512::<$zvecs>(
                                t.b_mode,
                                ks,
                                a_panel,
                                t.rows.as_ptr(),
                                $col,
                                &mut acc512,
                            );
                            let rp = if HAS_RES { res_plane.add($col) } else { std::ptr::null() };
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
                                if OUT_U8 { q_plane.add($col) } else { std::ptr::null_mut() },
                                &mut fv512,
                            );
                            if !OUT_U8 {
                                for (rr, row) in fv512.iter().enumerate() {
                                    std::ptr::copy_nonoverlapping(
                                        row.as_ptr(),
                                        f_base.add(rr * ohw + $col),
                                        w,
                                    );
                                }
                            }
                        }};
                    }
                    let mut col = c0;
                    while col + NR512 <= c1 {
                        full_z!(col, 3);
                        col += NR512;
                    }
                    let mut rem = c1 - col;
                    if rem >= 32 {
                        full_z!(col, 2);
                        col += 32;
                        rem -= 32;
                    } else if rem >= 16 {
                        full_z!(col, 1);
                        col += 16;
                        rem -= 16;
                    }
                    if rem > 0 {
                        if rem <= 8 {
                            k_loop_bmode_8(t.b_mode, ks, a_panel, t.rows.as_ptr(), col, &mut acc24);
                            let rp = if HAS_RES {
                                if rem == 8 {
                                    res_plane.add(col)
                                } else {
                                    resw24 = [0u32; NR];
                                    std::ptr::copy_nonoverlapping(
                                        res_plane.add(col),
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
                                        q_plane.add(col),
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
                                    std::ptr::copy_nonoverlapping(qw24.as_ptr(), q_plane.add(col), rem);
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
                                        f_base.add(rr * ohw + col),
                                        rem,
                                    );
                                }
                            }
                        } else {
                            // `9 <= rem <= 15`: `col < r_n` and `SLACK = 64 >= 16` guarantees 16 readable B words.
                            k_loop_bmode_512::<1>(t.b_mode, ks, a_panel, t.rows.as_ptr(), col, &mut acc512);
                            let rp = if HAS_RES {
                                resw512 = [0u32; NR512];
                                std::ptr::copy_nonoverlapping(res_plane.add(col), resw512.as_mut_ptr(), rem);
                                resw512.as_ptr()
                            } else {
                                std::ptr::null()
                            };
                            apply_epi_512::<SILU, HAS_RES, OUT_U8, 1>(
                                &acc512,
                                &corr,
                                &scale,
                                &bias,
                                rp,
                                &rs,
                                &rz,
                                &inv,
                                &zero,
                                qw512.as_mut_ptr(),
                                &mut fv512,
                            );
                            if OUT_U8 {
                                std::ptr::copy_nonoverlapping(qw512.as_ptr(), q_plane.add(col), rem);
                            } else {
                                for (rr, row) in fv512.iter().enumerate() {
                                    std::ptr::copy_nonoverlapping(
                                        row.as_ptr(),
                                        f_base.add(rr * ohw + col),
                                        rem,
                                    );
                                }
                            }
                        }
                    }
                }
                c0 = c1;
            }
            return;
        }
        // General padded-width `Rows` mode (`t.wq != t.ow`): pair contiguous 48-col tiles into ZMMs and use
        // the hybrid 1×ZMM + 1×YMM 24-col kernel on cross-row/tail chunks.
        let cb_tile = (24 * 1024 / (ks.max(1) * NR * 4)).clamp(1, 16);
        let mut c0 = t.cb_start;
        while c0 < t.cb_end {
            let c1 = (c0 + cb_tile).min(t.cb_end);
            for g in t.ob_start..t.ob_end {
                let a_panel = t.a.as_ptr().add(g * ks * MR);
                let corr: [i32; MR] = t.corr[4 * g..4 * g + 4].try_into().unwrap();
                let scale: [f32; MR] = t.scale[4 * g..4 * g + 4].try_into().unwrap();
                let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                let (inv, zero, z01_arr, z23_arr) = if OUT_U8 {
                    let (inv, zero) = &t.group_out[g];
                    let (z01, z23) = zero_pairs(zero);
                    (*inv, *zero, z01, z23)
                } else {
                    ([0.0; MR], [0; MR], [0i16; 16], [0i16; 16])
                };
                let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                let mut chunk = c0;
                while chunk < c1 {
                    let col = chunk * NR;
                    let (oy0, ox0) = (col / t.wq, col % t.wq);
                    if chunk + 1 < c1 && col + NR512 <= t.r_n && oy0 < t.oh && ox0 + NR512 <= t.ow {
                        let pix0 = oy0 * t.ow + ox0;
                        k_loop_rows_512::<3>(ks, a_panel, t.rows.as_ptr(), col, &mut acc512);
                        let rp = if HAS_RES { res_plane.add(pix0) } else { std::ptr::null() };
                        apply_epi_512::<SILU, HAS_RES, OUT_U8, 3>(
                            &acc512,
                            &corr,
                            &scale,
                            &bias,
                            rp,
                            &rs,
                            &rz,
                            &inv,
                            &zero,
                            if OUT_U8 { q_plane.add(pix0) } else { std::ptr::null_mut() },
                            &mut fv512,
                        );
                        if !OUT_U8 {
                            for (rr, row) in fv512.iter().enumerate() {
                                std::ptr::copy_nonoverlapping(
                                    row.as_ptr(),
                                    f_base.add(rr * ohw + pix0),
                                    NR512,
                                );
                            }
                        }
                        chunk += 2;
                        continue;
                    }
                    let cols = NR.min(t.r_n - col);
                    let contiguous = cols == NR && oy0 < t.oh && ox0 + NR <= t.ow;
                    let pix0 = oy0 * t.ow + ox0;
                    kernel_avx512vnni_zmm(ks, a_panel, t.rows.as_ptr(), col, &mut acc24.0);
                    let rp = if HAS_RES {
                        if contiguous {
                            res_plane.add(pix0)
                        } else {
                            read_u8_noncontiguous(&mut resw24, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane);
                            resw24.as_ptr()
                        }
                    } else {
                        std::ptr::null()
                    };
                    if OUT_U8 {
                        if contiguous {
                            apply_epi_aligned::<SILU, HAS_RES, true, 3>(
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
                                q_plane.add(pix0),
                                &mut fv24,
                            );
                        } else {
                            apply_epi_aligned::<SILU, HAS_RES, true, 3>(
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
                            write_u8_noncontiguous(&qw24, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane);
                        }
                    } else {
                        apply_epi_aligned::<SILU, HAS_RES, false, 3>(
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
                        if contiguous {
                            for (rr, row) in fv24.iter().enumerate() {
                                std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + pix0), NR);
                            }
                        } else {
                            write_f32_noncontiguous(&fv24, cols, oy0, ox0, t.oh, t.ow, t.wq, ohw, f_base);
                        }
                    }
                    chunk += 1;
                }
            }
            c0 = c1;
        }
    }
}

/// Execute one single-part fused task on 512-bit AVX-512 VNNI.
///
/// # Safety
/// CPU supports `avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni`; task bounds and pointers are valid and disjoint.
pub(in super::super) unsafe fn run_single_part(t: &SinglePartTask<'_>) {
    let has_res = !t.res_img_ptr.is_null();
    // SAFETY: forwarded from caller.
    unsafe {
        match (t.silu, has_res, t.is_u8) {
            (true, false, true) => run_single_spec::<true, false, true>(t),
            (false, false, true) => run_single_spec::<false, false, true>(t),
            (true, true, true) => run_single_spec::<true, true, true>(t),
            (false, true, true) => run_single_spec::<false, true, true>(t),
            (true, false, false) => run_single_spec::<true, false, false>(t),
            (false, false, false) => run_single_spec::<false, false, false>(t),
            (true, true, false) => run_single_spec::<true, true, false>(t),
            (false, true, false) => run_single_spec::<false, true, false>(t),
        }
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn run_multi_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(t: &MultiPartTask<'_>) {
    let ohw = t.oh * t.ow;
    let total_ks: usize = t.parts.iter().map(|p| p.ks).sum();
    let mut acc512 = AlignedTile512([[0; NR512]; MR]);
    let mut tile512 = AlignedTileF32_512([[0.0; NR512]; MR]);
    let mut acc24 = AlignedTile([[0; NR]; MR]);
    let mut tile24: [[f32; NR]; MR];
    let mut qw512 = [0u32; NR512];
    let mut qw24 = [0u32; NR];
    let mut resw512: [u32; NR512];
    let mut resw24 = [0u32; NR];
    let mut fv512 = [[0f32; NR512]; MR];
    let mut fv24 = [[0f32; NR]; MR];
    // SAFETY: caller validated all slice extents and disjoint task ownership.
    unsafe {
        if t.wq == t.ow {
            let col_start = t.cb_start * NR;
            let col_end = (t.cb_end * NR).min(t.r_n);
            let col_tile = ((24 * 1024 / (total_ks.max(1) * NR512 * 4)).clamp(1, 8)) * NR512;
            let mut c0 = col_start;
            while c0 < col_end {
                let c1 = (c0 + col_tile).min(col_end);
                for g in t.ob_start..t.ob_end {
                    let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                    let (inv, zero) = if OUT_U8 { t.group_out[g] } else { ([0.0; MR], [0; MR]) };
                    let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                    let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                    let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                    let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                    macro_rules! multi_z {
                        ($col:expr, $zvecs:literal, $len:expr, $direct:expr) => {{
                            let zf = _mm512_setzero_ps();
                            for r in 0..MR {
                                for k in 0..$zvecs {
                                    _mm512_store_ps(tile512.0[r].as_mut_ptr().add(16 * k), zf);
                                }
                            }
                            for part in t.parts {
                                if t.pointwise_hw > 0 {
                                    k_loop_1x1_512::<$zvecs>(
                                        part.ks,
                                        part.a.as_ptr().add(g * part.ks * MR),
                                        part.b_base,
                                        t.pointwise_hw,
                                        $col,
                                        &mut acc512,
                                    );
                                } else {
                                    k_loop_rows_512::<$zvecs>(
                                        part.ks,
                                        part.a.as_ptr().add(g * part.ks * MR),
                                        t.rows.as_ptr().add(part.row_offset),
                                        $col,
                                        &mut acc512,
                                    );
                                }
                                for r in 0..MR {
                                    let vc = _mm512_set1_epi32(part.corr[4 * g + r]);
                                    let vs = _mm512_set1_ps(part.scale[4 * g + r]);
                                    for k in 0..$zvecs {
                                        let off = 16 * k;
                                        let av = _mm512_load_si512(
                                            acc512.0[r].as_ptr().add(off) as *const __m512i
                                        );
                                        let cur = _mm512_load_ps(tile512.0[r].as_ptr().add(off));
                                        let next = _mm512_fmadd_ps(
                                            _mm512_cvtepi32_ps(_mm512_sub_epi32(av, vc)),
                                            vs,
                                            cur,
                                        );
                                        _mm512_store_ps(tile512.0[r].as_mut_ptr().add(off), next);
                                    }
                                }
                            }
                            let rp = if HAS_RES {
                                if $direct {
                                    res_plane.add($col)
                                } else {
                                    resw512 = [0u32; NR512];
                                    std::ptr::copy_nonoverlapping(
                                        res_plane.add($col),
                                        resw512.as_mut_ptr(),
                                        $len,
                                    );
                                    resw512.as_ptr()
                                }
                            } else {
                                std::ptr::null()
                            };
                            let q_dst = if OUT_U8 {
                                if $direct { q_plane.add($col) } else { qw512.as_mut_ptr() }
                            } else {
                                std::ptr::null_mut()
                            };
                            let vb0 = _mm512_set1_ps(bias[0]);
                            let vb1 = _mm512_set1_ps(bias[1]);
                            let vb2 = _mm512_set1_ps(bias[2]);
                            let vb3 = _mm512_set1_ps(bias[3]);
                            for k in 0..$zvecs {
                                let off = 16 * k;
                                let y0 = _mm512_add_ps(_mm512_load_ps(tile512.0[0].as_ptr().add(off)), vb0);
                                let y1 = _mm512_add_ps(_mm512_load_ps(tile512.0[1].as_ptr().add(off)), vb1);
                                let y2 = _mm512_add_ps(_mm512_load_ps(tile512.0[2].as_ptr().add(off)), vb2);
                                let y3 = _mm512_add_ps(_mm512_load_ps(tile512.0[3].as_ptr().add(off)), vb3);
                                finish_col16::<SILU, HAS_RES, OUT_U8>(
                                    y0,
                                    y1,
                                    y2,
                                    y3,
                                    if HAS_RES { rp.add(off) } else { rp },
                                    &rs,
                                    &rz,
                                    &inv,
                                    &zero,
                                    if OUT_U8 { q_dst.add(off) } else { q_dst },
                                    &mut fv512,
                                    off,
                                );
                            }
                            if OUT_U8 {
                                if !$direct {
                                    std::ptr::copy_nonoverlapping(qw512.as_ptr(), q_plane.add($col), $len);
                                }
                            } else {
                                for (rr, row) in fv512.iter().enumerate() {
                                    std::ptr::copy_nonoverlapping(
                                        row.as_ptr(),
                                        f_base.add(rr * ohw + $col),
                                        $len,
                                    );
                                }
                            }
                        }};
                    }
                    let mut col = c0;
                    while col + NR512 <= c1 {
                        multi_z!(col, 3, NR512, true);
                        col += NR512;
                    }
                    let mut rem = c1 - col;
                    if rem >= 32 {
                        multi_z!(col, 2, 32, true);
                        col += 32;
                        rem -= 32;
                    } else if rem >= 16 {
                        multi_z!(col, 1, 16, true);
                        col += 16;
                        rem -= 16;
                    }
                    if rem > 0 {
                        multi_z!(col, 1, rem, false);
                    }
                }
                c0 = c1;
            }
            return;
        }
        // General padded-width `Rows` mode (`t.wq != t.ow`).
        let cb_tile = (24 * 1024 / (total_ks.max(1) * NR * 4)).clamp(1, 16);
        let mut c0 = t.cb_start;
        while c0 < t.cb_end {
            let c1 = (c0 + cb_tile).min(t.cb_end);
            for g in t.ob_start..t.ob_end {
                let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                let (inv, zero, z01_arr, z23_arr) = if OUT_U8 {
                    let (inv, zero) = &t.group_out[g];
                    let (z01, z23) = zero_pairs(zero);
                    (*inv, *zero, z01, z23)
                } else {
                    ([0.0; MR], [0; MR], [0i16; 16], [0i16; 16])
                };
                let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                let mut chunk = c0;
                while chunk < c1 {
                    let col = chunk * NR;
                    let (oy0, ox0) = (col / t.wq, col % t.wq);
                    if chunk + 1 < c1 && col + NR512 <= t.r_n && oy0 < t.oh && ox0 + NR512 <= t.ow {
                        let pix0 = oy0 * t.ow + ox0;
                        let zf = _mm512_setzero_ps();
                        for row in &mut tile512.0 {
                            for k in 0..3 {
                                _mm512_store_ps(row.as_mut_ptr().add(16 * k), zf);
                            }
                        }
                        for part in t.parts {
                            k_loop_rows_512::<3>(
                                part.ks,
                                part.a.as_ptr().add(g * part.ks * MR),
                                t.rows.as_ptr().add(part.row_offset),
                                col,
                                &mut acc512,
                            );
                            for r in 0..MR {
                                let vc = _mm512_set1_epi32(part.corr[4 * g + r]);
                                let vs = _mm512_set1_ps(part.scale[4 * g + r]);
                                for k in 0..3 {
                                    let off = 16 * k;
                                    let av =
                                        _mm512_load_si512(acc512.0[r].as_ptr().add(off) as *const __m512i);
                                    let cur = _mm512_load_ps(tile512.0[r].as_ptr().add(off));
                                    let next = _mm512_fmadd_ps(
                                        _mm512_cvtepi32_ps(_mm512_sub_epi32(av, vc)),
                                        vs,
                                        cur,
                                    );
                                    _mm512_store_ps(tile512.0[r].as_mut_ptr().add(off), next);
                                }
                            }
                        }
                        let rp = if HAS_RES { res_plane.add(pix0) } else { std::ptr::null() };
                        let q_dst = if OUT_U8 { q_plane.add(pix0) } else { std::ptr::null_mut() };
                        for k in 0..3 {
                            let off = 16 * k;
                            let y0 = _mm512_add_ps(
                                _mm512_load_ps(tile512.0[0].as_ptr().add(off)),
                                _mm512_set1_ps(bias[0]),
                            );
                            let y1 = _mm512_add_ps(
                                _mm512_load_ps(tile512.0[1].as_ptr().add(off)),
                                _mm512_set1_ps(bias[1]),
                            );
                            let y2 = _mm512_add_ps(
                                _mm512_load_ps(tile512.0[2].as_ptr().add(off)),
                                _mm512_set1_ps(bias[2]),
                            );
                            let y3 = _mm512_add_ps(
                                _mm512_load_ps(tile512.0[3].as_ptr().add(off)),
                                _mm512_set1_ps(bias[3]),
                            );
                            finish_col16::<SILU, HAS_RES, OUT_U8>(
                                y0,
                                y1,
                                y2,
                                y3,
                                if HAS_RES { rp.add(off) } else { rp },
                                &rs,
                                &rz,
                                &inv,
                                &zero,
                                if OUT_U8 { q_dst.add(off) } else { q_dst },
                                &mut fv512,
                                off,
                            );
                        }
                        if !OUT_U8 {
                            for (rr, row) in fv512.iter().enumerate() {
                                std::ptr::copy_nonoverlapping(
                                    row.as_ptr(),
                                    f_base.add(rr * ohw + pix0),
                                    NR512,
                                );
                            }
                        }
                        chunk += 2;
                        continue;
                    }
                    let cols = NR.min(t.r_n - col);
                    let contiguous = cols == NR && oy0 < t.oh && ox0 + NR <= t.ow;
                    let pix0 = oy0 * t.ow + ox0;
                    tile24 = [[0.0; NR]; MR];
                    for part in t.parts {
                        kernel_avx512vnni_zmm(
                            part.ks,
                            part.a.as_ptr().add(g * part.ks * MR),
                            t.rows.as_ptr().add(part.row_offset),
                            col,
                            &mut acc24.0,
                        );
                        for r in 0..MR {
                            let vc = _mm256_set1_epi32(part.corr[4 * g + r]);
                            let vs = _mm256_set1_ps(part.scale[4 * g + r]);
                            for k in 0..3 {
                                let off = 8 * k;
                                let av = _mm256_load_si256(acc24.0[r].as_ptr().add(off) as *const __m256i);
                                let cur = _mm256_loadu_ps(tile24[r].as_ptr().add(off));
                                let next =
                                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(av, vc)), vs, cur);
                                _mm256_storeu_ps(tile24[r].as_mut_ptr().add(off), next);
                            }
                        }
                    }
                    let rp = if HAS_RES {
                        if contiguous {
                            res_plane.add(pix0)
                        } else {
                            read_u8_noncontiguous(&mut resw24, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane);
                            resw24.as_ptr()
                        }
                    } else {
                        std::ptr::null()
                    };
                    let q_dst = if OUT_U8 {
                        if contiguous { q_plane.add(pix0) } else { qw24.as_mut_ptr() }
                    } else {
                        std::ptr::null_mut()
                    };
                    for k in 0..3 {
                        let off = 8 * k;
                        let y0 = _mm256_add_ps(
                            _mm256_loadu_ps(tile24[0].as_ptr().add(off)),
                            _mm256_set1_ps(bias[0]),
                        );
                        let y1 = _mm256_add_ps(
                            _mm256_loadu_ps(tile24[1].as_ptr().add(off)),
                            _mm256_set1_ps(bias[1]),
                        );
                        let y2 = _mm256_add_ps(
                            _mm256_loadu_ps(tile24[2].as_ptr().add(off)),
                            _mm256_set1_ps(bias[2]),
                        );
                        let y3 = _mm256_add_ps(
                            _mm256_loadu_ps(tile24[3].as_ptr().add(off)),
                            _mm256_set1_ps(bias[3]),
                        );
                        finish_col8::<SILU, HAS_RES, OUT_U8>(
                            y0,
                            y1,
                            y2,
                            y3,
                            if HAS_RES { rp.add(off) } else { rp },
                            &rs,
                            &rz,
                            &inv,
                            &z01_arr,
                            &z23_arr,
                            if OUT_U8 { q_dst.add(off) } else { q_dst },
                            &mut fv24,
                            off,
                        );
                    }
                    if OUT_U8 {
                        if !contiguous {
                            write_u8_noncontiguous(&qw24, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane);
                        }
                    } else if contiguous {
                        for (rr, row) in fv24.iter().enumerate() {
                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + pix0), NR);
                        }
                    } else {
                        write_f32_noncontiguous(&fv24, cols, oy0, ox0, t.oh, t.ow, t.wq, ohw, f_base);
                    }
                    chunk += 1;
                }
            }
            c0 = c1;
        }
    }
}

/// Execute one decoupled multi-part fused task on 512-bit AVX-512 VNNI.
///
/// # Safety
/// CPU supports `avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni`; task bounds and pointers are valid and disjoint.
pub(in super::super) unsafe fn run_multi_part(t: &MultiPartTask<'_>) {
    let has_res = !t.res_img_ptr.is_null();
    // SAFETY: forwarded from caller.
    unsafe {
        match (t.silu, has_res, t.is_u8) {
            (true, false, true) => run_multi_spec::<true, false, true>(t),
            (false, false, true) => run_multi_spec::<false, false, true>(t),
            (true, true, true) => run_multi_spec::<true, true, true>(t),
            (false, true, true) => run_multi_spec::<false, true, true>(t),
            (true, false, false) => run_multi_spec::<true, false, false>(t),
            (false, false, false) => run_multi_spec::<false, false, false>(t),
            (true, true, false) => run_multi_spec::<true, true, false>(t),
            (false, true, false) => run_multi_spec::<false, true, false>(t),
        }
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_stem9_512<const ZVECS: usize>(
    a: *const i32,
    row_ptrs: &[*const u32; 9],
    col: usize,
    acc: &mut AlignedTile512,
) {
    let z = _mm512_setzero_si512();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: caller guarantees `9 * MR` weights and `col + ZVECS * 16 <= strip_w` per tap strip.
    unsafe {
        for (idx, &rp) in row_ptrs.iter().enumerate() {
            let bp = rp.add(col);
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            let ap = a.add(idx * MR);
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
        }
        store_tile512!(ZVECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn stem_row_spec<const SILU: bool, const OUT_U8: bool>(
    a: &[i32],
    scale: &[f32],
    bias: &[f32],
    og: usize,
    ow: usize,
    ohw: usize,
    img: usize,
    oy: usize,
    row_ptrs: &[*const u32; 9],
    group_out: &[([f32; MR], [i32; MR])],
    qptr: *mut u32,
    fptr: *mut f32,
) {
    let zero_corr = [0i32; MR];
    let zero_rs = [0.0f32; MR];
    let zero_rz = [0i32; MR];
    let mut acc512 = AlignedTile512([[0; NR512]; MR]);
    let mut acc24 = AlignedTile([[0; NR]; MR]);
    let mut qw512 = [0u32; NR512];
    let mut qw24 = [0u32; NR];
    let mut fv512 = [[0f32; NR512]; MR];
    let mut fv24 = [[0f32; NR]; MR];
    // SAFETY: caller validated all buffer bounds and disjoint row ownership; `strip_w = ow.next_multiple_of(24)`
    // guarantees `col + 16 <= strip_w` for `rem >= 9` and `col + 8 <= strip_w` for `rem <= 8` when `col % 16 == 0`.
    unsafe {
        for g in 0..og {
            let panel = a.as_ptr().add(g * 9 * MR);
            let sc: [f32; MR] = scale[4 * g..4 * g + 4].try_into().unwrap();
            let bi: [f32; MR] = bias[4 * g..4 * g + 4].try_into().unwrap();
            let (inv, zero, z01_arr, z23_arr) = if OUT_U8 {
                let (inv, zero) = &group_out[g];
                let (z01, z23) = zero_pairs(zero);
                (*inv, *zero, z01, z23)
            } else {
                ([0.0; MR], [0; MR], [0i16; 16], [0i16; 16])
            };
            let q_row = if OUT_U8 { qptr.add((img * og + g) * ohw + oy * ow) } else { std::ptr::null_mut() };
            let f_row =
                if !OUT_U8 { fptr.add((img * og * 4 + 4 * g) * ohw + oy * ow) } else { std::ptr::null_mut() };
            macro_rules! stem_z {
                ($col:expr, $zvecs:literal, $len:expr, $direct:expr) => {{
                    k_loop_stem9_512::<$zvecs>(panel, row_ptrs, $col, &mut acc512);
                    apply_epi_512::<SILU, false, OUT_U8, $zvecs>(
                        &acc512,
                        &zero_corr,
                        &sc,
                        &bi,
                        std::ptr::null(),
                        &zero_rs,
                        &zero_rz,
                        &inv,
                        &zero,
                        if OUT_U8 {
                            if $direct { q_row.add($col) } else { qw512.as_mut_ptr() }
                        } else {
                            std::ptr::null_mut()
                        },
                        &mut fv512,
                    );
                    if OUT_U8 {
                        if !$direct {
                            std::ptr::copy_nonoverlapping(qw512.as_ptr(), q_row.add($col), $len);
                        }
                    } else {
                        for (rr, row) in fv512.iter().enumerate() {
                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_row.add(rr * ohw + $col), $len);
                        }
                    }
                }};
            }
            let mut col = 0;
            while col + NR512 <= ow {
                stem_z!(col, 3, NR512, true);
                col += NR512;
            }
            let mut rem = ow - col;
            if rem >= 32 {
                stem_z!(col, 2, 32, true);
                col += 32;
                rem -= 32;
            } else if rem >= 16 {
                stem_z!(col, 1, 16, true);
                col += 16;
                rem -= 16;
            }
            if rem > 0 {
                if rem <= 8 {
                    k_loop_bmode_8(BMode::Rows, 9, panel, row_ptrs.as_ptr(), col, &mut acc24);
                    if OUT_U8 {
                        if rem == 8 {
                            apply_epi_aligned::<SILU, false, true, 1>(
                                &acc24,
                                &zero_corr,
                                &sc,
                                &bi,
                                std::ptr::null(),
                                &zero_rs,
                                &zero_rz,
                                &inv,
                                &z01_arr,
                                &z23_arr,
                                q_row.add(col),
                                &mut fv24,
                            );
                        } else {
                            apply_epi_aligned::<SILU, false, true, 1>(
                                &acc24,
                                &zero_corr,
                                &sc,
                                &bi,
                                std::ptr::null(),
                                &zero_rs,
                                &zero_rz,
                                &inv,
                                &z01_arr,
                                &z23_arr,
                                qw24.as_mut_ptr(),
                                &mut fv24,
                            );
                            std::ptr::copy_nonoverlapping(qw24.as_ptr(), q_row.add(col), rem);
                        }
                    } else {
                        apply_epi_aligned::<SILU, false, false, 1>(
                            &acc24,
                            &zero_corr,
                            &sc,
                            &bi,
                            std::ptr::null(),
                            &zero_rs,
                            &zero_rz,
                            &inv,
                            &z01_arr,
                            &z23_arr,
                            std::ptr::null_mut(),
                            &mut fv24,
                        );
                        for (rr, row) in fv24.iter().enumerate() {
                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_row.add(rr * ohw + col), rem);
                        }
                    }
                } else {
                    stem_z!(col, 1, rem, false);
                }
            }
        }
    }
}

/// Execute one `k=3, s=2, p=1` direct `u8` stem output row on 512-bit AVX-512 VNNI.
///
/// # Safety
/// CPU supports `avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni`; buffer bounds and row ownership are valid.
#[allow(clippy::too_many_arguments)]
pub(in super::super) unsafe fn run_stem_row(
    a: &[i32],
    scale: &[f32],
    bias: &[f32],
    og: usize,
    ow: usize,
    ohw: usize,
    img: usize,
    oy: usize,
    silu: bool,
    is_u8: bool,
    row_ptrs: &[*const u32; 9],
    group_out: &[([f32; MR], [i32; MR])],
    qptr: *mut u32,
    fptr: *mut f32,
) {
    // SAFETY: forwarded from caller.
    unsafe {
        match (silu, is_u8) {
            (true, true) => stem_row_spec::<true, true>(
                a, scale, bias, og, ow, ohw, img, oy, row_ptrs, group_out, qptr, fptr,
            ),
            (false, true) => stem_row_spec::<false, true>(
                a, scale, bias, og, ow, ohw, img, oy, row_ptrs, group_out, qptr, fptr,
            ),
            (true, false) => stem_row_spec::<true, false>(
                a, scale, bias, og, ow, ohw, img, oy, row_ptrs, group_out, qptr, fptr,
            ),
            (false, false) => stem_row_spec::<false, false>(
                a, scale, bias, og, ow, ohw, img, oy, row_ptrs, group_out, qptr, fptr,
            ),
        }
    }
}
