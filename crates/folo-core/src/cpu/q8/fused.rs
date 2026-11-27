//! Fused x86-64 tile-block drivers for int8 convolutions (`q8::conv` / `q8::conv_per_channel`) and the direct
//! `u8` RGB0 stem (`q8::stem_u8`).
//!
//! Running the entire `(c_tile, g, chunk)` task loop inside a single `#[target_feature]` function eliminates
//! per-tile `vzeroupper` transitions, Windows x64 ABI `xmm6..xmm15` save/restore overhead, indirect calls,
//! per-tile 8-way epilogue dispatch, `r.rows` pointer-table loads on single-part 1×1 convs,
//! and 24-iteration `pix` scatter loops on non-contiguous `3×3` chunks.
//!
//! Each K-loop stores its 12 finished `i32` accumulators into a 32-byte aligned L1 buffer ([`AlignedTile`])
//! with zero YMM live-ins across the K-loop, preventing LLVM from spilling `vpdpbusd` accumulators on long-K
//! `3×3` convs. Arithmetic order is bit-for-bit identical to [`super::kernel::kernel_scalar`] +
//! [`super::kernel::Epi::apply`].

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

pub(super) use super::fused_epi::{
    AlignedTile, AlignedTileF32, BMode, FusedIsa, MAX_INLINE_1X1_PARTS, MultiPartTask, PartSlice,
    SinglePartTask, active_isa, apply_epi_aligned, finish_col8, read_u8_noncontiguous,
    write_f32_noncontiguous, write_u8_noncontiguous, zero_pairs,
};
use super::kernel::{MR, NR};

/// Generates one ISA-specific fused suite (K-loop helpers + single-part task + multi-part task + stem row).
macro_rules! define_fused_isa {
    (
        mod $mod_name:ident,
        feature = $feat:literal,
        dp_mode = $dp_mode:ident
    ) => {
        mod $mod_name {
            use super::*;

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_rows(
                ks: usize,
                a: *const i32,
                rows: *const *const u32,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `ks * MR` weights and `col + NR` words per row.
                unsafe {
                    for p in 0..ks {
                        let bp = (*rows.add(p)).add(col) as *const __m256i;
                        let b0 = _mm256_loadu_si256(bp);
                        let b1 = _mm256_loadu_si256(bp.add(1));
                        let b2 = _mm256_loadu_si256(bp.add(2));
                        let ap = a.add(p * MR);
                        define_fused_isa!(@dp $dp_mode, b0, b1, b2, ap,
                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                    }
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    _mm256_store_si256(out.add(1), c01);
                    _mm256_store_si256(out.add(2), c02);
                    _mm256_store_si256(out.add(3), c10);
                    _mm256_store_si256(out.add(4), c11);
                    _mm256_store_si256(out.add(5), c12);
                    _mm256_store_si256(out.add(6), c20);
                    _mm256_store_si256(out.add(7), c21);
                    _mm256_store_si256(out.add(8), c22);
                    _mm256_store_si256(out.add(9), c30);
                    _mm256_store_si256(out.add(10), c31);
                    _mm256_store_si256(out.add(11), c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_1x1(
                ks: usize,
                a: *const i32,
                base: *const u32,
                hw: usize,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `ks` planes of stride `hw` with `col + NR` readable words in each
                // plane (within `ks * hw + SLACK`). Pointer stays strictly within `0..ks * hw + col + NR`.
                unsafe {
                    if ks > 0 {
                        let mut bp = base.add(col);
                        let mut ap = a;
                        for p in 0..ks {
                            let b0 = _mm256_loadu_si256(bp as *const __m256i);
                            let b1 = _mm256_loadu_si256(bp.add(8) as *const __m256i);
                            let b2 = _mm256_loadu_si256(bp.add(16) as *const __m256i);
                            define_fused_isa!(@dp $dp_mode, b0, b1, b2, ap,
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            if p + 1 < ks {
                                bp = bp.add(hw);
                                ap = ap.add(MR);
                            }
                        }
                    }
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    _mm256_store_si256(out.add(1), c01);
                    _mm256_store_si256(out.add(2), c02);
                    _mm256_store_si256(out.add(3), c10);
                    _mm256_store_si256(out.add(4), c11);
                    _mm256_store_si256(out.add(5), c12);
                    _mm256_store_si256(out.add(6), c20);
                    _mm256_store_si256(out.add(7), c21);
                    _mm256_store_si256(out.add(8), c22);
                    _mm256_store_si256(out.add(9), c30);
                    _mm256_store_si256(out.add(10), c31);
                    _mm256_store_si256(out.add(11), c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_1x1_parts(
                parts: &[(*const u32, usize)],
                hw: usize,
                a: *const i32,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees each part `(base, g_cnt)` has `g_cnt` planes of stride `hw` with
                // `col + NR` readable words, and `a` holds `sum(g_cnt) * MR` words.
                unsafe {
                    let mut ap = a;
                    for &(base, g_cnt) in parts {
                        if g_cnt == 0 {
                            continue;
                        }
                        let mut bp = base.add(col);
                        for p in 0..g_cnt {
                            let b0 = _mm256_loadu_si256(bp as *const __m256i);
                            let b1 = _mm256_loadu_si256(bp.add(8) as *const __m256i);
                            let b2 = _mm256_loadu_si256(bp.add(16) as *const __m256i);
                            define_fused_isa!(@dp $dp_mode, b0, b1, b2, ap,
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            ap = ap.add(MR);
                            if p + 1 < g_cnt {
                                bp = bp.add(hw);
                            }
                        }
                    }
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    _mm256_store_si256(out.add(1), c01);
                    _mm256_store_si256(out.add(2), c02);
                    _mm256_store_si256(out.add(3), c10);
                    _mm256_store_si256(out.add(4), c11);
                    _mm256_store_si256(out.add(5), c12);
                    _mm256_store_si256(out.add(6), c20);
                    _mm256_store_si256(out.add(7), c21);
                    _mm256_store_si256(out.add(8), c22);
                    _mm256_store_si256(out.add(9), c30);
                    _mm256_store_si256(out.add(10), c31);
                    _mm256_store_si256(out.add(11), c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_stem9<const VECS: usize>(
                a: *const i32,
                row_ptrs: &[*const u32; 9],
                col: usize,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `9 * MR` weights and `col + VECS * 8` readable words per tap strip.
                unsafe {
                    macro_rules! tap {
                        ($idx:expr) => {
                            let bp = row_ptrs[$idx].add(col) as *const __m256i;
                            let b0 = _mm256_loadu_si256(bp);
                            let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(1)) } else { z };
                            let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(2)) } else { z };
                            let ap = a.add(($idx) * MR);
                            define_fused_isa!(@dp_v $dp_mode, VECS, b0, b1, b2, ap,
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                        };
                    }
                    tap!(0);
                    tap!(1);
                    tap!(2);
                    tap!(3);
                    tap!(4);
                    tap!(5);
                    tap!(6);
                    tap!(7);
                    tap!(8);
                    let out = acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, c00);
                    if VECS >= 2 { _mm256_store_si256(out.add(1), c01); }
                    if VECS >= 3 { _mm256_store_si256(out.add(2), c02); }
                    _mm256_store_si256(out.add(3), c10);
                    if VECS >= 2 { _mm256_store_si256(out.add(4), c11); }
                    if VECS >= 3 { _mm256_store_si256(out.add(5), c12); }
                    _mm256_store_si256(out.add(6), c20);
                    if VECS >= 2 { _mm256_store_si256(out.add(7), c21); }
                    if VECS >= 3 { _mm256_store_si256(out.add(8), c22); }
                    _mm256_store_si256(out.add(9), c30);
                    if VECS >= 2 { _mm256_store_si256(out.add(10), c31); }
                    if VECS >= 3 { _mm256_store_si256(out.add(11), c32); }
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn run_single_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
                t: &SinglePartTask<'_>,
            ) {
                let ohw = t.oh * t.ow;
                let ks = t.ks;
                // L1-resident chunk tile: keeps `cb_tile * ks * NR * 4 <= 24 KiB` so B stays hot in L1 across `g`,
                // while hoisting channel metadata and writing sequentially along each output plane `g`.
                let cb_tile = (24 * 1024 / (ks.max(1) * NR * 4)).clamp(1, 16);
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut resw = [0u32; NR];
                let mut fv = [[0f32; NR]; MR];
                let mut c0 = t.cb_start;
                // SAFETY: caller validated all slice extents and disjoint task ownership.
                unsafe {
                    while c0 < t.cb_end {
                        let c1 = (c0 + cb_tile).min(t.cb_end);
                        for g in t.ob_start..t.ob_end {
                            let a_panel = t.a.as_ptr().add(g * ks * MR);
                            let corr: [i32; MR] = t.corr[4 * g..4 * g + 4].try_into().unwrap();
                            let scale: [f32; MR] = t.scale[4 * g..4 * g + 4].try_into().unwrap();
                            let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                            let (inv, z01_arr, z23_arr) = if OUT_U8 {
                                let (inv, zero) = &t.group_out[g];
                                let (z01, z23) = zero_pairs(zero);
                                (*inv, z01, z23)
                            } else {
                                ([0.0; MR], [0i16; 16], [0i16; 16])
                            };
                            let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                            let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                            let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                            let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                            for chunk in c0..c1 {
                                let col = chunk * NR;
                                let cols = NR.min(t.r_n - col);
                                let (oy0, ox0) = if t.wq == t.ow { (0, col) } else { (col / t.wq, col % t.wq) };
                                let contiguous = cols == NR && (t.wq == t.ow || (oy0 < t.oh && ox0 + NR <= t.ow));
                                let pix0 = if t.wq == t.ow { col } else { oy0 * t.ow + ox0 };
                                match t.b_mode {
                                    BMode::Pointwise1x1 { base, hw } => {
                                        k_loop_1x1(ks, a_panel, base, hw, col, &mut acc);
                                    }
                                    BMode::Pointwise1x1Parts { parts, hw } => {
                                        k_loop_1x1_parts(parts, hw, a_panel, col, &mut acc);
                                    }
                                    BMode::Rows => {
                                        k_loop_rows(ks, a_panel, t.rows.as_ptr(), col, &mut acc);
                                    }
                                }
                                let res_ptr = if HAS_RES {
                                    if contiguous {
                                        res_plane.add(pix0)
                                    } else {
                                        if t.wq == t.ow {
                                            resw = [0u32; NR];
                                            std::ptr::copy_nonoverlapping(res_plane.add(col), resw.as_mut_ptr(), cols);
                                        } else {
                                            read_u8_noncontiguous(&mut resw, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane);
                                        }
                                        resw.as_ptr()
                                    }
                                } else {
                                    std::ptr::null()
                                };
                                if OUT_U8 {
                                    if contiguous {
                                        apply_epi_aligned::<SILU, HAS_RES, true, 3>(
                                            &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                            &z01_arr, &z23_arr, q_plane.add(pix0), &mut fv,
                                        );
                                    } else {
                                        apply_epi_aligned::<SILU, HAS_RES, true, 3>(
                                            &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                            &z01_arr, &z23_arr, qw.as_mut_ptr(), &mut fv,
                                        );
                                        if t.wq == t.ow {
                                            std::ptr::copy_nonoverlapping(qw.as_ptr(), q_plane.add(col), cols);
                                        } else {
                                            write_u8_noncontiguous(&qw, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane);
                                        }
                                    }
                                } else {
                                    apply_epi_aligned::<SILU, HAS_RES, false, 3>(
                                        &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                        &z01_arr, &z23_arr, std::ptr::null_mut(), &mut fv,
                                    );
                                    if contiguous {
                                        for (rr, row) in fv.iter().enumerate() {
                                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + pix0), NR);
                                        }
                                    } else if t.wq == t.ow {
                                        for (rr, row) in fv.iter().enumerate() {
                                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + col), cols);
                                        }
                                    } else {
                                        write_f32_noncontiguous(&fv, cols, oy0, ox0, t.oh, t.ow, t.wq, ohw, f_base);
                                    }
                                }
                            }
                        }
                        c0 = c1;
                    }
                }
            }

            pub(super) unsafe fn run_single(t: &SinglePartTask<'_>) {
                let has_res = !t.res_img_ptr.is_null();
                // SAFETY: ISA checked by caller.
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

            #[target_feature(enable = $feat)]
            unsafe fn run_multi_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
                t: &MultiPartTask<'_>,
            ) {
                let ohw = t.oh * t.ow;
                let total_ks: usize = t.parts.iter().map(|p| p.ks).sum();
                let cb_tile = (24 * 1024 / (total_ks.max(1) * NR * 4)).clamp(1, 16);
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut tile_f32 = AlignedTileF32([[0.0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut resw = [0u32; NR];
                let mut fv = [[0f32; NR]; MR];
                let mut c0 = t.cb_start;
                // SAFETY: caller validated all slice extents and disjoint task ownership.
                unsafe {
                    while c0 < t.cb_end {
                        let c1 = (c0 + cb_tile).min(t.cb_end);
                        for g in t.ob_start..t.ob_end {
                            let bias: [f32; MR] = t.bias[4 * g..4 * g + 4].try_into().unwrap();
                            let (inv, z01_arr, z23_arr) = if OUT_U8 {
                                let (inv, zero) = &t.group_out[g];
                                let (z01, z23) = zero_pairs(zero);
                                (*inv, z01, z23)
                            } else {
                                ([0.0; MR], [0i16; 16], [0i16; 16])
                            };
                            let (rs, rz) = if HAS_RES { t.group_res[g] } else { ([0.0; MR], [0; MR]) };
                            let res_plane = if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                            let q_plane = if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                            let f_base = if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                            for chunk in c0..c1 {
                                let col = chunk * NR;
                                let cols = NR.min(t.r_n - col);
                                let (oy0, ox0) = if t.wq == t.ow { (0, col) } else { (col / t.wq, col % t.wq) };
                                let contiguous = cols == NR && (t.wq == t.ow || (oy0 < t.oh && ox0 + NR <= t.ow));
                                let pix0 = if t.wq == t.ow { col } else { oy0 * t.ow + ox0 };
                                let zf = _mm256_setzero_ps();
                                let tfp = tile_f32.0.as_mut_ptr() as *mut f32;
                                for i in 0..12 {
                                    _mm256_store_ps(tfp.add(i * 8), zf);
                                }
                                for part in t.parts {
                                    if t.pointwise_hw > 0 {
                                        k_loop_1x1(
                                            part.ks,
                                            part.a.as_ptr().add(g * part.ks * MR),
                                            part.b_base,
                                            t.pointwise_hw,
                                            col,
                                            &mut acc,
                                        );
                                    } else {
                                        k_loop_rows(
                                            part.ks,
                                            part.a.as_ptr().add(g * part.ks * MR),
                                            t.rows.as_ptr().add(part.row_offset),
                                            col,
                                            &mut acc,
                                        );
                                    }
                                    for r in 0..MR {
                                        let vc = _mm256_set1_epi32(part.corr[4 * g + r]);
                                        let vs = _mm256_set1_ps(part.scale[4 * g + r]);
                                        for k in 0..3 {
                                            let a_v = _mm256_load_si256(acc.0[r].as_ptr().add(8 * k) as *const __m256i);
                                            let cur = _mm256_load_ps(tile_f32.0[r].as_ptr().add(8 * k));
                                            let next = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a_v, vc)), vs, cur);
                                            _mm256_store_ps(tile_f32.0[r].as_mut_ptr().add(8 * k), next);
                                        }
                                    }
                                }
                                let res_ptr = if HAS_RES {
                                    if contiguous {
                                        res_plane.add(pix0)
                                    } else {
                                        if t.wq == t.ow {
                                            resw = [0u32; NR];
                                            std::ptr::copy_nonoverlapping(res_plane.add(col), resw.as_mut_ptr(), cols);
                                        } else {
                                            read_u8_noncontiguous(&mut resw, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane);
                                        }
                                        resw.as_ptr()
                                    }
                                } else {
                                    std::ptr::null()
                                };
                                let q_dst = if OUT_U8 {
                                    if contiguous { q_plane.add(pix0) } else { qw.as_mut_ptr() }
                                } else {
                                    std::ptr::null_mut()
                                };
                                for k in 0..3 {
                                    let off = 8 * k;
                                    let y0 = _mm256_add_ps(_mm256_load_ps(tile_f32.0[0].as_ptr().add(off)), _mm256_set1_ps(bias[0]));
                                    let y1 = _mm256_add_ps(_mm256_load_ps(tile_f32.0[1].as_ptr().add(off)), _mm256_set1_ps(bias[1]));
                                    let y2 = _mm256_add_ps(_mm256_load_ps(tile_f32.0[2].as_ptr().add(off)), _mm256_set1_ps(bias[2]));
                                    let y3 = _mm256_add_ps(_mm256_load_ps(tile_f32.0[3].as_ptr().add(off)), _mm256_set1_ps(bias[3]));
                                    finish_col8::<SILU, HAS_RES, OUT_U8>(
                                        y0, y1, y2, y3,
                                        if HAS_RES { res_ptr.add(off) } else { res_ptr },
                                        &rs, &rz, &inv, &z01_arr, &z23_arr,
                                        if OUT_U8 { q_dst.add(off) } else { q_dst },
                                        &mut fv, off,
                                    );
                                }
                                if OUT_U8 {
                                    if !contiguous {
                                        if t.wq == t.ow {
                                            std::ptr::copy_nonoverlapping(qw.as_ptr(), q_plane.add(col), cols);
                                        } else {
                                            write_u8_noncontiguous(&qw, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane);
                                        }
                                    }
                                } else if contiguous {
                                    for (rr, row) in fv.iter().enumerate() {
                                        std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + pix0), NR);
                                    }
                                } else if t.wq == t.ow {
                                    for (rr, row) in fv.iter().enumerate() {
                                        std::ptr::copy_nonoverlapping(row.as_ptr(), f_base.add(rr * ohw + col), cols);
                                    }
                                } else {
                                    write_f32_noncontiguous(&fv, cols, oy0, ox0, t.oh, t.ow, t.wq, ohw, f_base);
                                }
                            }
                        }
                        c0 = c1;
                    }
                }
            }

            pub(super) unsafe fn run_multi(t: &MultiPartTask<'_>) {
                let has_res = !t.res_img_ptr.is_null();
                // SAFETY: ISA checked by caller.
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

            #[target_feature(enable = $feat)]
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
                let full_chunks = ow / NR;
                let tail = ow % NR;
                let zero_corr = [0i32; MR];
                let zero_rs = [0.0f32; MR];
                let zero_rz = [0i32; MR];
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut fv = [[0f32; NR]; MR];
                // SAFETY: caller validated all buffer bounds and disjoint row ownership.
                unsafe {
                    for g in 0..og {
                        let panel = a.as_ptr().add(g * 9 * MR);
                        let sc: [f32; MR] = scale[4 * g..4 * g + 4].try_into().unwrap();
                        let bi: [f32; MR] = bias[4 * g..4 * g + 4].try_into().unwrap();
                        let (inv, z01_arr, z23_arr) = if OUT_U8 {
                            let (inv, zero) = &group_out[g];
                            let (z01, z23) = zero_pairs(zero);
                            (*inv, z01, z23)
                        } else {
                            ([0.0; MR], [0i16; 16], [0i16; 16])
                        };
                        let q_row = if OUT_U8 {
                            qptr.add((img * og + g) * ohw + oy * ow)
                        } else {
                            std::ptr::null_mut()
                        };
                        let f_row = if !OUT_U8 {
                            fptr.add((img * og * 4 + 4 * g) * ohw + oy * ow)
                        } else {
                            std::ptr::null_mut()
                        };
                        for chunk in 0..full_chunks {
                            let col = chunk * NR;
                            k_loop_stem9::<3>(panel, row_ptrs, col, &mut acc);
                            if OUT_U8 {
                                apply_epi_aligned::<SILU, false, true, 3>(
                                    &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                    &inv, &z01_arr, &z23_arr, q_row.add(col), &mut fv,
                                );
                            } else {
                                apply_epi_aligned::<SILU, false, false, 3>(
                                    &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                    &inv, &z01_arr, &z23_arr, std::ptr::null_mut(), &mut fv,
                                );
                                for (rr, row) in fv.iter().enumerate() {
                                    std::ptr::copy_nonoverlapping(row.as_ptr(), f_row.add(rr * ohw + col), NR);
                                }
                            }
                        }
                        if tail > 0 {
                            let col = full_chunks * NR;
                            macro_rules! stem_tail {
                                ($vecs:literal) => {{
                                    k_loop_stem9::<$vecs>(panel, row_ptrs, col, &mut acc);
                                    if OUT_U8 {
                                        if tail == $vecs * 8 {
                                            apply_epi_aligned::<SILU, false, true, $vecs>(
                                                &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                                &inv, &z01_arr, &z23_arr, q_row.add(col), &mut fv,
                                            );
                                        } else {
                                            apply_epi_aligned::<SILU, false, true, $vecs>(
                                                &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                                &inv, &z01_arr, &z23_arr, qw.as_mut_ptr(), &mut fv,
                                            );
                                            std::ptr::copy_nonoverlapping(qw.as_ptr(), q_row.add(col), tail);
                                        }
                                    } else {
                                        apply_epi_aligned::<SILU, false, false, $vecs>(
                                            &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                            &inv, &z01_arr, &z23_arr, std::ptr::null_mut(), &mut fv,
                                        );
                                        for (rr, row) in fv.iter().enumerate() {
                                            std::ptr::copy_nonoverlapping(row.as_ptr(), f_row.add(rr * ohw + col), tail);
                                        }
                                    }
                                }};
                            }
                            if tail <= 8 {
                                stem_tail!(1);
                            } else if tail <= 16 {
                                stem_tail!(2);
                            } else {
                                stem_tail!(3);
                            }
                        }
                    }
                }
            }

            #[allow(clippy::too_many_arguments)]
            pub(super) unsafe fn run_stem_row(
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
                // SAFETY: ISA checked by caller.
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
        }
    };

    (@dp $mode:ident, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        define_fused_isa!(@dp_v $mode, 3, $b0, $b1, $b2, $ap,
            $c00, $c01, $c02, $c10, $c11, $c12, $c20, $c21, $c22, $c30, $c31, $c32);
    };

    (@dp_v avxvnni, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_avx_epi32($c00, $b0, w0);
        if $vecs >= 2 { $c01 = _mm256_dpbusd_avx_epi32($c01, $b1, w0); }
        if $vecs >= 3 { $c02 = _mm256_dpbusd_avx_epi32($c02, $b2, w0); }
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_avx_epi32($c10, $b0, w1);
        if $vecs >= 2 { $c11 = _mm256_dpbusd_avx_epi32($c11, $b1, w1); }
        if $vecs >= 3 { $c12 = _mm256_dpbusd_avx_epi32($c12, $b2, w1); }
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_avx_epi32($c20, $b0, w2);
        if $vecs >= 2 { $c21 = _mm256_dpbusd_avx_epi32($c21, $b1, w2); }
        if $vecs >= 3 { $c22 = _mm256_dpbusd_avx_epi32($c22, $b2, w2); }
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_avx_epi32($c30, $b0, w3);
        if $vecs >= 2 { $c31 = _mm256_dpbusd_avx_epi32($c31, $b1, w3); }
        if $vecs >= 3 { $c32 = _mm256_dpbusd_avx_epi32($c32, $b2, w3); }
    };

    (@dp_v avx512vnni, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_epi32($c00, $b0, w0);
        if $vecs >= 2 { $c01 = _mm256_dpbusd_epi32($c01, $b1, w0); }
        if $vecs >= 3 { $c02 = _mm256_dpbusd_epi32($c02, $b2, w0); }
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_epi32($c10, $b0, w1);
        if $vecs >= 2 { $c11 = _mm256_dpbusd_epi32($c11, $b1, w1); }
        if $vecs >= 3 { $c12 = _mm256_dpbusd_epi32($c12, $b2, w1); }
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_epi32($c20, $b0, w2);
        if $vecs >= 2 { $c21 = _mm256_dpbusd_epi32($c21, $b1, w2); }
        if $vecs >= 3 { $c22 = _mm256_dpbusd_epi32($c22, $b2, w2); }
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_epi32($c30, $b0, w3);
        if $vecs >= 2 { $c31 = _mm256_dpbusd_epi32($c31, $b1, w3); }
        if $vecs >= 3 { $c32 = _mm256_dpbusd_epi32($c32, $b2, w3); }
    };

    (@dp_v avx2, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
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
        if $vecs >= 2 { col_step!($b1, $c01, $c11, $c21, $c31); }
        if $vecs >= 3 { col_step!($b2, $c02, $c12, $c22, $c32); }
    }};
}

#[inline(always)]
fn split_w_avx2(w: i32) -> (i32, i32) {
    let b = w.to_le_bytes();
    let ev = (b[0] as i8 as i16 as u16 as u32) | ((b[2] as i8 as i16 as u16 as u32) << 16);
    let od = (b[1] as i8 as i16 as u16 as u32) | ((b[3] as i8 as i16 as u16 as u32) << 16);
    (ev as i32, od as i32)
}

define_fused_isa!(mod vnni_impl, feature = "avx2,fma,avxvnni", dp_mode = avxvnni);
define_fused_isa!(mod avx512vnni_impl, feature = "avx2,fma,avx512vnni,avx512vl", dp_mode = avx512vnni);
define_fused_isa!(mod avx2_impl, feature = "avx2,fma", dp_mode = avx2);

/// Execute one single-part fused task on the active x86-64 ISA.
///
/// # Safety
/// `isa` must match [`active_isa()`]; all buffer pointers and task bounds in `t` must be valid and disjoint.
pub(super) unsafe fn run_single_part(isa: FusedIsa, t: &SinglePartTask<'_>) {
    // SAFETY: forwarded from caller.
    unsafe {
        match isa {
            FusedIsa::AvxVnni => vnni_impl::run_single(t),
            FusedIsa::Avx512Vnni => avx512vnni_impl::run_single(t),
            FusedIsa::Avx2 => avx2_impl::run_single(t),
        }
    }
}

/// Execute one decoupled multi-part fused task on the active x86-64 ISA.
///
/// # Safety
/// `isa` must match [`active_isa()`]; all buffer pointers and task bounds in `t` must be valid and disjoint.
pub(super) unsafe fn run_multi_part(isa: FusedIsa, t: &MultiPartTask<'_>) {
    // SAFETY: forwarded from caller.
    unsafe {
        match isa {
            FusedIsa::AvxVnni => vnni_impl::run_multi(t),
            FusedIsa::Avx512Vnni => avx512vnni_impl::run_multi(t),
            FusedIsa::Avx2 => avx2_impl::run_multi(t),
        }
    }
}

/// Execute one `k=3, s=2, p=1` direct `u8` stem output row across all output channel groups and column chunks.
///
/// # Safety
/// `isa` must match [`active_isa()`]; all buffer pointers and row bounds must be valid and disjoint.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_stem_row(
    isa: FusedIsa,
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
        match isa {
            FusedIsa::AvxVnni => vnni_impl::run_stem_row(
                a, scale, bias, og, ow, ohw, img, oy, silu, is_u8, row_ptrs, group_out, qptr, fptr,
            ),
            FusedIsa::Avx512Vnni => avx512vnni_impl::run_stem_row(
                a, scale, bias, og, ow, ohw, img, oy, silu, is_u8, row_ptrs, group_out, qptr, fptr,
            ),
            FusedIsa::Avx2 => avx2_impl::run_stem_row(
                a, scale, bias, og, ow, ohw, img, oy, silu, is_u8, row_ptrs, group_out, qptr, fptr,
            ),
        }
    }
}
