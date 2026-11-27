//! Fused x86-64 tile-block drivers for int8 convolutions (`q8::conv` / `q8::conv_per_channel`) and the direct
//! `u8` RGB0 stem (`q8::stem_u8`).
//!
//! Running the entire `(c_tile, g, chunk)` task loop inside a single `#[target_feature]` function eliminates
//! per-tile `vzeroupper` transitions, Windows x64 ABI `xmm6..xmm15` save/restore overhead, indirect calls,
//! per-tile 8-way epilogue dispatch, `r.rows` pointer-table loads on single-part 1×1 convs,
//! and 24-iteration `pix` scatter loops on non-contiguous `3×3` chunks.
//!
//! Each K-loop stores its finished `i32` accumulators into a 32-byte aligned L1 buffer ([`AlignedTile`])
//! with zero YMM live-ins across the K-loop, preventing LLVM from spilling `vpdpbusd` accumulators on long-K
//! `3×3` convs. Arithmetic order is bit-for-bit identical to [`super::kernel::kernel_scalar`] +
//! [`super::kernel::Epi::apply`].

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

pub(super) use super::fused_epi::{
    AlignedTile, AlignedTileF32, BMode, FusedIsa, MAX_INLINE_1X1_PARTS, MultiPartTask, PartSlice,
    SinglePartTask, active_isa, apply_epi_aligned_fstride, apply_epi_f32_aligned, read_u8_noncontiguous,
    write_f32_noncontiguous, write_u8_noncontiguous, zero_pairs,
};
use super::fused_pack::{avx2_pack, avx512vnni_pack, should_pack_1x1, vnni_pack};
use super::kernel::{MR, NR};

/// Generates one ISA-specific 256-bit fused suite (K-loop helpers + single-part + multi-part + stem row).
macro_rules! define_fused_isa {
    (
        mod $mod_name:ident,
        pack_mod = $pack_mod:ident,
        feature = $feat:literal,
        dp_mode = $dp_mode:ident,
        unroll2 = $unroll2:literal
    ) => {
        mod $mod_name {
            use super::*;
            const UNROLL2: bool = $unroll2;

            macro_rules! store_tile {
                ($vecs:expr, $acc:ident,
                 $c00:ident, $c01:ident, $c02:ident,
                 $c10:ident, $c11:ident, $c12:ident,
                 $c20:ident, $c21:ident, $c22:ident,
                 $c30:ident, $c31:ident, $c32:ident) => {
                    let out = $acc.0.as_mut_ptr() as *mut __m256i;
                    _mm256_store_si256(out, $c00);
                    if $vecs >= 2 { _mm256_store_si256(out.add(1), $c01); }
                    if $vecs >= 3 { _mm256_store_si256(out.add(2), $c02); }
                    _mm256_store_si256(out.add(3), $c10);
                    if $vecs >= 2 { _mm256_store_si256(out.add(4), $c11); }
                    if $vecs >= 3 { _mm256_store_si256(out.add(5), $c12); }
                    _mm256_store_si256(out.add(6), $c20);
                    if $vecs >= 2 { _mm256_store_si256(out.add(7), $c21); }
                    if $vecs >= 3 { _mm256_store_si256(out.add(8), $c22); }
                    _mm256_store_si256(out.add(9), $c30);
                    if $vecs >= 2 { _mm256_store_si256(out.add(10), $c31); }
                    if $vecs >= 3 { _mm256_store_si256(out.add(11), $c32); }
                };
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_rows<const VECS: usize>(
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
                macro_rules! step_row {
                    ($p:expr) => {
                        let bp = (*rows.add($p)).add(col) as *const __m256i;
                        let b0 = _mm256_loadu_si256(bp);
                        let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(1)) } else { z };
                        let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(2)) } else { z };
                        let ap = a.add(($p) * MR);
                        define_fused_isa!(@dp_v $dp_mode, VECS, b0, b1, b2, ap,
                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                    };
                }
                // SAFETY: caller guarantees `ks * MR` weights and `col + VECS * 8` words per row.
                unsafe {
                    if UNROLL2 {
                        let mut p = 0;
                        while p + 4 <= ks {
                            step_row!(p);
                            step_row!(p + 1);
                            step_row!(p + 2);
                            step_row!(p + 3);
                            p += 4;
                        }
                        while p < ks {
                            step_row!(p);
                            p += 1;
                        }
                    } else {
                        for p in 0..ks {
                            step_row!(p);
                        }
                    }
                    store_tile!(VECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_1x1<const VECS: usize>(
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
                macro_rules! step_bp {
                    ($bp:expr, $ap:expr) => {
                        let bp = $bp;
                        let ap = $ap;
                        let b0 = _mm256_loadu_si256(bp as *const __m256i);
                        let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(8) as *const __m256i) } else { z };
                        let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(16) as *const __m256i) } else { z };
                        define_fused_isa!(@dp_v $dp_mode, VECS, b0, b1, b2, ap,
                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                    };
                }
                // SAFETY: caller guarantees `ks` planes of stride `hw` with `col + VECS * 8` readable words
                // (within `ks * hw + SLACK`). Guarding with `ks > 0` avoids forming an out-of-bounds pointer
                // on a 0-channel view backed only by `SLACK` words; trailing `wrapping_add` pointers are never
                // dereferenced.
                unsafe {
                    if ks > 0 {
                        let mut bp = base.add(col);
                        let mut ap = a;
                        if UNROLL2 {
                            let hw2 = hw.wrapping_mul(2);
                            let hw3 = hw.wrapping_mul(3);
                            let hw4 = hw.wrapping_mul(4);
                            let mut p = 0;
                            while p + 4 <= ks {
                                step_bp!(bp, ap);
                                step_bp!(bp.add(hw), ap.add(MR));
                                step_bp!(bp.add(hw2), ap.add(2 * MR));
                                step_bp!(bp.add(hw3), ap.add(3 * MR));
                                bp = bp.wrapping_add(hw4);
                                ap = ap.wrapping_add(4 * MR);
                                p += 4;
                            }
                            while p < ks {
                                step_bp!(bp, ap);
                                bp = bp.wrapping_add(hw);
                                ap = ap.wrapping_add(MR);
                                p += 1;
                            }
                        } else {
                            for _ in 0..ks {
                                step_bp!(bp, ap);
                                bp = bp.wrapping_add(hw);
                                ap = ap.wrapping_add(MR);
                            }
                        }
                    }
                    store_tile!(VECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_1x1_parts<const VECS: usize>(
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
                macro_rules! step_bp {
                    ($bp:expr, $ap:expr) => {
                        let bp = $bp;
                        let ap = $ap;
                        let b0 = _mm256_loadu_si256(bp as *const __m256i);
                        let b1 = if VECS >= 2 { _mm256_loadu_si256(bp.add(8) as *const __m256i) } else { z };
                        let b2 = if VECS >= 3 { _mm256_loadu_si256(bp.add(16) as *const __m256i) } else { z };
                        define_fused_isa!(@dp_v $dp_mode, VECS, b0, b1, b2, ap,
                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                    };
                }
                // SAFETY: caller guarantees each part `(base, g_cnt)` has `g_cnt` planes of stride `hw` with
                // `col + VECS * 8` readable words, and `a` holds `sum(g_cnt) * MR` words.
                unsafe {
                    let hw2 = hw.wrapping_mul(2);
                    let hw3 = hw.wrapping_mul(3);
                    let hw4 = hw.wrapping_mul(4);
                    let mut ap = a;
                    for &(base, g_cnt) in parts {
                        if g_cnt == 0 {
                            continue;
                        }
                        let mut bp = base.add(col);
                        if UNROLL2 {
                            let mut p = 0;
                            while p + 4 <= g_cnt {
                                step_bp!(bp, ap);
                                step_bp!(bp.add(hw), ap.add(MR));
                                step_bp!(bp.add(hw2), ap.add(2 * MR));
                                step_bp!(bp.add(hw3), ap.add(3 * MR));
                                bp = bp.wrapping_add(hw4);
                                ap = ap.wrapping_add(4 * MR);
                                p += 4;
                            }
                            while p < g_cnt {
                                step_bp!(bp, ap);
                                bp = bp.wrapping_add(hw);
                                ap = ap.wrapping_add(MR);
                                p += 1;
                            }
                        } else {
                            for _ in 0..g_cnt {
                                step_bp!(bp, ap);
                                bp = bp.wrapping_add(hw);
                                ap = ap.wrapping_add(MR);
                            }
                        }
                    }
                    store_tile!(VECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_bmode<const VECS: usize>(
                b_mode: BMode<'_>,
                ks: usize,
                a: *const i32,
                rows: *const *const u32,
                col: usize,
                acc: &mut AlignedTile,
            ) {
                // SAFETY: forwarded from caller.
                unsafe {
                    match b_mode {
                        BMode::Pointwise1x1 { base, hw } => k_loop_1x1::<VECS>(ks, a, base, hw, col, acc),
                        BMode::Pointwise1x1Parts { parts, hw } => {
                            k_loop_1x1_parts::<VECS>(parts, hw, a, col, acc)
                        }
                        BMode::Rows => k_loop_rows::<VECS>(ks, a, rows, col, acc),
                    }
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
                    store_tile!(VECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                }
            }

            #[target_feature(enable = $feat)]
            unsafe fn run_single_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
                t: &SinglePartTask<'_>,
            ) {
                if should_pack_1x1(t) {
                    // SAFETY: ISA and buffer bounds forwarded from caller; `should_pack_1x1(t)` holds.
                    return unsafe { $pack_mod::run_packed_1x1_spec::<SILU, HAS_RES, OUT_U8>(t) };
                }
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
                let full_end = (t.r_n / NR).min(t.cb_end);
                let flat = t.wq == t.ow;
                // SAFETY: caller validated all slice extents and disjoint task ownership.
                unsafe {
                    while c0 < t.cb_end {
                        let c1 = (c0 + cb_tile).min(t.cb_end);
                        let c1_full = c1.min(full_end);
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
                            if flat {
                                macro_rules! run_full_loop {
                                    ($k_call:expr) => {
                                        for chunk in c0..c1_full {
                                            let col = chunk * NR;
                                            $k_call(col, &mut acc);
                                            let res_ptr = if HAS_RES { res_plane.add(col) } else { std::ptr::null() };
                                            apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, 3>(
                                                &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                                &z01_arr, &z23_arr,
                                                if OUT_U8 { q_plane.add(col) } else { std::ptr::null_mut() },
                                                if !OUT_U8 { f_base.add(col) } else { std::ptr::null_mut() },
                                                ohw,
                                            );
                                        }
                                    };
                                }
                                match t.b_mode {
                                    BMode::Pointwise1x1 { base, hw } => {
                                        run_full_loop!(|col, acc| k_loop_1x1::<3>(ks, a_panel, base, hw, col, acc));
                                    }
                                    BMode::Pointwise1x1Parts { parts, hw } => {
                                        run_full_loop!(|col, acc| k_loop_1x1_parts::<3>(
                                            parts, hw, a_panel, col, acc
                                        ));
                                    }
                                    BMode::Rows => {
                                        run_full_loop!(|col, acc| k_loop_rows::<3>(
                                            ks, a_panel, t.rows.as_ptr(), col, acc
                                        ));
                                    }
                                }
                                for chunk in c1_full..c1 {
                                    let col = chunk * NR;
                                    let cols = NR.min(t.r_n - col);
                                    macro_rules! run_tail {
                                        ($vecs:literal) => {{
                                            k_loop_bmode::<$vecs>(t.b_mode, ks, a_panel, t.rows.as_ptr(), col, &mut acc);
                                            let direct = cols == $vecs * 8;
                                            let res_ptr = if HAS_RES {
                                                if direct {
                                                    res_plane.add(col)
                                                } else {
                                                    resw = [0u32; NR];
                                                    std::ptr::copy_nonoverlapping(
                                                        res_plane.add(col),
                                                        resw.as_mut_ptr(),
                                                        cols,
                                                    );
                                                    resw.as_ptr()
                                                }
                                            } else {
                                                std::ptr::null()
                                            };
                                            let q_dst = if OUT_U8 {
                                                if direct { q_plane.add(col) } else { qw.as_mut_ptr() }
                                            } else {
                                                std::ptr::null_mut()
                                            };
                                            let (f_dst, f_stride) = if !OUT_U8 {
                                                if direct { (f_base.add(col), ohw) } else { (fv.as_mut_ptr() as *mut f32, NR) }
                                            } else {
                                                (std::ptr::null_mut(), 0)
                                            };
                                            apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, $vecs>(
                                                &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                                &z01_arr, &z23_arr, q_dst, f_dst, f_stride,
                                            );
                                            if OUT_U8 {
                                                if !direct {
                                                    std::ptr::copy_nonoverlapping(qw.as_ptr(), q_plane.add(col), cols);
                                                }
                                            } else if !direct {
                                                for (rr, row) in fv.iter().enumerate() {
                                                    std::ptr::copy_nonoverlapping(
                                                        row.as_ptr(),
                                                        f_base.add(rr * ohw + col),
                                                        cols,
                                                    );
                                                }
                                            }
                                        }};
                                    }
                                    if cols <= 8 {
                                        run_tail!(1);
                                    } else if cols <= 16 {
                                        run_tail!(2);
                                    } else {
                                        run_tail!(3);
                                    }
                                }
                            } else {
                                for chunk in c0..c1 {
                                    let col = chunk * NR;
                                    let cols = NR.min(t.r_n - col);
                                    let (oy0, ox0) = (col / t.wq, col % t.wq);
                                    let contiguous = cols == NR && oy0 < t.oh && ox0 + NR <= t.ow;
                                    let pix0 = oy0 * t.ow + ox0;
                                    k_loop_bmode::<3>(t.b_mode, ks, a_panel, t.rows.as_ptr(), col, &mut acc);
                                    let res_ptr = if HAS_RES {
                                        if contiguous {
                                            res_plane.add(pix0)
                                        } else {
                                            read_u8_noncontiguous(&mut resw, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane);
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
                                    let (f_dst, f_stride) = if !OUT_U8 {
                                        if contiguous { (f_base.add(pix0), ohw) } else { (fv.as_mut_ptr() as *mut f32, NR) }
                                    } else {
                                        (std::ptr::null_mut(), 0)
                                    };
                                    apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, 3>(
                                        &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                        &z01_arr, &z23_arr, q_dst, f_dst, f_stride,
                                    );
                                    if OUT_U8 {
                                        if !contiguous {
                                            write_u8_noncontiguous(&qw, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane);
                                        }
                                    } else if !contiguous {
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
                                macro_rules! multi_vecs {
                                    ($vecs:literal) => {{
                                        let zf = _mm256_setzero_ps();
                                        for r in 0..MR {
                                            for k in 0..$vecs {
                                                _mm256_store_ps(tile_f32.0[r].as_mut_ptr().add(8 * k), zf);
                                            }
                                        }
                                        for part in t.parts {
                                            if t.pointwise_hw > 0 {
                                                k_loop_1x1::<$vecs>(
                                                    part.ks,
                                                    part.a.as_ptr().add(g * part.ks * MR),
                                                    part.b_base,
                                                    t.pointwise_hw,
                                                    col,
                                                    &mut acc,
                                                );
                                            } else {
                                                k_loop_rows::<$vecs>(
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
                                                for k in 0..$vecs {
                                                    let a_v = _mm256_load_si256(
                                                        acc.0[r].as_ptr().add(8 * k) as *const __m256i,
                                                    );
                                                    let cur = _mm256_load_ps(tile_f32.0[r].as_ptr().add(8 * k));
                                                    let next = _mm256_fmadd_ps(
                                                        _mm256_cvtepi32_ps(_mm256_sub_epi32(a_v, vc)),
                                                        vs,
                                                        cur,
                                                    );
                                                    _mm256_store_ps(tile_f32.0[r].as_mut_ptr().add(8 * k), next);
                                                }
                                            }
                                        }
                                        let direct_u8 = contiguous || (t.wq == t.ow && cols == $vecs * 8);
                                        let res_ptr = if HAS_RES {
                                            if direct_u8 {
                                                res_plane.add(pix0)
                                            } else {
                                                if t.wq == t.ow {
                                                    resw = [0u32; NR];
                                                    std::ptr::copy_nonoverlapping(
                                                        res_plane.add(col),
                                                        resw.as_mut_ptr(),
                                                        cols,
                                                    );
                                                } else {
                                                    read_u8_noncontiguous(
                                                        &mut resw, cols, oy0, ox0, t.oh, t.ow, t.wq, res_plane,
                                                    );
                                                }
                                                resw.as_ptr()
                                            }
                                        } else {
                                            std::ptr::null()
                                        };
                                        let q_dst = if OUT_U8 {
                                            if direct_u8 { q_plane.add(pix0) } else { qw.as_mut_ptr() }
                                        } else {
                                            std::ptr::null_mut()
                                        };
                                        let (f_dst, f_stride) = if !OUT_U8 {
                                            if direct_u8 {
                                                (f_base.add(pix0), ohw)
                                            } else {
                                                (fv.as_mut_ptr() as *mut f32, NR)
                                            }
                                        } else {
                                            (std::ptr::null_mut(), 0)
                                        };
                                        apply_epi_f32_aligned::<SILU, HAS_RES, OUT_U8, $vecs>(
                                            &tile_f32, &bias, res_ptr, &rs, &rz, &inv, &z01_arr, &z23_arr,
                                            q_dst, f_dst, f_stride,
                                        );
                                        if OUT_U8 {
                                            if !direct_u8 {
                                                if t.wq == t.ow {
                                                    std::ptr::copy_nonoverlapping(
                                                        qw.as_ptr(),
                                                        q_plane.add(col),
                                                        cols,
                                                    );
                                                } else {
                                                    write_u8_noncontiguous(
                                                        &qw, cols, oy0, ox0, t.oh, t.ow, t.wq, q_plane,
                                                    );
                                                }
                                            }
                                        } else if !direct_u8 {
                                            if t.wq == t.ow {
                                                for (rr, row) in fv.iter().enumerate() {
                                                    std::ptr::copy_nonoverlapping(
                                                        row.as_ptr(),
                                                        f_base.add(rr * ohw + col),
                                                        cols,
                                                    );
                                                }
                                            } else {
                                                write_f32_noncontiguous(
                                                    &fv, cols, oy0, ox0, t.oh, t.ow, t.wq, ohw, f_base,
                                                );
                                            }
                                        }
                                    }};
                                }
                                if cols > 16 {
                                    multi_vecs!(3);
                                } else if cols > 8 {
                                    multi_vecs!(2);
                                } else {
                                    multi_vecs!(1);
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
                            apply_epi_aligned_fstride::<SILU, false, OUT_U8, 3>(
                                &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                &inv, &z01_arr, &z23_arr,
                                if OUT_U8 { q_row.add(col) } else { std::ptr::null_mut() },
                                if !OUT_U8 { f_row.add(col) } else { std::ptr::null_mut() },
                                ohw,
                            );
                        }
                        if tail > 0 {
                            let col = full_chunks * NR;
                            macro_rules! stem_tail {
                                ($vecs:literal) => {{
                                    k_loop_stem9::<$vecs>(panel, row_ptrs, col, &mut acc);
                                    let direct = tail == $vecs * 8;
                                    let q_dst = if OUT_U8 {
                                        if direct { q_row.add(col) } else { qw.as_mut_ptr() }
                                    } else {
                                        std::ptr::null_mut()
                                    };
                                    let (f_dst, f_stride) = if !OUT_U8 {
                                        if direct { (f_row.add(col), ohw) } else { (fv.as_mut_ptr() as *mut f32, NR) }
                                    } else {
                                        (std::ptr::null_mut(), 0)
                                    };
                                    apply_epi_aligned_fstride::<SILU, false, OUT_U8, $vecs>(
                                        &acc, &zero_corr, &sc, &bi, std::ptr::null(), &zero_rs, &zero_rz,
                                        &inv, &z01_arr, &z23_arr, q_dst, f_dst, f_stride,
                                    );
                                    if OUT_U8 {
                                        if !direct {
                                            std::ptr::copy_nonoverlapping(qw.as_ptr(), q_row.add(col), tail);
                                        }
                                    } else if !direct {
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

define_fused_isa!(mod vnni_impl, pack_mod = vnni_pack, feature = "avx2,fma,avxvnni", dp_mode = avxvnni, unroll2 = true);
define_fused_isa!(mod avx512vnni_impl, pack_mod = avx512vnni_pack, feature = "avx2,fma,avx512vnni,avx512vl", dp_mode = avx512vnni, unroll2 = true);
define_fused_isa!(mod avx2_impl, pack_mod = avx2_pack, feature = "avx2,fma", dp_mode = avx2, unroll2 = false);

/// Execute one single-part fused task on the active x86-64 ISA.
///
/// # Safety
/// `isa` must match [`active_isa()`]; all buffer pointers and task bounds in `t` must be valid and disjoint.
pub(super) unsafe fn run_single_part(isa: FusedIsa, t: &SinglePartTask<'_>) {
    // SAFETY: forwarded from caller.
    unsafe {
        match isa {
            FusedIsa::Avx512Vnni512 => super::vnni512::run_single_part(t),
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
            FusedIsa::Avx512Vnni512 => super::vnni512::run_multi_part(t),
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
            FusedIsa::Avx512Vnni512 => super::vnni512::run_stem_row(
                a, scale, bias, og, ow, ohw, img, oy, silu, is_u8, row_ptrs, group_out, qptr, fptr,
            ),
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
