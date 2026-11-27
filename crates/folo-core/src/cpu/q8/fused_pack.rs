//! L1-resident contiguous B-panel packing and 4× K-unrolled aligned 256-bit microkernels for fused x86-64
//! `1×1` pointwise int8 convolutions (`AVX-VNNI`, `AVX-512 VNNI VL`, and `AVX2`).
//!
//! Planar C4 feature maps at power-of-two / 4 KiB-aligned spatial sizes (notably `160×160` = 100 KiB = `25 × 4 KiB`
//! and `80×80` = 25.6 KiB, or deep-channel `40×40` / `20×20` maps spanning > 64 4 KiB pages) suffer L1D
//! set-associativity conflicts and dTLB thrashing when `k_loop_1x1` strides `bp += hw` across every output
//! channel group `g`. Packing up to `24 KiB` of contiguous 32-byte-aligned `[chunk][ks][NR]` panels on the
//! stack once per chunk block eliminates 4K aliasing, reduces TLB pressure to 1–6 pages, enables 32-byte
//! aligned `_mm256_load_si256` with compile-time constant offsets, and 4× K-unrolls `vpdpbusd` while staying
//! bit-for-bit identical to [`super::kernel::kernel_scalar`].

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::fused_epi::{
    AlignedTile, AlignedTileF32, BMode, MultiPartTask, PartSlice, SinglePartTask, apply_epi_aligned_fstride,
    apply_epi_f32_aligned, zero_pairs,
};
use super::kernel::{MR, NR};

/// Stack-allocated packed B buffer capacity in `u32` words (`24 KiB`, exact multiple of `NR = 24`).
pub(super) const PACK_WORDS: usize = 6144;
/// Maximum `ks` (`c_in / 4`) eligible for stack B-panel packing (`256` groups = `1024` channels).
pub(super) const PACK_MAX_KS: usize = PACK_WORDS / NR;
/// Minimum output channel groups (`og = c_out / 4`) in a task to amortize B-panel packing.
pub(super) const PACK_MIN_OG: usize = 4;

/// 32-byte aligned L1-resident stack buffer for packed `[chunk][ks][NR]` `u32` C4 words.
/// Stored as `MaybeUninit` so each task avoids a 24 KiB stack `memset` prologue on every call.
#[repr(align(32))]
struct PackedB(std::mem::MaybeUninit<[u32; PACK_WORDS]>);

impl PackedB {
    #[inline(always)]
    fn uninit() -> Self {
        Self(std::mem::MaybeUninit::uninit())
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u32 {
        self.0.as_ptr() as *const u32
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u32 {
        self.0.as_mut_ptr() as *mut u32
    }
}

/// Returns true when `t` benefits from L1-resident B-panel packing on 256-bit x86-64 ISAs:
/// requires enough output channel groups (`>= PACK_MIN_OG`), `ks` in `8..=PACK_MAX_KS`, and either large
/// planes (`hw >= 1024`, where strided reads hit 4K L1D set conflicts / dTLB misses) or deep reductions
/// (`ks >= 48`, spanning > 48 planes/pages).
#[inline]
pub(super) fn should_pack_1x1(t: &SinglePartTask<'_>) -> bool {
    let hw = match t.b_mode {
        BMode::Pointwise1x1 { hw, .. } | BMode::Pointwise1x1Parts { hw, .. } => hw,
        BMode::Rows => return false,
    };
    t.wq == t.ow
        && t.ks >= 8
        && t.ks <= PACK_MAX_KS
        && t.ob_end.saturating_sub(t.ob_start) >= PACK_MIN_OG
        && (hw >= 1024 || t.ks >= 48)
}

/// Returns true when a decoupled multi-part `1×1` conv benefits from packing all parts' B panels once per
/// chunk block into L1D (`PackedB`), eliminating 4K aliasing and strided dTLB misses across output groups.
#[inline]
pub(super) fn should_pack_1x1_multi(t: &MultiPartTask<'_>) -> bool {
    if t.pointwise_hw == 0 || t.wq != t.ow || t.parts.is_empty() {
        return false;
    }
    if t.ob_end.saturating_sub(t.ob_start) < PACK_MIN_OG {
        return false;
    }
    let total_ks: usize = t.parts.iter().map(|p| p.ks).sum();
    (8..=PACK_MAX_KS).contains(&total_ks) && (t.pointwise_hw >= 1024 || total_ks >= 48)
}

/// Pack `count` full `NR`-word chunks starting at `col0` from `b_mode` into `dst[0..count * ks * NR]`
/// with layout `dst[(i * ks + p) * NR .. +NR]` (every row 32-byte aligned).
///
/// # Safety
/// Caller guarantees AVX2, `dst` is 32-byte aligned with at least `count * ks * NR` writable words, and
/// `b_mode` has `ks` planes with `col0 + count * NR` readable words per plane.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_chunks_24(b_mode: BMode<'_>, ks: usize, col0: usize, count: usize, dst: *mut u32) {
    // SAFETY: pointer bounds and alignment guaranteed by caller.
    unsafe {
        match b_mode {
            BMode::Pointwise1x1 { base, hw } => {
                let mut src_p = base.add(col0);
                for p in 0..ks {
                    let mut sp = src_p;
                    let mut dp = dst.add(p * NR);
                    let stride = ks * NR;
                    for _ in 0..count {
                        let v0 = _mm256_loadu_si256(sp as *const __m256i);
                        let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                        let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                        _mm256_store_si256(dp as *mut __m256i, v0);
                        _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                        _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                        sp = sp.add(NR);
                        dp = dp.wrapping_add(stride);
                    }
                    src_p = src_p.wrapping_add(hw);
                }
            }
            BMode::Pointwise1x1Parts { parts, hw } => {
                let mut p_global = 0usize;
                let stride = ks * NR;
                for &(base, g_cnt) in parts {
                    if g_cnt == 0 {
                        continue;
                    }
                    let mut src_p = base.add(col0);
                    for _ in 0..g_cnt {
                        let mut sp = src_p;
                        let mut dp = dst.add(p_global * NR);
                        for _ in 0..count {
                            let v0 = _mm256_loadu_si256(sp as *const __m256i);
                            let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                            let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                            _mm256_store_si256(dp as *mut __m256i, v0);
                            _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                            _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                            sp = sp.add(NR);
                            dp = dp.wrapping_add(stride);
                        }
                        src_p = src_p.wrapping_add(hw);
                        p_global += 1;
                    }
                }
            }
            BMode::Rows => unreachable!("pack_chunks_24 only called on pointwise BMode"),
        }
    }
}

/// Pack one tail chunk (`VECS * 8` readable words at `col`) from `b_mode` into `dst[0..ks * NR]`
/// with row stride `NR` (32-byte aligned per K-step).
///
/// # Safety
/// Caller guarantees AVX2, `dst` is 32-byte aligned with at least `ks * NR` writable words, and `b_mode`
/// has `ks` planes with `col + VECS * 8` readable words (guaranteed by `SLACK`).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_tail_vecs<const VECS: usize>(b_mode: BMode<'_>, ks: usize, col: usize, dst: *mut u32) {
    // SAFETY: pointer bounds and alignment guaranteed by caller.
    unsafe {
        match b_mode {
            BMode::Pointwise1x1 { base, hw } => {
                let mut sp = base.add(col);
                let mut dp = dst;
                for _ in 0..ks {
                    let v0 = _mm256_loadu_si256(sp as *const __m256i);
                    _mm256_store_si256(dp as *mut __m256i, v0);
                    if VECS >= 2 {
                        let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                        _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                    }
                    if VECS >= 3 {
                        let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                        _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                    }
                    sp = sp.wrapping_add(hw);
                    dp = dp.wrapping_add(NR);
                }
            }
            BMode::Pointwise1x1Parts { parts, hw } => {
                let mut dp = dst;
                for &(base, g_cnt) in parts {
                    if g_cnt == 0 {
                        continue;
                    }
                    let mut sp = base.add(col);
                    for _ in 0..g_cnt {
                        let v0 = _mm256_loadu_si256(sp as *const __m256i);
                        _mm256_store_si256(dp as *mut __m256i, v0);
                        if VECS >= 2 {
                            let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                            _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                        }
                        if VECS >= 3 {
                            let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                            _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                        }
                        sp = sp.wrapping_add(hw);
                        dp = dp.wrapping_add(NR);
                    }
                }
            }
            BMode::Rows => unreachable!("pack_tail_vecs only called on pointwise BMode"),
        }
    }
}

/// Pack `count` full `NR`-word chunks from decoupled multi-part `1×1` inputs into `dst[0..count * total_ks * NR]`
/// with layout `dst[(i * total_ks + p_global) * NR .. +NR]` (every row 32-byte aligned).
///
/// # Safety
/// Caller guarantees AVX2, `dst` is 32-byte aligned with at least `count * total_ks * NR` writable words, and
/// each `part` has `part.ks` planes of stride `hw` with `col0 + count * NR` readable words.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_multi_chunks_24(
    parts: &[PartSlice<'_>],
    total_ks: usize,
    hw: usize,
    col0: usize,
    count: usize,
    dst: *mut u32,
) {
    let stride = total_ks * NR;
    let mut p_global = 0usize;
    // SAFETY: pointer bounds and alignment guaranteed by caller.
    unsafe {
        for part in parts {
            if part.ks == 0 {
                continue;
            }
            let mut src_p = part.b_base.add(col0);
            for _ in 0..part.ks {
                let mut sp = src_p;
                let mut dp = dst.add(p_global * NR);
                for _ in 0..count {
                    let v0 = _mm256_loadu_si256(sp as *const __m256i);
                    let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                    let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                    _mm256_store_si256(dp as *mut __m256i, v0);
                    _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                    _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                    sp = sp.add(NR);
                    dp = dp.wrapping_add(stride);
                }
                src_p = src_p.wrapping_add(hw);
                p_global += 1;
            }
        }
    }
}

/// Pack one tail chunk (`VECS * 8` readable words at `col`) from decoupled multi-part `1×1` inputs into
/// `dst[0..total_ks * NR]` with row stride `NR` (32-byte aligned per K-step).
///
/// # Safety
/// Caller guarantees AVX2, `dst` is 32-byte aligned with at least `total_ks * NR` writable words, and each
/// `part` has `part.ks` planes of stride `hw` with `col + VECS * 8` readable words (`SLACK`).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_multi_tail_vecs<const VECS: usize>(
    parts: &[PartSlice<'_>],
    hw: usize,
    col: usize,
    dst: *mut u32,
) {
    let mut dp = dst;
    // SAFETY: pointer bounds and alignment guaranteed by caller.
    unsafe {
        for part in parts {
            if part.ks == 0 {
                continue;
            }
            let mut sp = part.b_base.add(col);
            for _ in 0..part.ks {
                let v0 = _mm256_loadu_si256(sp as *const __m256i);
                _mm256_store_si256(dp as *mut __m256i, v0);
                if VECS >= 2 {
                    let v1 = _mm256_loadu_si256(sp.add(8) as *const __m256i);
                    _mm256_store_si256(dp.add(8) as *mut __m256i, v1);
                }
                if VECS >= 3 {
                    let v2 = _mm256_loadu_si256(sp.add(16) as *const __m256i);
                    _mm256_store_si256(dp.add(16) as *mut __m256i, v2);
                }
                sp = sp.wrapping_add(hw);
                dp = dp.wrapping_add(NR);
            }
        }
    }
}

#[inline(always)]
fn split_w_avx2(w: i32) -> (i32, i32) {
    let b = w.to_le_bytes();
    let ev = (b[0] as i8 as i16 as u16 as u32) | ((b[2] as i8 as i16 as u16 as u32) << 16);
    let od = (b[1] as i8 as i16 as u16 as u32) | ((b[3] as i8 as i16 as u16 as u32) << 16);
    (ev as i32, od as i32)
}

macro_rules! define_packed_1x1_isa {
    (
        mod $mod_name:ident,
        feature = $feat:literal,
        dp_mode = $dp_mode:ident,
        unroll4 = $unroll4:literal
    ) => {
        pub(super) mod $mod_name {
            use super::*;
            const UNROLL4: bool = $unroll4;

            #[target_feature(enable = $feat)]
            #[inline]
            unsafe fn k_loop_packed<const VECS: usize>(
                ks: usize,
                a: *const i32,
                b_panel: *const u32,
                acc: &mut AlignedTile,
            ) {
                let z = _mm256_setzero_si256();
                let (mut c00, mut c01, mut c02) = (z, z, z);
                let (mut c10, mut c11, mut c12) = (z, z, z);
                let (mut c20, mut c21, mut c22) = (z, z, z);
                let (mut c30, mut c31, mut c32) = (z, z, z);
                // SAFETY: caller guarantees `b_panel` is 32-byte aligned with `ks * NR` words and `a` holds
                // `ks * MR` weights. Trailing `wrapping_add` pointers are never dereferenced.
                unsafe {
                    let mut bp = b_panel;
                    let mut ap = a;
                    let mut p = 0;
                    if UNROLL4 {
                        while p + 4 <= ks {
                            define_packed_1x1_isa!(@step $dp_mode, VECS, z, bp, ap,
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            define_packed_1x1_isa!(@step $dp_mode, VECS, z, bp.add(NR), ap.add(MR),
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            define_packed_1x1_isa!(@step $dp_mode, VECS, z, bp.add(2 * NR), ap.add(2 * MR),
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            define_packed_1x1_isa!(@step $dp_mode, VECS, z, bp.add(3 * NR), ap.add(3 * MR),
                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                            bp = bp.wrapping_add(4 * NR);
                            ap = ap.wrapping_add(4 * MR);
                            p += 4;
                        }
                    }
                    while p < ks {
                        define_packed_1x1_isa!(@step $dp_mode, VECS, z, bp, ap,
                            c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                        bp = bp.wrapping_add(NR);
                        ap = ap.wrapping_add(MR);
                        p += 1;
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
            pub(in super::super) unsafe fn run_packed_1x1_spec<
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
            >(
                t: &SinglePartTask<'_>,
            ) {
                let ohw = t.oh * t.ow;
                let ks = t.ks;
                let cb_pack = (PACK_WORDS / (ks * NR)).clamp(1, 16);
                let mut packed = PackedB::uninit();
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut fv = [[0f32; NR]; MR];
                let full_end = (t.r_n / NR).min(t.cb_end);
                let mut c0 = t.cb_start;
                // SAFETY: caller validated all buffer bounds, disjoint task ranges, and `should_pack_1x1(t)`.
                unsafe {
                    while c0 < full_end {
                        let c1 = (c0 + cb_pack).min(full_end);
                        let count = c1 - c0;
                        pack_chunks_24(t.b_mode, ks, c0 * NR, count, packed.as_mut_ptr());
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
                            let res_plane =
                                if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                            let q_plane =
                                if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                            let f_base =
                                if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                            for i in 0..count {
                                let col = (c0 + i) * NR;
                                k_loop_packed::<3>(ks, a_panel, packed.as_ptr().add(i * ks * NR), &mut acc);
                                let res_ptr = if HAS_RES { res_plane.add(col) } else { std::ptr::null() };
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
                                    if OUT_U8 { q_plane.add(col) } else { std::ptr::null_mut() },
                                    if !OUT_U8 { f_base.add(col) } else { std::ptr::null_mut() },
                                    ohw,
                                );
                            }
                        }
                        c0 = c1;
                    }
                    // Optional single partial tail chunk (`r_n % NR != 0`).
                    for chunk in full_end.max(t.cb_start)..t.cb_end {
                        let col = chunk * NR;
                        let cols = NR.min(t.r_n - col);
                        if cols <= 8 {
                            run_packed_tail::<SILU, HAS_RES, OUT_U8, 1>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut qw,
                                &mut fv,
                            );
                        } else if cols <= 16 {
                            run_packed_tail::<SILU, HAS_RES, OUT_U8, 2>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut qw,
                                &mut fv,
                            );
                        } else {
                            run_packed_tail::<SILU, HAS_RES, OUT_U8, 3>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut qw,
                                &mut fv,
                            );
                        }
                    }
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            #[allow(clippy::too_many_arguments)]
            unsafe fn run_packed_tail<
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
                const VECS: usize,
            >(
                t: &SinglePartTask<'_>,
                col: usize,
                cols: usize,
                packed: &mut PackedB,
                acc: &mut AlignedTile,
                qw: &mut [u32; NR],
                fv: &mut [[f32; NR]; MR],
            ) {
                let ohw = t.oh * t.ow;
                let ks = t.ks;
                let direct = cols == VECS * 8;
                let mut resw: [u32; NR];
                // SAFETY: caller validated all buffer bounds and ISA.
                unsafe {
                    pack_tail_vecs::<VECS>(t.b_mode, ks, col, packed.as_mut_ptr());
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
                        k_loop_packed::<VECS>(ks, a_panel, packed.as_ptr(), acc);
                        let res_ptr = if HAS_RES {
                            if direct {
                                res_plane.add(col)
                            } else {
                                resw = [0u32; NR];
                                std::ptr::copy_nonoverlapping(res_plane.add(col), resw.as_mut_ptr(), cols);
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
                            if direct {
                                (f_base.add(col), ohw)
                            } else {
                                (fv.as_mut_ptr() as *mut f32, NR)
                            }
                        } else {
                            (std::ptr::null_mut(), 0)
                        };
                        apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, VECS>(
                            acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv, &z01_arr, &z23_arr, q_dst,
                            f_dst, f_stride,
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
                    }
                }
            }

            #[target_feature(enable = $feat)]
            pub(in super::super) unsafe fn run_packed_1x1_multi_spec<
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
            >(
                t: &MultiPartTask<'_>,
            ) {
                let ohw = t.oh * t.ow;
                let total_ks: usize = t.parts.iter().map(|p| p.ks).sum();
                let cb_pack = (PACK_WORDS / (total_ks * NR)).clamp(1, 16);
                let mut packed = PackedB::uninit();
                let mut acc = AlignedTile([[0; NR]; MR]);
                let mut tile_f32 = AlignedTileF32([[0.0; NR]; MR]);
                let mut qw = [0u32; NR];
                let mut fv = [[0f32; NR]; MR];
                let full_end = (t.r_n / NR).min(t.cb_end);
                let mut c0 = t.cb_start;
                let zf = _mm256_setzero_ps();
                // SAFETY: caller validated all buffer bounds, disjoint task ranges, and `should_pack_1x1_multi(t)`.
                unsafe {
                    while c0 < full_end {
                        let c1 = (c0 + cb_pack).min(full_end);
                        let count = c1 - c0;
                        pack_multi_chunks_24(
                            t.parts,
                            total_ks,
                            t.pointwise_hw,
                            c0 * NR,
                            count,
                            packed.as_mut_ptr(),
                        );
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
                            let res_plane =
                                if HAS_RES { t.res_img_ptr.add(g * ohw) } else { std::ptr::null() };
                            let q_plane =
                                if OUT_U8 { t.q_img_ptr.add(g * ohw) } else { std::ptr::null_mut() };
                            let f_base =
                                if !OUT_U8 { t.f_img_ptr.add(4 * g * ohw) } else { std::ptr::null_mut() };
                            for i in 0..count {
                                let col = (c0 + i) * NR;
                                let chunk_b = packed.as_ptr().add(i * total_ks * NR);
                                let mut p_off = 0usize;
                                for (pi, part) in t.parts.iter().enumerate() {
                                    k_loop_packed::<3>(
                                        part.ks,
                                        part.a.as_ptr().add(g * part.ks * MR),
                                        chunk_b.add(p_off * NR),
                                        &mut acc,
                                    );
                                    p_off += part.ks;
                                    for r in 0..MR {
                                        let vc = _mm256_set1_epi32(part.corr[4 * g + r]);
                                        let vs = _mm256_set1_ps(part.scale[4 * g + r]);
                                        for k in 0..3 {
                                            let a_v = _mm256_load_si256(
                                                acc.0[r].as_ptr().add(8 * k) as *const __m256i,
                                            );
                                            let cur = if pi == 0 {
                                                zf
                                            } else {
                                                _mm256_load_ps(tile_f32.0[r].as_ptr().add(8 * k))
                                            };
                                            let next = _mm256_fmadd_ps(
                                                _mm256_cvtepi32_ps(_mm256_sub_epi32(a_v, vc)),
                                                vs,
                                                cur,
                                            );
                                            _mm256_store_ps(tile_f32.0[r].as_mut_ptr().add(8 * k), next);
                                        }
                                    }
                                }
                                let res_ptr = if HAS_RES { res_plane.add(col) } else { std::ptr::null() };
                                apply_epi_f32_aligned::<SILU, HAS_RES, OUT_U8, 3>(
                                    &tile_f32,
                                    &bias,
                                    res_ptr,
                                    &rs,
                                    &rz,
                                    &inv,
                                    &z01_arr,
                                    &z23_arr,
                                    if OUT_U8 { q_plane.add(col) } else { std::ptr::null_mut() },
                                    if !OUT_U8 { f_base.add(col) } else { std::ptr::null_mut() },
                                    ohw,
                                );
                            }
                        }
                        c0 = c1;
                    }
                    for chunk in full_end.max(t.cb_start)..t.cb_end {
                        let col = chunk * NR;
                        let cols = NR.min(t.r_n - col);
                        if cols <= 8 {
                            run_packed_multi_tail::<SILU, HAS_RES, OUT_U8, 1>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut tile_f32,
                                &mut qw,
                                &mut fv,
                            );
                        } else if cols <= 16 {
                            run_packed_multi_tail::<SILU, HAS_RES, OUT_U8, 2>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut tile_f32,
                                &mut qw,
                                &mut fv,
                            );
                        } else {
                            run_packed_multi_tail::<SILU, HAS_RES, OUT_U8, 3>(
                                t,
                                col,
                                cols,
                                &mut packed,
                                &mut acc,
                                &mut tile_f32,
                                &mut qw,
                                &mut fv,
                            );
                        }
                    }
                }
            }

            #[target_feature(enable = $feat)]
            #[inline]
            #[allow(clippy::too_many_arguments)]
            unsafe fn run_packed_multi_tail<
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
                const VECS: usize,
            >(
                t: &MultiPartTask<'_>,
                col: usize,
                cols: usize,
                packed: &mut PackedB,
                acc: &mut AlignedTile,
                tile_f32: &mut AlignedTileF32,
                qw: &mut [u32; NR],
                fv: &mut [[f32; NR]; MR],
            ) {
                let ohw = t.oh * t.ow;
                let direct = cols == VECS * 8;
                let mut resw: [u32; NR];
                let zf = _mm256_setzero_ps();
                // SAFETY: caller validated all buffer bounds and ISA.
                unsafe {
                    pack_multi_tail_vecs::<VECS>(t.parts, t.pointwise_hw, col, packed.as_mut_ptr());
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
                        let mut p_off = 0usize;
                        for (pi, part) in t.parts.iter().enumerate() {
                            k_loop_packed::<VECS>(
                                part.ks,
                                part.a.as_ptr().add(g * part.ks * MR),
                                packed.as_ptr().add(p_off * NR),
                                acc,
                            );
                            p_off += part.ks;
                            for r in 0..MR {
                                let vc = _mm256_set1_epi32(part.corr[4 * g + r]);
                                let vs = _mm256_set1_ps(part.scale[4 * g + r]);
                                for k in 0..VECS {
                                    let a_v = _mm256_load_si256(
                                        acc.0[r].as_ptr().add(8 * k) as *const __m256i,
                                    );
                                    let cur = if pi == 0 {
                                        zf
                                    } else {
                                        _mm256_load_ps(tile_f32.0[r].as_ptr().add(8 * k))
                                    };
                                    let next = _mm256_fmadd_ps(
                                        _mm256_cvtepi32_ps(_mm256_sub_epi32(a_v, vc)),
                                        vs,
                                        cur,
                                    );
                                    _mm256_store_ps(tile_f32.0[r].as_mut_ptr().add(8 * k), next);
                                }
                            }
                        }
                        let res_ptr = if HAS_RES {
                            if direct {
                                res_plane.add(col)
                            } else {
                                resw = [0u32; NR];
                                std::ptr::copy_nonoverlapping(res_plane.add(col), resw.as_mut_ptr(), cols);
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
                            if direct {
                                (f_base.add(col), ohw)
                            } else {
                                (fv.as_mut_ptr() as *mut f32, NR)
                            }
                        } else {
                            (std::ptr::null_mut(), 0)
                        };
                        apply_epi_f32_aligned::<SILU, HAS_RES, OUT_U8, VECS>(
                            tile_f32, &bias, res_ptr, &rs, &rz, &inv, &z01_arr, &z23_arr, q_dst, f_dst,
                            f_stride,
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
                    }
                }
            }
        }
    };

    (@step $dp_mode:ident, $vecs:expr, $z:ident, $bp:expr, $ap:expr,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {{
        let bp = ($bp) as *const __m256i;
        let b0 = _mm256_load_si256(bp);
        let b1 = if $vecs >= 2 { _mm256_load_si256(bp.add(1)) } else { $z };
        let b2 = if $vecs >= 3 { _mm256_load_si256(bp.add(2)) } else { $z };
        let ap = $ap;
        define_packed_1x1_isa!(@dp $dp_mode, $vecs, b0, b1, b2, ap,
            $c00, $c01, $c02, $c10, $c11, $c12, $c20, $c21, $c22, $c30, $c31, $c32);
    }};

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

define_packed_1x1_isa!(mod vnni_pack, feature = "avx2,fma,avxvnni", dp_mode = avxvnni, unroll4 = true);
define_packed_1x1_isa!(mod avx512vnni_pack, feature = "avx2,fma,avx512vnni,avx512vl", dp_mode = avx512vnni, unroll4 = true);
define_packed_1x1_isa!(mod avx2_pack, feature = "avx2,fma", dp_mode = avx2, unroll4 = false);
