//! Single-barrier, L1-hot rotating row-strip fused x86-64 driver for single-part `3×3` (`pad = 1`)
//! int8 convolutions (`q8::conv` / `q8::conv_per_channel`):
//! - All `stride = 2, pad = 1` convs (3 rotating `(ev, od)` deinterleaved input row slots per task, `15–31 KiB`
//!   in L1D, reusing `ky = 2 -> ky = 0` across consecutive rows).
//! - Large-input `stride = 1, pad = 1` convs (`input > 384 KiB`, e.g. `64->64 80×80` and `folo-s` high-res layers),
//!   where the global padded copy would otherwise spill across cores through L3 and pay a second rayon barrier.
//!
//! Arithmetic order is bit-for-bit identical to [`super::kernel::kernel_scalar`] + [`super::kernel::Epi::apply`].

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;
use std::cell::RefCell;

use super::conv::OutPtr;
use super::fused_epi::{AlignedTile, FusedIsa, apply_epi_aligned_fstride, zero_pairs};
use super::kernel::{MR, NR};
use super::{C4, deinterleave_even_odd, recycle, words};

/// Cap on per-thread retained scratch (`4 MiB`) so pathological huge shapes do not bloat thread-local memory.
const MAX_RETAINED_WORDS: usize = 1 << 20;
/// Maximum scratch words allowed for the fused row-strip path before falling back to the general path (`16 MiB`).
const MAX_SCRATCH_WORDS: usize = 1 << 22;

thread_local! {
    static FUSED_3X3_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a reusable per-thread `u32` scratch slice of length `len` (re-entry-safe).
#[inline]
fn with_scratch_u32<R>(len: usize, f: impl FnOnce(&mut [u32]) -> R) -> R {
    let mut loaned = FUSED_3X3_SCRATCH
        .try_with(|cell| {
            let Ok(mut v) = cell.try_borrow_mut() else { return None };
            if v.len() < len {
                v.resize(len, 0);
            }
            Some(std::mem::take(&mut *v))
        })
        .ok()
        .flatten();
    if let Some(mut v) = loaned.take() {
        let r = f(&mut v[..len]);
        if v.capacity() <= MAX_RETAINED_WORDS {
            let _ = FUSED_3X3_SCRATCH.try_with(|cell| {
                if let Ok(mut slot) = cell.try_borrow_mut()
                    && v.capacity() >= slot.capacity()
                {
                    *slot = v;
                }
            });
        }
        return r;
    }
    let mut v = vec![0u32; len];
    f(&mut v[..len])
}

/// Read-only parameters for one `3×3` (`stride ∈ {1, 2}`, `pad = 1`) fused row-strip conv.
pub(super) struct Conv3x3Params<'a> {
    pub xs: &'a [C4<'a>],
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub oh: usize,
    pub ow: usize,
    pub groups: usize,
    pub og: usize,
    pub zeros: &'a [u8],
    pub a: &'a [i32],
    pub corr: &'a [i32],
    pub scale: &'a [f32],
    pub bias: &'a [f32],
    pub silu: bool,
    pub is_u8: bool,
    pub group_out: &'a [([f32; MR], [i32; MR])],
    pub group_res: &'a [([f32; MR], [i32; MR])],
    pub res_view: Option<C4<'a>>,
    pub qptr: OutPtr<u32>,
    pub fptr: OutPtr<f32>,
}

/// Per-row arguments passed into the 512-bit AVX-512 VNNI `run_3x3_row` kernel.
pub(super) struct RowArgs<'a> {
    pub(super) p: &'a Conv3x3Params<'a>,
    pub(super) img: usize,
    pub(super) oy: usize,
    pub(super) ob_start: usize,
    pub(super) ob_end: usize,
    pub(super) group_stride: usize,
    pub(super) ev_w: usize,
    pub(super) slot_words: usize,
    pub(super) ord: [usize; 3],
    pub(super) buf: &'a [u32],
}

/// Multi-row batch arguments (`rows_len ∈ 1..=4`) passed into 256-bit `compute_rows` kernels so each `g`'s
/// weight panel is loaded once and reused across `rows_len` output rows inside L1D.
struct BatchRowArgs<'a> {
    p: &'a Conv3x3Params<'a>,
    img: usize,
    oy_start: usize,
    rows_len: usize,
    ob_start: usize,
    ob_end: usize,
    group_stride: usize,
    ev_w: usize,
    slot_words: usize,
    ord: [usize; 9],
    buf: &'a [u32],
}

/// Source group lookup `(part_index, local_group_index)` and precomputed packed zero word.
#[derive(Clone, Copy)]
struct GroupSrc {
    part: usize,
    local_g: usize,
    zw: u32,
}

macro_rules! define_3x3_strip_isa {
    (
        mod $mod_name:ident,
        feature = $feat:literal,
        dp_mode = $dp_mode:ident
    ) => {
        mod $mod_name {
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
                        let ptrs = [p0, p1, p2];
                        for ev in ptrs {
                            let od = ev.add(ev_w);
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
                            tap!(od, 0);
                            tap!(ev, 1);
                            tap!(od.add(1), 2);
                            ap = ap.wrapping_add(3 * MR);
                        }
                        p0 = p0.wrapping_add(ch_stride);
                        p1 = p1.wrapping_add(ch_stride);
                        p2 = p2.wrapping_add(ch_stride);
                    }
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
                        for row_ptr in [p0, p1, p2] {
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
                            tap!(row_ptr, 0);
                            tap!(row_ptr.add(1), 1);
                            tap!(row_ptr.add(2), 2);
                            ap = ap.wrapping_add(3 * MR);
                        }
                        p0 = p0.wrapping_add(slot_w);
                        p1 = p1.wrapping_add(slot_w);
                        p2 = p2.wrapping_add(slot_w);
                    }
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
                                    k_loop_s2::<3>(p.groups, a_panel, slots, group_stride, ev_w, col, &mut acc);
                                } else {
                                    k_loop_s1::<3>(p.groups, a_panel, slots, group_stride, col, &mut acc);
                                }
                                let res_ptr = if HAS_RES { res_plane.add(row_off + col) } else { std::ptr::null() };
                                apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, 3>(
                                    &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                    &z01_arr, &z23_arr,
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
                                            k_loop_s2::<$vecs>(p.groups, a_panel, slots, group_stride, ev_w, col, &mut acc);
                                        } else {
                                            k_loop_s1::<$vecs>(p.groups, a_panel, slots, group_stride, col, &mut acc);
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
                                            if direct { (f_base.add(row_off + col), ohw) } else { (fv.as_mut_ptr() as *mut f32, NR) }
                                        } else {
                                            (std::ptr::null_mut(), 0)
                                        };
                                        apply_epi_aligned_fstride::<SILU, HAS_RES, OUT_U8, $vecs>(
                                            &acc, &corr, &scale, &bias, res_ptr, &rs, &rz, &inv,
                                            &z01_arr, &z23_arr, q_dst, f_dst, f_stride,
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

            pub(super) unsafe fn run_rows<const IS_S2: bool>(r: &BatchRowArgs<'_>) {
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

    (@dp avx512vnni, $vecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
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

define_3x3_strip_isa!(mod vnni_3x3, feature = "avx2,fma,avxvnni", dp_mode = avxvnni);
define_3x3_strip_isa!(mod avx512vnni_3x3, feature = "avx2,fma,avx512vnni,avx512vl", dp_mode = avx512vnni);
define_3x3_strip_isa!(mod avx2_3x3, feature = "avx2,fma", dp_mode = avx2);

/// Plan `(row_blk, og_blk)` for the single-barrier fused `3×3` row-strip driver.
fn plan_3x3_strip(
    n: usize,
    oh: usize,
    ow: usize,
    groups: usize,
    og: usize,
    is_s2: bool,
    (threads, min_macs): (usize, usize),
) -> (usize, usize) {
    let macs = n
        .saturating_mul(oh)
        .saturating_mul(ow)
        .saturating_mul(og * 4)
        .saturating_mul(groups * 4)
        .saturating_mul(9);
    let serial_floor = if threads >= 16 { min_macs.saturating_mul(8) } else { min_macs.saturating_mul(2) };
    let max_tasks = (3 * threads).max(1);
    let want = if threads > 1 && macs >= serial_floor {
        max_tasks.min(macs / min_macs.max(1)).max(1)
    } else {
        n.min(threads.max(1)).max(1)
    };
    let per_img = want.div_ceil(n);
    // Keep at least 2 output rows per band for stride-2 (so `ky = 2 -> ky = 0` rotation is actually reused)
    // and at least 4 rows per band for stride-1 (so the 3-row fill prologue is amortized and weight re-reads
    // stay bounded).
    let min_row_blk = if is_s2 { 2 } else { 4 };
    let max_rbs = (oh / min_row_blk).max(1);
    let rbs = max_rbs.min(per_img).max(1);
    let row_blk = oh.div_ceil(rbs);
    let rbs = oh.div_ceil(row_blk);
    let total_rbs = n * rbs;
    let og_blk = if total_rbs < want && og > 1 && macs / total_rbs >= min_macs.saturating_mul(3) {
        // Cap duplicate row-band packing across output channel blocks at 4× so small `oh` (e.g. rect 640×384
        // or 20×20/40×40 feature maps) never re-packs the same input rows excessively.
        let max_obs = (max_tasks / total_rbs).clamp(1, 4);
        let want_o = ((per_img + rbs / 2) / rbs).clamp(1, max_obs).min(og);
        og.div_ceil(want_o)
    } else {
        og
    };
    (row_blk, og_blk)
}

/// Fill one padded `s = 1, p = 1` input row `iy` across all `groups` into `dst[0..groups * slot_w]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fill_s1_row(
    xs: &[C4<'_>],
    gsrcs: &[GroupSrc],
    img: usize,
    h: usize,
    w: usize,
    slot_w: usize,
    iy: isize,
    dst: &mut [u32],
) {
    let hw = h * w;
    if iy < 0 || iy >= h as isize {
        for (g, gs) in gsrcs.iter().enumerate() {
            dst[g * slot_w..(g + 1) * slot_w].fill(gs.zw);
        }
        return;
    }
    let y = iy as usize;
    for (g, gs) in gsrcs.iter().enumerate() {
        let x = &xs[gs.part];
        let base = x.offset + img * x.img_stride + gs.local_g * hw + y * w;
        let srow = &x.data[base..base + w];
        let drow = &mut dst[g * slot_w..(g + 1) * slot_w];
        drow[0] = gs.zw;
        drow[1..1 + w].copy_from_slice(srow);
        drow[1 + w..].fill(gs.zw);
    }
}

/// Fill one deinterleaved `s = 2, p = 1` input row `iy` across all `groups` into `dst[0..groups * ch_stride]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fill_s2_row(
    xs: &[C4<'_>],
    gsrcs: &[GroupSrc],
    img: usize,
    h: usize,
    w: usize,
    ow: usize,
    ev_w: usize,
    ch_stride: usize,
    iy: isize,
    dst: &mut [u32],
) {
    let hw = h * w;
    if iy < 0 || iy >= h as isize {
        for (g, gs) in gsrcs.iter().enumerate() {
            dst[g * ch_stride..(g + 1) * ch_stride].fill(gs.zw);
        }
        return;
    }
    let y = iy as usize;
    for (g, gs) in gsrcs.iter().enumerate() {
        let x = &xs[gs.part];
        let base = x.offset + img * x.img_stride + gs.local_g * hw + y * w;
        let srow = &x.data[base..base + w];
        let dslot = &mut dst[g * ch_stride..(g + 1) * ch_stride];
        let (ev, od) = dslot.split_at_mut(ev_w);
        od[0] = gs.zw;
        deinterleave_even_odd(srow, &mut ev[..ow], &mut od[1..1 + ow], gs.zw);
        ev[ow..].fill(gs.zw);
        od[1 + ow..].fill(gs.zw);
    }
}

/// Attempt to run a single-part `k = 3, pad = 1, stride ∈ {1, 2}` conv on the fused single-barrier row-strip
/// x86-64 driver. Returns `false` if the shape falls outside scratch limits so the caller can fall back.
///
/// # Safety
/// `isa` must match [`super::fused_epi::active_isa()`]; all buffer extents in `p` must be validated by the caller.
pub(super) unsafe fn try_run_3x3_strip(
    isa: FusedIsa,
    p: &Conv3x3Params<'_>,
    stride: usize,
    min_macs: usize,
) -> bool {
    if stride != 1 && stride != 2 {
        return false;
    }
    let Some(ow8) = p.ow.checked_next_multiple_of(8) else { return false };
    let (group_stride, ev_w) = if stride == 2 {
        let ev_w = ow8;
        let Some(od_w) = ow8.checked_add(1) else { return false };
        let Some(ch_stride) = ev_w.checked_add(od_w).and_then(|v| v.checked_next_multiple_of(8)) else {
            return false;
        };
        (ch_stride, ev_w)
    } else {
        let Some(slot_w) = ow8.checked_add(2).and_then(|v| v.checked_next_multiple_of(8)) else {
            return false;
        };
        (slot_w, 0)
    };
    let Some(slot_words) = p.groups.checked_mul(group_stride) else { return false };
    let Some(base_scratch_words) = slot_words.checked_mul(3) else { return false };
    if base_scratch_words > MAX_SCRATCH_WORDS {
        return false;
    }

    const STACK_G: usize = 128;
    let mut gsrc_stack = [GroupSrc { part: 0, local_g: 0, zw: 0 }; STACK_G];
    let mut gsrc_heap = Vec::new();
    let gsrcs: &[GroupSrc] = {
        let slice = if p.groups <= STACK_G {
            &mut gsrc_stack[..p.groups]
        } else {
            gsrc_heap.resize(p.groups, GroupSrc { part: 0, local_g: 0, zw: 0 });
            &mut gsrc_heap[..p.groups]
        };
        let mut gi = 0usize;
        for (pi, x) in p.xs.iter().enumerate() {
            for lg in 0..x.c / 4 {
                let zw = u32::from_le_bytes([
                    p.zeros[4 * gi],
                    p.zeros[4 * gi + 1],
                    p.zeros[4 * gi + 2],
                    p.zeros[4 * gi + 3],
                ]);
                slice[gi] = GroupSrc { part: pi, local_g: lg, zw };
                gi += 1;
            }
        }
        debug_assert_eq!(gi, p.groups);
        slice
    };

    let threads = rayon::current_num_threads();
    let is_s2 = stride == 2;

    // Small-spatial stride-1 fast path (e.g. `20×20` P5 layers: `64->64` × 9 calls, `256->64` × 1 call):
    // when the entire `(oh + 2)` padded strip fits in `<= 256 KiB` (`65536` words), pack each image's padded
    // rows ONCE into a pooled buffer and split tasks by output channel group `og` first. This avoids 4×–16×
    // duplicate input row packing and keeps each task's weight panel hot across all rows of its plane.
    let shared_s1_words = (!is_s2 && p.oh <= 24)
        .then(|| p.oh.checked_add(2).and_then(|rows| rows.checked_mul(slot_words)))
        .flatten()
        .filter(|&img_w| {
            img_w <= 65_536 && p.n.checked_mul(img_w).is_some_and(|tot| tot <= MAX_SCRATCH_WORDS)
        });
    if let Some(img_words) = shared_s1_words
        && isa != FusedIsa::Avx512Vnni512
    {
        let macs =
            p.n.saturating_mul(p.oh)
                .saturating_mul(p.ow)
                .saturating_mul(p.og * 4)
                .saturating_mul(p.groups * 4)
                .saturating_mul(9);
        let serial_floor =
            if threads >= 16 { min_macs.saturating_mul(8) } else { min_macs.saturating_mul(2) };
        let max_tasks = (3 * threads).max(1);
        let want = if threads > 1 && macs >= serial_floor {
            max_tasks.min(macs / min_macs.max(1)).max(1)
        } else {
            p.n.min(threads.max(1)).max(1)
        };
        let per_img = want.div_ceil(p.n);
        let og_blk = p.og.div_ceil(per_img.min(p.og));
        let obs = p.og.div_ceil(og_blk);
        let max_rbs = (max_tasks / (p.n * obs)).min((p.oh / 4).max(1)).max(1);
        let want_r = ((per_img + obs / 2) / obs).clamp(1, max_rbs);
        let row_blk = p.oh.div_ceil(want_r.min(p.oh));
        let rbs = p.oh.div_ceil(row_blk);
        let tasks = p.n * rbs * obs;
        let padded_rows = p.oh + 2;
        let total_words = p.n * img_words;
        let mut shared = words(total_words);
        let sptr = OutPtr(shared.as_mut_ptr());
        let sptr = &sptr;
        let fill_one = |ir: usize| {
            let (img, r) = (ir / padded_rows, ir % padded_rows);
            // SAFETY: each `(img, r)` owns disjoint `slot_words` inside `shared[..total_words]`.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(sptr.at(img * img_words + r * slot_words), slot_words)
            };
            fill_s1_row(p.xs, gsrcs, img, p.h, p.w, group_stride, r as isize - 1, dst);
        };
        if total_words >= (1 << 14) && threads > 1 {
            crate::cpu::par_for(p.n * padded_rows, &fill_one);
        } else {
            (0..p.n * padded_rows).for_each(fill_one);
        }
        let shared_slice: &[u32] = &shared[..total_words];
        let run_shared = |t: usize| {
            let (img, rest) = (t / (rbs * obs), t % (rbs * obs));
            let (ob, rb) = (rest / rbs, rest % rbs);
            let oy0 = rb * row_blk;
            let oy1 = (oy0 + row_blk).min(p.oh);
            let ob_start = ob * og_blk;
            let ob_end = (ob_start + og_blk).min(p.og);
            let img_buf = &shared_slice[img * img_words..(img + 1) * img_words];
            let ord = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
            let mut oy = oy0;
            while oy < oy1 {
                let b_len = 4usize.min(oy1 - oy);
                let sub = &img_buf[oy * slot_words..(oy + b_len + 2) * slot_words];
                let batch_args = BatchRowArgs {
                    p,
                    img,
                    oy_start: oy,
                    rows_len: b_len,
                    ob_start,
                    ob_end,
                    group_stride,
                    ev_w: 0,
                    slot_words,
                    ord,
                    buf: sub,
                };
                // SAFETY: ISA checked by caller; `(img, oy..oy+b_len, ob_start..ob_end)` is disjointly owned.
                unsafe {
                    match isa {
                        FusedIsa::AvxVnni => vnni_3x3::run_rows::<false>(&batch_args),
                        FusedIsa::Avx512Vnni => avx512vnni_3x3::run_rows::<false>(&batch_args),
                        FusedIsa::Avx2 => avx2_3x3::run_rows::<false>(&batch_args),
                        FusedIsa::Avx512Vnni512 => unreachable!(),
                    }
                }
                oy += b_len;
            }
        };
        if tasks <= 1 {
            (0..tasks).for_each(run_shared);
        } else {
            crate::cpu::par_for(tasks, &run_shared);
        }
        recycle(shared);
        return true;
    }

    let (row_blk, og_blk) = plan_3x3_strip(p.n, p.oh, p.ow, p.groups, p.og, is_s2, (threads, min_macs));
    let rbs = p.oh.div_ceil(row_blk);
    let obs = p.og.div_ceil(og_blk);
    let tasks = p.n * rbs * obs;
    // Choose L1-resident multi-row batch size `max_rb ∈ {1, 2, 4}` for 256-bit ISAs when `ow <= 96` so each
    // weight panel `a_panel` is reused across `max_rb` output rows without spilling the strip out of L1D (32 KiB).
    let max_rb = if isa == FusedIsa::Avx512Vnni512 || p.ow > 96 {
        1
    } else if row_blk >= 4 && (stride * 3 + 3).saturating_mul(slot_words).saturating_mul(4) <= (32 << 10) {
        4
    } else if row_blk >= 2 && (stride + 3).saturating_mul(slot_words).saturating_mul(4) <= (32 << 10) {
        2
    } else {
        1
    };
    let max_slots = stride * (max_rb - 1) + 3;
    let scratch_words = slot_words * max_slots;
    let overlap = 3 - stride;
    let run_task = |t: usize| {
        let (img, rest) = (t / (rbs * obs), t % (rbs * obs));
        let (rb, ob) = (rest / obs, rest % obs);
        let oy0 = rb * row_blk;
        let oy1 = (oy0 + row_blk).min(p.oh);
        let ob_start = ob * og_blk;
        let ob_end = (ob_start + og_blk).min(p.og);
        with_scratch_u32(scratch_words, |buf| {
            let fill_row = |iy: isize, dst: &mut [u32]| {
                if is_s2 {
                    fill_s2_row(p.xs, gsrcs, img, p.h, p.w, p.ow, ev_w, group_stride, iy, dst);
                } else {
                    fill_s1_row(p.xs, gsrcs, img, p.h, p.w, group_stride, iy, dst);
                }
            };
            if isa == FusedIsa::Avx512Vnni512 {
                let mut ord = [0usize, 1, 2];
                for (i, oy) in (oy0..oy1).enumerate() {
                    let fill_from = if i == 0 {
                        0
                    } else {
                        ord.rotate_left(stride);
                        overlap
                    };
                    for (s_idx, &sl) in ord.iter().enumerate().skip(fill_from) {
                        let iy = (stride * oy) as isize - 1 + s_idx as isize;
                        fill_row(iy, &mut buf[sl * slot_words..(sl + 1) * slot_words]);
                    }
                    let row_args = RowArgs {
                        p,
                        img,
                        oy,
                        ob_start,
                        ob_end,
                        group_stride,
                        ev_w: if is_s2 { ev_w } else { 0 },
                        slot_words,
                        ord,
                        buf,
                    };
                    // SAFETY: ISA checked by caller; `(img, oy, ob_start..ob_end)` is disjointly owned.
                    unsafe {
                        if is_s2 {
                            super::vnni512::run_3x3_row::<true>(&row_args);
                        } else {
                            super::vnni512::run_3x3_row::<false>(&row_args);
                        }
                    }
                }
                return;
            }
            let mut ord = [0usize, 1, 2, 3, 4, 5, 6, 7, 8];
            let mut oy = oy0;
            let mut first = true;
            while oy < oy1 {
                let b_len = max_rb.min(oy1 - oy);
                let need_slots = stride * (b_len - 1) + 3;
                let fill_from = if first {
                    first = false;
                    0
                } else {
                    ord[..max_slots].rotate_left(stride * max_rb);
                    overlap
                };
                for (s_idx, &sl) in ord[..need_slots].iter().enumerate().skip(fill_from) {
                    let iy = (stride * oy) as isize - 1 + s_idx as isize;
                    fill_row(iy, &mut buf[sl * slot_words..(sl + 1) * slot_words]);
                }
                let batch_args = BatchRowArgs {
                    p,
                    img,
                    oy_start: oy,
                    rows_len: b_len,
                    ob_start,
                    ob_end,
                    group_stride,
                    ev_w: if is_s2 { ev_w } else { 0 },
                    slot_words,
                    ord,
                    buf,
                };
                // SAFETY: ISA checked by caller; `(img, oy..oy+b_len, ob_start..ob_end)` is disjointly owned.
                unsafe {
                    if is_s2 {
                        match isa {
                            FusedIsa::AvxVnni => vnni_3x3::run_rows::<true>(&batch_args),
                            FusedIsa::Avx512Vnni => avx512vnni_3x3::run_rows::<true>(&batch_args),
                            FusedIsa::Avx2 => avx2_3x3::run_rows::<true>(&batch_args),
                            FusedIsa::Avx512Vnni512 => unreachable!(),
                        }
                    } else {
                        match isa {
                            FusedIsa::AvxVnni => vnni_3x3::run_rows::<false>(&batch_args),
                            FusedIsa::Avx512Vnni => avx512vnni_3x3::run_rows::<false>(&batch_args),
                            FusedIsa::Avx2 => avx2_3x3::run_rows::<false>(&batch_args),
                            FusedIsa::Avx512Vnni512 => unreachable!(),
                        }
                    }
                }
                oy += b_len;
            }
        });
    };
    if tasks <= 1 {
        (0..tasks).for_each(run_task);
    } else {
        crate::cpu::par_for(tasks, &run_task);
    }
    true
}
