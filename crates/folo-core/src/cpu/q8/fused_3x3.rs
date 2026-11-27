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
use super::fused_epi::{AlignedTile, FusedIsa, apply_epi_aligned, zero_pairs};
use super::kernel::{MR, NR};
use super::{C4, deinterleave_even_odd};

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

/// Per-row arguments passed into the target-feature-specialized `compute_row` / `run_row` kernels.
struct RowArgs<'a> {
    p: &'a Conv3x3Params<'a>,
    img: usize,
    oy: usize,
    ob_start: usize,
    ob_end: usize,
    group_stride: usize,
    ev_w: usize,
    slot_words: usize,
    ord: [usize; 3],
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
            unsafe fn k_loop_s2(
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
                // SAFETY: caller guarantees `groups * 9 * MR` weights and valid `(ev, od)` extents per slot.
                // Pointer advances are guarded by `g + 1 < groups` so no pointer steps past the buffer end.
                unsafe {
                    let mut p0 = slots[0].add(col);
                    let mut p1 = slots[1].add(col);
                    let mut p2 = slots[2].add(col);
                    let mut ap = a;
                    for g in 0..groups {
                        let ptrs = [p0, p1, p2];
                        for ev in ptrs {
                            let od = ev.add(ev_w);
                            macro_rules! tap {
                                        ($bp:expr, $aoff:expr) => {
                                            let bp = ($bp) as *const __m256i;
                                            let b0 = _mm256_loadu_si256(bp);
                                            let b1 = _mm256_loadu_si256(bp.add(1));
                                            let b2 = _mm256_loadu_si256(bp.add(2));
                                            let wp = ap.add(($aoff) * MR);
                                            define_3x3_strip_isa!(@dp $dp_mode, b0, b1, b2, wp,
                                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                                        };
                                    }
                            tap!(od, 0);
                            tap!(ev, 1);
                            tap!(od.add(1), 2);
                            ap = ap.add(3 * MR);
                        }
                        if g + 1 < groups {
                            p0 = p0.add(ch_stride);
                            p1 = p1.add(ch_stride);
                            p2 = p2.add(ch_stride);
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
            #[allow(clippy::too_many_arguments)]
            unsafe fn k_loop_s1(
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
                // SAFETY: caller guarantees `groups * 9 * MR` weights and `col + NR + 2 <= slot_w` per group row.
                // Pointer advances are guarded by `g + 1 < groups` so no pointer steps past the buffer end.
                unsafe {
                    let mut p0 = slots[0].add(col);
                    let mut p1 = slots[1].add(col);
                    let mut p2 = slots[2].add(col);
                    let mut ap = a;
                    for g in 0..groups {
                        for row_ptr in [p0, p1, p2] {
                            macro_rules! tap {
                                        ($bp:expr, $aoff:expr) => {
                                            let bp = ($bp) as *const __m256i;
                                            let b0 = _mm256_loadu_si256(bp);
                                            let b1 = _mm256_loadu_si256(bp.add(1));
                                            let b2 = _mm256_loadu_si256(bp.add(2));
                                            let wp = ap.add(($aoff) * MR);
                                            define_3x3_strip_isa!(@dp $dp_mode, b0, b1, b2, wp,
                                                c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                                        };
                                    }
                            tap!(row_ptr, 0);
                            tap!(row_ptr.add(1), 1);
                            tap!(row_ptr.add(2), 2);
                            ap = ap.add(3 * MR);
                        }
                        if g + 1 < groups {
                            p0 = p0.add(slot_w);
                            p1 = p1.add(slot_w);
                            p2 = p2.add(slot_w);
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
            unsafe fn compute_row<
                const IS_S2: bool,
                const SILU: bool,
                const HAS_RES: bool,
                const OUT_U8: bool,
            >(
                r: &RowArgs<'_>,
            ) {
                let p = r.p;
                let (img, oy, ob_start, ob_end) = (r.img, r.oy, r.ob_start, r.ob_end);
                let (group_stride, ev_w, slot_words, ord, buf) =
                    (r.group_stride, r.ev_w, r.slot_words, r.ord, r.buf);
                let ohw = p.oh * p.ow;
                let chunks = p.ow.div_ceil(NR);
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
                let row_off = oy * p.ow;
                // SAFETY: caller validated all buffer bounds and disjoint `(img, oy, ob_start..ob_end)` ownership.
                unsafe {
                    let slots = [
                        bptr.add(ord[0] * slot_words),
                        bptr.add(ord[1] * slot_words),
                        bptr.add(ord[2] * slot_words),
                    ];
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
                        for chunk in 0..chunks {
                            let col = chunk * NR;
                            let cols = NR.min(p.ow - col);
                            if IS_S2 {
                                k_loop_s2(p.groups, a_panel, slots, group_stride, ev_w, col, &mut acc);
                            } else {
                                k_loop_s1(p.groups, a_panel, slots, group_stride, col, &mut acc);
                            }
                            let res_ptr = if HAS_RES {
                                if cols == NR {
                                    res_plane.add(row_off + col)
                                } else {
                                    resw = [0u32; NR];
                                    std::ptr::copy_nonoverlapping(
                                        res_plane.add(row_off + col),
                                        resw.as_mut_ptr(),
                                        cols,
                                    );
                                    resw.as_ptr()
                                }
                            } else {
                                std::ptr::null()
                            };
                            if OUT_U8 {
                                if cols == NR {
                                    apply_epi_aligned::<SILU, HAS_RES, true>(
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
                                        q_plane.add(row_off + col),
                                        &mut fv,
                                    );
                                } else {
                                    apply_epi_aligned::<SILU, HAS_RES, true>(
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
                                        qw.as_mut_ptr(),
                                        &mut fv,
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        qw.as_ptr(),
                                        q_plane.add(row_off + col),
                                        cols,
                                    );
                                }
                            } else {
                                apply_epi_aligned::<SILU, HAS_RES, false>(
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
                                    std::ptr::null_mut(),
                                    &mut fv,
                                );
                                for (rr, row) in fv.iter().enumerate() {
                                    std::ptr::copy_nonoverlapping(
                                        row.as_ptr(),
                                        f_base.add(rr * ohw + row_off + col),
                                        cols,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            pub(super) unsafe fn run_row<const IS_S2: bool>(r: &RowArgs<'_>) {
                let has_res = r.p.res_view.is_some();
                // SAFETY: ISA checked by caller.
                unsafe {
                    match (r.p.silu, has_res, r.p.is_u8) {
                        (true, false, true) => compute_row::<IS_S2, true, false, true>(r),
                        (false, false, true) => compute_row::<IS_S2, false, false, true>(r),
                        (true, true, true) => compute_row::<IS_S2, true, true, true>(r),
                        (false, true, true) => compute_row::<IS_S2, false, true, true>(r),
                        (true, false, false) => compute_row::<IS_S2, true, false, false>(r),
                        (false, false, false) => compute_row::<IS_S2, false, false, false>(r),
                        (true, true, false) => compute_row::<IS_S2, true, true, false>(r),
                        (false, true, false) => compute_row::<IS_S2, false, true, false>(r),
                    }
                }
            }
        }
    };

    (@dp avxvnni, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_avx_epi32($c00, $b0, w0);
        $c01 = _mm256_dpbusd_avx_epi32($c01, $b1, w0);
        $c02 = _mm256_dpbusd_avx_epi32($c02, $b2, w0);
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_avx_epi32($c10, $b0, w1);
        $c11 = _mm256_dpbusd_avx_epi32($c11, $b1, w1);
        $c12 = _mm256_dpbusd_avx_epi32($c12, $b2, w1);
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_avx_epi32($c20, $b0, w2);
        $c21 = _mm256_dpbusd_avx_epi32($c21, $b1, w2);
        $c22 = _mm256_dpbusd_avx_epi32($c22, $b2, w2);
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_avx_epi32($c30, $b0, w3);
        $c31 = _mm256_dpbusd_avx_epi32($c31, $b1, w3);
        $c32 = _mm256_dpbusd_avx_epi32($c32, $b2, w3);
    };

    (@dp avx512vnni, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
        $c00:ident, $c01:ident, $c02:ident,
        $c10:ident, $c11:ident, $c12:ident,
        $c20:ident, $c21:ident, $c22:ident,
        $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm256_set1_epi32(*$ap);
        $c00 = _mm256_dpbusd_epi32($c00, $b0, w0);
        $c01 = _mm256_dpbusd_epi32($c01, $b1, w0);
        $c02 = _mm256_dpbusd_epi32($c02, $b2, w0);
        let w1 = _mm256_set1_epi32(*$ap.add(1));
        $c10 = _mm256_dpbusd_epi32($c10, $b0, w1);
        $c11 = _mm256_dpbusd_epi32($c11, $b1, w1);
        $c12 = _mm256_dpbusd_epi32($c12, $b2, w1);
        let w2 = _mm256_set1_epi32(*$ap.add(2));
        $c20 = _mm256_dpbusd_epi32($c20, $b0, w2);
        $c21 = _mm256_dpbusd_epi32($c21, $b1, w2);
        $c22 = _mm256_dpbusd_epi32($c22, $b2, w2);
        let w3 = _mm256_set1_epi32(*$ap.add(3));
        $c30 = _mm256_dpbusd_epi32($c30, $b0, w3);
        $c31 = _mm256_dpbusd_epi32($c31, $b1, w3);
        $c32 = _mm256_dpbusd_epi32($c32, $b2, w3);
    };

    (@dp avx2, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
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
        col_step!($b1, $c01, $c11, $c21, $c31);
        col_step!($b2, $c02, $c12, $c22, $c32);
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
    let og_blk = if total_rbs < want && og > 1 && macs / total_rbs >= min_macs.saturating_mul(6) {
        // Cap duplicate row-band packing across output channel blocks at 4× so small `oh` (e.g. rect 640×384)
        // never re-packs the same input rows excessively.
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
    let chunks = p.ow.div_ceil(NR);
    let Some(chunks_nr) = chunks.checked_mul(NR) else { return false };
    let (group_stride, ev_w) = if stride == 2 {
        let ev_w = chunks_nr;
        let Some(od_w) = chunks_nr.checked_add(1) else { return false };
        let Some(ch_stride) = ev_w.checked_add(od_w).and_then(|v| v.checked_next_multiple_of(8)) else {
            return false;
        };
        (ch_stride, ev_w)
    } else {
        let Some(slot_w) = chunks_nr.checked_add(2).and_then(|v| v.checked_next_multiple_of(8)) else {
            return false;
        };
        (slot_w, 0)
    };
    let Some(slot_words) = p.groups.checked_mul(group_stride) else { return false };
    let Some(scratch_words) = slot_words.checked_mul(3) else { return false };
    if scratch_words > MAX_SCRATCH_WORDS {
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
    let (row_blk, og_blk) = plan_3x3_strip(p.n, p.oh, p.ow, p.groups, p.og, is_s2, (threads, min_macs));
    let rbs = p.oh.div_ceil(row_blk);
    let obs = p.og.div_ceil(og_blk);
    let tasks = p.n * rbs * obs;
    let run_task = |t: usize| {
        let (img, rest) = (t / (rbs * obs), t % (rbs * obs));
        let (rb, ob) = (rest / obs, rest % obs);
        let oy0 = rb * row_blk;
        let oy1 = (oy0 + row_blk).min(p.oh);
        let ob_start = ob * og_blk;
        let ob_end = (ob_start + og_blk).min(p.og);
        with_scratch_u32(scratch_words, |buf| {
            let mut ord = [0usize, 1, 2];
            for (i, oy) in (oy0..oy1).enumerate() {
                if is_s2 {
                    if i == 0 {
                        fill_s2_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            p.ow,
                            ev_w,
                            group_stride,
                            (2 * oy) as isize - 1,
                            &mut buf[0..slot_words],
                        );
                        fill_s2_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            p.ow,
                            ev_w,
                            group_stride,
                            (2 * oy) as isize,
                            &mut buf[slot_words..2 * slot_words],
                        );
                        fill_s2_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            p.ow,
                            ev_w,
                            group_stride,
                            (2 * oy + 1) as isize,
                            &mut buf[2 * slot_words..3 * slot_words],
                        );
                    } else {
                        ord.rotate_left(2);
                        let (s1, s2) = (ord[1], ord[2]);
                        fill_s2_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            p.ow,
                            ev_w,
                            group_stride,
                            (2 * oy) as isize,
                            &mut buf[s1 * slot_words..(s1 + 1) * slot_words],
                        );
                        fill_s2_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            p.ow,
                            ev_w,
                            group_stride,
                            (2 * oy + 1) as isize,
                            &mut buf[s2 * slot_words..(s2 + 1) * slot_words],
                        );
                    }
                    let row_args =
                        RowArgs { p, img, oy, ob_start, ob_end, group_stride, ev_w, slot_words, ord, buf };
                    // SAFETY: ISA checked by caller; `(img, oy, ob_start..ob_end)` is disjointly owned.
                    unsafe {
                        match isa {
                            FusedIsa::AvxVnni => vnni_3x3::run_row::<true>(&row_args),
                            FusedIsa::Avx512Vnni => avx512vnni_3x3::run_row::<true>(&row_args),
                            FusedIsa::Avx2 => avx2_3x3::run_row::<true>(&row_args),
                        }
                    }
                } else {
                    if i == 0 {
                        fill_s1_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            group_stride,
                            oy as isize - 1,
                            &mut buf[0..slot_words],
                        );
                        fill_s1_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            group_stride,
                            oy as isize,
                            &mut buf[slot_words..2 * slot_words],
                        );
                        fill_s1_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            group_stride,
                            oy as isize + 1,
                            &mut buf[2 * slot_words..3 * slot_words],
                        );
                    } else {
                        ord.rotate_left(1);
                        let s2 = ord[2];
                        fill_s1_row(
                            p.xs,
                            gsrcs,
                            img,
                            p.h,
                            p.w,
                            group_stride,
                            oy as isize + 1,
                            &mut buf[s2 * slot_words..(s2 + 1) * slot_words],
                        );
                    }
                    let row_args =
                        RowArgs { p, img, oy, ob_start, ob_end, group_stride, ev_w: 0, slot_words, ord, buf };
                    // SAFETY: ISA checked by caller; `(img, oy, ob_start..ob_end)` is disjointly owned.
                    unsafe {
                        match isa {
                            FusedIsa::AvxVnni => vnni_3x3::run_row::<false>(&row_args),
                            FusedIsa::Avx512Vnni => avx512vnni_3x3::run_row::<false>(&row_args),
                            FusedIsa::Avx2 => avx2_3x3::run_row::<false>(&row_args),
                        }
                    }
                }
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
