//! Single-barrier, L1-hot rotating row-strip fused x86-64 driver for single-part `3×3` (`pad = 1`)
//! int8 convolutions (`q8::conv` / `q8::conv_per_channel`):
//! - All `stride = 2, pad = 1` convs (3 rotating `(ev, od)` deinterleaved input row slots per task, `15–31 KiB`
//!   in L1D, reusing `ky = 2 -> ky = 0` across consecutive rows).
//! - Large-input `stride = 1, pad = 1` convs (`input > 384 KiB`, e.g. `64->64 80×80` and `folo-s` high-res layers),
//!   where the global padded copy would otherwise spill across cores through L3 and pay a second rayon barrier.
//!
//! Arithmetic order is bit-for-bit identical to [`super::kernel::kernel_scalar`] + [`super::kernel::Epi::apply`].

#![cfg(target_arch = "x86_64")]

use std::cell::RefCell;

use super::conv::OutPtr;
use super::fused_epi::FusedIsa;
use super::kernel::MR;
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

/// Multi-row batch arguments (`rows_len <= 4` for both `s = 1` and `s = 2`) passed into 256-bit `compute_rows`
/// kernels so each `g`'s weight panel is loaded once and reused across `rows_len` output rows inside L1D.
pub(super) struct BatchRowArgs<'a> {
    pub(super) p: &'a Conv3x3Params<'a>,
    pub(super) img: usize,
    pub(super) oy_start: usize,
    pub(super) rows_len: usize,
    pub(super) ob_start: usize,
    pub(super) ob_end: usize,
    pub(super) group_stride: usize,
    pub(super) ev_w: usize,
    pub(super) slot_words: usize,
    pub(super) ord: [usize; 9],
    pub(super) buf: &'a [u32],
}

/// Source group lookup `(part_index, local_group_index)` and precomputed packed zero word.
#[derive(Clone, Copy)]
struct GroupSrc {
    part: usize,
    local_g: usize,
    zw: u32,
}

mod kernels;
use kernels::{avx2_3x3, avx512vnni_3x3, vnni_3x3};

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

    // Shared packed-strip fast path for small-spatial `s = 1` (`oh <= 24`, `<= 256 KiB` per image, e.g. `20×20`
    // P5 layers): packs once per image (serially below 16K words, in parallel above) and splits compute tasks by
    // output channel group `og` first. Medium/large `40×40` / `80×80` and `s = 2` convs stay on the single-barrier
    // L1-hot rotating strip below.
    let shared_s1_words = (!is_s2 && p.oh <= 24)
        .then(|| p.oh.checked_add(2).and_then(|rows| rows.checked_mul(slot_words)))
        .flatten()
        .filter(|&img_w| {
            img_w <= 65_536 && p.n.checked_mul(img_w).is_some_and(|tot| tot <= MAX_SCRATCH_WORDS)
        });
    if let Some(img_words) = shared_s1_words
        && isa != FusedIsa::Avx512Vnni512
    {
        let padded_rows = p.oh + 2;
        let macs_total =
            p.n.saturating_mul(p.oh)
                .saturating_mul(p.ow)
                .saturating_mul(p.og * 4)
                .saturating_mul(p.groups * 4)
                .saturating_mul(9);
        let serial_floor =
            if threads >= 16 { min_macs.saturating_mul(8) } else { min_macs.saturating_mul(2) };
        let max_tasks = (3 * threads).max(1);
        let want = if threads > 1 && macs_total >= serial_floor {
            max_tasks.min(macs_total / min_macs.max(1)).max(1)
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
        let max_shared_rb = 4usize;
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
                let b_len = max_shared_rb.min(oy1 - oy);
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
    // Choose L1-resident multi-row batch size `max_rb ∈ {1, 2, 3, 4}` for 256-bit ISAs when `ow <= 128` so each
    // weight panel `a_panel` is reused across `max_rb` output rows without spilling the strip out of L1D (48 KiB).
    let max_rb = if isa == FusedIsa::Avx512Vnni512 || p.ow > 128 {
        1
    } else if row_blk >= 4 && (stride * 3 + 3).saturating_mul(slot_words).saturating_mul(4) <= (48 << 10) {
        4
    } else if row_blk >= 3 && (stride * 2 + 3).saturating_mul(slot_words).saturating_mul(4) <= (48 << 10) {
        3
    } else if row_blk >= 2 && (stride + 3).saturating_mul(slot_words).saturating_mul(4) <= (48 << 10) {
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
