//! Input row-pointer preparation (`image_rows`) for int8 convolutions: zero-copy plane pointers for 1×1
//! stride-1 convs, and cache-line-aligned, parallelised zero-point-padded phase copies for general `k × k`
//! / strided convs.

use super::conv::{OutPtr, QConv};
use super::kernel::NR;
use super::{C4, SLACK, words};

/// The B rows of one image: row pointers into the input planes or a padded copy, the column count, and the
/// padded row width.
pub(super) struct Rows {
    pub(super) buf: Vec<u32>,
    pub(super) rows: Vec<*const u32>,
    pub(super) n: usize,
    pub(super) wq: usize,
}

// SAFETY: the row pointers point into `buf` or into input slices that outlive the conv; they are only read.
unsafe impl Sync for Rows {}
// SAFETY: as above.
unsafe impl Send for Rows {}

/// Fill rows `i0..i1` of a single-phase (`s == 1`) padded plane into the disjoint band slice
/// `dst_band[0..(i1 - i0) * wq]`.
#[allow(clippy::too_many_arguments)]
fn fill_s1_band(
    src: &[u32],
    zw: u32,
    h: usize,
    w: usize,
    p: usize,
    hq: usize,
    wq: usize,
    i0: usize,
    i1: usize,
    dst_band: &mut [u32],
) {
    debug_assert_eq!(dst_band.len(), (i1 - i0) * wq);
    let top_end = p.min(i1);
    if i0 < top_end {
        dst_band[..(top_end - i0) * wq].fill(zw);
    }
    let mid_start = i0.max(p);
    let mid_end = i1.min(p + h);
    let lp = p.min(wq);
    let copy_len = w.min(wq - lp);
    for i in mid_start..mid_end {
        let y = i - p;
        let srow = &src[y * w..(y + 1) * w];
        let bi = i - i0;
        let drow = &mut dst_band[bi * wq..(bi + 1) * wq];
        drow[..lp].fill(zw);
        drow[lp..lp + copy_len].copy_from_slice(&srow[..copy_len]);
        drow[lp + copy_len..].fill(zw);
    }
    let bot_start = i0.max((p + h).min(hq));
    if bot_start < i1 {
        dst_band[(bot_start - i0) * wq..].fill(zw);
    }
}

/// Fill rows `i0..i1` of all four `s == 2` phase planes into four strictly disjoint per-phase band slices
/// `bands[a * 2 + b]` (each of length `(i1 - i0) * wq`), reading `src` top-to-bottom in one sequential pass.
#[allow(clippy::too_many_arguments)]
fn fill_s2_band(
    src: &[u32],
    zw: u32,
    h: usize,
    w: usize,
    p: usize,
    wq: usize,
    i0: usize,
    i1: usize,
    mut bands: [&mut [u32]; 4],
) {
    let band_len = (i1 - i0) * wq;
    debug_assert!(bands.iter().all(|b| b.len() == band_len));
    let [b00, b01, b10, b11] = &mut bands;
    for i in i0..i1 {
        let bi = i - i0;
        for (a, (p0, p1)) in [(0usize, (&mut **b00, &mut **b01)), (1usize, (&mut **b10, &mut **b11))] {
            let d0 = &mut p0[bi * wq..(bi + 1) * wq];
            let d1 = &mut p1[bi * wq..(bi + 1) * wq];
            let y = (2 * i + a) as isize - p as isize;
            if y < 0 || y >= h as isize {
                d0.fill(zw);
                d1.fill(zw);
                continue;
            }
            let srow = &src[y as usize * w..(y as usize + 1) * w];
            if p == 1 {
                d0[0] = zw;
                let pairs = (wq - 1).min(w.div_ceil(2));
                super::deinterleave_even_odd(srow, &mut d1[..pairs], &mut d0[1..1 + pairs], zw);
                d0[1 + pairs..].fill(zw);
                for (j, d) in d1.iter_mut().enumerate().skip(pairs) {
                    let x = 2 * j;
                    *d = if x < w { srow[x] } else { zw };
                }
            } else if p == 0 {
                super::deinterleave_even_odd(srow, d0, d1, zw);
            } else {
                for j in 0..wq {
                    let x0 = (2 * j) as isize - p as isize;
                    let x1 = x0 + 1;
                    d0[j] = if x0 < 0 || x0 >= w as isize { zw } else { srow[x0 as usize] };
                    d1[j] = if x1 < 0 || x1 >= w as isize { zw } else { srow[x1 as usize] };
                }
            }
        }
    }
}

pub(super) fn image_rows(xs: &[C4<'_>], img: usize, h: usize, w: usize, cv: &QConv) -> Rows {
    let hw = h * w;
    if cv.pointwise() {
        let mut rows = Vec::with_capacity(cv.ks);
        for x in xs {
            let base = x.offset + img * x.img_stride;
            rows.extend((0..x.c / 4).map(|g| x.data[base + g * hw..].as_ptr()));
        }
        return Rows { buf: Vec::new(), rows, n: hw, wq: w };
    }
    let (s, p, k) = (cv.stride, cv.pad, cv.k);
    let (oh, _) = cv.out_hw(h, w);
    let (hq, wq) = ((h + 2 * p).div_ceil(s), (w + 2 * p).div_ceil(s));
    let plane = hq * wq;
    let mut slot_stack = [usize::MAX; 16];
    let mut slot_heap;
    let slot: &mut [usize] = if s * s <= 16 {
        &mut slot_stack[..s * s]
    } else {
        slot_heap = vec![usize::MAX; s * s];
        &mut slot_heap
    };
    let mut phases_stack = [0usize; 16];
    let mut phases_len = 0usize;
    let mut phases_heap = Vec::new();
    for ky in 0..k {
        for kx in 0..k {
            let ph = (ky % s) * s + kx % s;
            if slot[ph] == usize::MAX {
                if s * s <= 16 {
                    slot[ph] = phases_len;
                    phases_stack[phases_len] = ph;
                    phases_len += 1;
                } else {
                    slot[ph] = phases_heap.len();
                    phases_heap.push(ph);
                }
            }
        }
    }
    let phases: &[usize] = if s * s <= 16 { &phases_stack[..phases_len] } else { &phases_heap };
    // Pad each channel group's phase block to a 64-byte cache line (16 `u32` words) so parallel `fill` workers
    // writing adjacent groups never false-share the boundary cache line (e.g. `22×22` / `42×42` / `82×82` planes
    // all have `plane % 16 == 4`).
    let raw_per_g = phases.len().checked_mul(plane).expect("q8::conv: phase buffer overflow");
    let per_g = raw_per_g.checked_add(15).map(|v| v & !15).expect("q8::conv: phase buffer overflow");
    let groups = cv.c / 4;
    let buf_len = groups
        .checked_mul(per_g)
        .and_then(|v| v.checked_add(wq.checked_mul(k)?))
        .and_then(|v| v.checked_add(NR + SLACK))
        .expect("q8::conv: padded copy size overflow");
    let mut buf = words(buf_len);
    const STACK_GROUPS: usize = 128;
    let mut srcs_stack: [&[u32]; STACK_GROUPS] = [&[]; STACK_GROUPS];
    let mut srcs_heap = Vec::new();
    let srcs: &[&[u32]] = if groups <= STACK_GROUPS {
        let mut gi = 0;
        for x in xs {
            let base = x.offset + img * x.img_stride;
            for g in 0..x.c / 4 {
                srcs_stack[gi] = &x.data[base + g * hw..base + (g + 1) * hw];
                gi += 1;
            }
        }
        &srcs_stack[..groups]
    } else {
        srcs_heap.reserve(groups);
        for x in xs {
            let base = x.offset + img * x.img_stride;
            srcs_heap.extend((0..x.c / 4).map(|g| &x.data[base + g * hw..base + (g + 1) * hw]));
        }
        &srcs_heap
    };
    let zero_word = |g: usize| {
        u32::from_le_bytes([cv.zeros[4 * g], cv.zeros[4 * g + 1], cv.zeros[4 * g + 2], cv.zeros[4 * g + 3]])
    };
    let fill_group = |g: usize, dst: &mut [u32]| {
        let src = srcs[g];
        let zw = zero_word(g);
        if s == 1 {
            fill_s1_band(src, zw, h, w, p, hq, wq, 0, hq, &mut dst[..plane]);
            return;
        }
        if s == 2 && phases.len() == 4 {
            // Split `dst[..4 * plane]` into four disjoint phase slices, then order them by `slot[0..4]`.
            let (p0, rest) = dst[..4 * plane].split_at_mut(plane);
            let (p1, rest) = rest.split_at_mut(plane);
            let (p2, p3) = rest.split_at_mut(plane);
            let mut slots: [Option<&mut [u32]>; 4] = [Some(p0), Some(p1), Some(p2), Some(p3)];
            let bands = [
                slots[slot[0]].take().unwrap(),
                slots[slot[1]].take().unwrap(),
                slots[slot[2]].take().unwrap(),
                slots[slot[3]].take().unwrap(),
            ];
            fill_s2_band(src, zw, h, w, p, wq, 0, hq, bands);
            return;
        }
        for (si, &ph) in phases.iter().enumerate() {
            let (a, b) = (ph / s, ph % s);
            let d = &mut dst[si * plane..(si + 1) * plane];
            for (i, drow) in d.chunks_exact_mut(wq).enumerate() {
                let y = (s * i + a) as isize - p as isize;
                if y < 0 || y >= h as isize {
                    drow.fill(zw);
                    continue;
                }
                let srow = &src[y as usize * w..(y as usize + 1) * w];
                for (j, v) in drow.iter_mut().enumerate() {
                    let x = (s * j + b) as isize - p as isize;
                    *v = if x < 0 || x >= w as isize { zw } else { srow[x as usize] };
                }
            }
        }
    };
    let (body, _) = buf.split_at_mut(groups * per_g);
    let threads = rayon::current_num_threads();
    let par_words = 1 << 14;
    if body.len() >= par_words && threads > 1 {
        if groups < threads && hq >= 16 && (s == 1 || (s == 2 && phases.len() == 4)) {
            // Split each channel group across row bands so few-group high-res layers (e.g. 16->32 k3s2 320x320)
            // fan out across all threads instead of leaving most cores idle. Each task constructs strictly
            // disjoint `&mut [u32]` slices covering only its own `[i0..i1)` rows per phase.
            let rb = threads.div_ceil(groups).min(hq / 8).max(1);
            let row_step = hq.div_ceil(rb);
            let bands = hq.div_ceil(row_step);
            let bptr = OutPtr(body.as_mut_ptr());
            let bptr = &bptr;
            let s2_slots =
                if s == 2 && phases.len() == 4 { [slot[0], slot[1], slot[2], slot[3]] } else { [0; 4] };
            crate::cpu::par_for(groups * bands, &|t| {
                let (g, b) = (t / bands, t % bands);
                let i0 = b * row_step;
                let i1 = (i0 + row_step).min(hq);
                let len = (i1 - i0) * wq;
                let src = srcs[g];
                let zw = zero_word(g);
                if s == 1 {
                    // SAFETY: `(g, b)` exclusively owns `g * per_g + i0 * wq..+len` inside `body`.
                    let band = unsafe { std::slice::from_raw_parts_mut(bptr.at(g * per_g + i0 * wq), len) };
                    fill_s1_band(src, zw, h, w, p, hq, wq, i0, i1, band);
                } else {
                    // SAFETY: `s2_slots` is a permutation of `0..4`, so the 4 phase ranges for `(g, b)` are
                    // pairwise disjoint and disjoint from every other `(g', b')` task inside `body`.
                    let phase_bands = s2_slots.map(|sl| unsafe {
                        std::slice::from_raw_parts_mut(bptr.at(g * per_g + sl * plane + i0 * wq), len)
                    });
                    fill_s2_band(src, zw, h, w, p, wq, i0, i1, phase_bands);
                }
            });
        } else {
            crate::cpu::par_chunks(body, per_g, &fill_group);
        }
    } else {
        body.chunks_mut(per_g).enumerate().for_each(|(g, d)| fill_group(g, d));
    }
    let base = buf.as_ptr();
    let mut rows = Vec::with_capacity(cv.ks);
    for g in 0..groups {
        for ky in 0..k {
            for kx in 0..k {
                let off = g * per_g + slot[(ky % s) * s + kx % s] * plane + (ky / s) * wq + kx / s;
                // SAFETY: `off` is inside `buf`, which `Rows` keeps alive.
                rows.push(unsafe { base.add(off) });
            }
        }
    }
    Rows { buf, rows, n: oh * wq, wq }
}
