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
    let mut srcs = Vec::with_capacity(groups);
    for x in xs {
        let base = x.offset + img * x.img_stride;
        srcs.extend((0..x.c / 4).map(|g| &x.data[base + g * hw..base + (g + 1) * hw]));
    }
    let fill_rows = |g: usize, dst: &mut [u32], i0: usize, i1: usize| {
        let src = srcs[g];
        let zw = u32::from_le_bytes([
            cv.zeros[4 * g],
            cv.zeros[4 * g + 1],
            cv.zeros[4 * g + 2],
            cv.zeros[4 * g + 3],
        ]);
        if s == 1 {
            // Single phase `(0, 0)`: branchless row copies + padding fills (10× faster than per-pixel bounds).
            let d = &mut dst[..plane];
            let top_end = p.min(i1);
            if i0 < top_end {
                d[i0 * wq..top_end * wq].fill(zw);
            }
            let mid_start = i0.max(p);
            let mid_end = i1.min(p + h);
            let lp = p.min(wq);
            let copy_len = w.min(wq - lp);
            for i in mid_start..mid_end {
                let y = i - p;
                let srow = &src[y * w..(y + 1) * w];
                let drow = &mut d[i * wq..(i + 1) * wq];
                drow[..lp].fill(zw);
                drow[lp..lp + copy_len].copy_from_slice(&srow[..copy_len]);
                drow[lp + copy_len..].fill(zw);
            }
            let bot_start = i0.max((p + h).min(hq));
            if bot_start < i1 {
                d[bot_start * wq..i1 * wq].fill(zw);
            }
            return;
        }
        if s == 2 && phases.len() == 4 {
            // All 4 stride-2 phases: read `src` top-to-bottom in one sequential pass (`a = 0` then `a = 1` per
            // output row `i`), dealing both column phases with AVX2 de-interleaving.
            let p_off = [slot[0] * plane, slot[1] * plane, slot[2] * plane, slot[3] * plane];
            for i in i0..i1 {
                for a in 0..2 {
                    let (p0_off, p1_off) = (p_off[a * 2], p_off[a * 2 + 1]);
                    let y = (2 * i + a) as isize - p as isize;
                    if y < 0 || y >= h as isize {
                        dst[p0_off + i * wq..p0_off + (i + 1) * wq].fill(zw);
                        dst[p1_off + i * wq..p1_off + (i + 1) * wq].fill(zw);
                        continue;
                    }
                    let srow = &src[y as usize * w..(y as usize + 1) * w];
                    let (d0, d1) = if p0_off < p1_off {
                        let (left, right) = dst.split_at_mut(p1_off);
                        (&mut left[p0_off + i * wq..p0_off + (i + 1) * wq], &mut right[i * wq..(i + 1) * wq])
                    } else {
                        let (left, right) = dst.split_at_mut(p0_off);
                        (&mut right[i * wq..(i + 1) * wq], &mut left[p1_off + i * wq..p1_off + (i + 1) * wq])
                    };
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
            return;
        }
        for (si, &ph) in phases.iter().enumerate() {
            let (a, b) = (ph / s, ph % s);
            let d = &mut dst[si * plane..(si + 1) * plane];
            for i in i0..i1 {
                let drow = &mut d[i * wq..(i + 1) * wq];
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
            // fan out across all threads instead of leaving most cores idle.
            let rb = threads.div_ceil(groups).min(hq / 8).max(1);
            let row_step = hq.div_ceil(rb);
            let bands = hq.div_ceil(row_step);
            let bptr = OutPtr(body.as_mut_ptr());
            let bptr = &bptr;
            crate::cpu::par_for(groups * bands, &|t| {
                let (g, b) = (t / bands, t % bands);
                let i0 = b * row_step;
                let i1 = (i0 + row_step).min(hq);
                // SAFETY: `(g, b)` writes only rows `i0..i1` of group `g`'s phase planes, disjoint from every
                // other `(g', b')` task inside `body`.
                let dst = unsafe { std::slice::from_raw_parts_mut(bptr.at(g * per_g), per_g) };
                fill_rows(g, dst, i0, i1);
            });
        } else {
            crate::cpu::par_chunks(body, per_g, &|g, dst| fill_rows(g, dst, 0, hq));
        }
    } else {
        body.chunks_mut(per_g).enumerate().for_each(|(g, d)| fill_rows(g, d, 0, hq));
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
