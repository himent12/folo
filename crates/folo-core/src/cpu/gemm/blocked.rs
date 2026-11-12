//! The blocked driver: tile planning, per-thread packing scratch, the K split ([`gemm_ex`]) and
//! [`gemm_blocked`], which packs A once per K super-block and B per tile and runs the micro-kernel over them.

use std::cell::RefCell;

use rayon::prelude::*;

use super::arows::arows_kernel;
use super::kernel::kernel;
use super::{ARows, BiasAct, KC, MR, NR, Operand, PACK_BUDGET, PAR_FLOPS};
use crate::cpu::elementwise::SendPtr;

thread_local! {
    static PACK_BUFS: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// Scratch buffers kept per thread. Re-entrant calls (rayon runs other tasks, including other GEMMs, on a thread
/// while one of its tasks waits) take further buffers from the same stack instead of allocating.
const PACK_BUFS_KEPT: usize = 4;

/// Run `f` with a scratch buffer of at least `len` floats, reused across calls on the same thread.
pub(super) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    // The smallest kept buffer that is large enough, else the largest (grown below).
    let mut buf = PACK_BUFS
        .with(|b| {
            let mut b = b.borrow_mut();
            let fit = (0..b.len()).filter(|&i| b[i].len() >= len).min_by_key(|&i| b[i].len());
            let pick = fit.or_else(|| (0..b.len()).max_by_key(|&i| b[i].len()));
            pick.map(|i| b.swap_remove(i))
        })
        .unwrap_or_default();
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    let r = f(&mut buf[..len]);
    PACK_BUFS.with(|b| {
        let mut b = b.borrow_mut();
        b.push(buf);
        if b.len() > PACK_BUFS_KEPT {
            // Drop the smallest.
            let i = (0..b.len()).min_by_key(|&i| b[i].len()).unwrap_or(0);
            b.swap_remove(i);
        }
    });
    r
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Geometry {
    m: usize,
    n: usize,
    pub(super) mc: usize,
    pub(super) nc: usize,
}

impl Geometry {
    pub(super) fn tiles(&self) -> (usize, usize) {
        (self.m.div_ceil(self.mc), self.n.div_ceil(self.nc))
    }
}

fn choose_tiles(m: usize, n: usize, threads: usize) -> Geometry {
    let mut g = Geometry { m, n, mc: 144, nc: 512 };
    let want = 3 * threads;
    loop {
        let (tm, tn) = g.tiles();
        if tm * tn >= want || (g.mc <= 18 && g.nc <= 64) {
            return g;
        }
        let can_n = g.nc > 64 && g.n > g.nc / 2;
        let can_m = g.mc > 18 && g.m > g.mc / 2;
        if can_n && (g.nc >= 2 * g.mc || !can_m) {
            g.nc /= 2;
        } else if can_m {
            g.mc /= 2;
        } else {
            return g;
        }
    }
}

/// Widest column tile: its packed `KC x NC` B block (512 KiB) stays in L2 while the A panels stream past it.
const NC_MAX: usize = 32 * NR;
/// Smallest amount of work (flops) worth a task of its own; below `threads` of these the pool is not filled.
const TASK_FLOPS: usize = 1 << 20;

/// Tile sizes for [`gemm_blocked`]: column tiles first (every task packs the B block of its own columns, so
/// splitting rows packs the same B more than once), rows only when there are too few columns to give every
/// task a tile. Aims at about three tasks per thread so work stealing can even out fast and slow cores, but
/// never at tasks smaller than [`TASK_FLOPS`].
pub(super) fn plan_tiles(m: usize, n: usize, k: usize, threads: usize) -> Geometry {
    let flops = 2 * m * n * k;
    let want = if threads > 1 { (3 * threads).min(flops / TASK_FLOPS).max(1) } else { 1 };
    let panels = n.div_ceil(NR);
    let nc = (panels.div_ceil(want) * NR).min(NC_MAX);
    let tn = n.div_ceil(nc);
    let mc = if tn >= want { m } else { (m.div_ceil(want.div_ceil(tn))).div_ceil(MR) * MR };
    Geometry { m, n, mc: mc.clamp(MR, m.max(MR)), nc }
}

/// Floats of packed A from which the panels are packed in parallel, and about how many each task packs.
const PACK_A_PAR_MIN: usize = 1 << 17;
const PACK_A_TASK: usize = 1 << 15;

/// Pack the A panels (`panel` floats each) of `pa` with `pack(index, panel)`: in parallel, several panels a
/// task, when there is enough of it.
pub(super) fn pack_lhs_panels(
    pa: &mut [f32],
    panel: usize,
    parallel: bool,
    pack: impl Fn((usize, &mut [f32])) + Sync,
) {
    if parallel && pa.len() >= PACK_A_PAR_MIN {
        let per_task = PACK_A_TASK.div_ceil(panel);
        pa.par_chunks_mut(per_task * panel).enumerate().for_each(|(t, chunk)| {
            chunk.chunks_mut(panel).enumerate().for_each(|(p, dst)| pack((t * per_task + p, dst)))
        });
    } else {
        pa.chunks_mut(panel).enumerate().for_each(pack);
    }
}

/// [`gemm`] with a thread budget (`1` = the calling thread only; the tiling is sized for `threads` tasks, so
/// GEMMs that already run side by side can share the pool) and an optional [`BiasAct`] epilogue.
///
/// The result does not depend on `threads`: every element of C is computed by one task in a fixed K order, and
/// the K split (see [`k_split_parts`]) is a function of the shape only.
///
/// [`gemm`]: super::gemm
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_ex<A: Operand, B: Operand>(
    m: usize,
    n: usize,
    k: usize,
    a: &A,
    b: &B,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
    threads: usize,
    epi: Option<BiasAct<'_>>,
) {
    if m == 0 || n == 0 {
        return;
    }
    let needed = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n));
    assert!(ldc >= n && needed.is_some_and(|e| e <= c.len()), "gemm: C buffer too small");
    let finish_rows = |c: &mut [f32]| {
        if let Some(e) = epi {
            for i in 0..m {
                e.apply(i, &mut c[i * ldc..i * ldc + n]);
            }
        }
    };
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        finish_rows(c);
        return;
    }
    let parts = k_split_parts(m, n, k);
    if parts > 1 {
        // Split K: private accumulators, summed in part order. The split depends only on (m, n, k), never on
        // the thread budget, so the result is bitwise the same however it is scheduled.
        let step = k.div_ceil(parts).div_ceil(KC) * KC;
        let part = |p: usize| {
            let k0 = p * step;
            let mut buf = crate::cpu::alloc_uninit(m * n);
            gemm_blocked(m, n, k0..(k0 + step).min(k), a, b, &mut buf, n, false, 1, None);
            buf
        };
        let parts = k.div_ceil(step);
        let partials: Vec<Vec<f32>> = if threads > 1 {
            (0..parts).into_par_iter().map(part).collect()
        } else {
            (0..parts).map(part).collect()
        };
        for i in 0..m {
            let row = &mut c[i * ldc..i * ldc + n];
            if !accumulate {
                row.fill(0.0);
            }
            for p in &partials {
                for (r, &v) in row.iter_mut().zip(&p[i * n..i * n + n]) {
                    *r += v;
                }
            }
        }
        partials.into_iter().for_each(crate::cpu::recycle);
        finish_rows(c);
        return;
    }
    let threads = if 2 * m * n * k < PAR_FLOPS { 1 } else { threads };
    gemm_blocked(m, n, 0..k, a, b, c, ldc, accumulate, threads, epi);
}

/// Number of K parts for a GEMM whose C is too small to keep every thread busy with tiles but whose K is long
/// (conv weight gradients: K = N·OH·OW), or 1. A function of the shape only, for reproducibility.
fn k_split_parts(m: usize, n: usize, k: usize) -> usize {
    if k < 8 * KC || 2 * m * n * k < PAR_FLOPS {
        return 1;
    }
    let (tm, tn) = choose_tiles(m, n, K_SPLIT_THREADS).tiles();
    if tm * tn >= K_SPLIT_THREADS {
        return 1;
    }
    (k / (2 * KC)).clamp(2, K_SPLIT_THREADS)
}

/// The nominal thread count the K-split decision is made for (see [`k_split_parts`]).
const K_SPLIT_THREADS: usize = 32;

/// The 64-byte-aligned start of a scratch buffer of at least `len + 16` floats, as a `len`-float slice.
pub(super) fn aligned(buf: &mut [f32], len: usize) -> &mut [f32] {
    let off = buf.as_ptr().align_offset(64).min(16);
    &mut buf[off..off + len]
}

/// The blocked GEMM over `krange`. A is packed once per K super-block (as much K as fits [`PACK_BUDGET`]) and
/// shared by all tasks. Each task owns a tile of C and, for every `KC` block, packs the B panels of its own
/// columns into a thread-local, L2-sized buffer, then sweeps the A panels past each B panel (which stays in
/// L1). For conv2d this expands the implicit im2col operand one small block at a time, while it is in cache,
/// instead of materialising the whole unfolded matrix first.
#[allow(clippy::too_many_arguments)]
fn gemm_blocked<A: Operand, B: Operand>(
    m: usize,
    n: usize,
    krange: std::ops::Range<usize>,
    a: &A,
    b: &B,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
    threads: usize,
    epi: Option<BiasAct<'_>>,
) {
    let (kern, akern) = (kernel(), arows_kernel());
    let k = krange.len();
    // A read in place (a row-major A: a conv's weights) is never packed.
    let ain = ARows::of(a, m, krange.end);
    let pm = m.div_ceil(MR);
    let reg = epi.filter(BiasAct::in_kernel);
    // K super-block: as many KC blocks of packed A as fit the pack budget (all of K for A in place).
    let blocks_fit = (PACK_BUDGET / (pm * MR * KC)).max(1);
    let ks = if ain.is_some() { k } else { (blocks_fit * KC).min(k.div_ceil(KC) * KC) };
    let g = plan_tiles(m, n, k, threads);
    let (tm, tn) = g.tiles();
    let parallel = threads > 1 && tm * tn > 1;
    let cptr = SendPtr(c.as_mut_ptr());
    let a_len = if ain.is_some() { 0 } else { pm * MR * ks };
    with_scratch(a_len + 16, |abuf| {
        let pa_all = aligned(abuf, a_len);
        let mut s0 = krange.start;
        while s0 < krange.end {
            let ksz = ks.min(krange.end - s0);
            let last = s0 + ksz == krange.end;
            let first = s0 == krange.start;
            let pa = &mut pa_all[..a_len.min(pm * MR * ksz)];
            if ain.is_none() {
                let pack_a = |(p, dst): (usize, &mut [f32])| {
                    a.pack(p * MR, MR.min(m - p * MR), s0, ksz, MR, dst);
                };
                pack_lhs_panels(pa, MR * ksz, parallel, pack_a);
            }
            let pa = &*pa;
            let tile = |t: usize| {
                let (ti, tj) = (t / tn, t % tn);
                let (i0, i1) = (ti * g.mc, ((ti + 1) * g.mc).min(m));
                let (j0, j1) = (tj * g.nc, ((tj + 1) * g.nc).min(n));
                let (cp, ain) = (&cptr, &ain);
                let panels = (j1 - j0).div_ceil(NR);
                let b_len = panels * NR * KC.min(ksz);
                with_scratch(b_len + 16, |bbuf| {
                    let pb = aligned(bbuf, b_len);
                    let mut kb = 0;
                    while kb < ksz {
                        let kc = KC.min(ksz - kb);
                        b.pack_panels(j0, j1 - j0, s0 + kb, kc, pb);
                        let acc = accumulate || !first || kb != 0;
                        let finished = last && kb + kc == ksz;
                        for q in 0..panels {
                            let j = j0 + q * NR;
                            let nr = NR.min(j1 - j);
                            let bpanel = &pb[q * NR * kc..(q + 1) * NR * kc];
                            let mut i = i0;
                            while i < i1 {
                                let mr = MR.min(i1 - i);
                                let re = reg.filter(|_| finished).map(|e| e.kernel_rows(i, mr));
                                let cblock = cp.0.wrapping_add(i * ldc + j);
                                // SAFETY: the panels hold kc*MR / kc*NR floats (packed above), or A's rows
                                // i..i + mr hold K elements s0 + kb..s0 + kb + kc in place; the C block
                                // [i..i+mr, j..j+nr] lies inside `c` (asserted by `gemm_ex`) and belongs to
                                // this tile only, so no other task touches it.
                                unsafe {
                                    if let Some(ar) = ain {
                                        let ap = ar.panel(i, mr, s0 + kb);
                                        akern(
                                            kc,
                                            &ap,
                                            bpanel.as_ptr(),
                                            cblock,
                                            ldc,
                                            mr,
                                            nr,
                                            acc,
                                            re.as_ref(),
                                        );
                                    } else {
                                        let apanel = &pa[(i / MR) * MR * ksz + kb * MR..];
                                        debug_assert!(apanel.len() >= kc * MR);
                                        kern(
                                            kc,
                                            apanel.as_ptr(),
                                            bpanel.as_ptr(),
                                            cblock,
                                            ldc,
                                            mr,
                                            nr,
                                            acc,
                                            re.as_ref(),
                                        );
                                    }
                                };
                                i += MR;
                            }
                        }
                        kb += KC;
                    }
                });
                if let (true, Some(e), None) = (last, epi, reg) {
                    for i in i0..i1 {
                        // SAFETY: row segment [i, j0..j1] is inside `c` and owned by this tile (see above); no
                        // other reference to it exists while `e` runs.
                        let row = unsafe { std::slice::from_raw_parts_mut(cp.0.add(i * ldc + j0), j1 - j0) };
                        e.apply(i, row);
                    }
                }
            };
            if parallel {
                (0..tm * tn).into_par_iter().for_each(tile);
            } else {
                (0..tm * tn).for_each(tile);
            }
            s0 += ksz;
        }
    });
}
