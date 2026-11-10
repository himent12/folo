//! [`ShiftedRows`]: a conv input zero-padded and split into its stride phases once, so every `(channel, ky, kx)`
//! row of the GEMM is a contiguous run read in place (the forward's B, the weight gradient's A).

use rayon::prelude::*;

use super::ConvGeom;
use crate::cpu::gemm::{ColumnMap, NR};

/// The shifted-rows form of one image's (one group's) conv input, the B operand of [`gemm_rows`]: the input,
/// zero-padded and split into its `sy x sx` stride phases (`T[ci][a][b][i][j] = x_pad[sy·i + a][sx·j + b]`),
/// so that kernel tap `(ky, kx)` of channel `ci` is the *contiguous* run of phase
/// `(ky·dy mod sy, kx·dx mod sx)` starting at row `ky·dy / sy`, column `kx·dx / sx`: output pixel `(oy, ox)`
/// is column `oy·wq + ox` of every run. The `wq - ow` extra columns per output row are computed and dropped
/// (`ColumnMap`), and no unfolded matrix is ever written: B is read in place.
///
/// Planes are spaced an odd number of cache lines apart: rows of B that are a whole number of 4 KiB pages
/// apart (as the planes of a 160x160 or 80x80 image are) would all compete for the same L1 sets.
pub(super) struct ShiftedRows {
    /// The phase planes, with slack after the last one.
    pub(super) buf: Vec<f32>,
    /// Row `k = (ci, ky, kx)` of B.
    pub(super) rows: Vec<*const f32>,
    /// Columns of B: `oh · wq`.
    pub(super) n: usize,
    pub(super) map: ColumnMap,
}

impl ShiftedRows {
    pub(super) fn new(planes: &[&[f32]], g: &ConvGeom, parallel: bool) -> Self {
        let [sy, sx] = g.p.stride;
        let [ph, pw] = g.p.padding;
        let [dy, dx] = g.p.dilation;
        let (h, w) = (g.h, g.w);
        let (hq, wq) = ((h + 2 * ph).div_ceil(sy), (w + 2 * pw).div_ceil(sx));
        let plane = hq * wq;
        // A plane stride of an odd number of 64-byte lines.
        let lines = plane.div_ceil(16);
        let pstride = (lines | 1) * 16;
        // Only the phases some tap reads (at most kh·kw of them, however large the stride), each with its
        // slot in the per-channel block.
        let mut slot = std::collections::BTreeMap::new();
        for ky in 0..g.kh {
            for kx in 0..g.kw {
                let next = slot.len();
                slot.entry(((ky * dy) % sy, (kx * dx) % sx)).or_insert(next);
            }
        }
        let phases: Vec<(usize, usize, usize)> = slot.iter().map(|(&(a, b), &s)| (a, b, s)).collect();
        let per_ci = phases.len() * pstride;
        // Valid outputs read inside their plane; the dropped columns of the last row may run up to `wq`
        // floats past it (into the next plane, or this slack).
        let mut buf = super::alloc_uninit(planes.len() * per_ci + wq + NR);
        let fill = |(ci, dst): (usize, &mut [f32])| {
            let src = planes[ci];
            // The `pstride - plane` floats after each plane are zeros: the dropped columns of the last output
            // row read into them, and the weight gradient multiplies those by its zero columns (a recycled
            // buffer's NaN there would make 0 · NaN). The gap is not the only place dropped columns read: they
            // wrap into the next row of the plane and, when the gap is shorter than `(kw - 1) · dx`, into the next
            // plane. Those floats are the input itself, so with a non-finite input (an inf or NaN in `x`) the
            // weight gradient can be NaN at neighbouring taps of the same or the previous channel where the
            // naive sum stays finite. That is accepted: the naive path's gradient of the same weight tensor is
            // already non-finite then, and the trainer reacts to non-finite values per tensor, never per element.
            for s in 0..phases.len() {
                dst[s * pstride + plane..(s + 1) * pstride].iter_mut().for_each(|v| *v = 0.0);
            }
            if sx == 1 {
                for &(a, _, s) in &phases {
                    let d = &mut dst[s * pstride..s * pstride + plane];
                    for (i, drow) in d.chunks_exact_mut(wq).enumerate() {
                        let y = (sy * i + a) as isize - ph as isize;
                        if y < 0 || y >= h as isize {
                            drow.fill(0.0);
                            continue;
                        }
                        let srow = &src[y as usize * w..(y as usize + 1) * w];
                        // x = j - pw: zeros, the row, zeros (short runs: plain stores, no memset calls).
                        let (lo, hi) = (pw.min(wq), (pw + w).min(wq));
                        drow[..lo].iter_mut().for_each(|v| *v = 0.0);
                        drow[lo..hi].copy_from_slice(&srow[..hi - lo]);
                        drow[hi..].iter_mut().for_each(|v| *v = 0.0);
                    }
                }
                return;
            }
            // Column phases: each input row is zero-padded once into `padded` (`sx * wq >= pw + w` floats,
            // `padded[x + pw] = row[x]`), then dealt out to the phases of its row parity, `T[b][j] =
            // padded[sx·j + b]`, two at a time for stride 2.
            let mut padded = vec![0f32; sx * wq];
            let mut k = 0;
            while k < phases.len() {
                let a = phases[k].0;
                let group = phases[k..].iter().take_while(|p| p.0 == a).count();
                let bs = &phases[k..k + group];
                k += group;
                for i in 0..hq {
                    let y = (sy * i + a) as isize - ph as isize;
                    if y < 0 || y >= h as isize {
                        for &(_, _, s) in bs {
                            dst[s * pstride + i * wq..][..wq].fill(0.0);
                        }
                        continue;
                    }
                    let srow = &src[y as usize * w..(y as usize + 1) * w];
                    padded[pw..pw + w].copy_from_slice(srow);
                    if let [(_, 0, s0), (_, 1, s1)] = *bs
                        && sx == 2
                    {
                        let (d0, d1) = if s0 < s1 {
                            let (lo, hi) = dst.split_at_mut(s1 * pstride);
                            (&mut lo[s0 * pstride + i * wq..][..wq], &mut hi[i * wq..][..wq])
                        } else {
                            let (lo, hi) = dst.split_at_mut(s0 * pstride);
                            (&mut hi[i * wq..][..wq], &mut lo[s1 * pstride + i * wq..][..wq])
                        };
                        for ((e, o), p) in d0.iter_mut().zip(d1.iter_mut()).zip(padded.as_chunks::<2>().0) {
                            *e = p[0];
                            *o = p[1];
                        }
                    } else {
                        for &(_, b, s) in bs {
                            let d = &mut dst[s * pstride + i * wq..][..wq];
                            for (v, p) in d.iter_mut().zip(padded[b..].iter().step_by(sx)) {
                                *v = *p;
                            }
                        }
                    }
                }
            }
        };
        let (body, slack) = buf.split_at_mut(planes.len() * per_ci);
        slack.fill(0.0);
        // A copy this size takes tens of microseconds: in parallel only when it is large, a few channels a task.
        if parallel && body.len() >= SHIFTED_PAR_MIN {
            let per_task = SHIFTED_TASK.div_ceil(per_ci);
            body.par_chunks_mut(per_task * per_ci).enumerate().for_each(|(t, chunk)| {
                chunk.chunks_mut(per_ci).enumerate().for_each(|(c, dst)| fill((t * per_task + c, dst)))
            });
        } else {
            body.chunks_mut(per_ci).enumerate().for_each(fill);
        }
        // Where each tap's run starts within a channel's block.
        let mut taps = Vec::with_capacity(g.kh * g.kw);
        for ky in 0..g.kh {
            for kx in 0..g.kw {
                let (a, rs) = ((ky * dy) % sy, (ky * dy) / sy);
                let (b, cs) = ((kx * dx) % sx, (kx * dx) / sx);
                taps.push(slot[&(a, b)] * pstride + rs * wq + cs);
            }
        }
        let base = buf.as_ptr();
        let mut rows = Vec::with_capacity(planes.len() * g.kh * g.kw);
        for ci in 0..planes.len() {
            // SAFETY: every offset is inside `buf` (rs < hq, cs < wq), which outlives `rows`.
            rows.extend(taps.iter().map(|&t| unsafe { base.add(ci * per_ci + t) }));
        }
        ShiftedRows { buf, rows, n: g.oh * wq, map: ColumnMap { wp: wq, ow: g.ow } }
    }
}

/// Floats from which [`ShiftedRows::new`] fills in parallel, and about how many each task fills.
const SHIFTED_PAR_MIN: usize = 1 << 16;
const SHIFTED_TASK: usize = 1 << 14;

/// About the floats of one image's (one group's) [`ShiftedRows`] copy: its padded input planes.
pub(super) fn shifted_scratch(g: &ConvGeom) -> usize {
    let [sy, sx] = g.p.stride;
    g.cg() * (g.h + 2 * g.p.padding[0] + sy) * (g.w + 2 * g.p.padding[1] + sx)
}
