//! [`ShiftedRows`]: a conv input zero-padded and split into its stride phases once, so every `(channel, ky, kx)`
//! row of the GEMM is a contiguous run read in place (the forward's B, the weight gradient's A).

use std::cell::RefCell;

use smallvec::SmallVec;

use super::ConvGeom;
use crate::cpu::gemm::{ColumnMap, NR};
use crate::cpu::par_chunks;

/// Maximum per-thread `padded` row scratch retained in TLS (64K floats = 256 KiB).
const SHIFTED_TLS_CAP: usize = 1 << 16;

thread_local! {
    static SHIFTED_PAD_TLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Split `src[0..2 * pairs]` into `even[j] = src[2 * j]` and `odd[j] = src[2 * j + 1]`, using AVX2
/// `vshufps` + `vpermq` when available.
pub(super) fn deinterleave_f32_even_odd(src: &[f32], even: &mut [f32], odd: &mut [f32]) {
    let pairs = even.len();
    assert_eq!(odd.len(), pairs, "deinterleave_f32_even_odd: output length mismatch");
    assert!(src.len() >= 2 * pairs, "deinterleave_f32_even_odd: source too short");
    #[cfg(target_arch = "x86_64")]
    if pairs >= 8 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above and slice lengths were asserted.
        unsafe { deinterleave_f32_even_odd_avx2(src, even, odd) };
        return;
    }
    for ((e, o), p) in even.iter_mut().zip(odd.iter_mut()).zip(src.as_chunks::<2>().0) {
        *e = p[0];
        *o = p[1];
    }
}

/// # Safety
/// The CPU supports AVX2, `even.len() == odd.len()`, and `src.len() >= 2 * even.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn deinterleave_f32_even_odd_avx2(src: &[f32], even: &mut [f32], odd: &mut [f32]) {
    use std::arch::x86_64::*;
    let pairs = even.len();
    let vec_n = pairs / 8 * 8;
    // SAFETY: `j + 8 <= vec_n <= pairs` and `2 * (j + 8) <= 2 * pairs <= src.len()`.
    unsafe {
        for j in (0..vec_n).step_by(8) {
            let sptr = src.as_ptr().add(2 * j);
            let lo = _mm256_loadu_ps(sptr);
            let hi = _mm256_loadu_ps(sptr.add(8));
            let ev = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0x88>(lo, hi),
            )));
            let od = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0xDD>(lo, hi),
            )));
            _mm256_storeu_ps(even.as_mut_ptr().add(j), ev);
            _mm256_storeu_ps(odd.as_mut_ptr().add(j), od);
        }
    }
    for j in vec_n..pairs {
        even[j] = src[2 * j];
        odd[j] = src[2 * j + 1];
    }
}

/// The shifted-rows form of one image's (one group's) conv input, the B operand of [`gemm_rows`]: the input,
/// zero-padded and split into its `sy x sx` stride phases (`T[ci][a][b][i][j] = x_pad[sy·i + a][sx·j + b]`),
/// so that kernel tap `(ky, kx)` of channel `ci` is the *contiguous* run of phase
/// `(ky·dy mod sy, kx·dx mod sx)` starting at row `ky·dy / sy`, column `kx·dx / sx`: output pixel `(oy, ox)`
/// is column `oy·wq + ox` of every run. The `wq - ow` extra columns per output row are computed and dropped
/// (`ColumnMap`), and no unfolded matrix is ever written: B is read in place.
///
/// Planes are spaced an odd number of cache lines apart: rows of B that are a whole number of 4 KiB pages
/// apart (as the planes of a 160x160 or 80x80 image are) would all compete for the same L1 sets.
///
/// [`gemm_rows`]: crate::cpu::gemm::gemm_rows
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
        // slot in the per-channel block. Fast stack table for `sy * sx <= 16` (covers all strides <= 4).
        let mut phases: SmallVec<[(usize, usize, usize); 16]> = SmallVec::new();
        let mut taps: SmallVec<[usize; 16]> = SmallVec::with_capacity(g.kh * g.kw);
        if sy.checked_mul(sx).is_some_and(|p| p <= 16) {
            let mut slot = [usize::MAX; 16];
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let (a, b) = ((ky * dy) % sy, (kx * dx) % sx);
                    let key = a * sx + b;
                    if slot[key] == usize::MAX {
                        slot[key] = phases.len();
                        phases.push((a, b, slot[key]));
                    }
                }
            }
            // Keep row-parity groups `(a, b)` sorted so column phases of the same `a` are adjacent.
            phases.sort_unstable_by_key(|&(a, b, _)| (a, b));
            for (new_s, p) in phases.iter_mut().enumerate() {
                p.2 = new_s;
                slot[p.0 * sx + p.1] = new_s;
            }
            let per_ci = phases.len() * pstride;
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let (a, rs) = ((ky * dy) % sy, (ky * dy) / sy);
                    let (b, cs) = ((kx * dx) % sx, (kx * dx) / sx);
                    taps.push(slot[a * sx + b] * pstride + rs * wq + cs);
                }
            }
            return Self::build(
                planes,
                g,
                parallel,
                (sy, sx, ph, pw, h, w, hq, wq, plane, pstride, per_ci),
                &phases,
                &taps,
            );
        }
        let mut slot = std::collections::BTreeMap::new();
        for ky in 0..g.kh {
            for kx in 0..g.kw {
                let next = slot.len();
                slot.entry(((ky * dy) % sy, (kx * dx) % sx)).or_insert(next);
            }
        }
        phases.extend(slot.iter().map(|(&(a, b), &s)| (a, b, s)));
        let per_ci = phases.len() * pstride;
        for ky in 0..g.kh {
            for kx in 0..g.kw {
                let (a, rs) = ((ky * dy) % sy, (ky * dy) / sy);
                let (b, cs) = ((kx * dx) % sx, (kx * dx) / sx);
                taps.push(slot[&(a, b)] * pstride + rs * wq + cs);
            }
        }
        Self::build(
            planes,
            g,
            parallel,
            (sy, sx, ph, pw, h, w, hq, wq, plane, pstride, per_ci),
            &phases,
            &taps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        planes: &[&[f32]],
        g: &ConvGeom,
        parallel: bool,
        (sy, sx, ph, pw, h, w, hq, wq, plane, pstride, per_ci): (
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
        ),
        phases: &[(usize, usize, usize)],
        taps: &[usize],
    ) -> Self {
        // Valid outputs read inside their plane; the dropped columns of the last row may run up to `wq`
        // floats past it (into the next plane, or this slack).
        let mut buf = crate::cpu::alloc_uninit(planes.len() * per_ci + wq + NR);
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
                for &(a, _, s) in phases {
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
            // Column phases: each input row is zero-padded once into a stack (<= 1024 floats) or TLS scratch
            // row (`sx * wq >= pw + w` floats, `padded[x + pw] = row[x]`), then dealt out to the phases of its
            // row parity, `T[b][j] = padded[sx·j + b]`, via AVX2 even/odd deinterleave for stride 2.
            let pad_len = sx * wq;
            let mut fill_strided = |padded: &mut [f32]| {
                padded.fill(0.0);
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
                            deinterleave_f32_even_odd(padded, d0, d1);
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
            const STACK_PAD: usize = 1024;
            if pad_len <= STACK_PAD {
                let mut stack_buf = [0.0f32; STACK_PAD];
                fill_strided(&mut stack_buf[..pad_len]);
            } else {
                let mut f = Some(fill_strided);
                let ran = SHIFTED_PAD_TLS.try_with(|cell| {
                    if let Ok(mut tls) = cell.try_borrow_mut() {
                        if tls.len() < pad_len {
                            tls.resize(pad_len, 0.0);
                        }
                        (f.take().unwrap())(&mut tls[..pad_len]);
                        if tls.capacity() > SHIFTED_TLS_CAP {
                            *tls = Vec::new();
                        }
                        true
                    } else {
                        false
                    }
                });
                if !matches!(ran, Ok(true)) {
                    let mut heap = vec![0.0f32; pad_len];
                    (f.take().unwrap())(&mut heap);
                }
            }
        };
        let (body, slack) = buf.split_at_mut(planes.len() * per_ci);
        slack.fill(0.0);
        // A copy this size takes tens of microseconds: in parallel only when it is large, a few channels a task.
        if parallel && body.len() >= SHIFTED_PAR_MIN {
            let per_task = SHIFTED_TASK.div_ceil(per_ci);
            par_chunks(body, per_task * per_ci, &|t, chunk| {
                chunk.chunks_mut(per_ci).enumerate().for_each(|(c, dst)| fill((t * per_task + c, dst)))
            });
        } else {
            body.chunks_mut(per_ci).enumerate().for_each(fill);
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
