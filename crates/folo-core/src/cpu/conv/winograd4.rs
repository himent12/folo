//! Well-conditioned Winograd F(4x4, 3x3) forward for dense 3x3 stride-1 inference convs: each 4x4 output tile
//! is `Aᵀ [Σ_c U_c ⊙ V_c] A` with `U = G g Gᵀ` (weights transformed once in `f64`, rounded to `f32`) and
//! `V = Bᵀ d B` (6x6 input patch), reducing 144 multiplies per 4x4 tile and channel pair to 36 (4.0× FLOP
//! reduction vs direct, 1.78× vs F(2x2, 3x3)).
//!
//! Uses interpolation points `(0, 1/2, -1/2, 1, -1, ∞)` rather than `(0, ±1, ±2, ∞)`: every multiplier in
//! the factored `Bᵀ` and `Aᵀ` butterflies is an exact power of two (`±1, ±1/2, ±1/4, ±1/8`; unfactored `Bᵀ`
//! has `-5/4` entries), and the worst-case 1D error factor `max_k Σ_j |Aᵀ[k,j]| · ||Bᵀ[j]||_∞ · ||G[j]||_∞`
//! is `11.67` (`136.1` in 2D), keeping `f32` error tightly bounded (`≤ (C + 16) · 2^-23 · 140 · Σ_c D_c · G_c`,
//! where `D_c = Σ|d|` over the 6x6 patch and `G_c = Σ|g|` over the 3x3 kernel) while maintaining zero AP loss.
//!
//! Input and output transforms live in [`transform`] with explicit 8-tile AVX2/FMA + 4-tile SSE/FMA paths
//! bit-exact with the scalar path; results are bitwise identical across thread counts and batches.

mod transform;

use super::winograd::winograd_shape;
use super::{ChannelEpilogue, ConvGeom, ConvInput, for_each_image};
use crate::cpu::elementwise::SendPtr;
use crate::cpu::gemm::{MR, NR, gemm_rows_serial, with_scratch};
use crate::cpu::par_for;
use transform::{input_transform, output_transform};
#[cfg(test)]
use transform::{input_transform_scalar, output_transform_scalar};

/// Tiles a segment of one tile row is processed in at most (stack buffers stay in L1).
pub(super) const SEG: usize = 64;

/// Floats of `V` plus `M` a block aims to keep (about 1 MiB, inside a core's L2).
const BLOCK_FLOATS: usize = 1 << 18;

/// Input plus output channels an F(4x4, 3x3) conv may have (caps per-thread scratch at `tb = NR = 16`).
const MAX_CHANNELS: usize = 2048;

/// Weight transform matrix `G` (6x3) for interpolation points `(0, 1/2, -1/2, 1, -1, ∞)`.
const G4: [[f64; 3]; 6] = [
    [4.0, 0.0, 0.0],
    [8.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0],
    [8.0 / 3.0, -4.0 / 3.0, 2.0 / 3.0],
    [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
    [2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0],
    [0.0, 0.0, 1.0],
];

/// Whether the inference forward of `g` takes Winograd F(4x4, 3x3) over F(2x2, 3x3) and direct conv.
pub(crate) fn uses_winograd4(g: &ConvGeom) -> bool {
    let tiles = g.oh.div_ceil(4).saturating_mul(g.ow.div_ceil(4));
    winograd_shape(g)
        && g.oh >= 24
        && g.ow >= 24
        && g.c >= 16
        && g.o >= 8
        && g.c.saturating_add(g.o) <= MAX_CHANNELS
        && ((tiles >= 56 && g.c.saturating_mul(g.o) >= 2048) || tiles >= 200)
}

#[cfg(test)]
thread_local! {
    static CALLS4: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// `[transform_weights4 calls, winograd4_forward calls]` made on this thread so far.
#[cfg(test)]
pub(crate) fn calls4() -> [usize; 2] {
    CALLS4.get()
}

#[cfg(test)]
fn count4(i: usize) {
    CALLS4.with(|c| {
        let mut v = c.get();
        v[i] += 1;
        c.set(v);
    });
}

/// `U = G g Gᵀ` of every `(o, c)` 3x3 kernel of the OIHW weights `wt`, laid out `[36][o][c]`, computed in `f64`
/// and rounded once to `f32`.
pub(crate) fn transform_weights4(wt: &[f32], o: usize, c: usize) -> Vec<f32> {
    #[cfg(test)]
    {
        super::winograd::count_transform();
        count4(0);
    }
    let oc = o.checked_mul(c).expect("winograd4: o * c overflow");
    let wt_len = oc.checked_mul(9).expect("winograd4: weight length overflow");
    let u_len = oc.checked_mul(36).expect("winograd4: transformed weight length overflow");
    assert_eq!(wt.len(), wt_len, "winograd4: weights are not O x C x 3 x 3");
    let mut u = vec![0f32; u_len];
    for (k, g) in wt.as_chunks::<9>().0.iter().enumerate() {
        let g: [f64; 9] = std::array::from_fn(|i| g[i] as f64);
        let mut t = [[0f64; 3]; 6];
        for i in 0..6 {
            let [a0, a1, a2] = G4[i];
            for j in 0..3 {
                t[i][j] = a0 * g[j] + a1 * g[3 + j] + a2 * g[6 + j];
            }
        }
        for i in 0..6 {
            let [r0, r1, r2] = t[i];
            for j in 0..6 {
                let [b0, b1, b2] = G4[j];
                u[(i * 6 + j) * oc + k] = (b0 * r0 + b1 * r1 + b2 * r2) as f32;
            }
        }
    }
    u
}

/// Tiles per block (multiple of `NR`), balancing L2 residency of `V` and `M` with parallelism.
fn block_tiles(c: usize, o: usize, tiles: usize, threads: usize) -> usize {
    let cache = (BLOCK_FLOATS / (36 * (c + o))).max(NR) / NR * NR;
    let spread = tiles.div_ceil(threads.max(1)).div_ceil(NR).max(1) * NR;
    cache.min(spread).clamp(NR, 4 * NR)
}

#[derive(Clone, Copy)]
pub(super) enum Tiles<'a> {
    All(usize),
    List(&'a [u32]),
}

impl Tiles<'_> {
    fn len(&self) -> usize {
        match self {
            Tiles::All(n) => *n,
            Tiles::List(l) => l.len(),
        }
    }

    /// Runs of consecutive tiles within one tile row, at most [`SEG`] long: `(ty, tx0, len, pos)`.
    pub(super) fn segments(
        self,
        t0: usize,
        t1: usize,
        tw: usize,
    ) -> impl Iterator<Item = (usize, usize, usize, usize)> {
        let mut t = t0;
        std::iter::from_fn(move || {
            if t >= t1 {
                return None;
            }
            let (tile, len) = match self {
                Tiles::All(_) => (t, (tw - t % tw).min(t1 - t).min(SEG)),
                Tiles::List(l) => {
                    let first = l[t] as usize;
                    let room = (tw - first % tw).min(t1 - t).min(SEG);
                    let len = (1..room).find(|&k| l[t + k] as usize != first + k).unwrap_or(room);
                    (first, len)
                }
            };
            let seg = (tile / tw, tile % tw, len, t - t0);
            t += len;
            Some(seg)
        })
    }
}

/// Forward conv2d via Winograd F(4x4, 3x3) into `out` (`[N, O, OH, OW]`, overwritten) with transformed weights `u`.
pub(crate) fn winograd4_forward(
    x: ConvInput<'_>,
    u: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    winograd4_run(x, u, out, g, epi, None);
}

/// [`winograd4_forward`] of only the 4x4 output tiles `tiles[img]` (`ty · ⌈OW/4⌉ + tx`, ascending and distinct):
/// bit for bit [`winograd4_forward`] on those tiles, leaving every other element of `out` untouched.
pub(crate) fn winograd4_forward_at(
    x: ConvInput<'_>,
    u: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    tiles: &[Vec<u32>],
) {
    assert_eq!(tiles.len(), g.n, "winograd4: one tile list per image");
    let all = g.oh.div_ceil(4) * g.ow.div_ceil(4);
    assert!(
        tiles
            .iter()
            .all(|l| l.windows(2).all(|w| w[0] < w[1]) && l.last().is_none_or(|&t| (t as usize) < all)),
        "winograd4: tile list not strictly ascending and within bounds"
    );
    winograd4_run(x, u, out, g, epi, Some(tiles));
}

fn winograd4_run(
    x: ConvInput<'_>,
    u: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    lists: Option<&[Vec<u32>]>,
) {
    debug_assert!(winograd_shape(g));
    #[cfg(test)]
    {
        super::winograd::count_forward();
        count4(1);
    }
    let (c, o) = (g.c, g.o);
    let u_len = o.checked_mul(c).and_then(|v| v.checked_mul(36)).expect("winograd4: weight size overflow");
    assert_eq!(u.len(), u_len, "winograd4: transformed weights are not 36 x O x C");
    let out_len =
        g.n.checked_mul(o).and_then(|v| v.checked_mul(g.ohw())).expect("winograd4: output overflow");
    assert!(out.len() >= out_len, "winograd4: output buffer too short");
    let all = g.oh.div_ceil(4) * g.ow.div_ceil(4);
    let hw = g.h * g.w;
    for_each_image(out, o * g.ohw(), g.n, 0, |img, out_img, threads| {
        let ts = lists.map_or(Tiles::All(all), |l| Tiles::List(&l[img]));
        let tiles = ts.len();
        if tiles == 0 {
            return;
        }
        let planes = x.planes(img, hw);
        let tb = block_tiles(c, o, tiles, threads);
        let blocks = tiles.div_ceil(tb);
        let span = |b: usize| (b * tb, ((b + 1) * tb).min(tiles));
        let optr = SendPtr(out_img.as_mut_ptr());
        let optr = &optr;
        if threads > 1 && 2 * blocks <= threads {
            // Two-pass path for moderate tile grids (e.g. 40x40 on 28 threads): Pass 1 splits input transforms
            // by C into `vall`; Pass 2 fuses the 36 GEMMs for `o0..o1` (into L1/L2 thread-local scratch `m`)
            // directly with `output_transform`, eliminating a third barrier and global `M` traffic.
            let per_block = threads.div_ceil(blocks);
            let cstep = c.div_ceil(per_block.min(c.div_ceil(8)));
            let cparts = c.div_ceil(cstep);
            let ostep = o.div_ceil(per_block.min(o.div_ceil(MR))).div_ceil(MR) * MR;
            let oparts = o.div_ceil(ostep);
            let vlen = 36 * c * tb;
            let mut vall = crate::cpu::alloc_uninit(blocks * vlen);
            let vptr = SendPtr(vall.as_mut_ptr());
            let vptr = &vptr;
            par_for(blocks * cparts, &|q| {
                let (b, part) = (q / cparts, q % cparts);
                let (t0, t1) = span(b);
                let (ci0, ci1) = (part * cstep, ((part + 1) * cstep).min(c));
                // SAFETY: block b's V is at `b * vlen`; this task owns rows `(xi, ci0..ci1)`.
                unsafe { input_transform(&planes[ci0..ci1], ci0, g, ts, t0, t1, tb, vptr.0.add(b * vlen)) };
            });
            let vall_ref = &vall;
            par_for(blocks * oparts, &|q| {
                let (b, part) = (q / oparts, q % oparts);
                let (t0, t1) = span(b);
                let (o0, o1) = (part * ostep, ((part + 1) * ostep).min(o));
                let on = o1 - o0;
                let v = &vall_ref[b * vlen..(b + 1) * vlen];
                with_scratch(36 * on * tb, |m| {
                    let m = &mut m[..36 * on * tb];
                    let mut rows = Vec::with_capacity(c);
                    for xi in 0..36 {
                        rows.clear();
                        rows.extend((0..c).map(|ci| v[(xi * c + ci) * tb..].as_ptr()));
                        let a = &u[(xi * o + o0) * c..(xi * o + o1) * c];
                        // SAFETY: each row of `v` holds `tb` floats; `m` has `36 * on * tb` floats.
                        unsafe {
                            gemm_rows_serial(
                                on,
                                c,
                                a,
                                c,
                                &rows,
                                tb,
                                &mut m[xi * on * tb..(xi + 1) * on * tb],
                                tb,
                            );
                        }
                    }
                    // SAFETY: each `(b, o0..o1)` owns disjoint output elements.
                    unsafe { output_transform(m, on, o0..o1, g, epi, ts, t0, t1, tb, optr.0) };
                });
            });
            crate::cpu::recycle(vall);
            return;
        }
        let run = |b: usize| {
            let (t0, t1) = span(b);
            with_scratch(36 * (c + o) * tb, |buf| {
                let (v, m) = buf[..36 * (c + o) * tb].split_at_mut(36 * c * tb);
                // SAFETY: `v` is this block's entire V.
                unsafe { input_transform(&planes, 0, g, ts, t0, t1, tb, v.as_mut_ptr()) };
                let mut rows = Vec::with_capacity(c);
                for xi in 0..36 {
                    rows.clear();
                    rows.extend((0..c).map(|ci| v[(xi * c + ci) * tb..].as_ptr()));
                    let a = &u[xi * o * c..(xi + 1) * o * c];
                    // SAFETY: each row of `v` holds `tb` floats.
                    unsafe {
                        gemm_rows_serial(o, c, a, c, &rows, tb, &mut m[xi * o * tb..(xi + 1) * o * tb], tb)
                    };
                }
                // SAFETY: blocks own disjoint output tiles.
                unsafe { output_transform(m, o, 0..o, g, epi, ts, t0, t1, tb, optr.0) };
            });
        };
        if threads > 1 && blocks > 1 {
            par_for(blocks, &run);
        } else {
            (0..blocks).for_each(run);
        }
    });
}

/// Scalar-only forward for testing exact AVX2 == scalar bit parity.
#[cfg(test)]
pub(crate) fn winograd4_forward_scalar(
    x: ConvInput<'_>,
    u: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let (c, o, hw, all) = (g.c, g.o, g.h * g.w, g.oh.div_ceil(4) * g.ow.div_ceil(4));
    let ts = Tiles::All(all);
    let tb = block_tiles(c, o, all, 1);
    for (img, out_img) in out[..g.n * o * g.ohw()].chunks_exact_mut(o * g.ohw()).enumerate() {
        let planes = x.planes(img, hw);
        let mut v = vec![0f32; 36 * c * tb];
        let mut m = vec![0f32; 36 * o * tb];
        for b in 0..all.div_ceil(tb) {
            let (t0, t1) = (b * tb, ((b + 1) * tb).min(all));
            // SAFETY: single-threaded exclusive buffers.
            unsafe {
                input_transform_scalar(&planes, 0, g, ts, t0, t1, tb, v.as_mut_ptr());
                let mut rows = Vec::with_capacity(c);
                for xi in 0..36 {
                    rows.clear();
                    rows.extend((0..c).map(|ci| v[(xi * c + ci) * tb..].as_ptr()));
                    let a = &u[xi * o * c..(xi + 1) * o * c];
                    gemm_rows_serial(o, c, a, c, &rows, tb, &mut m[xi * o * tb..(xi + 1) * o * tb], tb);
                }
                output_transform_scalar(&m, o, 0..o, g, epi, ts, t0, t1, tb, out_img.as_mut_ptr());
            }
        }
    }
}
