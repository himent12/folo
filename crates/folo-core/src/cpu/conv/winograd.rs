//! Winograd F(2x2, 3x3) forward for dense 3x3 stride-1 convs (inference): each 2x2 output tile is
//! `Aᵀ [Σ_c U_c ⊙ V_c] A` with `U = G g Gᵀ` (the weights, transformed once) and `V = Bᵀ d B` (the tile's 4x4
//! input patch), so the 36 multiplies of a tile and channel pair become 16, done as 16 GEMMs (one per element of
//! the 4x4 transform domain) of `O x C` weights by `C x tiles` inputs.
//!
//! Not bit for bit the direct conv: the transforms add and subtract inputs (and halve weights) before the
//! products, so a result differs from the direct sum in the last bits. The error of an output (before bias and
//! activation) is at most `(C + 8) · 2^-23 · 9 · Σ_c D_c · G_c`, with `D_c` the sum of `|d|` over its tile's 4x4
//! input patch in channel `c` (padding counts 0) and `G_c` the sum of `|g|` over its 3x3 kernel from channel `c`:
//! with `u = 2^-24`, an element of `V` is a ±1 sum of 4 patch values in two roundings (error `≤ 2u · D_c`, and
//! `|V| ≤ D_c`); an element of `U` is `G g Gᵀ` (every `|G_ik · G_jl| ≤ 1`, so `|U| ≤ G_c`), rounded once from
//! f64; the GEMM sums `C` products (error `≤ C u Σ_c |U| |V|`); and the output transform adds 9 elements of `M`
//! with signs ±1 in four roundings. First order that is `(C + 7) u · 9 Σ_c D_c G_c`, and the bound's twice-as-large
//! unit covers the second-order terms and the rounding of the bias add (`u · |bias|` besides). It grows with `C`:
//! with every product of one sign nothing cancels (all-ones inputs, 512 channels: 20 · 2^-23 · Σ|terms| of the
//! direct sum, far inside the direct conv's own `K · 2^-23 · Σ|terms|`, `K = 9C`). Tested in `kernel_tests.rs`.
//! Like every rounding bound it assumes no step overflows or underflows: the transforms add up to 4 inputs (and
//! scale weights by up to 9/4 before the products), so inputs within a factor ~4 of `f32::MAX` can overflow to
//! inf (and inf - inf to NaN) where the direct sum stays finite, and products below the normal range
//! (|value| < 2^-126) lose their relative accuracy. Activations of a trained model are many orders of magnitude
//! inside both.
//!
//! The tiles of an image are cut into blocks of `tb` tiles (a multiple of `NR`). A block transforms its input
//! patches into `V` (`16 x C x tb`), runs the 16 GEMMs into `M` (`16 x O x tb`) and transforms `M` into its
//! output tiles, then bias and activation, while `V` and `M` stay in L2.

use rayon::prelude::*;

use super::{ChannelEpilogue, ConvGeom, ConvInput, for_each_image};
use crate::cpu::elementwise::SendPtr;
use crate::cpu::gemm::{MR, NR, gemm_rows_serial, with_scratch};

/// Tiles a segment of one tile row is processed in at most (stack buffers).
const SEG: usize = 64;

/// Floats of `V` plus `M` a block aims to keep (about 1 MiB, inside a core's L2).
const BLOCK_FLOATS: usize = 1 << 18;

/// Whether the 3x3 stride-1 dense conv `g` has the shape Winograd F(2x2, 3x3) handles: 3x3, stride 1,
/// dilation 1, one group. (Which of those take it is [`uses_winograd`]'s choice.)
pub(crate) fn winograd_shape(g: &ConvGeom) -> bool {
    g.kh == 3
        && g.kw == 3
        && g.p.stride == [1, 1]
        && g.p.dilation == [1, 1]
        && g.p.groups == 1
        && g.oh > 0
        && g.ow > 0
        && g.c > 0
}

/// Whether the inference forward of `g` runs as Winograd F(2x2, 3x3) rather than the direct shifted-rows GEMM:
/// where it measured faster, with at least 16 input and 8 output channels, and either 256 tiles (a 32x32 output)
/// or `c * o >= 4096`. At 28 threads it ran 5-41% faster than the direct conv on folo-n's convs from 40x40 up and
/// 11-29% on 12x12 to 24x24 outputs with 64 to 512 channels; folo-n's 20x20 64->64 convs tied, and 8 input
/// channels at 160x160 tied or lost (docs/hill/r2-cpu.md, second wave). At most [`MAX_CHANNELS`] input plus
/// output channels, which bounds the per-thread scratch a block keeps.
pub(crate) fn uses_winograd(g: &ConvGeom) -> bool {
    let tiles = g.oh.div_ceil(2) * g.ow.div_ceil(2);
    winograd_shape(g)
        && g.c >= 16
        && g.o >= 8
        && g.c + g.o <= MAX_CHANNELS
        && (tiles >= 256 || g.c * g.o >= 4096)
}

/// Input plus output channels a Winograd conv may have: a block's `V` and `M` (`16 · (C + O) · tb` floats, in each
/// worker thread's scratch, kept between calls) stay within 4 MiB at the smallest block (`tb = NR = 16`). Wider
/// convs (none in the FOLO models, whose layers have at most 768 channels) run direct.
const MAX_CHANNELS: usize = 4096;

#[cfg(test)]
thread_local! {
    /// Calls of [`transform_weights`] and of [`winograd_forward`] on this thread: tests observe dispatch and the
    /// weight cache with it.
    static CALLS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// `[transform_weights calls, winograd_forward calls]` made on this thread so far.
#[cfg(test)]
pub(crate) fn calls() -> [usize; 2] {
    CALLS.get()
}

#[cfg(test)]
fn count(i: usize) {
    CALLS.with(|c| {
        let mut v = c.get();
        v[i] += 1;
        c.set(v);
    });
}

/// `U = G g Gᵀ` of every `(o, c)` 3x3 kernel of the OIHW weights `wt`, laid out `[16][o][c]` (element `xi` of
/// the transform domain is the row-major `O x C` matrix `xi`), computed in f64 and rounded once.
pub(crate) fn transform_weights(wt: &[f32], o: usize, c: usize) -> Vec<f32> {
    #[cfg(test)]
    count(0);
    let oc = o * c;
    assert_eq!(wt.len(), oc * 9, "winograd: weights are not O x C x 3 x 3");
    let mut u = vec![0f32; 16 * oc];
    for (k, g) in wt.as_chunks::<9>().0.iter().enumerate() {
        let g: [f64; 9] = std::array::from_fn(|i| g[i] as f64);
        // t = G g (4 x 3), then U = t Gᵀ (4 x 4).
        let mut t = [[0f64; 3]; 4];
        for j in 0..3 {
            let (g0, g1, g2) = (g[j], g[3 + j], g[6 + j]);
            t[0][j] = g0;
            t[1][j] = 0.5 * (g0 + g1 + g2);
            t[2][j] = 0.5 * (g0 - g1 + g2);
            t[3][j] = g2;
        }
        for (i, r) in t.iter().enumerate() {
            let row = [r[0], 0.5 * (r[0] + r[1] + r[2]), 0.5 * (r[0] - r[1] + r[2]), r[2]];
            for (j, v) in row.into_iter().enumerate() {
                u[(i * 4 + j) * oc + k] = v as f32;
            }
        }
    }
    u
}

/// Tiles per block for `tiles` tiles of `c` in, `o` out channels on `threads` threads: a multiple of `NR`,
/// small enough for `V` and `M` to stay in L2, and small enough to give every thread a block when the image has
/// that many tiles.
fn block_tiles(c: usize, o: usize, tiles: usize, threads: usize) -> usize {
    let cache = (BLOCK_FLOATS / (16 * (c + o))).max(NR) / NR * NR;
    let spread = tiles.div_ceil(threads.max(1)).div_ceil(NR).max(1) * NR;
    cache.min(spread).clamp(NR, 8 * NR)
}

/// Forward conv2d of a [`winograd_shape`] conv into `out` (`[N, O, OH, OW]`, overwritten) with the transformed
/// weights `u` ([`transform_weights`]). `epi`, if given, sees every finished output element once.
/// An image with at least half as many blocks as threads runs each block as one task. A smaller one (a 20x20
/// output has 100 tiles, 7 blocks) runs in two passes, so that more tasks than blocks share the work without
/// repeating any of it: the input transforms of every block, split by input channels, into one `V`; then the
/// GEMMs and output transforms, split by output channels. The result does not depend on either: every tile's
/// transforms and every element's GEMM sum (over input channels in `KC` blocks, in order) are the same.
pub(crate) fn winograd_forward(
    x: ConvInput<'_>,
    u: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    debug_assert!(winograd_shape(g));
    #[cfg(test)]
    count(1);
    let (c, o) = (g.c, g.o);
    assert_eq!(u.len(), 16 * o * c, "winograd: transformed weights are not 16 x O x C");
    let tiles = g.oh.div_ceil(2) * g.ow.div_ceil(2);
    let hw = g.h * g.w;
    for_each_image(out, o * g.ohw(), g.n, 0, |img, out_img, threads| {
        let planes = x.planes(img, hw);
        let tb = block_tiles(c, o, tiles, threads);
        let blocks = tiles.div_ceil(tb);
        let span = |b: usize| (b * tb, ((b + 1) * tb).min(tiles));
        let optr = SendPtr(out_img.as_mut_ptr());
        let optr = &optr;
        if threads > 1 && 2 * blocks <= threads {
            // Parts of at least 8 input channels, and of a multiple of MR output channels: about one task per
            // thread in each pass.
            let per_block = threads.div_ceil(blocks);
            let cstep = c.div_ceil(per_block.min(c.div_ceil(8)));
            let cparts = c.div_ceil(cstep);
            let ostep = o.div_ceil(per_block.min(o.div_ceil(MR))).div_ceil(MR) * MR;
            let oparts = o.div_ceil(ostep);
            let vlen = 16 * c * tb;
            let mut vall = crate::cpu::alloc_uninit(blocks * vlen);
            let vptr = SendPtr(vall.as_mut_ptr());
            let vptr = &vptr;
            (0..blocks * cparts).into_par_iter().for_each(|q| {
                let (b, part) = (q / cparts, q % cparts);
                let (t0, t1) = span(b);
                let (ci0, ci1) = (part * cstep, ((part + 1) * cstep).min(c));
                // SAFETY: block b's V is the vlen floats from b * vlen, inside `vall`; this task is the only one
                // with its rows (xi, ci0..ci1), and nothing reads `vall` until every task is done.
                unsafe { input_transform(&planes[ci0..ci1], ci0, g, t0, t1, tb, vptr.0.add(b * vlen)) };
            });
            let vall_ref = &vall;
            (0..blocks * oparts).into_par_iter().for_each(|q| {
                let (b, part) = (q / oparts, q % oparts);
                let (t0, t1) = span(b);
                let (o0, o1) = (part * ostep, ((part + 1) * ostep).min(o));
                let v = &vall_ref[b * vlen..(b + 1) * vlen];
                with_scratch(16 * (o1 - o0) * tb, |m| {
                    // SAFETY: each (block, output channel part) owns its output elements.
                    unsafe { gemm_output(u, v, o0..o1, g, epi, t0, t1, tb, m, optr.0) };
                });
            });
            crate::cpu::recycle(vall);
            return;
        }
        let run = |b: usize| {
            let (t0, t1) = span(b);
            with_scratch(16 * (c + o) * tb, |buf| {
                let (v, m) = buf[..16 * (c + o) * tb].split_at_mut(16 * c * tb);
                // SAFETY: `v` is this block's V, all of it.
                unsafe { input_transform(&planes, 0, g, t0, t1, tb, v.as_mut_ptr()) };
                // SAFETY: blocks own disjoint output tiles, so the elements each one writes are its own.
                unsafe { gemm_output(u, v, 0..o, g, epi, t0, t1, tb, m, optr.0) };
            });
        };
        if threads > 1 && blocks > 1 {
            (0..blocks).into_par_iter().for_each(run);
        } else {
            (0..blocks).for_each(run);
        }
    });
}

/// The runs of tiles `t0..t1` (row-major over a `tw`-wide tile grid) that lie in one tile row, at most [`SEG`]
/// long: `(ty, tx0, len, pos)` with `pos` the run's first tile relative to `t0`.
fn segments(t0: usize, t1: usize, tw: usize) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    let mut t = t0;
    std::iter::from_fn(move || {
        if t >= t1 {
            return None;
        }
        let (ty, tx0) = (t / tw, t % tw);
        let len = (tw - tx0).min(t1 - t).min(SEG);
        let seg = (ty, tx0, len, t - t0);
        t += len;
        Some(seg)
    })
}

/// Defines `$name` as `$body` compiled for AVX2 + FMA and runs it when the CPU has them, else runs `$body` as
/// is: the same IEEE operations in the same order either way (Rust never fuses a multiply and an add on its
/// own), so the same results.
macro_rules! avx2_or_plain {
    ($name:ident ($($arg:ident: $ty:ty),* $(,)?) $body:block) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[target_feature(enable = "avx2,fma")]
            #[allow(clippy::too_many_arguments)]
            unsafe fn $name($($arg: $ty),*) $body
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                // SAFETY: the CPU has AVX2 and FMA.
                return unsafe { $name($($arg),*) };
            }
        }
        $body
    }};
}

/// Input transform of tiles `t0..t1` for the channels `ci0..` of `planes`: `V[i*4 + j][ci][pos] = (Bt d B)[i][j]`
/// of the 4x4 input patch `d` of tile `t0 + pos` in channel `ci`, into the block's `v` (`16 x C x tb`). Columns
/// past the last tile are zeros. Writes through slices of single rows `(xi, ci)` of `v`, so tasks that own other
/// channels of the same `V` can run at once.
///
/// # Safety
/// `v` points to a block's `V` (`16 · C · tb` floats), and no one else accesses its rows `(xi, ci0..ci0 + n)`
/// (every `xi`, `n = planes.len()`) while this runs.
unsafe fn input_transform(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    avx2_or_plain!(run(planes: &[&[f32]], ci0: usize, g: &ConvGeom, t0: usize, t1: usize, tb: usize, v: *mut f32) {
        // SAFETY: forwarded.
        unsafe { input_transform_body(planes, ci0, g, t0, t1, tb, v) }
    })
}

/// # Safety
/// As [`input_transform`].
#[inline(always)]
unsafe fn input_transform_body(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    let (c, h, w) = (g.c, g.h as isize, g.w as isize);
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let tw = g.ow.div_ceil(2);
    let n = t1 - t0;
    // tr[r][j][k]: input row r of the patches, transformed along the row (d·B).
    let mut tr = [[[0f32; SEG]; 4]; 4];
    let mut seg = [0f32; 2 * SEG + 2];
    for (cl, plane) in planes.iter().enumerate() {
        let ci = ci0 + cl;
        // SAFETY: row (xi, ci) of V is tb floats inside `v`, and the caller's alone; one is live at a time.
        let row = |xi: usize| unsafe { std::slice::from_raw_parts_mut(v.add((xi * c + ci) * tb), tb) };
        for (ty, tx0, len, pos) in segments(t0, t1, tw) {
            let x0 = 2 * tx0 as isize - pw;
            let (lo, hi) = (x0.max(0), (x0 + 2 * len as isize + 2).min(w));
            for (r, [t0r, t1r, t2r, t3r]) in tr.iter_mut().enumerate() {
                let y = 2 * ty as isize + r as isize - ph;
                // The 2·len + 2 input columns from x0, zero outside the plane.
                let seg = &mut seg[..2 * len + 2];
                if (0..h).contains(&y) && lo < hi {
                    let row = &plane[(y * w) as usize..((y + 1) * w) as usize];
                    let (a, b) = ((lo - x0) as usize, (hi - x0) as usize);
                    seg[..a].fill(0.0);
                    seg[a..b].copy_from_slice(&row[lo as usize..hi as usize]);
                    seg[b..].fill(0.0);
                } else {
                    seg.fill(0.0);
                }
                // Columns 2k..2k + 4 are tile k's: (e0, o0, e1, o1).
                let pairs = seg.as_chunks::<2>().0;
                for ((((p, q), a), b), (cc, d)) in pairs
                    .iter()
                    .zip(&pairs[1..])
                    .zip(t0r[..len].iter_mut())
                    .zip(t1r[..len].iter_mut())
                    .zip(t2r[..len].iter_mut().zip(t3r[..len].iter_mut()))
                {
                    *a = p[0] - q[0];
                    *b = p[1] + q[0];
                    *cc = q[0] - p[1];
                    *d = p[1] - q[1];
                }
            }
            let [tr0, tr1, tr2, tr3] = &tr;
            for (j, (((a0, a1), a2), a3)) in tr0.iter().zip(tr1).zip(tr2).zip(tr3).enumerate() {
                let at = |i: usize| &mut row(i * 4 + j)[pos..pos + len];
                let (a0, a1, a2, a3) = (&a0[..len], &a1[..len], &a2[..len], &a3[..len]);
                for ((d, x), y) in at(0).iter_mut().zip(a0).zip(a2) {
                    *d = x - y;
                }
                for ((d, x), y) in at(1).iter_mut().zip(a1).zip(a2) {
                    *d = x + y;
                }
                for ((d, x), y) in at(2).iter_mut().zip(a2).zip(a1) {
                    *d = x - y;
                }
                for ((d, x), y) in at(3).iter_mut().zip(a1).zip(a3) {
                    *d = x - y;
                }
            }
        }
        // The columns past the block's last tile are computed and dropped: keep them finite.
        if n < tb {
            for xi in 0..16 {
                row(xi)[n..].fill(0.0);
            }
        }
    }
}

/// The GEMMs `M[xi] = U[xi] · V[xi]` for the output channels `os` (into `m`, `16 x |os| x tb`), then the output
/// transform `Y = Aᵀ M A` of tiles `t0..t1` into the image's `out` (`[O, OH, OW]`), and the epilogue.
///
/// # Safety
/// `out` is the image's output, and no one else accesses its elements of channels `os` in tiles `t0..t1`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_output(
    u: &[f32],
    v: &[f32],
    os: std::ops::Range<usize>,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    t0: usize,
    t1: usize,
    tb: usize,
    m: &mut [f32],
    out: *mut f32,
) {
    let (c, o) = (g.c, g.o);
    let mo = os.len();
    let mut rows = Vec::with_capacity(c);
    for xi in 0..16 {
        rows.clear();
        rows.extend((0..c).map(|ci| v[(xi * c + ci) * tb..(xi * c + ci + 1) * tb].as_ptr()));
        let a = &u[(xi * o + os.start) * c..(xi * o + os.end) * c];
        // SAFETY: row ci of V[xi] holds tb floats.
        unsafe { gemm_rows_serial(mo, c, a, c, &rows, tb, &mut m[xi * mo * tb..(xi + 1) * mo * tb], tb) };
    }
    let (m, o0) = (&*m, os.start);
    avx2_or_plain!(run(m: &[f32], mo: usize, o0: usize, g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>, t0: usize, t1: usize, tb: usize, out: *mut f32) {
        // SAFETY: forwarded.
        unsafe { output_transform_body(m, mo, o0, g, epi, t0, t1, tb, out) }
    })
}

/// The output transform of `m` (`16 x mo x tb`, output channels `o0..o0 + mo`) into `out`, then the epilogue.
///
/// # Safety
/// As [`gemm_output`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform_body(
    m: &[f32],
    mo: usize,
    o0: usize,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    t0: usize,
    t1: usize,
    tb: usize,
    out: *mut f32,
) {
    let (oh, ow) = (g.oh, g.ow);
    let tw = ow.div_ceil(2);
    let mut sm = [[[0f32; SEG]; 4]; 2];
    let mut y = [0f32; 2 * SEG];
    for ol in 0..mo {
        let oc = o0 + ol;
        let mrow = |xi: usize| &m[(xi * mo + ol) * tb..][..tb];
        for (ty, tx0, len, pos) in segments(t0, t1, tw) {
            let [s0, s1] = &mut sm;
            for (j, (s0j, s1j)) in s0.iter_mut().zip(s1.iter_mut()).enumerate() {
                let r = |xi: usize| &mrow(xi)[pos..pos + len];
                let (m0, m1, m2, m3) = (r(j), r(4 + j), r(8 + j), r(12 + j));
                for (((d, a), b), cc) in s0j[..len].iter_mut().zip(m0).zip(m1).zip(m2) {
                    *d = a + b + cc;
                }
                for (((d, a), b), cc) in s1j[..len].iter_mut().zip(m1).zip(m2).zip(m3) {
                    *d = a - b - cc;
                }
            }
            let ox0 = 2 * tx0;
            let ox1 = (ox0 + 2 * len).min(ow);
            for (r, [s0, s1, s2, s3]) in sm.iter().enumerate() {
                let oy = 2 * ty + r;
                if oy >= oh {
                    break;
                }
                for (k, yy) in y.as_chunks_mut::<2>().0[..len].iter_mut().enumerate() {
                    yy[0] = s0[k] + s1[k] + s2[k];
                    yy[1] = s1[k] - s2[k] - s3[k];
                }
                // SAFETY: row oy, columns ox0..ox1 of output channel oc are inside the image, and the caller's.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.add((oc * oh + oy) * ow + ox0), ox1 - ox0) };
                dst.copy_from_slice(&y[..ox1 - ox0]);
                if let Some(e) = epi {
                    e.apply(oc, dst);
                }
            }
        }
    }
}
