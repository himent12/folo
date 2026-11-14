//! Forward convs evaluated only at some output sites (`Backend::conv2d_act_at`): the inference forward's own
//! arithmetic on fewer outputs, so every site gets the value the dense conv computes there, bit for bit.
//!
//! - A Winograd conv ([`super::uses_winograd`]) runs the block code over the list of 2x2 output tiles that hold a
//!   site ([`super::winograd_forward_at`]).
//! - A pointwise conv runs the dense path's GEMM ([`gemm_ex`] with the weights read in place and `B` packed from
//!   planes) over the sites' columns gathered into planes of their own: every element of C is the same K sum in
//!   the same `KC` blocks, through the same micro-kernel and epilogue, wherever its column is.

use super::im2col::PlaneRows;
use super::{ChannelEpilogue, ConvGeom, ConvInput};
use crate::cpu::gemm::{KC, Strided, gemm_ex};

/// Per image, the ascending distinct 2x2 output tiles (`ty · ⌈OW/2⌉ + tx`) holding one of the sites `at` (flat
/// `img · OH·OW + oy · OW + ox`, each `< N · OH · OW`).
pub(crate) fn winograd_tiles_at(g: &ConvGeom, at: &[usize]) -> Vec<Vec<u32>> {
    let (ohw, tw) = (g.ohw(), g.ow.div_ceil(2));
    let mut lists = vec![Vec::new(); g.n];
    for &s in at {
        let (img, pix) = (s / ohw, s % ohw);
        let (oy, ox) = (pix / g.ow, pix % g.ow);
        lists[img].push(((oy / 2) * tw + ox / 2) as u32);
    }
    for l in &mut lists {
        l.sort_unstable();
        l.dedup();
    }
    lists
}

/// A pointwise conv (`g` 1x1, stride 1, no padding, one group) at the output sites `at` (as in
/// [`winograd_tiles_at`]) into `out` (`[N, O, OH, OW]`): the sites' elements are written, every other element
/// is left as it was. Returns `false`, writing nothing, for a conv the dense path would not run as this GEMM
/// (grouped, or so deep in K that [`gemm_ex`]'s K split, which depends on the column count, could apply).
pub(crate) fn pointwise_forward_at(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    at: &[usize],
) -> bool {
    // `k_split_parts` is 1 below 8 · KC for every shape, the dense call's included.
    if !g.is_pointwise() || g.p.groups != 1 || g.kdim() >= 8 * KC {
        return false;
    }
    let (c, o, ohw, hw) = (g.c, g.o, g.ohw(), g.h * g.w);
    let mut by_image = vec![Vec::new(); g.n];
    for &s in at {
        by_image[s / ohw].push(s % ohw);
    }
    let threads = rayon::current_num_threads();
    for (img, pix) in by_image.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        let s = pix.len();
        let planes = x.planes(img, hw);
        let mut cols = crate::cpu::alloc_uninit(c * s);
        for (dst, plane) in cols.chunks_exact_mut(s).zip(&planes) {
            for (d, &p) in dst.iter_mut().zip(pix) {
                *d = plane[p];
            }
        }
        let rows: Vec<&[f32]> = cols.chunks_exact(s).collect();
        let mut y = crate::cpu::alloc_uninit(o * s);
        let a = Strided::lhs(wt, 0, c, 1);
        gemm_ex(o, s, c, &a, &PlaneRows { planes: &rows }, &mut y, s, false, threads, epi);
        let out_img = &mut out[img * o * ohw..(img + 1) * o * ohw];
        for (plane, row) in out_img.chunks_exact_mut(ohw).zip(y.chunks_exact(s)) {
            for (&p, &v) in pix.iter().zip(row) {
                plane[p] = v;
            }
        }
        crate::cpu::recycle(cols);
        crate::cpu::recycle(y);
    }
    true
}
