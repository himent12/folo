//! The input gradient: at stride 1 the forward conv of the output gradient with the flipped, channel-swapped
//! kernel (no col2im); pointwise convs as `Wᵀ · gy`; strided convs through an unfolded gradient and col2im.

use rayon::prelude::*;

use super::depthwise::depthwise_backward_input;
use super::{ConvGeom, conv2d_forward, for_each_image};
use crate::backend::Conv2dParams;
use crate::cpu::gemm::{Strided, gemm_ex};

/// True if [`conv2d_backward_input`] takes the transposed-convolution path: stride 1 and padding within the
/// kernel's reach (`p <= d (k - 1)`) on both axes, not depthwise.
fn transposed_dx(g: &ConvGeom) -> bool {
    !g.is_depthwise()
        && g.p.stride == [1, 1]
        && g.p.dilation[0] * (g.kh - 1) >= g.p.padding[0]
        && g.p.dilation[1] * (g.kw - 1) >= g.p.padding[1]
}

/// True if [`conv2d_backward_input`] writes every element of `gx`, so it need not be zeroed first.
pub(crate) fn backward_input_overwrites(g: &ConvGeom) -> bool {
    transposed_dx(g)
}

/// Gradient with respect to the input, written into `gx` (`[N, C, H, W]`, zeroed unless
/// [`backward_input_overwrites`]).
///
/// With stride 1 this is the convolution of `gy` with the flipped kernel, channels swapped within each group
/// (`dx[c, i] = sum_{o,t} W[o, c, t] gy[o, i + p - t d]`, i.e. offset `d (k - 1) - p`), run through the forward
/// kernel: no unfolded matrix and no scattered adds. Pointwise convs multiply by `W^T` directly. Strided convs
/// go through an unfolded gradient and col2im.
pub(crate) fn conv2d_backward_input(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    if g.is_depthwise() {
        return depthwise_backward_input(gy, wt, gx, g);
    }
    let (cg, og, groups) = (g.cg(), g.og(), g.p.groups);
    let hw = g.h * g.w;
    if g.is_pointwise() {
        // gx[img, grp] = W_g^T [cg x og] * gy[img, grp] [og x hw], reading W transposed in place.
        for_each_image(gx, g.c * hw, g.n, 0, |img, gx_img, threads| {
            for grp in 0..groups {
                let a = Strided::lhs(wt, grp * og * cg, 1, cg);
                let b = Strided::rhs(gy, (img * g.o + grp * og) * hw, hw, 1);
                let c = &mut gx_img[grp * cg * hw..(grp + 1) * cg * hw];
                gemm_ex(cg, hw, og, &a, &b, c, hw, false, threads, None);
            }
        });
        return;
    }
    if !transposed_dx(g) {
        return backward_input_col2im(gy, wt, gx, g);
    }
    // W'[c, o, ky', kx'] = W[o, c, kh - 1 - ky', kw - 1 - kx'] (within each group). Rows of W' (one per input
    // channel) are built in blocks of 8 so that both the reads of W and the writes stay in L1.
    let khw = g.kh * g.kw;
    let mut flipped = crate::cpu::alloc_uninit(g.c * og * khw);
    let block = |(b, dst): (usize, &mut [f32])| {
        let rows = dst.len() / (og * khw);
        for oi in 0..og {
            for r in 0..rows {
                let row = b * 8 + r;
                let (grp, ci) = (row / cg, row % cg);
                let src = &wt[((grp * og + oi) * cg + ci) * khw..][..khw];
                let d = &mut dst[(r * og + oi) * khw..][..khw];
                for (t, v) in d.iter_mut().enumerate() {
                    *v = src[khw - 1 - t];
                }
            }
        }
    };
    if flipped.len() >= crate::cpu::PAR_MIN {
        flipped.par_chunks_mut(8 * og * khw).enumerate().for_each(block);
    } else {
        flipped.chunks_mut(8 * og * khw).enumerate().for_each(block);
    }
    let (dh, dw) = (g.p.dilation[0], g.p.dilation[1]);
    let tg = ConvGeom {
        n: g.n,
        c: g.o,
        h: g.oh,
        w: g.ow,
        o: g.c,
        kh: g.kh,
        kw: g.kw,
        oh: g.h,
        ow: g.w,
        p: Conv2dParams {
            stride: [1, 1],
            padding: [dh * (g.kh - 1) - g.p.padding[0], dw * (g.kw - 1) - g.p.padding[1]],
            dilation: [dh, dw],
            groups,
        },
    };
    conv2d_forward(gy, &flipped, gx, &tg, false, None);
    crate::cpu::recycle(flipped);
}

/// [`conv2d_backward_input`] through an unfolded gradient (`W^T gy`) and col2im.
fn backward_input_col2im(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    let (cg, og, kdim, ohw, hw) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
    for_each_image(gx, g.c * hw, g.n, kdim * ohw, |img, gx_img, threads| {
        // Every element of `col` is overwritten, so a recycled buffer needs no clearing.
        let mut col = crate::cpu::alloc_uninit(kdim * ohw);
        for grp in 0..g.p.groups {
            // col = W_g^T [kdim x og] * gy[img, g] [og x ohw]
            let a = Strided::lhs(wt, grp * og * kdim, 1, kdim);
            let b = Strided::rhs(gy, (img * g.o + grp * og) * ohw, ohw, 1);
            gemm_ex(kdim, ohw, og, &a, &b, &mut col, ohw, false, threads, None);
            col2im(&col, &mut gx_img[grp * cg * hw..(grp + 1) * cg * hw], g, threads > 1);
        }
        crate::cpu::recycle(col);
    });
}

/// Accumulate the unfolded gradient `col` (`[cg*kh*kw, ohw]`) into the input planes of one group.
fn col2im(col: &[f32], gx: &mut [f32], g: &ConvGeom, parallel: bool) {
    let (khw, ohw, hw) = (g.kh * g.kw, g.ohw(), g.h * g.w);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    let (hi, wi) = (g.h as isize, g.w as isize);
    let plane = |(ci, gxp): (usize, &mut [f32])| {
        for r in 0..khw {
            let (ky, kx) = ((r / g.kw) as isize, (r % g.kw) as isize);
            let crow = &col[(ci * khw + r) * ohw..(ci * khw + r + 1) * ohw];
            for oy in 0..g.oh {
                let iy = oy as isize * sh - ph + ky * dh;
                if iy < 0 || iy >= hi {
                    continue;
                }
                let dst = &mut gxp[iy as usize * g.w..(iy as usize + 1) * g.w];
                let src = &crow[oy * g.ow..(oy + 1) * g.ow];
                for (ox, &v) in src.iter().enumerate() {
                    let ix = ox as isize * sw - pw + kx * dw;
                    if ix >= 0 && ix < wi {
                        dst[ix as usize] += v;
                    }
                }
            }
        }
    };
    if parallel {
        gx.par_chunks_mut(hw).enumerate().for_each(plane);
    } else {
        gx.chunks_mut(hw).enumerate().for_each(plane);
    }
}
