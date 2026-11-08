//! conv2d forward and backward: implicit-im2col GEMM, with direct paths for 1x1 and depthwise convolutions.

use rayon::prelude::*;

use super::gemm::{Epilogue, Operand, Strided, gemm, gemm_ex};
use crate::backend::Conv2dParams;

/// Geometry shared by the kernels. All tensors are contiguous NCHW / OIHW.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvGeom {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub o: usize,
    pub kh: usize,
    pub kw: usize,
    pub oh: usize,
    pub ow: usize,
    pub p: Conv2dParams,
}

impl ConvGeom {
    fn cg(&self) -> usize {
        self.c / self.p.groups
    }
    fn og(&self) -> usize {
        self.o / self.p.groups
    }
    fn kdim(&self) -> usize {
        self.cg() * self.kh * self.kw
    }
    fn ohw(&self) -> usize {
        self.oh * self.ow
    }
    fn is_pointwise(&self) -> bool {
        self.kh == 1
            && self.kw == 1
            && self.p.stride == [1, 1]
            && self.p.padding == [0, 0]
            && (self.oh, self.ow) == (self.h, self.w)
    }
    fn is_depthwise(&self) -> bool {
        self.p.groups == self.c && self.p.groups > 1
    }
}

/// Copy one output row segment of the unfolded input: input row `iy` of `plane`, output columns
/// `ox0..ox0 + len`, kernel column offset `dx` (already includes `-pad`). Writes `dst[t * stride]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_segment(
    x: &[f32],
    plane: usize,
    g: &ConvGeom,
    iy: isize,
    dx: isize,
    ox0: usize,
    len: usize,
    dst: &mut [f32],
    stride: usize,
) {
    let sw = g.p.stride[1] as isize;
    if iy < 0 || iy >= g.h as isize {
        for t in 0..len {
            dst[t * stride] = 0.0;
        }
        return;
    }
    let row = &x[plane + iy as usize * g.w..plane + (iy as usize + 1) * g.w];
    let wi = g.w as isize;
    // Valid output columns: 0 <= ox * sw + dx < w.
    let ix0 = ox0 as isize * sw + dx;
    let lo = if ix0 >= 0 { 0 } else { ((-ix0) + sw - 1) / sw } as usize;
    let hi = if ix0 >= wi { 0 } else { (((wi - ix0) + sw - 1) / sw) as usize };
    let (lo, hi) = (lo.min(len), hi.min(len).max(lo.min(len)));
    for t in 0..lo {
        dst[t * stride] = 0.0;
    }
    if stride == 1 && sw == 1 && hi > lo {
        let s = (ix0 + lo as isize) as usize;
        dst[lo..hi].copy_from_slice(&row[s..s + (hi - lo)]);
    } else {
        for t in lo..hi {
            dst[t * stride] = row[(ix0 + t as isize * sw) as usize];
        }
    }
    for t in hi..len {
        dst[t * stride] = 0.0;
    }
}

/// B operand of the forward GEMM: element `(j, kk)` is the unfolded input `col[kk][j]` of image `img`, group `grp`.
struct Im2Col<'a> {
    x: &'a [f32],
    base: usize,
    g: ConvGeom,
}

impl Operand for Im2Col<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        let g = &self.g;
        let khw = g.kh * g.kw;
        let (sh, ph, pw) = (g.p.stride[0] as isize, g.p.padding[0] as isize, g.p.padding[1] as isize);
        let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
        for kk in 0..kc {
            let kidx = k0 + kk;
            let (ci, r) = (kidx / khw, kidx % khw);
            let (ky, kx) = ((r / g.kw) as isize, (r % g.kw) as isize);
            let plane = self.base + ci * g.h * g.w;
            let d = &mut dst[kk * width..kk * width + width];
            d[w..].fill(0.0);
            let (mut oy, mut ox) = (o0 / g.ow, o0 % g.ow);
            let mut t = 0;
            while t < w {
                let len = (g.ow - ox).min(w - t);
                let iy = oy as isize * sh - ph + ky * dh;
                im2col_segment(self.x, plane, g, iy, kx * dw - pw, ox, len, &mut d[t..], 1);
                t += len;
                ox = 0;
                oy += 1;
            }
        }
    }
}

/// B operand of the weight-gradient GEMM: element `(kidx, t)` with `t = img * ohw + s` is `col_img[kidx][s]`.
struct Im2ColT<'a> {
    x: &'a [f32],
    grp: usize,
    g: ConvGeom,
}

impl Operand for Im2ColT<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        let g = &self.g;
        let khw = g.kh * g.kw;
        let ohw = g.ohw();
        let (sh, ph, pw) = (g.p.stride[0] as isize, g.p.padding[0] as isize, g.p.padding[1] as isize);
        let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
        let dst = &mut dst[..kc * width];
        if w < width {
            dst.fill(0.0);
        }
        for ii in 0..w {
            let kidx = o0 + ii;
            let (ci, r) = (kidx / khw, kidx % khw);
            let (ky, kx) = ((r / g.kw) as isize, (r % g.kw) as isize);
            let mut t = k0;
            let mut kk = 0;
            while kk < kc {
                let (img, s) = (t / ohw, t % ohw);
                let (oy, ox) = (s / g.ow, s % g.ow);
                let len = (g.ow - ox).min(kc - kk);
                let plane = (img * g.c + self.grp * g.cg() + ci) * g.h * g.w;
                let iy = oy as isize * sh - ph + ky * dh;
                im2col_segment(
                    self.x,
                    plane,
                    g,
                    iy,
                    kx * dw - pw,
                    ox,
                    len,
                    &mut dst[kk * width + ii..],
                    width,
                );
                kk += len;
                t += len;
            }
        }
    }
}

/// A operand of the weight-gradient GEMM: element `(o, t)` with `t = img * ohw + s` is
/// `grad[img][grp * og + o][s]`.
struct GradRows<'a> {
    gy: &'a [f32],
    grp: usize,
    g: ConvGeom,
}

impl Operand for GradRows<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        let g = &self.g;
        let ohw = g.ohw();
        let dst = &mut dst[..kc * width];
        if w < width {
            dst.fill(0.0);
        }
        let mut t = k0;
        let mut kk = 0;
        while kk < kc {
            let (img, s) = (t / ohw, t % ohw);
            let len = (ohw - s).min(kc - kk);
            for ii in 0..w {
                let row = (img * g.o + self.grp * g.og() + o0 + ii) * ohw + s;
                for (q, &v) in self.gy[row..row + len].iter().enumerate() {
                    dst[(kk + q) * width + ii] = v;
                }
            }
            kk += len;
            t += len;
        }
    }
}

/// Above this many flops per image, one image's GEMM is parallelised; below it, images run in parallel.
const IMAGE_PAR_FLOPS: usize = 1 << 25;

/// Applied to every finished run of output values of one channel: `f(channel, values)`. Each output element is
/// passed exactly once (conv2d fuses bias + activation here).
pub(crate) type ChannelEpilogue<'a> = &'a (dyn Fn(usize, &mut [f32]) + Sync);

/// Run `f(img, out_img, threads)` for every image: one after another with the whole pool for large images, side
/// by side otherwise, splitting the pool between the images when there are fewer images than threads.
fn for_each_image(
    out: &mut [f32],
    per_image: usize,
    n: usize,
    flops: usize,
    f: impl Fn(usize, &mut [f32], usize) + Sync,
) {
    let threads = rayon::current_num_threads();
    if per_image == 0 {
        return;
    }
    if flops >= IMAGE_PAR_FLOPS || n == 1 {
        for (img, o) in out.chunks_mut(per_image).enumerate() {
            f(img, o, threads);
        }
    } else {
        let share = threads.div_ceil(n).max(1);
        out.par_chunks_mut(per_image).enumerate().for_each(|(img, o)| f(img, o, share));
    }
}

/// Forward conv2d into `out` (`[N, O, OH, OW]`): overwritten, or accumulated into if `accumulate`. `epi`, if
/// given, sees every finished output element once.
pub(crate) fn conv2d_forward(
    x: &[f32],
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    if g.is_depthwise() {
        if !accumulate {
            out.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
        }
        return depthwise_forward(x, wt, out, g, epi);
    }
    let (cg, og, kdim, ohw) = (g.cg(), g.og(), g.kdim(), g.ohw());
    let per_image_flops = 2 * g.o * ohw * kdim;
    for_each_image(out, g.o * ohw, g.n, per_image_flops, |img, out_img, threads| {
        for grp in 0..g.p.groups {
            let a = Strided::lhs(wt, grp * og * kdim, kdim, 1);
            let c = &mut out_img[grp * og * ohw..(grp + 1) * og * ohw];
            let base = (img * g.c + grp * cg) * g.h * g.w;
            let rows = epi.map(|f| move |i: usize, _: usize, row: &mut [f32]| f(grp * og + i, row));
            let rows = rows.as_ref().map(|r| r as Epilogue<'_>);
            if g.is_pointwise() {
                let b = Strided::rhs(x, base, g.h * g.w, 1);
                gemm_ex(og, ohw, kdim, &a, &b, c, ohw, accumulate, threads, rows);
            } else {
                let b = Im2Col { x, base, g: *g };
                gemm_ex(og, ohw, kdim, &a, &b, c, ohw, accumulate, threads, rows);
            }
        }
    });
}

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
        for_each_image(gx, g.c * hw, g.n, 2 * g.c * hw * og, |img, gx_img, threads| {
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
    let mut flipped = super::alloc_uninit(g.c * og * khw);
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
    if flipped.len() >= super::PAR_MIN {
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
    super::recycle(flipped);
}

/// [`conv2d_backward_input`] through an unfolded gradient (`W^T gy`) and col2im.
fn backward_input_col2im(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    let (cg, og, kdim, ohw, hw) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
    let flops = 2 * g.c * ohw * og * g.kh * g.kw;
    for_each_image(gx, g.c * hw, g.n, flops, |img, gx_img, threads| {
        // Every element of `col` is overwritten, so a recycled buffer needs no clearing.
        let mut col = super::alloc_uninit(kdim * ohw);
        for grp in 0..g.p.groups {
            // col = W_g^T [kdim x og] * gy[img, g] [og x ohw]
            let a = Strided::lhs(wt, grp * og * kdim, 1, kdim);
            let b = Strided::rhs(gy, (img * g.o + grp * og) * ohw, ohw, 1);
            gemm_ex(kdim, ohw, og, &a, &b, &mut col, ohw, false, threads, None);
            col2im(&col, &mut gx_img[grp * cg * hw..(grp + 1) * cg * hw], g, threads > 1);
        }
        super::recycle(col);
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

/// Gradient with respect to the weight, written into `gw` (zeroed, `[O, C/groups, KH, KW]`).
pub(crate) fn conv2d_backward_weight(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom) {
    if g.is_depthwise() {
        return depthwise_backward_weight(x, gy, gw, g);
    }
    let (og, kdim) = (g.og(), g.kdim());
    let k = g.n * g.ohw();
    for grp in 0..g.p.groups {
        let a = GradRows { gy, grp, g: *g };
        let b = Im2ColT { x, grp, g: *g };
        gemm(og, kdim, k, &a, &b, &mut gw[grp * og * kdim..(grp + 1) * og * kdim], kdim, false, true);
    }
}

/// Dot product with independent accumulators so it vectorises.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for v in acc {
        s += v;
    }
    s
}

/// Output columns `ox` with `0 <= ox * s + d < w`, as a range.
#[inline]
fn valid_cols(ow: usize, s: isize, d: isize, w: isize) -> std::ops::Range<usize> {
    let lo = if d >= 0 { 0 } else { ((-d) + s - 1) / s } as usize;
    let hi = if d >= w { 0 } else { ((w - d) + s - 1) / s } as usize;
    let (lo, hi) = (lo.min(ow), hi.min(ow));
    lo..hi.max(lo)
}

fn depthwise_forward(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    out.par_chunks_mut(ohw).enumerate().for_each(|(plane, op)| {
        let (img, oc) = (plane / g.o, plane % g.o);
        let xp = &x[(img * g.c + oc / mult) * hw..][..hw];
        let wk = &wt[oc * khw..(oc + 1) * khw];
        for oy in 0..g.oh {
            let orow = &mut op[oy * g.ow..(oy + 1) * g.ow];
            for ky in 0..g.kh {
                let iy = oy as isize * sh - ph + ky as isize * dh;
                if iy < 0 || iy >= g.h as isize {
                    continue;
                }
                let xrow = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
                for kx in 0..g.kw {
                    let wv = wk[ky * g.kw + kx];
                    let d = kx as isize * dw - pw;
                    let r = valid_cols(g.ow, sw, d, g.w as isize);
                    if r.is_empty() {
                        continue;
                    }
                    if sw == 1 {
                        let s0 = (r.start as isize + d) as usize;
                        let len = r.len();
                        for (o, &xv) in orow[r].iter_mut().zip(&xrow[s0..s0 + len]) {
                            *o += wv * xv;
                        }
                    } else {
                        for ox in r {
                            orow[ox] += wv * xrow[(ox as isize * sw + d) as usize];
                        }
                    }
                }
            }
        }
        if let Some(f) = epi {
            f(oc, op);
        }
    });
}

fn depthwise_backward_input(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    gx.par_chunks_mut(hw).enumerate().for_each(|(plane, gxp)| {
        let (img, ic) = (plane / g.c, plane % g.c);
        for oc in ic * mult..(ic + 1) * mult {
            let gp = &gy[(img * g.o + oc) * ohw..][..ohw];
            let wk = &wt[oc * khw..(oc + 1) * khw];
            for oy in 0..g.oh {
                let grow = &gp[oy * g.ow..(oy + 1) * g.ow];
                for ky in 0..g.kh {
                    let iy = oy as isize * sh - ph + ky as isize * dh;
                    if iy < 0 || iy >= g.h as isize {
                        continue;
                    }
                    let xrow = &mut gxp[iy as usize * g.w..(iy as usize + 1) * g.w];
                    for kx in 0..g.kw {
                        let wv = wk[ky * g.kw + kx];
                        let d = kx as isize * dw - pw;
                        let r = valid_cols(g.ow, sw, d, g.w as isize);
                        if r.is_empty() {
                            continue;
                        }
                        if sw == 1 {
                            let s0 = (r.start as isize + d) as usize;
                            let len = r.len();
                            for (xv, &gv) in xrow[s0..s0 + len].iter_mut().zip(&grow[r]) {
                                *xv += wv * gv;
                            }
                        } else {
                            for ox in r {
                                xrow[(ox as isize * sw + d) as usize] += wv * grow[ox];
                            }
                        }
                    }
                }
            }
        }
    });
}

fn depthwise_backward_weight(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    gw.par_chunks_mut(khw).enumerate().for_each(|(oc, gk)| {
        let ic = oc / mult;
        for img in 0..g.n {
            let xp = &x[(img * g.c + ic) * hw..][..hw];
            let gp = &gy[(img * g.o + oc) * ohw..][..ohw];
            for ky in 0..g.kh {
                for kx in 0..g.kw {
                    let d = kx as isize * dw - pw;
                    let cols = valid_cols(g.ow, sw, d, g.w as isize);
                    let mut acc = 0f32;
                    for oy in 0..g.oh {
                        let iy = oy as isize * sh - ph + ky as isize * dh;
                        if iy < 0 || iy >= g.h as isize {
                            continue;
                        }
                        let xrow = &xp[iy as usize * g.w..(iy as usize + 1) * g.w];
                        let grow = &gp[oy * g.ow..(oy + 1) * g.ow];
                        if cols.is_empty() {
                            continue;
                        }
                        if sw == 1 {
                            let s0 = (cols.start as isize + d) as usize;
                            acc += dot(&grow[cols.clone()], &xrow[s0..s0 + cols.len()]);
                        } else {
                            for ox in cols.clone() {
                                acc += grow[ox] * xrow[(ox as isize * sw + d) as usize];
                            }
                        }
                    }
                    gk[ky * g.kw + kx] += acc;
                }
            }
        }
    });
}
