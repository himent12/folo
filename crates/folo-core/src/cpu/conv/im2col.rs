//! The forward's packed B operands read from the input planes: the implicit im2col of a strided or accumulating
//! conv ([`Im2Col`], never materialised: each block is unfolded as it is packed) and a pointwise conv's planes.

use super::ConvGeom;
use crate::cpu::gemm::{Operand, pack_rows_into_panels};

/// Copy one output row segment of the unfolded input: input row `iy` of `plane` (one `h x w` input plane),
/// output columns `ox0..ox0 + len`, kernel column offset `dx` (already includes `-pad`). Writes
/// `dst[t * stride]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn im2col_segment(
    plane: &[f32],
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
    let row = &plane[iy as usize * g.w..(iy as usize + 1) * g.w];
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

/// B operand of the forward GEMM: element `(j, kk)` is the unfolded input `col[kk][j]` of one image and group,
/// whose input channel `ci` is the plane `planes[ci]`.
pub(super) struct Im2Col<'a> {
    pub(super) planes: &'a [&'a [f32]],
    pub(super) g: ConvGeom,
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
            let plane = self.planes[ci];
            let d = &mut dst[kk * width..kk * width + width];
            d[w..].fill(0.0);
            let (mut oy, mut ox) = (o0 / g.ow, o0 % g.ow);
            let mut t = 0;
            while t < w {
                let len = (g.ow - ox).min(w - t);
                let iy = oy as isize * sh - ph + ky * dh;
                im2col_segment(plane, g, iy, kx * dw - pw, ox, len, &mut d[t..], 1);
                t += len;
                ox = 0;
                oy += 1;
            }
        }
    }
}

/// B operand of a pointwise forward GEMM: element `(j, kk)` is pixel `j` of input plane `planes[kk]`.
pub(super) struct PlaneRows<'a> {
    pub(super) planes: &'a [&'a [f32]],
}

impl Operand for PlaneRows<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        for kk in 0..kc {
            let d = &mut dst[kk * width..kk * width + width];
            d[..w].copy_from_slice(&self.planes[k0 + kk][o0..o0 + w]);
            d[w..].fill(0.0);
        }
    }

    fn pack_panels(&self, o0: usize, w: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        pack_rows_into_panels(w, kc, dst, |kk| &self.planes[k0 + kk][o0..o0 + w]);
    }
}
