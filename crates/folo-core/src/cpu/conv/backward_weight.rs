//! The weight gradient: `dWᵀ = X_unfolded · gyᵀ` over the input's shifted rows read in place ([`gemm_arows`]),
//! K cut into parts and the rows into tiles by shape alone; strided and narrow pointwise convs pack an unfolded
//! input as they go ([`gemm`] over [`Im2ColT`]).

use rayon::prelude::*;

use super::depthwise::depthwise_backward_weight;
use super::im2col::im2col_segment;
use super::shifted::{ShiftedRows, shifted_scratch};
use super::{ConvGeom, image_lanes};
use crate::cpu::elementwise::SendPtr;
use crate::cpu::gemm::{ColumnMap, KC, MR, NR, Operand, gemm, gemm_arows};

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
                    &self.x[plane..plane + g.h * g.w],
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

/// One image's output gradient as the packed B of the weight-gradient GEMM (see [`gemm_arows`]): panel `q`
/// (channels `q·NR..q·NR + NR` of the `og` in `gy`, `ohw` floats each) at `dst[q * cols * NR..]`, column `t` of
/// the shifted rows at `+ t * NR`: output pixel `(oy, ox)` for `t = oy · wq + ox` with `ox < ow`, zero in the
/// dropped columns and past channel `og`.
fn pack_grad_cols(gy: &[f32], og: usize, ohw: usize, map: ColumnMap, cols: usize, dst: &mut [f32]) {
    let (wq, ow) = (map.wp, map.ow);
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2");
    for (q, panel) in dst.chunks_mut(cols * NR).enumerate() {
        let o0 = q * NR;
        let w = NR.min(og - o0);
        for (oy, prow) in panel.chunks_mut(wq * NR).enumerate() {
            // Columns ox >= ow, and channels past og, are zero.
            prow[ow * NR..].fill(0.0);
            #[allow(unused_mut)]
            let mut done = 0;
            #[cfg(target_arch = "x86_64")]
            if avx2 && w == NR {
                let src: [&[f32]; NR] = std::array::from_fn(|jj| &gy[(o0 + jj) * ohw + oy * ow..][..ow]);
                // SAFETY: AVX2 was detected; `src` rows hold `ow` floats and `prow` at least `ow * NR`.
                unsafe { transpose16_avx2(&src, ow / 8, prow) };
                done = ow / 8 * 8;
            }
            for ox in done..ow {
                let d = &mut prow[ox * NR..ox * NR + NR];
                for (jj, v) in d[..w].iter_mut().enumerate() {
                    *v = gy[(o0 + jj) * ohw + oy * ow + ox];
                }
                d[w..].fill(0.0);
            }
        }
    }
}

/// `dst[x * NR + j] = src[j][x]` for `x < 8 * blocks`, `j < NR`: sixteen rows transposed eight columns at a
/// time through two 8x8 register transposes.
///
/// # Safety
/// The CPU supports AVX2; every `src[j]` holds `8 * blocks` floats and `dst` `8 * blocks * NR`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose16_avx2(src: &[&[f32]; NR], blocks: usize, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(dst.len() >= blocks * 8 * NR && src.iter().all(|r| r.len() >= blocks * 8));
    for b in 0..blocks {
        for half in 0..2 {
            // SAFETY: the loads read `src[j][8b..8b + 8]` and the stores write `dst[(8b + t) * NR + 8 half..][..8]`,
            // all in bounds (asserted above).
            unsafe {
                let r: [__m256; 8] =
                    std::array::from_fn(|j| _mm256_loadu_ps(src[half * 8 + j].as_ptr().add(8 * b)));
                let t0 = _mm256_unpacklo_ps(r[0], r[1]);
                let t1 = _mm256_unpackhi_ps(r[0], r[1]);
                let t2 = _mm256_unpacklo_ps(r[2], r[3]);
                let t3 = _mm256_unpackhi_ps(r[2], r[3]);
                let t4 = _mm256_unpacklo_ps(r[4], r[5]);
                let t5 = _mm256_unpackhi_ps(r[4], r[5]);
                let t6 = _mm256_unpacklo_ps(r[6], r[7]);
                let t7 = _mm256_unpackhi_ps(r[6], r[7]);
                let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                let cols = [
                    _mm256_permute2f128_ps::<0x20>(u0, u4),
                    _mm256_permute2f128_ps::<0x20>(u1, u5),
                    _mm256_permute2f128_ps::<0x20>(u2, u6),
                    _mm256_permute2f128_ps::<0x20>(u3, u7),
                    _mm256_permute2f128_ps::<0x31>(u0, u4),
                    _mm256_permute2f128_ps::<0x31>(u1, u5),
                    _mm256_permute2f128_ps::<0x31>(u2, u6),
                    _mm256_permute2f128_ps::<0x31>(u3, u7),
                ];
                for (t, v) in cols.iter().enumerate() {
                    _mm256_storeu_ps(dst.as_mut_ptr().add((8 * b + t) * NR + 8 * half), *v);
                }
            }
        }
    }
}

/// One image's operands of the weight-gradient GEMM: A, its shifted rows (or, for a pointwise conv, its input
/// planes), and B, its packed output gradient.
struct ImageRows {
    /// The shifted-rows copy, if one was made.
    sr: Option<ShiftedRows>,
    /// Row `kidx = (ci, ky, kx)`: `cols` floats, column `oy · map.wp + ox` of output pixel `(oy, ox)`.
    rows: Vec<*const f32>,
    /// [`pack_grad_cols`] of the image's output gradient.
    grad: Vec<f32>,
}

// SAFETY: the pointers are only read, into `sr` or the caller's input, both alive and unmodified while the rows
// are used.
unsafe impl Send for ImageRows {}
unsafe impl Sync for ImageRows {}

/// Tasks the weight gradient aims at (a function of the shape only, like every split here), and the most
/// floats of K-part partial sums it keeps.
const WGRAD_TASKS: usize = 128;
const WGRAD_PARTIAL_MAX: usize = 1 << 22;

/// `(parts, tiles)` of the `m x n x k` weight-gradient GEMM: K is cut into `parts` (each summed into its own
/// partial `C`, the partials added in order), the rows of C into `tiles`. Depends on the shape only.
fn wgrad_split(m: usize, n: usize, k: usize) -> (usize, usize) {
    let target = ((2 * m * n * k) >> 19).clamp(1, WGRAD_TASKS);
    let mut parts = (k / (2 * KC)).clamp(1, target);
    while parts > 1 && parts * m * n > WGRAD_PARTIAL_MAX {
        parts = parts.div_ceil(2);
    }
    let tiles = target.div_ceil(parts).min(m.div_ceil(2 * MR)).max(1);
    (parts, tiles)
}

/// Gradient with respect to the weight, written into `gw` (`[O, C/groups, KH, KW]`, zeroed).
///
/// Where [`wgrad_in_place`] holds, per group, `dWᵀ = X_unfolded · gyᵀ`: a `kdim x og` GEMM over every output
/// pixel of every image, whose A rows are the input's shifted rows (the forward's B operand, read in place, so
/// the unfolded input is never packed) and whose B, the output gradient, is packed with zeros in the dropped
/// columns. K (images × columns) is cut into parts and the rows into tiles by shape alone ([`wgrad_split`]);
/// each part sums into its own partial in a fixed order and the partials are added in order, so the result
/// does not depend on the thread count. Elsewhere, `dW = gy · X_unfoldedᵀ` through [`gemm`] with the unfolded
/// input packed block by block.
pub(crate) fn conv2d_backward_weight(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom) {
    if g.is_depthwise() {
        return depthwise_backward_weight(x, gy, gw, g);
    }
    let (og, kdim) = (g.og(), g.kdim());
    if !wgrad_in_place(g) {
        let k = g.n * g.ohw();
        for grp in 0..g.p.groups {
            let a = GradRows { gy, grp, g: *g };
            let b = Im2ColT { x, grp, g: *g };
            gemm(og, kdim, k, &a, &b, &mut gw[grp * og * kdim..(grp + 1) * og * kdim], kdim, false, true);
        }
        return;
    }
    for grp in 0..g.p.groups {
        backward_weight_group(x, gy, &mut gw[grp * og * kdim..(grp + 1) * og * kdim], g, grp);
    }
}

/// True if [`conv2d_backward_weight`] reads the input's shifted rows in place, false if it packs an unfolded
/// input as it goes (the older path). Measured on folo-n's training shapes: the shifted rows win for stride-1
/// kernels and for pointwise convs with many input channels; for strided convs and narrow pointwise ones, whose
/// GEMMs do few flops per input element, writing the shifted-rows copy and the transposed gradient costs more
/// than packing on the fly.
fn wgrad_in_place(g: &ConvGeom) -> bool {
    if g.is_pointwise() { g.cg() >= 64 } else { g.p.stride == [1, 1] }
}

fn backward_weight_group(x: &[f32], gy: &[f32], gw: &mut [f32], g: &ConvGeom, grp: usize) {
    let (cg, og, kdim, ohw, hw) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
    let planes = |img: usize| -> Vec<&[f32]> {
        (0..cg).map(|ci| &x[(img * g.c + grp * cg + ci) * hw..][..hw]).collect()
    };
    let (cols, map) = if g.is_pointwise() {
        (hw, ColumnMap { wp: g.w, ow: g.w })
    } else {
        let wq = (g.w + 2 * g.p.padding[1]).div_ceil(g.p.stride[1]);
        (g.oh * wq, ColumnMap { wp: wq, ow: g.ow })
    };
    let b_stride = cols * NR;
    // The images' operands are made a wave at a time, as many images as fit the scratch budget.
    let per_image = og.div_ceil(NR) * b_stride + if g.is_pointwise() { 0 } else { shifted_scratch(g) };
    let wave = image_lanes(g.n, per_image);
    let image = |img: usize| -> ImageRows {
        // Every element is written by `pack_grad_cols`.
        let mut grad = super::alloc_uninit(og.div_ceil(NR) * b_stride);
        pack_grad_cols(&gy[(img * g.o + grp * og) * ohw..], og, ohw, map, cols, &mut grad);
        if g.is_pointwise() {
            ImageRows { sr: None, rows: planes(img).iter().map(|p| p.as_ptr()).collect(), grad }
        } else {
            let sr = ShiftedRows::new(&planes(img), g, wave == 1);
            let rows = sr.rows.clone();
            ImageRows { sr: Some(sr), rows, grad }
        }
    };
    let (m, n, k) = (kdim, og, g.n * cols);
    let (parts, tiles) = wgrad_split(m, n, k);
    let step = k.div_ceil(parts).div_ceil(KC) * KC;
    let parts = k.div_ceil(step);
    let rpt = m.div_ceil(tiles).div_ceil(MR) * MR;
    let tiles = m.div_ceil(rpt);
    // Every element of every partial is written by the first segment of its task.
    let mut partial = super::alloc_uninit(parts * m * n);
    let pptr = SendPtr(partial.as_mut_ptr());
    for w0 in (0..g.n).step_by(wave) {
        let w1 = (w0 + wave).min(g.n);
        let images: Vec<ImageRows> =
            if w1 - w0 > 1 { (w0..w1).into_par_iter().map(image).collect() } else { vec![image(w0)] };
        debug_assert!(
            images.iter().all(|im| im.sr.as_ref().is_none_or(|s| s.n == cols && s.map.wp == map.wp))
        );
        // Each task sums the segments of its K range that fall in this wave's images, in K order, continuing
        // the previous wave's sum: the same calls in the same order as with every image at once. Only the K parts
        // that overlap the wave have work, so a wave of a few images cuts the rows into more tiles to keep the
        // pool busy (which rows share a call changes no element's sum).
        let (p0, p1) = (w0 * cols / step, (w1 * cols).div_ceil(step).min(parts));
        let active = p1 - p0;
        let (rpt, tiles) = if wave >= g.n {
            (rpt, tiles)
        } else {
            let t = m.div_ceil(2 * MR).min(WGRAD_TASKS.div_ceil(active)).max(1);
            let r = m.div_ceil(t).div_ceil(MR) * MR;
            (r, m.div_ceil(r))
        };
        let task = |t: usize| {
            let pptr = &pptr;
            let (ti, p) = (t / active, p0 + t % active);
            let (i0, i1) = (ti * rpt, ((ti + 1) * rpt).min(m));
            let (k0, k1) = (p * step, ((p + 1) * step).min(k));
            let (lo, hi) = (k0.max(w0 * cols), k1.min(w1 * cols));
            if lo >= hi {
                return;
            }
            // SAFETY: rows i0..i1 of partial p belong to this task alone.
            let c = unsafe { std::slice::from_raw_parts_mut(pptr.0.add(p * m * n + i0 * n), (i1 - i0) * n) };
            let mut kk = lo;
            while kk < hi {
                let (img, t0) = (kk / cols, kk % cols);
                let len = (cols - t0).min(hi - kk);
                let im = &images[img - w0];
                gemm_arows(
                    i1 - i0,
                    n,
                    len,
                    &im.rows[i0..i1],
                    t0,
                    &im.grad[t0 * NR..],
                    b_stride,
                    c,
                    n,
                    kk != k0,
                );
                kk += len;
            }
        };
        if 2 * m * n * k >= super::gemm::PAR_FLOPS && tiles * active > 1 {
            (0..tiles * active).into_par_iter().for_each(task);
        } else {
            (0..tiles * active).for_each(task);
        }
        for im in images {
            super::recycle(im.grad);
            if let Some(sr) = im.sr {
                super::recycle(sr.buf);
            }
        }
    }
    // The partials, added in order into the first (row by row), then transposed: gw[j][i] = dWᵀ[i][j].
    let (sum, rest) = partial.split_at_mut(m * n);
    let add = |(i, row): (usize, &mut [f32])| {
        for p in 0..parts - 1 {
            for (v, &w) in row.iter_mut().zip(&rest[p * m * n + i * n..][..n]) {
                *v += w;
            }
        }
    };
    if parts * m * n >= super::PAR_MIN {
        sum.par_chunks_mut(n).enumerate().for_each(add);
    } else {
        sum.chunks_mut(n).enumerate().for_each(add);
    }
    let sum = &*sum;
    let transpose = |(j, row): (usize, &mut [f32])| {
        for (i, v) in row.iter_mut().enumerate() {
            *v = sum[i * n + j];
        }
    };
    if m * n >= super::PAR_MIN {
        gw.par_chunks_mut(m).enumerate().for_each(transpose);
    } else {
        gw.chunks_mut(m).enumerate().for_each(transpose);
    }
    super::recycle(partial);
}
