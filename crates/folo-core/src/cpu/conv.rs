//! conv2d forward and backward: implicit-im2col GEMM, with direct paths for 1x1 and depthwise convolutions.

use super::elementwise::SendPtr;
use super::gemm::{
    BiasAct, ColumnMap, NR, Operand, RowPtrs, Strided, gemm, gemm_ex, gemm_rows, pack_rows_into_panels,
};
use super::par;
use crate::backend::Conv2dParams;
#[cfg(any(target_arch = "x86_64", test))]
use crate::backend::UnaryOp;

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

/// Copy one output row segment of the unfolded input: input row `iy` of `plane` (one `h x w` input plane),
/// output columns `ox0..ox0 + len`, kernel column offset `dx` (already includes `-pad`). Writes
/// `dst[t * stride]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn im2col_segment(
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
struct Im2Col<'a> {
    planes: &'a [&'a [f32]],
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
struct PlaneRows<'a> {
    planes: &'a [&'a [f32]],
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

/// One part of a conv input: `c` channels whose plane `(img, ci)` is `data[offset + img * img_stride + ci * h * w..]`
/// (each image's planes contiguous, as in a contiguous tensor or a channel slice of one).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvPart<'a> {
    pub data: &'a [f32],
    pub offset: usize,
    pub img_stride: usize,
    pub c: usize,
}

impl<'a> ConvPart<'a> {
    /// A contiguous `[N, c, H, W]` buffer.
    pub fn contiguous(data: &'a [f32], c: usize, hw: usize) -> Self {
        ConvPart { data, offset: 0, img_stride: c * hw, c }
    }
}

/// The input of a forward conv: the channel concatenation of `parts` (one part is the ordinary case). Reading
/// the planes in place lets a conv over `cat(parts, 1)` run without the concatenation ever being written.
#[derive(Clone, Copy)]
pub(crate) struct ConvInput<'a> {
    pub parts: &'a [ConvPart<'a>],
}

impl<'a> ConvInput<'a> {
    /// The `h x w` planes of image `img`, one per input channel.
    fn planes(&self, img: usize, hw: usize) -> Vec<&'a [f32]> {
        let mut v = Vec::with_capacity(self.parts.iter().map(|p| p.c).sum());
        for p in self.parts {
            let base = p.offset + img * p.img_stride;
            v.extend((0..p.c).map(|ci| &p.data[base + ci * hw..base + (ci + 1) * hw]));
        }
        v
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
struct ShiftedRows {
    /// The phase planes, with slack after the last one.
    buf: Vec<f32>,
    /// Row `k = (ci, ky, kx)` of B.
    rows: Vec<*const f32>,
    /// Columns of B: `oh · wq`.
    n: usize,
    map: ColumnMap,
}

impl ShiftedRows {
    fn new(planes: &[&[f32]], g: &ConvGeom, threads: usize) -> Self {
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
        let fill = |ci: usize, dst: &mut [f32]| {
            let src = planes[ci];
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
        let threads = if body.len() >= SHIFTED_PAR_MIN { threads } else { 1 };
        let per_task = SHIFTED_TASK.div_ceil(per_ci);
        par::for_each_chunk_mut(body, per_task * per_ci, threads, |t, chunk| {
            chunk.chunks_mut(per_ci).enumerate().for_each(|(c, dst)| fill(t * per_task + c, dst))
        });
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

/// Above this many flops per image, one image's GEMM is parallelised; below it, images run in parallel.
const IMAGE_PAR_FLOPS: usize = 1 << 25;

/// Bias (one per output channel) and activation applied to every output element once, as it is finished
/// (conv2d fuses bias + activation here).
pub(crate) type ChannelEpilogue<'a> = BiasAct<'a>;

/// Run `f(img, out_img, threads)` for every image: one after another with the whole pool for large images, side
/// by side otherwise, splitting the pool between the images when there are fewer images than threads.
fn for_each_image(
    out: &mut [f32],
    per_image: usize,
    n: usize,
    flops: usize,
    f: impl Fn(usize, &mut [f32], usize) + Sync,
) {
    let threads = par::threads();
    if per_image == 0 {
        return;
    }
    if flops >= IMAGE_PAR_FLOPS || n == 1 || threads == 1 {
        for (img, o) in out.chunks_mut(per_image).enumerate() {
            f(img, o, threads);
        }
    } else {
        par::for_each_chunk_mut(out, per_image, threads, |img, o| f(img, o, 1));
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
    let part = [ConvPart::contiguous(x, g.c, g.h * g.w)];
    conv2d_forward_parts(ConvInput { parts: &part }, wt, out, g, accumulate, epi);
}

/// [`conv2d_forward`] over the channel concatenation of the parts of `x` (their channel counts add up to
/// `g.c`), reading each part in place.
pub(crate) fn conv2d_forward_parts(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    debug_assert_eq!(x.parts.iter().map(|p| p.c).sum::<usize>(), g.c);
    if g.is_depthwise() {
        return depthwise_forward(x, wt, out, g, accumulate, epi);
    }
    if !accumulate && uses_small_c(g) {
        return small_c_forward(x, wt, out, g, epi);
    }
    let (cg, og, kdim, ohw, hw) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
    let per_image_flops = 2 * g.o * ohw * kdim;
    for_each_image(out, g.o * ohw, g.n, per_image_flops, |img, out_img, threads| {
        let planes = x.planes(img, hw);
        for grp in 0..g.p.groups {
            let a = Strided::lhs(wt, grp * og * kdim, kdim, 1);
            let c = &mut out_img[grp * og * ohw..(grp + 1) * og * ohw];
            let planes = &planes[grp * cg..(grp + 1) * cg];
            let rows = epi.map(|e| e.rows_from(grp * og));
            // Pointwise convs pack B (a plain copy); every other conv reads shifted rows in place.
            if !accumulate && !g.is_pointwise() {
                let sr = ShiftedRows::new(planes, g, threads);
                let b = RowPtrs { rows: &sr.rows, n: sr.n };
                gemm_rows(og, kdim, &a, &b, c, sr.map, threads, rows);
                super::recycle(sr.buf);
            } else if g.is_pointwise() {
                let b = PlaneRows { planes };
                gemm_ex(og, ohw, kdim, &a, &b, c, ohw, accumulate, threads, rows);
            } else {
                let b = Im2Col { planes, g: *g };
                gemm_ex(og, ohw, kdim, &a, &b, c, ohw, accumulate, threads, rows);
            }
        }
    });
}

/// Input channels up to which a (non-pointwise, ungrouped) forward conv runs [`small_c_forward`] instead of
/// the shifted-rows GEMM: the stem's 3-channel image input, whose GEMM has a K of only 27 and whose
/// shifted-rows copy is filled by one task per input channel (three).
const SMALL_C: usize = 4;
/// Output columns per register tile of [`small_c_forward`] (four AVX2 vectors).
const SMALL_C_TILE: usize = 32;
/// The deepest K [`small_c_forward`] takes: within one K block of the GEMM (256), so both accumulate every
/// output in the same order and agree bit for bit.
const SMALL_C_MAX_K: usize = 256;
/// Below this many flops [`small_c_forward`] runs on the calling thread (the GEMM's threshold).
const SMALL_C_PAR_FLOPS: usize = 1 << 21;

/// Whether [`conv2d_forward_parts`] (overwriting) takes [`small_c_forward`].
fn uses_small_c(g: &ConvGeom) -> bool {
    g.p.groups == 1 && g.c <= SMALL_C && !g.is_pointwise() && g.ohw() > 0 && g.kdim() <= SMALL_C_MAX_K
}

/// Direct forward conv for few input channels (see [`SMALL_C`]), overwriting `out`, in parallel over bands of
/// output rows (of all images). Per output row it gathers the `K = c·kh·kw` input rows the taps read (strided
/// and zero-padded, `K x ow` floats, in L1), then every output channel is `K` multiply-adds per pixel from zero
/// in `(ci, ky, kx)` order, in register tiles of [`SMALL_C_TILE`] columns. That is the order and the
/// arithmetic (fused multiply-add with AVX2 and FMA, else a multiply then an add) of the GEMM micro-kernels on
/// one K block, so the result is bit for bit the GEMM path's. `epi` sees each finished output row.
fn small_c_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let (hw, ohw, k) = (g.h * g.w, g.ohw(), g.kdim());
    let owp = g.ow.div_ceil(SMALL_C_TILE) * SMALL_C_TILE;
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let rows = g.n * g.oh;
    // Up to four output rows (of every channel) per task, and about four tasks per thread when rows allow.
    let threads = par::threads();
    let band = (rows / (4 * threads)).clamp(1, 4);
    let optr = SendPtr(out.as_mut_ptr());
    let task = |t: usize| {
        let optr = &optr;
        let mut gat = vec![0f32; k * owp];
        let mut orow = vec![0f32; owp];
        for r in t * band..((t + 1) * band).min(rows) {
            let (img, oy) = (r / g.oh, r % g.oh);
            small_c_gather(&planes[img], g, oy, owp, &mut gat);
            for oc in 0..g.o {
                let w = &wt[oc * k..(oc + 1) * k];
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 and FMA were detected above.
                    unsafe { small_c_row_avx2(w, &gat, owp, &mut orow) };
                } else {
                    small_c_row(w, &gat, owp, &mut orow);
                }
                #[cfg(not(target_arch = "x86_64"))]
                small_c_row(w, &gat, owp, &mut orow);
                // SAFETY: row `oy` of channel `oc` of image `img` is inside `out` (`[n, o, oh, ow]`), and only
                // this task (which owns output row `r` of every channel) touches it.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(optr.0.add((img * g.o + oc) * ohw + oy * g.ow), g.ow)
                };
                dst.copy_from_slice(&orow[..g.ow]);
                if let Some(e) = epi {
                    e.apply(oc, dst);
                }
            }
        }
    };
    let tasks = rows.div_ceil(band);
    par::for_each(tasks, if 2 * g.n * g.o * ohw * k >= SMALL_C_PAR_FLOPS { threads } else { 1 }, task);
}

/// The `K x owp` taps of output row `oy`: `gat[kk * owp + ox]` is the input element that tap
/// `kk = (ci, ky, kx)` of output pixel `(oy, ox)` reads, zero where it falls in the padding and for `ox >= ow`.
fn small_c_gather(planes: &[&[f32]], g: &ConvGeom, oy: usize, owp: usize, gat: &mut [f32]) {
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    let mut rows = gat.chunks_exact_mut(owp);
    for plane in planes {
        for ky in 0..g.kh {
            let iy = oy as isize * sh - ph + ky as isize * dh;
            let src =
                (0..g.h as isize).contains(&iy).then(|| &plane[iy as usize * g.w..(iy as usize + 1) * g.w]);
            for kx in 0..g.kw {
                let dst = rows.next().expect("one gathered row per tap");
                let Some(src) = src else {
                    dst.fill(0.0);
                    continue;
                };
                let d = kx as isize * dw - pw;
                let r = valid_cols(g.ow, sw, d, g.w as isize);
                dst[..r.start].fill(0.0);
                dst[r.end..].fill(0.0);
                if r.is_empty() {
                    // The tap sees only padding (it starts right of the row, or ends left of it).
                } else if sw == 1 {
                    let s0 = (r.start as isize + d) as usize;
                    dst[r.clone()].copy_from_slice(&src[s0..s0 + r.len()]);
                } else {
                    for ox in r {
                        dst[ox] = src[(ox as isize * sw + d) as usize];
                    }
                }
            }
        }
    }
}

/// One output row of one channel from the gathered taps: `orow[ox] = Σ_kk w[kk] · gat[kk][ox]`, a multiply
/// then an add from zero in `kk` order (the portable GEMM micro-kernel's arithmetic).
fn small_c_row(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    for (t, o) in orow.as_chunks_mut::<SMALL_C_TILE>().0.iter_mut().enumerate() {
        let mut acc = [0f32; SMALL_C_TILE];
        for (kk, &wv) in w.iter().enumerate() {
            let src = &gat[kk * owp + t * SMALL_C_TILE..][..SMALL_C_TILE];
            for (a, &v) in acc.iter_mut().zip(src) {
                *a += wv * v;
            }
        }
        *o = acc;
    }
}

/// [`small_c_row`] with AVX2 fused multiply-adds (the AVX2 GEMM micro-kernels' arithmetic): four 8-wide
/// accumulators per tile.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn small_c_row_avx2(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(gat.len() >= w.len() * owp && orow.len() == owp && owp.is_multiple_of(SMALL_C_TILE));
    for x0 in (0..owp).step_by(SMALL_C_TILE) {
        let (mut c0, mut c1, mut c2, mut c3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        for (kk, wv) in w.iter().enumerate() {
            // SAFETY: row `kk` of `gat` holds `owp` floats and `x0 + SMALL_C_TILE <= owp` (asserted above).
            unsafe {
                let p = gat.as_ptr().add(kk * owp + x0);
                let a = _mm256_broadcast_ss(wv);
                c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p), c0);
                c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(8)), c1);
                c2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(16)), c2);
                c3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(24)), c3);
            }
        }
        // SAFETY: `orow` holds `owp >= x0 + SMALL_C_TILE` floats.
        unsafe {
            let o = orow.as_mut_ptr().add(x0);
            _mm256_storeu_ps(o, c0);
            _mm256_storeu_ps(o.add(8), c1);
            _mm256_storeu_ps(o.add(16), c2);
            _mm256_storeu_ps(o.add(24), c3);
        }
    }
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
    let block = |b: usize, dst: &mut [f32]| {
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
    let threads = if flipped.len() >= super::PAR_MIN { par::threads() } else { 1 };
    par::for_each_chunk_mut(&mut flipped, 8 * og * khw, threads, block);
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
            col2im(&col, &mut gx_img[grp * cg * hw..(grp + 1) * cg * hw], g, threads);
        }
        super::recycle(col);
    });
}

/// Accumulate the unfolded gradient `col` (`[cg*kh*kw, ohw]`) into the input planes of one group.
fn col2im(col: &[f32], gx: &mut [f32], g: &ConvGeom, threads: usize) {
    let (khw, ohw, hw) = (g.kh * g.kw, g.ohw(), g.h * g.w);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    let (hi, wi) = (g.h as isize, g.w as isize);
    let plane = |ci: usize, gxp: &mut [f32]| {
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
    par::for_each_chunk_mut(gx, hw, threads, plane);
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

/// One output plane of a depthwise conv: `op += conv(xp, wk)` (`op` is `oh x ow`, `xp` one `h x w` input plane).
#[inline(always)]
fn depthwise_plane(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
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
}

/// [`depthwise_plane`] compiled for AVX2 (8-wide rows).
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_plane_avx2(xp: &[f32], wk: &[f32], op: &mut [f32], g: &ConvGeom) {
    depthwise_plane(xp, wk, op, g)
}

/// Output elements per [`depthwise_forward`] task (whole planes, at least one): small planes are grouped so a
/// task is not dominated by scheduling.
const DW_TASK_OUTPUTS: usize = 1 << 14;

fn depthwise_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(target_arch = "x86_64")]
    let fast = avx2 && g.p.stride == [1, 1];
    let total = g.n * g.o;
    let threads = par::threads();
    let per_task = DW_TASK_OUTPUTS.div_ceil(ohw).min(total.div_ceil(4 * threads)).max(1);
    par::for_each_chunk_mut(out, per_task * ohw, threads, |t, ops| {
        #[cfg(target_arch = "x86_64")]
        let mut pad = Vec::new();
        for (p, op) in ops.chunks_mut(ohw).enumerate() {
            let plane = t * per_task + p;
            let (img, oc) = (plane / g.o, plane % g.o);
            let xp = planes[img][oc / mult];
            let wk = &wt[oc * khw..(oc + 1) * khw];
            #[cfg(target_arch = "x86_64")]
            if fast {
                // SAFETY: AVX2 and FMA were detected above.
                unsafe {
                    depthwise_s1_avx2(xp, wk, op, g, accumulate, epi.map(|e| e.rows_from(oc)), &mut pad)
                };
                continue;
            }
            if !accumulate {
                op.fill(0.0);
            }
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                // SAFETY: AVX2 and FMA were detected above.
                unsafe { depthwise_plane_avx2(xp, wk, op, g) };
            } else {
                depthwise_plane(xp, wk, op, g);
            }
            #[cfg(not(target_arch = "x86_64"))]
            depthwise_plane(xp, wk, op, g);
            if let Some(e) = epi {
                e.apply(oc, op);
            }
        }
    });
}

/// One output plane of a stride-1 depthwise conv with AVX2: `op = conv(xp, wk)` (plus the previous `op` if
/// `accumulate`), then `epi` (its row 0 is this channel). The input plane is first copied into `pad` with its
/// zero padding (and slack up to a whole number of vectors), so every output vector is the `kh·kw` taps'
/// fused multiply-adds, in `(ky, kx)` order, of plain unaligned loads, held in registers four vectors at a
/// time; the identity and SiLU epilogues are applied there too (the lanes of `silu_bias_inplace`), others as a
/// pass over the finished plane.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn depthwise_s1_avx2(
    xp: &[f32],
    wk: &[f32],
    op: &mut [f32],
    g: &ConvGeom,
    accumulate: bool,
    epi: Option<BiasAct<'_>>,
    pad: &mut Vec<f32>,
) {
    use std::arch::x86_64::*;
    let (ph, pw) = (g.p.padding[0], g.p.padding[1]);
    let (dh, dw) = (g.p.dilation[0], g.p.dilation[1]);
    let (h, w, ow) = (g.h, g.w, g.ow);
    let owp = ow.div_ceil(8) * 8;
    // Padded row: `prow[X] = x[X - pw]`, for every column a tap of a computed output column reads.
    let wpd = (owp + dw * (g.kw - 1)).max(pw + w);
    pad.clear();
    pad.resize(h * wpd, 0.0);
    // A zero-width input (all padding) has nothing to copy.
    if w > 0 {
        for (prow, xrow) in pad.chunks_exact_mut(wpd).zip(xp.chunks_exact(w)) {
            prow[pw..pw + w].copy_from_slice(xrow);
        }
    }
    // In registers: bias + SiLU / bias; otherwise a pass after the plane.
    let (reg, bias) = match epi {
        Some(e) if matches!(e.act, None | Some(UnaryOp::Silu)) => (true, e.bias.map_or(0.0, |b| b[0])),
        _ => (false, 0.0),
    };
    let silu = epi.is_some_and(|e| e.act == Some(UnaryOp::Silu));
    // Without an activation `bias_act_inplace` skips a zero bias; `x + -0.0` is `x` bit for bit.
    let bv = _mm256_set1_ps(if !silu && bias == 0.0 { -0.0 } else { bias });
    for oy in 0..g.oh {
        let orow = &mut op[oy * ow..(oy + 1) * ow];
        let mut x0 = 0;
        while x0 < owp {
            let nv = ((owp - x0) / 8).min(4);
            let mut acc = [_mm256_setzero_ps(); 4];
            if accumulate {
                for (v, a) in acc.iter_mut().enumerate().take(nv) {
                    let c = x0 + 8 * v;
                    let mut tmp = [0f32; 8];
                    let len = ow.saturating_sub(c).min(8);
                    tmp[..len].copy_from_slice(&orow[c..c + len]);
                    // SAFETY: `tmp` holds 8 floats.
                    *a = unsafe { _mm256_loadu_ps(tmp.as_ptr()) };
                }
            }
            for ky in 0..g.kh {
                let iy = (oy + ky * dh) as isize - ph as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                let prow = &pad[iy as usize * wpd..(iy as usize + 1) * wpd];
                for kx in 0..g.kw {
                    let wv = _mm256_set1_ps(wk[ky * g.kw + kx]);
                    let base = x0 + kx * dw;
                    assert!(base + 8 * nv <= wpd);
                    for (v, a) in acc.iter_mut().enumerate().take(nv) {
                        // SAFETY: `base + 8 * nv <= wpd` (asserted), inside this padded row.
                        *a = _mm256_fmadd_ps(
                            wv,
                            unsafe { _mm256_loadu_ps(prow.as_ptr().add(base + 8 * v)) },
                            *a,
                        );
                    }
                }
            }
            for (v, a) in acc.iter().enumerate().take(nv) {
                let c = x0 + 8 * v;
                let r = match (reg, silu) {
                    (false, _) => *a,
                    (true, false) => _mm256_add_ps(*a, bv),
                    (true, true) => super::elementwise::silu8(_mm256_add_ps(*a, bv)),
                };
                if c + 8 <= ow {
                    // SAFETY: columns c..c + 8 are inside `orow`.
                    unsafe { _mm256_storeu_ps(orow.as_mut_ptr().add(c), r) };
                } else if c < ow {
                    let mut tmp = [0f32; 8];
                    // SAFETY: `tmp` holds 8 floats.
                    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), r) };
                    orow[c..].copy_from_slice(&tmp[..ow - c]);
                }
            }
            x0 += 8 * nv;
        }
    }
    if let (false, Some(e)) = (reg, epi) {
        e.apply(0, op);
    }
}

fn depthwise_backward_input(gy: &[f32], wt: &[f32], gx: &mut [f32], g: &ConvGeom) {
    let mult = g.o / g.c;
    let (hw, ohw, khw) = (g.h * g.w, g.ohw(), g.kh * g.kw);
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    par::for_each_chunk_mut(gx, hw, par::threads(), |plane, gxp| {
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
    par::for_each_chunk_mut(gw, khw, par::threads(), |oc, gk| {
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

#[cfg(test)]
mod small_c_tests {
    use super::*;

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    fn geom(n: usize, c: usize, hw: [usize; 2], o: usize, k: [usize; 2], p: Conv2dParams) -> ConvGeom {
        let out =
            |i: usize, k: usize| (hw[i] + 2 * p.padding[i] - p.dilation[i] * (k - 1) - 1) / p.stride[i] + 1;
        let (oh, ow) = (out(0, k[0]), out(1, k[1]));
        ConvGeom { n, c, h: hw[0], w: hw[1], o, kh: k[0], kw: k[1], oh, ow, p }
    }

    /// The shifted-rows GEMM path [`small_c_forward`] replaces (what [`conv2d_forward_parts`] ran before).
    fn gemm_path(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
        let (kdim, ohw, hw) = (g.kdim(), g.ohw(), g.h * g.w);
        let part = [ConvPart::contiguous(x, g.c, hw)];
        let x = ConvInput { parts: &part };
        for (img, out_img) in out.chunks_mut(g.o * ohw).enumerate() {
            let planes = x.planes(img, hw);
            let a = Strided::lhs(wt, 0, kdim, 1);
            let sr = ShiftedRows::new(&planes, g, par::threads());
            let b = RowPtrs { rows: &sr.rows, n: sr.n };
            gemm_rows(g.o, kdim, &a, &b, out_img, sr.map, par::threads(), epi);
        }
    }

    #[test]
    fn small_c_matches_the_gemm_path_bit_for_bit() {
        let p = |s: [usize; 2], pad: [usize; 2], d: [usize; 2]| Conv2dParams {
            stride: s,
            padding: pad,
            dilation: d,
            groups: 1,
        };
        // (n, c, [h, w], o, [kh, kw], params): the folo stem at 640 and 320, then odd shapes, strides,
        // paddings, dilations and channel counts 1..=4.
        let cases = [
            (1, 3, [640, 640], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (2, 3, [320, 384], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (1, 1, [37, 29], 5, [3, 3], p([1, 1], [1, 1], [1, 1])),
            (3, 2, [19, 23], 7, [5, 3], p([2, 1], [2, 0], [1, 2])),
            (1, 4, [33, 65], 9, [3, 3], p([3, 2], [0, 1], [1, 1])),
            (2, 3, [8, 8], 4, [3, 3], p([1, 1], [3, 3], [1, 1])),
            (1, 3, [70, 70], 33, [7, 7], p([2, 2], [3, 3], [1, 1])),
            (1, 3, [5, 200], 2, [1, 3], p([1, 4], [0, 1], [1, 1])),
            // Taps that see only padding on the left or the right of a row (and above / below it).
            (1, 2, [1, 1], 2, [5, 5], p([1, 1], [2, 2], [1, 1])),
            (2, 1, [4, 3], 3, [5, 7], p([1, 1], [2, 3], [1, 1])),
            (1, 3, [3, 2], 2, [3, 5], p([2, 2], [1, 3], [1, 1])),
        ];
        for (i, (n, c, hw, o, k, prm)) in cases.into_iter().enumerate() {
            let g = geom(n, c, hw, o, k, prm);
            assert!(uses_small_c(&g), "case {i}");
            let x = data(n * c * hw[0] * hw[1], 1 + i as u32);
            let wt = data(o * g.kdim(), 100 + i as u32);
            // Channel 4's bias is 0.0.
            let bias: Vec<f32> = (0..o).map(|c| c as f32 * 0.25 - 1.0).collect();
            let b = Some(&bias[..]);
            // The GEMM applies the identity and SiLU in its micro-kernels, ReLU as a pass over finished rows.
            let epis = [
                None,
                Some(BiasAct { bias: None, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: None }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Relu) }),
            ];
            for epi in epis {
                let mut want = vec![f32::NAN; n * o * g.ohw()];
                gemm_path(&x, &wt, &mut want, &g, epi);
                let mut got = vec![f32::NAN; want.len()];
                conv2d_forward(&x, &wt, &mut got, &g, false, epi);
                let same = got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "case {i} ({g:?}): not bit-identical with the GEMM path");
            }
        }
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    fn geom(n: usize, c: usize, hw: [usize; 2], o: usize, k: [usize; 2], p: Conv2dParams) -> ConvGeom {
        let out =
            |i: usize, k: usize| (hw[i] + 2 * p.padding[i] - p.dilation[i] * (k - 1) - 1) / p.stride[i] + 1;
        let (oh, ow) = (out(0, k[0]), out(1, k[1]));
        ConvGeom { n, c, h: hw[0], w: hw[1], o, kh: k[0], kw: k[1], oh, ow, p }
    }

    fn params(s: [usize; 2], pad: [usize; 2], d: [usize; 2], groups: usize) -> Conv2dParams {
        Conv2dParams { stride: s, padding: pad, dilation: d, groups }
    }

    /// `(n, c, [h, w], o, [kh, kw], params)`.
    type Case = (usize, usize, [usize; 2], usize, [usize; 2], Conv2dParams);

    /// The conv in f64, per output element: `(sum, sum of |terms|)`.
    fn reference(x: &[f32], wt: &[f32], g: &ConvGeom) -> Vec<(f64, f64)> {
        let (cg, og) = (g.cg(), g.og());
        let mut out = Vec::with_capacity(g.n * g.o * g.ohw());
        for img in 0..g.n {
            for oc in 0..g.o {
                let grp = oc / og;
                for oy in 0..g.oh {
                    for ox in 0..g.ow {
                        let (mut s, mut a) = (0f64, 0f64);
                        for ci in 0..cg {
                            let ic = grp * cg + ci;
                            for ky in 0..g.kh {
                                for kx in 0..g.kw {
                                    let iy = (oy * g.p.stride[0] + ky * g.p.dilation[0]) as isize
                                        - g.p.padding[0] as isize;
                                    let ix = (ox * g.p.stride[1] + kx * g.p.dilation[1]) as isize
                                        - g.p.padding[1] as isize;
                                    if iy < 0 || ix < 0 || iy >= g.h as isize || ix >= g.w as isize {
                                        continue;
                                    }
                                    let xv =
                                        x[((img * g.c + ic) * g.h + iy as usize) * g.w + ix as usize] as f64;
                                    let wv = wt[((oc * cg + ci) * g.kh + ky) * g.kw + kx] as f64;
                                    s += xv * wv;
                                    a += (xv * wv).abs();
                                }
                            }
                        }
                        out.push((s, a));
                    }
                }
            }
        }
        out
    }

    /// Runs `conv2d_forward` on every case with every epilogue (and accumulating into a prior output), against
    /// the f64 reference. Tolerance: the f32 summation bound `K · 2^-23 · Σ|terms|` for a K-term dot product in
    /// any order, carried through the activation (slope at most 1.1), plus SiLU's own relative error.
    fn check(cases: &[Case]) {
        let bias_of = |o: usize| (0..o).map(|c| c as f32 * 0.25 - 1.0).collect::<Vec<f32>>();
        let silu = |v: f64| v / (1.0 + (-v).exp());
        for (i, &(n, c, hw, o, k, prm)) in cases.iter().enumerate() {
            let g = geom(n, c, hw, o, k, prm);
            let x = data(n * c * hw[0] * hw[1], 7 + i as u32);
            let wt = data(o * g.kdim(), 300 + i as u32);
            let want = reference(&x, &wt, &g);
            let bias = bias_of(o);
            let ohw = g.ohw();
            let kterms = g.kdim() as f64;
            for (e, act) in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)].into_iter().enumerate() {
                for with_bias in [false, true] {
                    let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                    let epi = (with_bias || act.is_some()).then_some(epi);
                    let mut got = vec![f32::NAN; n * o * ohw];
                    conv2d_forward(&x, &wt, &mut got, &g, false, epi);
                    for (q, (&y, &(s, a))) in got.iter().zip(&want).enumerate() {
                        let b = if with_bias { bias[(q / ohw) % o] as f64 } else { 0.0 };
                        let t = s + b;
                        let (want, slope) = match act {
                            None => (t, 1.0),
                            Some(UnaryOp::Silu) => (silu(t), 1.1),
                            _ => (t.max(0.0), 1.0),
                        };
                        let tol = slope * kterms * 1.2e-7 * (a + b.abs()) + 5e-7 * want.abs() + 1e-7;
                        assert!(
                            (y as f64 - want).abs() <= tol,
                            "case {i} ({g:?}) epi {e}/{with_bias} at {q}: {y} vs {want} (tol {tol})"
                        );
                    }
                }
            }
            // Accumulating into a prior output (no epilogue).
            let prior = data(n * o * ohw, 900 + i as u32);
            let mut got = prior.clone();
            conv2d_forward(&x, &wt, &mut got, &g, true, None);
            for (q, ((&y, &(s, a)), &p0)) in got.iter().zip(&want).zip(&prior).enumerate() {
                let want = s + p0 as f64;
                let tol = (kterms + 1.0) * 1.2e-7 * (a + (p0 as f64).abs()) + 1e-7;
                assert!((y as f64 - want).abs() <= tol, "case {i} accumulate at {q}: {y} vs {want}");
            }
        }
    }

    #[test]
    fn depthwise_matches_reference() {
        // (n, c, [h, w], o, [kh, kw], params): the folo-n shapes, then rows narrower than one vector, ragged
        // tails (ow % 8 != 0, more than four vectors), taps that see only padding, dilation, a channel
        // multiplier, non-square kernels and the strided (non-fast) path.
        let cases = [
            (1, 64, [80, 80], 64, [3, 3], params([1, 1], [1, 1], [1, 1], 64)),
            (1, 256, [20, 20], 256, [3, 3], params([1, 1], [1, 1], [1, 1], 256)),
            (2, 3, [1, 1], 3, [3, 3], params([1, 1], [1, 1], [1, 1], 3)),
            (1, 2, [5, 7], 2, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
            (1, 3, [9, 37], 3, [3, 3], params([1, 1], [1, 1], [1, 1], 3)),
            (1, 2, [6, 41], 2, [5, 5], params([1, 1], [2, 2], [1, 1], 2)),
            (1, 2, [4, 3], 2, [5, 7], params([1, 1], [3, 4], [1, 1], 2)),
            (1, 3, [11, 19], 3, [3, 3], params([1, 1], [2, 2], [2, 2], 3)),
            (2, 2, [8, 9], 4, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
            (1, 4, [7, 13], 4, [1, 5], params([1, 1], [0, 2], [1, 1], 4)),
            (1, 3, [10, 10], 3, [3, 3], params([1, 1], [0, 0], [1, 1], 3)),
            (1, 3, [12, 17], 3, [3, 3], params([2, 2], [1, 1], [1, 1], 3)),
            (1, 2, [9, 8], 4, [3, 3], params([2, 1], [1, 1], [1, 1], 2)),
            // Zero-width inputs: every output is padding (then bias and activation).
            (1, 2, [3, 0], 2, [1, 1], params([1, 1], [0, 1], [1, 1], 2)),
            (2, 2, [3, 0], 4, [3, 3], params([1, 1], [1, 2], [1, 1], 2)),
        ];
        check(&cases);
    }

    #[test]
    fn gemm_paths_match_reference() {
        let cases = [
            // 1x1 (packed B, rows read once into panels): ragged M and N, K over one K block (the epilogue
            // runs in the micro-kernel on the last one).
            (1, 20, [7, 9], 13, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
            (2, 300, [5, 7], 7, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
            // Shifted rows, stride 1: ragged tiles, K = 360 over one block, two groups.
            (2, 8, [11, 13], 10, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
            (1, 40, [6, 7], 8, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
            (1, 8, [9, 9], 6, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
            // Stride-2 phases (both column phases dealt out together), odd sizes, no / wide padding.
            (1, 6, [13, 15], 9, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
            (1, 5, [12, 9], 7, [3, 3], params([2, 2], [0, 0], [1, 1], 1)),
            (1, 6, [10, 11], 5, [5, 5], params([2, 2], [2, 2], [1, 1], 1)),
            // One column phase only (dilation 2 at stride 2), stride 3, and mixed strides.
            (1, 5, [15, 14], 4, [3, 3], params([2, 2], [2, 2], [2, 2], 1)),
            (1, 5, [16, 17], 6, [3, 3], params([3, 3], [1, 1], [1, 1], 1)),
            (1, 6, [12, 13], 5, [3, 3], params([2, 1], [1, 1], [1, 1], 1)),
            (1, 6, [9, 14], 5, [3, 3], params([1, 2], [1, 1], [1, 1], 1)),
            // The few-channel direct path.
            (1, 3, [17, 19], 11, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
        ];
        check(&cases);
    }

    /// The fused paths (shifted rows or a 1x1's planes packed row by row, bias and SiLU applied in the
    /// micro-kernel) are bit for bit an unfused GEMM over an im2col B (packed panel by panel from
    /// `im2col_segment`, the elementwise formula) followed by the row epilogue: the same K blocks, the same FMAs,
    /// the same epilogue lanes. Covers ragged M and N, K over one `KC` block, every phase split and zero biases.
    #[test]
    fn fused_conv_is_bitwise_im2col_gemm_plus_row_epilogue() {
        let cases = [
            (1, 20, [7, 9], 13, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
            (2, 300, [5, 7], 7, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
            (2, 8, [11, 13], 10, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
            (1, 40, [6, 7], 8, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
            (1, 8, [9, 9], 6, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
            (1, 6, [13, 15], 9, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
            (1, 5, [12, 9], 7, [3, 3], params([2, 2], [0, 0], [1, 1], 1)),
            (1, 6, [10, 11], 5, [5, 5], params([2, 2], [2, 2], [1, 1], 1)),
            (1, 5, [15, 14], 4, [3, 3], params([2, 2], [2, 2], [2, 2], 1)),
            (1, 5, [16, 17], 6, [3, 3], params([3, 3], [1, 1], [1, 1], 1)),
            (1, 6, [12, 13], 5, [3, 3], params([2, 1], [1, 1], [1, 1], 1)),
            (1, 6, [9, 14], 5, [3, 3], params([1, 2], [1, 1], [1, 1], 1)),
        ];
        for (i, &(n, c, hw, o, k, prm)) in cases.iter().enumerate() {
            let g = geom(n, c, hw, o, k, prm);
            assert!(!g.is_depthwise() && !uses_small_c(&g));
            let x = data(n * c * hw[0] * hw[1], 40 + i as u32);
            let wt = data(o * g.kdim(), 500 + i as u32);
            // Every third bias is exactly zero.
            let bias: Vec<f32> =
                (0..o).map(|c| if c % 3 == 0 { 0.0 } else { c as f32 * 0.3 - 1.0 }).collect();
            let (cg, og, kdim, ohw, hwi) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
            for act in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)] {
                for with_bias in [false, true] {
                    let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                    let mut got = vec![f32::NAN; n * o * ohw];
                    conv2d_forward(&x, &wt, &mut got, &g, false, Some(epi));
                    let mut want = vec![f32::NAN; n * o * ohw];
                    let part = [ConvPart::contiguous(&x, c, hwi)];
                    let input = ConvInput { parts: &part };
                    for (img, out_img) in want.chunks_mut(o * ohw).enumerate() {
                        let planes = input.planes(img, hwi);
                        for grp in 0..g.p.groups {
                            let a = Strided::lhs(&wt, grp * og * kdim, kdim, 1);
                            let b = Im2Col { planes: &planes[grp * cg..(grp + 1) * cg], g };
                            let cblk = &mut out_img[grp * og * ohw..(grp + 1) * og * ohw];
                            gemm_ex(og, ohw, kdim, &a, &b, cblk, ohw, false, 1, None);
                            for r in 0..og {
                                epi.rows_from(grp * og).apply(r, &mut cblk[r * ohw..(r + 1) * ohw]);
                            }
                        }
                    }
                    let same = got.iter().zip(&want).all(|(p, q)| p.to_bits() == q.to_bits());
                    assert!(same, "case {i} ({g:?}) act {act:?} bias {with_bias}: not bit-identical");
                }
            }
        }
    }
}
