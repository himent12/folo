//! conv2d forward and backward: implicit-im2col GEMM, with direct paths for 1x1 and depthwise convolutions.
//!
//! - `mod.rs`: the shared geometry (`ConvGeom`) and the forward entry points that pick a path.
//! - `shifted.rs`: the input zero-padded and split into its stride phases once, read in place as GEMM rows.
//! - `im2col.rs`: the packed B operands read from the input planes (strided and accumulating convs).
//! - `small_c.rs`: the direct forward for few input channels (the 3-channel stem).
//! - `depthwise.rs`: depthwise convs (groups = channels), forward and gradients.
//! - `winograd.rs`: Winograd F(2x2, 3x3) for dense 3x3 stride-1 inference convs.
//! - `sites.rs`: inference convs evaluated only at some output sites (`Backend::conv2d_act_at`).
//! - `backward_input.rs`, `backward_weight.rs`: the input and weight gradients.
//! - `kernel_tests.rs`: unit tests of these paths against an `f64` reference.

mod backward_input;
mod backward_weight;
mod depthwise;
mod im2col;
#[cfg(test)]
mod kernel_tests;
mod shifted;
mod sites;
mod small_c;
mod winograd;

use rayon::prelude::*;

use super::gemm::{BiasAct, RowPtrs, Strided, gemm_ex, gemm_rows};
use crate::backend::Conv2dParams;
use crate::cpu::par_chunks;
use depthwise::depthwise_forward;
use im2col::{Im2Col, PlaneRows};
use shifted::{ShiftedRows, shifted_scratch};
pub(crate) use sites::{pointwise_forward_at, winograd_tiles_at};
use small_c::{small_c_forward, uses_small_c};
#[cfg(test)]
pub(crate) use winograd::calls as winograd_calls;
pub(crate) use winograd::{transform_weights, uses_winograd, winograd_forward, winograd_forward_at};

pub(crate) use backward_input::{backward_input_overwrites, conv2d_backward_input};
pub(crate) use backward_weight::conv2d_backward_weight;

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

/// Bias (one per output channel) and activation applied to every output element once, as it is finished
/// (conv2d fuses bias + activation here).
pub(crate) type ChannelEpilogue<'a> = BiasAct<'a>;

/// Floats of per-image scratch (shifted-rows copies, unfolded gradients) the images of one conv keep live at
/// once, 16 MiB: images run side by side only as far as their scratch fits (a 640 px batch of 16 would otherwise
/// hold sixteen 15 MB unfolded gradients at once, and the buffer cache keep them). Measured on the folo-n train
/// step (batch 16): 16 MiB holds peak memory at `main`'s (one large image at a time) and is faster than no limit at
/// 320 and 640 px (less scratch to fault in and keep in cache); 64 MiB keeps most of the extra memory.
const IMAGE_SCRATCH_BUDGET: usize = 1 << 22;

static SCRATCH_BUDGET: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(IMAGE_SCRATCH_BUDGET);

/// Overrides [`IMAGE_SCRATCH_BUDGET`] (tests: results must not depend on it).
#[cfg(test)]
pub(crate) fn set_image_scratch_budget(floats: usize) {
    SCRATCH_BUDGET.store(floats, std::sync::atomic::Ordering::Relaxed);
}

/// How many of `n` images whose scratch is `scratch` floats each may run at once (at least one).
fn image_lanes(n: usize, scratch: usize) -> usize {
    let budget = SCRATCH_BUDGET.load(std::sync::atomic::Ordering::Relaxed);
    (budget / scratch.max(1)).clamp(1, n.max(1))
}

/// Run `f(img, out_img, threads)` for every image: one image with the whole pool, several side by side with
/// the pool split between them (`threads / lanes` each, at least one; work stealing spreads the GEMM tiles of
/// a batch smaller than the pool). Images side by side keep each GEMM nearer one core's efficiency than one
/// image at a time across every thread. At most [`image_lanes`] images (by the `scratch` floats `f` allocates
/// per image) run at once, each lane taking its images one after another. Results do not depend on the split.
fn for_each_image(
    out: &mut [f32],
    per_image: usize,
    n: usize,
    scratch: usize,
    f: impl Fn(usize, &mut [f32], usize) + Sync,
) {
    let threads = rayon::current_num_threads();
    if per_image == 0 {
        return;
    }
    let lanes = image_lanes(n, scratch);
    if n == 1 {
        f(0, out, threads);
    } else if lanes == n {
        let share = threads.div_ceil(n).max(1);
        par_chunks(out, per_image, &|img, o| f(img, o, share));
    } else {
        let share = threads.div_ceil(lanes).max(1);
        let mut work: Vec<Vec<(usize, &mut [f32])>> = (0..lanes).map(|_| Vec::new()).collect();
        for (img, o) in out.chunks_mut(per_image).enumerate() {
            work[img % lanes].push((img, o));
        }
        work.into_par_iter().for_each(|lane| lane.into_iter().for_each(|(img, o)| f(img, o, share)));
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
    let scratch = if !accumulate && !g.is_pointwise() { shifted_scratch(g) } else { 0 };
    for_each_image(out, g.o * ohw, g.n, scratch, |img, out_img, threads| {
        let planes = x.planes(img, hw);
        for grp in 0..g.p.groups {
            let a = Strided::lhs(wt, grp * og * kdim, kdim, 1);
            let c = &mut out_img[grp * og * ohw..(grp + 1) * og * ohw];
            let planes = &planes[grp * cg..(grp + 1) * cg];
            let rows = epi.map(|e| e.rows_from(grp * og));
            // Pointwise convs pack B (a plain copy); every other conv reads shifted rows in place.
            if !accumulate && !g.is_pointwise() {
                let sr = ShiftedRows::new(planes, g, threads > 1);
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

/// Output columns `ox` with `0 <= ox * s + d < w`, as a range.
#[inline]
fn valid_cols(ow: usize, s: isize, d: isize, w: isize) -> std::ops::Range<usize> {
    let lo = if d >= 0 { 0 } else { ((-d) + s - 1) / s } as usize;
    let hi = if d >= w { 0 } else { ((w - d) + s - 1) / s } as usize;
    let (lo, hi) = (lo.min(ow), hi.min(ow));
    lo..hi.max(lo)
}
