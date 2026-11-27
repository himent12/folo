//! The quantised forward of layers 0–22 (docs/MODEL.md §9.6), written once over an executor so calibration, the
//! fake-quant reference and the int8 plan run the same graph with the same boundaries.
//!
//! Every tensor an [`Exec`] holds as an activation (`E::A`) is a quantised one: the output of a conv's epilogue
//! (activation, then the optional residual add), the stem's output, the input of each attention block's `proj`,
//! and the channel slices, max pools and upsamples of those, which keep their source's grid. What stays `f32`:
//! the network input and the stem's weights, each attention block's `qkv` output, its two matmuls, its softmax
//! and its depthwise `pe`, and the head (P3, P4 and P5 are dequantised for it).

use folo_core::Tensor;
use folo_nn::{ConvBnAct, Result};

use crate::model::FoloModel;
use crate::model::blocks::{Attention, Bottleneck, C2Psa, C3k, C3k2, C3k2Block, PsaBlock, Sppf};

/// The ops of the quantised graph. Names are the conv's `.folo` prefix (`backbone.stage2.cv1`), or that of the
/// `proj` conv plus `.in` for its quantised input; they identify a boundary's range in a [`super::Calibration`].
pub trait Exec {
    /// A quantised activation.
    type A;

    /// The stem: an `f32` conv of the network input, quantised in its epilogue.
    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Self::A>;
    /// `q(act(conv(cat(parts))) + residual)`: a conv over the channel concatenation of `parts`, each part on its
    /// own grid, with the residual (on its own grid) added after the activation.
    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&Self::A],
        res: Option<&Self::A>,
    ) -> Result<Self::A>;
    /// `act(conv(x))` dequantised: `qkv`, whose output stays `f32`.
    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &Self::A) -> Result<Tensor>;
    /// `act(conv(cat(parts)))` directly as `f32`: used for the head-only P5 output (`neck.out5.cv2`) so an int8
    /// conv writes `f32` directly without a `u8` round-trip, and an `f32` conv skips requantisation.
    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Self::A]) -> Result<Tensor> {
        let y = self.conv(name, conv, parts, None)?;
        self.dequantize(&y)
    }
    /// Quantise an `f32` tensor onto the grid of `name` (may be deferred until an int8 consumer reads it).
    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<Self::A>;
    /// The `f32` values of an activation.
    fn dequantize(&mut self, x: &Self::A) -> Result<Tensor>;
    /// Channels `start..start + len` (same grid).
    fn narrow(&mut self, x: &Self::A, start: usize, len: usize) -> Result<Self::A>;
    /// `k × k` max pool, stride 1, padding `k / 2` (same grid).
    fn max_pool(&mut self, x: &Self::A, k: usize) -> Result<Self::A>;
    /// Nearest-neighbour upsample by 2 (same grid).
    fn upsample2(&mut self, x: &Self::A) -> Result<Self::A>;
    /// `[N, C, H, W]` of an activation.
    fn dims(&self, x: &Self::A) -> [usize; 4];
}

fn bottleneck<E: Exec>(e: &mut E, p: &str, b: &Bottleneck, x: &E::A) -> Result<E::A> {
    let y = e.conv(&format!("{p}.cv1"), &b.cv1, &[x], None)?;
    e.conv(&format!("{p}.cv2"), &b.cv2, &[&y], b.residual.then_some(x))
}

fn c3k<E: Exec>(e: &mut E, p: &str, b: &C3k, x: &E::A) -> Result<E::A> {
    let mut a = e.conv(&format!("{p}.cv1"), &b.cv1, &[x], None)?;
    for (j, blk) in b.blocks.iter().enumerate() {
        a = bottleneck(e, &format!("{p}.block{j}"), blk, &a)?;
    }
    let c = e.conv(&format!("{p}.cv2"), &b.cv2, &[x], None)?;
    e.conv(&format!("{p}.cv3"), &b.cv3, &[&a, &c], None)
}

/// [`C3k2::forward_cat`] body: `cv1` over the parts and the block chain on its second half, returning `(y, outs)`
/// for `cv2`.
fn c3k2_parts<E: Exec>(e: &mut E, p: &str, b: &C3k2, xs: &[&E::A]) -> Result<(E::A, Vec<E::A>)> {
    let y = e.conv(&format!("{p}.cv1"), &b.cv1, xs, None)?;
    let c = b.cv1.conv.config().out_channels / 2;
    let mut outs: Vec<E::A> = Vec::with_capacity(b.blocks.len());
    for (j, blk) in b.blocks.iter().enumerate() {
        let bp = format!("{p}.block{j}");
        let first;
        let src = match outs.last() {
            Some(prev) => prev,
            None => {
                first = e.narrow(&y, c, c)?;
                &first
            }
        };
        let next = match blk {
            C3k2Block::Bottleneck(bt) => bottleneck(e, &bp, bt, src)?,
            C3k2Block::C3k(ck) => c3k(e, &bp, ck, src)?,
        };
        outs.push(next);
    }
    Ok((y, outs))
}

/// [`C3k2::forward_cat`]: `cv1` over the parts, the chain on its second half, `cv2` over `y` and the chain.
fn c3k2<E: Exec>(e: &mut E, p: &str, b: &C3k2, xs: &[&E::A]) -> Result<E::A> {
    let (y, outs) = c3k2_parts(e, p, b, xs)?;
    let mut parts: Vec<&E::A> = Vec::with_capacity(1 + outs.len());
    parts.push(&y);
    parts.extend(&outs);
    e.conv(&format!("{p}.cv2"), &b.cv2, &parts, None)
}

/// [`c3k2`] whose final `cv2` outputs `f32` directly (for `neck.out5`, which only feeds the `f32` head).
fn c3k2_f32<E: Exec>(e: &mut E, p: &str, b: &C3k2, xs: &[&E::A]) -> Result<Tensor> {
    let (y, outs) = c3k2_parts(e, p, b, xs)?;
    let mut parts: Vec<&E::A> = Vec::with_capacity(1 + outs.len());
    parts.push(&y);
    parts.extend(&outs);
    e.conv_out_f32(&format!("{p}.cv2"), &b.cv2, &parts)
}

fn sppf<E: Exec>(e: &mut E, p: &str, b: &Sppf, x: &E::A) -> Result<E::A> {
    let y0 = e.conv(&format!("{p}.cv1"), &b.cv1, &[x], None)?;
    let y1 = e.max_pool(&y0, b.k)?;
    let y2 = e.max_pool(&y1, b.k)?;
    let y3 = e.max_pool(&y2, b.k)?;
    e.conv(&format!("{p}.cv2"), &b.cv2, &[&y0, &y1, &y2, &y3], None)
}

/// [`Attention::forward`] with `x` quantised: `qkv` dequantised, the attention and `pe` in `f32`, `proj`'s input
/// quantised (lazily deferred if `proj` runs in `f32`), and the block's residual (`x`) added in `proj`'s epilogue.
fn attention<E: Exec>(e: &mut E, p: &str, a: &Attention, x: &E::A) -> Result<E::A> {
    let [b, _, h, w] = e.dims(x);
    let qkv = e.conv_f32(&format!("{p}.qkv"), &a.qkv, x)?;
    let pin = e.quantize(&format!("{p}.proj.in"), &a.attend_and_pe(&qkv, b, h, w)?)?;
    e.conv(&format!("{p}.proj"), &a.proj, &[&pin], Some(x))
}

fn psa_block<E: Exec>(e: &mut E, p: &str, b: &PsaBlock, x: &E::A) -> Result<E::A> {
    let x1 = attention(e, &format!("{p}.attn"), &b.attn, x)?;
    let f = e.conv(&format!("{p}.ffn1"), &b.ffn1, &[&x1], None)?;
    e.conv(&format!("{p}.ffn2"), &b.ffn2, &[&f], Some(&x1))
}

fn c2psa<E: Exec>(e: &mut E, p: &str, b: &C2Psa, x: &E::A) -> Result<E::A> {
    let y = e.conv(&format!("{p}.cv1"), &b.cv1, &[x], None)?;
    let c = b.cv1.conv.config().out_channels / 2;
    let a = e.narrow(&y, 0, c)?;
    let mut t = e.narrow(&y, c, c)?;
    for (j, blk) in b.blocks.iter().enumerate() {
        t = psa_block(e, &format!("{p}.block{j}"), blk, &t)?;
    }
    e.conv(&format!("{p}.cv2"), &b.cv2, &[&a, &t], None)
}

/// Layers 0–22 of `model` on `x` (`[B, 3, H, W]`, `f32`, already checked) through `e`: the head's three
/// inputs, dequantised. The same graph as `FoloModel::features`.
pub fn features<E: Exec>(model: &FoloModel, e: &mut E, x: &Tensor) -> Result<[Tensor; 3]> {
    let (bb, nk) = (&model.backbone, &model.neck);
    let mut y = e.stem("backbone.stem", &bb.stem, x)?;
    y = e.conv("backbone.down2", &bb.down2, &[&y], None)?;
    y = c3k2(e, "backbone.stage2", &bb.stage2, &[&y])?;
    y = e.conv("backbone.down3", &bb.down3, &[&y], None)?;
    let x4 = c3k2(e, "backbone.stage3", &bb.stage3, &[&y])?;
    y = e.conv("backbone.down4", &bb.down4, &[&x4], None)?;
    let x6 = c3k2(e, "backbone.stage4", &bb.stage4, &[&y])?;
    y = e.conv("backbone.down5", &bb.down5, &[&x6], None)?;
    y = c3k2(e, "backbone.stage5", &bb.stage5, &[&y])?;
    y = sppf(e, "backbone.sppf", &bb.sppf, &y)?;
    let x10 = c2psa(e, "backbone.psa", &bb.psa, &y)?;
    y = e.upsample2(&x10)?;
    let x13 = c3k2(e, "neck.td4", &nk.td4, &[&y, &x6])?;
    drop(x6);
    y = e.upsample2(&x13)?;
    let x16 = c3k2(e, "neck.out3", &nk.out3, &[&y, &x4])?;
    drop(x4);
    y = e.conv("neck.down3", &nk.down3, &[&x16], None)?;
    let x19 = c3k2(e, "neck.out4", &nk.out4, &[&y, &x13])?;
    drop(x13);
    y = e.conv("neck.down4", &nk.down4, &[&x19], None)?;
    let x22 = c3k2_f32(e, "neck.out5", &nk.out5, &[&y, &x10])?;
    Ok([e.dequantize(&x16)?, e.dequantize(&x19)?, x22])
}
