//! The quantised forward of FOLO v1 (layers 0–22) and FOLO v2-Iso (backbone + SPD-PAN neck, docs/MODEL.md
//! §9.6), written once over an executor so calibration, the fake-quant reference and the int8 plan run the same
//! graph with the same boundaries.
//!
//! Every tensor an [`Exec`] holds as an activation (`E::A`) is a quantised one: the output of a conv's epilogue
//! (activation, then the optional residual add), the stem's output, the input of each attention / hyper-context
//! output projection, and the channel slices, max pools, space-to-depth parts, and upsamples of those, which
//! keep their source's grid. What stays `f32`: the network input (unless the direct `u8` stem runs), attention
//! `qkv` / `HyperContext` `av_proj` outputs and their matmuls/softmax/FFN/`pe`, any `f32_convs` subgraph
//! (including int8 producers whose sole consumer runs in `f32`, writing `QOut::F32` directly), relocated
//! `FullPadGate` head modulation, and the head (P3, P4 and P5 are dequantised for it).

mod v2;

pub use v2::features_v2;

use std::fmt::Write as _;

use folo_core::Tensor;
use folo_nn::{ConvBnAct, Forward, Result};

use crate::model::FoloModel;
use crate::model::blocks::{Attention, Bottleneck, C2Psa, C3k, C3k2, C3k2Block, PsaBlock, Sppf};
use crate::model::v2::{FullPadGate, HyperContext, StemV2, WsQARepConv3x3};

/// The ops of the quantised graph. Names are the conv's `.folo` prefix (`backbone.stage2.cv1`), or that of the
/// `proj` conv plus `.in` for its quantised input; they identify a boundary's range in a [`super::Calibration`].
pub trait Exec {
    /// A quantised activation.
    type A;

    /// The stem: an `f32` or direct `u8` conv of the network input, quantised in its epilogue.
    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Self::A>;
    /// FOLO v2 stem (`SPD(2×2) + Conv1×1(12 → c0)` or `Conv3×3 s=2`).
    fn stem_v2(&mut self, name: &str, stem: &StemV2, x: &Tensor) -> Result<Self::A> {
        if !stem.spd {
            self.stem(name, &stem.conv, x)
        } else {
            let s = crate::model::v2::space_to_depth_2x2(x)?;
            self.stem(name, &stem.conv, &s)
        }
    }
    /// `q(act(conv(cat(parts))) + residual)`: a conv over the channel concatenation of `parts`, each part on its
    /// own grid, with the residual (on its own grid) added after the activation.
    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&Self::A],
        res: Option<&Self::A>,
    ) -> Result<Self::A>;
    /// Like [`Exec::conv`], when every downstream consumer of this conv's output already runs in `f32`: an int8
    /// conv may write `f32` directly (`QOut::F32`) instead of requantising to `u8` and immediately dequantising.
    fn conv_f32_out(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&Self::A],
        res: Option<&Self::A>,
    ) -> Result<Self::A> {
        self.conv(name, conv, parts, res)
    }
    /// FOLO v2 `WS-QARepConv3x3`: runs as a fused dense `3×3` int8 conv when fused and eligible (or in `f32`
    /// otherwise), writing `QOut::F32` directly when `out_f32` is true.
    fn rep_conv(&mut self, name: &str, rep: &WsQARepConv3x3, x: &Self::A, _out_f32: bool) -> Result<Self::A> {
        let xf = self.dequantize(x)?;
        let y = rep.forward(&xf)?;
        self.quantize(name, &y)
    }
    /// `act(conv(x))` dequantised: `qkv` / `av_proj`, whose output stays `f32`.
    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &Self::A) -> Result<Tensor>;
    /// Attention matmuls + positional encoding (`o + pe(v)` from `qkv`). Symbolic executors override this to
    /// avoid running `pe` (which would mutate BatchNorm running stats on unfused training-mode models).
    fn attend_and_pe(&mut self, a: &Attention, qkv: &Tensor, b: usize, h: usize, w: usize) -> Result<Tensor> {
        a.attend_and_pe(qkv, b, h, w)
    }
    /// FOLO v2 `HyperContext` gather + FFN + scene pooling + broadcast from `av`. Symbolic executors override
    /// this to avoid running tensor ops during sensitivity analysis.
    fn hyper_attend_and_scene(
        &mut self,
        h_ctx: &HyperContext,
        av: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, Tensor)> {
        h_ctx.attend_and_scene(av, b, h, w)
    }
    /// FOLO v2 relocated `FullPadGate` modulation on `f32` head inputs (`P3_head`/`P4_head`).
    fn fullpad_gate(&mut self, gate: &FullPadGate, feat: &Tensor, g_scene: &Tensor) -> Result<Tensor> {
        gate.forward(feat, g_scene)
    }
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
    /// Nearest-neighbour upsample by 2 with a hint of whether the downstream consumer runs in `f32`: when
    /// `consumer_is_f32` is `false` and `x` is deferred `f32`, an executor may quantise `x` at `(h, w)` first
    /// (4× less quantisation work and exact bit-for-bit codes) before running `q8::upsample2`.
    fn upsample2_for(&mut self, x: &Self::A, _consumer_is_f32: bool) -> Result<Self::A> {
        self.upsample2(x)
    }
    /// Lossless `2×2` space-to-depth split into its 4 spatial phase parts `[tl, tr, bl, br]` (`[N, C, H/2, W/2]`
    /// each, keeping `x`'s exact per-channel grids), with a hint of whether the downstream `p2_proj` runs in
    /// `f32`.
    fn space_to_depth2_parts(&mut self, x: &Self::A, _consumer_is_f32: bool) -> Result<[Self::A; 4]> {
        let _ = x;
        Err(folo_nn::Error::Invalid("space_to_depth2_parts not implemented for this Exec".into()))
    }
    /// Hint to materialise a deferred `f32` activation to `u8` once before multiple int8 consumers (such as
    /// `narrow` + `cv2`, or `SPPF`'s 3 chained `max_pool`s) read it, avoiding redundant quantisation passes.
    fn ensure_u8(&mut self, _x: &Self::A) -> Result<()> {
        Ok(())
    }
    /// Whether `conv` named `name` will execute in `f32` (either listed in `f32_convs` or statically ineligible
    /// for the int8 kernel). Used to avoid redundant `u8 <-> f32` round-trips at int8-to-f32 boundaries.
    fn wants_f32(&self, _name: &str, _conv: &ConvBnAct) -> bool {
        false
    }
    /// Whether `rep` (`WsQARepConv3x3`) named `name` will execute in `f32`.
    fn wants_f32_rep(&self, _name: &str, _rep: &WsQARepConv3x3) -> bool {
        false
    }
    /// `[N, C, H, W]` of an activation.
    fn dims(&self, x: &Self::A) -> [usize; 4];
}

/// Reusable dotted path buffer (`backbone.stage4.block0.block1.cv2`, ≤ 48 bytes) eliminating ~75 `format!`
/// heap allocations per forward pass.
pub(super) struct NameBuf(String);

impl NameBuf {
    pub(super) fn new() -> Self {
        Self(String::with_capacity(64))
    }

    #[inline]
    pub(super) fn set(&mut self, s: &str) {
        self.0.clear();
        self.0.push_str(s);
    }

    #[inline]
    pub(super) fn push_field(&mut self, field: &str) -> usize {
        let len = self.0.len();
        self.0.push('.');
        self.0.push_str(field);
        len
    }

    #[inline]
    pub(super) fn push_block(&mut self, idx: usize) -> usize {
        let len = self.0.len();
        let _ = write!(self.0, ".block{idx}");
        len
    }

    #[inline]
    pub(super) fn push_rep(&mut self, idx: usize) -> usize {
        let len = self.0.len();
        let _ = write!(self.0, ".rep{idx}");
        len
    }

    #[inline]
    pub(super) fn restore(&mut self, len: usize) {
        self.0.truncate(len);
    }

    #[inline]
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stack-backed container for `C3k2` inner-block outputs (`<= 8` blocks covers every standard FOLO scale
/// `n/s/m/l/x` with zero heap allocation; falls back to `Vec` for deeper custom graphs).
enum BlockOuts<A> {
    Inline { items: [Option<A>; 8], len: usize },
    Heap(Vec<A>),
}

impl<A> BlockOuts<A> {
    fn with_capacity(cap: usize) -> Self {
        if cap <= 8 {
            BlockOuts::Inline { items: [const { None }; 8], len: 0 }
        } else {
            BlockOuts::Heap(Vec::with_capacity(cap))
        }
    }

    fn push(&mut self, val: A) {
        match self {
            BlockOuts::Inline { items, len } => {
                items[*len] = Some(val);
                *len += 1;
            }
            BlockOuts::Heap(v) => v.push(val),
        }
    }

    fn last(&self) -> Option<&A> {
        match self {
            BlockOuts::Inline { items, len } => {
                if *len == 0 {
                    None
                } else {
                    items[*len - 1].as_ref()
                }
            }
            BlockOuts::Heap(v) => v.last(),
        }
    }

    fn len(&self) -> usize {
        match self {
            BlockOuts::Inline { len, .. } => *len,
            BlockOuts::Heap(v) => v.len(),
        }
    }

    fn get(&self, idx: usize) -> &A {
        match self {
            BlockOuts::Inline { items, .. } => items[idx].as_ref().expect("inline block out"),
            BlockOuts::Heap(v) => &v[idx],
        }
    }
}

/// Dispatch `conv` or `conv_f32_out` depending on whether every consumer of its output runs in `f32`.
#[inline]
pub(super) fn run_conv<E: Exec>(
    e: &mut E,
    name: &str,
    conv: &ConvBnAct,
    parts: &[&E::A],
    res: Option<&E::A>,
    out_f32: bool,
) -> Result<E::A> {
    if out_f32 { e.conv_f32_out(name, conv, parts, res) } else { e.conv(name, conv, parts, res) }
}

/// Whether all entry consumers of `blk` (at current `nb` prefix) run in `f32`.
fn block_entry_is_f32<E: Exec>(e: &E, nb: &mut NameBuf, blk: &C3k2Block) -> bool {
    match blk {
        C3k2Block::Bottleneck(bt) => {
            let l1 = nb.push_field("cv1");
            let cv1_f32 = e.wants_f32(nb.as_str(), &bt.cv1);
            nb.restore(l1);
            if !cv1_f32 {
                return false;
            }
            if bt.residual {
                let l2 = nb.push_field("cv2");
                let cv2_f32 = e.wants_f32(nb.as_str(), &bt.cv2);
                nb.restore(l2);
                cv2_f32
            } else {
                true
            }
        }
        C3k2Block::C3k(ck) => {
            let l1 = nb.push_field("cv1");
            let cv1_f32 = e.wants_f32(nb.as_str(), &ck.cv1);
            nb.restore(l1);
            if !cv1_f32 {
                return false;
            }
            let l2 = nb.push_field("cv2");
            let cv2_f32 = e.wants_f32(nb.as_str(), &ck.cv2);
            nb.restore(l2);
            cv2_f32
        }
    }
}

fn bottleneck<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &Bottleneck, x: &E::A, out_f32: bool) -> Result<E::A> {
    let l2 = nb.push_field("cv2");
    let cv2_f32 = e.wants_f32(nb.as_str(), &b.cv2);
    nb.restore(l2);

    let l1 = nb.push_field("cv1");
    let y = run_conv(e, nb.as_str(), &b.cv1, &[x], None, cv2_f32)?;
    nb.restore(l1);

    let l2 = nb.push_field("cv2");
    let out = run_conv(e, nb.as_str(), &b.cv2, &[&y], b.residual.then_some(x), out_f32)?;
    nb.restore(l2);
    Ok(out)
}

fn c3k<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &C3k, x: &E::A, out_f32: bool) -> Result<E::A> {
    let l3 = nb.push_field("cv3");
    let cv3_f32 = e.wants_f32(nb.as_str(), &b.cv3);
    nb.restore(l3);

    let cv1_out_f32 = match b.blocks.items().first() {
        Some(first) => {
            let lb = nb.push_block(0);
            let lf1 = nb.push_field("cv1");
            let f1 = e.wants_f32(nb.as_str(), &first.cv1);
            nb.restore(lf1);
            let ok = if f1 && first.residual {
                let lf2 = nb.push_field("cv2");
                let f2 = e.wants_f32(nb.as_str(), &first.cv2);
                nb.restore(lf2);
                f2
            } else {
                f1
            };
            nb.restore(lb);
            ok
        }
        None => cv3_f32,
    };

    let l1 = nb.push_field("cv1");
    let mut a = run_conv(e, nb.as_str(), &b.cv1, &[x], None, cv1_out_f32)?;
    nb.restore(l1);

    let n_blocks = b.blocks.len();
    for (j, blk) in b.blocks.iter().enumerate() {
        let next_f32 = if j + 1 < n_blocks {
            let next_blk = &b.blocks.items()[j + 1];
            let lb = nb.push_block(j + 1);
            let lf1 = nb.push_field("cv1");
            let f1 = e.wants_f32(nb.as_str(), &next_blk.cv1);
            nb.restore(lf1);
            let ok = if f1 && next_blk.residual {
                let lf2 = nb.push_field("cv2");
                let f2 = e.wants_f32(nb.as_str(), &next_blk.cv2);
                nb.restore(lf2);
                f2
            } else {
                f1
            };
            nb.restore(lb);
            ok
        } else {
            cv3_f32
        };
        let lb = nb.push_block(j);
        a = bottleneck(e, nb, blk, &a, next_f32)?;
        nb.restore(lb);
    }

    let l2 = nb.push_field("cv2");
    let c = run_conv(e, nb.as_str(), &b.cv2, &[x], None, cv3_f32)?;
    nb.restore(l2);

    let l3 = nb.push_field("cv3");
    let out = run_conv(e, nb.as_str(), &b.cv3, &[&a, &c], None, out_f32)?;
    nb.restore(l3);
    Ok(out)
}

fn c3k2_parts<E: Exec>(
    e: &mut E,
    nb: &mut NameBuf,
    b: &C3k2,
    xs: &[&E::A],
) -> Result<(E::A, BlockOuts<E::A>)> {
    let l2 = nb.push_field("cv2");
    let cv2_f32 = e.wants_f32(nb.as_str(), &b.cv2);
    nb.restore(l2);

    let first_blk_f32 = match b.blocks.items().first() {
        Some(first) => {
            let lb = nb.push_block(0);
            let ok = block_entry_is_f32(e, nb, first);
            nb.restore(lb);
            ok
        }
        None => true,
    };

    let l1 = nb.push_field("cv1");
    let y = run_conv(e, nb.as_str(), &b.cv1, xs, None, cv2_f32 && first_blk_f32)?;
    nb.restore(l1);

    if !cv2_f32 {
        e.ensure_u8(&y)?;
    }

    let c = b.cv1.conv.config().out_channels / 2;
    let n_blocks = b.blocks.len();
    let mut outs = BlockOuts::with_capacity(n_blocks);
    for (j, blk) in b.blocks.iter().enumerate() {
        let next_f32 = if !cv2_f32 {
            false
        } else if j + 1 < n_blocks {
            let lb = nb.push_block(j + 1);
            let ok = block_entry_is_f32(e, nb, &b.blocks.items()[j + 1]);
            nb.restore(lb);
            ok
        } else {
            true
        };
        let first;
        let src = match outs.last() {
            Some(prev) => prev,
            None => {
                first = e.narrow(&y, c, c)?;
                &first
            }
        };
        let lb = nb.push_block(j);
        let next = match blk {
            C3k2Block::Bottleneck(bt) => bottleneck(e, nb, bt, src, next_f32)?,
            C3k2Block::C3k(ck) => c3k(e, nb, ck, src, next_f32)?,
        };
        nb.restore(lb);
        outs.push(next);
    }
    Ok((y, outs))
}

fn c3k2<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &C3k2, xs: &[&E::A], out_f32: bool) -> Result<E::A> {
    let (y, outs) = c3k2_parts(e, nb, b, xs)?;
    let l2 = nb.push_field("cv2");
    let res = if outs.len() <= 8 {
        let mut arr = [&y; 9];
        for i in 0..outs.len() {
            arr[1 + i] = outs.get(i);
        }
        run_conv(e, nb.as_str(), &b.cv2, &arr[..1 + outs.len()], None, out_f32)
    } else {
        let mut parts: Vec<&E::A> = Vec::with_capacity(1 + outs.len());
        parts.push(&y);
        for i in 0..outs.len() {
            parts.push(outs.get(i));
        }
        run_conv(e, nb.as_str(), &b.cv2, &parts, None, out_f32)
    };
    nb.restore(l2);
    res
}

fn c3k2_f32<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &C3k2, xs: &[&E::A]) -> Result<Tensor> {
    let (y, outs) = c3k2_parts(e, nb, b, xs)?;
    let l2 = nb.push_field("cv2");
    let res = if outs.len() <= 8 {
        let mut arr = [&y; 9];
        for i in 0..outs.len() {
            arr[1 + i] = outs.get(i);
        }
        e.conv_out_f32(nb.as_str(), &b.cv2, &arr[..1 + outs.len()])
    } else {
        let mut parts: Vec<&E::A> = Vec::with_capacity(1 + outs.len());
        parts.push(&y);
        for i in 0..outs.len() {
            parts.push(outs.get(i));
        }
        e.conv_out_f32(nb.as_str(), &b.cv2, &parts)
    };
    nb.restore(l2);
    res
}

pub(super) fn sppf<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &Sppf, x: &E::A, out_f32: bool) -> Result<E::A> {
    let l2 = nb.push_field("cv2");
    let cv2_f32 = e.wants_f32(nb.as_str(), &b.cv2);
    nb.restore(l2);

    let l1 = nb.push_field("cv1");
    let y0 = run_conv(e, nb.as_str(), &b.cv1, &[x], None, cv2_f32)?;
    nb.restore(l1);

    if !cv2_f32 {
        e.ensure_u8(&y0)?;
    }

    let y1 = e.max_pool(&y0, b.k)?;
    let y2 = e.max_pool(&y1, b.k)?;
    let y3 = e.max_pool(&y2, b.k)?;

    let l2 = nb.push_field("cv2");
    let out = run_conv(e, nb.as_str(), &b.cv2, &[&y0, &y1, &y2, &y3], None, out_f32)?;
    nb.restore(l2);
    Ok(out)
}

fn attention<E: Exec>(e: &mut E, nb: &mut NameBuf, a: &Attention, x: &E::A, out_f32: bool) -> Result<E::A> {
    let [b, _, h, w] = e.dims(x);
    let lq = nb.push_field("qkv");
    let qkv = e.conv_f32(nb.as_str(), &a.qkv, x)?;
    nb.restore(lq);

    let attn_out = e.attend_and_pe(a, &qkv, b, h, w)?;
    let lp = nb.push_field("proj.in");
    let pin = e.quantize(nb.as_str(), &attn_out)?;
    nb.restore(lp);

    let lp = nb.push_field("proj");
    let out = run_conv(e, nb.as_str(), &a.proj, &[&pin], Some(x), out_f32)?;
    nb.restore(lp);
    Ok(out)
}

fn psa_entry_is_f32<E: Exec>(e: &E, nb: &mut NameBuf, b: &PsaBlock) -> bool {
    let la = nb.push_field("attn");
    let lq = nb.push_field("qkv");
    let qkv_f32 = e.wants_f32(nb.as_str(), &b.attn.qkv);
    nb.restore(lq);
    let ok = if qkv_f32 {
        let lp = nb.push_field("proj");
        let p_f32 = e.wants_f32(nb.as_str(), &b.attn.proj);
        nb.restore(lp);
        p_f32
    } else {
        false
    };
    nb.restore(la);
    ok
}

fn psa_block<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &PsaBlock, x: &E::A, out_f32: bool) -> Result<E::A> {
    let lf1 = nb.push_field("ffn1");
    let ffn1_f32 = e.wants_f32(nb.as_str(), &b.ffn1);
    nb.restore(lf1);
    let lf2 = nb.push_field("ffn2");
    let ffn2_f32 = e.wants_f32(nb.as_str(), &b.ffn2);
    nb.restore(lf2);

    let la = nb.push_field("attn");
    let x1 = attention(e, nb, &b.attn, x, ffn1_f32 && ffn2_f32)?;
    nb.restore(la);

    let lf1 = nb.push_field("ffn1");
    let f = run_conv(e, nb.as_str(), &b.ffn1, &[&x1], None, ffn2_f32)?;
    nb.restore(lf1);

    let lf2 = nb.push_field("ffn2");
    let out = run_conv(e, nb.as_str(), &b.ffn2, &[&f], Some(&x1), out_f32)?;
    nb.restore(lf2);
    Ok(out)
}

fn c2psa<E: Exec>(e: &mut E, nb: &mut NameBuf, b: &C2Psa, x: &E::A, out_f32: bool) -> Result<E::A> {
    let l2 = nb.push_field("cv2");
    let cv2_f32 = e.wants_f32(nb.as_str(), &b.cv2);
    nb.restore(l2);

    let first_blk_f32 = match b.blocks.items().first() {
        Some(first) => {
            let lb = nb.push_block(0);
            let ok = psa_entry_is_f32(e, nb, first);
            nb.restore(lb);
            ok
        }
        None => true,
    };

    let l1 = nb.push_field("cv1");
    let y = run_conv(e, nb.as_str(), &b.cv1, &[x], None, cv2_f32 && first_blk_f32)?;
    nb.restore(l1);

    if !cv2_f32 {
        e.ensure_u8(&y)?;
    }

    let c = b.cv1.conv.config().out_channels / 2;
    let a = e.narrow(&y, 0, c)?;
    let mut t = e.narrow(&y, c, c)?;
    drop(y);

    let n_blocks = b.blocks.len();
    for (j, blk) in b.blocks.iter().enumerate() {
        let next_f32 = if j + 1 < n_blocks {
            let lb = nb.push_block(j + 1);
            let ok = psa_entry_is_f32(e, nb, &b.blocks.items()[j + 1]);
            nb.restore(lb);
            ok
        } else {
            cv2_f32
        };
        let lb = nb.push_block(j);
        t = psa_block(e, nb, blk, &t, next_f32)?;
        nb.restore(lb);
    }

    let l2 = nb.push_field("cv2");
    let out = run_conv(e, nb.as_str(), &b.cv2, &[&a, &t], None, out_f32)?;
    nb.restore(l2);
    Ok(out)
}

/// Layers 0–22 of `model` on `x` (`[B, 3, H, W]`, `f32`, already checked) through `e`: the head's three
/// inputs, dequantised. The same graph as `FoloModel::features`.
pub fn features<E: Exec>(model: &FoloModel, e: &mut E, x: &Tensor) -> Result<[Tensor; 3]> {
    let (bb, nk) = (&model.backbone, &model.neck);
    let mut nb = NameBuf::new();

    let mut y = e.stem("backbone.stem", &bb.stem, x)?;
    y = run_conv(
        e,
        "backbone.down2",
        &bb.down2,
        &[&y],
        None,
        e.wants_f32("backbone.stage2.cv1", &bb.stage2.cv1),
    )?;

    nb.set("backbone.stage2");
    y = c3k2(e, &mut nb, &bb.stage2, &[&y], e.wants_f32("backbone.down3", &bb.down3))?;

    y = run_conv(
        e,
        "backbone.down3",
        &bb.down3,
        &[&y],
        None,
        e.wants_f32("backbone.stage3.cv1", &bb.stage3.cv1),
    )?;

    let x4_f32 = e.wants_f32("backbone.down4", &bb.down4) && e.wants_f32("neck.out3.cv1", &nk.out3.cv1);
    nb.set("backbone.stage3");
    let x4 = c3k2(e, &mut nb, &bb.stage3, &[&y], x4_f32)?;

    y = run_conv(
        e,
        "backbone.down4",
        &bb.down4,
        &[&x4],
        None,
        e.wants_f32("backbone.stage4.cv1", &bb.stage4.cv1),
    )?;

    let x6_f32 = e.wants_f32("backbone.down5", &bb.down5) && e.wants_f32("neck.td4.cv1", &nk.td4.cv1);
    nb.set("backbone.stage4");
    let x6 = c3k2(e, &mut nb, &bb.stage4, &[&y], x6_f32)?;

    y = run_conv(
        e,
        "backbone.down5",
        &bb.down5,
        &[&x6],
        None,
        e.wants_f32("backbone.stage5.cv1", &bb.stage5.cv1),
    )?;

    nb.set("backbone.stage5");
    y = c3k2(e, &mut nb, &bb.stage5, &[&y], e.wants_f32("backbone.sppf.cv1", &bb.sppf.cv1))?;

    nb.set("backbone.sppf");
    y = sppf(e, &mut nb, &bb.sppf, &y, e.wants_f32("backbone.psa.cv1", &bb.psa.cv1))?;

    let td4_f32 = e.wants_f32("neck.td4.cv1", &nk.td4.cv1);
    let out5_f32 = e.wants_f32("neck.out5.cv1", &nk.out5.cv1);
    nb.set("backbone.psa");
    let x10 = c2psa(e, &mut nb, &bb.psa, &y, td4_f32 && out5_f32)?;

    y = e.upsample2_for(&x10, td4_f32)?;

    let out3_f32 = e.wants_f32("neck.out3.cv1", &nk.out3.cv1);
    let out4_f32 = e.wants_f32("neck.out4.cv1", &nk.out4.cv1);
    nb.set("neck.td4");
    let x13 = c3k2(e, &mut nb, &nk.td4, &[&y, &x6], out3_f32 && out4_f32)?;
    drop(x6);

    y = e.upsample2_for(&x13, out3_f32)?;

    nb.set("neck.out3");
    let x16 = c3k2(e, &mut nb, &nk.out3, &[&y, &x4], e.wants_f32("neck.down3", &nk.down3))?;
    drop(x4);

    y = run_conv(e, "neck.down3", &nk.down3, &[&x16], None, out4_f32)?;
    // Dequantise P3 while `x16` is hot in L2 cache and release its C4 buffer before P4/P5 run.
    let p3 = e.dequantize(&x16)?;
    drop(x16);

    nb.set("neck.out4");
    let x19 = c3k2(e, &mut nb, &nk.out4, &[&y, &x13], e.wants_f32("neck.down4", &nk.down4))?;
    drop(x13);

    y = run_conv(e, "neck.down4", &nk.down4, &[&x19], None, out5_f32)?;
    let p4 = e.dequantize(&x19)?;
    drop(x19);

    nb.set("neck.out5");
    let p5 = c3k2_f32(e, &mut nb, &nk.out5, &[&y, &x10])?;
    drop(x10);
    Ok([p3, p4, p5])
}
