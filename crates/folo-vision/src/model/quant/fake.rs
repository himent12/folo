//! The fake-quant reference (docs/MODEL.md §9.6): the int8 plan's arithmetic in `f32`. Every activation is
//! rounded onto its boundary's u8 grid (per-channel in `version: 2`, per-tensor in `version: 1`), every
//! quantised conv's weights onto their s8 grid (mirroring per-part concat scale decoupling and direct `u8` stem),
//! and everything else is the `f32` model.

use std::collections::{HashMap, HashSet};

use folo_core::cpu::q8::{PIXEL_SCALE, should_decouple_parts, silu_q8};
use folo_core::{Tensor, TensorId};
use folo_nn::{ConvBnAct, Forward, Result};

use super::walk::{self, Exec};
use super::{
    Calibration, ConvView, QParam, int8_able_view, quantize_weights, spd_stem_int8_able, static_int8_able,
    static_int8_able_rep, stem_int8_able,
};
use crate::model::FoloModel;
use crate::model::v2::{FoloV2Model, StemV2, WsQARepConv3x3, space_to_depth_2x2};

/// A fake-quantised activation: its values (on the grid, or raw `f32` until an int8 consumer reads it) and its
/// per-tensor fallback grid.
#[derive(Clone)]
pub struct Fq {
    /// The activation values.
    pub t: Tensor,
    /// The grid.
    pub q: QParam,
}

/// The fake-quant executor of a calibrated model.
pub struct FakeQuant<'a> {
    cal: &'a Calibration,
    /// Per-channel grids tracked by tensor identity (keeps `Fq`'s public fields unchanged).
    ch_grids: HashMap<TensorId, Vec<QParam>>,
    /// Tensors kept in unquantised `f32` (produced by `f32` convs, `f32` stem, or deferred `quantize`) until an
    /// int8 op consumes them, matching [`super::Int8Plan`]'s lazy boundary quantisation.
    f32_acts: HashSet<TensorId>,
}

/// Round every element of `t` onto `q`.
pub fn fake_quantize(t: &Tensor, q: QParam) -> Result<Tensor> {
    let v: Vec<f32> = t.to_vec::<f32>()?.into_iter().map(|x| q.dequantize(q.quantize(x))).collect();
    Ok(Tensor::from_vec(v, t.dims().to_vec())?)
}

/// Round every channel `ch` of `t` (`[N, C, H, W]`) onto `grids[ch]`.
pub fn fake_quantize_per_channel(t: &Tensor, grids: &[QParam]) -> Result<Tensor> {
    let (n, c, h, w) = t.dims4()?;
    assert_eq!(grids.len(), c, "fake_quantize_per_channel: grids length");
    let hw = h * w;
    let mut v = t.to_vec::<f32>()?;
    for img in 0..n {
        for (ch, g) in grids.iter().enumerate() {
            let base = (img * c + ch) * hw;
            for x in &mut v[base..base + hw] {
                *x = g.dequantize(g.quantize(*x));
            }
        }
    }
    Ok(Tensor::from_vec(v, [n, c, h, w])?)
}

/// `w` (`[O, C, kh, kw]`) on the s8 grid of the int8 plan, given the scale of each input channel:
/// `w'[o, c] = w[o, c] · s(c)`, quantised per output channel, then divided by `s(c)` again.
pub fn fake_weights(w: &Tensor, in_scales: &[f32]) -> Result<Tensor> {
    let d = w.dims().to_vec();
    let (o, c) = (d[0], d[1]);
    let taps = d[2] * d[3];
    let mut v = w.to_vec::<f32>()?;
    for (i, x) in v.iter_mut().enumerate() {
        *x *= in_scales[(i / taps) % c];
    }
    let (q, s) = quantize_weights(&v, o);
    let k = c * taps;
    for (i, x) in v.iter_mut().enumerate() {
        *x = q[i] as f32 * s[i / k] / in_scales[(i / taps) % c];
    }
    Ok(Tensor::from_vec(v, d)?)
}

fn dims4(t: &Tensor) -> [usize; 4] {
    let d = t.dims();
    [d[0], d[1], d[2], d[3]]
}

impl<'a> FakeQuant<'a> {
    /// The executor for `cal`.
    pub fn new(cal: &'a Calibration) -> Self {
        FakeQuant { cal, ch_grids: HashMap::new(), f32_acts: HashSet::new() }
    }

    /// Layers 0–22 of `model` on `x` with fake quantisation: the head's three inputs.
    pub fn features(&mut self, model: &FoloModel, x: &Tensor) -> Result<[Tensor; 3]> {
        self.ch_grids.clear();
        self.f32_acts.clear();
        let _g = folo_core::no_grad();
        let res = folo_core::cpu::in_pool(|| walk::features(model, self, x));
        self.ch_grids.clear();
        self.f32_acts.clear();
        res
    }

    /// Backbone and SPD-PAN neck of a [`FoloV2Model`] on `x` with fake quantisation: the head's three inputs.
    pub fn features_v2(&mut self, model: &FoloV2Model, x: &Tensor) -> Result<[Tensor; 3]> {
        self.ch_grids.clear();
        self.f32_acts.clear();
        let _g = folo_core::no_grad();
        let res = folo_core::cpu::in_pool(|| walk::features_v2(model, self, x));
        self.ch_grids.clear();
        self.f32_acts.clear();
        res
    }

    fn grids_of(&self, a: &Fq) -> Vec<QParam> {
        if let Some(gs) = self.ch_grids.get(&a.t.id()) {
            return gs.clone();
        }
        vec![a.q; a.t.dims()[1]]
    }

    fn wrap_u8(&mut self, t: Tensor, q: QParam, grids: Vec<QParam>) -> Fq {
        self.ch_grids.insert(t.id(), grids);
        Fq { t, q }
    }

    fn wrap_f32(&mut self, t: Tensor, q: QParam, grids: Vec<QParam>) -> Fq {
        let id = t.id();
        self.ch_grids.insert(id, grids);
        self.f32_acts.insert(id);
        Fq { t, q }
    }

    /// Quantise `a` onto its boundary grid if it is still deferred `f32`, else return it as-is.
    fn as_quantized_tensor(&self, a: &Fq, grids: &[QParam]) -> Result<Tensor> {
        if self.f32_acts.contains(&a.t.id()) {
            fake_quantize_per_channel(&a.t, grids)
        } else {
            Ok(a.t.clone())
        }
    }

    fn is_f32_conv_view(&self, name: &str, conv: ConvView<'_>, parts: &[&Fq], res: Option<&Fq>) -> bool {
        let d = parts[0].t.dims();
        let c: usize = parts.iter().map(|p| p.t.dims()[1]).sum();
        self.cal.is_f32(name)
            || !parts.iter().all(|p| p.t.dims()[1].is_multiple_of(4))
            || int8_able_view(conv, [d[0], c, d[2], d[3]], res.map(|r| dims4(&r.t))).is_none()
    }

    fn run(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&Fq],
        res: Option<&Fq>,
    ) -> Result<(Tensor, bool)> {
        self.run_view(name, ConvView::from_cba(conv), parts, res)
    }

    fn run_view(
        &mut self,
        name: &str,
        conv: ConvView<'_>,
        parts: &[&Fq],
        res: Option<&Fq>,
    ) -> Result<(Tensor, bool)> {
        let c: usize = parts.iter().map(|p| p.t.dims()[1]).sum();
        let p = conv.conv.config().params();
        if self.is_f32_conv_view(name, conv, parts, res) {
            let ts: Vec<&Tensor> = parts.iter().map(|p| &p.t).collect();
            let y = Tensor::conv2d_act_cat(
                &ts,
                &conv.conv.weight,
                conv.conv.bias.as_ref(),
                &p,
                conv.act.unary_op(),
            )?;
            return Ok((y, true));
        }
        // Compute in the well-conditioned integer-code domain `(q_x - z) * (q_w * s_out)` so tiny/dead
        // per-channel `in_scales[ic]` (down to `1e-8`) never blow up `w / in_scales[ic]` inside f32 Winograd.
        let owned_grids: Vec<Vec<QParam>> = parts.iter().map(|a| self.grids_of(a)).collect();
        let grid_refs: Vec<&[QParam]> = owned_grids.iter().map(Vec::as_slice).collect();
        let decouple = self.cal.version >= super::VERSION_V2 && should_decouple_parts(&grid_refs);
        let norm_parts: Vec<Tensor> = parts
            .iter()
            .zip(&owned_grids)
            .map(|(a, gs)| {
                let qt = self.as_quantized_tensor(a, gs)?;
                divide_by_scales(&qt, gs)
            })
            .collect::<Result<_>>()?;
        let o = conv.conv.config().out_channels;
        let taps = conv.conv.config().kernel * conv.conv.config().kernel;
        let orig_w = conv.conv.weight.to_vec::<f32>()?;
        let bias_v: Vec<f32> = match &conv.conv.bias {
            Some(b) => b.to_vec::<f32>()?,
            None => vec![0.0; o],
        };
        let acc = if decouple && parts.len() > 1 {
            let mut sum: Option<Vec<f32>> = None;
            let mut out_dims = Vec::new();
            let mut ch0 = 0;
            for (part_t, pg) in norm_parts.iter().zip(&owned_grids) {
                let pc = pg.len();
                let (qw, sc) = quantized_folded_weight(&orig_w, o, c, taps, ch0, pg)?;
                let raw = Tensor::conv2d_act_cat(&[part_t], &qw, None, &p, None)?;
                let (n, oc_n, oh, ow) = raw.dims4()?;
                let hw = oh * ow;
                out_dims = vec![n, oc_n, oh, ow];
                let rv = raw.to_vec::<f32>()?;
                let dst = sum.get_or_insert_with(|| vec![0.0; rv.len()]);
                for img in 0..n {
                    for (oc, &s) in sc.iter().enumerate().take(oc_n) {
                        let base = (img * oc_n + oc) * hw;
                        for idx in base..base + hw {
                            dst[idx] = rv[idx].round_ties_even().mul_add(s, dst[idx]);
                        }
                    }
                }
                ch0 += pc;
            }
            let mut dst = sum.expect("at least one part");
            let (n, oc_n, hw) = (out_dims[0], out_dims[1], out_dims[2] * out_dims[3]);
            for img in 0..n {
                for (oc, &b) in bias_v.iter().enumerate().take(oc_n) {
                    let base = (img * oc_n + oc) * hw;
                    for v in &mut dst[base..base + hw] {
                        *v += b;
                    }
                }
            }
            Tensor::from_vec(dst, out_dims)?
        } else {
            let all_grids: Vec<QParam> = owned_grids.iter().flat_map(|gs| gs.iter().copied()).collect();
            let (qw, sc) = quantized_folded_weight(&orig_w, o, c, taps, 0, &all_grids)?;
            let ts: Vec<&Tensor> = norm_parts.iter().collect();
            let raw = Tensor::conv2d_act_cat(&ts, &qw, None, &p, None)?;
            let (n, oc_n, oh, ow) = raw.dims4()?;
            let hw = oh * ow;
            let mut rv = raw.to_vec::<f32>()?;
            for img in 0..n {
                for oc in 0..oc_n {
                    let (s, b) = (sc[oc], bias_v[oc]);
                    let base = (img * oc_n + oc) * hw;
                    for v in &mut rv[base..base + hw] {
                        *v = v.round_ties_even().mul_add(s, b);
                    }
                }
            }
            Tensor::from_vec(rv, [n, oc_n, oh, ow])?
        };
        let out = if conv.act == folo_nn::Activation::Silu {
            let d = acc.dims().to_vec();
            let v: Vec<f32> = acc.to_vec::<f32>()?.into_iter().map(silu_q8).collect();
            Tensor::from_vec(v, d)?
        } else {
            conv.act.forward(&acc)?
        };
        Ok((out, false))
    }
}

/// Recover integer `(q - z)` values from dequantised `t` without `1 / scale` weight blowup.
fn divide_by_scales(t: &Tensor, grids: &[QParam]) -> Result<Tensor> {
    let (n, c, h, w) = t.dims4()?;
    let hw = h * w;
    let mut v = t.to_vec::<f32>()?;
    for img in 0..n {
        for (ch, g) in grids.iter().enumerate().take(c) {
            let inv = g.inv();
            let base = (img * c + ch) * hw;
            for x in &mut v[base..base + hw] {
                *x = (*x * inv).round_ties_even();
            }
        }
    }
    Ok(Tensor::from_vec(v, [n, c, h, w])?)
}

/// Quantise `w[:, ch0..ch0 + pg.len()] * pg[ic].scale` per output channel and return `(q as f32, scale [o])`
/// so the conv accumulates integers before applying `mul_add(scale, bias)` (matching `QConv`'s exact epilogue).
fn quantized_folded_weight(
    orig: &[f32],
    o: usize,
    c: usize,
    taps: usize,
    ch0: usize,
    pg: &[QParam],
) -> Result<(Tensor, Vec<f32>)> {
    let pc = pg.len();
    let k_side = (taps as f64).sqrt().round() as usize;
    let mut folded = Vec::with_capacity(o * pc * taps);
    for oc in 0..o {
        let base = (oc * c + ch0) * taps;
        for (ic, g) in pg.iter().enumerate() {
            folded.extend(orig[base + ic * taps..base + (ic + 1) * taps].iter().map(|&v| v * g.scale));
        }
    }
    let (q, s) = quantize_weights(&folded, o);
    let q_f32: Vec<f32> = q.into_iter().map(f32::from).collect();
    Ok((Tensor::from_vec(q_f32, [o, pc, k_side, k_side])?, s))
}

impl Exec for FakeQuant<'_> {
    type A = Fq;

    fn wants_f32(&self, name: &str, conv: &ConvBnAct) -> bool {
        self.cal.is_f32(name) || !static_int8_able(conv)
    }

    fn wants_f32_rep(&self, name: &str, rep: &WsQARepConv3x3) -> bool {
        self.cal.is_f32(name) || !static_int8_able_rep(rep)
    }

    fn stem_v2(&mut self, name: &str, stem: &StemV2, x: &Tensor) -> Result<Fq> {
        if !stem.spd {
            return self.stem(name, &stem.conv, x);
        }
        let conv = &stem.conv;
        let q = self.cal.param(name)?;
        let o = conv.conv.config().out_channels;
        let grids = self.cal.channel_params(name, o)?;
        if self.cal.u8_stem && !self.cal.is_f32(name) && spd_stem_int8_able(conv, dims4(x)).is_some() {
            let xq_vec: Vec<f32> = x
                .to_vec::<f32>()?
                .into_iter()
                .map(|v| (v * 255.0).clamp(0.0, 255.0).round_ties_even())
                .collect();
            let xq = space_to_depth_2x2(&Tensor::from_vec(xq_vec, x.dims().to_vec())?)?;
            let orig_w = conv.conv.weight.to_vec::<f32>()?;
            let taps = conv.conv.config().kernel * conv.conv.config().kernel;
            let pixel_grids = [QParam { scale: PIXEL_SCALE, zero: 0 }; 12];
            let (qw, sc) = quantized_folded_weight(&orig_w, o, 12, taps, 0, &pixel_grids)?;
            let bias_v: Vec<f32> = match &conv.conv.bias {
                Some(b) => b.to_vec::<f32>()?,
                None => vec![0.0; o],
            };
            let p = conv.conv.config().params();
            let raw = Tensor::conv2d_act_cat(&[&xq], &qw, None, &p, None)?;
            let (n, oc_n, oh, ow) = raw.dims4()?;
            let hw = oh * ow;
            let mut rv = raw.to_vec::<f32>()?;
            let is_silu = conv.act == folo_nn::Activation::Silu;
            for img in 0..n {
                for oc in 0..oc_n {
                    let (s, b) = (sc[oc], bias_v[oc]);
                    let base = (img * oc_n + oc) * hw;
                    for v in &mut rv[base..base + hw] {
                        let y = v.round_ties_even().mul_add(s, b);
                        *v = if is_silu { silu_q8(y) } else { y };
                    }
                }
            }
            let y = Tensor::from_vec(rv, [n, oc_n, oh, ow])?;
            let t = fake_quantize_per_channel(&y, &grids)?;
            Ok(self.wrap_u8(t, q, grids))
        } else {
            let y = stem.forward(x)?;
            Ok(self.wrap_f32(y, q, grids))
        }
    }

    fn rep_conv(&mut self, name: &str, rep: &WsQARepConv3x3, x: &Fq, out_f32: bool) -> Result<Fq> {
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, rep.out_channels)?;
        let Some(view) = ConvView::from_rep(rep) else {
            let y = rep.forward(&x.t)?;
            return Ok(self.wrap_f32(y, q, grids));
        };
        let (y, ran_f32) = self.run_view(name, view, &[x], None)?;
        if ran_f32 || out_f32 {
            Ok(self.wrap_f32(y, q, grids))
        } else {
            let t = fake_quantize_per_channel(&y, &grids)?;
            Ok(self.wrap_u8(t, q, grids))
        }
    }

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        let o = conv.conv.config().out_channels;
        let grids = self.cal.channel_params(name, o)?;
        if self.cal.u8_stem && !self.cal.is_f32(name) && stem_int8_able(conv, dims4(x)).is_some() {
            // Compute in the exact integer-code domain `(u8_pixel * s8_weight).round_ties_even() * s_o + bias`
            // followed by `silu_q8` (matching `q8::stem_u8`'s exact `i32` sums and epilogue).
            let xq_vec: Vec<f32> = x
                .to_vec::<f32>()?
                .into_iter()
                .map(|v| (v * 255.0).clamp(0.0, 255.0).round_ties_even())
                .collect();
            let xq = Tensor::from_vec(xq_vec, x.dims().to_vec())?;
            let orig_w = conv.conv.weight.to_vec::<f32>()?;
            let taps = conv.conv.config().kernel * conv.conv.config().kernel;
            let pixel_grids = [QParam { scale: PIXEL_SCALE, zero: 0 }; 3];
            let (qw, sc) = quantized_folded_weight(&orig_w, o, 3, taps, 0, &pixel_grids)?;
            let bias_v: Vec<f32> = match &conv.conv.bias {
                Some(b) => b.to_vec::<f32>()?,
                None => vec![0.0; o],
            };
            let p = conv.conv.config().params();
            let raw = Tensor::conv2d_act_cat(&[&xq], &qw, None, &p, None)?;
            let (n, oc_n, oh, ow) = raw.dims4()?;
            let hw = oh * ow;
            let mut rv = raw.to_vec::<f32>()?;
            let is_silu = conv.act == folo_nn::Activation::Silu;
            for img in 0..n {
                for oc in 0..oc_n {
                    let (s, b) = (sc[oc], bias_v[oc]);
                    let base = (img * oc_n + oc) * hw;
                    for v in &mut rv[base..base + hw] {
                        let y = v.round_ties_even().mul_add(s, b);
                        *v = if is_silu { silu_q8(y) } else { y };
                    }
                }
            }
            let y = Tensor::from_vec(rv, [n, oc_n, oh, ow])?;
            let t = fake_quantize_per_channel(&y, &grids)?;
            Ok(self.wrap_u8(t, q, grids))
        } else {
            let y = conv.forward(x)?;
            Ok(self.wrap_f32(y, q, grids))
        }
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Fq> {
        let (y, ran_f32) = self.run(name, conv, parts, res)?;
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, y.dims()[1])?;
        if ran_f32 {
            let y = match res {
                Some(r) => r.t.add(&y)?,
                None => y,
            };
            Ok(self.wrap_f32(y, q, grids))
        } else {
            let y = match res {
                Some(r) => {
                    let rg = self.grids_of(r);
                    let rt = self.as_quantized_tensor(r, &rg)?;
                    rt.add(&y)?
                }
                None => y,
            };
            let t = fake_quantize_per_channel(&y, &grids)?;
            Ok(self.wrap_u8(t, q, grids))
        }
    }

    fn conv_f32_out(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Fq> {
        let (y, ran_f32) = self.run(name, conv, parts, res)?;
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, y.dims()[1])?;
        let y = match res {
            Some(r) if ran_f32 => r.t.add(&y)?,
            Some(r) => {
                let rg = self.grids_of(r);
                let rt = self.as_quantized_tensor(r, &rg)?;
                rt.add(&y)?
            }
            None => y,
        };
        Ok(self.wrap_f32(y, q, grids))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &Fq) -> Result<Tensor> {
        Ok(self.run(name, conv, &[x], None)?.0)
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq]) -> Result<Tensor> {
        let _ = self.cal.param(name)?;
        let _ = self.cal.channel_params(name, conv.conv.config().out_channels)?;
        Ok(self.run(name, conv, parts, None)?.0)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, x.dims()[1])?;
        Ok(self.wrap_f32(x.clone(), q, grids))
    }

    fn dequantize(&mut self, x: &Fq) -> Result<Tensor> {
        Ok(x.t.clone())
    }

    fn narrow(&mut self, x: &Fq, start: usize, len: usize) -> Result<Fq> {
        let t = x.t.narrow(1, start, len)?;
        let gs = self.grids_of(x)[start..start + len].to_vec();
        if self.f32_acts.contains(&x.t.id()) {
            Ok(self.wrap_f32(t, x.q, gs))
        } else {
            Ok(self.wrap_u8(t, x.q, gs))
        }
    }

    fn max_pool(&mut self, x: &Fq, k: usize) -> Result<Fq> {
        let t = x.t.max_pool2d(k, 1, k / 2)?;
        let gs = self.grids_of(x);
        if self.f32_acts.contains(&x.t.id()) {
            Ok(self.wrap_f32(t, x.q, gs))
        } else {
            Ok(self.wrap_u8(t, x.q, gs))
        }
    }

    fn upsample2(&mut self, x: &Fq) -> Result<Fq> {
        let t = x.t.upsample_nearest2d(2)?;
        let gs = self.grids_of(x);
        if self.f32_acts.contains(&x.t.id()) {
            Ok(self.wrap_f32(t, x.q, gs))
        } else {
            Ok(self.wrap_u8(t, x.q, gs))
        }
    }

    fn space_to_depth2_parts(&mut self, x: &Fq, _consumer_is_f32: bool) -> Result<[Fq; 4]> {
        let c = x.t.dims4()?.1;
        let spd = space_to_depth_2x2(&x.t)?;
        let gs = self.grids_of(x);
        let is_f32 = self.f32_acts.contains(&x.t.id());
        let mut mk = |phase: usize| -> Result<Fq> {
            let t = spd.narrow(1, phase * c, c)?;
            if is_f32 { Ok(self.wrap_f32(t, x.q, gs.clone())) } else { Ok(self.wrap_u8(t, x.q, gs.clone())) }
        };
        Ok([mk(0)?, mk(1)?, mk(2)?, mk(3)?])
    }

    fn dims(&self, x: &Fq) -> [usize; 4] {
        dims4(&x.t)
    }
}
