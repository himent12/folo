//! The fake-quant reference (docs/MODEL.md §9.6): the int8 plan's arithmetic in `f32`. Every activation is
//! rounded onto its boundary's u8 grid (per-channel in `version: 2`, per-tensor in `version: 1`), every
//! quantised conv's weights onto their s8 grid (mirroring per-part concat scale decoupling and direct `u8` stem),
//! and everything else is the `f32` model.

use std::collections::HashMap;

use folo_core::cpu::q8::{PIXEL_SCALE, should_decouple_parts};
use folo_core::{Tensor, TensorId};
use folo_nn::{ConvBnAct, Forward, Result};

use super::walk::{self, Exec};
use super::{Calibration, QParam, int8_able, quantize_weights, stem_int8_able};
use crate::model::FoloModel;

/// A fake-quantised activation: its values (on the grid) and its per-tensor fallback grid.
#[derive(Clone)]
pub struct Fq {
    /// The dequantised values.
    pub t: Tensor,
    /// The grid.
    pub q: QParam,
}

/// The fake-quant executor of a calibrated model.
pub struct FakeQuant<'a> {
    cal: &'a Calibration,
    /// Per-channel grids tracked by tensor identity (keeps `Fq`'s public fields unchanged).
    ch_grids: HashMap<TensorId, Vec<QParam>>,
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
        FakeQuant { cal, ch_grids: HashMap::new() }
    }

    /// Layers 0–22 of `model` on `x` with fake quantisation: the head's three inputs.
    pub fn features(&mut self, model: &FoloModel, x: &Tensor) -> Result<[Tensor; 3]> {
        self.ch_grids.clear();
        let _g = folo_core::no_grad();
        let res = folo_core::cpu::in_pool(|| walk::features(model, self, x));
        self.ch_grids.clear();
        res
    }

    fn grids_of(&self, a: &Fq) -> Vec<QParam> {
        if let Some(gs) = self.ch_grids.get(&a.t.id()) {
            return gs.clone();
        }
        vec![a.q; a.t.dims()[1]]
    }

    fn wrap(&mut self, t: Tensor, q: QParam, grids: Vec<QParam>) -> Fq {
        self.ch_grids.insert(t.id(), grids);
        Fq { t, q }
    }

    fn run(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Tensor> {
        let d = parts[0].t.dims();
        let c: usize = parts.iter().map(|p| p.t.dims()[1]).sum();
        let p = conv.conv.config().params();
        if self.cal.is_f32(name) || int8_able(conv, [d[0], c, d[2], d[3]], res.map(|r| dims4(&r.t))).is_none()
        {
            let ts: Vec<&Tensor> = parts.iter().map(|p| &p.t).collect();
            return Ok(Tensor::conv2d_act_cat(
                &ts,
                &conv.conv.weight,
                conv.conv.bias.as_ref(),
                &p,
                conv.act.unary_op(),
            )?);
        }
        // Compute in the well-conditioned integer-code domain `(q_x - z) * (q_w * s_out)` so tiny/dead
        // per-channel `in_scales[ic]` (down to `1e-8`) never blow up `w / in_scales[ic]` inside f32 Winograd.
        let owned_grids: Vec<Vec<QParam>> = parts.iter().map(|a| self.grids_of(a)).collect();
        let grid_refs: Vec<&[QParam]> = owned_grids.iter().map(Vec::as_slice).collect();
        let decouple = self.cal.version >= super::VERSION_V2 && should_decouple_parts(&grid_refs);
        let norm_parts: Vec<Tensor> = parts
            .iter()
            .zip(&owned_grids)
            .map(|(a, gs)| divide_by_scales(&a.t, gs))
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
        conv.act.forward(&acc)
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

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        let o = conv.conv.config().out_channels;
        let grids = self.cal.channel_params(name, o)?;
        let y = if self.cal.u8_stem && !self.cal.is_f32(name) && stem_int8_able(conv, dims4(x)).is_some() {
            let xq_vec: Vec<f32> = x
                .to_vec::<f32>()?
                .into_iter()
                .map(|v| (v * 255.0).clamp(0.0, 255.0).round_ties_even() * PIXEL_SCALE)
                .collect();
            let xq = Tensor::from_vec(xq_vec, x.dims().to_vec())?;
            let w = fake_weights(&conv.conv.weight, &[PIXEL_SCALE; 3])?;
            let p = conv.conv.config().params();
            Tensor::conv2d_act_cat(&[&xq], &w, conv.conv.bias.as_ref(), &p, conv.act.unary_op())?
        } else {
            conv.forward(x)?
        };
        let t = fake_quantize_per_channel(&y, &grids)?;
        Ok(self.wrap(t, q, grids))
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Fq> {
        let y = self.run(name, conv, parts, res)?;
        let y = match res {
            Some(r) => r.t.add(&y)?,
            None => y,
        };
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, y.dims()[1])?;
        let t = fake_quantize_per_channel(&y, &grids)?;
        Ok(self.wrap(t, q, grids))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &Fq) -> Result<Tensor> {
        self.run(name, conv, &[x], None)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        let grids = self.cal.channel_params(name, x.dims()[1])?;
        let t = fake_quantize_per_channel(x, &grids)?;
        Ok(self.wrap(t, q, grids))
    }

    fn dequantize(&mut self, x: &Fq) -> Result<Tensor> {
        Ok(x.t.clone())
    }

    fn narrow(&mut self, x: &Fq, start: usize, len: usize) -> Result<Fq> {
        let t = x.t.narrow(1, start, len)?;
        let gs = self.grids_of(x)[start..start + len].to_vec();
        Ok(self.wrap(t, x.q, gs))
    }

    fn max_pool(&mut self, x: &Fq, k: usize) -> Result<Fq> {
        let t = x.t.max_pool2d(k, 1, k / 2)?;
        let gs = self.grids_of(x);
        Ok(self.wrap(t, x.q, gs))
    }

    fn upsample2(&mut self, x: &Fq) -> Result<Fq> {
        let t = x.t.upsample_nearest2d(2)?;
        let gs = self.grids_of(x);
        Ok(self.wrap(t, x.q, gs))
    }

    fn dims(&self, x: &Fq) -> [usize; 4] {
        dims4(&x.t)
    }
}
