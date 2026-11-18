//! The fake-quant reference (docs/MODEL.md §9.6): the int8 plan's arithmetic in `f32`. Every activation is
//! rounded onto its boundary's u8 grid, every quantised conv's weights onto their s8 grid (each input part's
//! scale folded into its columns first, as the int8 plan does), and everything else is the `f32` model.

use std::collections::HashMap;

use folo_core::Tensor;
use folo_nn::{ConvBnAct, Forward, Result};

use super::walk::{self, Exec};
use super::{Calibration, QParam, int8_able, quantize_weights};
use crate::model::FoloModel;

/// A fake-quantised activation: its values (on the grid) and the grid.
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
    weights: HashMap<String, Tensor>,
}

/// Round every element of `t` onto `q`.
pub fn fake_quantize(t: &Tensor, q: QParam) -> Result<Tensor> {
    let v: Vec<f32> = t.to_vec::<f32>()?.into_iter().map(|x| q.dequantize(q.quantize(x))).collect();
    Ok(Tensor::from_vec(v, t.dims().to_vec())?)
}

/// `w` (`[O, C, kh, kw]`) on the s8 grid of the int8 plan, given the scale of each input channel's part:
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
        FakeQuant { cal, weights: HashMap::new() }
    }

    /// Layers 0–22 of `model` on `x` with fake quantisation: the head's three inputs.
    pub fn features(&mut self, model: &FoloModel, x: &Tensor) -> Result<[Tensor; 3]> {
        let _g = folo_core::no_grad();
        folo_core::cpu::in_pool(|| walk::features(model, self, x))
    }

    /// The conv's weights on their s8 grid, or its `f32` weights when it stays `f32` in the int8 plan.
    fn weight(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Tensor> {
        let d = parts[0].t.dims();
        let c = parts.iter().map(|p| p.t.dims()[1]).sum();
        if self.cal.is_f32(name) || int8_able(conv, [d[0], c, d[2], d[3]], res.map(|r| dims4(&r.t))).is_none()
        {
            return Ok(conv.conv.weight.clone());
        }
        if let Some(w) = self.weights.get(name) {
            return Ok(w.clone());
        }
        let scales: Vec<f32> =
            parts.iter().flat_map(|p| std::iter::repeat_n(p.q.scale, p.t.dims()[1])).collect();
        let w = fake_weights(&conv.conv.weight, &scales)?;
        self.weights.insert(name.to_string(), w.clone());
        Ok(w)
    }

    fn run(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Tensor> {
        let w = self.weight(name, conv, parts, res)?;
        let ts: Vec<&Tensor> = parts.iter().map(|p| &p.t).collect();
        let p = conv.conv.config().params();
        Ok(Tensor::conv2d_act_cat(&ts, &w, conv.conv.bias.as_ref(), &p, conv.act.unary_op())?)
    }
}

impl Exec for FakeQuant<'_> {
    type A = Fq;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        Ok(Fq { t: fake_quantize(&conv.forward(x)?, q)?, q })
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&Fq], res: Option<&Fq>) -> Result<Fq> {
        let y = self.run(name, conv, parts, res)?;
        let y = match res {
            Some(r) => r.t.add(&y)?,
            None => y,
        };
        let q = self.cal.param(name)?;
        Ok(Fq { t: fake_quantize(&y, q)?, q })
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &Fq) -> Result<Tensor> {
        self.run(name, conv, &[x], None)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<Fq> {
        let q = self.cal.param(name)?;
        Ok(Fq { t: fake_quantize(x, q)?, q })
    }

    fn dequantize(&mut self, x: &Fq) -> Result<Tensor> {
        Ok(x.t.clone())
    }

    fn narrow(&mut self, x: &Fq, start: usize, len: usize) -> Result<Fq> {
        Ok(Fq { t: x.t.narrow(1, start, len)?, q: x.q })
    }

    fn max_pool(&mut self, x: &Fq, k: usize) -> Result<Fq> {
        Ok(Fq { t: x.t.max_pool2d(k, 1, k / 2)?, q: x.q })
    }

    fn upsample2(&mut self, x: &Fq) -> Result<Fq> {
        Ok(Fq { t: x.t.upsample_nearest2d(2)?, q: x.q })
    }

    fn dims(&self, x: &Fq) -> [usize; 4] {
        dims4(&x.t)
    }
}
