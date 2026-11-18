//! The int8 executor (docs/MODEL.md §9.6): the quantised graph ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]). Activations are u8 C4 buffers from the CPU buffer cache; each conv's packed weights
//! are built on its first call and kept with its weight tensor ([`Tensor::derived_f32`]) until that is written.

use std::sync::Arc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut};
use folo_nn::{Activation, ConvBnAct, Error, Result};

use super::calib::conv_f32;
use super::walk::{self, Exec};
use super::{Calibration, int8_able};
use crate::model::FoloModel;

/// A calibrated model's int8 plan. A conv's packed weights are kept with its weight tensor
/// ([`Tensor::derived_f32`]) under a key of everything else they are made from (the bias's bits, the geometry,
/// the activation, and the input parts' channels and grids): a write to the weights drops them, even one that
/// overlaps a forward, and the next forward packs them again.
pub struct Int8Plan {
    cal: Calibration,
}

impl Int8Plan {
    /// The plan of `cal`.
    pub fn new(cal: Calibration) -> Self {
        Int8Plan { cal }
    }

    /// The calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.cal
    }
}

/// A C4 buffer that goes back to the CPU buffer cache when its last activation is dropped.
struct Buf(Vec<u32>);

impl Drop for Buf {
    fn drop(&mut self) {
        q8::recycle(std::mem::take(&mut self.0));
    }
}

/// A u8 C4 activation (or a channel slice of one).
#[derive(Clone)]
pub struct QAct {
    buf: Arc<Buf>,
    offset: usize,
    img_stride: usize,
    dims: [usize; 4],
    grid: Grid,
}

impl QAct {
    fn view(&self) -> C4<'_> {
        C4 {
            data: &self.buf.0,
            offset: self.offset,
            img_stride: self.img_stride,
            c: self.dims[1],
            zero: self.grid.zero,
        }
    }

    fn new(buf: Vec<u32>, dims: [usize; 4], grid: Grid) -> QAct {
        let img_stride = dims[1] / 4 * dims[2] * dims[3];
        QAct { buf: Arc::new(Buf(buf)), offset: 0, img_stride, dims, grid }
    }

    fn from_f32(x: &Tensor, grid: Grid) -> Result<QAct> {
        let (n, c, h, w) = x.dims4()?;
        let mut out = q8::alloc(n, c, h * w);
        x.with_f32_slice(|v| q8::quantize(v, n, c, h * w, grid, &mut out))?;
        Ok(QAct::new(out, [n, c, h, w], grid))
    }

    fn to_f32(&self) -> Result<Tensor> {
        let [n, c, h, w] = self.dims;
        // From the CPU buffer cache (every element is written); it goes back when the tensor drops.
        let mut out = folo_core::cpu::take_buffer(n * c * h * w);
        q8::dequantize(self.view(), n, h * w, self.grid, &mut out);
        Ok(Tensor::from_vec(out, [n, c, h, w])?)
    }
}

/// Everything a conv's packed weights are made from besides the weights' elements, kept with the packing so a
/// hash collision cannot pass for a hit.
#[derive(Clone, PartialEq, Hash)]
struct Key {
    /// The weight view's dims, offset and strides (the elements it reads).
    layout: (Vec<usize>, usize, Vec<usize>),
    geometry: [usize; 5],
    silu: bool,
    /// Each input part's channels, scale bits and zero point.
    inputs: Vec<(usize, u32, u8)>,
    bias_bits: Vec<u32>,
}

/// A packed conv and its [`Key`].
struct Packed {
    key: Key,
    conv: Arc<QConv>,
}

/// The int8 executor of one forward.
struct Int8<'a> {
    plan: &'a Int8Plan,
}

impl Int8<'_> {
    /// `conv`'s packed weights for inputs `parts`: kept with the weight tensor, keyed by the rest.
    fn qconv(&self, conv: &ConvBnAct, parts: &[&QAct]) -> Result<Arc<QConv>> {
        use std::hash::{Hash, Hasher};
        let cfg = conv.conv.config();
        let bias = match &conv.conv.bias {
            Some(b) => b.to_vec::<f32>()?,
            None => vec![0.0; cfg.out_channels],
        };
        let geometry = [cfg.in_channels, cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
        let silu = conv.act == Activation::Silu;
        let inputs: Vec<(usize, u32, u8)> =
            parts.iter().map(|a| (a.dims[1], a.grid.scale.to_bits(), a.grid.zero)).collect();
        let bias_bits: Vec<u32> = bias.iter().map(|b| b.to_bits()).collect();
        let l = conv.conv.weight.layout();
        let layout = (l.shape().dims().to_vec(), l.offset(), l.strides().to_vec());
        let key = Key { layout, geometry, silu, inputs, bias_bits };
        let mut h = std::hash::DefaultHasher::new();
        ("folo-vision int8 QConv", &key).hash(&mut h);
        let pack = |w: &[f32]| {
            let grids: Vec<Grid> =
                parts.iter().flat_map(|a| std::iter::repeat_n(a.grid, a.dims[1])).collect();
            let shape = (cfg.out_channels, cfg.in_channels, cfg.kernel);
            Arc::new(QConv::new(w, &bias, shape, cfg.stride, cfg.padding, silu, &grids))
        };
        let kept =
            conv.conv.weight.derived_f32(h.finish(), |w| Packed { key: key.clone(), conv: pack(w) })?;
        if kept.key == key {
            return Ok(kept.conv.clone());
        }
        // Another key under the same hash: pack these weights without keeping them.
        Ok(pack(&conv.conv.weight.to_vec::<f32>()?))
    }

    /// The int8 conv into `out`.
    fn run(
        &self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QAct],
        res: Option<&QAct>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let [n, _, h, w] = parts[0].dims;
        if parts.iter().any(|a| a.dims[0] != n || a.dims[2] != h || a.dims[3] != w) {
            return Err(Error::Invalid(format!("int8 {name}: parts of different shapes")));
        }
        let qc = self.qconv(conv, parts)?;
        let views: Vec<C4> = parts.iter().map(|a| a.view()).collect();
        let (oh, ow) = qc.out_hw(h, w);
        let res = res.map(|r| (r.view(), r.grid));
        q8::conv(&views, n, h, w, &qc, res, out);
        Ok([n, qc.out_channels(), oh, ow])
    }
}

impl Exec for Int8<'_> {
    type A = QAct;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<QAct> {
        use folo_nn::Forward;
        QAct::from_f32(&conv.forward(x)?, self.plan.cal.param(name)?)
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct], res: Option<&QAct>) -> Result<QAct> {
        let grid = self.plan.cal.param(name)?;
        let [n, _, h, w] = parts[0].dims;
        let c = parts.iter().map(|a| a.dims[1]).sum();
        let o = conv.conv.config().out_channels;
        let able = int8_able(conv, [n, c, h, w], res.map(|r| r.dims));
        let (Some((oh, ow)), false) = (able, self.plan.cal.is_f32(name)) else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32(&y, grid);
        };
        let mut out = q8::alloc(n, o, oh * ow);
        let dims = self.run(name, conv, parts, res, QOut::U8 { data: &mut out, grid })?;
        Ok(QAct::new(out, dims, grid))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct) -> Result<Tensor> {
        let (Some((oh, ow)), false) = (int8_able(conv, x.dims, None), self.plan.cal.is_f32(name)) else {
            return conv_f32(conv, &[&x.to_f32()?], None);
        };
        let (n, o) = (x.dims[0], conv.conv.config().out_channels);
        // Every element of the output is written.
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims = self.run(name, conv, &[x], None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<QAct> {
        QAct::from_f32(x, self.plan.cal.param(name)?)
    }

    fn dequantize(&mut self, x: &QAct) -> Result<Tensor> {
        x.to_f32()
    }

    fn narrow(&mut self, x: &QAct, start: usize, len: usize) -> Result<QAct> {
        let [n, c, h, w] = x.dims;
        if !start.is_multiple_of(4) || !len.is_multiple_of(4) || start + len > c {
            return Err(Error::Invalid(format!(
                "int8: channels {start}..{} of {c} are not whole C4 groups",
                start + len
            )));
        }
        let mut y = x.clone();
        y.offset += start / 4 * h * w;
        y.dims = [n, len, h, w];
        Ok(y)
    }

    fn max_pool(&mut self, x: &QAct, k: usize) -> Result<QAct> {
        let [n, c, h, w] = x.dims;
        let mut out = q8::alloc(n, c, h * w);
        q8::max_pool(x.view(), n, h, w, k, &mut out);
        Ok(QAct::new(out, x.dims, x.grid))
    }

    fn upsample2(&mut self, x: &QAct) -> Result<QAct> {
        let [n, c, h, w] = x.dims;
        let mut out = q8::alloc(n, c, 4 * h * w);
        q8::upsample2(x.view(), n, h, w, &mut out);
        Ok(QAct::new(out, [n, c, 2 * h, 2 * w], x.grid))
    }

    fn dims(&self, x: &QAct) -> [usize; 4] {
        x.dims
    }
}

/// Layers 0–22 of `model` on `x` with the int8 plan: the head's three inputs (`f32`). Call it on the CPU, inside
/// folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features(model: &FoloModel, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    walk::features(model, &mut Int8 { plan }, x)
}
