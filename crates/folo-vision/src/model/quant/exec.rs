//! The int8 executor (docs/MODEL.md §9.6): the quantised graph ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]). Activations are u8 C4 buffers from the CPU buffer cache with per-channel grids
//! (`version: 2`, or uniform grids for `version: 1`); each conv's packed weights (and direct `u8` stem) are built
//! on its first call and kept with its weight tensor ([`Tensor::derived_f32`]) until that is written.

use std::sync::Arc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut, QStem};
use folo_nn::{Activation, ConvBnAct, Error, Result};

use super::calib::conv_f32;
use super::walk::{self, Exec};
use super::{Calibration, VERSION_V2, int8_able, stem_int8_able};
use crate::model::FoloModel;

/// A calibrated model's int8 plan. A conv's packed weights are kept with its weight tensor
/// ([`Tensor::derived_f32`]) under a key of everything else they are made from: a write to the weights drops
/// them, even one that overlaps a forward, and the next forward packs them again.
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

/// A u8 C4 activation (or a channel slice of one) with per-channel affine grids.
#[derive(Clone)]
pub struct QAct {
    buf: Arc<Buf>,
    offset: usize,
    img_stride: usize,
    dims: [usize; 4],
    grids: Arc<[Grid]>,
    zeros: Arc<[u8]>,
    grid_offset: usize,
}

impl QAct {
    fn grids(&self) -> &[Grid] {
        &self.grids[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    fn zeros(&self) -> &[u8] {
        &self.zeros[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    fn view(&self) -> C4<'_> {
        C4 {
            data: &self.buf.0,
            offset: self.offset,
            img_stride: self.img_stride,
            c: self.dims[1],
            zero: self.zeros()[0],
        }
    }

    fn new(buf: Vec<u32>, dims: [usize; 4], grids: Vec<Grid>) -> QAct {
        let img_stride = dims[1] / 4 * dims[2] * dims[3];
        let zeros: Vec<u8> = grids.iter().map(|g| g.zero).collect();
        QAct {
            buf: Arc::new(Buf(buf)),
            offset: 0,
            img_stride,
            dims,
            grids: grids.into(),
            zeros: zeros.into(),
            grid_offset: 0,
        }
    }

    fn from_f32(x: &Tensor, grids: Vec<Grid>) -> Result<QAct> {
        let (n, c, h, w) = x.dims4()?;
        let mut out = q8::alloc(n, c, h * w);
        x.with_f32_slice(|v| q8::quantize_per_channel(v, n, c, h * w, &grids, &mut out))?;
        Ok(QAct::new(out, [n, c, h, w], grids))
    }

    fn to_f32(&self) -> Result<Tensor> {
        let [n, c, h, w] = self.dims;
        let mut out = folo_core::cpu::take_buffer(n * c * h * w);
        q8::dequantize_per_channel(self.view(), n, h * w, self.grids(), &mut out);
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
    decouple: bool,
    /// Per-part channel counts.
    part_lens: Vec<usize>,
    /// Per-input-channel `(scale_bits, zero)`.
    in_grids: Vec<(u32, u8)>,
    bias_bits: Vec<u32>,
}

/// A packed conv and its [`Key`].
struct Packed {
    key: Key,
    conv: Arc<QConv>,
}

/// Cache key for a packed [`QStem`].
#[derive(Clone, PartialEq, Hash)]
struct StemKey {
    layout: (Vec<usize>, usize, Vec<usize>),
    geometry: [usize; 4],
    silu: bool,
    bias_bits: Vec<u32>,
}

struct PackedStem {
    key: StemKey,
    stem: Arc<QStem>,
}

/// The int8 executor of one forward.
struct Int8<'a> {
    plan: &'a Int8Plan,
    /// Optional pre-packed `[n, h, w]` `[R, G, B, 0]` input from `Letterbox::apply_rgb0_u8`.
    rgb0_input: Option<(&'a [u32], [usize; 3])>,
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
        let part_grids: Vec<&[Grid]> = parts.iter().map(|a| a.grids()).collect();
        let decouple = self.plan.cal.version >= VERSION_V2 && q8::should_decouple_parts(&part_grids);
        let part_lens: Vec<usize> = parts.iter().map(|a| a.dims[1]).collect();
        let in_grids: Vec<(u32, u8)> =
            part_grids.iter().flat_map(|gs| gs.iter().map(|g| (g.scale.to_bits(), g.zero))).collect();
        let bias_bits: Vec<u32> = bias.iter().map(|b| b.to_bits()).collect();
        let l = conv.conv.weight.layout();
        let layout = (l.shape().dims().to_vec(), l.offset(), l.strides().to_vec());
        let key = Key { layout, geometry, silu, decouple, part_lens, in_grids, bias_bits };
        let mut h = std::hash::DefaultHasher::new();
        ("folo-vision int8 QConv v2", &key).hash(&mut h);
        let pack = |w: &[f32]| {
            let shape = (cfg.out_channels, cfg.in_channels, cfg.kernel);
            Arc::new(QConv::new_parts(w, &bias, shape, cfg.stride, cfg.padding, silu, &part_grids, decouple))
        };
        let kept =
            conv.conv.weight.derived_f32(h.finish(), |w| Packed { key: key.clone(), conv: pack(w) })?;
        if kept.key == key {
            return Ok(kept.conv.clone());
        }
        Ok(pack(&conv.conv.weight.to_vec::<f32>()?))
    }

    /// The stem's packed `u8` weights, cached on the weight tensor.
    fn qstem(&self, conv: &ConvBnAct) -> Result<Arc<QStem>> {
        use std::hash::{Hash, Hasher};
        let cfg = conv.conv.config();
        let bias = match &conv.conv.bias {
            Some(b) => b.to_vec::<f32>()?,
            None => vec![0.0; cfg.out_channels],
        };
        let geometry = [cfg.out_channels, cfg.kernel, cfg.stride, cfg.padding];
        let silu = conv.act == Activation::Silu;
        let bias_bits: Vec<u32> = bias.iter().map(|b| b.to_bits()).collect();
        let l = conv.conv.weight.layout();
        let layout = (l.shape().dims().to_vec(), l.offset(), l.strides().to_vec());
        let key = StemKey { layout, geometry, silu, bias_bits };
        let mut h = std::hash::DefaultHasher::new();
        ("folo-vision int8 QStem v2", &key).hash(&mut h);
        let pack = |w: &[f32]| {
            Arc::new(QStem::new(w, &bias, (cfg.out_channels, cfg.kernel), cfg.stride, cfg.padding, silu))
        };
        let kept =
            conv.conv.weight.derived_f32(h.finish(), |w| PackedStem { key: key.clone(), stem: pack(w) })?;
        if kept.key == key {
            return Ok(kept.stem.clone());
        }
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
        let zeros: Vec<&[u8]> = parts.iter().map(|a| a.zeros()).collect();
        let (oh, ow) = qc.out_hw(h, w);
        let res = res.map(|r| (r.view(), r.grids()));
        q8::conv_per_channel(&views, &zeros, n, h, w, &qc, res, out);
        Ok([n, qc.out_channels(), oh, ow])
    }
}

impl Exec for Int8<'_> {
    type A = QAct;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<QAct> {
        use folo_nn::Forward;
        let o = conv.conv.config().out_channels;
        let grids = self.plan.cal.channel_params(name, o)?;
        let (n, c, h, w) = match self.rgb0_input {
            Some((_, [n, h, w])) => (n, 3, h, w),
            None => x.dims4()?,
        };
        if self.plan.cal.u8_stem
            && !self.plan.cal.is_f32(name)
            && let Some((oh, ow)) = stem_int8_able(conv, [n, c, h, w])
        {
            let qs = self.qstem(conv)?;
            let mut out = q8::alloc(n, o, oh * ow);
            if let Some((rgb0, _)) = self.rgb0_input {
                q8::stem_u8(rgb0, n, h, w, &qs, QOut::U8PerChannel { data: &mut out, grids: &grids });
            } else {
                let mut rgb0 = q8::alloc(n, 4, h * w);
                x.with_f32_slice(|v| q8::pack_f32_to_rgb0_u8(v, n, h, w, &mut rgb0))?;
                q8::stem_u8(&rgb0, n, h, w, &qs, QOut::U8PerChannel { data: &mut out, grids: &grids });
                q8::recycle(rgb0);
            }
            return Ok(QAct::new(out, [n, o, oh, ow], grids));
        }
        if let Some((rgb0, [n, h, w])) = self.rgb0_input {
            let hw = h * w;
            let mut planar = folo_core::cpu::take_buffer(n * 3 * hw);
            const PIXEL_SCALE: f32 = 1.0 / 255.0;
            for img in 0..n {
                let src = &rgb0[img * hw..(img + 1) * hw];
                let base = img * 3 * hw;
                for (p, &word) in src.iter().enumerate() {
                    let [r, g, b, _] = word.to_le_bytes();
                    planar[base + p] = f32::from(r) * PIXEL_SCALE;
                    planar[base + hw + p] = f32::from(g) * PIXEL_SCALE;
                    planar[base + 2 * hw + p] = f32::from(b) * PIXEL_SCALE;
                }
            }
            let x_f32 = Tensor::from_vec(planar, vec![n, 3, h, w])?;
            return QAct::from_f32(&conv.forward(&x_f32)?, grids);
        }
        QAct::from_f32(&conv.forward(x)?, grids)
    }

    fn conv(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct], res: Option<&QAct>) -> Result<QAct> {
        let [n, _, h, w] = parts[0].dims;
        let c = parts.iter().map(|a| a.dims[1]).sum();
        let o = conv.conv.config().out_channels;
        let grids = self.plan.cal.channel_params(name, o)?;
        let able = int8_able(conv, [n, c, h, w], res.map(|r| r.dims));
        let (Some((oh, ow)), false) = (able, self.plan.cal.is_f32(name)) else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32(&y, grids);
        };
        let mut out = q8::alloc(n, o, oh * ow);
        let dims = self.run(name, conv, parts, res, QOut::U8PerChannel { data: &mut out, grids: &grids })?;
        Ok(QAct::new(out, dims, grids))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct) -> Result<Tensor> {
        let (Some((oh, ow)), false) = (int8_able(conv, x.dims, None), self.plan.cal.is_f32(name)) else {
            return conv_f32(conv, &[&x.to_f32()?], None);
        };
        let (n, o) = (x.dims[0], conv.conv.config().out_channels);
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims = self.run(name, conv, &[x], None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<QAct> {
        let c = x.dims4()?.1;
        QAct::from_f32(x, self.plan.cal.channel_params(name, c)?)
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
        y.grid_offset += start;
        y.dims = [n, len, h, w];
        Ok(y)
    }

    fn max_pool(&mut self, x: &QAct, k: usize) -> Result<QAct> {
        let [n, c, h, w] = x.dims;
        let mut out = q8::alloc(n, c, h * w);
        q8::max_pool(x.view(), n, h, w, k, &mut out);
        Ok(QAct {
            buf: Arc::new(Buf(out)),
            offset: 0,
            img_stride: c / 4 * h * w,
            dims: x.dims,
            grids: x.grids.clone(),
            zeros: x.zeros.clone(),
            grid_offset: x.grid_offset,
        })
    }

    fn upsample2(&mut self, x: &QAct) -> Result<QAct> {
        let [n, c, h, w] = x.dims;
        let mut out = q8::alloc(n, c, 4 * h * w);
        q8::upsample2(x.view(), n, h, w, &mut out);
        Ok(QAct {
            buf: Arc::new(Buf(out)),
            offset: 0,
            img_stride: c / 4 * 4 * h * w,
            dims: [n, c, 2 * h, 2 * w],
            grids: x.grids.clone(),
            zeros: x.zeros.clone(),
            grid_offset: x.grid_offset,
        })
    }

    fn dims(&self, x: &QAct) -> [usize; 4] {
        x.dims
    }
}

/// Layers 0–22 of `model` on `x` with the int8 plan: the head's three inputs (`f32`). Call it on the CPU, inside
/// folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features(model: &FoloModel, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    walk::features(model, &mut Int8 { plan, rgb0_input: None }, x)
}

/// [`features`] reading a pre-packed `[n, h, w]` `u32` `[R, G, B, 0]` canvas (from `Letterbox::apply_rgb0_u8`)
/// straight into the direct `u8` AVX-VNNI stem with zero `f32` image materialisation.
pub fn features_rgb0_u8(
    model: &FoloModel,
    plan: &Int8Plan,
    rgb0: &[u32],
    n: usize,
    h: usize,
    w: usize,
) -> Result<[Tensor; 3]> {
    let dummy = Tensor::zeros([0]);
    walk::features(model, &mut Int8 { plan, rgb0_input: Some((rgb0, [n, h, w])) }, &dummy)
}
