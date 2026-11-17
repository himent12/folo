//! The int8 executor (docs/MODEL.md §9.6): the quantised graph ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]). Activations are u8 C4 buffers from the CPU buffer cache; each conv's packed weights
//! are built on its first call and kept in the [`Int8Plan`] while its weight and bias tensors hold the same data.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut};
use folo_nn::{Activation, ConvBnAct, Error, Result};

use super::Calibration;
use super::calib::conv_f32;
use super::walk::{self, Exec};
use crate::model::FoloModel;

/// A calibrated model's int8 plan: its calibration and the packed weights of every int8 conv, built on first
/// use (the graph is fixed, so a conv's input grids are too) and built again when the conv's tensors change.
pub struct Int8Plan {
    cal: Calibration,
    convs: Mutex<HashMap<String, (Stamp, Arc<QConv>)>>,
}

/// Which data a conv's packed weights were made from: the weight and bias tensors' identities and storage versions
/// (a load, an in-place write or a replaced tensor changes one of them).
type Stamp = [Option<(folo_core::TensorId, u64)>; 2];

fn stamp(conv: &ConvBnAct) -> Stamp {
    let s = |t: &Tensor| (t.id(), t.storage_version());
    [Some(s(&conv.conv.weight)), conv.conv.bias.as_ref().map(s)]
}

impl Int8Plan {
    /// The plan of `cal`.
    pub fn new(cal: Calibration) -> Self {
        Int8Plan { cal, convs: Mutex::new(HashMap::new()) }
    }

    /// The calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.cal
    }

    /// Forget the packed weights (after the model's weights changed).
    pub fn clear(&self) {
        self.convs.lock().unwrap_or_else(|e| e.into_inner()).clear();
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

/// Whether `conv` runs on the int8 kernel: fused, dense, square, stride and padding uniform, SiLU or identity,
/// channels in multiples of 4.
fn int8_able(conv: &ConvBnAct) -> bool {
    let cfg = conv.conv.config();
    conv.bn.is_none()
        && cfg.groups == 1
        && cfg.dilation == 1
        && cfg.in_channels.is_multiple_of(4)
        && cfg.out_channels.is_multiple_of(4)
        && matches!(conv.act, Activation::Silu | Activation::Identity)
}

/// The int8 executor of one forward.
struct Int8<'a> {
    plan: &'a Int8Plan,
}

impl Int8<'_> {
    fn qconv(&self, name: &str, conv: &ConvBnAct, parts: &[&QAct]) -> Result<Arc<QConv>> {
        let st = stamp(conv);
        if let Some((s, c)) = self.plan.convs.lock().unwrap_or_else(|e| e.into_inner()).get(name)
            && *s == st
        {
            return Ok(c.clone());
        }
        let cfg = conv.conv.config();
        let w = conv.conv.weight.to_vec::<f32>()?;
        let bias = match &conv.conv.bias {
            Some(b) => b.to_vec::<f32>()?,
            None => vec![0.0; cfg.out_channels],
        };
        let grids: Vec<Grid> = parts.iter().flat_map(|a| std::iter::repeat_n(a.grid, a.dims[1])).collect();
        let qc = Arc::new(QConv::new(
            &w,
            &bias,
            (cfg.out_channels, cfg.in_channels, cfg.kernel),
            cfg.stride,
            cfg.padding,
            conv.act == Activation::Silu,
            &grids,
        ));
        self.plan.convs.lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string(), (st, qc.clone()));
        Ok(qc)
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
        let qc = self.qconv(name, conv, parts)?;
        let [n, _, h, w] = parts[0].dims;
        if parts.iter().any(|a| a.dims[0] != n || a.dims[2] != h || a.dims[3] != w) {
            return Err(Error::Invalid(format!("int8 {name}: parts of different shapes")));
        }
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
        if self.plan.cal.is_f32(name) || !int8_able(conv) {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32(&y, grid);
        }
        let [n, _, h, w] = parts[0].dims;
        let cfg = conv.conv.config();
        let (k, s, pd) = (cfg.kernel, cfg.stride, cfg.padding);
        let (oh, ow) = ((h + 2 * pd - k) / s + 1, (w + 2 * pd - k) / s + 1);
        let mut out = q8::alloc(n, cfg.out_channels, oh * ow);
        let dims = self.run(name, conv, parts, res, QOut::U8 { data: &mut out, grid })?;
        Ok(QAct::new(out, dims, grid))
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct) -> Result<Tensor> {
        if self.plan.cal.is_f32(name) || !int8_able(conv) {
            return conv_f32(conv, &[&x.to_f32()?], None);
        }
        let [n, _, h, w] = x.dims;
        let o = conv.conv.config().out_channels;
        // Every element of the output is written (a stride-1 conv's output is no larger than its input).
        let mut out = folo_core::cpu::take_buffer(n * o * h * w);
        let dims = self.run(name, conv, &[x], None, QOut::F32 { data: &mut out })?;
        out.truncate(dims.iter().product());
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
