//! The int8 executor (docs/MODEL.md §9.6): the quantised v1/v2 graphs ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]).
//!
//! Activations live in reusable C4 `u32` scratch buffers ([`super::pool::ScratchPool`]) checked out once per
//! forward pass from [`Int8Plan`], eliminating per-layer `Arc` heap allocations and small-buffer `malloc`/`free`
//! churn. Per-channel boundary grids (`version: 2`, or lazily cached uniform grids for `version: 1`) are stored in
//! a single FNV-1a table with precomputed half-split hashes and per-part max scales; each conv's packed weights
//! (and direct `u8` stem) are kept with its weight tensor ([`Tensor::derived_f32`]) until written.

mod plan;
mod v2;

pub use plan::Int8Plan;
pub use v2::{features_v2, features_v2_rgb0_u8};

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut, QStem};
use folo_nn::{ConvBnAct, Error, Result};

use self::plan::GridOwner;
use super::cache::{PartKeyRef, load_qconv_view, load_qstem};
use super::calib::conv_f32;
use super::pool::{ScratchPool, SlotHandle};
use super::walk::{self, Exec};
use super::{ConvView, VERSION_V2, int8_able, static_int8_able, static_int8_able_rep, stem_int8_able};
use crate::model::FoloModel;
use crate::model::v2::{StemV2, WsQARepConv3x3};

/// A materialised u8 C4 activation (or a channel slice of one) backed by a pooled [`SlotHandle`].
#[derive(Clone)]
pub(super) struct QActU8<'a> {
    pub(super) buf: SlotHandle,
    pub(super) offset: usize,
    pub(super) img_stride: usize,
    pub(super) dims: [usize; 4],
    pub(super) bg: GridOwner<'a>,
    pub(super) grid_offset: usize,
    pub(super) grids_hash: u64,
    pub(super) max_scale: f32,
}

impl<'a> QActU8<'a> {
    #[inline]
    pub(super) fn grids(&self) -> &[Grid] {
        &self.bg.as_ref().grids[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    #[inline]
    pub(super) fn zeros(&self) -> &[u8] {
        &self.bg.as_ref().zeros[self.grid_offset..self.grid_offset + self.dims[1]]
    }

    #[inline]
    pub(super) fn view<'p>(&self, pool: &'p ScratchPool) -> C4<'p> {
        C4 {
            data: pool.slice(self.buf.slot()),
            offset: self.offset,
            img_stride: self.img_stride,
            c: self.dims[1],
            zero: self.zeros().first().copied().unwrap_or(0),
        }
    }

    pub(super) fn new(
        pool: &Rc<RefCell<ScratchPool>>,
        buf: Vec<u32>,
        dims: [usize; 4],
        bg: GridOwner<'a>,
    ) -> Self {
        let img_stride = dims[1] / 4 * dims[2] * dims[3];
        let (grids_hash, max_scale) = {
            let r = bg.as_ref();
            (r.hash, r.max_scale)
        };
        QActU8 {
            buf: SlotHandle::new(pool, buf),
            offset: 0,
            img_stride,
            dims,
            bg,
            grid_offset: 0,
            grids_hash,
            max_scale,
        }
    }

    pub(super) fn from_f32(
        pool: &Rc<RefCell<ScratchPool>>,
        x: &Tensor,
        bg: GridOwner<'a>,
        grid_offset: usize,
        grids_hash: u64,
        max_scale: f32,
    ) -> Result<Self> {
        let (n, c, h, w) = x.dims4()?;
        if !c.is_multiple_of(4) {
            return Err(Error::Invalid(format!("int8: {c} channels is not a multiple of 4")));
        }
        let mut out = pool.borrow_mut().take_buf(n, c, h * w);
        let res = {
            let slice = &bg.as_ref().grids[grid_offset..grid_offset + c];
            x.with_f32_slice(|v| q8::quantize_per_channel(v, n, c, h * w, slice, &mut out))
        };
        if let Err(e) = res {
            pool.borrow_mut().return_buf(out);
            return Err(e.into());
        }
        let img_stride = c / 4 * h * w;
        Ok(QActU8 {
            buf: SlotHandle::new(pool, out),
            offset: 0,
            img_stride,
            dims: [n, c, h, w],
            bg,
            grid_offset,
            grids_hash,
            max_scale,
        })
    }

    pub(super) fn to_f32(&self) -> Result<Tensor> {
        let [n, c, h, w] = self.dims;
        let mut out = folo_core::cpu::take_buffer(n * c * h * w);
        let pool = self.buf.pool().borrow();
        q8::dequantize_per_channel(self.view(&pool), n, h * w, self.grids(), &mut out);
        drop(pool);
        Ok(Tensor::from_vec(out, [n, c, h, w])?)
    }

    pub(super) fn narrow_slice(&self, start: usize, len: usize) -> Result<Self> {
        let [n, c, h, w] = self.dims;
        if !start.is_multiple_of(4) || !len.is_multiple_of(4) || start + len > c {
            return Err(Error::Invalid(format!(
                "int8: channels {start}..{} of {c} are not whole C4 groups",
                start + len
            )));
        }
        let grid_offset = self.grid_offset + start;
        let (grids_hash, max_scale) = self.bg.as_ref().slice_meta(grid_offset, len);
        Ok(QActU8 {
            buf: self.buf.clone(),
            offset: self.offset + start / 4 * h * w,
            img_stride: self.img_stride,
            dims: [n, len, h, w],
            bg: self.bg.clone(),
            grid_offset,
            grids_hash,
            max_scale,
        })
    }
}

/// An unquantised `f32` activation with deferred `u8` quantisation (materialised at most once in an inline
/// [`OnceCell`] if an int8 op consumes it, and passed through directly in `f32` to `f32` convs and the `f32` head).
pub(super) struct QActF32<'a> {
    pub(super) tensor: Tensor,
    pub(super) dims: [usize; 4],
    pub(super) bg: GridOwner<'a>,
    pub(super) grid_offset: usize,
    pub(super) grids_hash: u64,
    pub(super) max_scale: f32,
    pub(super) u8_cache: OnceCell<QActU8<'a>>,
}

pub(super) enum QActRepr<'a> {
    U8(QActU8<'a>),
    F32(QActF32<'a>),
}

/// An activation in the int8 executor: either a materialised u8 C4 buffer or a deferred `f32` tensor carrying
/// its boundary's per-channel affine grids.
pub struct QAct<'a> {
    pub(super) repr: QActRepr<'a>,
}

impl<'a> QAct<'a> {
    pub(super) fn new(
        pool: &Rc<RefCell<ScratchPool>>,
        buf: Vec<u32>,
        dims: [usize; 4],
        bg: GridOwner<'a>,
    ) -> Self {
        QAct { repr: QActRepr::U8(QActU8::new(pool, buf, dims, bg)) }
    }

    pub(super) fn from_f32_lazy(tensor: Tensor, bg: GridOwner<'a>) -> Result<Self> {
        let (n, c, h, w) = tensor.dims4()?;
        let (grids_hash, max_scale) = {
            let r = bg.as_ref();
            (r.hash, r.max_scale)
        };
        Ok(QAct {
            repr: QActRepr::F32(QActF32 {
                tensor,
                dims: [n, c, h, w],
                bg,
                grid_offset: 0,
                grids_hash,
                max_scale,
                u8_cache: OnceCell::new(),
            }),
        })
    }

    #[inline]
    pub(super) fn dims(&self) -> [usize; 4] {
        match &self.repr {
            QActRepr::U8(u) => u.dims,
            QActRepr::F32(f) => f.dims,
        }
    }

    pub(super) fn to_f32(&self) -> Result<Tensor> {
        match &self.repr {
            QActRepr::U8(u) => u.to_f32(),
            QActRepr::F32(f) => Ok(f.tensor.clone()),
        }
    }

    pub(super) fn as_u8(&self, pool: &Rc<RefCell<ScratchPool>>) -> Result<&QActU8<'a>> {
        match &self.repr {
            QActRepr::U8(u) => Ok(u),
            QActRepr::F32(f) => {
                if let Some(u) = f.u8_cache.get() {
                    return Ok(u);
                }
                let u = QActU8::from_f32(
                    pool,
                    &f.tensor,
                    f.bg.clone(),
                    f.grid_offset,
                    f.grids_hash,
                    f.max_scale,
                )?;
                Ok(f.u8_cache.get_or_init(|| u))
            }
        }
    }
}

/// The int8 executor of one forward pass.
pub(super) struct Int8<'a> {
    pub(super) plan: &'a Int8Plan,
    /// Optional pre-packed `[n, h, w]` `[R, G, B, 0]` input from `Letterbox::apply_rgb0_u8`.
    pub(super) rgb0_input: Option<(&'a [u32], [usize; 3])>,
    /// Per-forward reusable C4 scratch buffer pool.
    pub(super) pool: Rc<RefCell<ScratchPool>>,
}

impl<'a> Int8<'a> {
    /// `conv`'s packed weights for inputs `parts` (cached on `conv.conv.weight`).
    pub(super) fn qconv_view(&self, conv: ConvView<'_>, parts: &[&QActU8<'a>]) -> Result<Arc<QConv>> {
        fn to_ref<'b, 'a>(a: &'b QActU8<'a>) -> PartKeyRef<'b> {
            PartKeyRef {
                grids_arc: &a.bg.as_ref().grids,
                grid_offset: a.grid_offset,
                channels: a.dims[1],
                grids_hash: a.grids_hash,
                max_scale: a.max_scale,
            }
        }
        let allow_decouple = self.plan.cal.version >= VERSION_V2;
        if parts.len() <= 8 {
            let first = to_ref(parts[0]);
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = to_ref(a);
            }
            load_qconv_view(conv, &arr[..parts.len()], allow_decouple)
        } else {
            let pk: Vec<PartKeyRef<'_>> = parts.iter().map(|a| to_ref(a)).collect();
            load_qconv_view(conv, &pk, allow_decouple)
        }
    }

    #[cfg(test)]
    fn qconv(&self, conv: &ConvBnAct, parts: &[&QActU8<'a>]) -> Result<Arc<QConv>> {
        self.qconv_view(ConvView::from_cba(conv), parts)
    }

    /// The stem's packed `u8` weights (cached on `conv.conv.weight`).
    pub(super) fn qstem(&self, conv: &ConvBnAct) -> Result<Arc<QStem>> {
        load_qstem(conv)
    }

    fn max_pool_u8(&mut self, u: &QActU8<'a>, k: usize) -> QAct<'a> {
        let [n, c, h, w] = u.dims;
        let mut out = self.pool.borrow_mut().take_buf(n, c, h * w);
        {
            let pool = self.pool.borrow();
            q8::max_pool(u.view(&pool), n, h, w, k, &mut out);
        }
        QAct {
            repr: QActRepr::U8(QActU8 {
                buf: SlotHandle::new(&self.pool, out),
                offset: 0,
                img_stride: c / 4 * h * w,
                dims: u.dims,
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        }
    }

    /// Execute the int8 conv into `out`, borrowing all input C4 views from `self.pool` simultaneously.
    pub(super) fn run_view(
        &self,
        name: &str,
        conv: ConvView<'_>,
        parts: &[&QActU8<'a>],
        res: Option<&QActU8<'a>>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let [n, _, h, w] = parts[0].dims;
        if parts.iter().any(|a| a.dims[0] != n || a.dims[2] != h || a.dims[3] != w) {
            return Err(Error::Invalid(format!("int8 {name}: parts of different shapes")));
        }
        let qc = self.qconv_view(conv, parts)?;
        let (oh, ow) = qc.out_hw(h, w);
        let pool = self.pool.borrow();
        let res = res.map(|r| (r.view(&pool), r.grids()));
        if parts.len() <= 8 {
            let mut v_arr = [parts[0].view(&pool); 8];
            let mut z_arr: [&[u8]; 8] = [&[]; 8];
            for (i, a) in parts.iter().enumerate() {
                v_arr[i] = a.view(&pool);
                z_arr[i] = a.zeros();
            }
            q8::conv_per_channel(&v_arr[..parts.len()], &z_arr[..parts.len()], n, h, w, &qc, res, out);
        } else {
            let views: Vec<C4<'_>> = parts.iter().map(|a| a.view(&pool)).collect();
            let zeros: Vec<&[u8]> = parts.iter().map(|a| a.zeros()).collect();
            q8::conv_per_channel(&views, &zeros, n, h, w, &qc, res, out);
        }
        Ok([n, qc.out_channels(), oh, ow])
    }

    pub(super) fn run_parts_into(
        &self,
        name: &str,
        conv: ConvView<'_>,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let u8_res = res.map(|r| r.as_u8(&self.pool)).transpose()?;
        if parts.len() <= 8 {
            let first = parts[0].as_u8(&self.pool)?;
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = a.as_u8(&self.pool)?;
            }
            self.run_view(name, conv, &arr[..parts.len()], u8_res, out)
        } else {
            let u8_parts: Vec<&QActU8<'a>> =
                parts.iter().map(|a| a.as_u8(&self.pool)).collect::<Result<_>>()?;
            self.run_view(name, conv, &u8_parts, u8_res, out)
        }
    }

    fn upsample2_u8(&mut self, u: &QActU8<'a>) -> QAct<'a> {
        let [n, c, h, w] = u.dims;
        let mut out = self.pool.borrow_mut().take_buf(n, c, 4 * h * w);
        {
            let pool = self.pool.borrow();
            q8::upsample2(u.view(&pool), n, h, w, &mut out);
        }
        QAct {
            repr: QActRepr::U8(QActU8 {
                buf: SlotHandle::new(&self.pool, out),
                offset: 0,
                img_stride: c / 4 * 4 * h * w,
                dims: [n, c, 2 * h, 2 * w],
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        }
    }

    pub(super) fn rgb0_to_f32_tensor(rgb0: &[u32], [n, h, w]: [usize; 3]) -> Result<Tensor> {
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
        Ok(Tensor::from_vec(planar, vec![n, 3, h, w])?)
    }
}

impl<'a> Exec for Int8<'a> {
    type A = QAct<'a>;

    fn wants_f32(&self, name: &str, conv: &ConvBnAct) -> bool {
        self.plan.is_f32(name) || !static_int8_able(conv)
    }

    fn wants_f32_rep(&self, name: &str, rep: &WsQARepConv3x3) -> bool {
        self.plan.is_f32(name) || !static_int8_able_rep(rep)
    }

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<QAct<'a>> {
        use folo_nn::Forward;
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.plan.boundary_for(name, o)?;
        let (n, c, h, w) = match self.rgb0_input {
            Some((_, [n, h, w])) => (n, 3, h, w),
            None => x.dims4()?,
        };
        if self.plan.cal.u8_stem
            && !is_f32
            && let Some((oh, ow)) = stem_int8_able(conv, [n, c, h, w])
        {
            let qs = self.qstem(conv)?;
            let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
            if let Some((rgb0, _)) = self.rgb0_input {
                q8::stem_u8(
                    rgb0,
                    n,
                    h,
                    w,
                    &qs,
                    QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
                );
            } else {
                let mut rgb0 = self.pool.borrow_mut().take_buf(n, 4, h * w);
                let res = x.with_f32_slice(|v| q8::pack_f32_to_rgb0_u8(v, n, h, w, &mut rgb0));
                if let Err(e) = res {
                    let mut p = self.pool.borrow_mut();
                    p.return_buf(rgb0);
                    p.return_buf(out);
                    return Err(e.into());
                }
                q8::stem_u8(
                    &rgb0,
                    n,
                    h,
                    w,
                    &qs,
                    QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
                );
                self.pool.borrow_mut().return_buf(rgb0);
            }
            return Ok(QAct::new(&self.pool, out, [n, o, oh, ow], bg));
        }
        let y = if let Some((rgb0, dims)) = self.rgb0_input {
            let x_f32 = Self::rgb0_to_f32_tensor(rgb0, dims)?;
            conv.forward(&x_f32)?
        } else {
            conv.forward(x)?
        };
        QAct::from_f32_lazy(y, bg)
    }

    fn stem_v2(&mut self, name: &str, stem: &StemV2, x: &Tensor) -> Result<QAct<'a>> {
        self.stem_v2_impl(name, stem, x)
    }

    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
    ) -> Result<QAct<'a>> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.plan.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], res.map(QAct::dims)))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32_lazy(y, bg);
        };
        let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
        let dims = match self.run_parts_into(
            name,
            ConvView::from_cba(conv),
            parts,
            res,
            QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
        ) {
            Ok(d) => d,
            Err(e) => {
                self.pool.borrow_mut().return_buf(out);
                return Err(e);
            }
        };
        Ok(QAct::new(&self.pool, out, dims, bg))
    }

    fn conv_f32_out(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
    ) -> Result<QAct<'a>> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.plan.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], res.map(QAct::dims)))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let r = res.map(QAct::to_f32).transpose()?;
            let y = conv_f32(conv, &ts.iter().collect::<Vec<_>>(), r.as_ref())?;
            return QAct::from_f32_lazy(y, bg);
        };
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_parts_into(name, ConvView::from_cba(conv), parts, res, QOut::F32 { data: &mut out })?;
        let y = Tensor::from_vec(out, dims.to_vec())?;
        QAct::from_f32_lazy(y, bg)
    }

    fn rep_conv(
        &mut self,
        name: &str,
        rep: &WsQARepConv3x3,
        x: &QAct<'a>,
        out_f32: bool,
    ) -> Result<QAct<'a>> {
        self.rep_conv_impl(name, rep, x, out_f32)
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct<'a>) -> Result<Tensor> {
        let xd = x.dims();
        let (Some((oh, ow)), false) = (int8_able(conv, xd, None), self.plan.is_f32(name)) else {
            return conv_f32(conv, &[&x.to_f32()?], None);
        };
        let u8_x = x.as_u8(&self.pool)?;
        let (n, o) = (xd[0], conv.conv.config().out_channels);
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_view(name, ConvView::from_cba(conv), &[u8_x], None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct<'a>]) -> Result<Tensor> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (_, is_f32) = self.plan.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], None))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            return conv_f32(conv, &ts.iter().collect::<Vec<_>>(), None);
        };
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_parts_into(name, ConvView::from_cba(conv), parts, None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<QAct<'a>> {
        let c = x.dims4()?.1;
        let (bg, _) = self.plan.boundary_for(name, c)?;
        QAct::from_f32_lazy(x.clone(), bg)
    }

    fn dequantize(&mut self, x: &QAct<'a>) -> Result<Tensor> {
        x.to_f32()
    }

    fn ensure_u8(&mut self, x: &QAct<'a>) -> Result<()> {
        if let QActRepr::F32(f) = &x.repr
            && f.dims[1].is_multiple_of(4)
        {
            let _ = x.as_u8(&self.pool)?;
        }
        Ok(())
    }

    fn narrow(&mut self, x: &QAct<'a>, start: usize, len: usize) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(QAct { repr: QActRepr::U8(u.narrow_slice(start, len)?) }),
            QActRepr::F32(f) => {
                let [n, c, h, w] = f.dims;
                if start + len > c {
                    return Err(Error::Invalid(format!(
                        "int8: channels {start}..{} out of bounds for {c}",
                        start + len
                    )));
                }
                let grid_offset = f.grid_offset + start;
                let (grids_hash, max_scale) = f.bg.as_ref().slice_meta(grid_offset, len);
                let u8_cache = OnceCell::new();
                if let Some(u) = f.u8_cache.get()
                    && start.is_multiple_of(4)
                    && len.is_multiple_of(4)
                {
                    let _ = u8_cache.set(u.narrow_slice(start, len)?);
                }
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.narrow(1, start, len)?,
                        dims: [n, len, h, w],
                        bg: f.bg.clone(),
                        grid_offset,
                        grids_hash,
                        max_scale,
                        u8_cache,
                    }),
                })
            }
        }
    }

    fn max_pool(&mut self, x: &QAct<'a>, k: usize) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(self.max_pool_u8(u, k)),
            QActRepr::F32(f) => {
                if let Some(u) = f.u8_cache.get() {
                    return Ok(self.max_pool_u8(u, k));
                }
                let tensor = f.tensor.max_pool2d(k, 1, k / 2)?;
                let (n, c, oh, ow) = tensor.dims4()?;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor,
                        dims: [n, c, oh, ow],
                        bg: f.bg.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        max_scale: f.max_scale,
                        u8_cache: OnceCell::new(),
                    }),
                })
            }
        }
    }

    fn upsample2(&mut self, x: &QAct<'a>) -> Result<QAct<'a>> {
        self.upsample2_for(x, true)
    }

    fn upsample2_for(&mut self, x: &QAct<'a>, consumer_is_f32: bool) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(self.upsample2_u8(u)),
            QActRepr::F32(f) => {
                if !consumer_is_f32 && f.dims[1].is_multiple_of(4) {
                    let u = x.as_u8(&self.pool)?.clone();
                    return Ok(self.upsample2_u8(&u));
                }
                let [n, c, h, w] = f.dims;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.upsample_nearest2d(2)?,
                        dims: [n, c, 2 * h, 2 * w],
                        bg: f.bg.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        max_scale: f.max_scale,
                        u8_cache: OnceCell::new(),
                    }),
                })
            }
        }
    }

    fn space_to_depth2_parts(&mut self, x: &QAct<'a>, consumer_is_f32: bool) -> Result<[QAct<'a>; 4]> {
        self.space_to_depth2_parts_impl(x, consumer_is_f32)
    }

    fn dims(&self, x: &QAct<'a>) -> [usize; 4] {
        x.dims()
    }
}

fn run_walk(
    model: &FoloModel,
    plan: &Int8Plan,
    rgb0_input: Option<(&[u32], [usize; 3])>,
    x: &Tensor,
) -> Result<[Tensor; 3]> {
    let pool = plan.checkout_scratch();
    let prev_stats = pool.stats();
    let pool_rc = Rc::new(RefCell::new(pool));
    let mut exec = Int8 { plan, rgb0_input, pool: Rc::clone(&pool_rc) };
    let res = walk::features(model, &mut exec, x);
    drop(exec);
    if let Ok(cell) = Rc::try_unwrap(pool_rc) {
        plan.checkin_scratch(cell.into_inner(), prev_stats);
    }
    res
}

/// Layers 0–22 of `model` on `x` with the int8 plan: the head's three inputs (`f32`). Call it on the CPU, inside
/// folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features(model: &FoloModel, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    run_walk(model, plan, None, x)
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
    run_walk(model, plan, Some((rgb0, [n, h, w])), &dummy)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::blocks::Init;
    use crate::model::quant::cache::BoundaryGrids;
    use crate::model::quant::{Calibration, Method};
    use folo_core::rng::Rng;

    /// Weight and bias views sharing the same underlying storage `RwLock` must never deadlock during `qconv` or
    /// `qstem` packing (cache miss) or lookup (cache hit).
    #[test]
    fn aliased_weight_and_bias_storage_does_not_deadlock() {
        let mut rng = Rng::seed(99);
        let cal = Calibration {
            version: VERSION_V2,
            method: Method::MinMax,
            images: 1,
            ranges: BTreeMap::new(),
            channel_ranges: BTreeMap::new(),
            u8_stem: true,
            f32_convs: Vec::new(),
            weights: None,
        };
        let plan = Int8Plan::new(cal);
        let pool = Rc::new(RefCell::new(ScratchPool::default()));
        let exec = Int8 { plan: &plan, rgb0_input: None, pool: Rc::clone(&pool) };

        // Aliased conv (`bias` is a view into `weight`'s own storage).
        let mut conv = Init::new(&mut rng).conv(8, 8, 3, 1);
        conv.fuse_().unwrap();
        conv.conv.bias = Some(conv.conv.weight.reshape([8 * 8 * 9]).unwrap().narrow(0, 0, 8).unwrap());
        let g = Grid::from_range(-1.0, 1.0);
        let bg = Arc::new(BoundaryGrids::new(vec![g; 8]));
        let buf = pool.borrow_mut().take_buf(1, 8, 16);
        let act = QActU8::new(&pool, buf, [1, 8, 4, 4], GridOwner::Owned(bg));
        let first = exec.qconv(&conv, &[&act]).unwrap();
        let second = exec.qconv(&conv, &[&act]).unwrap();
        assert!(Arc::ptr_eq(&first, &second), "second qconv call must hit cache");

        // Aliased stem (`bias` is a view into `weight`'s own storage).
        let mut stem = Init::new(&mut rng).conv(3, 16, 3, 2);
        stem.fuse_().unwrap();
        stem.conv.bias = Some(stem.conv.weight.reshape([16 * 3 * 9]).unwrap().narrow(0, 0, 16).unwrap());
        let s1 = exec.qstem(&stem).unwrap();
        let s2 = exec.qstem(&stem).unwrap();
        assert!(Arc::ptr_eq(&s1, &s2), "second qstem call must hit cache");
    }
}
