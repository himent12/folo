//! Quantised activation representations (`QAct`, `QActU8`, `QActF32`) for the int8 executor.

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid};
use folo_nn::{Error, Result};

use super::plan::GridOwner;
use crate::model::quant::pool::{ScratchPool, SlotHandle};

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
