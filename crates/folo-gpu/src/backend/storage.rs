//! Storages and views on wgpu buffers: a view's buffer, new result storages, dense copies of strided views and
//! index operands.

use std::sync::atomic::{AtomicBool, Ordering};

use folo_core::backend::View;
use folo_core::{DType as CoreDType, DeviceStorage, Error, Result, Storage};

use super::convert::{Core, elem, strided_of};
use super::{GpuBackend, GpuStorage, core_err};
use crate::error::GpuError;
use crate::kernels::strided;
use crate::pool::GpuBuffer;

static POISON: AtomicBool = AtomicBool::new(false);

/// Test aid (as `folo_core::cpu::poison_uninit`): while on, every buffer the GPU backend allocates for a result is
/// filled with NaN first, so an element a kernel never wrote (that would otherwise hold whatever a recycled
/// buffer held, such as a dense result of the same size) shows up as NaN.
pub fn poison_uninit(on: bool) {
    POISON.store(on, Ordering::Relaxed);
}

/// A dense (contiguous, offset 0) f32-or-other buffer for a view: borrowed when it already is, else a copy.
pub(super) enum Dense<'a> {
    Borrowed(&'a wgpu::Buffer),
    Owned(GpuBuffer),
}

impl Dense<'_> {
    pub(super) fn raw(&self) -> &wgpu::Buffer {
        match self {
            Dense::Borrowed(b) => b,
            Dense::Owned(b) => b.raw(),
        }
    }
}

impl GpuBackend {
    pub(super) fn buf<'a>(&self, v: &View<'a>) -> Result<&'a wgpu::Buffer> {
        match v.storage {
            Storage::Device(ds) if ds.device() == self.device => ds
                .downcast_ref::<GpuStorage>()
                .map(|s| s.buf.raw())
                .ok_or_else(|| core_err(GpuError::Shape("device storage is not a wgpu buffer".into()))),
            s => Err(Error::DeviceMismatch { op: "wgpu", lhs: self.device, rhs: s.device() }),
        }
    }

    /// The wgpu buffer of a whole storage on this device.
    pub(super) fn storage_buf<'a>(&self, s: &'a Storage) -> Result<&'a wgpu::Buffer> {
        match s {
            Storage::Device(ds) if ds.device() == self.device => ds
                .downcast_ref::<GpuStorage>()
                .map(|s| s.buf.raw())
                .ok_or_else(|| core_err(GpuError::Shape("device storage is not a wgpu buffer".into()))),
            s => Err(Error::DeviceMismatch { op: "wgpu", lhs: self.device, rhs: s.device() }),
        }
    }

    pub(super) fn alloc(&self, dtype: CoreDType, len: usize) -> Result<GpuBuffer> {
        let bytes = elem(dtype).buffer_bytes(len).core()?;
        let b = self.ctx.alloc(bytes).core()?;
        if POISON.load(Ordering::Relaxed) {
            // A quiet NaN in every f32 word and in both halves of every f16 pair.
            let nan = if elem(dtype).bytes() == 2 { 0x7e00_7e00 } else { 0x7fc0_0000 };
            strided::fill_words(&self.ctx, b.raw(), bytes as usize / 4, nan, nan).core()?;
        }
        Ok(b)
    }

    pub(super) fn wrap(&self, dtype: CoreDType, len: usize, buf: GpuBuffer) -> Storage {
        Storage::Device(DeviceStorage::new(self.device, dtype, len, GpuStorage { buf }))
    }

    pub(super) fn dense<'a>(&self, v: &View<'a>) -> Result<Dense<'a>> {
        let b = self.buf(v)?;
        if v.layout.is_contiguous() && v.layout.offset() == 0 {
            return Ok(Dense::Borrowed(b));
        }
        let dtype = v.storage.dtype();
        let out = self.alloc(dtype, v.layout.numel())?;
        strided::gather_strided(&self.ctx, elem(dtype), b, &strided_of(v.layout), out.raw()).core()?;
        Ok(Dense::Owned(out))
    }

    pub(super) fn check_fits(&self, v: &View<'_>, op: &'static str) -> Result<()> {
        if !v.layout.fits(v.storage.len()) {
            return Err(Error::InvalidShape {
                op,
                shape: v.layout.shape().clone(),
                msg: "layout reaches outside its storage".into(),
            });
        }
        Ok(())
    }

    pub(super) fn indices<'a>(&self, v: &View<'a>, op: &'static str) -> Result<(Dense<'a>, bool)> {
        let i64 = match v.storage.dtype() {
            CoreDType::I64 => true,
            CoreDType::U32 => false,
            got => return Err(Error::DTypeMismatch { op, expected: CoreDType::I64, got }),
        };
        Ok((self.dense(v)?, i64))
    }
}
