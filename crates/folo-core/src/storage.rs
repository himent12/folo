//! Raw element buffers behind tensors.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use half::f16;

use crate::{DType, Device, Error, Result};

/// A host buffer of one dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F16(Vec<f16>),
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
}

/// Run `$body` with `$v` bound to the inner `Vec` of a [`CpuStorage`], whatever its dtype.
#[macro_export]
#[doc(hidden)]
macro_rules! with_cpu_storage {
    ($s:expr, $v:ident => $body:expr) => {
        match $s {
            $crate::CpuStorage::F32($v) => $body,
            $crate::CpuStorage::F16($v) => $body,
            $crate::CpuStorage::U8($v) => $body,
            $crate::CpuStorage::U32($v) => $body,
            $crate::CpuStorage::I64($v) => $body,
        }
    };
}

impl CpuStorage {
    /// The element type.
    pub fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::U8(_) => DType::U8,
            CpuStorage::U32(_) => DType::U32,
            CpuStorage::I64(_) => DType::I64,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        with_cpu_storage!(self, v => v.len())
    }

    /// True if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A buffer of `len` copies of `value` (converted to `dtype`).
    pub fn full(dtype: DType, len: usize, value: f64) -> Self {
        match dtype {
            DType::F32 => CpuStorage::F32(vec![value as f32; len]),
            DType::F16 => CpuStorage::F16(vec![f16::from_f64(value); len]),
            DType::U8 => CpuStorage::U8(vec![value as u8; len]),
            DType::U32 => CpuStorage::U32(vec![value as u32; len]),
            DType::I64 => CpuStorage::I64(vec![value as i64; len]),
        }
    }

    /// The raw little-endian bytes (on little-endian hosts) of the buffer.
    pub fn as_bytes(&self) -> &[u8] {
        with_cpu_storage!(self, v => bytemuck::cast_slice(v))
    }

    /// A buffer from raw native-endian bytes. `bytes.len()` must be a multiple of the element size.
    pub fn from_bytes(bytes: &[u8], dtype: DType) -> Result<Self> {
        if !bytes.len().is_multiple_of(dtype.size_in_bytes()) {
            return Err(Error::invalid_arg(
                "from_bytes",
                format!("{} bytes is not a multiple of the {dtype} element size", bytes.len()),
            ));
        }
        fn read<T: bytemuck::Pod>(b: &[u8]) -> Vec<T> {
            // pod_collect_to_vec handles unaligned input.
            bytemuck::pod_collect_to_vec(b)
        }
        Ok(match dtype {
            DType::F32 => CpuStorage::F32(read(bytes)),
            DType::F16 => CpuStorage::F16(read(bytes)),
            DType::U8 => CpuStorage::U8(bytes.to_vec()),
            DType::U32 => CpuStorage::U32(read(bytes)),
            DType::I64 => CpuStorage::I64(read(bytes)),
        })
    }

    /// The `f32` slice, or a dtype error naming `op`.
    pub fn as_f32(&self, op: &'static str) -> Result<&[f32]> {
        match self {
            CpuStorage::F32(v) => Ok(v),
            s => Err(Error::DTypeMismatch { op, expected: DType::F32, got: s.dtype() }),
        }
    }

    /// The mutable `f32` slice, or a dtype error naming `op`.
    pub fn as_f32_mut(&mut self, op: &'static str) -> Result<&mut [f32]> {
        match self {
            CpuStorage::F32(v) => Ok(v),
            s => Err(Error::DTypeMismatch { op, expected: DType::F32, got: s.dtype() }),
        }
    }
}

/// A buffer owned by a device backend. folo-core never looks inside `buffer`; the backend that created it
/// downcasts it back to its own type.
pub struct DeviceStorage {
    device: Device,
    dtype: DType,
    len: usize,
    buffer: Box<dyn Any + Send + Sync>,
}

impl DeviceStorage {
    /// Wrap a backend buffer holding `len` elements of `dtype` on `device`.
    pub fn new(device: Device, dtype: DType, len: usize, buffer: impl Any + Send + Sync) -> Self {
        DeviceStorage { device, dtype, len, buffer: Box::new(buffer) }
    }

    /// The device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The backend buffer, if it is a `T`.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.buffer.downcast_ref()
    }

    /// The backend buffer, mutably, if it is a `T`.
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.buffer.downcast_mut()
    }
}

impl std::fmt::Debug for DeviceStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceStorage({:?}, {:?}, len {})", self.device, self.dtype, self.len)
    }
}

/// The buffer behind one or more tensors.
#[derive(Debug)]
pub enum Storage {
    /// Host memory.
    Cpu(CpuStorage),
    /// Memory owned by a registered device backend.
    Device(DeviceStorage),
}

impl Storage {
    /// The element type.
    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
            Storage::Device(s) => s.dtype(),
        }
    }

    /// The device.
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Device(s) => s.device(),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len(),
            Storage::Device(s) => s.len(),
        }
    }

    /// True if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host buffer, or an error naming `op` and `backend` for device storage.
    pub fn as_cpu(&self, op: &'static str) -> Result<&CpuStorage> {
        match self {
            Storage::Cpu(s) => Ok(s),
            Storage::Device(_) => Err(Error::NotSupported { op, backend: "cpu" }),
        }
    }

    /// The host buffer mutably, or an error naming `op` for device storage.
    pub fn as_cpu_mut(&mut self, op: &'static str) -> Result<&mut CpuStorage> {
        match self {
            Storage::Cpu(s) => Ok(s),
            Storage::Device(_) => Err(Error::NotSupported { op, backend: "cpu" }),
        }
    }
}

/// Shared, lockable storage with a version counter bumped on every in-place write. Autograd records the
/// version when an op saves its inputs and refuses to run backward if it changed.
///
/// The device is kept next to the data (not in each tensor) so [`crate::Tensor::to_device_`] can move a
/// storage in place: every tensor and view sharing it follows. It is an atomic so `Tensor::device()` never
/// takes the lock.
pub(crate) struct StorageCell {
    data: RwLock<Storage>,
    version: AtomicU64,
    device: AtomicU64,
}

/// `Cpu` as 0, `Gpu(i)` as `i + 1`.
fn encode_device(d: Device) -> u64 {
    match d {
        Device::Cpu => 0,
        Device::Gpu(i) => i as u64 + 1,
    }
}

fn decode_device(v: u64) -> Device {
    match v {
        0 => Device::Cpu,
        i => Device::Gpu((i - 1) as usize),
    }
}

impl StorageCell {
    pub fn new(s: Storage) -> Self {
        let device = AtomicU64::new(encode_device(s.device()));
        StorageCell { data: RwLock::new(s), version: AtomicU64::new(0), device }
    }

    /// The device of the data.
    #[inline]
    pub fn device(&self) -> Device {
        decode_device(self.device.load(Ordering::Acquire))
    }

    /// Replace the data with `new` (same dtype and length, any device) through an already-held write guard.
    pub fn replace(&self, guard: &mut RwLockWriteGuard<'_, Storage>, new: Storage) {
        let device = new.device();
        let old = std::mem::replace(&mut **guard, new);
        self.device.store(encode_device(device), Ordering::Release);
        if let Storage::Cpu(CpuStorage::F32(v)) = old {
            crate::cpu::recycle(v);
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, Storage> {
        self.data.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Write access; bumps the version.
    pub fn write(&self) -> RwLockWriteGuard<'_, Storage> {
        self.version.fetch_add(1, Ordering::Relaxed);
        self.data.write().unwrap_or_else(|e| e.into_inner())
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
}

impl Drop for StorageCell {
    fn drop(&mut self) {
        let data = self.data.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Storage::Cpu(CpuStorage::F32(v)) = data {
            crate::cpu::recycle(std::mem::take(v));
        }
    }
}
