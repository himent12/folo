//! Element types.

use half::f16;

use crate::storage::CpuStorage;

/// The element type of a tensor.
///
/// `F32` is the compute type: every arithmetic op works on it. `F16` is for compact storage and GPU inference,
/// `U8` for images, `U32`/`I64` for indices. Convert with [`crate::Tensor::to_dtype`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE half float (storage only on the CPU).
    F16,
    /// Unsigned byte.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer (the index type returned by argmax / topk).
    I64,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 | DType::U32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
            DType::I64 => 8,
        }
    }

    /// True for floating point types.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16)
    }

    /// Short lowercase name (`"f32"`, `"u8"`, ...), the same spelling safetensors uses in lowercase.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that can be a tensor element: `f32`, `f16`, `u8`, `u32`, `i64`.
pub trait Element:
    sealed::Sealed + Copy + Default + Send + Sync + PartialOrd + std::fmt::Debug + bytemuck::Pod + 'static
{
    /// The matching [`DType`].
    const DTYPE: DType;
    /// Lossy conversion to `f64`.
    fn to_f64(self) -> f64;
    /// Lossy (saturating, rounding toward zero for integers) conversion from `f64`.
    fn from_f64(v: f64) -> Self;
    #[doc(hidden)]
    fn into_storage(v: Vec<Self>) -> CpuStorage;
    #[doc(hidden)]
    fn slice(s: &CpuStorage) -> Option<&[Self]>;
    #[doc(hidden)]
    fn slice_mut(s: &mut CpuStorage) -> Option<&mut [Self]>;
}

macro_rules! element {
    ($t:ty, $variant:ident, $to:expr, $from:expr) => {
        impl sealed::Sealed for $t {}
        impl Element for $t {
            const DTYPE: DType = DType::$variant;
            #[inline]
            fn to_f64(self) -> f64 {
                ($to)(self)
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                ($from)(v)
            }
            fn into_storage(v: Vec<Self>) -> CpuStorage {
                CpuStorage::$variant(v)
            }
            fn slice(s: &CpuStorage) -> Option<&[Self]> {
                match s {
                    CpuStorage::$variant(v) => Some(v),
                    _ => None,
                }
            }
            fn slice_mut(s: &mut CpuStorage) -> Option<&mut [Self]> {
                match s {
                    CpuStorage::$variant(v) => Some(v),
                    _ => None,
                }
            }
        }
    };
}

element!(f32, F32, |x: f32| x as f64, |v: f64| v as f32);
element!(f16, F16, |x: f16| x.to_f64(), f16::from_f64);
element!(u8, U8, |x: u8| x as f64, |v: f64| v as u8);
element!(u32, U32, |x: u32| x as f64, |v: f64| v as u32);
element!(i64, I64, |x: i64| x as f64, |v: f64| v as i64);
