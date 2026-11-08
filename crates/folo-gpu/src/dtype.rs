//! Element types of GPU buffers.

/// Element type of a [`crate::GpuTensor`] or a kernel operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float: the compute type, supported by every kernel.
    F32,
    /// 16-bit float storage (IEEE binary16). GEMM and conv have f16 variants (f32 accumulation) when the
    /// adapter has `shader-f16`; everything else works on f32.
    F16,
}

impl DType {
    /// Bytes per element.
    pub const fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }

    /// Bytes for `n` elements, rounded up to wgpu's 4-byte copy alignment (saturating, so an absurd count
    /// fails the allocation limit instead of wrapping).
    pub const fn bytes(self, n: usize) -> u64 {
        (n as u64).saturating_mul(self.size() as u64).saturating_add(3) & !3
    }
}
