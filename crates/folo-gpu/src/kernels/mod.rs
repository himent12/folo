//! The kernel layer: one Rust entry point per WGSL kernel, operating on raw `wgpu::Buffer`s plus explicit
//! sizes. Nothing here knows about [`crate::GpuTensor`]; the tensor wrapper (and later the folo-core
//! `Backend` implementation) validate shapes, allocate outputs and call into this layer.
//!
//! Buffer layout convention: every operand is a dense row-major array starting at byte 0 of its buffer
//! (NCHW for images, `[out, in/groups, kh, kw]` for conv weights). Buffers may be larger than the operand
//! (pool size classes); kernels never touch the slack.

pub mod conv;
pub mod elementwise;
pub mod ew;
pub mod gemm;
pub mod layout;
pub mod norm;
pub mod optim;
pub mod pool_grad;
pub mod pooling;
pub mod reduce;
pub mod strided;

use crate::DType;
use crate::error::{GpuError, Result};

/// `E` / `V4` / `OUT` aliases: element type of a storage array and conversions to and from `vec4<f32>`.
pub(crate) fn vec_prelude(vec: bool) -> &'static str {
    if vec {
        "alias E = vec4<f32>;\nconst VEC: bool = true;\n\
         fn V4(x: E) -> vec4<f32> { return x; }\nfn OUT(v: vec4<f32>) -> E { return v; }\n"
    } else {
        "alias E = f32;\nconst VEC: bool = false;\n\
         fn V4(x: E) -> vec4<f32> { return vec4<f32>(x); }\nfn OUT(v: vec4<f32>) -> E { return v.x; }\n"
    }
}

/// WGSL for the unary function table shared by unary, bias+activation, GEMM and conv epilogues.
pub(crate) const UNARY_FN: &str = include_str!("../shaders/unary_fn.wgsl");

/// Checked product of `dims`; fails with [`GpuError::Limit`] instead of wrapping.
pub(crate) fn prod(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| overflow(dims))
}

/// Validates a tensor shape: the product of its dims (zeros counted as 1) must fit in `usize`, so every
/// sub-product of the shape is safe to compute unchecked afterwards. Returns the element count.
pub(crate) fn shape_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d.max(1))).ok_or_else(|| overflow(shape))?;
    Ok(shape.iter().product())
}

fn overflow(dims: &[usize]) -> GpuError {
    GpuError::Limit(format!("size {dims:?} overflows"))
}

/// Fails unless `buf` holds at least `elems` elements of `dtype`.
pub(crate) fn check_buf(buf: &wgpu::Buffer, elems: usize, dtype: DType, what: &str) -> Result<()> {
    let need = elems.checked_mul(dtype.size()).ok_or_else(|| overflow(&[elems, dtype.size()]))? as u64;
    if buf.size() < need {
        return Err(GpuError::Shape(format!(
            "{what}: buffer of {} bytes is smaller than {need} bytes",
            buf.size()
        )));
    }
    Ok(())
}
