//! The kernel layer: one Rust entry point per WGSL kernel, operating on raw `wgpu::Buffer`s plus explicit
//! sizes. Nothing here knows about [`crate::GpuTensor`]; the tensor wrapper (and later the folo-core
//! `Backend` implementation) validate shapes, allocate outputs and call into this layer.
//!
//! Buffer layout convention: every operand is a dense row-major array starting at byte 0 of its buffer
//! (NCHW for images, `[out, in/groups, kh, kw]` for conv weights). Buffers may be larger than the operand
//! (pool size classes); kernels never touch the slack.

pub(crate) mod bounds;
pub mod conv;
pub mod coop;
pub mod detect;
pub mod detect_sites;
pub mod elementwise;
pub mod ew;
pub mod gemm;
pub mod gemm_cat;
pub mod layout;
pub mod muon;
pub mod norm;
pub mod optim;
pub mod pool_grad;
pub mod pooling;
pub mod reduce;
pub mod strided;
pub mod vision;

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::DType;
use crate::error::{GpuError, Result};
use crate::pipeline::{KernelDef, Spec};

/// The kernel named `name` as compiled with `spec`, for [`crate::precompile`], which records kernels by name and
/// compiles them again in a later process: every fixed definition (each module's `DEFS`), and the multi-tensor
/// optimizer and Muon kernels built per layout. `None` for a name this build does not have (a list from another
/// build). What a list may ask a found kernel for is [`bounds::precompilable`].
pub(crate) fn find_def(name: &str, spec: &Spec) -> Option<&'static KernelDef> {
    static TABLE: OnceLock<HashMap<&'static str, &'static KernelDef>> = OnceLock::new();
    let table = TABLE.get_or_init(|| all_defs().map(|d| (d.name, d)).collect());
    table
        .get(name)
        .copied()
        .or_else(|| optim::many_def_for(name, spec))
        .or_else(|| muon::many_def_for(name, spec))
}

/// Every fixed kernel definition.
pub(crate) fn all_defs() -> impl Iterator<Item = &'static KernelDef> {
    [
        conv::DEFS,
        coop::DEFS,
        detect::DEFS,
        detect_sites::DEFS,
        elementwise::DEFS,
        ew::DEFS,
        gemm::DEFS,
        layout::DEFS,
        muon::DEFS,
        norm::DEFS,
        optim::DEFS,
        pool_grad::DEFS,
        pooling::DEFS,
        reduce::DEFS,
        strided::DEFS,
        vision::DEFS,
    ]
    .into_iter()
    .flatten()
    .copied()
}

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

/// [`vec_prelude`] for elements stored as f16 (`E` is `f16` or `vec4<f16>`; `V4` widens to f32, `OUT` rounds).
pub(crate) fn vec_prelude_f16(vec: bool) -> &'static str {
    if vec {
        "alias E = vec4<f16>;\nconst VEC: bool = true;\n\
         fn V4(x: E) -> vec4<f32> { return vec4<f32>(x); }\nfn OUT(v: vec4<f32>) -> E { return vec4<f16>(v); }\n"
    } else {
        "alias E = f16;\nconst VEC: bool = false;\n\
         fn V4(x: E) -> vec4<f32> { return vec4<f32>(f32(x)); }\nfn OUT(v: vec4<f32>) -> E { return f16(v.x); }\n"
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
