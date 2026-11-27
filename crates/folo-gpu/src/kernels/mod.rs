//! The kernel layer: one Rust entry point per WGSL kernel, operating on raw `wgpu::Buffer`s plus explicit
//! sizes. Nothing here knows about [`crate::GpuTensor`]; the tensor wrapper and the folo-core `Backend`
//! implementation ([`crate::backend`]) validate shapes, allocate outputs and call into this layer.
//!
//! Buffer layout convention: every operand is a dense row-major array starting at byte 0 of its buffer
//! (NCHW for images, `[out, in/groups, kh, kw]` for conv weights). Buffers may be larger than the operand
//! (pool size classes); kernels never touch the slack.
//!
//! Each module declares its kernels as `KernelDef`s (the fixed ones collected in its `DEFS` and in `all_defs`)
//! and gets their WGSL one of two ways: `include_str!` of a file in `src/shaders/`, often prefixed by shared
//! snippets (`unary_fn.wgsl`, `bn_common.wgsl`, `bn_welford.wgsl`, `detect_common.wgsl`, the `E` / `V4` / `OUT`
//! aliases of `vec_prelude`), or a generator in Rust that writes the source from the spec with no `.wgsl` file
//! (the GEMM and implicit-GEMM convs, tensor cores, `ew.rs`, `strided/`, `pool_grad/`, Muon, the
//! multi-tensor optimizer kernels). At compile time `pipeline::render` puts the specialisation constants and the
//! shared prelude in front of either.
//!
//! Modules used by the folo-core backend:
//! - `ew.rs`: strided elementwise ops (unary and their gradients, binary, comparisons, `where`).
//! - `strided/`: strided gathers and scatters, fills, casts, tiled transpose and index ops for every dtype.
//! - `reduce/`: reductions, arg-reductions, softmax and top-k (also the tensor wrapper's reduce and softmax).
//! - `gemm/`: the tiled GEMM descriptors/config (`config.rs`), WGSL generator (`shader.rs`) and dispatch (`mod.rs`);
//!   `gemm_cat.rs`: GEMMs whose `B` is a concatenation read in place.
//! - `conv/`: conv2d forward, gradients, split-K, depthwise and per-site convs.
//! - `coop/`: tensor-core (cooperative-matrix) f16 GEMM and conv.
//! - `pool_grad/`: pooling with folo-core's semantics and the pooling and upsampling gradients.
//! - `layout.rs`: permute, concat, nearest upsampling and f32 / f16 casts (upsampling for the backend too).
//! - `norm/`: train-mode BatchNorm + activation; `optim.rs`: fused optimizer updates, clipping and
//!   unscaling; `muon.rs`: MuSGD's Muon step.
//! - `vision.rs`: letterboxing and detection-head candidates; `detect.rs`: the assigner and detection loss;
//!   `detect_sites.rs`: candidate-first detection.
//!
//! Modules only the tensor wrapper (and the benches) call: `elementwise.rs` (unary, broadcasting binary and bias +
//! activation; its [`elementwise::UnaryOp`] is also the fused activation of the GEMM and conv epilogues) and
//! `pooling.rs` (max pooling; its [`pooling::Window2d`] also describes conv windows). `bounds.rs` limits what a
//! kernel list may ask a generator for.

pub(crate) mod bounds;
pub mod conv;
pub mod coop;
pub mod detect;
pub mod detect_sites;
pub mod distill;
pub mod elementwise;
pub mod ew;
pub mod gemm;
pub mod gemm_cat;
pub mod layer_norm;
pub mod layout;
pub mod muon;
pub mod norm;
pub mod optim;
pub mod pool_grad;
pub mod pooling;
pub mod reduce;
pub mod softmax_bwd;
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
        distill::DEFS,
        elementwise::DEFS,
        ew::DEFS,
        gemm::DEFS,
        layer_norm::DEFS,
        layout::DEFS,
        muon::DEFS,
        norm::DEFS,
        optim::DEFS,
        pool_grad::DEFS,
        pooling::DEFS,
        reduce::DEFS,
        softmax_bwd::DEFS,
        strided::DEFS,
        vision::DEFS,
    ]
    .into_iter()
    .flatten()
    .copied()
}

// Tests set this thread-locally to exercise the portable shared-memory fallback even on hardware with 32-wide
// subgroups, without racing concurrent test threads or leaking across panics.
#[cfg(test)]
thread_local! {
    pub(crate) static FORCE_NO_SUBGROUPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether `ctx` should dispatch 32-wide subgroup reduction variants (always `false` on wasm32/browser WebGPU,
/// and overridable via [`FORCE_NO_SUBGROUPS`] in tests).
pub(crate) fn use_subgroups(ctx: &crate::context::GpuContext) -> bool {
    // Keep browser WebGPU explicitly on the portable path until wgpu exposes both `SUBGROUP` and WGSL `subgroup_id`.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ctx;
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(test)]
        if FORCE_NO_SUBGROUPS.with(std::cell::Cell::get) {
            return false;
        }
        let caps = ctx.caps();
        caps.subgroups && caps.subgroup_size == (32, 32)
    }
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
