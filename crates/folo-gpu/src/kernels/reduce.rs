//! Reductions along an arbitrary dim, and softmax / log-softmax along the last dim.
//!
//! A reduction over dim `d` of a dense tensor views it as `[outer, R, inner]` (`R` = size of `d`). When
//! `inner == 1` the reduced axis is contiguous and one workgroup reduces each row with a tree in workgroup
//! memory; otherwise each thread owns one output column and walks `R` with coalesced loads across columns.

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Reduction kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
    /// Arithmetic mean.
    Mean,
}

impl ReduceOp {
    fn code(self) -> u32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
        }
    }

    fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }
}

static ROWS: KernelDef = KernelDef {
    name: "reduce_rows",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/reduce_rows.wgsl")),
};
static COLS: KernelDef = KernelDef {
    name: "reduce_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/reduce_cols.wgsl")),
};
static SOFTMAX: KernelDef = KernelDef {
    name: "softmax",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/softmax.wgsl")),
};

/// Reduces `x` viewed as `[outer, r, inner]` over the middle axis into `y` (`[outer, inner]`), f32.
pub fn reduce(
    ctx: &GpuContext,
    op: ReduceOp,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "reduce input")?;
    check_buf(y, prod(&[outer, inner])?, DType::F32, "reduce output")?;
    if outer * inner == 0 {
        return Ok(());
    }
    let scale = if op == ReduceOp::Mean { 1.0 / r as f32 } else { 1.0 };
    let spec = Spec::new().u32("OP", op.code());
    if inner == 1 {
        let k = ctx.kernel(&ROWS, spec)?;
        let p = Params::new().uz(outer)?.uz(r)?.f(scale).f(op.identity());
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(outer as u64 * 256, 256)?)
    } else {
        let chunks = inner.div_ceil(64);
        let k = ctx.kernel(&COLS, spec)?;
        let p = Params::new().uz(outer)?.uz(r)?.f(scale).uz(inner)?.uz(chunks)?.f(op.identity());
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d((outer * chunks) as u64 * 256, 256)?)
    }
}

/// Splits `shape` around `dim` into `(outer, r, inner)`.
pub fn split_dim(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    if dim >= shape.len() {
        return Err(shape_err(format!("dim {dim} out of range for {shape:?}")));
    }
    Ok((shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product()))
}

/// Softmax (or log-softmax when `log`) over contiguous rows of length `n`, f32.
pub fn softmax(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    n: usize,
    log: bool,
) -> Result<()> {
    check_buf(x, prod(&[rows, n])?, DType::F32, "softmax input")?;
    check_buf(y, prod(&[rows, n])?, DType::F32, "softmax output")?;
    if rows * n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&SOFTMAX, Spec::new().bool("LOG", log))?;
    let p = Params::new().uz(rows)?.uz(n)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(rows as u64 * 256, 256)?)
}
