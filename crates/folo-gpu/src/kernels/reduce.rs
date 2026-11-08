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

// ----- folo-core backend: arg-reductions, softmax over any dim, long-row splitting, top-k -----------------------

static ARG: KernelDef = KernelDef {
    name: "arg_reduce",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/arg_reduce.wgsl")),
};
static SOFTMAX_COLS: KernelDef = KernelDef {
    name: "softmax_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/softmax_cols.wgsl")),
};
static TOPK: KernelDef = KernelDef {
    name: "topk",
    storage: &[Access::Read, Access::ReadWrite, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/topk.wgsl")),
};

/// A divisor `s` of `r` with `r / s >= min_len`, the largest one up to `want` (1 if none).
fn split_factor(r: usize, want: usize, min_len: usize) -> usize {
    let mut best = 1;
    let mut s = 2;
    while s <= want && r / s >= min_len {
        if r.is_multiple_of(s) {
            best = s;
        }
        s += 1;
    }
    best
}

/// [`reduce`] (sum / max / min) that splits a long reduction over few outputs into two passes so the GPU
/// fills up: `[outer, r, inner]` is reduced as `[outer · s, r / s, inner]`, then `[outer, s, inner]`.
pub fn reduce_split(
    ctx: &GpuContext,
    op: ReduceOp,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    // Workgroups the single-pass kernels would launch.
    let groups = if inner == 1 { outer } else { outer.saturating_mul(inner.div_ceil(64)) };
    let s =
        if groups < 256 && r >= 8192 { split_factor(r, 512usize.div_ceil(groups.max(1)), 1024) } else { 1 };
    if s <= 1 || op == ReduceOp::Mean {
        return reduce(ctx, op, x, y, outer, r, inner);
    }
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "reduce input")?;
    let tmp = ctx.alloc(DType::F32.bytes(prod(&[outer, s, inner])?))?;
    reduce(ctx, op, x, tmp.raw(), outer * s, r / s, inner)?;
    reduce(ctx, op, tmp.raw(), y, outer, s, inner)
}

/// Index of the first maximum (`max`) or minimum over the middle axis of dense f32 `[outer, r, inner]`, as
/// `i64` (two words per element) into `y` (`[outer, inner]`). NaN counts as the extreme (the first NaN wins).
pub fn arg_reduce(
    ctx: &GpuContext,
    max: bool,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "arg_reduce input")?;
    check_buf(y, prod(&[outer, inner, 2])?, DType::F32, "arg_reduce output")?;
    let n = outer * inner;
    if n == 0 {
        return Ok(());
    }
    if r == 0 {
        return Err(shape_err("arg_reduce over an empty dimension"));
    }
    let k = ctx.kernel(&ARG, Spec::new().bool("MAX", max))?;
    let p = Params::new().uz(n)?.uz(r)?.uz(inner)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Softmax (or log-softmax) over the middle axis of dense f32 `[outer, d, inner]`.
pub fn softmax_dim(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    log: bool,
) -> Result<()> {
    if inner == 1 {
        return softmax(ctx, x, y, outer, d, log);
    }
    check_buf(x, prod(&[outer, d, inner])?, DType::F32, "softmax input")?;
    check_buf(y, prod(&[outer, d, inner])?, DType::F32, "softmax output")?;
    let n = outer * inner;
    if n * d == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&SOFTMAX_COLS, Spec::new().bool("LOG", log))?;
    let p = Params::new().uz(n)?.uz(d)?.uz(inner)?.f(f32::NEG_INFINITY);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// The `k` largest (`largest`) or smallest entries of each of `rows` dense f32 rows of length `d`, sorted,
/// ties broken by lower index (total order: `-NaN < -inf < .. < -0 < +0 < .. < inf < NaN`, as
/// `f32::total_cmp`). Values go to `values` (`[rows, k]`), `i64` indices to `indices`. One thread per row and
/// `O(k · d)` work: sized for assigner-style problems (`k` up to a few dozen).
#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    values: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    rows: usize,
    d: usize,
    k: usize,
    largest: bool,
) -> Result<()> {
    if k > d {
        return Err(shape_err(format!("topk: k = {k} exceeds the row length {d}")));
    }
    check_buf(x, prod(&[rows, d])?, DType::F32, "topk input")?;
    check_buf(values, prod(&[rows, k])?, DType::F32, "topk values")?;
    check_buf(indices, prod(&[rows, k, 2])?, DType::F32, "topk indices")?;
    if rows * k == 0 {
        return Ok(());
    }
    let kern = ctx.kernel(&TOPK, Spec::new().bool("LARGEST", largest))?;
    let p = Params::new().uz(rows)?.uz(d)?.uz(k)?;
    ctx.dispatch(&kern, &[x, values, indices], &p, ctx.grid_1d(rows as u64, 64)?)
}
