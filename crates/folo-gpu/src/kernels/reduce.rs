//! Reductions along an arbitrary dim, and softmax / log-softmax along the last dim.
//!
//! A reduction over dim `d` of a dense tensor views it as `[outer, R, inner]` (`R` = size of `d`). When
//! `inner == 1` the reduced axis is contiguous and one workgroup reduces each row with a tree in workgroup
//! memory; otherwise each thread owns one output column and walks `R` with coalesced loads across columns.
//!
//! The `*_as` / dtype-taking entry points also read and write f16 storage (with `shader-f16`); the arithmetic
//! is always f32, so an f16 reduction or softmax accumulates in f32 and rounds once on the store.

use super::{check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// `alias XT` / `alias YT` (the storage types of `X` and `Y`) in front of `body`, from the spec's `XH` / `YH`.
fn typed(spec: &Spec, body: &str) -> String {
    let t = |h: bool| if h { "f16" } else { "f32" };
    format!("alias XT = {};\nalias YT = {};\n{body}", t(spec.get_bool("XH")), t(spec.get_bool("YH")))
}

/// The spec switches for an `x` of `xdt` and a `y` of `ydt`.
fn typed_spec(ctx: &GpuContext, xdt: DType, ydt: DType, op: &'static str) -> Result<Spec> {
    check_dtype(ctx, xdt, op)?;
    check_dtype(ctx, ydt, op)?;
    let (xh, yh) = (xdt == DType::F16, ydt == DType::F16);
    Ok(Spec::new().bool("XH", xh).bool("YH", yh).f16(xh || yh))
}

const REDUCE_ROWS_MAIN_SUBGROUPS: &str = r#"
var<workgroup> partial: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let r = pu(1u);
    let acc = reduce_rows_thread(row, rows, r, lid);
    var r_sg: f32;
    switch OP {
        case 1u: {
            let nan_v = isnan1(acc);
            r_sg = select(subgroupMax(acc), subgroupAdd(select(0.0, acc, nan_v)), subgroupAny(nan_v));
        }
        case 2u: {
            let nan_v = isnan1(acc);
            r_sg = select(subgroupMin(acc), subgroupAdd(select(0.0, acc, nan_v)), subgroupAny(nan_v));
        }
        default: { r_sg = subgroupAdd(acc); }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) { partial[sg] = r_sg; }
    workgroupBarrier();
    if (lid == 0u && row < rows) {
        let total = combine(
            combine(combine(partial[0], partial[1]), combine(partial[2], partial[3])),
            combine(combine(partial[4], partial[5]), combine(partial[6], partial[7])),
        );
        Y[row] = YT(total * pf(2u));
    }
}
"#;

const REDUCE_ROWS_MAIN_PORTABLE: &str = r#"
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let r = pu(1u);
    partial[lid] = reduce_rows_thread(row, rows, r, lid);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            partial[lid] = combine(partial[lid], partial[lid + s]);
        }
        workgroupBarrier();
    }
    if (lid == 0u && row < rows) {
        Y[row] = YT(partial[0] * pf(2u));
    }
}
"#;

const SOFTMAX_MAIN_SUBGROUPS: &str = r#"
var<workgroup> partial: array<vec2<f32>, 8>;

fn reduce_online(lid: u32, sgid: u32, lane: u32, v: vec2<f32>) -> vec2<f32> {
    var st = v;
    for (var off = 16u; off > 0u; off >>= 1u) {
        let om = subgroupShuffleDown(st.x, off);
        let od = subgroupShuffleDown(st.y, off);
        if (lane < off) { st = online_merge(st, vec2<f32>(om, od)); }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) { partial[sg] = st; }
    workgroupBarrier();
    // Butterfly-reduce the 8 subgroup partials across every subgroup so all 256 threads receive the final
    // `(m, d)` with a single workgroupBarrier().
    var r = partial[lane & 7u];
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 4u), subgroupShuffleXor(r.y, 4u)));
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 2u), subgroupShuffleXor(r.y, 2u)));
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 1u), subgroupShuffleXor(r.y, 1u)));
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    softmax_run(wid, nwg, lid, sgid, lane);
}
"#;

const SOFTMAX_MAIN_PORTABLE: &str = r#"
var<workgroup> partial: array<vec2<f32>, 256>;

fn reduce_online(lid: u32, sgid: u32, lane: u32, v: vec2<f32>) -> vec2<f32> {
    partial[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { partial[lid] = online_merge(partial[lid], partial[lid + s]); }
        workgroupBarrier();
    }
    return partial[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    softmax_run(wid, nwg, lid, 0u, 0u);
}
"#;

fn gen_rows(s: &Spec) -> String {
    let main = if s.is_subgroups() { REDUCE_ROWS_MAIN_SUBGROUPS } else { REDUCE_ROWS_MAIN_PORTABLE };
    typed(s, &format!("{}{main}", include_str!("../shaders/reduce_rows.wgsl")))
}
fn gen_cols(s: &Spec) -> String {
    typed(s, include_str!("../shaders/reduce_cols.wgsl"))
}
/// Maximum row length cached in per-thread registers (4 elements × 256 threads) for single-pass [`SOFTMAX`].
const SOFTMAX_ROW_CACHE_MAX: usize = 1024;
/// Maximum column length cached in a per-thread local array for single-pass [`SOFTMAX_COLS`] (covers DFL `d = 16`).
const SOFTMAX_COL_SMALL_MAX: usize = 32;

fn gen_softmax(s: &Spec) -> String {
    let cache = s.get_bool("CACHE");
    let main = if s.is_subgroups() { SOFTMAX_MAIN_SUBGROUPS } else { SOFTMAX_MAIN_PORTABLE };
    typed(
        s,
        &format!("const CACHE_ROW: bool = {cache};\n{main}\n{}", include_str!("../shaders/softmax.wgsl")),
    )
}
fn gen_softmax_cols(s: &Spec) -> String {
    let small = s.get_bool("SMALL");
    typed(s, &format!("const SMALL_COL: bool = {small};\n{}", include_str!("../shaders/softmax_cols.wgsl")))
}

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
    source: Source::Generated(gen_rows),
};
static COLS: KernelDef = KernelDef {
    name: "reduce_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_cols),
};
static SOFTMAX: KernelDef = KernelDef {
    name: "softmax",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_softmax),
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
    reduce_as(ctx, op, (x, DType::F32), (y, DType::F32), outer, r, inner)
}

/// [`reduce`] with `x` and `y` stored as the given dtypes (f32 or f16; f32 arithmetic either way).
pub fn reduce_as(
    ctx: &GpuContext,
    op: ReduceOp,
    (x, xdt): (&wgpu::Buffer, DType),
    (y, ydt): (&wgpu::Buffer, DType),
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, xdt, "reduce input")?;
    check_buf(y, prod(&[outer, inner])?, ydt, "reduce output")?;
    if outer * inner == 0 {
        return Ok(());
    }
    let scale = if op == ReduceOp::Mean { 1.0 / r as f32 } else { 1.0 };
    let spec = typed_spec(ctx, xdt, ydt, "reduce")?.u32("OP", op.code());
    if inner == 1 {
        let k = ctx.kernel(&ROWS, spec.subgroups(use_subgroups(ctx)))?;
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
    softmax_rows(ctx, x, y, rows, n, log, DType::F32)
}

/// [`softmax`] for `x` and `y` of `dtype` (f32 or f16; the max, the sum and the exponentials are f32).
pub fn softmax_rows(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    n: usize,
    log: bool,
    dtype: DType,
) -> Result<()> {
    check_buf(x, prod(&[rows, n])?, dtype, "softmax input")?;
    check_buf(y, prod(&[rows, n])?, dtype, "softmax output")?;
    if rows * n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(
        &SOFTMAX,
        typed_spec(ctx, dtype, dtype, "softmax")?
            .bool("LOG", log)
            .bool("CACHE", n <= SOFTMAX_ROW_CACHE_MAX)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new().uz(rows)?.uz(n)?.f(f32::NEG_INFINITY);
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
    source: Source::Generated(gen_softmax_cols),
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
/// fills up: `[outer, r, inner]` is reduced as `[outer · s, r / s, inner]`, then `[outer, s, inner]`. `x` and
/// `y` hold `dtype` (f32 or f16); the partial results between the passes are f32.
#[allow(clippy::too_many_arguments)]
pub fn reduce_split(
    ctx: &GpuContext,
    op: ReduceOp,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
    dtype: DType,
) -> Result<()> {
    // Workgroups the single-pass kernels would launch.
    let groups = if inner == 1 { outer } else { outer.saturating_mul(inner.div_ceil(64)) };
    let s =
        if groups < 256 && r >= 8192 { split_factor(r, 512usize.div_ceil(groups.max(1)), 1024) } else { 1 };
    if s <= 1 || op == ReduceOp::Mean {
        return reduce_as(ctx, op, (x, dtype), (y, dtype), outer, r, inner);
    }
    check_buf(x, prod(&[outer, r, inner])?, dtype, "reduce input")?;
    let tmp = ctx.alloc(DType::F32.bytes(prod(&[outer, s, inner])?))?;
    reduce_as(ctx, op, (x, dtype), (tmp.raw(), DType::F32), outer * s, r / s, inner)?;
    reduce_as(ctx, op, (tmp.raw(), DType::F32), (y, dtype), outer, s, inner)
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

/// Softmax (or log-softmax) over the middle axis of dense `[outer, d, inner]` of `dtype` (f32 or f16, f32
/// arithmetic).
pub fn softmax_dim(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    log: bool,
    dtype: DType,
) -> Result<()> {
    if inner == 1 {
        return softmax_rows(ctx, x, y, outer, d, log, dtype);
    }
    check_buf(x, prod(&[outer, d, inner])?, dtype, "softmax input")?;
    check_buf(y, prod(&[outer, d, inner])?, dtype, "softmax output")?;
    let n = outer * inner;
    if n * d == 0 {
        return Ok(());
    }
    let k = ctx.kernel(
        &SOFTMAX_COLS,
        typed_spec(ctx, dtype, dtype, "softmax")?.bool("LOG", log).bool("SMALL", d <= SOFTMAX_COL_SMALL_MAX),
    )?;
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

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&ROWS, &COLS, &SOFTMAX, &ARG, &SOFTMAX_COLS, &TOPK];
