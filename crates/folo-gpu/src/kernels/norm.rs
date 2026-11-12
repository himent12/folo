//! Train-mode BatchNorm over the channels of NCHW data, with a fused activation, forward and backward. The data
//! (`x`, `y`, the gradients `g` and `dx`) is f32 or f16 ([`BnDesc::dtype`], mixed-precision training); the
//! parameters, the statistics and every sum are f32.
//!
//! Forward, three dispatches:
//! 1. `bn_stats`: each workgroup takes one channel and one chunk of its `N·H·W` values and writes a Welford
//!    partial (count, mean, M2), merged across the workgroup with Chan et al.'s pairwise formula. Chunks are
//!    sized so the GPU fills up even when there are few channels and long planes.
//! 2. `bn_finalize`: one workgroup per channel merges its partials into the mean and biased variance,
//!    writes `mean` and `invstd = 1 / sqrt(var + eps)`, and optionally updates the running statistics in
//!    place.
//! 3. `bn_apply`: `y = act(((x − mean) · invstd) · w + b)`, `vec4` when `H·W` is a multiple of 4. Centring
//!    first matters: the expanded `x · scale + shift` loses everything to cancellation when the mean is large
//!    against the spread.
//!
//! Backward (the analytic BatchNorm gradient through the activation), three dispatches:
//! 1. `bn_bwd_reduce`: per (channel, chunk) partial sums of `dz` and `dz · x̂`, where
//!    `x̂ = (x − mean) · invstd`, `z = x̂ · w + b` is recomputed and `dz = g · act'(z)`;
//! 2. `bn_bwd_finalize`: `db = Σdz` and `dw = Σdz·x̂`;
//! 3. `bn_bwd_dx`: `dx = w · invstd · (dz − db / n − x̂ · dw / n)` per element, with `x̂` and `z` recomputed by
//!    the same expressions as the forward and the reduction pass.
//!
//! So a forward reads `x` twice and writes `y` once, and a backward reads `x` and `g` twice and writes `dx`
//! once, against about ten passes forward and twenty backward for BatchNorm + SiLU as separate ops.

use super::{check_buf, prod, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

const COMMON: &str = include_str!("../shaders/bn_common.wgsl");
const WELFORD: &str = include_str!("../shaders/bn_welford.wgsl");

/// The activation fused after the normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnAct {
    /// None.
    Identity,
    /// `z · sigmoid(z)`.
    Silu,
    /// `max(z, 0)`.
    Relu,
}

impl BnAct {
    fn code(self) -> u32 {
        match self {
            BnAct::Identity => 0,
            BnAct::Silu => 1,
            BnAct::Relu => 2,
        }
    }

    /// CPU reference of the activation.
    pub fn apply(self, z: f32) -> f32 {
        match self {
            BnAct::Identity => z,
            BnAct::Silu => z / (1.0 + (-z).exp()),
            BnAct::Relu => z.max(0.0),
        }
    }
}

/// A BatchNorm problem: `[n, c, hw]` values, normalised per channel over `n · hw`.
#[derive(Clone, Copy, Debug)]
pub struct BnDesc {
    /// Batch size.
    pub n: usize,
    /// Channels.
    pub c: usize,
    /// Values per plane (`H·W`).
    pub hw: usize,
    /// Activation after the affine transform.
    pub act: BnAct,
    /// Storage type of the data (`x`, `y`, `g`, `dx`): f32 or f16. Weight, bias, statistics and running statistics
    /// are always f32.
    pub dtype: DType,
}

/// The running statistics the forward updates in place.
pub struct BnRunning<'a> {
    /// Running mean, `[c]`.
    pub mean: &'a wgpu::Buffer,
    /// Running unbiased variance, `[c]`.
    pub var: &'a wgpu::Buffer,
    /// Weight of the batch statistic.
    pub momentum: f32,
}

fn gen_stats(_: &Spec) -> String {
    format!("{COMMON}{WELFORD}{}", include_str!("../shaders/bn_stats.wgsl"))
}

fn gen_finalize(_: &Spec) -> String {
    format!("{COMMON}{WELFORD}{}", include_str!("../shaders/bn_finalize.wgsl"))
}

/// The element prelude of the elementwise passes: f32 or f16 storage (`spec.is_f16()`), scalar or vec4.
fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

fn gen_apply(spec: &Spec) -> String {
    format!("{}{COMMON}{}", prelude(spec), include_str!("../shaders/bn_apply.wgsl"))
}

fn gen_bwd_reduce(_: &Spec) -> String {
    format!("{COMMON}{}", include_str!("../shaders/bn_bwd_reduce.wgsl"))
}

fn gen_bwd_finalize(_: &Spec) -> String {
    format!("{COMMON}{}", include_str!("../shaders/bn_bwd_finalize.wgsl"))
}

fn gen_bwd_dx(spec: &Spec) -> String {
    format!("{}{COMMON}{}", prelude(spec), include_str!("../shaders/bn_bwd_dx.wgsl"))
}

use Access::{Read as R, ReadWrite as RW};

static STATS: KernelDef =
    KernelDef { name: "bn_stats", storage: &[R, RW], source: Source::Generated(gen_stats) };
static FINALIZE: KernelDef = KernelDef {
    name: "bn_finalize",
    storage: &[R, R, R, RW, RW, RW, RW, RW],
    source: Source::Generated(gen_finalize),
};
static APPLY: KernelDef =
    KernelDef { name: "bn_apply", storage: &[R, R, RW], source: Source::Generated(gen_apply) };
static BWD_REDUCE: KernelDef = KernelDef {
    name: "bn_bwd_reduce",
    storage: &[R, R, R, R, R, R, RW],
    source: Source::Generated(gen_bwd_reduce),
};
static BWD_FINALIZE: KernelDef = KernelDef {
    name: "bn_bwd_finalize",
    storage: &[R, R, R, R, R, RW, RW, RW],
    source: Source::Generated(gen_bwd_finalize),
};
static BWD_DX: KernelDef =
    KernelDef { name: "bn_bwd_dx", storage: &[R, R, R, RW], source: Source::Generated(gen_bwd_dx) };

/// Chunking of each channel's `m = n · hw` values: `(chunks, chunk length)`. About 2048 workgroups in total
/// and at least 4 values per thread.
fn plan(m: usize, c: usize) -> (usize, usize) {
    let want = 2048usize.div_ceil(c.max(1));
    let chunks = want.min(m.div_ceil(256 * 4)).clamp(1, 65535);
    let len = m.div_ceil(chunks);
    (m.div_ceil(len), len)
}

impl BnDesc {
    /// Validates the sizes and the dtype and returns `(total elements, n · hw)`.
    fn check(&self, ctx: &GpuContext) -> Result<(usize, usize)> {
        super::gemm::check_dtype(ctx, self.dtype, "batch_norm")?;
        let total = prod(&[self.n, self.c, self.hw])?;
        let m = self.n * self.hw;
        if m == 0 || self.c == 0 {
            return Err(shape_err(format!("batch_norm: empty problem {self:?}")));
        }
        // Indices and the walk's `e + 256` stay below 2^32.
        if total >= (1usize << 31) || self.c > 65535 {
            return Err(GpuError::Limit(format!("batch_norm: {self:?} is too large for one dispatch")));
        }
        Ok((total, m))
    }

    fn walk_params(&self, m: usize, chunks: usize, len: usize) -> Result<Params> {
        Params::new()
            .uz(self.c)?
            .uz(self.hw)?
            .uz(m)?
            .uz(len)?
            .uz(chunks)?
            .uz(256 / self.hw)?
            .uz(256 % self.hw)
    }

    fn elementwise(&self, total: usize) -> (bool, usize) {
        let vec = self.hw.is_multiple_of(4);
        (vec, if vec { total / 4 } else { total })
    }
}

/// Train-mode forward: `y = act((x − mean) · invstd · w + b)`, writing `mean` and `invstd` (`[c]` each) and
/// updating `running` if given.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward(
    ctx: &GpuContext,
    d: &BnDesc,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    eps: f32,
    running: Option<BnRunning<'_>>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
) -> Result<()> {
    let (total, m) = d.check(ctx)?;
    check_buf(x, total, d.dtype, "batch_norm input")?;
    check_buf(y, total, d.dtype, "batch_norm output")?;
    for (buf, what) in [(w, "weight"), (b, "bias"), (mean, "mean"), (invstd, "invstd")] {
        check_buf(buf, d.c, DType::F32, what)?;
    }
    if let Some(r) = &running {
        check_buf(r.mean, d.c, DType::F32, "running_mean")?;
        check_buf(r.var, d.c, DType::F32, "running_var")?;
    }
    let (chunks, len) = plan(m, d.c);
    let part = ctx.alloc(DType::F32.bytes(prod(&[d.c, chunks, 3])?))?;
    let coef = ctx.alloc(DType::F32.bytes(4 * d.c))?;

    let half = d.dtype == DType::F16;
    let k = ctx.kernel(&STATS, Spec::new().u32("ACT", 0).f16(half))?;
    ctx.dispatch(&k, &[x, part.raw()], &d.walk_params(m, chunks, len)?, [chunks as u32, d.c as u32, 1])?;

    let k = ctx.kernel(&FINALIZE, Spec::new().u32("ACT", 0).bool("RUNNING", running.is_some()))?;
    let mom = running.as_ref().map_or(0.0, |r| r.momentum);
    let unbias = if m > 1 { m as f64 / (m as f64 - 1.0) } else { 1.0 };
    let p = Params::new().uz(d.c)?.uz(chunks)?.f(eps).f(1.0 - mom).f(mom).f(mom * unbias as f32);
    let (rm, rv) = running.as_ref().map_or((ctx.dummy(), ctx.dummy()), |r| (r.mean, r.var));
    ctx.dispatch(&k, &[part.raw(), w, b, mean, invstd, coef.raw(), rm, rv], &p, [d.c as u32, 1, 1])?;

    let (vec, units) = d.elementwise(total);
    let k = ctx.kernel(&APPLY, Spec::new().u32("ACT", d.act.code()).bool("VEC4", vec).f16(half))?;
    let p = Params::new().uz(units)?.uz(d.c)?.uz(d.hw)?;
    ctx.dispatch(&k, &[x, coef.raw(), y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Backward of [`batch_norm_forward`]: `dx` (like `x`), `dw` and `db` (`[c]`) from the output gradient `g`
/// and the forward's `mean` / `invstd`.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward(
    ctx: &GpuContext,
    d: &BnDesc,
    x: &wgpu::Buffer,
    g: &wgpu::Buffer,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dw: &wgpu::Buffer,
    db: &wgpu::Buffer,
) -> Result<()> {
    let (total, m) = d.check(ctx)?;
    for (buf, what) in [(x, "input"), (g, "output gradient"), (dx, "input gradient")] {
        check_buf(buf, total, d.dtype, what)?;
    }
    for (buf, what) in
        [(w, "weight"), (b, "bias"), (mean, "mean"), (invstd, "invstd"), (dw, "dw"), (db, "db")]
    {
        check_buf(buf, d.c, DType::F32, what)?;
    }
    let (chunks, len) = plan(m, d.c);
    let part = ctx.alloc(DType::F32.bytes(prod(&[d.c, chunks, 2])?))?;
    let coef = ctx.alloc(DType::F32.bytes(6 * d.c))?;
    let act = d.act.code();
    let half = d.dtype == DType::F16;

    let k = ctx.kernel(&BWD_REDUCE, Spec::new().u32("ACT", act).f16(half))?;
    let p = d.walk_params(m, chunks, len)?;
    ctx.dispatch(&k, &[x, g, mean, invstd, w, b, part.raw()], &p, [chunks as u32, d.c as u32, 1])?;

    let k = ctx.kernel(&BWD_FINALIZE, Spec::new().u32("ACT", 0))?;
    let p = Params::new().uz(d.c)?.uz(chunks)?.f((1.0 / m as f64) as f32);
    ctx.dispatch(&k, &[part.raw(), mean, invstd, w, b, dw, db, coef.raw()], &p, [d.c as u32, 1, 1])?;

    let (vec, units) = d.elementwise(total);
    let k = ctx.kernel(&BWD_DX, Spec::new().u32("ACT", act).bool("VEC4", vec).f16(half))?;
    let p = Params::new().uz(units)?.uz(d.c)?.uz(d.hw)?;
    ctx.dispatch(&k, &[x, g, coef.raw(), dx], &p, ctx.grid_1d(units as u64, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&STATS, &FINALIZE, &APPLY, &BWD_REDUCE, &BWD_FINALIZE, &BWD_DX];
