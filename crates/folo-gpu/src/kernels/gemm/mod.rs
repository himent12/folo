//! Tiled GEMM: `C = alpha · op(A) · op(B) (+ bias) → act`, batched, with transposed variants and f16 storage.
//!
//! One WGSL generator ([`shader`]) covers every variant. A workgroup computes a `BM × BN` tile of `C`; each
//! thread owns a `TM × TN` block held in registers as `vec4<f32>` accumulators. `K` is walked in steps of `BK`:
//! the next `A` and `B` tiles are fetched from global memory into registers while the current tiles in workgroup
//! memory are consumed (double-buffered workgroup memory, one barrier per step). Tiles are stored `k`-major
//! (`As[k][m]`, `Bs[k][n]`) so the inner loop reads `vec4`s of both operands. Global loads are `vec4` when the
//! leading dimension and offsets allow it; bounds checks are compiled out when `M`, `N`, `K` are tile
//! multiples. The same generator also does implicit-GEMM convolution (see [`super::conv`]): `CONV_MODE` 1
//! gathers `B` straight from an NCHW image (forward), 2 gathers it from the output gradient with the
//! transposed-conv rule (input gradient), and 3 gathers an im2col-transposed image tile while reading `A` from
//! the output gradient across the batch (weight gradient). `SPLIT` splits `K` into chunks, one per extra
//! workgroup layer, each writing its own partial result (split-K; the caller sums the partials). `SITES` (with
//! `CONV_MODE` 1) computes the forward conv at listed output pixels only (see [`super::conv::conv2d_sites`]).
//!
//! Layout:
//! - [`config`]: `GemmDesc`, `GemmBias`, `BatchStrides`, `TileConfig` and tile selection.
//! - [`shader`]: WGSL source generator for plain GEMM, concatenated `B`, and implicit-GEMM convolutions.

mod config;
mod shader;

pub(crate) use config::check_config;
pub use config::{
    BatchStrides, GemmBias, GemmDesc, ResidualDesc, ResidualMode, TileConfig, pick_config, pick_config_conv,
};
pub(crate) use shader::gen_gemm;

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::coop;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

pub(crate) static GEMM: KernelDef = KernelDef {
    name: "gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_gemm),
};

/// [`GEMM`] with `B` in up to [`super::gemm_cat::MAX_CAT_PARTS`] buffers (`CAT`; unused parts bind the dummy).
pub(crate) static GEMM_CAT: KernelDef = KernelDef {
    name: "gemm_cat",
    storage: &[
        Access::Read,
        Access::Read,
        Access::Read,
        Access::ReadWrite,
        Access::Read,
        Access::Read,
        Access::Read,
    ],
    source: Source::Generated(gen_gemm),
};

/// [`GEMM`] as the forward conv at listed output pixels (`CONV_MODE` 1 with `SITES`; binding 5 is the list).
pub(crate) static GEMM_SITES: KernelDef = KernelDef {
    name: "gemm_sites",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Generated(gen_gemm),
};

/// [`GEMM`] with a fused residual operand at binding 5 (`RES_MODE` 1 = pre-activation, 2 = post-activation).
pub(crate) static GEMM_EPILOGUE: KernelDef = KernelDef {
    name: "gemm_epilogue",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Generated(gen_gemm),
};

/// Highest element index (exclusive) a strided batched matrix operand touches (0 when it touches none).
pub(crate) fn extent(
    s: &BatchStrides,
    batch: (usize, usize),
    rows: usize,
    cols: usize,
    ld: usize,
) -> Result<usize> {
    if rows == 0 || cols == 0 || batch.0 == 0 || batch.1 == 0 {
        return Ok(0);
    }
    (|| {
        s.offset
            .checked_add((batch.0 - 1).checked_mul(s.outer)?)?
            .checked_add((batch.1 - 1).checked_mul(s.inner)?)?
            .checked_add((rows - 1).checked_mul(ld)?)?
            .checked_add(cols)
    })()
    .ok_or_else(|| GpuError::Limit(format!("gemm operand extent overflows ({s:?}, {rows}x{cols}, ld {ld})")))
}

pub(crate) fn base_spec(d: &GemmDesc, cfg: TileConfig) -> Spec {
    Spec::new()
        .u32("BM", cfg.bm)
        .u32("BN", cfg.bn)
        .u32("BK", cfg.bk)
        .u32("TM", cfg.tm)
        .u32("TN", cfg.tn)
        .u32("ACT", d.act.code())
        .u32(
            "BIAS_KIND",
            match d.bias {
                GemmBias::None => 0,
                GemmBias::Row => 1,
                GemmBias::Col => 2,
            },
        )
        .f16(d.dtype == DType::F16)
}

pub(crate) fn gemm_params(d: &GemmDesc) -> Result<Params> {
    let p = Params::new()
        .uz(d.m)?
        .uz(d.n)?
        .uz(d.k)?
        .uz(d.lda)?
        .uz(d.ldb)?
        .uz(d.ldc)?
        .uz(d.batch.1)?
        .f(d.alpha)
        .uz(d.a.offset)?
        .uz(d.a.outer)?
        .uz(d.a.inner)?
        .uz(d.b.offset)?
        .uz(d.b.outer)?
        .uz(d.b.inner)?
        .uz(d.c.offset)?
        .uz(d.c.outer)?
        .uz(d.c.inner)?
        .f(d.act.alpha())
        .uz(d.bias_strides.offset)?
        .uz(d.bias_strides.outer)?
        .uz(d.bias_strides.inner)?;
    Ok(p)
}

pub(crate) fn check_dtype(ctx: &GpuContext, dtype: DType, op: &'static str) -> Result<()> {
    if dtype == DType::F16 && !ctx.caps().shader_f16 {
        return Err(GpuError::Unsupported("shader-f16"));
    }
    let _ = op;
    Ok(())
}

/// Encodes a GEMM described by `d`. `bias` is required when `d.bias != GemmBias::None`.
pub fn gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
) -> Result<()> {
    gemm_impl(ctx, d, a, b, bias, c, None)
}

/// [`gemm`] with an explicit tile configuration (benchmarks and tuning).
pub fn gemm_with_config(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: TileConfig,
) -> Result<()> {
    gemm_impl(ctx, d, a, b, bias, c, Some(cfg))
}

/// [`gemm`] (`cfg == None`: tensor cores for f16 when the context has them on, else a picked tile config) or
/// [`gemm_with_config`].
fn gemm_impl(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "gemm")?;
    if let Some(cfg) = cfg {
        check_config(ctx, cfg)?;
    }
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    if d.lda < ac || d.ldb < bc || d.ldc < d.n {
        return Err(shape_err(format!("gemm: leading dimensions too small in {d:?}")));
    }
    if d.batch.1 == 0 {
        return Err(shape_err("gemm: inner batch count must be non-zero"));
    }
    if d.m == 0 || d.n == 0 || d.batch.0 == 0 {
        return Ok(());
    }
    check_buf(a, extent(&d.a, d.batch, ar, ac, d.lda)?, d.dtype, "gemm A")?;
    check_buf(b, extent(&d.b, d.batch, br, bc, d.ldb)?, d.dtype, "gemm B")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, d.dtype, "gemm C")?;
    let bias_len = match d.bias {
        GemmBias::None => 0,
        GemmBias::Row => d.m,
        GemmBias::Col => d.n,
    };
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (_, Some(bb)) => {
            check_buf(bb, extent(&d.bias_strides, d.batch, 1, bias_len, bias_len)?, d.dtype, "gemm bias")?;
            bb
        }
        (_, None) => return Err(shape_err("gemm: bias kind set but no bias buffer")),
    };
    if cfg.is_none()
        && d.k > 0
        && d.dtype == DType::F16
        && ctx.tensor_cores()
        && coop::encode_gemm(ctx, d, DType::F16, a, b, bias_buf, c, None)?
    {
        return Ok(());
    }
    let cfg = match cfg {
        Some(cfg) => cfg,
        None => {
            let cfg = pick_config(ctx, d);
            check_config(ctx, cfg)?;
            cfg
        }
    };
    let vec_a = d.lda.is_multiple_of(4)
        && d.a.aligned4()
        && if d.trans_a { d.m.is_multiple_of(4) } else { d.k.is_multiple_of(4) };
    let vec_b = d.ldb.is_multiple_of(4)
        && d.b.aligned4()
        && if d.trans_b { d.k.is_multiple_of(4) } else { d.n.is_multiple_of(4) };
    let vec_c = d.ldc.is_multiple_of(4) && d.c.aligned4() && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", d.trans_a)
        .bool("TRANS_B", d.trans_b)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0);
    let k = ctx.kernel(&GEMM, spec)?;
    let groups = [
        d.n.div_ceil(cfg.bn as usize) as u32,
        d.m.div_ceil(cfg.bm as usize) as u32,
        d.batch
            .0
            .checked_mul(d.batch.1)
            .and_then(|b| u32::try_from(b).ok())
            .ok_or_else(|| GpuError::Limit("gemm batch too large".into()))?,
    ];
    ctx.dispatch(&k, &[a, b, bias_buf, c], &gemm_params(d)?, groups)
}

/// [`gemm`] with an optional fused residual epilogue (`C = act(alpha * A * B + bias + beta * res)` for
/// [`ResidualMode::PreAct`], or `C = act(alpha * A * B + bias) + beta * res` for [`ResidualMode::PostAct`]).
///
/// Runs in a single WGSL compute dispatch with zero intermediate global-memory round-trips. When `residual`
/// is `None` or has [`ResidualMode::None`], delegates directly to [`gemm`] / [`gemm_with_config`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_epilogue(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    residual: Option<(&wgpu::Buffer, &ResidualDesc)>,
    c: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<()> {
    let Some((res_buf, res)) = residual.filter(|(_, r)| r.mode != ResidualMode::None) else {
        return gemm_impl(ctx, d, a, b, bias, c, cfg);
    };
    check_dtype(ctx, d.dtype, "gemm_epilogue")?;
    if let Some(cfg) = cfg {
        check_config(ctx, cfg)?;
    }
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    if d.lda < ac || d.ldb < bc || d.ldc < d.n || res.ldr < d.n {
        return Err(shape_err(format!("gemm_epilogue: leading dimensions too small in {d:?}, {res:?}")));
    }
    if d.batch.1 == 0 {
        return Err(shape_err("gemm_epilogue: inner batch count must be non-zero"));
    }
    if d.m == 0 || d.n == 0 || d.batch.0 == 0 {
        return Ok(());
    }
    check_buf(a, extent(&d.a, d.batch, ar, ac, d.lda)?, d.dtype, "gemm_epilogue A")?;
    check_buf(b, extent(&d.b, d.batch, br, bc, d.ldb)?, d.dtype, "gemm_epilogue B")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, d.dtype, "gemm_epilogue C")?;
    check_buf(res_buf, extent(&res.strides, d.batch, d.m, d.n, res.ldr)?, d.dtype, "gemm_epilogue residual")?;
    let bias_len = match d.bias {
        GemmBias::None => 0,
        GemmBias::Row => d.m,
        GemmBias::Col => d.n,
    };
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (_, Some(bb)) => {
            check_buf(
                bb,
                extent(&d.bias_strides, d.batch, 1, bias_len, bias_len)?,
                d.dtype,
                "gemm_epilogue bias",
            )?;
            bb
        }
        (_, None) => return Err(shape_err("gemm_epilogue: bias kind set but no bias buffer")),
    };
    let cfg = match cfg {
        Some(cfg) => cfg,
        None => {
            let cfg = pick_config(ctx, d);
            check_config(ctx, cfg)?;
            cfg
        }
    };
    let vec_a = d.lda.is_multiple_of(4)
        && d.a.aligned4()
        && if d.trans_a { d.m.is_multiple_of(4) } else { d.k.is_multiple_of(4) };
    let vec_b = d.ldb.is_multiple_of(4)
        && d.b.aligned4()
        && if d.trans_b { d.k.is_multiple_of(4) } else { d.n.is_multiple_of(4) };
    let vec_c = d.ldc.is_multiple_of(4) && d.c.aligned4() && d.n.is_multiple_of(4);
    let vec_r = res.ldr.is_multiple_of(4) && res.strides.aligned4() && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("VEC_R", vec_r)
        .u32("RES_MODE", res.mode.code())
        .bool("TRANS_A", d.trans_a)
        .bool("TRANS_B", d.trans_b)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0);
    let k = ctx.kernel(&GEMM_EPILOGUE, spec)?;
    let groups = [
        d.n.div_ceil(cfg.bn as usize) as u32,
        d.m.div_ceil(cfg.bm as usize) as u32,
        d.batch
            .0
            .checked_mul(d.batch.1)
            .and_then(|b| u32::try_from(b).ok())
            .ok_or_else(|| GpuError::Limit("gemm_epilogue batch too large".into()))?,
    ];
    let p = gemm_params(d)?
        .at(38)
        .f(res.beta)
        .uz(res.ldr)?
        .uz(res.strides.offset)?
        .uz(res.strides.outer)?
        .uz(res.strides.inner)?;
    ctx.dispatch(&k, &[a, b, bias_buf, c, res_buf], &p, groups)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&GEMM, &GEMM_CAT, &GEMM_SITES, &GEMM_EPILOGUE];
