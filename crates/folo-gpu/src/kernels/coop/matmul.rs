//! The tensor-core GEMM: validation, the kernel's specialisation and its dispatch.

use std::sync::Arc;

use super::shader::COOP;
use super::{CoopConfig, pick_config, resolve};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::check_buf;
use crate::kernels::gemm::{BatchStrides, GemmBias, GemmDesc, gemm_params};
use crate::pipeline::{Kernel, KernelDef, Spec};

pub(super) fn base_spec(d: &GemmDesc, out: DType, cfg: CoopConfig) -> Spec {
    Spec::new()
        .u32("BM", cfg.bm)
        .u32("BN", cfg.bn)
        .u32("BK", cfg.bk)
        .u32("WGM", cfg.wgm)
        .u32("WGN", cfg.wgn)
        .u32("ACT", d.act.code())
        .u32(
            "BIAS_KIND",
            match d.bias {
                GemmBias::None => 0,
                GemmBias::Row => 1,
                GemmBias::Col => 2,
            },
        )
        .bool("OUT_F16", out == DType::F16)
        .f16(true)
        .coop(true)
}

/// Builds (or fetches) a tensor-core pipeline. A build failure turns tensor cores off for the context and returns
/// `None`, so the caller can fall back to the regular kernel.
pub(super) fn build(ctx: &GpuContext, spec: Spec) -> Option<Arc<Kernel>> {
    build_def(ctx, &COOP, spec)
}

/// [`build`] of the generator's `def` variant.
pub(super) fn build_def(ctx: &GpuContext, def: &'static KernelDef, spec: Spec) -> Option<Arc<Kernel>> {
    match ctx.kernel(def, spec) {
        Ok(k) => Some(k),
        Err(e) => {
            ctx.disable_tensor_cores(format!("a tensor-core kernel failed to build ({e})"));
            None
        }
    }
}

pub(super) fn check_io(ctx: &GpuContext, input: DType, out: DType) -> Result<()> {
    if !ctx.caps().coop_f16 {
        return Err(GpuError::Unsupported("tensor cores (cooperative matrices; see FOLO_GPU_TENSOR_CORES)"));
    }
    if input != DType::F16 {
        return Err(GpuError::DType { op: "tensor-core gemm input", dtype: input });
    }
    let _ = out; // f16 or f32: both are supported.
    Ok(())
}

/// Highest element index (exclusive) a strided batched matrix operand touches.
fn extent(s: &BatchStrides, batch: (usize, usize), rows: usize, cols: usize, ld: usize) -> Result<usize> {
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

fn aligned4(s: &BatchStrides) -> bool {
    s.offset.is_multiple_of(4) && s.outer.is_multiple_of(4) && s.inner.is_multiple_of(4)
}

/// Tensor-core GEMM: `C = alpha · op(A) · op(B) (+ bias) → act` with f16 `A` and `B` (`d.dtype` must be
/// [`DType::F16`]), f32 accumulation, and `C` and the bias stored as `out` (f16 or f32). Fails with
/// [`GpuError::Unsupported`] unless the context has tensor cores ([`crate::Capabilities::coop_f16`]); ignores the
/// run-time switch ([`GpuContext::set_tensor_cores`]), which only governs the automatic routing in
/// [`crate::kernels::gemm::gemm`].
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    out: DType,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<()> {
    check_io(ctx, d.dtype, out)?;
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
    check_buf(a, extent(&d.a, d.batch, ar, ac, d.lda)?, DType::F16, "gemm A")?;
    check_buf(b, extent(&d.b, d.batch, br, bc, d.ldb)?, DType::F16, "gemm B")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, out, "gemm C")?;
    let bias_len = match d.bias {
        GemmBias::None => 0,
        GemmBias::Row => d.m,
        GemmBias::Col => d.n,
    };
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (_, Some(bb)) => {
            check_buf(bb, extent(&d.bias_strides, d.batch, 1, bias_len, bias_len)?, out, "gemm bias")?;
            bb
        }
        (_, None) => return Err(shape_err("gemm: bias kind set but no bias buffer")),
    };
    if !encode_gemm(ctx, d, out, a, b, bias_buf, c, cfg)? {
        return Err(GpuError::Unsupported("tensor cores (a tensor-core kernel failed to build)"));
    }
    Ok(())
}

/// Encodes a validated GEMM on tensor cores. Returns `false` (nothing encoded) when the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    out: DType,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    c: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<bool> {
    let batch = d.batch.0.saturating_mul(d.batch.1);
    let Some(cfg) = resolve(ctx, cfg, || pick_config(d.m, d.n, batch))? else { return Ok(false) };
    let vec_a = d.lda.is_multiple_of(4)
        && aligned4(&d.a)
        && if d.trans_a { d.m.is_multiple_of(4) } else { d.k.is_multiple_of(4) };
    let vec_b = d.ldb.is_multiple_of(4)
        && aligned4(&d.b)
        && if d.trans_b { d.k.is_multiple_of(4) } else { d.n.is_multiple_of(4) };
    let vec_c = d.ldc.is_multiple_of(4) && aligned4(&d.c) && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, out, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", d.trans_a)
        .bool("TRANS_B", d.trans_b)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0);
    let Some(k) = build(ctx, spec) else { return Ok(false) };
    let groups = [
        u32::try_from(d.n.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("gemm N too large".into()))?,
        u32::try_from(d.m.div_ceil(cfg.bm as usize))
            .map_err(|_| GpuError::Limit("gemm M too large".into()))?,
        u32::try_from(batch).map_err(|_| GpuError::Limit("gemm batch too large".into()))?,
    ];
    ctx.dispatch(&k, &[a, b, bias, c], &gemm_params(d)?, groups)?;
    Ok(true)
}

/// [`encode_gemm`] with `B` read in place from row blocks in separate buffers (`crate::kernels::gemm_cat`): `b0` is
/// part 0, `rest` the other part bindings (the dummy where unused), `p` the full parameter block with the part
/// table, `parts` the number of parts, `vec_b` whether every part allows `vec4` loads. `C` and the bias are f16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm_cat(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b0: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    c: &wgpu::Buffer,
    rest: &[&wgpu::Buffer],
    p: &crate::pipeline::Params,
    parts: u32,
    vec_b: bool,
) -> Result<bool> {
    let Some(cfg) = resolve(ctx, None, || super::pick_config_conv_batched(d.m, d.n, d.batch.0))? else {
        return Ok(false);
    };
    let vec_a = d.lda.is_multiple_of(4) && aligned4(&d.a) && d.k.is_multiple_of(4);
    let vec_c = d.ldc.is_multiple_of(4) && aligned4(&d.c) && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, DType::F16, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0)
        .u32("CAT", parts);
    let k = match ctx.kernel(&super::shader::COOP_CAT, spec) {
        Ok(k) => k,
        Err(e) => {
            ctx.disable_tensor_cores(format!("a tensor-core kernel failed to build ({e})"));
            return Ok(false);
        }
    };
    let groups = [
        u32::try_from(d.n.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("gemm N too large".into()))?,
        u32::try_from(d.m.div_ceil(cfg.bm as usize))
            .map_err(|_| GpuError::Limit("gemm M too large".into()))?,
        u32::try_from(d.batch.0).map_err(|_| GpuError::Limit("gemm batch too large".into()))?,
    ];
    let mut all = vec![a, b0, bias, c];
    all.extend_from_slice(rest);
    ctx.dispatch(&k, &all, p, groups)?;
    Ok(true)
}
