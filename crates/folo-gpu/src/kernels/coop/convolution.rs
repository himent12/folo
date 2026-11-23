//! The tensor-core conv2d: the forward pass (pointwise convs as the GEMM, the others as the implicit-GEMM conv)
//! and the input and weight gradients.

#[cfg(doc)]
use super::matmul::gemm;
use super::matmul::{base_spec, build, build_def, check_io, encode_gemm};
use super::shader::COOP_SITES;
use super::{CoopConfig, conv_config_batched, pick_config_conv, pick_config_conv_batched, resolve};
use crate::DType;
use crate::UnaryOp;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::kernels::conv::{Conv2dDesc, split};
use crate::kernels::gemm::{BatchStrides, GemmBias, GemmDesc, gemm_params};
use crate::pipeline::Spec;

/// Tensor-core conv2d: `y = act(conv2d(x, weight) + bias)` with f16 `x` and `weight` (`d.dtype` must be
/// [`DType::F16`]), f32 accumulation, and `y` and the bias stored as `out` (f16 or f32). 1×1 stride-1 convs run as
/// the tensor-core GEMM, other non-depthwise convs as the implicit-GEMM conv; depthwise convs have no tensor-core
/// path (one output row per group) and fail with [`GpuError::Unsupported`], as does a context without tensor cores.
/// Without `cfg`, both paths take [`pick_config_conv_batched`]. Ignores the run-time switch like [`gemm`].
/// Returns `(OH, OW)`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    out: DType,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<(usize, usize)> {
    check_io(ctx, d.dtype, out)?;
    let (oh, ow) = d.check_buffers(x, weight, bias, y, out)?;
    if d.is_depthwise() {
        return Err(GpuError::Unsupported("tensor-core depthwise conv (use conv::conv2d)"));
    }
    if d.n * d.c_out * oh * ow == 0 {
        return Ok((oh, ow));
    }
    let g = d.gemm_view(oh, ow, bias.is_some());
    let bias = bias.unwrap_or(ctx.dummy());
    let built = if d.is_pointwise() {
        encode_gemm(
            ctx,
            &g,
            out,
            weight,
            x,
            bias,
            y,
            cfg.or_else(|| conv_config_batched(ctx, g.m, g.n, d.n * d.groups)),
        )?
    } else {
        encode_conv(ctx, d, &g, out, x, weight, bias, y, cfg, None)?
    };
    if !built {
        return Err(GpuError::Unsupported("tensor cores (a tensor-core kernel failed to build)"));
    }
    Ok((oh, ow))
}

/// The implicit-GEMM conv on tensor cores for a validated, non-depthwise, non-pointwise f16 conv `d` whose GEMM
/// view is `g` (as built by [`crate::kernels::conv::conv2d`]), split along `K` when it has few output tiles
/// (`want_splits` forces the number of chunks, as [`crate::kernels::conv::ConvPlan::splits`]). Returns `false`
/// (nothing encoded) when the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    out: DType,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
    want_splits: Option<usize>,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let (batch, out_elems) = (d.n * d.groups, d.n * d.c_out * oh * ow);
    let Some(cfg) = resolve(ctx, cfg, || pick_config_conv_batched(g.m, g.n, batch))? else {
        return Ok(false);
    };
    // Split K when the conv has few output tiles (as the regular kernels do; `crate::kernels::conv::split`):
    // f32 partial slabs without bias and activation, then one pass sums them into `y` with both.
    let tiles = g.m.div_ceil(cfg.bm as usize) * g.n.div_ceil(cfg.bn as usize);
    let (splits, chunk) = split::plan(cfg.bk as usize, g.k, tiles, batch, out_elems, want_splits);
    let mut g = g.clone();
    let has_bias = g.bias != GemmBias::None;
    if splits > 1 {
        g.bias = GemmBias::None;
        g.act = UnaryOp::Identity;
    }
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let mut spec = base_spec(&g, if splits > 1 { DType::F32 } else { out }, cfg)
        .bool("VEC_A", g.k.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SPLIT", splits > 1);
    spec = with_window(spec, d);
    let Some(kernel) = build(ctx, spec) else { return Ok(false) };
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = grid(&g, cfg, batch * splits)?;
    if splits == 1 {
        ctx.dispatch(&kernel, &[weight, x, bias, y], &p, groups)?;
        return Ok(true);
    }
    let part = ctx.alloc(DType::F32.bytes(crate::kernels::prod(&[splits, out_elems])?))?;
    let p = p.uz(splits)?.uz(chunk)?.uz(out_elems)?;
    ctx.dispatch(&kernel, &[weight, x, ctx.dummy(), part.raw()], &p, groups)?;
    let f = split::Finish { splits, out_elems, plane: oh * ow, channels: d.c_out, act: d.act, out };
    split::finish(ctx, &f, part.raw(), has_bias.then_some(bias), y)?;
    Ok(true)
}

/// [`encode_conv`] at listed output pixels (`crate::kernels::conv::conv2d_sites`): `dense` is the conv's GEMM view
/// (it picks the tile config and the split-K chunks, as [`encode_conv`] would), `g` the view with one column per
/// site slot; `sites` lists the pixels, `y` takes the `[N, C_out, slots]` result. Returns `false` (nothing encoded)
/// when the kernel failed to build.
pub(crate) fn encode_conv_sites(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    (dense, g): (&GemmDesc, &GemmDesc),
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    (sites, y): (&wgpu::Buffer, &wgpu::Buffer),
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let (batch, dense_elems) = (d.n * d.groups, d.n * d.c_out * oh * ow);
    let Some(cfg) = resolve(ctx, None, || pick_config_conv_batched(dense.m, dense.n, batch))? else {
        return Ok(false);
    };
    let tiles = dense.m.div_ceil(cfg.bm as usize) * dense.n.div_ceil(cfg.bn as usize);
    let (splits, chunk) = split::plan(cfg.bk as usize, g.k, tiles, batch, dense_elems, None);
    let mut g = g.clone();
    let has_bias = g.bias != GemmBias::None;
    if splits > 1 {
        g.bias = GemmBias::None;
        g.act = UnaryOp::Identity;
    }
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(&g, if splits > 1 { DType::F32 } else { DType::F16 }, cfg)
        .bool("VEC_A", g.k.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SITES", true)
        .bool("SPLIT", splits > 1);
    let Some(kernel) = build_def(ctx, &COOP_SITES, with_window(spec, d)) else { return Ok(false) };
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = grid(&g, cfg, batch * splits)?;
    if splits == 1 {
        ctx.dispatch(&kernel, &[weight, x, bias, y, sites], &p, groups)?;
        return Ok(true);
    }
    let out_elems = d.n * d.c_out * g.n;
    let part = ctx.alloc(DType::F32.bytes(crate::kernels::prod(&[splits, out_elems])?))?;
    let p = p.uz(splits)?.uz(chunk)?.uz(out_elems)?;
    ctx.dispatch(&kernel, &[weight, x, ctx.dummy(), part.raw(), sites], &p, groups)?;
    let f = split::Finish { splits, out_elems, plane: g.n, channels: d.c_out, act: d.act, out: DType::F16 };
    split::finish(ctx, &f, part.raw(), has_bias.then_some(bias), y)?;
    Ok(true)
}

/// The conv window as the generator's specialisation constants.
fn with_window(spec: Spec, d: &Conv2dDesc) -> Spec {
    let (kh, kw) = d.win.kernel;
    let mut spec = spec;
    for (name, v) in [
        ("KH", kh as u32),
        ("KW", kw as u32),
        ("SH", d.win.stride.0 as u32),
        ("SW", d.win.stride.1 as u32),
        ("DH", d.win.dilation.0 as u32),
        ("DW", d.win.dilation.1 as u32),
    ] {
        spec = spec.u32(name, v);
    }
    spec.i32("PH", d.win.padding.0 as i32).i32("PW", d.win.padding.1 as i32)
}

/// Workgroup grid of a GEMM view `g` with `layers` z-layers.
fn grid(g: &GemmDesc, cfg: CoopConfig, layers: usize) -> Result<[u32; 3]> {
    Ok([
        u32::try_from(g.n.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("conv output too large".into()))?,
        u32::try_from(g.m.div_ceil(cfg.bm as usize))
            .map_err(|_| GpuError::Limit("conv rows too large".into()))?,
        u32::try_from(layers)
            .ok()
            .filter(|&z| z <= 65535)
            .ok_or_else(|| GpuError::Limit("conv batch too large".into()))?,
    ])
}

/// The input gradient of the f16 conv `d` on tensor cores (`CONV_MODE` 2, a transposed conv as implicit GEMM):
/// `g` is the GEMM view built by [`crate::kernels::conv::conv2d_backward_input`] (`M = C_in/groups`, `N = H·W`,
/// `K = C_out/groups·KH·KW`), `wt` the weight permuted to `[groups, C_in/groups, C_out/groups, KH·KW]`, `grad` the
/// output gradient and `gx` the f16 result. Uses [`pick_config_conv`] (`C64X64`/`C64X128`, avoiding `C128X128`
/// occupancy cliffs) and `tap_mask` on stride-1 convs with `KH·KW <= 32`. Returns `false` (nothing encoded) when
/// the kernel failed to build.
pub(crate) fn encode_conv_backward_input(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    wt: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let Some(cfg) = resolve(ctx, None, || pick_config_conv(g.m, g.n))? else { return Ok(false) };
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(g, DType::F16, cfg)
        .bool("VEC_A", g.k.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 2);
    let Some(kernel) = build(ctx, with_window(spec, d)) else { return Ok(false) };
    let p = gemm_params(g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    ctx.dispatch(&kernel, &[wt, grad, ctx.dummy(), gx], &p, grid(g, cfg, d.n * d.groups)?)?;
    Ok(true)
}

/// Tile pick for stride-2 `dX` (sub-pixel and phase): `C64X128` starting at `N >= 1600` (40×40) when `M >= 128`
/// to halve weight traffic on wide layers (`128→128 @ 80`, `256→256 @ 80`), else [`pick_config_conv`].
fn pick_config_stride2_dx(m: usize, n: usize) -> CoopConfig {
    if m >= 128 && n >= 1600 { CoopConfig::C64X128 } else { pick_config_conv(m, n) }
}

/// One stride phase of the f16 conv `d`'s input gradient on tensor cores (`CONV_MODE` 2 with `PHASE`, see
/// [`crate::kernels::conv`]'s stride-phase decomposition): `g` is the phase's GEMM view (`M = C_in/groups`, `N` its
/// pixels, `K = C_out/groups · th · tw`), `packed` the phase-packed weight. Returns `false` (nothing encoded) when
/// the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_backward_input_phase(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    phase: &crate::kernels::conv::Phase,
    vec_a: bool,
    packed: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let Some(cfg) = resolve(ctx, None, || pick_config_stride2_dx(g.m, g.n))? else { return Ok(false) };
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize)
        || g.k == 0;
    let spec = base_spec(g, DType::F16, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", false)
        .bool("VEC_C", false)
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 2);
    let Some(kernel) = build(ctx, phase.spec(with_window(spec, d))) else { return Ok(false) };
    let p = phase.params(gemm_params(g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?)?;
    ctx.dispatch(&kernel, &[packed, grad, ctx.dummy(), gx], &p, grid(g, cfg, d.n * d.groups)?)?;
    Ok(true)
}

/// Fused 2×2 sub-pixel stride-2 input gradient on tensor cores (`CONV_MODE` 1 with `SUBPIXEL`): one 2×2
/// stride-1 conv (`C_out → 4·C_in`) over `grad` whose epilogue shuffles directly into `gx`. Returns `false`
/// (nothing encoded) when the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_backward_input_subpixel(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    (oh, ow, uh, uw, bh, bw): (usize, usize, usize, usize, i32, i32),
    packed: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<bool> {
    let Some(cfg) = resolve(ctx, None, || pick_config_stride2_dx(g.m, g.n))? else { return Ok(false) };
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(g, DType::F16, cfg)
        .bool("VEC_A", true)
        .bool("VEC_B", false)
        .bool("VEC_C", d.w.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SUBPIXEL", true)
        .u32("KH", 2)
        .u32("KW", 2)
        .u32("SH", 1)
        .u32("SW", 1)
        .u32("DH", 1)
        .u32("DW", 1)
        .i32("PH", -bh)
        .i32("PW", -bw);
    let Some(kernel) = build(ctx, spec) else { return Ok(false) };
    let p = gemm_params(g)?.at(24).uz(oh)?.uz(ow)?.uz(uh)?.uz(uw)?.uz(d.h)?.uz(d.w)?;
    ctx.dispatch(&kernel, &[packed, grad, ctx.dummy(), gx], &p, grid(g, cfg, d.n * d.groups)?)?;
    Ok(true)
}

/// Tile configuration for a windowed weight gradient (`M = C_out/groups` rows, `N = C_in/groups·KH·KW` columns):
/// `C32X32` for narrow weights (e.g. `folo-s`'s `3→32` stem with `N = 27`), otherwise capped at `C64X128`/`C64X64`
/// (never `C128X128`, avoiding Ada shared-memory occupancy cliffs in the `CONV_MODE` 3 gather).
fn pick_config_weight_grad(m: usize, n: usize) -> CoopConfig {
    if n <= 32 || (m <= 32 && n <= 64) {
        return CoopConfig::C32X32;
    }
    match m {
        0..=16 => CoopConfig::C16X256,
        17..=32 => CoopConfig::C32X128,
        _ if n >= 128 => CoopConfig::C64X128,
        _ => CoopConfig::C64X64,
    }
}

/// Tile configuration for the 1×1 `A · Bᵀ` weight gradient: `C32X32` for narrow `N <= 32` (avoids wasted
/// columns), `C64X128` for wide `N >= 128`, `C64X64` elsewhere (stays at 24–35 KiB shared memory so Ada keeps
/// full occupancy, never `C128X128`).
fn pick_config_pointwise_wgrad(m: usize, n: usize) -> CoopConfig {
    if m <= 16 && n >= 128 {
        CoopConfig::C16X256
    } else if m <= 32 && n >= 96 {
        CoopConfig::C32X128
    } else if n <= 32 || (m <= 32 && n <= 64) {
        CoopConfig::C32X32
    } else if n >= 128 {
        CoopConfig::C64X128
    } else {
        CoopConfig::C64X64
    }
}

/// Upper bound for split-K partial slabs, in f32 elements (64 MiB), as for the regular kernels.
const MAX_PARTIAL_ELEMS: usize = 16 << 20;

/// The weight gradient of the f16 conv `d` on tensor cores: 1×1 stride-1 unpadded convs run as a division-free
/// contiguous `vec4` per-image split-K `A · Bᵀ` GEMM (`CONV_MODE` 0), other convs as implicit GEMM (`CONV_MODE` 3,
/// per-image when the split count already covers `N` so `/ OHOW` and `% OHOW` vanish). Partial sums stay f32 and
/// are reduced into `gw` (`out`, f32 or f16). Returns `false` (nothing encoded) when the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_backward_weight(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    out: DType,
    grad: &wgpu::Buffer,
    x: &wgpu::Buffer,
    gw: &wgpu::Buffer,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let (ohow, np, wn) = (oh * ow, g.k, g.m * g.n * d.groups);
    if let Some(pw_cfg) = resolve(ctx, None, || pick_config_pointwise_wgrad(g.m, g.n))?
        && crate::kernels::conv::can_pointwise_wgrad(d, pw_cfg.bk as usize, wn)
    {
        let (cin_g, cout_g, hw) = (d.c_in / d.groups, d.c_out / d.groups, ohow);
        let (sp, chunk_p) = crate::kernels::conv::plan_pointwise_wgrad(
            d,
            pw_cfg.bm as usize,
            pw_cfg.bn as usize,
            pw_cfg.bk as usize,
            pw_cfg.threads() as usize,
            wn,
        );
        let total_slabs = d.n * sp;
        let mut gp = GemmDesc::new(cout_g, cin_g, hw, false, true, d.n);
        gp.batch = (d.n, d.groups);
        gp.lda = hw;
        gp.ldb = hw;
        gp.ldc = cin_g;
        gp.a = BatchStrides { offset: 0, outer: d.c_out * hw, inner: cout_g * hw };
        gp.b = BatchStrides { offset: 0, outer: d.c_in * hw, inner: cin_g * hw };
        gp.c = BatchStrides { offset: 0, outer: wn, inner: cout_g * cin_g };
        gp.dtype = DType::F16;
        let tmp = if total_slabs > 1 {
            Some(ctx.alloc(DType::F32.bytes(crate::kernels::prod(&[total_slabs, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let out_dt = if total_slabs > 1 { DType::F32 } else { out };
        let check = !cout_g.is_multiple_of(pw_cfg.bm as usize)
            || !cin_g.is_multiple_of(pw_cfg.bn as usize)
            || !hw.is_multiple_of(pw_cfg.bk as usize);
        let spec = base_spec(&gp, out_dt, pw_cfg)
            .bool("VEC_A", hw.is_multiple_of(4))
            .bool("VEC_B", hw.is_multiple_of(4))
            .bool("VEC_C", cin_g.is_multiple_of(4))
            .bool("TRANS_A", false)
            .bool("TRANS_B", true)
            .bool("CHECK", check)
            .u32("CONV_MODE", 0)
            .bool("SPLIT", sp > 1);
        let Some(kernel) = build(ctx, spec) else { return Ok(false) };
        #[cfg(test)]
        crate::kernels::conv::LAST_ROUTE.set(Some((true, (pw_cfg.bm, pw_cfg.bn), total_slabs)));
        let p = if sp > 1 {
            gemm_params(&gp)?.at(28).uz(sp)?.uz(chunk_p)?.uz(d.n * wn)?
        } else {
            gemm_params(&gp)?
        };
        ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, grid(&gp, pw_cfg, d.n * d.groups * sp)?)?;
        if let Some(t) = &tmp {
            crate::kernels::conv::finish_wgrad(ctx, t.raw(), total_slabs, wn, gw, out)?;
        }
        return Ok(true);
    }
    let Some(cfg) = resolve(ctx, None, || pick_config_weight_grad(g.m, g.n))? else { return Ok(false) };
    let bk = cfg.bk as usize;
    // Aim for ~512 workgroups, each at least 16 K tiles deep.
    let tiles = g.m.div_ceil(cfg.bm as usize) * g.n.div_ceil(cfg.bn as usize) * d.groups;
    let want = 512usize
        .div_ceil(tiles)
        .min(np.div_ceil(16 * bk))
        .min(MAX_PARTIAL_ELEMS / wn.max(1))
        .min(65535 / d.groups)
        .max(1);
    let per_img = ohow >= bk
        && d.n.saturating_mul(d.groups) <= 65535
        && d.n.saturating_mul(wn) <= MAX_PARTIAL_ELEMS
        && want >= d.n;
    if per_img {
        let (cin_g, cout_g, ncols) = (d.c_in / d.groups, d.c_out / d.groups, g.n);
        let (sp, chunk_p) =
            crate::kernels::conv::plan_wgrad_per_img(d, ohow, tiles, bk, cfg.threads() as usize, wn, 16);
        let total_slabs = d.n * sp;
        let mut gp = GemmDesc::new(cout_g, ncols, ohow, false, false, d.n);
        gp.batch = (d.n, d.groups);
        gp.lda = ohow;
        gp.ldb = ncols;
        gp.ldc = ncols;
        gp.a = BatchStrides { offset: 0, outer: d.c_out * ohow, inner: cout_g * ohow };
        gp.b = BatchStrides { offset: 0, outer: d.c_in * d.h * d.w, inner: cin_g * d.h * d.w };
        gp.c = BatchStrides { offset: 0, outer: wn, inner: cout_g * ncols };
        gp.dtype = DType::F16;
        let tmp = if total_slabs > 1 {
            Some(ctx.alloc(DType::F32.bytes(crate::kernels::prod(&[total_slabs, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let out_dt = if total_slabs > 1 { DType::F32 } else { out };
        let check = !cout_g.is_multiple_of(cfg.bm as usize)
            || !ncols.is_multiple_of(cfg.bn as usize)
            || !ohow.is_multiple_of(bk);
        let spec = base_spec(&gp, out_dt, cfg)
            .bool("VEC_A", ohow.is_multiple_of(4))
            .bool("VEC_B", false)
            .bool("VEC_C", ncols.is_multiple_of(4))
            .bool("TRANS_A", false)
            .bool("TRANS_B", false)
            .bool("CHECK", check)
            .u32("CONV_MODE", 3)
            .bool("PER_IMG", true)
            .bool("SPLIT", sp > 1);
        let spec = crate::kernels::conv::weight_grad_spec(with_window(spec, d), d, oh, ow);
        let Some(kernel) = build(ctx, spec) else { return Ok(false) };
        #[cfg(test)]
        crate::kernels::conv::LAST_ROUTE.set(Some((true, (cfg.bm, cfg.bn), total_slabs)));
        let p = gemm_params(&gp)?
            .at(24)
            .uz(d.h)?
            .uz(d.w)?
            .uz(oh)?
            .uz(ow)?
            .uz(sp)?
            .uz(chunk_p)?
            .uz(d.n * wn)?
            .uz(ohow)?;
        ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, grid(&gp, cfg, d.n * d.groups * sp)?)?;
        if let Some(t) = &tmp {
            crate::kernels::conv::finish_wgrad(ctx, t.raw(), total_slabs, wn, gw, out)?;
        }
        return Ok(true);
    }
    let ks = np.div_ceil(want).next_multiple_of(bk);
    let splits = np.div_ceil(ks);
    let tmp = if splits > 1 || out != DType::F32 {
        Some(ctx.alloc(DType::F32.bytes(crate::kernels::prod(&[splits, wn])?))?)
    } else {
        None
    };
    let target = tmp.as_ref().map_or(gw, |t| t.raw());
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !np.is_multiple_of(bk);
    let spec = base_spec(g, DType::F32, cfg)
        .bool("VEC_A", ohow.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 3)
        .bool("SPLIT", splits > 1);
    let spec = crate::kernels::conv::weight_grad_spec(with_window(spec, d), d, oh, ow);
    let Some(kernel) = build(ctx, spec) else { return Ok(false) };
    #[cfg(test)]
    crate::kernels::conv::LAST_ROUTE.set(Some((true, (cfg.bm, cfg.bn), splits)));
    let p = gemm_params(g)?
        .at(24)
        .uz(d.h)?
        .uz(d.w)?
        .uz(oh)?
        .uz(ow)?
        .uz(splits)?
        .uz(ks)?
        .uz(wn)?
        .uz(ohow)?
        .uz(d.c_out * ohow)?
        .uz(d.c_in * d.h * d.w)?;
    ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, grid(g, cfg, d.groups * splits)?)?;
    if let Some(t) = &tmp {
        crate::kernels::conv::finish_wgrad(ctx, t.raw(), splits, wn, gw, out)?;
    }
    Ok(true)
}
