//! The forward conv ([`conv2d`]): the direct depthwise kernel, tensor cores for f16, a GEMM for 1×1 convs and
//! the implicit GEMM (`CONV_MODE` 1) for everything else, split along `K` when it has few output tiles
//! ([`super::split`]).

use super::depthwise::{DEPTHWISE, DW_VX};
use super::{Conv2dDesc, conv_spec, split};
use crate::context::GpuContext;
use crate::error::Result;
use crate::error::shape_err;
use crate::kernels::gemm::{self, GEMM, GemmBias, TileConfig, base_spec, check_dtype, gemm_params};
use crate::kernels::gemm_cat::{CatPart, MAX_CAT_PARTS, gemm_cat};
use crate::kernels::{coop, prod};
use crate::pipeline::Params;
use crate::{DType, UnaryOp};

/// Where a forward conv reads its input in its buffer: from element `offset`, images `n_stride` elements apart,
/// each image's `C_in·H·W` elements contiguous. A dense `[N, C, H, W]` tensor is `offset` 0, `n_stride` `C·H·W`;
/// a channel slice of a larger one (`narrow` along dim 1) is read in place with its own offset and the larger
/// tensor's image size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvInput {
    /// Element offset of image 0.
    pub offset: usize,
    /// Elements between images.
    pub n_stride: usize,
}

/// One part of the channel concatenation a 1×1 conv reads in place ([`conv2d_cat`]): `channels` channels of
/// `buf` read like a [`ConvInput`].
#[derive(Clone, Copy, Debug)]
pub struct ConvPart<'a> {
    /// The part's buffer.
    pub buf: &'a wgpu::Buffer,
    /// Its channels.
    pub channels: usize,
    /// Where its images are.
    pub at: ConvInput,
}

/// `y = act(conv2d(x, weight) + bias)`. Returns `(OH, OW)`.
///
/// `x`: `[N, C_in, H, W]`, `weight`: `[C_out, C_in/groups, KH, KW]`, `bias`: `[C_out]`, `y`: `[N, C_out, OH, OW]`.
pub fn conv2d(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
) -> Result<(usize, usize)> {
    conv2d_with_config(ctx, d, x, weight, bias, y, None)
}

/// [`conv2d`] with an explicit GEMM tile configuration for the GEMM paths (benchmarks and tests).
pub fn conv2d_with_config(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<(usize, usize)> {
    conv2d_with_plan(ctx, d, x, weight, bias, y, ConvPlan { tiles: cfg, splits: None })
}

/// How [`conv2d_with_plan`] runs a conv's GEMM: the regular kernels' tile config, and the split-K chunks on the
/// regular kernels or, with `tiles` `None`, on tensor cores; `None` fields are picked from the shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConvPlan {
    /// The regular kernels' GEMM tile config (`Some` also keeps an f16 conv off tensor cores; depthwise convs,
    /// routed first, ignore it).
    pub tiles: Option<TileConfig>,
    /// `K` chunks of the implicit GEMM, on the regular kernels or on tensor cores (split-K; clamped to `1..=8` and
    /// to what `K` allows). `Some(1)` turns split-K off. Only 1×1 stride-1 convs without padding (the pointwise
    /// GEMM) and depthwise convs ignore it; a strided 1×1 conv is implicit GEMM and honours it.
    pub splits: Option<usize>,
}

/// [`conv2d`] with an explicit [`ConvPlan`] (benchmarks and tests).
pub fn conv2d_with_plan(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    plan: ConvPlan,
) -> Result<(usize, usize)> {
    let dense = ConvInput { offset: 0, n_stride: d.c_in * d.h * d.w };
    conv2d_at(ctx, d, (x, dense), weight, bias, y, plan)
}

/// [`conv2d_with_config`] on an input read from `x` as `at` says (a channel slice read in place). Depthwise convs
/// take only dense inputs.
pub fn conv2d_input(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    (x, at): (&wgpu::Buffer, ConvInput),
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<(usize, usize)> {
    conv2d_at(ctx, d, (x, at), weight, bias, y, ConvPlan { tiles: cfg, splits: None })
}

/// The forward conv on an input at `at` in `x`, run as `plan` says.
fn conv2d_at(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    (x, at): (&wgpu::Buffer, ConvInput),
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    plan: ConvPlan,
) -> Result<(usize, usize)> {
    let cfg = plan.tiles;
    check_dtype(ctx, d.dtype, "conv2d")?;
    let (oh, ow) = d.check_input_buffers((x, at.offset, at.n_stride), weight, bias, y)?;
    if d.n * d.c_out * oh * ow == 0 {
        return Ok((oh, ow));
    }
    let dense = at.offset == 0 && (d.n <= 1 || at.n_stride == d.c_in * d.h * d.w);

    if d.is_depthwise() {
        if !dense {
            return Err(shape_err("conv2d: a depthwise conv reads only dense inputs"));
        }
        let spec = d
            .win
            .spec()
            .u32("MULT", (d.c_out / d.c_in) as u32)
            .bool("HAS_BIAS", bias.is_some())
            .u32("ACT", d.act.code())
            .f16(d.dtype == DType::F16);
        let k = ctx.kernel(&DEPTHWISE, spec)?;
        let qw = ow.div_ceil(DW_VX as usize);
        let total = d.n * d.c_out * oh * qw;
        let p = Params::new()
            .uz(total)?
            .uz(d.h)?
            .uz(d.w)?
            .uz(oh)?
            .uz(ow)?
            .uz(d.c_out)?
            .f(d.act.alpha())
            .uz(qw)?;
        ctx.dispatch(&k, &[x, weight, bias.unwrap_or(ctx.dummy()), y], &p, ctx.grid_1d(total as u64, 256)?)?;
        return Ok((oh, ow));
    }

    let (kh, kw) = d.win.kernel;
    let (k_dim, hw_out) = (d.c_in / d.groups * kh * kw, oh * ow);
    let cout_g = d.c_out / d.groups;
    let mut g = d.gemm_view(oh, ow, bias.is_some());
    g.b.offset = at.offset;
    g.b.outer = at.n_stride;

    // Tensor cores for f16 when the context has them on (an explicit tile config means the regular kernels).
    if cfg.is_none() && d.dtype == DType::F16 && ctx.tensor_cores() {
        let bias_buf = bias.unwrap_or(ctx.dummy());
        let done = if d.is_pointwise() {
            coop::encode_gemm(ctx, &g, DType::F16, weight, x, bias_buf, y, None)?
        } else {
            coop::encode_conv(ctx, d, &g, DType::F16, x, weight, bias_buf, y, None, plan.splits)?
        };
        if done {
            return Ok((oh, ow));
        }
    }

    if d.is_pointwise() {
        let cfg = cfg.unwrap_or_else(|| gemm::pick_config_conv(ctx, &g));
        gemm::gemm_with_config(ctx, &g, weight, x, bias, y, cfg)?;
        return Ok((oh, ow));
    }

    // Implicit GEMM, split along K when it has few output tiles: every chunk then writes an f32 partial slab of
    // the whole output (no bias, no activation), and one pass sums the slabs into `y` with the bias and
    // activation.
    let cfg = cfg.unwrap_or_else(|| gemm::pick_config_conv(ctx, &g));
    gemm::check_config(ctx, cfg)?;
    let (gx, gy, batch) =
        (hw_out.div_ceil(cfg.bn as usize), cout_g.div_ceil(cfg.bm as usize), d.n * d.groups);
    let out_elems = d.n * d.c_out * hw_out;
    let (splits, chunk) = split::plan(cfg.bk as usize, k_dim, gx * gy, batch, out_elems, plan.splits);
    if splits > 1 {
        g.bias = GemmBias::None;
        g.act = UnaryOp::Identity;
    }
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let vec_a = k_dim.is_multiple_of(4);
    let vec_c = hw_out % 4 == 0;
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", false)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SPLIT", splits > 1)
        .bool("OUT_F32", splits > 1 && d.dtype == DType::F16);
    let kernel = ctx.kernel(&GEMM, conv_spec(spec, d))?;
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = [gx as u32, gy as u32, u32::try_from(batch * splits).unwrap_or(u32::MAX)];
    if splits == 1 {
        ctx.dispatch(&kernel, &[weight, x, bias.unwrap_or(ctx.dummy()), y], &p, groups)?;
        return Ok((oh, ow));
    }
    let part = ctx.alloc(DType::F32.bytes(prod(&[splits, out_elems])?))?;
    let p = p.uz(splits)?.uz(chunk)?.uz(out_elems)?;
    ctx.dispatch(&kernel, &[weight, x, ctx.dummy(), part.raw()], &p, groups)?;
    let f = split::Finish { splits, out_elems, plane: hw_out, channels: d.c_out, act: d.act, out: d.dtype };
    split::finish(ctx, &f, part.raw(), bias, y)?;
    Ok((oh, ow))
}

/// `y = act(conv2d(cat(parts, 1), weight) + bias)` for a 1×1, stride-1, unpadded, ungrouped conv, reading the
/// parts of the channel concatenation in place (no concatenation is written): one GEMM whose `B` rows come from
/// the parts' buffers ([`crate::kernels::gemm_cat`]). `d.c_in` is the parts' total; 2 to
/// [`MAX_CAT_PARTS`] parts, each with at least one channel. `y` must not be the buffer of a part, the weight or
/// the bias (it is written in the dispatch that reads them).
pub fn conv2d_cat(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    parts: &[ConvPart<'_>],
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
) -> Result<(usize, usize)> {
    check_dtype(ctx, d.dtype, "conv2d_cat")?;
    if !d.is_pointwise() || d.groups != 1 {
        return Err(shape_err("conv2d_cat: only 1x1, stride-1, unpadded, ungrouped convs"));
    }
    if parts.len() < 2 || parts.len() > MAX_CAT_PARTS {
        return Err(shape_err(format!("conv2d_cat: {} parts (2 to {MAX_CAT_PARTS})", parts.len())));
    }
    if parts.iter().map(|p| p.channels).sum::<usize>() != d.c_in {
        return Err(shape_err(format!("conv2d_cat: the parts do not add up to {} channels", d.c_in)));
    }
    let hw = d.h * d.w;
    // `gemm_cat` checks the weight, bias, output and every part against its extent.
    d.validate()?;
    let (oh, ow) = d.output()?;
    let cat: Vec<CatPart<'_>> = parts
        .iter()
        .map(|p| CatPart { buf: p.buf, rows: p.channels, offset: p.at.offset, outer: p.at.n_stride })
        .collect();
    let g = d.gemm_view(oh, ow, bias.is_some());
    debug_assert_eq!(g.ldb, hw);
    gemm_cat(ctx, &g, weight, &cat, bias, y, None)?;
    Ok((oh, ow))
}
