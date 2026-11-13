//! The forward conv at listed output pixels ([`conv2d_sites`]): the implicit GEMM (`CONV_MODE` 1 with `SITES`)
//! whose column `j` of image `n` is the output pixel the image's site list names, instead of pixel `j`. A
//! detector's box branch needs its convs only where a candidate is (docs/MODEL.md §9.4).
//!
//! The values are the dense conv's bit for bit: the GEMM sums every output in `K` order, one fused multiply-add
//! after another, whatever the column, so only the split-K chunks (each chunk sums on its own, then the finish
//! pass adds the chunks in order) and the tensor-core choice can change a value. Both are taken from the dense
//! conv of the same batch: the tile config and the chunks it picks for the full output.

use super::{Conv2dDesc, ConvInput, conv_spec, split};
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::{self, BatchStrides, GEMM_SITES, GemmBias, base_spec, check_dtype, gemm_params};
use crate::kernels::{check_buf, coop, prod};
use crate::pipeline::Params;
use crate::{DType, UnaryOp};

/// The site list of a `SITES` kernel: per image a count, then up to `N` (the GEMM's columns) output pixels.
pub(crate) const SITES_BINDING: &str = "@group(0) @binding(5) var<storage, read> SITE_LIST: array<u32>;\n";

/// A `SITES` kernel's live columns: image `b1`'s count, at most `N`.
pub(crate) const SITES_LIVE: &str = "  let soff = b1 * (N + 1u); let live = min(SITE_LIST[soff], N);\n";

/// Retires a workgroup of the regular generator whose columns all lie past the live ones (`b1` and `n0` derive
/// from the workgroup id and uniforms, and `SITE_LIST` is read-only, so the branch is workgroup-uniform). The
/// tensor-core generator runs no `K` tile there instead: naga rejects cooperative-matrix operations after a
/// `return` that depends on a buffer.
pub(crate) const SITES_RETIRE: &str = "  if (n0 >= live) { return; }\n";

/// The forward conv's per-column setup for column `n0 + inner`: `cjok` (a live column), the output pixel
/// `(coh, cow)` (the listed one with `sites`) and its window's top-left input pixel `(cih0, ciw0)`.
pub(crate) fn forward_column(inner: &str, sites: bool) -> String {
    let pixel = if sites {
        format!(
            "  let cj = n0 + {inner}; let cjok = cj < live;\n\
             \x20 let cpx = SITE_LIST[soff + 1u + select(0u, cj, cjok)];\n\
             \x20 let coh = cpx / OW; let cow = cpx % OW;\n"
        )
    } else {
        format!("  let cj = n0 + {inner}; let cjok = cj < N;\n  let coh = cj / OW; let cow = cj % OW;\n")
    };
    pixel + "  let cih0 = i32(coh * SH) - PH; let ciw0 = i32(cow * SW) - PW;\n"
}

/// `y = act(conv2d(x, weight) + bias)` at listed output pixels. `sites` (u32 `[N, 1 + cap]`) holds per image a
/// count, then up to `cap` output pixels `oy·OW + ox`; `y` is `[N, C_out, cap]`: column `j < min(count, cap)` of
/// image `n` is the conv at the image's `j`-th pixel, bit for bit [`super::conv2d`]'s value there for the same
/// batch (same tile config, same split-K chunks, same tensor-core choice); the other columns are unspecified.
/// `x` is read as `at` says. Ungrouped convs, at most 65535 images.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_sites(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    (x, at): (&wgpu::Buffer, ConvInput),
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    sites: &wgpu::Buffer,
    cap: usize,
    y: &wgpu::Buffer,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "conv2d_sites")?;
    if d.groups != 1 || d.n > 65535 {
        return Err(shape_err("conv2d_sites: grouped convs and batches over 65535 images are not supported"));
    }
    let (oh, ow) = d.check_operands((x, at.offset, at.n_stride), weight, bias)?;
    check_buf(sites, prod(&[d.n, cap + 1])?, DType::F32, "conv2d_sites sites")?;
    let out_elems = prod(&[d.n, d.c_out, cap])?;
    check_buf(y, out_elems, d.dtype, "conv2d_sites output")?;
    if out_elems == 0 || oh * ow == 0 {
        return Ok(());
    }
    // The dense conv's GEMM view (it picks the tiles and the chunks), and this one's: `cap` columns per image.
    let mut dense = d.gemm_view(oh, ow, bias.is_some());
    dense.b.offset = at.offset;
    dense.b.outer = at.n_stride;
    let mut g = dense.clone();
    g.n = cap;
    g.ldc = cap;
    g.c = BatchStrides { offset: 0, outer: d.c_out * cap, inner: d.c_out * cap };

    if d.dtype == DType::F16 && ctx.tensor_cores() {
        if d.is_pointwise() {
            // The dense conv is the tensor-core GEMM there, a different kernel.
            return Err(GpuError::Unsupported("conv2d_sites: a 1x1 conv on tensor cores"));
        }
        let bias_buf = bias.unwrap_or(ctx.dummy());
        if coop::encode_conv_sites(ctx, d, (&dense, &g), x, weight, bias_buf, (sites, y))? {
            return Ok(());
        }
    }

    let cfg = gemm::pick_config_conv(ctx, &dense);
    gemm::check_config(ctx, cfg)?;
    let k_dim = g.k;
    // The dense 1×1 conv is the plain GEMM, never split.
    let (splits, chunk) = if d.is_pointwise() {
        (1, k_dim)
    } else {
        let tiles = (oh * ow).div_ceil(cfg.bn as usize) * d.c_out.div_ceil(cfg.bm as usize);
        split::plan(cfg.bk as usize, k_dim, tiles, d.n, prod(&[d.n, d.c_out, oh, ow])?, None)
    };
    if splits > 1 {
        g.bias = GemmBias::None;
        g.act = UnaryOp::Identity;
    }
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !cap.is_multiple_of(cfg.bn as usize)
        || !k_dim.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", k_dim.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", cap.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1)
        .bool("SITES", true)
        .bool("SPLIT", splits > 1)
        .bool("OUT_F32", splits > 1 && d.dtype == DType::F16);
    let kernel = ctx.kernel(&GEMM_SITES, conv_spec(spec, d))?;
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = [
        u32::try_from(cap.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("conv2d_sites: too many sites".into()))?,
        d.c_out.div_ceil(cfg.bm as usize) as u32,
        (d.n * splits) as u32,
    ];
    if splits == 1 {
        ctx.dispatch(&kernel, &[weight, x, bias.unwrap_or(ctx.dummy()), y, sites], &p, groups)?;
        return Ok(());
    }
    let part = ctx.alloc(DType::F32.bytes(prod(&[splits, out_elems])?))?;
    let p: Params = p.uz(splits)?.uz(chunk)?.uz(out_elems)?;
    ctx.dispatch(&kernel, &[weight, x, ctx.dummy(), part.raw(), sites], &p, groups)?;
    let f = split::Finish { splits, out_elems, plane: cap, channels: d.c_out, act: d.act, out: d.dtype };
    split::finish(ctx, &f, part.raw(), bias, y)
}
