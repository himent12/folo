//! 2-D convolution, NCHW, with stride, padding, dilation, groups and fused bias + activation.
//!
//! Three paths:
//! - **1×1, stride 1, no padding**: a plain batched GEMM per (image, group): `Y[n,g] = W[g] · X[n,g]`, with
//!   `vec4` loads when `H·W` is a multiple of 4.
//! - **Depthwise** (`groups == C_in`, `C_out` a multiple of `C_in`): a direct kernel, one output per thread
//!   (the GEMM view would have a single row per group).
//! - **Everything else**: implicit GEMM. `M = C_out/groups`, `N = OH·OW`, `K = C_in/groups·KH·KW`; the GEMM
//!   generator's `CONV` mode gathers the `B` tile straight from the image (each thread keeps one output pixel
//!   for the whole K loop, so the index math is a few constant divisions per element), so no im2col buffer is
//!   ever written. Weight `[C_out, C_in/groups, KH, KW]` is the row-major `A` matrix of each group as is.

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::UNARY_FN;
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::gemm::{
    self, BatchStrides, GEMM, GemmBias, GemmDesc, TileConfig, base_spec, check_dtype, gemm_params,
};
use crate::kernels::pooling::Window2d;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// A conv2d problem.
#[derive(Clone, Debug)]
pub struct Conv2dDesc {
    /// Batch size.
    pub n: usize,
    /// Input channels.
    pub c_in: usize,
    /// Input height.
    pub h: usize,
    /// Input width.
    pub w: usize,
    /// Output channels.
    pub c_out: usize,
    /// Channel groups.
    pub groups: usize,
    /// Kernel, stride, padding, dilation.
    pub win: Window2d,
    /// Activation applied after the bias.
    pub act: UnaryOp,
    /// Storage type of input, weight, bias and output (accumulation is f32).
    pub dtype: DType,
}

impl Conv2dDesc {
    /// Output `(OH, OW)`.
    pub fn output(&self) -> Result<(usize, usize)> {
        self.win.output(self.h, self.w)
    }

    /// Multiply-accumulates ×2.
    pub fn flops(&self) -> Result<f64> {
        let (oh, ow) = self.output()?;
        Ok(2.0
            * (self.n * self.c_out * oh * ow) as f64
            * (self.c_in / self.groups.max(1) * self.win.kernel.0 * self.win.kernel.1) as f64)
    }

    fn validate(&self) -> Result<()> {
        if self.groups == 0
            || !self.c_in.is_multiple_of(self.groups)
            || !self.c_out.is_multiple_of(self.groups)
        {
            return Err(shape_err(format!(
                "conv2d: groups {} must divide C_in {} and C_out {}",
                self.groups, self.c_in, self.c_out
            )));
        }
        self.win.check_limits()
    }

    fn is_depthwise(&self) -> bool {
        self.groups > 1 && self.groups == self.c_in && self.c_out.is_multiple_of(self.c_in)
    }

    fn is_pointwise(&self) -> bool {
        self.win == Window2d::square(1, 1, 0)
    }
}

fn gen_depthwise(_: &Spec) -> String {
    format!("{UNARY_FN}{}", include_str!("../shaders/depthwise.wgsl"))
}

static DEPTHWISE: KernelDef = KernelDef {
    name: "conv2d_depthwise",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise),
};

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
    check_dtype(ctx, d.dtype, "conv2d")?;
    d.validate()?;
    let (oh, ow) = d.output()?;
    let (kh, kw) = d.win.kernel;
    let cin_g = d.c_in / d.groups;
    let cout_g = d.c_out / d.groups;
    check_buf(x, prod(&[d.n, d.c_in, d.h, d.w])?, d.dtype, "conv2d input")?;
    check_buf(weight, prod(&[d.c_out, cin_g, kh, kw])?, d.dtype, "conv2d weight")?;
    check_buf(y, prod(&[d.n, d.c_out, oh, ow])?, d.dtype, "conv2d output")?;
    if let Some(b) = bias {
        check_buf(b, d.c_out, d.dtype, "conv2d bias")?;
    }
    if d.n * d.c_out * oh * ow == 0 {
        return Ok((oh, ow));
    }

    if d.is_depthwise() {
        let spec = d
            .win
            .spec()
            .u32("MULT", (d.c_out / d.c_in) as u32)
            .bool("HAS_BIAS", bias.is_some())
            .u32("ACT", d.act.code())
            .f16(d.dtype == DType::F16);
        let k = ctx.kernel(&DEPTHWISE, spec)?;
        let total = d.n * d.c_out * oh * ow;
        let p = Params::new().uz(total)?.uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?.uz(d.c_out)?.f(d.act.alpha());
        ctx.dispatch(&k, &[x, weight, bias.unwrap_or(ctx.dummy()), y], &p, ctx.grid_1d(total as u64, 256)?)?;
        return Ok((oh, ow));
    }

    let k_dim = cin_g * kh * kw;
    let hw_in = d.h * d.w;
    let hw_out = oh * ow;
    let mut g = GemmDesc::new(cout_g, hw_out, k_dim, false, false, d.n);
    g.batch = (d.n, d.groups);
    g.lda = k_dim;
    g.ldb = hw_in;
    g.ldc = hw_out;
    g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * k_dim };
    g.b = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
    g.c = BatchStrides { offset: 0, outer: d.c_out * hw_out, inner: cout_g * hw_out };
    g.bias_strides = BatchStrides { offset: 0, outer: 0, inner: cout_g };
    g.bias = if bias.is_some() { GemmBias::Row } else { GemmBias::None };
    g.act = d.act;
    g.dtype = d.dtype;

    if d.is_pointwise() {
        let cfg = cfg.unwrap_or_else(|| gemm::pick_config(ctx, &g));
        gemm::gemm_with_config(ctx, &g, weight, x, bias, y, cfg)?;
        return Ok((oh, ow));
    }

    // Implicit GEMM.
    let cfg = cfg.unwrap_or_else(|| gemm::pick_config(ctx, &g));
    gemm::check_config(ctx, cfg)?;
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
        .bool("CONV", true);
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
    let spec = spec.i32("PH", d.win.padding.0 as i32).i32("PW", d.win.padding.1 as i32);
    let kernel = ctx.kernel(&GEMM, spec)?;
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = [
        hw_out.div_ceil(cfg.bn as usize) as u32,
        cout_g.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.n * d.groups).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[weight, x, bias.unwrap_or(ctx.dummy()), y], &p, groups)?;
    Ok((oh, ow))
}
