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
//!
//! Gradients ([`conv2d_backward_input`], [`conv2d_backward_weight`]) reuse the same machinery:
//! - **Input gradient**: a transposed convolution as implicit GEMM (`CONV_MODE` 2): `M = C_in/groups`,
//!   `N = H·W`, `K = C_out/groups·KH·KW`; each thread owns one input pixel and gathers the output-gradient
//!   pixels that read it (strided convs skip the positions the stride never touched). The weight is permuted
//!   once to `[groups, C_in/groups, C_out/groups, KH·KW]`. 1×1 convs are a transposed-A GEMM; depthwise convs
//!   have a direct kernel.
//! - **Weight gradient**: implicit GEMM (`CONV_MODE` 3) with `K = N·OH·OW` folding the batch into the
//!   reduction, `M = C_out/groups`, `N = C_in/groups·KH·KW`, split along `K` into partial slabs when the
//!   output alone has too few tiles to fill the GPU, then summed. Depthwise convs have a direct kernel that
//!   reduces each tap in workgroup memory.
//!
//! Gradients take f32 or f16 operands (f32 accumulation either way; mixed-precision training). With tensor cores
//! on, the f16 input and weight gradients run on [`coop`]'s `CONV_MODE` 2 / 3 kernels (1×1 input gradients as a
//! GEMM over the transposed weight), and the weight gradient can be written as f32 directly
//! ([`conv2d_backward_weight_as`]).

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::gemm::{
    self, BatchStrides, GEMM, GemmBias, GemmDesc, TileConfig, base_spec, check_dtype, gemm_params,
};
use crate::kernels::pooling::Window2d;
use crate::kernels::{UNARY_FN, coop};
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

    /// Checks the operand buffers (input, weight and the `dtype` input storage; output and bias as `out`) and
    /// returns `(OH, OW)`.
    pub(crate) fn check_buffers(
        &self,
        x: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
        out: DType,
    ) -> Result<(usize, usize)> {
        self.validate()?;
        let (oh, ow) = self.output()?;
        let (kh, kw) = self.win.kernel;
        check_buf(x, prod(&[self.n, self.c_in, self.h, self.w])?, self.dtype, "conv2d input")?;
        check_buf(
            weight,
            prod(&[self.c_out, self.c_in / self.groups, kh, kw])?,
            self.dtype,
            "conv2d weight",
        )?;
        check_buf(y, prod(&[self.n, self.c_out, oh, ow])?, out, "conv2d output")?;
        if let Some(b) = bias {
            check_buf(b, self.c_out, out, "conv2d bias")?;
        }
        Ok((oh, ow))
    }

    /// The forward conv as a batched GEMM over `(image, group)`: `M = C_out/groups`, `N = OH·OW`,
    /// `K = C_in/groups·KH·KW`, the weight as `A`, the image as `B` (gathered for non-1×1 kernels), a row bias.
    pub(crate) fn gemm_view(&self, oh: usize, ow: usize, bias: bool) -> GemmDesc {
        let (kh, kw) = self.win.kernel;
        let (cin_g, cout_g) = (self.c_in / self.groups, self.c_out / self.groups);
        let k_dim = cin_g * kh * kw;
        let hw_in = self.h * self.w;
        let hw_out = oh * ow;
        let mut g = GemmDesc::new(cout_g, hw_out, k_dim, false, false, self.n);
        g.batch = (self.n, self.groups);
        g.lda = k_dim;
        g.ldb = hw_in;
        g.ldc = hw_out;
        g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * k_dim };
        g.b = BatchStrides { offset: 0, outer: self.c_in * hw_in, inner: cin_g * hw_in };
        g.c = BatchStrides { offset: 0, outer: self.c_out * hw_out, inner: cout_g * hw_out };
        g.bias_strides = BatchStrides { offset: 0, outer: 0, inner: cout_g };
        g.bias = if bias { GemmBias::Row } else { GemmBias::None };
        g.act = self.act;
        g.dtype = self.dtype;
        g
    }

    pub(crate) fn is_depthwise(&self) -> bool {
        self.groups > 1 && self.groups == self.c_in && self.c_out.is_multiple_of(self.c_in)
    }

    pub(crate) fn is_pointwise(&self) -> bool {
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
    let (oh, ow) = d.check_buffers(x, weight, bias, y, d.dtype)?;
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

    let (kh, kw) = d.win.kernel;
    let (k_dim, hw_out) = (d.c_in / d.groups * kh * kw, oh * ow);
    let cout_g = d.c_out / d.groups;
    let g = d.gemm_view(oh, ow, bias.is_some());

    // Tensor cores for f16 when the context has them on (an explicit tile config means the regular kernels).
    if cfg.is_none() && d.dtype == DType::F16 && ctx.tensor_cores() {
        let bias_buf = bias.unwrap_or(ctx.dummy());
        let done = if d.is_pointwise() {
            coop::encode_gemm(ctx, &g, DType::F16, weight, x, bias_buf, y, None)?
        } else {
            coop::encode_conv(ctx, d, &g, DType::F16, x, weight, bias_buf, y, None)?
        };
        if done {
            return Ok((oh, ow));
        }
    }

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
        .u32("CONV_MODE", 1);
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

// ----- gradients -----------------------------------------------------------------------------------------------

/// Direct depthwise kernels: input gradient (one thread per input element) and weight-gradient partials
/// (one workgroup per output channel and pixel chunk, the taps unrolled into registers).
fn gen_depthwise_grad(spec: &Spec) -> String {
    let (kh, kw) = (spec.get_u32("KH", 1), spec.get_u32("KW", 1));
    let mut s = String::new();
    if spec.get_u32("MODE", 0) == 0 {
        // Params: 0 = input elements, 1 = H, 2 = W, 3 = OH, 4 = OW, 5 = C_in.
        s.push_str(
            "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
             @group(0) @binding(2) var<storage, read> Wt: array<T>;\n\
             @group(0) @binding(3) var<storage, read_write> GX: array<T>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n\
             \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
             \x20 if (i >= pu(0u)) { return; }\n\
             \x20 let h = pu(1u); let w = pu(2u); let oh = pu(3u); let ow = pu(4u); let cin = pu(5u);\n\
             \x20 let ix = i % w; let t = i / w; let iy = t % h; let nc = t / h; let ci = nc % cin; let n = nc / cin;\n\
             \x20 var acc = 0.0;\n\
             \x20 for (var m = 0u; m < MULT; m++) {\n\
             \x20   let co = ci * MULT + m;\n\
             \x20   let gbase = (n * cin * MULT + co) * oh * ow;\n\
             \x20   let wbase = co * KH * KW;\n\
             \x20   for (var ky = 0u; ky < KH; ky++) {\n\
             \x20     let th = i32(iy) + PH - i32(ky * DH);\n\
             \x20     if (th < 0) { continue; }\n\
             \x20     let uh = u32(th);\n\
             \x20     if (uh % SH != 0u || uh / SH >= oh) { continue; }\n\
             \x20     for (var kx = 0u; kx < KW; kx++) {\n\
             \x20       let tw = i32(ix) + PW - i32(kx * DW);\n\
             \x20       if (tw < 0) { continue; }\n\
             \x20       let uw = u32(tw);\n\
             \x20       if (uw % SW != 0u || uw / SW >= ow) { continue; }\n\
             \x20       acc = fma(f32(G[gbase + (uh / SH) * ow + uw / SW]), f32(Wt[wbase + ky * KW + kx]), acc);\n\
             \x20     }\n\
             \x20   }\n\
             \x20 }\n\
             \x20 GX[i] = T(acc);\n}\n",
        );
        return s;
    }
    // Params: 0 = C_out, 1 = chunks S, 2 = pixels per chunk, 3 = N·OH·OW, 4 = OH·OW, 5 = OW, 6 = H, 7 = W.
    // Output: partial[s][co][tap].
    s.push_str(
        "@group(0) @binding(1) var<storage, read> G: array<T>;\n\
         @group(0) @binding(2) var<storage, read> X: array<T>;\n\
         @group(0) @binding(3) var<storage, read_write> OUT: array<f32>;\n\
         var<workgroup> red: array<f32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let g = wid.y * nwg.x + wid.x;\n\
         \x20 let cout = pu(0u); let chunks = pu(1u);\n\
         \x20 let co = g / chunks; let sidx = g % chunks;\n\
         \x20 let valid = co < cout;\n\
         \x20 let ps = pu(2u); let np = pu(3u); let ohow = pu(4u); let ow = pu(5u); let h = pu(6u); let w = pu(7u);\n",
    );
    for t in 0..kh * kw {
        s.push_str(&format!("  var acc{t} = 0.0;\n"));
    }
    s.push_str(
        "  if (valid) {\n\
         \x20   let p0 = sidx * ps; let p1 = min(p0 + ps, np);\n\
         \x20   let cin = cout / MULT; let ci = co / MULT;\n\
         \x20   for (var p = p0 + lid; p < p1; p += 256u) {\n\
         \x20     let n = p / ohow; let r = p % ohow;\n\
         \x20     let gv = f32(G[(n * cout + co) * ohow + r]);\n\
         \x20     let base = (n * cin + ci) * h * w;\n\
         \x20     let iy0 = i32((r / ow) * SH) - PH; let ix0 = i32((r % ow) * SW) - PW;\n",
    );
    for ky in 0..kh {
        s.push_str(&format!(
            "      {{ let iy = iy0 + {}; if (iy >= 0 && iy < i32(h)) {{ let row = base + u32(iy) * w;\n",
            ky as i64 * spec.get_u32("DH", 1) as i64
        ));
        for kx in 0..kw {
            let t = ky * kw + kx;
            s.push_str(&format!(
                "        {{ let ix = ix0 + {}; if (ix >= 0 && ix < i32(w)) {{ acc{t} = fma(gv, f32(X[row + u32(ix)]), acc{t}); }} }}\n",
                kx as i64 * spec.get_u32("DW", 1) as i64
            ));
        }
        s.push_str("      } }\n");
    }
    s.push_str("    }\n  }\n");
    for t in 0..kh * kw {
        s.push_str(&format!(
            "  red[lid] = acc{t};\n  workgroupBarrier();\n\
             \x20 for (var s = 128u; s > 0u; s >>= 1u) {{ if (lid < s) {{ red[lid] += red[lid + s]; }} workgroupBarrier(); }}\n\
             \x20 if (lid == 0u && valid) {{ OUT[(sidx * cout + co) * KH * KW + {t}u] = red[0]; }}\n\
             \x20 workgroupBarrier();\n"
        ));
    }
    s.push_str("}\n");
    s
}

static DEPTHWISE_GRAD: KernelDef = KernelDef {
    name: "conv2d_depthwise_grad",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_depthwise_grad),
};

/// Largest `KH · KW` the direct depthwise weight-gradient kernel unrolls; bigger kernels use the GEMM path.
const MAX_DEPTHWISE_TAPS: usize = 49;
/// Upper bound for split-K partial slabs, in f32 elements (64 MiB).
const MAX_PARTIAL_ELEMS: usize = 16 << 20;

fn check_grad_desc(ctx: &GpuContext, d: &Conv2dDesc) -> Result<(usize, usize)> {
    check_dtype(ctx, d.dtype, "conv2d backward")?;
    d.validate()?;
    d.output()
}

/// Whether an f16 gradient of `d` runs on tensor cores: the context has them on, and the conv is not depthwise
/// (those have direct kernels).
fn coop_grad(ctx: &GpuContext, d: &Conv2dDesc) -> bool {
    d.dtype == DType::F16 && ctx.tensor_cores() && !d.is_depthwise()
}

fn conv_spec(spec: Spec, d: &Conv2dDesc) -> Spec {
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

/// Gradient of [`conv2d`] with respect to its input. `d` describes the forward problem (its bias and
/// activation are ignored). `grad`: `[N, C_out, OH, OW]`, `weight`: `[C_out, C_in/groups, KH, KW]`,
/// `gx`: `[N, C_in, H, W]` (fully written), all stored as `d.dtype`: f32, or f16 with f32 accumulation (on
/// tensor cores when the context has them on, as a transposed-conv implicit GEMM or, for 1×1 convs, a GEMM over
/// the transposed weight; otherwise the regular kernels).
pub fn conv2d_backward_input(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    grad: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<()> {
    let (oh, ow) = check_grad_desc(ctx, d)?;
    let dt = d.dtype;
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let total = prod(&[d.n, d.c_in, d.h, d.w])?;
    check_buf(grad, prod(&[d.n, d.c_out, oh, ow])?, dt, "conv2d_backward_input grad")?;
    check_buf(weight, prod(&[d.c_out, cin_g, kh, kw])?, dt, "conv2d_backward_input weight")?;
    check_buf(gx, total, dt, "conv2d_backward_input output")?;
    if total == 0 {
        return Ok(());
    }
    if d.c_out * oh * ow == 0 {
        ctx.clear(gx, dt.bytes(total));
        return Ok(());
    }
    let (hw_in, hw_out, khkw) = (d.h * d.w, oh * ow, kh * kw);
    if d.is_depthwise() {
        let spec = d.win.spec().u32("MULT", (d.c_out / d.c_in) as u32).u32("MODE", 0).f16(dt == DType::F16);
        let k = ctx.kernel(&DEPTHWISE_GRAD, spec)?;
        let p = Params::new().uz(total)?.uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?.uz(d.c_in)?;
        return ctx.dispatch(&k, &[grad, weight, gx], &p, ctx.grid_1d(total as u64, 256)?);
    }
    let coop = coop_grad(ctx, d);
    if d.is_pointwise() && !coop {
        // gx[n, g] = W[g]^T · G[n, g].
        let mut g = GemmDesc::new(cin_g, hw_in, cout_g, true, false, d.n);
        g.batch = (d.n, d.groups);
        g.lda = cin_g;
        g.ldb = hw_in;
        g.ldc = hw_in;
        g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * cin_g };
        g.b = BatchStrides { offset: 0, outer: d.c_out * hw_in, inner: cout_g * hw_in };
        g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
        g.dtype = dt;
        return gemm::gemm(ctx, &g, weight, grad, None, gx);
    }
    // A[g][ci][(co, kh, kw)] = W[g * cout_g + co][ci][kh, kw].
    let wt = ctx.alloc(dt.bytes(prod(&[d.c_out, cin_g, khkw])?))?;
    super::layout::permute(ctx, weight, wt.raw(), &[d.groups, cout_g, cin_g, khkw], &[0, 2, 1, 3], dt)?;
    let k_dim = cout_g * khkw;
    let mut g = GemmDesc::new(cin_g, hw_in, k_dim, false, false, d.n);
    g.batch = (d.n, d.groups);
    g.lda = k_dim;
    g.ldb = hw_out;
    g.ldc = hw_in;
    g.a = BatchStrides { offset: 0, outer: 0, inner: cin_g * k_dim };
    g.b = BatchStrides { offset: 0, outer: d.c_out * hw_out, inner: cout_g * hw_out };
    g.c = BatchStrides { offset: 0, outer: d.c_in * hw_in, inner: cin_g * hw_in };
    g.dtype = dt;
    if coop {
        // A 1×1 conv (stride 1, no padding) has the same view as a plain GEMM over the transposed weight.
        let done = if d.is_pointwise() {
            coop::encode_gemm(ctx, &g, DType::F16, wt.raw(), grad, ctx.dummy(), gx, None)?
        } else {
            coop::encode_conv_backward_input(ctx, d, &g, wt.raw(), grad, gx)?
        };
        if done {
            return Ok(());
        }
        // The tensor-core kernel failed to build (tensor cores are off now): the regular kernel below.
    }
    let cfg = gemm::pick_config(ctx, &g);
    gemm::check_config(ctx, cfg)?;
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", k_dim.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", hw_in.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 2);
    let kernel = ctx.kernel(&GEMM, conv_spec(spec, d))?;
    let p = gemm_params(&g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = [
        hw_in.div_ceil(cfg.bn as usize) as u32,
        cin_g.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.n * d.groups).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[wt.raw(), grad, ctx.dummy(), gx], &p, groups)
}

/// Gradient of [`conv2d`] with respect to its weight, written as `d.dtype`: [`conv2d_backward_weight_as`].
pub fn conv2d_backward_weight(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gw: &wgpu::Buffer,
) -> Result<()> {
    conv2d_backward_weight_as(ctx, d, x, grad, gw, d.dtype)
}

/// Gradient of [`conv2d`] with respect to its weight. `d` describes the forward problem. `x`:
/// `[N, C_in, H, W]` and `grad`: `[N, C_out, OH, OW]` are stored as `d.dtype` (f32, or f16 with f32
/// accumulation, on tensor cores when the context has them on); `gw`: `[C_out, C_in/groups, KH, KW]` (fully
/// written) as `out`, which is f32 or `d.dtype` (mixed precision keeps f32 master weights, so an f16 conv's
/// weight gradient can be written straight to f32 without an f16 rounding or overflow in between). Split-K
/// partial sums are f32.
pub fn conv2d_backward_weight_as(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gw: &wgpu::Buffer,
    out: DType,
) -> Result<()> {
    let (oh, ow) = check_grad_desc(ctx, d)?;
    if out != DType::F32 && out != d.dtype {
        return Err(crate::error::GpuError::DType { op: "conv2d_backward_weight output", dtype: out });
    }
    let dt = d.dtype;
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let khkw = kh * kw;
    let wn = prod(&[d.c_out, cin_g, khkw])?;
    check_buf(x, prod(&[d.n, d.c_in, d.h, d.w])?, dt, "conv2d_backward_weight input")?;
    check_buf(grad, prod(&[d.n, d.c_out, oh, ow])?, dt, "conv2d_backward_weight grad")?;
    check_buf(gw, wn, out, "conv2d_backward_weight output")?;
    if wn == 0 {
        return Ok(());
    }
    let (ohow, np) = (oh * ow, prod(&[d.n, oh, ow])?);
    if np == 0 {
        ctx.clear(gw, out.bytes(wn));
        return Ok(());
    }
    // f32 partials (several chunks, or an f16 result) end in a column reduction or a cast into `gw`.
    let finish = |tmp: &crate::GpuBuffer, parts: usize| -> Result<()> {
        if parts > 1 {
            let op = super::reduce::ReduceOp::Sum;
            super::reduce::reduce_as(ctx, op, (tmp.raw(), DType::F32), (gw, out), 1, parts, wn)
        } else {
            super::layout::cast(ctx, tmp.raw(), gw, wn, true)
        }
    };
    if d.is_depthwise() && khkw <= MAX_DEPTHWISE_TAPS {
        // Enough workgroups to fill the GPU, at least 4096 pixels each.
        let chunks =
            1024usize.div_ceil(d.c_out).clamp(1, np.div_ceil(4096)).min(MAX_PARTIAL_ELEMS / wn).max(1);
        let ps = np.div_ceil(chunks);
        let chunks = np.div_ceil(ps);
        let tmp = if chunks > 1 || out != DType::F32 {
            Some(ctx.alloc(DType::F32.bytes(prod(&[chunks, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let spec = d.win.spec().u32("MULT", (d.c_out / d.c_in) as u32).u32("MODE", 1).f16(dt == DType::F16);
        let k = ctx.kernel(&DEPTHWISE_GRAD, spec)?;
        let p = Params::new().uz(d.c_out)?.uz(chunks)?.uz(ps)?.uz(np)?.uz(ohow)?.uz(ow)?.uz(d.h)?.uz(d.w)?;
        ctx.dispatch(&k, &[grad, x, target], &p, ctx.grid_1d((d.c_out * chunks) as u64 * 256, 256)?)?;
        if let Some(t) = &tmp {
            finish(t, chunks)?;
        }
        return Ok(());
    }
    let ncols = cin_g * khkw;
    let mut g = GemmDesc::new(cout_g, ncols, np, false, false, 1);
    g.batch = (1, d.groups);
    g.lda = ohow;
    g.ldb = ncols;
    g.ldc = ncols;
    g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * ohow };
    g.b = BatchStrides { offset: 0, outer: 0, inner: cin_g * d.h * d.w };
    g.c = BatchStrides { offset: 0, outer: 0, inner: cout_g * ncols };
    g.dtype = dt;
    if coop_grad(ctx, d) && coop::encode_conv_backward_weight(ctx, d, &g, out, grad, x, gw)? {
        return Ok(());
    }
    let cfg = gemm::pick_config(ctx, &g);
    gemm::check_config(ctx, cfg)?;
    let bk = cfg.bk as usize;
    // Split K when the weight alone has too few tiles: aim for ~512 workgroups, each at least 16 K tiles deep.
    let tiles = cout_g.div_ceil(cfg.bm as usize) * ncols.div_ceil(cfg.bn as usize) * d.groups;
    let want =
        512usize.div_ceil(tiles).min(np.div_ceil(16 * bk)).min(MAX_PARTIAL_ELEMS / wn).min(65535 / d.groups);
    let ks = np.div_ceil(want.max(1)).next_multiple_of(bk);
    let splits = np.div_ceil(ks);
    // The kernel writes `C` as f32 for partial slabs and for an f32 result of f16 inputs.
    let c_f32 = splits > 1 || out == DType::F32;
    let tmp = if splits > 1 { Some(ctx.alloc(DType::F32.bytes(prod(&[splits, wn])?))?) } else { None };
    let target = tmp.as_ref().map_or(gw, |t| t.raw());
    let check = !cout_g.is_multiple_of(cfg.bm as usize)
        || !ncols.is_multiple_of(cfg.bn as usize)
        || !np.is_multiple_of(bk);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", ohow.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", ncols.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 3)
        .bool("SPLIT", splits > 1)
        .bool("OUT_F32", c_f32 && dt == DType::F16);
    let kernel = ctx.kernel(&GEMM, conv_spec(spec, d))?;
    let p = gemm_params(&g)?
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
    let groups = [
        ncols.div_ceil(cfg.bn as usize) as u32,
        cout_g.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.groups * splits).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, groups)?;
    if let Some(t) = &tmp {
        finish(t, splits)?;
    }
    Ok(())
}
