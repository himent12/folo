//! 2-D convolution, NCHW, with stride, padding, dilation, groups and fused bias + activation.
//!
//! Three paths:
//! - **1×1, stride 1, no padding**: a plain batched GEMM per (image, group): `Y[n,g] = W[g] · X[n,g]`, with
//!   `vec4` loads when `H·W` is a multiple of 4.
//! - **Depthwise** (`groups == C_in`, `C_out` a multiple of `C_in`): a direct kernel, four adjacent outputs of a
//!   row per thread (the GEMM view would have a single row per group).
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
//! on, the f16 input and weight gradients run on [`coop`](crate::kernels::coop)'s `CONV_MODE` 2 / 3 kernels (1×1 input gradients as a
//! GEMM over the transposed weight), and the weight gradient can be written as f32 directly
//! ([`conv2d_backward_weight_as`]).

mod depthwise;
mod forward;
mod input_grad;
pub(crate) mod split;
mod weight_grad;

pub use forward::{
    ConvInput, ConvPart, ConvPlan, conv2d, conv2d_cat, conv2d_input, conv2d_with_config, conv2d_with_plan,
};
pub use input_grad::conv2d_backward_input;
pub(crate) use input_grad::{PHASE_PIXEL, Phase};
#[cfg(test)]
pub(crate) use weight_grad::{LAST_ROUTE, WgradRoute};
pub use weight_grad::{conv2d_backward_weight, conv2d_backward_weight_as};
pub(crate) use weight_grad::{weight_grad_dims, weight_grad_spec, wgrad_b_load, wgrad_b_units};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&crate::pipeline::KernelDef] =
    &[&depthwise::DEPTHWISE, &depthwise::DEPTHWISE_GRAD, &input_grad::PHASE_PACK, &split::SPLIT_FINISH];

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::gemm::{BatchStrides, GemmBias, GemmDesc, check_dtype};
use crate::kernels::pooling::Window2d;
use crate::pipeline::Spec;

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

    /// [`Conv2dDesc::check_buffers`] for an input at `x`'s element `offset` with images `n_stride` elements apart
    /// (each image's `C_in·H·W` contiguous).
    pub(crate) fn check_input_buffers(
        &self,
        (x, offset, n_stride): (&wgpu::Buffer, usize, usize),
        weight: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        y: &wgpu::Buffer,
    ) -> Result<(usize, usize)> {
        self.validate()?;
        let (oh, ow) = self.output()?;
        let (kh, kw) = self.win.kernel;
        let image = prod(&[self.c_in, self.h, self.w])?;
        if self.n > 1 && n_stride < image {
            return Err(shape_err(format!(
                "conv2d: images {n_stride} elements apart overlap ({image} each)"
            )));
        }
        // An empty batch (or empty images) reads nothing.
        let last = match self.n.checked_sub(1) {
            Some(last_image) if image > 0 => last_image
                .checked_mul(n_stride)
                .and_then(|v| v.checked_add(offset)?.checked_add(image))
                .ok_or_else(|| shape_err("conv2d: input extent overflows"))?,
            _ => 0,
        };
        check_buf(x, last, self.dtype, "conv2d input")?;
        check_buf(
            weight,
            prod(&[self.c_out, self.c_in / self.groups, kh, kw])?,
            self.dtype,
            "conv2d weight",
        )?;
        check_buf(y, prod(&[self.n, self.c_out, oh, ow])?, self.dtype, "conv2d output")?;
        if let Some(b) = bias {
            check_buf(b, self.c_out, self.dtype, "conv2d bias")?;
        }
        Ok((oh, ow))
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
