//! [`GpuTensor`]: a dense row-major buffer + shape + dtype, with methods that call the kernel layer.

use std::sync::Arc;

use crate::context::GpuContext;
use crate::dtype::DType;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::conv::{self, Conv2dDesc};
use crate::kernels::elementwise::{self, BinaryOp, UnaryOp};
use crate::kernels::gemm::{self, BatchStrides, GemmBias, GemmDesc};
use crate::kernels::layout;
use crate::kernels::pooling::{self, Window2d};
use crate::kernels::reduce::{self, ReduceOp};
use crate::kernels::{prod, shape_numel};
use crate::pool::GpuBuffer;

/// A dense, row-major tensor in GPU memory.
///
/// Cloning is cheap (the buffer is shared). Operations allocate their output from the context's pool and are
/// encoded into the open batch; nothing waits for the GPU until [`GpuTensor::to_vec`] or [`GpuContext::sync`].
#[derive(Clone)]
pub struct GpuTensor {
    ctx: GpuContext,
    buf: Arc<GpuBuffer>,
    shape: Vec<usize>,
    dtype: DType,
}

impl std::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor").field("shape", &self.shape).field("dtype", &self.dtype).finish()
    }
}

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl GpuTensor {
    /// Wraps an existing buffer. Fails if it is too small for `shape` and `dtype`.
    pub fn from_buffer(ctx: &GpuContext, buf: Arc<GpuBuffer>, shape: &[usize], dtype: DType) -> Result<Self> {
        let need = dtype.bytes(shape_numel(shape)?);
        if buf.size() < need {
            return Err(shape_err(format!(
                "buffer of {} bytes is too small for {shape:?} {dtype:?}",
                buf.size()
            )));
        }
        Ok(Self { ctx: ctx.clone(), buf, shape: shape.to_vec(), dtype })
    }

    /// A tensor with unspecified contents.
    pub fn empty(ctx: &GpuContext, shape: &[usize], dtype: DType) -> Result<Self> {
        let buf = ctx.alloc(dtype.bytes(shape_numel(shape)?))?;
        Ok(Self { ctx: ctx.clone(), buf: Arc::new(buf), shape: shape.to_vec(), dtype })
    }

    /// A tensor of zeros.
    pub fn zeros(ctx: &GpuContext, shape: &[usize], dtype: DType) -> Result<Self> {
        let t = Self::empty(ctx, shape, dtype)?;
        ctx.clear(t.raw(), dtype.bytes(t.numel()));
        Ok(t)
    }

    /// Uploads f32 data.
    pub fn from_slice(ctx: &GpuContext, data: &[f32], shape: &[usize]) -> Result<Self> {
        if data.len() != shape_numel(shape)? {
            return Err(shape_err(format!("{} values for shape {shape:?}", data.len())));
        }
        let buf = ctx.upload(bytemuck::cast_slice(data))?;
        Ok(Self { ctx: ctx.clone(), buf: Arc::new(buf), shape: shape.to_vec(), dtype: DType::F32 })
    }

    /// Uploads f16 data.
    pub fn from_f16(ctx: &GpuContext, data: &[half::f16], shape: &[usize]) -> Result<Self> {
        if data.len() != shape_numel(shape)? {
            return Err(shape_err(format!("{} values for shape {shape:?}", data.len())));
        }
        let buf = ctx.upload(bytemuck::cast_slice(data))?;
        Ok(Self { ctx: ctx.clone(), buf: Arc::new(buf), shape: shape.to_vec(), dtype: DType::F16 })
    }

    /// Downloads as f32 (f16 tensors are widened on the host). Waits for pending work.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let bytes = self.ctx.read(self.raw(), 0, (self.numel() * self.dtype.size()) as u64)?;
        Ok(match self.dtype {
            DType::F32 => bytemuck::pod_collect_to_vec(&bytes),
            DType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        })
    }

    /// Downloads an f16 tensor without widening.
    pub fn to_f16_vec(&self) -> Result<Vec<half::f16>> {
        self.require(DType::F16, "to_f16_vec")?;
        let bytes = self.ctx.read(self.raw(), 0, (self.numel() * 2) as u64)?;
        Ok(bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect())
    }

    /// The context this tensor lives on.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    /// The pooled buffer.
    pub fn buffer(&self) -> &Arc<GpuBuffer> {
        &self.buf
    }

    /// The raw wgpu buffer (dense row-major data from byte 0).
    pub fn raw(&self) -> &wgpu::Buffer {
        self.buf.raw()
    }

    /// Same data, new shape (no copy). One dimension may be `usize::MAX` to infer it.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let mut shape = shape.to_vec();
        let known = prod(&shape.iter().copied().filter(|&d| d != usize::MAX).collect::<Vec<_>>())?;
        let infer = shape.iter().filter(|&&d| d == usize::MAX).count();
        if infer > 1 {
            return Err(shape_err("reshape: at most one inferred dimension"));
        }
        if infer == 1 {
            if known == 0 || !self.numel().is_multiple_of(known) {
                return Err(shape_err(format!("reshape {:?} -> {shape:?}", self.shape)));
            }
            let d = self.numel() / known;
            shape.iter_mut().filter(|d| **d == usize::MAX).for_each(|x| *x = d);
        }
        if shape_numel(&shape)? != self.numel() {
            return Err(shape_err(format!("reshape {:?} -> {shape:?}", self.shape)));
        }
        Ok(Self { ctx: self.ctx.clone(), buf: self.buf.clone(), shape, dtype: self.dtype })
    }

    pub(crate) fn require(&self, dtype: DType, op: &'static str) -> Result<()> {
        if self.dtype != dtype {
            return Err(GpuError::DType { op, dtype: self.dtype });
        }
        Ok(())
    }

    pub(crate) fn same_ctx(&self, other: &GpuTensor) -> Result<()> {
        if !self.ctx.same_device(&other.ctx) {
            return Err(shape_err("tensors live on different GPU contexts"));
        }
        Ok(())
    }

    // ---- elementwise --------------------------------------------------------------------------------------

    /// Elementwise unary op (f32).
    pub fn unary(&self, op: UnaryOp) -> Result<Self> {
        self.require(DType::F32, "unary")?;
        let out = Self::empty(&self.ctx, &self.shape, DType::F32)?;
        elementwise::unary(&self.ctx, op, self.raw(), out.raw(), self.numel())?;
        Ok(out)
    }

    /// Elementwise binary op with NumPy broadcasting (f32).
    pub fn binary(&self, op: BinaryOp, rhs: &GpuTensor) -> Result<Self> {
        self.require(DType::F32, "binary")?;
        rhs.require(DType::F32, "binary")?;
        self.same_ctx(rhs)?;
        let shape = elementwise::broadcast_shape(&self.shape, &rhs.shape)?;
        let out = Self::empty(&self.ctx, &shape, DType::F32)?;
        elementwise::binary(&self.ctx, op, self.raw(), &self.shape, rhs.raw(), &rhs.shape, out.raw())?;
        Ok(out)
    }

    /// `self + rhs` (broadcasting).
    pub fn add(&self, rhs: &GpuTensor) -> Result<Self> {
        self.binary(BinaryOp::Add, rhs)
    }

    /// `self - rhs` (broadcasting).
    pub fn sub(&self, rhs: &GpuTensor) -> Result<Self> {
        self.binary(BinaryOp::Sub, rhs)
    }

    /// `self * rhs` (broadcasting).
    pub fn mul(&self, rhs: &GpuTensor) -> Result<Self> {
        self.binary(BinaryOp::Mul, rhs)
    }

    /// `self / rhs` (broadcasting).
    pub fn div(&self, rhs: &GpuTensor) -> Result<Self> {
        self.binary(BinaryOp::Div, rhs)
    }

    /// Fused `act(self + bias)` with `bias` of shape `[C]` broadcast along dim `dim` (1 for NCHW, the last dim
    /// for `[rows, features]`). `bias = None` applies only the activation.
    pub fn bias_act(&self, bias: Option<&GpuTensor>, dim: usize, act: UnaryOp) -> Result<Self> {
        self.require(DType::F32, "bias_act")?;
        let (channels, inner) = if let Some(b) = bias {
            b.require(DType::F32, "bias_act")?;
            self.same_ctx(b)?;
            if dim >= self.rank() || b.numel() != self.shape[dim] {
                return Err(shape_err(format!(
                    "bias {:?} does not match dim {dim} of {:?}",
                    b.shape, self.shape
                )));
            }
            (self.shape[dim], numel(&self.shape[dim + 1..]))
        } else {
            (1, 1)
        };
        let out = Self::empty(&self.ctx, &self.shape, DType::F32)?;
        elementwise::bias_act(
            &self.ctx,
            self.raw(),
            bias.map(|b| b.raw()),
            out.raw(),
            self.numel(),
            channels,
            inner,
            act,
        )?;
        Ok(out)
    }

    // ---- GEMM ---------------------------------------------------------------------------------------------

    /// Matrix product over the last two dims: `[.., M, K] · [.., K, N] -> [.., M, N]`. Batch dims must match,
    /// or one side may be a plain matrix that broadcasts over the other's batch. f32 or f16 (both the same).
    pub fn matmul(&self, rhs: &GpuTensor) -> Result<Self> {
        self.matmul_t(rhs, false, false)
    }

    /// [`GpuTensor::matmul`] with either operand stored transposed in its last two dims
    /// (`trans_a`: `self` is `[.., K, M]`; `trans_b`: `rhs` is `[.., N, K]`).
    pub fn matmul_t(&self, rhs: &GpuTensor, trans_a: bool, trans_b: bool) -> Result<Self> {
        self.gemm_impl(rhs, trans_a, trans_b, None, GemmBias::None, UnaryOp::Identity)
    }

    /// `act(self · weightᵀ + bias)` for `self: [.., in]`, `weight: [out, in]`, `bias: [out]`.
    pub fn linear(&self, weight: &GpuTensor, bias: Option<&GpuTensor>, act: UnaryOp) -> Result<Self> {
        if self.rank() < 1 || weight.rank() != 2 {
            return Err(shape_err(format!("linear: input {:?}, weight {:?}", self.shape, weight.shape)));
        }
        let rows = numel(&self.shape[..self.rank() - 1]);
        let x = self.reshape(&[rows, self.shape[self.rank() - 1]])?;
        let kind = if bias.is_some() { GemmBias::Col } else { GemmBias::None };
        let y = x.gemm_impl(weight, false, true, bias, kind, act)?;
        let mut shape = self.shape.clone();
        *shape.last_mut().expect("rank >= 1") = weight.shape[0];
        y.reshape(&shape)
    }

    fn gemm_impl(
        &self,
        rhs: &GpuTensor,
        trans_a: bool,
        trans_b: bool,
        bias: Option<&GpuTensor>,
        bias_kind: GemmBias,
        act: UnaryOp,
    ) -> Result<Self> {
        self.same_ctx(rhs)?;
        if self.dtype != rhs.dtype {
            return Err(GpuError::DType { op: "matmul (mixed dtypes)", dtype: rhs.dtype });
        }
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(shape_err(format!(
                "matmul needs rank >= 2, got {:?} and {:?}",
                self.shape, rhs.shape
            )));
        }
        let (ra, rb) = (self.rank(), rhs.rank());
        let (m, ka) = if trans_a {
            (self.shape[ra - 1], self.shape[ra - 2])
        } else {
            (self.shape[ra - 2], self.shape[ra - 1])
        };
        let (kb, n) = if trans_b {
            (rhs.shape[rb - 1], rhs.shape[rb - 2])
        } else {
            (rhs.shape[rb - 2], rhs.shape[rb - 1])
        };
        if ka != kb {
            return Err(shape_err(format!(
                "matmul: inner dims differ ({:?}{} x {:?}{})",
                self.shape,
                if trans_a { "^T" } else { "" },
                rhs.shape,
                if trans_b { "^T" } else { "" }
            )));
        }
        let (bat_a, bat_b) = (&self.shape[..ra - 2], &rhs.shape[..rb - 2]);
        let (na, nb) = (numel(bat_a), numel(bat_b));
        let batch_shape: Vec<usize> = if bat_a == bat_b || nb == 1 {
            bat_a.to_vec()
        } else if na == 1 {
            bat_b.to_vec()
        } else {
            return Err(shape_err(format!("matmul: batch dims {bat_a:?} and {bat_b:?} do not match")));
        };
        if let Some(b) = bias {
            self.same_ctx(b)?;
            let want = if bias_kind == GemmBias::Row { m } else { n };
            if b.numel() != want || b.dtype != self.dtype {
                return Err(shape_err(format!("matmul bias {:?} {:?} does not match", b.shape, b.dtype)));
            }
        }
        let batch = numel(&batch_shape);
        let mut d = GemmDesc::new(m, n, ka, trans_a, trans_b, batch);
        d.a = if na == 1 { BatchStrides::default() } else { BatchStrides::packed(m * ka) };
        d.b = if nb == 1 { BatchStrides::default() } else { BatchStrides::packed(ka * n) };
        d.dtype = self.dtype;
        d.bias = bias_kind;
        d.act = act;
        let mut shape = batch_shape;
        shape.extend([m, n]);
        let out = Self::empty(&self.ctx, &shape, self.dtype)?;
        gemm::gemm(&self.ctx, &d, self.raw(), rhs.raw(), bias.map(|b| b.raw()), out.raw())?;
        Ok(out)
    }

    // ---- reductions ---------------------------------------------------------------------------------------

    /// Reduces dim `dim` (f32). With `keepdim` the reduced dim stays as size 1.
    pub fn reduce(&self, op: ReduceOp, dim: usize, keepdim: bool) -> Result<Self> {
        self.require(DType::F32, "reduce")?;
        let (outer, r, inner) = reduce::split_dim(&self.shape, dim)?;
        let mut shape = self.shape.clone();
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }
        let out = Self::empty(&self.ctx, &shape, DType::F32)?;
        reduce::reduce(&self.ctx, op, self.raw(), out.raw(), outer, r, inner)?;
        Ok(out)
    }

    /// Sum over `dim`.
    pub fn sum(&self, dim: usize, keepdim: bool) -> Result<Self> {
        self.reduce(ReduceOp::Sum, dim, keepdim)
    }

    /// Maximum over `dim`.
    pub fn max(&self, dim: usize, keepdim: bool) -> Result<Self> {
        self.reduce(ReduceOp::Max, dim, keepdim)
    }

    /// Mean over `dim`.
    pub fn mean(&self, dim: usize, keepdim: bool) -> Result<Self> {
        self.reduce(ReduceOp::Mean, dim, keepdim)
    }

    /// Softmax over the last dim (f32).
    pub fn softmax(&self) -> Result<Self> {
        self.softmax_impl(false)
    }

    /// Log-softmax over the last dim (f32).
    pub fn log_softmax(&self) -> Result<Self> {
        self.softmax_impl(true)
    }

    fn softmax_impl(&self, log: bool) -> Result<Self> {
        self.require(DType::F32, "softmax")?;
        let n = *self.shape.last().ok_or_else(|| shape_err("softmax of a scalar"))?;
        let rows = self.numel().checked_div(n).unwrap_or(0);
        let out = Self::empty(&self.ctx, &self.shape, DType::F32)?;
        reduce::softmax(&self.ctx, self.raw(), out.raw(), rows, n, log)?;
        Ok(out)
    }

    // ---- convolution and pooling --------------------------------------------------------------------------

    /// `act(conv2d(self, weight) + bias)` for NCHW `self` and `[C_out, C_in/groups, KH, KW]` weights.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        weight: &GpuTensor,
        bias: Option<&GpuTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        act: UnaryOp,
    ) -> Result<Self> {
        self.same_ctx(weight)?;
        if self.rank() != 4 || weight.rank() != 4 {
            return Err(shape_err(format!("conv2d: input {:?}, weight {:?}", self.shape, weight.shape)));
        }
        if weight.dtype != self.dtype || bias.is_some_and(|b| b.dtype != self.dtype) {
            return Err(GpuError::DType { op: "conv2d (mixed dtypes)", dtype: weight.dtype });
        }
        let [n, c_in, h, w] = [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
        let [c_out, cin_g, kh, kw] = [weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]];
        if groups == 0 || cin_g * groups != c_in {
            return Err(shape_err(format!(
                "conv2d: weight {:?} with {groups} groups for {c_in} input channels",
                weight.shape
            )));
        }
        if let Some(b) = bias {
            self.same_ctx(b)?;
            if b.numel() != c_out {
                return Err(shape_err(format!("conv2d: bias {:?} for {c_out} output channels", b.shape)));
            }
        }
        let d = Conv2dDesc {
            n,
            c_in,
            h,
            w,
            c_out,
            groups,
            win: Window2d { kernel: (kh, kw), stride, padding, dilation },
            act,
            dtype: self.dtype,
        };
        let (oh, ow) = d.output()?;
        let out = Self::empty(&self.ctx, &[n, c_out, oh, ow], self.dtype)?;
        conv::conv2d(&self.ctx, &d, self.raw(), weight.raw(), bias.map(|b| b.raw()), out.raw())?;
        Ok(out)
    }

    /// `max_pool2d` on NCHW (floor mode; padded positions never win, as with -inf padding).
    pub fn max_pool2d(&self, win: Window2d) -> Result<Self> {
        let [n, c, h, w] = self.nchw("max_pool2d")?;
        let (oh, ow) = win.output(h, w)?;
        let out = Self::empty(&self.ctx, &[n, c, oh, ow], self.dtype)?;
        pooling::max_pool2d(&self.ctx, self.raw(), out.raw(), n * c, h, w, &win, self.dtype)?;
        Ok(out)
    }

    /// Nearest-neighbour upsampling of NCHW to `(oh, ow)` (`src = dst * in / out`, as PyTorch `nearest`).
    pub fn upsample_nearest2d(&self, oh: usize, ow: usize) -> Result<Self> {
        let [n, c, h, w] = self.nchw("upsample_nearest2d")?;
        let out = Self::empty(&self.ctx, &[n, c, oh, ow], self.dtype)?;
        layout::upsample_nearest2d(&self.ctx, self.raw(), out.raw(), n * c, h, w, oh, ow, self.dtype)?;
        Ok(out)
    }

    fn nchw(&self, op: &str) -> Result<[usize; 4]> {
        if self.rank() != 4 {
            return Err(shape_err(format!("{op} expects NCHW, got {:?}", self.shape)));
        }
        Ok([self.shape[0], self.shape[1], self.shape[2], self.shape[3]])
    }

    // ---- layout -------------------------------------------------------------------------------------------

    /// Concatenates along `dim` (all other dims equal, same dtype).
    pub fn concat(parts: &[&GpuTensor], dim: usize) -> Result<Self> {
        let first = parts.first().ok_or_else(|| shape_err("concat of nothing"))?;
        if dim >= first.rank() {
            return Err(shape_err(format!("concat dim {dim} out of range for {:?}", first.shape)));
        }
        let mut shape = first.shape.clone();
        shape[dim] = 0;
        for p in parts {
            first.same_ctx(p)?;
            let same_rest = p.rank() == first.rank()
                && p.shape.iter().zip(&first.shape).enumerate().all(|(i, (a, b))| i == dim || a == b);
            if !same_rest || p.dtype != first.dtype {
                return Err(shape_err(format!(
                    "concat: {:?} does not match {:?} outside dim {dim}",
                    p.shape, first.shape
                )));
            }
            shape[dim] += p.shape[dim];
        }
        let outer = numel(&shape[..dim]);
        let inner = numel(&shape[dim + 1..]);
        let out = Self::empty(&first.ctx, &shape, first.dtype)?;
        let mut off = 0;
        for p in parts {
            let row = p.shape[dim] * inner;
            layout::copy_rows(
                &first.ctx,
                p.raw(),
                out.raw(),
                outer,
                row,
                shape[dim] * inner,
                off,
                first.dtype,
            )?;
            off += row;
        }
        Ok(out)
    }

    /// Materialises `self.permute(perm)` densely.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        layout::coalesce_permute(&self.shape, perm)?;
        let shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let out = Self::empty(&self.ctx, &shape, self.dtype)?;
        layout::permute(&self.ctx, self.raw(), out.raw(), &self.shape, perm, self.dtype)?;
        Ok(out)
    }

    /// Swaps two dims (materialised).
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Self> {
        if d0 >= self.rank() || d1 >= self.rank() {
            return Err(shape_err(format!("transpose({d0}, {d1}) of {:?}", self.shape)));
        }
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        perm.swap(d0, d1);
        self.permute(&perm)
    }

    /// Converts to `dtype` (f32 <-> f16; a cheap clone when already `dtype`).
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        let out = Self::empty(&self.ctx, &self.shape, dtype)?;
        layout::cast(&self.ctx, self.raw(), out.raw(), self.numel(), dtype == DType::F16)?;
        Ok(out)
    }
}
