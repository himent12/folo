//! The device-kernel interface and the runtime backend registry.
//!
//! Every tensor op validates its arguments (shapes, dtypes, devices) in folo-core, records autograd, and then
//! calls one method of the [`Backend`] for the tensor's device. The CPU backend ([`crate::cpu::CpuBackend`]) is
//! built in and implements everything. A GPU crate implements the same trait over its own buffers and registers
//! it with [`register_backend`]; folo-core never depends on a GPU API.
//!
//! # Contract for implementors
//! - Inputs arrive as [`View`]s: a storage plus a [`Layout`] (shape, strides, offset). Layouts can be
//!   non-contiguous and can broadcast (stride 0). Unless a method says otherwise, outputs are fresh, contiguous
//!   (row-major, offset 0) storages on the same device as the inputs.
//! - folo-core has already checked shapes, dtypes and devices; backends may assume valid arguments. All compute
//!   ops take `F32` data unless the method says otherwise; index tensors are `I64` or `U32`. A backend whose
//!   [`Backend::computes_f16`] is true also gets `F16` data (every float operand of a call has the same dtype)
//!   in the forward ops folo-core lets through (unary, binary, `where_cond`, reductions, softmax, matmul,
//!   conv2d and the fused conv, pooling, upsampling, `detect_candidates`) and returns results of that dtype;
//!   it may refuse `F16` elsewhere with [`Error::UnsupportedDType`]. Mixed-precision training
//!   ([`crate::autocast`]) also sends `F16` data to the gradients it uses (conv2d input and weight gradients,
//!   `unary_backward`, max-pooling and upsampling backward) and to the fused BatchNorm (`F16` `x` / `grad` with
//!   `F32` parameters and statistics; `y` and `dx` take `x`'s dtype).
//! - Methods under "Compute" have a default implementation returning [`Error::NotSupported`], so a new backend
//!   can start with the memory methods and grow. The user sees a clear "op X is not supported by backend Y"
//!   and can move the tensor to the CPU with [`crate::Tensor::to_device`].
//! - Gradients are built from the same methods (plus the `*_backward` ones), so a backend that implements an op
//!   and its backward supports training through it.

mod types;

pub use types::*;

use std::sync::{Arc, LazyLock, RwLock};

use crate::{CpuStorage, DType, Device, Error, Layout, Result, Shape, Storage};

macro_rules! unsupported {
    ($self:ident, $op:literal) => {
        Err(Error::NotSupported { op: $op, backend: $self.name() })
    };
}

/// Kernels for one kind of device. See the [module docs](self) for the contract.
#[allow(unused_variables)]
pub trait Backend: Send + Sync + 'static {
    /// Short name used in errors (`"cpu"`, `"wgpu"`).
    fn name(&self) -> &'static str;

    /// Block until all work queued on this backend has finished, and report any asynchronous error it hit.
    /// Backends that run synchronously (the CPU) keep the default no-op. Reading data back
    /// ([`Backend::download`]) synchronises implicitly; call this to time device work or to surface errors
    /// early.
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Whether the compute ops accept `F16` data (half-precision inference; see the module docs). The CPU
    /// keeps `F16` for storage only.
    fn computes_f16(&self) -> bool {
        false
    }

    // ----- Memory (required) -----

    /// A contiguous buffer of `len` elements of `dtype`, every element set to `value`.
    fn full(&self, device: Device, dtype: DType, len: usize, value: f64) -> Result<Storage>;
    /// Copy host data to `device`.
    fn upload(&self, device: Device, data: &CpuStorage) -> Result<Storage>;
    /// Copy the elements of `src` (in logical row-major order) to host memory.
    fn download(&self, src: View<'_>) -> Result<CpuStorage>;
    /// A contiguous copy of `src`, same dtype and device.
    fn copy_strided(&self, src: View<'_>) -> Result<Storage>;
    /// Write `src` into the elements of `dst` selected by `dst_layout` (same shape as `src`; never broadcast).
    fn copy_into(&self, dst: &mut Storage, dst_layout: &Layout, src: View<'_>) -> Result<()>;
    /// A contiguous copy of `src` converted to `dtype` (floats to integers truncate toward zero and saturate).
    fn cast(&self, src: View<'_>, dtype: DType) -> Result<Storage>;

    // ----- Compute (optional; default NotSupported) -----

    /// `op(x)` elementwise.
    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        unsupported!(self, "unary")
    }
    /// The input gradient `grad * op'(x)` of a unary op (same shape as `x`).
    fn unary_backward(&self, op: UnaryOp, x: View<'_>, grad: View<'_>) -> Result<Storage> {
        unsupported!(self, "unary_backward")
    }
    /// `op(a, b)` elementwise; `a` and `b` already have the output shape (broadcast dims have stride 0).
    fn binary(&self, op: BinaryOp, a: View<'_>, b: View<'_>) -> Result<Storage> {
        unsupported!(self, "binary")
    }
    /// `cond != 0 ? a : b` elementwise; all three already have the output shape. `cond` is `F32` or `U8`.
    fn where_cond(&self, cond: View<'_>, a: View<'_>, b: View<'_>) -> Result<Storage> {
        unsupported!(self, "where_cond")
    }
    /// Reduce dimension `dim`; the output has `x`'s shape with `dim` set to 1 (`I64` for arg-reductions).
    fn reduce(&self, op: ReduceOp, x: View<'_>, dim: usize) -> Result<Storage> {
        unsupported!(self, "reduce")
    }
    /// Variance over dimension `dim` (biased or Bessel-corrected); the output has `x`'s shape with `dim` set to 1.
    /// The default returns [`Error::NotSupported`] and callers fall back to composite ops.
    fn reduce_var(&self, x: View<'_>, dim: usize, unbiased: bool) -> Result<Storage> {
        unsupported!(self, "reduce_var")
    }
    /// Softmax (or log-softmax if `log`) along `dim`, numerically stable.
    fn softmax(&self, x: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        unsupported!(self, "softmax")
    }
    /// Gradient of [`Backend::softmax`] (`y` is the forward output: probabilities if `!log`, log-probabilities if
    /// `log`). The default returns [`Error::NotSupported`] and callers fall back to composite ops.
    fn softmax_backward(&self, y: View<'_>, grad: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        unsupported!(self, "softmax_backward")
    }
    /// Layer normalisation over the last dimension `d` of `x` (`[.., d]`): `y = ((x - mean) * rstd) * weight + bias`
    /// with biased variance and `rstd = 1 / sqrt(var + eps)`. Returns `(y, mean [outer], rstd [outer])`.
    /// Optional; callers fall back to composite ops on [`Error::NotSupported`].
    fn layer_norm(
        &self,
        x: View<'_>,
        weight: Option<View<'_>>,
        bias: Option<View<'_>>,
        eps: f32,
    ) -> Result<(Storage, Storage, Storage)> {
        unsupported!(self, "layer_norm")
    }
    /// Gradient of [`Backend::layer_norm`]: returns `(dx, dweight, dbias)`. `dweight` / `dbias` are computed only
    /// when `need_weight` / `need_bias` are set.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        mean: View<'_>,
        rstd: View<'_>,
        need_weight: bool,
        need_bias: bool,
    ) -> Result<(Storage, Option<Storage>, Option<Storage>)> {
        unsupported!(self, "layer_norm_backward")
    }
    /// RMS normalisation over the last dimension `d` of `x` (`[.., d]`): `y = (x * rstd) * weight` with
    /// `rstd = 1 / sqrt(mean(x^2) + eps)`. Returns `(y, rstd [outer])`.
    /// Optional; callers fall back to composite ops on [`Error::NotSupported`].
    fn rms_norm(&self, x: View<'_>, weight: Option<View<'_>>, eps: f32) -> Result<(Storage, Storage)> {
        unsupported!(self, "rms_norm")
    }
    /// Gradient of [`Backend::rms_norm`]: returns `(dx, dweight)`. `dweight` is computed only when `need_weight`
    /// is set.
    fn rms_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        rstd: View<'_>,
        need_weight: bool,
    ) -> Result<(Storage, Option<Storage>)> {
        unsupported!(self, "rms_norm_backward")
    }
    /// Batched matrix product: `a` is `[B.., m, k]`, `b` is `[B.., k, n]` with identical (already broadcast)
    /// batch dims; the output is `[B.., m, n]`.
    fn matmul(&self, a: View<'_>, b: View<'_>) -> Result<Storage> {
        unsupported!(self, "matmul")
    }
    /// 2-D convolution of `x` `[N, C, H, W]` with `w` `[O, C/groups, KH, KW]`, plus `bias` `[O]` if given.
    fn conv2d(&self, x: View<'_>, w: View<'_>, bias: Option<View<'_>>, p: &Conv2dParams) -> Result<Storage> {
        unsupported!(self, "conv2d")
    }
    /// Gradient of conv2d with respect to its input (of shape `input_shape`).
    fn conv2d_backward_input(
        &self,
        grad: View<'_>,
        w: View<'_>,
        input_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_backward_input")
    }
    /// Gradient of conv2d with respect to its weight (of shape `weight_shape`).
    fn conv2d_backward_weight(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_backward_weight")
    }
    /// [`Backend::conv2d_backward_weight`] with the result stored as `out` (`F32`, or the operands' dtype): mixed
    /// precision writes an `F16` conv's weight gradient straight to its `F32` master weight. The default computes
    /// in the operands' dtype and casts.
    fn conv2d_backward_weight_as(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
        out: DType,
    ) -> Result<Storage> {
        let s = self.conv2d_backward_weight(x, grad, weight_shape, p)?;
        if s.dtype() == out {
            return Ok(s);
        }
        let whole = Layout::contiguous(weight_shape.clone());
        self.cast(View::new(&s, &whole), out)
    }
    /// Max pooling of `[N, C, H, W]`: returns the values and, per output element, the flat index `h * W + w`
    /// of the chosen input element within its plane (`I64`).
    fn max_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<(Storage, Storage)> {
        unsupported!(self, "max_pool2d")
    }
    /// [`Backend::max_pool2d`]'s values alone, bit for bit, for a result nobody differentiates (no grad mode, or
    /// an input that does not require grad). An implementation may skip making the indices (the CPU backend
    /// does); the default runs `max_pool2d` and drops them.
    fn max_pool2d_values(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        self.max_pool2d(x, p).map(|(v, _)| v)
    }
    /// Scatter `grad` back to the positions in `indices` (from [`Backend::max_pool2d`] with window `p`),
    /// summing collisions. `p` lets a backend gather instead of scatter (each input element visits only the
    /// windows that cover it); the CPU backend scatters and ignores it.
    fn max_pool2d_backward(
        &self,
        grad: View<'_>,
        indices: View<'_>,
        input_shape: &Shape,
        p: &Pool2dParams,
    ) -> Result<Storage> {
        unsupported!(self, "max_pool2d_backward")
    }
    /// Average pooling (padding zeros count toward the divisor).
    fn avg_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        unsupported!(self, "avg_pool2d")
    }
    /// Gradient of [`Backend::avg_pool2d`].
    fn avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape, p: &Pool2dParams) -> Result<Storage> {
        unsupported!(self, "avg_pool2d_backward")
    }
    /// Adaptive average pooling to `[oh, ow]` (PyTorch's window rule: `floor(i*H/oh)..ceil((i+1)*H/oh)`).
    fn adaptive_avg_pool2d(&self, x: View<'_>, out_hw: [usize; 2]) -> Result<Storage> {
        unsupported!(self, "adaptive_avg_pool2d")
    }
    /// Gradient of [`Backend::adaptive_avg_pool2d`].
    fn adaptive_avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape) -> Result<Storage> {
        unsupported!(self, "adaptive_avg_pool2d_backward")
    }
    /// Nearest-neighbour upsampling of `[N, C, H, W]` by integer factors `[sh, sw]`.
    fn upsample_nearest2d(&self, x: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        unsupported!(self, "upsample_nearest2d")
    }
    /// Gradient of [`Backend::upsample_nearest2d`] (sums each `sh x sw` block).
    fn upsample_nearest2d_backward(&self, grad: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        unsupported!(self, "upsample_nearest2d_backward")
    }
    /// Select entries `idx` (1-D) along `dim`. Works for every dtype of `x`.
    fn index_select(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        unsupported!(self, "index_select")
    }
    /// A copy of `x` with `src` slices added at positions `idx` (1-D) along `dim` (duplicates accumulate).
    fn index_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        unsupported!(self, "index_add")
    }
    /// `out[.., i, ..] = x[.., idx[.., i, ..], ..]` along `dim`; the output has `idx`'s shape.
    fn gather(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        unsupported!(self, "gather")
    }
    /// A copy of `x` with `out[.., idx[.., i, ..], ..] += src[.., i, ..]` along `dim`.
    fn scatter_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        unsupported!(self, "scatter_add")
    }
    /// Gradient of [`Backend::index_select`]: zeros of `input_shape` with the slices of `grad` added at `idx`
    /// along `dim` (duplicates accumulate). The default composes [`Backend::full`] and [`Backend::index_add`];
    /// the CPU backend scatters straight into a zeroed buffer.
    fn index_select_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let device = grad.storage.device();
        let zeros = self.full(device, DType::F32, input_shape.numel(), 0.0)?;
        let layout = Layout::contiguous(input_shape.clone());
        self.index_add(View::new(&zeros, &layout), dim, idx, grad)
    }
    /// Gradient of [`Backend::gather`]: zeros of `input_shape` with `out[.., idx[.., i, ..], ..] += grad[.., i, ..]`
    /// along `dim`. The default composes [`Backend::full`] and [`Backend::scatter_add`].
    fn gather_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let device = grad.storage.device();
        let zeros = self.full(device, DType::F32, input_shape.numel(), 0.0)?;
        let layout = Layout::contiguous(input_shape.clone());
        self.scatter_add(View::new(&zeros, &layout), dim, idx, grad)
    }
    /// The `k` largest (or smallest) entries along the last dim, sorted, as `(values, I64 indices)`.
    fn topk(&self, x: View<'_>, k: usize, largest: bool) -> Result<(Storage, Storage)> {
        unsupported!(self, "topk")
    }

    // ----- Fused (optional; default NotSupported, and callers fall back to the composite ops) -----

    /// `act(conv2d(x, w) + bias)` in one kernel (the conv epilogue applies bias and activation). `act` is
    /// `None` for the identity. Same shapes as [`Backend::conv2d`]; no gradient is ever needed (callers use it
    /// only when nothing is recorded). Reached through `Tensor::conv2d_act`.
    fn conv2d_act(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_act")
    }
    /// [`Backend::conv2d_act`] over the channel concatenation of `xs` (each `[N, C_i, H, W]` with the same `N`,
    /// `H`, `W`; `w` has `ΣC_i / groups` input channels), without writing the concatenation. Reached through
    /// `Tensor::conv2d_act_cat`, which falls back to `cat` then [`Backend::conv2d_act`].
    fn conv2d_act_cat(
        &self,
        xs: &[View<'_>],
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_act_cat")
    }
    /// [`Backend::conv2d_act`] where only the output sites `at` are needed (`U32 [S]`, flat output indices
    /// `n · OH·OW + oy · OW + ox`, each `< N·OH·OW`, any order, repeats allowed): each site depends only on its
    /// kernel's own receptive field in `x`, and every other element of the `[N, O, OH, OW]` output is unspecified.
    /// Reached through `Tensor::conv2d_act_at`, which falls back to the dense [`Backend::conv2d_act`].
    fn conv2d_act_at(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        at: View<'_>,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_act_at")
    }
    /// Like [`Backend::conv2d_act_at`], but permits tiled algorithms (such as CPU Winograd F(4×4, 3×3)) that
    /// combine inputs across each requested 4×4 tile's full 6×6 patch to match dense [`Backend::conv2d_act`] bit
    /// for bit at every site of `at`. Defaults to [`Backend::conv2d_act_at`].
    fn conv2d_act_at_tiles(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        at: View<'_>,
    ) -> Result<Storage> {
        self.conv2d_act_at(x, w, bias, p, act, at)
    }
    /// [`Backend::conv2d_act`] at listed output pixels: `sites` `U32 [N, 1 + cap]` holds per image a count, then
    /// up to `cap` pixels `oy·OW + ox` (as [`Backend::detect_class_sites`] lists them). Returns `[N, O, cap]`:
    /// column `j < min(count, cap)` of image `n` is, bit for bit, the value [`Backend::conv2d_act`] computes on
    /// this batch at the image's `j`-th pixel; with `to_map`, the full `[N, O, OH, OW]` output with those sites
    /// written instead. Everything else is unspecified. Reached through `Tensor::conv2d_act_sites`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_act_sites(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        sites: View<'_>,
        to_map: bool,
    ) -> Result<Storage> {
        unsupported!(self, "conv2d_act_sites")
    }
    /// Train-mode batch normalisation of `x` `[N, C, H, W]` over `N·H·W` per channel, then `act` (`None` is
    /// the identity): `y = act((x − mean) · invstd · weight + bias)` with the batch mean and
    /// `invstd = 1 / sqrt(var + eps)` of the *biased* variance. Returns `(y, mean [C], invstd [C])`.
    /// With `running`, also updates the running statistics in place:
    /// `mean ← (1 − m) · mean + m · batch_mean`, `var ← (1 − m) · var + m · batch_var · n / (n − 1)`. Their
    /// storages hold `[C]` contiguously from element 0. `x` may be `F16` on a backend that computes in it (`y` is
    /// then `F16`); `weight`, `bias`, the statistics and the running statistics are always `F32`. Reached through
    /// `Tensor::batch_norm_act_train`.
    fn batch_norm_act(
        &self,
        x: View<'_>,
        weight: View<'_>,
        bias: View<'_>,
        eps: f32,
        act: Option<UnaryOp>,
        running: Option<RunningUpdate<'_>>,
    ) -> Result<(Storage, Storage, Storage)> {
        unsupported!(self, "batch_norm_act")
    }
    /// The gradients `(dx, dweight, dbias)` of [`Backend::batch_norm_act`] given the output gradient `grad`
    /// and the forward's `mean` / `invstd`; `act` is differentiated at its recomputed input. `x` and `grad` share
    /// a dtype (`F32`, or `F16`), which `dx` takes; `dweight` and `dbias` are `F32`.
    #[allow(clippy::too_many_arguments)]
    fn batch_norm_act_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: View<'_>,
        bias: View<'_>,
        mean: View<'_>,
        invstd: View<'_>,
        act: Option<UnaryOp>,
    ) -> Result<(Storage, Storage, Storage)> {
        unsupported!(self, "batch_norm_act_backward")
    }
    /// One in-place optimizer update of `param` from `grad` (see [`OptimStep`] for the formulas, which are
    /// folo-nn's optimizers' and EMA's, op for op). `param` and `state` hold their `n = grad.numel()` elements
    /// contiguously from element 0; `state` is the momentum buffer (SGD with momentum), `[exp_avg,
    /// exp_avg_sq]` (AdamW) or empty. Reached through `Tensor::optimizer_step_`.
    fn optimizer_step(
        &self,
        param: &mut Storage,
        grad: View<'_>,
        state: &mut [&mut Storage],
        step: &OptimStep,
    ) -> Result<()> {
        unsupported!(self, "optimizer_step")
    }

    /// [`Backend::optimizer_step`] for many parameters with the same update, in as few kernels as the backend
    /// can (the GPU: one per 16 tensors). Every item holds what one `optimizer_step` call takes; no written
    /// storage appears twice and no gradient views a written one (gradients may view one storage). Reached through
    /// `Tensor::optimizer_step_many_`.
    fn optimizer_step_many(&self, items: &mut [OptimItem<'_>], step: &OptimStep) -> Result<()> {
        unsupported!(self, "optimizer_step_many")
    }

    /// Mixed precision (the loss scaler's unscale and overflow check): `x *= factor` in place over its first `n`
    /// `F32` elements and, if any result is not finite, set the single `U32` element of `flag` to a nonzero value
    /// (it is never cleared here). Reached through `Tensor::unscale_check_`.
    fn unscale_check(&self, x: &mut Storage, n: usize, factor: f32, flag: &mut Storage) -> Result<()> {
        unsupported!(self, "unscale_check")
    }

    /// [`Backend::unscale_check`] for many tensors (`xs[i]` over its first `ns[i]` elements) into one flag, in as
    /// few kernels as the backend can. Reached through `Tensor::unscale_check_many_`.
    fn unscale_check_many(
        &self,
        xs: &mut [&mut Storage],
        ns: &[usize],
        factor: f32,
        flag: &mut Storage,
    ) -> Result<()> {
        unsupported!(self, "unscale_check_many")
    }

    /// Global-norm clipping's reductions over many `F32` tensors (each dense from element 0): one partial per 256
    /// elements of each tensor, in order (`sum of ceil(numel / 256)` of them, `F32`), each the largest `|x|`
    /// (NaN propagating) or the sum of `((x * h) * h)^2` over those elements, reduced in a fixed order. `h` is a
    /// one-element `F32` view, required for [`NormReduce::ScaledSquares`]. The caller reduces the partials.
    /// Reached through `Tensor::norm_partials_many`.
    fn norm_partials_many(&self, xs: &[View<'_>], op: NormReduce, h: Option<View<'_>>) -> Result<Storage> {
        unsupported!(self, "norm_partials_many")
    }

    /// `x = (x * f1) * f2` in place over the first `ns[i]` `F32` elements of each `xs[i]`, with one-element `F32`
    /// factors on the device: the roundings of `x *= f1` then `x *= f2`, in as few kernels as the backend can.
    /// Reached through `Tensor::scale_many_`.
    fn scale_many(&self, xs: &mut [&mut Storage], ns: &[usize], f1: View<'_>, f2: View<'_>) -> Result<()> {
        unsupported!(self, "scale_many")
    }

    /// MuSGD's Muon update of many matrices in one call (folo-nn's `MuSgd`): for each item,
    /// `u = nesterov ? (g · (1 − μ)) · 1 + m · μ : m` (`rows × cols`), orthogonalised by `settings` (`steps`
    /// odd-quintic steps with `step_coeffs` and optional step-0 Schatten-4 rescaling), scaled by `scale` (when
    /// not 1), then `param = param · 1 + O · (−lr)` in place. Parameters hold `rows · cols` elements from 0 and
    /// do not alias. `Error::NotSupported`, writing nothing, declines the call so the caller falls back.
    fn muon_step_many(&self, items: &mut [MuonItem<'_>], settings: &MuonSettings) -> Result<()> {
        unsupported!(self, "muon_step_many")
    }

    // ----- Detector pre- and post-processing (optional; callers fall back to host code) -----

    /// Letterbox `U8` RGB images (`images[i]` is `[H_i, W_i, 3]`) onto one `[B, 3, OH, OW]` canvas of `dtype`
    /// (`F32`, or `F16` when [`Backend::computes_f16`]; the values are formed in `f32` and rounded once)
    /// (`out_hw = [OH, OW]`), scaled by `scale` (`1 / 255` for network input): every pixel is `pad_value`
    /// except the resized image at `geoms[i].pad`, which is sampled exactly as folo-data's bilinear resize
    /// (integer box prefilter, then 11-bit fixed-point bilinear taps; see [`LetterboxImage`]). `taps` is `U32`.
    /// Reached through `Tensor::letterbox_rgb8`.
    #[allow(clippy::too_many_arguments)]
    fn letterbox_rgb8(
        &self,
        images: &[View<'_>],
        taps: View<'_>,
        geoms: &[LetterboxImage],
        out_hw: [usize; 2],
        pad_value: u8,
        scale: f32,
        dtype: DType,
    ) -> Result<Storage> {
        unsupported!(self, "letterbox_rgb8")
    }
    /// Decode an anchor-free detection head and keep only the candidates: for every image `b` and anchor `a`,
    /// the best class `c` (the first maximum of the `nc` sigmoid scores) when its score is `> conf`, with the
    /// anchor's box (DFL softmax expectation of each side's `reg_max` bins, `ltrb` around the anchor point,
    /// times the stride). `box_raw` is `F32 [B, 4·reg_max, A]` (or `F16`, with `cls_raw` alike, when
    /// [`Backend::computes_f16`]; the decode runs in `f32`), `cls_raw` `F32 [B, nc, A]`, `anchors` `F32
    /// [3, A]` (point x, point y in grid units, stride). Returns `U32 [B, 1 + 7·A]`: per image the candidate
    /// count `n`, then `n` records `(a, c, score, x1, y1, x2, y2)` (floats as their bits) in unspecified order.
    /// Reached through `Tensor::detect_candidates`.
    fn detect_candidates(
        &self,
        box_raw: View<'_>,
        cls_raw: View<'_>,
        anchors: View<'_>,
        p: &DetectDecode,
    ) -> Result<Storage> {
        unsupported!(self, "detect_candidates")
    }
    /// The class pass of one head level for candidate-first prediction (docs/MODEL.md §9.5): of `cls` (`[B, nc,
    /// H, W]` logits, `F32` or `F16`) every pixel whose best class score (first maximum of the `nc` sigmoids) is
    /// `> p.conf`. Returns three `U32` tensors: `sites [B, 1 + cap]` (per image the count of all candidates, then
    /// the pixels `y·W + x` of the first `cap`, in unspecified order), `recs [B, 2·cap]` (class and score bits per
    /// slot) and `field [B, 1 + fcap]` (the pixels of the slotted candidates' 3×3 windows inside the map, each
    /// once, counted likewise; `fcap = min(9·cap, H·W)`, so it never overflows). Reached through
    /// `Tensor::detect_class_sites`.
    fn detect_class_sites(&self, cls: View<'_>, p: &ClassSites) -> Result<(Storage, Storage, Storage)> {
        unsupported!(self, "detect_class_sites")
    }
    /// The boxes of every level's candidates (`levels`, from [`Backend::detect_class_sites`] with each level's box
    /// logits at the slots, `[B, 4·reg_max, cap]`): `U32 [B, 2 + 7·Σ cap]`, per image the record count, a word
    /// with bit `l` set when level `l` had more candidates than slots, then the records `(a, c, score, x1, y1,
    /// x2, y2)` of [`Backend::detect_candidates`] (the decode in `f32`), in unspecified order. Reached through
    /// `Tensor::detect_decode_sites`.
    fn detect_decode_sites(&self, levels: &[SiteLevel<'_>], reg_max: usize) -> Result<Storage> {
        unsupported!(self, "detect_decode_sites")
    }
}

static GPU_BACKENDS: RwLock<Vec<Option<Arc<dyn Backend>>>> = RwLock::new(Vec::new());
static CPU: LazyLock<Arc<dyn Backend>> = LazyLock::new(|| Arc::new(crate::cpu::CpuBackend));

/// Register the backend that serves `device` (a [`Device::Gpu`] ordinal). Replaces any previous one.
///
/// The CPU backend is built in and cannot be replaced.
pub fn register_backend(device: Device, backend: Arc<dyn Backend>) -> Result<()> {
    match device {
        Device::Cpu => Err(Error::invalid_arg("register_backend", "the CPU backend is built in")),
        Device::Gpu(i) => {
            let mut reg = GPU_BACKENDS.write().unwrap_or_else(|e| e.into_inner());
            if reg.len() <= i {
                reg.resize(i + 1, None);
            }
            reg[i] = Some(backend);
            Ok(())
        }
    }
}

/// The backend serving `device`.
pub fn backend(device: Device) -> Result<Arc<dyn Backend>> {
    match device {
        Device::Cpu => Ok(CPU.clone()),
        Device::Gpu(i) => GPU_BACKENDS
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(i)
            .cloned()
            .flatten()
            .ok_or(Error::BackendNotRegistered(device)),
    }
}

/// Whether `device`'s backend computes in `F16` ([`Backend::computes_f16`]); `false` for an unregistered device.
pub fn computes_f16(device: Device) -> bool {
    backend(device).is_ok_and(|b| b.computes_f16())
}

/// Wait for all queued work on `device` to finish (see [`Backend::synchronize`]).
pub fn synchronize(device: Device) -> Result<()> {
    backend(device)?.synchronize()
}

/// Every device with a backend: `Cpu` first, then registered GPUs.
pub fn available_devices() -> Vec<Device> {
    let reg = GPU_BACKENDS.read().unwrap_or_else(|e| e.into_inner());
    std::iter::once(Device::Cpu)
        .chain(reg.iter().enumerate().filter(|(_, b)| b.is_some()).map(|(i, _)| Device::Gpu(i)))
        .collect()
}
