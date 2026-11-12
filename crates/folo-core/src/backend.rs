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

use std::sync::{Arc, LazyLock, RwLock};

use crate::storage::Derived;
use crate::{CpuStorage, DType, Device, Error, Layout, Result, Shape, Storage};

/// A storage plus the layout to read it with.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    /// The buffer.
    pub storage: &'a Storage,
    /// How to read it.
    pub layout: &'a Layout,
    /// For a view of a tensor: the storage's derived values, and the storage they belong to (`storage` is a
    /// public field, so it may since have been pointed elsewhere).
    derived: Option<(&'a Derived, &'a Storage)>,
}

impl<'a> View<'a> {
    /// Pair a storage with a layout.
    pub fn new(storage: &'a Storage, layout: &'a Layout) -> Self {
        View { storage, layout, derived: None }
    }

    /// A view of a tensor's storage, with the storage's derived values (see [`View::derived`]).
    pub(crate) fn of_cell(storage: &'a Storage, layout: &'a Layout, derived: &'a Derived) -> Self {
        View { storage, layout, derived: Some((derived, storage)) }
    }

    /// A value derived from this view's data, kept with the storage until its data is next written, under its
    /// type and `key` (which must name what the value is and everything it depends on besides the data, such
    /// as the layout); made by `make` on a miss. A view without derived values of its storage ([`View::new`],
    /// or one whose `storage` was replaced after [`View::of_cell`]) makes it every time and keeps nothing.
    pub(crate) fn derived<T: std::any::Any + Send + Sync>(
        &self,
        key: u64,
        make: impl FnOnce() -> T,
    ) -> Arc<T> {
        match self.derived {
            Some((d, owner)) if std::ptr::eq(self.storage, owner) => d.get_or_make(key, make),
            _ => Arc::new(make()),
        }
    }

    /// The logical shape.
    pub fn shape(&self) -> &'a Shape {
        self.layout.shape()
    }
}

/// Elementwise ops on one `F32` tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Recip,
    Square,
    Sin,
    Cos,
    Tanh,
    /// Arctangent, in `(-pi/2, pi/2)`.
    Atan,
    /// Arcsine of `x` in `[-1, 1]`.
    Asin,
    /// Arccosine of `x` in `[-1, 1]`.
    Acos,
    /// The error function `2/sqrt(pi) * integral_0^x e^(-t^2) dt`.
    Erf,
    Sigmoid,
    /// `x * sigmoid(x)`.
    Silu,
    Relu,
    /// `x` for `x > 0`, else `slope * x`.
    LeakyRelu(f32),
    /// `x * relu6(x + 3) / 6`.
    Hardswish,
    /// `ln(1 + e^x)`, computed stably.
    Softplus,
    /// `min(max(x, min), max)`.
    Clamp(f32, f32),
    /// `x^p` for a scalar `p`.
    PowScalar(f32),
    /// `x + c`.
    AddScalar(f32),
    /// `x * c`.
    MulScalar(f32),
    /// `c - x`.
    RSubScalar(f32),
    /// `c / x`.
    RDivScalar(f32),
    /// `x / c`.
    DivScalar(f32),
    /// `max(x, c)`.
    MaxScalar(f32),
    /// `min(x, c)`.
    MinScalar(f32),
    Floor,
    Ceil,
    Round,
    /// -1, 0 or 1.
    Sign,
}

impl UnaryOp {
    /// Stable op name used in errors and graph dumps.
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Rsqrt => "rsqrt",
            UnaryOp::Recip => "recip",
            UnaryOp::Square => "square",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Atan => "atan",
            UnaryOp::Asin => "asin",
            UnaryOp::Acos => "acos",
            UnaryOp::Erf => "erf",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Silu => "silu",
            UnaryOp::Relu => "relu",
            UnaryOp::LeakyRelu(_) => "leaky_relu",
            UnaryOp::Hardswish => "hardswish",
            UnaryOp::Softplus => "softplus",
            UnaryOp::Clamp(..) => "clamp",
            UnaryOp::PowScalar(_) => "pow_scalar",
            UnaryOp::AddScalar(_) => "add_scalar",
            UnaryOp::MulScalar(_) => "mul_scalar",
            UnaryOp::RSubScalar(_) => "rsub_scalar",
            UnaryOp::RDivScalar(_) => "rdiv_scalar",
            UnaryOp::DivScalar(_) => "div_scalar",
            UnaryOp::MaxScalar(_) => "max_scalar",
            UnaryOp::MinScalar(_) => "min_scalar",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Round => "round",
            UnaryOp::Sign => "sign",
        }
    }
}

/// Elementwise ops on two `F32` tensors of the same (already broadcast) shape. Comparisons produce `F32`
/// masks: 1.0 where true, 0.0 where false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    /// `atan2(a, b)`: the angle of the point `(b, a)` (`a` is y, `b` is x), in `[-pi, pi]`.
    Atan2,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl BinaryOp {
    /// Stable op name used in errors and graph dumps.
    pub fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
            BinaryOp::Atan2 => "atan2",
            BinaryOp::Eq => "eq",
            BinaryOp::Ne => "ne",
            BinaryOp::Gt => "gt",
            BinaryOp::Ge => "ge",
            BinaryOp::Lt => "lt",
            BinaryOp::Le => "le",
        }
    }

    /// True for comparisons (results are 0/1 masks without gradient).
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le
        )
    }
}

/// Reductions over one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    /// Index of the first maximum, as `I64`.
    ArgMax,
    /// Index of the first minimum, as `I64`.
    ArgMin,
}

/// Geometry of a 2-D convolution (NCHW input, OIHW weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv2dParams {
    /// Stride `[h, w]`.
    pub stride: [usize; 2],
    /// Zero padding `[h, w]` on both sides.
    pub padding: [usize; 2],
    /// Dilation `[h, w]`.
    pub dilation: [usize; 2],
    /// Number of groups; `C_in` and `C_out` are divisible by it.
    pub groups: usize,
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 }
    }
}

impl Conv2dParams {
    /// Output spatial size for an input of `[h, w]` and kernel `[kh, kw]`, or `None` if it would be empty or
    /// its arithmetic overflows `usize`.
    pub fn output_hw(&self, h: usize, w: usize, kh: usize, kw: usize) -> Option<[usize; 2]> {
        let f = |i: usize, k: usize, s: usize, p: usize, d: usize| {
            let eff = d.checked_mul(k.checked_sub(1)?)?.checked_add(1)?;
            i.checked_add(p.checked_mul(2)?)?.checked_sub(eff).map(|v| v / s + 1)
        };
        Some([
            f(h, kh, self.stride[0], self.padding[0], self.dilation[0])?,
            f(w, kw, self.stride[1], self.padding[1], self.dilation[1])?,
        ])
    }
}

/// Geometry of a 2-D pooling window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pool2dParams {
    /// Window `[h, w]`.
    pub kernel: [usize; 2],
    /// Stride `[h, w]`.
    pub stride: [usize; 2],
    /// Implicit padding `[h, w]` on both sides (`-inf` for max pooling, zeros counted for average pooling).
    pub padding: [usize; 2],
}

impl Pool2dParams {
    /// Output spatial size for an input of `[h, w]`, or `None` if it would be empty or its arithmetic overflows
    /// `usize`.
    pub fn output_hw(&self, h: usize, w: usize) -> Option<[usize; 2]> {
        let f = |i: usize, k: usize, s: usize, p: usize| {
            i.checked_add(p.checked_mul(2)?)?.checked_sub(k).map(|v| v / s + 1)
        };
        Some([
            f(h, self.kernel[0], self.stride[0], self.padding[0])?,
            f(w, self.kernel[1], self.stride[1], self.padding[1])?,
        ])
    }
}

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
    /// Softmax (or log-softmax if `log`) along `dim`, numerically stable.
    fn softmax(&self, x: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        unsupported!(self, "softmax")
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

    /// MuSGD's Muon update of many matrices in one call (folo-nn's `MuSgd`): for each item, the matrix
    /// `u = nesterov ? (g · (1 − μ)) · 1 + m · μ : m` (`rows × cols`, from its momentum buffer and gradient),
    /// orthogonalised as `folo_nn::optim::orthogonalize` does it (divided by its Frobenius norm plus `eps`,
    /// transposed so that rows ≤ cols, `steps` quintic Newton–Schulz steps with `coeffs`, transposed back), times
    /// its `scale` (when not 1), then `param = param · 1 + O · (−lr)` in place. The parameters hold their
    /// `rows · cols` elements contiguously from element 0 and are no other item's storage. Reached through
    /// `Tensor::muon_step_many_`. `Error::NotSupported`, with no parameter written, declines the call (also where
    /// a backend cannot take these matrices on this device), and the caller uses tensor ops.
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
}

/// One image of [`Backend::letterbox_rgb8`]. The sampling is folo-data's `ResizePlan` for a `src` →
/// `resized` resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LetterboxImage {
    /// Source `[width, height]`.
    pub src: [usize; 2],
    /// Size `[width, height]` of the resized image inside the canvas.
    pub resized: [usize; 2],
    /// Its top-left corner `[x, y]` in the canvas.
    pub pad: [usize; 2],
    /// Box-prefilter factors `[fx, fy]` (`[1, 1]`: none).
    pub prefilter: [usize; 2],
    /// Offset in `taps` of this image's six tap arrays: `x0, x1, wx` (`resized[0]` entries each), then
    /// `y0, y1, wy` (`resized[1]` each). Indices are prefiltered pixels; weights are in 1/2048.
    pub taps: usize,
}

/// Settings of [`Backend::detect_candidates`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectDecode {
    /// DFL bins per box side.
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Keep a candidate when its score is `> conf`.
    pub conf: f32,
}

/// Words per record of [`Backend::detect_candidates`]: anchor, class, score, x1, y1, x2, y2.
pub const DETECT_RECORD_WORDS: usize = 7;

/// The update [`Backend::optimizer_step`] applies, per element, in `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OptimStep {
    /// SGD: `d = g + wd · p` (when `wd ≠ 0`); with a momentum buffer `b = μ · b + d` and
    /// `d = nesterov ? d + μ · b : b`; then `p = p + (−lr) · d`.
    Sgd {
        /// Learning rate.
        lr: f32,
        /// `μ` (used only with a momentum buffer).
        momentum: f32,
        /// Coupled L2 weight decay.
        weight_decay: f32,
        /// Nesterov momentum.
        nesterov: bool,
    },
    /// AdamW: `p = p · decay` (when set); `m = β1 · m + (1 − β1) · g`; `v = β2 · v + (1 − β2) · g²`;
    /// `p = p + (−step_size) · m / (sqrt(v) / bc2_sqrt + eps)`.
    AdamW {
        /// Decoupled weight decay factor `1 − lr · wd`.
        decay: Option<f32>,
        /// β1.
        beta1: f32,
        /// `1 − β1`.
        one_minus_beta1: f32,
        /// β2.
        beta2: f32,
        /// `1 − β2`.
        one_minus_beta2: f32,
        /// `lr / (1 − β1^t)`.
        step_size: f32,
        /// `sqrt(1 − β2^t)`.
        bc2_sqrt: f32,
        /// ε.
        eps: f32,
    },
    /// `p = p · a + g · b` (the EMA update with `a = d`, `b = 1 − d`).
    Axpby {
        /// Weight of the old value.
        a: f32,
        /// Weight of the new value.
        b: f32,
    },
}

/// What [`Backend::norm_partials_many`] reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormReduce {
    /// The largest `|x|`, NaN propagating.
    AbsMax,
    /// The sum of `((x * h) * h)^2`.
    ScaledSquares,
}

/// One parameter of [`Backend::optimizer_step_many`]: the arguments of one [`Backend::optimizer_step`] call.
#[derive(Debug)]
pub struct OptimItem<'a> {
    /// The parameter (`grad.numel()` elements, contiguous from element 0).
    pub param: &'a mut Storage,
    /// Its gradient.
    pub grad: View<'a>,
    /// Its state buffers, as for [`Backend::optimizer_step`].
    pub state: Vec<&'a mut Storage>,
}

/// One matrix of [`Backend::muon_step_many`].
#[derive(Debug)]
pub struct MuonItem<'a> {
    /// The parameter, updated in place.
    pub param: &'a mut Storage,
    /// The Muon momentum buffer (`rows · cols` `F32` elements, already updated this step).
    pub momentum: View<'a>,
    /// The gradient (`rows · cols` `F32` elements).
    pub grad: View<'a>,
    /// Rows of the matrix (the parameter's first dimension).
    pub rows: usize,
    /// Columns (the product of its other dimensions).
    pub cols: usize,
    /// Factor of the orthogonalised direction.
    pub scale: f32,
}

/// The settings of [`Backend::muon_step_many`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuonSettings {
    /// `μ`.
    pub momentum: f32,
    /// Nesterov: `u = (g · (1 − μ)) · 1 + m · μ`; else `u = m`.
    pub nesterov: bool,
    /// The Newton–Schulz coefficients `(a, b, c)`: `X ← a·X + (b·A + c·A·A)·X`, `A = X·Xᵀ`.
    pub coeffs: (f32, f32, f32),
    /// Newton–Schulz steps.
    pub steps: usize,
    /// Added to the Frobenius norm before the division.
    pub eps: f32,
    /// The learning rate of the update (MuSGD's `0.2 · lr`).
    pub lr: f32,
}

/// The running statistics [`Backend::batch_norm_act`] updates in place.
#[derive(Debug)]
pub struct RunningUpdate<'a> {
    /// Running mean, `[C]` contiguous.
    pub mean: &'a mut Storage,
    /// Running (unbiased) variance, `[C]` contiguous.
    pub var: &'a mut Storage,
    /// Weight of the batch statistic.
    pub momentum: f32,
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
