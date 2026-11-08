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
//!   ops take `F32` data unless the method says otherwise; index tensors are `I64` or `U32`.
//! - Methods under "Compute" have a default implementation returning [`Error::NotSupported`], so a new backend
//!   can start with the memory methods and grow. The user sees a clear "op X is not supported by backend Y"
//!   and can move the tensor to the CPU with [`crate::Tensor::to_device`].
//! - Gradients are built from the same methods (plus the `*_backward` ones), so a backend that implements an op
//!   and its backward supports training through it.

use std::sync::{Arc, LazyLock, RwLock};

use crate::{CpuStorage, DType, Device, Error, Layout, Result, Shape, Storage};

/// A storage plus the layout to read it with.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    /// The buffer.
    pub storage: &'a Storage,
    /// How to read it.
    pub layout: &'a Layout,
}

impl<'a> View<'a> {
    /// Pair a storage with a layout.
    pub fn new(storage: &'a Storage, layout: &'a Layout) -> Self {
        View { storage, layout }
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
    /// Output spatial size for an input of `[h, w]` and kernel `[kh, kw]`, or `None` if it would be empty.
    pub fn output_hw(&self, h: usize, w: usize, kh: usize, kw: usize) -> Option<[usize; 2]> {
        let f = |i: usize, k: usize, s: usize, p: usize, d: usize| {
            let eff = d * (k - 1) + 1;
            (i + 2 * p).checked_sub(eff).map(|v| v / s + 1)
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
    /// Output spatial size for an input of `[h, w]`, or `None` if it would be empty.
    pub fn output_hw(&self, h: usize, w: usize) -> Option<[usize; 2]> {
        let f = |i: usize, k: usize, s: usize, p: usize| (i + 2 * p).checked_sub(k).map(|v| v / s + 1);
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
    /// Max pooling of `[N, C, H, W]`: returns the values and, per output element, the flat index `h * W + w`
    /// of the chosen input element within its plane (`I64`).
    fn max_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<(Storage, Storage)> {
        unsupported!(self, "max_pool2d")
    }
    /// Scatter `grad` back to the positions in `indices` (from [`Backend::max_pool2d`]), summing collisions.
    fn max_pool2d_backward(&self, grad: View<'_>, indices: View<'_>, input_shape: &Shape) -> Result<Storage> {
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
    /// The `k` largest (or smallest) entries along the last dim, sorted, as `(values, I64 indices)`.
    fn topk(&self, x: View<'_>, k: usize, largest: bool) -> Result<(Storage, Storage)> {
        unsupported!(self, "topk")
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

/// Every device with a backend: `Cpu` first, then registered GPUs.
pub fn available_devices() -> Vec<Device> {
    let reg = GPU_BACKENDS.read().unwrap_or_else(|e| e.into_inner());
    std::iter::once(Device::Cpu)
        .chain(reg.iter().enumerate().filter(|(_, b)| b.is_some()).map(|(i, _)| Device::Gpu(i)))
        .collect()
}
