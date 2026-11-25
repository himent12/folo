//! View, op descriptions, parameter structs, and optimizer/detector descriptors for [`super::Backend`].

use std::sync::Arc;

use crate::storage::Derived;
use crate::{Layout, Shape, Storage};

pub use crate::ops::muon::{MAX_MUON_STEPS, MuonItem, MuonSettings};

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

/// Settings of [`super::Backend::detect_class_sites`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassSites {
    /// A pixel is a candidate when its best class score is `> conf`.
    pub conf: f32,
    /// Candidate slots per image.
    pub cap: usize,
}

/// One level of [`super::Backend::detect_decode_sites`].
#[derive(Clone, Copy)]
pub struct SiteLevel<'a> {
    /// Box logits at the slots, `[B, 4·reg_max, cap]`.
    pub raw: View<'a>,
    /// The level's `sites` from [`super::Backend::detect_class_sites`].
    pub sites: View<'a>,
    /// The level's `recs`.
    pub recs: View<'a>,
    /// Map width.
    pub width: usize,
    /// The level's stride (anchor points are pixel centres, `(x + 0.5, y + 0.5)`, in grid units).
    pub stride: f32,
    /// The level's first anchor.
    pub start: usize,
}

/// One image of [`super::Backend::letterbox_rgb8`]. The sampling is folo-data's `ResizePlan` for a `src` →
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

/// Settings of [`super::Backend::detect_candidates`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectDecode {
    /// DFL bins per box side.
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Keep a candidate when its score is `> conf`.
    pub conf: f32,
}

/// Words per record of [`super::Backend::detect_candidates`]: anchor, class, score, x1, y1, x2, y2.
pub const DETECT_RECORD_WORDS: usize = 7;

/// The update [`super::Backend::optimizer_step`] applies, per element, in `f32`.
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
    /// Decoupled or Cautious-Decoupled SGD (P3): momentum accumulates raw `g` only (no coupled `wd · p`);
    /// parameter decay subtracts `wd_step · p` (gated by `u · p > 0` when `cautious` is true), independent of
    /// `MuSgd`'s `sgd` step weight.
    SgdDecoupled {
        /// Learning rate for the gradient / momentum direction `u`.
        lr: f32,
        /// `μ` (used only with a momentum buffer).
        momentum: f32,
        /// Pre-multiplied decoupled weight-decay step `lr_unscaled · wd`.
        wd_step: f32,
        /// Nesterov momentum.
        nesterov: bool,
        /// Cautious mask: apply decay only where `u · p > 0`.
        cautious: bool,
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

/// What [`super::Backend::norm_partials_many`] reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NormReduce {
    /// The largest `|x|`, NaN propagating.
    AbsMax,
    /// The sum of `((x * h) * h)^2`.
    ScaledSquares,
}

/// One parameter of [`super::Backend::optimizer_step_many`]: the arguments of one
/// [`super::Backend::optimizer_step`] call.
#[derive(Debug)]
pub struct OptimItem<'a> {
    /// The parameter (`grad.numel()` elements, contiguous from element 0).
    pub param: &'a mut Storage,
    /// Its gradient.
    pub grad: View<'a>,
    /// Its state buffers, as for [`super::Backend::optimizer_step`].
    pub state: Vec<&'a mut Storage>,
}

/// The running statistics [`super::Backend::batch_norm_act`] updates in place.
#[derive(Debug)]
pub struct RunningUpdate<'a> {
    /// Running mean, `[C]` contiguous.
    pub mean: &'a mut Storage,
    /// Running (unbiased) variance, `[C]` contiguous.
    pub var: &'a mut Storage,
    /// Weight of the batch statistic.
    pub momentum: f32,
}
