//! Linear, activations, pooling, upsampling, concatenation and identity.

use std::borrow::Borrow;

use folo_core::cpu::gemm::{Strided, gemm};
use folo_core::rng::Rng;
use folo_core::{DType, Tensor, UnaryOp, is_grad_enabled};

use crate::init;
use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// A fully connected layer: `y = x · Wᵀ + b` over the last dimension. Weight `[out, in]`, bias `[out]`.
/// Tensor names: `weight`, `bias`.
pub struct Linear {
    /// `[out, in]`.
    pub weight: Tensor,
    /// `[out]`, if enabled.
    pub bias: Option<Tensor>,
}

impl Linear {
    /// PyTorch's default init: weight then bias, `U(−b, b)` with `b = 1 / sqrt(in)`.
    pub fn new(in_features: usize, out_features: usize, bias: bool, rng: &mut Rng) -> Self {
        let b = init::default_conv_bound(in_features);
        let weight = init::uniform([out_features, in_features], b, rng).requires_grad_();
        let bias = bias.then(|| init::uniform([out_features], b, rng).requires_grad_());
        Linear { weight, bias }
    }

    /// A layer over existing tensors.
    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (o, _) = weight.dims2().map_err(|_| {
            Error::Invalid(format!("Linear: weight must be [out, in], got {:?}", weight.dims()))
        })?;
        if let Some(b) = &bias
            && b.dims() != [o]
        {
            return Err(Error::Invalid(format!("Linear: bias {:?} does not match [{o}]", b.dims())));
        }
        Ok(Linear { weight, bias })
    }
}

impl Module for Linear {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("weight", &self.weight, TensorKind::Parameter);
        if let Some(b) = &self.bias {
            f("bias", b, TensorKind::Parameter);
        }
    }
}

impl Forward for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (o, i) = self.weight.dims2()?;
        let last = *x.dims().last().ok_or_else(|| Error::Invalid("Linear: scalar input".into()))?;
        if last != i {
            return Err(Error::Invalid(format!(
                "Linear: input {:?} has {last} features, layer expects {i}",
                x.dims()
            )));
        }
        let rows = x.numel() / i.max(1);
        let mut out_shape = x.dims().to_vec();
        *out_shape.last_mut().expect("rank >= 1") = o;
        if !is_grad_enabled()
            && rows > 0
            && i > 0
            && o > 0
            && x.device().is_cpu()
            && x.dtype() == DType::F32
            && self.weight.device().is_cpu()
            && self.weight.dtype() == DType::F32
            && !x.same_storage(&self.weight)
            && self.bias.as_ref().is_none_or(|b| {
                b.device().is_cpu()
                    && b.dtype() == DType::F32
                    && b.dims() == [o]
                    && !b.same_storage(x)
                    && !b.same_storage(&self.weight)
            })
        {
            let mut out = folo_core::cpu::take_buffer(rows * o);
            x.with_f32_slice(|xs| {
                self.weight.with_f32_slice(|ws| {
                    let lhs = Strided::lhs(xs, 0, i, 1);
                    let rhs = Strided::rhs(ws, 0, 1, i);
                    gemm(rows, o, i, &lhs, &rhs, &mut out, o, false, true);
                })
            })??;
            if let Some(b) = &self.bias {
                b.with_f32_slice(|bs| {
                    for row in out.chunks_exact_mut(o) {
                        for (dst, &bv) in row.iter_mut().zip(bs) {
                            *dst += bv;
                        }
                    }
                })?;
            }
            return Ok(Tensor::from_vec(out, out_shape)?);
        }
        let mut y = x.reshape([rows, i])?.matmul(&self.weight.t()?)?;
        if let Some(b) = &self.bias {
            let b = if b.dtype() == DType::F32
                && y.dtype() == DType::F16
                && folo_core::autocast_dtype() == Some(DType::F16)
            {
                b.to_dtype(DType::F16)?
            } else {
                b.clone()
            };
            y = y.add(&b)?;
        }
        Ok(y.reshape(out_shape)?)
    }
}

/// An elementwise activation (no tensors).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// `x` (docs/MODEL.md `act=none`).
    Identity,
    /// `x · sigmoid(x)`, the FOLO default.
    Silu,
    /// `max(x, 0)`.
    Relu,
    /// `x` if `x > 0`, else `slope · x`.
    LeakyRelu(f32),
    /// `1 / (1 + e^−x)`.
    Sigmoid,
    /// `tanh(x)`.
    Tanh,
    /// `x · relu6(x + 3) / 6`.
    Hardswish,
}

impl Activation {
    /// The folo-core op this activation applies (`None` for the identity), for fused kernels
    /// ([`Tensor::conv2d_act`], [`Tensor::batch_norm_act_train`]).
    pub fn unary_op(self) -> Option<UnaryOp> {
        Some(match self {
            Activation::Identity => return None,
            Activation::Silu => UnaryOp::Silu,
            Activation::Relu => UnaryOp::Relu,
            Activation::LeakyRelu(s) => UnaryOp::LeakyRelu(s),
            Activation::Sigmoid => UnaryOp::Sigmoid,
            Activation::Tanh => UnaryOp::Tanh,
            Activation::Hardswish => UnaryOp::Hardswish,
        })
    }
}

impl Module for Activation {}

impl Forward for Activation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(match *self {
            Activation::Identity => x.clone(),
            Activation::Silu => x.silu()?,
            Activation::Relu => x.relu()?,
            Activation::LeakyRelu(s) => x.leaky_relu(s)?,
            Activation::Sigmoid => x.sigmoid()?,
            Activation::Tanh => x.tanh()?,
            Activation::Hardswish => x.hardswish()?,
        })
    }
}

/// Returns its input.
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Module for Identity {}

impl Forward for Identity {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

/// Max pooling with a square window. Padded positions count as −∞, so padding never wins (docs/MODEL.md §2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2d {
    /// Window size.
    pub kernel: usize,
    /// Stride.
    pub stride: usize,
    /// Padding on every side.
    pub padding: usize,
}

impl MaxPool2d {
    /// A `kernel × kernel` max pool.
    pub fn new(kernel: usize, stride: usize, padding: usize) -> Self {
        MaxPool2d { kernel, stride, padding }
    }
}

impl Module for MaxPool2d {}

impl Forward for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.max_pool2d(self.kernel, self.stride, self.padding)?)
    }
}

/// Nearest-neighbour upsampling by an integer factor: `out[y][x] = in[y / s][x / s]` (docs/MODEL.md §2.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Upsample {
    /// Scale factor on both axes.
    pub scale: usize,
}

impl Upsample {
    /// Nearest upsampling by `scale`.
    pub fn nearest(scale: usize) -> Self {
        Upsample { scale }
    }
}

impl Module for Upsample {}

impl Forward for Upsample {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.upsample_nearest2d(self.scale)?)
    }
}

/// Concatenation along a dimension (channels, `dim = 1`, in FOLO), operands in the order given. It has
/// several inputs, so it is not a [`Forward`]; call [`Concat::forward`] with a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Concat {
    /// The dimension to concatenate along.
    pub dim: usize,
}

impl Concat {
    /// Concatenate along channels (`dim = 1` of `NCHW`).
    pub fn channels() -> Self {
        Concat { dim: 1 }
    }

    /// Concatenate `xs` in order.
    pub fn forward<T: Borrow<Tensor>>(&self, xs: &[T]) -> Result<Tensor> {
        Ok(Tensor::cat(xs, self.dim)?)
    }
}

impl Module for Concat {}
