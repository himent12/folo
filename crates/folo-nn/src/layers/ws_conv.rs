//! Weight Standardization (`standardize_weight` and [`WeightStandardizedConv2d`]; Qiao et al.,
//! arXiv:1903.10520).
//!
//! Normalizes each output filter `W_o` of a convolution weight `[C_out, C_in / groups, kH, kW]` to zero
//! mean and unit variance across its fan-in axes `(C_in / groups, kH, kW)`:
//!
//! ```text
//! mu_o    = mean(W_o)
//! var_o   = mean((W_o - mu_o)^2)
//! W_hat_o = (W_o - mu_o) / sqrt(var_o + eps)
//! ```
//!
//! Implemented with pure differentiable two-pass tensor operations (avoiding `E[W^2] - E[W]^2` cancellation
//! and working on both CPU and GPU with full autograd through both the standardized weight and the input).

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};

use crate::init;
use crate::layers::Conv2dConfig;
use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Default numerical stability epsilon for Weight Standardization (`1e-5`, Qiao et al. 2019).
pub const WS_CONV_DEFAULT_EPS: f32 = 1e-5;

/// Differentiably standardize every row (output filter) of `weight` (rank >= 2, `F32` or `F16`) to zero
/// mean and unit variance across all trailing dimensions `1..rank`:
///
/// ```text
/// W_hat[o, ...] = (W[o, ...] - mean(W[o, ...])) / sqrt(var(W[o, ...]) + eps)
/// ```
///
/// Uses a two-pass centered variance (`mean((W - mean(W))^2)`) so constant filters, large offsets, and
/// `fan_in == 1` stay numerically well-conditioned with zero `NaN` or `±Inf`. `F16` weights are standardized
/// in `F32` and cast back to `F16`.
pub fn standardize_weight(weight: &Tensor, eps: f32) -> Result<Tensor> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!("standardize_weight: eps must be finite and > 0, got {eps}")));
    }
    let orig_dtype = weight.dtype();
    if !matches!(orig_dtype, DType::F32 | DType::F16) {
        return Err(Error::Invalid(format!(
            "standardize_weight: weight dtype must be f32 or f16, got {orig_dtype}"
        )));
    }
    let dims = weight.dims();
    if dims.len() < 2 {
        return Err(Error::Invalid(format!(
            "standardize_weight: weight must have rank >= 2, got shape {dims:?}"
        )));
    }
    let out_c = dims[0];
    let fan_in: usize = dims[1..].iter().product();
    if out_c == 0 || fan_in == 0 {
        return Err(Error::Invalid(format!(
            "standardize_weight: weight must be non-empty, got shape {dims:?}"
        )));
    }

    let work = if orig_dtype == DType::F16 { weight.to_dtype(DType::F32)? } else { weight.clone() };
    let flat = work.reshape([out_c, fan_in])?;
    let mean = flat.mean(1, true)?;
    let centered = flat.sub(&mean)?;
    let var = centered.square()?.mean(1, true)?;
    let denom = var.add_scalar(eps)?.sqrt()?;
    let normed = centered.div(&denom)?.reshape(weight.shape())?;
    if orig_dtype == DType::F16 { Ok(normed.to_dtype(DType::F16)?) } else { Ok(normed) }
}

/// 2-D convolution with Weight Standardization (Qiao et al. 2019).
///
/// Standardizes `self.weight` per output channel via [`standardize_weight`] on every forward pass before
/// applying 2-D convolution, keeping the underlying trainable parameter `weight` unconstrained.
///
/// Parameter names: `weight`, `bias` (identical to [`crate::Conv2d`]).
pub struct WeightStandardizedConv2d {
    /// Raw trainable weight `[out, in / groups, k, k]`.
    pub weight: Tensor,
    /// Optional trainable bias `[out]`.
    pub bias: Option<Tensor>,
    config: Conv2dConfig,
    eps: f32,
}

impl WeightStandardizedConv2d {
    /// Create a [`WeightStandardizedConv2d`] initialized identically to [`crate::Conv2d`] with
    /// `eps = WS_CONV_DEFAULT_EPS`.
    pub fn new(config: Conv2dConfig, rng: &mut Rng) -> Result<Self> {
        Self::with_eps(config, WS_CONV_DEFAULT_EPS, rng)
    }

    /// Create a [`WeightStandardizedConv2d`] with an explicit variance floor `eps > 0`.
    pub fn with_eps(config: Conv2dConfig, eps: f32, rng: &mut Rng) -> Result<Self> {
        validate_ws_conv(&config, eps)?;
        let b = init::default_conv_bound(config.fan_in());
        let weight = init::uniform(config.weight_shape(), b, rng).requires_grad_();
        let bias = config.bias.then(|| init::uniform([config.out_channels], b, rng).requires_grad_());
        Ok(Self { weight, bias, config, eps })
    }

    /// Construct a [`WeightStandardizedConv2d`] from existing `weight` and optional `bias` tensors.
    pub fn from_tensors(
        mut config: Conv2dConfig,
        weight: Tensor,
        bias: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        config.bias = bias.is_some();
        validate_ws_conv(&config, eps)?;
        if weight.dims() != config.weight_shape() || !matches!(weight.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!(
                "WeightStandardizedConv2d: weight {:?} {} does not match {:?}",
                weight.dims(),
                weight.dtype(),
                config.weight_shape()
            )));
        }
        if let Some(b) = &bias
            && (b.dims() != [config.out_channels] || b.dtype() != weight.dtype())
        {
            return Err(Error::Invalid(format!(
                "WeightStandardizedConv2d: bias {:?} does not match [{}]",
                b.dims(),
                config.out_channels
            )));
        }
        Ok(Self { weight, bias, config, eps })
    }

    /// Convolution geometry configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    /// Variance floor `eps`.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Return the current standardized weight `W_hat`.
    pub fn standardized_weight(&self) -> Result<Tensor> {
        standardize_weight(&self.weight, self.eps)
    }
}

fn validate_ws_conv(c: &Conv2dConfig, eps: f32) -> Result<()> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!(
            "WeightStandardizedConv2d: eps must be finite and > 0, got {eps}"
        )));
    }
    if c.in_channels == 0
        || c.out_channels == 0
        || c.kernel == 0
        || c.stride == 0
        || c.dilation == 0
        || c.groups == 0
        || !c.in_channels.is_multiple_of(c.groups)
        || !c.out_channels.is_multiple_of(c.groups)
    {
        return Err(Error::Invalid(format!("WeightStandardizedConv2d: invalid config {c:?}")));
    }
    Ok(())
}

impl Module for WeightStandardizedConv2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("weight", &self.weight, TensorKind::Parameter);
        if let Some(b) = &self.bias {
            f("bias", b, TensorKind::Parameter);
        }
    }
}

impl Forward for WeightStandardizedConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_hat = standardize_weight(&self.weight, self.eps)?;
        Ok(x.conv2d_with(&w_hat, self.bias.as_ref(), &self.config.params())?)
    }
}
