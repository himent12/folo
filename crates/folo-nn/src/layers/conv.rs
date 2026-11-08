//! 2-D convolution.

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, DType, Tensor};

use crate::init;
use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Hyperparameters of a [`Conv2d`]. Square kernels, the same stride / padding / dilation on both axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Kernel size `k` (a `k × k` kernel).
    pub kernel: usize,
    /// Stride.
    pub stride: usize,
    /// Zero padding on every side.
    pub padding: usize,
    /// Dilation.
    pub dilation: usize,
    /// Groups (`in_channels` for a depthwise conv).
    pub groups: usize,
    /// Whether the conv has a bias.
    pub bias: bool,
}

impl Conv2dConfig {
    /// `in → out` with a `k × k` kernel, stride 1, padding `k / 2` (docs/MODEL.md conventions), no dilation,
    /// one group, with a bias.
    pub fn new(in_channels: usize, out_channels: usize, kernel: usize) -> Self {
        Conv2dConfig {
            in_channels,
            out_channels,
            kernel,
            stride: 1,
            padding: kernel / 2,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }

    /// Set the stride.
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }

    /// Set the padding.
    pub fn padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }

    /// Set the dilation.
    pub fn dilation(mut self, d: usize) -> Self {
        self.dilation = d;
        self
    }

    /// Set the number of groups.
    pub fn groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    /// Enable or disable the bias.
    pub fn bias(mut self, on: bool) -> Self {
        self.bias = on;
        self
    }

    /// The weight shape `[out, in / groups, k, k]`.
    pub fn weight_shape(&self) -> [usize; 4] {
        [self.out_channels, self.in_channels / self.groups.max(1), self.kernel, self.kernel]
    }

    /// `(in / groups) · k · k`.
    pub fn fan_in(&self) -> usize {
        self.in_channels / self.groups.max(1) * self.kernel * self.kernel
    }

    fn validate(&self) -> Result<()> {
        let c = self;
        if c.in_channels == 0
            || c.out_channels == 0
            || c.kernel == 0
            || c.stride == 0
            || c.dilation == 0
            || c.groups == 0
            || !c.in_channels.is_multiple_of(c.groups)
            || !c.out_channels.is_multiple_of(c.groups)
        {
            return Err(Error::Invalid(format!("Conv2d: invalid config {c:?}")));
        }
        Ok(())
    }

    fn params(&self) -> Conv2dParams {
        Conv2dParams {
            stride: [self.stride; 2],
            padding: [self.padding; 2],
            dilation: [self.dilation; 2],
            groups: self.groups,
        }
    }
}

/// A 2-D convolution over `NCHW` input with weight `[out, in / groups, k, k]` and an optional bias `[out]`.
/// Tensor names: `weight`, `bias`.
pub struct Conv2d {
    /// The weight, `[out, in / groups, k, k]`.
    pub weight: Tensor,
    /// The bias, `[out]`, if configured.
    pub bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    /// A conv initialised as docs/MODEL.md §2.9 specifies: weight then bias, each `U(−b, b)` with
    /// `b = 1 / sqrt(fan_in)`, drawn from `rng` in that order.
    ///
    /// Panics on an invalid config (zero sizes, channels not divisible by groups); use
    /// [`Conv2d::try_new`] to get an error instead.
    pub fn new(config: Conv2dConfig, rng: &mut Rng) -> Self {
        Self::try_new(config, rng).expect("Conv2d::new")
    }

    /// [`Conv2d::new`] returning an error for an invalid config.
    pub fn try_new(config: Conv2dConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let b = init::default_conv_bound(config.fan_in());
        let weight = init::uniform(config.weight_shape(), b, rng).requires_grad_();
        let bias = config.bias.then(|| init::uniform([config.out_channels], b, rng).requires_grad_());
        Ok(Conv2d { weight, bias, config })
    }

    /// A conv over existing tensors (shapes are checked against `config`; `config.bias` follows `bias`). The
    /// tensors are used as they are: mark them `requires_grad_()` to train them.
    pub fn from_tensors(mut config: Conv2dConfig, weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        config.bias = bias.is_some();
        config.validate()?;
        if weight.dims() != config.weight_shape() || weight.dtype() != DType::F32 {
            return Err(Error::Invalid(format!(
                "Conv2d: weight {:?} {} does not match {:?}",
                weight.dims(),
                weight.dtype(),
                config.weight_shape()
            )));
        }
        if let Some(b) = &bias
            && (b.dims() != [config.out_channels] || b.dtype() != DType::F32)
        {
            return Err(Error::Invalid(format!(
                "Conv2d: bias {:?} does not match [{}]",
                b.dims(),
                config.out_channels
            )));
        }
        Ok(Conv2d { weight, bias, config })
    }

    /// The configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }
}

impl Module for Conv2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("weight", &self.weight, TensorKind::Parameter);
        if let Some(b) = &self.bias {
            f("bias", b, TensorKind::Parameter);
        }
    }
}

impl Forward for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.conv2d_with(&self.weight, self.bias.as_ref(), &self.config.params())?)
    }
}
