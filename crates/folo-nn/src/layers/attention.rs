//! Differentiable vision attention blocks:
//!
//! - [`SqueezeExcitation`]: Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).
//! - [`Eca`] & [`eca_kernel_size`]: Efficient Channel Attention with adaptive 1-D channel convolution
//!   (Wang et al., CVPR 2020).
//! - [`ChannelAttention`], [`SpatialAttention`], and [`Cbam`]: Convolutional Block Attention Module
//!   combining channel and spatial attention sequentially (Woo et al., ECCV 2018).
//!
//! All blocks operate on `NCHW` tensors with full autograd support on inputs and every trainable parameter,
//! clean hierarchical parameter names for `.folo` state dict round-trips, and explicit handling of zero-batch
//! (`N == 0`), odd spatial dimensions, and invalid shapes/hyperparameters.

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, DType, Tensor};

use crate::init;
use crate::layers::{Activation, Conv2d, Conv2dConfig};
use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Validate an `NCHW` input tensor against expected channel count `expected_c` (`None` accepts any `C >= 1`).
/// Returns `(n, c, h, w)`.
fn check_nchw(x: &Tensor, expected_c: Option<usize>, op: &str) -> Result<(usize, usize, usize, usize)> {
    if !matches!(x.dtype(), DType::F32 | DType::F16) {
        return Err(Error::Invalid(format!("{op}: input dtype must be f32 or f16, got {}", x.dtype())));
    }
    let (n, c, h, w) =
        x.dims4().map_err(|_| Error::Invalid(format!("{op}: input must be 4-D NCHW, got {:?}", x.dims())))?;
    if let Some(ec) = expected_c
        && c != ec
    {
        return Err(Error::Invalid(format!("{op}: input has {c} channels, expected {ec}")));
    }
    if c == 0 || h == 0 || w == 0 {
        return Err(Error::Invalid(format!(
            "{op}: channels and spatial dimensions must be >= 1, got [{n}, {c}, {h}, {w}]"
        )));
    }
    Ok((n, c, h, w))
}

/// Squeeze-and-Excitation (SE) channel attention block (Hu et al., CVPR 2018).
///
/// ```text
/// s = sigmoid(fc2(act(fc1(global_avg_pool(x)))))   // [N, C, 1, 1]
/// y = x * s                                         // [N, C, H, W]
/// ```
///
/// Parameter names: `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`.
pub struct SqueezeExcitation {
    /// Squeeze 1×1 convolution (`channels -> hidden`).
    pub fc1: Conv2d,
    /// Excitation 1×1 convolution (`hidden -> channels`).
    pub fc2: Conv2d,
    act: Activation,
    channels: usize,
    hidden: usize,
}

impl SqueezeExcitation {
    /// Create a [`SqueezeExcitation`] block with bottleneck `hidden = (channels / reduction).max(1)`
    /// and `ReLU` hidden activation (Hu et al. 2018 default).
    pub fn new(channels: usize, reduction: usize, rng: &mut Rng) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Invalid("SqueezeExcitation: channels must be >= 1, got 0".into()));
        }
        if reduction == 0 {
            return Err(Error::Invalid("SqueezeExcitation: reduction must be >= 1, got 0".into()));
        }
        let hidden = (channels / reduction).max(1);
        Self::with_hidden(channels, hidden, Activation::Relu, rng)
    }

    /// Create a [`SqueezeExcitation`] block with explicit `hidden` bottleneck channels and hidden activation.
    pub fn with_hidden(channels: usize, hidden: usize, act: Activation, rng: &mut Rng) -> Result<Self> {
        if channels == 0 || hidden == 0 {
            return Err(Error::Invalid(format!(
                "SqueezeExcitation: channels ({channels}) and hidden ({hidden}) must be >= 1"
            )));
        }
        let fc1 = Conv2d::try_new(Conv2dConfig::new(channels, hidden, 1).padding(0).bias(true), rng)?;
        let fc2 = Conv2d::try_new(Conv2dConfig::new(hidden, channels, 1).padding(0).bias(true), rng)?;
        Ok(Self { fc1, fc2, act, channels, hidden })
    }

    /// Input/output channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Bottleneck channel count.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Hidden activation function.
    pub fn activation(&self) -> Activation {
        self.act
    }
}

impl Module for SqueezeExcitation {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("fc1", &self.fc1);
        f("fc2", &self.fc2);
    }
}

impl Forward for SqueezeExcitation {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, _, _, _) = check_nchw(x, Some(self.channels), "SqueezeExcitation")?;
        if n == 0 {
            return Ok(x.mul_scalar(0.0)?);
        }
        let s = x.mean([2, 3], true)?;
        let s = self.fc1.forward(&s)?;
        let s = self.act.forward(&s)?;
        let s = self.fc2.forward(&s)?.sigmoid()?;
        Ok(x.mul(&s)?)
    }
}

/// Compute the adaptive odd 1-D kernel size for Efficient Channel Attention (Wang et al., CVPR 2020 Eq. 12):
///
/// ```text
/// t = floor(|(log2(channels) + b) / gamma|)
/// k = t if t is odd else t + 1, clamped to >= 1
/// ```
///
/// Paper defaults are `gamma = 2.0, b = 1.0` (e.g. `C = 64 -> k = 3`, `C = 256 -> k = 5`).
pub fn eca_kernel_size(channels: usize, gamma: f32, b: f32) -> Result<usize> {
    if channels == 0 {
        return Err(Error::Invalid("eca_kernel_size: channels must be >= 1, got 0".into()));
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(Error::Invalid(format!("eca_kernel_size: gamma must be finite and > 0, got {gamma}")));
    }
    if !b.is_finite() {
        return Err(Error::Invalid(format!("eca_kernel_size: b must be finite, got {b}")));
    }
    let raw = (((channels as f64).log2() + b as f64) / gamma as f64).abs().floor() as usize;
    let k = if raw == 0 {
        1
    } else if raw % 2 == 1 {
        raw
    } else {
        raw + 1
    };
    Ok(k)
}

/// Efficient Channel Attention (ECA) block (Wang et al., CVPR 2020).
///
/// Avoids dimensionality reduction by applying a local cross-channel 1-D convolution of odd kernel size `k`
/// to the global-average-pooled channel descriptor:
///
/// ```text
/// s = sigmoid(conv1d_k(global_avg_pool(x)))   // [N, C, 1, 1]
/// y = x * s                                    // [N, C, H, W]
/// ```
///
/// Parameter name: `weight` (`[1, 1, 1, k]`, shared 1-D kernel over the channel axis).
pub struct Eca {
    /// 1-D cross-channel convolution kernel stored as `[1, 1, 1, k]`.
    pub weight: Tensor,
    channels: usize,
    kernel_size: usize,
}

impl Eca {
    /// Create an [`Eca`] block with the adaptive odd kernel size `eca_kernel_size(channels, 2.0, 1.0)`.
    pub fn new(channels: usize, rng: &mut Rng) -> Result<Self> {
        let k = eca_kernel_size(channels, 2.0, 1.0)?;
        Self::with_kernel_size(channels, k, rng)
    }

    /// Create an [`Eca`] block with an explicit odd `kernel_size >= 1`.
    pub fn with_kernel_size(channels: usize, kernel_size: usize, rng: &mut Rng) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Invalid("Eca: channels must be >= 1, got 0".into()));
        }
        if kernel_size == 0 || kernel_size.is_multiple_of(2) {
            return Err(Error::Invalid(format!("Eca: kernel_size must be odd and >= 1, got {kernel_size}")));
        }
        let b = init::default_conv_bound(kernel_size);
        let weight = init::uniform([1, 1, 1, kernel_size], b, rng).requires_grad_();
        Ok(Self { weight, channels, kernel_size })
    }

    /// Input/output channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Odd 1-D cross-channel kernel size `k`.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for Eca {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("weight", &self.weight, TensorKind::Parameter);
    }
}

impl Forward for Eca {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, c, _, _) = check_nchw(x, Some(self.channels), "Eca")?;
        if self.weight.dims() != [1, 1, 1, self.kernel_size] {
            return Err(Error::Invalid(format!(
                "Eca: weight shape {:?} does not match [1, 1, 1, {}]",
                self.weight.dims(),
                self.kernel_size
            )));
        }
        if x.device() != self.weight.device() {
            return Err(Error::Invalid(format!(
                "Eca: input on {:?} does not match weight on {:?}",
                x.device(),
                self.weight.device()
            )));
        }
        if n == 0 {
            return Ok(x.mul_scalar(0.0)?);
        }
        let y = x.mean([2, 3], true)?; // [N, C, 1, 1]
        let y_1d = y.reshape([n, 1, 1, c])?;
        let params =
            Conv2dParams { stride: [1, 1], padding: [0, self.kernel_size / 2], dilation: [1, 1], groups: 1 };
        let w = if y_1d.dtype() == DType::F16 && self.weight.dtype() == DType::F32 {
            self.weight.to_dtype(DType::F16)?
        } else {
            self.weight.clone()
        };
        let s = y_1d.conv2d_with(&w, None, &params)?.sigmoid()?.reshape([n, c, 1, 1])?;
        Ok(x.mul(&s)?)
    }
}

/// CBAM Channel Attention sub-module (Woo et al., ECCV 2018).
///
/// Pools spatial features with both average and max pooling, passes both through a shared 2-layer
/// 1×1 bottleneck MLP (`fc1 -> ReLU -> fc2`), sums the logits, and applies sigmoid gating.
///
/// Parameter names: `fc1.weight`, `fc2.weight`.
pub struct ChannelAttention {
    /// Shared bottleneck reduction conv (`channels -> hidden`, 1×1, no bias).
    pub fc1: Conv2d,
    /// Shared bottleneck expansion conv (`hidden -> channels`, 1×1, no bias).
    pub fc2: Conv2d,
    channels: usize,
    hidden: usize,
}

impl ChannelAttention {
    /// Create a [`ChannelAttention`] module with `hidden = (channels / reduction).max(1)`.
    pub fn new(channels: usize, reduction: usize, rng: &mut Rng) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Invalid("ChannelAttention: channels must be >= 1, got 0".into()));
        }
        if reduction == 0 {
            return Err(Error::Invalid("ChannelAttention: reduction must be >= 1, got 0".into()));
        }
        let hidden = (channels / reduction).max(1);
        let fc1 = Conv2d::try_new(Conv2dConfig::new(channels, hidden, 1).padding(0).bias(false), rng)?;
        let fc2 = Conv2d::try_new(Conv2dConfig::new(hidden, channels, 1).padding(0).bias(false), rng)?;
        Ok(Self { fc1, fc2, channels, hidden })
    }

    /// Input/output channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Bottleneck channel count.
    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

impl Module for ChannelAttention {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("fc1", &self.fc1);
        f("fc2", &self.fc2);
    }
}

impl Forward for ChannelAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, _, _, _) = check_nchw(x, Some(self.channels), "ChannelAttention")?;
        if n == 0 {
            return Ok(x.mul_scalar(0.0)?);
        }
        let avg_pool = x.mean([2, 3], true)?;
        let max_pool = x.max([2, 3], true)?;
        let avg_out = self.fc2.forward(&self.fc1.forward(&avg_pool)?.relu()?)?;
        let max_out = self.fc2.forward(&self.fc1.forward(&max_pool)?.relu()?)?;
        let mc = avg_out.add(&max_out)?.sigmoid()?;
        Ok(x.mul(&mc)?)
    }
}

/// CBAM Spatial Attention sub-module (Woo et al., ECCV 2018).
///
/// Concatenates channel-wise average and max maps (`[N, 2, H, W]`), applies a `k × k` convolution
/// (`2 -> 1`, odd `k`, default `7`), and gates the input with the resulting sigmoid spatial map.
///
/// Parameter name: `conv.weight`.
pub struct SpatialAttention {
    /// Spatial attention convolution (`2 -> 1`, `k × k`, padding `k / 2`, no bias).
    pub conv: Conv2d,
    kernel_size: usize,
}

impl SpatialAttention {
    /// Create a [`SpatialAttention`] module with odd `kernel_size >= 1` (paper default `7`).
    pub fn new(kernel_size: usize, rng: &mut Rng) -> Result<Self> {
        if kernel_size == 0 || kernel_size.is_multiple_of(2) {
            return Err(Error::Invalid(format!(
                "SpatialAttention: kernel_size must be odd and >= 1, got {kernel_size}"
            )));
        }
        let conv =
            Conv2d::try_new(Conv2dConfig::new(2, 1, kernel_size).padding(kernel_size / 2).bias(false), rng)?;
        Ok(Self { conv, kernel_size })
    }

    /// Odd spatial kernel size `k`.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for SpatialAttention {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("conv", &self.conv);
    }
}

impl Forward for SpatialAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, _, _, _) = check_nchw(x, None, "SpatialAttention")?;
        if n == 0 {
            return Ok(x.mul_scalar(0.0)?);
        }
        let avg_s = x.mean(1, true)?;
        let max_s = x.max(1, true)?;
        let cat = Tensor::cat(&[avg_s, max_s], 1)?;
        let ms = self.conv.forward(&cat)?.sigmoid()?;
        Ok(x.mul(&ms)?)
    }
}

/// Convolutional Block Attention Module (CBAM; Woo et al., ECCV 2018).
///
/// Applies [`ChannelAttention`] followed sequentially by [`SpatialAttention`].
///
/// Parameter names: `channel.fc1.weight`, `channel.fc2.weight`, `spatial.conv.weight`.
pub struct Cbam {
    /// Channel attention stage.
    pub channel: ChannelAttention,
    /// Spatial attention stage.
    pub spatial: SpatialAttention,
}

impl Cbam {
    /// Create a [`Cbam`] block for `channels` with channel `reduction` and odd `spatial_kernel` (e.g. `7`).
    pub fn new(channels: usize, reduction: usize, spatial_kernel: usize, rng: &mut Rng) -> Result<Self> {
        let channel = ChannelAttention::new(channels, reduction, rng)?;
        let spatial = SpatialAttention::new(spatial_kernel, rng)?;
        Ok(Self { channel, spatial })
    }

    /// Input/output channel count.
    pub fn channels(&self) -> usize {
        self.channel.channels()
    }

    /// Spatial attention kernel size.
    pub fn spatial_kernel(&self) -> usize {
        self.spatial.kernel_size()
    }
}

impl Module for Cbam {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("channel", &self.channel);
        f("spatial", &self.spatial);
    }
}

impl Forward for Cbam {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.channel.forward(x)?;
        self.spatial.forward(&h)
    }
}
