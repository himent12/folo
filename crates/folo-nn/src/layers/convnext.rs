//! Layer Normalization ([`LayerNorm`]) and modern ConvNeXt residual blocks ([`ConvNeXtBlock`];
//! Liu et al., CVPR 2022 / arXiv:2201.03545).
//!
//! [`ConvNeXtBlock`] operates on `NCHW` feature maps `[N, C, H, W]`:
//! ```text
//! h = dwconv_{k×k}(x)                       // depthwise Conv2d (default k = 7)
//! h = permute(h, [0, 2, 3, 1])              // NCHW -> NHWC
//! h = norm(h)                               // LayerNorm (default) or RmsNorm over C
//! h = pwconv2(GELU_tanh(pwconv1(h)))        // inverted bottleneck C -> expand*C -> C
//! h = layer_scale(h)                        // optional learnable per-channel gamma
//! h = permute(h, [0, 3, 1, 2])              // NHWC -> NCHW
//! y = x + drop_path(h)                      // optional stochastic depth + residual
//! ```
//!
//! All sub-modules are fully differentiable with clean dotted parameter names for `.folo` round-trips,
//! and [`crate::optim::StandardGroups::from_module`] automatically places `LayerNorm::weight`,
//! `RmsNorm::weight`, and `LayerScale::gamma` in the no-decay `bn` group and marks `dwconv.weight` as
//! depthwise.

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use super::conv::{Conv2d, Conv2dConfig};
use super::glu::gelu_tanh;
use super::regularize::{ChannelLayout, DropPath, LayerScale};
use super::rmsnorm::RmsNorm;
use super::simple::Linear;
use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Default numerical stability epsilon for [`LayerNorm`] (`1e-6`, ConvNeXt default).
pub const LAYERNORM_DEFAULT_EPS: f32 = 1e-6;
/// Default [`ConvNeXtBlock`] depthwise kernel size (`7`).
pub const CONVNEXT_DEFAULT_KERNEL: usize = 7;
/// Default [`ConvNeXtBlock`] pointwise expansion ratio (`4`).
pub const CONVNEXT_DEFAULT_EXPAND: usize = 4;
/// Default [`ConvNeXtBlock`] LayerScale initialization value (`1e-6`, Liu et al. 2022).
pub const CONVNEXT_DEFAULT_LAYER_SCALE: f32 = 1e-6;

/// Layer Normalization over the last dimension (`dim >= 1`; Ba, Kiros & Hinton 2016).
///
/// Delegates directly to [`Tensor::layer_norm`] with full autograd support for input, optional learnable
/// `weight: [dim]` (initialized to 1), and optional learnable `bias: [dim]` (initialized to 0).
/// [`crate::optim::StandardGroups::from_module`] automatically routes `weight` into the no-decay `bn` group
/// and `bias` into the `bias` group.
pub struct LayerNorm {
    /// Optional learnable scale `[dim]`.
    pub weight: Option<Tensor>,
    /// Optional learnable shift `[dim]`.
    pub bias: Option<Tensor>,
    dim: usize,
    eps: f32,
}

impl LayerNorm {
    /// Create a [`LayerNorm`] with learnable `weight` (1s) and `bias` (0s) over `dim >= 1` and `eps > 0`.
    pub fn new(dim: usize, eps: f32) -> Result<Self> {
        validate_layernorm(dim, eps)?;
        Ok(Self {
            weight: Some(Tensor::ones([dim]).requires_grad_()),
            bias: Some(Tensor::zeros([dim]).requires_grad_()),
            dim,
            eps,
        })
    }

    /// Create a [`LayerNorm`] without learnable affine parameters.
    pub fn without_affine(dim: usize, eps: f32) -> Result<Self> {
        validate_layernorm(dim, eps)?;
        Ok(Self { weight: None, bias: None, dim, eps })
    }

    /// Wrap existing 1-D `[dim]` tensors (`F32` or `F16`) in a [`LayerNorm`].
    pub fn from_tensors(weight: Option<Tensor>, bias: Option<Tensor>, dim: usize, eps: f32) -> Result<Self> {
        validate_layernorm(dim, eps)?;
        for (name, t) in [("weight", &weight), ("bias", &bias)] {
            if let Some(t) = t {
                if t.dims() != [dim] {
                    return Err(Error::Invalid(format!(
                        "LayerNorm: {name} shape {:?} does not match [{dim}]",
                        t.dims()
                    )));
                }
                if !matches!(t.dtype(), DType::F32 | DType::F16) {
                    return Err(Error::Invalid(format!(
                        "LayerNorm: {name} dtype must be f32 or f16, got {}",
                        t.dtype()
                    )));
                }
            }
        }
        if let (Some(w), Some(b)) = (&weight, &bias) {
            if w.dtype() != b.dtype() {
                return Err(Error::Invalid(format!(
                    "LayerNorm: weight dtype {} does not match bias dtype {}",
                    w.dtype(),
                    b.dtype()
                )));
            }
            if w.device() != b.device() {
                return Err(Error::Invalid(format!(
                    "LayerNorm: weight device {:?} does not match bias device {:?}",
                    w.device(),
                    b.device()
                )));
            }
        }
        Ok(Self { weight, bias, dim, eps })
    }

    /// Normalized feature dimension (last axis size).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Configured epsilon floor.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

fn validate_layernorm(dim: usize, eps: f32) -> Result<()> {
    if dim == 0 {
        return Err(Error::Invalid("LayerNorm: dim must be >= 1, got 0".into()));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!("LayerNorm: eps must be finite and > 0, got {eps}")));
    }
    Ok(())
}

impl Module for LayerNorm {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        if let Some(w) = &self.weight {
            f("weight", w, TensorKind::Parameter);
        }
        if let Some(b) = &self.bias {
            f("bias", b, TensorKind::Parameter);
        }
    }
}

impl Forward for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last = *x.dims().last().ok_or_else(|| Error::Invalid("LayerNorm: scalar input".into()))?;
        if last != self.dim {
            return Err(Error::Invalid(format!(
                "LayerNorm: input {:?} has last dim {last}, expected {}",
                x.dims(),
                self.dim
            )));
        }
        let widen_for_amp = x.dtype() == DType::F16
            && folo_core::autocast_dtype() == Some(DType::F16)
            && (self.weight.as_ref().is_some_and(|w| w.dtype() == DType::F32)
                || self.bias.as_ref().is_some_and(|b| b.dtype() == DType::F32));
        if widen_for_amp {
            return Ok(x
                .to_dtype(DType::F32)?
                .layer_norm(self.weight.as_ref(), self.bias.as_ref(), self.eps)?
                .to_dtype(DType::F16)?);
        }
        Ok(x.layer_norm(self.weight.as_ref(), self.bias.as_ref(), self.eps)?)
    }
}

/// Normalization choice inside [`ConvNeXtBlock`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvNeXtNorm {
    /// Channels-last [`LayerNorm`] (Liu et al. 2022 default).
    #[default]
    LayerNorm,
    /// Channels-last [`RmsNorm`].
    RmsNorm,
}

/// Configuration for [`ConvNeXtBlock`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConvNeXtConfig {
    /// Input and output channel dimension `C >= 1`.
    pub dim: usize,
    /// Odd depthwise convolution kernel size `>= 1` (default `7`).
    pub kernel_size: usize,
    /// Pointwise inverted-bottleneck expansion factor `>= 1` (default `4`).
    pub expand_ratio: usize,
    /// Normalization layer applied after the depthwise conv (default `LayerNorm`).
    pub norm: ConvNeXtNorm,
    /// Epsilon for the normalization layer (`> 0`, default `1e-6`).
    pub norm_eps: f32,
    /// Optional learnable [`LayerScale`] initial value `>= 0` (default `Some(1e-6)`).
    pub layer_scale_init: Option<f32>,
    /// Optional residual [`DropPath`] probability in `[0, 1)` (default `0.0`, disabled).
    pub drop_path: f32,
    /// Seed for [`DropPath`] when `drop_path > 0.0` (default `0`).
    pub drop_path_seed: u64,
}

impl Default for ConvNeXtConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            kernel_size: CONVNEXT_DEFAULT_KERNEL,
            expand_ratio: CONVNEXT_DEFAULT_EXPAND,
            norm: ConvNeXtNorm::LayerNorm,
            norm_eps: LAYERNORM_DEFAULT_EPS,
            layer_scale_init: Some(CONVNEXT_DEFAULT_LAYER_SCALE),
            drop_path: 0.0,
            drop_path_seed: 0,
        }
    }
}

impl ConvNeXtConfig {
    /// Create a default ConvNeXt block config for `dim` channels (`k = 7`, `expand = 4`, `LayerNorm`,
    /// `LayerScale = 1e-6`, no `DropPath`).
    pub fn new(dim: usize) -> Self {
        Self { dim, ..Default::default() }
    }

    /// Set the odd depthwise kernel size.
    pub fn kernel_size(mut self, k: usize) -> Self {
        self.kernel_size = k;
        self
    }

    /// Set the pointwise expansion ratio.
    pub fn expand_ratio(mut self, ratio: usize) -> Self {
        self.expand_ratio = ratio;
        self
    }

    /// Choose `LayerNorm` or `RmsNorm`.
    pub fn norm(mut self, norm: ConvNeXtNorm) -> Self {
        self.norm = norm;
        self
    }

    /// Set normalization epsilon.
    pub fn norm_eps(mut self, eps: f32) -> Self {
        self.norm_eps = eps;
        self
    }

    /// Set optional `LayerScale` initial value (`None` disables LayerScale).
    pub fn layer_scale_init(mut self, init: Option<f32>) -> Self {
        self.layer_scale_init = init;
        self
    }

    /// Enable residual `DropPath` with probability `p in [0, 1)` and RNG `seed`.
    pub fn drop_path(mut self, p: f32, seed: u64) -> Self {
        self.drop_path = p;
        self.drop_path_seed = seed;
        self
    }

    /// Validate all fields and return the expanded hidden dimension `dim * expand_ratio`.
    pub fn validate(&self) -> Result<usize> {
        if self.dim == 0 {
            return Err(Error::Invalid("ConvNeXtBlock: dim must be >= 1, got 0".into()));
        }
        if self.kernel_size == 0
            || self.kernel_size.is_multiple_of(2)
            || self.kernel_size.checked_mul(self.kernel_size).is_none()
        {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: kernel_size must be odd and >= 1 without overflow, got {}",
                self.kernel_size
            )));
        }
        if self.expand_ratio == 0 {
            return Err(Error::Invalid("ConvNeXtBlock: expand_ratio must be >= 1, got 0".into()));
        }
        let hidden = self.dim.checked_mul(self.expand_ratio).ok_or_else(|| {
            Error::Invalid(format!(
                "ConvNeXtBlock: dim ({}) * expand_ratio ({}) overflows usize",
                self.dim, self.expand_ratio
            ))
        })?;
        if !(self.norm_eps.is_finite() && self.norm_eps > 0.0) {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: norm_eps must be finite and > 0, got {}",
                self.norm_eps
            )));
        }
        if let Some(ls) = self.layer_scale_init
            && !(ls.is_finite() && ls >= 0.0)
        {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: layer_scale_init must be finite and >= 0, got {ls}"
            )));
        }
        if !(self.drop_path.is_finite() && (0.0..1.0).contains(&self.drop_path)) {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: drop_path must be finite and in [0, 1), got {}",
                self.drop_path
            )));
        }
        Ok(hidden)
    }
}

/// Internal normalization wrapper exposed as child `"norm"`.
pub enum ConvNeXtNormLayer {
    /// Channels-last LayerNorm.
    Layer(LayerNorm),
    /// Channels-last RmsNorm.
    Rms(RmsNorm),
}

impl Module for ConvNeXtNormLayer {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        match self {
            Self::Layer(l) => f("", l),
            Self::Rms(r) => f("", r),
        }
    }
}

impl Forward for ConvNeXtNormLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Layer(l) => l.forward(x),
            Self::Rms(r) => r.forward(x),
        }
    }
}

/// Modern ConvNeXt residual block (Liu et al., CVPR 2022).
///
/// Child module names: `dwconv`, `norm`, `pwconv1`, `pwconv2`, optional `layer_scale`, optional `drop_path`.
pub struct ConvNeXtBlock {
    /// Large-kernel depthwise convolution (`dim -> dim`, `groups = dim`, `padding = k / 2`).
    pub dwconv: Conv2d,
    /// Channels-last normalization (`LayerNorm` or `RmsNorm`).
    pub norm: ConvNeXtNormLayer,
    /// Pointwise expansion projection (`dim -> dim * expand_ratio`).
    pub pwconv1: Linear,
    /// Pointwise bottleneck projection (`dim * expand_ratio -> dim`).
    pub pwconv2: Linear,
    /// Optional learnable per-channel LayerScale (`ChannelsLast`).
    pub layer_scale: Option<LayerScale>,
    /// Optional residual stochastic depth (`DropPath`).
    pub drop_path: Option<DropPath>,
    dim: usize,
    hidden: usize,
}

impl ConvNeXtBlock {
    /// Construct a [`ConvNeXtBlock`] from `cfg` and `rng`.
    pub fn new(cfg: ConvNeXtConfig, rng: &mut Rng) -> Result<Self> {
        let hidden = cfg.validate()?;
        let dw_cfg = Conv2dConfig::new(cfg.dim, cfg.dim, cfg.kernel_size)
            .padding(cfg.kernel_size / 2)
            .groups(cfg.dim)
            .bias(true);
        let dwconv = Conv2d::try_new(dw_cfg, rng)?;
        let norm = match cfg.norm {
            ConvNeXtNorm::LayerNorm => ConvNeXtNormLayer::Layer(LayerNorm::new(cfg.dim, cfg.norm_eps)?),
            ConvNeXtNorm::RmsNorm => ConvNeXtNormLayer::Rms(RmsNorm::new(cfg.dim, cfg.norm_eps)?),
        };
        let pwconv1 = Linear::new(cfg.dim, hidden, true, rng);
        let pwconv2 = Linear::new(hidden, cfg.dim, true, rng);
        let layer_scale = cfg
            .layer_scale_init
            .map(|init| LayerScale::with_layout(cfg.dim, init, ChannelLayout::ChannelsLast))
            .transpose()?;
        let drop_path = (cfg.drop_path > 0.0)
            .then(|| DropPath::try_new(cfg.drop_path, cfg.drop_path_seed))
            .transpose()?;
        Ok(Self { dwconv, norm, pwconv1, pwconv2, layer_scale, drop_path, dim: cfg.dim, hidden })
    }

    /// Channel dimension `C`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Expanded pointwise hidden dimension `dim * expand_ratio`.
    pub fn hidden_dim(&self) -> usize {
        self.hidden
    }
}

impl Module for ConvNeXtBlock {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("dwconv", &self.dwconv);
        match &self.norm {
            ConvNeXtNormLayer::Layer(l) => f("norm", l),
            ConvNeXtNormLayer::Rms(r) => f("norm", r),
        }
        f("pwconv1", &self.pwconv1);
        f("pwconv2", &self.pwconv2);
        if let Some(ls) = &self.layer_scale {
            f("layer_scale", ls);
        }
        if let Some(dp) = &self.drop_path {
            f("drop_path", dp);
        }
    }
}

impl Forward for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !matches!(x.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: input dtype must be f32 or f16, got {}",
                x.dtype()
            )));
        }
        let (n, c, h, w) = x.dims4().map_err(|_| {
            Error::Invalid(format!("ConvNeXtBlock: input must be 4-D NCHW, got {:?}", x.dims()))
        })?;
        if c != self.dim {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: input has {c} channels, expected {}",
                self.dim
            )));
        }
        if h == 0 || w == 0 {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: spatial dims must be >= 1, got [{n}, {c}, {h}, {w}]"
            )));
        }
        if x.device() != self.dwconv.weight.device() {
            return Err(Error::Invalid(format!(
                "ConvNeXtBlock: input on {:?} does not match weight on {:?}",
                x.device(),
                self.dwconv.weight.device()
            )));
        }
        if n == 0 {
            let dtype = x.dtype();
            let dev = x.device();
            let shape = x.shape().clone();
            let out_t = Tensor::zeros_with(&shape, dtype, dev)?;
            return Ok(Tensor::custom_op("convnext_empty", &[x], out_t, move |_g| {
                Ok(vec![Some(Tensor::zeros_with(&shape, dtype, dev)?)])
            })?);
        }

        let out = self.dwconv.forward(x)?;
        let out = out.permute([0, 2, 3, 1])?.contiguous()?;
        let out = self.norm.forward(&out)?;
        let out = self.pwconv1.forward(&out)?;
        let out = if out.device().is_cpu() { gelu_tanh(&out)? } else { gelu_tanh_device(&out)? };
        let out = self.pwconv2.forward(&out)?;
        let out = match &self.layer_scale {
            Some(ls) => ls.forward(&out)?,
            None => out,
        };
        let out = out.permute([0, 3, 1, 2])?.contiguous()?;
        let out = match &self.drop_path {
            Some(dp) => dp.forward(&out)?,
            None => out,
        };
        let out = if out.dtype() != x.dtype() { out.to_dtype(x.dtype())? } else { out };
        Ok(x.add(&out)?)
    }
}

/// Device-native tanh-GELU (`x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))`) using pure on-device
/// tensor ops with zero host readback (safe on browser WebGPU as well as native GPU / CPU).
pub(super) fn gelu_tanh_device(x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::F16 {
        return Ok(gelu_tanh_device(&x.to_dtype(DType::F32)?)?.to_dtype(DType::F16)?);
    }
    // Clamp only inside sigmoid's argument so `x^3` never overflows for large |x| while keeping
    // `sigmoid(2u) = 1.0` for `x > 12` and `0.0` for `x < -12` (since `sigmoid(-142.3) == 0.0f32`).
    let xc = x.clamp(-12.0, 12.0)?;
    let two_sqrt_2_over_pi = (2.0 * (2.0 / std::f64::consts::PI).sqrt()) as f32;
    let two_u =
        xc.mul(&xc.square()?.mul_scalar(0.044_715)?.add_scalar(1.0)?)?.mul_scalar(two_sqrt_2_over_pi)?;
    let gate = two_u.sigmoid()?;
    let y = x.mul(&gate)?;
    let is_nan = x.ne(x)?;
    Ok(is_nan.where_cond(x, &y)?)
}
