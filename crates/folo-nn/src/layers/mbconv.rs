//! Inverted Residual / MBConv block (MobileNetV2, Sandler et al. CVPR 2018; EfficientNet, Tan & Le
//! ICML 2019).
//!
//! ```text
//! [expand:  ConvBnAct(in -> hidden, 1×1, act)]    (omitted when expand_ratio == 1)
//!  dwconv:  ConvBnAct(hidden -> hidden, k×k, stride=s, groups=hidden, act)
//! [se:      SqueezeExcitation(hidden, reduction, act)]
//!  project: ConvBnAct(hidden -> out, 1×1, Identity)
//!  output = x + drop_path(h)  if stride == 1 && in == out  else  h
//! ```
//!
//! Supports full autograd on inputs and all trainable parameters, optional Squeeze-and-Excitation channel
//! attention, optional residual [`DropPath`], BatchNorm fusion via [`InvertedResidual::fuse_`], and clean
//! dotted names for `.folo` state dict round-trips.

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};

use super::attention::SqueezeExcitation;
use super::conv_bn_act::ConvBnAct;
use super::regularize::DropPath;
use super::simple::Activation;
use crate::module::{Forward, Module};
use crate::{Error, Result};

/// Configuration for [`InvertedResidual`] (MBConv).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvertedResidualConfig {
    /// Input channels `>= 1`.
    pub in_channels: usize,
    /// Output channels `>= 1`.
    pub out_channels: usize,
    /// Depthwise convolution stride `>= 1` (default `1`).
    pub stride: usize,
    /// Expansion ratio `>= 1` (`1` omits the expansion 1×1 stage; default `6`).
    pub expand_ratio: usize,
    /// Odd depthwise convolution kernel size `>= 1` (default `3`).
    pub kernel_size: usize,
    /// Optional Squeeze-and-Excitation reduction ratio (`>= 1`; `None` disables SE).
    pub se_reduction: Option<usize>,
    /// Activation used in expansion, depthwise conv, and SE bottleneck (default `Activation::Silu`).
    pub activation: Activation,
    /// Optional residual [`DropPath`] probability in `[0, 1)` (applied only when `stride == 1 && in == out`;
    /// default `0.0`).
    pub drop_path: f32,
    /// RNG seed for [`DropPath`] (default `0`).
    pub drop_path_seed: u64,
}

impl InvertedResidualConfig {
    /// Create a default MBConv configuration (`in -> out`, `stride = 1`, `expand_ratio = 6`,
    /// `kernel_size = 3`, no SE, `SiLU`, no `DropPath`).
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            stride: 1,
            expand_ratio: 6,
            kernel_size: 3,
            se_reduction: None,
            activation: Activation::Silu,
            drop_path: 0.0,
            drop_path_seed: 0,
        }
    }

    /// Set the depthwise stride.
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }

    /// Set the expansion ratio (`1` omits the 1×1 expansion layer).
    pub fn expand_ratio(mut self, ratio: usize) -> Self {
        self.expand_ratio = ratio;
        self
    }

    /// Set the odd depthwise kernel size.
    pub fn kernel_size(mut self, k: usize) -> Self {
        self.kernel_size = k;
        self
    }

    /// Enable Squeeze-and-Excitation with the given channel reduction ratio (`>= 1`).
    pub fn se_reduction(mut self, reduction: Option<usize>) -> Self {
        self.se_reduction = reduction;
        self
    }

    /// Set the block activation (`SiLU`, `ReLU`, `Hardswish`, ...).
    pub fn activation(mut self, act: Activation) -> Self {
        self.activation = act;
        self
    }

    /// Set residual `DropPath` probability `p in [0, 1)` and RNG `seed`.
    pub fn drop_path(mut self, p: f32, seed: u64) -> Self {
        self.drop_path = p;
        self.drop_path_seed = seed;
        self
    }

    /// Validate all fields and return `(hidden_channels, use_residual)`.
    pub fn validate(&self) -> Result<(usize, bool)> {
        if self.in_channels == 0 || self.out_channels == 0 {
            return Err(Error::Invalid(format!(
                "InvertedResidual: channels must be >= 1, got in={}, out={}",
                self.in_channels, self.out_channels
            )));
        }
        if self.stride == 0 {
            return Err(Error::Invalid("InvertedResidual: stride must be >= 1, got 0".into()));
        }
        if self.expand_ratio == 0 {
            return Err(Error::Invalid("InvertedResidual: expand_ratio must be >= 1, got 0".into()));
        }
        let hidden = self.in_channels.checked_mul(self.expand_ratio).ok_or_else(|| {
            Error::Invalid(format!(
                "InvertedResidual: in_channels ({}) * expand_ratio ({}) overflows usize",
                self.in_channels, self.expand_ratio
            ))
        })?;
        if self.kernel_size == 0
            || self.kernel_size.is_multiple_of(2)
            || self.kernel_size.checked_mul(self.kernel_size).is_none()
        {
            return Err(Error::Invalid(format!(
                "InvertedResidual: kernel_size must be odd and >= 1 without overflow, got {}",
                self.kernel_size
            )));
        }
        if let Some(r) = self.se_reduction
            && r == 0
        {
            return Err(Error::Invalid("InvertedResidual: se_reduction must be >= 1, got 0".into()));
        }
        if !(self.drop_path.is_finite() && (0.0..1.0).contains(&self.drop_path)) {
            return Err(Error::Invalid(format!(
                "InvertedResidual: drop_path must be finite and in [0, 1), got {}",
                self.drop_path
            )));
        }
        let use_residual = self.stride == 1 && self.in_channels == self.out_channels;
        Ok((hidden, use_residual))
    }
}

/// MobileNetV2 / EfficientNet MBConv inverted residual block.
///
/// Child module names: optional `expand`, `dwconv`, optional `se`, `project`, optional `drop_path`.
pub struct InvertedResidual {
    /// Optional 1×1 expansion stage (`None` when `expand_ratio == 1`).
    pub expand: Option<ConvBnAct>,
    /// Depthwise `k×k` convolution stage (`groups = hidden`).
    pub dwconv: ConvBnAct,
    /// Optional Squeeze-and-Excitation channel attention stage.
    pub se: Option<SqueezeExcitation>,
    /// Linear 1×1 projection stage (`Activation::Identity`).
    pub project: ConvBnAct,
    /// Optional residual [`DropPath`] (only present when `use_residual && drop_path > 0.0`).
    pub drop_path: Option<DropPath>,
    in_channels: usize,
    out_channels: usize,
    hidden_channels: usize,
    stride: usize,
    kernel_size: usize,
    use_residual: bool,
}

impl InvertedResidual {
    /// Construct an [`InvertedResidual`] block from `cfg` and `rng`.
    pub fn new(cfg: InvertedResidualConfig, rng: &mut Rng) -> Result<Self> {
        let (hidden, use_residual) = cfg.validate()?;
        let expand = (cfg.expand_ratio != 1)
            .then(|| ConvBnAct::new(cfg.in_channels, hidden, 1, 1, 1, cfg.activation, rng));
        let dwconv = ConvBnAct::new(hidden, hidden, cfg.kernel_size, cfg.stride, hidden, cfg.activation, rng);
        let se = cfg
            .se_reduction
            .map(|r| SqueezeExcitation::with_hidden(hidden, (hidden / r).max(1), cfg.activation, rng))
            .transpose()?;
        let project = ConvBnAct::new(hidden, cfg.out_channels, 1, 1, 1, Activation::Identity, rng);
        let drop_path = (use_residual && cfg.drop_path > 0.0)
            .then(|| DropPath::try_new(cfg.drop_path, cfg.drop_path_seed))
            .transpose()?;
        Ok(Self {
            expand,
            dwconv,
            se,
            project,
            drop_path,
            in_channels: cfg.in_channels,
            out_channels: cfg.out_channels,
            hidden_channels: hidden,
            stride: cfg.stride,
            kernel_size: cfg.kernel_size,
            use_residual,
        })
    }

    /// Input channel count.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Output channel count.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Expanded bottleneck channel count (`in_channels * expand_ratio`).
    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    /// Depthwise kernel size `k`.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// True when the block adds the identity residual shortcut (`stride == 1 && in == out`).
    pub fn use_residual(&self) -> bool {
        self.use_residual
    }

    /// Fold all BatchNorm layers into their preceding convolutions in place for inference export.
    pub fn fuse_(&mut self) -> Result<()> {
        if let Some(e) = &mut self.expand {
            e.fuse_()?;
        }
        self.dwconv.fuse_()?;
        self.project.fuse_()?;
        Ok(())
    }
}

impl Module for InvertedResidual {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        if let Some(e) = &self.expand {
            f("expand", e);
        }
        f("dwconv", &self.dwconv);
        if let Some(se) = &self.se {
            f("se", se);
        }
        f("project", &self.project);
        if let Some(dp) = &self.drop_path {
            f("drop_path", dp);
        }
    }
}

impl Forward for InvertedResidual {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !matches!(x.dtype(), DType::F32 | DType::F16) {
            return Err(Error::Invalid(format!(
                "InvertedResidual: input dtype must be f32 or f16, got {}",
                x.dtype()
            )));
        }
        let (n, c, h, w) = x.dims4().map_err(|_| {
            Error::Invalid(format!("InvertedResidual: input must be 4-D NCHW, got {:?}", x.dims()))
        })?;
        if c != self.in_channels {
            return Err(Error::Invalid(format!(
                "InvertedResidual: input has {c} channels, expected {}",
                self.in_channels
            )));
        }
        if h == 0 || w == 0 {
            return Err(Error::Invalid(format!(
                "InvertedResidual: spatial dims must be >= 1, got [{n}, {c}, {h}, {w}]"
            )));
        }
        if x.device() != self.dwconv.conv.weight.device() {
            return Err(Error::Invalid(format!(
                "InvertedResidual: input on {:?} does not match weight on {:?}",
                x.device(),
                self.dwconv.conv.weight.device()
            )));
        }
        if n == 0 {
            let h_out = (h - 1) / self.stride + 1;
            let w_out = (w - 1) / self.stride + 1;
            let dtype = x.dtype();
            let dev = x.device();
            let in_shape = x.shape().clone();
            let out_t = Tensor::zeros_with([0, self.out_channels, h_out, w_out], dtype, dev)?;
            return Ok(Tensor::custom_op("mbconv_empty", &[x], out_t, move |_g| {
                Ok(vec![Some(Tensor::zeros_with(&in_shape, dtype, dev)?)])
            })?);
        }

        let out = match &self.expand {
            Some(e) => e.forward(x)?,
            None => x.clone(),
        };
        let out = self.dwconv.forward(&out)?;
        let out = match &self.se {
            Some(se) => se.forward(&out)?,
            None => out,
        };
        let out = self.project.forward(&out)?;
        if self.use_residual {
            let out = match &self.drop_path {
                Some(dp) => dp.forward(&out)?,
                None => out,
            };
            let out = if out.dtype() != x.dtype() { out.to_dtype(x.dtype())? } else { out };
            Ok(x.add(&out)?)
        } else {
            Ok(out)
        }
    }
}
