//! Re-parameterizable structural convolution block (`RepConv` / RepVGG; Ding et al., CVPR 2021 /
//! arXiv:2101.03697).
//!
//! During training, [`RepConv`] computes a multi-branch residual sum before the activation:
//! ```text
//! y = act( BN_3x3(Conv3x3(x)) + BN_1x1(Conv1x1(x)) + [BN_id(x)] )
//! ```
//! where the optional identity BatchNorm branch is active whenever `in_channels == out_channels && stride == 1`
//! and `use_identity == true`.
//!
//! Calling [`RepConv::switch_to_deploy`] algebraically fuses all branches (including grouped and depthwise
//! convolutions) in `f64` into a single `3×3` [`Conv2d`] with bias (`rbr_reparam`), producing inference
//! outputs numerically equivalent to the eval-mode multi-branch block with zero extra latency.
//!
//! Tensor names:
//! - Training (multi-branch): `rbr_dense.conv.weight`, `rbr_dense.bn.{weight,bias,running_mean,running_var}`,
//!   `rbr_1x1.conv.weight`, `rbr_1x1.bn.{weight,bias,running_mean,running_var}`, and optional
//!   `rbr_identity.{weight,bias,running_mean,running_var}`.
//! - Deploy (fused): `rbr_reparam.weight`, `rbr_reparam.bias`.

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};

use super::batchnorm::{BN_EPS, BN_MOMENTUM, BatchNorm2d};
use super::conv::{Conv2d, Conv2dConfig};
use super::conv_bn_act::ConvBnAct;
use super::simple::Activation;
use crate::module::{Forward, Module};
use crate::{Error, Result};

/// Hyper-parameters for constructing a [`RepConv`] block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepConvConfig {
    /// Input channels (`>= 1`, divisible by `groups`).
    pub in_channels: usize,
    /// Output channels (`>= 1`, divisible by `groups`).
    pub out_channels: usize,
    /// Convolution stride (`>= 1`, default `1`).
    pub stride: usize,
    /// Convolution groups (`>= 1`, default `1`; set `== in_channels == out_channels` for depthwise).
    pub groups: usize,
    /// Enable the identity BatchNorm skip branch when `in_channels == out_channels && stride == 1`
    /// (default `true`).
    pub use_identity: bool,
    /// Post-sum activation (default [`Activation::Relu`]).
    pub act: Activation,
    /// BatchNorm epsilon (default [`BN_EPS`]).
    pub bn_eps: f64,
    /// BatchNorm running-stats momentum (default [`BN_MOMENTUM`]).
    pub bn_momentum: f64,
}

impl RepConvConfig {
    /// Default `RepConv(in_channels -> out_channels)` with stride 1, groups 1, identity branch enabled when
    /// shapes match, and ReLU activation.
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            stride: 1,
            groups: 1,
            use_identity: true,
            act: Activation::Relu,
            bn_eps: BN_EPS,
            bn_momentum: BN_MOMENTUM,
        }
    }

    /// Set the stride.
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }

    /// Set the number of groups.
    pub fn groups(mut self, g: usize) -> Self {
        self.groups = g;
        self
    }

    /// Enable or disable the identity BatchNorm branch.
    pub fn use_identity(mut self, on: bool) -> Self {
        self.use_identity = on;
        self
    }

    /// Set the post-sum activation.
    pub fn act(mut self, act: Activation) -> Self {
        self.act = act;
        self
    }

    /// Set BatchNorm `eps` and `momentum`.
    pub fn bn_options(mut self, eps: f64, momentum: f64) -> Self {
        self.bn_eps = eps;
        self.bn_momentum = momentum;
        self
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.in_channels == 0
            || self.out_channels == 0
            || self.stride == 0
            || self.groups == 0
            || !self.in_channels.is_multiple_of(self.groups)
            || !self.out_channels.is_multiple_of(self.groups)
        {
            return Err(Error::Invalid(format!("RepConv: invalid config {self:?}")));
        }
        if !(self.bn_eps.is_finite() && self.bn_eps > 0.0) {
            return Err(Error::Invalid(format!(
                "RepConv: bn_eps must be finite and > 0, got {}",
                self.bn_eps
            )));
        }
        if !(self.bn_momentum.is_finite() && (0.0..=1.0).contains(&self.bn_momentum)) {
            return Err(Error::Invalid(format!(
                "RepConv: bn_momentum must be finite and in [0, 1], got {}",
                self.bn_momentum
            )));
        }
        Ok(())
    }

    fn conv3x3_cfg(&self, bias: bool) -> Conv2dConfig {
        Conv2dConfig::new(self.in_channels, self.out_channels, 3)
            .stride(self.stride)
            .padding(1)
            .groups(self.groups)
            .bias(bias)
    }

    fn conv1x1_cfg(&self) -> Conv2dConfig {
        Conv2dConfig::new(self.in_channels, self.out_channels, 1)
            .stride(self.stride)
            .padding(0)
            .groups(self.groups)
            .bias(false)
    }
}

/// Re-parameterizable RepVGG / RepConv block supporting multi-branch training and single-`3×3` fused
/// deployment via [`RepConv::switch_to_deploy`].
pub struct RepConv {
    /// `3×3` Conv + BN branch (`None` after [`RepConv::switch_to_deploy`]).
    pub rbr_dense: Option<ConvBnAct>,
    /// `1×1` Conv + BN branch (`None` after [`RepConv::switch_to_deploy`]).
    pub rbr_1x1: Option<ConvBnAct>,
    /// Optional identity BN branch (`Some` only when `in_channels == out_channels && stride == 1 && use_identity`
    /// and not yet deployed).
    pub rbr_identity: Option<BatchNorm2d>,
    /// Single fused `3×3` Conv with bias (`Some` after [`RepConv::switch_to_deploy`] or [`RepConv::new_deploy`]).
    pub rbr_reparam: Option<Conv2d>,
    /// Post-sum activation.
    pub act: Activation,
    config: RepConvConfig,
}

impl RepConv {
    /// Create a multi-branch training-time [`RepConv`] block initialized from `rng`.
    pub fn new(config: RepConvConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let rbr_dense = ConvBnAct::with_config(
            config.conv3x3_cfg(false),
            Activation::Identity,
            config.bn_eps,
            config.bn_momentum,
            rng,
        );
        let rbr_1x1 = ConvBnAct::with_config(
            config.conv1x1_cfg(),
            Activation::Identity,
            config.bn_eps,
            config.bn_momentum,
            rng,
        );
        let rbr_identity =
            (config.use_identity && config.in_channels == config.out_channels && config.stride == 1)
                .then(|| BatchNorm2d::with_options(config.in_channels, config.bn_eps, config.bn_momentum));
        Ok(Self {
            rbr_dense: Some(rbr_dense),
            rbr_1x1: Some(rbr_1x1),
            rbr_identity,
            rbr_reparam: None,
            act: config.act,
            config,
        })
    }

    /// Create a [`RepConv`] block directly in deploy (fused single-`3×3`) layout, e.g. for loading saved
    /// deploy `.folo` weights.
    pub fn new_deploy(config: RepConvConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let conv_cfg = config.conv3x3_cfg(true);
        let init = Conv2d::try_new(conv_cfg, rng)?;
        let conv = Conv2d::from_tensors(conv_cfg, init.weight.detach(), init.bias.map(|b| b.detach()))?;
        Ok(Self {
            rbr_dense: None,
            rbr_1x1: None,
            rbr_identity: None,
            rbr_reparam: Some(conv),
            act: config.act,
            config,
        })
    }

    /// Block configuration.
    #[inline]
    pub fn config(&self) -> RepConvConfig {
        self.config
    }

    /// `true` once fused into a single `3×3` conv (`rbr_reparam`).
    #[inline]
    pub fn is_deployed(&self) -> bool {
        self.rbr_reparam.is_some()
    }

    /// Compute the equivalent fused `(weight: [O, I/g, 3, 3], bias: [O])` in `f64` without mutating `self`.
    pub fn equivalent_kernel_bias(&self) -> Result<(Tensor, Tensor)> {
        if let Some(reparam) = &self.rbr_reparam {
            let b = match &reparam.bias {
                Some(b) => b.detach(),
                None => Tensor::zeros_with(
                    [self.config.out_channels],
                    reparam.weight.dtype(),
                    reparam.weight.device(),
                )?,
            };
            return Ok((reparam.weight.detach(), b));
        }
        let dense =
            self.rbr_dense.as_ref().ok_or_else(|| Error::Invalid("RepConv: missing rbr_dense".into()))?;
        let b1x1 = self.rbr_1x1.as_ref().ok_or_else(|| Error::Invalid("RepConv: missing rbr_1x1".into()))?;

        let o = self.config.out_channels;
        let in_per_g = self.config.in_channels / self.config.groups;
        let dtype = dense.conv.weight.dtype();
        let dev = dense.conv.weight.device();

        let (mut w_acc, mut b_acc) = fuse_branch_f64(dense, o, in_per_g, 3)?;
        let (w1, b1) = fuse_branch_f64(b1x1, o, in_per_g, 1)?;
        for oc in 0..o {
            b_acc[oc] += b1[oc];
            for ic in 0..in_per_g {
                let dst = ((oc * in_per_g + ic) * 3 + 1) * 3 + 1;
                w_acc[dst] += w1[oc * in_per_g + ic];
            }
        }

        if let Some(bn_id) = &self.rbr_identity {
            if self.config.in_channels != self.config.out_channels || self.config.stride != 1 {
                return Err(Error::Invalid(format!(
                    "RepConv: rbr_identity requires in_channels == out_channels and stride == 1, got in={} out={} stride={}",
                    self.config.in_channels, self.config.out_channels, self.config.stride
                )));
            }
            let (gamma, beta, rm, rv) = read_bn_f64(bn_id, o)?;
            for oc in 0..o {
                let k = gamma[oc] / (rv[oc] + bn_id.eps).sqrt();
                let ic = oc % in_per_g;
                let dst = ((oc * in_per_g + ic) * 3 + 1) * 3 + 1;
                w_acc[dst] += k;
                b_acc[oc] += beta[oc] - rm[oc] * k;
            }
        }

        let w_f32: Vec<f32> = w_acc.into_iter().map(|v| v as f32).collect();
        let b_f32: Vec<f32> = b_acc.into_iter().map(|v| v as f32).collect();
        let w_t = Tensor::from_vec(w_f32, [o, in_per_g, 3, 3])?.to_dtype(dtype)?;
        let b_t = Tensor::from_vec(b_f32, [o])?.to_dtype(dtype)?;
        let to_dev =
            |t: Tensor| -> Result<Tensor> { Ok(if dev == Device::Cpu { t } else { t.to_device(dev)? }) };
        Ok((to_dev(w_t)?, to_dev(b_t)?))
    }

    /// Algebraically fuse all branches into a single `3×3` [`Conv2d`] with bias (`rbr_reparam`) in place.
    ///
    /// Idempotent: calling on an already-deployed block is a no-op.
    pub fn switch_to_deploy(&mut self) -> Result<()> {
        if self.is_deployed() {
            return Ok(());
        }
        let (w, b) = self.equivalent_kernel_bias()?;
        let conv = Conv2d::from_tensors(self.config.conv3x3_cfg(true), w, Some(b))?;
        self.rbr_dense = None;
        self.rbr_1x1 = None;
        self.rbr_identity = None;
        self.rbr_reparam = Some(conv);
        Ok(())
    }
}

type BnStatsF64 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn read_bn_f64(bn: &BatchNorm2d, o: usize) -> Result<BnStatsF64> {
    let to_f64 = |t: &Tensor, label: &str| -> Result<Vec<f64>> {
        if t.dims() != [o] {
            return Err(Error::Invalid(format!("RepConv: bn.{label} shape {:?} != [{o}]", t.dims())));
        }
        Ok(t.to_dtype(DType::F32)?.to_vec::<f32>()?.into_iter().map(f64::from).collect())
    };
    Ok((
        to_f64(&bn.weight, "weight")?,
        to_f64(&bn.bias, "bias")?,
        to_f64(&bn.running_mean, "running_mean")?,
        to_f64(&bn.running_var, "running_var")?,
    ))
}

fn fuse_branch_f64(
    branch: &ConvBnAct,
    o: usize,
    in_per_g: usize,
    k_size: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if branch.act != Activation::Identity {
        return Err(Error::Invalid(format!(
            "RepConv: cannot fuse branch with non-identity activation {:?}",
            branch.act
        )));
    }
    let expected = [o, in_per_g, k_size, k_size];
    if branch.conv.weight.dims() != expected {
        return Err(Error::Invalid(format!(
            "RepConv: branch conv weight {:?} != {expected:?}",
            branch.conv.weight.dims()
        )));
    }
    let mut w: Vec<f64> =
        branch.conv.weight.to_dtype(DType::F32)?.to_vec::<f32>()?.into_iter().map(f64::from).collect();
    let cb: Option<Vec<f64>> = match &branch.conv.bias {
        Some(b) => Some(b.to_dtype(DType::F32)?.to_vec::<f32>()?.into_iter().map(f64::from).collect()),
        None => None,
    };
    let per = in_per_g * k_size * k_size;
    let mut b_out = vec![0.0f64; o];
    if let Some(bn) = &branch.bn {
        let (gamma, beta, rm, rv) = read_bn_f64(bn, o)?;
        for oc in 0..o {
            let k = gamma[oc] / (rv[oc] + bn.eps).sqrt();
            for v in &mut w[oc * per..(oc + 1) * per] {
                *v *= k;
            }
            let conv_b = cb.as_ref().map_or(0.0, |v| v[oc]);
            b_out[oc] = beta[oc] + (conv_b - rm[oc]) * k;
        }
    } else if let Some(cb) = cb {
        b_out.copy_from_slice(&cb);
    }
    Ok((w, b_out))
}

impl Module for RepConv {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        if let Some(reparam) = &self.rbr_reparam {
            f("rbr_reparam", reparam);
        } else {
            if let Some(dense) = &self.rbr_dense {
                f("rbr_dense", dense);
            }
            if let Some(b1x1) = &self.rbr_1x1 {
                f("rbr_1x1", b1x1);
            }
            if let Some(id) = &self.rbr_identity {
                f("rbr_identity", id);
            }
        }
    }
}

fn conv2d_act_autocast(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    p: &folo_core::Conv2dParams,
    act: Option<folo_core::UnaryOp>,
) -> Result<Tensor> {
    if folo_core::autocast_dtype() == Some(DType::F16)
        && !x.device().is_cpu()
        && !(folo_core::is_grad_enabled()
            && (x.requires_grad() || w.requires_grad() || b.is_some_and(Tensor::requires_grad)))
        && folo_core::backend::backend(x.device())?.computes_f16()
    {
        let x16 = x.to_dtype(DType::F16)?;
        let w16 = w.to_dtype(DType::F16)?;
        let b16 = b.map(|t| t.to_dtype(DType::F16)).transpose()?;
        return Ok(x16.conv2d_act(&w16, b16.as_ref(), p, act)?);
    }
    Ok(x.conv2d_act(w, b, p, act)?)
}

impl Forward for RepConv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(reparam) = &self.rbr_reparam {
            let p = reparam.config().params();
            return conv2d_act_autocast(x, &reparam.weight, reparam.bias.as_ref(), &p, self.act.unary_op());
        }
        let dense =
            self.rbr_dense.as_ref().ok_or_else(|| Error::Invalid("RepConv: missing rbr_dense".into()))?;
        let b1x1 = self.rbr_1x1.as_ref().ok_or_else(|| Error::Invalid("RepConv: missing rbr_1x1".into()))?;
        let mut y = dense.forward(x)?.add(&b1x1.forward(x)?)?;
        if let Some(id) = &self.rbr_identity {
            let x_id = if !id.is_training() && x.dtype() != id.weight.dtype() {
                x.to_dtype(id.weight.dtype())?
            } else {
                x.clone()
            };
            let mut id_out = id.forward(&x_id)?;
            if id_out.dtype() != y.dtype() {
                id_out = id_out.to_dtype(y.dtype())?;
            }
            y = y.add(&id_out)?;
        }
        self.act.forward(&y)
    }
}
