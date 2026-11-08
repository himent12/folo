//! `ConvBnAct`, docs/MODEL.md's `Conv` block (§2.1), and Conv+BN fusion (§9.1).

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, is_grad_enabled};

use super::batchnorm::{BN_EPS, BN_MOMENTUM, BatchNorm2d};
use super::conv::{Conv2d, Conv2dConfig};
use super::simple::Activation;
use crate::module::{Forward, Module};
use crate::{Error, Result};

/// docs/MODEL.md's `Conv(c1→c2, k, s, g, act)`: `act(BN(conv2d(x, W, stride=s, padding=k/2, groups=g)))` with
/// a bias-free conv.
///
/// Tensor names: `conv.weight`, `bn.weight`, `bn.bias`, `bn.running_mean`, `bn.running_var`. After
/// [`ConvBnAct::fuse_`] (or built with [`ConvBnAct::new_fused`]) the BatchNorm is folded into the conv and the
/// names are `conv.weight`, `conv.bias` (docs/MODEL.md §5, fused files).
pub struct ConvBnAct {
    /// The convolution (no bias unless fused).
    pub conv: Conv2d,
    /// The BatchNorm; `None` once fused.
    pub bn: Option<BatchNorm2d>,
    /// The activation applied last.
    pub act: Activation,
}

impl ConvBnAct {
    /// `Conv(c1→c2, k, s, g=groups, act)` with padding `k / 2`, docs/MODEL.md's BatchNorm (`eps = 1e-3`,
    /// `momentum = 0.03`) and the conv weight drawn from `rng` (§2.9).
    pub fn new(
        c1: usize,
        c2: usize,
        k: usize,
        s: usize,
        groups: usize,
        act: Activation,
        rng: &mut Rng,
    ) -> Self {
        let cfg = Conv2dConfig::new(c1, c2, k).stride(s).groups(groups).bias(false);
        Self::with_config(cfg, act, BN_EPS, BN_MOMENTUM, rng)
    }

    /// From a conv config (its `bias` flag is ignored: the block's conv has none) and BatchNorm options.
    pub fn with_config(cfg: Conv2dConfig, act: Activation, eps: f64, momentum: f64, rng: &mut Rng) -> Self {
        let conv = Conv2d::new(cfg.bias(false), rng);
        let bn = BatchNorm2d::with_options(cfg.out_channels, eps, momentum);
        ConvBnAct { conv, bn: Some(bn), act }
    }

    /// The fused structure (conv with bias, no BatchNorm), for loading `folo.fused = "true"` files. The conv
    /// is initialised from `rng` like [`ConvBnAct::new`] (weight, then bias).
    pub fn new_fused(
        c1: usize,
        c2: usize,
        k: usize,
        s: usize,
        groups: usize,
        act: Activation,
        rng: &mut Rng,
    ) -> Self {
        let cfg = Conv2dConfig::new(c1, c2, k).stride(s).groups(groups).bias(true);
        ConvBnAct { conv: Conv2d::new(cfg, rng), bn: None, act }
    }

    /// True once the BatchNorm has been folded into the conv.
    pub fn is_fused(&self) -> bool {
        self.bn.is_none()
    }

    /// Fold the BatchNorm into the conv (in `F32`, with its running statistics), in place. A no-op if already
    /// fused. The result matches eval-mode output of the unfused block.
    pub fn fuse_(&mut self) -> Result<()> {
        let Some(bn) = self.bn.take() else { return Ok(()) };
        let (w, b) = fuse_conv_bn(
            &self.conv.weight,
            self.conv.bias.as_ref(),
            &bn.weight,
            &bn.bias,
            &bn.running_mean,
            &bn.running_var,
            bn.eps,
        )?;
        self.conv = Conv2d::from_tensors(*self.conv.config(), w, Some(b))?;
        Ok(())
    }

    /// `forward(&Tensor::cat(xs, 1))`. When the conv runs as one fused kernel (fused block, or eval mode with
    /// the BatchNorm folded on the fly, as in [`Forward::forward`]), it reads the parts in place through
    /// [`Tensor::conv2d_act_cat`] and the concatenation is never written (on backends with that kernel).
    pub fn forward_cat(&self, xs: &[&Tensor]) -> Result<Tensor> {
        let first = *xs.first().ok_or_else(|| Error::Invalid("ConvBnAct::forward_cat: no inputs".into()))?;
        let p = self.conv.config().params();
        let act = self.act.unary_op();
        match &self.bn {
            None => Ok(Tensor::conv2d_act_cat(xs, &self.conv.weight, self.conv.bias.as_ref(), &p, act)?),
            Some(bn) if !bn.is_training() && (!first.device().is_cpu() || !is_grad_enabled()) => {
                let (w, b) = self.folded_eval(bn)?;
                Ok(Tensor::conv2d_act_cat(xs, &w, Some(&b), &p, act)?)
            }
            Some(_) => self.forward(&Tensor::cat(xs, 1)?),
        }
    }

    /// The eval-mode BatchNorm folded into the conv on the fly: `(W · k, β − mean · k (+ b · k))`, small
    /// per-channel ops that stay differentiable.
    fn folded_eval(&self, bn: &BatchNorm2d) -> Result<(Tensor, Tensor)> {
        let (scale, shift) = bn.eval_scale_shift()?;
        let o = scale.numel();
        let w = self.conv.weight.mul(&scale.reshape([o, 1, 1, 1])?)?;
        let b = match &self.conv.bias {
            Some(cb) => cb.mul(&scale)?.add(&shift)?,
            None => shift,
        };
        Ok((w, b))
    }
}

impl Module for ConvBnAct {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("conv", &self.conv);
        if let Some(bn) = &self.bn {
            f("bn", bn);
        }
    }
}

impl Forward for ConvBnAct {
    /// - Fused (no BatchNorm): `act(conv2d(x) + bias)` through [`Tensor::conv2d_act`], one kernel with bias and
    ///   activation in the epilogue on the GPU when nothing is recorded.
    /// - Training: the conv, then [`BatchNorm2d::forward_act`] (BatchNorm + activation fused on the GPU).
    /// - Evaluation on a device other than the CPU, or on the CPU when nothing is recorded: the BatchNorm is
    ///   folded into the conv weights on the fly (`W · k`, `β − mean · k`, small per-channel ops,
    ///   differentiable) and the conv runs fused as above (the CPU applies bias + activation as its GEMM
    ///   epilogue).
    /// - Evaluation on the CPU with autograd recording: conv, BatchNorm, activation.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.conv.config().params();
        let act = self.act.unary_op();
        match &self.bn {
            None => Ok(x.conv2d_act(&self.conv.weight, self.conv.bias.as_ref(), &p, act)?),
            Some(bn) if bn.is_training() => bn.forward_act(&self.conv.forward(x)?, self.act),
            Some(bn) if !x.device().is_cpu() || !is_grad_enabled() => {
                let (w, b) = self.folded_eval(bn)?;
                Ok(x.conv2d_act(&w, Some(&b), &p, act)?)
            }
            Some(bn) => self.act.forward(&bn.forward(&self.conv.forward(x)?)?),
        }
    }
}

/// Fold a BatchNorm into the preceding convolution, in `F32` (docs/MODEL.md §9.1):
/// `k = γ / sqrt(running_var + eps)`, `W' = W · k` per output channel, `b' = β + (b − running_mean) · k`
/// (`b = 0` for a bias-free conv, giving `β − running_mean · k`). Returns new tensors `(W', b')` on the
/// weight's device; the inputs are not modified and the outputs do not require grad.
pub fn fuse_conv_bn(
    weight: &Tensor,
    conv_bias: Option<&Tensor>,
    gamma: &Tensor,
    beta: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let o = *weight.dims().first().ok_or_else(|| Error::Invalid("fuse_conv_bn: scalar weight".into()))?;
    let vec = |t: &Tensor, what: &str| -> Result<Vec<f32>> {
        if t.dims() != [o] {
            return Err(Error::Invalid(format!("fuse_conv_bn: {what} is {:?}, expected [{o}]", t.dims())));
        }
        Ok(t.to_dtype(DType::F32)?.to_vec::<f32>()?)
    };
    let (g, be, rm, rv) = (
        vec(gamma, "weight")?,
        vec(beta, "bias")?,
        vec(running_mean, "running_mean")?,
        vec(running_var, "running_var")?,
    );
    let cb = conv_bias.map(|b| vec(b, "conv bias")).transpose()?;
    let mut w = weight.to_dtype(DType::F32)?.to_vec::<f32>()?;
    let per = w.len() / o.max(1);
    let eps = eps as f32;
    let mut b = vec![0f32; o];
    for c in 0..o {
        let k = g[c] / (rv[c] + eps).sqrt();
        for v in &mut w[c * per..(c + 1) * per] {
            *v *= k;
        }
        b[c] = match &cb {
            Some(cb) => (cb[c] - rm[c]) * k + be[c],
            None => be[c] - rm[c] * k,
        };
    }
    let dev = weight.device();
    let to = |t: Tensor| -> Result<Tensor> { Ok(if dev == Device::Cpu { t } else { t.to_device(dev)? }) };
    Ok((to(Tensor::from_vec(w, weight.shape().clone())?)?, to(Tensor::from_vec(b, [o])?)?))
}
