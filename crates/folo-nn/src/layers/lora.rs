//! Low-Rank Adaptation (LoRA; Hu et al., ICLR 2022 / arXiv:2106.09685) adapters for [`Linear`]
//! and [`Conv2d`] layers.
//!
//! Wraps a frozen pretrained base layer (`base.weight` / `base.bias` exposed as non-learnable
//! [`TensorKind::Buffer`] tensors with `requires_grad = false`) alongside a trainable low-rank
//! residual branch (`lora_a`, `lora_b` exposed as [`TensorKind::Parameter`] tensors):
//!
//! - **`LoraLinear`**: `y = base(x) + (alpha / rank) * (dropout(x) @ lora_a^T @ lora_b^T)`
//!   with `lora_a: [rank, in_features]` (Kaiming uniform `a = sqrt(5)`) and `lora_b: [out_features, rank]`
//!   (initialized to zero so initial `delta_W = 0`).
//! - **`LoraConv2d`** (`groups == 1`): `y = base(x) + (alpha / rank) * conv1x1(conv_kxk(dropout(x), lora_a), lora_b)`
//!   with `lora_a: [rank, in_channels, k, k]` (Kaiming uniform `a = sqrt(5)`) and
//!   `lora_b: [out_channels, rank, 1, 1]` (initialized to zero).
//!
//! Both adapters support lossless in-place weight folding via [`merge_weights`](LoraLinear::merge_weights)
//! and [`unmerge_weights`](LoraLinear::unmerge_weights) (exact mathematical equivalence in eval mode)
//! and full `.folo` state-dict round-tripping through [`Module::named_tensors`].

use std::sync::Mutex;

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, DType, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::conv::Conv2d;
use super::simple::Linear;
use crate::init::{self, FanMode};
use crate::module::{Forward, Mode, Module, TensorKind};
use crate::{Error, Result};

/// Configuration for [`LoraLinear`] and [`LoraConv2d`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoraConfig {
    /// Low-rank bottleneck dimension `r >= 1` (default `4`).
    pub rank: usize,
    /// LoRA scaling numerator `alpha >= 0` (`scaling = alpha / rank`, default `8.0`).
    pub alpha: f32,
    /// Optional inverted dropout probability `p in [0, 1)` applied to the LoRA branch input during
    /// training (`0.0` disables dropout).
    pub dropout: f32,
    /// Deterministic PRNG seed for LoRA dropout masks (default `0`).
    pub seed: u64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self { rank: 4, alpha: 8.0, dropout: 0.0, seed: 0 }
    }
}

impl LoraConfig {
    /// Validate `rank >= 1`, finite `alpha >= 0`, and finite `dropout in [0, 1)`.
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(Error::Invalid("lora: rank must be >= 1, got 0".into()));
        }
        if !(self.alpha.is_finite() && self.alpha >= 0.0) {
            return Err(Error::Invalid(format!("lora: alpha must be finite and >= 0, got {}", self.alpha)));
        }
        if !(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout)) {
            return Err(Error::Invalid(format!(
                "lora: dropout must be finite and in [0, 1), got {}",
                self.dropout
            )));
        }
        Ok(())
    }

    /// Effective scaling factor `alpha / rank`.
    #[inline]
    pub fn scaling(&self) -> f32 {
        self.alpha / (self.rank as f32)
    }
}

/// Apply seeded inverted elementwise dropout to `x` during training when `p > 0`.
fn apply_lora_dropout(x: &Tensor, p: f32, training: bool, rng: &Mutex<Rng>) -> Result<Tensor> {
    if !training || p <= 0.0 {
        return Ok(x.clone());
    }
    let keep = 1.0 - p;
    let scale = 1.0 / keep;
    let n = x.numel();
    let mut mask_data = vec![0.0f32; n];
    {
        let mut guard = rng.lock().unwrap_or_else(|e| e.into_inner());
        for m in &mut mask_data {
            if guard.next_f32() >= p {
                *m = scale;
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, x.shape())?.to_device(x.device())?;
    let mask = if x.dtype() != DType::F32 { mask.to_dtype(x.dtype())? } else { mask };
    Ok(x.mul(&mask)?)
}

/// Low-Rank Adaptation wrapper around a frozen [`Linear`] layer.
///
/// Tensor names (via [`Module::visit_tensors`]):
/// - `base.weight` (`Buffer`, frozen)
/// - `base.bias` (`Buffer`, frozen, when present)
/// - `lora_a` (`Parameter`, trainable `[rank, in_features]`)
/// - `lora_b` (`Parameter`, trainable `[out_features, rank]`)
pub struct LoraLinear {
    base: Linear,
    /// Down-projection matrix `[rank, in_features]` (Kaiming uniform init).
    pub lora_a: Tensor,
    /// Up-projection matrix `[out_features, rank]` (zero init).
    pub lora_b: Tensor,
    config: LoraConfig,
    merged: bool,
    mode: Mode,
    rng: Mutex<Rng>,
}

impl LoraLinear {
    /// Wrap an existing [`Linear`] layer with freshly initialized LoRA adapters, freezing `base`
    /// (`requires_grad = false`).
    pub fn new(base: Linear, config: LoraConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let (out_f, in_f) = base.weight.dims2().map_err(|_| {
            Error::Invalid(format!("LoraLinear: base weight must be 2-D, got {:?}", base.weight.dims()))
        })?;
        if out_f == 0 || in_f == 0 {
            return Err(Error::Invalid(format!(
                "LoraLinear: base weight dims must be >= 1, got [{out_f}, {in_f}]"
            )));
        }
        let dev = base.weight.device();
        let frozen_base = Linear::from_tensors(base.weight.detach(), base.bias.as_ref().map(Tensor::detach))?;
        let lora_a = init::kaiming_uniform([config.rank, in_f], 5.0f64.sqrt(), FanMode::FanIn, rng)?
            .to_device(dev)?
            .requires_grad_();
        let lora_b = Tensor::zeros_with([out_f, config.rank], DType::F32, dev)?.requires_grad_();
        Ok(Self {
            base: frozen_base,
            lora_a,
            lora_b,
            config,
            merged: false,
            mode: Mode::new(),
            rng: Mutex::new(Rng::seed(config.seed)),
        })
    }

    /// Reference to the wrapped frozen [`Linear`] base layer.
    pub fn base(&self) -> &Linear {
        &self.base
    }

    /// The LoRA configuration.
    pub fn config(&self) -> LoraConfig {
        self.config
    }

    /// Effective scaling factor `alpha / rank`.
    pub fn scaling(&self) -> f32 {
        self.config.scaling()
    }

    /// Whether the adapter delta `(alpha / rank) * B @ A` is currently folded into `base.weight`.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Trainable adapter parameters `[lora_a, lora_b]`.
    pub fn adapter_parameters(&self) -> Vec<Tensor> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Named trainable adapter parameters `[("lora_a", lora_a), ("lora_b", lora_b)]`.
    pub fn named_adapter_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("lora_a".into(), self.lora_a.clone()), ("lora_b".into(), self.lora_b.clone())]
    }

    /// Reset the internal dropout RNG stream to `seed`.
    pub fn reseed_dropout(&self, seed: u64) {
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) = Rng::seed(seed);
    }

    /// Compute the scaled low-rank weight delta `(alpha / rank) * (lora_b @ lora_a)` of shape `[out, in]`.
    pub fn delta_weight(&self) -> Result<Tensor> {
        Ok(self.lora_b.matmul(&self.lora_a)?.mul_scalar(self.scaling())?)
    }

    /// Fold `(alpha / rank) * (lora_b @ lora_a)` into `base.weight` in place (idempotent).
    ///
    /// While merged, [`Forward::forward`] executes only `base.forward(x)` with zero adapter overhead.
    pub fn merge_weights(&mut self) -> Result<()> {
        if self.merged {
            return Ok(());
        }
        let _ng = no_grad();
        let delta = self.delta_weight()?;
        let delta = if delta.dtype() != self.base.weight.dtype() {
            delta.to_dtype(self.base.weight.dtype())?
        } else {
            delta
        };
        let merged = self.base.weight.add(&delta)?;
        self.base.weight.copy_from(&merged)?;
        self.merged = true;
        Ok(())
    }

    /// Subtract `(alpha / rank) * (lora_b @ lora_a)` from `base.weight` to restore the unmerged
    /// base weights (idempotent).
    pub fn unmerge_weights(&mut self) -> Result<()> {
        if !self.merged {
            return Ok(());
        }
        let _ng = no_grad();
        let delta = self.delta_weight()?;
        let delta = if delta.dtype() != self.base.weight.dtype() {
            delta.to_dtype(self.base.weight.dtype())?
        } else {
            delta
        };
        let unmerged = self.base.weight.sub(&delta)?;
        self.base.weight.copy_from(&unmerged)?;
        self.merged = false;
        Ok(())
    }
}

impl Module for LoraLinear {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("base.weight", &self.base.weight, TensorKind::Buffer);
        if let Some(b) = &self.base.bias {
            f("base.bias", b, TensorKind::Buffer);
        }
        f("lora_a", &self.lora_a, TensorKind::Parameter);
        f("lora_b", &self.lora_b, TensorKind::Parameter);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl Forward for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        if self.merged {
            return Ok(base_out);
        }
        let (o, i) = self.base.weight.dims2()?;
        let x_drop = apply_lora_dropout(x, self.config.dropout, self.mode.is_training(), &self.rng)?;
        let rows = x_drop.numel() / i.max(1);
        let h = x_drop.reshape([rows, i])?.matmul(&self.lora_a.t()?)?;
        let delta = h.matmul(&self.lora_b.t()?)?.mul_scalar(self.scaling())?;
        let mut out_shape = x.dims().to_vec();
        *out_shape.last_mut().expect("rank >= 1") = o;
        let delta = delta.reshape(out_shape)?;
        let delta = if delta.dtype() != base_out.dtype() { delta.to_dtype(base_out.dtype())? } else { delta };
        Ok(base_out.add(&delta)?)
    }
}

/// Low-Rank Adaptation wrapper around a frozen standard (`groups == 1`) [`Conv2d`] layer.
///
/// Tensor names (via [`Module::visit_tensors`]):
/// - `base.weight` (`Buffer`, frozen)
/// - `base.bias` (`Buffer`, frozen, when present)
/// - `lora_a` (`Parameter`, trainable `[rank, in_channels, k, k]`)
/// - `lora_b` (`Parameter`, trainable `[out_channels, rank, 1, 1]`)
pub struct LoraConv2d {
    base: Conv2d,
    /// Spatial down-projection conv kernel `[rank, in_channels, k, k]` (Kaiming uniform init).
    pub lora_a: Tensor,
    /// Pointwise up-projection conv kernel `[out_channels, rank, 1, 1]` (zero init).
    pub lora_b: Tensor,
    config: LoraConfig,
    merged: bool,
    mode: Mode,
    rng: Mutex<Rng>,
}

impl LoraConv2d {
    /// Wrap an existing standard (`groups == 1`) [`Conv2d`] layer with freshly initialized LoRA
    /// adapters, freezing `base` (`requires_grad = false`).
    pub fn new(base: Conv2d, config: LoraConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let c = base.config();
        if c.groups != 1 {
            return Err(Error::Invalid(format!(
                "LoraConv2d: only groups == 1 is supported, got groups = {}",
                c.groups
            )));
        }
        let dev = base.weight.device();
        let frozen_base =
            Conv2d::from_tensors(*c, base.weight.detach(), base.bias.as_ref().map(Tensor::detach))?;
        let lora_a = init::kaiming_uniform(
            [config.rank, c.in_channels, c.kernel, c.kernel],
            5.0f64.sqrt(),
            FanMode::FanIn,
            rng,
        )?
        .to_device(dev)?
        .requires_grad_();
        let lora_b =
            Tensor::zeros_with([c.out_channels, config.rank, 1, 1], DType::F32, dev)?.requires_grad_();
        Ok(Self {
            base: frozen_base,
            lora_a,
            lora_b,
            config,
            merged: false,
            mode: Mode::new(),
            rng: Mutex::new(Rng::seed(config.seed)),
        })
    }

    /// Reference to the wrapped frozen [`Conv2d`] base layer.
    pub fn base(&self) -> &Conv2d {
        &self.base
    }

    /// The LoRA configuration.
    pub fn config(&self) -> LoraConfig {
        self.config
    }

    /// Effective scaling factor `alpha / rank`.
    pub fn scaling(&self) -> f32 {
        self.config.scaling()
    }

    /// Whether the adapter delta is currently folded into `base.weight`.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Trainable adapter parameters `[lora_a, lora_b]`.
    pub fn adapter_parameters(&self) -> Vec<Tensor> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Named trainable adapter parameters `[("lora_a", lora_a), ("lora_b", lora_b)]`.
    pub fn named_adapter_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("lora_a".into(), self.lora_a.clone()), ("lora_b".into(), self.lora_b.clone())]
    }

    /// Reset the internal dropout RNG stream to `seed`.
    pub fn reseed_dropout(&self, seed: u64) {
        *self.rng.lock().unwrap_or_else(|e| e.into_inner()) = Rng::seed(seed);
    }

    /// Compute the exact merged weight delta `[out_channels, in_channels, k, k]`:
    /// `(alpha / rank) * reshape(lora_b, [out, r]) @ reshape(lora_a, [r, in * k * k])`.
    pub fn delta_weight(&self) -> Result<Tensor> {
        let c = self.base.config();
        let r = self.config.rank;
        let b_mat = self.lora_b.reshape([c.out_channels, r])?;
        let a_mat = self.lora_a.reshape([r, c.in_channels * c.kernel * c.kernel])?;
        let delta = b_mat.matmul(&a_mat)?.mul_scalar(self.scaling())?;
        Ok(delta.reshape(c.weight_shape())?)
    }

    /// Fold the LoRA weight delta into `base.weight` in place (idempotent).
    pub fn merge_weights(&mut self) -> Result<()> {
        if self.merged {
            return Ok(());
        }
        let _ng = no_grad();
        let delta = self.delta_weight()?;
        let delta = if delta.dtype() != self.base.weight.dtype() {
            delta.to_dtype(self.base.weight.dtype())?
        } else {
            delta
        };
        let merged = self.base.weight.add(&delta)?;
        self.base.weight.copy_from(&merged)?;
        self.merged = true;
        Ok(())
    }

    /// Subtract the LoRA weight delta from `base.weight` to restore unmerged base weights (idempotent).
    pub fn unmerge_weights(&mut self) -> Result<()> {
        if !self.merged {
            return Ok(());
        }
        let _ng = no_grad();
        let delta = self.delta_weight()?;
        let delta = if delta.dtype() != self.base.weight.dtype() {
            delta.to_dtype(self.base.weight.dtype())?
        } else {
            delta
        };
        let unmerged = self.base.weight.sub(&delta)?;
        self.base.weight.copy_from(&unmerged)?;
        self.merged = false;
        Ok(())
    }
}

impl Module for LoraConv2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("base.weight", &self.base.weight, TensorKind::Buffer);
        if let Some(b) = &self.base.bias {
            f("base.bias", b, TensorKind::Buffer);
        }
        f("lora_a", &self.lora_a, TensorKind::Parameter);
        f("lora_b", &self.lora_b, TensorKind::Parameter);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl Forward for LoraConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        if self.merged {
            return Ok(base_out);
        }
        let c = self.base.config();
        let x_drop = apply_lora_dropout(x, self.config.dropout, self.mode.is_training(), &self.rng)?;
        let h = x_drop.conv2d_with(&self.lora_a, None, &c.params())?;
        let p1 = Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 };
        let delta = h.conv2d_with(&self.lora_b, None, &p1)?.mul_scalar(self.scaling())?;
        let delta = if delta.dtype() != base_out.dtype() { delta.to_dtype(base_out.dtype())? } else { delta };
        Ok(base_out.add(&delta)?)
    }
}
