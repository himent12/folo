//! Muon (Momentum Orthogonalized by Newton–Schulz; Keller Jordan et al., 2024) standalone optimizer.
//!
//! For matrix/convolution parameters of rank `>= 2` (flattened to `[d_0, prod(d_1..)]`), Muon maintains a
//! momentum buffer `m`, forms an optional Nesterov lookahead `u`, orthogonalizes `u` via quintic
//! Newton–Schulz iterations (or an explicit [`MuonOrtho`] schedule from [`crate::optim::ortho`]), applies
//! aspect-ratio scaling, and performs decoupled weight decay:
//!
//! ```text
//! m_t = beta * m_{t-1} + (1 - beta) * g_t
//! u_t = beta * m_t + (1 - beta) * g_t              (when nesterov = true, else m_t)
//! O_t = NewtonSchulz(u_t as [R, C], ns_steps) * aspect_scale(R, C)
//! p_t = p_{t-1} * (1 - lr * weight_decay) - lr * O_t
//! ```
//!
//! Parameters with rank `< 2` (1-D biases, normalization gains, or scalars) are routed cleanly to a
//! configurable auxiliary fallback ([`MuonFallback::AdamW`] by default, or [`MuonFallback::Sgd`]).
//! Parameters that receive no gradient in a step are skipped completely (no decay and no state mutation).

mod checkpoint;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::ortho::{MuonOrtho, NS_COEFFS, NS_EPS, NS_STEPS, orthogonalize_with};
use super::{GradSource, ParamGroup, storage_key, validate_groups};
use crate::{Error, Result};

/// Default Muon learning rate (`0.02`, standard Keller Jordan 2024 recipe).
pub const MUON_DEFAULT_LR: f64 = 0.02;
/// Default Muon momentum `beta` (`0.95`).
pub const MUON_DEFAULT_MOMENTUM: f64 = 0.95;
/// Default Muon decoupled weight decay (`0.01`).
pub const MUON_DEFAULT_WEIGHT_DECAY: f64 = 0.01;
/// Default auxiliary AdamW `beta1` for 1-D / scalar fallback parameters (`0.9`).
pub const MUON_AUX_DEFAULT_BETA1: f64 = 0.9;
/// Default auxiliary AdamW `beta2` for 1-D / scalar fallback parameters (`0.999`).
pub const MUON_AUX_DEFAULT_BETA2: f64 = 0.999;
/// Default auxiliary AdamW `eps` for 1-D / scalar fallback parameters (`1e-8`).
pub const MUON_AUX_DEFAULT_EPS: f64 = 1e-8;

/// Aspect-ratio scaling applied to orthogonalized `[rows, cols]` updates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MuonAspectScale {
    /// Keller Jordan / FOLO geometric aspect scale: `sqrt(max(1, rows / cols))`.
    #[default]
    SqrtRatio,
    /// Moonlight RMS-matched scale: `0.2 * sqrt(max(rows, cols))` (matches AdamW's ~0.2 update RMS).
    MoonlightRms,
    /// Unit scaling (`1.0`).
    None,
}

impl MuonAspectScale {
    /// Compute the scalar multiplier for a parameter reshaped to `[rows, cols]`.
    #[inline]
    pub fn factor(self, rows: usize, cols: usize) -> f64 {
        if rows == 0 || cols == 0 {
            return 1.0;
        }
        match self {
            MuonAspectScale::SqrtRatio => (rows as f64 / cols as f64).max(1.0).sqrt(),
            MuonAspectScale::MoonlightRms => 0.2 * (rows.max(cols) as f64).sqrt(),
            MuonAspectScale::None => 1.0,
        }
    }
}

/// Update rule for non-matrix parameters (`rank < 2`, such as biases and normalization gains).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MuonFallback {
    /// AdamW with bias correction and decoupled weight decay (default).
    #[default]
    AdamW,
    /// Momentum SGD (honoring the group's `nesterov` flag) with decoupled weight decay.
    Sgd,
}

/// Hyper-parameters for constructing a standalone [`Muon`] optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MuonConfig {
    /// Learning rate `>= 0` for Muon updates (default `0.02`).
    pub lr: f64,
    /// Momentum `beta in [0, 1)` for Muon updates (default `0.95`).
    pub momentum: f64,
    /// Use Nesterov momentum lookahead (default `true`).
    pub nesterov: bool,
    /// Decoupled weight decay `>= 0` (default `0.01`).
    pub weight_decay: f64,
    /// Number of quintic Newton–Schulz iterations when `ortho` is `None` (`>= 1`, default `5`).
    pub ns_steps: usize,
    /// Optional preset [`MuonOrtho`] schedule (e.g. `Polar4` / `Polar5`). When `None`, runs `ns_steps`
    /// iterations of classic quintic [`NS_COEFFS`].
    pub ortho: Option<MuonOrtho>,
    /// Aspect-ratio scaling applied to orthogonalized matrix updates (default [`MuonAspectScale::SqrtRatio`]).
    pub aspect_scale: MuonAspectScale,
    /// Auxiliary optimizer rule for 1-D / 0-D fallback parameters (default [`MuonFallback::AdamW`]).
    pub fallback: MuonFallback,
    /// Optional dedicated learning rate for fallback parameters (`None` uses the group's `lr`).
    pub aux_lr: Option<f64>,
    /// Auxiliary AdamW `[beta1, beta2]` for fallback parameters (default `[0.9, 0.999]`).
    pub aux_betas: [f64; 2],
    /// Auxiliary AdamW `eps > 0` for fallback parameters (default `1e-8`).
    pub aux_eps: f64,
    /// Optional dedicated weight decay for fallback parameters (`None` uses the group's `weight_decay`).
    pub aux_weight_decay: Option<f64>,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: MUON_DEFAULT_LR,
            momentum: MUON_DEFAULT_MOMENTUM,
            nesterov: true,
            weight_decay: MUON_DEFAULT_WEIGHT_DECAY,
            ns_steps: NS_STEPS,
            ortho: None,
            aspect_scale: MuonAspectScale::SqrtRatio,
            fallback: MuonFallback::AdamW,
            aux_lr: None,
            aux_betas: [MUON_AUX_DEFAULT_BETA1, MUON_AUX_DEFAULT_BETA2],
            aux_eps: MUON_AUX_DEFAULT_EPS,
            aux_weight_decay: None,
        }
    }
}

impl MuonConfig {
    /// Validate all Muon configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.lr.is_finite() && self.lr >= 0.0 && (self.lr as f32).is_finite()) {
            return Err(Error::Invalid(format!("muon: lr must be finite f32 and >= 0, got {}", self.lr)));
        }
        if !(self.momentum.is_finite() && (0.0..1.0).contains(&self.momentum)) {
            return Err(Error::Invalid(format!(
                "muon: momentum must be finite and in [0, 1), got {}",
                self.momentum
            )));
        }
        if !(self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && (self.weight_decay as f32).is_finite())
        {
            return Err(Error::Invalid(format!(
                "muon: weight_decay must be finite f32 and >= 0, got {}",
                self.weight_decay
            )));
        }
        if self.ortho.is_none() && self.ns_steps == 0 {
            return Err(Error::Invalid("muon: ns_steps must be >= 1 when ortho is None, got 0".into()));
        }
        if let Some(alr) = self.aux_lr
            && !(alr.is_finite() && alr >= 0.0 && (alr as f32).is_finite())
        {
            return Err(Error::Invalid(format!("muon: aux_lr must be finite f32 and >= 0, got {alr}")));
        }
        if !(self.aux_betas[0].is_finite() && (0.0..1.0).contains(&self.aux_betas[0])) {
            return Err(Error::Invalid(format!(
                "muon: aux_betas[0] must be finite and in [0, 1), got {}",
                self.aux_betas[0]
            )));
        }
        if !(self.aux_betas[1].is_finite() && (0.0..1.0).contains(&self.aux_betas[1])) {
            return Err(Error::Invalid(format!(
                "muon: aux_betas[1] must be finite and in [0, 1), got {}",
                self.aux_betas[1]
            )));
        }
        if !(self.aux_eps.is_finite() && self.aux_eps > 0.0 && (self.aux_eps as f32) > 0.0) {
            return Err(Error::Invalid(format!(
                "muon: aux_eps must be finite and > 0 in f32, got {}",
                self.aux_eps
            )));
        }
        if let Some(awd) = self.aux_weight_decay
            && !(awd.is_finite() && awd >= 0.0 && (awd as f32).is_finite())
        {
            return Err(Error::Invalid(format!(
                "muon: aux_weight_decay must be finite f32 and >= 0, got {awd}"
            )));
        }
        Ok(())
    }
}

/// Serializable group contract for [`Muon`] checkpointing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MuonGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate.
    pub lr: f64,
    /// Current scheduled learning rate (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base momentum.
    pub momentum: f64,
    /// Current scheduled momentum (defaults to `momentum` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_momentum: Option<f64>,
    /// Nesterov lookahead flag.
    pub nesterov: bool,
    /// Decoupled weight decay.
    pub weight_decay: f64,
}

/// Serializable contract for [`Muon`] checkpointing and resume parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MuonContract {
    /// Parameter groups.
    pub groups: Vec<MuonGroupContract>,
    /// Global Muon settings.
    pub ns_steps: usize,
    /// Optional orthogonalization schedule override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ortho: Option<MuonOrtho>,
    /// Aspect-ratio scaling mode.
    pub aspect_scale: MuonAspectScale,
    /// Fallback mode for rank `< 2` parameters.
    pub fallback: MuonFallback,
    /// Optional dedicated fallback learning rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_lr: Option<f64>,
    /// Auxiliary AdamW `[beta1, beta2]`.
    pub aux_betas: [f64; 2],
    /// Auxiliary AdamW `eps`.
    pub aux_eps: f64,
    /// Optional dedicated fallback weight decay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_weight_decay: Option<f64>,
    /// Completed optimizer steps.
    pub step: u64,
    /// Per-parameter active step counts (flattened in canonical group/param order).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_steps: Vec<u64>,
}

enum ParamState {
    Matrix { momentum_buf: Tensor, param_step: u64 },
    AuxAdamW { exp_avg: Tensor, exp_avg_sq: Tensor, param_step: u64 },
    AuxSgd { momentum_buf: Tensor, param_step: u64 },
}

impl ParamState {
    fn param_step(&self) -> u64 {
        match self {
            ParamState::Matrix { param_step, .. }
            | ParamState::AuxAdamW { param_step, .. }
            | ParamState::AuxSgd { param_step, .. } => *param_step,
        }
    }

    fn set_param_step(&mut self, s: u64) {
        match self {
            ParamState::Matrix { param_step, .. }
            | ParamState::AuxAdamW { param_step, .. }
            | ParamState::AuxSgd { param_step, .. } => *param_step = s,
        }
    }
}

struct StagedGroup {
    base_lr: f64,
    lr: f64,
    base_momentum: f64,
    momentum: f64,
    nesterov: bool,
    weight_decay: f64,
}

struct PendingResume {
    step: u64,
    cfg: MuonConfig,
    group_names: Vec<String>,
    param_names: Vec<Vec<String>>,
    groups: Vec<StagedGroup>,
    param_steps: Vec<Vec<u64>>,
}

/// Standalone Muon (Momentum Orthogonalized by Newton–Schulz) optimizer with aspect-ratio scaling,
/// decoupled weight decay, and AdamW/SGD fallback for rank `< 2` parameters.
pub struct Muon {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    cfg: MuonConfig,
    state_keys: HashSet<usize>,
    step: u64,
    pending_resume: Option<PendingResume>,
}

/// Run `ns_steps` iterations of classic quintic Newton–Schulz orthogonalization `(a, b, c) = NS_COEFFS`
/// on an `F32` rank-2+ tensor reshaped to `[d_0, prod(d_1..)]` (without aspect scaling).
pub fn newton_schulz_quintic(u: &Tensor, ns_steps: usize) -> Result<Tensor> {
    if ns_steps == 0 {
        return Err(Error::Invalid("newton_schulz_quintic: ns_steps must be >= 1, got 0".into()));
    }
    if u.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("newton_schulz_quintic: expected f32, got {}", u.dtype())));
    }
    let dims = u.dims();
    if dims.len() < 2 {
        return Err(Error::Invalid(format!("newton_schulz_quintic: expected rank >= 2, got {dims:?}")));
    }
    if u.numel() == 0 {
        return Ok(Tensor::zeros_with(u.shape(), DType::F32, u.device())?);
    }
    let _g = no_grad();
    let rows = dims[0];
    let cols = u.numel() / rows;
    let m = u.reshape([rows, cols])?;
    let transposed = rows > cols;
    let mut x = if transposed { m.transpose(0, 1)?.contiguous()? } else { m.contiguous()? };
    // Overflow-safe Frobenius normalization so large finite inputs (e.g. 2e20) do not square to +inf.
    const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
    const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
    let max_abs = x.abs()?.max([0, 1], true)?;
    let is_big = max_abs.gt_scalar(BIG_THRESH)?;
    let one = max_abs.ones_like()?;
    let scale = is_big.where_cond(&one.mul_scalar(DOWN_SCALE)?, &one)?;
    let xs = x.mul(&scale)?;
    let eps_s = scale.mul_scalar(NS_EPS)?;
    let norm = xs.square()?.sum([0, 1], true)?.sqrt()?.add(&eps_s)?;
    x = xs.div(&norm)?;
    let (a, b, c) = NS_COEFFS;
    for _ in 0..ns_steps {
        let xt = x.transpose(0, 1)?.contiguous()?;
        let am = x.matmul(&xt)?;
        let aa = am.matmul(&am)?;
        let bm = am.mul_scalar(b)?.add(&aa.mul_scalar(c)?)?;
        x = x.mul_scalar(a)?.add(&bm.matmul(&x)?)?;
    }
    let out = if transposed { x.transpose(0, 1)?.contiguous()? } else { x };
    Ok(out.reshape(u.shape().clone())?)
}

impl Muon {
    /// Create a [`Muon`] optimizer over `groups` with explicit [`MuonConfig`] settings (each group uses its
    /// own `lr`, `momentum`, `nesterov`, and `weight_decay`).
    pub fn new(groups: Vec<ParamGroup>, cfg: MuonConfig) -> Result<Self> {
        cfg.validate()?;
        validate_groups(&groups)?;
        for g in &groups {
            validate_muon_group(g)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "muon: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = init_state(&groups, cfg.fallback)?;
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 2).sum());
        Ok(Self { groups, state, cfg, state_keys, step: 0, pending_resume: None })
    }

    /// Convenience constructor for a single `"params"` group configured by [`MuonConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: MuonConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.momentum)
            .nesterov(cfg.nesterov)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(true);
        Self::new(vec![group], cfg)
    }

    /// Current global Muon configuration.
    #[inline]
    pub fn config(&self) -> MuonConfig {
        self.cfg
    }

    /// Parameter groups.
    #[inline]
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable parameter groups (e.g., for [`crate::LrSchedule::apply_to_groups`]).
    #[inline]
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Completed [`Muon::step`] calls.
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count (incremented only when that parameter receives a gradient).
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(ParamState::param_step)
    }

    /// Whether parameter `(group, param)` uses orthogonalized Muon updates (`rank >= 2`).
    pub fn is_matrix_param(&self, group: usize, param: usize) -> Option<bool> {
        self.state.get(group)?.get(param).map(|s| matches!(s, ParamState::Matrix { .. }))
    }

    /// Perform one Muon optimization step over every parameter with a gradient in `grads`.
    ///
    /// Validates all groups, hyperparameters, parameter/state alignments, and incoming gradients before
    /// mutating any parameter or state tensor.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.cfg.validate()?;
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "muon: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        let mut param_keys = HashSet::with_capacity(self.groups.iter().map(|g| g.params.len()).sum());
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_muon_group(grp)?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "muon: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "muon: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "muon: parameter `{name}` has an overlapping/broadcast layout"
                    )));
                }
                if !param_keys.insert(storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "muon: parameter `{name}` aliases another parameter"
                    )));
                }
                check_state_alignment(name, p, ps, &mut self.state_keys)?;
                if let Some(g) = grads.grad(p) {
                    if g.dtype() != DType::F32 {
                        return Err(Error::Invalid(format!(
                            "muon: gradient of `{name}` is {}, expected f32",
                            g.dtype()
                        )));
                    }
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "muon: gradient of `{name}` has shape {:?}, expected {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "muon: gradient of `{name}` is on {:?}, parameter on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    check_finite_f32(g, name)?;
                }
            }
        }
        for grp in &self.groups {
            for (name, p) in &grp.params {
                if self.state_keys.contains(&storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "muon: parameter `{name}` aliases an optimizer state buffer"
                    )));
                }
            }
        }

        let _ng = no_grad();
        // Snapshot any gradient that shares storage with a parameter or state buffer before writing.
        let mut active_grads: Vec<Vec<Option<Tensor>>> = Vec::with_capacity(self.groups.len());
        for grp in &self.groups {
            let mut gg = Vec::with_capacity(grp.params.len());
            for (_, p) in &grp.params {
                let snapped = match grads.grad(p) {
                    Some(g) => {
                        let gk = storage_key(g);
                        if param_keys.contains(&gk) || self.state_keys.contains(&gk) {
                            Some(g.mul_scalar(1.0)?)
                        } else {
                            Some(g.clone())
                        }
                    }
                    None => None,
                };
                gg.push(snapped);
            }
            active_grads.push(gg);
        }

        let cfg = self.cfg;
        for ((grp, st), g_list) in self.groups.iter().zip(&mut self.state).zip(active_grads) {
            let lr = grp.lr;
            let beta = grp.momentum;
            let wd = grp.weight_decay;
            let nesterov = grp.nesterov;

            for (((_, p), ps), g_opt) in grp.params.iter().zip(st.iter_mut()).zip(g_list) {
                let Some(g) = g_opt else { continue };
                if p.numel() == 0 {
                    ps.set_param_step(ps.param_step().saturating_add(1));
                    continue;
                }

                match ps {
                    ParamState::Matrix { momentum_buf, param_step } => {
                        let m_new =
                            momentum_buf.mul_scalar(beta as f32)?.add(&g.mul_scalar((1.0 - beta) as f32)?)?;
                        let u = if nesterov {
                            m_new.mul_scalar(beta as f32)?.add(&g.mul_scalar((1.0 - beta) as f32)?)?
                        } else {
                            m_new.clone()
                        };

                        let rows = p.dims()[0];
                        let cols = p.numel() / rows;
                        let (ortho_u, builtin_scale) = if let Some(sched) = cfg.ortho {
                            // `orthogonalize_with` multiplies by `sqrt(max(1, rows / cols))`; fold the undo
                            // factor into `step_scale` to avoid an extra full-tensor pass.
                            let raw = orthogonalize_with(&u.reshape([rows, cols])?, sched)?
                                .reshape(p.shape().clone())?;
                            (raw, (rows as f64 / cols as f64).max(1.0).sqrt())
                        } else {
                            (newton_schulz_quintic(&u, cfg.ns_steps)?, 1.0)
                        };

                        let scale = cfg.aspect_scale.factor(rows, cols);
                        let step_scale = (lr * scale / builtin_scale) as f32;

                        let decayed =
                            if wd != 0.0 { p.mul_scalar((1.0 - lr * wd) as f32)? } else { p.clone() };
                        let p_new = decayed.sub(&ortho_u.mul_scalar(step_scale)?)?;
                        momentum_buf.copy_from(&m_new)?;
                        p.copy_from(&p_new)?;
                        *param_step = param_step.saturating_add(1);
                    }
                    ParamState::AuxAdamW { exp_avg, exp_avg_sq, param_step } => {
                        let t = param_step.saturating_add(1);
                        let alr = cfg.aux_lr.unwrap_or(lr);
                        let awd = cfg.aux_weight_decay.unwrap_or(wd);
                        let b1 = cfg.aux_betas[0];
                        let b2 = cfg.aux_betas[1];
                        let eps = cfg.aux_eps as f32;

                        let m_new = exp_avg.mul_scalar(b1 as f32)?.add(&g.mul_scalar((1.0 - b1) as f32)?)?;
                        let g_scaled = g.mul_scalar((1.0 - b2).sqrt() as f32)?;
                        let v_raw = exp_avg_sq.mul_scalar(b2 as f32)?.add(&g_scaled.square()?)?;
                        let max_t = v_raw.ones_like()?.mul_scalar(f32::MAX)?;
                        let v_new = v_raw.lt_scalar(f32::MAX)?.where_cond(&v_raw, &max_t)?;

                        let bc1 = 1.0 - b1.powi(t.min(i32::MAX as u64) as i32);
                        let bc2 = 1.0 - b2.powi(t.min(i32::MAX as u64) as i32);
                        let step_size = (alr / bc1) as f32;
                        let denom = v_new.sqrt()?.div_scalar(bc2.sqrt() as f32)?.add_scalar(eps)?;
                        let decayed =
                            if awd != 0.0 { p.mul_scalar((1.0 - alr * awd) as f32)? } else { p.clone() };
                        let p_new = decayed.sub(&m_new.div(&denom)?.mul_scalar(step_size)?)?;
                        exp_avg.copy_from(&m_new)?;
                        exp_avg_sq.copy_from(&v_new)?;
                        p.copy_from(&p_new)?;
                        *param_step = t;
                    }
                    ParamState::AuxSgd { momentum_buf, param_step } => {
                        // Matches FOLO `Sgd` undampened momentum equations (`buf = beta * buf + g`,
                        // Nesterov `u = g + beta * buf`, else `u = buf`) with decoupled weight decay.
                        let alr = cfg.aux_lr.unwrap_or(lr);
                        let awd = cfg.aux_weight_decay.unwrap_or(wd);
                        let m_new = momentum_buf.mul_scalar(beta as f32)?.add(&g)?;
                        let u =
                            if nesterov { g.add(&m_new.mul_scalar(beta as f32)?)? } else { m_new.clone() };
                        let decayed =
                            if awd != 0.0 { p.mul_scalar((1.0 - alr * awd) as f32)? } else { p.clone() };
                        let p_new = decayed.sub(&u.mul_scalar(alr as f32)?)?;
                        momentum_buf.copy_from(&m_new)?;
                        p.copy_from(&p_new)?;
                        *param_step = param_step.saturating_add(1);
                    }
                }
            }
        }

        self.step = self.step.saturating_add(1);
        Ok(())
    }

    /// Move all parameters and state tensors to `device` in place.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        for g in &self.groups {
            for (_, p) in &g.params {
                p.to_device_(device)?;
            }
        }
        for (_, t) in self.state_tensors()? {
            t.to_device_(device)?;
        }
        Ok(())
    }
}

fn validate_muon_scalars(name: &str, lr: f64, momentum: f64, weight_decay: f64) -> Result<()> {
    let ok = lr.is_finite()
        && lr >= 0.0
        && (lr as f32).is_finite()
        && momentum.is_finite()
        && (0.0..1.0).contains(&momentum)
        && weight_decay.is_finite()
        && weight_decay >= 0.0
        && (weight_decay as f32).is_finite();
    if !ok {
        return Err(Error::Invalid(format!(
            "muon group `{name}`: invalid lr={lr}, momentum={momentum}, wd={weight_decay}"
        )));
    }
    Ok(())
}

fn validate_muon_group(g: &ParamGroup) -> Result<()> {
    validate_muon_scalars(&g.name, g.lr, g.momentum, g.weight_decay)
}

fn check_finite_f32(t: &Tensor, name: &str) -> Result<()> {
    if t.numel() == 0 {
        return Ok(());
    }
    let all_finite = if t.device().is_cpu() {
        t.with_f32_slice(|s| s.iter().all(|v| v.is_finite()))?
    } else {
        t.to_vec::<f32>()?.iter().all(|v| v.is_finite())
    };
    if !all_finite {
        return Err(Error::Invalid(format!("muon: gradient of `{name}` contains non-finite values")));
    }
    Ok(())
}

fn init_state(groups: &[ParamGroup], fallback: MuonFallback) -> Result<Vec<Vec<ParamState>>> {
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let mut gs = Vec::with_capacity(g.params.len());
        for (_, p) in &g.params {
            let dev = p.device();
            let ps = if p.rank() >= 2 {
                ParamState::Matrix {
                    momentum_buf: Tensor::zeros_with(p.shape(), DType::F32, dev)?,
                    param_step: 0,
                }
            } else {
                match fallback {
                    MuonFallback::AdamW => ParamState::AuxAdamW {
                        exp_avg: Tensor::zeros_with(p.shape(), DType::F32, dev)?,
                        exp_avg_sq: Tensor::zeros_with(p.shape(), DType::F32, dev)?,
                        param_step: 0,
                    },
                    MuonFallback::Sgd => ParamState::AuxSgd {
                        momentum_buf: Tensor::zeros_with(p.shape(), DType::F32, dev)?,
                        param_step: 0,
                    },
                }
            };
            gs.push(ps);
        }
        out.push(gs);
    }
    Ok(out)
}

fn check_state_alignment(name: &str, p: &Tensor, ps: &ParamState, seen: &mut HashSet<usize>) -> Result<()> {
    let check = |label: &str, buf: &Tensor, seen: &mut HashSet<usize>| -> Result<()> {
        if buf.shape() != p.shape() || buf.dtype() != DType::F32 || buf.device() != p.device() {
            return Err(Error::Invalid(format!("muon: state `{name}.{label}` is misaligned with parameter")));
        }
        if !seen.insert(storage_key(buf)) {
            return Err(Error::Invalid(format!("muon: state `{name}.{label}` aliases another buffer")));
        }
        Ok(())
    };
    match ps {
        ParamState::Matrix { momentum_buf, .. } | ParamState::AuxSgd { momentum_buf, .. } => {
            check("momentum_buffer", momentum_buf, seen)
        }
        ParamState::AuxAdamW { exp_avg, exp_avg_sq, .. } => {
            check("exp_avg", exp_avg, seen)?;
            check("exp_avg_sq", exp_avg_sq, seen)
        }
    }
}
