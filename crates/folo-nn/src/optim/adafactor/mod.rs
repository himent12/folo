//! Adafactor memory-efficient adaptive optimizer (Shazeer & Stern, ICML 2018 / arXiv:1804.04235).
//!
//! For parameters of rank `>= 2` (logically viewed as `[R, C]` where `R = dims[0]` and
//! `C = prod(dims[1..])`), Adafactor replaces the full `R × C` second-moment matrix with row and column
//! running averages `v_r: [R]` and `v_c: [C]` (`O(R + C)` memory):
//!
//! ```text
//! beta2_t   = 1 - t^decay_rate                         (t >= 1, decay_rate < 0, default -0.8)
//! v_r[r]    = beta2_t * v_r[r] + (1 - beta2_t) * mean_c(g[r, c]^2 + eps1)
//! v_c[c]    = beta2_t * v_c[c] + (1 - beta2_t) * mean_r(g[r, c]^2 + eps1)
//! v_hat[r,c]= (v_r[r] / mean(v_r)) * v_c[c]
//! u         = g / sqrt(v_hat)
//! u_clip    = u / max(1, RMS(u) / clip_threshold)      (RMS update clipping, default clip_threshold = 1.0)
//! ```
//!
//! Rank `< 2` parameters (1-D biases, normalization gains, or scalars) store a full unfactored second moment
//! `v`. Supports optional first-moment EMA (`beta1`, applied to the `alpha_t`-scaled update `alpha_t * u_clip`
//! matching HuggingFace Transformers / Shazeer & Stern §7), relative step-size schedules (`relative_step` /
//! `warmup_init`), parameter-RMS scaling (`scale_parameter`), and decoupled weight decay. Parameters without
//! a gradient in a step are skipped completely (no decay and no state update). Note: the `f64` update kernel
//! currently stages GPU tensors on the host per step for numerical accuracy.

mod checkpoint;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{GradSource, ParamGroup, storage_key, validate_groups};
use crate::{Error, Result};

/// Default Adafactor explicit learning rate when `relative_step = false` (`1e-3`).
pub const ADAFACTOR_DEFAULT_LR: f64 = 1e-3;
/// Default Adafactor `(eps1, eps2)` regularization floors (`[1e-30, 1e-3]`, Shazeer & Stern 2018).
pub const ADAFACTOR_DEFAULT_EPS: [f64; 2] = [1e-30, 1e-3];
/// Default Adafactor RMS update clipping threshold `d` (`1.0`).
pub const ADAFACTOR_DEFAULT_CLIP_THRESHOLD: f64 = 1.0;
/// Default Adafactor second-moment decay exponent `c` (`-0.8`, giving `beta2_t = 1 - t^-0.8`).
pub const ADAFACTOR_DEFAULT_DECAY_RATE: f64 = -0.8;

/// Hyper-parameters for constructing an [`Adafactor`] optimizer (Shazeer & Stern, ICML 2018).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdafactorConfig {
    /// Explicit learning rate `>= 0` when `relative_step = false` (default `Some(1e-3)`). Must be `None`
    /// when `relative_step = true`.
    pub lr: Option<f64>,
    /// Regularization floors `[eps1, eps2]`: `eps1 > 0` added to `g²`, `eps2 > 0` floor on `RMS(param)`
    /// (paper default `[1e-30, 1e-3]`).
    pub eps: [f64; 2],
    /// Maximum allowed RMS of the normalized update `u` (`> 0`, paper default `1.0`).
    pub clip_threshold: f64,
    /// Negative decay exponent `c < 0` in `beta2_t = 1 - t^c` (paper default `-0.8`).
    pub decay_rate: f64,
    /// Optional first-moment EMA coefficient `beta1 in [0, 1)` (`None` = memory-free Adafactor default).
    pub beta1: Option<f64>,
    /// Decoupled weight decay `>= 0` (default `0.0`).
    pub weight_decay: f64,
    /// Multiply step size by `max(eps2, RMS(param))` (paper default `true`).
    pub scale_parameter: bool,
    /// Compute step size automatically via `rho_t = min(1e-2, 1 / sqrt(t))` (or linear warmup when
    /// `warmup_init = true`; default `false`).
    pub relative_step: bool,
    /// Linearly warm up the relative step size (`min(1e-6 * t, 1 / sqrt(t))`); requires `relative_step = true`.
    pub warmup_init: bool,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: Some(ADAFACTOR_DEFAULT_LR),
            eps: ADAFACTOR_DEFAULT_EPS,
            clip_threshold: ADAFACTOR_DEFAULT_CLIP_THRESHOLD,
            decay_rate: ADAFACTOR_DEFAULT_DECAY_RATE,
            beta1: None,
            weight_decay: 0.0,
            scale_parameter: true,
            relative_step: false,
            warmup_init: false,
        }
    }
}

impl AdafactorConfig {
    /// Standard Shazeer & Stern (2018) hyperparameter-free relative-step preset (`relative_step = true`,
    /// `scale_parameter = true`, `warmup_init = true`).
    pub fn relative_step() -> Self {
        Self { lr: None, relative_step: true, scale_parameter: true, warmup_init: true, ..Self::default() }
    }

    /// Validate all Adafactor hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if self.relative_step {
            if self.lr.is_some() {
                return Err(Error::Invalid("adafactor: lr must be None when relative_step is true".into()));
            }
        } else {
            let lr = self.lr.ok_or_else(|| {
                Error::Invalid("adafactor: lr must be Some(>= 0) when relative_step is false".into())
            })?;
            if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite()) {
                return Err(Error::Invalid(format!("adafactor: lr must be finite f32 and >= 0, got {lr}")));
            }
        }
        if self.warmup_init && !self.relative_step {
            return Err(Error::Invalid("adafactor: warmup_init requires relative_step = true".into()));
        }
        if !(self.eps[0].is_finite() && self.eps[0] >= 1e-300) {
            return Err(Error::Invalid(format!(
                "adafactor: eps[0] must be finite and >= 1e-300, got {}",
                self.eps[0]
            )));
        }
        if !(self.eps[1].is_finite() && self.eps[1] > 0.0) {
            return Err(Error::Invalid(format!(
                "adafactor: eps[1] must be finite and > 0, got {}",
                self.eps[1]
            )));
        }
        if !(self.clip_threshold.is_finite() && self.clip_threshold > 0.0) {
            return Err(Error::Invalid(format!(
                "adafactor: clip_threshold must be finite and > 0, got {}",
                self.clip_threshold
            )));
        }
        if !(self.decay_rate.is_finite() && self.decay_rate < 0.0) {
            return Err(Error::Invalid(format!(
                "adafactor: decay_rate must be finite and < 0, got {}",
                self.decay_rate
            )));
        }
        if let Some(b1) = self.beta1
            && !(b1.is_finite() && (0.0..1.0).contains(&b1))
        {
            return Err(Error::Invalid(format!("adafactor: beta1 must be finite and in [0, 1), got {b1}")));
        }
        if !(self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && (self.weight_decay as f32).is_finite())
        {
            return Err(Error::Invalid(format!(
                "adafactor: weight_decay must be finite f32 and >= 0, got {}",
                self.weight_decay
            )));
        }
        Ok(())
    }
}

/// Serializable group contract for [`Adafactor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdafactorGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate (used when `relative_step = false`).
    pub lr: f64,
    /// Current scheduled learning rate (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Decoupled weight decay.
    pub weight_decay: f64,
}

/// Serializable contract for [`Adafactor`] checkpointing and resume parity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdafactorContract {
    /// Parameter groups.
    pub groups: Vec<AdafactorGroupContract>,
    /// Adafactor configuration.
    pub config: AdafactorConfig,
    /// Completed optimizer steps.
    pub step: u64,
    /// Per-parameter active step counts (flattened in canonical order).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_steps: Vec<u64>,
}

enum SecondMoment {
    Factored { row: Tensor, col: Tensor },
    Unfactored(Tensor),
}

struct ParamState {
    second: SecondMoment,
    exp_avg: Option<Tensor>,
    param_step: u64,
}

struct StagedGroup {
    base_lr: f64,
    lr: f64,
    weight_decay: f64,
}

struct PendingResume {
    step: u64,
    cfg: AdafactorConfig,
    group_names: Vec<String>,
    param_names: Vec<Vec<String>>,
    groups: Vec<StagedGroup>,
    param_steps: Vec<Vec<u64>>,
}

/// Adafactor memory-efficient optimizer (Shazeer & Stern, ICML 2018).
pub struct Adafactor {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    cfg: AdafactorConfig,
    state_keys: HashSet<usize>,
    step: u64,
    pending_resume: Option<PendingResume>,
}

/// Compute Adafactor's second-moment retention `beta2_t = 1 - t^decay_rate` for active step `t >= 1`.
#[inline]
pub fn adafactor_beta2(step: u64, decay_rate: f64) -> f64 {
    if step <= 1 { 0.0 } else { (1.0 - (step as f64).powf(decay_rate)).clamp(0.0, 1.0 - f64::EPSILON) }
}

/// Compute Adafactor's relative step-size multiplier `rho_t` for active step `t >= 1`.
#[inline]
pub fn adafactor_relative_lr(step: u64, warmup_init: bool) -> f64 {
    let t = step.max(1) as f64;
    let min_step = if warmup_init { 1e-6 * t } else { 1e-2 };
    min_step.min(t.sqrt().recip())
}

impl Adafactor {
    /// Create an [`Adafactor`] optimizer over `groups` with [`AdafactorConfig`] settings.
    pub fn new(groups: Vec<ParamGroup>, cfg: AdafactorConfig) -> Result<Self> {
        cfg.validate()?;
        validate_groups(&groups)?;
        for g in &groups {
            validate_adafactor_group(g, cfg.relative_step)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = init_state(&groups, cfg.beta1.is_some())?;
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 3).sum());
        Ok(Self { groups, state, cfg, state_keys, step: 0, pending_resume: None })
    }

    /// Convenience constructor for a single `"params"` group configured by [`AdafactorConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: AdafactorConfig) -> Result<Self> {
        cfg.validate()?;
        let lr = cfg.lr.unwrap_or(0.0);
        let group =
            ParamGroup::new("params", params).lr(lr).weight_decay(cfg.weight_decay).decoupled_wd(true);
        Self::new(vec![group], cfg)
    }

    /// Current Adafactor configuration.
    #[inline]
    pub fn config(&self) -> AdafactorConfig {
        self.cfg
    }

    /// Parameter groups.
    #[inline]
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable parameter groups (for LR schedules when `relative_step = false`).
    #[inline]
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Completed [`Adafactor::step`] calls.
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count.
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// Factored `(exp_avg_sq_row, exp_avg_sq_col)` buffers for a rank `>= 2` parameter (`None` if unfactored).
    pub fn factored_moments(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        match &self.state.get(group)?.get(param)?.second {
            SecondMoment::Factored { row, col } => Some((row, col)),
            SecondMoment::Unfactored(_) => None,
        }
    }

    /// Unfactored `exp_avg_sq` buffer for a rank `< 2` parameter (`None` if factored).
    pub fn unfactored_moment(&self, group: usize, param: usize) -> Option<&Tensor> {
        match &self.state.get(group)?.get(param)?.second {
            SecondMoment::Unfactored(v) => Some(v),
            SecondMoment::Factored { .. } => None,
        }
    }

    /// Perform one Adafactor optimization step over every parameter with a gradient in `grads`.
    ///
    /// Validates all groups, state buffers, and gradients before mutating any parameter or state tensor.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.cfg.validate()?;
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "adafactor: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        let mut param_keys = HashSet::with_capacity(self.groups.iter().map(|g| g.params.len()).sum());
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_adafactor_group(grp, self.cfg.relative_step)?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "adafactor: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` has an overlapping/broadcast layout"
                    )));
                }
                if !param_keys.insert(storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` aliases another parameter"
                    )));
                }
                check_state_alignment(name, p, ps, &mut self.state_keys)?;
                if let Some(g) = grads.grad(p) {
                    if g.dtype() != DType::F32 {
                        return Err(Error::Invalid(format!(
                            "adafactor: gradient of `{name}` is {}, expected f32",
                            g.dtype()
                        )));
                    }
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "adafactor: gradient of `{name}` has shape {:?}, expected {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "adafactor: gradient of `{name}` is on {:?}, parameter on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                }
            }
        }
        for grp in &self.groups {
            for (name, p) in &grp.params {
                if self.state_keys.contains(&storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` aliases an optimizer state buffer"
                    )));
                }
            }
        }

        let _ng = no_grad();
        let cfg = self.cfg;
        let eps1 = cfg.eps[0];
        let eps2 = cfg.eps[1];
        let clip_thr = cfg.clip_threshold;

        // Compute and validate every staged parameter/state update before mutating any tensor.
        enum StagedSecond {
            Factored(Vec<f32>, Vec<f32>),
            Unfactored(Vec<f32>),
            Empty,
        }
        struct StagedUpdate {
            gi: usize,
            pi: usize,
            new_step: u64,
            second: StagedSecond,
            exp_avg: Option<Vec<f32>>,
            param: Option<Vec<f32>>,
        }

        let mut updates = Vec::new();
        for (gi, (grp, st)) in self.groups.iter().zip(&self.state).enumerate() {
            let group_lr = grp.lr;
            let wd = grp.weight_decay;

            for (pi, ((name, p), ps)) in grp.params.iter().zip(st).enumerate() {
                let Some(g) = grads.grad(p) else { continue };
                let n = p.numel();
                let t = ps.param_step.saturating_add(1);
                if n == 0 {
                    updates.push(StagedUpdate {
                        gi,
                        pi,
                        new_step: t,
                        second: StagedSecond::Empty,
                        exp_avg: None,
                        param: None,
                    });
                    continue;
                }

                let p_vec = p.to_vec::<f32>()?;
                let g_vec = g.to_vec::<f32>()?;
                if g_vec.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Invalid(format!(
                        "adafactor: gradient of `{name}` contains non-finite values"
                    )));
                }
                if p_vec.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Invalid(format!(
                        "adafactor: parameter `{name}` contains non-finite values"
                    )));
                }

                let beta2_t = adafactor_beta2(t, cfg.decay_rate);
                let base_step =
                    if cfg.relative_step { adafactor_relative_lr(t, cfg.warmup_init) } else { group_lr };
                let rms_p = if cfg.scale_parameter { rms_f32_safe(&p_vec).max(eps2) } else { 1.0 };
                let alpha_t = base_step * rms_p;

                let mut u = vec![0.0f64; n];
                let staged_second = match &ps.second {
                    SecondMoment::Factored { row, col } => {
                        let rows = p.dims()[0];
                        let cols = n / rows;
                        let mut vr = row.to_vec::<f32>()?;
                        let mut vc = col.to_vec::<f32>()?;

                        let mut row_means = vec![0.0f64; rows];
                        let mut col_means = vec![0.0f64; cols];
                        for (r, rm) in row_means.iter_mut().enumerate() {
                            let base = r * cols;
                            let mut r_sum = 0.0f64;
                            for (c, cm) in col_means.iter_mut().enumerate() {
                                let gv = g_vec[base + c] as f64;
                                let g2 = gv * gv + eps1;
                                r_sum += g2;
                                *cm += g2;
                            }
                            *rm = r_sum / cols as f64;
                        }
                        let inv_rows = 1.0 / rows as f64;
                        for cm in &mut col_means {
                            *cm *= inv_rows;
                        }

                        let mut vr_f64 = vec![0.0f64; rows];
                        let mut vc_f64 = vec![0.0f64; cols];
                        let one_minus_b2 = 1.0 - beta2_t;
                        let mut vr_sum = 0.0f64;
                        for r in 0..rows {
                            let v = beta2_t * (vr[r] as f64) + one_minus_b2 * row_means[r];
                            vr_f64[r] = v;
                            vr[r] = clamp_f32_finite(v);
                            vr_sum += v;
                        }
                        let vr_mean = vr_sum * inv_rows;
                        for c in 0..cols {
                            let v = beta2_t * (vc[c] as f64) + one_minus_b2 * col_means[c];
                            vc_f64[c] = v;
                            vc[c] = clamp_f32_finite(v);
                        }

                        let row_scale: Vec<f64> =
                            vr_f64
                                .iter()
                                .map(|&v| {
                                    if v > 0.0 && vr_mean.is_finite() {
                                        (vr_mean / v).sqrt()
                                    } else {
                                        f64::NAN
                                    }
                                })
                                .collect();
                        let col_scale: Vec<f64> = vc_f64
                            .iter()
                            .map(|&v| if v > 0.0 { v.sqrt().recip() } else { f64::NAN })
                            .collect();

                        for (r, &rs) in row_scale.iter().enumerate() {
                            let base = r * cols;
                            for (c, &cs) in col_scale.iter().enumerate() {
                                u[base + c] = (g_vec[base + c] as f64) * rs * cs;
                            }
                        }
                        if vr.iter().any(|v| !v.is_finite()) || vc.iter().any(|v| !v.is_finite()) {
                            return Err(Error::Invalid(format!(
                                "adafactor: non-finite factored moment for `{name}`"
                            )));
                        }
                        StagedSecond::Factored(vr, vc)
                    }
                    SecondMoment::Unfactored(v_buf) => {
                        let mut v_vec = v_buf.to_vec::<f32>()?;
                        let one_minus_b2 = 1.0 - beta2_t;
                        for i in 0..n {
                            let gv = g_vec[i] as f64;
                            let v = beta2_t * (v_vec[i] as f64) + one_minus_b2 * (gv * gv + eps1);
                            v_vec[i] = clamp_f32_finite(v);
                            u[i] = gv / v.sqrt();
                        }
                        if v_vec.iter().any(|v| !v.is_finite()) {
                            return Err(Error::Invalid(format!(
                                "adafactor: non-finite second moment for `{name}`"
                            )));
                        }
                        StagedSecond::Unfactored(v_vec)
                    }
                };

                // Overflow-safe RMS update clipping: u /= max(1, RMS(u) / clip_threshold).
                let rms_u = rms_f64_safe(&u);
                let clip_div = (rms_u / clip_thr).max(1.0);
                let inv_clip = 1.0 / clip_div;
                for v in &mut u {
                    *v *= inv_clip * alpha_t;
                }

                // Optional first-moment EMA on the scaled update `alpha_t * u_clip` (Transformers parity).
                let staged_m = if let (Some(b1), Some(m_buf)) = (cfg.beta1, &ps.exp_avg) {
                    let mut m_vec = m_buf.to_vec::<f32>()?;
                    let one_minus_b1 = 1.0 - b1;
                    for i in 0..n {
                        let m = b1 * (m_vec[i] as f64) + one_minus_b1 * u[i];
                        m_vec[i] = m as f32;
                        u[i] = m;
                    }
                    if m_vec.iter().any(|v| !v.is_finite()) {
                        return Err(Error::Invalid(format!(
                            "adafactor: non-finite first moment for `{name}`"
                        )));
                    }
                    Some(m_vec)
                } else {
                    None
                };

                // Decoupled weight decay and parameter step.
                let decay = if wd != 0.0 { 1.0 - alpha_t * wd } else { 1.0 };
                let mut p_out = vec![0.0f32; n];
                for i in 0..n {
                    p_out[i] = ((p_vec[i] as f64) * decay - u[i]) as f32;
                }
                if p_out.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Invalid(format!(
                        "adafactor: non-finite parameter update for `{name}`"
                    )));
                }
                updates.push(StagedUpdate {
                    gi,
                    pi,
                    new_step: t,
                    second: staged_second,
                    exp_avg: staged_m,
                    param: Some(p_out),
                });
            }
        }

        // Commit all validated updates.
        for upd in updates {
            let (_, p) = &self.groups[upd.gi].params[upd.pi];
            let ps = &mut self.state[upd.gi][upd.pi];
            let dev = p.device();
            match (&mut ps.second, upd.second) {
                (SecondMoment::Factored { row, col }, StagedSecond::Factored(vr, vc)) => {
                    let rows = row.dims()[0];
                    let cols = col.dims()[0];
                    row.copy_from(&Tensor::from_vec(vr, [rows])?.to_device(dev)?)?;
                    col.copy_from(&Tensor::from_vec(vc, [cols])?.to_device(dev)?)?;
                }
                (SecondMoment::Unfactored(v_buf), StagedSecond::Unfactored(v_vec)) => {
                    v_buf.copy_from(&Tensor::from_vec(v_vec, p.shape().clone())?.to_device(dev)?)?;
                }
                (_, StagedSecond::Empty) => {}
                _ => unreachable!("staged second-moment variant matches parameter state"),
            }
            if let (Some(m_buf), Some(m_vec)) = (&mut ps.exp_avg, upd.exp_avg) {
                m_buf.copy_from(&Tensor::from_vec(m_vec, p.shape().clone())?.to_device(dev)?)?;
            }
            if let Some(p_out) = upd.param {
                p.copy_from(&Tensor::from_vec(p_out, p.shape().clone())?.to_device(dev)?)?;
            }
            ps.param_step = upd.new_step;
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

#[inline]
fn clamp_f32_finite(v: f64) -> f32 {
    if v.is_finite() { v.min(f32::MAX as f64) as f32 } else { v as f32 }
}

fn rms_f64_safe(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = xs.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let sum_sq: f64 = xs.iter().map(|&v| (v / m) * (v / m)).sum();
    m * (sum_sq / xs.len() as f64).sqrt()
}

fn rms_f32_safe(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = xs.iter().fold(0.0f64, |acc, &v| acc.max((v as f64).abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let sum_sq: f64 = xs.iter().map(|&v| ((v as f64) / m).powi(2)).sum();
    m * (sum_sq / xs.len() as f64).sqrt()
}

fn validate_adafactor_scalars(name: &str, lr: f64, wd: f64, relative_step: bool) -> Result<()> {
    let lr_ok = relative_step || (lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite());
    let wd_ok = wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite();
    if !(lr_ok && wd_ok) {
        return Err(Error::Invalid(format!("adafactor group `{name}`: invalid lr={lr}, weight_decay={wd}")));
    }
    Ok(())
}

fn validate_adafactor_group(g: &ParamGroup, relative_step: bool) -> Result<()> {
    validate_adafactor_scalars(&g.name, g.lr, g.weight_decay, relative_step)
}

fn init_state(groups: &[ParamGroup], has_beta1: bool) -> Result<Vec<Vec<ParamState>>> {
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let mut gs = Vec::with_capacity(g.params.len());
        for (_, p) in &g.params {
            let dev = p.device();
            let second = if p.rank() >= 2 && p.numel() > 0 {
                let rows = p.dims()[0];
                let cols = p.numel() / rows;
                SecondMoment::Factored {
                    row: Tensor::zeros_with([rows], DType::F32, dev)?,
                    col: Tensor::zeros_with([cols], DType::F32, dev)?,
                }
            } else {
                SecondMoment::Unfactored(Tensor::zeros_with(p.shape(), DType::F32, dev)?)
            };
            let exp_avg =
                if has_beta1 { Some(Tensor::zeros_with(p.shape(), DType::F32, dev)?) } else { None };
            gs.push(ParamState { second, exp_avg, param_step: 0 });
        }
        out.push(gs);
    }
    Ok(out)
}

fn check_state_alignment(name: &str, p: &Tensor, ps: &ParamState, seen: &mut HashSet<usize>) -> Result<()> {
    let dev = p.device();
    match &ps.second {
        SecondMoment::Factored { row, col } => {
            if p.rank() < 2 || p.numel() == 0 {
                return Err(Error::Invalid(format!(
                    "adafactor: parameter `{name}` shape {:?} no longer matches factored state",
                    p.dims()
                )));
            }
            let rows = p.dims()[0];
            let cols = p.numel() / rows;
            if row.dims() != [rows] || row.dtype() != DType::F32 || row.device() != dev {
                return Err(Error::Invalid(format!("adafactor: `{name}.exp_avg_sq_row` is misaligned")));
            }
            if col.dims() != [cols] || col.dtype() != DType::F32 || col.device() != dev {
                return Err(Error::Invalid(format!("adafactor: `{name}.exp_avg_sq_col` is misaligned")));
            }
            if !seen.insert(storage_key(row)) || !seen.insert(storage_key(col)) {
                return Err(Error::Invalid(format!("adafactor: state for `{name}` aliases another buffer")));
            }
        }
        SecondMoment::Unfactored(v) => {
            if v.shape() != p.shape() || v.dtype() != DType::F32 || v.device() != dev {
                return Err(Error::Invalid(format!("adafactor: `{name}.exp_avg_sq` is misaligned")));
            }
            if !seen.insert(storage_key(v)) {
                return Err(Error::Invalid(format!("adafactor: state for `{name}` aliases another buffer")));
            }
        }
    }
    if let Some(m) = &ps.exp_avg {
        if m.shape() != p.shape() || m.dtype() != DType::F32 || m.device() != dev {
            return Err(Error::Invalid(format!("adafactor: `{name}.exp_avg` is misaligned")));
        }
        if !seen.insert(storage_key(m)) {
            return Err(Error::Invalid(format!("adafactor: `{name}.exp_avg` aliases another buffer")));
        }
    }
    Ok(())
}
