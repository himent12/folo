//! RAdam (Rectified Adam; Liu et al., ICLR 2020 / arXiv:1908.03265).
//!
//! RAdam tracks the analytical variance of the adaptive learning rate via the approximated Simple Moving
//! Average (SMA) length `rho_t`:
//!
//! ```text
//! rho_inf = 2 / (1 - beta2) - 1
//! rho_t   = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
//! ```
//!
//! For each parameter that receives a gradient at its active step `t >= 1`:
//! ```text
//! m_t = beta1 * m_{t-1} + (1 - beta1) * g_eff
//! v_t = beta2 * v_{t-1} + (1 - beta2) * g_eff^2
//! m_hat = m_t / (1 - prod_{j=1}^t beta1_j)
//!
//! If rho_t > 4 (tractable adaptive variance — rectified step):
//!   r_t   = sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
//!   theta = theta - lr * r_t * m_hat / (sqrt(v_t / (1 - beta2^t)) + eps)
//! Else (rho_t <= 4 — unrectified momentum step):
//!   theta = theta - lr * m_hat
//! ```
//!
//! Supports both AdamW-style decoupled weight decay (`decoupled_wd = true`, default) and classic coupled
//! L2 weight decay (`decoupled_wd = false`). Parameters without a gradient in a step are skipped completely
//! (no weight decay, no moment update, and no per-parameter step increment).

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradRef, GradSource, ParamGroup, ParamPlan, check_param_sets, cpu_fast, for_each_param_planned,
    storage_key, take_state, validate_groups,
};
use crate::{Error, Result};

/// Default RAdam learning rate (`1e-3`).
pub const RADAM_DEFAULT_LR: f64 = 1e-3;
/// Default RAdam first-moment retention `beta1` (`0.9`).
pub const RADAM_DEFAULT_BETA1: f64 = 0.9;
/// Default RAdam second-moment retention `beta2` (`0.999`).
pub const RADAM_DEFAULT_BETA2: f64 = 0.999;
/// Default RAdam numerical stability epsilon (`1e-8`).
pub const RADAM_DEFAULT_EPS: f64 = 1e-8;
/// Default RAdam weight decay (`0.0`).
pub const RADAM_DEFAULT_WEIGHT_DECAY: f64 = 0.0;

/// Analytical RAdam variance-rectification quantities for active step `t >= 1` and retention `beta2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamRectification {
    /// Maximum SMA length `rho_inf = 2 / (1 - beta2) - 1`.
    pub rho_inf: f64,
    /// Current SMA length `rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)`.
    pub rho_t: f64,
    /// `1 - beta2^t` computed without catastrophic cancellation.
    pub one_minus_beta2_t: f64,
    /// Rectification factor `r_t in (0, 1]` when `rho_t > 4`, or `None` for unrectified steps (`rho_t <= 4`).
    pub rect: Option<f64>,
}

/// Compute RAdam's exact analytical variance-rectification quantities `(rho_inf, rho_t, r_t)` in `f64`
/// for active step `step >= 1` and `beta2 in [0, 1)`.
///
/// Uses `-expm1(t * ln(beta2))` to avoid cancellation when `beta2` is close to `1` and `t` is small, and
/// cleanly saturates to `(rho_t = rho_inf, r_t = 1.0)` once `beta2^t` underflows to `0.0`.
pub fn radam_rectification(step: u64, beta2: f64) -> Result<RAdamRectification> {
    if step == 0 {
        return Err(Error::Invalid("radam_rectification: step must be >= 1, got 0".into()));
    }
    if !(beta2.is_finite() && (0.0..1.0).contains(&beta2)) {
        return Err(Error::Invalid(format!(
            "radam_rectification: beta2 must be finite and in [0, 1), got {beta2}"
        )));
    }
    let one_minus_b2 = 1.0 - beta2;
    let rho_inf = 2.0 / one_minus_b2 - 1.0;
    if beta2 == 0.0 {
        return Ok(RAdamRectification { rho_inf, rho_t: 1.0, one_minus_beta2_t: 1.0, rect: None });
    }
    let t = step as f64;
    let ln_b2 = (-one_minus_b2).ln_1p();
    let u = -ln_b2;
    let t_ln_b2 = t * ln_b2;
    let b2_t = t_ln_b2.exp();
    let one_minus_b2_t = -t_ln_b2.exp_m1();
    let rho_t = if b2_t == 0.0 {
        rho_inf
    } else if step == 1 {
        1.0
    } else if t * u < 1e-3 {
        // Exact small-argument expansion cancelling the 2/u singularity in `rho_inf - 2*t*b2^t/(1-b2^t)`:
        //   rho_t = t - (t^2 - 1)*u/6 + (t^4 - 1)*u^3/360 - O((t*u)^5 / t)
        // Machine-accurate in `f64` even when `beta2` is within a few ULPs of `1.0`.
        let t2 = t * t;
        let u2 = u * u;
        (t - ((t2 - 1.0) * u) / 6.0 + ((t2 * t2 - 1.0) * u * u2) / 360.0).clamp(1.0, rho_inf)
    } else {
        (rho_inf - (2.0 * t * b2_t) / one_minus_b2_t).clamp(1.0, rho_inf)
    };
    let rect = if rho_t > 4.0 && rho_inf > 4.0 {
        if b2_t == 0.0 || (rho_inf - rho_t) == 0.0 {
            Some(1.0)
        } else {
            let ratio =
                ((rho_t - 4.0) / (rho_inf - 4.0)) * ((rho_t - 2.0) / (rho_inf - 2.0)) * (rho_inf / rho_t);
            Some(ratio.sqrt())
        }
    } else {
        None
    };
    Ok(RAdamRectification { rho_inf, rho_t, one_minus_beta2_t: one_minus_b2_t, rect })
}

/// Hyper-parameters for constructing a [`RAdam`] optimizer via [`RAdam::from_params`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RAdamConfig {
    /// Learning rate `>= 0` (default `1e-3`).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2]`, each in `[0, 1)` (default `[0.9, 0.999]`).
    pub betas: [f64; 2],
    /// Denominator floor `eps > 0` (must remain positive in `f32`; default `1e-8`).
    pub eps: f64,
    /// Weight decay `>= 0` (default `0.0`).
    pub weight_decay: f64,
    /// `true` (default): AdamW-style decoupled decay `theta *= (1 - lr * wd)` before update.
    /// `false`: coupled L2 decay `g_eff = g + wd * theta`.
    pub decoupled_wd: bool,
}

impl Default for RAdamConfig {
    fn default() -> Self {
        Self {
            lr: RADAM_DEFAULT_LR,
            betas: [RADAM_DEFAULT_BETA1, RADAM_DEFAULT_BETA2],
            eps: RADAM_DEFAULT_EPS,
            weight_decay: RADAM_DEFAULT_WEIGHT_DECAY,
            decoupled_wd: true,
        }
    }
}

impl RAdamConfig {
    /// Validate all RAdam hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_radam_hyperparams(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
        )
    }
}

/// Serializable contract for one [`RAdam`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RAdamGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate `>= 0`.
    pub lr: f64,
    /// Current effective learning rate `>= 0` (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base `[beta1, beta2]`, each in `[0, 1)`.
    pub betas: [f64; 2],
    /// Current effective `beta1` in `[0, 1)` (defaults to `betas[0]` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_beta1: Option<f64>,
    /// Numerical stability floor `eps > 0`.
    pub eps: f64,
    /// Weight decay `>= 0`.
    pub weight_decay: f64,
    /// Decoupled (`true`) vs coupled L2 (`false`) weight decay.
    pub decoupled_wd: bool,
    /// Active step counts per parameter (aligned with `params`).
    pub param_steps: Vec<u64>,
    /// Cumulative `prod_{j=1}^{param_step} beta1_j` per parameter (aligned with `params`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beta1_prods: Vec<f64>,
}

/// Serializable contract for [`RAdam`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RAdamContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<RAdamGroupContract>,
    /// Total completed [`RAdam::step`] calls.
    pub step: u64,
}

struct ParamState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    param_step: u64,
    beta1_prod: f64,
}

struct StagedGroup {
    base_lr: f64,
    lr: f64,
    base_momentum: f64,
    momentum: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    decoupled_wd: bool,
}

struct PendingResume {
    step: u64,
    group_names: Vec<String>,
    param_names: Vec<Vec<String>>,
    groups: Vec<StagedGroup>,
    param_steps: Vec<Vec<u64>>,
    beta1_prods: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
struct GroupCoef {
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
    decoupled_wd: bool,
}

/// RAdam (Rectified Adam) optimizer (Liu et al., ICLR 2020).
///
/// Uses [`ParamGroup::momentum`] / [`ParamGroup::base_momentum`] for `beta1`, [`ParamGroup::beta2`] for
/// `beta2`, [`ParamGroup::eps`] for `eps`, [`ParamGroup::weight_decay`] for `weight_decay`, and
/// [`ParamGroup::decoupled_wd`] for decoupled (`true`) vs coupled L2 (`false`) weight decay.
pub struct RAdam {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    pending_resume: Option<PendingResume>,
    plan: ParamPlan,
    coefs: Vec<GroupCoef>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl RAdam {
    /// Create a [`RAdam`] optimizer over `groups`.
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        validate_groups(&groups)?;
        for g in &groups {
            validate_radam_hyperparams(&g.name, g.base_lr, g.base_momentum, g.beta2, g.eps, g.weight_decay)?;
            validate_radam_hyperparams(&g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "radam: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = zero_state(&groups)?;
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(groups.len());
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 2).sum());
        Ok(RAdam { groups, state, pending_resume: None, plan, coefs, state_keys, step: 0 })
    }

    /// Convenience constructor for a single `"params"` group configured by [`RAdamConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: RAdamConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(cfg.decoupled_wd);
        Self::new(vec![group])
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (e.g. for LR / momentum schedules).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Total completed [`RAdam::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count (incremented only when the parameter receives a gradient).
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// `(exp_avg, exp_avg_sq)` moment buffers for `(group, param)`.
    pub fn moments(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        self.state.get(group)?.get(param).map(|s| (&s.exp_avg, &s.exp_avg_sq))
    }

    /// Perform one RAdam optimization step over every parameter with a gradient in `grads`.
    ///
    /// Validates all groups, hyperparameters, parameter/state alignments, and incoming gradients before
    /// mutating any parameter or state tensor.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "radam: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_radam_hyperparams(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                grp.eps,
                grp.weight_decay,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "radam: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "radam: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "radam: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                for (sname, buf) in [("exp_avg", &ps.exp_avg), ("exp_avg_sq", &ps.exp_avg_sq)] {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "radam: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
                            p.dims(),
                            p.device(),
                            buf.dims(),
                            buf.device(),
                            buf.dtype()
                        )));
                    }
                    self.state_keys.insert(storage_key(buf));
                }
                if let Some(g) = grads.grad(p) {
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "radam: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "radam: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "radam: gradient of `{name}` has non-float dtype {}",
                            g.dtype()
                        )));
                    }
                }
            }
        }

        for grp in &self.groups {
            for (name, p) in &grp.params {
                if self.state_keys.contains(&storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "radam: parameter `{name}` aliases an optimizer state buffer"
                    )));
                }
            }
        }
        let aliases_state = self
            .groups
            .iter()
            .flat_map(|g| &g.params)
            .any(|(_, p)| grads.grad(p).is_some_and(|g| self.state_keys.contains(&storage_key(g))));
        self.plan.sync(&self.groups);
        self.coefs.clear();
        for g in &self.groups {
            self.coefs.push(GroupCoef {
                lr: g.lr,
                b1: g.momentum,
                b2: g.beta2,
                eps: g.eps,
                wd: g.weight_decay,
                decoupled_wd: g.decoupled_wd,
            });
        }

        let _ng = no_grad();
        let (groups, state, plan, coefs) = (&self.groups, &self.state, &self.plan, &self.coefs);
        if aliases_state {
            for (gi, grp) in groups.iter().enumerate() {
                for (pi, (_, p)) in grp.params.iter().enumerate() {
                    let Some(g) = grads.grad(p) else { continue };
                    radam_step_one(p, g, &state[gi][pi], coefs[gi])?;
                }
            }
        } else {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                radam_step_one(p, g, &state[gi][pi], coefs[gi])
            })?;
        }
        for (grp, st) in self.groups.iter().zip(&mut self.state) {
            let b1 = grp.momentum;
            for ((_, p), ps) in grp.params.iter().zip(st) {
                if grads.grad(p).is_some() {
                    ps.param_step = ps.param_step.saturating_add(1);
                    ps.beta1_prod *= b1;
                }
            }
        }
        self.pending_resume = None;
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
        for grp in &self.state {
            for ps in grp {
                ps.exp_avg.to_device_(device)?;
                ps.exp_avg_sq.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Export the serializable [`RAdamContract`].
    pub fn contract(&self) -> RAdamContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.state)
            .map(|(g, st)| RAdamGroupContract {
                name: g.name.clone(),
                params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                lr: g.base_lr,
                current_lr: (g.lr != g.base_lr).then_some(g.lr),
                betas: [g.base_momentum, g.beta2],
                current_beta1: (g.momentum != g.base_momentum).then_some(g.momentum),
                eps: g.eps,
                weight_decay: g.weight_decay,
                decoupled_wd: g.decoupled_wd,
                param_steps: st.iter().map(|ps| ps.param_step).collect(),
                beta1_prods: st.iter().map(|ps| ps.beta1_prod).collect(),
            })
            .collect();
        RAdamContract { groups, step: self.step }
    }

    /// Validate `c` against this optimizer's parameter sets and stage its hyperparameters and per-parameter
    /// step counts / `beta1` products for atomic commit in [`RAdam::load_state`].
    pub fn apply_contract(&mut self, c: &RAdamContract) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Load(format!(
                "radam: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        for (g, st) in self.groups.iter().zip(&self.state) {
            if g.params.len() != st.len() {
                return Err(Error::Load(format!(
                    "radam: group `{}` parameter count {} does not match state {}",
                    g.name,
                    g.params.len(),
                    st.len()
                )));
            }
        }
        let as_group_contracts: Vec<super::GroupContract> = c
            .groups
            .iter()
            .map(|gc| super::GroupContract {
                name: gc.name.clone(),
                params: gc.params.clone(),
                lr: gc.lr,
                momentum: None,
                nesterov: None,
                weight_decay: gc.weight_decay,
                betas: Some(gc.betas),
                eps: Some(gc.eps),
                decoupled_wd: gc.decoupled_wd,
                cautious_wd: false,
                cautious: false,
            })
            .collect();
        check_param_sets(&self.groups, &as_group_contracts)?;

        let mut staged_steps = Vec::with_capacity(self.groups.len());
        let mut staged_prods = Vec::with_capacity(self.groups.len());
        for (gc, grp) in c.groups.iter().zip(&self.groups) {
            if gc.param_steps.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "radam contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            if !gc.beta1_prods.is_empty() && gc.beta1_prods.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "radam contract group `{}`: beta1_prods length {} != {} params",
                    gc.name,
                    gc.beta1_prods.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_radam_hyperparams(&gc.name, gc.lr, gc.betas[0], gc.betas[1], gc.eps, gc.weight_decay)
                .map_err(|e| Error::Load(e.to_string()))?;
            validate_radam_hyperparams(&gc.name, cur_lr, cur_b1, gc.betas[1], gc.eps, gc.weight_decay)
                .map_err(|e| Error::Load(e.to_string()))?;

            let mut by_name: HashMap<&str, (u64, f64)> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "radam contract group `{}` param `{pname}`: param_step {k} > step {}",
                        gc.name, c.step
                    )));
                }
                let prod = if gc.beta1_prods.is_empty() {
                    if k == 0 {
                        1.0
                    } else if gc.betas[0] == 0.0 {
                        0.0
                    } else {
                        ((k as f64) * gc.betas[0].ln()).exp()
                    }
                } else {
                    let p = gc.beta1_prods[idx];
                    if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                        return Err(Error::Load(format!(
                            "radam contract group `{}` param `{pname}`: beta1_prod must be in [0, 1], got {p}",
                            gc.name
                        )));
                    }
                    if k == 0 && p != 1.0 {
                        return Err(Error::Load(format!(
                            "radam contract group `{}` param `{pname}`: param_step == 0 requires beta1_prod == 1.0, got {p}",
                            gc.name
                        )));
                    }
                    if k > 0 && p >= 1.0 {
                        return Err(Error::Load(format!(
                            "radam contract group `{}` param `{pname}`: param_step > 0 requires beta1_prod < 1.0, got {p}",
                            gc.name
                        )));
                    }
                    p
                };
                by_name.insert(pname.as_str(), (k, prod));
            }
            let mut grp_steps = Vec::with_capacity(grp.params.len());
            let mut grp_prods = Vec::with_capacity(grp.params.len());
            for (pname, _) in &grp.params {
                let (k, prod) = by_name[pname.as_str()];
                grp_steps.push(k);
                grp_prods.push(prod);
            }
            staged_steps.push(grp_steps);
            staged_prods.push(grp_prods);
        }

        let staged_groups: Vec<StagedGroup> = c
            .groups
            .iter()
            .map(|gc| StagedGroup {
                base_lr: gc.lr,
                lr: gc.current_lr.unwrap_or(gc.lr),
                base_momentum: gc.betas[0],
                momentum: gc.current_beta1.unwrap_or(gc.betas[0]),
                beta2: gc.betas[1],
                eps: gc.eps,
                weight_decay: gc.weight_decay,
                decoupled_wd: gc.decoupled_wd,
            })
            .collect();
        self.pending_resume = Some(PendingResume {
            step: c.step,
            group_names: self.groups.iter().map(|g| g.name.clone()).collect(),
            param_names: self
                .groups
                .iter()
                .map(|g| g.params.iter().map(|(n, _)| n.clone()).collect())
                .collect(),
            groups: staged_groups,
            param_steps: staged_steps,
            beta1_prods: staged_prods,
        });
        Ok(())
    }

    /// Export all optimizer state tensors (`<param>.exp_avg`, `<param>.exp_avg_sq`).
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(self.groups.iter().map(|g| g.params.len() * 2).sum());
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), ps.exp_avg.clone()));
                out.push((format!("{name}.exp_avg_sq"), ps.exp_avg_sq.clone()));
            }
        }
        Ok(out)
    }

    /// Restore step counter and state tensors from a checkpoint map.
    ///
    /// Resuming at `step > 0` requires staging the matching [`RAdamContract`] via [`RAdam::apply_contract`]
    /// first so per-parameter active step counts and cumulative `beta1` products are restored accurately.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "RAdam state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "RAdam state: step {step} does not match staged contract step {}",
                    pending.step
                )));
            }
            let same_layout = self.groups.len() == pending.group_names.len()
                && self.groups.iter().zip(&pending.group_names).zip(&pending.param_names).all(
                    |((g, gname), pnames)| {
                        g.name == *gname
                            && g.params.len() == pnames.len()
                            && g.params.iter().zip(pnames).all(|((n, _), pn)| n == pn)
                    },
                );
            if !same_layout {
                return Err(Error::Load(
                    "RAdam state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                let req = p_step > 0;
                for (k, what) in ["exp_avg", "exp_avg_sq"].into_iter().enumerate() {
                    if let Some(t) = take_state(state, name, p, what, req, &mut problems) {
                        if k == 1 {
                            let vals = t.to_vec::<f32>()?;
                            if vals.iter().any(|&v| v < 0.0 || v.is_nan()) {
                                problems.push(format!(
                                    "`optim.{name}.exp_avg_sq` contains negative or NaN values"
                                ));
                                continue;
                            }
                        }
                        loaded.push((gi, pi, k, t));
                    }
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("RAdam state: {}", problems.join("; "))));
        }
        let mut fresh = zero_state(&self.groups)?;
        let _ng = no_grad();
        for (gi, pi, k, t) in loaded {
            let ps = &fresh[gi][pi];
            let dst = if k == 0 { &ps.exp_avg } else { &ps.exp_avg_sq };
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        if let Some(pending) = self.pending_resume.take() {
            for (g, sg) in self.groups.iter_mut().zip(pending.groups) {
                g.base_lr = sg.base_lr;
                g.lr = sg.lr;
                g.base_momentum = sg.base_momentum;
                g.momentum = sg.momentum;
                g.beta2 = sg.beta2;
                g.eps = sg.eps;
                g.weight_decay = sg.weight_decay;
                g.decoupled_wd = sg.decoupled_wd;
            }
            for ((st, psteps), bprods) in fresh.iter_mut().zip(pending.param_steps).zip(pending.beta1_prods) {
                for ((ps, k), prod) in st.iter_mut().zip(psteps).zip(bprods) {
                    ps.param_step = k;
                    ps.beta1_prod = prod;
                }
            }
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}

fn validate_radam_hyperparams(group: &str, lr: f64, b1: f64, b2: f64, eps: f64, wd: f64) -> Result<()> {
    if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "radam group `{group}`: lr must be finite in f32 and >= 0, got {lr}"
        )));
    }
    if !(b1.is_finite() && (0.0..1.0).contains(&b1) && (b1 as f32) < 1.0 && ((1.0 - b1) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "radam group `{group}`: beta1 must be finite in [0, 1) with f32(beta1) < 1, got {b1}"
        )));
    }
    if !(b2.is_finite() && (0.0..1.0).contains(&b2) && (b2 as f32) < 1.0 && ((1.0 - b2) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "radam group `{group}`: beta2 must be finite in [0, 1) with f32(beta2) < 1, got {b2}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "radam group `{group}`: eps must be finite and > 0 in f32, got {eps}"
        )));
    }
    if !(wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite() && ((1.0 - lr * wd) as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "radam group `{group}`: weight_decay must be finite in f32 and >= 0 with finite (1 - lr * wd), got {wd}"
        )));
    }
    Ok(())
}

fn zero_state(groups: &[ParamGroup]) -> Result<Vec<Vec<ParamState>>> {
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(_, p)| {
                    let dev = p.device();
                    let shape = p.shape();
                    Ok(ParamState {
                        exp_avg: Tensor::zeros_with(shape, DType::F32, dev)?,
                        exp_avg_sq: Tensor::zeros_with(shape, DType::F32, dev)?,
                        param_step: 0,
                        beta1_prod: 1.0,
                    })
                })
                .collect()
        })
        .collect()
}

fn radam_step_one(p: &Tensor, g: &Tensor, ps: &ParamState, c: GroupCoef) -> Result<()> {
    let t = ps.param_step.saturating_add(1);
    let b1_prod = ps.beta1_prod * c.b1;
    let bc1 = 1.0 - b1_prod;
    let info = radam_rectification(t, c.b2)?;

    let b1 = c.b1 as f32;
    let one_minus_b1 = (1.0 - c.b1) as f32;
    let b2 = c.b2 as f32;
    let one_minus_b2 = (1.0 - c.b2) as f32;
    let eps = c.eps as f32;
    let wd = c.wd as f32;
    let decay = (c.decoupled_wd && c.wd != 0.0).then_some((1.0 - c.lr * c.wd) as f32);
    let coupled_wd = !c.decoupled_wd && c.wd != 0.0;
    let bc2_sqrt = info.one_minus_beta2_t.sqrt() as f32;
    let step_size = match info.rect {
        Some(r_t) => (c.lr * r_t / bc1) as f32,
        None => (c.lr / bc1) as f32,
    };
    let rectified = info.rect.is_some();

    let (m, v) = (&ps.exp_avg, &ps.exp_avg_sq);
    let can_fast = cpu_fast(&[p, m, v])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !p.same_storage(m)
        && !p.same_storage(v)
        && !m.same_storage(g)
        && !m.same_storage(v)
        && !v.same_storage(g);

    if can_fast {
        return g.with_f32_slice(|gs| -> Result<()> {
            m.update_f32(|ms| {
                v.update_f32(|vs| {
                    p.update_f32(|ps_slice| {
                        for (((pi, &gi), mi), vi) in
                            ps_slice.iter_mut().zip(gs).zip(ms.iter_mut()).zip(vs.iter_mut())
                        {
                            let g_eff = if coupled_wd { gi + wd * *pi } else { gi };
                            if let Some(d) = decay {
                                *pi *= d;
                            }
                            *mi = b1 * *mi + one_minus_b1 * g_eff;
                            *vi = b2 * *vi + one_minus_b2 * (g_eff * g_eff);
                            if rectified {
                                let denom = (*vi).sqrt() / bc2_sqrt + eps;
                                *pi -= step_size * (*mi / denom);
                            } else {
                                *pi -= step_size * *mi;
                            }
                        }
                    })
                })
            })???;
            Ok(())
        })?;
    }

    // Device-generic / aliased fallback using out-of-place tensor ops under `no_grad` so
    // `g`, `p`, `m`, and `v` can alias arbitrarily without clobbering pre-step values.
    let g_ref = GradRef::as_f32(g)?;
    let g_eff =
        if coupled_wd { g_ref.as_tensor().add(&p.mul_scalar(wd)?)? } else { g_ref.as_tensor().clone() };
    let p_base = if let Some(d) = decay { p.mul_scalar(d)? } else { p.clone() };
    let m_new = m.mul_scalar(b1)?.add(&g_eff.mul_scalar(one_minus_b1)?)?;
    let v_new = v.mul_scalar(b2)?.add(&g_eff.square()?.mul_scalar(one_minus_b2)?)?;
    let p_new = if rectified {
        let denom = v_new.sqrt()?.div_scalar(bc2_sqrt)?.add_scalar(eps)?;
        p_base.sub(&m_new.div(&denom)?.mul_scalar(step_size)?)?
    } else {
        p_base.sub(&m_new.mul_scalar(step_size)?)?
    };
    p.copy_from(&p_new)?;
    m.copy_from(&m_new)?;
    v.copy_from(&v_new)?;
    Ok(())
}
