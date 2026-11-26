//! Adan (Adaptive Nesterov Momentum; Xie et al., IEEE TPAMI 2024 / arXiv:2208.06677).
//!
//! Adan maintains three moment buffers per parameter — first moment `m_k`, gradient-difference moment
//! `v_k`, and Nesterov-extrapolated second moment `n_k` — plus the previous gradient `g_{k-1}`:
//!
//! ```text
//! diff_k = 0                                               if k == 1 (first active step for param)
//!        = g_k - g_{k-1}                                   if k >= 2
//! m_k    = beta1 * m_{k-1} + (1 - beta1) * g_k
//! v_k    = beta2 * v_{k-1} + (1 - beta2) * diff_k
//! u_k    = g_k + beta2 * diff_k
//! n_k    = beta3 * n_{k-1} + (1 - beta3) * u_k^2
//!
//! m_hat  = m_k / (1 - beta1^k)
//! v_hat  = v_k / (1 - beta2^k)
//! n_hat  = n_k / (1 - beta3^k)
//! step_k = (m_hat + beta2 * v_hat) / (sqrt(n_hat) + eps)
//!
//! Decoupled weight decay (default, `decoupled_wd = true`):
//!   theta_k = theta_{k-1} * (1 - lr * wd) - lr * step_k
//! Proximal weight decay (`decoupled_wd = false`, Algorithm 1 of Xie et al.):
//!   theta_k = (theta_{k-1} - lr * step_k) / (1 + lr * wd)
//! ```
//!
//! Following PyTorch / `sail-sg/adan` and FOLO's other optimizers (`AdamW`, `Lion`, `ScheduleFreeAdamW`),
//! `(beta1, beta2, beta3)` are EMA retention factors in `[0, 1)` with defaults `(0.98, 0.92, 0.99)`
//! (corresponding to `(1 - 0.02, 1 - 0.08, 1 - 0.01)` in paper notation). Parameters with no gradient
//! in a step are skipped completely (no weight decay, no state update, and no per-parameter step
//! increment).

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradSource, ParamGroup, ParamPlan, check_param_sets, cpu_fast, for_each_param_planned, storage_key,
    take_state, validate_groups,
};
use crate::{Error, Result};

/// Default Adan learning rate.
pub const ADAN_DEFAULT_LR: f64 = 1e-3;
/// Default Adan first-moment EMA retention `beta1` (`1 - 0.02` in Xie et al. 2024).
pub const ADAN_DEFAULT_BETA1: f64 = 0.98;
/// Default Adan gradient-difference EMA retention `beta2` (`1 - 0.08` in Xie et al. 2024).
pub const ADAN_DEFAULT_BETA2: f64 = 0.92;
/// Default Adan second-moment EMA retention `beta3` (`1 - 0.01` in Xie et al. 2024).
pub const ADAN_DEFAULT_BETA3: f64 = 0.99;
/// Default Adan numerical stability epsilon.
pub const ADAN_DEFAULT_EPS: f64 = 1e-8;
/// Default Adan weight decay.
pub const ADAN_DEFAULT_WEIGHT_DECAY: f64 = 0.02;

/// Hyper-parameters for constructing an [`Adan`] optimizer via [`Adan::from_params`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdanConfig {
    /// Learning rate `>= 0` (default `1e-3`).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2, beta3]`, each in `[0, 1)` (default `[0.98, 0.92, 0.99]`).
    pub betas: [f64; 3],
    /// Denominator floor `eps > 0` (must remain positive in `f32`; default `1e-8`).
    pub eps: f64,
    /// Weight decay `>= 0` (default `0.02`).
    pub weight_decay: f64,
    /// `true` (default): AdamW-style decoupled decay `theta * (1 - lr * wd) - lr * step`.
    /// `false`: proximal decay `(theta - lr * step) / (1 + lr * wd)` (Xie et al. Algorithm 1).
    pub decoupled_wd: bool,
}

impl Default for AdanConfig {
    fn default() -> Self {
        Self {
            lr: ADAN_DEFAULT_LR,
            betas: [ADAN_DEFAULT_BETA1, ADAN_DEFAULT_BETA2, ADAN_DEFAULT_BETA3],
            eps: ADAN_DEFAULT_EPS,
            weight_decay: ADAN_DEFAULT_WEIGHT_DECAY,
            decoupled_wd: true,
        }
    }
}

impl AdanConfig {
    /// Validate all Adan hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_adan_hyperparams(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.betas[2],
            self.eps,
            self.weight_decay,
        )
    }
}

/// Serializable contract for one [`Adan`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdanGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate `>= 0`.
    pub lr: f64,
    /// Current effective learning rate `>= 0` (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base `[beta1, beta2, beta3]`, each in `[0, 1)`.
    pub betas: [f64; 3],
    /// Current effective `beta1` in `[0, 1)` (defaults to `betas[0]` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_beta1: Option<f64>,
    /// Numerical stability floor `eps > 0`.
    pub eps: f64,
    /// Weight decay `>= 0`.
    pub weight_decay: f64,
    /// Decoupled (`true`) vs proximal (`false`) weight decay.
    pub decoupled_wd: bool,
    /// Active step counts per parameter (aligned with `params`).
    pub param_steps: Vec<u64>,
    /// Cumulative `prod_{j=1}^{param_step} beta1_j` per parameter (aligned with `params`) for exact
    /// bias correction under schedules that vary `beta1`. Defaults to `betas[0]^param_step` if omitted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beta1_prods: Vec<f64>,
}

/// Serializable contract for [`Adan`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdanContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<AdanGroupContract>,
    /// Total completed [`Adan::step`] calls.
    pub step: u64,
}

struct ParamState {
    exp_avg: Tensor,
    exp_avg_diff: Tensor,
    exp_avg_sq: Tensor,
    pre_grad: Tensor,
    param_step: u64,
    beta1_prod: f64,
}

struct PendingResume {
    step: u64,
    param_steps: Vec<Vec<u64>>,
    beta1_prods: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
struct GroupCoef {
    lr: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    eps: f64,
    wd: f64,
    decoupled_wd: bool,
}

/// Adan (Adaptive Nesterov Momentum) optimizer (Xie et al., IEEE TPAMI 2024).
///
/// Uses [`ParamGroup::momentum`] / [`ParamGroup::base_momentum`] for `beta1`, [`ParamGroup::beta2`] for
/// `beta2`, [`ParamGroup::eps`] for `eps`, [`ParamGroup::weight_decay`] for `weight_decay`,
/// [`ParamGroup::decoupled_wd`] for decoupled vs proximal decay, and an explicit per-group `beta3`
/// (default [`ADAN_DEFAULT_BETA3`]). Integrates directly with [`crate::LrSchedule::apply_to_groups`]
/// and [`crate::SgdrSchedule::apply_to_groups`].
pub struct Adan {
    groups: Vec<ParamGroup>,
    beta3: Vec<f64>,
    state: Vec<Vec<ParamState>>,
    pending_resume: Option<PendingResume>,
    plan: ParamPlan,
    coefs: Vec<GroupCoef>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl Adan {
    /// Create an [`Adan`] optimizer over `groups` sharing the same `beta3` in `[0, 1)`.
    pub fn new(groups: Vec<ParamGroup>, beta3: f64) -> Result<Self> {
        let beta3s = vec![beta3; groups.len()];
        Self::with_group_beta3(groups, beta3s)
    }

    /// Create an [`Adan`] optimizer with an explicit `beta3` per parameter group.
    pub fn with_group_beta3(groups: Vec<ParamGroup>, beta3: Vec<f64>) -> Result<Self> {
        if groups.len() != beta3.len() {
            return Err(Error::Invalid(format!(
                "adan: beta3 length {} does not match {} parameter groups",
                beta3.len(),
                groups.len()
            )));
        }
        validate_groups(&groups)?;
        for (g, &b3) in groups.iter().zip(&beta3) {
            validate_adan_hyperparams(
                &g.name,
                g.base_lr,
                g.base_momentum,
                g.beta2,
                b3,
                g.eps,
                g.weight_decay,
            )?;
            validate_adan_hyperparams(&g.name, g.lr, g.momentum, g.beta2, b3, g.eps, g.weight_decay)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adan: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = zero_state(&groups)?;
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(groups.len());
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 4).sum());
        Ok(Adan { groups, beta3, state, pending_resume: None, plan, coefs, state_keys, step: 0 })
    }

    /// Convenience constructor for a single `"params"` group configured by [`AdanConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: AdanConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(cfg.decoupled_wd);
        Self::new(vec![group], cfg.betas[2])
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (for LR schedules such as [`crate::SgdrSchedule::apply_to_groups`]).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Per-group `beta3` retention factors.
    pub fn beta3(&self) -> &[f64] {
        &self.beta3
    }

    /// Total completed [`Adan::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count (incremented only when the parameter receives a gradient).
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// First-moment buffer `m_k` (`exp_avg`) for `(group, param)`.
    pub fn exp_avg(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.state.get(group)?.get(param).map(|s| &s.exp_avg)
    }

    /// Gradient-difference moment buffer `v_k` (`exp_avg_diff`) for `(group, param)`.
    pub fn exp_avg_diff(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.state.get(group)?.get(param).map(|s| &s.exp_avg_diff)
    }

    /// Second-moment buffer `n_k` (`exp_avg_sq`) for `(group, param)`.
    pub fn exp_avg_sq(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.state.get(group)?.get(param).map(|s| &s.exp_avg_sq)
    }

    /// Previous gradient buffer `g_{k-1}` (`pre_grad`) for `(group, param)`.
    pub fn pre_grad(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.state.get(group)?.get(param).map(|s| &s.pre_grad)
    }

    /// Perform one Adan optimization step over every parameter with a gradient in `grads`.
    ///
    /// Validates all groups, hyperparameters, parameter/state alignments, and incoming gradients before
    /// mutating any parameter or state tensor.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if self.groups.len() != self.state.len() || self.groups.len() != self.beta3.len() {
            return Err(Error::Invalid(format!(
                "adan: group count {} does not match state {} / beta3 {}",
                self.groups.len(),
                self.state.len(),
                self.beta3.len()
            )));
        }
        self.state_keys.clear();
        for ((grp, st), &b3) in self.groups.iter().zip(&self.state).zip(&self.beta3) {
            validate_adan_hyperparams(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                b3,
                grp.eps,
                grp.weight_decay,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "adan: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "adan: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adan: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                for (sname, buf) in [
                    ("exp_avg", &ps.exp_avg),
                    ("exp_avg_diff", &ps.exp_avg_diff),
                    ("exp_avg_sq", &ps.exp_avg_sq),
                    ("pre_grad", &ps.pre_grad),
                ] {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "adan: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
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
                            "adan: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "adan: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "adan: gradient of `{name}` has non-float dtype {}",
                            g.dtype()
                        )));
                    }
                }
            }
        }

        let aliases_state = self.groups.iter().flat_map(|g| &g.params).any(|(_, p)| {
            self.state_keys.contains(&storage_key(p))
                || grads.grad(p).is_some_and(|g| self.state_keys.contains(&storage_key(g)))
        });
        self.plan.sync(&self.groups);
        self.coefs.clear();
        for (g, &b3) in self.groups.iter().zip(&self.beta3) {
            self.coefs.push(GroupCoef {
                lr: g.lr,
                b1: g.momentum,
                b2: g.beta2,
                b3,
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
                    adan_step_one(p, g, &state[gi][pi], coefs[gi])?;
                }
            }
        } else {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                adan_step_one(p, g, &state[gi][pi], coefs[gi])
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
                ps.exp_avg_diff.to_device_(device)?;
                ps.exp_avg_sq.to_device_(device)?;
                ps.pre_grad.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Export the serializable [`AdanContract`].
    pub fn contract(&self) -> AdanContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.beta3)
            .zip(&self.state)
            .map(|((g, &b3), st)| AdanGroupContract {
                name: g.name.clone(),
                params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                lr: g.base_lr,
                current_lr: (g.lr != g.base_lr).then_some(g.lr),
                betas: [g.base_momentum, g.beta2, b3],
                current_beta1: (g.momentum != g.base_momentum).then_some(g.momentum),
                eps: g.eps,
                weight_decay: g.weight_decay,
                decoupled_wd: g.decoupled_wd,
                param_steps: st.iter().map(|ps| ps.param_step).collect(),
                beta1_prods: st.iter().map(|ps| ps.beta1_prod).collect(),
            })
            .collect();
        AdanContract { groups, step: self.step }
    }

    /// Validate `c` against this optimizer's parameter sets, adopt its hyperparameters, and stage its
    /// per-parameter step counts and cumulative `beta1` products (mapped by parameter name so reordered
    /// parameter lists resume identically) to be committed atomically by [`Adan::load_state`].
    pub fn apply_contract(&mut self, c: &AdanContract) -> Result<()> {
        if self.groups.len() != self.state.len() || self.groups.len() != self.beta3.len() {
            return Err(Error::Load(format!(
                "adan: group count {} does not match state {} / beta3 {}",
                self.groups.len(),
                self.state.len(),
                self.beta3.len()
            )));
        }
        for (g, st) in self.groups.iter().zip(&self.state) {
            if g.params.len() != st.len() {
                return Err(Error::Load(format!(
                    "adan: group `{}` parameter count {} does not match state {}",
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
                betas: Some([gc.betas[0], gc.betas[1]]),
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
                    "adan contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            if !gc.beta1_prods.is_empty() && gc.beta1_prods.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "adan contract group `{}`: beta1_prods length {} != {} params",
                    gc.name,
                    gc.beta1_prods.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_adan_hyperparams(
                &gc.name,
                gc.lr,
                gc.betas[0],
                gc.betas[1],
                gc.betas[2],
                gc.eps,
                gc.weight_decay,
            )
            .map_err(|e| Error::Load(e.to_string()))?;
            validate_adan_hyperparams(
                &gc.name,
                cur_lr,
                cur_b1,
                gc.betas[1],
                gc.betas[2],
                gc.eps,
                gc.weight_decay,
            )
            .map_err(|e| Error::Load(e.to_string()))?;

            let mut by_name: HashMap<&str, (u64, f64)> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "adan contract group `{}` param `{pname}`: param_step {k} > step {}",
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
                            "adan contract group `{}` param `{pname}`: beta1_prod must be in [0, 1], got {p}",
                            gc.name
                        )));
                    }
                    if k == 0 && p != 1.0 {
                        return Err(Error::Load(format!(
                            "adan contract group `{}` param `{pname}`: param_step == 0 requires beta1_prod == 1.0, got {p}",
                            gc.name
                        )));
                    }
                    if k > 0 && p >= 1.0 {
                        return Err(Error::Load(format!(
                            "adan contract group `{}` param `{pname}`: param_step > 0 requires beta1_prod < 1.0, got {p}",
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

        for ((g, b3), gc) in self.groups.iter_mut().zip(&mut self.beta3).zip(&c.groups) {
            g.base_lr = gc.lr;
            g.lr = gc.current_lr.unwrap_or(gc.lr);
            g.base_momentum = gc.betas[0];
            g.momentum = gc.current_beta1.unwrap_or(gc.betas[0]);
            g.beta2 = gc.betas[1];
            *b3 = gc.betas[2];
            g.eps = gc.eps;
            g.weight_decay = gc.weight_decay;
            g.decoupled_wd = gc.decoupled_wd;
        }
        self.pending_resume =
            Some(PendingResume { step: c.step, param_steps: staged_steps, beta1_prods: staged_prods });
        Ok(())
    }

    /// Export all optimizer state tensors (`<param>.exp_avg`, `<param>.exp_avg_diff`, `<param>.exp_avg_sq`,
    /// `<param>.pre_grad`).
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(self.groups.iter().map(|g| g.params.len() * 4).sum());
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), ps.exp_avg.clone()));
                out.push((format!("{name}.exp_avg_diff"), ps.exp_avg_diff.clone()));
                out.push((format!("{name}.exp_avg_sq"), ps.exp_avg_sq.clone()));
                out.push((format!("{name}.pre_grad"), ps.pre_grad.clone()));
            }
        }
        Ok(out)
    }

    /// Restore step counter and state tensors from a checkpoint map.
    ///
    /// - If [`Adan::apply_contract`] was called beforehand, `step` must match the staged contract's `step`,
    ///   and per-parameter step counts and `beta1` products are committed atomically upon success.
    /// - When `step == 0`, missing state tensors stay zero and every parameter's active step count resets to `0`.
    /// - When `step > 0` without a prior [`Adan::apply_contract`], all state tensors are required and every
    ///   parameter adopts `param_step = step` and `beta1_prod = beta1^step`.
    /// - Any negative value (including `-Inf`) in `exp_avg_sq` is rejected.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(ref pending) = self.pending_resume
            && pending.step != step
        {
            return Err(Error::Load(format!(
                "adan: load_state step {step} does not match applied contract step {}",
                pending.step
            )));
        }
        type LoadedSlot = [Option<Tensor>; 4];
        let mut problems = Vec::new();
        let mut loaded: Vec<Vec<LoadedSlot>> = Vec::with_capacity(self.groups.len());
        let req = step > 0;
        for g in &self.groups {
            let mut grp = Vec::with_capacity(g.params.len());
            for (name, p) in &g.params {
                let m = take_state(state, name, p, "exp_avg", req, &mut problems);
                let v = take_state(state, name, p, "exp_avg_diff", req, &mut problems);
                let n = take_state(state, name, p, "exp_avg_sq", req, &mut problems);
                let pg = take_state(state, name, p, "pre_grad", req, &mut problems);
                if let Some(ref nt) = n {
                    let nv = nt.to_vec::<f32>()?;
                    if nv.iter().any(|&val| val < 0.0) {
                        problems.push(format!("`optim.{name}.exp_avg_sq` contains negative values"));
                    }
                }
                grp.push([m, v, n, pg]);
            }
            loaded.push(grp);
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("adan state load failed: {}", problems.join("; "))));
        }
        let _ng = no_grad();
        let mut fresh = zero_state(&self.groups)?;
        for (st_new, grp_loaded) in fresh.iter_mut().zip(loaded) {
            for (ps_new, [m, v, n, pg]) in st_new.iter_mut().zip(grp_loaded) {
                let dev = ps_new.exp_avg.device();
                if let Some(t) = m {
                    ps_new.exp_avg.copy_from(&t.to_device(dev)?)?;
                }
                if let Some(t) = v {
                    ps_new.exp_avg_diff.copy_from(&t.to_device(dev)?)?;
                }
                if let Some(t) = n {
                    ps_new.exp_avg_sq.copy_from(&t.to_device(dev)?)?;
                }
                if let Some(t) = pg {
                    ps_new.pre_grad.copy_from(&t.to_device(dev)?)?;
                }
            }
        }
        if let Some(pending) = self.pending_resume.take() {
            for ((st_new, steps), prods) in fresh.iter_mut().zip(pending.param_steps).zip(pending.beta1_prods)
            {
                for ((ps_new, k), prod) in st_new.iter_mut().zip(steps).zip(prods) {
                    ps_new.param_step = k;
                    ps_new.beta1_prod = prod;
                }
            }
        } else {
            for (g, st_new) in self.groups.iter().zip(&mut fresh) {
                let prod = if step == 0 {
                    1.0
                } else if g.momentum == 0.0 {
                    0.0
                } else {
                    ((step as f64) * g.momentum.ln()).exp()
                };
                for ps_new in st_new {
                    ps_new.param_step = step;
                    ps_new.beta1_prod = prod;
                }
            }
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}

fn validate_adan_hyperparams(
    name: &str,
    lr: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<()> {
    let lr_wd = lr * weight_decay;
    let ok = lr.is_finite()
        && lr >= 0.0
        && (lr as f32).is_finite()
        && beta1.is_finite()
        && (0.0..1.0).contains(&beta1)
        && (beta1 as f32) < 1.0
        && beta2.is_finite()
        && (0.0..1.0).contains(&beta2)
        && (beta2 as f32) < 1.0
        && beta3.is_finite()
        && (0.0..1.0).contains(&beta3)
        && (beta3 as f32) < 1.0
        && eps.is_finite()
        && eps > 0.0
        && (eps as f32) > 0.0
        && (eps as f32).is_finite()
        && weight_decay.is_finite()
        && weight_decay >= 0.0
        && (weight_decay as f32).is_finite()
        && lr_wd.is_finite()
        && (lr_wd as f32).is_finite()
        && ((1.0 - lr_wd) as f32).is_finite()
        && ((1.0 + lr_wd) as f32).is_finite();
    if !ok {
        return Err(Error::Invalid(format!(
            "adan group `{name}`: invalid hyperparameters (lr={lr}, betas=[{beta1}, {beta2}, {beta3}], eps={eps}, wd={weight_decay})"
        )));
    }
    Ok(())
}

fn zero_state(groups: &[ParamGroup]) -> Result<Vec<Vec<ParamState>>> {
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let mut grp = Vec::with_capacity(g.params.len());
        for (_, p) in &g.params {
            let dev = p.device();
            let s = p.shape();
            grp.push(ParamState {
                exp_avg: Tensor::zeros_with(s, DType::F32, dev)?,
                exp_avg_diff: Tensor::zeros_with(s, DType::F32, dev)?,
                exp_avg_sq: Tensor::zeros_with(s, DType::F32, dev)?,
                pre_grad: Tensor::zeros_with(s, DType::F32, dev)?,
                param_step: 0,
                beta1_prod: 1.0,
            });
        }
        out.push(grp);
    }
    Ok(out)
}

/// Exact bias correction `1 - beta^k` in `f64` without overflow or cancellation for `beta in [0, 1)` and `k >= 1`.
#[inline]
fn bias_correction(beta: f64, k: u64) -> f64 {
    if beta == 0.0 { 1.0 } else { -((k as f64) * beta.ln()).exp_m1() }
}

fn adan_step_one(p: &Tensor, g: &Tensor, ps: &ParamState, c: GroupCoef) -> Result<()> {
    let k = ps.param_step.saturating_add(1);
    let is_first = ps.param_step == 0;
    let bc1 = 1.0 - ps.beta1_prod * c.b1;
    let bc2 = bias_correction(c.b2, k);
    let bc3 = bias_correction(c.b3, k);
    let one_m_b1 = 1.0 - c.b1;
    let one_m_b2 = 1.0 - c.b2;
    let one_m_b3 = 1.0 - c.b3;
    let wd_scale = if c.decoupled_wd { 1.0 - c.lr * c.wd } else { 1.0 / (1.0 + c.lr * c.wd) };

    if cpu_fast(&[p, g, &ps.exp_avg, &ps.exp_avg_diff, &ps.exp_avg_sq, &ps.pre_grad])
        && storage_key(p) != storage_key(g)
    {
        let gv = g.to_vec::<f32>()?;
        let mut mv = ps.exp_avg.to_vec::<f32>()?;
        let mut vv = ps.exp_avg_diff.to_vec::<f32>()?;
        let mut nv = ps.exp_avg_sq.to_vec::<f32>()?;
        let mut pgv = ps.pre_grad.to_vec::<f32>()?;
        let mut pv = p.to_vec::<f32>()?;

        let bc3_sqrt = bc3.sqrt();
        for i in 0..pv.len() {
            let gi = gv[i] as f64;
            let diff = if is_first { 0.0 } else { gi - pgv[i] as f64 };
            let m_new = c.b1 * (mv[i] as f64) + one_m_b1 * gi;
            let v_new = c.b2 * (vv[i] as f64) + one_m_b2 * diff;
            let u = gi + c.b2 * diff;
            let n_new = c.b3 * (nv[i] as f64) + one_m_b3 * (u * u);

            let m_hat = m_new / bc1;
            let v_hat = v_new / bc2;
            let n_hat_sqrt = n_new.sqrt() / bc3_sqrt;
            let denom = n_hat_sqrt + c.eps;
            let step_dir = (m_hat + c.b2 * v_hat) / denom;
            let theta = pv[i] as f64;
            let theta_new = if c.decoupled_wd {
                theta * wd_scale - c.lr * step_dir
            } else {
                (theta - c.lr * step_dir) * wd_scale
            };

            pv[i] = theta_new as f32;
            mv[i] = m_new as f32;
            vv[i] = v_new as f32;
            nv[i] = n_new as f32;
            pgv[i] = gi as f32;
        }

        let shape = p.shape();
        p.copy_from(&Tensor::from_vec(pv, shape)?)?;
        ps.exp_avg.copy_from(&Tensor::from_vec(mv, shape)?)?;
        ps.exp_avg_diff.copy_from(&Tensor::from_vec(vv, shape)?)?;
        ps.exp_avg_sq.copy_from(&Tensor::from_vec(nv, shape)?)?;
        ps.pre_grad.copy_from(&Tensor::from_vec(pgv, shape)?)?;
        return Ok(());
    }

    // Device-generic composite path (supports GPU tensors and non-contiguous / aliased layouts).
    // Snapshot `g32` whenever it shares storage with any destination mutated before `pre_grad`,
    // pre-scale `u` by `sqrt(1 - beta3)` before squaring, and divide by `sqrt(bc3)` after `sqrt` so
    // representable `exp_avg_sq` values never overflow `f32` prematurely.
    let g32 = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
    let g_aliases_dest = [p, &ps.exp_avg, &ps.exp_avg_diff, &ps.exp_avg_sq]
        .iter()
        .any(|d| storage_key(d) == storage_key(&g32));
    let g_save = if g_aliases_dest { g32.mul_scalar(1.0)? } else { g32.clone() };

    let diff = if is_first {
        Tensor::zeros_with(p.shape(), DType::F32, p.device())?
    } else {
        g32.sub(&ps.pre_grad)?
    };
    let m_new = ps.exp_avg.mul_scalar(c.b1 as f32)?.add(&g32.mul_scalar(one_m_b1 as f32)?)?;
    let v_new = ps.exp_avg_diff.mul_scalar(c.b2 as f32)?.add(&diff.mul_scalar(one_m_b2 as f32)?)?;
    let u = g32.add(&diff.mul_scalar(c.b2 as f32)?)?;
    let u_scaled_sq = u.mul_scalar(one_m_b3.sqrt() as f32)?.square()?;
    let n_new = ps.exp_avg_sq.mul_scalar(c.b3 as f32)?.add(&u_scaled_sq)?;
    let num = m_new.div_scalar(bc1 as f32)?.add(&v_new.mul_scalar((c.b2 / bc2) as f32)?)?;
    let denom = n_new.sqrt()?.div_scalar(bc3.sqrt() as f32)?.add_scalar(c.eps as f32)?;
    let step_dir = num.div(&denom)?;
    let p_new = if c.decoupled_wd {
        p.mul_scalar(wd_scale as f32)?.sub(&step_dir.mul_scalar(c.lr as f32)?)?
    } else {
        p.sub(&step_dir.mul_scalar(c.lr as f32)?)?.mul_scalar(wd_scale as f32)?
    };

    p.copy_from(&p_new)?;
    ps.exp_avg.copy_from(&m_new)?;
    ps.exp_avg_diff.copy_from(&v_new)?;
    ps.exp_avg_sq.copy_from(&n_new)?;
    ps.pre_grad.copy_from(&g_save)?;
    Ok(())
}
