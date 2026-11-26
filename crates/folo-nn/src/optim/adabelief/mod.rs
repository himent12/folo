//! AdaBelief optimizer (Zhuang et al., NeurIPS 2020 / arXiv:2010.07468): adapts the step size
//! according to the "belief" in observed gradients by tracking the exponential moving average of
//! the squared prediction residual `(g_t - m_t)^2`:
//!
//! ```text
//! g_eff = g + wd * theta                                         (if !decoupled_wd)
//! m_t   = beta1 * m_{t-1} + (1 - beta1) * g_eff
//! s_t   = beta2 * s_{t-1} + (1 - beta2) * (g_eff - m_t)^2 + eps
//! m_hat = m_t / (1 - prod_{j=1}^t beta1_j)
//! s_hat = s_t / (1 - beta2^t)
//! denom = sqrt(max_t s_hat) + eps                                (if amsgrad, else sqrt(s_hat) + eps)
//! theta = theta * (1 - lr * wd) - lr * m_hat / denom             (if decoupled_wd)
//! ```
//!
//! Parameters without a gradient or with non-finite gradient/parameter values are skipped cleanly
//! without mutating parameters, moments, or active step counters.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradRef, GradSource, ParamGroup, ParamPlan, check_param_sets, cpu_fast, for_each_param_planned,
    storage_key, take_state, validate_groups,
};
use crate::{Error, Result};

/// Default AdaBelief learning rate (`1e-3`).
pub const ADABELIEF_DEFAULT_LR: f64 = 1e-3;
/// Default AdaBelief first-moment retention `beta1` (`0.9`).
pub const ADABELIEF_DEFAULT_BETA1: f64 = 0.9;
/// Default AdaBelief second-moment retention `beta2` (`0.999`).
pub const ADABELIEF_DEFAULT_BETA2: f64 = 0.999;
/// Default AdaBelief numerical stability epsilon (`1e-8`, positive in `f32`).
pub const ADABELIEF_DEFAULT_EPS: f64 = 1e-8;
/// Default AdaBelief weight decay (`0.0`).
pub const ADABELIEF_DEFAULT_WEIGHT_DECAY: f64 = 0.0;

/// Hyper-parameters for constructing an [`AdaBelief`] optimizer via [`AdaBelief::from_params`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdaBeliefConfig {
    /// Learning rate `>= 0` (default `1e-3`).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2]`, each in `[0, 1)` (default `[0.9, 0.999]`).
    pub betas: [f64; 2],
    /// Positive additive floor `eps > 0` added inside both `s_t` and `sqrt(s_hat) + eps` (default `1e-8`).
    pub eps: f64,
    /// Weight decay `>= 0` (default `0.0`).
    pub weight_decay: f64,
    /// `true` (default): AdamW-style decoupled weight decay; `false`: coupled L2 decay.
    pub decoupled_wd: bool,
    /// Enable AMSGrad monotonic upper bound on bias-corrected belief variance `s_hat` (default `false`).
    pub amsgrad: bool,
}

impl Default for AdaBeliefConfig {
    fn default() -> Self {
        Self {
            lr: ADABELIEF_DEFAULT_LR,
            betas: [ADABELIEF_DEFAULT_BETA1, ADABELIEF_DEFAULT_BETA2],
            eps: ADABELIEF_DEFAULT_EPS,
            weight_decay: ADABELIEF_DEFAULT_WEIGHT_DECAY,
            decoupled_wd: true,
            amsgrad: false,
        }
    }
}

impl AdaBeliefConfig {
    /// Validate all AdaBelief hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_adabelief_hyperparams(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
        )
    }
}

/// Serializable contract for one [`AdaBelief`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaBeliefGroupContract {
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

/// Serializable contract for [`AdaBelief`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaBeliefContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<AdaBeliefGroupContract>,
    /// Whether AMSGrad monotonic variance tracking is enabled.
    pub amsgrad: bool,
    /// Total completed [`AdaBelief::step`] calls.
    pub step: u64,
}

struct ParamState {
    exp_avg: Tensor,
    exp_avg_var: Tensor,
    max_exp_avg_var: Option<Tensor>,
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
    amsgrad: bool,
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
    amsgrad: bool,
}

/// AdaBelief optimizer (Zhuang et al., NeurIPS 2020).
///
/// Uses [`ParamGroup::momentum`] / [`ParamGroup::base_momentum`] for `beta1`, [`ParamGroup::beta2`]
/// for `beta2`, [`ParamGroup::eps`] for `eps`, [`ParamGroup::weight_decay`] for `weight_decay`, and
/// [`ParamGroup::decoupled_wd`] for decoupled (`true`) vs coupled L2 (`false`) weight decay.
pub struct AdaBelief {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    amsgrad: bool,
    pending_resume: Option<PendingResume>,
    plan: ParamPlan,
    coefs: Vec<GroupCoef>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl AdaBelief {
    /// Create an [`AdaBelief`] optimizer over `groups` with optional `amsgrad`.
    pub fn new(groups: Vec<ParamGroup>, amsgrad: bool) -> Result<Self> {
        validate_groups(&groups)?;
        for g in &groups {
            validate_adabelief_hyperparams(
                &g.name,
                g.base_lr,
                g.base_momentum,
                g.beta2,
                g.eps,
                g.weight_decay,
            )?;
            validate_adabelief_hyperparams(&g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adabelief: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = zero_state(&groups, amsgrad)?;
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(groups.len());
        let mult = if amsgrad { 3 } else { 2 };
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * mult).sum());
        Ok(AdaBelief { groups, state, amsgrad, pending_resume: None, plan, coefs, state_keys, step: 0 })
    }

    /// Convenience constructor for a single `"params"` group configured by [`AdaBeliefConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: AdaBeliefConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(cfg.decoupled_wd);
        Self::new(vec![group], cfg.amsgrad)
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (for LR / momentum schedules).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Whether AMSGrad monotonic variance tracking is enabled.
    pub fn amsgrad(&self) -> bool {
        self.amsgrad
    }

    /// Total completed [`AdaBelief::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count (incremented only on finite updates).
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// `(exp_avg, exp_avg_var)` moment buffers for `(group, param)`.
    pub fn moments(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        self.state.get(group)?.get(param).map(|s| (&s.exp_avg, &s.exp_avg_var))
    }

    /// Perform one AdaBelief optimization step over every parameter with a gradient in `grads`.
    ///
    /// Parameters with non-finite gradients or parameters are skipped cleanly without mutating
    /// weights, moments, or active step counters.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "adabelief: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_adabelief_hyperparams(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                grp.eps,
                grp.weight_decay,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "adabelief: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "adabelief: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "adabelief: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                let mut check_buf = |sname: &str, buf: &Tensor| -> Result<()> {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "adabelief: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
                            p.dims(),
                            p.device(),
                            buf.dims(),
                            buf.device(),
                            buf.dtype()
                        )));
                    }
                    self.state_keys.insert(storage_key(buf));
                    Ok(())
                };
                check_buf("exp_avg", &ps.exp_avg)?;
                check_buf("exp_avg_var", &ps.exp_avg_var)?;
                if let Some(m_buf) = &ps.max_exp_avg_var {
                    check_buf("max_exp_avg_var", m_buf)?;
                }
                if let Some(g) = grads.grad(p) {
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "adabelief: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "adabelief: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "adabelief: gradient of `{name}` has non-float dtype {}",
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
                        "adabelief: parameter `{name}` aliases an optimizer state buffer"
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
                amsgrad: self.amsgrad,
            });
        }

        let updated: Vec<Mutex<bool>> =
            self.groups.iter().flat_map(|g| g.params.iter().map(|_| Mutex::new(false))).collect();
        let mut group_offsets = Vec::with_capacity(self.groups.len());
        let mut off = 0usize;
        for g in &self.groups {
            group_offsets.push(off);
            off += g.params.len();
        }

        let _ng = no_grad();
        let (groups, state, plan, coefs) = (&self.groups, &self.state, &self.plan, &self.coefs);
        if aliases_state {
            for (gi, grp) in groups.iter().enumerate() {
                for (pi, (_, p)) in grp.params.iter().enumerate() {
                    let Some(g) = grads.grad(p) else { continue };
                    if adabelief_step_one(p, g, &state[gi][pi], coefs[gi])? {
                        *updated[group_offsets[gi] + pi].lock().unwrap_or_else(|e| e.into_inner()) = true;
                    }
                }
            }
        } else {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                if adabelief_step_one(p, g, &state[gi][pi], coefs[gi])? {
                    *updated[group_offsets[gi] + pi].lock().unwrap_or_else(|e| e.into_inner()) = true;
                }
                Ok(())
            })?;
        }

        for (gi, (grp, st)) in self.groups.iter().zip(&mut self.state).enumerate() {
            let b1 = grp.momentum;
            let base = group_offsets[gi];
            for (pi, ps) in st.iter_mut().enumerate() {
                if *updated[base + pi].lock().unwrap_or_else(|e| e.into_inner()) {
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
                ps.exp_avg_var.to_device_(device)?;
                if let Some(m) = &ps.max_exp_avg_var {
                    m.to_device_(device)?;
                }
            }
        }
        Ok(())
    }

    /// Export the serializable [`AdaBeliefContract`].
    pub fn contract(&self) -> AdaBeliefContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.state)
            .map(|(g, st)| AdaBeliefGroupContract {
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
        AdaBeliefContract { groups, amsgrad: self.amsgrad, step: self.step }
    }

    /// Validate `c` and stage its hyperparameters and per-parameter counters for atomic commit in [`AdaBelief::load_state`].
    pub fn apply_contract(&mut self, c: &AdaBeliefContract) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Load(format!(
                "adabelief: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
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
                    "adabelief contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            if !gc.beta1_prods.is_empty() && gc.beta1_prods.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "adabelief contract group `{}`: beta1_prods length {} != {} params",
                    gc.name,
                    gc.beta1_prods.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_adabelief_hyperparams(
                &gc.name,
                gc.lr,
                gc.betas[0],
                gc.betas[1],
                gc.eps,
                gc.weight_decay,
            )
            .map_err(|e| Error::Load(e.to_string()))?;
            validate_adabelief_hyperparams(&gc.name, cur_lr, cur_b1, gc.betas[1], gc.eps, gc.weight_decay)
                .map_err(|e| Error::Load(e.to_string()))?;

            let mut by_name: HashMap<&str, (u64, f64)> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "adabelief contract group `{}` param `{pname}`: param_step {k} > step {}",
                        gc.name, c.step
                    )));
                }
                let prod = if gc.beta1_prods.is_empty() {
                    if k == 0 {
                        1.0
                    } else if gc.betas[0] == 0.0 {
                        0.0
                    } else {
                        ((k as f64) * (-(1.0 - gc.betas[0])).ln_1p()).exp()
                    }
                } else {
                    let p = gc.beta1_prods[idx];
                    if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                        return Err(Error::Load(format!(
                            "adabelief contract group `{}` param `{pname}`: beta1_prod must be in [0, 1], got {p}",
                            gc.name
                        )));
                    }
                    if k == 0 && p != 1.0 {
                        return Err(Error::Load(format!(
                            "adabelief contract group `{}` param `{pname}`: param_step == 0 requires beta1_prod == 1.0, got {p}",
                            gc.name
                        )));
                    }
                    if k > 0 && p >= 1.0 {
                        return Err(Error::Load(format!(
                            "adabelief contract group `{}` param `{pname}`: param_step > 0 requires beta1_prod < 1.0, got {p}",
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
            amsgrad: c.amsgrad,
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

    /// Export all optimizer state tensors (`<param>.exp_avg`, `<param>.exp_avg_var`, plus
    /// `<param>.max_exp_avg_var` when `amsgrad` is enabled).
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mult = if self.amsgrad { 3 } else { 2 };
        let mut out = Vec::with_capacity(self.groups.iter().map(|g| g.params.len() * mult).sum());
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), ps.exp_avg.clone()));
                out.push((format!("{name}.exp_avg_var"), ps.exp_avg_var.clone()));
                if let Some(m) = &ps.max_exp_avg_var {
                    out.push((format!("{name}.max_exp_avg_var"), m.clone()));
                }
            }
        }
        Ok(out)
    }

    /// Restore step counter and state tensors from a checkpoint map.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "AdaBelief state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "AdaBelief state: step {step} does not match staged contract step {}",
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
                    "AdaBelief state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }
        let target_amsgrad = self.pending_resume.as_ref().map_or(self.amsgrad, |pr| pr.amsgrad);
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                let req = p_step > 0;
                let keys: &[&str] = if target_amsgrad {
                    &["exp_avg", "exp_avg_var", "max_exp_avg_var"]
                } else {
                    &["exp_avg", "exp_avg_var"]
                };
                for (k, &what) in keys.iter().enumerate() {
                    if let Some(t) = take_state(state, name, p, what, req, &mut problems) {
                        let vals = t.to_vec::<f32>()?;
                        let bad = if k == 0 {
                            vals.iter().any(|v| !v.is_finite())
                        } else {
                            vals.iter().any(|&v| !v.is_finite() || v < 0.0)
                        };
                        if bad {
                            problems.push(format!("`optim.{name}.{what}` contains invalid values"));
                            continue;
                        }
                        loaded.push((gi, pi, k, t));
                    }
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("AdaBelief state: {}", problems.join("; "))));
        }
        let mut fresh = zero_state(&self.groups, target_amsgrad)?;
        let _ng = no_grad();
        for (gi, pi, k, t) in loaded {
            let ps = &fresh[gi][pi];
            let dst = match k {
                0 => &ps.exp_avg,
                1 => &ps.exp_avg_var,
                _ => ps.max_exp_avg_var.as_ref().expect("amsgrad buffer exists"),
            };
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        if let Some(pending) = self.pending_resume.take() {
            self.amsgrad = pending.amsgrad;
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

fn validate_adabelief_hyperparams(group: &str, lr: f64, b1: f64, b2: f64, eps: f64, wd: f64) -> Result<()> {
    if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "adabelief group `{group}`: lr must be finite in f32 and >= 0, got {lr}"
        )));
    }
    if !(b1.is_finite() && (0.0..1.0).contains(&b1) && (b1 as f32) < 1.0 && ((1.0 - b1) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "adabelief group `{group}`: beta1 must be finite in [0, 1) with f32(beta1) < 1, got {b1}"
        )));
    }
    if !(b2.is_finite() && (0.0..1.0).contains(&b2) && (b2 as f32) < 1.0 && ((1.0 - b2) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "adabelief group `{group}`: beta2 must be finite in [0, 1) with f32(beta2) < 1, got {b2}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "adabelief group `{group}`: eps must be finite and > 0 in f32, got {eps}"
        )));
    }
    if !(wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite() && ((1.0 - lr * wd) as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "adabelief group `{group}`: weight_decay must be finite in f32 and >= 0 with finite (1 - lr * wd), got {wd}"
        )));
    }
    Ok(())
}

fn zero_state(groups: &[ParamGroup], amsgrad: bool) -> Result<Vec<Vec<ParamState>>> {
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(_, p)| {
                    let dev = p.device();
                    let shape = p.shape();
                    let max_var =
                        if amsgrad { Some(Tensor::zeros_with(shape, DType::F32, dev)?) } else { None };
                    Ok(ParamState {
                        exp_avg: Tensor::zeros_with(shape, DType::F32, dev)?,
                        exp_avg_var: Tensor::zeros_with(shape, DType::F32, dev)?,
                        max_exp_avg_var: max_var,
                        param_step: 0,
                        beta1_prod: 1.0,
                    })
                })
                .collect()
        })
        .collect()
}

/// Execute one AdaBelief step on a single parameter. Returns `Ok(true)` if updated, or `Ok(false)`
/// if skipped due to non-finite inputs/update (leaving `p`, `m`, `s` unchanged).
fn adabelief_step_one(p: &Tensor, g: &Tensor, ps: &ParamState, c: GroupCoef) -> Result<bool> {
    let t = ps.param_step.saturating_add(1);
    let bc1 = 1.0 - ps.beta1_prod * c.b1;
    let bc2 = if c.b2 == 0.0 { 1.0 } else { -((t as f64) * (-(1.0 - c.b2)).ln_1p()).exp_m1() };
    let decay = if c.decoupled_wd && c.wd != 0.0 { Some(1.0 - c.lr * c.wd) } else { None };
    let coupled_wd = !c.decoupled_wd && c.wd != 0.0;

    let (m, s) = (&ps.exp_avg, &ps.exp_avg_var);
    let max_s = ps.max_exp_avg_var.as_ref();
    let can_fast = cpu_fast(&[p, m, s])
        && max_s.is_none_or(|ms| cpu_fast(&[ms]) && !ms.same_storage(p) && !ms.same_storage(g))
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !p.same_storage(m)
        && !p.same_storage(s)
        && !m.same_storage(g)
        && !m.same_storage(s)
        && !s.same_storage(g);

    let g_ref = if can_fast { None } else { Some(GradRef::as_f32(g)?) };
    let pv = p.to_vec::<f32>()?;
    let gv = match &g_ref {
        Some(gr) => gr.as_tensor().to_vec::<f32>()?,
        None => g.to_vec::<f32>()?,
    };
    let mv = m.to_vec::<f32>()?;
    let sv = s.to_vec::<f32>()?;
    let max_sv = match max_s {
        Some(ms) => Some(ms.to_vec::<f32>()?),
        None => None,
    };

    if !gv.iter().all(|x| x.is_finite()) || !pv.iter().all(|x| x.is_finite()) {
        return Ok(false);
    }

    let n = pv.len();
    let mut m_next = vec![0.0f32; n];
    let mut s_next = vec![0.0f32; n];
    let mut max_s_next = if c.amsgrad { vec![0.0f32; n] } else { Vec::new() };
    let mut p_next = vec![0.0f32; n];

    let b1 = c.b1;
    let one_minus_b1 = 1.0 - b1;
    let b2 = c.b2;
    let one_minus_b2 = 1.0 - b2;
    let step_size = c.lr / bc1;

    for i in 0..n {
        let pi = pv[i] as f64;
        let gi = gv[i] as f64;
        let g_eff = if coupled_wd { gi + c.wd * pi } else { gi };
        let mi = b1 * (mv[i] as f64) + one_minus_b1 * g_eff;
        let diff = g_eff - mi;
        let si = b2 * (sv[i] as f64) + one_minus_b2 * (diff * diff) + c.eps;
        if !(mi.is_finite() && si.is_finite() && (mi as f32).is_finite() && (si as f32).is_finite()) {
            return Ok(false);
        }
        let s_hat = si / bc2;
        let denom_var = if let Some(ref prev_max) = max_sv {
            let mx = (prev_max[i] as f64).max(s_hat);
            if !(mx.is_finite() && (mx as f32).is_finite()) {
                return Ok(false);
            }
            max_s_next[i] = mx as f32;
            mx
        } else {
            s_hat
        };
        let p_base = if let Some(d) = decay { pi * d } else { pi };
        let pn = p_base - step_size * (mi / (denom_var.sqrt() + c.eps));
        let pn_f32 = pn as f32;
        if !pn_f32.is_finite() {
            return Ok(false);
        }
        m_next[i] = mi as f32;
        s_next[i] = si as f32;
        p_next[i] = pn_f32;
    }

    if can_fast {
        m.update_f32(|ms| ms.copy_from_slice(&m_next))?;
        s.update_f32(|ss| ss.copy_from_slice(&s_next))?;
        if let Some(ms) = max_s {
            ms.update_f32(|dst| dst.copy_from_slice(&max_s_next))?;
        }
        p.update_f32(|ps_slice| ps_slice.copy_from_slice(&p_next))?;
    } else {
        let dev = p.device();
        let shape = p.shape();
        m.copy_from(&Tensor::from_vec(m_next, shape)?.to_device(dev)?)?;
        s.copy_from(&Tensor::from_vec(s_next, shape)?.to_device(dev)?)?;
        if let Some(ms) = max_s {
            ms.copy_from(&Tensor::from_vec(max_s_next, shape)?.to_device(dev)?)?;
        }
        p.copy_from(&Tensor::from_vec(p_next, shape)?.to_device(dev)?)?;
    }
    Ok(true)
}
