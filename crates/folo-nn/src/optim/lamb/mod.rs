//! LAMB (Layer-wise Adaptive Moments optimizer for Batch training; You et al., ICLR 2020 /
//! arXiv:1904.00962).
//!
//! For each parameter tensor `theta` receiving a finite gradient `g` at its active step `t >= 1`:
//! ```text
//! m_t   = beta1 * m_{t-1} + (1 - beta1) * g
//! v_t   = beta2 * v_{t-1} + (1 - beta2) * g^2
//! m_hat = m_t / (1 - prod_{j=1}^t beta1_j)         (if bias_correction)
//! v_hat = v_t / (1 - beta2^t)                      (if bias_correction)
//! u_t   = m_hat / (sqrt(v_hat) + eps) + wd * theta
//! r_t   = clamp(||theta||_2 / ||u_t||_2, max = trust_clip)   if ||theta||_2 > 0 && ||u_t||_2 > 0
//!       = 1.0                                                otherwise
//! theta = theta - lr * r_t * u_t
//! ```
//!
//! Both parameter and update norms are computed via overflow-free rescaled `f64` 2-norms.
//! Parameters without a gradient or with non-finite gradient/parameter values are skipped safely
//! without corrupting parameters, moments, or active step counters.

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

/// Default LAMB learning rate (`1e-3`).
pub const LAMB_DEFAULT_LR: f64 = 1e-3;
/// Default LAMB first-moment retention `beta1` (`0.9`).
pub const LAMB_DEFAULT_BETA1: f64 = 0.9;
/// Default LAMB second-moment retention `beta2` (`0.999`).
pub const LAMB_DEFAULT_BETA2: f64 = 0.999;
/// Default LAMB numerical stability epsilon (`1e-6`, per You et al. 2020).
pub const LAMB_DEFAULT_EPS: f64 = 1e-6;
/// Default LAMB decoupled weight decay (`0.01`).
pub const LAMB_DEFAULT_WEIGHT_DECAY: f64 = 0.01;
/// Default LAMB trust-ratio upper clamp (`10.0`).
pub const LAMB_DEFAULT_TRUST_CLIP: f64 = 10.0;

/// Hyper-parameters for constructing a [`Lamb`] optimizer via [`Lamb::from_params`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LambConfig {
    /// Learning rate `>= 0` (default `1e-3`).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2]`, each in `[0, 1)` (default `[0.9, 0.999]`).
    pub betas: [f64; 2],
    /// Denominator floor `eps > 0` (must remain positive in `f32`; default `1e-6`).
    pub eps: f64,
    /// Decoupled weight decay `>= 0` added to the trust-normalized direction `u_t` (default `0.01`).
    pub weight_decay: f64,
    /// Optional upper bound `> 0` on the layer-wise trust ratio `r_t = ||w|| / ||u||` (default `Some(10.0)`).
    pub trust_clip: Option<f64>,
    /// Enable Adam bias correction on `m_t` and `v_t` (default `true`).
    pub bias_correction: bool,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: LAMB_DEFAULT_LR,
            betas: [LAMB_DEFAULT_BETA1, LAMB_DEFAULT_BETA2],
            eps: LAMB_DEFAULT_EPS,
            weight_decay: LAMB_DEFAULT_WEIGHT_DECAY,
            trust_clip: Some(LAMB_DEFAULT_TRUST_CLIP),
            bias_correction: true,
        }
    }
}

impl LambConfig {
    /// Validate all LAMB hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_lamb_hyperparams(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.trust_clip,
        )
    }
}

/// Serializable contract for one [`Lamb`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambGroupContract {
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
    /// Decoupled weight decay `>= 0`.
    pub weight_decay: f64,
    /// Active step counts per parameter (aligned with `params`).
    pub param_steps: Vec<u64>,
    /// Cumulative `prod_{j=1}^{param_step} beta1_j` per parameter (aligned with `params`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beta1_prods: Vec<f64>,
}

/// Serializable contract for [`Lamb`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LambContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<LambGroupContract>,
    /// Optional trust-ratio upper clamp `> 0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_clip: Option<f64>,
    /// Whether bias correction is active.
    pub bias_correction: bool,
    /// Total completed [`Lamb::step`] calls.
    pub step: u64,
}

struct ParamState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    param_step: u64,
    beta1_prod: f64,
    last_trust_ratio: f64,
}

struct StagedGroup {
    base_lr: f64,
    lr: f64,
    base_momentum: f64,
    momentum: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

struct PendingResume {
    step: u64,
    trust_clip: Option<f64>,
    bias_correction: bool,
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
    trust_clip: Option<f64>,
    bias_correction: bool,
}

/// LAMB (Layer-wise Adaptive Moments) optimizer (You et al., ICLR 2020).
///
/// Uses [`ParamGroup::momentum`] / [`ParamGroup::base_momentum`] for `beta1`, [`ParamGroup::beta2`]
/// for `beta2`, [`ParamGroup::eps`] for `eps`, and [`ParamGroup::weight_decay`] for decoupled weight
/// decay inside the trust-normalized step direction `u_t`.
pub struct Lamb {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    trust_clip: Option<f64>,
    bias_correction: bool,
    pending_resume: Option<PendingResume>,
    plan: ParamPlan,
    coefs: Vec<GroupCoef>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl Lamb {
    /// Create a [`Lamb`] optimizer over `groups` with optional `trust_clip` and `bias_correction`.
    pub fn new(groups: Vec<ParamGroup>, trust_clip: Option<f64>, bias_correction: bool) -> Result<Self> {
        validate_groups(&groups)?;
        for g in &groups {
            validate_lamb_hyperparams(
                &g.name,
                g.base_lr,
                g.base_momentum,
                g.beta2,
                g.eps,
                g.weight_decay,
                trust_clip,
            )?;
            validate_lamb_hyperparams(&g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay, trust_clip)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "lamb: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = zero_state(&groups)?;
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(groups.len());
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 2).sum());
        Ok(Lamb {
            groups,
            state,
            trust_clip,
            bias_correction,
            pending_resume: None,
            plan,
            coefs,
            state_keys,
            step: 0,
        })
    }

    /// Convenience constructor for a single `"params"` group configured by [`LambConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: LambConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay);
        Self::new(vec![group], cfg.trust_clip, cfg.bias_correction)
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (for LR / momentum schedules).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Configured optional upper clamp on the layer-wise trust ratio.
    pub fn trust_clip(&self) -> Option<f64> {
        self.trust_clip
    }

    /// Whether Adam bias correction is enabled.
    pub fn bias_correction(&self) -> bool {
        self.bias_correction
    }

    /// Total completed [`Lamb::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Per-parameter active step count (incremented only when the parameter receives a finite update).
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// Most recent layer-wise trust ratio `r_t` applied to `(group, param)` (`1.0` before its first step).
    pub fn last_trust_ratio(&self, group: usize, param: usize) -> Option<f64> {
        self.state.get(group)?.get(param).map(|s| s.last_trust_ratio)
    }

    /// `(exp_avg, exp_avg_sq)` moment buffers for `(group, param)`.
    pub fn moments(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        self.state.get(group)?.get(param).map(|s| (&s.exp_avg, &s.exp_avg_sq))
    }

    /// Perform one LAMB optimization step over every parameter with a gradient in `grads`.
    ///
    /// Parameters with non-finite gradients or parameters are skipped cleanly (leaving parameter and
    /// moment tensors intact) so numerical corruption never propagates.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "lamb: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_lamb_hyperparams(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                grp.eps,
                grp.weight_decay,
                self.trust_clip,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "lamb: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "lamb: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "lamb: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                for (sname, buf) in [("exp_avg", &ps.exp_avg), ("exp_avg_sq", &ps.exp_avg_sq)] {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "lamb: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
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
                            "lamb: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "lamb: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "lamb: gradient of `{name}` has non-float dtype {}",
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
                        "lamb: parameter `{name}` aliases an optimizer state buffer"
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
                trust_clip: self.trust_clip,
                bias_correction: self.bias_correction,
            });
        }

        let updated: Vec<Mutex<Option<f64>>> =
            self.groups.iter().flat_map(|g| g.params.iter().map(|_| Mutex::new(None))).collect();
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
                    if let Some(ratio) = lamb_step_one(p, g, &state[gi][pi], coefs[gi])? {
                        *updated[group_offsets[gi] + pi].lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(ratio);
                    }
                }
            }
        } else {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                if let Some(ratio) = lamb_step_one(p, g, &state[gi][pi], coefs[gi])? {
                    *updated[group_offsets[gi] + pi].lock().unwrap_or_else(|e| e.into_inner()) = Some(ratio);
                }
                Ok(())
            })?;
        }

        for (gi, (grp, st)) in self.groups.iter().zip(&mut self.state).enumerate() {
            let b1 = grp.momentum;
            let base = group_offsets[gi];
            for (pi, ps) in st.iter_mut().enumerate() {
                if let Some(ratio) = *updated[base + pi].lock().unwrap_or_else(|e| e.into_inner()) {
                    ps.param_step = ps.param_step.saturating_add(1);
                    ps.beta1_prod *= b1;
                    ps.last_trust_ratio = ratio;
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

    /// Export the serializable [`LambContract`].
    pub fn contract(&self) -> LambContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.state)
            .map(|(g, st)| LambGroupContract {
                name: g.name.clone(),
                params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                lr: g.base_lr,
                current_lr: (g.lr != g.base_lr).then_some(g.lr),
                betas: [g.base_momentum, g.beta2],
                current_beta1: (g.momentum != g.base_momentum).then_some(g.momentum),
                eps: g.eps,
                weight_decay: g.weight_decay,
                param_steps: st.iter().map(|ps| ps.param_step).collect(),
                beta1_prods: st.iter().map(|ps| ps.beta1_prod).collect(),
            })
            .collect();
        LambContract {
            groups,
            trust_clip: self.trust_clip,
            bias_correction: self.bias_correction,
            step: self.step,
        }
    }

    /// Validate `c` against this optimizer's parameter sets and stage its hyperparameters and
    /// per-parameter step counts / `beta1` products for atomic commit in [`Lamb::load_state`].
    pub fn apply_contract(&mut self, c: &LambContract) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Load(format!(
                "lamb: group count {} does not match state {}",
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
                decoupled_wd: true,
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
                    "lamb contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            if !gc.beta1_prods.is_empty() && gc.beta1_prods.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "lamb contract group `{}`: beta1_prods length {} != {} params",
                    gc.name,
                    gc.beta1_prods.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_lamb_hyperparams(
                &gc.name,
                gc.lr,
                gc.betas[0],
                gc.betas[1],
                gc.eps,
                gc.weight_decay,
                c.trust_clip,
            )
            .map_err(|e| Error::Load(e.to_string()))?;
            validate_lamb_hyperparams(
                &gc.name,
                cur_lr,
                cur_b1,
                gc.betas[1],
                gc.eps,
                gc.weight_decay,
                c.trust_clip,
            )
            .map_err(|e| Error::Load(e.to_string()))?;

            let mut by_name: HashMap<&str, (u64, f64)> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "lamb contract group `{}` param `{pname}`: param_step {k} > step {}",
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
                            "lamb contract group `{}` param `{pname}`: beta1_prod must be in [0, 1], got {p}",
                            gc.name
                        )));
                    }
                    if k == 0 && p != 1.0 {
                        return Err(Error::Load(format!(
                            "lamb contract group `{}` param `{pname}`: param_step == 0 requires beta1_prod == 1.0, got {p}",
                            gc.name
                        )));
                    }
                    if k > 0 && p >= 1.0 {
                        return Err(Error::Load(format!(
                            "lamb contract group `{}` param `{pname}`: param_step > 0 requires beta1_prod < 1.0, got {p}",
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
            })
            .collect();
        self.pending_resume = Some(PendingResume {
            step: c.step,
            trust_clip: c.trust_clip,
            bias_correction: c.bias_correction,
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
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "LAMB state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "LAMB state: step {step} does not match staged contract step {}",
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
                    "LAMB state: parameter groups changed after `apply_contract`".into(),
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
            return Err(Error::Load(format!("LAMB state: {}", problems.join("; "))));
        }
        let mut fresh = zero_state(&self.groups)?;
        let _ng = no_grad();
        for (gi, pi, k, t) in loaded {
            let ps = &fresh[gi][pi];
            let dst = if k == 0 { &ps.exp_avg } else { &ps.exp_avg_sq };
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        if let Some(pending) = self.pending_resume.take() {
            self.trust_clip = pending.trust_clip;
            self.bias_correction = pending.bias_correction;
            for (g, sg) in self.groups.iter_mut().zip(pending.groups) {
                g.base_lr = sg.base_lr;
                g.lr = sg.lr;
                g.base_momentum = sg.base_momentum;
                g.momentum = sg.momentum;
                g.beta2 = sg.beta2;
                g.eps = sg.eps;
                g.weight_decay = sg.weight_decay;
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

fn validate_lamb_hyperparams(
    group: &str,
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
    trust_clip: Option<f64>,
) -> Result<()> {
    if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: lr must be finite in f32 and >= 0, got {lr}"
        )));
    }
    if !(b1.is_finite() && (0.0..1.0).contains(&b1) && (b1 as f32) < 1.0 && ((1.0 - b1) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: beta1 must be finite in [0, 1) with f32(beta1) < 1, got {b1}"
        )));
    }
    if !(b2.is_finite() && (0.0..1.0).contains(&b2) && (b2 as f32) < 1.0 && ((1.0 - b2) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: beta2 must be finite in [0, 1) with f32(beta2) < 1, got {b2}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: eps must be finite and > 0 in f32, got {eps}"
        )));
    }
    if !(wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: weight_decay must be finite in f32 and >= 0, got {wd}"
        )));
    }
    if let Some(clip) = trust_clip
        && !(clip.is_finite() && clip > 0.0)
    {
        return Err(Error::Invalid(format!(
            "lamb group `{group}`: trust_clip must be finite and > 0, got {clip}"
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
                        last_trust_ratio: 1.0,
                    })
                })
                .collect()
        })
        .collect()
}

/// Overflow-free rescaled L2 norm of an `f64` iterator. Returns `f64::NAN` if any element is non-finite.
fn rescaled_l2_norm(vals: &[f64]) -> f64 {
    let mut max_abs = 0.0f64;
    for &v in vals {
        if !v.is_finite() {
            return f64::NAN;
        }
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    let inv = 1.0 / max_abs;
    let mut sum_sq = 0.0f64;
    for &v in vals {
        let s = v * inv;
        sum_sq += s * s;
    }
    max_abs * sum_sq.sqrt()
}

/// Execute one LAMB step on a single parameter. Returns `Ok(Some(trust_ratio))` if updated, or
/// `Ok(None)` if skipped due to non-finite inputs/update (leaving `p`, `m`, `v` unchanged).
fn lamb_step_one(p: &Tensor, g: &Tensor, ps: &ParamState, c: GroupCoef) -> Result<Option<f64>> {
    let t = ps.param_step.saturating_add(1);
    let (bc1, bc2_sqrt) = if c.bias_correction {
        let bc1 = 1.0 - ps.beta1_prod * c.b1;
        let bc2 = if c.b2 == 0.0 { 1.0 } else { -((t as f64) * (-(1.0 - c.b2)).ln_1p()).exp_m1() };
        (bc1, bc2.sqrt())
    } else {
        (1.0, 1.0)
    };

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

    let g_ref = if can_fast { None } else { Some(GradRef::as_f32(g)?) };
    let pv = p.to_vec::<f32>()?;
    let gv = match &g_ref {
        Some(gr) => gr.as_tensor().to_vec::<f32>()?,
        None => g.to_vec::<f32>()?,
    };
    let mv = m.to_vec::<f32>()?;
    let vv = v.to_vec::<f32>()?;

    if !gv.iter().all(|x| x.is_finite()) || !pv.iter().all(|x| x.is_finite()) {
        return Ok(None);
    }

    let n = pv.len();
    let mut m_next = vec![0.0f32; n];
    let mut v_next = vec![0.0f32; n];
    let mut u_dir = vec![0.0f64; n];
    let mut w_vals = vec![0.0f64; n];

    let b1 = c.b1;
    let one_minus_b1 = 1.0 - b1;
    let b2 = c.b2;
    let one_minus_b2 = 1.0 - b2;

    for i in 0..n {
        let pi = pv[i] as f64;
        let gi = gv[i] as f64;
        let mi = b1 * (mv[i] as f64) + one_minus_b1 * gi;
        let vi = b2 * (vv[i] as f64) + one_minus_b2 * (gi * gi);
        if !(mi.is_finite() && vi.is_finite() && (mi as f32).is_finite() && (vi as f32).is_finite()) {
            return Ok(None);
        }
        let m_hat = mi / bc1;
        let denom = vi.sqrt() / bc2_sqrt + c.eps;
        let ui = m_hat / denom + c.wd * pi;
        if !ui.is_finite() {
            return Ok(None);
        }
        m_next[i] = mi as f32;
        v_next[i] = vi as f32;
        u_dir[i] = ui;
        w_vals[i] = pi;
    }

    let w_norm = rescaled_l2_norm(&w_vals);
    let u_norm = rescaled_l2_norm(&u_dir);
    if !w_norm.is_finite() || !u_norm.is_finite() {
        return Ok(None);
    }
    let trust_ratio = if w_norm > 0.0 && u_norm > 0.0 {
        let raw = w_norm / u_norm;
        if let Some(clip) = c.trust_clip {
            if raw.is_nan() {
                return Ok(None);
            }
            raw.min(clip)
        } else if raw.is_finite() {
            raw
        } else {
            return Ok(None);
        }
    } else {
        1.0
    };

    let eff_lr = c.lr * trust_ratio;
    let mut p_next = vec![0.0f32; n];
    for i in 0..n {
        let pn = w_vals[i] - eff_lr * u_dir[i];
        let pn_f32 = pn as f32;
        if !pn_f32.is_finite() {
            return Ok(None);
        }
        p_next[i] = pn_f32;
    }

    if can_fast {
        m.update_f32(|ms| ms.copy_from_slice(&m_next))?;
        v.update_f32(|vs| vs.copy_from_slice(&v_next))?;
        p.update_f32(|ps_slice| ps_slice.copy_from_slice(&p_next))?;
    } else {
        let dev = p.device();
        let shape = p.shape();
        let m_t = Tensor::from_vec(m_next, shape)?.to_device(dev)?;
        let v_t = Tensor::from_vec(v_next, shape)?.to_device(dev)?;
        let p_t = Tensor::from_vec(p_next, shape)?.to_device(dev)?;
        m.copy_from(&m_t)?;
        v.copy_from(&v_t)?;
        p.copy_from(&p_t)?;
    }
    Ok(Some(trust_ratio))
}
