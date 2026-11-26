//! Sophia-G / Sophia-H: Second-order Clipped Stochastic Optimization (Liu et al., ICLR 2024 /
//! arXiv:2305.14342).
//!
//! For each parameter receiving a gradient `g_t` at step `t`:
//! ```text
//! m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//! (every k steps) h_t = beta2 * h_{t-1} + (1 - beta2) * h_est
//! p_t = p_{t-1} * (1 - lr * wd) - lr * clip(m_t / max(h_t, eps), -rho, rho)
//! ```
//!
//! Supports both:
//! - **Gauss-Newton diagonal estimator (`Sophia-G`)**: `h_est = g_hat^2` via
//!   [`SophiaG::update_hessian_gauss_newton`] (or automatically every `update_period` steps inside
//!   [`SophiaG::step`] when `auto_gauss_newton = true` and no explicit Hessian update was supplied for the
//!   current step), and
//! - **Seeded Hutchinson diagonal estimator (`Sophia-H`)**: `h_est = v ⊙ (H v)` with deterministic Rademacher
//!   (`±1`) or Gaussian (`N(0, 1)`) probes via [`SophiaG::sample_hutchinson_probes`] +
//!   [`SophiaG::update_hessian_hutchinson`] (or [`SophiaG::update_hessian_hutchinson_with`]).

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradRef, GradSource, ParamGroup, check_param_sets, cpu_fast, storage_key, take_state, validate_groups,
};
use crate::{Error, Result};

/// Default Sophia-G learning rate (`1e-4`).
pub const SOPHIA_DEFAULT_LR: f64 = 1e-4;
/// Default Sophia-G momentum retention `beta1` (`0.965`, Liu et al. 2024).
pub const SOPHIA_DEFAULT_BETA1: f64 = 0.965;
/// Default Sophia-G Hessian EMA retention `beta2` (`0.99`, Liu et al. 2024).
pub const SOPHIA_DEFAULT_BETA2: f64 = 0.99;
/// Default Sophia-G clipping threshold `rho` (`0.04`, Liu et al. 2024).
pub const SOPHIA_DEFAULT_RHO: f64 = 0.04;
/// Default Sophia-G denominator floor `eps` (`1e-12`).
pub const SOPHIA_DEFAULT_EPS: f64 = 1e-12;
/// Default Sophia-G Hessian update cadence `k` (`10` steps, Liu et al. 2024).
pub const SOPHIA_DEFAULT_UPDATE_PERIOD: u64 = 10;
/// Default Sophia-G decoupled weight decay (`0.1`).
pub const SOPHIA_DEFAULT_WEIGHT_DECAY: f64 = 0.1;

/// Probe distribution for Sophia-H Hutchinson diagonal Hessian estimation (`E[v ⊙ (H v)] = diag(H)`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HutchinsonDist {
    /// Rademacher `±1` with equal probability (minimum variance for diagonal estimation).
    #[default]
    Rademacher,
    /// Standard normal `N(0, 1)`.
    Gaussian,
}

/// Hyper-parameters for constructing a [`SophiaG`] optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SophiaConfig {
    /// Learning rate `>= 0` (default `1e-4`).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2]`, each in `[0, 1)` (default `[0.965, 0.99]`).
    pub betas: [f64; 2],
    /// Clipping bound `rho > 0` in `clip(m_t / max(h_t, eps), -rho, rho)` (default `0.04`).
    pub rho: f64,
    /// Denominator floor `eps > 0` (must remain positive in `f32`; default `1e-12`).
    pub eps: f64,
    /// Decoupled weight decay `>= 0` (default `0.1`).
    pub weight_decay: f64,
    /// Hessian estimation cadence `k >= 1`: update diagonal Hessian every `k` steps (`step % k == 0`;
    /// default `10`).
    pub update_period: u64,
    /// When `true` (default), [`SophiaG::step`] automatically performs a diagonal Gauss-Newton EMA update
    /// (`h_est = g_t^2`) whenever `step % update_period == 0` and no explicit `update_hessian_*` call has
    /// already been made for the current step. Set `false` when driving Hessian updates exclusively through
    /// explicit [`SophiaG::update_hessian_gauss_newton`] or [`SophiaG::update_hessian_hutchinson`] calls.
    pub auto_gauss_newton: bool,
    /// Probe distribution for [`SophiaG::sample_hutchinson_probes`] (default `Rademacher`).
    pub hutchinson_dist: HutchinsonDist,
    /// Seed for the deterministic internal Hutchinson probe RNG (default `0`).
    pub seed: u64,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: SOPHIA_DEFAULT_LR,
            betas: [SOPHIA_DEFAULT_BETA1, SOPHIA_DEFAULT_BETA2],
            rho: SOPHIA_DEFAULT_RHO,
            eps: SOPHIA_DEFAULT_EPS,
            weight_decay: SOPHIA_DEFAULT_WEIGHT_DECAY,
            update_period: SOPHIA_DEFAULT_UPDATE_PERIOD,
            auto_gauss_newton: true,
            hutchinson_dist: HutchinsonDist::Rademacher,
            seed: 0,
        }
    }
}

impl SophiaConfig {
    /// Validate all Sophia-G hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_sophia_global(self.rho, self.update_period)?;
        validate_sophia_group(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.rho,
        )
    }
}

/// Serializable contract for one [`SophiaG`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SophiaGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate `>= 0`.
    pub lr: f64,
    /// Current effective learning rate (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base `[beta1, beta2]`.
    pub betas: [f64; 2],
    /// Current effective `beta1` (defaults to `betas[0]` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_beta1: Option<f64>,
    /// Numerical stability floor `eps > 0`.
    pub eps: f64,
    /// Decoupled weight decay `>= 0`.
    pub weight_decay: f64,
    /// Active step counts per parameter (aligned with `params`).
    pub param_steps: Vec<u64>,
}

/// Serializable contract for [`SophiaG`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SophiaContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<SophiaGroupContract>,
    /// Total completed [`SophiaG::step`] calls.
    pub step: u64,
    /// Total completed Hessian EMA updates.
    pub hessian_updates: u64,
    /// Step index on which the most recent Hessian update occurred, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_hessian_step: Option<u64>,
    /// Clipping bound `rho > 0`.
    pub rho: f64,
    /// Hessian update cadence `k >= 1`.
    pub update_period: u64,
    /// Whether `step()` automatically runs diagonal Gauss-Newton when due.
    pub auto_gauss_newton: bool,
    /// Hutchinson probe distribution.
    pub hutchinson_dist: HutchinsonDist,
    /// Internal xoshiro256++ RNG state words for bit-exact Hutchinson probe resume.
    pub rng_state: [u64; 4],
}

struct ParamState {
    exp_avg: Tensor,
    hessian: Tensor,
    param_step: u64,
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
    hessian_updates: u64,
    last_hessian_step: Option<u64>,
    rho: f64,
    update_period: u64,
    auto_gauss_newton: bool,
    hutchinson_dist: HutchinsonDist,
    rng_state: [u64; 4],
    group_names: Vec<String>,
    param_names: Vec<Vec<String>>,
    groups: Vec<StagedGroup>,
    param_steps: Vec<Vec<u64>>,
}

/// Sophia-G / Sophia-H second-order clipped stochastic optimizer (Liu et al., ICLR 2024).
pub struct SophiaG {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    rho: f64,
    update_period: u64,
    auto_gauss_newton: bool,
    hutchinson_dist: HutchinsonDist,
    rng: Rng,
    hessian_updates: u64,
    last_hessian_step: Option<u64>,
    pending_resume: Option<PendingResume>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl SophiaG {
    /// Construct a [`SophiaG`] optimizer over `groups` with global Sophia settings from `cfg`.
    ///
    /// Group-level `lr`, `momentum` (`beta1`), `beta2`, `eps`, and `weight_decay` are read from each
    /// [`ParamGroup`]; global settings (`rho`, `update_period`, `auto_gauss_newton`, `hutchinson_dist`,
    /// `seed`) come from `cfg`.
    pub fn new(groups: Vec<ParamGroup>, cfg: SophiaConfig) -> Result<Self> {
        validate_groups(&groups)?;
        validate_sophia_global(cfg.rho, cfg.update_period)?;
        for g in &groups {
            validate_sophia_group(
                &g.name,
                g.base_lr,
                g.base_momentum,
                g.beta2,
                g.eps,
                g.weight_decay,
                cfg.rho,
            )?;
            validate_sophia_group(&g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay, cfg.rho)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "sophia: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = zero_state(&groups)?;
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 2).sum());
        Ok(SophiaG {
            groups,
            state,
            rho: cfg.rho,
            update_period: cfg.update_period,
            auto_gauss_newton: cfg.auto_gauss_newton,
            hutchinson_dist: cfg.hutchinson_dist,
            rng: Rng::seed(cfg.seed),
            hessian_updates: 0,
            last_hessian_step: None,
            pending_resume: None,
            state_keys,
            step: 0,
        })
    }

    /// Convenience constructor for a single `"params"` group configured by [`SophiaConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: SophiaConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(true);
        Self::new(vec![group], cfg)
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (e.g. for LR / momentum schedules).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Total completed [`SophiaG::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Total completed Hessian EMA updates (explicit or automatic).
    pub fn hessian_updates(&self) -> u64 {
        self.hessian_updates
    }

    /// Configured clipping threshold `rho`.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Configured Hessian update cadence `k`.
    pub fn update_period(&self) -> u64 {
        self.update_period
    }

    /// True when the current step is scheduled for a Hessian update (`step % update_period == 0`) and has
    /// not yet received one.
    pub fn should_update_hessian(&self) -> bool {
        self.step.is_multiple_of(self.update_period) && self.last_hessian_step != Some(self.step)
    }

    /// `(exp_avg, hessian)` state buffers for `(group, param)`.
    pub fn state_for(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        self.state.get(group)?.get(param).map(|s| (&s.exp_avg, &s.hessian))
    }

    /// Reseed the internal deterministic Hutchinson probe RNG.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::seed(seed);
    }

    /// Snapshot the internal xoshiro256++ RNG state words.
    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.state()
    }

    /// Sample one deterministic Hutchinson probe tensor `v` (Rademacher `±1` or Gaussian `N(0, 1)`) per
    /// parameter, matching each parameter's shape and device (`F32`, detached).
    pub fn sample_hutchinson_probes(&mut self) -> Result<HashMap<String, Tensor>> {
        let _ng = no_grad();
        let total = self.groups.iter().map(|g| g.params.len()).sum();
        let mut probes = HashMap::with_capacity(total);
        for grp in &self.groups {
            for (name, p) in &grp.params {
                let n = p.numel();
                let mut vals = Vec::with_capacity(n);
                match self.hutchinson_dist {
                    HutchinsonDist::Rademacher => {
                        for _ in 0..n {
                            vals.push(if self.rng.bernoulli(0.5) { 1.0f32 } else { -1.0f32 });
                        }
                    }
                    HutchinsonDist::Gaussian => {
                        for _ in 0..n {
                            vals.push(self.rng.normal());
                        }
                    }
                }
                let t = Tensor::from_vec(vals, p.shape().clone())?.to_device(p.device())?;
                probes.insert(name.clone(), t);
            }
        }
        Ok(probes)
    }

    /// Update the diagonal Hessian EMA using empirical Gauss-Newton (`h_est = g_hat^2`, Liu et al. 2024
    /// Algorithm 2) for every parameter present in `gn_grads`:
    ///
    /// ```text
    /// h_t = beta2 * h_{t-1} + (1 - beta2) * g_hat^2
    /// ```
    pub fn update_hessian_gauss_newton(&mut self, gn_grads: &dyn GradSource) -> Result<()> {
        self.validate_alignment(Some(gn_grads))?;
        let _ng = no_grad();
        let mut any_updated = false;
        for (grp, st) in self.groups.iter().zip(&self.state) {
            let b2 = grp.beta2 as f32;
            let one_m_b2 = (1.0 - grp.beta2) as f32;
            for ((_, p), ps) in grp.params.iter().zip(st) {
                let Some(g) = gn_grads.grad(p) else { continue };
                any_updated = true;
                update_hessian_gn_one(&ps.hessian, g, b2, one_m_b2)?;
            }
        }
        if any_updated {
            self.hessian_updates = self.hessian_updates.saturating_add(1);
            self.last_hessian_step = Some(self.step);
        }
        self.pending_resume = None;
        Ok(())
    }

    /// Update the diagonal Hessian EMA using a Hutchinson Hessian-vector product (`h_est = v ⊙ (H v)`):
    ///
    /// ```text
    /// h_t = beta2 * h_{t-1} + (1 - beta2) * (probe * hvp)
    /// ```
    ///
    /// `probes` is keyed by parameter name (as returned by [`SophiaG::sample_hutchinson_probes`]). Retaining
    /// the signed product `v ⊙ (H v)` inside the EMA keeps `E[h_est] = diag(H)` unbiased; the step
    /// denominator `max(h_t, eps)` guards against negative curvature or estimation noise.
    pub fn update_hessian_hutchinson(
        &mut self,
        hvp: &dyn GradSource,
        probes: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.validate_alignment(Some(hvp))?;
        for grp in &self.groups {
            for (name, p) in &grp.params {
                if hvp.grad(p).is_none() {
                    continue;
                }
                let Some(v) = probes.get(name) else {
                    return Err(Error::Invalid(format!(
                        "sophia: missing Hutchinson probe for parameter `{name}`"
                    )));
                };
                if v.shape() != p.shape() || v.device() != p.device() || !v.dtype().is_float() {
                    return Err(Error::Invalid(format!(
                        "sophia: probe for `{name}` ({:?} {:?} {}) does not match parameter ({:?} {:?})",
                        v.dims(),
                        v.device(),
                        v.dtype(),
                        p.dims(),
                        p.device()
                    )));
                }
            }
        }
        let _ng = no_grad();
        let mut any_updated = false;
        for (grp, st) in self.groups.iter().zip(&self.state) {
            let b2 = grp.beta2 as f32;
            let one_m_b2 = (1.0 - grp.beta2) as f32;
            for ((name, p), ps) in grp.params.iter().zip(st) {
                let Some(hv) = hvp.grad(p) else { continue };
                let v = &probes[name];
                any_updated = true;
                update_hessian_hutch_one(&ps.hessian, hv, v, b2, one_m_b2)?;
            }
        }
        if any_updated {
            self.hessian_updates = self.hessian_updates.saturating_add(1);
            self.last_hessian_step = Some(self.step);
        }
        self.pending_resume = None;
        Ok(())
    }

    /// Convenience wrapper that samples deterministic Hutchinson probes via
    /// [`SophiaG::sample_hutchinson_probes`], invokes `compute_hvp(&probes)` to evaluate `H v`, and updates
    /// the diagonal Hessian EMA via [`SophiaG::update_hessian_hutchinson`].
    pub fn update_hessian_hutchinson_with<G, F>(&mut self, mut compute_hvp: F) -> Result<()>
    where
        G: GradSource,
        F: FnMut(&HashMap<String, Tensor>) -> Result<G>,
    {
        let probes = self.sample_hutchinson_probes()?;
        let hvp = compute_hvp(&probes)?;
        self.update_hessian_hutchinson(&hvp, &probes)
    }

    /// Perform one Sophia-G optimization step over every parameter with a gradient in `grads`.
    ///
    /// When `auto_gauss_newton` is `true` and [`SophiaG::should_update_hessian`] is `true`, updates the
    /// diagonal Hessian EMA from `grads` (`h_est = g_t^2`) before taking the clipped second-order step.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.validate_alignment(Some(grads))?;
        let auto_gn_now = self.auto_gauss_newton && self.should_update_hessian();

        let _ng = no_grad();
        let mut any_active = false;
        for (grp, st) in self.groups.iter().zip(&mut self.state) {
            let lr = grp.lr as f32;
            let b1 = grp.momentum as f32;
            let one_m_b1 = (1.0 - grp.momentum) as f32;
            let b2 = grp.beta2 as f32;
            let one_m_b2 = (1.0 - grp.beta2) as f32;
            let eps = grp.eps as f32;
            let rho = self.rho as f32;
            let decay = (grp.weight_decay != 0.0).then_some((1.0 - grp.lr * grp.weight_decay) as f32);
            for ((_, p), ps) in grp.params.iter().zip(st.iter_mut()) {
                let Some(g) = grads.grad(p) else { continue };
                any_active = true;
                if auto_gn_now {
                    update_hessian_gn_one(&ps.hessian, g, b2, one_m_b2)?;
                }
                sophia_step_one(p, g, &ps.exp_avg, &ps.hessian, lr, b1, one_m_b1, eps, rho, decay)?;
                ps.param_step = ps.param_step.saturating_add(1);
            }
        }
        if auto_gn_now && any_active {
            self.hessian_updates = self.hessian_updates.saturating_add(1);
            self.last_hessian_step = Some(self.step);
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
                ps.hessian.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Export the serializable [`SophiaContract`].
    pub fn contract(&self) -> SophiaContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.state)
            .map(|(g, st)| SophiaGroupContract {
                name: g.name.clone(),
                params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                lr: g.base_lr,
                current_lr: (g.lr != g.base_lr).then_some(g.lr),
                betas: [g.base_momentum, g.beta2],
                current_beta1: (g.momentum != g.base_momentum).then_some(g.momentum),
                eps: g.eps,
                weight_decay: g.weight_decay,
                param_steps: st.iter().map(|ps| ps.param_step).collect(),
            })
            .collect();
        SophiaContract {
            groups,
            step: self.step,
            hessian_updates: self.hessian_updates,
            last_hessian_step: self.last_hessian_step,
            rho: self.rho,
            update_period: self.update_period,
            auto_gauss_newton: self.auto_gauss_newton,
            hutchinson_dist: self.hutchinson_dist,
            rng_state: self.rng.state(),
        }
    }

    /// Validate `c` and stage its hyperparameters, cadence, and RNG state for atomic commit in
    /// [`SophiaG::load_state`].
    pub fn apply_contract(&mut self, c: &SophiaContract) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Load(format!(
                "sophia: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        for (g, st) in self.groups.iter().zip(&self.state) {
            if g.params.len() != st.len() {
                return Err(Error::Load(format!(
                    "sophia: group `{}` parameter count {} does not match state {}",
                    g.name,
                    g.params.len(),
                    st.len()
                )));
            }
        }
        validate_sophia_global(c.rho, c.update_period).map_err(|e| Error::Load(e.to_string()))?;
        if Rng::from_state(c.rng_state).is_none() {
            return Err(Error::Load("sophia contract: all-zero rng_state is invalid".into()));
        }
        if let Some(last) = c.last_hessian_step
            && last > c.step
        {
            return Err(Error::Load(format!("sophia contract: last_hessian_step {last} > step {}", c.step)));
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
        for (gc, grp) in c.groups.iter().zip(&self.groups) {
            if gc.param_steps.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "sophia contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_sophia_group(&gc.name, gc.lr, gc.betas[0], gc.betas[1], gc.eps, gc.weight_decay, c.rho)
                .map_err(|e| Error::Load(e.to_string()))?;
            validate_sophia_group(&gc.name, cur_lr, cur_b1, gc.betas[1], gc.eps, gc.weight_decay, c.rho)
                .map_err(|e| Error::Load(e.to_string()))?;
            let mut by_name: HashMap<&str, u64> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "sophia contract group `{}` param `{pname}`: param_step {k} > step {}",
                        gc.name, c.step
                    )));
                }
                by_name.insert(pname.as_str(), k);
            }
            staged_steps.push(grp.params.iter().map(|(pname, _)| by_name[pname.as_str()]).collect());
        }

        let staged_groups = c
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
            hessian_updates: c.hessian_updates,
            last_hessian_step: c.last_hessian_step,
            rho: c.rho,
            update_period: c.update_period,
            auto_gauss_newton: c.auto_gauss_newton,
            hutchinson_dist: c.hutchinson_dist,
            rng_state: c.rng_state,
            group_names: self.groups.iter().map(|g| g.name.clone()).collect(),
            param_names: self
                .groups
                .iter()
                .map(|g| g.params.iter().map(|(n, _)| n.clone()).collect())
                .collect(),
            groups: staged_groups,
            param_steps: staged_steps,
        });
        Ok(())
    }

    /// Export all optimizer state tensors (`<param>.exp_avg`, `<param>.hessian`).
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(self.groups.iter().map(|g| g.params.len() * 2).sum());
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), ps.exp_avg.clone()));
                out.push((format!("{name}.hessian"), ps.hessian.clone()));
            }
        }
        Ok(out)
    }

    /// Restore step counter and state tensors from a checkpoint map.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "SophiaG state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "SophiaG state: step {step} does not match staged contract step {}",
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
                    "SophiaG state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }

        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        let h_updates = self.pending_resume.as_ref().map_or(0, |pr| pr.hessian_updates);
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                if let Some(t) = take_state(state, name, p, "exp_avg", p_step > 0, &mut problems) {
                    loaded.push((gi, pi, 0usize, t));
                }
                if let Some(t) =
                    take_state(state, name, p, "hessian", p_step > 0 || h_updates > 0, &mut problems)
                {
                    let vals = t.to_vec::<f32>()?;
                    if vals.iter().any(|v| v.is_nan()) {
                        problems.push(format!("`optim.{name}.hessian` contains NaN values"));
                    } else {
                        loaded.push((gi, pi, 1usize, t));
                    }
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("SophiaG state: {}", problems.join("; "))));
        }

        let mut fresh = zero_state(&self.groups)?;
        let _ng = no_grad();
        for (gi, pi, k, t) in loaded {
            let ps = &fresh[gi][pi];
            let dst = if k == 0 { &ps.exp_avg } else { &ps.hessian };
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
            }
            for (st, psteps) in fresh.iter_mut().zip(pending.param_steps) {
                for (ps, k) in st.iter_mut().zip(psteps) {
                    ps.param_step = k;
                }
            }
            self.hessian_updates = pending.hessian_updates;
            self.last_hessian_step = pending.last_hessian_step;
            self.rho = pending.rho;
            self.update_period = pending.update_period;
            self.auto_gauss_newton = pending.auto_gauss_newton;
            self.hutchinson_dist = pending.hutchinson_dist;
            self.rng = Rng::from_state(pending.rng_state).expect("validated in apply_contract");
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }

    fn validate_alignment(&mut self, grads: Option<&dyn GradSource>) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "sophia: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_sophia_group(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                grp.eps,
                grp.weight_decay,
                self.rho,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "sophia: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "sophia: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "sophia: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                for (sname, buf) in [("exp_avg", &ps.exp_avg), ("hessian", &ps.hessian)] {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "sophia: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
                            p.dims(),
                            p.device(),
                            buf.dims(),
                            buf.device(),
                            buf.dtype()
                        )));
                    }
                    self.state_keys.insert(storage_key(buf));
                }
                if let Some(src) = grads
                    && let Some(g) = src.grad(p)
                {
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "sophia: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "sophia: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "sophia: gradient of `{name}` has non-float dtype {}",
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
                        "sophia: parameter `{name}` aliases an optimizer state buffer"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn validate_sophia_global(rho: f64, update_period: u64) -> Result<()> {
    if !(rho.is_finite() && rho > 0.0 && (rho as f32).is_finite() && (rho as f32) > 0.0) {
        return Err(Error::Invalid(format!("sophia: rho must be finite and > 0 in f32, got {rho}")));
    }
    if update_period == 0 {
        return Err(Error::Invalid("sophia: update_period must be >= 1, got 0".into()));
    }
    Ok(())
}

fn validate_sophia_group(group: &str, lr: f64, b1: f64, b2: f64, eps: f64, wd: f64, rho: f64) -> Result<()> {
    if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite() && ((lr * rho) as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "sophia group `{group}`: lr must be finite in f32 and >= 0 with finite lr * rho, got {lr}"
        )));
    }
    if !(b1.is_finite() && (0.0..1.0).contains(&b1) && (b1 as f32) < 1.0 && ((1.0 - b1) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "sophia group `{group}`: beta1 must be finite in [0, 1) with f32(beta1) < 1, got {b1}"
        )));
    }
    if !(b2.is_finite() && (0.0..1.0).contains(&b2) && (b2 as f32) < 1.0 && ((1.0 - b2) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "sophia group `{group}`: beta2 must be finite in [0, 1) with f32(beta2) < 1, got {b2}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "sophia group `{group}`: eps must be finite and > 0 in f32, got {eps}"
        )));
    }
    if !(wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite() && ((1.0 - lr * wd) as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "sophia group `{group}`: weight_decay must be finite in f32 and >= 0 with finite (1 - lr * wd), got {wd}"
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
                        hessian: Tensor::zeros_with(shape, DType::F32, dev)?,
                        param_step: 0,
                    })
                })
                .collect()
        })
        .collect()
}

fn update_hessian_gn_one(h: &Tensor, g: &Tensor, b2: f32, one_m_b2: f32) -> Result<()> {
    if cpu_fast(&[h]) && g.device().is_cpu() && g.dtype() == DType::F32 && !h.same_storage(g) {
        return g.with_f32_slice(|gs| -> Result<()> {
            h.update_f32(|hs| {
                for (hi, &gi) in hs.iter_mut().zip(gs) {
                    *hi = b2 * *hi + one_m_b2 * (gi * gi);
                }
            })?;
            Ok(())
        })?;
    }
    let g_ref = GradRef::as_f32(g)?;
    let h_new = h.mul_scalar(b2)?.add(&g_ref.as_tensor().square()?.mul_scalar(one_m_b2)?)?;
    h.copy_from(&h_new)?;
    Ok(())
}

fn update_hessian_hutch_one(h: &Tensor, hv: &Tensor, v: &Tensor, b2: f32, one_m_b2: f32) -> Result<()> {
    if cpu_fast(&[h, v])
        && hv.device().is_cpu()
        && hv.dtype() == DType::F32
        && v.dtype() == DType::F32
        && !h.same_storage(hv)
        && !h.same_storage(v)
    {
        return hv.with_f32_slice(|hvs| -> Result<()> {
            v.with_f32_slice(|vs| -> Result<()> {
                h.update_f32(|hs| {
                    for ((hi, &hvi), &vi) in hs.iter_mut().zip(hvs).zip(vs) {
                        *hi = b2 * *hi + one_m_b2 * (vi * hvi);
                    }
                })?;
                Ok(())
            })??;
            Ok(())
        })?;
    }
    let hv_ref = GradRef::as_f32(hv)?;
    let v_ref = GradRef::as_f32(v)?;
    let est = v_ref.as_tensor().mul(hv_ref.as_tensor())?;
    let h_new = h.mul_scalar(b2)?.add(&est.mul_scalar(one_m_b2)?)?;
    h.copy_from(&h_new)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sophia_step_one(
    p: &Tensor,
    g: &Tensor,
    m: &Tensor,
    h: &Tensor,
    lr: f32,
    b1: f32,
    one_m_b1: f32,
    eps: f32,
    rho: f32,
    decay: Option<f32>,
) -> Result<()> {
    let can_fast = cpu_fast(&[p, m, h])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !m.same_storage(g)
        && !h.same_storage(g);
    if can_fast {
        return g.with_f32_slice(|gs| -> Result<()> {
            h.with_f32_slice(|hs| -> Result<()> {
                m.update_f32(|ms| {
                    p.update_f32(|ps_slice| {
                        for (((pi, &gi), mi), &hi) in ps_slice.iter_mut().zip(gs).zip(ms.iter_mut()).zip(hs) {
                            if let Some(d) = decay {
                                *pi *= d;
                            }
                            *mi = b1 * *mi + one_m_b1 * gi;
                            let denom = if hi.is_nan() {
                                f32::NAN
                            } else if hi > eps {
                                hi
                            } else {
                                eps
                            };
                            let u = (*mi / denom).clamp(-rho, rho);
                            *pi -= lr * u;
                        }
                    })
                })??;
                Ok(())
            })??;
            Ok(())
        })?;
    }

    let g_ref = GradRef::as_f32(g)?;
    let p_base = if let Some(d) = decay { p.mul_scalar(d)? } else { p.clone() };
    let m_new = m.mul_scalar(b1)?.add(&g_ref.as_tensor().mul_scalar(one_m_b1)?)?;
    let denom = h.maximum_scalar(eps)?;
    let u = m_new.div(&denom)?.clamp(-rho, rho)?;
    let p_new = p_base.sub(&u.mul_scalar(lr)?)?;
    p.copy_from(&p_new)?;
    m.copy_from(&m_new)?;
    Ok(())
}
