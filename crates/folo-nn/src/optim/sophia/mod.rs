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
//! Supports both published diagonal curvature estimators:
//! - **Gauss–Newton–Bartlett (`Sophia-G`, §2.2 Algorithm 2)**: `h_est = B * g_hat^2` from gradients `g_hat`
//!   evaluated on model-sampled labels over a mini-batch of size `B`, via
//!   [`SophiaG::update_hessian_gauss_newton_scaled`] (or [`SophiaG::update_hessian_gauss_newton`] for
//!   already-scaled `B = 1` gradients), and
//! - **Seeded Hutchinson (`Sophia-H`)**: `h_est = v ⊙ (H v)` with deterministic Rademacher (`±1`) or
//!   Gaussian (`N(0, 1)`) probes via [`SophiaG::sample_hutchinson_probes`] +
//!   [`SophiaG::update_hessian_hutchinson`] (or [`SophiaG::update_hessian_hutchinson_with`]).
//!
//! By default (`auto_gauss_newton = false`), Hessian updates are driven explicitly so the caller supplies a
//! proper Gauss–Newton–Bartlett or Hutchinson estimate rather than squaring the ordinary training gradient.
//! Setting `auto_gauss_newton = true` enables an opt-in empirical-Fisher convenience heuristic (`g_t^2`) on
//! `step % update_period == 0` when no explicit Hessian update was supplied for the current step.

mod contract;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{GradRef, GradSource, ParamGroup, cpu_fast, storage_key, validate_groups};
use crate::{Error, Result};
use contract::PendingResume;
pub use contract::{SophiaContract, SophiaGroupContract};

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
    /// Opt-in empirical-Fisher convenience heuristic (default `false`). When `true`, [`SophiaG::step`]
    /// automatically updates `h_t` from the squared training gradient (`g_t^2`) whenever
    /// `step % update_period == 0` and no explicit `update_hessian_*` call was made for the current step.
    /// Keep `false` (default) to drive exact Gauss–Newton–Bartlett (`update_hessian_gauss_newton_scaled`)
    /// or Hutchinson (`update_hessian_hutchinson`) updates explicitly.
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
            auto_gauss_newton: false,
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

pub(super) struct ParamState {
    pub(super) exp_avg: Tensor,
    pub(super) hessian: Tensor,
    pub(super) param_step: u64,
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

    /// Total completed Hessian EMA updates (explicit or opt-in automatic).
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

    /// Update the diagonal Hessian EMA using the published Gauss–Newton–Bartlett estimator
    /// (Liu et al., ICLR 2024 §2.2 Algorithm 2):
    ///
    /// ```text
    /// h_t = beta2 * h_{t-1} + (1 - beta2) * batch_size * g_hat^2
    /// ```
    ///
    /// where `gn_grads` holds the mean mini-batch gradient `g_hat` evaluated on labels sampled from the
    /// model's predictive distribution, and `batch_size > 0` is the mini-batch size `B`.
    pub fn update_hessian_gauss_newton_scaled(
        &mut self,
        gn_grads: &dyn GradSource,
        batch_size: f64,
    ) -> Result<()> {
        if !(batch_size.is_finite()
            && batch_size > 0.0
            && (batch_size as f32).is_finite()
            && (batch_size as f32) > 0.0)
        {
            return Err(Error::Invalid(format!(
                "sophia: batch_size must be finite and > 0 in f32, got {batch_size}"
            )));
        }
        self.validate_alignment(Some(gn_grads))?;
        let _ng = no_grad();
        let mut any_updated = false;
        for (grp, st) in self.groups.iter().zip(&self.state) {
            let b2 = grp.beta2 as f32;
            let coeff = ((1.0 - grp.beta2) * batch_size) as f32;
            if !coeff.is_finite() {
                return Err(Error::Invalid(format!(
                    "sophia: (1 - beta2) * batch_size overflows f32 for group `{}`",
                    grp.name
                )));
            }
            for ((_, p), ps) in grp.params.iter().zip(st) {
                let Some(g) = gn_grads.grad(p) else { continue };
                any_updated = true;
                update_hessian_gn_one(&ps.hessian, g, b2, coeff)?;
            }
        }
        if any_updated {
            self.hessian_updates = self.hessian_updates.saturating_add(1);
            self.last_hessian_step = Some(self.step);
        }
        self.pending_resume = None;
        Ok(())
    }

    /// Convenience alias for [`SophiaG::update_hessian_gauss_newton_scaled`] with `batch_size = 1.0`
    /// (for per-example or pre-scaled Gauss–Newton–Bartlett gradients `g_hat`).
    pub fn update_hessian_gauss_newton(&mut self, gn_grads: &dyn GradSource) -> Result<()> {
        self.update_hessian_gauss_newton_scaled(gn_grads, 1.0)
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
    /// When `auto_gauss_newton` is explicitly enabled and [`SophiaG::should_update_hessian`] is `true`,
    /// updates the diagonal Hessian EMA from the squared training gradient (`g_t^2`) before taking the
    /// clipped second-order step (safely snapshotting `g` first if it aliases `ps.hessian`).
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
                    if ps.hessian.same_storage(g) {
                        // Snapshot pre-step gradient before updating `ps.hessian` so momentum reads the
                        // unmutated gradient even when `g` aliases `ps.hessian`.
                        let g_snap = Tensor::zeros_with(g.shape(), g.dtype(), g.device())?;
                        g_snap.copy_from(g)?;
                        update_hessian_gn_one(&ps.hessian, &g_snap, b2, one_m_b2)?;
                        sophia_step_one(
                            p,
                            &g_snap,
                            &ps.exp_avg,
                            &ps.hessian,
                            lr,
                            b1,
                            one_m_b1,
                            eps,
                            rho,
                            decay,
                        )?;
                    } else {
                        update_hessian_gn_one(&ps.hessian, g, b2, one_m_b2)?;
                        sophia_step_one(p, g, &ps.exp_avg, &ps.hessian, lr, b1, one_m_b1, eps, rho, decay)?;
                    }
                } else {
                    sophia_step_one(p, g, &ps.exp_avg, &ps.hessian, lr, b1, one_m_b1, eps, rho, decay)?;
                }
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

pub(super) fn validate_sophia_global(rho: f64, update_period: u64) -> Result<()> {
    if !(rho.is_finite() && rho > 0.0 && (rho as f32).is_finite() && (rho as f32) > 0.0) {
        return Err(Error::Invalid(format!("sophia: rho must be finite and > 0 in f32, got {rho}")));
    }
    if update_period == 0 {
        return Err(Error::Invalid("sophia: update_period must be >= 1, got 0".into()));
    }
    Ok(())
}

pub(super) fn validate_sophia_group(
    group: &str,
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
    rho: f64,
) -> Result<()> {
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

pub(super) fn zero_state(groups: &[ParamGroup]) -> Result<Vec<Vec<ParamState>>> {
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

fn update_hessian_gn_one(h: &Tensor, g: &Tensor, b2: f32, coeff: f32) -> Result<()> {
    if cpu_fast(&[h]) && g.device().is_cpu() && g.dtype() == DType::F32 && !h.same_storage(g) {
        let b2_64 = b2 as f64;
        let coeff_64 = coeff as f64;
        return g.with_f32_slice(|gs| -> Result<()> {
            h.update_f32(|hs| {
                for (hi, &gi) in hs.iter_mut().zip(gs) {
                    let g64 = gi as f64;
                    *hi = (b2_64 * (*hi as f64) + coeff_64 * (g64 * g64)) as f32;
                }
            })?;
            Ok(())
        })?;
    }
    let g_ref = GradRef::as_f32(g)?;
    let h_new = h.mul_scalar(b2)?.add(&g_ref.as_tensor().mul_scalar(coeff.sqrt())?.square()?)?;
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

    // Device-generic / aliased fallback with explicit NaN propagation (`maximum_scalar` and `clamp` use
    // NaN-ignoring IEEE `maxNum`/`minNum`, so preserve NaN entries via `ne(self)` + `where_cond`).
    let g_ref = GradRef::as_f32(g)?;
    let p_base = if let Some(d) = decay { p.mul_scalar(d)? } else { p.clone() };
    let m_new = m.mul_scalar(b1)?.add(&g_ref.as_tensor().mul_scalar(one_m_b1)?)?;
    let h_is_nan = h.ne(h)?;
    let denom = h_is_nan.where_cond(h, &h.maximum_scalar(eps)?)?;
    let ratio = m_new.div(&denom)?;
    let ratio_is_nan = ratio.ne(&ratio)?;
    let u = ratio_is_nan.where_cond(&ratio, &ratio.clamp(-rho, rho)?)?;
    let p_new = p_base.sub(&u.mul_scalar(lr)?)?;
    p.copy_from(&p_new)?;
    m.copy_from(&m_new)?;
    Ok(())
}
