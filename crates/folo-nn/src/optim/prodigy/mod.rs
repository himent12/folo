//! Prodigy: an expeditiously adaptive parameter-free AdamW optimizer (Mishchenko & Defazio, ICML 2024 /
//! arXiv:2306.06101).
//!
//! Prodigy estimates the distance to the solution `D = ||x_0 - x_*||` on the fly via a monotonically
//! non-decreasing step-size scale `d_k` starting from `d_0` (default `1e-6`), so the user typically keeps
//! `lr = 1.0` (with an optional cosine/warmup schedule multiplier in `[0, 1]`):
//!
//! ```text
//! bc_k   = sqrt(1 - beta2^t) / (1 - beta1^t)          (when bias_correction = true, else 1)
//! dlr_k  = d_k * lr * bc_k
//! r_k    = beta3 * r_{k-1} + (d_k / d_0) * dlr_k * <g_k, x_0 - p_k>
//! s_k    = beta3 * s_{k-1} + (d_k / d_0) * s_scale * g_k
//!          where s_scale = d_k if safeguard_warmup else dlr_k
//! d_hat  = d_coef * r_k / ||s_k||_1
//! d_{k+1}= max(d_k, min(d_hat, d_k * growth_rate))
//!
//! m_k    = beta1 * m_{k-1} + (1 - beta1) * d_k * g_k
//! v_k    = beta2 * v_{k-1} + (1 - beta2) * (d_k * g_k)^2
//! p_{k+1}= p_k * (1 - d_k * lr * wd) - dlr_k * m_k / (sqrt(v_k) + d_k * eps)   (decoupled_wd = true)
//! ```
//!
//! Parameters without a gradient in a step are skipped completely (no weight decay, no moment update, and no
//! per-parameter step increment). Distance statistics (`r_k` and `||s_k||_1`) are accumulated in `f64` in
//! canonical parameter order for deterministic updates.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradRef, GradSource, ParamGroup, check_param_sets, cpu_fast, storage_key, take_state, validate_groups,
};
use crate::{Error, Result};

/// Default Prodigy learning-rate schedule multiplier (`1.0`).
pub const PRODIGY_DEFAULT_LR: f64 = 1.0;
/// Default Prodigy first-moment retention `beta1` (`0.9`).
pub const PRODIGY_DEFAULT_BETA1: f64 = 0.9;
/// Default Prodigy second-moment retention `beta2` (`0.999`).
pub const PRODIGY_DEFAULT_BETA2: f64 = 0.999;
/// Default Prodigy numerical stability floor `eps` (`1e-8`).
pub const PRODIGY_DEFAULT_EPS: f64 = 1e-8;
/// Default Prodigy initial distance estimate `d0` (`1e-6`).
pub const PRODIGY_DEFAULT_D0: f64 = 1e-6;
/// Default Prodigy distance multiplier `d_coef` (`1.0`).
pub const PRODIGY_DEFAULT_D_COEF: f64 = 1.0;
/// Default Prodigy decoupled weight decay (`0.0`).
pub const PRODIGY_DEFAULT_WEIGHT_DECAY: f64 = 0.0;

/// Hyper-parameters for constructing a [`Prodigy`] optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProdigyConfig {
    /// Learning-rate multiplier `>= 0` (default `1.0`; cosine/warmup schedules multiply this value).
    pub lr: f64,
    /// EMA retention factors `[beta1, beta2]`, each in `[0, 1)` (default `[0.9, 0.999]`).
    pub betas: [f64; 2],
    /// Optional EMA factor `beta3 in [0, 1)` for `r_k` and `s_k`. When `None`, defaults to `sqrt(beta2)`.
    pub beta3: Option<f64>,
    /// Denominator floor `eps > 0` (default `1e-8`).
    pub eps: f64,
    /// Weight decay `>= 0` (default `0.0`).
    pub weight_decay: f64,
    /// `true` (default): AdamW-style decoupled decay `p *= 1 - d * lr * wd`.
    /// `false`: coupled L2 decay `g_eff = g + wd * p`.
    pub decoupled_wd: bool,
    /// Initial estimate `d0 > 0` of `||x_0 - x_*||` (default `1e-6`).
    pub d0: f64,
    /// Scaling coefficient `d_coef > 0` applied to `d_hat` (default `1.0`).
    pub d_coef: f64,
    /// Maximum multiplicative growth of `d_k` per step (`>= 1.0`, default `f64::INFINITY`).
    pub growth_rate: f64,
    /// When `true`, scales `s_k` updates by `d_k` instead of `d_k * lr * bc` so small warmup LRs cannot
    /// cause `d_hat` to spike (default `false`).
    pub safeguard_warmup: bool,
    /// Apply Adam-style bias correction `sqrt(1 - beta2^t) / (1 - beta1^t)` to `dlr` (default `true`).
    pub bias_correction: bool,
}

impl Default for ProdigyConfig {
    fn default() -> Self {
        Self {
            lr: PRODIGY_DEFAULT_LR,
            betas: [PRODIGY_DEFAULT_BETA1, PRODIGY_DEFAULT_BETA2],
            beta3: None,
            eps: PRODIGY_DEFAULT_EPS,
            weight_decay: PRODIGY_DEFAULT_WEIGHT_DECAY,
            decoupled_wd: true,
            d0: PRODIGY_DEFAULT_D0,
            d_coef: PRODIGY_DEFAULT_D_COEF,
            growth_rate: f64::INFINITY,
            safeguard_warmup: false,
            bias_correction: true,
        }
    }
}

impl ProdigyConfig {
    /// Validate all Prodigy hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        validate_prodigy_global(self.beta3, self.d0, self.d_coef, self.growth_rate)?;
        validate_prodigy_group(
            "params",
            self.lr,
            self.betas[0],
            self.betas[1],
            self.eps,
            self.weight_decay,
            self.d0,
        )
    }
}

/// Serializable contract for one [`Prodigy`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProdigyGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning-rate multiplier `>= 0`.
    pub lr: f64,
    /// Current effective learning-rate multiplier (defaults to `lr` when omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base `[beta1, beta2]`.
    pub betas: [f64; 2],
    /// Current effective `beta1` (defaults to `betas[0]` when omitted).
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

/// Serializable contract for [`Prodigy`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProdigyContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<ProdigyGroupContract>,
    /// Total completed [`Prodigy::step`] calls.
    pub step: u64,
    /// Current distance estimate `d_k >= d0`.
    pub d: f64,
    /// Initial distance estimate `d0 > 0`.
    pub d0: f64,
    /// Running weighted numerator `r_k`.
    pub d_numerator: f64,
    /// Scaling coefficient `d_coef > 0`.
    pub d_coef: f64,
    /// Optional `beta3` override (`None` = `sqrt(beta2)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta3: Option<f64>,
    /// Optional finite `growth_rate >= 1.0` (`None` encodes `f64::INFINITY` in JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub growth_rate: Option<f64>,
    /// Warmup safeguard flag.
    pub safeguard_warmup: bool,
    /// Bias-correction flag.
    pub bias_correction: bool,
}

struct ParamState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    s: Tensor,
    x0: Tensor,
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
    d: f64,
    d0: f64,
    d_numerator: f64,
    d_coef: f64,
    beta3: Option<f64>,
    growth_rate: f64,
    safeguard_warmup: bool,
    bias_correction: bool,
    group_names: Vec<String>,
    param_names: Vec<Vec<String>>,
    groups: Vec<StagedGroup>,
    param_steps: Vec<Vec<u64>>,
    beta1_prods: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
struct StepParams {
    lr: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    eps: f64,
    wd: f64,
    decoupled_wd: bool,
    d: f64,
    d0: f64,
    safeguard_warmup: bool,
    bias_correction: bool,
}

/// Prodigy parameter-free adaptive AdamW optimizer (Mishchenko & Defazio, ICML 2024).
pub struct Prodigy {
    groups: Vec<ParamGroup>,
    state: Vec<Vec<ParamState>>,
    d: f64,
    d0: f64,
    d_numerator: f64,
    d_coef: f64,
    beta3: Option<f64>,
    growth_rate: f64,
    safeguard_warmup: bool,
    bias_correction: bool,
    pending_resume: Option<PendingResume>,
    state_keys: HashSet<usize>,
    step: u64,
}

impl Prodigy {
    /// Construct a [`Prodigy`] optimizer over `groups` with global Prodigy settings from `cfg`.
    ///
    /// Group-level `lr`, `momentum` (`beta1`), `beta2`, `eps`, `weight_decay`, and `decoupled_wd` are read
    /// from each [`ParamGroup`]; global distance-estimation settings (`d0`, `d_coef`, `beta3`, `growth_rate`,
    /// `safeguard_warmup`, `bias_correction`) are taken from `cfg`.
    pub fn new(groups: Vec<ParamGroup>, cfg: ProdigyConfig) -> Result<Self> {
        validate_groups(&groups)?;
        validate_prodigy_global(cfg.beta3, cfg.d0, cfg.d_coef, cfg.growth_rate)?;
        for g in &groups {
            validate_prodigy_group(
                &g.name,
                g.base_lr,
                g.base_momentum,
                g.beta2,
                g.eps,
                g.weight_decay,
                cfg.d0,
            )?;
            validate_prodigy_group(&g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay, cfg.d0)?;
            for (name, p) in &g.params {
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "prodigy: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
            }
        }
        let state = init_state(&groups)?;
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len() * 4).sum());
        Ok(Prodigy {
            groups,
            state,
            d: cfg.d0,
            d0: cfg.d0,
            d_numerator: 0.0,
            d_coef: cfg.d_coef,
            beta3: cfg.beta3,
            growth_rate: cfg.growth_rate,
            safeguard_warmup: cfg.safeguard_warmup,
            bias_correction: cfg.bias_correction,
            pending_resume: None,
            state_keys,
            step: 0,
        })
    }

    /// Convenience constructor for a single `"params"` group configured by [`ProdigyConfig`].
    pub fn from_params(params: Vec<(String, Tensor)>, cfg: ProdigyConfig) -> Result<Self> {
        cfg.validate()?;
        let group = ParamGroup::new("params", params)
            .lr(cfg.lr)
            .momentum(cfg.betas[0])
            .adam(cfg.betas[1], cfg.eps)
            .weight_decay(cfg.weight_decay)
            .decoupled_wd(cfg.decoupled_wd);
        Self::new(vec![group], cfg)
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups (e.g. for LR / warmup schedules).
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Total completed [`Prodigy::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Current adaptive distance / step-size estimate `d_k`.
    pub fn d(&self) -> f64 {
        self.d
    }

    /// Initial distance estimate `d0`.
    pub fn d0(&self) -> f64 {
        self.d0
    }

    /// Current running weighted numerator `r_k`.
    pub fn d_numerator(&self) -> f64 {
        self.d_numerator
    }

    /// Per-parameter active step count.
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.state.get(group)?.get(param).map(|s| s.param_step)
    }

    /// Perform one Prodigy optimization step over every parameter with a gradient in `grads`.
    ///
    /// Validates all groups, hyperparameters, state alignments, and incoming gradients before mutating any
    /// parameter or state tensor.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Invalid(format!(
                "prodigy: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        self.state_keys.clear();
        let mut any_active = false;
        for (grp, st) in self.groups.iter().zip(&self.state) {
            validate_prodigy_group(
                &grp.name,
                grp.lr,
                grp.momentum,
                grp.beta2,
                grp.eps,
                grp.weight_decay,
                self.d,
            )?;
            if grp.params.len() != st.len() {
                return Err(Error::Invalid(format!(
                    "prodigy: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st.len()
                )));
            }
            for ((name, p), ps) in grp.params.iter().zip(st) {
                if p.dtype() != DType::F32 {
                    return Err(Error::Invalid(format!(
                        "prodigy: parameter `{name}` is {}, expected f32",
                        p.dtype()
                    )));
                }
                if !p.layout().is_non_overlapping() {
                    return Err(Error::Invalid(format!(
                        "prodigy: parameter `{name}` has an overlapping/broadcast layout and cannot be updated in place"
                    )));
                }
                for (sname, buf) in
                    [("exp_avg", &ps.exp_avg), ("exp_avg_sq", &ps.exp_avg_sq), ("s", &ps.s), ("x0", &ps.x0)]
                {
                    if buf.dtype() != DType::F32
                        || buf.shape() != p.shape()
                        || buf.device() != p.device()
                        || !buf.layout().is_non_overlapping()
                    {
                        return Err(Error::Invalid(format!(
                            "prodigy: parameter `{name}` ({:?} {:?}) does not match `{sname}` ({:?} {:?} {})",
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
                    any_active = true;
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "prodigy: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "prodigy: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "prodigy: gradient of `{name}` has non-float dtype {}",
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
                        "prodigy: parameter `{name}` aliases an optimizer state buffer"
                    )));
                }
            }
        }

        if !any_active {
            self.pending_resume = None;
            self.step = self.step.saturating_add(1);
            return Ok(());
        }

        let _ng = no_grad();
        let d_cur = self.d;
        let b3_num = self.beta3.unwrap_or_else(|| {
            self.groups
                .iter()
                .find(|g| g.params.iter().any(|(_, p)| grads.grad(p).is_some()))
                .map_or(PRODIGY_DEFAULT_BETA2, |g| g.beta2)
                .sqrt()
        });

        let mut d_num_delta = 0.0f64;
        let mut d_denom = 0.0f64;

        for (grp, st) in self.groups.iter().zip(&mut self.state) {
            let b3 = self.beta3.unwrap_or_else(|| grp.beta2.sqrt());
            let sp = StepParams {
                lr: grp.lr,
                b1: grp.momentum,
                b2: grp.beta2,
                b3,
                eps: grp.eps,
                wd: grp.weight_decay,
                decoupled_wd: grp.decoupled_wd,
                d: d_cur,
                d0: self.d0,
                safeguard_warmup: self.safeguard_warmup,
                bias_correction: self.bias_correction,
            };
            for ((_, p), ps) in grp.params.iter().zip(st.iter_mut()) {
                let Some(g) = grads.grad(p) else { continue };
                let (delta_r, s_l1) = prodigy_step_one(p, g, ps, sp)?;
                d_num_delta += delta_r;
                d_denom += s_l1;
                ps.param_step = ps.param_step.saturating_add(1);
                ps.beta1_prod *= sp.b1;
            }
        }

        let new_d_num = self.d_numerator * b3_num + d_num_delta;
        if new_d_num.is_finite() {
            self.d_numerator = new_d_num;
        }
        if d_denom.is_finite() && d_denom > 0.0 && new_d_num.is_finite() {
            let d_hat = self.d_coef * new_d_num / d_denom;
            if d_hat.is_finite() {
                let cap = if self.growth_rate.is_finite() { d_cur * self.growth_rate } else { f64::INFINITY };
                let cand = d_cur.max(d_hat.min(cap));
                if cand.is_finite() && (cand as f32).is_finite() {
                    self.d = cand;
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
                ps.s.to_device_(device)?;
                ps.x0.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Export the serializable [`ProdigyContract`].
    pub fn contract(&self) -> ProdigyContract {
        let groups = self
            .groups
            .iter()
            .zip(&self.state)
            .map(|(g, st)| ProdigyGroupContract {
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
        ProdigyContract {
            groups,
            step: self.step,
            d: self.d,
            d0: self.d0,
            d_numerator: self.d_numerator,
            d_coef: self.d_coef,
            beta3: self.beta3,
            growth_rate: self.growth_rate.is_finite().then_some(self.growth_rate),
            safeguard_warmup: self.safeguard_warmup,
            bias_correction: self.bias_correction,
        }
    }

    /// Validate `c` against this optimizer's parameter sets and stage its hyperparameters and distance
    /// statistics for atomic commit in [`Prodigy::load_state`].
    pub fn apply_contract(&mut self, c: &ProdigyContract) -> Result<()> {
        if self.groups.len() != self.state.len() {
            return Err(Error::Load(format!(
                "prodigy: group count {} does not match state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        for (g, st) in self.groups.iter().zip(&self.state) {
            if g.params.len() != st.len() {
                return Err(Error::Load(format!(
                    "prodigy: group `{}` parameter count {} does not match state {}",
                    g.name,
                    g.params.len(),
                    st.len()
                )));
            }
        }
        let growth_rate = c.growth_rate.unwrap_or(f64::INFINITY);
        validate_prodigy_global(c.beta3, c.d0, c.d_coef, growth_rate)
            .map_err(|e| Error::Load(e.to_string()))?;
        if !(c.d.is_finite() && c.d >= c.d0 && (c.d as f32).is_finite()) {
            return Err(Error::Load(format!(
                "prodigy contract: d must be finite in f32 and >= d0 ({}), got {}",
                c.d0, c.d
            )));
        }
        if !c.d_numerator.is_finite() {
            return Err(Error::Load(format!(
                "prodigy contract: d_numerator must be finite, got {}",
                c.d_numerator
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
                    "prodigy contract group `{}`: param_steps length {} != {} params",
                    gc.name,
                    gc.param_steps.len(),
                    gc.params.len()
                )));
            }
            if !gc.beta1_prods.is_empty() && gc.beta1_prods.len() != gc.params.len() {
                return Err(Error::Load(format!(
                    "prodigy contract group `{}`: beta1_prods length {} != {} params",
                    gc.name,
                    gc.beta1_prods.len(),
                    gc.params.len()
                )));
            }
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_b1 = gc.current_beta1.unwrap_or(gc.betas[0]);
            validate_prodigy_group(&gc.name, gc.lr, gc.betas[0], gc.betas[1], gc.eps, gc.weight_decay, c.d)
                .map_err(|e| Error::Load(e.to_string()))?;
            validate_prodigy_group(&gc.name, cur_lr, cur_b1, gc.betas[1], gc.eps, gc.weight_decay, c.d)
                .map_err(|e| Error::Load(e.to_string()))?;

            let mut by_name: HashMap<&str, (u64, f64)> = HashMap::with_capacity(gc.params.len());
            for (idx, pname) in gc.params.iter().enumerate() {
                let k = gc.param_steps[idx];
                if k > c.step {
                    return Err(Error::Load(format!(
                        "prodigy contract group `{}` param `{pname}`: param_step {k} > step {}",
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
                            "prodigy contract group `{}` param `{pname}`: beta1_prod must be in [0, 1], got {p}",
                            gc.name
                        )));
                    }
                    if k == 0 && p != 1.0 {
                        return Err(Error::Load(format!(
                            "prodigy contract group `{}` param `{pname}`: param_step == 0 requires beta1_prod == 1.0, got {p}",
                            gc.name
                        )));
                    }
                    if k > 0 && p >= 1.0 {
                        return Err(Error::Load(format!(
                            "prodigy contract group `{}` param `{pname}`: param_step > 0 requires beta1_prod < 1.0, got {p}",
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
                decoupled_wd: gc.decoupled_wd,
            })
            .collect();
        self.pending_resume = Some(PendingResume {
            step: c.step,
            d: c.d,
            d0: c.d0,
            d_numerator: c.d_numerator,
            d_coef: c.d_coef,
            beta3: c.beta3,
            growth_rate,
            safeguard_warmup: c.safeguard_warmup,
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

    /// Export all optimizer state tensors (`<param>.exp_avg`, `<param>.exp_avg_sq`, `<param>.s`, `<param>.x0`).
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::with_capacity(self.groups.iter().map(|g| g.params.len() * 4).sum());
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), ps.exp_avg.clone()));
                out.push((format!("{name}.exp_avg_sq"), ps.exp_avg_sq.clone()));
                out.push((format!("{name}.s"), ps.s.clone()));
                out.push((format!("{name}.x0"), ps.x0.clone()));
            }
        }
        Ok(out)
    }

    /// Restore step counter and state tensors from a checkpoint map.
    ///
    /// Resuming at `step > 0` requires staging the matching [`ProdigyContract`] via
    /// [`Prodigy::apply_contract`] first so `d`, `d_numerator`, and per-parameter active step counts are
    /// restored accurately.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "Prodigy state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "Prodigy state: step {step} does not match staged contract step {}",
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
                    "Prodigy state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }

        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                let req = p_step > 0;
                for (k, what) in ["exp_avg", "exp_avg_sq", "s", "x0"].into_iter().enumerate() {
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
            return Err(Error::Load(format!("Prodigy state: {}", problems.join("; "))));
        }

        let mut fresh = init_state(&self.groups)?;
        let _ng = no_grad();
        for (gi, pi, k, t) in loaded {
            let ps = &fresh[gi][pi];
            let dst = match k {
                0 => &ps.exp_avg,
                1 => &ps.exp_avg_sq,
                2 => &ps.s,
                _ => &ps.x0,
            };
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
            self.d = pending.d;
            self.d0 = pending.d0;
            self.d_numerator = pending.d_numerator;
            self.d_coef = pending.d_coef;
            self.beta3 = pending.beta3;
            self.growth_rate = pending.growth_rate;
            self.safeguard_warmup = pending.safeguard_warmup;
            self.bias_correction = pending.bias_correction;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}

fn validate_prodigy_global(beta3: Option<f64>, d0: f64, d_coef: f64, growth_rate: f64) -> Result<()> {
    if let Some(b3) = beta3
        && !(b3.is_finite() && (0.0..1.0).contains(&b3) && (b3 as f32) < 1.0)
    {
        return Err(Error::Invalid(format!(
            "prodigy: beta3 must be finite in [0, 1) with f32(beta3) < 1, got {b3}"
        )));
    }
    if !(d0.is_finite() && d0 > 0.0 && (d0 as f32).is_finite() && (d0 as f32) > 0.0) {
        return Err(Error::Invalid(format!("prodigy: d0 must be finite and > 0 in f32, got {d0}")));
    }
    if !(d_coef.is_finite() && d_coef > 0.0) {
        return Err(Error::Invalid(format!("prodigy: d_coef must be finite and > 0, got {d_coef}")));
    }
    if growth_rate.is_nan() || growth_rate < 1.0 {
        return Err(Error::Invalid(format!(
            "prodigy: growth_rate must be >= 1.0 (or +inf), got {growth_rate}"
        )));
    }
    Ok(())
}

fn validate_prodigy_group(group: &str, lr: f64, b1: f64, b2: f64, eps: f64, wd: f64, d: f64) -> Result<()> {
    if !(lr.is_finite() && lr >= 0.0 && (lr as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: lr must be finite in f32 and >= 0, got {lr}"
        )));
    }
    if !(b1.is_finite() && (0.0..1.0).contains(&b1) && (b1 as f32) < 1.0 && ((1.0 - b1) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: beta1 must be finite in [0, 1) with f32(beta1) < 1, got {b1}"
        )));
    }
    if !(b2.is_finite() && (0.0..1.0).contains(&b2) && (b2 as f32) < 1.0 && ((1.0 - b2) as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: beta2 must be finite in [0, 1) with f32(beta2) < 1, got {b2}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: eps must be finite and > 0 in f32, got {eps}"
        )));
    }
    let d_eps = d * eps;
    if !(d_eps.is_finite() && d_eps > 0.0 && (d_eps as f32) > 0.0) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: d * eps ({d_eps:e}) must remain positive and finite in f32"
        )));
    }
    if !(wd.is_finite() && wd >= 0.0 && (wd as f32).is_finite() && ((1.0 - d * lr * wd) as f32).is_finite()) {
        return Err(Error::Invalid(format!(
            "prodigy group `{group}`: weight_decay must be finite in f32 and >= 0 with finite (1 - d * lr * wd), got {wd}"
        )));
    }
    Ok(())
}

fn init_state(groups: &[ParamGroup]) -> Result<Vec<Vec<ParamState>>> {
    let _ng = no_grad();
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(_, p)| {
                    let dev = p.device();
                    let shape = p.shape();
                    let x0 = Tensor::zeros_with(shape, DType::F32, dev)?;
                    x0.copy_from(&p.detach())?;
                    Ok(ParamState {
                        exp_avg: Tensor::zeros_with(shape, DType::F32, dev)?,
                        exp_avg_sq: Tensor::zeros_with(shape, DType::F32, dev)?,
                        s: Tensor::zeros_with(shape, DType::F32, dev)?,
                        x0,
                        param_step: 0,
                        beta1_prod: 1.0,
                    })
                })
                .collect()
        })
        .collect()
}

fn prodigy_step_one(p: &Tensor, g: &Tensor, ps: &ParamState, sp: StepParams) -> Result<(f64, f64)> {
    let t = ps.param_step.saturating_add(1);
    let bc = if sp.bias_correction {
        let b1_prod = ps.beta1_prod * sp.b1;
        let bc1 = 1.0 - b1_prod;
        let bc2 = if sp.b2 == 0.0 {
            1.0
        } else {
            let t_ln_b2 = (t as f64) * (-(1.0 - sp.b2)).ln_1p();
            -t_ln_b2.exp_m1()
        };
        bc2.sqrt() / bc1
    } else {
        1.0
    };

    let dlr = sp.d * sp.lr * bc;
    let d_ratio = sp.d / sp.d0;
    let r_scale = d_ratio * dlr;
    let s_scale = d_ratio * if sp.safeguard_warmup { sp.d } else { dlr };
    let d_eps = sp.d * sp.eps;
    let coupled_wd = !sp.decoupled_wd && sp.wd != 0.0;
    let decay = (sp.decoupled_wd && sp.wd != 0.0).then_some(1.0 - sp.d * sp.lr * sp.wd);

    let (m, v, s, x0) = (&ps.exp_avg, &ps.exp_avg_sq, &ps.s, &ps.x0);
    let can_fast = cpu_fast(&[p, m, v, s, x0])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !m.same_storage(g)
        && !v.same_storage(g)
        && !s.same_storage(g)
        && !x0.same_storage(g);

    if can_fast {
        let mut dot_r = 0.0f64;
        let mut l1_s = 0.0f64;
        g.with_f32_slice(|gs| -> Result<()> {
            x0.with_f32_slice(|x0s| -> Result<()> {
                m.update_f32(|ms| {
                    v.update_f32(|vs| {
                        s.update_f32(|ss| {
                            p.update_f32(|ps_slice| {
                                for (i, pi) in ps_slice.iter_mut().enumerate() {
                                    let p_val = *pi as f64;
                                    let g_raw = gs[i] as f64;
                                    let g_eff = if coupled_wd { g_raw + sp.wd * p_val } else { g_raw };
                                    dot_r += g_eff * (x0s[i] as f64 - p_val);

                                    let s_new = sp.b3 * (ss[i] as f64) + s_scale * g_eff;
                                    ss[i] = s_new as f32;
                                    l1_s += (ss[i] as f64).abs();

                                    let dg = sp.d * g_eff;
                                    let m_new = sp.b1 * (ms[i] as f64) + (1.0 - sp.b1) * dg;
                                    let v_new = sp.b2 * (vs[i] as f64) + (1.0 - sp.b2) * (dg * dg);
                                    ms[i] = m_new as f32;
                                    vs[i] = v_new as f32;

                                    let p_decayed =
                                        if let Some(d_fac) = decay { p_val * d_fac } else { p_val };
                                    let denom = (vs[i] as f64).sqrt() + d_eps;
                                    *pi = (p_decayed - dlr * ((ms[i] as f64) / denom)) as f32;
                                }
                            })
                        })
                    })
                })????;
                Ok(())
            })??;
            Ok(())
        })??;
        return Ok((r_scale * dot_r, l1_s));
    }

    // Device-generic / aliased fallback using out-of-place tensor ops under `no_grad`.
    let g_ref = GradRef::as_f32(g)?;
    let g_eff = if coupled_wd {
        g_ref.as_tensor().add(&p.mul_scalar(sp.wd as f32)?)?
    } else {
        g_ref.as_tensor().clone()
    };
    let dot_vec = g_eff.mul(&x0.sub(p)?)?.to_vec::<f32>()?;
    let dot_r: f64 = dot_vec.iter().map(|&v| v as f64).sum();

    let s_new = s.mul_scalar(sp.b3 as f32)?.add(&g_eff.mul_scalar(s_scale as f32)?)?;
    let l1_s: f64 = s_new.to_vec::<f32>()?.iter().map(|&v| (v as f64).abs()).sum();

    let dg = g_eff.mul_scalar(sp.d as f32)?;
    let m_new = m.mul_scalar(sp.b1 as f32)?.add(&dg.mul_scalar((1.0 - sp.b1) as f32)?)?;
    let v_new = v.mul_scalar(sp.b2 as f32)?.add(&dg.square()?.mul_scalar((1.0 - sp.b2) as f32)?)?;
    let p_base = if let Some(d_fac) = decay { p.mul_scalar(d_fac as f32)? } else { p.clone() };
    let denom = v_new.sqrt()?.add_scalar(d_eps as f32)?;
    let p_new = p_base.sub(&m_new.div(&denom)?.mul_scalar(dlr as f32)?)?;

    p.copy_from(&p_new)?;
    m.copy_from(&m_new)?;
    v.copy_from(&v_new)?;
    s.copy_from(&s_new)?;
    Ok((r_scale * dot_r, l1_s))
}
