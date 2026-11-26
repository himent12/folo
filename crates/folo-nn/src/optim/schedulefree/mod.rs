//! Schedule-Free AdamW (Defazio et al., NeurIPS 2024, "The Road Less Scheduled").
//!
//! Eliminates the need for a learning-rate decay schedule by maintaining an unaveraged primary iterate `z`,
//! an averaged evaluation iterate `x`, and evaluating gradients at their interpolation `y`:
//!
//! ```text
//! y_t = lerp(z_t, x_t, beta1)                            (held in model parameters during train mode)
//! g_t = grad f(y_t)
//! v_{t+1} = beta2 * v_t + (1 - beta2) * g_t^2
//! u_t = g_t / (sqrt(v_{t+1}) / sqrt(1 - beta2^{t+1}) + eps) + wd * y_t
//! z_{t+1} = z_t - gamma_t * u_t
//! lr_max_{t+1} = max(lr_max_t, gamma_t)
//! w_{t+1} = (t + 1)^r * lr_max_{t+1}^{weight_lr_power}
//! c_{t+1} = w_{t+1} / sum_{j=1}^{t+1} w_j
//! x_{t+1} = lerp(x_t, z_{t+1}, c_{t+1})
//! y_{t+1} = lerp(z_{t+1}, x_{t+1}, beta1)
//! ```
//!
//! where `gamma_t = lr * min((t + 1) / warmup_steps, 1)` when `warmup_steps > 0`, else `lr`.
//!
//! Both `z` and `x` are stored explicitly in optimizer state alongside `exp_avg_sq` (`v`), so switching
//! between training weights `y` ([`ScheduleFreeAdamW::train_mode`]) and evaluation weights `x`
//! ([`ScheduleFreeAdamW::eval_mode`]) is exact, idempotent, drift-free, and valid even when `beta1 == 0`.
//! Calling [`ScheduleFreeAdamW::step`] while in eval mode returns [`Error::Invalid`] so gradients are never
//! accidentally applied to `x` instead of `y`.

use std::collections::{HashMap, HashSet};

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{
    GradSource, ParamGroup, ParamPlan, check_param_sets, cpu_fast, for_each_param_planned, storage_key,
    take_state, validate_groups,
};
use crate::{Error, Result};

/// Default Schedule-Free AdamW learning rate (Defazio et al. 2024 recommend ~1-5x AdamW LR).
pub const SF_DEFAULT_LR: f64 = 2.5e-3;
/// Default interpolation parameter `beta1` (Defazio et al. 2024).
pub const SF_DEFAULT_BETA1: f64 = 0.9;
/// Default second-moment EMA decay `beta2`.
pub const SF_DEFAULT_BETA2: f64 = 0.999;
/// Default numerical stability epsilon `eps`.
pub const SF_DEFAULT_EPS: f64 = 1e-8;
/// Default polynomial step weighting exponent `r` (`0.0` in Defazio et al. 2024).
pub const SF_DEFAULT_R: f64 = 0.0;
/// Default LR-weighting exponent `weight_lr_power` (`2.0` in Defazio et al. 2024).
pub const SF_DEFAULT_WEIGHT_LR_POWER: f64 = 2.0;

/// Schedule-Free averaging and warmup hyperparameters shared across parameter groups.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScheduleFreeConfig {
    /// Linear LR warmup length in optimizer steps (`0` = no internal warmup).
    pub warmup_steps: u64,
    /// Polynomial step weighting exponent `r >= 0` in `w_t = t^r * lr_max_t^{weight_lr_power}` (`0.0` default).
    pub r: f64,
    /// Step-size weighting exponent `p >= 0` in `w_t = t^r * lr_max_t^p` (`2.0` default, down-weighting warmup).
    pub weight_lr_power: f64,
}

impl Default for ScheduleFreeConfig {
    fn default() -> Self {
        Self { warmup_steps: 0, r: SF_DEFAULT_R, weight_lr_power: SF_DEFAULT_WEIGHT_LR_POWER }
    }
}

impl ScheduleFreeConfig {
    /// Validate that `r` and `weight_lr_power` are finite and non-negative.
    pub fn validate(&self) -> Result<()> {
        if !(self.r.is_finite() && self.r >= 0.0) {
            return Err(Error::Invalid(format!("schedulefree: r must be finite and >= 0, got {}", self.r)));
        }
        if !(self.weight_lr_power.is_finite() && self.weight_lr_power >= 0.0) {
            return Err(Error::Invalid(format!(
                "schedulefree: weight_lr_power must be finite and >= 0, got {}",
                self.weight_lr_power
            )));
        }
        Ok(())
    }
}

/// Serializable contract for a [`ScheduleFreeAdamW`] parameter group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleFreeGroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate `>= 0`.
    pub lr: f64,
    /// Current effective learning rate `>= 0` (defaults to `lr` if omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_lr: Option<f64>,
    /// Base `[beta1, beta2]`, each in `[0, 1)`.
    pub betas: [f64; 2],
    /// Current effective `beta1` in `[0, 1)` (defaults to `betas[0]` if omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_beta1: Option<f64>,
    /// Numerical stability constant `eps > 0` (must remain positive in `f32`).
    pub eps: f64,
    /// Decoupled weight decay `>= 0`.
    pub weight_decay: f64,
    /// Active steps completed by this group (`<= step`).
    pub group_step: u64,
    /// Accumulated averaging weight sum `W_t = sum_{j=1}^t w_j` in `f64`.
    pub weight_sum: f64,
    /// Running maximum effective learning rate `max_{j <= t} gamma_j` (Defazio et al. 2024).
    #[serde(default)]
    pub lr_max: f64,
}

/// Serializable contract for [`ScheduleFreeAdamW`] checkpointing and bit-exact resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleFreeContract {
    /// Parameter groups, in canonical order.
    pub groups: Vec<ScheduleFreeGroupContract>,
    /// Schedule-Free warmup and averaging exponents.
    pub config: ScheduleFreeConfig,
    /// Total completed [`ScheduleFreeAdamW::step`] calls.
    pub step: u64,
    /// Whether parameters currently hold training weights `y` (`true`) or eval weights `x` (`false`).
    pub train_mode: bool,
}

struct ParamState {
    z: Tensor,
    x: Tensor,
    exp_avg_sq: Tensor,
}

struct Coef {
    gamma: f32,
    wd: f32,
    b1: f32,
    b2_f64: f64,
    one_minus_b2_f64: f64,
    b2: f32,
    sqrt_one_minus_b2: f32,
    bc2_sqrt_f64: f64,
    bc2_sqrt: f32,
    eps_f64: f64,
    eps: f32,
    c: f32,
}

/// Schedule-Free AdamW optimizer (Defazio et al., NeurIPS 2024).
pub struct ScheduleFreeAdamW {
    groups: Vec<ParamGroup>,
    config: ScheduleFreeConfig,
    state: Vec<Vec<ParamState>>,
    param_ids: Vec<Vec<(String, folo_core::TensorId)>>,
    group_steps: Vec<u64>,
    weight_sums: Vec<f64>,
    lr_maxes: Vec<f64>,
    plan: ParamPlan,
    coefs: Vec<Coef>,
    state_keys: HashSet<usize>,
    step: u64,
    train_mode: bool,
    pending_contract_step: Option<u64>,
}

impl ScheduleFreeAdamW {
    /// Construct a [`ScheduleFreeAdamW`] optimizer over `groups` with `config`.
    ///
    /// Starts in train mode with `z_0 = x_0 = param` and `exp_avg_sq = 0`. Rejects duplicate parameter
    /// identities, overlapping parameter storage, broadcast/non-writable views, and invalid hyperparameters.
    pub fn new(groups: Vec<ParamGroup>, config: ScheduleFreeConfig) -> Result<Self> {
        config.validate()?;
        validate_sf_groups(&groups)?;
        let state = init_state(&groups)?;
        let param_ids =
            groups.iter().map(|g| g.params.iter().map(|(n, p)| (n.clone(), p.id())).collect()).collect();
        let n_groups = groups.len();
        let plan = ParamPlan::from_groups(&groups);
        let coefs = Vec::with_capacity(n_groups);
        let state_keys = HashSet::with_capacity(groups.iter().map(|g| g.params.len()).sum::<usize>() * 3);
        Ok(Self {
            groups,
            config,
            state,
            param_ids,
            group_steps: vec![0; n_groups],
            weight_sums: vec![0.0; n_groups],
            lr_maxes: vec![0.0; n_groups],
            plan,
            coefs,
            state_keys,
            step: 0,
            train_mode: true,
            pending_contract_step: None,
        })
    }

    /// Convenience constructor for a single `"params"` group with explicit hyperparameters.
    pub fn from_params(
        params: Vec<(String, Tensor)>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        config: ScheduleFreeConfig,
    ) -> Result<Self> {
        let group = ParamGroup::new("params", params)
            .lr(lr)
            .momentum(beta1)
            .adam(beta2, eps)
            .weight_decay(weight_decay);
        Self::new(vec![group], config)
    }

    /// The parameter groups.
    pub fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    /// Mutable access to parameter groups.
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    /// Schedule-Free warmup and weighting configuration.
    pub fn config(&self) -> ScheduleFreeConfig {
        self.config
    }

    /// Total completed [`ScheduleFreeAdamW::step`] calls.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Active step count for parameter group `group`.
    pub fn group_step(&self, group: usize) -> Option<u64> {
        self.group_steps.get(group).copied()
    }

    /// Accumulated averaging weight sum `W_t` for parameter group `group`.
    pub fn weight_sum(&self, group: usize) -> Option<f64> {
        self.weight_sums.get(group).copied()
    }

    /// Running maximum effective learning rate `lr_max` for parameter group `group`.
    pub fn group_lr_max(&self, group: usize) -> Option<f64> {
        self.lr_maxes.get(group).copied()
    }

    /// Whether parameters currently hold training iterates `y` (`true`) or evaluation iterates `x` (`false`).
    pub fn is_train_mode(&self) -> bool {
        self.train_mode
    }

    /// Per-parameter state `(z, x, exp_avg_sq)`.
    pub fn param_state(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor, &Tensor)> {
        let st = self.state.get(group)?.get(param)?;
        Some((&st.z, &st.x, &st.exp_avg_sq))
    }

    /// Switch parameters in place to the averaged evaluation iterate `x` (`p <- x`).
    ///
    /// Idempotent and drift-free: calling `eval_mode()` repeatedly is a safe no-op.
    pub fn eval_mode(&mut self) -> Result<()> {
        if !self.train_mode {
            return Ok(());
        }
        self.validate_structure()?;
        let _g = no_grad();
        for (grp, st_grp) in self.groups.iter().zip(&self.state) {
            for ((_, p), st) in grp.params.iter().zip(st_grp) {
                p.copy_from(&st.x)?;
            }
        }
        self.train_mode = false;
        Ok(())
    }

    /// Switch parameters in place to the interpolated training iterate `y = lerp(z, x, beta1)`.
    ///
    /// Idempotent and drift-free: calling `train_mode()` repeatedly is a safe no-op.
    pub fn train_mode(&mut self) -> Result<()> {
        if self.train_mode {
            return Ok(());
        }
        validate_sf_step_groups(&self.groups)?;
        self.validate_structure()?;
        let _g = no_grad();
        for (grp, st_grp) in self.groups.iter().zip(&self.state) {
            let b1 = grp.momentum as f32;
            for ((_, p), st) in grp.params.iter().zip(st_grp) {
                write_lerp(p, &st.z, &st.x, b1)?;
            }
        }
        self.train_mode = true;
        Ok(())
    }

    /// Perform one Schedule-Free AdamW step over every parameter that has a gradient in `grads`.
    ///
    /// Requires train mode (`is_train_mode() == true`); validates hyperparameters, writable disjoint layouts,
    /// and all incoming gradients atomically before mutating any parameter or state buffer.
    pub fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        if !self.train_mode {
            return Err(Error::Invalid(
                "schedulefree: optimizer is in eval mode; call train_mode() before step()".into(),
            ));
        }
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("schedulefree: step counter overflow".into()))?;
        self.config.validate()?;
        validate_sf_step_groups(&self.groups)?;
        self.validate_structure()?;

        self.state_keys.clear();
        for (grp, st_grp) in self.groups.iter().zip(&self.state) {
            for ((name, p), st) in grp.params.iter().zip(st_grp) {
                self.state_keys.insert(storage_key(&st.z));
                self.state_keys.insert(storage_key(&st.x));
                self.state_keys.insert(storage_key(&st.exp_avg_sq));
                if let Some(g) = grads.grad(p) {
                    if g.shape() != p.shape() {
                        return Err(Error::Invalid(format!(
                            "schedulefree: gradient of `{name}` is {:?}, parameter is {:?}",
                            g.dims(),
                            p.dims()
                        )));
                    }
                    if g.device() != p.device() {
                        return Err(Error::Invalid(format!(
                            "schedulefree: gradient of `{name}` is on {:?}, parameter is on {:?}",
                            g.device(),
                            p.device()
                        )));
                    }
                    if !g.dtype().is_float() {
                        return Err(Error::Invalid(format!(
                            "schedulefree: gradient of `{name}` has non-float dtype {}",
                            g.dtype()
                        )));
                    }
                }
            }
        }

        let aliases_state = self
            .groups
            .iter()
            .flat_map(|g| &g.params)
            .any(|(_, p)| grads.grad(p).is_some_and(|g| self.state_keys.contains(&storage_key(g))));

        // Compute per-group effective LR, running max LR, averaging weight c_{t+1}, and next counters atomically.
        let mut next_group_steps = self.group_steps.clone();
        let mut next_weight_sums = self.weight_sums.clone();
        let mut next_lr_maxes = self.lr_maxes.clone();
        self.coefs.clear();
        for (gi, grp) in self.groups.iter().enumerate() {
            let has_grad = grp.params.iter().any(|(_, p)| grads.grad(p).is_some());
            if !has_grad {
                self.coefs.push(Coef {
                    gamma: 0.0,
                    wd: 0.0,
                    b1: 0.0,
                    b2_f64: 0.0,
                    one_minus_b2_f64: 1.0,
                    b2: 0.0,
                    sqrt_one_minus_b2: 1.0,
                    bc2_sqrt_f64: 1.0,
                    bc2_sqrt: 1.0,
                    eps_f64: 1.0,
                    eps: 1.0,
                    c: 0.0,
                });
                continue;
            }
            let k_next = self.group_steps[gi].checked_add(1).ok_or_else(|| {
                Error::Invalid(format!("schedulefree group `{}`: group_step overflow", grp.name))
            })?;
            let sched = if self.config.warmup_steps > 0 && k_next < self.config.warmup_steps {
                k_next as f64 / self.config.warmup_steps as f64
            } else {
                1.0
            };
            let gamma = grp.lr * sched;
            let gamma32 = gamma as f32;
            if !gamma32.is_finite() {
                return Err(Error::Invalid(format!(
                    "schedulefree group `{}`: effective lr {gamma} overflows f32",
                    grp.name
                )));
            }
            let lr_max = self.lr_maxes[gi].max(gamma);
            let w_t = (k_next as f64).powf(self.config.r) * lr_max.powf(self.config.weight_lr_power);
            let w_sum = self.weight_sums[gi] + w_t;
            if !w_t.is_finite() || !w_sum.is_finite() {
                return Err(Error::Invalid(format!(
                    "schedulefree group `{}`: averaging weight overflowed at step {k_next}",
                    grp.name
                )));
            }
            let c = if w_sum > 0.0 { (w_t / w_sum).clamp(0.0, 1.0) } else { 0.0 };
            let (b1, b2) = (grp.momentum, grp.beta2);
            let bc2 = if b2 == 0.0 { 1.0 } else { -(k_next as f64 * b2.ln()).exp_m1() };
            let bc2_sqrt_f64 = bc2.sqrt();
            let bc2_sqrt = bc2_sqrt_f64 as f32;
            let sqrt_one_minus_b2 = (1.0 - b2).sqrt() as f32;
            if !(bc2_sqrt.is_finite()
                && bc2_sqrt > 0.0
                && sqrt_one_minus_b2.is_finite()
                && sqrt_one_minus_b2 > 0.0)
            {
                return Err(Error::Invalid(format!(
                    "schedulefree group `{}`: beta2={b2} underflows bias correction in f32",
                    grp.name
                )));
            }
            next_group_steps[gi] = k_next;
            next_weight_sums[gi] = w_sum;
            next_lr_maxes[gi] = lr_max;
            self.coefs.push(Coef {
                gamma: gamma32,
                wd: grp.weight_decay as f32,
                b1: b1 as f32,
                b2_f64: b2,
                one_minus_b2_f64: 1.0 - b2,
                b2: b2 as f32,
                sqrt_one_minus_b2,
                bc2_sqrt_f64,
                bc2_sqrt,
                eps_f64: grp.eps,
                eps: grp.eps as f32,
                c: c as f32,
            });
        }

        self.plan.sync(&self.groups);
        let (groups, state, plan, coefs) = (&self.groups, &self.state, &self.plan, &self.coefs);
        if aliases_state {
            let _g = no_grad();
            for (gi, grp) in groups.iter().enumerate() {
                for (pi, (_, p)) in grp.params.iter().enumerate() {
                    let Some(g) = grads.grad(p) else { continue };
                    update_param(p, g, &state[gi][pi], &coefs[gi])?;
                }
            }
        } else {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                update_param(p, g, &state[gi][pi], &coefs[gi])
            })?;
        }

        self.group_steps = next_group_steps;
        self.weight_sums = next_weight_sums;
        self.lr_maxes = next_lr_maxes;
        self.step = next_step;
        self.pending_contract_step = None;
        Ok(())
    }

    /// Export state tensors as `(<param>.z, z)`, `(<param>.x, x)`, and `(<param>.exp_avg_sq, v)`.
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        self.validate_structure()?;
        let mut out = Vec::new();
        for (grp, st_grp) in self.groups.iter().zip(&self.state) {
            for ((name, _), st) in grp.params.iter().zip(st_grp) {
                out.push((format!("{name}.z"), st.z.clone()));
                out.push((format!("{name}.x"), st.x.clone()));
                out.push((format!("{name}.exp_avg_sq"), st.exp_avg_sq.clone()));
            }
        }
        Ok(out)
    }

    /// Atomically validate and restore both a [`ScheduleFreeContract`] and its state tensors from a
    /// checkpoint, keeping effective hyperparameters, active group counters, `weight_sum`, `lr_max`,
    /// `train_mode`, and `(z, x, exp_avg_sq)` consistent.
    pub fn restore_checkpoint(
        &mut self,
        contract: &ScheduleFreeContract,
        state: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.validate_structure()?;
        validate_contract_against(&self.groups, contract)?;
        let fresh = load_state_tensors(&self.groups, contract.step, state)?;
        apply_validated_contract(
            &mut self.groups,
            &mut self.group_steps,
            &mut self.weight_sums,
            &mut self.lr_maxes,
            &mut self.config,
            &mut self.train_mode,
            contract,
        );
        self.state = fresh;
        self.step = contract.step;
        self.pending_contract_step = None;
        if contract.step == 0 {
            self.group_steps.fill(0);
            self.weight_sums.fill(0.0);
            self.lr_maxes.fill(0.0);
        }
        Ok(())
    }

    /// Restore optimizer state tensors and step count from a checkpoint map.
    ///
    /// When `step == 0`, missing state tensors initialize to `z = x = param`, `exp_avg_sq = 0` and group
    /// counters reset to zero. When `step > 0`, requires a preceding [`ScheduleFreeAdamW::apply_contract`]
    /// with matching `contract.step == step` (or use [`ScheduleFreeAdamW::restore_checkpoint`]) and all three
    /// tensors per parameter.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        self.validate_structure()?;
        if step > 0 {
            if self.pending_contract_step != Some(step) {
                return Err(Error::Load(format!(
                    "ScheduleFreeAdamW state: apply_contract() with matching step {step} must precede load_state(step > 0)"
                )));
            }
            if self.group_steps.iter().any(|&k| k > step) {
                return Err(Error::Load(format!(
                    "ScheduleFreeAdamW state: group_step exceeds checkpoint step {step}"
                )));
            }
        }
        let fresh = load_state_tensors(&self.groups, step, state)?;
        self.state = fresh;
        self.step = step;
        self.pending_contract_step = None;
        if step == 0 {
            self.group_steps.fill(0);
            self.weight_sums.fill(0.0);
            self.lr_maxes.fill(0.0);
        }
        Ok(())
    }

    /// Move all parameters and state tensors (`z`, `x`, `exp_avg_sq`) to `device` in place.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.validate_structure()?;
        for (grp, st_grp) in self.groups.iter().zip(&self.state) {
            for ((_, p), st) in grp.params.iter().zip(st_grp) {
                p.to_device_(device)?;
                st.z.to_device_(device)?;
                st.x.to_device_(device)?;
                st.exp_avg_sq.to_device_(device)?;
            }
        }
        Ok(())
    }

    /// Build a serializable [`ScheduleFreeContract`] capturing base & current hyperparameters, group counters,
    /// `weight_sum`, `lr_max`, `config`, `step`, and `train_mode`.
    pub fn contract(&self) -> ScheduleFreeContract {
        ScheduleFreeContract {
            groups: self
                .groups
                .iter()
                .enumerate()
                .map(|(gi, g)| ScheduleFreeGroupContract {
                    name: g.name.clone(),
                    params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                    lr: g.base_lr,
                    current_lr: (g.lr != g.base_lr).then_some(g.lr),
                    betas: [g.base_momentum, g.beta2],
                    current_beta1: (g.momentum != g.base_momentum).then_some(g.momentum),
                    eps: g.eps,
                    weight_decay: g.weight_decay,
                    group_step: self.group_steps.get(gi).copied().unwrap_or(0),
                    weight_sum: self.weight_sums.get(gi).copied().unwrap_or(0.0),
                    lr_max: self.lr_maxes.get(gi).copied().unwrap_or(0.0),
                })
                .collect(),
            config: self.config,
            step: self.step,
            train_mode: self.train_mode,
        }
    }

    /// Validate that `c` matches this optimizer's groups and parameter sets, then adopt its base & current
    /// hyperparameters, group counters, `weight_sum`, `lr_max`, `config`, and `train_mode`.
    pub fn apply_contract(&mut self, c: &ScheduleFreeContract) -> Result<()> {
        self.validate_structure()?;
        validate_contract_against(&self.groups, c)?;
        apply_validated_contract(
            &mut self.groups,
            &mut self.group_steps,
            &mut self.weight_sums,
            &mut self.lr_maxes,
            &mut self.config,
            &mut self.train_mode,
            c,
        );
        self.pending_contract_step = Some(c.step);
        Ok(())
    }

    fn validate_structure(&self) -> Result<()> {
        if self.groups.len() != self.state.len() || self.groups.len() != self.param_ids.len() {
            return Err(Error::Invalid(format!(
                "schedulefree: group count {} does not match allocated state {}",
                self.groups.len(),
                self.state.len()
            )));
        }
        let mut seen_ids = HashSet::new();
        let mut seen_storage = HashSet::new();
        for ((grp, st_grp), id_grp) in self.groups.iter().zip(&self.state).zip(&self.param_ids) {
            if grp.params.len() != st_grp.len() || grp.params.len() != id_grp.len() {
                return Err(Error::Invalid(format!(
                    "schedulefree: group `{}` parameter count {} does not match state {}",
                    grp.name,
                    grp.params.len(),
                    st_grp.len()
                )));
            }
            for (((name, p), st), (exp_name, exp_id)) in grp.params.iter().zip(st_grp).zip(id_grp) {
                if name != exp_name || p.id() != *exp_id {
                    return Err(Error::Invalid(format!(
                        "schedulefree: group `{}` parameter `{name}` does not match original parameter `{exp_name}` bound to this state slot",
                        grp.name
                    )));
                }
                validate_param_tensor(name, p, &mut seen_ids, &mut seen_storage)?;
                for (label, buf) in [("z", &st.z), ("x", &st.x), ("exp_avg_sq", &st.exp_avg_sq)] {
                    if p.shape() != buf.shape() || p.device() != buf.device() || buf.dtype() != DType::F32 {
                        return Err(Error::Invalid(format!(
                            "schedulefree: parameter `{name}` ({:?} on {:?}) does not match `{label}` ({:?} {} on {:?})",
                            p.dims(),
                            p.device(),
                            buf.dims(),
                            buf.dtype(),
                            buf.device()
                        )));
                    }
                    if !buf.layout().is_non_overlapping() {
                        return Err(Error::Invalid(format!(
                            "schedulefree: state `{name}.{label}` is not writable (overlapping/broadcast view)"
                        )));
                    }
                    if !seen_storage.insert(storage_key(buf)) {
                        return Err(Error::Invalid(format!(
                            "schedulefree: state `{name}.{label}` aliases another parameter or state buffer"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_param_tensor(
    name: &str,
    p: &Tensor,
    seen_ids: &mut HashSet<folo_core::TensorId>,
    seen_storage: &mut HashSet<usize>,
) -> Result<()> {
    if p.dtype() != DType::F32 {
        return Err(Error::Invalid(format!(
            "schedulefree: parameter `{name}` is {}, expected f32",
            p.dtype()
        )));
    }
    if !p.layout().is_non_overlapping() {
        return Err(Error::Invalid(format!(
            "schedulefree: parameter `{name}` is a broadcast/overlapping view and cannot be updated in place"
        )));
    }
    if !seen_ids.insert(p.id()) || !seen_storage.insert(storage_key(p)) {
        return Err(Error::Invalid(format!(
            "schedulefree: parameter `{name}` aliases another parameter's storage"
        )));
    }
    Ok(())
}

fn validate_contract_against(groups: &[ParamGroup], c: &ScheduleFreeContract) -> Result<()> {
    c.config.validate().map_err(|e| Error::Load(e.to_string()))?;
    let compat: Vec<super::GroupContract> = c
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
            decoupled_wd: false,
            cautious_wd: false,
            cautious: false,
        })
        .collect();
    check_param_sets(groups, &compat)?;
    for gc in &c.groups {
        let [b1, b2] = gc.betas;
        let cur_lr = gc.current_lr.unwrap_or(gc.lr);
        let cur_b1 = gc.current_beta1.unwrap_or(b1);
        if !valid_sf_hparams(gc.lr, b1, b2, gc.eps, gc.weight_decay)
            || !valid_sf_hparams(cur_lr, cur_b1, b2, gc.eps, gc.weight_decay)
            || !(gc.weight_sum.is_finite() && gc.weight_sum >= 0.0)
            || !(gc.lr_max.is_finite() && gc.lr_max >= 0.0)
            || (gc.group_step == 0 && (gc.weight_sum != 0.0 || gc.lr_max != 0.0))
            || gc.group_step > c.step
        {
            return Err(Error::Load(format!(
                "schedulefree contract group `{}`: invalid hyperparameters, group_step, weight_sum, or lr_max",
                gc.name
            )));
        }
    }
    Ok(())
}

fn apply_validated_contract(
    groups: &mut [ParamGroup],
    group_steps: &mut [u64],
    weight_sums: &mut [f64],
    lr_maxes: &mut [f64],
    config: &mut ScheduleFreeConfig,
    train_mode: &mut bool,
    c: &ScheduleFreeContract,
) {
    for (gi, (g, gc)) in groups.iter_mut().zip(&c.groups).enumerate() {
        let [b1, b2] = gc.betas;
        g.base_lr = gc.lr;
        g.lr = gc.current_lr.unwrap_or(gc.lr);
        g.base_momentum = b1;
        g.momentum = gc.current_beta1.unwrap_or(b1);
        g.beta2 = b2;
        g.eps = gc.eps;
        g.weight_decay = gc.weight_decay;
        group_steps[gi] = gc.group_step;
        weight_sums[gi] = gc.weight_sum;
        lr_maxes[gi] = gc.lr_max;
    }
    *config = c.config;
    *train_mode = c.train_mode;
}

fn load_state_tensors(
    groups: &[ParamGroup],
    step: u64,
    state: &HashMap<String, Tensor>,
) -> Result<Vec<Vec<ParamState>>> {
    let mut problems = Vec::new();
    let mut loaded = Vec::new();
    for (gi, grp) in groups.iter().enumerate() {
        for (pi, (name, p)) in grp.params.iter().enumerate() {
            let z = take_state(state, name, p, "z", step > 0, &mut problems);
            let x = take_state(state, name, p, "x", step > 0, &mut problems);
            let v = take_state(state, name, p, "exp_avg_sq", step > 0, &mut problems);
            loaded.push((gi, pi, z, x, v));
        }
    }
    if !problems.is_empty() {
        return Err(Error::Load(format!("ScheduleFreeAdamW state: {}", problems.join("; "))));
    }
    let fresh = init_state(groups)?;
    let _g = no_grad();
    for (gi, pi, z, x, v) in loaded {
        let dst = &fresh[gi][pi];
        if let Some(t) = z {
            dst.z.copy_from(&t.to_device(dst.z.device())?)?;
        }
        if let Some(t) = x {
            dst.x.copy_from(&t.to_device(dst.x.device())?)?;
        }
        if let Some(t) = v {
            dst.exp_avg_sq.copy_from(&t.to_device(dst.exp_avg_sq.device())?)?;
        }
    }
    Ok(fresh)
}

#[inline]
fn valid_sf_hparams(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> bool {
    let eps32 = eps as f32;
    let lr32 = lr as f32;
    let wd32 = weight_decay as f32;
    lr.is_finite()
        && lr >= 0.0
        && lr32.is_finite()
        && beta1.is_finite()
        && (0.0..1.0).contains(&beta1)
        && (beta1 as f32) < 1.0
        && beta2.is_finite()
        && (0.0..1.0).contains(&beta2)
        && (beta2 as f32) < 1.0
        && eps.is_finite()
        && eps > 0.0
        && eps32.is_normal()
        && eps32 > 0.0
        && weight_decay.is_finite()
        && weight_decay >= 0.0
        && wd32.is_finite()
}

fn validate_sf_step_groups(groups: &[ParamGroup]) -> Result<()> {
    for g in groups {
        if g.cautious {
            return Err(Error::Invalid(format!(
                "schedulefree group `{}`: cautious masking is not supported",
                g.name
            )));
        }
        if !valid_sf_hparams(g.lr, g.momentum, g.beta2, g.eps, g.weight_decay) {
            return Err(Error::Invalid(format!(
                "schedulefree group `{}`: invalid hyperparameters (lr={}, beta1={}, beta2={}, eps={}, wd={})",
                g.name, g.lr, g.momentum, g.beta2, g.eps, g.weight_decay
            )));
        }
    }
    Ok(())
}

fn validate_sf_groups(groups: &[ParamGroup]) -> Result<()> {
    validate_groups(groups)?;
    validate_sf_step_groups(groups)?;
    let mut seen_ids = HashSet::new();
    let mut seen_storage = HashSet::new();
    for g in groups {
        for (name, p) in &g.params {
            validate_param_tensor(name, p, &mut seen_ids, &mut seen_storage)?;
        }
    }
    Ok(())
}

fn init_state(groups: &[ParamGroup]) -> Result<Vec<Vec<ParamState>>> {
    let _g = no_grad();
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(_, p)| {
                    let z = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
                    z.copy_from(p)?;
                    let x = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
                    x.copy_from(p)?;
                    let exp_avg_sq = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
                    Ok(ParamState { z, x, exp_avg_sq })
                })
                .collect()
        })
        .collect()
}

/// Fixed-point-preserving, endpoint-exact `lerp(start, end, weight)`:
/// returns exact `start` at `weight == 0`, exact `end` at `weight == 1`, and exact `start` when `start == end`.
#[inline]
fn lerp_f32(start: f32, end: f32, weight: f32) -> f32 {
    if weight < 0.5 { start + weight * (end - start) } else { end - (1.0 - weight) * (end - start) }
}

/// Write `dst <- lerp(start, end, weight)` in place under `no_grad`.
fn write_lerp(dst: &Tensor, start: &Tensor, end: &Tensor, weight: f32) -> Result<()> {
    if cpu_fast(&[dst, start, end]) && !dst.same_storage(start) && !dst.same_storage(end) {
        start.with_f32_slice(|ls| {
            end.with_f32_slice(|rs| {
                dst.update_f32(|ds| {
                    for ((d, &l), &r) in ds.iter_mut().zip(ls).zip(rs) {
                        *d = lerp_f32(l, r, weight);
                    }
                })
            })
        })???;
        return Ok(());
    }
    let val = if weight < 0.5 {
        start.add(&end.sub(start)?.mul_scalar(weight)?)?
    } else {
        end.sub(&end.sub(start)?.mul_scalar(1.0 - weight)?)?
    };
    dst.copy_from(&val)?;
    Ok(())
}

fn schedulefree_slice(p: &mut [f32], g: &[f32], z: &mut [f32], x: &mut [f32], v: &mut [f32], c: &Coef) {
    let inv_bc2 = 1.0f64 / c.bc2_sqrt_f64;
    let has_wd = c.wd != 0.0;
    for ((((pi, &gi), zi), xi), vi) in
        p.iter_mut().zip(g).zip(z.iter_mut()).zip(x.iter_mut()).zip(v.iter_mut())
    {
        let y_prev = *pi;
        // Accumulate `v` and `denom` in `f64` so large gradients (e.g. `1e20` with `beta2 = 0.999`) do not
        // overflow `g * g` to `+Inf` when `(1 - beta2) * g^2` is representable in `f32`.
        let g64 = gi as f64;
        let v_new_f64 = c.b2_f64 * (*vi as f64) + c.one_minus_b2_f64 * (g64 * g64);
        let v_new = v_new_f64 as f32;
        *vi = v_new;
        let denom = v_new_f64.sqrt() * inv_bc2 + c.eps_f64;
        let mut u = (g64 / denom) as f32;
        if has_wd {
            u += c.wd * y_prev;
        }
        let z_new = *zi - c.gamma * u;
        *zi = z_new;
        let x_new = lerp_f32(*xi, z_new, c.c);
        *xi = x_new;
        *pi = lerp_f32(z_new, x_new, c.b1);
    }
}

fn update_param(p: &Tensor, g: &Tensor, st: &ParamState, c: &Coef) -> Result<()> {
    let (z, x, v) = (&st.z, &st.x, &st.exp_avg_sq);
    if cpu_fast(&[p, z, x, v])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !z.same_storage(g)
        && !x.same_storage(g)
        && !v.same_storage(g)
    {
        return g.with_f32_slice(|g| -> Result<()> {
            v.update_f32(|v| {
                z.update_f32(|z| x.update_f32(|x| p.update_f32(|p| schedulefree_slice(p, g, z, x, v, c))))
            })????;
            Ok(())
        })?;
    }
    let mut g_f32 = g.to_dtype(DType::F32)?;
    if p.same_storage(&g_f32) || z.same_storage(&g_f32) || x.same_storage(&g_f32) || v.same_storage(&g_f32) {
        g_f32 = g_f32.mul_scalar(1.0)?;
    }
    let y_prev = if c.wd != 0.0 { Some(p.mul_scalar(1.0)?) } else { None };
    // Pre-scale `g` by `sqrt(1 - beta2)` before squaring so large `g` (e.g. `1e20` at `beta2 = 0.999`)
    // keeps `(sqrt(1 - beta2) * g)^2` finite in `f32`.
    let g_scaled_sq = g_f32.mul_scalar(c.sqrt_one_minus_b2)?.square()?;
    let v_new = v.mul_scalar(c.b2)?.add(&g_scaled_sq)?;
    v.copy_from(&v_new)?;
    let inv_bc2 = 1.0f32 / c.bc2_sqrt;
    let denom = v.sqrt()?.mul_scalar(inv_bc2)?.add_scalar(c.eps)?;
    let mut u = g_f32.div(&denom)?;
    if let Some(yp) = y_prev {
        u = u.add(&yp.mul_scalar(c.wd)?)?;
    }
    let z_new = z.sub(&u.mul_scalar(c.gamma)?)?;
    z.copy_from(&z_new)?;
    let x_new = if c.c < 0.5 {
        x.add(&z_new.sub(x)?.mul_scalar(c.c)?)?
    } else {
        z_new.sub(&z_new.sub(x)?.mul_scalar(1.0 - c.c)?)?
    };
    x.copy_from(&x_new)?;
    let y_new = if c.b1 < 0.5 {
        z_new.add(&x_new.sub(&z_new)?.mul_scalar(c.b1)?)?
    } else {
        x_new.sub(&x_new.sub(&z_new)?.mul_scalar(1.0 - c.b1)?)?
    };
    p.copy_from(&y_new)?;
    Ok(())
}

#[cfg(test)]
mod tests;
