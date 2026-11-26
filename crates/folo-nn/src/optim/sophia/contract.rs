//! Checkpoint contract and state serialization / restoration for [`SophiaG`].

use std::collections::HashMap;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::super::{GroupContract, check_param_sets, take_state};
use super::{HutchinsonDist, SophiaG, validate_sophia_global, validate_sophia_group, zero_state};
use crate::{Error, Result};

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

/// Serializable contract for [`SophiaG`] checkpointing and resume.
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
    /// Whether `step()` automatically runs the empirical-Fisher `g^2` heuristic when due.
    pub auto_gauss_newton: bool,
    /// Hutchinson probe distribution.
    pub hutchinson_dist: HutchinsonDist,
    /// Internal xoshiro256++ RNG state words for deterministic Hutchinson probe resume.
    pub rng_state: [u64; 4],
}

pub(super) struct StagedGroup {
    base_lr: f64,
    lr: f64,
    base_momentum: f64,
    momentum: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

pub(super) struct PendingResume {
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

impl SophiaG {
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
        if (c.hessian_updates == 0) != c.last_hessian_step.is_none() {
            return Err(Error::Load(format!(
                "sophia contract: inconsistent hessian_updates ({}) and last_hessian_step ({:?})",
                c.hessian_updates, c.last_hessian_step
            )));
        }
        if let Some(last) = c.last_hessian_step
            && last > c.step
        {
            return Err(Error::Load(format!("sophia contract: last_hessian_step {last} > step {}", c.step)));
        }
        let as_group_contracts: Vec<GroupContract> = c
            .groups
            .iter()
            .map(|gc| GroupContract {
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
    ///
    /// Resuming at `step > 0` requires staging the matching [`SophiaContract`] via
    /// [`SophiaG::apply_contract`] first. Calling `load_state(0, &empty_map)` without `apply_contract`
    /// resets the optimizer cleanly to step `0` (`hessian_updates = 0`, `last_hessian_step = None`).
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
        } else {
            self.hessian_updates = 0;
            self.last_hessian_step = None;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}
