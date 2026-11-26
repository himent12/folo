//! Checkpoint contract and state serialization / restoration for [`Prodigy`].

use std::collections::HashMap;

use folo_core::{Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::super::{GroupContract, check_param_sets, take_state};
use super::{Prodigy, init_state, validate_prodigy_global, validate_prodigy_group};
use crate::{Error, Result};

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

/// Serializable contract for [`Prodigy`] checkpointing and resume.
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
    /// Optional `beta3` override (`None` = `sqrt(max_g(beta2_g))`).
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

pub(super) struct StagedGroup {
    base_lr: f64,
    lr: f64,
    base_momentum: f64,
    momentum: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    decoupled_wd: bool,
}

pub(super) struct PendingResume {
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

impl Prodigy {
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
            validate_prodigy_group(
                &gc.name,
                gc.lr,
                gc.betas[0],
                gc.betas[1],
                gc.eps,
                gc.weight_decay,
                c.d,
                c.d0,
            )
            .map_err(|e| Error::Load(e.to_string()))?;
            validate_prodigy_group(&gc.name, cur_lr, cur_b1, gc.betas[1], gc.eps, gc.weight_decay, c.d, c.d0)
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
    /// restored accurately. Calling `load_state(0, &empty_map)` without `apply_contract` resets the
    /// optimizer cleanly to step `0` (`d = d0`, `d_numerator = 0.0`).
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
        } else {
            self.d = self.d0;
            self.d_numerator = 0.0;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}
