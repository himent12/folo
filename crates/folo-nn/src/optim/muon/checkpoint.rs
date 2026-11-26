//! Checkpoint contract and atomic state restoration for [`Muon`].

use std::collections::HashMap;

use folo_core::{Tensor, no_grad};

use super::super::{GroupContract, check_param_sets, take_state};
use super::{
    Muon, MuonConfig, MuonContract, MuonGroupContract, ParamState, PendingResume, StagedGroup, init_state,
    validate_muon_scalars,
};
use crate::{Error, Result};

impl Muon {
    /// Export all optimizer state tensors keyed `<param_name>.<state>`.
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                match ps {
                    ParamState::Matrix { momentum_buf, .. } | ParamState::AuxSgd { momentum_buf, .. } => {
                        out.push((format!("{name}.momentum_buffer"), momentum_buf.clone()));
                    }
                    ParamState::AuxAdamW { exp_avg, exp_avg_sq, .. } => {
                        out.push((format!("{name}.exp_avg"), exp_avg.clone()));
                        out.push((format!("{name}.exp_avg_sq"), exp_avg_sq.clone()));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Build a serializable [`MuonContract`] for checkpointing.
    pub fn contract(&self) -> MuonContract {
        let param_steps = self.state.iter().flat_map(|g| g.iter().map(ParamState::param_step)).collect();
        MuonContract {
            groups: self
                .groups
                .iter()
                .map(|g| MuonGroupContract {
                    name: g.name.clone(),
                    params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                    lr: g.base_lr,
                    current_lr: (g.lr != g.base_lr).then_some(g.lr),
                    momentum: g.base_momentum,
                    current_momentum: (g.momentum != g.base_momentum).then_some(g.momentum),
                    nesterov: g.nesterov,
                    weight_decay: g.weight_decay,
                })
                .collect(),
            ns_steps: self.cfg.ns_steps,
            ortho: self.cfg.ortho,
            aspect_scale: self.cfg.aspect_scale,
            fallback: self.cfg.fallback,
            aux_lr: self.cfg.aux_lr,
            aux_betas: self.cfg.aux_betas,
            aux_eps: self.cfg.aux_eps,
            aux_weight_decay: self.cfg.aux_weight_decay,
            step: self.step,
            param_steps,
        }
    }

    /// Validate and stage a checkpoint [`MuonContract`] to be committed atomically by [`Muon::load_state`].
    pub fn apply_contract(&mut self, c: &MuonContract) -> Result<()> {
        let group_contracts: Vec<GroupContract> = c
            .groups
            .iter()
            .map(|gc| GroupContract {
                name: gc.name.clone(),
                params: gc.params.clone(),
                lr: gc.lr,
                momentum: Some(gc.momentum),
                nesterov: Some(gc.nesterov),
                weight_decay: gc.weight_decay,
                betas: None,
                eps: None,
                decoupled_wd: true,
                cautious_wd: false,
                cautious: false,
            })
            .collect();
        check_param_sets(&self.groups, &group_contracts)?;
        if c.fallback != self.cfg.fallback {
            return Err(Error::Load(format!(
                "muon contract fallback {:?} does not match optimizer {:?}",
                c.fallback, self.cfg.fallback
            )));
        }
        let total_params: usize = self.groups.iter().map(|g| g.params.len()).sum();
        if !c.param_steps.is_empty() && c.param_steps.len() != total_params {
            return Err(Error::Load(format!(
                "muon contract param_steps length {} != total params {total_params}",
                c.param_steps.len()
            )));
        }
        let new_cfg = MuonConfig {
            lr: c.groups.first().map_or(self.cfg.lr, |g| g.lr),
            momentum: c.groups.first().map_or(self.cfg.momentum, |g| g.momentum),
            nesterov: c.groups.first().map_or(self.cfg.nesterov, |g| g.nesterov),
            weight_decay: c.groups.first().map_or(self.cfg.weight_decay, |g| g.weight_decay),
            ns_steps: c.ns_steps,
            ortho: c.ortho,
            aspect_scale: c.aspect_scale,
            fallback: c.fallback,
            aux_lr: c.aux_lr,
            aux_betas: c.aux_betas,
            aux_eps: c.aux_eps,
            aux_weight_decay: c.aux_weight_decay,
        };
        new_cfg.validate().map_err(|e| Error::Load(e.to_string()))?;

        let mut staged_groups = Vec::with_capacity(c.groups.len());
        let mut staged_steps = Vec::with_capacity(c.groups.len());
        let mut offset = 0usize;
        for (grp, gc) in self.groups.iter().zip(&c.groups) {
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            let cur_mom = gc.current_momentum.unwrap_or(gc.momentum);
            validate_muon_scalars(&gc.name, gc.lr, gc.momentum, gc.weight_decay)
                .map_err(|e| Error::Load(e.to_string()))?;
            validate_muon_scalars(&gc.name, cur_lr, cur_mom, gc.weight_decay)
                .map_err(|e| Error::Load(e.to_string()))?;
            staged_groups.push(StagedGroup {
                base_lr: gc.lr,
                lr: cur_lr,
                base_momentum: gc.momentum,
                momentum: cur_mom,
                nesterov: gc.nesterov,
                weight_decay: gc.weight_decay,
            });
            let grp_steps = if c.param_steps.is_empty() {
                vec![c.step; grp.params.len()]
            } else {
                let mut by_name = HashMap::with_capacity(gc.params.len());
                for pname in &gc.params {
                    let k = c.param_steps[offset];
                    offset += 1;
                    if k > c.step {
                        return Err(Error::Load(format!(
                            "muon contract param `{pname}` step {k} exceeds global step {}",
                            c.step
                        )));
                    }
                    by_name.insert(pname.as_str(), k);
                }
                grp.params.iter().map(|(n, _)| by_name[n.as_str()]).collect()
            };
            staged_steps.push(grp_steps);
        }

        self.pending_resume = Some(PendingResume {
            step: c.step,
            cfg: new_cfg,
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

    /// Atomically restore step count, staged contract settings, and state tensors from a checkpoint map.
    pub fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        if step > 0 && self.pending_resume.is_none() {
            return Err(Error::Load(
                "muon state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "muon state: step {step} does not match staged contract step {}",
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
                    "muon state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }

        let mut problems = Vec::new();
        let mut staged: Vec<Vec<(Option<Tensor>, Option<Tensor>)>> = Vec::with_capacity(self.groups.len());
        let check_vals =
            |t: &Tensor, name: &str, label: &str, non_neg: bool, problems: &mut Vec<String>| -> Result<()> {
                let vals = t.to_vec::<f32>()?;
                if vals.iter().any(|&v| !v.is_finite() || (non_neg && v < 0.0)) {
                    let kind = if non_neg { "negative or non-finite" } else { "non-finite" };
                    problems.push(format!("`optim.{name}.{label}` contains {kind} values"));
                }
                Ok(())
            };

        for (gi, (g, st)) in self.groups.iter().zip(&self.state).enumerate() {
            let mut g_staged = Vec::with_capacity(g.params.len());
            for (pi, ((name, p), ps)) in g.params.iter().zip(st).enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                let required = p_step > 0;
                match ps {
                    ParamState::Matrix { .. } | ParamState::AuxSgd { .. } => {
                        let mb = take_state(state, name, p, "momentum_buffer", required, &mut problems);
                        if let Some(t) = &mb {
                            check_vals(t, name, "momentum_buffer", false, &mut problems)?;
                        }
                        g_staged.push((mb, None));
                    }
                    ParamState::AuxAdamW { .. } => {
                        let m = take_state(state, name, p, "exp_avg", required, &mut problems);
                        let v = take_state(state, name, p, "exp_avg_sq", required, &mut problems);
                        if let Some(t) = &m {
                            check_vals(t, name, "exp_avg", false, &mut problems)?;
                        }
                        if let Some(t) = &v {
                            check_vals(t, name, "exp_avg_sq", true, &mut problems)?;
                        }
                        g_staged.push((m, v));
                    }
                }
            }
            staged.push(g_staged);
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("muon state mismatch: {}", problems.join("; "))));
        }

        let mut fresh = init_state(&self.groups, self.cfg.fallback)?;
        let _ng = no_grad();
        for ((g, st), g_staged) in self.groups.iter().zip(&mut fresh).zip(staged) {
            for (((_, p), ps), (t1, t2)) in g.params.iter().zip(st.iter_mut()).zip(g_staged) {
                let dev = p.device();
                match ps {
                    ParamState::Matrix { momentum_buf, .. } | ParamState::AuxSgd { momentum_buf, .. } => {
                        if let Some(t) = t1 {
                            momentum_buf.copy_from(&t.to_device(dev)?)?;
                        }
                    }
                    ParamState::AuxAdamW { exp_avg, exp_avg_sq, .. } => {
                        if let Some(t) = t1 {
                            exp_avg.copy_from(&t.to_device(dev)?)?;
                        }
                        if let Some(t) = t2 {
                            exp_avg_sq.copy_from(&t.to_device(dev)?)?;
                        }
                    }
                }
            }
        }
        if let Some(pending) = self.pending_resume.take() {
            for (g, sg) in self.groups.iter_mut().zip(pending.groups) {
                g.base_lr = sg.base_lr;
                g.lr = sg.lr;
                g.base_momentum = sg.base_momentum;
                g.momentum = sg.momentum;
                g.nesterov = sg.nesterov;
                g.weight_decay = sg.weight_decay;
            }
            for (st, psteps) in fresh.iter_mut().zip(pending.param_steps) {
                for (ps, k) in st.iter_mut().zip(psteps) {
                    ps.set_param_step(k);
                }
            }
            self.cfg = pending.cfg;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}
