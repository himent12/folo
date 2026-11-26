//! Checkpoint contract and atomic state restoration for [`Adafactor`].

use std::collections::HashMap;

use folo_core::{DType, Tensor, no_grad};

use super::super::{GroupContract, check_param_sets};
use super::{
    Adafactor, AdafactorContract, AdafactorGroupContract, PendingResume, SecondMoment, StagedGroup,
    init_state, validate_adafactor_scalars,
};
use crate::{Error, Result};

impl Adafactor {
    /// Export all optimizer state tensors keyed `<param_name>.<state>`.
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), ps) in g.params.iter().zip(st) {
                match &ps.second {
                    SecondMoment::Factored { row, col } => {
                        out.push((format!("{name}.exp_avg_sq_row"), row.clone()));
                        out.push((format!("{name}.exp_avg_sq_col"), col.clone()));
                    }
                    SecondMoment::Unfactored(v) => {
                        out.push((format!("{name}.exp_avg_sq"), v.clone()));
                    }
                }
                if let Some(m) = &ps.exp_avg {
                    out.push((format!("{name}.exp_avg"), m.clone()));
                }
            }
        }
        Ok(out)
    }

    /// Build a serializable [`AdafactorContract`] for checkpointing.
    pub fn contract(&self) -> AdafactorContract {
        let param_steps = self.state.iter().flat_map(|g| g.iter().map(|s| s.param_step)).collect();
        AdafactorContract {
            groups: self
                .groups
                .iter()
                .map(|g| AdafactorGroupContract {
                    name: g.name.clone(),
                    params: g.params.iter().map(|(n, _)| n.clone()).collect(),
                    lr: g.base_lr,
                    current_lr: (g.lr != g.base_lr).then_some(g.lr),
                    weight_decay: g.weight_decay,
                })
                .collect(),
            config: self.cfg,
            step: self.step,
            param_steps,
        }
    }

    /// Validate and stage a checkpoint [`AdafactorContract`] to be committed atomically by
    /// [`Adafactor::load_state`].
    pub fn apply_contract(&mut self, c: &AdafactorContract) -> Result<()> {
        c.config.validate().map_err(|e| Error::Load(e.to_string()))?;
        let group_contracts: Vec<GroupContract> = c
            .groups
            .iter()
            .map(|gc| GroupContract {
                name: gc.name.clone(),
                params: gc.params.clone(),
                lr: gc.lr,
                momentum: None,
                nesterov: None,
                weight_decay: gc.weight_decay,
                betas: None,
                eps: None,
                decoupled_wd: true,
                cautious_wd: false,
                cautious: false,
            })
            .collect();
        check_param_sets(&self.groups, &group_contracts)?;
        if c.config.beta1.is_some() != self.cfg.beta1.is_some() {
            return Err(Error::Load(
                "adafactor contract beta1 presence does not match optimizer state".into(),
            ));
        }
        let total_params: usize = self.groups.iter().map(|g| g.params.len()).sum();
        if !c.param_steps.is_empty() && c.param_steps.len() != total_params {
            return Err(Error::Load(format!(
                "adafactor contract param_steps length {} != total params {total_params}",
                c.param_steps.len()
            )));
        }

        let mut staged_groups = Vec::with_capacity(c.groups.len());
        let mut staged_steps = Vec::with_capacity(c.groups.len());
        let mut offset = 0usize;
        for (grp, gc) in self.groups.iter().zip(&c.groups) {
            let cur_lr = gc.current_lr.unwrap_or(gc.lr);
            validate_adafactor_scalars(&gc.name, gc.lr, gc.weight_decay, c.config.relative_step)
                .map_err(|e| Error::Load(e.to_string()))?;
            validate_adafactor_scalars(&gc.name, cur_lr, gc.weight_decay, c.config.relative_step)
                .map_err(|e| Error::Load(e.to_string()))?;
            staged_groups.push(StagedGroup { base_lr: gc.lr, lr: cur_lr, weight_decay: gc.weight_decay });
            let grp_steps = if c.param_steps.is_empty() {
                vec![c.step; grp.params.len()]
            } else {
                let mut by_name = HashMap::with_capacity(gc.params.len());
                for pname in &gc.params {
                    let k = c.param_steps[offset];
                    offset += 1;
                    if k > c.step {
                        return Err(Error::Load(format!(
                            "adafactor contract param `{pname}` step {k} exceeds global step {}",
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
            cfg: c.config,
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
                "adafactor state: non-zero step resume requires `apply_contract` before `load_state`".into(),
            ));
        }
        if let Some(pending) = &self.pending_resume {
            if pending.step != step {
                return Err(Error::Load(format!(
                    "adafactor state: step {step} does not match staged contract step {}",
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
                    "adafactor state: parameter groups changed after `apply_contract`".into(),
                ));
            }
        }

        let mut problems = Vec::new();
        let mut staged = Vec::with_capacity(self.groups.len());

        let fetch = |key: &str,
                     expected: &[usize],
                     required: bool,
                     non_neg: bool,
                     problems: &mut Vec<String>|
         -> Result<Option<Tensor>> {
            match state.get(key) {
                Some(t) if t.dims() == expected && t.dtype() == DType::F32 => {
                    let vals = t.to_vec::<f32>()?;
                    if vals.iter().any(|&v| !v.is_finite() || (non_neg && v < 0.0)) {
                        let kind = if non_neg { "negative or non-finite" } else { "non-finite" };
                        problems.push(format!("`optim.{key}` contains {kind} values"));
                        return Ok(None);
                    }
                    Ok(Some(t.clone()))
                }
                Some(t) => {
                    problems.push(format!(
                        "`optim.{key}` is {:?} {}, expected {expected:?} f32",
                        t.dims(),
                        t.dtype()
                    ));
                    Ok(None)
                }
                None => {
                    if required {
                        problems.push(format!("`optim.{key}` is missing"));
                    }
                    Ok(None)
                }
            }
        };

        for (gi, (g, st)) in self.groups.iter().zip(&self.state).enumerate() {
            let mut gs = Vec::with_capacity(g.params.len());
            for (pi, ((name, p), ps)) in g.params.iter().zip(st).enumerate() {
                let p_step = self.pending_resume.as_ref().map_or(step, |pr| pr.param_steps[gi][pi]);
                let required = p_step > 0;
                let s_t = match &ps.second {
                    SecondMoment::Factored { row, col } => {
                        let r = fetch(
                            &format!("{name}.exp_avg_sq_row"),
                            row.dims(),
                            required,
                            true,
                            &mut problems,
                        )?;
                        let c = fetch(
                            &format!("{name}.exp_avg_sq_col"),
                            col.dims(),
                            required,
                            true,
                            &mut problems,
                        )?;
                        (r, c)
                    }
                    SecondMoment::Unfactored(_) => {
                        let v =
                            fetch(&format!("{name}.exp_avg_sq"), p.dims(), required, true, &mut problems)?;
                        (v, None)
                    }
                };
                let m_t = if ps.exp_avg.is_some() {
                    fetch(&format!("{name}.exp_avg"), p.dims(), required, false, &mut problems)?
                } else {
                    None
                };
                gs.push((s_t.0, s_t.1, m_t));
            }
            staged.push(gs);
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("adafactor state mismatch: {}", problems.join("; "))));
        }

        let mut fresh = init_state(&self.groups, self.cfg.beta1.is_some())?;
        let _ng = no_grad();
        for ((g, st), gs) in self.groups.iter().zip(&mut fresh).zip(staged) {
            for (((_, p), ps), (t1, t2, tm)) in g.params.iter().zip(st.iter_mut()).zip(gs) {
                let dev = p.device();
                match &mut ps.second {
                    SecondMoment::Factored { row, col } => {
                        if let Some(t) = t1 {
                            row.copy_from(&t.to_device(dev)?)?;
                        }
                        if let Some(t) = t2 {
                            col.copy_from(&t.to_device(dev)?)?;
                        }
                    }
                    SecondMoment::Unfactored(v) => {
                        if let Some(t) = t1 {
                            v.copy_from(&t.to_device(dev)?)?;
                        }
                    }
                }
                if let (Some(m), Some(t)) = (&mut ps.exp_avg, tm) {
                    m.copy_from(&t.to_device(dev)?)?;
                }
            }
        }
        if let Some(pending) = self.pending_resume.take() {
            for (g, sg) in self.groups.iter_mut().zip(pending.groups) {
                g.base_lr = sg.base_lr;
                g.lr = sg.lr;
                g.weight_decay = sg.weight_decay;
            }
            for (st, psteps) in fresh.iter_mut().zip(pending.param_steps) {
                for (ps, k) in st.iter_mut().zip(psteps) {
                    ps.param_step = k;
                }
            }
            self.cfg = pending.cfg;
        }
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}
