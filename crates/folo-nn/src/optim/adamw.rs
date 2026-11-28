//! AdamW: Adam with decoupled weight decay.

use std::collections::{BTreeMap, HashMap, HashSet};

use folo_core::backend::OptimStep;
use folo_core::{DType, Tensor};

use super::cautious::{adamw_cautious_slice, cautious_mask_and_scale_tensor};
use super::simd::adamw_slice;
use super::{
    GradSource, OptimContract, OptimKind, Optimizer, ParamGroup, ParamPlan, apply_group_contract, cpu_fast,
    for_each_param_planned, group_contract, step_many_per_param, take_state, validate_groups,
};
use crate::{Error, Result};

/// AdamW, docs/MODEL.md §8.2, with each parameter's own step counter `t` = number of updates that parameter has
/// received + 1 (so parameters frozen during early epochs keep exact bias correction when unfrozen) and the
/// group's current `lr`, `β1` (its `momentum`, which warmup ramps), `β2`, `ε`:
/// ```text
/// p = p · (1 − lr · wd)                                    (if the group's weight_decay > 0: decoupled)
/// m = β1 · m + (1 − β1) · g;   v = β2 · v + (1 − β2) · g²
/// p = p − lr · (m / (1 − β1^t)) / (sqrt(v / (1 − β2^t)) + ε)
/// ```
/// The last line is evaluated as PyTorch does: `p − (lr / (1 − β1^t)) · m / (sqrt(v) / sqrt(1 − β2^t) + ε)`.
pub struct AdamW {
    groups: Vec<ParamGroup>,
    /// `(exp_avg, exp_avg_sq)` per parameter.
    state: Vec<Vec<(Tensor, Tensor)>>,
    /// Completed updates per parameter (for accurate bias correction when parameters are selectively frozen).
    param_steps: Vec<Vec<u64>>,
    /// Pending `(contract_step, param_steps)` overrides validated in [`Optimizer::apply_contract`] for
    /// [`Optimizer::load_state`].
    loaded_param_steps: Option<(u64, BTreeMap<String, u64>)>,
    plan: ParamPlan,
    step: u64,
}

impl AdamW {
    /// An optimizer over `groups` (no parameter in two groups; all `F32`).
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        validate_groups(&groups)?;
        let state = zero_state(&groups)?;
        let param_steps = groups.iter().map(|g| vec![0; g.params.len()]).collect();
        let plan = ParamPlan::from_groups(&groups);
        Ok(AdamW { groups, state, param_steps, loaded_param_steps: None, plan, step: 0 })
    }

    /// `(exp_avg, exp_avg_sq)` of a parameter.
    pub fn moments(&self, group: usize, param: usize) -> Option<(&Tensor, &Tensor)> {
        self.state.get(group)?.get(param).map(|(m, v)| (m, v))
    }

    /// Number of updates applied to parameter `(group, param)`.
    pub fn param_step(&self, group: usize, param: usize) -> Option<u64> {
        self.param_steps.get(group)?.get(param).copied()
    }
}

fn coef_for(g: &ParamGroup, step_before: u64) -> Coef {
    let t = (step_before.saturating_add(1)).min(i32::MAX as u64) as i32;
    let (b1, b2) = (g.momentum, g.beta2);
    Coef {
        decay: (g.weight_decay != 0.0).then_some((1.0 - g.lr * g.weight_decay) as f32),
        b1: b1 as f32,
        one_minus_b1: (1.0 - b1) as f32,
        b2: b2 as f32,
        one_minus_b2: (1.0 - b2) as f32,
        step_size: (g.lr / (1.0 - b1.powi(t))) as f32,
        bc2_sqrt: (1.0 - b2.powi(t)).sqrt() as f32,
        eps: g.eps as f32,
        cautious: g.cautious,
    }
}

fn zero_state(groups: &[ParamGroup]) -> Result<Vec<Vec<(Tensor, Tensor)>>> {
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(_, p)| {
                    Ok((
                        Tensor::zeros_with(p.shape(), DType::F32, p.device())?,
                        Tensor::zeros_with(p.shape(), DType::F32, p.device())?,
                    ))
                })
                .collect()
        })
        .collect()
}

struct Coef {
    decay: Option<f32>,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
    cautious: bool,
}

fn update(p: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, c: &Coef) -> Result<()> {
    if cpu_fast(&[p, m, v])
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
        && !p.same_storage(m)
        && !p.same_storage(v)
        && !m.same_storage(g)
        && !m.same_storage(v)
        && !v.same_storage(g)
    {
        return g.with_f32_slice(|g| -> Result<()> {
            m.update_f32(|m| {
                v.update_f32(|v| {
                    p.update_f32(|p| {
                        if c.cautious {
                            adamw_cautious_slice(
                                p,
                                g,
                                m,
                                v,
                                c.decay,
                                c.b1,
                                c.one_minus_b1,
                                c.b2,
                                c.one_minus_b2,
                                c.step_size,
                                c.bc2_sqrt,
                                c.eps,
                            );
                        } else {
                            adamw_slice(
                                p,
                                g,
                                m,
                                v,
                                c.decay,
                                c.b1,
                                c.one_minus_b1,
                                c.b2,
                                c.one_minus_b2,
                                c.step_size,
                                c.bc2_sqrt,
                                c.eps,
                            );
                        }
                    })
                })
            })???;
            Ok(())
        })?;
    }
    let g = g.to_dtype(DType::F32)?;
    if !c.cautious && !p.device().is_cpu() {
        let step = OptimStep::AdamW {
            decay: c.decay,
            beta1: c.b1,
            one_minus_beta1: c.one_minus_b1,
            beta2: c.b2,
            one_minus_beta2: c.one_minus_b2,
            step_size: c.step_size,
            bc2_sqrt: c.bc2_sqrt,
            eps: c.eps,
        };
        if p.optimizer_step_(&g, &[m, v], &step)? {
            return Ok(());
        }
    }
    if let Some(d) = c.decay {
        p.mul_scalar_assign(d)?;
    }
    m.mul_scalar_assign(c.b1)?;
    m.add_scaled_assign(&g, c.one_minus_b1)?;
    v.mul_scalar_assign(c.b2)?;
    v.add_scaled_assign(&g.square()?, c.one_minus_b2)?;
    let denom = v.sqrt()?.div_scalar(c.bc2_sqrt)?.add_scalar(c.eps)?;
    let u = m.div(&denom)?;
    if c.cautious {
        let (mask, scale) = cautious_mask_and_scale_tensor(&u, &g)?;
        let neg_step = scale.mul_scalar(-c.step_size)?;
        p.add_assign(&u.mul(&mask)?.mul(&neg_step)?)?;
    } else {
        p.add_scaled_assign(&u, -c.step_size)?;
    }
    Ok(())
}

impl Optimizer for AdamW {
    fn kind(&self) -> OptimKind {
        OptimKind::AdamW
    }

    fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.plan.sync(&self.groups);
        // Preflight every active gradient shape before any parameter or moment buffer is mutated.
        for g in &self.groups {
            for (name, p) in &g.params {
                if let Some(grad) = grads.grad(p)
                    && grad.shape() != p.shape()
                {
                    return Err(Error::Invalid(format!(
                        "adamw: gradient of `{name}` is {:?}, parameter is {:?}",
                        grad.dims(),
                        p.dims()
                    )));
                }
            }
        }
        let (groups, state, param_steps, plan) = (&self.groups, &self.state, &self.param_steps, &self.plan);
        let adamw = |gi: usize, pi: usize| {
            let c = coef_for(&groups[gi], param_steps[gi][pi]);
            OptimStep::AdamW {
                decay: c.decay,
                beta1: c.b1,
                one_minus_beta1: c.one_minus_b1,
                beta2: c.b2,
                one_minus_beta2: c.one_minus_b2,
                step_size: c.step_size,
                bc2_sqrt: c.bc2_sqrt,
                eps: c.eps,
            }
        };
        let done = if groups.iter().any(|g| g.cautious) {
            None
        } else {
            step_many_per_param("adamw", groups, grads, adamw, |gi, pi| {
                vec![&state[gi][pi].0, &state[gi][pi].1]
            })?
        };
        if done.as_ref().is_none_or(|d| !d.all_done()) {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                if done.as_ref().is_some_and(|d| d.is_done(gi, pi)) {
                    return Ok(());
                }
                let (_, p) = &groups[gi].params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                let (m, v) = &state[gi][pi];
                let c = coef_for(&groups[gi], param_steps[gi][pi]);
                update(p, g, m, v, &c)
            })?;
        }
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (_, p)) in g.params.iter().enumerate() {
                if grads.grad(p).is_some() {
                    self.param_steps[gi][pi] += 1;
                }
            }
        }
        self.step += 1;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn contract(&self) -> OptimContract {
        let mut c = group_contract(self);
        for (g, steps) in self.groups.iter().zip(&self.param_steps) {
            for ((name, _), &s) in g.params.iter().zip(steps) {
                if s != self.step {
                    c.param_steps.insert(name.clone(), s);
                }
            }
        }
        c
    }

    fn apply_contract(&mut self, c: &OptimContract) -> Result<()> {
        if c.kind == OptimKind::AdamW && !c.param_steps.is_empty() {
            let known: HashSet<&str> =
                self.groups.iter().flat_map(|g| g.params.iter().map(|(n, _)| n.as_str())).collect();
            for (name, &s) in &c.param_steps {
                if !known.contains(name.as_str()) {
                    return Err(Error::Load(format!(
                        "folo.optim: `param_steps` names `{name}`, which is not a parameter of this run"
                    )));
                }
                if s > c.step {
                    return Err(Error::Load(format!(
                        "folo.optim: `param_steps[{name}]` ({s}) exceeds optimizer step ({})",
                        c.step
                    )));
                }
            }
        }
        apply_group_contract(self, c)?;
        self.loaded_param_steps = (!c.param_steps.is_empty()).then(|| (c.step, c.param_steps.clone()));
        Ok(())
    }

    fn reset_param_steps(&mut self, pred: &dyn Fn(&str) -> bool) {
        for (g, steps) in self.groups.iter().zip(&mut self.param_steps) {
            for ((name, _), s) in g.params.iter().zip(steps.iter_mut()) {
                if pred(name) {
                    *s = 0;
                }
            }
        }
    }

    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (g, st) in self.groups.iter().zip(&self.state) {
            for ((name, _), (m, v)) in g.params.iter().zip(st) {
                out.push((format!("{name}.exp_avg"), m.clone()));
                out.push((format!("{name}.exp_avg_sq"), v.clone()));
            }
        }
        Ok(out)
    }

    fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        // Consume any stashed contract overrides up front so a failed load never leaves stale overrides armed.
        let overrides = self.loaded_param_steps.take();
        if let Some((expected_step, ref map)) = overrides {
            if expected_step != step {
                return Err(Error::Load(format!(
                    "AdamW state: step ({step}) disagrees with contract step ({expected_step}) for `param_steps`"
                )));
            }
            for (name, &s) in map {
                if s > step {
                    return Err(Error::Load(format!(
                        "AdamW state: `param_steps[{name}]` ({s}) exceeds step ({step})"
                    )));
                }
            }
        }
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                for (k, what) in ["exp_avg", "exp_avg_sq"].into_iter().enumerate() {
                    if let Some(t) = take_state(state, name, p, what, step > 0, &mut problems) {
                        loaded.push((gi, pi, k, t));
                    }
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("AdamW state: {}", problems.join("; "))));
        }
        let fresh = zero_state(&self.groups)?;
        let _g = folo_core::no_grad();
        for (gi, pi, k, t) in loaded {
            let (m, v) = &fresh[gi][pi];
            let dst = if k == 0 { m } else { v };
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        self.param_steps = self
            .groups
            .iter()
            .map(|g| {
                g.params
                    .iter()
                    .map(|(name, _)| {
                        overrides.as_ref().and_then(|(_, m)| m.get(name).copied()).unwrap_or(step)
                    })
                    .collect()
            })
            .collect();
        self.state = fresh;
        self.step = step;
        Ok(())
    }
}
