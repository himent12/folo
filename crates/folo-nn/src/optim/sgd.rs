//! SGD with momentum (PyTorch semantics, no dampening), Nesterov, and L2 weight decay.

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor};

use super::cautious::{
    cautious_mask_and_scale_tensor, sgd_coupled_cautious_slice, sgd_decoupled_cautious_slice,
};
use super::simd::{sgd_coupled_slice, sgd_decoupled_slice};
use super::{
    GradSource, OptimKind, Optimizer, ParamGroup, ParamPlan, cpu_fast, for_each_param_planned, step_many,
    take_state, validate_groups,
};
use crate::{Error, Result};

/// SGD, docs/MODEL.md §8.2. Per parameter `p` with gradient `g`, using the group's current `lr` and
/// momentum `μ`:
/// ```text
/// g   = g + wd · p                 (if the group's weight_decay > 0: L2, coupled)
/// buf = μ · buf + g                (buf starts at zero, so the first step gives buf = g)
/// p   = p − lr · (g + μ · buf)     (Nesterov; plain momentum uses p − lr · buf; μ = 0 uses p − lr · g)
/// ```
pub struct Sgd {
    groups: Vec<ParamGroup>,
    bufs: Vec<Vec<Option<Tensor>>>,
    plan: ParamPlan,
    step: u64,
}

impl Sgd {
    /// An optimizer over `groups` (no parameter in two groups; all `F32`).
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        validate_groups(&groups)?;
        let plan = ParamPlan::from_groups(&groups);
        let mut s =
            Sgd { bufs: groups.iter().map(|g| vec![None; g.params.len()]).collect(), groups, plan, step: 0 };
        s.ensure_buffers(|g| g.base_momentum > 0.0)?;
        Ok(s)
    }

    /// Allocate zero momentum buffers for every parameter of the groups selected by `want`.
    fn ensure_buffers(&mut self, want: impl Fn(&ParamGroup) -> bool) -> Result<()> {
        for (g, bufs) in self.groups.iter().zip(&mut self.bufs) {
            if !want(g) {
                continue;
            }
            for ((_, p), b) in g.params.iter().zip(bufs.iter_mut()) {
                if b.is_none() {
                    *b = Some(Tensor::zeros_with(p.shape(), DType::F32, p.device())?);
                }
            }
        }
        Ok(())
    }

    /// The momentum buffer of a parameter, if allocated.
    pub fn momentum_buffer(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.bufs.get(group)?.get(param)?.as_ref()
    }

    /// Step with optional per-group LR scales on the gradient/momentum direction `u` (used by [`super::MuSgd`]
    /// so decoupled/cautious weight decay uses the unscaled `g.lr · g.weight_decay` independently of `w_sgd`).
    pub(crate) fn step_with_lr_scales(
        &mut self,
        grads: &dyn GradSource,
        lr_scales: Option<&[f64]>,
    ) -> Result<()> {
        self.ensure_buffers(|g| g.momentum > 0.0)?;
        self.plan.sync(&self.groups);
        let (groups, bufs, plan) = (&self.groups, &self.bufs, &self.plan);
        let scale_of = |gi: usize| lr_scales.map_or(1.0, |s| s[gi]);
        let sgd = |gi: usize| {
            let g = &groups[gi];
            let eff_lr = (g.lr * scale_of(gi)) as f32;
            let momentum = g.momentum as f32;
            if g.decoupled_wd || g.cautious_wd {
                OptimStep::SgdDecoupled {
                    lr: eff_lr,
                    momentum,
                    wd_step: (g.lr * g.weight_decay) as f32,
                    nesterov: g.nesterov,
                    cautious: g.cautious_wd,
                }
            } else {
                OptimStep::Sgd {
                    lr: eff_lr,
                    momentum,
                    weight_decay: g.weight_decay as f32,
                    nesterov: g.nesterov,
                }
            }
        };
        let state = |gi: usize, pi: usize| -> Vec<&Tensor> {
            bufs[gi][pi].as_ref().filter(|_| groups[gi].momentum as f32 != 0.0).into_iter().collect()
        };
        let done = if groups.iter().any(|g| g.cautious) {
            None
        } else {
            step_many("sgd", groups, grads, sgd, state)?
        };
        if done.as_ref().is_none_or(|d| d.iter().any(|&v| !v)) {
            for_each_param_planned(plan, groups, grads, |gi, pi| {
                if done.as_ref().is_some_and(|d| d[gi]) {
                    return Ok(());
                }
                let grp = &groups[gi];
                let (name, p) = &grp.params[pi];
                let Some(g) = grads.grad(p) else { return Ok(()) };
                if g.shape() != p.shape() {
                    return Err(Error::Invalid(format!(
                        "sgd: gradient of `{name}` is {:?}, parameter is {:?}",
                        g.dims(),
                        p.dims()
                    )));
                }
                let eff_lr = (grp.lr * scale_of(gi)) as f32;
                let mu = grp.momentum as f32;
                if grp.decoupled_wd || grp.cautious_wd {
                    update_decoupled(
                        p,
                        g,
                        bufs[gi][pi].as_ref(),
                        eff_lr,
                        mu,
                        (grp.lr * grp.weight_decay) as f32,
                        grp.nesterov,
                        grp.cautious_wd,
                        grp.cautious,
                    )
                } else {
                    update(
                        p,
                        g,
                        bufs[gi][pi].as_ref(),
                        eff_lr,
                        mu,
                        grp.weight_decay as f32,
                        grp.nesterov,
                        grp.cautious,
                    )
                }
            })?;
        }
        self.step += 1;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn update_decoupled(
    p: &Tensor,
    g: &Tensor,
    buf: Option<&Tensor>,
    lr: f32,
    mu: f32,
    wd_step: f32,
    nesterov: bool,
    cautious_wd: bool,
    cautious_update: bool,
) -> Result<()> {
    if cpu_fast(&[p])
        && buf.is_none_or(|b| cpu_fast(&[b]) && !b.same_storage(p) && !b.same_storage(g))
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
    {
        return g.with_f32_slice(|g| -> Result<()> {
            if cautious_update {
                match buf {
                    Some(b) => b.update_f32(|b| {
                        p.update_f32(|p| {
                            sgd_decoupled_cautious_slice(
                                p,
                                g,
                                Some(b),
                                lr,
                                mu,
                                wd_step,
                                nesterov,
                                cautious_wd,
                            )
                        })
                    })??,
                    None => p.update_f32(|p| {
                        sgd_decoupled_cautious_slice(p, g, None, lr, mu, wd_step, nesterov, cautious_wd)
                    })?,
                }
            } else {
                match buf {
                    Some(b) => b.update_f32(|b| {
                        p.update_f32(|p| {
                            sgd_decoupled_slice(p, g, Some(b), lr, mu, wd_step, nesterov, cautious_wd)
                        })
                    })??,
                    None => p.update_f32(|p| {
                        sgd_decoupled_slice(p, g, None, lr, mu, wd_step, nesterov, cautious_wd)
                    })?,
                }
            }
            Ok(())
        })?;
    }
    let g = g.to_dtype(DType::F32)?;
    if !cautious_update && !p.device().is_cpu() {
        let state: Vec<&Tensor> = buf.filter(|_| mu != 0.0).into_iter().collect();
        let step = OptimStep::SgdDecoupled { lr, momentum: mu, wd_step, nesterov, cautious: cautious_wd };
        if p.optimizer_step_(&g, &state, &step)? {
            return Ok(());
        }
    }
    let u = if let Some(b) = buf
        && mu != 0.0
    {
        b.mul_scalar_assign(mu)?;
        b.add_assign(&g)?;
        if nesterov { g.add(&b.mul_scalar(mu)?)? } else { b.clone() }
    } else {
        g.clone()
    };
    let (u_eff, scale_t) = if cautious_update {
        let (mask, scale) = cautious_mask_and_scale_tensor(&u, &g)?;
        (u.mul(&mask)?, Some(scale))
    } else {
        (u, None)
    };
    let decay = if wd_step != 0.0 {
        if cautious_wd {
            let cond = u_eff.mul(p)?.gt_scalar(0.0)?;
            let zero = p.zeros_like()?;
            Some(cond.where_cond(&p.mul_scalar(wd_step)?, &zero)?)
        } else {
            Some(p.mul_scalar(wd_step)?)
        }
    } else {
        None
    };
    if let Some(scale) = scale_t {
        let neg_lr = scale.mul_scalar(-lr)?;
        p.add_assign(&u_eff.mul(&neg_lr)?)?;
    } else {
        p.add_scaled_assign(&u_eff, -lr)?;
    }
    if let Some(decay) = decay {
        p.add_scaled_assign(&decay, -1.0)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update(
    p: &Tensor,
    g: &Tensor,
    buf: Option<&Tensor>,
    lr: f32,
    mu: f32,
    wd: f32,
    nesterov: bool,
    cautious_update: bool,
) -> Result<()> {
    if cpu_fast(&[p])
        && buf.is_none_or(|b| cpu_fast(&[b]) && !b.same_storage(p) && !b.same_storage(g))
        && g.device().is_cpu()
        && g.dtype() == DType::F32
        && !p.same_storage(g)
    {
        return g.with_f32_slice(|g| -> Result<()> {
            if cautious_update {
                match buf {
                    Some(b) => b.update_f32(|b| {
                        p.update_f32(|p| sgd_coupled_cautious_slice(p, g, Some(b), lr, mu, wd, nesterov))
                    })??,
                    None => p.update_f32(|p| sgd_coupled_cautious_slice(p, g, None, lr, mu, wd, nesterov))?,
                }
            } else {
                match buf {
                    Some(b) => b.update_f32(|b| {
                        p.update_f32(|p| sgd_coupled_slice(p, g, Some(b), lr, mu, wd, nesterov))
                    })??,
                    None => p.update_f32(|p| sgd_coupled_slice(p, g, None, lr, mu, wd, nesterov))?,
                }
            }
            Ok(())
        })?;
    }
    let g = g.to_dtype(DType::F32)?;
    // A device with a fused kernel: one dispatch, same arithmetic.
    if !cautious_update && !p.device().is_cpu() {
        let state: Vec<&Tensor> = buf.filter(|_| mu != 0.0).into_iter().collect();
        let step = OptimStep::Sgd { lr, momentum: mu, weight_decay: wd, nesterov };
        if p.optimizer_step_(&g, &state, &step)? {
            return Ok(());
        }
    }
    // Any device: the same arithmetic, op by op.
    let d_grad = if wd != 0.0 { g.add(&p.mul_scalar(wd)?)? } else { g };
    let u = if let Some(b) = buf
        && mu != 0.0
    {
        b.mul_scalar_assign(mu)?;
        b.add_assign(&d_grad)?;
        if nesterov { d_grad.add(&b.mul_scalar(mu)?)? } else { b.clone() }
    } else {
        d_grad.clone()
    };
    if cautious_update {
        let (mask, scale) = cautious_mask_and_scale_tensor(&u, &d_grad)?;
        let neg_lr = scale.mul_scalar(-lr)?;
        p.add_assign(&u.mul(&mask)?.mul(&neg_lr)?)?;
    } else {
        p.add_scaled_assign(&u, -lr)?;
    }
    Ok(())
}

impl Optimizer for Sgd {
    fn kind(&self) -> OptimKind {
        OptimKind::Sgd
    }

    fn groups(&self) -> &[ParamGroup] {
        &self.groups
    }

    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }

    fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.step_with_lr_scales(grads, None)
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        for (g, bufs) in self.groups.iter().zip(&self.bufs) {
            let needed = g.base_momentum > 0.0;
            for ((name, p), b) in g.params.iter().zip(bufs) {
                match b {
                    Some(b) => out.push((format!("{name}.momentum_buffer"), b.clone())),
                    // A zero buffer is exactly the state of a parameter that never received a gradient.
                    None if needed => out.push((
                        format!("{name}.momentum_buffer"),
                        Tensor::zeros_with(p.shape(), DType::F32, p.device())?,
                    )),
                    None => {}
                }
            }
        }
        Ok(out)
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        for (g, bufs) in self.groups.iter().zip(&self.bufs) {
            for ((_, p), b) in g.params.iter().zip(bufs) {
                p.to_device_(device)?;
                if let Some(b) = b {
                    b.to_device_(device)?;
                }
            }
        }
        Ok(())
    }

    fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            let required = step > 0 && g.base_momentum > 0.0;
            for (pi, (name, p)) in g.params.iter().enumerate() {
                if let Some(t) = take_state(state, name, p, "momentum_buffer", required, &mut problems) {
                    loaded.push((gi, pi, t));
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("SGD state: {}", problems.join("; "))));
        }
        self.bufs = self.groups.iter().map(|g| vec![None; g.params.len()]).collect();
        self.ensure_buffers(|g| g.base_momentum > 0.0)?;
        let _g = folo_core::no_grad();
        for (gi, pi, t) in loaded {
            let p = &self.groups[gi].params[pi].1;
            let b = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
            b.copy_from(&t.to_device(p.device())?)?;
            self.bufs[gi][pi] = Some(b);
        }
        self.step = step;
        Ok(())
    }
}
