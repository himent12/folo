//! SGD with momentum (PyTorch semantics, no dampening), Nesterov, and L2 weight decay.

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor};

use super::{
    GradSource, OptimKind, Optimizer, ParamGroup, cpu_fast, for_each_param, step_many, take_state,
    validate_groups,
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
    step: u64,
    /// The momentum buffers the last step allocated (warmup momentum on a momentum-free group), dropped again by
    /// [`Optimizer::unstep`] so that a skipped guarded step leaves the state as it was.
    fresh: Vec<(usize, usize)>,
}

impl Sgd {
    /// An optimizer over `groups` (no parameter in two groups; all `F32`).
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        validate_groups(&groups)?;
        let bufs = groups.iter().map(|g| vec![None; g.params.len()]).collect();
        let mut s = Sgd { bufs, groups, step: 0, fresh: Vec::new() };
        s.ensure_buffers(|g| g.base_momentum > 0.0)?;
        s.fresh.clear();
        Ok(s)
    }

    /// Allocate zero momentum buffers for every parameter of the groups selected by `want`.
    fn ensure_buffers(&mut self, want: impl Fn(&ParamGroup) -> bool) -> Result<()> {
        for (gi, (g, bufs)) in self.groups.iter().zip(&mut self.bufs).enumerate() {
            if !want(g) {
                continue;
            }
            for (pi, ((_, p), b)) in g.params.iter().zip(bufs.iter_mut()).enumerate() {
                if b.is_none() {
                    *b = Some(Tensor::zeros_with(p.shape(), DType::F32, p.device())?);
                    self.fresh.push((gi, pi));
                }
            }
        }
        Ok(())
    }

    /// The momentum buffer of a parameter, if allocated.
    pub fn momentum_buffer(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.bufs.get(group)?.get(param)?.as_ref()
    }
}

fn update(
    p: &Tensor,
    g: &Tensor,
    buf: Option<&Tensor>,
    lr: f32,
    mu: f32,
    wd: f32,
    nesterov: bool,
) -> Result<()> {
    if cpu_fast(&[p]) && buf.is_none_or(|b| cpu_fast(&[b])) && g.device().is_cpu() && g.dtype() == DType::F32
    {
        return g.with_f32_slice(|g| -> Result<()> {
            let step = |p: &mut [f32], b: Option<&mut [f32]>| match b {
                Some(b) if mu != 0.0 => {
                    for ((p, &g), b) in p.iter_mut().zip(g).zip(b) {
                        let d = if wd != 0.0 { g + wd * *p } else { g };
                        *b = mu * *b + d;
                        let d = if nesterov { d + mu * *b } else { *b };
                        *p += -lr * d;
                    }
                }
                _ => {
                    for (p, &g) in p.iter_mut().zip(g) {
                        let d = if wd != 0.0 { g + wd * *p } else { g };
                        *p += -lr * d;
                    }
                }
            };
            match buf {
                Some(b) => b.update_f32(|b| p.update_f32(|p| step(p, Some(b))))??,
                None => p.update_f32(|p| step(p, None))?,
            }
            Ok(())
        })?;
    }
    let g = g.to_dtype(DType::F32)?;
    // A device with a fused kernel: one dispatch, same arithmetic.
    if !p.device().is_cpu() {
        let state: Vec<&Tensor> = buf.filter(|_| mu != 0.0).into_iter().collect();
        let step = OptimStep::Sgd { lr, momentum: mu, weight_decay: wd, nesterov };
        if p.optimizer_step_(&g, &state, &step)? {
            return Ok(());
        }
    }
    // Any device: the same arithmetic, op by op (under a step guard, only if it lets the step through).
    if !folo_core::step_guard_allows()? {
        return Ok(());
    }
    let mut d = if wd != 0.0 { g.add(&p.mul_scalar(wd)?)? } else { g };
    if let Some(b) = buf
        && mu != 0.0
    {
        b.mul_scalar_assign(mu)?;
        b.add_assign(&d)?;
        d = if nesterov { d.add(&b.mul_scalar(mu)?)? } else { b.clone() };
    }
    p.add_scaled_assign(&d, -lr)?;
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
        // Warmup can give a momentum-free group a non-zero momentum: give it buffers first.
        self.fresh.clear();
        self.ensure_buffers(|g| g.momentum > 0.0)?;
        let (groups, bufs) = (&self.groups, &self.bufs);
        let sgd = |gi: usize| {
            let g = &groups[gi];
            let (lr, momentum, weight_decay) = (g.lr as f32, g.momentum as f32, g.weight_decay as f32);
            OptimStep::Sgd { lr, momentum, weight_decay, nesterov: g.nesterov }
        };
        let state = |gi: usize, pi: usize| -> Vec<&Tensor> {
            bufs[gi][pi].as_ref().filter(|_| groups[gi].momentum as f32 != 0.0).into_iter().collect()
        };
        let done = step_many("sgd", groups, grads, sgd, state)?;
        for_each_param(groups, |gi, pi| {
            if done[gi] {
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
            let (lr, mu, wd) = (grp.lr as f32, grp.momentum as f32, grp.weight_decay as f32);
            update(p, g, bufs[gi][pi].as_ref(), lr, mu, wd, grp.nesterov)
        })?;
        self.step += 1;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn unstep(&mut self) {
        self.step = self.step.saturating_sub(1);
        for (gi, pi) in self.fresh.drain(..) {
            self.bufs[gi][pi] = None;
        }
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
