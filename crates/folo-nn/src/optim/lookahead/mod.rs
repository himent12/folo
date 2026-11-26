//! Lookahead optimizer wrapper (Zhang et al., NeurIPS 2019 / arXiv:1907.08610).
//!
//! Wraps any inner [`Optimizer`] `O` and maintains a shadow copy of slow weights `phi` alongside the
//! inner optimizer's fast weights `theta`. Every `k` inner steps, slow weights interpolate toward the
//! fast weights and synchronize back:
//!
//! ```text
//! phi   <- phi + alpha * (theta - phi)
//! theta <- phi
//! ```
//!
//! Includes zero-allocation steady-state CPU step loops, [`Lookahead::sync_slow_to_fast`],
//! [`Lookahead::sync_fast_to_slow`], and panic/error-safe [`Lookahead::with_slow_weights`] for evaluating
//! or checkpointing slow weights without losing the inner optimizer's fast exploration state.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::{GradSource, OptimContract, OptimKind, Optimizer, ParamGroup, cpu_fast, storage_key, take_state};
use crate::{Error, Result};

/// Default number of fast inner steps `k` per slow-weight synchronization (`5`).
pub const LOOKAHEAD_DEFAULT_K: u64 = 5;
/// Default slow-weight interpolation factor `alpha` (`0.5`).
pub const LOOKAHEAD_DEFAULT_ALPHA: f64 = 0.5;

/// Hyper-parameters for [`Lookahead`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LookaheadConfig {
    /// Number of fast inner optimizer steps per slow-weight interpolation (`>= 1`, default `5`).
    pub k: u64,
    /// Slow-weight interpolation factor `alpha` in `(0, 1]` (must remain positive in `f32`; default `0.5`).
    pub alpha: f64,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self { k: LOOKAHEAD_DEFAULT_K, alpha: LOOKAHEAD_DEFAULT_ALPHA }
    }
}

impl LookaheadConfig {
    /// Validate Lookahead hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if self.k == 0 {
            return Err(Error::Invalid("lookahead: k must be >= 1, got 0".into()));
        }
        if !(self.alpha.is_finite()
            && self.alpha > 0.0
            && self.alpha <= 1.0
            && (self.alpha as f32).is_finite()
            && (self.alpha as f32) > 0.0)
        {
            return Err(Error::Invalid(format!(
                "lookahead: alpha must be finite in (0, 1] and positive in f32, got {}",
                self.alpha
            )));
        }
        Ok(())
    }
}

/// Serializable Lookahead wrapper contract stored in [`OptimContract::lookahead`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookaheadContract {
    /// Fast inner steps `k >= 1` per slow-weight synchronization.
    pub k: u64,
    /// Slow-weight interpolation factor `alpha` in `(0, 1]` (positive in `f32`).
    pub alpha: f64,
    /// Completed fast steps under this Lookahead wrapper.
    pub fast_steps: u64,
    /// Completed slow-weight synchronizations (`fast_steps / k`).
    pub slow_steps: u64,
}

impl LookaheadContract {
    /// Validate `self` against the outer optimizer's `optim_step` count.
    pub fn validate(&self, optim_step: u64) -> Result<()> {
        LookaheadConfig { k: self.k, alpha: self.alpha }.validate()?;
        if self.fast_steps > optim_step {
            return Err(Error::Invalid(format!(
                "lookahead contract: fast_steps {} exceeds optimizer step {}",
                self.fast_steps, optim_step
            )));
        }
        if self.slow_steps != self.fast_steps / self.k {
            return Err(Error::Invalid(format!(
                "lookahead contract: slow_steps {} does not match fast_steps {} / k {}",
                self.slow_steps, self.fast_steps, self.k
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PendingLookahead {
    step: u64,
    cfg: LookaheadConfig,
    fast_steps: u64,
    slow_steps: u64,
}

/// Lookahead optimizer wrapper (`k` steps forward with `O`, 1 interpolation step back toward slow weights).
pub struct Lookahead<O: Optimizer> {
    inner: O,
    cfg: LookaheadConfig,
    /// Detached slow weight tensors aligned 1-to-1 with `inner.groups()[gi].params[pi]`.
    slow_weights: Vec<Vec<Tensor>>,
    /// Number of completed fast steps.
    fast_steps: u64,
    /// Number of completed slow-weight synchronizations (`fast_steps / k`).
    slow_steps: u64,
    pending_resume: Option<PendingLookahead>,
}

impl<O: Optimizer> Lookahead<O> {
    /// Wrap an inner [`Optimizer`] `inner` with Lookahead slow weights initialized from `inner`'s current
    /// parameters.
    pub fn new(inner: O, cfg: LookaheadConfig) -> Result<Self> {
        cfg.validate()?;
        if inner.contract().lookahead.is_some() {
            return Err(Error::Invalid(
                "lookahead: cannot wrap an optimizer that is already wrapped in Lookahead".into(),
            ));
        }
        let slow_weights = clone_params_detached(inner.groups())?;
        Ok(Self { inner, cfg, slow_weights, fast_steps: 0, slow_steps: 0, pending_resume: None })
    }

    /// Wrap `inner` with [`LookaheadConfig::default`] (`k = 5, alpha = 0.5`).
    pub fn with_defaults(inner: O) -> Result<Self> {
        Self::new(inner, LookaheadConfig::default())
    }

    /// Configured `(k, alpha)`.
    pub fn config(&self) -> LookaheadConfig {
        self.cfg
    }

    /// Borrow the wrapped inner optimizer.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Mutably borrow the wrapped inner optimizer.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// Unwrap and return the inner optimizer (leaving fast weights as-is; call [`Self::sync_slow_to_fast`]
    /// first if you want the slow weights installed in the model).
    pub fn into_inner(self) -> O {
        self.inner
    }

    /// Number of completed fast inner steps.
    pub fn fast_step_count(&self) -> u64 {
        self.fast_steps
    }

    /// Number of completed slow-weight interpolation steps.
    pub fn slow_step_count(&self) -> u64 {
        self.slow_steps
    }

    /// Slow-weight tensor for `(group, param)`.
    pub fn slow_weight(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.slow_weights.get(group)?.get(param)
    }

    /// Copy slow weights `phi` into fast parameters `theta` in place (zero per-element heap allocations).
    pub fn sync_slow_to_fast(&self) -> Result<()> {
        self.check_alignment("sync_slow_to_fast")?;
        let _ng = no_grad();
        for (grp, slows) in self.inner.groups().iter().zip(&self.slow_weights) {
            for ((_, p), phi) in grp.params.iter().zip(slows) {
                copy_tensor_in_place(p, phi)?;
            }
        }
        Ok(())
    }

    /// Copy current fast parameters `theta` into slow weights `phi` in place.
    pub fn sync_fast_to_slow(&mut self) -> Result<()> {
        self.check_alignment("sync_fast_to_slow")?;
        let _ng = no_grad();
        for (grp, slows) in self.inner.groups().iter().zip(&self.slow_weights) {
            for ((_, p), phi) in grp.params.iter().zip(slows) {
                copy_tensor_in_place(phi, p)?;
            }
        }
        Ok(())
    }

    /// Temporarily swap slow weights into the model's parameters, execute `f()`, and unconditionally
    /// swap the fast weights back before returning (even if `f()` returns `Err` or unwinds).
    ///
    /// Performs zero per-element heap allocations on the CPU fast path.
    pub fn with_slow_weights<R>(&mut self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.check_alignment("with_slow_weights")?;
        swap_all(self.inner.groups(), &self.slow_weights)?;
        struct RestoreGuard<'a> {
            groups: &'a [ParamGroup],
            slows: &'a [Vec<Tensor>],
            armed: bool,
        }
        impl Drop for RestoreGuard<'_> {
            fn drop(&mut self) {
                if self.armed {
                    let _ = swap_all(self.groups, self.slows);
                }
            }
        }
        let mut guard = RestoreGuard { groups: self.inner.groups(), slows: &self.slow_weights, armed: true };
        let res = f();
        guard.armed = false;
        swap_all(self.inner.groups(), &self.slow_weights)?;
        res
    }

    fn check_alignment(&self, op: &str) -> Result<()> {
        let groups = self.inner.groups();
        if groups.len() != self.slow_weights.len() {
            return Err(Error::Invalid(format!(
                "lookahead.{op}: group count {} does not match slow_weights {}",
                groups.len(),
                self.slow_weights.len()
            )));
        }
        let mut seen_params = HashSet::new();
        for (grp, slows) in groups.iter().zip(&self.slow_weights) {
            if grp.params.len() != slows.len() {
                return Err(Error::Invalid(format!(
                    "lookahead.{op}: group `{}` param count {} does not match slow_weights {}",
                    grp.name,
                    grp.params.len(),
                    slows.len()
                )));
            }
            for ((name, p), phi) in grp.params.iter().zip(slows) {
                if !seen_params.insert(storage_key(p)) {
                    return Err(Error::Invalid(format!(
                        "lookahead.{op}: parameter `{name}` aliases another parameter's storage"
                    )));
                }
                if p.dtype() != DType::F32
                    || phi.dtype() != DType::F32
                    || p.shape() != phi.shape()
                    || p.device() != phi.device()
                    || !p.layout().is_non_overlapping()
                    || !phi.layout().is_non_overlapping()
                {
                    return Err(Error::Invalid(format!(
                        "lookahead.{op}: parameter `{name}` ({:?} {:?} {}) does not match slow weight ({:?} {:?} {})",
                        p.dims(),
                        p.device(),
                        p.dtype(),
                        phi.dims(),
                        phi.device(),
                        phi.dtype()
                    )));
                }
            }
        }
        for (grp, slows) in groups.iter().zip(&self.slow_weights) {
            for ((name, _), phi) in grp.params.iter().zip(slows) {
                if seen_params.contains(&storage_key(phi)) {
                    return Err(Error::Invalid(format!(
                        "lookahead.{op}: slow weight for `{name}` aliases a parameter's storage"
                    )));
                }
            }
        }
        Ok(())
    }

    fn interpolate_slow_and_sync(&mut self) -> Result<()> {
        let _ng = no_grad();
        let alpha = self.cfg.alpha as f32;
        for (grp, slows) in self.inner.groups().iter().zip(&self.slow_weights) {
            for ((_, p), phi) in grp.params.iter().zip(slows) {
                interpolate_pair(p, phi, alpha)?;
            }
        }
        Ok(())
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn kind(&self) -> OptimKind {
        self.inner.kind()
    }

    fn groups(&self) -> &[ParamGroup] {
        self.inner.groups()
    }

    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        self.inner.groups_mut()
    }

    fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.check_alignment("step")?;
        self.inner.step(grads)?;
        self.pending_resume = None;
        self.fast_steps = self.fast_steps.saturating_add(1);
        if self.fast_steps.is_multiple_of(self.cfg.k) {
            self.interpolate_slow_and_sync()?;
            self.slow_steps = self.slow_steps.saturating_add(1);
        }
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        self.check_alignment("state_tensors")?;
        let mut out = self.inner.state_tensors()?;
        for (grp, slows) in self.inner.groups().iter().zip(&self.slow_weights) {
            for ((name, _), phi) in grp.params.iter().zip(slows) {
                out.push((format!("{name}.lookahead_slow"), phi.clone()));
            }
        }
        Ok(out)
    }

    fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        self.check_alignment("load_state").map_err(|e| Error::Load(e.to_string()))?;
        if let Some(pending) = &self.pending_resume
            && pending.step != step
        {
            return Err(Error::Load(format!(
                "Lookahead state: step {step} does not match staged contract step {}",
                pending.step
            )));
        }
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.inner.groups().iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                if let Some(t) = take_state(state, name, p, "lookahead_slow", step > 0, &mut problems) {
                    loaded.push((gi, pi, t));
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("Lookahead state: {}", problems.join("; "))));
        }
        self.inner.load_state(step, state)?;
        let fresh = clone_params_detached(self.inner.groups())?;
        let _ng = no_grad();
        for (gi, pi, t) in loaded {
            let dst = &fresh[gi][pi];
            dst.copy_from(&t.to_device(dst.device())?)?;
        }
        self.slow_weights = fresh;
        if let Some(pending) = self.pending_resume.take() {
            self.cfg = pending.cfg;
            self.fast_steps = pending.fast_steps;
            self.slow_steps = pending.slow_steps;
        } else {
            self.fast_steps = step;
            self.slow_steps = step / self.cfg.k;
        }
        Ok(())
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        self.inner.to_device(device)?;
        for slows in &self.slow_weights {
            for phi in slows {
                phi.to_device_(device)?;
            }
        }
        Ok(())
    }

    fn contract(&self) -> OptimContract {
        let mut c = self.inner.contract();
        c.lookahead = Some(LookaheadContract {
            k: self.cfg.k,
            alpha: self.cfg.alpha,
            fast_steps: self.fast_steps,
            slow_steps: self.slow_steps,
        });
        c
    }

    fn apply_contract(&mut self, c: &OptimContract) -> Result<()> {
        self.check_alignment("apply_contract").map_err(|e| Error::Load(e.to_string()))?;
        let lc = c.lookahead.ok_or_else(|| {
            Error::Load("folo.optim: Lookahead optimizer requires a `lookahead` contract".into())
        })?;
        lc.validate(c.step).map_err(|e| Error::Load(e.to_string()))?;
        let mut inner_c = c.clone();
        inner_c.lookahead = None;
        self.inner.apply_contract(&inner_c)?;
        self.pending_resume = Some(PendingLookahead {
            step: c.step,
            cfg: LookaheadConfig { k: lc.k, alpha: lc.alpha },
            fast_steps: lc.fast_steps,
            slow_steps: lc.slow_steps,
        });
        Ok(())
    }
}

fn clone_params_detached(groups: &[ParamGroup]) -> Result<Vec<Vec<Tensor>>> {
    let _ng = no_grad();
    let mut seen_params = HashSet::new();
    groups
        .iter()
        .map(|g| {
            g.params
                .iter()
                .map(|(name, p)| {
                    if !seen_params.insert(storage_key(p)) {
                        return Err(Error::Invalid(format!(
                            "lookahead: parameter `{name}` aliases another parameter's storage"
                        )));
                    }
                    if p.dtype() != DType::F32 {
                        return Err(Error::Invalid(format!(
                            "lookahead: parameter `{name}` is {}, expected f32",
                            p.dtype()
                        )));
                    }
                    if !p.layout().is_non_overlapping() {
                        return Err(Error::Invalid(format!(
                            "lookahead: parameter `{name}` has an overlapping/broadcast layout"
                        )));
                    }
                    let phi = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
                    phi.copy_from(&p.detach())?;
                    Ok(phi)
                })
                .collect()
        })
        .collect()
}

fn copy_tensor_in_place(dst: &Tensor, src: &Tensor) -> Result<()> {
    if cpu_fast(&[dst, src]) && !dst.same_storage(src) {
        src.with_f32_slice(|s| {
            dst.update_f32(|d| {
                d.copy_from_slice(s);
            })
        })??;
        return Ok(());
    }
    dst.copy_from(&src.detach())?;
    Ok(())
}

fn interpolate_pair(p: &Tensor, phi: &Tensor, alpha: f32) -> Result<()> {
    // At the `alpha = 1` endpoint, `phi <- theta` directly without computing `phi + (theta - phi)` so
    // small fast weights cannot cancel against large slow weights.
    if alpha == 1.0 {
        return copy_tensor_in_place(phi, p);
    }
    if cpu_fast(&[p, phi]) && !p.same_storage(phi) {
        phi.update_f32(|s| {
            p.update_f32(|f| {
                for (fi, si) in f.iter_mut().zip(s.iter_mut()) {
                    let next = *si + alpha * (*fi - *si);
                    *si = next;
                    *fi = next;
                }
            })
        })??;
        return Ok(());
    }
    let diff = p.sub(phi)?;
    phi.add_scaled_assign(&diff, alpha)?;
    p.copy_from(&phi.detach())?;
    Ok(())
}

fn swap_all(groups: &[ParamGroup], slow_weights: &[Vec<Tensor>]) -> Result<()> {
    let _ng = no_grad();
    for (grp, slows) in groups.iter().zip(slow_weights) {
        for ((_, p), phi) in grp.params.iter().zip(slows) {
            if cpu_fast(&[p, phi]) && !p.same_storage(phi) {
                phi.update_f32(|s| {
                    p.update_f32(|f| {
                        f.swap_with_slice(s);
                    })
                })??;
            } else {
                let tmp = Tensor::zeros_with(p.shape(), DType::F32, p.device())?;
                tmp.copy_from(&p.detach())?;
                p.copy_from(&phi.detach())?;
                phi.copy_from(&tmp)?;
            }
        }
    }
    Ok(())
}
