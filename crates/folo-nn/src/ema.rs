//! Exponential moving average of model weights (docs/MODEL.md §8.5).

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use crate::module::{Module, TensorKind};
use crate::{Error, Result};

/// docs/MODEL.md §8.5: `decay = 0.9999`.
pub const EMA_DECAY: f64 = 0.9999;
/// docs/MODEL.md §8.5: `tau = 2000`.
pub const EMA_TAU: f64 = 2000.0;

/// `folo.ema` (docs/FORMAT.md).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmaContract {
    /// Asymptotic decay.
    pub decay: f64,
    /// Ramp time constant, in updates.
    pub tau: f64,
}

/// An EMA copy of a model: every parameter and floating buffer (BatchNorm running statistics) of `M`.
///
/// The EMA model is a second instance of the same architecture, kept in eval mode with gradients off. After
/// every optimizer step, [`ModelEma::update`] sets `updates += 1`,
/// `d = decay · (1 − exp(−updates / tau))` and `ema = d · ema + (1 − d) · model` for every tracked tensor.
/// Validation and export use [`ModelEma::module`].
///
/// ```
/// use folo_nn::{BatchNorm2d, ModelEma, Module};
/// let live = BatchNorm2d::new(4);
/// let mut ema = ModelEma::new(BatchNorm2d::new(4), &live).unwrap();
/// ema.update(&live).unwrap();
/// assert_eq!(ema.updates(), 1);
/// assert!((ema.decay_at(1) - 4.99825e-4).abs() < 1e-9);
/// assert!(!ema.module().is_training());
/// ```
pub struct ModelEma<M: Module> {
    module: M,
    decay: f64,
    tau: f64,
    updates: u64,
}

impl<M: Module> ModelEma<M> {
    /// Track `source` in `module` (a fresh instance of the same architecture) with docs/MODEL.md's decay
    /// 0.9999 and tau 2000. Moves `module` to `source`'s device, copies `source`'s tensors into it, puts it
    /// in eval mode and turns its gradients off.
    pub fn new(module: M, source: &dyn Module) -> Result<Self> {
        Self::with_params(module, source, EMA_DECAY, EMA_TAU)
    }

    /// [`ModelEma::new`] with an explicit decay and tau.
    pub fn with_params(module: M, source: &dyn Module, decay: f64, tau: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) || !(tau.is_finite() && tau > 0.0) {
            return Err(Error::Invalid(format!("ModelEma: invalid decay {decay} / tau {tau}")));
        }
        let mut ema = ModelEma { module, decay, tau, updates: 0 };
        if let Some(t) = source.named_tensors().first() {
            ema.module.to_device(t.tensor.device())?;
        }
        for t in ema.module.named_tensors() {
            t.tensor.set_requires_grad(false);
        }
        ema.module.eval();
        ema.zip(source, |e, m| Ok(e.copy_from(m)?))?;
        Ok(ema)
    }

    /// Restart the average from `source` (`updates = 0`), as at the start of training.
    pub fn reset(&mut self, source: &dyn Module) -> Result<()> {
        self.zip(source, |e, m| Ok(e.copy_from(m)?))?;
        self.updates = 0;
        Ok(())
    }

    /// Move the averaged model to `device`, in place (call it when the source model moves).
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.module.to_device(device)
    }

    /// The EMA model (eval mode).
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Updates so far (`ema.__updates`).
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// `(decay, tau)`.
    pub fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau }
    }

    /// The effective decay after `updates` updates: `decay · (1 − exp(−updates / tau))`.
    pub fn decay_at(&self, updates: u64) -> f64 {
        self.decay * (1.0 - (-(updates as f64) / self.tau).exp())
    }

    /// Fold the current `source` weights into the average.
    pub fn update(&mut self, source: &dyn Module) -> Result<()> {
        let d = self.decay_at(self.updates + 1);
        let (d32, w) = (d as f32, (1.0 - d) as f32);
        // On a device, every tracked tensor in one multi-tensor update (16 per dispatch on the GPU).
        let (mine, theirs) = (tracked(&self.module), tracked(source));
        if mine.iter().any(|(_, e)| !e.device().is_cpu()) && matches(&mine, &theirs) {
            let _g = no_grad();
            let es: Vec<&Tensor> = mine.iter().map(|(_, t)| t).collect();
            let ms: Vec<&Tensor> = theirs.iter().map(|(_, t)| t).collect();
            let none = vec![Vec::new(); es.len()];
            if Tensor::optimizer_step_many_(&es, &ms, &none, &OptimStep::Axpby { a: d32, b: w })? {
                self.updates += 1;
                return Ok(());
            }
        }
        self.zip(source, |e, m| {
            if e.device().is_cpu() && e.is_contiguous() && m.device().is_cpu() {
                return m
                    .with_f32_slice(|m| {
                        e.update_f32(|e| {
                            for (e, &m) in e.iter_mut().zip(m) {
                                *e = *e * d32 + m * w;
                            }
                        })
                    })?
                    .map_err(Into::into);
            }
            if !e.device().is_cpu() && e.optimizer_step_(m, &[], &OptimStep::Axpby { a: d32, b: w })? {
                return Ok(());
            }
            e.mul_scalar_assign(d32)?;
            e.add_scaled_assign(m, w)?;
            Ok(())
        })?;
        self.updates += 1;
        Ok(())
    }

    /// Run `f(ema_tensor, model_tensor)` for every tracked pair, under `no_grad`. The two models must have the
    /// same tracked names, shapes and dtypes.
    fn zip(&self, source: &dyn Module, f: impl Fn(&Tensor, &Tensor) -> Result<()>) -> Result<()> {
        let (mine, theirs) = (tracked(&self.module), tracked(source));
        if !matches(&mine, &theirs) {
            return Err(Error::Invalid(
                "ModelEma: the source model's tensors do not match the EMA model's".into(),
            ));
        }
        let _g = no_grad();
        for ((_, e), (_, m)) in mine.iter().zip(&theirs) {
            f(e, m)?;
        }
        Ok(())
    }
}

/// Whether two tracked lists pair up: the same names, shapes and dtypes.
fn matches(mine: &[(String, Tensor)], theirs: &[(String, Tensor)]) -> bool {
    mine.len() == theirs.len()
        && mine
            .iter()
            .zip(theirs)
            .all(|((a, x), (b, y))| a == b && x.shape() == y.shape() && x.dtype() == y.dtype())
}

/// Parameters and `F32` buffers, canonical order.
fn tracked<M: Module + ?Sized>(m: &M) -> Vec<(String, Tensor)> {
    m.named_tensors()
        .into_iter()
        .filter(|t| t.kind == TensorKind::Parameter || t.tensor.dtype() == DType::F32)
        .map(|t| (t.name, t.tensor))
        .collect()
}

/// What a checkpoint needs from an EMA: object-safe, so checkpoint code does not depend on the model type.
pub trait EmaState {
    /// Every tracked tensor with its model name (no `ema.` prefix), canonical order.
    fn ema_tensors(&self) -> Vec<(String, Tensor)>;
    /// `ema.__updates`.
    fn updates(&self) -> u64;
    /// `folo.ema`.
    fn contract(&self) -> EmaContract;
    /// Restore: `tensors` maps model names to saved values and must cover every tracked tensor exactly, with
    /// the same shapes.
    fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        updates: u64,
        contract: EmaContract,
    ) -> Result<()>;
}

impl<M: Module> EmaState for ModelEma<M> {
    fn ema_tensors(&self) -> Vec<(String, Tensor)> {
        tracked(&self.module)
    }

    fn updates(&self) -> u64 {
        self.updates
    }

    fn contract(&self) -> EmaContract {
        EmaContract { decay: self.decay, tau: self.tau }
    }

    fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        updates: u64,
        contract: EmaContract,
    ) -> Result<()> {
        let mine = tracked(&self.module);
        let mut problems = Vec::new();
        for (n, t) in &mine {
            match tensors.get(n) {
                None => problems.push(format!("`ema.{n}` is missing")),
                Some(s) if s.shape() != t.shape() => {
                    problems.push(format!("`ema.{n}` is {:?}, model has {:?}", s.dims(), t.dims()))
                }
                Some(_) => {}
            }
        }
        let names: std::collections::HashSet<&str> = mine.iter().map(|(n, _)| n.as_str()).collect();
        for n in tensors.keys() {
            if !names.contains(n.as_str()) {
                problems.push(format!("`ema.{n}` is not a tensor of this model"));
            }
        }
        if !problems.is_empty() {
            problems.sort();
            return Err(Error::Load(format!("EMA state: {}", problems.join("; "))));
        }
        if !(0.0..=1.0).contains(&contract.decay) || !(contract.tau.is_finite() && contract.tau > 0.0) {
            return Err(Error::Load(format!(
                "folo.ema: invalid decay {} / tau {}",
                contract.decay, contract.tau
            )));
        }
        let _g = no_grad();
        for (n, t) in &mine {
            t.copy_from(&tensors[n].to_dtype(DType::F32)?.to_device(t.device())?)?;
        }
        self.decay = contract.decay;
        self.tau = contract.tau;
        self.updates = updates;
        Ok(())
    }
}
