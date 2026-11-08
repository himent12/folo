//! Gradient accumulation over several backward passes, and global-norm clipping.

use std::collections::HashMap;

use folo_core::{DType, Grads, Tensor, TensorId, no_grad};

use super::GradSource;
use crate::Result;

struct Slot {
    param: Tensor,
    buf: Option<Tensor>,
    valid: bool,
}

/// Sums the gradients of a fixed set of parameters over several backward passes (docs/MODEL.md §8.3
/// accumulation windows), then feeds an optimizer step.
///
/// Buffers are private copies (gradients from one backward pass may share storage between parameters) and
/// are reused across windows: [`GradAccumulator::zero`] only marks them empty, so a parameter that receives
/// no gradient in a window has none ([`GradSource::grad`] returns `None`) and the optimizer skips it.
pub struct GradAccumulator {
    slots: Vec<Slot>,
    index: HashMap<TensorId, usize>,
    passes: usize,
}

impl GradAccumulator {
    /// An accumulator for `params` (gradients of other tensors are ignored).
    pub fn new(params: &[Tensor]) -> Self {
        let mut index = HashMap::with_capacity(params.len());
        let mut slots = Vec::with_capacity(params.len());
        for p in params {
            // A repeated handle keeps its first slot: one parameter, one gradient buffer.
            if let std::collections::hash_map::Entry::Vacant(e) = index.entry(p.id()) {
                e.insert(slots.len());
                slots.push(Slot { param: p.clone(), buf: None, valid: false });
            }
        }
        GradAccumulator { slots, index, passes: 0 }
    }

    /// Add the gradients of one backward pass.
    pub fn add(&mut self, grads: &Grads) -> Result<()> {
        let _g = no_grad();
        for s in &mut self.slots {
            let Some(g) = grads.get(&s.param) else { continue };
            if s.valid {
                s.buf.as_ref().expect("valid slot has a buffer").add_assign(g)?;
            } else {
                let buf = match &s.buf {
                    Some(b) => b.clone(),
                    None => {
                        let b = Tensor::zeros_with(s.param.shape(), DType::F32, s.param.device())?;
                        s.buf = Some(b.clone());
                        b
                    }
                };
                buf.copy_from(&g.to_dtype(DType::F32)?)?;
                s.valid = true;
            }
        }
        self.passes += 1;
        Ok(())
    }

    /// Backward passes added since the last [`GradAccumulator::zero`].
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// True if at least one parameter has a gradient.
    pub fn has_grads(&self) -> bool {
        self.slots.iter().any(|s| s.valid)
    }

    /// Forget the accumulated gradients (buffers are kept for reuse).
    pub fn zero(&mut self) {
        for s in &mut self.slots {
            s.valid = false;
        }
        self.passes = 0;
    }

    /// The global L2 norm `sqrt(Σ g²)` over every accumulated gradient, summed in `f64`.
    pub fn global_norm(&self) -> Result<f64> {
        let mut sq = 0f64;
        for s in self.slots.iter().filter(|s| s.valid) {
            sq += sum_sq(s.buf.as_ref().expect("valid"))?;
        }
        Ok(sq.sqrt())
    }

    /// docs/MODEL.md §8.3: if the global norm `n` exceeds `max_norm`, scale every gradient by
    /// `max_norm / (n + 1e-6)`. Returns `n` (before clipping). `max_norm <= 0` disables clipping.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> Result<f64> {
        let n = self.global_norm()?;
        if max_norm > 0.0 && n > max_norm {
            let c = (max_norm / (n + 1e-6)) as f32;
            let _g = no_grad();
            for s in self.slots.iter().filter(|s| s.valid) {
                s.buf.as_ref().expect("valid").mul_scalar_assign(c)?;
            }
        }
        Ok(n)
    }
}

impl GradSource for GradAccumulator {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        let s = &self.slots[*self.index.get(&param.id())?];
        if s.valid { s.buf.as_ref() } else { None }
    }
}

/// The global L2 norm of the gradients of `params` found in `grads` (summed in `f64`).
pub fn global_grad_norm(params: &[Tensor], grads: &dyn GradSource) -> Result<f64> {
    let mut sq = 0f64;
    for p in params {
        if let Some(g) = grads.grad(p) {
            sq += sum_sq(g)?;
        }
    }
    Ok(sq.sqrt())
}

fn sum_sq(g: &Tensor) -> Result<f64> {
    let f = |s: &[f32]| s.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
    if g.device().is_cpu() && g.dtype() == DType::F32 {
        return Ok(g.with_f32_slice(f)?);
    }
    Ok(f(&g.to_dtype(DType::F32)?.to_vec::<f32>()?))
}
