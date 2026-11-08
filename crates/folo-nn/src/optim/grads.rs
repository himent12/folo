//! Gradient accumulation over several backward passes, and global-norm clipping.

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Grads, Tensor, TensorId, no_grad};

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
                let buf = s.buf.as_ref().expect("valid slot has a buffer");
                // One fused in-place `buf = buf + g` where the backend has it (the GPU), else a tensor op.
                let fused = !buf.device().is_cpu()
                    && buf.optimizer_step_(g, &[], &OptimStep::Axpby { a: 1.0, b: 1.0 })?;
                if !fused {
                    buf.add_assign(g)?;
                }
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

    /// Move the tracked parameters and the accumulation buffers to `device`, in place (see
    /// [`crate::Module::to_device`]).
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        for s in &self.slots {
            s.param.to_device_(device)?;
            if let Some(b) = &s.buf {
                b.to_device_(device)?;
            }
        }
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
    /// `max_norm / (n + 1e-6)`. Returns `n` (before clipping). `max_norm <= 0` disables clipping. Gradients on a
    /// device are clipped there without reading the norm back, and the result is then NaN (use
    /// [`GradAccumulator::global_norm`] to read it).
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> Result<f64> {
        if self.slots.iter().any(|s| s.valid && !s.param.device().is_cpu()) {
            self.clip_grad_norm_device(max_norm)?;
            return Ok(f64::NAN);
        }
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

    /// [`GradAccumulator::clip_grad_norm`] for gradients on a device, without a host round trip: the norm (in
    /// `f32`) and the factor `max_norm / (n + 1e-6)` (or 1) are device scalars and every gradient is scaled by it.
    fn clip_grad_norm_device(&mut self, max_norm: f64) -> Result<()> {
        if max_norm <= 0.0 {
            return Ok(());
        }
        let _g = no_grad();
        let mut sq: Vec<Tensor> = Vec::new();
        for s in self.slots.iter().filter(|s| s.valid) {
            let b = s.buf.as_ref().expect("valid");
            sq.push(b.square()?.sum_all()?.reshape([1])?);
        }
        let refs: Vec<&Tensor> = sq.iter().collect();
        let n = Tensor::cat(&refs, 0)?.sum_all()?.sqrt()?;
        let max = max_norm as f32;
        let factor =
            n.gt_scalar(max)?.where_cond(&n.add_scalar(1e-6)?.recip()?.mul_scalar(max)?, &n.ones_like()?)?;
        for s in self.slots.iter().filter(|s| s.valid) {
            s.buf.as_ref().expect("valid").mul_assign(&factor)?;
        }
        Ok(())
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
