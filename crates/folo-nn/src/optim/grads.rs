//! Gradient accumulation over several backward passes, and global-norm clipping.

use std::collections::HashMap;

use folo_core::backend::{NormReduce, OptimStep};
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
        // On a device, every `buf += g` in one multi-tensor update (16 per dispatch on the GPU).
        let mut summed = false;
        if self.slots.iter().any(|s| s.valid && !s.param.device().is_cpu()) {
            let (mut bufs, mut gs) = (Vec::new(), Vec::new());
            for s in self.slots.iter().filter(|s| s.valid) {
                if let Some(g) = grads.get(&s.param) {
                    bufs.push(s.buf.as_ref().expect("valid slot has a buffer"));
                    gs.push(g);
                }
            }
            let none = vec![Vec::new(); bufs.len()];
            summed = Tensor::optimizer_step_many_(&bufs, &gs, &none, &OptimStep::Axpby { a: 1.0, b: 1.0 })?;
        }
        for s in &mut self.slots {
            let Some(g) = grads.get(&s.param) else { continue };
            if s.valid && summed {
                continue;
            }
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
        if self.on_device() {
            if max_norm > 0.0 {
                self.clip_grad_norm_device(max_norm, 1.0)?;
            }
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

    /// Mixed precision (docs/MODEL.md §8.8): the accumulated gradients came from a loss multiplied by a loss scale
    /// `S`. Unscale them by `inv_scale = 1 / S`, then clip like [`GradAccumulator::clip_grad_norm`] (`max_norm <= 0`
    /// disables clipping), and return whether every gradient is finite. When it returns `false` the step must be
    /// skipped (the buffers then hold garbage until the next [`GradAccumulator::zero`]).
    ///
    /// On a device this is one host readback of a single value. With clipping, the unscale is folded into the
    /// clipping factors (the norm of the unscaled gradients is `n / S`, so the threshold and the `1e-6` move by `S`
    /// instead of every buffer being scaled twice), and the finiteness check is the clipping norm itself, which
    /// the overflow-safe sum below makes non-finite exactly when some gradient is. Without clipping it is
    /// [`crate::amp::unscale_`]: one fused unscale-and-check kernel per buffer and a shared flag.
    pub fn unscale_clip(&mut self, inv_scale: f32, max_norm: f64) -> Result<bool> {
        let _g = no_grad();
        let bufs: Vec<&Tensor> =
            self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid")).collect();
        if bufs.is_empty() {
            return Ok(true);
        }
        if self.on_device() && max_norm > 0.0 {
            let r = self.clip_grad_norm_device(max_norm, inv_scale)?;
            return Ok(r.item::<f32>()?.is_finite());
        }
        if !crate::amp::unscale_(&bufs, inv_scale)? {
            return Ok(false);
        }
        self.clip_grad_norm(max_norm)?;
        Ok(true)
    }

    /// [`GradAccumulator::unscale_clip`] without the readback, for gradients on a device with clipping on
    /// (`max_norm > 0`): unscales and clips them there and returns the device scalar that is finite exactly when
    /// every gradient is, to guard the step with (`folo_core::StepGuard`) and read later. `None`, having done
    /// nothing, otherwise (the CPU, or no clipping): use [`GradAccumulator::unscale_clip`]. With no gradients it
    /// returns a finite scalar.
    pub fn unscale_clip_deferred(&mut self, inv_scale: f32, max_norm: f64) -> Result<Option<Tensor>> {
        if !(self.on_device() && max_norm > 0.0) {
            return Ok(None);
        }
        Ok(Some(self.clip_grad_norm_device(max_norm, inv_scale)?))
    }

    fn on_device(&self) -> bool {
        self.slots.iter().any(|s| s.valid && !s.param.device().is_cpu())
    }

    /// [`GradAccumulator::clip_grad_norm`] for gradients on a device, without a host round trip: the norm (in
    /// `f32`) and the factor `max_norm / (n + 1e-6)` (or 1) are device scalars and every gradient is scaled by it.
    /// The squares are summed in units of the largest magnitude `s` (`n = s · sqrt(Σ (g / s)²)`), so finite
    /// gradients whose squares overflow `f32` still give their norm, as the CPU's `f64` sum does.
    ///
    /// `pre` (1, or the loss scaler's `1 / S`) is a factor the buffers still need: the norm is that of
    /// `pre · g`, and every buffer ends up as `pre · g · min(1, max_norm / (n + 1e-6))`. Returns `n / s`, a device
    /// scalar that is finite exactly when every gradient is (a non-finite peak makes the units 0 or NaN).
    fn clip_grad_norm_device(&mut self, max_norm: f64, pre: f32) -> Result<Tensor> {
        let _g = no_grad();
        let bufs: Vec<&Tensor> =
            self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid")).collect();
        let cat = |v: Vec<Tensor>| -> Result<Tensor> {
            let refs: Vec<&Tensor> = v.iter().collect();
            Ok(Tensor::cat(&refs, 0)?)
        };
        // On the GPU each reduction is one multi-tensor call (16 buffers per dispatch) and a reduction of its
        // partials, instead of a few tensor ops per buffer; the maximum is exact either way.
        let peak = match Tensor::norm_partials_many(&bufs, NormReduce::AbsMax, None)? {
            Some(p) => p.max_all()?,
            None => {
                let peaks =
                    bufs.iter().map(|b| Ok(b.abs()?.max_all()?.reshape([1])?)).collect::<Result<Vec<_>>>()?;
                cat(peaks)?.max_all()?
            }
        };
        let scale = peak.gt_scalar(0.0)?.where_cond(&peak, &peak.ones_like()?)?;
        // Work in units of s = the largest magnitude, through h = 1 / sqrt(s): every intermediate (g·h, g·h², max·h,
        // r·sqrt(s)) stays a normal f32 for any finite s, where 1 / s or max / s could be subnormal (and flushed to
        // zero by the GPU's division) and n = s·r could overflow.
        let root = scale.sqrt()?;
        let h = root.recip()?;
        // The same terms in another summation order (per 256 elements, then over the partials).
        let sum = match Tensor::norm_partials_many(&bufs, NormReduce::ScaledSquares, Some(&h))? {
            Some(p) => p.sum_all()?,
            None => {
                let sq = bufs
                    .iter()
                    .map(|b| Ok(b.mul(&h)?.mul(&h)?.square()?.sum_all()?.reshape([1])?))
                    .collect::<Result<Vec<_>>>()?;
                cat(sq)?.sum_all()?
            }
        };
        let r = sum.sqrt()?; // n_b / s, the norm of the buffers as they are
        // The norm of `pre · g` is n = pre · n_b, so with max' = max / pre and eps' = 1e-6 / pre:
        // n > max ⇔ r·sqrt(s) > max'·h, and pre · max / (n + 1e-6) = max / (n_b + eps') = h · (max·h / (r + eps'·h²)),
        // applied as two factors (or `pre` alone below the threshold).
        let (max, pre_f) = (max_norm as f32, pre as f64);
        let clip = r.mul(&root)?.gt(&h.mul_scalar((max_norm / pre_f) as f32)?)?;
        let one = r.ones_like()?;
        let k = h.mul_scalar(max)?.div(&r.add(&h.mul(&h)?.mul_scalar((1e-6 / pre_f) as f32)?)?)?;
        let keep = if pre == 1.0 { one.clone() } else { one.mul_scalar(pre)? };
        let (f1, f2) = (clip.where_cond(&h, &keep)?, clip.where_cond(&k, &one)?);
        if !Tensor::scale_many_(&bufs, &f1, &f2)? {
            for b in &bufs {
                b.mul_assign(&f1)?;
                b.mul_assign(&f2)?;
            }
        }
        Ok(r)
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
