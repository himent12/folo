//! Gradient accumulation over several backward passes, and global-norm clipping.

use std::collections::{HashMap, HashSet};

use folo_core::backend::{NormReduce, OptimStep};
use folo_core::{DType, Device, Grads, Tensor, TensorId, no_grad};
use rayon::prelude::*;

use super::simd::{
    ScaledSumSq, axpby_slice, check_finite_slice, check_sum_sq_f64_slice, scale_slice, scale2_slice,
    sum_sq_f64_slice, unscale_check_slice, unscale_check_sum_sq_slice,
};
use super::{GradSource, PAR_THRESHOLD, cpu_fast, storage_key};
use crate::Result;

struct Slot {
    param: Tensor,
    buf: Option<Tensor>,
    valid: bool,
    /// True when `buf` was adopted directly from [`GradAccumulator::add_owned`] rather than allocated as a
    /// private accumulator buffer; [`GradAccumulator::zero`] drops adopted buffers immediately so they return to
    /// the CPU/GPU buffer pool before the next forward/backward pass.
    adopted: bool,
}

/// Sums the gradients of a fixed set of parameters over several backward passes (docs/MODEL.md §8.3
/// accumulation windows), then feeds an optimizer step.
///
/// Buffers added via [`GradAccumulator::add`] are private copies (gradients from one backward pass may share
/// storage between parameters) and are reused across windows: [`GradAccumulator::zero`] only marks them empty,
/// so a parameter that receives no gradient in a window has none ([`GradSource::grad`] returns `None`) and the
/// optimizer skips it. [`GradAccumulator::add_owned`] takes ownership of a [`Grads`] map and adopts unaliased
/// gradient buffers on empty slots with zero copy, releasing them back to the backend's buffer pool on
/// [`GradAccumulator::zero`].
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
                slots.push(Slot { param: p.clone(), buf: None, valid: false, adopted: false });
            }
        }
        GradAccumulator { slots, index, passes: 0 }
    }

    /// Add the gradients of one backward pass, copying into private accumulation buffers.
    pub fn add(&mut self, grads: &Grads) -> Result<()> {
        let _g = no_grad();
        let pairs: Vec<(usize, &Tensor)> =
            self.slots.iter().enumerate().filter_map(|(i, s)| grads.get(&s.param).map(|g| (i, g))).collect();
        let empty_snaps = self.snapshot_aliased_empty(&pairs)?;
        self.add_valid_pairs(&pairs)?;
        for (pos, &(i, g)) in pairs.iter().enumerate() {
            let s = &mut self.slots[i];
            if !s.valid {
                let src = empty_snaps.get(pos).and_then(|o| o.as_ref()).unwrap_or(g);
                copy_into_private_slot(s, src)?;
            }
        }
        self.passes += 1;
        Ok(())
    }

    /// Add the gradients of one backward pass, taking ownership of `grads`.
    ///
    /// On empty slots (pass 0 of an accumulation window, or every batch when `accumulate == 1`), a gradient
    /// tensor that is contiguous, `F32`, has zero offset and full storage ownership, matches the parameter's
    /// shape and device, does not require grad, and does not alias any tracked parameter, accumulator buffer, or
    /// other tracked gradient in `grads` is adopted directly with zero copy and zero allocation. Subsequent
    /// passes or clipping mutate the adopted buffer in place, and [`GradAccumulator::zero`] drops it so its
    /// storage returns to the buffer pool before the next forward pass. Callers that retain external handles to
    /// tensors inside `grads` should use [`GradAccumulator::add`].
    pub fn add_owned(&mut self, mut grads: Grads) -> Result<()> {
        let _g = no_grad();
        let need_adopt = self.slots.iter().any(|s| !s.valid && grads.get(&s.param).is_some());
        let key_counts = if need_adopt {
            let mut counts: HashMap<usize, u32> = HashMap::with_capacity(self.slots.len() * 3);
            for s in &self.slots {
                *counts.entry(storage_key(&s.param)).or_insert(0) += 2;
                if let Some(b) = &s.buf {
                    *counts.entry(storage_key(b)).or_insert(0) += 2;
                }
                if let Some(g) = grads.get(&s.param) {
                    *counts.entry(storage_key(g)).or_insert(0) += 1;
                }
            }
            counts
        } else {
            HashMap::new()
        };

        let mut extracted: Vec<(usize, Tensor)> = Vec::with_capacity(self.slots.len().min(grads.len()));
        for (i, s) in self.slots.iter().enumerate() {
            if let Some(g) = grads.remove(&s.param) {
                extracted.push((i, g));
            }
        }
        // Drop any untracked gradients in `grads` before copying/accumulating.
        drop(grads);

        let all_pairs: Vec<(usize, &Tensor)> = extracted.iter().map(|(i, g)| (*i, g)).collect();
        let empty_snaps = self.snapshot_aliased_empty(&all_pairs)?;
        self.add_valid_pairs(&all_pairs)?;

        for (pos, (i, g)) in extracted.into_iter().enumerate() {
            let s = &mut self.slots[i];
            if s.valid {
                continue;
            }
            if let Some(snap) = empty_snaps.get(pos).and_then(|o| o.as_ref()) {
                copy_into_private_slot(s, snap)?;
                continue;
            }
            let can_adopt = !g.requires_grad()
                && g.dtype() == DType::F32
                && g.device() == s.param.device()
                && g.shape() == s.param.shape()
                && g.is_contiguous()
                && g.layout().offset() == 0
                && g.with_view(|v| v.storage.len() == g.numel())
                && key_counts.get(&storage_key(&g)).copied().unwrap_or(0) == 1;
            if can_adopt {
                s.buf = Some(g);
                s.valid = true;
                s.adopted = true;
            } else {
                copy_into_private_slot(s, &g)?;
            }
        }
        self.passes += 1;
        Ok(())
    }

    /// Snapshot any incoming gradient for an empty slot whose storage aliases an accumulator buffer that will
    /// be mutated by this pass (`add_valid_pairs` or an earlier empty-slot copy). Returns an empty `Vec` with no
    /// allocations when no aliasing is present.
    fn snapshot_aliased_empty(&self, pairs: &[(usize, &Tensor)]) -> Result<Vec<Option<Tensor>>> {
        if !pairs.iter().any(|&(i, _)| !self.slots[i].valid)
            || !pairs.iter().any(|&(i, _)| self.slots[i].buf.is_some())
        {
            return Ok(Vec::new());
        }
        let mut written_keys: HashSet<usize> = HashSet::new();
        for &(i, _) in pairs {
            if let Some(buf) = &self.slots[i].buf {
                written_keys.insert(storage_key(buf));
            }
        }
        let mut snaps = Vec::new();
        for (pos, &(i, g)) in pairs.iter().enumerate() {
            if !self.slots[i].valid && written_keys.contains(&storage_key(g)) {
                if snaps.is_empty() {
                    snaps.resize(pairs.len(), None);
                }
                snaps[pos] = Some(g.mul_scalar(1.0)?);
            }
        }
        Ok(snaps)
    }

    /// Accumulate `g` into every already-valid slot referenced by `pairs`.
    fn add_valid_pairs(&mut self, pairs: &[(usize, &Tensor)]) -> Result<()> {
        // Detect any incoming gradient that aliases ANY destination buffer (self-slot or cross-slot) before
        // mutating any buffer, and snapshot those reads up front so neither CPU parallel RwLocks nor GPU
        // storage bindings / sequential updates can deadlock or observe read-after-write corruption.
        let mut written_keys: HashSet<usize> = HashSet::new();
        let mut valid_count = 0usize;
        for &(i, _) in pairs {
            if self.slots[i].valid {
                let buf = self.slots[i].buf.as_ref().expect("valid slot has a buffer");
                written_keys.insert(storage_key(buf));
                valid_count += 1;
            }
        }
        if valid_count == 0 {
            return Ok(());
        }
        let mut snapshots: Vec<Option<Tensor>> = vec![None; pairs.len()];
        for (pos, &(i, g)) in pairs.iter().enumerate() {
            if self.slots[i].valid && written_keys.contains(&storage_key(g)) {
                snapshots[pos] = Some(g.mul_scalar(1.0)?);
            }
        }
        let grad_at = |pos: usize| -> &Tensor { snapshots[pos].as_ref().unwrap_or(pairs[pos].1) };

        let mut dev_done: Vec<bool> = vec![false; pairs.len()];
        if pairs.iter().any(|&(i, _)| self.slots[i].valid && !self.slots[i].param.device().is_cpu()) {
            let (mut idxs, mut bufs, mut gs) = (Vec::new(), Vec::new(), Vec::new());
            for (pos, &(i, _)) in pairs.iter().enumerate() {
                let s = &self.slots[i];
                if s.valid && !s.param.device().is_cpu() {
                    let buf = s.buf.as_ref().expect("valid slot has a buffer");
                    idxs.push(pos);
                    bufs.push(buf);
                    gs.push(grad_at(pos));
                }
            }
            if !bufs.is_empty() {
                let none = vec![Vec::new(); bufs.len()];
                if Tensor::optimizer_step_many_(&bufs, &gs, &none, &OptimStep::Axpby { a: 1.0, b: 1.0 })? {
                    for pos in idxs {
                        dev_done[pos] = true;
                    }
                }
            }
        }

        // Parallel AVX2 fast path for contiguous CPU F32 accumulations when total elements exceed PAR_THRESHOLD
        // and every mutated buffer has a distinct storage (aliased reads were snapshotted above).
        let bufs_unique = written_keys.len() == valid_count;
        let cpu_fast_total: usize = if bufs_unique {
            pairs
                .iter()
                .enumerate()
                .filter_map(|(pos, &(i, _))| {
                    if dev_done[pos] {
                        return None;
                    }
                    let s = &self.slots[i];
                    if !s.valid {
                        return None;
                    }
                    let buf = s.buf.as_ref().expect("valid slot has a buffer");
                    let g = grad_at(pos);
                    if cpu_fast(&[buf])
                        && g.device().is_cpu()
                        && g.dtype() == DType::F32
                        && buf.shape() == g.shape()
                    {
                        Some(buf.numel())
                    } else {
                        None
                    }
                })
                .sum()
        } else {
            0
        };

        if cpu_fast_total >= PAR_THRESHOLD {
            let mut fast_items: Vec<(&Tensor, &Tensor)> = Vec::new();
            for (pos, &(i, _)) in pairs.iter().enumerate() {
                if dev_done[pos] {
                    continue;
                }
                let s = &self.slots[i];
                if !s.valid {
                    continue;
                }
                let buf = s.buf.as_ref().expect("valid slot has a buffer");
                let g = grad_at(pos);
                if cpu_fast(&[buf])
                    && g.device().is_cpu()
                    && g.dtype() == DType::F32
                    && buf.shape() == g.shape()
                {
                    fast_items.push((buf, g));
                    dev_done[pos] = true;
                }
            }
            folo_core::cpu::in_pool(|| {
                fast_items.par_iter().try_for_each(|&(buf, g)| -> Result<()> {
                    let _g = no_grad();
                    g.with_f32_slice(|gs| buf.update_f32(|bs| axpby_slice(bs, gs, 1.0, 1.0)))??;
                    Ok(())
                })
            })?;
        }

        for (pos, &(i, _)) in pairs.iter().enumerate() {
            if dev_done[pos] || !self.slots[i].valid {
                continue;
            }
            let buf = self.slots[i].buf.as_ref().expect("valid slot has a buffer");
            let g = grad_at(pos);
            // One fused in-place `buf = buf + g` where the backend has it (the GPU), else a tensor op.
            let fused = !buf.device().is_cpu()
                && buf.optimizer_step_(g, &[], &OptimStep::Axpby { a: 1.0, b: 1.0 })?;
            if !fused {
                if cpu_fast(&[buf])
                    && g.device().is_cpu()
                    && g.dtype() == DType::F32
                    && buf.shape() == g.shape()
                {
                    g.with_f32_slice(|gs| buf.update_f32(|bs| axpby_slice(bs, gs, 1.0, 1.0)))??;
                } else {
                    buf.add_assign(g)?;
                }
            }
        }
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

    /// Forget the accumulated gradients. Private buffers allocated by [`GradAccumulator::add`] are kept for
    /// reuse; buffers adopted by [`GradAccumulator::add_owned`] are released immediately so their storage
    /// returns to the backend's pool before the next forward pass.
    pub fn zero(&mut self) {
        for s in &mut self.slots {
            s.valid = false;
            if s.adopted {
                s.buf = None;
                s.adopted = false;
            }
        }
        self.passes = 0;
    }

    /// The global L2 norm `sqrt(Σ g²)` over every accumulated gradient, summed in `f64` with overflow- and
    /// underflow-safe online rescaling (and a single 2-float readback on a GPU device).
    pub fn global_norm(&self) -> Result<f64> {
        if self.on_device() {
            let bufs: Vec<&Tensor> =
                self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid")).collect();
            if let Some(n) = device_global_norm(&bufs)? {
                return Ok(n);
            }
        }
        let total: usize =
            self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid").numel()).sum();
        if total >= PAR_THRESHOLD && !self.on_device() {
            let partials: Vec<ScaledSumSq> = folo_core::cpu::in_pool(|| {
                self.slots
                    .par_iter()
                    .filter(|s| s.valid)
                    .map(|s| scaled_sum_sq(s.buf.as_ref().expect("valid")))
                    .collect::<Result<Vec<_>>>()
            })?;
            return Ok(partials.into_iter().fold(ScaledSumSq::ZERO, ScaledSumSq::merge).norm());
        }
        let mut acc = ScaledSumSq::ZERO;
        for s in self.slots.iter().filter(|s| s.valid) {
            acc = acc.merge(scaled_sum_sq(s.buf.as_ref().expect("valid"))?);
        }
        Ok(acc.norm())
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
        if max_norm > 0.0 && n.is_finite() && n > max_norm {
            self.scale_cpu_valid_f64(max_norm / (n + 1e-6))?;
        }
        Ok(n)
    }

    /// Adaptive Gradient Clipping (Brock et al. 2021, NFNet) across every accumulated gradient in place:
    /// scales each output unit `i` whose gradient norm exceeds `clip_factor * max(||W_i||_F, eps)` down to that
    /// bound. When `skip_1d` is true, 0-D and 1-D parameters (biases, BatchNorm scales) are left unclipped.
    pub fn clip_grad_agc(&mut self, clip_factor: f64, eps: f64, skip_1d: bool) -> Result<super::AgcStats> {
        super::agc::validate_agc_params(clip_factor, eps)?;
        let mut stats = super::AgcStats::default();
        for s in self.slots.iter().filter(|s| s.valid) {
            if skip_1d && s.param.dims().len() < 2 {
                continue;
            }
            let buf = s.buf.as_ref().expect("valid");
            stats = stats.merge(super::agc::clip_grad_agc_tensor(&s.param, buf, clip_factor, eps)?);
        }
        Ok(stats)
    }

    /// Scale every valid CPU buffer by `c64` in place (AVX2, parallelized when large). When `c64` is a positive
    /// subnormal or underflows `f32`, uses two normal `f32` factors `f1 · f2 = c64` in one pass via
    /// [`scale2_slice`] when both stay normal (`c64 >= f32::MIN_POSITIVE²`), and falls back to widened `f64`
    /// scaling for tinier positive factors so finite gradients never turn into `NaN` via `0 * Inf`.
    fn scale_cpu_valid_f64(&self, c64: f64) -> Result<()> {
        #[derive(Clone, Copy)]
        enum ScaleMode {
            Single(f32),
            Split(f32, f32),
            Wide(f64),
        }
        let _g = no_grad();
        let total: usize =
            self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid").numel()).sum();
        let mode = if c64 > 0.0 && c64 < f32::MIN_POSITIVE as f64 {
            let f1 = c64.sqrt() as f32;
            let f2 = if f1 >= f32::MIN_POSITIVE { (c64 / f1 as f64) as f32 } else { 0.0 };
            if f1 >= f32::MIN_POSITIVE && f2 >= f32::MIN_POSITIVE && f2.is_finite() {
                ScaleMode::Split(f1, f2)
            } else {
                ScaleMode::Wide(c64)
            }
        } else {
            ScaleMode::Single(c64 as f32)
        };
        let scale_one = |b: &Tensor| -> Result<()> {
            match mode {
                ScaleMode::Single(c32) => {
                    if cpu_fast(&[b]) {
                        b.update_f32(|s| scale_slice(s, c32))?;
                    } else {
                        b.mul_scalar_assign(c32)?;
                    }
                }
                ScaleMode::Split(f1, f2) => {
                    if cpu_fast(&[b]) {
                        b.update_f32(|s| scale2_slice(s, f1, f2))?;
                    } else {
                        b.mul_scalar_assign(f1)?;
                        b.mul_scalar_assign(f2)?;
                    }
                }
                ScaleMode::Wide(c) => {
                    if cpu_fast(&[b]) {
                        b.update_f32(|s| {
                            for v in s {
                                *v = (*v as f64 * c) as f32;
                            }
                        })?;
                    } else {
                        let vals: Vec<f32> = b
                            .to_dtype(DType::F32)?
                            .to_vec::<f32>()?
                            .into_iter()
                            .map(|v| (v as f64 * c) as f32)
                            .collect();
                        let scaled = Tensor::from_vec(vals, b.shape().clone())?.to_dtype(b.dtype())?;
                        b.copy_from(&scaled)?;
                    }
                }
            }
            Ok(())
        };
        if total >= PAR_THRESHOLD {
            folo_core::cpu::in_pool(|| {
                self.slots.par_iter().filter(|s| s.valid).try_for_each(|s| {
                    let _g = no_grad();
                    scale_one(s.buf.as_ref().expect("valid"))
                })
            })
        } else {
            for s in self.slots.iter().filter(|s| s.valid) {
                scale_one(s.buf.as_ref().expect("valid"))?;
            }
            Ok(())
        }
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
        if !self.slots.iter().any(|s| s.valid) {
            return Ok(true);
        }
        if !self.on_device()
            && self.slots.iter().filter(|s| s.valid).all(|s| cpu_fast(&[s.buf.as_ref().expect("valid")]))
        {
            let total: usize =
                self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid").numel()).sum();
            let unit_scale = inv_scale == 1.0;
            if max_norm <= 0.0 {
                let check_one = |b: &Tensor| -> Result<bool> {
                    if unit_scale {
                        Ok(b.with_f32_slice(check_finite_slice)?)
                    } else {
                        Ok(b.update_f32(|buf| unscale_check_slice(buf, inv_scale))?)
                    }
                };
                if total >= PAR_THRESHOLD {
                    return folo_core::cpu::in_pool(|| {
                        let flags: Vec<bool> = self
                            .slots
                            .par_iter()
                            .filter(|s| s.valid)
                            .map(|s| {
                                let _g = no_grad();
                                check_one(s.buf.as_ref().expect("valid"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(flags.into_iter().all(|ok| ok))
                    });
                }
                let mut finite = true;
                for s in self.slots.iter().filter(|s| s.valid) {
                    finite &= check_one(s.buf.as_ref().expect("valid"))?;
                }
                return Ok(finite);
            }
            let scan_one = |b: &Tensor| -> Result<(bool, ScaledSumSq)> {
                let (ok, sq) = if unit_scale {
                    b.with_f32_slice(check_sum_sq_f64_slice)?
                } else {
                    b.update_f32(|buf| unscale_check_sum_sq_slice(buf, inv_scale))?
                };
                Ok((ok, ScaledSumSq::from_sq_f64(sq)))
            };
            let (finite, acc_sq) = if total >= PAR_THRESHOLD {
                let res: Vec<(bool, ScaledSumSq)> = folo_core::cpu::in_pool(|| {
                    self.slots
                        .par_iter()
                        .filter(|s| s.valid)
                        .map(|s| {
                            let _g = no_grad();
                            scan_one(s.buf.as_ref().expect("valid"))
                        })
                        .collect::<Result<Vec<_>>>()
                })?;
                res.into_iter().fold((true, ScaledSumSq::ZERO), |(ok, acc), (f, q)| (ok & f, acc.merge(q)))
            } else {
                let mut finite = true;
                let mut acc = ScaledSumSq::ZERO;
                for s in self.slots.iter().filter(|s| s.valid) {
                    let (f, q) = scan_one(s.buf.as_ref().expect("valid"))?;
                    finite &= f;
                    acc = acc.merge(q);
                }
                (finite, acc)
            };
            if !finite {
                return Ok(false);
            }
            let n = acc_sq.norm();
            if !n.is_finite() {
                return Ok(false);
            }
            if n > max_norm {
                self.scale_cpu_valid_f64(max_norm / (n + 1e-6))?;
            }
            return Ok(true);
        }
        let bufs: Vec<&Tensor> =
            self.slots.iter().filter(|s| s.valid).map(|s| s.buf.as_ref().expect("valid")).collect();
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
        // applied as two factors (or `pre` alone below the threshold). If `(max / pre) as f32` rounds to infinity (a
        // loss scale near `MAX_SCALE`, or a very large `clip_grad_norm`), the comparison is false for finite buffers,
        // so the factors are `pre` and 1 and no NaN comes from it. The step is then not clipped, even if the buffers'
        // norm is above `max · S`, a known limit: such buffers hold values within a few orders of `f32::MAX`.
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

/// Copy `g` into a private accumulator buffer on `s` (reusing `s.buf` when it is already a private buffer).
fn copy_into_private_slot(s: &mut Slot, g: &Tensor) -> Result<()> {
    let buf = match &s.buf {
        Some(b) if !s.adopted => b.clone(),
        _ => {
            let b = Tensor::zeros_with(s.param.shape(), DType::F32, s.param.device())?;
            s.buf = Some(b.clone());
            s.adopted = false;
            b
        }
    };
    let g_f32;
    let src = if buf.same_storage(g) {
        g_f32 = g.to_dtype(DType::F32)?.mul_scalar(1.0)?;
        &g_f32
    } else {
        g
    };
    if cpu_fast(&[&buf]) && src.device().is_cpu() && src.dtype() == DType::F32 && buf.shape() == src.shape() {
        src.with_f32_slice(|gs| buf.update_f32(|bs| bs.copy_from_slice(gs)))??;
    } else {
        buf.copy_from(&src.to_dtype(DType::F32)?)?;
    }
    s.valid = true;
    Ok(())
}

impl GradSource for GradAccumulator {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        let s = &self.slots[*self.index.get(&param.id())?];
        if s.valid { s.buf.as_ref() } else { None }
    }
}

/// The global L2 norm of the gradients of `params` found in `grads` (summed in `f64` with overflow-safe
/// rescaling, or via a single 2-float device readback when all gradients live on one GPU).
pub fn global_grad_norm(params: &[Tensor], grads: &dyn GradSource) -> Result<f64> {
    let gs: Vec<&Tensor> = params.iter().filter_map(|p| grads.grad(p)).collect();
    if gs.is_empty() {
        return Ok(0.0);
    }
    if !gs[0].device().is_cpu()
        && let Some(n) = device_global_norm(&gs)?
    {
        return Ok(n);
    }
    let all_cpu = gs.iter().all(|g| g.device().is_cpu());
    let total: usize = gs.iter().map(|g| g.numel()).sum();
    if all_cpu && total >= PAR_THRESHOLD {
        let partials: Vec<ScaledSumSq> =
            folo_core::cpu::in_pool(|| gs.par_iter().map(|g| scaled_sum_sq(g)).collect::<Result<Vec<_>>>())?;
        return Ok(partials.into_iter().fold(ScaledSumSq::ZERO, ScaledSumSq::merge).norm());
    }
    let mut acc = ScaledSumSq::ZERO;
    for g in gs {
        acc = acc.merge(scaled_sum_sq(g)?);
    }
    Ok(acc.norm())
}

/// Overflow- and underflow-safe global L2 norm on a GPU device in a single 2-float readback (`[peak, sum]`),
/// returning `Ok(None)` if the tensors do not share one device, repeat any underlying storage (avoiding
/// recursive same-thread `RwLock::read` in `norm_partials_many`), cannot use the multi-tensor kernel, or have a
/// subnormal/zero peak where WGSL may flush subnormals to zero (falling back to exact host `f64` summation).
fn device_global_norm(bufs: &[&Tensor]) -> Result<Option<f64>> {
    let non_empty: Vec<&Tensor> = bufs.iter().copied().filter(|b| b.numel() > 0).collect();
    let Some(first) = non_empty.first() else {
        return Ok(Some(0.0));
    };
    let dev = first.device();
    if dev.is_cpu() || !non_empty.iter().all(|b| b.device() == dev && b.dtype() == DType::F32) {
        return Ok(None);
    }
    let mut seen = HashSet::with_capacity(non_empty.len());
    if !non_empty.iter().all(|b| seen.insert(storage_key(b))) {
        return Ok(None);
    }
    let _g = no_grad();
    let Some(peak_partials) = Tensor::norm_partials_many(&non_empty, NormReduce::AbsMax, None)? else {
        return Ok(None);
    };
    let peak = peak_partials.max_all()?;
    let scale = peak.gt_scalar(0.0)?.where_cond(&peak, &peak.ones_like()?)?;
    let root = scale.sqrt()?;
    let h = root.recip()?;
    let Some(sum_partials) = Tensor::norm_partials_many(&non_empty, NormReduce::ScaledSquares, Some(&h))?
    else {
        return Ok(None);
    };
    let sum = sum_partials.sum_all()?;
    let pair = Tensor::cat(&[&peak.reshape([1])?, &sum.reshape([1])?], 0)?.to_vec::<f32>()?;
    let (p, s) = (pair[0], pair[1]);
    if p.is_nan() {
        return Ok(Some(f64::NAN));
    }
    if p.is_infinite() {
        return Ok(Some(f64::INFINITY));
    }
    // If `p` is zero or subnormal, WGSL is allowed to flush subnormals to zero in `AbsMax` or `sqrt`, so fall
    // back to exact host `f64` summation (zero cost on all normal finite gradients).
    if p < f32::MIN_POSITIVE || !s.is_finite() {
        return Ok(None);
    }
    Ok(Some((p as f64) * (s as f64).sqrt()))
}

fn scaled_sum_sq(g: &Tensor) -> Result<ScaledSumSq> {
    let sq = if g.device().is_cpu() && g.dtype() == DType::F32 {
        g.with_f32_slice(sum_sq_f64_slice)?
    } else {
        sum_sq_f64_slice(&g.to_dtype(DType::F32)?.to_vec::<f32>()?)
    };
    Ok(ScaledSumSq::from_sq_f64(sq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_grad_norm_does_not_zero_finite_grads_when_norm_is_infinite() {
        let p_fin = Tensor::from_vec(vec![1.0f32, 2.0], [2]).unwrap().requires_grad_();
        let p_inf = Tensor::from_vec(vec![3.0f32, 4.0], [2]).unwrap().requires_grad_();
        let mut acc = GradAccumulator::new(&[p_fin.clone(), p_inf.clone()]);

        // Build gradients via trivial backward passes so `Grads` contains a finite tensor and an +Inf tensor.
        let g_fin = Tensor::from_vec(vec![3.0f32, -4.0], [2]).unwrap();
        let g_inf = Tensor::from_vec(vec![f32::INFINITY, 1.0], [2]).unwrap();
        let loss = p_fin
            .mul(&g_fin)
            .unwrap()
            .sum_all()
            .unwrap()
            .add(&p_inf.mul(&g_inf).unwrap().sum_all().unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();
        acc.add(&grads).unwrap();

        let n = acc.clip_grad_norm(1.0).unwrap();
        assert!(n.is_infinite() && n.is_sign_positive(), "expected +Inf norm, got {n}");
        // Finite gradients must remain untouched (not multiplied by 1.0 / Inf = 0.0), and +Inf must not become NaN.
        assert_eq!(acc.grad(&p_fin).unwrap().to_vec::<f32>().unwrap(), vec![3.0, -4.0]);
        let inf_vals = acc.grad(&p_inf).unwrap().to_vec::<f32>().unwrap();
        assert!(inf_vals[0].is_infinite() && inf_vals[0].is_sign_positive(), "{inf_vals:?}");
        assert_eq!(inf_vals[1], 1.0);
    }
}
