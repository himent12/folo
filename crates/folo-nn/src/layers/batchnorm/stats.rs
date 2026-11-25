//! Running-statistics snapshot and finiteness checks for non-finite batch rollback (docs/MODEL.md §8.3).

use folo_core::backend::NormReduce;
use folo_core::{DType, Tensor, no_grad};

use crate::Result;
use crate::module::{Mode, Module, TensorKind};

/// A copy of every floating buffer of a model (the BatchNorm running statistics), taken before a train-mode
/// forward so a non-finite batch can be undone (docs/MODEL.md §8.3, "Non-finite batches").
///
/// A single `RunningStats` instance can be captured once before an epoch and refreshed in place across batches
/// with [`RunningStats::recapture`] to avoid per-batch heap and tensor allocations, and checked with
/// [`RunningStats::all_finite_live`] without re-walking the module tree.
///
/// ```
/// use folo_core::Tensor;
/// use folo_nn::{BatchNorm2d, Forward, RunningStats};
///
/// let bn = BatchNorm2d::new(2);
/// let snap = RunningStats::capture(&bn).unwrap();
/// bn.forward(&Tensor::full([2, 2, 2, 2], f32::NAN)).unwrap();
/// assert!(!RunningStats::all_finite(&bn).unwrap());
/// snap.restore().unwrap();
/// assert!(RunningStats::all_finite(&bn).unwrap());
/// ```
pub struct RunningStats {
    saved: Vec<(Tensor, Tensor)>,
}

impl RunningStats {
    /// Copy every floating buffer of `model`.
    pub fn capture<M: Module + ?Sized>(model: &M) -> Result<Self> {
        Self::capture_filtered(model, false)
    }

    /// Copy every floating buffer belonging to modules that are not in evaluation mode (i.e. active training
    /// `BatchNorm2d` layers). Frozen/eval layers never mutate their running statistics during forward, so
    /// skipping them avoids both unnecessary copies and unnecessary device finiteness syncs when all BNs are
    /// frozen (such as `bn_freeze_close_mosaic`).
    pub fn capture_training<M: Module + ?Sized>(model: &M) -> Result<Self> {
        Self::capture_filtered(model, true)
    }

    fn capture_filtered<M: Module + ?Sized>(model: &M, training_only: bool) -> Result<Self> {
        let _g = no_grad();
        let mut saved = Vec::new();
        let mut err: Option<crate::Error> = None;
        visit_float_buffers(model, training_only, &mut |b| {
            if err.is_some() {
                return;
            }
            match allocate_copy(b) {
                Ok(copy) => saved.push((b.clone(), copy)),
                Err(e) => err = Some(e),
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        Ok(RunningStats { saved })
    }

    /// Number of captured buffers.
    pub fn len(&self) -> usize {
        self.saved.len()
    }

    /// True if no buffers were captured (e.g. all BatchNorms are frozen or fused).
    pub fn is_empty(&self) -> bool {
        self.saved.is_empty()
    }

    /// Refresh every saved snapshot in place from its live buffer, reusing the existing copy buffers with zero
    /// heap or tensor allocations in the steady state. If a live buffer changed shape, dtype, or device since
    /// capture, that slot's backing copy is reallocated cleanly.
    pub fn recapture(&mut self) -> Result<()> {
        if self.saved.is_empty() {
            return Ok(());
        }
        let _g = no_grad();
        for (live, copy) in &mut self.saved {
            if copy.shape() == live.shape() && copy.dtype() == live.dtype() && copy.device() == live.device()
            {
                copy_tensor_fast(copy, live)?;
            } else {
                *copy = allocate_copy(live)?;
            }
        }
        Ok(())
    }

    /// Write the copies back into the model's buffers.
    pub fn restore(&self) -> Result<()> {
        if self.saved.is_empty() {
            return Ok(());
        }
        let _g = no_grad();
        for (live, copy) in &self.saved {
            copy_tensor_fast(live, copy)?;
        }
        Ok(())
    }

    /// True if every live `F32` buffer tracked by this snapshot is finite, without re-walking the module tree.
    /// When [`RunningStats::is_empty`] is true, returns `Ok(true)` immediately with zero device dispatches or
    /// host-sync stalls.
    pub fn all_finite_live(&self) -> Result<bool> {
        if self.saved.is_empty() {
            return Ok(true);
        }
        let mut device: Vec<&Tensor> = Vec::new();
        for (live, _) in &self.saved {
            if live.dtype() != DType::F32 {
                continue;
            }
            if live.device().is_cpu() {
                if !cpu_f32_all_finite(live)? {
                    return Ok(false);
                }
            } else if live.numel() > 0 {
                device.push(live);
            }
        }
        device_f32_all_finite(&device)
    }

    /// True if every floating (`F32`) buffer of `model` is finite. Buffers on a device are checked there with
    /// one readback per device (via multi-tensor `AbsMax` partials when supported, else `Σ 0 · x == 0`).
    pub fn all_finite<M: Module + ?Sized>(model: &M) -> Result<bool> {
        let mut device: Vec<Tensor> = Vec::new();
        let mut finite = true;
        let mut err: Option<crate::Error> = None;
        visit_float_buffers(model, false, &mut |b| {
            if !finite || err.is_some() || b.dtype() != DType::F32 {
                return;
            }
            if b.device().is_cpu() {
                match cpu_f32_all_finite(b) {
                    Ok(true) => {}
                    Ok(false) => finite = false,
                    Err(e) => err = Some(e),
                }
            } else if b.numel() > 0 {
                device.push(b.clone());
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        if !finite {
            return Ok(false);
        }
        let refs: Vec<&Tensor> = device.iter().collect();
        device_f32_all_finite(&refs)
    }
}

/// Walk `model` and call `f` on every floating buffer without formatting dotted names or cloning parameters.
/// When `training_only` is true, buffers directly owned by modules currently in eval mode are skipped.
fn visit_float_buffers<M: Module + ?Sized>(model: &M, training_only: bool, f: &mut dyn FnMut(&Tensor)) {
    let include_own = !training_only || model.mode().is_none_or(Mode::is_training);
    if include_own {
        model.visit_tensors(&mut |_, t, kind| {
            if kind == TensorKind::Buffer && t.dtype().is_float() {
                f(t);
            }
        });
    }
    model.visit_children(&mut |_, child| {
        visit_float_buffers(child, training_only, f);
    });
}

fn allocate_copy(src: &Tensor) -> Result<Tensor> {
    let copy = Tensor::zeros_with(src.shape(), src.dtype(), src.device())?;
    copy_tensor_fast(&copy, src)?;
    Ok(copy)
}

/// Copy `src` into `dst` (same shape, dtype, device), using a direct slice copy on contiguous CPU `F32`
/// tensors to avoid per-call tensor dispatch overhead, and falling back to [`Tensor::copy_from`].
fn copy_tensor_fast(dst: &Tensor, src: &Tensor) -> Result<()> {
    if dst.device().is_cpu()
        && src.device().is_cpu()
        && dst.dtype() == DType::F32
        && src.dtype() == DType::F32
        && dst.shape() == src.shape()
        && dst.is_contiguous()
        && src.is_contiguous()
        && !dst.same_storage(src)
    {
        src.with_f32_slice(|s| dst.update_f32(|d| d.copy_from_slice(s)))??;
        return Ok(());
    }
    dst.copy_from(src)?;
    Ok(())
}

/// True if every element of a CPU `F32` tensor is finite.
fn cpu_f32_all_finite(t: &Tensor) -> Result<bool> {
    Ok(t.with_f32_slice(|s| s.iter().all(|v| v.is_finite()))?)
}

/// True if every element of every non-empty device `F32` tensor in `bufs` is finite.
fn device_f32_all_finite(bufs: &[&Tensor]) -> Result<bool> {
    if bufs.is_empty() {
        return Ok(true);
    }
    let _g = no_grad();
    // Fast path: all buffers on one device, contiguous from storage start -> multi-tensor AbsMax kernel
    // (16 buffers per GPU dispatch, no cat/mul intermediate allocations; propagates both NaN and +Inf).
    if let Some(partials) = Tensor::norm_partials_many(bufs, NormReduce::AbsMax, None)? {
        return Ok(partials.max_all()?.item::<f32>()?.is_finite());
    }
    // Fallback: group by device (in case buffers span multiple devices) and check Σ 0 · x == 0.
    let mut remaining: Vec<&Tensor> = bufs.to_vec();
    while !remaining.is_empty() {
        let dev = remaining[0].device();
        let mut group = Vec::new();
        let mut next = Vec::new();
        for t in remaining {
            if t.device() == dev {
                group.push(t);
            } else {
                next.push(t);
            }
        }
        remaining = next;
        if let Some(partials) = Tensor::norm_partials_many(&group, NormReduce::AbsMax, None)? {
            if !partials.max_all()?.item::<f32>()?.is_finite() {
                return Ok(false);
            }
            continue;
        }
        let flat: Vec<Tensor> = group.iter().map(|b| Ok(b.flatten_all()?)).collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = flat.iter().collect();
        if Tensor::cat(&refs, 0)?.mul_scalar(0.0)?.sum_all()?.item::<f32>()? != 0.0 {
            return Ok(false);
        }
    }
    Ok(true)
}
