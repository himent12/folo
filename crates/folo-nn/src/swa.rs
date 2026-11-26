//! Stochastic Weight Averaging (SWA; Izmailov et al. 2018).
//!
//! Maintains an equal-weight running arithmetic mean of `n` model snapshots:
//!
//! ```text
//! theta_SWA^(1) = theta^(1)
//! theta_SWA^(n) = theta_SWA^(n - 1) + (theta^(n) - theta_SWA^(n - 1)) / n     (n >= 2)
//! ```
//!
//! ## Precision & BatchNorm contract
//! - On CPU (both contiguous and strided destinations), each step uses the incremental Welford update in `f64`
//!   before storing back to `f32`, so constant weights remain bit-identical across arbitrarily many updates
//!   (zero `((n - 1)/n + 1/n)` drift) while very long runs with sub-ULP increments are bounded by `f32` storage
//!   rounding at each step.
//! - On GPU devices, updates use the fused multi-tensor `OptimStep::Axpby` kernel (`a = (n - 1)/n`, `b = 1/n`
//!   in `f32`).
//! - All tracked tensors must be `F32`; like the original SWA procedure (Izmailov et al. 2018 §3.2), BatchNorm
//!   running statistics should be recomputed with a forward pass over the training set using the final averaged
//!   weights before inference.

use std::collections::HashSet;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor, no_grad};

use crate::module::{Module, TensorKind};
use crate::optim::storage_key;
use crate::{Error, Result};

struct Tracked {
    name: String,
    tensor: Tensor,
    kind: TensorKind,
}

#[inline]
fn is_tracked(kind: TensorKind, tensor: &Tensor) -> bool {
    kind == TensorKind::Parameter || tensor.dtype() == DType::F32
}

/// An equal-weight Stochastic Weight Average (Izmailov et al. 2018) of a model `M`.
///
/// The averaged model is kept in eval mode with `requires_grad = false`. Before any [`SwaModel::update`], it
/// holds a copy of `source` with `num_averaged() == 0`. The first [`SwaModel::update`] copies `source` cleanly
/// (`num_averaged() == 1`), and each subsequent update folds `source` into the running arithmetic mean.
pub struct SwaModel<M: Module> {
    module: M,
    cached: Vec<Tracked>,
    num_averaged: u64,
}

impl<M: Module> SwaModel<M> {
    /// Create an SWA container initialized from `source` (`num_averaged == 0`), moved to `source`'s device
    /// and placed in eval mode with gradients disabled.
    ///
    /// # Errors
    /// Returns [`Error::Invalid`] if `module` and `source` have mismatched tracked tensors, non-`F32` tracked
    /// parameters, or share any underlying tensor storage (preventing accidental mutation of `source`'s
    /// weights or gradient flags).
    pub fn new(mut module: M, source: &dyn Module) -> Result<Self> {
        let pre_cached = collect_cached(&module);
        let src_tracked: Vec<_> =
            source.named_tensors().into_iter().filter(|t| is_tracked(t.kind, &t.tensor)).collect();
        let dst_tracked: Vec<_> = pre_cached.iter().filter(|c| is_tracked(c.kind, &c.tensor)).collect();

        // Validate schema, F32 dtype, and disjoint storage BEFORE touching requires_grad or moving devices.
        if src_tracked.len() != dst_tracked.len() {
            return Err(Error::Invalid("SwaModel: source tensors do not match averaged model".into()));
        }
        let src_keys: HashSet<usize> = src_tracked.iter().map(|t| storage_key(&t.tensor)).collect();
        for (dst, src) in dst_tracked.iter().zip(&src_tracked) {
            if src.name != dst.name
                || src.tensor.shape() != dst.tensor.shape()
                || src.tensor.dtype() != DType::F32
                || dst.tensor.dtype() != DType::F32
            {
                return Err(Error::Invalid(format!(
                    "SwaModel: incompatible tensor `{}` ({:?} {}) vs `{}` ({:?} {}), expected matching F32",
                    src.name,
                    src.tensor.dims(),
                    src.tensor.dtype(),
                    dst.name,
                    dst.tensor.dims(),
                    dst.tensor.dtype(),
                )));
            }
            if src_keys.contains(&storage_key(&dst.tensor)) {
                return Err(Error::Invalid(format!(
                    "SwaModel: destination tensor `{}` shares storage with source model",
                    dst.name
                )));
            }
        }

        if let Some(dev) = first_device(source) {
            module.to_device(dev)?;
        }
        module.eval();
        let cached = collect_cached(&module);
        for c in &cached {
            c.tensor.set_requires_grad(false);
        }
        let mut swa = Self { module, cached, num_averaged: 0 };
        swa.copy_from_source(source)?;
        Ok(swa)
    }

    /// The averaged model (always in eval mode).
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Number of model snapshots folded into the average so far.
    pub fn num_averaged(&self) -> u64 {
        self.num_averaged
    }

    /// Reset the average back to `source` with `num_averaged == 0`.
    pub fn reset(&mut self, source: &dyn Module) -> Result<()> {
        self.copy_from_source(source)?;
        self.num_averaged = 0;
        Ok(())
    }

    /// Move the averaged model to `device` in place.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.module.to_device(device)?;
        self.cached = collect_cached(&self.module);
        Ok(())
    }

    /// Fold one snapshot of `source` into the running arithmetic mean.
    ///
    /// Validates every tensor's name, shape, `F32` dtype, and device before mutating any destination tensor,
    /// and snapshots any source tensor whose storage aliases a destination tensor (including cross-tensor
    /// aliasing and offset/transposed views).
    pub fn update(&mut self, source: &dyn Module) -> Result<()> {
        let next_n = self
            .num_averaged
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("SwaModel: snapshot counter overflow".into()))?;
        let pairs = match_and_snapshot_source(source, &self.cached)?;
        let _g = no_grad();

        if next_n == 1 {
            for (idx, src) in pairs {
                self.cached[idx].tensor.copy_from(&src)?;
            }
            self.num_averaged = 1;
            return Ok(());
        }

        let inv_n = 1.0f64 / (next_n as f64);
        let a32 = ((next_n - 1) as f64 / next_n as f64) as f32;
        let b32 = inv_n as f32;

        // Multi-tensor device fast path when all destinations are distinct storages on a non-CPU backend.
        let any_device = pairs.iter().any(|&(idx, _)| !self.cached[idx].tensor.device().is_cpu());
        if any_device {
            let mut dst_keys = HashSet::with_capacity(pairs.len());
            let dsts_unique =
                pairs.iter().all(|&(idx, _)| dst_keys.insert(storage_key(&self.cached[idx].tensor)));
            if dsts_unique {
                let dsts: Vec<&Tensor> = pairs.iter().map(|&(idx, _)| &self.cached[idx].tensor).collect();
                let srcs: Vec<&Tensor> = pairs.iter().map(|(_, s)| s).collect();
                let none = vec![Vec::new(); dsts.len()];
                if Tensor::optimizer_step_many_(&dsts, &srcs, &none, &OptimStep::Axpby { a: a32, b: b32 })? {
                    self.num_averaged = next_n;
                    return Ok(());
                }
            }
        }

        for (idx, src) in pairs {
            let dst = &self.cached[idx].tensor;
            if dst.device().is_cpu() {
                if dst.is_contiguous() {
                    src.with_f32_slice(|ms| {
                        dst.update_f32(|es| {
                            for (e, &m) in es.iter_mut().zip(ms) {
                                let e64 = *e as f64;
                                *e = (e64 + (m as f64 - e64) * inv_n) as f32;
                            }
                        })
                    })??;
                } else {
                    // Non-contiguous CPU destination: exact f64 Welford update matching contiguous CPU.
                    let mut dst_v = dst.to_vec::<f32>()?;
                    let src_v = src.to_vec::<f32>()?;
                    for (e, &m) in dst_v.iter_mut().zip(&src_v) {
                        let e64 = *e as f64;
                        *e = (e64 + (m as f64 - e64) * inv_n) as f32;
                    }
                    let updated = Tensor::from_vec(dst_v, dst.shape().clone())?;
                    dst.copy_from(&updated)?;
                }
            } else if dst.optimizer_step_(&src, &[], &OptimStep::Axpby { a: a32, b: b32 })? {
                continue;
            } else {
                dst.mul_scalar_assign(a32)?;
                dst.add_scaled_assign(&src, b32)?;
            }
        }
        self.num_averaged = next_n;
        Ok(())
    }

    fn copy_from_source(&mut self, source: &dyn Module) -> Result<()> {
        let pairs = match_and_snapshot_source(source, &self.cached)?;
        let _g = no_grad();
        for (idx, src) in pairs {
            self.cached[idx].tensor.copy_from(&src)?;
        }
        Ok(())
    }
}

fn collect_cached(m: &dyn Module) -> Vec<Tracked> {
    m.named_tensors().into_iter().map(|t| Tracked { name: t.name, tensor: t.tensor, kind: t.kind }).collect()
}

fn first_device(m: &dyn Module) -> Option<Device> {
    let mut found = None;
    m.visit_tensors(&mut |_, t, _| {
        if found.is_none() {
            found = Some(t.device());
        }
    });
    if found.is_some() {
        return found;
    }
    m.visit_children(&mut |_, child| {
        if found.is_none() {
            found = first_device(child);
        }
    });
    found
}

/// Validate every tracked pair up front (count, name, shape, `F32` dtype, and matching device) and snapshot
/// any source tensor whose storage aliases any destination tensor so in-place updates never deadlock or
/// corrupt unread source views.
fn match_and_snapshot_source(source: &dyn Module, cached: &[Tracked]) -> Result<Vec<(usize, Tensor)>> {
    let src_list: Vec<_> =
        source.named_tensors().into_iter().filter(|t| is_tracked(t.kind, &t.tensor)).collect();
    let dst_list: Vec<(usize, &Tracked)> =
        cached.iter().enumerate().filter(|(_, c)| is_tracked(c.kind, &c.tensor)).collect();
    if src_list.len() != dst_list.len() {
        return Err(Error::Invalid("SwaModel: source tensors do not match averaged model".into()));
    }
    let dst_keys: HashSet<usize> = dst_list.iter().map(|(_, d)| storage_key(&d.tensor)).collect();
    let mut out = Vec::with_capacity(dst_list.len());
    for ((idx, dst), src) in dst_list.into_iter().zip(src_list) {
        if src.name != dst.name || src.tensor.shape() != dst.tensor.shape() {
            return Err(Error::Invalid(format!(
                "SwaModel: mismatch at `{}` {:?} vs `{}` {:?}",
                src.name,
                src.tensor.dims(),
                dst.name,
                dst.tensor.dims(),
            )));
        }
        if src.tensor.dtype() != DType::F32 || dst.tensor.dtype() != DType::F32 {
            return Err(Error::Invalid(format!(
                "SwaModel: `{}` has dtype {} (destination {}), expected f32",
                src.name,
                src.tensor.dtype(),
                dst.tensor.dtype(),
            )));
        }
        if src.tensor.device() != dst.tensor.device() {
            return Err(Error::Invalid(format!(
                "SwaModel: device mismatch at `{}`: source {:?} vs SWA {:?} (call SwaModel::to_device first)",
                src.name,
                src.tensor.device(),
                dst.tensor.device(),
            )));
        }
        let src_t = if dst_keys.contains(&storage_key(&src.tensor)) {
            let _g = no_grad();
            let snap = Tensor::zeros_with(src.tensor.shape(), DType::F32, src.tensor.device())?;
            snap.copy_from(&src.tensor)?;
            snap
        } else {
            src.tensor
        };
        out.push((idx, src_t));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchNorm2d, Linear, ModuleList};
    use folo_core::rng::Rng;

    #[test]
    fn swa_exact_arithmetic_mean_and_constant_invariance() {
        let mut rng = Rng::seed(7);
        let bn_live = BatchNorm2d::new(4);
        let swa_bn = SwaModel::new(BatchNorm2d::new(4), &bn_live).unwrap();
        assert!(!swa_bn.module().is_training());

        let live = Linear::new(3, 2, true, &mut rng);
        let mut swa = SwaModel::new(Linear::new(3, 2, true, &mut Rng::seed(99)), &live).unwrap();
        assert_eq!(swa.num_averaged(), 0);

        let snapshots = [
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ];
        let mut sum = [0.0f64; 6];
        for (k, snap) in snapshots.iter().enumerate() {
            {
                let _g = no_grad();
                live.weight.copy_from(&Tensor::from_slice(snap, [2, 3]).unwrap()).unwrap();
            }
            swa.update(&live).unwrap();
            assert_eq!(swa.num_averaged(), (k + 1) as u64);
            for i in 0..6 {
                sum[i] += snap[i] as f64;
            }
            let got = swa.module().weight.to_vec::<f32>().unwrap();
            for i in 0..6 {
                let want = (sum[i] / (k + 1) as f64) as f32;
                assert!((got[i] - want).abs() < 1e-6, "step {k} idx {i}: got {} vs {want}", got[i]);
            }
        }

        // Constant weight remains bit-identical across 50 updates (zero drift).
        {
            let _g = no_grad();
            live.weight.copy_from(&Tensor::from_slice(&[0.3f32; 6], [2, 3]).unwrap()).unwrap();
        }
        swa.reset(&live).unwrap();
        assert_eq!(swa.num_averaged(), 0);
        for _ in 0..50 {
            swa.update(&live).unwrap();
        }
        assert_eq!(swa.num_averaged(), 50);
        assert_eq!(swa.module().weight.to_vec::<f32>().unwrap(), vec![0.3f32; 6]);

        // Mismatched model rejected.
        let other = Linear::new(4, 2, true, &mut rng);
        assert!(swa.update(&other).is_err());
    }

    #[test]
    fn swa_rejects_shared_storage_in_new_and_handles_views_and_f16() {
        let mut rng = Rng::seed(11);
        let live = Linear::new(2, 2, false, &mut rng);
        assert!(live.weight.requires_grad());

        // Sharing storage with live in SwaModel::new must fail WITHOUT disabling live's requires_grad!
        let aliased_dst = Linear { weight: live.weight.clone(), bias: None };
        assert!(SwaModel::new(aliased_dst, &live).is_err());
        assert!(live.weight.requires_grad(), "failed SwaModel::new must not mutate live requires_grad");

        // Transposed view sharing destination storage during update must be snapshotted and averaged, not skipped.
        let dst = Linear::new(2, 2, false, &mut Rng::seed(22));
        let mut swa = SwaModel::new(dst, &live).unwrap();
        {
            let _g = no_grad();
            live.weight.copy_from(&Tensor::from_slice(&[1.0f32, 3.0, 5.0, 7.0], [2, 2]).unwrap()).unwrap();
        }
        swa.update(&live).unwrap();
        // Update using the transposed view of SWA's own weight `[1, 5, 3, 7]` -> average is `[1, 4, 4, 7]`.
        let transposed_src = Linear { weight: swa.module().weight.t().unwrap(), bias: None };
        swa.update(&transposed_src).unwrap();
        assert_eq!(swa.num_averaged(), 2);
        assert_eq!(swa.module().weight.to_vec::<f32>().unwrap(), vec![1.0, 4.0, 4.0, 7.0]);

        // Cross-tensor aliasing in a 2-layer module: src[0] aliases dst[1] and src[1] aliases dst[0].
        let m_live = ModuleList::new(vec![
            Linear::new(2, 2, false, &mut Rng::seed(31)),
            Linear::new(2, 2, false, &mut Rng::seed(32)),
        ]);
        let m_dst = ModuleList::new(vec![
            Linear::new(2, 2, false, &mut Rng::seed(33)),
            Linear::new(2, 2, false, &mut Rng::seed(34)),
        ]);
        let mut swa_list = SwaModel::new(m_dst, &m_live).unwrap();
        {
            let _g = no_grad();
            m_live[0].weight.copy_from(&Tensor::from_slice(&[2.0f32; 4], [2, 2]).unwrap()).unwrap();
            m_live[1].weight.copy_from(&Tensor::from_slice(&[6.0f32; 4], [2, 2]).unwrap()).unwrap();
        }
        swa_list.update(&m_live).unwrap();
        let swapped = ModuleList::new(vec![
            Linear { weight: swa_list.module()[1].weight.clone(), bias: None },
            Linear { weight: swa_list.module()[0].weight.clone(), bias: None },
        ]);
        swa_list.update(&swapped).unwrap();
        assert_eq!(swa_list.module()[0].weight.to_vec::<f32>().unwrap(), vec![4.0; 4]);
        assert_eq!(swa_list.module()[1].weight.to_vec::<f32>().unwrap(), vec![4.0; 4]);

        // Non-contiguous destination tensor averages cleanly and matches contiguous arithmetic.
        let base_buf = Tensor::zeros([2, 2]);
        let non_contig_dst = Linear { weight: base_buf.t().unwrap(), bias: None };
        let mut swa_nc = SwaModel::new(non_contig_dst, &live).unwrap();
        swa_nc.update(&live).unwrap();
        {
            let _g = no_grad();
            live.weight.copy_from(&Tensor::from_slice(&[3.0f32, 5.0, 7.0, 9.0], [2, 2]).unwrap()).unwrap();
        }
        swa_nc.update(&live).unwrap();
        assert_eq!(swa_nc.module().weight.to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

        // F16 parameters rejected up front in SwaModel::new (never pass new and fail mid-update).
        let f16_live =
            Linear { weight: Tensor::zeros_with([2, 2], DType::F16, Device::Cpu).unwrap(), bias: None };
        let f16_dst =
            Linear { weight: Tensor::zeros_with([2, 2], DType::F16, Device::Cpu).unwrap(), bias: None };
        assert!(SwaModel::new(f16_dst, &f16_live).is_err());
    }
}
