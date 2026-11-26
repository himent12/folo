//! Stochastic Weight Averaging (SWA; Izmailov et al. 2018).
//!
//! Maintains an exact equal-weight arithmetic mean of `n` model snapshots:
//!
//! ```text
//! theta_SWA^(1) = theta^(1)
//! theta_SWA^(n) = theta_SWA^(n - 1) + (theta^(n) - theta_SWA^(n - 1)) / n     (n >= 2)
//! ```
//!
//! Using the incremental Welford form in `f64` on CPU guarantees constant weights remain bit-identical across
//! arbitrarily many updates (zero `((n - 1)/n + 1/n)` drift) and avoids overflow on large snapshot counts.
//! Tracks every parameter and `F32` buffer of `M` (same tensor selection and aliasing safety as
//! [`crate::ModelEma`]).

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor, no_grad};

use crate::module::{Module, TensorKind};
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
/// (`num_averaged() == 1`), and each subsequent update folds `source` into the exact arithmetic mean.
pub struct SwaModel<M: Module> {
    module: M,
    cached: Vec<Tracked>,
    num_averaged: u64,
}

impl<M: Module> SwaModel<M> {
    /// Create an SWA container initialized from `source` (`num_averaged == 0`), moved to `source`'s device
    /// and placed in eval mode with gradients disabled.
    pub fn new(mut module: M, source: &dyn Module) -> Result<Self> {
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
    pub fn update(&mut self, source: &dyn Module) -> Result<()> {
        let next_n = self
            .num_averaged
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("SwaModel: snapshot counter overflow".into()))?;
        let pairs = match_source(source, &self.cached)?;
        let _g = no_grad();

        if next_n == 1 {
            for (idx, src) in pairs {
                let dst = &self.cached[idx].tensor;
                if !dst.same_storage(&src) {
                    dst.copy_from(&src)?;
                }
            }
            self.num_averaged = 1;
            return Ok(());
        }

        let inv_n = 1.0f64 / (next_n as f64);
        let a32 = ((next_n - 1) as f64 / next_n as f64) as f32;
        let b32 = inv_n as f32;

        // Device multi-tensor fast path when on a non-CPU backend with no storage aliasing.
        let any_device = pairs.iter().any(|&(idx, _)| !self.cached[idx].tensor.device().is_cpu());
        let no_alias = pairs.iter().all(|(idx, src)| !self.cached[*idx].tensor.same_storage(src));
        if any_device && no_alias {
            let dsts: Vec<&Tensor> = pairs.iter().map(|&(idx, _)| &self.cached[idx].tensor).collect();
            let srcs: Vec<&Tensor> = pairs.iter().map(|(_, s)| s).collect();
            let none = vec![Vec::new(); dsts.len()];
            if Tensor::optimizer_step_many_(&dsts, &srcs, &none, &OptimStep::Axpby { a: a32, b: b32 })? {
                self.num_averaged = next_n;
                return Ok(());
            }
        }

        for (idx, src) in pairs {
            let dst = &self.cached[idx].tensor;
            if dst.same_storage(&src) {
                continue;
            }
            if dst.device().is_cpu()
                && src.device().is_cpu()
                && dst.dtype() == DType::F32
                && src.dtype() == DType::F32
                && dst.is_contiguous()
            {
                src.with_f32_slice(|ms| {
                    dst.update_f32(|es| {
                        for (e, &m) in es.iter_mut().zip(ms) {
                            let e64 = *e as f64;
                            *e = (e64 + (m as f64 - e64) * inv_n) as f32;
                        }
                    })
                })??;
            } else if !dst.device().is_cpu()
                && dst.optimizer_step_(&src, &[], &OptimStep::Axpby { a: a32, b: b32 })?
            {
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
        let pairs = match_source(source, &self.cached)?;
        let _g = no_grad();
        for (idx, src) in pairs {
            let dst = &self.cached[idx].tensor;
            if !dst.same_storage(&src) {
                dst.copy_from(&src)?;
            }
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

fn match_source(source: &dyn Module, cached: &[Tracked]) -> Result<Vec<(usize, Tensor)>> {
    let src_list: Vec<_> =
        source.named_tensors().into_iter().filter(|t| is_tracked(t.kind, &t.tensor)).collect();
    let dst_list: Vec<(usize, &Tracked)> =
        cached.iter().enumerate().filter(|(_, c)| is_tracked(c.kind, &c.tensor)).collect();
    if src_list.len() != dst_list.len() {
        return Err(Error::Invalid("SwaModel: source tensors do not match averaged model".into()));
    }
    let mut out = Vec::with_capacity(dst_list.len());
    for ((idx, dst), src) in dst_list.into_iter().zip(src_list) {
        if src.name != dst.name
            || src.tensor.shape() != dst.tensor.shape()
            || src.tensor.dtype() != dst.tensor.dtype()
        {
            return Err(Error::Invalid(format!("SwaModel: mismatch at `{}` vs `{}`", src.name, dst.name)));
        }
        out.push((idx, src.tensor));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchNorm2d, Linear};
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

        // Feed 4 known weight snapshots; verify exact arithmetic mean after every step.
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
}
