//! Autocast's cached `F16` copies of `F32` parameters ([`Tensor::autocast_f16`], [`Tensor::refresh_autocast_f16`]).
//!
//! An autocast conv uses an `F16` copy of its `F32` weight and bias. Each copy is kept on the parameter's storage
//! with the storage version it was cast at, so it is cast again only after the parameter changed (every write
//! bumps the version), and an optimizer refreshes the copies of all the parameters it updated in a few
//! multi-tensor casts right after its step instead of one cast per conv in the next forward. A refresh makes new
//! storages: a graph that still holds the previous copy keeps it.

use crate::{DType, Result, Tensor, no_grad};

impl Tensor {
    /// The `F16` copy autocast uses for this `F32` parameter: the cached one while this storage is unchanged since
    /// it was cast, else a fresh cast (cached for next time). Only dense tensors that start at element 0 of a
    /// device storage are cached; anything else is cast every time. Not recorded (taken under `no_grad`).
    pub(crate) fn autocast_f16(&self) -> Result<Tensor> {
        let _g = no_grad();
        if !self.caches_f16() {
            return self.to_dtype(DType::F16);
        }
        let cell = self.cell();
        let version = cell.version();
        if let Some((v, t)) = &*cell.half.lock().unwrap_or_else(|e| e.into_inner())
            && *v == version
            && t.shape() == self.shape()
        {
            return Ok(t.clone());
        }
        let t = self.to_dtype(DType::F16)?;
        *cell.half.lock().unwrap_or_else(|e| e.into_inner()) = Some((version, t.clone()));
        Ok(t)
    }

    fn caches_f16(&self) -> bool {
        self.dtype() == DType::F32
            && !self.device().is_cpu()
            && self.layout().contiguous_range() == Some(0..self.numel())
    }

    /// Recasts the stale autocast `F16` copies ([`Tensor::autocast_f16`]) of `params`, all of them in one backend
    /// call where the backend has one (the GPU: 16 tensors per dispatch), else one cast each. Parameters without
    /// a copy (never used under autocast) are skipped, so an optimizer calls this after every step at no cost
    /// outside mixed precision.
    pub fn refresh_autocast_f16(params: &[&Tensor]) -> Result<()> {
        let _g = no_grad();
        let stale: Vec<&Tensor> = params
            .iter()
            .copied()
            .filter(|p| {
                p.caches_f16()
                    && p.cell()
                        .half
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .as_ref()
                        .is_some_and(|(v, t)| *v != p.cell().version() && t.shape() == p.shape())
            })
            .collect();
        let Some(first) = stale.first() else { return Ok(()) };
        let versions: Vec<u64> = stale.iter().map(|p| p.cell().version()).collect();
        let fresh: Vec<Tensor> = if stale.iter().all(|p| p.device() == first.device()) {
            let b = first.backend()?;
            let guards: Vec<_> = stale.iter().map(|p| p.cell().read()).collect();
            let views: Vec<crate::backend::View<'_>> =
                guards.iter().zip(&stale).map(|(g, p)| crate::backend::View::new(g, p.layout())).collect();
            match b.cast_f16_many(&views) {
                Ok(ss) => {
                    drop(views);
                    drop(guards);
                    ss.into_iter()
                        .zip(&stale)
                        .map(|(s, p)| Tensor::wrap(s, p.shape().clone(), "to_dtype"))
                        .collect::<Result<_>>()?
                }
                Err(crate::Error::NotSupported { .. }) => {
                    drop(views);
                    drop(guards);
                    stale.iter().map(|p| p.to_dtype(DType::F16)).collect::<Result<_>>()?
                }
                Err(e) => return Err(e),
            }
        } else {
            stale.iter().map(|p| p.to_dtype(DType::F16)).collect::<Result<_>>()?
        };
        for ((p, v), t) in stale.iter().zip(versions).zip(fresh) {
            *p.cell().half.lock().unwrap_or_else(|e| e.into_inner()) = Some((v, t));
        }
        Ok(())
    }
}
