//! MuSGD's fused Muon update ([`Tensor::muon_step_many_`]), behind the optional `Backend::muon_step_many` hook.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::autograd::is_grad_enabled;
use crate::backend::View;
use crate::{DType, Error, Result, Storage, Tensor};

/// One matrix (or depthwise channel bank) of [`crate::backend::Backend::muon_step_many`].
#[derive(Debug)]
pub struct MuonItem<'a> {
    /// Parameter, updated in place.
    pub param: &'a mut Storage,
    /// Muon momentum buffer (`rows · cols` `F32`, already updated this step).
    pub momentum: View<'a>,
    /// Gradient (`rows · cols` `F32`).
    pub grad: View<'a>,
    /// Rows (first dimension).
    pub rows: usize,
    /// Columns (product of remaining dimensions).
    pub cols: usize,
    /// Factor of the orthogonalised direction.
    pub scale: f32,
    /// Exact per-row block-diagonal Polar LMO (depthwise convs; skips matrix orthogonalisation).
    pub depthwise: bool,
}

/// Maximum odd-quintic orthogonalisation steps a single [`MuonSettings`] holds.
pub const MAX_MUON_STEPS: usize = 8;

/// The settings of [`crate::backend::Backend::muon_step_many`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuonSettings {
    /// `μ`.
    pub momentum: f32,
    /// Nesterov: `u = (g · (1 − μ)) · 1 + m · μ`; else `u = m`.
    pub nesterov: bool,
    /// Per-step odd-quintic coefficients `(a_k, b_k, c_k)` for `k < steps`.
    pub step_coeffs: [(f32, f32, f32); MAX_MUON_STEPS],
    /// Number of polynomial steps (`<= MAX_MUON_STEPS`).
    pub steps: usize,
    /// Step-0 Schatten-4 spectral rescaling.
    pub schatten4: bool,
    /// Added to Frobenius / row norm before division.
    pub eps: f32,
    /// Learning rate (`p += (−lr) · scale · O`).
    pub lr: f32,
}

impl MuonSettings {
    /// Per-step schedule `schedule` (`len <= MAX_MUON_STEPS`) with optional step-0 Schatten-4 rescaling.
    pub fn from_schedule(
        mu: f32,
        nesterov: bool,
        schedule: &[(f32, f32, f32)],
        schatten4: bool,
        eps: f32,
        lr: f32,
    ) -> Self {
        let mut step_coeffs = [(0.0, 0.0, 0.0); MAX_MUON_STEPS];
        let n = schedule.len().min(MAX_MUON_STEPS);
        step_coeffs[..n].copy_from_slice(&schedule[..n]);
        MuonSettings { momentum: mu, nesterov, step_coeffs, steps: schedule.len(), schatten4, eps, lr }
    }
}

impl Tensor {
    /// MuSGD's Muon update of many matrix-shaped parameters in one backend call
    /// ([`crate::backend::Backend::muon_step_many`]): for each parameter `params[i]`, its Muon momentum buffer
    /// `momenta[i]` (already updated this step) and gradient `grads[i]` (same shape, rank ≥ 2, viewed as
    /// `[d₀, rest]`), the Nesterov mix (or the buffer itself), orthogonalised by `settings` (`steps` odd-quintic
    /// steps with `step_coeffs` and optional step-0 Schatten-4 spectral rescaling, matching
    /// `folo_nn::optim::orthogonalize_with`), times `scales[i]`, then `p = p · 1 + O · (−lr)` in place. Returns
    /// `Ok(false)`, touching nothing, when the backend has no such kernel or a tensor does not fit (all `F32` on
    /// one device, parameters and momenta dense from the start of their own storages, no parameter storage used
    /// twice or shared with a momentum or gradient); the caller then uses tensor ops. The products are summed in
    /// another order than the tensor ops' matmuls, so results agree to rounding, not bit for bit.
    pub fn muon_step_many_(
        params: &[&Tensor],
        momenta: &[&Tensor],
        grads: &[&Tensor],
        scales: &[f32],
        settings: &MuonSettings,
    ) -> Result<bool> {
        let no_dw = vec![false; params.len()];
        Self::muon_step_many_geo_(params, momenta, grads, scales, &no_dw, settings)
    }

    /// Extended [`Tensor::muon_step_many_`] with per-parameter `depthwise` flags for zero-GEMM per-channel
    /// Polar LMO (`Conv-GeoMuon`).
    pub fn muon_step_many_geo_(
        params: &[&Tensor],
        momenta: &[&Tensor],
        grads: &[&Tensor],
        scales: &[f32],
        depthwise: &[bool],
        settings: &MuonSettings,
    ) -> Result<bool> {
        const OP: &str = "muon_step_many_";
        if params.len() != momenta.len()
            || params.len() != grads.len()
            || params.len() != scales.len()
            || params.len() != depthwise.len()
        {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "{} parameters, {} momenta, {} gradients, {} scales, {} depthwise",
                    params.len(),
                    momenta.len(),
                    grads.len(),
                    scales.len(),
                    depthwise.len()
                ),
            ));
        }
        let Some(first) = params.first() else { return Ok(true) };
        if is_grad_enabled() && params.iter().any(|t| t.requires_grad()) {
            return Err(Error::Autograd(format!(
                "{OP}: in-place update of a tensor that requires grad; run it under no_grad()"
            )));
        }
        let dev = first.device();
        let ok = |t: &Tensor| t.dtype() == DType::F32 && t.device() == dev;
        let dense = |t: &Tensor| t.layout().contiguous_range() == Some(0..t.numel());
        for ((p, m), g) in params.iter().zip(momenta).zip(grads) {
            if p.rank() < 2 || m.shape() != p.shape() || g.shape() != p.shape() {
                return Err(Error::invalid_arg(
                    OP,
                    format!("parameter {:?}, momentum {:?}, gradient {:?}", p.dims(), m.dims(), g.dims()),
                ));
            }
            if !ok(p) || !ok(m) || !ok(g) || !dense(p) || !dense(m) {
                return Ok(false);
            }
        }
        let mut written: HashSet<*const _> = HashSet::with_capacity(params.len());
        for t in params {
            if !written.insert(Arc::as_ptr(t.cell())) {
                return Ok(false);
            }
        }
        if momenta.iter().chain(grads).any(|t| written.contains(&Arc::as_ptr(t.cell()))) {
            return Ok(false);
        }
        let b = first.backend()?;
        // One read guard per read storage (taking a lock twice on one thread can deadlock).
        let mut slot = HashMap::new();
        let mut guards = Vec::new();
        let mut ids = Vec::with_capacity(params.len());
        for (&m, &g) in momenta.iter().zip(grads) {
            let mut pair = [0; 2];
            for (k, t) in [m, g].into_iter().enumerate() {
                pair[k] = *slot.entry(Arc::as_ptr(t.cell())).or_insert_with(|| {
                    guards.push(t.cell().read());
                    guards.len() - 1
                });
            }
            ids.push((pair[0], pair[1]));
        }
        let mut pw: Vec<_> = params.iter().map(|p| p.cell().write_unversioned()).collect();
        let mut items: Vec<MuonItem<'_>> = pw
            .iter_mut()
            .zip(&ids)
            .zip(params.iter().zip(momenta.iter().zip(grads)))
            .zip(scales.iter().zip(depthwise))
            .map(|(((pg, &(mi, gi)), (p, (m, g))), (&scale, &dw))| {
                let rows = p.dims()[0];
                MuonItem {
                    param: pg,
                    momentum: View::new(&guards[mi], m.layout()),
                    grad: View::new(&guards[gi], g.layout()),
                    rows,
                    cols: p.numel() / rows.max(1),
                    scale,
                    depthwise: dw,
                }
            })
            .collect();
        let r = b.muon_step_many(&mut items, settings);
        // Bump versions only if the backend may have written, so a fallback keeps existing graphs valid.
        if !matches!(&r, Err(Error::NotSupported { .. })) {
            for t in params {
                t.cell().mark_modified();
            }
        }
        match r {
            Ok(()) => Ok(true),
            Err(Error::NotSupported { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }
}
