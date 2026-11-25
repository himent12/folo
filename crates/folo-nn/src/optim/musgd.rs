//! MuSGD: SGD plus a Muon (orthogonalised momentum) update on every matrix-shaped weight, as Ultralytics 8.4's
//! `MuSGD` (docs/MODEL.md §8.2), with optional Polar-FOLO per-step minimax Remez orthogonalisation schedules and
//! step-0 Schatten-4 spectral rescaling.

use std::collections::HashMap;

use folo_core::backend::{MuonSettings, OptimStep};
use folo_core::{DType, Device, Tensor};

use serde::{Deserialize, Serialize};

use super::ortho::{MuonGeoMode, MuonOrtho, NS_EPS, muon_scale, muon_scale_geo, orthogonalize_many_geo};
use super::simd::{
    axpby_slice, muon_dw_polar_fused_slice, muon_dw_polar_step_slice, muon_momentum_nesterov_slice,
};
use super::{
    GradSource, OptimContract, OptimKind, Optimizer, PAR_THRESHOLD, ParamGroup, Sgd, apply_group_contract,
    cpu_fast, group_contract, is_depthwise_conv_shape, take_state,
};
use crate::{Error, Result};

/// Weight of the Muon update: `p −= MUON_SCALE · lr · O`.
pub const MUON_SCALE: f64 = 0.2;

/// Whether `name` is a Muon group of [`MuSgd`]: `decay` (every conv weight) and its `decay.<suffix>` splits.
pub fn is_muon_group(name: &str) -> bool {
    name == "decay" || name.starts_with("decay.")
}

/// MuSGD, docs/MODEL.md §8.2. Every group takes the [`Sgd`] update (Nesterov momentum `momentum_buffer`, coupled
/// weight decay) at the group's `lr`. Before it, each 2-D or 4-D parameter of a Muon group ([`is_muon_group`])
/// with gradient `g` and the group's momentum `μ` also takes
/// ```text
/// m = μ · m + (1 − μ) · g                       (muon_buffer, starts at zero)
/// u = μ · m + (1 − μ) · g                       (Nesterov; plain: u = m)
/// O = Ortho(u as [rows, cols]) · sqrt(max(1, d₋₂ / d₋₁))
/// p = p − 0.2 · lr · O                         (0.2: MuSgdWeights::muon)
/// ```
/// where the matrix is `u.flatten(1)` (a conv weight `[c_out, c_in·k·k]`), `d₋₂ / d₋₁` are the parameter's last
/// two dimensions (1 for a square kernel) and `Ortho` is [`orthogonalize_with`] (default [`MuonOrtho::Polar4`]).
/// The SGD update then reads the parameter after the Muon step (its weight decay too), as the reference's in-place
/// `foreach` ops do.
pub struct MuSgd {
    sgd: Sgd,
    muon: Vec<Vec<Option<Tensor>>>,
    weights: MuSgdWeights,
    ortho: MuonOrtho,
    geo: MuonGeoMode,
}

/// The two factors of a [`MuSgd`] step on a Muon parameter: `p −= muon · lr · O`, then the SGD update at
/// `sgd · lr` (other parameters take SGD at `lr`). The default is Ultralytics 8.4's trainer (`muon = 0.2`,
/// `sgd = 1.0`); YOLO26's published COCO run used 0.528 and 0.674 (`muon_w`, `sgd_w` of its `train_args`).
/// Other weights are saved in a checkpoint's `folo.optim` as `"musgd": {"muon": .., "sgd": ..}` (docs/FORMAT.md).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MuSgdWeights {
    /// Weight of the orthogonalised update.
    pub muon: f64,
    /// Weight of the SGD update of a Muon parameter.
    pub sgd: f64,
}

impl Default for MuSgdWeights {
    fn default() -> Self {
        MuSgdWeights { muon: MUON_SCALE, sgd: 1.0 }
    }
}

impl MuSgdWeights {
    /// An error unless both weights are finite and non-negative.
    pub fn validate(&self) -> Result<()> {
        if [self.muon, self.sgd].iter().all(|w| w.is_finite() && *w >= 0.0) {
            Ok(())
        } else {
            Err(Error::Invalid(format!("musgd: weights must be finite and >= 0, got {self:?}")))
        }
    }
}

impl MuSgd {
    /// An optimizer over `groups` (no parameter in two groups; all `F32`) with the default [`MuSgdWeights`] and
    /// [`MuonOrtho::Polar4`].
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        Self::with_weights(groups, MuSgdWeights::default())
    }

    /// [`MuSgd::new`] with other step weights (finite, non-negative) and the default [`MuonOrtho::Polar4`].
    pub fn with_weights(groups: Vec<ParamGroup>, weights: MuSgdWeights) -> Result<Self> {
        Self::with_ortho(groups, weights, MuonOrtho::default())
    }

    /// [`MuSgd::with_weights`] with a chosen orthogonalisation schedule [`MuonOrtho`].
    pub fn with_ortho(groups: Vec<ParamGroup>, weights: MuSgdWeights, ortho: MuonOrtho) -> Result<Self> {
        Self::with_ortho_and_geo(groups, weights, ortho, MuonGeoMode::Legacy)
    }

    /// [`MuSgd::with_ortho`] with a chosen geometric conditioning mode [`MuonGeoMode`].
    pub fn with_ortho_and_geo(
        groups: Vec<ParamGroup>,
        weights: MuSgdWeights,
        ortho: MuonOrtho,
        geo: MuonGeoMode,
    ) -> Result<Self> {
        weights.validate()?;
        if geo == MuonGeoMode::Geo {
            Self::validate_depthwise(&groups)?;
        }
        let muon = groups.iter().map(|g| vec![None; g.params.len()]).collect();
        let mut s = MuSgd { sgd: Sgd::new(groups)?, muon, weights, ortho, geo };
        s.ensure_buffers()?;
        Ok(s)
    }

    fn validate_depthwise(groups: &[ParamGroup]) -> Result<()> {
        for g in groups {
            if !is_muon_group(&g.name) {
                continue;
            }
            for (pi, (name, p)) in g.params.iter().enumerate() {
                if g.is_depthwise(pi) && !is_depthwise_conv_shape(p.dims()) {
                    return Err(Error::Invalid(format!(
                        "muon depthwise parameter `{name}` has shape {:?}, expected depthwise conv weight `[C > 1, 1, kH, kW]`",
                        p.dims()
                    )));
                }
            }
        }
        Ok(())
    }

    /// The step weights.
    pub fn weights(&self) -> MuSgdWeights {
        self.weights
    }

    /// The orthogonalisation schedule.
    pub fn ortho(&self) -> MuonOrtho {
        self.ortho
    }

    /// Set the orthogonalisation schedule.
    pub fn set_ortho(&mut self, ortho: MuonOrtho) {
        self.ortho = ortho;
    }

    /// The geometric conditioning mode (`Conv-GeoMuon`).
    pub fn geo_mode(&self) -> MuonGeoMode {
        self.geo
    }

    /// Set the geometric conditioning mode (`Conv-GeoMuon`).
    pub fn set_geo_mode(&mut self, geo: MuonGeoMode) -> Result<()> {
        if geo == MuonGeoMode::Geo {
            Self::validate_depthwise(self.sgd.groups())?;
        }
        self.geo = geo;
        self.ensure_buffers()
    }

    fn uses_muon(g: &ParamGroup, p: &Tensor) -> bool {
        is_muon_group(&g.name) && matches!(p.rank(), 2 | 4)
    }

    fn ensure_buffers(&mut self) -> Result<()> {
        for (gi, g) in self.sgd.groups().iter().enumerate() {
            for (pi, (_, p)) in g.params.iter().enumerate() {
                if !Self::uses_muon(g, p) {
                    continue;
                }
                if self.muon[gi][pi].is_none() {
                    self.muon[gi][pi] = Some(Tensor::zeros_with(p.shape(), DType::F32, p.device())?);
                }
            }
        }
        Ok(())
    }

    /// The Muon momentum buffer of a parameter, if it has one.
    pub fn muon_buffer(&self, group: usize, param: usize) -> Option<&Tensor> {
        self.muon.get(group)?.get(param)?.as_ref()
    }

    /// The Muon half of a step: momentum, orthogonalisation and `p −= muon · lr · O` for every Muon parameter
    /// with a gradient.
    fn muon_step(&self, grads: &dyn GradSource) -> Result<()> {
        let _g = folo_core::no_grad();
        struct Active<'a> {
            mu: f32,
            nesterov: bool,
            lr: f32,
            ps: Vec<&'a Tensor>,
            bs: Vec<&'a Tensor>,
            dws: Vec<bool>,
            gs: Vec<Tensor>,
        }
        let mut active: Vec<Active<'_>> = Vec::new();
        for (gi, g) in self.sgd.groups().iter().enumerate() {
            let mu = g.momentum as f32;
            let (mut ps, mut bs, mut dws, mut gs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (pi, (name, p)) in g.params.iter().enumerate() {
                let (Some(b), Some(gr)) = (&self.muon[gi][pi], grads.grad(p)) else { continue };
                if gr.shape() != p.shape() {
                    return Err(Error::Invalid(format!(
                        "musgd: gradient of `{name}` is {:?}, parameter is {:?}",
                        gr.dims(),
                        p.dims()
                    )));
                }
                let dw = self.geo != MuonGeoMode::Legacy && g.is_depthwise(pi);
                ps.push(p);
                bs.push(b);
                dws.push(dw);
                gs.push(gr.to_dtype(DType::F32)?);
            }
            if !ps.is_empty() {
                active.push(Active {
                    mu,
                    nesterov: g.nesterov,
                    lr: (self.weights.muon * g.lr) as f32,
                    ps,
                    bs,
                    dws,
                    gs,
                });
            }
        }
        if active.is_empty() {
            return Ok(());
        }
        let scale_of = |p: &Tensor, dw: bool| -> f32 {
            if dw {
                1.0
            } else if self.geo == MuonGeoMode::Legacy {
                muon_scale(p.dims())
            } else {
                muon_scale_geo(p.dims())
            }
        };
        let first_dev = active[0].ps[0].device();
        let same_lr = active.iter().all(|a| a.lr.to_bits() == active[0].lr.to_bits());
        let (step_lr, fold_lr) = if same_lr { (active[0].lr, false) } else { (1.0, true) };
        let no_alias = |active: &[Active<'_>]| -> bool {
            let mut ps: Vec<&Tensor> = Vec::new();
            let mut bs: Vec<&Tensor> = Vec::new();
            let mut gs: Vec<&Tensor> = Vec::new();
            for a in active {
                ps.extend(a.ps.iter().copied());
                bs.extend(a.bs.iter().copied());
                gs.extend(a.gs.iter());
            }
            for (i, p) in ps.iter().enumerate() {
                if ps[i + 1..].iter().any(|q| p.same_storage(q))
                    || bs.iter().any(|b| p.same_storage(b))
                    || gs.iter().any(|g| p.same_storage(g))
                {
                    return false;
                }
            }
            for (i, b) in bs.iter().enumerate() {
                if bs[i + 1..].iter().any(|q| b.same_storage(q)) || gs.iter().any(|g| b.same_storage(g)) {
                    return false;
                }
            }
            true
        };
        let can_fuse_all = self.ortho != MuonOrtho::Ns5
            && active.len() >= 2
            && !first_dev.is_cpu()
            && active.iter().all(|a| {
                a.mu.to_bits() == active[0].mu.to_bits()
                    && a.nesterov == active[0].nesterov
                    && a.ps.iter().all(|p| p.device() == first_dev)
            })
            && no_alias(&active);
        let fused_scales: Option<Vec<f32>> = if can_fuse_all {
            let total: usize = active.iter().map(|a| a.ps.len()).sum();
            let mut scales = Vec::with_capacity(total);
            let mut ok = true;
            'outer: for a in &active {
                for (p, &dw) in a.ps.iter().zip(&a.dws) {
                    let sc = scale_of(p, dw);
                    let s = if fold_lr { sc * a.lr } else { sc };
                    if fold_lr && !s.is_finite() {
                        ok = false;
                        break 'outer;
                    }
                    scales.push(s);
                }
            }
            ok.then_some(scales)
        } else {
            None
        };
        if let Some(scales) = fused_scales {
            let total = scales.len();
            let (mut ps, mut bs, mut dws, mut gr) = (
                Vec::with_capacity(total),
                Vec::with_capacity(total),
                Vec::with_capacity(total),
                Vec::with_capacity(total),
            );
            for a in &active {
                ps.extend_from_slice(&a.ps);
                bs.extend_from_slice(&a.bs);
                dws.extend_from_slice(&a.dws);
                gr.extend(a.gs.iter());
            }
            let (mu, nesterov) = (active[0].mu, active[0].nesterov);
            axpby_many(&bs, &gr, mu, 1.0 - mu)?;
            if muon_fused_with_scales(&ps, &bs, &gr, &scales, &dws, mu, nesterov, step_lr, self.ortho)? {
                return Ok(());
            }
            for a in &active {
                muon_step_tensor_ops(
                    &a.ps, &a.bs, &a.gs, &a.dws, a.mu, a.nesterov, a.lr, self.ortho, self.geo,
                )?;
            }
            return Ok(());
        }
        for a in &active {
            if a.ps.iter().all(|p| p.device().is_cpu()) {
                muon_step_cpu(&a.ps, &a.bs, &a.gs, &a.dws, a.mu, a.nesterov, a.lr, self.ortho, self.geo)?;
                continue;
            }
            let gr: Vec<&Tensor> = a.gs.iter().collect();
            axpby_many(&a.bs, &gr, a.mu, 1.0 - a.mu)?;
            let scales: Vec<f32> = a.ps.iter().zip(&a.dws).map(|(p, &dw)| scale_of(p, dw)).collect();
            if muon_fused_with_scales(&a.ps, &a.bs, &gr, &scales, &a.dws, a.mu, a.nesterov, a.lr, self.ortho)?
            {
                continue;
            }
            muon_step_tensor_ops(&a.ps, &a.bs, &a.gs, &a.dws, a.mu, a.nesterov, a.lr, self.ortho, self.geo)?;
        }
        Ok(())
    }
}

/// Vectorized CPU path for [`MuSgd::muon_step`]: fuses momentum + Nesterov, avoids copying `b` when
/// `!nesterov`, runs depthwise Polar LMO in-place with zero tensor allocations, and uses AVX2 `axpby`.
#[allow(clippy::too_many_arguments)]
fn muon_step_cpu(
    ps: &[&Tensor],
    bs: &[&Tensor],
    gs: &[Tensor],
    dws: &[bool],
    mu: f32,
    nesterov: bool,
    lr: f32,
    ortho: MuonOrtho,
    geo: MuonGeoMode,
) -> Result<()> {
    use rayon::prelude::*;

    let one_minus_mu = 1.0 - mu;
    // Check cross-parameter storage aliasing across `ps`, `bs`, and `gs`: if any gradient aliases any
    // parameter or momentum buffer in this group (e.g. a caller passing another parameter's buffer as a
    // gradient), fall back to the staged path so an in-place update neither deadlocks nor mutates a
    // later parameter's gradient before it is read.
    let grad_clean = (0..ps.len()).all(|i| {
        let (pi, bi) = (ps[i], bs[i]);
        gs.iter().all(|g| !pi.same_storage(g) && !bi.same_storage(g))
    });
    if !grad_clean {
        let gr: Vec<&Tensor> = gs.iter().collect();
        axpby_many(bs, &gr, mu, one_minus_mu)?;
        return muon_step_tensor_ops(ps, bs, gs, dws, mu, nesterov, lr, ortho, geo);
    }
    let is_fast = |i: usize| -> bool {
        let (p, b, g) = (ps[i], bs[i], &gs[i]);
        cpu_fast(&[p, b]) && g.device().is_cpu() && g.dtype() == DType::F32 && !p.same_storage(b)
    };
    let all_fast = (0..ps.len()).all(is_fast);
    let total_numel: usize = ps.iter().map(|p| p.numel()).sum();
    let can_par = all_fast
        && total_numel >= PAR_THRESHOLD
        && ps.len() > 1
        && (0..ps.len()).all(|i| {
            let (pi, bi) = (ps[i], bs[i]);
            (i + 1..ps.len()).all(|j| {
                !pi.same_storage(ps[j])
                    && !pi.same_storage(bs[j])
                    && !bi.same_storage(ps[j])
                    && !bi.same_storage(bs[j])
            })
        });

    let prepare_one = |i: usize| -> Result<Option<(&Tensor, Tensor)>> {
        let (p, b, g, dw) = (ps[i], bs[i], &gs[i], dws[i]);
        if dw {
            let rows = p.dims()[0];
            let cols = p.numel() / rows.max(1);
            g.with_f32_slice(|gs| {
                b.update_f32(|bs| {
                    p.update_f32(|ps| {
                        muon_dw_polar_fused_slice(ps, bs, gs, rows, cols, mu, nesterov, lr, NS_EPS)
                    })
                })
            })???;
            return Ok(None);
        }
        if nesterov {
            let mut u_vec = vec![0.0f32; p.numel()];
            g.with_f32_slice(|gs| b.update_f32(|bs| muon_momentum_nesterov_slice(bs, gs, &mut u_vec, mu)))??;
            Ok(Some((p, Tensor::from_vec(u_vec, p.shape().clone())?)))
        } else {
            g.with_f32_slice(|gs| b.update_f32(|bs| axpby_slice(bs, gs, mu, one_minus_mu)))??;
            Ok(Some((p, b.clone())))
        }
    };

    let mut mat_ps: Vec<&Tensor> = Vec::with_capacity(ps.len());
    let mut mat_us: Vec<Tensor> = Vec::with_capacity(ps.len());
    if can_par {
        let prepared: Vec<Option<(&Tensor, Tensor)>> = folo_core::cpu::in_pool(|| {
            (0..ps.len())
                .into_par_iter()
                .map(|i| {
                    let _g = folo_core::no_grad();
                    prepare_one(i)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for item in prepared.into_iter().flatten() {
            mat_ps.push(item.0);
            mat_us.push(item.1);
        }
    } else {
        for i in 0..ps.len() {
            if is_fast(i) {
                if let Some((p, u)) = prepare_one(i)? {
                    mat_ps.push(p);
                    mat_us.push(u);
                }
            } else {
                axpby_many(&[bs[i]], &[&gs[i]], mu, one_minus_mu)?;
                muon_step_tensor_ops(
                    &[ps[i]],
                    &[bs[i]],
                    std::slice::from_ref(&gs[i]),
                    &[dws[i]],
                    mu,
                    nesterov,
                    lr,
                    ortho,
                    geo,
                )?;
            }
        }
    }
    if mat_ps.is_empty() {
        return Ok(());
    }
    let os = orthogonalize_many_geo(&mat_us, ortho, geo)?;
    let or: Vec<&Tensor> = os.iter().collect();
    axpby_many(&mat_ps, &or, 1.0, -lr)
}

/// Exact zero-GEMM per-channel block-diagonal Polar LMO for depthwise conv weights (CPU / tensor fallback).
fn muon_dw_step_tensor(p: &Tensor, u: &Tensor, lr: f32) -> Result<()> {
    let rows = p.dims()[0];
    let cols = p.numel() / rows.max(1);
    if cpu_fast(&[p]) && u.device().is_cpu() && u.dtype() == DType::F32 && !p.same_storage(u) {
        u.with_f32_slice(|us| p.update_f32(|ps| muon_dw_polar_step_slice(ps, us, rows, cols, lr, NS_EPS)))??;
        return Ok(());
    }
    let dev = p.device();
    let u_cpu = u.to_device(Device::Cpu)?.to_vec::<f32>()?;
    let mut o_cpu = vec![0.0f32; u_cpu.len()];
    const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
    const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
    for r in 0..rows {
        let row = &u_cpu[r * cols..(r + 1) * cols];
        let max_abs = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let sc_big = if max_abs > BIG_THRESH { DOWN_SCALE } else { 1.0 };
        let mut sq = 0.0f32;
        for &v in row {
            let vs = if sc_big != 1.0 { v * sc_big } else { v };
            sq += vs * vs;
        }
        let denom = sq.sqrt() + NS_EPS * sc_big;
        let out = &mut o_cpu[r * cols..(r + 1) * cols];
        for (dst, &v) in out.iter_mut().zip(row) {
            let vs = if sc_big != 1.0 { v * sc_big } else { v };
            *dst = vs / denom;
        }
    }
    let o = Tensor::from_vec(o_cpu, p.shape().clone())?.to_device(dev)?;
    p.add_scaled_assign(&o, -lr)?;
    Ok(())
}

/// Tensor-op fallback for the post-momentum half of [`MuSgd::muon_step`].
#[allow(clippy::too_many_arguments)]
fn muon_step_tensor_ops(
    ps: &[&Tensor],
    bs: &[&Tensor],
    gs: &[Tensor],
    dws: &[bool],
    mu: f32,
    nesterov: bool,
    lr: f32,
    ortho: MuonOrtho,
    geo: MuonGeoMode,
) -> Result<()> {
    let us = if nesterov {
        let us = gs.iter().map(|x| x.mul_scalar(1.0 - mu)).collect::<folo_core::Result<Vec<_>>>()?;
        let ur: Vec<&Tensor> = us.iter().collect();
        axpby_many(&ur, bs, 1.0, mu)?;
        us
    } else {
        bs.iter().map(|b| b.mul_scalar(1.0)).collect::<folo_core::Result<Vec<_>>>()?
    };
    let mut mat_ps = Vec::new();
    let mut mat_us = Vec::new();
    for ((&p, u), &dw) in ps.iter().zip(us).zip(dws) {
        if dw {
            muon_dw_step_tensor(p, &u, lr)?;
        } else {
            mat_ps.push(p);
            mat_us.push(u);
        }
    }
    if mat_ps.is_empty() {
        return Ok(());
    }
    let os = orthogonalize_many_geo(&mat_us, ortho, geo)?;
    let or: Vec<&Tensor> = os.iter().collect();
    axpby_many(&mat_ps, &or, 1.0, -lr)
}

/// The rest of [`MuSgd::muon_step`] after the momentum update through the backend's fused kernels
/// (`Tensor::muon_step_many_geo_`); `false`, touching nothing, where the backend has none.
#[allow(clippy::too_many_arguments)]
fn muon_fused_with_scales(
    ps: &[&Tensor],
    bs: &[&Tensor],
    gs: &[&Tensor],
    scales: &[f32],
    dws: &[bool],
    mu: f32,
    nesterov: bool,
    lr: f32,
    ortho: MuonOrtho,
) -> Result<bool> {
    if ps.iter().all(|p| p.device().is_cpu()) {
        return Ok(false);
    }
    let settings = MuonSettings::from_schedule(mu, nesterov, ortho.schedule(), ortho.schatten4(), NS_EPS, lr);
    Ok(Tensor::muon_step_many_geo_(ps, bs, gs, scales, dws, &settings)?)
}

/// `p = a · p + b · g` for every pair: one multi-tensor call on a device that has it, AVX2 on CPU, else tensor ops.
fn axpby_many(ps: &[&Tensor], gs: &[&Tensor], a: f32, b: f32) -> Result<()> {
    use rayon::prelude::*;

    if ps.iter().any(|p| !p.device().is_cpu()) {
        let none: Vec<Vec<&Tensor>> = vec![Vec::new(); ps.len()];
        if Tensor::optimizer_step_many_(ps, gs, &none, &OptimStep::Axpby { a, b })? {
            return Ok(());
        }
    }
    let all_cpu = ps
        .iter()
        .zip(gs)
        .all(|(p, g)| cpu_fast(&[p]) && g.device().is_cpu() && g.dtype() == DType::F32 && !p.same_storage(g));
    if all_cpu {
        let total: usize = ps.iter().map(|p| p.numel()).sum();
        let disjoint = ps.len() > 1
            && total >= PAR_THRESHOLD
            && (0..ps.len()).all(|i| {
                (i + 1..ps.len()).all(|j| !ps[i].same_storage(ps[j]))
                    && gs.iter().all(|g| !ps[i].same_storage(g))
            });
        if disjoint {
            return folo_core::cpu::in_pool(|| {
                ps.par_iter().zip(gs.par_iter()).try_for_each(|(p, g)| -> Result<()> {
                    let _g = folo_core::no_grad();
                    g.with_f32_slice(|gs| p.update_f32(|ps| axpby_slice(ps, gs, a, b)))??;
                    Ok(())
                })
            });
        }
        for (p, g) in ps.iter().zip(gs) {
            g.with_f32_slice(|gs| p.update_f32(|ps| axpby_slice(ps, gs, a, b)))??;
        }
        return Ok(());
    }
    for (p, g) in ps.iter().zip(gs) {
        if cpu_fast(&[p]) && g.device().is_cpu() && g.dtype() == DType::F32 && !p.same_storage(g) {
            g.with_f32_slice(|gs| p.update_f32(|ps| axpby_slice(ps, gs, a, b)))??;
        } else {
            if a != 1.0 {
                p.mul_scalar_assign(a)?;
            }
            p.add_scaled_assign(g, b)?;
        }
    }
    Ok(())
}

impl Optimizer for MuSgd {
    fn kind(&self) -> OptimKind {
        OptimKind::MuSgd
    }

    fn groups(&self) -> &[ParamGroup] {
        self.sgd.groups()
    }

    fn groups_mut(&mut self) -> &mut [ParamGroup] {
        self.sgd.groups_mut()
    }

    fn step(&mut self, grads: &dyn GradSource) -> Result<()> {
        self.ensure_buffers()?;
        self.muon_step(grads)?;
        if self.weights.sgd == 1.0 {
            return self.sgd.step(grads);
        }
        // The SGD half of a Muon group runs at `sgd · lr`, while decoupled/cautious weight decay uses unscaled `lr`.
        let scales: Vec<f64> = self
            .sgd
            .groups()
            .iter()
            .map(|g| if is_muon_group(&g.name) { self.weights.sgd } else { 1.0 })
            .collect();
        self.sgd.step_with_lr_scales(grads, Some(&scales))
    }

    fn step_count(&self) -> u64 {
        self.sgd.step_count()
    }

    fn contract(&self) -> OptimContract {
        let mut c = group_contract(self);
        c.musgd = (self.weights != MuSgdWeights::default()).then_some(self.weights);
        // Omit `muon_ortho` only for legacy `Ns5` so pre-Polar `folo.optim` contracts stay byte-compatible and
        // older readers cleanly reject `Polar4` / `Polar5` checkpoints instead of silently resuming with `Ns5`.
        c.muon_ortho = (self.ortho != MuonOrtho::Ns5).then_some(self.ortho);
        c.muon_geo = (!self.geo.is_legacy()).then_some(self.geo);
        if self.geo == MuonGeoMode::Geo {
            let mut dw = Vec::new();
            for g in self.sgd.groups() {
                if !is_muon_group(&g.name) {
                    continue;
                }
                for (pi, (name, p)) in g.params.iter().enumerate() {
                    if Self::uses_muon(g, p) && g.is_depthwise(pi) {
                        dw.push(name.clone());
                    }
                }
            }
            c.muon_depthwise = dw;
        }
        c
    }

    /// Also adopts the contract's step weights, orthogonalisation schedule, geometric conditioning mode and
    /// depthwise parameter selection.
    fn apply_contract(&mut self, c: &OptimContract) -> Result<()> {
        let weights = c.musgd.unwrap_or(self.weights);
        weights.validate().map_err(|e| Error::Load(format!("folo.optim: {e}")))?;
        apply_group_contract(self, c)?;
        self.weights = weights;
        self.ortho = c.muon_ortho.unwrap_or(MuonOrtho::Ns5);
        self.geo = c.muon_geo.unwrap_or(MuonGeoMode::Legacy);
        if self.geo == MuonGeoMode::Geo {
            let dw_set: std::collections::HashSet<&str> =
                c.muon_depthwise.iter().map(String::as_str).collect();
            for g in self.sgd.groups_mut() {
                if !dw_set.is_empty() && is_muon_group(&g.name) {
                    let mask: Vec<bool> = g.params.iter().map(|(n, _)| dw_set.contains(n.as_str())).collect();
                    if mask.iter().any(|&b| b) {
                        g.depthwise = mask;
                    } else {
                        g.depthwise.clear();
                    }
                } else {
                    g.depthwise.clear();
                }
            }
        }
        self.ensure_buffers()?;
        Ok(())
    }

    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = self.sgd.state_tensors()?;
        for (gi, g) in self.sgd.groups().iter().enumerate() {
            for (pi, (name, _)) in g.params.iter().enumerate() {
                if let Some(b) = &self.muon[gi][pi] {
                    out.push((format!("{name}.muon_buffer"), b.clone()));
                }
            }
        }
        Ok(out)
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        self.sgd.to_device(device)?;
        for b in self.muon.iter().flatten().flatten() {
            b.to_device_(device)?;
        }
        Ok(())
    }

    fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()> {
        let mut problems = Vec::new();
        let mut loaded = Vec::new();
        for (gi, g) in self.sgd.groups().iter().enumerate() {
            for (pi, (name, p)) in g.params.iter().enumerate() {
                if !Self::uses_muon(g, p) {
                    continue;
                }
                if let Some(t) = take_state(state, name, p, "muon_buffer", step > 0, &mut problems) {
                    loaded.push((gi, pi, t));
                }
            }
        }
        if !problems.is_empty() {
            return Err(Error::Load(format!("MuSGD state: {}", problems.join("; "))));
        }
        self.sgd.load_state(step, state)?;
        self.muon = self.sgd.groups().iter().map(|g| vec![None; g.params.len()]).collect();
        self.ensure_buffers()?;
        let _g = folo_core::no_grad();
        for (gi, pi, t) in loaded {
            let b = self.muon[gi][pi].as_ref().expect("allocated above");
            b.copy_from(&t.to_device(b.device())?)?;
        }
        Ok(())
    }
}
