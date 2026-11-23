//! MuSGD: SGD plus a Muon (orthogonalised momentum) update on every matrix-shaped weight, as Ultralytics 8.4's
//! `MuSGD` (docs/MODEL.md §8.2), with optional Polar-FOLO per-step minimax Remez orthogonalisation schedules and
//! step-0 Schatten-4 spectral rescaling.

use std::collections::HashMap;

use folo_core::backend::{MuonSettings, OptimStep};
use folo_core::{DType, Device, Tensor};

use serde::{Deserialize, Serialize};

use super::{
    GradSource, OptimContract, OptimKind, Optimizer, ParamGroup, Sgd, apply_group_contract, group_contract,
    take_state,
};
use crate::{Error, Result};

/// Geometric spectral conditioning mode of [`MuSgd`] (`Conv-GeoMuon`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MuonGeoMode {
    /// Legacy `muon_scale` (`dims[n-2]/dims[n-1]`) and dense orthogonalisation on all rank-2/4 decay weights.
    #[default]
    Legacy,
    /// True 2-D flattened aspect scale `sqrt(max(1, m/n))` + exact zero-GEMM per-channel Polar LMO for depthwise convs.
    Geo,
}

impl MuonGeoMode {
    /// True when this is legacy mode.
    pub fn is_legacy(&self) -> bool {
        *self == MuonGeoMode::Legacy
    }

    /// Canonical lowercase name (`"legacy"`, `"geo"`).
    pub fn as_str(self) -> &'static str {
        match self {
            MuonGeoMode::Legacy => "legacy",
            MuonGeoMode::Geo => "geo",
        }
    }
}

/// Weight of the Muon update: `p −= MUON_SCALE · lr · O`.
pub const MUON_SCALE: f64 = 0.2;
/// Newton–Schulz iterations of [`orthogonalize`].
pub const NS_STEPS: usize = 5;
/// The quintic Newton–Schulz coefficients `(a, b, c)`: `X ← a·X + (b·A + c·A²)·X`, `A = X·Xᵀ`.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
/// Fixed 5-step Newton–Schulz schedule (`NS_COEFFS` repeated 5 times).
pub const NS5_SCHEDULE: [(f32, f32, f32); NS_STEPS] = [NS_COEFFS; NS_STEPS];
/// Polar-FOLO 4-step minimax Remez odd-quintic schedule (`l₀ = 5e-3`, safety cushion `1.005`, paired with step-0
/// Schatten-4 spectral rescaling): 12 GEMMs (20% fewer than `NS5`) with ~3× lower polar Frobenius distance and
/// tighter spectral accuracy across `κ = 10..10⁴`.
pub const POLAR4_COEFFS: [(f32, f32, f32); 4] = [
    (4.216_272_4, -12.289_586, 9.009_986),
    (4.152_54, -11.770_327, 8.558_353),
    (3.887_979, -9.771_957, 6.839_154_7),
    (3.287_126, -5.261_409, 3.046_717_4),
];
/// Polar-FOLO 5-step minimax Remez odd-quintic schedule (`l₀ = 1e-3`, safety cushion `1.005`, paired with step-0
/// Schatten-4 spectral rescaling): 15 GEMMs with ultra-sharp orthogonalisation (`≤ 0.12` spectral error down to
/// `κ = 100`, 3.3× lower polar error at `κ = 1000`).
pub const POLAR5_COEFFS: [(f32, f32, f32); 5] = [
    (4.232_037, -12.420_36, 9.124_008),
    (4.219_302, -12.314_649, 9.031_83),
    (4.165_359_5, -11.873_566, 8.648_002),
    (3.940_082_3, -10.145_948, 7.158_359_5),
    (3.537_624_1, -6.125_873_6, 3.688_620_8),
];
/// Added to the Frobenius norm before [`orthogonalize`] divides by it.
pub const NS_EPS: f32 = 1e-7;
/// A bucket of [`orthogonalize_many`] keeps its row counts within this factor (bounds the zero padding).
const BUCKET_SPAN: usize = 16;

/// Orthogonalisation polynomial schedule used by [`MuSgd`]'s Muon step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MuonOrtho {
    /// Legacy fixed 5-step quintic Newton–Schulz `(3.4445, -4.7750, 2.0315)` (15 GEMMs).
    Ns5,
    /// Default Polar-FOLO 4-step minimax Remez odd-quintic schedule with step-0 Schatten-4 spectral rescaling
    /// (12 GEMMs, 20% fewer GEMMs than `Ns5` with ~3× lower polar Frobenius error).
    #[default]
    Polar4,
    /// Polar-FOLO 5-step minimax Remez odd-quintic schedule with step-0 Schatten-4 spectral rescaling (15 GEMMs).
    Polar5,
}

impl MuonOrtho {
    /// True when this is the default [`MuonOrtho::Polar4`] schedule.
    pub fn is_default(&self) -> bool {
        *self == MuonOrtho::Polar4
    }

    /// Per-step odd-quintic coefficients `(a_k, b_k, c_k)`.
    pub fn schedule(self) -> &'static [(f32, f32, f32)] {
        match self {
            MuonOrtho::Ns5 => &NS5_SCHEDULE,
            MuonOrtho::Polar4 => &POLAR4_COEFFS,
            MuonOrtho::Polar5 => &POLAR5_COEFFS,
        }
    }

    /// Whether step 0 applies zero-extra-GEMM Schatten-4 spectral rescaling
    /// `α = ‖X₀·X₀ᵀ‖_F^(1/2) + eps / (‖U‖_F + eps)`.
    pub fn schatten4(self) -> bool {
        !matches!(self, MuonOrtho::Ns5)
    }

    /// Canonical lowercase name (`"ns5"`, `"polar4"`, `"polar5"`).
    pub fn as_str(self) -> &'static str {
        match self {
            MuonOrtho::Ns5 => "ns5",
            MuonOrtho::Polar4 => "polar4",
            MuonOrtho::Polar5 => "polar5",
        }
    }
}

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
        let muon = groups.iter().map(|g| vec![None; g.params.len()]).collect();
        let mut s = MuSgd { sgd: Sgd::new(groups)?, muon, weights, ortho, geo };
        s.ensure_buffers()?;
        Ok(s)
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

/// Exact zero-GEMM per-channel block-diagonal Polar LMO for depthwise conv weights (CPU / tensor fallback).
fn muon_dw_step_tensor(p: &Tensor, u: &Tensor, lr: f32) -> Result<()> {
    let rows = p.dims()[0];
    let cols = p.numel() / rows.max(1);
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

/// `p = a · p + b · g` for every pair: one multi-tensor call on a device that has it, else tensor ops.
fn axpby_many(ps: &[&Tensor], gs: &[&Tensor], a: f32, b: f32) -> Result<()> {
    let none: Vec<Vec<&Tensor>> = vec![Vec::new(); ps.len()];
    if ps.iter().any(|p| !p.device().is_cpu())
        && Tensor::optimizer_step_many_(ps, gs, &none, &OptimStep::Axpby { a, b })?
    {
        return Ok(());
    }
    for (p, g) in ps.iter().zip(gs) {
        if a != 1.0 {
            p.mul_scalar_assign(a)?;
        }
        p.add_scaled_assign(g, b)?;
    }
    Ok(())
}

/// The scale [`MuSgd`] gives a parameter's orthogonalised update: `sqrt(max(1, d₋₂ / d₋₁))` of its last two
/// dimensions.
pub fn muon_scale(dims: &[usize]) -> f32 {
    let n = dims.len();
    if n < 2 {
        return 1.0;
    }
    (dims[n - 2] as f32 / dims[n - 1] as f32).max(1.0).sqrt()
}

/// True 2-D flattened aspect scale for `Conv-GeoMuon`: `sqrt(max(1, m / n))` where `m = dims[0]` and
/// `n = prod(dims[1..])`.
pub fn muon_scale_geo(dims: &[usize]) -> f32 {
    if dims.len() < 2 {
        return 1.0;
    }
    let rows = dims[0];
    let cols = dims.iter().product::<usize>() / rows.max(1);
    if cols == 0 {
        return 1.0;
    }
    (rows as f32 / cols as f32).max(1.0).sqrt()
}

/// Newton–Schulz orthogonalisation of one matrix `g` (`[rows, cols]`): `X = g / (‖g‖_F + 1e-7)`, transposed so
/// that `rows ≤ cols`, then [`NS_STEPS`] times `A = X·Xᵀ`, `X = a·X + (b·A + c·A·A)·X` with [`NS_COEFFS`], and
/// transposed back. In `f32` (the reference computes in `bfloat16`).
pub fn orthogonalize(g: &Tensor) -> Result<Tensor> {
    orthogonalize_with(g, MuonOrtho::Ns5)
}

/// [`orthogonalize`] with a chosen [`MuonOrtho`] schedule (e.g. [`MuonOrtho::Polar4`] or [`MuonOrtho::Polar5`]).
pub fn orthogonalize_with(g: &Tensor, ortho: MuonOrtho) -> Result<Tensor> {
    Ok(orthogonalize_many_with(std::slice::from_ref(g), ortho)?.remove(0))
}

/// [`MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2) using the legacy [`MuonOrtho::Ns5`] schedule.
pub fn orthogonalize_many(us: &[Tensor]) -> Result<Vec<Tensor>> {
    orthogonalize_many_with(us, MuonOrtho::Ns5)
}

/// [`MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2): each flattened to `[d₀, rest]`,
/// orthogonalised with `ortho`, scaled by [`muon_scale`] and reshaped back. Matrices are oriented so that
/// rows ≤ cols and batched by column count, zero-padding the rows (a zero row changes neither the Frobenius norm,
/// nor the Schatten-4 norm `‖X·Xᵀ‖_F^(1/2)`, nor the other rows' result).
pub fn orthogonalize_many_with(us: &[Tensor], ortho: MuonOrtho) -> Result<Vec<Tensor>> {
    orthogonalize_many_geo(us, ortho, MuonGeoMode::Legacy)
}

/// [`orthogonalize_many_with`] extended with [`MuonGeoMode`] (true 2-D aspect scaling).
fn orthogonalize_many_geo(us: &[Tensor], ortho: MuonOrtho, geo: MuonGeoMode) -> Result<Vec<Tensor>> {
    let _g = folo_core::no_grad();
    // (index, oriented matrix, transposed)
    let mut mats = Vec::with_capacity(us.len());
    for (i, u) in us.iter().enumerate() {
        let d = u.dims();
        if d.len() < 2 {
            return Err(Error::Invalid(format!("musgd: orthogonalize needs rank >= 2, got {d:?}")));
        }
        let rows = d[0];
        let m = u.reshape([rows, u.numel() / rows.max(1)])?;
        let t = m.dims()[0] > m.dims()[1];
        let m = if t { m.transpose(0, 1)?.contiguous()? } else { m.contiguous()? };
        mats.push((i, m, t));
    }
    let mut by_cols: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut order: Vec<usize> = (0..mats.len()).collect();
    order.sort_by_key(|&k| (mats[k].1.dims()[1], std::cmp::Reverse(mats[k].1.dims()[0]), k));
    for k in order {
        let (r, c) = (mats[k].1.dims()[0], mats[k].1.dims()[1]);
        match by_cols.last_mut() {
            Some((cc, ks)) if *cc == c && mats[ks[0]].1.dims()[0] <= BUCKET_SPAN * r => ks.push(k),
            _ => by_cols.push((c, vec![k])),
        }
    }
    let mut out: Vec<Option<Tensor>> = vec![None; us.len()];
    let schedule = ortho.schedule();
    let use_s4 = ortho.schatten4();
    for (_, ks) in by_cols {
        let rmax = mats[ks[0]].1.dims()[0];
        let padded = ks
            .iter()
            .map(|&k| {
                let m = &mats[k].1;
                let r = m.dims()[0];
                if r == rmax { Ok(m.clone()) } else { m.pad(&[(0, rmax - r), (0, 0)], 0.0) }
            })
            .collect::<folo_core::Result<Vec<_>>>()?;
        let mut x = Tensor::stack(&padded, 0)?;
        if x.numel() > 0 {
            let (norm, eps_s) = if use_s4 {
                const BIG_THRESH: f32 = f32::from_bits(0x5780_0000);
                const DOWN_SCALE: f32 = f32::from_bits(0x1780_0000);
                let max_abs = x.abs()?.max([1, 2], true)?;
                let is_big = max_abs.gt_scalar(BIG_THRESH)?;
                let one = max_abs.ones_like()?;
                let scale = is_big.where_cond(&one.mul_scalar(DOWN_SCALE)?, &one)?;
                let xs = x.mul(&scale)?;
                let eps_s = scale.mul_scalar(NS_EPS)?;
                let norm = xs.square()?.sum([1, 2], true)?.sqrt()?.add(&eps_s)?;
                x = xs.div(&norm)?;
                (norm, Some(eps_s))
            } else {
                let norm = x.square()?.sum([1, 2], true)?.sqrt()?.add_scalar(NS_EPS)?;
                x = x.div(&norm)?;
                (norm, None)
            };
            for (step, &(a, b, c)) in schedule.iter().enumerate() {
                let xt = x.transpose(1, 2)?.contiguous()?;
                let am = x.matmul(&xt)?;
                if step == 0 && use_s4 {
                    let a_frob = am.square()?.sum([1, 2], true)?.sqrt()?;
                    let eps_s = eps_s.as_ref().expect("set when use_s4");
                    let alpha = a_frob.sqrt()?.add(&eps_s.div(&norm)?)?;
                    let zeros = Tensor::zeros_with(alpha.shape(), DType::F32, alpha.device())?;
                    let inv_a = alpha.eq_scalar(0.0)?.where_cond(&zeros, &alpha.recip()?)?;
                    let inv_a2 = inv_a.square()?;
                    let inv_a4 = inv_a2.square()?;
                    let bm =
                        am.mul(&inv_a2.mul_scalar(b)?)?.add(&am.matmul(&am)?.mul(&inv_a4.mul_scalar(c)?)?)?;
                    x = x.mul(&inv_a.mul_scalar(a)?)?.add(&bm.matmul(&x)?.mul(&inv_a)?)?;
                } else {
                    let bm = am.mul_scalar(b)?.add(&am.matmul(&am)?.mul_scalar(c)?)?;
                    x = x.mul_scalar(a)?.add(&bm.matmul(&x)?)?;
                }
            }
        }
        for (j, &k) in ks.iter().enumerate() {
            let (i, m, t) = &mats[k];
            let r = m.dims()[0];
            let c = m.dims()[1];
            let o = x.narrow(0, j, 1)?.narrow(1, 0, r)?.reshape([r, c])?;
            let o = if *t { o.transpose(0, 1)? } else { o };
            let u = &us[*i];
            let s = if geo == MuonGeoMode::Legacy { muon_scale(u.dims()) } else { muon_scale_geo(u.dims()) };
            let o = o.contiguous()?.reshape(u.shape().clone())?;
            out[*i] = Some(if s != 1.0 { o.mul_scalar(s)? } else { o });
        }
    }
    Ok(out.into_iter().map(|o| o.expect("every matrix is in a bucket")).collect())
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
