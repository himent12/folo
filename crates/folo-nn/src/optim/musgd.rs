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
        weights.validate()?;
        let muon = groups.iter().map(|g| vec![None; g.params.len()]).collect();
        let mut s = MuSgd { sgd: Sgd::new(groups)?, muon, weights, ortho };
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

    fn uses_muon(g: &ParamGroup, p: &Tensor) -> bool {
        is_muon_group(&g.name) && matches!(p.rank(), 2 | 4)
    }

    fn ensure_buffers(&mut self) -> Result<()> {
        for (g, bufs) in self.sgd.groups().iter().zip(&mut self.muon) {
            for ((_, p), b) in g.params.iter().zip(bufs.iter_mut()) {
                if b.is_none() && Self::uses_muon(g, p) {
                    *b = Some(Tensor::zeros_with(p.shape(), DType::F32, p.device())?);
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
    /// with a gradient. On non-CPU devices, Muon groups sharing `(momentum, nesterov)` (such as `decay` and
    /// `decay.cls`) fuse into one `muon_step_many_` call so small `.cls` groups do not pay a second wave of
    /// grouped GEMM dispatches.
    fn muon_step(&self, grads: &dyn GradSource) -> Result<()> {
        let _g = folo_core::no_grad();
        // Collect active Muon groups (those with at least one parameter that received a gradient).
        struct Active<'a> {
            mu: f32,
            nesterov: bool,
            lr: f32,
            ps: Vec<&'a Tensor>,
            bs: Vec<&'a Tensor>,
            gs: Vec<Tensor>,
        }
        let mut active: Vec<Active<'_>> = Vec::new();
        for (g, bufs) in self.sgd.groups().iter().zip(&self.muon) {
            let mu = g.momentum as f32;
            let (mut ps, mut bs, mut gs) = (Vec::new(), Vec::new(), Vec::new());
            for ((name, p), b) in g.params.iter().zip(bufs) {
                let (Some(b), Some(gr)) = (b, grads.grad(p)) else { continue };
                if gr.shape() != p.shape() {
                    return Err(Error::Invalid(format!(
                        "musgd: gradient of `{name}` is {:?}, parameter is {:?}",
                        gr.dims(),
                        p.dims()
                    )));
                }
                ps.push(p);
                bs.push(b);
                gs.push(gr.to_dtype(DType::F32)?);
            }
            if !ps.is_empty() {
                active.push(Active {
                    mu,
                    nesterov: g.nesterov,
                    lr: (self.weights.muon * g.lr) as f32,
                    ps,
                    bs,
                    gs,
                });
            }
        }
        if active.is_empty() {
            return Ok(());
        }
        // Fast path: when >= 2 Muon groups share (mu, nesterov) off the CPU, fuse both momentum update and
        // orthogonalisation into a single multi-tensor device pass (carrying per-group lr in each matrix's scale,
        // with settings.lr = 1.0 whose negation is an exact IEEE-754 sign flip).
        let first_dev = active[0].ps[0].device();
        let can_fuse_all = active.len() >= 2
            && !first_dev.is_cpu()
            && active.iter().all(|a| {
                a.mu.to_bits() == active[0].mu.to_bits()
                    && a.nesterov == active[0].nesterov
                    && a.ps.iter().all(|p| p.device() == first_dev)
            });
        if can_fuse_all {
            let total: usize = active.iter().map(|a| a.ps.len()).sum();
            let (mut ps, mut bs, mut gr, mut scales) = (
                Vec::with_capacity(total),
                Vec::with_capacity(total),
                Vec::with_capacity(total),
                Vec::with_capacity(total),
            );
            for a in &active {
                ps.extend_from_slice(&a.ps);
                bs.extend_from_slice(&a.bs);
                gr.extend(a.gs.iter());
                scales.extend(a.ps.iter().map(|p| muon_scale(p.dims()) * a.lr));
            }
            let (mu, nesterov) = (active[0].mu, active[0].nesterov);
            axpby_many(&bs, &gr, mu, 1.0 - mu)?;
            if muon_fused_with_scales(&ps, &bs, &gr, &scales, mu, nesterov, 1.0, self.ortho)? {
                return Ok(());
            }
            // Fallback if backend declined: momentum buffers `bs` are already updated; run tensor ops per group.
            for a in &active {
                muon_step_tensor_ops(&a.ps, &a.bs, &a.gs, a.mu, a.nesterov, a.lr, self.ortho)?;
            }
            return Ok(());
        }
        for a in &active {
            let gr: Vec<&Tensor> = a.gs.iter().collect();
            // m = μ · m + (1 − μ) · g
            axpby_many(&a.bs, &gr, a.mu, 1.0 - a.mu)?;
            // The rest of the Muon step in a few kernels for all of them, where the backend has it.
            let scales: Vec<f32> = a.ps.iter().map(|p| muon_scale(p.dims())).collect();
            if muon_fused_with_scales(&a.ps, &a.bs, &gr, &scales, a.mu, a.nesterov, a.lr, self.ortho)? {
                continue;
            }
            muon_step_tensor_ops(&a.ps, &a.bs, &a.gs, a.mu, a.nesterov, a.lr, self.ortho)?;
        }
        Ok(())
    }
}

/// Tensor-op fallback for the post-momentum half of [`MuSgd::muon_step`].
#[allow(clippy::too_many_arguments)]
fn muon_step_tensor_ops(
    ps: &[&Tensor],
    bs: &[&Tensor],
    gs: &[Tensor],
    mu: f32,
    nesterov: bool,
    lr: f32,
    ortho: MuonOrtho,
) -> Result<()> {
    // u = μ · m + (1 − μ) · g (Nesterov), else m.
    let us = if nesterov {
        let us = gs.iter().map(|x| x.mul_scalar(1.0 - mu)).collect::<folo_core::Result<Vec<_>>>()?;
        let ur: Vec<&Tensor> = us.iter().collect();
        axpby_many(&ur, bs, 1.0, mu)?;
        us
    } else {
        bs.iter().map(|b| b.mul_scalar(1.0)).collect::<folo_core::Result<Vec<_>>>()?
    };
    let os = orthogonalize_many_with(&us, ortho)?;
    let or: Vec<&Tensor> = os.iter().collect();
    axpby_many(ps, &or, 1.0, -lr)
}

/// The rest of [`MuSgd::muon_step`] after the momentum update through the backend's fused kernels
/// (`Tensor::muon_step_many_`); `false`, touching nothing, where the backend has none.
#[allow(clippy::too_many_arguments)]
fn muon_fused_with_scales(
    ps: &[&Tensor],
    bs: &[&Tensor],
    gs: &[&Tensor],
    scales: &[f32],
    mu: f32,
    nesterov: bool,
    lr: f32,
    ortho: MuonOrtho,
) -> Result<bool> {
    if ps.iter().all(|p| p.device().is_cpu()) {
        return Ok(false);
    }
    let settings = MuonSettings::from_schedule(mu, nesterov, ortho.schedule(), ortho.schatten4(), NS_EPS, lr);
    Ok(Tensor::muon_step_many_(ps, bs, gs, scales, &settings)?)
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

/// [`MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2) using the default [`MuonOrtho::Ns5`] schedule.
pub fn orthogonalize_many(us: &[Tensor]) -> Result<Vec<Tensor>> {
    orthogonalize_many_with(us, MuonOrtho::Ns5)
}

/// [`MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2): each flattened to `[d₀, rest]`,
/// orthogonalised with `ortho`, scaled by [`muon_scale`] and reshaped back. Matrices are oriented so that
/// rows ≤ cols and batched by column count, zero-padding the rows (a zero row changes neither the Frobenius norm,
/// nor the Schatten-4 norm `‖X·Xᵀ‖_F^(1/2)`, nor the other rows' result).
pub fn orthogonalize_many_with(us: &[Tensor], ortho: MuonOrtho) -> Result<Vec<Tensor>> {
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
        let norm = x.square()?.sum([1, 2], true)?.sqrt()?.add_scalar(NS_EPS)?;
        x = x.div(&norm)?;
        for (step, &(a, b, c)) in schedule.iter().enumerate() {
            let xt = x.transpose(1, 2)?.contiguous()?;
            let am = x.matmul(&xt)?;
            if step == 0 && use_s4 {
                // Zero-extra-GEMM step-0 Schatten-4 spectral rescaling:
                // α = ‖A₀‖_F^(1/2) + eps / (‖U‖_F + eps) guarantees σ₁(X₀ / α) < 1 strictly.
                let a_frob = am.square()?.sum([1, 2], true)?.sqrt()?;
                let alpha = a_frob.sqrt()?.add(&norm.recip()?.mul_scalar(NS_EPS)?)?;
                let inv_a = alpha.recip()?;
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
        for (j, &k) in ks.iter().enumerate() {
            let (i, m, t) = &mats[k];
            let r = m.dims()[0];
            let o = x.narrow(0, j, 1)?.narrow(1, 0, r)?.reshape([r, m.dims()[1]])?;
            let o = if *t { o.transpose(0, 1)? } else { o };
            let u = &us[*i];
            let s = muon_scale(u.dims());
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
        // Omit `muon_ortho` only for legacy `Ns5` so pre-Polar checkpoints match byte-for-byte and older readers
        // cleanly reject `Polar4` / `Polar5` checkpoints instead of silently resuming them with `Ns5`.
        c.muon_ortho = (self.ortho != MuonOrtho::Ns5).then_some(self.ortho);
        c
    }

    /// Also adopts the contract's step weights and orthogonalisation schedule. A contract without `musgd` keeps
    /// the weights this optimizer was built with; one without `muon_ortho` (legacy pre-Polar or explicit `Ns5`
    /// checkpoint) resumes with [`MuonOrtho::Ns5`] for bit-exact backward compatibility.
    fn apply_contract(&mut self, c: &OptimContract) -> Result<()> {
        let weights = c.musgd.unwrap_or(self.weights);
        weights.validate().map_err(|e| Error::Load(format!("folo.optim: {e}")))?;
        apply_group_contract(self, c)?;
        self.weights = weights;
        self.ortho = c.muon_ortho.unwrap_or(MuonOrtho::Ns5);
        Ok(())
    }

    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let mut out = self.sgd.state_tensors()?;
        for (g, bufs) in self.sgd.groups().iter().zip(&self.muon) {
            for ((name, _), b) in g.params.iter().zip(bufs) {
                if let Some(b) = b {
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
