//! MuSGD: SGD plus a Muon (orthogonalised momentum) update on every matrix-shaped weight, as Ultralytics 8.4's
//! `MuSGD` (docs/MODEL.md §8.2).

use std::collections::HashMap;

use folo_core::backend::OptimStep;
use folo_core::{DType, Device, Tensor};

use super::{GradSource, OptimKind, Optimizer, ParamGroup, Sgd, take_state};
use crate::{Error, Result};

/// Weight of the Muon update: `p −= MUON_SCALE · lr · O`.
pub const MUON_SCALE: f64 = 0.2;
/// Newton–Schulz iterations of [`orthogonalize`].
pub const NS_STEPS: usize = 5;
/// The quintic Newton–Schulz coefficients `(a, b, c)`: `X ← a·X + (b·A + c·A²)·X`, `A = X·Xᵀ`.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);
/// Added to the Frobenius norm before [`orthogonalize`] divides by it.
pub const NS_EPS: f32 = 1e-7;
/// A bucket of [`orthogonalize_many`] keeps its row counts within this factor (bounds the zero padding).
const BUCKET_SPAN: usize = 16;

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
/// O = NS5(u as [rows, cols]) · sqrt(max(1, d₋₂ / d₋₁))
/// p = p − 0.2 · lr · O                         (0.2: MuSgdWeights::muon)
/// ```
/// where the matrix is `u.flatten(1)` (a conv weight `[c_out, c_in·k·k]`), `d₋₂ / d₋₁` are the parameter's last
/// two dimensions (1 for a square kernel) and `NS5` is [`orthogonalize`]. The SGD update then reads the
/// parameter after the Muon step (its weight decay too), as the reference's in-place `foreach` ops do.
pub struct MuSgd {
    sgd: Sgd,
    muon: Vec<Vec<Option<Tensor>>>,
    weights: MuSgdWeights,
}

/// The two factors of a [`MuSgd`] step on a Muon parameter: `p −= muon · lr · O`, then the SGD update at
/// `sgd · lr` (other parameters take SGD at `lr`). The default is Ultralytics 8.4's trainer (`muon = 0.2`,
/// `sgd = 1.0`); YOLO26's published COCO run used 0.528 and 0.674 (`muon_w`, `sgd_w` of its `train_args`).
#[derive(Clone, Copy, Debug, PartialEq)]
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

impl MuSgd {
    /// An optimizer over `groups` (no parameter in two groups; all `F32`) with the default [`MuSgdWeights`].
    pub fn new(groups: Vec<ParamGroup>) -> Result<Self> {
        Self::with_weights(groups, MuSgdWeights::default())
    }

    /// [`MuSgd::new`] with other step weights (finite, non-negative).
    pub fn with_weights(groups: Vec<ParamGroup>, weights: MuSgdWeights) -> Result<Self> {
        if ![weights.muon, weights.sgd].iter().all(|w| w.is_finite() && *w >= 0.0) {
            return Err(Error::Invalid(format!("musgd: weights must be finite and >= 0, got {weights:?}")));
        }
        let muon = groups.iter().map(|g| vec![None; g.params.len()]).collect();
        let mut s = MuSgd { sgd: Sgd::new(groups)?, muon, weights };
        s.ensure_buffers()?;
        Ok(s)
    }

    /// The step weights.
    pub fn weights(&self) -> MuSgdWeights {
        self.weights
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

    /// The Muon half of a step: momentum, orthogonalisation and `p −= 0.2 · lr · O` for every Muon parameter
    /// with a gradient.
    fn muon_step(&self, grads: &dyn GradSource) -> Result<()> {
        let _g = folo_core::no_grad();
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
            if ps.is_empty() {
                continue;
            }
            let gr: Vec<&Tensor> = gs.iter().collect();
            // m = μ · m + (1 − μ) · g
            axpby_many(&bs, &gr, mu, 1.0 - mu)?;
            // u = μ · m + (1 − μ) · g (Nesterov), else m.
            let us = if g.nesterov {
                let us = gs.iter().map(|x| x.mul_scalar(1.0 - mu)).collect::<folo_core::Result<Vec<_>>>()?;
                let ur: Vec<&Tensor> = us.iter().collect();
                axpby_many(&ur, &bs, 1.0, mu)?;
                us
            } else {
                bs.iter().map(|b| b.mul_scalar(1.0)).collect::<folo_core::Result<Vec<_>>>()?
            };
            let os = orthogonalize_many(&us)?;
            let lr = (self.weights.muon * g.lr) as f32;
            let or: Vec<&Tensor> = os.iter().collect();
            axpby_many(&ps, &or, 1.0, -lr)?;
        }
        Ok(())
    }
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
    Ok(orthogonalize_many(std::slice::from_ref(g))?.remove(0))
}

/// [`MuSgd`]'s update directions for `us` (any shapes of rank ≥ 2): each flattened to `[d₀, rest]`,
/// orthogonalised ([`orthogonalize`]), scaled by [`muon_scale`] and reshaped back. Matrices are oriented so that
/// rows ≤ cols and batched by column count, zero-padding the rows (a zero row changes neither the norm nor the
/// other rows' result), as the reference does.
pub fn orthogonalize_many(us: &[Tensor]) -> Result<Vec<Tensor>> {
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
    let (a, b, c) = NS_COEFFS;
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
        for _ in 0..NS_STEPS {
            let xt = x.transpose(1, 2)?.contiguous()?;
            let am = x.matmul(&xt)?;
            let bm = am.mul_scalar(b)?.add(&am.matmul(&am)?.mul_scalar(c)?)?;
            x = x.mul_scalar(a)?.add(&bm.matmul(&x)?)?;
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
        // The SGD half of a Muon group runs at `sgd · lr`; the groups' own LRs are put back afterwards.
        let lrs: Vec<f64> = self.sgd.groups().iter().map(|g| g.lr).collect();
        for g in self.sgd.groups_mut() {
            if is_muon_group(&g.name) {
                g.lr *= self.weights.sgd;
            }
        }
        let r = self.sgd.step(grads);
        for (g, lr) in self.sgd.groups_mut().iter_mut().zip(lrs) {
            g.lr = lr;
        }
        r
    }

    fn step_count(&self) -> u64 {
        self.sgd.step_count()
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
