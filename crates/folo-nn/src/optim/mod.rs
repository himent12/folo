//! Optimizers (SGD with Nesterov momentum, AdamW, MuSGD), parameter groups, gradient accumulation and clipping.
//!
//! Update rules are exactly docs/MODEL.md §8.2. Hyperparameters are `f64`; the per-element arithmetic is
//! `f32`, like the tensors. A parameter with no gradient in a step is skipped entirely (no decay, no state
//! change), as in PyTorch.
//!
//! - `sgd.rs`: [`Sgd`]. `adamw.rs`: [`AdamW`].
//! - `musgd.rs`: [`MuSgd`], SGD plus the Muon update. `ortho.rs`: Newton–Schulz / Polar-FOLO [`orthogonalize`].
//! - `grads.rs`: [`GradAccumulator`] (accumulation over several backward passes, clipping) and
//!   [`global_grad_norm`].
//! - `simd/`: AVX2 / scalar CPU kernels shared by the optimizers, clipping and AMP unscaling.
//! - This file: the [`Optimizer`] trait, [`OptimKind`], [`ParamGroup`], [`StandardGroups`],
//!   [`effective_weight_decay`] and the `auto` optimizer rule ([`auto_optimizer_rule`]).
//!
//! ```
//! use folo_core::{rng::Rng, Tensor};
//! use folo_nn::optim::{GradAccumulator, Optimizer, Sgd, StandardGroups};
//! use folo_nn::{Conv2d, Conv2dConfig, Forward, Module};
//!
//! let mut rng = Rng::seed(0);
//! let conv = Conv2d::new(Conv2dConfig::new(3, 4, 3), &mut rng);
//! let groups = StandardGroups::from_module(&conv).into_groups(0.01, 0.937, 5e-4);
//! let mut opt = Sgd::new(groups).unwrap();
//! let mut acc = GradAccumulator::new(&conv.parameters());
//!
//! let x = Tensor::randn([2, 3, 8, 8], 0.0, 1.0, &mut rng);
//! let loss = conv.forward(&x).unwrap().square().unwrap().mean_all().unwrap();
//! acc.add(&loss.backward().unwrap()).unwrap();
//! acc.clip_grad_norm(10.0).unwrap();
//! opt.step(&acc).unwrap();
//! acc.zero();
//! assert_eq!(opt.step_count(), 1);
//! ```

mod adamw;
mod agc;
mod cautious;
mod dispatch;
mod grads;
mod musgd;
mod ortho;
mod sgd;
pub(crate) mod simd;

use std::collections::HashMap;

use folo_core::{Device, Grads, Tensor};
use serde::{Deserialize, Serialize};

pub use adamw::AdamW;
pub use agc::{AGC_DEFAULT_CLIP_FACTOR, AGC_DEFAULT_EPS, AgcStats, clip_grad_agc_tensor};
pub(crate) use dispatch::{
    GradRef, PAR_THRESHOLD, ParamPlan, cpu_fast, for_each_param_planned, step_many, storage_key,
};
pub use grads::{DeviceUnscaleClip, GradAccumulator, global_grad_norm};
pub use musgd::{MUON_SCALE, MuSgd, MuSgdWeights, is_muon_group};
pub use ortho::{
    MuonGeoMode, MuonOrtho, NS_COEFFS, NS_EPS, NS_STEPS, NS5_SCHEDULE, POLAR3_CS9_COEFFS, POLAR4_COEFFS,
    POLAR5_COEFFS, muon_scale, muon_scale_geo, orthogonalize, orthogonalize_many, orthogonalize_many_with,
    orthogonalize_with,
};
pub use sgd::Sgd;

use crate::module::Module;
use crate::{BatchNorm2d, Error, Result};

/// The optimizer family, as written in `folo.optim.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimKind {
    /// SGD with (Nesterov) momentum and L2 weight decay.
    Sgd,
    /// Adam with decoupled weight decay.
    AdamW,
    /// SGD plus a Muon (orthogonalised momentum) update on matrix-shaped weights ([`MuSgd`]).
    MuSgd,
}

impl OptimKind {
    /// `"sgd"`, `"adamw"` or `"musgd"`.
    pub fn as_str(self) -> &'static str {
        match self {
            OptimKind::Sgd => "sgd",
            OptimKind::AdamW => "adamw",
            OptimKind::MuSgd => "musgd",
        }
    }
}

/// The name of the bias group, which docs/MODEL.md §8.4 warms up from `warmup_bias_lr`.
pub const BIAS_GROUP: &str = "bias";

/// Whether a group warms up from `warmup_bias_lr`: [`BIAS_GROUP`] and its `bias.<suffix>` splits (MuSGD's
/// `bias.cls`).
pub fn is_bias_group(name: &str) -> bool {
    name == BIAS_GROUP || name.strip_prefix(BIAS_GROUP).is_some_and(|r| r.starts_with('.'))
}

/// A named set of parameters sharing hyperparameters.
///
/// `base_lr` / `base_momentum` are the configured values (written to `folo.optim`); `lr` / `momentum` are
/// the values the next step uses, set by the schedule ([`crate::LrSchedule::apply`]) and initially equal to
/// the base values.
#[derive(Debug, Clone)]
pub struct ParamGroup {
    /// Group name (`decay`, `bn`, `bias` for the standard groups).
    pub name: String,
    /// `(name, tensor)` of every parameter, in canonical order.
    pub params: Vec<(String, Tensor)>,
    /// Base learning rate (before the schedule multiplier).
    pub base_lr: f64,
    /// Base SGD momentum μ / AdamW β1.
    pub base_momentum: f64,
    /// Weight decay (L2 for SGD, decoupled for AdamW).
    pub weight_decay: f64,
    /// SGD: Nesterov momentum.
    pub nesterov: bool,
    /// AdamW: β2.
    pub beta2: f64,
    /// AdamW: ε.
    pub eps: f64,
    /// Current learning rate.
    pub lr: f64,
    /// Current momentum / β1.
    pub momentum: f64,
    /// Decouple weight decay from SGD/MuSGD momentum (`p -= lr·u + lr_unscaled·wd·p`).
    pub decoupled_wd: bool,
    /// Cautious decoupled weight decay: apply decoupled decay only where `u · p > 0`.
    pub cautious_wd: bool,
    /// Opt-in Cautious update masking (Liang et al. 2024): mask updates where the update direction and reference
    /// gradient disagree in sign (`g` for [`AdamW`] and decoupled [`Sgd`]; L2-regularized `g + wd · p` for
    /// coupled [`Sgd`]) and rescale active entries by `n / max(active, 1)` (default `false`; supported by
    /// [`AdamW`] and [`Sgd`], rejected by [`MuSgd`]).
    pub cautious: bool,
    /// Per-parameter depthwise conv mask (`groups > 1 && groups == in_channels == out_channels`), aligned with
    /// `params` (empty means all false). Used by `MuSgd`'s `Conv-GeoMuon` depthwise Polar LMO.
    pub depthwise: Vec<bool>,
}

/// AdamW's β2 of every [`ParamGroup`] unless [`ParamGroup::adam`] sets another (FOLO's trainer never does).
pub const DEFAULT_BETA2: f64 = 0.999;

impl ParamGroup {
    /// A group with `lr = 0.01`, momentum 0.937, Nesterov on, no weight decay, `β2 = 0.999`, `ε = 1e-8`
    /// (docs/MODEL.md §8.1–8.2 defaults). Adjust with the builder methods.
    pub fn new(name: impl Into<String>, params: Vec<(String, Tensor)>) -> Self {
        ParamGroup {
            name: name.into(),
            params,
            base_lr: 0.01,
            base_momentum: 0.937,
            weight_decay: 0.0,
            nesterov: true,
            beta2: DEFAULT_BETA2,
            eps: 1e-8,
            lr: 0.01,
            momentum: 0.937,
            decoupled_wd: false,
            cautious_wd: false,
            cautious: false,
            depthwise: Vec::new(),
        }
    }

    /// Whether parameter `i` is a depthwise convolution weight.
    pub fn is_depthwise(&self, i: usize) -> bool {
        self.depthwise.get(i).copied().unwrap_or(false)
    }

    /// Set the per-parameter depthwise conv mask.
    pub fn with_depthwise(mut self, mask: Vec<bool>) -> Self {
        self.depthwise = mask;
        self
    }

    /// Set the base (and current) learning rate.
    pub fn lr(mut self, lr: f64) -> Self {
        self.base_lr = lr;
        self.lr = lr;
        self
    }

    /// Set the base (and current) momentum / β1.
    pub fn momentum(mut self, m: f64) -> Self {
        self.base_momentum = m;
        self.momentum = m;
        self
    }

    /// Set the weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable or disable Nesterov momentum (SGD).
    pub fn nesterov(mut self, on: bool) -> Self {
        self.nesterov = on;
        self
    }

    /// Set AdamW's β2 and ε.
    pub fn adam(mut self, beta2: f64, eps: f64) -> Self {
        self.beta2 = beta2;
        self.eps = eps;
        self
    }

    /// Enable or disable decoupled weight decay (SGD / MuSGD).
    pub fn decoupled_wd(mut self, on: bool) -> Self {
        self.decoupled_wd = on;
        self
    }

    /// Enable or disable cautious decoupled weight decay (SGD / MuSGD).
    pub fn cautious_wd(mut self, on: bool) -> Self {
        self.cautious_wd = on;
        self
    }

    /// Enable or disable Cautious update masking (`u · g > 0`, rescaled by `n / max(active, 1)`) for
    /// [`AdamW`] and [`Sgd`].
    pub fn cautious(mut self, on: bool) -> Self {
        self.cautious = on;
        self
    }

    fn validate(&self) -> Result<()> {
        let ok = self.base_lr.is_finite()
            && self.base_lr >= 0.0
            && (0.0..1.0).contains(&self.base_momentum)
            && self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && (0.0..1.0).contains(&self.beta2)
            && self.eps.is_finite()
            && self.eps > 0.0;
        if !ok {
            return Err(Error::Invalid(format!("optimizer group `{}`: invalid hyperparameters", self.name)));
        }
        if !self.depthwise.is_empty() && self.depthwise.len() != self.params.len() {
            return Err(Error::Invalid(format!(
                "optimizer group `{}`: depthwise mask length {} does not match {} parameters",
                self.name,
                self.depthwise.len(),
                self.params.len()
            )));
        }
        Ok(())
    }
}

/// Anything that maps a parameter to its gradient: [`Grads`] from one backward pass, or a
/// [`GradAccumulator`].
pub trait GradSource: Sync {
    /// The gradient of `param`, if it has one.
    fn grad(&self, param: &Tensor) -> Option<&Tensor>;
}

impl GradSource for Grads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.get(param)
    }
}

/// An optimizer over [`ParamGroup`]s.
pub trait Optimizer: Send {
    /// SGD, AdamW or MuSGD.
    fn kind(&self) -> OptimKind;

    /// The groups.
    fn groups(&self) -> &[ParamGroup];

    /// The groups, mutable (the schedule sets `lr` and `momentum` through this).
    fn groups_mut(&mut self) -> &mut [ParamGroup];

    /// Update every parameter that has a gradient in `grads`, under `no_grad`, then count the step.
    fn step(&mut self, grads: &dyn GradSource) -> Result<()>;

    /// Number of completed `step()` calls (`optim.__step`).
    fn step_count(&self) -> u64;

    /// The state tensors as `(<param name>.<state>, tensor)`: `momentum_buffer` (SGD with momentum > 0),
    /// `exp_avg` and `exp_avg_sq` (AdamW). Complete: every parameter of every group that uses a state has it.
    fn state_tensors(&self) -> Result<Vec<(String, Tensor)>>;

    /// Restore from a checkpoint: the step count and the state tensors (keyed `<param name>.<state>`).
    /// Missing or mismatched state is an error unless `step == 0`, where missing state stays zero
    /// (docs/FORMAT.md, optimizer contract).
    fn load_state(&mut self, step: u64, state: &HashMap<String, Tensor>) -> Result<()>;

    /// Move every group's parameters and the optimizer state (SGD momentum buffers, AdamW moments) to
    /// `device`, in place ([`Tensor::to_device_`]): identities are kept, so gradients from a model moved with
    /// [`crate::Module::to_device`] still match. State created later (a momentum buffer that warmup turns
    /// on) is allocated on its parameter's device.
    fn to_device(&mut self, device: Device) -> Result<()> {
        for g in self.groups() {
            for (_, p) in &g.params {
                p.to_device_(device)?;
            }
        }
        for (_, t) in self.state_tensors()? {
            t.to_device_(device)?;
        }
        Ok(())
    }

    /// The `folo.optim` contract.
    fn contract(&self) -> OptimContract {
        group_contract(self)
    }

    /// Check that `c` describes this optimizer (type, group names, parameter sets) and adopt its group
    /// hyperparameters (and [`MuSgd`]'s step weights). Used on resume before [`Optimizer::load_state`].
    fn apply_contract(&mut self, c: &OptimContract) -> Result<()> {
        apply_group_contract(self, c)
    }
}

/// The part of `folo.optim` every optimizer has: type, groups and step (no `musgd`).
fn group_contract<O: Optimizer + ?Sized>(o: &O) -> OptimContract {
    let kind = o.kind();
    OptimContract {
        kind,
        groups: o.groups().iter().map(|g| GroupContract::from_group(g, kind)).collect(),
        step: o.step_count(),
        musgd: None,
        muon_ortho: None,
        muon_geo: None,
        muon_depthwise: Vec::new(),
    }
}

/// Whether `shape` is structurally compatible with a depthwise `Conv2d` weight (`[C, 1, kH, kW]` with `C > 1`).
pub(crate) fn is_depthwise_conv_shape(shape: &[usize]) -> bool {
    matches!(shape, &[c, 1, kh, kw] if c > 1 && kh > 0 && kw > 0)
}

/// Check the type and parameter sets of `c` against `o` and adopt the group hyperparameters. A `musgd` object
/// is an error here: only [`MuSgd`] reads it.
fn apply_group_contract<O: Optimizer + ?Sized>(o: &mut O, c: &OptimContract) -> Result<()> {
    let kind = o.kind();
    if c.kind != kind {
        return Err(Error::Load(format!(
            "checkpoint optimizer is {}, but the run uses {}",
            c.kind.as_str(),
            kind.as_str()
        )));
    }
    if c.musgd.is_some() && kind != OptimKind::MuSgd {
        return Err(Error::Load(format!("folo.optim: a `{}` contract cannot have `musgd`", kind.as_str())));
    }
    if c.muon_ortho.is_some() && kind != OptimKind::MuSgd {
        return Err(Error::Load(format!(
            "folo.optim: a `{}` contract cannot have `muon_ortho`",
            kind.as_str()
        )));
    }
    if c.muon_geo.is_some() && kind != OptimKind::MuSgd {
        return Err(Error::Load(format!(
            "folo.optim: a `{}` contract cannot have `muon_geo`",
            kind.as_str()
        )));
    }
    if !c.muon_depthwise.is_empty() && (kind != OptimKind::MuSgd || c.muon_geo != Some(MuonGeoMode::Geo)) {
        return Err(Error::Load(
            "folo.optim: `muon_depthwise` is only valid on a `musgd` contract with `muon_geo: \"geo\"`"
                .into(),
        ));
    }
    check_param_sets(o.groups(), &c.groups)?;
    if !c.muon_depthwise.is_empty() {
        let mut seen = std::collections::HashSet::with_capacity(c.muon_depthwise.len());
        let muon_shapes: std::collections::HashMap<&str, &[usize]> = o
            .groups()
            .iter()
            .filter(|g| is_muon_group(&g.name))
            .flat_map(|g| g.params.iter().map(|(n, p)| (n.as_str(), p.dims())))
            .collect();
        for name in &c.muon_depthwise {
            if !seen.insert(name.as_str()) {
                return Err(Error::Load(format!("folo.optim: duplicate `{name}` in `muon_depthwise`")));
            }
            let Some(&shape) = muon_shapes.get(name.as_str()) else {
                return Err(Error::Load(format!(
                    "folo.optim: `muon_depthwise` names `{name}`, which is not in a Muon group"
                )));
            };
            if !is_depthwise_conv_shape(shape) {
                return Err(Error::Load(format!(
                    "folo.optim: `muon_depthwise` names `{name}` with shape {shape:?}, expected depthwise conv weight `[C > 1, 1, kH, kW]`"
                )));
            }
        }
    }
    if kind == OptimKind::MuSgd {
        for gc in &c.groups {
            if gc.cautious {
                return Err(Error::Load(format!(
                    "folo.optim group `{}`: `cautious` is only supported for `sgd` and `adamw`",
                    gc.name
                )));
            }
        }
    }
    for (g, gc) in o.groups_mut().iter_mut().zip(&c.groups) {
        gc.apply_to(g, kind)?;
    }
    Ok(())
}

/// `folo.optim` (docs/FORMAT.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimContract {
    /// `"sgd"`, `"adamw"` or `"musgd"`.
    #[serde(rename = "type")]
    pub kind: OptimKind,
    /// The groups, in order.
    pub groups: Vec<GroupContract>,
    /// Completed optimizer steps; equals `optim.__step`.
    pub step: u64,
    /// MuSGD's step weights. Written only when they differ from [`MuSgdWeights::default`], so a file with the
    /// default weights reads as before; absent, `apply_contract` keeps the weights the optimizer was built with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musgd: Option<MuSgdWeights>,
    /// MuSGD's orthogonalisation schedule when not legacy [`MuonOrtho::Ns5`] (`"polar4"` or `"polar5"`). Omitted
    /// for `Ns5` so `folo.optim` stays byte-identical to pre-Polar contracts and absent `muon_ortho` resumes as
    /// `Ns5`, while readers predating the key reject Polar-FOLO checkpoints cleanly instead of silently switching
    /// schedules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub muon_ortho: Option<MuonOrtho>,
    /// Optional `Conv-GeoMuon` mode (`"geo"`). Omitted for `Legacy` for byte-exact compat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub muon_geo: Option<MuonGeoMode>,
    /// Parameter names in Muon groups routed to the zero-GEMM depthwise Polar LMO under `muon_geo: "geo"`.
    /// Omitted when empty or in legacy mode so existing checkpoints stay byte-for-byte identical.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub muon_depthwise: Vec<String>,
}

/// One group of `folo.optim`. Only the fields that apply to the optimizer type are present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupContract {
    /// Group name.
    pub name: String,
    /// Parameter names, in canonical order.
    pub params: Vec<String>,
    /// Base learning rate.
    pub lr: f64,
    /// SGD momentum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub momentum: Option<f64>,
    /// SGD Nesterov flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nesterov: Option<bool>,
    /// Weight decay.
    pub weight_decay: f64,
    /// AdamW `[β1, β2]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub betas: Option<[f64; 2]>,
    /// AdamW ε.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eps: Option<f64>,
    /// Decoupled weight decay (SGD / MuSGD; omitted when false so older files match byte for byte).
    #[serde(default, skip_serializing_if = "is_false")]
    pub decoupled_wd: bool,
    /// Cautious decoupled weight decay (SGD / MuSGD; omitted when false).
    #[serde(default, skip_serializing_if = "is_false")]
    pub cautious_wd: bool,
    /// Cautious update masking (AdamW / SGD; omitted when false so older files match byte for byte).
    #[serde(default, skip_serializing_if = "is_false")]
    pub cautious: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl GroupContract {
    fn from_group(g: &ParamGroup, kind: OptimKind) -> Self {
        let sgd = kind != OptimKind::AdamW;
        GroupContract {
            name: g.name.clone(),
            params: g.params.iter().map(|(n, _)| n.clone()).collect(),
            lr: g.base_lr,
            momentum: sgd.then_some(g.base_momentum),
            nesterov: sgd.then_some(g.nesterov),
            weight_decay: g.weight_decay,
            betas: (!sgd).then_some([g.base_momentum, g.beta2]),
            eps: (!sgd).then_some(g.eps),
            decoupled_wd: sgd && g.decoupled_wd,
            cautious_wd: sgd && g.cautious_wd,
            cautious: kind != OptimKind::MuSgd && g.cautious,
        }
    }

    fn apply_to(&self, g: &mut ParamGroup, kind: OptimKind) -> Result<()> {
        let missing = |f: &str| Error::Load(format!("folo.optim group `{}` has no `{f}`", self.name));
        g.base_lr = self.lr;
        g.weight_decay = self.weight_decay;
        match kind {
            OptimKind::Sgd | OptimKind::MuSgd => {
                if kind == OptimKind::MuSgd && self.cautious {
                    return Err(Error::Load(format!(
                        "folo.optim group `{}`: `cautious` is only supported for `sgd` and `adamw`",
                        self.name
                    )));
                }
                g.base_momentum = self.momentum.ok_or_else(|| missing("momentum"))?;
                g.nesterov = self.nesterov.ok_or_else(|| missing("nesterov"))?;
                g.decoupled_wd = self.decoupled_wd;
                g.cautious_wd = self.cautious_wd;
            }
            OptimKind::AdamW => {
                let [b1, b2] = self.betas.ok_or_else(|| missing("betas"))?;
                g.base_momentum = b1;
                g.beta2 = b2;
                g.eps = self.eps.ok_or_else(|| missing("eps"))?;
            }
        }
        g.cautious = self.cautious;
        g.lr = g.base_lr;
        g.momentum = g.base_momentum;
        g.validate().map_err(|e| Error::Load(e.to_string()))
    }
}

fn check_param_sets(groups: &[ParamGroup], file: &[GroupContract]) -> Result<()> {
    let mut problems = Vec::new();
    let names = |gs: Vec<&str>| gs.join(", ");
    if groups.len() != file.len() || groups.iter().zip(file).any(|(g, f)| g.name != f.name) {
        problems.push(format!(
            "groups differ: run has [{}], checkpoint has [{}]",
            names(groups.iter().map(|g| g.name.as_str()).collect()),
            names(file.iter().map(|g| g.name.as_str()).collect())
        ));
    } else {
        for (g, f) in groups.iter().zip(file) {
            let ours: std::collections::BTreeSet<&str> = g.params.iter().map(|(n, _)| n.as_str()).collect();
            let theirs: std::collections::BTreeSet<&str> = f.params.iter().map(String::as_str).collect();
            if theirs.len() != f.params.len() {
                problems.push(format!("group `{}` lists a parameter twice", g.name));
            }
            for n in ours.difference(&theirs) {
                problems.push(format!("`{n}` (group `{}`) is missing from the checkpoint", g.name));
            }
            for n in theirs.difference(&ours) {
                problems.push(format!("`{n}` (group `{}`) is not a parameter of this run", g.name));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(Error::Load(format!("optimizer parameter set mismatch: {}", problems.join("; "))))
    }
}

/// Check that no parameter is in two groups, and validate hyperparameters.
fn validate_groups(groups: &[ParamGroup]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for g in groups {
        g.validate()?;
        if !names.insert(g.name.as_str()) {
            return Err(Error::Invalid(format!("optimizer: two groups are named `{}`", g.name)));
        }
        for (n, t) in &g.params {
            if !seen.insert(n.as_str()) {
                return Err(Error::Invalid(format!("optimizer: parameter `{n}` is in more than one group")));
            }
            if t.dtype() != folo_core::DType::F32 {
                return Err(Error::Invalid(format!(
                    "optimizer: parameter `{n}` is {}, expected f32",
                    t.dtype()
                )));
            }
        }
    }
    Ok(())
}

/// Look up a state tensor in a checkpoint map and check it (shape of the parameter, `F32`).
fn take_state(
    state: &HashMap<String, Tensor>,
    pname: &str,
    param: &Tensor,
    what: &str,
    required: bool,
    problems: &mut Vec<String>,
) -> Option<Tensor> {
    let key = format!("{pname}.{what}");
    match state.get(&key) {
        Some(t) if t.shape() == param.shape() && t.dtype() == folo_core::DType::F32 => Some(t.clone()),
        Some(t) => {
            problems.push(format!(
                "`optim.{key}` is {:?} {}, expected {:?} f32",
                t.dims(),
                t.dtype(),
                param.dims()
            ));
            None
        }
        None => {
            if required {
                problems.push(format!("`optim.{key}` is missing"));
            }
            None
        }
    }
}

/// Returns true for 1-D scale / gate parameters that should receive no weight decay and standard warmup
/// (like BatchNorm γ) rather than weight decay or bias warmup: FOLO v2-Iso's `WsQARepConv3x3::gamma_id`
/// and `PeakSuppress::alpha_raw`.
fn is_no_decay_scale(name: &str, t: &Tensor) -> bool {
    t.dims().len() == 1 && matches!(name.rsplit('.').next(), Some("gamma_id" | "alpha_raw"))
}

/// docs/MODEL.md §8.2's three groups, assigned by role: `decay` (conv / linear weights), `bn` (BatchNorm
/// weights and 1-D scale/gate parameters such as `gamma_id` and `alpha_raw`), `bias` (every tensor named
/// `bias`: BatchNorm biases and conv / linear biases).
#[derive(Debug, Clone, Default)]
pub struct StandardGroups {
    /// Weights that get weight decay.
    pub decay: Vec<(String, Tensor)>,
    /// BatchNorm γ and 1-D scale/gate parameters (`gamma_id`, `alpha_raw`) with no weight decay.
    pub bn: Vec<(String, Tensor)>,
    /// Biases (BatchNorm β and layer biases).
    pub bias: Vec<(String, Tensor)>,
    /// Names of depthwise conv weights (`groups > 1 && groups == in_channels == out_channels`) in `decay`.
    pub depthwise: std::collections::HashSet<String>,
}

impl StandardGroups {
    /// Split `model`'s parameters. A parameter is `bn` if it is the `weight` of a [`BatchNorm2d`] or a 1-D
    /// no-decay scale/gate parameter whose last name segment is `gamma_id` or `alpha_raw`, `bias` if its last
    /// name segment is `bias`, and `decay` otherwise. For the FOLO v1 detector this is exactly the name rule
    /// of §8.2 (`*.bn.weight` / `*.bn.bias`, `*.out.bias` / every other weight); for FOLO v2-Iso it also
    /// protects `gamma_id` and `alpha_raw` from weight decay and bias warmup.
    pub fn from_module(model: &dyn Module) -> Self {
        let mut bn_ids = std::collections::HashSet::new();
        let mut dw_ids = std::collections::HashSet::new();
        let mut note = |m: &dyn Module| {
            if let Some(bn) = (m as &dyn std::any::Any).downcast_ref::<BatchNorm2d>() {
                bn_ids.insert(bn.weight.id());
            } else if let Some(conv) = (m as &dyn std::any::Any).downcast_ref::<crate::Conv2d>() {
                let c = conv.config();
                if c.groups > 1 && c.groups == c.in_channels && c.groups == c.out_channels {
                    dw_ids.insert(conv.weight.id());
                }
            }
        };
        note(model);
        model.for_each_module(&mut |_, m| note(m));
        let mut out = StandardGroups::default();
        for (name, t) in model.named_parameters() {
            if bn_ids.contains(&t.id()) || is_no_decay_scale(&name, &t) {
                out.bn.push((name, t));
            } else if name == "bias" || name.ends_with(".bias") {
                out.bias.push((name, t));
            } else {
                if dw_ids.contains(&t.id()) {
                    out.depthwise.insert(name.clone());
                }
                out.decay.push((name, t));
            }
        }
        out
    }

    /// The three groups `decay`, `bn`, `bias` (in that order) with the same `lr` and `momentum` (μ or β1),
    /// Nesterov on, and `weight_decay` on `decay` only.
    pub fn into_groups(self, lr: f64, momentum: f64, weight_decay: f64) -> Vec<ParamGroup> {
        let mut decay =
            ParamGroup::new("decay", self.decay).lr(lr).momentum(momentum).weight_decay(weight_decay);
        let dw_mask: Vec<bool> = decay.params.iter().map(|(n, _)| self.depthwise.contains(n)).collect();
        if dw_mask.iter().any(|&b| b) {
            decay = decay.with_depthwise(dw_mask);
        }
        vec![
            decay,
            ParamGroup::new("bn", self.bn).lr(lr).momentum(momentum),
            ParamGroup::new(BIAS_GROUP, self.bias).lr(lr).momentum(momentum),
        ]
    }

    /// MuSGD's groups (docs/MODEL.md §8.2): [`StandardGroups::into_groups`], with the parameters `boost` selects
    /// moved from each group into a `<group>.cls` group at `boost_factor · lr` (the reference trains the class
    /// branch of the head at 3x the LR). Order: `decay`, `decay.cls`, `bn`, `bn.cls`, `bias`, `bias.cls`; empty
    /// `.cls` groups are left out.
    pub fn into_boosted_groups(
        self,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        boost: impl Fn(&str) -> bool,
        boost_factor: f64,
    ) -> Vec<ParamGroup> {
        let dw = self.depthwise;
        let mut out = Vec::new();
        for (name, params, wd) in
            [("decay", self.decay, weight_decay), ("bn", self.bn, 0.0), (BIAS_GROUP, self.bias, 0.0)]
        {
            let (b, rest): (Vec<_>, Vec<_>) = params.into_iter().partition(|(n, _)| boost(n));
            let is_decay = name == "decay";
            let mut g_rest = ParamGroup::new(name, rest).lr(lr).momentum(momentum).weight_decay(wd);
            if is_decay {
                let mask: Vec<bool> = g_rest.params.iter().map(|(n, _)| dw.contains(n)).collect();
                if mask.iter().any(|&d| d) {
                    g_rest = g_rest.with_depthwise(mask);
                }
            }
            out.push(g_rest);
            if !b.is_empty() {
                let mut g = ParamGroup::new(format!("{name}.cls"), b)
                    .lr(lr * boost_factor)
                    .momentum(momentum)
                    .weight_decay(wd);
                if is_decay {
                    let mask: Vec<bool> = g.params.iter().map(|(n, _)| dw.contains(n)).collect();
                    if mask.iter().any(|&d| d) {
                        g = g.with_depthwise(mask);
                    }
                }
                out.push(g);
            }
        }
        out
    }
}

/// Gradient accumulation length, docs/MODEL.md §8.1: `max(round(nbs / batch), 1)`.
pub fn accumulate_steps(nbs: usize, batch: usize) -> usize {
    ((nbs as f64 / batch.max(1) as f64).round() as usize).max(1)
}

/// Effective weight decay, docs/MODEL.md §8.2: `weight_decay · batch · accumulate / nbs`.
pub fn effective_weight_decay(weight_decay: f64, batch: usize, accumulate: usize, nbs: usize) -> f64 {
    weight_decay * batch as f64 * accumulate as f64 / nbs.max(1) as f64
}

/// What `optimizer: auto` chooses (docs/MODEL.md §8.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoChoice {
    /// MuSGD for every run (rule 4); rules 1 to 3: SGD (1, 2) or MuSGD (3) for long runs (> 10000 iterations),
    /// AdamW otherwise.
    pub kind: OptimKind,
    /// The base LR.
    pub lr0: f64,
    /// SGD momentum / AdamW β1 (0.937 under rule 4, 0.9 before).
    pub momentum: f64,
    /// 0.0 under `auto` rules 1 to 3, [`AUTO_MUSGD_WARMUP_BIAS_LR`] under rule 4.
    pub warmup_bias_lr: f64,
}

/// The current version of the `auto` rule (docs/MODEL.md §8.2). A run's settings record the rule it started with
/// (`TrainConfig.auto_rule`), so a resume keeps it; settings written before the key existed mean rule 1.
pub const AUTO_RULE: u32 = 5;

/// Rule 5's MuSGD step weights for a run from scratch (the `muon` / `sgd` defaults of Ultralytics' `MuSGD` class,
/// which its trainer overrides to 0.2 / 1.0): on the COCO proxy folo-n trains to 0.1234 AP with them against 0.1135
/// with 0.2 / 1.0; fine-tuning from pretrained weights (BCCD) is better with 0.2 / 1.0 (docs/notes/hill/r2-learn.md).
pub const AUTO_MUSGD_SCRATCH_WEIGHTS: MuSgdWeights = MuSgdWeights { muon: 0.5, sgd: 0.5 };

/// Rule 4's momentum, the configured default (docs/MODEL.md §8.1; rules 1 to 3 use 0.9).
pub const AUTO_MUSGD_MOMENTUM: f64 = 0.937;

/// Rule 4's bias-group warmup start: the configured default (docs/MODEL.md §8.1), no longer zeroed. On the COCO
/// proxy MuSGD reaches 0.113 AP with it and 0.103 with 0 (docs/notes/hill/r2-learn.md).
pub const AUTO_MUSGD_WARMUP_BIAS_LR: f64 = 0.1;

/// Rule 2's floor on the AdamW `lr0`: the reference's fit `0.002 · 5 / (4 + nc)` falls below it from 7 classes on
/// (0.000119 for 80 classes), where it trains from scratch about 5x slower than SGD (docs/notes/hill/learn.md).
pub const AUTO_ADAMW_MIN_LR0: f64 = 0.001;

/// docs/MODEL.md §8.2 `auto`, the current rule ([`AUTO_RULE`]); see [`auto_optimizer_rule`].
pub fn auto_optimizer(
    n_train: usize,
    batch: usize,
    nbs: usize,
    epochs: usize,
    num_classes: usize,
) -> AutoChoice {
    auto_optimizer_rule(AUTO_RULE, n_train, batch, nbs, epochs, num_classes)
}

/// docs/MODEL.md §8.2 `auto` by rule version. Rules 5 and 4 resolve the same optimizer here (rule 5 also picks
/// MuSGD's step weights, [`AUTO_MUSGD_SCRATCH_WEIGHTS`] from scratch, which the trainer applies). Rule 4: MuSGD at `lr0 = 0.01`, momentum 0.937
/// ([`AUTO_MUSGD_MOMENTUM`]) and `warmup_bias_lr = 0.1` ([`AUTO_MUSGD_WARMUP_BIAS_LR`]) for every run. Rules 1 to 3:
/// `iterations = ceil(n_train / max(batch, nbs)) · epochs`; above 10000, SGD at `lr0 = 0.01` (rules 1 and 2) or
/// MuSGD at `lr0 = 0.01` (rule 3, the reference trainer's choice since Ultralytics 8.4), else AdamW at
/// `round(0.002 · 5 / (4 + nc), 6)` (rule 1, the reference trainer's), raised to at least [`AUTO_ADAMW_MIN_LR0`]
/// from rule 2 on; momentum 0.9 and `warmup_bias_lr = 0`.
pub fn auto_optimizer_rule(
    rule: u32,
    n_train: usize,
    batch: usize,
    nbs: usize,
    epochs: usize,
    num_classes: usize,
) -> AutoChoice {
    if rule >= 4 {
        return AutoChoice {
            kind: OptimKind::MuSgd,
            lr0: 0.01,
            momentum: AUTO_MUSGD_MOMENTUM,
            warmup_bias_lr: AUTO_MUSGD_WARMUP_BIAS_LR,
        };
    }
    let iterations = n_train.div_ceil(batch.max(nbs).max(1)) * epochs;
    if iterations > 10_000 {
        let kind = if rule >= 3 { OptimKind::MuSgd } else { OptimKind::Sgd };
        AutoChoice { kind, lr0: 0.01, momentum: 0.9, warmup_bias_lr: 0.0 }
    } else {
        let fit = (0.002 * 5.0 / (4.0 + num_classes as f64) * 1e6).round() / 1e6;
        let lr0 = if rule == 1 { fit } else { fit.max(AUTO_ADAMW_MIN_LR0) };
        AutoChoice { kind: OptimKind::AdamW, lr0, momentum: 0.9, warmup_bias_lr: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_helpers() {
        // docs/MODEL.md §10.4: N = 800, batch 16, nbs 64.
        assert_eq!(accumulate_steps(64, 16), 4);
        assert_eq!(accumulate_steps(64, 128), 1);
        assert_eq!(effective_weight_decay(0.0005, 16, 4, 64), 0.0005);
        // Rule 4: MuSGD at lr0 0.01, momentum 0.937, warmup_bias_lr 0.1, for short and long runs alike.
        let m = AutoChoice { kind: OptimKind::MuSgd, lr0: 0.01, momentum: 0.937, warmup_bias_lr: 0.1 };
        for (n, epochs, nc) in [(2000, 30, 4), (205, 100, 3), (20_000, 20, 80), (118_287, 100, 80)] {
            assert_eq!(auto_optimizer(n, 16, 64, epochs, nc), m, "{n} images, {epochs} epochs, {nc} classes");
        }
        let a = auto_optimizer_rule(3, 2000, 16, 64, 30, 4);
        assert_eq!((a.kind, a.lr0, a.momentum, a.warmup_bias_lr), (OptimKind::AdamW, 0.00125, 0.9, 0.0));
        // Rule 2 floors the AdamW lr0 at 0.001 (from 7 classes on); rule 1 is the reference fit.
        assert_eq!(auto_optimizer_rule(3, 2000, 16, 64, 30, 80).lr0, 0.001);
        assert_eq!(auto_optimizer_rule(3, 2000, 16, 64, 30, 6).lr0, 0.001);
        assert_eq!(auto_optimizer_rule(3, 2000, 16, 64, 30, 2).lr0, 0.001667);
        assert_eq!(auto_optimizer_rule(1, 2000, 16, 64, 30, 80).lr0, 0.000119);
        assert_eq!(auto_optimizer_rule(1, 2000, 16, 64, 30, 4), a);
        // Long runs: MuSGD from rule 3 on (bias warmup 0), SGD before.
        let l = auto_optimizer_rule(3, 118_287, 16, 64, 100, 80);
        assert_eq!((l.kind, l.warmup_bias_lr), (OptimKind::MuSgd, 0.0));
        assert_eq!(auto_optimizer_rule(2, 118_287, 16, 64, 100, 80).kind, OptimKind::Sgd);
        assert_eq!(
            auto_optimizer_rule(2, 2000, 16, 64, 30, 80),
            auto_optimizer_rule(3, 2000, 16, 64, 30, 80)
        );
    }
}
