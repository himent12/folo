//! The training configuration: docs/API.md's `TrainConfig` plus docs/MODEL.md §8.1's recipe keys.

use folo_core::Device;
use folo_data::AugmentConfig;
use folo_nn::optim::{AUTO_RULE, OptimContract, OptimKind, auto_optimizer_rule, effective_weight_decay};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::loss::LossConfig;
use crate::model::HeadMode;

/// `optimizer`: `auto`, `sgd`, `adamw` or `musgd` (docs/MODEL.md §8.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerChoice {
    /// MuSGD at lr0 0.01 for every run (`auto_rule` 5 and 4; rules 1 to 3 picked AdamW for short runs), §8.2.
    #[default]
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW (decoupled weight decay).
    Adamw,
    /// MuSGD: SGD plus a Muon (orthogonalised momentum) update on conv weights.
    Musgd,
}

/// A training run's settings. The JSON shape is docs/API.md's `TrainConfig` (`dataset_id` and `weights` are the
/// caller's business: [`super::Trainer`] takes the data and the starting weights directly) extended with the
/// recipe keys of docs/MODEL.md §8.1, every one optional with the default listed there. The resolved value
/// (every default filled in) is stored in checkpoints as `folo.train` and in a run's `config.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    /// Run name (the CLI and server fill it in; informational here).
    pub name: String,
    /// The dataset: a path or a registered id (informational: the caller opens it and passes [`super::TrainData`];
    /// `folo train --resume` reopens it from here).
    pub dataset: String,
    /// How `dataset` is read (informational, like `dataset`: `folo train --resume` reopens it so, and a folder that
    /// holds both YOLO and COCO labels reads the same). Missing in configs from before it: `auto`.
    pub data_format: folo_data::Format,
    /// Architecture, `folo-n` … `folo-x`.
    pub arch: String,
    /// Epochs, `1..=10000`.
    pub epochs: u32,
    /// Images per forward / backward, `1..=1024`.
    pub batch: usize,
    /// Square training and validation size: a multiple of 32, at least 64 (MODEL.md §8.1).
    pub imgsz: u32,
    /// `auto`, `cpu` or `gpu:<n>` ([`resolve_device`]).
    pub device: String,
    /// Optimizer.
    pub optimizer: OptimizerChoice,
    /// Version of the `auto` rule (MODEL.md §8.2): 5, the current one, is 4 (MuSGD for every run, momentum 0.937,
    /// bias warmup from 0.1) with MuSGD step weights 0.5 / 0.5 from scratch; 3 picks MuSGD for long runs and AdamW
    /// (`lr0` floored at 0.001) for short ones; 2 picks SGD for long runs; 1 is the reference fit without the floor,
    /// with SGD. Missing in configs from before it: 1, so a run resumes with the rule it started with.
    #[serde(default = "legacy_auto_rule")]
    pub auto_rule: u32,
    /// Base LR (ignored by `auto`).
    pub lr0: f64,
    /// Final LR fraction: the last iteration runs at `lr0 · lrf`.
    pub lrf: f64,
    /// SGD momentum / AdamW β1 (ignored by `auto`, which uses 0.937 under rule 4 and 0.9 before).
    pub momentum: f64,
    /// Weight decay before the batch scaling of §8.2.
    pub weight_decay: f64,
    /// Warmup length in epochs (converted to iterations by §8.4: at least `warmup_floor`, or 30% of a short run).
    pub warmup_epochs: f64,
    /// The §8.4 warmup floor in iterations (still capped to 30% of the run); `0` (the default): `warmup_epochs`
    /// alone, as Ultralytics 8.4 counts it. Missing in configs from before it: 100, the floor those runs had.
    #[serde(default = "legacy_warmup_floor")]
    pub warmup_floor: u64,
    /// Momentum / β1 at the start of warmup.
    pub warmup_momentum: f64,
    /// `bias`-group LR at the start of warmup (0 under `auto`).
    pub warmup_bias_lr: f64,
    /// Cosine instead of linear LR decay.
    pub cos_lr: bool,
    /// Nominal batch: gradients accumulate over `max(round(nbs / batch), 1)` batches.
    pub nbs: usize,
    /// Keep an EMA of the weights for validation, checkpoints and export (§8.5).
    pub ema: bool,
    /// EMA decay schedule (`legacy` exponential ramp or `power` for `PC-PEMA`).
    pub ema_mode: folo_nn::EmaMode,
    /// A fine-tune's EMA window as a fraction of its optimizer steps (§8.5): a run from pretrained weights
    /// (`transfer`) averages over `ema_run_fraction · T` steps, at most the default window of 10,000, which runs of
    /// `10000 / ema_run_fraction` steps or more and every run from scratch keep; `0`: the default window for every
    /// run. Missing in configs from before it: 0, the window those runs had.
    #[serde(default = "legacy_ema_run_fraction")]
    pub ema_run_fraction: f64,
    /// Global gradient-norm clip; `0` disables it.
    pub clip_grad_norm: f64,
    /// Mixed-precision training (docs/MODEL.md §8.8): `F16` activations and convolutions / matrix products on the
    /// GPU's tensor cores with `F32` master weights, optimizer state, BatchNorm statistics, loss and EMA, and
    /// dynamic loss scaling. Needs a GPU whose context has tensor cores (folo-gpu's cooperative matrices); anything
    /// else fails with `not_supported`.
    pub amp: bool,
    /// Detection head (docs/MODEL.md §6.6): `o2m`, or `o2o` for a model with the NMS-free one-to-one branch too,
    /// trained with dual assignment (`L = w_m · L_o2m + w_o · L_o2o`) and predicting without NMS.
    pub head: HeadMode,
    /// `head = "o2o"` only: for the first this many epochs the one-to-one branch's inputs are not detached, so its
    /// loss trains the backbone and neck as well (MODEL.md §6.6); `0` (the default) detaches them throughout.
    pub o2o_attached_epochs: u32,
    /// `head = "o2o"` only: the one-to-one branch's assigner takes `loss.tal`'s `alpha`, `beta`, `eps` and
    /// `min_box_size` (its `topk = 7` and `topk2 = 1` are fixed, MODEL.md §6.6). Missing in configs from before it:
    /// `false`, the §7.3 defaults those runs trained that branch with, so they resume bit-exactly. Like the other
    /// versioned keys, a partial JSON decoded straight into this type reads as such a config; new runs of the CLI, of
    /// the API (`from_api`) and from `TrainConfig::default()` default to `true`.
    #[serde(default = "legacy_o2o_keeps_tal")]
    pub o2o_keeps_tal: bool,
    /// Stop after this many epochs without a fitness gain.
    pub patience: u32,
    /// Seed of the initial weights, the epoch order and the augmentation.
    pub seed: u64,
    /// Loader worker threads (results do not depend on it; `0` loads on the training thread).
    pub workers: usize,
    /// Augmentation (docs/DATA.md; `close_mosaic` is here too).
    pub augment: AugmentConfig,
    /// Validation batch size; `0` means `2 · batch` (§8.7).
    pub val_batch: usize,
    /// Loss gains and assigner settings (docs/MODEL.md §7; `reg_max` is the model's, whatever is given here).
    /// Missing in configs from before it: the §7 defaults.
    pub loss: LossConfig,
    /// MuSGD's step weights, class-branch LR factor and orthogonalisation schedule (docs/MODEL.md §8.2); used only
    /// when the run resolves to MuSGD (under `auto` rule 5 the step weights are `auto`'s, see
    /// [`TrainConfig::resolve_optimizer`]). Missing in configs from before it: the §8.2 defaults with legacy `ns5`
    /// orthogonalisation so older runs resume with legacy `ns5`.
    #[serde(default = "legacy_musgd_config")]
    pub musgd: MuSgdConfig,
    /// Whether the run started from pretrained weights (transfer learning). Set by [`super::Trainer::new`] from
    /// [`super::TrainOptions::weights`], whatever is given here, and stored with the run, so a resume resolves `auto`
    /// as the run started. Missing in configs from before it: `false`.
    pub transfer: bool,
    /// Dense distillation from a frozen teacher (docs/MODEL.md §8.9; `folo train --teacher`). `None`, the
    /// default, trains without one and is not written, so such a run's `folo.train` and `config.json` are those
    /// of a build without the key. Stored with the run, so a resume loads the same teacher.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distill: Option<crate::loss::DistillConfig>,
    /// Decouple weight decay from SGD/MuSGD Nesterov momentum (P3; default `false` keeps v1 bit-identical).
    pub decoupled_wd: bool,
    /// Cautious decoupled weight decay (`u · p > 0` mask) on SGD/MuSGD (P3; default `false`).
    pub cautious_wd: bool,
    /// Optional explicit effective weight decay overriding `effective_weight_decay(weight_decay, ...)` (P3 sweep
    /// `{2e-3, 4e-3, 6e-3}` for decoupled/cautious WD). Default `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wd_eff: Option<f64>,
    /// Apply exact function-preserving Floor Conv+BN Gauge Normalization on `--weights` load (P3; default `false`).
    pub gauge_norm: bool,
    /// Freeze BatchNorm running stats / use eval-mode BN during closed-mosaic epochs (P3; default `false`).
    pub bn_freeze_close_mosaic: bool,
    /// Multiplier applied to `.cls` parameter groups' LR during closed-mosaic epochs (e.g. `1/3` decays `3x -> 1x`;
    /// default `1.0` = no change).
    pub cls_lr_close_scale: f64,
    /// Differential learning-rate multiplier applied to `backbone.*` parameter groups (e.g. `0.1` for transfer
    /// fine-tuning; default `1.0` keeps the standard parameter groups unchanged).
    pub backbone_lr_scale: f64,
    /// Number of initial epochs (`0..freeze_backbone_epochs`) during which `backbone.*` weights and BatchNorm
    /// running statistics stay frozen (default `0` = train the whole model from epoch 1).
    pub freeze_backbone_epochs: u32,
    /// Opt-in GPU auto-batch sizing (`false` by default): before placing the model on a GPU in [`super::Trainer::new`],
    /// probes VRAM headroom and per-buffer limits via `folo-gpu`'s `MemoryBudget`, picks a safe physical micro-batch
    /// up to `max(batch, nbs)`, and pairs it with gradient accumulation so effective batch size and LR schedule stay
    /// consistent without OOM. Ignored on CPU and on checkpoint resume.
    pub auto_batch: bool,
    /// Target fraction of dedicated VRAM for [`TrainConfig::auto_batch`] (default `0.80`, valid in `0.25..=0.95`;
    /// `1.0 - auto_batch_target` is reserved as safety headroom).
    pub auto_batch_target: f64,
}

/// `musgd.*` (docs/MODEL.md §8.2): `p −= muon · lr · O`, the SGD half of a conv weight at `sgd · lr`, the
/// head's class branches at `cls_lr · lr0`, and the orthogonalisation schedule `ortho` (`polar4` default for new
/// runs, `polar5`, `polar3_cs9`, or legacy `ns5`). Missing `ortho` in configs written before the key reads as `ns5`
/// so older runs resume with legacy `ns5`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MuSgdConfig {
    /// Weight of the orthogonalised (Muon) update.
    pub muon: f64,
    /// Weight of the SGD update of a conv weight.
    pub sgd: f64,
    /// LR factor of the head's class branches.
    pub cls_lr: f64,
    /// Muon orthogonalisation schedule (`polar4` default for new runs; missing in older configs reads as `ns5`).
    #[serde(default = "legacy_muon_ortho")]
    pub ortho: folo_nn::optim::MuonOrtho,
    /// Geometric conditioning mode (`legacy` default; `geo` = conv aspect + depthwise Polar LMO).
    #[serde(default)]
    pub geo: folo_nn::optim::MuonGeoMode,
}

impl Default for MuSgdConfig {
    fn default() -> Self {
        let w = folo_nn::optim::MuSgdWeights::default();
        MuSgdConfig {
            muon: w.muon,
            sgd: w.sgd,
            cls_lr: super::MUSGD_CLS_LR_FACTOR,
            ortho: folo_nn::optim::MuonOrtho::default(),
            geo: folo_nn::optim::MuonGeoMode::default(),
        }
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            name: String::new(),
            dataset: String::new(),
            data_format: folo_data::Format::Auto,
            arch: "folo-n".into(),
            epochs: 100,
            batch: 16,
            imgsz: 640,
            device: "auto".into(),
            optimizer: OptimizerChoice::Auto,
            auto_rule: AUTO_RULE,
            lr0: 0.01,
            lrf: 0.01,
            momentum: 0.937,
            weight_decay: 0.0005,
            warmup_epochs: 3.0,
            warmup_floor: 0,
            warmup_momentum: 0.8,
            warmup_bias_lr: 0.1,
            cos_lr: false,
            nbs: 64,
            ema: true,
            ema_mode: folo_nn::EmaMode::Legacy,
            ema_run_fraction: 0.25,
            clip_grad_norm: 10.0,
            amp: false,
            head: HeadMode::O2m,
            o2o_attached_epochs: 0,
            o2o_keeps_tal: true,
            patience: 100,
            seed: 0,
            workers: default_workers(),
            augment: AugmentConfig::default(),
            val_batch: 0,
            loss: LossConfig::default(),
            musgd: MuSgdConfig::default(),
            transfer: false,
            distill: None,
            decoupled_wd: false,
            cautious_wd: false,
            wd_eff: None,
            gauge_norm: false,
            bn_freeze_close_mosaic: false,
            cls_lr_close_scale: 1.0,
            backbone_lr_scale: 1.0,
            freeze_backbone_epochs: 0,
            auto_batch: false,
            auto_batch_target: super::DEFAULT_AUTO_BATCH_UTILIZATION,
        }
    }
}

/// An `f32` as the `f64` of its shortest decimal form (`0.01f32` → `0.01`, not `0.009999999776482582`).
pub fn f32_decimal(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(v as f64)
}

/// The `warmup_floor` of settings written before the key existed.
fn legacy_warmup_floor() -> u64 {
    folo_nn::WARMUP_FLOOR
}

/// The `ema_run_fraction` of settings written before the key existed: the default window for every run.
fn legacy_ema_run_fraction() -> f64 {
    0.0
}

/// The `o2o_keeps_tal` of settings written before the key existed: the one-to-one branch ignored `loss.tal`.
fn legacy_o2o_keeps_tal() -> bool {
    false
}

/// The `auto_rule` of settings written before the key existed.
fn legacy_auto_rule() -> u32 {
    1
}

/// The `musgd.ortho` of settings written before Polar-FOLO existed: legacy 5-step Newton–Schulz (`ns5`).
fn legacy_muon_ortho() -> folo_nn::optim::MuonOrtho {
    folo_nn::optim::MuonOrtho::Ns5
}

/// The `musgd` block of settings written before `musgd` existed: default weights with legacy `ns5`.
fn legacy_musgd_config() -> MuSgdConfig {
    MuSgdConfig { ortho: legacy_muon_ortho(), ..MuSgdConfig::default() }
}

/// `min(available cores, 16)`: decoding and mosaic of large images (four decodes per sample) need more than 8
/// workers to keep up with a GPU (1920x1080 JPEGs: 83 mosaic samples/s on 8 workers, 140 on 16), and the batches
/// do not depend on the count.
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get().min(16))
}

/// What the optimizer settings resolve to for a given training set (docs/MODEL.md §8.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedOptim {
    /// SGD, AdamW or MuSGD.
    pub kind: OptimKind,
    /// Base LR of every group.
    pub lr0: f64,
    /// SGD μ / AdamW β1 after warmup.
    pub momentum: f64,
    /// `bias`-group LR at iteration 0.
    pub warmup_bias_lr: f64,
    /// MuSGD's step weights (used only by MuSGD).
    pub musgd: folo_nn::optim::MuSgdWeights,
}

impl TrainConfig {
    /// The API.md `TrainConfig` mapped onto the trainer's: the shared fields are copied, the augmentation fields
    /// it names override [`AugmentConfig::default`], and every recipe key keeps its MODEL.md default. An
    /// optimizer this build does not know ([`folo_api::Optimizer::Unknown`]) is [`Error::Config`].
    pub fn from_api(c: &folo_api::TrainConfig) -> Result<Self> {
        let a = &c.augment;
        Ok(TrainConfig {
            name: c.name.clone(),
            dataset: c.dataset_id.clone(),
            arch: c.arch.as_str().to_string(),
            epochs: c.epochs,
            batch: c.batch as usize,
            imgsz: c.imgsz,
            device: c.device.clone(),
            optimizer: match c.optimizer {
                folo_api::Optimizer::Auto => OptimizerChoice::Auto,
                folo_api::Optimizer::Sgd => OptimizerChoice::Sgd,
                folo_api::Optimizer::Adamw => OptimizerChoice::Adamw,
                folo_api::Optimizer::Musgd => OptimizerChoice::Musgd,
                folo_api::Optimizer::Unknown => {
                    return Err(Error::Config(
                        "optimizer: unknown value (this build knows auto, sgd, adamw and musgd)".into(),
                    ));
                }
            },
            lr0: f32_decimal(c.lr0),
            amp: c.amp,
            head: match c.head {
                folo_api::Head::O2m => HeadMode::O2m,
                folo_api::Head::O2o => HeadMode::O2o,
            },
            seed: c.seed,
            augment: AugmentConfig {
                mosaic: a.mosaic,
                mixup: a.mixup,
                hsv: a.hsv,
                fliplr: a.fliplr,
                scale: a.scale,
                translate: a.translate,
                close_mosaic: a.close_mosaic,
                ..AugmentConfig::default()
            },
            ..TrainConfig::default()
        })
    }

    /// Whether `arch` selects a FOLO v2-Iso model (`folo2-n` … `folo2-x`).
    pub fn is_v2(&self) -> bool {
        self.arch.starts_with("folo2-")
    }

    /// The scale letter of `arch` (`folo-n` or `folo2-n` → `n`).
    pub fn scale(&self) -> Result<&str> {
        let s = self
            .arch
            .strip_prefix("folo2-")
            .or_else(|| self.arch.strip_prefix("folo-"))
            .unwrap_or(&self.arch);
        match s {
            "n" | "s" | "m" | "l" | "x" => Ok(s),
            _ => Err(Error::Config(format!("arch `{}` is not one of folo-n..x or folo2-n..x", self.arch))),
        }
    }

    /// Check every field against docs/API.md's limits and MODEL.md §8.1.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::Config(m));
        self.scale()?;
        if !(1..=10_000).contains(&self.epochs) {
            return bad(format!("epochs must be in 1..=10000, got {}", self.epochs));
        }
        if !(1..=1024).contains(&self.batch) {
            return bad(format!("batch must be in 1..=1024, got {}", self.batch));
        }
        if !self.imgsz.is_multiple_of(32) || !(64..=4096).contains(&self.imgsz) {
            return bad(format!(
                "imgsz must be a multiple of 32 in 64..=4096 for training, got {}",
                self.imgsz
            ));
        }
        if !(self.lr0 > 0.0 && self.lr0 < 10.0) {
            return bad(format!("lr0 must be in (0, 10), got {}", self.lr0));
        }
        if !(self.lrf > 0.0 && self.lrf <= 1.0) {
            return bad(format!("lrf must be in (0, 1], got {}", self.lrf));
        }
        for (name, v) in [("momentum", self.momentum), ("warmup_momentum", self.warmup_momentum)] {
            if !(0.0..1.0).contains(&v) {
                return bad(format!("{name} must be in [0, 1), got {v}"));
            }
        }
        for (name, v) in [
            ("weight_decay", self.weight_decay),
            ("warmup_epochs", self.warmup_epochs),
            ("warmup_bias_lr", self.warmup_bias_lr),
            ("clip_grad_norm", self.clip_grad_norm),
            ("ema_run_fraction", self.ema_run_fraction),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(format!("{name} must be a finite number >= 0, got {v}"));
            }
        }
        if self.nbs == 0 {
            return bad("nbs must be at least 1".into());
        }
        let m = &self.musgd;
        for (name, v) in [("musgd.muon", m.muon), ("musgd.sgd", m.sgd), ("musgd.cls_lr", m.cls_lr)] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(format!("{name} must be a finite number >= 0, got {v}"));
            }
        }
        if self.val_batch > 4096 {
            return bad(format!("val_batch must be at most 4096, got {}", self.val_batch));
        }
        self.augment.validate().map_err(|e| Error::Config(format!("augment: {e}")))?;
        if !(1..=AUTO_RULE).contains(&self.auto_rule) {
            return bad(format!("auto_rule must be in 1..={AUTO_RULE}, got {}", self.auto_rule));
        }
        let l = &self.loss;
        for (name, v) in [
            ("loss.box_gain", l.box_gain),
            ("loss.cls_gain", l.cls_gain),
            ("loss.dfl_gain", l.dfl_gain),
            ("loss.tal.alpha", l.tal.alpha),
            ("loss.tal.beta", l.tal.beta),
            ("loss.tal.min_box_size", l.tal.min_box_size),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(format!("{name} must be a finite number >= 0, got {v}"));
            }
        }
        // The soft targets divide by `m_max + eps`, and `m_max` can be 0.
        if !(l.tal.eps.is_finite() && l.tal.eps > 0.0) {
            return bad(format!("loss.tal.eps must be a finite number > 0, got {}", l.tal.eps));
        }
        if l.tal.topk == 0 || l.tal.topk2 == Some(0) {
            return bad("loss.tal.topk and loss.tal.topk2 must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&l.nwd_weight) {
            return bad(format!("loss.nwd_weight must be in 0..=1, got {}", l.nwd_weight));
        }
        if !(0.0..=1.0).contains(&l.tal.nwd_weight) {
            return bad(format!("loss.tal.nwd_weight must be in 0..=1, got {}", l.tal.nwd_weight));
        }
        l.validate_params().map_err(Error::Config)?;
        if let Some(w) = self.wd_eff
            && !(w.is_finite() && w >= 0.0)
        {
            return bad(format!("wd_eff must be a finite number >= 0, got {w}"));
        }
        if !(self.cls_lr_close_scale.is_finite() && self.cls_lr_close_scale > 0.0) {
            return bad(format!(
                "cls_lr_close_scale must be a finite number > 0, got {}",
                self.cls_lr_close_scale
            ));
        }
        if !(self.backbone_lr_scale.is_finite() && self.backbone_lr_scale > 0.0) {
            return bad(format!(
                "backbone_lr_scale must be a finite number > 0, got {}",
                self.backbone_lr_scale
            ));
        }
        if !(self.auto_batch_target.is_finite()
            && (super::MIN_AUTO_BATCH_UTILIZATION..=super::MAX_AUTO_BATCH_UTILIZATION)
                .contains(&self.auto_batch_target))
        {
            return bad(format!(
                "auto_batch_target must be a finite number in {}..={}, got {}",
                super::MIN_AUTO_BATCH_UTILIZATION,
                super::MAX_AUTO_BATCH_UTILIZATION,
                self.auto_batch_target
            ));
        }
        if let Some(d) = &self.distill {
            d.validate().map_err(Error::Config)?;
        }
        let device = resolve_device(&self.device)?;
        if !self.device.trim().eq_ignore_ascii_case("auto") {
            super::can_place(device)?;
        }
        Ok(())
    }

    /// `accumulate = max(round(nbs / batch), 1)`.
    pub fn accumulate(&self) -> usize {
        folo_nn::optim::accumulate_steps(self.nbs, self.batch)
    }

    /// The validation batch: `val_batch`, or `2 · batch` when it is 0.
    pub fn val_batch(&self) -> usize {
        if self.val_batch == 0 { 2 * self.batch } else { self.val_batch }
    }

    /// docs/API.md's `Run.resolved`: what this config trains with on `n_train` images of `num_classes` classes
    /// (the optimizer of [`TrainConfig::resolve_optimizer`], the scaled weight decay, accumulation and warmup), as
    /// [`super::Trainer::resolved_train`] reports it once built.
    pub fn resolved_train(&self, n_train: usize, num_classes: usize) -> folo_api::ResolvedTrain {
        let accumulate = self.accumulate();
        let spe = n_train.div_ceil(self.batch.max(1)).div_ceil(accumulate);
        self.resolve_optimizer(n_train, num_classes).to_api(self, accumulate, spe as u64)
    }

    /// The optimizer this config trains with on `n_train` images of `num_classes` classes (§8.2 `auto`). MuSGD's
    /// step weights are `musgd.muon` / `musgd.sgd`, except under `auto` rule 5 for a run from scratch
    /// (`transfer = false`): [`folo_nn::optim::AUTO_MUSGD_SCRATCH_WEIGHTS`].
    pub fn resolve_optimizer(&self, n_train: usize, num_classes: usize) -> ResolvedOptim {
        let configured = folo_nn::optim::MuSgdWeights { muon: self.musgd.muon, sgd: self.musgd.sgd };
        match self.optimizer {
            OptimizerChoice::Auto => {
                let a = auto_optimizer_rule(
                    self.auto_rule,
                    n_train,
                    self.batch,
                    self.nbs,
                    self.epochs as usize,
                    num_classes,
                );
                let scratch = self.auto_rule >= 5 && !self.transfer;
                ResolvedOptim {
                    kind: a.kind,
                    lr0: a.lr0,
                    momentum: a.momentum,
                    warmup_bias_lr: a.warmup_bias_lr,
                    musgd: if scratch { folo_nn::optim::AUTO_MUSGD_SCRATCH_WEIGHTS } else { configured },
                }
            }
            OptimizerChoice::Sgd | OptimizerChoice::Adamw | OptimizerChoice::Musgd => ResolvedOptim {
                kind: match self.optimizer {
                    OptimizerChoice::Sgd => OptimKind::Sgd,
                    OptimizerChoice::Musgd => OptimKind::MuSgd,
                    _ => OptimKind::AdamW,
                },
                lr0: self.lr0,
                momentum: self.momentum,
                warmup_bias_lr: self.warmup_bias_lr,
                musgd: configured,
            },
        }
    }
}

impl TrainConfig {
    /// The warmup length `W` of docs/MODEL.md §8.4 in iterations, for `steps_per_epoch` optimizer steps per epoch.
    pub fn warmup_iters(&self, steps_per_epoch: u64) -> u64 {
        folo_nn::warmup_iters_floor(
            self.warmup_epochs,
            self.epochs as u64,
            steps_per_epoch,
            self.warmup_floor,
        )
    }
}

impl TrainConfig {
    /// The effective weight decay actually passed to the optimizer and recorded in run metadata (`wd_eff` when
    /// explicitly set, otherwise batch-scaled [`effective_weight_decay`]).
    pub fn effective_wd(&self, accumulate: usize) -> f64 {
        self.wd_eff
            .unwrap_or_else(|| effective_weight_decay(self.weight_decay, self.batch, accumulate, self.nbs))
    }

    /// The EMA's `(decay, tau)` in a run of `total_iters` optimizer steps (docs/MODEL.md §8.5): a run from
    /// pretrained weights (`transfer`) takes [`folo_nn::ema_for_run`] with `ema_run_fraction`, a run from scratch
    /// the default (0.9999, 2000).
    pub fn ema_params(&self, total_iters: u64) -> (f64, f64) {
        if self.transfer {
            folo_nn::ema_for_run(total_iters, self.ema_run_fraction)
        } else {
            (folo_nn::EMA_DECAY, folo_nn::EMA_TAU)
        }
    }
}

impl ResolvedOptim {
    /// docs/API.md's `Run.resolved` of a run with `cfg` that resolved to `self`, accumulating `accumulate` batches
    /// over `steps_per_epoch` optimizer steps per epoch.
    pub fn to_api(
        &self,
        cfg: &TrainConfig,
        accumulate: usize,
        steps_per_epoch: u64,
    ) -> folo_api::ResolvedTrain {
        let adamw = self.kind == OptimKind::AdamW;
        let ema = (cfg.ema && cfg.ema_mode == folo_nn::EmaMode::Legacy)
            .then(|| cfg.ema_params(cfg.epochs as u64 * steps_per_epoch));
        folo_api::ResolvedTrain {
            optimizer: match self.kind {
                OptimKind::Sgd => folo_api::ResolvedOptimizer::Sgd,
                OptimKind::AdamW => folo_api::ResolvedOptimizer::Adamw,
                OptimKind::MuSgd => folo_api::ResolvedOptimizer::Musgd,
            },
            lr0: self.lr0,
            lrf: cfg.lrf,
            cos_lr: cfg.cos_lr,
            momentum: self.momentum,
            betas: adamw.then_some([self.momentum, folo_nn::optim::DEFAULT_BETA2]),
            weight_decay: cfg.effective_wd(accumulate),
            batch: cfg.batch as u32,
            accumulate: accumulate as u32,
            nbs: cfg.nbs as u32,
            optimizer_steps_per_epoch: steps_per_epoch,
            warmup_iters: cfg.warmup_iters(steps_per_epoch),
            warmup_epochs: cfg.warmup_epochs,
            warmup_momentum: cfg.warmup_momentum,
            warmup_bias_lr: self.warmup_bias_lr,
            close_mosaic: Some(cfg.augment.close_mosaic_epochs(cfg.epochs)),
            ema_decay: ema.map(|e| e.0),
            ema_tau: ema.map(|e| e.1),
            musgd: (self.kind == OptimKind::MuSgd).then_some(folo_api::ResolvedMusgd {
                muon: self.musgd.muon,
                sgd: self.musgd.sgd,
                cls_lr: cfg.musgd.cls_lr,
            }),
        }
    }

    /// The `Run.resolved` of a run folder's `config.json` ([`super::Trainer::run_config`]): its `resolved` object,
    /// or, in files written before it, the same numbers rebuilt from `optimizer`, `accumulate`, `steps_per_epoch`,
    /// `warmup_iters` and `train`. `None` when the file has neither.
    pub fn from_run_config(v: &serde_json::Value) -> Option<folo_api::ResolvedTrain> {
        // `close_mosaic_epochs` is the capped value the trainer used (files from before `resolved.close_mosaic`).
        let close_mosaic = v.get("close_mosaic_epochs").and_then(|x| x.as_u64()).map(|n| n as u32);
        if let Some(mut r) =
            v.get("resolved").and_then(|r| serde_json::from_value::<folo_api::ResolvedTrain>(r.clone()).ok())
        {
            r.close_mosaic = r.close_mosaic.or(close_mosaic);
            // Files from before `resolved.ema_decay`: the EMA their `train` settings give (the default window).
            if r.ema_decay.is_none()
                && let Some(t) =
                    v.get("train").and_then(|t| serde_json::from_value::<TrainConfig>(t.clone()).ok())
                && t.ema
                && t.ema_mode == folo_nn::EmaMode::Legacy
            {
                let (d, tau) = t.ema_params(t.epochs as u64 * r.optimizer_steps_per_epoch);
                (r.ema_decay, r.ema_tau) = (Some(d), Some(tau));
            }
            return Some(r);
        }
        let o = v.get("optimizer")?;
        let train: TrainConfig = serde_json::from_value(v.get("train")?.clone()).ok()?;
        let f = |obj: &serde_json::Value, k: &str| obj.get(k).and_then(|x| x.as_f64());
        let n = |k: &str| v.get(k).and_then(|x| x.as_u64());
        let kind = match o.get("type")?.as_str()? {
            "sgd" => OptimKind::Sgd,
            "adamw" => OptimKind::AdamW,
            "musgd" => OptimKind::MuSgd,
            _ => return None,
        };
        let resolved = ResolvedOptim {
            kind,
            lr0: f(o, "lr0")?,
            momentum: f(o, "momentum")?,
            warmup_bias_lr: f(o, "warmup_bias_lr")?,
            musgd: train.resolve_optimizer(0, 1).musgd,
        };
        let accumulate = n("accumulate")? as usize;
        let mut r = resolved.to_api(&train, accumulate, n("steps_per_epoch")?);
        // The numbers the run trained with, where the file has them.
        r.weight_decay = f(o, "weight_decay").unwrap_or(r.weight_decay);
        r.warmup_iters = n("warmup_iters").unwrap_or(r.warmup_iters);
        r.close_mosaic = close_mosaic.or(r.close_mosaic);
        // A file this old does not say which step weights the run trained with.
        r.musgd = None;
        Some(r)
    }
}

/// MuSGD's step weights and class-head LR factor as an optimizer holds them. `recipe` is the factor the run was
/// configured with, kept when the groups hold exactly it (a ratio of two LRs can land one ulp off).
pub(super) fn musgd_in_use(c: &OptimContract, recipe: f64) -> folo_api::ResolvedMusgd {
    let w = c.musgd.unwrap_or_default();
    let lrs = c.groups.iter().find_map(|g| {
        let plain = g.name.strip_suffix(".cls")?;
        Some((c.groups.iter().find(|b| b.name == plain)?.lr, g.lr))
    });
    let cls_lr = match lrs {
        Some((base, cls)) if base > 0.0 && base * recipe != cls => cls / base,
        _ => recipe,
    };
    folo_api::ResolvedMusgd { muon: w.muon, sgd: w.sgd, cls_lr }
}

/// Parse a device string: `cpu`, `gpu` (= `gpu:0`), `gpu:<n>`, or `auto` (the first GPU with a registered
/// folo-core backend, else the CPU). A GPU without a registered backend is an error.
pub fn resolve_device(s: &str) -> Result<Device> {
    let gpu = |i: usize| {
        let d = Device::Gpu(i);
        folo_core::backend::backend(d).map(|_| d).map_err(|_| super::device_not_found(d))
    };
    match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "auto" | "" => Ok(gpu(0).unwrap_or(Device::Cpu)),
        "gpu" | "cuda" => gpu(0),
        other => {
            let n = other.strip_prefix("gpu:").or_else(|| other.strip_prefix("cuda:"));
            match n.and_then(|n| n.parse::<usize>().ok()) {
                Some(i) => gpu(i),
                None => Err(Error::Config(format!("device must be auto, cpu or gpu:<n>, got `{s}`"))),
            }
        }
    }
}

#[cfg(test)]
mod tests;
