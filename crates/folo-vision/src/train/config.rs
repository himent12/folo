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
    /// MuSGD's step weights and class-branch LR factor (docs/MODEL.md §8.2); used only when the run resolves to
    /// MuSGD (under `auto` rule 5 the step weights are `auto`'s, see [`TrainConfig::resolve_optimizer`]). Missing in
    /// configs from before it: the §8.2 defaults.
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
}

/// `musgd.*` (docs/MODEL.md §8.2): `p −= muon · lr · O`, the SGD half of a conv weight at `sgd · lr`, and the
/// head's class branches at `cls_lr · lr0`. The defaults are the reference trainer's (Ultralytics 8.4).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MuSgdConfig {
    /// Weight of the orthogonalised (Muon) update.
    pub muon: f64,
    /// Weight of the SGD update of a conv weight.
    pub sgd: f64,
    /// LR factor of the head's class branches.
    pub cls_lr: f64,
}

impl Default for MuSgdConfig {
    fn default() -> Self {
        let w = folo_nn::optim::MuSgdWeights::default();
        MuSgdConfig { muon: w.muon, sgd: w.sgd, cls_lr: super::MUSGD_CLS_LR_FACTOR }
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
            ema_run_fraction: 0.25,
            clip_grad_norm: 10.0,
            amp: false,
            head: HeadMode::O2m,
            o2o_attached_epochs: 0,
            patience: 100,
            seed: 0,
            workers: default_workers(),
            augment: AugmentConfig::default(),
            val_batch: 0,
            loss: LossConfig::default(),
            musgd: MuSgdConfig::default(),
            transfer: false,
            distill: None,
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

/// The `auto_rule` of settings written before the key existed.
fn legacy_auto_rule() -> u32 {
    1
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

    /// The scale letter of `arch` (`folo-n` → `n`).
    pub fn scale(&self) -> Result<&str> {
        let s = self.arch.strip_prefix("folo-").unwrap_or(&self.arch);
        match s {
            "n" | "s" | "m" | "l" | "x" => Ok(s),
            _ => Err(Error::Config(format!(
                "arch `{}` is not one of folo-n, folo-s, folo-m, folo-l, folo-x",
                self.arch
            ))),
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
        let ema = cfg.ema.then(|| cfg.ema_params(cfg.epochs as u64 * steps_per_epoch));
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
            weight_decay: effective_weight_decay(cfg.weight_decay, cfg.batch, accumulate, cfg.nbs),
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
mod tests {
    use super::*;

    /// `auto_rule`: new configs use rule 4 (MuSGD for every run, momentum 0.937, bias warmup from 0.1), settings stored before
    /// the key existed (a checkpoint's `folo.train`) resume with rule 1, the reference fit; rule 3 keeps AdamW for
    /// short runs; other versions are invalid.
    #[test]
    fn auto_rule_versions() {
        let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
        assert_eq!(cfg.auto_rule, AUTO_RULE);
        let m = cfg.resolve_optimizer(2000, 80);
        assert_eq!((m.kind, m.lr0, m.momentum, m.warmup_bias_lr), (OptimKind::MuSgd, 0.01, 0.937, 0.1));
        let r = cfg.resolved_train(2000, 80);
        assert_eq!((r.optimizer, r.betas, r.warmup_bias_lr), (folo_api::ResolvedOptimizer::Musgd, None, 0.1));
        let r3 = TrainConfig { auto_rule: 3, ..cfg.clone() };
        assert_eq!(r3.resolve_optimizer(2000, 80).lr0, 0.001);
        assert_eq!(r3.resolve_optimizer(2000, 4).lr0, 0.00125, "4 classes are above the floor");
        let mut old = serde_json::to_value(&cfg).unwrap();
        old.as_object_mut().unwrap().remove("auto_rule");
        let old: TrainConfig = serde_json::from_value(old).unwrap();
        assert_eq!(old.auto_rule, 1);
        assert_eq!(old.resolve_optimizer(2000, 80).lr0, 0.000119);
        // Long runs: MuSGD (rules 3 and 4) vs SGD (rules 1 and 2), lr0 0.01 and momentum 0.9 either way.
        let long = TrainConfig { epochs: 300, auto_rule: 3, ..cfg.clone() };
        let m = long.resolve_optimizer(118_287, 80);
        assert_eq!((m.kind, m.lr0, m.momentum, m.warmup_bias_lr), (OptimKind::MuSgd, 0.01, 0.9, 0.0));
        for rule in [1, 2] {
            assert_eq!(
                TrainConfig { auto_rule: rule, ..long.clone() }.resolve_optimizer(118_287, 80).kind,
                OptimKind::Sgd
            );
        }
        assert!(TrainConfig { auto_rule: 0, ..cfg.clone() }.validate().is_err());
        assert!(TrainConfig { auto_rule: AUTO_RULE + 1, ..cfg }.validate().is_err());
    }

    /// `Run.resolved` says which MuSGD step weights a run trains with (not the configured `train.musgd`), only for
    /// MuSGD, and a `config.json` written before the field still loads, without it.
    #[test]
    fn resolved_names_the_musgd_step_weights() {
        let musgd = |muon, sgd| Some(folo_api::ResolvedMusgd { muon, sgd, cls_lr: 3.0 });
        let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
        assert_eq!(cfg.resolved_train(2000, 80).musgd, musgd(0.5, 0.5), "auto, from scratch");
        let tuned = TrainConfig { transfer: true, ..cfg.clone() };
        assert_eq!(tuned.resolved_train(2000, 80).musgd, musgd(0.2, 1.0), "auto, a fine-tune");
        let named = TrainConfig { optimizer: OptimizerChoice::Musgd, ..cfg.clone() };
        assert_eq!(named.resolved_train(2000, 80).musgd, musgd(0.2, 1.0), "asked for by name");
        let sgd = TrainConfig { optimizer: OptimizerChoice::Sgd, ..cfg.clone() };
        assert_eq!(sgd.resolved_train(2000, 80).musgd, None);
        let json = serde_json::to_value(sgd.resolved_train(2000, 80)).unwrap();
        assert!(json.get("musgd").is_none(), "{json}");
        // Written before the field: a `resolved` without it, and a file from before `resolved` itself.
        let mut old = serde_json::to_value(cfg.resolved_train(2000, 80)).unwrap();
        old.as_object_mut().unwrap().remove("musgd");
        let file = serde_json::json!({ "resolved": old });
        assert_eq!(ResolvedOptim::from_run_config(&file).unwrap().musgd, None);
        let file = serde_json::json!({
            "optimizer": { "type": "musgd", "lr0": 0.01, "momentum": 0.937, "warmup_bias_lr": 0.1 },
            "accumulate": 4, "steps_per_epoch": 13, "train": serde_json::to_value(&cfg).unwrap(),
        });
        assert_eq!(ResolvedOptim::from_run_config(&file).unwrap().musgd, None);
    }

    /// `auto` rule 5 picks MuSGD's step weights: 0.5 / 0.5 from scratch, `musgd.*` (0.2 / 1.0) for a run that starts
    /// from weights and for any explicit `musgd` or older rule; `transfer` is missing (false) in older configs.
    #[test]
    fn auto_rule_5_musgd_weights() {
        use folo_nn::optim::{AUTO_MUSGD_SCRATCH_WEIGHTS, MuSgdWeights};
        let cfg = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
        assert_eq!(cfg.auto_rule, 5);
        assert_eq!(cfg.resolve_optimizer(2000, 80).musgd, AUTO_MUSGD_SCRATCH_WEIGHTS);
        let tuned = TrainConfig { transfer: true, ..cfg.clone() };
        assert_eq!(tuned.resolve_optimizer(2000, 80).musgd, MuSgdWeights::default());
        let rule4 = TrainConfig { auto_rule: 4, ..cfg.clone() };
        assert_eq!(rule4.resolve_optimizer(2000, 80).musgd, MuSgdWeights::default());
        assert_eq!(rule4.resolve_optimizer(2000, 80).kind, cfg.resolve_optimizer(2000, 80).kind);
        let mut explicit = TrainConfig { optimizer: OptimizerChoice::Musgd, ..cfg.clone() };
        explicit.musgd.muon = 0.3;
        assert_eq!(explicit.resolve_optimizer(2000, 80).musgd, MuSgdWeights { muon: 0.3, sgd: 1.0 });
        let mut old = serde_json::to_value(&tuned).unwrap();
        old.as_object_mut().unwrap().remove("transfer");
        assert!(!serde_json::from_value::<TrainConfig>(old).unwrap().transfer);
    }

    /// `warmup_floor`: 0 by default, so `warmup_epochs` alone sets the warmup (a 205-image fine-tune, 4 steps per
    /// epoch: 12 warmup steps); configs from before the key keep §8.4's 100-iteration floor.
    #[test]
    fn warmup_floor_versions() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.warmup_iters(4), 12);
        assert_eq!(TrainConfig { warmup_floor: 100, ..cfg.clone() }.warmup_iters(4), 100);
        let mut old = serde_json::to_value(cfg).unwrap();
        old.as_object_mut().unwrap().remove("warmup_floor");
        let old: TrainConfig = serde_json::from_value(old).unwrap();
        assert_eq!(old.warmup_floor, 100);
    }

    /// `ema_run_fraction`: 0.25 by default, so a fine-tune of 400 optimizer steps averages over 100 (decay 0.99,
    /// tau 20) and one of 40,000 steps or more, or any run from scratch, keeps the default EMA exactly; configs from
    /// before the key keep the default for every run; a negative or non-finite fraction is refused.
    #[test]
    fn ema_run_fraction_versions() {
        use folo_nn::{EMA_DECAY, EMA_TAU};
        let tuned = TrainConfig { transfer: true, ..TrainConfig::default() };
        assert_eq!(tuned.ema_run_fraction, 0.25);
        assert_eq!(tuned.ema_params(400), (0.99, 20.0));
        let default = (EMA_DECAY.to_bits(), EMA_TAU.to_bits());
        let bits = |(d, t): (f64, f64)| (d.to_bits(), t.to_bits());
        assert_eq!(bits(tuned.ema_params(40_000)), default, "a COCO-length fine-tune");
        assert_eq!(bits(tuned.ema_params(1_000_000)), default);
        assert_eq!(tuned.ema_params(39_996), (1.0 - 1.0 / 9999.0, 9999.0 / 5.0), "just under the cap");
        let scratch = TrainConfig::default();
        assert_eq!(bits(scratch.ema_params(400)), default, "a run from scratch");
        let mut old = serde_json::to_value(&tuned).unwrap();
        old.as_object_mut().unwrap().remove("ema_run_fraction");
        let old: TrainConfig = serde_json::from_value(old).unwrap();
        assert_eq!((old.ema_run_fraction, bits(old.ema_params(400))), (0.0, default));
        // `Run.resolved` carries the pair; a run without an EMA has none.
        let r = tuned.resolved_train(205, 3);
        assert_eq!((r.ema_decay, r.ema_tau), (Some(0.99), Some(20.0)), "BCCD: 4 steps an epoch, 400 in all");
        let off = TrainConfig { ema: false, ..tuned.clone() }.resolved_train(205, 3);
        assert_eq!((off.ema_decay, off.ema_tau), (None, None));
        // A config.json from before the fields: the pair its `train` gives (a fine-tune before the key: the default).
        let mut file = serde_json::json!({ "resolved": r, "train": serde_json::to_value(&tuned).unwrap() });
        file["resolved"].as_object_mut().unwrap().retain(|k, _| !k.starts_with("ema_"));
        file["train"].as_object_mut().unwrap().remove("ema_run_fraction");
        let back = ResolvedOptim::from_run_config(&file).unwrap();
        assert_eq!((back.ema_decay, back.ema_tau), (Some(EMA_DECAY), Some(EMA_TAU)));
        for f in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(TrainConfig { ema_run_fraction: f, ..tuned.clone() }.validate().is_err(), "{f} passed");
        }
    }

    /// `loss.tal.eps` divides the soft targets (`m · u_max / (m_max + eps)`): at 0 an anchor whose prediction
    /// misses its box gets a NaN target score, so validation takes only a finite, strictly positive value.
    #[test]
    fn tal_eps_must_be_finite_and_positive() {
        use crate::boxes::Xyxy;
        let zero = crate::assign::TalConfig { eps: 0.0, ..Default::default() };
        let gt = Xyxy::new(0.0, 0.0, 32.0, 32.0);
        let miss = Xyxy::new(100.0, 100.0, 110.0, 110.0);
        let a = crate::assign::assign(&[0.5], 1, &[miss], &[[16.0, 16.0]], &[gt], &[0], &zero);
        assert!(a.target_scores[0].is_nan(), "eps 0 divides 0 by 0");
        let cfg = TrainConfig::default();
        assert!(cfg.validate().is_ok());
        for eps in [0.0, -1e-9, f32::NAN, f32::INFINITY] {
            let mut c = cfg.clone();
            c.loss.tal.eps = eps;
            assert!(c.validate().is_err(), "loss.tal.eps = {eps} passed");
        }
    }

    /// `Run.resolved`: `auto` resolved for a short run, and the same numbers back from a `config.json` with the
    /// `resolved` object and from one written before it (rebuilt from its other keys).
    #[test]
    fn resolved_train_and_run_config() {
        // Rule 3 and the 100-step warmup floor, the settings of runs made before rule 4.
        let cfg =
            TrainConfig { epochs: 30, batch: 8, auto_rule: 3, warmup_floor: 100, ..TrainConfig::default() };
        let r = cfg.resolved_train(1000, 3);
        // 125 batches · 30 epochs over max(batch, nbs) = 64 is 480 iterations: a short run, so AdamW at
        // 0.002 · 5 / (4 + 3), its β1 0.9 and β2 0.999, no warmup bias LR.
        assert_eq!(r.optimizer, folo_api::ResolvedOptimizer::Adamw);
        assert_eq!((r.lr0, r.momentum, r.betas, r.warmup_bias_lr), (0.001429, 0.9, Some([0.9, 0.999]), 0.0));
        // accumulate round(64 / 8) = 8; 125 batches → 16 optimizer steps; weight decay 0.0005 · 8 · 8 / 64.
        assert_eq!((r.batch, r.accumulate, r.nbs, r.optimizer_steps_per_epoch), (8, 8, 64, 16));
        assert!((r.weight_decay - 0.0005).abs() < 1e-12, "{}", r.weight_decay);
        // 3 warmup epochs of 16 steps = 48 < the 100-step floor (30% of 480 = 144 allows it).
        assert_eq!(r.warmup_iters, 100);
        assert_eq!((r.warmup_epochs, r.warmup_momentum, r.lrf, r.cos_lr), (3.0, 0.8, 0.01, false));
        // An explicit optimizer keeps the config's LR and momentum; SGD has no betas.
        let sgd =
            TrainConfig { optimizer: OptimizerChoice::Sgd, lr0: 0.02, ..cfg.clone() }.resolved_train(1000, 3);
        assert_eq!(
            (sgd.optimizer, sgd.lr0, sgd.momentum, sgd.betas),
            (folo_api::ResolvedOptimizer::Sgd, 0.02, 0.937, None)
        );

        let new = serde_json::json!({ "resolved": r, "optimizer": { "type": "sgd" } });
        assert_eq!(ResolvedOptim::from_run_config(&new), Some(r.clone()), "the resolved object wins");
        let old = serde_json::json!({
            "train": cfg,
            "optimizer": { "type": "adamw", "lr0": 0.001429, "momentum": 0.9, "warmup_bias_lr": 0.0,
                           "weight_decay": 0.0005 },
            "batches_per_epoch": 125, "accumulate": 8, "steps_per_epoch": 16, "warmup_iters": 100,
        });
        assert_eq!(ResolvedOptim::from_run_config(&old), Some(r));
        for broken in
            [serde_json::json!({}), serde_json::json!({ "optimizer": { "type": "auto" }, "train": {} })]
        {
            assert_eq!(ResolvedOptim::from_run_config(&broken), None, "{broken}");
        }
    }
}
