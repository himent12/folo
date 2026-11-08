//! The training configuration: docs/API.md's `TrainConfig` plus docs/MODEL.md §8.1's recipe keys.

use folo_core::Device;
use folo_data::AugmentConfig;
use folo_nn::optim::{OptimKind, auto_optimizer};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// `optimizer`: `auto`, `sgd` or `adamw` (docs/MODEL.md §8.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerChoice {
    /// AdamW for short runs (≤ 10000 iterations), SGD otherwise, with the LR of §8.2.
    #[default]
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW (decoupled weight decay).
    Adamw,
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
    /// Base LR (ignored by `auto`).
    pub lr0: f64,
    /// Final LR fraction: the last iteration runs at `lr0 · lrf`.
    pub lrf: f64,
    /// SGD momentum / AdamW β1 (ignored by `auto`, which uses 0.9).
    pub momentum: f64,
    /// Weight decay before the batch scaling of §8.2.
    pub weight_decay: f64,
    /// Warmup length in epochs (converted to iterations, at least 100, by §8.4).
    pub warmup_epochs: f64,
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
    /// Global gradient-norm clip; `0` disables it.
    pub clip_grad_norm: f64,
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
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            name: String::new(),
            dataset: String::new(),
            arch: "folo-n".into(),
            epochs: 100,
            batch: 16,
            imgsz: 640,
            device: "auto".into(),
            optimizer: OptimizerChoice::Auto,
            lr0: 0.01,
            lrf: 0.01,
            momentum: 0.937,
            weight_decay: 0.0005,
            warmup_epochs: 3.0,
            warmup_momentum: 0.8,
            warmup_bias_lr: 0.1,
            cos_lr: false,
            nbs: 64,
            ema: true,
            clip_grad_norm: 10.0,
            patience: 100,
            seed: 0,
            workers: default_workers(),
            augment: AugmentConfig::default(),
            val_batch: 0,
        }
    }
}

/// An `f32` as the `f64` of its shortest decimal form (`0.01f32` → `0.01`, not `0.009999999776482582`).
pub fn f32_decimal(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(v as f64)
}

/// `min(available cores, 8)`.
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get().min(8))
}

/// What the optimizer settings resolve to for a given training set (docs/MODEL.md §8.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedOptim {
    /// SGD or AdamW.
    pub kind: OptimKind,
    /// Base LR of every group.
    pub lr0: f64,
    /// SGD μ / AdamW β1 after warmup.
    pub momentum: f64,
    /// `bias`-group LR at iteration 0.
    pub warmup_bias_lr: f64,
}

impl TrainConfig {
    /// The API.md `TrainConfig` mapped onto the trainer's: the shared fields are copied, the augmentation fields
    /// it names override [`AugmentConfig::default`], and every recipe key keeps its MODEL.md default.
    pub fn from_api(c: &folo_api::TrainConfig) -> Self {
        let a = &c.augment;
        TrainConfig {
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
            },
            lr0: f32_decimal(c.lr0),
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
        }
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
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(format!("{name} must be a finite number >= 0, got {v}"));
            }
        }
        if self.nbs == 0 {
            return bad("nbs must be at least 1".into());
        }
        if self.val_batch > 4096 {
            return bad(format!("val_batch must be at most 4096, got {}", self.val_batch));
        }
        self.augment.validate().map_err(|e| Error::Config(format!("augment: {e}")))?;
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

    /// The optimizer this config trains with on `n_train` images of `num_classes` classes (§8.2 `auto`).
    pub fn resolve_optimizer(&self, n_train: usize, num_classes: usize) -> ResolvedOptim {
        match self.optimizer {
            OptimizerChoice::Auto => {
                let a = auto_optimizer(n_train, self.batch, self.nbs, self.epochs as usize, num_classes);
                ResolvedOptim {
                    kind: a.kind,
                    lr0: a.lr0,
                    momentum: a.momentum,
                    warmup_bias_lr: a.warmup_bias_lr,
                }
            }
            OptimizerChoice::Sgd | OptimizerChoice::Adamw => ResolvedOptim {
                kind: if self.optimizer == OptimizerChoice::Sgd { OptimKind::Sgd } else { OptimKind::AdamW },
                lr0: self.lr0,
                momentum: self.momentum,
                warmup_bias_lr: self.warmup_bias_lr,
            },
        }
    }
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
