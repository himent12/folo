//! The trainer: docs/MODEL.md §8 end to end.
//!
//! [`Trainer`] wires folo-data's [`DataLoader`] to the model, the loss, folo-nn's optimizers, schedule, EMA and
//! checkpoints:
//!
//! - **Batches** (§8.3): the loader shuffles with `(seed, epoch)` and augments each sample from
//!   `(seed, epoch, position)`, so a resumed epoch reproduces its batches exactly. `accumulate =
//!   max(round(nbs / batch), 1)` batches form a window; windows never cross an epoch boundary.
//! - **Step** (§8.3–8.5): train-mode forward, [`detection_loss`], and (only if the loss and every BatchNorm
//!   statistic are finite) backward into a [`GradAccumulator`]. At the end of a window with at least one finite
//!   batch: the schedule sets each group's LR and momentum for iteration `t`, the global gradient norm is clipped,
//!   the optimizer steps, the EMA updates and `t` advances. A non-finite batch restores the running statistics it
//!   touched, emits a `warn` log and no `step` event; ten in a row, or an epoch without a finite batch, fail the
//!   run.
//! - **Validation** (§8.7) after every epoch with the EMA model (or the live model in eval mode), then the
//!   fitness (mAP@0.5:0.95), `weights/last.folo` every epoch and `weights/best.folo` on improvement (complete
//!   FORMAT.md checkpoints, written atomically), and early stopping after `patience` epochs without a gain.
//! - **Events**: docs/API.md's `status`, `step` (rate-limited), `epoch` and `log` objects go to an [`EventSink`]
//!   and, when the run has a directory, every one of them (every step) to `metrics.jsonl`.
//! - **Resume** ([`Trainer::resume`]) restores the model, EMA, optimizer, schedule and early-stopping state from
//!   an epoch-boundary checkpoint; the following steps are bit-identical to an uninterrupted run.
//!
//! Run directory layout (when [`TrainOptions::run_dir`] is set): `config.json` (the resolved config and the
//! derived schedule numbers), `metrics.jsonl`, `weights/{last,best}.folo` and `samples/epoch_NNN.json` (the
//! predictions on the first 16 validation images, docs/API.md `/samples`).
//!
//! The device is a parameter: inputs, the loss and the optimizers follow the model's device, which
//! [`place_model`] sets (`Module::to_device`).

mod config;
mod events;
mod feed;

pub use config::{OptimizerChoice, ResolvedOptim, TrainConfig, default_workers, f32_decimal, resolve_device};
pub use events::{EventSink, channel_sink, event_record, rfc3339_now};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use folo_api::{LogLevel, RunEvent, RunStatus};
use folo_core::Device;
use folo_core::rng::Rng;
use folo_data::{Batch, DataLoader, Dataset, LoaderConfig};
use folo_nn::io::{BestState, CheckpointParts, Metrics, ModelConfig, ModelFile, RngState, save_checkpoint};
use folo_nn::optim::{
    AdamW, GradAccumulator, OptimKind, Optimizer, Sgd, StandardGroups, effective_weight_decay,
};
use folo_nn::{DecayKind, EmaState, LrSchedule, ModelEma, Module, RunningStats};

use crate::error::{Error, Result};
use crate::loss::{LossConfig, LossItems, detection_loss, targets_from_normalized};
use crate::model::FoloModel;
use crate::val::{ValConfig, ValResult, ValSample, batch_tensor, val_metrics, validate};
use events::Emitter;

/// Consecutive non-finite batches that fail the run (§8.3).
pub const MAX_NONFINITE_BATCHES: usize = 10;

/// The extra `folo.train` key holding the number of training images, checked on resume.
pub const TRAIN_IMAGES_KEY: &str = "train_images";

/// The extra `folo.train` key holding the epochs at the end of the run with mosaic off
/// ([`folo_data::AugmentConfig::close_mosaic_epochs`]), so a resume keeps them. A checkpoint without it predates
/// the cap of docs/MODEL.md §8.6 ([`resumed_close_mosaic`]).
pub const CLOSE_MOSAIC_KEY: &str = "close_mosaic_epochs";

/// The closing epochs a resumed run keeps: [`CLOSE_MOSAIC_KEY`] of its `folo.train`, or for a checkpoint from
/// before the cap, the old rule `min(close_mosaic, epochs)`.
pub fn resumed_close_mosaic(train: &serde_json::Value, cfg: &TrainConfig) -> u32 {
    match train.get(CLOSE_MOSAIC_KEY).and_then(|v| v.as_u64()) {
        Some(n) => u32::try_from(n).unwrap_or(u32::MAX).min(cfg.epochs),
        None => cfg.augment.close_mosaic.min(cfg.epochs),
    }
}

/// Checkpoint metadata key (FORMAT.md: unknown `folo.*` keys are preserved) holding the run of non-finite batches
/// at the end of the checkpoint's epoch, so a streak crossing a resume still fails at [`MAX_NONFINITE_BATCHES`].
pub const NONFINITE_STREAK_KEY: &str = "folo.nonfinite_streak";

/// Runs from scratch with fewer optimizer steps than this get a warning ([`schedule_warnings`]).
pub const SHORT_RUN_STEPS: u64 = 200;

/// Warnings for a run too short to learn from, emitted as `warn` log lines after the run's first log line and
/// kept in `config.json` (`warnings`):
/// - a run from scratch (`from_scratch`) with fewer than [`SHORT_RUN_STEPS`] optimizer steps;
/// - a warmup longer than half of the run (docs/MODEL.md §8.4 caps only its 100-step floor, at 30% of the run,
///   so this takes an explicit `warmup_epochs` close to `epochs`).
pub fn schedule_warnings(
    cfg: &TrainConfig,
    sched: &LrSchedule,
    train_images: usize,
    from_scratch: bool,
) -> Vec<String> {
    let c = sched.contract();
    let (total, warmup, spe) = (sched.total_iters(), c.warmup_iters, c.steps_per_epoch);
    let mut out = Vec::new();
    if from_scratch && total < SHORT_RUN_STEPS {
        let acc = cfg.accumulate();
        let batch = if acc > 1 {
            format!("batch {} × accumulate {acc}", cfg.batch)
        } else {
            format!("batch {}", cfg.batch)
        };
        out.push(format!(
            "this run takes only {total} optimizer steps ({} epochs × {spe}: {train_images} training images, {batch}); \
             a detector trained from scratch usually needs a few hundred steps to learn anything, so expect an mAP \
             near 0. Train for at least {} epochs, add images, or start from pretrained weights",
            cfg.epochs,
            SHORT_RUN_STEPS.div_ceil(spe.max(1)),
        ));
    }
    if warmup.saturating_mul(2) > total {
        out.push(format!(
            "warmup takes {warmup} of the run's {total} optimizer steps, so the learning rate reaches lr0 only in the \
             second half of the run; lower warmup_epochs (now {}) or train for more epochs",
            cfg.warmup_epochs
        ));
    }
    out
}

/// Early stopping (§8.7): record `fitness` of 0-based epoch `e` in `best`. Returns `(improved, stop)`: whether it
/// is a new best, and whether the run stops after this epoch (`e − best.epoch >= patience`).
pub fn update_best(best: &mut BestState, e: u32, fitness: f64, patience: u32) -> (bool, bool) {
    let improved = best.fitness.is_none_or(|b| fitness > b);
    if improved {
        *best = BestState { epoch: e as i64, fitness: Some(fitness) };
    }
    (improved, e as i64 - best.epoch >= patience as i64)
}

/// The data of a run.
#[derive(Clone)]
pub struct TrainData {
    /// Training split.
    pub train: Arc<dyn Dataset>,
    /// Validation split; when empty, the training split is validated instead (with a warning).
    pub val: Arc<dyn Dataset>,
    /// Class names, index = class id.
    pub classes: Vec<String>,
}

/// How a run is observed and controlled (none of this changes the numbers it produces).
pub struct TrainOptions {
    /// Write `config.json`, `metrics.jsonl`, checkpoints and samples here. `None` keeps everything in memory.
    pub run_dir: Option<PathBuf>,
    /// Event consumer.
    pub events: Option<EventSink>,
    /// Minimum interval between `step` events sent to `events` (docs/API.md: about 10 per second).
    pub step_interval: Duration,
    /// Set to stop the run: the current batch finishes, the partial epoch is discarded and the last epoch-boundary
    /// checkpoint is kept (status `stopped`).
    pub stop: Arc<AtomicBool>,
    /// Start from these weights (transfer learning: matching tensors load, the class outputs are re-initialised
    /// when the class list differs). Ignored by [`Trainer::resume`].
    pub weights: Option<PathBuf>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            run_dir: None,
            events: None,
            step_interval: Duration::from_millis(100),
            stop: Arc::new(AtomicBool::new(false)),
            weights: None,
        }
    }
}

/// Settings a resume may change (they do not affect the numbers).
#[derive(Clone, Debug, Default)]
pub struct ResumeOverrides {
    /// Loader worker threads.
    pub workers: Option<usize>,
    /// Device.
    pub device: Option<String>,
}

/// How a run ended.
#[derive(Clone, Debug)]
pub struct TrainSummary {
    /// `completed` (all epochs, or early stopping) or `stopped` (the stop flag).
    pub status: RunStatus,
    /// Epochs completed over the whole run (including those before a resume).
    pub epochs_completed: u32,
    /// True if patience ended the run before `epochs`.
    pub early_stopped: bool,
    /// Best epoch (1-based) and its fitness (mAP@0.5:0.95), if any epoch was validated.
    pub best: Option<(u32, f64)>,
    /// The best epoch's metrics.
    pub best_metrics: Option<folo_api::ValMetrics>,
    /// Every epoch validated by this process, in order.
    pub epochs: Vec<folo_api::Epoch>,
    /// `weights/last.folo`, if written.
    pub last: Option<PathBuf>,
    /// `weights/best.folo`, if written.
    pub best_weights: Option<PathBuf>,
}

/// Put a freshly built (CPU) model on `device` (`Module::to_device`: parameters, buffers and, through them, the
/// optimizer state and EMA built from it). [`can_place`] says whether that device can be used.
pub fn place_model(mut model: FoloModel, device: Device) -> Result<FoloModel> {
    can_place(device)?;
    if !device.is_cpu() {
        model.to_device(device)?;
    }
    Ok(model)
}

/// Whether [`place_model`] can put a model on `device`: the CPU always, a GPU once its folo-core backend is
/// registered (folo-gpu's `init()`).
pub fn can_place(device: Device) -> Result<()> {
    if device.is_cpu() || folo_core::backend::backend(device).is_ok() {
        Ok(())
    } else {
        Err(device_not_found(device))
    }
}

/// `device` cannot be used: no GPU backend is registered for it (no such adapter, or no usable GPU at all).
pub(crate) fn device_not_found(device: Device) -> Error {
    let have: Vec<String> = folo_core::backend::available_devices().iter().map(|d| d.to_string()).collect();
    let why = if have.len() > 1 { "this machine has no such GPU" } else { "no usable GPU adapter was found" };
    Error::DeviceNotFound(format!(
        "device {device} is not available: {why}; FOLO can use {}",
        have.join(", ")
    ))
}

/// The training loop of docs/MODEL.md §8. See the module docs.
pub struct Trainer {
    cfg: TrainConfig,
    data: TrainData,
    device: Device,
    model: FoloModel,
    ema: Option<ModelEma<FoloModel>>,
    opt: Box<dyn Optimizer>,
    sched: LrSchedule,
    acc: GradAccumulator,
    loader: DataLoader,
    loss_cfg: LossConfig,
    resolved: ResolvedOptim,
    nb: usize,
    accumulate: usize,
    /// Completed epochs.
    epoch: u32,
    best: BestState,
    best_metrics: Option<folo_api::ValMetrics>,
    emitter: Emitter,
    opts_dir: Option<PathBuf>,
    stop: Arc<AtomicBool>,
    pending_logs: Vec<(LogLevel, String)>,
    /// [`schedule_warnings`], logged after the run's first log line.
    warnings: Vec<String>,
    /// The live state is exactly that of `epoch` completed epochs (false while an epoch runs, and for good once
    /// one was interrupted or failed): checkpoints may only be written then.
    at_boundary: bool,
    /// Non-finite batches in a row, across epoch boundaries (§8.3).
    consecutive_bad: usize,
    /// Unknown `folo.*` metadata of the checkpoint a run resumed from, written back into its checkpoints
    /// (FORMAT.md: tools that rewrite a file preserve them).
    extra_meta: BTreeMap<String, String>,
    /// The last checkpoint this trainer wrote or resumed from, and the epochs it holds (named when a lost GPU
    /// device fails the run).
    last_saved: Option<(PathBuf, u32)>,
}

impl Trainer {
    /// A new run of `cfg` on `data`.
    pub fn new(cfg: TrainConfig, data: TrainData, opts: TrainOptions) -> Result<Self> {
        let weights = opts.weights.clone();
        let mut t = Self::build(cfg, data, opts)?;
        t.warnings = schedule_warnings(&t.cfg, &t.sched, t.data.train.len(), weights.is_none());
        if let Some(w) = weights {
            let file = ModelFile::open(&w)?;
            let report = t.model.transfer_from(&file)?;
            if let Some(ema) = &mut t.ema {
                ema.reset(&t.model)?;
            }
            t.pending_logs.push((
                LogLevel::Info,
                format!(
                    "transferred {} of {} tensors from {} ({} skipped, {} missing)",
                    report.loaded.len(),
                    t.model.named_tensors().len(),
                    folo_api::json::path(&w),
                    report.skipped.len(),
                    report.missing.len()
                ),
            ));
        }
        Ok(t)
    }

    /// Continue the run a checkpoint (`last.folo` or `best.folo`) belongs to, with its stored config
    /// (`folo.train`). `data` must be the same dataset (same class list, same number of training images).
    pub fn resume(
        checkpoint: impl AsRef<Path>,
        data: TrainData,
        opts: TrainOptions,
        overrides: ResumeOverrides,
    ) -> Result<Self> {
        let path = checkpoint.as_ref();
        let file = ModelFile::open(path)?;
        let c = file.checkpoint().ok_or_else(|| {
            Error::Config(format!(
                "{} is a weights file, not a training checkpoint",
                folo_api::json::path(path)
            ))
        })?;
        let mut cfg: TrainConfig = serde_json::from_value(c.train.clone()).map_err(|e| {
            Error::Config(format!("{}: folo.train is not a TrainConfig: {e}", folo_api::json::path(path)))
        })?;
        if let Some(w) = overrides.workers {
            cfg.workers = w;
        }
        if let Some(d) = overrides.device {
            cfg.device = d;
        }
        cfg.augment.close_mosaic_fixed = Some(resumed_close_mosaic(&c.train, &cfg));
        if let Some(n) = c.train.get(TRAIN_IMAGES_KEY).and_then(|v| v.as_u64())
            && n != data.train.len() as u64
        {
            return Err(Error::Dataset(format!(
                "the checkpoint was trained on {n} training images, the dataset has {}; resume on the dataset the \
                 run started with",
                data.train.len()
            )));
        }
        if file.metadata().classes != data.classes {
            return Err(Error::Dataset(format!(
                "the checkpoint was trained on classes {:?}, the dataset has {:?}",
                file.metadata().classes,
                data.classes
            )));
        }
        let mut t = Self::build(cfg, data, opts)?;
        let ema = t.ema.as_mut().map(|e| e as &mut dyn EmaState);
        let r = file.resume(&t.model, t.opt.as_mut(), ema, Some(t.data.classes.clone()))?;
        let (want, got) = (t.sched.contract(), r.schedule.contract());
        if (want.epochs, want.steps_per_epoch, want.kind) != (got.epochs, got.steps_per_epoch, got.kind) {
            return Err(Error::Dataset(format!(
                "the checkpoint's schedule ({} epochs of {} steps) does not match this dataset ({} of {}); resume \
                 on the dataset the run started with",
                got.epochs, got.steps_per_epoch, want.epochs, want.steps_per_epoch
            )));
        }
        t.sched = r.schedule;
        // Whether the run started from weights is not recorded, so a resume repeats only the warmup warning.
        t.warnings = schedule_warnings(&t.cfg, &t.sched, t.data.train.len(), false);
        t.epoch = u32::try_from(r.epoch).map_err(|_| Error::Config("folo.epoch out of range".into()))?;
        t.best = r.best;
        t.extra_meta = file.metadata().extra.clone();
        t.last_saved = Some((path.to_path_buf(), t.epoch));
        t.consecutive_bad = match file.metadata().extra.get(NONFINITE_STREAK_KEY) {
            Some(v) => v
                .parse()
                .map_err(|_| Error::Config(format!("{NONFINITE_STREAK_KEY} `{v}` is not a count")))?,
            None => 0,
        };
        t.best_metrics = file.metadata().metrics.map(|m| folo_api::ValMetrics {
            map50: m.map50 as f32,
            map50_95: m.map50_95 as f32,
            precision: m.precision as f32,
            recall: m.recall as f32,
            per_class: Vec::new(),
        });
        t.pending_logs.push((
            LogLevel::Info,
            format!(
                "resumed from {} after epoch {} (optimizer step {})",
                folo_api::json::path(path),
                t.epoch,
                t.sched.step()
            ),
        ));
        Ok(t)
    }

    fn build(cfg: TrainConfig, data: TrainData, opts: TrainOptions) -> Result<Self> {
        cfg.validate()?;
        let n_train = data.train.len();
        let nc = data.classes.len();
        if nc == 0 {
            return Err(Error::Dataset("the dataset has no classes".into()));
        }
        if n_train == 0 {
            return Err(Error::Dataset("the dataset has no training images".into()));
        }
        if data.train.class_names().len() > nc {
            return Err(Error::Dataset(format!(
                "the training split names {} classes but the run has {nc}",
                data.train.class_names().len()
            )));
        }
        let mut pending_logs = Vec::new();
        let (close, close_eff) = (cfg.augment.close_mosaic, cfg.augment.close_mosaic_epochs(cfg.epochs));
        if cfg.augment.mosaic > 0.0 && cfg.augment.close_mosaic_fixed.is_none() && close > close_eff {
            pending_logs.push((
                LogLevel::Info,
                format!(
                    "close_mosaic {close} is more than half of the {} epochs: mosaic closes for the last {close_eff} \
                     instead",
                    cfg.epochs
                ),
            ));
        }
        let mut device = resolve_device(&cfg.device)?;
        if !device.is_cpu() && cfg.device.trim().eq_ignore_ascii_case("auto") {
            // `auto` must never pick a device the model cannot be placed on.
            if let Err(e) = can_place(device) {
                pending_logs.push((LogLevel::Warn, format!("{e}; training on the CPU")));
                device = Device::Cpu;
            }
        }

        let mconf = ModelConfig::new(cfg.scale()?, nc)?;
        let mut init = Rng::seed(cfg.seed);
        let mut model = FoloModel::with_rng(mconf.clone(), &mut init)?;
        model.set_classes(data.classes.clone())?;
        model.set_imgsz(cfg.imgsz);
        let model = place_model(model, device)?;
        model.train();
        let ema = if cfg.ema {
            let mut shadow = FoloModel::new(mconf)?;
            shadow.set_classes(data.classes.clone())?;
            shadow.set_imgsz(cfg.imgsz);
            Some(ModelEma::new(place_model(shadow, device)?, &model)?)
        } else {
            None
        };

        let accumulate = cfg.accumulate();
        let resolved = cfg.resolve_optimizer(n_train, nc);
        let wd = effective_weight_decay(cfg.weight_decay, cfg.batch, accumulate, cfg.nbs);
        let groups = StandardGroups::from_module(&model).into_groups(resolved.lr0, resolved.momentum, wd);
        let opt: Box<dyn Optimizer> = match resolved.kind {
            OptimKind::Sgd => Box::new(Sgd::new(groups)?),
            OptimKind::AdamW => Box::new(AdamW::new(groups)?),
        };
        let nb = n_train.div_ceil(cfg.batch);
        let spe = nb.div_ceil(accumulate);
        let sched = LrSchedule::new(
            if cfg.cos_lr { DecayKind::Cosine } else { DecayKind::Linear },
            cfg.epochs as u64,
            spe as u64,
            cfg.lrf,
            cfg.warmup_epochs,
            resolved.warmup_bias_lr,
            cfg.warmup_momentum,
        )?;
        let loader = DataLoader::new(
            data.train.clone(),
            LoaderConfig {
                batch_size: cfg.batch,
                imgsz: cfg.imgsz,
                workers: cfg.workers,
                shuffle: true,
                drop_last: false,
                seed: cfg.seed,
                augment: Some(cfg.augment),
                epochs: cfg.epochs,
                prefetch_batches: 2,
                u8_images: folo_gpu::context_for(device).is_some(),
            },
        )?;
        let acc = GradAccumulator::new(&model.parameters());
        let loss_cfg = LossConfig { reg_max: model.head.reg_max, ..LossConfig::default() };

        let file = match &opts.run_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir.join("weights"))
                    .map_err(|e| Error::io(dir.join("weights"), e))?;
                let p = dir.join("metrics.jsonl");
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .map_err(|e| Error::io(&p, e))?;
                Some(std::io::BufWriter::new(f))
            }
            None => None,
        };
        let emitter = Emitter::new(opts.events, file, opts.step_interval, (cfg.epochs, nb as u64));
        Ok(Trainer {
            cfg,
            data,
            device,
            model,
            ema,
            opt,
            sched,
            acc,
            loader,
            loss_cfg,
            resolved,
            nb,
            accumulate,
            epoch: 0,
            best: BestState::default(),
            best_metrics: None,
            emitter,
            opts_dir: opts.run_dir,
            stop: opts.stop,
            pending_logs,
            warnings: Vec::new(),
            at_boundary: true,
            consecutive_bad: 0,
            extra_meta: BTreeMap::new(),
            last_saved: None,
        })
    }

    /// The resolved configuration (`folo.train`).
    pub fn config(&self) -> &TrainConfig {
        &self.cfg
    }

    /// The device the run trains on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The live training model.
    pub fn model(&self) -> &FoloModel {
        &self.model
    }

    /// The EMA model (eval mode), if the run keeps one.
    pub fn ema_model(&self) -> Option<&FoloModel> {
        self.ema.as_ref().map(|e| e.module())
    }

    /// The optimizer the config resolved to.
    pub fn optimizer(&self) -> ResolvedOptim {
        self.resolved
    }

    /// Training batches per epoch, `nb` (docs/API.md `steps_per_epoch`).
    pub fn batches_per_epoch(&self) -> usize {
        self.nb
    }

    /// Batches per accumulation window.
    pub fn accumulate(&self) -> usize {
        self.accumulate
    }

    /// The LR schedule (its `step()` is the optimizer iteration `t`).
    pub fn schedule(&self) -> &LrSchedule {
        &self.sched
    }

    /// Epochs completed so far.
    pub fn epochs_completed(&self) -> u32 {
        self.epoch
    }

    /// The early-stopping state.
    pub fn best(&self) -> BestState {
        self.best
    }

    /// Non-finite batches in a row so far (they fail the run at [`MAX_NONFINITE_BATCHES`]).
    pub fn nonfinite_streak(&self) -> usize {
        self.consecutive_bad
    }

    /// The [`schedule_warnings`] of this run.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The flag [`TrainOptions::stop`] that stops this run.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// The `config.json` of the run: the resolved config plus the derived numbers.
    pub fn run_config(&self) -> serde_json::Value {
        let c = self.sched.contract();
        serde_json::json!({
            "train": self.cfg,
            "arch": self.model.arch(),
            "classes": self.data.classes,
            "params": self.model.num_parameters(),
            "device": self.device.to_string(),
            "train_images": self.data.train.len(),
            "val_images": self.data.val.len(),
            "optimizer": {
                "type": self.resolved.kind.as_str(),
                "lr0": self.resolved.lr0,
                "momentum": self.resolved.momentum,
                "warmup_bias_lr": self.resolved.warmup_bias_lr,
                "weight_decay": effective_weight_decay(self.cfg.weight_decay, self.cfg.batch, self.accumulate, self.cfg.nbs),
            },
            "batches_per_epoch": self.nb,
            "accumulate": self.accumulate,
            "steps_per_epoch": c.steps_per_epoch,
            "total_iters": self.sched.total_iters(),
            "warmup_iters": c.warmup_iters,
            "close_mosaic_epochs": self.cfg.augment.close_mosaic_epochs(self.cfg.epochs),
            "warnings": self.warnings,
        })
    }

    /// Run to the end (all epochs, early stopping, or the stop flag). Emits `status` `running` first and
    /// `completed` / `stopped` / `failed` last. On error the status `failed` is emitted and the error returned;
    /// the last checkpoint stays valid. A run stopped, or failed, inside an epoch continues only through
    /// [`Trainer::resume`] (its in-memory state is past the checkpoint), so calling `run` again on it is an error.
    /// A resumed run appends to `metrics.jsonl`, so the steps of an interrupted epoch appear there twice.
    pub fn run(&mut self) -> Result<TrainSummary> {
        if !self.at_boundary {
            return Err(Error::Config(format!(
                "this run stopped inside epoch {}; continue it with Trainer::resume from its last checkpoint",
                self.epoch + 1
            )));
        }
        self.emitter.status(RunStatus::Running, None);
        let r = self.run_inner().map_err(|e| self.explain_device_loss(e));
        match &r {
            Ok(s) => self.emitter.status(s.status, None),
            Err(e) => {
                // The status carries the (capped) message; the log line only names the failure, so a driver
                // report is not repeated in every record.
                let summary = e.summary();
                self.emitter.log(
                    LogLevel::Error,
                    format!("run failed ({}): {}", e.code(), folo_api::codes::cap(&summary, 120)),
                );
                self.emitter.status(RunStatus::Failed, Some(summary));
            }
        }
        r
    }

    /// A lost GPU device (driver reset, TDR) fails the run with a message that says so and names the checkpoint to
    /// resume from. Checkpoints are written only at epoch boundaries, so the last one written stays intact.
    fn explain_device_loss(&self, e: Error) -> Error {
        if !device_lost(&e) {
            return e;
        }
        Error::Failed(device_loss_message(&e, self.last_saved.as_ref().map(|(p, n)| (p.as_path(), *n))))
    }

    fn run_inner(&mut self) -> Result<TrainSummary> {
        if let Some(dir) = &self.opts_dir {
            let p = dir.join("config.json");
            let text = serde_json::to_string_pretty(&self.run_config()).expect("config serialises");
            std::fs::write(&p, text).map_err(|e| Error::io(&p, e))?;
        }
        for (level, msg) in std::mem::take(&mut self.pending_logs) {
            self.emitter.log(level, msg);
        }
        let c = self.sched.contract();
        self.emitter.log(
            LogLevel::Info,
            format!(
                "{} ({} params, {} classes) on {}: {} train / {} val images at {}px, batch {} × accumulate {}, {} {} \
                 lr0 {} momentum {}, {} batches = {} optimizer steps per epoch, warmup {} steps",
                self.model.arch(),
                self.model.num_parameters(),
                self.data.classes.len(),
                self.device,
                self.data.train.len(),
                self.data.val.len(),
                self.cfg.imgsz,
                self.cfg.batch,
                self.accumulate,
                if self.cfg.optimizer == OptimizerChoice::Auto { "auto →" } else { "optimizer" },
                self.resolved.kind.as_str(),
                self.resolved.lr0,
                self.resolved.momentum,
                self.nb,
                c.steps_per_epoch,
                c.warmup_iters
            ),
        );
        for w in &self.warnings {
            self.emitter.log(LogLevel::Warn, w.clone());
        }
        if self.data.val.is_empty() {
            self.emitter.log(
                LogLevel::Warn,
                "the dataset has no validation images; validating on the training split",
            );
        }

        let mut summary = TrainSummary {
            status: RunStatus::Completed,
            epochs_completed: self.epoch,
            early_stopped: false,
            best: None,
            best_metrics: None,
            epochs: Vec::new(),
            last: None,
            best_weights: None,
        };
        let run_start = Instant::now();
        let mut clock = Clock::default();
        let epochs = self.cfg.epochs;
        let mut stopped = false;
        // A resumed run whose last completed epoch already met the patience rule stops without training.
        if self.epoch > 0
            && self.best.fitness.is_some()
            && (self.epoch as i64 - 1) - self.best.epoch >= self.cfg.patience as i64
            && self.epoch < epochs
        {
            self.emitter.log(
                LogLevel::Info,
                format!(
                    "early stopping: the run already stopped after epoch {} (best epoch {})",
                    self.epoch,
                    self.best.epoch + 1
                ),
            );
            summary.early_stopped = true;
        }
        while self.epoch < epochs && !summary.early_stopped {
            if self.stopping() {
                stopped = true;
                break;
            }
            let e = self.epoch;
            let a = &self.cfg.augment;
            if e > 0 && a.mosaic > 0.0 && a.mosaic_active(e - 1, epochs) && !a.mosaic_active(e, epochs) {
                self.emitter.log(
                    LogLevel::Info,
                    format!("closing mosaic for the last {} epochs", a.close_mosaic_epochs(epochs)),
                );
            }
            let t0 = Instant::now();
            self.at_boundary = false;
            let Some(train_loss) = self.train_epoch(e, &mut clock)? else {
                stopped = true;
                break;
            };
            let val = self.validate_epoch()?;
            clock.val_time += val.seconds;
            clock.val_count += 1;
            let metrics = val_metrics(&val.report);
            let fitness = val.report.map50_95 as f64;
            let (improved, stop) = update_best(&mut self.best, e, fitness, self.cfg.patience);
            if improved {
                self.best_metrics = Some(metrics.clone());
            }
            self.epoch = e + 1;
            self.at_boundary = true;
            self.write_epoch_files(e, improved, &val.samples, &mut summary)?;
            let ep = folo_api::Epoch {
                epoch: e + 1,
                train_loss,
                val: metrics,
                duration_s: t0.elapsed().as_secs_f64(),
            };
            summary.epochs.push(ep.clone());
            self.emitter.emit(RunEvent::Epoch(ep));
            if stop {
                self.emitter.log(
                    LogLevel::Info,
                    format!(
                        "early stopping: no fitness gain for {} epochs (best epoch {})",
                        self.cfg.patience,
                        self.best.epoch + 1
                    ),
                );
                summary.early_stopped = true;
                break;
            }
        }
        if stopped {
            self.emitter.log(LogLevel::Info, format!("stopped after {} completed epochs", self.epoch));
            summary.status = RunStatus::Stopped;
        } else {
            self.emitter.log(
                LogLevel::Info,
                format!("finished {} epochs in {:.1} s", self.epoch, run_start.elapsed().as_secs_f64()),
            );
        }
        summary.epochs_completed = self.epoch;
        summary.best = self.best.fitness.map(|f| (self.best.epoch as u32 + 1, f));
        summary.best_metrics = self.best_metrics.clone();
        if let Some(dir) = &self.opts_dir {
            let (last, best) = (dir.join("weights/last.folo"), dir.join("weights/best.folo"));
            summary.last = last.is_file().then_some(last);
            summary.best_weights = best.is_file().then_some(best);
        }
        Ok(summary)
    }

    /// One epoch of §8.3. Returns the mean reported loss over its finite batches, or `None` if the stop flag
    /// interrupted it.
    fn train_epoch(&mut self, e: u32, clock: &mut Clock) -> Result<Option<folo_api::Loss>> {
        self.model.train();
        self.acc.zero();
        let s = self.cfg.imgsz;
        let (nb, acc_n) = (self.nb, self.accumulate);
        let decay = self.opt.groups().iter().position(|g| g.name == "decay").unwrap_or(0);
        let mut sum = [0f64; 3];
        let (mut finite_batches, mut window_finite) = (0usize, false);
        let (epoch_start, mut epoch_images) = (Instant::now(), 0usize);
        // On a folo-gpu device the loader delivers u8 pixels and a feeder thread stages each batch while the
        // previous step runs; the step then only encodes the copy and the conversion.
        let ctx = folo_gpu::context_for(self.device);
        let iter = self.loader.epoch(e);
        let mut source: Box<dyn Iterator<Item = Result<(Batch, Option<folo_gpu::Staged>)>>> = match &ctx {
            Some(c) => Box::new(feed::Feeder::new(iter, c.clone())?.map(|r| r.map(|(b, s)| (b, Some(s))))),
            None => Box::new(iter.map(|r| r.map(|b| (b, None)).map_err(Error::from))),
        };
        let mut k = 0usize;
        for item in source.by_ref() {
            let batch_start = Instant::now();
            let (mut batch, staged) = item?;
            let n = batch.len();
            let targets = targets_from_normalized(&batch.targets, n, s, s)?;
            let x = match (&ctx, staged) {
                (Some(c), Some(st)) => feed::staged_tensor(c, self.device, &st, batch.shape)?,
                _ => batch_tensor(&mut batch, self.device, folo_core::DType::F32)?,
            };
            let lr = self.sched.lr_at(self.sched.step(), &self.opt.groups()[decay]);

            let snapshot = RunningStats::capture(&self.model)?;
            let out = self.model.forward(&x)?;
            drop(x);
            let loss = detection_loss(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &self.loss_cfg)?;
            let finite = loss.is_finite() && RunningStats::all_finite(&self.model)?;
            let items = loss.items;
            if finite {
                let grads = loss.total.backward()?;
                drop(loss);
                drop(out);
                self.acc.add(&grads)?;
                window_finite = true;
                self.consecutive_bad = 0;
                finite_batches += 1;
                sum[0] += items.box_ as f64;
                sum[1] += items.cls as f64;
                sum[2] += items.dfl as f64;
            } else {
                drop(loss);
                drop(out);
                snapshot.restore()?;
                self.consecutive_bad += 1;
                let consecutive_bad = self.consecutive_bad;
                self.emitter.log(
                    LogLevel::Warn,
                    format!(
                        "epoch {} batch {}: non-finite loss {:?}; skipped ({consecutive_bad} in a row)",
                        e + 1,
                        k + 1,
                        items
                    ),
                );
                if consecutive_bad >= MAX_NONFINITE_BATCHES {
                    return Err(Error::Failed(format!(
                        "{MAX_NONFINITE_BATCHES} consecutive non-finite batches (epoch {}, batch {}); lower lr0 or \
                         check the data",
                        e + 1,
                        k + 1
                    )));
                }
            }
            if (k + 1).is_multiple_of(acc_n) || k + 1 == nb {
                if window_finite {
                    self.optimizer_step()?;
                }
                self.acc.zero();
                window_finite = false;
            }

            epoch_images += n;
            clock.batches += 1;
            clock.train_time += batch_start.elapsed().as_secs_f64();
            if finite {
                let elapsed = epoch_start.elapsed().as_secs_f64().max(1e-9);
                let step = folo_api::Step {
                    epoch: e + 1,
                    step: e as u64 * nb as u64 + k as u64 + 1,
                    lr: lr as f32,
                    loss: loss_json(&items),
                    imgs_per_s: (epoch_images as f64 / elapsed) as f32,
                    eta_s: self.eta(e, k, clock),
                    steps_per_epoch: nb as u64,
                    epochs: self.cfg.epochs,
                };
                self.emitter.step(RunEvent::Step(step), k + 1 == nb);
            }
            k += 1;
            if self.stopping() {
                return Ok(None);
            }
        }
        drop(source);
        if k != nb {
            return Err(Error::Failed(format!("epoch {} delivered {k} of {nb} batches", e + 1)));
        }
        if finite_batches == 0 {
            return Err(Error::Failed(format!("epoch {} had no finite batch", e + 1)));
        }
        let f = finite_batches as f64;
        let (b, c, d) = ((sum[0] / f) as f32, (sum[1] / f) as f32, (sum[2] / f) as f32);
        Ok(Some(folo_api::Loss { box_: b, cls: c, dfl: d, total: b + c + d }))
    }

    /// §8.3's end-of-window step: schedule, clip, optimizer, EMA, `t += 1`.
    fn optimizer_step(&mut self) -> Result<()> {
        self.sched.apply(self.opt.as_mut());
        if self.cfg.clip_grad_norm > 0.0 {
            self.acc.clip_grad_norm(self.cfg.clip_grad_norm)?;
        }
        self.opt.step(&self.acc)?;
        if let Some(ema) = &mut self.ema {
            ema.update(&self.model)?;
        }
        self.sched.advance();
        Ok(())
    }

    /// Remaining seconds: batches left at the run's mean batch time, plus the validations left at their mean.
    fn eta(&self, e: u32, k: usize, clock: &Clock) -> f64 {
        let per_batch = clock.train_time / clock.batches.max(1) as f64;
        let left_epochs = (self.cfg.epochs - e - 1) as f64;
        let left_batches = (self.nb - k - 1) as f64 + left_epochs * self.nb as f64;
        let per_val = if clock.val_count > 0 {
            clock.val_time / clock.val_count as f64
        } else {
            0.1 * per_batch * self.nb as f64
        };
        left_batches * per_batch + (left_epochs + 1.0) * per_val
    }

    /// Validate the EMA model (or the live model in eval mode) on the validation split.
    fn validate_epoch(&mut self) -> Result<ValResult> {
        let ds = if self.data.val.is_empty() { self.data.train.clone() } else { self.data.val.clone() };
        let vc =
            ValConfig { workers: self.cfg.workers, ..ValConfig::new(self.cfg.imgsz, self.cfg.val_batch()) };
        match &self.ema {
            Some(ema) => validate(ema.module(), ds, &vc),
            None => {
                self.model.eval();
                let r = validate(&self.model, ds, &vc);
                self.model.train();
                r
            }
        }
    }

    /// `weights/last.folo` (and `best.folo` when `improved`) and the epoch's samples.
    fn write_epoch_files(
        &mut self,
        e: u32,
        improved: bool,
        samples: &[ValSample],
        summary: &mut TrainSummary,
    ) -> Result<()> {
        let Some(dir) = &self.opts_dir else { return Ok(()) };
        let weights = dir.join("weights");
        let last = weights.join("last.folo");
        self.save_checkpoint(&last)?;
        self.last_saved = Some((last.clone(), self.epoch));
        if improved {
            let best = weights.join("best.folo");
            let (tmp, mut f) = folo_nn::io::safetensors::create_temp_beside(&best)?;
            // The file is closed before the rename (on wasm `File` has nothing to close, hence the allow).
            #[cfg_attr(target_arch = "wasm32", allow(clippy::drop_non_drop))]
            let copied = std::fs::File::open(&last)
                .and_then(|mut src| std::io::copy(&mut src, &mut f))
                .and_then(|_| f.sync_all())
                .and_then(|_| {
                    drop(f);
                    std::fs::rename(&tmp, &best)
                });
            if let Err(err) = copied {
                let _ = std::fs::remove_file(&tmp);
                return Err(Error::io(&best, err));
            }
            summary.best_weights = Some(best);
        }
        summary.last = Some(last);
        let sdir = dir.join("samples");
        std::fs::create_dir_all(&sdir).map_err(|err| Error::io(&sdir, err))?;
        let p = sdir.join(format!("epoch_{:03}.json", e + 1));
        let names = &self.data.classes;
        let body = serde_json::json!({
            "epoch": e + 1,
            "classes": names,
            "samples": samples,
        });
        std::fs::write(&p, serde_json::to_string(&body).expect("samples serialise"))
            .map_err(|err| Error::io(&p, err))
    }

    /// Write the complete training state (FORMAT.md checkpoint) to `path`, atomically.
    ///
    /// Only possible at an epoch boundary (FORMAT.md v1 stores no loader position or pending gradients): before
    /// the first epoch, between epochs, or after `run` returned at one. A trainer stopped or failed inside an
    /// epoch refuses.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.at_boundary {
            return Err(Error::Config(format!(
                "a checkpoint can only be written at an epoch boundary; this run stopped inside epoch {}",
                self.epoch + 1
            )));
        }
        let mut meta = self.model.metadata()?;
        meta.metrics = self.best_metrics.as_ref().map(|m| Metrics {
            map50: m.map50 as f64,
            map50_95: m.map50_95 as f64,
            precision: m.precision as f64,
            recall: m.recall as f64,
        });
        meta.created = Some(rfc3339_now());
        meta.extra = self.extra_meta.clone();
        meta.extra.insert(NONFINITE_STREAK_KEY.into(), self.consecutive_bad.to_string());
        let mut streams = BTreeMap::new();
        // Every random draw after initialisation is derived from (seed, epoch, ...), so the seed is the whole
        // RNG state; the init stream is recorded for completeness.
        streams.insert("init".to_string(), Rng::seed(self.cfg.seed));
        let rng = RngState { seed: self.cfg.seed, streams };
        let mut train = serde_json::to_value(&self.cfg).expect("config serialises");
        train[TRAIN_IMAGES_KEY] = serde_json::json!(self.data.train.len());
        train[CLOSE_MOSAIC_KEY] = serde_json::json!(self.cfg.augment.close_mosaic_epochs(self.cfg.epochs));
        let parts = CheckpointParts {
            model: &self.model,
            ema: self.ema.as_ref().map(|e| e as &dyn EmaState),
            optimizer: self.opt.as_ref(),
            schedule: &self.sched,
            rng: &rng,
            epoch: self.epoch as u64,
            best: self.best,
            train_config: &train,
        };
        Ok(save_checkpoint(path, &meta, &parts)?)
    }
}

/// Whether `e` is folo-gpu reporting a lost device, from the trainer's own tensor work, a layer, or the batch
/// feeder.
fn device_lost(e: &Error) -> bool {
    match e {
        Error::Core(c) | Error::Nn(folo_nn::Error::Core(c)) => folo_gpu::is_device_lost(c),
        _ => false,
    }
}

/// The failure message of a run whose GPU was lost: the cause, and the checkpoint to resume from (the last one
/// written or resumed from, with the epochs it holds) or that there is none.
fn device_loss_message(e: &Error, last: Option<(&Path, u32)>) -> String {
    let resume = match last {
        Some((p, n)) => format!(
            "{} holds the state after epoch {n}: resume with `folo train --resume {}`",
            folo_api::json::path(p),
            folo_api::json::path(p)
        ),
        None => "no checkpoint was written yet: start the run again".into(),
    };
    format!("the GPU device was lost ({e}); {resume}")
}

fn loss_json(i: &LossItems) -> folo_api::Loss {
    folo_api::Loss { box_: i.box_, cls: i.cls, dfl: i.dfl, total: i.total() }
}

/// Wall-clock accounting for the ETA.
#[derive(Default)]
struct Clock {
    batches: usize,
    train_time: f64,
    val_count: usize,
    val_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resumed run keeps the closing epochs its checkpoint records; one from before the cap keeps the old rule.
    #[test]
    fn resumes_keep_their_close_mosaic() {
        let cfg = TrainConfig { epochs: 8, ..TrainConfig::default() };
        let old = serde_json::to_value(&cfg).unwrap();
        assert_eq!(resumed_close_mosaic(&old, &cfg), 8, "close_mosaic 10 of 8 epochs: no mosaic at all");
        let mut new = old.clone();
        new[CLOSE_MOSAIC_KEY] = serde_json::json!(cfg.augment.close_mosaic_epochs(cfg.epochs));
        assert_eq!(resumed_close_mosaic(&new, &cfg), 4);
        // The key is not a TrainConfig field: folo.train with it still parses.
        assert!(serde_json::from_value::<TrainConfig>(new).is_ok());
    }

    /// A lost GPU device is recognised wherever it surfaces (tensor work, a layer, the batch feeder) and the
    /// message names the checkpoint that was actually written; other errors are left alone.
    #[test]
    fn device_loss_is_explained() {
        let lost = || folo_core::Error::from(folo_gpu::GpuError::DeviceLost("Destroyed".into()));
        let other = folo_core::Error::from(folo_gpu::GpuError::Wgpu("validation".into()));
        assert!(device_lost(&Error::Core(lost())));
        assert!(device_lost(&Error::Nn(folo_nn::Error::Core(lost()))));
        assert!(!device_lost(&Error::Core(other)));
        assert!(!device_lost(&Error::Failed("GPU device lost".into())));
        let p = Path::new("runs/r6/weights/last.folo");
        let m = device_loss_message(&Error::Core(lost()), Some((p, 266)));
        assert!(m.contains("GPU device lost") && m.contains("after epoch 266"), "{m}");
        assert!(m.contains(&format!("--resume {}", p.display())), "{m}");
        let m = device_loss_message(&Error::Core(lost()), None);
        assert!(m.contains("no checkpoint was written yet"), "{m}");
    }

    /// MODEL.md §10.4: patience 2 with fitness [0.1, 0.3, 0.2, 0.2] stops after epoch 3 (best epoch 1), and a
    /// resume from the epoch-2 state (the `folo.best` a checkpoint persists) stops at the same epoch.
    #[test]
    fn early_stopping() {
        let fit = [0.1, 0.3, 0.2, 0.2, 0.9];
        let run = |best: &mut BestState, from: u32| {
            for e in from..fit.len() as u32 {
                if update_best(best, e, fit[e as usize], 2).1 {
                    return e;
                }
            }
            u32::MAX
        };
        let mut best = BestState::default();
        assert_eq!(run(&mut best, 0), 3);
        assert_eq!(best, BestState { epoch: 1, fitness: Some(0.3) });
        let mut at_epoch_2 = BestState::default();
        for e in 0..3 {
            update_best(&mut at_epoch_2, e, fit[e as usize], 2);
        }
        assert_eq!(run(&mut at_epoch_2, 3), 3);
        // An equal fitness is not a gain.
        let mut b = BestState { epoch: 0, fitness: Some(0.5) };
        assert_eq!(update_best(&mut b, 1, 0.5, 10), (false, false));
    }
}
