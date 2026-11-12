//! The trainer: docs/MODEL.md §8 end to end.
//!
//! [`Trainer`] wires folo-data's [`DataLoader`] to the model, the loss, folo-nn's optimizers, schedule, EMA and
//! checkpoints:
//!
//! - **Batches** (§8.3): the loader shuffles with `(seed, epoch)` and augments each sample from
//!   `(seed, epoch, position)`, so a resumed epoch reproduces its batches exactly. `accumulate =
//!   max(round(nbs / batch), 1)` batches form a window; windows never cross an epoch boundary.
//! - **Step** (§8.3–8.5): train-mode forward, [`detection_loss`] (for a `head = "o2o"` model the [`dual_loss`] of
//!   both branches, §6.6), and (only if the loss and every BatchNorm
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
//! - **Mixed precision** (`amp`, §8.8): the forward runs under [`folo_core::autocast`] (f16 activations, convs
//!   and matrix products on the GPU's tensor cores), the loss in f32 on f32 copies of the head outputs, backward on
//!   the loss times a dynamic loss scale ([`GradScaler`]); at the end of a window the gradients are unscaled and
//!   clipped and, if any is not finite, the step is skipped (no `t` advance, the scale backs off). Master weights,
//!   optimizer state, BatchNorm statistics and the EMA stay f32. Needs a GPU context with tensor cores
//!   ([`amp_support`]).
//! - **Resume** ([`Trainer::resume`]) restores the model, EMA, optimizer, schedule, loss scaler and early-stopping
//!   state from an epoch-boundary checkpoint; the following steps are bit-identical to an uninterrupted run.
//!
//! Run directory layout (when [`TrainOptions::run_dir`] is set): `config.json` (the resolved config and the
//! derived schedule numbers), `metrics.jsonl`, `weights/{last,best}.folo` and `samples/epoch_NNN.json` (the
//! predictions on the first 16 validation images, docs/API.md `/samples`).
//!
//! The device is a parameter: inputs, the loss and the optimizers follow the model's device, which
//! [`place_model`] sets (`Module::to_device`).
//!
//! [`detection_loss`]: crate::loss::detection_loss
//! [`dual_loss`]: crate::loss::dual_loss

mod checkpoint;
mod config;
mod device;
mod epoch;
mod events;
mod feed;
mod run;
mod setup;
#[cfg(test)]
mod tests;

pub use config::{
    MuSgdConfig, OptimizerChoice, ResolvedOptim, TrainConfig, default_workers, f32_decimal, resolve_device,
};
pub use device::{amp_available, amp_not_supported, amp_support, can_place, place_model};
pub use events::{EventSink, channel_sink, event_record, rfc3339_now};
pub use run::update_best;
pub use setup::{resumed_close_mosaic, schedule_warnings};

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{LogLevel, RunStatus};
use folo_core::Device;
use folo_data::{Batch, DataLoader, Dataset};
use folo_nn::io::BestState;
use folo_nn::optim::{GradAccumulator, Optimizer, effective_weight_decay};

/// MuSGD's LR factor for the head's class branches (docs/MODEL.md §8.2, the reference's `lr * 3`).
pub const MUSGD_CLS_LR_FACTOR: f64 = 3.0;
use folo_nn::{GradScaler, LrSchedule, ModelEma, Module};

use crate::error::Result;
use crate::loss::LossConfig;
use crate::model::FoloModel;
use device::device_not_found;
use events::Emitter;

/// Consecutive non-finite batches that fail the run (§8.3).
pub const MAX_NONFINITE_BATCHES: usize = 10;

/// The extra `folo.train` key holding the number of training images, checked on resume.
pub const TRAIN_IMAGES_KEY: &str = "train_images";

/// The extra `folo.train` key holding the epochs at the end of the run with mosaic off
/// ([`folo_data::AugmentConfig::close_mosaic_epochs`]), so a resume keeps them. A checkpoint without it predates
/// the cap of docs/MODEL.md §8.6 ([`resumed_close_mosaic`]).
pub const CLOSE_MOSAIC_KEY: &str = "close_mosaic_epochs";

/// Checkpoint metadata key (FORMAT.md: unknown `folo.*` keys are preserved) holding the run of non-finite batches
/// at the end of the checkpoint's epoch, so a streak crossing a resume still fails at [`MAX_NONFINITE_BATCHES`].
pub const NONFINITE_STREAK_KEY: &str = "folo.nonfinite_streak";

/// Runs from scratch with fewer optimizer steps than this get a warning ([`schedule_warnings`]).
pub const SHORT_RUN_STEPS: u64 = 200;

/// Checkpoint metadata key (docs/FORMAT.md) of a mixed-precision run's loss scaler: [`GradScaler::to_json`].
pub const AMP_KEY: &str = "folo.amp";

/// Consecutive optimizer steps skipped for non-finite gradients that fail a mixed-precision run (§8.8): by then the
/// loss scale has backed off by `2^50`, so the gradients are not finite at any scale.
pub const MAX_AMP_SKIPS: usize = 50;

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
    /// How the dataset is read ([`TrainConfig::data_format`]), kept in the resumed run's checkpoints.
    pub data_format: Option<folo_data::Format>,
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
    /// The loss scaler of a mixed-precision run (`amp`). Its skip streak (fails the run at [`MAX_AMP_SKIPS`]) is
    /// part of the checkpointed state.
    scaler: Option<GradScaler>,
    /// The next epoch's batches, started when the previous epoch took its last batch, so that the loader works
    /// while validation and the checkpoints run (the epoch it belongs to, and its source).
    prefetched: Option<(u32, BatchSource)>,
}

/// A started epoch's batches: the loader's, staged by the feeder thread on a folo-gpu device.
type BatchSource = Box<dyn Iterator<Item = Result<(Batch, Option<folo_gpu::Staged>)>> + Send>;

impl Trainer {
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

    /// The optimizer's parameter groups (names, parameters, base and current LR).
    pub fn param_groups(&self) -> &[folo_nn::optim::ParamGroup] {
        self.opt.groups()
    }

    /// docs/API.md's `Run.resolved`: the optimizer, effective LR and weight decay, accumulation and warmup this run
    /// trains with (also `resolved` in its `config.json`).
    pub fn resolved_train(&self) -> folo_api::ResolvedTrain {
        self.resolved.to_api(&self.cfg, self.accumulate, self.sched.contract().steps_per_epoch)
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

    /// The loss scaler of a mixed-precision run (`amp`): the current scale and the steps skipped so far.
    pub fn loss_scaler(&self) -> Option<&GradScaler> {
        self.scaler.as_ref()
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
            "resolved": self.resolved_train(),
        })
    }
}
