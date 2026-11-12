//! The run loop ([`Trainer::run`]): epochs, validation, early stopping, the summary, and the status and log
//! events around them.

use std::time::Instant;

use folo_api::{LogLevel, RunEvent, RunStatus};
use folo_nn::Module;
use folo_nn::io::BestState;

use super::device::{device_loss_message, device_lost};
use super::epoch::Clock;
use super::{OptimizerChoice, TrainSummary, Trainer};
use crate::error::{Error, Result};
use crate::val::{ValConfig, ValResult, val_metrics, validate};

/// Early stopping (§8.7): record `fitness` of 0-based epoch `e` in `best`. Returns `(improved, stop)`: whether it
/// is a new best, and whether the run stops after this epoch (`e − best.epoch >= patience`).
pub fn update_best(best: &mut BestState, e: u32, fitness: f64, patience: u32) -> (bool, bool) {
    let improved = best.fitness.is_none_or(|b| fitness > b);
    if improved {
        *best = BestState { epoch: e as i64, fitness: Some(fitness) };
    }
    (improved, e as i64 - best.epoch >= patience as i64)
}

impl Trainer {
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
        let r = self.run_inner();
        // A run that failed or stopped inside an epoch still delivers the previous epoch's event once its files are
        // written (a failed write fails the run).
        let written = self.settle_write();
        let r = match (r, written) {
            (Ok(s), Ok(())) => Ok(s),
            (Err(e), _) | (Ok(_), Err(e)) => Err(e),
        }
        .map_err(|e| self.explain_device_loss(e));
        // A next epoch started before the run ended (finished, stopped, stopped early or failed) is dropped, which
        // stops and joins its loader workers and feeder: nothing keeps decoding or staging once `run` returns.
        self.prefetched = None;
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
                self.emitter.failed(e.code(), summary);
            }
        }
        r
    }

    /// A lost GPU device (driver reset, TDR) fails the run with a message that says so and names the checkpoint to
    /// resume from. Checkpoints are written only at epoch boundaries (from a host snapshot, so a write in flight
    /// finishes), and the last one written stays intact.
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
                 lr0 {} momentum {}{}, {} batches = {} optimizer steps per epoch, warmup {} steps",
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
                if self.resolved.kind == folo_nn::optim::OptimKind::MuSgd {
                    format!(" muon {} sgd {}", self.resolved.musgd.muon, self.resolved.musgd.sgd)
                } else {
                    String::new()
                },
                self.nb,
                c.steps_per_epoch,
                c.warmup_iters
            ),
        );
        if self.model.has_o2o() {
            self.emitter.log(
                LogLevel::Info,
                "head o2o: dual assignment (one-to-many + one-to-one losses, weights 0.8 / 0.2 → 0.1 / 0.9 over the \
                 run); validation uses the one-to-one head without NMS"
                    .to_string(),
            );
        }
        for w in &self.warnings {
            self.emitter.log(LogLevel::Warn, w.clone());
        }
        if let Some(s) = &self.scaler {
            self.emitter.log(
                LogLevel::Info,
                format!(
                    "mixed precision: f16 activations, convolutions and matrix products on tensor cores, f32 master \
                     weights, loss and BatchNorm statistics; loss scale {} (×{} every {} finite steps, ×{} on overflow)",
                    s.scale(),
                    s.growth_factor(),
                    s.growth_interval(),
                    s.backoff_factor()
                ),
            );
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
            let batches_before = clock.wall[0] + clock.wall[1];
            self.at_boundary = false;
            let Some(train_loss) = self.train_epoch(e, &mut clock)? else {
                stopped = true;
                break;
            };
            let val = self.validate_epoch()?;
            let metrics = val_metrics(&val.report);
            let fitness = val.report.map50_95 as f64;
            let (improved, stop) = update_best(&mut self.best, e, fitness, self.cfg.patience);
            if improved {
                self.best_metrics = Some(metrics.clone());
            }
            self.epoch = e + 1;
            self.at_boundary = true;
            // The checkpoint's snapshot is taken here; its files are written while the next epoch trains, and the
            // epoch event is delivered once they are on disk (`train/writer.rs`). The epoch's time ends with the
            // snapshot.
            let files = self.epoch_files(e, improved, &val.samples)?;
            let duration_s = t0.elapsed().as_secs_f64();
            let batches = clock.wall[0] + clock.wall[1] - batches_before;
            clock.ended_epoch(val.seconds, (duration_s - batches - val.seconds).max(0.0));
            let ep = folo_api::Epoch { epoch: e + 1, train_loss, val: metrics, duration_s };
            summary.epochs.push(ep.clone());
            self.write_epoch_files(files, RunEvent::Epoch(ep))?;
            if let Some(mut hook) = self.after_epoch.take() {
                let written = self.settle_write();
                hook(&self.stop);
                self.after_epoch = Some(hook);
                written?;
            }
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
        self.settle_write()?;
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
}
