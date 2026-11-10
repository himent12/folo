//! The worker's training of one run: building its trainer and measuring its throughput.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use folo_api::{Epoch, LogLevel, PhaseKind, RunEvent, RunStatus, TrainConfig};
use folo_data::{Dataset as _, Split};
use folo_vision::train::ResumeOverrides;
use folo_vision::{TrainData, TrainOptions, Trainer};

use crate::engine;
use crate::throughput::{Throughput, TrainKey, TrainSample};

use super::{DatasetRef, EpochHold, Measure, RunEntry, data_format};

impl Measure {
    /// Epoch `e` (1-based) trains with mosaic.
    fn mosaic_in(&self, e: u32) -> bool {
        self.mosaic && e + self.close_mosaic_epochs <= self.epochs
    }

    pub(super) fn epoch(&mut self, e: &Epoch) {
        // Only the epoch's final step covers all of it (steps of non-finite batches are not sent).
        let step = self.last_step.take().filter(|s| {
            s.epoch == e.epoch && s.step == e.epoch as u64 * s.steps_per_epoch && s.imgs_per_s > 0.0
        });
        let Some((first, first_s)) = self.first else {
            self.first = Some((e.epoch, e.duration_s));
            return;
        };
        // `imgs_per_s` of the epoch's last step is its training rate: images over the time since the epoch began.
        // Without it the epoch is not a sample at all.
        let Some(s) = step else { return };
        let ips = s.imgs_per_s as f64;
        let val_s = e.duration_s - self.train_images as f64 / ips;
        let val = (val_s > 0.01).then(|| self.val_images as f64 / val_s);
        let mosaic = self.mosaic_in(e.epoch);
        let startup =
            (first + 1 == e.epoch && mosaic == self.mosaic_in(first)).then_some(first_s - e.duration_s);
        let peak = folo_gpu::context_for(self.device).map(|c| c.pool_stats().peak_live_bytes);
        self.throughput.record_train(TrainSample {
            key: self.key.clone(),
            phase: if mosaic { PhaseKind::Mosaic } else { PhaseKind::CloseMosaic },
            train_imgs_per_s: Some(ips),
            val_imgs_per_s: val,
            startup_s: startup,
            peak_vram_bytes: peak,
            probe: false,
        });
    }
}

/// Trains one run on the worker thread.
pub(super) fn execute(e: &Arc<RunEntry>, throughput: &Arc<Throughput>, hold: &Arc<EpochHold>) {
    let (config, dataset, weights) = {
        let st = e.state.lock().unwrap();
        (st.file.run.config.clone(), st.file.dataset.clone(), st.file.weights.clone())
    };
    let trainer = build(e, &config, &dataset, weights, hold);
    let mut trainer = match trainer {
        Ok(t) => t,
        Err((code, msg)) => {
            // A trainer that failed after placing its model (weights that do not load, a dataset error after the
            // model was built) left its buffers in the pool too.
            if let Ok(device) = engine::resolve(&config.device) {
                engine::release_gpu_memory(device);
            }
            e.log(LogLevel::Error, msg.clone());
            e.state.lock().unwrap().outcome = Some((RunStatus::Failed, Some(msg), Some(code)));
            return;
        }
    };
    {
        let mut st = e.state.lock().unwrap();
        st.file.run.steps_per_epoch = trainer.batches_per_epoch() as u64;
        st.file.run.device = engine::describe(trainer.device());
        st.file.run.resolved = Some(trainer.resolved_train());
        e.persist(&st);
    }
    let device = trainer.device();
    if let Some(c) = folo_gpu::context_for(device) {
        c.reset_peak_memory();
    }
    let rc = trainer.run_config();
    let count = |k: &str| rc[k].as_u64().unwrap_or(0);
    let (train_images, val_images) = (count("train_images"), count("val_images"));
    *e.measure.lock().unwrap() = Some(Measure {
        throughput: throughput.clone(),
        key: TrainKey {
            device: engine::device_id(device),
            arch: config.arch,
            imgsz: config.imgsz,
            batch: config.batch,
            amp: config.amp,
        },
        train_images,
        // An empty val split validates the training split.
        val_images: if val_images == 0 { train_images } else { val_images },
        epochs: config.epochs,
        mosaic: config.augment.mosaic > 0.0,
        close_mosaic_epochs: count("close_mosaic_epochs") as u32,
        device,
        first: None,
        last_step: None,
    });
    // The trainer reports the outcome itself (the final `status` event, the error as a log line); the worker
    // publishes it after dropping the trainer (and with it the run's open files).
    if let Err(err) = trainer.run() {
        tracing::warn!(run = %e.id, "training failed: {err}");
    }
    drop(trainer);
    // The run's model, optimizer state, EMA and activations are back in the pool now; hand them to the driver
    // before the run is reported finished, so the next job on this GPU (a CLI run, another process) has its memory.
    engine::release_gpu_memory(device);
}

fn build(
    e: &Arc<RunEntry>,
    config: &TrainConfig,
    dataset: &DatasetRef,
    weights: Option<PathBuf>,
    hold: &Arc<EpochHold>,
) -> std::result::Result<Trainer, (String, String)> {
    let device =
        engine::resolve(&config.device).map_err(|err| (err.body.code.to_string(), err.body.message))?;
    let opts = folo_data::OpenOptions { format: data_format(dataset.format), verify_images: true };
    let format = opts.format;
    let ds = folo_data::open_with(Path::new(&dataset.path), &opts)
        .map_err(|err| ("data_error".to_string(), format!("opening the dataset {}: {err}", dataset.path)))?;
    if ds.train.is_empty() {
        return Err(("invalid_dataset".into(), "the dataset has no training images".into()));
    }
    let data = TrainData {
        train: Arc::new(ds.split(Split::Train).clone()),
        val: Arc::new(ds.split(Split::Val).clone()),
        classes: ds.classes.clone(),
    };
    let (sink, hold, stop) = (e.clone(), hold.clone(), e.stop.clone());
    let opts = TrainOptions {
        run_dir: Some(e.dir.clone()),
        events: Some(Box::new(move |ev: &RunEvent| {
            sink.on_event(ev);
            if matches!(ev, RunEvent::Epoch(_)) {
                hold.wait(&stop);
            }
        })),
        stop: e.stop.clone(),
        weights,
        ..TrainOptions::default()
    };
    let last = e.dir.join("weights").join("last.folo");
    let t = if last.is_file() {
        let overrides =
            ResumeOverrides { workers: None, device: Some(device.to_string()), data_format: Some(format) };
        Trainer::resume(&last, data, opts, overrides)
    } else {
        let mut cfg = folo_vision::TrainConfig::from_api(config)
            .map_err(|err| (err.code().to_string(), err.summary()))?;
        cfg.device = device.to_string();
        cfg.dataset = dataset.id.clone();
        cfg.data_format = format;
        Trainer::new(cfg, data, opts)
    };
    t.map_err(|err| (err.code().to_string(), err.summary()))
}
