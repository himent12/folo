//! Starting training runs (here or on a cloud connection) and reading their validation samples.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::http::StatusCode;
use folo_api::{
    BBox, CloudOffer, CloudPhase, Compute, DatasetFormat, LogLevel, Run, RunCloud, RunSamples, RunStatus,
    Sample, ScoredBox, TrainConfig,
};
use folo_core::Device;
use serde::Deserialize;

use crate::cloud::{self, Job, Mode};
use crate::engine::{self, engine_error};
use crate::error::{Error, Result};
use crate::index::{Index, absolute};
use crate::models::ModelRef;
use crate::runs::{CheckpointSnapshot, Ended, RunEntry};

use super::RealStore;

/// `invalid_config` for a dataset a run cannot train on (`POST /api/runs` and its estimate).
pub(super) fn check_trainable(idx: &Index) -> Result<()> {
    let bad = |m: &str| Err(Error::new(StatusCode::BAD_REQUEST, "invalid_config", m));
    if idx.classes.is_empty() {
        return bad("the dataset has no classes");
    }
    if idx.splits.train == 0 {
        return bad("the dataset has no training images");
    }
    Ok(())
}

impl RealStore {
    /// `POST /api/runs`: checks `config` and queues a new run, which continues the training state in the
    /// validated `checkpoint` snapshot when there is one (it becomes the run's `weights/last.folo`).
    pub(super) fn start_run(
        &self,
        config: TrainConfig,
        checkpoint: Option<CheckpointSnapshot<'_>>,
    ) -> Result<Run> {
        self.start_run_with_options(config, checkpoint, crate::store::RunOptions::default())
    }

    /// Like [`RealStore::start_run`], with optional GPU VRAM-budget auto-batch sizing (`auto_batch`).
    pub(super) fn start_run_with_auto_batch(
        &self,
        config: TrainConfig,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        auto_batch: crate::store::AutoBatchOptions,
    ) -> Result<Run> {
        self.start_run_with_options(
            config,
            checkpoint,
            crate::store::RunOptions { auto_batch, arch: None, max_preempt_retries: None },
        )
    }

    /// Like [`RealStore::start_run`], with optional auto-batch, `folo2-*` architecture, and cloud preemption overrides.
    pub(super) fn start_run_with_options(
        &self,
        mut config: TrainConfig,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        mut options: crate::store::RunOptions,
    ) -> Result<Run> {
        crate::store::check_auto_batch_options(&options.auto_batch)?;
        crate::store::check_arch_override(options.arch.as_deref())?;
        crate::store::check_max_preempt_retries(options.max_preempt_retries, config.compute.is_some())?;
        if options.auto_batch.enabled && config.compute.is_some() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                "auto_batch is not supported for cloud runs yet (pass an explicit integer `batch` for cloud compute)",
            )
            .hint("pick an integer `batch` for cloud runs, or omit `compute` to use auto-batch locally"));
        }
        // Shared with registration; a deletion of dataset files waits until the run is in the registry (see
        // `RealStore::files`).
        let _files = self.files.read().unwrap_or_else(|p| p.into_inner());
        crate::store::check_run_name(&config.name)?;
        let e = self.entry(&config.dataset_id)?;
        let idx = self.fresh_index(&e)?;
        check_trainable(&idx)?;
        let (weights, target) = self.run_settings(&mut config, &mut options.arch)?;
        let weights = weights.map(|m| m.path);
        let ds = self.dataset_ref(&e);
        if config.name.trim().is_empty() {
            config.name = crate::store::default_run_name_for_arch(
                &ds.name,
                crate::store::effective_arch(config.arch, options.arch.as_deref()),
            );
        }
        let val_offset = if idx.splits.val == 0 { 0 } else { idx.splits.train };
        match target {
            RunTarget::Cloud(cloud) => {
                let (compute, connected, offer) = *cloud;
                let device = format!("{} at {}", offer.gpu, connected.provider.info().name);
                let fingerprint = upload_fingerprint(ds.format, &idx);
                let cloud = run_cloud(&compute, &connected, offer);
                let entry = self.runs.create_cloud(
                    config,
                    ds,
                    weights,
                    checkpoint,
                    val_offset,
                    device,
                    cloud,
                    options.max_preempt_retries,
                )?;
                self.start_cloud(entry, connected, Some(fingerprint))
            }
            RunTarget::Local(device) => self.runs.create_with_options(
                config,
                ds,
                weights,
                checkpoint,
                val_offset,
                engine::describe(device),
                options,
            ),
        }
    }

    /// The checks of `POST /api/runs` that do not need the dataset: the starting weights (a fine-tune takes their
    /// architecture, as `folo train --weights` does, so `config.arch` and `arch_override` change), the settings
    /// (on the CPU, so a rejected config never initialises a GPU), and where the run would train: a cloud run's
    /// target, else the device, with `amp` checked as `folo train --amp` checks it (the same answer `/api/system`'s
    /// `capabilities.amp` gives). Returns the weights file and the target.
    pub(super) fn run_settings(
        &self,
        config: &mut TrainConfig,
        arch_override: &mut Option<String>,
    ) -> Result<(Option<ModelRef>, RunTarget)> {
        let weights = match &config.weights {
            Some(w) => {
                let m = self.model(w)?;
                config.arch = m.model.arch;
                *arch_override = m.arch_name.starts_with("folo2-").then(|| m.arch_name.clone());
                Some(m)
            }
            None => {
                if let Some(raw) = arch_override.as_deref() {
                    if let Some(parsed) = crate::store::check_arch_override(Some(raw))? {
                        config.arch = parsed;
                    }
                    if !raw.starts_with("folo2-") {
                        *arch_override = None;
                    }
                }
                None
            }
        };
        let mut check = folo_vision::TrainConfig::from_api(config).map_err(engine_error)?;
        if let Some(arch) = arch_override.as_deref() {
            check.arch = arch.to_string();
        }
        check.device = "cpu".into();
        check.validate().map_err(engine_error)?;
        // A cloud run's device and `amp` are the worker's to check (docs/CLOUD.md, worker check).
        if let Some(compute) = config.compute.clone() {
            if arch_override.is_some() {
                return Err(Error::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_config",
                    "folo2 architectures are not supported for cloud runs yet (train locally, or use folo-n..folo-x for cloud compute)",
                )
                .hint("omit `compute` to train folo2 locally, or select a folo-n..folo-x architecture for cloud runs"));
            }
            cloud::check_device(config)?;
            let (connected, offer) = self.cloud.target(&compute)?;
            return Ok((weights, RunTarget::Cloud(Box::new((compute, connected, offer)))));
        }
        let device = engine::resolve(&config.device)?;
        if config.amp {
            folo_vision::train::amp_available(device).map_err(engine_error)?;
        }
        Ok((weights, RunTarget::Local(device)))
    }

    /// Hands a registered (and claimed) cloud run to its executor. A run whose executor cannot start fails with the
    /// reason, which is also the answer.
    pub(super) fn start_cloud(
        &self,
        entry: Arc<RunEntry>,
        connected: cloud::Connected,
        fingerprint: Option<String>,
    ) -> Result<Run> {
        let job = Job { entry: entry.clone(), connected, upload: fingerprint };
        if let Err(e) = self.cloud.start(job, Mode::Start) {
            entry.execute(|| {
                entry.log(LogLevel::Error, e.body.message.clone());
                entry.set_outcome(RunStatus::Failed, Some(e.body.message.clone()), Some(e.body.code.clone()));
                Ended::Finished
            });
            return Err(e);
        }
        self.runs.snapshot(entry.id())
    }

    /// The validation samples of epoch `epoch` of a run, with image ids of its dataset.
    pub(super) fn samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples> {
        let e = self.runs.get(id)?;
        let dir = e.dir().join("samples");
        let available = |n: u32| dir.join(format!("epoch_{n:03}.json")).is_file();
        let done = e.run().epoch;
        let n = match epoch {
            Some(n) => n,
            None => (1..=done.max(1)).rev().find(|&n| available(n)).unwrap_or(0),
        };
        let not_available = || {
            Error::new(
                StatusCode::NOT_FOUND,
                "epoch_not_available",
                format!("run '{id}' has no samples for epoch {n}"),
            )
            .hint("samples exist once an epoch is validated")
        };
        if n == 0 || !available(n) {
            return Err(not_available());
        }
        let text =
            std::fs::read_to_string(dir.join(format!("epoch_{n:03}.json"))).map_err(|_| not_available())?;
        let file: SamplesFile = serde_json::from_str(&text)
            .map_err(|err| Error::internal(format!("samples of epoch {n} of run '{id}': {err}")))?;
        let ds = e.dataset();
        // Map files to the dataset's image ids; fall back to the split offset when the dataset is gone.
        let by_path: std::collections::HashMap<PathBuf, String> =
            match self.entry(&ds.id).and_then(|x| x.current()) {
                Ok(idx) => idx
                    .images
                    .iter()
                    .enumerate()
                    .map(|(i, im)| (absolute(&idx.root.join(&im.file)), i.to_string()))
                    .collect(),
                Err(_) => Default::default(),
            };
        let offset = e.val_offset();
        let name = |c: u32| file.classes.get(c as usize).cloned().unwrap_or_else(|| format!("class{c}"));
        let samples = file
            .samples
            .into_iter()
            .map(|s| Sample {
                image_id: s
                    .file
                    .as_ref()
                    .and_then(|f| by_path.get(&absolute(Path::new(f))).cloned())
                    .unwrap_or_else(|| (offset + s.index as u64).to_string()),
                dataset_id: ds.id.clone(),
                width: s.width,
                height: s.height,
                truth: s
                    .truth
                    .iter()
                    .map(|t| BBox { class: t.class, x1: t.x1, y1: t.y1, x2: t.x2, y2: t.y2 })
                    .collect(),
                pred: s
                    .pred
                    .iter()
                    .map(|p| ScoredBox {
                        class: p.class,
                        name: name(p.class),
                        score: p.score,
                        x1: p.x1,
                        y1: p.y1,
                        x2: p.x2,
                        y2: p.y2,
                    })
                    .collect(),
            })
            .collect();
        Ok(RunSamples { epoch: n, samples })
    }
}

/// Where a run trains ([`RealStore::run_settings`]).
pub(super) enum RunTarget {
    Cloud(Box<(Compute, cloud::Connected, CloudOffer)>),
    Local(Device),
}

/// `samples/epoch_NNN.json` as the trainer writes it.
#[derive(Deserialize)]
struct SamplesFile {
    classes: Vec<String>,
    samples: Vec<SampleRecord>,
}

#[derive(Deserialize)]
struct SampleRecord {
    index: usize,
    file: Option<String>,
    width: u32,
    height: u32,
    truth: Vec<BoxRecord>,
    pred: Vec<BoxRecord>,
}

#[derive(Deserialize)]
struct BoxRecord {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    class: u32,
    #[serde(default)]
    score: f32,
}

/// The fingerprint of a dataset upload: the dataset's format and the index fingerprint of its files, so the worker
/// is sent the dataset again exactly when its files (or how they are read) changed.
pub(super) fn upload_fingerprint(format: DatasetFormat, idx: &Index) -> String {
    let format =
        serde_json::to_value(format).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    format!("dataset-{format}-{}", idx.fingerprint.hash)
}

/// A new [`RunCloud`] for a run on `connected`'s `offer`: `provisioning` (`booting` for a server that already
/// runs).
pub(super) fn run_cloud(compute: &Compute, connected: &cloud::Connected, offer: CloudOffer) -> RunCloud {
    let info = connected.provider.info();
    RunCloud {
        connection_id: compute.connection_id.clone(),
        provider: info.id,
        offer,
        instance_id: None,
        phase: if info.provisions { CloudPhase::Provisioning } else { CloudPhase::Booting },
        progress: None,
        max_hours: compute.max_hours,
        deadline: None,
        cost_usd: None,
        synced: None,
        ended_in: None,
        unanswered_since: None,
    }
}
