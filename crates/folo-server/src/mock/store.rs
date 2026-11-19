//! [`Store`] for [`MockStore`].

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use axum::http::StatusCode;
use folo_api::{
    BBox, BenchmarkReport, BenchmarkRequest, Capabilities, CliCommand, CloudConnection, CloudInstance,
    CloudProvider, ConnectCloud, Dataset, DatasetFormat, ExportFormat, ExportResult, ExportSpec, GpuInfo,
    ImagePage, IssuePage, Model, ModelEval, PhaseKind, PredictResponse, RegisterDataset, Run, RunEstimate,
    RunLogs, RunSamples, Sample, ScoredBox, Splits, SyntheticDataset, ThroughputTable, TrainConfig,
};

use crate::error::{Error, Result};
use crate::index::{Index, default_name, free_name, requested_path};
use crate::store::{
    ImageData, ImageQuery, IssueQuery, LogsQuery, MetricsQuery, PredictInput, PredictSource, QueueMove,
    Store, Subscription, Weights, WeightsFormat,
};
use crate::util::{Rng, fnv1a, mix64, noise, rfc3339, unique_id, unix_now};

use super::cloud::{self, MockCloud, PHASE_S, SimCloud};
use super::datasets::{DatasetEntry, ProcDataset};
use super::models::{EvalBasis, fake_detections};
use super::runs::{self, Curve, LiveClock, SimRun, Timeline, arch_spec};
use super::scene::Theme;
use super::{
    DEVICE, LIVE_RUN, MOCK_GPU, MockStore, MockTensorCores, mock_weights, model_bytes, quality_of,
    steps_per_epoch, throughput, val,
};

impl Store for MockStore {
    fn is_mock(&self) -> bool {
        true
    }

    fn gpus(&self) -> Vec<GpuInfo> {
        let busy =
            self.runs.read().unwrap().iter().any(|r| r.0.snap().status == folo_api::RunStatus::Running);
        let t = unix_now() as u64;
        let util = if busy { 0.9 + 0.06 * noise(&[t, 1]) } else { 0.03 + 0.02 * noise(&[t, 2]).abs() };
        vec![GpuInfo {
            ordinal: 0,
            name: MOCK_GPU.into(),
            backend: self.tensor_cores.backend(),
            memory_bytes: 17_171_480_576,
            util: Some((util * 100.0).round() / 100.0),
        }]
    }

    fn capabilities(&self) -> Option<Capabilities> {
        Some(self.tensor_cores.capabilities())
    }

    fn features(&self) -> &'static [&'static str] {
        // Cursors and ranges and (unless `FOLO_MOCK_CLOUD=off`) the cloud, not uploads or checkpoints (docs/API.md,
        // Mock mode).
        match self.cloud {
            MockCloud::Off => &[folo_api::FEATURE_RUN_CURSORS],
            _ => &[folo_api::FEATURE_RUN_CURSORS, folo_api::FEATURE_CLOUD],
        }
    }

    fn throughput(&self) -> Result<ThroughputTable> {
        Ok(self.throughput.table(&throughput::gpu(), &[throughput::gpu(), throughput::cpu()]))
    }

    fn estimate_run(&self, config: TrainConfig) -> Result<RunEstimate> {
        let ds = self.dataset(&config.dataset_id)?;
        self.validate(&config)?;
        let splits = ds.meta().splits;
        if ds.meta().classes.is_empty() || splits.train == 0 {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("dataset_id: '{}' has no classes or no training images", ds.meta().id),
            ));
        }
        let augment = folo_data::AugmentConfig {
            close_mosaic: config.augment.close_mosaic,
            ..folo_data::AugmentConfig::default()
        };
        // A cloud run's time is the reference's, as on the real server.
        let cloud = match &config.compute {
            Some(c) => Some((self.cloud_offer(&config, c)?, c.max_hours)),
            None => None,
        };
        let device = match (&cloud, config.device.as_str()) {
            (Some((offer, _)), _) => crate::cloud::estimate_device(offer),
            (None, "cpu") => throughput::cpu(),
            _ => throughput::gpu(),
        };
        let cli = crate::cli::train_command(
            &config,
            &ds.meta().path,
            ds.meta().format,
            config.weights.as_deref().map(mock_weights).as_deref(),
        );
        let mut estimate = self.throughput.estimate(&crate::throughput::EstimateInput {
            device,
            arch: config.arch,
            imgsz: config.imgsz,
            amp: config.amp,
            batch: config.batch,
            epochs: config.epochs,
            mosaic: config.augment.mosaic > 0.0,
            close_mosaic_epochs: augment.close_mosaic_epochs(config.epochs),
            train_images: splits.train,
            val_images: if splits.val == 0 { splits.train } else { splits.val },
        });
        estimate.cli = Some(cli.cli);
        Ok(match cloud {
            Some((offer, max_hours)) => crate::cloud::cloud_estimate(estimate, &offer, max_hours),
            None => estimate,
        })
    }

    /// Answers at once with numbers near the table's (or the reference's): no device is used. Refused while a run
    /// created through the API is queued or training (the scripted live run does not count).
    fn benchmark(&self, req: BenchmarkRequest) -> Result<BenchmarkReport> {
        crate::probe::check(&req)?;
        let busy =
            self.runs.read().unwrap().iter().any(|r| r.0.id != LIVE_RUN && r.0.snap().status.is_active());
        if busy {
            return Err(Error::conflict(
                "run_active",
                "a training run is queued or running; a probe now would measure both slower",
            )
            .hint("wait until it finishes (or stop it), then probe again"));
        }
        let amp = req.amp && self.tensor_cores == MockTensorCores::On;
        if req.amp && !amp {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                self.tensor_cores.capabilities().amp.reason.unwrap_or_default(),
            ));
        }
        let gpu = throughput::gpu();
        let reference = crate::throughput::reference::train(true, req.arch, req.imgsz, amp);
        let seed = fnv1a(format!("{:?}{}{}{}", req.arch, req.imgsz, req.batch, amp).as_bytes());
        let ips = (reference.mosaic * (1.0 + 0.04 * noise(&[seed, 1]) as f64) * 10.0).round() / 10.0;
        let trained = self.throughput.record_train(crate::throughput::TrainSample {
            key: crate::throughput::TrainKey {
                device: gpu.clone(),
                arch: req.arch,
                imgsz: req.imgsz,
                batch: req.batch,
                amp,
            },
            phase: PhaseKind::Mosaic,
            train_imgs_per_s: Some(ips),
            val_imgs_per_s: None,
            startup_s: None,
            peak_vram_bytes: Some(crate::throughput::reference::memory(req.arch, req.imgsz, req.batch, amp)),
            probe: true,
        });
        let (latency, forward, _) =
            crate::throughput::reference::infer(true, req.arch, req.imgsz, folo_api::InferPrecision::F32);
        let calls = (req.budget_s * 0.4 * 1000.0 / latency).clamp(5.0, 500.0) as u64;
        let mut inferred = None;
        for i in 0..calls.min(crate::throughput::WINDOW as u64) {
            let k = 1.0 + 0.08 * noise(&[seed, 100 + i]) as f64;
            inferred = Some(self.throughput.record_infer(crate::throughput::InferSample {
                key: crate::throughput::InferKey {
                    device: gpu.clone(),
                    arch: req.arch,
                    imgsz: req.imgsz,
                    precision: folo_api::InferPrecision::F32,
                    tensor_cores: false,
                },
                total_ms: (latency + 0.9) * k,
                inference_ms: forward * k,
            }));
        }
        let inference = crate::throughput::infer_row(&inferred.expect("at least 5 predictions"));
        Ok(BenchmarkReport {
            device: gpu.device,
            device_name: gpu.device_name,
            train_imgs_per_s: ips,
            train_batches: ((req.budget_s * 0.6 * 0.7 * ips) / req.batch as f64).max(1.0) as u64,
            latency_ms: inference.latency_ms,
            predictions: calls,
            elapsed_s: ((req.budget_s + 1.8) * 10.0).round() / 10.0,
            training: crate::throughput::train_row(&trained),
            inference,
        })
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>> {
        Ok(self.datasets.read().unwrap().iter().map(|d| d.meta().clone()).collect())
    }

    fn register_dataset(&self, mut req: RegisterDataset) -> Result<Dataset> {
        req.path = requested_path(&req.path)?;
        // Registering a folder again (by its canonical path) returns the dataset it already is: checked before
        // indexing and again under the write lock.
        let want = std::fs::canonicalize(&req.path).ok();
        let same = |list: &[Arc<DatasetEntry>]| {
            let want = want.as_ref()?;
            list.iter().find_map(|d| match &**d {
                DatasetEntry::Disk(meta, _)
                    if std::fs::canonicalize(&meta.path).ok().as_ref() == Some(want)
                        && (req.format == DatasetFormat::Auto || req.format == meta.format) =>
                {
                    Some(d.with_stats())
                }
                _ => None,
            })
        };
        if let Some(d) = same(&self.datasets.read().unwrap()) {
            return Ok(d);
        }
        let index = Index::build(&req.path, req.format, &[])?;
        let mut list = self.datasets.write().unwrap();
        if let Some(d) = same(&list) {
            return Ok(d);
        }
        // As in real mode: the same folder registered again gets its id back; another gets an id never used.
        let mut retired = self.retired.write().unwrap();
        let back = retired.iter().position(|(path, format, id)| {
            path.is_some()
                && *path == want
                && *format == index.format
                && !list.iter().any(|d| d.meta().id == *id)
        });
        let id = match back {
            Some(i) => retired.remove(i).2,
            None => unique_id("ds", fnv1a(req.path.as_bytes()) ^ unix_now().to_bits(), |id| {
                list.iter().any(|d| d.meta().id == id) || retired.iter().any(|r| r.2 == id)
            }),
        };
        let name = req.name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
            default_name(Path::new(&req.path), |n| list.iter().any(|d| d.meta().name.eq_ignore_ascii_case(n)))
        });
        let meta = Dataset {
            id,
            name,
            path: req.path,
            format: index.format,
            classes: index.classes.clone(),
            created: rfc3339(unix_now()),
            splits: index.splits,
            stats: None,
            revision: None,
        };
        let entry = DatasetEntry::Disk(meta, Box::new(index));
        let out = entry.with_stats();
        list.push(Arc::new(entry));
        Ok(out)
    }

    fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset> {
        let bad = |m: &str| Err(Error::new(StatusCode::BAD_REQUEST, "invalid_body", m.to_string()));
        if req.name.trim().is_empty() {
            return bad("name must not be empty");
        }
        if !(1..=100_000).contains(&req.images) {
            return bad("images must be in 1..=100000");
        }
        if req.classes.is_empty() || req.classes.len() > 80 || req.classes.iter().any(|c| c.trim().is_empty())
        {
            return bad("classes must hold 1..=80 non-empty names");
        }
        if !(32..=4096).contains(&req.imgsz) {
            return bad("imgsz must be in 32..=4096");
        }
        let seed = fnv1a(req.name.as_bytes()) ^ req.seed ^ unix_now().to_bits();
        let train = (req.images as f64 * 0.8).round() as u64;
        let splits = Splits { train, val: req.images - train, test: 0 };
        let mut ds = ProcDataset::build(
            String::new(),
            req.name,
            Theme::Shapes,
            req.classes,
            req.imgsz,
            req.imgsz,
            splits,
            mix64(req.seed ^ 0x5717),
            rfc3339(unix_now()),
            0,
        );
        let mut list = self.datasets.write().unwrap();
        let retired = self.retired.read().unwrap();
        ds.meta.id = unique_id("ds", seed, |id| {
            list.iter().any(|d| d.meta().id == id) || retired.iter().any(|r| r.2 == id)
        });
        // As the real store names a sample set: numbered while a dataset has the name.
        let lower = |n: &str| n.to_lowercase();
        ds.meta.name = free_name(std::mem::take(&mut ds.meta.name), |n| {
            list.iter().any(|d| lower(&d.meta().name) == lower(n))
        });
        let entry = Arc::new(DatasetEntry::Proc(ds));
        let out = entry.with_stats();
        list.push(entry);
        Ok(out)
    }

    fn get_dataset(&self, id: &str) -> Result<Dataset> {
        Ok(self.dataset(id)?.with_stats())
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage> {
        Ok(self.dataset(id)?.list_images(q))
    }

    fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<IssuePage> {
        Ok(self.dataset(id)?.list_issues(q))
    }

    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        self.dataset(id)?.image(image_id, thumb)
    }

    fn delete_dataset_files(&self, id: &str) -> Result<()> {
        // Mock mode writes no files: a generated set (made in memory) goes like any other; a folder on disk is
        // never this server's to delete.
        if matches!(&*self.dataset(id)?, DatasetEntry::Disk(..)) {
            let m = "a folder registered from disk is not one this server wrote; its files are never deleted";
            return Err(Error::new(StatusCode::BAD_REQUEST, "invalid_query", m.to_string()));
        }
        // As in real mode, never while a run is active (the scripted live run does not count).
        let runs = self.runs.read().unwrap();
        if let Some(r) = runs.iter().find(|r| r.0.id != LIVE_RUN && r.0.snap().status.is_active()) {
            let m = format!(
                "run {} is queued or training; files are deleted only while no run is",
                r.0.config.name
            );
            return Err(Error::conflict("run_active", m).hint("wait for it to end, or stop it first"));
        }
        drop(runs);
        self.delete_dataset(id)
    }

    fn delete_dataset(&self, id: &str) -> Result<()> {
        let mut ds = self.datasets.write().unwrap();
        let i = ds.iter().position(|d| d.meta().id == id).ok_or_else(|| Error::not_found("dataset", id))?;
        let gone = ds.remove(i);
        let path = match &*gone {
            DatasetEntry::Disk(meta, _) => std::fs::canonicalize(&meta.path).ok(),
            DatasetEntry::Proc(_) => None,
        };
        self.retired.write().unwrap().push((path, gone.meta().format, id.to_string()));
        Ok(())
    }

    fn list_models(&self) -> Result<Vec<Model>> {
        Ok(self.all_models())
    }

    fn get_model(&self, id: &str) -> Result<Model> {
        self.all_models().into_iter().find(|m| m.id == id).ok_or_else(|| Error::not_found("model", id))
    }

    fn model_eval(&self, id: &str) -> Result<ModelEval> {
        if let Some(m) = self.models.read().unwrap().iter().find(|m| m.model.id == id) {
            return Ok(m.basis.eval());
        }
        let runs = self.runs.read().unwrap();
        let (run, ds) = runs
            .iter()
            .find(|(r, _)| r.user && r.current_model(&r.snap()).as_deref() == Some(id))
            .ok_or_else(|| Error::not_found("model", id))?;
        Ok(EvalBasis::new(run, ds).eval())
    }

    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult> {
        crate::store::check_export(&spec)?;
        let m = self.get_model(id)?;
        let (ext, bytes) = match spec.format {
            ExportFormat::FoloF16 => ("f16.folo", model_bytes(m.params, true)),
            ExportFormat::Onnx if spec.onnx.half => ("f16.onnx", m.params * 2 + 61_440),
            ExportFormat::Onnx => ("onnx", m.params * 4 + 61_440),
        };
        Ok(ExportResult { path: format!("mock://exports/{}.{ext}", m.name), bytes })
    }

    fn model_weights(&self, id: &str, format: WeightsFormat) -> Result<Weights> {
        self.weights.get(&self.get_model(id)?, format)
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        let mut runs: Vec<Run> = self.runs.read().unwrap().iter().map(|r| r.0.to_api()).collect();
        runs.sort_by(|a, b| b.created.cmp(&a.created).then(b.id.cmp(&a.id)));
        Ok(runs)
    }

    fn create_run(&self, mut config: TrainConfig) -> Result<Run> {
        let ds = self.dataset(&config.dataset_id)?;
        crate::store::check_run_name(&config.name)?;
        self.validate(&config)?;
        // A cloud run rents an offer of its connection, and its worker checks `amp`.
        let offer = match &config.compute {
            Some(c) => Some((self.cloud_offer(&config, c)?, c.clone())),
            None => None,
        };
        // As the real store: a fine-tune takes the weights' architecture, and `amp` is refused up front where
        // `capabilities.amp` says it is unavailable.
        if let Some(w) = &config.weights {
            config.arch = self.get_model(w)?.arch;
        }
        if offer.is_none()
            && config.amp
            && (config.device == "cpu" || self.tensor_cores != MockTensorCores::On)
        {
            let why = if config.device == "cpu" {
                "the run is on the CPU".to_string()
            } else {
                format!(
                    "gpu:0 has tensor cores {}",
                    self.tensor_cores.capabilities().tensor_cores.reason.unwrap_or_default()
                )
            };
            return Err(crate::engine::engine_error(folo_vision::train::amp_not_supported(&why)));
        }
        let meta = ds.meta();
        if meta.classes.is_empty() || meta.splits.train == 0 {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("dataset_id: '{}' has no classes or no training images", meta.id),
            ));
        }
        if config.name.trim().is_empty() {
            config.name = crate::store::default_run_name(&meta.name, config.arch);
        }
        // Ids are allocated under the runs write lock so concurrent creates cannot pick the same one, and model
        // ids are checked against every model that exists or will exist when a run completes.
        let mut runs = self.runs.write().unwrap();
        let id = unique_id("run", fnv1a(config.name.as_bytes()) ^ unix_now().to_bits(), |id| {
            runs.iter().any(|r| r.0.id == id)
        });
        let models_taken: std::collections::HashSet<String> = self
            .models
            .read()
            .unwrap()
            .iter()
            .map(|m| m.model.id.clone())
            .chain(runs.iter().filter_map(|r| r.0.model_id.clone()))
            .collect();
        let model_id = unique_id("m", fnv1a(id.as_bytes()), |m| models_taken.contains(m));
        let spe = steps_per_epoch(meta.splits.train, config.batch);
        let quality = 0.45 + 0.4 * (fnv1a(meta.id.as_bytes()) % 100) as f32 / 100.0;
        let curve = Curve::new(fnv1a(id.as_bytes()), quality, meta.classes.len(), &config, spe);
        let device = match (&offer, config.device.as_str()) {
            (Some((o, _)), _) => format!("NVIDIA GeForce {} (vulkan)", o.gpu),
            (None, "cpu") => "CPU".to_string(),
            _ => DEVICE.to_string(),
        };
        // A cloud run hands over for 3 phases before it trains and 2 after, outside the training queue.
        let clock = match offer {
            Some(_) => LiveClock::new(6.0, 3.0 * PHASE_S, 0.0, false, 0.0).with_tail(2.0 * PHASE_S),
            None => LiveClock::new(6.0, 2.0, 0.0, false, 0.0),
        };
        let cloud = offer.map(|(o, c)| SimCloud::new(&id, &c.connection_id, o, c.max_hours));
        let local = cloud.is_none();
        let mut sim = SimRun::new(
            id,
            config,
            meta.classes.clone(),
            spe,
            curve,
            device,
            Some(model_id),
            true,
            Timeline::Live(std::sync::Mutex::new(clock)),
        )
        .resolved_for(meta.splits.train);
        if let Some(c) = cloud {
            sim = sim.with_cloud(c);
        }
        let run = Arc::new(sim);
        runs.push((Arc::clone(&run), Arc::clone(&ds)));
        drop(runs);
        if local {
            let mut queue = self.queue.lock().unwrap();
            queue.push(Arc::clone(&run));
            runs::reschedule(&queue, std::time::Instant::now());
        }
        Ok(run.to_api())
    }

    fn get_run(&self, id: &str) -> Result<Run> {
        Ok(self.run(id)?.0.to_api())
    }

    fn run_cli(&self, id: &str) -> Result<CliCommand> {
        let (run, ds) = self.run(id)?;
        crate::cli::check_optimizer(&run.config)?;
        let weights = run.config.weights.as_deref().map(mock_weights);
        Ok(crate::cli::train_command(&run.config, &ds.meta().path, ds.meta().format, weights.as_deref()))
    }

    fn stop_run(&self, id: &str) -> Result<Run> {
        let (run, _) = self.run(id)?;
        if !run.stop() {
            return Err(Error::run_not_active(
                id,
                serde_json::to_value(run.to_api().status).unwrap().as_str().unwrap_or(""),
            ));
        }
        // The runs behind it start earlier now.
        runs::reschedule(&self.queue.lock().unwrap(), std::time::Instant::now());
        Ok(run.to_api())
    }

    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run> {
        let (run, _) = self.run(id)?;
        let moved = runs::move_waiting(&mut self.queue.lock().unwrap(), id, mv);
        if let Some((from, to, waiting)) = moved {
            let after = run.to_api();
            let place = format!("{} of {waiting}", to + 1);
            let message = match mv {
                QueueMove::Start if after.status == folo_api::RunStatus::Running => {
                    "start requested: starting now".to_string()
                }
                QueueMove::Start => {
                    format!(
                        "start requested: moved to the front ({place}); it starts when the active run ends"
                    )
                }
                _ if from == to => format!("queue position unchanged ({place})"),
                _ => format!("moved in the queue to position {place}"),
            };
            run.note(folo_api::LogLevel::Info, message);
            return Ok(after);
        }
        let what = match run.to_api().status {
            folo_api::RunStatus::Queued => "is not in the training queue".to_string(),
            s => format!("is {}", serde_json::to_value(s).unwrap().as_str().unwrap_or("finished")),
        };
        Err(Error::conflict("run_not_queued", format!("run '{id}' {what}, not waiting in the queue"))
            .hint("only runs waiting in the queue (with a queue_position) can be moved"))
    }

    fn delete_run(&self, id: &str) -> Result<()> {
        let (run, _) = self.run(id)?;
        // A local run waiting in the queue that never started leaves it (as in real mode); any other active run stays.
        let api = run.to_api();
        let never_started = api.status == folo_api::RunStatus::Queued
            && api.started.is_none()
            && api.config.compute.is_none();
        if run.is_active() && !never_started {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                .hint(format!("stop it first: POST /api/runs/{id}/stop")));
        }
        run.delete();
        self.runs.write().unwrap().retain(|r| r.0.id != id);
        let mut queue = self.queue.lock().unwrap();
        queue.retain(|r| r.id != id);
        // The runs behind it start earlier now.
        runs::reschedule(&queue, std::time::Instant::now());
        Ok(())
    }

    fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<folo_api::RunMetrics> {
        Ok(q.select(self.run(id)?.0.metrics()))
    }

    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples> {
        let (run, ds) = self.run(id)?;
        let done = run.epochs_done(&run.snap());
        let epoch = epoch.unwrap_or(done);
        if epoch == 0 || epoch > done {
            return Err(Error::new(
                StatusCode::NOT_FOUND,
                "epoch_not_available",
                format!("run '{id}' has no validation results for epoch {epoch} (finished epochs: {done})"),
            ));
        }
        let quality = run.curve.val(epoch).map50;
        let samples = ds
            .sample_ids(16)
            .into_iter()
            .filter_map(|image_id| {
                let (w, h, truth) = ds.truth(&image_id).ok()?;
                let seed = mix64(fnv1a(run.id.as_bytes()) ^ epoch as u64) ^ fnv1a(image_id.as_bytes());
                let pred: Vec<ScoredBox> = fake_detections(&truth, w, h, &run.classes, quality, seed)
                    .into_iter()
                    .filter(|b| b.score >= 0.25)
                    .collect();
                Some(Sample { image_id, dataset_id: ds.meta().id.clone(), width: w, height: h, truth, pred })
            })
            .collect();
        Ok(RunSamples { epoch, samples })
    }

    fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs> {
        Ok(q.select(self.run(id)?.0.logs()))
    }

    fn subscribe(&self, id: &str) -> Result<Subscription> {
        Ok(self.run(id)?.0.subscribe())
    }

    fn predict(&self, input: PredictInput) -> Result<PredictResponse> {
        let model = self.get_model(&input.model_id)?;
        let (w, h, truth, seed) = match &input.source {
            PredictSource::Dataset { dataset_id, image_id } => {
                let ds = self.dataset(dataset_id)?;
                let (w, h, truth) = ds.truth(image_id)?;
                // The model only knows its own classes: map the truth by class name and drop the rest.
                let names = &ds.meta().classes;
                let truth: Vec<BBox> = truth
                    .into_iter()
                    .filter_map(|b| {
                        let name = names.get(b.class as usize)?;
                        let k = model.classes.iter().position(|c| c == name)?;
                        Some(BBox { class: k as u32, ..b })
                    })
                    .collect();
                (w, h, truth, fnv1a(format!("{dataset_id}/{image_id}").as_bytes()))
            }
            PredictSource::Upload(bytes) => {
                let (w, h) = image::ImageReader::new(Cursor::new(bytes))
                    .with_guessed_format()
                    .map_err(|e| Error::internal(e.to_string()))?
                    .into_dimensions()
                    .map_err(|e| {
                        Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
                            .hint("send a JPEG, PNG, BMP or WebP file in the `image` field")
                    })?;
                let seed = fnv1a(bytes);
                let mut r = Rng::new(seed);
                let n = 1 + r.below(6);
                let nc = model.classes.len().max(1) as u64;
                let truth = (0..n)
                    .map(|_| {
                        let (bw, bh) = (r.range(0.08, 0.4) * w as f32, r.range(0.08, 0.4) * h as f32);
                        let (x, y) = (r.range(0.0, w as f32 - bw), r.range(0.0, h as f32 - bh));
                        BBox { class: r.below(nc) as u32, x1: x, y1: y, x2: x + bw, y2: y + bh }
                    })
                    .collect();
                (w, h, truth, seed)
            }
        };
        let seed = seed ^ fnv1a(model.id.as_bytes());
        let mut boxes: Vec<ScoredBox> =
            fake_detections(&truth, w, h, &model.classes, quality_of(&model), seed)
                .into_iter()
                .filter(|b| b.score >= input.conf)
                .collect();
        boxes.truncate(input.max_det as usize);
        let mut r = Rng::new(seed ^ 0x7131);
        let (.., inf_ms) = arch_spec(model.arch);
        let round = |v: f64| (v * 100.0).round() / 100.0;
        let mp = w as f64 * h as f64 / 1e6;
        let preprocess_ms = round(0.35 + 0.28 * mp + 0.1 * r.f32() as f64);
        let inference_ms =
            round(inf_ms * (model.imgsz as f64 / 640.0).powi(2) * (0.94 + 0.12 * r.f32() as f64));
        let postprocess_ms = round(0.12 + 0.015 * boxes.len() as f64 + 0.05 * r.f32() as f64);
        Ok(PredictResponse {
            model_id: model.id,
            image: folo_api::ImageSize { width: w, height: h },
            boxes,
            timing: folo_api::Timing {
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms: round(preprocess_ms + inference_ms + postprocess_ms),
            },
            device: "gpu:0".into(),
            cold: false,
        })
    }

    fn cloud_providers(&self) -> Result<Vec<CloudProvider>> {
        self.providers()
    }

    fn cloud_connections(&self) -> Result<Vec<CloudConnection>> {
        self.providers()?;
        Ok(self.connections.read().unwrap().clone())
    }

    fn connect_cloud(&self, req: ConnectCloud) -> Result<CloudConnection> {
        let provider = self.providers()?.into_iter().find(|p| p.id == req.provider).ok_or_else(|| {
            Error::not_found("provider", &req.provider).hint("GET /api/cloud/providers lists the providers")
        })?;
        let mut list = self.connections.write().unwrap();
        let seed = fnv1a(req.provider.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("cc", seed, |id| list.iter().any(|c| c.id == id));
        let c = cloud::connect(req, &provider, id, &list, unix_now())?;
        list.push(c.clone());
        Ok(c)
    }

    fn check_cloud(&self, id: &str) -> Result<CloudConnection> {
        self.connection(id)?;
        let mut list = self.connections.write().unwrap();
        let c = list.iter_mut().find(|c| c.id == id).ok_or_else(|| crate::cloud::connection_not_found(id))?;
        c.checked = Some(rfc3339(unix_now()));
        Ok(c.clone())
    }

    fn disconnect_cloud(&self, id: &str) -> Result<()> {
        self.connection(id)?;
        let runs = self.runs.read().unwrap();
        let busy: Vec<&str> = runs
            .iter()
            .filter(|(r, _)| r.cloud().is_some_and(|c| c.connection_id == id))
            .filter(|(r, _)| r.is_active() || r.instance().is_some_and(|i| i.status.is_alive()))
            .map(|(r, _)| r.id.as_str())
            .collect();
        if !busy.is_empty() {
            return Err(Error::conflict(
                "connection_in_use",
                format!("connection '{id}' has active run(s) {}", busy.join(", ")),
            )
            .hint("stop its runs and wait until GET /api/cloud/instances shows its instances terminated"));
        }
        self.connections.write().unwrap().retain(|c| c.id != id);
        Ok(())
    }

    fn cloud_offers(&self, id: &str, _refresh: bool) -> Result<folo_api::CloudOfferList> {
        let c = self.connection(id)?;
        Ok(folo_api::CloudOfferList { offers: MockStore::offers_of(&c), fetched: rfc3339(unix_now()) })
    }

    fn cloud_instances(&self) -> Result<Vec<CloudInstance>> {
        self.providers()?;
        Ok(self.instances())
    }

    /// A cloud run whose instance is terminated by hand fails with `cloud_instance_lost`.
    fn terminate_instance(&self, id: &str) -> Result<CloudInstance> {
        self.providers()?;
        let runs = self.runs.read().unwrap();
        let (run, _) = runs
            .iter()
            .find(|(r, _)| r.cloud().is_some_and(|c| c.instance_id == id))
            .ok_or_else(|| crate::cloud::instance_not_found(id))?;
        run.lose();
        Ok(run.instance().expect("a cloud run has its instance"))
    }

    fn val(&self, req: folo_api::ValRequest) -> Result<folo_api::ValReport> {
        val::val(self, req)
    }
}
