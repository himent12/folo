//! [`Store`] for [`RealStore`].

use std::path::{Path, PathBuf};
use std::time::Instant;

use axum::body::Bytes;
use axum::http::StatusCode;
use folo_api::{
    BenchmarkReport, BenchmarkRequest, Capabilities, CliCommand, CloudConnection, CloudInstance,
    CloudOfferList, CloudProvider, ConnectCloud, CreateUpload, Dataset, DatasetFormat, ExportResult,
    ExportSpec, GpuInfo, GpuPool, ImagePage, ImageSize, InferPrecision, Model, ModelEval, PredictResponse,
    RegisterDataset, Run, RunEstimate, RunLogs, RunMetrics, RunSamples, SyntheticDataset, ThroughputTable,
    TrainConfig, Upload, UploadCreated, UploadKind, Uploaded, ValReport, ValRequest,
};
use folo_core::Device;
use folo_vision::PredictConfig;

use crate::bench::Benches;
use crate::cloud::{self};
use crate::engine::{self, engine_error};
use crate::error::{Error, Result};
use crate::index::{data_error, read_image_with, requested_path, write_atomic};
use crate::label::Labeler;
use crate::runs::run_model_id;
use crate::store::{
    ImageData, ImageQuery, IssueQuery, LogsQuery, MetricsQuery, PredictInput, PredictSource, QueueMove,
    Store, Subscription, Weights, WeightsFormat, weights_file_name,
};
use crate::throughput::{EstimateInput, InferKey, InferSample};
use crate::uploads::Uploads;
use crate::util::slug;

use super::datasets::Naming;
use super::models::cli_weights;
use super::training::{run_cloud, upload_fingerprint};
use super::{MAX_SYNTHETIC_IMAGES, RealStore, SHUTDOWN_WAIT, State};

impl Store for RealStore {
    fn is_mock(&self) -> bool {
        false
    }

    fn gpus(&self) -> Vec<GpuInfo> {
        engine::gpus()
    }

    fn prepare_gpus(&self) {
        engine::adapters();
    }

    fn capabilities(&self) -> Option<Capabilities> {
        Some(engine::capabilities())
    }

    fn gpu_pool(&self) -> Option<GpuPool> {
        engine::gpu_pool()
    }

    fn data_dir(&self) -> Option<&Path> {
        Some(&self.data_dir)
    }

    fn features(&self) -> &'static [&'static str] {
        &[
            folo_api::FEATURE_UPLOADS,
            folo_api::FEATURE_RUN_CURSORS,
            folo_api::FEATURE_RUN_CHECKPOINT,
            folo_api::FEATURE_RUN_RESUME,
            folo_api::FEATURE_CLOUD,
        ]
    }

    fn throughput(&self) -> Result<ThroughputTable> {
        let auto = engine::device_id(engine::resolve("auto")?);
        let mut devices = vec![auto.clone()];
        if auto.is_gpu() {
            devices.push(engine::device_id(Device::Cpu));
        }
        Ok(self.throughput.table(&auto, &devices))
    }

    fn estimate_run(&self, mut config: TrainConfig) -> Result<RunEstimate> {
        let e = self.entry(&config.dataset_id)?;
        let idx = self.fresh_index(&e)?;
        let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_config", m);
        if idx.classes.is_empty() {
            return Err(bad("the dataset has no classes".into()));
        }
        if idx.splits.train == 0 {
            return Err(bad("the dataset has no training images".into()));
        }
        // A fine-tune trains the weights' architecture (see create_run).
        let weights = match &config.weights {
            Some(w) => {
                let m = self.model(w)?;
                config.arch = m.model.arch;
                Some(m)
            }
            None => None,
        };
        // The same checks as POST /api/runs, the settings first (on the CPU).
        let mut check = folo_vision::TrainConfig::from_api(&config).map_err(engine_error)?;
        check.device = "cpu".into();
        check.validate().map_err(engine_error)?;
        // A cloud run's time is the reference's: nothing measured here says how fast the offer's GPU is.
        let cloud = match &config.compute {
            Some(compute) => {
                cloud::check_device(&config)?;
                Some((self.cloud.target(compute)?.1, compute.max_hours))
            }
            None => None,
        };
        let device = match &cloud {
            Some((offer, _)) => cloud::estimate_device(offer),
            None => engine::device_id(engine::resolve(&config.device)?),
        };
        let (train, val) = (idx.splits.train, idx.splits.val);
        let cli = crate::cli::train_command(
            &config,
            &e.reg().path,
            e.reg().format,
            weights.as_ref().map(cli_weights).as_deref(),
        );
        let mut estimate = self.throughput.estimate(&EstimateInput {
            device,
            arch: config.arch,
            imgsz: config.imgsz,
            amp: config.amp,
            batch: config.batch,
            epochs: config.epochs,
            mosaic: config.augment.mosaic > 0.0,
            close_mosaic_epochs: check.augment.close_mosaic_epochs(config.epochs),
            train_images: train,
            val_images: if val == 0 { train } else { val },
        });
        estimate.cli = Some(cli.cli);
        Ok(match cloud {
            Some((offer, max_hours)) => cloud::cloud_estimate(estimate, &offer, max_hours),
            None => estimate,
        })
    }

    fn run_cli(&self, id: &str) -> Result<CliCommand> {
        let e = self.runs.get(id)?;
        let config = e.run().config;
        crate::cli::check_optimizer(&config)?;
        // A published model by its name; any other by the file the run started from.
        let weights = match (&config.weights, e.weights()) {
            (Some(w), _) if folo_api::weights::find(w).is_some() => Some(w.clone()),
            (_, Some(p)) => Some(folo_api::json::path(&p)),
            _ => None,
        };
        let ds = e.dataset();
        let cmd = crate::cli::train_command(&config, &ds.path, ds.format, weights.as_deref());
        // An `auto` run repeats with the rule it trained with (made before rule 3: SGD, not MuSGD, for long runs),
        // and every run with its warmup floor (made before the key: 100 iterations).
        let rule = crate::cli::recorded_auto_rule(e.dir());
        let cmd = crate::cli::with_auto_rule(cmd, &config, rule);
        // A transfer run whose command has no `--weights` keeps its MuSGD step weights (rule 5 vs 4, §8.2).
        let (transfer, weights) =
            (crate::cli::recorded_transfer(e.dir()), crate::cli::recorded_musgd_weights(e.dir()));
        let cmd = crate::cli::with_transfer(cmd, &config, rule, transfer, weights);
        let cmd = crate::cli::with_warmup_floor(cmd, crate::cli::recorded_warmup_floor(e.dir()));
        // A run continued from a distillation run's checkpoint keeps its teacher.
        Ok(crate::cli::with_distill(cmd, crate::cli::recorded_distill(e.dir())))
    }

    fn benchmark(&self, req: BenchmarkRequest) -> Result<BenchmarkReport> {
        crate::probe::check(&req)?;
        let _gpu = self.idle.using();
        let device = engine::resolve(&self.device)?;
        _gpu.on(device);
        let _gate = self.runs.probe_gate()?;
        crate::probe::run(&req, device, self.precision, &self.throughput)
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>> {
        // A registry that could not be read at start loaded no dataset: that is an error to show (with how to fix
        // it), not "no datasets yet" (x3-server-7).
        self.registry_loaded()?;
        let list = self.entries.read().unwrap().clone();
        Ok(list
            .iter()
            .map(|e| {
                let idx = match &e.slot.lock().unwrap().state {
                    State::Ready(i) => Some(i.clone()),
                    _ => None,
                };
                Dataset { stats: None, revision: None, ..Self::meta(&e.reg(), idx.as_deref()) }
            })
            .collect())
    }

    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset> {
        self.register(req.name.map_or(Naming::Default, Naming::Given), requested_path(&req.path)?, req.format)
    }

    fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset> {
        let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_body", m);
        if req.name.trim().is_empty() {
            return Err(bad("name must not be empty".into()));
        }
        if !(1..=MAX_SYNTHETIC_IMAGES).contains(&req.images) {
            return Err(bad(format!("images must be in 1..={MAX_SYNTHETIC_IMAGES}")));
        }
        if !(32..=4096).contains(&req.imgsz) {
            return Err(bad("imgsz must be in 32..=4096".into()));
        }
        let names: Vec<String> = req.classes.iter().map(|c| c.trim().to_string()).collect();
        let shapes = folo_data::synthetic::shapes_for(&names).map_err(|e| bad(e.to_string()))?;
        let cfg = folo_data::synthetic::SyntheticConfig {
            images: req.images as usize,
            classes: shapes,
            names,
            imgsz: req.imgsz,
            seed: req.seed,
            vary_aspect: false,
            val_fraction: 0.2,
            // A sample set passes FOLO's own health checks: Studio warns past 10% tiny boxes, and the generator's
            // default (a fifth of the objects 8 to 28 px across) makes about 27%. A few small ones stay (about 6%).
            small_fraction: 0.05,
            ..Default::default()
        };
        // Registration would refuse it (a registry not loaded at start): said before anything is generated.
        self.registry_loaded()?;
        let dir = self.claim_dataset_dir(&slug(&req.name, "synthetic"))?;
        let claim = claim_generated(&dir)?;
        let t0 = Instant::now();
        // Only a folder this call claimed is removed: no other writer can have put a file in it.
        if let Err(e) = folo_data::synthetic::write_yolo_with_progress(
            &cfg,
            claim,
            folo_data::ImageFormat::Jpeg(90),
            |_| {},
        ) {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(data_error(e, &dir.to_string_lossy()));
        }
        tracing::info!(path = %dir.display(), images = cfg.images, secs = t0.elapsed().as_secs_f32(), "generated");
        let path = dir.to_string_lossy().into_owned();
        // A registration that fails writes nothing to the registry: the generated folder would be nobody's.
        self.register(Naming::Unique(req.name.trim().to_string()), path, DatasetFormat::Yolo).inspect_err(
            |_| {
                let _ = std::fs::remove_dir_all(&dir);
            },
        )
    }

    fn get_dataset(&self, id: &str) -> Result<Dataset> {
        let e = self.entry(id)?;
        let idx = self.fresh_index(&e)?;
        Ok(Self::meta(&e.reg(), Some(&idx)))
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage> {
        let e = self.entry(id)?;
        Ok(self.fresh_index(&e)?.list_images(q))
    }

    fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<folo_api::IssuePage> {
        let e = self.entry(id)?;
        Ok(self.fresh_index(&e)?.list_issues(q))
    }

    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        let e = self.entry(id)?;
        let path = e.current()?.image_path(image_id)?;
        let cache = thumb.map(|_| self.thumbs_dir(id));
        read_image_with(&path, thumb, cache.as_deref(), |file, bytes| {
            e.write_cache(|| {
                if let Err(err) = write_atomic(file, bytes) {
                    tracing::warn!("caching thumbnail {}: {err}", file.display());
                }
            })
        })
    }

    fn delete_dataset(&self, id: &str) -> Result<()> {
        // Exclusive (briefly), so no run is admitted for the dataset while its identity changes (`RealStore::files`).
        let _files = self.files.write().unwrap_or_else(|p| p.into_inner());
        self.unregister(id)
    }

    fn delete_dataset_files(&self, id: &str) -> Result<()> {
        self.delete_own_dataset(id)
    }

    fn list_models(&self) -> Result<Vec<Model>> {
        Ok(self.all_models().into_iter().map(|m| m.model).collect())
    }

    fn get_model(&self, id: &str) -> Result<Model> {
        Ok(self.model(id)?.model)
    }

    fn model_eval(&self, id: &str) -> Result<ModelEval> {
        let _gpu = self.idle.using();
        let m = self.model(id)?;
        let (ds, key) = self.eval_dataset(&m)?;
        let device = engine::resolve(&self.device)?;
        _gpu.on(device);
        self.models.eval(&m, &ds, key.as_deref(), device)
    }

    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult> {
        crate::store::check_export(&spec)?;
        let m = self.model(id)?;
        self.models.export(&m, spec)
    }

    fn model_weights(&self, id: &str, format: WeightsFormat) -> Result<Weights> {
        let m = self.model(id)?;
        let name = weights_file_name(&m.model.name, format);
        match format {
            WeightsFormat::Stored => Weights::open(&m.path, id, name),
            WeightsFormat::F16 | WeightsFormat::Onnx(_) => {
                let (path, file) = self.models.derived_weights(&m, format)?;
                Weights::from_file(file, &path, id, name)
            }
        }
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        Ok(self.runs.list())
    }

    fn create_run(&self, config: TrainConfig) -> Result<Run> {
        self.start_run(config, None)
    }

    fn continue_run(&self, checkpoint: &str, config: TrainConfig) -> Result<Run> {
        if config.weights.is_some() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "checkpoint and weights cannot be combined: the checkpoint holds the weights",
            ));
        }
        let m = self.model(checkpoint)?;
        if engine::read_model_file(&m.path)?.checkpoint().is_none() {
            return Err(Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_weights",
                format!("model '{checkpoint}' ({}) has no training state to continue", m.model.name),
            )
            .hint("name a full checkpoint (a run's last.folo), or start a new run with `weights` set to the model"));
        }
        self.start_run(config, Some(&m.path))
    }

    fn resume_run(&self, id: &str, fields: &serde_json::Map<String, serde_json::Value>) -> Result<Run> {
        // Shared with registration; a deletion of dataset files waits for it (see `RealStore::files`).
        let _files = self.files.read().unwrap_or_else(|p| p.into_inner());
        let (entry, wanted) = self.runs.resume_config(id, fields)?;
        let Some(compute) = wanted.compute.clone() else { return self.runs.resume(&entry, wanted) };
        cloud::check_device(&wanted)?;
        let (connected, offer) = self.cloud.target(&compute)?;
        // The dataset as it is now (a run whose dataset is no longer registered uploads without a fingerprint).
        let ds = entry.dataset();
        let fingerprint = self
            .entry(&ds.id)
            .and_then(|e| self.fresh_index(&e))
            .ok()
            .map(|idx| upload_fingerprint(ds.format, &idx));
        self.runs.resume_cloud(&entry, wanted, run_cloud(&compute, &connected, offer))?;
        self.start_cloud(entry, connected, fingerprint)
    }

    fn get_run(&self, id: &str) -> Result<Run> {
        self.runs.snapshot(id)
    }

    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run> {
        self.runs.move_queued(id, mv)
    }

    fn stop_run(&self, id: &str) -> Result<Run> {
        self.runs.stop(id)
    }

    fn delete_run(&self, id: &str) -> Result<()> {
        let dir = self.runs.delete(id)?;
        for (last, f) in [(false, "best.folo"), (true, "last.folo")] {
            self.predictors.forget(&dir.join("weights").join(f));
            let model = run_model_id(id, last);
            let _ =
                std::fs::remove_file(self.data_dir.join("cache").join("eval").join(format!("{model}.json")));
            self.models.forget_weights(&model);
        }
        Ok(())
    }

    fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics> {
        self.runs.metrics(id, q)
    }

    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples> {
        self.samples(id, epoch)
    }

    fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs> {
        self.runs.logs(id, q)
    }

    fn subscribe(&self, id: &str) -> Result<Subscription> {
        self.runs.subscribe(id)
    }

    fn predict(&self, input: PredictInput) -> Result<PredictResponse> {
        let _gpu = self.idle.using();
        let path = match &input.source {
            PredictSource::Dataset { dataset_id, image_id } => {
                let e = self.entry(dataset_id)?;
                Some(e.current()?.image_path(image_id)?)
            }
            PredictSource::Upload(_) => None,
        };
        let m = self.model(&input.model_id)?;
        let device = engine::resolve(&self.device)?;
        _gpu.on(device);
        // Throughput samples: only a warm predictor's calls with no run queued or training from before the decode
        // (it counts in `total_ms`) to the end, and none taken up or let go meanwhile.
        let activity = self.runs.activity();
        let idle = !self.runs.busy();
        let t0 = Instant::now();
        let decoded = match (&input.source, &path) {
            (PredictSource::Upload(bytes), _) => folo_data::decode(bytes).map_err(|err| {
                // An upload has no path, so the codec's message stands alone.
                let why = match err {
                    folo_data::DataError::Image { message, .. } => message,
                    e => e.to_string(),
                };
                Error::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "image_decode_failed",
                    format!("not an image: {why}"),
                )
                .hint("send a JPEG, PNG, BMP or WebP file")
            })?,
            (_, Some(p)) => folo_data::read_image(p).map_err(|err| data_error(err, &p.to_string_lossy()))?,
            _ => unreachable!("dataset sources have a path"),
        };
        let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
        let loaded = self.predictors.get(&m.path, device, self.precision)?;
        let mut guard = loaded.lock().unwrap();
        // A model's first prediction includes warm-up.
        let warm = std::mem::replace(&mut guard.warm, true);
        let p = &mut guard.predictor;
        let cfg = PredictConfig {
            imgsz: p.model().imgsz(),
            conf: input.conf,
            iou: input.iou,
            max_det: input.max_det as usize,
            rect: true,
            // The model's own head: o2o models predict without NMS.
            head: None,
            ..PredictConfig::default()
        };
        p.set_config(cfg).map_err(engine_error)?;
        let pred = p.predict(&decoded).map_err(engine_error)?;
        let mut timing = pred.timing.to_api();
        timing.preprocess_ms += decode_ms;
        timing.total_ms += decode_ms;
        if warm && idle && !self.runs.busy() && self.runs.activity() == activity {
            let precision = match p.precision() {
                folo_vision::Precision::F16 => InferPrecision::F16,
                _ => InferPrecision::F32,
            };
            self.throughput.record_infer(InferSample {
                key: InferKey {
                    device: engine::device_id(p.device()),
                    arch: m.model.arch,
                    imgsz: p.model().imgsz(),
                    precision,
                    tensor_cores: engine::tensor_cores_on(p.device()),
                },
                total_ms: timing.total_ms,
                inference_ms: timing.inference_ms,
            });
        }
        Ok(PredictResponse {
            model_id: m.model.id.clone(),
            image: ImageSize { width: pred.width, height: pred.height },
            boxes: pred.scored_boxes(p.classes()),
            timing,
            device: p.device().to_string(),
            cold: !warm,
        })
    }

    fn val(&self, req: ValRequest) -> Result<ValReport> {
        let _gpu = self.idle.using();
        let m = self.model(&req.model_id)?;
        let e = self.entry(&req.dataset_id)?;
        // Waits for an index still being built, so a dataset that cannot be read fails here.
        self.fresh_index(&e)?;
        let device = engine::resolve(&self.device)?;
        _gpu.on(device);
        let (ds, p) = (self.dataset_ref(&e), self.precision);
        crate::models::val(&m, &ds, req.split, req.imgsz, req.conf, req.iou, device, p)
    }

    fn labeler(&self) -> Result<&Labeler> {
        Ok(&self.labeler)
    }

    fn benches(&self) -> Result<&Benches> {
        Ok(&self.benches)
    }

    fn check_run(&self, config: &TrainConfig) -> Result<()> {
        self.run_settings(&mut config.clone()).map(|_| ())
    }

    fn dataset_image_path(&self, id: &str, image_id: &str) -> Result<PathBuf> {
        self.entry(id)?.current()?.image_path(image_id)
    }

    fn create_upload(&self, req: CreateUpload) -> Result<UploadCreated> {
        Uploads::check(&req)?;
        if let Some(existing) = req.fingerprint.as_deref().and_then(|f| self.uploaded(req.kind, f)) {
            return Ok(UploadCreated { upload: None, existing: Some(existing) });
        }
        Ok(UploadCreated { upload: Some(self.uploads.create(&req)?), existing: None })
    }

    fn get_upload(&self, id: &str) -> Result<Upload> {
        self.uploads.get(id)
    }

    fn upload_part(&self, id: &str, part: u32, bytes: Bytes) -> Result<Upload> {
        self.uploads.put(id, part, bytes)
    }

    fn complete_upload(&self, id: &str) -> Result<Uploaded> {
        let done = self.uploads.finish(id)?;
        let (made, uploaded) = match done.kind() {
            UploadKind::Dataset => {
                let d = self.adopt_dataset(&done)?;
                (d.id.clone(), Uploaded { dataset: Some(d), model: None })
            }
            UploadKind::Model => {
                let m = self.adopt_model(&done)?;
                (m.id.clone(), Uploaded { dataset: None, model: Some(m) })
            }
        };
        if let Some(fingerprint) = done.fingerprint() {
            self.uploads.remember(done.kind(), fingerprint, &made);
        }
        tracing::info!(upload = %id, made = %made, "upload completed");
        Ok(uploaded)
    }

    fn delete_upload(&self, id: &str) -> Result<()> {
        self.uploads.delete(id)
    }

    fn cloud_providers(&self) -> Result<Vec<CloudProvider>> {
        Ok(self.cloud.providers())
    }

    fn cloud_connections(&self) -> Result<Vec<CloudConnection>> {
        self.cloud.connections()
    }

    fn connect_cloud(&self, req: ConnectCloud) -> Result<CloudConnection> {
        self.cloud.connect(req)
    }

    fn check_cloud(&self, id: &str) -> Result<CloudConnection> {
        self.cloud.check(id)
    }

    fn disconnect_cloud(&self, id: &str) -> Result<()> {
        self.cloud.disconnect(id)
    }

    fn cloud_offers(&self, id: &str, refresh: bool) -> Result<CloudOfferList> {
        self.cloud.offers(id, refresh)
    }

    fn cloud_instances(&self) -> Result<Vec<CloudInstance>> {
        Ok(self.cloud.instances())
    }

    fn terminate_instance(&self, id: &str) -> Result<CloudInstance> {
        self.cloud.terminate(id)
    }

    fn shutdown(&self) {
        self.uploads.shutdown();
        // Cloud runs train on without this server; their threads let go of them (each logs it).
        self.cloud.shutdown();
        // Before the runs: a loop stops the run it follows.
        self.benches.shutdown();
        self.labeler.shutdown();
        self.runs.shutdown(Some(SHUTDOWN_WAIT));
        self.throughput.flush();
    }
}

/// Claims `dir`, the new folder of a sample set ([`folo_data::unfinished::claim`]). Refused when another writer (such
/// as `folo dataset synth` into this path) took it first: `409 output_exists`, the folder left as it is.
pub(super) fn claim_generated(dir: &Path) -> Result<folo_data::unfinished::Claim> {
    use std::io::ErrorKind;
    folo_data::unfinished::claim(dir).map_err(|e| match e {
        folo_data::DataError::Io { source, .. }
            if matches!(source.kind(), ErrorKind::AlreadyExists | ErrorKind::DirectoryNotEmpty) =>
        {
            let e = Error::new(StatusCode::CONFLICT, "output_exists", source.to_string());
            if source.kind() == ErrorKind::AlreadyExists { e.hint(folo_data::unfinished::HINT) } else { e }
        }
        e => data_error(e, &dir.to_string_lossy()),
    })
}
