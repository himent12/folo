//! The [`Store`] trait: everything the HTTP handlers need, independent of where the data comes from.
//!
//! Two implementations: [`crate::mock::MockStore`] (deterministic synthetic data, `folo serve --mock`) and
//! [`crate::real::RealStore`] (datasets on disk, training runs, models and inference on the FOLO engine). Methods
//! are synchronous; handlers run the ones that may touch the disk or burn CPU on tokio's blocking pool.

use std::path::Path;

use axum::body::Bytes;
use folo_api::{
    BenchmarkReport, BenchmarkRequest, Capabilities, CliCommand, CloudConnection, CloudInstance,
    CloudOfferList, CloudProvider, ConnectCloud, CreateUpload, Dataset, DatasetFormat, ExportResult,
    ExportSpec, GpuInfo, GpuPool, LabelDatasetRequest, LogLine, Model, ModelEval, PredictResponse,
    RegisterDataset, Run, RunEstimate, RunEvent, RunLogs, RunMetrics, RunSamples, Split, StatusEvent,
    SyntheticDataset, ThroughputTable, TrainConfig, Upload, UploadCreated, Uploaded, ValReport, ValRequest,
};
use tokio::sync::broadcast;

use crate::error::Result;

/// Filter and page of `GET /api/datasets/{id}/images`.
#[derive(Clone, Debug, Default)]
pub struct ImageQuery {
    /// Only this split (all splits when `None`).
    pub split: Option<Split>,
    /// Skip this many matching images.
    pub offset: u64,
    /// Return at most this many.
    pub limit: u64,
    /// Only images with at least one box of this class.
    pub class: Option<u32>,
}

/// Filter and page of `GET /api/datasets/{id}/issues`.
#[derive(Clone, Debug, Default)]
pub struct IssueQuery {
    /// Only issues of this kind (one of [`folo_api::ISSUE_KINDS`]).
    pub kind: Option<String>,
    /// Only issues of this severity.
    pub severity: Option<folo_api::IssueSeverity>,
    /// Skip this many matching issues.
    pub offset: u64,
    /// Return at most this many.
    pub limit: u64,
}

impl IssueQuery {
    /// The page of `issues` this query selects, with the counts of all of them.
    pub fn page<'a>(
        &self,
        issues: impl Iterator<Item = &'a folo_api::DatasetIssue> + Clone,
    ) -> folo_api::IssuePage {
        let mut total = 0u64;
        let mut page = Vec::new();
        let keep = |i: &&folo_api::DatasetIssue| {
            self.kind.as_ref().is_none_or(|k| *k == i.kind) && self.severity.is_none_or(|s| s == i.severity)
        };
        for i in issues.clone().filter(keep) {
            if total >= self.offset && (page.len() as u64) < self.limit {
                page.push(i.clone());
            }
            total += 1;
        }
        folo_api::IssuePage { total, issues: page, counts: folo_api::IssueCounts::of(issues) }
    }
}

/// Selection of `GET /api/runs/{id}/metrics`: the cursors first (feature `run_cursors`), then the thinning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricsQuery {
    /// Keep every `every`-th step (and always the latest one); 1 keeps them all.
    pub every: u64,
    /// Only steps with `step > after_step`.
    pub after_step: Option<u64>,
    /// Only epochs with `epoch > after_epoch`.
    pub after_epoch: Option<u32>,
}

impl Default for MetricsQuery {
    fn default() -> Self {
        MetricsQuery { every: 1, after_step: None, after_epoch: None }
    }
}

impl MetricsQuery {
    /// The whole history, thinned to every `every`-th step.
    pub fn every(every: u64) -> MetricsQuery {
        MetricsQuery { every, ..MetricsQuery::default() }
    }

    /// What this query selects of a run's full history `m`.
    pub fn select(&self, mut m: RunMetrics) -> RunMetrics {
        if let Some(after) = self.after_step {
            m.steps.retain(|s| s.step > after);
        }
        if let Some(after) = self.after_epoch {
            m.epochs.retain(|e| e.epoch > after);
        }
        if self.every > 1 {
            let latest = m.steps.last().map(|s| s.step);
            m.steps.retain(|s| s.step % self.every == 0 || Some(s.step) == latest);
        }
        m
    }
}

/// Selection of `GET /api/runs/{id}/logs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogsQuery {
    /// Most lines answered.
    pub limit: usize,
    /// The lines after the first `after` of the log (the oldest `limit` of them) instead of the last `limit` lines
    /// (feature `run_cursors`).
    pub after: Option<u64>,
}

impl LogsQuery {
    /// The last `limit` lines.
    pub fn last(limit: usize) -> LogsQuery {
        LogsQuery { limit, after: None }
    }

    /// What this query selects of a run's whole log `lines` (oldest first), with the log's length.
    pub fn select(&self, lines: Vec<LogLine>) -> RunLogs {
        let total = lines.len() as u64;
        let skip = match self.after {
            Some(after) => usize::try_from(after).unwrap_or(usize::MAX),
            None => lines.len().saturating_sub(self.limit),
        };
        RunLogs { lines: lines.into_iter().skip(skip).take(self.limit).collect(), total }
    }
}

/// Encoded image bytes with their MIME type.
#[derive(Clone, Debug)]
pub struct ImageData {
    /// e.g. `image/jpeg`.
    pub content_type: &'static str,
    /// File contents.
    pub bytes: Vec<u8>,
    /// True when the bytes never change for this URL (mock images), so clients may cache them forever.
    pub immutable: bool,
}

/// Which weights file `GET /api/models/{id}/weights` serves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeightsFormat {
    /// The model's `.folo` file as stored (a run model is its full training checkpoint).
    #[default]
    Stored,
    /// The `folo-f16` export (fused EMA weights in F16), created on first request and cached.
    F16,
    /// The ONNX export with these switches (`POST /api/models/{id}/export` with `format: onnx`), created on first
    /// request and cached.
    Onnx(folo_api::OnnxSwitches),
}

/// Where the bytes of a [`Weights`] come from.
#[derive(Debug)]
pub enum WeightsBody {
    /// An open file, streamed from its start (at most [`Weights::len`] bytes are sent).
    File(std::fs::File),
    /// Bytes in memory (the mock store's generated files).
    Bytes(axum::body::Bytes),
}

/// A model's weights file with its HTTP validators (`GET /api/models/{id}/weights`).
#[derive(Debug)]
pub struct Weights {
    /// The bytes.
    pub body: WeightsBody,
    /// Their length (`Content-Length`).
    pub len: u64,
    /// Last modification (`Last-Modified`).
    pub modified: std::time::SystemTime,
    /// Strong ETag, quoted (`"..."`): it changes whenever the bytes do.
    pub etag: String,
    /// Suggested file name (`Content-Disposition`), e.g. `shapes-n-best.folo`.
    pub file_name: String,
}

impl Weights {
    /// Opens `path` for streaming ([`Weights::from_file`]). A missing file is `model_not_found` with `id`.
    pub fn open(path: &std::path::Path, id: &str, file_name: String) -> Result<Weights> {
        let file = std::fs::File::open(path).map_err(|e| weights_io(e, path, id))?;
        Weights::from_file(file, path, id, file_name)
    }

    /// Weights streamed from `file` (opened from `path`). The length, modification time and ETag come from the
    /// open handle, so they describe the bytes sent even when the file is replaced or removed meanwhile. The ETag
    /// is the size and the modification time in nanoseconds: every writer of `.folo` files replaces them whole,
    /// which gives the new file a new modification time (a replacement that keeps both the size and the exact
    /// modification time, e.g. `touch -r`, is not noticed).
    pub fn from_file(
        file: std::fs::File,
        path: &std::path::Path,
        id: &str,
        file_name: String,
    ) -> Result<Weights> {
        let meta = file.metadata().map_err(|e| weights_io(e, path, id))?;
        let modified = meta.modified().map_err(|e| weights_io(e, path, id))?;
        let nanos = modified.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let len = meta.len();
        Ok(Weights {
            body: WeightsBody::File(file),
            len,
            modified,
            etag: format!("\"{len:x}-{nanos:x}\""),
            file_name,
        })
    }
}

/// `400 invalid_body` for ONNX switches on a `folo-f16` export (every [`Store::export_model`] checks this).
pub fn check_export(spec: &ExportSpec) -> Result<()> {
    if spec.format != folo_api::ExportFormat::Onnx && spec.onnx.any() {
        return Err(crate::error::Error::new(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_body",
            "decode, dynamic and half only apply to format onnx",
        )
        .hint("folo-f16 is already the half-precision .folo export; leave the switches out"));
    }
    Ok(())
}

/// An IO error on the weights file of model `id`: `model_not_found` when the file is gone, else `internal`.
pub fn weights_io(e: std::io::Error, path: &std::path::Path, id: &str) -> crate::error::Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        crate::error::Error::not_found("model", id)
    } else {
        crate::error::Error::internal(format!("{}: {e}", path.display()))
    }
}

/// The file name a model's weights are offered under: its name reduced to `A-Z a-z 0-9 . _ -` (anything else
/// becomes `_`, outer dots are dropped, at most 200 characters, so it is a file name on every OS), then `.folo`,
/// `-f16.folo` for the f16 export, or, as `POST /api/models/{id}/export` names its files, `.onnx` (`-f16.onnx` with
/// `half`).
pub fn weights_file_name(model_name: &str, format: WeightsFormat) -> String {
    let stem: String = model_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .take(200)
        .collect();
    // No leading dot (a hidden file) and no trailing one (Windows drops it).
    let stem = match stem.trim_matches('.') {
        "" => "model",
        s => s,
    };
    match format {
        WeightsFormat::Stored => format!("{stem}.folo"),
        WeightsFormat::F16 => format!("{stem}-f16.folo"),
        WeightsFormat::Onnx(s) if s.half => format!("{stem}-f16.onnx"),
        WeightsFormat::Onnx(_) => format!("{stem}.onnx"),
    }
}

/// Input of `POST /api/predict`.
#[derive(Clone, Debug)]
pub struct PredictInput {
    /// Model to run.
    pub model_id: String,
    /// The image.
    pub source: PredictSource,
    /// Confidence threshold.
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Maximum detections.
    pub max_det: u32,
}

/// Where the image of a prediction comes from.
#[derive(Clone, Debug)]
pub enum PredictSource {
    /// Uploaded file bytes.
    Upload(Vec<u8>),
    /// An image of a registered dataset.
    Dataset {
        /// Dataset id.
        dataset_id: String,
        /// Image id.
        image_id: String,
    },
}

/// A validated `POST /api/runs/{id}/queue` body ([`QueueMove::parse`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueMove {
    /// To the front.
    Top,
    /// One place forward.
    Up,
    /// One place back.
    Down,
    /// To the back.
    Bottom,
    /// To the front, starting it at once when nothing trains (never preempting).
    Start,
    /// To this 1-based place (at least 1; past the end means the end).
    To(u32),
}

impl QueueMove {
    /// The move a request asks for: exactly one of `action` and `position`, `position` at least 1; anything
    /// else is `400 invalid_body`.
    pub fn parse(req: folo_api::QueueRequest) -> Result<QueueMove> {
        use folo_api::QueueAction as A;
        let bad = |m: &str| {
            crate::error::Error::new(axum::http::StatusCode::BAD_REQUEST, "invalid_body", m)
                .hint(r#"send {"action": "top" | "up" | "down" | "bottom" | "start"} or {"position": 1}"#)
        };
        match (req.action, req.position) {
            (Some(_), Some(_)) => Err(bad("give either action or position, not both")),
            (None, None) => Err(bad("give action or position")),
            (None, Some(0)) => Err(bad("position is 1-based (1 starts next)")),
            (None, Some(p)) => Ok(QueueMove::To(p)),
            (Some(a), None) => Ok(match a {
                A::Top => QueueMove::Top,
                A::Up => QueueMove::Up,
                A::Down => QueueMove::Down,
                A::Bottom => QueueMove::Bottom,
                A::Start => QueueMove::Start,
            }),
        }
    }

    /// The 0-based index a run at index `from` of a queue of `len` (> `from`) moves to.
    pub fn target(self, from: usize, len: usize) -> usize {
        let last = len.saturating_sub(1);
        match self {
            QueueMove::Top | QueueMove::Start => 0,
            QueueMove::Up => from.saturating_sub(1),
            QueueMove::Down => (from + 1).min(last),
            QueueMove::Bottom => last,
            QueueMove::To(p) => (p.max(1) as usize - 1).min(last),
        }
    }
}

/// `501 not_implemented` of the upload methods of a store without uploads.
fn not_uploads() -> crate::error::Error {
    crate::error::Error::not_implemented("uploading")
        .hint("uploads need a real server (`folo serve` without --mock)")
}

/// A live subscription to a run's events.
pub struct Subscription {
    /// The run's state right now (sent first).
    pub initial: StatusEvent,
    /// Subsequent events. Subscribe before computing `initial`, so nothing between the two is lost.
    pub events: broadcast::Receiver<RunEvent>,
}

/// The data behind the API. See the module docs.
pub trait Store: Send + Sync + 'static {
    /// True for the mock store.
    fn is_mock(&self) -> bool;
    /// GPUs to report in `/api/system`.
    fn gpus(&self) -> Vec<GpuInfo>;
    /// What runs with `device: auto` can use, for `/api/system`'s `capabilities` (`None`: not reported). May block
    /// (the real store creates the GPU context on first use).
    fn capabilities(&self) -> Option<Capabilities> {
        None
    }
    /// FOLO's GPU buffer pool, for `/api/system`'s `gpu_pool` (`None`: no GPU in use, or the mock store).
    fn gpu_pool(&self) -> Option<GpuPool> {
        None
    }
    /// The data dir to report in `/api/system` (`None`: the mock store).
    fn data_dir(&self) -> Option<&Path> {
        None
    }
    /// The optional features of `/api/system`'s `features` that depend on the store (`folo_api::FEATURE_*`:
    /// uploads, run cursors, checkpoints); the ones every store has are listed by the server.
    fn features(&self) -> &'static [&'static str] {
        &[]
    }
    /// The throughput table (`GET /api/system/throughput`): measured rows, then the reference rows of this server's
    /// devices. May block (the real store resolves `device: auto`). Stores without one answer `501
    /// not_implemented` (the default).
    fn throughput(&self) -> Result<ThroughputTable> {
        Err(crate::error::Error::not_implemented("the throughput table"))
    }
    /// How long a run of `config` would take (`POST /api/runs/estimate`), checked like `POST /api/runs`.
    fn estimate_run(&self, config: TrainConfig) -> Result<RunEstimate> {
        let _ = config;
        Err(crate::error::Error::not_implemented("run estimates"))
    }
    /// A few-second probe of training and inference speed (`POST /api/system/benchmark`); `409 run_active` while
    /// a run is queued or training.
    fn benchmark(&self, req: BenchmarkRequest) -> Result<BenchmarkReport> {
        let _ = req;
        Err(crate::error::Error::not_implemented("benchmark probes"))
    }

    /// All datasets (without stats), oldest first.
    fn list_datasets(&self) -> Result<Vec<Dataset>>;
    /// Registers a folder on disk.
    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset>;
    /// Generates a synthetic dataset.
    fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset>;
    /// One dataset with stats.
    fn get_dataset(&self, id: &str) -> Result<Dataset>;
    /// A page of images.
    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<folo_api::ImagePage>;
    /// A page of the dataset's issues (the `folo dataset check` findings, computed with the stats).
    fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<folo_api::IssuePage>;
    /// An image (original bytes, or a JPEG thumbnail whose long side is `thumb`).
    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData>;
    /// Unregisters a dataset (never deletes files).
    fn delete_dataset(&self, id: &str) -> Result<()>;

    /// All models, newest first.
    fn list_models(&self) -> Result<Vec<Model>>;
    /// One model.
    fn get_model(&self, id: &str) -> Result<Model>;
    /// Validation results of a model.
    fn model_eval(&self, id: &str) -> Result<ModelEval>;
    /// Exports a model.
    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult>;
    /// A model's weights file (`GET /api/models/{id}/weights`). Stores without files answer
    /// `501 not_implemented` (the default).
    fn model_weights(&self, id: &str, format: WeightsFormat) -> Result<Weights> {
        let _ = format;
        self.get_model(id)?;
        Err(crate::error::Error::not_implemented("downloading weights"))
    }

    /// All runs, newest first.
    fn list_runs(&self) -> Result<Vec<Run>>;
    /// Starts a run.
    fn create_run(&self, config: TrainConfig) -> Result<Run>;
    /// Resumes a `stopped` or `failed` run (same id) from its last checkpoint: `POST /api/runs` with
    /// `resume_run_id`. `fields` are the body's other fields, which must match the run's config (`device` may
    /// change). Stores without resumable runs answer `501 not_implemented`.
    fn resume_run(&self, id: &str, fields: &serde_json::Map<String, serde_json::Value>) -> Result<Run> {
        let _ = fields;
        self.get_run(id)?;
        Err(crate::error::Error::not_implemented("resuming runs"))
    }
    /// Starts a new run that continues the training state stored in model `checkpoint`'s file (`POST /api/runs`
    /// with `checkpoint`): `404 model_not_found`, `422 invalid_weights` when the file is not a full checkpoint.
    /// `config` is checked as for [`Store::create_run`]. Stores without checkpoints answer `501 not_implemented`.
    fn continue_run(&self, checkpoint: &str, config: TrainConfig) -> Result<Run> {
        let _ = (checkpoint, config);
        Err(crate::error::Error::not_implemented("continuing a checkpoint"))
    }
    /// One run.
    fn get_run(&self, id: &str) -> Result<Run>;
    /// `GET /api/runs/{id}/cli`: the `folo train` command of the run's config on this machine
    /// ([`crate::cli::train_command`]). Stores without one answer `501 not_implemented` (the default).
    fn run_cli(&self, id: &str) -> Result<CliCommand> {
        let _ = id;
        Err(crate::error::Error::not_implemented("the CLI command of a run"))
    }
    /// Requests a stop.
    fn stop_run(&self, id: &str) -> Result<Run>;
    /// Moves a run that waits in the training queue (`POST /api/runs/{id}/queue`): `409 run_not_queued` when it
    /// does not wait there. Stores without a queue answer `501 not_implemented` (the default).
    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run> {
        let _ = mv;
        self.get_run(id)?;
        Err(crate::error::Error::not_implemented("moving queued runs"))
    }
    /// Deletes a run that is not active.
    fn delete_run(&self, id: &str) -> Result<()>;
    /// The run's metric history, as `q` selects it.
    fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics>;
    /// Validation samples of an epoch (latest finished epoch when `None`).
    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples>;
    /// The run's log lines `q` selects, oldest first, with the length of the whole log.
    fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs>;
    /// Subscribes to a run's live events. Must be called inside a tokio runtime.
    fn subscribe(&self, id: &str) -> Result<Subscription>;

    /// Runs inference on one image.
    fn predict(&self, input: PredictInput) -> Result<PredictResponse>;

    /// Evaluates a model on a dataset split (`POST /api/val`, the MCP `val` tool). Stores without an evaluator
    /// answer `501 not_implemented` (the default).
    fn val(&self, req: ValRequest) -> Result<ValReport> {
        let _ = req;
        Err(crate::error::Error::not_implemented("validation"))
    }

    /// Starts an upload session (`POST /api/uploads`), or answers with what an earlier upload with the same
    /// `fingerprint` became. Stores without uploads answer `501 not_implemented` (the default, also for the other
    /// upload methods).
    fn create_upload(&self, req: CreateUpload) -> Result<UploadCreated> {
        let _ = req;
        Err(not_uploads())
    }
    /// An upload session (`GET /api/uploads/{id}`).
    fn get_upload(&self, id: &str) -> Result<Upload> {
        let _ = id;
        Err(not_uploads())
    }
    /// Part `part` of an upload's stream (`PUT /api/uploads/{id}/parts/{n}`).
    fn upload_part(&self, id: &str, part: u32, bytes: Bytes) -> Result<Upload> {
        let _ = (id, part, bytes);
        Err(not_uploads())
    }
    /// Ends an upload and registers what it holds (`POST /api/uploads/{id}/complete`).
    fn complete_upload(&self, id: &str) -> Result<Uploaded> {
        let _ = id;
        Err(not_uploads())
    }
    /// Drops an upload session and what it received (`DELETE /api/uploads/{id}`).
    fn delete_upload(&self, id: &str) -> Result<()> {
        let _ = id;
        Err(not_uploads())
    }

    /// The cloud providers this server knows (`GET /api/cloud/providers`). Stores without cloud compute answer `501
    /// not_implemented` (the default, also for the other cloud methods).
    fn cloud_providers(&self) -> Result<Vec<CloudProvider>> {
        Err(not_cloud())
    }
    /// The cloud connections, oldest first (`GET /api/cloud/connections`).
    fn cloud_connections(&self) -> Result<Vec<CloudConnection>> {
        Err(not_cloud())
    }
    /// Makes a cloud connection, its values checked with the provider first (`POST /api/cloud/connections`).
    fn connect_cloud(&self, req: ConnectCloud) -> Result<CloudConnection> {
        let _ = req;
        Err(not_cloud())
    }
    /// Checks a connection's credentials again (`POST /api/cloud/connections/{id}/check`).
    fn check_cloud(&self, id: &str) -> Result<CloudConnection> {
        let _ = id;
        Err(not_cloud())
    }
    /// Forgets a connection (`DELETE /api/cloud/connections/{id}`): `409 connection_in_use` while it has live
    /// instances or active runs.
    fn disconnect_cloud(&self, id: &str) -> Result<()> {
        let _ = id;
        Err(not_cloud())
    }
    /// What a connection can run on (`GET /api/cloud/connections/{id}/offers`), cached unless `refresh`.
    fn cloud_offers(&self, id: &str, refresh: bool) -> Result<CloudOfferList> {
        let _ = (id, refresh);
        Err(not_cloud())
    }
    /// The instances this server launched (`GET /api/cloud/instances`).
    fn cloud_instances(&self) -> Result<Vec<CloudInstance>> {
        Err(not_cloud())
    }
    /// Terminates an instance now (`DELETE /api/cloud/instances/{id}`).
    fn terminate_instance(&self, id: &str) -> Result<CloudInstance> {
        let _ = id;
        Err(not_cloud())
    }

    /// The agent-labeling service behind `/api/label` (docs/API.md, Agent labeling). Stores without one answer
    /// `501 not_implemented` (the default).
    fn labeler(&self) -> Result<&crate::label::Labeler> {
        Err(crate::error::Error::not_implemented("agent labeling"))
    }
    /// `POST /api/label/jobs/{id}/dataset`: writes the job's labeled images as a YOLO dataset under the data dir
    /// and registers it. A folder that cannot be registered is removed again.
    fn label_dataset(&self, id: &str, req: LabelDatasetRequest) -> Result<Dataset> {
        let labeler = self.labeler()?;
        let written = labeler.write_dataset(id, &req)?;
        let registered = self.register_dataset(RegisterDataset {
            name: Some(written.name.clone()),
            path: written.path.to_string_lossy().into_owned(),
            format: DatasetFormat::Yolo,
        });
        let dataset = registered.inspect_err(|_| {
            let _ = std::fs::remove_dir_all(&written.path);
        })?;
        labeler.set_dataset(written, &dataset);
        Ok(dataset)
    }

    /// The vision-benchmarks service behind `/api/bench` (docs/API.md, Vision benchmarks). Stores without one
    /// answer `501 not_implemented` (the default).
    fn benches(&self) -> Result<&crate::bench::Benches> {
        Err(crate::error::Error::not_implemented("vision benchmarks")
            .hint("vision benchmarks need a real server (`folo serve` without --mock)"))
    }
    /// Checks a run's settings as `POST /api/runs` would, all but its dataset (a self-improvement loop checks its
    /// students' runs before their datasets exist). The default takes them as they are.
    fn check_run(&self, config: &TrainConfig) -> Result<()> {
        let _ = config;
        Ok(())
    }
    /// The file of image `image_id` of dataset `id` (what a benchmark copies, and a loop's pool reads). Stores
    /// without files answer `501 not_implemented` (the default).
    fn dataset_image_path(&self, id: &str, image_id: &str) -> Result<std::path::PathBuf> {
        let _ = (id, image_id);
        Err(crate::error::Error::not_implemented("reading dataset files"))
    }

    /// Called once when the server shuts down: stop background work (the real store stops the active run at a
    /// batch boundary, keeping its last checkpoint, drops upload sessions and leaves cloud runs to their workers).
    /// Blocking; the default does nothing.
    fn shutdown(&self) {}
}

/// `501 not_implemented` of the cloud methods of a store without cloud compute.
fn not_cloud() -> crate::error::Error {
    crate::error::Error::not_implemented("cloud compute")
        .hint("cloud compute needs a server with the `cloud` feature (GET /api/system lists its features)")
}
