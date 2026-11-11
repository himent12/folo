//! The data seam: everything the UI knows about FOLO comes through [`Backend`].
//!
//! Two implementations: [`http::HttpBackend`] (`folo serve` over HTTP + SSE) and
//! [`mock::MockBackend`] (in-process fixtures with a live training simulator, for development,
//! deterministic screenshots and running without a server).

pub mod http;
pub mod mock;
mod mock_bench;
pub mod time;
pub mod types;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

pub use types::*;

/// A failed call, shaped like the API's error body.
#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub hint: Option<String>,
}

impl ApiError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.into(), message: message.into(), hint: None }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Image pixels as delivered by a backend: encoded bytes from HTTP, raw RGBA from the mock.
pub enum ImageData {
    /// Encoded bytes (JPEG/PNG), as `folo serve` sends them.
    #[allow(dead_code)]
    Encoded(Vec<u8>),
    Rgba {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageQuery {
    pub split: Option<Split>,
    pub offset: u32,
    pub limit: u32,
    pub class: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PredictInput {
    Dataset { dataset_id: String, image_id: String },
    File { name: String, bytes: std::sync::Arc<Vec<u8>> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictRequest {
    pub model_id: String,
    pub input: PredictInput,
    pub conf: f32,
    pub iou: f32,
    pub max_det: u32,
}

pub trait Backend: Send + Sync + 'static {
    /// Where the data comes from, for the status bar ("Mock data", "127.0.0.1:7878").
    fn describe(&self) -> String;
    /// Seconds since the Unix epoch on the backend's clock (the mock's is fixed for screenshots).
    fn now(&self) -> i64;

    fn health(&self) -> ApiResult<Health>;
    /// The machine, plus what runs can use on it (`capabilities`; absent from older servers).
    fn system(&self) -> ApiResult<SystemResponse>;

    fn datasets(&self) -> ApiResult<Vec<Dataset>>;
    fn dataset(&self, id: &str) -> ApiResult<Dataset>;
    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage>;
    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData>;

    fn models(&self) -> ApiResult<Vec<Model>>;
    fn model_eval(&self, id: &str) -> ApiResult<ModelEval>;
    fn export_model(&self, id: &str, format: &str) -> ApiResult<ExportResult>;

    fn runs(&self) -> ApiResult<Vec<Run>>;
    #[allow(dead_code)]
    fn run(&self, id: &str) -> ApiResult<Run>;
    fn create_run(&self, config: TrainConfig) -> ApiResult<Run>;
    fn stop_run(&self, id: &str) -> ApiResult<Run>;
    /// The `folo train` command that trains a run's config on the server's machine
    /// (`GET /api/runs/{id}/cli`, servers with `run_cli`).
    fn run_cli(&self, _id: &str) -> ApiResult<CliCommand> {
        Err(ApiError::new("not_supported", "This server can't build the run's command"))
    }

    /// How long a config would train (`POST /api/runs/estimate`, servers with `run_estimate`): from
    /// the server's measured throughput when it has one, else its reference rates, with a range and
    /// a confidence.
    fn estimate_run(&self, _config: &TrainConfig) -> ApiResult<RunEstimate> {
        Err(ApiError::new("not_supported", "This server can't estimate a run"))
    }
    /// The server's speed table (`GET /api/system/throughput`, servers with `throughput`): what it
    /// measured on its own devices, and the reference rows for everything it hasn't.
    fn throughput(&self) -> ApiResult<ThroughputTable> {
        Err(ApiError::new("not_supported", "This server doesn't keep a throughput table"))
    }

    /// A quick probe on the server's device (`POST /api/system/benchmark`, servers with
    /// `benchmark`): a few seconds of training and predictions, added to the table as measured.
    /// `409 run_active` while a run is queued or training.
    fn benchmark(&self, _req: &BenchmarkRequest) -> ApiResult<BenchmarkReport> {
        Err(ApiError::new("not_supported", "This server can't measure its speed"))
    }

    /// Moves a waiting run in the training queue (`POST /api/runs/{id}/queue`, servers with the
    /// `queue_control` feature): `409 run_not_queued` once the worker has taken it.
    fn queue_run(&self, _id: &str, _req: QueueRequest) -> ApiResult<Run> {
        Err(ApiError::new("not_supported", "This server can't reorder its queue"))
    }
    fn metrics(&self, id: &str) -> ApiResult<RunMetrics>;
    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<RunSamples>;
    /// Log history of a run. Not in API v1 yet (proposed as `GET /api/runs/{id}/logs`); the SSE
    /// stream only carries new lines.
    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>>;
    /// Live events for a run. The producer stops when the [`Subscription`] is dropped or the run
    /// finishes.
    fn events(&self, id: &str) -> ApiResult<Subscription>;

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse>;

    /// Register a dataset folder (`POST /api/datasets`). The built-in demo has a fixed set.
    fn register_dataset(&self, _req: &RegisterDataset) -> ApiResult<Dataset> {
        Err(ApiError::new("not_supported", "Registering folders needs folo serve")
            .with_hint("The built-in demo has a fixed set of datasets."))
    }
    /// Generate a synthetic shapes dataset (`POST /api/datasets/synthetic`).
    fn synth_dataset(&self, _req: &SyntheticDataset) -> ApiResult<Dataset> {
        Err(ApiError::new("not_supported", "Synthetic datasets need folo serve")
            .with_hint("Run `folo dataset synth`, or connect to folo serve in Settings."))
    }
    /// Resume a stopped or failed run from its last checkpoint (`POST /api/runs` with
    /// `resume_run_id`; the server answers 200 with the same run, queued again).
    fn resume_run(&self, _id: &str) -> ApiResult<Run> {
        Err(ApiError::new("not_supported", "Resuming runs needs folo serve"))
    }
    /// A dataset's issue counts by severity (`GET /api/datasets/{id}/issues`, one issue per page:
    /// `counts` covers them all).
    fn dataset_issues(&self, _id: &str) -> ApiResult<IssuePage> {
        Err(ApiError::new("not_supported", "Issue counts need folo serve"))
    }
    /// The kinds of compute the server can train on, with the fields a connection to each takes
    /// (`GET /api/cloud/providers`, servers with `cloud`).
    fn cloud_providers(&self) -> ApiResult<Vec<CloudProvider>> {
        Err(no_cloud())
    }
    /// The server's cloud connections, oldest first (`GET /api/cloud/connections`).
    fn cloud_connections(&self) -> ApiResult<Vec<CloudConnection>> {
        Err(no_cloud())
    }
    /// Checks credentials with the provider and stores them (`POST /api/cloud/connections`):
    /// `422 cloud_auth_failed` when the provider rejects them, `400 invalid_body` naming a field.
    fn connect_cloud(&self, _req: &ConnectCloud) -> ApiResult<CloudConnection> {
        Err(no_cloud())
    }
    /// Checks a connection's stored credentials again (`POST /api/cloud/connections/{id}/check`); a
    /// failed check is a connection with `status: error`, not an error.
    fn check_cloud(&self, _id: &str) -> ApiResult<CloudConnection> {
        Err(no_cloud())
    }
    /// Forgets a connection's credentials (`DELETE /api/cloud/connections/{id}`): `409
    /// connection_in_use` while it has live instances or active runs.
    fn disconnect_cloud(&self, _id: &str) -> ApiResult<()> {
        Err(no_cloud())
    }
    /// What a connection can run on now, cheapest first (`GET /api/cloud/connections/{id}/offers`);
    /// `refresh` asks the provider again instead of the server's minute-long cache.
    fn cloud_offers(&self, _connection: &str, _refresh: bool) -> ApiResult<CloudOfferList> {
        Err(no_cloud())
    }
    /// The instances the server launched: alive first, then those that ended in the last 7 days
    /// (`GET /api/cloud/instances`).
    fn cloud_instances(&self) -> ApiResult<Vec<CloudInstance>> {
        Err(no_cloud())
    }
    /// Terminates an instance now (`DELETE /api/cloud/instances/{id}`); its run fails with
    /// `cloud_instance_lost`.
    fn terminate_instance(&self, _id: &str) -> ApiResult<CloudInstance> {
        Err(no_cloud())
    }

    // ---------- vision benchmarks (`/api/bench`, servers with `vision_bench`) ----------

    fn bench_catalog(&self) -> ApiResult<BenchCatalog> {
        Err(no_bench())
    }
    fn benches(&self) -> ApiResult<Vec<Bench>> {
        Err(no_bench())
    }
    fn create_bench(&self, _req: &CreateBench) -> ApiResult<Bench> {
        Err(no_bench())
    }
    fn update_bench(&self, _id: &str, _req: &UpdateBench) -> ApiResult<Bench> {
        Err(no_bench())
    }
    fn delete_bench(&self, _id: &str) -> ApiResult<()> {
        Err(no_bench())
    }
    fn design_bench(&self, _id: &str, _req: &DesignBench) -> ApiResult<BenchDesign> {
        Err(no_bench())
    }
    fn bench_images(
        &self,
        _id: &str,
        _offset: u32,
        _limit: u32,
        _condition: Option<(&str, u8)>,
    ) -> ApiResult<BenchImagePage> {
        Err(no_bench())
    }
    fn bench_image(
        &self,
        _id: &str,
        _image_id: &str,
        _condition: Option<(&str, u8)>,
        _thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        Err(no_bench())
    }
    fn bench_entries(&self, _id: &str) -> ApiResult<Vec<BenchEntry>> {
        Err(no_bench())
    }
    fn start_bench_entry(&self, _id: &str, _req: &StartBenchEntry) -> ApiResult<BenchEntry> {
        Err(no_bench())
    }
    fn bench_result(&self, _id: &str, _entry_id: &str) -> ApiResult<BenchResult> {
        Err(no_bench())
    }
    fn bench_entry_images(
        &self,
        _id: &str,
        _entry_id: &str,
        _sort: &str,
        _offset: u32,
        _limit: u32,
    ) -> ApiResult<BenchEntryImagePage> {
        Err(no_bench())
    }
    fn stop_bench_entry(&self, _id: &str, _entry_id: &str) -> ApiResult<BenchEntry> {
        Err(no_bench())
    }
    fn delete_bench_entry(&self, _id: &str, _entry_id: &str) -> ApiResult<()> {
        Err(no_bench())
    }
    fn bench_loops(&self, _id: &str) -> ApiResult<Vec<BenchLoop>> {
        Err(no_bench())
    }
    fn start_bench_loop(&self, _id: &str, _req: &StartBenchLoop) -> ApiResult<BenchLoop> {
        Err(no_bench())
    }
    fn stop_bench_loop(&self, _id: &str, _loop_id: &str) -> ApiResult<BenchLoop> {
        Err(no_bench())
    }
    fn delete_bench_loop(&self, _id: &str, _loop_id: &str) -> ApiResult<()> {
        Err(no_bench())
    }
    fn bench_events(&self, _id: &str) -> ApiResult<BenchSubscription> {
        Err(no_bench())
    }

    // ---------- coding agents & labeling jobs (`/api/label`) ----------

    fn label_agents(&self) -> ApiResult<Vec<LabelAgent>> {
        Err(no_label())
    }
    fn label_jobs(&self) -> ApiResult<Vec<LabelJob>> {
        Err(no_label())
    }

    /// Whether models can be downloaded and run in this process ([`Backend::model_weights`]).
    fn local_inference(&self) -> bool {
        false
    }
    /// A model's f16 weights for local inference (`GET /api/models/{id}/weights?format=f16`),
    /// revalidated against `etag`: `None` when the copy the caller holds is still current.
    fn model_weights(&self, _id: &str, _etag: Option<&str>) -> ApiResult<Option<Weights>> {
        Err(ApiError::new("not_supported", "The built-in demo has no weights to run here")
            .with_hint("Connect to folo serve in Settings to run models in this app."))
    }
}

/// What a server without the `cloud` feature answers to every cloud call.
fn no_cloud() -> ApiError {
    ApiError::new("not_supported", "This server can't train in the cloud")
}

/// What a server without `vision_bench` answers to every vision-benchmark call.
fn no_bench() -> ApiError {
    ApiError::new("not_supported", "Vision benchmarks need a real folo serve")
        .with_hint("Start folo serve without --mock, or switch to the built-in demo in Settings.")
}

/// What a server without `agent_labeling` answers to every labeling call.
fn no_label() -> ApiError {
    ApiError::new("not_supported", "This server does not run coding agents")
}

/// A `.folo` file and its ETag.
pub struct Weights {
    pub bytes: Vec<u8>,
    pub etag: String,
}

/// A live event stream. Dropping it tells the producer to stop: the mock's worker at its next tick,
/// the HTTP reader at the next line it receives (at the latest the server's 15 s keep-alive).
pub struct EventStream<E> {
    rx: Receiver<E>,
    cancel: Arc<AtomicBool>,
}

pub type Subscription = EventStream<RunEvent>;
pub type BenchSubscription = EventStream<BenchEvent>;

impl<E> EventStream<E> {
    pub fn new(rx: Receiver<E>) -> (Self, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        (Self { rx, cancel: cancel.clone() }, cancel)
    }
}

impl<E> std::ops::Deref for EventStream<E> {
    type Target = Receiver<E>;
    fn deref(&self) -> &Receiver<E> {
        &self.rx
    }
}

impl<E> Drop for EventStream<E> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
