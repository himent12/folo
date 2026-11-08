//! The data seam: everything the UI knows about FOLO comes through [`Backend`].
//!
//! Two implementations: [`http::HttpBackend`] (`folo serve` over HTTP + SSE) and
//! [`mock::MockBackend`] (in-process fixtures with a live training simulator, for development,
//! deterministic screenshots and running without a server).

pub mod http;
pub mod mock;
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
    fn system(&self) -> ApiResult<SystemInfo>;

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
}

/// A live event stream. Dropping it tells the producer to stop: the mock's worker at its next tick,
/// the HTTP reader at the next line it receives (at the latest the server's 15 s keep-alive).
pub struct Subscription {
    rx: Receiver<RunEvent>,
    cancel: Arc<AtomicBool>,
}

impl Subscription {
    pub fn new(rx: Receiver<RunEvent>) -> (Self, Arc<AtomicBool>) {
        let cancel = Arc::new(AtomicBool::new(false));
        (Self { rx, cancel: cancel.clone() }, cancel)
    }
}

impl std::ops::Deref for Subscription {
    type Target = Receiver<RunEvent>;
    fn deref(&self) -> &Receiver<RunEvent> {
        &self.rx
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
