//! The [`Store`] trait: everything the HTTP handlers need, independent of where the data comes from.
//!
//! Two implementations: [`crate::mock::MockStore`] (deterministic synthetic data, `folo serve --mock`) and
//! [`crate::real::RealStore`] (datasets on disk; training and inference return `not_implemented` until the engine
//! is wired in). Methods are synchronous; handlers run the ones that may touch the disk or burn CPU on tokio's
//! blocking pool.

use folo_api::{
    Dataset, ExportFormat, ExportResult, GpuInfo, Model, PredictResponse, RegisterDataset, Run, RunEvent,
    RunMetrics, RunSamples, Split, StatusEvent, SyntheticDataset, TrainConfig,
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
    /// An image (original bytes, or a JPEG thumbnail whose long side is `thumb`).
    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData>;
    /// Unregisters a dataset (never deletes files).
    fn delete_dataset(&self, id: &str) -> Result<()>;

    /// All models, newest first.
    fn list_models(&self) -> Result<Vec<Model>>;
    /// One model.
    fn get_model(&self, id: &str) -> Result<Model>;
    /// Exports a model.
    fn export_model(&self, id: &str, format: ExportFormat) -> Result<ExportResult>;

    /// All runs, newest first.
    fn list_runs(&self) -> Result<Vec<Run>>;
    /// Starts a run.
    fn create_run(&self, config: TrainConfig) -> Result<Run>;
    /// One run.
    fn get_run(&self, id: &str) -> Result<Run>;
    /// Requests a stop.
    fn stop_run(&self, id: &str) -> Result<Run>;
    /// Deletes a run that is not active.
    fn delete_run(&self, id: &str) -> Result<()>;
    /// Full metric history; `every > 1` keeps every `every`-th step (and always the last one).
    fn run_metrics(&self, id: &str, every: u64) -> Result<RunMetrics>;
    /// Validation samples of an epoch (latest finished epoch when `None`).
    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples>;
    /// Subscribes to a run's live events. Must be called inside a tokio runtime.
    fn subscribe(&self, id: &str) -> Result<Subscription>;

    /// Runs inference on one image.
    fn predict(&self, input: PredictInput) -> Result<PredictResponse>;
}
