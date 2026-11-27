//! The [`Store`] trait: everything the HTTP handlers need, independent of where the data comes from.
//!
//! Two implementations: [`crate::mock::MockStore`] (deterministic synthetic data, `folo serve --mock`) and
//! [`crate::real::RealStore`] (datasets on disk, training runs, models and inference on the FOLO engine). Methods
//! are synchronous; handlers run the ones that may touch the disk or burn CPU on tokio's blocking pool. A third
//! lives in folo-mcp: `folo_mcp::proxy::HttpStore`, which forwards the request / response calls to a running
//! `folo serve` (its `subscribe` answers `not_implemented`).

use std::path::Path;
use std::sync::Arc;

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

#[path = "store_diag.rs"]
mod diag;
pub use diag::{
    AutoBatchOptions, DEFAULT_VAL_DIAG_BG_IOU, DEFAULT_VAL_DIAG_CONF, DEFAULT_VAL_DIAG_TP_IOU,
    DEFAULT_VAL_DIAG_WORST, DynamicAxes, FEATURE_BENCH_GENERATED_SOURCE, FEATURE_EXPORT_DYNAMIC_AXES,
    FEATURE_TRAIN_AUTO_BATCH, FEATURE_VAL_DIAGNOSTICS, ImageDiagnostics, MAX_VAL_DIAG_WORST, ValDiagConfig,
    ValDiagnostics, ValDiagnosticsSummary, check_auto_batch_options, check_export_axes,
    check_val_diag_config,
};
pub use folo_vision::tta::DEFAULT_TTA_SCALES;
pub use folo_vision::{PredictOptions, TileConfig, TileMergeMetric, TtaConfig};

/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/predict` honours optional
/// `tile` (SAHI) and `tta` (Test-Time Augmentation) fields.
pub const FEATURE_PREDICT_OPTIONS: &str = "predict_options";

/// Server capability flag advertised on `/api/health` and `/api/system` when `/api/bench` honours client-chosen
/// `?entry_id=<id>` on `POST .../entries` and conditional `?only_if_open=true` on `POST .../entries/{id}/stop`.
pub const FEATURE_BENCH_UPLOAD_SAFE: &str = "bench_upload_safe";

/// Maximum SAHI tiles permitted in a single server/MCP `predict` call (prevents resource exhaustion on tiny tiles).
pub const MAX_PREDICT_TILES: usize = 256;
/// Maximum cropped tiles predicted in a single batch in server/MCP `predict`.
pub const MAX_PREDICT_TILE_BATCH: usize = 32;
/// Maximum peak RGB crop bytes allocated per tiled batch chunk in server/MCP `predict` (256 MiB).
pub const MAX_PREDICT_CROP_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum TTA views permitted in a single server/MCP `predict` call (`unique_scales * (hflip ? 2 : 1)`).
pub const MAX_PREDICT_TTA_VIEWS: usize = 16;

/// Validates `options` independent of a specific model `imgsz` or image dimensions (`400 invalid_body` on invalid
/// ranges, exceeding per-request batch/scale caps, or combining `tile` with `tta`).
pub fn check_predict_options(options: &PredictOptions) -> Result<()> {
    fn bad(field: &str, msg: impl std::fmt::Display) -> crate::error::Error {
        crate::error::Error::new(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_body",
            format!("field `{field}`: {msg}"),
        )
    }
    if options.tile.is_some() && options.tta.is_some() {
        return Err(bad("tile", "cannot be combined with tta"));
    }
    if let Some(tile) = &options.tile {
        tile.validate().map_err(|e| bad("tile", e))?;
        if tile.tile_batch > MAX_PREDICT_TILE_BATCH {
            return Err(bad(
                "tile",
                format!("tile_batch must be at most {MAX_PREDICT_TILE_BATCH}, got {}", tile.tile_batch),
            ));
        }
    }
    if let Some(tta) = &options.tta {
        tta.validate().map_err(|e| bad("tta", e))?;
        if tta.scales.len() > MAX_PREDICT_TTA_VIEWS {
            return Err(bad(
                "tta",
                format!("at most {MAX_PREDICT_TTA_VIEWS} scales are allowed, got {}", tta.scales.len()),
            ));
        }
    }
    Ok(())
}

/// Tile count along one image axis matching `folo_vision::tile::compute_tiles` without allocating tile rects.
fn axis_tile_count(img_len: u32, tile_len: u32, overlap: f32) -> usize {
    let tw = tile_len.min(img_len);
    if tw == img_len || tw == 0 {
        return 1;
    }
    let overlap_px = ((tw as f64) * (overlap as f64)).round() as u32;
    let stride = tw.saturating_sub(overlap_px.min(tw - 1)).max(1);
    let span = img_len - tw;
    (span as usize).div_ceil(stride as usize) + 1
}

/// Validates `options` against `model_imgsz` and (when known) image `(width, height)`, enforcing server-side SAHI
/// tile-count and TTA view-count budgets before locking the predictor or allocating tile grids.
pub fn check_predict_workload(
    options: &PredictOptions,
    model_imgsz: u32,
    image_size: Option<(u32, u32)>,
) -> Result<()> {
    let bad = |msg: String, hint: &'static str| {
        crate::error::Error::new(axum::http::StatusCode::BAD_REQUEST, "invalid_config", msg).hint(hint)
    };
    options.validate(model_imgsz).map_err(|e| {
        bad(e.to_string(), "check `tile` / `tta` settings against the model's input size (32..=8192)")
    })?;
    if let (Some(tile), Some((w, h))) = (&options.tile, image_size) {
        let nx = axis_tile_count(w, tile.tile_width, tile.overlap_x);
        let ny = axis_tile_count(h, tile.tile_height, tile.overlap_y);
        let tiles = nx.saturating_mul(ny);
        if tiles > MAX_PREDICT_TILES {
            return Err(bad(
                format!(
                    "tiled prediction on {w}x{h} with {}x{} tiles produces {tiles} tiles, exceeding the server limit of {MAX_PREDICT_TILES}",
                    tile.tile_width, tile.tile_height
                ),
                "use a larger tile size or lower overlap",
            ));
        }
        let crop_w = tile.tile_width.min(w) as u64;
        let crop_h = tile.tile_height.min(h) as u64;
        let chunk = tiles.min(tile.tile_batch) as u64;
        let crop_bytes = chunk.saturating_mul(crop_w).saturating_mul(crop_h).saturating_mul(3);
        if crop_bytes > MAX_PREDICT_CROP_BYTES {
            return Err(bad(
                format!(
                    "tiled prediction on {w}x{h} with {}x{} tiles and tile_batch {} needs {:.1} MiB of crop buffers per batch chunk, exceeding the server limit of {} MiB",
                    tile.tile_width,
                    tile.tile_height,
                    tile.tile_batch,
                    crop_bytes as f64 / (1024.0 * 1024.0),
                    MAX_PREDICT_CROP_BYTES / (1024 * 1024),
                ),
                "reduce `tile_batch` or use a smaller tile size",
            ));
        }
    }
    if let Some(tta) = &options.tta {
        let sizes = tta.resolve_imgszs(model_imgsz).map_err(|e| {
            bad(e.to_string(), "check `tta.scales` against the model's input size (32..=8192)")
        })?;
        let views = sizes.len().saturating_mul(if tta.hflip { 2 } else { 1 });
        if views > MAX_PREDICT_TTA_VIEWS {
            return Err(bad(
                format!("TTA produces {views} views, exceeding the server limit of {MAX_PREDICT_TTA_VIEWS}"),
                "pass fewer scales or set `hflip: false`",
            ));
        }
    }
    Ok(())
}

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
    /// The page of `issues` this query selects, with the counts of all of them, computed in a single pass.
    pub fn page<'a>(&self, issues: impl Iterator<Item = &'a folo_api::DatasetIssue>) -> folo_api::IssuePage {
        let mut total = 0u64;
        let mut page = Vec::new();
        let mut counts = folo_api::IssueCounts::default();
        for i in issues {
            match i.severity {
                folo_api::IssueSeverity::Error => counts.error += 1,
                folo_api::IssueSeverity::Warning => counts.warning += 1,
                folo_api::IssueSeverity::Info => counts.info += 1,
            }
            if let Some(n) = counts.by_kind.get_mut(i.kind.as_str()) {
                *n += 1;
            } else {
                counts.by_kind.insert(i.kind.clone(), 1);
            }
            if self.kind.as_ref().is_none_or(|k| *k == i.kind)
                && self.severity.is_none_or(|s| s == i.severity)
            {
                if total >= self.offset && (page.len() as u64) < self.limit {
                    page.push(i.clone());
                }
                total += 1;
            }
        }
        folo_api::IssuePage { total, issues: page, counts }
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
    /// (feature `run_cursors`). `RunLogs::total` still reports the full log length; a paginated poller advances its
    /// cursor to `after + lines.len()`.
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

/// The longest run name `POST /api/runs` takes, in characters (as long as a dataset's name may be).
pub const MAX_RUN_NAME_CHARS: usize = 64;

/// `400 invalid_config` for a run name over [`MAX_RUN_NAME_CHARS`] (every store's `create_run` checks this): one long
/// name would otherwise widen every list that shows runs and their models.
pub fn check_run_name(name: &str) -> Result<()> {
    if name.chars().count() > MAX_RUN_NAME_CHARS {
        return Err(crate::error::Error::new(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_config",
            format!("name: must be at most {MAX_RUN_NAME_CHARS} characters"),
        )
        .hint("pick a shorter run name, or leave it empty for <dataset name>-<size>"));
    }
    Ok(())
}

/// The name of a run created without one: `<dataset name>-<size>` (`shapes-n`), the dataset's name shortened so the
/// whole stays within [`MAX_RUN_NAME_CHARS`] (a run's config read back can be sent as a new run again).
pub fn default_run_name(dataset: &str, arch: folo_api::Arch) -> String {
    let size = arch.as_str().trim_start_matches("folo-");
    let room = MAX_RUN_NAME_CHARS.saturating_sub(size.chars().count() + 1);
    let base: String = dataset.chars().take(room).collect();
    format!("{}-{size}", base.trim_end())
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
/// `-f16.folo` for the f16 export, or, as `POST /api/models/{id}/export` names its files, `.onnx` with the
/// switches' suffix (`-decoded`, `-f16`, `-decoded-f16`: [`folo_api::OnnxSwitches::file_suffix`]).
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
        WeightsFormat::Onnx(s) => format!("{stem}{}.onnx", s.file_suffix()),
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
    /// Optional SAHI tiled inference or Test-Time Augmentation settings.
    pub options: PredictOptions,
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

/// A pre-serialized SSE frame shared across concurrent subscribers so fan-out serializes each event once (`O(1)`
/// across subscribers instead of `O(N)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseFrame {
    /// SSE `event:` name (e.g. `"status"`, `"step"`, `"job"`, `"bench"`).
    pub event: &'static str,
    /// Pre-serialized JSON payload for `data:`.
    pub data: Arc<str>,
}

impl SseFrame {
    /// Builds a shared frame from an event name and its JSON payload.
    pub fn new(event: &'static str, data: impl Into<Arc<str>>) -> Self {
        Self { event, data: data.into() }
    }

    /// Encodes a [`RunEvent`] once for SSE broadcast.
    pub fn run(ev: &RunEvent) -> Self {
        Self::new(ev.name(), ev.data_json())
    }

    /// Encodes a [`folo_api::LabelEvent`] once for SSE broadcast.
    pub fn label(ev: &folo_api::LabelEvent) -> Self {
        Self::new(ev.name(), ev.data_json())
    }

    /// Encodes a [`folo_api::BenchEvent`] once for SSE broadcast.
    pub fn bench(ev: &folo_api::BenchEvent) -> Self {
        Self::new(ev.name(), ev.data_json())
    }

    /// Turns this shared frame into an axum SSE [`Event`](axum::response::sse::Event).
    pub(crate) fn to_sse(&self) -> axum::response::sse::Event {
        axum::response::sse::Event::default().event(self.event).data(self.data.as_ref())
    }
}

/// Live stream backing `GET /api/runs/{id}/events`: either shared pre-serialized frames or typed events for custom
/// [`Store`] implementations that only implement [`Store::subscribe`].
pub enum RunSseStream {
    /// Pre-serialized frames shared across concurrent HTTP subscribers.
    Frames(broadcast::Receiver<SseFrame>),
    /// Typed events fallback.
    Events(broadcast::Receiver<RunEvent>),
}

/// Callback atomically returning a run's current [`StatusEvent`] paired with a tail-positioned frame receiver
/// for lag recovery (`None` once the run is deleted).
pub type RunSseResync = Arc<dyn Fn() -> Option<(StatusEvent, broadcast::Receiver<SseFrame>)> + Send + Sync>;

/// A live SSE subscription to a run's events, sharing pre-serialized payloads across concurrent clients and
/// carrying a weak resync callback for lag recovery.
pub struct RunSseSubscription {
    /// The run's initial `status` frame (sent first).
    pub initial: SseFrame,
    /// Subsequent live events.
    pub stream: RunSseStream,
    /// Atomic status snapshot and tail-positioned receiver for lag recovery (`None` when the store does not
    /// provide one).
    pub resync: Option<RunSseResync>,
}

/// The data behind the API. See the module docs.
pub trait Store: Send + Sync + 'static {
    /// True for the mock store.
    fn is_mock(&self) -> bool;
    /// GPUs to report in `/api/system`.
    fn gpus(&self) -> Vec<GpuInfo>;
    /// Does the slow part of [`Store::gpus`] that doesn't depend on [`Store::capabilities`] having run (the real
    /// store enumerates the adapters), so `/api/system` overlaps the two. May block; the default does nothing.
    fn prepare_gpus(&self) {}
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
    /// uploads, run cursors, checkpoints, the cloud); the ones every store has are listed by the server.
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
    /// Registers a folder on disk and returns `(dataset, rollback_token)` where `rollback_token` is `Some` iff this
    /// call created a brand-new exclusive registration that may be rolled back via [`Store::delete_dataset_if_unused`].
    /// Defaults to `(register_dataset, None)` when ownership is not tracked.
    fn register_dataset_tracked(&self, req: RegisterDataset) -> Result<(Dataset, Option<String>)> {
        self.register_dataset(req).map(|ds| (ds, None))
    }
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
    /// Unregisters `id` only if its active registration matches `token`, has not been shared with another
    /// registration, and is not referenced by any training run or benchmark (`409 dataset_in_use` otherwise).
    /// Defaults to refusing so stores that cannot prove exclusivity never unregister a live dataset.
    fn delete_dataset_if_unused(&self, id: &str, token: &str) -> Result<()> {
        let _ = token;
        Err(crate::error::Error::conflict(
            "dataset_in_use",
            format!("dataset '{id}' cannot be verified unused"),
        ))
    }
    /// Unregisters a dataset and deletes its folder, which must be one this server wrote under `<data dir>/datasets`
    /// (a generated sample set, a labeling job's export, an upload); any other is `400 invalid_query`.
    fn delete_dataset_files(&self, id: &str) -> Result<()> {
        let _ = id;
        Err(crate::error::Error::new(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_query",
            "this server does not delete dataset files".to_string(),
        ))
    }

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
    /// Starts a run with optional GPU VRAM-budget auto-batch sizing (`batch: "auto"` / `auto_batch: true`).
    /// When `auto_batch` is disabled, delegates to [`Store::create_run`].
    fn create_run_with_auto_batch(&self, config: TrainConfig, auto_batch: AutoBatchOptions) -> Result<Run> {
        check_auto_batch_options(&auto_batch)?;
        if !auto_batch.enabled {
            return self.create_run(config);
        }
        Err(crate::error::Error::not_implemented("auto-batch training"))
    }
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
    /// Subscribes to a run's live events as pre-serialized [`SseFrame`]s for HTTP SSE fan-out, with an optional
    /// weak resync callback for lag recovery. The default delegates to [`Store::subscribe`].
    fn subscribe_sse(&self, id: &str) -> Result<RunSseSubscription> {
        let sub = self.subscribe(id)?;
        Ok(RunSseSubscription {
            initial: SseFrame::run(&RunEvent::Status(sub.initial)),
            stream: RunSseStream::Events(sub.events),
            resync: None,
        })
    }

    /// Runs inference on one image.
    fn predict(&self, input: PredictInput) -> Result<PredictResponse>;

    /// Exports a model (`POST /api/models/{id}/export`), optionally with independent ONNX dynamic axes
    /// (`dynamic_batch`, `dynamic_hw`). Uniform axes (`NONE` or `ALL`) delegate to [`Store::export_model`]; split
    /// axes return `501 not_implemented` unless overridden.
    fn export_model_with_axes(
        &self,
        id: &str,
        mut spec: ExportSpec,
        axes: DynamicAxes,
    ) -> Result<ExportResult> {
        check_export_axes(&spec, axes)?;
        if axes.batch == axes.hw {
            spec.onnx.dynamic = axes.batch;
            return self.export_model(id, spec);
        }
        Err(crate::error::Error::not_implemented("split ONNX dynamic axes"))
    }

    /// Evaluates a model on a dataset split (`POST /api/val`, the MCP `val` tool). Stores without an evaluator
    /// answer `501 not_implemented` (the default).
    fn val(&self, req: ValRequest) -> Result<ValReport> {
        let _ = req;
        Err(crate::error::Error::not_implemented("validation"))
    }

    /// Evaluates a model on a dataset split, optionally computing per-image error diagnostics. When `diag` is
    /// `None`, delegates to [`Store::val`].
    fn val_with_diagnostics(
        &self,
        req: ValRequest,
        diag: Option<ValDiagConfig>,
    ) -> Result<(ValReport, Option<ValDiagnostics>)> {
        let Some(cfg) = diag else {
            return self.val(req).map(|r| (r, None));
        };
        check_val_diag_config(&cfg)?;
        Err(crate::error::Error::not_implemented("validation diagnostics"))
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
    /// Effective root folder of dataset `id` (`Index.root`, which honors a YOLO `data.yaml`'s `path:` override)
    /// when known in this process (`None` for mock procedural datasets or remote proxy stores).
    fn dataset_root(&self, id: &str) -> Result<Option<std::path::PathBuf>> {
        let _ = id;
        Ok(None)
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

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
