//! What a training run reports while it trains (docs/API.md, *Training runs*): steps, epochs and their
//! validation metrics (`/metrics`), samples (`/samples`), logs (`/logs`) and the SSE events of `/events`.

use serde::{Deserialize, Serialize};

use crate::{BBox, QueueEvent, RunCloud, RunStatus, ScoredBox};

/// Loss components.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Loss {
    /// CIoU box loss.
    #[serde(rename = "box")]
    pub box_: f32,
    /// Classification BCE.
    pub cls: f32,
    /// Distribution focal loss.
    pub dfl: f32,
    /// Weighted sum.
    pub total: f32,
}

/// One optimizer step (SSE `step`, and `/metrics` history).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Step {
    /// Epoch the step belongs to, 1-based.
    pub epoch: u32,
    /// Global step number, 1-based.
    pub step: u64,
    /// Learning rate used for this step.
    pub lr: f32,
    /// Training loss for this batch.
    pub loss: Loss,
    /// Throughput.
    pub imgs_per_s: f32,
    /// Estimated seconds remaining.
    pub eta_s: f64,
    /// Optimizer steps per epoch (`Run.steps_per_epoch`); 0 from servers that predate the field.
    #[serde(default)]
    pub steps_per_epoch: u64,
    /// Total epochs of the run (`Run.epochs`); 0 from servers that predate the field.
    #[serde(default)]
    pub epochs: u32,
}

/// One finished epoch with validation (SSE `epoch`, and `/metrics` history).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Epoch {
    /// Epoch number, 1-based.
    pub epoch: u32,
    /// Mean training loss over the epoch.
    pub train_loss: Loss,
    /// Validation metrics.
    pub val: ValMetrics,
    /// Wall time of the epoch including validation and its checkpoint's snapshot (and a wait for the previous
    /// epoch's checkpoint write, if it is still running; docs/API.md).
    pub duration_s: f64,
}

/// Validation metrics, overall and per class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ValMetrics {
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision (class mean) at `best_f1_conf`.
    pub precision: f32,
    /// Recall (class mean) at `best_f1_conf`.
    pub recall: f32,
    /// Per-class metrics (their precision and recall are taken at `best_f1_conf` too).
    pub per_class: Vec<ClassMetrics>,
    /// The operating point of `precision` and `recall`: the confidence (on a 0.001 grid) that maximises the
    /// class-mean F1 at IoU 0.5, one threshold for every class (docs/METRICS.md). `null` when unknown (epochs
    /// recorded before the field). Additive: missing reads as `null`.
    #[serde(default)]
    pub best_f1_conf: Option<f32>,
}

/// Validation metrics of one class.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ClassMetrics {
    /// Class index.
    pub class: u32,
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision.
    pub precision: f32,
    /// Recall.
    pub recall: f32,
}

/// `GET /api/runs/{id}/metrics`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunMetrics {
    /// Step history, oldest first (every `every`-th step when that query parameter is given).
    pub steps: Vec<Step>,
    /// Epoch history, oldest first.
    pub epochs: Vec<Epoch>,
}

/// `GET /api/runs/{id}/samples`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunSamples {
    /// Epoch the predictions come from.
    pub epoch: u32,
    /// The fixed validation images with truth and predictions.
    pub samples: Vec<Sample>,
}

/// One validation image with ground truth and predictions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Sample {
    /// Image id (fetch it from `/api/datasets/{dataset_id}/images/{image_id}`).
    pub image_id: String,
    /// Dataset id.
    pub dataset_id: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Ground truth.
    pub truth: Vec<BBox>,
    /// Predictions after NMS.
    pub pred: Vec<ScoredBox>,
}

/// `GET /api/runs/{id}/logs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunLogs {
    /// The most recent log lines, oldest first (with `after`: the lines after that many).
    pub lines: Vec<LogLine>,
    /// Lines in the run's whole log, so `after: total` asks for what comes next. 0 (an empty log, a log without a
    /// cursor, a server that predates the field) is left out.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total: u64,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// SSE `status` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StatusEvent {
    /// New state.
    pub status: RunStatus,
    /// Error message when `failed`.
    pub error: Option<String>,
    /// Optimizer steps per epoch (`Run.steps_per_epoch`): 0 while unknown (a run that has not been built yet) and
    /// from servers that predate the field.
    #[serde(default)]
    pub steps_per_epoch: u64,
    /// Total epochs of the run (`Run.epochs`); 0 from servers that predate the field.
    #[serde(default)]
    pub epochs: u32,
    /// The stable error code when `failed` (`Run.error_code`); absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

impl StatusEvent {
    /// A status event with unknown run length (`steps_per_epoch` and `epochs` 0) and no error code.
    pub fn new(status: RunStatus, error: Option<String>) -> StatusEvent {
        StatusEvent { status, error, steps_per_epoch: 0, epochs: 0, error_code: None }
    }

    /// Sets the error code (`failed` events).
    pub fn with_error_code(mut self, code: Option<String>) -> StatusEvent {
        self.error_code = code;
        self
    }

    /// Sets the run's length.
    pub fn with_length(mut self, epochs: u32, steps_per_epoch: u64) -> StatusEvent {
        self.epochs = epochs;
        self.steps_per_epoch = steps_per_epoch;
        self
    }
}

/// Log level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Debug.
    Debug,
    /// Info.
    Info,
    /// Warning.
    Warn,
    /// Error.
    Error,
}

/// SSE `log` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LogLine {
    /// RFC 3339 time.
    pub t: String,
    /// Level.
    pub level: LogLevel,
    /// Message.
    pub message: String,
}

/// One event of `GET /api/runs/{id}/events`. On the wire the SSE `event:` field is [`RunEvent::name`] and `data:`
/// is the JSON of the payload alone.
#[derive(Clone, Debug, PartialEq)]
pub enum RunEvent {
    /// `status`.
    Status(StatusEvent),
    /// `step`.
    Step(Step),
    /// `epoch`.
    Epoch(Epoch),
    /// `log`.
    Log(LogLine),
    /// `queue`.
    Queue(QueueEvent),
    /// `cloud`: the run's [`RunCloud`] after it changed (a new phase, transfer progress, cost). Boxed: it is the
    /// largest payload by far and the rarest event, and the others (a `step` ten times a second) stay small.
    Cloud(Box<RunCloud>),
}

impl RunEvent {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            RunEvent::Status(_) => "status",
            RunEvent::Step(_) => "step",
            RunEvent::Epoch(_) => "epoch",
            RunEvent::Log(_) => "log",
            RunEvent::Queue(_) => "queue",
            RunEvent::Cloud(_) => "cloud",
        }
    }

    /// The JSON `data:` payload.
    pub fn data_json(&self) -> String {
        let r = match self {
            RunEvent::Status(p) => serde_json::to_string(p),
            RunEvent::Step(p) => serde_json::to_string(p),
            RunEvent::Epoch(p) => serde_json::to_string(p),
            RunEvent::Log(p) => serde_json::to_string(p),
            RunEvent::Queue(p) => serde_json::to_string(p),
            RunEvent::Cloud(p) => serde_json::to_string(p),
        };
        r.expect("event payloads always serialize")
    }

    /// Parses an SSE event from its `event:` name and `data:` JSON. Returns `None` for unknown event names (new
    /// event types may be added; clients should skip them) and an error for malformed data.
    pub fn from_sse(name: &str, data: &str) -> Option<Result<RunEvent, serde_json::Error>> {
        Some(match name {
            "status" => serde_json::from_str(data).map(RunEvent::Status),
            "step" => serde_json::from_str(data).map(RunEvent::Step),
            "epoch" => serde_json::from_str(data).map(RunEvent::Epoch),
            "log" => serde_json::from_str(data).map(RunEvent::Log),
            "queue" => serde_json::from_str(data).map(RunEvent::Queue),
            "cloud" => serde_json::from_str(data).map(RunEvent::Cloud),
            _ => return None,
        })
    }
}
