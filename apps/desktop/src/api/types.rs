//! Wire types for `docs/API.md` (v1).
//!
//! Interim copy until the shared API types crate lands on main; field names and shapes follow the
//! contract exactly so the swap is mechanical.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ApiErrorBody {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub hint: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Health {
    pub status: String,
    pub version: String,
    #[serde(default)]
    pub mock: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cpu {
    pub name: String,
    pub threads: u32,
    pub util: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gpu {
    pub ordinal: u32,
    pub name: String,
    pub backend: String,
    pub memory_bytes: u64,
    #[serde(default)]
    pub util: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SystemInfo {
    pub cpu: Cpu,
    pub memory: Memory,
    pub gpus: Vec<Gpu>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub fn label(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Splits {
    pub train: u32,
    pub val: u32,
    pub test: u32,
}

impl Splits {
    pub fn total(&self) -> u32 {
        self.train + self.val + self.test
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Hist {
    pub edges: Vec<f32>,
    pub counts: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DatasetStats {
    pub boxes: u32,
    pub empty_images: u32,
    pub broken_labels: u32,
    pub class_counts: Vec<u32>,
    pub box_size_hist: Hist,
    pub boxes_per_image_hist: Hist,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub path: String,
    pub format: String,
    pub classes: Vec<String>,
    pub created: String,
    pub splits: Splits,
    #[serde(default)]
    pub stats: Option<DatasetStats>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct LabelBox {
    pub class: u32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoredBox {
    pub class: u32,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImageMeta {
    pub id: String,
    pub file: String,
    pub split: Split,
    pub width: u32,
    pub height: u32,
    pub boxes: Vec<LabelBox>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImagePage {
    pub total: u32,
    pub images: Vec<ImageMeta>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelMetrics {
    pub map50: f32,
    pub map50_95: f32,
    pub precision: f32,
    pub recall: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub arch: String,
    pub imgsz: u32,
    pub classes: Vec<String>,
    pub params: u64,
    pub flops_g: f32,
    pub bytes: u64,
    pub created: String,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub metrics: Option<ModelMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExportResult {
    pub path: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Augment {
    pub mosaic: f32,
    pub mixup: f32,
    pub hsv: [f32; 3],
    pub fliplr: f32,
    pub scale: f32,
    pub translate: f32,
    pub close_mosaic: u32,
}

impl Default for Augment {
    fn default() -> Self {
        Self {
            mosaic: 1.0,
            mixup: 0.0,
            hsv: [0.015, 0.7, 0.4],
            fliplr: 0.5,
            scale: 0.5,
            translate: 0.1,
            close_mosaic: 10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainConfig {
    pub name: String,
    pub dataset_id: String,
    pub arch: String,
    pub epochs: u32,
    pub batch: u32,
    pub imgsz: u32,
    pub device: String,
    pub optimizer: String,
    pub lr0: f32,
    #[serde(default)]
    pub weights: Option<String>,
    pub augment: Augment,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Queued,
    Running,
    Stopping,
    Stopped,
    Completed,
    Failed,
}

impl RunStatus {
    pub fn label(self) -> &'static str {
        match self {
            RunStatus::Queued => "Queued",
            RunStatus::Running => "Running",
            RunStatus::Stopping => "Stopping",
            RunStatus::Stopped => "Stopped",
            RunStatus::Completed => "Completed",
            RunStatus::Failed => "Failed",
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, RunStatus::Queued | RunStatus::Running | RunStatus::Stopping)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Best {
    pub epoch: u32,
    pub map50_95: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Run {
    pub id: String,
    pub name: String,
    pub status: RunStatus,
    pub config: TrainConfig,
    pub created: String,
    #[serde(default)]
    pub started: Option<String>,
    #[serde(default)]
    pub finished: Option<String>,
    pub epoch: u32,
    pub epochs: u32,
    pub step: u32,
    pub steps_per_epoch: u32,
    #[serde(default)]
    pub eta_s: Option<f64>,
    #[serde(default)]
    pub best: Option<Best>,
    pub device: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl Run {
    /// Fraction of all training steps done, 0..=1.
    pub fn progress(&self) -> f32 {
        let total = (self.epochs * self.steps_per_epoch).max(1);
        (self.step as f32 / total as f32).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Loss {
    #[serde(rename = "box")]
    pub box_: f32,
    pub cls: f32,
    pub dfl: f32,
    pub total: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Step {
    pub epoch: u32,
    pub step: u32,
    pub lr: f32,
    pub loss: Loss,
    pub imgs_per_s: f32,
    #[serde(default)]
    pub eta_s: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassMetrics {
    pub class: u32,
    pub map50: f32,
    pub map50_95: f32,
    pub precision: f32,
    pub recall: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValMetrics {
    pub map50: f32,
    pub map50_95: f32,
    pub precision: f32,
    pub recall: f32,
    #[serde(default)]
    pub per_class: Vec<ClassMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Epoch {
    pub epoch: u32,
    pub train_loss: Loss,
    pub val: ValMetrics,
    pub duration_s: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Metrics {
    pub steps: Vec<Step>,
    pub epochs: Vec<Epoch>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    pub image_id: String,
    pub dataset_id: String,
    pub width: u32,
    pub height: u32,
    pub truth: Vec<LabelBox>,
    pub pred: Vec<ScoredBox>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Samples {
    pub epoch: u32,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogLine {
    pub t: String,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StatusEvent {
    pub status: RunStatus,
    #[serde(default)]
    pub error: Option<String>,
}

/// One SSE event from `GET /api/runs/{id}/events`, keyed by its `event:` name.
#[derive(Clone, Debug, PartialEq)]
pub enum RunEvent {
    Status(StatusEvent),
    Step(Step),
    Epoch(Epoch),
    Log(LogLine),
}

impl RunEvent {
    /// Parse one SSE event from its name and `data:` payload.
    pub fn parse(event: &str, data: &str) -> Option<Self> {
        Some(match event {
            "status" => RunEvent::Status(serde_json::from_str(data).ok()?),
            "step" => RunEvent::Step(serde_json::from_str(data).ok()?),
            "epoch" => RunEvent::Epoch(serde_json::from_str(data).ok()?),
            "log" => RunEvent::Log(serde_json::from_str(data).ok()?),
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DetBox {
    pub class: u32,
    pub name: String,
    pub score: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Timing {
    pub preprocess_ms: f32,
    pub inference_ms: f32,
    pub postprocess_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub model_id: String,
    pub image: ImageSize,
    pub boxes: Vec<DetBox>,
    pub timing: Timing,
    pub device: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_parses_contract_example() {
        let json = r#"{ "id": "run_7d1e", "name": "shapes-n", "status": "running",
          "config": { "name": "shapes-n", "dataset_id": "ds_8f2c", "arch": "folo-n", "epochs": 100,
            "batch": 16, "imgsz": 640, "device": "auto", "optimizer": "auto", "lr0": 0.01,
            "weights": null, "augment": { "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4],
            "fliplr": 0.5, "scale": 0.5, "translate": 0.1, "close_mosaic": 10 }, "seed": 0 },
          "created": "2026-11-08T06:00:00Z", "started": "2026-11-08T06:00:02Z", "finished": null,
          "epoch": 12, "epochs": 100, "step": 340, "steps_per_epoch": 50, "eta_s": 1840,
          "best": { "epoch": 11, "map50_95": 0.61 }, "device": "NVIDIA GeForce RTX 4080 SUPER (vulkan)",
          "model_id": null, "error": null }"#;
        let run: Run = serde_json::from_str(json).unwrap();
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.best.as_ref().unwrap().epoch, 11);
        assert!((run.progress() - 340.0 / 5000.0).abs() < 1e-6);
    }

    #[test]
    fn sse_events_parse() {
        let step = r#"{ "epoch": 12, "step": 340, "lr": 0.0098,
          "loss": { "box": 1.21, "cls": 0.83, "dfl": 1.02, "total": 3.06 },
          "imgs_per_s": 412.5, "eta_s": 1840 }"#;
        match RunEvent::parse("step", step) {
            Some(RunEvent::Step(s)) => assert_eq!(s.loss.box_, 1.21),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            RunEvent::parse("status", r#"{ "status": "completed", "error": null }"#),
            Some(RunEvent::Status(StatusEvent { status: RunStatus::Completed, .. }))
        ));
        assert_eq!(RunEvent::parse("nope", "{}"), None);
    }
}
