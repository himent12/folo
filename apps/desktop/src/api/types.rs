//! Wire types for `docs/API.md`: the shared `folo-api` crate, plus small client-side helpers.
//!
//! The desktop app uses the same serde types as `folo-server`, so a contract change is a compile
//! error here rather than a runtime surprise.

pub use folo_api::{
    Arch, Augment, BBox, BestEpoch, ClassMetrics, CpuInfo, Dataset, DatasetFormat, DatasetStats, Epoch,
    ExportResult, GpuBackend, GpuInfo, Health, Histogram, ImageInfo, ImagePage, ImageSize, IssueCounts,
    IssuePage, LogLevel, LogLine, Loss, MemoryInfo, Model, ModelEval, ModelMetrics, Optimizer, PrCurve,
    PredictResponse, RegisterDataset, Run, RunEvent, RunMetrics, RunSamples, RunStatus, Sample, ScoredBox,
    Split, Splits, StatusEvent, Step, SyntheticDataset, SystemInfo, Timing, TrainConfig, ValMetrics,
};
pub use folo_api::{Capabilities, SystemResponse};

/// Display helpers for wire enums.
pub trait Label {
    fn label(&self) -> &'static str;
}

impl Label for RunStatus {
    fn label(&self) -> &'static str {
        match self {
            RunStatus::Queued => "Queued",
            RunStatus::Running => "Running",
            RunStatus::Stopping => "Stopping",
            RunStatus::Stopped => "Stopped",
            RunStatus::Completed => "Completed",
            RunStatus::Failed => "Failed",
        }
    }
}

impl Label for Split {
    fn label(&self) -> &'static str {
        self.as_str()
    }
}

impl Label for DatasetFormat {
    fn label(&self) -> &'static str {
        match self {
            DatasetFormat::Auto => "Auto",
            DatasetFormat::Yolo => "YOLO",
            DatasetFormat::Coco => "COCO",
        }
    }
}

impl Label for GpuBackend {
    fn label(&self) -> &'static str {
        match self {
            GpuBackend::Vulkan => "Vulkan",
            GpuBackend::Dx12 => "DX12",
            GpuBackend::Metal => "Metal",
        }
    }
}

impl Label for Optimizer {
    fn label(&self) -> &'static str {
        match self {
            Optimizer::Auto => "auto",
            Optimizer::Sgd => "sgd",
            Optimizer::Adamw => "adamw",
        }
    }
}

/// Derived values of a run.
pub trait RunExt {
    /// Fraction of all training steps done, 0..=1.
    fn progress(&self) -> f32;
}

impl RunExt for Run {
    fn progress(&self) -> f32 {
        let total = (self.epochs as u64 * self.steps_per_epoch).max(1);
        (self.step as f64 / total as f64).clamp(0.0, 1.0) as f32
    }
}

pub trait SplitsExt {
    fn total(&self) -> u64;
}

impl SplitsExt for Splits {
    fn total(&self) -> u64 {
        self.train + self.val + self.test
    }
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
        assert_eq!(run.status.label(), "Running");
    }

    #[test]
    fn sse_events_parse() {
        let step = r#"{ "epoch": 12, "step": 340, "lr": 0.0098,
          "loss": { "box": 1.21, "cls": 0.83, "dfl": 1.02, "total": 3.06 },
          "imgs_per_s": 412.5, "eta_s": 1840 }"#;
        match RunEvent::from_sse("step", step) {
            Some(Ok(RunEvent::Step(s))) => assert_eq!(s.loss.box_, 1.21),
            other => panic!("unexpected {other:?}"),
        }
        assert!(RunEvent::from_sse("nope", "{}").is_none());
    }
}
