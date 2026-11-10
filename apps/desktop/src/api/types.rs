//! Wire types for `docs/API.md`: the shared `folo-api` crate, plus small client-side helpers.
//!
//! The desktop app uses the same serde types as `folo-server`, so a contract change is a compile
//! error here rather than a runtime surprise.

pub use folo_api::{
    AmpSupport, BenchmarkReport, BenchmarkRequest, Capabilities, CliCommand, Confidence, EstimateBasis,
    EstimatePhase, Head, InferPrecision, InferThroughput, ModelSource, PhaseKind, QueueAction, QueueRequest,
    ResolvedOptimizer, ResolvedTrain, RunEstimate, SystemResponse, TensorCoreSupport, ThroughputSource,
    ThroughputTable, TrainThroughput,
};
pub use folo_api::{
    Arch, Augment, BBox, BestEpoch, ClassMetrics, CpuInfo, Dataset, DatasetFormat, DatasetStats, Epoch,
    ExportResult, GpuBackend, GpuInfo, Health, Histogram, ImageInfo, ImagePage, ImageSize, IssueCounts,
    IssuePage, LogLevel, LogLine, Loss, MemoryInfo, Model, ModelEval, ModelMetrics, Optimizer, PrCurve,
    PredictResponse, RegisterDataset, Run, RunEvent, RunMetrics, RunSamples, RunStatus, Sample, ScoredBox,
    Split, Splits, StatusEvent, Step, SyntheticDataset, SystemInfo, Timing, TrainConfig, ValMetrics,
};
pub use folo_api::{
    CloudAvailability, CloudConnection, CloudConnectionStatus, CloudField, CloudInstance,
    CloudInstanceStatus, CloudOffer, CloudOfferList, CloudPhase, CloudProvider, Compute, ConnectCloud,
    CostEstimate, RunCloud, TransferProgress,
};

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

/// A cloud run's phase in the words people see (docs/CLOUD.md).
impl Label for CloudPhase {
    fn label(&self) -> &'static str {
        match self {
            CloudPhase::Provisioning => "Provisioning",
            CloudPhase::Booting => "Booting",
            CloudPhase::Syncing => "Uploading data",
            CloudPhase::Training => "Training",
            CloudPhase::Collecting => "Collecting weights",
            CloudPhase::Releasing => "Releasing",
            CloudPhase::Released => "Released",
            // A newer server's phase: say so plainly instead of guessing which one it is.
            CloudPhase::Unknown => "In progress",
        }
    }
}

impl Label for CloudInstanceStatus {
    fn label(&self) -> &'static str {
        match self {
            CloudInstanceStatus::Launching => "Launching",
            CloudInstanceStatus::Starting => "Starting",
            CloudInstanceStatus::Running => "Running",
            CloudInstanceStatus::Terminating => "Terminating",
            CloudInstanceStatus::Terminated => "Terminated",
            CloudInstanceStatus::Lost => "Lost",
            CloudInstanceStatus::Unknown => "Unknown",
        }
    }
}

impl Label for Optimizer {
    fn label(&self) -> &'static str {
        match self {
            Optimizer::Auto => "auto",
            Optimizer::Sgd => "sgd",
            Optimizer::Adamw => "adamw",
            Optimizer::Musgd => "musgd",
            Optimizer::Unknown => "unknown",
        }
    }
}

/// Derived values of a run.
pub trait RunExt {
    /// Fraction of all training steps done, 0..=1.
    fn progress(&self) -> f32;
    /// Epochs at the end trained without mosaic, as the engine runs them: the server's resolved
    /// value, else `augment.close_mosaic` capped at half the run (docs/MODEL.md §8.4); every epoch
    /// when mosaic is off.
    fn close_mosaic_epochs(&self) -> u32;
    /// A cloud run that hasn't reached training yet (provisioning, booting, uploading): it waits on
    /// its instance, not in the training queue.
    fn cloud_pending(&self) -> Option<&RunCloud>;
}

/// `close_mosaic` as the engine applies it to `epochs` (capped at half the run).
pub fn capped_close_mosaic(close_mosaic: u32, epochs: u32) -> u32 {
    close_mosaic.min(epochs / 2)
}

impl RunExt for Run {
    fn close_mosaic_epochs(&self) -> u32 {
        if self.config.augment.mosaic <= 0. {
            return self.epochs;
        }
        self.resolved
            .as_ref()
            .and_then(|r| r.close_mosaic)
            .unwrap_or_else(|| capped_close_mosaic(self.config.augment.close_mosaic, self.epochs))
    }

    fn progress(&self) -> f32 {
        let total = (self.epochs as u64 * self.steps_per_epoch).max(1);
        (self.step as f64 / total as f64).clamp(0.0, 1.0) as f32
    }

    fn cloud_pending(&self) -> Option<&RunCloud> {
        // Named, not compared: a newer server's phase (`Unknown`) has no known place in the order.
        let handing_over =
            |p: CloudPhase| matches!(p, CloudPhase::Provisioning | CloudPhase::Booting | CloudPhase::Syncing);
        self.cloud.as_ref().filter(|c| self.status == RunStatus::Queued && handing_over(c.phase))
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
