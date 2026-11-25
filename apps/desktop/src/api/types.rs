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
    PredictResponse, RegisterDataset, Run, RunDataset, RunEvent, RunLogs, RunMetrics, RunSamples, RunStatus,
    Sample, ScoredBox, Split, Splits, StatusEvent, Step, SyntheticDataset, SystemInfo, Timing, TrainConfig,
    ValMetrics,
};
pub use folo_api::{
    BENCH_SEVERITIES, Bench, BenchAccuracy, BenchAxis, BenchBest, BenchCalibration, BenchCalibrationBin,
    BenchCatalog, BenchClassAccuracy, BenchConditionGroup, BenchConditionResult, BenchContestant,
    BenchContestantSpec, BenchDesign, BenchDesigner, BenchDetection, BenchEntry, BenchEntryImage,
    BenchEntryImagePage, BenchEntryList, BenchEntryStatus, BenchEvent, BenchGeneration, BenchImage,
    BenchImagePage, BenchLabelPoint, BenchLabels, BenchList, BenchLocalization, BenchLoop, BenchLoopList,
    BenchLoopPhase, BenchLoopStatus, BenchLoopVerdict, BenchPlan, BenchPlanCondition, BenchPrediction,
    BenchProgress, BenchReadiness, BenchRemoved, BenchResult, BenchRobustness, BenchSelfLabeling,
    BenchSeverityPoint, BenchSource, BenchSpeed, BenchStudent, CreateBench, DesignBench,
    FEATURE_AGENT_LABELING, FEATURE_VISION_BENCH, LabelAgent, LabelAgentId, LabelAgentList, LabelCounts,
    LabelEffort, LabelJob, LabelJobList, LabelJobStatus, LabelModel, LabelUsage, StartBenchEntry,
    StartBenchLoop, UpdateBench,
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

impl Label for BenchEntryStatus {
    fn label(&self) -> &'static str {
        match self {
            BenchEntryStatus::Queued => "Queued",
            BenchEntryStatus::Running => "Running",
            BenchEntryStatus::Stopping => "Stopping",
            BenchEntryStatus::Done => "Done",
            BenchEntryStatus::Failed => "Failed",
            BenchEntryStatus::Stopped => "Stopped",
        }
    }
}

impl Label for BenchLoopStatus {
    fn label(&self) -> &'static str {
        match self {
            BenchLoopStatus::Running => "Running",
            BenchLoopStatus::Stopping => "Stopping",
            BenchLoopStatus::Done => "Done",
            BenchLoopStatus::Failed => "Failed",
            BenchLoopStatus::Stopped => "Stopped",
        }
    }
}

impl Label for BenchLoopPhase {
    fn label(&self) -> &'static str {
        match self {
            BenchLoopPhase::Benchmarking => "Benchmarking",
            BenchLoopPhase::Labeling => "Labeling",
            BenchLoopPhase::Training => "Training",
        }
    }
}

impl Label for BenchLoopVerdict {
    fn label(&self) -> &'static str {
        match self {
            BenchLoopVerdict::Improved => "Improved",
            BenchLoopVerdict::Held => "Held",
            BenchLoopVerdict::Regressed => "Regressed",
        }
    }
}

impl Label for BenchConditionGroup {
    fn label(&self) -> &'static str {
        match self {
            BenchConditionGroup::Light => "Light",
            BenchConditionGroup::Blur => "Blur",
            BenchConditionGroup::Noise => "Noise",
            BenchConditionGroup::Medium => "Medium",
            BenchConditionGroup::Digital => "Digital",
            BenchConditionGroup::Geometry => "Geometry",
        }
    }
}

impl Label for BenchReadiness {
    fn label(&self) -> &'static str {
        match self {
            BenchReadiness::Ready => "Ready to self-label",
            BenchReadiness::Partial => "Partly ready",
            BenchReadiness::NotReady => "Not ready",
        }
    }
}

impl Label for LabelEffort {
    fn label(&self) -> &'static str {
        match self {
            LabelEffort::Low => "Low",
            LabelEffort::Medium => "Medium",
            LabelEffort::High => "High",
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
    /// A checkpoint to continue from: a model, or a validated best epoch. A run stopped or failed
    /// before its first epoch ended has none, so it starts again rather than resumes (as in Studio).
    fn can_continue(&self) -> bool;
    /// A run waiting in this machine's queue that never started: Remove from queue deletes it
    /// (docs/API.md, `DELETE /api/runs/{id}`). A resumed run waiting to continue is stopped instead,
    /// and keeps what it trained.
    fn never_started(&self) -> bool;
    /// A stopped run's title, in Studio's words: a stop the server's shutdown or restart made
    /// (`error_code: "interrupted"`) is never said as the user's own.
    fn stopped_title(&self) -> String;
    /// What a stopped run kept, in Studio's words.
    fn stopped_detail(&self) -> &'static str;
    /// The epoch a failed run failed in: the one its last step was in (an older server reports the
    /// last finished epoch), as Studio's `failedEpoch` counts it, so both apps name the same one.
    fn failed_epoch(&self) -> u32;
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

    fn failed_epoch(&self) -> u32 {
        let from_steps = if self.steps_per_epoch > 0 { self.step.div_ceil(self.steps_per_epoch) } else { 0 };
        self.epoch.max(u32::try_from(from_steps).unwrap_or(u32::MAX))
    }

    fn cloud_pending(&self) -> Option<&RunCloud> {
        // Named, not compared: a newer server's phase (`Unknown`) has no known place in the order.
        let handing_over =
            |p: CloudPhase| matches!(p, CloudPhase::Provisioning | CloudPhase::Booting | CloudPhase::Syncing);
        self.cloud.as_ref().filter(|c| self.status == RunStatus::Queued && handing_over(c.phase))
    }

    fn can_continue(&self) -> bool {
        self.model_id.is_some() || self.best.is_some()
    }

    fn never_started(&self) -> bool {
        self.status == RunStatus::Queued && self.started.is_none() && self.cloud.is_none()
    }

    fn stopped_title(&self) -> String {
        match (self.error_code.as_deref() == Some("interrupted"), self.started.is_some()) {
            (true, true) => "Interrupted: the server stopped during training".into(),
            (true, false) => "Interrupted: the server stopped before it started".into(),
            _ if self.can_continue() => format!("Stopped at epoch {}", self.epoch),
            (_, true) => "Stopped during its first epoch".into(),
            (_, false) => "Stopped before it started".into(),
        }
    }

    fn stopped_detail(&self) -> &'static str {
        match (self.can_continue(), self.model_id.is_some(), self.started.is_some()) {
            (true, true, _) => "The last checkpoint was kept and saved as a model.",
            (true, false, _) => "The last checkpoint was kept.",
            (false, _, true) => "No epoch had finished, so nothing was saved.",
            (false, _, false) => "It never trained, so nothing was saved.",
        }
    }
}

/// The run on this machine that holds the training device (running or stopping), other than
/// `except`. Cloud runs train elsewhere and never hold it.
pub fn training_now<'a>(runs: &'a [Run], except: &str) -> Option<&'a Run> {
    runs.iter().find(|r| {
        r.id != except && r.cloud.is_none() && matches!(r.status, RunStatus::Running | RunStatus::Stopping)
    })
}

/// Whether a run the server just accepted waits for others rather than training at once.
/// `POST /api/runs` answers `queued` either way (docs/API.md), so its status alone can't say: it
/// waits behind a place past the first, a run holding the device, or (a server without queue
/// places, or a run whose trainer is being built) another queued run on this machine.
pub fn waits_to_start(run: &Run, runs: &[Run]) -> bool {
    run.status == RunStatus::Queued
        && (run.queue_position.is_some_and(|p| p > 1)
            || training_now(runs, &run.id).is_some()
            || runs.iter().any(|r| {
                r.id != run.id
                    && r.cloud.is_none()
                    && r.status == RunStatus::Queued
                    && r.queue_position.is_none()
            }))
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
    fn a_failed_run_names_the_epoch_its_last_step_was_in() {
        use crate::api::Backend;
        let api = crate::api::mock::MockBackend::new(4200, true);
        let base = api.run("run_a19f").unwrap();
        let failed = |epoch: u32, step: u64, steps_per_epoch: u64| {
            Run { status: RunStatus::Failed, epoch, step, steps_per_epoch, ..base.clone() }.failed_epoch()
        };
        // An older server reports the last finished epoch (2) for a run that failed 30 steps into epoch 3.
        assert_eq!(failed(2, 130, 50), 3);
        // A server that reports the epoch in progress, and a failure on an epoch's last step.
        assert_eq!(failed(3, 130, 50), 3);
        assert_eq!(failed(2, 100, 50), 2);
        // Before the first step, and without a step count.
        assert_eq!(failed(0, 0, 50), 0);
        assert_eq!(failed(4, 0, 0), 4);
    }

    #[test]
    fn a_stopped_run_says_what_it_kept_and_who_stopped_it() {
        use crate::api::Backend;
        let api = crate::api::mock::MockBackend::new(4200, true);
        let base = api.run("run_a19f").unwrap();
        let stopped = |started: bool, best: bool, code: Option<&str>| Run {
            status: RunStatus::Stopped,
            started: started.then(|| "2026-11-08T06:00:02Z".into()),
            best: if best { base.best } else { None },
            model_id: None,
            error_code: code.map(Into::into),
            ..base.clone()
        };
        let before = stopped(false, false, None);
        assert!(!before.can_continue());
        assert_eq!(before.stopped_title(), "Stopped before it started");
        assert_eq!(before.stopped_detail(), "It never trained, so nothing was saved.");
        assert_eq!(stopped(true, false, None).stopped_title(), "Stopped during its first epoch");
        let kept = stopped(true, true, None);
        assert!(kept.can_continue());
        assert_eq!(kept.stopped_title(), format!("Stopped at epoch {}", kept.epoch));
        assert_eq!(
            stopped(true, true, Some("interrupted")).stopped_title(),
            "Interrupted: the server stopped during training"
        );
        assert_eq!(
            stopped(false, false, Some("interrupted")).stopped_title(),
            "Interrupted: the server stopped before it started"
        );
        // Only a run that never started leaves the queue by being deleted.
        let queued = Run { status: RunStatus::Queued, ..before.clone() };
        assert!(queued.never_started());
        assert!(!Run { started: kept.started.clone(), ..queued }.never_started());
    }

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
    fn a_new_run_waits_only_behind_another() {
        let json = r#"{ "id": "run_a", "name": "a", "status": "queued", "queue_position": 1,
          "config": { "name": "a", "dataset_id": "ds", "arch": "folo-n", "epochs": 10, "batch": 16,
            "imgsz": 320, "device": "auto", "optimizer": "auto", "lr0": 0.01, "weights": null,
            "augment": { "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4], "fliplr": 0.5,
              "scale": 0.5, "translate": 0.1, "close_mosaic": 10 }, "seed": 0 },
          "created": "2026-11-08T06:00:00Z", "started": null, "finished": null, "epoch": 0,
          "epochs": 10, "step": 0, "steps_per_epoch": 0, "eta_s": null, "best": null,
          "device": "", "model_id": null, "error": null }"#;
        let new: Run = serde_json::from_str(json).unwrap();
        let other = |id: &str, status: RunStatus, place: Option<u32>| Run {
            id: id.into(),
            status,
            queue_position: place,
            ..new.clone()
        };
        let done = other("run_d", RunStatus::Completed, None);
        // The server's first answer is always `queued`, at the front: with nothing else on the
        // device it trains at once.
        assert!(!waits_to_start(&new, &[new.clone(), done.clone()]));
        // Behind a run that trains, or one whose trainer is being built.
        let busy = other("run_b", RunStatus::Running, None);
        assert!(waits_to_start(&new, &[new.clone(), busy.clone()]));
        assert_eq!(training_now(&[new.clone(), busy], "run_a").map(|r| r.id.as_str()), Some("run_b"));
        assert!(waits_to_start(&new, &[new.clone(), other("run_c", RunStatus::Queued, None)]));
        // Past the first place.
        let second = Run { queue_position: Some(2), ..new.clone() };
        assert!(waits_to_start(&second, std::slice::from_ref(&second)));
        // A run that already trains (a later status) never "waits".
        assert!(!waits_to_start(&Run { status: RunStatus::Running, ..new.clone() }, &[new, done]));
    }
}
