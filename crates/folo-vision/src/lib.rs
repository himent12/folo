//! FOLO detection models, losses, assigners, NMS and metrics.
//!
//! The detection math in this crate works on plain slices and small `Copy` structs, with no tensor types, so the
//! trainer, the predictor, the server and the browser (wasm) can all share it. The loss is the exception: it is
//! differentiable, so it works on folo-core tensors.
//!
//! - [`boxes`]: `xyxy` / `cxcywh` boxes, letterbox mapping, IoU / GIoU / DIoU / CIoU.
//! - [`decode`]: anchor grids and DFL decoding of raw head outputs.
//! - [`nms`]: class-aware batched NMS and the full detector post-process.
//! - [`fusion`]: cluster box fusion after NMS (docs/MODEL.md §9.4), opt-in through [`NmsConfig`].
//! - [`metrics`]: COCO-style mAP, precision / recall / F1, PR curves and a confusion matrix (`docs/METRICS.md`).
//! - [`calibration`]: post-hoc temperature scaling confidence calibration and ECE / MCE / Brier score metrics.
//! - [`cocoeval`]: the full pycocotools `COCOeval` (crowd regions, area buckets, AR) for reporting on COCO splits.
//! - [`assign`]: the task-aligned assigner and DFL target encoding.
//! - [`assigner`]: additional label assignment algorithms (SimOTA dynamic-`k`, exact Hungarian / linear-sum
//!   assignment, and [`OneToOneMatcher`]).
//! - [`loss`]: the detection loss (BCE + CIoU + DFL) on folo-core tensors, with autograd. The assigner runs on
//!   host copies of the detached head outputs; that read is the loss's only full-size device-to-host transfer.
//!
//! On top of folo-core / folo-nn, [`model`] defines the detector family itself (`folo-n/s/m/l/x`, docs/MODEL.md):
//! blocks, head, forward pass, fusion, raw prediction and `.folo` weight IO. With folo-data on top of that:
//!
//! - [`train`]: the [`Trainer`] (docs/MODEL.md §8: accumulation, schedule, EMA, clipping, validation,
//!   checkpoints, early stopping, resume) and its docs/API.md event stream.
//! - [`val`]: validation of a model on a dataset split.
//! - [`predict`]: the [`Predictor`] (letterbox, forward, decode, NMS, rescale, with a timing breakdown).
//! - [`export`]: ONNX export of a fused model ([`export::onnx`]).
//! - [`precompile`]: precompiles, on worker threads, the GPU kernels an earlier run of the same kind and shape
//!   used (docs/GPU.md, Pipeline cache).
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

pub mod assign;
pub mod assigner;
pub mod boxes;
pub mod calibration;
pub mod cocoeval;
pub mod decode;
mod error;
pub mod export;
pub mod fusion;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod nms;
pub mod precompile;
pub mod predict;
pub mod train;
pub mod val;

pub use assign::{Assignment, TalConfig};
pub use assigner::{
    AtssAssignment, AtssConfig, ClsCostKind, Gmm1dResult, HungarianResult, OneToOneAssignment,
    OneToOneConfig, OneToOneMatcher, PaaAssignment, PaaConfig, SimOtaAssignment, SimOtaConfig,
    anchor_boxes_from_points, assign_atss, assign_paa, assign_simota, fit_gmm_1d_two_component,
    hungarian_assign, hungarian_assign_f32,
};
pub use boxes::{Cxcywh, IouKind, Letterbox, Xyxy};
pub use calibration::{
    CalibrationConfig, CalibrationMetrics, ClassTemperatureFit, ClasswiseCalibrationConfig,
    ClasswiseCalibrationMetrics, ClasswiseTemperatureFit, ClasswiseTemperatureScaler,
    MulticlassCalibrationMetrics, TemperatureFit, TemperatureScaler,
};
pub use decode::Anchors;
pub use error::{Error, Result};
pub use loss::{
    DistillConfig, EmaSlideLoss, GwdTransform, ImageTargets, InnerIouBase, LossConfig, LossItems, LossOutput,
    SeesawConfig, SeesawLoss, SlideLossConfig, SlideWeightKind, detection_loss, gwd_loss, inner_iou,
    inner_iou_loss, kld_loss, prob_iou, prob_iou_l2_loss, prob_iou_loss, seesaw_loss, slide_bce_loss,
    slide_reweight_loss, wrap_inner_iou_loss,
};
pub use metrics::{
    BinningStrategy, CalibrationReport, CalibrationReportConfig, CalibrationSummary, ClassConfidenceSweep,
    ClassPrCurve, CocoScaleClassMetrics, CocoScaleDiagnostics, ConfidenceSweepConfig,
    ConfidenceSweepDiagnostics, ConfidenceSweepPoint, DetectionCalibrationDiagnostics,
    DetectionCalibrationReport, DetectionConfusionMatrix, EvalConfig, EvalReport, Evaluator, OptimalF1Point,
    PrCurveDiagnostics, PrCurvePoint, ReliabilityBin, ScaleAreaBounds, TideClassBreakdown, TideConfig,
    TideDiagnostics, TideErrorCounts, TideErrorImpacts, Truth, compute_calibration_report,
    compute_coco_scale_metrics, compute_confidence_sweep, compute_detection_calibration, compute_pr_curves,
    compute_tide_diagnostics, reliability_diagnostics_from_scores, reliability_from_scores,
};
pub use model::{FoloModel, HeadOutput, ModelConfig};
pub use nms::{
    ClusterNmsConfig, ClusterNmsEntry, ClusterScoreDecay, Detection, NmsConfig, NmsKind, batched_cluster_nms,
    batched_fast_nms, cluster_nms, cluster_nms_detections, fast_nms, fast_nms_detections,
};
pub use predict::{Precision, PredictConfig, Prediction, Predictor, PreparedBatch, Timing};
pub use train::{TrainConfig, TrainData, TrainOptions, Trainer};
