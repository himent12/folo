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
//! - [`eval`]: per-image error diagnosis ([`ImageDiagnostics`], [`eval::diagnose_image`], [`eval::worst_images`])
//!   wired into [`Evaluator`].
//! - [`cocoeval`]: the full pycocotools `COCOeval` (crowd regions, area buckets, AR) for reporting on COCO splits.
//! - [`assign`]: the task-aligned assigner and DFL target encoding.
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
//! - [`tile`]: SAHI tiled inference ([`TileConfig`], [`Predictor::predict_tiled`]) for small objects in large images.
//! - [`tta`]: Test-Time Augmentation ([`TtaConfig`], [`PredictOptions`], [`Predictor::predict_tta`]) with
//!   multi-scale + horizontal-flip ensembling and cross-view box fusion.
//! - [`track`]: ByteTrack multi-object tracking ([`ByteTracker`], [`TrackConfig`], [`TrackedDetection`],
//!   [`Predictor::predict_sequence`]).
//! - [`export`]: ONNX export of a fused model ([`export::onnx`]).
//! - [`precompile`]: precompiles, on worker threads, the GPU kernels an earlier run of the same kind and shape
//!   used (docs/GPU.md, Pipeline cache).
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

pub mod assign;
pub mod boxes;
pub mod cocoeval;
pub mod decode;
mod error;
pub mod eval;
pub mod export;
pub mod fusion;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod nms;
pub mod precompile;
pub mod predict;
pub mod tile;
pub mod track;
pub mod train;
pub mod tta;
pub mod val;

pub use assign::{Assignment, TalConfig};
pub use boxes::{Cxcywh, IouKind, Letterbox, Xyxy};
pub use decode::Anchors;
pub use error::{Error, Result};
pub use eval::ImageDiagnostics;
pub use loss::{DistillConfig, ImageTargets, LossConfig, LossItems, LossOutput, detection_loss};
pub use metrics::{EvalConfig, EvalReport, Evaluator, Truth};
pub use model::{FoloModel, HeadOutput, ModelConfig};
pub use nms::{Detection, NmsConfig, NmsKind};
pub use predict::{Precision, PredictConfig, Prediction, Predictor, PreparedBatch, Timing};
pub use tile::{TileConfig, TileMergeMetric, TileRect};
pub use track::{ByteTracker, SequenceFrame, TrackConfig, TrackedDetection};
pub use train::{
    AutoBatchPlan, AutoBatchRequest, BatchLimitFactor, TrainConfig, TrainData, TrainOptions, Trainer,
    auto_batch_plan, probe_batch_size,
};
pub use tta::{PredictOptions, TtaConfig};
