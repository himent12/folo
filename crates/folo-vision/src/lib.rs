//! FOLO detection models, losses, assigners, NMS and metrics.
//!
//! The detection math in this crate works on plain slices and small `Copy` structs, with no tensor types, so the
//! trainer, the predictor, the server and the browser (wasm) can all share it. The loss is the exception: it is
//! differentiable, so it works on folo-core tensors.
//!
//! - [`boxes`]: `xyxy` / `cxcywh` boxes, letterbox mapping, IoU / GIoU / DIoU / CIoU.
//! - [`decode`]: anchor grids and DFL decoding of raw head outputs.
//! - [`nms`]: class-aware batched NMS and the full detector post-process.
//! - [`metrics`]: COCO-style mAP, precision / recall / F1, PR curves and a confusion matrix (`docs/METRICS.md`).
//! - [`assign`]: the task-aligned assigner and DFL target encoding.
//! - [`loss`]: the detection loss (BCE + CIoU + DFL) on folo-core tensors, with autograd. The assigner runs on
//!   host copies of the detached head outputs; that read is the loss's only full-size device-to-host transfer.
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

pub mod assign;
pub mod boxes;
pub mod decode;
pub mod loss;
pub mod metrics;
pub mod nms;

pub use assign::{Assignment, TalConfig};
pub use boxes::{Cxcywh, IouKind, Letterbox, Xyxy};
pub use decode::Anchors;
pub use loss::{ImageTargets, LossConfig, LossItems, LossOutput, detection_loss};
pub use metrics::{EvalConfig, EvalReport, Evaluator, Truth};
pub use nms::{Detection, NmsConfig};
