//! Unified [`Detector`] abstraction over FOLO v1 ([`FoloModel`]) and FOLO v2-Iso ([`FoloV2Model`]).
//!
//! Lets [`crate::train::Trainer`], [`crate::Predictor`], [`crate::val::validate_detector`], and distillation
//! teachers support both `folo-*` and `folo2-*` end to end without breaking any existing `FoloModel` APIs.

use std::path::Path;

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};
use folo_nn::io::{LoadReport, Metadata, ModelConfig, ModelFile, Purpose};
use folo_nn::{Mode, Module, Result};

use crate::boxes::Xyxy;
use crate::model::v2::{DEFAULT_IOU_FUSION_ALPHA, FoloV2Config, FoloV2Model, HeadOutputV2};
use crate::model::{FoloModel, HeadMode, HeadOutput, quant};
use crate::nms::{Detection, NmsConfig};

/// Raw forward output of a [`Detector`]: standard [`HeadOutput`] plus optional FOLO v2 `iou_raw` logits.
#[derive(Clone)]
pub struct DetectorOutput {
    /// Box/class logits (`o2m` + optional training-mode `o2o`), anchors, and head metadata.
    pub head: HeadOutput,
    /// Optional `[B, 1, A]` localization-quality logits emitted by [`FoloV2Model`] when `iou_head == true`.
    pub iou_raw: Option<Tensor>,
}

impl DetectorOutput {
    /// Wrap a FOLO v1 [`HeadOutput`] (`iou_raw = None`).
    pub fn from_v1(head: HeadOutput) -> Self {
        Self { head, iou_raw: None }
    }

    /// Wrap a FOLO v2 [`HeadOutputV2`], preserving `iou_raw`.
    pub fn from_v2(out: HeadOutputV2) -> Self {
        let iou_raw = out.iou_raw.clone();
        Self { head: out.into_head_output(), iou_raw }
    }

    /// Decode `(boxes, scores)` per image, fusing `iou_raw` with exponent `iou_alpha` when present.
    pub fn decode_images(&self, iou_alpha: f32) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        if self.iou_raw.is_some() && iou_alpha > 0.0 {
            self.as_v2_output().decode_images(iou_alpha)
        } else {
            self.head.decode_images()
        }
    }

    /// Decode and post-process detections per image, fusing `iou_raw` with exponent `iou_alpha` when present.
    pub fn detections(&self, nms: &NmsConfig, iou_alpha: f32) -> Result<Vec<Vec<Detection>>> {
        if self.iou_raw.is_some() && iou_alpha > 0.0 {
            self.as_v2_output().detections(nms, iou_alpha)
        } else {
            self.head.detections(nms)
        }
    }

    /// Default IoU-fusion exponent ([`DEFAULT_IOU_FUSION_ALPHA`]).
    pub fn default_iou_alpha() -> f32 {
        DEFAULT_IOU_FUSION_ALPHA
    }

    fn as_v2_output(&self) -> HeadOutputV2 {
        HeadOutputV2 {
            box_raw: self.head.box_raw.clone(),
            cls_raw: self.head.cls_raw.clone(),
            iou_raw: self.iou_raw.clone(),
            mode: self.head.mode,
            o2o: self.head.o2o.clone(),
            anchors: self.head.anchors.clone(),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        }
    }
}

/// A FOLO detector: either v1 (`folo-n..x`) or v2-Iso (`folo2-n..x`).
pub enum Detector {
    /// FOLO v1 (`folo-n/s/m/l/x`).
    V1(Box<FoloModel>),
    /// FOLO v2-Iso (`folo2-n/s/m/l/x`).
    V2(Box<FoloV2Model>),
}

impl From<FoloModel> for Detector {
    fn from(m: FoloModel) -> Self {
        Detector::V1(Box::new(m))
    }
}

impl From<FoloV2Model> for Detector {
    fn from(m: FoloV2Model) -> Self {
        Detector::V2(Box::new(m))
    }
}

impl Detector {
    /// Construct a `V1` or `V2` detector from a validated `.folo` [`ModelConfig`] and RNG.
    pub fn from_model_config(mc: &ModelConfig, rng: &mut Rng) -> Result<Self> {
        mc.validate()?;
        if mc.is_v2() {
            let v2_cfg = FoloV2Config::from_model_config(mc)?;
            Ok(Detector::V2(Box::new(FoloV2Model::with_rng(v2_cfg, rng)?)))
        } else {
            Ok(Detector::V1(Box::new(FoloModel::with_rng(mc.clone(), rng)?)))
        }
    }

    /// Construct a `V1` or `V2` detector from a validated `.folo` [`ModelConfig`] using seed `0`.
    pub fn new(mc: &ModelConfig) -> Result<Self> {
        Self::from_model_config(mc, &mut Rng::seed(0))
    }

    /// Load a `.folo` file (auto-dispatching `folo-*` vs `folo2-*` from `folo.config.family`).
    pub fn from_file(file: &ModelFile, purpose: Purpose) -> Result<(Self, LoadReport)> {
        if file.metadata().config.is_v2() {
            let (m, r) = FoloV2Model::from_file(file, purpose)?;
            Ok((Detector::V2(Box::new(m)), r))
        } else {
            let (m, r) = FoloModel::from_file(file, purpose)?;
            Ok((Detector::V1(Box::new(m)), r))
        }
    }

    /// Open a `.folo` file for inference (auto-dispatching v1 vs v2).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::from_file(&ModelFile::open(path)?, Purpose::Inference)?.0)
    }

    /// Load a fused, eval-mode detector onto `device` at `dtype` (`F32` or `F16`).
    pub fn from_file_for_inference(file: &ModelFile, device: Device, dtype: DType) -> Result<Self> {
        if file.metadata().config.is_v2() {
            Ok(Detector::V2(Box::new(FoloV2Model::from_file_for_inference(file, device, dtype)?)))
        } else {
            Ok(Detector::V1(Box::new(FoloModel::from_file_for_inference(file, device, dtype)?)))
        }
    }

    /// Borrow the underlying FOLO v1 model, if this is `Detector::V1`.
    pub fn as_v1(&self) -> Option<&FoloModel> {
        match self {
            Detector::V1(m) => Some(m.as_ref()),
            Detector::V2(_) => None,
        }
    }

    /// Borrow the underlying FOLO v2 model, if this is `Detector::V2`.
    pub fn as_v2(&self) -> Option<&FoloV2Model> {
        match self {
            Detector::V1(_) => None,
            Detector::V2(m) => Some(m.as_ref()),
        }
    }

    /// True if this is a FOLO v2 (`folo2-*`) model.
    pub fn is_v2(&self) -> bool {
        matches!(self, Detector::V2(_))
    }

    /// Architecture identifier (`"folo-n"`, `"folo2-n"`, ...).
    pub fn arch(&self) -> String {
        match self {
            Detector::V1(m) => m.arch(),
            Detector::V2(m) => m.arch(),
        }
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        match self {
            Detector::V1(m) => m.num_classes(),
            Detector::V2(m) => m.num_classes(),
        }
    }

    /// Class names (`folo.classes`).
    pub fn classes(&self) -> &[String] {
        match self {
            Detector::V1(m) => m.classes(),
            Detector::V2(m) => m.classes(),
        }
    }

    /// Set class names (`folo.classes`).
    pub fn set_classes(&mut self, classes: Vec<String>) -> Result<()> {
        match self {
            Detector::V1(m) => m.set_classes(classes),
            Detector::V2(m) => m.set_classes(classes),
        }
    }

    /// Default image size (`folo.imgsz`).
    pub fn imgsz(&self) -> u32 {
        match self {
            Detector::V1(m) => m.imgsz(),
            Detector::V2(m) => m.imgsz(),
        }
    }

    /// Set default image size (`folo.imgsz`).
    pub fn set_imgsz(&mut self, imgsz: u32) {
        match self {
            Detector::V1(m) => m.set_imgsz(imgsz),
            Detector::V2(m) => m.set_imgsz(imgsz),
        }
    }

    /// Detection head strides (`[8, 16, 32]`).
    pub fn strides(&self) -> [u32; 3] {
        match self {
            Detector::V1(m) => [m.config().strides[0], m.config().strides[1], m.config().strides[2]],
            Detector::V2(m) => m.strides(),
        }
    }

    /// DFL bins per side (`reg_max`).
    pub fn reg_max(&self) -> usize {
        match self {
            Detector::V1(m) => m.head.reg_max,
            Detector::V2(m) => m.head.reg_max,
        }
    }

    /// Whether the model has the one-to-one head branch.
    pub fn has_o2o(&self) -> bool {
        match self {
            Detector::V1(m) => m.has_o2o(),
            Detector::V2(m) => m.has_o2o(),
        }
    }

    /// Active evaluation head mode.
    pub fn head_mode(&self) -> HeadMode {
        match self {
            Detector::V1(m) => m.head_mode(),
            Detector::V2(m) => m.head_mode(),
        }
    }

    /// Set active evaluation head mode.
    pub fn set_head_mode(&mut self, mode: HeadMode) -> Result<()> {
        match self {
            Detector::V1(m) => m.set_head_mode(mode),
            Detector::V2(m) => m.set_head_mode(mode),
        }
    }

    /// Set `o2o_detach` during training.
    pub fn set_o2o_detach(&mut self, detach: bool) {
        match self {
            Detector::V1(m) => m.head.o2o_detach = detach,
            Detector::V2(m) => m.head.o2o_detach = detach,
        }
    }

    /// Device holding the model weights.
    pub fn device(&self) -> Device {
        match self {
            Detector::V1(m) => m.device(),
            Detector::V2(m) => m.device(),
        }
    }

    /// Floating dtype of the model weights.
    pub fn dtype(&self) -> DType {
        match self {
            Detector::V1(m) => m.dtype(),
            Detector::V2(m) => m.dtype(),
        }
    }

    /// Move weights and buffers to `device` in place.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        match self {
            Detector::V1(m) => m.to_device(device),
            Detector::V2(m) => m.to_device(device),
        }
    }

    /// Cast floating weights and buffers to `dtype` in place.
    pub fn to_dtype(&mut self, dtype: DType) -> Result<()> {
        match self {
            Detector::V1(m) => Module::to_dtype(m.as_mut(), dtype),
            Detector::V2(m) => m.to_dtype(dtype),
        }
    }

    /// True once Conv+BN (and v2 `WS-QARepConv3x3`) are fused.
    pub fn is_fused(&self) -> bool {
        match self {
            Detector::V1(m) => m.is_fused(),
            Detector::V2(m) => m.is_fused(),
        }
    }

    /// Fold BatchNorms (and v2 `WS-QARepConv3x3` multi-branch convs) in place.
    pub fn fuse(&mut self) -> Result<()> {
        match self {
            Detector::V1(m) => m.fuse(),
            Detector::V2(m) => m.fuse(),
        }
    }

    /// Detached, fused, eval-mode clone on the same device and dtype.
    pub fn fused_eval_clone(&self) -> Result<Self> {
        match self {
            Detector::V1(m) => Ok(Detector::V1(Box::new(m.fused_eval_clone()?))),
            Detector::V2(m) => Ok(Detector::V2(Box::new(m.fused_eval_clone()?))),
        }
    }

    /// `.folo` weights metadata.
    pub fn metadata(&self) -> Result<Metadata> {
        match self {
            Detector::V1(m) => m.metadata(),
            Detector::V2(m) => m.metadata(),
        }
    }

    /// Partial transfer load from `file`.
    pub fn transfer_from(&self, file: &ModelFile) -> Result<LoadReport> {
        match self {
            Detector::V1(m) => m.transfer_from(file),
            Detector::V2(m) => m.transfer_from(file),
        }
    }

    /// Enable or disable GPU forward plans (no-op on `V2`).
    pub fn set_forward_plans(&self, on: bool) {
        if let Detector::V1(m) = self {
            m.set_forward_plans(on);
        }
    }

    /// Enable or disable CPU int8 calibration plan (`None` is a no-op on `V2`).
    pub fn set_int8(&mut self, cal: Option<quant::Calibration>) -> Result<()> {
        match self {
            Detector::V1(m) => m.set_int8(cal),
            Detector::V2(_) => {
                if cal.is_some() {
                    Err(folo_nn::Error::Invalid(
                        "int8 calibration plan is not yet supported for folo2".into(),
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Switch backbone BatchNorms to eval mode (for `freeze_backbone_epochs`).
    pub fn eval_backbone(&self) {
        match self {
            Detector::V1(m) => m.backbone.eval(),
            Detector::V2(m) => m.backbone.eval(),
        }
    }

    /// All trainable parameters inside `backbone.*`.
    pub fn backbone_parameters(&self) -> Vec<Tensor> {
        match self {
            Detector::V1(m) => m.backbone.parameters(),
            Detector::V2(m) => m.backbone.parameters(),
        }
    }

    /// Run forward pass using active `head_mode()`.
    pub fn forward(&self, x: &Tensor) -> Result<DetectorOutput> {
        self.forward_head(x, self.head_mode())
    }

    /// Run forward pass selecting `mode` in eval mode.
    pub fn forward_head(&self, x: &Tensor, mode: HeadMode) -> Result<DetectorOutput> {
        match self {
            Detector::V1(m) => Ok(DetectorOutput::from_v1(m.forward_head(x, mode)?)),
            Detector::V2(m) => Ok(DetectorOutput::from_v2(m.forward_head(x, mode)?)),
        }
    }
}

impl Module for Detector {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        match self {
            Detector::V1(m) => m.visit_children(f),
            Detector::V2(m) => m.visit_children(f),
        }
    }

    fn mode(&self) -> Option<&Mode> {
        match self {
            Detector::V1(m) => m.mode(),
            Detector::V2(m) => m.mode(),
        }
    }
}
