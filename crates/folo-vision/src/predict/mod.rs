//! The predictor: `.folo` weights → detections in original-image pixels (docs/MODEL.md §9).
//!
//! [`Predictor::open`] loads the EMA weights when the file has them (a checkpoint or an export), fuses Conv+BN
//! (§9.1) and keeps the model in eval mode. Each call letterboxes (§9.2: `rect` padding to multiples of 32 for
//! single images and for batches whose images share that canvas, a square `imgsz` canvas for other batches), runs
//! the forward pass, decodes and NMSes (§9.3) and maps the boxes back through the letterbox, clipped to the image. [`Timing`] splits the wall time into
//! preprocess / inference / postprocess like docs/API.md's `/api/predict`.
//!
//! On a GPU the whole pipeline stays on the device: the decoded `u8` images are uploaded as they are and
//! letterboxed there (`Tensor::letterbox_rgb8`, bit-exact with the CPU letterbox), and the head is decoded
//! there into above-threshold candidates, so only those come back ([`crate::model::HeadOutput::detections`]).
//! GPU work is queued asynchronously, so [`Predictor::predict`] and [`Predictor::predict_batch`] wait for the
//! device after the letterbox and after the forward pass: each stage of [`Timing`] holds its own GPU work, and
//! the wait for the forward pass counts as inference, not postprocess (the two waits cost a few tens of
//! microseconds of pipelining between stages).
//!
//! **Half precision.** [`Predictor::new_f16`] / [`Predictor::open_on`] with [`Precision::F16`] run the whole
//! model in f16 on a GPU that computes in it (`shader-f16`): f16 weights and activations, f32 accumulation in
//! every GEMM, convolution, reduction and softmax, the letterbox writing the f16 input and the candidate kernel
//! reading the f16 head. Where f16 is not available (the CPU, an adapter without `shader-f16`,
//! `FOLO_GPU_NO_F16=1`) the predictor runs in f32 and [`Predictor::fallback`] says why.

mod config;
mod letterbox;

use std::collections::BTreeMap;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_core::{DType, Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions};
use folo_nn::io::ModelFile;
use rayon::prelude::*;

pub use config::{Precision, PredictConfig, Prediction, Timing, f16_unavailable};
pub use letterbox::{letterbox_on_device, letterbox_on_device_as, letterbox_on_host};
use letterbox::{letterbox_on_device_as_refs, letterbox_on_host_refs, letterbox_on_host_rgb0};
pub(crate) use letterbox::{letterbox_uploaded_on_device, upload_images_on_device};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::eval::{ThresholdCalibration, validate_per_class_conf};
use crate::model::v2::{DEFAULT_IOU_FUSION_ALPHA, FoloV2Model};
use crate::model::{Detector, DetectorOutput, FoloModel, HeadOutput};
use crate::nms::{Detection, NmsConfig, postprocess};

/// Per-call confidence override passed internally by [`Predictor::predict_sequence`] so ByteTrack can lower
/// fallback and per-class thresholds independently without mutating `Predictor`.
pub(crate) struct ConfOverride {
    pub conf: f32,
    pub per_class_conf: Option<BTreeMap<u32, f32>>,
    pub conf_inclusive: bool,
}

/// A fused, eval-mode detector ready for inference (`folo-*` or `folo2-*`). See the module docs.
pub struct Predictor {
    model: Detector,
    cfg: PredictConfig,
    per_class_conf: Option<BTreeMap<u32, f32>>,
    conf_inclusive: bool,
    fallback: Option<String>,
}

impl Predictor {
    /// Load `.folo` weights (`folo-*` or `folo2-*`; EMA when present; weights, exports and checkpoints all work)
    /// and fuse them.
    pub fn open(path: impl AsRef<Path>, cfg: PredictConfig) -> Result<Self> {
        Self::from_detector(Detector::open(path)?, cfg)
    }

    /// Load a quantized `.folo` file (`folo-*` or `folo2-*`) carrying `folo.quant` metadata for CPU int8
    /// inference.
    pub fn open_int8(path: impl AsRef<Path>, cfg: PredictConfig) -> Result<Self> {
        Self::from_detector(Detector::open_int8(path)?, cfg)
    }

    /// Load `.folo` weights (`folo-*` or `folo2-*`) onto `device` at `precision`
    /// ([`Detector::from_file_for_inference`]: an f16 export goes to the GPU as f16, other files are fused in f32
    /// and converted). When f16 cannot run on `device` the model is loaded in f32 and [`Predictor::fallback`] says
    /// why.
    pub fn open_on(
        path: impl AsRef<Path>,
        cfg: PredictConfig,
        device: Device,
        precision: Precision,
    ) -> Result<Self> {
        cfg.validate()?;
        let fallback = if precision == Precision::F16 { f16_unavailable(device) } else { None };
        let dtype = if fallback.is_some() { DType::F32 } else { precision.dtype() };
        let model = Detector::from_file_for_inference(&ModelFile::open(path)?, device, dtype)?;
        let mut p = Self::from_detector(model, cfg)?;
        p.fallback = fallback;
        Ok(p)
    }

    /// Wrap a FOLO v1 model: it is fused (a no-op if it already is) and stays in eval mode.
    pub fn new(model: FoloModel, cfg: PredictConfig) -> Result<Self> {
        Self::from_detector(Detector::from(model), cfg)
    }

    /// Wrap a FOLO v2-Iso model ([`FoloV2Model`]): folds all `WS-QARepConv3x3` branches and BatchNorms and stays
    /// in eval mode. Supports `predict`, `predict_batch`, SAHI (`predict_tiled`), TTA (`predict_tta`), `o2m`/`o2o`,
    /// and `iou_head` score fusion.
    pub fn new_v2(model: FoloV2Model, cfg: PredictConfig) -> Result<Self> {
        Self::from_detector(Detector::from(model), cfg)
    }

    /// Wrap any [`Detector`] (`folo-*` or `folo2-*`): it is fused (a no-op if it already is) and stays in eval
    /// mode. The model keeps its dtype (an f16 model predicts in f16). GPU forwards and candidate-first site passes
    /// are recorded and replayed on both v1 and v2 models ([`Detector::set_forward_plans`]).
    pub fn from_detector(mut model: Detector, cfg: PredictConfig) -> Result<Self> {
        cfg.validate()?;
        if let Some(h) = cfg.head {
            model.set_head_mode(h)?;
        }
        model.fuse()?;
        model.set_forward_plans(true);
        Ok(Predictor { model, cfg, per_class_conf: None, conf_inclusive: false, fallback: None })
    }

    /// [`Predictor::new`] in half precision: the model is fused (in f32) and converted to f16 on its device. On
    /// a device without f16 compute ([`f16_unavailable`]) it stays f32 and [`Predictor::fallback`] says why.
    pub fn new_f16(model: FoloModel, cfg: PredictConfig) -> Result<Self> {
        Self::with_precision(model, cfg, Precision::F16)
    }

    /// [`Predictor::new`] at `precision` (see [`Predictor::new_f16`]). The whole model runs at `precision`: an
    /// int8 plan ([`FoloModel::set_int8`]) is turned off; [`Predictor::new`] keeps it.
    pub fn with_precision(model: FoloModel, cfg: PredictConfig, precision: Precision) -> Result<Self> {
        Self::from_detector_with_precision(Detector::from(model), cfg, precision)
    }

    /// [`Predictor::from_detector`] at `precision` (`F32` or `F16` with automatic fallback on devices without
    /// `shader-f16`).
    pub fn from_detector_with_precision(
        model: Detector,
        cfg: PredictConfig,
        precision: Precision,
    ) -> Result<Self> {
        let mut p = Self::from_detector(model, cfg)?;
        p.model.set_int8(None)?;
        let fallback = if precision == Precision::F16 { f16_unavailable(p.device()) } else { None };
        let dtype = if fallback.is_some() { DType::F32 } else { precision.dtype() };
        p.model.to_dtype(dtype)?;
        p.fallback = fallback;
        Ok(p)
    }

    /// The precision the model runs in.
    pub fn precision(&self) -> Precision {
        if self.model.dtype() == DType::F16 { Precision::F16 } else { Precision::F32 }
    }

    /// Why an f16 request runs in f32 ([`f16_unavailable`]), or `None` when it did not fall back.
    pub fn fallback(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    /// The (fused) v1 model (panics on a `folo2-*` predictor — use [`Predictor::detector`] or
    /// [`Predictor::model_v2`] for architecture-agnostic / v2 access).
    pub fn model(&self) -> &FoloModel {
        self.model
            .as_v1()
            .expect("Predictor::model() requires a FOLO v1 model; use Predictor::detector() or Predictor::model_v2() for folo2-*")
    }

    /// The (fused) detector (`folo-*` or `folo2-*`).
    pub fn detector(&self) -> &Detector {
        &self.model
    }

    /// The (fused) FOLO v2 model, if predicting with `folo2-*`.
    pub fn model_v2(&self) -> Option<&FoloV2Model> {
        self.model.as_v2()
    }

    /// Model architecture identifier (`"folo-n"`, `"folo2-n"`, ...).
    pub fn arch(&self) -> String {
        self.model.arch()
    }

    /// Model default image size (`folo.imgsz`).
    pub fn imgsz(&self) -> u32 {
        self.model.imgsz()
    }

    /// Class names.
    pub fn classes(&self) -> &[String] {
        self.model.classes()
    }

    /// The settings.
    pub fn config(&self) -> &PredictConfig {
        &self.cfg
    }

    /// Change the settings. `head: None` keeps the branch the predictor runs now.
    pub fn set_config(&mut self, cfg: PredictConfig) -> Result<()> {
        cfg.validate()?;
        if let Some(h) = cfg.head {
            self.model.set_head_mode(h)?;
        }
        self.cfg = cfg;
        Ok(())
    }

    /// Optional per-class confidence thresholds currently active on this predictor (`class_id -> conf`).
    pub fn per_class_conf(&self) -> Option<&BTreeMap<u32, f32>> {
        self.per_class_conf.as_ref()
    }

    /// Set or clear optional per-class confidence thresholds (`class_id -> conf`). Every entry must be finite and
    /// in `0.0..=1.0`; classes absent from the map continue to use [`PredictConfig::conf`].
    pub fn set_per_class_conf(&mut self, thresholds: Option<BTreeMap<u32, f32>>) -> Result<()> {
        if let Some(map) = &thresholds {
            validate_per_class_conf(map)?;
        }
        self.per_class_conf = thresholds;
        Ok(())
    }

    /// Builder attaching per-class confidence thresholds (`class_id -> conf`) to this predictor.
    pub fn with_per_class_conf(mut self, thresholds: BTreeMap<u32, f32>) -> Result<Self> {
        self.set_per_class_conf(Some(thresholds))?;
        Ok(self)
    }

    /// Whether confidence thresholding uses inclusive (`score >= conf`) final-score semantics (enabled
    /// automatically by [`Predictor::apply_calibrated_thresholds`] / [`Predictor::with_calibrated_thresholds`]).
    pub fn conf_inclusive(&self) -> bool {
        self.conf_inclusive
    }

    /// Explicitly enable or disable inclusive (`score >= conf`) final-score thresholding
    /// ([`NmsConfig::conf_inclusive`]).
    pub fn set_conf_inclusive(&mut self, inclusive: bool) {
        self.conf_inclusive = inclusive;
    }

    /// Apply a [`ThresholdCalibration`] (from [`crate::EvalReport::thresholds`] or [`crate::val::ValResult::thresholds`])
    /// to this predictor. When the calibration has at least one calibrated class, sets [`PredictConfig::conf`] to
    /// `cal.global.best_conf` (as fallback for uncalibrated classes), installs `cal.per_class_map()`, and enables
    /// inclusive final-score thresholding ([`NmsConfig::conf_inclusive`]) so boundary detections (`score == best_conf`,
    /// including `0.0` and `1.0`) and rescored detections match the evaluator's operating point exactly. When `cal`
    /// is empty (no class had ground truth), leaves [`PredictConfig::conf`] unchanged and clears per-class overrides.
    pub fn apply_calibrated_thresholds(&mut self, cal: &ThresholdCalibration) -> Result<()> {
        cal.validate()?;
        if cal.is_empty() {
            self.per_class_conf = None;
            self.conf_inclusive = false;
        } else {
            self.cfg.conf = cal.global.best_conf;
            self.per_class_conf = Some(cal.per_class_map());
            self.conf_inclusive = true;
        }
        Ok(())
    }

    /// Builder applying a [`ThresholdCalibration`] ([`Predictor::apply_calibrated_thresholds`]).
    pub fn with_calibrated_thresholds(mut self, cal: &ThresholdCalibration) -> Result<Self> {
        self.apply_calibrated_thresholds(cal)?;
        Ok(self)
    }

    /// Build the effective [`NmsConfig`] for inference, attaching optional per-class thresholds and applying any
    /// internal [`ConfOverride`] (e.g. ByteTrack independent per-class stage-2 floors).
    pub(crate) fn nms_config(&self, conf_override: Option<&ConfOverride>) -> NmsConfig {
        let mut nms = self.cfg.nms();
        if let Some(ov) = conf_override {
            nms.conf = ov.conf;
            nms.per_class_conf = ov.per_class_conf.clone();
            nms.conf_inclusive = ov.conf_inclusive;
        } else {
            nms.per_class_conf = self.per_class_conf.clone();
            nms.conf_inclusive = self.conf_inclusive;
        }
        nms
    }

    /// The device the model runs on.
    pub fn device(&self) -> Device {
        self.model.device()
    }

    /// Decode an image file and predict it.
    pub fn predict_file(&self, path: impl AsRef<Path>) -> Result<Prediction> {
        let t = Instant::now();
        let img = folo_data::read_image(path.as_ref())?;
        let decode_ms = ms(t);
        let mut p = self.predict(&img)?;
        p.timing =
            Timing::new(p.timing.preprocess_ms + decode_ms, p.timing.inference_ms, p.timing.postprocess_ms);
        Ok(p)
    }

    /// One image, letterboxed `rect` (or square when [`PredictConfig::rect`] is off).
    pub fn predict(&self, img: &Image) -> Result<Prediction> {
        Ok(self.run(std::slice::from_ref(img), self.cfg.rect)?.pop().expect("one image in, one out"))
    }

    /// A batch, in input order. With [`PredictConfig::rect`] (the default) a batch whose images all get the same
    /// `rect` canvas (frames of one camera, say) is letterboxed to that canvas, the input [`Predictor::predict`]
    /// gives each image alone (on the CPU the predictions are then the same bit for bit; on a GPU a kernel may sum
    /// in another order at another batch size, so scores can differ in their last bits); any other batch is
    /// letterboxed to the square `imgsz × imgsz` canvas. (Running each canvas as its own pass measured slower than one square pass for mixed shapes on the
    /// CPU.) On a GPU it is one forward pass; on the CPU, chunks of two images run side by side (the same
    /// predictions as one pass). Each prediction's timing is its pass's (or chunk's) divided by its size; with
    /// chunks side by side the images' totals add up to more than the call's wall time (docs/API.md, `timing`).
    pub fn predict_batch(&self, imgs: &[Image]) -> Result<Vec<Prediction>> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }
        self.run(imgs, self.cfg.rect && one_canvas(imgs, self.cfg.imgsz))
    }

    fn run(&self, imgs: &[Image], rect: bool) -> Result<Vec<Prediction>> {
        let refs: Vec<&Image> = imgs.iter().collect();
        self.run_with_imgsz(&refs, rect, self.cfg.imgsz)
    }

    pub(crate) fn run_with_imgsz(&self, imgs: &[&Image], rect: bool, imgsz: u32) -> Result<Vec<Prediction>> {
        self.run_with_imgsz_conf(imgs, rect, imgsz, None)
    }

    pub(crate) fn run_with_imgsz_conf(
        &self,
        imgs: &[&Image],
        rect: bool,
        imgsz: u32,
        conf_override: Option<&ConfOverride>,
    ) -> Result<Vec<Prediction>> {
        /// Images per forward when the CPU predicts a batch (see below). One, two and four measured within
        /// 5% of each other at batch 16 on 28 threads, and all about 20% faster than one batched forward.
        const CPU_BATCH_CHUNK: usize = 2;
        // On the CPU every stage (letterbox, forward, decode) runs from inside the rayon pool.
        if self.model.device().is_cpu() {
            // A batch runs as chunks of about two images whose forwards run side by side (each op's tasks
            // spread over the pool by work stealing), which keeps every kernel nearer one core's efficiency
            // than one batched forward across all threads. Every image's result is the same either way (the
            // CPU kernels compute each image on its own, in an order that depends only on its shape).
            let chunk = CPU_BATCH_CHUNK;
            if imgs.len() > chunk {
                return folo_core::cpu::in_pool(|| {
                    let parts: Vec<Vec<Prediction>> = imgs
                        .par_chunks(chunk)
                        .map(|c| self.run_here_with_imgsz(c, rect, imgsz, conf_override))
                        .collect::<Result<_>>()?;
                    Ok(parts.into_iter().flatten().collect())
                });
            }
            folo_core::cpu::in_pool(|| self.run_here_with_imgsz(imgs, rect, imgsz, conf_override))
        } else {
            self.run_here_with_imgsz(imgs, rect, imgsz, conf_override)
        }
    }

    fn run_here_with_imgsz(
        &self,
        imgs: &[&Image],
        rect: bool,
        imgsz: u32,
        conf_override: Option<&ConfOverride>,
    ) -> Result<Vec<Prediction>> {
        if self.model.has_int8_u8_stem() {
            return self.run_here_rgb0_u8_with_imgsz(imgs, rect, imgsz, conf_override);
        }
        let t0 = Instant::now();
        let batch = self.prepare_with_imgsz(imgs, rect, imgsz)?;
        self.wait_for_device()?;
        let pre = ms(t0);
        self.run_prepared(batch, pre, conf_override)
    }

    /// Predict already-uploaded `[H, W, 3]` `u8` device images `dev_imgs` (parallel to `imgs`) at `imgsz`,
    /// reusing the resident GPU pixels instead of re-uploading them across scales (multi-scale GPU TTA).
    pub(crate) fn run_uploaded_with_imgsz(
        &self,
        dev_imgs: &[Tensor],
        imgs: &[&Image],
        rect: bool,
        imgsz: u32,
    ) -> Result<Vec<Prediction>> {
        if imgs.is_empty() {
            return Err(Error::Config("cannot predict an empty batch".into()));
        }
        if dev_imgs.len() != imgs.len() {
            return Err(Error::Config(format!(
                "run_uploaded_with_imgsz: {} device images for {} host images",
                dev_imgs.len(),
                imgs.len()
            )));
        }
        let t0 = Instant::now();
        let rect = rect && one_canvas_refs(imgs, imgsz);
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        for img in imgs {
            if img.width == 0 || img.height == 0 {
                return Err(Error::Config("cannot predict an empty image".into()));
            }
        }
        let letterboxes: Vec<Letterbox> =
            imgs.iter().map(|i| Letterbox::compute(i.width, i.height, imgsz, imgsz, &opts)).collect();
        let (device, dtype) = (self.model.device(), self.model.dtype());
        let input = match letterbox_uploaded_on_device(dev_imgs, imgs, &letterboxes, device, dtype)? {
            Some(x) => x,
            None => {
                // On-device letterbox was unavailable for these shapes/device; letterbox on host and upload
                // once instead of recursing into `run_here_with_imgsz` (which would re-upload raw images).
                let x = letterbox_on_host_refs(imgs, &letterboxes)?;
                let x = if device.is_cpu() { x } else { x.to_device(device)? };
                x.to_dtype(dtype)?
            }
        };
        self.wait_for_device()?;
        let pre = ms(t0);
        let sizes = imgs.iter().map(|i| [i.width, i.height]).collect();
        self.run_prepared(PreparedBatch { input, letterboxes, sizes }, pre, None)
    }

    fn run_prepared(
        &self,
        batch: PreparedBatch,
        pre: f64,
        conf_override: Option<&ConfOverride>,
    ) -> Result<Vec<Prediction>> {
        let nms = self.nms_config(conf_override);
        let n = batch.len() as f64;
        let (dets, inf, post) = if let Some(m) = self.model.as_v1()
            && m.candidates_first_applies(batch.input.device(), &nms)
        {
            // Candidate-first (MODEL.md §9.5): the box branch runs once the candidates are known, so the
            // candidate extraction and the candidates' box decode count as inference (docs/API.md, `timing`).
            let t1 = Instant::now();
            let cands = m.predict_candidates(&batch.input, &nms)?;
            let inf = ms(t1);
            let t2 = Instant::now();
            let dets = cands.into_par_iter().map(|c| m.finish_candidates(&c, &nms)).collect();
            (self.finish_detections(&batch, dets), inf, ms(t2))
        } else if let Some(m) = self.model.as_v2()
            && m.candidates_first_applies(batch.input.device(), &nms)
        {
            let t1 = Instant::now();
            let cands = m.predict_candidates(&batch.input, &nms)?;
            let inf = ms(t1);
            let t2 = Instant::now();
            let dets = cands.into_par_iter().map(|c| m.finish_candidates(&c, &nms)).collect();
            (self.finish_detections(&batch, dets), inf, ms(t2))
        } else {
            let t1 = Instant::now();
            let out = self.forward_detector(&batch)?;
            self.wait_for_device()?;
            let inf = ms(t1);
            // docs/API.md counts the decode as post-processing.
            let t2 = Instant::now();
            let dets = self.finish_detections(&batch, out.detections(&nms, DEFAULT_IOU_FUSION_ALPHA)?);
            (dets, inf, ms(t2))
        };
        Ok(batch.predictions(dets, Timing::new(pre / n, inf / n, post / n)))
    }

    /// Direct `u8` RGB0 path for CPU int8 models (`folo-*` and `folo2-*`): letterboxes directly into packed
    /// `[R, G, B, 0]` `u32` words (zero `f32` canvas materialisation, bit-exact with `apply_planar_f32` +
    /// `pack_f32_to_rgb0_u8`). Validates the int8 plan immediately before execution so any concurrent weight
    /// write that invalidates calibration during letterboxing falls back cleanly to float inference.
    fn run_here_rgb0_u8_with_imgsz(
        &self,
        imgs: &[&Image],
        rect: bool,
        s: u32,
        conf_override: Option<&ConfOverride>,
    ) -> Result<Vec<Prediction>> {
        if imgs.is_empty() {
            return Err(Error::Config("cannot predict an empty batch".into()));
        }
        let t0 = Instant::now();
        let rect = rect && one_canvas_refs(imgs, s);
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        for img in imgs {
            if img.width == 0 || img.height == 0 {
                return Err(Error::Config("cannot predict an empty image".into()));
            }
        }
        let letterboxes: Vec<Letterbox> =
            imgs.iter().map(|i| Letterbox::compute(i.width, i.height, s, s, &opts)).collect();
        let (rgb0, h, w) = letterbox_on_host_rgb0(imgs, &letterboxes)?;
        let Some(plan) = self.model.int8_u8_stem_plan() else {
            folo_core::cpu::q8::recycle(rgb0);
            let batch = self.prepare_with_imgsz(imgs, rect, s)?;
            self.wait_for_device()?;
            return self.run_prepared(batch, ms(t0), conf_override);
        };
        let pre = ms(t0);
        let nms = self.nms_config(conf_override);
        let b = imgs.len();
        let (dets, inf, post) = match &self.model {
            Detector::V1(model) => {
                if model.candidates_first_applies(Device::Cpu, &nms) {
                    let t1 = Instant::now();
                    let cands = model.predict_candidates_rgb0_u8_with_plan(&plan, &rgb0, b, h, w, &nms)?;
                    let inf = ms(t1);
                    let t2 = Instant::now();
                    let dets = cands.into_par_iter().map(|c| model.finish_candidates(&c, &nms)).collect();
                    (dets, inf, ms(t2))
                } else {
                    let t1 = Instant::now();
                    let out = {
                        let _g = no_grad();
                        model.forward_rgb0_u8_with_plan(&plan, &rgb0, b, h, w)?
                    };
                    let inf = ms(t1);
                    let t2 = Instant::now();
                    let dets = out.detections(&nms)?;
                    (dets, inf, ms(t2))
                }
            }
            Detector::V2(model) => {
                if model.candidates_first_applies(Device::Cpu, &nms) {
                    let t1 = Instant::now();
                    let cands = model.predict_candidates_rgb0_u8_with_plan(&plan, &rgb0, b, h, w, &nms)?;
                    let inf = ms(t1);
                    let t2 = Instant::now();
                    let dets = cands.into_par_iter().map(|c| model.finish_candidates(&c, &nms)).collect();
                    (dets, inf, ms(t2))
                } else {
                    let t1 = Instant::now();
                    let out = {
                        let _g = no_grad();
                        model.forward_rgb0_u8_with_plan(&plan, &rgb0, b, h, w)?
                    };
                    let inf = ms(t1);
                    let t2 = Instant::now();
                    let dets = out.detections(&nms, DEFAULT_IOU_FUSION_ALPHA)?;
                    (dets, inf, ms(t2))
                }
            }
        };
        folo_core::cpu::q8::recycle(rgb0);
        let t_unmap = Instant::now();
        let mapped: Vec<Vec<Detection>> = dets
            .into_par_iter()
            .zip(letterboxes.par_iter())
            .map(|(d, lb)| {
                d.into_iter()
                    .map(|d| {
                        let [x1, y1, x2, y2] = lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]);
                        Detection { bbox: Xyxy::new(x1, y1, x2, y2), ..d }
                    })
                    .collect()
            })
            .collect();
        let post = post + ms(t_unmap);
        let n = imgs.len() as f64;
        let timing = Timing::new(pre / n, inf / n, post / n);
        Ok(imgs
            .iter()
            .zip(mapped)
            .map(|(img, detections)| Prediction { width: img.width, height: img.height, detections, timing })
            .collect())
    }

    /// Waits for the work queued on the model's device (a no-op on the CPU), so a stage's timer covers its
    /// GPU work instead of the next stage's first readback.
    fn wait_for_device(&self) -> Result<()> {
        let device = self.model.device();
        if !device.is_cpu() {
            folo_core::backend::synchronize(device)?;
        }
        Ok(())
    }

    /// The first step of a prediction, for callers that drive the steps themselves (the browser, which
    /// must await the GPU readback between [`Predictor::forward`] and [`Predictor::finish`]): letterbox the
    /// images (`rect` padding for one image or for images that share their `rect` canvas, else the square canvas)
    /// into the input tensor, on the model's
    /// device (on a GPU from the raw `u8` images, [`letterbox_on_device`]). [`Predictor::predict`] is
    /// `prepare` → `forward` → [`HeadOutput::detections`] → [`Predictor::finish_detections`]; [`Predictor::finish`]
    /// takes decoded outputs instead.
    pub fn prepare(&self, imgs: &[Image], rect: bool) -> Result<PreparedBatch> {
        let refs: Vec<&Image> = imgs.iter().collect();
        self.prepare_with_imgsz(&refs, rect, self.cfg.imgsz)
    }

    fn prepare_with_imgsz(&self, imgs: &[&Image], rect: bool, s: u32) -> Result<PreparedBatch> {
        if imgs.is_empty() {
            return Err(Error::Config("cannot predict an empty batch".into()));
        }
        let rect = rect && one_canvas_refs(imgs, s);
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        for img in imgs {
            if img.width == 0 || img.height == 0 {
                return Err(Error::Config("cannot predict an empty image".into()));
            }
        }
        let letterboxes: Vec<Letterbox> =
            imgs.iter().map(|i| Letterbox::compute(i.width, i.height, s, s, &opts)).collect();
        // `rect` is only used for images that share their canvas, so every canvas of a call has the same size.
        let (device, dtype) = (self.model.device(), self.model.dtype());
        let on_device = if device.is_cpu() {
            None
        } else {
            letterbox_on_device_as_refs(imgs, &letterboxes, device, dtype)?
        };
        let input = match on_device {
            Some(x) => x,
            None => {
                let x = letterbox_on_host_refs(imgs, &letterboxes)?;
                let x = if device.is_cpu() { x } else { x.to_device(device)? };
                x.to_dtype(dtype)?
            }
        };
        let sizes = imgs.iter().map(|i| [i.width, i.height]).collect();
        Ok(PreparedBatch { input, letterboxes, sizes })
    }

    /// The forward pass of a [`PreparedBatch`] under `no_grad`. On a GPU the result is queued work: nothing
    /// waits until its outputs are read back.
    pub fn forward(&self, batch: &PreparedBatch) -> Result<HeadOutput> {
        Ok(self.forward_detector(batch)?.head)
    }

    /// Like [`Predictor::forward`], preserving optional FOLO v2 `iou_raw` logits in [`DetectorOutput`].
    pub fn forward_detector(&self, batch: &PreparedBatch) -> Result<DetectorOutput> {
        let _g = no_grad();
        Ok(self.model.forward(&batch.input)?)
    }

    /// The last step: class-aware NMS (with the current [`PredictConfig`]; the top-k of docs/MODEL.md §6.6 when
    /// the model predicts with its `o2o` head, [`FoloModel::head_mode`]) on each image's decoded
    /// `(boxes, scores)` (from [`HeadOutput::decode_images`] or [`HeadOutput::split_decoded`]), then the boxes
    /// mapped back through the letterbox into original-image pixels. `nc` is the head's class count. Decoded boxes
    /// carry no DFL logits or anchor strides, so [`BoxFusion::Dfl`] and [`BoxFusion::Sebf`] omit per-side DFL
    /// precision weights (fusing as [`BoxFusion::Vote`] / unweighted `Sebf`), [`ScoreRule::DflConsensus`] scores as
    /// [`ScoreRule::Elc`], and [`ScoreRule::DflRepulse`] / [`ScoreRule::DflCgr`] omit their entropy and cross-stride
    /// terms here.
    pub fn finish(
        &self,
        batch: &PreparedBatch,
        decoded: Vec<(Vec<Xyxy>, Vec<f32>)>,
        nc: usize,
    ) -> Vec<Vec<Detection>> {
        let (nms, mode) = (self.nms_config(None), self.model.head_mode());
        let dets = decoded.into_par_iter().map(|(b, sc)| postprocess(&b, &sc, nc, &nms, mode)).collect();
        self.finish_detections(batch, dets)
    }

    /// The last step for detections in network-input pixels (from [`HeadOutput::detections`] with this predictor's
    /// NMS settings): each image's boxes mapped back through its letterbox.
    pub fn finish_detections(&self, batch: &PreparedBatch, dets: Vec<Vec<Detection>>) -> Vec<Vec<Detection>> {
        dets.into_par_iter()
            .zip(batch.letterboxes.par_iter())
            .map(|(d, lb)| {
                d.into_iter()
                    .map(|d| {
                        let [x1, y1, x2, y2] = lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]);
                        Detection { bbox: Xyxy::new(x1, y1, x2, y2), ..d }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Letterboxed images and their input tensor ([`Predictor::prepare`]).
pub struct PreparedBatch {
    /// `[B, 3, H, W]` in `0..1`, on the model's device, in the model's dtype.
    pub input: Tensor,
    letterboxes: Vec<Letterbox>,
    sizes: Vec<[u32; 2]>,
}

impl PreparedBatch {
    /// Images in the batch.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    /// Whether the batch is empty (never, for a batch from [`Predictor::prepare`]).
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// One [`Prediction`] per image from [`Predictor::finish`]'s detections and a per-image timing.
    pub fn predictions(&self, dets: Vec<Vec<Detection>>, timing: Timing) -> Vec<Prediction> {
        self.sizes
            .iter()
            .zip(dets)
            .map(|(&[width, height], detections)| Prediction { width, height, detections, timing })
            .collect()
    }
}

/// Letterbox canvas `[width, height]` that an image of size `w × h` receives for target `s` and `rect` mode.
/// Used by batched SAHI to group tiles and full-image passes that share one `rect` canvas without changing
/// any image's letterbox geometry.
#[inline]
pub(crate) fn rect_canvas(w: u32, h: u32, s: u32, rect: bool) -> [u32; 2] {
    if rect {
        let opts = LetterboxOptions { stride: Some(32), ..LetterboxOptions::default() };
        Letterbox::compute(w, h, s, s, &opts).out
    } else {
        [s, s]
    }
}

/// Whether every image of `imgs` gets the same `rect` letterbox canvas for target `s` (§9.2: the smallest multiple
/// of 32 around the resized image), so they can share one `rect` batch. An empty image never shares one.
pub(crate) fn one_canvas(imgs: &[Image], s: u32) -> bool {
    let opts = LetterboxOptions { stride: Some(32), ..LetterboxOptions::default() };
    let canvas = |i: &Image| {
        (i.width > 0 && i.height > 0).then(|| Letterbox::compute(i.width, i.height, s, s, &opts).out)
    };
    let Some((first, rest)) = imgs.split_first() else { return false };
    let first = canvas(first);
    first.is_some() && rest.iter().all(|i| canvas(i) == first)
}

fn one_canvas_refs(imgs: &[&Image], s: u32) -> bool {
    let opts = LetterboxOptions { stride: Some(32), ..LetterboxOptions::default() };
    let canvas = |i: &Image| {
        (i.width > 0 && i.height > 0).then(|| Letterbox::compute(i.width, i.height, s, s, &opts).out)
    };
    let Some((&first, rest)) = imgs.split_first() else { return false };
    let first = canvas(first);
    first.is_some() && rest.iter().all(|&i| canvas(i) == first)
}

/// Scale accumulated per-view `(pre, inf, post)` timings to actual `wall_ms` so parallel CPU sub-batches never
/// double-count overlapping wall intervals while keeping stage proportions faithful.
#[inline]
pub(crate) fn add_scaled_timing(
    pre_ms: &mut f64,
    inf_ms: &mut f64,
    post_ms: &mut f64,
    b_pre: f64,
    b_inf: f64,
    b_post: f64,
    wall_ms: f64,
) {
    let b_sum = b_pre + b_inf + b_post;
    if b_sum > 0.0 {
        let scale = wall_ms / b_sum;
        *pre_ms += b_pre * scale;
        *inf_ms += b_inf * scale;
        *post_ms += b_post * scale;
    } else {
        *inf_ms += wall_ms;
    }
}

pub(crate) fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
