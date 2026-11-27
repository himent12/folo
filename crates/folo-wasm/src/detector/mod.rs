//! The detector behind the JS bindings, free of `wasm-bindgen` types so the workspace's native tests run it.
//!
//! [`Detector::load`] parses a `.folo` file, builds the fused eval-mode model and places it on WebGPU or the CPU
//! (`auto` tries WebGPU and checks it with one small forward pass, falling back to the CPU).
//!
//! - [`Detector::detect`] runs single-pass inference (`folo_vision::Predictor` pipeline):
//!   - on WebGPU, RGBA pixels are stripped into a reusable RGB scratch buffer, uploaded, and decoded on the
//!     device (`Tensor::detect_candidates`) so only compact candidate records come back via an awaited
//!     [`folo_gpu::download_async`] (~28 KiB per frame instead of ~2.8 MiB for the dense `[B, A, 4 + nc]` head),
//!     falling back to dense readback only when a backend lacks the candidate kernel;
//!   - on the CPU, RGBA pixels are letterboxed directly into planar `f32` using reusable scratch buffers (and
//!     wasm SIMD128 when built with `+simd128`) without an intermediate RGB image copy, then run through
//!     candidate-first / sparse decode.
//! - [`Detector::detect_tiled`] ([`tiled`]) runs SAHI overlapping tiled inference with optional full-image pass
//!   and deterministic IoS/IoU merging (`folo_vision::tile`).
//! - [`Detector::detect_tta`] ([`tta`]) runs multi-scale and/or horizontal-flip Test-Time Augmentation with
//!   deterministic cross-view merging (`folo_vision::tta`).

mod options;
mod tiled;
mod tta;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

use folo_core::backend::DETECT_RECORD_WORDS;
use folo_core::{DType, DetectDecode, Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions, PAD_VALUE};
use folo_gpu::GpuContext;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::boxes::Xyxy;
use folo_vision::model::HeadMode;
use folo_vision::model::v2::fuse_cls_iou_scores;
use folo_vision::nms::{Candidate, Detection, NmsConfig, nms_candidates, topk_candidates};
use folo_vision::{
    Detector as VisionDetector, DetectorOutput, HeadOutput, Precision, PredictConfig, Predictor, Timing,
};

use crate::simd::{RgbaScratch, letterbox_rgba_planar_f32_with_scratch, rgba_to_rgb};

pub use options::{
    Backend, BackendChoice, DetectOptions, DetectResult, DetectorInfo, LoadOptions, TileDetectOptions,
    TtaDetectOptions,
};
pub use tiled::MAX_BROWSER_TILES;

/// `u8` pixel → network input (`folo_vision::predict::INV_255`).
const INV_255: f32 = 1.0 / 255.0;

/// Records per image in the first async GPU candidate download (`folo_vision`'s `CANDIDATE_READBACK`).
/// Counts plus up to this many candidates return in one readback; a second readback runs only on overflow.
const CANDIDATE_READBACK: usize = 1024;

/// Cached `[3, A]` GPU anchor tensor for one network input canvas `[H, W]`.
struct CachedAnchors {
    canvas: [usize; 2],
    num_anchors: usize,
    tensor: Tensor,
}

/// Typed GPU batch error preserving whether a failure is an OOM / device-buffer-limit error eligible for
/// chunk-halving fallback.
pub(crate) enum BatchError {
    Oom(String),
    Other(String),
}

impl BatchError {
    fn into_string(self) -> String {
        match self {
            BatchError::Oom(s) | BatchError::Other(s) => s,
        }
    }

    fn from_vision(e: folo_vision::Error) -> Self {
        let is_oom = e.is_gpu_out_of_memory()
            || matches!(&e, folo_vision::Error::Core(ce) | folo_vision::Error::Nn(folo_nn::Error::Core(ce)) if is_core_gpu_oom(ce));
        if is_oom { BatchError::Oom(e.to_string()) } else { BatchError::Other(e.to_string()) }
    }

    fn from_core(e: folo_core::Error) -> Self {
        if is_core_gpu_oom(&e) { BatchError::Oom(e.to_string()) } else { BatchError::Other(e.to_string()) }
    }
}

/// A loaded model ready to detect. See the module docs.
pub struct Detector {
    // Only `set_config` needs `&mut`; no borrow is held across an await, so concurrent `detect` calls are fine.
    predictor: RefCell<Predictor>,
    /// Immutable baseline config from `load` so concurrent `detect` / `detect_tta` calls never observe transient
    /// `imgsz` changes from another in-flight call.
    base_cfg: PredictConfig,
    anchors: RefCell<Option<CachedAnchors>>,
    /// Reusable synchronous scratch buffers (never borrowed across `.await`).
    pub(crate) scratch: RefCell<RgbaScratch>,
    /// Reusable RGBA input buffer for JS→WASM pixel copies across repeated frames.
    rgba_pool: RefCell<Vec<u8>>,
    backend: Backend,
    gpu: Option<GpuContext>,
    model_id: String,
    fallback: Option<String>,
}

pub(crate) fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub(crate) fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The fused eval-mode detector (`folo-*` or `folo2-*`) of `file` on the CPU (fusing never needs a device
/// readback this way).
fn cpu_model(file: &ModelFile) -> Result<VisionDetector, String> {
    let (mut model, _) = VisionDetector::from_file(file, Purpose::Inference).map_err(err)?;
    model.fuse().map_err(err)?;
    Ok(model)
}

/// Validate that `imgsz` is a valid stride-32 inference size (`32..=8192`).
fn validate_imgsz(imgsz: u32) -> Result<(), String> {
    if imgsz == 0 || !imgsz.is_multiple_of(32) || imgsz > 8192 {
        return Err(format!("imgsz must be a multiple of 32 in 32..=8192, got {imgsz}"));
    }
    Ok(())
}

/// Validate per-call detection thresholds (`conf`, `iou`, `max_det`).
pub(crate) fn validate_detect_params(conf: f32, iou: f32, max_det: usize) -> Result<(), String> {
    if !conf.is_finite() || !(0.0..=1.0).contains(&conf) {
        return Err(format!("conf must be finite and in 0..=1, got {conf}"));
    }
    if !iou.is_finite() || !(0.0..=1.0).contains(&iou) {
        return Err(format!("iou must be finite and in 0..=1, got {iou}"));
    }
    if max_det == 0 {
        return Err("max_det must be at least 1".into());
    }
    Ok(())
}

/// RAII guard returning a borrowed RGB staging buffer to `Detector`'s scratch pool on drop (even across `.await`
/// or early `?` returns), keeping the larger allocation if concurrent calls finish out of order.
pub(crate) struct PooledRgb<'a> {
    detector: &'a Detector,
    img: Option<Image>,
}

impl<'a> PooledRgb<'a> {
    pub(crate) fn new(detector: &'a Detector, rgba: &[u8], width: u32, height: u32) -> Result<Self, String> {
        let n = validate_rgba(rgba, width, height)?;
        let mut data = std::mem::take(&mut detector.scratch.borrow_mut().rgb);
        data.resize(n * 3, 0);
        rgba_to_rgb(rgba, &mut data);
        Ok(PooledRgb { detector, img: Some(Image { width, height, data }) })
    }

    pub(crate) fn image(&self) -> &Image {
        self.img.as_ref().expect("PooledRgb image present until drop")
    }

    /// Take ownership of the pooled `Image` (caller is responsible for returning `img.data` via
    /// [`Detector::return_rgb_buffer`]).
    pub(crate) fn into_image(mut self) -> Image {
        self.img.take().expect("PooledRgb image present until taken")
    }
}

impl Drop for PooledRgb<'_> {
    fn drop(&mut self) {
        if let Some(img) = self.img.take() {
            self.detector.return_rgb_buffer(img.data);
        }
    }
}

impl Detector {
    /// Parse `bytes` (a `.folo` weights file, export or checkpoint; EMA weights when present) and place the
    /// model on the chosen backend.
    pub async fn load(bytes: Vec<u8>, opts: LoadOptions) -> Result<Self, String> {
        let file = ModelFile::from_bytes(bytes).map_err(err)?;
        let imgsz = opts.imgsz.unwrap_or(file.metadata().imgsz);
        validate_imgsz(imgsz)?;
        let cfg = PredictConfig { imgsz, ..PredictConfig::default() };
        let model_id = opts.model_id.unwrap_or_default();
        let mut fallback = None;
        if opts.backend != BackendChoice::Cpu {
            match Self::load_gpu(&file, &cfg, &model_id, opts.precision).await {
                Ok(d) => return Ok(d),
                Err(e) if opts.backend == BackendChoice::Auto => fallback = Some(e),
                Err(e) => return Err(format!("WebGPU backend unavailable: {e}")),
            }
        }
        let predictor =
            Predictor::from_detector_with_precision(cpu_model(&file)?, cfg.clone(), opts.precision)
                .map_err(err)?;
        Ok(Detector {
            predictor: RefCell::new(predictor),
            base_cfg: cfg,
            anchors: RefCell::new(None),
            scratch: RefCell::new(RgbaScratch::default()),
            rgba_pool: RefCell::new(Vec::new()),
            backend: Backend::Cpu,
            gpu: None,
            model_id,
            fallback,
        })
    }

    async fn load_gpu(
        file: &ModelFile,
        cfg: &PredictConfig,
        model_id: &str,
        precision: Precision,
    ) -> Result<Self, String> {
        let device = folo_gpu::init_async().await.map_err(err)?;
        let ctx = GpuContext::shared().map_err(err)?;
        let mut model = cpu_model(file)?;
        model.to_device(device).map_err(err)?;
        let predictor =
            Predictor::from_detector_with_precision(model, cfg.clone(), precision).map_err(err)?;
        let d = Detector {
            predictor: RefCell::new(predictor),
            base_cfg: cfg.clone(),
            anchors: RefCell::new(None),
            scratch: RefCell::new(RgbaScratch::default()),
            rgba_pool: RefCell::new(Vec::new()),
            backend: Backend::Webgpu,
            gpu: Some(ctx),
            model_id: model_id.to_string(),
            fallback: None,
        };
        // One pass on a small gray image, letterboxed to the configured size like any input, compiles the
        // kernels a real call needs (so the first `detect` is fast) and surfaces shader or device errors now,
        // so `auto` can still fall back to the CPU.
        let gray = vec![114u8; 64 * 64 * 4];
        d.detect(&gray, 64, 64, DetectOptions::default())
            .await
            .map_err(|e| format!("test pass failed: {e}"))?;
        Ok(d)
    }

    /// Take a reusable RGBA byte buffer of length `len` (zero heap allocation once warmed up).
    pub(crate) fn take_rgba_buffer(&self, len: usize) -> Vec<u8> {
        let mut buf = std::mem::take(&mut *self.rgba_pool.borrow_mut());
        buf.resize(len, 0);
        buf
    }

    /// Return a previously taken RGBA byte buffer to the pool, retaining the larger backing capacity.
    pub(crate) fn return_rgba_buffer(&self, buf: Vec<u8>) {
        let mut slot = self.rgba_pool.borrow_mut();
        if buf.capacity() > slot.capacity() {
            *slot = buf;
        }
    }

    /// Return a previously taken RGB staging buffer to the scratch pool, retaining the larger backing capacity.
    pub(crate) fn return_rgb_buffer(&self, buf: Vec<u8>) {
        let mut scratch = self.scratch.borrow_mut();
        if buf.capacity() > scratch.rgb.capacity() {
            scratch.rgb = buf;
        }
    }

    /// The backend in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Backend, model and class details.
    pub fn info(&self) -> DetectorInfo {
        let p = self.predictor.borrow();
        let det = p.detector();
        DetectorInfo {
            backend: self.backend,
            device: p.device().to_string(),
            adapter: self.gpu.as_ref().map(|g| g.summary().name.clone()).filter(|n| !n.is_empty()),
            fallback: self.fallback.clone(),
            precision: p.precision(),
            precision_fallback: p.fallback().map(str::to_string),
            simd128: folo_core::cpu::gemm::uses_simd128(),
            // The wasm build has no threads (rayon runs on the calling thread there).
            threads: if cfg!(target_arch = "wasm32") {
                1
            } else {
                std::thread::available_parallelism().map_or(1, |n| n.get())
            },
            family: if det.is_v2() { "folo2" } else { "folo" }.to_string(),
            arch: p.arch(),
            head_mode: det.head_mode().as_str().to_string(),
            has_o2o: det.has_o2o(),
            has_iou_head: p.model_v2().is_some_and(|m| m.config().iou_head),
            imgsz: self.base_cfg.imgsz,
            classes: p.classes().to_vec(),
            model_id: self.model_id.clone(),
        }
    }

    /// Detect objects in an RGBA image (`width * height * 4` bytes, rows top to bottom, alpha ignored).
    /// Boxes are in image pixels, highest score first.
    pub async fn detect(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        opts: DetectOptions,
    ) -> Result<DetectResult, String> {
        let t0 = Instant::now();
        validate_rgba(rgba, width, height)?;
        let cfg = self.predict_config(opts.conf, opts.iou, opts.max_det, self.base_cfg.imgsz)?;
        let nms = nms_config(&cfg);
        self.detect_single_with_cfg(rgba, width, height, &cfg, &nms, t0).await
    }

    /// Build and validate a per-call [`PredictConfig`] from [`Detector::base_cfg`].
    pub(crate) fn predict_config(
        &self,
        conf: f32,
        iou: f32,
        max_det: usize,
        imgsz: u32,
    ) -> Result<PredictConfig, String> {
        validate_detect_params(conf, iou, max_det)?;
        validate_imgsz(imgsz)?;
        Ok(PredictConfig { conf, iou, max_det, imgsz, ..self.base_cfg.clone() })
    }

    /// Single-pass detection at an explicit validated `cfg` (used by `detect`, single-tile `detect_tiled`, and
    /// single-view `detect_tta`).
    pub(crate) async fn detect_single_with_cfg(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        cfg: &PredictConfig,
        nms: &NmsConfig,
        t0: Instant,
    ) -> Result<DetectResult, String> {
        let Some(ctx) = &self.gpu else {
            let (detections, timing) =
                self.detect_cpu_raw(rgba, width, height, cfg.imgsz, cfg.rect, nms, t0)?;
            return Ok(self.build_result(width, height, detections, timing));
        };

        let pooled = PooledRgb::new(self, rgba, width, height)?;
        let rgb_ms = ms(t0);
        let (mut dets, timing) = self
            .detect_gpu_batch_once(ctx, std::slice::from_ref(pooled.image()), cfg, nms)
            .await
            .map_err(BatchError::into_string)?;
        let detections = dets.pop().expect("one image in, one out");
        let timing = Timing::new(rgb_ms + timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms);
        Ok(self.build_result(width, height, detections, timing))
    }

    /// Wrap detections and timing into a [`DetectResult`] matching docs/API.md's `/api/predict`.
    pub(crate) fn build_result(
        &self,
        width: u32,
        height: u32,
        detections: Vec<Detection>,
        timing: Timing,
    ) -> DetectResult {
        let p = self.predictor.borrow();
        let pred = folo_vision::Prediction { width, height, detections, timing };
        DetectResult {
            response: folo_api::PredictResponse {
                model_id: self.model_id.clone(),
                image: folo_api::ImageSize { width, height },
                boxes: pred.scored_boxes(p.classes()),
                timing: pred.timing.to_api(),
                device: p.device().to_string(),
                cold: false,
            },
            backend: self.backend,
        }
    }

    /// Direct CPU path: letterboxes RGBA straight into planar `f32` (reusing [`Detector::scratch`], zero
    /// intermediate RGB allocation) and runs candidate-first / sparse CPU decode, matching
    /// `folo_vision::Predictor::predict` bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn detect_cpu_raw(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        imgsz: u32,
        rect: bool,
        nms: &NmsConfig,
        t0: Instant,
    ) -> Result<(Vec<Detection>, Timing), String> {
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        let lb = Letterbox::compute(width, height, imgsz, imgsz, &opts);
        let [ow, oh] = lb.out.map(|v| v as usize);
        let mut buf = folo_core::cpu::take_buffer(3 * ow * oh);
        {
            let mut scratch = self.scratch.borrow_mut();
            letterbox_rgba_planar_f32_with_scratch(
                rgba,
                width,
                height,
                &lb,
                PAD_VALUE,
                INV_255,
                &mut buf,
                &mut scratch,
            );
        }
        let input = Tensor::from_vec(buf, [1, 3, oh, ow]).map_err(err)?;
        let pre = ms(t0);

        let p = self.predictor.borrow();
        let det = p.detector();
        let (raw_dets, inf, post) = if let Some(model) = det.as_v1()
            && model.candidates_first_applies(Device::Cpu, nms)
        {
            let t1 = Instant::now();
            let cands = model.predict_candidates(&input, nms).map_err(err)?;
            let inf = ms(t1);
            let t2 = Instant::now();
            let dets = model.finish_candidates(&cands[0], nms);
            (dets, inf, ms(t2))
        } else {
            let t1 = Instant::now();
            let out = {
                let _g = no_grad();
                det.forward(&input).map_err(err)?
            };
            let inf = ms(t1);
            let t2 = Instant::now();
            let dets = out
                .detections(nms, DetectorOutput::default_iou_alpha())
                .map_err(err)?
                .pop()
                .expect("one image in, one out");
            (dets, inf, ms(t2))
        };

        let t_unmap = Instant::now();
        let detections: Vec<Detection> = raw_dets
            .into_iter()
            .map(|d| {
                let [x1, y1, x2, y2] = lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]);
                Detection { bbox: Xyxy::new(x1, y1, x2, y2), ..d }
            })
            .collect();
        let post = post + ms(t_unmap);
        Ok((detections, Timing::new(pre, inf, post)))
    }

    /// Run a batch of RGB images on WebGPU with automatic half-split fallback on GPU OOM / buffer-limit errors,
    /// preserving input order.
    pub(crate) async fn detect_gpu_batch(
        &self,
        ctx: &GpuContext,
        imgs: &[Image],
        cfg: &PredictConfig,
        nms: &NmsConfig,
    ) -> Result<(Vec<Vec<Detection>>, Timing), String> {
        run_chunks_with_oom_fallback(imgs, |chunk| self.detect_gpu_batch_once(ctx, chunk, cfg, nms)).await
    }

    /// Execute one WebGPU batch (`imgs.len() >= 1`) with device-side candidate extraction and compact async
    /// readback (falling back to dense readback only when candidate extraction is unavailable).
    async fn detect_gpu_batch_once(
        &self,
        ctx: &GpuContext,
        imgs: &[Image],
        cfg: &PredictConfig,
        nms: &NmsConfig,
    ) -> Result<(Vec<Vec<Detection>>, Timing), BatchError> {
        let t0 = Instant::now();
        let (batch, pre, t1, det_out, canvas) = {
            let mut p = self.predictor.borrow_mut();
            p.set_config(cfg.clone()).map_err(BatchError::from_vision)?;
            let batch = p.prepare(imgs, cfg.rect).map_err(BatchError::from_vision)?;
            let dims = batch.input.dims();
            let canvas = [dims[2], dims[3]];
            let pre = ms(t0);
            let t1 = Instant::now();
            let det_out = p.forward_detector(&batch).map_err(BatchError::from_vision)?;
            (batch, pre, t1, det_out, canvas)
        };

        // Prefer device-side candidate extraction when no `iou_raw` score-fusion branch is active (~28 KiB/img
        // instead of ~2.8 MiB/img for the dense head). Models with FOLO v2 `iou_head` decode on device and
        // download both `dec` and `iou_raw` asynchronously before applying `fuse_cls_iou_scores`.
        let gpu_cands = if det_out.iou_raw.is_none() {
            self.try_gpu_candidates(ctx, &det_out.head, canvas, nms)?
        } else {
            None
        };
        let (dets, inf, t2) = match gpu_cands {
            Some((cands_tensor, mode, b, a)) => {
                drop(det_out);
                let records = download_candidate_records_batch(ctx, &cands_tensor, b, a).await?;
                let inf = ms(t1);
                let t2 = Instant::now();
                let raw_dets: Vec<Vec<Detection>> = records
                    .into_iter()
                    .map(|r| {
                        let cands = candidates_of_records(&r);
                        match mode {
                            HeadMode::O2m => nms_candidates(&cands, nms),
                            HeadMode::O2o => topk_candidates(&cands, nms),
                        }
                    })
                    .collect();
                let p = self.predictor.borrow();
                (p.finish_detections(&batch, raw_dets), inf, t2)
            }
            None => {
                let nc = det_out.head.num_classes;
                let b = det_out.head.batch();
                let a = det_out.head.num_anchors();
                let dec = det_out.head.decoded_on_device().map_err(|e| BatchError::from_vision(e.into()))?;
                let iou_f32 = match det_out.iou_raw {
                    Some(iou) => {
                        if iou.dims() != [b, 1, a] {
                            return Err(BatchError::Other(format!(
                                "iou_raw {:?} does not match [{b}, 1, {a}]",
                                iou.dims()
                            )));
                        }
                        let _g = no_grad();
                        let t = if iou.dtype() == DType::F32 {
                            iou
                        } else {
                            iou.to_dtype(DType::F32).map_err(BatchError::from_core)?
                        };
                        Some(t)
                    }
                    None => None,
                };
                drop(det_out.head);
                let host = folo_gpu::download_async(ctx, &dec).await.map_err(BatchError::from_core)?;
                let iou_host = match iou_f32 {
                    Some(iou) => {
                        Some(folo_gpu::download_async(ctx, &iou).await.map_err(BatchError::from_core)?)
                    }
                    None => None,
                };
                let inf = ms(t1);
                let t2 = Instant::now();
                let all = host.to_vec::<f32>().map_err(BatchError::from_core)?;
                let mut decoded = HeadOutput::split_decoded(&all, a, nc);
                if let Some(iou_host) = iou_host {
                    let iou_vals = iou_host.to_vec::<f32>().map_err(BatchError::from_core)?;
                    if iou_vals.len() != b * a {
                        return Err(BatchError::Other(format!(
                            "iou_raw: expected {} values, got {}",
                            b * a,
                            iou_vals.len()
                        )));
                    }
                    let alpha = DetectorOutput::default_iou_alpha();
                    for ((_, scores), iou_img) in decoded.iter_mut().zip(iou_vals.chunks_exact(a)) {
                        fuse_cls_iou_scores(scores, iou_img, nc, alpha);
                    }
                }
                let mut p = self.predictor.borrow_mut();
                p.set_config(cfg.clone()).map_err(BatchError::from_vision)?;
                (p.finish(&batch, decoded, nc), inf, t2)
            }
        };

        let post = ms(t2);
        Ok((dets, Timing::new(pre, inf, post)))
    }

    /// Encodes device-side candidate extraction (`Tensor::detect_candidates`) if supported, reusing a cached
    /// `[3, A]` anchors tensor for repeated frames of the same canvas size.
    fn try_gpu_candidates(
        &self,
        _ctx: &GpuContext,
        out: &HeadOutput,
        canvas: [usize; 2],
        nms: &NmsConfig,
    ) -> Result<Option<(Tensor, HeadMode, usize, usize)>, BatchError> {
        let _g = no_grad();
        let (b, a, nc, rm) = (out.batch(), out.num_anchors(), out.num_classes, out.reg_max);
        if b == 0
            || out.box_raw.dims() != [b, 4 * rm, a]
            || out.cls_raw.dims() != [b, nc, a]
            || a == 0
            || nc == 0
        {
            return Ok(None);
        }
        let dev = out.box_raw.device();
        let anchors = self.gpu_anchors(out, canvas, dev)?;
        let params = DetectDecode { reg_max: rm, num_classes: nc, conf: nms.candidate_conf_for(out.mode) };
        let Some(cands) = Tensor::detect_candidates(&out.box_raw, &out.cls_raw, &anchors, &params)
            .map_err(BatchError::from_core)?
        else {
            return Ok(None);
        };
        Ok(Some((cands, out.mode, b, a)))
    }

    /// Returns a cached `[3, A]` device anchors tensor for `canvas`, uploading it once per canvas shape.
    fn gpu_anchors(&self, out: &HeadOutput, canvas: [usize; 2], dev: Device) -> Result<Tensor, BatchError> {
        let a = out.num_anchors();
        if let Some(cached) = self.anchors.borrow().as_ref()
            && cached.canvas == canvas
            && cached.num_anchors == a
            && cached.tensor.device() == dev
        {
            return Ok(cached.tensor.clone());
        }
        let mut v = Vec::with_capacity(3 * a);
        v.extend(out.anchors.points.iter().map(|p| p[0]));
        v.extend(out.anchors.points.iter().map(|p| p[1]));
        v.extend_from_slice(&out.anchors.strides);
        let tensor = Tensor::from_vec(v, [3, a])
            .map_err(BatchError::from_core)?
            .to_device(dev)
            .map_err(BatchError::from_core)?;
        *self.anchors.borrow_mut() = Some(CachedAnchors { canvas, num_anchors: a, tensor: tensor.clone() });
        Ok(tensor)
    }
}

/// Iterative chunk-halving driver for GPU batch execution: splits a failing chunk in half on [`BatchError::Oom`]
/// while preserving exact input order and failing immediately on non-OOM or single-element errors.
async fn run_chunks_with_oom_fallback<'a, T, R, F, Fut>(
    items: &'a [T],
    mut run_chunk: F,
) -> Result<(Vec<R>, Timing), String>
where
    F: FnMut(&'a [T]) -> Fut,
    Fut: Future<Output = Result<(Vec<R>, Timing), BatchError>>,
{
    let mut pending: VecDeque<&'a [T]> = VecDeque::from([items]);
    let mut all = Vec::with_capacity(items.len());
    let (mut pre, mut inf, mut post) = (0.0f64, 0.0f64, 0.0f64);
    while let Some(chunk) = pending.pop_front() {
        match run_chunk(chunk).await {
            Ok((out, t)) => {
                all.extend(out);
                pre += t.preprocess_ms;
                inf += t.inference_ms;
                post += t.postprocess_ms;
            }
            Err(BatchError::Oom(_)) if chunk.len() > 1 => {
                let mid = chunk.len() / 2;
                pending.push_front(&chunk[mid..]);
                pending.push_front(&chunk[..mid]);
            }
            Err(e) => return Err(e.into_string()),
        }
    }
    Ok((all, Timing::new(pre, inf, post)))
}

/// Converts a [`PredictConfig`] into its matching [`NmsConfig`].
pub(crate) fn nms_config(cfg: &PredictConfig) -> NmsConfig {
    NmsConfig {
        conf: cfg.conf,
        iou: cfg.iou,
        max_det: cfg.max_det,
        fusion: cfg.fusion,
        fusion_iou: cfg.fusion_iou,
        score: cfg.score,
        ..NmsConfig::default()
    }
}

/// Whether a `folo_core::Error` from `wgpu` indicates an out-of-memory or device buffer-limit failure
/// (`GpuError::Limit` / `HeadroomBlocker` / WebGPU OOM) suitable for chunk halving.
fn is_core_gpu_oom(e: &folo_core::Error) -> bool {
    let folo_core::Error::Backend { backend: "wgpu", msg } = e else {
        return false;
    };
    is_wgpu_oom_or_limit_msg(msg)
}

fn is_wgpu_oom_or_limit_msg(msg: &str) -> bool {
    if msg.contains("exceeds max_storage_buffer_binding_size")
        || msg.contains("exceeds max_buffer_size")
        || msg.contains("exceeds estimated remaining VRAM")
        || msg.contains("bytes exceeds the device limit")
    {
        return true;
    }
    let lower = msg.to_ascii_lowercase();
    lower.contains("out of memory") || lower.contains("outofmemory")
}

/// Downloads the compacted `[B, 1 + 7 * A]` candidate buffer asynchronously: one readback of the counts plus up
/// to `CANDIDATE_READBACK` records per image, and a second readback only for any image that overflows that cap.
async fn download_candidate_records_batch(
    ctx: &GpuContext,
    cands: &Tensor,
    b: usize,
    a: usize,
) -> Result<Vec<Vec<u32>>, BatchError> {
    const W: usize = DETECT_RECORD_WORDS;
    let cap = CANDIDATE_READBACK.min(a);
    let stride = 1 + W * cap;
    let head_view = cands.narrow(1, 0, stride).map_err(BatchError::from_core)?;
    let head = folo_gpu::download_async(ctx, &head_view)
        .await
        .map_err(BatchError::from_core)?
        .to_vec::<u32>()
        .map_err(BatchError::from_core)?;
    if head.len() != b * stride {
        return Err(BatchError::Other(format!(
            "detect_candidates: expected {} header words, got {}",
            b * stride,
            head.len()
        )));
    }
    let mut records = Vec::with_capacity(b);
    for (i, row) in head.chunks_exact(stride).enumerate() {
        let n = row[0] as usize;
        if n > a {
            return Err(BatchError::Other(format!("detect_candidates: {n} candidates for {a} anchors")));
        }
        if n <= cap {
            records.push(row[1..1 + W * n].to_vec());
        } else {
            let full_view = cands
                .narrow(0, i, 1)
                .map_err(BatchError::from_core)?
                .narrow(1, 1, W * n)
                .map_err(BatchError::from_core)?;
            let full = folo_gpu::download_async(ctx, &full_view)
                .await
                .map_err(BatchError::from_core)?
                .to_vec::<u32>()
                .map_err(BatchError::from_core)?;
            records.push(full);
        }
    }
    Ok(records)
}

/// Parses `DETECT_RECORD_WORDS`-word device candidate records and sorts by anchor (NMS tie-break order).
fn candidates_of_records(r: &[u32]) -> Vec<Candidate> {
    let mut c: Vec<Candidate> = r
        .as_chunks::<DETECT_RECORD_WORDS>()
        .0
        .iter()
        .map(|w| Candidate {
            anchor: w[0],
            class: w[1],
            score: f32::from_bits(w[2]),
            bbox: Xyxy::new(
                f32::from_bits(w[3]),
                f32::from_bits(w[4]),
                f32::from_bits(w[5]),
                f32::from_bits(w[6]),
            ),
        })
        .collect();
    c.sort_unstable_by_key(|c| c.anchor);
    c
}

/// Validates RGBA image dimensions and byte length, returning the pixel count `width * height`.
pub(crate) fn validate_rgba(rgba: &[u8], width: u32, height: u32) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err(format!("image is empty ({width}x{height})"));
    }
    // Checked: `usize` is 32 bits on wasm32, where 65536 x 65536 x 4 would wrap to 0; also cap at `isize::MAX`.
    let n = (width as usize)
        .checked_mul(height as usize)
        .filter(|n| n.checked_mul(4).is_some_and(|b| b <= isize::MAX as usize))
        .ok_or_else(|| format!("image is too large ({width}x{height})"))?;
    if rgba.len() != n * 4 {
        return Err(format!(
            "expected {} RGBA bytes for a {width}x{height} image, got {}",
            n * 4,
            rgba.len()
        ));
    }
    Ok(n)
}

/// RGBA → folo-data's RGB [`Image`], checking the size.
pub fn rgba_to_image(rgba: &[u8], width: u32, height: u32) -> Result<Image, String> {
    let n = validate_rgba(rgba, width, height)?;
    let mut data = vec![0u8; n * 3];
    rgba_to_rgb(rgba, &mut data);
    Ok(Image { width, height, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validators_and_oom_classifier_cover_edge_cases() {
        assert!(validate_imgsz(32).is_ok());
        assert!(validate_imgsz(640).is_ok());
        assert!(validate_imgsz(8192).is_ok());
        assert!(validate_imgsz(0).is_err());
        assert!(validate_imgsz(48).is_err());
        assert!(validate_imgsz(8224).is_err());

        assert!(validate_detect_params(0.0, 1.0, 1).is_ok());
        assert!(validate_detect_params(f32::NAN, 0.5, 10).is_err());
        assert!(validate_detect_params(0.5, f32::INFINITY, 10).is_err());
        assert!(validate_detect_params(0.5, 0.5, 0).is_err());

        for msg in [
            "storage buffer of 200 bytes (200 B) exceeds max_storage_buffer_binding_size (128 bytes / 128 B)",
            "buffer of 200 bytes (200 B) exceeds max_buffer_size (128 bytes / 128 B)",
            "storage buffer of 200 bytes (200 B) exceeds estimated remaining VRAM (128 bytes / 128 B)",
            "buffer of 200 bytes exceeds the device limit (128 bytes)",
            "wgpu Out of Memory",
        ] {
            let core_err = folo_core::Error::Backend { backend: "wgpu", msg: msg.into() };
            assert!(is_core_gpu_oom(&core_err), "must classify as GPU OOM/limit: {msg}");
            assert!(matches!(BatchError::from_core(core_err), BatchError::Oom(_)));
        }
        let non_oom = folo_core::Error::Backend { backend: "wgpu", msg: "invalid tensor shape".into() };
        assert!(!is_core_gpu_oom(&non_oom));
        assert!(matches!(BatchError::from_core(non_oom), BatchError::Other(_)));
    }

    #[test]
    fn oom_fallback_splits_storage_limit_batches_in_order() {
        let items = [10u32, 20, 30, 40, 50, 60];
        let mut seen_lens = Vec::new();
        let (out, timing) = pollster::block_on(run_chunks_with_oom_fallback(&items, |chunk| {
            seen_lens.push(chunk.len());
            let chunk_vec = chunk.to_vec();
            async move {
                if chunk_vec.len() > 2 {
                    let err = folo_core::Error::Backend {
                        backend: "wgpu",
                        msg: "storage buffer of 268435456 bytes exceeds max_storage_buffer_binding_size (134217728 bytes)".into(),
                    };
                    return Err(BatchError::from_core(err));
                }
                Ok((chunk_vec, Timing::new(1.0, 2.0, 0.5)))
            }
        }))
        .unwrap();
        assert_eq!(out, items);
        assert_eq!(seen_lens, vec![6, 3, 1, 2, 3, 1, 2]);
        assert!(timing.total_ms > 0.0);

        // Non-OOM errors must NOT split or retry.
        let mut calls = 0usize;
        let err = pollster::block_on(run_chunks_with_oom_fallback(&items, |_| {
            calls += 1;
            async { Err::<(Vec<u32>, Timing), _>(BatchError::Other("shader validation failed".into())) }
        }))
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(err.contains("shader validation failed"));

        // Single-element OOM cannot split further and fails cleanly.
        let err = pollster::block_on(run_chunks_with_oom_fallback(&items[..1], |_| async {
            Err::<(Vec<u32>, Timing), _>(BatchError::Oom("exceeds max_buffer_size".into()))
        }))
        .unwrap_err();
        assert!(err.contains("max_buffer_size"));
    }
}
