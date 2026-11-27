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

mod tiled;
mod tta;

use std::cell::RefCell;
use std::collections::VecDeque;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

use folo_core::backend::DETECT_RECORD_WORDS;
use folo_core::{DetectDecode, Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions, PAD_VALUE};
use folo_gpu::GpuContext;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::boxes::Xyxy;
use folo_vision::fusion::BoxFusion;
use folo_vision::model::HeadMode;
use folo_vision::nms::{Candidate, Detection, NmsConfig, nms_candidates, topk_candidates};
use folo_vision::tta::DEFAULT_TTA_SCALES;
use folo_vision::{
    FoloModel, HeadOutput, Precision, PredictConfig, Predictor, TileConfig, TileMergeMetric, Timing,
    TtaConfig,
};
use serde::{Deserialize, Serialize};

use crate::simd::{RgbaScratch, letterbox_rgba_planar_f32_with_scratch, rgba_to_rgb};

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

/// Which backend [`Detector::load`] should use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    /// WebGPU when it initialises and runs, otherwise the CPU.
    #[default]
    Auto,
    /// WebGPU or an error.
    Webgpu,
    /// The wasm CPU backend.
    Cpu,
}

/// The backend a [`Detector`] runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// folo-gpu on WebGPU (natively: Vulkan / DX12 / Metal).
    Webgpu,
    /// folo-core's CPU kernels (vectorised to wasm SIMD128 in browser builds).
    Cpu,
}

impl Backend {
    /// `"webgpu"` or `"cpu"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Webgpu => "webgpu",
            Backend::Cpu => "cpu",
        }
    }
}

/// [`Detector::load`] options (the JS `load` options object; unknown fields are ignored).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct LoadOptions {
    /// `auto` (default), `webgpu` or `cpu`.
    pub backend: BackendChoice,
    /// Echoed as `model_id` in every result (docs/API.md); empty by default.
    pub model_id: Option<String>,
    /// Letterbox size; defaults to the file's `folo.imgsz`.
    pub imgsz: Option<u32>,
    /// `f32` (default) or `f16`: half precision on WebGPU with `shader-f16`; f32 elsewhere (the CPU, an adapter
    /// without it), with the reason in [`DetectorInfo::precision_fallback`].
    pub precision: Precision,
}

/// Per-call options (the JS `detect` options object), with docs/API.md's `/api/predict` defaults.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DetectOptions {
    /// Score threshold in `0..=1`.
    pub conf: f32,
    /// NMS IoU threshold in `0..=1`.
    pub iou: f32,
    /// Detections kept, at least 1.
    pub max_det: usize,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions { conf: 0.25, iou: 0.7, max_det: 300 }
    }
}

/// Per-call options for SAHI tiled inference ([`Detector::detect_tiled`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TileDetectOptions {
    /// Score threshold in `0..=1` (default `0.25`).
    pub conf: f32,
    /// Per-tile NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per tile and after cross-tile merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Convenience square tile size in pixels (`tileWidth` / `tileHeight` override it when set; default `640`).
    pub tile_size: Option<u32>,
    /// Target tile width in pixels (defaults to `tileSize`, or `640`).
    pub tile_width: Option<u32>,
    /// Target tile height in pixels (defaults to `tileSize`, or `640`).
    pub tile_height: Option<u32>,
    /// Convenience symmetric overlap ratio in `[0, 1)` (`overlapX` / `overlapY` override it; default `0.2`).
    pub overlap: Option<f32>,
    /// Horizontal overlap ratio between adjacent tiles in `[0, 1)`.
    pub overlap_x: Option<f32>,
    /// Vertical overlap ratio between adjacent tiles in `[0, 1)`.
    pub overlap_y: Option<f32>,
    /// Also run a full-image pass alongside the tiles to preserve large boundary-spanning objects (default `true`).
    #[serde(alias = "fullImagePass")]
    pub full_image: bool,
    /// Maximum cropped tiles predicted per WebGPU batch, at least `1` (default `8`).
    pub tile_batch: usize,
    /// Cross-tile merge overlap metric: `"ios"` (default) or `"iou"`.
    pub merge_metric: TileMergeMetric,
    /// Cross-tile merge overlap threshold in `0..=1` (default `0.5`; also accepts JS alias `mergeIou`).
    #[serde(alias = "mergeIou")]
    pub merge_threshold: f32,
    /// Cross-tile box fusion rule: `"vote"` (default), `"none"`, `"dfl"`, or `"sebf"`.
    pub merge_fusion: BoxFusion,
    /// Merge overlapping detections across different classes (default `false`).
    pub class_agnostic: bool,
}

impl Default for TileDetectOptions {
    fn default() -> Self {
        let def = TileConfig::default();
        TileDetectOptions {
            conf: 0.25,
            iou: 0.7,
            max_det: def.max_det,
            tile_size: None,
            tile_width: None,
            tile_height: None,
            overlap: None,
            overlap_x: None,
            overlap_y: None,
            full_image: def.full_image,
            tile_batch: def.tile_batch,
            merge_metric: def.merge_metric,
            merge_threshold: def.merge_threshold,
            merge_fusion: def.merge_fusion,
            class_agnostic: def.class_agnostic,
        }
    }
}

impl TileDetectOptions {
    /// Validate and resolve into a [`TileConfig`].
    pub fn to_tile_config(&self) -> Result<TileConfig, String> {
        if let Some(0) = self.tile_size {
            return Err("tile_size must be positive, got 0".into());
        }
        if let Some(o) = self.overlap
            && (!o.is_finite() || !(0.0..1.0).contains(&o))
        {
            return Err(format!("tile overlap must be finite and in [0, 1), got {o}"));
        }
        let def = TileConfig::default();
        let tile_cfg = TileConfig {
            tile_width: self.tile_width.unwrap_or_else(|| self.tile_size.unwrap_or(def.tile_width)),
            tile_height: self.tile_height.unwrap_or_else(|| self.tile_size.unwrap_or(def.tile_height)),
            overlap_x: self.overlap_x.unwrap_or_else(|| self.overlap.unwrap_or(def.overlap_x)),
            overlap_y: self.overlap_y.unwrap_or_else(|| self.overlap.unwrap_or(def.overlap_y)),
            full_image: self.full_image,
            tile_batch: self.tile_batch,
            merge_metric: self.merge_metric,
            merge_threshold: self.merge_threshold,
            merge_fusion: self.merge_fusion,
            class_agnostic: self.class_agnostic,
            max_det: self.max_det,
        };
        tile_cfg.validate().map_err(err)?;
        Ok(tile_cfg)
    }
}

/// Per-call options for Test-Time Augmentation ([`Detector::detect_tta`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TtaDetectOptions {
    /// Score threshold in `0..=1` (default `0.25`).
    pub conf: f32,
    /// Per-view NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per view and after cross-view merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Also evaluate a horizontally flipped mirror at each scale (default `true`).
    pub hflip: bool,
    /// Multi-scale multipliers applied to the detector's `imgsz` (default `[0.83, 1.0, 1.17]`).
    pub scales: Vec<f32>,
    /// Cross-view merge IoU threshold in `0..=1` (defaults to `iou` when omitted).
    pub merge_iou: Option<f32>,
    /// Cross-view box fusion rule: `"vote"` (default), `"none"`, `"dfl"`, or `"sebf"`.
    pub merge_fusion: BoxFusion,
    /// Merge overlapping detections across different classes (default `false`).
    pub class_agnostic: bool,
}

impl Default for TtaDetectOptions {
    fn default() -> Self {
        let def = TtaConfig::default();
        TtaDetectOptions {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
            hflip: def.hflip,
            scales: DEFAULT_TTA_SCALES.to_vec(),
            merge_iou: def.merge_iou,
            merge_fusion: def.merge_fusion,
            class_agnostic: def.class_agnostic,
        }
    }
}

impl TtaDetectOptions {
    /// Validate and resolve into a [`TtaConfig`].
    pub fn to_tta_config(&self) -> Result<TtaConfig, String> {
        let tta_cfg = TtaConfig {
            hflip: self.hflip,
            scales: self.scales.clone(),
            merge_iou: self.merge_iou,
            merge_fusion: self.merge_fusion,
            class_agnostic: self.class_agnostic,
            max_det: Some(self.max_det),
        };
        tta_cfg.validate().map_err(err)?;
        Ok(tta_cfg)
    }
}

/// What [`Detector::info`] reports.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectorInfo {
    /// `webgpu` or `cpu`.
    pub backend: Backend,
    /// docs/API.md's device string: `gpu:0` or `cpu`.
    pub device: String,
    /// The GPU adapter's name (WebGPU only; `None` when the browser does not disclose it, as Chrome does).
    pub adapter: Option<String>,
    /// Why `auto` fell back to the CPU, if it did.
    pub fallback: Option<String>,
    /// The precision the model runs in: `f32` or `f16`.
    pub precision: Precision,
    /// Why a requested `f16` runs in `f32`, if it does.
    pub precision_fallback: Option<String>,
    /// Whether the CPU kernels were built with wasm SIMD128.
    pub simd128: bool,
    /// CPU threads (1: the wasm build is single-threaded).
    pub threads: usize,
    /// Model family and scale, e.g. `folo-n`.
    pub arch: String,
    /// Letterbox size.
    pub imgsz: u32,
    /// Class names, indexed by class id.
    pub classes: Vec<String>,
    /// The `model_id` echoed in results.
    pub model_id: String,
}

/// A detection result: docs/API.md's `/api/predict` response plus the backend that produced it.
#[derive(Clone, Debug, Serialize)]
pub struct DetectResult {
    /// The `/api/predict` fields.
    #[serde(flatten)]
    pub response: folo_api::PredictResponse,
    /// `webgpu` or `cpu`.
    pub backend: Backend,
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

/// The fused eval-mode model of `file` on the CPU (fusing never needs a device readback this way).
fn cpu_model(file: &ModelFile) -> Result<FoloModel, String> {
    let (mut model, _) = FoloModel::from_file(file, Purpose::Inference).map_err(err)?;
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
}

impl Drop for PooledRgb<'_> {
    fn drop(&mut self) {
        if let Some(img) = self.img.take() {
            let mut scratch = self.detector.scratch.borrow_mut();
            if img.data.capacity() > scratch.rgb.capacity() {
                scratch.rgb = img.data;
            }
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
            Predictor::with_precision(cpu_model(&file)?, cfg.clone(), opts.precision).map_err(err)?;
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
        let predictor = Predictor::with_precision(model, cfg.clone(), precision).map_err(err)?;
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

    /// The backend in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Backend, model and class details.
    pub fn info(&self) -> DetectorInfo {
        let p = self.predictor.borrow();
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
            arch: p.model().arch(),
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
        let (mut dets, timing) =
            self.detect_gpu_batch_once(ctx, std::slice::from_ref(pooled.image()), cfg, nms).await?;
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
        let model = p.model();
        let (raw_dets, inf, post) = if model.candidates_first_applies(Device::Cpu, nms) {
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
                model.forward(&input).map_err(err)?
            };
            let inf = ms(t1);
            let t2 = Instant::now();
            let dets = out.detections(nms).map_err(err)?.pop().expect("one image in, one out");
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
        let mut pending: VecDeque<&[Image]> = VecDeque::from([imgs]);
        let mut all_dets = Vec::with_capacity(imgs.len());
        let (mut pre, mut inf, mut post) = (0.0f64, 0.0f64, 0.0f64);
        while let Some(chunk) = pending.pop_front() {
            match self.detect_gpu_batch_once(ctx, chunk, cfg, nms).await {
                Ok((dets, t)) => {
                    all_dets.extend(dets);
                    pre += t.preprocess_ms;
                    inf += t.inference_ms;
                    post += t.postprocess_ms;
                }
                Err(e) if chunk.len() > 1 && is_gpu_oom_msg(&e) => {
                    let mid = chunk.len() / 2;
                    pending.push_front(&chunk[mid..]);
                    pending.push_front(&chunk[..mid]);
                }
                Err(e) => return Err(e),
            }
        }
        Ok((all_dets, Timing::new(pre, inf, post)))
    }

    /// Execute one WebGPU batch (`imgs.len() >= 1`) with device-side candidate extraction and compact async
    /// readback (falling back to dense readback only when candidate extraction is unavailable).
    async fn detect_gpu_batch_once(
        &self,
        ctx: &GpuContext,
        imgs: &[Image],
        cfg: &PredictConfig,
        nms: &NmsConfig,
    ) -> Result<(Vec<Vec<Detection>>, Timing), String> {
        let t0 = Instant::now();
        let (batch, pre, t1, out, canvas) = {
            let mut p = self.predictor.borrow_mut();
            p.set_config(cfg.clone()).map_err(err)?;
            let batch = p.prepare(imgs, cfg.rect).map_err(err)?;
            let dims = batch.input.dims();
            let canvas = [dims[2], dims[3]];
            let pre = ms(t0);
            let t1 = Instant::now();
            let out = p.forward(&batch).map_err(err)?;
            (batch, pre, t1, out, canvas)
        };

        // Prefer device-side candidate extraction (downloads ~28 KiB/img instead of ~2.8 MiB/img for dense head).
        let (dets, inf, t2) = match self.try_gpu_candidates(ctx, &out, canvas, nms)? {
            Some((cands_tensor, mode, b, a)) => {
                drop(out);
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
                let nc = out.num_classes;
                let dec = out.decoded_on_device().map_err(err)?;
                drop(out);
                let host = folo_gpu::download_async(ctx, &dec).await.map_err(err)?;
                let inf = ms(t1);
                let t2 = Instant::now();
                let all = host.to_vec::<f32>().map_err(err)?;
                let a = dec.dims()[1];
                let decoded = HeadOutput::split_decoded(&all, a, nc);
                let mut p = self.predictor.borrow_mut();
                p.set_config(cfg.clone()).map_err(err)?;
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
    ) -> Result<Option<(Tensor, HeadMode, usize, usize)>, String> {
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
        let Some(cands) =
            Tensor::detect_candidates(&out.box_raw, &out.cls_raw, &anchors, &params).map_err(err)?
        else {
            return Ok(None);
        };
        Ok(Some((cands, out.mode, b, a)))
    }

    /// Returns a cached `[3, A]` device anchors tensor for `canvas`, uploading it once per canvas shape.
    fn gpu_anchors(&self, out: &HeadOutput, canvas: [usize; 2], dev: Device) -> Result<Tensor, String> {
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
        let tensor = Tensor::from_vec(v, [3, a]).map_err(err)?.to_device(dev).map_err(err)?;
        *self.anchors.borrow_mut() = Some(CachedAnchors { canvas, num_anchors: a, tensor: tensor.clone() });
        Ok(tensor)
    }
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

/// Whether a GPU error string indicates an out-of-memory or device buffer-limit failure suitable for batch splitting.
fn is_gpu_oom_msg(msg: &str) -> bool {
    if msg.contains("bytes exceeds the device limit") {
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
) -> Result<Vec<Vec<u32>>, String> {
    const W: usize = DETECT_RECORD_WORDS;
    let cap = CANDIDATE_READBACK.min(a);
    let stride = 1 + W * cap;
    let head_view = cands.narrow(1, 0, stride).map_err(err)?;
    let head = folo_gpu::download_async(ctx, &head_view).await.map_err(err)?.to_vec::<u32>().map_err(err)?;
    if head.len() != b * stride {
        return Err(format!("detect_candidates: expected {} header words, got {}", b * stride, head.len()));
    }
    let mut records = Vec::with_capacity(b);
    for (i, row) in head.chunks_exact(stride).enumerate() {
        let n = row[0] as usize;
        if n > a {
            return Err(format!("detect_candidates: {n} candidates for {a} anchors"));
        }
        if n <= cap {
            records.push(row[1..1 + W * n].to_vec());
        } else {
            let full_view = cands.narrow(0, i, 1).map_err(err)?.narrow(1, 1, W * n).map_err(err)?;
            let full =
                folo_gpu::download_async(ctx, &full_view).await.map_err(err)?.to_vec::<u32>().map_err(err)?;
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
    // Checked: `usize` is 32 bits on wasm32, where 65536 x 65536 x 4 would wrap to 0.
    let n = (width as usize)
        .checked_mul(height as usize)
        .filter(|n| n.checked_mul(4).is_some())
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
    use super::{is_gpu_oom_msg, validate_detect_params, validate_imgsz};

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

        assert!(is_gpu_oom_msg("buffer of 200 bytes exceeds the device limit (128 bytes)"));
        assert!(is_gpu_oom_msg("wgpu Out of Memory"));
        assert!(!is_gpu_oom_msg("invalid tensor shape"));
    }
}
