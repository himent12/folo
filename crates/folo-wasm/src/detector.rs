//! The detector behind the JS bindings, free of `wasm-bindgen` types so the workspace's native tests run it.
//!
//! [`Detector::load`] parses a `.folo` file, builds the fused eval-mode model and places it on WebGPU or the CPU
//! (`auto` tries WebGPU and checks it with one small forward pass, falling back to the CPU). [`Detector::detect`]
//! is [`folo_vision::Predictor`]'s pipeline (letterbox, forward, decode, NMS, unmap):
//! - on WebGPU, the head is decoded into above-threshold candidates on the device (`Tensor::detect_candidates`)
//!   and only the compact candidate records come back via an awaited [`folo_gpu::download_async`] (~28 KiB per
//!   frame instead of ~2.8 MiB for the dense `[1, A, 4 + nc]` head), falling back to dense readback only when a
//!   backend lacks the candidate kernel;
//! - on the CPU, RGBA pixels are letterboxed directly into planar `f32` (using wasm SIMD128 when built with
//!   `+simd128`) without an intermediate RGB image copy, then run through candidate-first / sparse decode.

use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_core::backend::DETECT_RECORD_WORDS;
use folo_core::{DetectDecode, Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions, PAD_VALUE};
use folo_gpu::GpuContext;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::boxes::Xyxy;
use folo_vision::model::HeadMode;
use folo_vision::nms::{Candidate, Detection, NmsConfig, nms_candidates, topk_candidates};
use folo_vision::{FoloModel, HeadOutput, Precision, PredictConfig, Predictor, Timing};
use serde::{Deserialize, Serialize};

use crate::simd::{letterbox_rgba_planar_f32, rgba_to_rgb};

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
    anchors: RefCell<Option<CachedAnchors>>,
    backend: Backend,
    gpu: Option<GpuContext>,
    model_id: String,
    fallback: Option<String>,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The fused eval-mode model of `file` on the CPU (fusing never needs a device readback this way).
fn cpu_model(file: &ModelFile) -> Result<FoloModel, String> {
    let (mut model, _) = FoloModel::from_file(file, Purpose::Inference).map_err(err)?;
    model.fuse().map_err(err)?;
    Ok(model)
}

impl Detector {
    /// Parse `bytes` (a `.folo` weights file, export or checkpoint; EMA weights when present) and place the
    /// model on the chosen backend.
    pub async fn load(bytes: Vec<u8>, opts: LoadOptions) -> Result<Self, String> {
        let file = ModelFile::from_bytes(bytes).map_err(err)?;
        let imgsz = opts.imgsz.unwrap_or(file.metadata().imgsz);
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
        let predictor = Predictor::with_precision(cpu_model(&file)?, cfg, opts.precision).map_err(err)?;
        Ok(Detector {
            predictor: RefCell::new(predictor),
            anchors: RefCell::new(None),
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
            anchors: RefCell::new(None),
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
            imgsz: p.config().imgsz,
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
        let cfg = {
            let p = self.predictor.borrow();
            PredictConfig { conf: opts.conf, iou: opts.iou, max_det: opts.max_det, ..p.config().clone() }
        };
        // Validates the options now; they are applied again right before NMS, because another `detect` may
        // change the predictor's config while this one awaits its readback.
        self.predictor.borrow_mut().set_config(cfg.clone()).map_err(err)?;
        let nms = nms_config(&cfg);

        let Some(ctx) = &self.gpu else {
            return self.detect_cpu(rgba, width, height, &cfg, &nms, t0);
        };

        let img = rgba_to_image(rgba, width, height)?;
        let (batch, pre, t1, out, canvas) = {
            let p = self.predictor.borrow();
            let batch = p.prepare(std::slice::from_ref(&img), p.config().rect).map_err(err)?;
            let dims = batch.input.dims();
            let canvas = [dims[2], dims[3]];
            let pre = ms(t0);
            let t1 = Instant::now();
            let out = p.forward(&batch).map_err(err)?;
            (batch, pre, t1, out, canvas)
        };

        // Prefer device-side candidate extraction (downloads ~28 KiB instead of ~2.8 MiB for the dense head).
        let (dets, inf, t2) = match self.try_gpu_candidates(ctx, &out, canvas, &nms)? {
            Some((cands_tensor, mode, a)) => {
                drop(out);
                let records = download_candidate_records(ctx, &cands_tensor, a).await?;
                let inf = ms(t1);
                let t2 = Instant::now();
                let cands = candidates_of_records(&records);
                let raw_dets = match mode {
                    HeadMode::O2m => nms_candidates(&cands, &nms),
                    HeadMode::O2o => topk_candidates(&cands, &nms),
                };
                let mut p = self.predictor.borrow_mut();
                p.set_config(cfg).map_err(err)?;
                (p.finish_detections(&batch, vec![raw_dets]), inf, t2)
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
                p.set_config(cfg).map_err(err)?;
                (p.finish(&batch, decoded, nc), inf, t2)
            }
        };

        let post = ms(t2);
        let p = self.predictor.borrow();
        let pred = batch.predictions(dets, Timing::new(pre, inf, post)).pop().expect("one image in, one out");
        Ok(DetectResult {
            response: folo_api::PredictResponse {
                model_id: self.model_id.clone(),
                image: folo_api::ImageSize { width, height },
                boxes: pred.scored_boxes(p.classes()),
                timing: pred.timing.to_api(),
                device: p.device().to_string(),
                cold: false,
            },
            backend: self.backend,
        })
    }

    /// Direct CPU path: letterboxes RGBA straight into planar `f32` (zero intermediate RGB allocation) and uses
    /// candidate-first / sparse CPU decode, matching `folo_vision::Predictor::predict` bit for bit.
    fn detect_cpu(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        cfg: &PredictConfig,
        nms: &NmsConfig,
        t0: Instant,
    ) -> Result<DetectResult, String> {
        let opts = LetterboxOptions { stride: cfg.rect.then_some(32), ..LetterboxOptions::default() };
        let lb = Letterbox::compute(width, height, cfg.imgsz, cfg.imgsz, &opts);
        let [ow, oh] = lb.out.map(|v| v as usize);
        let mut buf = folo_core::cpu::take_buffer(3 * ow * oh);
        letterbox_rgba_planar_f32(rgba, width, height, &lb, PAD_VALUE, INV_255, &mut buf);
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
        let pred = folo_vision::Prediction { width, height, detections, timing: Timing::new(pre, inf, post) };
        Ok(DetectResult {
            response: folo_api::PredictResponse {
                model_id: self.model_id.clone(),
                image: folo_api::ImageSize { width, height },
                boxes: pred.scored_boxes(p.classes()),
                timing: pred.timing.to_api(),
                device: p.device().to_string(),
                cold: false,
            },
            backend: self.backend,
        })
    }

    /// Encodes device-side candidate extraction (`Tensor::detect_candidates`) if supported, reusing a cached
    /// `[3, A]` anchors tensor for repeated frames of the same canvas size.
    fn try_gpu_candidates(
        &self,
        _ctx: &GpuContext,
        out: &HeadOutput,
        canvas: [usize; 2],
        nms: &NmsConfig,
    ) -> Result<Option<(Tensor, HeadMode, usize)>, String> {
        let _g = no_grad();
        let (b, a, nc, rm) = (out.batch(), out.num_anchors(), out.num_classes, out.reg_max);
        if b != 1
            || out.box_raw.dims() != [1, 4 * rm, a]
            || out.cls_raw.dims() != [1, nc, a]
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
        Ok(Some((cands, out.mode, a)))
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
fn nms_config(cfg: &PredictConfig) -> NmsConfig {
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

/// Downloads the compacted `[1, 1 + 7 * A]` candidate buffer asynchronously: one readback of the count plus up
/// to `CANDIDATE_READBACK` records, and a second readback only when an image overflows that cap.
async fn download_candidate_records(ctx: &GpuContext, cands: &Tensor, a: usize) -> Result<Vec<u32>, String> {
    const W: usize = DETECT_RECORD_WORDS;
    let cap = CANDIDATE_READBACK.min(a);
    let head_view = cands.narrow(1, 0, 1 + W * cap).map_err(err)?;
    let head = folo_gpu::download_async(ctx, &head_view).await.map_err(err)?.to_vec::<u32>().map_err(err)?;
    let n = head[0] as usize;
    if n > a {
        return Err(format!("detect_candidates: {n} candidates for {a} anchors"));
    }
    if n <= cap {
        Ok(head[1..1 + W * n].to_vec())
    } else {
        let full_view = cands.narrow(1, 1, W * n).map_err(err)?;
        folo_gpu::download_async(ctx, &full_view).await.map_err(err)?.to_vec::<u32>().map_err(err)
    }
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
fn validate_rgba(rgba: &[u8], width: u32, height: u32) -> Result<usize, String> {
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
