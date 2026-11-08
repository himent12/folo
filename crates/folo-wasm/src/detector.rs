//! The detector behind the JS bindings, free of `wasm-bindgen` types so the workspace's native tests run it.
//!
//! [`Detector::load`] parses a `.folo` file, builds the fused eval-mode model and places it on WebGPU or the CPU
//! (`auto` tries WebGPU and checks it with one small forward pass, falling back to the CPU). [`Detector::detect`]
//! is [`folo_vision::Predictor`]'s pipeline (letterbox, forward, decode, NMS, unmap), except that on the GPU the
//! decoded head is read back with an awaited [`folo_gpu::download_async`]: the browser cannot block for the GPU.

use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_data::Image;
use folo_gpu::GpuContext;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::{FoloModel, HeadOutput, PredictConfig, Predictor, Timing};
use serde::{Deserialize, Serialize};

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
    /// folo-core's CPU kernels (wasm SIMD128 when built with it).
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
    /// Whether the CPU GEMM uses the wasm SIMD128 micro-kernel.
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
            match Self::load_gpu(&file, &cfg, &model_id).await {
                Ok(d) => return Ok(d),
                Err(e) if opts.backend == BackendChoice::Auto => fallback = Some(e),
                Err(e) => return Err(format!("WebGPU backend unavailable: {e}")),
            }
        }
        let predictor = Predictor::new(cpu_model(&file)?, cfg).map_err(err)?;
        Ok(Detector {
            predictor: RefCell::new(predictor),
            backend: Backend::Cpu,
            gpu: None,
            model_id,
            fallback,
        })
    }

    async fn load_gpu(file: &ModelFile, cfg: &PredictConfig, model_id: &str) -> Result<Self, String> {
        let device = folo_gpu::init_async().await.map_err(err)?;
        let ctx = GpuContext::shared().map_err(err)?;
        let mut model = cpu_model(file)?;
        model.to_device(device).map_err(err)?;
        let predictor = Predictor::new(model, cfg.clone()).map_err(err)?;
        let d = Detector {
            predictor: RefCell::new(predictor),
            backend: Backend::Webgpu,
            gpu: Some(ctx),
            model_id: model_id.to_string(),
            fallback: None,
        };
        // One small pass compiles the core kernels and surfaces shader or device errors now, so `auto` can
        // still fall back to the CPU.
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
        let img = rgba_to_image(rgba, width, height)?;
        let cfg = {
            let p = self.predictor.borrow();
            PredictConfig { conf: opts.conf, iou: opts.iou, max_det: opts.max_det, ..p.config().clone() }
        };
        // Validates the options now; they are applied again right before NMS, because another `detect` may
        // change the predictor's config while this one awaits its readback.
        self.predictor.borrow_mut().set_config(cfg.clone()).map_err(err)?;
        let (batch, pre, t1, out) = {
            let p = self.predictor.borrow();
            let batch = p.prepare(std::slice::from_ref(&img), p.config().rect).map_err(err)?;
            let pre = ms(t0);
            let t1 = Instant::now();
            let out = p.forward(&batch).map_err(err)?;
            (batch, pre, t1, out)
        };
        let nc = out.num_classes;
        let (decoded, inf, t2) = match &self.gpu {
            // The forward pass is queued work; decode on the GPU and await the one readback. The wait is
            // the GPU running the network, so it counts as inference here.
            Some(ctx) => {
                let dec = out.decoded_on_device().map_err(err)?;
                drop(out);
                let host = folo_gpu::download_async(ctx, &dec).await.map_err(err)?;
                let inf = ms(t1);
                let t2 = Instant::now();
                let all = host.to_vec::<f32>().map_err(err)?;
                let a = dec.dims()[1];
                (HeadOutput::split_decoded(&all, a, nc), inf, t2)
            }
            None => {
                let inf = ms(t1);
                let t2 = Instant::now();
                (out.decode_images().map_err(err)?, inf, t2)
            }
        };
        let mut p = self.predictor.borrow_mut();
        p.set_config(cfg).map_err(err)?;
        let dets = p.finish(&batch, decoded, nc);
        let post = ms(t2);
        let pred = batch.predictions(dets, Timing::new(pre, inf, post)).pop().expect("one image in, one out");
        Ok(DetectResult {
            response: folo_api::PredictResponse {
                model_id: self.model_id.clone(),
                image: folo_api::ImageSize { width, height },
                boxes: pred.scored_boxes(p.classes()),
                timing: pred.timing.to_api(),
                device: p.device().to_string(),
            },
            backend: self.backend,
        })
    }
}

/// RGBA → folo-data's RGB [`Image`], checking the size.
pub fn rgba_to_image(rgba: &[u8], width: u32, height: u32) -> Result<Image, String> {
    if width == 0 || height == 0 {
        return Err(format!("image is empty ({width}x{height})"));
    }
    let n = width as usize * height as usize;
    if rgba.len() != n * 4 {
        return Err(format!(
            "expected {} RGBA bytes for a {width}x{height} image, got {}",
            n * 4,
            rgba.len()
        ));
    }
    let mut data = Vec::with_capacity(n * 3);
    for px in rgba.as_chunks::<4>().0 {
        data.extend_from_slice(&px[..3]);
    }
    Ok(Image { width, height, data })
}
