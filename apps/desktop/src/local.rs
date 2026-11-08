//! Local inference: Playground can run a model in this process with folo-vision's predictor, on
//! this machine's GPU (folo-gpu) or CPU, instead of asking folo serve. Weights are the server's f16
//! export (`GET /api/models/{id}/weights?format=f16`), held in memory and revalidated by ETag
//! before each run, so a retrained model is picked up and an unreachable server still runs the
//! copy already loaded.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use folo_core::Device;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::{FoloModel, PredictConfig, Predictor};

use crate::api::{
    ApiError, ApiResult, Backend, ImageData, ImageSize, PredictInput, PredictRequest, PredictResponse,
    Weights,
};

/// Where a local prediction runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Gpu,
    Cpu,
}

/// Loaded predictors kept (a model on the GPU and on the CPU, or two models).
const KEEP: usize = 2;

struct Loaded {
    /// The backend the id belongs to (ids are only unique per server).
    server: String,
    model_id: String,
    target: Target,
    etag: String,
    predictor: Arc<Mutex<Predictor>>,
}

/// Most recently used last.
static CACHE: Mutex<Vec<Loaded>> = Mutex::new(Vec::new());

/// The GPU local inference uses (its folo-core device and adapter name), or why there is none. The
/// first call sets up folo-gpu, which blocks for a moment: call it off the UI thread.
pub fn gpu() -> Result<(Device, String), String> {
    static GPU: OnceLock<Result<(Device, String), String>> = OnceLock::new();
    GPU.get_or_init(|| {
        let device = folo_gpu::init().map_err(|e| e.to_string())?;
        let ctx = folo_gpu::GpuContext::shared().map_err(|e| e.to_string())?;
        Ok((device, ctx.summary().name.clone()))
    })
    .clone()
}

fn engine_error(e: impl Into<folo_vision::Error>) -> ApiError {
    let e = e.into();
    let code = match e.code() {
        "invalid_weights" => "invalid_weights",
        "invalid_config" => "invalid_config",
        _ => "local_inference_failed",
    };
    ApiError::new(code, e.to_string())
}

fn device(target: Target) -> ApiResult<(Device, String)> {
    match target {
        Target::Cpu => Ok((Device::Cpu, "CPU".into())),
        Target::Gpu => gpu().map(|(d, name)| (d, crate::ui::short_gpu(&name))).map_err(|e| {
            ApiError::new("no_gpu", format!("No GPU for local inference: {e}"))
                .with_hint("Choose Local CPU, or run on the server.")
        }),
    }
}

/// The predictor of `model_id` on `target`: the loaded one while the server says its weights are
/// unchanged (or can't be reached), else freshly downloaded and loaded.
fn predictor(api: &dyn Backend, model_id: &str, target: Target) -> ApiResult<Arc<Mutex<Predictor>>> {
    predictor_with(&api.describe(), model_id, target, |etag| api.model_weights(model_id, etag))
}

/// [`predictor`] over any weights source: `fetch(etag)` answers `None` when `etag` is current.
fn predictor_with(
    server: &str,
    model_id: &str,
    target: Target,
    fetch: impl FnOnce(Option<&str>) -> ApiResult<Option<Weights>>,
) -> ApiResult<Arc<Mutex<Predictor>>> {
    let is = |l: &Loaded| l.server == server && l.model_id == model_id && l.target == target;
    let held = CACHE.lock().unwrap().iter().find(|l| is(l)).map(|l| (l.etag.clone(), l.predictor.clone()));
    let fresh = match (fetch(held.as_ref().map(|h| h.0.as_str())), held) {
        (Ok(Some(w)), _) => w,
        // Unchanged, or the server is out of reach: the copy already here still runs.
        (Ok(None), Some(h)) => return Ok(touch(&is).unwrap_or(h.1)),
        (Err(e), Some(h)) if e.code == "server_unreachable" => return Ok(touch(&is).unwrap_or(h.1)),
        (Ok(None), None) => return Err(ApiError::new("bad_response", "The server sent no weights")),
        (Err(e), _) => return Err(e),
    };
    let (device, _) = device(target)?;
    let file = ModelFile::from_bytes(fresh.bytes).map_err(engine_error)?;
    let mut model = FoloModel::from_file(&file, Purpose::Inference).map_err(engine_error)?.0;
    model.fuse().map_err(engine_error)?;
    let model = folo_vision::train::place_model(model, device).map_err(engine_error)?;
    let cfg = PredictConfig { imgsz: model.imgsz(), ..PredictConfig::default() };
    let s = cfg.imgsz;
    let predictor = Predictor::new(model, cfg).map_err(engine_error)?;
    // A GPU's first pass builds its kernels (about a second): do it now, so the latency shown for
    // the first image is the model's, not the warm-up's.
    if target == Target::Gpu {
        predictor.predict(&folo_data::Image::new(s, s, [114; 3])).map_err(engine_error)?;
    }
    let predictor = Arc::new(Mutex::new(predictor));
    let mut cache = CACHE.lock().unwrap();
    cache.retain(|l| !is(l));
    cache.push(Loaded {
        server: server.into(),
        model_id: model_id.into(),
        target,
        etag: fresh.etag,
        predictor: predictor.clone(),
    });
    while cache.len() > KEEP {
        cache.remove(0);
    }
    Ok(predictor)
}

/// Mark a cached predictor most recently used and return it.
fn touch(is: &dyn Fn(&Loaded) -> bool) -> Option<Arc<Mutex<Predictor>>> {
    let mut cache = CACHE.lock().unwrap();
    let i = cache.iter().position(is)?;
    let l = cache.remove(i);
    let p = l.predictor.clone();
    cache.push(l);
    Some(p)
}

/// Drop every loaded model (the backend changed: its model ids mean something else now).
pub fn forget_all() {
    CACHE.lock().unwrap().clear();
}

/// `POST /api/predict`, answered in this process. `device` reads "local:<name>".
pub fn predict(api: &dyn Backend, req: &PredictRequest, target: Target) -> ApiResult<PredictResponse> {
    let (_, name) = device(target)?;
    let encoded;
    let bytes: &[u8] = match &req.input {
        PredictInput::File { bytes, .. } => bytes,
        PredictInput::Dataset { dataset_id, image_id } => match api.image(dataset_id, image_id, None)? {
            ImageData::Encoded(b) => {
                encoded = b;
                &encoded
            }
            ImageData::Rgba { .. } => {
                return Err(ApiError::new("not_supported", "Local inference needs an encoded image"));
            }
        },
    };
    run(&*predictor(api, &req.model_id, target)?, bytes, req, &name)
}

/// One prediction of encoded `bytes` with a loaded predictor.
fn run(
    predictor: &Mutex<Predictor>,
    bytes: &[u8],
    req: &PredictRequest,
    name: &str,
) -> ApiResult<PredictResponse> {
    let mut p = predictor.lock().unwrap();
    // Decoding counts as preprocessing, as it does on the server.
    let t = Instant::now();
    let img = folo_data::decode(bytes)
        .map_err(|e| ApiError::new("image_decode_failed", format!("Could not read the image: {e}")))?;
    let decode_ms = t.elapsed().as_secs_f64() * 1000.;
    let cfg = PredictConfig {
        conf: req.conf,
        iou: req.iou,
        max_det: req.max_det.max(1) as usize,
        ..p.config().clone()
    };
    p.set_config(cfg).map_err(engine_error)?;
    let out = p.predict(&img).map_err(engine_error)?;
    let timing = folo_vision::Timing::new(
        out.timing.preprocess_ms + decode_ms,
        out.timing.inference_ms,
        out.timing.postprocess_ms,
    );
    Ok(PredictResponse {
        model_id: req.model_id.clone(),
        image: ImageSize { width: out.width, height: out.height },
        boxes: out.scored_boxes(p.classes()),
        timing: timing.to_api(),
        device: format!("local:{name}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, untrained folo-n with two classes as f16 `.folo` bytes (what the server sends).
    fn weights() -> Vec<u8> {
        let mut m = FoloModel::new(folo_vision::ModelConfig::new("n", 2).unwrap()).unwrap();
        m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
        let path = std::env::temp_dir().join(format!("folo-desktop-local-{}.folo", std::process::id()));
        m.export(&path, folo_nn::io::FloatDType::F16).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 4) as u8, (y * 4) as u8, 128]));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Png).unwrap();
        out.into_inner()
    }

    #[test]
    fn weights_load_once_and_revalidate_by_etag() {
        let bytes = weights();
        let fresh = |etag: &str| Ok(Some(Weights { bytes: bytes.clone(), etag: etag.into() }));
        let srv = "test-local";
        let first = predictor_with(srv, "m_1", Target::Cpu, |held| {
            assert_eq!(held, None);
            fresh("\"a\"")
        })
        .unwrap();
        // Unchanged: the loaded model is reused, and the ETag it was loaded with is what's sent.
        let same = predictor_with(srv, "m_1", Target::Cpu, |held| {
            assert_eq!(held, Some("\"a\""));
            Ok(None)
        })
        .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        // The server is gone: the copy here still runs. Other errors (a deleted model) don't hide.
        let offline =
            predictor_with(srv, "m_1", Target::Cpu, |_| Err(ApiError::new("server_unreachable", "x")));
        assert!(Arc::ptr_eq(&first, &offline.unwrap()));
        let gone = predictor_with(srv, "m_1", Target::Cpu, |_| Err(ApiError::new("model_not_found", "x")));
        assert_eq!(gone.err().map(|e| e.code), Some("model_not_found".into()));
        // Retrained: a new ETag loads the new file.
        let retrained = predictor_with(srv, "m_1", Target::Cpu, |_| fresh("\"b\"")).unwrap();
        assert!(!Arc::ptr_eq(&first, &retrained));
        // Same id on another server is another model.
        let other = predictor_with("elsewhere", "m_1", Target::Cpu, |held| {
            assert_eq!(held, None);
            fresh("\"a\"")
        });
        assert!(other.is_ok());

        let req = PredictRequest {
            model_id: "m_1".into(),
            input: PredictInput::File { name: "x.png".into(), bytes: Arc::new(png(64, 48)) },
            conf: 0.05,
            iou: 0.7,
            max_det: 300,
        };
        let r = run(&retrained, &png(64, 48), &req, "CPU").unwrap();
        assert_eq!((r.image.width, r.image.height), (64, 48));
        assert_eq!(r.device, "local:CPU");
        assert!(r.timing.total_ms > 0.);
        assert!(r.boxes.iter().all(|b| b.score >= 0.05 && b.x1 >= 0. && b.x2 <= 64.));
        let bad = run(&retrained, b"not an image", &req, "CPU");
        assert_eq!(bad.err().map(|e| e.code), Some("image_decode_failed".into()));
    }

    /// Against a real `folo serve` with a trained model: `FOLO_SERVER=http://127.0.0.1:7879 cargo
    /// test real_server -- --ignored --nocapture`. Local CPU and GPU find what the server finds.
    #[test]
    #[ignore]
    fn real_server_local_matches_the_server() {
        use crate::api::{Backend, ImageQuery, http::HttpBackend};
        let api = HttpBackend::new(&std::env::var("FOLO_SERVER").unwrap_or("http://127.0.0.1:7878".into()));
        let model = api.models().unwrap().into_iter().next().expect("a trained model");
        let ds = api.datasets().unwrap().into_iter().next().expect("a dataset");
        let q = ImageQuery { split: None, offset: 0, limit: 1, class: None };
        let image = api.images(&ds.id, &q).unwrap().images.remove(0);
        let req = PredictRequest {
            model_id: model.id.clone(),
            input: PredictInput::Dataset { dataset_id: ds.id.clone(), image_id: image.id.clone() },
            conf: 0.05,
            iou: 0.7,
            max_det: 300,
        };
        let server = api.predict(&req).unwrap();
        let top = |r: &PredictResponse| r.boxes.iter().filter(|b| b.score >= 0.25).count();
        println!(
            "server  {:>3} boxes, {:>3} >= 0.25, {:.1} ms on {}",
            server.boxes.len(),
            top(&server),
            server.timing.total_ms,
            server.device
        );
        for target in [Target::Cpu, Target::Gpu] {
            if target == Target::Gpu && gpu().is_err() {
                println!("no GPU: {:?}", gpu().err());
                continue;
            }
            // Twice: the second run reuses the loaded model (a 304).
            for _ in 0..2 {
                let local = predict(&api, &req, target).unwrap();
                println!(
                    "{:<7} {:>3} boxes, {:>3} >= 0.25, {:.1} ms on {}",
                    format!("{target:?}"),
                    local.boxes.len(),
                    top(&local),
                    local.timing.total_ms,
                    local.device
                );
                assert_eq!(
                    (local.image.width, local.image.height),
                    (server.image.width, server.image.height)
                );
                // f16 weights against the server's f32 checkpoint: the confident boxes agree.
                let n = top(&server).max(1) as i64;
                assert!(
                    (top(&local) as i64 - top(&server) as i64).abs() <= (n / 5).max(1),
                    "{} vs {}",
                    top(&local),
                    top(&server)
                );
                if let (Some(a), Some(b)) = (server.boxes.first(), local.boxes.first()) {
                    assert_eq!(a.class, b.class);
                    assert!(
                        (a.score - b.score).abs() < 0.05
                            && (a.x1 - b.x1).abs() < 2.
                            && (a.y2 - b.y2).abs() < 2.
                    );
                }
            }
        }
    }
}
