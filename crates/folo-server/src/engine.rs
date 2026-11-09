//! The real store's link to the engine: device selection, GPU info from folo-gpu, error mapping and the predictor
//! cache of `POST /api/predict`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use axum::http::StatusCode;
use folo_api::{AmpSupport, Capabilities, GpuBackend, GpuInfo, TensorCoreSupport};
use folo_core::Device;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::predict::f16_unavailable;
use folo_vision::train::{can_place, place_model, resolve_device};
use folo_vision::{FoloModel, Precision, PredictConfig, Predictor};

use crate::error::{Error, Result};

/// Loaded models kept by [`Predictors`] (least recently used first out).
pub const PREDICTOR_CACHE: usize = 4;

/// Maps an engine error to the API envelope (codes and statuses of `folo_api::codes`): settings → `400
/// invalid_config`, unusable data → `400 invalid_dataset`, a device this machine does not have → `404
/// device_not_found` (the hint lists the ones it has), GPU memory exhausted → `507 gpu_out_of_memory`, a feature
/// the engine lacks → `501 not_implemented`, bad weights → `422 invalid_weights`, the rest → `500 internal`.
/// Messages are capped ([`folo_vision::Error::summary`]).
pub fn engine_error(e: folo_vision::Error) -> Error {
    use folo_vision::Error as E;
    let msg = e.summary();
    match &e {
        E::Config(_) => Error::new(StatusCode::BAD_REQUEST, "invalid_config", msg)
            .hint("TrainConfig in docs/API.md lists every field's accepted values (over MCP: the train tool's inputSchema)"),
        E::Dataset(_) => Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", msg),
        E::DeviceNotFound(_) => Error::new(StatusCode::NOT_FOUND, "device_not_found", msg).hint(format!(
            "this machine has {}; use device `auto` (the GPU when there is one) or `cpu`",
            crate::system::available_devices()
        )),
        _ if e.is_gpu_out_of_memory() => {
            Error::new(StatusCode::INSUFFICIENT_STORAGE, "gpu_out_of_memory", msg)
                .hint("lower `batch` or `imgsz`, or use a smaller `arch` (folo-n, folo-s)")
        }
        E::Unsupported(_) => Error::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", msg),
        E::NotSupported(_) => Error::new(StatusCode::NOT_IMPLEMENTED, "not_supported", msg)
            .hint("train without `amp`, or start `folo serve --tensor-cores` on a GPU with tensor cores"),
        _ if e.code() == "invalid_weights" => {
            Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_weights", msg)
        }
        // Everything else keeps the engine's code with its status from the shared vocabulary (`training_failed`,
        // `data_error`, `io_error`, `internal`, ...).
        _ => {
            let code = e.code();
            let status = folo_api::codes::find(code)
                .and_then(|c| StatusCode::from_u16(c.status).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Error::new(status, code, msg)
        }
    }
}

/// A folo-nn error through [`engine_error`].
pub fn nn_error(e: folo_nn::Error) -> Error {
    engine_error(e.into())
}

fn is_auto(device: &str) -> bool {
    matches!(device.trim().to_ascii_lowercase().as_str(), "" | "auto")
}

/// Registers folo-gpu's backend (once per process) when `device` may use a GPU (`auto`, `gpu`, `gpu:<n>`).
fn init_gpu_for(device: &str) {
    let d = device.trim().to_ascii_lowercase();
    if (is_auto(&d) || d.starts_with("gpu") || d.starts_with("cuda"))
        && let Err(e) = folo_gpu::init()
    {
        tracing::debug!("no GPU backend: {e}");
    }
}

/// The device a run or a model runs on. `auto` is the GPU when folo-gpu finds an adapter and models can be placed
/// on it ([`can_place`]), else the CPU; an explicit `gpu:<n>` that cannot be used is an error.
pub fn resolve(device: &str) -> Result<Device> {
    init_gpu_for(device);
    let d = resolve_device(device).map_err(engine_error)?;
    if is_auto(device) && can_place(d).is_err() {
        return Ok(Device::Cpu);
    }
    can_place(d).map_err(engine_error)?;
    Ok(d)
}

/// What `Run.device` says: the adapter name and backend for a GPU, the CPU's name for the CPU.
pub fn describe(device: Device) -> String {
    match device {
        Device::Cpu => format!("{} (cpu)", crate::system::host_info().cpu_name),
        Device::Gpu(_) => match folo_gpu::GpuContext::shared() {
            Ok(ctx) => format!("{} ({})", ctx.summary().name, ctx.summary().backend.to_ascii_lowercase()),
            Err(_) => device.to_string(),
        },
    }
}

fn api_backend(name: &str) -> Option<GpuBackend> {
    match name {
        "Vulkan" => Some(GpuBackend::Vulkan),
        "Dx12" => Some(GpuBackend::Dx12),
        "Metal" => Some(GpuBackend::Metal),
        _ => None,
    }
}

fn adapter_rank(a: &folo_gpu::AdapterSummary) -> (u8, u8) {
    let ty = match a.device_type.as_str() {
        "DiscreteGpu" => 0,
        "IntegratedGpu" => 1,
        "VirtualGpu" => 2,
        _ => 3,
    };
    // folo-gpu prefers Vulkan / Metal over DX12 for the same adapter.
    let backend = match a.backend.as_str() {
        "Vulkan" | "Metal" => 0,
        _ => 1,
    };
    (ty, backend)
}

/// `/api/system`'s `gpus`: each physical adapter once (through its preferred backend; software adapters are left
/// out), best first, so ordinal 0 is the adapter `gpu:0` uses. When the GPU backend is already running, its
/// adapter is ordinal 0. Enumerated once per process. wgpu does not report memory size or utilisation, so
/// `memory_bytes` is 0 and `util` is `null`.
pub fn gpus() -> Vec<GpuInfo> {
    static GPUS: OnceLock<Vec<GpuInfo>> = OnceLock::new();
    GPUS.get_or_init(|| {
        let mut adapters: Vec<folo_gpu::AdapterSummary> =
            folo_gpu::GpuContext::enumerate(&folo_gpu::GpuOptions::from_env())
                .into_iter()
                .filter(|a| a.device_type != "Cpu" && api_backend(&a.backend).is_some())
                .collect();
        adapters.sort_by_key(adapter_rank);
        if folo_core::backend::backend(Device::Gpu(0)).is_ok()
            && let Ok(ctx) = folo_gpu::GpuContext::shared()
        {
            let s = ctx.summary();
            if let Some(i) = adapters.iter().position(|a| a.name == s.name && a.backend == s.backend) {
                let a = adapters.remove(i);
                adapters.insert(0, a);
            }
        }
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for a in adapters {
            let key = (a.vendor, a.device, a.name.clone());
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            out.push(GpuInfo {
                ordinal: out.len() as u32,
                name: a.name.clone(),
                backend: api_backend(&a.backend).expect("filtered above"),
                memory_bytes: 0,
                util: None,
            });
        }
        out
    })
    .clone()
}

/// `/api/system`'s `capabilities`: what a run with `device: auto` gets on this server. The device comes from
/// [`resolve`] (so this registers the GPU backend on first use, as a run would); `shader_f16` is
/// [`f16_unavailable`] (what f16 inference falls back on), tensor cores come from the device's folo-gpu context
/// (`reason` is its status, the line `folo doctor` prints), and `amp` is the trainer's own
/// [`folo_vision::train::amp_support`] answer for that device ([`folo_vision::train::amp_available`]: the same check
/// without switching tensor cores back on, so a read changes nothing).
pub fn capabilities() -> Capabilities {
    let device = resolve("auto").unwrap_or(Device::Cpu);
    let ctx = (!device.is_cpu()).then(|| folo_gpu::context_for(device)).flatten();
    let tensor_cores = match &ctx {
        Some(ctx) => {
            let enabled = ctx.tensor_cores();
            TensorCoreSupport {
                available: ctx.summary().tensor_cores,
                enabled,
                reason: (!enabled).then(|| ctx.tensor_cores_status()),
            }
        }
        None => TensorCoreSupport {
            available: false,
            enabled: false,
            reason: Some(match folo_gpu::init() {
                Err(e) => format!("off: no usable GPU ({e})"),
                Ok(_) => "off: device auto runs on the CPU".to_string(),
            }),
        },
    };
    // Only the trainer's check decides; its `not_supported` message is the reason.
    let amp = match folo_vision::train::amp_available(device) {
        Ok(()) => AmpSupport { available: true, reason: None },
        Err(e) => AmpSupport { available: false, reason: Some(e.to_string()) },
    };
    Capabilities {
        device: device.to_string(),
        shader_f16: f16_unavailable(device).is_none(),
        tensor_cores,
        amp,
    }
}

/// Reads a `.folo` file into memory (never memory-mapped: on Windows a mapped file cannot be replaced, and a run's
/// checkpoints are replaced while it trains).
pub fn read_model_file(path: &Path) -> Result<ModelFile> {
    let bytes = std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::new(
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!("{} no longer exists", path.display()),
            )
        } else {
            Error::internal(format!("reading {}: {e}", path.display()))
        }
    })?;
    ModelFile::from_bytes(bytes).map_err(nn_error)
}

/// The throughput table's name for `device` ([`crate::throughput::DeviceId`]): `gpu:<n>` with the adapter's name,
/// or `cpu` with the CPU's.
pub fn device_id(device: Device) -> crate::throughput::DeviceId {
    static CPU_NAME: OnceLock<String> = OnceLock::new();
    let device_name = match device {
        Device::Cpu => CPU_NAME.get_or_init(|| crate::system::host_info().cpu_name).clone(),
        Device::Gpu(_) => {
            folo_gpu::context_for(device).map_or_else(|| device.to_string(), |c| c.summary().name.clone())
        }
    };
    crate::throughput::DeviceId { device: device.to_string(), device_name }
}

/// The tensor-core kernels are on for `device`.
pub fn tensor_cores_on(device: Device) -> bool {
    folo_gpu::context_for(device).is_some_and(|c| c.tensor_cores())
}

/// Opens `.folo` weights for inference (EMA when present, fused, eval mode) on `device` at `precision`
/// ([`effective_precision`]: f16 only where the device computes in it).
pub fn open_model(path: &Path, device: Device, precision: Precision) -> Result<FoloModel> {
    let file = read_model_file(path)?;
    if precision == Precision::F32 {
        let mut model = FoloModel::from_file(&file, Purpose::Inference).map_err(nn_error)?.0;
        model.fuse().map_err(nn_error)?;
        return place_model(model, device).map_err(engine_error);
    }
    can_place(device).map_err(engine_error)?;
    let dtype = effective_precision(device, precision).dtype();
    FoloModel::from_file_for_inference(&file, device, dtype).map_err(nn_error)
}

/// The precision a model runs in on `device` when `wanted` is asked for: f16 falls back to f32 (with a warning
/// in the log) where the device has no f16 compute (the CPU, an adapter without `shader-f16`).
pub fn effective_precision(device: Device, wanted: Precision) -> Precision {
    match (wanted, f16_unavailable(device)) {
        (Precision::F16, Some(why)) => {
            tracing::warn!("precision f16: {why}");
            Precision::F32
        }
        (p, _) => p,
    }
}

/// The identity of a weights file on disk: path, size and modification time. A changed file is a new key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileKey {
    /// The file.
    pub path: PathBuf,
    /// Its size.
    pub len: u64,
    /// Its modification time.
    pub modified: Option<SystemTime>,
}

impl FileKey {
    /// The key of `path` now.
    pub fn of(path: &Path) -> std::io::Result<FileKey> {
        let m = std::fs::metadata(path)?;
        Ok(FileKey { path: path.to_path_buf(), len: m.len(), modified: m.modified().ok() })
    }
}

/// A cached predictor and whether it has predicted yet (its first prediction includes warm-up: pipeline
/// compilation, allocations). The flag lives under the predictor's lock, so exactly one request sees it unset.
pub struct Loaded {
    /// The predictor.
    pub predictor: Predictor,
    /// It has predicted at least once.
    pub warm: bool,
}

type Cached = (FileKey, Device, Precision, Arc<Mutex<Loaded>>);

/// An LRU cache of loaded predictors keyed by weights file, device and precision, so repeated predictions skip
/// loading and fusing. A predictor serves one request at a time (its thresholds are set per call).
#[derive(Default)]
pub struct Predictors {
    entries: Mutex<Vec<Cached>>,
}

impl Predictors {
    /// The predictor of `path` on `device` at `precision`, loaded on a miss (evicting the least recently used
    /// beyond [`PREDICTOR_CACHE`]).
    pub fn get(&self, path: &Path, device: Device, precision: Precision) -> Result<Arc<Mutex<Loaded>>> {
        let key =
            FileKey::of(path).map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))?;
        {
            let mut entries = self.entries.lock().unwrap();
            if let Some(i) =
                entries.iter().position(|(k, d, q, _)| *k == key && *d == device && *q == precision)
            {
                let hit = entries.remove(i);
                let p = hit.3.clone();
                entries.push(hit);
                return Ok(p);
            }
        }
        // Loaded outside the lock: other models keep serving meanwhile (two misses on one file both load it).
        let model = open_model(path, device, precision)?;
        let cfg = PredictConfig { imgsz: model.imgsz(), ..PredictConfig::default() };
        let predictor = Predictor::new(model, cfg).map_err(engine_error)?;
        let p = Arc::new(Mutex::new(Loaded { predictor, warm: false }));
        let mut entries = self.entries.lock().unwrap();
        entries.retain(|(k, d, q, _)| !(k.path == key.path && *d == device && *q == precision));
        entries.push((key, device, precision, p.clone()));
        while entries.len() > PREDICTOR_CACHE {
            entries.remove(0);
        }
        Ok(p)
    }

    /// Drops every cached predictor of `path` (its file was replaced or deleted).
    pub fn forget(&self, path: &Path) {
        self.entries.lock().unwrap().retain(|(k, _, _, _)| k.path != path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every engine error keeps its stable code over HTTP; `not_supported` (an `amp` run without tensor cores) is
    /// 501 with a hint, not a 500.
    #[test]
    fn engine_errors_keep_their_codes() {
        use folo_vision::Error as E;
        let e = engine_error(E::NotSupported("mixed-precision training (amp) needs tensor cores".into()));
        assert_eq!((e.status, e.body.code.as_str()), (StatusCode::NOT_IMPLEMENTED, "not_supported"));
        assert!(e.body.hint.is_some());
        for err in [E::Config("x".into()), E::Unsupported("x".into()), E::Failed("x".into())] {
            let code = err.code();
            let got = engine_error(err);
            assert_eq!(got.body.code, code);
            assert_eq!(got.status.as_u16(), folo_api::codes::find(code).unwrap().status);
        }
    }
}
