//! The real store's link to the engine: device selection, GPU info from folo-gpu, error mapping and the predictor
//! cache of `POST /api/predict`.

use std::collections::HashMap;
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
use crate::util::Flight;

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
            .hint(
                "set `amp: false` to train in f32, or restart this server with tensor cores (`folo serve \
                 --tensor-cores`, or FOLO_GPU_TENSOR_CORES=1) on a GPU that has them; GET /api/system's \
                 `capabilities.amp` says when amp is available",
            ),
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

/// The physical adapters [`gpus`] lists, best first (software ones left out), before it moves the running GPU
/// backend's adapter to the front. Enumerated once per process (about 150 ms on Windows: a GPU instance of its own),
/// so `/api/system` runs it beside creating the device ([`crate::Store::prepare_gpus`]).
pub fn adapters() -> &'static [folo_gpu::AdapterSummary] {
    static ADAPTERS: OnceLock<Vec<folo_gpu::AdapterSummary>> = OnceLock::new();
    ADAPTERS.get_or_init(|| {
        let mut adapters: Vec<folo_gpu::AdapterSummary> =
            folo_gpu::GpuContext::enumerate(&folo_gpu::GpuOptions::from_env())
                .into_iter()
                .filter(|a| a.device_type != "Cpu" && api_backend(&a.backend).is_some())
                .collect();
        adapters.sort_by_key(adapter_rank);
        adapters
    })
}

/// `/api/system`'s `gpus`: each physical adapter once (through its preferred backend; software adapters are left
/// out), best first, so ordinal 0 is the adapter `gpu:0` uses. When the GPU backend is already running, its
/// adapter is ordinal 0. Enumerated once per process. `memory_bytes` is the adapter's largest device-local heap where
/// its backend is Vulkan (folo-gpu reads it there), else 0; wgpu does not report utilisation, so `util` is `null`.
pub fn gpus() -> Vec<GpuInfo> {
    static GPUS: OnceLock<Vec<GpuInfo>> = OnceLock::new();
    GPUS.get_or_init(|| {
        let mut adapters = adapters().to_vec();
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
                memory_bytes: a.memory_bytes,
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

/// Gives the GPU buffer pool of `device` back to the driver ([`folo_gpu::GpuContext::release_memory`]): after a
/// training run, and when inference has been idle for a while ([`crate::real::RealStore`]). Nothing for the CPU or a
/// device whose context was never created.
pub fn release_gpu_memory(device: Device) {
    let Some(ctx) = (!device.is_cpu()).then(|| folo_gpu::context_for(device)).flatten() else { return };
    match ctx.release_memory() {
        Ok(bytes) if bytes > 0 => {
            tracing::debug!(
                "released {:.1} MiB of pooled GPU memory on {device}",
                bytes as f64 / (1 << 20) as f64
            )
        }
        Ok(_) => {}
        Err(e) => tracing::warn!("releasing GPU memory on {device}: {e}"),
    }
}

/// `/api/system`'s `gpu_pool`: the buffer pool of the GPU `device: auto` uses, when its context exists (never
/// created by this call).
pub fn gpu_pool() -> Option<folo_api::GpuPool> {
    let ctx = folo_gpu::context_for(Device::Gpu(0))?;
    let s = ctx.pool_stats();
    Some(folo_api::GpuPool {
        device: Device::Gpu(0).to_string(),
        live_bytes: s.live_bytes,
        cached_bytes: s.cached_bytes,
        peak_live_bytes: s.peak_live_bytes,
    })
}

/// Opens `.folo` weights for inference (EMA when present, fused, eval mode) on `device` at `precision`
/// ([`effective_precision`]: f16 only where the device computes in it).
pub fn open_model(path: &Path, device: Device, precision: Precision) -> Result<FoloModel> {
    let file = read_model_file(path)?;
    // The forward kernels an earlier server used with this model compile on every core while it loads.
    let dtype = effective_precision(device, precision).dtype();
    folo_vision::precompile::precompile_inference(&file, device, dtype, "serve", "");
    if precision == Precision::F32 {
        let mut model = FoloModel::from_file(&file, Purpose::Inference).map_err(nn_error)?.0;
        model.fuse().map_err(nn_error)?;
        return place_model(model, device).map_err(engine_error);
    }
    can_place(device).map_err(engine_error)?;
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

type PredictorKey = (FileKey, Device, Precision);
type Cached = (FileKey, Device, Precision, Arc<Mutex<Loaded>>);
type PredictorFlight = Arc<Flight<Result<Arc<Mutex<Loaded>>>>>;

#[derive(Default)]
struct PredictorCache {
    entries: Vec<Cached>,
    in_flight: HashMap<PredictorKey, PredictorFlight>,
}

/// An LRU cache of loaded predictors keyed by weights file, device and precision, so repeated predictions skip
/// loading and fusing. Concurrent cache misses for the same key share one load (single-flight), while distinct
/// keys load in parallel. A predictor serves one request at a time (its thresholds are set per call).
#[derive(Default)]
pub struct Predictors {
    cache: Mutex<PredictorCache>,
}

impl Predictors {
    /// The predictor of `path` on `device` at `precision`, loaded on a miss (evicting the least recently used
    /// beyond [`PREDICTOR_CACHE`]). Concurrent misses for the same `(FileKey, Device, Precision)` share one load.
    pub fn get(&self, path: &Path, device: Device, precision: Precision) -> Result<Arc<Mutex<Loaded>>> {
        let key =
            FileKey::of(path).map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))?;
        self.get_or_load(key, device, precision, || {
            let model = open_model(path, device, precision)?;
            let cfg = PredictConfig { imgsz: model.imgsz(), ..PredictConfig::default() };
            let predictor = Predictor::new(model, cfg).map_err(engine_error)?;
            Ok(Arc::new(Mutex::new(Loaded { predictor, warm: false })))
        })
    }

    fn get_or_load(
        &self,
        key: FileKey,
        device: Device,
        precision: Precision,
        mut load: impl FnMut() -> Result<Arc<Mutex<Loaded>>>,
    ) -> Result<Arc<Mutex<Loaded>>> {
        let fkey = (key.clone(), device, precision);
        loop {
            let role = {
                let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(i) =
                    cache.entries.iter().position(|(k, d, q, _)| *k == key && *d == device && *q == precision)
                {
                    let hit = cache.entries.remove(i);
                    let p = hit.3.clone();
                    cache.entries.push(hit);
                    return Ok(p);
                }
                if let Some(existing) = cache.in_flight.get(&fkey).cloned() {
                    Err(existing)
                } else {
                    let created = Flight::new();
                    cache.in_flight.insert(fkey.clone(), created.clone());
                    Ok(created)
                }
            };
            match role {
                Err(existing) => {
                    if let Some(res) = existing.wait() {
                        return res;
                    }
                    // Leader aborted (panicked); retry so another caller can take over.
                }
                Ok(mine) => {
                    struct Cleanup<'a> {
                        cache: &'a Mutex<PredictorCache>,
                        key: PredictorKey,
                        flight: PredictorFlight,
                        done: bool,
                    }
                    impl Drop for Cleanup<'_> {
                        fn drop(&mut self) {
                            if !self.done {
                                let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
                                if cache
                                    .in_flight
                                    .get(&self.key)
                                    .is_some_and(|f| Arc::ptr_eq(f, &self.flight))
                                {
                                    cache.in_flight.remove(&self.key);
                                }
                                drop(cache);
                                self.flight.abort();
                            }
                        }
                    }
                    let mut guard =
                        Cleanup { cache: &self.cache, key: fkey.clone(), flight: mine.clone(), done: false };
                    let outcome = load();
                    {
                        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
                        let still_current = cache.in_flight.get(&fkey).is_some_and(|f| Arc::ptr_eq(f, &mine));
                        if still_current {
                            cache.in_flight.remove(&fkey);
                            if let Ok(p) = &outcome {
                                cache.entries.retain(|(k, d, q, _)| {
                                    !(k.path == key.path && *d == device && *q == precision)
                                });
                                cache.entries.push((key.clone(), device, precision, p.clone()));
                                while cache.entries.len() > PREDICTOR_CACHE {
                                    cache.entries.remove(0);
                                }
                            }
                        }
                    }
                    mine.finish(outcome.clone());
                    guard.done = true;
                    return outcome;
                }
            }
        }
    }

    /// Drops every cached predictor (inference has been idle; [`crate::real::PREDICTOR_IDLE`]). Returns whether one
    /// of them was on a GPU. Any load currently in flight will not be inserted into the cache when it finishes.
    pub fn clear(&self) -> bool {
        let (dropped, in_flight_gpu) = {
            let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            let dropped = std::mem::take(&mut cache.entries);
            let in_flight_gpu = cache.in_flight.keys().any(|(_, d, _)| !d.is_cpu());
            cache.in_flight.clear();
            (dropped, in_flight_gpu)
        };
        dropped.iter().any(|(_, d, _, _)| !d.is_cpu()) || in_flight_gpu
    }

    /// Drops every cached predictor of `path` (its file was replaced or deleted). Any load of `path` currently in
    /// flight will not be inserted into the cache when it finishes.
    pub fn forget(&self, path: &Path) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.entries.retain(|(k, _, _, _)| k.path != path);
        cache.in_flight.retain(|(k, _, _), _| k.path != path);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    fn tiny_loaded() -> Arc<Mutex<Loaded>> {
        let mut m = FoloModel::with_rng(
            folo_nn::io::ModelConfig::new("n", 1).unwrap(),
            &mut folo_core::rng::Rng::seed(1),
        )
        .unwrap();
        m.set_classes(vec!["cat".into()]).unwrap();
        m.set_imgsz(64);
        let cfg = PredictConfig { imgsz: 64, ..PredictConfig::default() };
        let predictor = Predictor::new(m, cfg).unwrap();
        Arc::new(Mutex::new(Loaded { predictor, warm: false }))
    }

    fn key(name: &str) -> FileKey {
        FileKey { path: PathBuf::from(name), len: 1, modified: None }
    }

    /// Concurrent misses for the same `(FileKey, Device, Precision)` run one load and share the same predictor;
    /// distinct keys load in parallel without blocking each other.
    #[test]
    fn predictors_singleflight_deduplicates_same_key_and_runs_distinct_keys_in_parallel() {
        let predictors = Predictors::default();
        let loads = AtomicUsize::new(0);
        let entered = Barrier::new(2);
        let release = Barrier::new(2);

        let loaded = tiny_loaded();
        let (p1, p2) = std::thread::scope(|s| {
            let h1 = s.spawn(|| {
                predictors
                    .get_or_load(key("a.folo"), Device::Cpu, Precision::F32, || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        entered.wait();
                        release.wait();
                        Ok(loaded.clone())
                    })
                    .unwrap()
            });
            // Wait until thread 1 is inside `load`, then launch a second caller for the same key.
            entered.wait();
            let h2 = s.spawn(|| {
                predictors
                    .get_or_load(key("a.folo"), Device::Cpu, Precision::F32, || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        Ok(loaded.clone())
                    })
                    .unwrap()
            });
            // Wait until thread 2 is parked in `Flight::wait` before releasing thread 1. Always release the
            // leader before asserting so a timeout never strands the scoped thread.
            let wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let fk = (key("a.folo"), Device::Cpu, Precision::F32);
            let mut parked = false;
            while std::time::Instant::now() < wait_deadline {
                if predictors.cache.lock().unwrap().in_flight[&fk].waiters() >= 1 {
                    parked = true;
                    break;
                }
                std::thread::yield_now();
            }
            release.wait();
            let res = (h1.join().unwrap(), h2.join().unwrap());
            assert!(parked, "waiter did not park in Flight::wait");
            res
        });
        assert_eq!(loads.load(Ordering::SeqCst), 1, "same key loads once");
        assert!(Arc::ptr_eq(&p1, &p2), "both callers receive the same predictor");

        // Distinct keys run in parallel (proved by a bounded 2-thread rendezvous inside both loaders).
        let both = crate::util::Rendezvous::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = ["b.folo", "c.folo"]
                .into_iter()
                .map(|name| {
                    let (preds, both, loaded) = (&predictors, &both, &loaded);
                    s.spawn(move || {
                        let mut met = false;
                        preds
                            .get_or_load(key(name), Device::Cpu, Precision::F32, || {
                                met = both.arrive(std::time::Duration::from_secs(5));
                                Ok(loaded.clone())
                            })
                            .unwrap();
                        assert!(met, "distinct predictor keys did not load concurrently");
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    /// If a leader panics while loading, waiting callers recover and retry cleanly; if `forget` runs during a
    /// load, the stale predictor is not cached afterwards.
    #[test]
    fn predictors_singleflight_recovers_from_panic_and_respects_forget() {
        let predictors = Predictors::default();
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        let loaded = tiny_loaded();

        // 1. Leader panics: waiter wakes and succeeds on retry.
        let panicking = AtomicBool::new(true);
        std::thread::scope(|s| {
            let leader = s.spawn(|| {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = predictors.get_or_load(key("panic.folo"), Device::Cpu, Precision::F32, || {
                        entered.wait();
                        release.wait();
                        if panicking.swap(false, Ordering::SeqCst) {
                            panic!("boom");
                        }
                        unreachable!()
                    });
                }));
            });
            entered.wait();
            let waiter = s.spawn(|| {
                predictors
                    .get_or_load(key("panic.folo"), Device::Cpu, Precision::F32, || Ok(loaded.clone()))
                    .unwrap()
            });
            let wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let fk = (key("panic.folo"), Device::Cpu, Precision::F32);
            let mut parked = false;
            while std::time::Instant::now() < wait_deadline {
                if predictors.cache.lock().unwrap().in_flight[&fk].waiters() >= 1 {
                    parked = true;
                    break;
                }
                std::thread::yield_now();
            }
            release.wait();
            leader.join().unwrap();
            let got = waiter.join().unwrap();
            assert!(parked, "waiter did not park in Flight::wait");
            assert!(Arc::ptr_eq(&got, &loaded));
        });

        // 2. `forget(path)` while loading prevents the stale result from populating the cache.
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        std::thread::scope(|s| {
            let h = s.spawn(|| {
                predictors
                    .get_or_load(key("stale.folo"), Device::Cpu, Precision::F32, || {
                        entered.wait();
                        release.wait();
                        Ok(loaded.clone())
                    })
                    .unwrap()
            });
            entered.wait();
            predictors.forget(Path::new("stale.folo"));
            release.wait();
            h.join().unwrap();
        });
        assert!(
            predictors
                .cache
                .lock()
                .unwrap()
                .entries
                .iter()
                .all(|(k, _, _, _)| k.path != Path::new("stale.folo")),
            "forgotten in-flight load must not repopulate the cache"
        );
    }

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
