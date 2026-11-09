//! [`GpuContext`]: adapter selection, device + queue, buffer pool, pipeline cache and command batching.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};

use crate::error::{GpuError, Result};
use crate::pipeline::{Kernel, KernelDef, KernelKey, Params, Spec, UNIFORM_SLOT};
use crate::pool::{BufferPool, GpuBuffer, PoolStats, size_class};
use crate::profile::{ProfileRow, Profiler};

/// Environment variable that picks the adapter: an index into [`GpuContext::enumerate`] or a case-insensitive
/// substring of the adapter name (for example `FOLO_GPU_ADAPTER=4080` or `FOLO_GPU_ADAPTER=1`).
pub const ENV_ADAPTER: &str = "FOLO_GPU_ADAPTER";
/// Environment variable that restricts the backends: a comma-separated list of `vulkan`, `dx12`, `metal`,
/// `webgpu` (for example `FOLO_GPU_BACKEND=dx12`).
pub const ENV_BACKEND: &str = "FOLO_GPU_BACKEND";
/// Environment variable that disables f16 kernels even when the adapter has `shader-f16` (`FOLO_GPU_NO_F16=1`).
pub const ENV_NO_F16: &str = "FOLO_GPU_NO_F16";
/// Environment variable that opts into the tensor-core (cooperative-matrix) f16 kernels
/// (`FOLO_GPU_TENSOR_CORES=1`); see [`GpuOptions::tensor_cores`].
pub const ENV_TENSOR_CORES: &str = "FOLO_GPU_TENSOR_CORES";

/// [`GpuContext::tensor_cores_status`] of a context created without [`GpuOptions::tensor_cores`].
pub const TENSOR_CORES_NOT_REQUESTED: &str = "off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)";

/// Set by [`request_tensor_cores`]: contexts built from [`GpuOptions::from_env`] opt in even without
/// [`ENV_TENSOR_CORES`].
static TENSOR_CORES_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Held while the shared context is created and while [`request_tensor_cores`] looks at it, so a request either
/// lands before the options are read or sees the finished context.
static SHARED_INIT: Mutex<()> = Mutex::new(());

/// Asks for tensor cores in every context later built from [`GpuOptions::from_env`], including the process-wide
/// [`GpuContext::shared`] one (what `--tensor-cores` on the CLI and the server option call). It only has an effect
/// before that context is created: returns `false` when the shared context already exists without them.
pub fn request_tensor_cores() -> bool {
    let _creating = SHARED_INIT.lock().unwrap_or_else(|p| p.into_inner());
    TENSOR_CORES_REQUESTED.store(true, Ordering::Relaxed);
    match SHARED.get() {
        Some(Ok(c)) => c.caps().coop_f16,
        Some(Err(_)) => false,
        None => true,
    }
}

/// Dispatches encoded before the open batch is submitted on its own (the default of
/// [`GpuContext::set_flush_interval`]).
pub const MAX_DISPATCHES_PER_BATCH: u32 = 1024;
/// Size of one uniform arena chunk (256 parameter slots).
const UNIFORM_CHUNK: u64 = 64 * 1024;

/// How to pick and configure the adapter.
#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Adapter filter: an index into [`GpuContext::enumerate`] or a name substring. `None` picks the best one.
    pub adapter: Option<String>,
    /// Backends to consider. `None` means Vulkan, Metal, DX12 and browser WebGPU.
    pub backends: Option<wgpu::Backends>,
    /// Power preference used to rank adapters when no filter is given.
    pub power_preference: wgpu::PowerPreference,
    /// Request `shader-f16` when the adapter has it.
    pub enable_f16: bool,
    /// Opt into the tensor-core f16 GEMM and conv kernels (wgpu's experimental cooperative matrices, see
    /// [`crate::kernels::coop`]). They are used only when the adapter reports 16×16×16 f16 → f32 cooperative
    /// matrices and `shader-f16`, and a start-up self-test on the device passes; otherwise the context keeps the
    /// regular kernels and [`GpuContext::tensor_cores_status`] says why.
    pub tensor_cores: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            adapter: None,
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            enable_f16: true,
            tensor_cores: false,
        }
    }
}

impl GpuOptions {
    /// Defaults overridden by [`ENV_ADAPTER`], [`ENV_BACKEND`], [`ENV_NO_F16`] and [`ENV_TENSOR_CORES`] (or
    /// [`request_tensor_cores`]).
    pub fn from_env() -> Self {
        let mut o = Self::default();
        if let Ok(a) = std::env::var(ENV_ADAPTER)
            && !a.trim().is_empty()
        {
            o.adapter = Some(a.trim().to_string());
        }
        if let Ok(b) = std::env::var(ENV_BACKEND) {
            o.backends = parse_backends(&b);
        }
        if std::env::var(ENV_NO_F16).is_ok_and(|v| v != "0" && !v.is_empty()) {
            o.enable_f16 = false;
        }
        o.tensor_cores = TENSOR_CORES_REQUESTED.load(Ordering::Relaxed)
            || std::env::var(ENV_TENSOR_CORES).is_ok_and(|v| v != "0" && !v.is_empty());
        o
    }
}

/// Parses `vulkan,dx12,...` into [`wgpu::Backends`]; `None` for an empty or unrecognised list.
pub fn parse_backends(s: &str) -> Option<wgpu::Backends> {
    let mut b = wgpu::Backends::empty();
    for part in s.split(',').map(|p| p.trim().to_ascii_lowercase()) {
        b |= match part.as_str() {
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "dx12" | "d3d12" => wgpu::Backends::DX12,
            "metal" | "mtl" => wgpu::Backends::METAL,
            "webgpu" | "browser" => wgpu::Backends::BROWSER_WEBGPU,
            "all" => wgpu::Backends::all(),
            _ => continue,
        };
    }
    (!b.is_empty()).then_some(b)
}

/// What `folo doctor` prints about an adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    /// Index in [`GpuContext::enumerate`] order (usable as `FOLO_GPU_ADAPTER`).
    pub index: usize,
    /// Adapter name as reported by the driver.
    pub name: String,
    /// PCI vendor id.
    pub vendor: u32,
    /// PCI device id.
    pub device: u32,
    /// `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu` or `Other`.
    pub device_type: String,
    /// Backend (`Vulkan`, `Dx12`, `Metal`, `Gl`, `BrowserWebGpu`).
    pub backend: String,
    /// Driver name.
    pub driver: String,
    /// Driver version / info string.
    pub driver_info: String,
    /// Whether the adapter supports `shader-f16`.
    pub shader_f16: bool,
    /// Whether the adapter offers the 16×16×16 f16 → f32 cooperative matrices the tensor-core kernels need
    /// (wgpu's experimental `EXPERIMENTAL_COOPERATIVE_MATRIX`; native Vulkan / Metal only).
    pub tensor_cores: bool,
    /// Largest buffer in bytes.
    pub max_buffer_size: u64,
    /// Largest storage buffer binding in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Workgroup (shared) memory in bytes.
    pub max_compute_workgroup_storage_size: u32,
    /// Invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Subgroup (warp / wave) size range.
    pub subgroup_size: (u32, u32),
}

impl AdapterSummary {
    fn new(index: usize, adapter: &wgpu::Adapter) -> Self {
        let i = adapter.get_info();
        let l = adapter.limits();
        Self {
            index,
            name: i.name,
            vendor: i.vendor,
            device: i.device,
            device_type: format!("{:?}", i.device_type),
            backend: format!("{:?}", i.backend),
            driver: i.driver,
            driver_info: i.driver_info,
            shader_f16: adapter.features().contains(wgpu::Features::SHADER_F16),
            tensor_cores: crate::kernels::coop::adapter_supports(adapter),
            max_buffer_size: l.max_buffer_size,
            max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
            subgroup_size: (i.subgroup_min_size, i.subgroup_max_size),
        }
    }
}

impl std::fmt::Display for AdapterSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({}, {}) driver {} {}; shader-f16 {}; tensor cores {}; max buffer {} MiB, storage binding {} MiB, \
             workgroup storage {} KiB, {} invocations/workgroup, subgroup {}..{}",
            self.index,
            self.name,
            self.backend,
            self.device_type,
            self.driver,
            self.driver_info,
            if self.shader_f16 { "yes" } else { "no" },
            if self.tensor_cores { "yes" } else { "no" },
            self.max_buffer_size >> 20,
            self.max_storage_buffer_binding_size >> 20,
            self.max_compute_workgroup_storage_size >> 10,
            self.max_compute_invocations_per_workgroup,
            self.subgroup_size.0,
            self.subgroup_size.1,
        )
    }
}

/// Device capabilities the kernels consult.
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    /// f16 kernels are available (`shader-f16` enabled on the device).
    pub shader_f16: bool,
    /// The device was created with cooperative matrices and passed the tensor-core self-test, so the f16 GEMM
    /// and conv kernels can run on tensor cores (see [`GpuContext::tensor_cores`] for the run-time switch).
    pub coop_f16: bool,
    /// Largest buffer in bytes.
    pub max_buffer_size: u64,
    /// Largest storage binding in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Workgroup memory in bytes.
    pub max_workgroup_storage: u32,
    /// Workgroups per dispatch dimension.
    pub max_workgroups_per_dim: u32,
    /// Uniform buffer offset alignment.
    pub uniform_align: u32,
}

struct UniformChunk {
    buf: wgpu::Buffer,
    cpu: Vec<u8>,
}

#[derive(Default)]
struct Batch {
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    dispatches: u32,
    chunks: Vec<UniformChunk>,
    /// Index of the chunk being filled and the byte cursor in it.
    chunk: usize,
    cursor: u64,
}

struct Inner {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    summary: AdapterSummary,
    caps: Capabilities,
    pool: Arc<Mutex<BufferPool>>,
    kernels: Mutex<HashMap<KernelKey, Arc<Kernel>>>,
    batch: Mutex<Batch>,
    errors: Arc<Mutex<Vec<String>>>,
    /// Set once the device is lost (by wgpu's device-lost callback, or by a poll that failed fatally).
    lost: Arc<Mutex<Option<String>>>,
    dummy: wgpu::Buffer,
    dispatches: AtomicU64,
    submissions: AtomicU64,
    /// Tensor-core kernels in use (only ever true when `caps.coop_f16`); cleared by
    /// [`GpuContext::set_tensor_cores`] or when a tensor-core kernel fails to build.
    tensor_cores: AtomicBool,
    /// `"on"`, or why tensor cores are off.
    tensor_cores_status: Mutex<String>,
    /// The device was created with wgpu's experimental cooperative matrices (see [`GpuContext::device`]).
    experimental: bool,
    /// Dispatches after which the open batch is submitted on its own ([`GpuContext::set_flush_interval`]).
    flush_every: AtomicU32,
    /// Per-kernel timestamp queries (`FOLO_GPU_PROFILE=1`, see [`crate::profile`]).
    profiler: Option<Mutex<Profiler>>,
}

/// Host data in a staging buffer, made by [`GpuContext::stage`] (on any thread) and copied to the device with
/// [`GpuContext::upload_staged`].
pub struct Staged {
    buf: wgpu::Buffer,
    len: u64,
}

impl Staged {
    /// Bytes staged.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether nothing was staged.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Work counters since the context was created (see [`GpuContext::work_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkStats {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// Command buffers submitted to the queue.
    pub submissions: u64,
}

/// A wgpu device and queue plus FOLO's buffer pool, pipeline cache and command batch.
///
/// Cheap to clone (an `Arc` inside) and `Send + Sync` on native targets. Kernels encode into an open batch;
/// the batch is submitted when it grows large, before uploads and readbacks, and on [`GpuContext::flush`] /
/// [`GpuContext::sync`].
#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext").field("adapter", &self.inner.summary.name).finish()
    }
}

fn instance(backends: wgpu::Backends) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

fn default_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12 | wgpu::Backends::BROWSER_WEBGPU
}

fn adapter_score(a: &wgpu::Adapter, power: wgpu::PowerPreference) -> (u32, u32) {
    let info = a.get_info();
    let ty = match (info.device_type, power) {
        (wgpu::DeviceType::DiscreteGpu, wgpu::PowerPreference::LowPower) => 3,
        (wgpu::DeviceType::IntegratedGpu, wgpu::PowerPreference::LowPower) => 4,
        (wgpu::DeviceType::DiscreteGpu, _) => 4,
        (wgpu::DeviceType::IntegratedGpu, _) => 3,
        (wgpu::DeviceType::VirtualGpu, _) => 2,
        (wgpu::DeviceType::Other, _) => 1,
        (wgpu::DeviceType::Cpu, _) => 0,
    };
    // Vulkan's generated SPIR-V is the fastest path on NVIDIA / AMD in our GEMM benchmarks.
    let backend = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::BrowserWebGpu => 3,
        wgpu::Backend::Dx12 => 2,
        _ => 1,
    };
    (ty, backend)
}

/// The context behind [`GpuContext::shared`] (set by the first `shared` / `shared_async` call).
static SHARED: OnceLock<std::result::Result<GpuContext, String>> = OnceLock::new();

/// The error the blocking entry points return in the browser, where nothing may block the page's thread.
#[cfg(target_arch = "wasm32")]
const NO_BLOCKING: &str = "blocking GPU calls are not available in the browser; use the async variants \
     (GpuContext::shared_async, folo_gpu::init_async, folo_gpu::download_async)";

impl GpuContext {
    /// Lists the adapters visible with `options.backends` (ignores the adapter filter). Empty in the browser,
    /// where WebGPU hands out one adapter on request instead of enumerating.
    pub fn enumerate(options: &GpuOptions) -> Vec<AdapterSummary> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let inst = instance(options.backends.unwrap_or_else(default_backends));
            let adapters = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()));
            adapters.iter().enumerate().map(|(i, a)| AdapterSummary::new(i, a)).collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = options;
            Vec::new()
        }
    }

    /// Creates a context with [`GpuOptions::from_env`]. Blocks; in the browser use [`GpuContext::new_async`].
    pub fn new() -> Result<Self> {
        Self::with_options(&GpuOptions::from_env())
    }

    /// [`GpuContext::new`] without blocking (the only way to create a context in the browser).
    pub async fn new_async() -> Result<Self> {
        Self::with_options_async(&GpuOptions::from_env()).await
    }

    /// The process-wide shared context, created on first use with [`GpuOptions::from_env`].
    ///
    /// Tests and the folo-core backend use this so the device is created once. In the browser it cannot be
    /// created here (that would block): it returns the context [`GpuContext::shared_async`] created, or an
    /// error before that.
    pub fn shared() -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let r = match SHARED.get() {
            Some(r) => r,
            None => {
                let _creating = SHARED_INIT.lock().unwrap_or_else(|p| p.into_inner());
                SHARED.get_or_init(|| Self::new().map_err(|e| e.reason()))
            }
        };
        #[cfg(target_arch = "wasm32")]
        let r = SHARED.get().ok_or(GpuError::Unsupported(NO_BLOCKING))?;
        r.clone().map_err(GpuError::NoAdapter)
    }

    /// [`GpuContext::shared`] without blocking: creates the shared context on first use.
    pub async fn shared_async() -> Result<Self> {
        // Natively the blocking form is the same (and serialised with `request_tensor_cores`).
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::shared()
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(r) = SHARED.get() {
                return r.clone().map_err(GpuError::NoAdapter);
            }
            let created = Self::new_async().await.map_err(|e| e.reason());
            // Two concurrent first calls may both create a device; the first one stored wins.
            SHARED.get_or_init(|| created).clone().map_err(GpuError::NoAdapter)
        }
    }

    /// Creates a context with explicit options. Blocks; in the browser use [`GpuContext::with_options_async`].
    pub fn with_options(options: &GpuOptions) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(Self::with_options_async(options))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = options;
            Err(GpuError::Unsupported(NO_BLOCKING))
        }
    }

    /// [`GpuContext::with_options`] without blocking.
    pub async fn with_options_async(options: &GpuOptions) -> Result<Self> {
        let backends = options.backends.unwrap_or_else(default_backends);
        let inst = instance(backends);
        let adapters = inst.enumerate_adapters(wgpu::Backends::all()).await;
        let (index, adapter) = if adapters.is_empty() {
            // The browser backend does not enumerate; ask for one.
            if options.adapter.is_some() {
                return Err(GpuError::NoAdapter("adapter filter given but no adapters enumerate".into()));
            }
            let a = inst
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: options.power_preference,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
            (0, a)
        } else if let Some(filter) = &options.adapter {
            let found = if let Ok(i) = filter.parse::<usize>() {
                adapters.into_iter().enumerate().nth(i)
            } else {
                let f = filter.to_ascii_lowercase();
                adapters
                    .into_iter()
                    .enumerate()
                    .filter(|(_, a)| a.get_info().name.to_ascii_lowercase().contains(&f))
                    .max_by_key(|(_, a)| adapter_score(a, options.power_preference))
            };
            found.ok_or_else(|| GpuError::NoAdapter(format!("{ENV_ADAPTER}={filter} matched no adapter")))?
        } else {
            adapters
                .into_iter()
                .enumerate()
                .max_by_key(|(i, a)| (adapter_score(a, options.power_preference), std::cmp::Reverse(*i)))
                .expect("non-empty")
        };
        Self::from_adapter(index, adapter, options).await
    }

    async fn from_adapter(index: usize, adapter: wgpu::Adapter, options: &GpuOptions) -> Result<Self> {
        let summary = AdapterSummary::new(index, &adapter);
        let limits = adapter.limits();
        let mut problems = Vec::new();
        if limits.max_storage_buffers_per_shader_stage < 4 {
            problems.push("fewer than 4 storage buffers per stage");
        }
        if limits.max_compute_workgroup_storage_size < 16 * 1024 {
            problems.push("less than 16 KiB workgroup storage");
        }
        if limits.max_compute_invocations_per_workgroup < 256 || limits.max_compute_workgroup_size_x < 256 {
            problems.push("fewer than 256 invocations per workgroup");
        }
        if !problems.is_empty() {
            return Err(GpuError::Requirements(format!("{}: {}", summary.name, problems.join(", "))));
        }
        let mut features = wgpu::Features::empty();
        if options.enable_f16 && adapter.features().contains(wgpu::Features::SHADER_F16) {
            features |= wgpu::Features::SHADER_F16;
        }
        let profile = Profiler::wanted(&adapter);
        if profile {
            features |= Profiler::features();
        }
        let mut status = String::from(TENSOR_CORES_NOT_REQUESTED);
        let want_coop = options.tensor_cores
            && match crate::kernels::coop::unavailable_reason(&adapter, features) {
                None => true,
                Some(why) => {
                    status = format!("off: {why}");
                    false
                }
            };
        // Ask for everything the adapter offers: big storage buffers and the full workgroup memory.
        let request = |features: wgpu::Features, coop: bool| {
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("folo-gpu"),
                required_features: features,
                required_limits: limits.clone(),
                experimental_features: if coop {
                    crate::kernels::coop::experimental_opt_in()
                } else {
                    wgpu::ExperimentalFeatures::disabled()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
        };
        let mut coop = false;
        let (device, queue) = if want_coop {
            match request(features | crate::kernels::coop::FEATURES, true).await {
                Ok(dq) => {
                    coop = true;
                    dq
                }
                Err(e) => {
                    // Fall back to a device without cooperative matrices rather than failing.
                    status = format!("off: the device refused cooperative matrices ({e})");
                    request(features, false).await.map_err(|e| GpuError::RequestDevice(e.to_string()))?
                }
            }
        } else {
            request(features, false).await.map_err(|e| GpuError::RequestDevice(e.to_string()))?
        };
        let errors: Arc<Mutex<Vec<String>>> = Arc::default();
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock() {
                v.push(e.to_string());
            }
        }));
        let lost: Arc<Mutex<Option<String>>> = Arc::default();
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            if let Ok(mut l) = flag.lock() {
                l.get_or_insert_with(|| format!("{reason:?}: {msg}"));
            }
        });
        let dl = device.limits();
        let caps = Capabilities {
            shader_f16: device.features().contains(wgpu::Features::SHADER_F16),
            // Set below once the self-test passes.
            coop_f16: false,
            max_buffer_size: dl.max_buffer_size,
            max_storage_buffer_binding_size: dl.max_storage_buffer_binding_size,
            max_workgroup_storage: dl.max_compute_workgroup_storage_size,
            max_workgroups_per_dim: dl.max_compute_workgroups_per_dimension,
            uniform_align: dl.min_uniform_buffer_offset_alignment,
        };
        let profiler = profile.then(|| Mutex::new(Profiler::new(&device, &queue)));
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let ctx = Self {
            inner: Arc::new(Inner {
                adapter,
                device,
                queue,
                summary,
                caps,
                pool: Arc::default(),
                kernels: Mutex::default(),
                batch: Mutex::default(),
                errors,
                lost,
                dummy,
                dispatches: AtomicU64::new(0),
                submissions: AtomicU64::new(0),
                tensor_cores: AtomicBool::new(false),
                tensor_cores_status: Mutex::new(status),
                experimental: coop,
                flush_every: AtomicU32::new(
                    std::env::var("FOLO_GPU_FLUSH_EVERY")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(MAX_DISPATCHES_PER_BATCH)
                        .clamp(1, MAX_DISPATCHES_PER_BATCH),
                ),
                profiler,
            }),
        };
        if !coop {
            return Ok(ctx);
        }
        // The kernels must agree with a CPU reference on this device before they are trusted. The test runs
        // the tensor-core path explicitly, so `caps.coop_f16` is still false here.
        let tested = crate::kernels::coop::self_test(&ctx);
        let mut inner = Arc::into_inner(ctx.inner).expect("the context is not shared during creation");
        match tested {
            Ok(()) => {
                inner.caps.coop_f16 = true;
                inner.tensor_cores = AtomicBool::new(true);
                inner.tensor_cores_status = Mutex::new("on".into());
                Ok(Self { inner: Arc::new(inner) })
            }
            Err(e) => {
                // The experimental device may be broken (or lost); replace it with a plain one.
                let adapter = inner.adapter.clone();
                drop(inner);
                let plain = GpuOptions { tensor_cores: false, ..options.clone() };
                let ctx = Box::pin(Self::from_adapter(index, adapter, &plain)).await?;
                *ctx.inner.tensor_cores_status.lock().expect("status lock") =
                    format!("off: the tensor-core self-test failed ({e})");
                Ok(ctx)
            }
        }
    }

    /// Whether the f16 GEMM and conv kernels currently run on tensor cores (see [`GpuOptions::tensor_cores`]).
    pub fn tensor_cores(&self) -> bool {
        self.inner.tensor_cores.load(Ordering::Relaxed)
    }

    /// Turns the tensor-core kernels on or off for later dispatches (A/B comparisons in one process). Turning
    /// them on has no effect unless [`Capabilities::coop_f16`] (and not after a tensor-core kernel failed to
    /// build); returns whether they are now on.
    pub fn set_tensor_cores(&self, on: bool) -> bool {
        let status = self.inner.tensor_cores_status.lock().expect("status lock");
        let on = on && self.inner.caps.coop_f16 && *status == "on";
        self.inner.tensor_cores.store(on, Ordering::Relaxed);
        on
    }

    /// Whether [`GpuContext::set_tensor_cores`]`(true)` would turn the tensor-core kernels on (they are on, or
    /// were only switched off at run time), without changing anything.
    pub fn can_enable_tensor_cores(&self) -> bool {
        self.inner.caps.coop_f16 && *self.inner.tensor_cores_status.lock().expect("status lock") == "on"
    }

    /// `"on"`, or why the tensor-core kernels are off (not requested, unsupported adapter, failed self-test, a
    /// kernel that did not build, or switched off with [`GpuContext::set_tensor_cores`]).
    pub fn tensor_cores_status(&self) -> String {
        let status = self.inner.tensor_cores_status.lock().expect("status lock").clone();
        if status == "on" && !self.tensor_cores() {
            return "off (switched off at run time)".into();
        }
        status
    }

    /// Turns tensor cores off for good after a tensor-core kernel failed to build, recording why.
    pub(crate) fn disable_tensor_cores(&self, why: String) {
        let mut status = self.inner.tensor_cores_status.lock().expect("status lock");
        self.inner.tensor_cores.store(false, Ordering::Relaxed);
        *status = format!("off: {why}");
    }

    /// Adapter description for `folo doctor`.
    pub fn summary(&self) -> &AdapterSummary {
        &self.inner.summary
    }

    /// Device capabilities.
    pub fn caps(&self) -> &Capabilities {
        &self.inner.caps
    }

    /// The wgpu adapter.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
    }

    /// The wgpu device, for work outside FOLO's kernels. `None` on a context with tensor cores: its device has
    /// wgpu's experimental cooperative matrices enabled, which arbitrary shaders could use without the bounds and
    /// uniformity guarantees FOLO's own kernels keep (see `kernels::coop::experimental_opt_in`), so it is only
    /// reachable through the `unsafe` [`GpuContext::device_unchecked`].
    pub fn device(&self) -> Option<&wgpu::Device> {
        (!self.inner.experimental).then_some(&self.inner.device)
    }

    /// The wgpu device, also on a context with tensor cores.
    ///
    /// # Safety
    ///
    /// Shaders created on the returned device must not use `enable wgpu_cooperative_matrix;` unless they meet the
    /// conditions of `kernels::coop::experimental_opt_in` (cooperative loads and stores in bounds of workgroup
    /// arrays, never storage buffers; cooperative operations in uniform control flow). Everything else wgpu
    /// offers is as safe as on any device.
    pub unsafe fn device_unchecked(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// The device for this crate's tests.
    #[cfg(test)]
    pub(crate) fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Destroys the device, as a driver reset would (for testing device-loss handling): every later wait fails
    /// with [`GpuError::DeviceLost`].
    pub fn destroy_device(&self) {
        self.inner.device.destroy();
    }

    /// Whether the device was created with wgpu's experimental cooperative matrices (even if the self-test then
    /// turned tensor cores off; a failed self-test replaces the device, so this is then `false`).
    pub fn experimental_features(&self) -> bool {
        self.inner.experimental
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// A small storage buffer bound where a kernel's optional input is absent (never written).
    pub(crate) fn dummy(&self) -> &wgpu::Buffer {
        &self.inner.dummy
    }

    /// Dispatch and submission counters (benchmarks use the difference across a pass).
    pub fn work_stats(&self) -> WorkStats {
        WorkStats {
            dispatches: self.inner.dispatches.load(Ordering::Relaxed),
            submissions: self.inner.submissions.load(Ordering::Relaxed),
        }
    }

    /// Buffer pool counters.
    pub fn pool_stats(&self) -> PoolStats {
        self.inner.pool.lock().expect("pool lock").stats()
    }

    /// Restart the pool's peak-memory counter ([`PoolStats::peak_live_bytes`]) from the current live bytes.
    pub fn reset_peak_memory(&self) {
        self.inner.pool.lock().expect("pool lock").reset_peak();
    }

    /// Drops every cached (free) buffer in the pool and lowers its cap back to the live bytes (wgpu frees the memory
    /// at its next submission or poll; [`GpuContext::release_memory`] also makes it happen now).
    pub fn trim_pool(&self) {
        self.inner.pool.lock().expect("pool lock").clear();
    }

    /// Gives the pool's cached buffers back to the driver now: [`GpuContext::trim_pool`], then submits the open
    /// batch and waits for the GPU, so wgpu frees the dropped buffers instead of holding them until the next
    /// submission. Returns the bytes the pool released.
    ///
    /// For a long-lived process between jobs (`folo serve` after a training run, or once inference has been idle):
    /// the pool otherwise keeps up to the job's high-water mark of buffers, gigabytes after a training run, which
    /// other processes on the same GPU then have to page around. Buffers still in use stay; a dropped buffer still
    /// read by encoded work is freed only once that work is done (wgpu keeps it alive until then).
    pub fn release_memory(&self) -> Result<u64> {
        let released = {
            // The batch lock first (the order `alloc` inside an encode takes them in), so nothing encodes between
            // the trim and the submission below.
            let mut b = self.inner.batch.lock().expect("batch lock");
            let released = {
                let mut pool = self.inner.pool.lock().expect("pool lock");
                let cached = pool.stats().cached_bytes;
                pool.clear();
                cached
            };
            self.submit(&mut b);
            // The uniform arena too (recreated on the next dispatch): its 64 KiB chunks are allocated mid-job, and a
            // live one keeps the driver allocator's whole memory block (up to 256 MiB) from being freed.
            b.chunks.clear();
            released
        };
        self.wait_idle()?;
        Ok(released)
    }

    fn profiler(&self) -> Option<std::sync::MutexGuard<'_, Profiler>> {
        self.inner.profiler.as_ref().map(|p| p.lock().expect("profiler lock"))
    }

    /// Whether this context times every kernel (`FOLO_GPU_PROFILE=1` on an adapter with timestamp queries inside
    /// passes; see [`crate::profile`]).
    pub fn profiling(&self) -> bool {
        self.inner.profiler.is_some()
    }

    /// Submits the open batch, waits for it, and returns the GPU time per label since the last call (largest
    /// first), resetting the totals. Empty unless [`GpuContext::profiling`].
    pub fn profile_take(&self) -> Vec<ProfileRow> {
        if !self.profiling() {
            return Vec::new();
        }
        self.flush();
        self.profiler().map(|mut p| p.take()).unwrap_or_default()
    }

    /// Number of compiled pipelines in the cache.
    pub fn pipeline_count(&self) -> usize {
        self.inner.kernels.lock().expect("kernel lock").len()
    }

    /// Allocates a storage buffer of at least `bytes` bytes from the pool. Its contents are unspecified.
    pub fn alloc(&self, bytes: u64) -> Result<GpuBuffer> {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        self.alloc_usage(bytes, usage)
    }

    fn alloc_usage(&self, bytes: u64, usage: wgpu::BufferUsages) -> Result<GpuBuffer> {
        let caps = &self.inner.caps;
        let storage = usage.contains(wgpu::BufferUsages::STORAGE);
        let limit = if storage {
            caps.max_buffer_size.min(caps.max_storage_buffer_binding_size)
        } else {
            caps.max_buffer_size
        };
        if bytes > limit {
            return Err(GpuError::Limit(format!(
                "buffer of {bytes} bytes exceeds the device limit ({limit} bytes)"
            )));
        }
        let mut class = size_class(bytes);
        if class > limit {
            // Near the limit, allocate the exact size instead of rounding past it.
            class = bytes.max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        }
        if class > limit {
            return Err(GpuError::Limit(format!(
                "buffer of {bytes} bytes exceeds the device limit ({limit} bytes)"
            )));
        }
        let buf = self.inner.pool.lock().expect("pool lock").take(&self.inner.device, class, usage);
        Ok(GpuBuffer::new(buf, class, self.inner.pool.clone()))
    }

    /// Uploads `data` into a new pooled storage buffer.
    pub fn upload(&self, data: &[u8]) -> Result<GpuBuffer> {
        let buf = self.alloc(data.len() as u64)?;
        self.write(&buf, 0, data)?;
        Ok(buf)
    }

    /// Copies `data` into a new host-visible staging buffer (mapped at creation), ready for
    /// [`GpuContext::upload_staged`]. This touches neither the command batch nor the queue, so another thread can
    /// stage the next batch of inputs while this one encodes and the GPU runs: the host-side copy leaves the
    /// critical path, and the device copy is one transfer command in stream order.
    pub fn stage(&self, data: &[u8]) -> Result<Staged> {
        let len = data.len() as u64;
        let size = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(wgpu::COPY_BUFFER_ALIGNMENT);
        let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu staging"),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let copied = buf
            .slice(..)
            .get_mapped_range_mut()
            .map(|mut view| view.slice(..data.len()).copy_from_slice(data))
            .map_err(|e| self.or_lost(GpuError::Map(e.to_string())));
        buf.unmap();
        copied?;
        Ok(Staged { buf, len })
    }

    /// A pooled storage buffer holding the bytes of `staged`, filled by a copy encoded after all work so far
    /// (no submission, no wait).
    pub fn upload_staged(&self, staged: &Staged) -> Result<GpuBuffer> {
        let out = self.alloc(staged.len.max(1))?;
        let len = staged.len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        self.copy(&staged.buf, 0, out.raw(), 0, len.min(staged.buf.size()));
        Ok(out)
    }

    /// Writes `data` at byte `offset` of `buf`, ordered after all work encoded so far.
    pub fn write(&self, buf: &GpuBuffer, offset: u64, data: &[u8]) -> Result<()> {
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Shape(format!("write offset {offset} is not 4-byte aligned")));
        }
        if offset + data.len() as u64 > buf.size() {
            return Err(GpuError::Shape(format!(
                "write of {} bytes at {offset} overruns a {}-byte buffer",
                data.len(),
                buf.size()
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        // `Queue::write_buffer` runs before the next submission's commands, so submit what is open first:
        // otherwise a pooled buffer still read by unsubmitted work would be overwritten early.
        self.flush();
        if data.len().is_multiple_of(4) {
            self.inner.queue.write_buffer(buf.raw(), offset, data);
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().next_multiple_of(4), 0);
            self.inner.queue.write_buffer(buf.raw(), offset, &padded);
        }
        Ok(())
    }

    /// Reads `len` bytes at `offset` of `buf` back to the host (flushes and waits). Not available in the
    /// browser, where [`GpuContext::read_async`] is the way to read.
    pub fn read(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(buf, offset, len, <[u8]>::to_vec)
    }

    /// [`GpuContext::read`] that hands the mapped bytes to `f` instead of copying them into a `Vec` first,
    /// so a caller converting them (to typed host storage) copies once.
    pub fn read_with<R>(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        let Some(pending) = self.begin_read(buf, offset, len)? else { return Ok(f(&[])) };
        pending.wait_blocking()?;
        pending.finish(f)
    }

    /// Reads `len` bytes at `offset` of `buf` back to the host without blocking: the copy is ordered after
    /// all work encoded so far, and the future resolves once the GPU has finished it. This is the browser's
    /// readback (WebGPU maps buffers asynchronously and the page's thread must not wait); natively it waits
    /// for the device like [`GpuContext::read`].
    pub async fn read_async(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with_async(buf, offset, len, <[u8]>::to_vec).await
    }

    /// [`GpuContext::read_async`] that hands the mapped bytes to `f` (see [`GpuContext::read_with`]).
    pub async fn read_with_async<R>(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        let Some(pending) = self.begin_read(buf, offset, len)? else { return Ok(f(&[])) };
        pending.wait().await?;
        pending.finish(f)
    }

    /// Encodes the copy of a readback into a staging buffer, submits it and requests the map. `None` for an
    /// empty read.
    pub(crate) fn begin_read(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Option<PendingRead>> {
        if len == 0 {
            return Ok(None);
        }
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Shape(format!("read offset {offset} is not 4-byte aligned")));
        }
        let len4 = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if offset + len4 > buf.size() {
            return Err(GpuError::Shape(format!(
                "read of {len} bytes at {offset} overruns a {}-byte buffer",
                buf.size()
            )));
        }
        let staging = self.alloc_usage(len4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
        {
            let mut b = self.inner.batch.lock().expect("batch lock");
            self.encoder(&mut b).copy_buffer_to_buffer(buf, offset, staging.raw(), 0, len4);
        }
        self.flush();
        let state: Arc<MapSignal> = Arc::default();
        let sink = state.clone();
        staging.raw().slice(..len4).map_async(wgpu::MapMode::Read, move |r| {
            let mut s = sink.state.lock().expect("map state lock");
            s.result = Some(r.map_err(|e| e.to_string()));
            if let Some(w) = s.waker.take() {
                w.wake();
            }
            sink.ready.notify_all();
        });
        Ok(Some(PendingRead { ctx: self.clone(), staging, len, len4, state }))
    }

    /// Copies `len` bytes between buffers, ordered with the other encoded work.
    pub fn copy(&self, src: &wgpu::Buffer, src_off: u64, dst: &wgpu::Buffer, dst_off: u64, len: u64) {
        if len == 0 {
            return;
        }
        let mut b = self.inner.batch.lock().expect("batch lock");
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin("copy"));
        let enc = self.encoder(&mut b);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q);
        }
        enc.copy_buffer_to_buffer(src, src_off, dst, dst_off, len);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.bump(&mut b);
    }

    /// Zeroes the first `bytes` bytes of `buf` (rounded up to 4), ordered with the other encoded work.
    pub fn clear(&self, buf: &wgpu::Buffer, bytes: u64) {
        let bytes = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).min(buf.size());
        if bytes == 0 {
            return;
        }
        let mut b = self.inner.batch.lock().expect("batch lock");
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin("clear"));
        let enc = self.encoder(&mut b);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q);
        }
        enc.clear_buffer(buf, 0, Some(bytes));
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.bump(&mut b);
    }

    /// Whether two handles refer to the same device.
    pub fn same_device(&self, other: &GpuContext) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Submits the open batch without waiting.
    pub fn flush(&self) {
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.submit(&mut b);
    }

    /// Submits the open batch and blocks until the GPU has finished all submitted work.
    ///
    /// Returns any validation / out-of-memory error wgpu reported since the last check. In the browser it
    /// cannot wait: it submits and reports the errors seen so far (an awaited readback,
    /// [`GpuContext::read_async`], is what waits there).
    pub fn sync(&self) -> Result<()> {
        self.flush();
        self.wait_idle()?;
        self.take_error()
    }

    /// Blocks until the device is idle (returns at once in the browser, where WebGPU's `poll` cannot wait).
    ///
    /// A lost device is [`GpuError::DeviceLost`], not a panic: wgpu treats a failed poll on a lost device as fatal
    /// and panics inside `Device::poll` (its message still reaches stderr through the panic hook), so the poll
    /// runs under `catch_unwind` and the loss is remembered, failing every later wait at once.
    fn wait_idle(&self) -> Result<()> {
        self.poll_checked(wgpu::PollType::wait_indefinitely())
    }

    /// One `Device::poll` that turns a lost device into [`GpuError::DeviceLost`]: before the poll if the loss is
    /// already known, if wgpu panics inside the poll (it treats a poll on a lost device as fatal), and after a
    /// successful poll if the device-lost callback fired during it. A loss found here is remembered.
    fn poll_checked(&self, how: wgpu::PollType) -> Result<()> {
        if let Some(why) = self.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.device.poll(how)));
        let res = match polled {
            Ok(r) => r.map(|_| ()).map_err(|e| GpuError::Poll(e.to_string())),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Device::poll panicked".into());
                let mut lost = self.inner.lost.lock().unwrap_or_else(|p| p.into_inner());
                return Err(GpuError::DeviceLost(lost.get_or_insert(msg).clone()));
            }
        };
        match self.lost_reason() {
            Some(why) => Err(GpuError::DeviceLost(why)),
            None => res,
        }
    }

    /// Why the device was lost (wgpu's reason and message), if it was. Every wait then fails with
    /// [`GpuError::DeviceLost`].
    pub fn lost_reason(&self) -> Option<String> {
        self.inner.lost.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `e`, or [`GpuError::DeviceLost`] if the device turns out to be lost (checked with a non-blocking poll): wgpu
    /// reports work on a lost device as ordinary validation or map errors.
    fn or_lost(&self, e: GpuError) -> GpuError {
        match self.poll_checked(wgpu::PollType::Poll) {
            Err(lost @ GpuError::DeviceLost(_)) => lost,
            _ => e,
        }
    }

    /// The result of a readback's buffer map, loss-aware: a recorded device loss wins over both a map failure and
    /// a map that succeeded meanwhile. This is the only loss check the browser gets, where waits cannot poll and
    /// the device-lost callback runs from the event loop.
    pub(crate) fn map_outcome(&self, r: std::result::Result<(), String>) -> Result<()> {
        if let Some(why) = self.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        r.map_err(GpuError::Map)
    }

    fn take_error(&self) -> Result<()> {
        let mut errs = self.inner.errors.lock().expect("error lock");
        if errs.is_empty() {
            return Ok(());
        }
        // The first error is the cause; once a buffer could not be created (out of memory), every later use of
        // it fails validation too, so the rest is only counted.
        let mut msg = errs[0].clone();
        if errs.len() > 1 {
            msg.push_str(&format!("\n({} more wgpu errors followed)", errs.len() - 1));
        }
        errs.clear();
        Err(GpuError::Wgpu(msg))
    }

    fn encoder<'b>(&self, b: &'b mut Batch) -> &'b mut wgpu::CommandEncoder {
        b.pass = None;
        b.encoder.get_or_insert_with(|| {
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("folo-gpu batch") })
        })
    }

    fn bump(&self, b: &mut Batch) {
        b.dispatches += 1;
        if b.dispatches >= self.inner.flush_every.load(Ordering::Relaxed) {
            self.submit(b);
        }
    }

    fn submit(&self, b: &mut Batch) {
        b.pass = None;
        let Some(mut enc) = b.encoder.take() else { return };
        let mut prof = self.profiler();
        let timed = prof.as_ref().and_then(|p| p.resolve(&mut enc));
        let used = if b.cursor == 0 { b.chunk } else { b.chunk + 1 };
        for (i, c) in b.chunks.iter().enumerate().take(used) {
            let len = if i == b.chunk { b.cursor } else { UNIFORM_CHUNK } as usize;
            if len > 0 {
                self.inner.queue.write_buffer(&c.buf, 0, &c.cpu[..len]);
            }
        }
        self.inner.queue.submit([enc.finish()]);
        self.inner.submissions.fetch_add(1, Ordering::Relaxed);
        if let (Some(p), Some(bytes)) = (prof.as_mut(), timed) {
            p.collect(&self.inner.device, bytes);
        }
        b.dispatches = 0;
        b.chunk = 0;
        b.cursor = 0;
    }

    /// Compiles (or fetches from the cache) the pipeline for `def` specialised with `spec`.
    pub(crate) fn kernel(&self, def: &'static KernelDef, spec: Spec) -> Result<Arc<Kernel>> {
        let key = KernelKey { name: def.name, spec };
        if let Some(k) = self.inner.kernels.lock().expect("kernel lock").get(&key) {
            return Ok(k.clone());
        }
        // Compile outside the lock: pipeline creation can take tens of milliseconds.
        let kernel = Arc::new(Kernel::compile(&self.inner.device, def, &key.spec)?);
        Ok(self.inner.kernels.lock().expect("kernel lock").entry(key).or_insert(kernel).clone())
    }

    /// Encodes one dispatch of `kernel` with `buffers` bound at `@binding(1..)` and `params` at `@binding(0)`.
    pub(crate) fn dispatch(
        &self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        params: &Params,
        groups: [u32; 3],
    ) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        let max = self.inner.caps.max_workgroups_per_dim;
        if groups.iter().any(|&g| g > max) {
            return Err(GpuError::Limit(format!(
                "dispatch {groups:?} exceeds {max} workgroups per dimension"
            )));
        }
        if buffers.len() != kernel.storage.len() {
            return Err(GpuError::Shape(format!(
                "kernel {} takes {} buffers, got {}",
                kernel.name,
                kernel.storage.len(),
                buffers.len()
            )));
        }
        let mut guard = self.inner.batch.lock().expect("batch lock");
        let b = &mut *guard;
        if b.cursor + UNIFORM_SLOT > UNIFORM_CHUNK {
            b.chunk += 1;
            b.cursor = 0;
        }
        if b.chunk == b.chunks.len() {
            b.chunks.push(UniformChunk {
                buf: self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("folo-gpu params"),
                    size: UNIFORM_CHUNK,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cpu: vec![0; UNIFORM_CHUNK as usize],
            });
        }
        let slot = b.cursor;
        let chunk = &mut b.chunks[b.chunk];
        chunk.cpu[slot as usize..(slot + UNIFORM_SLOT) as usize].copy_from_slice(params.bytes());
        let mut entries = Vec::with_capacity(buffers.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &chunk.buf,
                offset: slot,
                size: wgpu::BufferSize::new(UNIFORM_SLOT),
            }),
        });
        for (i, buf) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: buf.as_entire_binding() });
        }
        let bind = self.inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &entries,
        });
        b.cursor += UNIFORM_SLOT.max(self.inner.caps.uniform_align as u64);
        if b.pass.is_none() {
            let enc = self.encoder(b);
            let pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None })
                .forget_lifetime();
            b.pass = Some(pass);
        }
        let pass = b.pass.as_mut().expect("pass");
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin(kernel.name));
        if let (Some(p), Some(q)) = (&prof, q) {
            pass.write_timestamp(&p.set, q);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        if let (Some(p), Some(q)) = (&prof, q) {
            pass.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.inner.dispatches.fetch_add(1, Ordering::Relaxed);
        self.bump(b);
        Ok(())
    }

    /// A 1-D launch of `threads` invocations with `wg` threads per workgroup, folded into x/y when it exceeds
    /// the per-dimension limit. Shaders recover the linear id with `linear_id` from the prelude.
    pub(crate) fn grid_1d(&self, threads: u64, wg: u32) -> Result<[u32; 3]> {
        let groups = threads.div_ceil(wg as u64);
        let max = self.inner.caps.max_workgroups_per_dim as u64;
        if groups <= max {
            return Ok([groups as u32, 1, 1]);
        }
        let y = groups.div_ceil(max);
        if y > max {
            return Err(GpuError::Limit(format!("{threads} threads do not fit one dispatch")));
        }
        Ok([max as u32, y as u32, 1])
    }
}

/// What a readback's map callback reported, and the task to wake.
#[derive(Default)]
struct MapState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

/// [`MapState`] plus a condition variable for blocking waiters.
#[derive(Default)]
struct MapSignal {
    state: Mutex<MapState>,
    ready: Condvar,
}

/// A readback whose copy is submitted and whose staging buffer is being mapped ([`GpuContext::begin_read`]).
pub(crate) struct PendingRead {
    ctx: GpuContext,
    staging: GpuBuffer,
    len: u64,
    len4: u64,
    state: Arc<MapSignal>,
}

impl PendingRead {
    /// Waits for the map by blocking on the device (native targets).
    ///
    /// The callback can run on another thread: when several threads poll the device, the one that collected
    /// the callbacks may still be running them after this thread's poll returned. So wait on the condition
    /// variable, polling again now and then in case no other thread is.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait_blocking(&self) -> Result<()> {
        self.ctx.wait_idle()?;
        let mut s = self.state.state.lock().expect("map state lock");
        loop {
            if let Some(r) = s.result.take() {
                return self.ctx.map_outcome(r);
            }
            let (guard, timeout) = self
                .state
                .ready
                .wait_timeout(s, std::time::Duration::from_millis(10))
                .expect("map state lock");
            s = guard;
            if timeout.timed_out() && s.result.is_none() {
                drop(s);
                self.ctx.wait_idle()?;
                s = self.state.state.lock().expect("map state lock");
            }
        }
    }

    /// In the browser the map callback only runs from the page's event loop, which a blocking call never gives
    /// a turn.
    #[cfg(target_arch = "wasm32")]
    fn wait_blocking(&self) -> Result<()> {
        match self.state.state.lock().expect("map state lock").result.take() {
            Some(r) => self.ctx.map_outcome(r),
            None => Err(GpuError::Unsupported(NO_BLOCKING)),
        }
    }

    /// Waits for the map without blocking the browser. Natively the device is polled to completion first,
    /// so the future is ready at once; in the browser WebGPU resolves it from the event loop.
    pub(crate) async fn wait(&self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        self.ctx.wait_idle()?;
        let r = MapWait { state: &self.state.state }.await;
        self.ctx.map_outcome(r)
    }

    /// Hands the mapped bytes to `f`, unmaps, and reports the errors wgpu captured meanwhile.
    pub(crate) fn finish<R>(self, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        let out = self
            .staging
            .raw()
            .slice(..self.len4)
            .get_mapped_range()
            .map(|view| f(&view[..self.len as usize]))
            .map_err(|e| GpuError::Map(e.to_string()));
        let ctx = self.ctx.clone();
        drop(self);
        let out = out?;
        ctx.take_error()?;
        Ok(out)
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        // Unmap on every path (also a failed or abandoned wait, which cancels a pending map) so the staging
        // buffer goes back to the pool usable.
        self.staging.raw().unmap();
    }
}

/// Resolves when a readback's map callback has run.
struct MapWait<'a> {
    state: &'a Mutex<MapState>,
}

impl Future for MapWait<'_> {
    type Output = std::result::Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.state.lock().expect("map state lock");
        match s.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
