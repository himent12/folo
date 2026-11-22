//! Adapter selection and device creation: [`AdapterSummary`], [`GpuContext::enumerate`], the constructors, the
//! process-wide shared context and the tensor-core self-test at creation.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};

#[cfg(target_arch = "wasm32")]
use super::NO_BLOCKING;
#[cfg(not(target_arch = "wasm32"))]
use super::SHARED_INIT;
use super::{
    Capabilities, DEFAULT_FLUSH_INTERVAL, ENV_ADAPTER, ENV_FLUSH_EVERY, GpuContext, GpuOptions, Inner,
    MAX_DISPATCHES_PER_BATCH, SHARED, TENSOR_CORES_NOT_REQUESTED,
};
use crate::error::{GpuError, Result};
use crate::pool::BufferPool;
use crate::profile::Profiler;

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
    /// Dedicated memory in bytes: the largest device-local heap on Vulkan (on an integrated GPU, the share of system
    /// memory it may use); 0 where wgpu offers no way to ask (DX12, Metal, the browser).
    pub memory_bytes: u64,
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
            memory_bytes: device_memory(adapter),
            max_buffer_size: l.max_buffer_size,
            max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
            max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
            subgroup_size: (i.subgroup_min_size, i.subgroup_max_size),
        }
    }
}

/// [`AdapterSummary::memory_bytes`]: wgpu's own adapter info has no memory size, so it is read from the Vulkan
/// physical device behind the adapter (`vkGetPhysicalDeviceMemoryProperties`).
fn device_memory(adapter: &wgpu::Adapter) -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    // SAFETY: the guard is dropped at the end of this block and only the physical device's memory properties are read
    // through it: nothing is created, destroyed or used by the GPU.
    if let Some(a) = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() } {
        // SAFETY: `raw_physical_device` belongs to `raw_instance`, which outlives the call (the adapter holds it).
        let props = unsafe {
            a.shared_instance().raw_instance().get_physical_device_memory_properties(a.raw_physical_device())
        };
        /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
        const DEVICE_LOCAL: u32 = 0x1;
        return props
            .memory_heaps_as_slice()
            .iter()
            .filter(|h| h.flags.as_raw() & DEVICE_LOCAL != 0)
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
    }
    let _ = adapter;
    0
}

impl std::fmt::Display for AdapterSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The memory size where the backend tells it (Vulkan); nothing said where it cannot (0).
        let memory = match self.memory_bytes {
            0 => String::new(),
            b => format!("memory {:.1} GiB; ", b as f64 / (1u64 << 30) as f64),
        };
        write!(
            f,
            "[{}] {} ({}, {}) driver {} {}; {memory}shader-f16 {}; tensor cores {}; max buffer {} MiB, storage binding \
             {} MiB, workgroup storage {} KiB, {} invocations/workgroup, subgroup {}..{}",
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
        // Native only: wgpu 30.0.1's WebGPU backend omits `SUBGROUP` and BatchNorm also needs WGSL `subgroup_id`.
        #[cfg(not(target_arch = "wasm32"))]
        if adapter.features().contains(wgpu::Features::SUBGROUP) {
            features |= wgpu::Features::SUBGROUP;
        }
        let cache_file = crate::pipeline_cache::wanted(&adapter, options.pipeline_cache_dir.as_deref());
        if cache_file.is_some() {
            features |= wgpu::Features::PIPELINE_CACHE;
        }
        let profile = options.profile && Profiler::supported(&adapter);
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
            subgroups: device.features().contains(wgpu::Features::SUBGROUP),
            subgroup_size: summary.subgroup_size,
            max_buffer_size: dl.max_buffer_size,
            max_storage_buffer_binding_size: dl.max_storage_buffer_binding_size,
            max_workgroup_storage: dl.max_compute_workgroup_storage_size,
            max_workgroups_per_dim: dl.max_compute_workgroups_per_dimension,
            uniform_align: dl.min_uniform_buffer_offset_alignment,
            max_storage_buffers: dl.max_storage_buffers_per_shader_stage,
        };
        let profiler = profile.then(|| Mutex::new(Profiler::new(&device, &queue)));
        let pipeline_cache = cache_file
            .filter(|_| device.features().contains(wgpu::Features::PIPELINE_CACHE))
            .map(|path| crate::pipeline_cache::DiskCache::open(&device, &adapter.get_info(), path));
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pool = BufferPool::default();
        let retired = pool.retired_flag();
        let ctx = Self {
            inner: Arc::new(Inner {
                adapter,
                device,
                queue,
                summary,
                caps,
                pool: Arc::new(Mutex::new(pool)),
                retired,
                kernels: Mutex::default(),
                batch: Mutex::default(),
                errors,
                lost,
                dummy,
                dispatches: AtomicU64::new(0),
                submissions: AtomicU64::new(0),
                bind_groups: AtomicU64::new(0),
                spin_waits: AtomicU64::new(0),
                blocked_waits: AtomicU64::new(0),
                tensor_cores: AtomicBool::new(false),
                tensor_cores_status: Mutex::new(status),
                experimental: coop,
                tracing: AtomicBool::new(false),
                trace: Mutex::default(),
                staging: Arc::default(),
                flush_every: AtomicU32::new(
                    std::env::var(ENV_FLUSH_EVERY)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(DEFAULT_FLUSH_INTERVAL)
                        .clamp(1, MAX_DISPATCHES_PER_BATCH),
                ),
                wait_spin_ns: AtomicU64::new(
                    std::env::var(super::ENV_WAIT_SPIN_US)
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(|us| us.saturating_mul(1000))
                        .unwrap_or(super::DEFAULT_WAIT_SPIN.as_nanos() as u64),
                ),
                profiler,
                pipeline_cache,
                cache_dir: options.pipeline_cache_dir.clone(),
                precompile: Default::default(),
                plans: Mutex::default(),
                plan_epoch: AtomicU64::new(0),
                plans_on: AtomicBool::new(
                    !std::env::var(super::ENV_NO_PLAN).is_ok_and(|v| v != "0" && !v.is_empty()),
                ),
            }),
        };
        if std::env::var(super::ENV_NO_BIND_CACHE).is_ok_and(|v| v != "0" && !v.is_empty()) {
            ctx.set_bind_cache(false);
        }
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
}

#[cfg(test)]
mod tests {
    use super::AdapterSummary;

    /// `folo doctor` prints each adapter's memory size when the backend reports one, and nothing about it otherwise.
    #[test]
    fn an_adapter_line_names_its_memory_when_known() {
        let mut a = AdapterSummary {
            index: 0,
            name: "NVIDIA GeForce RTX 4080 SUPER".into(),
            vendor: 0x10de,
            device: 0x2702,
            device_type: "DiscreteGpu".into(),
            backend: "Vulkan".into(),
            driver: "NVIDIA".into(),
            driver_info: "591.44".into(),
            shader_f16: true,
            tensor_cores: true,
            memory_bytes: 16_869_490_688,
            max_buffer_size: 1 << 32,
            max_storage_buffer_binding_size: 1 << 31,
            max_compute_workgroup_storage_size: 49_152,
            max_compute_invocations_per_workgroup: 1024,
            subgroup_size: (32, 32),
        };
        assert!(a.to_string().contains("driver NVIDIA 591.44; memory 15.7 GiB; shader-f16 yes"), "{a}");
        a.memory_bytes = 0;
        assert!(!a.to_string().contains("memory"), "{a}");
    }
}
