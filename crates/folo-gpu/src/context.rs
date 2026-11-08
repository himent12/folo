//! [`GpuContext`]: adapter selection, device + queue, buffer pool, pipeline cache and command batching.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use crate::error::{GpuError, Result};
use crate::pipeline::{Kernel, KernelDef, KernelKey, Params, Spec, UNIFORM_SLOT};
use crate::pool::{BufferPool, GpuBuffer, PoolStats, size_class};

/// Environment variable that picks the adapter: an index into [`GpuContext::enumerate`] or a case-insensitive
/// substring of the adapter name (for example `FOLO_GPU_ADAPTER=4080` or `FOLO_GPU_ADAPTER=1`).
pub const ENV_ADAPTER: &str = "FOLO_GPU_ADAPTER";
/// Environment variable that restricts the backends: a comma-separated list of `vulkan`, `dx12`, `metal`,
/// `webgpu` (for example `FOLO_GPU_BACKEND=dx12`).
pub const ENV_BACKEND: &str = "FOLO_GPU_BACKEND";
/// Environment variable that disables f16 kernels even when the adapter has `shader-f16` (`FOLO_GPU_NO_F16=1`).
pub const ENV_NO_F16: &str = "FOLO_GPU_NO_F16";

/// Dispatches encoded before the open batch is submitted on its own.
const MAX_DISPATCHES_PER_BATCH: u32 = 1024;
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
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            adapter: None,
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            enable_f16: true,
        }
    }
}

impl GpuOptions {
    /// Defaults overridden by [`ENV_ADAPTER`], [`ENV_BACKEND`] and [`ENV_NO_F16`].
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
            "[{}] {} ({}, {}) driver {} {}; shader-f16 {}; max buffer {} MiB, storage binding {} MiB, \
             workgroup storage {} KiB, {} invocations/workgroup, subgroup {}..{}",
            self.index,
            self.name,
            self.backend,
            self.device_type,
            self.driver,
            self.driver_info,
            if self.shader_f16 { "yes" } else { "no" },
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
    dummy: wgpu::Buffer,
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

impl GpuContext {
    /// Lists the adapters visible with `options.backends` (ignores the adapter filter).
    pub fn enumerate(options: &GpuOptions) -> Vec<AdapterSummary> {
        let inst = instance(options.backends.unwrap_or_else(default_backends));
        let adapters = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()));
        adapters.iter().enumerate().map(|(i, a)| AdapterSummary::new(i, a)).collect()
    }

    /// Creates a context with [`GpuOptions::from_env`].
    pub fn new() -> Result<Self> {
        Self::with_options(&GpuOptions::from_env())
    }

    /// The process-wide shared context, created on first use with [`GpuOptions::from_env`].
    ///
    /// Tests and the future folo-core backend use this so the device is created once.
    pub fn shared() -> Result<Self> {
        static SHARED: OnceLock<std::result::Result<GpuContext, String>> = OnceLock::new();
        SHARED.get_or_init(|| Self::new().map_err(|e| e.to_string())).clone().map_err(GpuError::NoAdapter)
    }

    /// Creates a context with explicit options.
    pub fn with_options(options: &GpuOptions) -> Result<Self> {
        let backends = options.backends.unwrap_or_else(default_backends);
        let inst = instance(backends);
        let adapters = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()));
        let (index, adapter) = if adapters.is_empty() {
            // The browser backend does not enumerate; ask for one.
            if options.adapter.is_some() {
                return Err(GpuError::NoAdapter("adapter filter given but no adapters enumerate".into()));
            }
            let a = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            }))
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
        Self::from_adapter(index, adapter, options)
    }

    fn from_adapter(index: usize, adapter: wgpu::Adapter, options: &GpuOptions) -> Result<Self> {
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
        // Ask for everything the adapter offers: big storage buffers and the full workgroup memory.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("folo-gpu"),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| GpuError::RequestDevice(e.to_string()))?;
        let errors: Arc<Mutex<Vec<String>>> = Arc::default();
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if let Ok(mut v) = sink.lock() {
                v.push(e.to_string());
            }
        }));
        let dl = device.limits();
        let caps = Capabilities {
            shader_f16: device.features().contains(wgpu::Features::SHADER_F16),
            max_buffer_size: dl.max_buffer_size,
            max_storage_buffer_binding_size: dl.max_storage_buffer_binding_size,
            max_workgroup_storage: dl.max_compute_workgroup_storage_size,
            max_workgroups_per_dim: dl.max_compute_workgroups_per_dimension,
            uniform_align: dl.min_uniform_buffer_offset_alignment,
        };
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(Self {
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
                dummy,
            }),
        })
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

    /// The wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// The wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// A small storage buffer bound where a kernel's optional input is absent (never written).
    pub(crate) fn dummy(&self) -> &wgpu::Buffer {
        &self.inner.dummy
    }

    /// Buffer pool counters.
    pub fn pool_stats(&self) -> PoolStats {
        self.inner.pool.lock().expect("pool lock").stats()
    }

    /// Drops every cached (free) buffer in the pool.
    pub fn trim_pool(&self) {
        self.inner.pool.lock().expect("pool lock").clear();
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

    /// Reads `len` bytes at `offset` of `buf` back to the host (flushes and waits).
    pub fn read(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
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
        let (tx, rx) = mpsc::channel();
        staging.raw().slice(..len4).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Poll(e.to_string()))?;
        rx.recv()
            .map_err(|_| GpuError::Map("map callback did not run".into()))?
            .map_err(|e| GpuError::Map(e.to_string()))?;
        let out = staging
            .raw()
            .slice(..len4)
            .get_mapped_range()
            .map(|view| view[..len as usize].to_vec())
            .map_err(|e| GpuError::Map(e.to_string()));
        // Unmap on every path so the staging buffer goes back to the pool usable.
        staging.raw().unmap();
        let out = out?;
        self.take_error()?;
        Ok(out)
    }

    /// Copies `len` bytes between buffers, ordered with the other encoded work.
    pub fn copy(&self, src: &wgpu::Buffer, src_off: u64, dst: &wgpu::Buffer, dst_off: u64, len: u64) {
        if len == 0 {
            return;
        }
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.encoder(&mut b).copy_buffer_to_buffer(src, src_off, dst, dst_off, len);
        self.bump(&mut b);
    }

    /// Zeroes the first `bytes` bytes of `buf` (rounded up to 4), ordered with the other encoded work.
    pub fn clear(&self, buf: &wgpu::Buffer, bytes: u64) {
        let bytes = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).min(buf.size());
        if bytes == 0 {
            return;
        }
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.encoder(&mut b).clear_buffer(buf, 0, Some(bytes));
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
    /// Returns any validation / out-of-memory error wgpu reported since the last check.
    pub fn sync(&self) -> Result<()> {
        self.flush();
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Poll(e.to_string()))?;
        self.take_error()
    }

    fn take_error(&self) -> Result<()> {
        let mut errs = self.inner.errors.lock().expect("error lock");
        if errs.is_empty() {
            return Ok(());
        }
        let msg = errs.join("\n");
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
        if b.dispatches >= MAX_DISPATCHES_PER_BATCH {
            self.submit(b);
        }
    }

    fn submit(&self, b: &mut Batch) {
        b.pass = None;
        let Some(enc) = b.encoder.take() else { return };
        let used = if b.cursor == 0 { b.chunk } else { b.chunk + 1 };
        for (i, c) in b.chunks.iter().enumerate().take(used) {
            let len = if i == b.chunk { b.cursor } else { UNIFORM_CHUNK } as usize;
            if len > 0 {
                self.inner.queue.write_buffer(&c.buf, 0, &c.cpu[..len]);
            }
        }
        self.inner.queue.submit([enc.finish()]);
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
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
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
