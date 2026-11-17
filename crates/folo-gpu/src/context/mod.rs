//! [`GpuContext`]: adapter selection, device + queue, buffer pool, pipeline cache and command batching.
//!
//! - `mod.rs`: [`GpuContext`], [`GpuOptions`], [`Capabilities`], the `FOLO_GPU_*` environment variables and the
//!   shared context.
//! - `device.rs`: adapter selection and device creation ([`AdapterSummary`], [`GpuContext::enumerate`]).
//! - `memory.rs`: allocation from the buffer pool and uploads.
//! - `batch.rs`: the open command batch: `GpuContext::kernel` (compile or fetch a pipeline) and
//!   `GpuContext::dispatch`, which encode into it, and its submission.
//! - `binds.rs`: the bind-group cache.
//! - `readback.rs`: reading buffers back to the host; `wait.rs`: [`GpuContext::sync`] and device-loss handling.
//! - `plan.rs`: frozen dispatch lists and forward plans ([`Frozen`]).
//! - `precompile.rs`: [`GpuContext::precompile`], compiling a recorded kernel list on worker threads.
//! - `trace.rs`: recording a pass's operations ([`TracedOp`]) and the per-kernel timestamps.

mod batch;
pub(crate) mod binds;
mod device;
mod guard;
mod memory;
mod plan;
mod precompile;
mod readback;
mod trace;
mod wait;

pub(crate) use guard::SKIP_FN;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(doc)]
use crate::error::GpuError;
use crate::pipeline::{Kernel, KernelKey, Params};
use crate::pool::BufferPool;
use crate::profile::Profiler;

pub use device::AdapterSummary;
pub use plan::{Frozen, MAX_PLAN_BYTES, MAX_PLAN_BYTES_EACH, MAX_PLANS};
pub(crate) use plan::{Plan, PlanOutput};

/// A stable id of a buffer for the bind-group cache (see `context/binds.rs`).
pub(crate) fn buffer_id(buf: &wgpu::Buffer) -> u64 {
    binds::id_of(buf)
}

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

/// Environment variable that turns the bind-group cache off in contexts created afterwards
/// (`FOLO_GPU_NO_BIND_CACHE=1`, for A/B runs; see [`GpuContext::set_bind_cache`]).
pub const ENV_NO_BIND_CACHE: &str = "FOLO_GPU_NO_BIND_CACHE";
/// Environment variable that turns forward plans off in contexts created afterwards (`FOLO_GPU_NO_PLAN=1`, for A/B
/// runs; see [`GpuContext::set_forward_plans`]).
pub const ENV_NO_PLAN: &str = "FOLO_GPU_NO_PLAN";
/// Environment variable that overrides the flush interval ([`GpuContext::set_flush_interval`]) of contexts created
/// afterwards, for experiments (`FOLO_GPU_FLUSH_EVERY=1024`).
pub const ENV_FLUSH_EVERY: &str = "FOLO_GPU_FLUSH_EVERY";
/// Dispatches encoded before the open batch is submitted on its own, by default. Small batches keep the GPU busy:
/// it starts on a batch while the host encodes the next, instead of waiting for one long submission. 64 made the
/// folo-n training step about 20% faster than 1024 (docs/GPU.md, Command batching); 16 is slower again, because
/// every submission costs host time.
pub const DEFAULT_FLUSH_INTERVAL: u32 = 64;
/// Largest flush interval.
pub const MAX_DISPATCHES_PER_BATCH: u32 = 1024;
/// Environment variable that overrides the wait spin ([`GpuContext::set_wait_spin`]) of contexts created
/// afterwards, in microseconds (`FOLO_GPU_WAIT_SPIN_US=0` blocks at once).
pub const ENV_WAIT_SPIN_US: &str = "FOLO_GPU_WAIT_SPIN_US";
/// How long a wait polls the device before it blocks, by default.
pub const DEFAULT_WAIT_SPIN: std::time::Duration = std::time::Duration::from_millis(2);
/// Dispatches before the first submission after the device was idle (or the flush interval, when that is smaller).
/// The limit then doubles with every submission up to the flush interval, so the GPU starts on the head of a pass
/// while the host still encodes the rest (a synchronised batch-1 forward otherwise spends a third of its latency
/// encoding and submitting before the GPU has anything to do).
const FIRST_DISPATCHES_PER_BATCH: u32 = 16;
/// Idle staging buffers [`GpuContext::upload_with`] keeps mapped for reuse.
const STAGING_KEEP: usize = 4;
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
    /// Time every kernel ([`crate::profile`]), when the adapter has timestamp queries inside passes.
    pub profile: bool,
    /// Folder of the persistent pipeline cache ([`crate::pipeline_cache`]); `None` keeps it off. Defaults to
    /// [`crate::set_pipeline_cache_dir`]'s folder.
    pub pipeline_cache_dir: Option<std::path::PathBuf>,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            adapter: None,
            backends: None,
            power_preference: wgpu::PowerPreference::HighPerformance,
            enable_f16: true,
            tensor_cores: false,
            profile: false,
            pipeline_cache_dir: crate::pipeline_cache::pipeline_cache_dir(),
        }
    }
}

impl GpuOptions {
    /// Defaults overridden by [`ENV_ADAPTER`], [`ENV_BACKEND`], [`ENV_NO_F16`], [`ENV_TENSOR_CORES`] (or
    /// [`request_tensor_cores`]) and [`crate::profile::ENV_PROFILE`].
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
        o.profile = std::env::var(crate::profile::ENV_PROFILE).is_ok_and(|v| v != "0" && !v.is_empty());
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
    /// Storage buffers one kernel can bind.
    pub max_storage_buffers: u32,
}

struct UniformChunk {
    buf: wgpu::Buffer,
    cpu: Vec<u8>,
}

struct Batch {
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    dispatches: u32,
    /// Dispatches at which the open batch is submitted (see [`FIRST_DISPATCHES_PER_BATCH`]).
    limit: u32,
    chunks: Vec<UniformChunk>,
    /// Index of the chunk being filled and the byte cursor in it.
    chunk: usize,
    cursor: u64,
    /// Bind groups of earlier dispatches, for reuse.
    binds: binds::BindCache,
}

impl Default for Batch {
    fn default() -> Self {
        Self {
            encoder: None,
            pass: None,
            dispatches: 0,
            limit: FIRST_DISPATCHES_PER_BATCH,
            chunks: Vec::new(),
            chunk: 0,
            cursor: 0,
            binds: binds::BindCache::default(),
        }
    }
}

struct Inner {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    summary: AdapterSummary,
    caps: Capabilities,
    pool: Arc<Mutex<BufferPool>>,
    /// The pool's [`BufferPool::retired_flag`]: set when the pool let a buffer go that cached bind groups may hold.
    retired: Arc<AtomicBool>,
    kernels: Mutex<HashMap<KernelKey, Arc<Kernel>>>,
    batch: Mutex<Batch>,
    errors: Arc<Mutex<Vec<String>>>,
    /// Set once the device is lost (by wgpu's device-lost callback, or by a poll that failed fatally).
    lost: Arc<Mutex<Option<String>>>,
    dummy: wgpu::Buffer,
    dispatches: AtomicU64,
    submissions: AtomicU64,
    bind_groups: AtomicU64,
    spin_waits: AtomicU64,
    blocked_waits: AtomicU64,
    /// Tensor-core kernels in use (only ever true when `caps.coop_f16`); cleared by
    /// [`GpuContext::set_tensor_cores`] or when a tensor-core kernel fails to build.
    tensor_cores: AtomicBool,
    /// `"on"`, or why tensor cores are off.
    tensor_cores_status: Mutex<String>,
    /// The device was created with wgpu's experimental cooperative matrices (see [`GpuContext::device`]).
    experimental: bool,
    /// Set while [`GpuContext::trace_begin`] records dispatches into `trace`.
    tracing: AtomicBool,
    trace: Mutex<Vec<TracedOp>>,
    /// Mapped, idle staging buffers of [`GpuContext::upload_with`], ready for the next upload.
    staging: Arc<Mutex<StagingFree>>,
    /// Dispatches after which the open batch is submitted on its own ([`GpuContext::set_flush_interval`]).
    flush_every: AtomicU32,
    /// How long a wait polls the device before it blocks, in nanoseconds ([`GpuContext::set_wait_spin`]).
    wait_spin_ns: AtomicU64,
    /// Per-kernel timestamp queries (`FOLO_GPU_PROFILE=1`, see [`crate::profile`]).
    profiler: Option<Mutex<Profiler>>,
    /// Compiled pipelines kept on disk between processes ([`crate::pipeline_cache`]).
    pipeline_cache: Option<Arc<crate::pipeline_cache::DiskCache>>,
    /// The pipeline cache folder, also where kernel lists live ([`GpuContext::precompile`]).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    cache_dir: Option<std::path::PathBuf>,
    /// Kernels compiled ahead on worker threads and the kernel lists being recorded.
    precompile: precompile::Precompile,
    /// Recorded forward plans ([`crate::ForwardPlans`]).
    plans: Mutex<plan::PlanStore>,
    /// Bumped by changes that retire every plan (see `context/plan.rs`).
    plan_epoch: AtomicU64,
    /// Forward plans are on ([`GpuContext::set_forward_plans`]).
    plans_on: AtomicBool,
}

/// The idle staging buffers of [`GpuContext::upload_with`].
#[derive(Default)]
struct StagingFree {
    bufs: Vec<wgpu::Buffer>,
    /// Bumped by [`GpuContext::release_memory`]: a buffer whose re-map was requested before the bump is dropped
    /// when the map completes instead of joining `bufs`.
    generation: u64,
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

/// One operation recorded by [`GpuContext::trace_begin`]: a kernel dispatch or a buffer copy, with everything
/// needed to encode it again ([`GpuContext::replay`]). Profiling tools replay each operation of a forward pass
/// on its own to see where the GPU time goes.
#[derive(Clone)]
pub struct TracedOp {
    /// Kernel name, or `"copy"` for a buffer copy.
    pub name: &'static str,
    /// The kernel's specialisation constants (empty for a copy).
    pub spec: String,
    /// Workgroups dispatched (`[bytes / 4, 1, 1]` for a copy).
    pub groups: [u32; 3],
    kind: TracedKind,
}

#[derive(Clone)]
enum TracedKind {
    Dispatch { kernel: Kernel, buffers: Vec<wgpu::Buffer>, params: Box<Params> },
    Copy { src: wgpu::Buffer, src_off: u64, dst: wgpu::Buffer, dst_off: u64, len: u64 },
    Clear { buf: wgpu::Buffer, bytes: u64 },
}

/// Work counters since the context was created (see [`GpuContext::work_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkStats {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// Command buffers submitted to the queue.
    pub submissions: u64,
    /// Bind groups created; the other dispatches reused a cached one (see [`GpuContext::bind_cache_len`]).
    pub bind_groups: u64,
    /// Waits for the GPU ([`GpuContext::sync`], readbacks) that ended while polling, within the wait spin
    /// ([`GpuContext::set_wait_spin`]).
    pub spin_waits: u64,
    /// Waits that blocked in the driver: the wait spin is zero, or the GPU was still busy when it ran out.
    pub blocked_waits: u64,
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

/// The context behind [`GpuContext::shared`] (set by the first `shared` / `shared_async` call).
static SHARED: OnceLock<std::result::Result<GpuContext, String>> = OnceLock::new();

/// The error the blocking entry points return in the browser, where nothing may block the page's thread.
#[cfg(target_arch = "wasm32")]
const NO_BLOCKING: &str = "blocking GPU calls are not available in the browser; use the async variants \
     (GpuContext::shared_async, folo_gpu::init_async, folo_gpu::download_async)";

impl GpuContext {
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
        // Plans are keyed by the switch, so they need not be retired.
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
        self.bump_plan_epoch();
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

    /// Submit the open batch on its own once it holds `dispatches` dispatches (clamped to
    /// `1..=`[`MAX_DISPATCHES_PER_BATCH`]; the default is [`DEFAULT_FLUSH_INTERVAL`], or [`ENV_FLUSH_EVERY`]).
    pub fn set_flush_interval(&self, dispatches: u32) {
        self.inner.flush_every.store(dispatches.clamp(1, MAX_DISPATCHES_PER_BATCH), Ordering::Relaxed);
    }

    /// The current flush interval ([`GpuContext::set_flush_interval`]).
    pub fn flush_interval(&self) -> u32 {
        self.inner.flush_every.load(Ordering::Relaxed)
    }

    /// How long a wait for the GPU ([`GpuContext::sync`], a readback) polls the device before it blocks in the
    /// driver (the default is [`DEFAULT_WAIT_SPIN`], or [`ENV_WAIT_SPIN_US`]); zero blocks at once.
    pub fn set_wait_spin(&self, spin: std::time::Duration) {
        self.inner.wait_spin_ns.store(spin.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    }

    /// The current wait spin ([`GpuContext::set_wait_spin`]).
    pub fn wait_spin(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.inner.wait_spin_ns.load(Ordering::Relaxed))
    }

    /// Dispatch, submission and wait counters (benchmarks use the difference across a pass).
    pub fn work_stats(&self) -> WorkStats {
        WorkStats {
            dispatches: self.inner.dispatches.load(Ordering::Relaxed),
            submissions: self.inner.submissions.load(Ordering::Relaxed),
            bind_groups: self.inner.bind_groups.load(Ordering::Relaxed),
            spin_waits: self.inner.spin_waits.load(Ordering::Relaxed),
            blocked_waits: self.inner.blocked_waits.load(Ordering::Relaxed),
        }
    }

    /// Whether two handles refer to the same device.
    pub fn same_device(&self, other: &GpuContext) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
