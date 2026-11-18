//! System (docs/API.md, *System*): health, the host's CPU, memory and GPUs, what `device: auto` can do, and the
//! optional API features a server announces.

use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{OnnxSwitches, QueueRequest};

/// API version string reported in [`Health::api`].
pub const API_VERSION: &str = "v1";

/// `GET /api/health`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Health {
    /// Always `"ok"` when the server answers.
    pub status: String,
    /// Server (crate) version, e.g. `0.1.0`.
    pub version: String,
    /// API version, currently `v1`.
    #[serde(default = "default_api")]
    pub api: String,
    /// True when the server runs `--mock`.
    pub mock: bool,
    /// The same optional features `GET /api/system` lists, without its hardware probe; empty from servers that
    /// predate it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
}

fn default_api() -> String {
    API_VERSION.to_string()
}

/// `GET /api/system`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SystemInfo {
    /// Host CPU.
    pub cpu: CpuInfo,
    /// Host memory.
    pub memory: MemoryInfo,
    /// GPUs usable by FOLO (empty until GPU detection lands).
    pub gpus: Vec<GpuInfo>,
}

/// What `GET /api/system` sends: [`SystemInfo`] plus the server's data dir. (A separate type, so code that builds
/// a [`SystemInfo`] is unaffected.)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SystemResponse {
    /// CPU, memory and GPUs.
    #[serde(flatten)]
    pub system: SystemInfo,
    /// The server's data dir (`--data-dir`) with `/` separators, where its runs, models and exports live; absent
    /// in mock mode and from servers older than this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Optional API features this server implements (for example [`FEATURE_EXPORT_ONNX_SWITCHES`]), so a client
    /// can tell before it relies on one; empty from servers older than this field.
    #[serde(default)]
    pub features: Vec<String>,
    /// What runs with `device: auto` can use on this server (f16 inference, tensor cores, mixed-precision
    /// training), so a client can disable a switch, with the reason, before it submits a run; absent from servers
    /// older than this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Capabilities>,
    /// FOLO's GPU buffer pool on this server, when the GPU is in use (absent before the server touched it, on a
    /// CPU-only server, in mock mode and from servers older than this field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_pool: Option<GpuPool>,
}

/// [`SystemResponse::gpu_pool`]: the GPU memory FOLO's buffer pool holds. A finished run gives its cached buffers
/// back to the driver, and so does inference once it has been idle for a while, so with nothing running
/// `cached_bytes` drops to 0.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GpuPool {
    /// The device, e.g. `gpu:0`.
    #[serde(default)]
    pub device: String,
    /// Bytes in buffers in use (loaded models, a run's tensors).
    #[serde(default)]
    pub live_bytes: u64,
    /// Bytes in free buffers kept for reuse.
    #[serde(default)]
    pub cached_bytes: u64,
    /// The largest `live_bytes` since the last training run or benchmark probe started (since start-up before
    /// one).
    #[serde(default)]
    pub peak_live_bytes: u64,
}

/// [`SystemResponse::capabilities`]: what the device `device: auto` resolves to can do. Every field defaults, so a
/// client parses an object from a newer server that adds fields, and one from an older server that lacks some.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Capabilities {
    /// The device `device: auto` resolves to: `gpu:0` or `cpu`.
    #[serde(default)]
    pub device: String,
    /// The device computes in f16, so `precision: f16` inference runs in half precision (else it falls back to
    /// f32: the CPU, an adapter without `shader-f16`, or `FOLO_GPU_NO_F16`).
    #[serde(default)]
    pub shader_f16: bool,
    /// The tensor-core (cooperative-matrix) f16 kernels.
    #[serde(default)]
    pub tensor_cores: TensorCoreSupport,
    /// Mixed-precision training (`TrainConfig.amp`).
    #[serde(default)]
    pub amp: AmpSupport,
}

/// [`Capabilities::tensor_cores`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TensorCoreSupport {
    /// The adapter reports the 16×16×16 f16 → f32 cooperative matrices the tensor-core kernels need.
    #[serde(default)]
    pub available: bool,
    /// The kernels are on for this server (`folo serve --tensor-cores` / `FOLO_GPU_TENSOR_CORES=1`, and the
    /// start-up self-test passed).
    #[serde(default)]
    pub enabled: bool,
    /// Why they are off, as `folo doctor` phrases it (`off (not requested: ...)`, `off: ...`); `null` when
    /// `enabled`.
    #[serde(default)]
    pub reason: Option<String>,
}

/// [`Capabilities::amp`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AmpSupport {
    /// A run with `"amp": true` and `device: auto` can start (it fails with `not_supported` otherwise).
    #[serde(default)]
    pub available: bool,
    /// The trainer's `not_supported` message when not `available`; `null` when it is.
    #[serde(default)]
    pub reason: Option<String>,
}

/// [`SystemResponse::features`]: `POST /api/models/{id}/export` honours [`OnnxSwitches`] (older servers accept
/// and ignore them).
pub const FEATURE_EXPORT_ONNX_SWITCHES: &str = "export_onnx_switches";

/// [`SystemResponse::features`]: `POST /api/runs/{id}/queue` exists ([`QueueRequest`]), runs carry
/// `queue_position` and streams send `queue` events.
pub const FEATURE_QUEUE_CONTROL: &str = "queue_control";

/// [`SystemResponse::features`]: `GET /api/models/{id}/weights?format=onnx` exists (older servers answer `400
/// invalid_query`).
pub const FEATURE_WEIGHTS_ONNX: &str = "weights_onnx";

/// [`SystemResponse::features`]: `GET /api/runs/{id}/cli` exists ([`CliCommand`]) and `RunEstimate.cli` is set.
pub const FEATURE_RUN_CLI: &str = "run_cli";

/// [`SystemResponse::features`]: the Studio this server serves includes the in-browser runtime (folo-wasm,
/// docs/WASM.md), so its Playground can run models with WebGPU or the wasm CPU.
pub const FEATURE_WEB_WASM: &str = "web_wasm";

/// `GET /api/runs/{id}/cli`: the `folo train` command that trains a run's config on this machine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CliCommand {
    /// The command line, arguments quoted for a POSIX shell or PowerShell (single quotes where needed).
    pub cli: String,
    /// The same command as separate arguments (`argv[0]` is `folo`), for clients that spawn it without a shell.
    pub argv: Vec<String>,
}

/// CPU name, logical thread count and current utilisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CpuInfo {
    /// Marketing name, e.g. `AMD Ryzen 9 7950X 16-Core Processor`.
    pub name: String,
    /// Logical threads.
    pub threads: u32,
    /// Utilisation in `0..=1` averaged over all threads since the previous call.
    pub util: f32,
}

/// Host memory in bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryInfo {
    /// Bytes in use.
    pub used_bytes: u64,
    /// Total physical bytes.
    pub total_bytes: u64,
}

/// One GPU adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GpuInfo {
    /// Index used in device strings such as `gpu:0`.
    pub ordinal: u32,
    /// Adapter name.
    pub name: String,
    /// Graphics API the adapter is driven through.
    pub backend: GpuBackend,
    /// Dedicated memory in bytes.
    pub memory_bytes: u64,
    /// Utilisation in `0..=1`, or `null` when the platform does not report it.
    pub util: Option<f32>,
}

/// GPU graphics API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    /// Vulkan.
    Vulkan,
    /// Direct3D 12.
    Dx12,
    /// Metal.
    Metal,
}
