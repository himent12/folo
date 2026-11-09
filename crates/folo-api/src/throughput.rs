//! Measured speed and time estimates (docs/API.md, *Throughput and estimates*): `GET /api/system/throughput`,
//! `POST /api/system/benchmark` and `POST /api/runs/estimate`.

use serde::{Deserialize, Serialize};

use crate::Arch;

/// [`SystemResponse::features`](crate::SystemResponse::features): `GET /api/system/throughput` exists.
pub const FEATURE_THROUGHPUT: &str = "throughput";
/// [`SystemResponse::features`](crate::SystemResponse::features): `POST /api/runs/estimate` exists.
pub const FEATURE_RUN_ESTIMATE: &str = "run_estimate";
/// [`SystemResponse::features`](crate::SystemResponse::features): `POST /api/system/benchmark` exists.
pub const FEATURE_BENCHMARK: &str = "benchmark";

/// Where a number comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum ThroughputSource {
    /// Measured on this server: training runs, `/api/predict` calls, `POST /api/system/benchmark` probes.
    Measured,
    /// docs/BENCHMARKS.md's reference numbers (another machine, or this one at another time), possibly scaled to
    /// the size asked for; `note` says which.
    #[default]
    Reference,
}

/// How far to trust an estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Measured with these settings at least 3 times.
    High,
    /// Measured, but fewer than 3 times or with other settings (scaled by the reference shape).
    Medium,
    /// Reference numbers only, or a measurement far from these settings.
    #[default]
    Low,
}

/// Inference precision of a throughput row (`folo serve --precision`, after the f32 fallback of devices without
/// f16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum InferPrecision {
    /// f32.
    #[default]
    F32,
    /// f16 weights and activations, f32 accumulation.
    F16,
}

/// A rate with its spread: the median of the recent samples and their 10th / 90th percentiles (min / max with
/// fewer than 10). Reference rates carry the range the reference is expected to hold in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Rate {
    /// Images per second.
    pub imgs_per_s: f64,
    /// Low end of the range, images per second.
    pub low: f64,
    /// High end of the range, images per second.
    pub high: f64,
    /// Samples behind it (epochs, probes); 0 for a reference.
    pub samples: u64,
}

/// Training speed for one (device, arch, imgsz, batch, amp).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainThroughput {
    /// The device of this server the row is for: `gpu:<n>` or `cpu`.
    pub device: String,
    /// The adapter or CPU the numbers were measured on (for a reference: the machine of docs/BENCHMARKS.md).
    pub measured_on: String,
    /// Architecture.
    pub arch: Arch,
    /// Training image size.
    pub imgsz: u32,
    /// Training batch.
    pub batch: u32,
    /// Mixed precision.
    pub amp: bool,
    /// The headline training rate: the mosaic epochs' median when measured, else the close-mosaic epochs'.
    pub imgs_per_s: f64,
    /// Steady-state training rate in epochs with mosaic (the first epoch of a session and validation excluded).
    pub mosaic: Option<Rate>,
    /// The same in epochs without mosaic (the close-mosaic phase, or runs with `mosaic: 0`).
    pub close_mosaic: Option<Rate>,
    /// Validation pass rate (validation images per second of an epoch's validation).
    pub val: Option<Rate>,
    /// Extra seconds of a session's first epoch over a steady one (pipeline compilation, allocations).
    pub startup_s: Option<f64>,
    /// Peak GPU memory in bytes (GPU rows only; the buffer pool's peak, like torch's `max_memory_allocated`).
    pub peak_vram_bytes: Option<u64>,
    /// Measured or reference.
    pub source: ThroughputSource,
    /// Samples behind the row (epochs and probes); 0 for a reference.
    pub samples: u64,
    /// Of those, `POST /api/system/benchmark` probes.
    #[serde(default)]
    pub probes: u64,
    /// RFC 3339 time of the last sample; `null` for a reference.
    pub updated: Option<String>,
    /// Where a reference comes from (and how it was scaled); `null` for measured rows.
    pub note: Option<String>,
}

/// Inference speed (`POST /api/predict`, batch 1) for one (device, arch, imgsz, precision, tensor cores).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InferThroughput {
    /// The device of this server the row is for: `gpu:<n>` or `cpu`.
    pub device: String,
    /// The adapter or CPU the numbers were measured on.
    pub measured_on: String,
    /// Architecture.
    pub arch: Arch,
    /// The model's image size.
    pub imgsz: u32,
    /// Always 1 (`/api/predict` predicts one image).
    pub batch: u32,
    /// Precision it ran in.
    pub precision: InferPrecision,
    /// The tensor-core kernels were on.
    pub tensor_cores: bool,
    /// Median end-to-end latency (`timing.total_ms`: image decode, letterbox, forward, decode, NMS).
    pub latency_ms: f64,
    /// Low end of the range (10th percentile, or min).
    pub latency_ms_low: f64,
    /// High end of the range (90th percentile, or max).
    pub latency_ms_high: f64,
    /// Median forward pass alone (`timing.inference_ms`), when known.
    pub inference_ms: Option<f64>,
    /// `1000 / latency_ms`.
    pub imgs_per_s: f64,
    /// Measured or reference.
    pub source: ThroughputSource,
    /// Predictions behind the row; 0 for a reference.
    pub samples: u64,
    /// RFC 3339 time of the last sample; `null` for a reference.
    pub updated: Option<String>,
    /// Where a reference comes from; `null` for measured rows.
    pub note: Option<String>,
}

/// `GET /api/system/throughput`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ThroughputTable {
    /// What `device: auto` resolves to (`gpu:0` or `cpu`).
    pub device: String,
    /// Its adapter or CPU name.
    pub device_name: String,
    /// Measured rows first (newest first), then the reference rows no measured row covers.
    pub training: Vec<TrainThroughput>,
    /// Measured rows first (newest first), then the reference rows no measured row covers.
    pub inference: Vec<InferThroughput>,
}

/// One phase of an estimated run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    /// Epochs with mosaic.
    Mosaic,
    /// The last epochs, without mosaic (`augment.close_mosaic`; every epoch when `mosaic` is 0).
    CloseMosaic,
}

/// [`RunEstimate::phases`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EstimatePhase {
    /// Which phase.
    pub phase: PhaseKind,
    /// Epochs in it.
    pub epochs: u32,
    /// Training rate used.
    pub train_imgs_per_s: f64,
    /// Seconds of training per epoch.
    pub train_s: f64,
    /// Seconds of validation per epoch.
    pub val_s: f64,
    /// `train_s + val_s`.
    pub epoch_s: f64,
}

/// What an estimate is based on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EstimateBasis {
    /// Measured or reference.
    pub source: ThroughputSource,
    /// The row used: its architecture.
    pub arch: Arch,
    /// Its image size.
    pub imgsz: u32,
    /// Its batch.
    pub batch: u32,
    /// Its precision.
    pub amp: bool,
    /// Where it was measured.
    pub measured_on: String,
    /// Samples behind it.
    pub samples: u64,
    /// Its last sample, `null` for a reference.
    pub updated: Option<String>,
    /// The row has exactly the run's arch, imgsz, batch and amp (else it was carried over, see `note`).
    pub exact: bool,
    /// How the numbers were derived, in a sentence.
    pub note: String,
}

/// `POST /api/runs/estimate`: how long a run would take.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunEstimate {
    /// The device the run would train on (`gpu:<n>` or `cpu`).
    pub device: String,
    /// Its name.
    pub device_name: String,
    /// Training images (one epoch).
    pub train_images: u64,
    /// Images of each validation pass (the val split, or the training split when there is none).
    pub val_images: u64,
    /// Batches per epoch, `ceil(train_images / batch)`.
    pub batches_per_epoch: u64,
    /// Epochs.
    pub epochs: u32,
    /// Mosaic epochs, then close-mosaic epochs (phases with no epochs are left out).
    pub phases: Vec<EstimatePhase>,
    /// Validation rate used.
    pub val_imgs_per_s: f64,
    /// Extra seconds of the first epoch (pipeline compilation, allocations).
    pub startup_s: f64,
    /// Mean seconds per epoch, validation included (`(est_total_s - startup_s) / epochs`).
    pub est_epoch_s: f64,
    /// Estimated wall time of the run, seconds.
    pub est_total_s: f64,
    /// Low end of the range.
    pub est_total_s_low: f64,
    /// High end of the range.
    pub est_total_s_high: f64,
    /// Source of the training rate (the bulk of the time).
    pub source: ThroughputSource,
    /// How far to trust `est_total_s`.
    pub confidence: Confidence,
    /// The throughput row behind the training rate.
    pub basis: EstimateBasis,
    /// Source of the validation rate.
    pub val_source: ThroughputSource,
    /// Peak GPU memory of training in bytes (`null` on the CPU).
    pub est_peak_vram_bytes: Option<u64>,
    /// Source of `est_peak_vram_bytes` (`null` with it).
    pub vram_source: Option<ThroughputSource>,
    /// Close-mosaic epochs the run would train with (`augment.close_mosaic` capped at half the run; the
    /// `close_mosaic` phase's epochs). Additive: 0 from older servers.
    #[serde(default)]
    pub close_mosaic: u32,
    /// The `folo train` command that trains this config on the server's machine (dataset and weights as paths
    /// there, published weights by name; docs/API.md, CLI command). Left out by older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cli: Option<String>,
}

/// `POST /api/system/benchmark` body. Every field has a default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default)]
pub struct BenchmarkRequest {
    /// Architecture to probe.
    pub arch: Arch,
    /// Image size (a multiple of 32 in 64..=1280).
    pub imgsz: u32,
    /// Training batch.
    pub batch: u32,
    /// Probe mixed-precision training (needs tensor cores, like a run).
    pub amp: bool,
    /// Seconds of measuring, 1..=10 (setup comes on top: building the model, compiling GPU pipelines).
    pub budget_s: f64,
}

impl Default for BenchmarkRequest {
    fn default() -> Self {
        BenchmarkRequest { arch: Arch::N, imgsz: 320, batch: 16, amp: false, budget_s: 4.0 }
    }
}

/// `POST /api/system/benchmark` response: what the probe measured and the rows it updated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchmarkReport {
    /// The device probed (the server's inference device).
    pub device: String,
    /// Its name.
    pub device_name: String,
    /// Training images per second over the measured batches (warm-up excluded, no validation).
    pub train_imgs_per_s: f64,
    /// Training batches measured.
    pub train_batches: u64,
    /// Median `/api/predict`-path latency over the measured calls.
    pub latency_ms: f64,
    /// Predictions measured.
    pub predictions: u64,
    /// Wall time of the probe, setup included.
    pub elapsed_s: f64,
    /// The training row after the probe.
    pub training: TrainThroughput,
    /// The inference row after the probe.
    pub inference: InferThroughput,
}
