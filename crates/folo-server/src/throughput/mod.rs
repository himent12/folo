//! Measured throughput and run-time estimates (docs/API.md, *Throughput and estimates*).
//!
//! The table keeps, per (device, arch, imgsz, batch, amp), the steady-state training rate of every epoch the server
//! trains (with and without mosaic, the session's first epoch excluded), the validation rate, the first epoch's
//! extra time and the peak GPU memory; and per (device, arch, imgsz, precision) the latency of `/api/predict`
//! calls. The real store persists it in `<data dir>/throughput.json`; the mock keeps a fixed one in memory.
//!
//! Where nothing is measured, the numbers come from [`reference`]: docs/BENCHMARKS.md's FOLO figures, scaled to
//! other sizes by compute. A measured row also stands in for nearby sizes: its rate times the reference's ratio
//! between the two ([`estimate`]).

mod api_table;
mod recording;
/// The reference numbers: docs/BENCHMARKS.md's FOLO measurements on the dev machine.
pub mod reference;
mod run_estimate;

use std::time::Duration;

use folo_api::{Arch, InferPrecision, PhaseKind, Rate};
use serde::{Deserialize, Serialize};

pub use api_table::{REFERENCE_SIZES, infer_row, table, train_row};
pub use recording::Throughput;
pub use run_estimate::{EstimateInput, estimate, latency_growth, scale_rate, slowdown};

/// The file in the data dir.
pub const FILE: &str = "throughput.json";
/// Recent samples kept per series (older ones only count in `samples`).
pub const WINDOW: usize = 32;
/// Prediction samples are written at most this often (training samples, once per epoch, right away).
const WRITE_EVERY: Duration = Duration::from_secs(10);
/// The batch the reference rows are listed at (docs/BENCHMARKS.md measured at 16).
pub const REFERENCE_BATCH: u32 = 16;

/// Recent values of one measurement.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Series {
    /// The last [`WINDOW`] values, oldest first.
    pub values: Vec<f64>,
    /// Every value ever added.
    pub count: u64,
}

impl Series {
    /// Adds a value (non-finite and non-positive ones are ignored).
    pub fn push(&mut self, v: f64) {
        if !(v.is_finite() && v > 0.0) {
            return;
        }
        self.values.push(v);
        if self.values.len() > WINDOW {
            self.values.remove(0);
        }
        self.count += 1;
    }

    /// No values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn sorted(&self) -> Vec<f64> {
        let mut v = self.values.clone();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Median of the recent values.
    pub fn median(&self) -> Option<f64> {
        let v = self.sorted();
        let n = v.len();
        match n {
            0 => None,
            _ if n % 2 == 1 => Some(v[n / 2]),
            _ => Some((v[n / 2 - 1] + v[n / 2]) / 2.0),
        }
    }

    /// The 10th and 90th percentiles of the recent values (nearest rank), min and max with fewer than 10.
    pub fn spread(&self) -> Option<(f64, f64)> {
        let v = self.sorted();
        let n = v.len();
        if n == 0 {
            return None;
        }
        if n < 10 {
            return Some((v[0], v[n - 1]));
        }
        let rank = |p: f64| v[((p * n as f64).ceil() as usize).clamp(1, n) - 1];
        Some((rank(0.1), rank(0.9)))
    }

    /// The API rate.
    pub fn rate(&self) -> Option<Rate> {
        let (low, high) = self.spread()?;
        Some(Rate { imgs_per_s: r1(self.median()?), low: r1(low), high: r1(high), samples: self.count })
    }
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// A device as the table keys it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    /// `gpu:<n>` or `cpu`.
    pub device: String,
    /// Adapter or CPU name.
    pub device_name: String,
}

impl DeviceId {
    /// A GPU.
    pub fn is_gpu(&self) -> bool {
        self.device.starts_with("gpu")
    }
}

/// Key of a training row.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrainKey {
    /// Device.
    #[serde(flatten)]
    pub device: DeviceId,
    /// Architecture.
    pub arch: Arch,
    /// Training image size.
    pub imgsz: u32,
    /// Training batch.
    pub batch: u32,
    /// Mixed precision.
    pub amp: bool,
}

/// Key of an inference row.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InferKey {
    /// Device.
    #[serde(flatten)]
    pub device: DeviceId,
    /// Architecture.
    pub arch: Arch,
    /// The model's image size.
    pub imgsz: u32,
    /// Precision it ran in.
    pub precision: InferPrecision,
    /// Tensor-core kernels on.
    pub tensor_cores: bool,
}

/// A measured training row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainRecord {
    /// Key.
    #[serde(flatten)]
    pub key: TrainKey,
    /// Training img/s of epochs with mosaic.
    #[serde(default)]
    pub mosaic: Series,
    /// Training img/s of epochs without mosaic.
    #[serde(default)]
    pub close_mosaic: Series,
    /// Validation img/s.
    #[serde(default)]
    pub val: Series,
    /// Extra seconds of a session's first epoch.
    #[serde(default)]
    pub startup: Series,
    /// Peak GPU memory in bytes (the largest seen).
    #[serde(default)]
    pub peak_vram_bytes: Option<u64>,
    /// Probe samples among `mosaic`.
    #[serde(default)]
    pub probes: u64,
    /// RFC 3339 time of the last sample.
    pub updated: String,
}

/// A measured inference row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferRecord {
    /// Key.
    #[serde(flatten)]
    pub key: InferKey,
    /// `timing.total_ms`.
    #[serde(default)]
    pub total: Series,
    /// `timing.inference_ms`.
    #[serde(default)]
    pub inference: Series,
    /// RFC 3339 time of the last sample.
    pub updated: String,
}

/// `throughput.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TableFile {
    /// Format version (1).
    #[serde(default = "version")]
    pub version: u32,
    /// Training rows.
    #[serde(default)]
    pub training: Vec<TrainRecord>,
    /// Inference rows.
    #[serde(default)]
    pub inference: Vec<InferRecord>,
}

fn version() -> u32 {
    1
}

/// One measured epoch (or probe) of training.
#[derive(Clone, Debug)]
pub struct TrainSample {
    /// Row.
    pub key: TrainKey,
    /// Whether the epoch had mosaic.
    pub phase: PhaseKind,
    /// Training img/s over the epoch (validation excluded).
    pub train_imgs_per_s: Option<f64>,
    /// Validation img/s.
    pub val_imgs_per_s: Option<f64>,
    /// Extra seconds of the session's first epoch.
    pub startup_s: Option<f64>,
    /// Peak GPU memory so far in the run, bytes.
    pub peak_vram_bytes: Option<u64>,
    /// From `POST /api/system/benchmark`.
    pub probe: bool,
}

/// One measured prediction.
#[derive(Clone, Debug)]
pub struct InferSample {
    /// Row.
    pub key: InferKey,
    /// `timing.total_ms`.
    pub total_ms: f64,
    /// `timing.inference_ms`.
    pub inference_ms: f64,
}

#[cfg(test)]
mod tests;
