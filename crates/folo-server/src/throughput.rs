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

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use folo_api::{
    Arch, Confidence, EstimateBasis, EstimatePhase, InferPrecision, InferThroughput, PhaseKind, Rate,
    RunEstimate, ThroughputSource, ThroughputTable, TrainThroughput,
};
use serde::{Deserialize, Serialize};

use crate::index::write_atomic;
use crate::util::{rfc3339, unix_now};

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

struct Inner {
    file: TableFile,
    dirty: bool,
    last_write: Option<Instant>,
}

/// The table, persisted (real store) or in memory (mock).
pub struct Throughput {
    path: Option<PathBuf>,
    inner: Mutex<Inner>,
}

impl Throughput {
    /// Opens `path` (a missing file is an empty table; a malformed one is logged and replaced on the next write:
    /// it only holds measurements, which come back).
    pub fn open(path: PathBuf) -> Throughput {
        let file = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!("{}: {e}; starting a new throughput table", path.display());
                TableFile::default()
            }),
            Err(_) => TableFile::default(),
        };
        Throughput { path: Some(path), inner: Mutex::new(Inner { file, dirty: false, last_write: None }) }
    }

    /// A table that is never written.
    pub fn in_memory(file: TableFile) -> Throughput {
        Throughput { path: None, inner: Mutex::new(Inner { file, dirty: false, last_write: None }) }
    }

    /// A copy of the rows.
    pub fn file(&self) -> TableFile {
        self.inner.lock().unwrap().file.clone()
    }

    /// Adds a training sample and writes the file.
    pub fn record_train(&self, s: TrainSample) -> TrainRecord {
        let mut inner = self.inner.lock().unwrap();
        let rows = &mut inner.file.training;
        let i = match rows.iter().position(|r| r.key == s.key) {
            Some(i) => i,
            None => {
                rows.push(TrainRecord {
                    key: s.key.clone(),
                    mosaic: Series::default(),
                    close_mosaic: Series::default(),
                    val: Series::default(),
                    startup: Series::default(),
                    peak_vram_bytes: None,
                    probes: 0,
                    updated: String::new(),
                });
                rows.len() - 1
            }
        };
        let r = &mut rows[i];
        if let Some(v) = s.train_imgs_per_s {
            match s.phase {
                PhaseKind::Mosaic => r.mosaic.push(v),
                PhaseKind::CloseMosaic => r.close_mosaic.push(v),
            }
        }
        if let Some(v) = s.val_imgs_per_s {
            r.val.push(v);
        }
        if let Some(v) = s.startup_s {
            r.startup.push(v);
        }
        if let Some(b) = s.peak_vram_bytes.filter(|&b| b > 0) {
            r.peak_vram_bytes = Some(r.peak_vram_bytes.map_or(b, |m| m.max(b)));
        }
        r.probes += u64::from(s.probe);
        r.updated = rfc3339(unix_now());
        let out = r.clone();
        inner.dirty = true;
        self.write(&mut inner);
        out
    }

    /// Adds a prediction; the file is written at most every [`WRITE_EVERY`] (and by [`Throughput::flush`]).
    pub fn record_infer(&self, s: InferSample) -> InferRecord {
        let mut inner = self.inner.lock().unwrap();
        let rows = &mut inner.file.inference;
        let i = match rows.iter().position(|r| r.key == s.key) {
            Some(i) => i,
            None => {
                rows.push(InferRecord {
                    key: s.key.clone(),
                    total: Series::default(),
                    inference: Series::default(),
                    updated: String::new(),
                });
                rows.len() - 1
            }
        };
        let r = &mut rows[i];
        r.total.push(s.total_ms);
        r.inference.push(s.inference_ms);
        r.updated = rfc3339(unix_now());
        let out = r.clone();
        inner.dirty = true;
        if inner.last_write.is_none_or(|t| t.elapsed() >= WRITE_EVERY) {
            self.write(&mut inner);
        }
        out
    }

    /// Writes pending samples.
    pub fn flush(&self) {
        let mut inner = self.inner.lock().unwrap();
        self.write(&mut inner);
    }

    fn write(&self, inner: &mut Inner) {
        let Some(path) = &self.path else { return };
        if !inner.dirty {
            return;
        }
        let bytes = serde_json::to_vec_pretty(&inner.file).expect("the table serializes");
        match write_atomic(path, &bytes) {
            Ok(()) => {
                inner.dirty = false;
                inner.last_write = Some(Instant::now());
            }
            Err(e) => tracing::warn!("writing {}: {e}", path.display()),
        }
    }

    /// `GET /api/system/throughput`: measured rows, then reference rows for `devices` that no measured row covers.
    pub fn table(&self, auto: &DeviceId, devices: &[DeviceId]) -> ThroughputTable {
        table(&self.inner.lock().unwrap().file, auto, devices)
    }

    /// `POST /api/runs/estimate` on this table.
    pub fn estimate(&self, input: &EstimateInput) -> RunEstimate {
        estimate(&self.inner.lock().unwrap().file.training, input)
    }
}

/// The API row of a measured training record.
pub fn train_row(r: &TrainRecord) -> TrainThroughput {
    TrainThroughput {
        device: r.key.device.device.clone(),
        measured_on: r.key.device.device_name.clone(),
        arch: r.key.arch,
        imgsz: r.key.imgsz,
        batch: r.key.batch,
        amp: r.key.amp,
        imgs_per_s: r1(r.mosaic.median().or_else(|| r.close_mosaic.median()).unwrap_or(0.0)),
        mosaic: r.mosaic.rate(),
        close_mosaic: r.close_mosaic.rate(),
        val: r.val.rate(),
        startup_s: r.startup.median().map(r1),
        peak_vram_bytes: r.peak_vram_bytes,
        source: ThroughputSource::Measured,
        samples: r.mosaic.count + r.close_mosaic.count,
        probes: r.probes,
        updated: Some(r.updated.clone()),
        note: None,
    }
}

/// The API row of a measured inference record.
pub fn infer_row(r: &InferRecord) -> InferThroughput {
    let latency = r.total.median().unwrap_or(0.0);
    let (lo, hi) = r.total.spread().unwrap_or((latency, latency));
    InferThroughput {
        device: r.key.device.device.clone(),
        measured_on: r.key.device.device_name.clone(),
        arch: r.key.arch,
        imgsz: r.key.imgsz,
        batch: 1,
        precision: r.key.precision,
        tensor_cores: r.key.tensor_cores,
        latency_ms: r2(latency),
        latency_ms_low: r2(lo),
        latency_ms_high: r2(hi),
        inference_ms: r.inference.median().map(r2),
        imgs_per_s: if latency > 0.0 { r1(1000.0 / latency) } else { 0.0 },
        source: ThroughputSource::Measured,
        samples: r.total.count,
        updated: Some(r.updated.clone()),
        note: None,
    }
}

/// Image sizes the reference rows are listed at.
pub const REFERENCE_SIZES: [u32; 2] = [640, 320];

/// See [`Throughput::table`].
pub fn table(file: &TableFile, auto: &DeviceId, devices: &[DeviceId]) -> ThroughputTable {
    // Rows without a rate (none are written, but a hand-edited file may have them) neither list nor shadow.
    let rated = || file.training.iter().filter(|r| !(r.mosaic.is_empty() && r.close_mosaic.is_empty()));
    let mut training: Vec<TrainThroughput> = rated().map(train_row).collect();
    training.sort_by(|a, b| b.updated.cmp(&a.updated));
    let mut inference: Vec<InferThroughput> = file.inference.iter().map(infer_row).collect();
    inference.sort_by(|a, b| b.updated.cmp(&a.updated));
    for d in devices {
        let gpu = d.is_gpu();
        for arch in Arch::ALL {
            for imgsz in REFERENCE_SIZES {
                for amp in [false, true] {
                    if amp && !gpu {
                        continue;
                    }
                    // Only a measurement of this very adapter (or CPU) replaces the reference: rows of another
                    // one under the same ordinal (a moved data dir, a new GPU) stay listed with their
                    // `measured_on`, next to the reference.
                    let covered = rated().any(|r| {
                        r.key.device == *d
                            && (r.key.arch, r.key.imgsz, r.key.batch, r.key.amp)
                                == (arch, imgsz, REFERENCE_BATCH, amp)
                    });
                    if !covered {
                        training.push(reference::train_row(d, arch, imgsz, amp));
                    }
                }
                let precisions: &[InferPrecision] =
                    if gpu { &[InferPrecision::F32, InferPrecision::F16] } else { &[InferPrecision::F32] };
                for &p in precisions {
                    let covered = file.inference.iter().any(|r| {
                        r.key.device == *d && (r.key.arch, r.key.imgsz, r.key.precision) == (arch, imgsz, p)
                    });
                    if !covered {
                        inference.push(reference::infer_row(d, arch, imgsz, p));
                    }
                }
            }
        }
    }
    ThroughputTable {
        device: auto.device.clone(),
        device_name: auto.device_name.clone(),
        training,
        inference,
    }
}

/// What [`estimate`] needs to know about a run.
#[derive(Clone, Debug)]
pub struct EstimateInput {
    /// The device it would train on.
    pub device: DeviceId,
    /// Architecture.
    pub arch: Arch,
    /// Image size.
    pub imgsz: u32,
    /// Mixed precision.
    pub amp: bool,
    /// Batch size.
    pub batch: u32,
    /// Epochs.
    pub epochs: u32,
    /// Mosaic is on (`augment.mosaic > 0`).
    pub mosaic: bool,
    /// Epochs without mosaic at the end (the trainer's capped `close_mosaic`).
    pub close_mosaic_epochs: u32,
    /// Training images.
    pub train_images: u64,
    /// Validation images per pass (the training split's count when there is no val split).
    pub val_images: u64,
}

/// GFLOPs of one image of `arch` at `imgsz`.
fn compute(arch: Arch, imgsz: u32) -> f64 {
    crate::models::gflops_640(arch) * (imgsz as f64 / 640.0).powi(2)
}

/// How much slower a model of `q` times the compute runs (`q < 1`: faster). FOLO's trainer and predictor are not
/// compute-bound at these sizes: rates scale with the compute ratio to the 0.55 (fitted on the trainer references:
/// folo-n 640 → 320 px and folo-n → folo-s at 640 both fit 0.54–0.57), beyond 4x larger to the 0.8 (closer to
/// compute-bound), beyond 4x smaller to the 0.25 (per-batch overheads dominate).
pub fn slowdown(q: f64) -> f64 {
    if q >= 1.0 {
        q.min(4.0).powf(0.55) * (q / 4.0).max(1.0).powf(0.8)
    } else {
        let p = 1.0 / q;
        1.0 / (p.min(4.0).powf(0.55) * (p / 4.0).max(1.0).powf(0.25))
    }
}

/// How much longer one batch-1 prediction of a model of `q` times the compute takes. Batch-1 inference is bound
/// by kernel launches and transfers more than training is: the GPU references grow with the compute ratio to the
/// 0.35 (folo-n → folo-s at 640: 5.18 → 7.87 ms for 3.3x the compute), beyond 4x larger to the 0.8, and a smaller
/// input saves little (to the 0.1).
pub fn latency_growth(q: f64) -> f64 {
    if q >= 1.0 { q.min(4.0).powf(0.35) * (q / 4.0).max(1.0).powf(0.8) } else { q.powf(0.1) }
}

/// A rate measured at `from` (arch, imgsz) carried to `to` by [`slowdown`].
pub fn scale_rate(rate: f64, from: (Arch, u32), to: (Arch, u32)) -> f64 {
    rate / slowdown(compute(to.0, to.1) / compute(from.0, from.1))
}

/// The reference numbers: docs/BENCHMARKS.md's FOLO measurements on the dev machine.
pub mod reference {
    use super::*;

    /// The GPU of docs/BENCHMARKS.md.
    pub const GPU_NAME: &str = "NVIDIA GeForce RTX 4080 SUPER";
    /// The CPU of docs/BENCHMARKS.md.
    pub const CPU_NAME: &str = "Intel(R) Core(TM) i7-14700KF";

    /// A measured trainer rate (img/s, validation excluded).
    struct TrainAnchor {
        arch: Arch,
        imgsz: u32,
        amp: bool,
        mosaic: f64,
        close_mosaic: f64,
        note: &'static str,
    }

    const GPU_TRAIN: &[TrainAnchor] = &[
        TrainAnchor {
            arch: Arch::N,
            imgsz: 640,
            amp: false,
            mosaic: 208.0,
            close_mosaic: 215.0,
            note: "FOLO trainer, COCO train2017 (50 epochs, batch 16)",
        },
        TrainAnchor {
            arch: Arch::N,
            imgsz: 320,
            amp: false,
            mosaic: 440.0,
            close_mosaic: 470.0,
            note: "FOLO trainer, shapes (30 epochs, batch 16)",
        },
        TrainAnchor {
            arch: Arch::S,
            imgsz: 640,
            amp: false,
            mosaic: 105.0,
            close_mosaic: 110.0,
            note: "FOLO trainer, R6 (300 epochs, batch 16)",
        },
        TrainAnchor {
            arch: Arch::N,
            imgsz: 320,
            amp: true,
            mosaic: 406.0,
            close_mosaic: 430.0,
            note: "FOLO trainer with amp, shapes (30 epochs, batch 16)",
        },
        TrainAnchor {
            arch: Arch::N,
            imgsz: 640,
            amp: true,
            mosaic: 258.0,
            close_mosaic: 267.0,
            note: "the f32 trainer rate times the amp / f32 training-step ratio (1.24)",
        },
        TrainAnchor {
            arch: Arch::S,
            imgsz: 640,
            amp: true,
            mosaic: 169.0,
            close_mosaic: 177.0,
            note: "the f32 trainer rate times the amp / f32 training-step ratio (1.61)",
        },
    ];

    const CPU_TRAIN: &[TrainAnchor] = &[TrainAnchor {
        arch: Arch::N,
        imgsz: 320,
        amp: false,
        mosaic: 14.0,
        close_mosaic: 20.0,
        note: "FOLO trainer on the CPU under load from other jobs, shapes (30 epochs, batch 16)",
    }];

    /// Validation img/s of folo-n (the rest of an epoch after training: the validation pass and the checkpoint
    /// writes): on the GPU at 640 px (COCO val2017, 5000 images in about 18 s), on the CPU at 320 px (200 images in
    /// about 3.3 s, under load).
    const GPU_VAL: (u32, f64) = (640, 275.0);
    const CPU_VAL: (u32, f64) = (320, 60.0);
    /// Extra seconds of a session's first epoch (shader compilation, pool growth; shapes runs at 320 px).
    const GPU_STARTUP_S: f64 = 6.0;
    const CPU_STARTUP_S: f64 = 1.0;

    /// A reference training rate.
    #[derive(Clone, Debug)]
    pub struct RefTrain {
        /// Mosaic epochs, img/s.
        pub mosaic: f64,
        /// Close-mosaic epochs, img/s.
        pub close_mosaic: f64,
        /// Validation, img/s.
        pub val: f64,
        /// First-epoch extra seconds.
        pub startup_s: f64,
        /// Where it comes from.
        pub note: String,
    }

    /// The reference training rate of (arch, imgsz, amp) on a GPU or the CPU: the nearest anchor by compute (with
    /// the same amp; the CPU has no amp anchor), scaled by [`slowdown`].
    pub fn train(gpu: bool, arch: Arch, imgsz: u32, amp: bool) -> RefTrain {
        let anchors = if gpu { GPU_TRAIN } else { CPU_TRAIN };
        let amp = amp && gpu;
        let c = compute(arch, imgsz);
        let dist = |a: &TrainAnchor| (compute(a.arch, a.imgsz) / c).ln().abs();
        let a = anchors
            .iter()
            .filter(|a| a.amp == amp)
            .min_by(|x, y| dist(x).total_cmp(&dist(y)))
            .expect("an anchor per device and precision");
        let to = (arch, imgsz);
        let from = (a.arch, a.imgsz);
        let ((val_imgsz, val), startup_s) =
            if gpu { (GPU_VAL, GPU_STARTUP_S) } else { (CPU_VAL, CPU_STARTUP_S) };
        let device = if gpu { GPU_NAME } else { CPU_NAME };
        let note = if from == to {
            format!("docs/BENCHMARKS.md: {} on an {device}", a.note)
        } else {
            format!(
                "docs/BENCHMARKS.md: {} {} px ({}) on an {device}, scaled to {} {imgsz} px by compute",
                a.arch.as_str(),
                a.imgsz,
                a.note,
                arch.as_str()
            )
        };
        RefTrain {
            mosaic: scale_rate(a.mosaic, from, to),
            close_mosaic: scale_rate(a.close_mosaic, from, to),
            val: scale_rate(val, (Arch::N, val_imgsz), to),
            startup_s,
            note,
        }
    }

    /// A measured end-to-end predictor latency (ms, batch 1, decoded image in, boxes out) and forward pass.
    struct InferAnchor {
        arch: Arch,
        latency_ms: f64,
        inference_ms: f64,
    }

    /// GPU f32 at 640 (quiet GPU); f16 ties f32 at batch 1 (rows 2 and 5).
    const GPU_INFER: &[InferAnchor] = &[
        InferAnchor { arch: Arch::N, latency_ms: 5.18, inference_ms: 2.99 },
        InferAnchor { arch: Arch::S, latency_ms: 7.87, inference_ms: 5.51 },
    ];
    /// CPU f32 at 640, 28 threads, under load (rows 7 and 8).
    const CPU_INFER: &[InferAnchor] = &[InferAnchor { arch: Arch::N, latency_ms: 26.9, inference_ms: 21.4 }];

    /// The reference latency of (arch, imgsz) at batch 1: `(end to end ms, forward ms, note)`.
    pub fn infer(gpu: bool, arch: Arch, imgsz: u32, precision: InferPrecision) -> (f64, f64, String) {
        let anchors = if gpu { GPU_INFER } else { CPU_INFER };
        let c = compute(arch, imgsz);
        let dist = |a: &InferAnchor| (compute(a.arch, 640) / c).ln().abs();
        let a = anchors.iter().min_by(|x, y| dist(x).total_cmp(&dist(y))).expect("an anchor per device");
        // The CPU is closer to compute-bound: its latency scales like the training rates.
        let q = c / compute(a.arch, 640);
        let k = if gpu { latency_growth(q) } else { slowdown(q) };
        let device = if gpu { GPU_NAME } else { CPU_NAME };
        let what = if gpu {
            "GPU end to end, batch 1, f32"
        } else {
            "CPU end to end, batch 1, 28 threads, under load"
        };
        let mut note = format!("docs/BENCHMARKS.md: {} 640 px, {what}, on an {device}", a.arch.as_str());
        if (a.arch, 640) != (arch, imgsz) {
            note.push_str(&format!(", scaled to {} {imgsz} px by compute", arch.as_str()));
        }
        if precision == InferPrecision::F16 {
            note.push_str("; f16 ties f32 at batch 1");
        }
        note.push_str("; excludes decoding the image file");
        (a.latency_ms * k, a.inference_ms * k, note)
    }

    /// Learnable parameters at 80 classes (docs/MODEL.md §4).
    pub fn params(arch: Arch) -> u64 {
        match arch {
            Arch::N => 2_624_064,
            Arch::S => 9_458_736,
            Arch::M => 20_114_672,
            Arch::L => 25_372_144,
            Arch::X => 56_966_160,
        }
    }

    /// Activation MiB per image at 640 in f32, and the amp factor: folo-n and folo-s from the measured training
    /// step peaks (3474 / 2022 and 6541 / 3713 MiB at batch 16), folo-m / l / x by Studio's per-image ratios.
    fn activation(arch: Arch) -> (f64, f64) {
        match arch {
            Arch::N => (214.6, 0.577),
            Arch::S => (399.8, 0.558),
            Arch::M => (757.0, 0.56),
            Arch::L => (989.0, 0.56),
            Arch::X => (1683.0, 0.56),
        }
    }

    /// The reference peak GPU memory of training: weights, gradients, optimizer state and EMA (16 bytes a
    /// parameter) plus the activations of the batch.
    pub fn memory(arch: Arch, imgsz: u32, batch: u32, amp: bool) -> u64 {
        let (act, amp_factor) = activation(arch);
        let per_image = act * (1u64 << 20) as f64 * (imgsz as f64 / 640.0).powi(2);
        let per_image = if amp { per_image * amp_factor } else { per_image };
        params(arch) * 16 + (batch as f64 * per_image) as u64
    }

    /// The machine the references were measured on matches `d` (the adapter or CPU name).
    pub fn same_hardware(d: &DeviceId) -> bool {
        let marker = if d.is_gpu() { "RTX 4080 SUPER" } else { "i7-14700KF" };
        d.device_name.contains(marker)
    }

    /// Time multipliers of a reference estimate: about +-40% on the reference machine (other jobs moved the
    /// trainer rates 2x), much wider elsewhere.
    pub fn band(d: &DeviceId) -> (f64, f64) {
        if same_hardware(d) { (0.7, 1.6) } else { (0.4, 3.0) }
    }

    fn rate(v: f64, d: &DeviceId) -> Rate {
        let (lo, hi) = band(d);
        Rate { imgs_per_s: r1(v), low: r1(v / hi), high: r1(v / lo), samples: 0 }
    }

    /// The reference training row of (arch, imgsz, amp) on `d`, at [`REFERENCE_BATCH`].
    pub fn train_row(d: &DeviceId, arch: Arch, imgsz: u32, amp: bool) -> TrainThroughput {
        let gpu = d.is_gpu();
        let t = train(gpu, arch, imgsz, amp);
        TrainThroughput {
            device: d.device.clone(),
            measured_on: (if gpu { GPU_NAME } else { CPU_NAME }).to_string(),
            arch,
            imgsz,
            batch: REFERENCE_BATCH,
            amp,
            imgs_per_s: r1(t.mosaic),
            mosaic: Some(rate(t.mosaic, d)),
            close_mosaic: Some(rate(t.close_mosaic, d)),
            val: Some(rate(t.val, d)),
            startup_s: Some(t.startup_s),
            peak_vram_bytes: gpu.then(|| memory(arch, imgsz, REFERENCE_BATCH, amp)),
            source: ThroughputSource::Reference,
            samples: 0,
            probes: 0,
            updated: None,
            note: Some(t.note),
        }
    }

    /// The reference inference row.
    pub fn infer_row(d: &DeviceId, arch: Arch, imgsz: u32, precision: InferPrecision) -> InferThroughput {
        let gpu = d.is_gpu();
        let (latency, forward, note) = infer(gpu, arch, imgsz, precision);
        let (lo, hi) = band(d);
        InferThroughput {
            device: d.device.clone(),
            measured_on: (if gpu { GPU_NAME } else { CPU_NAME }).to_string(),
            arch,
            imgsz,
            batch: 1,
            precision,
            tensor_cores: false,
            latency_ms: r2(latency),
            latency_ms_low: r2(latency * lo),
            latency_ms_high: r2(latency * hi),
            inference_ms: Some(r2(forward)),
            imgs_per_s: r1(1000.0 / latency),
            source: ThroughputSource::Reference,
            samples: 0,
            updated: None,
            note: Some(note),
        }
    }
}

/// Estimates a run's wall time from the measured training rows of `records` (falling back to [`reference`]).
///
/// - The basis: the measured row of the same device with the run's arch, imgsz, batch and amp; else the nearest
///   one by compute and batch (preferring the same amp), its rates carried over by the reference's ratio between
///   the two settings (the reference does not depend on the batch); else the reference itself.
/// - Each epoch trains `train_images` at the phase's rate (mosaic, then the close-mosaic epochs) and validates
///   `val_images` at the validation rate; the first epoch adds `startup_s`.
/// - The range: the basis's spread (at least -10% / +10%) for an exact measured row with 3 or more samples, wider
///   for fewer samples, scaled rows and references ([`reference::band`]).
pub fn estimate(records: &[TrainRecord], input: &EstimateInput) -> RunEstimate {
    let d = &input.device;
    let gpu = d.is_gpu();
    let amp = input.amp && gpu;
    let rt = reference::train(gpu, input.arch, input.imgsz, amp);
    let c = compute(input.arch, input.imgsz);
    // Distance of a measured row from the run: the compute ratio, half the batch ratio, and the precision.
    let dist = |r: &TrainRecord| {
        (compute(r.key.arch, r.key.imgsz) / c).ln().abs()
            + 0.5 * (r.key.batch.max(1) as f64 / input.batch.max(1) as f64).ln().abs()
            + if r.key.amp == amp { 0.0 } else { 1.5 }
    };
    let basis = records
        .iter()
        .filter(|r| r.key.device == *d && !(r.mosaic.is_empty() && r.close_mosaic.is_empty()))
        .min_by(|a, b| dist(a).total_cmp(&dist(b)));

    let (mosaic_rate, close_rate, val_rate, val_source, startup_s, basis_api, confidence, (lo, hi));
    match basis {
        Some(m) => {
            let exact = (m.key.arch, m.key.imgsz, m.key.batch, m.key.amp)
                == (input.arch, input.imgsz, input.batch, amp);
            let rm = reference::train(gpu, m.key.arch, m.key.imgsz, m.key.amp);
            // Carry a measured rate to the target with the reference's shape: `measured * ref(target) / ref(m)`.
            let carry =
                |measured: &Series, ref_m: f64, ref_t: f64| measured.median().map(|v| v * ref_t / ref_m);
            let mosaic = carry(&m.mosaic, rm.mosaic, rt.mosaic)
                .or_else(|| carry(&m.close_mosaic, rm.close_mosaic, rt.mosaic))
                .expect("a basis has a rate");
            let close = carry(&m.close_mosaic, rm.close_mosaic, rt.close_mosaic)
                .or_else(|| carry(&m.mosaic, rm.mosaic, rt.close_mosaic))
                .expect("a basis has a rate");
            mosaic_rate = mosaic;
            close_rate = close;
            (val_rate, val_source) = match carry(&m.val, rm.val, rt.val) {
                Some(v) => (v, ThroughputSource::Measured),
                None => (rt.val, ThroughputSource::Reference),
            };
            startup_s = m.startup.median().filter(|_| exact).unwrap_or(rt.startup_s);
            // The main series of the basis: the one the run spends most epochs in.
            let main = if input.mosaic && input.close_mosaic_epochs * 2 < input.epochs && !m.mosaic.is_empty()
            {
                &m.mosaic
            } else if !m.close_mosaic.is_empty() {
                &m.close_mosaic
            } else {
                &m.mosaic
            };
            let samples = m.mosaic.count + m.close_mosaic.count;
            let far = m.key.amp != amp || dist(m) > 4f64.ln() + 1e-9;
            let (conf, band) = if exact {
                let med = main.median().unwrap_or(1.0);
                let (slow, fast) = main.spread().unwrap_or((med, med));
                let (min_lo, min_hi) = if samples >= 3 { (0.9, 1.1) } else { (0.8, 1.3) };
                let conf = if samples >= 3 { Confidence::High } else { Confidence::Medium };
                (conf, ((med / fast).min(min_lo), (med / slow).max(min_hi)))
            } else if far {
                (Confidence::Low, (0.6, 1.8))
            } else {
                (Confidence::Medium, (0.75, 1.4))
            };
            confidence = conf;
            (lo, hi) = band;
            let what = match (samples - m.probes, m.probes) {
                (e, 0) => format!("{e} epochs of runs"),
                (0, p) => format!("{p} benchmark probes"),
                (e, p) => format!("{e} epochs of runs and {p} benchmark probes"),
            };
            let note = if exact {
                format!("measured on this device with these settings: {what}")
            } else {
                format!(
                    "measured on this device for {} {} px batch {}{} ({what}), carried to {} {} px batch {}{} by \
                     the reference's ratio between the two",
                    m.key.arch.as_str(),
                    m.key.imgsz,
                    m.key.batch,
                    if m.key.amp { " amp" } else { "" },
                    input.arch.as_str(),
                    input.imgsz,
                    input.batch,
                    if amp { " amp" } else { "" }
                )
            };
            basis_api = EstimateBasis {
                source: ThroughputSource::Measured,
                arch: m.key.arch,
                imgsz: m.key.imgsz,
                batch: m.key.batch,
                amp: m.key.amp,
                measured_on: m.key.device.device_name.clone(),
                samples,
                updated: Some(m.updated.clone()),
                exact,
                note,
            };
        }
        None => {
            mosaic_rate = rt.mosaic;
            close_rate = rt.close_mosaic;
            val_rate = rt.val;
            val_source = ThroughputSource::Reference;
            startup_s = rt.startup_s;
            confidence = Confidence::Low;
            (lo, hi) = reference::band(d);
            let elsewhere =
                if reference::same_hardware(d) { "" } else { "; not measured on this device, which differs" };
            basis_api = EstimateBasis {
                source: ThroughputSource::Reference,
                arch: input.arch,
                imgsz: input.imgsz,
                batch: input.batch,
                amp,
                measured_on: (if gpu { reference::GPU_NAME } else { reference::CPU_NAME }).to_string(),
                samples: 0,
                updated: None,
                exact: true,
                note: format!("{}{elsewhere}", rt.note),
            };
        }
    }

    let close_epochs = if input.mosaic { input.close_mosaic_epochs.min(input.epochs) } else { input.epochs };
    let mosaic_epochs = input.epochs - close_epochs;
    let train_n = input.train_images as f64;
    let val_s = input.val_images as f64 / val_rate;
    let phase = |kind, epochs, rate: f64| EstimatePhase {
        phase: kind,
        epochs,
        train_imgs_per_s: r1(rate),
        train_s: r2(train_n / rate),
        val_s: r2(val_s),
        epoch_s: r2(train_n / rate + val_s),
    };
    let mut phases = Vec::new();
    let mut run_s = 0.0;
    for (kind, epochs, rate) in
        [(PhaseKind::Mosaic, mosaic_epochs, mosaic_rate), (PhaseKind::CloseMosaic, close_epochs, close_rate)]
    {
        if epochs > 0 {
            run_s += epochs as f64 * (train_n / rate + val_s);
            phases.push(phase(kind, epochs, rate));
        }
    }
    let total = startup_s + run_s;
    let (est_peak_vram_bytes, vram_source) = if gpu { peak_vram(records, input) } else { (None, None) };
    RunEstimate {
        device: d.device.clone(),
        device_name: d.device_name.clone(),
        train_images: input.train_images,
        val_images: input.val_images,
        batches_per_epoch: input.train_images.div_ceil(input.batch.max(1) as u64),
        epochs: input.epochs,
        phases,
        val_imgs_per_s: r1(val_rate),
        startup_s: r1(startup_s),
        est_epoch_s: r1(run_s / input.epochs.max(1) as f64),
        est_total_s: r1(total),
        est_total_s_low: r1(total * lo),
        est_total_s_high: r1(total * hi),
        source: basis_api.source,
        confidence,
        basis: basis_api,
        val_source,
        est_peak_vram_bytes,
        vram_source,
        close_mosaic: input.close_mosaic_epochs,
        cli: None,
    }
}

/// Peak GPU memory of the run: measured with its device, arch, imgsz and amp (at the nearest batch, scaled by the
/// reference's batch dependence), else the reference.
fn peak_vram(records: &[TrainRecord], input: &EstimateInput) -> (Option<u64>, Option<ThroughputSource>) {
    let reference = |batch| reference::memory(input.arch, input.imgsz, batch, input.amp) as f64;
    let measured = records
        .iter()
        .filter(|r| {
            r.key.device == input.device
                && (r.key.arch, r.key.imgsz, r.key.amp) == (input.arch, input.imgsz, input.amp)
        })
        .filter_map(|r| Some((r.key.batch, r.peak_vram_bytes?)))
        .min_by_key(|(b, _)| (*b as i64 - input.batch as i64).unsigned_abs());
    match measured {
        Some((batch, bytes)) => {
            let bytes = bytes as f64 * reference(input.batch) / reference(batch);
            (Some(bytes as u64), Some(ThroughputSource::Measured))
        }
        None => (Some(reference(input.batch) as u64), Some(ThroughputSource::Reference)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> DeviceId {
        DeviceId { device: "gpu:0".into(), device_name: "NVIDIA GeForce RTX 4080 SUPER".into() }
    }

    fn series(values: &[f64]) -> Series {
        let mut s = Series::default();
        for &v in values {
            s.push(v);
        }
        s
    }

    fn record(arch: Arch, imgsz: u32, amp: bool, mosaic: &[f64], close: &[f64], val: &[f64]) -> TrainRecord {
        TrainRecord {
            key: TrainKey { device: gpu(), arch, imgsz, batch: 16, amp },
            mosaic: series(mosaic),
            close_mosaic: series(close),
            val: series(val),
            startup: series(&[4.0]),
            peak_vram_bytes: Some(3_000_000_000),
            probes: 0,
            updated: "2026-11-08T06:00:00Z".into(),
        }
    }

    fn input(arch: Arch, imgsz: u32, amp: bool) -> EstimateInput {
        EstimateInput {
            device: gpu(),
            arch,
            imgsz,
            amp,
            batch: 16,
            epochs: 100,
            mosaic: true,
            close_mosaic_epochs: 10,
            train_images: 8000,
            val_images: 1000,
        }
    }

    #[test]
    fn series_median_and_spread() {
        let s = series(&[3.0, 1.0, 2.0, f64::NAN, -1.0, 4.0]);
        assert_eq!(s.count, 4, "non-finite and negative values are dropped");
        assert_eq!(s.median(), Some(2.5));
        assert_eq!(s.spread(), Some((1.0, 4.0)), "min and max below 10 samples");
        let s = series(&(1..=40).map(f64::from).collect::<Vec<_>>());
        assert_eq!((s.values.len(), s.count), (WINDOW, 40), "the window keeps the last 32");
        // Values 9..=40: nearest-rank p10 is the 4th (12), p90 the 29th (37).
        assert_eq!(s.spread(), Some((12.0, 37.0)));
        assert_eq!(s.median(), Some(24.5));
    }

    /// The estimate against a hand-computed case: an exact measured row, 100 epochs, the last 10 without mosaic.
    #[test]
    fn estimate_matches_a_hand_computed_case() {
        // Mosaic 200 img/s, close-mosaic 250 img/s, validation 500 img/s, startup 4 s, 3 samples each.
        let rec = record(Arch::N, 640, false, &[190.0, 200.0, 210.0], &[250.0, 250.0, 250.0], &[500.0; 3]);
        let e = estimate(&[rec], &input(Arch::N, 640, false));
        // Mosaic epoch: 8000 / 200 = 40 s training + 1000 / 500 = 2 s validation = 42 s, 90 of them.
        // Close-mosaic epoch: 8000 / 250 = 32 + 2 = 34 s, 10 of them. Total: 4 + 90 * 42 + 10 * 34 = 4124 s.
        assert_eq!(e.phases.len(), 2);
        assert_eq!(
            (e.phases[0].phase, e.phases[0].epochs, e.phases[0].epoch_s),
            (PhaseKind::Mosaic, 90, 42.0)
        );
        assert_eq!((e.phases[0].train_s, e.phases[0].val_s), (40.0, 2.0));
        assert_eq!(
            (e.phases[1].phase, e.phases[1].epochs, e.phases[1].epoch_s),
            (PhaseKind::CloseMosaic, 10, 34.0)
        );
        assert_eq!(e.est_total_s, 4124.0);
        assert_eq!(e.est_epoch_s, 41.2, "(4124 - 4) / 100");
        assert_eq!(e.batches_per_epoch, 500);
        assert_eq!(
            (e.source, e.val_source, e.confidence),
            (ThroughputSource::Measured, ThroughputSource::Measured, Confidence::High)
        );
        assert!(e.basis.exact);
        // Spread 190..210 around 200: -5% / +5.3%, widened to the -10% / +10% floor.
        assert_eq!((e.est_total_s_low, e.est_total_s_high), (r1(4124.0 * 0.9), r1(4124.0 * 1.1)));
        assert_eq!(
            (e.est_peak_vram_bytes, e.vram_source),
            (Some(3_000_000_000), Some(ThroughputSource::Measured))
        );
    }

    #[test]
    fn no_mosaic_trains_every_epoch_at_the_close_mosaic_rate() {
        let rec = record(Arch::N, 640, false, &[200.0; 3], &[250.0; 3], &[500.0; 3]);
        let e = estimate(&[rec], &EstimateInput { mosaic: false, ..input(Arch::N, 640, false) });
        assert_eq!(e.phases.len(), 1);
        assert_eq!((e.phases[0].phase, e.phases[0].epochs), (PhaseKind::CloseMosaic, 100));
        assert_eq!(e.est_total_s, 4.0 + 100.0 * 34.0);
    }

    #[test]
    fn without_measurements_the_estimate_is_the_labelled_reference() {
        let e = estimate(&[], &input(Arch::N, 640, false));
        assert_eq!(
            (e.source, e.val_source, e.confidence),
            (ThroughputSource::Reference, ThroughputSource::Reference, Confidence::Low)
        );
        assert_eq!(e.vram_source, Some(ThroughputSource::Reference));
        assert!(e.basis.note.contains("docs/BENCHMARKS.md"), "{}", e.basis.note);
        // The COCO anchor: 8000 / 208 + 1000 / 275 per mosaic epoch.
        let mosaic = 8000.0 / 208.0 + 1000.0 / 275.0;
        assert!((e.phases[0].epoch_s - mosaic).abs() < 0.06, "{} vs {mosaic}", e.phases[0].epoch_s);
        let near = |a: f64, b: f64| (a - b).abs() <= 0.2;
        assert!(
            near(e.est_total_s_low, e.est_total_s * 0.7) && near(e.est_total_s_high, e.est_total_s * 1.6),
            "{e:?}"
        );
        // The reference memory model reproduces the measured folo-n step peak (3474 MiB at batch 16).
        let mib = e.est_peak_vram_bytes.unwrap() as f64 / (1u64 << 20) as f64;
        assert!((mib - 3474.0).abs() < 2.0, "{mib}");
        // Another GPU: same numbers, a much wider range, and the note says so.
        let other = DeviceId { device: "gpu:0".into(), device_name: "AMD Radeon RX 7900 XTX".into() };
        let e2 = estimate(&[], &EstimateInput { device: other, ..input(Arch::N, 640, false) });
        assert_eq!(e2.est_total_s, e.est_total_s);
        assert!(near(e2.est_total_s_high, e.est_total_s * 3.0), "{e2:?}");
        assert!(e2.basis.note.contains("not measured on this device"), "{}", e2.basis.note);
    }

    #[test]
    fn a_nearby_measurement_is_carried_by_the_reference_shape() {
        // Measured folo-n at 640, f32: twice the reference's 208 img/s.
        let rec = record(Arch::N, 640, false, &[416.0; 5], &[], &[]);
        let e = estimate(&[rec], &input(Arch::N, 320, false));
        assert_eq!((e.source, e.confidence), (ThroughputSource::Measured, Confidence::Medium));
        assert!(!e.basis.exact);
        assert_eq!((e.basis.arch, e.basis.imgsz), (Arch::N, 640));
        // Twice the reference at 320 px (the shapes anchor, 440 img/s), the close phase from the mosaic series.
        assert_eq!(e.phases[0].train_imgs_per_s, 880.0);
        assert_eq!(e.phases[1].train_imgs_per_s, r1(416.0 * 470.0 / 208.0));
        assert_eq!(e.val_source, ThroughputSource::Reference, "no validation measured");
        // Asking for amp with only f32 measured: still carried, but low confidence.
        let rec = record(Arch::N, 640, false, &[416.0; 5], &[], &[]);
        let e = estimate(&[rec], &input(Arch::N, 640, true));
        assert_eq!((e.source, e.confidence), (ThroughputSource::Measured, Confidence::Low));
        assert_eq!(e.phases[0].train_imgs_per_s, r1(416.0 * 258.0 / 208.0));
        // Another batch: the same rate (the reference has no batch dependence), not exact, the VRAM scaled.
        let rec = record(Arch::N, 640, false, &[416.0; 5], &[], &[]);
        let e = estimate(&[rec], &EstimateInput { batch: 8, ..input(Arch::N, 640, false) });
        assert_eq!(
            (e.source, e.confidence, e.basis.exact, e.basis.batch),
            (ThroughputSource::Measured, Confidence::Medium, false, 16)
        );
        assert_eq!(e.phases[0].train_imgs_per_s, 416.0);
        let want = 3e9 * reference::memory(Arch::N, 640, 8, false) as f64
            / reference::memory(Arch::N, 640, 16, false) as f64;
        assert_eq!(
            (e.est_peak_vram_bytes, e.vram_source),
            (Some(want as u64), Some(ThroughputSource::Measured))
        );
        // A measurement on another device is never used.
        let mut rec = record(Arch::N, 640, false, &[416.0; 5], &[], &[]);
        rec.key.device.device = "cpu".into();
        assert_eq!(estimate(&[rec], &input(Arch::N, 640, false)).source, ThroughputSource::Reference);
    }

    #[test]
    fn slowdown_is_continuous_and_monotonic() {
        let mut last = 0.0;
        for i in 1..400 {
            let q = i as f64 / 20.0;
            let s = slowdown(q);
            assert!(s > last, "{q}");
            last = s;
        }
        assert!((slowdown(1.0) - 1.0).abs() < 1e-12);
        assert!((slowdown(4.0) - 4f64.powf(0.55)).abs() < 1e-12);
        assert!((slowdown(0.25) * 4f64.powf(0.55) - 1.0).abs() < 1e-12);
        let mut last = 0.0;
        for i in 1..400 {
            let g = latency_growth(i as f64 / 20.0);
            assert!(g > last, "{i}");
            last = g;
        }
        assert!((latency_growth(1.0) - 1.0).abs() < 1e-12 && latency_growth(0.25) > 0.85);
    }

    #[test]
    fn table_labels_references_and_lets_measured_rows_shadow_them() {
        let mut file = TableFile::default();
        file.training.push(record(Arch::N, 640, false, &[200.0; 3], &[], &[300.0; 3]));
        let cpu = DeviceId { device: "cpu".into(), device_name: "Some CPU".into() };
        let t = table(&file, &gpu(), &[gpu(), cpu]);
        let measured: Vec<_> = t.training.iter().filter(|r| r.source == ThroughputSource::Measured).collect();
        assert_eq!(measured.len(), 1);
        assert_eq!((measured[0].samples, measured[0].note.as_deref()), (3, None));
        let n640: Vec<_> = t
            .training
            .iter()
            .filter(|r| r.device == "gpu:0" && r.arch == Arch::N && r.imgsz == 640 && !r.amp)
            .collect();
        assert_eq!(n640.len(), 1, "the measured row replaces the reference one");
        // GPU: 5 arches x 2 sizes x 2 precisions - 1 measured; CPU: 5 x 2, f32 only.
        assert_eq!(t.training.len(), 1 + 19 + 10);
        for r in t.training.iter().filter(|r| r.source == ThroughputSource::Reference) {
            assert_eq!((r.samples, r.updated.as_ref()), (0, None));
            assert!(r.note.as_ref().is_some_and(|n| n.contains("docs/BENCHMARKS.md")));
        }
        assert_eq!(t.inference.len(), 20 + 10);
        // A row measured on another adapter under the same ordinal is listed, but the reference stays.
        let mut old = record(Arch::S, 640, false, &[100.0; 3], &[], &[]);
        old.key.device.device_name = "AMD Radeon RX 7900 XTX".into();
        file.training.push(old);
        // And an empty row (no rate) neither lists nor shadows.
        let mut empty = record(Arch::M, 640, false, &[], &[], &[]);
        empty.peak_vram_bytes = None;
        file.training.push(empty);
        let t = table(&file, &gpu(), &[gpu()]);
        let s640: Vec<_> =
            t.training.iter().filter(|r| r.arch == Arch::S && r.imgsz == 640 && !r.amp).collect();
        assert_eq!(s640.len(), 2, "{s640:?}");
        assert!(
            s640.iter().any(|r| r.source == ThroughputSource::Reference && r.measured_on.contains("4080"))
        );
        assert!(
            s640.iter().any(|r| r.source == ThroughputSource::Measured && r.measured_on.contains("7900"))
        );
        let m640: Vec<_> =
            t.training.iter().filter(|r| r.arch == Arch::M && r.imgsz == 640 && !r.amp).collect();
        assert_eq!((m640.len(), m640[0].source), (1, ThroughputSource::Reference));
        assert!(t.inference.iter().all(|r| r.source == ThroughputSource::Reference));
    }

    #[test]
    fn the_table_round_trips_through_its_file() {
        let dir = std::env::temp_dir().join(format!("folo-throughput-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join(FILE);
        let t = Throughput::open(path.clone());
        let key = TrainKey { device: gpu(), arch: Arch::S, imgsz: 640, batch: 8, amp: true };
        for (i, phase) in
            [PhaseKind::Mosaic, PhaseKind::Mosaic, PhaseKind::CloseMosaic].into_iter().enumerate()
        {
            t.record_train(TrainSample {
                key: key.clone(),
                phase,
                train_imgs_per_s: Some(150.0 + i as f64),
                val_imgs_per_s: Some(300.0),
                startup_s: (i == 0).then_some(5.0),
                peak_vram_bytes: Some(3_900_000_000 + i as u64),
                probe: i == 1,
            });
        }
        let ikey = InferKey {
            device: gpu(),
            arch: Arch::N,
            imgsz: 640,
            precision: InferPrecision::F16,
            tensor_cores: true,
        };
        // Predictions are written at most every WRITE_EVERY (the training sample was just written) or on a flush.
        t.record_infer(InferSample { key: ikey.clone(), total_ms: 6.0, inference_ms: 3.0 });
        t.record_infer(InferSample { key: ikey.clone(), total_ms: 8.0, inference_ms: 4.0 });
        assert!(Throughput::open(path.clone()).file().inference.is_empty());
        t.flush();
        let back = Throughput::open(path.clone()).file();
        assert_eq!(back, t.file());
        let r = &back.training[0];
        assert_eq!(
            (r.mosaic.count, r.close_mosaic.count, r.val.count, r.startup.count, r.probes),
            (2, 1, 3, 1, 1)
        );
        assert_eq!(r.peak_vram_bytes, Some(3_900_000_002), "the largest peak");
        let row = train_row(r);
        assert_eq!(
            (row.source, row.batch, row.measured_on.as_str()),
            (ThroughputSource::Measured, 8, "NVIDIA GeForce RTX 4080 SUPER")
        );
        assert_eq!(
            (row.imgs_per_s, row.mosaic.unwrap().imgs_per_s),
            (150.5, 150.5),
            "the headline rate is the mosaic one"
        );
        assert_eq!(row.samples, 3);
        let irow = infer_row(&back.inference[0]);
        assert_eq!((irow.latency_ms, irow.samples, irow.inference_ms, irow.batch), (7.0, 2, Some(3.5), 1));
        // A malformed file is an empty table (and is replaced on the next write).
        std::fs::write(&path, b"{ not json").unwrap();
        assert_eq!(Throughput::open(path).file(), TableFile::default());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
