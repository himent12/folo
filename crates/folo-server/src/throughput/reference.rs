use folo_api::{Arch, InferPrecision, InferThroughput, Rate, ThroughputSource, TrainThroughput};

use super::run_estimate::{compute, latency_growth, scale_rate, slowdown};
use super::{DeviceId, REFERENCE_BATCH, r1, r2};

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
        note: "FOLO trainer, 1080p JPEGs (300 epochs, batch 16)",
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
    let ((val_imgsz, val), startup_s) = if gpu { (GPU_VAL, GPU_STARTUP_S) } else { (CPU_VAL, CPU_STARTUP_S) };
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
    let what =
        if gpu { "GPU end to end, batch 1, f32" } else { "CPU end to end, batch 1, 28 threads, under load" };
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
