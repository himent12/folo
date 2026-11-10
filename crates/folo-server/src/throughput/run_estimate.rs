//! Estimating a run's wall time and peak GPU memory from the table.

use folo_api::{Arch, Confidence, EstimateBasis, EstimatePhase, PhaseKind, RunEstimate, ThroughputSource};

use super::{DeviceId, Series, TrainRecord, r1, r2, reference};

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
pub(super) fn compute(arch: Arch, imgsz: u32) -> f64 {
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
            // At most every sample is a probe (a hand-edited or older file can count more).
            let probes = m.probes.min(samples);
            let what = match (samples - probes, probes) {
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
        cost: None,
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
