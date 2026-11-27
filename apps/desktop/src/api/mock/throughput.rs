//! Speed: the throughput table (what the demo measured, and reference rows), benchmarks, and run
//! estimates.

use crate::api::time;
use crate::api::*;

use super::{CPU_NAME, GPU_NAME, MOCK_NOW, MockBackend, REF_CPU, REF_GPU, REF_NOTE};

impl MockBackend {
    /// [`Backend::benchmark`]'s answer.
    pub(super) fn benchmark_for(&self, req: &BenchmarkRequest) -> ApiResult<BenchmarkReport> {
        self.check()?;
        let busy = self.runs().unwrap_or_default().into_iter().any(|r| {
            !self.fixture_runs.contains(&r.id) && matches!(r.status, RunStatus::Queued | RunStatus::Running)
        });
        if busy {
            return Err(ApiError::new("run_active", "A run is queued or training")
                .with_hint("Measure when the queue is empty."));
        }
        if !req.imgsz.is_multiple_of(32)
            || !(64..=1280).contains(&req.imgsz)
            || !(1..=64).contains(&req.batch)
            || !(req.budget_s.is_finite() && (1.0..=10.0).contains(&req.budget_s))
        {
            return Err(ApiError::new(
                "invalid_body",
                "imgsz must be a multiple of 32 in 64..=1280, batch 1..=64, budget_s 1..=10",
            ));
        }
        // Near the reference, a little off, as a real probe lands.
        let r = reference_train(req.arch, req.imgsz, req.amp, false) * 0.97;
        let lat = reference_latency(req.arch, req.imgsz, req.amp, false) * 1.04;
        let now = time::format(self.sim_now());
        let mut m = self.measured.lock().unwrap();
        let same_train = |t: &TrainThroughput| {
            t.device == "gpu:0"
                && t.arch == req.arch
                && t.imgsz == req.imgsz
                && t.batch == req.batch
                && t.amp == req.amp
        };
        let train = match m.0.iter().position(same_train) {
            Some(i) => {
                let mut t = m.0.remove(i);
                t.imgs_per_s = (t.imgs_per_s * t.samples as f64 + r) / (t.samples + 1) as f64;
                t.samples += 1;
                t.probes += 1;
                t.updated = Some(now.clone());
                t
            }
            None => TrainThroughput {
                samples: 1,
                probes: 1,
                updated: Some(now.clone()),
                ..train_row(
                    "gpu:0",
                    GPU_NAME,
                    req.arch,
                    req.imgsz,
                    req.batch,
                    req.amp,
                    r,
                    ThroughputSource::Measured,
                )
            },
        };
        m.0.insert(0, train.clone());
        // Inference counts predictions (training counts one sample per probe).
        let predictions = (req.budget_s * 0.4 * 1000. / lat).round() as u64;
        let precision = if req.amp { InferPrecision::F16 } else { InferPrecision::F32 };
        let same_infer = |i: &InferThroughput| {
            i.device == "gpu:0" && i.arch == req.arch && i.imgsz == req.imgsz && i.precision == precision
        };
        let infer = match m.1.iter().position(same_infer) {
            Some(i) => {
                let mut row = m.1.remove(i);
                let n = row.samples as f64;
                row.latency_ms = (row.latency_ms * n + lat * predictions as f64) / (n + predictions as f64);
                row.imgs_per_s = 1000. / row.latency_ms;
                row.samples += predictions;
                row.updated = Some(now.clone());
                row
            }
            None => InferThroughput {
                samples: predictions,
                updated: Some(now),
                ..infer_row("gpu:0", GPU_NAME, req.arch, req.imgsz, req.amp, lat, ThroughputSource::Measured)
            },
        };
        m.1.insert(0, infer.clone());
        Ok(BenchmarkReport {
            device: "gpu:0".into(),
            device_name: GPU_NAME.into(),
            train_imgs_per_s: r,
            train_batches: (r * req.budget_s * 0.42 / req.batch as f64).round() as u64,
            latency_ms: lat,
            predictions,
            elapsed_s: req.budget_s + 4.0,
            training: train,
            inference: infer,
        })
    }

    /// [`Backend::estimate_run`] / [`Backend::estimate_run_with_arch`]'s answer.
    pub(super) fn estimate_for(
        &self,
        config: &TrainConfig,
        arch_override: Option<&str>,
    ) -> ApiResult<RunEstimate> {
        self.check()?;
        let model_arch = crate::arch::ModelArch::from_raw_or(arch_override, config.arch);
        if model_arch.is_v2() && config.compute.is_some() {
            return Err(ApiError::new(
                "invalid_config",
                "folo2 architectures are not supported for cloud runs yet (train locally, or use folo-n..folo-x for cloud compute)",
            ));
        }
        let ds = self
            .datasets
            .iter()
            .find(|d| d.ds.id == config.dataset_id)
            .ok_or_else(|| ApiError::new("dataset_not_found", "No such dataset"))?;
        let train = ds.ds.splits.train;
        // No validation split: validation runs on the training images (docs/API.md).
        let val = if ds.ds.splits.val > 0 { ds.ds.splits.val } else { train };
        // Reference rates (the app's own reference model) unless the table has a measured row of
        // this device with the run's arch, image size, batch and precision (docs/API.md: the
        // measured row first).
        let cpu = config.device == "cpu";
        let (secs, vram_gib) = crate::overlays::estimate_arch(
            model_arch,
            config.epochs as f32,
            config.batch,
            config.imgsz,
            train as u32,
        );
        let secs = if cpu { secs * 25. } else { secs };
        let vram_gib = if config.amp { vram_gib * 0.58 } else { vram_gib };
        let epochs = config.epochs.max(1);
        // Without mosaic every epoch is a close-mosaic one; else the last ones, capped at half.
        let close =
            if config.augment.mosaic <= 0. { epochs } else { config.augment.close_mosaic.min(epochs / 2) };
        let startup = 6.0;
        // A cloud config trains on its offer's GPU: nothing measured here applies (no row has
        // that device), so the estimate is the reference, plus what the run would cost.
        let offer = match &config.compute {
            Some(c) => Some(self.compute_offer(c, config)?.1),
            None => None,
        };
        let (device, here) = match &offer {
            Some(o) => ("cloud", o.gpu.as_str()),
            None if cpu => ("cpu", CPU_NAME),
            None => ("gpu:0", GPU_NAME),
        };
        // The basis (docs/API.md): the measured row of this device with the run's arch, image
        // size, batch and precision; else the nearest measured row on it by compute (the same
        // precision preferred), carried over by the reference's ratio between the two settings
        // (the reference has no batch dependence); else the reference itself.
        let rows: Vec<TrainThroughput> = self
            .measured
            .lock()
            .unwrap()
            .0
            .iter()
            .filter(|t| t.device == device && t.measured_on == here)
            .cloned()
            .collect();
        // Model compute: the architecture's GFLOPs at 640 px, scaled by image area.
        let compute = |arch: Arch, imgsz: u32| gflops(arch) * (imgsz as f64 / 640.).powi(2);
        let want = compute(config.arch, config.imgsz);
        let exact = rows.iter().find(|t| {
            t.arch == config.arch && t.imgsz == config.imgsz && t.batch == config.batch && t.amp == config.amp
        });
        // Nearest by compute, then by batch; the same precision before any other.
        let nearest = exact.or_else(|| {
            rows.iter().min_by(|a, b| {
                let key = |t: &TrainThroughput| {
                    (
                        t.amp != config.amp,
                        (compute(t.arch, t.imgsz) / want).ln().abs(),
                        (t.batch.max(1) as f64 / config.batch.max(1) as f64).ln().abs(),
                    )
                };
                let (ka, kb) = (key(a), key(b));
                ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(ka.2.total_cmp(&kb.2))
            })
        });
        // (row, rate carried to this config, within 4x compute with the same precision)
        let measured = nearest.map(|m| {
            let carried = if exact.is_some() {
                m.imgs_per_s
            } else {
                m.imgs_per_s * reference_train(config.arch, config.imgsz, config.amp, cpu)
                    / reference_train(m.arch, m.imgsz, m.amp, cpu)
            };
            let near = m.amp == config.amp && (compute(m.arch, m.imgsz) / want).ln().abs() <= 4f64.ln();
            (m.clone(), carried, near)
        });
        let exact = exact.is_some();
        let ref_epoch_s = secs / epochs as f64;
        // Validation stays at the reference rate (the table's rows time training).
        let val_s = ref_epoch_s * 0.08;
        let (rate, train_s) = match &measured {
            Some((_, r, _)) => (*r, train as f64 / r.max(1e-6)),
            None => (train as f64 / (ref_epoch_s * 0.92), ref_epoch_s * 0.92),
        };
        let epoch_s = train_s + val_s;
        let secs = epoch_s * epochs as f64;
        let phase = |kind: PhaseKind, n: u32| EstimatePhase {
            phase: kind,
            epochs: n,
            train_imgs_per_s: rate,
            train_s,
            val_s,
            epoch_s,
        };
        let phases = [(PhaseKind::Mosaic, epochs - close), (PhaseKind::CloseMosaic, close)]
            .into_iter()
            .filter(|p| p.1 > 0)
            .map(|(k, n)| phase(k, n))
            .collect();
        let device_name = here.to_string();
        // Confidence and range as docs/API.md has them: an exact row, high from 3 samples (±10 %)
        // else medium (-20 / +30 %); a carried row within 4x the compute with the same precision,
        // medium (-25 / +40 %); further, low (-40 / +80 %); the reference, low: -30 / +60 % on the
        // reference machine, -60 / +200 % elsewhere (the demo's CPU isn't the reference CPU).
        let (source, confidence, lo, hi) = match &measured {
            Some((m, _, _)) if exact && m.samples >= 3 => {
                (ThroughputSource::Measured, Confidence::High, 0.9, 1.1)
            }
            Some(_) if exact => (ThroughputSource::Measured, Confidence::Medium, 0.8, 1.3),
            Some((_, _, true)) => (ThroughputSource::Measured, Confidence::Medium, 0.75, 1.4),
            Some(_) => (ThroughputSource::Measured, Confidence::Low, 0.6, 1.8),
            None if (if cpu { REF_CPU } else { REF_GPU }) == here => {
                (ThroughputSource::Reference, Confidence::Low, 0.7, 1.6)
            }
            None => (ThroughputSource::Reference, Confidence::Low, 0.4, 3.0),
        };
        let basis = match &measured {
            Some((m, _, _)) => EstimateBasis {
                source: ThroughputSource::Measured,
                arch: m.arch,
                imgsz: m.imgsz,
                batch: m.batch,
                amp: m.amp,
                measured_on: m.measured_on.clone(),
                samples: m.samples,
                updated: m.updated.clone(),
                exact,
                note: if exact {
                    format!(
                        "Measured on this machine at these settings ({} {}).",
                        m.samples,
                        if m.samples == 1 { "sample" } else { "samples" }
                    )
                } else {
                    format!(
                        "Carried from {} at {} px, batch {}{} measured on this machine, by the reference's ratio.",
                        m.arch.as_str(),
                        m.imgsz,
                        m.batch,
                        if m.amp { " (mixed precision)" } else { "" }
                    )
                },
            },
            None => EstimateBasis {
                source: ThroughputSource::Reference,
                arch: config.arch,
                imgsz: 640,
                batch: 16,
                amp: false,
                measured_on: if cpu { REF_CPU } else { REF_GPU }.into(),
                samples: 0,
                updated: None,
                exact: false,
                note: match &offer {
                    Some(o) => format!(
                        "Reference rates of FOLO's reference machine ({}), scaled to this config; the {} may \
                         be faster or slower.",
                        crate::ui::short_gpu(REF_GPU),
                        o.gpu
                    ),
                    None => "Reference rates for this architecture at 640 px and batch 16, scaled to this \
                             config; nothing has been measured at these settings on this machine."
                        .into(),
                },
            },
        };
        let total_s = secs + startup;
        let cost = offer.as_ref().zip(config.compute.as_ref()).and_then(|(o, c)| {
            let price = o.price_per_hour?;
            let usd = |s: f64| price * s / 3600.;
            Some(CostEstimate {
                price_per_hour: price,
                est_usd: usd(total_s),
                est_usd_low: usd(total_s * lo),
                est_usd_high: usd(total_s * hi),
                max_usd: price * c.max_hours as f64,
            })
        });
        Ok(RunEstimate {
            device: device.into(),
            device_name,
            train_images: train,
            val_images: val,
            batches_per_epoch: train.div_ceil(config.batch.max(1) as u64),
            epochs,
            phases,
            // The rate the phases' validation time is at (val_images / val_imgs_per_s = val_s).
            val_imgs_per_s: val as f64 / val_s.max(1e-6),
            startup_s: startup,
            est_epoch_s: epoch_s,
            est_total_s: secs + startup,
            est_total_s_low: (secs + startup) * lo,
            est_total_s_high: (secs + startup) * hi,
            source,
            confidence,
            basis,
            val_source: ThroughputSource::Reference,
            est_peak_vram_bytes: (!cpu).then_some((vram_gib as f64 * (1u64 << 30) as f64) as u64),
            vram_source: (!cpu).then_some(ThroughputSource::Reference),
            close_mosaic: close,
            cli: Some(self.cli_for(config, arch_override).cli),
            cost,
        })
    }
}

/// An architecture's compute at 640 px, GFLOPs (as New run lists them).
pub(super) fn gflops(arch: Arch) -> f64 {
    match arch.as_str() {
        "folo-n" => 6.6,
        "folo-s" => 21.5,
        "folo-m" => 48.0,
        "folo-l" => 104.0,
        _ => 165.0,
    }
}

/// Reference training rate, img/s at batch 16 (the demo's stand-in for docs/BENCHMARKS.md).
pub(super) fn reference_train(arch: Arch, imgsz: u32, amp: bool, cpu: bool) -> f64 {
    let base = match arch.as_str() {
        "folo-n" => 420.,
        "folo-s" => 260.,
        "folo-m" => 140.,
        "folo-l" => 80.,
        _ => 52.,
    };
    let px = (640. / imgsz as f64).powi(2).min(3.6);
    let amp = if amp { 1.15 } else { 1. };
    base * px * amp / if cpu { 25. } else { 1. }
}

/// Reference latency of one prediction, ms (decode included).
pub(super) fn reference_latency(arch: Arch, imgsz: u32, f16: bool, cpu: bool) -> f64 {
    let base = match arch.as_str() {
        "folo-n" => 4.2,
        "folo-s" => 6.1,
        "folo-m" => 9.8,
        "folo-l" => 15.0,
        _ => 22.0,
    };
    let px = (imgsz as f64 / 640.).powi(2).max(0.3);
    base * px * if f16 { 0.7 } else { 1. } * if cpu { 12. } else { 1. }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn train_row(
    device: &str,
    on: &str,
    arch: Arch,
    imgsz: u32,
    batch: u32,
    amp: bool,
    rate: f64,
    source: ThroughputSource,
) -> TrainThroughput {
    TrainThroughput {
        device: device.into(),
        measured_on: on.into(),
        arch,
        imgsz,
        batch,
        amp,
        imgs_per_s: rate,
        mosaic: None,
        close_mosaic: None,
        val: None,
        startup_s: None,
        peak_vram_bytes: None,
        source,
        samples: 0,
        probes: 0,
        updated: None,
        note: (source == ThroughputSource::Reference).then(|| REF_NOTE.to_string()),
    }
}

pub(super) fn infer_row(
    device: &str,
    on: &str,
    arch: Arch,
    imgsz: u32,
    f16: bool,
    latency: f64,
    source: ThroughputSource,
) -> InferThroughput {
    InferThroughput {
        device: device.into(),
        measured_on: on.into(),
        arch,
        imgsz,
        batch: 1,
        precision: if f16 { InferPrecision::F16 } else { InferPrecision::F32 },
        tensor_cores: f16,
        latency_ms: latency,
        latency_ms_low: latency * 0.9,
        latency_ms_high: latency * 1.2,
        inference_ms: Some(latency * 0.8),
        imgs_per_s: 1000. / latency,
        source,
        samples: 0,
        updated: None,
        note: (source == ThroughputSource::Reference).then(|| REF_NOTE.to_string()),
    }
}

/// What the demo has measured, as `folo serve --mock` has it: folo-n and folo-s at 640 from its
/// runs, a folo-n 320 px amp probe, and folo-n 640 predictions.
pub(super) fn mock_measured() -> (Vec<TrainThroughput>, Vec<InferThroughput>) {
    let at = |hours: f64| Some(time::format(MOCK_NOW - (hours * 3600.) as i64));
    let training = vec![
        TrainThroughput {
            samples: 37,
            updated: at(4.0),
            ..train_row("gpu:0", GPU_NAME, Arch::N, 640, 16, false, 431.8, ThroughputSource::Measured)
        },
        TrainThroughput {
            samples: 118,
            updated: at(9.0),
            ..train_row("gpu:0", GPU_NAME, Arch::S, 640, 32, false, 247.3, ThroughputSource::Measured)
        },
        TrainThroughput {
            samples: 1,
            probes: 1,
            updated: at(26.0),
            ..train_row("gpu:0", GPU_NAME, Arch::N, 320, 16, true, 1392.0, ThroughputSource::Measured)
        },
    ];
    let inference = vec![InferThroughput {
        samples: 54,
        updated: at(2.0),
        ..infer_row("gpu:0", GPU_NAME, Arch::N, 640, false, 4.6, ThroughputSource::Measured)
    }];
    (training, inference)
}

/// The table as the server answers it: measured rows first (newest first), then a reference row
/// for every arch at 640 and 320 px (batch 16), f32 and (GPU) amp / f16, on the GPU and the CPU,
/// unless a measured row of that device has the same settings.
pub(super) fn with_reference(
    mut training: Vec<TrainThroughput>,
    mut inference: Vec<InferThroughput>,
) -> ThroughputTable {
    training.sort_by(|a, b| b.updated.cmp(&a.updated));
    inference.sort_by(|a, b| b.updated.cmp(&a.updated));
    let mut ref_train = Vec::new();
    let mut ref_infer = Vec::new();
    for (device, cpu) in [("gpu:0", false), ("cpu", true)] {
        let on = if cpu { REF_CPU } else { REF_GPU };
        // Only a row measured on this very adapter / CPU replaces the reference (one from
        // another adapter under the same ordinal stays listed beside it).
        let here = if cpu { CPU_NAME } else { GPU_NAME };
        for arch in Arch::ALL {
            for imgsz in [640, 320] {
                for amp in [false, true] {
                    if cpu && amp {
                        continue;
                    }
                    let have = training.iter().any(|t| {
                        t.device == device
                            && t.measured_on == here
                            && t.arch == arch
                            && t.imgsz == imgsz
                            && t.batch == 16
                            && t.amp == amp
                    });
                    if !have {
                        let r = reference_train(arch, imgsz, amp, cpu);
                        ref_train.push(train_row(
                            device,
                            on,
                            arch,
                            imgsz,
                            16,
                            amp,
                            r,
                            ThroughputSource::Reference,
                        ));
                    }
                    let precision = if amp { InferPrecision::F16 } else { InferPrecision::F32 };
                    let have = inference.iter().any(|i| {
                        i.device == device
                            && i.measured_on == here
                            && i.arch == arch
                            && i.imgsz == imgsz
                            && i.precision == precision
                    });
                    if !have {
                        let l = reference_latency(arch, imgsz, amp, cpu);
                        ref_infer.push(infer_row(
                            device,
                            on,
                            arch,
                            imgsz,
                            amp,
                            l,
                            ThroughputSource::Reference,
                        ));
                    }
                }
            }
        }
    }
    training.extend(ref_train);
    inference.extend(ref_infer);
    ThroughputTable { device: "gpu:0".into(), device_name: GPU_NAME.into(), training, inference }
}
