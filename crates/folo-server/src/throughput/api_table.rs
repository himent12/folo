//! The API's throughput table: the measured rows and the reference rows they shadow.

use folo_api::{Arch, InferPrecision, InferThroughput, ThroughputSource, ThroughputTable, TrainThroughput};

use super::{DeviceId, InferRecord, REFERENCE_BATCH, TableFile, TrainRecord, r1, r2, reference};

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
