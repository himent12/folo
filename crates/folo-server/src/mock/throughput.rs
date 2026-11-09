//! The mock's throughput table: measured rows for the sizes its finished runs trained (folo-n and folo-s at
//! 640 px on the RTX 4080 SUPER, a folo-n amp probe at 320 px) and for folo-n predictions, so clients see both
//! sources; every other size comes from the reference.

use folo_api::{Arch, InferPrecision};

use crate::throughput::{
    DeviceId, InferKey, InferRecord, Series, TableFile, TrainKey, TrainRecord, reference,
};
use crate::util::noise;

/// The mock GPU as the table keys it.
pub fn gpu() -> DeviceId {
    DeviceId { device: "gpu:0".into(), device_name: super::MOCK_GPU.into() }
}

/// The mock CPU (the reference machine's).
pub fn cpu() -> DeviceId {
    DeviceId { device: "cpu".into(), device_name: reference::CPU_NAME.into() }
}

/// `count` samples around `base` (±`spread`, deterministic in `seed`); the window keeps the last 32.
fn series(seed: u64, base: f64, spread: f64, count: u64) -> Series {
    let mut s = Series::default();
    for i in 0..count {
        let v = base * (1.0 + spread * noise(&[seed, i]) as f64);
        s.push((v * 10.0).round() / 10.0);
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn train(
    seed: u64,
    arch: Arch,
    imgsz: u32,
    amp: bool,
    (mosaic, mosaic_n): (f64, u64),
    (close, close_n): (f64, u64),
    (val, val_n): (f64, u64),
    memory: u64,
    probes: u64,
    updated: &str,
) -> TrainRecord {
    TrainRecord {
        key: TrainKey { device: gpu(), arch, imgsz, batch: 16, amp },
        mosaic: series(seed, mosaic, 0.04, mosaic_n),
        close_mosaic: series(seed + 1, close, 0.03, close_n),
        val: series(seed + 2, val, 0.05, val_n),
        startup: series(seed + 3, 5.4, 0.2, val_n.min(3)),
        peak_vram_bytes: Some(memory),
        probes,
        updated: updated.into(),
    }
}

/// The fixed table.
pub fn table() -> TableFile {
    TableFile {
        version: 1,
        training: vec![
            train(
                0x7c01,
                Arch::N,
                640,
                false,
                (212.0, 81),
                (219.0, 18),
                (281.0, 99),
                3_652_000_000,
                0,
                "2026-11-07T21:14:09Z",
            ),
            train(
                0x3c01,
                Arch::S,
                640,
                false,
                (108.0, 54),
                (0.0, 0),
                (146.0, 54),
                6_871_000_000,
                0,
                "2026-11-06T17:40:51Z",
            ),
            train(
                0x5a03,
                Arch::N,
                320,
                true,
                (415.0, 1),
                (0.0, 0),
                (0.0, 0),
                1_103_000_000,
                1,
                "2026-11-08T04:02:33Z",
            ),
        ],
        inference: vec![InferRecord {
            key: InferKey {
                device: gpu(),
                arch: Arch::N,
                imgsz: 640,
                precision: InferPrecision::F32,
                tensor_cores: false,
            },
            total: series(0x31aa, 6.1, 0.12, 214),
            inference: series(0x31ab, 3.1, 0.1, 214),
            updated: "2026-11-08T05:51:12Z".into(),
        }],
    }
}
