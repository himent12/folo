use super::*;
use folo_api::{Confidence, ThroughputSource};
use folo_test_support::TestDir;

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
    assert_eq!((e.phases[0].phase, e.phases[0].epochs, e.phases[0].epoch_s), (PhaseKind::Mosaic, 90, 42.0));
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
    assert_eq!((e.est_peak_vram_bytes, e.vram_source), (Some(want as u64), Some(ThroughputSource::Measured)));
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
    let s640: Vec<_> = t.training.iter().filter(|r| r.arch == Arch::S && r.imgsz == 640 && !r.amp).collect();
    assert_eq!(s640.len(), 2, "{s640:?}");
    assert!(s640.iter().any(|r| r.source == ThroughputSource::Reference && r.measured_on.contains("4080")));
    assert!(s640.iter().any(|r| r.source == ThroughputSource::Measured && r.measured_on.contains("7900")));
    let m640: Vec<_> = t.training.iter().filter(|r| r.arch == Arch::M && r.imgsz == 640 && !r.amp).collect();
    assert_eq!((m640.len(), m640[0].source), (1, ThroughputSource::Reference));
    assert!(t.inference.iter().all(|r| r.source == ThroughputSource::Reference));
}

/// A hand-edited or older `throughput.json` can count more probes than samples: the estimate's note must not
/// underflow (a panic in debug builds).
#[test]
fn a_row_with_more_probes_than_samples_still_estimates() {
    let dir = TestDir::new("throughput-probes");
    let path = dir.join(FILE);
    let mut file = TableFile::default();
    let mut rec = record(Arch::N, 640, false, &[400.0; 3], &[], &[]);
    rec.probes = 5;
    file.training.push(rec);
    std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
    let t = Throughput::open(path);
    assert_eq!(t.file().training[0].probes, 5);
    let e = t.estimate(&input(Arch::N, 640, false));
    assert_eq!(e.source, ThroughputSource::Measured);
    assert!(
        e.basis.note.contains("3 benchmark probes") && !e.basis.note.contains("epochs"),
        "{}",
        e.basis.note
    );
}

#[test]
fn the_table_round_trips_through_its_file() {
    let dir = TestDir::new("throughput");
    let path = dir.join(FILE);
    let t = Throughput::open(path.clone());
    let key = TrainKey { device: gpu(), arch: Arch::S, imgsz: 640, batch: 8, amp: true };
    for (i, phase) in [PhaseKind::Mosaic, PhaseKind::Mosaic, PhaseKind::CloseMosaic].into_iter().enumerate() {
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
