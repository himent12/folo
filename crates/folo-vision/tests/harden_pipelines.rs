//! End-to-end pipeline hardening tests for Sprint 11 (training resume with M4 options, ONNX export across
//! checkpoints & head modes, int8 calibration/inference edge cases, zero-GT & single-class datasets,
//! extreme/tiny images, corrupted `.folo` files, and post-processing numerical stability).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use folo_api::RunStatus;
use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, BBox, Image, InMemoryDataset, Sample, Split};
use folo_nn::io::{FloatDType, ModelFile, Purpose};
use folo_nn::{EmaMode, GradScaler};
use folo_test_support::TestDir;
use folo_vision::cocoeval::CocoEval;
use folo_vision::export::onnx::{OnnxOptions, export_file};
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::loss::{BoxLossKind, LossConfig};
use folo_vision::model::quant::{self, Calibration, META_KEY, Method};
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::train::{EpochHook, OptimizerChoice, ResumeOverrides};
use folo_vision::val::{ValConfig, val_metrics, validate};
use folo_vision::{PredictConfig, Predictor, TrainConfig, TrainData, TrainOptions, Trainer};

fn tiny_shapes(images: usize, val: usize, nc: usize) -> SyntheticConfig {
    const NAMES: [&str; 4] = ["circle", "square", "triangle", "star"];
    let classes = NAMES[..nc].iter().map(|&s| s.to_string()).collect();
    SyntheticConfig {
        images: images + val,
        classes,
        imgsz: 64,
        seed: 17,
        min_objects: 1,
        max_objects: 2,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: val as f32 / (images + val) as f32,
        ..SyntheticConfig::default()
    }
}

fn train_data(cfg: &SyntheticConfig) -> TrainData {
    TrainData {
        train: Arc::new(SyntheticDataset::new(cfg.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(cfg.clone(), Split::Val).unwrap()),
        classes: cfg.classes.clone(),
    }
}

fn fused_model(scale: &str, nc: usize, head: HeadMode, seed: u64) -> FoloModel {
    let mut cfg = ModelConfig::new(scale, nc).unwrap();
    cfg.head = head.as_str().to_string();
    cfg.bn_momentum = 0.999;
    let mut m = FoloModel::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    m.set_classes((0..nc).map(|i| format!("cls{i}")).collect()).unwrap();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(seed + 7));
    {
        let _g = no_grad();
        m.forward(&x).unwrap();
    }
    m.fuse().unwrap();
    m
}

struct StopAfter(u32);

impl EpochHook for StopAfter {
    fn pauses(&mut self, epoch: u32) -> bool {
        epoch == self.0
    }

    fn pause(&mut self, _: u32, stop: &AtomicBool) {
        stop.store(true, Ordering::Relaxed);
    }
}

/// Checkpoint resume parity with M4 training options enabled simultaneously:
/// `MuSgd`, `freeze_backbone_epochs = 1`, `backbone_lr_scale = 0.25`, `EmaMode::Power`, and `wiou_v3`.
#[test]
fn train_resume_parity_with_m4_options_musgd_freeze_backbone_power_ema_wiou() {
    let cfg_data = tiny_shapes(4, 2, 2);
    let data = train_data(&cfg_data);
    let cfg = TrainConfig {
        epochs: 2,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        optimizer: OptimizerChoice::Musgd,
        lr0: 0.005,
        warmup_epochs: 0.5,
        workers: 0,
        augment: AugmentConfig::none(),
        ema_mode: EmaMode::Power,
        backbone_lr_scale: 0.25,
        freeze_backbone_epochs: 1,
        loss: LossConfig { box_loss: BoxLossKind::WiouV3, ..LossConfig::default() },
        ..TrainConfig::default()
    };

    let dir_full = TestDir::new("harden-m4-full");
    let opts_full = TrainOptions { run_dir: Some(dir_full.path().to_path_buf()), ..TrainOptions::default() };
    let full = Trainer::new(cfg.clone(), data.clone(), opts_full).unwrap().run().unwrap();
    assert_eq!(full.epochs_completed, 2);

    // Split run: stop deterministically at the epoch 1 boundary via `after_epoch`, then resume from `last.folo`.
    let dir_split = TestDir::new("harden-m4-split");
    let opts1 = TrainOptions {
        run_dir: Some(dir_split.path().to_path_buf()),
        after_epoch: Some(Box::new(StopAfter(1))),
        ..TrainOptions::default()
    };
    let part1 = Trainer::new(cfg, data.clone(), opts1).unwrap().run().unwrap();
    assert_eq!((part1.status, part1.epochs_completed), (RunStatus::Stopped, 1));

    let last = dir_split.path().join("weights/last.folo");
    assert!(last.is_file());
    let opts2 = TrainOptions { run_dir: Some(dir_split.path().to_path_buf()), ..TrainOptions::default() };
    let part2 = Trainer::resume(&last, data, opts2, ResumeOverrides::default()).unwrap().run().unwrap();
    assert_eq!((part2.status, part2.epochs_completed), (RunStatus::Completed, 2));

    // Final losses, epoch/step counters, and model weights must match bit-for-bit across the resume boundary.
    assert_eq!(full.epochs[1].train_loss.total.to_bits(), part2.epochs[0].train_loss.total.to_bits());
    assert_eq!(full.epochs[1].train_loss.box_.to_bits(), part2.epochs[0].train_loss.box_.to_bits());
    assert_eq!(full.epochs[1].train_loss.cls.to_bits(), part2.epochs[0].train_loss.cls.to_bits());
    let mf_full = ModelFile::open(dir_full.path().join("weights/last.folo")).unwrap();
    let mf_split = ModelFile::open(&last).unwrap();
    let ck_full = mf_full.checkpoint().unwrap();
    let ck_split = mf_split.checkpoint().unwrap();
    assert_eq!(ck_full.epoch, ck_split.epoch);
    assert_eq!(ck_full.sched.step, ck_split.sched.step);
    let (m_full, _) = FoloModel::from_file(&mf_full, Purpose::Inference).unwrap();
    let (m_split, _) = FoloModel::from_file(&mf_split, Purpose::Inference).unwrap();
    for (a, b) in m_full.state().iter().zip(m_split.state().iter()) {
        assert_eq!(a.0, b.0);
        assert_eq!(
            a.1.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.1.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "weight mismatch after resume on {}",
            a.0
        );
    }
}

/// ONNX export graph validity, shapes, and metadata across sizes (`n`, `s`), `o2m`/`o2o`,
/// fused / f16 / quantized `.folo` checkpoints, and dynamic/static/half export options.
#[test]
fn onnx_export_file_handles_sizes_o2o_f16_and_quantized_checkpoints() {
    let dir = TestDir::new("harden-onnx");

    // 1. Export fused f32 and fused f16 checkpoints for `folo-n` (o2o) and `folo-s` (o2m).
    for (scale, head, nc) in [("n", HeadMode::O2o, 3), ("s", HeadMode::O2m, 1)] {
        let m = fused_model(scale, nc, head, 41);
        let p_f32 = dir.path().join(format!("{scale}-f32.folo"));
        let p_f16 = dir.path().join(format!("{scale}-f16.folo"));
        m.export(&p_f32, FloatDType::F32).unwrap();
        m.export(&p_f16, FloatDType::F16).unwrap();

        for (tag, p, half, dynamic) in
            [("f32-static", &p_f32, false, false), ("f16-dynamic", &p_f16, true, true)]
        {
            let mf = ModelFile::open(p).unwrap();
            let opts = OnnxOptions { half, dynamic, decode: true, max_det: 100, ..OnnxOptions::new(64) };
            let exp = export_file(&mf, &opts).unwrap();
            assert!(!exp.bytes.is_empty(), "{scale}/{tag} produced empty protobuf");
            assert_eq!(exp.inputs.len(), 1);
            assert_eq!(exp.outputs.len(), 1);
            assert_eq!(exp.outputs[0].name, "detections");
            assert_eq!(exp.outputs[0].shape.len(), 3);
            assert_eq!(
                exp.metadata.get("folo.onnx.outputs").map(String::as_str),
                Some(if head == HeadMode::O2o { "end2end" } else { "decoded" })
            );
            if head == HeadMode::O2o {
                // O2o decoded output is always f32 [B, K, 6] so class ids remain exact.
                assert_eq!(exp.outputs[0].dtype, "float");
                assert_eq!(format!("{:?}", exp.outputs[0].shape[2]), "Value(6)");
            } else {
                assert_eq!(format!("{:?}", exp.outputs[0].shape[2]), format!("Value({})", 4 + nc));
            }
            let out_path = dir.path().join(format!("{scale}-{tag}.onnx"));
            exp.write_to(&out_path).unwrap();
            assert_eq!(std::fs::read(&out_path).unwrap(), exp.bytes);
        }
    }

    // 2. Quantized `.folo` checkpoint (carrying `folo.quant` metadata) exports cleanly via `export_file`.
    let m = fused_model("n", 2, HeadMode::O2m, 99);
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(100));
    let cal = quant::calibrate(&m, || std::iter::once(Ok(x.clone())), Method::Mse).unwrap();
    let mut meta = m.metadata().unwrap();
    cal.store(&mut meta).unwrap();
    let p_q = dir.path().join("quantized.folo");
    folo_nn::io::save_weights(&p_q, &meta, &m).unwrap();
    let mf_q = ModelFile::open(&p_q).unwrap();
    assert!(Calibration::from_metadata(mf_q.metadata()).unwrap().is_some());
    let exp_q = export_file(&mf_q, &OnnxOptions { decode: true, ..OnnxOptions::new(64) }).unwrap();
    assert!(!exp_q.bytes.is_empty());
    assert_eq!(exp_q.outputs[0].name, "detections");
}

/// Zero-GT validation dataset (non-empty split with zero ground-truth boxes) runs cleanly through
/// `validate`, `val_metrics`, `CocoEval`, and `Trainer` without producing any NaN/Inf metrics.
#[test]
fn zero_gt_dataset_produces_clean_finite_val_and_train_metrics() {
    let classes = vec!["obj".into(), "other".into()];
    let empty_gt_val = Arc::new(InMemoryDataset {
        samples: vec![
            Sample { image: Image::new(64, 64, [32, 64, 96]), boxes: vec![] },
            Sample { image: Image::new(64, 48, [120, 80, 40]), boxes: vec![] },
        ],
        classes: classes.clone(),
        split: Split::Val,
    });
    let m = fused_model("n", 2, HeadMode::O2m, 55);
    let vc = ValConfig { workers: 0, keep_detections: true, ..ValConfig::new(64, 2) };
    let res = validate(&m, empty_gt_val.clone(), &vc).unwrap();
    let vm = val_metrics(&res.report);
    assert!(vm.map50.is_finite() && vm.map50 == 0.0);
    assert!(vm.map50_95.is_finite() && vm.map50_95 == 0.0);
    assert!(vm.precision.is_finite() && vm.recall.is_finite());
    assert_eq!(vm.best_f1_conf, None);

    let mut ce = CocoEval::new(classes.len());
    for (idx, dets) in &res.detections {
        ce.add_image(*idx as u64, dets, &[]);
    }
    let summary = ce.compute();
    assert!(summary.ap.is_finite());
    assert!(summary.ap50.is_finite());

    // Trainer running an epoch against a zero-GT val split completes cleanly with valid checkpoints.
    let train_ds = Arc::new(InMemoryDataset {
        samples: vec![
            Sample { image: Image::new(64, 64, [10, 20, 30]), boxes: vec![BBox::new(0, 0.5, 0.5, 0.3, 0.3)] },
            Sample {
                image: Image::new(64, 64, [40, 50, 60]),
                boxes: vec![BBox::new(1, 0.4, 0.6, 0.25, 0.25)],
            },
        ],
        classes: classes.clone(),
        split: Split::Train,
    });
    let dir = TestDir::new("harden-zero-gt-train");
    let cfg = TrainConfig {
        epochs: 1,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        optimizer: OptimizerChoice::Adamw,
        lr0: 1e-3,
        warmup_epochs: 0.0,
        workers: 0,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    let opts = TrainOptions { run_dir: Some(dir.path().to_path_buf()), ..TrainOptions::default() };
    let sum = Trainer::new(cfg, TrainData { train: train_ds, val: empty_gt_val, classes }, opts)
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(sum.epochs_completed, 1);
    assert!(sum.epochs[0].val.map50_95.is_finite());
    assert!(dir.path().join("weights/last.folo").is_file());
    assert!(dir.path().join("weights/best.folo").is_file());
}

/// Single-class (`nc = 1`) end-to-end across train, val, predict, int8 quantize, and ONNX export (`o2o`).
#[test]
fn single_class_model_works_across_train_val_predict_quantize_and_onnx() {
    let cfg_data = tiny_shapes(4, 2, 1);
    let data = train_data(&cfg_data);
    let dir = TestDir::new("harden-nc1");
    let cfg = TrainConfig {
        epochs: 1,
        batch: 2,
        nbs: 2,
        imgsz: 64,
        head: HeadMode::O2o,
        optimizer: OptimizerChoice::Adamw,
        lr0: 1e-3,
        warmup_epochs: 0.0,
        workers: 0,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    let opts = TrainOptions { run_dir: Some(dir.path().to_path_buf()), ..TrainOptions::default() };
    let sum = Trainer::new(cfg, data.clone(), opts).unwrap().run().unwrap();
    assert_eq!(sum.epochs_completed, 1);

    let last = dir.path().join("weights/last.folo");
    let mf = ModelFile::open(&last).unwrap();
    let (mut m, _) = FoloModel::from_file(&mf, Purpose::Inference).unwrap();
    m.fuse().unwrap();

    // ONNX export of single-class o2o model.
    let exp = export_file(&mf, &OnnxOptions { decode: true, ..OnnxOptions::new(64) }).unwrap();
    assert_eq!(format!("{:?}", exp.outputs[0].shape[2]), "Value(6)");

    // Int8 quantize + predict on single-class model.
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(77));
    let cal =
        quant::calibrate(&m, || std::iter::once(Ok(x.clone())), Method::Percentile { p: 1e-3 }).unwrap();
    m.set_int8(Some(cal)).unwrap();
    let pred =
        Predictor::new(m, PredictConfig { imgsz: 64, conf: 0.01, ..PredictConfig::default() }).unwrap();
    let out = pred.predict(&Image::new(48, 32, [100, 120, 140])).unwrap();
    for d in &out.detections {
        assert_eq!(d.class, 0);
        assert!(d.score.is_finite());
        assert!(
            d.bbox.x1.is_finite() && d.bbox.y1.is_finite() && d.bbox.x2.is_finite() && d.bbox.y2.is_finite()
        );
    }
}

/// Extreme tiny and non-square images (`1x1`, `8x8`, `1000x1`, `1x1000`, mixed batch) through `Predictor`
/// in both `f32` and `int8` (`u8_stem = true`), with `rect = true` and `rect = false`, plus `sebf` /
/// `dfl-cgr` / `dfl-repulse` determinism.
#[test]
fn predictor_handles_tiny_and_extreme_aspect_images_in_f32_int8_and_postproc_rules() {
    let imgs = vec![
        Image::new(1, 1, [200, 100, 50]),
        Image::new(8, 8, [10, 200, 30]),
        Image::new(1000, 1, [80, 90, 100]),
        Image::new(1, 1000, [50, 150, 250]),
    ];

    let mut m_f32 = fused_model("n", 2, HeadMode::O2m, 12);
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(13));
    let cal = quant::calibrate(&m_f32, || std::iter::once(Ok(x.clone())), Method::MinMax).unwrap();
    assert!(cal.u8_stem);

    for (use_int8, rect) in [(false, true), (false, false), (true, true), (true, false)] {
        let mut m = fused_model("n", 2, HeadMode::O2m, 12);
        if use_int8 {
            m.set_int8(Some(cal.clone())).unwrap();
        }
        let cfg = PredictConfig {
            imgsz: 64,
            conf: 0.05,
            rect,
            fusion: BoxFusion::Sebf,
            score: ScoreRule::DflCgr,
            ..PredictConfig::default()
        };
        let pred = Predictor::new(m, cfg).unwrap();
        for img in &imgs {
            let p1 = pred.predict(img).unwrap();
            let p2 = pred.predict(img).unwrap();
            assert_eq!(p1.detections, p2.detections, "non-deterministic sebf+dfl-cgr");
            assert_eq!((p1.width, p1.height), (img.width, img.height));
            for d in &p1.detections {
                assert!(d.score.is_finite() && (0.0..=1.0).contains(&d.score));
                assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0);
                assert!(d.bbox.x2 <= img.width as f32 + 1e-3 && d.bbox.y2 <= img.height as f32 + 1e-3);
            }
        }
        // Mixed-shape batch forces square canvas and runs cleanly.
        let batch_out = pred.predict_batch(&imgs).unwrap();
        assert_eq!(batch_out.len(), imgs.len());
    }

    // DflRepulse determinism on the same fused model.
    m_f32.set_int8(None).unwrap();
    let rep_cfg = PredictConfig {
        imgsz: 64,
        conf: 0.05,
        fusion: BoxFusion::Dfl,
        score: ScoreRule::DflRepulse,
        ..PredictConfig::default()
    };
    let pred_rep = Predictor::new(m_f32, rep_cfg).unwrap();
    let a = pred_rep.predict(&imgs[1]).unwrap();
    let b = pred_rep.predict(&imgs[1]).unwrap();
    assert_eq!(a.detections, b.detections, "non-deterministic dfl+dfl-repulse");
}

/// Corrupted/truncated `.folo` weight files and malformed/future `folo.quant` metadata fail cleanly.
#[test]
fn corrupted_and_truncated_folo_files_fail_cleanly() {
    let dir = TestDir::new("harden-corrupt");
    let m = fused_model("n", 2, HeadMode::O2m, 23);
    let good = dir.path().join("good.folo");
    m.save(&good).unwrap();
    let raw = std::fs::read(&good).unwrap();

    // 1. Empty file, truncated header (< 8 bytes), truncated JSON header, truncated tensor payload.
    for (name, slice) in [
        ("empty.folo", &raw[..0]),
        ("tiny.folo", &raw[..4]),
        ("trunc_hdr.folo", &raw[..24]),
        ("trunc_body.folo", &raw[..raw.len() - 16]),
    ] {
        let p = dir.path().join(name);
        std::fs::write(&p, slice).unwrap();
        assert!(ModelFile::open(&p).is_err(), "{name} should fail to open");
    }

    // 2. Corrupted JSON header bytes.
    let mut bad_json = raw.clone();
    bad_json[8] = b'!';
    let p_bad_json = dir.path().join("bad_json.folo");
    std::fs::write(&p_bad_json, &bad_json).unwrap();
    assert!(ModelFile::open(&p_bad_json).is_err());

    // 3. Unsupported future `folo.quant` schema version and invalid JSON in `folo.quant`.
    let mut meta_v99 = m.metadata().unwrap();
    meta_v99.extra.insert(
        META_KEY.into(),
        r#"{"version":99,"method":{"kind":"min_max"},"images":1,"ranges":{}}"#.into(),
    );
    assert!(Calibration::from_metadata(&meta_v99).is_err());

    let mut meta_bad = m.metadata().unwrap();
    meta_bad.extra.insert(META_KEY.into(), "{not json".into());
    assert!(Calibration::from_metadata(&meta_bad).is_err());

    // 4. Calibrator::observe refuses non-finite inputs and poisons finish().
    let mut cal = quant::calib::Calibrator::new();
    let nan_img = Tensor::full([1, 3, 64, 64], f32::NAN);
    assert!(cal.observe(&m, &nan_img).is_err());
    cal.next_pass();
    assert!(cal.finish(Method::MinMax).is_err());
}

/// GradScaler overflow handling skips the step, backs off scale cleanly, and stays usable/finite.
#[test]
fn grad_scaler_overflow_skips_and_backs_off_cleanly() {
    let mut scaler = GradScaler::new();
    let s0 = scaler.scale();
    scaler.update(true);
    assert_eq!(scaler.scale(), s0 * 0.5);
    assert_eq!(scaler.growth_tracker(), 0);
    assert_eq!(scaler.skipped_steps(), 1);
    assert_eq!(scaler.skip_streak(), 1);
    // Repeated overflows floor safely at MIN_SCALE without underflowing to zero or NaN.
    for _ in 0..200 {
        scaler.update(true);
    }
    assert!(scaler.scale().is_finite() && scaler.scale() > 0.0 && scaler.inv_scale().is_finite());
}
