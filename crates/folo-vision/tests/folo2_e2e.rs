//! End-to-end `folo2-*` (FOLO v2-Iso) parity tests across `.folo` IO, `Trainer`, EMA, bit-exact checkpoint
//! save/resume, opt-in `iou_head` supervision + fusion, `auto_batch` VRAM probing, `Evaluator` with per-image
//! error diagnosis, and `Predictor` (single, batch, SAHI tiled, TTA, and GPU).

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use folo_api::{RunEvent, RunStatus};
use folo_core::{DType, Device};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Dataset, Image, Split};
use folo_gpu::{MemoryBudget, MemoryLimits, MemoryUsage};
use folo_nn::Module;
use folo_nn::io::{BestState, CheckpointParts, FileKind, FloatDType, ModelConfig, ModelFile, RngState};
use folo_nn::optim::OptimKind;
use folo_test_support::TestDir;
use folo_vision::metrics::EvalConfig;
use folo_vision::model::HeadMode;
use folo_vision::tile::TileConfig;
use folo_vision::train::{
    AutoBatchRequest, DEFAULT_AUTO_BATCH_UTILIZATION, EpochHook, OptimizerChoice, ResumeOverrides,
    TrainConfig, TrainData, TrainOptions, Trainer, auto_batch_plan, estimate_training_footprint,
};
use folo_vision::tta::{PredictOptions, TtaConfig};
use folo_vision::val::{ValConfig, validate_detector, validate_v2};
use folo_vision::{Detector, FoloV2Config, FoloV2Model, Precision, PredictConfig, Predictor};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn shapes(images: usize, val: usize, imgsz: u32) -> SyntheticConfig {
    SyntheticConfig {
        images: images + val,
        classes: vec!["circle".into(), "square".into()],
        imgsz,
        seed: 7,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: val as f32 / (images + val) as f32,
        ..SyntheticConfig::default()
    }
}

fn data(cfg: &SyntheticConfig) -> TrainData {
    let train = SyntheticDataset::new(cfg.clone(), Split::Train).unwrap();
    let val = SyntheticDataset::new(cfg.clone(), Split::Val).unwrap();
    TrainData { train: Arc::new(train), val: Arc::new(val), classes: cfg.classes.clone() }
}

fn folo2_config(epochs: u32, head: HeadMode) -> TrainConfig {
    TrainConfig {
        arch: "folo2-n".into(),
        head,
        device: "cpu".into(),
        epochs,
        batch: 2,
        nbs: 4,
        imgsz: 64,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 1.0,
        workers: 1,
        augment: AugmentConfig { close_mosaic: 1, ..AugmentConfig::default() },
        ..TrainConfig::default()
    }
}

type Log = Arc<Mutex<Vec<RunEvent>>>;

fn options(dir: Option<&Path>) -> (TrainOptions, Log) {
    let log: Log = Arc::default();
    let sink = log.clone();
    let opts = TrainOptions {
        run_dir: dir.map(Path::to_path_buf),
        events: Some(Box::new(move |e| sink.lock().unwrap().push(e.clone()))),
        step_interval: std::time::Duration::ZERO,
        ..TrainOptions::default()
    };
    (opts, log)
}

struct StopAfter(u32);

impl EpochHook for StopAfter {
    fn pauses(&mut self, epoch: u32) -> bool {
        epoch == self.0
    }

    fn pause(&mut self, _: u32, stop: &std::sync::atomic::AtomicBool) {
        stop.store(true, Ordering::Relaxed);
    }
}

fn raw_tensors(path: &Path) -> Vec<(String, Vec<u8>)> {
    let f = ModelFile::open(path).unwrap();
    let mut names: Vec<String> = f.tensor_infos().iter().map(|i| i.name.clone()).collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let t = f.tensor(&n).unwrap();
            let bytes = match t.dtype() {
                DType::I64 => t.to_vec::<i64>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect(),
                _ => t.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            (n, bytes)
        })
        .collect()
}

fn base_auto_batch_req(arch: &str) -> AutoBatchRequest {
    AutoBatchRequest {
        arch: arch.into(),
        imgsz: 640,
        num_classes: 80,
        head: HeadMode::O2m,
        amp: false,
        ema: true,
        optimizer: OptimKind::MuSgd,
        requested_batch: 16,
        nominal_batch: 64,
        target_utilization: DEFAULT_AUTO_BATCH_UTILIZATION,
        max_batch: None,
        val_batch: 0,
        teacher_arch: None,
        teacher_head: HeadMode::O2m,
        teacher_params: None,
    }
}

#[test]
fn folo2_trainer_ema_checkpoint_and_bit_exact_resume() {
    let ds = shapes(5, 2, 64);
    let dir_a = TestDir::new("folo2-e2e-cont");
    let dir_b = TestDir::new("folo2-e2e-resume");

    // Continuous 2-epoch run.
    let (opts_a, log_a) = options(Some(&dir_a));
    let mut a = Trainer::new(folo2_config(2, HeadMode::O2m), data(&ds), opts_a).unwrap();
    assert!(a.detector().is_v2());
    assert!(a.model_v2().is_some());
    assert!(a.ema_model_v2().is_some());
    let sum_a = a.run().unwrap();
    assert_eq!(sum_a.status, RunStatus::Completed);
    assert_eq!(sum_a.epochs_completed, 2);

    let last_a = dir_a.join("weights/last.folo");
    let best_a = dir_a.join("weights/best.folo");
    assert!(last_a.exists() && best_a.exists());
    let mf = ModelFile::open(&last_a).unwrap();
    assert_eq!(mf.metadata().kind, FileKind::Checkpoint);
    assert_eq!(mf.metadata().arch, "folo2-n");
    assert!(mf.metadata().config.is_v2());
    assert_eq!(mf.metadata().classes, ds.classes);
    assert_eq!(mf.metadata().imgsz, 64);
    assert!(mf.tensor_infos().iter().any(|i| i.name.starts_with("ema.")));

    // Split run: pause after epoch 1, resume from `weights/last.folo`, finish epoch 2.
    let (mut opts_b1, _) = options(Some(&dir_b));
    opts_b1.after_epoch = Some(Box::new(StopAfter(1)));
    let mut b1 = Trainer::new(folo2_config(2, HeadMode::O2m), data(&ds), opts_b1).unwrap();
    let sum_b1 = b1.run().unwrap();
    assert_eq!(sum_b1.status, RunStatus::Stopped);
    assert_eq!(sum_b1.epochs_completed, 1);

    let (opts_b2, log_b2) = options(Some(&dir_b));
    let mut b2 =
        Trainer::resume(dir_b.join("weights/last.folo"), data(&ds), opts_b2, ResumeOverrides::default())
            .unwrap();
    assert!(b2.detector().is_v2());
    let sum_b2 = b2.run().unwrap();
    assert_eq!(sum_b2.status, RunStatus::Completed);
    assert_eq!(sum_b2.epochs_completed, 2);
    assert_eq!(a.schedule().step(), b2.schedule().step());

    // Every parameter, BN buffer, EMA tensor, optimizer buffer, and scheduler counter matches bit-for-bit.
    let ta = raw_tensors(&last_a);
    let tb = raw_tensors(&dir_b.join("weights/last.folo"));
    assert_eq!(ta.len(), tb.len(), "checkpoint tensor count mismatch");
    let mut diffs = Vec::new();
    for ((na, ba), (nb, bb)) in ta.iter().zip(tb.iter()) {
        if na != nb || ba != bb {
            diffs.push(format!("{na} vs {nb} (len {} vs {})", ba.len(), bb.len()));
        }
    }
    assert!(
        diffs.is_empty(),
        "resume diverged on {} tensors; first 10: {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(10)]
    );

    // Epoch 2 losses and val metrics match.
    let ep_a: Vec<_> = log_a
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Epoch(x) = e { Some(x.clone()) } else { None })
        .collect();
    let ep_b2: Vec<_> = log_b2
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Epoch(x) = e { Some(x.clone()) } else { None })
        .collect();
    assert_eq!(ep_a.len(), 2);
    assert_eq!(ep_b2.len(), 1);
    assert_eq!(ep_a[1].train_loss.total.to_bits(), ep_b2[0].train_loss.total.to_bits());
    assert_eq!(ep_a[1].val.map50_95.to_bits(), ep_b2[0].val.map50_95.to_bits());
}

#[test]
fn folo2_o2o_training_export_transfer_and_predictor_sahi_tta() {
    let ds_cfg = shapes(4, 2, 64);
    let dir = TestDir::new("folo2-o2o-e2e");

    // Train an o2o folo2-n model for 2 epochs (with MuSGD so v2 `.cls.` LR boost & ortho are exercised).
    let cfg = TrainConfig {
        optimizer: OptimizerChoice::Musgd,
        lr0: 0.005,
        warmup_epochs: 0.0,
        augment: AugmentConfig::none(),
        ..folo2_config(2, HeadMode::O2o)
    };
    let (opts, _) = options(Some(&dir));
    let mut trainer = Trainer::new(cfg, data(&ds_cfg), opts).unwrap();
    let sum = trainer.run().unwrap();
    assert_eq!(sum.status, RunStatus::Completed);
    assert_eq!(sum.epochs_completed, 2);

    // Export fused F32 and F16 weights from the trained FoloV2Model.
    let trained_v2 = trainer.model_v2().unwrap();
    let exp_f32 = dir.join("export_f32.folo");
    let exp_f16 = dir.join("export_f16.folo");
    trained_v2.export(&exp_f32, FloatDType::F32).unwrap();
    trained_v2.export(&exp_f16, FloatDType::F16).unwrap();

    let mf32 = ModelFile::open(&exp_f32).unwrap();
    assert!(mf32.metadata().fused);
    assert_eq!(mf32.metadata().arch, "folo2-n");
    let mf16 = ModelFile::open(&exp_f16).unwrap();
    assert!(mf16.metadata().fused);
    assert_eq!(mf16.metadata().dtype, FloatDType::F16);

    // Transfer load from the unfused checkpoint into a different-class-count o2o model:
    // class outputs (`head.cls.*.out.*`, `head.o2o.*.cls.out.*`, `head.o2o.*.suppress.alpha_raw`) are skipped.
    let ckpt = ModelFile::open(dir.join("weights/last.folo")).unwrap();
    let mut target_cfg = FoloV2Config::new("n", 5).unwrap();
    target_cfg.head = HeadMode::O2o;
    let target_v2 = FoloV2Model::new(target_cfg).unwrap();
    let transfer_report = target_v2.transfer_from(&ckpt).unwrap();
    assert!(!transfer_report.loaded.is_empty());
    for level in ["p3", "p4", "p5"] {
        assert!(
            transfer_report.missing.contains(&format!("head.cls.{level}.out.weight")),
            "expected head.cls.{level}.out.weight skipped on class mismatch"
        );
        assert!(
            transfer_report.missing.contains(&format!("head.o2o.{level}.cls.out.weight")),
            "expected head.o2o.{level}.cls.out.weight skipped on class mismatch"
        );
        assert!(
            transfer_report.missing.contains(&format!("head.o2o.{level}.suppress.alpha_raw")),
            "expected head.o2o.{level}.suppress.alpha_raw skipped on class mismatch"
        );
    }

    // Transfer from an o2m file into an o2o model copies `head.cls.*` -> `head.o2o.*.cls.*`.
    let o2m_path = dir.join("o2m_src.folo");
    let mut o2m_src = FoloV2Model::new(FoloV2Config::new("n", 2).unwrap()).unwrap();
    o2m_src.set_classes(ds_cfg.classes.clone()).unwrap();
    o2m_src.save(&o2m_path).unwrap();
    let mut o2o_dst_cfg = FoloV2Config::new("n", 2).unwrap();
    o2o_dst_cfg.head = HeadMode::O2o;
    let mut o2o_dst = FoloV2Model::new(o2o_dst_cfg).unwrap();
    o2o_dst.set_classes(ds_cfg.classes.clone()).unwrap();
    let rep_o2m_to_o2o = o2o_dst.transfer_from(&ModelFile::open(&o2m_path).unwrap()).unwrap();
    assert!(
        rep_o2m_to_o2o.loaded.iter().any(|n| n == "head.o2o.p3.cls.out.weight"),
        "o2m -> o2o transfer should initialize head.o2o.*.cls.* from head.cls.*"
    );

    // Predictor::open auto-dispatches folo2-n from the exported F32 file and supports single, batch, SAHI, TTA.
    let pcfg = PredictConfig { imgsz: 64, ..PredictConfig::default() };
    let predictor = Predictor::open(&exp_f32, pcfg.clone()).unwrap();
    assert!(predictor.detector().is_v2());
    assert_eq!(predictor.arch(), "folo2-n");

    let val_ds = SyntheticDataset::new(ds_cfg.clone(), Split::Val).unwrap();
    let img0 = val_ds.get(0).unwrap().image;
    let img1 = val_ds.get(1).unwrap().image;

    let r0 = predictor.predict(&img0).unwrap();
    let r_batch = predictor.predict_batch(&[img0.clone(), img1.clone()]).unwrap();
    assert_eq!(r_batch.len(), 2);
    assert_eq!(r0.detections.len(), r_batch[0].detections.len());

    // SAHI tiled inference on a larger image (160x128 with 64x64 tiles).
    let big = Image::new(160, 128, [128, 128, 128]);
    let tile_cfg = TileConfig {
        tile_width: 64,
        tile_height: 64,
        overlap_x: 0.25,
        overlap_y: 0.25,
        ..TileConfig::default()
    };
    let tiled = predictor.predict_tiled(&big, &tile_cfg).unwrap();
    assert!(tiled.timing.total_ms >= 0.0);

    // TTA (hflip + multi-scale [1.0, 1.5] -> [64, 96]) single, batch, and via PredictOptions.
    let tta_cfg = TtaConfig { scales: vec![1.0, 1.5], hflip: true, ..TtaConfig::default() };
    let tta_single = predictor.predict_tta(&img0, &tta_cfg).unwrap();
    let tta_batch = predictor.predict_batch_tta(&[img0.clone(), img1.clone()], &tta_cfg).unwrap();
    assert_eq!(tta_batch.len(), 2);
    assert_eq!(tta_single.detections.len(), tta_batch[0].detections.len());

    let opts_res =
        predictor.predict_with_options(&img0, &PredictOptions { tta: Some(tta_cfg), tile: None }).unwrap();
    assert_eq!(opts_res.detections.len(), tta_single.detections.len());

    // Also test o2m evaluation mode override via Predictor::new_v2 on the same o2o model.
    let loaded_v2 = FoloV2Model::open(&exp_f32).unwrap();
    let pred_o2m = Predictor::new_v2(loaded_v2, PredictConfig { head: Some(HeadMode::O2m), ..pcfg }).unwrap();
    let _ = pred_o2m.predict(&img0).unwrap();
}

#[test]
fn folo2_iou_head_supervision_validation_diagnostics_and_prediction() {
    let ds_cfg = shapes(4, 3, 64);
    let dir = TestDir::new("folo2-iou-diag");

    // Build an unfused folo2-n checkpoint with `iou_head = true` by running 1-epoch resume over a seed ckpt,
    // first creating a base checkpoint via Trainer, then re-saving with `iou_head = true`? Even cleaner:
    // test `FoloV2Model` with `iou_head = true` directly through `save`/`open`/`validate_v2`/`Predictor::new_v2`,
    // AND build a real checkpoint with `iou_head = true` to verify `Trainer::resume` preserves `iou_head`
    // and `iou_quality_loss` updates `head.box_.p3.iou_out.weight`!
    let mut v2_cfg = FoloV2Config::new("n", 2).unwrap();
    v2_cfg.iou_head = true;
    let mut v2 = FoloV2Model::new(v2_cfg).unwrap();
    v2.set_classes(ds_cfg.classes.clone()).unwrap();
    v2.set_imgsz(64);
    let iou_w_before = v2.head.levels[0].box_branch.iou_out.as_ref().unwrap().weight.to_vec::<f32>().unwrap();

    // Validate with per-image error diagnosis enabled (`EvalConfig::with_diagnostics()`).
    let val_ds: Arc<dyn Dataset> = Arc::new(SyntheticDataset::new(ds_cfg.clone(), Split::Val).unwrap());
    let v2_fused = v2.fused_eval_clone().unwrap();
    let vcfg = ValConfig { eval: EvalConfig::with_diagnostics(), workers: 1, ..ValConfig::new(64, 2) };
    let val_res = validate_v2(&v2_fused, val_ds.clone(), &vcfg).unwrap();
    assert_eq!(val_res.report.image_diagnostics.len(), val_ds.len());
    for diag in &val_res.report.image_diagnostics {
        assert_eq!(diag.num_truths, diag.tp + diag.fn_count);
        assert_eq!(diag.num_preds, diag.tp + diag.fp);
        assert_eq!(
            diag.fp,
            diag.localization_errors + diag.class_confusions + diag.duplicate_errors + diag.background_fp
        );
    }
    let worst = val_res.report.worst_images(2);
    assert_eq!(worst.len(), 2);

    // `validate_detector` matches `validate_v2` identically.
    let det_res = validate_detector(&Detector::from(v2_fused), val_ds.clone(), &vcfg).unwrap();
    assert_eq!(val_res.report.map50_95.to_bits(), det_res.report.map50_95.to_bits());
    assert_eq!(val_res.report.image_diagnostics.len(), det_res.report.image_diagnostics.len());

    // Save/open round-trip preserves `iou_head = true` and Predictor runs `iou_raw` score fusion.
    let iou_path = dir.join("folo2_iou.folo");
    v2.save(&iou_path).unwrap();
    let reopened = FoloV2Model::open(&iou_path).unwrap();
    assert!(reopened.config().iou_head);
    let pred = Predictor::open(&iou_path, PredictConfig { imgsz: 64, ..PredictConfig::default() }).unwrap();
    let img0 = val_ds.get(0).unwrap().image;
    let _ = pred.predict(&img0).unwrap();

    // Verify `Trainer::resume` preserves `iou_head = true` from `folo.config` and supervises `iou_out` via
    // `iou_quality_loss` so `head.levels.0.box_branch.iou_out.weight` receives non-zero gradient updates.
    let ckpt_path = dir.join("iou_ckpt.folo");
    let train_cfg = TrainConfig {
        epochs: 1,
        batch: 2,
        nbs: 2,
        warmup_epochs: 0.0,
        augment: AugmentConfig::none(),
        ..folo2_config(1, HeadMode::O2m)
    };
    let train_data = data(&ds_cfg);
    let mc = v2.config().to_model_config().unwrap();
    let meta = folo_nn::io::Metadata::new(FileKind::Checkpoint, mc, ds_cfg.classes.clone(), 64).unwrap();
    let mut train_json = serde_json::to_value(&train_cfg).unwrap();
    train_json
        .as_object_mut()
        .unwrap()
        .insert("train_images".into(), serde_json::json!(train_data.train.len()));
    let groups = folo_nn::optim::StandardGroups::from_module(&v2).into_groups(0.002, 0.9, 0.0005);
    let opt = folo_nn::optim::AdamW::new(groups).unwrap();
    let sched = folo_nn::LrSchedule::new(folo_nn::DecayKind::Linear, 1, 2, 0.01, 0.0, 0.1, 0.8).unwrap();
    let mut shadow = FoloV2Model::new(v2.config().clone()).unwrap();
    shadow.set_classes(ds_cfg.classes.clone()).unwrap();
    shadow.set_imgsz(64);
    let ema = folo_nn::ModelEma::with_params(shadow, &v2, 0.9999, 2000.0).unwrap();
    let rng_state = RngState { seed: 0, streams: Default::default() };
    let parts = CheckpointParts {
        model: &v2,
        ema: Some(&ema),
        optimizer: &opt,
        schedule: &sched,
        rng: &rng_state,
        epoch: 0,
        best: BestState::default(),
        train_config: &train_json,
    };
    folo_nn::io::save_checkpoint(&ckpt_path, &meta, &parts).unwrap();

    let (opts, _) = options(Some(&dir));
    let mut resumed = Trainer::resume(&ckpt_path, train_data, opts, ResumeOverrides::default()).unwrap();
    assert!(resumed.model_v2().unwrap().config().iou_head, "resume must preserve iou_head=true");
    let sum = resumed.run().unwrap();
    assert_eq!(sum.status, RunStatus::Completed);
    let iou_w_after = resumed.model_v2().unwrap().head.levels[0]
        .box_branch
        .iou_out
        .as_ref()
        .unwrap()
        .weight
        .to_vec::<f32>()
        .unwrap();
    assert_ne!(iou_w_before, iou_w_after, "iou_quality_loss must supervise and update iou_out weights");
}

#[test]
fn auto_batch_folo2_param_count_and_plan_parity_across_all_scales() {
    for scale in ["n", "s", "m", "l", "x"] {
        for (head, nc) in [(HeadMode::O2m, 80), (HeadMode::O2o, 4)] {
            let mut mconf = ModelConfig::new_v2(scale, nc).unwrap();
            mconf.head = head.as_str().into();
            let v2_cfg = FoloV2Config::from_model_config(&mconf).unwrap();
            let model = FoloV2Model::new(v2_cfg).unwrap();
            let req = AutoBatchRequest {
                arch: format!("folo2-{scale}"),
                num_classes: nc,
                head,
                ..base_auto_batch_req(&format!("folo2-{scale}"))
            };
            let fp = estimate_training_footprint(&req).unwrap();
            assert_eq!(
                fp.model_params,
                model.num_parameters(),
                "auto_batch param count mismatch for folo2-{scale} head={head:?} nc={nc}"
            );
            assert!(fp.fixed_bytes > (fp.model_params as u64) * 4);
            assert!(fp.per_sample_bytes >= fp.max_single_tensor_per_sample);
            assert!(fp.max_single_tensor_per_sample > 0);
        }
    }

    // Tight vs generous VRAM budget planning works for folo2-s.
    let req = AutoBatchRequest { requested_batch: 32, ..base_auto_batch_req("folo2-s") };
    let tight =
        MemoryBudget::new(MemoryLimits::new(Some(6 * GIB), 512 * MIB, 512 * MIB), MemoryUsage::default());
    let generous =
        MemoryBudget::new(MemoryLimits::new(Some(24 * GIB), 2 * GIB, 2 * GIB), MemoryUsage::default());
    let tight_plan = auto_batch_plan(&req, &tight).unwrap();
    let gen_plan = auto_batch_plan(&req, &generous).unwrap();
    assert!(tight_plan.fits && gen_plan.fits);
    assert!(tight_plan.chosen_batch < gen_plan.chosen_batch);

    // Trainer::new accepts `arch: "folo2-n"` with `auto_batch: true` (CPU graceful fallback path).
    let ds = shapes(4, 2, 64);
    let cfg = TrainConfig { auto_batch: true, ..folo2_config(1, HeadMode::O2m) };
    let (opts, _) = options(None);
    let trainer = Trainer::new(cfg, data(&ds), opts).unwrap();
    assert!(trainer.detector().is_v2());
}

#[test]
fn folo2_gpu_amp_train_and_f16_predict_smoke() {
    folo_gpu::request_tensor_cores();
    let Ok(dev) = folo_gpu::init() else {
        eprintln!("skipping folo2_gpu_amp_train_and_f16_predict_smoke: no GPU adapter");
        return;
    };
    let ds_cfg = shapes(4, 2, 64);
    let dir = TestDir::new("folo2-gpu-smoke");
    let use_amp = folo_vision::train::amp_support(dev).is_ok();
    let cfg = TrainConfig {
        device: "gpu".into(),
        amp: use_amp,
        epochs: 1,
        batch: 2,
        nbs: 2,
        warmup_epochs: 0.0,
        augment: AugmentConfig::none(),
        ..folo2_config(1, HeadMode::O2m)
    };
    let (opts, _) = options(Some(&dir));
    let mut trainer = Trainer::new(cfg, data(&ds_cfg), opts).unwrap();
    let sum = trainer.run().unwrap();
    assert_eq!(sum.status, RunStatus::Completed);

    let last = dir.join("weights/last.folo");
    let prec = if folo_core::backend::computes_f16(dev) { Precision::F16 } else { Precision::F32 };
    let pred = Predictor::open_on(&last, PredictConfig { imgsz: 64, ..PredictConfig::default() }, dev, prec)
        .unwrap();
    assert!(pred.detector().is_v2());
    assert_eq!(pred.detector().device(), dev);
    let val_ds = SyntheticDataset::new(ds_cfg, Split::Val).unwrap();
    let img = val_ds.get(0).unwrap().image;
    let _ = pred.predict(&img).unwrap();
    assert_eq!(Device::Gpu(0), dev);
}
