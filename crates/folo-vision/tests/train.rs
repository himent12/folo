//! Trainer tests (docs/MODEL.md §10.4) on tiny synthetic sets: a micro run that learns, checkpoint round trips,
//! bit-exact resume, the schedule numbers, the non-finite guard, and the predictor on trained weights.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use folo_api::{RunEvent, RunStatus};
use folo_core::no_grad;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, InMemoryDataset, Split};
use folo_nn::io::{Metadata, ModelFile, Namespace};
use folo_nn::{Module, RunningStats};
use folo_vision::train::{OptimizerChoice, ResumeOverrides};
use folo_vision::{PredictConfig, Predictor, TrainConfig, TrainData, TrainOptions, Trainer};

fn shapes(images: usize, val: usize, imgsz: u32) -> SyntheticConfig {
    SyntheticConfig {
        images: images + val,
        classes: vec!["circle".into(), "square".into()],
        imgsz,
        seed: 3,
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

/// 16 train / 4 val images at 64 px, batch 4 with accumulation 2, AdamW without warmup or augmentation.
fn micro_config(epochs: u32) -> TrainConfig {
    TrainConfig {
        epochs,
        batch: 4,
        nbs: 8,
        imgsz: 64,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    }
}

/// The resume run: 18 train images in batches of 4 (5 batches: windows of 2, 2 and a short 1), warmup over
/// the first 3 of 6 optimizer steps, the default augmentation with mosaic closing for the second epoch.
fn resume_config() -> TrainConfig {
    TrainConfig {
        warmup_epochs: 1.0,
        augment: AugmentConfig { close_mosaic: 1, ..AugmentConfig::default() },
        ..micro_config(2)
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("folo-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

type Log = Arc<Mutex<Vec<RunEvent>>>;

/// A parameter handle and its clean values.
type Saved = (folo_core::Tensor, Vec<f32>);

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

fn epochs_of(log: &Log) -> Vec<folo_api::Epoch> {
    log.lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Epoch(x) = e { Some(x.clone()) } else { None })
        .collect()
}

/// Every tensor of a `.folo` file as raw bytes, by name.
fn tensors(path: &Path) -> Vec<(String, Vec<u8>)> {
    let f = ModelFile::open(path).unwrap();
    let mut names: Vec<String> = f.tensor_infos().iter().map(|i| i.name.clone()).collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let t = f.tensor(&n).unwrap();
            let bytes = match t.dtype() {
                folo_core::DType::I64 => {
                    t.to_vec::<i64>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                _ => t.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            (n, bytes)
        })
        .collect()
}

#[test]
fn micro_run_learns_checkpoints_and_resumes_bit_exactly() {
    let ds = shapes(18, 4, 64);
    let dir_a = temp_dir("a");
    let dir_b = temp_dir("b");

    // A: two uninterrupted epochs.
    let (opts, log_a) = options(Some(&dir_a));
    let mut a = Trainer::new(resume_config(), data(&ds), opts).unwrap();
    let c = a.schedule().contract();
    assert_eq!((a.batches_per_epoch(), a.accumulate(), c.steps_per_epoch, c.warmup_iters), (5, 2, 3, 3));
    let sum_a = a.run().unwrap();
    assert_eq!(sum_a.status, RunStatus::Completed);
    assert_eq!(sum_a.epochs_completed, 2);
    assert_eq!(a.schedule().step(), 6);

    let events = log_a.lock().unwrap().clone();
    assert!(matches!(events.first(), Some(RunEvent::Status(s)) if s.status == RunStatus::Running));
    assert!(matches!(events.last(), Some(RunEvent::Status(s)) if s.status == RunStatus::Completed));
    let steps: Vec<_> =
        events.iter().filter_map(|e| if let RunEvent::Step(s) = e { Some(s) } else { None }).collect();
    assert_eq!(steps.iter().map(|s| s.step).collect::<Vec<_>>(), (1..=10).collect::<Vec<u64>>());
    for s in &steps {
        assert!(s.loss.total.is_finite() && s.lr.is_finite() && s.eta_s.is_finite() && s.imgs_per_s > 0.0);
    }
    assert_eq!(steps[0].lr, 0.0, "warmup starts the decay group at 0");
    assert_eq!(steps[9].lr, (0.002f64 * 0.01) as f32, "the last step runs at lr0 · lrf");
    let epochs = epochs_of(&log_a);
    assert_eq!(epochs.len(), 2);
    assert!(epochs.iter().all(|e| e.train_loss.total.is_finite() && e.val.map50.is_finite()));

    // The run directory.
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir_a.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["optimizer"]["type"], "adamw");
    assert_eq!(cfg["accumulate"], 2);
    let lines = std::fs::read_to_string(dir_a.join("metrics.jsonl")).unwrap();
    let records: Vec<serde_json::Value> = lines.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(records.iter().filter(|r| r["event"] == "step").count(), 10);
    assert_eq!(records.iter().filter(|r| r["event"] == "epoch").count(), 2);
    let samples: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir_a.join("samples/epoch_002.json")).unwrap())
            .unwrap();
    assert_eq!(samples["samples"].as_array().unwrap().len(), 4);

    // Checkpoints: a complete FORMAT.md checkpoint at the epoch boundary.
    let last = dir_a.join("weights/last.folo");
    let file = ModelFile::open(&last).unwrap();
    let c = file.checkpoint().expect("last.folo is a checkpoint");
    assert_eq!((c.epoch, c.optim.step, c.sched.step, c.ema_updates), (2, 6, 6, Some(6)));
    assert!(file.has_ema());
    assert!(!file.names(Namespace::Optim).is_empty());
    assert_eq!(file.metadata().classes, ds.classes);
    assert!(dir_a.join("weights/best.folo").is_file());
    // Round trip: the EMA weights of last.folo are exactly the trainer's EMA model.
    let (loaded, _) = folo_vision::FoloModel::from_file(&file, folo_nn::io::Purpose::Inference).unwrap();
    let ema = a.ema_model().unwrap();
    for ((n1, t1), (n2, t2)) in loaded.named_tensors().iter().map(|t| (&t.name, &t.tensor)).zip(
        ema.named_tensors().iter().map(|t| (t.name.clone(), t.tensor.clone())).collect::<Vec<_>>().iter(),
    ) {
        assert_eq!(n1, n2);
        assert_eq!(t1.to_vec::<f32>().unwrap(), t2.to_vec::<f32>().unwrap(), "{n1}");
    }
    let meta: &Metadata = file.metadata();
    assert!(meta.metrics.is_some());

    // B: the same run, stopped after epoch 1, then resumed from its last.folo.
    let (opts, log_b) = options(Some(&dir_b));
    let stop = opts.stop.clone();
    let sink_log = log_b.clone();
    let opts = TrainOptions {
        events: Some(Box::new(move |e| {
            if matches!(e, RunEvent::Epoch(_)) {
                stop.store(true, Ordering::Relaxed);
            }
            sink_log.lock().unwrap().push(e.clone());
        })),
        ..opts
    };
    let sum_b1 = Trainer::new(resume_config(), data(&ds), opts).unwrap().run().unwrap();
    assert_eq!((sum_b1.status, sum_b1.epochs_completed), (RunStatus::Stopped, 1));
    let (opts, log_b2) = options(Some(&dir_b));
    let mut b = Trainer::resume(dir_b.join("weights/last.folo"), data(&ds), opts, ResumeOverrides::default())
        .unwrap();
    assert_eq!((b.epochs_completed(), b.schedule().step()), (1, 3));
    let sum_b2 = b.run().unwrap();
    assert_eq!((sum_b2.status, sum_b2.epochs_completed), (RunStatus::Completed, 2));

    // Bit-exact: the same epoch-2 numbers and the same final checkpoint tensors.
    let (ea, eb) = (&epochs_of(&log_a)[1], &epochs_of(&log_b2)[0]);
    assert_eq!((ea.epoch, ea.train_loss, &ea.val), (eb.epoch, eb.train_loss, &eb.val));
    let (ta, tb) = (tensors(&last), tensors(&dir_b.join("weights/last.folo")));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }

    // The predictor on the run's best weights (EMA, fused) agrees with the unfused EMA model.
    let p = Predictor::open(
        dir_a.join("weights/best.folo"),
        PredictConfig { imgsz: 64, conf: 0.001, ..Default::default() },
    )
    .unwrap();
    assert!(p.model().is_fused());
    let sample = folo_data::synthetic::generate(&ds, 0).unwrap().image;
    let single = p.predict(&sample).unwrap();
    assert_eq!((single.width, single.height), (64, 64));
    for d in &single.detections {
        assert!(d.bbox.x1 >= 0.0 && d.bbox.x2 <= 64.0 && d.bbox.y1 >= 0.0 && d.bbox.y2 <= 64.0, "{d:?}");
    }
    assert!(single.detections.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(single.timing.total_ms > 0.0);
    // A square 64×64 image letterboxes the same way in rect and square mode, so batch and single agree.
    let batch = p.predict_batch(&[sample.clone(), sample]).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].detections, batch[1].detections);
    assert_eq!(batch[0].detections.len(), single.detections.len());
    for (x, y) in batch[0].detections.iter().zip(&single.detections) {
        assert_eq!(x.class, y.class);
        assert!((x.score - y.score).abs() < 1e-5 && (x.bbox.x1 - y.bbox.x1).abs() < 1e-3);
    }

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// docs/MODEL.md §10.4: N = 800, batch 16, 30 epochs → nb 50, accumulate 4, spe 13, T 390, W 100; and `auto`
/// picks AdamW at `round(0.002 · 5 / (4 + nc), 6)`. The same data for 10 epochs is a short run: W 39, a warning.
#[test]
fn schedule_numbers_match_the_spec() {
    let img = folo_data::Image::new(64, 64, [0; 3]);
    let samples = (0..800).map(|_| folo_data::Sample { image: img.clone(), boxes: vec![] }).collect();
    let classes: Vec<String> = ["a", "b", "c", "d"].map(String::from).to_vec();
    let ds = Arc::new(InMemoryDataset { samples, classes: classes.clone(), split: Split::Train });
    let cfg = TrainConfig { epochs: 30, batch: 16, imgsz: 64, workers: 0, ..TrainConfig::default() };
    let data = TrainData { train: ds.clone(), val: ds, classes };
    let t = Trainer::new(cfg.clone(), data.clone(), TrainOptions::default()).unwrap();
    assert_eq!((t.batches_per_epoch(), t.accumulate()), (50, 4));
    let c = t.schedule().contract();
    assert_eq!((c.steps_per_epoch, t.schedule().total_iters(), c.warmup_iters), (13, 390, 100));
    assert!(t.warnings().is_empty(), "{:?}", t.warnings());
    assert_eq!(t.config().augment.close_mosaic_epochs(30), 10);

    let short = Trainer::new(TrainConfig { epochs: 10, ..cfg }, data, TrainOptions::default()).unwrap();
    let c = short.schedule().contract();
    assert_eq!((short.schedule().total_iters(), c.warmup_iters), (130, 39));
    assert_eq!(short.config().augment.close_mosaic_epochs(10), 5);
    let w = short.warnings();
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("130 optimizer steps") && w[0].contains("16 epochs"), "{}", w[0]);
    let o = t.optimizer();
    assert_eq!(
        (o.kind, o.lr0, o.momentum, o.warmup_bias_lr),
        (folo_nn::optim::OptimKind::AdamW, 0.00125, 0.9, 0.0)
    );

    let smoke = TrainConfig { epochs: 30, batch: 16, ..TrainConfig::default() };
    let r = smoke.resolve_optimizer(2000, 4);
    assert_eq!((r.kind, r.lr0), (folo_nn::optim::OptimKind::AdamW, 0.00125));
    let long = TrainConfig { epochs: 300, ..TrainConfig::default() };
    assert_eq!(long.resolve_optimizer(118_287, 80).kind, folo_nn::optim::OptimKind::Sgd);
}

/// A model whose forward is NaN: every batch is skipped with a warning and no step event, the BatchNorm
/// statistics are restored, no optimizer step is taken, and the run fails with `status: failed`.
#[test]
fn non_finite_batches_are_skipped_then_fail_the_run() {
    let ds = shapes(12, 2, 64);
    let (opts, log) = options(None);
    let cfg = TrainConfig { batch: 1, nbs: 1, ..micro_config(1) };
    let mut t = Trainer::new(cfg, data(&ds), opts).unwrap();
    {
        let _g = no_grad();
        t.model().backbone.stem.conv.weight.fill_assign(f32::NAN).unwrap();
    }
    let before: Vec<Vec<f32>> =
        t.model().named_buffers().iter().map(|(_, b)| b.to_vec::<f32>().unwrap()).collect();
    let err = t.run().unwrap_err();
    assert_eq!(err.code(), "training_failed", "{err}");
    assert_eq!(t.nonfinite_streak(), folo_vision::train::MAX_NONFINITE_BATCHES);
    // A failed epoch is past its checkpoint: no retry, no checkpoint.
    assert_eq!(t.run().unwrap_err().code(), "invalid_config");
    assert_eq!(
        t.save_checkpoint(std::env::temp_dir().join("folo-never.folo")).unwrap_err().code(),
        "invalid_config"
    );
    assert_eq!(t.schedule().step(), 0);
    assert!(RunningStats::all_finite(t.model()).unwrap());
    let after: Vec<Vec<f32>> =
        t.model().named_buffers().iter().map(|(_, b)| b.to_vec::<f32>().unwrap()).collect();
    assert_eq!(before, after, "the running statistics are restored after every skipped batch");
    let events = log.lock().unwrap();
    assert!(!events.iter().any(|e| matches!(e, RunEvent::Step(_))));
    let warns = events
        .iter()
        .filter(|e| matches!(e, RunEvent::Log(l) if l.level == folo_api::LogLevel::Warn && l.message.contains("non-finite")))
        .count();
    assert_eq!(warns, folo_vision::train::MAX_NONFINITE_BATCHES);
    assert!(
        matches!(events.last(), Some(RunEvent::Status(s)) if s.status == RunStatus::Failed && s.error.is_some())
    );
}

#[test]
fn trainer_is_send() {
    fn send<T: Send>() {}
    send::<Trainer>();
    send::<Predictor>();
}

#[test]
fn invalid_configs_are_rejected() {
    let ds = shapes(4, 1, 64);
    for (cfg, what) in [
        (TrainConfig { imgsz: 48, ..micro_config(1) }, "imgsz"),
        (TrainConfig { epochs: 0, ..micro_config(1) }, "epochs"),
        (TrainConfig { arch: "folo-q".into(), ..micro_config(1) }, "arch"),
        (TrainConfig { device: "tpu".into(), ..micro_config(1) }, "device"),
    ] {
        let err = Trainer::new(cfg, data(&ds), TrainOptions::default()).err().expect(what);
        assert_eq!(err.code(), "invalid_config", "{what}: {err}");
        assert!(err.to_string().contains(what), "{err}");
    }
    let gpu = TrainConfig { device: "gpu:3".into(), ..micro_config(1) };
    let err = Trainer::new(gpu, data(&ds), TrainOptions::default()).err().unwrap();
    assert_eq!(err.code(), "device_not_found");
    assert!(err.to_string().contains("FOLO can use cpu"), "{err}");
    let empty = Arc::new(InMemoryDataset { samples: vec![], classes: vec!["a".into()], split: Split::Train });
    let no_images = TrainData { train: empty.clone(), val: empty, classes: vec!["a".into()] };
    let err = Trainer::new(micro_config(1), no_images, TrainOptions::default()).err().unwrap();
    assert_eq!(err.code(), "invalid_dataset");
}

/// The loss falls on a fixed 8-image set: 10 epochs of 2 AdamW steps at 128 px without augmentation. (At 64 px the
/// objects are so small that the first steps raise the reported box / DFL terms: they are normalised by the
/// target-score sum, which starts below its floor of 1 and grows as the IoUs improve.)
#[test]
fn micro_run_loss_decreases() {
    let ds = shapes(8, 2, 128);
    let (opts, log) = options(None);
    let cfg = TrainConfig { nbs: 4, imgsz: 128, ema: false, ..micro_config(10) };
    Trainer::new(cfg, data(&ds), opts).unwrap().run().unwrap();
    let losses: Vec<f32> = epochs_of(&log).iter().map(|e| e.train_loss.total).collect();
    let tail = losses[losses.len() - 3..].iter().sum::<f32>() / 3.0;
    assert!(tail < 0.6 * losses[0], "the loss must fall: {losses:?}");
}

/// The stop flag inside an epoch: the run ends `stopped` with no epoch completed, writes no checkpoint, and the
/// trainer refuses to continue from its (now past-the-checkpoint) state.
#[test]
fn stop_inside_an_epoch() {
    let ds = shapes(8, 2, 64);
    let dir = temp_dir("stop");
    let (opts, log) = options(Some(&dir));
    let stop = opts.stop.clone();
    let sink = log.clone();
    let opts = TrainOptions {
        events: Some(Box::new(move |e| {
            if matches!(e, RunEvent::Step(_)) {
                stop.store(true, Ordering::Relaxed);
            }
            sink.lock().unwrap().push(e.clone());
        })),
        ..opts
    };
    let mut t = Trainer::new(micro_config(2), data(&ds), opts).unwrap();
    let s = t.run().unwrap();
    assert_eq!((s.status, s.epochs_completed, s.last.is_none()), (RunStatus::Stopped, 0, true));
    assert!(!dir.join("weights/last.folo").exists());
    assert_eq!(t.run().unwrap_err().code(), "invalid_config");
    assert_eq!(t.save_checkpoint(dir.join("mid.folo")).unwrap_err().code(), "invalid_config");
    assert!(
        matches!(log.lock().unwrap().last(), Some(RunEvent::Status(s)) if s.status == RunStatus::Stopped)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// docs/MODEL.md §10.4 overfit: folo-n at 320 px on 8 images, augmentation off, AdamW `lr0 = 0.002`, 300 steps of
/// batch 8 without accumulation. The loss falls below 25 % of its first value and mAP@0.5 on the same images
/// exceeds 0.95. Minutes on a desktop CPU:
/// `cargo test --release -p folo-vision --test train -- --ignored overfit`.
#[test]
#[ignore]
fn overfit_eight_images() {
    let ds = SyntheticConfig {
        classes: ["circle", "square", "triangle", "star"].map(String::from).to_vec(),
        ..shapes(8, 0, 320)
    };
    let train = Arc::new(SyntheticDataset::new(ds.clone(), Split::Train).unwrap());
    let data = TrainData { train: train.clone(), val: train, classes: ds.classes.clone() };
    let (opts, log) = options(None);
    let cfg = TrainConfig { epochs: 300, batch: 8, nbs: 8, imgsz: 320, patience: 1000, ..micro_config(300) };
    let s = Trainer::new(cfg, data, opts).unwrap().run().unwrap();
    let steps: Vec<f32> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Step(s) = e { Some(s.loss.total) } else { None })
        .collect();
    let last = epochs_of(&log).last().unwrap().clone();
    println!(
        "first loss {}, last {}, final mAP50 {} mAP50-95 {}",
        steps[0],
        steps[steps.len() - 1],
        last.val.map50,
        last.val.map50_95
    );
    assert_eq!(s.epochs_completed, 300);
    assert!(steps[steps.len() - 1] < 0.25 * steps[0], "loss {} -> {}", steps[0], steps[steps.len() - 1]);
    assert!(last.val.map50 > 0.95, "mAP50 {}", last.val.map50);
}

/// One non-finite batch, then healthy ones (MODEL.md §10.4): the bad batch is skipped with a warning and no step
/// event, its window takes no optimizer step, the BatchNorm statistics stay finite, training continues, and a
/// checkpoint written after it resumes to the same state as the uninterrupted run. Also: a resume on a dataset
/// of another size is refused, and an interrupted trainer refuses to write a checkpoint.
#[test]
fn non_finite_batch_recovers_and_resumes() {
    let ds = shapes(8, 2, 64);
    // accumulate 1: the bad first batch is a window of its own.
    let cfg = || TrainConfig { nbs: 4, ..micro_config(2) };
    let dir_a = temp_dir("nan-a");
    let dir_b = temp_dir("nan-b");

    // The first `warn` (the skipped batch) puts the poisoned stem weights back.
    let run = |dir: &Path, stop_after_epoch: bool| {
        let (opts, log) = options(Some(dir));
        let stop = opts.stop.clone();
        let sink = log.clone();
        let w: Arc<Mutex<Option<Saved>>> = Arc::default();
        let w2 = w.clone();
        let opts = TrainOptions {
            events: Some(Box::new(move |e| {
                if let RunEvent::Log(l) = e
                    && l.level == folo_api::LogLevel::Warn
                    && l.message.contains("non-finite")
                    && let Some((t, clean)) = w2.lock().unwrap().take()
                {
                    let _g = no_grad();
                    t.copy_from(&folo_core::Tensor::from_vec(clean, t.shape()).unwrap()).unwrap();
                }
                if stop_after_epoch && matches!(e, RunEvent::Epoch(_)) {
                    stop.store(true, Ordering::Relaxed);
                }
                sink.lock().unwrap().push(e.clone());
            })),
            ..opts
        };
        let mut t = Trainer::new(cfg(), data(&ds), opts).unwrap();
        let weight = t.model().backbone.stem.conv.weight.clone();
        *w.lock().unwrap() = Some((weight.clone(), weight.to_vec::<f32>().unwrap()));
        {
            let _g = no_grad();
            weight.fill_assign(f32::NAN).unwrap();
        }
        let s = t.run().unwrap();
        (t, s, log)
    };

    let (a, sa, log_a) = run(&dir_a, false);
    assert_eq!((sa.status, sa.epochs_completed), (RunStatus::Completed, 2));
    // 2 epochs × 2 batches, the first skipped: 3 optimizer steps, 3 EMA updates, 3 step events.
    assert_eq!(a.schedule().step(), 3);
    assert!(RunningStats::all_finite(a.model()).unwrap());
    let events = log_a.lock().unwrap().clone();
    let steps: Vec<u64> =
        events.iter().filter_map(|e| if let RunEvent::Step(s) = e { Some(s.step) } else { None }).collect();
    assert_eq!(steps, [2, 3, 4]);
    let warns = events
        .iter()
        .filter(|e| matches!(e, RunEvent::Log(l) if l.level == folo_api::LogLevel::Warn && l.message.contains("non-finite")))
        .count();
    assert_eq!(warns, 1);
    let c = ModelFile::open(dir_a.join("weights/last.folo")).unwrap();
    assert_eq!(c.checkpoint().unwrap().ema_updates, Some(3));

    // B: the same, stopped after epoch 1 (whose only optimizer step came after the skipped window), resumed.
    let (_, sb, _) = run(&dir_b, true);
    assert_eq!((sb.status, sb.epochs_completed), (RunStatus::Stopped, 1));
    let ckpt = dir_b.join("weights/last.folo");
    assert_eq!(ModelFile::open(&ckpt).unwrap().checkpoint().unwrap().sched.step, 1);
    let other = shapes(12, 2, 64);
    let err = Trainer::resume(&ckpt, data(&other), TrainOptions::default(), ResumeOverrides::default())
        .err()
        .unwrap();
    assert_eq!(err.code(), "invalid_dataset", "{err}");
    let (opts, _) = options(Some(&dir_b));
    let mut b = Trainer::resume(&ckpt, data(&ds), opts, ResumeOverrides::default()).unwrap();
    b.run().unwrap();
    let (ta, tb) = (tensors(&dir_a.join("weights/last.folo")), tensors(&ckpt));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// A run of non-finite batches at the end of an epoch survives a checkpoint: the streak is stored with it and
/// restored on resume.
#[test]
fn nonfinite_streak_crosses_a_checkpoint() {
    let ds = shapes(8, 2, 64);
    let dir = temp_dir("streak");
    let (opts, log) = options(Some(&dir));
    let w: Arc<Mutex<Option<Saved>>> = Arc::default();
    let (w2, sink) = (w.clone(), log.clone());
    let opts = TrainOptions {
        events: Some(Box::new(move |e| {
            // After the first batch's step, poison the weights; the warning of the (last) bad batch heals them.
            let mut slot = w2.lock().unwrap();
            if let (RunEvent::Step(s), Some((t, _))) = (e, slot.as_ref())
                && s.step == 1
            {
                let _g = no_grad();
                t.fill_assign(f32::NAN).unwrap();
            }
            if let RunEvent::Log(l) = e
                && l.level == folo_api::LogLevel::Warn
                && l.message.contains("non-finite")
                && let Some((t, clean)) = slot.take()
            {
                let _g = no_grad();
                t.copy_from(&folo_core::Tensor::from_vec(clean, t.shape()).unwrap()).unwrap();
            }
            sink.lock().unwrap().push(e.clone());
        })),
        ..opts
    };
    // 2 batches per epoch, no accumulation: batch 2 of epoch 1 is the bad one.
    let mut t = Trainer::new(TrainConfig { nbs: 4, ..micro_config(1) }, data(&ds), opts).unwrap();
    let weight = t.model().backbone.stem.conv.weight.clone();
    *w.lock().unwrap() = Some((weight.clone(), weight.to_vec::<f32>().unwrap()));
    t.run().unwrap();
    assert_eq!((t.nonfinite_streak(), t.schedule().step()), (1, 1));
    let ckpt = dir.join("weights/last.folo");
    let file = ModelFile::open(&ckpt).unwrap();
    assert_eq!(
        file.metadata().extra.get(folo_vision::train::NONFINITE_STREAK_KEY).map(String::as_str),
        Some("1")
    );
    let r = Trainer::resume(&ckpt, data(&ds), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    assert_eq!(r.nonfinite_streak(), 1);
    assert!(
        log.lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e, RunEvent::Log(l) if l.level == folo_api::LogLevel::Warn && l.message.contains("non-finite")))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A run that stopped early (patience 0 stops after the first epoch) stays stopped when its checkpoint is resumed.
#[test]
fn resuming_an_early_stopped_run_trains_no_further() {
    let ds = shapes(8, 2, 64);
    let dir = temp_dir("patience");
    let (opts, _) = options(Some(&dir));
    let s =
        Trainer::new(TrainConfig { patience: 0, ..micro_config(3) }, data(&ds), opts).unwrap().run().unwrap();
    assert_eq!((s.epochs_completed, s.early_stopped), (1, true));
    let (opts, log) = options(Some(&dir));
    let mut r =
        Trainer::resume(dir.join("weights/last.folo"), data(&ds), opts, ResumeOverrides::default()).unwrap();
    let step = r.schedule().step();
    let s = r.run().unwrap();
    assert_eq!((s.status, s.epochs_completed, s.early_stopped), (RunStatus::Completed, 1, true));
    assert_eq!(r.schedule().step(), step);
    assert!(!log.lock().unwrap().iter().any(|e| matches!(e, RunEvent::Step(_))));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Unknown `folo.*` metadata of a resumed checkpoint survives into the checkpoints the run writes (FORMAT.md).
#[test]
fn resume_preserves_unknown_metadata() {
    let ds = shapes(8, 2, 64);
    let dir = temp_dir("meta");
    let (opts, _) = options(Some(&dir));
    Trainer::new(micro_config(1), data(&ds), opts).unwrap().run().unwrap();
    // Add a provenance key to last.folo, as another tool might.
    let last = dir.join("weights/last.folo");
    let file = ModelFile::open(&last).unwrap();
    // The full header metadata (checkpoint contracts included), read back from the file.
    let mut map = std::collections::BTreeMap::new();
    let raw = std::fs::read(&last).unwrap();
    let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + header_len]).unwrap();
    for (k, v) in header["__metadata__"].as_object().unwrap() {
        map.insert(k.clone(), v.as_str().unwrap().to_string());
    }
    map.insert("folo.dataset_hash".into(), "abc123".into());
    let tensors: Vec<(String, folo_core::Tensor)> =
        file.tensor_infos().iter().map(|i| (i.name.clone(), file.tensor(&i.name).unwrap())).collect();
    drop(file);
    let tagged = dir.join("weights/tagged.folo");
    folo_nn::io::safetensors::write_file(&tagged, Some(&map), &tensors).unwrap();

    let r = Trainer::resume(&tagged, data(&ds), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    let again = dir.join("weights/again.folo");
    r.save_checkpoint(&again).unwrap();
    let meta = ModelFile::open(&again).unwrap().metadata().clone();
    assert_eq!(meta.extra.get("folo.dataset_hash").map(String::as_str), Some("abc123"));
    let _ = std::fs::remove_dir_all(&dir);
}
