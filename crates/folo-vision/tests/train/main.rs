//! Trainer tests (docs/MODEL.md §10.4) on tiny synthetic sets: a micro run that learns, checkpoint round trips,
//! bit-exact resume, the schedule numbers, the non-finite guard, and the predictor on trained weights.

mod ema;

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use folo_api::{RunEvent, RunStatus};
use folo_core::no_grad;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, InMemoryDataset, Split};
use folo_nn::io::{Metadata, ModelFile, Namespace};
use folo_nn::{Module, RunningStats};
use folo_test_support::{TestDir, TestFile};
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

/// The resume run: 9 train images in batches of 2 (5 batches, the last one short: windows of 2, 2 and a short 1),
/// warmup over the first 3 of 6 optimizer steps, the default augmentation with mosaic closing for the second epoch.
fn resume_config() -> TrainConfig {
    TrainConfig {
        batch: 2,
        nbs: 4,
        warmup_epochs: 1.0,
        augment: AugmentConfig { close_mosaic: 1, ..AugmentConfig::default() },
        ..micro_config(2)
    }
}

fn run_dir(name: &str) -> TestDir {
    TestDir::new(&format!("train-{name}"))
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

/// Stops the run at the boundary after epoch `.0` ([`TrainOptions::after_epoch`]).
struct StopAfter(u32);

impl folo_vision::train::EpochHook for StopAfter {
    fn pauses(&mut self, epoch: u32) -> bool {
        epoch == self.0
    }

    fn pause(&mut self, _: u32, stop: &std::sync::atomic::AtomicBool) {
        stop.store(true, Ordering::Relaxed);
    }
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
    let ds = shapes(9, 2, 64);
    let dir_a = run_dir("a");
    let dir_b = run_dir("b");

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
    assert_eq!(samples["samples"].as_array().unwrap().len(), 2);

    // Checkpoints: a complete FORMAT.md checkpoint at the epoch boundary.
    let last = dir_a.join("weights/last.folo");
    let file = ModelFile::open(&last).unwrap();
    let c = file.checkpoint().expect("last.folo is a checkpoint");
    assert_eq!((c.epoch, c.optim.step, c.sched.step, c.ema_updates), (2, 6, 6, Some(6)));
    // The closing epochs a resume keeps (close_mosaic 1 of 2 epochs).
    assert_eq!(c.train[folo_vision::train::CLOSE_MOSAIC_KEY], 1);
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

    // B: the same run, stopped after epoch 1 (at the boundary: its event arrives once its files are written,
    // possibly after epoch 2 began), then resumed from its last.folo.
    let (opts, _log_b) = options(Some(&dir_b));
    let opts = TrainOptions { after_epoch: Some(Box::new(StopAfter(1))), ..opts };
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

/// docs/MODEL.md §10.4: N = 800, batch 16, 30 epochs → nb 50, accumulate 4, spe 13, T 390, W 39 (100 with the
/// pre-rule-4 floor); and `auto` picks MuSGD (rule 4). The same data for 10 epochs is a short run: W 39, a warning.
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
    assert_eq!((c.steps_per_epoch, t.schedule().total_iters(), c.warmup_iters), (13, 390, 39));
    let floored =
        Trainer::new(TrainConfig { warmup_floor: 100, ..cfg.clone() }, data.clone(), TrainOptions::default());
    assert_eq!(floored.unwrap().schedule().contract().warmup_iters, 100);
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
        (folo_nn::optim::OptimKind::MuSgd, 0.01, 0.937, 0.1)
    );

    let smoke = TrainConfig { epochs: 30, batch: 16, auto_rule: 3, ..TrainConfig::default() };
    let r = smoke.resolve_optimizer(2000, 4);
    assert_eq!((r.kind, r.lr0), (folo_nn::optim::OptimKind::AdamW, 0.00125), "rule 3: AdamW for a short run");
    let long = TrainConfig { epochs: 300, ..TrainConfig::default() };
    // Long runs: MuSGD under `auto` rules 3 and 4, SGD under the rules before.
    assert_eq!(long.resolve_optimizer(118_287, 80).kind, folo_nn::optim::OptimKind::MuSgd);
    let rule2 = TrainConfig { auto_rule: 2, ..long };
    assert_eq!(rule2.resolve_optimizer(118_287, 80).kind, folo_nn::optim::OptimKind::Sgd);
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
        t.save_checkpoint(TestFile::new("train-never", "never.folo")).unwrap_err().code(),
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

/// The loss falls on a fixed 4-image set: 10 epochs of one AdamW step at 128 px without augmentation. (At 64 px the
/// objects are so small that the first steps raise the reported box / DFL terms: they are normalised by the
/// target-score sum, which starts below its floor of 1 and grows as the IoUs improve.)
#[test]
fn micro_run_loss_decreases() {
    let ds = shapes(4, 1, 128);
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
    let dir = run_dir("stop");
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
    let root = run_dir("nan");

    // The first `warn` (the skipped batch) puts the poisoned stem weights back.
    let run = |stop_after_epoch: bool| {
        let (opts, log) = options(None);
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

    let (a, sa, log_a) = run(false);
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
    let last_a = root.join("a.folo");
    a.save_checkpoint(&last_a).unwrap();
    let c = ModelFile::open(&last_a).unwrap();
    assert_eq!(c.checkpoint().unwrap().ema_updates, Some(3));
    drop(c);

    // B: the same, stopped after epoch 1 (whose only optimizer step came after the skipped window), resumed.
    let (b, sb, _) = run(true);
    assert_eq!((sb.status, sb.epochs_completed), (RunStatus::Stopped, 1));
    let ckpt = root.join("b1.folo");
    b.save_checkpoint(&ckpt).unwrap();
    assert_eq!(ModelFile::open(&ckpt).unwrap().checkpoint().unwrap().sched.step, 1);
    let other = shapes(12, 2, 64);
    let err = Trainer::resume(&ckpt, data(&other), TrainOptions::default(), ResumeOverrides::default())
        .err()
        .unwrap();
    assert_eq!(err.code(), "invalid_dataset", "{err}");
    let mut r =
        Trainer::resume(&ckpt, data(&ds), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    r.run().unwrap();
    let resumed = root.join("b2.folo");
    r.save_checkpoint(&resumed).unwrap();
    let (ta, tb) = (tensors(&last_a), tensors(&resumed));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// A run of non-finite batches at the end of an epoch survives a checkpoint: the streak is stored with it and
/// restored on resume.
#[test]
fn nonfinite_streak_crosses_a_checkpoint() {
    let ds = shapes(8, 2, 64);
    let (opts, log) = options(None);
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
    let ckpt = TestFile::new("train-streak", "last.folo");
    t.save_checkpoint(&ckpt).unwrap();
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
}

/// A run that stopped early (patience 0 stops after the first epoch) stays stopped when its checkpoint is resumed.
#[test]
fn resuming_an_early_stopped_run_trains_no_further() {
    let ds = shapes(8, 2, 64);
    let mut t =
        Trainer::new(TrainConfig { patience: 0, ..micro_config(3) }, data(&ds), options(None).0).unwrap();
    let s = t.run().unwrap();
    assert_eq!((s.epochs_completed, s.early_stopped), (1, true));
    let ckpt = TestFile::new("train-patience", "last.folo");
    t.save_checkpoint(&ckpt).unwrap();
    let (opts, log) = options(None);
    let mut r = Trainer::resume(&ckpt, data(&ds), opts, ResumeOverrides::default()).unwrap();
    let step = r.schedule().step();
    let s = r.run().unwrap();
    assert_eq!((s.status, s.epochs_completed, s.early_stopped), (RunStatus::Completed, 1, true));
    assert_eq!(r.schedule().step(), step);
    assert!(!log.lock().unwrap().iter().any(|e| matches!(e, RunEvent::Step(_))));
}

/// Unknown `folo.*` metadata of a resumed checkpoint survives into the checkpoints the run writes (FORMAT.md).
#[test]
fn resume_preserves_unknown_metadata() {
    let ds = shapes(8, 2, 64);
    let dir = run_dir("meta");
    let mut t = Trainer::new(micro_config(1), data(&ds), TrainOptions::default()).unwrap();
    t.run().unwrap();
    // Add a provenance key to the checkpoint, as another tool might.
    let last = dir.join("last.folo");
    t.save_checkpoint(&last).unwrap();
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
    let tagged = dir.join("tagged.folo");
    folo_nn::io::safetensors::write_file(&tagged, Some(&map), &tensors).unwrap();

    let r = Trainer::resume(&tagged, data(&ds), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    let again = dir.join("again.folo");
    r.save_checkpoint(&again).unwrap();
    let meta = ModelFile::open(&again).unwrap().metadata().clone();
    assert_eq!(meta.extra.get("folo.dataset_hash").map(String::as_str), Some("abc123"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// MuSGD in the trainer (docs/MODEL.md §8.2), with either head: the head's class branches (`head.cls.`, and
/// `head.o2o.cls.` with the one-to-one head) and nothing else are in the `.cls` groups, at `MUSGD_CLS_LR_FACTOR`
/// times the LR; the losses are finite; the checkpoint holds a `muon_buffer` for every 2-D / 4-D parameter of the
/// `decay` groups; and a resume at an epoch boundary continues bit for bit. The dual-assignment run (`head: o2o`,
/// docs/MODEL.md §6.6) also leaves an `o2o` model (both branches) whose o2o branch trained and which predicts end
/// to end with the one-to-one head, no NMS (its dual-loss weight depends only on the completed epochs).
#[test]
fn musgd_runs_train_checkpoint_and_resume_with_either_head() {
    use folo_vision::model::{FoloModel, HeadMode};
    use folo_vision::train::MUSGD_CLS_LR_FACTOR;
    // 4 train images in batches of 2 with accumulation 2: one optimizer step an epoch (MuSGD's are the slow ones).
    let ds = shapes(4, 2, 64);
    let cls = |n: &str| n.starts_with("head.cls.") || n.starts_with("head.o2o.cls.");
    let root = run_dir("musgd");
    for head in [HeadMode::O2m, HeadMode::O2o] {
        let cfg = TrainConfig { optimizer: OptimizerChoice::Musgd, lr0: 0.01, head, ..resume_config() };
        // A: two uninterrupted epochs.
        let (opts, log_a) = options(None);
        let mut a = Trainer::new(cfg.clone(), data(&ds), opts).unwrap();
        let lr0 = a.optimizer().lr0;
        let groups: Vec<&str> = a.param_groups().iter().map(|g| g.name.as_str()).collect();
        assert_eq!(groups, ["decay", "decay.cls", "bn", "bn.cls", "bias", "bias.cls"], "{head:?}");
        let mut muon = 0;
        for g in a.param_groups() {
            let boosted = g.name.ends_with(".cls");
            let want = if boosted { lr0 * MUSGD_CLS_LR_FACTOR } else { lr0 };
            assert!((g.base_lr - want).abs() < 1e-12, "{head:?} {}: lr {} vs {want}", g.name, g.base_lr);
            for (n, p) in &g.params {
                assert_eq!(cls(n), boosted, "{head:?}: {n} in {}", g.name);
                muon += usize::from(g.name.starts_with("decay") && matches!(p.rank(), 2 | 4));
            }
        }
        let o2o_cls =
            a.param_groups().iter().flat_map(|g| &g.params).any(|(n, _)| n.starts_with("head.o2o.cls."));
        assert_eq!(o2o_cls, head == HeadMode::O2o);
        let sum = a.run().unwrap();
        assert_eq!((sum.status, sum.epochs_completed), (RunStatus::Completed, 2));
        let epochs = epochs_of(&log_a);
        assert!(
            epochs.iter().all(|e| e.train_loss.total.is_finite() && e.train_loss.total > 0.0),
            "{head:?}"
        );
        let last = root.join("a.folo");
        a.save_checkpoint(&last).unwrap();
        let ta = tensors(&last);
        assert!(muon > 0);
        assert_eq!(ta.iter().filter(|(n, _)| n.ends_with(".muon_buffer")).count(), muon, "{head:?}");

        // B: stopped after epoch 1, resumed from its checkpoint: the same epoch 2 and the same final tensors.
        let (opts, _) = options(None);
        let stop = opts.stop.clone();
        let opts = TrainOptions {
            events: Some(Box::new(move |e| {
                if matches!(e, RunEvent::Epoch(_)) {
                    stop.store(true, Ordering::Relaxed);
                }
            })),
            ..opts
        };
        let mut b = Trainer::new(cfg.clone(), data(&ds), opts).unwrap();
        let s1 = b.run().unwrap();
        assert_eq!((s1.status, s1.epochs_completed), (RunStatus::Stopped, 1));
        let ckpt = root.join("b1.folo");
        b.save_checkpoint(&ckpt).unwrap();
        let (opts, log_b) = options(None);
        let mut r = Trainer::resume(&ckpt, data(&ds), opts, ResumeOverrides::default()).unwrap();
        assert_eq!(r.model().has_o2o(), head == HeadMode::O2o);
        assert_eq!(r.optimizer().kind, folo_nn::optim::OptimKind::MuSgd);
        r.run().unwrap();
        let (ea, eb) = (&epochs[1], &epochs_of(&log_b)[0]);
        assert_eq!((ea.epoch, ea.train_loss, &ea.val), (eb.epoch, eb.train_loss, &eb.val), "{head:?}");
        let resumed = root.join("b2.folo");
        r.save_checkpoint(&resumed).unwrap();
        let tb = tensors(&resumed);
        assert_eq!(ta.len(), tb.len());
        for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
            assert!(na == nb && va == vb, "{head:?}: tensor {na} differs after resume");
        }
        if head != HeadMode::O2o {
            continue;
        }

        // The dual-assignment run: an `o2o` model whose o2o branch no longer equals its initialisation...
        assert_eq!(ModelFile::open(&last).unwrap().metadata().config.head, "o2o");
        let m = FoloModel::open(&last).unwrap();
        assert!(m.has_o2o() && m.head_mode() == HeadMode::O2o);
        let init = FoloModel::with_rng(m.config().clone(), &mut folo_core::rng::Rng::seed(cfg.seed)).unwrap();
        let get = |m: &FoloModel, n: &str| {
            m.state().into_iter().find(|(k, _)| k == n).unwrap().1.to_vec::<f32>().unwrap()
        };
        assert_ne!(get(&m, "head.o2o.cls.p3.out.weight"), get(&init, "head.o2o.cls.p3.out.weight"));
        // ... and whose weights predict end to end (no NMS).
        let p =
            Predictor::open(&last, PredictConfig { imgsz: 64, conf: 0.001, ..Default::default() }).unwrap();
        assert_eq!(p.model().head_mode(), HeadMode::O2o);
        let sample = folo_data::synthetic::generate(&ds, 0).unwrap().image;
        let pred = p.predict(&sample).unwrap();
        assert!(pred.detections.windows(2).all(|w| w[0].score >= w[1].score));
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// `o2o_attached_epochs` in the trainer (docs/MODEL.md §6.6): with 2 of 3 epochs attached, run one epoch at a time
/// through checkpoints, the one-to-one branch reads attached features in epochs 1 and 2 (also in epoch 2, the
/// first epoch of a resumed run, whose model starts detached) and detached ones in epoch 3.
#[test]
fn o2o_attached_epochs_switch_at_the_boundary() {
    use folo_vision::model::HeadMode;
    let ds = shapes(1, 1, 64);
    let cfg = TrainConfig {
        head: HeadMode::O2o,
        o2o_attached_epochs: 2,
        batch: 1,
        nbs: 1,
        ema: false,
        ..micro_config(3)
    };
    let detached = |t: &Trainer| t.model().head.o2o_detach;
    // Options that stop the run after its first epoch.
    let one_epoch = || {
        let (opts, _) = options(None);
        let stop = opts.stop.clone();
        let events: folo_vision::train::EventSink = Box::new(move |e| {
            if matches!(e, RunEvent::Epoch(_)) {
                stop.store(true, Ordering::Relaxed);
            }
        });
        TrainOptions { events: Some(events), ..opts }
    };
    let ckpt = TestFile::new("train-o2o-attached", "last.folo");
    let mut t = Trainer::new(cfg, data(&ds), one_epoch()).unwrap();
    assert_eq!(t.run().unwrap().epochs_completed, 1);
    assert!(!detached(&t), "epoch 1 of 2 attached");
    t.save_checkpoint(&ckpt).unwrap();
    let mut t = Trainer::resume(&ckpt, data(&ds), one_epoch(), ResumeOverrides::default()).unwrap();
    assert!(detached(&t), "a resumed model starts with the default");
    assert_eq!(t.run().unwrap().epochs_completed, 2);
    assert!(!detached(&t), "epoch 2, the resumed run's first, is still attached");
    t.save_checkpoint(&ckpt).unwrap();
    let mut t = Trainer::resume(&ckpt, data(&ds), options(None).0, ResumeOverrides::default()).unwrap();
    assert_eq!(t.run().unwrap().epochs_completed, 3);
    assert!(detached(&t), "epoch 3 is past the 2 attached epochs");
}

/// `loss.tal.*` reach the one-to-one branch (docs/MODEL.md §6.6). After one optimizer step the branch's weights
/// depend only on its own loss (its inputs are detached, the backbone has not moved yet and no global clipping
/// couples it to the other branch), so they change with `beta` exactly when its assigner reads it; a config from
/// before `o2o_keeps_tal` keeps the §7.3 values.
#[test]
fn o2o_branch_takes_the_runs_assigner_settings() {
    use folo_vision::assign::TalConfig;
    use folo_vision::loss::LossConfig;
    use folo_vision::model::HeadMode;
    let ds = shapes(4, 1, 64);
    let base = TrainConfig {
        head: HeadMode::O2o,
        device: "cpu".into(),
        batch: 4,
        nbs: 4,
        ema: false,
        clip_grad_norm: 0.0,
        ..micro_config(1)
    };
    let o2o_weights = |cfg: TrainConfig| {
        let mut t = Trainer::new(cfg, data(&ds), options(None).0).unwrap();
        t.run().unwrap();
        let weights = t.model().named_tensors().into_iter().filter(|x| x.name.starts_with("head.o2o."));
        weights.map(|x| x.tensor.to_vec::<f32>().unwrap()).collect::<Vec<_>>()
    };
    let loss = LossConfig { tal: TalConfig { beta: 0.0, ..TalConfig::default() }, ..base.loss };
    let custom = TrainConfig { loss, ..base.clone() };
    let plain = o2o_weights(base);
    assert_ne!(o2o_weights(custom.clone()), plain, "the branch ignored loss.tal.beta");
    assert_eq!(o2o_weights(TrainConfig { o2o_keeps_tal: false, ..custom }), plain, "an old run's branch");
}

/// `auto` rule 5 (docs/MODEL.md §8.2): MuSGD with the scratch step weights 0.5 / 0.5 for a run from scratch, the
/// configured 0.2 / 1.0 for one that starts from weights (`transfer`, which the trainer sets from
/// `TrainOptions::weights` whatever the config says, and which a resume keeps). Checked on the optimizer itself
/// (its `folo.optim` contract in each checkpoint, which carries the weights only when they are not 0.2 / 1.0), also
/// after a resume of each; no training needed.
#[test]
fn auto_musgd_weights_follow_transfer_and_survive_resume() {
    use folo_nn::optim::{AUTO_MUSGD_SCRATCH_WEIGHTS, MuSgdWeights, OptimKind};
    // The `musgd` weights of a checkpoint's `folo.optim`, `None` when the key is absent (0.2 / 1.0).
    let saved = |path: &Path| -> Option<MuSgdWeights> {
        let header = folo_nn::io::safetensors::Container::read_header(path).unwrap();
        let optim: serde_json::Value = serde_json::from_str(&header.metadata.unwrap()["folo.optim"]).unwrap();
        optim.get("musgd").map(|m| serde_json::from_value(m.clone()).unwrap())
    };
    let resume =
        |p: &Path| Trainer::resume(p, data(&shapes(2, 1, 64)), options(None).0, ResumeOverrides::default());
    let ds = shapes(2, 1, 64);
    let cfg = TrainConfig { optimizer: OptimizerChoice::Auto, transfer: true, ..micro_config(2) };
    let scratch = Trainer::new(cfg.clone(), data(&ds), options(None).0).unwrap();
    assert!(!scratch.config().transfer, "no weights: from scratch, whatever the config said");
    let o = scratch.optimizer();
    assert_eq!((o.kind, o.musgd), (OptimKind::MuSgd, AUTO_MUSGD_SCRATCH_WEIGHTS));
    let start = TestFile::new("train-auto-transfer", "start.folo");
    scratch.save_checkpoint(&start).unwrap();
    assert_eq!(saved(&start), Some(AUTO_MUSGD_SCRATCH_WEIGHTS), "the optimizer itself runs 0.5 / 0.5");
    let again = resume(&start).unwrap();
    assert!(!again.config().transfer);
    let again_ckpt = TestFile::new("train-auto-transfer", "again.folo");
    again.save_checkpoint(&again_ckpt).unwrap();
    assert_eq!(saved(&again_ckpt), Some(AUTO_MUSGD_SCRATCH_WEIGHTS), "a resumed run from scratch keeps them");

    let opts = TrainOptions { weights: Some(start.path().to_path_buf()), ..options(None).0 };
    let tuned = Trainer::new(TrainConfig { transfer: false, ..cfg }, data(&ds), opts).unwrap();
    assert!(tuned.config().transfer);
    assert_eq!(tuned.optimizer().musgd, MuSgdWeights::default());
    let ckpt = TestFile::new("train-auto-transfer", "last.folo");
    tuned.save_checkpoint(&ckpt).unwrap();
    assert_eq!(saved(&ckpt), None, "the optimizer itself runs 0.2 / 1.0");
    let resumed = resume(&ckpt).unwrap();
    assert!(resumed.config().transfer, "a resumed fine-tune stays one");
    let resumed_ckpt = TestFile::new("train-auto-transfer", "resumed.folo");
    resumed.save_checkpoint(&resumed_ckpt).unwrap();
    assert_eq!(saved(&resumed_ckpt), None, "and keeps 0.2 / 1.0");
}
