//! The trainer on the GPU: a tiny run whose loss falls, with the model, optimizer state, EMA and gradient
//! buffers on the device, checkpoints that reload on the CPU, and a resume that continues on the device.
//!
//! The runs have no run directory and write a checkpoint only where a test reads one (`Trainer::save_checkpoint`,
//! which also writes a run directory's `last.folo`): on the GPU an epoch of these runs costs about as much as one
//! checkpoint.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.

use std::sync::Arc;

use folo_api::{RunEvent, RunStatus};
use folo_core::Device;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::Module;
use folo_nn::io::ModelFile;
use folo_test_support::{TestDir, TestFile};
use folo_vision::train::{OptimizerChoice, ResumeOverrides};
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};

fn gpu() -> Option<Device> {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    folo_gpu::init().ok()
}

fn data() -> TrainData {
    let cfg = SyntheticConfig {
        images: 20,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 128,
        seed: 5,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 4.0 / 20.0,
        ..SyntheticConfig::default()
    };
    let train = SyntheticDataset::new(cfg.clone(), Split::Train).unwrap();
    let val = SyntheticDataset::new(cfg.clone(), Split::Val).unwrap();
    TrainData { train: Arc::new(train), val: Arc::new(val), classes: cfg.classes }
}

/// 16 train images at 128 px, batch 4 (no accumulation), AdamW without warmup or augmentation, on `gpu:0`: the
/// whole run stays on the device and the epoch loss falls (as `train.rs`'s CPU micro run).
#[test]
fn tiny_gpu_run_learns() {
    let Some(dev) = gpu() else { return };
    let cfg = TrainConfig {
        epochs: 10,
        batch: 4,
        nbs: 4,
        imgsz: 128,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        device: "gpu:0".into(),
        ..TrainConfig::default()
    };
    let mut t = Trainer::new(cfg, data(), TrainOptions::default()).unwrap();
    assert_eq!(t.device(), dev);
    assert!(t.model().named_tensors().iter().all(|n| n.tensor.device() == dev), "model on the device");
    assert!(
        t.ema_model().unwrap().named_tensors().iter().all(|n| n.tensor.device() == dev),
        "EMA on the device"
    );
    let s = t.run().unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Completed, 10));
    let losses: Vec<f32> = s.epochs.iter().map(|e| e.train_loss.total).collect();
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    let tail = losses[losses.len() - 3..].iter().sum::<f32>() / 3.0;
    assert!(tail < 0.6 * losses[0], "the loss falls: {losses:?}");
    assert!(s.epochs.iter().all(|e| e.val.map50.is_finite()));
    assert!(t.nonfinite_streak() == 0);
    // Every kernel the steps and validation used can be named in a kernel list and precompiled next time.
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    assert_eq!(ctx.kernels_without_registry(), Vec::<String>::new());

    // The checkpoint reloads on the CPU.
    let last = TestFile::new("gpu-train", "last.folo");
    t.save_checkpoint(&last).unwrap();
    assert_eq!(ModelFile::open(&last).unwrap().checkpoint().unwrap().epoch, 10);
    let cpu = folo_vision::model::FoloModel::open(&last).unwrap();
    assert!(cpu.named_tensors().iter().all(|n| n.tensor.device().is_cpu()));
}

/// Every tensor of a `.folo` file (weights, EMA, optimizer state) as raw `f32` / `i64` bytes, by name.
fn tensors(path: &std::path::Path) -> Vec<(String, Vec<u8>)> {
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

/// A GPU run stopped at an epoch boundary and resumed onto the GPU ends in the same state as an uninterrupted GPU
/// run, bit for bit: the resumed optimizer, EMA, schedule and BatchNorm statistics continue exactly (the device
/// kernels are deterministic: fixed reduction orders, integer atomics only).
#[test]
fn stopped_gpu_run_resumes_bit_exactly() {
    let Some(dev) = gpu() else { return };
    let root = TestDir::new("gpu-resume");
    let cfg = TrainConfig {
        epochs: 2,
        batch: 4,
        nbs: 8,
        imgsz: 128,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 1.0,
        workers: 2,
        augment: AugmentConfig { close_mosaic: 1, ..AugmentConfig::default() },
        device: "gpu:0".into(),
        ..TrainConfig::default()
    };
    let mut a = Trainer::new(cfg.clone(), data(), TrainOptions::default()).unwrap();
    assert_eq!(a.run().unwrap().epochs_completed, 2);
    let last_a = root.join("a.folo");
    a.save_checkpoint(&last_a).unwrap();

    // B: stopped after epoch 1 (the stop flag set on its epoch event), then resumed onto the GPU.
    let ob = TrainOptions::default();
    let stop = ob.stop.clone();
    let ob = TrainOptions {
        events: Some(Box::new(move |e| {
            if matches!(e, RunEvent::Epoch(x) if x.epoch == 1) {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        })),
        ..ob
    };
    let mut b = Trainer::new(cfg, data(), ob).unwrap();
    let sb = b.run().unwrap();
    assert_eq!((sb.status, sb.epochs_completed), (RunStatus::Stopped, 1));
    let ckpt = root.join("b1.folo");
    b.save_checkpoint(&ckpt).unwrap();
    let over = ResumeOverrides { device: Some("gpu:0".into()), ..Default::default() };
    let mut r = Trainer::resume(&ckpt, data(), TrainOptions::default(), over).unwrap();
    assert_eq!((r.device(), r.epochs_completed()), (dev, 1));
    assert_eq!(r.run().unwrap().epochs_completed, 2);
    let last_b = root.join("b2.folo");
    r.save_checkpoint(&last_b).unwrap();

    let (ta, tb) = (tensors(&last_a), tensors(&last_b));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Pipelined GPU f32 training with dual `o2o` head, partial final accumulation window (`nb = 3`, `acc = 2`),
/// and `folo2-n` produces finite losses, exact window step counts, and bit-exact epoch resume.
#[test]
fn pipelined_f32_o2o_and_partial_window_resume_bit_exactly() {
    let Some(dev) = gpu() else { return };
    let root = TestDir::new("gpu-f32-partial");
    let ds_cfg = SyntheticConfig {
        images: 16,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 128,
        seed: 11,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 4.0 / 16.0,
        ..SyntheticConfig::default()
    };
    let td = TrainData {
        train: Arc::new(SyntheticDataset::new(ds_cfg.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(ds_cfg.clone(), Split::Val).unwrap()),
        classes: ds_cfg.classes,
    };
    let cfg = TrainConfig {
        epochs: 2,
        batch: 4,
        nbs: 8,
        imgsz: 128,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        device: "gpu:0".into(),
        head: folo_vision::model::HeadMode::O2o,
        ..TrainConfig::default()
    };

    let mut a = Trainer::new(cfg.clone(), td.clone(), TrainOptions::default()).unwrap();
    assert_eq!((a.device(), a.batches_per_epoch(), a.accumulate()), (dev, 3, 2));
    let sa = a.run().unwrap();
    assert_eq!(sa.epochs_completed, 2);
    assert_eq!(a.schedule().step(), 4);
    assert!(sa.epochs.iter().all(|e| e.train_loss.total.is_finite()));
    let pa = root.join("a.folo");
    a.save_checkpoint(&pa).unwrap();

    let ob = TrainOptions::default();
    let stop = ob.stop.clone();
    let ob = TrainOptions {
        events: Some(Box::new(move |e| {
            if matches!(e, RunEvent::Epoch(x) if x.epoch == 1) {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        })),
        ..ob
    };
    let mut b = Trainer::new(cfg.clone(), td.clone(), ob).unwrap();
    assert_eq!(b.run().unwrap().epochs_completed, 1);
    let ckpt = root.join("b1.folo");
    b.save_checkpoint(&ckpt).unwrap();
    let over = ResumeOverrides { device: Some("gpu:0".into()), ..Default::default() };
    let mut r = Trainer::resume(&ckpt, td.clone(), TrainOptions::default(), over).unwrap();
    assert_eq!(r.run().unwrap().epochs_completed, 2);
    let pb = root.join("b2.folo");
    r.save_checkpoint(&pb).unwrap();
    assert_eq!(tensors(&pa), tensors(&pb));

    // Also verify `folo2-n` f32 pipelined training with partial-window flush.
    let v2_cfg =
        TrainConfig { arch: "folo2-n".into(), epochs: 1, head: folo_vision::model::HeadMode::O2m, ..cfg };
    let mut v2 = Trainer::new(v2_cfg, td, TrainOptions::default()).unwrap();
    let sv2 = v2.run().unwrap();
    assert_eq!(sv2.epochs_completed, 1);
    assert_eq!(v2.schedule().step(), 2);
    assert!(sv2.epochs[0].train_loss.total.is_finite());
}

/// Pipelined GPU non-finite batch recovery (`can_pipeline == true`):
/// 1. Poisoning the window-ending forward inside a live accumulation window (`acc = 2`) restores BN stats,
///    keeps `GradAccumulator` isolated to batch 1's clean gradients, steps the optimizer once cleanly, and
///    recovers when the callback restores the poisoned weight.
/// 2. Freezing every parameter while poisoning a forward (making speculative `backward()` return an error)
///    still resolves the probes first, restores BN stats, and invokes the warn handler rather than aborting.
/// 3. Consecutive non-finite batches hit `MAX_NONFINITE_BATCHES` with BN stats intact and zero optimizer steps.
#[test]
fn pipelined_gpu_nonfinite_mid_window_recovery_and_max_streak() {
    use folo_core::no_grad;
    use folo_nn::RunningStats;
    use std::sync::Mutex;

    let Some(_dev) = gpu() else { return };
    // 1. Mid-window rejection (batch 1 finite, batch 2 poisoned -> window-end optimizer step uses batch 1 only).
    let ds_cfg = SyntheticConfig {
        images: 12,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 128,
        seed: 13,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 4.0 / 12.0,
        ..SyntheticConfig::default()
    };
    let td = TrainData {
        train: Arc::new(SyntheticDataset::new(ds_cfg.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(ds_cfg.clone(), Split::Val).unwrap()),
        classes: ds_cfg.classes,
    };
    assert_eq!(td.train.len(), 8); // batch 4, nbs 8 -> nb = 2, acc = 2 (1 window of 2 batches).

    let stem_slot: Arc<Mutex<Option<(folo_core::Tensor, folo_core::Tensor)>>> = Arc::default();
    let warns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let slot_cb = stem_slot.clone();
    let warns_cb = warns.clone();
    let cfg = TrainConfig {
        epochs: 1,
        batch: 4,
        nbs: 8,
        imgsz: 128,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        device: "gpu:0".into(),
        ..TrainConfig::default()
    };
    let opts = TrainOptions {
        step_interval: std::time::Duration::ZERO,
        events: Some(Box::new(move |ev| match ev {
            RunEvent::Step(s) if s.step == 1 => {
                // Batch 1 is finite and accumulated; poison batch 2 (window-ending forward).
                if let Some((w, _)) = slot_cb.lock().unwrap().as_ref() {
                    let _g = no_grad();
                    w.fill_assign(f32::NAN).unwrap();
                }
            }
            RunEvent::Log(l) if l.message.contains("non-finite loss") => {
                warns_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if let Some((w, saved)) = slot_cb.lock().unwrap().as_ref() {
                    let _g = no_grad();
                    w.copy_from(saved).unwrap();
                }
            }
            _ => {}
        })),
        ..TrainOptions::default()
    };
    let mut t = Trainer::new(cfg.clone(), td.clone(), opts).unwrap();
    {
        let _g = no_grad();
        let w = t.model().backbone.stem.conv.weight.clone();
        let saved = w.mul_scalar(1.0).unwrap();
        *stem_slot.lock().unwrap() = Some((w, saved));
    }
    let s = t.run().unwrap();
    assert_eq!(s.epochs_completed, 1);
    assert_eq!(warns.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(t.schedule().step(), 1, "window must step once using batch 1's clean gradients");
    assert_eq!(t.nonfinite_streak(), 1);
    assert!(RunningStats::all_finite(t.model()).unwrap());
    assert!(t.model().parameters().iter().all(|p| p.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite())));

    // 2. Speculative backward error on a confirmed bad batch (all params temporarily frozen + poisoned weight):
    // must reach the warn callback and recover cleanly rather than returning a backward error.
    type FreezeState = (folo_core::Tensor, folo_core::Tensor, Vec<folo_core::Tensor>);
    let freeze_slot: Arc<Mutex<Option<FreezeState>>> = Arc::default();
    let fcb = freeze_slot.clone();
    let fwarns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fwarns_cb = fwarns.clone();
    let opts_freeze = TrainOptions {
        step_interval: std::time::Duration::ZERO,
        events: Some(Box::new(move |ev| {
            if let RunEvent::Log(l) = ev
                && l.message.contains("non-finite loss")
            {
                fwarns_cb.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if let Some((w, saved, ps)) = fcb.lock().unwrap().as_ref() {
                    let _g = no_grad();
                    w.copy_from(saved).unwrap();
                    for p in ps {
                        p.set_requires_grad(true);
                    }
                }
            }
        })),
        ..TrainOptions::default()
    };
    let mut tf = Trainer::new(cfg.clone(), td, opts_freeze).unwrap();
    {
        let _g = no_grad();
        let ps = tf.model().parameters();
        let w = tf.model().backbone.stem.conv.weight.clone();
        let saved = w.mul_scalar(1.0).unwrap();
        w.fill_assign(f32::NAN).unwrap();
        for p in &ps {
            p.set_requires_grad(false);
        }
        *freeze_slot.lock().unwrap() = Some((w, saved, ps));
    }
    let sf = tf.run().unwrap();
    assert_eq!(sf.epochs_completed, 1);
    assert_eq!(fwarns.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(tf.schedule().step(), 1, "batch 2 recovers and steps cleanly after unfreezing");
    assert!(RunningStats::all_finite(tf.model()).unwrap());

    // 3. `MAX_NONFINITE_BATCHES` consecutive rejected batches through the pipelined GPU path.
    let streak_ds = SyntheticConfig {
        images: 16,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 15,
        min_objects: 1,
        max_objects: 2,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 4.0 / 16.0,
        ..SyntheticConfig::default()
    };
    let td_streak = TrainData {
        train: Arc::new(SyntheticDataset::new(streak_ds.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(streak_ds.clone(), Split::Val).unwrap()),
        classes: streak_ds.classes,
    };
    let mut ts =
        Trainer::new(TrainConfig { batch: 1, nbs: 2, imgsz: 64, ..cfg }, td_streak, TrainOptions::default())
            .unwrap();
    let bn_before: Vec<Vec<f32>> =
        ts.model().named_buffers().iter().map(|(_, b)| b.to_vec::<f32>().unwrap()).collect();
    {
        let _g = no_grad();
        ts.model().backbone.stem.conv.weight.fill_assign(f32::NAN).unwrap();
    }
    let err = ts.run().unwrap_err();
    assert_eq!(err.code(), "training_failed", "{err}");
    assert_eq!(ts.nonfinite_streak(), folo_vision::train::MAX_NONFINITE_BATCHES);
    assert_eq!(ts.schedule().step(), 0);
    assert!(RunningStats::all_finite(ts.model()).unwrap());
    let bn_after: Vec<Vec<f32>> =
        ts.model().named_buffers().iter().map(|(_, b)| b.to_vec::<f32>().unwrap()).collect();
    assert_eq!(bn_before, bn_after);
}
