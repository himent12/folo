//! Unit tests of the trainer's internals on a micro run: the prefetched epoch, chunked downloads, the ETA by
//! mosaic phase, `close_mosaic` across resumes, lost-device messages and early stopping.

use std::path::Path;

use folo_api::RunEvent;

use super::checkpoint::{DOWNLOAD_CHUNK, download_all_chunked};
use super::device::{device_loss_message, device_lost, failure_log_line};
use super::epoch::{Clock, eta_s};
use super::*;
use crate::error::Error;

/// A trainer on 8 synthetic 64 px training images (2 batches of 4), on the CPU.
pub(super) fn micro(epochs: u32, patience: u32, opts: TrainOptions) -> Trainer {
    micro_on("cpu", epochs, patience, opts)
}

/// [`micro`] on `device`.
pub(super) fn micro_on(device: &str, epochs: u32, patience: u32, opts: TrainOptions) -> Trainer {
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    let shapes = SyntheticConfig {
        images: 10,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 3,
        val_fraction: 0.2,
        ..SyntheticConfig::default()
    };
    let split = |s| Arc::new(SyntheticDataset::new(shapes.clone(), s).unwrap());
    let data = TrainData {
        train: split(folo_data::Split::Train),
        val: split(folo_data::Split::Val),
        classes: shapes.classes.clone(),
    };
    let cfg = TrainConfig {
        epochs,
        patience,
        batch: 4,
        imgsz: 64,
        device: device.into(),
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        warmup_epochs: 0.0,
        workers: 2,
        augment: folo_data::AugmentConfig::none(),
        ..TrainConfig::default()
    };
    Trainer::new(cfg, data, opts).unwrap()
}

/// The next epoch, started while the previous one validates, does not outlive the run: after a stop requested
/// at an epoch's end, an early stop and a completed run, nothing is prefetched, so its loader workers and
/// feeder are joined (dropping an epoch joins them).
#[test]
fn prefetched_epoch_does_not_outlive_the_run() {
    // Stopped: the stop arrives with epoch 1's event, after epoch 2 was prefetched.
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let opts = TrainOptions {
        events: Some(Box::new(move |e: &RunEvent| {
            if matches!(e, RunEvent::Epoch(_)) {
                flag.store(true, Ordering::Relaxed);
            }
        })),
        stop,
        ..TrainOptions::default()
    };
    let mut t = micro(3, 100, opts);
    let s = t.run().unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Stopped, 1));
    assert!(t.prefetched.is_none(), "stopped run");
    // Early stop (patience 0: after epoch 1), with epoch 2 prefetched.
    let mut t = micro(3, 0, TrainOptions::default());
    let s = t.run().unwrap();
    assert!(s.early_stopped && s.epochs_completed == 1);
    assert!(t.prefetched.is_none(), "early-stopped run");
    // Completed: the last epoch prefetches nothing.
    let mut t = micro(2, 100, TrainOptions::default());
    let s = t.run().unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Completed, 2));
    assert!(t.prefetched.is_none(), "completed run");
}

/// The batched checkpoint download gives every tensor's exact bytes and shape: several dtypes, a view that is
/// not contiguous, an empty tensor and one already on the host; in one chunk per dtype, and in chunks so small
/// that some hold one tensor, some several, and one tensor is larger than a chunk.
#[test]
fn download_all_keeps_every_byte() {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping GPU test, no adapter: {e}"),
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        return eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
    }
    let dev = folo_gpu::init().unwrap();
    let mut rng = folo_core::rng::Rng::seed(9);
    let a = folo_core::Tensor::rand_uniform([3, 5, 7], -1.0, 1.0, &mut rng);
    let b = folo_core::Tensor::rand_uniform([4, 6], 0.0, 9.0, &mut rng);
    let host = vec![
        ("a".to_string(), a.clone()),
        ("b".to_string(), b.clone()),
        ("bt".to_string(), b.transpose(0, 1).unwrap()),
        ("n".to_string(), folo_core::Tensor::from_vec((0..11u32).collect::<Vec<_>>(), [11]).unwrap()),
        ("e".to_string(), folo_core::Tensor::from_vec(Vec::<f32>::new(), [0, 3]).unwrap()),
        ("c".to_string(), folo_core::Tensor::rand_uniform([2], 0.0, 1.0, &mut rng)),
    ];
    let mut tensors: Vec<(String, folo_core::Tensor)> = host
        .iter()
        .map(|(n, t)| (n.clone(), if n == "c" { t.clone() } else { t.to_device(dev).unwrap() }))
        .collect();
    // A transposed (strided) view on the device.
    tensors[2].1 = tensors[1].1.transpose(0, 1).unwrap();
    // 400 bytes: `a` (420) alone and over the limit, `b` + `bt` (96 + 96) together, `n` (44).
    for chunk in [DOWNLOAD_CHUNK, 400, 1] {
        let mut tensors = tensors.clone();
        download_all_chunked(&mut tensors, chunk).unwrap();
        for ((n, got), (_, want)) in tensors.iter().zip(&host) {
            assert!(got.device().is_cpu(), "{n}, chunk {chunk}");
            assert_eq!(got.dims(), want.dims(), "{n}, chunk {chunk}");
            assert_eq!(got.dtype(), want.dtype(), "{n}, chunk {chunk}");
            let (g, w) = (got.to_cpu_storage().unwrap(), want.to_cpu_storage().unwrap());
            assert_eq!(g.as_bytes(), w.as_bytes(), "{n}, chunk {chunk}");
        }
    }
}

/// The ETA counts the loader's wait and each epoch's work besides its batches and validation, and does not charge the
/// mosaic-free epochs at mosaic speed.
#[test]
fn eta_by_mosaic_phase() {
    // 10 epochs of 100 batches, the last 2 without mosaic. After epoch 0: 0.2 s per batch of wall time, of
    // which 0.08 s is the step (the rest is waiting for the loader), and a 1 s validation.
    let mut c = Clock { batches: [100, 0], wall: [20.0, 0.0], step: [8.0, 0.0], ..Clock::default() };
    c.ended_epoch(1.0, 0.0);
    let phase = |ep: u32| usize::from(ep >= 8);
    let eta = eta_s(&c, 100, 1, 0, 10, phase);
    // Epoch 1's 99 batches and epochs 2..8 at 0.2 s, epochs 8 and 9 at the step time, 9 validations.
    let want = 99.0 * 0.2 + 6.0 * 100.0 * 0.2 + 2.0 * 100.0 * 0.08 + 9.0;
    assert!((eta - want).abs() < 1e-9, "{eta} vs {want}");
    // Once the closed phase has run, its own wall time counts.
    c.batches[1] = 50;
    c.wall[1] = 5.0;
    c.step[1] = 4.0;
    let eta = eta_s(&c, 100, 8, 49, 10, phase);
    let want = 50.0 * 0.1 + 100.0 * 0.1 + 2.0;
    assert!((eta - want).abs() < 1e-9, "{eta} vs {want}");
    // Starting the loader and writing checkpoints and samples (0 s, then 1 s, then 2 s an epoch) counts for each epoch
    // left too. It and validation lean on the latest epochs (they grow as the model learns): validation 1, 1, 3 s → 2 s,
    // the other work 0, 1, 2 s → 1.25 s.
    c.ended_epoch(1.0, 1.0);
    c.ended_epoch(3.0, 2.0);
    let eta = eta_s(&c, 100, 8, 49, 10, phase);
    let want = 50.0 * 0.1 + 100.0 * 0.1 + 2.0 * (2.0 + 1.25);
    assert!((eta - want).abs() < 1e-9, "{eta} vs {want}");
    // No mosaic at all (phase 1 throughout), and nothing measured yet.
    let c = Clock::default();
    assert_eq!(eta_s(&c, 100, 0, 0, 10, |_| 1), 0.0);
}

/// A resumed run keeps the closing epochs its checkpoint records; one from before the cap keeps the old rule.
#[test]
fn resumes_keep_their_close_mosaic() {
    let cfg = TrainConfig { epochs: 8, ..TrainConfig::default() };
    let old = serde_json::to_value(&cfg).unwrap();
    assert_eq!(resumed_close_mosaic(&old, &cfg), 8, "close_mosaic 10 of 8 epochs: no mosaic at all");
    let mut new = old.clone();
    new[CLOSE_MOSAIC_KEY] = serde_json::json!(cfg.augment.close_mosaic_epochs(cfg.epochs));
    assert_eq!(resumed_close_mosaic(&new, &cfg), 4);
    // The key is not a TrainConfig field: folo.train with it still parses.
    assert!(serde_json::from_value::<TrainConfig>(new).is_ok());
}

/// A lost GPU device is recognised wherever it surfaces (tensor work, a layer, the batch feeder) and the
/// message names the checkpoint that was actually written; other errors are left alone.
#[test]
fn device_loss_is_explained() {
    let lost = || folo_core::Error::from(folo_gpu::GpuError::DeviceLost("Destroyed".into()));
    let other = folo_core::Error::from(folo_gpu::GpuError::Wgpu("validation".into()));
    assert!(device_lost(&Error::Core(lost())));
    assert!(device_lost(&Error::Nn(folo_nn::Error::Core(lost()))));
    assert!(!device_lost(&Error::Core(other)));
    assert!(!device_lost(&Error::Failed("GPU device lost".into())));
    let p = Path::new("runs/shapes/weights/last.folo");
    let m = device_loss_message(&Error::Core(lost()), Some((p, 266)));
    assert!(m.contains("GPU device lost") && m.contains("after epoch 266"), "{m}");
    assert!(m.contains(&format!("--resume {}", p.display())), "{m}");
    let m = device_loss_message(&Error::Core(lost()), None);
    assert!(m.contains("no checkpoint was written yet"), "{m}");
}

/// A failed run's log line cuts its message to 120 characters, but a lost GPU's keeps the command to resume with
/// whole, however long the driver's report (#382: the line was cut inside the command).
#[test]
fn a_lost_gpus_log_line_ends_in_the_resume_command() {
    let report = folo_core::Error::from(folo_gpu::GpuError::DeviceLost("x".repeat(40_000)));
    let p = Path::new("D:/projects/one long folder name/runs/shapes-folo-s-640-batch-16/weights/last.folo");
    let e = Error::Failed(device_loss_message(&Error::Core(report), Some((p, 266))));
    let line = failure_log_line(&e, true);
    let command =
        format!("resume from the state after epoch 266 with `folo train --resume \"{}\"`", p.display());
    assert!(
        line.starts_with("run failed (training_failed): training failed: the GPU device was lost ("),
        "{line}"
    );
    assert!(line.ends_with(&command), "{line}");
    assert!(line.chars().count() < 400, "the driver's report is cut: {line}");
    assert_eq!(e.summary(), e.to_string(), "the status message is whole too");
    // Any other failure names itself in at most 120 characters (and an ellipsis).
    let other = failure_log_line(&Error::Failed("y".repeat(300)), false);
    assert_eq!(other, format!("run failed (training_failed): training failed: {}…", "y".repeat(120 - 17)));
}

/// MODEL.md §10.4: patience 2 with fitness [0.1, 0.3, 0.2, 0.2] stops after epoch 3 (best epoch 1), and a
/// resume from the epoch-2 state (the `folo.best` a checkpoint persists) stops at the same epoch.
#[test]
fn early_stopping() {
    let fit = [0.1, 0.3, 0.2, 0.2, 0.9];
    let run = |best: &mut BestState, from: u32| {
        for e in from..fit.len() as u32 {
            if update_best(best, e, fit[e as usize], 2).1 {
                return e;
            }
        }
        u32::MAX
    };
    let mut best = BestState::default();
    assert_eq!(run(&mut best, 0), 3);
    assert_eq!(best, BestState { epoch: 1, fitness: Some(0.3) });
    let mut at_epoch_2 = BestState::default();
    for e in 0..3 {
        update_best(&mut at_epoch_2, e, fit[e as usize], 2);
    }
    assert_eq!(run(&mut at_epoch_2, 3), 3);
    // An equal fitness is not a gain.
    let mut b = BestState { epoch: 0, fitness: Some(0.5) };
    assert_eq!(update_best(&mut b, 1, 0.5, 10), (false, false));
}
