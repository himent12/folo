//! The trainer on the GPU: a tiny run whose loss falls, with the model, optimizer state, EMA and gradient
//! buffers on the device, checkpoints that reload on the CPU, and a resume that continues on the device.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.

use std::sync::{Arc, Mutex};

use folo_api::{RunEvent, RunStatus};
use folo_core::Device;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::Module;
use folo_nn::io::ModelFile;
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

fn epochs_of(log: &Mutex<Vec<RunEvent>>) -> Vec<folo_api::Epoch> {
    log.lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Epoch(x) = e { Some(x.clone()) } else { None })
        .collect()
}

/// 16 train images at 128 px, batch 4 (no accumulation), AdamW without warmup or augmentation, on `gpu:0`: the
/// whole run stays on the device and the epoch loss falls (as `train.rs`'s CPU micro run).
#[test]
fn tiny_gpu_run_learns_and_resumes() {
    let Some(dev) = gpu() else { return };
    let dir = std::env::temp_dir().join(format!("folo-gpu-train-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let log: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = log.clone();
    let opts = TrainOptions {
        run_dir: Some(dir.clone()),
        events: Some(Box::new(move |e| sink.lock().unwrap().push(e.clone()))),
        step_interval: std::time::Duration::ZERO,
        ..TrainOptions::default()
    };
    let mut t = Trainer::new(cfg, data(), opts).unwrap();
    assert_eq!(t.device(), dev);
    assert!(t.model().named_tensors().iter().all(|n| n.tensor.device() == dev), "model on the device");
    assert!(
        t.ema_model().unwrap().named_tensors().iter().all(|n| n.tensor.device() == dev),
        "EMA on the device"
    );
    let s = t.run().unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Completed, 10));
    let epochs = epochs_of(&log);
    let losses: Vec<f32> = epochs.iter().map(|e| e.train_loss.total).collect();
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    let tail = losses[losses.len() - 3..].iter().sum::<f32>() / 3.0;
    assert!(tail < 0.6 * losses[0], "the loss falls: {losses:?}");
    assert!(epochs.iter().all(|e| e.val.map50.is_finite()));
    assert!(t.nonfinite_streak() == 0);

    // The checkpoint reloads on the CPU, and resumes onto the device.
    let last = dir.join("weights/last.folo");
    let file = ModelFile::open(&last).unwrap();
    assert_eq!(file.checkpoint().unwrap().epoch, 10);
    let cpu = folo_vision::model::FoloModel::open(&last).unwrap();
    assert!(cpu.named_tensors().iter().all(|n| n.tensor.device().is_cpu()));
    drop(t);
    let opts = TrainOptions { run_dir: Some(dir.clone()), ..TrainOptions::default() };
    let mut r = Trainer::resume(
        &last,
        data(),
        opts,
        ResumeOverrides { device: Some("gpu:0".into()), ..Default::default() },
    )
    .unwrap();
    assert_eq!((r.device(), r.epochs_completed()), (dev, 10));
    // All epochs are done: the resumed run completes without training.
    assert_eq!(r.run().unwrap().epochs_completed, 10);
    let _ = std::fs::remove_dir_all(&dir);
}
