//! Deterministic analytic/mock tests and hardware-guarded live GPU tests for the wired GPU auto-batch prober
//! ([`folo_vision::train::auto_batch`]) and its [`Trainer`] integration.

use std::sync::Arc;

use folo_core::Device;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_gpu::{MemoryBudget, MemoryLimits, MemoryUsage, PoolStats};
use folo_nn::Module;
use folo_nn::optim::OptimKind;
use folo_test_support::TestDir;
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::train::{
    AutoBatchRequest, BatchLimitFactor, DEFAULT_AUTO_BATCH_UTILIZATION, MAX_AUTO_BATCH_UTILIZATION,
    MIN_AUTO_BATCH_UTILIZATION, ResumeOverrides, TrainConfig, TrainData, TrainOptions, Trainer,
    auto_batch_plan, estimate_training_footprint, probe_batch_size,
};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn base_req(arch: &str) -> AutoBatchRequest {
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
        teacher_arch: None,
        teacher_params: None,
    }
}

#[test]
fn analytic_param_count_matches_folo_model_across_all_scales_and_heads() {
    for scale in ["n", "s", "m", "l", "x"] {
        for (head, nc) in [(HeadMode::O2m, 80), (HeadMode::O2o, 4)] {
            let mut mconf = ModelConfig::new(scale, nc).unwrap();
            mconf.head = head.as_str().into();
            let model = FoloModel::new(mconf).unwrap();
            let req =
                AutoBatchRequest { arch: format!("folo-{scale}"), num_classes: nc, head, ..base_req(scale) };
            let fp = estimate_training_footprint(&req).unwrap();
            assert_eq!(
                fp.model_params,
                model.num_parameters(),
                "param count mismatch for folo-{scale} head={head:?} nc={nc}"
            );
            assert!(fp.fixed_bytes > (fp.model_params as u64) * 4);
            assert!(fp.per_sample_bytes > fp.max_single_tensor_per_sample);
            assert!(fp.max_single_tensor_per_sample > 0);
        }
    }
}

#[test]
fn footprint_scales_monotonically_with_amp_ema_optimizer_o2o_and_teacher() {
    let f32_fp = estimate_training_footprint(&base_req("folo-s")).unwrap();
    let amp_fp = estimate_training_footprint(&AutoBatchRequest { amp: true, ..base_req("folo-s") }).unwrap();
    assert!(
        amp_fp.per_sample_bytes < f32_fp.per_sample_bytes,
        "AMP should reduce per-sample activation bytes ({} vs {})",
        amp_fp.per_sample_bytes,
        f32_fp.per_sample_bytes
    );

    let no_ema = estimate_training_footprint(&AutoBatchRequest { ema: false, ..base_req("folo-s") }).unwrap();
    assert!(no_ema.fixed_bytes < f32_fp.fixed_bytes, "EMA adds resident shadow weights");

    let adamw =
        estimate_training_footprint(&AutoBatchRequest { optimizer: OptimKind::AdamW, ..base_req("folo-s") })
            .unwrap();
    assert!(adamw.fixed_bytes > f32_fp.fixed_bytes, "AdamW holds 2x optimizer state vs MuSGD/SGD");

    let o2o =
        estimate_training_footprint(&AutoBatchRequest { head: HeadMode::O2o, ..base_req("folo-s") }).unwrap();
    assert!(o2o.model_params > f32_fp.model_params);
    assert!(o2o.per_sample_bytes > f32_fp.per_sample_bytes);

    let distill = estimate_training_footprint(&AutoBatchRequest {
        teacher_arch: Some("folo-m".into()),
        ..base_req("folo-s")
    })
    .unwrap();
    assert!(distill.fixed_bytes > f32_fp.fixed_bytes, "frozen teacher weights add to fixed bytes");
    assert!(distill.per_sample_bytes > f32_fp.per_sample_bytes, "teacher outputs stay live per batch");
    assert!(
        distill.max_single_tensor_per_sample >= f32_fp.max_single_tensor_per_sample,
        "larger teacher peak single tensor is accounted for"
    );
}

#[test]
fn auto_batch_plan_handles_vram_buffer_limits_reclaim_and_grad_accum_pairing() {
    let req = base_req("folo-n");
    let fp = estimate_training_footprint(&req).unwrap();

    // 1. Abundant VRAM: grows from requested 16 up to nominal 64 (grad_accum = 1).
    let abundant =
        MemoryBudget::new(MemoryLimits::new(Some(24 * GIB), 4 * GIB, 2 * GIB), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &abundant).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 64);
    assert_eq!(plan.grad_accum, 1);
    assert_eq!(plan.effective_batch, 64);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::RequestedBatch);
    assert!(!plan.needs_reclaim);

    // 2. Tight VRAM fitting exactly 8 samples at 80% utilization: clamps to 8 and pairs grad_accum = 8.
    let target_free = fp.fixed_bytes + 8 * fp.per_sample_bytes + fp.per_sample_bytes / 2;
    let dedicated_for_8 = ((target_free as f64) / 0.80).ceil() as u64;
    let tight_vram =
        MemoryBudget::new(MemoryLimits::new(Some(dedicated_for_8), 4 * GIB, 2 * GIB), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &tight_vram).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 8);
    assert_eq!(plan.nominal_batch, 64);
    assert_eq!(plan.grad_accum, 8);
    assert_eq!(plan.effective_batch, 64);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::VramBudget);

    // 3. Clamping below `requested_batch` when `requested_batch > nominal_batch`: preserves requested effective batch.
    let big_req = AutoBatchRequest { requested_batch: 128, nominal_batch: 64, ..req.clone() };
    let plan = auto_batch_plan(&big_req, &tight_vram).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 8);
    assert_eq!(plan.nominal_batch, 128);
    assert_eq!(plan.grad_accum, 16);
    assert_eq!(plan.effective_batch, 128);

    // 4. Per-buffer `max_storage_buffer_binding_size` limit (< max_buffer_size) caps batch to 4 despite huge VRAM.
    let s_limit = 4 * fp.max_single_tensor_per_sample + 1024;
    let storage_bound =
        MemoryBudget::new(MemoryLimits::new(Some(64 * GIB), 4 * GIB, s_limit), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &storage_bound).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 4);
    assert_eq!(plan.grad_accum, 16);
    assert_eq!(plan.effective_batch, 64);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::StorageBindingLimit);

    // 5. Per-buffer `max_buffer_size` limit (== storage binding limit) caps batch to 2 with MaxBufferSize factor.
    let b_limit = 2 * fp.max_single_tensor_per_sample + 1024;
    let max_buf_bound =
        MemoryBudget::new(MemoryLimits::new(Some(64 * GIB), b_limit, b_limit), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &max_buf_bound).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 2);
    assert_eq!(plan.grad_accum, 32);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::MaxBufferSize);

    // 6. Unknown VRAM (DX12/Metal): keeps requested batch when within buffer limits, clamps when over buffer limit.
    let unknown_vram = MemoryBudget::new(MemoryLimits::new(None, 4 * GIB, 2 * GIB), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &unknown_vram).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 16);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::DeviceUnbounded);
    let unknown_tight_buf =
        MemoryBudget::new(MemoryLimits::new(None, 4 * GIB, s_limit), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &unknown_tight_buf).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 4);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::StorageBindingLimit);

    // 7. Reclaimable pool cache occupying headroom triggers `needs_reclaim = true`.
    let cached = dedicated_for_8 / 2;
    let usage_with_cache = MemoryUsage::from_pool(PoolStats {
        live_bytes: 0,
        cached_bytes: cached,
        peak_live_bytes: 0,
        peak_held_bytes: cached,
        ..PoolStats::default()
    });
    let reclaim_budget =
        MemoryBudget::new(MemoryLimits::new(Some(dedicated_for_8), 4 * GIB, 2 * GIB), usage_with_cache);
    let plan = auto_batch_plan(&req, &reclaim_budget).unwrap();
    assert!(plan.fits);
    assert_eq!(plan.chosen_batch, 8);
    assert!(plan.needs_reclaim, "cached pool bytes should flag needs_reclaim");
    assert_eq!(plan.reclaimable_bytes, cached);
    assert!(plan.summary_line().contains("reclaiming"), "{}", plan.summary_line());

    // 8. Even batch 1 does not fit: returns `fits = false`, `chosen_batch = 0`, actionable summary.
    let tiny =
        MemoryBudget::new(MemoryLimits::new(Some(64 * MIB), 64 * MIB, 64 * MIB), MemoryUsage::default());
    let plan = auto_batch_plan(&req, &tiny).unwrap();
    assert!(!plan.fits);
    assert_eq!(plan.chosen_batch, 0);
    assert_eq!(plan.grad_accum, 0);
    assert_eq!(plan.limiting_factor, BatchLimitFactor::VramBudget);
    assert!(plan.summary_line().contains("batch 1 does not fit"), "{}", plan.summary_line());
}

#[test]
fn validation_and_config_serde_backwards_compatibility() {
    for bad in [
        AutoBatchRequest { arch: "folo-z".into(), ..base_req("folo-n") },
        AutoBatchRequest { imgsz: 100, ..base_req("folo-n") },
        AutoBatchRequest { num_classes: 0, ..base_req("folo-n") },
        AutoBatchRequest { requested_batch: 0, ..base_req("folo-n") },
        AutoBatchRequest { nominal_batch: 0, ..base_req("folo-n") },
        AutoBatchRequest { target_utilization: MIN_AUTO_BATCH_UTILIZATION - 0.01, ..base_req("folo-n") },
        AutoBatchRequest { target_utilization: MAX_AUTO_BATCH_UTILIZATION + 0.01, ..base_req("folo-n") },
        AutoBatchRequest { target_utilization: f64::NAN, ..base_req("folo-n") },
        AutoBatchRequest { max_batch: Some(0), ..base_req("folo-n") },
        AutoBatchRequest { teacher_arch: Some("folo-q".into()), ..base_req("folo-n") },
    ] {
        assert!(estimate_training_footprint(&bad).is_err(), "expected error for {bad:?}");
    }

    // Legacy TrainConfig JSON without `auto_batch` / `auto_batch_target` deserializes cleanly.
    let legacy: TrainConfig = serde_json::from_str(r#"{"arch":"folo-n","epochs":10}"#).unwrap();
    assert!(!legacy.auto_batch);
    assert!((legacy.auto_batch_target - DEFAULT_AUTO_BATCH_UTILIZATION).abs() < 1e-12);

    assert!(TrainConfig { auto_batch_target: 0.1, ..TrainConfig::default() }.validate().is_err());
    assert!(TrainConfig { auto_batch_target: 0.99, ..TrainConfig::default() }.validate().is_err());
    assert!(TrainConfig { auto_batch_target: f64::NAN, ..TrainConfig::default() }.validate().is_err());
}

fn tiny_data() -> TrainData {
    let syn = SyntheticConfig {
        images: 4,
        imgsz: 64,
        classes: vec!["circle".into(), "square".into()],
        val_fraction: 0.5,
        seed: 7,
        ..SyntheticConfig::default()
    };
    TrainData {
        train: Arc::new(SyntheticDataset::new(syn.clone(), Split::Train).unwrap()),
        val: Arc::new(SyntheticDataset::new(syn, Split::Val).unwrap()),
        classes: vec!["circle".into(), "square".into()],
    }
}

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

#[test]
fn cpu_probe_returns_none_and_trainer_runs_and_resumes_cleanly() {
    assert!(probe_batch_size(Device::Cpu, &base_req("folo-n")).unwrap().is_none());

    let dir = TestDir::new("auto-batch-cpu");
    let cfg = TrainConfig {
        arch: "folo-n".into(),
        epochs: 1,
        batch: 2,
        nbs: 4,
        imgsz: 64,
        device: "cpu".into(),
        workers: 0,
        auto_batch: true,
        auto_batch_target: 0.80,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    let opts = TrainOptions { run_dir: Some(dir.to_path_buf()), ..TrainOptions::default() };
    let mut trainer = Trainer::new(cfg, tiny_data(), opts).unwrap();
    assert_eq!(trainer.config().batch, 2);
    assert!(trainer.auto_batch_plan().is_none());
    let summary = trainer.run().unwrap();
    assert_eq!(summary.epochs_completed, 1);

    let last = summary.last.expect("last.folo written");
    let resumed =
        Trainer::resume(&last, tiny_data(), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    assert_eq!(resumed.config().batch, 2);
    assert!(resumed.auto_batch_plan().is_none());
}

#[test]
fn gpu_probe_and_trainer_auto_batch_end_to_end() {
    let Some(dev) = gpu() else {
        return;
    };

    let req = AutoBatchRequest {
        imgsz: 128,
        num_classes: 2,
        requested_batch: 4,
        nominal_batch: 8,
        ..base_req("folo-n")
    };
    let plan = probe_batch_size(dev, &req).unwrap().expect("GPU context present");
    assert!(plan.fits, "folo-n at 128px should fit on GPU: {}", plan.summary_line());
    assert!(plan.chosen_batch >= 1);
    assert!(plan.grad_accum >= 1);
    assert!(plan.single_buffer_limit_bytes > 0);
    assert!(plan.per_sample_bytes > 0);

    // Wire end-to-end through `Trainer::new` with `auto_batch = true` on GPU, run 1 epoch, and verify resume.
    let dir = TestDir::new("auto-batch-gpu");
    let cfg = TrainConfig {
        arch: "folo-n".into(),
        epochs: 1,
        batch: 2,
        nbs: 4,
        imgsz: 64,
        device: dev.to_string(),
        workers: 0,
        auto_batch: true,
        auto_batch_target: 0.80,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    };
    let opts = TrainOptions { run_dir: Some(dir.to_path_buf()), ..TrainOptions::default() };
    let mut trainer = Trainer::new(cfg, tiny_data(), opts).unwrap();
    let t_plan = trainer.auto_batch_plan().cloned().expect("auto_batch_plan recorded on GPU");
    assert!(t_plan.fits);
    assert_eq!(trainer.config().batch, t_plan.chosen_batch);
    assert!(trainer.run_config().get("auto_batch").is_some());

    let summary = trainer.run().unwrap();
    assert_eq!(summary.epochs_completed, 1);

    let last = summary.last.expect("last.folo written");
    let resumed =
        Trainer::resume(&last, tiny_data(), TrainOptions::default(), ResumeOverrides::default()).unwrap();
    assert_eq!(resumed.config().batch, t_plan.chosen_batch);
    assert_eq!(resumed.accumulate(), t_plan.grad_accum);
}
