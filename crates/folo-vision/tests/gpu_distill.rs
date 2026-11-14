//! Distillation (docs/MODEL.md §8.9) on the GPU against the CPU reference: the teacher's targets, the term and its
//! gradients (also under autocast), the flip map, and a mixed-precision run with an f16 teacher.
//!
//! Tolerances: the term and its gradients are f32 compositions of softplus, sigmoid, log, log-softmax and row sums
//! whose GPU kernels round differently from the CPU's, so values agree to a relative 1e-5 and gradients to 1e-6 plus
//! 1e-4 of the largest one; the flip map only moves values, so it is exact.
//!
//! This test binary asks for tensor cores before the shared GPU context exists; it skips (with a note) without a
//! hardware adapter, and the mixed-precision run without tensor cores.

use std::sync::{Arc, Mutex};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::Module;
use folo_test_support::TestDir;
use folo_vision::decode::make_anchors;
use folo_vision::loss::distill::FlipMap;
use folo_vision::loss::{Consensus, DistillConfig, DistillTargets, TeacherView, distill_loss};
use folo_vision::train::OptimizerChoice;
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};

const R: usize = 16;

/// The GPU (tensor cores on when it has them) and whether it has them, or `None` without a hardware adapter.
fn gpu() -> Option<(Device, bool)> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    folo_gpu::request_tensor_cores();
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
    let dev = folo_gpu::init().ok()?;
    let tc = ctx.caps().coop_f16;
    if tc {
        ctx.set_tensor_cores(true);
    }
    Some((dev, tc))
}

macro_rules! gpu {
    () => {
        match gpu() {
            Some(d) => d,
            None => return,
        }
    };
}

fn raw(seed: u64, b: usize, nc: usize, a: usize) -> (Tensor, Tensor) {
    let mut rng = Rng::seed(seed);
    (Tensor::randn([b, 4 * R, a], 0.0, 2.0, &mut rng), Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng))
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6 + 1e-4 * scale, "{what} [{i}]: {g} vs {w} (largest {scale})");
    }
}

/// The term's values and both gradients, plus the targets, on `dev` (under autocast when `amp`).
fn term_on(dev: Device, amp: bool, temp: f32, consensus: Consensus) -> (f32, f32, Vec<Vec<f32>>) {
    let anchors = make_anchors(64, 96, &[8, 16, 32], 0.5);
    let (sb, sc) = raw(1, 2, 5, anchors.len());
    let (tb, tc) = raw(2, 2, 5, anchors.len());
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    let (z, x) = (on(&sb).requires_grad_(), on(&sc).requires_grad_());
    let _a = amp.then(folo_core::autocast);
    let mut view = TeacherView::new(&on(&tb), &on(&tc), R, temp).unwrap();
    if consensus == Consensus::Flip {
        let map = FlipMap::new(&anchors, 96, dev).unwrap();
        view = view.average(&view.mapped(&map).unwrap()).unwrap();
    }
    let t = DistillTargets::new(&view).unwrap();
    let cfg = DistillConfig { temperature: temp, consensus, ..DistillConfig::new("t") };
    let out = distill_loss(&z, &x, &t, 3.5, &cfg).unwrap();
    let g = out.total.backward().unwrap();
    let grads = [&x, &z].map(|v| g.get(v).unwrap().to_vec::<f32>().unwrap()).to_vec();
    let mut all = grads;
    all.push(t.cls_entropy.to_vec::<f32>().unwrap());
    all.push(t.box_weight.to_vec::<f32>().unwrap());
    (out.cls, out.dfl, all)
}

/// The term, its gradients and the targets on the GPU agree with the CPU, at T = 1 and 2, with one view and the
/// flip consensus, and under autocast (the term stays f32).
#[test]
fn term_and_gradients_match_the_cpu() {
    let (dev, _) = gpu!();
    for (temp, consensus) in [(1.0, Consensus::None), (2.0, Consensus::Flip)] {
        let (c0, d0, g0) = term_on(Device::Cpu, false, temp, consensus);
        assert!(c0 > 1e-2 && d0 > 1e-2, "the case has content: {c0} {d0}");
        for amp in [false, true] {
            let (c, d, g) = term_on(dev, amp, temp, consensus);
            let what = format!("T {temp} {} amp {amp}", consensus.as_str());
            assert!(
                (c - c0).abs() <= 1e-5 * c0 && (d - d0).abs() <= 1e-5 * d0,
                "{what}: {c} {d} vs {c0} {d0}"
            );
            for (k, name) in ["cls grad", "box grad", "cls entropy", "box weight"].iter().enumerate() {
                close(&g[k], &g0[k], &format!("{what} {name}"));
            }
        }
    }
}

/// Teacher outputs in f16 (the f16 teacher of a mixed-precision run) give the targets the CPU computes from the
/// same f16 values.
#[test]
fn f16_teacher_outputs_match_the_cpu() {
    let (dev, _) = gpu!();
    let (tb, tc) = raw(3, 1, 3, 126);
    let half = |t: &Tensor| t.to_dtype(DType::F16).unwrap();
    let cpu = TeacherView::new(&half(&tb), &half(&tc), R, 1.0).unwrap();
    let on = |t: &Tensor| half(t).to_device(dev).unwrap();
    let gpu = TeacherView::new(&on(&tb), &on(&tc), R, 1.0).unwrap();
    assert_eq!(gpu.cls_prob.dtype(), DType::F32);
    close(&gpu.cls_prob.to_vec::<f32>().unwrap(), &cpu.cls_prob.to_vec::<f32>().unwrap(), "cls");
    close(&gpu.box_prob.to_vec::<f32>().unwrap(), &cpu.box_prob.to_vec::<f32>().unwrap(), "box");
}

/// The flip map moves values exactly on the GPU: the input mirror (f32 and f16) and both output maps.
#[test]
fn flip_map_matches_the_cpu_exactly() {
    let (dev, _) = gpu!();
    let anchors = make_anchors(64, 96, &[8, 16, 32], 0.5);
    let (cpu, gpu) =
        (FlipMap::new(&anchors, 96, Device::Cpu).unwrap(), FlipMap::new(&anchors, 96, dev).unwrap());
    let (b, c) = raw(4, 2, 3, anchors.len());
    let x = Tensor::rand_uniform([2, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(5));
    let bits = |t: Tensor| t.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    assert_eq!(bits(gpu.map_box(&on(&b)).unwrap()), bits(cpu.map_box(&b).unwrap()));
    assert_eq!(bits(gpu.map_cls(&on(&c)).unwrap()), bits(cpu.map_cls(&c).unwrap()));
    assert_eq!(bits(gpu.flip_input(&on(&x)).unwrap()), bits(cpu.flip_input(&x).unwrap()));
    let h = x.to_dtype(DType::F16).unwrap();
    assert_eq!(bits(gpu.flip_input(&on(&h)).unwrap()), bits(cpu.flip_input(&h).unwrap()));
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
    let split = |s| Arc::new(SyntheticDataset::new(cfg.clone(), s).unwrap());
    TrainData { train: split(Split::Train), val: split(Split::Val), classes: cfg.classes.clone() }
}

/// A mixed-precision run on the GPU with an f16 flip-consensus teacher: it loads the teacher in f16, trains with
/// finite losses and weights, and reports both terms.
#[test]
fn amp_run_with_an_f16_teacher() {
    let (_, tc) = gpu!();
    if !tc {
        return eprintln!("note: skipping the mixed-precision run: no tensor cores");
    }
    let dir = TestDir::new("gpu-distill");
    let config = folo_nn::io::ModelConfig::new("n", 2).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(config, &mut Rng::seed(7)).unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    let teacher = dir.join("teacher.folo");
    m.save(&teacher).unwrap();
    // gpu_amp.rs's tiny run (16 images at 128 px, batch 4), one epoch.
    let cfg = TrainConfig {
        epochs: 1,
        batch: 4,
        nbs: 4,
        imgsz: 128,
        device: "gpu:0".into(),
        amp: true,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        distill: Some(DistillConfig {
            consensus: Consensus::Flip,
            ..DistillConfig::new(teacher.to_string_lossy())
        }),
        ..TrainConfig::default()
    };
    let log: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = log.clone();
    let opts = TrainOptions {
        events: Some(Box::new(move |e| {
            if let folo_api::RunEvent::Log(l) = e {
                sink.lock().unwrap().push(l.message.clone());
            }
        })),
        ..TrainOptions::default()
    };
    let mut t = Trainer::new(cfg, data(), opts).unwrap();
    let s = t.run().unwrap();
    assert!(s.epochs.iter().all(|e| e.train_loss.total.is_finite()));
    let w: Vec<Vec<f32>> =
        t.model().named_tensors().iter().filter_map(|n| n.tensor.to_vec::<f32>().ok()).collect();
    let log = log.lock().unwrap().clone();
    assert!(log.iter().any(|l| l.contains("distilling from") && l.contains("f16")), "{log:?}");
    assert!(log.iter().any(|l| l.contains("distillation cls")), "{log:?}");
    assert!(!log.iter().any(|l| l.contains("non-finite")), "{log:?}");
    assert!(w.iter().flatten().all(|v| v.is_finite()));
}
