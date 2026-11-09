//! Mixed-precision training (`amp`, docs/MODEL.md §8.8) on tensor cores: the gradients of a training step against
//! f32, the device unscale / clip / overflow check, a stopped run resuming bit-exactly with its loss scaler, and
//! the `not_supported` refusal without tensor cores.
//!
//! This test binary asks for tensor cores before the shared GPU context exists (`request_tensor_cores`) and skips
//! the GPU tests (with a note) when the adapter has none.

use std::sync::{Arc, Mutex, MutexGuard};

use folo_api::{RunEvent, RunStatus};
use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, autocast, no_grad};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::io::ModelFile;
use folo_nn::optim::GradAccumulator;
use folo_nn::{GradScaler, Module};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::train::{AMP_KEY, OptimizerChoice, ResumeOverrides};
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};

/// The GPU device with tensor cores on (and a lock: the tests share the run-time switch), or `None`.
fn gpu_tc() -> Option<(Device, MutexGuard<'static, ()>)> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    folo_gpu::request_tensor_cores();
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if !ctx.caps().coop_f16 {
        let why = folo_gpu::kernels::coop::adapter_unavailable_reason(ctx.adapter());
        let no_f16 = std::env::var("FOLO_GPU_NO_F16").is_ok_and(|v| v != "0" && !v.is_empty());
        assert!(why.is_some() || no_f16, "the adapter qualifies but: {}", ctx.tensor_cores_status());
        eprintln!("note: skipping tensor-core test: {}", ctx.tensor_cores_status());
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    ctx.set_tensor_cores(true);
    Some((dev, guard))
}

macro_rules! gpu_tc {
    () => {
        match gpu_tc() {
            Some(d) => d,
            None => return,
        }
    };
}

/// ‖a − b‖ / ‖b‖ (f64 sums).
fn rel_l2(a: &Tensor, b: &Tensor) -> f64 {
    let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        d += (*x as f64 - *y as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

/// `Σ_i ‖a_i − b_i‖² / Σ_i ‖b_i‖²`, square-rooted: the relative L2 error of a whole gradient (all tensors).
fn whole_rel_l2(a: &[(String, Tensor)], b: &[(String, Tensor)]) -> f64 {
    let (mut d2, mut n2) = (0f64, 0f64);
    for ((_, x), (_, y)) in a.iter().zip(b) {
        let e = rel_l2(x, y);
        let n = y.square().unwrap().sum_all().unwrap().item::<f32>().unwrap() as f64;
        d2 += e * e * n;
        n2 += n;
    }
    (d2 / n2).sqrt()
}

/// The loss scale of the gradient tests: large enough that no f16 gradient underflows, small enough that none
/// overflows on these random-init networks (a training run starts at 2^16 and backs off; see the training tests).
const TEST_SCALE: f32 = 1024.0;

/// `[mean((y − t)²)]` against fixed random targets, in f32: a smooth loss, so the comparison sees only the
/// arithmetic (the detection loss's task-aligned assignment flips on near-ties at random init).
fn smooth_loss(ys: &[&Tensor], dev: Device) -> Tensor {
    let mut total: Option<Tensor> = None;
    for (i, y) in ys.iter().enumerate() {
        let y = y.to_dtype(DType::F32).unwrap();
        let t =
            Tensor::randn(y.shape().clone(), 0.0, 1.0, &mut Rng::seed(40 + i as u64)).to_device(dev).unwrap();
        let l = y.sub(&t).unwrap().square().unwrap().mean_all().unwrap();
        total = Some(match total {
            Some(s) => s.add(&l).unwrap(),
            None => l,
        });
    }
    total.unwrap()
}

/// `(name, gradient)` of every parameter of `m`, each f32, divided by `scale`.
fn param_grads(m: &dyn Module, grads: &folo_core::Grads, scale: f32) -> Vec<(String, Tensor)> {
    let _g = no_grad();
    m.named_parameters()
        .into_iter()
        .map(|(n, p)| {
            let g = grads.get(&p).unwrap_or_else(|| panic!("no gradient for {n}"));
            assert_eq!(g.dtype(), DType::F32, "the master weights are f32, so is {n}'s gradient");
            (n, if scale == 1.0 { g.clone() } else { g.mul_scalar(1.0 / scale).unwrap() })
        })
        .collect()
}

/// A small net of the detector's parts (strided ConvBnAct + SiLU stem, a 1×1 ConvBnAct, a spatial self-attention
/// product with softmax and a residual add, nearest upsampling and concatenation, a depthwise conv, a 1×1 output
/// conv with bias) trained one step in f32 and with mixed precision (autocast forward on tensor cores, the loss in
/// f32 times the loss scale, backward, unscale): every parameter gradient is f32 and within 0.5% relative L2 of the
/// f32 one (0.3% for the whole gradient).
///
/// Tolerance: each activation and activation gradient is stored in f16 (relative rounding 2^-11 ≈ 5e-4) through
/// about ten layers, the products and sums in f32; measured errors are at most 2e-3 per tensor and 1.2e-3 whole.
/// A wrong kernel (a dropped tap, a transposed operand, a missing unscale) is off by O(1). Max pooling is left out
/// here: f16 rounding turns near-equal window values into ties, which routes their gradient to another position (a
/// discontinuity, not an error; the f16 pooling gradient has its own test in folo-gpu), and it moves this net's
/// gradient by about 2%.
#[test]
fn amp_gradients_match_f32_on_a_small_net() {
    use folo_nn::{Activation, Conv2d, Conv2dConfig, ConvBnAct, Forward};
    let (dev, _g) = gpu_tc!();
    let x = Tensor::rand_uniform([4, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(2)).to_device(dev).unwrap();
    struct Net {
        stem: ConvBnAct,
        down: ConvBnAct,
        pw: ConvBnAct,
        fuse: ConvBnAct,
        dw: ConvBnAct,
        out: Conv2d,
    }
    impl Module for Net {
        fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
            f("stem", &self.stem);
            f("down", &self.down);
            f("pw", &self.pw);
            f("fuse", &self.fuse);
            f("dw", &self.dw);
            f("out", &self.out);
        }
    }
    let run = |amp: bool| {
        let rng = &mut Rng::seed(9);
        let act = Activation::Silu;
        let mut net = Net {
            stem: ConvBnAct::new(3, 16, 3, 2, 1, act, rng),
            down: ConvBnAct::new(16, 32, 3, 2, 1, act, rng),
            pw: ConvBnAct::new(32, 32, 1, 1, 1, act, rng),
            fuse: ConvBnAct::new(48, 32, 3, 1, 1, act, rng),
            dw: ConvBnAct::new(32, 32, 3, 1, 32, act, rng),
            out: Conv2d::new(Conv2dConfig::new(32, 8, 1), rng),
        };
        net.to_device(dev).unwrap();
        net.train();
        let y = {
            let _a = amp.then(autocast);
            let s = net.stem.forward(&x).unwrap(); // [4, 16, 32, 32]
            let d = net.down.forward(&s).unwrap(); // [4, 32, 16, 16]
            let p = net.pw.forward(&d).unwrap();
            // Self-attention over the 256 positions: softmax(qᵀq / sqrt(32)) applied to q.
            let q = p.reshape([4, 32, 256]).unwrap();
            let a = q.transpose(1, 2).unwrap().matmul(&q).unwrap().mul_scalar(32f32.sqrt().recip()).unwrap();
            let att =
                q.matmul(&a.softmax(-1).unwrap().transpose(1, 2).unwrap()).unwrap().reshape([4, 32, 16, 16]);
            let u = att.unwrap().add(&p).unwrap().upsample_nearest2d(2).unwrap(); // [4, 32, 32, 32]
            let f = net.fuse.forward(&Tensor::cat(&[&u, &s], 1).unwrap()).unwrap();
            net.out.forward(&net.dw.forward(&f).unwrap()).unwrap()
        };
        assert_eq!(y.dtype(), if amp { DType::F16 } else { DType::F32 });
        let scale = if amp { TEST_SCALE } else { 1.0 };
        let grads = smooth_loss(&[&y], dev).mul_scalar(scale).unwrap().backward().unwrap();
        param_grads(&net, &grads, scale)
    };
    let (g16, g32) = (run(true), run(false));
    for ((n, a), (_, b)) in g16.iter().zip(&g32) {
        let e = rel_l2(a, b);
        assert!(e <= 5e-3, "gradient of {n}: relative L2 error {e:e}");
    }
    let whole = whole_rel_l2(&g16, &g32);
    eprintln!("small net, amp vs f32: whole-gradient relative L2 error {whole:e}");
    assert!(whole <= 3e-3, "whole gradient: relative L2 error {whole:e}");
}

/// folo-n (batch 2 at 128 px, random init): the mixed-precision gradient is as close to the f32 one as f16 allows.
/// A random-init detector this deep is badly conditioned (and SPPF's max pooling is discontinuous under rounding):
/// rounding only its input and weights to f16, then computing in f32 throughout, already moves the whole gradient
/// by about 4%. The bound is relative to that sensitivity, measured in the test: the AMP gradient (f16 activations
/// everywhere) must be within 4× of it (measured: about 2.4×), and so must the BatchNorm running statistics; the
/// loss within 1%.
#[test]
fn amp_gradients_of_folo_n_are_as_close_as_f16_allows() {
    let (dev, _g) = gpu_tc!();
    let x = Tensor::rand_uniform([2, 3, 128, 128], 0.0, 1.0, &mut Rng::seed(2)).to_device(dev).unwrap();
    // 0: f32; 1: f32 with the input and the weights rounded to f16; 2: mixed precision.
    let run = |mode: u8| {
        let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut Rng::seed(7)).unwrap();
        m.to_device(dev).unwrap();
        m.train();
        let q = |t: &Tensor| t.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
        if mode == 1 {
            let _g = no_grad();
            for p in m.parameters() {
                p.copy_from(&q(&p)).unwrap();
            }
        }
        let out = {
            let _a = (mode == 2).then(autocast);
            m.forward(&if mode == 1 { q(&x) } else { x.clone() }).unwrap()
        };
        let loss = smooth_loss(&[&out.box_raw, &out.cls_raw], dev);
        let value = loss.item::<f32>().unwrap();
        let scale = if mode == 2 { TEST_SCALE } else { 1.0 };
        let grads = loss.mul_scalar(scale).unwrap().backward().unwrap();
        let bufs: Vec<(String, Tensor)> =
            m.named_buffers().into_iter().map(|(n, t)| (n, t.clone())).collect();
        (value, param_grads(&m, &grads, scale), bufs)
    };
    let (l32, g32, b32) = run(0);
    let (_, grounded, brounded) = run(1);
    let (l16, g16, b16) = run(2);
    assert!((l16 - l32).abs() <= 1e-2 * l32.abs(), "loss {l16} vs {l32}");
    let (sensitivity, err) = (whole_rel_l2(&grounded, &g32), whole_rel_l2(&g16, &g32));
    eprintln!("folo-n: f16-rounded input and weights move the gradient by {sensitivity:e}, amp by {err:e}");
    assert!(err <= 4.0 * sensitivity, "amp gradient error {err:e} vs f16 sensitivity {sensitivity:e}");
    assert!(g16.iter().all(|(_, g)| g.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite())));
    let (bsens, berr) = (whole_rel_l2(&brounded, &b32), whole_rel_l2(&b16, &b32));
    eprintln!("folo-n BatchNorm statistics: sensitivity {bsens:e}, amp {berr:e}");
    assert!(berr <= 4.0 * bsens + 1e-5, "running statistics error {berr:e} vs f16 sensitivity {bsens:e}");
}

/// `GradAccumulator::unscale_clip` on the device: finite gradients are unscaled and clipped as on the CPU, and an
/// infinite or NaN gradient anywhere is reported (the step is skipped), with and without clipping.
#[test]
fn device_unscale_clip_matches_cpu_and_flags_overflow() {
    let (dev, _g) = gpu_tc!();
    let shapes: [&[usize]; 3] = [&[16, 8, 3, 3], &[16], &[40, 7]];
    let params: Vec<Tensor> = shapes.iter().map(|s| Tensor::zeros(*s).requires_grad_()).collect();
    let grads: Vec<Tensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| Tensor::randn(*s, 0.0, 3e4, &mut Rng::seed(20 + i as u64)))
        .collect();
    let inv = 1.0 / 65536.0;
    // `max_norm` 1 clips these gradients (their unscaled norm is about 10); 1e9 does not; 0 disables clipping.
    for max_norm in [1.0, 1e9, 0.0] {
        let run = |device: Device, poison: Option<f32>| {
            let ps: Vec<Tensor> =
                params.iter().map(|p| p.to_device(device).unwrap().detach().requires_grad_()).collect();
            let mut acc = GradAccumulator::new(&ps);
            let loss = ps
                .iter()
                .zip(&grads)
                .map(|(p, g)| p.mul(&g.to_device(device).unwrap()).unwrap().sum_all().unwrap())
                .reduce(|a, b| a.add(&b).unwrap())
                .unwrap();
            let mut gr = loss.backward().unwrap();
            if let Some(v) = poison {
                let g = gr.get(&ps[2]).unwrap().to_device(Device::Cpu).unwrap();
                let mut host = g.to_vec::<f32>().unwrap();
                host[17] = v;
                gr.insert(
                    &ps[2],
                    Tensor::from_vec(host, g.shape().clone()).unwrap().to_device(device).unwrap(),
                );
            }
            acc.add(&gr).unwrap();
            let finite = acc.unscale_clip(inv, max_norm).unwrap();
            let out: Vec<Vec<f32>> = ps
                .iter()
                .map(|p| {
                    use folo_nn::optim::GradSource;
                    acc.grad(p).unwrap().to_vec::<f32>().unwrap()
                })
                .collect();
            (finite, out)
        };
        let (ok_cpu, cpu) = run(Device::Cpu, None);
        let (ok_gpu, gpu) = run(dev, None);
        assert!(ok_cpu && ok_gpu, "max_norm {max_norm}");
        for (c, g) in cpu.iter().zip(&gpu) {
            for (a, b) in g.iter().zip(c) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "max_norm {max_norm}: {a} vs {b}");
            }
        }
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(!run(dev, Some(bad)).0, "max_norm {max_norm}: {bad} not reported");
            assert!(!run(Device::Cpu, Some(bad)).0, "CPU, max_norm {max_norm}: {bad} not reported");
        }
    }
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

fn amp_config() -> TrainConfig {
    TrainConfig {
        epochs: 4,
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
        amp: true,
        ..TrainConfig::default()
    }
}

/// Every tensor of a `.folo` file as raw bytes, by name.
fn tensors(path: &std::path::Path) -> Vec<(String, Vec<u8>)> {
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

/// A mixed-precision run stopped at an epoch boundary and resumed ends in the same state as an uninterrupted one,
/// bit for bit, loss scaler included: `folo.amp` holds the scale, the growth count and the skipped steps, and the
/// resumed run continues from them. Master weights, optimizer state and EMA in the checkpoint are f32.
#[test]
fn amp_run_resumes_bit_exactly_with_its_loss_scaler() {
    let (dev, _g) = gpu_tc!();
    let root = std::env::temp_dir().join(format!("folo-amp-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let (dir_a, dir_b) = (root.join("a"), root.join("b"));
    let log: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = log.clone();
    let opts_a = TrainOptions {
        run_dir: Some(dir_a.clone()),
        events: Some(Box::new(move |e| sink.lock().unwrap().push(e.clone()))),
        step_interval: std::time::Duration::ZERO,
        ..TrainOptions::default()
    };
    let mut a = Trainer::new(amp_config(), data(), opts_a).unwrap();
    assert_eq!(a.device(), dev);
    assert_eq!(a.loss_scaler().map(GradScaler::scale), Some(65536.0));
    let sa = a.run().unwrap();
    assert_eq!((sa.status, sa.epochs_completed), (RunStatus::Completed, 4));
    assert!(sa.epochs.iter().all(|e| e.train_loss.total.is_finite() && e.val.map50.is_finite()));
    let logs: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Log(l) = e { Some(l.message.clone()) } else { None })
        .collect();
    assert!(logs.iter().any(|m| m.starts_with("mixed precision:")), "{logs:?}");
    assert_eq!(
        logs.iter().filter(|m| m.contains("optimizer steps skipped for gradient overflow")).count(),
        4
    );
    let scaler_a = a.loss_scaler().unwrap().clone();

    let opts =
        |dir: &std::path::Path| TrainOptions { run_dir: Some(dir.to_path_buf()), ..TrainOptions::default() };
    let mut ob = opts(&dir_b);
    let stop = ob.stop.clone();
    ob.events = Some(Box::new(move |e| {
        if matches!(e, RunEvent::Epoch(x) if x.epoch == 2) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }));
    let sb = Trainer::new(amp_config(), data(), ob).unwrap().run().unwrap();
    assert_eq!((sb.status, sb.epochs_completed), (RunStatus::Stopped, 2));
    let last_b = dir_b.join("weights/last.folo");
    let file = ModelFile::open(&last_b).unwrap();
    let stored = GradScaler::from_json(file.metadata().extra.get(AMP_KEY).expect("folo.amp")).unwrap();
    assert_eq!(file.checkpoint().unwrap().train["amp"], serde_json::json!(true));
    let over = ResumeOverrides { device: Some("gpu:0".into()), ..Default::default() };
    let mut r = Trainer::resume(&last_b, data(), opts(&dir_b), over).unwrap();
    assert_eq!(r.loss_scaler(), Some(&stored));
    assert_eq!(r.run().unwrap().epochs_completed, 4);
    assert_eq!(r.loss_scaler(), Some(&scaler_a));

    let (ta, tb) = (tensors(&dir_a.join("weights/last.folo")), tensors(&last_b));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    let f = ModelFile::open(&last_b).unwrap();
    for info in f.tensor_infos() {
        assert!(matches!(f.tensor(&info.name).unwrap().dtype(), DType::F32 | DType::I64), "{}", info.name);
    }
    assert_eq!(GradScaler::from_json(f.metadata().extra.get(AMP_KEY).unwrap()).unwrap(), scaler_a);
    let _ = std::fs::remove_dir_all(&root);
}

/// Without tensor cores (here: on the CPU) a mixed-precision run is refused up front with `not_supported`.
#[test]
fn amp_without_tensor_cores_is_not_supported() {
    let cfg = TrainConfig { device: "cpu".into(), ..amp_config() };
    let e = Trainer::new(cfg, data(), TrainOptions::default()).err().expect("refused");
    assert_eq!(e.code(), "not_supported", "{e}");
    assert!(e.to_string().contains("tensor cores"), "{e}");
}

/// `gpu_train.rs`'s tiny run (16 images at 128 px, batch 4, AdamW, no warmup or augmentation, 10 epochs) with
/// mixed precision against the same run in f32: the loss falls as far (the last three epochs within 15% of f32's).
/// The first steps overflow at the initial scale 2^16 and are skipped while it backs off (a run of thousands of
/// steps does not notice; on this 40-step one it shows as a one-epoch delay), so the curves are compared at the end.
#[test]
fn tiny_amp_run_learns() {
    let (_, _g) = gpu_tc!();
    let cfg = TrainConfig {
        epochs: 10,
        nbs: 4,
        warmup_epochs: 0.0,
        augment: AugmentConfig::none(),
        ..amp_config()
    };
    let f32_losses: Vec<f32> =
        Trainer::new(TrainConfig { amp: false, ..cfg.clone() }, data(), TrainOptions::default())
            .unwrap()
            .run()
            .unwrap()
            .epochs
            .iter()
            .map(|e| e.train_loss.total)
            .collect();
    eprintln!("f32 losses {f32_losses:?}");
    let mut t = Trainer::new(cfg, data(), TrainOptions::default()).unwrap();
    let s = t.run().unwrap();
    let losses: Vec<f32> = s.epochs.iter().map(|e| e.train_loss.total).collect();
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    let tail = |l: &[f32]| l[l.len() - 3..].iter().sum::<f32>() / 3.0;
    let scaler = t.loss_scaler().unwrap();
    eprintln!("loss scale {} after {} skipped steps: amp {losses:?}", scaler.scale(), scaler.skipped_steps());
    assert!(tail(&losses) <= 1.15 * tail(&f32_losses), "amp {losses:?} vs f32 {f32_losses:?}");
    assert!(tail(&losses) < 0.7 * losses.iter().fold(0f32, |m, &l| m.max(l)), "the loss falls: {losses:?}");
    assert!(scaler.skipped_steps() < 20 && scaler.scale() >= 1.0, "{scaler:?}");
}

/// Rewrites the `folo.amp` entry of the checkpoint at `path` (the rest of the file is kept byte for byte).
fn edit_amp(path: &std::path::Path, edit: impl FnOnce(&mut folo_nn::AmpContract)) {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta = header["__metadata__"].as_object_mut().unwrap();
    let mut c: folo_nn::AmpContract = serde_json::from_str(meta[AMP_KEY].as_str().unwrap()).unwrap();
    edit(&mut c);
    meta.insert(AMP_KEY.into(), serde_json::Value::String(serde_json::to_string(&c).unwrap()));
    let mut h = serde_json::to_string(&header).unwrap().into_bytes();
    while !(8 + h.len()).is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(&bytes[8 + n..]);
    std::fs::write(path, out).unwrap();
}

/// A run stopped after its first epoch: `(root dir, its last.folo)`.
fn one_epoch_checkpoint(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("folo-amp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut o = TrainOptions { run_dir: Some(root.clone()), ..TrainOptions::default() };
    let stop = o.stop.clone();
    o.events = Some(Box::new(move |e| {
        if matches!(e, RunEvent::Epoch(_)) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }));
    let s = Trainer::new(amp_config(), data(), o).unwrap().run().unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Stopped, 1));
    let last = root.join("weights/last.folo");
    (root, last)
}

/// Resumes `ckpt` for one more epoch (stopped at its end); the run's result.
fn resume_one_epoch(
    root: &std::path::Path,
    ckpt: &std::path::Path,
) -> folo_vision::Result<folo_vision::train::TrainSummary> {
    let mut o = TrainOptions { run_dir: Some(root.to_path_buf()), ..TrainOptions::default() };
    let stop = o.stop.clone();
    o.events = Some(Box::new(move |e| {
        if matches!(e, RunEvent::Epoch(_)) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }));
    let over = ResumeOverrides { device: Some("gpu:0".into()), ..Default::default() };
    Trainer::resume(ckpt, data(), o, over)?.run()
}

/// An injected overflow (the checkpoint's loss scale set to 2^120, so every f16 gradient is infinite): every window
/// of the next epoch skips its step and backs the scale off, and the parameters, the optimizer state, the EMA and
/// the schedule are exactly as they were (only the BatchNorm running statistics, which the forward updates, move).
#[test]
fn an_overflowing_epoch_leaves_optimizer_ema_and_schedule_unchanged() {
    let (_dev, _g) = gpu_tc!();
    let (root, last) = one_epoch_checkpoint("overflow");
    let mut streak = 0;
    edit_amp(&last, |c| {
        c.scale = 2f64.powi(120);
        streak = c.skip_streak;
    });
    let before = tensors(&last);
    let s = resume_one_epoch(&root, &last).unwrap();
    assert_eq!((s.status, s.epochs_completed), (RunStatus::Stopped, 2));
    let after = tensors(&last);
    assert_eq!(before.len(), after.len());
    let mut compared = 0;
    for ((n, a), (m, b)) in before.iter().zip(&after) {
        assert_eq!(n, m);
        if n.contains("running_") && !n.starts_with("ema.") {
            continue;
        }
        assert!(a == b, "{n} changed although every step overflowed");
        compared += 1;
    }
    assert!(compared > 300 && before.iter().any(|(n, _)| n == "sched.__step"), "{compared} tensors compared");
    // 16 images, batch 4, nbs 8: two windows, both skipped, extending the checkpoint's streak.
    let f = ModelFile::open(&last).unwrap();
    let c = GradScaler::from_json(f.metadata().extra.get(AMP_KEY).unwrap()).unwrap().contract();
    assert_eq!((c.scale, c.skip_streak), (2f64.powi(118), streak + 2), "{c:?}");
    let _ = std::fs::remove_dir_all(&root);
}

/// The skip streak is part of `folo.amp`: a checkpoint written with 49 overflowing steps in a row resumes into a
/// run that fails at the 50th (`MAX_AMP_SKIPS`), as an uninterrupted run would, instead of counting from zero.
#[test]
fn an_overflow_streak_continues_across_a_resume() {
    let (_dev, _g) = gpu_tc!();
    let (root, last) = one_epoch_checkpoint("streak");
    let max = folo_vision::train::MAX_AMP_SKIPS as u64;
    edit_amp(&last, |c| {
        c.scale = 2f64.powi(120);
        c.skip_streak = max - 1;
        c.skipped_steps = c.skipped_steps.max(max - 1);
    });
    let e = resume_one_epoch(&root, &last).expect_err("the 50th skip in a row fails the run");
    assert_eq!(e.code(), "training_failed", "{e}");
    assert!(e.to_string().contains(&format!("{max} consecutive optimizer steps")), "{e}");
    let _ = std::fs::remove_dir_all(&root);
}
