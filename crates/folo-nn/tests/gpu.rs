//! folo-nn on the GPU (folo-gpu's backend): moving modules, optimizer state and EMAs between devices, the
//! fused train-mode BatchNorm kernels and the fused Conv + bias + activation inference path.
//!
//! Every test returns early, with a note, when there is no hardware adapter (CI). Software adapters (WARP,
//! llvmpipe) are skipped too unless `FOLO_GPU_TEST_SOFTWARE=1`, as in folo-gpu's own tests.
//!
//! Tolerances: the CPU and GPU sum in different orders (and the GPU fuses multiply-adds), so values agree to
//! `atol + rtol·|ref|` with the bounds stated per check rather than bit for bit.

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{Device, Error as CoreError, Tensor, no_grad};
use folo_nn::optim::*;
use folo_nn::*;

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

macro_rules! gpu {
    () => {
        match gpu() {
            Some(d) => d,
            None => return,
        }
    };
}

#[track_caller]
fn close(got: &Tensor, want: &Tensor, atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (g, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() <= atol + rtol * b.abs(),
            "{what}[{i}]: got {a}, want {b} (atol {atol}, rtol {rtol})"
        );
    }
}

/// Two Conv-BN-SiLU blocks and a plain conv head: every kind of tensor a detector has.
struct Net {
    a: ConvBnAct,
    b: ConvBnAct,
    head: Conv2d,
}

impl Net {
    fn new(seed: u64) -> Self {
        let rng = &mut Rng::seed(seed);
        Net {
            a: ConvBnAct::new(3, 8, 3, 1, 1, Activation::Silu, rng),
            b: ConvBnAct::new(8, 8, 3, 2, 1, Activation::Silu, rng),
            head: Conv2d::new(Conv2dConfig::new(8, 4, 1), rng),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.head.forward(&self.b.forward(&self.a.forward(x).unwrap()).unwrap()).unwrap()
    }
}

impl Module for Net {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("a", &self.a);
        f("b", &self.b);
        f("head", &self.head);
    }
}

#[test]
fn tensor_to_device_in_place() {
    let dev = gpu!();
    let t = Tensor::randn([4, 5], 0.0, 1.0, &mut Rng::seed(1)).requires_grad_();
    let view = t.narrow(1, 1, 2).unwrap();
    let (id, before) = (t.id(), t.to_vec::<f32>().unwrap());
    let y = t.mul(&t).unwrap().sum_all().unwrap();
    t.to_device_(dev).unwrap();
    assert_eq!((t.device(), view.device(), t.id()), (dev, dev, id));
    assert!(t.requires_grad() && t.is_leaf());
    assert_eq!(t.to_vec::<f32>().unwrap(), before);
    assert_eq!(view.to_vec::<f32>().unwrap(), t.narrow(1, 1, 2).unwrap().to_vec::<f32>().unwrap());
    // The graph recorded before the move saved the old storage version.
    assert!(matches!(y.backward(), Err(CoreError::ModifiedInPlace { .. })));
    // Results of recorded ops cannot move (their gradient would land on the old device).
    assert!(t.mul_scalar(2.0).unwrap().to_device_(Device::Cpu).is_err());
    // A fresh graph on the GPU works, and the gradient is keyed by the same id.
    let g = t.square().unwrap().sum_all().unwrap().backward().unwrap();
    assert_eq!(g.get(&t).unwrap().device(), dev);
    t.to_device_(Device::Cpu).unwrap();
    assert_eq!((t.device(), t.id()), (Device::Cpu, id));
    assert_eq!(t.to_vec::<f32>().unwrap(), before);
    // Every dtype moves.
    let i = Tensor::arange_i64(-3, 5);
    i.to_device_(dev).unwrap();
    assert_eq!(i.to_vec::<i64>().unwrap(), (-3..5).collect::<Vec<_>>());
}

#[test]
fn module_to_device_round_trip() {
    let dev = gpu!();
    let mut net = Net::new(3);
    net.a.bn.as_ref().unwrap().running_mean.copy_from(&Tensor::full([8], 0.25)).unwrap();
    net.eval();
    let before = net.named_tensors();
    let values: Vec<Vec<f32>> = before.iter().map(|t| t.tensor.to_vec::<f32>().unwrap()).collect();
    net.to_device(dev).unwrap();
    let after = net.named_tensors();
    assert_eq!(after.len(), before.len());
    for ((b, a), v) in before.iter().zip(&after).zip(&values) {
        assert_eq!((&a.name, a.kind, a.tensor.id()), (&b.name, b.kind, b.tensor.id()));
        assert_eq!(a.tensor.device(), dev, "{}", a.name);
        assert_eq!(a.tensor.requires_grad(), a.kind == TensorKind::Parameter, "{}", a.name);
        assert_eq!(&a.tensor.to_vec::<f32>().unwrap(), v, "{}", a.name);
    }
    assert!(!net.is_training(), "the mode survives the move");
    net.to_device(Device::Cpu).unwrap();
    for (t, v) in net.named_tensors().iter().zip(&values) {
        assert_eq!(t.tensor.device(), Device::Cpu);
        assert_eq!(&t.tensor.to_vec::<f32>().unwrap(), v);
    }
}

/// Training steps (forward with train-mode BatchNorm, backward, optimizer, EMA) on the CPU and, after the
/// first two, on the GPU for a second copy whose model, optimizer state and EMA were moved mid-run: every
/// parameter, buffer, state tensor and EMA tensor must still agree.
#[test]
fn training_steps_on_gpu_match_cpu() {
    let dev = gpu!();
    let x = Tensor::rand_uniform([2, 3, 16, 16], -1.0, 1.0, &mut Rng::seed(4));
    let xg = x.to_device(dev).unwrap();
    for adam in [false, true] {
        let make = || {
            let net = Net::new(5);
            let groups =
                StandardGroups::from_module(&net).into_groups(if adam { 1e-3 } else { 0.05 }, 0.9, 5e-4);
            let opt: Box<dyn Optimizer> = if adam {
                Box::new(AdamW::new(groups).unwrap())
            } else {
                Box::new(Sgd::new(groups).unwrap())
            };
            let ema = ModelEma::with_params(Net::new(6), &net, 0.9, 2.0).unwrap();
            (net, opt, ema)
        };
        let step = |net: &Net, opt: &mut Box<dyn Optimizer>, ema: &mut ModelEma<Net>, x: &Tensor| {
            let out = net.forward(x);
            let loss = out.square().unwrap().mean_all().unwrap().add(&out.mean_all().unwrap()).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
            ema.update(net).unwrap();
        };
        let (cpu, mut cpu_opt, mut cpu_ema) = make();
        let (mut gn, mut g_opt, mut g_ema) = make();
        for _ in 0..2 {
            step(&cpu, &mut cpu_opt, &mut cpu_ema, &x);
            step(&gn, &mut g_opt, &mut g_ema, &x);
        }
        gn.to_device(dev).unwrap();
        g_opt.to_device(dev).unwrap();
        g_ema.to_device(dev).unwrap();
        for (_, t) in g_opt.state_tensors().unwrap() {
            assert_eq!(t.device(), dev);
        }
        for _ in 0..3 {
            step(&cpu, &mut cpu_opt, &mut cpu_ema, &x);
            step(&gn, &mut g_opt, &mut g_ema, &xg);
        }
        let what = if adam { "adamw" } else { "sgd" };
        for (c, g) in cpu.named_tensors().iter().zip(gn.named_tensors()) {
            assert_eq!(g.tensor.device(), dev);
            close(&g.tensor, &c.tensor, 2e-4, 2e-3, &format!("{what} model {}", c.name));
        }
        for ((n, c), (_, g)) in cpu_opt.state_tensors().unwrap().iter().zip(g_opt.state_tensors().unwrap()) {
            close(&g, c, 2e-4, 2e-3, &format!("{what} state {n}"));
        }
        for (c, g) in cpu_ema.module().named_tensors().iter().zip(g_ema.module().named_tensors()) {
            assert_eq!(g.tensor.device(), dev);
            close(&g.tensor, &c.tensor, 2e-4, 2e-3, &format!("{what} ema {}", c.name));
        }
    }
}

/// An EMA built for a model that already lives on the GPU follows it there.
#[test]
fn ema_follows_its_source_device() {
    let dev = gpu!();
    let mut net = Net::new(7);
    net.to_device(dev).unwrap();
    let ema = ModelEma::new(Net::new(8), &net).unwrap();
    for (e, m) in ema.module().named_tensors().iter().zip(net.named_tensors()) {
        assert_eq!(e.tensor.device(), dev);
        assert_eq!(e.tensor.to_vec::<f32>().unwrap(), m.tensor.to_vec::<f32>().unwrap());
    }
}

/// BatchNorm in train mode as composite ops (the reference).
fn composite_bn(x: &Tensor, g: &Tensor, b: &Tensor, eps: f64) -> Tensor {
    let c = g.numel();
    let mean = x.mean([0, 2, 3], true).unwrap();
    let d = x.sub(&mean).unwrap();
    let var = d.square().unwrap().mean([0, 2, 3], true).unwrap();
    let inv = var.add_scalar(eps as f32).unwrap().rsqrt().unwrap();
    d.mul(&inv)
        .unwrap()
        .mul(&g.reshape([1, c, 1, 1]).unwrap())
        .unwrap()
        .add(&b.reshape([1, c, 1, 1]).unwrap())
        .unwrap()
}

/// The fused train-mode BatchNorm + activation kernels against composite ops on the GPU: output, every
/// gradient and the running statistics. Shapes cover odd planes (scalar path), planes that are multiples of 4
/// (vec4 path), many chunks per channel and tiny planes.
#[test]
fn fused_batch_norm_matches_composite() {
    let dev = gpu!();
    let cases: [(&[usize], Activation); 5] = [
        (&[3, 5, 7, 6], Activation::Silu),
        (&[2, 4, 8, 8], Activation::Identity),
        (&[4, 6, 48, 40], Activation::Relu),
        (&[16, 3, 2, 2], Activation::Silu),
        (&[2, 3, 5, 5], Activation::Hardswish), // not fusable: fused BN, then the activation
    ];
    for (i, &(shape, act)) in cases.iter().enumerate() {
        let rng = &mut Rng::seed(10 + i as u64);
        let c = shape[1];
        let leaf = |t: Tensor| t.to_device(dev).unwrap().detach().requires_grad_();
        let x = leaf(Tensor::randn(shape, 0.7, 2.0, rng));
        let g = leaf(Tensor::rand_uniform([c], 0.5, 1.5, rng));
        let b = leaf(Tensor::rand_uniform([c], -0.5, 0.5, rng));
        let rm = Tensor::rand_uniform([c], -0.2, 0.2, rng).to_device(dev).unwrap();
        let rv = Tensor::rand_uniform([c], 0.5, 1.5, rng).to_device(dev).unwrap();
        let (rm0, rv0) = (rm.to_vec::<f32>().unwrap(), rv.to_vec::<f32>().unwrap());
        let (eps, mom) = (1e-3, 0.1);
        let bn = BatchNorm2d::from_tensors(g.clone(), b.clone(), rm.clone(), rv.clone(), eps, mom).unwrap();
        let y = bn.forward_act(&x, act).unwrap();
        if act != Activation::Hardswish {
            assert_eq!(y.grad_fn_name(), Some("batch_norm2d_act"), "case {i}: the fused path ran");
        }
        let want = act.forward(&composite_bn(&x, &g, &b, eps)).unwrap();
        let what = format!("case {i} {shape:?} {act:?}");
        close(&y, &want, 2e-5, 2e-5, &format!("{what} y"));

        let w = Tensor::randn(shape, 0.0, 1.0, rng).to_device(dev).unwrap();
        let g1 = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g2 = want.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        for (t, name) in [(&x, "dx"), (&g, "dweight"), (&b, "dbias")] {
            close(g1.get(t).unwrap(), g2.get(t).unwrap(), 2e-4, 1e-3, &format!("{what} {name}"));
        }

        // Running statistics: (1 − m)·old + m·stat, with the unbiased variance.
        let n = (shape[0] * shape[2] * shape[3]) as f64;
        let xs = x.to_vec::<f32>().unwrap();
        let hw = shape[2] * shape[3];
        for ch in 0..c {
            let vals: Vec<f64> = (0..shape[0])
                .flat_map(|bi| xs[(bi * c + ch) * hw..(bi * c + ch + 1) * hw].iter().map(|&v| v as f64))
                .collect();
            let mean = vals.iter().sum::<f64>() / n;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let em = (1.0 - mom) * rm0[ch] as f64 + mom * mean;
            let ev = (1.0 - mom) * rv0[ch] as f64 + mom * var;
            let (gm, gv) = (rm.to_vec::<f32>().unwrap()[ch] as f64, rv.to_vec::<f32>().unwrap()[ch] as f64);
            assert!((gm - em).abs() <= 1e-5 + 1e-5 * em.abs(), "{what} running_mean[{ch}]: {gm} vs {em}");
            assert!((gv - ev).abs() <= 1e-5 + 1e-4 * ev.abs(), "{what} running_var[{ch}]: {gv} vs {ev}");
        }
    }
}

/// folo-core's finite-difference gradient check through the fused BatchNorm + SiLU on the GPU.
#[test]
fn fused_batch_norm_grad_check() {
    let dev = gpu!();
    let rng = &mut Rng::seed(21);
    let inputs = [
        Tensor::randn([2, 3, 5, 4], 0.3, 1.5, rng),
        Tensor::rand_uniform([3], 0.5, 1.5, rng),
        Tensor::rand_uniform([3], -0.5, 0.5, rng),
    ];
    let f = |v: &[Tensor]| {
        let to = |t: &Tensor| t.to_device(dev);
        let running =
            (Tensor::zeros_with([3], folo_core::DType::F32, dev)?, Tensor::ones([3]).to_device(dev)?);
        let bn = BatchNorm2d::from_tensors(to(&v[1])?, to(&v[2])?, running.0, running.1, 1e-3, 0.03)
            .map_err(|e| CoreError::Autograd(e.to_string()))?;
        let y =
            bn.forward_act(&to(&v[0])?, Activation::Silu).map_err(|e| CoreError::Autograd(e.to_string()))?;
        if folo_core::is_grad_enabled() {
            assert_eq!(y.grad_fn_name(), Some("batch_norm2d_act"));
        }
        y.to_device(Device::Cpu)
    };
    let opts = GradCheck { eps: 1e-2, atol: 3e-3, rtol: 2e-2, max_checks: 48, seed: 7 };
    grad_check_with(f, &inputs, &opts).unwrap();
}

/// ConvBnAct inference on the GPU (BatchNorm folded on the fly, and after `fuse_`) against the CPU's
/// unfused conv → BatchNorm → activation, for a regular, a strided depthwise and a pointwise conv.
#[test]
fn conv_bn_act_inference_parity() {
    let dev = gpu!();
    let cases = [
        (8, 16, 3, 1, 1, Activation::Silu),
        (16, 16, 3, 2, 16, Activation::Silu),
        (8, 12, 1, 1, 1, Activation::Identity),
    ];
    for (i, &(c1, c2, k, s, groups, act)) in cases.iter().enumerate() {
        let rng = &mut Rng::seed(30 + i as u64);
        let mut m = ConvBnAct::new(c1, c2, k, s, groups, act, rng);
        {
            let bn = m.bn.as_ref().unwrap();
            let _g = no_grad();
            bn.weight.copy_from(&Tensor::rand_uniform([c2], 0.5, 1.5, rng)).unwrap();
            bn.bias.copy_from(&Tensor::rand_uniform([c2], -0.5, 0.5, rng)).unwrap();
            bn.running_mean.copy_from(&Tensor::rand_uniform([c2], -0.3, 0.3, rng)).unwrap();
            bn.running_var.copy_from(&Tensor::rand_uniform([c2], 0.2, 2.0, rng)).unwrap();
        }
        m.eval();
        let x = Tensor::randn([2, c1, 13, 11], 0.0, 1.0, rng);
        let want = {
            let _g = no_grad();
            m.forward(&x).unwrap()
        };
        m.to_device(dev).unwrap();
        let xg = x.to_device(dev).unwrap();
        let what = format!("case {i}");
        {
            let _g = no_grad();
            close(&m.forward(&xg).unwrap(), &want, 1e-4, 1e-4, &format!("{what} folded on the fly"));
        }
        // With grad on, the same math runs as composite ops and gradients reach the BatchNorm parameters.
        let y = m.forward(&xg).unwrap();
        close(&y, &want, 1e-4, 1e-4, &format!("{what} composite"));
        let grads = y.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(&m.bn.as_ref().unwrap().weight).is_some());
        m.fuse_().unwrap();
        assert_eq!(m.conv.weight.device(), dev);
        let _g = no_grad();
        close(&m.forward(&xg).unwrap(), &want, 1e-4, 1e-4, &format!("{what} fused"));
    }
}

/// Every fused optimizer variant (SGD with and without momentum / decay / Nesterov, AdamW with and without
/// decay) against the CPU's loops over a few steps, on a tensor whose size is not a multiple of the
/// workgroup.
#[test]
fn fused_optimizer_kernels_match_cpu() {
    let dev = gpu!();
    let rng = &mut Rng::seed(40);
    let p0 = Tensor::randn([3, 7, 13], 0.0, 1.0, rng);
    let grads: Vec<Tensor> = (0..3).map(|_| Tensor::randn([3, 7, 13], 0.0, 0.5, rng)).collect();
    type Make = fn(ParamGroup) -> Box<dyn Optimizer>;
    let sgd: Make = |g| Box::new(Sgd::new(vec![g]).unwrap());
    let adamw: Make = |g| Box::new(AdamW::new(vec![g]).unwrap());
    let cases: [(&str, Make, f64, f64, bool); 6] = [
        ("sgd plain", sgd, 0.0, 0.0, false),
        ("sgd decay", sgd, 0.0, 1e-2, false),
        ("sgd momentum", sgd, 0.9, 0.0, false),
        ("sgd nesterov + decay", sgd, 0.937, 5e-4, true),
        ("adamw", adamw, 0.9, 0.0, false),
        ("adamw decay", adamw, 0.9, 0.05, false),
    ];
    for (what, make, mom, wd, nesterov) in cases {
        let run = |d: Device| {
            // A fresh copy: on the CPU `to_device` is a no-op and the step would write into `p0`.
            let p = Tensor::from_vec(p0.to_vec::<f32>().unwrap(), p0.shape().clone()).unwrap();
            let p = p.to_device(d).unwrap().detach().requires_grad_();
            let g = ParamGroup::new("g", vec![("p".into(), p.clone())])
                .lr(0.01)
                .momentum(mom)
                .weight_decay(wd)
                .nesterov(nesterov);
            let mut opt = make(g);
            for gr in &grads {
                let gr = gr.to_device(d).unwrap();
                let grads = p.mul(&gr).unwrap().sum_all().unwrap().backward().unwrap();
                opt.step(&grads).unwrap();
            }
            (p, opt.state_tensors().unwrap())
        };
        let ((pc, sc), (pg, sg)) = (run(Device::Cpu), run(dev));
        assert_eq!(pg.device(), dev);
        close(&pg, &pc, 1e-6, 1e-5, what);
        for ((n, c), (_, g)) in sc.iter().zip(&sg) {
            close(g, c, 1e-6, 1e-5, &format!("{what} {n}"));
        }
    }
    // The EMA's fused `a·p + b·g` update.
    let (e, m) = (Tensor::randn([1000], 0.0, 1.0, rng), Tensor::randn([1000], 0.0, 1.0, rng));
    let (eg, mg) = (e.to_device(dev).unwrap(), m.to_device(dev).unwrap());
    assert!(eg.optimizer_step_(&mg, &[], &folo_core::backend::OptimStep::Axpby { a: 0.9, b: 0.1 }).unwrap());
    let want = e.mul_scalar(0.9).unwrap().add(&m.mul_scalar(0.1).unwrap()).unwrap();
    close(&eg, &want, 1e-6, 1e-6, "axpby");
}

/// The fused BatchNorm forward without a running-statistics update (folo-core's raw entry point).
#[test]
fn fused_batch_norm_without_running_stats() {
    let dev = gpu!();
    let rng = &mut Rng::seed(41);
    let x = Tensor::randn([2, 3, 5, 3], 1.0, 3.0, rng);
    let (g, b) = (Tensor::rand_uniform([3], 0.5, 1.5, rng), Tensor::rand_uniform([3], -0.5, 0.5, rng));
    let to = |t: &Tensor| t.to_device(dev).unwrap();
    let out =
        to(&x).batch_norm_act_train(&to(&g), &to(&b), 1e-3, None, None).unwrap().expect("fused kernels");
    close(&out.y, &composite_bn(&x, &g, &b, 1e-3), 2e-5, 2e-5, "y");
    let (dx, dw, db) = to(&x)
        .batch_norm_act_train_backward(
            &to(&x).ones_like().unwrap(),
            &to(&g),
            &to(&b),
            &out.mean,
            &out.invstd,
            None,
        )
        .unwrap()
        .expect("fused kernels");
    // With a constant output gradient: dβ = n, dγ = Σ x̂ = 0, dx = 0.
    close(&db, &Tensor::full([3], 30.0), 1e-4, 1e-5, "db");
    close(&dw, &Tensor::zeros([3]), 1e-4, 0.0, "dw");
    close(&dx, &Tensor::zeros([2, 3, 5, 3]), 1e-5, 0.0, "dx");
}

/// Inputs with a large mean and a small spread (and constant planes): the fused kernels centre before scaling,
/// so they match an `f64` reference and the composite gradients instead of losing the signal to cancellation.
#[test]
fn fused_batch_norm_large_mean() {
    let dev = gpu!();
    let rng = &mut Rng::seed(42);
    let (eps, shape) = (1e-3f64, [4usize, 2, 6, 5]);
    let x = Tensor::randn(shape, 0.0, 0.5, rng).add_scalar(1000.0).unwrap();
    let to = |t: &Tensor| t.to_device(dev).unwrap().detach().requires_grad_();
    let (xg, g, b) = (to(&x), to(&Tensor::full([2], 1.5)), to(&Tensor::full([2], 0.25)));
    let bn = BatchNorm2d::from_tensors(
        g.clone(),
        b.clone(),
        Tensor::zeros([2]).to_device(dev).unwrap(),
        Tensor::ones([2]).to_device(dev).unwrap(),
        eps,
        0.1,
    )
    .unwrap();
    let y = bn.forward_act(&xg, Activation::Silu).unwrap();
    assert_eq!(y.grad_fn_name(), Some("batch_norm2d_act"));
    // f64 reference of the normalisation.
    let (xs, ys) = (x.to_vec::<f32>().unwrap(), y.to_vec::<f32>().unwrap());
    let hw = shape[2] * shape[3];
    for ch in 0..2 {
        let idx: Vec<usize> =
            (0..shape[0]).flat_map(|n| ((n * 2 + ch) * hw)..((n * 2 + ch + 1) * hw)).collect();
        let n = idx.len() as f64;
        let mean = idx.iter().map(|&i| xs[i] as f64).sum::<f64>() / n;
        let var = idx.iter().map(|&i| (xs[i] as f64 - mean).powi(2)).sum::<f64>() / n;
        for &i in &idx {
            let z = (xs[i] as f64 - mean) / (var + eps).sqrt() * 1.5 + 0.25;
            let want = z / (1.0 + (-z).exp());
            assert!((ys[i] as f64 - want).abs() <= 2e-3, "y[{i}]: {} vs {want}", ys[i]);
        }
    }
    let w = Tensor::randn(shape, 0.0, 1.0, rng).to_device(dev).unwrap();
    let g1 = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let want = Activation::Silu.forward(&composite_bn(&xg, &g, &b, eps)).unwrap();
    let g2 = want.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    for (t, name) in [(&xg, "dx"), (&g, "dweight"), (&b, "dbias")] {
        close(g1.get(t).unwrap(), g2.get(t).unwrap(), 5e-3, 5e-3, name);
    }
    // Constant planes, however large: x̂ = 0 exactly, so y = act(β).
    for v in [1e6f32, 1e30] {
        let xc = Tensor::full([2, 1, 2, 2], v).to_device(dev).unwrap();
        let out = xc
            .batch_norm_act_train(
                &Tensor::ones([1]).to_device(dev).unwrap(),
                &Tensor::zeros([1]).to_device(dev).unwrap(),
                eps,
                None,
                None,
            )
            .unwrap()
            .expect("fused kernels");
        assert_eq!(out.y.to_vec::<f32>().unwrap(), vec![0.0; 8], "constant {v}");
    }
}

/// A non-constant plane around a huge mean (1e20 ± 1e15): the Welford merges must not square the mean (an empty
/// accumulator merged as `d · d · 0` overflows to NaN), so the statistics, outputs, running variance and
/// gradients are those of a ±1e15 spread.
#[test]
fn fused_batch_norm_huge_mean() {
    let dev = gpu!();
    let v: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1e20 + 1e15 } else { 1e20 - 1e15 }).collect();
    let xs = v;
    let mean = xs.iter().map(|&a| a as f64).sum::<f64>() / 8.0;
    let var = xs.iter().map(|&a| (a as f64 - mean).powi(2)).sum::<f64>() / 8.0;
    let x = Tensor::from_vec(xs.clone(), [2, 1, 2, 2]).unwrap().to_device(dev).unwrap().requires_grad_();
    let (g, b) = (
        Tensor::ones([1]).to_device(dev).unwrap().detach().requires_grad_(),
        Tensor::zeros([1]).to_device(dev).unwrap().detach().requires_grad_(),
    );
    let (rm, rv) = (Tensor::zeros([1]).to_device(dev).unwrap(), Tensor::ones([1]).to_device(dev).unwrap());
    let bn = BatchNorm2d::from_tensors(g.clone(), b.clone(), rm, rv.clone(), 1e-3, 0.5).unwrap();
    let y = bn.forward_act(&x, Activation::Identity).unwrap();
    let ys = y.to_vec::<f32>().unwrap();
    for (i, (&yv, &xv)) in ys.iter().zip(&xs).enumerate() {
        let want = (xv as f64 - mean) / (var + 1e-3).sqrt();
        assert!((yv as f64 - want).abs() <= 1e-3, "y[{i}]: {yv} vs {want}");
    }
    let rv_want = 0.5 + 0.5 * var * 8.0 / 7.0;
    let rv_got = rv.to_vec::<f32>().unwrap()[0] as f64;
    assert!((rv_got - rv_want).abs() <= 1e-3 * rv_want, "running_var {rv_got} vs {rv_want}");
    let w = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.5], [2, 1, 2, 2]).unwrap();
    let grads = y.mul(&w.to_device(dev).unwrap()).unwrap().sum_all().unwrap().backward().unwrap();
    // dβ = Σw, dγ = Σ w · x̂; dx is finite and of order 1 / σ.
    let db = grads.get(&b).unwrap().to_vec::<f32>().unwrap()[0];
    let dg = grads.get(&g).unwrap().to_vec::<f32>().unwrap()[0] as f64;
    let dg_want: f64 = ys.iter().zip(w.to_vec::<f32>().unwrap()).map(|(&a, b)| a as f64 * b as f64).sum();
    let db_want: f32 = w.to_vec::<f32>().unwrap().iter().sum();
    assert!((db - db_want).abs() <= 1e-5, "dbias {db} vs {db_want}");
    assert!((dg - dg_want).abs() <= 1e-3 * (1.0 + dg_want.abs()), "dweight {dg} vs {dg_want}");
    assert!(grads.get(&x).unwrap().to_vec::<f32>().unwrap().iter().all(|d| d.is_finite() && d.abs() < 1e-13));
}

/// Gradient accumulation and global-norm clipping on the device (fused `buf += g`, the norm and the clip factor as
/// device scalars, no readback) give the CPU accumulator's gradients, clipped and unclipped.
#[test]
fn grad_accumulator_on_gpu_matches_cpu() {
    let dev = gpu!();
    let rng = &mut Rng::seed(21);
    let shapes: [&[usize]; 3] = [&[8, 3, 3, 3], &[8], &[5, 7]];
    let params: Vec<Tensor> =
        shapes.iter().map(|s| Tensor::randn(*s, 0.0, 1.0, rng).requires_grad_()).collect();
    let dparams: Vec<Tensor> =
        params.iter().map(|p| p.to_device(dev).unwrap().detach().requires_grad_()).collect();
    let passes: Vec<Vec<Tensor>> =
        (0..3).map(|_| shapes.iter().map(|s| Tensor::randn(*s, 0.0, 2.0, rng)).collect()).collect();
    // Gradients of Σ p · g for each pass: exactly g.
    let grads_of = |ps: &[Tensor], gs: &[Tensor]| {
        let mut loss = Tensor::zeros_with([], folo_core::DType::F32, ps[0].device()).unwrap();
        for (p, g) in ps.iter().zip(gs) {
            loss = loss.add(&p.mul(&g.to_device(p.device()).unwrap()).unwrap().sum_all().unwrap()).unwrap();
        }
        loss.backward().unwrap()
    };
    for max_norm in [1e9, 5.0] {
        let mut cpu = GradAccumulator::new(&params);
        let mut gpu = GradAccumulator::new(&dparams);
        for gs in &passes {
            cpu.add(&grads_of(&params, gs)).unwrap();
            gpu.add(&grads_of(&dparams, gs)).unwrap();
        }
        let n = cpu.clip_grad_norm(max_norm).unwrap();
        gpu.clip_grad_norm(max_norm).unwrap();
        assert!(max_norm > 1e8 || n > max_norm, "the clipped case clips (norm {n})");
        for (p, d) in params.iter().zip(&dparams) {
            let (c, g) = (cpu.grad(p).unwrap(), gpu.grad(d).unwrap());
            assert_eq!(g.device(), dev);
            close(g, c, 1e-5, 1e-5, &format!("grad, max_norm {max_norm}"));
        }
    }
}

/// The device check of `RunningStats::all_finite` sees a NaN or an infinity in any BatchNorm buffer.
#[test]
fn running_stats_all_finite_on_gpu() {
    let dev = gpu!();
    let mut m = Sequential::new().push(BatchNorm2d::new(3)).push(BatchNorm2d::new(4));
    m.to_device(dev).unwrap();
    assert!(RunningStats::all_finite(&m).unwrap());
    let snap = RunningStats::capture(&m).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (_, buf) = m.named_buffers().into_iter().nth(3).unwrap();
        let mut v = buf.to_vec::<f32>().unwrap();
        v[1] = bad;
        let _g = no_grad();
        buf.copy_from(&Tensor::from_vec(v, buf.shape()).unwrap().to_device(dev).unwrap()).unwrap();
        assert!(!RunningStats::all_finite(&m).unwrap(), "{bad}");
        snap.restore().unwrap();
        assert!(RunningStats::all_finite(&m).unwrap());
    }
}

/// Finite gradients whose squares overflow `f32` (1e20) still get the CPU's (`f64`) norm on the device, so they are
/// clipped, not zeroed.
#[test]
fn device_clip_survives_huge_finite_gradients() {
    let dev = gpu!();
    let p = Tensor::zeros([4]).requires_grad_();
    let dp = p.to_device(dev).unwrap().detach().requires_grad_();
    let g = Tensor::from_vec(vec![1e20f32, -2e20, 3e19, 0.0], [4]).unwrap();
    let (mut cpu, mut gpu) =
        (GradAccumulator::new(std::slice::from_ref(&p)), GradAccumulator::new(std::slice::from_ref(&dp)));
    cpu.add(&p.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    gpu.add(&dp.mul(&g.to_device(dev).unwrap()).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
    cpu.clip_grad_norm(10.0).unwrap();
    gpu.clip_grad_norm(10.0).unwrap();
    let (c, d) = (cpu.grad(&p).unwrap(), gpu.grad(&dp).unwrap());
    close(d, c, 1e-5, 1e-5, "clipped huge gradient");
    assert!(d.to_vec::<f32>().unwrap()[1] < -5.0, "clipped to norm 10, not zeroed");
}
