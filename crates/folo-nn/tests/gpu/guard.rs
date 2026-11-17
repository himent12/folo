//! The optimizer step guard (`folo_core::StepGuard`, docs/MODEL.md §8.8): on the GPU, SGD, AdamW and MuSGD steps
//! and the EMA update encoded under a guard whose flag is not finite leave every parameter, optimizer state and
//! average exactly as they were; under a finite flag they are bit for bit the unguarded step.

use super::*;
use folo_core::StepGuard;

type Setup = (Net, Box<dyn Optimizer>, ModelEma<Net>);

/// A net, its optimizer of `kind` and an EMA, on `dev`, after two plain steps (so every state buffer exists and
/// is not zero).
fn setup(dev: Device, kind: OptimKind, x: &Tensor) -> Setup {
    let mut net = Net::new(5);
    net.to_device(dev).unwrap();
    let sg = StandardGroups::from_module(&net);
    let mut opt: Box<dyn Optimizer> = match kind {
        OptimKind::AdamW => Box::new(AdamW::new(sg.into_groups(1e-3, 0.9, 5e-4)).unwrap()),
        OptimKind::Sgd => Box::new(Sgd::new(sg.into_groups(0.05, 0.9, 5e-4)).unwrap()),
        OptimKind::MuSgd => Box::new(
            MuSgd::new(sg.into_boosted_groups(0.05, 0.9, 5e-4, |n| n.starts_with("head."), 3.0)).unwrap(),
        ),
    };
    let mut ema = ModelEma::with_params(Net::new(6), &net, 0.9, 2.0).unwrap();
    ema.to_device(dev).unwrap();
    for _ in 0..2 {
        let grads = grads_of(&net, x);
        opt.step(&grads).unwrap();
        ema.update(&net).unwrap();
    }
    (net, opt, ema)
}

fn grads_of(net: &Net, x: &Tensor) -> folo_core::Grads {
    let out = net.forward(x);
    out.square().unwrap().mean_all().unwrap().add(&out.mean_all().unwrap()).unwrap().backward().unwrap()
}

/// Every value the step may write: the model, the optimizer state and the EMA, as bits.
fn snapshot((net, opt, ema): &Setup) -> Vec<(String, Vec<u32>)> {
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let mut v: Vec<(String, Vec<u32>)> =
        net.named_tensors().iter().map(|t| (format!("model {}", t.name), bits(&t.tensor))).collect();
    v.extend(opt.state_tensors().unwrap().iter().map(|(n, t)| (format!("state {n}"), bits(t))));
    v.extend(ema.module().named_tensors().iter().map(|t| (format!("ema {}", t.name), bits(&t.tensor))));
    v
}

/// Whether `dev` has a step guard (a GPU with fewer than five storage buffers per stage has none; the trainer then
/// reads the flag back, and these tests have nothing to check).
fn has_guard(dev: Device) -> bool {
    let one = Tensor::from_vec(vec![1.0f32], [1]).unwrap().to_device(dev).unwrap();
    let guard = StepGuard::new(&one).unwrap();
    if guard.is_none() {
        eprintln!("skipped: this GPU has no step guard");
    }
    guard.is_some()
}

/// One step with `grads` under a guard on `flag`.
fn guarded_step(s: &mut Setup, grads: &folo_core::Grads, flag: f32, dev: Device) {
    let f = Tensor::from_vec(vec![flag], [1]).unwrap().to_device(dev).unwrap();
    let guard = StepGuard::new(&f).unwrap().expect("the GPU has a step guard");
    s.1.step(grads).unwrap();
    s.2.update(&s.0).unwrap();
    drop(guard);
}

#[test]
fn guarded_steps_skip_exactly_and_otherwise_match() {
    let dev = gpu!();
    if !has_guard(dev) {
        return;
    }
    let x = Tensor::rand_uniform([2, 3, 16, 16], -1.0, 1.0, &mut Rng::seed(4)).to_device(dev).unwrap();
    for kind in [OptimKind::Sgd, OptimKind::AdamW, OptimKind::MuSgd] {
        let what = kind.as_str();
        // Not finite: nothing moves.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut s = setup(dev, kind, &x);
            // The forward updates the BatchNorm statistics (the trainer's batch, not its step).
            let grads = grads_of(&s.0, &x);
            let before = snapshot(&s);
            guarded_step(&mut s, &grads, bad, dev);
            let after = snapshot(&s);
            for ((name, a), (_, b)) in before.iter().zip(&after) {
                assert!(a == b, "{what}, flag {bad}: {name} changed under a skipping guard");
            }
            // The guard is gone: a plain step moves the model again.
            let grads = grads_of(&s.0, &x);
            s.1.step(&grads).unwrap();
            assert_ne!(snapshot(&s)[0], before[0], "{what}: a step after the guard");
        }
        // Finite: the plain step, bit for bit.
        let (mut a, mut b) = (setup(dev, kind, &x), setup(dev, kind, &x));
        let grads = grads_of(&a.0, &x);
        guarded_step(&mut a, &grads, 3.5, dev);
        let grads = grads_of(&b.0, &x);
        b.1.step(&grads).unwrap();
        b.2.update(&b.0).unwrap();
        for ((name, ga), (_, gb)) in snapshot(&a).iter().zip(&snapshot(&b)) {
            assert!(ga == gb, "{what}: {name} differs from the unguarded step");
        }
    }
    // The guarded kernels can be named in a kernel list and precompiled by the next run.
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    assert_eq!(ctx.kernels_without_registry(), Vec::<String>::new());
}

/// A guard needs one F32 element and is one per thread; without one, fallbacks may run.
#[test]
fn guard_rules() {
    let dev = gpu!();
    assert!(folo_core::step_guard_allows(dev).unwrap());
    let two = Tensor::zeros_with([2], folo_core::DType::F32, dev).unwrap();
    assert!(StepGuard::new(&two).is_err());
    let nan = Tensor::from_vec(vec![f32::NAN], [1]).unwrap().to_device(dev).unwrap();
    let Some(g) = StepGuard::new(&nan).unwrap() else {
        return eprintln!("skipped: this GPU has no step guard");
    };
    assert!(StepGuard::new(&nan).is_err(), "one guard per thread");
    assert!(!folo_core::step_guard_allows(dev).unwrap(), "a fallback under a skipping guard does nothing");
    assert!(
        folo_core::step_guard_allows(folo_core::Device::Cpu).unwrap(),
        "the guard covers its own device only"
    );
    // The guard holds the flag's value as of its start: a later write to the flag moves neither the device's
    // decision nor the fallbacks'.
    nan.copy_from(&Tensor::full_with([1], 1.0, folo_core::DType::F32, dev).unwrap()).unwrap();
    assert!(!folo_core::step_guard_allows(dev).unwrap(), "the fallback check after a write to the flag");
    let p = Tensor::zeros_with([300], folo_core::DType::F32, dev).unwrap();
    let ones = Tensor::full_with([300], 1.0, folo_core::DType::F32, dev).unwrap();
    assert!(p.optimizer_step_(&ones, &[], &folo_core::backend::OptimStep::Axpby { a: 1.0, b: 2.0 }).unwrap());
    assert!(
        p.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0),
        "the device's guard after a write to the flag"
    );
    // The guard is this thread's: an update another thread encodes meanwhile on the same device still runs.
    let moved = std::thread::spawn(move || {
        let p = Tensor::zeros_with([300], folo_core::DType::F32, dev).unwrap();
        let g = Tensor::full_with([300], 1.0, folo_core::DType::F32, dev).unwrap();
        let step = folo_core::backend::OptimStep::Axpby { a: 1.0, b: 2.0 };
        assert!(p.optimizer_step_(&g, &[], &step).unwrap());
        assert!(Tensor::optimizer_step_many_(&[&p], &[&g], &[vec![]], &step).unwrap());
        p.to_vec::<f32>().unwrap()
    })
    .join()
    .unwrap();
    assert!(moved.iter().all(|&v| v == 4.0), "{:?}", &moved[..4]);
    drop(g);
    assert!(folo_core::step_guard_allows(dev).unwrap());
    // The CPU has none: the caller reads the flag itself.
    assert!(StepGuard::new(&Tensor::from_vec(vec![1.0f32], [1]).unwrap()).unwrap().is_none());
}

/// Gradients by parameter shape (the test's parameters all differ in shape).
struct ByShape(Vec<Tensor>);

impl GradSource for ByShape {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.iter().find(|g| g.shape() == param.shape())
    }
}

/// A group the multi-tensor update declines (a transposed parameter) takes the per-parameter path, which runs on
/// worker threads for a large step: under a guard it stays on the guarded thread, so a skip still writes nothing.
#[test]
fn per_parameter_fallback_stays_guarded() {
    let dev = gpu!();
    if !has_guard(dev) {
        return;
    }
    let mut rng = Rng::seed(7);
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let one = |dims: [usize; 2]| Tensor::full_with(dims, 1.0, folo_core::DType::F32, dev).unwrap();
    for kind in [OptimKind::Sgd, OptimKind::AdamW] {
        // 90,000 elements: above the threshold for updating parameters in parallel.
        let big = Tensor::randn([300, 300], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
        let small = Tensor::randn([8, 4], 0.0, 1.0, &mut rng).to_device(dev).unwrap().t().unwrap();
        let params = vec![("big".to_string(), big.clone()), ("small".to_string(), small.clone())];
        let groups = vec![ParamGroup::new("decay", params).lr(0.1)];
        let mut opt: Box<dyn Optimizer> = match kind {
            OptimKind::Sgd => Box::new(Sgd::new(groups).unwrap()),
            _ => Box::new(AdamW::new(groups).unwrap()),
        };
        let grads = ByShape(vec![one([300, 300]), one([4, 8])]);
        let before = (bits(&big), bits(&small));
        let flag = Tensor::from_vec(vec![f32::NAN], [1]).unwrap().to_device(dev).unwrap();
        let guard = StepGuard::new(&flag).unwrap().expect("a guard");
        opt.step(&grads).unwrap();
        drop(guard);
        assert!(bits(&big) == before.0, "{kind:?}: the large parameter moved under a skipping guard");
        assert!(bits(&small) == before.1, "{kind:?}: the transposed parameter moved under a skipping guard");
        opt.step(&grads).unwrap();
        assert!(bits(&big) != before.0 && bits(&small) != before.1, "{kind:?}: a plain step moves both");
    }
}
