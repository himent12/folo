//! Optimizers, groups, accumulation, clipping, schedule application and EMA against hand-computed values.

use folo_core::rng::Rng;
use folo_core::{Grads, Tensor, no_grad};
use folo_nn::optim::*;
use folo_nn::*;

fn close(a: &[f32], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x as f64 - y).abs() <= tol * (1.0 + y.abs()), "element {i}: {x} vs {y}");
    }
}

/// Gradients for `p` from a backward pass of `sum(p * g)`, which is exactly `g`.
fn grads_for(p: &Tensor, g: &[f32]) -> Grads {
    let gt = Tensor::from_slice(g, p.shape().clone()).unwrap();
    p.mul(&gt).unwrap().sum_all().unwrap().backward().unwrap()
}

fn param(v: &[f32]) -> Tensor {
    Tensor::from_slice(v, [v.len()]).unwrap().requires_grad_()
}

const G1: [f32; 2] = [0.5, 0.25];
const G2: [f32; 2] = [-1.0, 2.0];

#[test]
fn sgd_nesterov_with_l2_weight_decay() {
    let p = param(&[1.0, -2.0]);
    let g = ParamGroup::new("decay", vec![("p".into(), p.clone())]).lr(0.1).momentum(0.9).weight_decay(0.01);
    let mut opt = Sgd::new(vec![g]).unwrap();
    opt.step(&grads_for(&p, &G1)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[0.9031, -2.0437], 1e-6);
    close(&opt.momentum_buffer(0, 0).unwrap().to_vec::<f32>().unwrap(), &[0.51, 0.23], 1e-6);
    opt.step(&grads_for(&p, &G2)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[1.05007411, -2.43844697], 1e-6);
    close(&opt.momentum_buffer(0, 0).unwrap().to_vec::<f32>().unwrap(), &[-0.531969, 2.186563], 1e-6);
    assert_eq!(opt.step_count(), 2);
}

#[test]
fn sgd_plain_momentum_and_missing_grads() {
    let (p, q) = (param(&[1.0, -2.0]), param(&[3.0]));
    let g = ParamGroup::new("g", vec![("p".into(), p.clone()), ("q".into(), q.clone())])
        .lr(0.1)
        .momentum(0.9)
        .nesterov(false);
    let mut opt = Sgd::new(vec![g]).unwrap();
    opt.step(&grads_for(&p, &G1)).unwrap();
    opt.step(&grads_for(&p, &G2)).unwrap();
    close(&p.to_vec::<f32>().unwrap(), &[1.005, -2.2475], 1e-6);
    // `q` never had a gradient: untouched, zero state (which is still written to checkpoints).
    assert_eq!(q.to_vec::<f32>().unwrap(), [3.0]);
    let st = opt.state_tensors().unwrap();
    assert_eq!(
        st.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["p.momentum_buffer", "q.momentum_buffer"]
    );
    assert_eq!(st[1].1.to_vec::<f32>().unwrap(), [0.0]);
}

#[test]
fn adamw_decoupled_decay_and_groups() {
    let (p, b) = (param(&[1.0, -2.0]), param(&[1.0, -2.0]));
    let groups = vec![
        ParamGroup::new("decay", vec![("w".into(), p.clone())]).lr(0.01).momentum(0.9).weight_decay(0.1),
        ParamGroup::new("bias", vec![("b".into(), b.clone())]).lr(0.01).momentum(0.9),
    ];
    let mut opt = AdamW::new(groups).unwrap();
    for g in [G1, G2] {
        let mut grads = grads_for(&p, &g);
        let gb = grads_for(&b, &g);
        grads.insert(&b, gb.get(&b).unwrap().clone());
        opt.step(&grads).unwrap();
    }
    close(&p.to_vec::<f32>().unwrap(), &[0.9916720354238566, -2.0142066571261106], 1e-6);
    close(&b.to_vec::<f32>().unwrap(), &[0.9936610354240566, -2.0182146571257107], 1e-6);
    let c = opt.contract();
    assert_eq!(c.kind, OptimKind::AdamW);
    assert_eq!(c.groups[0].betas, Some([0.9, 0.999]));
    assert_eq!(c.groups[0].momentum, None);
}

#[test]
fn generic_path_matches_fast_path_bitwise() {
    // A transposed (non-contiguous) parameter takes the op-by-op path; the result must be identical.
    let vals: Vec<f32> = (0..6).map(|i| i as f32 * 0.3 - 0.7).collect();
    let fast = Tensor::from_slice(&vals, [2, 3]).unwrap().requires_grad_();
    let base = Tensor::from_slice(&vals, [3, 2]).unwrap();
    let base_t = base.t().unwrap();
    {
        let _g = no_grad();
        base_t.copy_from(&fast.detach()).unwrap();
    }
    let slow = base_t.requires_grad_();
    assert!(!slow.is_contiguous());
    let gv: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.11).collect();
    for kind in [OptimKind::Sgd, OptimKind::AdamW] {
        let mk = |t: &Tensor| {
            ParamGroup::new("decay", vec![("w".into(), t.clone())]).lr(0.05).momentum(0.9).weight_decay(0.01)
        };
        let (mut a, mut b): (Box<dyn Optimizer>, Box<dyn Optimizer>) = match kind {
            OptimKind::Sgd => {
                (Box::new(Sgd::new(vec![mk(&fast)]).unwrap()), Box::new(Sgd::new(vec![mk(&slow)]).unwrap()))
            }
            OptimKind::AdamW => (
                Box::new(AdamW::new(vec![mk(&fast)]).unwrap()),
                Box::new(AdamW::new(vec![mk(&slow)]).unwrap()),
            ),
        };
        for _ in 0..3 {
            a.step(&grads_for(&fast, &gv)).unwrap();
            b.step(&grads_for(&slow, &gv)).unwrap();
            let (x, y) = (fast.to_vec::<f32>().unwrap(), slow.to_vec::<f32>().unwrap());
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{kind:?}"
            );
        }
    }
}

#[test]
fn standard_groups_and_schedule_application() {
    let mut rng = Rng::seed(0);
    let model = Sequential::new()
        .push_named("stem", ConvBnAct::new(3, 4, 3, 1, 1, Activation::Silu, &mut rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, 2, 1), &mut rng));
    let sg = StandardGroups::from_module(&model);
    let names = |v: &[(String, Tensor)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&sg.decay), ["stem.conv.weight", "out.weight"]);
    assert_eq!(names(&sg.bn), ["stem.bn.weight"]);
    assert_eq!(names(&sg.bias), ["stem.bn.bias", "out.bias"]);
    let groups = sg.into_groups(0.01, 0.937, 5e-4);
    assert_eq!(groups.iter().map(|g| g.weight_decay).collect::<Vec<_>>(), [5e-4, 0.0, 0.0]);
    let mut opt = Sgd::new(groups).unwrap();
    let mut sched = LrSchedule::new(DecayKind::Linear, 10, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
    sched.apply(&mut opt);
    let g = opt.groups();
    assert_eq!((g[0].lr, g[1].lr, g[2].lr), (0.0, 0.0, 0.1));
    assert_eq!(g[0].momentum, 0.8);
    for _ in 0..50 {
        sched.advance();
    }
    sched.apply(&mut opt);
    let target = 0.01 * ((1.0 - 50.0 / 129.0) * 0.99 + 0.01);
    assert!((opt.groups()[0].lr - target * 0.5).abs() < 1e-15);
    assert!((opt.groups()[2].lr - (0.1 + (target - 0.1) * 0.5)).abs() < 1e-15);
    assert!((opt.groups()[1].momentum - (0.8 + 0.137 * 0.5)).abs() < 1e-15);
    // Duplicate parameters are rejected.
    let p = param(&[1.0]);
    assert!(
        Sgd::new(vec![
            ParamGroup::new("a", vec![("p".into(), p.clone())]),
            ParamGroup::new("b", vec![("p".into(), p)])
        ])
        .is_err()
    );
}

#[test]
fn accumulation_and_clipping() {
    let (p, q) = (param(&[1.0, 2.0]), param(&[3.0]));
    let mut acc = GradAccumulator::new(&[p.clone(), q.clone()]);
    // One backward where p and q receive the same gradient tensor (shared storage): buffers must not alias.
    let x = Tensor::from_slice(&[1.0f32], [1]).unwrap();
    let y = p.sum_all().unwrap().add(&q.mul(&x).unwrap().sum_all().unwrap()).unwrap();
    acc.add(&y.backward().unwrap()).unwrap();
    acc.add(&grads_for(&p, &[3.0, 4.0])).unwrap();
    assert_eq!(acc.passes(), 2);
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0, 5.0]);
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [1.0]);
    // ‖g‖ = sqrt(16 + 25 + 1) = sqrt(42); clipped to 1 by 1 / (n + 1e-6).
    let n = acc.clip_grad_norm(1.0).unwrap();
    assert!((n - 42f64.sqrt()).abs() < 1e-12);
    let c = (1.0 / (42f64.sqrt() + 1e-6)) as f32;
    assert_eq!(acc.grad(&p).unwrap().to_vec::<f32>().unwrap(), [4.0 * c, 5.0 * c]);
    assert!((acc.global_norm().unwrap() - 1.0).abs() < 1e-5);
    assert_eq!(acc.clip_grad_norm(10.0).unwrap(), acc.global_norm().unwrap(), "below the limit: unchanged");
    acc.zero();
    assert!(!acc.has_grads() && acc.grad(&p).is_none());
    acc.add(&grads_for(&q, &[2.0])).unwrap();
    assert!(acc.grad(&p).is_none(), "a parameter without a gradient in this window has none");
    assert_eq!(acc.grad(&q).unwrap().to_vec::<f32>().unwrap(), [2.0]);
}

#[test]
fn ema_values() {
    let live = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::full([1, 1, 1, 1], 2.0).requires_grad_(),
        Some(Tensor::zeros([1]).requires_grad_()),
    )
    .unwrap();
    let shadow = Conv2d::from_tensors(
        Conv2dConfig::new(1, 1, 1),
        Tensor::zeros([1, 1, 1, 1]),
        Some(Tensor::zeros([1])),
    )
    .unwrap();
    let mut ema = ModelEma::new(shadow, &live).unwrap();
    assert_eq!(ema.module().weight.to_vec::<f32>().unwrap(), [2.0]);
    assert!(!ema.module().weight.requires_grad());
    let d = |u: f64| 0.9999 * (1.0 - (-u / 2000.0).exp());
    assert!((d(1.0) - 4.99825e-4).abs() < 1e-9);
    let mut want = 2.0f64;
    for (u, v) in [(1.0, 4.0f32), (2.0, 6.0)] {
        {
            let _g = no_grad();
            live.weight.fill_assign(v).unwrap();
        }
        ema.update(&live).unwrap();
        want = d(u) * want + (1.0 - d(u)) * v as f64;
        close(&ema.module().weight.to_vec::<f32>().unwrap(), &[want], 1e-6);
    }
    assert_eq!(ema.updates(), 2);
    // BatchNorm running statistics are tracked too.
    let bn = BatchNorm2d::new(2);
    let mut e2 = ModelEma::new(BatchNorm2d::new(2), &bn).unwrap();
    bn.forward(&Tensor::full([2, 2, 2, 2], 3.0)).unwrap();
    e2.update(&bn).unwrap();
    let rm = 0.03 * 3.0;
    close(&e2.module().running_mean.to_vec::<f32>().unwrap(), &[(1.0 - d(1.0)) * rm; 2], 1e-6);
}
