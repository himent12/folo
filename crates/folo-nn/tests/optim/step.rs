//! SGD, AdamW, parameter groups, and LR schedule application tests.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::optim::*;
use folo_nn::*;

use crate::{G1, G2, close, grads_for, param};

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
            OptimKind::MuSgd => unreachable!("not in the loop"),
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
    // 30 epochs of 13 steps: T = 390, W = 100 (docs/MODEL.md §10.4).
    let mut sched = LrSchedule::new(DecayKind::Linear, 30, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
    sched.apply(&mut opt);
    let g = opt.groups();
    assert_eq!((g[0].lr, g[1].lr, g[2].lr), (0.0, 0.0, 0.1));
    assert_eq!(g[0].momentum, 0.8);
    for _ in 0..50 {
        sched.advance();
    }
    sched.apply(&mut opt);
    let target = 0.01 * ((1.0 - 50.0 / 389.0) * 0.99 + 0.01);
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
fn standard_groups_puts_v2_scale_and_gate_params_in_no_decay_bn_group() {
    struct V2Like {
        w3x3: Tensor,
        gamma_id: Tensor,
        alpha_raw: Tensor,
        gamma_id_extra: Tensor,
        bias: Tensor,
    }
    impl Module for V2Like {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("block.w3x3", &self.w3x3, TensorKind::Parameter);
            f("block.gamma_id", &self.gamma_id, TensorKind::Parameter);
            f("head.o2o_cls.0.suppress.alpha_raw", &self.alpha_raw, TensorKind::Parameter);
            f("block.gamma_id_extra", &self.gamma_id_extra, TensorKind::Parameter);
            f("head.bias", &self.bias, TensorKind::Parameter);
        }
    }
    let m = V2Like {
        w3x3: Tensor::ones([4, 4, 3, 3]).requires_grad_(),
        gamma_id: Tensor::ones([4]).requires_grad_(),
        alpha_raw: Tensor::full([3], -2.0).requires_grad_(),
        gamma_id_extra: Tensor::ones([4, 4]).requires_grad_(),
        bias: Tensor::zeros([3]).requires_grad_(),
    };
    let sg = StandardGroups::from_module(&m);
    let names = |v: &[(String, Tensor)]| v.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&sg.bn), ["block.gamma_id", "head.o2o_cls.0.suppress.alpha_raw"]);
    assert_eq!(names(&sg.decay), ["block.w3x3", "block.gamma_id_extra"]);
    assert_eq!(names(&sg.bias), ["head.bias"]);
    // Top-level parameter names without dots match too.
    struct TopScale(Tensor, Tensor, Tensor);
    impl Module for TopScale {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("gamma_id", &self.0, TensorKind::Parameter);
            f("alpha_raw", &self.1, TensorKind::Parameter);
            // Rank-2 tensor with an exact scale name must keep weight decay.
            f("proj.gamma_id", &self.2, TensorKind::Parameter);
        }
    }
    let top = StandardGroups::from_module(&TopScale(
        Tensor::ones([4]).requires_grad_(),
        Tensor::ones([4]).requires_grad_(),
        Tensor::ones([4, 4]).requires_grad_(),
    ));
    assert_eq!(names(&top.bn), ["gamma_id", "alpha_raw"]);
    assert_eq!(names(&top.decay), ["proj.gamma_id"]);
    assert!(top.bias.is_empty());
}

#[test]
fn boosted_groups_and_bias_warmup() {
    let mut rng = Rng::seed(0);
    let model = Sequential::new()
        .push_named("stem", ConvBnAct::new(3, 4, 3, 1, 1, Activation::Silu, &mut rng))
        .push_named("out", Conv2d::new(Conv2dConfig::new(4, 2, 1), &mut rng));
    let gs = StandardGroups::from_module(&model).into_boosted_groups(
        0.01,
        0.9,
        5e-4,
        |n| n.starts_with("out."),
        3.0,
    );
    let summary: Vec<(&str, usize, f64, f64)> =
        gs.iter().map(|g| (g.name.as_str(), g.params.len(), g.base_lr, g.weight_decay)).collect();
    assert_eq!(
        summary,
        [
            ("decay", 1, 0.01, 5e-4),
            ("decay.cls", 1, 0.03, 5e-4),
            ("bn", 1, 0.01, 0.0),
            ("bias", 1, 0.01, 0.0),
            ("bias.cls", 1, 0.03, 0.0)
        ]
    );
    assert!(
        is_bias_group("bias")
            && is_bias_group("bias.cls")
            && !is_bias_group("biased")
            && !is_bias_group("bn")
    );
    assert!(is_muon_group("decay") && is_muon_group("decay.cls") && !is_muon_group("bn"));
    // Warmup starts both bias groups at warmup_bias_lr, the others at 0.
    let sched = LrSchedule::new(DecayKind::Linear, 30, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
    assert_eq!(sched.lr_at(0, &gs[4]), 0.1);
    assert_eq!(sched.lr_at(0, &gs[1]), 0.0);
}
