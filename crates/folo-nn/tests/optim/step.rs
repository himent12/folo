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

#[test]
fn large_multi_group_sgd_and_adamw_parallel_plan_is_exact() {
    let mut rng = Rng::seed(77);
    // Exceeds PAR_THRESHOLD (65,536) across 3 groups (decay, bn, bias) to exercise ParamPlan's parallel reuse
    // (8 * 9,216 + 12 * 512 = 79,872 >= 65,536).
    let make_params = |rng: &mut Rng| -> Vec<Vec<(String, Tensor)>> {
        vec![
            (0..8)
                .map(|i| (format!("w{i}"), Tensor::randn([32, 32, 3, 3], 0.0, 0.5, rng).requires_grad_()))
                .collect(),
            (0..6)
                .map(|i| (format!("bn{i}"), Tensor::randn([512], 1.0, 0.1, rng).requires_grad_()))
                .collect(),
            (0..6).map(|i| (format!("b{i}"), Tensor::randn([512], 0.0, 0.1, rng).requires_grad_())).collect(),
        ]
    };
    let ps_sgd = make_params(&mut rng);
    let total_numel: usize = ps_sgd.iter().flat_map(|g| g.iter()).map(|(_, t)| t.numel()).sum();
    assert!(total_numel >= 65_536, "fixture must exceed PAR_THRESHOLD: {total_numel}");
    let ps_adam = make_params(&mut rng);
    let mk_groups = |ps: &[Vec<(String, Tensor)>]| {
        vec![
            ParamGroup::new("decay", ps[0].clone()).lr(0.01).momentum(0.9).weight_decay(5e-4),
            ParamGroup::new("bn", ps[1].clone()).lr(0.01).momentum(0.9),
            ParamGroup::new("bias", ps[2].clone()).lr(0.01).momentum(0.9),
        ]
    };
    let mut sgd = Sgd::new(mk_groups(&ps_sgd)).unwrap();
    let mut adam = AdamW::new(mk_groups(&ps_adam)).unwrap();
    // Clone initial values for single-threaded reference comparison.
    let clone_ps = |ps: &[Vec<(String, Tensor)>]| -> Vec<Vec<(String, Tensor)>> {
        ps.iter()
            .map(|g| {
                g.iter()
                    .map(|(n, t)| {
                        (
                            n.clone(),
                            Tensor::from_vec(t.to_vec::<f32>().unwrap(), t.shape().clone())
                                .unwrap()
                                .requires_grad_(),
                        )
                    })
                    .collect()
            })
            .collect()
    };
    let ref_sgd_ps = clone_ps(&ps_sgd);
    let ref_adam_ps = clone_ps(&ps_adam);
    let mut ref_sgd = Sgd::new(mk_groups(&ref_sgd_ps)).unwrap();
    let mut ref_adam = AdamW::new(mk_groups(&ref_adam_ps)).unwrap();

    let pool_par = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let pool_ser = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    for _ in 0..3 {
        let mut gs_sgd = folo_core::Grads::default();
        let mut gs_ref_sgd = folo_core::Grads::default();
        for (g, rg) in ps_sgd.iter().zip(&ref_sgd_ps) {
            for ((_, p), (_, rp)) in g.iter().zip(rg) {
                let gv = Tensor::randn(p.shape().clone(), 0.0, 0.1, &mut rng);
                gs_sgd.insert(p, gv.clone());
                gs_ref_sgd.insert(rp, gv);
            }
        }
        pool_par.install(|| sgd.step(&gs_sgd).unwrap());
        pool_ser.install(|| ref_sgd.step(&gs_ref_sgd).unwrap());

        let mut gs_adam = folo_core::Grads::default();
        let mut gs_ref_adam = folo_core::Grads::default();
        for (g, rg) in ps_adam.iter().zip(&ref_adam_ps) {
            for ((_, p), (_, rp)) in g.iter().zip(rg) {
                let gv = Tensor::randn(p.shape().clone(), 0.0, 0.1, &mut rng);
                gs_adam.insert(p, gv.clone());
                gs_ref_adam.insert(rp, gv);
            }
        }
        pool_par.install(|| adam.step(&gs_adam).unwrap());
        pool_ser.install(|| ref_adam.step(&gs_ref_adam).unwrap());
    }
    for (g, rg) in ps_sgd.iter().zip(&ref_sgd_ps).chain(ps_adam.iter().zip(&ref_adam_ps)) {
        for ((n, p), (_, rp)) in g.iter().zip(rg) {
            let a: Vec<u32> = p.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
            let b: Vec<u32> = rp.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect();
            assert_eq!(a, b, "parallel ParamPlan diverged on `{n}`");
        }
    }
}

#[test]
fn adamw_per_param_step_bias_correction_and_contract() {
    // Parameter `bb` stays frozen (no grad) for 50 steps while `head` trains in the SAME group.
    // When `bb` unfreezes, its first update must use t = 1 bias correction (matching a fresh AdamW),
    // not t = 51 (which would overshoot by ~3x because sqrt(1 - 0.999^51) >> sqrt(1 - 0.999)).
    let bb = param(&[1.0, -2.0]);
    let head = param(&[0.5, 0.5]);
    let grp = ParamGroup::new("decay", vec![("bb".into(), bb.clone()), ("head".into(), head.clone())])
        .lr(0.01)
        .momentum(0.9)
        .weight_decay(0.05);
    let mut opt = AdamW::new(vec![grp]).unwrap();

    for _ in 0..50 {
        opt.step(&grads_for(&head, &G1)).unwrap();
    }
    assert_eq!(opt.step_count(), 50);
    assert_eq!(opt.param_step(0, 0), Some(0));
    assert_eq!(opt.param_step(0, 1), Some(50));
    assert_eq!(bb.to_vec::<f32>().unwrap(), [1.0, -2.0]);

    // Contract records only the differing `param_steps` entry and round-trips through JSON + resume.
    let c = opt.contract();
    assert_eq!(c.param_steps.len(), 1);
    assert_eq!(c.param_steps.get("bb"), Some(&0));
    let json = serde_json::to_string(&c).unwrap();
    let parsed: OptimContract = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, c);

    // Resume into a fresh AdamW and verify `bb`'s first unfrozen step matches a fresh 1-step AdamW bit-for-bit.
    let bb_res = param(&[1.0, -2.0]);
    let head_res = Tensor::from_vec(head.to_vec::<f32>().unwrap(), [2]).unwrap().requires_grad_();
    let mut opt_res = AdamW::new(vec![
        ParamGroup::new("decay", vec![("bb".into(), bb_res.clone()), ("head".into(), head_res.clone())])
            .lr(0.01)
            .momentum(0.9)
            .weight_decay(0.05),
    ])
    .unwrap();
    let st: std::collections::HashMap<String, Tensor> = opt.state_tensors().unwrap().into_iter().collect();
    opt_res.apply_contract(&parsed).unwrap();
    opt_res.load_state(parsed.step, &st).unwrap();
    assert_eq!(opt_res.param_step(0, 0), Some(0));
    assert_eq!(opt_res.param_step(0, 1), Some(50));

    let bb_ref = param(&[1.0, -2.0]);
    let mut opt_ref = AdamW::new(vec![
        ParamGroup::new("decay", vec![("bb".into(), bb_ref.clone())])
            .lr(0.01)
            .momentum(0.9)
            .weight_decay(0.05),
    ])
    .unwrap();

    for g in [G1, G2] {
        let mut both = grads_for(&bb, &g);
        both.insert(&head, Tensor::from_slice(&g, [2]).unwrap());
        opt.step(&both).unwrap();

        let mut both_res = grads_for(&bb_res, &g);
        both_res.insert(&head_res, Tensor::from_slice(&g, [2]).unwrap());
        opt_res.step(&both_res).unwrap();

        opt_ref.step(&grads_for(&bb_ref, &g)).unwrap();
    }
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>();
    assert_eq!(bits(&bb), bits(&bb_ref), "unfrozen param must match fresh AdamW bias correction");
    assert_eq!(bits(&bb_res), bits(&bb_ref), "resumed AdamW must preserve per-param step count");
    assert_eq!(bits(&head_res), bits(&head), "resumed head must match uninterrupted AdamW");
    assert_eq!(opt.param_step(0, 0), Some(2));
    assert_eq!(opt.param_step(0, 1), Some(52));

    // Uniform AdamW omits `param_steps` from JSON for byte-identical compatibility with legacy checkpoints.
    let uniform_json = serde_json::to_string(&opt_ref.contract()).unwrap();
    assert!(!uniform_json.contains("param_steps"), "{uniform_json}");

    // Reject unknown param name or step > optim.step, and reject `param_steps` on SGD.
    let mut bad_unknown = parsed.clone();
    bad_unknown.param_steps.insert("nope".into(), 0);
    assert!(opt_res.apply_contract(&bad_unknown).is_err());

    let mut bad_gt = parsed.clone();
    bad_gt.param_steps.insert("bb".into(), parsed.step + 1);
    assert!(opt_res.apply_contract(&bad_gt).is_err());

    let mut sgd = Sgd::new(vec![ParamGroup::new("decay", vec![("bb".into(), param(&[1.0, -2.0]))])]).unwrap();
    let mut sgd_c = sgd.contract();
    sgd_c.param_steps.insert("bb".into(), 0);
    assert!(sgd.apply_contract(&sgd_c).is_err());
}
