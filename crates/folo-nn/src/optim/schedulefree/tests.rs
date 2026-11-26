use super::*;
use crate::{Activation, ConvBnAct, Linear, Module, Sequential};
use folo_core::Grads;
use folo_core::rng::Rng;

fn grads_of(p: &Tensor, vals: &[f32]) -> Grads {
    let mut g = Grads::default();
    g.insert(p, Tensor::from_slice(vals, p.shape()).unwrap());
    g
}

#[test]
fn schedulefree_hand_computed_steps_warmup_and_fast_generic_parity() {
    // Hand-computed 2-step trace with warmup_steps = 2, r = 0, weight_lr_power = 2:
    // p0 = z0 = x0 = [2.0], lr = 0.2, beta1 = 0.9, beta2 = 0.99, eps = 1e-8, wd = 0.1
    // Step 1: sched = 1/2 = 0.5 -> gamma1 = 0.1, w1 = 0.01, W1 = 0.01 -> c1 = 1.0
    //   g1 = 1.0 -> v1 = 0.01, bc2 = 0.01 -> sqrt(v1)/sqrt(bc2) = 1.0
    //   u1 = 1.0/(1.0 + 1e-8) + 0.1*2.0 = 1.2
    //   z1 = 2.0 - 0.1*1.2 = 1.88
    //   x1 = 0*2.0 + 1.0*1.88 = 1.88
    //   y1 = 0.1*1.88 + 0.9*1.88 = 1.88
    // Step 2: sched = 1.0 -> gamma2 = 0.2, w2 = 0.04, W2 = 0.05 -> c2 = 0.8
    //   g2 = 0.5 -> v2 = 0.99*0.01 + 0.01*0.25 = 0.0124, bc2 = 1 - 0.99^2 = 0.0199
    //   denom2 = sqrt(0.0124 / 0.0199) + 1e-8 ≈ 0.7893761
    //   u2 = 0.5 / denom2 + 0.1*1.88
    //   z2 = 1.88 - 0.2*u2
    //   x2 = 0.2*1.88 + 0.8*z2
    //   y2 = 0.1*z2 + 0.9*x2
    let p = Tensor::from_slice(&[2.0f32], [1]).unwrap().requires_grad_();
    let cfg = ScheduleFreeConfig { warmup_steps: 2, r: 0.0, weight_lr_power: 2.0 };
    let mut opt =
        ScheduleFreeAdamW::from_params(vec![("w".into(), p.clone())], 0.2, 0.9, 0.99, 1e-8, 0.1, cfg)
            .unwrap();

    opt.step(&grads_of(&p, &[1.0])).unwrap();
    let (z1, x1, v1) = opt.param_state(0, 0).unwrap();
    assert!((p.to_vec::<f32>().unwrap()[0] - 1.88).abs() < 1e-6);
    assert!((z1.to_vec::<f32>().unwrap()[0] - 1.88).abs() < 1e-6);
    assert!((x1.to_vec::<f32>().unwrap()[0] - 1.88).abs() < 1e-6);
    assert!((v1.to_vec::<f32>().unwrap()[0] - 0.01).abs() < 1e-7);
    assert!((opt.weight_sum(0).unwrap() - 0.01).abs() < 1e-12);

    opt.step(&grads_of(&p, &[0.5])).unwrap();
    let denom2 = (0.0124f64 / (1.0 - 0.99f64 * 0.99)).sqrt() + 1e-8;
    let u2 = 0.5 / denom2 + 0.1 * 1.88;
    let z2_want = 1.88 - 0.2 * u2;
    let x2_want = 0.2 * 1.88 + 0.8 * z2_want;
    let y2_want = 0.1 * z2_want + 0.9 * x2_want;
    let (z2, x2, _) = opt.param_state(0, 0).unwrap();
    assert!((z2.to_vec::<f32>().unwrap()[0] as f64 - z2_want).abs() < 1e-5);
    assert!((x2.to_vec::<f32>().unwrap()[0] as f64 - x2_want).abs() < 1e-5);
    assert!((p.to_vec::<f32>().unwrap()[0] as f64 - y2_want).abs() < 1e-5);
    assert!((opt.weight_sum(0).unwrap() - 0.05).abs() < 1e-12);

    // Bitwise parity between contiguous CPU fast path and non-contiguous transposed generic path.
    let init = [1.0f32, -2.0, 0.5, 3.0, -1.5, 2.5];
    let fast = Tensor::from_slice(&init, [2, 3]).unwrap().requires_grad_();
    let base = Tensor::zeros([3, 2]);
    let slow_t = base.t().unwrap();
    {
        let _g = no_grad();
        slow_t.copy_from(&fast.detach()).unwrap();
    }
    let slow = slow_t.requires_grad_();
    assert!(!slow.is_contiguous());
    let mut opt_f =
        ScheduleFreeAdamW::from_params(vec![("w".into(), fast.clone())], 0.05, 0.9, 0.99, 1e-8, 0.1, cfg)
            .unwrap();
    let mut opt_s =
        ScheduleFreeAdamW::from_params(vec![("w".into(), slow.clone())], 0.05, 0.9, 0.99, 1e-8, 0.1, cfg)
            .unwrap();
    for gv in [[0.3f32, -0.4, 0.0, -0.0, 0.2, -0.1], [-0.5, 0.2, 0.0, 0.1, f32::INFINITY, f32::NAN]] {
        opt_f.step(&grads_of(&fast, &gv)).unwrap();
        opt_s.step(&grads_of(&slow, &gv)).unwrap();
        for (a, b) in fast.to_vec::<f32>().unwrap().iter().zip(slow.to_vec::<f32>().unwrap()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs().max(b.abs())), "parity mismatch: {a} vs {b}");
            }
        }
    }
}

#[test]
fn schedulefree_converges_quadratic_rosenbrock_and_module() {
    // 1. Quadratic bowl: min (x - 3)^2 + 4*(y + 2)^2 without any LR decay schedule.
    let p = Tensor::from_slice(&[-4.0f32, 5.0], [2]).unwrap().requires_grad_();
    let mut opt = ScheduleFreeAdamW::from_params(
        vec![("p".into(), p.clone())],
        0.5,
        0.9,
        0.99,
        1e-8,
        0.0,
        ScheduleFreeConfig { warmup_steps: 10, ..Default::default() },
    )
    .unwrap();
    let target = Tensor::from_slice(&[3.0f32, -2.0], [2]).unwrap();
    let scale = Tensor::from_slice(&[1.0f32, 4.0], [2]).unwrap();
    let eval_quad = |w: &Tensor| -> f32 {
        w.sub(&target)
            .unwrap()
            .square()
            .unwrap()
            .mul(&scale)
            .unwrap()
            .sum_all()
            .unwrap()
            .item::<f32>()
            .unwrap()
    };
    let init_loss = eval_quad(&p);
    for _ in 0..350 {
        let loss = p.sub(&target).unwrap().square().unwrap().mul(&scale).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    opt.eval_mode().unwrap();
    let eval_loss = eval_quad(&p);
    assert!(eval_loss < 1e-2 && eval_loss < init_loss * 1e-4, "quadratic: {init_loss} -> {eval_loss}");

    // 2. 2D Rosenbrock banana: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2 starting at (-0.5, 0.5).
    let x = Tensor::from_slice(&[-0.5f32], [1]).unwrap().requires_grad_();
    let y = Tensor::from_slice(&[0.5f32], [1]).unwrap().requires_grad_();
    let mut opt_r = ScheduleFreeAdamW::from_params(
        vec![("x".into(), x.clone()), ("y".into(), y.clone())],
        0.05,
        0.9,
        0.999,
        1e-8,
        0.0,
        ScheduleFreeConfig { warmup_steps: 15, ..Default::default() },
    )
    .unwrap();
    let eval_rosen = |x: &Tensor, y: &Tensor| -> Tensor {
        let t1 = x.rsub_scalar(1.0).unwrap().square().unwrap();
        let t2 = y.sub(&x.square().unwrap()).unwrap().square().unwrap().mul_scalar(100.0).unwrap();
        t1.add(&t2).unwrap().sum_all().unwrap()
    };
    let r0 = eval_rosen(&x, &y).item::<f32>().unwrap();
    for _ in 0..400 {
        let loss = eval_rosen(&x, &y);
        let grads = loss.backward().unwrap();
        opt_r.step(&grads).unwrap();
    }
    opt_r.eval_mode().unwrap();
    let r1 = eval_rosen(&x, &y).item::<f32>().unwrap();
    assert!(r1 < r0 * 0.02, "Rosenbrock eval iterate must converge: {r0} -> {r1}");

    // 3. Small folo-nn module (ConvBnAct + Linear) regression with StandardGroups, NO external LR schedule.
    let mut rng = Rng::seed(77);
    let net = Sequential::new()
        .push_named("stem", ConvBnAct::new(2, 4, 3, 1, 1, Activation::Silu, &mut rng))
        .push_named("proj", Linear::new(4, 2, true, &mut rng));
    let groups = super::super::StandardGroups::from_module(&net).into_groups(0.05, 0.9, 1e-3);
    let mut opt_m =
        ScheduleFreeAdamW::new(groups, ScheduleFreeConfig { warmup_steps: 10, ..Default::default() })
            .unwrap();
    let inp = Tensor::randn([2, 2, 4, 4], 0.0, 1.0, &mut rng);
    let tgt = Tensor::randn([2, 2], 0.0, 1.0, &mut rng);
    let forward_loss = |m: &Sequential| -> Tensor {
        let h = m.get(0).unwrap().forward(&inp).unwrap().mean(3, false).unwrap().mean(2, false).unwrap();
        let pred = m.get(1).unwrap().forward(&h).unwrap();
        pred.sub(&tgt).unwrap().square().unwrap().mean_all().unwrap()
    };
    let l0 = forward_loss(&net).item::<f32>().unwrap();
    for _ in 0..100 {
        let loss = forward_loss(&net);
        let grads = loss.backward().unwrap();
        opt_m.step(&grads).unwrap();
    }
    opt_m.eval_mode().unwrap();
    net.eval();
    let l1 = forward_loss(&net).item::<f32>().unwrap();
    assert!(l1 < l0 * 0.15, "module ScheduleFree training must converge: {l0} -> {l1}");
}

#[test]
fn schedulefree_mode_switch_idempotence_edge_cases_and_resume() {
    // 1. Mode switching is idempotent, drift-free across repeated toggles, works at beta1 = 0, and
    //    calling step() in eval mode errors cleanly without mutating state.
    for beta1 in [0.9f64, 0.0] {
        let p = Tensor::from_slice(&[1.5f32, -2.5], [2]).unwrap().requires_grad_();
        let mut opt = ScheduleFreeAdamW::from_params(
            vec![("w".into(), p.clone())],
            0.1,
            beta1,
            0.99,
            1e-8,
            0.05,
            ScheduleFreeConfig::default(),
        )
        .unwrap();
        opt.step(&grads_of(&p, &[0.5, -0.3])).unwrap();
        opt.step(&grads_of(&p, &[-0.2, 0.7])).unwrap();
        let y_ref = p.to_vec::<f32>().unwrap();
        let x_ref = opt.param_state(0, 0).unwrap().1.to_vec::<f32>().unwrap();

        assert!(opt.is_train_mode());
        opt.train_mode().unwrap(); // idempotent in train mode
        assert_eq!(p.to_vec::<f32>().unwrap(), y_ref);

        opt.eval_mode().unwrap();
        assert!(!opt.is_train_mode());
        assert_eq!(p.to_vec::<f32>().unwrap(), x_ref);
        opt.eval_mode().unwrap(); // idempotent in eval mode
        assert_eq!(p.to_vec::<f32>().unwrap(), x_ref);

        // step() in eval mode must error without advancing step count.
        assert!(opt.step(&grads_of(&p, &[0.1, 0.1])).is_err());
        assert_eq!(opt.step_count(), 2);

        // 100 repeated toggles produce bit-exact y_ref and x_ref (zero rounding drift).
        for _ in 0..100 {
            opt.train_mode().unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), y_ref);
            opt.eval_mode().unwrap();
            assert_eq!(p.to_vec::<f32>().unwrap(), x_ref);
        }
        opt.train_mode().unwrap();
    }

    // 2. Zero grad (wd = 0) leaves param finite; NaN grad poisons z, x, v, and p cleanly.
    let pz = Tensor::from_slice(&[3.0f32, -1.0], [2]).unwrap().requires_grad_();
    let mut opt_z = ScheduleFreeAdamW::from_params(
        vec![("p".into(), pz.clone())],
        0.1,
        0.9,
        0.99,
        1e-8,
        0.0,
        ScheduleFreeConfig::default(),
    )
    .unwrap();
    opt_z.step(&grads_of(&pz, &[0.0, 0.0])).unwrap();
    assert_eq!(pz.to_vec::<f32>().unwrap(), [3.0, -1.0]);
    opt_z.step(&grads_of(&pz, &[f32::NAN, 0.5])).unwrap();
    let pv = pz.to_vec::<f32>().unwrap();
    let (zv, xv, vv) = opt_z.param_state(0, 0).unwrap();
    assert!(pv[0].is_nan() && pv[1].is_finite());
    assert!(zv.to_vec::<f32>().unwrap()[0].is_nan());
    assert!(xv.to_vec::<f32>().unwrap()[0].is_nan());
    assert!(vv.to_vec::<f32>().unwrap()[0].is_nan());

    // 3. Missing grads skip the group cleanly without advancing weight_sum.
    let p_skip = Tensor::from_slice(&[4.0f32], [1]).unwrap().requires_grad_();
    let mut opt_skip = ScheduleFreeAdamW::from_params(
        vec![("a".into(), p_skip.clone())],
        0.1,
        0.9,
        0.99,
        1e-8,
        0.5,
        ScheduleFreeConfig::default(),
    )
    .unwrap();
    opt_skip.step(&Grads::default()).unwrap();
    assert_eq!(p_skip.to_vec::<f32>().unwrap(), [4.0]);
    assert_eq!(opt_skip.group_step(0), Some(0));
    assert_eq!(opt_skip.weight_sum(0), Some(0.0));

    // 4. Hyperparameter & config validation rejections.
    let q = Tensor::zeros([2]).requires_grad_();
    let def = ScheduleFreeConfig::default();
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], -0.1, 0.9, 0.99, 1e-8, 0.0, def)
            .is_err()
    );
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], 0.1, 1.0, 0.99, 1e-8, 0.0, def)
            .is_err()
    );
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], 0.1, 0.9, 1.0, 1e-8, 0.0, def).is_err()
    );
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], 0.1, 0.9, 0.99, 0.0, 0.0, def).is_err()
    );
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], 0.1, 0.9, 0.99, 1e-8, -0.1, def)
            .is_err()
    );
    assert!(
        ScheduleFreeAdamW::new(
            vec![ParamGroup::new("g", vec![("q".into(), q.clone())]).cautious(true)],
            def,
        )
        .is_err()
    );
    assert!(ScheduleFreeConfig { r: -0.1, ..def }.validate().is_err());
    assert!(ScheduleFreeConfig { weight_lr_power: f64::NAN, ..def }.validate().is_err());

    // 5. Checkpoint contract & state round-trip produces bit-exact resume.
    let pa = Tensor::from_slice(&[1.2f32, -0.7, 2.3], [3]).unwrap().requires_grad_();
    let pb = Tensor::from_slice(&[1.2f32, -0.7, 2.3], [3]).unwrap().requires_grad_();
    let cfg = ScheduleFreeConfig { warmup_steps: 5, r: 0.5, weight_lr_power: 2.0 };
    let mut opt_a =
        ScheduleFreeAdamW::from_params(vec![("w".into(), pa.clone())], 0.05, 0.9, 0.99, 1e-8, 0.1, cfg)
            .unwrap();
    opt_a.step(&grads_of(&pa, &[0.3, -0.6, 0.2])).unwrap();
    opt_a.step(&grads_of(&pa, &[-0.4, 0.1, 0.5])).unwrap();
    let json = serde_json::to_string(&opt_a.contract()).unwrap();
    let parsed: ScheduleFreeContract = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, opt_a.contract());
    let saved: HashMap<String, Tensor> = opt_a.state_tensors().unwrap().into_iter().collect();

    {
        let _g = no_grad();
        pb.copy_from(&pa.detach()).unwrap();
    }
    let mut opt_b =
        ScheduleFreeAdamW::from_params(vec![("w".into(), pb.clone())], 0.9, 0.5, 0.5, 1e-5, 0.0, def)
            .unwrap();
    opt_b.apply_contract(&parsed).unwrap();
    opt_b.load_state(parsed.step, &saved).unwrap();

    for g in [[0.2f32, 0.4, -0.3], [-0.5, 0.2, 0.1]] {
        opt_a.step(&grads_of(&pa, &g)).unwrap();
        opt_b.step(&grads_of(&pb, &g)).unwrap();
    }
    assert_eq!(pa.to_vec::<f32>().unwrap(), pb.to_vec::<f32>().unwrap());
    opt_a.eval_mode().unwrap();
    opt_b.eval_mode().unwrap();
    assert_eq!(pa.to_vec::<f32>().unwrap(), pb.to_vec::<f32>().unwrap());

    // 6. Resume after modifying current LR & from an eval-mode checkpoint via `restore_checkpoint`.
    let pc = Tensor::from_slice(&[1.2f32, -0.7, 2.3], [3]).unwrap().requires_grad_();
    let pd = Tensor::from_slice(&[1.2f32, -0.7, 2.3], [3]).unwrap().requires_grad_();
    let mut opt_c =
        ScheduleFreeAdamW::from_params(vec![("w".into(), pc.clone())], 0.05, 0.9, 0.99, 1e-8, 0.1, cfg)
            .unwrap();
    opt_c.step(&grads_of(&pc, &[0.3, -0.6, 0.2])).unwrap();
    opt_c.groups_mut()[0].lr = 0.0125; // runtime LR change
    opt_c.eval_mode().unwrap();
    let eval_contract = opt_c.contract();
    assert!(!eval_contract.train_mode);
    assert_eq!(eval_contract.groups[0].current_lr, Some(0.0125));
    let eval_saved: HashMap<String, Tensor> = opt_c.state_tensors().unwrap().into_iter().collect();
    {
        let _g = no_grad();
        pd.copy_from(&pc.detach()).unwrap();
    }
    let mut opt_d =
        ScheduleFreeAdamW::from_params(vec![("w".into(), pd.clone())], 0.9, 0.5, 0.5, 1e-5, 0.0, def)
            .unwrap();
    // Direct `load_state(step > 0)` without `apply_contract` must error.
    assert!(opt_d.load_state(eval_contract.step, &eval_saved).is_err());
    // Atomic `restore_checkpoint` restores current_lr, eval mode, group counters, and state together.
    opt_d.restore_checkpoint(&eval_contract, &eval_saved).unwrap();
    assert!(!opt_d.is_train_mode());
    assert_eq!(opt_d.groups()[0].lr, 0.0125);
    opt_c.train_mode().unwrap();
    opt_d.train_mode().unwrap();
    opt_c.step(&grads_of(&pc, &[-0.2, 0.5, 0.1])).unwrap();
    opt_d.step(&grads_of(&pd, &[-0.2, 0.5, 0.1])).unwrap();
    assert_eq!(pc.to_vec::<f32>().unwrap(), pd.to_vec::<f32>().unwrap());

    // 7. Duplicate parameter identity / overlapping storage / broadcast view / f32-underflowing eps rejections.
    let shared = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap().requires_grad_();
    assert!(
        ScheduleFreeAdamW::from_params(
            vec![("a".into(), shared.clone()), ("b".into(), shared.clone())],
            0.1,
            0.9,
            0.99,
            1e-8,
            0.0,
            def
        )
        .is_err()
    );
    let bcast = Tensor::from_slice(&[1.0f32], [1]).unwrap().broadcast_to([4]).unwrap().requires_grad_();
    assert!(
        ScheduleFreeAdamW::from_params(vec![("b".into(), bcast)], 0.1, 0.9, 0.99, 1e-8, 0.0, def).is_err()
    );
    // `eps = 1e-45` narrows to `0.0f32` and must be rejected.
    assert!(
        ScheduleFreeAdamW::from_params(vec![("q".into(), q.clone())], 0.1, 0.9, 0.99, 1e-45, 0.0, def)
            .is_err()
    );

    // 8. Representable large gradient `g = 1e20` at `beta2 = 0.999` keeps `v`, `z`, `x`, `p` finite
    //    (both contiguous CPU fast path and non-contiguous transposed generic path).
    let p_big = Tensor::from_slice(&[1.0f32, -1.0], [2]).unwrap().requires_grad_();
    let mut opt_big =
        ScheduleFreeAdamW::from_params(vec![("w".into(), p_big.clone())], 0.1, 0.9, 0.999, 1e-8, 0.0, def)
            .unwrap();
    opt_big.step(&grads_of(&p_big, &[1e20, -1e20])).unwrap();
    let (zb, xb, vb) = opt_big.param_state(0, 0).unwrap();
    assert!(p_big.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(zb.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(xb.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(vb.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite() && *v > 1e35));

    let base_big = Tensor::zeros([2, 2]);
    let view_big = base_big.t().unwrap().requires_grad_();
    assert!(!view_big.is_contiguous());
    let mut opt_big_gen =
        ScheduleFreeAdamW::from_params(vec![("w".into(), view_big.clone())], 0.1, 0.9, 0.999, 1e-8, 0.0, def)
            .unwrap();
    opt_big_gen.step(&grads_of(&view_big, &[1e20, -1e20, 1e20, -1e20])).unwrap();
    assert!(view_big.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(opt_big_gen.param_state(0, 0).unwrap().2.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));

    // 9. Running maximum LR (`lr_max`) averaging rule (Defazio et al. 2024): lowering LR from 0.1 -> 0.01
    //    with `r = 0, weight_lr_power = 2.0` keeps `lr_max = 0.1`, so `w_2 = w_1 = 0.01` and `c_2 = 0.5`.
    let p_lr = Tensor::from_slice(&[2.0f32], [1]).unwrap().requires_grad_();
    let mut opt_lr =
        ScheduleFreeAdamW::from_params(vec![("w".into(), p_lr.clone())], 0.1, 0.9, 0.99, 1e-8, 0.0, def)
            .unwrap();
    opt_lr.step(&grads_of(&p_lr, &[1.0])).unwrap();
    assert!((opt_lr.group_lr_max(0).unwrap() - 0.1).abs() < 1e-15);
    assert!((opt_lr.weight_sum(0).unwrap() - 0.01).abs() < 1e-15);
    opt_lr.groups_mut()[0].lr = 0.01;
    let x_before = opt_lr.param_state(0, 0).unwrap().1.to_vec::<f32>().unwrap()[0];
    opt_lr.step(&grads_of(&p_lr, &[1.0])).unwrap();
    assert!((opt_lr.group_lr_max(0).unwrap() - 0.1).abs() < 1e-15);
    assert!((opt_lr.weight_sum(0).unwrap() - 0.02).abs() < 1e-15);
    let z_after = opt_lr.param_state(0, 0).unwrap().0.to_vec::<f32>().unwrap()[0];
    let x_after = opt_lr.param_state(0, 0).unwrap().1.to_vec::<f32>().unwrap()[0];
    // c_2 = 0.01 / 0.02 = 0.5 -> x_2 = 0.5 * x_1 + 0.5 * z_2.
    assert!((x_after - 0.5 * (x_before + z_after)).abs() < 1e-6);
    assert!((opt_lr.contract().groups[0].lr_max - 0.1).abs() < 1e-15);

    // 10. Swapping or replacing same-shaped parameters via `groups_mut()` is rejected before touching state.
    let p_s0 = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap().requires_grad_();
    let p_s1 = Tensor::from_slice(&[3.0f32, 4.0], [2]).unwrap().requires_grad_();
    let mut opt_swap = ScheduleFreeAdamW::from_params(
        vec![("a".into(), p_s0.clone()), ("b".into(), p_s1.clone())],
        0.1,
        0.9,
        0.99,
        1e-8,
        0.0,
        def,
    )
    .unwrap();
    opt_swap.groups_mut()[0].params.swap(0, 1);
    assert!(opt_swap.step(&grads_of(&p_s0, &[1.0, 1.0])).is_err());
    assert!(opt_swap.eval_mode().is_err());
    assert!(opt_swap.state_tensors().is_err());
    assert_eq!(p_s0.to_vec::<f32>().unwrap(), [1.0, 2.0]);
    assert_eq!(p_s1.to_vec::<f32>().unwrap(), [3.0, 4.0]);

    // 11. Skipped-step checkpoint (`step == 1, group_step == 0`) round-trips cleanly via `apply_contract` +
    //     `load_state`, while calling `load_state(step > 0)` on an already-stepped optimizer without
    //     `apply_contract` is rejected, as is a contract with `group_step == 0 && weight_sum > 0`.
    let p_sk1 = Tensor::from_slice(&[5.0f32], [1]).unwrap().requires_grad_();
    let p_sk2 = Tensor::from_slice(&[5.0f32], [1]).unwrap().requires_grad_();
    let mut opt_sk1 =
        ScheduleFreeAdamW::from_params(vec![("w".into(), p_sk1.clone())], 0.1, 0.9, 0.99, 1e-8, 0.0, def)
            .unwrap();
    opt_sk1.step(&Grads::default()).unwrap(); // skipped step: step=1, group_step=0
    let sk_contract = opt_sk1.contract();
    let sk_state: HashMap<String, Tensor> = opt_sk1.state_tensors().unwrap().into_iter().collect();
    let mut opt_sk2 =
        ScheduleFreeAdamW::from_params(vec![("w".into(), p_sk2.clone())], 0.1, 0.9, 0.99, 1e-8, 0.0, def)
            .unwrap();
    opt_sk2.apply_contract(&sk_contract).unwrap();
    opt_sk2.load_state(sk_contract.step, &sk_state).unwrap();
    assert_eq!(opt_sk2.step_count(), 1);
    assert_eq!(opt_sk2.group_step(0), Some(0));

    // Already-stepped optimizer must reject `load_state(step > 0)` without a preceding `apply_contract`.
    opt_sk2.step(&grads_of(&p_sk2, &[1.0])).unwrap();
    assert!(opt_sk2.load_state(parsed.step, &saved).is_err());

    // Invalid contract with `group_step == 0 && weight_sum > 0` or `lr_max > 0` is rejected.
    let mut bad_contract = sk_contract.clone();
    bad_contract.groups[0].weight_sum = 1.0;
    assert!(opt_sk2.apply_contract(&bad_contract).is_err());
    let mut bad_lr_max = sk_contract;
    bad_lr_max.groups[0].lr_max = 0.1;
    assert!(opt_sk2.apply_contract(&bad_lr_max).is_err());
}
