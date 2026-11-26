//! Unit tests for the [`Adan`] optimizer:
//! hand-computed multi-step `f64` reference parity (both decoupled and proximal weight decay),
//! convergence on quadratic & Rosenbrock bowls, skipped-parameter per-param step tracking,
//! checkpoint contract / state bit-exact resume, config & gradient validation, and non-finite poison.

use std::collections::HashMap;

use folo_core::{DType, Grads, Tensor};

use super::*;
use crate::optim::GradSource;
use crate::{SgdrConfig, SgdrSchedule};

struct MapGrads(HashMap<folo_core::TensorId, Tensor>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.get(&param.id())
    }
}

/// Hand-computed `f64` reference check over 5 steps for both decoupled and proximal weight decay.
#[test]
fn adan_matches_f64_reference_decoupled_and_proximal() {
    for decoupled_wd in [true, false] {
        let init = [1.5f32, -2.0, 0.25, 4.0];
        let grads_seq: [[f32; 4]; 5] = [
            [0.4, -0.8, 0.1, 1.2],
            [0.3, -0.5, -0.2, 0.9],
            [-0.1, 0.2, 0.5, 0.4],
            [0.0, -0.3, 0.2, -0.6],
            [0.25, 0.15, -0.1, 0.05],
        ];
        let cfg =
            AdanConfig { lr: 0.05, betas: [0.98, 0.92, 0.99], eps: 1e-8, weight_decay: 0.02, decoupled_wd };
        let p = Tensor::from_slice(&init, [4]).unwrap();
        let mut opt = Adan::from_params(vec![("w".into(), p.clone())], cfg).unwrap();

        // Independent f64 reference.
        let mut ref_p: Vec<f64> = init.iter().map(|&v| v as f64).collect();
        let mut ref_m = [0.0f64; 4];
        let mut ref_v = [0.0f64; 4];
        let mut ref_n = [0.0f64; 4];
        let mut ref_pg = [0.0f64; 4];
        let [b1, b2, b3] = cfg.betas;

        for (step_idx, g_arr) in grads_seq.iter().enumerate() {
            let k = (step_idx + 1) as i32;
            let mut map = HashMap::new();
            map.insert(p.id(), Tensor::from_slice(g_arr, [4]).unwrap());
            opt.step(&MapGrads(map)).unwrap();

            let bc1 = 1.0 - b1.powi(k);
            let bc2 = 1.0 - b2.powi(k);
            let bc3 = 1.0 - b3.powi(k);
            for i in 0..4 {
                let gi = g_arr[i] as f64;
                let diff = if step_idx == 0 { 0.0 } else { gi - ref_pg[i] };
                ref_m[i] = b1 * ref_m[i] + (1.0 - b1) * gi;
                ref_v[i] = b2 * ref_v[i] + (1.0 - b2) * diff;
                let u = gi + b2 * diff;
                ref_n[i] = b3 * ref_n[i] + (1.0 - b3) * u * u;
                let step_dir = (ref_m[i] / bc1 + b2 * (ref_v[i] / bc2)) / ((ref_n[i] / bc3).sqrt() + cfg.eps);
                if decoupled_wd {
                    ref_p[i] = ref_p[i] * (1.0 - cfg.lr * cfg.weight_decay) - cfg.lr * step_dir;
                } else {
                    ref_p[i] = (ref_p[i] - cfg.lr * step_dir) / (1.0 + cfg.lr * cfg.weight_decay);
                }
                ref_pg[i] = gi;
            }

            let got = p.to_vec::<f32>().unwrap();
            for i in 0..4 {
                assert!(
                    (got[i] as f64 - ref_p[i]).abs() < 1e-5,
                    "decoupled={decoupled_wd} step={k} i={i}: got={} want={}",
                    got[i],
                    ref_p[i]
                );
            }
            assert_eq!(opt.step_count(), k as u64);
            assert_eq!(opt.param_step(0, 0), Some(k as u64));
        }
    }
}

/// Convergence test on an ill-conditioned quadratic bowl paired with SGDR warm restarts.
#[test]
fn adan_converges_on_quadratic_and_rosenbrock() {
    // 1. Anisotropic quadratic: f(x) = sum_i a_i * (x_i - target_i)^2.
    let target = Tensor::from_slice(&[2.0f32, -1.5, 0.75, -3.0], [4]).unwrap();
    let coeffs = Tensor::from_slice(&[1.0f32, 10.0, 25.0, 5.0], [4]).unwrap();
    let x = Tensor::from_slice(&[-3.0f32, 4.0, -2.5, 5.0], [4]).unwrap().requires_grad_();
    let cfg = AdanConfig { lr: 0.15, weight_decay: 0.0, ..AdanConfig::default() };
    let mut opt = Adan::from_params(vec![("x".into(), x.clone())], cfg).unwrap();
    let mut sched = SgdrSchedule::new(SgdrConfig {
        base_lr: 0.15,
        eta_min: 1e-4,
        t_0: 80,
        t_mult: 2,
        warmup_steps: 5,
        warmup_start_lr: 0.01,
    })
    .unwrap();

    let loss_fn =
        |v: &Tensor| v.sub(&target).unwrap().square().unwrap().mul(&coeffs).unwrap().sum_all().unwrap();
    let initial_loss = loss_fn(&x).item::<f32>().unwrap();
    // 5 warmup + 80 (cycle 0) + 160 (cycle 1) + 320 (cycle 2) = 565 steps, ending at the trough of cycle 2.
    for _ in 0..565 {
        sched.apply_to_groups(opt.groups_mut());
        let loss = loss_fn(&x);
        let grads: Grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        sched.advance();
    }
    let final_loss = loss_fn(&x).item::<f32>().unwrap();
    assert!(
        final_loss < initial_loss * 1e-4,
        "Adan failed to converge on anisotropic quadratic: {initial_loss} -> {final_loss}"
    );

    // 2. Classic 2D Rosenbrock banana: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2 starting at (-1.0, 1.0).
    let rx = Tensor::from_slice(&[-1.0f32], [1]).unwrap().requires_grad_();
    let ry = Tensor::from_slice(&[1.0f32], [1]).unwrap().requires_grad_();
    let mut ropt = Adan::from_params(
        vec![("x".into(), rx.clone()), ("y".into(), ry.clone())],
        AdanConfig { lr: 0.05, weight_decay: 0.0, ..AdanConfig::default() },
    )
    .unwrap();
    let rosen = |a: &Tensor, b: &Tensor| {
        let t1 = a.rsub_scalar(1.0).unwrap().square().unwrap();
        let t2 = b.sub(&a.square().unwrap()).unwrap().square().unwrap().mul_scalar(100.0).unwrap();
        t1.add(&t2).unwrap().sum_all().unwrap()
    };
    let r0 = rosen(&rx, &ry).item::<f32>().unwrap();
    for _ in 0..500 {
        let l = rosen(&rx, &ry);
        let g = l.backward().unwrap();
        ropt.step(&g).unwrap();
    }
    let r1 = rosen(&rx, &ry).item::<f32>().unwrap();
    assert!(r1 < 1e-2 && r1 < r0 * 1e-2, "Adan failed to descend Rosenbrock: {r0} -> {r1}");
}

/// Skipped parameters receive zero decay/state updates, keep `param_step == 0` until their first gradient,
/// and contract/state checkpoint resume is bit-exact.
#[test]
fn adan_skipped_param_and_bit_exact_resume() {
    let p1 = Tensor::from_slice(&[1.0f32, -2.0], [2]).unwrap();
    let p2 = Tensor::from_slice(&[3.0f32, 4.0], [2]).unwrap();
    let cfg = AdanConfig { lr: 0.02, weight_decay: 0.05, ..AdanConfig::default() };
    let mut opt = Adan::from_params(vec![("a".into(), p1.clone()), ("b".into(), p2.clone())], cfg).unwrap();

    // Step 1: only `a` gets a gradient; `b` is skipped completely.
    let mut g1 = HashMap::new();
    g1.insert(p1.id(), Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap());
    opt.step(&MapGrads(g1)).unwrap();
    assert_eq!(p2.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(opt.param_step(0, 0), Some(1));
    assert_eq!(opt.param_step(0, 1), Some(0));

    // Snapshot contract & state tensors after step 1, then run step 2 on both original and resumed optimizers.
    let contract = opt.contract();
    let json = serde_json::to_string(&contract).unwrap();
    let parsed: AdanContract = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, contract);
    let state_map: HashMap<String, Tensor> = opt.state_tensors().unwrap().into_iter().collect();

    let p1_r = p1.detach().to_vec::<f32>().unwrap();
    let p2_r = p2.detach().to_vec::<f32>().unwrap();
    let p1_clone = Tensor::from_vec(p1_r, [2]).unwrap();
    let p2_clone = Tensor::from_vec(p2_r, [2]).unwrap();
    let mut resumed = Adan::from_params(
        vec![("a".into(), p1_clone.clone()), ("b".into(), p2_clone.clone())],
        AdanConfig::default(),
    )
    .unwrap();
    resumed.apply_contract(&parsed).unwrap();
    resumed.load_state(parsed.step, &state_map).unwrap();

    // Step 2: both `a` and `b` receive gradients (`b` gets its first active step with diff = 0).
    let mut g2_orig = HashMap::new();
    g2_orig.insert(p1.id(), Tensor::from_slice(&[-0.2f32, 0.7], [2]).unwrap());
    g2_orig.insert(p2.id(), Tensor::from_slice(&[0.3f32, -0.9], [2]).unwrap());
    opt.step(&MapGrads(g2_orig)).unwrap();

    let mut g2_res = HashMap::new();
    g2_res.insert(p1_clone.id(), Tensor::from_slice(&[-0.2f32, 0.7], [2]).unwrap());
    g2_res.insert(p2_clone.id(), Tensor::from_slice(&[0.3f32, -0.9], [2]).unwrap());
    resumed.step(&MapGrads(g2_res)).unwrap();

    assert_eq!(
        p1.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        p1_clone.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        p2.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        p2_clone.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(resumed.param_step(0, 0), Some(2));
    assert_eq!(resumed.param_step(0, 1), Some(1));
}

/// Edge cases: `betas = [0, 0, 0]`, `lr = 0`, invalid configs, atomic failure on bad gradient,
/// negative `exp_avg_sq` rejection on load, and non-finite gradient poisoning.
#[test]
fn adan_edge_cases_validation_and_non_finite_poison() {
    // 1. Invalid configs rejected cleanly.
    let p = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap();
    for bad in [
        AdanConfig { lr: -0.01, ..AdanConfig::default() },
        AdanConfig { lr: f64::NAN, ..AdanConfig::default() },
        AdanConfig { betas: [-0.1, 0.92, 0.99], ..AdanConfig::default() },
        AdanConfig { betas: [1.0, 0.92, 0.99], ..AdanConfig::default() },
        AdanConfig { betas: [0.98, 1.0, 0.99], ..AdanConfig::default() },
        AdanConfig { betas: [0.98, 0.92, 1.0], ..AdanConfig::default() },
        AdanConfig { betas: [0.98, f64::NAN, 0.99], ..AdanConfig::default() },
        AdanConfig { eps: 0.0, ..AdanConfig::default() },
        AdanConfig { eps: 1e-60, ..AdanConfig::default() }, // underflows to 0.0 in f32
        AdanConfig { weight_decay: -0.01, ..AdanConfig::default() },
        AdanConfig { weight_decay: f64::INFINITY, ..AdanConfig::default() },
    ] {
        assert!(bad.validate().is_err(), "{bad:?} should be rejected");
        assert!(Adan::from_params(vec![("w".into(), p.clone())], bad).is_err());
    }

    // 2. betas = [0.0, 0.0, 0.0] stays finite (no 0/0 bias correction).
    let p0 = Tensor::from_slice(&[1.0f32, -1.0], [2]).unwrap();
    let mut opt0 = Adan::from_params(
        vec![("w".into(), p0.clone())],
        AdanConfig { lr: 0.1, betas: [0.0, 0.0, 0.0], eps: 1e-8, weight_decay: 0.0, decoupled_wd: true },
    )
    .unwrap();
    let mut m0 = HashMap::new();
    m0.insert(p0.id(), Tensor::from_slice(&[2.0f32, -2.0], [2]).unwrap());
    opt0.step(&MapGrads(m0)).unwrap();
    let v0 = p0.to_vec::<f32>().unwrap();
    assert!((v0[0] - 0.9).abs() < 1e-5 && (v0[1] - (-0.9)).abs() < 1e-5, "got {v0:?}");

    // 3. Atomic validation: a mismatched gradient shape on param 2 leaves param 1 untouched.
    let pa = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap();
    let pb = Tensor::from_slice(&[3.0f32, 4.0], [2]).unwrap();
    let mut opt_atomic =
        Adan::from_params(vec![("a".into(), pa.clone()), ("b".into(), pb.clone())], AdanConfig::default())
            .unwrap();
    let mut bad_g = HashMap::new();
    bad_g.insert(pa.id(), Tensor::from_slice(&[0.5f32, 0.5], [2]).unwrap());
    bad_g.insert(pb.id(), Tensor::zeros([3])); // wrong shape
    assert!(opt_atomic.step(&MapGrads(bad_g)).is_err());
    assert_eq!(pa.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(opt_atomic.step_count(), 0);

    // 4. Reject non-float gradient, f32-rounding beta, and negative / -Inf exp_avg_sq on load_state.
    let mut int_g = HashMap::new();
    int_g.insert(pa.id(), Tensor::zeros_with([2], DType::I64, pa.device()).unwrap());
    assert!(opt_atomic.step(&MapGrads(int_g)).is_err());

    assert!(AdanConfig { betas: [1.0 - 1e-10, 0.92, 0.99], ..AdanConfig::default() }.validate().is_err());

    let mut bad_state: HashMap<String, Tensor> = opt_atomic.state_tensors().unwrap().into_iter().collect();
    bad_state.insert("a.exp_avg_sq".into(), Tensor::from_slice(&[-1.0f32, 0.0], [2]).unwrap());
    assert!(opt_atomic.load_state(1, &bad_state).is_err());
    bad_state.insert("a.exp_avg_sq".into(), Tensor::from_slice(&[f32::NEG_INFINITY, 0.0], [2]).unwrap());
    assert!(opt_atomic.load_state(1, &bad_state).is_err());

    // 5. Non-finite gradient propagates NaN into updated parameter.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let pn = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap();
        let mut opt_n = Adan::from_params(vec![("w".into(), pn.clone())], AdanConfig::default()).unwrap();
        let mut gn = HashMap::new();
        gn.insert(pn.id(), Tensor::from_slice(&[bad, 0.1], [2]).unwrap());
        opt_n.step(&MapGrads(gn)).unwrap();
        let out = pn.to_vec::<f32>().unwrap();
        assert!(out[0].is_nan() && out[1].is_finite(), "bad grad {bad}: {out:?}");
    }
}

/// Regression tests for Round 1 review findings:
/// reordered parameter resume, broadcast-view atomic preflight, aliased `g == p` `pre_grad` snapshot,
/// composite-path large-gradient overflow safety + CPU parity + F16 gradient, varying-`beta1` exact
/// bias correction, and `load_state(0, empty)` / direct `load_state(step > 0)` handling.
#[test]
fn adan_reordered_resume_aliasing_composite_overflow_and_varying_beta1() {
    // 1. Reordered parameter resume maps param_step and beta1_prod by parameter name.
    let pa = Tensor::from_slice(&[1.0f32, -2.0], [2]).unwrap();
    let pb = Tensor::from_slice(&[3.0f32, 4.0], [2]).unwrap();
    let cfg = AdanConfig { lr: 0.05, weight_decay: 0.01, ..AdanConfig::default() };
    let mut opt = Adan::from_params(vec![("a".into(), pa.clone()), ("b".into(), pb.clone())], cfg).unwrap();
    let mut g1 = HashMap::new();
    g1.insert(pa.id(), Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap());
    opt.step(&MapGrads(g1)).unwrap(); // a has step 1, b has step 0

    let contract = opt.contract();
    let state_map: HashMap<String, Tensor> = opt.state_tensors().unwrap().into_iter().collect();
    let pa2 = Tensor::from_vec(pa.to_vec::<f32>().unwrap(), [2]).unwrap();
    let pb2 = Tensor::from_vec(pb.to_vec::<f32>().unwrap(), [2]).unwrap();
    // Construct resumed optimizer in REVERSED order [b, a].
    let mut resumed =
        Adan::from_params(vec![("b".into(), pb2.clone()), ("a".into(), pa2.clone())], AdanConfig::default())
            .unwrap();
    resumed.apply_contract(&contract).unwrap();
    resumed.load_state(contract.step, &state_map).unwrap();
    assert_eq!(resumed.param_step(0, 0), Some(0)); // "b"
    assert_eq!(resumed.param_step(0, 1), Some(1)); // "a"

    let mut g2_orig = HashMap::new();
    g2_orig.insert(pa.id(), Tensor::from_slice(&[-0.3f32, 0.8], [2]).unwrap());
    g2_orig.insert(pb.id(), Tensor::from_slice(&[0.4f32, -0.2], [2]).unwrap());
    opt.step(&MapGrads(g2_orig)).unwrap();

    let mut g2_res = HashMap::new();
    g2_res.insert(pa2.id(), Tensor::from_slice(&[-0.3f32, 0.8], [2]).unwrap());
    g2_res.insert(pb2.id(), Tensor::from_slice(&[0.4f32, -0.2], [2]).unwrap());
    resumed.step(&MapGrads(g2_res)).unwrap();
    assert_eq!(pa.to_vec::<f32>().unwrap(), pa2.to_vec::<f32>().unwrap());
    assert_eq!(pb.to_vec::<f32>().unwrap(), pb2.to_vec::<f32>().unwrap());

    // Resetting an already-stepped optimizer via load_state(0, empty) succeeds and resets param_steps.
    resumed.load_state(0, &HashMap::new()).unwrap();
    assert_eq!(resumed.step_count(), 0);
    assert_eq!(resumed.param_step(0, 0), Some(0));
    assert_eq!(resumed.param_step(0, 1), Some(0));

    // Failed load_state after apply_contract leaves existing optimizer hyperparameters and state untouched.
    let mut partial_res =
        Adan::from_params(vec![("b".into(), pb2.clone()), ("a".into(), pa2.clone())], AdanConfig::default())
            .unwrap();
    let orig_lr = partial_res.groups()[0].base_lr;
    partial_res.apply_contract(&contract).unwrap();
    assert!(partial_res.load_state(contract.step, &HashMap::new()).is_err());
    assert_eq!(partial_res.groups()[0].base_lr, orig_lr);
    assert_eq!(partial_res.step_count(), 0);

    // Contract rejecting param_step > step.
    let mut bad_contract = contract.clone();
    bad_contract.groups[0].param_steps[0] = bad_contract.step + 1;
    assert!(resumed.apply_contract(&bad_contract).is_err());

    // 2. Broadcast/overlapping view parameter rejected at construction AND preflight (atomic step).
    let bcast = Tensor::from_slice(&[1.0f32], [1]).unwrap().broadcast_to([2]).unwrap();
    assert!(Adan::from_params(vec![("bc".into(), bcast.clone())], cfg).is_err());

    let p_ok = Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap();
    let p_mut = Tensor::from_slice(&[3.0f32, 4.0], [2]).unwrap();
    let mut opt_bcast =
        Adan::from_params(vec![("ok".into(), p_ok.clone()), ("m".into(), p_mut.clone())], cfg).unwrap();
    // Replace second param with a broadcast view via groups_mut to test step() preflight atomicity.
    opt_bcast.groups_mut()[0].params[1].1 = bcast.clone();
    let mut gb = HashMap::new();
    gb.insert(p_ok.id(), Tensor::from_slice(&[1.0f32, 1.0], [2]).unwrap());
    gb.insert(bcast.id(), Tensor::from_slice(&[1.0f32, 1.0], [2]).unwrap());
    assert!(opt_bcast.step(&MapGrads(gb)).is_err());
    assert_eq!(p_ok.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(opt_bcast.step_count(), 0);

    // 3. Aliased gradient g == p saves pre-update gradient in pre_grad (not post-update p_new).
    let p_alias = Tensor::from_slice(&[2.0f32], [1]).unwrap();
    let mut opt_alias = Adan::from_params(
        vec![("w".into(), p_alias.clone())],
        AdanConfig { lr: 0.1, weight_decay: 0.0, ..AdanConfig::default() },
    )
    .unwrap();
    let mut ga = HashMap::new();
    ga.insert(p_alias.id(), p_alias.clone());
    opt_alias.step(&MapGrads(ga)).unwrap();
    assert!((p_alias.to_vec::<f32>().unwrap()[0] - 1.9).abs() < 1e-5);
    assert_eq!(opt_alias.pre_grad(0, 0).unwrap().to_vec::<f32>().unwrap(), vec![2.0]);

    // 4. Composite path (transposed non-contiguous gradient) matches CPU fast path at 2e19 without Inf.
    let p_cpu = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let p_comp = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let big_cfg = AdanConfig { lr: 0.1, weight_decay: 0.0, ..AdanConfig::default() };
    let mut opt_cpu = Adan::from_params(vec![("w".into(), p_cpu.clone())], big_cfg).unwrap();
    let mut opt_comp = Adan::from_params(vec![("w".into(), p_comp.clone())], big_cfg).unwrap();
    let g_contig = Tensor::from_slice(&[2e19f32, -2e19, 2e19, -2e19], [2, 2]).unwrap();
    let g_trans = Tensor::from_slice(&[2e19f32, 2e19, -2e19, -2e19], [2, 2]).unwrap().t().unwrap();
    assert!(!g_trans.is_contiguous());
    let mut mc = HashMap::new();
    mc.insert(p_cpu.id(), g_contig);
    let mut mt = HashMap::new();
    mt.insert(p_comp.id(), g_trans);
    opt_cpu.step(&MapGrads(mc)).unwrap();
    opt_comp.step(&MapGrads(mt)).unwrap();
    let n_comp = opt_comp.exp_avg_sq(0, 0).unwrap().to_vec::<f32>().unwrap();
    assert!(n_comp.iter().all(|v| v.is_finite()), "composite exp_avg_sq overflowed: {n_comp:?}");
    for (a, b) in p_cpu.to_vec::<f32>().unwrap().iter().zip(p_comp.to_vec::<f32>().unwrap()) {
        assert!((a - b).abs() < 1e-5, "CPU vs composite mismatch at 2e19: {a} vs {b}");
    }

    // F16 gradient through composite path matches F32 CPU path.
    let p_f16 = Tensor::from_slice(&[1.0f32, -1.0], [2]).unwrap();
    let mut opt_f16 = Adan::from_params(vec![("w".into(), p_f16.clone())], big_cfg).unwrap();
    let mut mf16 = HashMap::new();
    mf16.insert(p_f16.id(), Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap().to_dtype(DType::F16).unwrap());
    opt_f16.step(&MapGrads(mf16)).unwrap();
    let vf16 = p_f16.to_vec::<f32>().unwrap();
    assert!((vf16[0] - 0.9).abs() < 1e-4 && (vf16[1] - (-0.9)).abs() < 1e-4, "f16 grad: {vf16:?}");

    // 5. Varying beta1 schedule (0.5 on step 1, 0.9 on step 2, constant unit gradients):
    // Exact bc1 = 1 - 0.5 * 0.9 = 0.55 => m_hat = 1.0 => parameter reaches -0.2 (not -0.38947).
    let pv = Tensor::from_slice(&[0.0f32], [1]).unwrap();
    let mut opt_v = Adan::from_params(
        vec![("w".into(), pv.clone())],
        AdanConfig { lr: 0.1, betas: [0.5, 0.92, 0.99], eps: 1e-8, weight_decay: 0.0, decoupled_wd: true },
    )
    .unwrap();
    let mut gv = HashMap::new();
    gv.insert(pv.id(), Tensor::from_slice(&[1.0f32], [1]).unwrap());
    opt_v.step(&MapGrads(gv)).unwrap();
    opt_v.groups_mut()[0].momentum = 0.9;
    let mut gv2 = HashMap::new();
    gv2.insert(pv.id(), Tensor::from_slice(&[1.0f32], [1]).unwrap());
    opt_v.step(&MapGrads(gv2)).unwrap();
    let got_v = pv.to_vec::<f32>().unwrap()[0];
    assert!((got_v - (-0.2)).abs() < 1e-5, "varying beta1 expected -0.2, got {got_v}");
}
