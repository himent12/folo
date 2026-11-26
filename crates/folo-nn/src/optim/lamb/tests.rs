//! Deterministic unit tests for [`Lamb`].

use std::collections::HashMap;

use folo_core::Tensor;
use folo_core::rng::Rng;

use super::*;
use crate::optim::GradSource;

struct MapGrads(Vec<(Tensor, Tensor)>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.iter().find(|(k, _)| k.id() == param.id()).map(|(_, g)| g)
    }
}

fn l2_f64(xs: &[f64]) -> f64 {
    xs.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn lamb_matches_hand_computed_f64_reference_two_steps() {
    let w_init = [1.5f32, -0.5, 2.0, 0.25];
    let g1 = [0.2f32, -0.4, 0.1, 0.8];
    let g2 = [-0.3f32, 0.6, -0.2, 0.5];
    let lr = 0.05;
    let b1 = 0.9;
    let b2 = 0.999;
    let eps = 1e-6;
    let wd = 0.01;
    let clip = 10.0;

    let p = Tensor::from_slice(&w_init, [4]).unwrap().requires_grad_();
    let cfg = LambConfig {
        lr,
        betas: [b1, b2],
        eps,
        weight_decay: wd,
        trust_clip: Some(clip),
        bias_correction: true,
    };
    let mut opt = Lamb::from_params(vec![("w".into(), p.clone())], cfg).unwrap();

    // Independent f64 reference simulation.
    let mut w_ref: [f64; 4] = w_init.map(|v| v as f64);
    let mut m_ref = [0.0f64; 4];
    let mut v_ref = [0.0f64; 4];

    for (t, g_arr) in [g1, g2].into_iter().enumerate() {
        let step = (t + 1) as i32;
        let grads = MapGrads(vec![(p.clone(), Tensor::from_slice(&g_arr, [4]).unwrap())]);
        opt.step(&grads).unwrap();

        let bc1 = 1.0 - b1.powi(step);
        let bc2 = 1.0 - b2.powi(step);
        let mut u_ref = [0.0f64; 4];
        for i in 0..4 {
            let gi = g_arr[i] as f64;
            m_ref[i] = b1 * m_ref[i] + (1.0 - b1) * gi;
            v_ref[i] = b2 * v_ref[i] + (1.0 - b2) * gi * gi;
            let m_hat = m_ref[i] / bc1;
            let v_hat = v_ref[i] / bc2;
            u_ref[i] = m_hat / (v_hat.sqrt() + eps) + wd * w_ref[i];
        }
        let r_ref = (l2_f64(&w_ref) / l2_f64(&u_ref)).min(clip);
        for i in 0..4 {
            w_ref[i] -= lr * r_ref * u_ref[i];
            // Quantize to f32 between steps to match stored tensor precision.
            w_ref[i] = (w_ref[i] as f32) as f64;
            m_ref[i] = (m_ref[i] as f32) as f64;
            v_ref[i] = (v_ref[i] as f32) as f64;
        }
        assert!((opt.last_trust_ratio(0, 0).unwrap() - r_ref).abs() < 1e-6);
        let actual = p.to_vec::<f32>().unwrap();
        for i in 0..4 {
            assert!((actual[i] as f64 - w_ref[i]).abs() < 1e-6, "step {step} idx {i}");
        }
    }
    assert_eq!(opt.step_count(), 2);
    assert_eq!(opt.param_step(0, 0), Some(2));
}

#[test]
fn lamb_zero_norm_fallback_trust_clip_and_no_bias_correction() {
    // Zero-initialized weights => ||w|| = 0 => trust_ratio must fall back to 1.0.
    let w_zero = Tensor::zeros([4]).requires_grad_();
    let cfg = LambConfig {
        lr: 0.1,
        betas: [0.9, 0.999],
        eps: 1e-6,
        weight_decay: 0.0,
        trust_clip: Some(2.0),
        bias_correction: false,
    };
    let mut opt = Lamb::from_params(vec![("w".into(), w_zero.clone())], cfg).unwrap();
    let g = Tensor::from_slice(&[1.0f32, -1.0, 2.0, -2.0], [4]).unwrap();
    opt.step(&MapGrads(vec![(w_zero.clone(), g)])).unwrap();
    assert_eq!(opt.last_trust_ratio(0, 0), Some(1.0));
    let vals = w_zero.to_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite() && *v != 0.0));

    // Huge weights with tiny update => raw ratio >> clip => clamped to trust_clip.
    let w_big = Tensor::from_slice(&[100.0f32, 100.0], [2]).unwrap().requires_grad_();
    let mut opt2 = Lamb::from_params(
        vec![("w".into(), w_big.clone())],
        LambConfig { trust_clip: Some(1.5), weight_decay: 0.0, ..LambConfig::default() },
    )
    .unwrap();
    opt2.step(&MapGrads(vec![(w_big.clone(), Tensor::from_slice(&[0.5f32, 0.5], [2]).unwrap())])).unwrap();
    assert!((opt2.last_trust_ratio(0, 0).unwrap() - 1.5).abs() < 1e-12);

    // Zero gradient with wd = 0 => ||u|| = 0 => trust_ratio falls back to 1.0 and weights stay exact.
    let w_keep = Tensor::from_slice(&[3.0f32, -4.0], [2]).unwrap().requires_grad_();
    let mut opt3 = Lamb::from_params(
        vec![("w".into(), w_keep.clone())],
        LambConfig { weight_decay: 0.0, ..LambConfig::default() },
    )
    .unwrap();
    opt3.step(&MapGrads(vec![(w_keep.clone(), Tensor::zeros([2]))])).unwrap();
    assert_eq!(opt3.last_trust_ratio(0, 0), Some(1.0));
    assert_eq!(w_keep.to_vec::<f32>().unwrap(), vec![3.0, -4.0]);
}

#[test]
fn lamb_skips_non_finite_grads_cleanly() {
    let w1 = Tensor::from_slice(&[1.0f32, -2.0], [2]).unwrap().requires_grad_();
    let w2 = Tensor::from_slice(&[0.5f32, 1.5], [2]).unwrap().requires_grad_();
    let mut opt =
        Lamb::from_params(vec![("w1".into(), w1.clone()), ("w2".into(), w2.clone())], LambConfig::default())
            .unwrap();

    // First step: w1 gets NaN grad (skipped), w2 gets finite grad (updated).
    let g_nan = Tensor::from_slice(&[f32::NAN, 0.1], [2]).unwrap();
    let g_ok = Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap();
    opt.step(&MapGrads(vec![(w1.clone(), g_nan), (w2.clone(), g_ok)])).unwrap();

    assert_eq!(w1.to_vec::<f32>().unwrap(), vec![1.0, -2.0]);
    assert_eq!(opt.param_step(0, 0), Some(0));
    assert_eq!(opt.param_step(0, 1), Some(1));
    assert_eq!(opt.moments(0, 0).unwrap().0.to_vec::<f32>().unwrap(), vec![0.0, 0.0]);
    assert_ne!(w2.to_vec::<f32>().unwrap(), vec![0.5, 1.5]);

    // Second step: Inf gradient also skipped without touching w1.
    let g_inf = Tensor::from_slice(&[f32::INFINITY, 0.1], [2]).unwrap();
    opt.step(&MapGrads(vec![(w1.clone(), g_inf)])).unwrap();
    assert_eq!(w1.to_vec::<f32>().unwrap(), vec![1.0, -2.0]);
    assert_eq!(opt.param_step(0, 0), Some(0));
}

#[test]
fn lamb_converges_deterministically_on_quadratic() {
    let mut rng = Rng::seed(42);
    let target = Tensor::from_slice(&[2.5f32, -1.5, 0.75, -3.0, 1.25, 0.5], [6]).unwrap();
    let w = Tensor::randn([6], 0.0, 1.0, &mut rng).requires_grad_();
    let init_loss: f32 = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap().item().unwrap();

    let mut opt = Lamb::from_params(
        vec![("w".into(), w.clone())],
        LambConfig { lr: 0.06, weight_decay: 0.0, ..LambConfig::default() },
    )
    .unwrap();
    for _ in 0..150 {
        let loss = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    let final_loss: f32 = w.sub(&target).unwrap().square().unwrap().mean_all().unwrap().item().unwrap();
    assert!(
        final_loss < init_loss * 1e-3 && final_loss < 1e-3,
        "LAMB failed to converge: init={init_loss} final={final_loss}"
    );
}

#[test]
fn lamb_checkpoint_contract_and_state_bit_exact_resume() {
    let make_pair = || {
        let w1 = Tensor::from_slice(&[1.2f32, -0.8, 0.4], [3]).unwrap().requires_grad_();
        let w2 = Tensor::from_slice(&[-0.5f32, 2.0], [2]).unwrap().requires_grad_();
        let g1 = ParamGroup::new("decay", vec![("w1".into(), w1.clone())])
            .lr(0.02)
            .momentum(0.85)
            .adam(0.99, 1e-6)
            .weight_decay(0.01);
        let g2 = ParamGroup::new("bias", vec![("w2".into(), w2.clone())])
            .lr(0.01)
            .momentum(0.9)
            .adam(0.999, 1e-6)
            .weight_decay(0.0);
        let opt = Lamb::new(vec![g1, g2], Some(5.0), true).unwrap();
        (w1, w2, opt)
    };

    let (w1_a, w2_a, mut opt_a) = make_pair();
    let gs = [
        (vec![0.3f32, -0.5, 0.2], Some(vec![0.4f32, -0.1])),
        (vec![-0.2f32, 0.7, -0.4], None), // sparse grad on w2
        (vec![0.1f32, -0.3, 0.6], Some(vec![-0.2f32, 0.5])),
        (vec![-0.5f32, 0.2, -0.1], Some(vec![0.3f32, -0.4])),
    ];

    // Step 0 & 1 on A (with a scheduled LR change on step 1).
    for (idx, (g1v, g2o)) in gs[..2].iter().enumerate() {
        if idx == 1 {
            opt_a.groups_mut()[0].lr = 0.015;
            opt_a.groups_mut()[0].momentum = 0.88;
        }
        let mut entries = vec![(w1_a.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        if let Some(g2v) = g2o {
            entries.push((w2_a.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
        }
        opt_a.step(&MapGrads(entries)).unwrap();
    }

    // Serialize contract via JSON and export state tensors.
    let contract_json = serde_json::to_string(&opt_a.contract()).unwrap();
    let contract: LambContract = serde_json::from_str(&contract_json).unwrap();
    let state_map: HashMap<String, Tensor> = opt_a.state_tensors().unwrap().into_iter().collect();

    // Build B from current weights and resume.
    let (w1_b, w2_b, mut opt_b) = make_pair();
    {
        let _ng = folo_core::no_grad();
        w1_b.copy_from(&w1_a).unwrap();
        w2_b.copy_from(&w2_a).unwrap();
    }
    // Non-zero load_state without apply_contract must fail.
    assert!(opt_b.load_state(contract.step, &state_map).is_err());

    opt_b.apply_contract(&contract).unwrap();
    // Corrupt state rejection: negative or Inf exp_avg_sq or NaN exp_avg must fail atomically.
    let mut bad_state = state_map.clone();
    bad_state.insert("w1.exp_avg_sq".into(), Tensor::from_slice(&[-0.1f32, 0.0, 0.0], [3]).unwrap());
    assert!(opt_b.load_state(contract.step, &bad_state).is_err());
    let mut bad_nan = state_map.clone();
    bad_nan.insert("w1.exp_avg".into(), Tensor::from_slice(&[f32::NAN, 0.0, 0.0], [3]).unwrap());
    assert!(opt_b.load_state(contract.step, &bad_nan).is_err());

    // Valid load commits atomically and matches A bit-for-bit on subsequent steps.
    opt_b.load_state(contract.step, &state_map).unwrap();
    for (g1v, g2o) in &gs[2..] {
        let mut ea = vec![(w1_a.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        let mut eb = vec![(w1_b.clone(), Tensor::from_slice(g1v, [3]).unwrap())];
        if let Some(g2v) = g2o {
            ea.push((w2_a.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
            eb.push((w2_b.clone(), Tensor::from_slice(g2v, [2]).unwrap()));
        }
        opt_a.step(&MapGrads(ea)).unwrap();
        opt_b.step(&MapGrads(eb)).unwrap();
    }

    assert_eq!(
        w1_a.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w1_b.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        w2_a.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w2_b.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(opt_a.contract(), opt_b.contract());
}

#[test]
fn lamb_rejects_invalid_configs() {
    let w = Tensor::zeros([2]).requires_grad_();
    for bad in [
        LambConfig { lr: -1e-3, ..LambConfig::default() },
        LambConfig { betas: [1.0, 0.999], ..LambConfig::default() },
        LambConfig { betas: [0.9, -0.1], ..LambConfig::default() },
        LambConfig { eps: 0.0, ..LambConfig::default() },
        LambConfig { weight_decay: -0.01, ..LambConfig::default() },
        LambConfig { trust_clip: Some(0.0), ..LambConfig::default() },
        LambConfig { trust_clip: Some(f64::NAN), ..LambConfig::default() },
    ] {
        assert!(Lamb::from_params(vec![("w".into(), w.clone())], bad).is_err());
    }
}
