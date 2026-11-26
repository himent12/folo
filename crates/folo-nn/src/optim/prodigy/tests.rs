//! Unit tests for [`Prodigy`]: parameter-free `d_k` adaptation, deterministic quadratic & ill-conditioned
//! convergence, safeguard warmup, growth-rate cap, decoupled vs coupled weight decay, contract/state resume,
//! and config/group validation.

use std::collections::HashMap;

use folo_core::{Tensor, TensorId};

use super::*;
use crate::optim::ParamGroup;

struct MapGrads(HashMap<TensorId, Tensor>);

impl GradSource for MapGrads {
    fn grad(&self, param: &Tensor) -> Option<&Tensor> {
        self.0.get(&param.id())
    }
}

fn make_grads(pairs: &[(&Tensor, Tensor)]) -> MapGrads {
    let mut map = HashMap::with_capacity(pairs.len());
    for (p, g) in pairs {
        map.insert(p.id(), g.clone());
    }
    MapGrads(map)
}

/// On a quadratic bowl `0.5 * ||x - target||^2` starting far from the optimum (`||x_0 - x_*|| = 10`),
/// Prodigy starts at `d0 = 1e-6` with `lr = 1.0`, automatically expands `d_k` by orders of magnitude,
/// and converges deterministically near `target`.
#[test]
fn prodigy_adapts_d_and_converges_on_quadratic() {
    let target = Tensor::from_slice(&[5.0f32, -5.0, 4.0, -3.0], [4]).unwrap();
    let x = Tensor::zeros([4]).requires_grad_();
    let cfg = ProdigyConfig {
        lr: 1.0,
        betas: [0.9, 0.999],
        d0: 1e-6,
        d_coef: 1.0,
        weight_decay: 0.0,
        ..Default::default()
    };
    let mut opt = Prodigy::from_params(vec![("x".into(), x.clone())], cfg).unwrap();
    assert_eq!(opt.d(), 1e-6);

    let init_loss = x.sub(&target).unwrap().square().unwrap().sum_all().unwrap().item::<f32>().unwrap();
    let mut prev_d = opt.d();
    for _ in 0..160 {
        let loss = x.sub(&target).unwrap().square().unwrap().sum_all().unwrap().mul_scalar(0.5).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        assert!(opt.d() >= prev_d, "d must be monotonically non-decreasing: {} < {prev_d}", opt.d());
        prev_d = opt.d();
    }
    let final_loss = x.sub(&target).unwrap().square().unwrap().sum_all().unwrap().item::<f32>().unwrap();
    assert!(opt.d() > 1e-2, "expected Prodigy d to grow well past 1e-2, got {}", opt.d());
    assert!(
        final_loss < init_loss * 1e-4,
        "expected strong convergence: init={init_loss}, final={final_loss}"
    );
    assert_eq!(opt.step_count(), 160);
    assert_eq!(opt.param_step(0, 0), Some(160));
}

/// Verify exact hand-computed step-1 and step-2 equations against `Prodigy::step`.
#[test]
fn prodigy_matches_hand_computed_steps() {
    let x = Tensor::from_slice(&[2.0f32, -1.0], [2]).unwrap().requires_grad_();
    let cfg = ProdigyConfig {
        lr: 1.0,
        betas: [0.9, 0.99],
        beta3: Some(0.95),
        eps: 1e-8,
        weight_decay: 0.1,
        decoupled_wd: true,
        d0: 1e-4,
        d_coef: 1.0,
        growth_rate: f64::INFINITY,
        safeguard_warmup: false,
        bias_correction: true,
    };
    let mut opt = Prodigy::from_params(vec![("x".into(), x.clone())], cfg).unwrap();

    // Step 1: p == x0 so <g, x0 - p> == 0 and d stays d0.
    let g1 = Tensor::from_slice(&[0.5f32, -0.25], [2]).unwrap();
    opt.step(&make_grads(&[(&x, g1)])).unwrap();
    assert_eq!(opt.d(), 1e-4);
    assert_eq!(opt.d_numerator(), 0.0);

    let bc1 = (1.0 - 0.99f64).sqrt() / (1.0 - 0.9f64);
    let dlr1 = 1e-4 * bc1;
    let p1_want: Vec<f32> = [2.0f64, -1.0]
        .into_iter()
        .zip([0.5f64, -0.25])
        .map(|(p0, g)| {
            let dg = 1e-4 * g;
            let m = (0.1 * dg) as f32 as f64;
            let v = (0.01 * dg * dg) as f32 as f64;
            let p_dec = p0 * (1.0 - 1e-4 * 0.1);
            (p_dec - dlr1 * (m / (v.sqrt() + 1e-4 * 1e-8))) as f32
        })
        .collect();
    let p1_got = x.to_vec::<f32>().unwrap();
    for (got, want) in p1_got.iter().zip(&p1_want) {
        assert!((got - want).abs() < 1e-6, "step 1 mismatch: got={got} want={want}");
    }

    // Subsequent steps with same-direction gradient: d_numerator becomes positive on step 2
    // (d_hat ≈ 0.5 * (k - 1) * d0), and `d` strictly exceeds `d0` once k >= 4.
    let g_const = Tensor::from_slice(&[0.5f32, -0.25], [2]).unwrap();
    opt.step(&make_grads(&[(&x, g_const.clone())])).unwrap();
    assert!(opt.d_numerator() > 0.0);
    for _ in 0..3 {
        opt.step(&make_grads(&[(&x, g_const.clone())])).unwrap();
    }
    assert!(opt.d() > 1e-4, "expected d to grow after 5 consistent steps, got {}", opt.d());
}

/// `safeguard_warmup` prevents small warmup LRs from inflating `d_hat`, and `growth_rate` caps per-step growth.
#[test]
fn prodigy_safeguard_warmup_and_growth_rate_cap() {
    let run = |safeguard_warmup: bool, growth_rate: f64| {
        let target = Tensor::from_slice(&[10.0f32, -10.0], [2]).unwrap();
        let x = Tensor::zeros([2]).requires_grad_();
        let cfg = ProdigyConfig {
            lr: 1e-3, // small warmup LR
            d0: 1e-4,
            safeguard_warmup,
            growth_rate,
            ..Default::default()
        };
        let mut opt = Prodigy::from_params(vec![("x".into(), x.clone())], cfg).unwrap();
        for _ in 0..5 {
            let loss = x.sub(&target).unwrap().square().unwrap().sum_all().unwrap().mul_scalar(0.5).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        opt.d()
    };

    let d_unguarded = run(false, f64::INFINITY);
    let d_guarded = run(true, f64::INFINITY);
    assert!(
        d_guarded <= d_unguarded,
        "safeguard_warmup should keep d conservative during tiny warmup LR: guarded={d_guarded}, unguarded={d_unguarded}"
    );

    let d_capped = run(false, 1.5);
    let max_possible = 1e-4 * 1.5f64.powi(5);
    assert!(
        d_capped <= max_possible * (1.0 + 1e-12),
        "growth_rate=1.5 violated: d={d_capped} > {max_possible}"
    );
}

/// Contract + `state_tensors` / `load_state` produces bit-exact continuation after checkpoint resume.
#[test]
fn prodigy_checkpoint_contract_bit_exact_resume() {
    let target = Tensor::from_slice(&[3.0f32, -4.0, 2.5], [3]).unwrap();
    let x1 = Tensor::from_slice(&[0.5f32, 0.2, -0.3], [3]).unwrap().requires_grad_();
    let x2 = Tensor::from_slice(&[0.5f32, 0.2, -0.3], [3]).unwrap().requires_grad_();

    let cfg = ProdigyConfig {
        lr: 1.0,
        weight_decay: 0.01,
        safeguard_warmup: true,
        growth_rate: 4.0,
        ..Default::default()
    };
    let mut opt_ref = Prodigy::from_params(vec![("w".into(), x1.clone())], cfg).unwrap();
    let mut opt_res = Prodigy::from_params(vec![("w".into(), x2.clone())], ProdigyConfig::default()).unwrap();

    for step in 0..6 {
        let loss = x1.sub(&target).unwrap().square().unwrap().sum_all().unwrap().mul_scalar(0.5).unwrap();
        let grads = loss.backward().unwrap();
        if step == 2 {
            // Test missing-gradient step skipping.
            opt_ref.step(&make_grads(&[])).unwrap();
            continue;
        }
        opt_ref.step(&grads).unwrap();
    }

    // Snapshot contract (round-tripped through JSON) and state tensors.
    let contract = opt_ref.contract();
    let json = serde_json::to_string(&contract).unwrap();
    let decoded: ProdigyContract = serde_json::from_str(&json).unwrap();
    assert_eq!(contract, decoded);

    let state_map: HashMap<String, Tensor> = opt_ref.state_tensors().unwrap().into_iter().collect();
    // Non-zero step load without apply_contract must fail cleanly.
    assert!(opt_res.load_state(contract.step, &state_map).is_err());

    {
        let _ng = folo_core::no_grad();
        x2.copy_from(&x1).unwrap();
    }
    opt_res.apply_contract(&decoded).unwrap();
    opt_res.load_state(decoded.step, &state_map).unwrap();
    assert_eq!(opt_res.d(), opt_ref.d());
    assert_eq!(opt_res.d_numerator(), opt_ref.d_numerator());
    assert_eq!(opt_res.step_count(), opt_ref.step_count());

    // Continue 5 more steps and assert bit-for-bit equality.
    for _ in 0..5 {
        let g_ref = x1.sub(&target).unwrap().square().unwrap().sum_all().unwrap().backward().unwrap();
        let g_res = x2.sub(&target).unwrap().square().unwrap().sum_all().unwrap().backward().unwrap();
        opt_ref.step(&g_ref).unwrap();
        opt_res.step(&g_res).unwrap();
    }
    assert_eq!(x1.to_vec::<f32>().unwrap(), x2.to_vec::<f32>().unwrap());
    assert_eq!(opt_ref.d().to_bits(), opt_res.d().to_bits());
}

/// Invalid hyperparameters, non-finite values, and group/state mismatches are rejected before mutation.
#[test]
fn prodigy_validation_rejects_bad_configs() {
    let w = Tensor::ones([4]).requires_grad_();
    assert!(
        Prodigy::from_params(vec![("w".into(), w.clone())], ProdigyConfig { d0: 0.0, ..Default::default() })
            .is_err()
    );
    assert!(
        Prodigy::from_params(
            vec![("w".into(), w.clone())],
            ProdigyConfig { d_coef: -1.0, ..Default::default() }
        )
        .is_err()
    );
    assert!(
        Prodigy::from_params(
            vec![("w".into(), w.clone())],
            ProdigyConfig { growth_rate: 0.5, ..Default::default() }
        )
        .is_err()
    );
    assert!(
        Prodigy::from_params(
            vec![("w".into(), w.clone())],
            ProdigyConfig { growth_rate: f64::NAN, ..Default::default() }
        )
        .is_err()
    );
    assert!(
        Prodigy::from_params(
            vec![("w".into(), w.clone())],
            ProdigyConfig { beta3: Some(1.0), ..Default::default() }
        )
        .is_err()
    );
    assert!(
        Prodigy::from_params(
            vec![("w".into(), w.clone())],
            ProdigyConfig { d0: 1e-25, eps: 1e-25, ..Default::default() }
        )
        .is_err()
    );

    // Multi-group construction with distinct decoupled weight decays works cleanly.
    let w2 = Tensor::ones([3]).requires_grad_();
    let g_decay = ParamGroup::new("decay", vec![("w".into(), w.clone())])
        .lr(1.0)
        .momentum(0.9)
        .adam(0.999, 1e-8)
        .weight_decay(0.05)
        .decoupled_wd(true);
    let g_bias = ParamGroup::new("bias", vec![("b".into(), w2.clone())])
        .lr(1.0)
        .momentum(0.9)
        .adam(0.999, 1e-8)
        .weight_decay(0.0)
        .decoupled_wd(true);
    let mut opt = Prodigy::new(vec![g_decay, g_bias], ProdigyConfig::default()).unwrap();
    // Mismatched grad shape fails without mutating parameters.
    let bad_g = Tensor::ones([2]);
    let w_before = w.to_vec::<f32>().unwrap();
    assert!(opt.step(&make_grads(&[(&w, bad_g)])).is_err());
    assert_eq!(w.to_vec::<f32>().unwrap(), w_before);
    assert_eq!(opt.step_count(), 0);
}
