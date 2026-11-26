//! Unit tests for [`RAdam`]: exact analytical `rho_inf`/`rho_t`/`r_t` rectification formulas, hand-computed
//! unrectified (`rho_t <= 4`) and rectified (`rho_t > 4`) steps, decoupled vs coupled weight decay,
//! multi-group optimization convergence, missing-gradient skipping, serialization & bit-exact checkpoint
//! resume (`RAdamContract` + `state_tensors` / `load_state`), and hyperparameter validation.

use std::collections::HashMap;

use folo_core::Tensor;

use super::*;
use crate::optim::{GradAccumulator, ParamGroup};

/// Verify exact analytical `radam_rectification` formulas across unrectified (`rho_t <= 4`),
/// transition (`rho_t > 4`), asymptotic (`r_t -> 1.0`), `beta2 = 0.0`, and invalid inputs.
#[test]
fn radam_rectification_analytical_values_and_transition() {
    // beta2 = 0.999 -> rho_inf = 1999.0; steps 1..=4 have rho_t <= 4 (unrectified), step 5+ rectified!
    let r1 = radam_rectification(1, 0.999).unwrap();
    assert!((r1.rho_inf - 1999.0).abs() < 1e-10, "rho_inf={}", r1.rho_inf);
    assert!((r1.rho_t - 1.0).abs() < 1e-12);
    assert!(r1.rect.is_none());

    for t in 1..=4 {
        let rt = radam_rectification(t, 0.999).unwrap();
        assert!(rt.rho_t <= 4.0, "step {t}: expected rho_t <= 4, got {}", rt.rho_t);
        assert!(rt.rect.is_none(), "step {t}: expected unrectified");
    }

    let r5 = radam_rectification(5, 0.999).unwrap();
    assert!(r5.rho_t > 4.0, "step 5 should cross rho_t > 4, got {}", r5.rho_t);
    let b2_5 = 0.999f64.powi(5);
    let want_rho5 = 1999.0 - 10.0 * b2_5 / (1.0 - b2_5);
    let want_r5 = (((want_rho5 - 4.0) * (want_rho5 - 2.0) * 1999.0) / (1995.0 * 1997.0 * want_rho5)).sqrt();
    assert!((r5.rho_t - want_rho5).abs() < 1e-10);
    assert!((r5.rect.unwrap() - want_r5).abs() < 1e-12);

    // Strictly monotonic r_t approaching 1.0 as t grows; exact 1.0 when beta2^t underflows.
    let mut prev_r = 0.0;
    for &t in &[5u64, 10, 50, 500, 5_000, 1_000_000] {
        let rt = radam_rectification(t, 0.999).unwrap().rect.unwrap();
        assert!(rt > prev_r && rt <= 1.0, "step {t}: r_t={rt} vs prev {prev_r}");
        prev_r = rt;
    }
    let r_inf = radam_rectification(10_000_000, 0.999).unwrap();
    assert_eq!(r_inf.rect, Some(1.0));

    // beta2 = 0.5 -> rho_inf = 3.0 <= 4 -> never rectified at any step.
    assert!(radam_rectification(100, 0.5).unwrap().rect.is_none());
    // beta2 = 0.0 -> rho_inf = 1.0 -> unrectified.
    assert!(radam_rectification(1, 0.0).unwrap().rect.is_none());

    // Invalid inputs.
    assert!(radam_rectification(0, 0.999).is_err());
    assert!(radam_rectification(1, -0.1).is_err());
    assert!(radam_rectification(1, 1.0).is_err());
    assert!(radam_rectification(1, f64::NAN).is_err());
}

/// Hand-computed step-by-step comparison against `f64` reference across BOTH unrectified steps
/// (`t = 1..=4`) and rectified steps (`t = 5..=8`), for both decoupled and coupled weight decay.
#[test]
fn radam_matches_analytical_steps_unrectified_and_rectified() {
    for decoupled_wd in [true, false] {
        let cfg = RAdamConfig { lr: 0.05, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.02, decoupled_wd };
        let p = Tensor::from_slice(&[1.5f32, -0.75, 0.25], [3]).unwrap().requires_grad_();
        let mut opt = RAdam::from_params(vec![("w".into(), p.clone())], cfg).unwrap();

        let mut p_ref = [1.5f64, -0.75, 0.25];
        let mut m_ref = [0.0f64; 3];
        let mut v_ref = [0.0f64; 3];

        for step in 1..=8u64 {
            let g_vals = [0.3f32 * step as f32, -0.2 * step as f32, 0.1 + 0.05 * step as f32];
            let g = Tensor::from_slice(&g_vals, [3]).unwrap();
            let loss = p.mul(&g).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();

            let info = radam_rectification(step, cfg.betas[1]).unwrap();
            let bc1 = 1.0 - cfg.betas[0].powi(step as i32);
            let bc2_sqrt = info.one_minus_beta2_t.sqrt();
            for j in 0..3 {
                let g_eff = if !decoupled_wd {
                    g_vals[j] as f64 + cfg.weight_decay * p_ref[j]
                } else {
                    g_vals[j] as f64
                };
                if decoupled_wd {
                    p_ref[j] *= 1.0 - cfg.lr * cfg.weight_decay;
                }
                m_ref[j] = cfg.betas[0] * m_ref[j] + (1.0 - cfg.betas[0]) * g_eff;
                v_ref[j] = cfg.betas[1] * v_ref[j] + (1.0 - cfg.betas[1]) * g_eff * g_eff;
                if let Some(r_t) = info.rect {
                    let denom = v_ref[j].sqrt() / bc2_sqrt + cfg.eps;
                    p_ref[j] -= (cfg.lr * r_t / bc1) * (m_ref[j] / denom);
                } else {
                    p_ref[j] -= (cfg.lr / bc1) * m_ref[j];
                }
            }

            let got = p.to_vec::<f32>().unwrap();
            for j in 0..3 {
                assert!(
                    (got[j] as f64 - p_ref[j]).abs() < 1e-5,
                    "decoupled={decoupled_wd} step={step} idx={j}: got {} vs ref {}",
                    got[j],
                    p_ref[j]
                );
            }
        }
        assert_eq!(opt.step_count(), 8);
        assert_eq!(opt.param_step(0, 0), Some(8));
    }
}

/// Convergence on an ill-conditioned quadratic, skipped missing-gradient parameters, bit-exact
/// contract/state round-trip resume, and hyperparameter validation.
#[test]
fn radam_convergence_resume_and_validation() {
    // 1. Minimize f(x) = 10*(x0 - 2)^2 + 0.5*(x1 + 3)^2 + 4*(x2 - 1)^2.
    let target = Tensor::from_slice(&[2.0f32, -3.0, 1.0], [3]).unwrap();
    let scales = Tensor::from_slice(&[10.0f32, 0.5, 4.0], [3]).unwrap();
    let x = Tensor::from_slice(&[-4.0f32, 5.0, -3.0], [3]).unwrap().requires_grad_();
    let unused = Tensor::from_slice(&[7.0f32, 8.0], [2]).unwrap().requires_grad_();
    let mut opt = RAdam::from_params(
        vec![("x".into(), x.clone()), ("unused".into(), unused.clone())],
        RAdamConfig { lr: 0.2, weight_decay: 0.0, ..RAdamConfig::default() },
    )
    .unwrap();

    for _ in 0..250 {
        let diff = x.sub(&target).unwrap();
        let loss = diff.square().unwrap().mul(&scales).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
    }
    let xv = x.to_vec::<f32>().unwrap();
    assert!(
        (xv[0] - 2.0).abs() < 1e-2 && (xv[1] + 3.0).abs() < 1e-2 && (xv[2] - 1.0).abs() < 1e-2,
        "x={xv:?}"
    );
    // Unused parameter was never touched.
    assert_eq!(unused.to_vec::<f32>().unwrap(), vec![7.0, 8.0]);
    assert_eq!(opt.param_step(0, 1), Some(0));

    // 2. Bit-exact checkpoint save & resume across unrectified -> rectified boundary.
    let p_a = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], [4]).unwrap().requires_grad_();
    let p_b = Tensor::from_slice(&[1.0f32, -2.0, 0.5, 3.0], [4]).unwrap().requires_grad_();
    let cfg = RAdamConfig { lr: 0.03, weight_decay: 0.01, ..RAdamConfig::default() };
    let mut opt_a = RAdam::from_params(vec![("p".into(), p_a.clone())], cfg).unwrap();
    let mut opt_b = RAdam::from_params(vec![("p".into(), p_b.clone())], cfg).unwrap();
    let mut acc_a = GradAccumulator::new(std::slice::from_ref(&p_a));
    let mut acc_b = GradAccumulator::new(std::slice::from_ref(&p_b));

    for step in 0..3 {
        let g = Tensor::from_slice(&[0.5f32, -0.4, 0.2 + step as f32 * 0.1, -0.7], [4]).unwrap();
        acc_a.add(&p_a.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
        opt_a.step(&acc_a).unwrap();
        acc_a.zero();
    }
    // Serialize contract & state from opt_a at step 3 (unrectified), restore into opt_b.
    let contract_json = serde_json::to_string(&opt_a.contract()).unwrap();
    let contract: RAdamContract = serde_json::from_str(&contract_json).unwrap();
    let state_map: HashMap<String, Tensor> = opt_a.state_tensors().unwrap().into_iter().collect();
    {
        let _ng = folo_core::no_grad();
        p_b.copy_from(&p_a.detach()).unwrap();
    }
    opt_b.apply_contract(&contract).unwrap();
    opt_b.load_state(contract.step, &state_map).unwrap();

    // Run 5 more steps (crossing into rectified regime at step 5) — must match bit-for-bit!
    for step in 3..8 {
        let g = Tensor::from_slice(&[0.2f32, 0.6, -0.3, 0.1 * step as f32], [4]).unwrap();
        acc_a.add(&p_a.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
        acc_b.add(&p_b.mul(&g).unwrap().sum_all().unwrap().backward().unwrap()).unwrap();
        opt_a.step(&acc_a).unwrap();
        opt_b.step(&acc_b).unwrap();
        acc_a.zero();
        acc_b.zero();
    }
    assert_eq!(p_a.to_vec::<f32>().unwrap(), p_b.to_vec::<f32>().unwrap());

    // 3. Validation errors.
    let q = Tensor::zeros([2]).requires_grad_();
    for bad in [
        RAdamConfig { lr: -1e-3, ..RAdamConfig::default() },
        RAdamConfig { lr: f64::NAN, ..RAdamConfig::default() },
        RAdamConfig { betas: [1.0, 0.999], ..RAdamConfig::default() },
        RAdamConfig { betas: [0.9, 1.0], ..RAdamConfig::default() },
        RAdamConfig { betas: [-0.1, 0.999], ..RAdamConfig::default() },
        RAdamConfig { eps: 0.0, ..RAdamConfig::default() },
        RAdamConfig { eps: 1e-60, ..RAdamConfig::default() }, // underflows to 0.0 in f32
        RAdamConfig { weight_decay: -0.01, ..RAdamConfig::default() },
    ] {
        assert!(RAdam::from_params(vec![("q".into(), q.clone())], bad).is_err(), "expected err for {bad:?}");
    }
    // Group validation rejects mismatched step in load_state after apply_contract.
    let mut opt_c = RAdam::new(vec![ParamGroup::new("p", vec![("p".into(), q)])]).unwrap();
    assert!(opt_c.load_state(1, &HashMap::new()).is_err());
}
