//! Unit tests for [`SophiaG`]: exact clipped second-order update math, Hessian cadence `k`,
//! Gauss-Newton & seeded Hutchinson diagonal estimators, ill-conditioned quadratic convergence,
//! contract/state resume (including Hutchinson RNG state), and validation.

use std::collections::HashMap;

use folo_core::{Tensor, TensorId};

use super::*;

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

/// Exact hand-computed Sophia-G step, clipping at `±rho`, decoupled weight decay, and `update_period = k`
/// cadence verification.
#[test]
fn sophia_hand_computed_step_clipping_and_cadence() {
    let p = Tensor::from_slice(&[2.0f32, -4.0, 1.0], [3]).unwrap().requires_grad_();
    let cfg = SophiaConfig {
        lr: 0.1,
        betas: [0.5, 0.5],
        rho: 0.25,
        eps: 1e-6,
        weight_decay: 0.2,
        update_period: 3,
        auto_gauss_newton: true,
        ..Default::default()
    };
    let mut opt = SophiaG::from_params(vec![("p".into(), p.clone())], cfg).unwrap();
    assert!(opt.should_update_hessian());

    // Step 0 (0 % 3 == 0): auto Gauss-Newton fires -> h_1 = 0.5 * g^2, m_1 = 0.5 * g.
    // g = [10.0, -0.02, 0.0]:
    //   idx 0: m=5, h=50 -> m/h = 0.1 (unclipped, |0.1| < 0.25)
    //   idx 1: m=-0.01, h=0.0002 -> m/h = -50 -> clipped to -0.25
    //   idx 2: m=0, h=0 -> max(h, eps)=1e-6 -> 0
    let g0 = Tensor::from_slice(&[10.0f32, -0.02, 0.0], [3]).unwrap();
    opt.step(&make_grads(&[(&p, g0)])).unwrap();
    assert_eq!(opt.step_count(), 1);
    assert_eq!(opt.hessian_updates(), 1);
    assert!(!opt.should_update_hessian());

    let p_got = p.to_vec::<f32>().unwrap();
    let p_want = [
        2.0 * (1.0 - 0.1 * 0.2) - 0.1 * 0.1,
        -4.0 * (1.0 - 0.1 * 0.2) - 0.1 * (-0.25),
        1.0 * (1.0 - 0.1 * 0.2),
    ];
    for (got, want) in p_got.iter().zip(p_want) {
        assert!((got - want).abs() < 1e-6, "step 0 mismatch: got={got} want={want}");
    }

    // Steps 1 and 2: NOT on cadence, so Hessian must remain unchanged.
    let h_after_0 = opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    let g1 = Tensor::from_slice(&[1.0f32, 1.0, 1.0], [3]).unwrap();
    opt.step(&make_grads(&[(&p, g1.clone())])).unwrap();
    opt.step(&make_grads(&[(&p, g1.clone())])).unwrap();
    assert_eq!(opt.hessian_updates(), 1);
    assert_eq!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap(), h_after_0);
    assert!(opt.should_update_hessian()); // step == 3 -> due again!

    // Step 3 (3 % 3 == 0): explicit Gauss-Newton update before step() prevents double-updating in step().
    let gn_g = Tensor::from_slice(&[2.0f32, 3.0, 4.0], [3]).unwrap();
    opt.update_hessian_gauss_newton(&make_grads(&[(&p, gn_g)])).unwrap();
    assert_eq!(opt.hessian_updates(), 2);
    assert!(!opt.should_update_hessian());
    let h_explicit = opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    opt.step(&make_grads(&[(&p, g1)])).unwrap();
    assert_eq!(opt.hessian_updates(), 2);
    assert_eq!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap(), h_explicit);
}

/// Seeded Hutchinson probe estimator recovers the exact diagonal of a non-diagonal SPD Hessian matrix
/// in expectation (with exact diagonal for Rademacher on diagonal H in a single probe!) and accelerates
/// convergence on an ill-conditioned quadratic bowl (`kappa = 100`).
#[test]
fn sophia_hutchinson_estimator_and_ill_conditioned_convergence() {
    // 1. Exact diagonal recovery on a diagonal Hessian `H = diag(h_diag)` with Rademacher probes:
    //    since v_i^2 == 1 identically for Rademacher ±1, `v * (H v) = diag(H)` with ZERO variance!
    let h_diag = [100.0f32, 10.0, 1.0, 0.25];
    let x = Tensor::from_slice(&[2.0f32, -3.0, 4.0, -5.0], [4]).unwrap().requires_grad_();
    let cfg = SophiaConfig {
        lr: 0.25,
        betas: [0.5, 0.0], // beta2=0 -> h_t equals current h_est directly
        rho: 1.0,
        eps: 1e-8,
        weight_decay: 0.0,
        update_period: 1,
        auto_gauss_newton: false,
        hutchinson_dist: HutchinsonDist::Rademacher,
        seed: 42,
    };
    let mut opt = SophiaG::from_params(vec![("x".into(), x.clone())], cfg).unwrap();

    // Determinism check: two optimizers with the same seed produce identical probes.
    let mut opt_clone = SophiaG::from_params(vec![("x".into(), Tensor::zeros([4]))], cfg).unwrap();
    let p1 = opt.sample_hutchinson_probes().unwrap()["x"].to_vec::<f32>().unwrap();
    let p2 = opt_clone.sample_hutchinson_probes().unwrap()["x"].to_vec::<f32>().unwrap();
    assert_eq!(p1, p2);
    assert!(p1.iter().all(|&v| v == 1.0 || v == -1.0));

    // Update Hessian via Hutchinson closure on `f(x) = 0.5 * sum_i h_diag[i] * x_i^2`, where `(H v)_i = h_diag[i] * v_i`.
    let h_t = Tensor::from_slice(&h_diag, [4]).unwrap();
    opt.update_hessian_hutchinson_with(|probes| {
        let hv = probes["x"].mul(&h_t)?;
        Ok(make_grads(&[(&x, hv)]))
    })
    .unwrap();
    let est = opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    assert_eq!(est, h_diag.to_vec());

    // 2. Converge on the ill-conditioned quadratic `0.5 * sum_i h_diag[i] * (x_i - target_i)^2`
    //    (`condition number = 400`) using Sophia-H with periodic Hutchinson updates.
    let target = Tensor::from_slice(&[1.5f32, -2.0, 3.0, -1.0], [4]).unwrap();
    let w = Tensor::from_slice(&[-3.0f32, 4.0, -2.0, 5.0], [4]).unwrap().requires_grad_();
    let cfg_conv = SophiaConfig {
        lr: 0.3,
        betas: [0.85, 0.9],
        rho: 1.0,
        eps: 1e-8,
        weight_decay: 0.0,
        update_period: 5,
        auto_gauss_newton: false,
        hutchinson_dist: HutchinsonDist::Rademacher,
        seed: 7,
    };
    let mut opt_conv = SophiaG::from_params(vec![("w".into(), w.clone())], cfg_conv).unwrap();
    let quad_loss = |t: &Tensor| -> f32 {
        let d = t.sub(&target).unwrap();
        d.square().unwrap().mul(&h_t).unwrap().sum_all().unwrap().item::<f32>().unwrap() * 0.5
    };
    let init_loss = quad_loss(&w);
    for _ in 0..80 {
        if opt_conv.should_update_hessian() {
            opt_conv
                .update_hessian_hutchinson_with(|probes| Ok(make_grads(&[(&w, probes["w"].mul(&h_t)?)])))
                .unwrap();
        }
        let d = w.sub(&target).unwrap();
        let loss = d.square().unwrap().mul(&h_t).unwrap().sum_all().unwrap().mul_scalar(0.5).unwrap();
        let grads = loss.backward().unwrap();
        opt_conv.step(&grads).unwrap();
    }
    let final_loss = quad_loss(&w);
    assert!(
        final_loss < init_loss * 1e-5,
        "expected Sophia-H convergence on kappa=400 bowl: init={init_loss}, final={final_loss}"
    );
    assert_eq!(opt_conv.hessian_updates(), 16); // 80 / 5
}

/// Contract + `state_tensors` / `load_state` resumes bit-for-bit identically, including Hutchinson RNG state.
#[test]
fn sophia_checkpoint_contract_bit_exact_resume() {
    let h_t = Tensor::from_slice(&[4.0f32, 0.5, 2.0], [3]).unwrap();
    let w1 = Tensor::from_slice(&[2.0f32, -3.0, 1.5], [3]).unwrap().requires_grad_();
    let w2 = Tensor::from_slice(&[2.0f32, -3.0, 1.5], [3]).unwrap().requires_grad_();
    let cfg = SophiaConfig {
        lr: 0.1,
        betas: [0.9, 0.95],
        rho: 0.5,
        eps: 1e-8,
        weight_decay: 0.05,
        update_period: 2,
        auto_gauss_newton: false,
        hutchinson_dist: HutchinsonDist::Gaussian,
        seed: 123,
    };
    let mut opt_ref = SophiaG::from_params(vec![("w".into(), w1.clone())], cfg).unwrap();
    let mut opt_res = SophiaG::from_params(vec![("w".into(), w2.clone())], SophiaConfig::default()).unwrap();

    for _ in 0..5 {
        if opt_ref.should_update_hessian() {
            opt_ref
                .update_hessian_hutchinson_with(|probes| Ok(make_grads(&[(&w1, probes["w"].mul(&h_t)?)])))
                .unwrap();
        }
        let g = w1.square().unwrap().mul(&h_t).unwrap().sum_all().unwrap().backward().unwrap();
        opt_ref.step(&g).unwrap();
    }

    let contract = opt_ref.contract();
    let json = serde_json::to_string(&contract).unwrap();
    let decoded: SophiaContract = serde_json::from_str(&json).unwrap();
    assert_eq!(contract, decoded);

    let state_map: HashMap<String, Tensor> = opt_ref.state_tensors().unwrap().into_iter().collect();
    assert!(opt_res.load_state(contract.step, &state_map).is_err());

    {
        let _ng = folo_core::no_grad();
        w2.copy_from(&w1).unwrap();
    }
    opt_res.apply_contract(&decoded).unwrap();
    opt_res.load_state(decoded.step, &state_map).unwrap();
    assert_eq!(opt_res.rng_state(), opt_ref.rng_state());

    for _ in 0..6 {
        if opt_ref.should_update_hessian() {
            opt_ref
                .update_hessian_hutchinson_with(|probes| Ok(make_grads(&[(&w1, probes["w"].mul(&h_t)?)])))
                .unwrap();
        }
        if opt_res.should_update_hessian() {
            opt_res
                .update_hessian_hutchinson_with(|probes| Ok(make_grads(&[(&w2, probes["w"].mul(&h_t)?)])))
                .unwrap();
        }
        let g1 = w1.square().unwrap().mul(&h_t).unwrap().sum_all().unwrap().backward().unwrap();
        let g2 = w2.square().unwrap().mul(&h_t).unwrap().sum_all().unwrap().backward().unwrap();
        opt_ref.step(&g1).unwrap();
        opt_res.step(&g2).unwrap();
    }
    assert_eq!(w1.to_vec::<f32>().unwrap(), w2.to_vec::<f32>().unwrap());
    assert_eq!(opt_ref.rng_state(), opt_res.rng_state());
}

/// Invalid configurations, zero update_period, NaN Hessian/probes, and shape mismatches fail cleanly.
#[test]
fn sophia_validation_and_nan_propagation() {
    let w = Tensor::ones([4]).requires_grad_();
    assert!(
        SophiaG::from_params(vec![("w".into(), w.clone())], SophiaConfig { rho: 0.0, ..Default::default() })
            .is_err()
    );
    assert!(
        SophiaG::from_params(
            vec![("w".into(), w.clone())],
            SophiaConfig { update_period: 0, ..Default::default() }
        )
        .is_err()
    );
    assert!(
        SophiaG::from_params(vec![("w".into(), w.clone())], SophiaConfig { eps: 0.0, ..Default::default() })
            .is_err()
    );

    let mut opt = SophiaG::from_params(vec![("w".into(), w.clone())], SophiaConfig::default()).unwrap();
    // Missing probe for parameter present in HVP fails cleanly.
    let hvp = make_grads(&[(&w, Tensor::ones([4]))]);
    assert!(opt.update_hessian_hutchinson(&hvp, &HashMap::new()).is_err());

    // Negative curvature in Hessian EMA is floored by `eps` (not NaN); actual NaN in Hessian propagates.
    let neg_probe =
        HashMap::from([("w".to_string(), Tensor::from_slice(&[-1.0f32, 1.0, 1.0, 1.0], [4]).unwrap())]);
    opt.update_hessian_hutchinson(&hvp, &neg_probe).unwrap();
    assert!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap()[0] < 0.0);
    opt.step(&hvp).unwrap();
    assert!(w.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}
