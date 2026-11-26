//! Unit tests for [`SophiaG`]: exact clipped second-order update math, Hessian cadence `k`,
//! Gauss–Newton–Bartlett (`batch_size * g_hat^2`) & seeded Hutchinson diagonal estimators, ill-conditioned
//! quadratic convergence, aliasing safety, fast/fallback `NaN` propagation parity, `load_state(0)` reset,
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

/// Exact hand-computed Sophia-G step, clipping at `±rho`, decoupled weight decay, Gauss–Newton–Bartlett
/// `batch_size * g_hat^2` scaling, and `update_period = k` cadence verification.
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
        auto_gauss_newton: false,
        ..Default::default()
    };
    assert!(!SophiaConfig::default().auto_gauss_newton);
    let mut opt = SophiaG::from_params(vec![("p".into(), p.clone())], cfg).unwrap();
    assert!(opt.should_update_hessian());

    // Step 0: explicit Gauss–Newton–Bartlett with batch_size = 4 and mean g_hat = [5.0, -0.01, 0.0]:
    //   h_1 = 0.5 * 4 * g_hat^2 = [50.0, 0.0002, 0.0]
    let g_hat = Tensor::from_slice(&[5.0f32, -0.01, 0.0], [3]).unwrap();
    opt.update_hessian_gauss_newton_scaled(&make_grads(&[(&p, g_hat)]), 4.0).unwrap();
    assert_eq!(opt.hessian_updates(), 1);
    assert!(!opt.should_update_hessian());

    let g0 = Tensor::from_slice(&[10.0f32, -0.02, 0.0], [3]).unwrap();
    opt.step(&make_grads(&[(&p, g0)])).unwrap();
    assert_eq!(opt.step_count(), 1);
    assert_eq!(opt.hessian_updates(), 1);

    let p_got = p.to_vec::<f32>().unwrap();
    let p_want = [
        2.0 * (1.0 - 0.1 * 0.2) - 0.1 * 0.1,
        -4.0 * (1.0 - 0.1 * 0.2) - 0.1 * (-0.25),
        1.0 * (1.0 - 0.1 * 0.2),
    ];
    for (got, want) in p_got.iter().zip(p_want) {
        assert!((got - want).abs() < 1e-6, "step 0 mismatch: got={got} want={want}");
    }

    // Steps 1 and 2: NOT on cadence, so Hessian remains unchanged.
    let h_after_0 = opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    let g1 = Tensor::from_slice(&[1.0f32, 1.0, 1.0], [3]).unwrap();
    opt.step(&make_grads(&[(&p, g1.clone())])).unwrap();
    opt.step(&make_grads(&[(&p, g1.clone())])).unwrap();
    assert_eq!(opt.hessian_updates(), 1);
    assert_eq!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap(), h_after_0);
    assert!(opt.should_update_hessian()); // step == 3 -> due again!

    // Step 3: explicit Gauss-Newton update before step() keeps cadence exact.
    let gn_g = Tensor::from_slice(&[2.0f32, 3.0, 4.0], [3]).unwrap();
    opt.update_hessian_gauss_newton(&make_grads(&[(&p, gn_g)])).unwrap();
    assert_eq!(opt.hessian_updates(), 2);
    assert!(!opt.should_update_hessian());
    let h_explicit = opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    opt.step(&make_grads(&[(&p, g1)])).unwrap();
    assert_eq!(opt.hessian_updates(), 2);
    assert_eq!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap(), h_explicit);
}

/// Seeded Hutchinson probe estimator recovers the exact diagonal of a diagonal SPD Hessian in one probe
/// and converges on an ill-conditioned quadratic bowl (`kappa = 400`).
#[test]
fn sophia_hutchinson_estimator_and_ill_conditioned_convergence() {
    let h_diag = [100.0f32, 10.0, 1.0, 0.25];
    let x = Tensor::from_slice(&[2.0f32, -3.0, 4.0, -5.0], [4]).unwrap().requires_grad_();
    let cfg = SophiaConfig {
        lr: 0.25,
        betas: [0.5, 0.0],
        rho: 1.0,
        eps: 1e-8,
        weight_decay: 0.0,
        update_period: 1,
        auto_gauss_newton: false,
        hutchinson_dist: HutchinsonDist::Rademacher,
        seed: 42,
    };
    let mut opt = SophiaG::from_params(vec![("x".into(), x.clone())], cfg).unwrap();

    let mut opt_clone = SophiaG::from_params(vec![("x".into(), Tensor::zeros([4]))], cfg).unwrap();
    let p1 = opt.sample_hutchinson_probes().unwrap()["x"].to_vec::<f32>().unwrap();
    let p2 = opt_clone.sample_hutchinson_probes().unwrap()["x"].to_vec::<f32>().unwrap();
    assert_eq!(p1, p2);
    assert!(p1.iter().all(|&v| v == 1.0 || v == -1.0));

    let h_t = Tensor::from_slice(&h_diag, [4]).unwrap();
    opt.update_hessian_hutchinson_with(|probes| Ok(make_grads(&[(&x, probes["x"].mul(&h_t)?)]))).unwrap();
    assert_eq!(opt.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap(), h_diag.to_vec());

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
    assert!(final_loss < init_loss * 1e-5, "init={init_loss}, final={final_loss}");
    assert_eq!(opt_conv.hessian_updates(), 16);
}

/// Contract + `state_tensors` / `load_state` resumes bit-for-bit identically, including Hutchinson RNG state,
/// and `load_state(0, &empty_map)` cleanly resets `hessian_updates` and `last_hessian_step`.
#[test]
fn sophia_checkpoint_contract_resume_and_reset() {
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
    assert_eq!(contract.step, decoded.step);
    assert_eq!(contract.rng_state, decoded.rng_state);

    let state_map: HashMap<String, Tensor> = opt_ref.state_tensors().unwrap().into_iter().collect();
    assert!(opt_res.load_state(contract.step, &state_map).is_err());

    {
        let _ng = folo_core::no_grad();
        w2.copy_from(&w1).unwrap();
    }
    opt_res.apply_contract(&contract).unwrap();
    opt_res.load_state(contract.step, &state_map).unwrap();
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

    // Regression: `load_state(0, &empty_map)` without `apply_contract` resets Hessian tracking cleanly
    // and produces a valid contract that `apply_contract` accepts.
    opt_res.load_state(0, &HashMap::new()).unwrap();
    assert_eq!(opt_res.step_count(), 0);
    assert_eq!(opt_res.hessian_updates(), 0);
    assert!(opt_res.should_update_hessian());
    let reset_contract = opt_res.contract();
    assert_eq!(reset_contract.last_hessian_step, None);
    opt_res.apply_contract(&reset_contract).unwrap();

    // Regression: inconsistent contract (`hessian_updates == 0` with `last_hessian_step == Some(0)`) rejected.
    let mut bad_contract = reset_contract;
    bad_contract.last_hessian_step = Some(0);
    assert!(opt_res.apply_contract(&bad_contract).is_err());
}

/// Regression tests for `g` aliasing `ps.hessian` under `auto_gauss_newton = true`, `NaN` propagation
/// parity across fast & fallback paths, and validation rejections.
#[test]
fn sophia_aliasing_nan_propagation_and_validation() {
    // 1. Validation rejections.
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
    let ones4 = make_grads(&[(&w, Tensor::ones([4]))]);
    assert!(opt.update_hessian_hutchinson(&ones4, &HashMap::new()).is_err());
    assert!(opt.update_hessian_gauss_newton_scaled(&ones4, 0.0).is_err());

    // 2. Regression: `g` aliasing `ps.hessian` with `auto_gauss_newton = true` must NOT clobber `g`
    //    before `exp_avg` reads it (`h = g = 2.0`, `betas = [0.5, 0.5]` -> `exp_avg == 1.0`, not `1.5`).
    let p_alias = Tensor::ones([2]).requires_grad_();
    let mut opt_alias = SophiaG::from_params(
        vec![("p".into(), p_alias.clone())],
        SophiaConfig {
            lr: 0.1,
            betas: [0.5, 0.5],
            rho: 1.0,
            eps: 1e-6,
            weight_decay: 0.0,
            update_period: 1,
            auto_gauss_newton: true,
            ..Default::default()
        },
    )
    .unwrap();
    let h_buf = opt_alias.state_for(0, 0).unwrap().1.clone();
    {
        let _ng = folo_core::no_grad();
        h_buf.copy_from(&Tensor::full([2], 2.0f32)).unwrap();
    }
    opt_alias.step(&make_grads(&[(&p_alias, h_buf)])).unwrap();
    let m_after = opt_alias.state_for(0, 0).unwrap().0.to_vec::<f32>().unwrap();
    assert_eq!(m_after, vec![1.0, 1.0], "aliased g clobbered momentum: {m_after:?}");

    // 3. Regression: NaN in gradient AND NaN in Hessian must propagate to `p` on BOTH fast path
    //    (contiguous) and fallback path (transposed non-contiguous).
    for transposed in [false, true] {
        let p_nan = if transposed {
            Tensor::ones([2, 2]).transpose(0, 1).unwrap().requires_grad_()
        } else {
            Tensor::ones([2, 2]).requires_grad_()
        };
        let mut opt_nan = SophiaG::from_params(
            vec![("p".into(), p_nan.clone())],
            SophiaConfig { lr: 0.1, betas: [0.5, 0.5], rho: 0.5, weight_decay: 0.0, ..Default::default() },
        )
        .unwrap();
        // Inject NaN at index 0 via Hessian and index 1 via gradient.
        let h_nan = Tensor::from_slice(&[f32::NAN, 1.0, 1.0, 1.0], [2, 2]).unwrap();
        opt_nan.update_hessian_gauss_newton(&make_grads(&[(&p_nan, h_nan)])).unwrap();
        let g_nan = Tensor::from_slice(&[1.0f32, f32::NAN, 1.0, 1.0], [2, 2]).unwrap();
        opt_nan.step(&make_grads(&[(&p_nan, g_nan)])).unwrap();
        let pv = p_nan.to_vec::<f32>().unwrap();
        assert!(pv[0].is_nan(), "transposed={transposed}: NaN Hessian swallowed: {pv:?}");
        assert!(pv[1].is_nan(), "transposed={transposed}: NaN grad swallowed: {pv:?}");
        assert!(
            pv[2].is_finite() && pv[3].is_finite(),
            "transposed={transposed}: valid entries corrupted: {pv:?}"
        );
    }

    // 4. Regression: CPU fast path must not overflow `(1 - beta2) * g^2` to +inf when `g = 1e20` and
    //    `beta2 = 0.999` (where the true update `1e37` is representable in `f32`), matching fallback.
    let p_fast = Tensor::ones([2, 2]).requires_grad_();
    let p_fall = Tensor::ones([2, 2]).transpose(0, 1).unwrap().requires_grad_();
    let cfg_big = SophiaConfig { betas: [0.9, 0.999], ..Default::default() };
    let mut opt_f = SophiaG::from_params(vec![("p".into(), p_fast.clone())], cfg_big).unwrap();
    let mut opt_g = SophiaG::from_params(vec![("p".into(), p_fall.clone())], cfg_big).unwrap();
    let big_g = Tensor::full([2, 2], 1e20f32);
    opt_f.update_hessian_gauss_newton(&make_grads(&[(&p_fast, big_g.clone())])).unwrap();
    opt_g.update_hessian_gauss_newton(&make_grads(&[(&p_fall, big_g)])).unwrap();
    let hf = opt_f.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    let hg = opt_g.state_for(0, 0).unwrap().1.to_vec::<f32>().unwrap();
    assert!(hf.iter().all(|v| v.is_finite() && *v > 0.0), "fast path overflowed Hessian: {hf:?}");
    assert!(hg.iter().all(|v| v.is_finite() && *v > 0.0), "fallback overflowed Hessian: {hg:?}");
}
