//! Exact analytical, numerical, contract-serde, and determinism tests for:
//! - [`DecayKind::Polynomial`] and [`DecayKind::OneCycle`] (Smith & Topin 2019), and
//! - Adaptive Gradient Clipping (`clip_grad_agc`).

use folo_core::{Tensor, rng::Rng};
use folo_nn::optim::{
    AGC_DEFAULT_CLIP_FACTOR, AGC_DEFAULT_EPS, GradAccumulator, Optimizer, ParamGroup, Sgd, StandardGroups,
};
use folo_nn::{
    Conv2d, Conv2dConfig, DEFAULT_ONE_CYCLE_DIV_FACTOR, DEFAULT_ONE_CYCLE_PCT_START, DecayKind, Forward,
    LrSchedule, Module, SchedContract,
};

#[test]
fn polynomial_schedule_identities_monotonicity_and_serde() {
    let lrf = 0.01;
    // power = 1.0 matches Linear identically for every iteration (with and without warmup).
    let lin = LrSchedule::new(DecayKind::Linear, 10, 13, lrf, 2.0, 0.1, 0.8).unwrap();
    let poly1 = LrSchedule::new(DecayKind::Polynomial, 10, 13, lrf, 2.0, 0.1, 0.8).unwrap();
    let group = ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([1]))]).lr(0.01).momentum(0.9);
    for t in 0..=130 {
        assert!((poly1.lf(t) - lin.lf(t)).abs() < 1e-15, "poly(p=1) vs linear at t={t}");
        assert!((poly1.lr_at(t, &group) - lin.lr_at(t, &group)).abs() < 1e-15);
    }

    // Analytical values and monotonicity for convex (p = 2.0) and concave (p = 0.5).
    for p in [0.5f64, 2.0, 3.5] {
        let s = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 101, 1, lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_poly_power(p)
            .unwrap();
        assert_eq!(s.lf(0), 1.0);
        let want_mid = lrf + (1.0 - lrf) * 0.5f64.powf(p);
        assert!((s.lf(50) - want_mid).abs() < 1e-15, "p={p} midpoint");
        assert!((s.lf(100) - lrf).abs() < 1e-15, "p={p} end");
        assert_eq!(s.lf(101), lrf, "p={p} past end");
        let mut prev = s.lf(0);
        for t in 1..=101 {
            let cur = s.lf(t);
            assert!(cur <= prev + 1e-15, "p={p} non-monotonic at t={t}");
            prev = cur;
        }
    }

    // Convex p = 2 decays faster than Linear at midpoint; concave p = 0.5 decays slower.
    let s_lo = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 101, 1, lrf, 0, 0.0, 0.8)
        .unwrap()
        .with_poly_power(0.5)
        .unwrap();
    let s_hi = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 101, 1, lrf, 0, 0.0, 0.8)
        .unwrap()
        .with_poly_power(2.0)
        .unwrap();
    assert!(s_hi.lf(50) < 0.505 && s_lo.lf(50) > 0.505);

    // T = 1 stays at 1.0; T = 2 hits 1.0 then lrf; extreme lrf = 1e17 / f64::MAX preserves exact lf(0) = 1.0.
    let one = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 1, 1, lrf, 0, 0.0, 0.8)
        .unwrap()
        .with_poly_power(2.0)
        .unwrap();
    assert_eq!(one.lf(0), 1.0);
    let two = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 2, 1, lrf, 0, 0.0, 0.8)
        .unwrap()
        .with_poly_power(2.0)
        .unwrap();
    assert_eq!(two.lf(0), 1.0);
    assert_eq!(two.lf(1), lrf);
    for extreme_lrf in [0.0f64, 1.0, 2.0, 1e17, f64::MAX] {
        let ex = LrSchedule::with_warmup_iters(DecayKind::Polynomial, 10, 1, extreme_lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_poly_power(2.0)
            .unwrap();
        assert_eq!(ex.lf(0), 1.0, "lrf={extreme_lrf} must give exact lf(0)=1.0");
        assert_eq!(ex.lf(9), extreme_lrf, "lrf={extreme_lrf} must give exact lf(T-1)=lrf");
    }

    // Contract JSON round-trip and validation errors.
    let json = serde_json::to_string(s_hi.contract()).unwrap();
    assert!(json.contains(r#""type":"polynomial""#) && json.contains(r#""poly_power":2.0"#));
    let parsed: SchedContract = serde_json::from_str(&json).unwrap();
    assert_eq!(LrSchedule::from_contract(parsed).unwrap(), s_hi);

    // Legacy Linear omits poly_power / one_cycle_* keys.
    let lin_json = serde_json::to_string(lin.contract()).unwrap();
    assert!(!lin_json.contains("poly_power") && !lin_json.contains("one_cycle"));

    assert!(poly1.clone().with_poly_power(0.0).is_err());
    assert!(poly1.clone().with_poly_power(-1.0).is_err());
    assert!(poly1.clone().with_poly_power(f64::NAN).is_err());
    assert!(poly1.clone().with_poly_power(f64::INFINITY).is_err());
    assert!(lin.with_poly_power(2.0).is_err());
}

#[test]
fn one_cycle_schedule_phases_inverse_momentum_and_serde() {
    let lrf = 1e-4;
    // T = 101 so x = t / 100 is exact; default pct_start = 0.3 -> peak at t = 30, div_factor = 25.0.
    let s = LrSchedule::with_warmup_iters(DecayKind::OneCycle, 101, 1, lrf, 0, 0.0, 0.85).unwrap();
    let group = ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([1]))]).lr(0.1).momentum(0.95);

    // Exact endpoints and peak:
    assert!((s.lf(0) - 1.0 / DEFAULT_ONE_CYCLE_DIV_FACTOR).abs() < 1e-15);
    assert!((s.lf(30) - 1.0).abs() < 1e-15, "peak at x = pct_start = 0.3");
    assert!((s.lf(100) - lrf).abs() < 1e-15, "final lf == lrf");
    assert_eq!(s.lf(101), s.lf(100), "clamped past end");

    // Strictly increasing on [0, 30], strictly decreasing on [30, 100]:
    for t in 1..=30 {
        assert!(s.lf(t) > s.lf(t - 1), "must rise on [0, 30] at t={t}");
    }
    for t in 31..=100 {
        assert!(s.lf(t) < s.lf(t - 1), "must anneal on [30, 100] at t={t}");
    }

    // Halfway up (t = 15, u = 0.5 -> cos_up = 0.5) and halfway down (t = 65, v = 0.5 -> cos_down = 0.5):
    let init_f = 1.0 / DEFAULT_ONE_CYCLE_DIV_FACTOR;
    assert!((s.lf(15) - 0.5 * (init_f + 1.0)).abs() < 1e-14);
    assert!((s.lf(65) - 0.5 * (1.0 + lrf)).abs() < 1e-14);

    // Inverse cosine momentum cycling: 0.95 at t=0 -> 0.85 at t=30 -> 0.95 at t=100.
    assert!((s.momentum_at(0, &group) - 0.95).abs() < 1e-15);
    assert!((s.momentum_at(15, &group) - 0.90).abs() < 1e-14);
    assert!((s.momentum_at(30, &group) - 0.85).abs() < 1e-15);
    assert!((s.momentum_at(65, &group) - 0.90).abs() < 1e-14);
    assert!((s.momentum_at(100, &group) - 0.95).abs() < 1e-15);
    assert_eq!(s.momentum_at(101, &group), s.momentum_at(100, &group));

    // Custom pct_start & div_factor, T = 1 and T = 2 edge cases, and JSON contract round-trip.
    let custom = s.clone().with_one_cycle(0.25, 10.0).unwrap();
    assert!((custom.lf(0) - 0.1).abs() < 1e-15);
    assert!((custom.lf(25) - 1.0).abs() < 1e-15);
    assert!((custom.lf(100) - lrf).abs() < 1e-15);
    let json = serde_json::to_string(custom.contract()).unwrap();
    assert!(
        json.contains(r#""type":"one_cycle""#)
            && json.contains(r#""one_cycle_pct_start":0.25"#)
            && json.contains(r#""one_cycle_div_factor":10.0"#)
    );
    let parsed: SchedContract = serde_json::from_str(&json).unwrap();
    assert_eq!(LrSchedule::from_contract(parsed).unwrap(), custom);

    let one = LrSchedule::with_warmup_iters(DecayKind::OneCycle, 1, 1, lrf, 0, 0.0, 0.85).unwrap();
    assert_eq!(one.lf(0), 1.0);
    assert_eq!(one.momentum_at(0, &group), 0.95);
    let two = LrSchedule::with_warmup_iters(DecayKind::OneCycle, 2, 1, lrf, 0, 0.0, 0.85).unwrap();
    assert!((two.lf(0) - init_f).abs() < 1e-15);
    assert!((two.lf(1) - lrf).abs() < 1e-15);

    // Validation rejects out-of-range parameters, non-zero warmup, invalid lrf, and cross-kind fields.
    assert!(LrSchedule::with_warmup_iters(DecayKind::OneCycle, 100, 1, lrf, 10, 0.0, 0.85).is_err());
    assert!(LrSchedule::new(DecayKind::OneCycle, 10, 10, lrf, 1.0, 0.0, 0.85).is_err());
    assert!(LrSchedule::with_warmup_iters(DecayKind::OneCycle, 100, 1, -0.01, 0, 0.0, 0.85).is_err());
    assert!(LrSchedule::with_warmup_iters(DecayKind::OneCycle, 100, 1, 1.5, 0, 0.0, 0.85).is_err());
    assert!(s.clone().with_one_cycle(0.0, 10.0).is_err());
    assert!(s.clone().with_one_cycle(1.0, 10.0).is_err());
    assert!(s.clone().with_one_cycle(f64::NAN, 10.0).is_err());
    assert!(s.clone().with_one_cycle(DEFAULT_ONE_CYCLE_PCT_START, 0.5).is_err());
    assert!(s.clone().with_one_cycle(DEFAULT_ONE_CYCLE_PCT_START, f64::INFINITY).is_err());
    assert!(s.with_poly_power(2.0).is_err());
    let cos = LrSchedule::new(DecayKind::Cosine, 10, 5, 0.01, 0.0, 0.0, 0.8).unwrap();
    assert!(cos.with_one_cycle(0.3, 25.0).is_err());
}

#[test]
fn agc_and_one_cycle_training_loop_determinism() {
    let run_once = || -> Vec<f32> {
        let mut rng = Rng::seed(2026);
        let conv = Conv2d::new(Conv2dConfig::new(3, 6, 3).padding(1), &mut rng);
        let groups = StandardGroups::from_module(&conv).into_groups(0.05, 0.95, 1e-4);
        let mut opt = Sgd::new(groups).unwrap();
        let mut sched = LrSchedule::with_warmup_iters(DecayKind::OneCycle, 6, 1, 1e-3, 0, 0.0, 0.85).unwrap();
        let mut acc = GradAccumulator::new(&conv.parameters());

        for _ in 0..6 {
            let x = Tensor::rand_uniform([2, 3, 8, 8], -1.0, 1.0, &mut rng);
            let loss = conv.forward(&x).unwrap().square().unwrap().mean_all().unwrap();
            acc.add(&loss.backward().unwrap()).unwrap();
            sched.apply(&mut opt);
            let stats = acc.clip_grad_agc(AGC_DEFAULT_CLIP_FACTOR, AGC_DEFAULT_EPS, true).unwrap();
            assert_eq!(stats.total_units, 6);
            assert!(stats.max_ratio.is_finite());
            opt.step(&acc).unwrap();
            sched.advance();
            acc.zero();
        }
        conv.weight.to_vec::<f32>().unwrap()
    };

    let a = run_once();
    let b = run_once();
    assert_eq!(a, b, "AGC + OneCycle training loop must be bitwise deterministic");
    assert!(a.iter().all(|v| v.is_finite()));
}
