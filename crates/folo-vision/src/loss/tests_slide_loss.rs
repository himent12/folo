//! Unit tests for SlideLoss and EMA-SlideLoss (`slide_weight`, `slide_bce_loss`,
//! `slide_reweight_loss`, `EmaSlideLoss`): smooth vs standard weighting curve verification,
//! `f64` central finite-difference checks, tensor `grad_check_with` across all reduction modes,
//! extreme-logit cancellation-free stability, EMA tracking & warmup behavior, empty/non-finite
//! resilience, and invalid-config/target rejection.

use super::terms::{
    EmaSlideLoss, LossReduction, SlideLossConfig, SlideWeightKind, VflConfig, slide_bce_elem_f64,
    slide_bce_loss, slide_reweight_loss, slide_weight, slide_weight_f64, varifocal_loss,
};
use folo_core::Tensor;
use folo_core::testing::{GradCheck, grad_check_with};

/// Verify SlideLoss weight curves (Smooth & Standard): bounds `[1, e]`, peak at `mu`,
/// C¹ continuity of `Smooth`, exact piecewise values of `Standard`, and invalid-input NaN guards.
#[test]
fn slide_weight_curve_properties() {
    let mu = 0.6f64;
    let delta = 0.1f64;
    let peak = (1.0 - mu).exp();

    for kind in [SlideWeightKind::Smooth, SlideWeightKind::Standard] {
        // Easy negatives (<= mu - delta) get weight 1.0.
        assert_eq!(slide_weight_f64(0.0, mu, delta, kind), 1.0);
        assert_eq!(slide_weight_f64(0.45, mu, delta, kind), 1.0);
        assert_eq!(slide_weight_f64(0.50, mu, delta, kind), 1.0);
        // Peak at hard boundary x == mu equals exp(1 - mu).
        assert!((slide_weight_f64(mu, mu, delta, kind) - peak).abs() < 1e-15);
        // Easy positive x == 1.0 gets weight 1.0.
        assert!((slide_weight_f64(1.0, mu, delta, kind) - 1.0).abs() < 1e-15);
        // Decay for x > mu matches exp(1 - x).
        for x in [0.65, 0.8, 0.95] {
            assert!((slide_weight_f64(x, mu, delta, kind) - (1.0 - x).exp()).abs() < 1e-15);
        }
        // Bounded in [1, e] over the full grid.
        for k in 0..=100 {
            let x = k as f64 / 100.0;
            let w = slide_weight_f64(x, mu, delta, kind);
            assert!(w.is_finite() && (1.0..=std::f64::consts::E + 1e-12).contains(&w));
            assert!((slide_weight(x as f32, mu as f32, delta as f32, kind) as f64 - w).abs() < 1e-6);
        }
    }

    // Standard is flat at exp(1 - mu) on (mu - delta, mu).
    assert!((slide_weight_f64(0.55, mu, delta, SlideWeightKind::Standard) - peak).abs() < 1e-15);

    // Smooth is strictly increasing across (mu - delta, mu) and continuous at both endpoints.
    let w_lo = slide_weight_f64(0.50 + 1e-9, mu, delta, SlideWeightKind::Smooth);
    let w_mid = slide_weight_f64(0.55, mu, delta, SlideWeightKind::Smooth);
    let w_hi = slide_weight_f64(0.60 - 1e-9, mu, delta, SlideWeightKind::Smooth);
    assert!((w_lo - 1.0).abs() < 1e-6);
    assert!((w_mid - 0.5 * (1.0 + peak)).abs() < 1e-12, "Hermite midpoint must equal 0.5*(1+peak)");
    assert!((w_hi - peak).abs() < 1e-6);
    assert!(w_lo < w_mid && w_mid < w_hi);

    // Invalid inputs return NaN.
    assert!(slide_weight_f64(-0.1, mu, delta, SlideWeightKind::Smooth).is_nan());
    assert!(slide_weight_f64(1.1, mu, delta, SlideWeightKind::Smooth).is_nan());
    assert!(slide_weight_f64(0.5, -0.1, delta, SlideWeightKind::Smooth).is_nan());
    assert!(slide_weight_f64(0.5, mu, 0.0, SlideWeightKind::Smooth).is_nan());
    assert!(slide_weight_f64(f64::NAN, mu, delta, SlideWeightKind::Smooth).is_nan());
}

/// `f64` central finite-difference checks (including extreme `±50` / `±500` logits) and tensor
/// `grad_check_with` for `slide_bce_loss` and `slide_reweight_loss`.
#[test]
fn slide_loss_fd_and_tensor_grad_check() {
    let logits = [-50.0f64, -10.0, -2.5, -0.3, 0.0, 0.7, 3.1, 10.0, 50.0];
    let targets = [0.0f64, 0.2, 0.5, 0.85, 1.0];
    let weights = [1.0f64, 1.35, 2.1];

    for &w in &weights {
        for &x in &logits {
            for &q in &targets {
                let (l, g) = slide_bce_elem_f64(x, q, w);
                assert!(l.is_finite() && l >= 0.0);
                assert!(g.is_finite());
                let h = 1e-6f64;
                let fd = (slide_bce_elem_f64(x + h, q, w).0 - slide_bce_elem_f64(x - h, q, w).0) / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(g.abs());
                assert!((g - fd).abs() <= tol, "w={w} x={x} q={q}: ana={g} fd={fd}");
            }
        }
    }

    // Extreme ±500 and ±1e38 logits stay finite and cancellation-free.
    for x in [-1e38f64, -500.0, 500.0, 1e38] {
        for q in [0.0, 0.5, 1.0] {
            let (l, g) = slide_bce_elem_f64(x, q, 1.8);
            assert!(l.is_finite() && l >= 0.0);
            assert!(g.is_finite());
        }
    }

    let x = Tensor::from_slice(&[-2.4f32, -0.5, 0.2, 1.3, 2.7, -1.1], [2, 3]).unwrap();
    let t = Tensor::from_slice(&[0.0f32, 0.0, 0.55, 0.72, 1.0, 0.48], [2, 3]).unwrap();
    let ious = Tensor::from_slice(&[0.0f32, 0.25, 0.55, 0.62, 0.92, 0.48], [2, 3]).unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 6, seed: 19 };

    for kind in [SlideWeightKind::Smooth, SlideWeightKind::Standard] {
        for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
            let cfg = SlideLossConfig { delta: 0.1, weight_kind: kind, reduction };
            for mu in [None, Some(0.6f32)] {
                grad_check_with(
                    |ins| {
                        let out = slide_bce_loss(&ins[0], &t, &ious, mu, &cfg)?;
                        if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
                    },
                    std::slice::from_ref(&x),
                    &gc,
                )
                .unwrap();
            }
        }
    }

    // Composable slide_reweight_loss wrapping standalone varifocal_loss passes grad_check_with.
    let vfl_none = VflConfig { alpha: 0.75, gamma: 2.0, reduction: LossReduction::None };
    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let cfg = SlideLossConfig { delta: 0.15, weight_kind: SlideWeightKind::Smooth, reduction };
        grad_check_with(
            |ins| {
                let base = varifocal_loss(&ins[0], &t, &vfl_none)?;
                let out = slide_reweight_loss(&base, &ious, 0.58, &cfg)?;
                if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
            },
            std::slice::from_ref(&x),
            &gc,
        )
        .unwrap();
    }
}

/// Verify `EmaSlideLoss` state tracking (lazy init, explicit init, warmup tau, constant decay,
/// empty/all-negative batch preservation, non-finite IoU protection, eval mode, reset, and serde).
#[test]
fn ema_slide_loss_tracking_and_stability() {
    let cfg = SlideLossConfig::default();

    // 1. Lazy first-batch init + constant decay.
    let mut ema = EmaSlideLoss::new(cfg, 0.8, 0.0).unwrap();
    assert!(!ema.is_initialized());
    assert_eq!(ema.updates(), 0);

    let mu1 = ema.update_mu(&[0.0, 0.4, 0.8], Some(&[false, true, true])).unwrap();
    assert!(ema.is_initialized());
    assert_eq!(ema.updates(), 1);
    assert!((mu1 - 0.6).abs() < 1e-6, "first batch seeds directly to positive mean 0.6, got {mu1}");

    let mu2 = ema.update_mu(&[0.9, 0.9], None).unwrap();
    let want_mu2 = 0.8 * 0.6 + 0.2 * 0.9;
    assert_eq!(ema.updates(), 2);
    assert!((mu2 as f64 - want_mu2).abs() < 1e-6, "got {mu2} vs {want_mu2}");

    // Empty mask or all-false mask leaves EMA state unchanged.
    let mu_same = ema.update_mu(&[0.1, 0.2], Some(&[false, false])).unwrap();
    assert_eq!(ema.updates(), 2);
    assert_eq!(mu_same, mu2);

    // Non-finite IoU in selected entries leaves EMA state uncorrupted.
    let mu_nf = ema.update_mu(&[f32::NAN, 0.5], Some(&[true, true])).unwrap();
    assert_eq!(ema.updates(), 2);
    assert_eq!(mu_nf, mu2);

    // Reset returns to uninitialized state.
    ema.reset();
    assert!(!ema.is_initialized());
    assert_eq!(ema.updates(), 0);

    // 2. Explicit initial_mu + exponential warmup tau.
    let mut ema_w = EmaSlideLoss::with_initial_mu(cfg, 0.99, 10.0, 0.5).unwrap();
    assert!(ema_w.is_initialized());
    let d1 = ema_w.effective_decay();
    let expected_d1 = (0.99f32 as f64) * (1.0 - (-1.0f64 / (10.0f32 as f64)).exp());
    assert!((d1 - expected_d1).abs() < 1e-12);

    let logits = Tensor::from_slice(&[0.5f32, -0.5, 1.2, -1.0], [4]).unwrap().requires_grad_();
    let targets = Tensor::from_slice(&[1.0f32, 0.0, 0.8, 0.0], [4]).unwrap();
    let ious = Tensor::from_slice(&[0.8f32, 0.0, 0.6, 0.0], [4]).unwrap();
    let loss = ema_w.compute_bce(&logits, &targets, &ious).unwrap();
    assert_eq!(ema_w.updates(), 1);
    let batch_mu = 0.5 * (0.8f32 as f64 + 0.6f32 as f64);
    let want_after = d1 * 0.5 + (1.0 - d1) * batch_mu;
    assert!((ema_w.mu_f64() - want_after).abs() < 1e-12);
    assert!(loss.item::<f32>().unwrap().is_finite());

    // eval_bce uses current EMA mu without mutating updates or mu.
    let mu_before_eval = ema_w.mu_f64();
    let eval_loss = ema_w.eval_bce(&logits, &targets, &ious).unwrap();
    assert_eq!(ema_w.updates(), 1);
    assert_eq!(ema_w.mu_f64(), mu_before_eval);
    assert!((eval_loss.item::<f32>().unwrap() - loss.item::<f32>().unwrap()).abs() < 1e-6);

    // Serde round-trip preserves full EMA state.
    let json = serde_json::to_string(&ema_w).unwrap();
    assert_eq!(serde_json::from_str::<EmaSlideLoss>(&json).unwrap(), ema_w);
}

/// Edge cases: empty tensors, zero-positive batches, NaN/Inf propagation, and validation errors.
#[test]
fn slide_loss_edge_cases_and_validation() {
    let cfg = SlideLossConfig::default();

    // Empty tensors work cleanly for every reduction.
    let empty = Tensor::zeros([0]).requires_grad_();
    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let c = SlideLossConfig { reduction, ..cfg };
        let l = slide_bce_loss(&empty, &empty, &empty, None, &c).unwrap();
        let lr = slide_reweight_loss(&empty, &empty, 0.5, &c).unwrap();
        if reduction == LossReduction::None {
            assert_eq!(l.dims(), &[0]);
            assert_eq!(lr.dims(), &[0]);
        } else {
            assert_eq!(l.item::<f32>().unwrap(), 0.0);
            assert_eq!(lr.item::<f32>().unwrap(), 0.0);
        }
    }

    // All-negative batch with dynamic mu == None falls back to unweighted BCE.
    let x = Tensor::from_slice(&[-1.0f32, 0.5, 2.0], [3]).unwrap();
    let zeros = Tensor::zeros([3]);
    let l_dyn =
        slide_bce_loss(&x, &zeros, &zeros, None, &SlideLossConfig { reduction: LossReduction::None, ..cfg })
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
    let l_unw = slide_bce_loss(
        &x,
        &zeros,
        &zeros,
        Some(0.5),
        &SlideLossConfig { reduction: LossReduction::None, ..cfg },
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    for (a, b) in l_dyn.iter().zip(&l_unw) {
        assert!((a - b).abs() < 1e-6);
    }

    // NaN / ±Inf logits propagate NaN cleanly.
    let t1 = Tensor::from_slice(&[0.5f32], [1]).unwrap();
    let i1 = Tensor::from_slice(&[0.5f32], [1]).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bx = Tensor::from_slice(&[bad], [1]).unwrap().requires_grad_();
        let l = slide_bce_loss(&bx, &t1, &i1, Some(0.5), &cfg).unwrap();
        assert!(l.item::<f32>().unwrap().is_nan());
        let g = l.backward().unwrap().get(&bx).unwrap().to_vec::<f32>().unwrap();
        assert!(g[0].is_nan());
    }

    // Validation errors.
    let bad_cfg = SlideLossConfig { delta: 0.0, ..cfg };
    assert!(slide_bce_loss(&x, &zeros, &zeros, Some(0.5), &bad_cfg).is_err());
    assert!(slide_bce_loss(&x, &zeros, &zeros, Some(1.2), &cfg).is_err());
    assert!(slide_bce_loss(&x, &zeros, &zeros, Some(f32::NAN), &cfg).is_err());
    let oob_t = Tensor::from_slice(&[-0.1f32, 0.5, 0.5], [3]).unwrap();
    assert!(slide_bce_loss(&x, &oob_t, &zeros, Some(0.5), &cfg).is_err());
    let oob_i = Tensor::from_slice(&[0.2f32, 1.5, 0.5], [3]).unwrap();
    assert!(slide_bce_loss(&x, &zeros, &oob_i, Some(0.5), &cfg).is_err());
    assert!(EmaSlideLoss::new(cfg, 1.0, 0.0).is_err());
    assert!(EmaSlideLoss::new(cfg, -0.1, 0.0).is_err());
    assert!(EmaSlideLoss::new(cfg, 0.9, -1.0).is_err());
    assert!(EmaSlideLoss::with_initial_mu(cfg, 0.9, 0.0, 1.5).is_err());
}
