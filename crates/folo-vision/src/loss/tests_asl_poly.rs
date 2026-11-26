//! Unit tests for Asymmetric Loss (ASL), Poly-1 BCE, and Poly-1 Focal Loss:
//! `f64` central finite-difference gradient checks (including extreme `±50` logits, margin-shift clamp
//! boundary, fractional exponents), tensor `grad_check`, algebraic reduction identities, zero/all-negative/
//! all-positive targets, non-finite input safety, fused vs composite parity, and `detection_loss` / config
//! validation.

use folo_core::testing::{GradCheck, grad_check_with};

use super::terms::{asl_elem_f64, loss_impl, poly1_bce_elem_f64, poly1_focal_elem_f64};
use super::*;
use crate::decode::make_anchors;

/// `f64` central finite-difference gradient check for ASL across standard, fractional, extreme (`±50`),
/// and margin-boundary logits and binary/soft targets.
#[test]
fn asl_f64_central_fd_and_margin_boundary() {
    let params = [
        (0.0f64, 4.0f64, 0.05f64), // paper default
        (1.0, 4.0, 0.05),          // positive + negative focusing
        (0.5, 0.5, 0.1),           // fractional exponents (< 1)
        (0.0, 0.0, 0.0),           // exact BCE reduction
        (2.0, 3.0, 0.0),           // no margin shift
        (0.0, 2.0, 0.25),          // wide margin shift
    ];
    let logits = [-50.0f64, -12.0, -3.5, -1.0, 0.0, 0.8, 3.2, 12.0, 50.0];
    let targets = [0.0f64, 0.25, 0.5, 0.85, 1.0];

    for &(gp, gn, clip) in &params {
        for &x in &logits {
            for &t in &targets {
                let (loss, grad) = asl_elem_f64(x, t, gp, gn, clip);
                assert!(loss.is_finite() && loss >= 0.0, "asl ({gp},{gn},{clip}) x={x} t={t}: loss={loss}");
                assert!(grad.is_finite(), "asl ({gp},{gn},{clip}) x={x} t={t}: grad={grad}");
                let eps = 1e-6f64;
                let fp = asl_elem_f64(x + eps, t, gp, gn, clip).0;
                let fm = asl_elem_f64(x - eps, t, gp, gn, clip).0;
                let fd = (fp - fm) / (2.0 * eps);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad.abs());
                assert!(
                    (grad - fd).abs() <= tol,
                    "asl ({gp},{gn},{clip}) x={x} t={t}: ana={grad:e} fd={fd:e}"
                );
            }
        }
    }

    // Margin shift clamp boundary: at x_m = ln(m / (1 - m)), p == m (up to 1 ULP of ln/exp).
    // At exact p == m (e.g. x = 0, m = 0.5) and below x_m, negative loss & grad are exact 0.0; above x_m
    // with gamma_neg > 0, loss & grad grow smoothly (C^1 continuous, including fractional gamma_neg = 0.5).
    assert_eq!(asl_elem_f64(0.0, 0.0, 0.0, 4.0, 0.5), (0.0, 0.0));
    assert_eq!(asl_elem_f64(0.0, 0.0, 0.0, 0.0, 0.5), (0.0, 0.0));
    for &(gn, clip) in &[(4.0f64, 0.05f64), (0.5, 0.05), (1.0, 0.2), (0.0, 0.1)] {
        let xm = (clip / (1.0 - clip)).ln();
        for &dx in &[-1.0f64, -1e-3, -1e-9] {
            let (l, g) = asl_elem_f64(xm + dx, 0.0, 0.0, gn, clip);
            assert_eq!((l, g), (0.0, 0.0), "gn={gn} clip={clip} dx={dx}: expected exact 0 in clamped zone");
        }
        // Slightly above boundary: positive loss, positive grad, matches central FD away from boundary.
        let (l_hi, g_hi) = asl_elem_f64(xm + 1e-3, 0.0, 0.0, gn, clip);
        assert!(l_hi > 0.0 && l_hi.is_finite() && g_hi > 0.0 && g_hi.is_finite());
        let eps = 1e-6;
        let fd = (asl_elem_f64(xm + 1e-3 + eps, 0.0, 0.0, gn, clip).0
            - asl_elem_f64(xm + 1e-3 - eps, 0.0, 0.0, gn, clip).0)
            / (2.0 * eps);
        let tol = 1e-8 + 1e-5 * fd.abs().max(g_hi.abs());
        assert!((g_hi - fd).abs() <= tol, "above margin gn={gn} clip={clip}: ana={g_hi:e} fd={fd:e}");

        // For gn > 0, right-hand derivative vanishes continuously at the boundary.
        if gn > 0.0 {
            let (_, g_near) = asl_elem_f64(xm + 1e-10, 0.0, 0.0, gn, clip);
            assert!((0.0..1e-4).contains(&g_near), "C^1 margin check gn={gn}: g_near={g_near:e}");
        }
    }
}

/// `f64` central finite-difference gradient check for Poly-1 BCE and Poly-1 Focal across standard,
/// fractional, negative-epsilon (`-1.0`), and extreme (`±50`) logits.
#[test]
fn polyloss_f64_central_fd_grad_check() {
    let logits = [-50.0f64, -15.0, -2.5, -0.3, 0.0, 0.4, 2.7, 15.0, 50.0];
    let targets = [0.0f64, 0.2, 0.5, 0.9, 1.0];
    let epsilons = [-1.0f64, -0.5, 0.0, 1.0, 2.5];
    let gammas = [0.0f64, 0.5, 1.0, 2.0, 3.5];

    for &eps1 in &epsilons {
        for &x in &logits {
            for &t in &targets {
                let (loss, grad) = poly1_bce_elem_f64(x, t, eps1);
                assert!(loss.is_finite() && loss >= -1e-15, "poly1_bce eps={eps1} x={x} t={t}: {loss}");
                assert!(grad.is_finite());
                let h = 1e-6f64;
                let fd =
                    (poly1_bce_elem_f64(x + h, t, eps1).0 - poly1_bce_elem_f64(x - h, t, eps1).0) / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad.abs());
                assert!((grad - fd).abs() <= tol, "poly1_bce eps={eps1} x={x} t={t}: ana={grad:e} fd={fd:e}");
            }
        }
    }

    for &gamma in &gammas {
        for &eps1 in &epsilons {
            for &x in &logits {
                for &t in &targets {
                    let (loss, grad) = poly1_focal_elem_f64(x, t, gamma, eps1);
                    assert!(
                        loss.is_finite() && loss >= -1e-15,
                        "poly1_focal g={gamma} eps={eps1} x={x} t={t}: {loss}"
                    );
                    assert!(grad.is_finite());
                    let h = 1e-6f64;
                    let fd = (poly1_focal_elem_f64(x + h, t, gamma, eps1).0
                        - poly1_focal_elem_f64(x - h, t, gamma, eps1).0)
                        / (2.0 * h);
                    let tol = 1e-8 + 1e-5 * fd.abs().max(grad.abs());
                    assert!(
                        (grad - fd).abs() <= tol,
                        "poly1_focal g={gamma} eps={eps1} x={x} t={t}: ana={grad:e} fd={fd:e}"
                    );
                }
            }
        }
    }
}

/// Tensor-level `grad_check_with` and exact algebraic reduction identities for `asymmetric_loss`,
/// `poly1_bce_loss`, `poly1_focal_loss`, and `poly1_focal_loss_with_alpha`.
#[test]
fn standalone_tensor_grad_checks_and_identities() {
    let logits = Tensor::from_slice(&[-3.5f32, -1.2, -0.1, 0.6, 2.1, 4.3, -2.9, 1.4], [2, 4]).unwrap();
    let targets = Tensor::from_slice(&[0.0f32, 0.0, 0.3, 0.75, 1.0, 1.0, 0.0, 0.5], [2, 4]).unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 8, seed: 42 };

    grad_check_with(
        |inps| asymmetric_loss(&inps[0], &targets, 0.0, 4.0, 0.05)?.sum_all(),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();
    grad_check_with(
        |inps| asymmetric_loss(&inps[0], &targets, 1.0, 2.5, 0.1)?.sum_all(),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();
    grad_check_with(
        |inps| poly1_bce_loss(&inps[0], &targets, 1.0)?.sum_all(),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();
    grad_check_with(
        |inps| poly1_focal_loss(&inps[0], &targets, 2.0, 1.0)?.sum_all(),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();
    grad_check_with(
        |inps| poly1_focal_loss_with_alpha(&inps[0], &targets, 2.0, 1.0, 0.25)?.sum_all(),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();

    // Algebraic identities:
    // 1. ASL(0, 0, 0) == Poly1BCE(0) == BCE
    let bce =
        logits.softplus().unwrap().sub(&logits.mul(&targets).unwrap()).unwrap().to_vec::<f32>().unwrap();
    let asl_bce = asymmetric_loss(&logits, &targets, 0.0, 0.0, 0.0).unwrap().to_vec::<f32>().unwrap();
    let poly_bce0 = poly1_bce_loss(&logits, &targets, 0.0).unwrap().to_vec::<f32>().unwrap();
    for i in 0..bce.len() {
        assert!((asl_bce[i] - bce[i]).abs() < 1e-6, "ASL(0,0,0) != BCE at {i}");
        assert!((poly_bce0[i] - bce[i]).abs() < 1e-6, "Poly1BCE(0) != BCE at {i}");
    }
    // 2. Poly1Focal(gamma=0, eps) == Poly1BCE(eps)
    let p_bce = poly1_bce_loss(&logits, &targets, 1.5).unwrap().to_vec::<f32>().unwrap();
    let p_foc0 = poly1_focal_loss(&logits, &targets, 0.0, 1.5).unwrap().to_vec::<f32>().unwrap();
    for i in 0..p_bce.len() {
        assert!((p_bce[i] - p_foc0[i]).abs() < 1e-6, "Poly1Focal(0, eps) != Poly1BCE(eps) at {i}");
    }
    // 3. Poly1Focal(gamma, eps=0) == ASL(gamma, gamma, clip=0)
    let p_foc_e0 = poly1_focal_loss(&logits, &targets, 2.0, 0.0).unwrap().to_vec::<f32>().unwrap();
    let asl_sym = asymmetric_loss(&logits, &targets, 2.0, 2.0, 0.0).unwrap().to_vec::<f32>().unwrap();
    for i in 0..p_foc_e0.len() {
        assert!((p_foc_e0[i] - asl_sym[i]).abs() < 1e-6, "Poly1Focal(g,0) != ASL(g,g,0) at {i}");
    }
    // 4. Poly1FocalWithAlpha(alpha=0.5) == 0.5 * Poly1Focal
    let p_foc = poly1_focal_loss(&logits, &targets, 2.0, 1.0).unwrap().to_vec::<f32>().unwrap();
    let p_half =
        poly1_focal_loss_with_alpha(&logits, &targets, 2.0, 1.0, 0.5).unwrap().to_vec::<f32>().unwrap();
    for i in 0..p_foc.len() {
        assert!((p_half[i] - 0.5 * p_foc[i]).abs() < 1e-6, "alpha=0.5 identity failed at {i}");
    }
}

/// Edge cases: extreme `±50` / `±500` logits, all-negative / all-positive / zero targets, empty tensors,
/// non-finite (`NaN` / `±Inf`) poison propagation, and invalid hyperparameter / shape rejections.
#[test]
fn edge_cases_extreme_logits_and_non_finite_safety() {
    // 1. Extreme ±50 and ±500 logits with all-negative, all-positive, and mixed targets stay finite.
    let extreme_x = Tensor::from_slice(&[-500.0f32, -50.0, 0.0, 50.0, 500.0], [5]).unwrap().requires_grad_();
    let all_neg = Tensor::zeros([5]);
    let all_pos = Tensor::ones([5]);
    let mixed = Tensor::from_slice(&[0.0f32, 1.0, 0.5, 1.0, 0.0], [5]).unwrap();

    for t in [&all_neg, &all_pos, &mixed] {
        for (name, l) in [
            ("asl", asymmetric_loss(&extreme_x, t, 1.0, 4.0, 0.05).unwrap()),
            ("poly1_bce", poly1_bce_loss(&extreme_x, t, 1.0).unwrap()),
            ("poly1_focal", poly1_focal_loss(&extreme_x, t, 2.0, 1.0).unwrap()),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv.iter().all(|v| v.is_finite() && *v >= 0.0), "{name}: non-finite or neg loss {lv:?}");
            let gv =
                l.sum_all().unwrap().backward().unwrap().get(&extreme_x).unwrap().to_vec::<f32>().unwrap();
            assert!(gv.iter().all(|v| v.is_finite()), "{name}: non-finite grad {gv:?}");
        }
    }

    // 2. Non-finite inputs (NaN, +Inf, -Inf) in logits or targets poison both forward and backward.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bx = Tensor::from_slice(&[bad, 0.5], [2]).unwrap().requires_grad_();
        let ok_t = Tensor::from_slice(&[1.0f32, 0.0], [2]).unwrap();
        for l in [
            asymmetric_loss(&bx, &ok_t, 0.0, 4.0, 0.05).unwrap(),
            poly1_bce_loss(&bx, &ok_t, 1.0).unwrap(),
            poly1_focal_loss(&bx, &ok_t, 2.0, 1.0).unwrap(),
            poly1_focal_loss_with_alpha(&bx, &ok_t, 2.0, 1.0, 0.25).unwrap(),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv[0].is_nan() && lv[1].is_finite(), "bad logit {bad}: {lv:?}");
            let gv = l.sum_all().unwrap().backward().unwrap().get(&bx).unwrap().to_vec::<f32>().unwrap();
            assert!(gv[0].is_nan() && gv[1].is_finite(), "bad logit {bad} grad: {gv:?}");
        }

        let ok_x = Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap().requires_grad_();
        let bt = Tensor::from_slice(&[bad, 1.0], [2]).unwrap();
        for l in [
            asymmetric_loss(&ok_x, &bt, 0.0, 4.0, 0.05).unwrap(),
            poly1_bce_loss(&ok_x, &bt, 1.0).unwrap(),
            poly1_focal_loss(&ok_x, &bt, 2.0, 1.0).unwrap(),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv[0].is_nan() && lv[1].is_finite(), "bad target {bad}: {lv:?}");
            let gv = l.sum_all().unwrap().backward().unwrap().get(&ok_x).unwrap().to_vec::<f32>().unwrap();
            assert!(gv[0].is_nan() && gv[1].is_finite(), "bad target {bad} grad: {gv:?}");
        }
    }

    // 3. Cancellation-free relative precision at epsilon = -1.0 for large logits (x = ±40):
    // For x = 40, t = 1: r = sigmoid(-40) ≈ 4.248354e-18, true loss = -ln(1 - r) - r = r^2 / 2 + r^3 / 3 + ...,
    // true grad = -r^2 (both normal non-zero f32 values ~9.02e-36 and ~-1.80e-35).
    let r40 = (-40.0f64).exp() / (1.0 + (-40.0f64).exp());
    let ref_bce_loss = 0.5 * r40 * r40 * (1.0 + (2.0 / 3.0) * r40);
    let ref_bce_grad = -(r40 * r40);
    let (got_l, got_g) = poly1_bce_elem_f64(40.0, 1.0, -1.0);
    assert!(
        (got_l - ref_bce_loss).abs() <= 1e-12 * ref_bce_loss,
        "poly1_bce eps=-1 x=40 loss: got={got_l:e} ref={ref_bce_loss:e}"
    );
    assert!(
        (got_g - ref_bce_grad).abs() <= 1e-12 * ref_bce_grad.abs(),
        "poly1_bce eps=-1 x=40 grad: got={got_g:e} ref={ref_bce_grad:e}"
    );
    let (got_l_neg, got_g_neg) = poly1_bce_elem_f64(-40.0, 0.0, -1.0);
    assert!((got_l_neg - ref_bce_loss).abs() <= 1e-12 * ref_bce_loss);
    assert!((got_g_neg - (-ref_bce_grad)).abs() <= 1e-12 * ref_bce_grad.abs());

    // Poly-1 Focal at epsilon = -1, gamma = 1, x = 20, t = 1: loss = r * (r^2 / 2 + ...), grad = -r^3 * (1 + p / 2 + ...).
    let r20 = (-20.0f64).exp() / (1.0 + (-20.0f64).exp());
    let p20 = 1.0 - r20;
    let ref_fl_loss = r20 * (0.5 * r20 * r20 + r20.powi(3) / 3.0);
    let ref_fl_grad = -r20 * (p20 * (0.5 * r20 * r20 + r20.powi(3) / 3.0) + r20 * r20);
    let (fl_l, fl_g) = poly1_focal_elem_f64(20.0, 1.0, 1.0, -1.0);
    assert!(
        (fl_l - ref_fl_loss).abs() <= 1e-12 * ref_fl_loss,
        "poly1_focal eps=-1 loss: {fl_l:e} vs {ref_fl_loss:e}"
    );
    assert!(
        (fl_g - ref_fl_grad).abs() <= 1e-12 * ref_fl_grad.abs(),
        "poly1_focal eps=-1 grad: {fl_g:e} vs {ref_fl_grad:e}"
    );

    // 4. Empty tensors.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(asymmetric_loss(&empty, &empty, 0.0, 4.0, 0.05).unwrap().dims(), &[0, 4]);
    assert_eq!(poly1_bce_loss(&empty, &empty, 1.0).unwrap().dims(), &[0, 4]);
    assert_eq!(poly1_focal_loss(&empty, &empty, 2.0, 1.0).unwrap().dims(), &[0, 4]);

    // 4. Invalid hyperparameters and shape mismatch rejections.
    let x = Tensor::zeros([2, 3]);
    let t = Tensor::zeros([2, 3]);
    let bad_shape = Tensor::zeros([2, 4]);
    assert!(asymmetric_loss(&x, &bad_shape, 0.0, 4.0, 0.05).is_err());
    assert!(asymmetric_loss(&x, &t, -0.1, 4.0, 0.05).is_err());
    assert!(asymmetric_loss(&x, &t, 0.0, -0.1, 0.05).is_err());
    assert!(asymmetric_loss(&x, &t, 0.0, 4.0, -0.01).is_err());
    assert!(asymmetric_loss(&x, &t, 0.0, 4.0, 1.0).is_err());
    assert!(asymmetric_loss(&x, &t, f32::NAN, 4.0, 0.05).is_err());
    assert!(poly1_bce_loss(&x, &t, -1.1).is_err());
    assert!(poly1_bce_loss(&x, &t, f32::INFINITY).is_err());
    assert!(poly1_focal_loss(&x, &t, -0.5, 1.0).is_err());
    assert!(poly1_focal_loss(&x, &t, 2.0, -1.5).is_err());
    assert!(poly1_focal_loss_with_alpha(&x, &t, 2.0, 1.0, 0.0).is_err());
    assert!(poly1_focal_loss_with_alpha(&x, &t, 2.0, 1.0, 1.0).is_err());
}

/// Fused vs composite parity, `grad_check_with` through `loss_with_assignment`, all-negative (empty GT)
/// batch, end-to-end `detection_loss`, and `LossConfig` serde / validation for `Asl`, `Poly1Bce`, `Poly1Focal`.
#[test]
fn detection_loss_asl_and_polyloss_parity_grad_check_and_config() {
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let (b, nc, a, r) = (2, 4, anchors.len(), 16);
    let mut rng = folo_core::rng::Rng::seed(1337);
    let box_raw = Tensor::rand_uniform([b, 4 * r, a], 0.0, 3.0, &mut rng);
    let cls_raw = Tensor::rand_uniform([b, nc, a], -4.5, 3.5, &mut rng).requires_grad_();
    let targets = vec![
        ImageTargets { boxes: vec![Xyxy::new(8.0, 8.0, 40.0, 44.0)], labels: vec![1] },
        ImageTargets { boxes: vec![Xyxy::new(16.0, 12.0, 56.0, 52.0)], labels: vec![3] },
    ];
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    assert!(asg.num_fg() > 0);

    let configs = [
        LossConfig { cls_loss: ClsLossKind::Asl, ..LossConfig::default() },
        LossConfig {
            cls_loss: ClsLossKind::Asl,
            asl_gamma_pos: 1.0,
            asl_gamma_neg: 2.5,
            asl_clip: 0.1,
            ..LossConfig::default()
        },
        LossConfig {
            cls_loss: ClsLossKind::Asl,
            asl_gamma_pos: 0.0,
            asl_gamma_neg: 2.0,
            asl_clip: 0.0,
            ..LossConfig::default()
        },
        LossConfig { cls_loss: ClsLossKind::Poly1Bce, poly1_epsilon: 1.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Poly1Bce, poly1_epsilon: -0.5, ..LossConfig::default() },
        LossConfig {
            cls_loss: ClsLossKind::Poly1Focal,
            poly1_focal_gamma: 2.0,
            poly1_epsilon: 1.0,
            ..LossConfig::default()
        },
        LossConfig {
            cls_loss: ClsLossKind::Poly1Focal,
            poly1_focal_gamma: 0.5,
            poly1_epsilon: 1.5,
            ..LossConfig::default()
        },
    ];
    for cfg in configs {
        assert!(!device_loss_supported(&box_raw, &cfg));
        let fused = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, true).unwrap();
        let comp = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, false).unwrap();
        assert!(fused.is_finite() && comp.is_finite());
        assert!(
            (fused.items.cls - comp.items.cls).abs() <= 1e-5 * (1.0 + comp.items.cls.abs()),
            "{cfg:?}: fused {} vs comp {}",
            fused.items.cls,
            comp.items.cls
        );
        let gf = fused.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        let gc = comp.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        for (i, (&x, &y)) in gf.iter().zip(&gc).enumerate() {
            assert!(x.is_finite() && y.is_finite());
            assert!((x - y).abs() <= 2e-5 + 2e-4 * y.abs(), "{cfg:?} elem {i}: {x} vs {y}");
        }

        // Empty-GT (all-negative) batch stays finite on both paths and matches.
        let empty_targets = vec![ImageTargets::default(); b];
        let out_empty = detection_loss(&box_raw, &cls_raw, &anchors, &empty_targets, &cfg).unwrap();
        assert!(out_empty.is_finite() && out_empty.num_fg == 0);
        let g_empty = out_empty.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        assert!(g_empty.iter().all(|v| v.is_finite()));

        // Serde round-trip.
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), cfg);
    }

    // Tensor `grad_check_with` through `loss_impl` on small grid for all three new cls losses.
    let small_anchors = make_anchors(32, 32, &[16, 32], 0.5);
    let sa = small_anchors.len();
    let sbox = Tensor::rand_uniform([1, 4 * 4, sa], 0.0, 2.0, &mut rng);
    let scls = Tensor::rand_uniform([1, 2, sa], -2.0, 2.0, &mut rng);
    let sasg = BatchAssignment {
        batch: 1,
        num_anchors: sa,
        num_classes: 2,
        image: vec![0, 0],
        anchor: vec![0, sa as u32 - 1],
        label: vec![0, 1],
        score: vec![0.8, 0.35],
        target: vec![Xyxy::new(2.0, 2.0, 20.0, 22.0), Xyxy::new(6.0, 6.0, 28.0, 28.0)],
        inputs_finite: true,
    };
    for kind in [ClsLossKind::Asl, ClsLossKind::Poly1Bce, ClsLossKind::Poly1Focal] {
        let cfg = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: kind,
            ..LossConfig::default()
        };
        for fused in [true, false] {
            grad_check_with(
                |inps| Ok(loss_impl(&sbox, &inps[0], &small_anchors, &sasg, &cfg, fused)?.total),
                std::slice::from_ref(&scls),
                &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 19 },
            )
            .unwrap();
        }
    }

    // Non-finite logits (-Inf, +Inf, NaN) and non-finite target scores (NaN, -Inf) must poison BOTH
    // forward and backward identically on fused and composite paths for Asl, Poly1Bce, and Poly1Focal.
    for kind in [ClsLossKind::Asl, ClsLossKind::Poly1Bce, ClsLossKind::Poly1Focal] {
        let cfg = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: kind,
            ..LossConfig::default()
        };
        for bad_logit in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let mut cv = vec![0.0f32; 2 * sa];
            cv[1] = bad_logit; // background position
            let c_bad = Tensor::from_vec(cv, [1, 2, sa]).unwrap().requires_grad_();
            for fused in [true, false] {
                let out = loss_impl(&sbox, &c_bad, &small_anchors, &sasg, &cfg, fused).unwrap();
                assert!(out.items.cls.is_nan(), "{kind:?} fused={fused} bad_logit={bad_logit}: loss not NaN");
                let g = out.total.backward().unwrap().get(&c_bad).unwrap().to_vec::<f32>().unwrap();
                assert!(g[1].is_nan(), "{kind:?} fused={fused} bad_logit={bad_logit}: grad[1] not NaN");
            }
        }
        for bad_score in [f32::NAN, f32::NEG_INFINITY] {
            let mut bad_asg = sasg.clone();
            bad_asg.score[0] = bad_score;
            let c_ok = scls.detach().requires_grad_();
            for fused in [true, false] {
                let out = loss_impl(&sbox, &c_ok, &small_anchors, &bad_asg, &cfg, fused).unwrap();
                assert!(out.items.cls.is_nan(), "{kind:?} fused={fused} bad_score={bad_score}: loss not NaN");
                let g = out.total.backward().unwrap().get(&c_ok).unwrap().to_vec::<f32>().unwrap();
                assert!(g[0].is_nan(), "{kind:?} fused={fused} bad_score={bad_score}: grad[0] not NaN");
            }
        }
    }

    // Config validation rejects out-of-range ASL / PolyLoss fields and mal_gamma conflict.
    for bad in [
        LossConfig { cls_loss: ClsLossKind::Asl, mal_gamma: 1.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Poly1Bce, mal_gamma: 0.5, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Poly1Focal, mal_gamma: 0.5, ..LossConfig::default() },
        LossConfig { asl_gamma_pos: -0.1, ..LossConfig::default() },
        LossConfig { asl_gamma_neg: -0.1, ..LossConfig::default() },
        LossConfig { asl_clip: -0.01, ..LossConfig::default() },
        LossConfig { asl_clip: 1.0, ..LossConfig::default() },
        LossConfig { poly1_epsilon: -1.01, ..LossConfig::default() },
        LossConfig { poly1_focal_gamma: -0.1, ..LossConfig::default() },
        LossConfig { asl_clip: f32::NAN, ..LossConfig::default() },
        LossConfig { poly1_epsilon: f32::INFINITY, ..LossConfig::default() },
    ] {
        assert!(bad.validate_params().is_err(), "{bad:?} should be rejected");
        assert!(detection_loss(&box_raw, &cls_raw, &anchors, &targets, &bad).is_err());
    }
}
