//! Unit tests for standalone Varifocal Loss (VFL) and Quality Focal Loss (QFL):
//! exact `f64` central finite-difference checks (including `q = 0`, `q = 1`, fractional `q`,
//! extreme `±50` / `±500` logits, `beta == 1` cusp, fractional `gamma < 1`), tensor `grad_check_with`
//! across all [`LossReduction`] modes, algebraic reduction identities, parity with `detection_loss`'s
//! `ClsLossKind::Vfl` and `ClsLossKind::Qfl`, empty tensors, non-finite poison propagation, and
//! config/target validation.

use folo_core::testing::{GradCheck, grad_check_with};

use super::terms::{loss_impl, qfl_elem_f64, vfl_elem_f64};
use super::*;
use crate::decode::make_anchors;

/// `f64` central finite-difference gradient check for VFL across `q = 0`, continuous `q in (0, 1)`,
/// `q = 1`, fractional/zero/default `gamma`, and extreme logits `±50`.
#[test]
fn vfl_f64_central_fd_and_extreme_logits() {
    let params = [
        (0.75f64, 2.0f64), // paper default
        (0.25, 0.5),       // fractional gamma < 1
        (1.0, 0.0),        // gamma = 0 (unfocused negative BCE)
        (0.5, 1.0),        // linear focal negative
        (1.2, 3.0),        // steeper focal negative
    ];
    let logits = [-50.0f64, -12.0, -3.5, -0.7, 0.0, 0.9, 3.4, 12.0, 50.0];
    let targets = [0.0f64, 0.05, 0.35, 0.72, 1.0];

    for &(alpha, gamma) in &params {
        for &x in &logits {
            for &q in &targets {
                let (loss, grad) = vfl_elem_f64(x, q, alpha, gamma);
                assert!(loss.is_finite() && loss >= 0.0, "vfl ({alpha},{gamma}) x={x} q={q}: loss={loss}");
                assert!(grad.is_finite(), "vfl ({alpha},{gamma}) x={x} q={q}: grad={grad}");
                let h = 1e-6f64;
                let fp = vfl_elem_f64(x + h, q, alpha, gamma).0;
                let fm = vfl_elem_f64(x - h, q, alpha, gamma).0;
                let fd = (fp - fm) / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad.abs());
                assert!(
                    (grad - fd).abs() <= tol,
                    "vfl ({alpha},{gamma}) x={x} q={q}: ana={grad:e} fd={fd:e}"
                );
            }
        }
    }

    // Cancellation-free precision at x = +50, q = 1.0:
    // True loss = softplus(-50) ≈ e^-50, true grad = -sigmoid(-50) ≈ -e^-50.
    let e50 = (-50.0f64).exp();
    let want_l = e50.ln_1p();
    let want_g = -e50 / (1.0 + e50);
    let (got_l, got_g) = vfl_elem_f64(50.0, 1.0, 0.75, 2.0);
    assert!((got_l - want_l).abs() <= 1e-12 * want_l, "x=50 q=1 loss: {got_l:e} vs {want_l:e}");
    assert!((got_g - want_g).abs() <= 1e-12 * want_g.abs(), "x=50 q=1 grad: {got_g:e} vs {want_g:e}");

    // Extreme ±500 and ±1e38 logits remain completely finite.
    for x in [-1e38f64, -500.0, 500.0, 1e38] {
        for q in [0.0, 0.5, 1.0] {
            let (l, g) = vfl_elem_f64(x, q, 0.75, 0.5);
            assert!(l.is_finite() && l >= 0.0, "extreme vfl x={x} q={q}: l={l}");
            assert!(g.is_finite(), "extreme vfl x={x} q={q}: g={g}");
        }
    }
}

/// `f64` central finite-difference gradient check for QFL across `q = 0`, continuous `q in (0, 1)`,
/// `q = 1`, `beta = 0, 1, 1.5, 2, 3`, the exact `p == q` point, and extreme logits `±50` / `±500`.
#[test]
fn qfl_f64_central_fd_cusp_and_extreme_logits() {
    let betas = [0.0f64, 1.0, 1.5, 2.0, 3.0];
    let logits = [-50.0f64, -12.0, -3.1, -0.6, 0.2, 1.1, 3.7, 12.0, 50.0];
    let targets = [0.0f64, 0.15, 0.45, 0.85, 1.0];

    for &beta in &betas {
        for &x in &logits {
            for &q in &targets {
                let (loss, grad) = qfl_elem_f64(x, q, beta);
                assert!(loss.is_finite() && loss >= 0.0, "qfl beta={beta} x={x} q={q}: loss={loss}");
                assert!(grad.is_finite(), "qfl beta={beta} x={x} q={q}: grad={grad}");
                let h = 1e-6f64;
                let fp = qfl_elem_f64(x + h, q, beta).0;
                let fm = qfl_elem_f64(x - h, q, beta).0;
                let fd = (fp - fm) / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad.abs());
                assert!((grad - fd).abs() <= tol, "qfl beta={beta} x={x} q={q}: ana={grad:e} fd={fd:e}");
            }
        }
    }

    // Exact p == q zero-diff points (x = 0 with q = 0.5, and x = ln(3) with q = 0.75):
    // loss == 0 and grad == 0 for all beta >= 1 (including the beta = 1 cusp).
    for &(x_eq, q_eq) in &[(0.0f64, 0.5f64), (3.0f64.ln(), 0.75f64)] {
        for &beta in &[1.0f64, 1.5, 2.0] {
            let (l, g) = qfl_elem_f64(x_eq, q_eq, beta);
            assert!(l.abs() < 1e-15, "p==q beta={beta} q={q_eq}: l={l:e}");
            assert!(g.abs() < 1e-15, "p==q beta={beta} q={q_eq}: g={g:e}");
        }
    }

    // Tiny perturbations around the beta == 1 cusp (x = ±1e-20f32, q = 0.5) must NOT collapse to a false
    // zero-gradient dead zone: true loss ≈ 0.25 * |x| * ln(2) ≈ 1.7329e-21, true grad ≈ ±0.25 * ln(2) ≈ ±0.17329.
    let eps_x = 1e-20f32;
    let cusp_x = Tensor::from_slice(&[-eps_x, 0.0, eps_x], [3]).unwrap().requires_grad_();
    let cusp_q = Tensor::from_slice(&[0.5f32, 0.5, 0.5], [3]).unwrap();
    let cusp_loss =
        quality_focal_loss(&cusp_x, &cusp_q, &QflConfig { beta: 1.0, reduction: LossReduction::None })
            .unwrap();
    let cusp_lv = cusp_loss.to_vec::<f32>().unwrap();
    let cusp_gv =
        cusp_loss.sum_all().unwrap().backward().unwrap().get(&cusp_x).unwrap().to_vec::<f32>().unwrap();
    let want_cusp_l = (0.25 * (eps_x as f64) * 2.0f64.ln()) as f32;
    let want_cusp_g = (0.25 * 2.0f64.ln()) as f32;
    assert!((cusp_lv[0] - want_cusp_l).abs() <= 1e-5 * want_cusp_l, "cusp -eps loss: {}", cusp_lv[0]);
    assert_eq!(cusp_lv[1], 0.0, "exact midpoint loss must be 0");
    assert!((cusp_lv[2] - want_cusp_l).abs() <= 1e-5 * want_cusp_l, "cusp +eps loss: {}", cusp_lv[2]);
    assert!((cusp_gv[0] + want_cusp_g).abs() <= 1e-5 * want_cusp_g, "cusp -eps grad: {}", cusp_gv[0]);
    assert_eq!(cusp_gv[1], 0.0, "exact midpoint subgradient must be 0");
    assert!((cusp_gv[2] - want_cusp_g).abs() <= 1e-5 * want_cusp_g, "cusp +eps grad: {}", cusp_gv[2]);

    // Large exponents (1e22f32) at x = ±50 evaluate the focusing factor in log space without rounding p/r to 1.0.
    let big_exp = 1e22f32 as f64;
    let sp_m50 = (-50.0f64).exp().ln_1p();
    let sp_p50 = 50.0 + sp_m50;
    let r50_exact = (-50.0f64).exp() / (1.0 + (-50.0f64).exp());
    let p50_exact = 1.0 / (1.0 + (-50.0f64).exp());
    let pow_big = (-big_exp * sp_m50).exp();
    let want_big_l = pow_big * sp_p50;
    let want_big_g = pow_big * ((big_exp * r50_exact) * sp_p50 + p50_exact);
    let (vfl_bl, vfl_bg) = vfl_elem_f64(50.0, 0.0, 1.0, big_exp);
    assert!(
        (vfl_bl - want_big_l).abs() <= 1e-12 * want_big_l,
        "vfl big gamma loss: {vfl_bl} vs {want_big_l}"
    );
    assert!(
        (vfl_bg - want_big_g).abs() <= 1e-12 * want_big_g,
        "vfl big gamma grad: {vfl_bg} vs {want_big_g}"
    );
    let (qfl_bl0, qfl_bg0) = qfl_elem_f64(50.0, 0.0, big_exp);
    assert!(
        (qfl_bl0 - want_big_l).abs() <= 1e-12 * want_big_l,
        "qfl q=0 big beta loss: {qfl_bl0} vs {want_big_l}"
    );
    assert!(
        (qfl_bg0 - want_big_g).abs() <= 1e-12 * want_big_g,
        "qfl q=0 big beta grad: {qfl_bg0} vs {want_big_g}"
    );
    let (qfl_bl1, qfl_bg1) = qfl_elem_f64(-50.0, 1.0, big_exp);
    assert!(
        (qfl_bl1 - want_big_l).abs() <= 1e-12 * want_big_l,
        "qfl q=1 big beta loss: {qfl_bl1} vs {want_big_l}"
    );
    assert!(
        (qfl_bg1 + want_big_g).abs() <= 1e-12 * want_big_g,
        "qfl q=1 big beta grad: {qfl_bg1} vs -{want_big_g}"
    );

    // Cancellation-free precision at x = +50, q = 1.0, beta = 2.0:
    // True loss = r^2 * softplus(-50) ≈ e^-150, true grad = -r^2 * (2 * p * softplus(-50) + r) ≈ -3 * e^-150.
    let r50 = (-50.0f64).exp() / (1.0 + (-50.0f64).exp());
    let sp50 = (-50.0f64).exp().ln_1p();
    let want_l = r50 * r50 * sp50;
    let want_g = -(r50 * r50) * (2.0 * (1.0 - r50) * sp50 + r50);
    let (got_l, got_g) = qfl_elem_f64(50.0, 1.0, 2.0);
    assert!((got_l - want_l).abs() <= 1e-12 * want_l, "qfl x=50 q=1 loss: {got_l:e} vs {want_l:e}");
    assert!((got_g - want_g).abs() <= 1e-12 * want_g.abs(), "qfl x=50 q=1 grad: {got_g:e} vs {want_g:e}");

    // Extreme ±500 and ±1e38 logits remain completely finite.
    for x in [-1e38f64, -500.0, 500.0, 1e38] {
        for q in [0.0, 0.5, 1.0] {
            for beta in [0.0, 1.0, 1.5, 2.0] {
                let (l, g) = qfl_elem_f64(x, q, beta);
                assert!(l.is_finite() && l >= 0.0, "extreme qfl beta={beta} x={x} q={q}: l={l}");
                assert!(g.is_finite(), "extreme qfl beta={beta} x={x} q={q}: g={g}");
            }
        }
    }
}

/// Tensor-level `grad_check_with` across `LossReduction::{None, Sum, Mean}`, algebraic identities,
/// and parity with `detection_loss`'s `ClsLossKind::Vfl` and `ClsLossKind::Qfl`.
#[test]
fn vfl_qfl_tensor_grad_checks_reductions_and_detection_loss_parity() {
    let logits = Tensor::from_slice(&[-3.2f32, -1.1, -0.2, 0.5, 1.8, 4.1, -2.5, 0.9], [2, 4]).unwrap();
    let targets = Tensor::from_slice(&[0.0f32, 0.0, 0.25, 0.7, 1.0, 0.9, 0.0, 0.45], [2, 4]).unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 8, seed: 101 };

    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let vcfg = VflConfig { alpha: 0.75, gamma: 2.0, reduction };
        let qcfg = QflConfig { beta: 2.0, reduction };
        let qcfg15 = QflConfig { beta: 1.5, reduction };
        grad_check_with(
            |inps| {
                let l = varifocal_loss(&inps[0], &targets, &vcfg)?;
                if reduction == LossReduction::None { l.sum_all() } else { Ok(l) }
            },
            std::slice::from_ref(&logits),
            &gc,
        )
        .unwrap();
        grad_check_with(
            |inps| {
                let l = quality_focal_loss(&inps[0], &targets, &qcfg)?;
                if reduction == LossReduction::None { l.sum_all() } else { Ok(l) }
            },
            std::slice::from_ref(&logits),
            &gc,
        )
        .unwrap();
        grad_check_with(
            |inps| {
                let l = quality_focal_loss(&inps[0], &targets, &qcfg15)?;
                if reduction == LossReduction::None { l.sum_all() } else { Ok(l) }
            },
            std::slice::from_ref(&logits),
            &gc,
        )
        .unwrap();
    }

    // Reduction consistency: Sum == sum(None), Mean == Sum / N.
    let v_none = varifocal_loss(
        &logits,
        &targets,
        &VflConfig { reduction: LossReduction::None, ..VflConfig::default() },
    )
    .unwrap();
    assert_eq!(v_none.dims(), &[2, 4]);
    let v_sum = varifocal_loss(
        &logits,
        &targets,
        &VflConfig { reduction: LossReduction::Sum, ..VflConfig::default() },
    )
    .unwrap()
    .item::<f32>()
    .unwrap();
    let v_mean = varifocal_loss(
        &logits,
        &targets,
        &VflConfig { reduction: LossReduction::Mean, ..VflConfig::default() },
    )
    .unwrap()
    .item::<f32>()
    .unwrap();
    let v_none_sum: f32 = v_none.to_vec::<f32>().unwrap().iter().sum();
    assert!((v_sum - v_none_sum).abs() < 1e-6);
    assert!((v_mean - v_sum / 8.0).abs() < 1e-6);

    // Algebraic identities:
    // 1. QFL(beta = 0) == BCE
    let bce =
        logits.softplus().unwrap().sub(&logits.mul(&targets).unwrap()).unwrap().to_vec::<f32>().unwrap();
    let qfl0 =
        quality_focal_loss(&logits, &targets, &QflConfig { beta: 0.0, reduction: LossReduction::None })
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
    for i in 0..bce.len() {
        assert!((qfl0[i] - bce[i]).abs() < 1e-5, "QFL(0) != BCE at {i}: {} vs {}", qfl0[i], bce[i]);
    }
    // 2. VFL(alpha = 1, gamma = 0) on binary targets {0, 1} == BCE
    let bin_t = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [2, 4]).unwrap();
    let bin_bce =
        logits.softplus().unwrap().sub(&logits.mul(&bin_t).unwrap()).unwrap().to_vec::<f32>().unwrap();
    let vfl_bce = varifocal_loss(
        &logits,
        &bin_t,
        &VflConfig { alpha: 1.0, gamma: 0.0, reduction: LossReduction::None },
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    for i in 0..bin_bce.len() {
        assert!((vfl_bce[i] - bin_bce[i]).abs() < 1e-5, "VFL(1,0) on binary != BCE at {i}");
    }
    // 3. On all-negative targets (q == 0), VFL(alpha = 1, gamma = beta) == QFL(beta)
    let zeros_t = Tensor::zeros([2, 4]);
    let vfl_neg = varifocal_loss(
        &logits,
        &zeros_t,
        &VflConfig { alpha: 1.0, gamma: 2.0, reduction: LossReduction::None },
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let qfl_neg =
        quality_focal_loss(&logits, &zeros_t, &QflConfig { beta: 2.0, reduction: LossReduction::None })
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
    for i in 0..vfl_neg.len() {
        assert!((vfl_neg[i] - qfl_neg[i]).abs() < 1e-6, "VFL neg != QFL neg at {i}");
    }

    // 4. Parity with `detection_loss`'s `ClsLossKind::Vfl` and `ClsLossKind::Qfl`.
    let anchors = make_anchors(32, 32, &[16, 32], 0.5);
    let sa = anchors.len();
    let mut rng = folo_core::rng::Rng::seed(2026);
    let sbox = Tensor::rand_uniform([1, 4 * 4, sa], 0.0, 2.0, &mut rng);
    let scls = Tensor::rand_uniform([1, 2, sa], -3.0, 3.0, &mut rng).requires_grad_();
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
    let mut dense_t = vec![0.0f32; 2 * sa];
    dense_t[0] = 0.8;
    dense_t[sa + sa - 1] = 0.35;
    let t_tensor = Tensor::from_vec(dense_t, [1, 2, sa]).unwrap();
    let tss = sasg.target_score_sum().max(1.0) as f32;

    let det_vfl = loss_impl(
        &sbox,
        &scls,
        &anchors,
        &sasg,
        &LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: ClsLossKind::Vfl,
            vfl_alpha: 0.75,
            vfl_gamma: 2.0,
            ..LossConfig::default()
        },
        true,
    )
    .unwrap();
    let stand_vfl = varifocal_loss(
        &scls,
        &t_tensor,
        &VflConfig { alpha: 0.75, gamma: 2.0, reduction: LossReduction::Sum },
    )
    .unwrap()
    .div_scalar(tss)
    .unwrap();
    assert!((det_vfl.items.cls - stand_vfl.item::<f32>().unwrap()).abs() < 1e-5);
    let gv_det = det_vfl.total.backward().unwrap().get(&scls).unwrap().to_vec::<f32>().unwrap();
    let gv_st = stand_vfl.backward().unwrap().get(&scls).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in gv_det.iter().zip(&gv_st) {
        assert!((a - b).abs() < 1e-5 + 1e-4 * b.abs(), "VFL grad parity: {a} vs {b}");
    }

    let det_qfl = loss_impl(
        &sbox,
        &scls,
        &anchors,
        &sasg,
        &LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: ClsLossKind::Qfl,
            qfl_beta: 2.0,
            ..LossConfig::default()
        },
        true,
    )
    .unwrap();
    let stand_qfl =
        quality_focal_loss(&scls, &t_tensor, &QflConfig { beta: 2.0, reduction: LossReduction::Sum })
            .unwrap()
            .div_scalar(tss)
            .unwrap();
    assert!((det_qfl.items.cls - stand_qfl.item::<f32>().unwrap()).abs() < 1e-5);
    let gq_det = det_qfl.total.backward().unwrap().get(&scls).unwrap().to_vec::<f32>().unwrap();
    let gq_st = stand_qfl.backward().unwrap().get(&scls).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in gq_det.iter().zip(&gq_st) {
        assert!((a - b).abs() < 1e-5 + 1e-4 * b.abs(), "QFL grad parity: {a} vs {b}");
    }
}

/// Edge cases: extreme `±50` / `±500` tensor logits, empty tensors across all reductions, non-finite
/// input poison propagation, config serde round-trip, and invalid config / shape / target rejections.
#[test]
fn vfl_qfl_edge_cases_and_validation() {
    // 1. Extreme ±50 and ±500 logits stay finite in forward & backward.
    let extreme_x = Tensor::from_slice(&[-500.0f32, -50.0, 0.0, 50.0, 500.0], [5]).unwrap().requires_grad_();
    let all_neg = Tensor::zeros([5]);
    let all_pos = Tensor::ones([5]);
    let mixed = Tensor::from_slice(&[0.0f32, 1.0, 0.5, 1.0, 0.0], [5]).unwrap();

    for t in [&all_neg, &all_pos, &mixed] {
        for l in [
            varifocal_loss(
                &extreme_x,
                t,
                &VflConfig { reduction: LossReduction::None, ..VflConfig::default() },
            )
            .unwrap(),
            quality_focal_loss(
                &extreme_x,
                t,
                &QflConfig { reduction: LossReduction::None, ..QflConfig::default() },
            )
            .unwrap(),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv.iter().all(|v| v.is_finite() && *v >= 0.0), "extreme loss: {lv:?}");
            let gv =
                l.sum_all().unwrap().backward().unwrap().get(&extreme_x).unwrap().to_vec::<f32>().unwrap();
            assert!(gv.iter().all(|v| v.is_finite()), "extreme grad: {gv:?}");
        }
    }

    // 2. Empty tensors across all reduction modes produce finite 0 and valid empty gradient.
    let empty_x = Tensor::zeros([0, 4]).requires_grad_();
    let empty_t = Tensor::zeros([0, 4]);
    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let vl =
            varifocal_loss(&empty_x, &empty_t, &VflConfig { reduction, ..VflConfig::default() }).unwrap();
        let ql =
            quality_focal_loss(&empty_x, &empty_t, &QflConfig { reduction, ..QflConfig::default() }).unwrap();
        if reduction == LossReduction::None {
            assert_eq!(vl.dims(), &[0, 4]);
            assert_eq!(ql.dims(), &[0, 4]);
        } else {
            assert_eq!(vl.item::<f32>().unwrap(), 0.0);
            assert_eq!(ql.item::<f32>().unwrap(), 0.0);
            let gv = vl.backward().unwrap();
            assert_eq!(gv.get(&empty_x).unwrap().dims(), &[0, 4]);
            let gq = ql.backward().unwrap();
            assert_eq!(gq.get(&empty_x).unwrap().dims(), &[0, 4]);
        }
    }

    // 3. Non-finite logits and targets propagate NaN through both forward and backward.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bx = Tensor::from_slice(&[bad, 0.5], [2]).unwrap().requires_grad_();
        let ok_t = Tensor::from_slice(&[0.7f32, 0.0], [2]).unwrap();
        for l in [
            varifocal_loss(&bx, &ok_t, &VflConfig { reduction: LossReduction::None, ..VflConfig::default() })
                .unwrap(),
            quality_focal_loss(
                &bx,
                &ok_t,
                &QflConfig { reduction: LossReduction::None, ..QflConfig::default() },
            )
            .unwrap(),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv[0].is_nan() && lv[1].is_finite(), "bad logit {bad}: {lv:?}");
            let gv = l.sum_all().unwrap().backward().unwrap().get(&bx).unwrap().to_vec::<f32>().unwrap();
            assert!(gv[0].is_nan() && gv[1].is_finite(), "bad logit {bad} grad: {gv:?}");
        }

        let ok_x = Tensor::from_slice(&[0.5f32, -0.5], [2]).unwrap().requires_grad_();
        let bt = Tensor::from_slice(&[bad, 1.0], [2]).unwrap();
        for l in [
            varifocal_loss(&ok_x, &bt, &VflConfig { reduction: LossReduction::None, ..VflConfig::default() })
                .unwrap(),
            quality_focal_loss(
                &ok_x,
                &bt,
                &QflConfig { reduction: LossReduction::None, ..QflConfig::default() },
            )
            .unwrap(),
        ] {
            let lv = l.to_vec::<f32>().unwrap();
            assert!(lv[0].is_nan() && lv[1].is_finite(), "bad target {bad}: {lv:?}");
            let gv = l.sum_all().unwrap().backward().unwrap().get(&ok_x).unwrap().to_vec::<f32>().unwrap();
            assert!(gv[0].is_nan() && gv[1].is_finite(), "bad target {bad} grad: {gv:?}");
        }
    }

    // 4. Config validation, target range [0, 1] validation, shape mismatch, and serde round-trip.
    let x = Tensor::zeros([2, 3]);
    let t = Tensor::zeros([2, 3]);
    let bad_shape = Tensor::zeros([2, 4]);
    let t_neg = Tensor::from_slice(&[-0.01f32, 0.5, 0.0, 0.0, 0.0, 0.0], [2, 3]).unwrap();
    let t_gt1 = Tensor::from_slice(&[1.01f32, 0.5, 0.0, 0.0, 0.0, 0.0], [2, 3]).unwrap();

    assert!(varifocal_loss(&x, &bad_shape, &VflConfig::default()).is_err());
    assert!(quality_focal_loss(&x, &bad_shape, &QflConfig::default()).is_err());
    assert!(varifocal_loss(&x, &t_neg, &VflConfig::default()).is_err());
    assert!(varifocal_loss(&x, &t_gt1, &VflConfig::default()).is_err());
    assert!(quality_focal_loss(&x, &t_neg, &QflConfig::default()).is_err());
    assert!(quality_focal_loss(&x, &t_gt1, &QflConfig::default()).is_err());

    for bad_vfl in [
        VflConfig { alpha: 0.0, ..VflConfig::default() },
        VflConfig { alpha: -0.5, ..VflConfig::default() },
        VflConfig { alpha: f32::NAN, ..VflConfig::default() },
        VflConfig { alpha: f32::INFINITY, ..VflConfig::default() },
        VflConfig { gamma: -0.1, ..VflConfig::default() },
        VflConfig { gamma: f32::NAN, ..VflConfig::default() },
    ] {
        assert!(bad_vfl.validate().is_err());
        assert!(varifocal_loss(&x, &t, &bad_vfl).is_err());
    }
    for bad_qfl in [
        QflConfig { beta: -1.0, ..QflConfig::default() },
        QflConfig { beta: 0.5, ..QflConfig::default() },
        QflConfig { beta: 0.999, ..QflConfig::default() },
        QflConfig { beta: f32::NAN, ..QflConfig::default() },
        QflConfig { beta: f32::INFINITY, ..QflConfig::default() },
    ] {
        assert!(bad_qfl.validate().is_err());
        assert!(quality_focal_loss(&x, &t, &bad_qfl).is_err());
    }

    let vcfg = VflConfig { alpha: 0.6, gamma: 1.5, reduction: LossReduction::Sum };
    assert_eq!(serde_json::from_str::<VflConfig>(&serde_json::to_string(&vcfg).unwrap()).unwrap(), vcfg);
    let qcfg = QflConfig { beta: 1.5, reduction: LossReduction::None };
    assert_eq!(serde_json::from_str::<QflConfig>(&serde_json::to_string(&qcfg).unwrap()).unwrap(), qcfg);
}
