//! Unit tests for Normalized Wasserstein Distance (`NWD`) and Wise-IoU v1 / v3 (`WIoU v1` / `WIoU v3`):
//! `f64` central finite-difference gradient checks (including detached enclosing-diagonal verification),
//! `Tensor`-level `grad_check_with`, tiny/non-overlapping non-vanishing gradients, scale invariance across
//! 24 orders of magnitude, degenerate/inverted/zero-area/extreme edge cases, `NaN`/`±Inf` poison,
//! frozen-assignment DFL-logit finite-difference checks, and end-to-end `detection_loss` / config validation.

use super::terms::{nwd_box_f64, wiou_v1_box_f64, wiou_v3_box_f64};
use super::*;
use crate::decode::make_anchors;

/// Helper: evaluate WIoU v1 loss at `a` while holding the detached enclosing diagonal `c2_fixed` constant,
/// matching the exact detached gradient objective of Tong et al. (2023) §III-A.
fn wiou_v1_frozen_c2_f64(a: [f64; 4], b: [f64; 4], c2_fixed: f64) -> f64 {
    let (_, _, l_iou) = wiou_v1_box_f64(a, b);
    let r_dist = if c2_fixed <= 0.0 {
        1.0
    } else {
        let dcx = 0.5 * ((a[0] - b[0]) + (a[2] - b[2]));
        let dcy = 0.5 * ((a[1] - b[1]) + (a[3] - b[3]));
        ((dcx * dcx + dcy * dcy) / c2_fixed).exp()
    };
    r_dist * l_iou
}

/// Compute `c2 = cw² + ch²` matching `enclosing_axis` over all endpoints of `a` and `b`.
fn enclosing_c2_f64(a: [f64; 4], b: [f64; 4]) -> f64 {
    let cw = a[0].max(a[2]).max(b[0].max(b[2])) - a[0].min(a[2]).min(b[0].min(b[2]));
    let ch = a[1].max(a[3]).max(b[1].max(b[3])) - a[1].min(a[3]).min(b[1].min(b[3]));
    cw * cw + ch * ch
}

/// `f64` central finite-difference gradient check for `NWD` across overlapping, non-overlapping/disjoint,
/// concentric, tiny, huge, extreme aspect-ratio, and inverted boxes, plus `Tensor` `grad_check_with` wrt
/// both `b1` and `b2`, non-vanishing gradients for tiny/disjoint boxes, and joint scale invariance.
#[test]
fn nwd_f64_central_fd_and_tensor_grad_check() {
    use folo_core::testing::{GradCheck, grad_check_with};

    let cases: [([f64; 4], [f64; 4], f64); 10] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], 12.8), // overlapping
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5], 6.4),    // concentric different size
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], 12.8), // non-overlapping / disjoint
        ([0.0, 0.0, 1.0, 1.0], [2.5, 1.5, 3.5, 2.5], 4.0),      // tiny disjoint boxes
        ([1e-4, 2e-4, 5e-4, 6e-4], [2e-4, 1e-4, 7e-4, 5e-4], 1e-3), // sub-pixel tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7], 2e7), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5], 25.0), // extreme aspect ratio
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0], 12.8),  // inverted predicted box
        ([5.0, 5.0, 5.0, 5.0], [7.0, 8.0, 11.0, 14.0], 10.0),   // zero-area point vs box
        ([1.0, 2.0, 1.0, 8.0], [3.0, 2.0, 9.0, 8.0], 8.0),      // zero-width line vs box
    ];
    for (idx, &(a, b, c)) in cases.iter().enumerate() {
        let (loss, grad) = nwd_box_f64(a, b, c);
        assert!(loss.is_finite() && (0.0..=1.0).contains(&loss), "nwd case {idx}: bad loss {loss}");
        // Every non-identical case with moderate W_2/c must have strictly non-vanishing gradient!
        assert!(
            grad.iter().any(|g| g.abs() > 1e-12),
            "nwd case {idx}: expected non-vanishing gradient, got {grad:?}"
        );
        for coord in 0..4 {
            let span = if coord % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(c * 1e-3).max(1e-6)).max(1e-8 * a[coord].abs());
            let mut ap = a;
            let mut am = a;
            ap[coord] += eps;
            am[coord] -= eps;
            let fd = (nwd_box_f64(ap, b, c).0 - nwd_box_f64(am, b, c).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[coord].abs());
            assert!(
                (grad[coord] - fd).abs() <= tol,
                "nwd case {idx} coord {coord}: analytic {} vs fd {fd}",
                grad[coord]
            );
        }
    }

    // Joint scale invariance across 24 orders of magnitude when (a, b, c) scale together.
    let l1 = nwd_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.7, 1.8, 3.1], 3.0).0;
    let l_tiny = nwd_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.7e-12, 1.8e-12, 3.1e-12], 3e-12).0;
    let l_huge = nwd_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.7e12, 1.8e12, 3.1e12], 3e12).0;
    assert!((l1 - l_tiny).abs() < 1e-12, "nwd tiny scale invariance: {l1} vs {l_tiny}");
    assert!((l1 - l_huge).abs() < 1e-12, "nwd huge scale invariance: {l1} vs {l_huge}");

    // Verify 2D Gaussian Wasserstein formula directly against cx/cy/w/h definition.
    let (a, b, c) = ([1.0f64, 2.0, 7.0, 10.0], [4.0, 1.0, 12.0, 13.0], 12.8);
    let dcx = 0.5 * ((a[0] + a[2]) - (b[0] + b[2]));
    let dcy = 0.5 * ((a[1] + a[3]) - (b[1] + b[3]));
    let dw = (a[2] - a[0]) - (b[2] - b[0]);
    let dh = (a[3] - a[1]) - (b[3] - b[1]);
    let w2 = (dcx * dcx + dcy * dcy + 0.25 * (dw * dw + dh * dh)).sqrt();
    let want = 1.0 - (-w2 / c).exp();
    let (got, _) = nwd_box_f64(a, b, c);
    assert!((got - want).abs() < 1e-14, "nwd formula mismatch: {got} vs {want}");

    // Tensor-level grad_check_with wrt BOTH b1 and b2 (overlapping + disjoint + tiny rows).
    let tb1 = Tensor::from_slice(&[2.0f32, 3.0, 10.0, 12.0, 0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.5, 5.5], [3, 4])
        .unwrap();
    let tb2 = Tensor::from_slice(&[3.5f32, 2.5, 11.5, 13.0, 6.0, 7.0, 7.0, 8.0, 5.8, 5.9, 6.3, 6.4], [3, 4])
        .unwrap();
    let tc = Tensor::from_slice(&[12.8f32, 8.0, 2.0], [3]).unwrap();
    grad_check_with(
        |inputs| nwd_loss(&inputs[0], &inputs[1], &tc)?.sum_all(),
        &[tb1.clone(), tb2.clone()],
        &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 24, seed: 17 },
    )
    .unwrap();
    grad_check_with(
        |inputs| nwd_loss_with_c(&inputs[0], &tb2, 12.8)?.sum_all(),
        &[tb1],
        &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 12, seed: 19 },
    )
    .unwrap();
}

/// `f64` central finite-difference gradient checks for `WIoU v1` and `WIoU v3` across overlapping,
/// non-overlapping, concentric, interior, tiny, huge, extreme aspect, and inverted boxes; verifies exact
/// enclosing-diagonal detachment, non-monotonic focusing peak, `Tensor` `grad_check_with`, and 24-decade
/// scale invariance.
#[test]
fn wiou_v1_and_v3_f64_central_fd_and_focusing() {
    use folo_core::testing::{GradCheck, grad_check_with};

    let cases: [([f64; 4], [f64; 4]); 10] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0]), // overlapping
        ([3.0, 3.5, 7.5, 8.0], [1.0, 1.0, 10.0, 10.0]),   // strictly interior (c2* locally constant)
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5]),   // concentric different size
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0]), // non-overlapping / disjoint
        ([0.0, 2.0, 4.0, 6.0], [7.0, 2.0, 11.0, 6.0]),    // horizontally collinear
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7]), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5]), // extreme aspect ratio
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0]),  // inverted predicted box
        ([5.0, 5.0, 5.0, 5.0], [7.0, 8.0, 11.0, 14.0]),   // zero-area point vs box
    ];
    for (idx, &(a, b)) in cases.iter().enumerate() {
        let (l1, g1, l_iou) = wiou_v1_box_f64(a, b);
        assert!(l1.is_finite() && l1 >= 0.0, "wiou_v1 case {idx}: bad loss {l1}");
        assert!(l_iou.is_finite() && (0.0..=1.0).contains(&l_iou));
        let c2_fixed = enclosing_c2_f64(a, b);

        // Frozen-c2* central FD must match WIoU v1 analytical gradient on EVERY case.
        for coord in 0..4 {
            let span = if coord % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[coord].abs());
            let mut ap = a;
            let mut am = a;
            ap[coord] += eps;
            am[coord] -= eps;
            let fd = (wiou_v1_frozen_c2_f64(ap, b, c2_fixed) - wiou_v1_frozen_c2_f64(am, b, c2_fixed))
                / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(g1[coord].abs());
            assert!(
                (g1[coord] - fd).abs() <= tol,
                "wiou_v1 case {idx} coord {coord}: analytic {} vs frozen-c2 FD {fd}",
                g1[coord]
            );
        }

        // WIoU v3 gradient must equal exact detached `r * g1` across varied (alpha, delta, iou_mean).
        for &(alpha, delta, iou_mean) in &[(1.9, 3.0, 1.0), (1.9, 3.0, 0.35), (2.5, 2.0, 0.5)] {
            let (l3, g3, l_iou3) = wiou_v3_box_f64(a, b, alpha, delta, iou_mean);
            assert_eq!(l_iou3, l_iou);
            let beta = l_iou / iou_mean.max(1e-4);
            let r = beta / (delta * ((beta - delta) * alpha.ln()).clamp(-80.0, 80.0).exp());
            assert!((l3 - r * l1).abs() <= 1e-12 + 1e-12 * l3.abs());
            for coord in 0..4 {
                assert!(
                    (g3[coord] - r * g1[coord]).abs() <= 1e-12 + 1e-12 * g3[coord].abs(),
                    "wiou_v3 case {idx} coord {coord}: {} vs {}",
                    g3[coord],
                    r * g1[coord]
                );
            }
        }
    }

    // Verify enclosing-box detachment: on disjoint boxes (where b1 forms outer edges of enclosing box),
    // the undetached FD includes d(c2)/db1 whereas WIoU v1 explicitly detaches c2* (so analytic != undetached FD
    // while matching frozen-c2 FD to 1e-10).
    let (da, db) = ([0.0f64, 0.0, 4.0, 4.0], [10.0, 0.0, 14.0, 4.0]);
    let (_, g_det, _) = wiou_v1_box_f64(da, db);
    let eps = 1e-6;
    let fd_undetached = (wiou_v1_box_f64([da[0] + eps, da[1], da[2], da[3]], db).0
        - wiou_v1_box_f64([da[0] - eps, da[1], da[2], da[3]], db).0)
        / (2.0 * eps);
    assert!(
        (g_det[0] - fd_undetached).abs() > 1e-2,
        "c2* must be detached: analytic {} vs undetached FD {fd_undetached}",
        g_det[0]
    );
    // And on disjoint boxes, WIoU v1 still yields a negative gradient on ax1/ax2 pulling box `a` right toward `b`!
    assert!(g_det[0] < -0.01 && g_det[2] < -0.01, "disjoint WIoU v1 must pull toward target: {g_det:?}");

    // Direct Tensor-level grad_check_with for wiou_v1_loss on strictly interior predicted boxes (where c2* is
    // locally constant so undetached FD == detached analytical gradient!).
    let interior_a = Tensor::from_slice(&[2.5f32, 3.0, 7.0, 8.0, 3.0, 2.0, 8.5, 7.5], [2, 4]).unwrap();
    let outer_b = Tensor::from_slice(&[0.0f32, 0.0, 12.0, 12.0, 0.0, 0.0, 12.0, 12.0], [2, 4]).unwrap();
    grad_check_with(
        |inputs| wiou_v1_loss(&inputs[0], &outer_b)?.sum_all(),
        &[interior_a],
        &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 8, seed: 23 },
    )
    .unwrap();

    // Non-monotonic focusing property of WIoU v3: r(beta) peaks near beta = 1 / ln(alpha) ≈ 1.558 for alpha = 1.9,
    // assigning lower gradient gain to both high-quality (small beta) and outlier (large beta) anchors.
    let (alpha, delta) = (1.9f64, 3.0f64);
    let r_of = |beta: f64| beta / (delta * ((beta - delta) * alpha.ln()).exp());
    let peak = 1.0 / alpha.ln();
    assert!(r_of(peak) > r_of(0.2) && r_of(peak) > r_of(6.0), "WIoU v3 focusing must be non-monotonic");
    assert!((r_of(delta) - 1.0).abs() < 1e-12, "r(delta) == 1");

    // Scale invariance across 24 orders of magnitude (`1e-12` vs `1.0` vs `1e12`).
    let (v1_1, _, _) = wiou_v1_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.7, 1.8, 3.1]);
    let (v1_tiny, _, _) = wiou_v1_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.7e-12, 1.8e-12, 3.1e-12]);
    let (v1_huge, _, _) = wiou_v1_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.7e12, 1.8e12, 3.1e12]);
    assert!((v1_1 - v1_tiny).abs() < 1e-12 && (v1_1 - v1_huge).abs() < 1e-12);

    let (v3_1, _, _) = wiou_v3_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.7, 1.8, 3.1], 1.9, 3.0, 0.5);
    let (v3_tiny, _, _) =
        wiou_v3_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.7e-12, 1.8e-12, 3.1e-12], 1.9, 3.0, 0.5);
    let (v3_huge, _, _) =
        wiou_v3_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.7e12, 1.8e12, 3.1e12], 1.9, 3.0, 0.5);
    assert!((v3_1 - v3_tiny).abs() < 1e-12 && (v3_1 - v3_huge).abs() < 1e-12);
}

/// Edge cases for `NWD`, `WIoU v1`, and `WIoU v3`: identical (including tiny `1e-12`), zero-area point,
/// zero-width line, inverted, extreme `1e15` / `1e30`, empty `[0, 4]`, `NaN`/`±Inf` poison on both `b1`
/// and `b2`, and invalid parameter rejection.
#[test]
fn nwd_and_wiou_edge_cases_and_validation() {
    // 1. Huge 1e15 boxes and extreme inverted [1e30, 0, 0, 0] vs [0, 0, 1e-30, 0] stay finite in fwd & bwd.
    let huge_a =
        Tensor::from_slice(&[0.0f32, 0.0, 1e15, 1e15, 1e30, 0.0, 0.0, 0.0], [2, 4]).unwrap().requires_grad_();
    let huge_b = Tensor::from_slice(&[1e14f32, 2e14, 1.1e15, 1.2e15, 0.0, 0.0, 1e-30, 0.0], [2, 4]).unwrap();
    let huge_c = Tensor::from_slice(&[1e15f32, 1e30], [2]).unwrap();
    for (name, l) in [
        ("nwd", nwd_loss(&huge_a, &huge_b, &huge_c).unwrap()),
        ("wiou_v1", wiou_v1_loss(&huge_a, &huge_b).unwrap()),
        ("wiou_v3", wiou_v3_loss(&huge_a, &huge_b, 1.9, 3.0, 0.5).unwrap()),
    ] {
        let vs = l.to_vec::<f32>().unwrap();
        assert!(vs.iter().all(|&v| v.is_finite() && v >= 0.0), "{name} extreme losses: {vs:?}");
        let g = l.sum_all().unwrap().backward().unwrap().get(&huge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|x| x.is_finite()), "{name} extreme grad non-finite: {g:?}");
    }

    // 2. Identical tiny, zero-area point, inverted, zero-width line, disjoint, identical normal.
    let edge_a = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny identical -> exact 0.0 loss & 0.0 grad
            5.0, 5.0, 5.0, 5.0, // zero-area point == point
            10.0, 12.0, 4.0, 6.0, // inverted
            1.0, 1.0, 1.0, 8.0, // identical zero-width vertical line
            0.0, 0.0, 4.0, 4.0, // disjoint
            0.0, 0.0, 10.0, 10.0, // identical to GT
        ],
        [6, 4],
    )
    .unwrap()
    .requires_grad_();
    let edge_b = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, 5.0, 5.0, 5.0, 5.0, 2.0, 3.0, 9.0, 11.0, 1.0, 1.0, 1.0, 8.0, 10.0,
            0.0, 14.0, 4.0, 0.0, 0.0, 10.0, 10.0,
        ],
        [6, 4],
    )
    .unwrap();
    for (name, l) in [
        ("nwd", nwd_loss_with_c(&edge_a, &edge_b, 12.8).unwrap()),
        ("wiou_v1", wiou_v1_loss(&edge_a, &edge_b).unwrap()),
        ("wiou_v3", wiou_v3_loss(&edge_a, &edge_b, 1.9, 3.0, 0.5).unwrap()),
    ] {
        let vals = l.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite() && *v >= 0.0), "{name} degenerate losses: {vals:?}");
        assert_eq!(vals[0], 0.0, "{name}: identical tiny 1e-12 box must have 0.0 loss, got {}", vals[0]);
        assert_eq!(vals[5], 0.0, "{name}: identical box must have 0.0 loss, got {}", vals[5]);
        let g = l.sum_all().unwrap().backward().unwrap().get(&edge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{name} degenerate grads: {g:?}");
        assert!(g[0..4].iter().all(|v| *v == 0.0), "{name} tiny identical grad: {:?}", &g[0..4]);
        assert!(g[20..24].iter().all(|v| *v == 0.0), "{name} identical grad: {:?}", &g[20..24]);
    }

    // 3. NaN / ±Inf poison in either b1 or b2 propagates NaN to both forward and backward.
    let ok = Tensor::from_slice(&[0.0f32, 0.0, 4.0, 4.0], [1, 4]).unwrap();
    for bad in [[f32::NAN, 0.0, 4.0, 4.0], [0.0, f32::INFINITY, 4.0, 4.0], [f32::NEG_INFINITY, 0.0, 4.0, 4.0]]
    {
        let bp = Tensor::from_slice(&bad, [1, 4]).unwrap().requires_grad_();
        for l in [
            nwd_loss_with_c(&bp, &ok, 12.8).unwrap(),
            wiou_v1_loss(&bp, &ok).unwrap(),
            wiou_v3_loss(&bp, &ok, 1.9, 3.0, 1.0).unwrap(),
        ] {
            assert!(l.to_vec::<f32>().unwrap()[0].is_nan());
            let g = l.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for b1={bad:?}, got {g:?}");
        }
        // Bad GT box b2 also poisons forward and backward wrt b1.
        let p_ok = ok.detach().requires_grad_();
        let bg = Tensor::from_slice(&bad, [1, 4]).unwrap();
        for l in [
            nwd_loss_with_c(&p_ok, &bg, 12.8).unwrap(),
            wiou_v1_loss(&p_ok, &bg).unwrap(),
            wiou_v3_loss(&p_ok, &bg, 1.9, 3.0, 1.0).unwrap(),
        ] {
            assert!(l.to_vec::<f32>().unwrap()[0].is_nan());
            let g = l.sum_all().unwrap().backward().unwrap().get(&p_ok).unwrap().to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for b2={bad:?}, got {g:?}");
        }
    }

    // 4. Empty [0, 4] and invalid parameters / shape mismatches.
    let empty = Tensor::zeros([0, 4]);
    let empty_c = Tensor::zeros([0]);
    assert_eq!(nwd_loss(&empty, &empty, &empty_c).unwrap().dims(), &[0]);
    assert_eq!(nwd_loss_with_c(&empty, &empty, 12.8).unwrap().dims(), &[0]);
    assert_eq!(wiou_v1_loss(&empty, &empty).unwrap().dims(), &[0]);
    assert_eq!(wiou_v3_loss(&empty, &empty, 1.9, 3.0, 1.0).unwrap().dims(), &[0]);

    for bad_c in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert!(nwd_loss_with_c(&ok, &ok, bad_c).is_err());
        assert!(nwd_loss(&ok, &ok, &Tensor::from_slice(&[bad_c], [1]).unwrap()).is_err());
    }
    assert!(nwd_loss(&ok, &ok, &Tensor::from_slice(&[1.0f32, 2.0], [2]).unwrap()).is_err());
    assert!(wiou_v1_loss(&ok, &Tensor::zeros([2, 4])).is_err());
}

/// Frozen-assignment finite-difference check isolating `NWD` and `WIoU v1` gradients through the DFL softmax
/// Jacobian (`cls_gain = 0`, `dfl_gain = 0`), plus end-to-end `detection_loss` and `LossConfig` validation
/// for `BoxLossKind::Nwd`, `BoxLossKind::WiouV1`, and `BoxLossKind::WiouV3`.
#[test]
fn nwd_and_wiou_dfl_fd_and_detection_loss() {
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(4.0, 5.0, 26.0, 27.0)], labels: vec![0] }];
    let reg_max = 4usize;
    let mut rng = folo_core::rng::Rng::seed(1203);
    let raw_bx: Vec<f32> = (0..4 * reg_max * a).map(|_| rng.uniform(-1.5, 1.5)).collect();
    let raw_cx: Vec<f32> = (0..2 * a).map(|_| rng.uniform(-1.0, 1.0)).collect();

    // Exact frozen-assignment DFL-logit FD check for BoxLossKind::Nwd.
    let cfg_nwd = LossConfig {
        box_gain: 7.5,
        cls_gain: 0.0,
        dfl_gain: 0.0,
        reg_max,
        box_loss: BoxLossKind::Nwd,
        nwd_c: 12.8,
        ..LossConfig::default()
    };
    let bx = Tensor::from_vec(raw_bx.clone(), [1, 4 * reg_max, a]).unwrap().requires_grad_();
    let cx = Tensor::from_vec(raw_cx.clone(), [1, 2, a]).unwrap();
    let asg = assign_batch(&bx, &cx, &anchors, &targets, &cfg_nwd).unwrap();
    assert!(asg.num_fg() > 0);
    let out = loss_with_assignment(&bx, &cx, &anchors, &asg, &cfg_nwd).unwrap();
    assert!(out.is_finite() && out.items.cls == 0.0 && out.items.dfl == 0.0 && out.mean_iou_loss.is_none());
    let grad = out.total.backward().unwrap().get(&bx).unwrap().to_vec::<f32>().unwrap();
    let fg_a = asg.anchor[0] as usize;
    let eps = 2e-3f32;
    for ch in 0..4 * reg_max {
        let idx = ch * a + fg_a;
        let mut bp = raw_bx.clone();
        let mut bm = raw_bx.clone();
        bp[idx] += eps;
        bm[idx] -= eps;
        let lp = loss_with_assignment(
            &Tensor::from_vec(bp, [1, 4 * reg_max, a]).unwrap(),
            &cx,
            &anchors,
            &asg,
            &cfg_nwd,
        )
        .unwrap()
        .total_value;
        let lm = loss_with_assignment(
            &Tensor::from_vec(bm, [1, 4 * reg_max, a]).unwrap(),
            &cx,
            &anchors,
            &asg,
            &cfg_nwd,
        )
        .unwrap()
        .total_value;
        let fd = (lp - lm) / (2.0 * eps);
        let tol = 1e-3 + 2e-2 * fd.abs().max(grad[idx].abs());
        assert!((grad[idx] - fd).abs() <= tol, "Nwd DFL ch={ch}: analytic {} vs fd {fd}", grad[idx]);
    }

    // Exact DFL-logit FD check for BoxLossKind::WiouV1 on a large target enclosing all predicted boxes
    // (so c2* is locally constant wrt logits and central FD matches the detached analytical gradient!).
    let big_targets = vec![ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, 32.0, 32.0)], labels: vec![0] }];
    let cfg_v1 = LossConfig {
        box_gain: 7.5,
        cls_gain: 0.0,
        dfl_gain: 0.0,
        reg_max,
        box_loss: BoxLossKind::WiouV1,
        ..LossConfig::default()
    };
    let bx_in = Tensor::from_vec(raw_bx.clone(), [1, 4 * reg_max, a]).unwrap().requires_grad_();
    let asg_in = assign_batch(&bx_in, &cx, &anchors, &big_targets, &cfg_v1).unwrap();
    assert!(asg_in.num_fg() > 0);
    // Keep only the single central foreground anchor whose predicted box lies strictly inside [0, 4]x[0, 4] grid.
    let inner_pos = (0..asg_in.num_fg())
        .find(|&k| {
            let p = anchors.points[asg_in.anchor[k] as usize];
            p[0] > 1.0 && p[0] < 3.0 && p[1] > 1.0 && p[1] < 3.0
        })
        .expect("central fg anchor");
    let single_asg = BatchAssignment {
        batch: asg_in.batch,
        num_anchors: asg_in.num_anchors,
        num_classes: asg_in.num_classes,
        image: vec![asg_in.image[inner_pos]],
        anchor: vec![asg_in.anchor[inner_pos]],
        label: vec![asg_in.label[inner_pos]],
        score: vec![asg_in.score[inner_pos]],
        target: vec![Xyxy::new(-32.0, -32.0, 64.0, 64.0)],
        inputs_finite: true,
    };
    let out_v1 = loss_with_assignment(&bx_in, &cx, &anchors, &single_asg, &cfg_v1).unwrap();
    let grad_v1 = out_v1.total.backward().unwrap().get(&bx_in).unwrap().to_vec::<f32>().unwrap();
    let fg_a_in = single_asg.anchor[0] as usize;
    for ch in 0..4 * reg_max {
        let idx = ch * a + fg_a_in;
        let mut bp = raw_bx.clone();
        let mut bm = raw_bx.clone();
        bp[idx] += eps;
        bm[idx] -= eps;
        let lp = loss_with_assignment(
            &Tensor::from_vec(bp, [1, 4 * reg_max, a]).unwrap(),
            &cx,
            &anchors,
            &single_asg,
            &cfg_v1,
        )
        .unwrap()
        .total_value;
        let lm = loss_with_assignment(
            &Tensor::from_vec(bm, [1, 4 * reg_max, a]).unwrap(),
            &cx,
            &anchors,
            &single_asg,
            &cfg_v1,
        )
        .unwrap()
        .total_value;
        let fd = (lp - lm) / (2.0 * eps);
        let tol = 1e-3 + 2e-2 * fd.abs().max(grad_v1[idx].abs());
        assert!((grad_v1[idx] - fd).abs() <= tol, "WiouV1 DFL ch={ch}: analytic {} vs fd {fd}", grad_v1[idx]);
    }

    // End-to-end detection_loss with reg_max = 2, 16 and nwd_weight = 0.0, 0.5 + serde round-trip.
    let anchors64 = make_anchors(64, 64, &[8, 16], 0.5);
    let a64 = anchors64.len();
    let targets64 = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 44.0, 48.0)], labels: vec![0] }];
    for kind in [BoxLossKind::Nwd, BoxLossKind::WiouV1, BoxLossKind::WiouV3] {
        for r in [2usize, 16] {
            for nwd_weight in [0.0f32, 0.5] {
                let mut rng2 = folo_core::rng::Rng::seed(505 + r as u64);
                let bx2 = Tensor::rand_uniform([1, 4 * r, a64], -2.0, 3.0, &mut rng2).requires_grad_();
                let cx2 = Tensor::rand_uniform([1, 2, a64], -3.0, 2.0, &mut rng2).requires_grad_();
                let cfg2 = LossConfig {
                    reg_max: r,
                    box_loss: kind,
                    nwd_c: 12.8,
                    nwd_weight,
                    ..LossConfig::default()
                };
                assert!(!device_loss_supported(&bx2, &cfg2));
                let out2 = detection_loss(&bx2, &cx2, &anchors64, &targets64, &cfg2).unwrap();
                assert!(out2.is_finite() && out2.num_fg > 0, "{kind:?} reg_max={r} nwd={nwd_weight}");
                if kind == BoxLossKind::WiouV3 {
                    assert!(out2.mean_iou_loss.is_some_and(|m| m.is_finite() && (0.0..=1.0).contains(&m)));
                } else {
                    assert!(out2.mean_iou_loss.is_none());
                }
                let g2 = out2.total.backward().unwrap();
                let gb2 = g2.get(&bx2).unwrap().to_vec::<f32>().unwrap();
                assert!(gb2.iter().all(|v| v.is_finite()) && gb2.iter().any(|v| v.abs() > 1e-6));
            }
        }
        let sc = LossConfig { box_loss: kind, nwd_c: 16.0, ..LossConfig::default() };
        assert!(sc.validate_params().is_ok());
        let json = serde_json::to_string(&sc).unwrap();
        assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), sc);
    }

    for bad in [0.0f32, -5.0, f32::NAN, f32::INFINITY] {
        let cfg = LossConfig { nwd_c: bad, ..LossConfig::default() };
        assert!(cfg.validate_params().is_err(), "nwd_c={bad} should fail");
    }
}
