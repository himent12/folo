//! Unit tests for Powerful-IoU v1 (`piou_loss`) and v2 (`piou2_loss`):
//! `f64` central finite-difference gradient checks across general, nested, disjoint, collinear,
//! degenerate, zero-area, inverted, and extreme-scale boxes; non-monotonic attention focusing curve
//! `u(lambda * q)` verification; scale invariance across 24 orders of magnitude; `Tensor`-level
//! `grad_check_with`; optimization convergence from disjoint boxes; and `NaN`/`±Inf`/invalid-param safety.

use super::terms::{
    piou_attention, piou_attention_f64, piou_attention_from_p_f64, piou_box_f64, piou_loss, piou2_box_f64,
    piou2_loss,
};
use folo_core::Tensor;
use folo_core::testing::{GradCheck, grad_check_with};

/// `f64` central finite-difference gradient check for `PIoU v1` across general, nested (both directions),
/// concentric, disjoint, collinear, tiny, huge, extreme aspect-ratio, zero-area, and inverted boxes, plus
/// 24-decade scale invariance and direct `Tensor` `grad_check_with`.
#[test]
fn piou_v1_f64_central_fd_and_tensor_grad_check() {
    let cases: [([f64; 4], [f64; 4]); 12] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0]), // general overlapping
        ([3.0, 3.5, 7.5, 8.0], [1.0, 1.0, 10.0, 10.0]),   // predicted strictly nested inside GT
        ([0.5, 0.2, 12.0, 11.5], [3.0, 2.5, 8.0, 7.5]),   // GT strictly nested inside predicted
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5]),   // concentric different size
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0]), // disjoint
        ([0.0, 2.0, 4.0, 6.0], [7.0, 2.0, 11.0, 6.0]),    // horizontally collinear (y edges aligned)
        ([2.0, 0.0, 6.0, 4.0], [2.0, 8.0, 6.0, 14.0]),    // vertically collinear (x edges aligned)
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7]), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5]), // extreme aspect ratio
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0]),  // inverted predicted box
        ([5.0, 5.0, 5.0, 5.0], [2.0, 3.0, 8.0, 9.0]),     // zero-area point vs valid GT
    ];

    for (idx, &(a, b)) in cases.iter().enumerate() {
        let (loss, grad, p) = piou_box_f64(a, b);
        assert!(loss.is_finite() && (0.0..=2.0).contains(&loss), "piou case {idx}: bad loss {loss}");
        assert!(p.is_finite() && p >= 0.0, "piou case {idx}: bad P {p}");
        for c in 0..4 {
            let span = if c % 2 == 0 { (b[2] - b[0]).abs() } else { (b[3] - b[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (piou_box_f64(ap, b).0 - piou_box_f64(am, b).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "piou case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }

    // Strict scale invariance across 24 orders of magnitude (`1e-12` vs `1.0` vs `1e12`).
    let (l1, _, p1) = piou_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.7, 1.8, 3.1]);
    let (l_tiny, _, p_tiny) = piou_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.7e-12, 1.8e-12, 3.1e-12]);
    let (l_huge, _, p_huge) = piou_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.7e12, 1.8e12, 3.1e12]);
    assert!((l1 - l_tiny).abs() < 1e-12 && (p1 - p_tiny).abs() < 1e-12, "tiny scale mismatch");
    assert!((l1 - l_huge).abs() < 1e-12 && (p1 - p_huge).abs() < 1e-12, "huge scale mismatch");

    // Verify exact formula directly on a hand-checked pair.
    let a = [1.0f64, 2.0, 5.0, 6.0];
    let b = [2.0f64, 0.0, 6.0, 4.0]; // w_t = 4, h_t = 4
    // dw1 = 1, dw2 = 1 -> 2/4 = 0.5; dh1 = 2, dh2 = 2 -> 4/4 = 1.0; P = 0.25 * 1.5 = 0.375
    // inter = 3 * 2 = 6; union = 16 + 16 - 6 = 26; 1 - IoU = 20 / 26
    let want_p = 0.375f64;
    let want_l = (20.0 / 26.0) + (1.0 - (-want_p * want_p).exp());
    let (got_l, _, got_p) = piou_box_f64(a, b);
    assert!((got_p - want_p).abs() < 1e-14);
    assert!((got_l - want_l).abs() < 1e-14);

    // Tensor-level grad_check_with on general, nested, and disjoint boxes.
    let tb1 = Tensor::from_slice(&[2.3f32, 3.1, 10.7, 12.4, 3.0, 3.5, 7.5, 8.0, 0.0, 0.0, 2.5, 2.5], [3, 4])
        .unwrap();
    let tb2 =
        Tensor::from_slice(&[3.0f32, 2.5, 11.2, 13.0, 1.0, 1.0, 10.0, 10.0, 4.0, 5.0, 8.0, 9.0], [3, 4])
            .unwrap();
    grad_check_with(
        |inputs| piou_loss(&inputs[0], &tb2)?.sum_all(),
        std::slice::from_ref(&tb1),
        &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 12, seed: 31 },
    )
    .unwrap();
}

/// `PIoU v2` non-monotonic attention focusing curve `u(lambda * q)`, exact detached gradient identity
/// `grad_piou2 == u(lambda * q) * grad_piou1`, frozen-`u*` central finite-difference verification across
/// all box configurations, and scale invariance.
#[test]
fn piou_v2_attention_focusing_and_detached_fd_grad_check() {
    let lambda = 1.3f64;
    // Peak of u(x) = 3 * x * exp(-x²) occurs at x = 1 / sqrt(2) with peak value 3 / sqrt(2 * e).
    let q_peak = 1.0 / (lambda * std::f64::consts::SQRT_2);
    let p_peak = -q_peak.ln();
    let u_peak = piou_attention_f64(q_peak, lambda);
    let want_peak = 3.0 / (2.0 * std::f64::consts::E).sqrt();
    assert!((u_peak - want_peak).abs() < 1e-14, "u_peak {u_peak} vs {want_peak}");
    assert!((piou_attention_from_p_f64(p_peak, lambda) - want_peak).abs() < 1e-14);
    // `piou_attention` and `piou_attention_f64` agree on the exact same quality domain `q in [0, 1]`.
    for q in [0.0f32, 0.25, q_peak as f32, 0.75, 1.0] {
        assert!((piou_attention(q, 1.3) as f64 - piou_attention_f64(q as f64, 1.3)).abs() < 1e-6);
    }
    // Non-monotonic: peak strictly exceeds both high-quality (P -> 0, q -> 1) and low-quality (P = 3).
    let u_high_q = piou_attention_from_p_f64(0.0, lambda);
    let u_low_q = piou_attention_from_p_f64(3.0, lambda);
    assert!(
        u_peak > u_high_q && u_peak > u_low_q,
        "attention must be non-monotonic: {u_high_q}, {u_peak}, {u_low_q}"
    );
    // Extreme outlier decays smoothly to 0 without NaN/Inf.
    assert_eq!(piou_attention_from_p_f64(1e6, lambda), 0.0);
    assert_eq!(piou_attention_from_p_f64(f64::INFINITY, lambda), 0.0);
    assert_eq!(piou_attention_f64(0.0, lambda), 0.0);
    assert!(piou_attention_f64(-0.1, lambda).is_nan());
    assert!(piou_attention_f64(1.1, lambda).is_nan());
    assert!((piou_attention(q_peak as f32, 1.3) - want_peak as f32).abs() < 1e-6);

    let cases: [([f64; 4], [f64; 4]); 8] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0]),
        ([3.0, 3.5, 7.5, 8.0], [1.0, 1.0, 10.0, 10.0]),
        ([0.5, 0.2, 12.0, 11.5], [3.0, 2.5, 8.0, 7.5]),
        ([0.0, 0.0, 2.0, 2.0], [5.0, 6.0, 9.0, 10.0]),
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]),
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7]),
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0]),
        ([5.0, 5.0, 5.0, 5.0], [2.0, 3.0, 8.0, 9.0]),
    ];

    for &lam in &[0.8f64, 1.3, 2.0] {
        for (idx, &(a, b)) in cases.iter().enumerate() {
            let (l1, g1, p) = piou_box_f64(a, b);
            let (l2, g2, u) = piou2_box_f64(a, b, lam);
            assert!(l2.is_finite() && l2 >= 0.0, "piou2 case {idx} lam {lam}: bad loss {l2}");
            assert!((u - piou_attention_from_p_f64(p, lam)).abs() < 1e-15);
            assert!((l2 - u * l1).abs() <= 1e-14 + 1e-14 * l2.abs());
            for c in 0..4 {
                assert!((g2[c] - u * g1[c]).abs() <= 1e-14 + 1e-14 * g2[c].abs());
                // Frozen-u* central finite difference matches analytical gradient.
                let span = if c % 2 == 0 { (b[2] - b[0]).abs() } else { (b[3] - b[1]).abs() };
                let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
                let mut ap = a;
                let mut am = a;
                ap[c] += eps;
                am[c] -= eps;
                let fd = u * (piou_box_f64(ap, b).0 - piou_box_f64(am, b).0) / (2.0 * eps);
                let tol = 1e-9 + 1e-5 * fd.abs().max(g2[c].abs());
                assert!(
                    (g2[c] - fd).abs() <= tol,
                    "piou2 case {idx} lam {lam} coord {c}: analytic {} vs frozen-u FD {fd}",
                    g2[c]
                );
            }
        }
    }

    // Verify Tensor-level backward matches piou2_box_f64 and pulls disjoint boxes strictly toward GT.
    // Far disjoint box [0,0,2,2] vs [6,6,10,10] receives strictly restoring (negative) gradients on all 4 coords.
    let far_pred = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 2.0], [1, 4]).unwrap().requires_grad_();
    let gt = Tensor::from_slice(&[6.0f32, 6.0, 10.0, 10.0], [1, 4]).unwrap();
    let g_far = piou2_loss(&far_pred, &gt, 1.3)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&far_pred)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        g_far.iter().all(|&v| v.is_finite() && v < 0.0),
        "far disjoint PIoU2 must pull toward GT: {g_far:?}"
    );

    // Disjoint box [2.5, 2.5, 5.5, 5.5] (IoU == 0) converges cleanly onto [6, 6, 10, 10] under PIoU2.
    let mut coords = [2.5f32, 2.5, 5.5, 5.5];
    let init_l1 = piou_box_f64([2.5, 2.5, 5.5, 5.5], [6.0, 6.0, 10.0, 10.0]).0;
    for step in 0..300 {
        let pred = Tensor::from_slice(&coords, [1, 4]).unwrap().requires_grad_();
        let loss = piou2_loss(&pred, &gt, 1.3).unwrap();
        let g = loss.sum_all().unwrap().backward().unwrap().get(&pred).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()));
        let (_, g1, _) = piou_box_f64(
            [coords[0] as f64, coords[1] as f64, coords[2] as f64, coords[3] as f64],
            [6.0, 6.0, 10.0, 10.0],
        );
        // PIoU2 gradient must always align strictly positively with PIoU1 gradient (descent direction).
        let dot: f64 = g.iter().zip(&g1).map(|(&a, &b)| (a as f64) * b).sum();
        assert!(dot >= 0.0, "step {step}: PIoU2 must never oppose PIoU1 descent direction (dot={dot})");
        let lr = 1.0f32 / (1.0 + 0.05 * step as f32);
        for c in 0..4 {
            coords[c] -= lr * g[c];
        }
    }
    let final_l1 = piou_box_f64(
        [coords[0] as f64, coords[1] as f64, coords[2] as f64, coords[3] as f64],
        [6.0, 6.0, 10.0, 10.0],
    )
    .0;
    assert!(
        final_l1 < 0.05 && final_l1 < init_l1,
        "disjoint box should converge close to GT under PIoU2: init {init_l1}, final {final_l1}, coords {coords:?}"
    );
}

/// Edge cases for `piou_loss` and `piou2_loss`: identical (including tiny `1e-12`), zero-area point/line,
/// degenerate GT, inverted boxes, huge `1e15` & extreme-ratio boxes, empty `[0, 4]`, `NaN`/`±Inf` poison,
/// and invalid `lambda` rejection.
#[test]
fn piou_edge_cases_and_validation() {
    let edge_a = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny identical -> exact 0.0 loss & 0.0 grad
            5.0, 5.0, 5.0, 5.0, // zero-area point == point
            10.0, 12.0, 4.0, 6.0, // inverted prediction
            1.0, 1.0, 1.0, 8.0, // zero-width vertical line vs line
            0.0, 0.0, 1e15, 1e15, // huge boxes
            1e30, 0.0, 0.0, 0.0, // extreme inverted vs tiny target (huge P)
            0.0, 0.0, 10.0, 10.0, // identical valid box
            2.0, 2.0, 8.0, 8.0, // valid pred vs degenerate zero-area GT
        ],
        [8, 4],
    )
    .unwrap()
    .requires_grad_();
    let edge_b = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny identical
            5.0, 5.0, 5.0, 5.0, // zero-area point
            2.0, 3.0, 9.0, 11.0, // valid GT
            1.0, 1.0, 1.0, 8.0, // zero-width vertical line
            1e14, 2e14, 1.1e15, 1.2e15, // huge shifted GT
            0.0, 0.0, 1e-10, 1e-10, // tiny target causing huge P >> 30
            0.0, 0.0, 10.0, 10.0, // identical GT
            5.0, 5.0, 5.0, 5.0, // zero-area GT
        ],
        [8, 4],
    )
    .unwrap();

    for (name, l) in [
        ("piou", piou_loss(&edge_a, &edge_b).unwrap()),
        ("piou2", piou2_loss(&edge_a, &edge_b, 1.3).unwrap()),
    ] {
        let vals = l.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite() && *v >= 0.0), "{name} edge losses: {vals:?}");
        assert_eq!(vals[0], 0.0, "{name}: tiny identical loss must be 0.0, got {}", vals[0]);
        assert_eq!(vals[6], 0.0, "{name}: identical loss must be 0.0, got {}", vals[6]);
        let g = l.sum_all().unwrap().backward().unwrap().get(&edge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{name} edge grads non-finite: {g:?}");
        assert!(g[0..4].iter().all(|v| *v == 0.0), "{name}: tiny identical grad must be 0: {:?}", &g[0..4]);
        assert!(g[24..28].iter().all(|v| *v == 0.0), "{name}: identical grad must be 0: {:?}", &g[24..28]);
    }

    // NaN / ±Inf poison both forward and backward cleanly.
    for bad in [[f32::NAN, 0.0, 4.0, 4.0], [0.0, f32::INFINITY, 4.0, 4.0], [f32::NEG_INFINITY, 0.0, 4.0, 4.0]]
    {
        let bp = Tensor::from_slice(&bad, [1, 4]).unwrap().requires_grad_();
        let bg = Tensor::from_slice(&[0.0f32, 0.0, 4.0, 4.0], [1, 4]).unwrap();
        for l in [piou_loss(&bp, &bg).unwrap(), piou2_loss(&bp, &bg, 1.3).unwrap()] {
            assert!(l.to_vec::<f32>().unwrap()[0].is_nan());
            let g = l.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for {bad:?}, got {g:?}");
        }
    }

    // Empty [0, 4] and invalid lambda / shape checks.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(piou_loss(&empty, &empty).unwrap().dims(), &[0]);
    assert_eq!(piou2_loss(&empty, &empty, 1.3).unwrap().dims(), &[0]);
    let b = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 2.0], [1, 4]).unwrap();
    for bad_lam in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(piou2_loss(&b, &b, bad_lam).is_err(), "piou2_loss should reject lambda={bad_lam}");
    }
    // Extreme huge finite lambda should not overflow or produce NaN.
    let extreme =
        piou2_loss(&b, &Tensor::from_slice(&[1.0f32, 1.0, 3.0, 3.0], [1, 4]).unwrap(), 1e30).unwrap();
    assert!(extreme.to_vec::<f32>().unwrap()[0].is_finite());
}
