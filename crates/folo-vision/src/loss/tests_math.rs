//! Unit tests for `S-AEIoU` and `CRPS-DFL`: `f64` central finite-difference gradient checks,
//! extreme & degenerate edge-case regressions, and config validation / serde round-trips.

use super::terms::{crps_dfl, saeiou_box_f64};
use super::*;
use crate::decode::make_anchors;

/// `f64` central finite-difference gradient check for `S-AEIoU` across overlapping, disjoint, high-aspect,
/// tiny, and scale-normalized (`> 1e6`) boxes.
#[test]
fn saeiou_f64_central_fd_grad_check() {
    let cases = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], 0.5, 2.0, 0.5),
        ([0.5, 0.5, 2.5, 8.5], [1.0, 1.0, 2.0, 9.0], 0.0, 2.0, 0.5),
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], 1.0, 3.0, 0.5), // disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3], 0.5, 2.0, 0.5), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7], 0.5, 2.0, 0.5), // > 1e6 scale-normalized
        ([0.0, 0.0, 2e6, 0.2], [0.0, 0.0, 1.5e6, 0.2], 0.5, 2.0, 0.5),   // thin large box > 1e6
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5], 0.7, 1.5, 0.25), // extreme aspect ratio
    ];
    for (idx, &(a, b, beta, kappa, delta0)) in cases.iter().enumerate() {
        let (loss, grad) = saeiou_box_f64(a, b, beta, kappa, delta0);
        assert!(loss.is_finite() && loss >= 0.0, "case {idx}: non-finite or negative loss {loss}");
        let mut h = [0.0f64; 4];
        for c in 0..4 {
            h[c] = 1e-6 * a[c].abs().max(b[c].abs()).max(1.0);
        }
        let eps_mag = (a[2] - a[0]).abs().min((a[3] - a[1]).abs()).max(1e-3);
        let h_thin = [
            (1e-6 * eps_mag).max(1e-8 * a[0].abs()),
            (1e-6 * eps_mag).max(1e-8 * a[1].abs()),
            (1e-6 * eps_mag).max(1e-8 * a[2].abs()),
            (1e-6 * eps_mag).max(1e-8 * a[3].abs()),
        ];
        for c in 0..4 {
            let eps = if c % 2 == 0 {
                h_thin[c].max(h[c].min(1e-5 * (a[2] - a[0]).abs().max(1.0)))
            } else {
                h_thin[c].max(1e-7 * (a[3] - a[1]).abs().max(1e-4))
            };
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (saeiou_box_f64(ap, b, beta, kappa, delta0).0
                - saeiou_box_f64(am, b, beta, kappa, delta0).0)
                / (2.0 * eps);
            let tol = 1e-10 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }
    // Continuity across the `m_dim = 1e6` scaling boundary even for thin boxes (`h = 0.2`).
    let l_lo = saeiou_box_f64([0.0, 0.0, 9.99999e5, 0.2], [0.0, 0.0, 7.5e5, 0.2], 0.5, 2.0, 0.5).0;
    let l_hi = saeiou_box_f64([0.0, 0.0, 1.000001e6, 0.2], [0.0, 0.0, 7.5e5, 0.2], 0.5, 2.0, 0.5).0;
    assert!((l_lo - l_hi).abs() < 1e-5, "threshold jump across 1e6: {l_lo} vs {l_hi}");
}

/// `S-AEIoU` extreme & degenerate edge cases: `1e15`, `1e-12`, inverted/zero-area boxes, `NaN`, `±Inf`,
/// and end-to-end `detection_loss` with `reg_max = 2` and `16`.
#[test]
fn saeiou_extreme_and_degenerate_edge_cases() {
    // 1. Scale-normalized huge boxes (`1e15`) never overflow `f32` (`w*h = 1e30` normalized to `1.0`).
    let huge_a = Tensor::from_slice(&[0.0f32, 0.0, 1e15, 1e15], [1, 4]).unwrap().requires_grad_();
    let huge_b = Tensor::from_slice(&[1e14f32, 2e14, 1.1e15, 1.2e15], [1, 4]).unwrap();
    let l_huge = saeiou_loss(&huge_a, &huge_b, 0.5, 2.0, 0.5).unwrap();
    let lv = l_huge.to_vec::<f32>().unwrap()[0];
    assert!(lv.is_finite() && lv > 0.0 && lv < 2.0, "huge 1e15 S-AEIoU loss out of range: {lv}");
    let g_huge = l_huge.sum_all().unwrap().backward().unwrap().get(&huge_a).unwrap().to_vec::<f32>().unwrap();
    assert!(g_huge.iter().all(|v| v.is_finite()), "huge 1e15 S-AEIoU grad non-finite: {g_huge:?}");

    // 2. Tiny `1e-12` boxes, zero-area boxes, and inverted boxes stay finite in forward and backward.
    let edge_a = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny 1e-12
            5.0, 5.0, 5.0, 5.0, // zero-area point
            10.0, 12.0, 4.0, 6.0, // inverted pred
            1.0, 1.0, 1.0, 8.0, // zero-width line
            0.0, 0.0, 10.0, 10.0, // identical to GT
        ],
        [5, 4],
    )
    .unwrap()
    .requires_grad_();
    let edge_b = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny 1e-12
            5.0, 5.0, 8.0, 8.0, // normal GT
            2.0, 3.0, 9.0, 11.0, // normal GT
            0.0, 0.0, 5.0, 5.0, // normal GT
            0.0, 0.0, 10.0, 10.0, // identical GT
        ],
        [5, 4],
    )
    .unwrap();
    let l_edge = saeiou_loss(&edge_a, &edge_b, 0.5, 2.0, 0.5).unwrap();
    let l_vals = l_edge.to_vec::<f32>().unwrap();
    assert!(l_vals.iter().all(|v| v.is_finite() && *v >= 0.0), "degenerate S-AEIoU losses: {l_vals:?}");
    assert!(l_vals[4] < 1e-6, "identical box must have ~0 S-AEIoU loss, got {}", l_vals[4]);
    let g_edge = l_edge.sum_all().unwrap().backward().unwrap().get(&edge_a).unwrap().to_vec::<f32>().unwrap();
    assert!(g_edge.iter().all(|v| v.is_finite()), "degenerate S-AEIoU grads: {g_edge:?}");

    // 3. `NaN` and `±Inf` poison both forward and backward.
    for bad in [[f32::NAN, 0.0, 4.0, 4.0], [0.0, f32::INFINITY, 4.0, 4.0], [f32::NEG_INFINITY, 0.0, 4.0, 4.0]]
    {
        let bp = Tensor::from_slice(&bad, [1, 4]).unwrap().requires_grad_();
        let bg = Tensor::from_slice(&[0.0f32, 0.0, 4.0, 4.0], [1, 4]).unwrap();
        let lb = saeiou_loss(&bp, &bg, 0.5, 2.0, 0.5).unwrap();
        assert!(lb.to_vec::<f32>().unwrap()[0].is_nan(), "expected NaN loss for {bad:?}");
        let gb = lb.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
        assert!(gb.iter().all(|v| v.is_nan()), "expected NaN grad for {bad:?}, got {gb:?}");
    }

    // 4. End-to-end `detection_loss` with `BoxLossKind::Saeiou` (+ optional NWD) at `reg_max = 2` and `16`.
    let anchors = make_anchors(64, 64, &[8, 16], 0.5);
    let a = anchors.len();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 44.0, 48.0)], labels: vec![0] }];
    for reg_max in [2usize, 16] {
        for nwd_weight in [0.0f32, 0.5] {
            let mut rng = folo_core::rng::Rng::seed(101 + reg_max as u64);
            let bx = Tensor::rand_uniform([1, 4 * reg_max, a], -2.0, 3.0, &mut rng).requires_grad_();
            let cx = Tensor::rand_uniform([1, 2, a], -3.0, 2.0, &mut rng).requires_grad_();
            let cfg =
                LossConfig { reg_max, box_loss: BoxLossKind::Saeiou, nwd_weight, ..LossConfig::default() };
            let out = detection_loss(&bx, &cx, &anchors, &targets, &cfg).unwrap();
            assert!(out.is_finite() && out.num_fg > 0, "reg_max={reg_max} nwd={nwd_weight}");
            let g = out.total.backward().unwrap();
            let gb = g.get(&bx).unwrap().to_vec::<f32>().unwrap();
            assert!(gb.iter().all(|v| v.is_finite()) && gb.iter().any(|v| v.abs() > 1e-6));
        }
    }
}

/// Independent `f64` oracle for `CRPS-DFL` excess energy distance on a single side.
fn crps_side_oracle_f64(logits: &[f64], j: usize, wl: f64) -> f64 {
    let r = logits.len();
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - m).exp()).collect();
    let z: f64 = exps.iter().sum();
    let p: Vec<f64> = exps.iter().map(|&e| e / z).collect();
    let mut cdf = 0.0;
    let mut sum = 0.0;
    for (k, &pk) in p.iter().enumerate().take(r - 1) {
        cdf += pk;
        let target_cdf = if k < j {
            0.0
        } else if k == j {
            wl
        } else {
            1.0
        };
        let e = cdf - target_cdf;
        sum += e * e;
    }
    sum
}

/// `f64` central finite-difference gradient check and boundary/extreme cases for `CRPS-DFL`.
#[test]
fn crps_dfl_f64_fd_and_edge_cases() {
    // 1. Finite-difference gradient check for `reg_max = 2` and `16`, including clamp boundaries `dt = 0.0`
    //    (`j = 0, wl = 1.0`) and `dt = R - 1 - 0.01` (`j = R - 2, wl = 0.01`), plus interior bins.
    for r in [2usize, 16] {
        let mut rng = folo_core::rng::Rng::seed(2026 + r as u64);
        let raw: Vec<f32> = (0..4 * r).map(|_| rng.uniform(-3.0, 3.0)).collect();
        let left_bins = [0usize, r - 2, (r - 2) / 2, (r - 2).min(1)];
        let left_w = [1.0f32, 0.01, 0.73, 0.35];
        let side_wt = [0.25f32, 0.25, 0.25, 0.25];
        let (crps_w, tss, dfl_gain) = (0.5f32, 1.75f32, 1.5f32);

        let logits = Tensor::from_vec(raw.clone(), [1, 4, r]).unwrap().requires_grad_();
        let out = crps_dfl(&logits, &left_bins, &left_w, &side_wt, crps_w, tss, dfl_gain).unwrap();
        let loss_val = out.item::<f32>().unwrap() as f64;
        let grad = out.backward().unwrap().get(&logits).unwrap().to_vec::<f32>().unwrap();

        let scale = (crps_w as f64 * dfl_gain as f64) / tss as f64;
        let mut want_loss = 0.0;
        for side in 0..4 {
            let row: Vec<f64> = raw[side * r..(side + 1) * r].iter().map(|&v| v as f64).collect();
            want_loss += scale
                * (side_wt[side] as f64)
                * crps_side_oracle_f64(&row, left_bins[side], left_w[side] as f64);
            let eps = 1e-6;
            for k in 0..r {
                let mut rp = row.clone();
                let mut rm = row.clone();
                rp[k] += eps;
                rm[k] -= eps;
                let fd = scale
                    * (side_wt[side] as f64)
                    * (crps_side_oracle_f64(&rp, left_bins[side], left_w[side] as f64)
                        - crps_side_oracle_f64(&rm, left_bins[side], left_w[side] as f64))
                    / (2.0 * eps);
                let g = grad[side * r + k] as f64;
                let tol = 1e-5 + 1e-4 * fd.abs();
                assert!(
                    (g - fd).abs() <= tol,
                    "R={r} side={side} k={k}: analytic {g} vs fd {fd} (diff {})",
                    (g - fd).abs()
                );
            }
        }
        assert!(
            (loss_val - want_loss).abs() <= 1e-5 * (1.0 + want_loss),
            "R={r}: loss {loss_val} vs oracle {want_loss}"
        );
    }

    // 2. Exact target distribution `p == q` gives ~0 loss and ~0 gradient.
    let mut exact = vec![-60.0f32; 4 * 16];
    for side in 0..4 {
        exact[side * 16 + 3] = (0.7f32).ln();
        exact[side * 16 + 4] = (0.3f32).ln();
    }
    let logits_ex = Tensor::from_vec(exact, [1, 4, 16]).unwrap().requires_grad_();
    let out_ex = crps_dfl(&logits_ex, &[3; 4], &[0.7; 4], &[0.25; 4], 1.0, 1.0, 1.5).unwrap();
    assert!(out_ex.item::<f32>().unwrap().abs() < 1e-7);
    let g_ex = out_ex.backward().unwrap().get(&logits_ex).unwrap().to_vec::<f32>().unwrap();
    assert!(g_ex.iter().all(|v| v.abs() < 1e-6));

    // 3. Extreme `-1e38` logits stay completely finite.
    let mut extreme = vec![-1e38f32; 4 * 16];
    for side in 0..4 {
        extreme[side * 16 + 15] = 0.0;
    }
    let logits_ext = Tensor::from_vec(extreme, [1, 4, 16]).unwrap().requires_grad_();
    let out_ext =
        crps_dfl(&logits_ext, &[0, 7, 14, 0], &[1.0, 0.5, 0.01, 0.25], &[0.25; 4], 0.5, 1.0, 1.5).unwrap();
    assert!(out_ext.item::<f32>().unwrap().is_finite());
    let g_ext = out_ext.backward().unwrap().get(&logits_ext).unwrap().to_vec::<f32>().unwrap();
    assert!(g_ext.iter().all(|v| v.is_finite()));

    // 4. Partial `NaN` in direct `crps_dfl` call propagates `NaN` to both loss and gradient.
    let mut nan_row = vec![0.0f32; 4 * 16];
    nan_row[2] = f32::NAN;
    let logits_nan = Tensor::from_vec(nan_row, [1, 4, 16]).unwrap().requires_grad_();
    let out_nan = crps_dfl(&logits_nan, &[3; 4], &[0.5; 4], &[0.25; 4], 0.5, 1.0, 1.5).unwrap();
    assert!(out_nan.item::<f32>().unwrap().is_nan());
    let g_nan = out_nan.backward().unwrap().get(&logits_nan).unwrap().to_vec::<f32>().unwrap();
    assert!(g_nan[..16].iter().all(|v| v.is_nan()), "expected NaN grad on NaN side: {:?}", &g_nan[..16]);
}

/// Validation and serde round-trip for `S-AEIoU` and `CRPS-DFL`.
#[test]
fn saeiou_and_crps_config_validation_and_serde() {
    let def_json = serde_json::to_string(&LossConfig::default()).unwrap();
    assert!(def_json.contains("dfl_crps_weight"), "{def_json}");

    let cfg = LossConfig {
        box_loss: BoxLossKind::Saeiou,
        dfl_crps_weight: 0.25,
        cls_loss: ClsLossKind::Vfl,
        ..LossConfig::default()
    };
    assert!(cfg.validate_params().is_ok());
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("\"saeiou\"") && json.contains("\"vfl\""), "{json}");
    assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), cfg);

    for bad in [
        LossConfig { dfl_crps_weight: -0.01, ..LossConfig::default() },
        LossConfig { dfl_crps_weight: f32::NAN, ..LossConfig::default() },
        LossConfig { dfl_crps_weight: f32::INFINITY, ..LossConfig::default() },
    ] {
        assert!(bad.validate_params().is_err(), "{bad:?} should fail validate_params");
    }
}
