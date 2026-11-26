//! Unit tests for Seesaw Loss ([`seesaw_loss`], [`SeesawLoss`]) and Gaussian-box regression losses
//! ([`prob_iou`], [`prob_iou_loss`], [`prob_iou_l2_loss`], [`gwd_loss`], [`kld_loss`]):
//! exact `f64` central finite-difference gradient checks (including detached vs differentiable Seesaw
//! compensation), `Tensor`-level `grad_check_with`, scale invariance across 20+ orders of magnitude,
//! small/disjoint non-vanishing gradients, extreme-logit stability, stateful count accumulation, and
//! error validation.

use folo_core::Tensor;
use folo_core::testing::{GradCheck, grad_check_with};

use super::terms::{gwd_box_f64, kld_box_f64, prob_iou_l1_box_f64, prob_iou_l2_box_f64, seesaw_row_f64};
use super::*;

/// Frozen-`S*` Seesaw forward in `f64` matching Wang et al. (CVPR 2021) Eq. 5 detached compensation weights.
fn seesaw_frozen_weights_f64(z: &[f64], target: usize, log_s: &[f64]) -> f64 {
    let mut adj = vec![f64::NEG_INFINITY; z.len()];
    for j in 0..z.len() {
        if log_s[j].is_finite() {
            adj[j] = z[j] + log_s[j];
        }
    }
    let m = adj.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + adj.iter().filter(|v| v.is_finite()).map(|&v| (v - m).exp()).sum::<f64>().ln();
    lse - z[target]
}

#[test]
fn seesaw_f64_central_fd_mitigation_and_compensation() {
    let counts = [1000.0, 50.0, 5.0, 0.0];
    let z = [1.2, -0.4, 2.3, 0.8]; // class 2 > class 0 -> triggers compensation when target == 0

    // 1. Both detached (Eq. 5) and differentiable compensation modes verified against central FD.
    for target in 0..4 {
        for &(p, q, eps) in &[(0.8, 2.0, 1e-6), (0.8, 0.0, 1e-6), (0.0, 2.0, 0.0), (1.0, 1.5, 0.0)] {
            // End-to-end differentiable mode: direct central FD on seesaw_row_f64.
            let (loss_diff, grad_diff) = seesaw_row_f64(&z, target, &counts, p, q, eps, false);
            assert!(loss_diff.is_finite() && loss_diff >= 0.0);
            let sum_g: f64 = grad_diff.iter().sum();
            assert!(sum_g.abs() < 1e-13, "gradient sum must be 0 (translation invariance), got {sum_g:e}");
            for j in 0..z.len() {
                let h = 1e-6;
                let mut zp = z;
                let mut zm = z;
                zp[j] += h;
                zm[j] -= h;
                let fd = (seesaw_row_f64(&zp, target, &counts, p, q, eps, false).0
                    - seesaw_row_f64(&zm, target, &counts, p, q, eps, false).0)
                    / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad_diff[j].abs());
                assert!(
                    (grad_diff[j] - fd).abs() <= tol,
                    "seesaw diff t={target} p={p} q={q} j={j}: ana={} fd={fd}",
                    grad_diff[j]
                );
            }

            // Detached Eq. 5 mode: freeze log(S*) at z and verify central FD matches analytical gradient.
            let (loss_det, grad_det) = seesaw_row_f64(&z, target, &counts, p, q, eps, true);
            assert!((loss_det - loss_diff).abs() < 1e-14, "forward values must be identical");
            let mut log_s = vec![0.0f64; z.len()];
            for j in 0..z.len() {
                if j == target {
                    continue;
                }
                let log_m = if p == 0.0 || counts[j] >= counts[target] {
                    0.0
                } else if counts[j] == 0.0 {
                    if eps == 0.0 { f64::NEG_INFINITY } else { eps.ln() }
                } else {
                    let raw = p * (counts[j].ln() - counts[target].ln());
                    if eps > 0.0 { raw.max(eps.ln()) } else { raw }
                };
                let log_c = if q > 0.0 && z[j] > z[target] { q * (z[j] - z[target]) } else { 0.0 };
                log_s[j] = log_m + log_c;
            }
            for j in 0..z.len() {
                let h = 1e-6;
                let mut zp = z;
                let mut zm = z;
                zp[j] += h;
                zm[j] -= h;
                let fd = (seesaw_frozen_weights_f64(&zp, target, &log_s)
                    - seesaw_frozen_weights_f64(&zm, target, &log_s))
                    / (2.0 * h);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad_det[j].abs());
                assert!(
                    (grad_det[j] - fd).abs() <= tol,
                    "seesaw det t={target} p={p} q={q} j={j}: ana={} fd={fd}",
                    grad_det[j]
                );
            }
        }
    }

    // 2. Verify mitigation suppresses rare-class negative gradient when frequent class is GT, and
    //    compensation amplifies negative gradient when a false-positive class outscores GT.
    let (_, g_plain) = seesaw_row_f64(&[1.0, 0.5, 0.5], 0, &[1000.0, 1000.0, 10.0], 0.0, 0.0, 1e-6, true);
    let (_, g_mit) = seesaw_row_f64(&[1.0, 0.5, 0.5], 0, &[1000.0, 1000.0, 10.0], 0.8, 0.0, 1e-6, true);
    assert!(g_mit[2] < 0.1 * g_plain[2], "rare class 2 negative gradient must be strongly mitigated");

    let (_, g_nocomp) = seesaw_row_f64(&[0.5, 2.0, 0.0], 0, &[1000.0, 10.0, 1000.0], 0.8, 0.0, 1e-6, true);
    let (_, g_comp) = seesaw_row_f64(&[0.5, 2.0, 0.0], 0, &[1000.0, 10.0, 1000.0], 0.8, 2.0, 1e-6, true);
    assert!(g_comp[1] > g_nocomp[1], "misclassified class 1 must receive compensated penalty");

    // 3. Extreme logits & huge shared offsets remain finite, cancellation-free, and shift-invariant.
    let (l_ext, g_ext) =
        seesaw_row_f64(&[500.0, -500.0, 498.0], 0, &[100.0, 10.0, 1.0], 0.8, 2.0, 1e-6, true);
    assert!(l_ext.is_finite() && l_ext >= 0.0 && g_ext.iter().all(|v| v.is_finite()));

    let (l_off, g_off) = seesaw_row_f64(&[1e20, 1e20], 0, &[100.0, 1.0], 1.0, 2.0, 1e-6, true);
    let want_l_off = (0.01f64).ln_1p();
    assert!(
        (l_off - want_l_off).abs() < 1e-14,
        "large shared offset lost mitigation: {l_off} vs {want_l_off}"
    );
    assert!((g_off[1] - 0.01 / 1.01).abs() < 1e-14);
    assert!((g_off[0] + g_off[1]).abs() < 1e-15);

    // 4. Single-class (`C == 1`) and non-finite logit propagation.
    let (l_one, g_one) = seesaw_row_f64(&[2.5], 0, &[10.0], 0.8, 2.0, 1e-6, true);
    assert_eq!((l_one, g_one), (0.0, vec![0.0]));
    let (l_nan, g_nan) = seesaw_row_f64(&[f64::NAN, 1.0], 0, &[10.0, 10.0], 0.8, 2.0, 1e-6, true);
    assert!(l_nan.is_nan() && g_nan.iter().all(|v| v.is_nan()));
}

#[test]
fn seesaw_tensor_grad_check_and_stateful_tracker() {
    let logits = Tensor::from_slice(&[1.4f32, -0.3, 0.6, -0.8, 0.9, 1.7], [2, 3]).unwrap();
    let targets = [0usize, 2];
    let counts = [200.0, 20.0, 4.0];
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 6, seed: 41 };

    // End-to-end differentiable compensation passes grad_check_with for all reductions even with active FP.
    for reduction in [LossReduction::None, LossReduction::Sum, LossReduction::Mean] {
        let cfg = SeesawConfig { p: 0.8, q: 1.5, eps: 1e-6, detach_compensation: false, reduction };
        grad_check_with(
            |ins| {
                let out = seesaw_loss(&ins[0], &targets, &counts, &cfg)?;
                if reduction == LossReduction::None { out.sum_all() } else { Ok(out) }
            },
            std::slice::from_ref(&logits),
            &gc,
        )
        .unwrap();
    }

    // Detached mode passes grad_check_with directly when q == 0 (mitigation only).
    let cfg_mit =
        SeesawConfig { p: 0.8, q: 0.0, eps: 1e-6, detach_compensation: true, reduction: LossReduction::Mean };
    grad_check_with(
        |ins| seesaw_loss(&ins[0], &targets, &counts, &cfg_mit),
        std::slice::from_ref(&logits),
        &gc,
    )
    .unwrap();

    // Stateful SeesawLoss tracks counts and validates before mutating.
    let mut tracker = SeesawLoss::new(3, SeesawConfig::default()).unwrap();
    assert_eq!(tracker.counts(), &[1.0, 1.0, 1.0]);
    let l1 = tracker.forward(&logits, &[0, 0]).unwrap().item::<f32>().unwrap();
    assert!(l1.is_finite() && l1 >= 0.0);
    assert_eq!(tracker.counts(), &[3.0, 1.0, 1.0]);
    // Out-of-range target errors without mutating counts.
    assert!(tracker.forward(&logits, &[0, 5]).is_err());
    assert_eq!(tracker.counts(), &[3.0, 1.0, 1.0]);
    // Empty batch preserves autograd graph and returns 0.0 cleanly; with_counts / update_counts / reset work.
    let empty = Tensor::zeros([0, 3]).requires_grad_();
    let el = tracker.loss(&empty, &[]).unwrap();
    assert_eq!(el.item::<f32>().unwrap(), 0.0);
    assert_eq!(el.backward().unwrap().get(&empty).unwrap().dims(), &[0, 3]);

    let mut custom = SeesawLoss::with_counts(vec![10.0, 2.0, 1.0], SeesawConfig::default()).unwrap();
    custom.update_counts(&[1, 2]).unwrap();
    assert_eq!(custom.counts(), &[10.0, 3.0, 2.0]);
    custom.reset();
    assert_eq!(custom.counts(), &[1.0, 1.0, 1.0]);
}

#[test]
fn prob_iou_gwd_kld_f64_central_fd_and_properties() {
    let cases: [([f64; 4], [f64; 4]); 8] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0]), // overlapping
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5]),   // concentric
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0]), // disjoint
        ([0.0, 0.0, 1.0, 1.0], [2.5, 1.5, 3.5, 2.5]),     // tiny disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]), // sub-pixel tiny
        ([1e5, 2e5, 4e5, 5e5], [1.5e5, 1.8e5, 4.5e5, 5.2e5]), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5]), // extreme aspect
        ([1.0, 1.0, 5.0, 5.0], [1.0 + 1e-3, 1.0, 5.0, 5.0]), // nearly identical
    ];

    type BoxLossFn = dyn Fn([f64; 4], [f64; 4]) -> (f64, [f64; 4], [f64; 4]);
    for (idx, &(a, b)) in cases.iter().enumerate() {
        let diff_scale = (0..4).map(|k| (a[k] - b[k]).abs()).fold(0.0f64, f64::max);
        // ProbIoU L1, ProbIoU L2, GWD (Log1p + Sqrt), and KLD all match central FD wrt BOTH a and b.
        let check_fn = |name: &str, f: &BoxLossFn| {
            let (loss, ga, gb) = f(a, b);
            assert!(loss.is_finite() && loss >= 0.0, "{name} case {idx}: bad loss {loss}");
            assert!(ga.iter().chain(&gb).all(|v| v.is_finite()), "{name} case {idx}: non-finite grad");
            for coord in 0..4 {
                let span_a = if coord % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
                let eps_a =
                    (1e-6 * span_a.max(1e-4)).min(1e-3 * diff_scale.max(1e-6)).max(1e-8 * a[coord].abs());
                let (mut ap, mut am) = (a, a);
                ap[coord] += eps_a;
                am[coord] -= eps_a;
                let fd_a = (f(ap, b).0 - f(am, b).0) / (2.0 * eps_a);
                let tol_a = 1e-8 + 1e-5 * fd_a.abs().max(ga[coord].abs());
                assert!(
                    (ga[coord] - fd_a).abs() <= tol_a,
                    "{name} case {idx} da[{coord}]: ana={} fd={fd_a}",
                    ga[coord]
                );

                let span_b = if coord % 2 == 0 { (b[2] - b[0]).abs() } else { (b[3] - b[1]).abs() };
                let eps_b =
                    (1e-6 * span_b.max(1e-4)).min(1e-3 * diff_scale.max(1e-6)).max(1e-8 * b[coord].abs());
                let (mut bp, mut bm) = (b, b);
                bp[coord] += eps_b;
                bm[coord] -= eps_b;
                let fd_b = (f(a, bp).0 - f(a, bm).0) / (2.0 * eps_b);
                let tol_b = 1e-8 + 1e-5 * fd_b.abs().max(gb[coord].abs());
                assert!(
                    (gb[coord] - fd_b).abs() <= tol_b,
                    "{name} case {idx} db[{coord}]: ana={} fd={fd_b}",
                    gb[coord]
                );
            }
        };

        check_fn("prob_iou_l1", &prob_iou_l1_box_f64);
        check_fn("prob_iou_l2", &prob_iou_l2_box_f64);
        check_fn("gwd_log1p", &|x, y| gwd_box_f64(x, y, 1.0, GwdTransform::Log1p));
        check_fn("gwd_sqrt", &|x, y| gwd_box_f64(x, y, 2.0, GwdTransform::Sqrt));
        check_fn("kld", &|x, y| kld_box_f64(x, y, 1.0));
    }

    // Identical boxes yield exact 0 loss and 0 gradient across all Gaussian losses.
    let same = [3.0, 4.0, 9.0, 12.0];
    assert_eq!(prob_iou_l1_box_f64(same, same), (0.0, [0.0; 4], [0.0; 4]));
    assert_eq!(prob_iou_l2_box_f64(same, same), (0.0, [0.0; 4], [0.0; 4]));
    assert_eq!(gwd_box_f64(same, same, 1.0, GwdTransform::Log1p), (0.0, [0.0; 4], [0.0; 4]));
    assert_eq!(kld_box_f64(same, same, 1.0), (0.0, [0.0; 4], [0.0; 4]));

    // Strict scale invariance across 16 orders of magnitude for ProbIoU and KLD.
    let a0 = [1.0, 2.0, 5.0, 8.0];
    let b0 = [2.5, 1.5, 7.0, 9.0];
    let scale = |box_: [f64; 4], s: f64| [box_[0] * s, box_[1] * s, box_[2] * s, box_[3] * s];
    let p_ref = prob_iou_l1_box_f64(a0, b0).0;
    let k_ref = kld_box_f64(a0, b0, 1.0).0;
    for s in [1e-4, 1e4, 1e10] {
        let p_s = prob_iou_l1_box_f64(scale(a0, s), scale(b0, s)).0;
        let k_s = kld_box_f64(scale(a0, s), scale(b0, s), 1.0).0;
        assert!((p_ref - p_s).abs() < 1e-12, "ProbIoU scale invariance failed at s={s}: {p_ref} vs {p_s}");
        assert!((k_ref - k_s).abs() < 1e-12, "KLD scale invariance failed at s={s}: {k_ref} vs {k_s}");
    }

    // Small-object disjoint non-vanishing gradient (unlike plain IoU which has 0 gradient).
    let small_a = [0.0, 0.0, 2.0, 2.0];
    let small_b = [3.0, 0.0, 5.0, 2.0];
    assert!(prob_iou_l1_box_f64(small_a, small_b).1[0].abs() > 1e-3);
    assert!(gwd_box_f64(small_a, small_b, 1.0, GwdTransform::Log1p).1[0].abs() > 1e-3);
    assert!(kld_box_f64(small_a, small_b, 1.0).1[0].abs() > 1e-3);

    // KLD extreme small width ratio (w1 = 1 vs w2 = 1e18): must NOT saturate to 1.0 / zero grad.
    let kld_a = [-0.5, -0.5, 0.5, 0.5];
    let kld_b = [-0.5e18, -0.5, 0.5e18, 0.5];
    let (kl_val, kl_ga, kl_gb) = kld_box_f64(kld_a, kld_b, 1.0);
    let want_dkl = 0.5 * ((1e-18f64).powi(2) - 1.0 - 2.0 * (1e-18f64).ln());
    let want_kl = want_dkl.ln_1p() / (1.0 + want_dkl.ln_1p());
    assert!((kl_val - want_kl).abs() < 1e-12, "KLD extreme ratio mismatch: {kl_val} vs {want_kl}");
    assert!(kl_ga[0].abs() > 1e-4 && kl_gb[0].abs() > 0.0, "KLD extreme ratio must have nonzero grads");

    // Degenerate/inverted boxes and non-finite inputs remain finite / propagate NaN cleanly.
    let inv = [10.0, 10.0, 2.0, 2.0];
    let valid = [1.0, 1.0, 5.0, 5.0];
    assert!(prob_iou_l1_box_f64(inv, valid).0.is_finite());
    assert!(gwd_box_f64(inv, valid, 1.0, GwdTransform::Log1p).0.is_finite());
    assert!(kld_box_f64(inv, valid, 1.0).0.is_finite());
    let nan_box = [f64::NAN, 0.0, 2.0, 2.0];
    assert!(prob_iou_l1_box_f64(nan_box, valid).0.is_nan());
    assert!(gwd_box_f64(nan_box, valid, 1.0, GwdTransform::Log1p).0.is_nan());
    assert!(kld_box_f64(nan_box, valid, 1.0).0.is_nan());
}

#[test]
fn prob_iou_gwd_kld_tensor_grad_check_and_validation() {
    let b1 = Tensor::from_slice(&[2.0f32, 3.0, 10.0, 12.0, 0.0, 0.0, 2.0, 2.0, 5.0, 5.0, 6.5, 7.0], [3, 4])
        .unwrap();
    let b2 = Tensor::from_slice(&[3.5f32, 2.5, 11.5, 13.0, 3.5, 1.0, 5.5, 3.0, 5.8, 5.6, 7.2, 7.9], [3, 4])
        .unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 24, seed: 53 };

    grad_check_with(|ins| prob_iou(&ins[0], &ins[1])?.sum_all(), &[b1.clone(), b2.clone()], &gc).unwrap();
    grad_check_with(|ins| prob_iou_loss(&ins[0], &ins[1])?.sum_all(), &[b1.clone(), b2.clone()], &gc)
        .unwrap();
    grad_check_with(|ins| prob_iou_l2_loss(&ins[0], &ins[1])?.sum_all(), &[b1.clone(), b2.clone()], &gc)
        .unwrap();
    grad_check_with(
        |ins| gwd_loss(&ins[0], &ins[1], 1.0, GwdTransform::Log1p)?.sum_all(),
        &[b1.clone(), b2.clone()],
        &gc,
    )
    .unwrap();
    grad_check_with(
        |ins| gwd_loss(&ins[0], &ins[1], 1.5, GwdTransform::Sqrt)?.sum_all(),
        &[b1.clone(), b2.clone()],
        &gc,
    )
    .unwrap();
    grad_check_with(|ins| kld_loss(&ins[0], &ins[1], 1.0)?.sum_all(), &[b1.clone(), b2.clone()], &gc)
        .unwrap();

    // ProbIoU + ProbIoU_loss == 1.0 elementwise.
    let piou = prob_iou(&b1, &b2).unwrap().to_vec::<f32>().unwrap();
    let ploss = prob_iou_loss(&b1, &b2).unwrap().to_vec::<f32>().unwrap();
    for (p, l) in piou.iter().zip(&ploss) {
        assert!((p + l - 1.0).abs() < 1e-6);
    }

    // Rationalized ProbIoU similarity stays representably positive at large B_D = 37.5 (5 widths apart).
    let far_a = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], [1, 4]).unwrap().requires_grad_();
    let far_b = Tensor::from_slice(&[5.0f32, 0.0, 6.0, 1.0], [1, 4]).unwrap();
    let far_sim = prob_iou(&far_a, &far_b).unwrap();
    let sim_v = far_sim.to_vec::<f32>().unwrap()[0];
    let want_sim = ((-37.5f64).exp() / 2.0) as f32;
    assert!(sim_v > 0.0 && (sim_v - want_sim).abs() <= 1e-5 * want_sim, "far sim {sim_v:e} vs {want_sim:e}");

    // Invalid tau and empty autograd-tracked tensors handled properly.
    assert!(gwd_loss(&b1, &b2, 0.0, GwdTransform::Log1p).is_err());
    assert!(kld_loss(&b1, &b2, -1.0).is_err());
    let empty_a = Tensor::zeros([0, 4]).requires_grad_();
    let empty_b = Tensor::zeros([0, 4]).requires_grad_();
    let empty_out = prob_iou_loss(&empty_a, &empty_b).unwrap();
    assert_eq!(empty_out.dims(), &[0]);
    let empty_grads = empty_out.sum_all().unwrap().backward().unwrap();
    assert_eq!(empty_grads.get(&empty_a).unwrap().dims(), &[0, 4]);
    assert_eq!(empty_grads.get(&empty_b).unwrap().dims(), &[0, 4]);
    assert_eq!(kld_loss(&empty_a, &empty_b, 1.0).unwrap().dims(), &[0]);
}
