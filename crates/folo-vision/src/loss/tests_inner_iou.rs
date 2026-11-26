//! Unit tests for generalized Inner-IoU (`inner_iou`, `inner_iou_loss`, `wrap_inner_iou_loss`):
//! `f64` central finite-difference gradient checks across all 7 [`InnerIouBase`] variants and
//! multiple ratios (`0.5..=1.5`), tensor `grad_check_with`, exact `ratio == 1.0` parity against
//! base losses, wrapper composition parity, 24-decade scale invariance, degenerate/inverted boxes,
//! and `NaN` / `±Inf` / invalid-argument safety.

use super::terms::{
    InnerIouBase, ciou_aspect_penalty_f64, eiou_loss, inner_iou, inner_iou_box_f64, inner_iou_loss,
    inner_iou_loss_box_f64, inner_one_minus_iou_f64, mpdiou_loss, piou_loss, siou_loss, wrap_inner_iou_loss,
};
use crate::boxes::{IouKind, Xyxy, iou as scalar_iou};
use folo_core::Tensor;
use folo_core::testing::{GradCheck, grad_check_with};

/// `f64` central finite-difference gradient checks across `Iou`, `Giou`, `Diou`, `Eiou`, `Siou`,
/// and `Mpdiou` at ratios `0.5`, `0.7`, `1.0`, `1.25`, `1.5`, plus frozen-`alpha*` central FD
/// check for `Ciou`.
#[test]
fn inner_iou_f64_central_fd_all_bases_and_ratios() {
    let cases: [([f64; 4], [f64; 4]); 9] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.6]), // general overlapping (non-collinear centers)
        ([3.2, 3.6, 7.4, 8.1], [1.0, 1.0, 10.0, 10.0]),   // pred strictly inside GT
        ([0.5, 0.2, 12.0, 11.5], [3.0, 2.5, 8.0, 7.5]),   // GT strictly inside pred
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5]),   // concentric different sizes
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0]), // disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7]), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5]), // extreme aspect ratio
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0]),  // inverted predicted box
    ];

    let exact_bases = [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 4.0 },
        InnerIouBase::Mpdiou,
    ];
    let ratios = [0.5f64, 0.7, 1.0, 1.25, 1.5];

    for &ratio in &ratios {
        for &base in &exact_bases {
            for (idx, &(a, b)) in cases.iter().enumerate() {
                let (loss, grad) = inner_iou_loss_box_f64(a, b, base, ratio);
                assert!(loss.is_finite() && loss >= -1e-12, "{base:?} r={ratio} case {idx}: loss={loss}");
                for c in 0..4 {
                    let span = if c % 2 == 0 { (b[2] - b[0]).abs() } else { (b[3] - b[1]).abs() };
                    let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
                    let mut ap = a;
                    let mut am = a;
                    ap[c] += eps;
                    am[c] -= eps;
                    let fd = (inner_iou_loss_box_f64(ap, b, base, ratio).0
                        - inner_iou_loss_box_f64(am, b, base, ratio).0)
                        / (2.0 * eps);
                    let tol = 1e-8 + 1e-5 * fd.abs().max(grad[c].abs());
                    assert!(
                        (grad[c] - fd).abs() <= tol,
                        "{base:?} r={ratio} case {idx} coord {c}: ana={} fd={fd}",
                        grad[c]
                    );
                }
            }
        }

        // CIoU: verify exact frozen-alpha* central finite differences.
        for (idx, &(a, b)) in cases.iter().enumerate() {
            let (loss, grad) = inner_iou_loss_box_f64(a, b, InnerIouBase::Ciou, ratio);
            assert!(loss.is_finite() && loss >= -1e-12, "Ciou r={ratio} case {idx}: loss={loss}");
            let (_, _, alpha) = ciou_aspect_penalty_f64(a, b);
            let eval_frozen = |x: [f64; 4]| {
                let (l_diou, _) = inner_iou_loss_box_f64(x, b, InnerIouBase::Diou, ratio);
                let w1 = (x[2] - x[0]).max(0.0);
                let h1 = (x[3] - x[1]).max(0.0);
                let w2 = (b[2] - b[0]).max(0.0);
                let h2 = (b[3] - b[1]).max(0.0);
                let d = w2.atan2(h2) - w1.atan2(h1);
                let v = (4.0 / (std::f64::consts::PI * std::f64::consts::PI)) * d * d;
                l_diou + alpha * v
            };
            for c in 0..4 {
                let span = if c % 2 == 0 { (b[2] - b[0]).abs() } else { (b[3] - b[1]).abs() };
                let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
                let mut ap = a;
                let mut am = a;
                ap[c] += eps;
                am[c] -= eps;
                let fd = (eval_frozen(ap) - eval_frozen(am)) / (2.0 * eps);
                let tol = 1e-8 + 1e-5 * fd.abs().max(grad[c].abs());
                assert!(
                    (grad[c] - fd).abs() <= tol,
                    "Ciou frozen-alpha r={ratio} case {idx} coord {c}: ana={} fd={fd}",
                    grad[c]
                );
            }
        }
    }
}

/// Tensor-level `grad_check_with` for `inner_iou`, `inner_iou_loss` across all base variants,
/// and `wrap_inner_iou_loss` composing with `eiou_loss`, `mpdiou_loss`, and `piou_loss`.
#[test]
fn inner_iou_tensor_grad_check_and_wrapper_parity() {
    let b1 = Tensor::from_slice(&[2.3f32, 3.1, 10.7, 12.4, 3.2, 3.6, 7.4, 8.1, 0.5, 0.7, 3.1, 3.3], [3, 4])
        .unwrap();
    let b2 = Tensor::from_slice(&[3.0f32, 2.5, 11.2, 13.0, 1.0, 1.0, 10.0, 10.0, 4.2, 5.1, 8.4, 9.2], [3, 4])
        .unwrap();
    let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 12, seed: 41 };

    // Differentiable inner_iou overlap.
    for &ratio in &[0.7f32, 1.25] {
        grad_check_with(|ins| inner_iou(&ins[0], &b2, ratio)?.sum_all(), std::slice::from_ref(&b1), &gc)
            .unwrap();
    }

    // All non-detached InnerIouBase variants pass direct tensor grad_check_with.
    for base in [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 4.0 },
        InnerIouBase::Mpdiou,
    ] {
        for &ratio in &[0.7f32, 1.3] {
            grad_check_with(
                |ins| inner_iou_loss(&ins[0], &b2, base, ratio)?.sum_all(),
                std::slice::from_ref(&b1),
                &gc,
            )
            .unwrap();
        }
    }

    // Same-aspect-ratio boxes (v == 0, alpha == 0) pass direct grad_check_with for Ciou too.
    let sq1 = Tensor::from_slice(&[1.0f32, 1.5, 5.0, 5.5, 2.0, 2.0, 6.0, 6.0], [2, 4]).unwrap();
    let sq2 = Tensor::from_slice(&[2.2f32, 2.1, 6.2, 6.1, 2.8, 3.2, 6.8, 7.2], [2, 4]).unwrap();
    grad_check_with(
        |ins| inner_iou_loss(&ins[0], &sq2, InnerIouBase::Ciou, 0.7)?.sum_all(),
        std::slice::from_ref(&sq1),
        &gc,
    )
    .unwrap();

    // Composable wrapper parity: wrap_inner_iou_loss(eiou_loss) == inner_iou_loss(..., Eiou) in
    // both forward value and backward gradient, and passes grad_check_with on piou_loss too.
    let ratio = 0.75f32;
    let b1_a = b1.clone().requires_grad_();
    let wrapped_e = wrap_inner_iou_loss(&eiou_loss(&b1_a, &b2).unwrap(), &b1_a, &b2, ratio).unwrap();
    let g_wrapped_e =
        wrapped_e.sum_all().unwrap().backward().unwrap().get(&b1_a).unwrap().to_vec::<f32>().unwrap();

    let b1_b = b1.clone().requires_grad_();
    let direct_e = inner_iou_loss(&b1_b, &b2, InnerIouBase::Eiou, ratio).unwrap();
    let g_direct_e =
        direct_e.sum_all().unwrap().backward().unwrap().get(&b1_b).unwrap().to_vec::<f32>().unwrap();

    for (w, d) in wrapped_e.to_vec::<f32>().unwrap().iter().zip(direct_e.to_vec::<f32>().unwrap()) {
        assert!((w - d).abs() < 1e-6, "wrapped EIoU {w} != direct {d}");
    }
    for (gw, gd) in g_wrapped_e.iter().zip(&g_direct_e) {
        assert!((gw - gd).abs() < 1e-5, "wrapped EIoU grad {gw} != direct {gd}");
    }

    // Composing with MPDIoU and PIoU via wrap_inner_iou_loss passes grad_check_with.
    grad_check_with(
        |ins| {
            let base = mpdiou_loss(&ins[0], &b2)?;
            wrap_inner_iou_loss(&base, &ins[0], &b2, 0.8)?.sum_all()
        },
        std::slice::from_ref(&b1),
        &gc,
    )
    .unwrap();
    grad_check_with(
        |ins| {
            let base = piou_loss(&ins[0], &b2)?;
            wrap_inner_iou_loss(&base, &ins[0], &b2, 1.2)?.sum_all()
        },
        std::slice::from_ref(&b1),
        &gc,
    )
    .unwrap();
}

/// Exact `ratio == 1.0` reduction to standard losses, 24-decade scale invariance, identical-box
/// zero loss/grad, and NaN/Inf/degenerate/invalid-param safety.
#[test]
fn inner_iou_identities_scale_invariance_and_edge_cases() {
    let a_xy = Xyxy::new(1.5, 2.0, 9.5, 11.0);
    let b_xy = Xyxy::new(3.0, 3.5, 12.0, 14.0);
    let ta = Tensor::from_slice(&a_xy.to_array(), [1, 4]).unwrap();
    let tb = Tensor::from_slice(&b_xy.to_array(), [1, 4]).unwrap();

    // At ratio == 1.0, matches IoU / GIoU / DIoU / CIoU / EIoU / SIoU / MPDIoU.
    assert!(
        (inner_iou(&ta, &tb, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - scalar_iou(&a_xy, &b_xy, IouKind::Iou))
        .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Iou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - (1.0 - scalar_iou(&a_xy, &b_xy, IouKind::Iou)))
        .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Giou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - (1.0 - scalar_iou(&a_xy, &b_xy, IouKind::Giou)))
        .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Diou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - (1.0 - scalar_iou(&a_xy, &b_xy, IouKind::Diou)))
        .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Ciou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - (1.0 - scalar_iou(&a_xy, &b_xy, IouKind::Ciou)))
        .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Eiou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - eiou_loss(&ta, &tb).unwrap().to_vec::<f32>().unwrap()[0])
            .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Siou { theta: 4.0 }, 1.0).unwrap().to_vec::<f32>().unwrap()
            [0]
            - siou_loss(&ta, &tb, 4.0).unwrap().to_vec::<f32>().unwrap()[0])
            .abs()
            < 1e-6
    );
    assert!(
        (inner_iou_loss(&ta, &tb, InnerIouBase::Mpdiou, 1.0).unwrap().to_vec::<f32>().unwrap()[0]
            - mpdiou_loss(&ta, &tb).unwrap().to_vec::<f32>().unwrap()[0])
            .abs()
            < 1e-6
    );

    // Identical valid boxes have exact 0.0 loss and 0.0 gradient for every base & ratio.
    let same = [2.0f64, 3.0, 8.0, 11.0];
    for base in [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Ciou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 4.0 },
        InnerIouBase::Mpdiou,
    ] {
        for &r in &[0.6f64, 1.0, 1.4] {
            let (l, g) = inner_iou_loss_box_f64(same, same, base, r);
            assert_eq!(l, 0.0, "{base:?} r={r} identical boxes loss");
            assert_eq!(g, [0.0; 4], "{base:?} r={r} identical boxes grad");
        }
    }
    assert_eq!(inner_iou_box_f64(same, same, 0.7).0, 1.0);
    assert_eq!(inner_one_minus_iou_f64(same, same, 0.7).0, 0.0);

    // Zero-area point and line boxes yield finite loss and finite gradients across every base & ratio.
    let point = [5.0f64, 5.0, 5.0, 5.0];
    let line = [2.0f64, 5.0, 8.0, 5.0];
    let gt_box = [2.0f64, 3.0, 8.0, 9.0];
    for base in [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Ciou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 4.0 },
        InnerIouBase::Mpdiou,
    ] {
        for deg in [point, line] {
            for &r in &[0.5f64, 0.7, 1.0, 1.5] {
                let (l, g) = inner_iou_loss_box_f64(deg, gt_box, base, r);
                assert!(l.is_finite() && l >= 0.0, "{base:?} deg r={r}: loss={l}");
                assert!(g.iter().all(|v| v.is_finite()), "{base:?} deg r={r}: grad={g:?}");
            }
        }
    }

    // Scale invariance across 24 orders of magnitude (`1e-12` vs `1.0` vs `1e12`).
    for base in [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Ciou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 4.0 },
        InnerIouBase::Mpdiou,
    ] {
        let l1 = inner_iou_loss_box_f64([0.0, 0.0, 1.0, 2.0], [0.3, 0.5, 1.6, 2.8], base, 0.75).0;
        let l_tiny = inner_iou_loss_box_f64(
            [0.0, 0.0, 1e-12, 2e-12],
            [0.3e-12, 0.5e-12, 1.6e-12, 2.8e-12],
            base,
            0.75,
        )
        .0;
        let l_huge =
            inner_iou_loss_box_f64([0.0, 0.0, 1e12, 2e12], [0.3e12, 0.5e12, 1.6e12, 2.8e12], base, 0.75).0;
        assert!((l1 - l_tiny).abs() < 1e-11, "{base:?} tiny scale invariance: {l1} vs {l_tiny}");
        assert!((l1 - l_huge).abs() < 1e-11, "{base:?} huge scale invariance: {l1} vs {l_huge}");
    }

    // Empty tensors return empty [0] tensors cleanly.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(inner_iou(&empty, &empty, 0.7).unwrap().dims(), &[0]);
    assert_eq!(inner_iou_loss(&empty, &empty, InnerIouBase::Giou, 0.7).unwrap().dims(), &[0]);
    assert_eq!(wrap_inner_iou_loss(&Tensor::zeros([0]), &empty, &empty, 0.7).unwrap().dims(), &[0]);

    // NaN / ±Inf propagate NaN through both forward and backward.
    let good = Tensor::from_slice(&[1.0f32, 1.0, 5.0, 5.0], [1, 4]).unwrap();
    for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad = Tensor::from_slice(&[bad_val, 1.0, 5.0, 5.0], [1, 4]).unwrap().requires_grad_();
        let out = inner_iou_loss(&bad, &good, InnerIouBase::Diou, 0.7).unwrap();
        assert!(out.to_vec::<f32>().unwrap()[0].is_nan());
        let g = out.sum_all().unwrap().backward().unwrap().get(&bad).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for {bad_val}, got {g:?}");

        let bad_w = Tensor::from_slice(&[bad_val, 1.0, 5.0, 5.0], [1, 4]).unwrap().requires_grad_();
        let base_zero = Tensor::from_slice(&[0.5f32], [1]).unwrap();
        let out_w = wrap_inner_iou_loss(&base_zero, &bad_w, &good, 0.7).unwrap();
        assert!(out_w.to_vec::<f32>().unwrap()[0].is_nan());
        let gw = out_w.sum_all().unwrap().backward().unwrap().get(&bad_w).unwrap().to_vec::<f32>().unwrap();
        assert!(gw.iter().all(|v| v.is_nan()));
    }

    // Invalid parameters rejected.
    for bad_ratio in [0.0f32, -0.5, f32::NAN, f32::INFINITY] {
        assert!(inner_iou(&ta, &tb, bad_ratio).is_err());
        assert!(inner_iou_loss(&ta, &tb, InnerIouBase::Iou, bad_ratio).is_err());
        assert!(
            wrap_inner_iou_loss(&Tensor::from_slice(&[0.2f32], [1]).unwrap(), &ta, &tb, bad_ratio).is_err()
        );
    }
    assert!(inner_iou_loss(&ta, &tb, InnerIouBase::Siou { theta: 1.5 }, 0.7).is_err());
    assert!(inner_iou_loss(&ta, &tb, InnerIouBase::Siou { theta: 7.0 }, 0.7).is_err());
    assert!(inner_iou_loss(&ta, &tb, InnerIouBase::Siou { theta: f32::NAN }, 0.7).is_err());
    assert!(wrap_inner_iou_loss(&Tensor::from_slice(&[0.2f32, 0.3], [2]).unwrap(), &ta, &tb, 0.7).is_err());

    // Serde round-trip for InnerIouBase.
    for base in [
        InnerIouBase::Iou,
        InnerIouBase::Giou,
        InnerIouBase::Diou,
        InnerIouBase::Ciou,
        InnerIouBase::Eiou,
        InnerIouBase::Siou { theta: 3.5 },
        InnerIouBase::Mpdiou,
    ] {
        let json = serde_json::to_string(&base).unwrap();
        assert_eq!(serde_json::from_str::<InnerIouBase>(&json).unwrap(), base);
    }

    // Tiny ratio (1e-20) on identical unit boxes must not collapse auxiliary endpoints.
    let unit = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], [1, 4]).unwrap();
    assert_eq!(inner_iou(&unit, &unit, 1e-20).unwrap().to_vec::<f32>().unwrap()[0], 1.0);
    assert_eq!(
        inner_iou_loss(&unit, &unit, InnerIouBase::Iou, 1e-20).unwrap().to_vec::<f32>().unwrap()[0],
        0.0
    );

    // Tiny box of area 1e-20 nested inside unit box must preserve exact IoU (no 1 - (1 - IoU) cancellation).
    let tiny_nested = Tensor::from_slice(&[0.0f32, 0.0, 1e-10, 1e-10], [1, 4]).unwrap();
    let got_tiny = inner_iou_box_f64([0.0, 0.0, 1e-10, 1e-10], [0.0, 0.0, 1.0, 1.0], 1.0).0;
    assert!((got_tiny - 1e-20).abs() < 1e-34, "expected 1e-20, got {got_tiny:e}");
    let got_tiny_t = inner_iou(&tiny_nested, &unit, 1.0).unwrap().to_vec::<f32>().unwrap()[0];
    assert!((got_tiny_t - 1e-20).abs() < 1e-25, "expected 1e-20f32, got {got_tiny_t:e}");

    // Wrapper accumulates in f64 before casting to f32 (no cancellation of tiny base_loss + delta).
    let a_wrap = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0], [1, 4]).unwrap();
    let b_wrap = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0], [1, 4]).unwrap();
    let tiny_base = Tensor::from_slice(&[3e-9f32], [1]).unwrap();
    let wrapped = wrap_inner_iou_loss(&tiny_base, &a_wrap, &b_wrap, 0.7).unwrap().to_vec::<f32>().unwrap()[0];
    assert!((wrapped - 3e-9).abs() < 1e-14, "wrapper must preserve tiny base loss, got {wrapped:e}");

    // Extreme aspect ratio (w=1, h=1e8, ratio=0.5): direct EIoU/SIoU penalties preserve exact
    // O(1e-16) center-distance gradients without O(1) `full.grad - full.grad` cancellation.
    let a_ext = [0.0f64, 0.0, 1.0, 1e8];
    let b_ext = [0.6f64, 0.0, 1.6, 1e8];
    let (l_eiou, g_eiou) = inner_iou_loss_box_f64(a_ext, b_ext, InnerIouBase::Eiou, 0.5);
    assert!(l_eiou >= 1.0 && l_eiou.is_finite());
    let want_g0 = -0.6f64 / (1.6 * 1.6 + 1e16);
    assert!((g_eiou[0] - want_g0).abs() <= 1e-12 * want_g0.abs(), "g_eiou[0]={} vs {want_g0}", g_eiou[0]);
    let (l_siou, g_siou) = inner_iou_loss_box_f64(a_ext, b_ext, InnerIouBase::Siou { theta: 4.0 }, 0.5);
    assert!(l_siou > 1.0 && l_siou.is_finite());
    assert!(g_siou[0].is_finite() && g_siou[0] < 0.0);
}
