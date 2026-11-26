//! Unit tests for `MPDIoU`, `Shape-IoU`, and `Focaler-IoU`: `f64` central finite-difference gradient checks,
//! tensor autograd gradient checks, algebraic identities, degenerate & extreme edge cases, and end-to-end
//! `detection_loss` / config validation.

use super::terms::{mpdiou_box_f64, shape_iou_box_f64};
use super::*;
use crate::boxes::{IouKind, iou as scalar_iou};
use crate::decode::make_anchors;

/// `f64` central finite-difference gradient check for `MPDIoU` (scale-invariant and fixed-scale).
#[test]
fn mpdiou_f64_central_fd_grad_check() {
    let cases: [([f64; 4], [f64; 4], Option<f64>); 8] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], None), // overlapping
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5], None),   // concentric different size
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], None), // disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3], None), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7], None), // > 1e6 scale-normalized
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5], None), // extreme aspect ratio
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], Some(64.0 * 64.0 + 64.0 * 64.0)), // fixed scale
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], Some(100.0 * 100.0 + 80.0 * 80.0)), // fixed disjoint
    ];
    for (idx, &(a, b, s2)) in cases.iter().enumerate() {
        let (loss, grad) = mpdiou_box_f64(a, b, s2);
        assert!(loss.is_finite() && loss >= 0.0, "case {idx}: non-finite or negative loss {loss}");
        for c in 0..4 {
            let span = if c % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (mpdiou_box_f64(ap, b, s2).0 - mpdiou_box_f64(am, b, s2).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "mpdiou case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }
    // Continuity across the `1e6` scale-normalization boundary.
    let l_lo = mpdiou_box_f64([0.0, 0.0, 9.99999e5, 4e5], [1e5, 1e5, 8.5e5, 5e5], None).0;
    let l_hi = mpdiou_box_f64([0.0, 0.0, 1.000001e6, 4e5], [1e5, 1e5, 8.5e5, 5e5], None).0;
    assert!((l_lo - l_hi).abs() < 1e-5, "mpdiou threshold jump across 1e6: {l_lo} vs {l_hi}");
}

/// Algebraic identities for `MPDIoU`:
/// 1. Equal-sized shifted boxes: scale-invariant MPDIoU loss equals `1 - DIoU`.
/// 2. Concentric boxes with different sizes: penalty equals `(dw² + dh²) / (4 * c²) > 0`.
#[test]
fn mpdiou_algebraic_identities() {
    // 1. Equal-sized shifted boxes: (d1² + d2²) / (2 c²) == rho² / c².
    let a = Xyxy::new(1.0, 2.0, 7.0, 10.0);
    let b = Xyxy::new(4.0, 6.0, 10.0, 14.0);
    let ta = Tensor::from_slice(&a.to_array(), [1, 4]).unwrap();
    let tb = Tensor::from_slice(&b.to_array(), [1, 4]).unwrap();
    let l_mpd = mpdiou_loss(&ta, &tb).unwrap().to_vec::<f32>().unwrap()[0];
    let l_diou = 1.0 - scalar_iou(&a, &b, IouKind::Diou);
    assert!((l_mpd - l_diou).abs() < 1e-5, "equal-size shifted MPDIoU {l_mpd} vs 1-DIoU {l_diou}");

    // 2. Concentric boxes: DIoU has zero distance penalty, MPDIoU has (dw² + dh²) / (4 c²).
    let outer = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let inner = Xyxy::new(2.0, 3.0, 8.0, 7.0);
    let to = Tensor::from_slice(&outer.to_array(), [1, 4]).unwrap();
    let ti = Tensor::from_slice(&inner.to_array(), [1, 4]).unwrap();
    let l_conc = mpdiou_loss(&to, &ti).unwrap().to_vec::<f32>().unwrap()[0];
    let want_iou = scalar_iou(&outer, &inner, IouKind::Iou);
    let want_pen = (4.0f32 * 4.0 + 6.0 * 6.0) / (4.0 * (10.0 * 10.0 + 10.0 * 10.0));
    assert!((l_conc - ((1.0 - want_iou) + want_pen)).abs() < 1e-5);
}

/// `f64` central finite-difference gradient check for `Shape-IoU` across scale values and geometries.
#[test]
fn shape_iou_f64_central_fd_grad_check() {
    let cases = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], 0.0), // isotropic scale = 0
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0], 1.0), // scale = 1
        ([1.0, 0.5, 9.0, 3.5], [1.5, 1.0, 11.5, 3.0], 1.5),    // elongated GT, scale = 1.5
        ([2.0, 2.3, 6.0, 8.3], [3.0, 2.0, 7.0, 8.0], 0.75),    // equal size (dw=dh=0, C³ smooth), shifted
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], 1.0), // disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3], 0.5), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7], 1.0), // > 1e6 scale-normalized
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5], 2.0), // extreme aspect, max scale
    ];
    for (idx, &(a, b, scale)) in cases.iter().enumerate() {
        let (loss, grad) = shape_iou_box_f64(a, b, scale);
        assert!(loss.is_finite() && loss >= 0.0, "shape_iou case {idx}: bad loss {loss}");
        for c in 0..4 {
            let span = if c % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (shape_iou_box_f64(ap, b, scale).0 - shape_iou_box_f64(am, b, scale).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "shape_iou case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }
}

/// Edge cases for `MPDIoU` and `Shape-IoU`: identical, disjoint, zero-area, inverted, `1e15`, `1e-12`,
/// empty `[0, 4]`, `NaN`/`±Inf` poison, and invalid parameter rejection.
#[test]
fn mpdiou_and_shape_iou_edge_cases_and_autograd() {
    // 1. Huge 1e15 boxes stay finite in forward & backward.
    let huge_a = Tensor::from_slice(&[0.0f32, 0.0, 1e15, 1e15], [1, 4]).unwrap().requires_grad_();
    let huge_b = Tensor::from_slice(&[1e14f32, 2e14, 1.1e15, 1.2e15], [1, 4]).unwrap();
    for l in [mpdiou_loss(&huge_a, &huge_b).unwrap(), shape_iou_loss(&huge_a, &huge_b, 1.0).unwrap()] {
        let v = l.to_vec::<f32>().unwrap()[0];
        assert!(v.is_finite() && v > 0.0 && v < 4.0, "huge 1e15 loss out of range: {v}");
        let g = l.sum_all().unwrap().backward().unwrap().get(&huge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|x| x.is_finite()), "huge 1e15 grad non-finite: {g:?}");
    }

    // 2. Degenerate / tiny / inverted / zero-area / identical boxes.
    let edge_a = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny
            5.0, 5.0, 5.0, 5.0, // zero-area point
            10.0, 12.0, 4.0, 6.0, // inverted
            1.0, 1.0, 1.0, 8.0, // zero-width line
            0.0, 0.0, 10.0, 10.0, // identical to GT
        ],
        [5, 4],
    )
    .unwrap()
    .requires_grad_();
    let edge_b = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny
            5.0, 5.0, 8.0, 8.0, // normal GT
            2.0, 3.0, 9.0, 11.0, // normal GT
            0.0, 0.0, 5.0, 5.0, // normal GT
            0.0, 0.0, 10.0, 10.0, // identical GT
        ],
        [5, 4],
    )
    .unwrap();
    for (name, l) in [
        ("mpdiou", mpdiou_loss(&edge_a, &edge_b).unwrap()),
        ("mpdiou_scale", mpdiou_loss_with_scale(&edge_a, &edge_b, 32.0, 32.0).unwrap()),
        ("shape_iou", shape_iou_loss(&edge_a, &edge_b, 1.0).unwrap()),
    ] {
        let vals = l.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite() && *v >= 0.0), "{name} degenerate losses: {vals:?}");
        assert!(vals[4] < 1e-6, "{name}: identical box must have ~0 loss, got {}", vals[4]);
        let g = l.sum_all().unwrap().backward().unwrap().get(&edge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{name} degenerate grads: {g:?}");
        assert!(
            g[16..20].iter().all(|v| v.abs() < 1e-6),
            "{name} identical box grad must be ~0: {:?}",
            &g[16..20]
        );
    }

    // 3. NaN / ±Inf poison both forward and backward.
    for bad in [[f32::NAN, 0.0, 4.0, 4.0], [0.0, f32::INFINITY, 4.0, 4.0], [f32::NEG_INFINITY, 0.0, 4.0, 4.0]]
    {
        let bp = Tensor::from_slice(&bad, [1, 4]).unwrap().requires_grad_();
        let bg = Tensor::from_slice(&[0.0f32, 0.0, 4.0, 4.0], [1, 4]).unwrap();
        for l in [
            mpdiou_loss(&bp, &bg).unwrap(),
            mpdiou_loss_with_scale(&bp, &bg, 16.0, 16.0).unwrap(),
            shape_iou_loss(&bp, &bg, 0.5).unwrap(),
        ] {
            assert!(l.to_vec::<f32>().unwrap()[0].is_nan());
            let g = l.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for {bad:?}, got {g:?}");
        }
    }

    // 4. Empty [0, 4] and invalid parameters.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(mpdiou_loss(&empty, &empty).unwrap().dims(), &[0]);
    assert_eq!(shape_iou_loss(&empty, &empty, 0.5).unwrap().dims(), &[0]);
    let b = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 2.0], [1, 4]).unwrap();
    assert!(mpdiou_loss_with_scale(&b, &b, 0.0, 10.0).is_err());
    assert!(mpdiou_loss_with_scale(&b, &b, 10.0, f32::NAN).is_err());
    assert!(shape_iou_loss(&b, &b, -0.1).is_err());
    assert!(shape_iou_loss(&b, &b, 2.5).is_err());
    assert!(shape_iou_loss(&b, &b, f32::INFINITY).is_err());
}

/// `Focaler-IoU` remapping, finite-difference gradient check, identity at `(d=0, u=1)`, and `NaN` propagation.
#[test]
fn focaler_iou_grad_check_and_properties() {
    let raw = [0.05f32, 0.25, 0.50, 0.85, 0.98];
    let iou_t = Tensor::from_slice(&raw, [5]).unwrap().requires_grad_();
    let (d, u) = (0.1f32, 0.9f32);
    let remapped = focaler_iou(&iou_t, d, u).unwrap();
    let vals = remapped.to_vec::<f32>().unwrap();
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] - (0.25 - 0.1) / 0.8).abs() < 1e-6);
    assert!((vals[2] - 0.5).abs() < 1e-6);
    assert!((vals[3] - (0.85 - 0.1) / 0.8).abs() < 1e-6);
    assert_eq!(vals[4], 1.0);

    // Analytic gradient vs central finite differences.
    let weights = [1.0f32, 2.0, 0.5, 1.5, 3.0];
    let wt = Tensor::from_slice(&weights, [5]).unwrap();
    let grad = remapped
        .mul(&wt)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&iou_t)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let eps = 1e-3f32;
    for i in 0..raw.len() {
        let mut p = raw;
        let mut m = raw;
        p[i] += eps;
        m[i] -= eps;
        let fp =
            focaler_iou(&Tensor::from_slice(&p, [5]).unwrap(), d, u).unwrap().to_vec::<f32>().unwrap()[i];
        let fm =
            focaler_iou(&Tensor::from_slice(&m, [5]).unwrap(), d, u).unwrap().to_vec::<f32>().unwrap()[i];
        let fd = weights[i] * (fp - fm) / (2.0 * eps);
        assert!((grad[i] - fd).abs() < 1e-4, "focaler_iou idx {i}: analytic {} vs fd {fd}", grad[i]);
    }

    // Composable wrapper identity at (0.0, 1.0) and analytical gradient check for `focaler_iou_loss`.
    let base = Tensor::from_slice(&[1.2f32, 0.9, 0.6, 0.2, 0.05], [5]).unwrap().requires_grad_();
    let id = focaler_iou_loss(&base, &iou_t, 0.0, 1.0).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in id.iter().zip(base.to_vec::<f32>().unwrap()) {
        assert!((a - b).abs() < 1e-6);
    }
    let fl = focaler_iou_loss(&base, &iou_t, d, u).unwrap();
    let fg = fl.sum_all().unwrap().backward().unwrap();
    let g_base = fg.get(&base).unwrap().to_vec::<f32>().unwrap();
    let g_iou = fg.get(&iou_t).unwrap().to_vec::<f32>().unwrap();
    assert!(g_base.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    let expected_g_iou = [1.0f32, 1.0 - 1.0 / 0.8, 1.0 - 1.0 / 0.8, 1.0 - 1.0 / 0.8, 1.0];
    for (got, want) in g_iou.iter().zip(expected_g_iou) {
        assert!((got - want).abs() < 1e-5, "focaler_iou_loss d/diou: {got} vs {want}");
    }

    // NaN propagation and parameter validation.
    let nan_t = Tensor::from_slice(&[0.5f32, f32::NAN], [2]).unwrap().requires_grad_();
    let out_nan = focaler_iou(&nan_t, 0.1, 0.9).unwrap();
    let ov = out_nan.to_vec::<f32>().unwrap();
    assert!(ov[0].is_finite() && ov[1].is_nan());
    let g_nan = out_nan.sum_all().unwrap().backward().unwrap().get(&nan_t).unwrap().to_vec::<f32>().unwrap();
    assert!(g_nan[0].is_finite() && g_nan[1].is_nan());
    assert!(focaler_iou(&iou_t, 0.5, 0.5).is_err());
    assert!(focaler_iou(&iou_t, -0.1, 0.9).is_err());
    assert!(focaler_iou(&iou_t, 0.1, 1.1).is_err());
    assert!(focaler_iou_loss(&Tensor::zeros([3]), &iou_t, 0.1, 0.9).is_err());
}

/// End-to-end `detection_loss` with `BoxLossKind::Mpdiou` and `BoxLossKind::ShapeIou`, plus config serde.
#[test]
fn mpdiou_and_shape_iou_detection_loss_and_serde() {
    let anchors = make_anchors(64, 64, &[8, 16], 0.5);
    let a = anchors.len();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 44.0, 48.0)], labels: vec![0] }];

    for kind in [BoxLossKind::Mpdiou, BoxLossKind::ShapeIou] {
        for reg_max in [2usize, 16] {
            for nwd_weight in [0.0f32, 0.5] {
                let mut rng = folo_core::rng::Rng::seed(303 + reg_max as u64);
                let bx = Tensor::rand_uniform([1, 4 * reg_max, a], -2.0, 3.0, &mut rng).requires_grad_();
                let cx = Tensor::rand_uniform([1, 2, a], -3.0, 2.0, &mut rng).requires_grad_();
                let cfg = LossConfig {
                    reg_max,
                    box_loss: kind,
                    shape_scale: 0.75,
                    nwd_weight,
                    ..LossConfig::default()
                };
                let out = detection_loss(&bx, &cx, &anchors, &targets, &cfg).unwrap();
                assert!(out.is_finite() && out.num_fg > 0, "{kind:?} reg_max={reg_max} nwd={nwd_weight}");
                let g = out.total.backward().unwrap();
                let gb = g.get(&bx).unwrap().to_vec::<f32>().unwrap();
                assert!(gb.iter().all(|v| v.is_finite()) && gb.iter().any(|v| v.abs() > 1e-6));
            }
        }
        let cfg = LossConfig { box_loss: kind, shape_scale: 1.0, ..LossConfig::default() };
        assert!(cfg.validate_params().is_ok());
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), cfg);
    }

    for bad in [-0.1f32, 2.1, f32::NAN, f32::INFINITY] {
        let cfg = LossConfig { shape_scale: bad, ..LossConfig::default() };
        assert!(cfg.validate_params().is_err(), "shape_scale={bad} should fail");
    }
}
