//! Unit tests for `EIoU` and `SIoU`: `f64` central finite-difference gradient checks, algebraic identities,
//! scale invariance across 24 orders of magnitude, degenerate/collinear/zero-area/extreme edge cases,
//! frozen-assignment DFL-logit finite-difference checks, and end-to-end `detection_loss` / config validation.

use super::terms::{eiou_box_f64, siou_box_f64};
use super::*;
use crate::boxes::{IouKind, iou as scalar_iou};
use crate::decode::make_anchors;

/// `f64` central finite-difference gradient check for `EIoU` across overlapping, concentric, disjoint,
/// collinear, tiny, huge, extreme aspect-ratio, and inverted boxes, plus scale invariance.
#[test]
fn eiou_f64_central_fd_grad_check() {
    let cases: [([f64; 4], [f64; 4]); 10] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.0]), // overlapping
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5]),   // concentric different size
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0]), // disjoint
        ([0.0, 2.0, 4.0, 6.0], [7.0, 2.0, 11.0, 6.0]),    // horizontally collinear, equal size
        ([2.0, 0.0, 6.0, 4.0], [2.0, 8.0, 6.0, 14.0]),    // vertically collinear, different height
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3]), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7]), // huge
        ([0.0, 0.0, 100.0, 2.0], [10.0, 0.5, 90.0, 2.5]), // extreme aspect ratio
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0]),  // inverted predicted box
        ([1e15, 2.0, 0.0, 1.0], [2e15, 2.0, 0.0, 1.0]),   // inverted zero-enclosing box
    ];
    for (idx, &(a, b)) in cases.iter().enumerate() {
        let (loss, grad) = eiou_box_f64(a, b);
        assert!(loss.is_finite() && loss >= 0.0, "eiou case {idx}: bad loss {loss}");
        for c in 0..4 {
            let span = if c % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (eiou_box_f64(ap, b).0 - eiou_box_f64(am, b).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "eiou case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }
    // Scale invariance across 24 orders of magnitude (`1e-12` vs `1.0` vs `1e12`).
    let l1 = eiou_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.2, 1.8, 2.5]).0;
    let l_tiny = eiou_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.2e-12, 1.8e-12, 2.5e-12]).0;
    let l_huge = eiou_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.2e12, 1.8e12, 2.5e12]).0;
    assert!((l1 - l_tiny).abs() < 1e-12, "eiou tiny scale invariance: {l1} vs {l_tiny}");
    assert!((l1 - l_huge).abs() < 1e-12, "eiou huge scale invariance: {l1} vs {l_huge}");
}

/// `f64` central finite-difference gradient check for `SIoU` across angles (general, 45° diagonal,
/// horizontal & vertical collinear), concentric, disjoint, equal-size, tiny, huge, inverted, and `theta = 2, 4, 6`.
#[test]
fn siou_f64_central_fd_grad_check() {
    let cases: [([f64; 4], [f64; 4], f64); 12] = [
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.6], 4.0), // general overlapping, theta=4
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.6], 2.0), // theta=2
        ([2.3, 3.1, 10.7, 12.4], [3.0, 2.5, 11.2, 13.6], 6.0), // theta=6
        ([0.0, 0.0, 4.0, 6.0], [3.0, 3.0, 7.0, 9.0], 4.0),     // exact 45-degree diagonal (|dx| == |dy|)
        ([2.0, 2.0, 10.0, 10.0], [3.5, 3.5, 8.5, 8.5], 4.0),   // concentric different size (dx = dy = 0)
        ([0.0, 2.0, 4.0, 6.0], [7.0, 2.0, 11.0, 6.0], 4.0),    // horizontally collinear (dy = 0)
        ([2.0, 0.0, 6.0, 4.0], [2.0, 8.0, 6.0, 14.0], 4.0),    // vertically collinear (dx = 0)
        ([0.0, 0.0, 1.2, 1.4], [15.0, 18.0, 20.0, 22.0], 4.0), // disjoint
        ([1e-3, 2e-3, 5e-3, 6e-3], [2e-3, 1e-3, 7e-3, 5e-3], 4.0), // tiny
        ([1e7, 2e7, 4e7, 5e7], [1.5e7, 1.8e7, 4.5e7, 5.2e7], 4.0), // huge
        ([10.0, 12.0, 4.0, 6.0], [2.0, 3.0, 9.0, 11.0], 4.0),  // inverted predicted box
        ([1e15, 2.0, 0.0, 1.0], [2e15, 2.0, 0.0, 1.0], 4.0),   // inverted zero-enclosing box
    ];
    for (idx, &(a, b, theta)) in cases.iter().enumerate() {
        let (loss, grad) = siou_box_f64(a, b, theta);
        assert!(loss.is_finite() && loss >= 0.0, "siou case {idx}: bad loss {loss}");
        for c in 0..4 {
            let span = if c % 2 == 0 { (a[2] - a[0]).abs() } else { (a[3] - a[1]).abs() };
            let eps = (1e-6 * span.max(1e-3)).max(1e-8 * a[c].abs());
            let mut ap = a;
            let mut am = a;
            ap[c] += eps;
            am[c] -= eps;
            let fd = (siou_box_f64(ap, b, theta).0 - siou_box_f64(am, b, theta).0) / (2.0 * eps);
            let tol = 1e-9 + 1e-5 * fd.abs().max(grad[c].abs());
            assert!(
                (grad[c] - fd).abs() <= tol,
                "siou case {idx} coord {c}: analytic {} vs fd {fd} (diff {})",
                grad[c],
                (grad[c] - fd).abs()
            );
        }
    }
    // Scale invariance across 24 orders of magnitude (`1e-12` vs `1.0` vs `1e12`).
    let l1 = siou_box_f64([0.0, 0.0, 1.0, 2.0], [0.4, 0.7, 1.8, 3.1], 4.0).0;
    let l_tiny = siou_box_f64([0.0, 0.0, 1e-12, 2e-12], [0.4e-12, 0.7e-12, 1.8e-12, 3.1e-12], 4.0).0;
    let l_huge = siou_box_f64([0.0, 0.0, 1e12, 2e12], [0.4e12, 0.7e12, 1.8e12, 3.1e12], 4.0).0;
    assert!((l1 - l_tiny).abs() < 1e-12, "siou tiny scale invariance: {l1} vs {l_tiny}");
    assert!((l1 - l_huge).abs() < 1e-12, "siou huge scale invariance: {l1} vs {l_huge}");
}

/// Algebraic identities for `EIoU` and `SIoU`:
/// 1. Equal-sized shifted boxes: `EIoU` reduces identically to `1 - DIoU`.
/// 2. Concentric different-sized boxes: `EIoU` equals `(1 - IoU) + (dw/cw)² + (dh/ch)²`.
/// 3. `SIoU` matches Gevorgyan (2022)'s explicit `arcsin` trigonometric formula to `1e-12` across angles.
#[test]
fn eiou_and_siou_algebraic_identities() {
    // 1. Equal-sized shifted boxes: EIoU == 1 - DIoU.
    let a = Xyxy::new(1.0, 2.0, 7.0, 10.0);
    let b = Xyxy::new(4.0, 6.0, 10.0, 14.0);
    let ta = Tensor::from_slice(&a.to_array(), [1, 4]).unwrap();
    let tb = Tensor::from_slice(&b.to_array(), [1, 4]).unwrap();
    let l_eiou = eiou_loss(&ta, &tb).unwrap().to_vec::<f32>().unwrap()[0];
    let l_diou = 1.0 - scalar_iou(&a, &b, IouKind::Diou);
    assert!((l_eiou - l_diou).abs() < 1e-6, "equal-size shifted EIoU {l_eiou} vs 1-DIoU {l_diou}");

    // 2. Concentric different-sized boxes: exact decoupled width/height terms.
    let outer = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let inner = Xyxy::new(2.0, 3.0, 8.0, 7.0);
    let to = Tensor::from_slice(&outer.to_array(), [1, 4]).unwrap();
    let ti = Tensor::from_slice(&inner.to_array(), [1, 4]).unwrap();
    let l_conc = eiou_loss(&to, &ti).unwrap().to_vec::<f32>().unwrap()[0];
    let want_iou = scalar_iou(&outer, &inner, IouKind::Iou);
    let want = (1.0 - want_iou) + (4.0f32 / 10.0).powi(2) + (6.0f32 / 10.0).powi(2);
    assert!((l_conc - want).abs() < 1e-6, "concentric EIoU {l_conc} vs {want}");

    // 3. SIoU matches Gevorgyan (2022) Eq. 1-13 with explicit arcsin across all octants & angles.
    let pairs = [
        ([0.0f64, 0.0, 4.0, 4.0], [6.0, 0.0, 10.0, 4.0]), // 0 deg (collinear)
        ([0.0, 0.0, 4.0, 4.0], [5.0, 2.0, 9.0, 7.0]),     // ~22 deg
        ([0.0, 0.0, 4.0, 4.0], [4.0, 4.0, 8.0, 8.0]),     // 45 deg
        ([0.0, 0.0, 4.0, 4.0], [1.5, 6.0, 6.5, 12.0]),    // ~76 deg
        ([5.0, 7.0, 11.0, 15.0], [1.0, 2.0, 5.0, 9.0]),   // negative dx, dy
    ];
    for (pa, pb) in pairs {
        let theta = 4.0f64;
        let (got, _) = siou_box_f64(pa, pb, theta);
        // Reference Gevorgyan (2022) trig form.
        let (w1, h1) = (pa[2] - pa[0], pa[3] - pa[1]);
        let (w2, h2) = (pb[2] - pb[0], pb[3] - pb[1]);
        let iw = (pa[2].min(pb[2]) - pa[0].max(pb[0])).max(0.0);
        let ih = (pa[3].min(pb[3]) - pa[1].max(pb[1])).max(0.0);
        let inter = iw * ih;
        let iou = inter / (w1 * h1 + w2 * h2 - inter);
        let dx = 0.5 * ((pb[0] + pb[2]) - (pa[0] + pa[2]));
        let dy = 0.5 * ((pb[1] + pb[3]) - (pa[1] + pa[3]));
        let sigma = dx.hypot(dy);
        let x = dx.abs().min(dy.abs()) / sigma;
        let lambda = 1.0 - 2.0 * (x.asin() - std::f64::consts::FRAC_PI_4).sin().powi(2);
        let gamma = 2.0 - lambda;
        let cw = pa[2].max(pb[2]) - pa[0].min(pb[0]);
        let ch = pa[3].max(pb[3]) - pa[1].min(pb[1]);
        let delta = (1.0 - (-gamma * (dx / cw).powi(2)).exp()) + (1.0 - (-gamma * (dy / ch).powi(2)).exp());
        let ow = (w1 - w2).abs() / w1.max(w2);
        let oh = (h1 - h2).abs() / h1.max(h2);
        let omega = (1.0 - (-ow).exp()).powf(theta) + (1.0 - (-oh).exp()).powf(theta);
        let want = (1.0 - iou) + 0.5 * (delta + omega);
        assert!((got - want).abs() < 1e-12, "SIoU trig ref mismatch: got {got} vs want {want}");
    }
}

/// Edge cases for `EIoU` and `SIoU`: identical (including tiny `1e-12`), disjoint, collinear, zero-area point,
/// zero-width line, inverted, huge `1e15`, empty `[0, 4]`, `NaN`/`±Inf` poison, and invalid `theta` rejection.
#[test]
fn eiou_and_siou_edge_cases_and_autograd() {
    // 1. Huge 1e15 boxes and extreme inverted [1e30, 0, 0, 0] vs [0, 0, 1e-30, 0] stay finite in forward & backward.
    let huge_a =
        Tensor::from_slice(&[0.0f32, 0.0, 1e15, 1e15, 1e30, 0.0, 0.0, 0.0], [2, 4]).unwrap().requires_grad_();
    let huge_b = Tensor::from_slice(&[1e14f32, 2e14, 1.1e15, 1.2e15, 0.0, 0.0, 1e-30, 0.0], [2, 4]).unwrap();
    for l in [eiou_loss(&huge_a, &huge_b).unwrap(), siou_loss(&huge_a, &huge_b, 4.0).unwrap()] {
        let vs = l.to_vec::<f32>().unwrap();
        assert!(
            vs.iter().all(|&v| v.is_finite() && (0.0..=4.0).contains(&v)),
            "extreme losses out of range: {vs:?}"
        );
        let g = l.sum_all().unwrap().backward().unwrap().get(&huge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|x| x.is_finite()), "extreme grad non-finite: {g:?}");
    }

    // 2. Degenerate / tiny identical / collinear / zero-area / inverted / identical boxes.
    let edge_a = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny identical -> exact 0.0 loss & 0.0 grad
            5.0, 5.0, 5.0, 5.0, // zero-area point
            10.0, 12.0, 4.0, 6.0, // inverted
            1.0, 1.0, 1.0, 8.0, // zero-width vertical line
            0.0, 0.0, 4.0, 4.0, // collinear disjoint
            0.0, 0.0, 10.0, 10.0, // identical to GT
        ],
        [6, 4],
    )
    .unwrap()
    .requires_grad_();
    let edge_b = Tensor::from_slice(
        &[
            0.0f32, 0.0, 1e-12, 1e-12, // tiny identical
            5.0, 5.0, 5.0, 5.0, // zero-area point == point
            2.0, 3.0, 9.0, 11.0, // normal GT
            1.0, 1.0, 1.0, 8.0, // identical zero-width line
            10.0, 0.0, 14.0, 4.0, // horizontally collinear disjoint
            0.0, 0.0, 10.0, 10.0, // identical GT
        ],
        [6, 4],
    )
    .unwrap();
    for (name, l) in
        [("eiou", eiou_loss(&edge_a, &edge_b).unwrap()), ("siou", siou_loss(&edge_a, &edge_b, 4.0).unwrap())]
    {
        let vals = l.to_vec::<f32>().unwrap();
        assert!(vals.iter().all(|v| v.is_finite() && *v >= 0.0), "{name} degenerate losses: {vals:?}");
        assert_eq!(vals[0], 0.0, "{name}: identical tiny 1e-12 box must have 0.0 loss, got {}", vals[0]);
        assert_eq!(vals[5], 0.0, "{name}: identical box must have 0.0 loss, got {}", vals[5]);
        let g = l.sum_all().unwrap().backward().unwrap().get(&edge_a).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|v| v.is_finite()), "{name} degenerate grads: {g:?}");
        assert!(g[0..4].iter().all(|v| v.abs() == 0.0), "{name} tiny identical grad: {:?}", &g[0..4]);
        assert!(g[20..24].iter().all(|v| v.abs() == 0.0), "{name} identical grad: {:?}", &g[20..24]);
    }

    // 3. NaN / ±Inf poison both forward and backward.
    for bad in [[f32::NAN, 0.0, 4.0, 4.0], [0.0, f32::INFINITY, 4.0, 4.0], [f32::NEG_INFINITY, 0.0, 4.0, 4.0]]
    {
        let bp = Tensor::from_slice(&bad, [1, 4]).unwrap().requires_grad_();
        let bg = Tensor::from_slice(&[0.0f32, 0.0, 4.0, 4.0], [1, 4]).unwrap();
        for l in [eiou_loss(&bp, &bg).unwrap(), siou_loss(&bp, &bg, 4.0).unwrap()] {
            assert!(l.to_vec::<f32>().unwrap()[0].is_nan());
            let g = l.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_nan()), "expected NaN grad for {bad:?}, got {g:?}");
        }
    }

    // 4. Empty [0, 4] and invalid parameters.
    let empty = Tensor::zeros([0, 4]);
    assert_eq!(eiou_loss(&empty, &empty).unwrap().dims(), &[0]);
    assert_eq!(siou_loss(&empty, &empty, 4.0).unwrap().dims(), &[0]);
    let b = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 2.0], [1, 4]).unwrap();
    for bad_theta in [1.9f32, 6.1, -1.0, f32::NAN, f32::INFINITY] {
        assert!(siou_loss(&b, &b, bad_theta).is_err(), "siou_loss should reject theta={bad_theta}");
    }
}

/// Frozen-assignment finite-difference check isolating `EIoU` and `SIoU` gradients through the DFL softmax
/// Jacobian (`cls_gain = 0`, `dfl_gain = 0`), plus end-to-end `detection_loss` and `LossConfig` validation.
#[test]
fn eiou_and_siou_dfl_fd_and_detection_loss() {
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(4.0, 5.0, 26.0, 27.0)], labels: vec![0] }];
    let reg_max = 4usize;
    let mut rng = folo_core::rng::Rng::seed(909);
    let raw_bx: Vec<f32> = (0..4 * reg_max * a).map(|_| rng.uniform(-1.5, 1.5)).collect();
    let raw_cx: Vec<f32> = (0..2 * a).map(|_| rng.uniform(-1.0, 1.0)).collect();

    for kind in [BoxLossKind::Eiou, BoxLossKind::Siou] {
        let cfg = LossConfig {
            box_gain: 7.5,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            reg_max,
            box_loss: kind,
            siou_theta: 4.0,
            ..LossConfig::default()
        };
        let bx = Tensor::from_vec(raw_bx.clone(), [1, 4 * reg_max, a]).unwrap().requires_grad_();
        let cx = Tensor::from_vec(raw_cx.clone(), [1, 2, a]).unwrap();
        let asg = assign_batch(&bx, &cx, &anchors, &targets, &cfg).unwrap();
        assert!(asg.num_fg() > 0);

        let out = loss_with_assignment(&bx, &cx, &anchors, &asg, &cfg).unwrap();
        assert!(out.is_finite() && out.items.cls == 0.0 && out.items.dfl == 0.0);
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
                &cfg,
            )
            .unwrap()
            .total_value;
            let lm = loss_with_assignment(
                &Tensor::from_vec(bm, [1, 4 * reg_max, a]).unwrap(),
                &cx,
                &anchors,
                &asg,
                &cfg,
            )
            .unwrap()
            .total_value;
            let fd = (lp - lm) / (2.0 * eps);
            let tol = 1e-3 + 2e-2 * fd.abs().max(grad[idx].abs());
            assert!((grad[idx] - fd).abs() <= tol, "{kind:?} DFL ch={ch}: analytic {} vs fd {fd}", grad[idx]);
        }

        // End-to-end detection_loss with reg_max=2,16 and nwd_weight=0,0.5 + serde round-trip.
        let anchors64 = make_anchors(64, 64, &[8, 16], 0.5);
        let a64 = anchors64.len();
        let targets64 =
            vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 44.0, 48.0)], labels: vec![0] }];
        for r in [2usize, 16] {
            for nwd_weight in [0.0f32, 0.5] {
                let mut rng2 = folo_core::rng::Rng::seed(404 + r as u64);
                let bx2 = Tensor::rand_uniform([1, 4 * r, a64], -2.0, 3.0, &mut rng2).requires_grad_();
                let cx2 = Tensor::rand_uniform([1, 2, a64], -3.0, 2.0, &mut rng2).requires_grad_();
                let cfg2 = LossConfig {
                    reg_max: r,
                    box_loss: kind,
                    siou_theta: 3.5,
                    nwd_weight,
                    ..LossConfig::default()
                };
                let out2 = detection_loss(&bx2, &cx2, &anchors64, &targets64, &cfg2).unwrap();
                assert!(out2.is_finite() && out2.num_fg > 0, "{kind:?} reg_max={r} nwd={nwd_weight}");
                let g2 = out2.total.backward().unwrap();
                let gb2 = g2.get(&bx2).unwrap().to_vec::<f32>().unwrap();
                assert!(gb2.iter().all(|v| v.is_finite()) && gb2.iter().any(|v| v.abs() > 1e-6));
            }
        }
        let sc = LossConfig { box_loss: kind, siou_theta: 5.0, ..LossConfig::default() };
        assert!(sc.validate_params().is_ok());
        let json = serde_json::to_string(&sc).unwrap();
        assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), sc);
    }

    for bad in [1.9f32, 6.1, f32::NAN, f32::INFINITY] {
        let cfg = LossConfig { siou_theta: bad, ..LossConfig::default() };
        assert!(cfg.validate_params().is_err(), "siou_theta={bad} should fail");
    }
}
