//! Unit tests of the loss internals: the fused CPU kernels against the composite path, confident positives,
//! huge foreground logits, duplicate anchors, and the fixed-order sum.

use super::terms::{loss_impl, sum_fixed_order};
use super::*;
use crate::decode::make_anchors;

/// The fused CPU kernels and the device-generic composite give the same loss and gradients.
#[test]
fn fused_matches_composite() {
    let anchors = make_anchors(96, 96, &[8, 16, 32], 0.5);
    let (b, nc, a, r) = (3, 5, anchors.len(), 16);
    let mut rng = folo_core::rng::Rng::seed(3);
    let box_raw = Tensor::rand_uniform([b, 4 * r, a], 0.0, 4.0, &mut rng).requires_grad_();
    let cls_raw = Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng).requires_grad_();
    let gt = |boxes: Vec<Xyxy>, labels: Vec<u32>| ImageTargets { boxes, labels };
    let targets = vec![
        gt(vec![Xyxy::new(10.0, 12.0, 60.0, 70.0), Xyxy::new(40.0, 5.0, 90.0, 40.0)], vec![1, 4]),
        gt(vec![], vec![]),
        gt(vec![Xyxy::new(20.0, 30.0, 34.0, 41.0)], vec![0]),
    ];
    let cfg = LossConfig::default();
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    assert!(asg.num_fg() > 0);
    let fused = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, true).unwrap();
    let composite = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, false).unwrap();
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs()));
    let (fi, ci) = (fused.items, composite.items);
    assert!(close(fi.box_, ci.box_) && close(fi.cls, ci.cls) && close(fi.dfl, ci.dfl), "{fi:?} {ci:?}");
    let (gf, gc) = (fused.total.backward().unwrap(), composite.total.backward().unwrap());
    for t in [&box_raw, &cls_raw] {
        let (x, y) =
            (gf.get(t).unwrap().to_vec::<f32>().unwrap(), gc.get(t).unwrap().to_vec::<f32>().unwrap());
        for (i, (x, y)) in x.iter().zip(&y).enumerate() {
            assert!((x - y).abs() <= 1e-6 + 1e-5 * y.abs(), "element {i}: {x} vs {y}");
        }
    }
}

/// Confident positives (logit ≈ 80, t = 1) among background logits: BCE is non-negative and tiny, on both
/// kernel paths. Summing softplus and `x · t` separately would leave an error of about ulp(80) ≈ 8e-6, of
/// either sign.
#[test]
fn confident_positives_do_not_cancel() {
    let anchors = make_anchors(64, 64, &[8], 0.5);
    let a = anchors.len();
    let mut cls = vec![-30.0f32; 2 * a];
    let fg = [3usize, 17, 40, 63];
    for (k, &an) in fg.iter().enumerate() {
        cls[a + an] = 80.4 + 0.1 * k as f32; // class 1
    }
    let cls_raw = Tensor::from_vec(cls, [1, 2, a]).unwrap().requires_grad_();
    let box_raw = Tensor::zeros([1, 64, a]);
    let asg = BatchAssignment {
        batch: 1,
        num_anchors: a,
        num_classes: 2,
        image: vec![0; 4],
        anchor: fg.iter().map(|&an| an as u32).collect(),
        label: vec![1; 4],
        score: vec![1.0; 4],
        target: vec![Xyxy::new(0.0, 0.0, 64.0, 64.0); 4],
        inputs_finite: true,
    };
    let cfg = LossConfig::default();
    // Background: 2A − 4 logits at −30; foreground terms ≈ e^-80 ≈ 0. tss = 4.
    let want = 0.5 * (2 * a - 4) as f64 * (-30f64).exp().ln_1p() / 4.0;
    for fused in [true, false] {
        let out = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &cfg, fused).unwrap();
        let got = out.items.cls as f64;
        assert!(got >= 0.0 && (got - want).abs() <= 1e-6 * want, "fused {fused}: {got} vs {want}");
        // d/dx = B · 0.5 · (sigmoid(x) − T) / tss ≈ 0 at the positives.
        let g = out.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        assert!(fg.iter().all(|&an| g[a + an].abs() < 1e-9), "fused {fused}");
    }
}

/// Huge finite foreground logits (2^32, where `x − 100` rounds back to `x` in f32) are excluded from the dense
/// sum exactly on both paths: the loss is the background part alone and the positives' gradient is
/// `sigmoid(x) − t = 0`.
#[test]
fn huge_foreground_logits_are_masked_exactly() {
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    let mut cls = vec![-5.0f32; a];
    cls[2] = 4_294_967_296.0;
    cls[9] = -4_294_967_296.0;
    let cls_raw = Tensor::from_vec(cls, [1, 1, a]).unwrap().requires_grad_();
    let box_raw = Tensor::zeros([1, 64, a]);
    let asg = BatchAssignment {
        batch: 1,
        num_anchors: a,
        num_classes: 1,
        image: vec![0, 0],
        anchor: vec![2, 9],
        label: vec![0, 0],
        score: vec![1.0, 0.0],
        target: vec![Xyxy::new(0.0, 0.0, 32.0, 32.0); 2],
        inputs_finite: true,
    };
    let want = 0.5 * (a - 2) as f64 * (-5f64).exp().ln_1p(); // tss = 1
    for fused in [true, false] {
        let out = loss_impl(&box_raw, &cls_raw, &anchors, &asg, &LossConfig::default(), fused).unwrap();
        assert!(
            (out.items.cls as f64 - want).abs() <= 1e-6 * want,
            "fused {fused}: {} vs {want}",
            out.items.cls
        );
        let g = out.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        assert_eq!((g[2], g[9]), (0.0, 0.0), "fused {fused}");
    }
}

/// Duplicate foreground anchors are rejected (an anchor has at most one gt).
#[test]
fn rejects_duplicate_anchors() {
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    let asg = BatchAssignment {
        batch: 1,
        num_anchors: a,
        num_classes: 2,
        image: vec![0, 0],
        anchor: vec![5, 5],
        label: vec![0, 1],
        score: vec![0.5, 0.5],
        target: vec![Xyxy::new(0.0, 0.0, 32.0, 32.0); 2],
        inputs_finite: true,
    };
    let (box_raw, cls_raw) = (Tensor::zeros([1, 64, a]), Tensor::zeros([1, 2, a]));
    assert!(loss_with_assignment(&box_raw, &cls_raw, &anchors, &asg, &LossConfig::default()).is_err());
}

#[test]
fn sum_fixed_order_matches_sum() {
    for n in [1, 4096, 4097, 3 * 4096 + 5, 40_000_000 / 1000] {
        let x = Tensor::arange(0.0, n as f32, 1.0).unwrap().mul_scalar(1e-3).unwrap();
        let want = (n as f64 - 1.0) * n as f64 / 2.0 * 1e-3;
        let got = sum_fixed_order(&x).unwrap().item::<f32>().unwrap() as f64;
        assert!((got - want).abs() <= 1e-5 * want.max(1.0), "{n}: {got} vs {want}");
    }
}
