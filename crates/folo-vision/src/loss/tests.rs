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

/// Finite-difference gradient verification + fused/composite parity for `R-PIoU` and `OT-DFL` (`reg_max = 16`,
/// `reg_max = 2`, extreme logits `±500` on an active foreground anchor, and `[-1e8, -1e8, -1e8, 0]` exact ratios).
#[test]
fn new_losses_finite_diff_and_edge_cases() {
    let anchors = make_anchors(64, 64, &[8, 16], 0.5);
    let a = anchors.len();
    let targets = vec![ImageTargets {
        boxes: vec![Xyxy::new(10.0, 12.0, 46.0, 50.0), Xyxy::new(22.0, 18.0, 34.0, 30.0)],
        labels: vec![0, 1],
    }];
    let eps = 1e-3f32;
    let check_fd = |cfg: LossConfig, seed: u64, label: &str| {
        let mut rng = folo_core::rng::Rng::seed(seed);
        let r = cfg.reg_max;
        let box_v: Vec<f32> = (0..4 * r * a).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let cls_v: Vec<f32> = (0..2 * a).map(|_| rng.uniform(-5.0, 3.0)).collect();
        let b0 = Tensor::from_vec(box_v.clone(), [1, 4 * r, a]).unwrap().requires_grad_();
        let c0 = Tensor::from_vec(cls_v.clone(), [1, 2, a]).unwrap().requires_grad_();
        let asg = assign_batch(&b0, &c0, &anchors, &targets, &cfg).unwrap();
        assert!(asg.num_fg() > 0, "{label}: no positives");
        let out_f = loss_impl(&b0, &c0, &anchors, &asg, &cfg, true).unwrap();
        let out_c = loss_impl(&b0, &c0, &anchors, &asg, &cfg, false).unwrap();
        assert!((out_f.total_value - out_c.total_value).abs() <= 1e-4 * out_c.total_value.abs().max(1.0));
        let grads = out_f.total.backward().unwrap();
        let gb = grads.get(&b0).unwrap().to_vec::<f32>().unwrap();
        let gc = grads.get(&c0).unwrap().to_vec::<f32>().unwrap();
        let fg_a = asg.anchor[0] as usize;
        for ch in [0, 1, r - 1, r + 2, 2 * r + 1, 3 * r].into_iter().filter(|&c| c < 4 * r) {
            let idx = ch * a + fg_a;
            let (mut vp, mut vm) = (box_v.clone(), box_v.clone());
            vp[idx] += eps;
            vm[idx] -= eps;
            let lp =
                loss_impl(&Tensor::from_vec(vp, [1, 4 * r, a]).unwrap(), &c0, &anchors, &asg, &cfg, true)
                    .unwrap()
                    .total_value;
            let lm =
                loss_impl(&Tensor::from_vec(vm, [1, 4 * r, a]).unwrap(), &c0, &anchors, &asg, &cfg, true)
                    .unwrap()
                    .total_value;
            let num = (lp - lm) / (2.0 * eps);
            let ana = gb[idx];
            assert!((ana - num).abs() <= 5e-3 + 2e-2 * num.abs(), "{label} box[{ch}]: ana={ana} num={num}");
        }
        for idx in [fg_a, a + fg_a, 0, a + 1] {
            let (mut vp, mut vm) = (cls_v.clone(), cls_v.clone());
            vp[idx] += eps;
            vm[idx] -= eps;
            let lp = loss_impl(&b0, &Tensor::from_vec(vp, [1, 2, a]).unwrap(), &anchors, &asg, &cfg, true)
                .unwrap()
                .total_value;
            let lm = loss_impl(&b0, &Tensor::from_vec(vm, [1, 2, a]).unwrap(), &anchors, &asg, &cfg, true)
                .unwrap()
                .total_value;
            let num = (lp - lm) / (2.0 * eps);
            let ana = gc[idx];
            assert!((ana - num).abs() <= 5e-3 + 2e-2 * num.abs(), "{label} cls[{idx}]: ana={ana} num={num}");
        }
    };
    check_fd(
        LossConfig { box_loss: BoxLossKind::Rpiou, dfl_ot_weight: 0.25, ..LossConfig::default() },
        7,
        "rpiou+ot",
    );
    check_fd(
        LossConfig { box_loss: BoxLossKind::Rpiou, dfl_ot_weight: 0.25, reg_max: 2, ..LossConfig::default() },
        8,
        "reg_max=2",
    );

    // Extreme logits (±500) placed directly on an active foreground anchor remain finite for R-PIoU + OT-DFL.
    let fg_anchor = 2 * 8 + 3; // stride-8 anchor at (28, 20), inside both GT boxes.
    let mut extreme_box = vec![-500.0f32; 4 * 16 * a];
    let mut extreme_cls = vec![-25.0f32; 2 * a];
    for side in 0..4 {
        extreme_box[(side * 16 + 15) * a + fg_anchor] = 500.0;
    }
    extreme_cls[fg_anchor] = 25.0;
    let bx = Tensor::from_vec(extreme_box, [1, 64, a]).unwrap().requires_grad_();
    let cx = Tensor::from_vec(extreme_cls, [1, 2, a]).unwrap().requires_grad_();
    let cfg = LossConfig { box_loss: BoxLossKind::Rpiou, dfl_ot_weight: 0.35, ..LossConfig::default() };
    let out = detection_loss_host(&bx, &cx, &anchors, &targets, &cfg).unwrap();
    assert!(out.num_fg > 0 && out.is_finite(), "extreme logits produced non-finite loss: {:?}", out.items);
    let g = out.total.backward().unwrap();
    let gb = g.get(&bx).unwrap().to_vec::<f32>().unwrap();
    assert!(gb.iter().all(|v| v.is_finite()));
    assert!(gb[(15 * a) + fg_anchor].abs() > 0.0 || gb[fg_anchor].abs() > 0.0, "fg anchor got gradient");
    assert!(g.get(&cx).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));

    // Exact OT-DFL gradient ratios on extreme finite logits `[-1e8, -1e8, -1e8, 0]` with `R = 4, j = 0`:
    // unweighted OT gradient must be `[-5/6, -5/6, -1/3, +2]` (zero-sum shift invariance, no f32 ln(2)/ln(3) loss).
    let r4 = 4usize;
    let mut asg_ot = BatchAssignment {
        batch: 1,
        num_classes: 1,
        num_anchors: a,
        inputs_finite: true,
        ..Default::default()
    };
    asg_ot.image.push(0);
    asg_ot.anchor.push(fg_anchor as u32);
    asg_ot.label.push(0);
    asg_ot.score.push(1.0);
    // Anchor at (28, 20) stride 8 -> grid (3.5, 2.5); target box centered with ltrb = (0.5, 0.5, 0.5, 0.5) => j = 0.
    asg_ot.target.push(Xyxy::new(24.0, 16.0, 32.0, 24.0));
    let mut b4 = vec![0.0f32; 4 * r4 * a];
    for side in 0..4 {
        b4[(side * r4) * a + fg_anchor] = -1e8;
        b4[(side * r4 + 1) * a + fg_anchor] = -1e8;
        b4[(side * r4 + 2) * a + fg_anchor] = -1e8;
        b4[(side * r4 + 3) * a + fg_anchor] = 0.0;
    }
    let bx4 = Tensor::from_vec(b4, [1, 4 * r4, a]).unwrap().requires_grad_();
    let cx1 = Tensor::zeros([1, 1, a]).requires_grad_();
    let cfg_base =
        LossConfig { box_gain: 0.0, cls_gain: 0.0, dfl_gain: 1.0, reg_max: r4, ..LossConfig::default() };
    let cfg_ot = LossConfig { dfl_ot_weight: 1.0, ..cfg_base };
    let g_base = loss_with_assignment(&bx4, &cx1, &anchors, &asg_ot, &cfg_base)
        .unwrap()
        .total
        .backward()
        .unwrap()
        .get(&bx4)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let g_ot = loss_with_assignment(&bx4, &cx1, &anchors, &asg_ot, &cfg_ot)
        .unwrap()
        .total
        .backward()
        .unwrap()
        .get(&bx4)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let want = [-5.0f32 / 6.0, -5.0 / 6.0, -1.0 / 3.0, 2.0];
    for side in 0..4 {
        let mut sum = 0.0f32;
        for (k, &w) in want.iter().enumerate() {
            let idx = (side * r4 + k) * a + fg_anchor;
            // Side weight is t / 4 = 0.25.
            let got = (g_ot[idx] - g_base[idx]) * 4.0;
            sum += got;
            assert!((got - w).abs() < 1e-5, "side {side} bin {k}: got {got}, want {w}");
        }
        assert!(sum.abs() < 1e-5, "side {side}: OT gradient sum must be 0, got {sum}");
    }

    // Regression: CPU R-PIoU must reuse the tensor-decoded box (`softmax` -> `dist` -> `pd`) bit-for-bit so a
    // ground truth matched to the tensor-decoded box incurs zero edge penalty even at `kappa = 1e16`.
    let p_left =
        Tensor::from_slice(&[0.0f32, -2.0], [1, 2]).unwrap().softmax(1).unwrap().to_vec::<f32>().unwrap();
    let d_left = p_left[1];
    let anchors1 =
        crate::decode::Anchors { points: vec![[d_left, 0.5]], strides: vec![1.0], ..Default::default() };
    let bx2 = Tensor::from_slice(&[0.0f32, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 8, 1]).unwrap();
    let cx1b = Tensor::zeros([1, 1, 1]);
    let mut asg1 = BatchAssignment { batch: 1, num_classes: 1, num_anchors: 1, ..Default::default() };
    asg1.image.push(0);
    asg1.anchor.push(0);
    asg1.label.push(0);
    asg1.score.push(1.0);
    asg1.target.push(Xyxy::new(0.0, 0.0, d_left + 0.5, 1.0));
    let cfg_rpi = LossConfig {
        reg_max: 2,
        box_loss: BoxLossKind::Rpiou,
        rpiou_kappa: 1e16,
        rpiou_delta: 0.1,
        box_gain: 1.0,
        cls_gain: 0.0,
        dfl_gain: 0.0,
        ..LossConfig::default()
    };
    let out_rpi = loss_with_assignment(&bx2, &cx1b, &anchors1, &asg1, &cfg_rpi).unwrap();
    assert!(
        out_rpi.total_value.abs() < 1e-6,
        "exact tensor-decoded match must have 0 R-PIoU loss, got {}",
        out_rpi.total_value
    );
}
