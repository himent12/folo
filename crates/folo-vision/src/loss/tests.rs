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

    // Public `rpiou_loss` rejects mismatched batch lengths, bad column counts, and wrong rank without panicking.
    let b2x4 = Tensor::zeros([2, 4]);
    let b1x4 = Tensor::zeros([1, 4]);
    let b1x3 = Tensor::zeros([1, 3]);
    let b4 = Tensor::zeros([4]);
    assert!(rpiou_loss(&b2x4, &b1x4, 0.5, 2.0, 0.5).is_err());
    assert!(rpiou_loss(&b1x4, &b2x4, 0.5, 2.0, 0.5).is_err());
    assert!(rpiou_loss(&b1x3, &b1x3, 0.5, 2.0, 0.5).is_err());
    assert!(rpiou_loss(&b4, &b1x4, 0.5, 2.0, 0.5).is_err());
}

#[test]
fn vfl_and_qfl_fused_matches_composite_and_grad_checks() {
    use folo_core::testing::{GradCheck, grad_check_with};

    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let (b, nc, a, r) = (2, 4, anchors.len(), 16);
    let mut rng = folo_core::rng::Rng::seed(707);
    let box_raw = Tensor::rand_uniform([b, 4 * r, a], 0.0, 3.0, &mut rng);
    let cls_raw = Tensor::rand_uniform([b, nc, a], -4.0, 3.0, &mut rng).requires_grad_();
    let targets = vec![
        ImageTargets { boxes: vec![Xyxy::new(8.0, 8.0, 40.0, 44.0)], labels: vec![1] },
        ImageTargets { boxes: vec![Xyxy::new(16.0, 12.0, 56.0, 52.0)], labels: vec![3] },
    ];
    let base_asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    assert!(base_asg.num_fg() > 0);

    let configs = [
        LossConfig { cls_loss: ClsLossKind::Vfl, vfl_alpha: 0.75, vfl_gamma: 2.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Vfl, vfl_alpha: 0.5, vfl_gamma: 0.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Qfl, qfl_beta: 2.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Qfl, qfl_beta: 1.5, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Qfl, qfl_beta: 1.0, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Qfl, qfl_beta: 0.0, ..LossConfig::default() },
    ];
    for cfg in configs {
        let fused = loss_impl(&box_raw, &cls_raw, &anchors, &base_asg, &cfg, true).unwrap();
        let comp = loss_impl(&box_raw, &cls_raw, &anchors, &base_asg, &cfg, false).unwrap();
        assert!(
            (fused.items.cls - comp.items.cls).abs() <= 1e-5 * (1.0 + comp.items.cls.abs()),
            "{cfg:?}: fused {} vs comp {}",
            fused.items.cls,
            comp.items.cls
        );
        let gf = fused.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        let gc = comp.total.backward().unwrap().get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
        for (i, (&x, &y)) in gf.iter().zip(&gc).enumerate() {
            assert!(x.is_finite() && y.is_finite(), "{cfg:?}: non-finite grad at {i}");
            assert!((x - y).abs() <= 2e-5 + 2e-4 * y.abs(), "{cfg:?} elem {i}: {x} vs {y}");
        }
    }

    // Finite-difference gradient check on both VFL and QFL (fused CPU and composite).
    let small_anchors = make_anchors(32, 32, &[16, 32], 0.5);
    let sa = small_anchors.len();
    let sbox = Tensor::rand_uniform([1, 4 * 4, sa], 0.0, 2.0, &mut rng);
    let scls = Tensor::rand_uniform([1, 2, sa], -2.0, 2.0, &mut rng);
    let sasg = BatchAssignment {
        batch: 1,
        num_anchors: sa,
        num_classes: 2,
        image: vec![0, 0],
        anchor: vec![0, sa as u32 - 1],
        label: vec![0, 1],
        score: vec![0.7, 0.35],
        target: vec![Xyxy::new(2.0, 2.0, 20.0, 22.0), Xyxy::new(6.0, 6.0, 28.0, 28.0)],
        inputs_finite: true,
    };
    for cfg in [
        LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: ClsLossKind::Vfl,
            ..LossConfig::default()
        },
        LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: ClsLossKind::Qfl,
            qfl_beta: 2.0,
            ..LossConfig::default()
        },
    ] {
        for fused in [true, false] {
            grad_check_with(
                |inputs| Ok(loss_impl(&sbox, &inputs[0], &small_anchors, &sasg, &cfg, fused)?.total),
                std::slice::from_ref(&scls),
                &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 11 },
            )
            .unwrap();
        }
    }

    // QFL exact p == t zero-diff point produces exact 0 loss and finite 0 gradient on both paths,
    // including the beta == 1.0 cusp at x = ln(3), t = 0.75.
    let p_t = 0.75f32;
    let logit_exact = (p_t / (1.0 - p_t)).ln();
    let c_exact = Tensor::from_vec(vec![logit_exact], [1, 1, 1]).unwrap().requires_grad_();
    let b_exact = Tensor::zeros([1, 8, 1]);
    let a1 = crate::decode::Anchors { points: vec![[0.5, 0.5]], strides: vec![1.0], ..Default::default() };
    let asg1 = BatchAssignment {
        batch: 1,
        num_anchors: 1,
        num_classes: 1,
        image: vec![0],
        anchor: vec![0],
        label: vec![0],
        score: vec![p_t],
        target: vec![Xyxy::new(0.0, 0.0, 1.0, 1.0)],
        inputs_finite: true,
    };
    for qfl_beta in [1.0, 1.5, 2.0] {
        let qfl_cfg = LossConfig {
            reg_max: 2,
            box_gain: 0.0,
            dfl_gain: 0.0,
            cls_gain: 1.0,
            cls_loss: ClsLossKind::Qfl,
            qfl_beta,
            ..LossConfig::default()
        };
        let mut grads = Vec::new();
        for fused in [true, false] {
            let out = loss_impl(&b_exact, &c_exact, &a1, &asg1, &qfl_cfg, fused).unwrap();
            assert!(
                out.items.cls.abs() < 1e-6,
                "beta={qfl_beta} fused={fused}: expected 0 at p==t, got {}",
                out.items.cls
            );
            let g = out.total.backward().unwrap().get(&c_exact).unwrap().to_vec::<f32>().unwrap()[0];
            assert!(
                g.is_finite() && g.abs() < 1e-5,
                "beta={qfl_beta} fused={fused}: expected ~0 grad at p==t, got {g}"
            );
            grads.push(g);
        }
        assert!(
            (grads[0] - grads[1]).abs() < 1e-6,
            "beta={qfl_beta}: fused grad {} != composite grad {}",
            grads[0],
            grads[1]
        );
    }
}

#[test]
fn inner_ciou_and_wiou_v3_parity_and_grad_checks() {
    use folo_core::testing::{GradCheck, grad_check_with};

    // 1. Inner-CIoU at ratio == 1.0 matches `1 - ciou` identically, including thin high-offset boxes.
    let b1 = Tensor::from_slice(
        &[2.0f32, 3.0, 14.0, 17.0, 10.0, 10.0, 30.0, 26.0, 1.0, 2.0, 6.0, 9.0, 100.0, 0.0, 100.00001, 1.0],
        [4, 4],
    )
    .unwrap();
    let b2 = Tensor::from_slice(
        &[3.0f32, 2.0, 15.0, 18.0, 14.0, 12.0, 32.0, 29.0, 8.0, 8.0, 15.0, 16.0, 100.0, 0.0, 100.00001, 1.0],
        [4, 4],
    )
    .unwrap();
    let l_ciou = ciou(&b1, &b2).unwrap().rsub_scalar(1.0).unwrap().to_vec::<f32>().unwrap();
    let l_inner1 = inner_ciou_loss(&b1, &b2, 1.0).unwrap().to_vec::<f32>().unwrap();
    for (a, b) in l_ciou.iter().zip(&l_inner1) {
        assert!((a - b).abs() < 1e-6, "inner_ratio=1.0 must match 1 - ciou: {a} vs {b}");
    }

    // Inverted boxes in Inner-CIoU stay finite with finite gradients (never 0/0 or -EPS cancellation).
    let inv_pred =
        Tensor::from_slice(&[0.0f32, 0.0, 1.0, -1e-7, 5.0, 0.0, 1.0, 4.0], [2, 4]).unwrap().requires_grad_();
    let inv_gt = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 4.0, 4.0], [2, 4]).unwrap();
    let l_inv = inner_ciou_loss(&inv_pred, &inv_gt, 0.7).unwrap();
    assert!(l_inv.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    let g_inv = l_inv.sum_all().unwrap().backward().unwrap().get(&inv_pred).unwrap().to_vec::<f32>().unwrap();
    assert!(g_inv.iter().all(|v| v.is_finite()), "inverted Inner-CIoU grads must be finite: {g_inv:?}");

    // 2. Finite-difference gradient check for Inner-CIoU (ratio = 0.7) wrt b1 on square boxes (v = 0, alpha = 0
    //    so detached alpha introduces zero bias), plus general-aspect finite-gradient check.
    let sq1 = Tensor::from_slice(&[2.0f32, 3.0, 14.0, 15.0], [1, 4]).unwrap();
    let sq2 = Tensor::from_slice(&[4.0f32, 5.0, 16.0, 17.0], [1, 4]).unwrap();
    grad_check_with(
        |inputs| inner_ciou_loss(&inputs[0], &sq2, 0.7)?.sum_all(),
        &[sq1],
        &GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 4, seed: 5 },
    )
    .unwrap();
    let b1_leaf = b1.detach().requires_grad_();
    let li = inner_ciou_loss(&b1_leaf, &b2, 0.7).unwrap();
    let gi = li.sum_all().unwrap().backward().unwrap().get(&b1_leaf).unwrap().to_vec::<f32>().unwrap();
    assert!(gi.iter().all(|v| v.is_finite()) && gi.iter().any(|v| v.abs() > 1e-4));

    // 3. WIoU v3 with explicit running-EMA `iou_mean`: differentiates single-anchor near-perfect vs disjoint
    //    boxes and propagates NaN/Inf in both forward and backward.
    let w_pred = Tensor::from_slice(
        &[
            0.0f32, 0.0, 10.0, 10.0, // near-perfect (IoU ≈ 1.0 -> beta ≈ 0)
            1.5, 1.5, 11.5, 11.5, // moderate overlap
            30.0, 30.0, 40.0, 40.0, // disjoint outlier (L_IoU = 1.0)
        ],
        [3, 4],
    )
    .unwrap();
    let w_gt =
        Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0], [3, 4])
            .unwrap();
    let l_wiou = wiou_v3_loss(&w_pred, &w_gt, 1.9, 3.0, 0.5).unwrap().to_vec::<f32>().unwrap();
    assert!(l_wiou.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(l_wiou[0] < 1e-5, "identical box should have ~0 WIoU v3 loss, got {}", l_wiou[0]);

    // Single-anchor differentiation via running EMA denominator (Tong et al. §III-B).
    let good_one = Tensor::from_slice(&[0.2f32, 0.2, 10.0, 10.0], [1, 4]).unwrap();
    let bad_one = Tensor::from_slice(&[20.0f32, 20.0, 30.0, 30.0], [1, 4]).unwrap();
    let gt_one = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 10.0], [1, 4]).unwrap();
    let l_good = wiou_v3_loss(&good_one, &gt_one, 1.9, 3.0, 0.5).unwrap().to_vec::<f32>().unwrap()[0];
    let l_bad = wiou_v3_loss(&bad_one, &gt_one, 1.9, 3.0, 0.5).unwrap().to_vec::<f32>().unwrap()[0];
    assert!(
        l_good > 0.0 && l_bad > l_good * 10.0,
        "single-anchor WIoU v3 must differentiate: {l_good} vs {l_bad}"
    );

    // Non-finite inputs (NaN and +Inf) must poison both forward and backward in both new box losses.
    for bad_row in [[f32::NAN, 0.0, 2.0, 2.0], [f32::INFINITY, 20.0, 30.0, 30.0]] {
        let bp = Tensor::from_slice(&bad_row, [1, 4]).unwrap().requires_grad_();
        let bg = Tensor::from_slice(&[0.0f32, 0.0, 2.0, 2.0], [1, 4]).unwrap();
        let lw = wiou_v3_loss(&bp, &bg, 1.9, 3.0, 1.0).unwrap();
        assert!(lw.to_vec::<f32>().unwrap()[0].is_nan(), "wiou_v3_loss must return NaN for {bad_row:?}");
        let gw = lw.sum_all().unwrap().backward().unwrap().get(&bp).unwrap().to_vec::<f32>().unwrap();
        assert!(gw.iter().all(|v| v.is_nan()), "wiou_v3_loss backward must be NaN for {bad_row:?}: {gw:?}");

        let bp2 = Tensor::from_slice(&bad_row, [1, 4]).unwrap().requires_grad_();
        let li2 = inner_ciou_loss(&bp2, &bg, 0.7).unwrap();
        assert!(li2.to_vec::<f32>().unwrap()[0].is_nan(), "inner_ciou_loss must return NaN for {bad_row:?}");
        let gi2 = li2.sum_all().unwrap().backward().unwrap().get(&bp2).unwrap().to_vec::<f32>().unwrap();
        assert!(
            gi2.iter().all(|v| v.is_nan()),
            "inner_ciou_loss backward must be NaN for {bad_row:?}: {gi2:?}"
        );
    }

    // Shape validation rejects extra columns and mismatched row counts.
    let bad5 = Tensor::zeros([2, 5]);
    let ok2 = Tensor::zeros([2, 4]);
    let ok1 = Tensor::zeros([1, 4]);
    assert!(inner_ciou_loss(&bad5, &ok2, 0.7).is_err());
    assert!(inner_ciou_loss(&ok2, &ok1, 0.7).is_err());
    assert!(wiou_v3_loss(&bad5, &ok2, 1.9, 3.0, 1.0).is_err());
    assert!(wiou_v3_loss(&ok2, &ok1, 1.9, 3.0, 1.0).is_err());
    assert!(wiou_v3_loss(&ok2, &ok2, 1.9, 3.0, 0.0).is_err());

    // Scale-invariant tiny identical boxes give exact 0.0 for both WIoU v3 and public NWD,
    // extreme WIoU v3 outlier beta monotonically attenuates to 0, and public NWD rejects c <= 0.
    let tiny = Tensor::from_slice(&[0.0f32, 0.0, 1e-3, 1e-3], [1, 4]).unwrap();
    assert_eq!(wiou_v3_loss(&tiny, &tiny, 1.9, 3.0, 1.0).unwrap().to_vec::<f32>().unwrap()[0], 0.0);
    let huge_outlier = wiou_v3_loss(
        &Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], [1, 4]).unwrap(),
        &Tensor::from_slice(&[10.0f32, 10.0, 11.0, 11.0], [1, 4]).unwrap(),
        1.9,
        3.0,
        1e-4,
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap()[0];
    assert!(huge_outlier.is_finite() && huge_outlier < 1e-10, "outlier must attenuate: {huge_outlier}");
    let c_ok = Tensor::from_slice(&[12.8f32], [1]).unwrap();
    assert_eq!(nwd_loss(&tiny, &tiny, &c_ok).unwrap().to_vec::<f32>().unwrap()[0], 0.0);
    assert!(nwd_loss(&tiny, &tiny, &Tensor::from_slice(&[0.0f32], [1]).unwrap()).is_err());
    assert!(nwd_loss(&tiny, &tiny, &Tensor::from_slice(&[-1.0f32], [1]).unwrap()).is_err());
    // Finite-difference gradient check for public `nwd_loss` wrt `b1`, `b2`, and `c` on non-trivial boxes.
    let nb1 = Tensor::from_slice(&[2.0f32, 3.0, 10.0, 12.0, 0.0, 0.0, 1.0, 1.0], [2, 4]).unwrap();
    let nb2 = Tensor::from_slice(&[3.5f32, 2.5, 11.5, 13.0, 6.0, 7.0, 7.0, 8.0], [2, 4]).unwrap();
    let nc = Tensor::from_slice(&[12.8f32, 8.0], [2]).unwrap();
    let nwd_vals = nwd_loss(&nb1, &nb2, &nc).unwrap().to_vec::<f32>().unwrap();
    assert!(nwd_vals.iter().all(|&v| v.is_finite() && v > 0.0 && v < 1.0));
    grad_check_with(
        |inputs| nwd_loss(&inputs[0], &inputs[1], &inputs[2])?.sum_all(),
        &[nb1, nb2, nc],
        &GradCheck { eps: 1e-3, atol: 1e-4, rtol: 2e-2, ..GradCheck::default() },
    )
    .unwrap();

    // Gradient descent check on WIoU v3.
    let w_pair = Tensor::from_slice(&[1.0f32, 2.0, 11.0, 14.0], [1, 4]).unwrap();
    let w_target = Tensor::from_slice(&[2.5f32, 3.0, 13.0, 15.0], [1, 4]).unwrap();
    let w_leaf = w_pair.detach().requires_grad_();
    let l0 = wiou_v3_loss(&w_leaf, &w_target, 1.9, 3.0, 1.0).unwrap().sum_all().unwrap();
    let v0 = l0.item::<f32>().unwrap();
    let gw = l0.backward().unwrap().get(&w_leaf).unwrap().to_vec::<f32>().unwrap();
    assert!(gw.iter().all(|v| v.is_finite()) && gw.iter().any(|v| v.abs() > 1e-4));
    let stepped: Vec<f32> =
        w_pair.to_vec::<f32>().unwrap().iter().zip(&gw).map(|(p, g)| p - 0.1 * g).collect();
    let v1 = wiou_v3_loss(&Tensor::from_vec(stepped, [1, 4]).unwrap(), &w_target, 1.9, 3.0, 1.0)
        .unwrap()
        .sum_all()
        .unwrap()
        .item::<f32>()
        .unwrap();
    assert!(v1 < v0, "WIoU v3 gradient step must decrease loss: {v1} >= {v0}");

    // 4. End-to-end detection_loss + serde round-trip + CLI-discoverable keys + config validation.
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let mut rng = folo_core::rng::Rng::seed(99);
    let bx = Tensor::rand_uniform([1, 64, anchors.len()], 0.0, 3.0, &mut rng).requires_grad_();
    let cx = Tensor::rand_uniform([1, 3, anchors.len()], -3.0, 2.0, &mut rng).requires_grad_();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 10.0, 42.0, 46.0)], labels: vec![0] }];
    for cfg in [
        LossConfig {
            cls_loss: ClsLossKind::Vfl,
            box_loss: BoxLossKind::InnerCiou,
            inner_ratio: 0.7,
            ..LossConfig::default()
        },
        LossConfig {
            cls_loss: ClsLossKind::Qfl,
            box_loss: BoxLossKind::WiouV3,
            wiou_alpha: 1.9,
            wiou_delta: 3.0,
            ..LossConfig::default()
        },
    ] {
        assert!(!device_loss_supported(&bx, &cfg));
        let out = detection_loss(&bx, &cx, &anchors, &targets, &cfg).unwrap();
        assert!(out.is_finite(), "{cfg:?}: non-finite loss");
        if cfg.box_loss == BoxLossKind::WiouV3 {
            assert!(out.mean_iou_loss.is_some_and(|m| m.is_finite() && (0.0..=1.0).contains(&m)));
        } else {
            assert!(out.mean_iou_loss.is_none());
        }
        let g = out.total.backward().unwrap();
        assert!(g.get(&bx).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
        assert!(g.get(&cx).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(serde_json::from_str::<LossConfig>(&json).unwrap(), cfg);
    }

    // Default LossConfig exposes all keys (for CLI `--set loss.<key>=...`) and deserializes legacy JSON without them.
    let def_json = serde_json::to_string(&LossConfig::default()).unwrap();
    for key in [
        "cls_loss",
        "vfl_alpha",
        "vfl_gamma",
        "qfl_beta",
        "inner_ratio",
        "wiou_alpha",
        "wiou_delta",
        "wiou_momentum",
    ] {
        assert!(def_json.contains(key), "default LossConfig JSON must expose `{key}` for CLI `--set`");
    }
    assert_eq!(serde_json::from_str::<LossConfig>("{}").unwrap(), LossConfig::default());
    for bad in [
        LossConfig { mal_gamma: -0.5, ..LossConfig::default() },
        LossConfig { mal_gamma: f32::NAN, ..LossConfig::default() },
        LossConfig { mal_gamma: 5.5, ..LossConfig::default() },
        LossConfig { cls_loss: ClsLossKind::Vfl, mal_gamma: 1.0, ..LossConfig::default() },
        LossConfig { vfl_alpha: 0.0, ..LossConfig::default() },
        LossConfig { vfl_gamma: -0.1, ..LossConfig::default() },
        LossConfig { qfl_beta: 0.5, ..LossConfig::default() },
        LossConfig { inner_ratio: 0.4, ..LossConfig::default() },
        LossConfig { inner_ratio: 1.6, ..LossConfig::default() },
        LossConfig { wiou_alpha: 1.0, ..LossConfig::default() },
        LossConfig { wiou_delta: 0.0, ..LossConfig::default() },
        LossConfig { wiou_momentum: -0.01, ..LossConfig::default() },
        LossConfig { wiou_momentum: 1.1, ..LossConfig::default() },
    ] {
        assert!(bad.validate_params().is_err(), "{bad:?} should be rejected");
        assert!(detection_loss(&bx, &cx, &anchors, &targets, &bad).is_err());
    }
}
