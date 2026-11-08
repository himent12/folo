//! Detection loss (`docs/MODEL.md` §7.4) against an independent `f64` reference, the loss cases of §10.3, gradient
//! checks, edge cases and determinism.

mod common;

use common::Rng;
use folo_core::Tensor;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_vision::Xyxy;
use folo_vision::assign::{TalConfig, assign};
use folo_vision::boxes::ciou_eps;
use folo_vision::decode::{Anchors, decode_boxes, make_anchors, sigmoid_all};
use folo_vision::loss::{
    BatchAssignment, ImageTargets, LossConfig, assign_batch, ciou, detection_loss, loss_with_assignment,
    targets_from_normalized,
};

const R: usize = 16;

fn tensor(v: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(v, shape).unwrap()
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Random head outputs: box logits around the §6.5 prior, class logits in `[lo, hi)`.
fn random_head(rng: &mut Rng, b: usize, nc: usize, a: usize, cls: (f32, f32)) -> (Tensor, Tensor) {
    let reg: Vec<f32> = (0..b * 4 * R * a).map(|_| 2.0 + rng.range(-2.0, 2.0)).collect();
    let logits: Vec<f32> = (0..b * nc * a).map(|_| rng.range(cls.0, cls.1)).collect();
    (tensor(reg, &[b, 4 * R, a]), tensor(logits, &[b, nc, a]))
}

fn random_targets(rng: &mut Rng, counts: &[usize], nc: usize, size: f32) -> Vec<ImageTargets> {
    counts
        .iter()
        .map(|&g| ImageTargets {
            boxes: (0..g).map(|_| rng.bbox(size, 0.1 * size, 0.6 * size)).collect(),
            labels: (0..g).map(|_| rng.below(nc) as u32).collect(),
        })
        .collect()
}

// ---- f64 reference of §7.2 / §7.4 -------------------------------------------------------------------------------

fn ciou_ref(a: [f64; 4], b: [f64; 4]) -> f64 {
    ciou_parts(a, b, None).0
}

/// `(CIoU, α)`; with `alpha` given, that value is used instead of the computed one.
fn ciou_parts(a: [f64; 4], b: [f64; 4], alpha: Option<f64>) -> (f64, f64) {
    let eps = 1e-7;
    let (w1, h1, w2, h2) = (a[2] - a[0], a[3] - a[1] + eps, b[2] - b[0], b[3] - b[1] + eps);
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let union = w1 * h1 + w2 * h2 - inter + eps;
    let iou = inter / union;
    let cw = a[2].max(b[2]) - a[0].min(b[0]);
    let ch = a[3].max(b[3]) - a[1].min(b[1]);
    let c2 = cw * cw + ch * ch + eps;
    let rho2 = ((b[0] + b[2] - a[0] - a[2]).powi(2) + (b[1] + b[3] - a[1] - a[3]).powi(2)) / 4.0;
    let v = 4.0 / std::f64::consts::PI.powi(2) * ((w2 / h2).atan() - (w1 / h1).atan()).powi(2);
    let alpha = alpha.unwrap_or(v / (1.0 - iou + v + eps));
    (iou - (rho2 / c2 + v * alpha), alpha)
}

fn log_softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    x.iter().map(|v| v - lse).collect()
}

/// `(L_box, L_cls, L_dfl)` of §7.4, unweighted, straight from the formulas with a dense target `T`.
fn reference(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
) -> (f64, f64, f64) {
    let (b, nc, a) = cls_raw.dims3().unwrap();
    let (reg, cls) = (box_raw.to_vec::<f32>().unwrap(), cls_raw.to_vec::<f32>().unwrap());
    let mut dense = vec![0f64; b * nc * a];
    for k in 0..asg.num_fg() {
        dense[(asg.image[k] as usize * nc + asg.label[k] as usize) * a + asg.anchor[k] as usize] =
            asg.score[k] as f64;
    }
    let tss = dense.iter().sum::<f64>().max(1.0);
    let bce = |x: f64, t: f64| x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln();
    let l_cls = cls.iter().zip(&dense).map(|(&x, &t)| bce(x as f64, t)).sum::<f64>() / tss;
    let (mut l_box, mut l_dfl) = (0.0, 0.0);
    for k in 0..asg.num_fg() {
        let (i, an, t) = (asg.image[k] as usize, asg.anchor[k] as usize, asg.score[k] as f64);
        let [px, py] = anchors.points[an].map(|v| v as f64);
        let s = anchors.strides[an] as f64;
        let g = asg.target[k];
        let tb = [g.x1 as f64 / s, g.y1 as f64 / s, g.x2 as f64 / s, g.y2 as f64 / s];
        let ltrb = [px - tb[0], py - tb[1], tb[2] - px, tb[3] - py];
        let mut d = [0f64; 4];
        let mut dfl = 0.0;
        for side in 0..4 {
            let logits: Vec<f64> =
                (0..R).map(|j| reg[((i * 4 * R) + side * R + j) * a + an] as f64).collect();
            let lsm = log_softmax(&logits);
            d[side] = lsm.iter().enumerate().map(|(j, l)| j as f64 * l.exp()).sum();
            let delta = ltrb[side].clamp(0.0, R as f64 - 1.0 - 0.01);
            let j = delta.floor() as usize;
            let wl = j as f64 + 1.0 - delta;
            dfl += -(wl * lsm[j] + (1.0 - wl) * lsm[j + 1]);
        }
        let pd = [px - d[0], py - d[1], px + d[2], py + d[3]];
        l_box += t * (1.0 - ciou_ref(pd, tb));
        l_dfl += t * dfl / 4.0;
    }
    (l_box / tss, l_cls, l_dfl / tss)
}

fn check_against_reference(box_raw: &Tensor, cls_raw: &Tensor, anchors: &Anchors, targets: &[ImageTargets]) {
    let cfg = LossConfig::default();
    let asg = assign_batch(box_raw, cls_raw, anchors, targets, &cfg).unwrap();
    let out = loss_with_assignment(box_raw, cls_raw, anchors, &asg, &cfg).unwrap();
    let (lb, lc, ld) = reference(box_raw, cls_raw, anchors, &asg);
    let it = out.items;
    assert!(rel_close(it.box_ as f64, 7.5 * lb, 2e-5), "box {} vs {}", it.box_, 7.5 * lb);
    assert!(rel_close(it.cls as f64, 0.5 * lc, 2e-5), "cls {} vs {}", it.cls, 0.5 * lc);
    assert!(rel_close(it.dfl as f64, 1.5 * ld, 2e-5), "dfl {} vs {}", it.dfl, 1.5 * ld);
    let b = box_raw.dims()[0] as f64;
    let want = b * (7.5 * lb + 0.5 * lc + 1.5 * ld);
    assert!(rel_close(out.total.item::<f32>().unwrap() as f64, want, 2e-5));
    assert_eq!(out.num_fg, asg.num_fg());
    assert!((out.target_score_sum as f64 - asg.target_score_sum()).abs() < 1e-4);
}

// ---- CIoU ------------------------------------------------------------------------------------------------------

#[test]
fn tensor_ciou_matches_ciou_eps() {
    let mut rng = Rng::new(7);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..400 {
        let p = rng.bbox(100.0, 0.5, 60.0);
        // Overlapping, identical, contained and disjoint pairs.
        let q = match i % 4 {
            0 => rng.jitter(&p, 0.3),
            1 => p,
            2 => Xyxy::new(p.x1 + 0.1 * p.width(), p.y1 + 0.2 * p.height(), p.x2 - 0.1 * p.width(), p.y2),
            _ => rng.bbox(100.0, 0.5, 20.0).translate(150.0, 0.0),
        };
        a.push(p);
        b.push(q);
    }
    let flat = |v: &[Xyxy]| tensor(v.iter().flat_map(|b| [b.x1, b.y1, b.x2, b.y2]).collect(), &[v.len(), 4]);
    let got = ciou(&flat(&a), &flat(&b)).unwrap().to_vec::<f32>().unwrap();
    for (i, g) in got.iter().enumerate() {
        let want = ciou_eps(&a[i], &b[i]);
        assert!((g - want).abs() <= 2e-6 * (1.0 + want.abs()), "pair {i}: {g} vs {want}");
        match i % 4 {
            // §10.3: identical boxes give ≈ 1 (exactly 1 minus the eps effects), disjoint ones a negative CIoU.
            1 => assert!(*g > 1.0 - 1e-5 && *g <= 1.0, "identical: {g}"),
            3 => assert!(*g < 0.0, "disjoint: {g}"),
            _ => {}
        }
    }
}

/// §7.2 treats α as a constant, so the autograd gradient of [`ciou`] is the derivative with α frozen at its value.
/// Compare it with `f64` central differences of the reference with α frozen.
#[test]
fn tensor_ciou_gradients_with_alpha_constant() {
    let mut rng = Rng::new(8);
    let n = 64;
    let (mut p, mut q) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let a = rng.bbox(50.0, 2.0, 30.0);
        p.push(a);
        q.push(rng.jitter(&a, 0.6));
    }
    let flat = |v: &[Xyxy]| tensor(v.iter().flat_map(|b| [b.x1, b.y1, b.x2, b.y2]).collect(), &[n, 4]);
    let (pt, qt) = (flat(&p).requires_grad_(), flat(&q).requires_grad_());
    let grads = ciou(&pt, &qt).unwrap().sum_all().unwrap().backward().unwrap();
    let (gp, gq) =
        (grads.get(&pt).unwrap().to_vec::<f32>().unwrap(), grads.get(&qt).unwrap().to_vec::<f32>().unwrap());
    let arr = |b: &Xyxy| [b.x1 as f64, b.y1 as f64, b.x2 as f64, b.y2 as f64];
    let mut checked = 0;
    for i in 0..n {
        let (a, b) = (arr(&p[i]), arr(&q[i]));
        let alpha = ciou_parts(a, b, None).1;
        // Skip pairs within the finite-difference step of a min/max/clamp kink.
        let h = 1e-6;
        let kink = |x: f64, y: f64| (x - y).abs() < 1e-3;
        if (0..4).any(|k| kink(a[k], b[k]))
            || kink(a[2], b[0])
            || kink(a[0], b[2])
            || kink(a[3], b[1])
            || kink(a[1], b[3])
        {
            continue;
        }
        for k in 0..8 {
            let probe = |d: f64| {
                let (mut a, mut b) = (a, b);
                if k < 4 {
                    a[k] += d
                } else {
                    b[k - 4] += d
                }
                ciou_parts(a, b, Some(alpha)).0
            };
            let fd = (probe(h) - probe(-h)) / (2.0 * h);
            let got = if k < 4 { gp[i * 4 + k] } else { gq[i * 4 + k - 4] } as f64;
            assert!((got - fd).abs() <= 1e-4 + 1e-3 * fd.abs(), "pair {i} coord {k}: {got} vs {fd}");
        }
        checked += 1;
    }
    assert!(checked > n / 2, "{checked}");
}

// ---- the loss against the reference ------------------------------------------------------------------------------

#[test]
fn matches_reference_on_a_batch_with_differing_gt_counts() {
    let mut rng = Rng::new(11);
    let anchors = make_anchors(128, 128, &[8, 16, 32], 0.5);
    let counts = [0, 1, 5, 3];
    let (box_raw, cls_raw) = random_head(&mut rng, counts.len(), 4, anchors.len(), (-4.0, 2.0));
    let targets = random_targets(&mut rng, &counts, 4, 128.0);
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    assert!(asg.num_fg() > 0);
    // Image 0 has no gt: no foreground; every image with gts has some.
    for (i, &g) in counts.iter().enumerate() {
        let fg = asg.image.iter().filter(|&&im| im as usize == i).count();
        assert_eq!(fg == 0, g == 0, "image {i}: {fg} fg for {g} gts");
    }
    check_against_reference(&box_raw, &cls_raw, &anchors, &targets);
}

#[test]
fn matches_reference_at_640_with_80_classes() {
    let mut rng = Rng::new(12);
    let anchors = make_anchors(640, 640, &[8, 16, 32], 0.5);
    let (box_raw, cls_raw) = random_head(&mut rng, 2, 80, anchors.len(), (-10.0, -2.0));
    let targets = random_targets(&mut rng, &[7, 2], 80, 640.0);
    check_against_reference(&box_raw, &cls_raw, &anchors, &targets);
}

/// `assign_batch` (channel-major host decode, only the classes present) gives what `decode_boxes` + `assign` on the
/// full anchor-major score matrix give.
#[test]
fn assign_batch_matches_per_image_assign() {
    let mut rng = Rng::new(13);
    let nc = 6;
    let anchors = make_anchors(96, 96, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let (box_raw, cls_raw) = random_head(&mut rng, 3, nc, a, (-3.0, 3.0));
    let targets = random_targets(&mut rng, &[4, 2, 6], nc, 96.0);
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    let (reg, cls) = (box_raw.to_vec::<f32>().unwrap(), cls_raw.to_vec::<f32>().unwrap());
    let pts = anchors.points_px();
    let mut k = 0;
    for (i, gt) in targets.iter().enumerate() {
        let (reg, cls) = (&reg, &cls);
        let reg_am: Vec<f32> =
            (0..a).flat_map(|an| (0..4 * R).map(move |c| reg[(i * 4 * R + c) * a + an])).collect();
        let cls_am: Vec<f32> =
            (0..a).flat_map(|an| (0..nc).map(move |c| cls[(i * nc + c) * a + an])).collect();
        let boxes = decode_boxes(&reg_am, &anchors, R);
        let want =
            assign(&sigmoid_all(&cls_am), nc, &boxes, &pts, &gt.boxes, &gt.labels, &TalConfig::default());
        for an in (0..a).filter(|&an| want.fg_mask[an]) {
            assert_eq!((asg.image[k], asg.anchor[k]), (i as u32, an as u32));
            assert_eq!(asg.label[k], want.target_labels[an]);
            assert_eq!(asg.target[k], want.target_bboxes[an]);
            assert!((asg.score[k] - want.target_scores[an]).abs() < 1e-5);
            k += 1;
        }
    }
    assert_eq!(k, asg.num_fg());
}

// ---- §10.3 loss cases --------------------------------------------------------------------------------------------

/// §10.3 "an empty batch": `L_box = L_dfl = 0`, `L_cls = Σ softplus(cls_raw)` with `tss = 1`, and the gradient
/// `B · 0.5 · sigmoid(x)` is positive everywhere, so descent pushes every score towards 0.
#[test]
fn no_gts_gives_only_the_cls_loss() {
    let mut rng = Rng::new(21);
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let (box_raw, cls_raw) = random_head(&mut rng, 3, 5, anchors.len(), (-6.0, 6.0));
    let (box_raw, cls_raw) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
    let targets = vec![ImageTargets::default(); 3];
    let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    assert_eq!(out.num_fg, 0);
    assert_eq!((out.items.box_, out.items.dfl, out.target_score_sum), (0.0, 0.0, 0.0));
    let x = cls_raw.to_vec::<f32>().unwrap();
    let softplus: f64 = x.iter().map(|&v| (v as f64).max(0.0) + (1.0 + (-(v as f64).abs()).exp()).ln()).sum();
    assert!(rel_close(out.items.cls as f64, 0.5 * softplus, 1e-5), "{} vs {}", out.items.cls, 0.5 * softplus);
    assert!(rel_close(out.total.item::<f32>().unwrap() as f64, 3.0 * 0.5 * softplus, 1e-5));

    let grads = out.total.backward().unwrap();
    let g = grads.get(&cls_raw).unwrap().to_vec::<f32>().unwrap();
    for (gv, &xv) in g.iter().zip(&x) {
        let want = 1.5 / (1.0 + (-xv as f64).exp());
        assert!(*gv > 0.0 && (*gv as f64 - want).abs() < 1e-6 * (1.0 + want), "{gv} vs {want}");
    }
    // No foreground: nothing reaches the box branch.
    assert!(grads.get(&box_raw).is_none_or(|g| g.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0)));
}

/// One level (stride 8) so every distance can be an exact bin: a gt whose corners sit on anchor coordinates and box
/// logits peaked at the exact distances give a CIoU of ≈ 1 and a DFL of ≈ 0 (one-hot targets, `w_r = 0`).
fn perfect_case(peak: f32) -> (Anchors, Tensor, Tensor, Vec<ImageTargets>) {
    let anchors = make_anchors(128, 128, &[8], 0.5);
    let a = anchors.len();
    let gts = [Xyxy::new(1.5, 1.5, 6.5, 7.5), Xyxy::new(8.5, 3.5, 14.5, 13.5)]; // grid units
    let mut reg = vec![0f32; 4 * R * a];
    for (an, p) in anchors.points.iter().enumerate() {
        let Some(g) = gts.iter().find(|g| g.contains_point(p[0], p[1], 0.0)) else { continue };
        let d = [p[0] - g.x1, p[1] - g.y1, g.x2 - p[0], g.y2 - p[1]];
        for (side, &d) in d.iter().enumerate() {
            for j in 0..R {
                reg[(side * R + j) * a + an] = if j as f32 == d { peak } else { -peak };
            }
        }
    }
    let cls = vec![0f32; 2 * a];
    let targets =
        vec![ImageTargets { boxes: gts.iter().map(|g| g.scale(8.0, 8.0)).collect(), labels: vec![0, 1] }];
    (anchors, tensor(reg, &[1, 4 * R, a]), tensor(cls, &[1, 2, a]), targets)
}

#[test]
fn perfect_prediction_has_near_zero_box_and_dfl_loss() {
    let (anchors, box_raw, cls_raw, targets) = perfect_case(30.0);
    let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
    assert!(out.num_fg >= 10, "{}", out.num_fg);
    assert!(out.items.box_ >= 0.0 && out.items.box_ < 1e-5, "box {}", out.items.box_);
    assert!(out.items.dfl >= 0.0 && out.items.dfl < 1e-5, "dfl {}", out.items.dfl);
    // Every positive has u ≈ 1, so its target score is ≈ 1.
    assert!(out.target_score_sum > out.num_fg as f32 - 1e-3);
}

/// §10.3 loss scaling: duplicating a batch (B → 2B) keeps the reported items and doubles `L` when `Σ T >= 1`; below
/// the floor (all background) the reported `cls` doubles and `L` quadruples.
#[test]
fn loss_scaling_with_batch_duplication() {
    let cfg = LossConfig::default();
    let dup = |t: &Tensor| Tensor::cat(&[t, t], 0).unwrap();
    let (anchors, box_raw, cls_raw, targets) = perfect_case(2.0);
    let one = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    assert!(one.target_score_sum >= 1.0);
    let targets2 = [targets.clone(), targets].concat();
    let two = detection_loss(&dup(&box_raw), &dup(&cls_raw), &anchors, &targets2, &cfg).unwrap();
    for (x, y) in
        [(one.items.box_, two.items.box_), (one.items.cls, two.items.cls), (one.items.dfl, two.items.dfl)]
    {
        assert!(rel_close(x as f64, y as f64, 1e-5), "{x} vs {y}");
    }
    let (l1, l2) = (one.total.item::<f32>().unwrap(), two.total.item::<f32>().unwrap());
    assert!(rel_close(2.0 * l1 as f64, l2 as f64, 1e-5), "{l1} {l2}");

    let empty = vec![ImageTargets::default()];
    let one = detection_loss(&box_raw, &cls_raw.sub_scalar(4.0).unwrap(), &anchors, &empty, &cfg).unwrap();
    let two = detection_loss(
        &dup(&box_raw),
        &dup(&cls_raw.sub_scalar(4.0).unwrap()),
        &anchors,
        &[empty.clone(), empty].concat(),
        &cfg,
    )
    .unwrap();
    assert!(rel_close(2.0 * one.items.cls as f64, two.items.cls as f64, 1e-5));
    let (l1, l2) = (one.total.item::<f32>().unwrap(), two.total.item::<f32>().unwrap());
    assert!(rel_close(4.0 * l1 as f64, l2 as f64, 1e-5), "{l1} {l2}");
}

/// §10.3 DFL target `δ = 3.3`: weights 0.7 on bin 3 and 0.3 on bin 4 (and `δ = 15.2` clamps to 14.99), through the
/// loss with a hand-built assignment.
#[test]
fn dfl_term_by_hand() {
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let a = anchors.len();
    // Anchor 5 is at grid (1.5, 1.5). Target ltrb (3.3, 1.0, 15.2, 2.0) grid units.
    let p = anchors.points[5];
    let tb = Xyxy::new(p[0] - 3.3, p[1] - 1.0, p[0] + 15.2, p[1] + 2.0);
    let asg = BatchAssignment {
        batch: 1,
        num_anchors: a,
        num_classes: 1,
        image: vec![0],
        anchor: vec![5],
        label: vec![0],
        score: vec![0.5],
        target: vec![tb.scale(8.0, 8.0)],
        inputs_finite: true,
    };
    // Logits l_j = j / 4 on every side: log_softmax_j = j/4 − lse.
    let mut reg = vec![0f32; 4 * R * a];
    for c in 0..4 * R {
        reg[c * a + 5] = (c % R) as f32 / 4.0;
    }
    let box_raw = tensor(reg, &[1, 4 * R, a]);
    let cls_raw = tensor(vec![-5.0; a], &[1, 1, a]);
    let out = loss_with_assignment(&box_raw, &cls_raw, &anchors, &asg, &LossConfig::default()).unwrap();
    let lse = (0..R).map(|j| (j as f64 / 4.0).exp()).sum::<f64>().ln();
    let ls = |j: usize| j as f64 / 4.0 - lse;
    let ce = |wl: f64, j: usize| -(wl * ls(j) + (1.0 - wl) * ls(j + 1));
    // l: 3.3 → bins 3/4 at 0.7/0.3; t: 1.0 → bin 1 (w_l = 1); r: 15.2 → 14.99 → bins 14/15 at 0.01/0.99; b: 2.0.
    let dfl = (ce(0.7, 3) + ce(1.0, 1) + ce(0.01, 14) + ce(1.0, 2)) / 4.0;
    // tss = max(0.5, 1) = 1.
    let want = 1.5 * 0.5 * dfl;
    assert!(rel_close(out.items.dfl as f64, want, 1e-5), "{} vs {want}", out.items.dfl);
}

// ---- gradients ---------------------------------------------------------------------------------------------------

/// Finite-difference check of the whole loss (all three terms) on a tiny case, A = 20 anchors, C = 3 classes,
/// B = 2, every element of both inputs. The assignment is held fixed: it is a constant of the loss (§7.4). CIoU's α
/// is also a constant for autograd but not for finite differences; its effect is second order in the aspect-ratio
/// gap and well inside the tolerance here, and `tensor_ciou_gradients_with_alpha_constant` checks it exactly.
#[test]
fn grad_check_full_loss() {
    let mut rng = Rng::new(31);
    let anchors = make_anchors(32, 32, &[8, 16], 0.5);
    assert_eq!(anchors.len(), 20);
    let (box_raw, cls_raw) = random_head(&mut rng, 2, 3, 20, (-2.0, 2.0));
    let targets = vec![
        ImageTargets {
            boxes: vec![Xyxy::new(3.0, 4.0, 22.0, 27.0), Xyxy::new(14.0, 10.0, 30.0, 18.0)],
            labels: vec![0, 2],
        },
        ImageTargets { boxes: vec![Xyxy::new(6.0, 2.0, 26.0, 20.0)], labels: vec![1] },
    ];
    let cfg = LossConfig::default();
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    assert!(asg.num_fg() >= 4 && asg.target_score_sum() > 0.0);
    let opts = GradCheck { eps: 2e-3, atol: 5e-3, rtol: 1e-2, max_checks: 0, ..Default::default() };
    grad_check_with(
        |v| Ok(loss_with_assignment(&v[0], &v[1], &anchors, &asg, &cfg)?.total),
        &[box_raw.clone(), cls_raw.clone()],
        &opts,
    )
    .unwrap();

    // detection_loss differentiates exactly that function.
    let (bx, cx) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
    let g1 = detection_loss(&bx, &cx, &anchors, &targets, &cfg).unwrap().total.backward().unwrap();
    let g2 = loss_with_assignment(&bx, &cx, &anchors, &asg, &cfg).unwrap().total.backward().unwrap();
    for t in [&bx, &cx] {
        assert_eq!(g1.get(t).unwrap().to_vec::<f32>().unwrap(), g2.get(t).unwrap().to_vec::<f32>().unwrap());
    }
    // Only the box logits of foreground anchors get a gradient, and all of those with t > 0 do (a zero-metric
    // positive has t = 0 and contributes nothing, §7.3).
    let gb = g1.get(&bx).unwrap().to_vec::<f32>().unwrap();
    for (e, g) in gb.iter().enumerate() {
        let (i, an) = (e / (4 * R * 20), e % 20);
        let t = (0..asg.num_fg())
            .find(|&k| asg.image[k] as usize == i && asg.anchor[k] as usize == an)
            .map(|k| asg.score[k]);
        assert_eq!(*g != 0.0, t.is_some_and(|t| t > 0.0), "element {e}: grad {g}, t {t:?}");
    }
}

// ---- guards, inputs, determinism ---------------------------------------------------------------------------------

#[test]
fn non_finite_head_output_gives_a_non_finite_loss() {
    let mut rng = Rng::new(41);
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let targets = random_targets(&mut rng, &[2, 3], 3, 64.0);
    for (which, bad) in [(0, f32::NAN), (1, f32::INFINITY), (1, f32::NAN), (0, f32::NEG_INFINITY)] {
        let (box_raw, cls_raw) = random_head(&mut rng, 2, 3, anchors.len(), (-3.0, 3.0));
        let poison = |t: &Tensor| {
            let mut v = t.to_vec::<f32>().unwrap();
            let at = v.len() - 7;
            v[at] = bad;
            tensor(v, t.dims())
        };
        let (box_raw, cls_raw) =
            if which == 0 { (poison(&box_raw), cls_raw) } else { (box_raw, poison(&cls_raw)) };
        let cfg = LossConfig::default();
        let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
        assert!(!out.is_finite() && !out.total.item::<f32>().unwrap().is_finite(), "{which} {bad}");
        assert!(!assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap().inputs_finite);
    }
}

#[test]
fn targets_from_normalized_converts_and_validates() {
    let rows =
        [[1.0, 2.0, 0.5, 0.25, 0.5, 0.5], [0.0, 0.0, 0.1, 0.1, 0.2, 0.1], [1.0, 1.0, 0.5, 0.5, 1.0, 1.0]];
    // 320 high, 640 wide: x scales by 640, y by 320.
    let t = targets_from_normalized(&rows, 3, 320, 640).unwrap();
    assert_eq!(t[0], ImageTargets { boxes: vec![Xyxy::new(0.0, 16.0, 128.0, 48.0)], labels: vec![0] });
    assert_eq!(t[1].boxes, vec![Xyxy::new(160.0, 0.0, 480.0, 160.0), Xyxy::new(0.0, 0.0, 640.0, 320.0)]);
    assert_eq!(t[1].labels, vec![2, 1]);
    assert!(t[2].is_empty());
    for bad in [
        [3.0, 0.0, 0.5, 0.5, 0.1, 0.1],
        [0.5, 0.0, 0.5, 0.5, 0.1, 0.1],
        [0.0, -1.0, 0.5, 0.5, 0.1, 0.1],
        [0.0, 0.0, 0.5, 0.5, 0.0, 0.1],
        [0.0, 0.0, f32::NAN, 0.5, 0.1, 0.1],
    ] {
        assert!(targets_from_normalized(&[bad], 3, 320, 640).is_err(), "{bad:?}");
    }
}

#[test]
fn rejects_mismatched_inputs() {
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let cfg = LossConfig::default();
    let (box_raw, cls_raw) = (Tensor::zeros([2, 64, a]), Tensor::zeros([2, 3, a]));
    let gt = |l: u32| ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, 9.0, 9.0)], labels: vec![l] };
    assert!(detection_loss(&box_raw, &cls_raw, &anchors, &[gt(0), gt(2)], &cfg).is_ok());
    assert!(detection_loss(&box_raw, &cls_raw, &anchors, &[gt(0), gt(3)], &cfg).is_err()); // class >= nc
    assert!(detection_loss(&box_raw, &cls_raw, &anchors, &[gt(0)], &cfg).is_err()); // B mismatch
    assert!(detection_loss(&Tensor::zeros([2, 60, a]), &cls_raw, &anchors, &[gt(0), gt(0)], &cfg).is_err());
    assert!(
        detection_loss(&box_raw, &Tensor::zeros([2, 3, a - 1]), &anchors, &[gt(0), gt(0)], &cfg).is_err()
    );
    let asg = assign_batch(&box_raw, &cls_raw, &anchors, &[gt(0), gt(2)], &cfg).unwrap();
    assert!(loss_with_assignment(&box_raw, &Tensor::zeros([2, 4, a]), &anchors, &asg, &cfg).is_err());
}

/// Two runs on the same inputs give bitwise-identical items, totals and gradients (parallel paths included).
#[test]
fn deterministic() {
    let anchors = make_anchors(640, 640, &[8, 16, 32], 0.5);
    let run = || {
        let mut rng = Rng::new(51);
        let (box_raw, cls_raw) = random_head(&mut rng, 4, 80, anchors.len(), (-12.0, 0.0));
        let (box_raw, cls_raw) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let targets = random_targets(&mut rng, &[9, 0, 3, 20], 80, 640.0);
        let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &LossConfig::default()).unwrap();
        let g = out.total.backward().unwrap();
        let bits = |t: &Tensor| {
            g.get(t).unwrap().to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        (out.items, out.total.item::<f32>().unwrap().to_bits(), bits(&box_raw), bits(&cls_raw))
    };
    let first = run();
    for _ in 0..3 {
        assert!(run() == first);
    }
}
