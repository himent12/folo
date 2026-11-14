//! The distillation term against finite differences and its closed-form gradients, its zero at a teacher equal to
//! the student, the blended-target identity with the supervised BCE, and the flip map.

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{Device, Tensor};

use super::*;
use crate::boxes::Xyxy;
use crate::decode::{make_anchors, softmax_expectation};
use crate::loss::{BatchAssignment, LossConfig, loss_with_assignment};

const R: usize = 16;

/// Student and teacher raw outputs `(box_raw, cls_raw)` for `b` images of `nc` classes over `a` anchors.
fn raw(seed: u64, b: usize, nc: usize, a: usize) -> (Tensor, Tensor) {
    let mut rng = Rng::seed(seed);
    (Tensor::randn([b, 4 * R, a], 0.0, 2.0, &mut rng), Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng))
}

fn targets_of(box_raw: &Tensor, cls_raw: &Tensor, temperature: f32) -> DistillTargets {
    DistillTargets::new(&TeacherView::new(box_raw, cls_raw, R, temperature).unwrap()).unwrap()
}

fn cfg(cls_gain: f32, dfl_gain: f32, temperature: f32) -> DistillConfig {
    DistillConfig { cls_gain, dfl_gain, temperature, ..DistillConfig::new("t.folo") }
}

fn grads(loss: &DistillLoss, of: &[&Tensor]) -> Vec<Vec<f32>> {
    let g = loss.total.backward().unwrap();
    of.iter().map(|t| g.get(t).unwrap().to_vec::<f32>().unwrap()).collect()
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

/// The whole term against central finite differences in both inputs, at T = 1 and T = 3. Few anchors, so the f32
/// total is small enough for the differences to resolve.
#[test]
fn gradients_match_finite_differences() {
    let (sb, sc) = raw(1, 2, 3, 5);
    let (tb, tc) = raw(2, 2, 3, 5);
    for temp in [1.0, 3.0] {
        let t = targets_of(&tb, &tc, temp);
        let c = cfg(0.5, 1.5, temp);
        let f = |v: &[Tensor]| Ok(distill_loss(&v[0], &v[1], &t, 1.0, &c)?.total);
        let opts = GradCheck { eps: 1e-2, atol: 2e-4, rtol: 1e-2, max_checks: 0, ..GradCheck::default() };
        grad_check_with(f, &[sb.clone(), sc.clone()], &opts).unwrap_or_else(|e| panic!("T = {temp}: {e}"));
    }
}

/// The method's closed-form gradients: `dL/dx = B λc (σ(x) − p~) / N`, `dL/dz = B λd T q (softmax(z/T) − s~) / (4N)`.
#[test]
fn gradients_are_the_closed_form() {
    let a = make_anchors(64, 96, &[8, 16, 32], 0.5).len();
    let (b, nc, n) = (2, 3, 2.5f32);
    let (sb, sc) = raw(3, b, nc, a);
    let (tb, tc) = raw(4, b, nc, a);
    for temp in [1.0f32, 3.0] {
        let (x, z) = (sc.detach().requires_grad_(), sb.detach().requires_grad_());
        let c = cfg(0.25, 1.5, temp);
        let t = targets_of(&tb, &tc, temp);
        let g = grads(&distill_loss(&z, &x, &t, n, &c).unwrap(), &[&x, &z]);
        let p = tc.sigmoid().unwrap().to_vec::<f32>().unwrap();
        let xs = sc.sigmoid().unwrap().to_vec::<f32>().unwrap();
        let want: Vec<f32> = xs.iter().zip(&p).map(|(s, p)| b as f32 * c.cls_gain * (s - p) / n).collect();
        assert!(max_abs_diff(&g[0], &want) < 1e-6, "cls, T = {temp}");
        let soft =
            |t: &Tensor| t.reshape([b, 4, R, a]).unwrap().div_scalar(temp).unwrap().softmax(2).unwrap();
        let (ss, st) = (soft(&sb).to_vec::<f32>().unwrap(), soft(&tb).to_vec::<f32>().unwrap());
        let q = tc.sigmoid().unwrap().max(1, false).unwrap().to_vec::<f32>().unwrap();
        let want: Vec<f32> = (0..ss.len())
            .map(|i| {
                let (img, an) = (i / (4 * R * a), i % a);
                b as f32 * c.dfl_gain * temp * q[img * a + an] * (ss[i] - st[i]) / (4.0 * n)
            })
            .collect();
        assert!(max_abs_diff(&g[1], &want) < 1e-6, "box, T = {temp}");
    }
}

/// A teacher equal to the student: both reported terms are (about) 0 and so is every gradient.
#[test]
fn a_teacher_equal_to_the_student_teaches_nothing() {
    let a = make_anchors(64, 64, &[8, 16, 32], 0.5).len();
    let (sb, sc) = raw(5, 2, 4, a);
    for temp in [1.0, 2.0] {
        let (x, z) = (sc.detach().requires_grad_(), sb.detach().requires_grad_());
        let out = distill_loss(&z, &x, &targets_of(&sb, &sc, temp), 4.0, &cfg(0.5, 1.5, temp)).unwrap();
        assert!(out.cls.abs() < 1e-5 && out.dfl.abs() < 1e-5, "T = {temp}: {out:?}");
        for (g, name) in grads(&out, &[&x, &z]).iter().zip(["cls", "box"]) {
            assert!(g.iter().all(|v| v.abs() < 1e-6), "{name}, T = {temp}");
        }
    }
    // And a different teacher is not 0: the class KL and the DFL KL are positive.
    let (tb, tc) = raw(6, 2, 4, a);
    let out = distill_loss(&sb, &sc, &targets_of(&tb, &tc, 1.0), 4.0, &cfg(0.5, 1.5, 1.0)).unwrap();
    assert!(out.cls > 1e-2 && out.dfl > 1e-2, "{out:?}");
}

/// Supervised BCE at class gain g plus `L_kc` at λc has the gradient (w.r.t. `cls_raw`) of the BCE against the
/// blended target `(g T + λc p~) / (g + λc)` at gain `g + λc`, both over the supervised normaliser.
#[test]
fn class_term_is_a_blended_bce_target() {
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let (b, nc, a) = (2, 3, anchors.len());
    let (sb, sc) = raw(7, b, nc, a);
    let (tb, tc) = raw(8, b, nc, a);
    let fg = [(0u32, 5u32, 1u32, 0.7f32), (0, 30, 2, 0.4), (1, 70, 0, 0.9), (1, 80, 1, 0.2)];
    let asg = BatchAssignment {
        batch: b,
        num_anchors: a,
        num_classes: nc,
        image: fg.iter().map(|f| f.0).collect(),
        anchor: fg.iter().map(|f| f.1).collect(),
        label: fg.iter().map(|f| f.2).collect(),
        score: fg.iter().map(|f| f.3).collect(),
        target: vec![Xyxy::new(4.0, 6.0, 30.0, 40.0); fg.len()],
        inputs_finite: true,
    };
    let tss = asg.target_score_sum() as f32;
    let (g, lambda) = (0.5f32, 0.25f32);
    let x = sc.detach().requires_grad_();
    let sup =
        loss_with_assignment(&sb, &x, &anchors, &asg, &LossConfig { cls_gain: g, ..LossConfig::default() })
            .unwrap();
    let kd = distill_loss(&sb, &x, &targets_of(&tb, &tc, 1.0), tss, &cfg(lambda, 0.0, 1.0)).unwrap();
    let got = sup.total.add(&kd.total).unwrap().backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap();
    // The reference, built directly: Σ BCE(x, blend) · (g + λc) / tss · B.
    let p = tc.sigmoid().unwrap().to_vec::<f32>().unwrap();
    let mut hard = vec![0f32; b * nc * a];
    for &(i, an, c, s) in &fg {
        hard[(i as usize * nc + c as usize) * a + an as usize] = s;
    }
    let blend: Vec<f32> = hard.iter().zip(&p).map(|(t, p)| (g * t + lambda * p) / (g + lambda)).collect();
    let blend = Tensor::from_vec(blend, [b, nc, a]).unwrap();
    let y = sc.detach().requires_grad_();
    let bce = y.softplus().unwrap().sub(&y.mul(&blend).unwrap()).unwrap().sum_all().unwrap();
    let reference = bce.mul_scalar((g + lambda) / tss * b as f32).unwrap();
    let want = reference.backward().unwrap().get(&y).unwrap().to_vec::<f32>().unwrap();
    assert!(max_abs_diff(&got, &want) < 1e-6, "{}", max_abs_diff(&got, &want));
    // The identity has content: without the teacher the gradients differ.
    assert!(max_abs_diff(&got, &sup_only(&sb, &sc, &anchors, &asg, g + lambda)) > 1e-3);
}

fn sup_only(
    sb: &Tensor,
    sc: &Tensor,
    anchors: &crate::decode::Anchors,
    asg: &BatchAssignment,
    g: f32,
) -> Vec<f32> {
    let x = sc.detach().requires_grad_();
    let l = loss_with_assignment(sb, &x, anchors, asg, &LossConfig { cls_gain: g, ..LossConfig::default() })
        .unwrap();
    l.total.backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap()
}

/// The flip map on a rectangular input (64 × 96): a permutation that is its own inverse, applied twice the identity,
/// and the mapped outputs decode to the mirror images of the boxes they came from.
#[test]
fn flip_map_mirrors_boxes_exactly() {
    let (h, w) = (64u32, 96u32);
    let anchors = make_anchors(h, w, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let perm = flip_permutation(&anchors);
    assert!(perm.iter().enumerate().all(|(i, &p)| perm[p] == i), "an involution");
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..a).collect::<Vec<_>>(), "a permutation");
    assert!(perm.iter().enumerate().any(|(i, &p)| p != i));

    let map = FlipMap::new(&anchors, w as usize, Device::Cpu).unwrap();
    let (box_raw, cls_raw) = raw(9, 2, 3, a);
    let twice = map.map_box(&map.map_box(&box_raw).unwrap()).unwrap();
    assert_eq!(twice.to_vec::<f32>().unwrap(), box_raw.to_vec::<f32>().unwrap());
    let twice = map.map_cls(&map.map_cls(&cls_raw).unwrap()).unwrap();
    assert_eq!(twice.to_vec::<f32>().unwrap(), cls_raw.to_vec::<f32>().unwrap());
    let x = Tensor::rand_uniform([1, 3, h as usize, w as usize], 0.0, 1.0, &mut Rng::seed(1));
    let flipped = map.flip_input(&x).unwrap().to_vec::<f32>().unwrap();
    let xv = x.to_vec::<f32>().unwrap();
    let wu = w as usize;
    assert!((0..xv.len()).all(|i| flipped[i] == xv[i - i % wu + (wu - 1 - i % wu)]), "columns reversed");

    let decode = |t: &Tensor| -> Vec<[f32; 4]> {
        let v = t.to_vec::<f32>().unwrap();
        (0..a)
            .map(|an| {
                let d: Vec<f32> = (0..4)
                    .map(|s| {
                        softmax_expectation(&(0..R).map(|i| v[(s * R + i) * a + an]).collect::<Vec<_>>())
                    })
                    .collect();
                let ([px, py], st) = (anchors.points[an], anchors.strides[an]);
                [(px - d[0]) * st, (py - d[1]) * st, (px + d[2]) * st, (py + d[3]) * st]
            })
            .collect()
    };
    let one = box_raw.narrow(0, 0, 1).unwrap().contiguous().unwrap();
    let (orig, mapped) = (decode(&one), decode(&map.map_box(&one).unwrap()));
    let wf = w as f32;
    for an in 0..a {
        let (m, o) = (mapped[an], orig[perm[an]]);
        let mirror = [wf - o[2], o[1], wf - o[0], o[3]];
        assert!(m.iter().zip(&mirror).all(|(x, y)| (x - y).abs() < 1e-3), "anchor {an}: {m:?} vs {mirror:?}");
    }
    // A width that is not a multiple of every stride has no exact map.
    assert!(FlipMap::new(&make_anchors(64, 80, &[8, 16, 32], 0.5), 80, Device::Cpu).is_err());
}

/// The flip consensus averages in probability space: a view averaged with itself is unchanged, and the targets of
/// two disagreeing views sit between them.
#[test]
fn consensus_averages_probabilities() {
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let (tb, tc) = raw(10, 1, 2, anchors.len());
    let v = TeacherView::new(&tb, &tc, R, 1.0).unwrap();
    let same = v.average(&v).unwrap();
    assert_eq!(same.cls_prob.to_vec::<f32>().unwrap(), v.cls_prob.to_vec::<f32>().unwrap());
    let map = FlipMap::new(&anchors, 64, Device::Cpu).unwrap();
    let avg = v.average(&v.mapped(&map).unwrap()).unwrap();
    // Each side's averaged distribution still sums to 1.
    let s = avg.box_prob.sum(2, false).unwrap().to_vec::<f32>().unwrap();
    assert!(s.iter().all(|x| (x - 1.0).abs() < 1e-5));
}

#[test]
fn config_is_validated() {
    assert!(DistillConfig::new("folo-s-coco").validate().is_ok());
    let bad = [
        DistillConfig::default(),
        DistillConfig { temperature: 0.0, ..DistillConfig::new("t") },
        DistillConfig { temperature: f32::NAN, ..DistillConfig::new("t") },
        DistillConfig { cls_gain: -0.1, ..DistillConfig::new("t") },
        DistillConfig { dfl_gain: f32::INFINITY, ..DistillConfig::new("t") },
    ];
    for c in bad {
        assert!(c.validate().is_err(), "{c:?}");
    }
    let c: DistillConfig = serde_json::from_str(r#"{"teacher":"t.folo","consensus":"flip"}"#).unwrap();
    assert_eq!((c.cls_gain, c.dfl_gain, c.temperature, c.consensus), (0.5, 1.5, 1.0, Consensus::Flip));
}
