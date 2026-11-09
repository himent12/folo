//! The NMS-free one-to-one head (docs/MODEL.md §6.6): the one-to-one assignment on hand-checked cases, the
//! dual-assignment loss and its gradients, the detached branch, the reserved tensor names, `.folo` round trips,
//! backwards compatibility with `o2m` files, and the top-k post-process that replaces NMS.

use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::{FloatDType, ModelFile};
use folo_vision::Xyxy;
use folo_vision::assign::{Assignment, TalConfig, assign};
use folo_vision::boxes::ciou_eps;
use folo_vision::decode::make_anchors;
use folo_vision::loss::{
    ImageTargets, LossConfig, assign_batch, detection_loss, dual_loss, loss_with_assignment, o2m_weight,
};
use folo_vision::model::{FoloModel, HeadMode, HeadOutput, ModelConfig};
use folo_vision::nms::{NmsConfig, non_max_suppression, topk_detections};
use folo_vision::{PredictConfig, Predictor};

fn config(scale: &str, nc: usize, head: &str) -> ModelConfig {
    let mut c = ModelConfig::new(scale, nc).unwrap();
    c.head = head.into();
    c
}

fn positives(a: &Assignment) -> Vec<usize> {
    a.fg_mask.iter().enumerate().filter(|(_, f)| **f).map(|(i, _)| i).collect()
}

fn temp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("folo-o2o-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// Moves every output bias away from the prior and perturbs every weight, so outputs depend on the tensors that
/// loaded (a freshly initialised head would agree with any other one through its priors).
fn perturb(m: &FoloModel, seed: u64) {
    let mut rng = Rng::seed(seed);
    let _g = no_grad();
    for t in m.named_tensors() {
        if t.name.ends_with("running_var") {
            continue;
        }
        let noise = Tensor::rand_uniform(t.tensor.shape().clone(), -0.05, 0.05, &mut rng);
        t.tensor.copy_from(&t.tensor.add(&noise).unwrap()).unwrap();
    }
}

fn same_values(a: &Tensor, b: &Tensor) -> bool {
    a.dims() == b.dims() && a.to_vec::<f32>().unwrap() == b.to_vec::<f32>().unwrap()
}

// ---- one-to-one assignment (TalConfig::o2o: topk 7, then topk2 1) ----------------------------------------------

/// One gt, four candidates of equal score: the one-to-many assignment keeps all four, the one-to-one one keeps
/// only the best-aligned anchor, whose soft target is about its IoU (m · u / (m + eps) with a single positive).
#[test]
fn one_positive_per_gt_its_best_metric() {
    let gt = Xyxy::new(0.0, 0.0, 40.0, 40.0);
    let anchors = [[10.0, 10.0], [30.0, 10.0], [10.0, 30.0], [30.0, 30.0]];
    let shift = |d: f32| Xyxy::new(d, d, 40.0 + d, 40.0 + d);
    let preds = [shift(6.0), gt, shift(3.0), shift(10.0)];
    let us: Vec<f32> = preds.iter().map(|p| ciou_eps(&gt, p).max(0.0)).collect();
    assert!(us[1] > us[2] && us[2] > us[0] && us[0] > us[3], "{us:?}");
    let scores = [0.5f32; 4];

    let o2m = assign(&scores, 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(positives(&o2m), vec![0, 1, 2, 3]);
    let o2o = assign(&scores, 1, &preds, &anchors, &[gt], &[0], &TalConfig::o2o());
    assert_eq!(positives(&o2o), vec![1]);
    assert_eq!((o2o.num_fg, o2o.target_gt[1], o2o.target_labels[1], o2o.target_bboxes[1]), (1, 0, 0, gt));
    let m = 0.5f32.sqrt() * us[1].powi(6);
    let t = m * us[1] / (m + 1e-9);
    assert!((o2o.target_scores[1] - t).abs() < 1e-6 && (t - us[1]).abs() < 1e-6, "{}", o2o.target_scores[1]);
    // topk2 = 2 keeps the two best, and their targets are normalised over the two.
    let two =
        assign(&scores, 1, &preds, &anchors, &[gt], &[0], &TalConfig { topk2: Some(2), ..TalConfig::o2o() });
    assert_eq!(positives(&two), vec![1, 2]);
    assert!((two.target_scores[1] - us[1]).abs() < 1e-5 && two.target_scores[2] < two.target_scores[1]);
    // A second filter at least as wide as the first changes nothing.
    let wide = TalConfig { topk2: Some(7), ..TalConfig::o2o() };
    let plain = TalConfig { topk2: None, ..TalConfig::o2o() };
    assert_eq!(
        assign(&scores, 1, &preds, &anchors, &[gt], &[0], &wide),
        assign(&scores, 1, &preds, &anchors, &[gt], &[0], &plain)
    );
}

/// The second filter runs after conflict resolution: gt 0's best-metric anchor X goes to gt 1 (larger IoU), so
/// gt 0 keeps its next best positive Y instead of ending with none; gt 1 keeps X over Z.
#[test]
fn second_filter_runs_after_conflicts() {
    let g0 = Xyxy::new(0.0, 0.0, 40.0, 40.0);
    let g1 = Xyxy::new(20.0, 0.0, 60.0, 40.0);
    // X is inside both gts, Y inside g0 only, Z inside g1 only.
    let anchors = [[30.0, 20.0], [10.0, 20.0], [50.0, 20.0]];
    let preds = [g1, g0, Xyxy::new(24.0, 4.0, 56.0, 36.0)];
    // Class 0 is certain at X and nearly impossible at Y, so for gt 0 the metric ranks X (poor overlap) above Y.
    let scores = [1.0, 1.0, 1e-12, 1.0, 0.5, 1.0];
    let (u0x, u0y, u1x) = (ciou_eps(&g0, &preds[0]), ciou_eps(&g0, &preds[1]), ciou_eps(&g1, &preds[0]));
    assert!(u0x.powi(6) > 1e-6 * u0y.powi(6) && u1x > u0x, "{u0x} {u0y} {u1x}");

    let a = assign(&scores, 2, &preds, &anchors, &[g0, g1], &[0, 1], &TalConfig::o2o());
    assert_eq!(positives(&a), vec![0, 1]);
    assert_eq!((a.target_gt[0], a.target_labels[0]), (1, 1));
    assert_eq!((a.target_gt[1], a.target_labels[1]), (0, 0));
    // The one-to-many assignment also gives Z to gt 1.
    let m = assign(&scores, 2, &preds, &anchors, &[g0, g1], &[0, 1], &TalConfig::default());
    assert_eq!(positives(&m), vec![0, 1, 2]);
}

/// Equal metrics: the positive is the candidate with the smaller anchor index, wherever it lies.
#[test]
fn metric_ties_go_to_the_smaller_anchor() {
    let gt = Xyxy::new(0.0, 0.0, 40.0, 40.0);
    let anchors = [[30.0, 30.0], [10.0, 10.0], [20.0, 20.0]];
    let a = assign(&[0.7; 3], 1, &[gt; 3], &anchors, &[gt], &[0], &TalConfig::o2o());
    assert_eq!(positives(&a), vec![0]);
}

// ---- dual-assignment loss -------------------------------------------------------------------------------------

#[test]
fn o2m_weight_follows_the_schedule() {
    let close = |a: f32, b: f64| (a as f64 - b).abs() < 1e-6;
    assert!(close(o2m_weight(0, 30), 0.8));
    assert!(close(o2m_weight(15, 30), (1.0 - 15.0 / 29.0) * 0.7 + 0.1));
    assert!(close(o2m_weight(29, 30), 0.1));
    assert!(close(o2m_weight(40, 30), 0.1));
    // One-epoch and two-epoch runs: E − 1 is floored at 1.
    assert!(close(o2m_weight(0, 1), 0.8) && close(o2m_weight(1, 2), 0.1));
}

fn random_pair(rng: &mut Rng, b: usize, nc: usize, a: usize) -> (Tensor, Tensor) {
    let reg = Tensor::rand_uniform([b, 64, a], 0.0, 4.0, rng);
    let cls = Tensor::rand_uniform([b, nc, a], -3.0, 2.0, rng);
    (reg, cls)
}

/// `dual_loss` is `w_m · L_o2m + (1 − w_m) · L_o2o` in value, items and gradients, and a finite-difference check
/// of the dual total over all four head outputs (fixed assignments, as in §7.4) passes.
#[test]
fn dual_loss_value_items_and_gradients() {
    let anchors = make_anchors(32, 32, &[8, 16], 0.5);
    let a = anchors.len();
    let mut rng = Rng::seed(11);
    let (bm, cm) = random_pair(&mut rng, 2, 3, a);
    let (bo, co) = random_pair(&mut rng, 2, 3, a);
    let targets = vec![
        ImageTargets {
            boxes: vec![Xyxy::new(3.0, 4.0, 22.0, 27.0), Xyxy::new(14.0, 10.0, 30.0, 18.0)],
            labels: vec![0, 2],
        },
        ImageTargets { boxes: vec![Xyxy::new(6.0, 2.0, 26.0, 20.0)], labels: vec![1] },
    ];
    let (cfg_m, cfg_o) = (LossConfig::default(), LossConfig::o2o());
    let am = assign_batch(&bm, &cm, &anchors, &targets, &cfg_m).unwrap();
    let ao = assign_batch(&bo, &co, &anchors, &targets, &cfg_o).unwrap();
    // The one-to-one assignment: at most one positive per gt (3 gts).
    assert!(
        ao.num_fg() <= 3 && ao.num_fg() >= 2 && am.num_fg() > ao.num_fg(),
        "{} {}",
        am.num_fg(),
        ao.num_fg()
    );
    let w = 0.65f32;

    let leaves: Vec<Tensor> = [&bm, &cm, &bo, &co].iter().map(|t| (*t).clone().requires_grad_()).collect();
    let lm = detection_loss(&leaves[0], &leaves[1], &anchors, &targets, &cfg_m).unwrap();
    let lo = detection_loss(&leaves[2], &leaves[3], &anchors, &targets, &cfg_o).unwrap();
    let d = dual_loss(&lm, &lo, w).unwrap();
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs()));
    assert!(close(d.total.item::<f32>().unwrap(), w * lm.total_value + (1.0 - w) * lo.total_value));
    assert!(close(d.total_value, w * lm.total_value + (1.0 - w) * lo.total_value));
    assert!(close(d.items.box_, w * lm.items.box_ + (1.0 - w) * lo.items.box_));
    assert!(close(d.items.cls, w * lm.items.cls + (1.0 - w) * lo.items.cls));
    assert!(close(d.items.dfl, w * lm.items.dfl + (1.0 - w) * lo.items.dfl));
    assert_eq!(d.num_fg, lm.num_fg);
    let g = d.total.backward().unwrap();
    let gm = lm.total.backward().unwrap();
    let go = lo.total.backward().unwrap();
    for (i, (single, scale)) in [(&gm, w), (&gm, w), (&go, 1.0 - w), (&go, 1.0 - w)].into_iter().enumerate() {
        let (got, want) =
            (g.get(&leaves[i]).unwrap().to_vec::<f32>().unwrap(), single.get(&leaves[i]).unwrap());
        for (x, y) in got.iter().zip(want.to_vec::<f32>().unwrap()) {
            assert!((x - scale * y).abs() <= 1e-6 + 1e-5 * y.abs(), "input {i}: {x} vs {scale} · {y}");
        }
    }

    let opts = GradCheck { eps: 2e-3, atol: 5e-3, rtol: 1e-2, max_checks: 0, ..Default::default() };
    grad_check_with(
        |v| {
            let m = loss_with_assignment(&v[0], &v[1], &anchors, &am, &cfg_m)?;
            let o = loss_with_assignment(&v[2], &v[3], &anchors, &ao, &cfg_o)?;
            Ok(dual_loss(&m, &o, w)?.total)
        },
        &[bm, cm, bo, co],
        &opts,
    )
    .unwrap();
}

fn grad_nonzero(g: &folo_core::Grads, t: &Tensor) -> bool {
    g.get(t).is_some_and(|v| v.to_vec::<f32>().unwrap().iter().any(|x| *x != 0.0))
}

/// In training mode an `o2o` model returns both branches; the one-to-one branch reads detached features, so its
/// loss trains only `head.o2o.*`, while the one-to-many loss reaches the backbone and never the o2o branch.
#[test]
fn o2o_branch_is_detached_from_backbone_and_neck() {
    let m = FoloModel::new(config("n", 2, "o2o")).unwrap();
    m.train();
    let x = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(4));
    let out = m.forward(&x).unwrap();
    assert_eq!(out.mode, HeadMode::O2m);
    let (bo, co) = out.o2o.clone().expect("training output has the o2o branch");
    assert_eq!((bo.dims(), co.dims()), (out.box_raw.dims(), out.cls_raw.dims()));
    let targets = vec![
        ImageTargets { boxes: vec![Xyxy::new(8.0, 8.0, 40.0, 44.0)], labels: vec![1] },
        ImageTargets { boxes: vec![Xyxy::new(20.0, 4.0, 60.0, 30.0)], labels: vec![0] },
    ];
    let lo = detection_loss(&bo, &co, &out.anchors, &targets, &LossConfig::o2o()).unwrap();
    let g = lo.total.backward().unwrap();
    let named = |n: &str| m.named_tensors().into_iter().find(|t| t.name == n).unwrap().tensor;
    assert!(grad_nonzero(&g, &named("head.o2o.box.p3.conv0.conv.weight")));
    assert!(grad_nonzero(&g, &named("head.o2o.cls.p5.out.bias")));
    for n in ["backbone.stem.conv.weight", "neck.out3.cv1.conv.weight", "head.box.p3.conv0.conv.weight"] {
        assert!(!grad_nonzero(&g, &named(n)), "{n} got a gradient from the o2o loss");
    }
    let lm =
        detection_loss(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &LossConfig::default()).unwrap();
    let g = lm.total.backward().unwrap();
    assert!(grad_nonzero(&g, &named("backbone.stem.conv.weight")));
    assert!(!grad_nonzero(&g, &named("head.o2o.box.p3.conv0.conv.weight")));
}

// ---- tensor names, initialisation, inference switch ------------------------------------------------------------

/// MODEL.md §5's reservation: an `o2o` model is the `o2m` model's tensors, in the same order and drawn from the
/// same stream, followed by the same head tensors under `head.o2o.`, ending with `head.o2o.cls.p5.out.bias`.
#[test]
fn tensor_names_follow_the_reservation() {
    let m2m = FoloModel::new(config("n", 80, "o2m")).unwrap();
    let m2o = FoloModel::new(config("n", 80, "o2o")).unwrap();
    let (a, b) = (m2m.named_tensors(), m2o.named_tensors());
    let head: Vec<_> = a.iter().filter(|t| t.name.starts_with("head.")).collect();
    assert_eq!(b.len(), a.len() + head.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.name, y.name);
        assert!(same_values(&x.tensor, &y.tensor), "{} differs from the o2m model's init", x.name);
    }
    for (x, y) in head.iter().zip(&b[a.len()..]) {
        assert_eq!(y.name, format!("head.o2o.{}", &x.name["head.".len()..]));
        assert_eq!(x.tensor.dims(), y.tensor.dims(), "{}", y.name);
    }
    assert_eq!(b.first().unwrap().name, "backbone.stem.conv.weight");
    assert_eq!(b.last().unwrap().name, "head.o2o.cls.p5.out.bias");
    assert!(folo_nn::io::is_class_output("head.o2o.cls.p3.out.weight"));
    // Same priors in both branches (§6.5).
    let named = |n: &str| b.iter().find(|t| t.name == n).unwrap().tensor.to_vec::<f32>().unwrap();
    for l in ["p3", "p4", "p5"] {
        assert_eq!(named(&format!("head.o2o.cls.{l}.out.bias")), named(&format!("head.cls.{l}.out.bias")));
        assert_eq!(named(&format!("head.o2o.box.{l}.out.bias")), named(&format!("head.box.{l}.out.bias")));
    }
    assert!(m2o.has_o2o() && !m2m.has_o2o());
    assert_eq!((m2o.head_mode(), m2m.head_mode()), (HeadMode::O2o, HeadMode::O2m));
}

/// Eval mode runs only the selected branch: an `o2o` model predicting with `o2m` gives exactly the `o2m` model's
/// outputs (same seed), and with `o2o` the one-to-one branch's, marked for the top-k post-process.
#[test]
fn eval_forward_runs_the_selected_branch() {
    let m2m = FoloModel::new(config("n", 3, "o2m")).unwrap();
    let mut m2o = FoloModel::new(config("n", 3, "o2o")).unwrap();
    perturb(&m2o, 9);
    for (x, y) in m2m.named_tensors().iter().zip(m2o.named_tensors()) {
        init_copy(&x.tensor, &y.tensor);
    }
    m2m.eval();
    m2o.eval();
    let _g = no_grad();
    let x = Tensor::rand_uniform([1, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(2));
    let o = m2o.forward(&x).unwrap();
    assert_eq!((o.mode, o.o2o.is_none()), (HeadMode::O2o, true));
    let r = m2m.forward(&x).unwrap();
    assert!(!same_values(&o.cls_raw, &r.cls_raw), "the o2o branch has its own weights");
    m2o.set_head_mode(HeadMode::O2m).unwrap();
    let p = m2o.forward(&x).unwrap();
    assert_eq!(p.mode, HeadMode::O2m);
    assert!(same_values(&p.box_raw, &r.box_raw) && same_values(&p.cls_raw, &r.cls_raw));
    // forward_head picks a branch without changing the model's mode.
    let q = m2o.forward_head(&x, HeadMode::O2o).unwrap();
    assert!(same_values(&q.cls_raw, &o.cls_raw) && q.mode == HeadMode::O2o);
    assert_eq!(m2o.head_mode(), HeadMode::O2m);
    let mut m2m = m2m;
    assert!(m2m.set_head_mode(HeadMode::O2o).is_err());
    assert!(m2m.forward_head(&x, HeadMode::O2o).is_err());
}

fn init_copy(dst: &Tensor, src: &Tensor) {
    folo_nn::init::copy_(dst, src).unwrap();
}

// ---- .folo files ----------------------------------------------------------------------------------------------

/// An `o2o` model survives save / open (config, mode, every tensor, outputs) and a fused f32 export.
#[test]
fn folo_round_trip() {
    let mut m = FoloModel::new(config("n", 2, "o2o")).unwrap();
    m.set_classes(vec!["a".into(), "b".into()]).unwrap();
    perturb(&m, 5);
    let path = temp("round-trip.folo");
    m.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    assert_eq!(file.metadata().config.head, "o2o");
    let r = FoloModel::open(&path).unwrap();
    assert!(r.has_o2o() && r.head_mode() == HeadMode::O2o && !r.is_training());
    let (a, b) = (m.named_tensors(), r.named_tensors());
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert!(x.name == y.name && same_values(&x.tensor, &y.tensor), "{}", x.name);
    }
    m.eval();
    let _g = no_grad();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(6));
    let (want, got) = (m.forward(&x).unwrap(), r.forward(&x).unwrap());
    assert!(same_values(&want.cls_raw, &got.cls_raw) && same_values(&want.box_raw, &got.box_raw));

    let fused = temp("round-trip-fused.folo");
    m.export(&fused, FloatDType::F32).unwrap();
    let f = FoloModel::open(&fused).unwrap();
    assert!(f.is_fused() && f.has_o2o() && f.head_mode() == HeadMode::O2o);
    let got = f.forward(&x).unwrap();
    for (p, q) in [(&want.box_raw, &got.box_raw), (&want.cls_raw, &got.cls_raw)] {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        for (u, v) in p.iter().zip(&q) {
            assert!((u - v).abs() <= 1e-4 * u.abs().max(v.abs()).max(1.0), "{u} vs {v}");
        }
    }
}

/// `o2m` files (every file written before the one-to-one head) load and run as before: no `head.o2o.*` tensors,
/// NMS post-processing, and no way to switch to a branch they do not have. An unknown head is refused.
#[test]
fn o2m_files_load_and_run_as_before() {
    let m = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    assert_eq!(m.config().head, "o2m");
    perturb(&m, 7);
    let path = temp("o2m.folo");
    m.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    assert!(file.metadata().config.head == "o2m" && !file.metadata().config.has_o2o());
    let mut r = FoloModel::open(&path).unwrap();
    assert!(!r.has_o2o() && r.head_mode() == HeadMode::O2m);
    assert!(r.named_tensors().iter().all(|t| !t.name.starts_with("head.o2o.")));
    assert_eq!(r.named_tensors().len(), m.named_tensors().len());
    m.eval();
    let _g = no_grad();
    let x = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(8));
    let out = r.forward(&x).unwrap();
    assert_eq!((out.mode, out.o2o.is_none()), (HeadMode::O2m, true));
    let nms = NmsConfig { conf: 0.01, ..NmsConfig::default() };
    let want: Vec<_> =
        out.decode_images().unwrap().iter().map(|(b, s)| non_max_suppression(b, s, 2, &nms)).collect();
    assert_eq!(r.predict_detections(&x, &nms).unwrap(), want);
    assert!(r.set_head_mode(HeadMode::O2o).is_err());
    assert!(
        Predictor::new(r, PredictConfig { head: Some(HeadMode::O2o), ..PredictConfig::default() }).is_err()
    );

    let mut bad = ModelConfig::new("n", 2).unwrap();
    bad.head = "dual".into();
    assert!(FoloModel::new(bad).is_err());
}

/// Fine-tuning an `o2o` model from `o2m` weights starts the one-to-one branch as a copy of the one-to-many head.
#[test]
fn transfer_from_o2m_weights_copies_the_head_into_o2o() {
    let src = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    perturb(&src, 12);
    let path = temp("transfer.folo");
    src.save(&path).unwrap();
    let dst = FoloModel::new(config("n", 2, "o2o")).unwrap();
    let report = dst.transfer_from(&ModelFile::open(&path).unwrap()).unwrap();
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    let all: std::collections::HashMap<_, _> = dst.state().into_iter().collect();
    for (name, t) in src.state() {
        assert!(same_values(&all[&name], &t), "{name}");
        if let Some(rest) = name.strip_prefix("head.") {
            assert!(same_values(&all[&format!("head.o2o.{rest}")], &t), "head.o2o.{rest}");
        }
    }
}

// ---- the top-k post-process (no NMS) ----------------------------------------------------------------------------

/// MODEL.md §6.6 by hand: the k best anchors by max-class score, then the k best (anchor, class) pairs (validation)
/// or each anchor once with its best class (prediction), then conf. Overlapping boxes are not suppressed.
#[test]
fn topk_selection_by_hand() {
    let same = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let boxes = [same, same, Xyxy::new(20.0, 20.0, 30.0, 30.0), Xyxy::new(40.0, 0.0, 50.0, 10.0)];
    let scores = [0.9, 0.8, 0.3, 0.95, 0.5, 0.1, 0.05, 0.02];
    let sel = |max_det: usize, conf: f32, multi_label: bool| -> Vec<(u32, f32, f32)> {
        let cfg = NmsConfig { max_det, conf, multi_label, ..NmsConfig::default() };
        topk_detections(&boxes, &scores, 2, &cfg).iter().map(|d| (d.class, d.score, d.bbox.x1)).collect()
    };
    // k = 2: anchors 1 (0.95) and 0 (0.9), which overlap completely and are both kept.
    assert_eq!(sel(2, 0.1, true), vec![(1, 0.95, 0.0), (0, 0.9, 0.0)]);
    assert_eq!(sel(2, 0.1, false), vec![(1, 0.95, 0.0), (0, 0.9, 0.0)]);
    // k = 3: anchors 1, 0, 2. Pairs: 0.95, 0.9, then anchor 0's second class 0.8 (not anchor 2's 0.5).
    assert_eq!(sel(3, 0.1, true), vec![(1, 0.95, 0.0), (0, 0.9, 0.0), (1, 0.8, 0.0)]);
    assert_eq!(sel(3, 0.1, false), vec![(1, 0.95, 0.0), (0, 0.9, 0.0), (0, 0.5, 20.0)]);
    // conf applies after the selection.
    assert_eq!(sel(3, 0.85, true), vec![(1, 0.95, 0.0), (0, 0.9, 0.0)]);
    // k is capped at A.
    assert_eq!(sel(300, 0.0, true).len(), 4);
    // Equal scores: the lower anchor first.
    let tie = [0.6f32, 0.6];
    let d = topk_detections(&boxes[2..], &tie, 1, &NmsConfig { max_det: 1, ..NmsConfig::default() });
    assert_eq!(d[0].bbox, boxes[2]);
    // The class filter applies after the selection.
    let cfg = NmsConfig {
        max_det: 2,
        conf: 0.1,
        multi_label: true,
        classes: Some(vec![0]),
        ..NmsConfig::default()
    };
    assert_eq!(topk_detections(&boxes, &scores, 2, &cfg).len(), 1);
}

fn random_head(b: usize, nc: usize, mean: f32, seed: u64) -> HeadOutput {
    let m = FoloModel::new(ModelConfig::new("n", nc).unwrap()).unwrap();
    let anchors = m.anchors(96, 96);
    let a = anchors.len();
    let rng = &mut Rng::seed(seed);
    HeadOutput {
        box_raw: Tensor::randn([b, 64, a], 0.0, 2.0, rng),
        cls_raw: Tensor::randn([b, nc, a], mean, 2.0, rng),
        mode: HeadMode::O2o,
        o2o: None,
        anchors,
        reg_max: 16,
        num_classes: nc,
    }
}

/// `HeadOutput::detections` of one-to-one outputs (the CPU candidate path, single-label; the full decode,
/// multi-label) equals the full decode followed by `topk_detections`.
#[test]
fn o2o_detections_match_the_full_decode() {
    for (b, nc, mean, cfg) in [
        (2usize, 5usize, -2.5f32, NmsConfig { conf: 0.25, max_det: 7, ..NmsConfig::default() }),
        (1, 5, 0.0, NmsConfig { conf: 0.3, ..NmsConfig::default() }),
        (3, 5, -3.0, NmsConfig::val()),
        (1, 5, -1.0, NmsConfig { max_det: 9, ..NmsConfig::val() }),
    ] {
        let out = random_head(b, nc, mean, (nc * b) as u64 + mean.to_bits() as u64);
        let want: Vec<_> =
            out.decode_images().unwrap().iter().map(|(bx, sc)| topk_detections(bx, sc, nc, &cfg)).collect();
        let got = out.detections(&cfg).unwrap();
        assert_eq!(got, want);
        assert!(got.iter().all(|d| d.len() <= cfg.max_det) && got.iter().any(|d| !d.is_empty()));
    }
}

/// The predictor of an `o2o` model predicts end to end (top-k, no NMS) and can be switched to its o2m branch.
#[test]
fn predictor_uses_the_model_head_and_can_switch() {
    let m = FoloModel::new(config("n", 2, "o2o")).unwrap();
    perturb(&m, 13);
    let img =
        folo_data::Image::from_raw(80, 64, (0..80 * 64 * 3).map(|i| (i * 7 % 251) as u8).collect()).unwrap();
    // The head is near its priors (scores about 1e-3), so a tiny conf keeps detections.
    let cfg = PredictConfig { imgsz: 64, conf: 1e-6, max_det: 50, ..PredictConfig::default() };
    let mut p = Predictor::new(m, cfg.clone()).unwrap();
    assert_eq!(p.model().head_mode(), HeadMode::O2o);
    let e2e = p.predict(&img).unwrap();
    p.set_config(PredictConfig { head: Some(HeadMode::O2m), ..cfg.clone() }).unwrap();
    assert_eq!(p.model().head_mode(), HeadMode::O2m);
    let nms = p.predict(&img).unwrap();
    assert_eq!(e2e.detections.len(), 50);
    assert!(!nms.detections.is_empty() && e2e.detections != nms.detections);
    // `head: None` keeps the current branch.
    p.set_config(cfg).unwrap();
    assert_eq!(p.model().head_mode(), HeadMode::O2m);
}
