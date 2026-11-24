//! The task-aligned assigner and the detection loss on the GPU (folo-gpu's `detect` kernels) against the host
//! reference (`assign_batch` + `loss_with_assignment`): the same assignment (positives, gts, labels; soft targets
//! to float rounding), the same reported items, and the same gradients.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.

use folo_core::rng::Rng;
use folo_core::{Device, Tensor};
use folo_vision::boxes::Xyxy;
use folo_vision::decode::{Anchors, make_anchors};
use folo_vision::loss::{
    BatchAssignment, ImageTargets, LossConfig, assign_batch, assign_batch_device, detection_loss,
    detection_loss_host,
};

fn gpu() -> Option<Device> {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    folo_gpu::init().ok()
}

macro_rules! gpu {
    () => {
        match gpu() {
            Some(d) => d,
            None => return,
        }
    };
}

/// A batch: head outputs that decode to boxes of a few cells around each anchor, and ground truth with overlapping
/// boxes (conflicts), a box smaller than 16 px (the candidate enlargement), boxes on several levels and an empty
/// image.
fn case(seed: u64, size: u32, nc: usize) -> (Anchors, Tensor, Tensor, Vec<ImageTargets>) {
    let anchors = make_anchors(size, size, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let mut rng = Rng::seed(seed);
    let b = 4;
    let box_raw = Tensor::randn([b, 64, a], 0.0, 1.5, &mut rng);
    let cls_raw = Tensor::rand_uniform([b, nc, a], -6.0, 2.0, &mut rng);
    let s = size as f32;
    let mut targets = Vec::new();
    for i in 0..b {
        let n = if i == 2 { 0 } else { 3 + 2 * i };
        let mut boxes = Vec::new();
        let mut labels = Vec::new();
        for j in 0..n {
            let (w, h) =
                if j == 0 { (10.0, 7.0) } else { (rng.uniform(16.0, s * 0.6), rng.uniform(16.0, s * 0.6)) };
            let (cx, cy) = (rng.uniform(w / 2.0, s - w / 2.0), rng.uniform(h / 2.0, s - h / 2.0));
            boxes.push(Xyxy::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0));
            labels.push((rng.uniform(0.0, nc as f32) as u32).min(nc as u32 - 1));
        }
        // Two near-duplicates of the last box make anchors contested.
        if let Some(&last) = boxes.last() {
            boxes.push(Xyxy::new(last.x1 + 1.5, last.y1 - 1.0, last.x2 + 2.0, last.y2 + 0.5));
            labels.push(0);
        }
        targets.push(ImageTargets { boxes, labels });
    }
    (anchors, box_raw, cls_raw, targets)
}

fn assert_same_assignment(got: &BatchAssignment, want: &BatchAssignment) {
    assert!(want.num_fg() > 0, "the case has positives");
    assert_eq!(got.image, want.image, "images");
    assert_eq!(got.anchor, want.anchor, "anchors");
    assert_eq!(got.label, want.label, "labels");
    assert_eq!(got.target, want.target, "target boxes");
    for (k, (g, w)) in got.score.iter().zip(&want.score).enumerate() {
        assert!((g - w).abs() <= 1e-5 + 1e-4 * w.abs(), "score {k}: {g} vs {w}");
    }
}

/// The device assigner picks the same positives, gts and labels as the host assigner, with the same soft targets
/// to float rounding, on several sizes and class counts, for the one-to-many assignment and the one-to-one one
/// (docs/MODEL.md §6.6: topk 7, then one positive per gt after conflicts).
#[test]
fn assigner_matches_host() {
    let dev = gpu!();
    for cfg in [LossConfig::default(), LossConfig::o2o()] {
        for (seed, size, nc) in [(1, 96, 3), (2, 160, 5), (3, 128, 1), (4, 224, 12)] {
            let (anchors, box_raw, cls_raw, targets) = case(seed, size, nc);
            let want = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
            let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
            let got = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
            assert_same_assignment(&got, &want);
            if cfg.tal.topk2 == Some(1) {
                // At most one positive per gt: no image has more positives than gts.
                for (i, t) in targets.iter().enumerate() {
                    assert!(want.image.iter().filter(|&&j| j as usize == i).count() <= t.len());
                }
            }
        }
    }
}

/// Other assigner settings take the kernels' general paths (pow for non-integer beta and alpha != 0.5, a topk
/// above 16, zero exponents with `powf`'s `0^0 = 1`) and still agree.
#[test]
fn assigner_matches_host_with_other_settings() {
    let dev = gpu!();
    let (anchors, box_raw, cls_raw, targets) = case(5, 160, 4);
    let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
    for (topk, alpha, beta) in [(24, 1.0, 4.5), (10, 0.0, 6.0), (10, 0.5, 0.0), (7, 0.0, 0.0)] {
        let mut cfg = LossConfig::default();
        cfg.tal.topk = topk;
        cfg.tal.alpha = alpha;
        cfg.tal.beta = beta;
        let want = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
        let got = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert_same_assignment(&got, &want);
    }
}

#[track_caller]
fn close_vec(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, f32::NEG_INFINITY);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}[{i}]: {g}");
        let e = (g - w).abs() - rtol * w.abs();
        if e > worst.1 {
            worst = (i, e);
        }
    }
    let i = worst.0;
    assert!(worst.1 <= atol, "{what}[{i}]: got {}, want {} (atol {atol}, rtol {rtol})", got[i], want[i]);
}

/// The device loss reports the host loss's items and total, and its fused backward gives the host autograd's
/// gradients for both head outputs.
#[test]
fn loss_and_gradients_match_host() {
    let dev = gpu!();
    for (seed, size, nc, cfg) in [
        (11, 96, 3, LossConfig::default()),
        (12, 160, 7, LossConfig::default()),
        (13, 160, 7, LossConfig::o2o()),
    ] {
        let (anchors, box_raw, cls_raw, targets) = case(seed, size, nc);
        // Device leaves first: a transfer of a tensor that requires grad is recorded (not a leaf).
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite());
        assert_eq!(got.num_fg, want.num_fg, "positives");
        let rel = |g: f32, w: f32| (g - w).abs() <= 1e-4 * w.abs().max(1e-3);
        let (gi, wi) = (got.items, want.items);
        assert!(rel(gi.box_, wi.box_) && rel(gi.cls, wi.cls) && rel(gi.dfl, wi.dfl), "{gi:?} vs {wi:?}");
        assert!(rel(got.total_value, want.total_value), "{} vs {}", got.total_value, want.total_value);
        assert!(rel(got.target_score_sum, want.target_score_sum));
        assert!(rel(got.total.to_vec::<f32>().unwrap()[0], want.total_value));
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        for (name, d, h) in [("box_raw", &bd, &bh), ("cls_raw", &cd, &ch)] {
            let g = gg.get(d).unwrap().to_vec::<f32>().unwrap();
            let w = gw.get(h).unwrap().to_vec::<f32>().unwrap();
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            close_vec(&g, &w, 1e-4 * scale, 1e-3, name);
        }
    }
}

/// A batch without any gt: only the background classification term, no positives.
#[test]
fn empty_batch_matches_host() {
    let dev = gpu!();
    let (anchors, box_raw, cls_raw, _) = case(21, 96, 2);
    let targets = vec![ImageTargets::default(); 4];
    let cfg = LossConfig::default();
    let want = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    let cd = cls_raw.to_device(dev).unwrap().requires_grad_();
    let got = detection_loss(&box_raw.to_device(dev).unwrap(), &cd, &anchors, &targets, &cfg).unwrap();
    assert_eq!((got.num_fg, got.items.box_, got.items.dfl), (0, 0.0, 0.0));
    assert!((got.items.cls - want.items.cls).abs() <= 1e-4 * want.items.cls);
    let g = got.total.backward().unwrap().get(&cd).unwrap().to_vec::<f32>().unwrap();
    assert!(g.iter().all(|v| v.is_finite() && *v >= 0.0));
}

/// A non-finite head value gives a non-finite loss, as on the host.
#[test]
fn non_finite_head_gives_non_finite_loss() {
    let dev = gpu!();
    let (anchors, box_raw, cls_raw, targets) = case(31, 96, 2);
    let cfg = LossConfig::default();
    let mut v = cls_raw.to_vec::<f32>().unwrap();
    v[17] = f32::NAN;
    let cls_bad = Tensor::from_vec(v, cls_raw.shape()).unwrap();
    let bd = box_raw.to_device(dev).unwrap();
    let out = detection_loss(&bd, &cls_bad.to_device(dev).unwrap(), &anchors, &targets, &cfg).unwrap();
    assert!(!out.is_finite());
    let mut v = box_raw.to_vec::<f32>().unwrap();
    v[5] = f32::INFINITY;
    let box_bad = Tensor::from_vec(v, box_raw.shape()).unwrap().to_device(dev).unwrap();
    let out = detection_loss(&box_bad, &cls_raw.to_device(dev).unwrap(), &anchors, &targets, &cfg).unwrap();
    assert!(!out.is_finite());
}

/// `folo_gpu::detect::detection_loss` hands out a copy of its stats: overwriting it changes neither the backward nor
/// a second read.
#[test]
fn returned_stats_do_not_feed_the_backward() {
    let dev = gpu!();
    let (anchors, box_raw, cls_raw, targets) = case(41, 96, 3);
    let bd = box_raw.to_device(dev).unwrap().requires_grad_();
    let cd = cls_raw.to_device(dev).unwrap().requires_grad_();
    let boxes: Vec<Vec<[f32; 4]>> =
        targets.iter().map(|t| t.boxes.iter().map(|b| [b.x1, b.y1, b.x2, b.y2]).collect()).collect();
    let gts: Vec<_> = boxes
        .iter()
        .zip(&targets)
        .map(|(b, t)| folo_gpu::detect::ImageGt { boxes: b, labels: &t.labels })
        .collect();
    let at = folo_gpu::detect::AnchorTable { points: &anchors.points, strides: &anchors.strides };
    let cfg = LossConfig::default();
    let s = folo_gpu::detect::LossSettings {
        reg_max: 16,
        gains: [cfg.box_gain, cfg.cls_gain, cfg.dfl_gain],
        mal_gamma: cfg.mal_gamma,
        nwd_weight: cfg.nwd_weight,
        box_loss: folo_gpu::detect::BoxLossKind::Ciou,
        rpiou_beta: cfg.rpiou_beta,
        rpiou_kappa: cfg.rpiou_kappa,
        rpiou_delta: cfg.rpiou_delta,
        dfl_ot_weight: cfg.dfl_ot_weight,
        tal: folo_gpu::detect::TalSettings {
            topk: cfg.tal.topk,
            topk2: cfg.tal.topk2,
            alpha: cfg.tal.alpha,
            beta: cfg.tal.beta,
            eps: cfg.tal.eps,
            min_box_size: cfg.tal.min_box_size,
            eps_free: cfg.tal.eps_free,
            nwd_weight: cfg.tal.nwd_weight,
        },
    };
    let grads = |mutate: bool| {
        let out = folo_gpu::detect::detection_loss(&bd, &cd, &at, &gts, &s).unwrap();
        if mutate {
            out.stats.fill_assign(0.0).unwrap();
        }
        let g = out.total.backward().unwrap();
        (g.get(&bd).unwrap().to_vec::<f32>().unwrap(), g.get(&cd).unwrap().to_vec::<f32>().unwrap())
    };
    let (b0, c0) = grads(false);
    let (b1, c1) = grads(true);
    assert!(b0.iter().any(|v| *v != 0.0) && c0.iter().any(|v| *v != 0.0));
    assert_eq!((b0, c0), (b1, c1));
}

/// Settings the kernels do not take (`reg_max` above 32) fall back to the host assigner on GPU tensors instead of
/// failing.
#[test]
fn unsupported_settings_fall_back_to_the_host() {
    let dev = gpu!();
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let mut rng = Rng::seed(51);
    let cfg = LossConfig { reg_max: 33, ..LossConfig::default() };
    let box_raw = Tensor::randn([1, 4 * 33, anchors.len()], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
    let cls_raw = Tensor::rand_uniform([1, 2, anchors.len()], -4.0, 1.0, &mut rng).to_device(dev).unwrap();
    let targets = vec![ImageTargets { boxes: vec![Xyxy::new(8.0, 10.0, 40.0, 50.0)], labels: vec![1] }];
    assert!(!folo_vision::loss::device_loss_supported(&box_raw, &cfg));
    let got = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    let want = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    assert!(got.is_finite() && got.num_fg > 0);
    assert_eq!(got.total_value, want.total_value);
}

/// CPU-vs-GPU parity for `R-PIoU`, `OT-DFL`, `R-PIoU + OT-DFL`, `R-PIoU + NWD`, and `reg_max = 2`.
#[test]
fn new_math_formulas_match_host() {
    let dev = gpu!();
    use folo_vision::loss::BoxLossKind;
    let rpiou = LossConfig { box_loss: BoxLossKind::Rpiou, ..LossConfig::default() };
    let otdfl = LossConfig { dfl_ot_weight: 0.25, ..LossConfig::default() };
    let box_combo = LossConfig { dfl_ot_weight: 0.25, ..rpiou };
    let rpiou_nwd = LossConfig { nwd_weight: 0.5, ..rpiou };

    for (name, seed, cfg) in [
        ("rpiou", 61, rpiou),
        ("otdfl", 62, otdfl),
        ("rpiou+otdfl", 63, box_combo),
        ("rpiou+nwd", 64, rpiou_nwd),
    ] {
        let (anchors, box_raw, cls_raw, targets) = case(seed, 128, 5);
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want_asg = assign_batch(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got_asg = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert_same_assignment(&got_asg, &want_asg);
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite(), "{name}: non-finite");
        assert_eq!(got.num_fg, want.num_fg, "{name}: num_fg");
        let rel = |g: f32, w: f32| (g - w).abs() <= 2e-4 * w.abs().max(1e-3);
        let (gi, wi) = (got.items, want.items);
        assert!(
            rel(gi.box_, wi.box_) && rel(gi.cls, wi.cls) && rel(gi.dfl, wi.dfl),
            "{name}: {gi:?} vs {wi:?}"
        );
        assert!(
            rel(got.total_value, want.total_value),
            "{name}: {} vs {}",
            got.total_value,
            want.total_value
        );
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        for (tensor_name, d, h) in [("box_raw", &bd, &bh), ("cls_raw", &cd, &ch)] {
            let g = gg.get(d).unwrap().to_vec::<f32>().unwrap();
            let w = gw.get(h).unwrap().to_vec::<f32>().unwrap();
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            close_vec(&g, &w, 2e-4 * scale.max(1e-4), 2e-3, &format!("{name}/{tensor_name}"));
        }
    }

    // `reg_max = 2` (OT-DFL vanishes identically; R-PIoU + DFL still match CPU).
    {
        let anchors = make_anchors(64, 64, &[8, 16], 0.5);
        let mut rng = Rng::seed(65);
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            dfl_ot_weight: 0.5,
            ..LossConfig::default()
        };
        let box_raw = Tensor::randn([1, 8, anchors.len()], 0.0, 1.0, &mut rng);
        let cls_raw = Tensor::rand_uniform([1, 3, anchors.len()], -4.0, 1.0, &mut rng);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 38.0, 44.0)], labels: vec![1] }];
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0);
        assert!((got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1e-3));
        let (gg, gw) = (got.total.backward().unwrap(), want.total.backward().unwrap());
        let g = gg.get(&bd).unwrap().to_vec::<f32>().unwrap();
        let w = gw.get(&bh).unwrap().to_vec::<f32>().unwrap();
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g, &w, 2e-4 * scale.max(1e-4), 2e-3, "reg_max=2/box_raw");
    }
}

/// Edge-case CPU-vs-GPU regressions:
/// 1. Extreme OT-DFL logits `[-1e8, -1e8, -1e8, 0]` on an active foreground anchor (`R = 4`, left bin `j = 0`):
///    shifted tail accumulators keep `ln(2)` / `ln(3)` exact, matching CPU and preserving zero-sum gradients.
/// 2. Saturated `R-PIoU` edge penalty (`rpiou_kappa = 3e38`, `box_gain = 7.5`): GPU backward must stay finite
///    (no `Inf * 0 = NaN` overflow) and match CPU.
#[test]
fn new_math_extreme_edge_cases_match_host() {
    let dev = gpu!();
    use folo_vision::loss::BoxLossKind;

    // 1. Extreme OT-DFL logits on active foreground anchor (`R = 4`, `j = 0`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 16 * a];
        for side in 0..4 {
            for k in 0..3 {
                for j in 0..a {
                    box_data[(side * 4 + k) * a + j] = -1e8;
                }
            }
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 16, a]).unwrap();
        let cls_raw = Tensor::from_slice(&vec![0.0f32; a], [1, 1, a]).unwrap();
        // Anchor (3, 3) sits at px (28, 28); target box gives exact ltrb = 0.25 grid cells -> j = 0.
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(26.0, 26.0, 30.0, 30.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 1.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "extreme OT-DFL must be finite with fg > 0");
        assert!(
            (got.total_value - want.total_value).abs() <= 1e-4 * want.total_value.abs().max(1.0),
            "extreme OT-DFL loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_gpu.iter().all(|v| v.is_finite()), "extreme OT-DFL GPU grad must be finite");
        assert!(g_gpu.iter().any(|v| v.abs() > 0.0), "extreme OT-DFL GPU grad must be non-zero");
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-4, "extreme_otdfl/box_raw");
        // Verify shift invariance: per-(anchor, side) sum of logits gradient is 0.
        for j in 0..a {
            for side in 0..4 {
                let s: f32 = (0..4).map(|k| g_gpu[(side * 4 + k) * a + j]).sum();
                assert!(s.abs() < 1e-4, "anchor {j} side {side}: non-zero logit grad sum {s}");
            }
        }
    }

    // 2. Saturated `R-PIoU` (`rpiou_kappa = 3e38`, `box_gain = 7.5`).
    {
        let anchors = make_anchors(64, 64, &[8, 16], 0.5);
        let mut rng = Rng::seed(71);
        let cfg = LossConfig {
            box_loss: BoxLossKind::Rpiou,
            rpiou_kappa: 3e38,
            box_gain: 7.5,
            ..LossConfig::default()
        };
        let box_raw = Tensor::randn([1, 64, anchors.len()], 0.0, 1.0, &mut rng);
        let cls_raw = Tensor::rand_uniform([1, 2, anchors.len()], -3.0, 1.0, &mut rng);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 46.0, 50.0)], labels: vec![0] }];
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "saturated R-PIoU must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1e-3),
            "saturated R-PIoU loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_gpu.iter().all(|v| v.is_finite()), "saturated R-PIoU GPU grad must be finite");
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale.max(1e-4), 2e-3, "saturated_rpiou/box_raw");
    }

    // 3. Exact-match `R-PIoU` (`reg_max = 2`, equal logits, target half a grid cell from anchor on every side,
    //    `rpiou_kappa = 3e38`, `box_gain = 7.5`): `ke2_raw == 0` must not overflow into `Inf * 0 = NaN` on CPU or GPU.
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        // Equal logits -> softmax = [0.5, 0.5] -> decoded ltrb = 0.5 grid cells (4 px) on every side.
        let box_raw = Tensor::zeros([1, 8, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        // Anchor (3, 3) is at px (28, 28); target [24, 24, 32, 32] matches predicted box identically.
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_kappa: 3e38,
            box_gain: 7.5,
            dfl_gain: 0.0,
            cls_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "exact-match R-PIoU must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 1e-5,
            "{} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "exact-match R-PIoU CPU grad must be finite");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "exact-match R-PIoU GPU grad must be finite");
        close_vec(&g_gpu, &g_cpu, 1e-5, 1e-4, "exact_match_rpiou/box_raw");
    }

    // 4. Extreme `OT-DFL` weight (`reg_max = 3`, uniform logits, target distances 0.5 bins, `dfl_ot_weight = 3e38`,
    //    `dfl_gain = 1.5`): both loss and gradients fit in `f32` and must match across CPU and GPU without Inf.
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 3,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.5,
            dfl_ot_weight: 3e38,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert!(got.is_finite() && got.num_fg > 0, "huge-weight OT-DFL must be finite");
        assert!(
            (got.total_value - want.total_value).abs() <= 2e-4 * want.total_value.abs().max(1.0),
            "huge-weight OT-DFL loss: {} vs {}",
            got.total_value,
            want.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "huge-weight OT-DFL CPU grad must be finite");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "huge-weight OT-DFL GPU grad must be finite");
        assert!(g_gpu.iter().any(|v| v.abs() > 0.0), "huge-weight OT-DFL GPU grad must be non-zero");
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale.max(1.0), 2e-3, "huge_weight_otdfl/box_raw");
    }

    // 5. Multi-positive extreme `OT-DFL` (`reg_max = 3`, uniform logits, 3 exact-match positives with `t = 1`,
    //    `dfl_ot_weight = 3e38`, `dfl_gain = 1.5`): raw per-positive sum would overflow `f32`, so both CPU and GPU
    //    must normalize by `tss` before multiplying by `dfl_ot_weight * dfl_gain`, matching an `f64` reference.
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        // Three well-separated 16x16 boxes centered on anchors (1,1), (5,1), (1,5): each has ltrb = 1.0 grid cells
        // (matching uniform R=3 decoded d = 1.0 exactly, IoU = 1, t = 1, j = 1, l_tail = ln(3/2)).
        let targets = vec![ImageTargets {
            boxes: vec![
                Xyxy::new(4.0, 4.0, 20.0, 20.0),
                Xyxy::new(36.0, 4.0, 52.0, 20.0),
                Xyxy::new(4.0, 36.0, 20.0, 52.0),
            ],
            labels: vec![0, 0, 0],
        }];
        let cfg = LossConfig {
            reg_max: 3,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.5,
            dfl_ot_weight: 3e38,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let ref_loss = (1.5f64 * 3e38f64 * 1.5f64.ln()) as f32;
        assert!(got.is_finite() && want.is_finite(), "3-pos extreme OT-DFL must be finite");
        assert_eq!((got.num_fg, want.num_fg), (3, 3));
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "CPU 3-pos OT-DFL {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "GPU 3-pos OT-DFL {} vs f64 ref {ref_loss}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()));
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e35, "expected large finite OT-DFL gradient, got {scale}");
        close_vec(&g_gpu, &g_cpu, 2e-4 * scale, 2e-3, "three_pos_otdfl/box_raw");
    }

    // 6. Extreme `R-PIoU` floor (`rpiou_delta = 1e20`, `rpiou_kappa = 3e38`, unit-size GT `[3,3,4,4]` and
    //    predicted `[2.5,2.5,4.5,4.5]` in grid units -> `|dx| = 0.5` on all 4 edges): naive `delta^2` overflows
    //    `f32` and drops the penalty to 0; scaled coordinates match the independent `f64` reference (`0.00744417`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let box_raw = Tensor::zeros([1, 12, a]);
        let cls_raw = Tensor::zeros([1, 1, a]);
        // Anchor (3, 3) sits at (28, 28) px = (3.5, 3.5) grid units; uniform R=3 decodes d = 1.0 -> [2.5,2.5,4.5,4.5].
        // GT [24,24,32,32] px = [3,3,4,4] grid units (w=h=1); all 4 edge residuals have magnitude 0.5 grid units.
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 3,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 3e38,
            rpiou_delta: 1e20,
            box_gain: 1.0,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let ke2_ref = 3e38f64 * 0.25f64 * (4.0 * 0.25f64 / (1.0f64 + 1e40f64));
        let edge_ref = ke2_ref / (1.0 + ke2_ref); // 0.007444168734491315
        let focal_ref = (1.0 - 0.25f64) * (1.0 + 0.5 * 0.25f64); // 0.84375
        let t_w = want.target_score_sum as f64; // u_max = CIoU = 0.25, tss = max(0.25, 1.0) = 1.0
        let ref_loss = (t_w * (focal_ref + edge_ref)) as f32;
        assert_eq!((got.num_fg, want.num_fg), (1, 1));
        assert!((t_w - 0.25).abs() < 1e-5, "expected target_score_sum 0.25, got {t_w}");
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-5,
            "CPU scaled R-PIoU {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-5,
            "GPU scaled R-PIoU {} vs f64 ref {ref_loss}",
            got.total_value
        );
        // Also check gradient against a pure-focal baseline (kappa -> 0) to confirm non-zero edge gradient.
        let cfg_no_edge = LossConfig { rpiou_kappa: 1e-30, ..cfg };
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        let bh0 = Tensor::zeros([1, 12, a]).requires_grad_();
        let ch0 = Tensor::zeros([1, 1, a]).requires_grad_();
        let g_no_edge = detection_loss_host(&bh0, &ch0, &anchors, &targets, &cfg_no_edge)
            .unwrap()
            .total
            .backward()
            .unwrap()
            .get(&bh0)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let fg = 3 * 8 + 3;
        // Analytical d(t * edge_pen)/d(d_side) = t * 0.5 * 3e38 * 0.5 / ((1 + 1e40) * 1.0075^2),
        // and d(d_side)/d(logit_2) = (1/3) * (2 - 1) = +1/3.
        let edge_d = (t_w * 0.5f64 * 3e38 * 0.5 / ((1.0 + 1e40) * (1.0 + ke2_ref).powi(2)) / 3.0) as f32;
        let got_edge_d = g_gpu[2 * a + fg] - g_no_edge[2 * a + fg];
        assert!(
            (got_edge_d - edge_d).abs() <= 1e-5,
            "GPU scaled R-PIoU edge grad delta {got_edge_d} vs analytical {edge_d}"
        );
        close_vec(&g_gpu, &g_cpu, 1e-5, 1e-4, "scaled_rpiou_delta1e20/box_raw");
    }

    // 7. Extreme `R-PIoU` ratio (`gt = [0, 0, 2.5e-20, 1]`, `pred = [1.25e-38, 0, 2.5e-20, 1]`, `kappa = 1e38`,
    //    `delta0 = 1e-30`): `sqrt(kappa) / sx` overflows `f32`, so exponent-split scaling on GPU and `f64` on CPU
    //    must keep the finite penalty (`~0.862069`) and finite edge gradient (`~1.9025e37`) instead of saturating.
    {
        // Direct CPU `rpiou_loss` check against exact f64 reference (`iou ≈ 2.5e-13` due to `eps = 1e-7`,
        // so `focal_iou ≈ 1.0` and `loss ≈ 1.0 + edge_ref ≈ 1.862069`).
        let b1 = Tensor::from_slice(&[1.25e-38f32, 0.0, 2.5e-20, 1.0], [1, 4]).unwrap().requires_grad_();
        let b2 = Tensor::from_slice(&[0.0f32, 0.0, 2.5e-20, 1.0], [1, 4]).unwrap();
        let l = folo_vision::loss::rpiou_loss(&b1, &b2, 0.5, 1e38, 1e-30).unwrap();
        let lv = l.item::<f32>().unwrap();
        let g = l.backward().unwrap().get(&b1).unwrap().to_vec::<f32>().unwrap();
        let ke2_ref = 1e38f64 * 0.25 * ((1.25e-38f64).powi(2) / ((2.5e-20f64).powi(2) + (1e-30f64).powi(2)));
        let edge_ref = (ke2_ref / (1.0 + ke2_ref)) as f32; // ~0.86206895
        let loss_ref = 1.0 + edge_ref;
        let grad_ref = (0.5f64 * 1e38 * 1.25e-38
            / (((2.5e-20f64).powi(2) + (1e-30f64).powi(2)) * (1.0 + ke2_ref).powi(2)))
            as f32; // ~1.902497e37
        assert!((lv - loss_ref).abs() <= 1e-5, "CPU rpiou_loss {lv} vs f64 ref {loss_ref}");
        assert!((g[0] - grad_ref).abs() <= 1e-4 * grad_ref, "CPU rpiou grad {} vs f64 ref {grad_ref}", g[0]);

        // End-to-end GPU vs CPU `detection_loss` exercising the exact same `sqrt(kappa) / sx = 4e38 > f32::MAX`
        // exponent-split path (`sx = max(w2, delta0) = 2.5e-20`, `kappa = 1e38`, `dx1 = d_left ≈ 1.25e-38`,
        // `dx2 = dy1 = dy2 = 0` bit-exact via `r = [0, -1000]` -> `d2 = 0`, `tal.nwd_weight = 1.0` -> `t ≈ 1`).
        let l_left = (1.25e-38f32).ln();
        let d_left = Tensor::from_slice(&[0.0f32, l_left], [1, 2])
            .unwrap()
            .softmax(1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[1];
        let ax = 2.0 * d_left; // a.x = ax - d_left = d_left (~1.25e-38), a.z = ax + 0 = ax
        let anchors = Anchors { points: vec![[ax, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let box_raw =
            Tensor::from_slice(&[0.0, l_left, 0.0, 0.0, 0.0, -1000.0, 0.0, 0.0], [1, 8, 1]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, 1]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, ax, 1.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 1e38,
            rpiou_delta: 2.5e-20,
            box_gain: 1.0,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(t_w > 0.9 && got.num_fg == 1 && want.num_fg == 1);
        let ke2_g = 1e38f64 * 0.25 * ((d_left as f64).powi(2) / ((ax as f64).powi(2) + (2.5e-20f64).powi(2)));
        let loss_g_ref = 1.0 + ke2_g / (1.0 + ke2_g);
        let total_ref = (t_w * loss_g_ref) as f32;
        assert!(
            (want.total_value - total_ref).abs() <= 1e-4 * total_ref,
            "CPU exponent-scaled R-PIoU {} vs ref {total_ref}",
            want.total_value
        );
        assert!(
            (got.total_value - total_ref).abs() <= 1e-3 * total_ref,
            "GPU exponent-scaled R-PIoU {} vs ref {total_ref}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        let grad_g_ref = 0.5f64 * 1e38 * (d_left as f64)
            / (((ax as f64).powi(2) + (2.5e-20f64).powi(2)) * (1.0 + ke2_g).powi(2));
        // dL/d(logit_left_1) = -t_w * grad_g_ref * d_left ≈ -0.2378 (non-zero; false saturation would give ~0).
        let logit_g_ref = -(t_w * grad_g_ref * d_left as f64) as f32;
        assert!(
            (g_gpu[1] - logit_g_ref).abs() <= 2e-2 * logit_g_ref.abs(),
            "GPU exponent-scaled R-PIoU logit grad {} vs ref {logit_g_ref}",
            g_gpu[1]
        );
        close_vec(&g_gpu, &g_cpu, 2e-2 * logit_g_ref.abs(), 2e-2, "exponent_scaled_rpiou/box_raw");
    }

    // 8. Extreme `OT-DFL` logits (`reg_max = 6`, logits `[-1e38; 5, 0]` per side, target distance `0.5` per side,
    //    `dfl_ot_weight = 0.25`, `dfl_gain = 1.0`): naive `l_tail` sums four `1e38` terms and overflows `f32`;
    //    widened CPU `f64` and GPU `2^-32` scaled reduction must match the finite `f64` reference (`~2e38 * t`).
    {
        let anchors = make_anchors(64, 64, &[8], 0.5);
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 24 * a];
        for side in 0..4 {
            for k in 0..5 {
                for j in 0..a {
                    box_data[(side * 6 + k) * a + j] = -1e38;
                }
            }
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 24, a]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, a]);
        // Anchor (3, 3) sits at px (28, 28); target [24, 24, 32, 32] gives exact ltrb = 0.5 grid cells -> j = 0.
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(24.0, 24.0, 32.0, 32.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 6,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 0.25,
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(t_w > 0.0 && got.num_fg == 1 && want.num_fg == 1);
        // CE contributes t * 1e38; OT contributes 0.25 * t * (4 * 1e38) = t * 1e38 -> total = 2e38 * t.
        let ref_loss = (2e38f64 * t_w) as f32;
        assert!(got.is_finite() && want.is_finite(), "extreme -1e38 OT-DFL must be finite");
        assert!(
            (want.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "CPU -1e38 OT-DFL {} vs f64 ref {ref_loss}",
            want.total_value
        );
        assert!(
            (got.total_value - ref_loss).abs() <= 1e-4 * ref_loss,
            "GPU -1e38 OT-DFL {} vs f64 ref {ref_loss}",
            got.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()));
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-3, "extreme_neg1e38_otdfl/box_raw");
    }

    // 9. Equal tail and full sums in `OT-DFL` (`reg_max = 4`, target distance `1.5`, logits `[-1000, 0, 0, -1000]`,
    //    `dfl_ot_weight = 1e8`, `dfl_gain = 1.0`): all probability mass sits in bins `{1, 2}` so both retained tails
    //    equal the full sum (`2`). CPU and GPU must both report exact `0` OT contribution (matching `dfl_ot_weight = 0`).
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![8.0], ..Anchors::default() };
        let a = anchors.len();
        let mut box_data = vec![0.0f32; 16 * a];
        for side in 0..4 {
            box_data[side * 4] = -1000.0;
            box_data[side * 4 + 3] = -1000.0;
        }
        let box_raw = Tensor::from_slice(&box_data, [1, 16, a]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, a]);
        // Anchor at grid (3.5, 3.5) = px (28, 28); target [16, 16, 40, 40] gives exact ltrb = 1.5 grid cells -> j = 1.
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(16.0, 16.0, 40.0, 40.0)], labels: vec![0] }];
        let cfg0 = LossConfig {
            reg_max: 4,
            box_gain: 0.0,
            cls_gain: 0.0,
            dfl_gain: 1.0,
            dfl_ot_weight: 0.0,
            ..LossConfig::default()
        };
        let cfg_ot = LossConfig { dfl_ot_weight: 1e8, ..cfg0 };
        let base_cpu = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg0).unwrap();
        let ot_cpu = detection_loss_host(&box_raw, &cls_raw, &anchors, &targets, &cfg_ot).unwrap();
        let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
        let ot_gpu = detection_loss(&bd, &cd, &anchors, &targets, &cfg_ot).unwrap();
        assert_eq!((ot_cpu.num_fg, ot_gpu.num_fg), (1, 1));
        assert!(
            (ot_cpu.total_value - base_cpu.total_value).abs() <= 1e-6,
            "CPU equal-sum OT-DFL added spurious loss: {} vs base {}",
            ot_cpu.total_value,
            base_cpu.total_value
        );
        assert!(
            (ot_gpu.total_value - base_cpu.total_value).abs() <= 1e-5,
            "GPU equal-sum OT-DFL {} vs base {}",
            ot_gpu.total_value,
            base_cpu.total_value
        );
    }

    // 10. Extreme `box_gain * kappa` where intermediate coordinate gradient (`~3.2e44`) overflows `f32` while the
    //     final logit gradient after the DFL softmax Jacobian (`~-3.2e24`) fits in `f32`:
    //     `d_left ≈ 1e-20`, `rpiou_delta = 0.1`, `rpiou_kappa = 1e38`, `box_gain = 1e25`.
    {
        let l_left = (1e-20f32).ln();
        let d_left = Tensor::from_slice(&[0.0f32, l_left], [1, 2])
            .unwrap()
            .softmax(1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()[1];
        let ax = 2.0 * d_left;
        let anchors = Anchors { points: vec![[ax, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let box_raw =
            Tensor::from_slice(&[0.0, l_left, 0.0, 0.0, 0.0, -1000.0, 0.0, 0.0], [1, 8, 1]).unwrap();
        let cls_raw = Tensor::zeros([1, 1, 1]);
        let targets = vec![ImageTargets { boxes: vec![Xyxy::new(0.0, 0.0, ax, 1.0)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.5,
            rpiou_kappa: 1e38,
            rpiou_delta: 0.1,
            box_gain: 1e25,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let (bd, cd) = (
            box_raw.to_device(dev).unwrap().requires_grad_(),
            cls_raw.to_device(dev).unwrap().requires_grad_(),
        );
        let (bh, ch) = (box_raw.requires_grad_(), cls_raw.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &targets, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        let t_w = want.target_score_sum as f64;
        assert!(got.is_finite() && want.is_finite() && t_w > 0.9);
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()), "CPU logit grad must be finite: {g_cpu:?}");
        assert!(g_gpu.iter().all(|v| v.is_finite()), "GPU logit grad must be finite: {g_gpu:?}");
        let d2 = (0.1f32 as f64).powi(2);
        let ke2 = 1e38f64 * 0.25 * (d_left as f64).powi(2) / ((ax as f64).powi(2) + d2);
        let coord_g = 0.5f64 * (t_w * 1e25f64) * 1e38f64 * (d_left as f64)
            / (((ax as f64).powi(2) + d2) * (1.0 + ke2).powi(2));
        let logit_g_ref = -(coord_g * d_left as f64) as f32;
        assert!(
            (g_cpu[1] - logit_g_ref).abs() <= 1e-3 * logit_g_ref.abs(),
            "CPU logit grad {} vs f64 ref {logit_g_ref}",
            g_cpu[1]
        );
        assert!(
            (g_gpu[1] - logit_g_ref).abs() <= 2e-2 * logit_g_ref.abs(),
            "GPU logit grad {} vs f64 ref {logit_g_ref}",
            g_gpu[1]
        );
        close_vec(&g_gpu, &g_cpu, 2e-2 * logit_g_ref.abs(), 2e-2, "extreme_box_gain_rpiou/box_raw");
    }

    // 11. Small excluded OT-DFL mass (`reg_max = 4`, target distance `1.5`, logits `[-20, 0, 0, -1000]` on all
    //     sides, `dfl_ot_weight = 1e8`, `dfl_gain = 1.0`): naive `ln(Z) - ln(S)` rounds both sums to `2.0` and
    //     drops the barrier to `0`; `softplus(log_excl - log_ret)` preserves `~0.10306` on both CPU and GPU.
    {
        let anchors = Anchors { points: vec![[3.5, 3.5]], strides: vec![8.0], ..Anchors::default() };
        let mut bd_v = vec![0.0f32; 16];
        for s in 0..4 {
            bd_v[s * 4] = -20.0;
            bd_v[s * 4 + 3] = -1000.0;
        }
        let bx = Tensor::from_slice(&bd_v, [1, 16, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(16.0, 16.0, 40.0, 40.0)], labels: vec![0] }];
        let cfg0 =
            LossConfig { reg_max: 4, box_gain: 0.0, cls_gain: 0.0, dfl_gain: 1.0, ..LossConfig::default() };
        let cfg1 = LossConfig { dfl_ot_weight: 1e8, ..cfg0 };
        let b0 = detection_loss_host(&bx, &cx, &anchors, &tgts, &cfg0).unwrap();
        let (bd, cd) =
            (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
        let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg1).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg1).unwrap();
        let ot_ref = (1e8f64 * ((-20.0f64 - 2.0f64.ln()).exp().ln_1p())) as f32; // ~0.1030577
        assert!(
            ((want.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "CPU small-excl OT {} vs {ot_ref}",
            want.total_value - b0.total_value
        );
        assert!(
            ((got.total_value - b0.total_value) - ot_ref).abs() <= 1e-4,
            "GPU small-excl OT {} vs {ot_ref}",
            got.total_value - b0.total_value
        );
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        close_vec(&g_gpu, &g_cpu, 1e-4, 1e-3, "small_excl_otdfl/box_raw");
    }

    // 12. Extreme `box_gain = 2e38` on R-PIoU focal-IoU (`0.25x0.25` centered inside `0.5x0.5`, `beta = 0`,
    //     `kappa = 1e-30`): loss is `~1.5e38` and per-bin logit grad is `~±2.19e37` (finite on CPU and GPU).
    {
        let anchors = Anchors { points: vec![[0.5, 0.5]], strides: vec![1.0], ..Anchors::default() };
        let l1 = (1.0f32 / 7.0).ln(); // softmax([0, l1])[1] = 0.125 -> 0.25x0.25 box centered at (0.5, 0.5)
        let bx = Tensor::from_slice(&[0.0, l1, 0.0, l1, 0.0, l1, 0.0, l1], [1, 8, 1]).unwrap();
        let cx = Tensor::zeros([1, 1, 1]);
        let tgts = vec![ImageTargets { boxes: vec![Xyxy::new(0.25, 0.25, 0.75, 0.75)], labels: vec![0] }];
        let cfg = LossConfig {
            reg_max: 2,
            box_loss: BoxLossKind::Rpiou,
            rpiou_beta: 0.0,
            rpiou_kappa: 1e-30,
            box_gain: 2e38,
            cls_gain: 0.0,
            dfl_gain: 0.0,
            tal: folo_vision::assign::TalConfig {
                nwd_weight: 1.0,
                ..folo_vision::assign::TalConfig::default()
            },
            ..LossConfig::default()
        };
        let (bd, cd) =
            (bx.to_device(dev).unwrap().requires_grad_(), cx.to_device(dev).unwrap().requires_grad_());
        let (bh, ch) = (bx.requires_grad_(), cx.requires_grad_());
        let want = detection_loss_host(&bh, &ch, &anchors, &tgts, &cfg).unwrap();
        let got = detection_loss(&bd, &cd, &anchors, &tgts, &cfg).unwrap();
        assert!(got.is_finite() && want.is_finite());
        let g_gpu = got.total.backward().unwrap().get(&bd).unwrap().to_vec::<f32>().unwrap();
        let g_cpu = want.total.backward().unwrap().get(&bh).unwrap().to_vec::<f32>().unwrap();
        assert!(g_cpu.iter().all(|v| v.is_finite()) && g_gpu.iter().all(|v| v.is_finite()));
        let scale = g_cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e36, "expected ~2.19e37 finite focal-IoU logit grad, got {scale}");
        close_vec(&g_gpu, &g_cpu, 1e-3 * scale, 2e-3, "extreme_focal_iou_box_gain/box_raw");
    }
}
