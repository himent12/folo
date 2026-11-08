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
/// to float rounding, on several sizes and class counts.
#[test]
fn assigner_matches_host() {
    let dev = gpu!();
    for (seed, size, nc) in [(1, 96, 3), (2, 160, 5), (3, 128, 1), (4, 224, 12)] {
        let (anchors, box_raw, cls_raw, targets) = case(seed, size, nc);
        let cfg = LossConfig::default();
        let want = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
        let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
        let got = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
        assert_same_assignment(&got, &want);
    }
}

/// Other assigner settings take the kernels' general paths (pow for non-integer beta and alpha != 0.5, a topk
/// above 16) and still agree.
#[test]
fn assigner_matches_host_with_other_settings() {
    let dev = gpu!();
    let (anchors, box_raw, cls_raw, targets) = case(5, 160, 4);
    let mut cfg = LossConfig::default();
    cfg.tal.topk = 24;
    cfg.tal.alpha = 1.0;
    cfg.tal.beta = 4.5;
    let want = assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    let (bd, cd) = (box_raw.to_device(dev).unwrap(), cls_raw.to_device(dev).unwrap());
    let got = assign_batch_device(&bd, &cd, &anchors, &targets, &cfg).unwrap();
    assert_same_assignment(&got, &want);
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
    for (seed, size, nc) in [(11, 96, 3), (12, 160, 7)] {
        let (anchors, box_raw, cls_raw, targets) = case(seed, size, nc);
        let cfg = LossConfig::default();
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
