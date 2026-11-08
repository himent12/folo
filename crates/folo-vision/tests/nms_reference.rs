//! NMS against a naive O(n²) greedy reference on seeded random boxes.

mod common;

use common::Rng;
use folo_vision::boxes::box_iou;
use folo_vision::nms::{batched_nms, nms_detections, non_max_suppression};
use folo_vision::{Detection, NmsConfig, Xyxy};

/// Textbook greedy NMS: visit by (score desc, index asc), keep a box unless a kept box of the same class (any class
/// when agnostic) overlaps it with IoU > thr; then the first `max_det`.
fn naive(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    thr: f32,
    max_det: usize,
    agnostic: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        let suppressed = keep
            .iter()
            .any(|&k| (agnostic || classes[k] == classes[i]) && box_iou(&boxes[k], &boxes[i]) > thr);
        if !suppressed {
            keep.push(i);
        }
    }
    keep.truncate(max_det);
    keep
}

fn random_case(
    rng: &mut Rng,
    n: usize,
    num_classes: usize,
    quantise: bool,
) -> (Vec<Xyxy>, Vec<f32>, Vec<u32>) {
    let boxes: Vec<Xyxy> = (0..n).map(|_| rng.bbox(640.0, 4.0, 160.0)).collect();
    let scores: Vec<f32> = (0..n)
        .map(|_| {
            let s = rng.f32();
            // Coarse scores force many ties, exercising the tie-break.
            if quantise { (s * 20.0).floor() / 20.0 } else { s }
        })
        .collect();
    let classes: Vec<u32> = (0..n).map(|_| rng.below(num_classes) as u32).collect();
    (boxes, scores, classes)
}

#[test]
fn batched_nms_matches_naive() {
    let mut rng = Rng::new(0xF010);
    for case in 0..300 {
        let n = match case % 4 {
            0 => rng.below(8),
            1 => rng.below(64),
            2 => rng.below(600),
            _ => rng.below(2500),
        };
        let nc = 1 + rng.below(6);
        let (boxes, scores, classes) = random_case(&mut rng, n, nc, case % 3 == 0);
        let thr = [0.0, 0.3, 0.45, 0.5, 0.7, 0.9, 1.0][case % 7];
        let max_det = [1, 5, 50, 300, usize::MAX][case % 5];
        let agnostic = case % 6 == 1;
        let got = batched_nms(&boxes, &scores, &classes, thr, max_det, agnostic);
        let want = naive(&boxes, &scores, &classes, thr, max_det, agnostic);
        assert_eq!(got, want, "case {case}: n={n} nc={nc} thr={thr} max_det={max_det} agnostic={agnostic}");
    }
}

#[test]
fn large_bucket_parallel_suppression_matches_naive() {
    // One class with > 16k boxes runs the threaded suppression pass.
    let mut rng = Rng::new(7);
    let (boxes, scores, _) = random_case(&mut rng, 20_000, 1, false);
    let classes = vec![0u32; boxes.len()];
    for (thr, max_det) in [(0.5, 300), (0.7, usize::MAX)] {
        let got = batched_nms(&boxes, &scores, &classes, thr, max_det, false);
        let want = naive(&boxes, &scores, &classes, thr, max_det, false);
        assert_eq!(got, want);
    }
}

/// Reference for the full post-process built from the naive NMS.
fn naive_post(boxes: &[Xyxy], scores: &[f32], nc: usize, cfg: &NmsConfig) -> Vec<Detection> {
    let mut cands = Vec::new();
    for (a, row) in scores.chunks(nc).enumerate() {
        if cfg.multi_label {
            for (c, &s) in row.iter().enumerate() {
                if s > cfg.conf && cfg.classes.as_ref().is_none_or(|cs| cs.contains(&(c as u32))) {
                    cands.push(Detection { bbox: boxes[a], score: s, class: c as u32 });
                }
            }
        } else {
            let mut best = 0;
            for c in 1..nc {
                if row[c] > row[best] {
                    best = c;
                }
            }
            if row[best] > cfg.conf && cfg.classes.as_ref().is_none_or(|cs| cs.contains(&(best as u32))) {
                cands.push(Detection { bbox: boxes[a], score: row[best], class: best as u32 });
            }
        }
    }
    // Cap: the best `max_candidates` by (score desc, position asc), kept in position order.
    let mut idx: Vec<usize> = (0..cands.len()).collect();
    idx.sort_by(|&a, &b| cands[b].score.total_cmp(&cands[a].score).then(a.cmp(&b)));
    idx.truncate(cfg.max_candidates);
    idx.sort();
    let cands: Vec<Detection> = idx.into_iter().map(|i| cands[i]).collect();
    let b: Vec<Xyxy> = cands.iter().map(|d| d.bbox).collect();
    let s: Vec<f32> = cands.iter().map(|d| d.score).collect();
    let c: Vec<u32> = cands.iter().map(|d| d.class).collect();
    naive(&b, &s, &c, cfg.iou, cfg.max_det, cfg.agnostic).into_iter().map(|i| cands[i]).collect()
}

#[test]
fn post_process_matches_naive() {
    let mut rng = Rng::new(42);
    for case in 0..40 {
        let a = [10, 300, 2000, 5000][case % 4];
        let nc = [1, 3, 20][case % 3];
        let boxes: Vec<Xyxy> = (0..a).map(|_| rng.bbox(640.0, 4.0, 200.0)).collect();
        let scores: Vec<f32> = (0..a * nc).map(|_| rng.f32().powi(4)).collect();
        let cfg = NmsConfig {
            conf: [0.001, 0.05, 0.25][case % 3],
            iou: [0.45, 0.7][case % 2],
            max_det: [300, 20][case % 2],
            max_candidates: [30_000, 500][(case / 2) % 2],
            classes: if case % 5 == 0 { Some(vec![0, 2]) } else { None },
            agnostic: case % 7 == 3,
            multi_label: case % 2 == 1,
        };
        let got = non_max_suppression(&boxes, &scores, nc, &cfg);
        let want = naive_post(&boxes, &scores, nc, &cfg);
        assert_eq!(got, want, "case {case}: a={a} nc={nc} cfg={cfg:?}");
        // The list API agrees when fed the multi-label candidates.
        if cfg.multi_label {
            let all: Vec<Detection> = scores
                .chunks(nc)
                .enumerate()
                .flat_map(|(i, row)| {
                    let bbox = boxes[i];
                    row.iter().enumerate().map(move |(c, &s)| Detection { bbox, score: s, class: c as u32 })
                })
                .collect();
            assert_eq!(nms_detections(&all, &cfg), want, "case {case}: nms_detections");
        }
    }
}
