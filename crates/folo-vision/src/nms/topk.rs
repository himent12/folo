//! One-to-one (NMS-free) top-k post-processing (docs/MODEL.md §6.6).

use crate::boxes::Xyxy;
use crate::nms::{Candidate, Detection, NmsConfig, class_allowed};

/// Ranks `(score, key)` pairs by descending score, ties by ascending key (NaN never reaches here: it fails every
/// `> conf` test).
fn by_score<K: Ord>(a: &(f32, K), b: &(f32, K)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// The `k` best of `v` by [`by_score`], in that order.
fn top_sorted<K: Ord + Copy>(v: &mut Vec<(f32, K)>, k: usize) {
    if v.len() > k {
        if k == 0 {
            v.clear();
            return;
        }
        v.select_nth_unstable_by(k - 1, by_score);
        v.truncate(k);
    }
    v.sort_unstable_by(by_score);
}

/// The post-process of the one-to-one (NMS-free) head for one image (docs/MODEL.md §6.6): with `k =
/// min(max_det, A)`, the `k` anchors with the highest max-class score (ties: lower anchor), then
///
/// - `multi_label` (validation): the `k` best (anchor, class) pairs among those anchors (ties: the anchor's rank,
///   then the class), each kept if its score is `> conf`;
/// - single-label (prediction): each anchor once, with its best class (first maximum), kept if `> conf`.
///
/// Then the class filter. No NMS and no candidate cap (`iou`, `max_candidates` and `agnostic` are unused).
/// Returns at most `max_det` detections sorted by descending score. Anchors whose best score is not `> conf` can
/// contribute no pair above it, so they are dropped before ranking: the result is the spec's.
///
/// # Panics
/// If `scores.len() != boxes.len() * num_classes`, or there are more than `u32::MAX` anchors.
pub fn topk_detections(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    assert_eq!(scores.len(), boxes.len() * num_classes, "topk_detections: scores must be [A, num_classes]");
    assert!(boxes.len() <= u32::MAX as usize, "topk_detections: too many anchors");
    if num_classes == 0 || boxes.is_empty() {
        return Vec::new();
    }
    let k = cfg.max_det.min(boxes.len());
    // (best score, (anchor, best class)): the class is a function of the anchor, so ties rank by anchor.
    let mut anchors: Vec<(f32, (u32, u32))> = Vec::new();
    for (a, row) in scores.chunks_exact(num_classes).enumerate() {
        let (mut bc, mut bs) = (0usize, row[0]);
        for (c, &s) in row.iter().enumerate().skip(1) {
            if s > bs {
                (bc, bs) = (c, s);
            }
        }
        if bs > cfg.conf {
            anchors.push((bs, (a as u32, bc as u32)));
        }
    }
    top_sorted(&mut anchors, k);
    let dets: Vec<Detection> = if cfg.multi_label {
        let mut pairs: Vec<(f32, (u32, u32))> = Vec::new();
        for (rank, &(_, (a, _))) in anchors.iter().enumerate() {
            let row = &scores[a as usize * num_classes..(a as usize + 1) * num_classes];
            for (c, &s) in row.iter().enumerate() {
                if s > cfg.conf {
                    pairs.push((s, (rank as u32, c as u32)));
                }
            }
        }
        top_sorted(&mut pairs, k);
        pairs
            .into_iter()
            .map(|(s, (r, c))| Detection {
                bbox: boxes[anchors[r as usize].1.0 as usize],
                score: s,
                class: c,
            })
            .collect()
    } else {
        anchors.iter().map(|&(s, (a, c))| Detection { bbox: boxes[a as usize], score: s, class: c }).collect()
    };
    dets.into_iter().filter(|d| class_allowed(cfg, d.class)).collect()
}

/// [`topk_detections`] single-label from the candidates already extracted (one per anchor whose best class scores
/// `> conf`, any order; folo-core's `Tensor::detect_candidates`): the same detections.
pub fn topk_candidates(cands: &[Candidate], cfg: &NmsConfig) -> Vec<Detection> {
    assert!(cands.len() <= u32::MAX as usize, "topk_candidates: too many candidates");
    // (score, (anchor, position)): ties rank by anchor, as the full selection does.
    let mut ranked: Vec<(f32, (u32, u32))> = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score > cfg.conf)
        .map(|(i, c)| (c.score, (c.anchor, i as u32)))
        .collect();
    top_sorted(&mut ranked, cfg.max_det);
    ranked
        .into_iter()
        .map(|(_, (_, i))| cands[i as usize])
        .map(|c| Detection { bbox: c.bbox, score: c.score, class: c.class })
        .filter(|d| class_allowed(cfg, d.class))
        .collect()
}
