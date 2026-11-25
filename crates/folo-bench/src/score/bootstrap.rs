//! Fast, bit-exact bootstrap confidence interval for `mAP@0.5:0.95`.
//!
//! Each image's per-class greedy matching (`tp` bitmask across the 10 IoU thresholds, capped at COCO's `maxDets = 100`)
//! depends only on that image, never on which other images appear in a bootstrap resample. Precomputing those compact
//! `(score, tp)` records once per image eliminates 200× redundant IoU matching, confusion-matrix updates, and F1
//! confidence sweeps.
//!
//! For every class with no score ties between detections of differing `tp` bitmasks (the standard case for scored
//! detectors), detections are also pre-sorted once globally by descending score: each resample merely accumulates
//! per-image multiplicities `mult[img]` and scans the pre-sorted slice with zero per-resample sorting. Classes with
//! differing-`tp` score ties (such as unscored contestants where all scores equal `1.0`) gather per-image records in
//! exact draw order so stable tie-breaking matches [`folo_vision::Evaluator`] bit for bit.

use folo_core::rng::Rng;
use folo_vision::boxes::Xyxy;
use folo_vision::metrics::{NUM_IOU_THRESHOLDS, NUM_RECALL_POINTS, iou_thresholds, recall_points};
use rayon::prelude::*;

use super::{BOOTSTRAP_RESAMPLES, BOOTSTRAP_STREAM, Prepared, percentile};
use crate::pixels::mix;

/// COCO `maxDets` per image and class, matching [`folo_vision::EvalConfig::default`].
const MAX_DETS_PER_CLASS: usize = 100;

/// One pre-matched detection of a class in a specific image (during construction).
#[derive(Clone, Copy)]
struct PreDet {
    score: f32,
    tp: u16,
    img: u32,
}

/// Bitmask covering the `NUM_IOU_THRESHOLDS` (10) TP bits packed into a compact `TieFreeDets::Compact` word.
const TP_MASK: u32 = (1u32 << NUM_IOU_THRESHOLDS) - 1;
/// Maximum image count representable in the upper `32 - NUM_IOU_THRESHOLDS` bits of a compact `u32`.
const COMPACT_MAX_IMAGES: usize = 1usize << (32 - NUM_IOU_THRESHOLDS);

/// Compact pre-sorted detection for tie-free classes when `images.len() > COMPACT_MAX_IMAGES`.
#[derive(Clone, Copy)]
struct TieFreeDet {
    img: u32,
    tp: u16,
}

/// Pre-sorted tie-free detections (`score` is no longer needed once sorted, and trailing `tp == 0` entries are
/// trimmed because they can never affect any TP's cumulative weight).
enum TieFreeDets {
    /// `(img << NUM_IOU_THRESHOLDS) | tp` packed into 4 bytes per detection.
    Compact(Vec<u32>),
    /// Fallback when `images.len() > COMPACT_MAX_IMAGES`.
    Wide(Vec<TieFreeDet>),
}

/// Precomputed per-class data for bootstrap resampling.
enum ClassData {
    /// Every score tie in this class has identical `tp` bits: globally pre-sorted by descending score,
    /// so each resample runs a weighted linear scan without sorting.
    TieFree { gt_entries: Vec<(u32, u64)>, dets: TieFreeDets },
    /// Score ties exist with differing `tp` bits: per-image CSR slices preserve exact draw-order stability.
    WithTies {
        gt_entries: Vec<(u32, u64)>,
        /// For image `i`, its pre-matched detections (already in descending score / input order) live at
        /// `img_dets[img_offsets[i]..img_offsets[i + 1]]`.
        img_offsets: Vec<u32>,
        img_dets: Vec<(f32, u16)>,
        /// True when every detection in this class shares the exact same score (e.g. unscored `1.0`),
        /// so gathering in draw order is already stably sorted.
        all_same_score: bool,
    },
}

/// Per-image matching output produced once in parallel before resampling.
struct ImagePre {
    /// `(class, gt_count)` for classes with `gt_count > 0` in this image.
    gts: Vec<(u32, u64)>,
    /// `(class, score, tp_bits)` in descending score order per class (capped at `MAX_DETS_PER_CLASS`).
    dets: Vec<(u32, f32, u16)>,
}

/// Reusable per-worker scratch for `precompute_image`.
#[derive(Default)]
struct PreScratch {
    pred_order: Vec<u32>,
    gt_order: Vec<u32>,
    gt_areas: Vec<f64>,
    ious: Vec<f64>,
    tp_bits: Vec<u16>,
    gt_matched: Vec<bool>,
}

/// IoU in `f64` with precomputed areas, bit-identical to `folo_vision::metrics::iou_f64`.
#[inline]
fn iou_f64_with_areas(a: &Xyxy, a_area: f64, b: &Xyxy, b_area: f64) -> f64 {
    let w = (a.x2 as f64).min(b.x2 as f64) - (a.x1 as f64).max(b.x1 as f64);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (a.y2 as f64).min(b.y2 as f64) - (a.y1 as f64).max(b.y1 as f64);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let union = a_area + b_area - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[inline]
fn box_area_clamped(b: &Xyxy) -> f64 {
    (b.x2 as f64 - b.x1 as f64).max(0.0) * (b.y2 as f64 - b.y1 as f64).max(0.0)
}

/// Runs the exact per-image, per-class greedy matching of `Evaluator::add_image` (`EvalConfig::default()`).
fn precompute_image(p: &Prepared, thresholds: &[f64; NUM_IOU_THRESHOLDS], sc: &mut PreScratch) -> ImagePre {
    let preds = &p.dets;
    let truths = &p.truths;
    let mut gts = Vec::new();
    let mut dets = Vec::new();
    if truths.is_empty() && preds.is_empty() {
        return ImagePre { gts, dets };
    }

    sc.pred_order.clear();
    sc.pred_order.extend((0..preds.len() as u32).filter(|&i| !preds[i as usize].score.is_nan()));
    sc.pred_order.sort_by(|&a, &b| {
        let (pa, pb) = (&preds[a as usize], &preds[b as usize]);
        pa.class.cmp(&pb.class).then(pb.score.total_cmp(&pa.score))
    });

    sc.gt_order.clear();
    sc.gt_order.extend(0..truths.len() as u32);
    sc.gt_order.sort_by_key(|&i| truths[i as usize].class);

    // Record per-class ground truth counts.
    let mut gi = 0;
    while gi < sc.gt_order.len() {
        let class = truths[sc.gt_order[gi] as usize].class;
        let g_end = gi + sc.gt_order[gi..].partition_point(|&i| truths[i as usize].class == class);
        gts.push((class, (g_end - gi) as u64));
        gi = g_end;
    }

    // Match detections per class across all 10 IoU thresholds.
    dets.reserve(sc.pred_order.len());
    let (mut pi, mut gi) = (0, 0);
    while pi < sc.pred_order.len() {
        let class = preds[sc.pred_order[pi] as usize].class;
        let p_end = pi + sc.pred_order[pi..].partition_point(|&i| preds[i as usize].class == class);
        while gi < sc.gt_order.len() && truths[sc.gt_order[gi] as usize].class < class {
            gi += 1;
        }
        let g_end = gi + sc.gt_order[gi..].partition_point(|&i| truths[i as usize].class == class);
        let np = (p_end - pi).min(MAX_DETS_PER_CLASS);
        let ng = g_end - gi;

        sc.tp_bits.clear();
        sc.tp_bits.resize(np, 0u16);
        if ng > 0 {
            sc.gt_areas.clear();
            sc.gt_areas
                .extend(sc.gt_order[gi..g_end].iter().map(|&g| box_area_clamped(&truths[g as usize].bbox)));
            sc.ious.clear();
            sc.ious.reserve(np * ng);
            for &p_idx in &sc.pred_order[pi..pi + np] {
                let pb = &preds[p_idx as usize].bbox;
                let pa = box_area_clamped(pb);
                for (lg, &g_idx) in sc.gt_order[gi..g_end].iter().enumerate() {
                    sc.ious.push(iou_f64_with_areas(pb, pa, &truths[g_idx as usize].bbox, sc.gt_areas[lg]));
                }
            }
            for (t, &thr) in thresholds.iter().enumerate() {
                sc.gt_matched.clear();
                sc.gt_matched.resize(ng, false);
                let min_iou = thr.min(1.0 - 1e-10);
                for (d, tp) in sc.tp_bits.iter_mut().enumerate() {
                    let mut best = min_iou;
                    let mut m = usize::MAX;
                    let row = d * ng;
                    for (g, matched) in sc.gt_matched.iter().enumerate() {
                        if *matched {
                            continue;
                        }
                        let v = sc.ious[row + g];
                        if v < best {
                            continue;
                        }
                        best = v;
                        m = g;
                    }
                    if m != usize::MAX {
                        sc.gt_matched[m] = true;
                        *tp |= 1 << t;
                    }
                }
            }
        }
        for (k, &p_idx) in sc.pred_order[pi..pi + np].iter().enumerate() {
            dets.push((class, preds[p_idx as usize].score, sc.tp_bits[k]));
        }
        pi = p_end;
        gi = g_end;
    }

    ImagePre { gts, dets }
}

/// Reusable per-thread scratch for evaluating one bootstrap resample.
struct ResampleScratch {
    draws: Vec<u32>,
    mult: Vec<u32>,
    /// Sparse true-positive entries `(tp_bits, weight, cum_weight)` in descending score order.
    /// Detections with `tp_bits == 0` never raise recall or the precision envelope and only contribute to
    /// `cum_weight`.
    active: Vec<(u16, u64, u64)>,
    tied: Vec<(f32, u16)>,
    rc: Vec<f64>,
    pr: Vec<f64>,
}

impl ResampleScratch {
    fn new(n_images: usize) -> Self {
        ResampleScratch {
            draws: Vec::with_capacity(n_images),
            mult: vec![0u32; n_images],
            active: Vec::new(),
            tied: Vec::new(),
            rc: Vec::new(),
            pr: Vec::new(),
        }
    }
}

/// Computes `AP@0.5:0.95` over sparse true-positive entries `(tp_bits, weight, cum_weight)` already in descending
/// score order, where `any_tp` is the bitwise OR of all `tp_bits`.
///
/// Skipping false positives (`(tp_bits >> t) & 1 == 0`) is bit-identical to evaluating the full PR curve:
/// at any FP, `tp` and `rc` are unchanged and `pr = tp / (tp + fp + eps)` can only decrease, so an FP neither
/// raises the monotone precision envelope nor satisfies `rc[j] >= r` before the preceding TP (and leading FPs have
/// `pr = 0 <= pr[first_tp]`, so `env[0] == env[first_tp]`).
fn ap_sparse(
    active: &[(u16, u64, u64)],
    any_tp: u16,
    n_gt: u64,
    recall_pts: &[f64; NUM_RECALL_POINTS],
    rc: &mut Vec<f64>,
    pr: &mut Vec<f64>,
) -> f64 {
    if any_tp == 0 || active.is_empty() {
        return 0.0;
    }
    let n_gt_f = n_gt as f64;
    let mut ap_sum = 0.0f64;
    for t in 0..NUM_IOU_THRESHOLDS {
        if (any_tp >> t) & 1 == 0 {
            continue;
        }
        rc.clear();
        pr.clear();
        let mut tp = 0u64;
        for &(bits, w, cum_w) in active {
            if (bits >> t) & 1 != 0 {
                tp += w;
                let tp_f = tp as f64;
                let fp_f = (cum_w - tp) as f64;
                rc.push(tp_f / n_gt_f);
                pr.push(tp_f / (tp_f + fp_f + f64::EPSILON));
            }
        }
        let m = rc.len();
        if m == 0 {
            continue;
        }
        for i in (1..m).rev() {
            if pr[i] > pr[i - 1] {
                pr[i - 1] = pr[i];
            }
        }
        let mut q_sum = 0.0f64;
        let mut j = 0;
        for &r in recall_pts {
            while j < m && rc[j] < r {
                j += 1;
            }
            if j == m {
                break;
            }
            q_sum += pr[j];
        }
        ap_sum += q_sum / NUM_RECALL_POINTS as f64;
    }
    ap_sum / NUM_IOU_THRESHOLDS as f64
}

/// Builds per-class precomputed structures for all classes that have at least one ground truth in `images`.
fn build_class_data(images: &[Prepared], num_classes: usize) -> Vec<ClassData> {
    let n = images.len();
    let thresholds = iou_thresholds();
    let per_img: Vec<ImagePre> = images
        .par_iter()
        .map_init(PreScratch::default, |sc, p| precompute_image(p, &thresholds, sc))
        .collect();

    let mut gt_counts = vec![0usize; num_classes];
    let mut det_counts = vec![0usize; num_classes];
    for pre in &per_img {
        for &(c, _) in &pre.gts {
            gt_counts[c as usize] += 1;
        }
        for &(c, _, _) in &pre.dets {
            det_counts[c as usize] += 1;
        }
    }
    let mut gt_by_class: Vec<Vec<(u32, u64)>> = gt_counts.into_iter().map(Vec::with_capacity).collect();
    let mut det_by_class: Vec<Vec<PreDet>> = det_counts.into_iter().map(Vec::with_capacity).collect();
    for (img_idx, pre) in per_img.into_iter().enumerate() {
        let img = img_idx as u32;
        for (c, cnt) in pre.gts {
            gt_by_class[c as usize].push((img, cnt));
        }
        for (c, score, tp) in pre.dets {
            det_by_class[c as usize].push(PreDet { score, tp, img });
        }
    }

    gt_by_class
        .into_iter()
        .zip(det_by_class)
        .filter(|(gts, _)| !gts.is_empty())
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(gt_entries, mut raw_dets)| {
            raw_dets.sort_by(|a, b| b.score.total_cmp(&a.score));
            let tie_free = raw_dets
                .windows(2)
                .all(|w| w[0].score.to_bits() != w[1].score.to_bits() || w[0].tp == w[1].tp);
            if tie_free {
                // Trailing false positives after the last true positive never affect any TP's cumulative weight.
                let keep = raw_dets.iter().rposition(|d| d.tp != 0).map_or(0, |i| i + 1);
                raw_dets.truncate(keep);
                let dets = if n <= COMPACT_MAX_IMAGES {
                    TieFreeDets::Compact(
                        raw_dets
                            .into_iter()
                            .map(|d| (d.img << NUM_IOU_THRESHOLDS) | (d.tp as u32 & TP_MASK))
                            .collect(),
                    )
                } else {
                    TieFreeDets::Wide(
                        raw_dets.into_iter().map(|d| TieFreeDet { img: d.img, tp: d.tp }).collect(),
                    )
                };
                ClassData::TieFree { gt_entries, dets }
            } else {
                let all_same_score = raw_dets
                    .first()
                    .is_none_or(|first| raw_dets.iter().all(|d| d.score.to_bits() == first.score.to_bits()));
                // Stable sort by `img` restores exact per-image descending-score / input order without cloning.
                raw_dets.sort_by_key(|d| d.img);
                let mut img_offsets = vec![0u32; n + 1];
                let mut img_dets = Vec::with_capacity(raw_dets.len());
                let mut cur_img = 0usize;
                for d in raw_dets {
                    let img = d.img as usize;
                    while cur_img < img {
                        cur_img += 1;
                        img_offsets[cur_img] = img_dets.len() as u32;
                    }
                    img_dets.push((d.score, d.tp));
                }
                while cur_img < n {
                    cur_img += 1;
                    img_offsets[cur_img] = img_dets.len() as u32;
                }
                ClassData::WithTies { gt_entries, img_offsets, img_dets, all_same_score }
            }
        })
        .collect()
}

/// mAP@0.5:0.95's 95% interval around `measured`: [`BOOTSTRAP_RESAMPLES`] resamples of the images with
/// replacement, their 2.5th and 97.5th percentile (linear interpolation between order statistics, numpy's
/// default), shifted by the difference between `measured` and the resamples' mean. The shift matters when classes
/// are rare: a resample that happens to hold none of a hard class leaves it out of the class mean, so resamples
/// run higher than the measured value and their plain percentiles would sit beside it instead of around it. The
/// interval is clamped to `[0, 1]` and always holds `measured`. `[0, 0]` without images.
pub(super) fn bootstrap(images: &[Prepared], num_classes: usize, seed: u64, measured: f32) -> [f32; 2] {
    let n = images.len();
    if n == 0 || num_classes == 0 {
        return [0.0; 2];
    }
    let classes = build_class_data(images, num_classes);
    let needs_draws = classes.iter().any(|c| matches!(c, ClassData::WithTies { .. }));
    let recall_pts = recall_points();

    let mut maps: Vec<f32> = (0..BOOTSTRAP_RESAMPLES)
        .into_par_iter()
        .map_init(
            || ResampleScratch::new(n),
            |s, r| {
                let mut rng = Rng::seed(mix(seed ^ BOOTSTRAP_STREAM, r as u64));
                s.mult.fill(0);
                if needs_draws {
                    s.draws.clear();
                    for _ in 0..n {
                        let idx = rng.below(n as u64) as usize;
                        s.draws.push(idx as u32);
                        s.mult[idx] += 1;
                    }
                } else {
                    for _ in 0..n {
                        let idx = rng.below(n as u64) as usize;
                        s.mult[idx] += 1;
                    }
                }

                let mut ap_sum = 0.0f64;
                let mut n_eval = 0usize;
                for cls in &classes {
                    match cls {
                        ClassData::TieFree { gt_entries, dets } => {
                            let mut n_gt = 0u64;
                            for &(img, cnt) in gt_entries {
                                n_gt += cnt * s.mult[img as usize] as u64;
                            }
                            if n_gt == 0 {
                                continue;
                            }
                            n_eval += 1;
                            s.active.clear();
                            let mut cum_w = 0u64;
                            let mut any_tp = 0u16;
                            match dets {
                                TieFreeDets::Compact(packed) => {
                                    for &entry in packed {
                                        let w = s.mult[(entry >> NUM_IOU_THRESHOLDS) as usize] as u64;
                                        cum_w += w;
                                        let tp = (entry & TP_MASK) as u16;
                                        if tp != 0 && w != 0 {
                                            any_tp |= tp;
                                            s.active.push((tp, w, cum_w));
                                        }
                                    }
                                }
                                TieFreeDets::Wide(wide) => {
                                    for d in wide {
                                        let w = s.mult[d.img as usize] as u64;
                                        cum_w += w;
                                        if d.tp != 0 && w != 0 {
                                            any_tp |= d.tp;
                                            s.active.push((d.tp, w, cum_w));
                                        }
                                    }
                                }
                            }
                            ap_sum += ap_sparse(&s.active, any_tp, n_gt, &recall_pts, &mut s.rc, &mut s.pr);
                        }
                        ClassData::WithTies { gt_entries, img_offsets, img_dets, all_same_score } => {
                            let mut n_gt = 0u64;
                            for &(img, cnt) in gt_entries {
                                n_gt += cnt * s.mult[img as usize] as u64;
                            }
                            if n_gt == 0 {
                                continue;
                            }
                            n_eval += 1;
                            s.active.clear();
                            let mut cum_w = 0u64;
                            let mut any_tp = 0u16;
                            if *all_same_score {
                                for &img in &s.draws {
                                    let lo = img_offsets[img as usize] as usize;
                                    let hi = img_offsets[img as usize + 1] as usize;
                                    for &(_, tp) in &img_dets[lo..hi] {
                                        cum_w += 1;
                                        if tp != 0 {
                                            any_tp |= tp;
                                            s.active.push((tp, 1, cum_w));
                                        }
                                    }
                                }
                            } else {
                                s.tied.clear();
                                for &img in &s.draws {
                                    let lo = img_offsets[img as usize] as usize;
                                    let hi = img_offsets[img as usize + 1] as usize;
                                    if lo < hi {
                                        s.tied.extend_from_slice(&img_dets[lo..hi]);
                                    }
                                }
                                s.tied.sort_by(|a, b| b.0.total_cmp(&a.0));
                                for &(_, tp) in &s.tied {
                                    cum_w += 1;
                                    if tp != 0 {
                                        any_tp |= tp;
                                        s.active.push((tp, 1, cum_w));
                                    }
                                }
                            }
                            ap_sum += ap_sparse(&s.active, any_tp, n_gt, &recall_pts, &mut s.rc, &mut s.pr);
                        }
                    }
                }
                if n_eval == 0 { 0.0 } else { (ap_sum / n_eval as f64) as f32 }
            },
        )
        .collect();

    maps.sort_by(f32::total_cmp);
    // Summed in sorted order, in f64: the same for every thread count.
    let mean = maps.iter().map(|&m| m as f64).sum::<f64>() / maps.len() as f64;
    let shift = measured as f64 - mean;
    let at = |q: f64| (percentile(&maps, q) as f64 + shift).clamp(0.0, 1.0) as f32;
    [at(0.025).min(measured), at(0.975).max(measured)]
}
