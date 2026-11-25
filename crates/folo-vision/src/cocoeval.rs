//! The full pycocotools `COCOeval` (bbox): crowd regions, the small / medium / large area buckets and the
//! `maxDets = 1 / 10 / 100` recalls, i.e. the twelve numbers `COCOeval.summarize()` prints.
//!
//! [`crate::metrics::Evaluator`] is the training-time evaluator (area range "all", no crowd boxes, plus precision /
//! recall / F1, curves and a confusion matrix). [`CocoEval`] is for reporting on a real COCO-format split: ground
//! truths carry the annotation's `area` (the segmentation area for COCO instances, which is what pycocotools buckets
//! by) and `iscrowd`, and every box is in original-image pixels. It is a line-by-line port of
//! `COCOeval.evaluateImg` / `accumulate` / `summarize`:
//!
//! - Per image and class, detections in descending score order (stable), capped at 100. A ground truth is ignored
//!   for an area range when it is a crowd or its `area` lies outside the range (bounds inclusive).
//! - IoU is pycocotools' `bbIou` in f64: `inter / union`, or `inter / area(det)` against a crowd.
//! - For each IoU threshold, each detection takes the unmatched truth with the highest IoU `>= t` (a crowd can be
//!   matched any number of times); once it holds a regular truth it stops at the ignored ones (sorted last). A
//!   detection matched to an ignored truth, or unmatched with its own `w·h` outside the range, is ignored.
//! - Per class and area range, detections of all images are ranked by score (ties: image key, then rank in the
//!   image) and AP is the 101-point interpolated precision; `maxDets = m` keeps each image's first `m` detections
//!   of the class. Recall is the final recall; classes without a regular truth are left out of every mean.
//!
//! `tests/coco_full.rs` checks all twelve numbers against pycocotools.

use crate::boxes::Xyxy;
use crate::metrics::{NUM_IOU_THRESHOLDS, NUM_RECALL_POINTS, desc_f32_key, iou_thresholds, recall_points};
use crate::nms::Detection;
use rayon::prelude::*;
use serde::Serialize;

/// pycocotools' area ranges, `[lo, hi]` inclusive: all, small, medium, large.
pub const AREA_RANGES: [(&str, f64, f64); 4] =
    [("all", 0.0, 1e10), ("small", 0.0, 1024.0), ("medium", 1024.0, 9216.0), ("large", 9216.0, 1e10)];
/// pycocotools' `maxDets`.
pub const MAX_DETS: [usize; 3] = [1, 10, 100];
const NA: usize = AREA_RANGES.len();
const ALL_BITS: u16 = (1 << NUM_IOU_THRESHOLDS) - 1;

/// One ground-truth annotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CocoTruth {
    /// The box as COCO stores it, `[x, y, w, h]` in f64 (the annotation file's numbers, not rounded to f32, so IoUs
    /// near a threshold come out exactly as pycocotools computes them), in the same coordinates as the detections.
    pub xywh: [f64; 4],
    /// Class index.
    pub class: u32,
    /// The annotation's `area` (COCO instances: the segmentation area), used for the area buckets.
    pub area: f64,
    /// A crowd region (`iscrowd: 1`): never counted, and detections on it are neither TPs nor FPs.
    pub iscrowd: bool,
}

impl CocoTruth {
    /// A regular (non-crowd) truth from an `xyxy` box, whose area is its box area.
    pub fn from_box(b: Xyxy, class: u32) -> Self {
        let (x, y) = (b.x1 as f64, b.y1 as f64);
        let (w, h) = (b.x2 as f64 - x, b.y2 as f64 - y);
        CocoTruth { xywh: [x, y, w, h], class, area: w * h, iscrowd: false }
    }
}

/// `w * h` in f64 as pycocotools computes it from `xywh`.
#[inline]
fn box_area(b: &Xyxy) -> f64 {
    (b.x2 as f64 - b.x1 as f64) * (b.y2 as f64 - b.y1 as f64)
}

#[inline]
fn in_range(area: f64, a: usize) -> bool {
    let (_, lo, hi) = AREA_RANGES[a];
    // pycocotools ignores when `area < lo or area > hi`, so a NaN area is in every range.
    !(area < lo || area > hi)
}

/// One detection as stored for `accumulate`: per area range, whether it is a TP / ignored at each IoU threshold.
#[derive(Clone, Copy, Debug)]
struct Rec {
    score: f32,
    /// Rank within its (image, class), `0..100`.
    rank: u16,
    image: u64,
    tp: [u16; NA],
    ig: [u16; NA],
}

/// Accumulates images; [`CocoEval::compute`] returns the summary. See the module docs.
#[derive(Clone, Debug)]
pub struct CocoEval {
    num_classes: usize,
    thresholds: [f64; NUM_IOU_THRESHOLDS],
    max_dets: usize,
    num_images: u64,
    num_dets: u64,
    num_crowd: u64,
    /// Per class: detections in insertion order.
    recs: Vec<Vec<Rec>>,
    /// Per class and area range: regular (not ignored) truths.
    npig: Vec<[u64; NA]>,
    // Scratch buffers reused across images and classes.
    d_order: Vec<(u64, u32)>,
    g_order: Vec<(u32, u32)>,
    ious: Vec<f64>,
    gt_order: Vec<usize>,
    gt_ignored: Vec<bool>,
    gt_matched: Vec<bool>,
}

/// Per-class AP (area "all", `maxDets = 100`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CocoClassStats {
    /// Class index.
    pub class: u32,
    /// Regular (non-crowd) truths.
    pub truths: u64,
    /// AP at IoU 0.50:0.05:0.95.
    pub ap: f64,
    /// AP at IoU 0.5.
    pub ap50: f64,
}

/// `COCOeval.summarize()`'s twelve numbers (`-1` when nothing qualifies, as in pycocotools), plus per-class AP.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CocoStats {
    /// AP @[IoU=0.50:0.95 | area=all | maxDets=100] (COCO's "mAP").
    pub ap: f64,
    /// AP @[IoU=0.50 | area=all | maxDets=100].
    pub ap50: f64,
    /// AP @[IoU=0.75 | area=all | maxDets=100].
    pub ap75: f64,
    /// AP @[IoU=0.50:0.95 | area=small | maxDets=100] (area < 32²).
    pub ap_small: f64,
    /// AP @[IoU=0.50:0.95 | area=medium | maxDets=100] (32² ≤ area ≤ 96²).
    pub ap_medium: f64,
    /// AP @[IoU=0.50:0.95 | area=large | maxDets=100] (area > 96²).
    pub ap_large: f64,
    /// AR @[IoU=0.50:0.95 | area=all | maxDets=1].
    pub ar1: f64,
    /// AR @[IoU=0.50:0.95 | area=all | maxDets=10].
    pub ar10: f64,
    /// AR @[IoU=0.50:0.95 | area=all | maxDets=100].
    pub ar100: f64,
    /// AR @[IoU=0.50:0.95 | area=small | maxDets=100].
    pub ar_small: f64,
    /// AR @[IoU=0.50:0.95 | area=medium | maxDets=100].
    pub ar_medium: f64,
    /// AR @[IoU=0.50:0.95 | area=large | maxDets=100].
    pub ar_large: f64,
    /// Images evaluated.
    pub images: u64,
    /// Regular truths (area "all").
    pub truths: u64,
    /// Crowd regions.
    pub crowd: u64,
    /// Detections that entered the evaluation (after the per-class cap of 100 per image).
    pub detections: u64,
    /// Classes with at least one regular truth.
    pub per_class: Vec<CocoClassStats>,
}

impl CocoStats {
    /// The twelve numbers in `COCOeval.stats` order.
    pub fn as_array(&self) -> [f64; 12] {
        [
            self.ap,
            self.ap50,
            self.ap75,
            self.ap_small,
            self.ap_medium,
            self.ap_large,
            self.ar1,
            self.ar10,
            self.ar100,
            self.ar_small,
            self.ar_medium,
            self.ar_large,
        ]
    }

    /// `COCOeval.summarize()`'s printout.
    pub fn summary(&self) -> String {
        let rows = [
            ("Average Precision", "0.50:0.95", "all", 100),
            ("Average Precision", "0.50     ", "all", 100),
            ("Average Precision", "0.75     ", "all", 100),
            ("Average Precision", "0.50:0.95", "small", 100),
            ("Average Precision", "0.50:0.95", "medium", 100),
            ("Average Precision", "0.50:0.95", "large", 100),
            ("Average Recall   ", "0.50:0.95", "all", 1),
            ("Average Recall   ", "0.50:0.95", "all", 10),
            ("Average Recall   ", "0.50:0.95", "all", 100),
            ("Average Recall   ", "0.50:0.95", "small", 100),
            ("Average Recall   ", "0.50:0.95", "medium", 100),
            ("Average Recall   ", "0.50:0.95", "large", 100),
        ];
        let mut s = String::new();
        for ((title, iou, area, md), v) in rows.iter().zip(self.as_array()) {
            let kind = if title.starts_with("Average Precision") { "(AP)" } else { "(AR)" };
            s.push_str(&format!(
                " {title} {kind} @[ IoU={iou} | area={area:>6} | maxDets={md:>3} ] = {v:.3}\n"
            ));
        }
        s
    }
}

impl CocoEval {
    /// An evaluator for `num_classes` classes.
    pub fn new(num_classes: usize) -> Self {
        CocoEval {
            num_classes,
            thresholds: iou_thresholds(),
            max_dets: *MAX_DETS.last().unwrap(),
            num_images: 0,
            num_dets: 0,
            num_crowd: 0,
            recs: vec![Vec::new(); num_classes],
            npig: vec![[0; NA]; num_classes],
            d_order: Vec::new(),
            g_order: Vec::new(),
            ious: Vec::new(),
            gt_order: Vec::new(),
            gt_ignored: Vec::new(),
            gt_matched: Vec::new(),
        }
    }

    /// Number of images added so far.
    pub fn num_images(&self) -> u64 {
        self.num_images
    }

    /// Add one image. `image` orders detections with equal scores across images (pycocotools ranks them by
    /// ascending image id), so pass the COCO image id for exact parity; any distinct keys work otherwise.
    /// Detections with a NaN score are ignored.
    ///
    /// # Panics
    /// If a class index is `>= num_classes`.
    pub fn add_image(&mut self, image: u64, dets: &[Detection], truths: &[CocoTruth]) {
        let nc = self.num_classes;
        for d in dets {
            assert!((d.class as usize) < nc, "CocoEval: detection class {} >= num_classes {nc}", d.class);
        }
        for t in truths {
            assert!((t.class as usize) < nc, "CocoEval: truth class {} >= num_classes {nc}", t.class);
        }
        self.num_images += 1;
        for t in truths {
            if t.iscrowd {
                self.num_crowd += 1;
            } else {
                for (a, n) in self.npig[t.class as usize].iter_mut().enumerate() {
                    if in_range(t.area, a) {
                        *n += 1;
                    }
                }
            }
        }
        // Detections by (class, score desc, input order) and truths by (class, input order) over packed keys:
        // zero indirection and zero allocation.
        self.d_order.clear();
        for (i, d) in dets.iter().enumerate() {
            if !d.score.is_nan() {
                let key = ((d.class as u64) << 32) | desc_f32_key(d.score) as u64;
                self.d_order.push((key, i as u32));
            }
        }
        self.d_order.sort_unstable();
        self.g_order.clear();
        for (i, t) in truths.iter().enumerate() {
            self.g_order.push((t.class, i as u32));
        }
        self.g_order.sort_unstable();

        let (mut di, mut gi) = (0, 0);
        while di < self.d_order.len() {
            let class = (self.d_order[di].0 >> 32) as u32;
            let d_end = di + self.d_order[di..].partition_point(|&(k, _)| (k >> 32) as u32 == class);
            while gi < self.g_order.len() && self.g_order[gi].0 < class {
                gi += 1;
            }
            let g_end = gi + self.g_order[gi..].partition_point(|&(c, _)| c == class);
            let n_d = (d_end - di).min(self.max_dets);
            self.match_class(image, dets, truths, di, di + n_d, gi, g_end, class as usize);
            self.num_dets += n_d as u64;
            di = d_end;
            gi = g_end;
        }
    }

    /// `evaluateImg` for one (image, class) and every area range, writing directly into `self.recs[class]`.
    #[allow(clippy::too_many_arguments)]
    fn match_class(
        &mut self,
        image: u64,
        dets: &[Detection],
        truths: &[CocoTruth],
        d0: usize,
        d1: usize,
        g0: usize,
        g1: usize,
        class: usize,
    ) {
        let nd = d1 - d0;
        let ng = g1 - g0;
        let d_idx = &self.d_order[d0..d1];
        let g_idx = &self.g_order[g0..g1];
        let base = self.recs[class].len();
        self.recs[class].reserve(nd);
        if ng == 0 {
            for (k, &(_, di)) in d_idx.iter().enumerate() {
                let d = &dets[di as usize];
                let area = box_area(&d.bbox);
                let ig = std::array::from_fn(|a| if in_range(area, a) { 0 } else { ALL_BITS });
                self.recs[class].push(Rec { score: d.score, rank: k as u16, image, tp: [0; NA], ig });
            }
            return;
        }
        self.ious.clear();
        let mut max_iou = 0.0f64;
        let mut has_nan = false;
        for &(_, di) in d_idx {
            let db = &dets[di as usize].bbox;
            for &(_, gi) in g_idx {
                let g = &truths[gi as usize];
                let v = iou_crowd(db, &g.xywh, g.iscrowd);
                if v.is_nan() {
                    has_nan = true;
                } else if v > max_iou {
                    max_iou = v;
                }
                self.ious.push(v);
            }
        }
        for (k, &(_, di)) in d_idx.iter().enumerate() {
            self.recs[class].push(Rec {
                score: dets[di as usize].score,
                rank: k as u16,
                image,
                tp: [0; NA],
                ig: [0; NA],
            });
        }
        let recs = &mut self.recs[class][base..base + nd];
        self.gt_ignored.resize(ng, false);
        self.gt_matched.resize(ng, false);
        let can_match_any = has_nan || max_iou >= self.thresholds[0].min(1.0 - 1e-10);
        for a in 0..NA {
            if can_match_any {
                for (g, &(_, gi)) in g_idx.iter().enumerate() {
                    let t = &truths[gi as usize];
                    self.gt_ignored[g] = t.iscrowd || !in_range(t.area, a);
                }
                // Regular truths first, each group in input order (pycocotools' mergesort on `_ignore`).
                self.gt_order.clear();
                self.gt_order.extend((0..ng).filter(|&g| !self.gt_ignored[g]));
                self.gt_order.extend((0..ng).filter(|&g| self.gt_ignored[g]));
                for (t, &thr) in self.thresholds.iter().enumerate() {
                    let thr_cap = thr.min(1.0 - 1e-10);
                    if !has_nan && max_iou < thr_cap {
                        break;
                    }
                    self.gt_matched.fill(false);
                    for (d, rec) in recs.iter_mut().enumerate() {
                        let row = &self.ious[d * ng..(d + 1) * ng];
                        let mut best = thr_cap;
                        let mut m = usize::MAX;
                        for &g in &self.gt_order {
                            if self.gt_matched[g] && !truths[g_idx[g].1 as usize].iscrowd {
                                continue;
                            }
                            if m != usize::MAX && !self.gt_ignored[m] && self.gt_ignored[g] {
                                break;
                            }
                            let v = row[g];
                            if v < best {
                                continue;
                            }
                            best = v;
                            m = g;
                        }
                        if m != usize::MAX {
                            self.gt_matched[m] = true;
                            if self.gt_ignored[m] {
                                rec.ig[a] |= 1 << t;
                            } else {
                                rec.tp[a] |= 1 << t;
                            }
                        }
                    }
                }
            }
            for (rec, &(_, di)) in recs.iter_mut().zip(d_idx) {
                if !in_range(box_area(&dets[di as usize].bbox), a) {
                    rec.ig[a] |= !(rec.tp[a] | rec.ig[a]) & ALL_BITS;
                }
            }
        }
    }

    /// Fold another evaluator's images into this one.
    ///
    /// # Panics
    /// If the class counts differ.
    pub fn merge(&mut self, other: CocoEval) {
        assert_eq!(self.num_classes, other.num_classes, "CocoEval::merge: class count mismatch");
        self.num_images += other.num_images;
        self.num_dets += other.num_dets;
        self.num_crowd += other.num_crowd;
        for (a, b) in self.recs.iter_mut().zip(other.recs) {
            a.extend(b);
        }
        for (a, b) in self.npig.iter_mut().zip(other.npig) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }

    /// `accumulate` + `summarize`.
    pub fn compute(&self) -> CocoStats {
        let recall_pts = recall_points();
        // Per class: [area][maxDet] -> Some((mean precision per threshold, recall per threshold)) or None (-1).
        // Only `(a = 0, m = 0..3)` and `(a = 1..4, m = 2)` are reported by `CocoStats`.
        type Cell = Option<([f64; NUM_IOU_THRESHOLDS], [f64; NUM_IOU_THRESHOLDS])>;
        let per: Vec<[[Cell; MAX_DETS.len()]; NA]> = (0..self.num_classes)
            .into_par_iter()
            .map(|c| {
                let mut sorted = self.recs[c].clone();
                sorted.sort_by(|x, y| {
                    y.score.total_cmp(&x.score).then(x.image.cmp(&y.image)).then(x.rank.cmp(&y.rank))
                });
                std::array::from_fn(|a| {
                    std::array::from_fn(|m| {
                        let used = a == 0 || m == MAX_DETS.len() - 1;
                        let npig = self.npig[c][a];
                        (used && npig > 0).then(|| accumulate(&sorted, a, MAX_DETS[m], npig, &recall_pts))
                    })
                })
            })
            .collect();
        let mean = |f: &dyn Fn(&[[Cell; 3]; NA]) -> Option<f64>| -> f64 {
            let (sum, count) = per.iter().filter_map(f).fold((0.0f64, 0usize), |(s, n), v| (s + v, n + 1));
            if count == 0 { -1.0 } else { sum / count as f64 }
        };
        let avg = |x: &[f64; NUM_IOU_THRESHOLDS]| x.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64;
        let ap = |a: usize| mean(&|p| p[a][2].as_ref().map(|(pr, _)| avg(pr)));
        let ap_t = |t: usize| mean(&|p| p[0][2].as_ref().map(|(pr, _)| pr[t]));
        let ar = |a: usize, m: usize| mean(&|p| p[a][m].as_ref().map(|(_, rc)| avg(rc)));
        let per_class = per
            .iter()
            .enumerate()
            .filter_map(|(c, p)| {
                p[0][2].as_ref().map(|(pr, _)| CocoClassStats {
                    class: c as u32,
                    truths: self.npig[c][0],
                    ap: avg(pr),
                    ap50: pr[0],
                })
            })
            .collect();
        CocoStats {
            ap: ap(0),
            ap50: ap_t(0),
            ap75: ap_t(5),
            ap_small: ap(1),
            ap_medium: ap(2),
            ap_large: ap(3),
            ar1: ar(0, 0),
            ar10: ar(0, 1),
            ar100: ar(0, 2),
            ar_small: ar(1, 2),
            ar_medium: ar(2, 2),
            ar_large: ar(3, 2),
            images: self.num_images,
            truths: self.npig.iter().map(|n| n[0]).sum(),
            crowd: self.num_crowd,
            detections: self.num_dets,
            per_class,
        }
    }
}

/// AP per threshold (mean of the 101-point interpolated precision) and the final recall per threshold, for one
/// class and area range, over the detections ranked below `max_det` in their image.
///
/// Single pass over `sorted` across all 10 IoU thresholds with zero heap allocations: precision can only peak on
/// a TP, and `{ i : i >= j_m } == { i : bucket(i) >= m }`, so a 101-point suffix max over per-bucket maxima is
/// bit-for-bit identical to pycocotools' envelope + left-sided recall search.
fn accumulate(
    sorted: &[Rec],
    a: usize,
    max_det: usize,
    npig: u64,
    recall_pts: &[f64; NUM_RECALL_POINTS],
) -> ([f64; NUM_IOU_THRESHOLDS], [f64; NUM_IOU_THRESHOLDS]) {
    let n = npig as f64;
    let mut tp = [0u64; NUM_IOU_THRESHOLDS];
    let mut fp = [0u64; NUM_IOU_THRESHOLDS];
    let mut cur_b = [0usize; NUM_IOU_THRESHOLDS];
    let mut bucket_max = [[0.0f64; NUM_RECALL_POINTS]; NUM_IOU_THRESHOLDS];
    for r in sorted {
        if (r.rank as usize) >= max_det {
            continue;
        }
        let (tp_bits, ig_bits) = (r.tp[a], r.ig[a]);
        if tp_bits == 0 && ig_bits == 0 {
            for f in &mut fp {
                *f += 1;
            }
            continue;
        }
        for t in 0..NUM_IOU_THRESHOLDS {
            if (ig_bits >> t) & 1 == 0 {
                if (tp_bits >> t) & 1 == 1 {
                    tp[t] += 1;
                    let tp_f = tp[t] as f64;
                    let rc = tp_f / n;
                    let mut b = cur_b[t];
                    while b + 1 < NUM_RECALL_POINTS && rc >= recall_pts[b + 1] {
                        b += 1;
                    }
                    cur_b[t] = b;
                    let p = tp_f / (fp[t] as f64 + tp_f + f64::EPSILON);
                    if p > bucket_max[t][b] {
                        bucket_max[t][b] = p;
                    }
                } else {
                    fp[t] += 1;
                }
            }
        }
    }
    let mut ap = [0.0; NUM_IOU_THRESHOLDS];
    let mut recall = [0.0; NUM_IOU_THRESHOLDS];
    for t in 0..NUM_IOU_THRESHOLDS {
        recall[t] = tp[t] as f64 / n;
        let bm = &mut bucket_max[t];
        let mut running = 0.0f64;
        for v in bm.iter_mut().rev() {
            if *v > running {
                running = *v;
            } else {
                *v = running;
            }
        }
        ap[t] = bm.iter().sum::<f64>() / NUM_RECALL_POINTS as f64;
    }
    (ap, recall)
}

/// pycocotools' `bbIou` in f64 on `xywh` boxes, operation for operation (`x + w` as the right edge): against a
/// crowd region the union is the detection's area. A detection is `[x1, y1, x2 - x1, y2 - y1]` in f64, which is
/// what a COCO results file written by `folo val --save-json` holds.
#[inline]
fn iou_crowd(d: &Xyxy, g: &[f64; 4], crowd: bool) -> f64 {
    let (dx, dy) = (d.x1 as f64, d.y1 as f64);
    let (dw, dh) = (d.x2 as f64 - dx, d.y2 as f64 - dy);
    let [gx, gy, gw, gh] = *g;
    let w = (dw + dx).min(gw + gx) - dx.max(gx);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (dh + dy).min(gh + gy) - dy.max(gy);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let da = dw * dh;
    let union = if crowd { da } else { da + gw * gh - inter };
    inter / union
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
        Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
    }

    #[test]
    fn perfect_and_buckets() {
        let mut ev = CocoEval::new(1);
        // A small (20x20), a medium (50x50) and a large (200x200) truth, each found exactly.
        let boxes = [
            Xyxy::new(0.0, 0.0, 20.0, 20.0),
            Xyxy::new(100.0, 100.0, 150.0, 150.0),
            Xyxy::new(300.0, 300.0, 500.0, 500.0),
        ];
        let truths: Vec<CocoTruth> = boxes.iter().map(|&b| CocoTruth::from_box(b, 0)).collect();
        let dets: Vec<Detection> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| det(b.x1, b.y1, b.x2, b.y2, 0.9 - i as f32 * 0.1, 0))
            .collect();
        ev.add_image(1, &dets, &truths);
        let s = ev.compute();
        for v in [s.ap, s.ap50, s.ap75, s.ap_small, s.ap_medium, s.ap_large, s.ar10, s.ar100] {
            assert!((v - 1.0).abs() < 1e-12, "{s:?}");
        }
        // maxDets 1 recalls one of three.
        assert!((s.ar1 - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(s.truths, 3);
    }

    #[test]
    fn crowd_absorbs_detections() {
        // A crowd region with a regular truth inside it. Extra detections inside the crowd are ignored, not FPs.
        let truths = [
            CocoTruth { xywh: [0.0, 0.0, 400.0, 400.0], class: 0, area: 1e5, iscrowd: true },
            CocoTruth::from_box(Xyxy::new(10.0, 10.0, 60.0, 60.0), 0),
        ];
        let dets = [
            det(50.0, 50.0, 150.0, 150.0, 0.95, 0),  // inside the crowd: ignored
            det(10.0, 10.0, 60.0, 60.0, 0.9, 0),     // the regular truth
            det(200.0, 200.0, 300.0, 300.0, 0.8, 0), // inside the crowd again (crowds re-match)
        ];
        let mut ev = CocoEval::new(1);
        ev.add_image(1, &dets, &truths);
        let s = ev.compute();
        assert!((s.ap - 1.0).abs() < 1e-12, "{s:?}");
        assert_eq!((s.truths, s.crowd), (1, 1));
        // Without the crowd both extra detections are FPs, one of them ranked first.
        let mut ev = CocoEval::new(1);
        ev.add_image(1, &dets, &truths[1..]);
        assert!(ev.compute().ap < 0.6);
    }

    #[test]
    fn truth_geometry_stays_f64() {
        // pycocotools: IoU of [0, 0, 10, 20] with the truth [0, 0, 20.0000001, 20] is 0.4999999975 < 0.5, a FP.
        // Rounded to f32 the truth would be 20 wide and the IoU exactly 0.5, a TP.
        let truth =
            CocoTruth { xywh: [0.0, 0.0, 20.0000001, 20.0], class: 0, area: 400.000002, iscrowd: false };
        let mut ev = CocoEval::new(1);
        ev.add_image(1, &[det(0.0, 0.0, 10.0, 20.0, 0.9, 0)], &[truth]);
        assert_eq!(ev.compute().ap50, 0.0);
        let mut ev = CocoEval::new(1);
        ev.add_image(
            1,
            &[det(0.0, 0.0, 10.0, 20.0, 0.9, 0)],
            &[CocoTruth::from_box(Xyxy::new(0.0, 0.0, 20.0, 20.0), 0)],
        );
        assert!(ev.compute().ap50 > 0.999);
    }

    #[test]
    fn no_truths_is_minus_one() {
        let mut ev = CocoEval::new(2);
        ev.add_image(1, &[det(0.0, 0.0, 10.0, 10.0, 0.5, 1)], &[]);
        let s = ev.compute();
        assert!(s.as_array().iter().all(|&v| v == -1.0));
        assert!(s.per_class.is_empty());
    }

    #[test]
    fn merge_equals_sequential() {
        let t = |x: f32, c| CocoTruth::from_box(Xyxy::new(x, x, x + 40.0, x + 30.0), c);
        let images: Vec<(Vec<Detection>, Vec<CocoTruth>)> = (0..6)
            .map(|i| {
                let x = i as f32 * 7.0;
                (
                    vec![
                        det(x + 1.0, x, x + 41.0, x + 30.0, 0.3 + 0.1 * i as f32, i % 2),
                        det(x, x, x + 9.0, x + 9.0, 0.5, 0),
                    ],
                    vec![t(x, i % 2), t(x + 100.0, 1)],
                )
            })
            .collect();
        let mut seq = CocoEval::new(2);
        for (i, (d, g)) in images.iter().enumerate() {
            seq.add_image(i as u64, d, g);
        }
        let (mut a, mut b) = (CocoEval::new(2), CocoEval::new(2));
        for (i, (d, g)) in images.iter().enumerate().rev() {
            if i % 2 == 0 { a.add_image(i as u64, d, g) } else { b.add_image(i as u64, d, g) }
        }
        a.merge(b);
        assert_eq!(seq.compute(), a.compute());
    }
}
