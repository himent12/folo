//! COCO scale-bucketed AP/AR diagnostics (`AP_S`, `AP_M`, `AP_L`, `AR@1/10/100`) operating
//! directly on `(preds, truths)` image streams with pycocotools bucket-ignore semantics.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{NUM_IOU_THRESHOLDS, NUM_RECALL_POINTS, Truth, iou_f64, iou_thresholds, recall_points};
use crate::nms::Detection;

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Box area `(x2 - x1) * (y2 - y1)` in `f64` (matching pycocotools).
#[inline]
fn box_area_f64(b: &Xyxy) -> f64 {
    (b.x2 as f64 - b.x1 as f64) * (b.y2 as f64 - b.y1 as f64)
}

/// Area boundaries defining COCO's `small`, `medium`, and `large` object scale buckets.
///
/// Defaults match pycocotools:
/// - `all`: `[0, max_area]` (`max_area = 1e10 = 1e5Â²`)
/// - `small`: `[0, small_max]` (`32Â² = 1024`)
/// - `medium`: `[small_max, medium_max]` (`32Â²..=96Â² = 1024..=9216`)
/// - `large`: `[medium_max, max_area]` (`96Â²..=1e5Â²`)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScaleAreaBounds {
    /// Upper area bound of the `small` bucket (and lower bound of `medium`; default `1024.0`).
    pub small_max: f64,
    /// Upper area bound of the `medium` bucket (and lower bound of `large`; default `9216.0`).
    pub medium_max: f64,
    /// Upper area bound of the `all` and `large` buckets (default `1e10`).
    pub max_area: f64,
}

impl Default for ScaleAreaBounds {
    fn default() -> Self {
        Self { small_max: 1024.0, medium_max: 9216.0, max_area: 1e10 }
    }
}

impl ScaleAreaBounds {
    /// Validate `0 < small_max < medium_max < max_area`, all finite.
    pub fn validate(&self) -> Result<()> {
        if !(self.small_max.is_finite()
            && self.medium_max.is_finite()
            && self.max_area.is_finite()
            && self.small_max > 0.0
            && self.small_max < self.medium_max
            && self.medium_max < self.max_area)
        {
            return Err(Error::Config(format!(
                "ScaleAreaBounds: require finite 0 < small_max < medium_max < max_area, got \
                 ({}, {}, {})",
                self.small_max, self.medium_max, self.max_area
            )));
        }
        Ok(())
    }

    #[inline]
    fn ranges(&self) -> [(f64, f64); 4] {
        [
            (0.0, self.max_area),
            (0.0, self.small_max),
            (self.small_max, self.medium_max),
            (self.medium_max, self.max_area),
        ]
    }
}

/// Per-class scale-bucketed COCO AP summary (`None` when the class has zero truths in a bucket).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CocoScaleClassMetrics {
    /// Class index.
    pub class: u32,
    /// Ground-truth count in `all`.
    pub truths_all: u64,
    /// Ground-truth count in `small`.
    pub truths_small: u64,
    /// Ground-truth count in `medium`.
    pub truths_medium: u64,
    /// Ground-truth count in `large`.
    pub truths_large: u64,
    /// `AP@0.50:0.95` (all areas, `maxDets = 100`).
    pub ap: Option<f64>,
    /// `AP@0.50` (all areas, `maxDets = 100`).
    pub ap50: Option<f64>,
    /// `AP@0.75` (all areas, `maxDets = 100`).
    pub ap75: Option<f64>,
    /// `AP_S@0.50:0.95` (`small` bucket, `maxDets = 100`).
    pub ap_small: Option<f64>,
    /// `AP_M@0.50:0.95` (`medium` bucket, `maxDets = 100`).
    pub ap_medium: Option<f64>,
    /// `AP_L@0.50:0.95` (`large` bucket, `maxDets = 100`).
    pub ap_large: Option<f64>,
}

/// COCO scale-bucketed AP/AR diagnostics (`AP`, `AP50`, `AP75`, `AP_S`, `AP_M`, `AP_L`, `AR@1`,
/// `AR@10`, `AR@100`, `AR_S`, `AR_M`, `AR_L`) with per-class and per-bucket truth counts.
///
/// Any metric whose bucket has zero qualifying ground truths is `None` (unlike pycocotools' `-1.0`
/// sentinel, preventing accidental inclusion in downstream arithmetic).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CocoScaleDiagnostics {
    /// Area boundaries used for bucketing.
    pub bounds: ScaleAreaBounds,
    /// `AP @[IoU=0.50:0.95 | area=all | maxDets=100]`.
    pub ap: Option<f64>,
    /// `AP @[IoU=0.50 | area=all | maxDets=100]`.
    pub ap50: Option<f64>,
    /// `AP @[IoU=0.75 | area=all | maxDets=100]`.
    pub ap75: Option<f64>,
    /// `AP_S @[IoU=0.50:0.95 | area=small | maxDets=100]`.
    pub ap_small: Option<f64>,
    /// `AP_M @[IoU=0.50:0.95 | area=medium | maxDets=100]`.
    pub ap_medium: Option<f64>,
    /// `AP_L @[IoU=0.50:0.95 | area=large | maxDets=100]`.
    pub ap_large: Option<f64>,
    /// `AR@1 @[IoU=0.50:0.95 | area=all | maxDets=1]`.
    pub ar_1: Option<f64>,
    /// `AR@10 @[IoU=0.50:0.95 | area=all | maxDets=10]`.
    pub ar_10: Option<f64>,
    /// `AR@100 @[IoU=0.50:0.95 | area=all | maxDets=100]`.
    pub ar_100: Option<f64>,
    /// `AR_S @[IoU=0.50:0.95 | area=small | maxDets=100]`.
    pub ar_small: Option<f64>,
    /// `AR_M @[IoU=0.50:0.95 | area=medium | maxDets=100]`.
    pub ar_medium: Option<f64>,
    /// `AR_L @[IoU=0.50:0.95 | area=large | maxDets=100]`.
    pub ar_large: Option<f64>,
    /// Number of images evaluated.
    pub num_images: u64,
    /// Total non-ignored ground truths in `all`.
    pub truths_all: u64,
    /// Total non-ignored ground truths in `small`.
    pub truths_small: u64,
    /// Total non-ignored ground truths in `medium`.
    pub truths_medium: u64,
    /// Total non-ignored ground truths in `large`.
    pub truths_large: u64,
    /// Per-class scale metrics for classes `0..num_classes`.
    pub per_class: Vec<CocoScaleClassMetrics>,
}

const SCALE_MAX_DETS: [usize; 3] = [1, 10, 100];
const ALL_IOU_BITS: u16 = (1 << NUM_IOU_THRESHOLDS) - 1;

#[derive(Clone, Copy, Debug)]
struct ScaleDetRec {
    score: f32,
    image_idx: usize,
    rank: u16,
    tp: [u16; 4],
    ig: [u16; 4],
}

/// Evaluate exact COCO 10-IoU (`0.50:0.05:0.95`) 101-point AP and AR across `all`, `small`,
/// `medium`, and `large` area buckets and `maxDets = 1 / 10 / 100` directly from `(preds, truths)`
/// image pairs.
///
/// Follows pycocotools bucket-ignore rules: for each area bucket `[lo, hi]`, ground truths outside
/// `[lo, hi]` are marked ignored (sorted after regular truths during greedy matching so predictions
/// matching them are neither TP nor FP), and unmatched predictions outside `[lo, hi]` are also
/// ignored. Non-finite scores or boxes are skipped cleanly.
pub fn compute_coco_scale_metrics<P, T>(
    num_classes: usize,
    images: &[(P, T)],
    bounds: Option<ScaleAreaBounds>,
) -> Result<CocoScaleDiagnostics>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    let bounds = bounds.unwrap_or_default();
    bounds.validate()?;
    let ranges = bounds.ranges();
    let thresholds = iou_thresholds();
    let r_grid = recall_points();

    let mut per_class_recs: Vec<Vec<ScaleDetRec>> = vec![Vec::new(); num_classes];
    let mut gt_counts: Vec<[u64; 4]> = vec![[0u64; 4]; num_classes];

    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        let preds = preds_ref.as_ref();
        let truths = truths_ref.as_ref();
        for (i, p) in preds.iter().enumerate() {
            if (p.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_coco_scale_metrics: image {img_idx} pred[{i}].class = {} >= num_classes ({num_classes})",
                    p.class
                )));
            }
        }
        for (i, t) in truths.iter().enumerate() {
            if (t.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_coco_scale_metrics: image {img_idx} truth[{i}].class = {} >= num_classes ({num_classes})",
                    t.class
                )));
            }
        }

        let mut img_preds: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        for (pi, p) in preds.iter().enumerate() {
            if p.score.is_finite() && box_is_finite(&p.bbox) {
                img_preds[p.class as usize].push(pi);
            }
        }
        let mut img_gts: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        for (gi, g) in truths.iter().enumerate() {
            if !box_is_finite(&g.bbox) {
                continue;
            }
            let area = box_area_f64(&g.bbox);
            for (a, &(lo, hi)) in ranges.iter().enumerate() {
                if (lo..=hi).contains(&area) {
                    gt_counts[g.class as usize][a] += 1;
                }
            }
            img_gts[g.class as usize].push(gi);
        }

        for c in 0..num_classes {
            let p_list = &mut img_preds[c];
            if p_list.is_empty() {
                continue;
            }
            p_list
                .sort_unstable_by(|&a, &b| preds[b].score.total_cmp(&preds[a].score).then_with(|| a.cmp(&b)));
            p_list.truncate(100);
            let np = p_list.len();
            let g_list = &img_gts[c];
            let ng = g_list.len();

            let mut tp = vec![[0u16; 4]; np];
            let mut ig = vec![[0u16; 4]; np];

            if ng == 0 {
                for (d, &pi) in p_list.iter().enumerate() {
                    let area = box_area_f64(&preds[pi].bbox);
                    for (a, &(lo, hi)) in ranges.iter().enumerate() {
                        if !(lo..=hi).contains(&area) {
                            ig[d][a] = ALL_IOU_BITS;
                        }
                    }
                }
            } else {
                let mut ious = vec![0.0f64; np * ng];
                for (d, &pi) in p_list.iter().enumerate() {
                    let pb = &preds[pi].bbox;
                    for (g, &gi) in g_list.iter().enumerate() {
                        ious[d * ng + g] = iou_f64(pb, &truths[gi].bbox);
                    }
                }
                let gt_areas: Vec<f64> = g_list.iter().map(|&gi| box_area_f64(&truths[gi].bbox)).collect();
                let det_areas: Vec<f64> = p_list.iter().map(|&pi| box_area_f64(&preds[pi].bbox)).collect();

                for (a, &(lo, hi)) in ranges.iter().enumerate() {
                    // Regular GTs (inside area range) first, ignored GTs second (stable order).
                    let mut gt_ord: Vec<(usize, bool)> =
                        (0..ng).map(|g| (g, !(lo..=hi).contains(&gt_areas[g]))).collect();
                    gt_ord.sort_by_key(|&(_, ignored)| ignored);

                    for (t, &thr) in thresholds.iter().enumerate() {
                        let thr_cap = thr.min(1.0 - 1e-10);
                        let mut matched = vec![false; ng];
                        for d in 0..np {
                            let row = &ious[d * ng..(d + 1) * ng];
                            let mut best = thr_cap;
                            let mut match_pos: Option<(usize, bool)> = None;
                            for (ord_idx, &(orig_g, is_ign)) in gt_ord.iter().enumerate() {
                                if matched[ord_idx] {
                                    continue;
                                }
                                if let Some((_, false)) = match_pos
                                    && is_ign
                                {
                                    break;
                                }
                                let v = row[orig_g];
                                if v < best {
                                    continue;
                                }
                                best = v;
                                match_pos = Some((ord_idx, is_ign));
                            }
                            if let Some((ord_idx, is_ign)) = match_pos {
                                matched[ord_idx] = true;
                                if is_ign {
                                    ig[d][a] |= 1 << t;
                                } else {
                                    tp[d][a] |= 1 << t;
                                }
                            } else if !(lo..=hi).contains(&det_areas[d]) {
                                ig[d][a] |= 1 << t;
                            }
                        }
                    }
                }
            }

            let out = &mut per_class_recs[c];
            out.reserve(np);
            for (rank, &pi) in p_list.iter().enumerate() {
                out.push(ScaleDetRec {
                    score: preds[pi].score,
                    image_idx: img_idx,
                    rank: rank as u16,
                    tp: tp[rank],
                    ig: ig[rank],
                });
            }
        }
    }

    // Accumulate AP (at maxDets = 100) and AR (at maxDets = 1 / 10 / 100) per class and bucket.
    let mut ap_sum = [[0.0f64; 4]; NUM_IOU_THRESHOLDS];
    let mut ap_cnt = [0usize; 4];
    let mut ar_sum = [[0.0f64; 4]; 3];
    let mut ar_cnt = [[0usize; 4]; 3];
    let mut per_class = Vec::with_capacity(num_classes);

    for c in 0..num_classes {
        let recs = &mut per_class_recs[c];
        recs.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.image_idx.cmp(&b.image_idx))
                .then_with(|| a.rank.cmp(&b.rank))
        });

        let mut cls_ap_mean = [None::<f64>; 4];
        let mut cls_ap50 = None;
        let mut cls_ap75 = None;

        for a in 0..4 {
            let n_gt = gt_counts[c][a];
            if n_gt == 0 {
                continue;
            }
            let n_gt_f = n_gt as f64;
            ap_cnt[a] += 1;

            // Compute AP at maxDets = 100 across all 10 IoU thresholds.
            let mut ap_t = [0.0f64; NUM_IOU_THRESHOLDS];
            for t in 0..NUM_IOU_THRESHOLDS {
                let mut tp_c = 0u64;
                let mut fp_c = 0u64;
                let mut cur_b = 0usize;
                let mut bucket_max = [0.0f64; NUM_RECALL_POINTS];
                for r in recs.iter() {
                    if (r.tp[a] >> t) & 1 == 1 {
                        tp_c += 1;
                        let tp_f = tp_c as f64;
                        let rc = tp_f / n_gt_f;
                        while cur_b + 1 < NUM_RECALL_POINTS && rc >= r_grid[cur_b + 1] {
                            cur_b += 1;
                        }
                        let prec = tp_f / (tp_f + fp_c as f64 + f64::EPSILON);
                        if prec > bucket_max[cur_b] {
                            bucket_max[cur_b] = prec;
                        }
                    } else if (r.ig[a] >> t) & 1 == 0 {
                        fp_c += 1;
                    }
                }
                let mut running = 0.0f64;
                for v in bucket_max.iter_mut().rev() {
                    if *v > running {
                        running = *v;
                    } else {
                        *v = running;
                    }
                }
                let ap_val = bucket_max.iter().sum::<f64>() / NUM_RECALL_POINTS as f64;
                ap_t[t] = ap_val;
                ap_sum[t][a] += ap_val;
            }
            let mean_t = ap_t.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64;
            cls_ap_mean[a] = Some(mean_t);
            if a == 0 {
                cls_ap50 = Some(ap_t[0]);
                cls_ap75 = Some(ap_t[5]);
            }

            // Compute AR at maxDets = 1 / 10 / 100 across all 10 IoU thresholds.
            for (mi, &md) in SCALE_MAX_DETS.iter().enumerate() {
                let mut tp_by_t = [0u64; NUM_IOU_THRESHOLDS];
                for r in recs.iter() {
                    if (r.rank as usize) < md {
                        let bits = r.tp[a];
                        for (t, slot) in tp_by_t.iter_mut().enumerate() {
                            *slot += u64::from((bits >> t) & 1);
                        }
                    }
                }
                let mean_rec =
                    tp_by_t.iter().map(|&tp| tp as f64 / n_gt_f).sum::<f64>() / NUM_IOU_THRESHOLDS as f64;
                ar_sum[mi][a] += mean_rec;
                ar_cnt[mi][a] += 1;
            }
        }

        per_class.push(CocoScaleClassMetrics {
            class: c as u32,
            truths_all: gt_counts[c][0],
            truths_small: gt_counts[c][1],
            truths_medium: gt_counts[c][2],
            truths_large: gt_counts[c][3],
            ap: cls_ap_mean[0],
            ap50: cls_ap50,
            ap75: cls_ap75,
            ap_small: cls_ap_mean[1],
            ap_medium: cls_ap_mean[2],
            ap_large: cls_ap_mean[3],
        });
    }

    let mean_ap_bucket = |a: usize| -> Option<f64> {
        if ap_cnt[a] == 0 {
            None
        } else {
            let total: f64 = (0..NUM_IOU_THRESHOLDS).map(|t| ap_sum[t][a]).sum();
            Some(total / (NUM_IOU_THRESHOLDS * ap_cnt[a]) as f64)
        }
    };
    let ap_at_t = |t: usize| -> Option<f64> {
        if ap_cnt[0] == 0 { None } else { Some(ap_sum[t][0] / ap_cnt[0] as f64) }
    };
    let mean_ar = |mi: usize, a: usize| -> Option<f64> {
        if ar_cnt[mi][a] == 0 { None } else { Some(ar_sum[mi][a] / ar_cnt[mi][a] as f64) }
    };

    Ok(CocoScaleDiagnostics {
        bounds,
        ap: mean_ap_bucket(0),
        ap50: ap_at_t(0),
        ap75: ap_at_t(5),
        ap_small: mean_ap_bucket(1),
        ap_medium: mean_ap_bucket(2),
        ap_large: mean_ap_bucket(3),
        ar_1: mean_ar(0, 0),
        ar_10: mean_ar(1, 0),
        ar_100: mean_ar(2, 0),
        ar_small: mean_ar(2, 1),
        ar_medium: mean_ar(2, 2),
        ar_large: mean_ar(2, 3),
        num_images: images.len() as u64,
        truths_all: gt_counts.iter().map(|g| g[0]).sum(),
        truths_small: gt_counts.iter().map(|g| g[1]).sum(),
        truths_medium: gt_counts.iter().map(|g| g[2]).sum(),
        truths_large: gt_counts.iter().map(|g| g[3]).sum(),
        per_class,
    })
}
