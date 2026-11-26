//! Seeded image-level bootstrap confidence intervals for COCO-style `mAP@50` and `mAP@50:95`.
//!
//! Each image's detections are matched once against its ground truths using [`Evaluator`]'s exact
//! per-image matching rules across all 10 IoU thresholds (`0.50:0.05:0.95`). Bootstrap replicates
//! then resample image indices with replacement using a deterministic per-replicate SplitMix64 seed
//! (guaranteeing bitwise identical results regardless of thread count), evaluate exact 101-point
//! COCO AP on every replicate without repeating IoU calculations, and report percentile confidence
//! intervals (linearly interpolated Type-7 quantiles).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::metrics::{
    DetRecord, EvalConfig, Evaluator, NUM_IOU_THRESHOLDS, NUM_RECALL_POINTS, Truth, iou_thresholds,
    recall_points,
};
use crate::nms::Detection;

/// Configuration for [`compute_bootstrap_map`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BootstrapConfig {
    /// Number of bootstrap replicates (`>= 1`, default `1000`).
    pub num_samples: usize,
    /// Two-sided confidence level in `(0.0, 1.0)` (default `0.95` for a 95% CI: `[2.5%, 97.5%]`).
    pub confidence_level: f64,
    /// Deterministic PRNG seed (default `0xF010_B007_574A_0001`).
    pub seed: u64,
    /// Underlying [`Evaluator`] matching configuration (`max_dets_per_class`, `coco_compatible`, etc.).
    pub eval: EvalConfig,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            confidence_level: 0.95,
            seed: 0xF010_B007_574A_0001,
            eval: EvalConfig::default(),
        }
    }
}

impl BootstrapConfig {
    /// Validate bootstrap hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_samples == 0 {
            return Err(Error::Config("compute_bootstrap_map: num_samples must be >= 1".into()));
        }
        if !(self.confidence_level.is_finite() && self.confidence_level > 0.0 && self.confidence_level < 1.0)
        {
            return Err(Error::Config(format!(
                "compute_bootstrap_map: confidence_level must be finite and in (0, 1), got {}",
                self.confidence_level
            )));
        }
        Ok(())
    }
}

/// Point estimate and bootstrap percentile confidence interval for one scalar metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Full-dataset point estimate (matches [`Evaluator::compute`]).
    pub point: f64,
    /// Mean across bootstrap replicates.
    pub mean: f64,
    /// Sample standard deviation across bootstrap replicates (`0.0` when `num_samples <= 1`).
    pub std: f64,
    /// Lower percentile bound `(1 - confidence_level) / 2`.
    pub lower: f64,
    /// Upper percentile bound `1 - (1 - confidence_level) / 2`.
    pub upper: f64,
}

/// Per-class bootstrap confidence intervals for `AP@50` and `AP@50:95`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassBootstrapCi {
    /// Foreground class index.
    pub class: u32,
    /// Total ground truths for this class in the full dataset.
    pub num_truths: u64,
    /// Number of bootstrap replicates in which this class had at least one ground truth.
    pub valid_samples: usize,
    /// Bootstrap confidence interval for `AP@50`.
    pub ap50: ConfidenceInterval,
    /// Bootstrap confidence interval for `AP@50:95`.
    pub ap50_95: ConfidenceInterval,
}

/// Full seeded bootstrap mAP confidence-interval report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BootstrapMapReport {
    /// Number of images evaluated.
    pub num_images: usize,
    /// Number of foreground classes.
    pub num_classes: usize,
    /// Number of bootstrap replicates executed.
    pub num_samples: usize,
    /// Two-sided confidence level in `(0, 1)`.
    pub confidence_level: f64,
    /// PRNG seed used for resampling.
    pub seed: u64,
    /// Confidence interval for `mAP@50`.
    pub map50: ConfidenceInterval,
    /// Confidence interval for `mAP@50:95`.
    pub map50_95: ConfidenceInterval,
    /// Confidence interval for `mAP` at each IoU threshold `0.50:0.05:0.95`.
    pub map_per_iou: Vec<ConfidenceInterval>,
    /// IoU thresholds (`0.50:0.05:0.95`).
    pub iou_thresholds: Vec<f32>,
    /// Per-class bootstrap CIs for classes `0..num_classes`.
    pub per_class: Vec<ClassBootstrapCi>,
}

/// Deterministic SplitMix64 mixing step (thread-independent per-replicate RNG).
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Unbiased bounded uniform integer in `0..bound` using Lemire's rejection method.
#[inline]
fn next_bounded(state: &mut u64, bound: usize) -> usize {
    let b = bound as u64;
    let threshold = b.wrapping_neg() % b;
    loop {
        let r = splitmix64(state);
        let prod = (r as u128) * (b as u128);
        if (prod as u64) >= threshold {
            return (prod >> 64) as usize;
        }
    }
}

/// Pre-matched detection tagged with its originating image and per-image order for exact tie preservation.
#[derive(Clone, Copy, Debug)]
struct IndexedDet {
    score: f32,
    img_idx: usize,
    det_idx: usize,
    tp: u16,
    ig: u16,
}

/// Evaluate exact 101-point COCO AP across all 10 IoU thresholds for one class given per-image
/// bootstrap multiplicities `counts[img_idx]`.
///
/// Matches [`Evaluator::compute`] bit-for-bit on the resampled image multiset in ascending image
/// order: single detections with unique `(score, img_idx)` apply multiplicity `w` in `O(1)` (proven
/// identical under the backward suffix-max precision envelope), while tied-score runs within the
/// same image replay `0..w` copies in exact `(copy, det_idx)` order.
fn class_ap_weighted(
    sorted_dets: &[IndexedDet],
    counts: &[u32],
    n_gt: u64,
    recall_pts: &[f64; NUM_RECALL_POINTS],
) -> [f64; NUM_IOU_THRESHOLDS] {
    if n_gt == 0 || sorted_dets.is_empty() {
        return [0.0; NUM_IOU_THRESHOLDS];
    }
    let n_gt_f = n_gt as f64;
    let mut tp_c = [0u64; NUM_IOU_THRESHOLDS];
    let mut fp_c = [0u64; NUM_IOU_THRESHOLDS];
    let mut cur_b = [0usize; NUM_IOU_THRESHOLDS];
    let mut bucket_max = [[0.0f64; NUM_RECALL_POINTS]; NUM_IOU_THRESHOLDS];

    let apply_one = |tp_bits: u16,
                     ig_bits: u16,
                     w: u64,
                     tp_c: &mut [u64; NUM_IOU_THRESHOLDS],
                     fp_c: &mut [u64; NUM_IOU_THRESHOLDS],
                     cur_b: &mut [usize; NUM_IOU_THRESHOLDS],
                     bucket_max: &mut [[f64; NUM_RECALL_POINTS]; NUM_IOU_THRESHOLDS]| {
        if tp_bits == 0 && ig_bits == 0 {
            for f in fp_c.iter_mut() {
                *f += w;
            }
            return;
        }
        for t in 0..NUM_IOU_THRESHOLDS {
            if (tp_bits >> t) & 1 == 1 {
                tp_c[t] += w;
                let tp_f = tp_c[t] as f64;
                let rc = tp_f / n_gt_f;
                let mut b = cur_b[t];
                while b + 1 < NUM_RECALL_POINTS && rc >= recall_pts[b + 1] {
                    b += 1;
                }
                cur_b[t] = b;
                let p = tp_f / (tp_f + fp_c[t] as f64 + f64::EPSILON);
                if p > bucket_max[t][b] {
                    bucket_max[t][b] = p;
                }
            } else if (ig_bits >> t) & 1 == 0 {
                fp_c[t] += w;
            }
        }
    };

    let len = sorted_dets.len();
    let mut i = 0usize;
    while i < len {
        let d0 = sorted_dets[i];
        let w = counts[d0.img_idx] as u64;
        let mut end = i + 1;
        while end < len
            && sorted_dets[end].img_idx == d0.img_idx
            && sorted_dets[end].score.to_bits() == d0.score.to_bits()
        {
            end += 1;
        }
        if w > 0 {
            if end - i == 1 || w == 1 {
                for d in &sorted_dets[i..end] {
                    apply_one(d.tp, d.ig, w, &mut tp_c, &mut fp_c, &mut cur_b, &mut bucket_max);
                }
            } else {
                for _ in 0..w {
                    for d in &sorted_dets[i..end] {
                        apply_one(d.tp, d.ig, 1, &mut tp_c, &mut fp_c, &mut cur_b, &mut bucket_max);
                    }
                }
            }
        }
        i = end;
    }

    let mut ap = [0.0f64; NUM_IOU_THRESHOLDS];
    for (t, ap_t) in ap.iter_mut().enumerate() {
        let bm = &mut bucket_max[t];
        let mut running = 0.0f64;
        for v in bm.iter_mut().rev() {
            if *v > running {
                running = *v;
            } else {
                *v = running;
            }
        }
        *ap_t = bm.iter().sum::<f64>() / NUM_RECALL_POINTS as f64;
    }
    ap
}

/// Linearly interpolated Type-7 quantile on a non-empty ascending slice `sorted`.
#[inline]
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = (q * (sorted.len() - 1) as f64).clamp(0.0, (sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn summarize_samples(point: f64, samples: &mut [f64], confidence_level: f64) -> ConfidenceInterval {
    if samples.is_empty() {
        return ConfidenceInterval { point, mean: 0.0, std: 0.0, lower: 0.0, upper: 0.0 };
    }
    samples.sort_unstable_by(f64::total_cmp);
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let std = if n > 1 {
        let var = samples.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
        var.max(0.0).sqrt()
    } else {
        0.0
    };
    let alpha = 1.0 - confidence_level;
    let lower = percentile_sorted(samples, 0.5 * alpha);
    let upper = percentile_sorted(samples, 1.0 - 0.5 * alpha);
    ConfidenceInterval { point, mean, std, lower, upper }
}

/// Compute seeded image-level bootstrap percentile confidence intervals for `mAP@50`, `mAP@50:95`,
/// per-IoU mAP, and per-class `AP@50` / `AP@50:95`.
///
/// Results are strictly deterministic for a given `(images, cfg)` regardless of CPU thread count.
pub fn compute_bootstrap_map<P, T>(
    num_classes: usize,
    images: &[(P, T)],
    cfg: &BootstrapConfig,
) -> Result<BootstrapMapReport>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    cfg.validate()?;
    let thresholds = iou_thresholds();
    let recall_pts = recall_points();
    let num_images = images.len();

    // Validate all class ids before running matching.
    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        for (i, p) in preds_ref.as_ref().iter().enumerate() {
            if (p.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_bootstrap_map: image {img_idx} pred[{i}].class = {} >= num_classes ({num_classes})",
                    p.class
                )));
            }
        }
        for (i, t) in truths_ref.as_ref().iter().enumerate() {
            if (t.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_bootstrap_map: image {img_idx} truth[{i}].class = {} >= num_classes ({num_classes})",
                    t.class
                )));
            }
        }
    }

    if num_images == 0 || num_classes == 0 {
        return Ok(BootstrapMapReport {
            num_images,
            num_classes,
            num_samples: cfg.num_samples,
            confidence_level: cfg.confidence_level,
            seed: cfg.seed,
            map50: ConfidenceInterval::default(),
            map50_95: ConfidenceInterval::default(),
            map_per_iou: vec![ConfidenceInterval::default(); NUM_IOU_THRESHOLDS],
            iou_thresholds: thresholds.iter().map(|&t| t as f32).collect(),
            per_class: (0..num_classes)
                .map(|c| ClassBootstrapCi {
                    class: c as u32,
                    num_truths: 0,
                    valid_samples: 0,
                    ap50: ConfidenceInterval::default(),
                    ap50_95: ConfidenceInterval::default(),
                })
                .collect(),
        });
    }

    // Match each image once using Evaluator's exact per-image matching kernel.
    let mut img_gt_counts: Vec<Vec<u64>> = vec![vec![0u64; num_classes]; num_images];
    let mut class_dets: Vec<Vec<IndexedDet>> = vec![Vec::new(); num_classes];
    let mut total_gt_counts = vec![0u64; num_classes];

    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        let mut ev = Evaluator::with_config(num_classes, cfg.eval.clone());
        ev.add_image(preds_ref.as_ref(), truths_ref.as_ref());
        let (dets_per_class, gts_per_class): (Vec<Vec<DetRecord>>, Vec<u64>) = ev.into_matched_records();
        for (c, (dets_c, n_gt)) in dets_per_class.into_iter().zip(gts_per_class).enumerate() {
            img_gt_counts[img_idx][c] = n_gt;
            total_gt_counts[c] += n_gt;
            let dst = &mut class_dets[c];
            dst.reserve(dets_c.len());
            for (det_idx, rec) in dets_c.into_iter().enumerate() {
                dst.push(IndexedDet { score: rec.score, img_idx, det_idx, tp: rec.tp, ig: rec.ig });
            }
        }
    }

    // Sort each class's detections once by (score DESC, img_idx ASC, det_idx ASC).
    for dets in &mut class_dets {
        dets.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.img_idx.cmp(&b.img_idx))
                .then_with(|| a.det_idx.cmp(&b.det_idx))
        });
    }

    // Full-dataset point estimates (counts = all 1s).
    let ones = vec![1u32; num_images];
    let mut point_class_ap = vec![[0.0f64; NUM_IOU_THRESHOLDS]; num_classes];
    let mut point_map_per_iou = [0.0f64; NUM_IOU_THRESHOLDS];
    let mut active_full = 0usize;
    for c in 0..num_classes {
        if total_gt_counts[c] > 0 {
            let ap = class_ap_weighted(&class_dets[c], &ones, total_gt_counts[c], &recall_pts);
            point_class_ap[c] = ap;
            for t in 0..NUM_IOU_THRESHOLDS {
                point_map_per_iou[t] += ap[t];
            }
            active_full += 1;
        }
    }
    if active_full > 0 {
        for v in &mut point_map_per_iou {
            *v /= active_full as f64;
        }
    }
    let point_map50 = point_map_per_iou[0];
    let point_map50_95 = point_map_per_iou.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64;

    // Run bootstrap replicates in parallel with deterministic per-replicate SplitMix64 streams.
    struct RepResult {
        map_per_iou: [f64; NUM_IOU_THRESHOLDS],
        class_ap50: Vec<Option<f64>>,
        class_ap50_95: Vec<Option<f64>>,
    }

    let seed = cfg.seed;
    let replicates: Vec<RepResult> = (0..cfg.num_samples)
        .into_par_iter()
        .map(|rep_idx| {
            let mut rng = seed ^ ((rep_idx as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let _ = splitmix64(&mut rng);
            let mut counts = vec![0u32; num_images];
            for _ in 0..num_images {
                let idx = next_bounded(&mut rng, num_images);
                counts[idx] += 1;
            }

            let mut map_per_iou = [0.0f64; NUM_IOU_THRESHOLDS];
            let mut class_ap50 = vec![None; num_classes];
            let mut class_ap50_95 = vec![None; num_classes];
            let mut active = 0usize;

            for c in 0..num_classes {
                if total_gt_counts[c] == 0 {
                    continue;
                }
                let mut n_gt_b = 0u64;
                for (img_idx, &w) in counts.iter().enumerate() {
                    if w > 0 {
                        n_gt_b += w as u64 * img_gt_counts[img_idx][c];
                    }
                }
                if n_gt_b == 0 {
                    continue;
                }
                let ap = class_ap_weighted(&class_dets[c], &counts, n_gt_b, &recall_pts);
                for t in 0..NUM_IOU_THRESHOLDS {
                    map_per_iou[t] += ap[t];
                }
                class_ap50[c] = Some(ap[0]);
                class_ap50_95[c] = Some(ap.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64);
                active += 1;
            }

            if active > 0 {
                for v in &mut map_per_iou {
                    *v /= active as f64;
                }
            }
            RepResult { map_per_iou, class_ap50, class_ap50_95 }
        })
        .collect();

    let mut map50_samples: Vec<f64> = replicates.iter().map(|r| r.map_per_iou[0]).collect();
    let mut map50_95_samples: Vec<f64> =
        replicates.iter().map(|r| r.map_per_iou.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64).collect();

    let map50 = summarize_samples(point_map50, &mut map50_samples, cfg.confidence_level);
    let map50_95 = summarize_samples(point_map50_95, &mut map50_95_samples, cfg.confidence_level);

    let map_per_iou: Vec<ConfidenceInterval> = (0..NUM_IOU_THRESHOLDS)
        .map(|t| {
            let mut s: Vec<f64> = replicates.iter().map(|r| r.map_per_iou[t]).collect();
            summarize_samples(point_map_per_iou[t], &mut s, cfg.confidence_level)
        })
        .collect();

    let per_class: Vec<ClassBootstrapCi> = (0..num_classes)
        .map(|c| {
            let mut s50: Vec<f64> = replicates.iter().filter_map(|r| r.class_ap50[c]).collect();
            let mut s50_95: Vec<f64> = replicates.iter().filter_map(|r| r.class_ap50_95[c]).collect();
            let valid_samples = s50.len();
            let pt50 = point_class_ap[c][0];
            let pt50_95 = point_class_ap[c].iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64;
            ClassBootstrapCi {
                class: c as u32,
                num_truths: total_gt_counts[c],
                valid_samples,
                ap50: summarize_samples(pt50, &mut s50, cfg.confidence_level),
                ap50_95: summarize_samples(pt50_95, &mut s50_95, cfg.confidence_level),
            }
        })
        .collect();

    Ok(BootstrapMapReport {
        num_images,
        num_classes,
        num_samples: cfg.num_samples,
        confidence_level: cfg.confidence_level,
        seed: cfg.seed,
        map50,
        map50_95,
        map_per_iou,
        iou_thresholds: thresholds.iter().map(|&t| t as f32).collect(),
        per_class,
    })
}
