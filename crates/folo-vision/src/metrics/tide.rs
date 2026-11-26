//! TIDE-style object detection error attribution (Bolya et al., ECCV 2020) operating directly on
//! `(preds, truths)` image streams.
//!
//! Classifies every false positive into one of five mutually exclusive error categories (`Cls`,
//! `Loc`, `Both`, `Dup`, `Bkg`), identifies unexplained missed ground truths (`Miss`), and
//! computes exact oracle-fix `dAP = AP_oracle - AP_base >= 0` attribution for every error type
//! plus summary `FP` and `FN` impacts.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{NUM_RECALL_POINTS, Truth, iou_f64, recall_points};
use crate::nms::Detection;

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Configuration for TIDE-style error attribution ([`compute_tide_diagnostics`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TideConfig {
    /// Foreground / positive IoU threshold `t_pos` (default `0.50`).
    pub pos_iou: f32,
    /// Background IoU threshold `t_bg` (default `0.10`; must satisfy `0 < bg_iou < pos_iou <= 1`).
    pub bg_iou: f32,
    /// Optional per-image, per-class detection cap (default `Some(100)`, matching COCO).
    pub max_dets_per_class: Option<usize>,
}

impl Default for TideConfig {
    fn default() -> Self {
        Self { pos_iou: 0.50, bg_iou: 0.10, max_dets_per_class: Some(100) }
    }
}

impl TideConfig {
    /// Validate `0 < bg_iou < pos_iou <= 1` (all finite).
    pub fn validate(&self) -> Result<()> {
        if !(self.pos_iou.is_finite()
            && self.bg_iou.is_finite()
            && self.bg_iou > 0.0
            && self.bg_iou < self.pos_iou
            && self.pos_iou <= 1.0)
        {
            return Err(Error::Config(format!(
                "TideConfig: require finite 0 < bg_iou < pos_iou <= 1, got bg_iou={}, pos_iou={}",
                self.bg_iou, self.pos_iou
            )));
        }
        Ok(())
    }
}

/// Mutually exclusive TIDE error category counts (Bolya et al., ECCV 2020).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TideErrorCounts {
    /// True positives matched at `pos_iou`.
    pub tp: u64,
    /// Classification errors: wrong class, localized with `IoU >= pos_iou` to another class's GT.
    pub cls: u64,
    /// Localization errors: right class, `bg_iou <= IoU < pos_iou` with a same-class GT.
    pub loc: u64,
    /// Both Cls and Loc errors: wrong class and `bg_iou <= IoU < pos_iou` with another class's GT.
    pub both: u64,
    /// Duplicate detections: right class, `IoU >= pos_iou` with an already-matched same-class GT.
    pub dup: u64,
    /// Background false positives: `IoU < bg_iou` with all ground truths of every class.
    pub bkg: u64,
    /// Missed ground truths: unmatched GTs not explained by any `Cls` or `Loc` false positive.
    pub miss: u64,
    /// Total false positives (`cls + loc + both + dup + bkg`).
    pub fp: u64,
    /// Total false negatives (all unmatched GTs at `pos_iou`, `>= miss`).
    pub fn_count: u64,
}

/// TIDE `dAP = AP_oracle - AP_base >= 0` error impacts for each error type and summary `FP` / `FN`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TideErrorImpacts {
    /// `dAP_Cls`: AP gain from fixing classification errors.
    pub cls: f64,
    /// `dAP_Loc`: AP gain from fixing localization errors.
    pub loc: f64,
    /// `dAP_Both`: AP gain from removing `Both` (misclassified + mislocalized) false positives.
    pub both: f64,
    /// `dAP_Dup`: AP gain from removing duplicate detections.
    pub dup: f64,
    /// `dAP_Bkg`: AP gain from removing background false positives.
    pub bkg: f64,
    /// `dAP_Miss`: AP gain from fixing unexplained missed ground truths.
    pub miss: f64,
    /// `dAP_FP`: AP gain from removing all false positives.
    pub fp: f64,
    /// `dAP_FN`: AP gain from fixing all false negatives (missed ground truths).
    pub fn_impact: f64,
}

/// Per-class TIDE error attribution breakdown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TideClassBreakdown {
    /// Class index.
    pub class: u32,
    /// Number of ground truths for this class.
    pub num_truths: u64,
    /// Number of evaluated predictions for this class.
    pub num_preds: u64,
    /// Baseline 101-point AP at `pos_iou` (`0.0` when `num_truths == 0`).
    pub ap_base: f64,
    /// Error counts attributed to this class.
    pub counts: TideErrorCounts,
    /// Per-class `dAP` impact of each error type (`0.0` when `num_truths == 0`).
    pub impacts: TideErrorImpacts,
}

/// Complete TIDE-style error attribution report across the dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TideDiagnostics {
    /// Foreground / positive IoU threshold used.
    pub pos_iou: f32,
    /// Background IoU threshold used.
    pub bg_iou: f32,
    /// Baseline mAP across classes with `num_truths > 0` at `pos_iou` (`0.0` if none).
    pub map_base: f64,
    /// Aggregate error counts summed across all classes.
    pub counts: TideErrorCounts,
    /// Mean `dAP = mAP_oracle - mAP_base >= 0` across active classes for each error category.
    pub impacts: TideErrorImpacts,
    /// Per-class TIDE breakdown for classes `0..num_classes`.
    pub per_class: Vec<TideClassBreakdown>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TideDetKind {
    Tp,
    Cls { target_gt: usize },
    Loc { target_gt: usize },
    Both,
    Dup,
    Bkg,
}

#[derive(Clone, Copy, Debug)]
struct TideDetRecord {
    score: f32,
    image_idx: usize,
    det_idx: usize,
    kind: TideDetKind,
}

/// Exact 101-point COCO-interpolated AP from a score-sorted stream of `(score, is_tp)` flags.
///
/// When `original_gt > 0` and `effective_gt == 0` (every ground truth was fixed by a `Miss`/`FN`
/// oracle), returns `1.0` per TIDE conventions.
fn ap_101_from_stream(dets: &[(f32, usize, usize, bool)], original_gt: u64, effective_gt: u64) -> f64 {
    if original_gt == 0 {
        return 0.0;
    }
    if effective_gt == 0 {
        return 1.0;
    }
    if dets.is_empty() {
        return 0.0;
    }
    let r_grid = recall_points();
    let n_gt_f = effective_gt as f64;
    let mut tp_c = 0u64;
    let mut fp_c = 0u64;
    let mut cur_b = 0usize;
    let mut bucket_max = [0.0f64; NUM_RECALL_POINTS];

    for &(_, _, _, is_tp) in dets {
        if is_tp {
            tp_c += 1;
            let tp_f = tp_c as f64;
            let rc = (tp_f / n_gt_f).min(1.0);
            while cur_b + 1 < NUM_RECALL_POINTS && rc >= r_grid[cur_b + 1] {
                cur_b += 1;
            }
            let prec = tp_f / (tp_f + fp_c as f64 + f64::EPSILON);
            if prec > bucket_max[cur_b] {
                bucket_max[cur_b] = prec;
            }
        } else {
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
    bucket_max.iter().sum::<f64>() / NUM_RECALL_POINTS as f64
}

fn sort_stream(stream: &mut [(f32, usize, usize, bool)]) {
    stream
        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
}

/// Compute TIDE-style detection error attribution (`Cls`, `Loc`, `Both`, `Dup`, `Bkg`, `Miss`,
/// plus `FP` and `FN` summaries) with exact `dAP = AP_oracle - AP_base >= 0` oracle-fix impacts.
pub fn compute_tide_diagnostics<P, T>(
    num_classes: usize,
    images: &[(P, T)],
    cfg: &TideConfig,
) -> Result<TideDiagnostics>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    cfg.validate()?;
    let pos_thr = cfg.pos_iou as f64;
    let bg_thr = cfg.bg_iou as f64;

    let mut per_class_dets: Vec<Vec<TideDetRecord>> = vec![Vec::new(); num_classes];
    let mut gt_counts = vec![0u64; num_classes];
    let mut gt_global_matched: Vec<bool> = Vec::new();
    let mut gt_global_class: Vec<usize> = Vec::new();
    let mut gt_explained_by_cls_or_loc: Vec<bool> = Vec::new();

    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        let preds = preds_ref.as_ref();
        let truths = truths_ref.as_ref();
        for (i, p) in preds.iter().enumerate() {
            if (p.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_tide_diagnostics: image {img_idx} pred[{i}].class = {} >= num_classes ({num_classes})",
                    p.class
                )));
            }
        }
        for (i, t) in truths.iter().enumerate() {
            if (t.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "compute_tide_diagnostics: image {img_idx} truth[{i}].class = {} >= num_classes ({num_classes})",
                    t.class
                )));
            }
        }

        let gt_base = gt_global_matched.len();
        let mut img_gts_by_class: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        let mut finite_gts: Vec<usize> = Vec::with_capacity(truths.len());
        for (gi, g) in truths.iter().enumerate() {
            let cls = g.class as usize;
            gt_counts[cls] += 1;
            gt_global_matched.push(false);
            gt_global_class.push(cls);
            gt_explained_by_cls_or_loc.push(false);
            if box_is_finite(&g.bbox) {
                img_gts_by_class[cls].push(gi);
                finite_gts.push(gi);
            }
        }

        let mut img_preds_by_class: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        for (pi, p) in preds.iter().enumerate() {
            if p.score.is_finite() && box_is_finite(&p.bbox) {
                img_preds_by_class[p.class as usize].push(pi);
            }
        }

        // Pass 1: standard greedy per-class matching at `pos_thr` to identify TPs and matched GTs.
        let mut is_tp_flag: Vec<Vec<bool>> = vec![Vec::new(); num_classes];
        for c in 0..num_classes {
            let p_list = &mut img_preds_by_class[c];
            p_list
                .sort_unstable_by(|&a, &b| preds[b].score.total_cmp(&preds[a].score).then_with(|| a.cmp(&b)));
            if let Some(cap) = cfg.max_dets_per_class {
                p_list.truncate(cap);
            }
            let g_list = &img_gts_by_class[c];
            let flags = &mut is_tp_flag[c];
            flags.resize(p_list.len(), false);
            for (local_p, &pi) in p_list.iter().enumerate() {
                let pb = &preds[pi].bbox;
                let mut best_iou = -1.0f64;
                let mut best_gi = None;
                for &gi in g_list {
                    if gt_global_matched[gt_base + gi] {
                        continue;
                    }
                    let v = iou_f64(pb, &truths[gi].bbox);
                    if v >= pos_thr && v > best_iou {
                        best_iou = v;
                        best_gi = Some(gi);
                    }
                }
                if let Some(gi) = best_gi {
                    gt_global_matched[gt_base + gi] = true;
                    flags[local_p] = true;
                }
            }
        }

        // Pass 2: classify every FP into mutually exclusive TIDE categories.
        for c in 0..num_classes {
            let p_list = &img_preds_by_class[c];
            let flags = &is_tp_flag[c];
            for (local_p, &pi) in p_list.iter().enumerate() {
                let p = &preds[pi];
                let kind = if flags[local_p] {
                    TideDetKind::Tp
                } else {
                    let mut best_same_iou = -1.0f64;
                    let mut best_same_gi = None;
                    let mut best_other_iou = -1.0f64;
                    let mut best_other_gi = None;
                    for &gi in &finite_gts {
                        let v = iou_f64(&p.bbox, &truths[gi].bbox);
                        if truths[gi].class as usize == c {
                            if v > best_same_iou {
                                best_same_iou = v;
                                best_same_gi = Some(gi);
                            }
                        } else if v > best_other_iou {
                            best_other_iou = v;
                            best_other_gi = Some(gi);
                        }
                    }
                    if best_same_iou >= pos_thr {
                        TideDetKind::Dup
                    } else if best_same_iou >= bg_thr {
                        let target_gt = gt_base + best_same_gi.unwrap();
                        gt_explained_by_cls_or_loc[target_gt] = true;
                        TideDetKind::Loc { target_gt }
                    } else if best_other_iou >= pos_thr {
                        let gi = best_other_gi.unwrap();
                        let target_gt = gt_base + gi;
                        gt_explained_by_cls_or_loc[target_gt] = true;
                        TideDetKind::Cls { target_gt }
                    } else if best_other_iou >= bg_thr {
                        TideDetKind::Both
                    } else {
                        TideDetKind::Bkg
                    }
                };
                per_class_dets[c].push(TideDetRecord {
                    score: p.score,
                    image_idx: img_idx,
                    det_idx: pi,
                    kind,
                });
            }
        }
    }

    // Count FN and unexplained Miss per class.
    let mut fn_by_class = vec![0u64; num_classes];
    let mut miss_by_class = vec![0u64; num_classes];
    for (g_idx, &matched) in gt_global_matched.iter().enumerate() {
        if !matched {
            let cls = gt_global_class[g_idx];
            fn_by_class[cls] += 1;
            if !gt_explained_by_cls_or_loc[g_idx] {
                miss_by_class[cls] += 1;
            }
        }
    }

    // Pre-resolve Cls oracle promotions: for each unmatched GT targeted by one or more Cls FPs,
    // promote the single highest-scoring Cls FP into `target_class`.
    let mut best_cls_for_gt: Vec<Option<(f32, usize, usize)>> = vec![None; gt_global_matched.len()];
    for dets in &per_class_dets {
        for d in dets {
            if let TideDetKind::Cls { target_gt, .. } = d.kind
                && !gt_global_matched[target_gt]
            {
                let cand = (d.score, d.image_idx, d.det_idx);
                let replace = match best_cls_for_gt[target_gt] {
                    None => true,
                    Some(prev) => {
                        cand.0.total_cmp(&prev.0).is_gt()
                            || (cand.0.total_cmp(&prev.0).is_eq() && (cand.1, cand.2) < (prev.1, prev.2))
                    }
                };
                if replace {
                    best_cls_for_gt[target_gt] = Some(cand);
                }
            }
        }
    }
    let mut cls_promoted_by_class: Vec<Vec<(f32, usize, usize, bool)>> = vec![Vec::new(); num_classes];
    for (g_idx, promoted) in best_cls_for_gt.into_iter().enumerate() {
        if let Some((score, img_idx, det_idx)) = promoted {
            let cls = gt_global_class[g_idx];
            cls_promoted_by_class[cls].push((score, img_idx, det_idx, true));
        }
    }

    let mut per_class = Vec::with_capacity(num_classes);
    let mut total_counts = TideErrorCounts::default();
    let mut sum_base = 0.0f64;
    let mut sum_imp = TideErrorImpacts::default();
    let mut active_classes = 0usize;

    for c in 0..num_classes {
        let dets = &mut per_class_dets[c];
        dets.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.image_idx.cmp(&b.image_idx))
                .then_with(|| a.det_idx.cmp(&b.det_idx))
        });

        let mut counts = TideErrorCounts {
            fn_count: fn_by_class[c],
            miss: miss_by_class[c],
            ..TideErrorCounts::default()
        };
        for d in dets.iter() {
            match d.kind {
                TideDetKind::Tp => counts.tp += 1,
                TideDetKind::Cls { .. } => counts.cls += 1,
                TideDetKind::Loc { .. } => counts.loc += 1,
                TideDetKind::Both => counts.both += 1,
                TideDetKind::Dup => counts.dup += 1,
                TideDetKind::Bkg => counts.bkg += 1,
            }
        }
        counts.fp = counts.cls + counts.loc + counts.both + counts.dup + counts.bkg;

        total_counts.tp += counts.tp;
        total_counts.cls += counts.cls;
        total_counts.loc += counts.loc;
        total_counts.both += counts.both;
        total_counts.dup += counts.dup;
        total_counts.bkg += counts.bkg;
        total_counts.miss += counts.miss;
        total_counts.fp += counts.fp;
        total_counts.fn_count += counts.fn_count;

        let n_gt = gt_counts[c];
        if n_gt == 0 {
            per_class.push(TideClassBreakdown {
                class: c as u32,
                num_truths: 0,
                num_preds: dets.len() as u64,
                ap_base: 0.0,
                counts,
                impacts: TideErrorImpacts::default(),
            });
            continue;
        }

        let base_stream: Vec<(f32, usize, usize, bool)> =
            dets.iter().map(|d| (d.score, d.image_idx, d.det_idx, d.kind == TideDetKind::Tp)).collect();
        let ap_base = ap_101_from_stream(&base_stream, n_gt, n_gt);

        // Oracle 1: Fix Cls (drop Cls FPs from this class, inject promoted Cls TPs targeting this class).
        let mut cls_stream: Vec<(f32, usize, usize, bool)> = dets
            .iter()
            .filter(|d| !matches!(d.kind, TideDetKind::Cls { .. }))
            .map(|d| (d.score, d.image_idx, d.det_idx, d.kind == TideDetKind::Tp))
            .collect();
        cls_stream.extend_from_slice(&cls_promoted_by_class[c]);
        sort_stream(&mut cls_stream);
        let ap_cls = ap_101_from_stream(&cls_stream, n_gt, n_gt);

        // Oracle 2: Fix Loc (for each unmatched GT targeted by Loc FPs, promote highest-scoring Loc FP to TP; drop other Loc FPs).
        let mut loc_claimed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut loc_stream: Vec<(f32, usize, usize, bool)> = Vec::with_capacity(dets.len());
        for d in dets.iter() {
            match d.kind {
                TideDetKind::Loc { target_gt } => {
                    if !gt_global_matched[target_gt] && loc_claimed.insert(target_gt) {
                        loc_stream.push((d.score, d.image_idx, d.det_idx, true));
                    }
                }
                _ => loc_stream.push((d.score, d.image_idx, d.det_idx, d.kind == TideDetKind::Tp)),
            }
        }
        let ap_loc = ap_101_from_stream(&loc_stream, n_gt, n_gt);

        // Oracles 3â€“5 & FP: drop matching FP category.
        let ap_drop = |drop_fn: fn(TideDetKind) -> bool| -> f64 {
            let stream: Vec<(f32, usize, usize, bool)> = dets
                .iter()
                .filter(|d| !drop_fn(d.kind))
                .map(|d| (d.score, d.image_idx, d.det_idx, d.kind == TideDetKind::Tp))
                .collect();
            ap_101_from_stream(&stream, n_gt, n_gt)
        };
        let ap_both = ap_drop(|k| k == TideDetKind::Both);
        let ap_dup = ap_drop(|k| k == TideDetKind::Dup);
        let ap_bkg = ap_drop(|k| k == TideDetKind::Bkg);
        let ap_fp = ap_drop(|k| k != TideDetKind::Tp);

        // Oracles 6 & FN: reduce effective GT denominator by missed / FN count.
        let ap_miss = ap_101_from_stream(&base_stream, n_gt, n_gt.saturating_sub(counts.miss));
        let ap_fn = ap_101_from_stream(&base_stream, n_gt, n_gt.saturating_sub(counts.fn_count));

        let impacts = TideErrorImpacts {
            cls: (ap_cls - ap_base).max(0.0),
            loc: (ap_loc - ap_base).max(0.0),
            both: (ap_both - ap_base).max(0.0),
            dup: (ap_dup - ap_base).max(0.0),
            bkg: (ap_bkg - ap_base).max(0.0),
            miss: (ap_miss - ap_base).max(0.0),
            fp: (ap_fp - ap_base).max(0.0),
            fn_impact: (ap_fn - ap_base).max(0.0),
        };

        sum_base += ap_base;
        sum_imp.cls += impacts.cls;
        sum_imp.loc += impacts.loc;
        sum_imp.both += impacts.both;
        sum_imp.dup += impacts.dup;
        sum_imp.bkg += impacts.bkg;
        sum_imp.miss += impacts.miss;
        sum_imp.fp += impacts.fp;
        sum_imp.fn_impact += impacts.fn_impact;
        active_classes += 1;

        per_class.push(TideClassBreakdown {
            class: c as u32,
            num_truths: n_gt,
            num_preds: dets.len() as u64,
            ap_base,
            counts,
            impacts,
        });
    }

    let (map_base, impacts) = if active_classes == 0 {
        (0.0, TideErrorImpacts::default())
    } else {
        let inv = 1.0 / active_classes as f64;
        (
            sum_base * inv,
            TideErrorImpacts {
                cls: sum_imp.cls * inv,
                loc: sum_imp.loc * inv,
                both: sum_imp.both * inv,
                dup: sum_imp.dup * inv,
                bkg: sum_imp.bkg * inv,
                miss: sum_imp.miss * inv,
                fp: sum_imp.fp * inv,
                fn_impact: sum_imp.fn_impact * inv,
            },
        )
    };

    Ok(TideDiagnostics {
        pos_iou: cfg.pos_iou,
        bg_iou: cfg.bg_iou,
        map_base,
        counts: total_counts,
        impacts,
        per_class,
    })
}
