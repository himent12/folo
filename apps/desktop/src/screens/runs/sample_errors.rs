//! Run Detail Per-Image Validation Error Explorer: greedy IoU-0.50 TP/FP/FN matching on loaded
//! `Sample` ground-truth vs predicted boxes at the current confidence threshold, plus filter pills
//! (`All`, `FP`, `FN`, `TP`) and worst-error sorting.

use crate::api::Sample;
use crate::screens::bench::{MATCH_IOU_THRESHOLD, box_iou};

/// Filter pill for a run's Validation samples panel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SampleErrorFilter {
    #[default]
    All,
    /// Samples with at least one false positive detection (FP > 0).
    Fp,
    /// Samples with at least one missed ground-truth box (FN > 0).
    Fn,
    /// Samples with at least one matched true positive detection (TP > 0).
    Tp,
}

impl SampleErrorFilter {
    pub fn matches(self, st: &SampleErrorStats) -> bool {
        match self {
            Self::All => true,
            Self::Fp => st.fp > 0,
            Self::Fn => st.fn_count > 0,
            Self::Tp => st.tp > 0,
        }
    }
}

/// Sort mode for a run's Validation samples panel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SampleSort {
    /// Server / fixed sample order.
    #[default]
    Default,
    /// Most total errors (`FP + FN`) first, breaking ties by `FN`, `FP`, lowest IoU, then index.
    WorstErrors,
    /// Most false positives (`FP`) first.
    MostFp,
    /// Most missed ground-truth boxes (`FN`) first.
    MostFn,
}

impl SampleSort {
    pub const ALL: [Self; 4] = [Self::Default, Self::WorstErrors, Self::MostFp, Self::MostFn];

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Order",
            Self::WorstErrors => "Worst",
            Self::MostFp => "Most FP",
            Self::MostFn => "Most FN",
        }
    }
}

/// Per-sample TP / FP / FN counts and mean matched IoU at the active confidence threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleErrorStats {
    pub tp: u32,
    pub fp: u32,
    pub fn_count: u32,
    pub mean_iou: Option<f32>,
}

impl SampleErrorStats {
    pub fn total_errors(&self) -> u32 {
        self.fp.saturating_add(self.fn_count)
    }

    pub fn has_errors(&self) -> bool {
        self.fp > 0 || self.fn_count > 0
    }
}

/// Number of loaded samples matching each error filter pill at the active confidence threshold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SampleFilterCounts {
    pub all: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub tp: usize,
}

#[inline]
fn valid_box(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
    x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite() && x2 > x1 && y2 > y1
}

/// Classifies a validation sample's predictions (`score >= min_conf`) against its ground truth
/// using greedy highest-confidence same-class matching at IoU ≥ 0.50 (reusing `bench::box_iou`).
pub fn analyze_sample(sample: &Sample, min_conf: f32) -> SampleErrorStats {
    let gt_valid: Vec<bool> = sample.truth.iter().map(|g| valid_box(g.x1, g.y1, g.x2, g.y2)).collect();
    let mut gt_matched = vec![false; sample.truth.len()];

    // Consider only predictions currently visible at `min_conf`.
    let mut kept: Vec<usize> = sample
        .pred
        .iter()
        .enumerate()
        .filter(|(_, p)| p.score.is_finite() && p.score >= min_conf)
        .map(|(i, _)| i)
        .collect();

    // Sort descending by score (canonicalizing -0.0 to +0.0, deterministic tie-break by index).
    kept.sort_by(|&a, &b| {
        let norm = |s: f32| if s == 0.0 { 0.0f32 } else { s };
        norm(sample.pred[b].score).total_cmp(&norm(sample.pred[a].score)).then_with(|| a.cmp(&b))
    });

    let mut tp = 0u32;
    let mut iou_sum = 0.0f64;
    for pred_idx in &kept {
        let p = &sample.pred[*pred_idx];
        let pr = (p.x1, p.y1, p.x2, p.y2);
        if !valid_box(pr.0, pr.1, pr.2, pr.3) {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut best_threshold = MATCH_IOU_THRESHOLD;
        for (gt_idx, gt) in sample.truth.iter().enumerate() {
            if !gt_valid[gt_idx] || gt.class != p.class || gt_matched[gt_idx] {
                continue;
            }
            let iou = box_iou(pr, (gt.x1, gt.y1, gt.x2, gt.y2));
            if iou >= best_threshold {
                best_threshold = iou;
                best = Some((gt_idx, iou));
            }
        }
        if let Some((gt_idx, iou)) = best {
            gt_matched[gt_idx] = true;
            tp += 1;
            iou_sum += iou;
        }
    }

    let valid_gt = gt_valid.iter().filter(|&&v| v).count() as u32;
    let fn_count = valid_gt.saturating_sub(tp);
    let fp = (kept.len() as u32).saturating_sub(tp);
    let mean_iou = (tp > 0).then(|| (iou_sum / tp as f64) as f32);

    SampleErrorStats { tp, fp, fn_count, mean_iou }
}

/// Counts how many analyzed samples match each filter pill (`All`, `FP`, `FN`, `TP`).
pub fn count_sample_filters(analyzed: &[(usize, &Sample, SampleErrorStats)]) -> SampleFilterCounts {
    let mut c = SampleFilterCounts { all: analyzed.len(), ..Default::default() };
    for (_, _, st) in analyzed {
        if st.fp > 0 {
            c.fp += 1;
        }
        if st.fn_count > 0 {
            c.fn_count += 1;
        }
        if st.tp > 0 {
            c.tp += 1;
        }
    }
    c
}

/// Filters and sorts already-analyzed validation samples according to `filter` and `sort`.
pub fn filter_and_sort_samples(
    analyzed: Vec<(usize, &Sample, SampleErrorStats)>,
    filter: SampleErrorFilter,
    sort: SampleSort,
) -> Vec<(usize, &Sample, SampleErrorStats)> {
    let mut out: Vec<_> = analyzed.into_iter().filter(|(_, _, st)| filter.matches(st)).collect();
    let iou_key = |st: &SampleErrorStats| -> f32 {
        match st.mean_iou {
            Some(v) => v,
            None if st.has_errors() => 0.0,
            None => 2.0,
        }
    };
    match sort {
        SampleSort::Default => {}
        SampleSort::WorstErrors => {
            out.sort_by(|a, b| {
                b.2.total_errors()
                    .cmp(&a.2.total_errors())
                    .then_with(|| b.2.fn_count.cmp(&a.2.fn_count))
                    .then_with(|| b.2.fp.cmp(&a.2.fp))
                    .then_with(|| iou_key(&a.2).total_cmp(&iou_key(&b.2)))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        SampleSort::MostFp => {
            out.sort_by(|a, b| {
                b.2.fp
                    .cmp(&a.2.fp)
                    .then_with(|| b.2.total_errors().cmp(&a.2.total_errors()))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        SampleSort::MostFn => {
            out.sort_by(|a, b| {
                b.2.fn_count
                    .cmp(&a.2.fn_count)
                    .then_with(|| b.2.total_errors().cmp(&a.2.total_errors()))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
    }
    out
}

/// Returns the filtered and sorted clone of `samples` at `min_conf` (shared by the samples grid
/// and the image inspector's frozen sample snapshot).
pub fn ordered_filtered_samples(
    samples: &[Sample],
    min_conf: f32,
    filter: SampleErrorFilter,
    sort: SampleSort,
) -> Vec<Sample> {
    let analyzed: Vec<(usize, &Sample, SampleErrorStats)> =
        samples.iter().enumerate().map(|(i, s)| (i, s, analyze_sample(s, min_conf))).collect();
    filter_and_sort_samples(analyzed, filter, sort).into_iter().map(|(_, s, _)| s.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{BBox, ScoredBox};

    fn gt(class: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { class, x1, y1, x2, y2 }
    }

    fn pred(class: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> ScoredBox {
        ScoredBox { class, name: format!("c{class}"), score, x1, y1, x2, y2 }
    }

    fn sample(id: &str, truth: Vec<BBox>, pred: Vec<ScoredBox>) -> Sample {
        Sample { image_id: id.into(), dataset_id: "ds_1".into(), width: 640, height: 480, truth, pred }
    }

    #[test]
    fn analyzes_tp_fp_fn_and_reacts_to_confidence_threshold() {
        let s = sample(
            "img_a",
            vec![
                gt(0, 0.0, 0.0, 100.0, 100.0),     // matched at conf 0.25, missed at conf 0.80
                gt(1, 200.0, 200.0, 300.0, 300.0), // always missed (FN)
            ],
            vec![
                pred(0, 0.60, 0.0, 0.0, 100.0, 100.0), // TP at 0.25, filtered out at 0.80
                pred(0, 0.50, 0.0, 0.0, 100.0, 100.0), // duplicate -> FP at 0.25
                pred(0, 0.90, 200.0, 200.0, 300.0, 300.0), // wrong class -> always FP
            ],
        );

        let st_low = analyze_sample(&s, 0.25);
        assert_eq!((st_low.tp, st_low.fp, st_low.fn_count), (1, 2, 1));
        assert!((st_low.mean_iou.unwrap() - 1.0).abs() < 1e-5);

        let st_high = analyze_sample(&s, 0.80);
        assert_eq!((st_high.tp, st_high.fp, st_high.fn_count), (0, 1, 2));
        assert!(st_high.mean_iou.is_none());
    }

    #[test]
    fn handles_nan_and_degenerate_boxes_safely() {
        let s = sample(
            "nan",
            vec![gt(0, 0.0, 0.0, 100.0, 100.0)],
            vec![
                pred(0, 0.9, f32::NAN, 0.0, 100.0, 100.0),
                pred(0, f32::NAN, 0.0, 0.0, 100.0, 100.0),
                pred(0, 0.8, 50.0, 50.0, 50.0, 100.0),
            ],
        );
        let st = analyze_sample(&s, 0.25);
        assert_eq!((st.tp, st.fp, st.fn_count), (0, 2, 1));
    }

    #[test]
    fn filters_and_sorts_samples_by_error_modes() {
        let clean = sample("clean", vec![gt(0, 0., 0., 100., 100.)], vec![pred(0, 0.9, 0., 0., 100., 100.)]);
        let only_fn = sample("only_fn", vec![gt(0, 0., 0., 50., 50.), gt(0, 100., 100., 150., 150.)], vec![]);
        let only_fp = sample(
            "only_fp",
            vec![],
            vec![
                pred(0, 0.9, 0., 0., 50., 50.),
                pred(0, 0.8, 60., 60., 110., 110.),
                pred(0, 0.7, 120., 120., 170., 170.),
            ],
        );
        let mixed = sample(
            "mixed",
            vec![gt(0, 0., 0., 100., 100.), gt(1, 200., 200., 300., 300.)],
            vec![pred(0, 0.9, 0., 0., 100., 100.), pred(0, 0.7, 400., 400., 450., 450.)],
        );

        let all = [clean, only_fn, only_fp, mixed];
        let analyzed =
            || all.iter().enumerate().map(|(i, s)| (i, s, analyze_sample(s, 0.25))).collect::<Vec<_>>();

        let counts = count_sample_filters(&analyzed());
        assert_eq!(counts, SampleFilterCounts { all: 4, fp: 2, fn_count: 2, tp: 2 });

        let worst: Vec<&str> =
            filter_and_sort_samples(analyzed(), SampleErrorFilter::All, SampleSort::WorstErrors)
                .into_iter()
                .map(|(_, s, _)| s.image_id.as_str())
                .collect();
        assert_eq!(worst, vec!["only_fp", "only_fn", "mixed", "clean"]);

        let fp_only: Vec<&str> =
            filter_and_sort_samples(analyzed(), SampleErrorFilter::Fp, SampleSort::MostFp)
                .into_iter()
                .map(|(_, s, _)| s.image_id.as_str())
                .collect();
        assert_eq!(fp_only, vec!["only_fp", "mixed"]);

        let fn_only: Vec<&str> =
            filter_and_sort_samples(analyzed(), SampleErrorFilter::Fn, SampleSort::MostFn)
                .into_iter()
                .map(|(_, s, _)| s.image_id.as_str())
                .collect();
        assert_eq!(fn_only, vec!["only_fn", "mixed"]);

        let tp_only: Vec<&str> =
            filter_and_sort_samples(analyzed(), SampleErrorFilter::Tp, SampleSort::Default)
                .into_iter()
                .map(|(_, s, _)| s.image_id.as_str())
                .collect();
        assert_eq!(tp_only, vec!["clean", "mixed"]);
    }
}
