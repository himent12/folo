//! Benchmark entry image error analysis: greedy IoU-0.50 TP/FP/FN classification matching
//! `folo-bench` semantics (coordinate validation, `f64` IoU, `>=` tie-breaking), per-image
//! localization IoU, filter chips, and error-focused sorting computed directly from
//! `BenchEntryImage` without backend API changes.

use crate::api::{BBox, BenchDetection, BenchEntryImage};

/// IoU threshold used for matching predictions to ground truth (matches `folo-bench` outcome rules).
pub const MATCH_IOU_THRESHOLD: f64 = 0.50;

/// Filter chip for a benchmark entry's Images panel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum EntryImageFilter {
    #[default]
    All,
    /// Images with at least one missed ground-truth box (FN > 0).
    MissedGt,
    /// Images with at least one false-positive detection (FP > 0).
    FalsePositives,
    /// Images with any error (FN > 0 or FP > 0).
    HasErrors,
}

/// Sort mode for a benchmark entry's Images panel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum EntryImageOrder {
    /// Most total errors (`missed + wrong`) first (backed by server `worst` sort).
    #[default]
    MostErrors,
    /// Most missed ground-truth boxes (FN) first (within loaded images).
    MostFn,
    /// Most false-positive detections (FP) first (within loaded images).
    MostFp,
    /// Lowest mean matched IoU first (within loaded images).
    LowestIou,
    /// Original benchmark order (backed by server `order` sort).
    InOrder,
}

impl EntryImageOrder {
    /// Server-side sort parameter to request for this ordering.
    pub fn server_sort(self) -> crate::bench::ImageSort {
        match self {
            EntryImageOrder::InOrder => crate::bench::ImageSort::Order,
            _ => crate::bench::ImageSort::Worst,
        }
    }

    /// True when this order re-sorts only the loaded prefix client-side.
    pub fn is_client_subsort(self) -> bool {
        matches!(self, EntryImageOrder::MostFn | EntryImageOrder::MostFp | EntryImageOrder::LowestIou)
    }
}

/// Per-detection classification after greedy same-class IoU ≥ 0.50 matching.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetMatch {
    Tp { gt_index: usize, iou: f32 },
    Fp { best_same_class_iou: f32 },
}

impl DetMatch {
    pub fn is_tp(self) -> bool {
        matches!(self, DetMatch::Tp { .. })
    }
}

/// Per-image TP / FP / FN counts, mean matched IoU, and per-box match status.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageErrorStats {
    pub tp: u32,
    pub fp: u32,
    pub fn_count: u32,
    pub mean_iou: Option<f32>,
    /// Match status indexed by the image's `detections` slice order.
    pub det_matches: Vec<DetMatch>,
    /// Whether each ground-truth box (indexed by `image.boxes`) was matched by a detection.
    pub gt_matched: Vec<bool>,
}

impl ImageErrorStats {
    pub fn total_errors(&self) -> u32 {
        self.fp.saturating_add(self.fn_count)
    }

    pub fn has_errors(&self) -> bool {
        self.fp > 0 || self.fn_count > 0
    }
}

/// Counts of loaded images matching each filter chip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterCounts {
    pub all: usize,
    pub missed_gt: usize,
    pub false_positives: usize,
    pub has_errors: usize,
}

#[inline]
fn valid_box(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
    x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite() && x2 > x1 && y2 > y1
}

/// Axis-aligned Intersection-over-Union in `f64` matching `folo-bench::score::iou_with_areas`.
/// Returns `0.0` if any coordinate is non-finite or either box has non-positive area.
pub fn box_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f64 {
    if !valid_box(a.0, a.1, a.2, a.3) || !valid_box(b.0, b.1, b.2, b.3) {
        return 0.0;
    }
    let w = (a.2 as f64).min(b.2 as f64) - (a.0 as f64).max(b.0 as f64);
    if w <= 0.0 {
        return 0.0;
    }
    let h = (a.3 as f64).min(b.3 as f64) - (a.1 as f64).max(b.1 as f64);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let area_a = (a.2 as f64 - a.0 as f64) * (a.3 as f64 - a.1 as f64);
    let area_b = (b.2 as f64 - b.0 as f64) * (b.3 as f64 - b.1 as f64);
    let union = area_a + area_b - inter;
    if union <= 0.0 || !union.is_finite() { 0.0 } else { (inter / union).clamp(0.0, 1.0) }
}

fn gt_rect(b: &BBox) -> (f32, f32, f32, f32) {
    (b.x1, b.y1, b.x2, b.y2)
}

fn det_rect(d: &BenchDetection) -> (f32, f32, f32, f32) {
    (d.x1, d.y1, d.x2, d.y2)
}

/// Classifies every detection (TP/FP) and ground-truth box (matched/FN) in `im` using `folo-bench`'s
/// exact greedy highest-confidence same-class matching at IoU ≥ 0.50 (`f64` IoU, `>=` tie-break).
pub fn analyze_entry_image(im: &BenchEntryImage) -> ImageErrorStats {
    let n_gt = im.image.boxes.len();
    let n_det = im.detections.len();
    let mut gt_matched = vec![false; n_gt];
    let mut det_matches = vec![DetMatch::Fp { best_same_class_iou: 0.0 }; n_det];

    let gt_valid: Vec<bool> = im.image.boxes.iter().map(|g| valid_box(g.x1, g.y1, g.x2, g.y2)).collect();

    // Sort detection indices by descending score (canonicalizing `-0.0` to `+0.0`, stable on ties).
    let mut order: Vec<usize> = (0..n_det).collect();
    order.sort_by(|&a, &b| {
        let norm = |s: f32| {
            if !s.is_finite() {
                -1.0f32
            } else if s == 0.0 {
                0.0f32
            } else {
                s
            }
        };
        norm(im.detections[b].score).total_cmp(&norm(im.detections[a].score)).then_with(|| a.cmp(&b))
    });

    let mut tp = 0u32;
    let mut iou_sum = 0.0f64;
    for det_idx in order {
        let d = &im.detections[det_idx];
        let dr = det_rect(d);
        if !d.score.is_finite() || !valid_box(dr.0, dr.1, dr.2, dr.3) {
            continue;
        }
        let mut best_unmatched: Option<(usize, f64)> = None;
        let mut best_threshold = MATCH_IOU_THRESHOLD;
        let mut best_any_same = 0.0f64;
        for (gt_idx, gt) in im.image.boxes.iter().enumerate() {
            if !gt_valid[gt_idx] || gt.class != d.class {
                continue;
            }
            let iou = box_iou(dr, gt_rect(gt));
            if iou > best_any_same {
                best_any_same = iou;
            }
            // Use `>=` to match `folo_bench::score::match_image_into` exact tie-breaking.
            if !gt_matched[gt_idx] && iou >= best_threshold {
                best_threshold = iou;
                best_unmatched = Some((gt_idx, iou));
            }
        }
        if let Some((gt_idx, iou)) = best_unmatched {
            gt_matched[gt_idx] = true;
            det_matches[det_idx] = DetMatch::Tp { gt_index: gt_idx, iou: iou as f32 };
            tp += 1;
            iou_sum += iou;
        } else {
            det_matches[det_idx] = DetMatch::Fp { best_same_class_iou: best_any_same as f32 };
        }
    }

    let valid_gt_count = gt_valid.iter().filter(|&&v| v).count() as u32;
    let fn_count = valid_gt_count.saturating_sub(tp);
    let fp = (n_det as u32).saturating_sub(tp);
    let mean_iou = (tp > 0).then(|| (iou_sum / tp as f64) as f32);

    ImageErrorStats { tp, fp, fn_count, mean_iou, det_matches, gt_matched }
}

/// Returns whether an entry image matches `filter` (shared by rendering and `Esc` handling).
pub fn entry_image_matches_filter(im: &BenchEntryImage, filter: EntryImageFilter) -> bool {
    match filter {
        EntryImageFilter::All => true,
        EntryImageFilter::MissedGt => analyze_entry_image(im).fn_count > 0,
        EntryImageFilter::FalsePositives => analyze_entry_image(im).fp > 0,
        EntryImageFilter::HasErrors => analyze_entry_image(im).has_errors(),
    }
}

/// Counts how many loaded images match each error-analysis filter chip.
#[cfg(test)]
pub fn count_filter_matches(images: &[BenchEntryImage]) -> FilterCounts {
    let mut counts = FilterCounts { all: images.len(), ..Default::default() };
    for im in images {
        let st = analyze_entry_image(im);
        if st.fn_count > 0 {
            counts.missed_gt += 1;
        }
        if st.fp > 0 {
            counts.false_positives += 1;
        }
        if st.has_errors() {
            counts.has_errors += 1;
        }
    }
    counts
}

/// Filters and sorts already-analyzed entry images according to `filter` and `order`.
pub fn filter_and_sort_analyzed<'a>(
    analyzed: Vec<(usize, &'a BenchEntryImage, ImageErrorStats)>,
    filter: EntryImageFilter,
    order: EntryImageOrder,
) -> Vec<(usize, &'a BenchEntryImage, ImageErrorStats)> {
    let mut out: Vec<(usize, &'a BenchEntryImage, ImageErrorStats)> = analyzed
        .into_iter()
        .filter(|(_, _, st)| match filter {
            EntryImageFilter::All => true,
            EntryImageFilter::MissedGt => st.fn_count > 0,
            EntryImageFilter::FalsePositives => st.fp > 0,
            EntryImageFilter::HasErrors => st.has_errors(),
        })
        .collect();

    match order {
        EntryImageOrder::InOrder => {}
        EntryImageOrder::MostErrors => {
            out.sort_by(|a, b| {
                b.2.total_errors()
                    .cmp(&a.2.total_errors())
                    .then_with(|| b.2.fn_count.cmp(&a.2.fn_count))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        EntryImageOrder::MostFn => {
            out.sort_by(|a, b| {
                b.2.fn_count
                    .cmp(&a.2.fn_count)
                    .then_with(|| b.2.total_errors().cmp(&a.2.total_errors()))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        EntryImageOrder::MostFp => {
            out.sort_by(|a, b| {
                b.2.fp
                    .cmp(&a.2.fp)
                    .then_with(|| b.2.total_errors().cmp(&a.2.total_errors()))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
        EntryImageOrder::LowestIou => {
            let key = |st: &ImageErrorStats| -> f32 {
                match st.mean_iou {
                    Some(v) => v,
                    None if st.has_errors() => 0.0,
                    None => 2.0,
                }
            };
            out.sort_by(|a, b| {
                key(&a.2)
                    .total_cmp(&key(&b.2))
                    .then_with(|| b.2.total_errors().cmp(&a.2.total_errors()))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }
    }

    out
}

/// Counts how many already-analyzed images match each error-analysis filter chip.
pub fn count_analyzed_filters(analyzed: &[(usize, &BenchEntryImage, ImageErrorStats)]) -> FilterCounts {
    let mut counts = FilterCounts { all: analyzed.len(), ..Default::default() };
    for (_, _, st) in analyzed {
        if st.fn_count > 0 {
            counts.missed_gt += 1;
        }
        if st.fp > 0 {
            counts.false_positives += 1;
        }
        if st.has_errors() {
            counts.has_errors += 1;
        }
    }
    counts
}

/// Filters and sorts loaded entry images according to `filter` and `order`.
#[cfg(test)]
pub fn filter_and_sort_entry_images(
    images: &[BenchEntryImage],
    filter: EntryImageFilter,
    order: EntryImageOrder,
) -> Vec<(usize, &BenchEntryImage, ImageErrorStats)> {
    let analyzed: Vec<_> =
        images.iter().enumerate().map(|(i, im)| (i, im, analyze_entry_image(im))).collect();
    filter_and_sort_analyzed(analyzed, filter, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::BenchImage;

    fn gt(class: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { class, x1, y1, x2, y2 }
    }

    fn det(class: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> BenchDetection {
        BenchDetection { class, score, x1, y1, x2, y2 }
    }

    fn entry_img(id: &str, gts: Vec<BBox>, dets: Vec<BenchDetection>) -> BenchEntryImage {
        let mut im = BenchEntryImage {
            image: BenchImage {
                id: id.into(),
                file: format!("{id}.jpg"),
                width: 640,
                height: 480,
                boxes: gts,
            },
            detections: dets,
            missed: 0,
            wrong: 0,
        };
        let st = analyze_entry_image(&im);
        im.missed = st.fn_count;
        im.wrong = st.fp;
        im
    }

    #[test]
    fn greedy_matching_classifies_tp_fp_fn_and_duplicates() {
        let im = entry_img(
            "img_1",
            vec![
                gt(0, 0.0, 0.0, 100.0, 100.0),     // matched by d0 (IoU = 1.0)
                gt(1, 200.0, 200.0, 300.0, 300.0), // missed (FN)
            ],
            vec![
                det(0, 0.95, 0.0, 0.0, 100.0, 100.0),     // TP
                det(0, 0.80, 0.0, 0.0, 100.0, 100.0),     // duplicate on same GT -> FP
                det(0, 0.70, 200.0, 200.0, 300.0, 300.0), // wrong class on GT 1 -> FP
            ],
        );
        let st = analyze_entry_image(&im);
        assert_eq!((st.tp, st.fp, st.fn_count), (1, 2, 1));
        assert_eq!(st.gt_matched, vec![true, false]);
        assert!(matches!(st.det_matches[0], DetMatch::Tp { gt_index: 0, iou } if (iou - 1.0).abs() < 1e-5));
        assert!(
            matches!(st.det_matches[1], DetMatch::Fp { best_same_class_iou } if best_same_class_iou > 0.99)
        );
        assert!(
            matches!(st.det_matches[2], DetMatch::Fp { best_same_class_iou } if best_same_class_iou == 0.0)
        );
        assert!((st.mean_iou.unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nan_and_degenerate_boxes_never_match_and_ties_match_folo_bench() {
        // NaN coordinate must yield IoU 0.0, never 1.0.
        assert_eq!(box_iou((f32::NAN, 0.0, 100.0, 100.0), (0.0, 0.0, 200.0, 100.0)), 0.0);
        assert_eq!(box_iou((0.0, 0.0, 0.0, 100.0), (0.0, 0.0, 100.0, 100.0)), 0.0);
        let nan_im = entry_img(
            "nan",
            vec![gt(0, 0.0, 0.0, 200.0, 100.0)],
            vec![det(0, 0.9, f32::NAN, 0.0, 100.0, 100.0)],
        );
        let nan_st = analyze_entry_image(&nan_im);
        assert_eq!((nan_st.tp, nan_st.fp, nan_st.fn_count), (0, 1, 1));
        // Finite coordinates with NaN score must also be rejected (never claim a GT box).
        let nan_score_im = entry_img(
            "nan_score",
            vec![gt(0, 0.0, 0.0, 100.0, 100.0)],
            vec![det(0, f32::NAN, 0.0, 0.0, 100.0, 100.0)],
        );
        let nan_score_st = analyze_entry_image(&nan_score_im);
        assert_eq!((nan_score_st.tp, nan_score_st.fp, nan_score_st.fn_count), (0, 1, 1));

        // Exact IoU tie (2/3 with both GT 0 and GT 1): `folo-bench` takes the last GT (GT 1),
        // leaving GT 0 free for the second detection -> 2 TP, 0 FP, 0 FN.
        let tie_im = entry_img(
            "tie",
            vec![gt(0, 0.0, 0.0, 100.0, 100.0), gt(0, 50.0, 0.0, 150.0, 100.0)],
            vec![
                det(0, 0.9, 0.0, 0.0, 150.0, 100.0), // ties at 2/3 with GT 0 and GT 1 -> takes GT 1
                det(0, 0.8, 0.0, 0.0, 100.0, 100.0), // matches GT 0 exactly
            ],
        );
        let tie_st = analyze_entry_image(&tie_im);
        assert_eq!((tie_st.tp, tie_st.fp, tie_st.fn_count), (2, 0, 0));
    }

    #[test]
    fn filters_and_orders_entry_images_accurately() {
        let clean =
            entry_img("clean", vec![gt(0, 0., 0., 100., 100.)], vec![det(0, 0.9, 0., 0., 100., 100.)]);
        let low_iou =
            entry_img("low_iou", vec![gt(0, 0., 0., 100., 100.)], vec![det(0, 0.9, 25., 0., 125., 100.)]);
        let only_fn =
            entry_img("only_fn", vec![gt(0, 0., 0., 50., 50.), gt(0, 100., 100., 150., 150.)], vec![]);
        let only_fp = entry_img(
            "only_fp",
            vec![],
            vec![
                det(0, 0.9, 0., 0., 50., 50.),
                det(0, 0.8, 60., 60., 110., 110.),
                det(0, 0.7, 120., 120., 170., 170.),
            ],
        );

        let all = vec![clean, low_iou, only_fn, only_fp];
        let counts = count_filter_matches(&all);
        assert_eq!(counts, FilterCounts { all: 4, missed_gt: 1, false_positives: 1, has_errors: 2 });

        let fn_list =
            filter_and_sort_entry_images(&all, EntryImageFilter::MissedGt, EntryImageOrder::MostErrors);
        assert_eq!(fn_list.iter().map(|x| x.1.image.id.as_str()).collect::<Vec<_>>(), vec!["only_fn"]);

        let fp_list =
            filter_and_sort_entry_images(&all, EntryImageFilter::FalsePositives, EntryImageOrder::MostFp);
        assert_eq!(fp_list.iter().map(|x| x.1.image.id.as_str()).collect::<Vec<_>>(), vec!["only_fp"]);

        let most_errors =
            filter_and_sort_entry_images(&all, EntryImageFilter::All, EntryImageOrder::MostErrors);
        assert_eq!(
            most_errors.iter().map(|x| x.1.image.id.as_str()).collect::<Vec<_>>(),
            vec!["only_fp", "only_fn", "clean", "low_iou"]
        );

        let most_fn = filter_and_sort_entry_images(&all, EntryImageFilter::All, EntryImageOrder::MostFn);
        assert_eq!(most_fn[0].1.image.id, "only_fn");

        let lowest_iou =
            filter_and_sort_entry_images(&all, EntryImageFilter::All, EntryImageOrder::LowestIou);
        assert_eq!(
            lowest_iou.iter().map(|x| x.1.image.id.as_str()).collect::<Vec<_>>(),
            vec!["only_fp", "only_fn", "low_iou", "clean"]
        );
    }
}
