//! Benchmark entry image error analysis: greedy IoU-0.50 TP/FP/FN classification, per-image
//! localization IoU, filter chips, and error-focused sorting computed directly from
//! `BenchEntryImage` without backend API changes.

use crate::api::{BBox, BenchDetection, BenchEntryImage};

/// IoU threshold used for matching predictions to ground truth (matches `folo-bench` outcome rules).
pub const MATCH_IOU_THRESHOLD: f32 = 0.50;

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
    /// Most missed ground-truth boxes (FN) first.
    MostFn,
    /// Most false-positive detections (FP) first.
    MostFp,
    /// Lowest mean matched IoU first (images with errors or poor localization first).
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

/// Axis-aligned Intersection-over-Union between two `[x1, y1, x2, y2]` boxes.
pub fn box_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 || !inter.is_finite() {
        return 0.0;
    }
    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 || !union.is_finite() { 0.0 } else { (inter / union).clamp(0.0, 1.0) }
}

fn gt_rect(b: &BBox) -> (f32, f32, f32, f32) {
    (b.x1, b.y1, b.x2, b.y2)
}

fn det_rect(d: &BenchDetection) -> (f32, f32, f32, f32) {
    (d.x1, d.y1, d.x2, d.y2)
}

/// Classifies every detection (TP/FP) and ground-truth box (matched/FN) in `im` using standard
/// greedy highest-confidence same-class matching at IoU ≥ 0.50.
pub fn analyze_entry_image(im: &BenchEntryImage) -> ImageErrorStats {
    let n_gt = im.image.boxes.len();
    let n_det = im.detections.len();
    let mut gt_matched = vec![false; n_gt];
    let mut det_matches = vec![DetMatch::Fp { best_same_class_iou: 0.0 }; n_det];

    // Sort detection indices by descending score (stable on input order for ties/unscored entries).
    let mut order: Vec<usize> = (0..n_det).collect();
    order.sort_by(|&a, &b| {
        let sa = if im.detections[a].score.is_finite() { im.detections[a].score } else { -1.0 };
        let sb = if im.detections[b].score.is_finite() { im.detections[b].score } else { -1.0 };
        sb.total_cmp(&sa).then_with(|| a.cmp(&b))
    });

    let mut tp = 0u32;
    let mut iou_sum = 0.0f32;
    for det_idx in order {
        let d = &im.detections[det_idx];
        let dr = det_rect(d);
        let mut best_unmatched: Option<(usize, f32)> = None;
        let mut best_any_same = 0.0f32;
        for (gt_idx, gt) in im.image.boxes.iter().enumerate() {
            if gt.class != d.class {
                continue;
            }
            let iou = box_iou(dr, gt_rect(gt));
            if iou > best_any_same {
                best_any_same = iou;
            }
            if !gt_matched[gt_idx]
                && iou >= MATCH_IOU_THRESHOLD
                && best_unmatched.is_none_or(|(_, best)| iou > best)
            {
                best_unmatched = Some((gt_idx, iou));
            }
        }
        if let Some((gt_idx, iou)) = best_unmatched {
            gt_matched[gt_idx] = true;
            det_matches[det_idx] = DetMatch::Tp { gt_index: gt_idx, iou };
            tp += 1;
            iou_sum += iou;
        } else {
            det_matches[det_idx] = DetMatch::Fp { best_same_class_iou: best_any_same };
        }
    }

    let fn_count = gt_matched.iter().filter(|&&m| !m).count() as u32;
    let fp = (n_det as u32).saturating_sub(tp);
    let mean_iou = (tp > 0).then(|| iou_sum / tp as f32);

    ImageErrorStats { tp, fp, fn_count, mean_iou, det_matches, gt_matched }
}

/// Counts how many loaded images match each error-analysis filter chip.
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

/// Filters and sorts loaded entry images according to `filter` and `order`.
pub fn filter_and_sort_entry_images(
    images: &[BenchEntryImage],
    filter: EntryImageFilter,
    order: EntryImageOrder,
) -> Vec<(usize, &BenchEntryImage, ImageErrorStats)> {
    let mut out: Vec<(usize, &BenchEntryImage, ImageErrorStats)> = images
        .iter()
        .enumerate()
        .map(|(i, im)| (i, im, analyze_entry_image(im)))
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
            // Images with no TP despite having errors sort first (effective IoU 0.0), then ascending mean IoU,
            // and clean background images (no GT, no detections) sort last.
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
    fn filters_and_orders_entry_images_accurately() {
        let clean = entry_img(
            "clean",
            vec![gt(0, 0., 0., 100., 100.)],
            vec![det(0, 0.9, 0., 0., 100., 100.)], // TP=1, FP=0, FN=0, IoU=1.0
        );
        let low_iou = entry_img(
            "low_iou",
            vec![gt(0, 0., 0., 100., 100.)],
            vec![det(0, 0.9, 25., 0., 125., 100.)], // IoU = 75/125 = 0.60, TP=1, FP=0, FN=0
        );
        let only_fn = entry_img(
            "only_fn",
            vec![gt(0, 0., 0., 50., 50.), gt(0, 100., 100., 150., 150.)],
            vec![], // TP=0, FP=0, FN=2
        );
        let only_fp = entry_img(
            "only_fp",
            vec![],
            vec![
                det(0, 0.9, 0., 0., 50., 50.),
                det(0, 0.8, 60., 60., 110., 110.),
                det(0, 0.7, 120., 120., 170., 170.),
            ], // TP=0, FP=3, FN=0
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
