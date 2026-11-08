//! Task-aligned assigner on small hand-checked cases.

use folo_vision::Xyxy;
use folo_vision::assign::{TalConfig, assign, dfl_targets};
use folo_vision::decode::make_anchors;

fn close(a: f32, b: f64, tol: f64) -> bool {
    (a as f64 - b).abs() <= tol
}

#[test]
fn alignment_and_normalisation() {
    // Anchors along y = 5; the gt (0,0)-(20,10) contains the first two centres. The one at x = 20 sits on the edge
    // and is not a candidate (strict inside test).
    let anchors = [[5.0, 5.0], [15.0, 5.0], [25.0, 5.0], [35.0, 5.0], [20.0, 5.0]];
    let gt = Xyxy::new(0.0, 0.0, 20.0, 10.0);
    let preds = [gt, Xyxy::new(5.0, 0.0, 25.0, 10.0), gt, gt, gt];
    // Two classes; the gt is class 1.
    let scores = [0.0, 0.25, 0.0, 0.81, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9];
    let a = assign(&scores, 2, &preds, &anchors, &[gt], &[1], &TalConfig::default());

    assert_eq!(a.fg_mask, [true, true, false, false, false]);
    assert_eq!(a.num_fg, 2);
    // Anchor 0: CIoU 1, align = 0.25^0.5 * 1 = 0.5.
    // Anchor 1: shifted copy, IoU 150/250 = 0.6; centres 5 apart, enclosing 25x10: CIoU = 0.6 - 25/725 (v = 0).
    //           align = 0.81^0.5 * CIoU^6.
    let ov1 = 0.6f64 - 25.0 / 725.0;
    let align0 = 0.5f64;
    let align1 = 0.9 * ov1.powi(6);
    // Normalised: align * max_overlap / max_align, with max_overlap = 1 and max_align = align0.
    assert!(close(a.target_scores[0], 1.0, 1e-6));
    assert!(
        close(a.target_scores[1], align1 / align0, 1e-6),
        "{} vs {}",
        a.target_scores[1],
        align1 / align0
    );
    assert_eq!(&a.target_labels[..2], &[1, 1]);
    assert_eq!(a.target_bboxes[1], gt);
    // Background entries.
    assert_eq!((a.target_labels[2], a.target_scores[2], a.target_bboxes[2]), (2, 0.0, Xyxy::default()));
    let dense = a.dense_scores(2);
    assert_eq!(dense.len(), 10);
    assert_eq!((dense[0], dense[2]), (0.0, 0.0));
    assert!(close(dense[1], 1.0, 1e-6));
}

#[test]
fn topk_keeps_best_aligned() {
    // 12 anchors inside one gt, all predicting the gt exactly (CIoU 1); align = sqrt(score) with rising scores.
    let gt = Xyxy::new(0.0, 0.0, 120.0, 10.0);
    let anchors: Vec<[f32; 2]> = (0..12).map(|i| [5.0 + 10.0 * i as f32, 5.0]).collect();
    let preds = vec![gt; 12];
    let scores: Vec<f32> = (0..12).map(|i| (i + 1) as f32 / 12.0).collect();
    let a = assign(&scores, 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.num_fg, 10);
    assert!(!a.fg_mask[0] && !a.fg_mask[1] && a.fg_mask[2..].iter().all(|&f| f));
    for i in 2..12 {
        let want = ((i + 1) as f64 / 12.0).sqrt() / 1.0f64.sqrt();
        assert!(close(a.target_scores[i], want, 1e-6), "anchor {i}");
    }
    // topk = 3 with equal scores: ties go to the lowest anchor indices.
    let flat = vec![0.5; 12];
    let cfg = TalConfig { topk: 3, ..TalConfig::default() };
    let a = assign(&flat, 1, &preds, &anchors, &[gt], &[0], &cfg);
    assert_eq!(a.fg_mask.iter().positions(), vec![0, 1, 2]);
}

#[test]
fn conflicts_go_to_highest_overlap() {
    // Anchor 1 at (15, 10) lies in both gts and predicts gt B exactly, so it is assigned to B.
    let ga = Xyxy::new(0.0, 0.0, 20.0, 20.0);
    let gb = Xyxy::new(10.0, 0.0, 30.0, 20.0);
    let anchors = [[5.0, 10.0], [15.0, 10.0], [25.0, 10.0]];
    let preds = [ga, gb, gb];
    let scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let a = assign(&scores, 2, &preds, &anchors, &[ga, gb], &[0, 1], &TalConfig::default());
    assert_eq!(a.fg_mask, [true, true, true]);
    assert_eq!(a.target_gt, [0, 1, 1]);
    assert_eq!(a.target_labels, [0, 1, 1]);
    // Both of B's positives have CIoU 1 and equal scores: normalised score 1 each.
    assert!(a.target_scores.iter().all(|&s| close(s, 1.0, 1e-6)));
}

#[test]
fn zero_alignment_candidates_are_kept_with_zero_score() {
    // 12 disjoint gts with one anchor each; every prediction misses every gt, so all alignments are 0. FOLO ranks
    // only candidates (documented parity exception), so each gt keeps its single anchor, with target score 0.
    let gts: Vec<Xyxy> =
        (0..12).map(|i| Xyxy::new(10.0 * i as f32, 0.0, 10.0 * i as f32 + 10.0, 10.0)).collect();
    let anchors: Vec<[f32; 2]> = (0..12).map(|i| [10.0 * i as f32 + 5.0, 5.0]).collect();
    let preds = vec![Xyxy::new(500.0, 500.0, 510.0, 510.0); 12];
    let labels = vec![0u32; 12];
    let a = assign(&[0.9; 12], 1, &preds, &anchors, &gts, &labels, &TalConfig::default());
    assert_eq!(a.num_fg, 12);
    assert_eq!(a.target_gt, (0..12).collect::<Vec<_>>());
    assert!(a.target_scores.iter().all(|&s| s == 0.0));
    assert!(a.dense_scores(1).iter().all(|&s| s == 0.0));
}

#[test]
fn no_gts_is_all_background() {
    let a = assign(
        &[0.9; 6],
        3,
        &[Xyxy::new(0.0, 0.0, 1.0, 1.0); 2],
        &[[0.5, 0.5]; 2],
        &[],
        &[],
        &TalConfig::default(),
    );
    assert_eq!(a.num_fg, 0);
    assert_eq!(a.target_labels, [3, 3]);
    assert_eq!(a.target_scores, [0.0, 0.0]);
}

#[test]
fn grid_assignment_and_dfl_targets() {
    // A 32x32 input at stride 8: a 4x4 grid. The gt (0,0)-(16,16) contains the centres of cells (0,0), (1,0),
    // (0,1), (1,1) → anchors 0, 1, 4, 5.
    let anchors = make_anchors(32, 32, &[8], 0.5);
    let gt = Xyxy::new(0.0, 0.0, 16.0, 16.0);
    let preds = vec![gt; anchors.len()];
    let scores = vec![0.5; anchors.len()];
    let a = assign(&scores, 1, &preds, &anchors.points_px(), &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![0, 1, 4, 5]);
    let d = dfl_targets(&a, &anchors, 16);
    // Anchor 0 at cell (0.5, 0.5); the gt in cells is (0,0)-(2,2).
    assert_eq!(d[0], [0.5, 0.5, 1.5, 1.5]);
    assert_eq!(d[5], [1.5, 1.5, 0.5, 0.5]);
    assert_eq!(d[2], [0.0; 4]);
}

trait Positions {
    fn positions(self) -> Vec<usize>;
}

impl<'a, I: Iterator<Item = &'a bool>> Positions for I {
    fn positions(self) -> Vec<usize> {
        self.enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect()
    }
}
