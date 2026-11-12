//! Task-aligned assigner on small hand-checked cases, one group per rule of `docs/MODEL.md` §7.3 plus the assigner
//! cases of §10.3. All boxes and anchor points are in input pixels.

use folo_vision::Xyxy;
use folo_vision::assign::{TalConfig, assign, dfl_targets};
use folo_vision::boxes::ciou_eps;
use folo_vision::decode::make_anchors;

fn close(a: f32, b: f64, tol: f64) -> bool {
    (a as f64 - b).abs() <= tol
}

fn with_topk(topk: usize) -> TalConfig {
    TalConfig { topk, ..TalConfig::default() }
}

// ---- Rule 1: candidates (enlarged box, strict inside test) ----------------------------------------------------

#[test]
fn rule1_strict_inside_with_eps() {
    // A 16x16 gt is not enlarged. Anchors on its edges are not candidates; a point 1e-3 inside is.
    let gt = Xyxy::new(0.0, 0.0, 16.0, 16.0);
    let anchors = [[0.0, 8.0], [16.0, 8.0], [8.0, 0.0], [8.0, 16.0], [8.0, 8.0], [1e-3, 8.0], [8.0, 15.999]];
    let preds = vec![gt; anchors.len()];
    let a = assign(&[0.5; 7], 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![4, 5, 6]);
    // The margin is eps = 1e-9 itself, not 0: from the edge x1 = 0, only the point 2·eps inside qualifies.
    let anchors = [[5e-10, 8.0], [1e-9, 8.0], [2e-9, 8.0]];
    let a = assign(&[0.5; 3], 1, &[gt; 3], &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![2]);
}

#[test]
fn rule1_enlargement_is_about_the_centre_and_for_candidates_only() {
    // 6x6 gt centred at (7, 7): its candidate box is (-1, -1)-(15, 15).
    let gt = Xyxy::new(4.0, 4.0, 10.0, 10.0);
    // In the original box; outside it but inside the enlarged one; on the enlarged edge; outside both.
    let anchors = [[7.0, 7.0], [14.5, 7.0], [-0.5, -0.5], [15.0, 7.0], [7.0, -1.0], [16.0, 7.0]];
    let preds = vec![gt; anchors.len()];
    let a = assign(&[0.5; 6], 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![0, 1, 2]);
    // The box target is the original gt, not the enlarged box.
    assert!(a.fg_mask.iter().zip(&a.target_bboxes).filter(|(f, _)| **f).all(|(_, b)| *b == gt));
    // With the enlargement disabled only the anchor inside the original box remains.
    let off = TalConfig { min_box_size: 0.0, ..TalConfig::default() };
    let a = assign(&[0.5; 6], 1, &preds, &anchors, &[gt], &[0], &off);
    assert_eq!(a.fg_mask.iter().positions(), vec![0]);
}

#[test]
fn rule1_only_the_short_side_is_enlarged() {
    // 40 x 6 gt centred at (20, 20): candidate box (0, 12)-(40, 28). x is not widened.
    let gt = Xyxy::new(0.0, 17.0, 40.0, 23.0);
    let anchors = [[20.0, 13.0], [20.0, 27.0], [20.0, 11.0], [-1.0, 20.0], [41.0, 20.0], [39.0, 20.0]];
    let preds = vec![gt; anchors.len()];
    let a = assign(&[0.5; 6], 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![0, 1, 5]);
}

#[test]
fn spec_small_gt_gets_p3_candidates_and_original_box_targets() {
    // MODEL.md §10.3: a 6x6 px gt. On a 64x64 input (P3 centres 4 + 8k, P4 8 + 16k, P5 16 + 32k) the box
    // (28.5, 28.5)-(34.5, 34.5) contains no anchor centre of any level. Enlarged about (31.5, 31.5) to
    // (23.5, 23.5)-(39.5, 39.5) it contains the P3 centres {28, 36}² (cells (3|4, 3|4) → anchors 27, 28, 35, 36) and
    // the P4 centre (24, 24) (cell (1, 1) → anchor 64 + 5 = 69); P5 centres (16, 48) stay outside.
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let gt = Xyxy::new(28.5, 28.5, 34.5, 34.5);
    let preds = vec![gt; anchors.len()];
    let scores = vec![0.5; anchors.len()];
    let pts = anchors.points_px();
    let a = assign(&scores, 1, &preds, &pts, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask.iter().positions(), vec![27, 28, 35, 36, 69]);
    let off = TalConfig { min_box_size: 0.0, ..TalConfig::default() };
    assert_eq!(assign(&scores, 1, &preds, &pts, &[gt], &[0], &off).num_fg, 0);

    // Rule 6: box targets are the original gt divided by each anchor's own stride, clamped at 0 for the anchors
    // that only the enlarged box contains.
    let d = dfl_targets(&a, &anchors, 16);
    // Anchor 27: grid (3.5, 3.5), gt / 8 = (3.5625, 3.5625)-(4.3125, 4.3125).
    assert_eq!(d[27], [0.0, 0.0, 0.8125, 0.8125]);
    // Anchor 36: grid (4.5, 4.5).
    assert_eq!(d[36], [0.9375, 0.9375, 0.0, 0.0]);
    // Anchor 69: grid (1.5, 1.5) at stride 16, gt / 16 = (1.78125, 1.78125)-(2.15625, 2.15625).
    assert_eq!(d[69], [0.0, 0.0, 0.65625, 0.65625]);
    assert_eq!(d[0], [0.0; 4]);
}

#[test]
fn spec_100px_gt_candidates_on_all_levels() {
    // MODEL.md §10.3: a 100x100 px gt centred on the P3 cell centre (100, 100), on a 256x256 input. Its candidates
    // are exactly the anchor centres strictly inside (50, 150)²: P3 centres 52..148 (13 per axis, 169), P4 56..136
    // (6 per axis, 36), P5 80, 112, 144 (3 per axis, 9). 214 in all.
    let anchors = make_anchors(256, 256, &[8, 16, 32], 0.5);
    let gt = Xyxy::new(50.0, 50.0, 150.0, 150.0);
    let pts = anchors.points_px();
    // Predictions are the gt shifted by a few pixels, so overlaps differ (and repeat, exercising the tie rule).
    let preds: Vec<Xyxy> = (0..anchors.len()).map(|i| gt.translate((i % 7) as f32, (i % 5) as f32)).collect();
    let scores = vec![0.5; anchors.len()];
    let inside: Vec<usize> =
        (0..pts.len()).filter(|&i| pts[i].iter().all(|&v| v > 50.0 && v < 150.0)).collect();
    assert_eq!(inside.len(), 214);
    let all = assign(&scores, 1, &preds, &pts, &[gt], &[0], &with_topk(1000));
    assert_eq!(all.fg_mask.iter().positions(), inside);

    // Default top-k: 10 positives, the 10 largest overlaps (equal scores), ties to the smaller anchor index.
    let a = assign(&scores, 1, &preds, &pts, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.num_fg, 10);
    let u = |i: usize| ciou_eps(&gt, &preds[i]).max(0.0);
    let mut ranked = inside.clone();
    ranked.sort_by(|&x, &y| u(y).total_cmp(&u(x)).then(x.cmp(&y)));
    let mut want = ranked[..10].to_vec();
    want.sort();
    assert_eq!(a.fg_mask.iter().positions(), want);
    // max t = u_max.
    let t_max = a.target_scores.iter().copied().fold(0.0f32, f32::max);
    let u_max = want.iter().map(|&i| u(i)).fold(0.0f32, f32::max);
    assert!((t_max - u_max).abs() < 1e-6, "{t_max} vs {u_max}");
}

// ---- Rule 2: the metric ---------------------------------------------------------------------------------------

#[test]
fn rule2_alignment_and_rule5_normalisation() {
    // Anchors along y = 5; the gt (0,0)-(20,10) contains the first two centres. Its height 10 is enlarged to
    // (0,-3)-(20,13) for candidate selection, which changes nothing here; the anchor at x = 20 sits on the edge and
    // is not a candidate (strict inside test).
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
fn spec_ciou_identical_and_disjoint() {
    // MODEL.md §10.3: identical boxes give CIoU 1 up to the eps terms; disjoint boxes a negative CIoU.
    let b = Xyxy::new(0.0, 0.0, 20.0, 10.0);
    assert!(close(ciou_eps(&b, &b), 1.0, 1e-6));
    assert!(ciou_eps(&b, &Xyxy::new(30.0, 0.0, 50.0, 10.0)) < 0.0);
}

#[test]
fn rule2_and_rule3_zero_metric_candidates_are_eligible() {
    // 12 disjoint gts with one anchor each; every prediction misses every gt (negative CIoU, so u = m = 0). Each gt
    // still keeps its single candidate, with target score 0 (m_max = 0, so t = 0 / eps = 0). The 10x10 gts are
    // enlarged to 16x16 about their centres, which still contain only their own anchor.
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

// ---- Rule 3: top-k ----------------------------------------------------------------------------------------------

#[test]
fn rule3_topk_keeps_best_aligned() {
    // 12 anchors inside one gt, all predicting the gt exactly (CIoU 1); align = sqrt(score) with rising scores.
    let gt = Xyxy::new(0.0, 0.0, 120.0, 10.0);
    let anchors: Vec<[f32; 2]> = (0..12).map(|i| [5.0 + 10.0 * i as f32, 5.0]).collect();
    let preds = vec![gt; 12];
    let scores: Vec<f32> = (0..12).map(|i| (i + 1) as f32 / 12.0).collect();
    let a = assign(&scores, 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.num_fg, 10);
    assert!(!a.fg_mask[0] && !a.fg_mask[1] && a.fg_mask[2..].iter().all(|&f| f));
    for (i, &score) in a.target_scores.iter().enumerate().take(12).skip(2) {
        let want = ((i + 1) as f64 / 12.0).sqrt() / 1.0f64.sqrt();
        assert!(close(score, want, 1e-6), "anchor {i}");
    }
    // topk = 3 with equal scores: ties go to the lowest anchor indices.
    let a = assign(&[0.5; 12], 1, &preds, &anchors, &[gt], &[0], &with_topk(3));
    assert_eq!(a.fg_mask.iter().positions(), vec![0, 1, 2]);
    // topk = 0 selects nothing.
    assert_eq!(assign(&[0.5; 12], 1, &preds, &anchors, &[gt], &[0], &with_topk(0)).num_fg, 0);
}

#[test]
fn rule3_ties_break_by_anchor_index_in_any_order() {
    // Six candidates with metrics sqrt(score): 0.5, 0.9, 0.5, 0.9, 0.8, 0.5.
    let gt = Xyxy::new(0.0, 0.0, 60.0, 20.0);
    let anchors: Vec<[f32; 2]> = (0..6).map(|i| [5.0 + 10.0 * i as f32, 10.0]).collect();
    let preds = vec![gt; 6];
    let scores = [0.25, 0.81, 0.25, 0.81, 0.64, 0.25];
    let pick =
        |k| assign(&scores, 1, &preds, &anchors, &[gt], &[0], &with_topk(k)).fg_mask.iter().positions();
    assert_eq!(pick(1), vec![1]);
    assert_eq!(pick(3), vec![1, 3, 4]);
    assert_eq!(pick(4), vec![0, 1, 3, 4]);
    assert_eq!(pick(5), vec![0, 1, 2, 3, 4]);
    // min(top_k, #candidates).
    assert_eq!(pick(100), vec![0, 1, 2, 3, 4, 5]);
}

// ---- Rule 4: conflicts --------------------------------------------------------------------------------------------

#[test]
fn rule4_conflicts_go_to_highest_overlap() {
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
fn rule4_conflicts_resolve_only_among_selecting_gts() {
    // Gts A = (0,0)-(40,40) and B = (12,0)-(52,40), class 0; C = (5,0)-(45,40), class 1. Two anchors, X (25, 20) and
    // Y (30, 20), both inside all three gts, both predicting C exactly. top_k = 1.
    // Class scores: X = [0.9, 0.01], Y = [0.01, 0.9]. So A and B select X; C selects Y (sqrt(0.9) > sqrt(0.01)).
    // X is contested by A and B only:
    //   u_A(X) = CIoU(A, C) = 1400/1800 - 25/3625 = 0.770881,  u_B(X) = 1320/1880 - 49/3809 = 0.689264  → A.
    // Resolving over all gts (Ultralytics) would hand X to C (u_C(X) = 1), a gt that never selected it.
    let (ga, gb, gc) =
        (Xyxy::new(0.0, 0.0, 40.0, 40.0), Xyxy::new(12.0, 0.0, 52.0, 40.0), Xyxy::new(5.0, 0.0, 45.0, 40.0));
    let anchors = [[25.0, 20.0], [30.0, 20.0]];
    let preds = [gc, gc];
    let scores = [0.9, 0.01, 0.01, 0.9];
    let a = assign(&scores, 2, &preds, &anchors, &[ga, gb, gc], &[0, 0, 1], &with_topk(1));
    assert!(close(ciou_eps(&ga, &gc), 0.770881, 1e-5) && close(ciou_eps(&gb, &gc), 0.689264, 1e-5));
    assert_eq!(a.fg_mask, [true, true]);
    assert_eq!(a.target_gt, [0, 2]);
    assert_eq!(a.target_labels, [0, 1]);
    assert_eq!(a.target_bboxes, [ga, gc]);
    // Each gt has one positive, so its target is (about) its overlap.
    assert!(close(a.target_scores[0], 0.770881, 1e-5), "{}", a.target_scores[0]);
    assert!(close(a.target_scores[1], 1.0, 1e-5));
}

#[test]
fn rule4_equal_overlap_goes_to_smaller_gt_index() {
    // A = (0,0)-(40,40), B = (10,0)-(50,40): the prediction (5,0)-(45,40) sits exactly halfway, so both overlaps are
    // bitwise equal and the smaller gt index wins, whichever box it is.
    let (ga, gb) = (Xyxy::new(0.0, 0.0, 40.0, 40.0), Xyxy::new(10.0, 0.0, 50.0, 40.0));
    let pred = Xyxy::new(5.0, 0.0, 45.0, 40.0);
    assert_eq!(ciou_eps(&ga, &pred), ciou_eps(&gb, &pred));
    let anchors = [[25.0, 20.0]];
    let a = assign(&[0.5, 0.5], 2, &[pred], &anchors, &[ga, gb], &[0, 1], &TalConfig::default());
    assert_eq!((a.target_gt[0], a.target_labels[0], a.target_bboxes[0]), (0, 0, ga));
    let a = assign(&[0.5, 0.5], 2, &[pred], &anchors, &[gb, ga], &[1, 0], &TalConfig::default());
    assert_eq!((a.target_gt[0], a.target_labels[0], a.target_bboxes[0]), (0, 1, gb));
}

// ---- Rule 5: normalisation over the final positives -------------------------------------------------------------

#[test]
fn rule5_normalises_over_final_positives() {
    // A = (0,0)-(40,40) and B = (20,0)-(60,40), one class, scores 0.5. Anchors X (30, 20) in both, Y (10, 20) in A
    // only, Z (50, 20) in B only.
    //   X predicts (11,0)-(51,40): u_A = 1160/2040 - 121/4201 = 0.539824, u_B = 1240/1960 - 81/4001 = 0.612408 → B.
    //   Y predicts (16,0)-(56,40): u_A = 960/2240 - 256/4736 = 0.374517.
    //   Z predicts B exactly: u_B = 1.
    // A's final positives are {Y}: m_max and u_max are Y's own, so t(Y) = u_A(Y). Had the maxima been taken
    // before conflict resolution (including X), t(Y) would be (u_A(Y) / u_A(X))^6 · u_A(X) ≈ 0.060.
    // B's final positives are {X, Z}: u_max = 1, m_max = m(Z), so t(X) = u_B(X)^6 = 0.052752 and t(Z) = 1.
    let (ga, gb) = (Xyxy::new(0.0, 0.0, 40.0, 40.0), Xyxy::new(20.0, 0.0, 60.0, 40.0));
    let anchors = [[30.0, 20.0], [10.0, 20.0], [50.0, 20.0]];
    let preds = [Xyxy::new(11.0, 0.0, 51.0, 40.0), Xyxy::new(16.0, 0.0, 56.0, 40.0), gb];
    let a = assign(&[0.5; 3], 1, &preds, &anchors, &[ga, gb], &[0, 0], &TalConfig::default());
    assert_eq!(a.target_gt, [1, 0, 1]);
    assert!(a.fg_mask.iter().all(|&f| f));
    assert!(close(a.target_scores[1], 0.374517, 1e-5), "{}", a.target_scores[1]);
    assert!(close(a.target_scores[0], 0.052752, 1e-5), "{}", a.target_scores[0]);
    assert!(close(a.target_scores[2], 1.0, 1e-5));
}

#[test]
fn rule5_m_max_and_u_max_come_from_different_anchors() {
    // One gt (0,0)-(40,40), two positives:
    //   P (10, 20) predicts the gt exactly, u = 1, score 0.01:  m = 0.1.
    //   Q (30, 20) predicts (2,0)-(42,40), u = 1520/1680 - 4/3364 = 0.903573, score 0.81:  m = 0.9 · u^6 = 0.489803.
    // m_max = m(Q) but u_max = u(P) = 1, so t(Q) = 1 and t(P) = 0.1 / 0.489803 = 0.204164. (Taking u_max from the
    // best-metric anchor would give t(Q) = 0.903573.)
    let gt = Xyxy::new(0.0, 0.0, 40.0, 40.0);
    let anchors = [[10.0, 20.0], [30.0, 20.0]];
    let preds = [gt, Xyxy::new(2.0, 0.0, 42.0, 40.0)];
    let a = assign(&[0.01, 0.81], 1, &preds, &anchors, &[gt], &[0], &TalConfig::default());
    assert_eq!(a.fg_mask, [true, true]);
    assert!(close(a.target_scores[1], 1.0, 1e-5), "{}", a.target_scores[1]);
    assert!(close(a.target_scores[0], 0.204164, 1e-5), "{}", a.target_scores[0]);
}

// ---- Empty inputs and the grid --------------------------------------------------------------------------------

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
