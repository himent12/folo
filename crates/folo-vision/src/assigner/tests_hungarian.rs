//! Unit tests for exact Hungarian linear-sum assignment (`hungarian_assign`, `hungarian_assign_f32`)
//! and [`OneToOneMatcher`]: exhaustive brute-force optimality verification on rectangular & square
//! matrices, deterministic tie-breaking, negative & extreme-scale costs, finite-cost validation,
//! planted-optimum large matrices, and `OneToOneMatcher` DETR/RT-DETR matching invariants.

use folo_core::rng::Rng;

use super::hungarian::{
    ClsCostKind, OneToOneConfig, OneToOneMatcher, hungarian_assign, hungarian_assign_f32,
};
use crate::boxes::{IouKind, Xyxy};

/// Exhaustive brute-force minimum linear-sum assignment cost for small `rows x cols` matrices.
fn brute_force_min_cost(costs: &[f64], rows: usize, cols: usize) -> f64 {
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        costs: &[f64],
        rows: usize,
        cols: usize,
        transposed: bool,
        r: usize,
        used: &mut [bool],
        cur: f64,
        best: &mut f64,
    ) {
        let n = if transposed { cols } else { rows };
        let m = if transposed { rows } else { cols };
        if r == n {
            if cur < *best {
                *best = cur;
            }
            return;
        }
        for c in 0..m {
            if !used[c] {
                used[c] = true;
                let v = if transposed { costs[c * cols + r] } else { costs[r * cols + c] };
                dfs(costs, rows, cols, transposed, r + 1, used, cur + v, best);
                used[c] = false;
            }
        }
    }
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let transposed = rows > cols;
    let m = rows.max(cols);
    let mut used = vec![false; m];
    let mut best = f64::INFINITY;
    dfs(costs, rows, cols, transposed, 0, &mut used, 0.0, &mut best);
    best
}

fn assert_valid_matching(pairs: &[(usize, usize)], rows: usize, cols: usize) {
    assert_eq!(pairs.len(), rows.min(cols));
    let mut used_r = vec![false; rows];
    let mut used_c = vec![false; cols];
    for w in pairs.windows(2) {
        assert!(w[0].0 < w[1].0, "pairs must be strictly sorted by row: {pairs:?}");
    }
    for &(r, c) in pairs {
        assert!(r < rows && c < cols);
        assert!(!used_r[r], "duplicate row {r}");
        assert!(!used_c[c], "duplicate col {c}");
        used_r[r] = true;
        used_c[c] = true;
    }
}

/// Exhaustively verify `hungarian_assign` matches brute-force global optimum across square, wide
/// (`rows < cols`), and tall (`rows > cols`) matrices with positive, negative, integer-tie, and uniform costs.
#[test]
fn hungarian_matches_brute_force_on_small_matrices() {
    let mut rng = Rng::seed(2026);
    let shapes = [
        (1, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (2, 5),
        (5, 2),
        (3, 3),
        (3, 6),
        (6, 3),
        (4, 4),
        (4, 6),
        (6, 4),
        (5, 5),
        (6, 6),
    ];

    for &(rows, cols) in &shapes {
        for trial in 0..8 {
            let costs: Vec<f64> = match trial {
                // Continuous mixed positive/negative costs
                0..=3 => (0..rows * cols).map(|_| rng.uniform(-50.0, 50.0) as f64).collect(),
                // Discrete small integers with heavy ties
                4..=5 => (0..rows * cols).map(|_| ((rng.next_u64() % 5) as i64 - 2) as f64).collect(),
                // All-equal costs
                6 => vec![7.5; rows * cols],
                // Large offset + small variations
                _ => (0..rows * cols).map(|_| 1e6 + rng.uniform(-1.0, 1.0) as f64).collect(),
            };
            let res = hungarian_assign(&costs, rows, cols).unwrap();
            assert_valid_matching(&res.pairs, rows, cols);
            let recomputed: f64 = res.pairs.iter().map(|&(r, c)| costs[r * cols + c]).sum();
            assert!((res.total_cost - recomputed).abs() <= 1e-9 + 1e-12 * recomputed.abs());

            let opt = brute_force_min_cost(&costs, rows, cols);
            let tol = 1e-9 + 1e-10 * opt.abs();
            assert!(
                (res.total_cost - opt).abs() <= tol,
                "shape {rows}x{cols} trial {trial}: got {} vs brute-force {opt}",
                res.total_cost
            );
        }
    }
}

/// Deterministic tie-breaking, extreme-magnitude rescaling, empty matrices, `f32` wrapper, and
/// rejection of `NaN` / `±Inf` / mismatched lengths.
#[test]
fn hungarian_tie_breaking_extremes_and_validation() {
    // All-zero matrix -> deterministic identity matching (0,0), (1,1), ...
    let zeros = vec![0.0; 12];
    let r_wide = hungarian_assign(&zeros, 3, 4).unwrap();
    assert_eq!(r_wide.pairs, vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(r_wide.total_cost, 0.0);
    // Repeated runs produce bit-identical results.
    assert_eq!(r_wide, hungarian_assign(&zeros, 3, 4).unwrap());

    let r_tall = hungarian_assign(&zeros, 4, 3).unwrap();
    assert_eq!(r_tall.pairs, vec![(0, 0), (1, 1), (2, 2)]);

    // Empty matrices.
    assert_eq!(hungarian_assign(&[], 0, 5).unwrap().pairs, Vec::<(usize, usize)>::new());
    assert_eq!(hungarian_assign(&[], 5, 0).unwrap().pairs, Vec::<(usize, usize)>::new());
    assert_eq!(hungarian_assign(&[], 0, 0).unwrap().total_cost, 0.0);

    // Extreme magnitudes (~1e200 and f64::MAX) must not overflow in `cost - u - v`.
    let huge = vec![3e200, 1e200, 2e200, 1e200, 5e200, 4e200]; // 2x3: opt is (0,1)+(1,0) = 2e200
    let r_huge = hungarian_assign(&huge, 2, 3).unwrap();
    assert_eq!(r_huge.pairs, vec![(0, 1), (1, 0)]);
    assert!((r_huge.total_cost - 2e200).abs() <= 1e186);

    // Regression: large negative entry in row 2 must NOT absorb small differences (1 vs 0).
    let mixed = vec![
        9e17, 9e17, 9e17, 1e17, // row 0 -> col 3 (1e17)
        -1e17, 9e17, 9e17, 9e17, // row 1 -> col 0 (-1e17)
        -1e16, 1.0, 0.0, 9e17, // row 2 -> col 2 (0.0), NOT col 1 (1.0)
    ];
    let r_mixed = hungarian_assign(&mixed, 3, 4).unwrap();
    assert_eq!(r_mixed.pairs, vec![(0, 3), (1, 0), (2, 2)]);
    assert_eq!(r_mixed.total_cost, 0.0);

    // Regression: extreme scale (1e308) with tiny negative entry (-1e-20) must preserve -1e-20 (both orientations).
    let wide_ext = vec![0.0, -1e-20, 0.0, 0.0, 0.0, 1e308]; // 2x3: opt is (0,1)+(1,0) = -1e-20
    let r_we = hungarian_assign(&wide_ext, 2, 3).unwrap();
    assert_eq!(r_we.pairs, vec![(0, 1), (1, 0)]);
    assert_eq!(r_we.total_cost, -1e-20);
    let tall_ext = vec![0.0, 0.0, -1e-20, 0.0, 0.0, 1e308]; // 3x2: opt is (1,0)+(0,1) = -1e-20
    let r_te = hungarian_assign(&tall_ext, 3, 2).unwrap();
    assert_eq!(r_te.pairs, vec![(0, 1), (1, 0)]);
    assert_eq!(r_te.total_cost, -1e-20);

    // f32 wrapper parity.
    let f32_costs = [4.0f32, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
    let r32 = hungarian_assign_f32(&f32_costs, 3, 3).unwrap();
    assert_eq!(r32.pairs, vec![(0, 1), (1, 0), (2, 2)]);
    assert_eq!(r32.total_cost, 5.0);

    // Planted global optimum on a larger rectangular 25 x 40 matrix.
    let (nr, nc) = (25usize, 40usize);
    let mut rng = Rng::seed(777);
    let mut big = vec![0.0f64; nr * nc];
    for v in &mut big {
        *v = rng.uniform(10.0, 100.0) as f64;
    }
    // Plant a unique optimal permutation with costs in [0.0, 1.0).
    let mut planted_cols: Vec<usize> = (0..nc).collect();
    for i in (1..nc).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        planted_cols.swap(i, j);
    }
    let mut planted_sum = 0.0f64;
    for r in 0..nr {
        let c = planted_cols[r];
        let val = 0.01 * (r as f64);
        big[r * nc + c] = val;
        planted_sum += val;
    }
    let r_big = hungarian_assign(&big, nr, nc).unwrap();
    assert_valid_matching(&r_big.pairs, nr, nc);
    for r in 0..nr {
        assert_eq!(r_big.pairs[r], (r, planted_cols[r]));
    }
    assert!((r_big.total_cost - planted_sum).abs() < 1e-12);

    // Validation errors: NaN, ±Inf, length mismatch, overflow.
    assert!(hungarian_assign(&[1.0, f64::NAN, 2.0, 3.0], 2, 2).is_err());
    assert!(hungarian_assign(&[1.0, f64::INFINITY, 2.0, 3.0], 2, 2).is_err());
    assert!(hungarian_assign(&[1.0, f64::NEG_INFINITY, 2.0, 3.0], 2, 2).is_err());
    assert!(hungarian_assign(&[1.0, 2.0, 3.0], 2, 2).is_err());
    assert!(hungarian_assign(&[], usize::MAX, 2).is_err());
    assert!(hungarian_assign_f32(&[1.0, f32::NAN], 1, 2).is_err());
    assert!(hungarian_assign_f32(&[1.0, f32::INFINITY], 1, 2).is_err());
    assert!(hungarian_assign_f32(&[1.0], 1, 2).is_err());
}

/// [`OneToOneMatcher`] end-to-end tests: global optimality over greedy matching, GIoU & CIoU modes,
/// Focal / BCE / Prob classification costs, image-size normalization, `Q < G` and `Q > G` regimes,
/// degenerate GT skipping, non-finite input handling, `to_assignment` / `dense_quality_targets`,
/// and config validation + serde round-trip.
#[test]
fn one_to_one_matcher_optimality_modes_and_edge_cases() {
    // 1. Classic case where greedy matching fails but Hungarian finds the global optimum:
    //    GT 0 at [0,0,10,10] (cls 0), GT 1 at [12,0,22,10] (cls 1).
    //    Query 0 is near GT 0 (moderate cls 0, moderate cls 1), Query 1 is strictly inside GT 0 (strong cls 0),
    //    Query 2 is near GT 1 (strong cls 1), Query 3 is far away background.
    let gts = [Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(12.0, 0.0, 22.0, 10.0)];
    let gt_labels = [0u32, 1u32];
    let pds = [
        Xyxy::new(1.0, 1.0, 9.0, 9.0),     // Q0: good for GT0, weaker cls
        Xyxy::new(0.1, 0.1, 9.9, 9.9),     // Q1: best for GT0
        Xyxy::new(12.2, 0.2, 21.8, 9.8),   // Q2: best for GT1
        Xyxy::new(50.0, 50.0, 60.0, 60.0), // Q3: background
    ];
    let logits = [
        0.5f32, -2.0, // Q0
        4.0, -4.0, // Q1 -> GT0
        -4.0, 4.0, // Q2 -> GT1
        -5.0, -5.0, // Q3
    ];

    for (cls_cost, iou_kind) in [
        (ClsCostKind::Focal { alpha: 0.25, gamma: 2.0 }, IouKind::Giou),
        (ClsCostKind::Bce, IouKind::Ciou),
        (ClsCostKind::Prob, IouKind::Diou),
        (ClsCostKind::Focal { alpha: 0.5, gamma: 0.0 }, IouKind::Iou),
    ] {
        let cfg = OneToOneConfig {
            cls_weight: 2.0,
            l1_weight: 5.0,
            iou_weight: 2.0,
            iou_kind,
            cls_cost,
            image_size: Some((64.0, 64.0)),
        };
        let matcher = OneToOneMatcher::new(cfg).unwrap();
        let asg = matcher.match_image(&logits, 2, &pds, &gts, &gt_labels).unwrap();
        assert!(asg.inputs_finite);
        assert_eq!(asg.num_fg, 2);
        assert_eq!(asg.pairs, vec![(1, 0), (2, 1)]);
        assert_eq!(asg.fg_mask, vec![false, true, true, false]);
        assert_eq!(asg.target_labels, vec![2, 0, 1, 2]);
        assert!(asg.target_ious[1] > 0.9 && asg.target_ious[2] > 0.9);
        assert_eq!(asg.target_ious[0], 0.0);
        assert_eq!(asg.target_ious[3], 0.0);

        let generic = asg.to_assignment();
        assert_eq!(generic.num_fg, 2);
        assert_eq!(generic.target_scores, asg.target_ious);

        let dense = asg.dense_quality_targets(2);
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[2], asg.target_ious[1]);
        assert_eq!(dense[5], asg.target_ious[2]);
        assert_eq!(dense[0], 0.0);
        assert_eq!(dense[1], 0.0);
    }

    // 2. Fewer queries than GTs (Q = 1 < G = 3): matches the single best GT cleanly.
    let matcher = OneToOneMatcher::new(OneToOneConfig::default()).unwrap();
    let gts3 = [
        Xyxy::new(0.0, 0.0, 10.0, 10.0),
        Xyxy::new(20.0, 20.0, 30.0, 30.0),
        Xyxy::new(40.0, 40.0, 50.0, 50.0),
    ];
    let asg_small = matcher
        .match_image(&[-3.0, 3.0], 2, &[Xyxy::new(20.5, 20.5, 29.5, 29.5)], &gts3, &[0, 1, 0])
        .unwrap();
    assert_eq!(asg_small.pairs, vec![(0, 1)]);
    assert_eq!(asg_small.num_fg, 1);

    // 3. Degenerate GT in the middle is skipped while preserving original GT indices (0 and 2).
    let gts_degen = [
        Xyxy::new(0.0, 0.0, 10.0, 10.0),
        Xyxy::new(5.0, 5.0, 5.0, 5.0), // degenerate zero-area GT
        Xyxy::new(20.0, 20.0, 30.0, 30.0),
    ];
    let pds2 = [Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(20.0, 20.0, 30.0, 30.0)];
    let asg_degen = matcher.match_image(&[2.0, -2.0, -2.0, 2.0], 2, &pds2, &gts_degen, &[0, 0, 1]).unwrap();
    assert_eq!(asg_degen.pairs, vec![(0, 0), (1, 2)]);
    assert_eq!(asg_degen.target_gt, vec![0, 2]);
    assert_eq!(asg_degen.target_labels, vec![0, 1]);

    // 4. Non-finite inputs return empty assignment with `inputs_finite = false` (never panics or errors).
    let asg_nan = matcher.match_image(&[f32::NAN, 0.0, 0.0, 0.0], 2, &pds2, &gts, &gt_labels).unwrap();
    assert!(!asg_nan.inputs_finite);
    assert_eq!(asg_nan.num_fg, 0);
    assert!(asg_nan.pairs.is_empty());

    // 5. Empty GTs or empty queries.
    let asg_empty_gt = matcher.match_image(&[0.0, 0.0, 0.0, 0.0], 2, &pds2, &[], &[]).unwrap();
    assert!(asg_empty_gt.inputs_finite && asg_empty_gt.num_fg == 0);
    let asg_empty_q = matcher.match_image(&[], 2, &[], &gts, &gt_labels).unwrap();
    assert!(asg_empty_q.inputs_finite && asg_empty_q.num_fg == 0);

    // 6. Config validation & serde round-trip.
    let cfg = OneToOneConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<OneToOneConfig>(&json).unwrap(), cfg);
    assert!(OneToOneMatcher::new(OneToOneConfig { cls_weight: -1.0, ..cfg }).is_err());
    assert!(OneToOneMatcher::new(OneToOneConfig { l1_weight: f32::NAN, ..cfg }).is_err());
    assert!(OneToOneMatcher::new(OneToOneConfig { iou_weight: f32::INFINITY, ..cfg }).is_err());
    assert!(OneToOneMatcher::new(OneToOneConfig { image_size: Some((0.0, 64.0)), ..cfg }).is_err());
    assert!(
        OneToOneMatcher::new(OneToOneConfig {
            cls_cost: ClsCostKind::Focal { alpha: 1.0, gamma: 2.0 },
            ..cfg
        })
        .is_err()
    );
    // Out-of-range GT label and length mismatch errors.
    assert!(matcher.match_image(&[0.0, 0.0], 2, &pds2[..1], &gts[..1], &[2]).is_err());
    assert!(matcher.match_image(&[0.0], 2, &pds2[..1], &gts[..1], &[0]).is_err());
}
