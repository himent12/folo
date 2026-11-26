use super::*;

fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> Xyxy {
    Xyxy::new(x1, y1, x2, y2)
}

#[test]
fn suppresses_overlapping_same_class_only() {
    let boxes = [b(0., 0., 10., 10.), b(1., 1., 11., 11.), b(0., 0., 10., 10.), b(50., 50., 60., 60.)];
    let scores = [0.9, 0.8, 0.7, 0.6];
    let classes = [0, 0, 1, 0];
    // Box 1 overlaps box 0 with IoU 81/119 = 0.68 > 0.5; box 2 is another class.
    assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 300, false), vec![0, 2, 3]);
    assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 300, true), vec![0, 3]);
    // Threshold is strict: IoU 0.68 is not > 0.7.
    assert_eq!(batched_nms(&boxes, &scores, &classes, 0.7, 300, false), vec![0, 1, 2, 3]);
    assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 2, false), vec![0, 2]);
}

#[test]
fn sparse_class_ids() {
    let boxes = [b(0., 0., 10., 10.), b(1., 1., 11., 11.), b(0., 0., 10., 10.)];
    let classes = [u32::MAX, u32::MAX, 7];
    assert_eq!(batched_nms(&boxes, &[0.9, 0.8, 0.7], &classes, 0.5, 300, false), vec![0, 2]);
}

/// NMS from `[A, 4 + nc]` rows is NMS from the split boxes and scores, single- and multi-label, with ties
/// and enough anchors for the parallel extraction.
#[test]
fn rows_match_split_outputs() {
    let (a, nc) = (5000usize, 3usize);
    let mut state = 12345u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut rows = Vec::with_capacity(a * (4 + nc));
    let (mut boxes, mut scores) = (Vec::new(), Vec::new());
    for _ in 0..a {
        let (x, y) = (next() * 600.0, next() * 600.0);
        let bx = Xyxy::new(x, y, x + 5.0 + next() * 60.0, y + 5.0 + next() * 60.0);
        rows.extend_from_slice(&[bx.x1, bx.y1, bx.x2, bx.y2]);
        boxes.push(bx);
        for _ in 0..nc {
            let s = (next() * 16.0).floor() / 16.0; // many ties
            rows.push(s);
            scores.push(s);
        }
    }
    for multi_label in [false, true] {
        let cfg = NmsConfig { conf: 0.05, multi_label, max_candidates: 4000, ..NmsConfig::default() };
        let want = non_max_suppression(&boxes, &scores, nc, &cfg);
        assert!(want.len() > 50);
        assert_eq!(non_max_suppression_rows(&rows, nc, &cfg), want, "multi_label {multi_label}");
    }
    assert!(non_max_suppression_rows(&[], nc, &NmsConfig::val()).is_empty());
}

/// NMS from shuffled `(anchor, class, score)` pairs is NMS from the rows they come from: tied scores on
/// overlapping boxes (the winner is the first in `(anchor, class)` order), scores below, at and above `conf`,
/// NaNs, a class filter, agnostic NMS, the candidate cap, and no pairs at all.
#[test]
fn pairs_match_rows_nms() {
    let (a, nc) = (40usize, 3usize);
    let conf = 0.25f32;
    let levels = [0.5, 0.5, conf, 0.3, f32::NAN, 0.1, 0.5, 0.0, f32::from_bits(conf.to_bits() + 1)];
    let mut rows = Vec::with_capacity(a * (4 + nc));
    let mut pairs = Vec::new();
    for i in 0..a {
        // Groups of four boxes one pixel apart (IoU about 0.9), the groups far apart.
        let x = (i / 4 * 30 + i % 4) as f32;
        rows.extend_from_slice(&[x, 0.0, x + 20.0, 20.0]);
        for c in 0..nc {
            let s = levels[(i * 3 + c * 5) % levels.len()];
            rows.push(s);
            pairs.push((i as u32, c as u32, s));
        }
    }
    // Two lone boxes with nothing above conf: no detections, though nothing would suppress them.
    for (i, s) in [(a, [conf, f32::NAN, 0.1]), (a + 1, [0.0, conf, f32::NAN])] {
        let x = (i * 30) as f32;
        rows.extend_from_slice(&[x, 0.0, x + 20.0, 20.0]);
        rows.extend_from_slice(&s);
        pairs.extend((0..nc).map(|c| (i as u32, c as u32, s[c])));
    }
    // A fixed shuffle that puts later anchors before earlier ones.
    let mut state = 7u64;
    for k in (1..pairs.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        pairs.swap(k, (state >> 33) as usize % (k + 1));
    }
    let bbox = |i: u32| {
        let r = &rows[i as usize * (4 + nc)..];
        Xyxy::new(r[0], r[1], r[2], r[3])
    };
    let val = NmsConfig { conf, ..NmsConfig::val() };
    for cfg in [
        val.clone(),
        NmsConfig { classes: Some(vec![0, 2]), ..val.clone() },
        NmsConfig { agnostic: true, ..val.clone() },
        NmsConfig { max_candidates: 9, max_det: 5, ..val.clone() },
    ] {
        let want = non_max_suppression_rows(&rows, nc, &cfg);
        assert!(want.len() >= 5, "{cfg:?}: {want:?}");
        assert_eq!(non_max_suppression_pairs(pairs.clone(), bbox, &cfg), want, "{cfg:?}");
    }
    // Nothing above conf, as rows or as (no) pairs.
    let none = NmsConfig { conf: 0.5, ..val };
    assert!(non_max_suppression_rows(&rows, nc, &none).is_empty());
    assert!(non_max_suppression_pairs(pairs, bbox, &none).is_empty());
    assert!(non_max_suppression_pairs(Vec::new(), bbox, &NmsConfig::val()).is_empty());
}

#[test]
fn candidates_match_full_nms() {
    // 50 anchors, 4 classes, many overlapping boxes and tied scores.
    let (a, nc) = (50usize, 4usize);
    let boxes: Vec<Xyxy> = (0..a)
        .map(|i| b((i % 7) as f32, (i % 5) as f32, (i % 7) as f32 + 9.0, (i % 5) as f32 + 9.0))
        .collect();
    let scores: Vec<f32> = (0..a * nc).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
    for cfg in [
        NmsConfig { conf: 0.25, iou: 0.5, ..NmsConfig::default() },
        NmsConfig { conf: 0.05, iou: 0.3, max_det: 7, max_candidates: 20, ..NmsConfig::default() },
        NmsConfig { classes: Some(vec![1, 3]), agnostic: true, ..NmsConfig::default() },
    ] {
        let want = non_max_suppression(&boxes, &scores, nc, &cfg);
        let cands: Vec<Candidate> = (0..a)
            .filter_map(|i| {
                let row = &scores[i * nc..(i + 1) * nc];
                let (mut best, mut bs) = (0, row[0]);
                for (c, &s) in row.iter().enumerate().skip(1) {
                    if s > bs {
                        (best, bs) = (c, s);
                    }
                }
                (bs > cfg.conf).then_some(Candidate {
                    anchor: i as u32,
                    class: best as u32,
                    score: bs,
                    bbox: boxes[i],
                })
            })
            .collect();
        assert_eq!(nms_candidates(&cands, &cfg), want, "{cfg:?}");
    }
}

#[test]
fn ties_keep_lower_index() {
    let boxes = [b(0., 0., 10., 10.), b(0., 0., 10., 10.)];
    assert_eq!(batched_nms(&boxes, &[0.5, 0.5], &[0, 0], 0.5, 10, false), vec![0]);
}

#[test]
fn post_process_filters() {
    // Three anchors, two classes.
    let boxes = [b(0., 0., 10., 10.), b(0., 0., 10., 10.), b(20., 20., 30., 30.)];
    let scores = [0.9, 0.3, 0.2, 0.8, 0.1, 0.05];
    let cfg = NmsConfig { conf: 0.25, iou: 0.5, ..NmsConfig::default() };
    let out = non_max_suppression(&boxes, &scores, 2, &cfg);
    assert_eq!(out.len(), 2);
    assert_eq!((out[0].class, out[0].score), (0, 0.9));
    assert_eq!((out[1].class, out[1].score), (1, 0.8));
    // Multi-label adds anchor 0's class-1 candidate (0.3), which anchor 1's identical 0.8 box then suppresses.
    let ml = NmsConfig { multi_label: true, ..cfg.clone() };
    let out = non_max_suppression(&boxes, &scores, 2, &ml);
    assert_eq!(out.iter().map(|d| d.score).collect::<Vec<_>>(), vec![0.9, 0.8]);
    // Class filter after argmax: anchor 1's best class is 1, anchor 0's is 0.
    let only1 = NmsConfig { classes: Some(vec![1]), ..cfg.clone() };
    let out = non_max_suppression(&boxes, &scores, 2, &only1);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].class, 1);
    // Candidate cap keeps the best.
    let capped = NmsConfig { max_candidates: 1, ..cfg };
    assert_eq!(non_max_suppression(&boxes, &scores, 2, &capped).len(), 1);
    assert!(non_max_suppression(&[], &[], 2, &NmsConfig::default()).is_empty());
}

#[test]
fn detection_serialises_flat() {
    let d = Detection { bbox: b(1., 2., 3., 4.), score: 0.5, class: 7 };
    let v = serde_json::to_value(d).unwrap();
    assert_eq!(
        v,
        serde_json::json!({ "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "score": 0.5, "class": 7 })
    );
    assert_eq!(serde_json::from_value::<Detection>(v).unwrap(), d);
}

#[test]
fn diou_nms_preserves_distinct_centers_and_suppresses_coincident() {
    // Two heavily overlapping boxes (IoU = 6/10 = 0.6 > 0.55) with shifted centres:
    // a = [0, 0, 8, 10], b = [2, 0, 10, 10] -> inter = 60, union = 100, IoU = 0.6.
    // Enclosing [0, 0, 10, 10]: c² = 200; centres (4,5) and (6,5) -> rho² = 4; DIoU = 0.6 - 4/200 = 0.58.
    // With thr = 0.59: standard NMS suppresses b (0.6 > 0.59), DIoU-NMS keeps BOTH (0.58 <= 0.59)!
    let boxes = [b(0., 0., 8., 10.), b(2., 0., 10., 10.), b(0., 0., 8., 10.)];
    assert_eq!(nms_sorted(&boxes, 0.59, 10), vec![0]);
    assert_eq!(diou_nms_sorted(&boxes, 0.59, 10), vec![0, 1]);
    // Coincident box 2 (DIoU = 1.0 > 0.59) is still suppressed by box 0.
    let (kept, own) = diou_nms_sorted_with_owner(&boxes, 0.59, 10);
    assert_eq!(kept, vec![0, 1]);
    assert_eq!(own, vec![0, 1, 0]);
    // Through NmsConfig::nms_kind = NmsKind::Diou.
    let dets = [
        Detection { bbox: boxes[0], score: 0.9, class: 0 },
        Detection { bbox: boxes[1], score: 0.8, class: 0 },
        Detection { bbox: boxes[2], score: 0.7, class: 0 },
    ];
    let cfg = NmsConfig { conf: 0.25, iou: 0.59, nms_kind: NmsKind::Diou, ..NmsConfig::default() };
    let out = nms_detections(&dets, &cfg);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].score, 0.9);
    assert_eq!(out[1].score, 0.8);
}

#[test]
fn soft_nms_linear_and_gaussian_decay_and_reorder() {
    // Box 0 and box 1 overlap with IoU = 0.5 (a=[0,0,10,10], b=[0,0,10,5] -> inter=50, union=100).
    // Box 2 is disjoint with score 0.6.
    let boxes = [b(0., 0., 10., 10.), b(0., 0., 10., 5.), b(50., 50., 60., 60.)];
    let scores = [0.9, 0.8, 0.6];

    // Linear Soft-NMS with iou_thr = 0.4: box 1 decays to 0.8 * (1 - 0.5) = 0.4, so box 2 (0.6) moves ahead!
    let lin = soft_nms_sorted(&boxes, &scores, 0.4, 0.25, None, 10);
    assert_eq!(lin.len(), 3);
    assert_eq!(lin[0], (0, 0.9));
    assert_eq!(lin[1], (2, 0.6));
    assert_eq!(lin[2].0, 1);
    assert!((lin[2].1 - 0.4).abs() < 1e-6);

    // Higher score_thr = 0.45 drops decayed box 1.
    let lin_pruned = soft_nms_sorted(&boxes, &scores, 0.4, 0.45, None, 10);
    assert_eq!(lin_pruned, vec![(0, 0.9), (2, 0.6)]);

    // Gaussian Soft-NMS with sigma = 0.5: box 1 decays by exp(-0.25 / 0.5) = exp(-0.5) ≈ 0.6065307 -> ~0.4852245.
    let gauss = soft_nms_sorted(&boxes, &scores, 0.7, 0.25, Some(0.5), 10);
    assert_eq!(gauss.len(), 3);
    assert_eq!(gauss[0], (0, 0.9));
    assert_eq!(gauss[1], (2, 0.6));
    assert_eq!(gauss[2].0, 1);
    let want = 0.8 * (-0.5f32).exp();
    assert!((gauss[2].1 - want).abs() < 1e-6);

    // Through NmsConfig pipeline and JSON round-trip.
    let dets = [
        Detection { bbox: boxes[0], score: 0.9, class: 0 },
        Detection { bbox: boxes[1], score: 0.8, class: 0 },
        Detection { bbox: boxes[2], score: 0.6, class: 0 },
    ];
    let cfg = NmsConfig {
        conf: 0.25,
        iou: 0.4,
        nms_kind: NmsKind::SoftGaussian { sigma: 0.5 },
        ..NmsConfig::default()
    };
    let out = nms_detections(&dets, &cfg);
    assert_eq!(out.len(), 3);
    assert!((out[2].score - want).abs() < 1e-6);

    // Default NmsConfig omits nms_kind from JSON; non-default kinds round-trip cleanly.
    let def_json = serde_json::to_string(&NmsConfig::default()).unwrap();
    assert!(!def_json.contains("nms_kind"));
    let cfg_json = serde_json::to_string(&cfg).unwrap();
    assert!(cfg_json.contains("soft_gaussian"));
    assert_eq!(serde_json::from_str::<NmsConfig>(&cfg_json).unwrap(), cfg);

    // Negative scores with negative score_thr never increase under decay, preserving descending order.
    let same = [b(0., 0., 10., 10.), b(0., 0., 10., 10.)];
    let neg_lin = soft_nms_sorted(&same, &[-0.1, -0.2], 0.5, -1.0, None, 10);
    assert_eq!(neg_lin, vec![(0, -0.1), (1, -0.2)]);
    let neg_gauss = soft_nms_sorted(&same, &[-0.1, -0.2], 0.5, -1.0, Some(0.5), 10);
    assert_eq!(neg_gauss, vec![(0, -0.1), (1, -0.2)]);

    // Soft-NMS preserves decayed scores even if ScoreRule/Sebf are set, and Gaussian Soft-NMS is invariant to cfg.iou.
    let mut cfg_sebf_a = NmsConfig {
        conf: 0.25,
        iou: 0.2,
        nms_kind: NmsKind::SoftGaussian { sigma: 0.5 },
        fusion: crate::fusion::BoxFusion::Sebf,
        score: crate::fusion::ScoreRule::Elc,
        ..NmsConfig::default()
    };
    let out_a = nms_detections(&dets, &cfg_sebf_a);
    cfg_sebf_a.iou = 0.9;
    let out_b = nms_detections(&dets, &cfg_sebf_a);
    assert_eq!(out_a, out_b, "Gaussian Soft-NMS must not depend on cfg.iou");
    assert!((out_a[2].score - want).abs() < 1e-6, "Soft-NMS decayed score must not be overwritten");
}

#[test]
fn agnostic_diou_calibrated_score_uses_diou_class_local_ownership() {
    // Class 0 boxes A=[0,0,10,10] and B=[1.2,0,11.2,10] have IoU = 8.8/11.2 = 0.785714,
    // enclosing diag^2 = 11.2^2 + 10^2 = 225.44, center dist^2 = 1.2^2 = 1.44 -> DIoU = 0.779327.
    // Threshold 0.782 lies strictly between DIoU (0.7793) and IoU (0.7857).
    // Class 1 box C=[1.2,0,11.2,10] (score 0.95) suppresses B via agnostic DIoU (DIoU=1.0 > 0.782),
    // while A (score 0.90) survives. A must NOT credit B in class-local ELC because DIoU(A, B) <= 0.782.
    let dets = [
        Detection { bbox: b(1.2, 0.0, 11.2, 10.0), score: 0.95, class: 1 },
        Detection { bbox: b(0.0, 0.0, 10.0, 10.0), score: 0.90, class: 0 },
        Detection { bbox: b(1.2, 0.0, 11.2, 10.0), score: 0.80, class: 0 },
    ];
    let cfg = NmsConfig {
        conf: 0.01,
        iou: 0.782,
        agnostic: true,
        nms_kind: NmsKind::Diou,
        score: crate::fusion::ScoreRule::Elc,
        ..NmsConfig::default()
    };
    let out = nms_detections(&dets, &cfg);
    let a_det = out.iter().find(|d| d.class == 0).expect("A kept");
    let solo = crate::fusion::elc_scores(&[0], &[b(0.0, 0.0, 10.0, 10.0)], &[0.90], &[0], None, 0.782)[0];
    assert!(
        (a_det.score - solo).abs() < 1e-6,
        "agnostic DIoU-NMS ELC must not credit B (DIoU <= thr): got {} vs solo {solo}",
        a_det.score
    );
}

#[test]
fn nms_sorted_avx2_matches_naive_across_vector_lengths() {
    // Exercise non-multiple-of-8 and multi-vector lengths, both with and without owner tracking.
    for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65] {
        let mut s = 0x1234_5678u64 ^ n as u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32 * 0.5)
        };
        let boxes: Vec<Xyxy> = (0..n)
            .map(|_| {
                let x = next() * 40.0;
                let y = next() * 40.0;
                let w = 2.0 + next() * 20.0;
                let h = 2.0 + next() * 20.0;
                Xyxy::new(x, y, x + w, y + h)
            })
            .collect();
        let thr = 0.45f32;
        let (kept, own) = nms_sorted_with_owner(&boxes, thr, n);
        // Naive scalar reference.
        let mut ref_sup = vec![false; n];
        let mut ref_kept = Vec::new();
        let mut ref_own = vec![u32::MAX; n];
        for i in 0..n {
            if ref_sup[i] {
                continue;
            }
            let kp = ref_kept.len() as u32;
            ref_kept.push(i);
            ref_own[i] = kp;
            for j in i + 1..n {
                if !ref_sup[j] && crate::boxes::box_iou(&boxes[i], &boxes[j]) > thr {
                    ref_sup[j] = true;
                    ref_own[j] = kp;
                }
            }
        }
        assert_eq!(kept, ref_kept, "n={n} kept mismatch");
        assert_eq!(own, ref_own, "n={n} owner mismatch");
        assert_eq!(nms_sorted(&boxes, thr, n), ref_kept, "n={n} nms_sorted mismatch");
    }
}

#[test]
fn soft_nms_crowded_monotonicity_boundaries_ties_and_degenerate() {
    // 1. Strict IoU threshold boundary in Linear Soft-NMS: IoU == 0.5 must NOT decay at iou_thr = 0.5.
    let pair = [b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 5.0)]; // exact IoU = 0.5
    let at_thr = soft_nms_sorted(&pair, &[0.9, 0.8], 0.5, 0.0, None, 10);
    assert_eq!(at_thr, vec![(0, 0.9), (1, 0.8)], "IoU == iou_thr must not decay");

    // 2. Score decay monotonicity in crowded/occluded cluster (4 overlapping boxes with increasing overlap).
    // Box 0 is the anchor [0,0,10,10]; boxes 1, 2, 3 overlap it with IoU = 1/3, 0.5, 9/11 respectively.
    let crowd = [
        b(0.0, 0.0, 10.0, 10.0),
        b(5.0, 0.0, 15.0, 10.0), // IoU with 0 = 50/150 = 1/3
        b(0.0, 0.0, 10.0, 5.0),  // IoU with 0 = 50/100 = 1/2
        b(1.0, 0.0, 11.0, 10.0), // IoU with 0 = 90/110 = 9/11
    ];
    let scores = [0.95f32, 0.80, 0.80, 0.80];
    for sigma in [None, Some(0.5f32), Some(0.25f32)] {
        let out = soft_nms_sorted(&crowd, &scores, 0.2, 0.0, sigma, 10);
        assert_eq!(out.len(), 4, "sigma={sigma:?}");
        // Output must be strictly sorted descending by decayed score, and every score <= original.
        assert!(out.windows(2).all(|w| w[0].1 >= w[1].1), "descending order broken: {out:?}");
        for &(idx, s) in &out {
            assert!(s <= scores[idx], "score increased for idx={idx}: {s} > {}", scores[idx]);
        }
        // Lower-overlap box 1 must finish strictly above higher-overlap box 3 (which overlaps 0 by 9/11).
        let s1 = out.iter().find(|&&(i, _)| i == 1).unwrap().1;
        let s3 = out.iter().find(|&&(i, _)| i == 3).unwrap().1;
        assert!(s1 > s3, "monotonic decay violated (sigma={sigma:?}): s1={s1} vs s3={s3}");
    }

    // 3. Ties preserve lower input index (both initial ties and post-decay ties).
    let tie_boxes = [b(0.0, 0.0, 10.0, 10.0), b(50.0, 0.0, 60.0, 10.0), b(100.0, 0.0, 110.0, 10.0)];
    assert_eq!(
        soft_nms_sorted(&tie_boxes, &[0.7, 0.7, 0.7], 0.5, 0.1, None, 10),
        vec![(0, 0.7), (1, 0.7), (2, 0.7)]
    );
    // Two disjoint boxes that overlap box 0 by identical IoU = 0.5 get identical decayed scores; lower index wins.
    let sym = [
        b(0.0, 0.0, 10.0, 10.0),
        b(0.0, 0.0, 10.0, 5.0),  // top half (IoU = 0.5 with box 0, 0 with box 2)
        b(0.0, 5.0, 10.0, 10.0), // bottom half (IoU = 0.5 with box 0, 0 with box 1)
    ];
    let sym_lin = soft_nms_sorted(&sym, &[0.9, 0.8, 0.8], 0.4, 0.1, None, 10);
    assert_eq!(sym_lin, vec![(0, 0.9), (1, 0.4), (2, 0.4)]);

    // 4. Empty inputs, limit == 0, non-finite scores (+Inf / NaN), near-identical f64 precision, huge/degenerate boxes.
    assert!(soft_nms_sorted(&[], &[], 0.5, 0.1, None, 10).is_empty());
    assert!(soft_nms_sorted(&pair, &[0.9, 0.8], 0.5, 0.1, None, 0).is_empty());
    // +Inf and NaN scores are dropped and never suppress valid detections or evict at max_candidates.
    let dup = [b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 10.0)];
    assert_eq!(soft_nms_sorted(&dup, &[f32::INFINITY, f32::NAN, 0.8], 0.5, 0.1, None, 10), vec![(2, 0.8)]);
    let dets_inf = [
        Detection { bbox: dup[0], score: f32::INFINITY, class: 0 },
        Detection { bbox: dup[1], score: 0.8, class: 0 },
    ];
    let cfg_cap = NmsConfig {
        conf: 0.25,
        max_candidates: 1,
        nms_kind: NmsKind::SoftGaussian { sigma: 0.5 },
        ..NmsConfig::default()
    };
    let kept_inf = nms_detections(&dets_inf, &cfg_cap);
    assert_eq!(kept_inf.len(), 1);
    assert_eq!(kept_inf[0].score, 0.8);

    // Near-identical boxes `[0,0,1,1]` and `[1e-8,0,1,1]` as well as `[1e-20,0,1,1]`: direct non-overlap
    // deficit calculation preserves exact representable `f32` decayed scores (~8e-9 and ~8e-21) instead of
    // cancelling `1 - IoU` to 0.
    let near = [b(0.0, 0.0, 1.0, 1.0), b(1e-8, 0.0, 1.0, 1.0)];
    let near_lin = soft_nms_sorted(&near, &[0.9, 0.8], 0.5, 0.0, None, 10);
    assert_eq!(near_lin.len(), 2);
    assert!((near_lin[1].1 - 8e-9).abs() < 1e-9, "near-identical f64 decay: {}", near_lin[1].1);
    let ultra_near = [b(0.0, 0.0, 1.0, 1.0), b(1e-20, 0.0, 1.0, 1.0)];
    let ultra_lin = soft_nms_sorted(&ultra_near, &[0.9, 0.8], 0.5, 0.0, None, 10);
    assert_eq!(ultra_lin.len(), 2, "1e-20 shift must not round decayed score to 0");
    assert!((ultra_lin[1].1 - 8e-21).abs() < 1e-22, "ultra-near decay: {:e}", ultra_lin[1].1);

    // Single-label non_max_suppression with [+Inf, 0.8] and [NaN, 0.8] under Soft/Matrix-NMS keeps valid class 1.
    let single_boxes = [b(0.0, 0.0, 10.0, 10.0), b(50.0, 50.0, 60.0, 60.0)];
    let bad_scores = [f32::INFINITY, 0.8, f32::NAN, 0.75];
    for kind in [NmsKind::SoftLinear, NmsKind::SoftGaussian { sigma: 0.5 }, NmsKind::MatrixLinear] {
        let cfg = NmsConfig { conf: 0.25, nms_kind: kind, ..NmsConfig::default() };
        let out = non_max_suppression(&single_boxes, &bad_scores, 2, &cfg);
        assert_eq!(out.len(), 2, "{kind:?} must keep valid finite classes on anchors with +Inf/NaN");
        assert_eq!((out[0].class, out[0].score), (1, 0.8));
        assert_eq!((out[1].class, out[1].score), (1, 0.75));
    }

    // Huge 1e25 identical boxes must decay properly (not overflow f32 area to NaN / IoU=0).
    let huge = [b(0.0, 0.0, 1e25, 1e25), b(0.0, 0.0, 1e25, 1e25)];
    assert_eq!(soft_nms_sorted(&huge, &[0.9, 0.8], 0.5, 0.01, None, 10), vec![(0, 0.9)]);

    // Degenerate (point, inverted, NaN coord) boxes never produce NaN scores.
    let messy = [
        b(0.0, 0.0, 10.0, 10.0),
        b(5.0, 5.0, 5.0, 5.0),
        b(10.0, 10.0, 2.0, 2.0),
        b(f32::NAN, 0.0, 10.0, 10.0),
    ];
    let messy_out = soft_nms_sorted(&messy, &[0.9, 0.7, 0.6, 0.5], 0.5, 0.1, Some(0.5), 10);
    assert_eq!(messy_out.len(), 4);
    assert!(messy_out.iter().all(|(_, s)| s.is_finite() && *s > 0.1));
}

#[test]
fn diou_nms_matches_naive_reference_and_handles_edge_cases() {
    use crate::boxes::{IouKind, iou};

    // 1. Exact threshold strictness: DIoU == thr must NOT suppress.
    // a=[0,0,8,10], b=[2,0,10,10]: IoU = 0.6, rho²/c² = 4/200 = 0.02 -> DIoU = 0.58.
    let pair = [b(0.0, 0.0, 8.0, 10.0), b(2.0, 0.0, 10.0, 10.0)];
    let exact_diou = iou(&pair[0], &pair[1], IouKind::Diou);
    assert_eq!(diou_nms_sorted(&pair, exact_diou, 10), vec![0, 1], "DIoU == thr must not suppress");
    assert_eq!(diou_nms_sorted(&pair, exact_diou - 1e-5, 10), vec![0], "DIoU > thr must suppress");

    // 2. Empty, limit == 0, ties, degenerate (point / line / inverted / huge / NaN).
    assert!(diou_nms_sorted(&[], 0.5, 10).is_empty());
    assert!(diou_nms_sorted(&pair, 0.5, 0).is_empty());
    let dup = [b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 10.0)];
    assert_eq!(
        batched_nms_with_kind(&dup, &[0.7, 0.7], &[0, 0], 0.5, 0.0, 10, false, NmsKind::Diou),
        vec![(0, 0.7)]
    );

    // 3. Seeded random crowded scenes across positive, zero, and negative DIoU thresholds,
    //    matching a naive scalar `boxes::iou(..., IouKind::Diou)` reference bit for bit!
    let mut s = 0xD100_BEEFu64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / (1u64 << 24) as f32
    };
    for (case_idx, &thr) in [-0.25f32, 0.0, 0.35, 0.55, 0.85].iter().enumerate() {
        let n = 45;
        let mut boxes: Vec<Xyxy> = (0..n)
            .map(|i| {
                if i == 3 {
                    b(5.0, 5.0, 5.0, 5.0) // zero-area point
                } else if i == 7 {
                    b(20.0, 20.0, 10.0, 10.0) // inverted
                } else {
                    let x = next() * 25.0;
                    let y = next() * 25.0;
                    b(x, y, x + 5.0 + next() * 12.0, y + 5.0 + next() * 12.0)
                }
            })
            .collect();
        boxes.push(b(f32::NAN, 0.0, 10.0, 10.0));
        let scores: Vec<f32> = (0..boxes.len()).map(|_| (next() * 10.0).floor() / 10.0).collect();
        let classes: Vec<u32> = (0..boxes.len()).map(|i| (i % 3) as u32).collect();

        for agnostic in [false, true] {
            let got = batched_nms_with_kind(&boxes, &scores, &classes, thr, 0.0, 20, agnostic, NmsKind::Diou);
            // Naive reference.
            let mut order: Vec<usize> = (0..boxes.len()).collect();
            order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]).then(i.cmp(&j)));
            let mut want = Vec::new();
            for i in order {
                let suppressed = want.iter().any(|&(k, _): &(usize, f32)| {
                    (agnostic || classes[k] == classes[i]) && iou(&boxes[k], &boxes[i], IouKind::Diou) > thr
                });
                if !suppressed {
                    want.push((i, scores[i]));
                }
            }
            want.truncate(20);
            assert_eq!(got, want, "DIoU-NMS case={case_idx} thr={thr} agnostic={agnostic}");
        }
    }
}
