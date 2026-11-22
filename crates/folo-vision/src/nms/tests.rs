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
