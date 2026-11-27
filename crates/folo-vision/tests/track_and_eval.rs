//! End-to-end integration tests for ByteTrack multi-object tracking ([`folo_vision::track`] wired into
//! [`Predictor::predict_sequence`]) and per-image error diagnosis ([`folo_vision::eval`] wired into [`Evaluator`]).

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_data::Image;
use folo_vision::model::quant::{self, Method};
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::{
    ByteTracker, Detection, EvalConfig, EvalReport, Evaluator, PredictConfig, Predictor, TrackConfig, Truth,
    Xyxy,
};

fn fused_model(nc: usize, head: HeadMode, seed: u64) -> FoloModel {
    let mut cfg = ModelConfig::new("n", nc).unwrap();
    cfg.head = head.as_str().to_string();
    let mut m = FoloModel::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    m.set_classes((0..nc).map(|i| format!("cls{i}")).collect()).unwrap();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(seed + 17));
    {
        let _g = no_grad();
        m.forward(&x).unwrap();
    }
    m.fuse().unwrap();
    m
}

fn sequence_frames(count: usize, w: u32, h: u32) -> Vec<Image> {
    (0..count)
        .map(|i| {
            let mut img = Image::new(w, h, [20, 28, 42]);
            let dx = (i as i64) * 2;
            img.fill_rect(10 + dx, 14, 34 + dx, 40, [235, 45, 45]);
            img.fill_rect(
                w as i64 - 40 - dx,
                h as i64 - 38,
                w as i64 - 12 - dx,
                h as i64 - 10,
                [45, 225, 85],
            );
            img
        })
        .collect()
}

/// `Predictor::predict_sequence` matches running single-frame prediction at `conf.min(low_thresh)` +
/// `ByteTracker::update` across multi-chunk batches (> 8 frames), preserves `Predictor::config()`, and works
/// across `f32`, `int8` (`u8_stem`), and `o2o` models.
#[test]
fn predict_sequence_matches_manual_tracking_across_f32_int8_and_o2o() {
    let frames = sequence_frames(10, 96, 64); // > SEQUENCE_BATCH (8) to exercise chunked batching

    // 1. f32 o2m: Predictor has higher conf (0.25) than TrackConfig::low_thresh (0.0); predict_sequence must
    // automatically lower detection conf without mutating Predictor::config().
    let m_f32 = fused_model(2, HeadMode::O2m, 61);
    let m_ref = fused_model(2, HeadMode::O2m, 61);
    let pred =
        Predictor::new(m_f32, PredictConfig { imgsz: 64, conf: 0.25, max_det: 20, ..Default::default() })
            .unwrap();
    let pred_low =
        Predictor::new(m_ref, PredictConfig { imgsz: 64, conf: 0.0, max_det: 20, ..Default::default() })
            .unwrap();

    let track_cfg = TrackConfig {
        low_thresh: 0.0,
        high_thresh: 0.005,
        new_track_thresh: 0.005,
        match_iou: 0.2,
        second_match_iou: 0.3,
        ..Default::default()
    };
    let t_wall = std::time::Instant::now();
    let seq = pred.predict_sequence(&frames, &track_cfg).unwrap();
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1e3;
    assert_eq!(pred.config().conf, 0.25, "predict_sequence must not mutate Predictor::config");
    assert_eq!(seq.len(), frames.len());

    // Compare bit-for-bit against manual frame-by-frame prediction at det_conf + ByteTracker::update.
    let mut manual_tracker = ByteTracker::new(track_cfg.clone()).unwrap();
    let mut total_tracks = 0usize;
    for (idx, (sf, img)) in seq.iter().zip(&frames).enumerate() {
        assert_eq!(sf.frame_index, idx);
        assert_eq!((sf.width, sf.height), (img.width, img.height));
        let manual_det = pred_low.predict(img).unwrap();
        let manual_tracks = manual_tracker.update(&manual_det.detections);
        assert_eq!(sf.tracks, manual_tracks, "frame {idx} diverged from manual tracking");
        assert!(sf.tracks.windows(2).all(|w| w[0].track_id < w[1].track_id));
        total_tracks += sf.tracks.len();
    }
    assert!(total_tracks > 0, "sequence should emit tracked detections");

    // Repeated frames preserve persistent track IDs and increment hits & age monotonically.
    let same_frames = vec![frames[0].clone(); 4];
    let stable = pred.predict_sequence(&same_frames, &track_cfg).unwrap();
    assert!(!stable[0].tracks.is_empty());
    let first_id = stable[0].tracks[0].track_id;
    for (i, sf) in stable.iter().enumerate() {
        let tr = sf.tracks.iter().find(|t| t.track_id == first_id).expect("track ID must persist");
        assert_eq!(tr.age, (i + 1) as u32);
        assert_eq!(tr.hits, (i + 1) as u32);
    }

    // Parallel CPU wall-time check across the 10-frame sequence.
    let sum_ms: f64 = seq.iter().map(|f| f.timing.total_ms).sum();
    assert!(
        sum_ms <= wall_ms + 5.0,
        "sequence sum total_ms ({sum_ms:.2}) must not double-count parallel CPU wall time ({wall_ms:.2})"
    );

    // 2. int8 (u8_stem) model.
    let mut m_q8 = fused_model(2, HeadMode::O2m, 61);
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(62));
    let cal = quant::calibrate(&m_q8, || std::iter::once(Ok(x.clone())), Method::MinMax).unwrap();
    assert!(cal.u8_stem);
    m_q8.set_int8(Some(cal)).unwrap();
    let pred_q8 =
        Predictor::new(m_q8, PredictConfig { imgsz: 64, conf: 0.25, max_det: 20, ..Default::default() })
            .unwrap();
    let q8_seq = pred_q8.predict_sequence(&frames[..3], &track_cfg).unwrap();
    assert_eq!(q8_seq.len(), 3);
    assert!(q8_seq.iter().any(|f| !f.tracks.is_empty()));

    // 3. o2o (NMS-free top-k) model (untrained o2o head has low bias prior ~1e-3, so lower track thresholds).
    let m_o2o = fused_model(3, HeadMode::O2o, 71);
    let pred_o2o =
        Predictor::new(m_o2o, PredictConfig { imgsz: 64, conf: 0.25, max_det: 15, ..Default::default() })
            .unwrap();
    let o2o_track_cfg =
        TrackConfig { low_thresh: 0.0, high_thresh: 0.0, new_track_thresh: 0.0, ..track_cfg.clone() };
    let o2o_seq = pred_o2o.predict_sequence(&frames[..3], &o2o_track_cfg).unwrap();
    assert_eq!(o2o_seq.len(), 3);
    assert!(o2o_seq.iter().any(|f| !f.tracks.is_empty()));

    // 4. Mixed-resolution sequence: every frame keeps its exact single-frame `rect` canvas regardless of
    // adjacent frames' dimensions, matching manual single-frame prediction bit-for-bit.
    let mixed = vec![
        sequence_frames(1, 128, 64).pop().unwrap(),
        sequence_frames(1, 64, 128).pop().unwrap(),
        sequence_frames(1, 128, 64).pop().unwrap(),
    ];
    let mixed_seq = pred.predict_sequence(&mixed, &track_cfg).unwrap();
    let mut mixed_manual = ByteTracker::new(track_cfg.clone()).unwrap();
    for (sf, img) in mixed_seq.iter().zip(&mixed) {
        assert_eq!((sf.width, sf.height), (img.width, img.height));
        let manual_det = pred_low.predict(img).unwrap();
        assert_eq!(sf.tracks, mixed_manual.update(&manual_det.detections));
    }

    // 5. Calibrated ScoreRule (`Elc`) works seamlessly through `predict_sequence`.
    let pred_elc = Predictor::new(
        fused_model(2, HeadMode::O2m, 61),
        PredictConfig {
            imgsz: 64,
            conf: 0.25,
            max_det: 20,
            score: folo_vision::fusion::ScoreRule::Elc,
            ..Default::default()
        },
    )
    .unwrap();
    let elc_seq = pred_elc.predict_sequence(&frames[..3], &track_cfg).unwrap();
    assert_eq!(elc_seq.len(), 3);

    // 6. Empty sequence, invalid config, and bad image buffer errors.
    assert!(pred.predict_sequence(&[], &track_cfg).unwrap().is_empty());
    let bad_cfg = TrackConfig { low_thresh: 0.8, high_thresh: 0.5, ..Default::default() };
    assert!(pred.predict_sequence(&[], &bad_cfg).is_err());
    assert!(pred.predict_sequence(&frames[..1], &bad_cfg).is_err());
    let empty_img = Image { width: 0, height: 64, data: Vec::new() };
    assert!(pred.predict_sequence(&[empty_img], &track_cfg).is_err());
    let bad_buf = Image { width: 64, height: 64, data: vec![0u8; 10] };
    assert!(pred.predict_sequence(&[bad_buf], &track_cfg).is_err());
}

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn gt(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

/// Wired per-image diagnostics in `Evaluator`: opt-in zero-overhead default, exact error categorization,
/// parallel `Evaluator::merge` index offset parity, `coco_compatible` alignment, and `EvalReport` JSON round-trip.
#[test]
fn evaluator_diagnostics_merge_parity_coco_filter_and_serde() {
    // 1. Default config leaves diagnostics off and omits `image_diagnostics` from JSON.
    let mut ev_off = Evaluator::new(3);
    ev_off.add_image(&[det(0.0, 0.0, 50.0, 50.0, 0.9, 0)], &[gt(0.0, 0.0, 50.0, 50.0, 0)]);
    assert!(ev_off.image_diagnostics().is_empty());
    assert!(ev_off.worst_images(10).is_empty());
    let rep_off = ev_off.compute();
    let json_off = serde_json::to_value(&rep_off).unwrap();
    assert!(json_off.get("image_diagnostics").is_none());

    // 2. Opt-in diagnostics across 4 images: compare sequential vs split-and-merge parity.
    let cases: Vec<(&str, Vec<Detection>, Vec<Truth>)> = vec![
        (
            "img_0_perfect.jpg",
            vec![det(10.0, 10.0, 60.0, 60.0, 0.95, 0)],
            vec![gt(10.0, 10.0, 60.0, 60.0, 0)],
        ),
        (
            "img_1_loc_and_bg.jpg",
            vec![
                det(35.0, 10.0, 85.0, 60.0, 0.85, 0), // IoU = 1/3 -> localization_errors
                det(200.0, 200.0, 240.0, 240.0, 0.70, 1), // pure background_fp
            ],
            vec![gt(10.0, 10.0, 60.0, 60.0, 0)],
        ),
        (
            "img_2_cls_and_dup.jpg",
            vec![
                det(0.0, 0.0, 100.0, 100.0, 0.92, 0),   // TP
                det(1.0, 1.0, 99.0, 99.0, 0.81, 0),     // duplicate_errors
                det(200.0, 0.0, 300.0, 100.0, 0.88, 2), // class_confusions (truth is class 1)
            ],
            vec![gt(0.0, 0.0, 100.0, 100.0, 0), gt(200.0, 0.0, 300.0, 100.0, 1)],
        ),
        (
            "img_3_all_missed.jpg",
            vec![],
            vec![
                gt(0.0, 0.0, 40.0, 40.0, 0),
                gt(60.0, 0.0, 100.0, 40.0, 1),
                gt(120.0, 0.0, 160.0, 40.0, 2),
                gt(180.0, 0.0, 220.0, 40.0, 0),
            ],
        ),
    ];

    let cfg = EvalConfig::with_diagnostics();
    let mut seq_ev = Evaluator::with_config(3, cfg.clone());
    for (id, preds, truths) in &cases {
        seq_ev.add_image_with_id(*id, preds, truths);
    }

    let mut left = Evaluator::with_config(3, cfg.clone());
    let mut right = Evaluator::with_config(3, cfg);
    for (id, preds, truths) in &cases[..2] {
        left.add_image_with_id(*id, preds, truths);
    }
    for (id, preds, truths) in &cases[2..] {
        right.add_image_with_id(*id, preds, truths);
    }
    left.merge(right);

    assert_eq!(left.image_diagnostics(), seq_ev.image_diagnostics());
    assert_eq!(left.worst_images(10), seq_ev.worst_images(10));

    let diags = seq_ev.image_diagnostics();
    assert_eq!(diags.len(), 4);
    assert_eq!((diags[0].image_index, diags[0].tp, diags[0].fp, diags[0].fn_count), (0, 1, 0, 0));
    assert_eq!(
        (
            diags[1].image_index,
            diags[1].tp,
            diags[1].localization_errors,
            diags[1].background_fp,
            diags[1].fn_count
        ),
        (1, 0, 1, 1, 1)
    );
    assert_eq!(
        (
            diags[2].image_index,
            diags[2].tp,
            diags[2].duplicate_errors,
            diags[2].class_confusions,
            diags[2].fn_count
        ),
        (2, 1, 1, 1, 1)
    );
    assert_eq!((diags[3].image_index, diags[3].tp, diags[3].fp, diags[3].fn_count), (3, 0, 0, 4));

    // Worst-image ranking: img_3 (4 errors) > img_1 (3 errors, f1=0.0) > img_2 (3 errors, f1=0.4);
    // img_0 (0 errors) is excluded.
    let rep = seq_ev.compute();
    let worst_ids: Vec<&str> = rep.worst_images(10).iter().map(|d| d.image_id.as_deref().unwrap()).collect();
    assert_eq!(worst_ids, vec!["img_3_all_missed.jpg", "img_1_loc_and_bg.jpg", "img_2_cls_and_dup.jpg"]);

    // JSON round-trip preserves `image_diagnostics` exactly.
    let json = serde_json::to_string(&rep).unwrap();
    let back: EvalReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.image_diagnostics, rep.image_diagnostics);
    assert_eq!(back.worst_images(2), rep.worst_images(2));

    // 3. `coco_compatible: true` ignores out-of-range truths and detections matching them consistently across
    // AP metrics and per-image diagnostics.
    let mut coco_ev =
        Evaluator::with_config(1, EvalConfig { record_diagnostics: true, ..EvalConfig::coco() });
    coco_ev.add_image(
        &[
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(0.0, 0.0, 90_000.0, 100_000.0, 0.85, 0), // matches COCO-ignored truth at IoU 0.75 -> ignored
        ],
        &[
            gt(0.0, 0.0, 10.0, 10.0, 0),
            gt(0.0, 0.0, 120_000.0, 100_000.0, 0), // area > 1e10 -> COCO-ignored
        ],
    );
    let coco_rep = coco_ev.compute();
    assert_eq!(coco_rep.num_truths, 1);
    assert_eq!(coco_rep.image_diagnostics[0].num_truths, 1);
    assert_eq!(coco_rep.image_diagnostics[0].num_preds, 1);
    assert_eq!(coco_rep.image_diagnostics[0].tp, 1);
    assert_eq!(coco_rep.image_diagnostics[0].fp, 0);
    assert_eq!(coco_rep.image_diagnostics[0].fn_count, 0);
    assert!(coco_rep.worst_images(5).is_empty());

    // 4. Evaluators built with non-finite diagnostic thresholds normalize cleanly and merge without panicking.
    let nan_cfg = EvalConfig {
        record_diagnostics: true,
        diag_conf: f32::NAN,
        diag_tp_iou: f32::NAN,
        diag_bg_iou: f32::NAN,
        ..Default::default()
    };
    let mut nan_a = Evaluator::with_config(1, nan_cfg.clone());
    let mut nan_b = Evaluator::with_config(1, nan_cfg);
    nan_a.add_image(&[det(0.0, 0.0, 10.0, 10.0, 0.9, 0)], &[gt(0.0, 0.0, 10.0, 10.0, 0)]);
    nan_b.add_image(&[], &[gt(0.0, 0.0, 10.0, 10.0, 0)]);
    nan_a.merge(nan_b);
    assert_eq!(nan_a.image_diagnostics().len(), 2);
    assert_eq!(nan_a.image_diagnostics()[1].image_index, 1);
}

#[test]
#[should_panic(expected = "config mismatch")]
fn evaluator_merge_panics_on_mismatched_diagnostics_config() {
    let mut a = Evaluator::with_config(1, EvalConfig::with_diagnostics());
    let b = Evaluator::new(1);
    a.merge(b);
}
