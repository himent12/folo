//! Native integration tests for `folo-wasm`'s stateful ByteTrack [`Tracker`]:
//! multi-frame ID persistence, crossing stability, stage-2 low-confidence occlusion recovery, coasting and
//! retirement, `min_hits` confirmation, class-agnostic tracking, `reset()`, JSON shape / JS options parsing,
//! invalid option rejection, and end-to-end `Detector` (v1 & `folo2`) → `Tracker` wiring.

use folo_core::no_grad;
use folo_core::rng::Rng;
use folo_nn::Module;
use folo_test_support::TestDir;
use folo_vision::boxes::Xyxy;
use folo_vision::nms::Detection;
use folo_vision::{ByteTracker, FoloV2Config, FoloV2Model, TrackConfig};
use folo_wasm::{
    BackendChoice, DetectOptions, Detector, LoadOptions, TrackInputBox, TrackStateLabel, Tracker,
    TrackerOptions,
};

fn ibox(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32, name: Option<&str>) -> TrackInputBox {
    TrackInputBox { class, name: name.map(str::to_string), score, x1, y1, x2, y2 }
}

fn det(b: &TrackInputBox) -> Detection {
    Detection { bbox: Xyxy::new(b.x1, b.y1, b.x2, b.y2), score: b.score, class: b.class }
}

#[test]
fn matches_folo_vision_bytetracker_across_motion_occlusion_and_reset() {
    let opts = TrackerOptions {
        track_buffer: 3,
        classes: vec!["person".into(), "car".into()],
        ..Default::default()
    };
    let mut wasm_trk = Tracker::new(opts.clone()).unwrap();
    let mut native_trk = ByteTracker::new(TrackConfig { track_buffer: 3, ..Default::default() }).unwrap();

    assert_eq!(wasm_trk.info().frame_id, 0);
    assert_eq!(wasm_trk.info().active_tracks, 0);
    assert_eq!(wasm_trk.info().lost_tracks, 0);
    assert_eq!(wasm_trk.config().track_buffer, 3);

    let mut id_a = 0u64;
    let mut id_b = 0u64;
    for f in 0..8u32 {
        let ax = 50.0 + f as f32 * 12.0;
        let bx = 170.0 - f as f32 * 12.0;
        let mut frame = vec![
            ibox(ax, 100.0, ax + 50.0, 160.0, 0.90, 0, None),
            ibox(bx, 110.0, bx + 50.0, 170.0, 0.88, 1, None),
        ];
        if f % 2 == 1 {
            frame.reverse();
        }
        let native_dets: Vec<Detection> = frame.iter().map(det).collect();
        let got = wasm_trk.update(&frame);
        let want = native_trk.update(&native_dets);

        assert_eq!(got.len(), want.len());
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.track_id, w.track_id);
            assert_eq!(g.age, w.age);
            assert_eq!(g.hits, w.hits);
            assert_eq!(g.state, TrackStateLabel::Tracked);
            assert_eq!(g.class, w.class_id);
            assert_eq!(g.name, if w.class_id == 0 { "person" } else { "car" });
            assert_eq!(
                (g.score, g.x1, g.y1, g.x2, g.y2),
                (w.score, w.bbox.x1, w.bbox.y1, w.bbox.x2, w.bbox.y2)
            );
        }
        if f == 0 {
            id_a = got.iter().find(|t| t.class == 0).unwrap().track_id;
            id_b = got.iter().find(|t| t.class == 1).unwrap().track_id;
            assert_ne!(id_a, id_b);
        }
    }

    // Complete 2-frame occlusion: tracks coast in lost buffer and re-identify with identical IDs.
    assert!(wasm_trk.update(&[]).is_empty());
    assert!(native_trk.update(&[]).is_empty());
    assert_eq!((wasm_trk.active_tracks(), wasm_trk.lost_tracks(), wasm_trk.frame_id()), (0, 2, 9));

    assert!(wasm_trk.update(&[]).is_empty());
    assert!(native_trk.update(&[]).is_empty());
    assert_eq!(wasm_trk.info().lost_tracks, 2);

    let f_re = 10u32;
    let ax = 50.0 + f_re as f32 * 12.0;
    let bx = 170.0 - f_re as f32 * 12.0;
    let reappear = vec![
        ibox(ax, 100.0, ax + 50.0, 160.0, 0.91, 0, None),
        ibox(bx, 110.0, bx + 50.0, 170.0, 0.89, 1, None),
    ];
    let got_re = wasm_trk.update(&reappear);
    let want_re = native_trk.update(&reappear.iter().map(det).collect::<Vec<_>>());
    assert_eq!(got_re.len(), 2);
    assert_eq!(want_re.len(), 2);
    assert!(got_re.iter().any(|t| t.track_id == id_a && t.hits == 9 && t.age == 11));
    assert!(got_re.iter().any(|t| t.track_id == id_b && t.hits == 9 && t.age == 11));

    // Exceed track_buffer (3 missed frames -> retired on 4th missed frame).
    for _ in 0..4 {
        assert!(wasm_trk.update(&[]).is_empty());
    }
    assert_eq!((wasm_trk.active_tracks(), wasm_trk.lost_tracks()), (0, 0));

    // Reset restores frame_id = 0 and track_id = 1 while keeping configured `classes`.
    wasm_trk.reset();
    assert_eq!(wasm_trk.info().frame_id, 0);
    let fresh = wasm_trk.update(&[ibox(10.0, 10.0, 50.0, 50.0, 0.9, 1, None)]);
    assert_eq!(fresh.len(), 1);
    assert_eq!((fresh[0].track_id, fresh[0].class, fresh[0].name.as_str()), (1, 1, "car"));
}

#[test]
fn stage_two_low_confidence_occlusion_recovery_and_dynamic_names() {
    let mut trk = Tracker::default();

    for step in 0..=6u32 {
        let cx = 100.0 + step as f32 * 10.0;
        let out = trk.update(&[
            ibox(cx - 30.0, 60.0, cx + 30.0, 140.0, 0.90, 2, Some("bicycle")),
            // Lone low-confidence noise must never spawn a track.
            ibox(400.0, 400.0, 450.0, 450.0, 0.20, 2, Some("bicycle")),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].track_id, out[0].age, out[0].hits, out[0].name.as_str()),
            (1, step + 1, step + 1, "bicycle")
        );
    }

    // Low-confidence occluded detection (0.20 in [0.1, 0.5)) recovers ID 1 and retains dynamic class name.
    let recovered = trk.update(&[ibox(149.0 - 30.0, 60.0, 149.0 + 30.0, 140.0, 0.20, 2, None)]);
    assert_eq!(recovered.len(), 1);
    assert_eq!((recovered[0].track_id, recovered[0].age, recovered[0].hits), (1, 8, 8));
    assert_eq!(recovered[0].name, "bicycle");
    assert!((recovered[0].score - 0.20).abs() < 1e-6);

    // Reset clears dynamically learned names (falls back to "class2" when no name is passed).
    trk.reset();
    let after_reset = trk.update(&[ibox(10.0, 10.0, 60.0, 60.0, 0.9, 2, None)]);
    assert_eq!((after_reset[0].track_id, after_reset[0].name.as_str()), (1, "class2"));
}

#[test]
fn discarded_boxes_cannot_relabel_tracks_and_same_frame_labels_are_order_independent() {
    let mut trk = Tracker::default();
    let t1 = trk.update(&[ibox(20.0, 20.0, 80.0, 80.0, 0.9, 0, Some("person"))]);
    assert_eq!((t1.len(), t1[0].name.as_str()), (1, "person"));

    // Degenerate, NaN, and sub-low_thresh detections carrying bogus names must NOT overwrite "person".
    let t2 = trk.update(&[
        ibox(22.0, 20.0, 82.0, 80.0, 0.9, 0, None),
        ibox(80.0, 20.0, 20.0, 80.0, 0.95, 0, Some("degenerate")),
        ibox(20.0, 20.0, 80.0, 80.0, f32::NAN, 0, Some("nan_score")),
        ibox(20.0, 20.0, 80.0, 80.0, 0.05, 0, Some("too_low")),
    ]);
    assert_eq!((t2.len(), t2[0].track_id, t2[0].name.as_str()), (1, 1, "person"));

    // Conflicting eligible labels in one frame resolve deterministically independent of input order.
    let a = ibox(10.0, 10.0, 60.0, 60.0, 0.95, 3, Some("truck"));
    let b = ibox(100.0, 100.0, 150.0, 150.0, 0.80, 3, Some("van"));
    let mut fwd = Tracker::default();
    let mut rev = Tracker::default();
    let out_fwd = fwd.update(&[a.clone(), b.clone()]);
    let out_rev = rev.update(&[b, a]);
    assert_eq!(out_fwd, out_rev);
    assert!(out_fwd.iter().all(|t| t.name == "truck"));
}

#[test]
fn min_hits_class_agnostic_degenerate_filtering_and_json_contract() {
    // min_hits = 2 delays emission until second consecutive match and ignores degenerate/NaN boxes.
    let mut trk =
        Tracker::new(TrackerOptions { min_hits: 2, class_aware: true, ..Default::default() }).unwrap();
    let good = ibox(20.0, 20.0, 80.0, 80.0, 0.9, 0, Some("dog"));
    let bad_deg = ibox(80.0, 20.0, 20.0, 80.0, 0.9, 0, Some("dog"));
    let bad_nan = ibox(20.0, 20.0, 80.0, 80.0, f32::NAN, 0, Some("dog"));
    assert!(trk.update(&[good.clone(), bad_deg, bad_nan]).is_empty());
    let confirmed = trk.update(&[good]);
    assert_eq!(confirmed.len(), 1);
    assert_eq!((confirmed[0].track_id, confirmed[0].hits, confirmed[0].age), (1, 2, 2));

    // Verify JSON field names match TS_TYPES (`FoloTrackedBox`, `FoloTrackerInfo`, `FoloTrackConfig`).
    let box_json = serde_json::to_value(&confirmed[0]).unwrap();
    let mut keys: Vec<&str> = box_json.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["age", "class", "hits", "name", "score", "state", "track_id", "x1", "x2", "y1", "y2"]);
    assert_eq!(box_json["state"], "tracked");
    assert_eq!(box_json["track_id"], 1);

    let info_json = serde_json::to_value(trk.info()).unwrap();
    assert_eq!(info_json["frameId"], 2);
    assert_eq!(info_json["activeTracks"], 1);
    assert_eq!(info_json["lostTracks"], 0);
    assert_eq!(info_json["config"]["minHits"], 2);
    assert_eq!(info_json["config"]["classAware"], true);

    // Class-agnostic tracking allows class change across stages and track_buffer = 0 retires immediately.
    let mut ag =
        Tracker::new(TrackerOptions { class_aware: false, track_buffer: 0, ..Default::default() }).unwrap();
    let t1 = ag.update(&[ibox(10.0, 10.0, 60.0, 60.0, 0.9, 0, Some("a"))]);
    assert_eq!((t1[0].track_id, t1[0].class, t1[0].name.as_str()), (1, 0, "a"));
    let t2 = ag.update(&[ibox(12.0, 10.0, 62.0, 60.0, 0.25, 2, Some("c"))]);
    assert_eq!((t2.len(), t2[0].track_id, t2[0].class, t2[0].name.as_str()), (1, 1, 2, "c"));
    assert!(ag.update(&[]).is_empty());
    assert_eq!((ag.active_tracks(), ag.lost_tracks()), (0, 0));
}

#[test]
fn parses_js_json_options_and_boxes_and_rejects_invalid_configs() {
    let opts: TrackerOptions = serde_json::from_str(
        r#"{"highThresh":0.55,"lowThresh":0.15,"newTrackThresh":0.65,"matchIou":0.25,"secondMatchIou":0.45,"tentativeMatchIou":0.35,"trackBuffer":20,"minHits":2,"classAware":false,"classes":["cat"],"unknown":1}"#,
    )
    .unwrap();
    let cfg = opts.to_track_config().unwrap();
    assert_eq!(cfg.high_thresh, 0.55);
    assert_eq!(cfg.low_thresh, 0.15);
    assert_eq!(cfg.new_track_thresh, 0.65);
    assert_eq!(cfg.match_iou, 0.25);
    assert_eq!(cfg.second_match_iou, 0.45);
    assert_eq!(cfg.tentative_match_iou, 0.35);
    assert_eq!((cfg.track_buffer, cfg.min_hits, cfg.class_aware), (20, 2, false));

    // Input boxes accept `class`, `class_id`, or `classId`, with optional `name`.
    let boxes: Vec<TrackInputBox> = serde_json::from_str(
        r#"[
            {"class":0,"name":"person","score":0.9,"x1":1,"y1":2,"x2":30,"y2":40},
            {"class_id":1,"score":0.8,"x1":5,"y1":6,"x2":50,"y2":60},
            {"classId":2,"score":0.7,"x1":7,"y1":8,"x2":70,"y2":80}
        ]"#,
    )
    .unwrap();
    assert_eq!(boxes.len(), 3);
    assert_eq!((boxes[0].class, boxes[1].class, boxes[2].class), (0, 1, 2));

    // Invalid configs reject cleanly.
    for bad in [
        TrackerOptions { high_thresh: -0.1, ..Default::default() },
        TrackerOptions { high_thresh: 1.1, ..Default::default() },
        TrackerOptions { high_thresh: f32::NAN, ..Default::default() },
        TrackerOptions { low_thresh: Some(-0.01), ..Default::default() },
        TrackerOptions { low_thresh: Some(0.6), high_thresh: 0.5, ..Default::default() },
        TrackerOptions { new_track_thresh: Some(0.4), high_thresh: 0.5, ..Default::default() },
        TrackerOptions { match_iou: 1.2, ..Default::default() },
        TrackerOptions { second_match_iou: f32::INFINITY, ..Default::default() },
        TrackerOptions { tentative_match_iou: -0.5, ..Default::default() },
        TrackerOptions { min_hits: 0, ..Default::default() },
    ] {
        assert!(Tracker::new(bad.clone()).is_err(), "expected error for {bad:?}");
    }
}

#[test]
fn folo2_detector_boxes_feed_directly_into_tracker() {
    let mut m = FoloV2Model::with_rng(FoloV2Config::new("n", 2).unwrap(), &mut Rng::seed(11)).unwrap();
    m.set_classes(vec!["alpha".into(), "beta".into()]).unwrap();
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            if t.name.starts_with("head.cls.") && t.name.ends_with(".out.bias") {
                t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
            }
        }
    }
    let dir = TestDir::new("wasm-v2-track");
    let path = dir.join("v2.folo");
    m.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    let det = pollster::block_on(Detector::load(
        bytes,
        LoadOptions { backend: BackendChoice::Cpu, imgsz: Some(128), ..Default::default() },
    ))
    .unwrap();
    let (w, h) = (128u32, 96u32);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8, 255]);
        }
    }
    let res = pollster::block_on(det.detect(
        &rgba,
        w,
        h,
        DetectOptions { conf: 0.3, iou: 0.7, max_det: 20, ..Default::default() },
    ))
    .unwrap();
    assert!(!res.response.boxes.is_empty(), "test needs detections from folo2");

    let mut trk = Tracker::new(TrackerOptions {
        high_thresh: 0.35,
        low_thresh: Some(0.2),
        new_track_thresh: Some(0.35),
        classes: det.info().classes,
        ..Default::default()
    })
    .unwrap();
    let input_boxes: Vec<TrackInputBox> = res.response.boxes.iter().map(TrackInputBox::from).collect();
    let frame1 = trk.update(&input_boxes);
    assert!(!frame1.is_empty());
    let frame2 = trk.update(&input_boxes);
    assert_eq!(frame2.len(), frame1.len());
    for (a, b) in frame1.iter().zip(&frame2) {
        assert_eq!(a.track_id, b.track_id);
        assert_eq!((a.age, b.age), (1, 2));
        assert_eq!((a.hits, b.hits), (1, 2));
        assert!(b.name == "alpha" || b.name == "beta");
    }
}
