use super::*;
use crate::api::{Backend, ImageSize};

fn fake_model(id: &str, name: &str, classes: &[&str]) -> Model {
    let mut base = crate::api::mock::MockBackend::new(4200, true).models().unwrap().remove(0);
    base.id = id.into();
    base.name = name.into();
    base.classes = classes.iter().map(|s| (*s).into()).collect();
    base
}

fn fake_box(class: u32, name: &str, score: f32) -> ScoredBox {
    ScoredBox { class, name: name.into(), score, x1: 10., y1: 10., x2: 50., y2: 50. }
}

fn fake_pred(model_id: &str, total_ms: f64, boxes: Vec<ScoredBox>) -> PredictResponse {
    PredictResponse {
        model_id: model_id.into(),
        image: ImageSize { width: 640, height: 480 },
        boxes,
        timing: Timing {
            preprocess_ms: 1.0,
            inference_ms: (total_ms - 1.5).max(0.1),
            postprocess_ms: 0.5,
            total_ms,
        },
        device: "cpu".into(),
        cold: false,
    }
}

#[test]
fn default_model_b_prefers_same_classes_then_distinct_then_fallback() {
    let a = fake_model("m_a", "shapes-n", &["circle", "square"]);
    let b_other_cls = fake_model("m_b", "coco-n", &["person"]);
    let c_same_cls = fake_model("m_c", "shapes-s", &["circle", "square"]);
    let models = vec![a.clone(), b_other_cls.clone(), c_same_cls.clone()];
    assert_eq!(pick_default_model_b(Some(&a), &models).map(|m| m.id.as_str()), Some("m_c"));
    let diff_only = vec![a.clone(), b_other_cls];
    assert_eq!(pick_default_model_b(Some(&a), &diff_only).map(|m| m.id.as_str()), Some("m_b"));
    let single = vec![a.clone()];
    assert_eq!(pick_default_model_b(Some(&a), &single).map(|m| m.id.as_str()), Some("m_a"));
}

#[test]
fn summarize_comparison_computes_deltas_speedup_and_class_sets() {
    let a_boxes = vec![fake_box(0, "circle", 0.9), fake_box(1, "square", 0.8)];
    let b_boxes =
        vec![fake_box(0, "circle", 0.95), fake_box(0, "circle", 0.85), fake_box(2, "triangle", 0.7)];
    let pa = fake_pred("m_a", 20.0, a_boxes.clone());
    let pb = fake_pred("m_b", 10.0, b_boxes.clone());
    let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
    let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

    let s = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
    assert_eq!((s.count_a, s.count_b, s.count_delta), (Some(2), Some(3), Some(1)));
    assert_eq!(format_signed_delta(s.count_delta), "+1");
    assert_eq!(s.shared_classes, vec!["circle"]);
    assert_eq!(s.only_a_classes, vec!["square"]);
    assert_eq!(s.only_b_classes, vec!["triangle"]);
    assert_eq!(format_latency_comparison(&s), "-10.0 ms · B 2.0× faster");

    // Partial availability (B pending/failed) never fabricates zero counts or deltas.
    let partial = summarize_comparison(Some(&ra), None, Some(&pa), None);
    assert_eq!((partial.count_a, partial.count_b, partial.count_delta), (Some(2), None, None));
    assert_eq!(format_signed_delta(partial.count_delta), "—");
    assert_eq!(format_latency_comparison(&partial), "—");
    assert!(partial.shared_classes.is_empty());
}

#[test]
fn compare_json_serializes_both_sides_and_partial_states() {
    let a_boxes = vec![fake_box(0, "circle", 0.91)];
    let b_boxes = vec![fake_box(0, "circle", 0.88), fake_box(1, "square", 0.77)];
    let pa = fake_pred("m_a", 12.0, a_boxes.clone());
    let pb = fake_pred("m_b", 18.0, b_boxes.clone());
    let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
    let rb: Vec<&ScoredBox> = b_boxes.iter().collect();
    let summary = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
    let json = compare_detections_json(
        "shapes / val_001.png",
        (640, 480),
        0.25,
        None,
        Some(0.7),
        None,
        &summary,
        Some(&CompareSideExport {
            model: "shapes-n",
            model_id: "m_a",
            iou: Some(0.7),
            per_class_conf: None,
            pred: &pa,
            boxes: &ra,
        }),
        Some(&CompareSideExport {
            model: "shapes-s",
            model_id: "m_b",
            iou: None,
            per_class_conf: None,
            pred: &pb,
            boxes: &rb,
        }),
    )
    .expect("serialize compare JSON");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["mode"], "compare");
    assert_eq!(v["comparison"]["count_delta"], 1);
    assert_eq!(v["comparison"]["shared_classes"][0], "circle");
    assert_eq!(v["comparison"]["only_b_classes"][0], "square");
    assert_eq!(v["a"]["model"], "shapes-n");
    assert_eq!(v["a"]["cold"], false);
    assert!(v["a"].get("iou").is_some());
    assert_eq!(v["b"]["model"], "shapes-s");
    assert_eq!(v["b"]["cold"], false);
    assert!(v["b"].get("iou").is_none());
}

#[test]
fn cold_start_prediction_suppresses_speedup_claim_and_serializes_cold_flags() {
    let a_boxes = vec![fake_box(0, "circle", 0.91)];
    let b_boxes = vec![fake_box(0, "circle", 0.88)];
    let mut pa = fake_pred("m_a", 2.0, a_boxes.clone());
    pa.cold = true;
    let pb = fake_pred("m_b", 0.1, b_boxes.clone());
    let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
    let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

    let s_a = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
    assert_eq!(s_a.cold_a, Some(true));
    assert_eq!(s_a.cold_b, Some(false));
    assert_eq!(s_a.speedup_ratio, None, "cold Model A must not produce a speedup ratio");
    assert_eq!(format_latency_comparison(&s_a), "Warm-up (A)");

    let mut pb_cold = pb.clone();
    pb_cold.cold = true;
    let s_both = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb_cold));
    assert_eq!(s_both.speedup_ratio, None);
    assert_eq!(format_latency_comparison(&s_both), "Warm-up (A & B)");

    let mut pa_warm = pa.clone();
    pa_warm.cold = false;
    let s_b = summarize_comparison(Some(&ra), Some(&rb), Some(&pa_warm), Some(&pb_cold));
    assert_eq!(s_b.speedup_ratio, None);
    assert_eq!(format_latency_comparison(&s_b), "Warm-up (B)");

    let cap = compare_pane_caption_parts("A", "shapes-n", Some(1), Some(2.0), true);
    assert_eq!(cap.title, "A · shapes-n");
    assert_eq!(cap.meta, "1 det · 2.0 ms (cold)");

    let json = compare_detections_json(
        "val_001.png",
        (640, 480),
        0.25,
        None,
        None,
        None,
        &s_a,
        Some(&CompareSideExport {
            model: "shapes-n",
            model_id: "m_a",
            iou: None,
            per_class_conf: None,
            pred: &pa,
            boxes: &ra,
        }),
        Some(&CompareSideExport {
            model: "shapes-s",
            model_id: "m_b",
            iou: None,
            per_class_conf: None,
            pred: &pb,
            boxes: &rb,
        }),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["comparison"]["cold_a"], true);
    assert_eq!(v["comparison"]["cold_b"], false);
    assert!(v["comparison"]["speedup_ratio"].is_null());
    assert_eq!(v["a"]["cold"], true);
    assert_eq!(v["b"]["cold"], false);
}

#[test]
fn class_diff_status_filtering_and_json_payload_cover_both_only_a_and_only_b() {
    let a_boxes = vec![fake_box(0, "circle", 0.90), fake_box(1, "square", 0.80)];
    let b_boxes =
        vec![fake_box(0, "circle", 0.95), fake_box(0, "circle", 0.85), fake_box(2, "triangle", 0.70)];
    let pa = fake_pred("m_a", 12.0, a_boxes.clone());
    let pb = fake_pred("m_b", 10.0, b_boxes.clone());
    let ra: Vec<&ScoredBox> = a_boxes.iter().collect();
    let rb: Vec<&ScoredBox> = b_boxes.iter().collect();

    let summary = summarize_comparison(Some(&ra), Some(&rb), Some(&pa), Some(&pb));
    assert_eq!(summary.filtered_classes(CompareClassFilter::All).len(), 3);
    let both = summary.filtered_classes(CompareClassFilter::Both);
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].name, "circle");
    assert_eq!(both[0].status, Some(CompareClassStatus::Both));
    assert!((both[0].mean_score_a.unwrap() - 0.90).abs() < 1e-4);
    assert!((both[0].mean_score_b.unwrap() - 0.90).abs() < 1e-4);

    let only_a = summary.filtered_classes(CompareClassFilter::OnlyA);
    assert_eq!(only_a.len(), 1);
    assert_eq!(only_a[0].name, "square");
    assert_eq!(only_a[0].status, Some(CompareClassStatus::OnlyA));

    let only_b = summary.filtered_classes(CompareClassFilter::OnlyB);
    assert_eq!(only_b.len(), 1);
    assert_eq!(only_b[0].name, "triangle");
    assert_eq!(only_b[0].status, Some(CompareClassStatus::OnlyB));

    let json = compare_detections_json(
        "val_001.png",
        (640, 480),
        0.25,
        None,
        Some(0.7),
        None,
        &summary,
        Some(&CompareSideExport {
            model: "shapes-n",
            model_id: "m_a",
            iou: Some(0.7),
            per_class_conf: None,
            pred: &pa,
            boxes: &ra,
        }),
        Some(&CompareSideExport {
            model: "shapes-s",
            model_id: "m_b",
            iou: Some(0.7),
            per_class_conf: None,
            pred: &pb,
            boxes: &rb,
        }),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["comparison"]["class_diff"]["both"], 1);
    assert_eq!(v["comparison"]["class_diff"]["only_a"], 1);
    assert_eq!(v["comparison"]["class_diff"]["only_b"], 1);
    let rows = v["comparison"]["classes"].as_array().unwrap();
    assert!(rows.iter().any(|r| r["name"] == "circle" && r["status"] == "both"));
    assert!(rows.iter().any(|r| r["name"] == "square" && r["status"] == "only_a"));
    assert!(rows.iter().any(|r| r["name"] == "triangle" && r["status"] == "only_b"));
}

#[test]
fn playground_compare_renders_class_diff_filters_and_exports_ab_json() {
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::store::Load;

    let (mut app, window) = open();
    let dir = folo_test_support::TestDir::new("desktop-play-cmp-diff-ui");
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.set_play_compare(true, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, cx| {
        let ds = this.play_sample_dataset().expect("sample dataset for default model");
        let img = this
            .store
            .images
            .get(&ds)
            .and_then(|l| l.ready())
            .and_then(|list| list.first())
            .map(|m| m.id.clone())
            .unwrap_or_else(|| "img_00801".into());
        this.play_dataset_image(&ds, &img, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    // Inject deterministic real-shaped A & B predictions covering Both, Only A, and Only B.
    drive(&mut app, window, |this, _, cx| {
        let mid_a = this.play_shown_model().map(|m| m.id.clone()).unwrap_or_else(|| "m_a".into());
        let mid_b = this.play_shown_model_b().map(|m| m.id.clone()).unwrap_or_else(|| "m_b".into());
        this.play_pred = Load::Ready(fake_pred(
            &mid_a,
            11.0,
            vec![fake_box(0, "circle", 0.91), fake_box(1, "square", 0.82)],
        ));
        this.play_compare.pred_b = Load::Ready(fake_pred(
            &mid_b,
            8.5,
            vec![fake_box(0, "circle", 0.94), fake_box(2, "triangle", 0.77)],
        ));
        cx.notify();
    });
    app.frame(window.into());

    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "play-cmp-cls-all",
            "play-cmp-cls-both",
            "play-cmp-cls-a",
            "play-cmp-cls-b",
            "play-export-json",
            "play-copy-json",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
    });

    let out = dir.path().join("compare-ab.json");
    app.platform().answer_prompts(Some(vec![out.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_json(cx));
    app.run_until_idle();
    let text = std::fs::read_to_string(&out).expect("compare JSON written");
    let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    assert_eq!(v["mode"], "compare");
    assert_eq!(v["comparison"]["class_diff"]["both"], 1);
    assert_eq!(v["comparison"]["class_diff"]["only_a"], 1);
    assert_eq!(v["comparison"]["class_diff"]["only_b"], 1);
    drive(&mut app, window, |this, _, _| {
        assert!(this.toasts.iter().any(|t| t.title == "Exported A/B compare JSON"));
    });
}

#[test]
fn playground_compare_delta_ignores_stale_side_during_rerun_interval() {
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::store::Load;

    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.set_play_compare(true, cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, _, cx| {
        let ds = this.play_sample_dataset().expect("sample dataset");
        this.play_dataset_image(&ds, "img_00801", cx);
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }
    // Simulate an in-flight setting change (device switch): Model A already answered fresh for
    // the new target while Model B still holds an older Load::Ready prediction from the old target.
    drive(&mut app, window, |this, _, cx| {
        let mid_a = this.play_model.clone().unwrap();
        let mid_b = this.play_compare.model_b.clone().unwrap();
        let img_gen =
            this.play_compare.pred_a_meta.as_ref().expect("loaded image populates pred_a_meta").image_gen;
        let iou_bits = this.play_iou.to_bits();
        let floor = this.play_conf.min(crate::app::PLAY_CONF_FLOOR);
        this.play_pred_pending = false;
        this.play_compare.pred_b_pending = false;
        this.play_pred = Load::Ready(fake_pred(&mid_a, 9.0, vec![fake_box(0, "circle", 0.91)]));
        this.play_compare.pred_a_meta = Some(PlayPredMeta {
            image_gen: img_gen,
            model_id: mid_a,
            iou_bits,
            target: this.play_target,
            floor,
            per_class_conf: BTreeMap::new(),
        });
        // Model B still has a Ready prediction, but from a different device target (stale!).
        let stale_target = Some(crate::local::Target::Cpu);
        assert_ne!(stale_target, this.play_target);
        this.play_compare.pred_b = Load::Ready(fake_pred(&mid_b, 12.0, vec![fake_box(1, "square", 0.88)]));
        this.play_compare.pred_b_meta = Some(PlayPredMeta {
            image_gen: img_gen,
            model_id: mid_b,
            iou_bits,
            target: stale_target,
            floor,
            per_class_conf: BTreeMap::new(),
        });
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());

    drive(&mut app, window, |this, _, _| {
        assert!(this.pred_a_is_fresh());
        assert!(!this.pred_b_is_fresh(), "stale Model B must not be treated as fresh");
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(
            !keys.iter().any(|k| k == "play-cmp-cls-both"),
            "Both/Only A/Only B filter chips must not mix fresh A with stale B: {keys:?}"
        );
    });
}
