//! Deterministic tests for Benchmark Compare and External Predictions entry support.

use folo_test_support::TestDir;

use crate::api::mock::MockBackend;
use crate::api::*;
use crate::app::Route;
use crate::app::tests::{drive, open};
use crate::bench::*;

#[test]
fn compare_selection_caveats_best_in_row_and_intervals() {
    let api = MockBackend::new(4200, true);
    let benches = api.benches().unwrap();
    let bench = benches.iter().find(|b| b.id == "bn_3c90").unwrap();
    let entries = api.bench_entries(&bench.id).unwrap();

    // Default selection picks top COMPARE_MAX finished entries; preferred baseline goes first.
    let defaults = default_compare_entries(&entries, None);
    assert_eq!(defaults.len(), COMPARE_MAX);
    assert_eq!(defaults[0], "be_3c01");
    let pinned = default_compare_entries(&entries, Some("be_3c03"));
    assert_eq!(pinned[0], "be_3c03");
    assert_eq!(pinned.len(), COMPARE_MAX);

    // Reconcile drops missing / unfinished entries and deduplicates.
    let reconciled = reconcile_compare_entries(
        &["be_3c01".into(), "be_3c06".into(), "be_3c01".into(), "missing".into(), "be_3c02".into()],
        &entries,
    );
    assert_eq!(reconciled, vec!["be_3c01".to_string(), "be_3c02".to_string()]);

    // Caveats detect fewer images, conditions left out, no confidences, missing classes, stale fingerprint.
    let lead = entries.iter().find(|e| e.id == "be_3c01").unwrap().clone();
    let partial_cls = entries.iter().find(|e| e.id == "be_3c04").unwrap().clone();
    let agent = entries.iter().find(|e| e.id == "be_3c05").unwrap().clone();
    let mut old = lead.clone();
    old.id = "be_old".into();
    old.name = "older-run".into();
    old.fingerprint = "old-fp".into();
    let cavs = compare_caveats(&[lead.clone(), partial_cls, agent, old.clone()], bench);
    assert!(cavs.iter().any(|c| c.contains("earlier version")), "{cavs:?}");
    assert!(cavs.iter().any(|c| c.contains("Fewer images")), "{cavs:?}");
    assert!(cavs.iter().any(|c| c.contains("Conditions left out")), "{cavs:?}");
    assert!(cavs.iter().any(|c| c.contains("No confidences")), "{cavs:?}");
    assert!(cavs.iter().any(|c| c.contains("does not know")), "{cavs:?}");

    // best_in_row skips stale entries and marks ties accurately.
    let best = best_in_row(&[Some(80.0), Some(95.0), Some(80.04)], &[false, true, false], 1, true);
    assert_eq!(best, vec![true, false, true], "stale 95.0 is skipped; 80.0 and 80.04 tie at 80.0");
    let speed_best = best_in_row(&[Some(4.1), Some(2.3), None], &[false, false, false], 1, false);
    assert_eq!(speed_best, vec![false, true, false], "lower ms/img wins speed");

    // Intervals domain and verdict separate non-overlapping CIs and note overlapping ones.
    let mut a = lead.clone();
    a.map50_95 = Some(0.80);
    a.ci95 = Some([0.75, 0.85]);
    let mut b_lo = lead.clone();
    b_lo.id = "be_b".into();
    b_lo.name = "weaker".into();
    b_lo.map50_95 = Some(0.60);
    b_lo.ci95 = Some([0.55, 0.65]);
    let dom = intervals_domain(&[a.clone(), b_lo.clone()]).expect("domain");
    assert!(dom.0 < 0.55 && dom.1 > 0.85, "{dom:?}");
    assert!(intervals_verdict(&[a.clone(), b_lo], bench).contains("non-overlapping"));

    let mut b_ov = lead;
    b_ov.id = "be_ov".into();
    b_ov.name = "close".into();
    b_ov.map50_95 = Some(0.78);
    b_ov.ci95 = Some([0.72, 0.84]);
    assert!(intervals_verdict(&[a, b_ov], bench).contains("overlapping"));
}

#[test]
fn compare_route_renders_switches_baseline_and_adds_removes_entries() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    // Open Compare from the benchmark leaderboard.
    drive(&mut app, window, |this, w, cx| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "bench-compare-open"), "Compare button visible: {keys:?}");
        assert!(
            keys.iter().any(|k| k == "bench-external-open"),
            "Import predictions button visible: {keys:?}"
        );
        this.open_bench_compare("bn_3c90", None, w, cx);
        assert_eq!(this.route, Route::BenchCompare("bn_3c90".into()));
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    // Verify 4 compared entries loaded, then promote entry 1 to baseline and remove one to show Add entry.
    drive(&mut app, window, |this, _, cx| {
        let cmp = this.bench.compare.get("bn_3c90").cloned().unwrap_or_default();
        assert_eq!(cmp.len(), 4, "default compare loads top 4 finished entries");
        let second = cmp[1].clone();
        this.compare_make_baseline("bn_3c90", &second, cx);
        assert_eq!(this.bench.compare.get("bn_3c90").unwrap()[0], second);
        let last = this.bench.compare.get("bn_3c90").unwrap()[3].clone();
        this.compare_remove_entry("bn_3c90", &last, cx);
        assert_eq!(this.bench.compare.get("bn_3c90").unwrap().len(), 3);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(
            keys.iter().any(|k| k == "bench-cmp-add"),
            "Add entry picker appears when < 4 entries: {keys:?}"
        );
    });

    // Route parsing round-trip.
    assert_eq!(Route::parse("bench-compare:bn_3c90"), Some(Route::BenchCompare("bn_3c90".into())));
    assert_eq!(Route::parse("bench-compare:"), None);
}

#[test]
fn external_predictions_parse_validate_chunk_and_upload_end_to_end() {
    let api = MockBackend::new(4200, true);
    let bench = api.benches().unwrap().into_iter().find(|b| b.id == "bn_7a04").unwrap();
    let cat: Vec<String> = api.bench_catalog().unwrap().conditions.into_iter().map(|c| c.id).collect();
    let imgs = api.bench_images(&bench.id, 0, 1000, None).unwrap().images;
    let img_ids: Vec<String> = imgs.iter().map(|i| i.id.clone()).collect();

    // Name suggestion strips extensions and prediction suffixes.
    assert_eq!(suggest_external_name("yolo26n-predictions.json"), "yolo26n");
    assert_eq!(suggest_external_name("rtdetr_prediction.jsonl"), "rtdetr");
    assert_eq!(suggest_external_name("predictions.json"), "predictions");

    // 1. JSON wrapper format passes validation & detects confidences.
    let recs: Vec<BenchPrediction> = imgs
        .iter()
        .map(|im| BenchPrediction {
            image: im.id.clone(),
            condition: None,
            severity: None,
            boxes: vec![BenchDetection { class: 0, score: 0.82, x1: 10.0, y1: 12.0, x2: 40.0, y2: 52.0 }],
        })
        .collect();
    let json_text = serde_json::to_string(&BenchPredictions { predictions: recs.clone() }).unwrap();
    let parsed = parse_predictions_text(&json_text).unwrap();
    assert!(parsed.has_scores);
    let checked = check_predictions(parsed, &bench, Some(&img_ids), &cat, true).unwrap();
    assert_eq!(checked.summary.images, imgs.len());
    assert_eq!(checked.summary.boxes, imgs.len());
    assert!(checked.summary.conditions.is_empty());

    // 2. JSONL format also parses and validates cleanly.
    let jsonl_text = recs.iter().map(|r| serde_json::to_string(r).unwrap()).collect::<Vec<_>>().join("\n");
    let parsed_jl = parse_predictions_text(&jsonl_text).unwrap();
    let checked_jl = check_predictions(parsed_jl, &bench, Some(&img_ids), &cat, true).unwrap();
    assert_eq!(checked_jl.summary.images, imgs.len());

    // 3. Invalid corners, out-of-range class, missing image all produce clear record errors.
    let bad_corners = format!(
        r#"[{{"image":"{}","boxes":[{{"class":0,"score":0.9,"x1":50,"y1":10,"x2":10,"y2":40}}]}}]"#,
        img_ids[0]
    );
    let err =
        check_predictions(parse_predictions_text(&bad_corners).unwrap(), &bench, Some(&img_ids), &cat, true)
            .unwrap_err();
    assert!(err.contains("Record 1, box 1") && err.contains("corners"), "{err}");

    let bad_class = format!(
        r#"[{{"image":"{}","boxes":[{{"class":999,"score":0.9,"x1":10,"y1":10,"x2":20,"y2":20}}]}}]"#,
        img_ids[0]
    );
    let err =
        check_predictions(parse_predictions_text(&bad_class).unwrap(), &bench, Some(&img_ids), &cat, true)
            .unwrap_err();
    assert!(err.contains("class 999"), "{err}");

    let missing_img = format!(r#"[{{"image":"{}","boxes":[]}}]"#, img_ids[0]);
    let err =
        check_predictions(parse_predictions_text(&missing_img).unwrap(), &bench, Some(&img_ids), &cat, true)
            .unwrap_err();
    assert!(err.contains("unchanged record"), "{err}");

    // 4. Multi-part upload (forced via tiny max_bytes) creates open entry, sends parts, finishes & scores!
    let parts = parts_of(&checked.predictions, &bench.plan, 256);
    assert!(parts.len() > 1, "tiny max_bytes splits into multiple parts");
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let done = upload_external_predictions(
        &api,
        &bench.id,
        "yolo26n-ext".into(),
        true,
        &bench.plan,
        &checked.predictions,
        256,
        &cancel,
    )
    .unwrap();
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert_eq!(done.contestant, BenchContestant::External);
    assert!(done.score.is_some());
    assert_eq!(api.bench_entry(&bench.id, &done.id).unwrap().name, "yolo26n-ext");

    // 5. Cooperative cancel aborts before creating/publishing any entry.
    let cancelled = std::sync::atomic::AtomicBool::new(true);
    let err = upload_external_predictions(
        &api,
        &bench.id,
        "should-not-exist".into(),
        true,
        &bench.plan,
        &checked.predictions,
        256,
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(err.code, "cancelled");
    assert!(api.bench_entries(&bench.id).unwrap().iter().all(|e| e.name != "should-not-exist"));
}

#[test]
fn external_entry_inline_form_validates_file_and_submits_into_leaderboard() {
    let dir = TestDir::new("folo-desktop-ext");
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_7a04".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    // Build a valid predictions file for bn_7a04 on disk inside TestDir.
    let file_path = dir.path().join("yolo26s-predictions.json");
    let imgs = drive(&mut app, window, |this, _, _| {
        this.store.api.bench_images("bn_7a04", 0, 1000, None).unwrap().images
    });
    let recs: Vec<BenchPrediction> = imgs
        .iter()
        .map(|im| BenchPrediction {
            image: im.id.clone(),
            condition: None,
            severity: None,
            boxes: vec![BenchDetection { class: 0, score: 0.77, x1: 8.0, y1: 8.0, x2: 32.0, y2: 32.0 }],
        })
        .collect();
    std::fs::write(&file_path, serde_json::to_string(&BenchPredictions { predictions: recs }).unwrap())
        .unwrap();

    // Open inline form, set path, validate, verify suggested name, submit, and confirm entry appears.
    drive(&mut app, window, |this, w, cx| {
        this.open_external_entry("bn_7a04", w, cx);
        assert_eq!(this.shown_bench_form(), Some(BenchForm::External));
        let path_str = file_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&path_str, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, _, cx| {
        let f = this.bench.external.as_ref().expect("form open");
        assert!(f.check.as_ref().is_some_and(Result::is_ok), "file validated: {:?}", f.check);
        assert_eq!(f.name.read(cx).text(), "yolo26s", "auto-suggested name");
        this.submit_external_entry(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    drive(&mut app, window, |this, _, _| {
        assert!(this.bench.external.is_none(), "form closes after successful score");
        let list = this.bench.entries.get("bn_7a04").and_then(|l| l.ready()).unwrap();
        assert!(
            list.iter().any(|e| e.name == "yolo26s" && e.contestant == BenchContestant::External),
            "scored external entry appears on leaderboard"
        );
    });
}

/// Regression tests for all 6 Playtest Round 9 findings:
/// 1. Changing file_path after validation blocks scoring the stale file.
/// 2. Explicit No confidences stays off on files with out-of-range scores, and toggling Uses confidences catches them.
/// 3. Custom leaderboard name is preserved when rechecking another file.
/// 4. Partial /result failure keeps rendering details for loaded entries alongside a Retry button.
#[test]
fn playtest_r9_external_guards_and_compare_partial_failure_regressions() {
    let dir = TestDir::new("folo-desktop-r9");
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_7a04".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    let imgs = drive(&mut app, window, |this, _, _| {
        this.store.api.bench_images("bn_7a04", 0, 1000, None).unwrap().images
    });
    let valid_path = dir.path().join("first-predictions.json");
    let second_path = dir.path().join("second-predictions.json");
    let bad_path = dir.path().join("malformed.jsonl");
    let bad_score_path = dir.path().join("over-one-score.json");

    let make_file = |score: f32| {
        let recs: Vec<BenchPrediction> = imgs
            .iter()
            .map(|im| BenchPrediction {
                image: im.id.clone(),
                condition: None,
                severity: None,
                boxes: vec![BenchDetection { class: 0, score, x1: 8.0, y1: 8.0, x2: 32.0, y2: 32.0 }],
            })
            .collect();
        serde_json::to_string(&BenchPredictions { predictions: recs }).unwrap()
    };
    std::fs::write(&valid_path, make_file(0.8)).unwrap();
    std::fs::write(&second_path, make_file(0.75)).unwrap();
    std::fs::write(&bad_path, "{ not valid jsonl").unwrap();
    std::fs::write(&bad_score_path, make_file(2.0)).unwrap();

    // 1. Check valid file, customize name, check second file -> custom name preserved!
    drive(&mut app, window, |this, w, cx| {
        this.open_external_entry("bn_7a04", w, cx);
        let p = valid_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&p, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let f = this.bench.external.as_ref().unwrap();
        assert_eq!(f.name.read(cx).text(), "first");
        f.name.update(cx, |i, cx| i.set_text("my-custom-detector", cx));
        let p2 = second_path.to_string_lossy().to_string();
        f.file_path.update(cx, |i, cx| i.set_text(&p2, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let f = this.bench.external.as_ref().unwrap();
        assert_eq!(f.name.read(cx).text(), "my-custom-detector", "custom name survives recheck");
    });

    // 2. Replace path with malformed file without rechecking -> submit MUST refuse and NOT score previous file!
    drive(&mut app, window, |this, _, cx| {
        let bad = bad_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&bad, cx));
        this.submit_external_entry(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let f = this.bench.external.as_ref().expect("form must stay open on stale path");
        assert!(f.error.as_ref().is_some_and(|e| e.message.contains("Check file")), "{:?}", f.error);
        let list = this.bench.entries.get("bn_7a04").and_then(|l| l.ready()).unwrap();
        assert!(list.is_empty(), "stale file must not be scored");
    });

    // 3. Explicit No confidences stays off when checking score=2.0 file; switching back to Uses confidences rejects it.
    drive(&mut app, window, |this, _, cx| {
        this.set_external_scored(false, cx);
        let p = bad_score_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&p, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let f = this.bench.external.as_ref().unwrap();
        assert!(!f.scored, "explicit No confidences must NOT silently flip back on");
        assert!(f.check.as_ref().is_some_and(Result::is_ok), "unscored check accepts score=2.0");
        // Switching back to Uses confidences triggers re-validation of the file and rejects score=2.0!
        this.set_external_scored(true, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let f = this.bench.external.as_ref().expect("form stays open on score > 1");
        assert!(
            f.check.as_ref().is_some_and(|r| r.as_ref().is_err_and(|m| m.contains("between 0 and 1"))),
            "{:?}",
            f.check
        );
        this.submit_external_entry(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let list = this.bench.entries.get("bn_7a04").and_then(|l| l.ready()).unwrap();
        assert!(list.is_empty(), "invalid scored entry must not be submitted");
    });

    // 4. Compare partial /result failure shows Retry button for the failed entry without hiding loaded entries.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        this.open_bench_compare("bn_3c90", None, w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let cmp = this.bench.compare.get("bn_3c90").cloned().unwrap();
        let fail_id = cmp[2].clone();
        this.bench.results.insert(
            ("bn_3c90".into(), fail_id),
            crate::store::Load::Failed(ApiError::new("not_found", "Result missing")),
        );
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let cmp = this.bench.compare.get("bn_3c90").unwrap();
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(
            keys.iter().any(|k| k == &format!("bench-cmp-res-retry-{}", cmp[2])),
            "failed entry gets Retry button: {keys:?}"
        );
        // Loaded entries' results remain ready and rendered (per-class panel & robustness stay up).
        assert!(
            this.bench.results.get(&("bn_3c90".into(), cmp[0].clone())).and_then(|l| l.ready()).is_some()
        );
    });
}

/// Regression tests for PR #746 Round 1 review findings:
/// - Deleting a compared entry persists reconciliation so Add entry can fill the freed slot.
/// - Switching No confidences -> Uses confidences re-reads and rejects duplicate records with differing scores.
/// - Empty-path Check file surfaces its error cleanly.
#[test]
fn pr746_r1_deleted_compare_slot_and_confidence_recheck_regressions() {
    let dir = TestDir::new("folo-desktop-pr746-r1");
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    // 1. Open Compare (4 entries), delete one entry from loaded entries, re-render, and add a 5th entry.
    drive(&mut app, window, |this, w, cx| {
        this.open_bench_compare("bn_3c90", None, w, cx);
        let cmp = this.bench.compare.get("bn_3c90").cloned().unwrap();
        assert_eq!(cmp.len(), 4);
        let removed_id = cmp[3].clone();
        if let Some(list) = this.bench.entries.get_mut("bn_3c90").and_then(|l| l.ready_mut()) {
            list.retain(|e| e.id != removed_id);
        }
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        let cmp = this.bench.compare.get("bn_3c90").cloned().unwrap();
        assert_eq!(cmp.len(), 3, "deleted entry ID must be pruned from saved compare list");
        this.compare_add_entry("bn_3c90", "be_3c05".into(), cx);
        assert_eq!(this.bench.compare.get("bn_3c90").unwrap().len(), 4, "freed slot accepts a new entry");
    });

    // 2. Duplicate record differing only in score passes under No confidences, then fails when toggling Uses confidences!
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench("bn_7a04".into()), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    let imgs = drive(&mut app, window, |this, _, _| {
        this.store.api.bench_images("bn_7a04", 0, 1000, None).unwrap().images
    });
    let mut recs: Vec<BenchPrediction> = imgs
        .iter()
        .map(|im| BenchPrediction {
            image: im.id.clone(),
            condition: None,
            severity: None,
            boxes: vec![BenchDetection { class: 0, score: 0.8, x1: 8.0, y1: 8.0, x2: 32.0, y2: 32.0 }],
        })
        .collect();
    recs.push(BenchPrediction {
        image: imgs[0].id.clone(),
        condition: None,
        severity: None,
        boxes: vec![BenchDetection { class: 0, score: 0.3, x1: 8.0, y1: 8.0, x2: 32.0, y2: 32.0 }],
    });
    let dup_path = dir.path().join("dup-diff-score.json");
    std::fs::write(&dup_path, serde_json::to_string(&BenchPredictions { predictions: recs }).unwrap())
        .unwrap();

    drive(&mut app, window, |this, w, cx| {
        this.open_external_entry("bn_7a04", w, cx);
        this.check_typed_external_file(cx);
        assert!(this.bench.external.as_ref().unwrap().check.as_ref().is_some_and(Result::is_err));
        this.set_external_scored(false, cx);
        let p = dup_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&p, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        assert!(this.bench.external.as_ref().unwrap().check.as_ref().is_some_and(Result::is_ok));
        this.set_external_scored(true, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let chk = this.bench.external.as_ref().unwrap().check.as_ref().expect("rechecked");
        let err = chk.as_ref().unwrap_err();
        assert!(err.contains("different boxes"), "recheck must catch differing-score duplicate: {err}");
    });

    // 3. Edit path -> toggle Uses confidences -> restore path still blocks submit (`checked_scored == false`).
    let cancel_probe = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancel_probe_ref = cancel_probe.clone();
    drive(&mut app, window, |this, _, cx| {
        this.set_external_scored(false, cx);
        let p = dup_path.to_string_lossy().to_string();
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&p, cx));
        this.check_typed_external_file(cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        let orig = dup_path.to_string_lossy().to_string();
        // Edit path away, enable confidences, then undo path edit back to orig.
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text("other.json", cx));
        this.set_external_scored(true, cx);
        this.bench.external.as_ref().unwrap().file_path.update(cx, |i, cx| i.set_text(&orig, cx));
        this.submit_external_entry(cx);
        let f = this.bench.external.as_mut().expect("form blocks submit");
        assert!(
            f.error.as_ref().is_some_and(|e| e.message.contains("Check file")),
            "must refuse unchecked confidence mode after path undo: {:?}",
            f.error
        );
        // Navigating directly to another benchmark drops the form and signals cancel via Drop.
        f.cancel = Some(cancel_probe_ref);
        this.navigate(Route::Bench("bn_3c90".into()), w, cx);
    });
    assert!(
        cancel_probe.load(std::sync::atomic::Ordering::Relaxed),
        "direct Bench(A) -> Bench(B) navigation cancels in-flight upload via Drop"
    );
}
