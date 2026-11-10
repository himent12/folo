//! Model entries: run on the queue, and agree with validation.

use std::time::Duration;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{
    Events, TIMEOUT, axis, bench, dataset, ended, err, fixture, get, model, model_with, ok, page, post,
    small_plan,
};

#[tokio::test(flavor = "multi_thread")]
async fn model_entries_run_on_the_queue_and_agree_with_validation() {
    let t = TestDir::new("bench-model");
    let home = t.join("home");
    let f = fixture(&home);
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let src = json!({ "kind": "dataset", "dataset_id": ds.id });
    // The images in the split's order: an untrained model scores many anchors alike, and AP breaks ties in score
    // by the order the images come in (validation's is the split's).
    let val: ImagePage = ok(app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    let mut seed = 0;
    let b = loop {
        let b = bench(app, json!({ "name": "b", "source": src, "plan": small_plan(), "seed": seed })).await;
        let order: Vec<String> = page(app, &b.id, "").await.images.into_iter().map(|i| i.file).collect();
        if val.images.iter().zip(&order).all(|(v, o)| v.file.ends_with(o.as_str())) {
            break b;
        }
        seed += 1;
        assert!(seed < 500, "no seed keeps the split's order");
    };
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    // A model that finds something, so the comparison with validation means something.
    let m = model_with(app, &home, "shapes-n", &["circle", "square"], 1, true).await;
    let n = b.images;

    let mut events = Events::open(app, &b.id).await;
    assert_eq!(events.next().await.0, "bench");
    let e: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    assert!(e.id.starts_with("be_") && e.status == BenchEntryStatus::Queued, "{e:?}");
    assert_eq!(
        (e.name.as_str(), e.images, e.conditions, e.scored, e.progress.total),
        ("shapes-n", n, true, true, 4 * n)
    );
    assert!(
        matches!(&e.contestant, BenchContestant::Model { model_id, arch, imgsz: 64, .. } if *model_id == m && arch == "folo-n")
    );
    assert!(e.missing_classes.is_empty() && e.fingerprint == b.fingerprint);
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    let score = done.score.clone().unwrap();
    // Every image and condition ran: partial only when no detection reached the calibration floor.
    assert_eq!(score.partial, axis(&score, BenchAxis::Calibration).is_none(), "{score:?}");
    let speed = done.speed.clone().unwrap();
    assert!(speed.device == "cpu" && speed.ms_per_image > 0.0, "{speed:?}");
    assert_eq!((done.progress.done, done.progress.total), (4 * n, 4 * n));
    // Its events: queued, running, progress under a condition, done.
    let (mut seen, mut active) = (Vec::new(), Vec::new());
    loop {
        let (name, data) = events.next().await;
        if name == "bench" {
            active.push(data["active"].as_bool().unwrap());
        }
        if name == "entry" {
            let status = data["status"].as_str().unwrap().to_string();
            seen.push((status.clone(), data["progress"]["condition"].as_str().map(str::to_string)));
            if status == "done" {
                break;
            }
        }
    }
    assert_eq!(seen.first().unwrap().0, "queued");
    // The benchmark is sent as it becomes active, and again as it is idle once more.
    assert_eq!(active, [true]);
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["active"].as_bool(), data["entries"].as_u64()),
        ("bench", Some(false), Some(1))
    );
    assert_eq!((done.conditions_run, done.ci95.is_some()), (2, true));
    assert!(seen.iter().any(|(s, _)| s == "running"), "{seen:?}");

    // The result's shape, and per-image outcomes that agree with it.
    let r: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(r.entry, done);
    assert_eq!(r.robustness.as_ref().unwrap().conditions.len(), 2);
    assert!(r.calibration.is_some() || r.accuracy.map50_95 == 0.0);
    assert_eq!(r.self_labeling.curve.len(), 101);
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images", e.id))).await;
    assert_eq!(gallery.total, n);
    for im in &gallery.images {
        assert!(im.detections.iter().all(|d| d.score >= r.accuracy.best_conf && d.class < 2));
        assert!(im.missed as usize <= im.image.boxes.len());
        let right = im.detections.len() as u32 - im.wrong;
        assert_eq!(right + im.missed, im.image.boxes.len() as u32, "{im:?}");
    }

    // The unchanged images' mAP is what `folo val` gives on the same images.
    let v: ValReport =
        ok(app, post("/api/val", json!({ "model_id": m, "dataset_id": ds.id, "split": "val" }))).await;
    assert_eq!(v.images, n);
    assert!(v.val.map50 > 0.0, "the model finds something: {:?}", v.val);
    assert!(
        (r.accuracy.map50_95 - v.val.map50_95).abs() < 1e-5,
        "{} vs {}",
        r.accuracy.map50_95,
        v.val.map50_95
    );
    assert!((r.accuracy.map50 - v.val.map50).abs() < 1e-5, "{} vs {}", r.accuracy.map50, v.val.map50);

    // Classes match by name without case; a model that knows none is refused.
    let partly = model(app, &home, "squares", &["Square", "triangle"], 2).await;
    let p: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": partly }, "conditions": false, "max_images": 2 }))).await;
    assert_eq!(
        (p.missing_classes.clone(), p.images, p.conditions, p.progress.total),
        (vec!["circle".to_string()], 2, false, 2)
    );
    let pets = model(app, &home, "pets", &["dog", "cat"], 3).await;
    let e2 = err(
        app,
        post(&entries, json!({ "contestant": { "kind": "model", "model_id": pets } })),
        StatusCode::BAD_REQUEST,
        "bench_class_mismatch",
    )
    .await;
    assert!(e2.message.contains("dog"), "{}", e2.message);
    err(
        app,
        post(&entries, json!({ "contestant": { "kind": "model", "model_id": "m_none" } })),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;
    let pd = ended(app, &b.id, &p.id).await;
    assert!(pd.score.as_ref().unwrap().partial, "a part of the images and no conditions: partial");
    let r2: BenchResult = ok(app, get(&format!("{entries}/{}/result", p.id))).await;
    assert!(r2.robustness.is_none());
    assert!(r2.accuracy.per_class.iter().all(|c| c.class < 2));

    // One at a time: a second entry waits behind the first; stopping a waiting entry stops it at once.
    let a: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    let w: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    let stopped: BenchEntry = ok(app, post(&format!("{entries}/{}/stop", w.id), json!({}))).await;
    assert!(matches!(stopped.status, BenchEntryStatus::Stopped | BenchEntryStatus::Stopping), "{stopped:?}");
    assert_eq!(ended(app, &b.id, &w.id).await.status, BenchEntryStatus::Stopped);
    err(app, get(&format!("{entries}/{}/result", w.id)), StatusCode::CONFLICT, "bench_entry_not_done").await;
    assert_eq!(ended(app, &b.id, &a.id).await.status, BenchEntryStatus::Done);

    // A running entry that is stopped ends `stopped`, never `done`: also when the stop lands after its last pass,
    // while it is graded (the worker is held there).
    f.store.hold_bench_entries(true);
    let r: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    let t0 = std::time::Instant::now();
    loop {
        let now: BenchEntry = ok(app, get(&format!("{entries}/{}", r.id))).await;
        if now.status == BenchEntryStatus::Running && now.progress.done == now.progress.total {
            break;
        }
        assert!(t0.elapsed() < TIMEOUT, "{now:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let s: BenchEntry = ok(app, post(&format!("{entries}/{}/stop", r.id), json!({}))).await;
    assert_eq!(s.status, BenchEntryStatus::Stopping);
    f.store.hold_bench_entries(false);
    let s = ended(app, &b.id, &r.id).await;
    assert_eq!((s.status, s.score.clone(), s.map50_95), (BenchEntryStatus::Stopped, None, None), "{s:?}");
}
