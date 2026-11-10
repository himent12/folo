//! External entries: checked and scored, filled in parts, expiry, and finish racing stop.

use std::time::Duration;

use axum::Router;
use axum::http::StatusCode;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    Events, Step, TIMEOUT, axis, bench, dataset, delete, ended, err, external, fixture, fixture_with, get,
    ok, open_entry, page, parts, post, put, send, small_plan, truth_predictions,
};

#[tokio::test(flavor = "multi_thread")]
async fn external_entries_are_checked_and_scored() {
    let t = TestDir::new("bench-external");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let truth = page(app, &b.id, "").await;
    let images = &truth.images;
    let n_boxes: usize = images.iter().map(|i| i.boxes.len()).sum();
    assert!(n_boxes >= 3, "the synthetic images have boxes");

    // The truth itself, under every planned condition: every axis 1, a score of 100, complete.
    let mut perfect = truth_predictions(images, None, 1.0);
    for (c, s) in [("haze", 1), ("haze", 3), ("distance", 2)] {
        let moved = page(app, &b.id, &format!("&condition={c}&severity={s}")).await;
        perfect.extend(truth_predictions(&moved.images, Some((c, s)), 1.0));
    }
    let p: BenchEntry = ok(app, post(&entries, external("perfect", true, perfect.clone()))).await;
    assert_eq!(
        (p.status, p.contestant.clone(), p.conditions, p.scored),
        (BenchEntryStatus::Done, BenchContestant::External, true, true)
    );
    let score = p.score.clone().unwrap();
    assert!(!score.partial, "{score:?}");
    for a in BenchAxis::ALL {
        assert!((axis(&score, a).unwrap() - 1.0).abs() < 1e-6, "{a:?}: {score:?}");
    }
    assert!((score.total - 100.0).abs() < 1e-3 && (p.map50_95.unwrap() - 1.0).abs() < 1e-6);
    assert_eq!((p.ci95, p.conditions_run), (Some([1.0, 1.0]), 2));
    assert_eq!((p.progress.done, p.progress.total, p.images), (16, 16, 4));
    let r: BenchResult = ok(app, get(&format!("{entries}/{}/result", p.id))).await;
    let rob = r.robustness.clone().unwrap();
    assert_eq!(rob.conditions.iter().map(|c| c.condition.as_str()).collect::<Vec<_>>(), ["haze", "distance"]);
    assert_eq!(rob.conditions[0].points.len(), 2);
    assert_eq!((r.self_labeling.readiness, r.localization.mean_iou), (BenchReadiness::Ready, Some(1.0)));

    // Every box shifted right by a tenth of its width: IoU 0.9 / 1.1 for each, so mAP@0.5:0.95 is 0.7 (matched at
    // 0.5 to 0.8), localization (0.818 - 0.5) / 0.5, calibration 1 - |0.9 - 1|; unchanged images only: partial.
    let shifted: Vec<Value> = images
        .iter()
        .map(|im| {
            let boxes: Vec<Value> = im
                .boxes
                .iter()
                .map(|b| {
                    let d = (b.x2 - b.x1) * 0.1;
                    json!({ "class": b.class, "score": 0.9, "x1": b.x1 + d, "y1": b.y1, "x2": b.x2 + d, "y2": b.y2 })
                })
                .collect();
            json!({ "image": im.id, "boxes": boxes })
        })
        .collect();
    let s: BenchEntry = ok(app, post(&entries, external("shifted", true, shifted))).await;
    let sc = s.score.clone().unwrap();
    let iou = 0.9f32 / 1.1;
    let near = |a: Option<f32>, b: f32| (a.unwrap() - b).abs() < 1e-3;
    assert!(sc.partial && axis(&sc, BenchAxis::Robustness).is_none() && !s.conditions);
    assert_eq!(s.conditions_run, 0);
    let ci = s.ci95.unwrap();
    assert!(ci[0] <= s.map50_95.unwrap() && s.map50_95.unwrap() <= ci[1], "{ci:?}");
    assert!(near(axis(&sc, BenchAxis::Accuracy), 0.7), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::Localization), (iou - 0.5) / 0.5), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::Calibration), 0.9), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::SelfLabeling), 1.0), "{sc:?}");
    let want = 100.0 * (0.35 * 0.7 + 0.2 * 1.0 + 0.1 * 0.9 + 0.1 * (iou - 0.5) / 0.5) / 0.75;
    assert!((sc.total - want).abs() < 0.1, "{} vs {want}", sc.total);

    // Without confidences, one box missing: self-labeling is the share of boxes found, calibration unmeasured.
    let mut missing = truth_predictions(images, None, 1.0);
    let with = images.iter().position(|i| !i.boxes.is_empty()).unwrap();
    missing[with]["boxes"].as_array_mut().unwrap().remove(0);
    let m: BenchEntry = ok(app, post(&entries, external("missing", false, missing))).await;
    let ms = m.score.clone().unwrap();
    assert!(!m.scored && ms.partial && axis(&ms, BenchAxis::Calibration).is_none());
    assert!(near(axis(&ms, BenchAxis::SelfLabeling), (n_boxes - 1) as f32 / n_boxes as f32), "{ms:?}");
    assert!(near(axis(&ms, BenchAxis::Localization), 1.0));
    let acc = axis(&ms, BenchAxis::Accuracy).unwrap();
    assert!(acc < 1.0 && acc > 0.5, "{acc}");
    let want = 100.0 * (0.35 * acc + 0.2 * (n_boxes - 1) as f32 / n_boxes as f32 + 0.1) / 0.65;
    assert!((ms.total - want).abs() < 1e-3, "{} vs {want}", ms.total);
    // Its gallery: the image with the missing box comes first by `worst`, with one missed.
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images?sort=worst", m.id))).await;
    assert_eq!(
        (gallery.total, gallery.images[0].image.id.as_str(), gallery.images[0].missed),
        (4, images[with].id.as_str(), 1)
    );
    assert!(gallery.images[1..].iter().all(|i| i.missed == 0 && i.wrong == 0));
    let ordered: BenchEntryImagePage =
        ok(app, get(&format!("{entries}/{}/images?limit=2&offset=1", p.id))).await;
    assert_eq!(ordered.images.iter().map(|i| i.image.id.as_str()).collect::<Vec<_>>(), ["2", "3"]);
    err(app, get(&format!("{entries}/{}/images?sort=best", m.id)), StatusCode::BAD_REQUEST, "invalid_query")
        .await;

    // The leaderboard: by score; the benchmark's best is the perfect entry.
    let board: BenchEntryList = ok(app, get(&entries)).await;
    assert_eq!(
        board.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        ["perfect", "missing", "shifted"]
    );
    let bb: Bench = ok(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await;
    let best = bb.best.unwrap();
    assert_eq!((bb.entries, best.entry_id, best.partial, bb.active), (3, p.id.clone(), false, false));
    // A plan is frozen once there are entries; the name is not.
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    err(app, put(&uri, json!({ "plan": { "conditions": [] } })), StatusCode::CONFLICT, "bench_frozen").await;
    ok::<Bench>(app, put(&uri, json!({ "name": "still mine" }))).await;

    // A wrong submission is named (every way a record can be wrong: the unit tests of `bench::external`), and so
    // is a wrong request.
    let clean = truth_predictions(images, None, 1.0);
    let mut extra = clean.clone();
    extra.push(json!({ "image": "99", "boxes": [] }));
    let cases: Vec<(Value, &str)> = vec![
        (external("x", true, extra), "the benchmark has no image '99'"),
        (json!({ "contestant": { "kind": "external", "predictions": clean.clone() } }), "name is required"),
        (
            json!({ "name": "x", "max_images": 2, "contestant": { "kind": "external", "predictions": clean.clone() } }),
            "not for an external contestant",
        ),
    ];
    for (body, want) in cases {
        let e = err(app, post(&entries, body), StatusCode::BAD_REQUEST, "invalid_body").await;
        assert!(e.message.contains(want), "{} / {want}", e.message);
    }
    err(app, get(&format!("{entries}/be_none")), StatusCode::NOT_FOUND, "bench_entry_not_found").await;
    err(
        app,
        post(&format!("{entries}/{}/stop", p.id), json!({})),
        StatusCode::CONFLICT,
        "bench_entry_not_active",
    )
    .await;

    // Up to 64 MiB: a submission over the 2 MiB of other requests is taken.
    let box_ = json!({ "class": 0, "score": 0.0123456789, "x1": 1.0000001, "y1": 1.0000001, "x2": 3.0000001, "y2": 9.0000001 });
    let mut big = clean.clone();
    for im in &mut big {
        im["boxes"] = json!(vec![box_.clone(); 1000]);
    }
    let mut body = external("big", true, big);
    body["comment"] = json!("x".repeat(2 << 20));
    assert!(body.to_string().len() > 2 << 20);
    let e: BenchEntry = ok(app, post(&entries, body)).await;
    assert_eq!(e.status, BenchEntryStatus::Done);
}

/// Every record of a benchmark with [`small_plan`] whose boxes are the truth moved by `shift` px and scored
/// `score`, plus a wrong box per image (so precision, calibration and the curves have something to say).
async fn records(app: &Router, bench: &str, shift: f32) -> Vec<Value> {
    let mut out = Vec::new();
    for v in [None, Some(("haze", 1)), Some(("haze", 3)), Some(("distance", 2))] {
        let q = v.map_or(String::new(), |(c, s)| format!("&condition={c}&severity={s}"));
        for (k, im) in page(app, bench, &q).await.images.iter().enumerate() {
            let mut boxes: Vec<Value> = im
                .boxes
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let score = 0.3 + 0.07 * ((k + j) % 9) as f32;
                    json!({ "class": b.class, "score": score, "x1": b.x1 + shift, "y1": b.y1, "x2": b.x2 + shift, "y2": b.y2 })
                })
                .collect();
            boxes.push(json!({ "class": 0, "score": 0.2, "x1": 1, "y1": 1, "x2": 9, "y2": 9 }));
            let mut r = json!({ "image": im.id, "boxes": boxes });
            if let Some((c, s)) = v {
                r["condition"] = json!(c);
                r["severity"] = json!(s);
            }
            out.push(r);
        }
    }
    out
}

/// A result as JSON without what differs between two entries of the same predictions (ids, names and times).
fn comparable(mut r: Value) -> Value {
    for k in ["id", "name", "created", "finished"] {
        r["entry"].as_object_mut().unwrap().remove(k);
    }
    r
}

#[tokio::test(flavor = "multi_thread")]
async fn open_external_entries_are_filled_in_parts() {
    let t = TestDir::new("bench-open");
    let home = t.join("home");
    let f = fixture(&home);
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let all = records(app, &b.id, 2.0).await;
    assert_eq!(all.len(), 16);

    // The same predictions in one request, and in parts in another order (variants first, so each is graded as
    // its last image arrives, the unchanged images last): the same entry, bit for bit.
    let whole: BenchEntry = ok(app, post(&entries, external("whole", true, all.clone()))).await;
    let mut events = Events::open(app, &b.id).await;
    assert_eq!(events.next().await.0, "bench");
    let mut order = all.clone();
    order.reverse();
    assert!(order[12..].iter().all(|r| r.get("condition").is_none()), "the unchanged images last");
    let o: BenchEntry = ok(app, post(&entries, open_entry("parts", order[..3].to_vec()))).await;
    assert_eq!(
        (o.status, o.progress.done, o.progress.total, o.score.clone()),
        (BenchEntryStatus::Running, 3, 16, None)
    );
    // Open counts as active, and the benchmark says so.
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["status"].as_str()), ("entry", Some("running")));
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["active"].as_bool()), ("bench", Some(true)));
    let uri = format!("{entries}/{}", o.id);
    for chunk in order[3..].chunks(5) {
        let e: BenchEntry = ok(app, post(&format!("{uri}/predictions"), parts(chunk))).await;
        assert_eq!(e.status, BenchEntryStatus::Running);
    }
    let done: BenchEntry = ok(app, post(&format!("{uri}/finish"), json!({}))).await;
    assert_eq!((done.status, done.progress.done, done.progress.total), (BenchEntryStatus::Done, 16, 16));
    let a: Value = ok(app, get(&format!("{entries}/{}/result", whole.id))).await;
    let c: Value = ok(app, get(&format!("{uri}/result"))).await;
    assert_eq!(comparable(a), comparable(c), "parts score exactly as one request does");
    let ga: Value = ok(app, get(&format!("{entries}/{}/images?sort=worst", whole.id))).await;
    let gc: Value = ok(app, get(&format!("{uri}/images?sort=worst"))).await;
    assert_eq!(ga, gc);
    let mut seen = Vec::new();
    loop {
        let (name, data) = events.next().await;
        seen.push((name.clone(), data["status"].as_str().map(str::to_string), data["active"].as_bool()));
        if name == "bench" {
            break;
        }
    }
    assert_eq!(seen.last().unwrap(), &("bench".to_string(), None, Some(false)), "{seen:?}");
    assert!(seen.iter().any(|s| s.1.as_deref() == Some("done")), "{seen:?}");

    // The truth through the open path scores 100.
    let mut perfect = truth_predictions(&page(app, &b.id, "").await.images, None, 1.0);
    for (c, s) in [("haze", 1), ("haze", 3), ("distance", 2)] {
        let moved = page(app, &b.id, &format!("&condition={c}&severity={s}")).await;
        perfect.extend(truth_predictions(&moved.images, Some((c, s)), 1.0));
    }
    let p: BenchEntry = ok(app, post(&entries, open_entry("perfect", vec![]))).await;
    let puri = format!("{entries}/{}", p.id);
    assert_eq!((p.progress.done, p.progress.total), (0, 16));
    for pass in perfect.chunks(4).rev() {
        ok::<BenchEntry>(app, post(&format!("{puri}/predictions"), parts(pass))).await;
    }
    let p: BenchEntry = ok(app, post(&format!("{puri}/finish"), json!({}))).await;
    let score = p.score.unwrap();
    assert!((score.total - 100.0).abs() < 1e-3 && !score.partial, "{score:?}");
    assert_eq!((p.conditions, p.conditions_run), (true, 2));

    // A part sent again as it was (a retry after a lost answer) is taken as it is; a record received before with
    // other boxes is named, across parts too; refused parts take nothing; an unfinished entry stays open.
    let e: BenchEntry = ok(app, post(&entries, open_entry("fixable", all[..2].to_vec()))).await;
    let euri = format!("{entries}/{}", e.id);
    let retried: BenchEntry = ok(app, post(&format!("{euri}/predictions"), parts(&all[..2]))).await;
    assert_eq!(retried.progress.done, 2, "a retry adds nothing");
    let mut changed = all[1].clone();
    changed["boxes"] = json!([]);
    let again = err(
        app,
        post(&format!("{euri}/predictions"), parts(&[all[2].clone(), changed])),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(
        again
            .message
            .contains("prediction 2 (image '2', unchanged) was received in an earlier part with other boxes"),
        "{}",
        again.message
    );
    let mut bad = all[2..4].to_vec();
    bad[1]["boxes"][0]["class"] = json!(7);
    let e2 =
        err(app, post(&format!("{euri}/predictions"), parts(&bad)), StatusCode::BAD_REQUEST, "invalid_body")
            .await;
    assert!(e2.message.contains("prediction 2 (image '4', unchanged), box 1: class 7"), "{}", e2.message);
    let now: BenchEntry = ok(app, get(&euri)).await;
    assert_eq!((now.status, now.progress.done), (BenchEntryStatus::Running, 2), "refused parts took nothing");
    let missing =
        err(app, post(&format!("{euri}/finish"), json!({})), StatusCode::BAD_REQUEST, "invalid_body").await;
    assert!(missing.message.contains("no prediction for image '3' (unchanged)"), "{}", missing.message);
    // Half a condition is the "answered in part" error, also at finish.
    ok::<BenchEntry>(app, post(&format!("{euri}/predictions"), parts(&all[2..5]))).await;
    let half =
        err(app, post(&format!("{euri}/finish"), json!({})), StatusCode::BAD_REQUEST, "invalid_body").await;
    assert!(
        half.message.contains("condition 'haze' is answered, but not for image '2' at severity 1"),
        "{}",
        half.message
    );
    assert_eq!(ok::<BenchEntry>(app, get(&euri)).await.status, BenchEntryStatus::Running);
    ok::<BenchEntry>(app, post(&format!("{euri}/predictions"), parts(&all[5..]))).await;
    let fixed: BenchEntry = ok(app, post(&format!("{euri}/finish"), json!({}))).await;
    assert_eq!(fixed.status, BenchEntryStatus::Done);
    assert_eq!(fixed.score, done.score, "the same predictions, the same score");

    // Not open: a finished external entry, a stopped one, another kind of contestant.
    for target in [&euri, &format!("{entries}/{}", whole.id)] {
        err(
            app,
            post(&format!("{target}/predictions"), parts(&all[..1])),
            StatusCode::CONFLICT,
            "bench_entry_not_open",
        )
        .await;
        err(app, post(&format!("{target}/finish"), json!({})), StatusCode::CONFLICT, "bench_entry_not_open")
            .await;
    }
    f.agent.script(vec![Step::Hold, Step::Hold]);
    let agent: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" } }))).await;
    let auri = format!("{entries}/{}", agent.id);
    let e3 = err(
        app,
        post(&format!("{auri}/predictions"), parts(&all[..1])),
        StatusCode::CONFLICT,
        "bench_entry_not_open",
    )
    .await;
    assert!(e3.message.contains("an agent entry"), "{}", e3.message);
    err(app, post(&format!("{auri}/finish"), json!({})), StatusCode::CONFLICT, "bench_entry_not_open").await;
    ok::<BenchEntry>(app, post(&format!("{auri}/stop"), json!({}))).await;
    assert_eq!(ended(app, &b.id, &agent.id).await.status, BenchEntryStatus::Stopped);
    err(
        app,
        post(&format!("{entries}/be_none/finish"), json!({})),
        StatusCode::NOT_FOUND,
        "bench_entry_not_found",
    )
    .await;

    // Stop discards an open entry; it is active until then (no delete, no deleting the benchmark).
    let s: BenchEntry = ok(app, post(&entries, open_entry("stopped", all[..4].to_vec()))).await;
    let suri = format!("{entries}/{}", s.id);
    assert!(ok::<Bench>(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await.active);
    err(app, delete(&suri), StatusCode::CONFLICT, "bench_entry_active").await;
    err(app, delete(&format!("/api/bench/benchmarks/{}", b.id)), StatusCode::CONFLICT, "bench_active").await;
    let stopped: BenchEntry = ok(app, post(&format!("{suri}/stop"), json!({}))).await;
    assert_eq!((stopped.status, stopped.score.clone()), (BenchEntryStatus::Stopped, None));
    err(
        app,
        post(&format!("{suri}/predictions"), parts(&all[4..5])),
        StatusCode::CONFLICT,
        "bench_entry_not_open",
    )
    .await;
    err(app, post(&format!("{suri}/stop"), json!({})), StatusCode::CONFLICT, "bench_entry_not_active").await;
    assert!(!ok::<Bench>(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await.active);
    let (status, _, _) = send(app, delete(&suri)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    // A result that cannot be written leaves the entry failed, never done without a result on disk (the entry's
    // file is a folder here, so writing it fails).
    let w: BenchEntry = ok(app, post(&entries, open_entry("unwritable", all.clone()))).await;
    let file = home.join("bench").join(&b.id).join("entries").join(format!("{}.json", w.id));
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir(&file).unwrap();
    let wf: BenchEntry = ok(app, post(&format!("{entries}/{}/finish", w.id), json!({}))).await;
    assert_eq!((wf.status, wf.score.clone()), (BenchEntryStatus::Failed, None), "{wf:?}");
    assert!(wf.error.as_deref().unwrap().starts_with("the entry's result could not be written"), "{wf:?}");
    err(app, get(&format!("{entries}/{}/result", w.id)), StatusCode::CONFLICT, "bench_entry_not_done").await;
    std::fs::remove_dir(&file).unwrap();

    // One left open when the server stops is failed at the next start (its predictions were in memory).
    let left: BenchEntry = ok(app, post(&entries, open_entry("left", all[..4].to_vec()))).await;
    drop(events);
    drop(f);
    let f = fixture(&home);
    let l: BenchEntry = ok(&f.app, get(&format!("{entries}/{}", left.id))).await;
    assert_eq!(
        (l.status, l.error.as_deref()),
        (BenchEntryStatus::Failed, Some("the server stopped before the entry finished"))
    );
    err(
        &f.app,
        post(&format!("{entries}/{}/finish", left.id), json!({})),
        StatusCode::CONFLICT,
        "bench_entry_not_open",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn open_entries_expire_without_parts() {
    let t = TestDir::new("bench-expire");
    let f = fixture_with(&t.join("home"), |s| s.bench_open_timeout(Duration::from_millis(300)));
    let app = &f.app;
    let ds = dataset(app, "shapes", 10).await;
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": { "conditions": [] } }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let all = truth_predictions(&page(app, &b.id, "").await.images, None, 0.9);
    let mut events = Events::open(app, &b.id).await;
    assert_eq!(events.next().await.0, "bench");
    // Parts keep an entry open: every image's, then one every 50 ms until the other entry, which gets none, has
    // expired. (A part sent again as it was counts as a part: a client that retries is alive.)
    let kept: BenchEntry = ok(app, post(&entries, open_entry("kept", vec![]))).await;
    let left: BenchEntry = ok(app, post(&entries, open_entry("left", vec![]))).await;
    ok::<BenchEntry>(app, post(&format!("{entries}/{}/predictions", kept.id), parts(&all))).await;
    let t0 = std::time::Instant::now();
    let mut i = 0;
    let l = loop {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let r = &all[i % all.len()];
        i += 1;
        ok::<BenchEntry>(
            app,
            post(&format!("{entries}/{}/predictions", kept.id), parts(std::slice::from_ref(r))),
        )
        .await;
        let l: BenchEntry = ok(app, get(&format!("{entries}/{}", left.id))).await;
        if !l.status.is_active() {
            break l;
        }
        assert!(t0.elapsed() < TIMEOUT, "{l:?}");
    };
    // ... failed, with what it held dropped, and the benchmark idle again.
    assert_eq!(
        (l.status, l.error.as_deref()),
        (BenchEntryStatus::Failed, Some("no predictions arrived for 300 ms"))
    );
    assert!(l.hint.is_some() && l.finished.is_some());
    err(
        app,
        post(&format!("{entries}/{}/finish", left.id), json!({})),
        StatusCode::CONFLICT,
        "bench_entry_not_open",
    )
    .await;
    let done: BenchEntry = ok(app, post(&format!("{entries}/{}/finish", kept.id), json!({}))).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(!ok::<Bench>(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await.active);
    // Its events: the `bench` that goes inactive comes once both have ended.
    let mut actives = Vec::new();
    while actives.last() != Some(&false) {
        let (name, data) = events.next().await;
        if name == "bench" {
            actives.push(data["active"].as_bool().unwrap());
        }
    }
    assert!(actives.contains(&true), "{actives:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn finish_and_stop_race_without_a_hang_and_one_wins() {
    let t = TestDir::new("bench-race");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 10).await;
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": { "conditions": [] } }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let all = truth_predictions(&page(app, &b.id, "").await.images, None, 0.9);
    for i in 0..40 {
        let e: BenchEntry = ok(app, post(&entries, open_entry(&format!("race {i}"), all.clone()))).await;
        let uri = format!("{entries}/{}", e.id);
        let race = async {
            tokio::join!(
                send(app, post(&format!("{uri}/finish"), json!({}))),
                send(app, post(&format!("{uri}/stop"), json!({})))
            )
        };
        let ((fs, _, fb), (ss, _, sb)) =
            tokio::time::timeout(Duration::from_secs(20), race).await.expect("no hang");
        let end: BenchEntry = ok(app, get(&uri)).await;
        match end.status {
            // Finish won: the stop came after it and found nothing to stop.
            BenchEntryStatus::Done => {
                assert_eq!(fs, StatusCode::OK, "{}", String::from_utf8_lossy(&fb));
                assert_eq!(ss, StatusCode::CONFLICT, "{}", String::from_utf8_lossy(&sb));
            }
            // Stop won: the finish was refused, nothing was scored.
            BenchEntryStatus::Stopped => {
                assert_eq!(ss, StatusCode::OK, "{}", String::from_utf8_lossy(&sb));
                assert_eq!(fs, StatusCode::CONFLICT, "{}", String::from_utf8_lossy(&fb));
                assert!(end.score.is_none());
            }
            other => panic!("ended {other:?}"),
        }
    }
}
