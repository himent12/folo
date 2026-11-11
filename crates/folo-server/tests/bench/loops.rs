//! Self-improvement loops.

use std::path::Path;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    TIMEOUT, bench, bytes, dataset, delete, ended, err, fixture, get, loop_ended, model, model_with, ok,
    post, put, send,
};

/// A labeling job whose images are `copies` (bytes) plus the images of `ids` of dataset `ds`, every one labeled
/// by hand with the same wrong box.
async fn pool_job(app: &Router, ds: &Dataset, ids: &[String], copies: &[Vec<u8>]) -> LabelJob {
    let job: LabelJob =
        ok(app, post("/api/label/jobs", json!({ "name": "pool", "classes": ["circle", "square"] }))).await;
    let boundary = "XfoloPool";
    let mut body = Vec::new();
    let mut files: Vec<Vec<u8>> = copies.to_vec();
    for id in ids {
        files.push(bytes(app, &format!("/api/datasets/{}/images/{id}", ds.id)).await.1);
    }
    for (i, b) in files.iter().enumerate() {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"p{i}.jpg\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let req = Request::post(format!("/api/label/jobs/{}/images", job.id))
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let added: LabelImagesAdded = ok(app, req).await;
    // A label no teacher would give: if it reached the student's dataset, the pool's own labels were read.
    for im in &added.added {
        let wrong = json!({ "boxes": [{ "class": 1, "x1": 1, "y1": 1, "x2": 7, "y2": 7 }] });
        ok::<LabelImage>(app, put(&format!("/api/label/jobs/{}/images/{}", job.id, im.id), wrong)).await;
    }
    added.job
}

#[tokio::test(flavor = "multi_thread")]
async fn a_loop_teaches_a_student_on_the_teachers_labels_only() {
    let t = TestDir::new("bench-loop");
    let home = t.join("home");
    let f = fixture(&home);
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": plan }),
    )
    .await;
    let teacher = model(app, &home, "teacher", &["circle", "square"], 7).await;
    let loops = format!("/api/bench/benchmarks/{}/loops", b.id);
    let train: ImagePage =
        ok(app, get(&format!("/api/datasets/{}/images?split=train&limit=12", ds.id))).await;
    let train_ids: Vec<String> = train.images.iter().map(|i| i.id.clone()).collect();
    let copies = vec![
        bytes(app, &format!("/api/bench/benchmarks/{}/images/1", b.id)).await.1,
        bytes(app, &format!("/api/bench/benchmarks/{}/images/2", b.id)).await.1,
    ];
    let pool = pool_job(app, &ds, &train_ids, &copies).await;
    let student = json!({ "epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu" });

    // Checked up front: the teacher's classes, the pool, the numbers.
    let wrong_order = model(app, &home, "reversed", &["square", "circle"], 8).await;
    let pool_src = json!({ "kind": "label_job", "job_id": pool.id });
    let e = err(
        app,
        post(&loops, json!({ "teacher": wrong_order, "pool": pool_src })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e.message.contains("same names, same order"), "{}", e.message);
    let same_images = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "val" });
    let e = err(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": same_images, "student": student })),
        StatusCode::BAD_REQUEST,
        "invalid_dataset",
    )
    .await;
    assert!(e.message.contains("every image of the pool"), "{}", e.message);
    err(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "generations": 6 })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(
        app,
        post(&loops, json!({ "teacher": "m_none", "pool": pool_src })),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;
    // Seeds a browser can hold; the students' runs checked as POST /api/runs checks them, before anything runs.
    let big =
        json!({ "teacher": teacher, "pool": pool_src, "student": { "seed": 9_007_199_254_740_992u64 } });
    let e = err(app, post(&loops, big), StatusCode::BAD_REQUEST, "invalid_body").await;
    assert!(e.message.contains("student.seed must be at most 9007199254740991"), "{}", e.message);
    for (student, status, code) in [
        (json!({ "weights": "m_none", "device": "cpu" }), StatusCode::NOT_FOUND, "model_not_found"),
        (json!({ "batch": 0, "device": "cpu" }), StatusCode::BAD_REQUEST, "invalid_config"),
    ] {
        let body =
            json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 0.001, "student": student });
        err(app, post(&loops, body), status, code).await;
    }
    let none: BenchLoopList = ok(app, get(&loops)).await;
    assert!(none.loops.is_empty(), "a refused loop is not made");

    let started: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 0.001, "student": student, "name": "selfie" })),
    )
    .await;
    assert!(started.id.starts_with("bl_") && started.status == BenchLoopStatus::Running);
    assert_eq!(started.config.generations, Some(1));
    assert_eq!(
        (started.config.student.arch.as_deref(), started.config.student.epochs),
        (Some("folo-n"), Some(1))
    );
    let done = loop_ended(app, &b.id, &started.id).await;
    assert_eq!(done.status, BenchLoopStatus::Done, "{done:?}");
    assert!(done.verdict.is_some() && done.phase.is_none() && done.finished.is_some());
    assert_eq!(done.generations.len(), 2);
    let g0 = &done.generations[0];
    assert_eq!(
        (g0.model_id.as_deref(), g0.labels.clone(), g0.run_id.clone()),
        (Some(teacher.as_str()), None, None)
    );
    assert!(g0.entry_id.is_some() && g0.score.is_some() && g0.map50_95.is_some() && g0.ci95.is_some());
    let g1 = &done.generations[1];
    let labels = g1.labels.clone().unwrap();
    assert_eq!((labels.images, labels.excluded, labels.threshold), (12, 2, 0.001));
    assert!(labels.boxes > 0 && labels.labeled_images > 0);
    assert!(g1.model_id.is_some() && g1.entry_id.is_some() && g1.score.is_some() && g1.ci95.is_some());
    // The student's run is an ordinary run.
    let run: Run = ok(app, get(&format!("/api/runs/{}", g1.run_id.clone().unwrap()))).await;
    assert_eq!(
        (run.status, run.name.as_str(), run.config.dataset_id.as_str()),
        (RunStatus::Completed, "selfie gen 1", labels.dataset_id.as_str())
    );
    assert_eq!(g1.model_id, run.model_id);
    // Its dataset has the teacher's labels and nothing of the pool's own.
    let student_ds: Dataset = ok(app, get(&format!("/api/datasets/{}", labels.dataset_id))).await;
    assert_eq!(
        (student_ds.classes.clone(), student_ds.splits.train + student_ds.splits.val),
        (vec!["circle".to_string(), "square".to_string()], labels.labeled_images)
    );
    assert!(labels.labeled_images >= 10 && labels.labeled_images <= labels.images);
    // Every label line read back as numbers: none is the pool's own label (class 1 at (1, 1)-(7, 7) of 64 px).
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for e in walk(Path::new(&student_ds.path).join("labels")) {
        for line in std::fs::read_to_string(e).unwrap().lines() {
            let row: Vec<f64> = line.split_whitespace().map(|v| v.parse().unwrap()).collect();
            assert_eq!(row.len(), 5, "{line}");
            rows.push(row);
        }
    }
    assert_eq!(rows.len() as u64, labels.boxes);
    let wrong = [1.0, 4.0 / 64.0, 4.0 / 64.0, 6.0 / 64.0, 6.0 / 64.0];
    assert!(
        !rows.iter().any(|r| r.iter().zip(wrong).all(|(a, b)| (a - b).abs() < 1e-4)),
        "the pool's own label reached the student"
    );
    let yaml = std::fs::read_to_string(Path::new(&student_ds.path).join("data.yaml")).unwrap();
    assert!(yaml.starts_with("# made by a FOLO self-improvement loop"), "{yaml}");
    // The teacher's entry was a fresh one, the student's another; both are on the leaderboard.
    let board: BenchEntryList = ok(app, get(&format!("/api/bench/benchmarks/{}/entries", b.id))).await;
    assert_eq!(board.entries.len(), 2);

    // Without min_confidence, a teacher whose entry has no self-labeling threshold (this one never reaches 90%
    // precision on ten labels) ends its loop before labeling: there is no confidence to label from.
    let guess: BenchLoop =
        ok(app, post(&loops, json!({ "teacher": teacher, "pool": pool_src, "student": student }))).await;
    let guess = loop_ended(app, &b.id, &guess.id).await;
    assert_eq!(guess.status, BenchLoopStatus::Failed, "{guess:?}");
    assert!(
        guess
            .error
            .as_deref()
            .unwrap()
            .starts_with("'teacher' never reaches 90% precision on at least 10 labels"),
        "{guess:?}"
    );
    assert!(guess.hint.as_deref().unwrap().contains("min_confidence"));
    assert_eq!((guess.generations.len(), guess.generations[1].labels.clone()), (2, None));

    // A teacher that keeps (almost) nothing ends its loop before anything is trained: an image it is silent on
    // is not a background image.
    let silent: BenchLoop = ok(
        app,
        post(
            &loops,
            json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 1.0, "student": student }),
        ),
    )
    .await;
    let silent = loop_ended(app, &b.id, &silent.id).await;
    assert_eq!(silent.status, BenchLoopStatus::Failed, "{silent:?}");
    assert!(silent.error.as_deref().unwrap().contains("too few to train a student on"), "{silent:?}");
    assert!(silent.hint.is_some() && silent.generations[1].run_id.is_none());

    // A second loop reuses the teacher's finished entry; stopping it while its student trains stops the run.
    f.store.hold_runs_after_epochs(true);
    let second: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 0.001, "student": json!({ "epochs": 3, "imgsz": 64, "batch": 4, "device": "cpu" }) })),
    )
    .await;
    let t0 = std::time::Instant::now();
    let run_id = loop {
        let l: BenchLoop = ok(app, get(&format!("{loops}/{}", second.id))).await;
        if let Some(id) = l.generations.get(1).and_then(|g| g.run_id.clone()) {
            assert_eq!(l.generations[0].entry_id, g0.entry_id, "the teacher's entry is reused");
            break id;
        }
        assert!(t0.elapsed() < TIMEOUT && l.status.is_active(), "{l:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    err(app, delete(&format!("{loops}/{}", second.id)), StatusCode::CONFLICT, "bench_loop_active").await;
    let used = g0.entry_id.clone().unwrap();
    err(
        app,
        delete(&format!("/api/bench/benchmarks/{}/entries/{used}", b.id)),
        StatusCode::CONFLICT,
        "bench_entry_in_loop",
    )
    .await;
    let stopping: BenchLoop = ok(app, post(&format!("{loops}/{}/stop", second.id), json!({}))).await;
    assert_eq!(stopping.status, BenchLoopStatus::Stopping);
    f.store.hold_runs_after_epochs(false);
    let stopped = loop_ended(app, &b.id, &second.id).await;
    assert_eq!((stopped.status, stopped.verdict), (BenchLoopStatus::Stopped, None));
    let run: Run = ok(app, get(&format!("/api/runs/{run_id}"))).await;
    assert_eq!(run.status, RunStatus::Stopped);
    err(
        app,
        post(&format!("{loops}/{}/stop", second.id), json!({})),
        StatusCode::CONFLICT,
        "bench_loop_not_active",
    )
    .await;
    let list: BenchLoopList = ok(app, get(&loops)).await;
    assert_eq!(
        list.loops.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
        [second.id.as_str(), silent.id.as_str(), guess.id.as_str(), started.id.as_str()]
    );
    let (status, _, _) = send(app, delete(&format!("{loops}/{}", second.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(app, get(&format!("{loops}/{}", second.id)), StatusCode::NOT_FOUND, "bench_loop_not_found").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_loop_labels_from_each_teachers_own_threshold_unless_told() {
    let t = TestDir::new("bench-loop-chain");
    let home = t.join("home");
    let (bench_id, entry_id, teacher, pool_src) = {
        let f = fixture(&home);
        let app = &f.app;
        let ds = dataset(app, "shapes", 20).await;
        let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
        let b = bench(
            app,
            json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": plan }),
        )
        .await;
        // A teacher that finds something everywhere (scores near 0.5).
        let teacher = model_with(app, &home, "teacher", &["circle", "square"], 3, true).await;
        let e: BenchEntry = ok(
            app,
            post(
                &format!("/api/bench/benchmarks/{}/entries", b.id),
                json!({ "contestant": { "kind": "model", "model_id": teacher } }),
            ),
        )
        .await;
        assert_eq!(ended(app, &b.id, &e.id).await.status, BenchEntryStatus::Done);
        let pool = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "train" });
        (b.id, e.id, teacher, pool)
    };
    // Its entry's self-labeling threshold, set to 0.37 (what a teacher that reaches the target would have).
    let path = home.join("bench").join(&bench_id).join("entries").join(format!("{entry_id}.json"));
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["result"]["self_labeling"]["threshold"] = json!(0.37);
    std::fs::write(&path, file.to_string()).unwrap();
    let f = fixture(&home);
    let app = &f.app;
    let loops = format!("/api/bench/benchmarks/{bench_id}/loops");
    let student = json!({ "epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu" });

    // Generation 1 labels from the teacher's threshold; generation 2 from its student's, which has none: the loop
    // ends there, naming the student.
    let own: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "generations": 2, "student": student })),
    )
    .await;
    let own = loop_ended(app, &bench_id, &own.id).await;
    assert_eq!(
        own.generations[0].entry_id.as_deref(),
        Some(entry_id.as_str()),
        "the patched entry is reused"
    );
    assert_eq!(own.generations[1].labels.as_ref().map(|l| l.threshold), Some(0.37), "{own:?}");
    let student_name = own.generations[1].model_name.clone().expect("generation 1 trained");
    assert_eq!(own.status, BenchLoopStatus::Failed, "{own:?}");
    assert!(
        own.error.as_deref().unwrap().starts_with(&format!("'{student_name}' never reaches 90% precision")),
        "{own:?}"
    );
    assert!(own.generations.len() == 3 && own.generations[2].labels.is_none());

    // An explicit min_confidence overrides both: generation 1 and generation 2 label from it.
    let told: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "generations": 2, "min_confidence": 0.2, "student": student })),
    )
    .await;
    let told = loop_ended(app, &bench_id, &told.id).await;
    assert_eq!(told.generations[1].labels.as_ref().map(|l| l.threshold), Some(0.2), "{told:?}");
    match told.status {
        BenchLoopStatus::Done => {
            assert_eq!(told.generations[2].labels.as_ref().map(|l| l.threshold), Some(0.2))
        }
        // The student keeps too little at 0.2 to train on: it said so at that confidence.
        _ => assert!(told.error.as_deref().unwrap().contains("at confidence 0.2"), "{told:?}"),
    }
}

/// Every file under `dir`.
fn walk(dir: std::path::PathBuf) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(p));
        } else {
            out.push(p);
        }
    }
    out
}
