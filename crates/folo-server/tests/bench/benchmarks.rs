//! Benchmarks: from a dataset split and from a labeling job, images, variants, plans, seeds, the
//! leaderboard and events.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{MockStore, Options, Server};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    Events, bench, bytes, dataset, delete, err, external, fixture, get, ok, open_entry, page, post, put,
    send, small_plan, truth_predictions,
};

#[tokio::test(flavor = "multi_thread")]
async fn benchmarks_from_a_dataset_split_their_images_and_plans() {
    let t = TestDir::new("bench-create");
    let f = fixture(&t.join("home"));
    let app = &f.app;

    let system: Value = ok(app, get("/api/system")).await;
    assert!(system["features"].as_array().unwrap().contains(&json!("vision_bench")), "{system}");
    let catalog: BenchCatalog = ok(app, get("/api/bench/catalog")).await;
    assert_eq!((catalog.conditions.len(), catalog.severities, catalog.axes.len()), (12, 5, 5));
    // The mock has no benchmarks.
    let mock = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    err(&mock, get("/api/bench/benchmarks"), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
    let mock_system: Value = ok(&mock, get("/api/system")).await;
    assert!(!mock_system["features"].as_array().unwrap().contains(&json!("vision_bench")));

    let ds = dataset(app, "shapes", 20).await;
    let src = json!({ "kind": "dataset", "dataset_id": ds.id });
    // Inputs are checked before anything is made.
    for (body, code, status) in [
        (json!({ "name": " ", "source": src }), "invalid_body", StatusCode::BAD_REQUEST),
        (json!({ "name": "b", "source": src, "max_images": 0 }), "invalid_body", StatusCode::BAD_REQUEST),
        (json!({ "name": "b", "source": src, "max_images": 5001 }), "invalid_body", StatusCode::BAD_REQUEST),
        (
            json!({ "name": "b", "source": src, "plan": { "conditions": [{ "condition": "fog", "severities": [1] }] } }),
            "invalid_body",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": src, "description": "x".repeat(4001) }),
            "invalid_body",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id, "split": "test" } }),
            "invalid_dataset",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": "ds_none" } }),
            "dataset_not_found",
            StatusCode::NOT_FOUND,
        ),
        (
            json!({ "name": "b", "source": { "kind": "label_job", "job_id": "lb_none" } }),
            "label_job_not_found",
            StatusCode::NOT_FOUND,
        ),
    ] {
        err(app, post("/api/bench/benchmarks", body), status, code).await;
    }
    let none: BenchList = ok(app, get("/api/bench/benchmarks")).await;
    assert!(none.benchmarks.is_empty(), "a refused request makes nothing");

    // From the val split (the default), every image, with its truth.
    let val: ImagePage = ok(app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    let b =
        bench(app, json!({ "name": "shapes val", "description": "toy shapes", "source": src, "seed": 3 }))
            .await;
    assert!(b.id.starts_with("bn_"));
    assert_eq!(b.source, BenchSource::Dataset { dataset_id: ds.id.clone(), split: Some(Split::Val) });
    assert_eq!((b.images, b.source_name.as_str(), b.classes.len()), (4, "shapes", 2));
    let truth_boxes: u64 = val.images.iter().map(|i| i.boxes.len() as u64).sum();
    assert_eq!((b.boxes, b.class_boxes.iter().sum::<u64>()), (truth_boxes, truth_boxes));
    assert_eq!(b.plan, folo_bench::default_plan());
    assert_eq!((b.entries, b.best.clone(), b.fingerprint.len()), (0, None, 64));
    let images = page(app, &b.id, "").await;
    assert_eq!(images.images.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), ["1", "2", "3", "4"]);

    // The same source and seed: the same images in the same order and the same fingerprint; another seed is
    // another benchmark.
    let again = bench(app, json!({ "name": "again", "source": src, "seed": 3 })).await;
    assert_eq!(again.fingerprint, b.fingerprint);
    assert_eq!(page(app, &again.id, "").await.images, images.images);
    let other = bench(app, json!({ "name": "other", "source": src, "seed": 4 })).await;
    assert_ne!(other.fingerprint, b.fingerprint);
    // A sample of a larger split: seeded, and a smaller benchmark is the first images of a larger one.
    let train = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "train" });
    let ten = bench(app, json!({ "name": "ten", "source": train, "max_images": 10, "seed": 9 })).await;
    let five = bench(app, json!({ "name": "five", "source": train, "max_images": 5, "seed": 9 })).await;
    assert_eq!((ten.images, five.images), (10, 5));
    let files = |p: BenchImagePage| p.images.into_iter().map(|i| i.file).collect::<Vec<_>>();
    assert_eq!(files(page(app, &ten.id, "").await)[..5], files(page(app, &five.id, "").await));
    let list: BenchList = ok(app, get("/api/bench/benchmarks")).await;
    assert_eq!(list.benchmarks[0].id, five.id, "newest first");

    // Plans: checked, and the fingerprint follows the plan, not the name.
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    let bad = json!({ "plan": { "conditions": [{ "condition": "haze", "severities": [3, 1] }] } });
    err(app, put(&uri, bad), StatusCode::BAD_REQUEST, "invalid_body").await;
    let renamed: Bench = ok(app, put(&uri, json!({ "name": "renamed", "description": "a pond" }))).await;
    assert_eq!((renamed.name.as_str(), renamed.fingerprint.as_str()), ("renamed", b.fingerprint.as_str()));
    let planned: Bench = ok(app, put(&uri, json!({ "plan": small_plan() }))).await;
    assert_ne!(planned.fingerprint, b.fingerprint);
    assert_eq!(planned.plan.conditions.len(), 2);

    // Images: the stored file, the variant every contestant gets, thumbnails; all immutable.
    let truth = page(app, &b.id, "").await;
    let first = &truth.images[0];
    let (headers, stored) = bytes(app, &format!("{uri}/images/1")).await;
    assert!(headers[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    let stored_etag = headers[header::ETAG].to_str().unwrap().to_string();
    let (s304, h304, b304) = send(
        app,
        Request::get(format!("{uri}/images/1"))
            .header(header::IF_NONE_MATCH, &stored_etag)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s304, StatusCode::NOT_MODIFIED);
    assert_eq!(h304[header::ETAG], stored_etag.as_str());
    assert!(h304[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    assert!(b304.is_empty());
    let source_image = val.images.iter().find(|i| i.file.ends_with(&first.file)).unwrap();
    let (_, original) = bytes(app, &format!("/api/datasets/{}/images/{}", ds.id, source_image.id)).await;
    assert_eq!(stored, original, "a benchmark copies its files byte for byte");
    let (headers, haze) = bytes(app, &format!("{uri}/images/1?condition=haze&severity=3")).await;
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
    assert!(headers[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    let haze_etag = headers[header::ETAG].to_str().unwrap().to_string();
    assert_ne!(haze_etag, stored_etag);
    let (s304, _, b304) = send(
        app,
        Request::get(format!("{uri}/images/1?condition=haze&severity=3"))
            .header(header::IF_NONE_MATCH, &haze_etag)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s304, StatusCode::NOT_MODIFIED);
    assert!(b304.is_empty());
    let seed = folo_bench::conditions::variant_seed(3, "1", folo_bench::Condition::Haze, 3);
    let want =
        folo_bench::conditions::render(&stored, folo_bench::Condition::Haze, 3, &first.boxes, seed).unwrap();
    assert_eq!(haze, want.jpeg, "the variant is the engine's render of the stored file");
    let (_, again_haze) = bytes(app, &format!("{uri}/images/1?condition=haze&severity=3")).await;
    assert_eq!(again_haze, haze);
    for q in
        ["?thumb=32", "?condition=distance&severity=2&thumb=32", "?condition=distance&severity=2&thumb=32"]
    {
        let (h, thumb) = bytes(app, &format!("{uri}/images/1{q}")).await;
        assert!(
            thumb.starts_with(&[0xff, 0xd8])
                && h[header::CACHE_CONTROL].to_str().unwrap().contains("immutable")
        );
    }
    // The truth moves under distance (and only there).
    let far = page(app, &b.id, "&condition=distance&severity=2").await;
    let hazy = page(app, &b.id, "&condition=haze&severity=5").await;
    assert_eq!(hazy.images, truth.images);
    let with_boxes = truth.images.iter().position(|i| !i.boxes.is_empty()).unwrap();
    assert_ne!(far.images[with_boxes].boxes, truth.images[with_boxes].boxes);
    let (_, dist) =
        bytes(app, &format!("{uri}/images/{}?condition=distance&severity=2", truth.images[with_boxes].id))
            .await;
    let im = &truth.images[with_boxes];
    let seed = folo_bench::conditions::variant_seed(3, &im.id, folo_bench::Condition::Distance, 2);
    let (_, file) = bytes(app, &format!("{uri}/images/{}", im.id)).await;
    let r =
        folo_bench::conditions::render(&file, folo_bench::Condition::Distance, 2, &im.boxes, seed).unwrap();
    assert_eq!((dist, r.boxes), (r.jpeg.clone(), far.images[with_boxes].boxes.clone()));
    for q in ["?condition=haze", "?severity=2", "?condition=fog&severity=1", "?condition=haze&severity=9"] {
        err(app, get(&format!("{uri}/images/1{q}")), StatusCode::BAD_REQUEST, "invalid_query").await;
    }
    err(app, get(&format!("{uri}/images/99")), StatusCode::NOT_FOUND, "image_not_found").await;

    // Deleting a benchmark removes it and its folder.
    let (status, _, _) = send(app, delete(&format!("/api/bench/benchmarks/{}", other.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(app, get(&format!("/api/bench/benchmarks/{}", other.id)), StatusCode::NOT_FOUND, "bench_not_found")
        .await;
    assert!(!t.join("home").join("bench").join(&other.id).exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_benchmark_from_a_labeling_job_and_its_events() {
    let t = TestDir::new("bench-job");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let job: LabelJob =
        ok(app, post("/api/label/jobs", json!({ "name": "pond", "classes": ["fish", "frog"] }))).await;
    let ds = dataset(app, "pond-images", 10).await;
    let imgs: ImagePage = ok(app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    // Three images uploaded, two labeled by hand (one of them as a background image), one left pending.
    let boundary = "XfoloBench";
    let mut body = Vec::new();
    for im in &imgs.images[..3] {
        let (_, b) = bytes(app, &format!("/api/datasets/{}/images/{}", ds.id, im.id)).await;
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{}.jpg\"\r\n\r\n",
                im.id
            )
            .as_bytes(),
        );
        body.extend_from_slice(&b);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let req = Request::post(format!("/api/label/jobs/{}/images", job.id))
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let added: LabelImagesAdded = ok(app, req).await;
    let ids: Vec<String> = added.added.iter().map(|i| i.id.clone()).collect();
    let drawn = json!({ "boxes": [
        { "class": 1, "x1": 4, "y1": 4, "x2": 30, "y2": 20 },
        { "class": 0, "x1": 50, "y1": 10, "x2": 60, "y2": 30 } ] });
    ok::<LabelImage>(app, put(&format!("/api/label/jobs/{}/images/{}", job.id, ids[0]), drawn)).await;
    ok::<LabelImage>(
        app,
        put(&format!("/api/label/jobs/{}/images/{}", job.id, ids[2]), json!({ "boxes": [] })),
    )
    .await;

    let b = bench(
        app,
        json!({ "name": "pond", "source": { "kind": "label_job", "job_id": job.id }, "plan": { "conditions": [] } }),
    )
    .await;
    assert_eq!(
        (b.images, b.boxes, b.class_boxes.clone(), b.classes.clone()),
        (2, 2, vec![1, 1], job.classes.clone())
    );
    assert_eq!(b.source_name, "pond");
    let truth = page(app, &b.id, "").await;
    let mut boxes: Vec<BBox> = truth.images.iter().flat_map(|i| i.boxes.clone()).collect();
    boxes.sort_by_key(|b| b.class);
    assert_eq!(boxes[1], BBox { class: 1, x1: 4.0, y1: 4.0, x2: 30.0, y2: 20.0 });
    // A job with nothing labeled is no source.
    let empty: LabelJob = ok(app, post("/api/label/jobs", json!({ "name": "e", "classes": ["a"] }))).await;
    let src = json!({ "kind": "label_job", "job_id": empty.id });
    err(
        app,
        post("/api/bench/benchmarks", json!({ "name": "e", "source": src })),
        StatusCode::BAD_REQUEST,
        "invalid_dataset",
    )
    .await;

    // Events: the benchmark first, then each change in order.
    let mut events = Events::open(app, &b.id).await;
    let (name, first) = events.next().await;
    assert_eq!((name.as_str(), first["id"].as_str()), ("bench", Some(b.id.as_str())));
    let e: BenchEntry = ok(
        app,
        post(
            &format!("/api/bench/benchmarks/{}/entries", b.id),
            external("truth", true, truth_predictions(&truth.images, None, 0.9)),
        ),
    )
    .await;
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["id"].as_str(), data["status"].as_str()),
        ("entry", Some(e.id.as_str()), Some("done"))
    );
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["entries"].as_u64(), data["best"]["entry_id"].as_str()),
        ("bench", Some(1), Some(e.id.as_str()))
    );
    ok::<Bench>(app, put(&format!("/api/bench/benchmarks/{}", b.id), json!({ "name": "pond 2" }))).await;
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["name"].as_str()), ("bench", Some("pond 2")));
    let (status, _, _) = send(app, delete(&format!("/api/bench/benchmarks/{}/entries/{}", b.id, e.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["id"].as_str()), ("removed", Some(e.id.as_str())));
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["entries"].as_u64(), data["best"].is_null()), ("bench", Some(0), true));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_subscriber_that_falls_behind_gets_the_benchmark_again() {
    let t = TestDir::new("bench-lag");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 10).await;
    let b = bench(app, json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id } })).await;
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    let mut events = Events::open(app, &b.id).await;
    assert_eq!(events.next().await.0, "bench");
    // More changes than the stream buffers (1024) while the subscriber reads nothing.
    for i in 0..1100 {
        ok::<Bench>(app, put(&uri, json!({ "name": format!("name {i}") }))).await;
    }
    // The first thing it reads is the benchmark as it is now.
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["name"].as_str()), ("bench", Some("name 1099")));
}

#[tokio::test(flavor = "multi_thread")]
async fn plans_seeds_and_the_leaderboard() {
    let t = TestDir::new("bench-board");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let src = json!({ "kind": "dataset", "dataset_id": ds.id });
    // A seed a browser cannot hold is refused.
    let e = err(
        app,
        post(
            "/api/bench/benchmarks",
            json!({ "name": "b", "source": src, "seed": 9_007_199_254_740_992u64 }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e.message.contains("seed must be at most 9007199254740991"), "{}", e.message);
    let b = bench(
        app,
        json!({ "name": "b", "source": src, "seed": 9_007_199_254_740_991u64, "plan": small_plan() }),
    )
    .await;
    assert!(b.id.len() > 12, "a long random id: {}", b.id);
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    let entries = format!("{uri}/entries");
    let images = page(app, &b.id, "").await.images;

    // The plan is frozen by an active entry (an open one here), not by a stopped one ...
    let open: BenchEntry = ok(app, post(&entries, open_entry("open", vec![]))).await;
    let e =
        err(app, put(&uri, json!({ "plan": { "conditions": [] } })), StatusCode::CONFLICT, "bench_frozen")
            .await;
    assert!(e.message.contains("running entry ('open')"), "{}", e.message);
    ok::<BenchEntry>(app, post(&format!("{entries}/{}/stop", open.id), json!({}))).await;
    ok::<Bench>(app, put(&uri, json!({ "plan": small_plan() }))).await;

    // ... and by a done one. The leaderboard: complete entries first by score, then partial ones by score (equal
    // scores: the older first), then the unfinished; `best` is the first.
    let mut whole = truth_predictions(&images, None, 1.0);
    for (c, s) in [("haze", 1), ("haze", 3), ("distance", 2)] {
        let moved = page(app, &b.id, &format!("&condition={c}&severity={s}")).await;
        whole.extend(truth_predictions(&moved.images, Some((c, s)), 0.6));
    }
    let shifted: Vec<Value> = whole
        .iter()
        .map(|r| {
            let mut r = r.clone();
            for b in r["boxes"].as_array_mut().unwrap() {
                b["x1"] = json!(b["x1"].as_f64().unwrap() + 2.0);
                b["x2"] = json!(b["x2"].as_f64().unwrap() + 2.0);
            }
            r
        })
        .collect();
    let complete: BenchEntry = ok(app, post(&entries, external("complete", true, shifted))).await;
    let partial_a: BenchEntry =
        ok(app, post(&entries, external("partial a", true, truth_predictions(&images, None, 1.0)))).await;
    let partial_b: BenchEntry =
        ok(app, post(&entries, external("partial b", true, truth_predictions(&images, None, 1.0)))).await;
    let c = complete.score.clone().unwrap();
    let p = partial_a.score.clone().unwrap();
    assert!(!c.partial && p.partial && p.total > c.total, "{c:?} {p:?}");
    assert_eq!(partial_b.score.clone().unwrap().total, p.total);
    let running: BenchEntry = ok(app, post(&entries, open_entry("running", vec![]))).await;
    let board: BenchEntryList = ok(app, get(&entries)).await;
    let names: Vec<&str> = board.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["complete", "partial a", "partial b", "running", "open"]);
    let now: Bench = ok(app, get(&uri)).await;
    let best = now.best.unwrap();
    assert_eq!((best.entry_id, best.partial), (complete.id.clone(), false));
    ok::<BenchEntry>(app, post(&format!("{entries}/{}/stop", running.id), json!({}))).await;
    err(app, put(&uri, json!({ "plan": { "conditions": [] } })), StatusCode::CONFLICT, "bench_frozen").await;
}
