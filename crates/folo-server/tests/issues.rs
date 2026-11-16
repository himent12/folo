//! `GET /api/datasets/{id}/issues` in mock and real mode.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{MockStore, Options, RealStore, Server};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    (status, res.into_body().collect().await.unwrap().to_bytes().to_vec())
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn post_json(uri: &str, v: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().to_string();
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap()
}

async fn err_code(app: &Router, req: Request<Body>) -> (StatusCode, String) {
    let (status, body) = send(app, req).await;
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    (status, e.error.code)
}

fn count(page: &IssuePage, kind: &str) -> usize {
    page.issues.iter().filter(|i| i.kind == kind).count()
}

#[tokio::test]
async fn mock_issues_agree_with_the_stats() {
    let app = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    for d in &list.datasets {
        let ds: Dataset = ok(&app, get(&format!("/api/datasets/{}", d.id))).await;
        let stats = ds.stats.unwrap();
        let all: IssuePage = ok(&app, get(&format!("/api/datasets/{}/issues?limit=1000", d.id))).await;
        assert!(all.total <= 1000 || all.issues.len() == 1000);
        let empty: IssuePage =
            ok(&app, get(&format!("/api/datasets/{}/issues?kind=empty_image&limit=1000", d.id))).await;
        assert_eq!(empty.total, stats.empty_images, "{}", d.name);
        assert!(empty.issues.iter().all(|i| i.severity == IssueSeverity::Info && i.line.is_none()));
        let errors = all.issues.iter().filter(|i| i.severity == IssueSeverity::Error).count() as u64;
        assert_eq!(errors, stats.broken_labels, "{}: one error per broken label file", d.name);
        for i in &all.issues {
            assert!(ISSUE_KINDS.contains(&i.kind.as_str()), "{}", i.kind);
            assert!(i.file.starts_with("images/") || i.file.starts_with("labels/"), "{}", i.file);
            assert_eq!(i.line.is_some(), i.file.starts_with("labels/"), "{i:?}");
        }
        // Deterministic.
        let again: IssuePage = ok(&app, get(&format!("/api/datasets/{}/issues?limit=1000", d.id))).await;
        assert_eq!(again, all);
    }
    let traffic = list.datasets.iter().find(|d| d.name == "traffic-cams").unwrap();
    let all: IssuePage = ok(&app, get(&format!("/api/datasets/{}/issues?limit=1000", traffic.id))).await;
    assert!(count(&all, "clipped_box") + count(&all, "duplicate_image") > 0, "a few warnings");
    // Paging.
    let page: IssuePage =
        ok(&app, get(&format!("/api/datasets/{}/issues?offset=2&limit=3", traffic.id))).await;
    assert_eq!((page.total, page.issues.as_slice()), (all.total, &all.issues[2..5]));
    // Severity filter and counts (over every issue, whatever the filter).
    let errors: IssuePage =
        ok(&app, get(&format!("/api/datasets/{}/issues?severity=error&limit=1000", traffic.id))).await;
    assert_eq!(errors.total, 3);
    assert!(errors.issues.iter().all(|i| i.severity == IssueSeverity::Error && i.fix.is_some()));
    assert_eq!(errors.counts, all.counts);
    assert_eq!(all.counts.error + all.counts.warning + all.counts.info, all.total);
    assert_eq!(all.counts.by_kind.values().sum::<u64>(), all.total);
    assert_eq!(
        all.counts.by_kind.get("empty_image").copied().unwrap_or(0),
        count(&all, "empty_image") as u64
    );
    assert!(all.issues.iter().filter(|i| i.kind == "empty_image").all(|i| i.fix.is_none()));
    // Errors.
    let bad = format!("/api/datasets/{}/issues?kind=nope", traffic.id);
    assert_eq!(err_code(&app, get(&bad)).await, (StatusCode::BAD_REQUEST, "invalid_query".into()));
    let bad = format!("/api/datasets/{}/issues?severity=fatal", traffic.id);
    assert_eq!(err_code(&app, get(&bad)).await, (StatusCode::BAD_REQUEST, "invalid_query".into()));
    let missing = err_code(&app, get("/api/datasets/ds_none/issues")).await;
    assert_eq!(missing, (StatusCode::NOT_FOUND, "dataset_not_found".into()));
}

fn write_shapes(dir: &Path) {
    let cfg = folo_data::synthetic::SyntheticConfig { images: 10, imgsz: 64, seed: 2, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, dir, folo_data::ImageFormat::Png).unwrap();
}

/// `stats.broken_labels` counts images left out for a broken label (docs/API.md), one per broken label file however
/// many broken lines it has, while `/issues` lists every broken line: a dataset where files, lines and kept images
/// all differ (d1-datasets-3).
#[tokio::test]
async fn real_broken_labels_count_the_images_left_out_not_the_lines() {
    let t = TestDir::new("issues-broken-count");
    let (home, data) = (t.join("home"), t.join("shapes"));
    write_shapes(&data);
    let clean = folo_data::open(&data).unwrap().split_sizes().iter().sum::<usize>() as u64;
    // Three broken lines (a class out of range, too few values, a box outside the image) beside a good one, and a
    // second file with one broken line.
    let three = "0 0.5 0.5 0.2 0.2\n42 0.5 0.5 0.1 0.1\n0 0.5 0.5\n0 1.5 1.5 0.2 0.2\n";
    std::fs::write(data.join("labels/train/000001.txt"), three).unwrap();
    std::fs::write(data.join("labels/train/000002.txt"), "42 0.5 0.5 0.1 0.1\n").unwrap();
    let store = RealStore::open(home).recheck_every(Duration::ZERO);
    let app = Server::new(Arc::new(store), Options::default()).router();
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": data.to_str().unwrap() }))).await;
    let stats = ds.stats.clone().unwrap();
    assert_eq!(stats.broken_labels, 2, "two broken label files");
    assert_eq!(ds.splits.train + ds.splits.val + ds.splits.test, clean - 2, "their two images are left out");
    let errors: IssuePage =
        ok(&app, get(&format!("/api/datasets/{}/issues?severity=error&limit=1000", ds.id))).await;
    let mut lines: Vec<(&str, Option<u32>)> =
        errors.issues.iter().map(|i| (i.file.as_str(), i.line)).collect();
    lines.sort();
    assert_eq!(
        lines,
        [
            ("labels/train/000001.txt", Some(2)),
            ("labels/train/000001.txt", Some(3)),
            ("labels/train/000001.txt", Some(4)),
            ("labels/train/000002.txt", Some(1)),
        ],
        "every broken line"
    );
}

#[tokio::test]
async fn real_issues_are_cached_and_refreshed() {
    let t = TestDir::new("issues-real");
    let (home, data) = (t.join("home"), t.join("shapes"));
    write_shapes(&data);
    std::fs::write(data.join("labels/train/000001.txt"), "0 0.5 0.5 0.2 0.2\n42 0.5 0.5 0.1 0.1\n").unwrap();
    std::fs::write(data.join("images/train/000002.png"), b"not a png").unwrap();
    std::fs::copy(data.join("images/train/000003.png"), data.join("images/train/000003b.png")).unwrap();
    let app = |home: &Path| {
        let store = RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO);
        Server::new(Arc::new(store), Options::default()).router()
    };
    let a = app(&home);
    let ds: Dataset = ok(&a, post_json("/api/datasets", json!({ "path": data.to_str().unwrap() }))).await;
    let uri = format!("/api/datasets/{}/issues?limit=1000", ds.id);
    let all: IssuePage = ok(&a, get(&uri)).await;
    let find = |page: &IssuePage, kind: &str| page.issues.iter().find(|i| i.kind == kind).cloned();
    let overflow = find(&all, "class_overflow").expect("class_overflow");
    assert_eq!(
        (overflow.file.as_str(), overflow.line, overflow.severity),
        ("labels/train/000001.txt", Some(2), IssueSeverity::Error)
    );
    assert_eq!(overflow.fix.as_deref(), Some("delete line 2"));
    assert_eq!(all.counts.by_kind["class_overflow"], 1);
    assert_eq!(find(&all, "corrupt_image").unwrap().file, "images/train/000002.png");
    assert_eq!(find(&all, "duplicate_image").unwrap().file, "images/train/000003b.png");
    assert_eq!(count(&all, "empty_image") as u64, ds.stats.unwrap().empty_images);
    let only: IssuePage = ok(&a, get(&format!("/api/datasets/{}/issues?kind=corrupt_image", ds.id))).await;
    assert_eq!((only.total, only.issues.len()), (1, 1));

    // Fixing the files clears their issues (re-checked on request), and the result survives a restart.
    std::fs::write(data.join("labels/train/000001.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
    std::fs::remove_file(data.join("images/train/000002.png")).unwrap();
    let fixed: IssuePage = ok(&a, get(&uri)).await;
    assert!(find(&fixed, "class_overflow").is_none() && find(&fixed, "corrupt_image").is_none(), "{fixed:?}");
    drop(a);
    let restarted: IssuePage = ok(&app(&home), get(&uri)).await;
    assert_eq!(restarted, fixed);
    assert_eq!(err_code(&app(&home), get("/api/datasets/ds_none/issues")).await.1, "dataset_not_found");
}
