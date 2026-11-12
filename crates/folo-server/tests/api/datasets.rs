//! Datasets and their images, in mock mode and on the real store's disk.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Options, RealStore, Server};
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{err, get, mock, ok, post_json, send};

#[tokio::test]
async fn datasets_and_images() {
    let (server, _) = mock();
    let app = server.router();
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    let names: Vec<_> = list.datasets.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["shapes-1k", "traffic-cams", "fasteners"]);
    assert!(list.datasets.iter().all(|d| d.stats.is_none()));
    let fasteners = list.datasets.iter().find(|d| d.name == "fasteners").unwrap();
    assert_eq!(fasteners.classes, ["bolt", "nut"]);

    let full: Dataset = ok(&app, get(&format!("/api/datasets/{}", fasteners.id))).await;
    let stats = full.stats.unwrap();
    assert_eq!(stats.class_counts.iter().sum::<u64>(), stats.boxes);
    assert_eq!(stats.box_size_hist.counts.iter().sum::<u64>(), stats.boxes);
    let n = full.splits.train + full.splits.val + full.splits.test;
    assert_eq!(stats.boxes_per_image_hist.counts.iter().sum::<u64>(), n);

    let page: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?split=val&offset=10&limit=5", fasteners.id))).await;
    assert_eq!(page.total, full.splits.val);
    assert_eq!(page.images.len(), 5);
    assert!(page.images.iter().all(|i| i.split == Split::Val && i.width == 1920 && i.height == 1080));
    let nuts: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?class=1&limit=1000", fasteners.id))).await;
    assert!(nuts.images.iter().all(|i| i.boxes.iter().any(|b| b.class == 1)));
    assert!(nuts.total < n);
    let again: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?split=val&offset=10&limit=5", fasteners.id))).await;
    assert_eq!(again, page, "mock data is deterministic");
    err(
        &app,
        get(&format!("/api/datasets/{}/images?split=nope", fasteners.id)),
        StatusCode::BAD_REQUEST,
        "invalid_query",
    )
    .await;

    let img = &page.images[0];
    let (status, headers, body) =
        send(&app, get(&format!("/api/datasets/{}/images/{}", fasteners.id, img.id))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
    let decoded = image::load_from_memory(&body).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (1920, 1080));
    let (_, _, thumb) =
        send(&app, get(&format!("/api/datasets/{}/images/{}?thumb=256", fasteners.id, img.id))).await;
    let t = image::load_from_memory(&thumb).unwrap();
    assert_eq!((t.width(), t.height()), (256, 144));
    err(
        &app,
        get(&format!("/api/datasets/{}/images/zzz", fasteners.id)),
        StatusCode::NOT_FOUND,
        "image_not_found",
    )
    .await;
    err(&app, get("/api/datasets/ds_none"), StatusCode::NOT_FOUND, "dataset_not_found").await;

    // Synthetic datasets.
    let body =
        json!({ "name": "tiny", "images": 20, "classes": ["circle", "star"], "imgsz": 320, "seed": 4 });
    let (status, _, b) = send(&app, post_json("/api/datasets/synthetic", body)).await;
    assert_eq!(status, StatusCode::CREATED);
    let syn: Dataset = serde_json::from_slice(&b).unwrap();
    assert_eq!(syn.splits, Splits { train: 16, val: 4, test: 0 });
    assert_eq!(syn.stats.as_ref().unwrap().class_counts.len(), 2);
    let tp: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images", syn.id))).await;
    assert_eq!(tp.total, 20);
    err(
        &app,
        post_json("/api/datasets/synthetic", json!({ "name": "" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(
        &app,
        post_json("/api/datasets/synthetic", json!({ "images": 3 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;

    // Register from disk: a missing path is a clean 404, a relative one a 400 that does not name the server's
    // working directory.
    let gone = TestDir::new("gone");
    let missing = gone.join("definitely/not/here");
    err(
        &app,
        post_json("/api/datasets", json!({ "path": missing })),
        StatusCode::NOT_FOUND,
        "path_not_found",
    )
    .await;
    let e = err(
        &app,
        post_json("/api/datasets", json!({ "path": "bccd" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert_eq!(e.message, "'bccd' is not a full path");

    // Delete.
    let (status, _, _) =
        send(&app, Request::delete(format!("/api/datasets/{}", syn.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 3);
    err(
        &app,
        Request::delete(format!("/api/datasets/{}", syn.id)).body(Body::empty()).unwrap(),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
}

#[tokio::test]
async fn real_store_disk_dataset_and_engine_checks() {
    // More real-store dataset tests: tests/real_datasets.rs.
    let root = TestDir::new("real-store");
    let ds_dir = root.join("ds");
    std::fs::create_dir_all(ds_dir.join("images/train")).unwrap();
    std::fs::create_dir_all(ds_dir.join("labels/train")).unwrap();
    std::fs::write(ds_dir.join("data.yaml"), "names:\n  0: bolt\n  1: nut\n").unwrap();
    image::RgbImage::from_pixel(64, 32, image::Rgb([1, 2, 3]))
        .save(ds_dir.join("images/train/x.png"))
        .unwrap();
    std::fs::write(ds_dir.join("labels/train/x.txt"), "0 0.5 0.5 0.5 0.5\n1 0.5 0.25 0.1 0.1\n").unwrap();

    let data_dir = root.join("home");
    let server = Server::new(Arc::new(RealStore::open(data_dir.clone())), Options::default());
    let app = server.router();
    let h: Health = ok(&app, get("/api/health")).await;
    assert!(!h.mock);
    // Pasted as Windows' "Copy as path" gives it: in quotes, with a stray space.
    let quoted = format!(" \"{}\"", ds_dir.to_str().unwrap());
    let (status, _, b) = send(&app, post_json("/api/datasets", json!({ "path": quoted }))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let ds: Dataset = serde_json::from_slice(&b).unwrap();
    assert_eq!((ds.name.as_str(), ds.format), ("ds", DatasetFormat::Yolo));
    assert_eq!(ds.path, folo_api::json::path(&ds_dir));
    assert_eq!(ds.classes, ["bolt", "nut"]);
    assert_eq!(ds.stats.unwrap().class_counts, [1, 1]);
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    assert_eq!(page.images[0].boxes[0].x1, 16.0);
    let (status, headers, _) = send(&app, get(&format!("/api/datasets/{}/images/0", ds.id))).await;
    assert_eq!((status, headers[header::CONTENT_TYPE].to_str().unwrap()), (StatusCode::OK, "image/png"));

    // Registrations persist across restarts.
    let reopened = Server::new(Arc::new(RealStore::open(data_dir.clone())), Options::default()).router();
    let list: DatasetList = ok(&reopened, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 1);

    // Training and inference: tests/real_runs/. Here only the checks before any work starts.
    let cfg = json!({ "dataset_id": ds.id, "epochs": 0 });
    err(&app, post_json("/api/runs", cfg), StatusCode::BAD_REQUEST, "invalid_config").await;
    let p = json!({ "model_id": "m_1", "dataset_id": ds.id, "image_id": "0" });
    err(&app, post_json("/api/predict", p), StatusCode::NOT_FOUND, "model_not_found").await;
    let p = json!({ "model_id": "m_1", "dataset_id": ds.id, "image_id": "9" });
    err(&app, post_json("/api/predict", p), StatusCode::NOT_FOUND, "image_not_found").await;
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert!(runs.runs.is_empty());
    let models: ModelList = ok(&app, get("/api/models")).await;
    assert!(models.models.is_empty());
    let s: SystemInfo = ok(&app, get("/api/system")).await;
    assert!(s.gpus.iter().enumerate().all(|(i, g)| g.ordinal == i as u32 && !g.name.is_empty()));

    let (status, _, _) =
        send(&app, Request::delete(format!("/api/datasets/{}", ds.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    assert!(list.datasets.is_empty());
    assert!(ds_dir.join("images/train/x.png").exists(), "unregistering never deletes files");
    std::fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn datasets_without_classes_cannot_train() {
    let dir = TestDir::new("empty-ds");
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::create_dir_all(dir.join("labels")).unwrap();
    let (server, _) = mock();
    let app = server.router();
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": dir.to_str().unwrap() }))).await;
    assert!(ds.classes.is_empty());
    let e = err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("no classes"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn malformed_registry_is_never_overwritten() {
    let root = TestDir::new("bad-registry");
    std::fs::create_dir_all(root.join("ds/images")).unwrap();
    std::fs::create_dir_all(root.join("ds/labels")).unwrap();
    std::fs::write(root.join("datasets.json"), "{ not json").unwrap();
    let app = Server::new(Arc::new(RealStore::open(root.to_path_buf())), Options::default()).router();
    let body = json!({ "path": root.join("ds").to_str().unwrap() });
    err(&app, post_json("/api/datasets", body), StatusCode::INTERNAL_SERVER_ERROR, "internal").await;
    assert_eq!(std::fs::read_to_string(root.join("datasets.json")).unwrap(), "{ not json");
    std::fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn mock_mode_deletes_a_generated_set_with_files_but_never_a_folder() {
    let dir = TestDir::new("mock-delete-files");
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::create_dir_all(dir.join("labels")).unwrap();
    let (server, _) = mock();
    let app = server.router();
    let body = json!({ "name": "tiny", "images": 10, "imgsz": 64 });
    let made: Dataset = ok(&app, post_json("/api/datasets/synthetic", body)).await;
    let disk: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": dir.to_str().unwrap() }))).await;
    let del =
        |id: &str| Request::delete(format!("/api/datasets/{id}?files=true")).body(Body::empty()).unwrap();
    err(&app, del(&disk.id), StatusCode::BAD_REQUEST, "invalid_query").await;
    assert!(dir.join("images").is_dir());
    let (status, _, _) = send(&app, del(&made.id)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
}
