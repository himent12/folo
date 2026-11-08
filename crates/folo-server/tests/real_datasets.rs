//! The real store's datasets: registration, synthetic generation, listing, images and thumbnails, the registry and
//! the index cache across restarts, and re-indexing when files change.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Options, RealStore, Server};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("folo-real-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A new server on the same data dir (a restart), for tests whose `app` variable shadows [`app`].
fn restart(home: &Path) -> Router {
    app(home)
}

fn app(home: &Path) -> Router {
    let store = RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO);
    Server::new(Arc::new(store), Options::default()).router()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
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

fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().to_string();
    let (status, _, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().to_string();
    let (got, _, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}");
    e.error
}

/// A small synthetic YOLO dataset written with folo-data.
fn write_shapes(dir: &Path, images: usize) {
    let cfg = folo_data::synthetic::SyntheticConfig { images, imgsz: 64, seed: 5, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, dir, folo_data::ImageFormat::Png).unwrap();
}

#[tokio::test]
async fn synthetic_datasets_images_and_thumbnails() {
    let t = TempDir::new("synth");
    let home = t.join("home");
    let app = app(&home);
    let body = json!({ "name": "Shapes Mini", "images": 12, "imgsz": 64, "classes": ["apple", "square"], "seed": 1 });
    let (status, _, b) = send(&app, post_json("/api/datasets/synthetic", body.clone())).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let ds: Dataset = serde_json::from_slice(&b).unwrap();
    assert_eq!(ds.name, "Shapes Mini");
    assert_eq!(ds.format, DatasetFormat::Yolo);
    assert_eq!(ds.classes, ["apple", "square"]);
    assert_eq!(ds.splits, Splits { train: 10, val: 2, test: 0 });
    assert_eq!(Path::new(&ds.path), home.join("datasets").join("shapes-mini"));
    let stats = ds.stats.clone().unwrap();
    assert!(stats.boxes > 0);
    assert_eq!(stats.class_counts.iter().sum::<u64>(), stats.boxes);
    // The same name again gets its own folder.
    let again: Dataset = ok(&app, post_json("/api/datasets/synthetic", body)).await;
    assert_eq!(Path::new(&again.path), home.join("datasets").join("shapes-mini-2"));
    assert_ne!(again.id, ds.id);

    let all: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?limit=1000", ds.id))).await;
    assert_eq!((all.total, all.images.len()), (12, 12));
    for im in &all.images {
        assert_eq!((im.width, im.height), (64, 64));
        assert!(im.file.starts_with("images/"), "{}", im.file);
        for b in &im.boxes {
            assert!(
                0.0 <= b.x1 && b.x1 < b.x2 && b.x2 <= 64.0 && 0.0 <= b.y1 && b.y1 < b.y2 && b.y2 <= 64.0,
                "{b:?}"
            );
        }
    }
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?offset=10&limit=5", ds.id))).await;
    assert_eq!((page.total, page.images.len()), (12, 2));
    assert_eq!(page.images[0].id, all.images[10].id);
    let val: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    assert_eq!(val.total, 2);
    assert!(val.images.iter().all(|i| i.split == Split::Val));
    let squares: ImagePage =
        ok(&app, get(&format!("/api/datasets/{}/images?class=1&limit=1000", ds.id))).await;
    let want = all.images.iter().filter(|i| i.boxes.iter().any(|b| b.class == 1)).count() as u64;
    assert_eq!(squares.total, want);
    assert!(squares.images.iter().all(|i| i.boxes.iter().any(|b| b.class == 1)));

    let id = &all.images[3].id;
    let (status, headers, bytes) = send(&app, get(&format!("/api/datasets/{}/images/{id}", ds.id))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
    assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 64);
    let thumb = format!("/api/datasets/{}/images/{id}?thumb=32", ds.id);
    let (status, headers, bytes) = send(&app, get(&thumb)).await;
    assert_eq!((status, &headers[header::CONTENT_TYPE]), (StatusCode::OK, &"image/jpeg".parse().unwrap()));
    assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 32);
    let cached: Vec<_> = std::fs::read_dir(home.join("cache/thumbs").join(&ds.id)).unwrap().collect();
    assert_eq!(cached.len(), 1, "the thumbnail is cached");
    assert_eq!(send(&app, get(&thumb)).await.2, bytes);
    // Thumbnails are never upscaled.
    let (_, _, big) = send(&app, get(&format!("/api/datasets/{}/images/{id}?thumb=2048", ds.id))).await;
    assert_eq!(image::load_from_memory(&big).unwrap().width(), 64);
    for bad in ["999", "abc", "03"] {
        let uri = format!("/api/datasets/{}/images/{bad}", ds.id);
        err(&app, get(&uri), StatusCode::NOT_FOUND, "image_not_found").await;
    }

    for (body, what) in [
        (json!({ "name": "x", "classes": ["a", "b", "c", "d", "e", "f", "g", "h", "i"] }), "classes"),
        (json!({ "name": "x", "classes": ["a", "a"] }), "twice"),
        (json!({ "name": "x", "images": 0 }), "images"),
        (json!({ "name": "x", "imgsz": 8 }), "imgsz"),
        (json!({ "name": " " }), "name"),
    ] {
        let e =
            err(&app, post_json("/api/datasets/synthetic", body), StatusCode::BAD_REQUEST, "invalid_body")
                .await;
        assert!(e.message.contains(what), "{what}: {}", e.message);
    }
    // Runs are checked against the engine's limits before anything is queued (tests/real_runs.rs trains).
    let cfg = json!({ "dataset_id": ds.id, "imgsz": 48 });
    err(&app, post_json("/api/runs", cfg), StatusCode::BAD_REQUEST, "invalid_config").await;
}

#[tokio::test]
async fn registry_and_index_cache_survive_restarts_and_track_changes() {
    let t = TempDir::new("registry");
    let (home, data) = (t.join("home"), t.join("shapes"));
    write_shapes(&data, 10);
    let app1 = app(&home);
    let ds: Dataset =
        ok(&app1, post_json("/api/datasets", json!({ "path": data.to_str().unwrap(), "name": "mine" })))
            .await;
    let boxes = ds.stats.unwrap().boxes;
    let cache = home.join("cache/datasets").join(format!("{}.json", ds.id));
    assert!(cache.is_file(), "the index is cached");
    let registry: Value =
        serde_json::from_slice(&std::fs::read(home.join("datasets.json")).unwrap()).unwrap();
    assert_eq!(registry[0]["id"], ds.id.as_str());
    assert_eq!(registry[0]["name"], "mine");
    drop(app1);

    // Unchanged files: the restart loads the cache instead of rebuilding it.
    let stamp = std::fs::metadata(&cache).unwrap().modified().unwrap();
    let app2 = app(&home);
    let list: DatasetList = ok(&app2, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 1);
    assert_eq!((list.datasets[0].name.as_str(), list.datasets[0].splits.train), ("mine", 8));
    let got: Dataset = ok(&app2, get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.stats.unwrap().boxes, boxes);
    assert_eq!(std::fs::metadata(&cache).unwrap().modified().unwrap(), stamp, "cache reused");
    drop(app2);

    // A label edited while the server was down: the restart re-indexes.
    let label = data.join("labels/train/000000.txt");
    let mut text = std::fs::read_to_string(&label).unwrap();
    text.push_str("0 0.5 0.5 0.123 0.321\n");
    std::fs::write(&label, &text).unwrap();
    let app3 = app(&home);
    let got: Dataset = ok(&app3, get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.stats.unwrap().boxes, boxes + 1);

    // ... and while it runs (this store re-checks on every request).
    std::fs::remove_file(data.join("images/val/000009.png")).unwrap();
    let got: Dataset = ok(&app3, get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.splits, Splits { train: 8, val: 1, test: 0 });
    let page: ImagePage = ok(&app3, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    assert_eq!(page.total, 1);

    // Unregistering removes FOLO's caches, never the dataset's files.
    let (status, _, _) = send(&app3, delete(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!cache.exists());
    assert!(data.join("images/train/000000.png").is_file());
    let list: DatasetList = ok(&app(&home), get("/api/datasets")).await;
    assert!(list.datasets.is_empty());
    let leftovers: Vec<_> = std::fs::read_dir(&home)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "atomic writes leave no temporary files");
}

#[tokio::test]
async fn coco_folders_and_registration_errors() {
    let t = TempDir::new("coco");
    let home = t.join("home");
    let yolo = t.join("yolo");
    write_shapes(&yolo, 8);
    let loaded = folo_data::open(&yolo).unwrap();
    folo_data::convert::convert(&loaded, &t.join("coco"), folo_data::Format::Coco).unwrap();
    let app = app(&home);
    let a: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": yolo.to_str().unwrap() }))).await;
    let body = json!({ "path": t.join("coco").to_str().unwrap(), "format": "coco" });
    let b: Dataset = ok(&app, post_json("/api/datasets", body)).await;
    assert_eq!(b.format, DatasetFormat::Coco);
    assert_eq!(b.name, "coco");
    assert_eq!((a.classes.clone(), a.splits), (b.classes.clone(), b.splits));
    let (sa, sb) = (a.stats.unwrap(), b.stats.unwrap());
    assert_eq!((sa.boxes, sa.class_counts), (sb.boxes, sb.class_counts));
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?limit=1", b.id))).await;
    assert!(page.images[0].file.starts_with("images/train/"), "{}", page.images[0].file);

    let missing = json!({ "path": t.join("nope").to_str().unwrap() });
    err(&app, post_json("/api/datasets", missing), StatusCode::NOT_FOUND, "path_not_found").await;
    std::fs::create_dir_all(t.join("empty")).unwrap();
    let empty = json!({ "path": t.join("empty").to_str().unwrap() });
    err(&app, post_json("/api/datasets", empty), StatusCode::BAD_REQUEST, "unknown_format").await;
    std::fs::write(yolo.join("data.yaml"), "train: images/train\nnc: 3\nnames: [a]\n").unwrap();
    let bad = json!({ "path": yolo.to_str().unwrap() });
    err(&app, post_json("/api/datasets", bad), StatusCode::BAD_REQUEST, "invalid_dataset").await;
    let list: DatasetList = ok(&app, get("/api/datasets")).await;
    assert_eq!(list.datasets.len(), 2, "failed registrations are not recorded");
}

#[tokio::test]
async fn external_split_sources_are_watched() {
    let t = TempDir::new("external");
    let (home, shared, ds_dir) = (t.join("home"), t.join("shared"), t.join("ds"));
    write_shapes(&shared, 10);
    std::fs::create_dir_all(&ds_dir).unwrap();
    // The registered folder holds only the YAML; images and labels live elsewhere.
    let yaml = serde_json::to_string(shared.join("images/train").to_str().unwrap()).unwrap();
    std::fs::write(
        ds_dir.join("data.yaml"),
        format!("train: {yaml}\nnames: [circle, square, triangle, star, ring, cross]\n"),
    )
    .unwrap();
    let app = app(&home);
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": ds_dir.to_str().unwrap() }))).await;
    assert_eq!(ds.splits.train, 8);
    let boxes = ds.stats.unwrap().boxes;
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?limit=1", ds.id))).await;
    assert!(Path::new(&page.images[0].file).is_absolute(), "outside the folder: {}", page.images[0].file);
    let (status, _, _) = send(&app, get(&format!("/api/datasets/{}/images/0?thumb=32", ds.id))).await;
    assert_eq!(status, StatusCode::OK);

    // Editing an external label re-indexes, live and across a restart.
    let label = shared.join("labels/train/000000.txt");
    let mut text = std::fs::read_to_string(&label).unwrap();
    text.push_str("1 0.5 0.5 0.111 0.222\n");
    std::fs::write(&label, &text).unwrap();
    let got: Dataset = ok(&app, get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.stats.unwrap().boxes, boxes + 1);
    text.push_str("1 0.4 0.4 0.111 0.222\n");
    std::fs::write(&label, &text).unwrap();
    let got: Dataset = ok(&restart(&home), get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.stats.unwrap().boxes, boxes + 2);
}

#[tokio::test]
async fn relative_paths_are_registered_absolute() {
    // Under the working directory (the package folder), so a relative path to it exists.
    let name = format!(".folo-rel-test-{}", std::process::id());
    let local = TempDir(std::env::current_dir().unwrap().join(&name));
    let _ = std::fs::remove_dir_all(&local.0);
    let yolo = local.join("yolo");
    write_shapes(&yolo, 6);
    let loaded = folo_data::open(&yolo).unwrap();
    folo_data::convert::convert(&loaded, &local.join("coco"), folo_data::Format::Coco).unwrap();
    let t = TempDir::new("relative");
    let app = app(&t.join("home"));
    let rel = format!("./{name}/coco/annotations/instances_train.json");
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": rel }))).await;
    assert!(Path::new(&ds.path).is_absolute(), "{}", ds.path);
    assert_eq!((ds.format, ds.splits.train), (DatasetFormat::Coco, loaded.split_sizes()[0] as u64));
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?limit=100", ds.id))).await;
    for im in &page.images {
        let (status, _, b) =
            send(&app, get(&format!("/api/datasets/{}/images/{}?thumb=16", ds.id, im.id))).await;
        assert_eq!(status, StatusCode::OK, "{}: {}", im.file, String::from_utf8_lossy(&b));
    }
}

#[tokio::test]
async fn lists_recursive_roots_and_rejected_images_are_watched() {
    let t = TempDir::new("sources");
    let (home, shared, ds_dir) = (t.join("home"), t.join("shared"), t.join("ds"));
    write_shapes(&shared, 10);
    std::fs::create_dir_all(&ds_dir).unwrap();
    let q = |p: PathBuf| serde_json::to_string(p.to_str().unwrap()).unwrap();
    // train: an image list outside the folder; val: a recursive root outside it.
    let list = shared.join("train.txt");
    let first: Vec<String> = (0..4)
        .map(|i| shared.join(format!("images/train/{i:06}.png")).to_string_lossy().into_owned())
        .collect();
    std::fs::write(&list, first.join("\n")).unwrap();
    let yaml = format!(
        "train: {}\nval: {}\nnames: [circle, square, triangle, star, ring, cross]\n",
        q(list.clone()),
        q(shared.join("images/val"))
    );
    std::fs::write(ds_dir.join("data.yaml"), yaml).unwrap();
    let app = app(&home);
    let id =
        ok::<Dataset>(&app, post_json("/api/datasets", json!({ "path": ds_dir.to_str().unwrap() }))).await.id;
    let splits = |d: Dataset| d.splits;
    let get_ds = || get(&format!("/api/datasets/{id}"));
    assert_eq!(splits(ok(&app, get_ds()).await), Splits { train: 4, val: 2, test: 0 });

    // A list gains a line.
    let mut more = first.clone();
    more.push(shared.join("images/train/000004.png").to_string_lossy().into_owned());
    std::fs::write(&list, more.join("\n")).unwrap();
    assert_eq!(splits(ok(&app, get_ds()).await).train, 5);

    // A new image in a new sub-folder of the recursive root.
    std::fs::create_dir_all(shared.join("images/val/extra")).unwrap();
    std::fs::copy(shared.join("images/train/000005.png"), shared.join("images/val/extra/x.png")).unwrap();
    assert_eq!(splits(ok(&app, get_ds()).await).val, 3);

    // A folder holding only a corrupt image (skipped by the loader), repaired later.
    std::fs::create_dir_all(shared.join("images/val/broken")).unwrap();
    std::fs::write(shared.join("images/val/broken/y.png"), b"not a png").unwrap();
    assert_eq!(splits(ok(&app, get_ds()).await).val, 3);
    std::fs::copy(shared.join("images/train/000006.png"), shared.join("images/val/broken/y.png")).unwrap();
    assert_eq!(splits(ok(&app, get_ds()).await).val, 4);
    // ... and the same holds across a restart.
    std::fs::remove_file(shared.join("images/val/broken/y.png")).unwrap();
    assert_eq!(splits(ok(&restart(&home), get_ds()).await).val, 3);
}

#[tokio::test]
async fn a_yaml_root_created_later_is_picked_up() {
    let t = TempDir::new("later");
    let (home, ds_dir, later) = (t.join("home"), t.join("ds"), t.join("later"));
    write_shapes(&ds_dir, 5);
    let names = "names: [circle, square, triangle, star, ring, cross]";
    let yaml = format!(
        "path: {}\ntrain: images/train\nval: images/val\n{names}\n",
        serde_json::to_string(later.to_str().unwrap()).unwrap()
    );
    std::fs::write(ds_dir.join("data.yaml"), yaml).unwrap();
    let app = app(&home);
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": ds_dir.to_str().unwrap() }))).await;
    assert_eq!(ds.splits, Splits { train: 4, val: 1, test: 0 }, "falls back to the YAML's folder");
    write_shapes(&later, 10);
    let got: Dataset = ok(&app, get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.splits, Splits { train: 8, val: 2, test: 0 });
    std::fs::remove_dir_all(&later).unwrap();
    let got: Dataset = ok(&restart(&home), get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.splits, Splits { train: 4, val: 1, test: 0 }, "and back, across a restart");
}

#[tokio::test]
async fn files_are_relative_to_the_dataset_root() {
    let t = TempDir::new("root");
    let yolo = t.join("yolo");
    write_shapes(&yolo, 6);
    let loaded = folo_data::open(&yolo).unwrap();
    let coco = t.join("coco");
    folo_data::convert::convert(&loaded, &coco, folo_data::Format::Coco).unwrap();
    std::fs::remove_file(coco.join("images/train/000000.png")).unwrap();
    let app = app(&t.join("home"));
    let json_path = coco.join("annotations/instances_train.json");
    let ds: Dataset =
        ok(&app, post_json("/api/datasets", json!({ "path": json_path.to_str().unwrap() }))).await;
    let page: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    assert!(page.images.iter().all(|i| i.file.starts_with("images/train/")), "{:?}", page.images[0].file);
    let issues: IssuePage =
        ok(&app, get(&format!("/api/datasets/{}/issues?kind=missing_image", ds.id))).await;
    assert_eq!(issues.issues[0].file, "images/train/000000.png");
    // The same object `folo dataset check --json` reports.
    let report = folo_data::check::check(&json_path, &Default::default()).unwrap();
    let cli = report.issues.iter().find(|i| i.kind == folo_data::check::CheckKind::MissingImage).unwrap();
    assert_eq!(folo_data::paths::relative_to(&cli.file, &report.root), issues.issues[0].file);
}

#[tokio::test]
async fn a_data_dir_inside_the_dataset_keeps_its_cache() {
    let t = TempDir::new("inside");
    let data = t.join("ds");
    write_shapes(&data, 6);
    let home = data.join(".folo");
    let a = app(&home);
    let ds: Dataset = ok(&a, post_json("/api/datasets", json!({ "path": data.to_str().unwrap() }))).await;
    let (status, _, _) = send(&a, get(&format!("/api/datasets/{}/images/0?thumb=32", ds.id))).await;
    assert_eq!(status, StatusCode::OK);
    let _: Dataset = ok(&a, get(&format!("/api/datasets/{}", ds.id))).await;
    drop(a);
    let cache = home.join("cache/datasets").join(format!("{}.json", ds.id));
    let stamp = std::fs::metadata(&cache).unwrap().modified().unwrap();
    let got: Dataset = ok(&restart(&home), get(&format!("/api/datasets/{}", ds.id))).await;
    assert_eq!(got.splits, ds.splits);
    assert_eq!(std::fs::metadata(&cache).unwrap().modified().unwrap(), stamp, "cache reused, not rebuilt");
}

#[tokio::test]
async fn an_empty_root_folder_created_later_counts_as_a_change() {
    let t = TempDir::new("empty-root");
    let (home, ds_dir) = (t.join("home"), t.join("ds"));
    write_shapes(&ds_dir, 5);
    let yaml = "path: later\ntrain: images/train\nval: images/val\nnames: [circle, square, triangle, star, ring, cross]\n";
    std::fs::write(ds_dir.join("data.yaml"), yaml).unwrap();
    let app = app(&home);
    let ds: Dataset = ok(&app, post_json("/api/datasets", json!({ "path": ds_dir.to_str().unwrap() }))).await;
    assert_eq!(ds.splits.train, 4);
    let first_file = |page: ImagePage| page.images[0].file.clone();
    let uri = format!("/api/datasets/{}/images?limit=1", ds.id);
    assert_eq!(first_file(ok(&app, get(&uri)).await), "images/train/000000.png");
    // The YAML's `path:` now exists (empty): folo-data takes it as the root (split folders still resolve next to
    // the YAML), so image files are named from the new root.
    std::fs::create_dir_all(ds_dir.join("later")).unwrap();
    let moved = first_file(ok(&app, get(&uri)).await);
    assert!(Path::new(&moved).is_absolute(), "re-indexed with the new root: {moved}");
    assert_eq!(first_file(ok(&restart(&home), get(&uri)).await), moved, "also after a restart");
    // Removing it again restores the old root.
    std::fs::remove_dir(ds_dir.join("later")).unwrap();
    assert_eq!(first_file(ok(&app, get(&uri)).await), "images/train/000000.png");
}
