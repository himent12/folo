//! `/api/uploads` on the real store: a packed dataset sent in parts registers like its source, the part protocol
//! (retries, order, a stream completed early), hostile archives, model uploads, fingerprints (also after a
//! restart), expiry and the cleanup of session folders.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_nn::io::ModelConfig;
use folo_server::api::*;
use folo_server::{MockStore, Options, RealStore, Server};
use folo_test_support::TestDir;
use folo_vision::FoloModel;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app(store: RealStore) -> Router {
    Server::new(Arc::new(store), Options::default()).router()
}

fn open(home: &Path) -> RealStore {
    RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO).device("cpu")
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, body.collect().await.unwrap().to_bytes().to_vec())
}

fn post_json(uri: &str, v: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

fn put(uri: &str, bytes: Vec<u8>) -> Request<Body> {
    Request::put(uri)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from(bytes))
        .unwrap()
}

fn bare(method: &str, uri: &str) -> Request<Body> {
    Request::builder().method(method).uri(uri).body(Body::empty()).unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>, want: StatusCode) -> T {
    let uri = req.uri().to_string();
    let (status, body) = send(app, req).await;
    assert_eq!(status, want, "{uri}: {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().to_string();
    let (got, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// Starts an upload session.
async fn start(app: &Router, body: Value) -> Upload {
    let created: UploadCreated = ok(app, post_json("/api/uploads", body), StatusCode::CREATED).await;
    assert!(created.existing.is_none());
    created.upload.unwrap()
}

/// Sends `bytes` as parts of `size` bytes from part `from` on.
async fn send_parts(app: &Router, id: &str, bytes: &[u8], size: usize, from: u32) -> Upload {
    let mut last = None;
    for (i, chunk) in bytes.chunks(size).enumerate() {
        let uri = format!("/api/uploads/{id}/parts/{}", from + i as u32);
        last = Some(ok::<Upload>(app, put(&uri, chunk.to_vec()), StatusCode::OK).await);
    }
    last.unwrap()
}

/// A synthetic YOLO dataset of 12 images under `dir`, and its packed stream.
fn packed_dataset(dir: &Path) -> Vec<u8> {
    let cfg = folo_data::synthetic::SyntheticConfig { images: 12, imgsz: 64, seed: 4, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, dir, folo_data::ImageFormat::Jpeg(90)).unwrap();
    let ds = folo_data::open(dir).unwrap();
    let mut bytes = Vec::new();
    folo_data::pack::pack(&ds, &mut bytes).unwrap();
    bytes
}

/// Session folders left under `<home>/uploads`.
fn staging(home: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(home.join("uploads")) else { return Vec::new() };
    rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_packed_dataset_uploaded_in_parts_registers_like_its_source() {
    let t = TestDir::new("uploads-dataset");
    let home = t.join("home");
    let bytes = packed_dataset(&t.join("src"));
    let app = app(open(&home));
    let source: Dataset = ok(
        &app,
        post_json("/api/datasets", json!({ "path": t.join("src").to_string_lossy() })),
        StatusCode::CREATED,
    )
    .await;

    let u = start(
        &app,
        json!({ "kind": "dataset", "name": "Shapes Up!", "fingerprint": "fp-shapes", "bytes": bytes.len() }),
    )
    .await;
    assert_eq!((u.part, u.received_bytes, u.max_part_bytes), (0, 0, 32 << 20));
    assert!(u.id.starts_with("up_") && staging(&home) == [u.id.clone()], "{:?}", staging(&home));
    let size = bytes.len() / 3 + 1;
    let parts: Vec<&[u8]> = bytes.chunks(size).collect();
    let part = |n: usize| put(&format!("/api/uploads/{}/parts/{n}", u.id), parts[n].to_vec());
    ok::<Upload>(&app, part(0), StatusCode::OK).await;
    let after1: Upload = ok(&app, part(1), StatusCode::OK).await;
    assert_eq!((after1.part, after1.received_bytes), (2, (2 * size) as u64));
    // A retry of the last part (its answer was lost) changes nothing; anything else names the part to send.
    let again: Upload = ok(&app, part(1), StatusCode::OK).await;
    assert_eq!((again.part, again.received_bytes), (2, after1.received_bytes));
    let e = err(
        &app,
        put(&format!("/api/uploads/{}/parts/1", u.id), vec![0; 3]),
        StatusCode::CONFLICT,
        "upload_conflict",
    )
    .await;
    assert!(e.hint.unwrap().contains("send part 2"));
    for n in [0, 3, 7] {
        let req = put(&format!("/api/uploads/{}/parts/{n}", u.id), parts[2].to_vec());
        err(&app, req, StatusCode::CONFLICT, "upload_conflict").await;
    }
    let status: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
    assert_eq!(status.part, 2);
    ok::<Upload>(&app, part(2), StatusCode::OK).await;
    let done: Uploaded =
        ok(&app, bare("POST", &format!("/api/uploads/{}/complete", u.id)), StatusCode::CREATED).await;
    let ds = done.dataset.unwrap();
    assert!(done.model.is_none());
    assert_eq!(ds.name, "Shapes Up!");
    assert!(ds.path.ends_with("/datasets/shapes-up"), "{}", ds.path);
    assert_eq!((&ds.classes, ds.splits), (&source.classes, source.splits));
    assert_eq!(ds.stats, source.stats);
    assert!(staging(&home).is_empty(), "the session folder is gone: {:?}", staging(&home));
    err(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::NOT_FOUND, "upload_not_found")
        .await;
    // The images, in the same order, with the same boxes.
    let images = |id: &str| bare("GET", &format!("/api/datasets/{id}/images?limit=1000"));
    let (a, b): (ImagePage, ImagePage) =
        (ok(&app, images(&source.id), StatusCode::OK).await, ok(&app, images(&ds.id), StatusCode::OK).await);
    let shape = |p: &ImagePage| {
        p.images.iter().map(|i| (i.split, i.width, i.height, i.boxes.clone())).collect::<Vec<_>>()
    };
    assert_eq!(shape(&a), shape(&b));

    // The same fingerprint: what it became, and nothing to send (also after a restart).
    let again = json!({ "kind": "dataset", "name": "other name", "fingerprint": "fp-shapes" });
    let existing: UploadCreated = ok(&app, post_json("/api/uploads", again.clone()), StatusCode::OK).await;
    assert!(existing.upload.is_none());
    assert_eq!(existing.existing.unwrap().dataset.unwrap().id, ds.id);
    drop(app);
    let app = self::app(open(&home));
    let existing: UploadCreated = ok(&app, post_json("/api/uploads", again.clone()), StatusCode::OK).await;
    assert_eq!(existing.existing.unwrap().dataset.unwrap().id, ds.id);
    // A model with that fingerprint is another upload; a dataset no longer registered is sent again.
    start(&app, json!({ "kind": "model", "name": "m", "fingerprint": "fp-shapes" })).await;
    assert_eq!(
        send(&app, bare("DELETE", &format!("/api/datasets/{}", ds.id))).await.0,
        StatusCode::NO_CONTENT
    );
    let fresh = start(&app, again).await;
    assert_eq!(
        send(&app, bare("DELETE", &format!("/api/uploads/{}", fresh.id))).await.0,
        StatusCode::NO_CONTENT
    );
    err(
        &app,
        bare("DELETE", &format!("/api/uploads/{}", fresh.id)),
        StatusCode::NOT_FOUND,
        "upload_not_found",
    )
    .await;
    assert!(!staging(&home).contains(&fresh.id));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stream_completed_inside_a_file_can_still_be_finished() {
    let t = TestDir::new("uploads-truncated");
    let home = t.join("home");
    let bytes = packed_dataset(&t.join("src"));
    let app = app(open(&home));
    let u = start(&app, json!({ "kind": "dataset", "name": "cut" })).await;
    let complete = || bare("POST", &format!("/api/uploads/{}/complete", u.id));
    // Cut inside the first image (after data.yaml's header and text, and the image's header), and after the last
    // label, before the end-of-archive blocks: the archive is not whole, and the session stays open.
    let first_image = 512 + 512 + 512 + 100;
    let mut sent = 0;
    for (cut, at) in [
        (first_image, "inside images/train/0000000.jpg"),
        (bytes.len() - 1024, "after labels/val/0000001.txt"),
    ] {
        let status: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
        send_parts(&app, &u.id, &bytes[sent..cut], 64 * 1024, status.part).await;
        sent = cut;
        let e = err(&app, complete(), StatusCode::CONFLICT, "upload_conflict").await;
        assert!(e.message.contains(at), "{at}: {}", e.message);
        let status: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
        assert_eq!(status.received_bytes, cut as u64);
    }
    let status: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
    send_parts(&app, &u.id, &bytes[sent..], 64 * 1024, status.part).await;
    let done: Uploaded = ok(&app, complete(), StatusCode::CREATED).await;
    let [train, val, _] = folo_data::open(t.join("src")).unwrap().split_sizes();
    let splits = done.dataset.unwrap().splits;
    assert_eq!((splits.train, splits.val), (train as u64, val as u64));
}

/// A tar archive with one entry of type `kind` named `name` (written as raw bytes, so any name goes in).
fn hostile(name: &str, kind: tar::EntryType, link: Option<&str>) -> Vec<u8> {
    let mut h = tar::Header::new_gnu();
    h.as_mut_bytes()[..name.len()].copy_from_slice(name.as_bytes());
    h.set_entry_type(kind);
    h.set_mode(0o644);
    let data: &[u8] = if kind.is_file() { b"pwned" } else { b"" };
    h.set_size(data.len() as u64);
    if let Some(l) = link {
        h.set_link_name_literal(l).unwrap();
    }
    h.set_cksum();
    let mut out = h.as_bytes().to_vec();
    out.extend_from_slice(data);
    out.resize(out.len().div_ceil(512) * 512 + 1024, 0);
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn hostile_archives_are_refused_and_leave_nothing() {
    let t = TestDir::new("uploads-hostile");
    let home = t.join("home");
    let app = app(open(&home));
    let outside = t.join("outside.txt");
    let absolute = outside.to_string_lossy().replace('\\', "/");
    let cases = [
        ("../outside.txt".to_string(), tar::EntryType::Regular, None, "relative paths"),
        ("images/../../outside.txt".to_string(), tar::EntryType::Regular, None, "relative paths"),
        (absolute, tar::EntryType::Regular, None, "relative paths"),
        ("/etc/folo-test".to_string(), tar::EntryType::Regular, None, "relative paths"),
        ("data.yaml:stream".to_string(), tar::EntryType::Regular, None, "relative paths"),
        ("link".to_string(), tar::EntryType::Symlink, Some("../outside.txt"), "symbolic link"),
        ("hard".to_string(), tar::EntryType::Link, Some("data.yaml"), "hard link"),
        ("fifo".to_string(), tar::EntryType::Fifo, None, "special file"),
        ("dev".to_string(), tar::EntryType::Char, None, "special file"),
    ];
    for (name, kind, link, why) in cases {
        let u = start(&app, json!({ "kind": "dataset", "name": "evil" })).await;
        let req = put(&format!("/api/uploads/{}/parts/0", u.id), hostile(&name, kind, link));
        let e = err(&app, req, StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload").await;
        assert!(e.message.contains(why), "{name}: {}", e.message);
        // The session is gone, with its folder.
        err(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::NOT_FOUND, "upload_not_found")
            .await;
        assert!(staging(&home).is_empty(), "{name}: {:?}", staging(&home));
    }
    assert!(!outside.exists() && !t.join("hard").exists());
    // A stream that is not a tar archive at all.
    let u = start(&app, json!({ "kind": "dataset", "name": "noise" })).await;
    let noise: Vec<u8> = (0..2048u32).map(|i| (i * 7 + 3) as u8).collect();
    let e = err(
        &app,
        put(&format!("/api/uploads/{}/parts/0", u.id), noise),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_upload",
    )
    .await;
    assert!(e.message.contains("not a readable tar archive"), "{}", e.message);
    assert!(staging(&home).is_empty());
    assert!(
        !home.join("datasets").exists() || std::fs::read_dir(home.join("datasets")).unwrap().next().is_none()
    );
}

/// Entries that differ only by letter case or a trailing dot or space name one file on Windows (and on macOS's
/// default volumes): unpacking both would silently lose an image, so the upload is refused, naming both.
#[tokio::test(flavor = "multi_thread")]
async fn entries_that_would_overwrite_each_other_are_refused() {
    let t = TestDir::new("uploads-collide");
    let home = t.join("home");
    let app = app(open(&home));
    let cases = [
        ("images/train/A.jpg", "images/train/a.jpg"),
        ("images/train/b.jpg", "images/train/b.jpg."),
        ("images/train/c.jpg", "images/train/c.jpg "),
        ("Images/train/d.jpg", "images/Train/D.JPG"),
        ("labels/train/e.txt", "labels/train./e.txt"),
    ];
    for (first, second) in cases {
        let mut b = tar::Builder::new(Vec::new());
        for (name, data) in [("data.yaml", &b"names: [x]\n"[..]), (first, b"one"), (second, b"two")] {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            b.append_data(&mut h, name, data).unwrap();
        }
        let bytes = b.into_inner().unwrap();
        let u = start(&app, json!({ "kind": "dataset", "name": "collide" })).await;
        let req = put(&format!("/api/uploads/{}/parts/0", u.id), bytes);
        let e = err(&app, req, StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload").await;
        assert!(e.message.contains(first) && e.message.contains(second.trim_end()), "{}", e.message);
        assert!(staging(&home).is_empty(), "{first}: {:?}", staging(&home));
    }
}

/// An archive with a file and a folder of one name is refused like the other hostile archives (422, naming it),
/// not answered with the operating system's error as `500 io_error` (x3-server-5).
#[tokio::test(flavor = "multi_thread")]
async fn a_file_and_a_folder_of_one_name_are_refused() {
    let t = TestDir::new("uploads-file-folder");
    let home = t.join("home");
    let app = app(open(&home));
    let (dir, file) = (tar::EntryType::Directory, tar::EntryType::Regular);
    let cases = [
        ([("images/", dir), ("images", file)], "images"),
        ([("x", file), ("x/y.txt", file)], "x"),
        ([("labels/a.txt", file), ("labels/a.txt/b", dir)], "labels/a.txt"),
    ];
    for (entries, clash) in cases {
        let mut b = tar::Builder::new(Vec::new());
        for (name, kind) in entries {
            let data: &[u8] = if kind.is_dir() { b"" } else { b"one" };
            let mut h = tar::Header::new_gnu();
            h.set_entry_type(kind);
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            b.append_data(&mut h, name, data).unwrap();
        }
        let u = start(&app, json!({ "kind": "dataset", "name": "clash" })).await;
        let req = put(&format!("/api/uploads/{}/parts/0", u.id), b.into_inner().unwrap());
        let e = err(&app, req, StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload").await;
        assert!(e.message.starts_with(&format!("{clash} is both a file and a folder")), "{}", e.message);
        assert!(staging(&home).is_empty(), "{clash}: {:?}", staging(&home));
    }
}

/// An archive that wraps the dataset in a folder (a zip of the dataset folder) is not registered, and the folder it
/// was unpacked into is removed: the error says to pack the folder's contents at the top and upload again, not to
/// register a folder that is gone (x3-server-6, review round 1).
#[tokio::test(flavor = "multi_thread")]
async fn an_archive_wrapping_the_dataset_in_a_folder_says_how_to_pack_it() {
    let t = TestDir::new("uploads-wrapped");
    let home = t.join("home");
    let app = app(open(&home));
    let mut b = tar::Builder::new(Vec::new());
    for (name, data) in [("my-set/data.yaml", &b"names: [x]\n"[..]), ("my-set/images/train/a.jpg", b"jpg")] {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        b.append_data(&mut h, name, data).unwrap();
    }
    let u = start(&app, json!({ "kind": "dataset", "name": "wrapped" })).await;
    ok::<Upload>(
        &app,
        put(&format!("/api/uploads/{}/parts/0", u.id), b.into_inner().unwrap()),
        StatusCode::OK,
    )
    .await;
    let complete = bare("POST", &format!("/api/uploads/{}/complete", u.id));
    let e = err(&app, complete, StatusCode::BAD_REQUEST, "unknown_format").await;
    assert!(e.message.contains("its folder 'my-set' holds one"), "{}", e.message);
    let hint = e.hint.unwrap();
    assert!(hint.contains("top level") && hint.contains("upload it again"), "{hint}");
    let left: Vec<_> = std::fs::read_dir(home.join("datasets")).into_iter().flatten().flatten().collect();
    assert!(left.is_empty(), "nothing stays in datasets/: {left:?}");
}

/// Writes a folo-n weights file with two classes.
fn model_bytes(dir: &Path) -> Vec<u8> {
    let path = dir.join("pets.folo");
    let mut m =
        FoloModel::with_rng(ModelConfig::new("n", 2).unwrap(), &mut folo_core::rng::Rng::seed(3)).unwrap();
    m.set_classes(vec!["cat".into(), "dog".into()]).unwrap();
    m.set_imgsz(320);
    m.save(&path).unwrap();
    std::fs::read(path).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn models_upload_into_the_models_folder() {
    let t = TestDir::new("uploads-model");
    let home = t.join("home");
    let bytes = model_bytes(&t);
    let app = app(open(&home));

    let u = start(&app, json!({ "kind": "model", "name": "pets n/v2", "fingerprint": "fp-model" })).await;
    send_parts(&app, &u.id, &bytes, 3 << 20, 0).await;
    let done: Uploaded =
        ok(&app, bare("POST", &format!("/api/uploads/{}/complete", u.id)), StatusCode::CREATED).await;
    let m = done.model.unwrap();
    assert_eq!((m.name.as_str(), m.source), ("pets_n_v2", Some(ModelSource::File)));
    assert_eq!(m.classes, ["cat", "dog"]);
    assert_eq!(std::fs::read(home.join("models").join("pets_n_v2.folo")).unwrap(), bytes);
    let list: ModelList = ok(&app, bare("GET", "/api/models"), StatusCode::OK).await;
    assert!(list.models.iter().any(|x| x.id == m.id));
    let existing: UploadCreated = ok(
        &app,
        post_json("/api/uploads", json!({ "kind": "model", "name": "x", "fingerprint": "fp-model" })),
        StatusCode::OK,
    )
    .await;
    assert_eq!(existing.existing.unwrap().model.unwrap().id, m.id);
    // The same name again: `-2`.
    let u = start(&app, json!({ "kind": "model", "name": "pets n/v2" })).await;
    send_parts(&app, &u.id, &bytes, 32 << 20, 0).await;
    let done: Uploaded =
        ok(&app, bare("POST", &format!("/api/uploads/{}/complete", u.id)), StatusCode::CREATED).await;
    assert_eq!(done.model.unwrap().name, "pets_n_v2-2");
    // A file that took the next name meanwhile keeps its bytes, and the upload moves on to `-4`.
    let models = home.join("models");
    std::fs::write(models.join("pets_n_v2-3.folo"), b"someone else's").unwrap();
    let u = start(&app, json!({ "kind": "model", "name": "pets n/v2" })).await;
    send_parts(&app, &u.id, &bytes, 32 << 20, 0).await;
    let done: Uploaded =
        ok(&app, bare("POST", &format!("/api/uploads/{}/complete", u.id)), StatusCode::CREATED).await;
    assert_eq!(done.model.unwrap().name, "pets_n_v2-4");
    assert_eq!(std::fs::read(models.join("pets_n_v2-3.folo")).unwrap(), b"someone else's");
    // The upload is moved to its name in one step: no name is claimed with an empty file first.
    for e in std::fs::read_dir(&models).unwrap() {
        let e = e.unwrap();
        assert!(e.metadata().unwrap().len() > 0, "{:?} is empty", e.file_name());
    }
    std::fs::remove_file(models.join("pets_n_v2-3.folo")).unwrap();

    // Not a .folo file: refused on completion, nothing listed, the session gone.
    let u = start(&app, json!({ "kind": "model", "name": "junk" })).await;
    send_parts(&app, &u.id, b"definitely not safetensors", 1024, 0).await;
    let e = err(
        &app,
        bare("POST", &format!("/api/uploads/{}/complete", u.id)),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_weights",
    )
    .await;
    assert!(e.message.contains("not a readable .folo file"), "{}", e.message);
    assert!(!home.join("models").join("junk.folo").exists());
    assert!(staging(&home).is_empty());
}

/// An upload's name is any text, a long one too: the stream still becomes a file in the models folder.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_with_a_long_name_uploads() {
    let t = TestDir::new("uploads-model-long");
    let home = t.join("home");
    let bytes = model_bytes(&t);
    let app = app(open(&home));
    let u = start(&app, json!({ "kind": "model", "name": "pets".repeat(80) })).await;
    send_parts(&app, &u.id, &bytes, 32 << 20, 0).await;
    let done: Uploaded =
        ok(&app, bare("POST", &format!("/api/uploads/{}/complete", u.id)), StatusCode::CREATED).await;
    let m = done.model.unwrap();
    assert!(m.name.starts_with("petspets") && m.name.len() < 255, "{}", m.name);
    assert!(staging(&home).is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_expiry_and_leftovers() {
    let t = TestDir::new("uploads-misc");
    let home = t.join("home");
    // A session folder a previous process left behind is removed on start.
    std::fs::create_dir_all(home.join("uploads").join("up_dead")).unwrap();
    std::fs::write(home.join("uploads").join("up_dead").join("data.yaml"), "names: [a]").unwrap();
    let app = app(open(&home).upload_idle(Duration::from_millis(300)));
    assert!(staging(&home).is_empty());

    for (body, why) in [
        (json!({ "kind": "dataset", "name": "  " }), "name"),
        (json!({ "kind": "model", "name": "m", "format": "yolo" }), "format"),
        (json!({ "kind": "model", "name": "m", "fingerprint": "f".repeat(129) }), "fingerprint"),
    ] {
        let e = err(&app, post_json("/api/uploads", body), StatusCode::BAD_REQUEST, "invalid_body").await;
        assert!(e.message.contains(why), "{}", e.message);
    }
    err(
        &app,
        post_json("/api/uploads", json!({ "kind": "folder", "name": "x" })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    for (method, uri) in [
        ("GET", "/api/uploads/up_none"),
        ("POST", "/api/uploads/up_none/complete"),
        ("DELETE", "/api/uploads/up_none"),
    ] {
        err(&app, bare(method, uri), StatusCode::NOT_FOUND, "upload_not_found").await;
    }
    err(&app, put("/api/uploads/up_none/parts/0", vec![1]), StatusCode::NOT_FOUND, "upload_not_found").await;

    // A part over the limit.
    let u = start(&app, json!({ "kind": "model", "name": "big" })).await;
    let e = err(
        &app,
        put(&format!("/api/uploads/{}/parts/0", u.id), vec![0; (32 << 20) + 1]),
        StatusCode::PAYLOAD_TOO_LARGE,
        "payload_too_large",
    )
    .await;
    assert!(e.hint.unwrap().contains("max_part_bytes"));

    // A session without a part for the idle limit expires, with its folder.
    let u = start(&app, json!({ "kind": "dataset", "name": "idle" })).await;
    assert!(staging(&home).contains(&u.id));
    let t0 = std::time::Instant::now();
    while !staging(&home).is_empty() {
        assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", staging(&home));
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    err(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::NOT_FOUND, "upload_not_found")
        .await;
}

#[tokio::test]
async fn the_mock_has_no_uploads() {
    let app = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let e = err(
        &app,
        post_json("/api/uploads", json!({ "kind": "dataset", "name": "x" })),
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented",
    )
    .await;
    assert!(e.hint.unwrap().contains("--mock"));
    let sys: SystemResponse = ok(&app, bare("GET", "/api/system"), StatusCode::OK).await;
    assert!(sys.features.iter().any(|f| f == FEATURE_RUN_CURSORS));
    assert!(!sys.features.iter().any(|f| f == FEATURE_UPLOADS || f == FEATURE_RUN_CHECKPOINT));
}

/// On Windows, tar entries that fail with `500 io_error` or write to `\Device\Null` (`< > " | ? *`, ASCII control
/// chars, dot/space-only components, bare `NUL` with optional trailing dots/spaces, and parent folders ending in
/// spaces or multiple dots/spaces) are refused with `422 invalid_upload`, while names that unpack cleanly on
/// Windows 11 (`CON.png`, `NUL.png`, `aux/`, bare `CON`, files with a trailing dot/space, `foo./a.png`) stay
/// accepted. On non-Windows OSes, all of these names stay accepted.
#[tokio::test(flavor = "multi_thread")]
async fn entries_with_windows_unsafe_names_are_refused() {
    let t = TestDir::new("uploads-windows-unsafe");
    let home = t.join("home");
    let app = app(open(&home));
    let pack = |entries: &[(&str, &[u8])]| {
        let mut b = tar::Builder::new(Vec::new());
        for &(entry, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            b.append_data(&mut h, entry, data).unwrap();
        }
        b.into_inner().unwrap()
    };

    let refused_on_windows = [
        "images/train/a?b.png",
        "images/train/a*b.png",
        "images/train/a<b.png",
        "images/train/a>b.png",
        "images/train/a\"b.png",
        "images/train/a|b.png",
        "images/train/Icon\r",
        "images/train/a\tb.png",
        "images/train/NUL",
        "images/train/nul.",
        "images/train/NUL ",
        "NUL/a.png",
        "NUL./a.png",
        "images/train/...",
        "images/train/   ",
        ".../a.png",
        "foo /a.png",
        "foo../a.png",
        "foo ./a.png",
    ];
    for name in refused_on_windows {
        let bytes = pack(&[("data.yaml", b"names: [x]\n"), (name, b"one")]);
        let u = start(&app, json!({ "kind": "dataset", "name": "unsafe" })).await;
        let req = put(&format!("/api/uploads/{}/parts/0", u.id), bytes);
        if cfg!(windows) {
            err(&app, req, StatusCode::UNPROCESSABLE_ENTITY, "invalid_upload").await;
            err(
                &app,
                bare("GET", &format!("/api/uploads/{}", u.id)),
                StatusCode::NOT_FOUND,
                "upload_not_found",
            )
            .await;
            assert!(staging(&home).is_empty(), "{name}: {:?}", staging(&home));
        } else {
            ok::<Upload>(&app, req, StatusCode::OK).await;
            let (status, _) = send(&app, bare("DELETE", &format!("/api/uploads/{}", u.id))).await;
            assert_eq!(status, StatusCode::NO_CONTENT, "{name}");
        }
    }

    // Names that unpack cleanly on Windows 11 (and every other OS) stay accepted on every OS.
    let accepted_everywhere = [
        "images/train/CON.png",
        "images/train/NUL.png",
        "images/train/aux/a.png",
        "images/train/CON",
        "images/train/aux .txt",
        "com1/a.png",
        "images/train/foo.",
        "images/train/bar ",
        "images/train/dir./a.png",
    ];
    let mut entries: Vec<(&str, &[u8])> = vec![("data.yaml", b"names: [x]\n")];
    for name in accepted_everywhere {
        entries.push((name, b"one"));
    }
    let u = start(&app, json!({ "kind": "dataset", "name": "safe-names" })).await;
    ok::<Upload>(&app, put(&format!("/api/uploads/{}/parts/0", u.id), pack(&entries)), StatusCode::OK).await;
    let stage = home.join("uploads").join(&u.id);
    for expected in [
        "images/train/CON.png",
        "images/train/NUL.png",
        "images/train/aux/a.png",
        "images/train/CON",
        "com1/a.png",
    ] {
        assert_eq!(std::fs::read(stage.join(expected)).unwrap(), b"one", "{expected}");
    }
    let (status, _) = send(&app, bare("DELETE", &format!("/api/uploads/{}", u.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(staging(&home).is_empty());
}

/// Legacy persisted fingerprints (including whitespace-only or control-character keys recorded by older versions)
/// remain reachable on retry via `POST /api/uploads`, while new sessions still reject invalid fingerprints.
#[tokio::test(flavor = "multi_thread")]
async fn legacy_persisted_fingerprints_remain_reachable_on_create_upload() {
    let t = TestDir::new("uploads-legacy-fp");
    let home = t.join("home");
    let bytes = model_bytes(&t);
    std::fs::create_dir_all(home.join("models")).unwrap();
    std::fs::write(home.join("models").join("legacy.folo"), &bytes).unwrap();
    let first = app(open(&home));
    let models: ModelList = ok(&first, bare("GET", "/api/models"), StatusCode::OK).await;
    let model_id = models.models[0].id.clone();

    // Persist legacy whitespace-only and control-char keys directly in `uploads.json`, then reopen.
    std::fs::write(
        home.join("uploads.json"),
        json!([
            { "kind": "model", "fingerprint": "   ", "id": model_id },
            { "kind": "model", "fingerprint": "legacy\nfp", "id": model_id }
        ])
        .to_string(),
    )
    .unwrap();
    let reopened = app(open(&home));

    for fp in ["   ", "legacy\nfp"] {
        let res: UploadCreated = ok(
            &reopened,
            post_json("/api/uploads", json!({ "kind": "model", "name": "retry", "fingerprint": fp })),
            StatusCode::OK,
        )
        .await;
        assert_eq!(res.existing.unwrap().model.unwrap().id, model_id, "{fp:?}");
    }
    // An unknown invalid fingerprint still fails with 400 invalid_body instead of opening a new session.
    err(
        &reopened,
        post_json("/api/uploads", json!({ "kind": "model", "name": "new", "fingerprint": "\t" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn upload_parts_require_octet_stream_media_type() {
    let t = TestDir::new("uploads-media-type");
    let app = app(open(&t.join("home")));
    let u = start(&app, json!({ "kind": "model", "name": "media-type" })).await;
    let uri = format!("/api/uploads/{}/parts/0", u.id);

    // Wrong Content-Type -> 415 unsupported_media_type and session does not advance.
    let wrong = Request::put(&uri)
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(b"x".to_vec()))
        .unwrap();
    err(&app, wrong, StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type").await;
    let after_wrong: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
    assert_eq!((after_wrong.part, after_wrong.received_bytes), (0, 0));

    // Missing Content-Type -> 415 unsupported_media_type and session still does not advance.
    let missing = Request::put(&uri).body(Body::from(b"x".to_vec())).unwrap();
    err(&app, missing, StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type").await;
    let after_missing: Upload =
        ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
    assert_eq!((after_missing.part, after_missing.received_bytes), (0, 0));

    // Malformed parameter (unterminated quote) and duplicate conflicting headers -> 415 and no state advance.
    let bad_param = Request::put(&uri)
        .header(header::CONTENT_TYPE, "application/octet-stream; charset=\"binary")
        .body(Body::from(b"x".to_vec()))
        .unwrap();
    err(&app, bad_param, StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type").await;

    let dup_headers = Request::put(&uri)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(b"x".to_vec()))
        .unwrap();
    err(&app, dup_headers, StatusCode::UNSUPPORTED_MEDIA_TYPE, "unsupported_media_type").await;
    let after_bad: Upload = ok(&app, bare("GET", &format!("/api/uploads/{}", u.id)), StatusCode::OK).await;
    assert_eq!((after_bad.part, after_bad.received_bytes), (0, 0));

    // Case-insensitive media type with valid RFC 9110 parameters (including quoted semicolon) is accepted.
    let valid = Request::put(&uri)
        .header(header::CONTENT_TYPE, "Application/Octet-Stream; charset=binary; name=\"a;b\"")
        .body(Body::from(b"x".to_vec()))
        .unwrap();
    let accepted: Upload = ok(&app, valid, StatusCode::OK).await;
    assert_eq!((accepted.part, accepted.received_bytes), (1, 1));
}
