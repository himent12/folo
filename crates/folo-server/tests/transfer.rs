//! `folo_client::transfer` against a real server on a loopback port: uploads in parts through a proxy that loses
//! a request and an answer (nothing is sent twice or lost), fingerprints, stops, and downloads in `Range` parts,
//! also of a file replaced while it is fetched.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::Request;
use axum::http::{Method, StatusCode};
use axum::middleware::Next;
use axum::response::IntoResponse;
use folo_client::Client;
use folo_client::transfer::{Control, TransferError, download, upload};
use folo_nn::io::ModelConfig;
use folo_server::api::*;
use folo_server::{Options, RealStore, Server, Token};
use folo_test_support::TestDir;
use folo_vision::FoloModel;

const TOKEN: &str = "worker-token";

/// A router served on a loopback port by its own runtime, until dropped.
struct Live {
    url: String,
    rt: Option<tokio::runtime::Runtime>,
}

impl Live {
    fn start(router: Router) -> Live {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        rt.spawn(async move { axum::serve(listener, router).await.unwrap() });
        Live { url, rt: Some(rt) }
    }

    fn client(&self) -> Client {
        Client::new(&self.url).token(TOKEN)
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}

/// The real store on `home` behind a token.
fn server(home: &Path) -> Router {
    let store = RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO).device("cpu");
    Server::new(Arc::new(store), Options { token: Token::new(TOKEN), ..Options::default() }).router()
}

/// `router` behind a proxy that fails the `PUT`s with these 0-based numbers like a proxy does (`502`): `lost` never
/// reaches the server, `unanswered` does but its answer is lost.
fn lossy(router: Router, lost: usize, unanswered: usize) -> Router {
    let puts = Arc::new(AtomicUsize::new(0));
    router.layer(axum::middleware::from_fn(move |req: Request, next: Next| {
        let puts = puts.clone();
        async move {
            if req.method() != Method::PUT {
                return next.run(req).await;
            }
            let n = puts.fetch_add(1, Ordering::SeqCst);
            if n == lost {
                return (StatusCode::BAD_GATEWAY, "proxy: no upstream").into_response();
            }
            let res = next.run(req).await;
            if n == unanswered {
                return (StatusCode::BAD_GATEWAY, "proxy: upstream reset").into_response();
            }
            res
        }
    }))
}

/// Writes a folo-n weights file (about 10 MB: two 8 MiB parts) initialised from `seed`.
fn write_model(path: &Path, seed: u64) {
    write_sized(path, "n", seed);
}

/// A model of architecture `size` (`n` is about 10 MB, `s` about 38).
fn write_sized(path: &Path, size: &str, seed: u64) {
    let mut m = FoloModel::with_rng(ModelConfig::new(size, 2).unwrap(), &mut folo_core::rng::Rng::seed(seed))
        .unwrap();
    m.set_classes(vec!["cat".into(), "dog".into()]).unwrap();
    m.save(path).unwrap();
}

fn never() -> bool {
    false
}

fn staging(home: &Path) -> usize {
    std::fs::read_dir(home.join("uploads")).map_or(0, |rd| rd.count())
}

#[test]
fn uploads_survive_a_lost_request_and_a_lost_answer() {
    let t = TestDir::new("transfer-up");
    let home = t.join("home");
    let file = t.join("pets.folo");
    write_model(&file, 1);
    let bytes = std::fs::read(&file).unwrap();
    assert!(bytes.len() > 8 << 20 && bytes.len() < 16 << 20, "two parts: {}", bytes.len());
    // PUT 0 (part 0) is lost; its retry, PUT 1, arrives but its answer is lost; part 1 goes through.
    let live = Live::start(lossy(server(&home), 0, 1));
    let client = live.client();

    let mut seen = Vec::new();
    let mut progress = |p: TransferProgress| seen.push(p);
    let mut ctl = Control { progress: &mut progress, stop: &never };
    let req = CreateUpload {
        kind: UploadKind::Model,
        name: "pets".into(),
        format: DatasetFormat::Auto,
        fingerprint: Some("sha-pets".into()),
        bytes: Some(bytes.len() as u64),
    };
    let done = upload(&client, &req, |w| w.write_all(&bytes), &mut ctl).unwrap();
    let model = done.model.unwrap();
    assert_eq!(
        std::fs::read(home.join("models").join("pets.folo")).unwrap(),
        bytes,
        "no part twice, none lost"
    );
    let total = bytes.len() as u64;
    assert_eq!(
        seen,
        [(8 << 20, total), (total, total)].map(|(d, t)| TransferProgress { done_bytes: d, total_bytes: t })
    );
    assert_eq!(staging(&home), 0);

    // The same fingerprint: the server already has it, and nothing is produced or sent.
    let mut ctl = Control { progress: &mut |_| panic!("nothing is sent"), stop: &never };
    let again = upload(&client, &req, |_| panic!("not produced"), &mut ctl).unwrap();
    assert_eq!(again.model.unwrap().id, model.id);

    // Without the token: refused before anything is sent.
    let mut ctl = Control { progress: &mut |_| {}, stop: &never };
    let anonymous = Client::new(&live.url);
    match upload(&anonymous, &req, |_| panic!("not produced"), &mut ctl) {
        Err(TransferError::Failed(e)) => assert_eq!(e.code(), "unauthorized"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_dataset_stream_uploads_and_a_stopped_upload_leaves_nothing() {
    let t = TestDir::new("transfer-ds");
    let home = t.join("home");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 10, imgsz: 64, seed: 2, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &t.join("src"), folo_data::ImageFormat::Png).unwrap();
    let ds = folo_data::open(t.join("src")).unwrap();
    let live = Live::start(server(&home));
    let client = live.client();

    let req = CreateUpload {
        kind: UploadKind::Dataset,
        name: "shapes".into(),
        format: DatasetFormat::Yolo,
        fingerprint: None,
        bytes: Some(folo_data::pack::packed_len(&ds).unwrap()),
    };
    let mut ctl = Control { progress: &mut |_| {}, stop: &never };
    let pack = |w: &mut dyn Write| folo_data::pack::pack(&ds, w).map(|_| ()).map_err(std::io::Error::other);
    let done = upload(&client, &req, pack, &mut ctl).unwrap().dataset.unwrap();
    assert_eq!((done.format, done.classes.clone()), (DatasetFormat::Yolo, ds.classes.clone()));
    let [train, val, _] = ds.split_sizes();
    assert_eq!((done.splits.train, done.splits.val), (train as u64, val as u64));

    // Stopped after the first part: the session is deleted with what it received.
    let parts = Arc::new(AtomicBool::new(false));
    let sent = parts.clone();
    let mut progress = move |_: TransferProgress| sent.store(true, Ordering::SeqCst);
    let stop = move || parts.load(Ordering::SeqCst);
    let mut ctl = Control { progress: &mut progress, stop: &stop };
    let big = vec![7u8; (8 << 20) + 10];
    let model = CreateUpload {
        kind: UploadKind::Model,
        name: "big".into(),
        format: DatasetFormat::Auto,
        fingerprint: None,
        bytes: None,
    };
    assert!(matches!(upload(&client, &model, |w| w.write_all(&big), &mut ctl), Err(TransferError::Stopped)));
    assert_eq!(staging(&home), 0);
}

#[test]
fn downloads_in_parts_start_over_when_the_file_is_replaced() {
    let t = TestDir::new("transfer-down");
    let home = t.join("home");
    let path = home.join("models").join("pets.folo");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    write_model(&path, 1);
    let live = Live::start(server(&home));
    let client = live.client();
    let models: ModelList = client.get("/models").unwrap();
    let uri = format!("/models/{}/weights", models.models[0].id);
    let dest = t.join("out").join("pets.folo");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let leftovers = || std::fs::read_dir(dest.parent().unwrap()).unwrap().count();

    let mut parts = 0;
    let mut ctl = Control { progress: &mut |_| parts += 1, stop: &never };
    let first = download(&client, &uri, &dest, &mut ctl).unwrap();
    let on_disk = std::fs::read(&path).unwrap();
    assert!(std::fs::read(&dest).unwrap() == on_disk);
    assert_eq!((first.bytes, parts), (on_disk.len() as u64, 2));
    assert!(first.etag.starts_with('"'), "{}", first.etag);
    assert_eq!(leftovers(), 1, "only the file");

    // Replaced after the first part: the second part's If-Match fails and the download starts over.
    let mut replaced = false;
    let mut progress = |_: TransferProgress| {
        if !replaced {
            replaced = true;
            write_model(&path, 2);
        }
    };
    let mut ctl = Control { progress: &mut progress, stop: &never };
    let second = download(&client, &uri, &dest, &mut ctl).unwrap();
    let new = std::fs::read(&path).unwrap();
    assert!(new != on_disk && std::fs::read(&dest).unwrap() == new, "the new version, not a mix");
    assert_ne!(second.etag, first.etag);
    assert_eq!(leftovers(), 1);

    // A file of several parts comes in two requests on a link this fast: the first part, then the rest at once, so
    // a file replaced every few seconds (a run's last.folo) has one moment to get in the way, not one per part.
    write_sized(&path, "s", 3);
    let large = std::fs::read(&path).unwrap();
    assert!(large.len() as u64 > 3 * folo_client::transfer::PART_BYTES, "{} bytes", large.len());
    let mut seen = Vec::new();
    let mut progress = |p: TransferProgress| seen.push((p.done_bytes, p.total_bytes));
    let mut ctl = Control { progress: &mut progress, stop: &never };
    let third = download(&client, &uri, &dest, &mut ctl).unwrap();
    let len = large.len() as u64;
    assert_eq!(seen, [(folo_client::transfer::PART_BYTES, len), (len, len)]);
    assert!(third.bytes == len && std::fs::read(&dest).unwrap() == large);
    assert_eq!(leftovers(), 1);

    // Stopped, or a model that is not there: no file, no leftovers.
    let other = t.join("out").join("other.folo");
    let mut ctl = Control { progress: &mut |_| {}, stop: &|| true };
    assert!(matches!(download(&client, &uri, &other, &mut ctl), Err(TransferError::Stopped)));
    let mut ctl = Control { progress: &mut |_| {}, stop: &never };
    match download(&client, "/models/m_none/weights", &other, &mut ctl) {
        Err(TransferError::Failed(e)) => assert_eq!(e.code(), "model_not_found"),
        r => panic!("{r:?}"),
    }
    assert!(!other.exists());
    assert_eq!(leftovers(), 1);
}
