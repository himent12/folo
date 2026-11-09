//! `GET /api/models/{id}/weights`: the mock store's generated folo-n files and the real store's `.folo` files
//! (streamed), their validators (ETag, Last-Modified, 304s), the cached f16 export, errors, compression and CORS.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use folo_nn::io::{FloatDType, ModelConfig, ModelFile, Purpose};
use folo_server::api::*;
use folo_server::store::{WeightsBody, WeightsFormat};
use folo_server::{MockStore, Options, RealStore, Server, Store};
use folo_vision::FoloModel;
use http_body_util::BodyExt;
use tower::ServiceExt;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("folo-weights-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A response's status, headers, body and the sizes of its body frames.
async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>, Vec<usize>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, mut body) = res.into_parts();
    let (mut bytes, mut frames) = (Vec::new(), Vec::new());
    while let Some(f) = body.frame().await {
        if let Ok(data) = f.unwrap().into_data() {
            frames.push(data.len());
            bytes.extend_from_slice(&data);
        }
    }
    (parts.status, parts.headers, bytes, frames)
}

fn get(uri: &str, headers: &[(header::HeaderName, &str)]) -> Request<Body> {
    let mut req = Request::get(uri);
    for (k, v) in headers {
        req = req.header(k, *v);
    }
    req.body(Body::empty()).unwrap()
}

fn header(h: &HeaderMap, name: header::HeaderName) -> &str {
    h.get(&name).unwrap_or_else(|| panic!("no {name}")).to_str().unwrap()
}

async fn error_code(app: &Router, uri: &str, status: StatusCode) -> String {
    let (got, _, body, _) = send(app, get(uri, &[])).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    serde_json::from_slice::<ErrorBody>(&body).unwrap().error.code
}

/// Checks a 200 weights response and the conditional requests against it; returns the ETag.
async fn check_validators(app: &Router, uri: &str, headers: &HeaderMap, body: &[u8]) -> String {
    assert_eq!(header(headers, header::CONTENT_TYPE), "application/octet-stream");
    assert_eq!(header(headers, header::CONTENT_LENGTH), body.len().to_string());
    assert_eq!(header(headers, header::CACHE_CONTROL), "no-cache");
    let etag = header(headers, header::ETAG).to_string();
    assert!(etag.starts_with('"') && etag.ends_with('"') && etag.len() > 2, "a strong ETag: {etag}");
    let modified = header(headers, header::LAST_MODIFIED).to_string();
    let lm = httpdate::parse_http_date(&modified).unwrap();

    for inm in [etag.clone(), format!("W/{etag}"), format!("\"x\", {etag}"), "*".into()] {
        let (s, h, b, _) = send(app, get(uri, &[(header::IF_NONE_MATCH, &inm)])).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED, "{inm}");
        assert!(b.is_empty());
        assert_eq!(
            (header(&h, header::ETAG), header(&h, header::LAST_MODIFIED)),
            (etag.as_str(), &*modified)
        );
    }
    let (s, _, b, _) = send(app, get(uri, &[(header::IF_NONE_MATCH, "\"other\"")])).await;
    assert_eq!((s, b.len()), (StatusCode::OK, body.len()));

    let (s, _, _, _) = send(app, get(uri, &[(header::IF_MODIFIED_SINCE, &modified)])).await;
    assert_eq!(s, StatusCode::NOT_MODIFIED);
    let later = httpdate::fmt_http_date(lm + std::time::Duration::from_secs(60));
    assert_eq!(send(app, get(uri, &[(header::IF_MODIFIED_SINCE, &later)])).await.0, StatusCode::NOT_MODIFIED);
    let earlier = httpdate::fmt_http_date(lm - std::time::Duration::from_secs(1));
    assert_eq!(send(app, get(uri, &[(header::IF_MODIFIED_SINCE, &earlier)])).await.0, StatusCode::OK);
    assert_eq!(send(app, get(uri, &[(header::IF_MODIFIED_SINCE, "yesterday")])).await.0, StatusCode::OK);
    // If-None-Match wins over If-Modified-Since.
    let both = [(header::IF_NONE_MATCH, "\"other\""), (header::IF_MODIFIED_SINCE, &*modified)];
    assert_eq!(send(app, get(uri, &both)).await.0, StatusCode::OK);

    // Never gzip-encoded, so Content-Length and the strong ETag hold on the wire.
    let (s, h, b, _) = send(app, get(uri, &[(header::ACCEPT_ENCODING, "gzip")])).await;
    assert_eq!(s, StatusCode::OK);
    assert!(h.get(header::CONTENT_ENCODING).is_none());
    assert_eq!((header(&h, header::ETAG), b.as_slice()), (etag.as_str(), body));

    // HEAD: the same headers, no body.
    let head = Request::head(uri).body(Body::empty()).unwrap();
    let (s, h, b, _) = send(app, head).await;
    assert_eq!((s, b.len()), (StatusCode::OK, 0));
    assert_eq!(header(&h, header::CONTENT_LENGTH), body.len().to_string());
    assert_eq!(header(&h, header::ETAG), etag);

    // The Vite dev server's origin may fetch it and read the validators, also with a conditional HEAD.
    let preflight = Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header(header::ORIGIN, "http://localhost:5173")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "HEAD")
        .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "if-none-match")
        .body(Body::empty())
        .unwrap();
    let (s, h, _, _) = send(app, preflight).await;
    assert!(s.is_success(), "{s}");
    assert_eq!(header(&h, header::ACCESS_CONTROL_ALLOW_ORIGIN), "http://localhost:5173");
    assert!(header(&h, header::ACCESS_CONTROL_ALLOW_METHODS).contains("HEAD"), "{h:?}");
    let (s, h, _, _) = send(app, get(uri, &[(header::ORIGIN, "http://localhost:5173")])).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(header(&h, header::ACCESS_CONTROL_ALLOW_ORIGIN), "http://localhost:5173");
    let exposed = header(&h, header::ACCESS_CONTROL_EXPOSE_HEADERS).to_ascii_lowercase();
    assert!(exposed.contains("etag") && exposed.contains("content-disposition"), "{exposed}");
    etag
}

/// Parses a served file and builds the model it describes, as the browser detector does.
fn load(bytes: &[u8]) -> ModelFile {
    let file = ModelFile::from_bytes(bytes.to_vec()).unwrap();
    FoloModel::from_file(&file, Purpose::Inference).unwrap();
    file
}

#[tokio::test]
async fn mock_weights_are_a_real_folo_n_file() {
    let app = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let model: Model = {
        let (_, _, b, _) = send(&app, get("/api/models/m_31aa", &[])).await;
        serde_json::from_slice(&b).unwrap()
    };
    let uri = "/api/models/m_31aa/weights";
    let (status, headers, body, _) = send(&app, get(uri, &[])).await;
    assert_eq!(status, StatusCode::OK);
    let file = load(&body);
    let meta = file.metadata();
    assert_eq!((meta.arch.as_str(), meta.dtype), ("folo-n", FloatDType::F32));
    assert_eq!((&meta.classes, meta.imgsz), (&model.classes, model.imgsz));
    assert_eq!(
        header(&headers, header::CONTENT_DISPOSITION),
        format!("attachment; filename=\"{}.folo\"", model.name)
    );
    let etag = check_validators(&app, uri, &headers, &body).await;

    // Deterministic: another store serves the same bytes under the same ETag.
    let other = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let (_, h2, b2, _) = send(&other, get(uri, &[])).await;
    assert_eq!((header(&h2, header::ETAG), &b2), (etag.as_str(), &body));

    // f16: the fused export of the same weights, half the size.
    let uri16 = "/api/models/m_31aa/weights?format=f16";
    let (status, h16, b16, _) = send(&app, get(uri16, &[])).await;
    assert_eq!(status, StatusCode::OK);
    let f16 = load(&b16);
    assert_eq!((f16.metadata().dtype, f16.metadata().fused), (FloatDType::F16, true));
    assert_eq!(f16.metadata().classes, model.classes);
    assert!(b16.len() < body.len() / 2 + 64 * 1024, "{} vs {}", b16.len(), body.len());
    assert!(header(&h16, header::CONTENT_DISPOSITION).ends_with("-f16.folo\""));
    let etag16 = check_validators(&app, uri16, &h16, &b16).await;
    assert_ne!(etag16, etag);

    // An o2o model's file has the one-to-one branch.
    let (status, _, o2o, _) = send(&app, get("/api/models/m_31ab/weights", &[])).await;
    assert_eq!(status, StatusCode::OK);
    assert!(load(&o2o).metadata().config.has_o2o() && !meta.config.has_o2o());

    assert_eq!(
        error_code(&app, "/api/models/m_none/weights", StatusCode::NOT_FOUND).await,
        "model_not_found"
    );
    let bad = error_code(&app, "/api/models/m_31aa/weights?format=onnx", StatusCode::BAD_REQUEST).await;
    assert_eq!(bad, "invalid_query");
}

/// Writes a folo-n weights file with two classes, initialised from `seed`.
fn write_model(path: &Path, seed: u64) {
    let mut m =
        FoloModel::with_rng(ModelConfig::new("n", 2).unwrap(), &mut folo_core::rng::Rng::seed(seed)).unwrap();
    m.set_classes(vec!["cat".into(), "dog".into()]).unwrap();
    m.set_imgsz(320);
    m.save(path).unwrap();
}

fn f16_cache(home: &Path) -> Vec<String> {
    let Ok(dir) = std::fs::read_dir(home.join("cache/weights")) else { return Vec::new() };
    dir.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn real_weights_stream_with_validators_and_a_cached_f16_export() {
    let t = TempDir::new("real");
    let home = t.0.join("home");
    let path = home.join("models").join("zoo").join("pets n+v2.folo");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    write_model(&path, 1);
    let on_disk = std::fs::read(&path).unwrap();
    assert!(on_disk.len() > 8 << 20, "a large-ish file: {} bytes", on_disk.len());

    let store = Arc::new(RealStore::open(home.clone()).device("cpu"));
    let app = Server::new(store.clone(), Options::default()).router();
    let models: ModelList = {
        let (_, _, b, _) = send(&app, get("/api/models", &[])).await;
        serde_json::from_slice(&b).unwrap()
    };
    let id = models.models[0].id.clone();
    let uri = format!("/api/models/{id}/weights");

    // Streamed in chunks (not one buffer), byte for byte the file.
    let (status, headers, body, frames) = send(&app, get(&uri, &[])).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body == on_disk, "the file's bytes");
    assert!(frames.len() >= on_disk.len() / (256 * 1024), "{} frames", frames.len());
    assert!(frames.iter().all(|&n| n <= 256 * 1024), "chunks of at most 256 KiB: {frames:?}");
    assert_eq!(header(&headers, header::CONTENT_DISPOSITION), "attachment; filename=\"pets_n_v2.folo\"");
    let etag = check_validators(&app, &uri, &headers, &body).await;

    // f16: exported on the first request, then served from cache/weights/.
    let uri16 = format!("{uri}?format=f16");
    let (status, h16, b16, _) = send(&app, get(&uri16, &[])).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(load(&b16).metadata().dtype, FloatDType::F16);
    let cache = f16_cache(&home);
    assert_eq!(cache.len(), 1, "{cache:?}");
    assert!(cache[0].starts_with(&format!("{id}.")) && cache[0].ends_with(".f16.folo"), "{cache:?}");
    let etag16 = check_validators(&app, &uri16, &h16, &b16).await;
    let (_, again, _, _) = send(&app, get(&uri16, &[])).await;
    assert_eq!(header(&again, header::ETAG), etag16, "served from the cache");
    assert_eq!(f16_cache(&home), cache);

    // New weights under the same name: new validators, the old copy is no longer current, the f16 is redone.
    // A download in progress keeps its bytes when a newer export replaces (and removes) its cache file.
    let held = store.model_weights(&id, WeightsFormat::F16).unwrap();
    write_model(&path, 2);
    let (status, h2, b2, _) = send(&app, get(&uri, &[(header::IF_NONE_MATCH, &etag)])).await;
    assert_eq!(status, StatusCode::OK);
    assert!(b2 == std::fs::read(&path).unwrap() && b2 != on_disk);
    assert_ne!(header(&h2, header::ETAG), etag);
    let (status, h16b, b16b, _) = send(&app, get(&uri16, &[(header::IF_NONE_MATCH, &etag16)])).await;
    assert_eq!(status, StatusCode::OK);
    assert_ne!(header(&h16b, header::ETAG), etag16);
    assert_ne!(b16b, b16);
    let cache2 = f16_cache(&home);
    assert_eq!(cache2.len(), 1, "the stale export is removed: {cache2:?}");
    assert_ne!(cache2, cache);
    let WeightsBody::File(mut f) = held.body else { panic!("real weights are files") };
    let mut old = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut old).unwrap();
    assert!(old == b16, "the held handle still reads the old export");
    assert_eq!(held.etag, etag16);

    // Gone: 404, also for the export.
    std::fs::remove_file(&path).unwrap();
    assert_eq!(error_code(&app, &uri, StatusCode::NOT_FOUND).await, "model_not_found");
    assert_eq!(error_code(&app, &uri16, StatusCode::NOT_FOUND).await, "model_not_found");
    let bad = error_code(&app, "/api/models/m_none/weights?format=f16", StatusCode::NOT_FOUND).await;
    assert_eq!(bad, "model_not_found");
}
