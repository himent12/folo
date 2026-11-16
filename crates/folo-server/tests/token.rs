//! `folo serve --token`: every `/api` request but `GET /api/health` and CORS preflights needs `Authorization:
//! Bearer <token>`; anything else is `401 unauthorized`. The token never reaches the log.

use std::io::Write;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::ErrorBody;
use folo_server::{MockStore, Options, Server, Token};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

const SECRET: &str = "s3cret-Token_42";

fn app() -> Router {
    let opts = Options { token: Token::new(SECRET), ..Options::default() };
    Server::new(Arc::new(MockStore::new()), opts).router()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
}

fn get(uri: &str, auth: Option<&str>) -> Request<Body> {
    let mut req = Request::get(uri);
    if let Some(a) = auth {
        req = req.header(header::AUTHORIZATION, a);
    }
    req.body(Body::empty()).unwrap()
}

fn create_run(auth: Option<&str>) -> Request<Body> {
    let mut req = Request::post("/api/runs").header(header::CONTENT_TYPE, "application/json");
    if let Some(a) = auth {
        req = req.header(header::AUTHORIZATION, a);
    }
    req.body(Body::from(json!({ "dataset_id": "ds_5a1e", "epochs": 3 }).to_string())).unwrap()
}

/// Asserts a `401 unauthorized` answer with the error envelope, `WWW-Authenticate: Bearer` and a hint.
async fn unauthorized(app: &Router, req: Request<Body>) -> String {
    let what = format!("{} {}", req.method(), req.uri());
    let (status, headers, body) = send(app, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{what}");
    assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");
    let e = serde_json::from_slice::<ErrorBody>(&body).unwrap().error;
    assert_eq!(e.code, "unauthorized");
    assert!(e.hint.as_deref().is_some_and(|h| h.contains("--token")), "{e:?}");
    assert!(!String::from_utf8_lossy(&body).contains(SECRET));
    e.message
}

/// A log sink shared with the test.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn api_requests_need_the_token_and_it_is_never_logged() {
    // Everything the server logs while it answers, down to TRACE (single-threaded runtime: the requests run on
    // this thread, where the subscriber is the default).
    let log = Captured::default();
    let sink = log.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || sink.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let app = app();
    let bearer = format!("Bearer {SECRET}");

    // Without the token or with another one: reads and writes, known routes and unknown ones.
    let missing = unauthorized(&app, get("/api/datasets", None)).await;
    assert!(missing.contains("needs a token"), "{missing}");
    let wrong = unauthorized(&app, get("/api/datasets", Some("Bearer s3cret-Token_43"))).await;
    assert!(wrong.contains("not this server's"), "{wrong}");
    unauthorized(&app, get("/api/datasets", Some(SECRET))).await;
    unauthorized(&app, get("/api/datasets", Some(&format!("Basic {SECRET}")))).await;
    unauthorized(&app, get("/api/datasets", Some(&format!("Bearer {SECRET}x")))).await;
    unauthorized(&app, create_run(None)).await;
    unauthorized(&app, create_run(Some("Bearer nope"))).await;
    unauthorized(&app, get("/api/no-such-route", None)).await;
    // A token in the query string is not accepted, and not logged with the request's URL either (x3-server-2).
    unauthorized(&app, get(&format!("/api/runs?token={SECRET}"), None)).await;
    let mut post_health = get("/api/health", None);
    *post_health.method_mut() = axum::http::Method::POST;
    unauthorized(&app, post_health).await;

    // Health is open, so a client can tell a server that is up from one it may not use; so are preflights.
    let (status, _, body) = send(&app, get("/api/health", None)).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let preflight = Request::builder()
        .method("OPTIONS")
        .uri("/api/runs")
        .header(header::ORIGIN, "http://localhost:5173")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "authorization")
        .body(Body::empty())
        .unwrap();
    assert!(send(&app, preflight).await.0.is_success());
    // Outside /api the web app stays public (here: not built, 404 with a hint, not 401).
    assert_eq!(send(&app, get("/", None)).await.0, StatusCode::NOT_FOUND);

    // The token passes, with the scheme in any case.
    assert_eq!(send(&app, get("/api/datasets", Some(&bearer))).await.0, StatusCode::OK);
    assert_eq!(send(&app, get("/api/runs", Some(&format!("bearer {SECRET}")))).await.0, StatusCode::OK);
    let (status, _, body) = send(&app, create_run(Some(&bearer))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&body));

    let text = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    assert!(text.contains("/api/datasets"), "the requests were logged: {text}");
    assert!(!text.contains(SECRET), "the token reached the log: {text}");
}

#[test]
fn tokens_are_never_shown_and_never_empty() {
    assert!(Token::new("").is_none());
    let t = Token::new(SECRET).unwrap();
    assert!(!format!("{t:?}").contains(SECRET));
    let opts = Options { token: Some(t), ..Options::default() };
    assert!(!format!("{opts:?}").contains(SECRET));
}
