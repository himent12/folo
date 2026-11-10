//! The harness: a mock server, HTTP helpers and an SSE reader.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{MockStore, Options, Server, Store};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tower::ServiceExt;

pub(super) fn mock() -> (Server, Arc<MockStore>) {
    let store = Arc::new(MockStore::new());
    let server = Server::new(store.clone() as Arc<dyn Store>, Options::default());
    (server, store)
}

pub(super) async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let headers = res.headers().clone();
    let body = res.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, headers, body)
}

pub(super) fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

pub(super) fn post_json(uri: &str, v: Value) -> Request<Body> {
    Request::post(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

pub(super) async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().clone();
    let (status, _, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

pub(super) async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().clone();
    let (got, headers, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().starts_with("application/json"));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}");
    assert!(!e.error.message.is_empty());
    e.error
}

/// Reads SSE frames until `n` events arrived (or the timeout hits). Returns `(event, data)` pairs and whether a
/// keep-alive comment was seen.
pub(super) async fn read_events(body: &mut Body, n: usize, timeout: Duration) -> Vec<(String, String)> {
    let mut buf = String::new();
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    while out.len() < n {
        let frame = match tokio::time::timeout_at(deadline, body.frame()).await {
            Ok(Some(f)) => f.unwrap(),
            Ok(None) | Err(_) => break,
        };
        if let Ok(data) = frame.into_data() {
            buf.push_str(std::str::from_utf8(&data).unwrap());
        }
        while let Some(end) = buf.find("\n\n") {
            let block: String = buf.drain(..end + 2).collect();
            let mut ev = None;
            let mut data = String::new();
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event: ") {
                    ev = Some(v.to_string());
                } else if let Some(v) = line.strip_prefix("data: ") {
                    data.push_str(v);
                }
            }
            if let Some(ev) = ev {
                out.push((ev, data));
            }
        }
    }
    out
}
