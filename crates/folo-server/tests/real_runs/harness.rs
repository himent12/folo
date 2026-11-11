//! The harness: a CPU store and its router, HTTP helpers, a tiny dataset and run, and an SSE reader.

use std::path::Path;
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

/// Longest wait for a run to reach a state (a 2-epoch run here takes a few seconds).
const TIMEOUT: Duration = Duration::from_secs(300);

pub(super) fn store(home: &Path) -> Arc<RealStore> {
    Arc::new(RealStore::open(home.to_path_buf()).recheck_every(Duration::ZERO).device("cpu"))
}

pub(super) fn app(store: &Arc<RealStore>) -> Router {
    Server::new(store.clone(), Options::default()).router()
}

pub(super) async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, body.collect().await.unwrap().to_bytes().to_vec())
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

pub(super) fn post(uri: &str) -> Request<Body> {
    Request::post(uri).body(Body::empty()).unwrap()
}

pub(super) fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

pub(super) async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = req.uri().to_string();
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

pub(super) async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = req.uri().to_string();
    let (got, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A synthetic dataset of `images` 64 px images (80 / 20 train / val) with two classes.
pub(super) async fn dataset(app: &Router, name: &str, images: u32) -> Dataset {
    let body =
        json!({ "name": name, "images": images, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 });
    ok(app, post_json("/api/datasets/synthetic", body)).await
}

/// A tiny CPU run's config.
pub(super) fn tiny(ds: &Dataset, epochs: u32) -> Value {
    json!({ "dataset_id": ds.id, "epochs": epochs, "batch": 4, "imgsz": 64, "device": "cpu",
            "augment": { "close_mosaic": 1 } })
}

/// An SSE stream of `GET /api/runs/{id}/events`.
pub(super) struct Events {
    body: Body,
    buf: String,
}

impl Events {
    pub(super) async fn open(app: &Router, run: &str) -> Events {
        let res = app.clone().oneshot(get(&format!("/api/runs/{run}/events"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        Events { body: res.into_body(), buf: String::new() }
    }

    /// The next event (name, data), skipping comments.
    pub(super) async fn next(&mut self) -> (String, Value) {
        loop {
            if let Some(end) = self.buf.find("\n\n") {
                let block: String = self.buf.drain(..end + 2).collect();
                let (mut name, mut data) = (None, String::new());
                for line in block.lines() {
                    if let Some(n) = line.strip_prefix("event:") {
                        name = Some(n.trim().to_string());
                    } else if let Some(d) = line.strip_prefix("data:") {
                        data.push_str(d.trim_start());
                    }
                }
                if let Some(name) = name {
                    return (name, serde_json::from_str(&data).unwrap());
                }
                continue;
            }
            let frame = tokio::time::timeout(TIMEOUT, self.body.frame())
                .await
                .expect("SSE event in time")
                .expect("SSE stream open")
                .unwrap();
            if let Ok(data) = frame.into_data() {
                self.buf.push_str(&String::from_utf8_lossy(&data).replace("\r\n", "\n"));
            }
        }
    }

    /// Reads events until a `status` event with one of `statuses`; returns every event read.
    pub(super) async fn until_status(&mut self, statuses: &[&str]) -> Vec<(String, Value)> {
        let mut seen = Vec::new();
        loop {
            let ev = self.next().await;
            let done = ev.0 == "status" && statuses.contains(&ev.1["status"].as_str().unwrap());
            seen.push(ev);
            if done {
                return seen;
            }
        }
    }

    /// Reads events until the first `epoch` event.
    pub(super) async fn until_epoch(&mut self) {
        while self.next().await.0 != "epoch" {}
    }
}

pub(super) fn count(events: &[(String, Value)], name: &str) -> usize {
    events.iter().filter(|(n, _)| n == name).count()
}
