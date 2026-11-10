//! The harness: scripted agents, a server on a home folder, HTTP helpers, images and jobs.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::label::{Agent, AgentError, AgentReply, AgentRequest};
use folo_server::{Options, RealStore, Server};
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// What a scripted agent does on its next call.
pub(super) enum Step {
    /// Replies with this text.
    Reply(String),
    /// The CLI reports an error.
    Fail(&'static str),
    /// No answer in time.
    Timeout,
    /// Works until the run is cancelled.
    Hold,
    /// Waits for the switch, then labels as an unscripted call does.
    Until(Arc<AtomicBool>),
}

/// What a scripted agent was asked.
#[derive(Clone, Debug)]
pub(super) struct Asked {
    pub(super) prompt: String,
    pub(super) images: usize,
    pub(super) model: Option<String>,
    pub(super) effort: LabelEffort,
}

/// An agent that follows a script, then labels every image with one box of the first class.
pub(super) struct Scripted {
    id: LabelAgentId,
    available: bool,
    steps: Mutex<VecDeque<Step>>,
    pub(super) asked: Mutex<Vec<Asked>>,
    calls: AtomicUsize,
    /// When set, the next probe ([`Agent::info`]) says it has begun (`.0`) and waits to be let go (`.1`).
    pub(super) slow_probe: Mutex<Option<(Arc<AtomicBool>, Arc<AtomicBool>)>>,
}

impl Scripted {
    fn new(id: LabelAgentId, steps: Vec<Step>) -> Arc<Scripted> {
        Arc::new(Scripted {
            id,
            available: true,
            steps: Mutex::new(steps.into()),
            asked: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            slow_probe: Mutex::default(),
        })
    }

    pub(super) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// A reply with one box of `label` per image, at the same relative place in each.
pub(super) fn boxes_reply(images: usize, label: &str) -> String {
    let entries: Vec<Value> = (1..=images)
        .map(|i| json!({ "image": i, "boxes": [{ "label": label, "box": [250, 100, 750, 600] }] }))
        .collect();
    json!({ "images": entries }).to_string()
}

impl Agent for Scripted {
    fn id(&self) -> LabelAgentId {
        self.id
    }

    fn info(&self) -> LabelAgent {
        if let Some((begun, go)) = self.slow_probe.lock().unwrap().take() {
            begun.store(true, Ordering::SeqCst);
            while !go.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        LabelAgent {
            id: self.id,
            name: self.id.display_name().into(),
            available: self.available,
            version: self.available.then(|| "1.0.0".into()),
            models: vec![LabelModel { id: "small".into(), label: "Small".into() }],
            default_model: Some("small".into()),
            default_effort: LabelEffort::Medium,
            hint: (!self.available).then(|| "install it".into()),
        }
    }

    fn default_batch(&self) -> usize {
        2
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(req.scratch.is_dir(), "the call's scratch folder exists");
        assert!(req.images.iter().all(|jpeg| jpeg.starts_with(&[0xff, 0xd8])), "images arrive as JPEG");
        self.asked.lock().unwrap().push(Asked {
            prompt: req.prompt.to_string(),
            images: req.images.len(),
            model: req.model.map(str::to_string),
            effort: req.effort,
        });
        let step = self.steps.lock().unwrap().pop_front();
        let text = match step {
            Some(Step::Reply(text)) => text,
            Some(Step::Fail(why)) => return Err(AgentError::Failed(why.into())),
            Some(Step::Timeout) => return Err(AgentError::Timeout),
            Some(Step::Hold) => {
                while !req.cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                return Err(AgentError::Cancelled);
            }
            Some(Step::Until(_)) | None => {
                if let Some(Step::Until(open)) = &step {
                    while !open.load(Ordering::SeqCst) {
                        // A test that fails before it opens the gate ends the run instead (the store's drop).
                        if req.cancel.load(Ordering::Relaxed) {
                            return Err(AgentError::Cancelled);
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
                let first_class = req.prompt.lines().next().unwrap().trim_start_matches("Classes: ");
                boxes_reply(req.images.len(), first_class.split(", ").next().unwrap())
            }
        };
        let usage = LabelUsage {
            calls: 1,
            input_tokens: 1000,
            cached_input_tokens: 400,
            output_tokens: 50,
            cost_usd: Some(0.01),
        };
        Ok(AgentReply { text, usage })
    }

    fn hint(&self, message: &str) -> Option<String> {
        message.contains("/login").then(|| "sign in first".into())
    }
}

pub(super) struct Fixture {
    pub(super) app: Router,
    pub(super) store: Arc<RealStore>,
    pub(super) agent: Arc<Scripted>,
}

/// A server on `home` whose Claude Code agent follows `steps`.
pub(super) fn fixture(home: &Path, steps: Vec<Step>) -> Fixture {
    fixture_with(home, steps, |store| store)
}

/// [`fixture`] with the store changed by `tune` before it serves.
pub(super) fn fixture_with(
    home: &Path,
    steps: Vec<Step>,
    tune: impl FnOnce(RealStore) -> RealStore,
) -> Fixture {
    let agent = Scripted::new(LabelAgentId::Claude, steps);
    let missing = Arc::new(Scripted {
        id: LabelAgentId::Codex,
        available: false,
        steps: Mutex::default(),
        asked: Mutex::default(),
        calls: AtomicUsize::new(0),
        slow_probe: Mutex::default(),
    });
    let store = Arc::new(tune(
        RealStore::open(home.to_path_buf())
            .recheck_every(Duration::ZERO)
            .label_agents(vec![agent.clone(), missing]),
    ));
    Fixture { app: Server::new(store.clone(), Options::default()).router(), store, agent }
}

pub(super) async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, body.collect().await.unwrap().to_bytes().to_vec())
}

pub(super) fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn json_request(method: &str, uri: &str, v: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

pub(super) fn post(uri: &str, v: Value) -> Request<Body> {
    json_request("POST", uri, v)
}

pub(super) fn put(uri: &str, v: Value) -> Request<Body> {
    json_request("PUT", uri, v)
}

pub(super) fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

pub(super) async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = format!("{} {}", req.method(), req.uri());
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

pub(super) async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = format!("{} {}", req.method(), req.uri());
    let (got, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A `width` x `height` PNG whose colour (and so whose bytes) depends on `n`.
pub(super) fn png(n: u8, width: u32, height: u32) -> Vec<u8> {
    let image = folo_data::Image::new(width, height, [n.wrapping_mul(37), 120, 255 - n]);
    folo_data::encode(&image, folo_data::ImageFormat::Png).unwrap()
}

/// A multipart upload of `files` (name, bytes) as `image` parts.
pub(super) fn upload(uri: &str, files: &[(&str, Vec<u8>)]) -> Request<Body> {
    let boundary = "XfoloLabelBoundary";
    let mut body = Vec::new();
    for (name, bytes) in files {
        let head = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        );
        body.extend_from_slice(head.as_bytes());
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::post(uri)
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap()
}

/// A job named `street` with classes `car` and `person` and `images` uploaded 200x100 images.
pub(super) async fn job_with_images(app: &Router, images: u8) -> LabelJob {
    let body = json!({ "name": "street", "classes": ["car", "person"], "instructions": "Only parked cars." });
    let job: LabelJob = ok(app, post("/api/label/jobs", body)).await;
    if images == 0 {
        return job;
    }
    let files: Vec<(String, Vec<u8>)> =
        (0..images).map(|n| (format!("img{n}.png"), png(n, 200, 100))).collect();
    let files: Vec<(&str, Vec<u8>)> = files.iter().map(|(n, b)| (n.as_str(), b.clone())).collect();
    let added: LabelImagesAdded =
        ok(app, upload(&format!("/api/label/jobs/{}/images", job.id), &files)).await;
    assert_eq!((added.added.len(), added.skipped.len()), (images as usize, 0));
    added.job
}

/// The job once its run has ended.
pub(super) async fn idle(app: &Router, id: &str) -> LabelJob {
    for _ in 0..2000 {
        let job: LabelJob = ok(app, get(&format!("/api/label/jobs/{id}"))).await;
        if job.status == LabelJobStatus::Idle {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("job {id} did not become idle");
}

pub(super) async fn images(app: &Router, id: &str, query: &str) -> LabelImagePage {
    ok(app, get(&format!("/api/label/jobs/{id}/images{query}"))).await
}
