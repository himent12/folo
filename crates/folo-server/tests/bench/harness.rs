//! The harness: scripted agents, a CPU server, HTTP helpers, datasets, models, events and predictions.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_nn::io::ModelConfig;
use folo_server::api::*;
use folo_server::label::agent::{Failure, fault_hint};
use folo_server::label::{Agent, AgentError, AgentReply, AgentRequest};
use folo_server::{Options, RealStore, Server};
use folo_vision::FoloModel;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Longest wait for an entry, a run or a loop to end.
pub(super) const TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------

/// What a scripted agent does on its next call.
pub(super) enum Step {
    /// Replies with this text.
    Reply(String),
    /// The CLI reports an error.
    Fail(&'static str),
    /// The CLI reports an error after the call reached the model: it used [`paid`].
    FailPaid(&'static str),
    /// The CLI reports an error with an HTTP status in a field of its own (Claude Code's `api_error_status`).
    FailStatus(&'static str, u16),
    /// Works until the call is cancelled.
    Hold,
}

/// An agent that follows a script, then labels every image with one box of the first class.
pub(super) struct Scripted {
    id: LabelAgentId,
    available: bool,
    steps: Mutex<VecDeque<Step>>,
    pub(super) calls: AtomicUsize,
    pub(super) prompts: Mutex<Vec<(String, usize)>>,
}

impl Scripted {
    fn new(id: LabelAgentId, available: bool) -> Arc<Scripted> {
        Arc::new(Scripted {
            id,
            available,
            steps: Mutex::default(),
            calls: AtomicUsize::new(0),
            prompts: Mutex::default(),
        })
    }

    pub(super) fn script(&self, steps: Vec<Step>) {
        *self.steps.lock().unwrap() = steps.into();
    }
}

/// A labeling reply with one box of `label` per image.
fn boxes_reply(images: usize, label: &str) -> String {
    let entries: Vec<Value> = (1..=images)
        .map(|i| json!({ "image": i, "boxes": [{ "label": label, "box": [50, 10, 150, 60] }] }))
        .collect();
    json!({ "images": entries }).to_string()
}

/// What a [`Step::FailPaid`] call used.
pub(super) fn paid() -> LabelUsage {
    LabelUsage {
        calls: 1,
        input_tokens: 300,
        cached_input_tokens: 0,
        output_tokens: 0,
        cost_usd: Some(0.002),
    }
}

impl Agent for Scripted {
    fn id(&self) -> LabelAgentId {
        self.id
    }

    fn info(&self) -> LabelAgent {
        LabelAgent {
            id: self.id,
            name: self.id.display_name().into(),
            available: self.available,
            version: self.available.then(|| "1.0.0".into()),
            models: vec![LabelModel { id: "small".into(), label: "Small".into(), fast: false }],
            default_model: Some("small".into()),
            default_effort: LabelEffort::Medium,
            fast: false,
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
        self.prompts.lock().unwrap().push((req.prompt.to_string(), req.images.len()));
        let step = self.steps.lock().unwrap().pop_front();
        let text = match step {
            Some(Step::Reply(text)) => text,
            Some(Step::Fail(why)) => return Err(AgentError::failed(why)),
            Some(Step::FailPaid(why)) => {
                return Err(AgentError::Failed(Failure { used: paid(), ..Failure::new(why) }));
            }
            Some(Step::FailStatus(why, status)) => {
                return Err(AgentError::Failed(Failure { status: Some(status), ..Failure::new(why) }));
            }
            Some(Step::Hold) => {
                while !req.cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                return Err(AgentError::Cancelled(Default::default()));
            }
            None => {
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
        // As the real agents: their own hints, then the ones every agent gives (a limit, a passing fault).
        message.contains("/login").then(|| "sign in first".into()).or_else(|| fault_hint(message, None))
    }
}

pub(super) struct Fixture {
    pub(super) app: Router,
    pub(super) store: Arc<RealStore>,
    pub(super) agent: Arc<Scripted>,
}

/// A CPU server on `home` whose Claude Code agent is scripted (Codex is not installed).
pub(super) fn fixture(home: &Path) -> Fixture {
    fixture_with(home, |s| s)
}

/// [`fixture`] with the store changed by `change` before it serves.
pub(super) fn fixture_with(home: &Path, change: impl FnOnce(RealStore) -> RealStore) -> Fixture {
    let agent = Scripted::new(LabelAgentId::Claude, true);
    let store = Arc::new(change(
        RealStore::open(home.to_path_buf())
            .recheck_every(Duration::ZERO)
            .device("cpu")
            .label_agents(vec![agent.clone(), Scripted::new(LabelAgentId::Codex, false)])
            .label_transient_backoff(Duration::from_millis(20)),
    ));
    Fixture { app: Server::new(store.clone(), Options::default()).router(), store, agent }
}

pub(super) async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
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
    let (status, _, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

pub(super) async fn bytes(app: &Router, uri: &str) -> (axum::http::HeaderMap, Vec<u8>) {
    let (status, headers, body) = send(app, get(uri)).await;
    assert_eq!(status, StatusCode::OK, "{uri}: {}", String::from_utf8_lossy(&body));
    (headers, body)
}

pub(super) async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = format!("{} {}", req.method(), req.uri());
    let (got, _, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A synthetic dataset of `images` 64 px images (80 / 20 train / val) with classes circle and square.
pub(super) async fn dataset(app: &Router, name: &str, images: u32) -> Dataset {
    let body =
        json!({ "name": name, "images": images, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 });
    ok(app, post("/api/datasets/synthetic", body)).await
}

/// A small plan: haze at 1 and 3, distance at 2.
pub(super) fn small_plan() -> Value {
    json!({ "conditions": [
        { "condition": "haze", "severities": [1, 3] },
        { "condition": "distance", "severities": [2], "weight": 2 } ] })
}

pub(super) async fn bench(app: &Router, body: Value) -> Bench {
    ok(app, post("/api/bench/benchmarks", body)).await
}

/// Writes an untrained folo-n at 64 px with `classes` into the models folder; returns its id.
pub(super) async fn model(app: &Router, home: &Path, name: &str, classes: &[&str], seed: u64) -> String {
    model_with(app, home, name, classes, seed, false).await
}

/// [`model`]; `confident` sets the class outputs' biases to 0, so the untrained model scores every anchor near
/// 0.5 instead of below validation's 0.001 (MODEL.md §6.5's prior), and the box outputs' to boxes of a few
/// strides: hundreds of detections per image, some of them right.
pub(super) async fn model_with(
    app: &Router,
    home: &Path,
    name: &str,
    classes: &[&str],
    seed: u64,
    confident: bool,
) -> String {
    let mut m = FoloModel::with_rng(
        ModelConfig::new("n", classes.len()).unwrap(),
        &mut folo_core::rng::Rng::seed(seed),
    )
    .unwrap();
    m.set_classes(classes.iter().map(|c| c.to_string()).collect()).unwrap();
    m.set_imgsz(64);
    if confident {
        for (name, t) in folo_nn::Module::named_parameters(&m) {
            if name.contains(".cls.") && name.ends_with("out.bias") {
                folo_nn::init::constant_(&t, 0.0).unwrap();
            }
            // Class weights that tell anchors apart (an initialised output layer is zero: every score would tie,
            // and AP of tied scores depends on the order of the images).
            if name.contains(".cls.") && name.ends_with("out.weight") {
                let n: usize = t.dims().iter().product();
                let v: Vec<f32> = (0..n).map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 0.5).collect();
                folo_nn::init::copy_(&t, &folo_core::Tensor::from_vec(v, t.dims().to_vec()).unwrap())
                    .unwrap();
            }
            // Box distances of about one stride (DFL bin 1), so boxes have the shapes' sizes.
            if name.contains(".box.") && name.ends_with("out.bias") {
                let n = t.dims()[0];
                let v: Vec<f32> = (0..n).map(|k| if k % (n / 4) == 1 { 8.0 } else { 0.0 }).collect();
                folo_nn::init::copy_(&t, &folo_core::Tensor::from_vec(v, [n]).unwrap()).unwrap();
            }
        }
    }
    std::fs::create_dir_all(home.join("models")).unwrap();
    m.save(home.join("models").join(format!("{name}.folo"))).unwrap();
    let models: ModelList = ok(app, get("/api/models")).await;
    models.models.into_iter().find(|m| m.name == name).unwrap().id
}

/// The entry once it has ended.
pub(super) async fn ended(app: &Router, bench: &str, id: &str) -> BenchEntry {
    let t0 = std::time::Instant::now();
    loop {
        let e: BenchEntry = ok(app, get(&format!("/api/bench/benchmarks/{bench}/entries/{id}"))).await;
        if !e.status.is_active() {
            return e;
        }
        assert!(t0.elapsed() < TIMEOUT, "entry {id} did not end: {e:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The loop once it has ended.
pub(super) async fn loop_ended(app: &Router, bench: &str, id: &str) -> BenchLoop {
    let t0 = std::time::Instant::now();
    loop {
        let l: BenchLoop = ok(app, get(&format!("/api/bench/benchmarks/{bench}/loops/{id}"))).await;
        if !l.status.is_active() {
            return l;
        }
        assert!(t0.elapsed() < TIMEOUT, "loop {id} did not end: {l:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// An SSE stream of a benchmark's events.
pub(super) struct Events {
    body: Body,
    buf: String,
}

impl Events {
    pub(super) async fn open(app: &Router, bench: &str) -> Events {
        let res = app.clone().oneshot(get(&format!("/api/bench/benchmarks/{bench}/events"))).await.unwrap();
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
}

/// Predictions that are exactly the truth of `images` (unchanged, or under `condition` / `severity`).
pub(super) fn truth_predictions(
    images: &[BenchImage],
    variant: Option<(&str, u8)>,
    score: f32,
) -> Vec<Value> {
    images
        .iter()
        .map(|im| {
            let boxes: Vec<Value> = im
                .boxes
                .iter()
                .map(|b| json!({ "class": b.class, "score": score, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2 }))
                .collect();
            match variant {
                Some((c, s)) => json!({ "image": im.id, "condition": c, "severity": s, "boxes": boxes }),
                None => json!({ "image": im.id, "boxes": boxes }),
            }
        })
        .collect()
}

pub(super) fn external(name: &str, scored: bool, predictions: Vec<Value>) -> Value {
    json!({ "name": name, "contestant": { "kind": "external", "scored": scored, "predictions": predictions } })
}

pub(super) fn axis(score: &BenchScore, axis: BenchAxis) -> Option<f32> {
    score.axes.iter().find(|a| a.axis == axis).unwrap().value
}

pub(super) async fn page(app: &Router, bench: &str, query: &str) -> BenchImagePage {
    ok(app, get(&format!("/api/bench/benchmarks/{bench}/images?limit=1000{query}"))).await
}

pub(super) fn open_entry(name: &str, predictions: Vec<Value>) -> Value {
    json!({ "name": name, "contestant": { "kind": "external", "open": true, "predictions": predictions } })
}

pub(super) fn parts(predictions: &[Value]) -> Value {
    json!({ "predictions": predictions })
}

/// Waits until model entry `id` has run every pass and is held before its grade ([`RealStore::hold_bench_entries`]).
pub(super) async fn held(app: &Router, bench: &str, id: &str) {
    let t0 = std::time::Instant::now();
    loop {
        let e: BenchEntry = ok(app, get(&format!("/api/bench/benchmarks/{bench}/entries/{id}"))).await;
        if e.status == BenchEntryStatus::Running && e.progress.done == e.progress.total {
            return;
        }
        assert!(t0.elapsed() < TIMEOUT, "{e:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
