//! Vision benchmarks end to end over HTTP (docs/API.md, Vision benchmarks; docs/BENCH.md), on the CPU with tiny
//! synthetic datasets and untrained (or one-epoch) folo-n models, and scripted agents in place of the Claude Code
//! and Codex CLIs: benchmarks from a dataset split and from a labeling job, their images and variants, external,
//! model and agent entries, designing a plan, events, restarts, and a self-improvement loop.

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
use folo_server::label::{Agent, AgentError, AgentReply, AgentRequest};
use folo_server::{MockStore, Options, RealStore, Server, Store};
use folo_test_support::TestDir;
use folo_vision::FoloModel;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Longest wait for an entry, a run or a loop to end.
const TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------

/// What a scripted agent does on its next call.
enum Step {
    /// Replies with this text.
    Reply(String),
    /// The CLI reports an error.
    Fail(&'static str),
    /// Works until the call is cancelled.
    Hold,
}

/// An agent that follows a script, then labels every image with one box of the first class.
struct Scripted {
    id: LabelAgentId,
    available: bool,
    steps: Mutex<VecDeque<Step>>,
    calls: AtomicUsize,
    prompts: Mutex<Vec<(String, usize)>>,
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

    fn script(&self, steps: Vec<Step>) {
        *self.steps.lock().unwrap() = steps.into();
    }
}

/// A labeling reply with one box of `label` per image.
fn boxes_reply(images: usize, label: &str) -> String {
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
        self.prompts.lock().unwrap().push((req.prompt.to_string(), req.images.len()));
        let step = self.steps.lock().unwrap().pop_front();
        let text = match step {
            Some(Step::Reply(text)) => text,
            Some(Step::Fail(why)) => return Err(AgentError::Failed(why.into())),
            Some(Step::Hold) => {
                while !req.cancel.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                return Err(AgentError::Cancelled);
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
        message.contains("/login").then(|| "sign in first".into())
    }
}

struct Fixture {
    app: Router,
    store: Arc<RealStore>,
    agent: Arc<Scripted>,
}

/// A CPU server on `home` whose Claude Code agent is scripted (Codex is not installed).
fn fixture(home: &Path) -> Fixture {
    let agent = Scripted::new(LabelAgentId::Claude, true);
    let store = Arc::new(
        RealStore::open(home.to_path_buf())
            .recheck_every(Duration::ZERO)
            .device("cpu")
            .label_agents(vec![agent.clone(), Scripted::new(LabelAgentId::Codex, false)]),
    );
    Fixture { app: Server::new(store.clone(), Options::default()).router(), store, agent }
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
}

fn get(uri: &str) -> Request<Body> {
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

fn post(uri: &str, v: Value) -> Request<Body> {
    json_request("POST", uri, v)
}

fn put(uri: &str, v: Value) -> Request<Body> {
    json_request("PUT", uri, v)
}

fn delete(uri: &str) -> Request<Body> {
    Request::delete(uri).body(Body::empty()).unwrap()
}

async fn ok<T: DeserializeOwned>(app: &Router, req: Request<Body>) -> T {
    let uri = format!("{} {}", req.method(), req.uri());
    let (status, _, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn bytes(app: &Router, uri: &str) -> (axum::http::HeaderMap, Vec<u8>) {
    let (status, headers, body) = send(app, get(uri)).await;
    assert_eq!(status, StatusCode::OK, "{uri}: {}", String::from_utf8_lossy(&body));
    (headers, body)
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = format!("{} {}", req.method(), req.uri());
    let (got, _, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A synthetic dataset of `images` 64 px images (80 / 20 train / val) with classes circle and square.
async fn dataset(app: &Router, name: &str, images: u32) -> Dataset {
    let body =
        json!({ "name": name, "images": images, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 });
    ok(app, post("/api/datasets/synthetic", body)).await
}

/// A small plan: haze at 1 and 3, distance at 2.
fn small_plan() -> Value {
    json!({ "conditions": [
        { "condition": "haze", "severities": [1, 3] },
        { "condition": "distance", "severities": [2], "weight": 2 } ] })
}

async fn bench(app: &Router, body: Value) -> Bench {
    ok(app, post("/api/bench/benchmarks", body)).await
}

/// Writes an untrained folo-n at 64 px with `classes` into the models folder; returns its id.
async fn model(app: &Router, home: &Path, name: &str, classes: &[&str], seed: u64) -> String {
    model_with(app, home, name, classes, seed, false).await
}

/// [`model`]; `confident` sets the class outputs' biases to 0, so the untrained model scores every anchor near
/// 0.5 instead of below validation's 0.001 (MODEL.md §6.5's prior), and the box outputs' to boxes of a few
/// strides: hundreds of detections per image, some of them right.
async fn model_with(
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
async fn ended(app: &Router, bench: &str, id: &str) -> BenchEntry {
    let t0 = std::time::Instant::now();
    loop {
        let e: BenchEntry = ok(app, get(&format!("/api/bench/benchmarks/{bench}/entries/{id}"))).await;
        if !e.status.is_active() {
            return e;
        }
        assert!(t0.elapsed() < TIMEOUT, "entry {id} did not end: {e:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The loop once it has ended.
async fn loop_ended(app: &Router, bench: &str, id: &str) -> BenchLoop {
    let t0 = std::time::Instant::now();
    loop {
        let l: BenchLoop = ok(app, get(&format!("/api/bench/benchmarks/{bench}/loops/{id}"))).await;
        if !l.status.is_active() {
            return l;
        }
        assert!(t0.elapsed() < TIMEOUT, "loop {id} did not end: {l:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// An SSE stream of a benchmark's events.
struct Events {
    body: Body,
    buf: String,
}

impl Events {
    async fn open(app: &Router, bench: &str) -> Events {
        let res = app.clone().oneshot(get(&format!("/api/bench/benchmarks/{bench}/events"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        Events { body: res.into_body(), buf: String::new() }
    }

    /// The next event (name, data), skipping comments.
    async fn next(&mut self) -> (String, Value) {
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
fn truth_predictions(images: &[BenchImage], variant: Option<(&str, u8)>, score: f32) -> Vec<Value> {
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

fn external(name: &str, scored: bool, predictions: Vec<Value>) -> Value {
    json!({ "name": name, "contestant": { "kind": "external", "scored": scored, "predictions": predictions } })
}

fn axis(score: &BenchScore, axis: BenchAxis) -> Option<f32> {
    score.axes.iter().find(|a| a.axis == axis).unwrap().value
}

async fn page(app: &Router, bench: &str, query: &str) -> BenchImagePage {
    ok(app, get(&format!("/api/bench/benchmarks/{bench}/images?limit=1000{query}"))).await
}

// ---------------------------------------------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn benchmarks_from_a_dataset_split_their_images_and_plans() {
    let t = TestDir::new("bench-create");
    let f = fixture(&t.join("home"));
    let app = &f.app;

    let system: Value = ok(app, get("/api/system")).await;
    assert!(system["features"].as_array().unwrap().contains(&json!("vision_bench")), "{system}");
    let catalog: BenchCatalog = ok(app, get("/api/bench/catalog")).await;
    assert_eq!((catalog.conditions.len(), catalog.severities, catalog.axes.len()), (12, 5, 5));
    // The mock has no benchmarks.
    let mock = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    err(&mock, get("/api/bench/benchmarks"), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
    let mock_system: Value = ok(&mock, get("/api/system")).await;
    assert!(!mock_system["features"].as_array().unwrap().contains(&json!("vision_bench")));

    let ds = dataset(app, "shapes", 20).await;
    let src = json!({ "kind": "dataset", "dataset_id": ds.id });
    // Inputs are checked before anything is made.
    for (body, code, status) in [
        (json!({ "name": " ", "source": src }), "invalid_body", StatusCode::BAD_REQUEST),
        (json!({ "name": "b", "source": src, "max_images": 0 }), "invalid_body", StatusCode::BAD_REQUEST),
        (json!({ "name": "b", "source": src, "max_images": 5001 }), "invalid_body", StatusCode::BAD_REQUEST),
        (
            json!({ "name": "b", "source": src, "plan": { "conditions": [{ "condition": "fog", "severities": [1] }] } }),
            "invalid_body",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": src, "description": "x".repeat(4001) }),
            "invalid_body",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id, "split": "test" } }),
            "invalid_dataset",
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": "ds_none" } }),
            "dataset_not_found",
            StatusCode::NOT_FOUND,
        ),
        (
            json!({ "name": "b", "source": { "kind": "label_job", "job_id": "lb_none" } }),
            "label_job_not_found",
            StatusCode::NOT_FOUND,
        ),
    ] {
        err(app, post("/api/bench/benchmarks", body), status, code).await;
    }
    let none: BenchList = ok(app, get("/api/bench/benchmarks")).await;
    assert!(none.benchmarks.is_empty(), "a refused request makes nothing");

    // From the val split (the default), every image, with its truth.
    let val: ImagePage = ok(app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    let b =
        bench(app, json!({ "name": "shapes val", "description": "toy shapes", "source": src, "seed": 3 }))
            .await;
    assert!(b.id.starts_with("bn_"));
    assert_eq!(b.source, BenchSource::Dataset { dataset_id: ds.id.clone(), split: Some(Split::Val) });
    assert_eq!((b.images, b.source_name.as_str(), b.classes.len()), (4, "shapes", 2));
    let truth_boxes: u64 = val.images.iter().map(|i| i.boxes.len() as u64).sum();
    assert_eq!((b.boxes, b.class_boxes.iter().sum::<u64>()), (truth_boxes, truth_boxes));
    assert_eq!(b.plan, folo_bench::default_plan());
    assert_eq!((b.entries, b.best.clone(), b.fingerprint.len()), (0, None, 64));
    let images = page(app, &b.id, "").await;
    assert_eq!(images.images.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), ["1", "2", "3", "4"]);

    // The same source and seed: the same images in the same order and the same fingerprint; another seed is
    // another benchmark.
    let again = bench(app, json!({ "name": "again", "source": src, "seed": 3 })).await;
    assert_eq!(again.fingerprint, b.fingerprint);
    assert_eq!(page(app, &again.id, "").await.images, images.images);
    let other = bench(app, json!({ "name": "other", "source": src, "seed": 4 })).await;
    assert_ne!(other.fingerprint, b.fingerprint);
    // A sample of a larger split: seeded, and a smaller benchmark is the first images of a larger one.
    let train = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "train" });
    let ten = bench(app, json!({ "name": "ten", "source": train, "max_images": 10, "seed": 9 })).await;
    let five = bench(app, json!({ "name": "five", "source": train, "max_images": 5, "seed": 9 })).await;
    assert_eq!((ten.images, five.images), (10, 5));
    let files = |p: BenchImagePage| p.images.into_iter().map(|i| i.file).collect::<Vec<_>>();
    assert_eq!(files(page(app, &ten.id, "").await)[..5], files(page(app, &five.id, "").await));
    let list: BenchList = ok(app, get("/api/bench/benchmarks")).await;
    assert_eq!(list.benchmarks[0].id, five.id, "newest first");

    // Plans: checked, and the fingerprint follows the plan, not the name.
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    let bad = json!({ "plan": { "conditions": [{ "condition": "haze", "severities": [3, 1] }] } });
    err(app, put(&uri, bad), StatusCode::BAD_REQUEST, "invalid_body").await;
    let renamed: Bench = ok(app, put(&uri, json!({ "name": "renamed", "description": "a pond" }))).await;
    assert_eq!((renamed.name.as_str(), renamed.fingerprint.as_str()), ("renamed", b.fingerprint.as_str()));
    let planned: Bench = ok(app, put(&uri, json!({ "plan": small_plan() }))).await;
    assert_ne!(planned.fingerprint, b.fingerprint);
    assert_eq!(planned.plan.conditions.len(), 2);

    // Images: the stored file, the variant every contestant gets, thumbnails; all immutable.
    let truth = page(app, &b.id, "").await;
    let first = &truth.images[0];
    let (headers, stored) = bytes(app, &format!("{uri}/images/1")).await;
    assert!(headers[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    let source_image = val.images.iter().find(|i| i.file.ends_with(&first.file)).unwrap();
    let (_, original) = bytes(app, &format!("/api/datasets/{}/images/{}", ds.id, source_image.id)).await;
    assert_eq!(stored, original, "a benchmark copies its files byte for byte");
    let (headers, haze) = bytes(app, &format!("{uri}/images/1?condition=haze&severity=3")).await;
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
    assert!(headers[header::CACHE_CONTROL].to_str().unwrap().contains("immutable"));
    let seed = folo_bench::conditions::variant_seed(3, "1", folo_bench::Condition::Haze, 3);
    let want =
        folo_bench::conditions::render(&stored, folo_bench::Condition::Haze, 3, &first.boxes, seed).unwrap();
    assert_eq!(haze, want.jpeg, "the variant is the engine's render of the stored file");
    let (_, again_haze) = bytes(app, &format!("{uri}/images/1?condition=haze&severity=3")).await;
    assert_eq!(again_haze, haze);
    for q in
        ["?thumb=32", "?condition=distance&severity=2&thumb=32", "?condition=distance&severity=2&thumb=32"]
    {
        let (h, thumb) = bytes(app, &format!("{uri}/images/1{q}")).await;
        assert!(
            thumb.starts_with(&[0xff, 0xd8])
                && h[header::CACHE_CONTROL].to_str().unwrap().contains("immutable")
        );
    }
    // The truth moves under distance (and only there).
    let far = page(app, &b.id, "&condition=distance&severity=2").await;
    let hazy = page(app, &b.id, "&condition=haze&severity=5").await;
    assert_eq!(hazy.images, truth.images);
    let with_boxes = truth.images.iter().position(|i| !i.boxes.is_empty()).unwrap();
    assert_ne!(far.images[with_boxes].boxes, truth.images[with_boxes].boxes);
    let (_, dist) =
        bytes(app, &format!("{uri}/images/{}?condition=distance&severity=2", truth.images[with_boxes].id))
            .await;
    let im = &truth.images[with_boxes];
    let seed = folo_bench::conditions::variant_seed(3, &im.id, folo_bench::Condition::Distance, 2);
    let (_, file) = bytes(app, &format!("{uri}/images/{}", im.id)).await;
    let r =
        folo_bench::conditions::render(&file, folo_bench::Condition::Distance, 2, &im.boxes, seed).unwrap();
    assert_eq!((dist, r.boxes), (r.jpeg.clone(), far.images[with_boxes].boxes.clone()));
    for q in ["?condition=haze", "?severity=2", "?condition=fog&severity=1", "?condition=haze&severity=9"] {
        err(app, get(&format!("{uri}/images/1{q}")), StatusCode::BAD_REQUEST, "invalid_query").await;
    }
    err(app, get(&format!("{uri}/images/99")), StatusCode::NOT_FOUND, "image_not_found").await;

    // Deleting a benchmark removes it and its folder.
    let (status, _, _) = send(app, delete(&format!("/api/bench/benchmarks/{}", other.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(app, get(&format!("/api/bench/benchmarks/{}", other.id)), StatusCode::NOT_FOUND, "bench_not_found")
        .await;
    assert!(!t.join("home").join("bench").join(&other.id).exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_benchmark_from_a_labeling_job_and_its_events() {
    let t = TestDir::new("bench-job");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let job: LabelJob =
        ok(app, post("/api/label/jobs", json!({ "name": "pond", "classes": ["fish", "frog"] }))).await;
    let ds = dataset(app, "pond-images", 10).await;
    let imgs: ImagePage = ok(app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    // Three images uploaded, two labeled by hand (one of them as a background image), one left pending.
    let boundary = "XfoloBench";
    let mut body = Vec::new();
    for im in &imgs.images[..3] {
        let (_, b) = bytes(app, &format!("/api/datasets/{}/images/{}", ds.id, im.id)).await;
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{}.jpg\"\r\n\r\n",
                im.id
            )
            .as_bytes(),
        );
        body.extend_from_slice(&b);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let req = Request::post(format!("/api/label/jobs/{}/images", job.id))
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let added: LabelImagesAdded = ok(app, req).await;
    let ids: Vec<String> = added.added.iter().map(|i| i.id.clone()).collect();
    let drawn = json!({ "boxes": [
        { "class": 1, "x1": 4, "y1": 4, "x2": 30, "y2": 20 },
        { "class": 0, "x1": 50, "y1": 10, "x2": 60, "y2": 30 } ] });
    ok::<LabelImage>(app, put(&format!("/api/label/jobs/{}/images/{}", job.id, ids[0]), drawn)).await;
    ok::<LabelImage>(
        app,
        put(&format!("/api/label/jobs/{}/images/{}", job.id, ids[2]), json!({ "boxes": [] })),
    )
    .await;

    let b = bench(
        app,
        json!({ "name": "pond", "source": { "kind": "label_job", "job_id": job.id }, "plan": { "conditions": [] } }),
    )
    .await;
    assert_eq!(
        (b.images, b.boxes, b.class_boxes.clone(), b.classes.clone()),
        (2, 2, vec![1, 1], job.classes.clone())
    );
    assert_eq!(b.source_name, "pond");
    let truth = page(app, &b.id, "").await;
    let mut boxes: Vec<BBox> = truth.images.iter().flat_map(|i| i.boxes.clone()).collect();
    boxes.sort_by_key(|b| b.class);
    assert_eq!(boxes[1], BBox { class: 1, x1: 4.0, y1: 4.0, x2: 30.0, y2: 20.0 });
    // A job with nothing labeled is no source.
    let empty: LabelJob = ok(app, post("/api/label/jobs", json!({ "name": "e", "classes": ["a"] }))).await;
    let src = json!({ "kind": "label_job", "job_id": empty.id });
    err(
        app,
        post("/api/bench/benchmarks", json!({ "name": "e", "source": src })),
        StatusCode::BAD_REQUEST,
        "invalid_dataset",
    )
    .await;

    // Events: the benchmark first, then each change in order.
    let mut events = Events::open(app, &b.id).await;
    let (name, first) = events.next().await;
    assert_eq!((name.as_str(), first["id"].as_str()), ("bench", Some(b.id.as_str())));
    let e: BenchEntry = ok(
        app,
        post(
            &format!("/api/bench/benchmarks/{}/entries", b.id),
            external("truth", true, truth_predictions(&truth.images, None, 0.9)),
        ),
    )
    .await;
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["id"].as_str(), data["status"].as_str()),
        ("entry", Some(e.id.as_str()), Some("done"))
    );
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["entries"].as_u64(), data["best"]["entry_id"].as_str()),
        ("bench", Some(1), Some(e.id.as_str()))
    );
    ok::<Bench>(app, put(&format!("/api/bench/benchmarks/{}", b.id), json!({ "name": "pond 2" }))).await;
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["name"].as_str()), ("bench", Some("pond 2")));
    let (status, _, _) = send(app, delete(&format!("/api/bench/benchmarks/{}/entries/{}", b.id, e.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["id"].as_str()), ("removed", Some(e.id.as_str())));
    let (name, data) = events.next().await;
    assert_eq!((name.as_str(), data["entries"].as_u64(), data["best"].is_null()), ("bench", Some(0), true));
}

// ---------------------------------------------------------------------------------------------------------------
// External entries
// ---------------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn external_entries_are_checked_and_scored() {
    let t = TestDir::new("bench-external");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let truth = page(app, &b.id, "").await;
    let images = &truth.images;
    let n_boxes: usize = images.iter().map(|i| i.boxes.len()).sum();
    assert!(n_boxes >= 3, "the synthetic images have boxes");

    // The truth itself, under every planned condition: every axis 1, a score of 100, complete.
    let mut perfect = truth_predictions(images, None, 1.0);
    for (c, s) in [("haze", 1), ("haze", 3), ("distance", 2)] {
        let moved = page(app, &b.id, &format!("&condition={c}&severity={s}")).await;
        perfect.extend(truth_predictions(&moved.images, Some((c, s)), 1.0));
    }
    let p: BenchEntry = ok(app, post(&entries, external("perfect", true, perfect.clone()))).await;
    assert_eq!(
        (p.status, p.contestant.clone(), p.conditions, p.scored),
        (BenchEntryStatus::Done, BenchContestant::External, true, true)
    );
    let score = p.score.clone().unwrap();
    assert!(!score.partial, "{score:?}");
    for a in BenchAxis::ALL {
        assert!((axis(&score, a).unwrap() - 1.0).abs() < 1e-6, "{a:?}: {score:?}");
    }
    assert!((score.total - 100.0).abs() < 1e-3 && (p.map50_95.unwrap() - 1.0).abs() < 1e-6);
    assert_eq!((p.ci95, p.conditions_run), (Some([1.0, 1.0]), 2));
    assert_eq!((p.progress.done, p.progress.total, p.images), (16, 16, 4));
    let r: BenchResult = ok(app, get(&format!("{entries}/{}/result", p.id))).await;
    let rob = r.robustness.clone().unwrap();
    assert_eq!(rob.conditions.iter().map(|c| c.condition.as_str()).collect::<Vec<_>>(), ["haze", "distance"]);
    assert_eq!(rob.conditions[0].points.len(), 2);
    assert_eq!((r.self_labeling.readiness, r.localization.mean_iou), (BenchReadiness::Ready, Some(1.0)));

    // Every box shifted right by a tenth of its width: IoU 0.9 / 1.1 for each, so mAP@0.5:0.95 is 0.7 (matched at
    // 0.5 to 0.8), localization (0.818 - 0.5) / 0.5, calibration 1 - |0.9 - 1|; unchanged images only: partial.
    let shifted: Vec<Value> = images
        .iter()
        .map(|im| {
            let boxes: Vec<Value> = im
                .boxes
                .iter()
                .map(|b| {
                    let d = (b.x2 - b.x1) * 0.1;
                    json!({ "class": b.class, "score": 0.9, "x1": b.x1 + d, "y1": b.y1, "x2": b.x2 + d, "y2": b.y2 })
                })
                .collect();
            json!({ "image": im.id, "boxes": boxes })
        })
        .collect();
    let s: BenchEntry = ok(app, post(&entries, external("shifted", true, shifted))).await;
    let sc = s.score.clone().unwrap();
    let iou = 0.9f32 / 1.1;
    let near = |a: Option<f32>, b: f32| (a.unwrap() - b).abs() < 1e-3;
    assert!(sc.partial && axis(&sc, BenchAxis::Robustness).is_none() && !s.conditions);
    assert_eq!(s.conditions_run, 0);
    let ci = s.ci95.unwrap();
    assert!(ci[0] <= s.map50_95.unwrap() && s.map50_95.unwrap() <= ci[1], "{ci:?}");
    assert!(near(axis(&sc, BenchAxis::Accuracy), 0.7), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::Localization), (iou - 0.5) / 0.5), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::Calibration), 0.9), "{sc:?}");
    assert!(near(axis(&sc, BenchAxis::SelfLabeling), 1.0), "{sc:?}");
    let want = 100.0 * (0.35 * 0.7 + 0.2 * 1.0 + 0.1 * 0.9 + 0.1 * (iou - 0.5) / 0.5) / 0.75;
    assert!((sc.total - want).abs() < 0.1, "{} vs {want}", sc.total);

    // Without confidences, one box missing: self-labeling is the share of boxes found, calibration unmeasured.
    let mut missing = truth_predictions(images, None, 1.0);
    let with = images.iter().position(|i| !i.boxes.is_empty()).unwrap();
    missing[with]["boxes"].as_array_mut().unwrap().remove(0);
    let m: BenchEntry = ok(app, post(&entries, external("missing", false, missing))).await;
    let ms = m.score.clone().unwrap();
    assert!(!m.scored && ms.partial && axis(&ms, BenchAxis::Calibration).is_none());
    assert!(near(axis(&ms, BenchAxis::SelfLabeling), (n_boxes - 1) as f32 / n_boxes as f32), "{ms:?}");
    assert!(near(axis(&ms, BenchAxis::Localization), 1.0));
    let acc = axis(&ms, BenchAxis::Accuracy).unwrap();
    assert!(acc < 1.0 && acc > 0.5, "{acc}");
    let want = 100.0 * (0.35 * acc + 0.2 * (n_boxes - 1) as f32 / n_boxes as f32 + 0.1) / 0.65;
    assert!((ms.total - want).abs() < 1e-3, "{} vs {want}", ms.total);
    // Its gallery: the image with the missing box comes first by `worst`, with one missed.
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images?sort=worst", m.id))).await;
    assert_eq!(
        (gallery.total, gallery.images[0].image.id.as_str(), gallery.images[0].missed),
        (4, images[with].id.as_str(), 1)
    );
    assert!(gallery.images[1..].iter().all(|i| i.missed == 0 && i.wrong == 0));
    let ordered: BenchEntryImagePage =
        ok(app, get(&format!("{entries}/{}/images?limit=2&offset=1", p.id))).await;
    assert_eq!(ordered.images.iter().map(|i| i.image.id.as_str()).collect::<Vec<_>>(), ["2", "3"]);
    err(app, get(&format!("{entries}/{}/images?sort=best", m.id)), StatusCode::BAD_REQUEST, "invalid_query")
        .await;

    // The leaderboard: by score; the benchmark's best is the perfect entry.
    let board: BenchEntryList = ok(app, get(&entries)).await;
    assert_eq!(
        board.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
        ["perfect", "missing", "shifted"]
    );
    let bb: Bench = ok(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await;
    let best = bb.best.unwrap();
    assert_eq!((bb.entries, best.entry_id, best.partial, bb.active), (3, p.id.clone(), false, false));
    // A plan is frozen once there are entries; the name is not.
    let uri = format!("/api/bench/benchmarks/{}", b.id);
    err(app, put(&uri, json!({ "plan": { "conditions": [] } })), StatusCode::CONFLICT, "bench_frozen").await;
    ok::<Bench>(app, put(&uri, json!({ "name": "still mine" }))).await;

    // Every way a submission can be wrong, each named.
    let clean = truth_predictions(images, None, 1.0);
    let with_extra = |extra: Value| {
        let mut v = clean.clone();
        v.push(extra);
        v
    };
    let mut bad_class = clean.clone();
    bad_class[1]["boxes"] = json!([{ "class": 2, "x1": 1, "y1": 1, "x2": 5, "y2": 5 }]);
    let cases: Vec<(Value, &str)> = vec![
        (
            external("x", true, with_extra(json!({ "image": "99", "boxes": [] }))),
            "the benchmark has no image '99'",
        ),
        (
            external(
                "x",
                true,
                with_extra(json!({ "image": "1", "condition": "fog", "severity": 1, "boxes": [] })),
            ),
            "'fog' is not a condition",
        ),
        (
            external(
                "x",
                true,
                with_extra(json!({ "image": "1", "condition": "jpeg", "severity": 1, "boxes": [] })),
            ),
            "does not run jpeg at severity 1",
        ),
        (
            external("x", true, with_extra(json!({ "image": "1", "boxes": [] }))),
            "prediction 5 (image '1', unchanged) repeats prediction 1",
        ),
        (external("x", true, bad_class), "box 1: class 2 is not one of the benchmark's 2 classes"),
        (external("x", true, clean[..3].to_vec()), "no prediction for image '4'"),
        (
            external(
                "x",
                true,
                with_extra(json!({ "image": "1", "condition": "haze", "severity": 1, "boxes": [] })),
            ),
            "condition 'haze' is answered, but not for image '2' at severity 1",
        ),
        (json!({ "contestant": { "kind": "external", "predictions": clean.clone() } }), "name is required"),
        (
            json!({ "name": "x", "max_images": 2, "contestant": { "kind": "external", "predictions": clean.clone() } }),
            "not for an external contestant",
        ),
    ];
    for (body, want) in cases {
        let e = err(app, post(&entries, body), StatusCode::BAD_REQUEST, "invalid_body").await;
        assert!(e.message.contains(want), "{} / {want}", e.message);
    }
    err(app, get(&format!("{entries}/be_none")), StatusCode::NOT_FOUND, "bench_entry_not_found").await;
    err(
        app,
        post(&format!("{entries}/{}/stop", p.id), json!({})),
        StatusCode::CONFLICT,
        "bench_entry_not_active",
    )
    .await;

    // Up to 64 MiB: a submission over the 2 MiB of other requests is taken.
    let mut big = clean.clone();
    let many: Vec<Value> = (0..60_000)
        .map(|i| json!({ "class": 0, "score": 0.01, "x1": i % 50, "y1": 1, "x2": i % 50 + 3, "y2": 9 }))
        .collect();
    big[0]["boxes"] = json!(many);
    let body = external("big", true, big);
    assert!(body.to_string().len() > 2 << 20);
    let e: BenchEntry = ok(app, post(&entries, body)).await;
    assert_eq!(e.status, BenchEntryStatus::Done);
}

// ---------------------------------------------------------------------------------------------------------------
// Model entries
// ---------------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn model_entries_run_on_the_queue_and_agree_with_validation() {
    let t = TestDir::new("bench-model");
    let home = t.join("home");
    let f = fixture(&home);
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let src = json!({ "kind": "dataset", "dataset_id": ds.id });
    // The images in the split's order: an untrained model scores many anchors alike, and AP breaks ties in score
    // by the order the images come in (validation's is the split's).
    let val: ImagePage = ok(app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    let mut seed = 0;
    let b = loop {
        let b = bench(app, json!({ "name": "b", "source": src, "plan": small_plan(), "seed": seed })).await;
        let order: Vec<String> = page(app, &b.id, "").await.images.into_iter().map(|i| i.file).collect();
        if val.images.iter().zip(&order).all(|(v, o)| v.file.ends_with(o.as_str())) {
            break b;
        }
        seed += 1;
        assert!(seed < 500, "no seed keeps the split's order");
    };
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    // A model that finds something, so the comparison with validation means something.
    let m = model_with(app, &home, "shapes-n", &["circle", "square"], 1, true).await;
    let n = b.images;

    let mut events = Events::open(app, &b.id).await;
    assert_eq!(events.next().await.0, "bench");
    let e: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    assert!(e.id.starts_with("be_") && e.status == BenchEntryStatus::Queued, "{e:?}");
    assert_eq!(
        (e.name.as_str(), e.images, e.conditions, e.scored, e.progress.total),
        ("shapes-n", n, true, true, 4 * n)
    );
    assert!(
        matches!(&e.contestant, BenchContestant::Model { model_id, arch, imgsz: 64, .. } if *model_id == m && arch == "folo-n")
    );
    assert!(e.missing_classes.is_empty() && e.fingerprint == b.fingerprint);
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    let score = done.score.clone().unwrap();
    // Every image and condition ran: partial only when no detection reached the calibration floor.
    assert_eq!(score.partial, axis(&score, BenchAxis::Calibration).is_none(), "{score:?}");
    let speed = done.speed.clone().unwrap();
    assert!(speed.device == "cpu" && speed.ms_per_image > 0.0, "{speed:?}");
    assert_eq!((done.progress.done, done.progress.total), (4 * n, 4 * n));
    // Its events: queued, running, progress under a condition, done.
    let (mut seen, mut active) = (Vec::new(), Vec::new());
    loop {
        let (name, data) = events.next().await;
        if name == "bench" {
            active.push(data["active"].as_bool().unwrap());
        }
        if name == "entry" {
            let status = data["status"].as_str().unwrap().to_string();
            seen.push((status.clone(), data["progress"]["condition"].as_str().map(str::to_string)));
            if status == "done" {
                break;
            }
        }
    }
    assert_eq!(seen.first().unwrap().0, "queued");
    // The benchmark is sent as it becomes active, and again as it is idle once more.
    assert_eq!(active, [true]);
    let (name, data) = events.next().await;
    assert_eq!(
        (name.as_str(), data["active"].as_bool(), data["entries"].as_u64()),
        ("bench", Some(false), Some(1))
    );
    assert_eq!((done.conditions_run, done.ci95.is_some()), (2, true));
    assert!(seen.iter().any(|(s, _)| s == "running"), "{seen:?}");

    // The result's shape, and per-image outcomes that agree with it.
    let r: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(r.entry, done);
    assert_eq!(r.robustness.as_ref().unwrap().conditions.len(), 2);
    assert!(r.calibration.is_some() || r.accuracy.map50_95 == 0.0);
    assert_eq!(r.self_labeling.curve.len(), 101);
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images", e.id))).await;
    assert_eq!(gallery.total, n);
    for im in &gallery.images {
        assert!(im.detections.iter().all(|d| d.score >= r.accuracy.best_conf && d.class < 2));
        assert!(im.missed as usize <= im.image.boxes.len());
        let right = im.detections.len() as u32 - im.wrong;
        assert_eq!(right + im.missed, im.image.boxes.len() as u32, "{im:?}");
    }

    // The unchanged images' mAP is what `folo val` gives on the same images.
    let v: ValReport =
        ok(app, post("/api/val", json!({ "model_id": m, "dataset_id": ds.id, "split": "val" }))).await;
    assert_eq!(v.images, n);
    assert!(v.val.map50 > 0.0, "the model finds something: {:?}", v.val);
    assert!(
        (r.accuracy.map50_95 - v.val.map50_95).abs() < 1e-5,
        "{} vs {}",
        r.accuracy.map50_95,
        v.val.map50_95
    );
    assert!((r.accuracy.map50 - v.val.map50).abs() < 1e-5, "{} vs {}", r.accuracy.map50, v.val.map50);

    // Classes match by name without case; a model that knows none is refused.
    let partly = model(app, &home, "squares", &["Square", "triangle"], 2).await;
    let p: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": partly }, "conditions": false, "max_images": 2 }))).await;
    assert_eq!(
        (p.missing_classes.clone(), p.images, p.conditions, p.progress.total),
        (vec!["circle".to_string()], 2, false, 2)
    );
    let pets = model(app, &home, "pets", &["dog", "cat"], 3).await;
    let e2 = err(
        app,
        post(&entries, json!({ "contestant": { "kind": "model", "model_id": pets } })),
        StatusCode::BAD_REQUEST,
        "bench_class_mismatch",
    )
    .await;
    assert!(e2.message.contains("dog"), "{}", e2.message);
    err(
        app,
        post(&entries, json!({ "contestant": { "kind": "model", "model_id": "m_none" } })),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;
    let pd = ended(app, &b.id, &p.id).await;
    assert!(pd.score.as_ref().unwrap().partial, "a part of the images and no conditions: partial");
    let r2: BenchResult = ok(app, get(&format!("{entries}/{}/result", p.id))).await;
    assert!(r2.robustness.is_none());
    assert!(r2.accuracy.per_class.iter().all(|c| c.class < 2));

    // One at a time: a second entry waits behind the first; stopping a waiting entry stops it at once.
    let a: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    let w: BenchEntry =
        ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
    let stopped: BenchEntry = ok(app, post(&format!("{entries}/{}/stop", w.id), json!({}))).await;
    assert!(matches!(stopped.status, BenchEntryStatus::Stopped | BenchEntryStatus::Stopping), "{stopped:?}");
    assert_eq!(ended(app, &b.id, &w.id).await.status, BenchEntryStatus::Stopped);
    err(app, get(&format!("{entries}/{}/result", w.id)), StatusCode::CONFLICT, "bench_entry_not_done").await;
    assert_eq!(ended(app, &b.id, &a.id).await.status, BenchEntryStatus::Done);
}

// ---------------------------------------------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn agent_entries_follow_the_labeling_policy() {
    let t = TestDir::new("bench-agent");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let b = bench(
        app,
        json!({ "name": "b", "description": "Count the shapes.", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() }),
    )
    .await;
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let agent = json!({ "contestant": { "kind": "agent", "agent": "claude" } });

    // Defaults: the first 48 images (here all 4), unchanged only, unscored; usage summed over the calls.
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    assert_eq!((e.name.as_str(), e.images, e.conditions, e.scored), ("Claude Code (small)", 4, false, false));
    assert!(
        matches!(&e.contestant, BenchContestant::Agent { agent: LabelAgentId::Claude, model: Some(m), effort: LabelEffort::Medium } if m == "small")
    );
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    let usage = done.usage.unwrap();
    assert_eq!((usage.calls, usage.input_tokens, usage.output_tokens), (2, 2000, 100));
    let score = done.score.unwrap();
    assert!(
        score.partial
            && axis(&score, BenchAxis::Calibration).is_none()
            && axis(&score, BenchAxis::Robustness).is_none()
    );
    // Asked as agent labeling asks: the classes, the description as the instructions, two images a call.
    let prompts = f.agent.prompts.lock().unwrap().clone();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0], ("Classes: circle, square\nInstructions: Count the shapes.\n2 images follow, in order: image 1 to image 2.".to_string(), 2));
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images", e.id))).await;
    assert!(gallery.images.iter().all(|i| i.detections.len() == 1 && i.detections[0].score == 1.0));

    // With conditions: every variant is a call per batch too.
    let n = f.agent.calls.load(Ordering::SeqCst);
    let c: BenchEntry = ok(app, post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "conditions": true, "max_images": 2 }))).await;
    let cd = ended(app, &b.id, &c.id).await;
    assert_eq!((cd.status, cd.progress.total), (BenchEntryStatus::Done, 8));
    assert_eq!(f.agent.calls.load(Ordering::SeqCst) - n, 4);
    assert!(cd.score.unwrap().partial, "a prefix, without confidences: partial");

    // A CLI that fails ends the entry at once, with its message and the agent's hint.
    f.agent.script(vec![Step::Fail("not signed in: run /login")]);
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    let failed = ended(app, &b.id, &e.id).await;
    assert_eq!(failed.status, BenchEntryStatus::Failed);
    assert_eq!(
        (failed.error.as_deref(), failed.hint.as_deref()),
        (Some("not signed in: run /login"), Some("sign in first"))
    );
    // One image left out of a reply is within the policy (scored as nothing found)...
    f.agent.script(vec![Step::Reply(json!({ "images": [{ "image": 1, "boxes": [] }] }).to_string())]);
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    assert_eq!(ended(app, &b.id, &e.id).await.status, BenchEntryStatus::Done);
    // ... a call whose reply cannot be read even when asked once more is not: two images without an answer.
    f.agent.script(vec![Step::Reply("no".into()), Step::Reply("still no".into())]);
    let two = json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 });
    let e: BenchEntry = ok(app, post(&entries, two)).await;
    let failed = ended(app, &b.id, &e.id).await;
    assert_eq!(failed.status, BenchEntryStatus::Failed);
    assert!(failed.error.as_deref().unwrap().starts_with("2 of 2 images got no usable answer"), "{failed:?}");

    // Stop, delete and their conflicts.
    f.agent.script(vec![Step::Hold, Step::Hold]);
    let h: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    assert_eq!(h.status, BenchEntryStatus::Running);
    let busy: Bench = ok(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await;
    assert!(busy.active, "an entry is running");
    err(app, delete(&format!("{entries}/{}", h.id)), StatusCode::CONFLICT, "bench_entry_active").await;
    err(app, delete(&format!("/api/bench/benchmarks/{}", b.id)), StatusCode::CONFLICT, "bench_active").await;
    err(app, get(&format!("{entries}/{}/result", h.id)), StatusCode::CONFLICT, "bench_entry_not_done").await;
    let s: BenchEntry = ok(app, post(&format!("{entries}/{}/stop", h.id), json!({}))).await;
    assert_eq!(s.status, BenchEntryStatus::Stopping);
    let s = ended(app, &b.id, &h.id).await;
    assert_eq!((s.status, s.score), (BenchEntryStatus::Stopped, None));
    err(
        app,
        post(&format!("{entries}/{}/stop", h.id), json!({})),
        StatusCode::CONFLICT,
        "bench_entry_not_active",
    )
    .await;
    let idle: Bench = ok(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await;
    assert!(!idle.active);
    let (status, _, _) = send(app, delete(&format!("{entries}/{}", h.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(app, get(&format!("{entries}/{}", h.id)), StatusCode::NOT_FOUND, "bench_entry_not_found").await;

    // An agent that is not installed, and bad inputs.
    let codex = json!({ "contestant": { "kind": "agent", "agent": "codex" } });
    err(app, post(&entries, codex), StatusCode::NOT_IMPLEMENTED, "not_supported").await;
    let bad = json!({ "contestant": { "kind": "agent", "agent": "claude", "model": "a b" } });
    err(app, post(&entries, bad), StatusCode::BAD_REQUEST, "invalid_body").await;
    let zero = json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 0 });
    err(app, post(&entries, zero), StatusCode::BAD_REQUEST, "invalid_body").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn designing_a_plan_with_an_agent() {
    let t = TestDir::new("bench-design");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let b = bench(app, json!({ "name": "pond", "description": "A camera under murky water.", "source": { "kind": "dataset", "dataset_id": ds.id } })).await;
    let uri = format!("/api/bench/benchmarks/{}/design", b.id);

    let answer = json!({ "rationale": "Murky water scatters light.", "conditions": [
        { "condition": "haze", "severities": [2, 4], "weight": 3, "why": "turbidity" },
        { "condition": "color_cast", "severities": [1, 3, 5], "weight": 2, "why": "green water" } ] });
    f.agent.script(vec![Step::Reply(format!("Here it is:\n```json\n{answer}\n```"))]);
    let d: BenchDesign = ok(app, post(&uri, json!({ "agent": "claude", "note": "mostly at dusk" }))).await;
    assert_eq!(
        d.plan.designed_by,
        Some(BenchDesigner { agent: LabelAgentId::Claude, model: Some("small".into()) })
    );
    assert_eq!(d.plan.rationale.as_deref(), Some("Murky water scatters light."));
    assert_eq!(
        d.plan.conditions.iter().map(|c| (c.condition.as_str(), c.weight)).collect::<Vec<_>>(),
        [("haze", 3.0), ("color_cast", 2.0)]
    );
    assert_eq!(d.usage.calls, 1);
    let (prompt, images) = f.agent.prompts.lock().unwrap().last().cloned().unwrap();
    assert_eq!(images, 4);
    assert!(prompt.starts_with("About this camera: A camera under murky water.\nFor this request: mostly at dusk\nClasses to find: circle ("), "{prompt}");
    // Nothing is saved; PUT adopts it.
    let now: Bench = ok(app, get(&format!("/api/bench/benchmarks/{}", b.id))).await;
    assert_eq!(now.plan, b.plan);
    let adopted: Bench =
        ok(app, put(&format!("/api/bench/benchmarks/{}", b.id), json!({ "plan": d.plan }))).await;
    assert_eq!(adopted.plan.designed_by.unwrap().agent, LabelAgentId::Claude);

    // A messy answer is cleaned into a valid plan.
    let messy = r#"{"rationale":"x","conditions":[{"condition":"fog"},{"condition":"Haze","severities":[9,1,1],"weight":40},{"condition":"jpeg","weight":0}]}"#;
    f.agent.script(vec![Step::Reply(messy.into())]);
    let d: BenchDesign =
        ok(app, post(&uri, json!({ "agent": "claude", "model": "big", "effort": "high" }))).await;
    let got: Vec<(&str, Vec<u8>, f32)> =
        d.plan.conditions.iter().map(|c| (c.condition.as_str(), c.severities.clone(), c.weight)).collect();
    assert_eq!(got, [("haze", vec![1, 5], 10.0), ("jpeg", vec![1, 3, 5], 0.1)]);
    assert_eq!(d.plan.designed_by.unwrap().model.as_deref(), Some("big"));

    // An unreadable answer is asked for once more; twice is a 502 with the reason. Usage of a failed design is
    // the agent's, not reported.
    f.agent.script(vec![Step::Reply("I would rather not.".into()), Step::Reply(answer.to_string())]);
    let d: BenchDesign = ok(app, post(&uri, json!({ "agent": "claude" }))).await;
    assert_eq!(d.usage.calls, 2);
    let calls = f.agent.calls.load(Ordering::SeqCst);
    f.agent.script(vec![Step::Reply("no".into()), Step::Reply("{\"conditions\":[]}".into())]);
    let e = err(app, post(&uri, json!({ "agent": "claude" })), StatusCode::BAD_GATEWAY, "agent_failed").await;
    assert!(e.message.contains("not a plan") && e.hint.is_some(), "{e:?}");
    assert_eq!(f.agent.calls.load(Ordering::SeqCst) - calls, 2, "asked once more, no more");
    f.agent.script(vec![Step::Fail("rate limited; /login again")]);
    let e = err(app, post(&uri, json!({ "agent": "claude" })), StatusCode::BAD_GATEWAY, "agent_failed").await;
    assert_eq!(
        (e.message.as_str(), e.hint.as_deref()),
        ("rate limited; /login again", Some("sign in first"))
    );
    err(app, post(&uri, json!({ "agent": "codex" })), StatusCode::NOT_IMPLEMENTED, "not_supported").await;
    err(app, post(&uri, json!({ "agent": "robot" })), StatusCode::UNPROCESSABLE_ENTITY, "invalid_body").await;
}

// ---------------------------------------------------------------------------------------------------------------
// Restarts and shutdown
// ---------------------------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn entries_and_loops_left_active_fail_at_the_next_start() {
    let t = TestDir::new("bench-restart");
    let home = t.join("home");
    let (bench_id, entry_id) = {
        let f = fixture(&home);
        let app = &f.app;
        let ds = dataset(app, "shapes", 10).await;
        let b = bench(app, json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() })).await;
        // Shutting down ends an agent call in flight: the entry fails as the server stopped.
        f.agent.script(vec![Step::Hold]);
        let e: BenchEntry = ok(
            app,
            post(
                &format!("/api/bench/benchmarks/{}/entries", b.id),
                json!({ "contestant": { "kind": "agent", "agent": "claude" } }),
            ),
        )
        .await;
        f.store.shutdown();
        let s: BenchEntry = ok(app, get(&format!("/api/bench/benchmarks/{}/entries/{}", b.id, e.id))).await;
        assert_eq!(
            (s.status, s.error.as_deref()),
            (BenchEntryStatus::Failed, Some("the server stopped before the entry finished"))
        );
        (b.id, e.id)
    };
    // A process that died left an entry running and a loop running with a finished generation.
    let dir = home.join("bench").join(&bench_id);
    let path = dir.join("entries").join(format!("{entry_id}.json"));
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["entry"]["status"] = json!("running");
    file["entry"]["error"] = Value::Null;
    file["entry"]["finished"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let lp = json!({ "version": 1, "loop": {
        "id": "bl_1", "bench_id": bench_id, "name": "l", "status": "running", "phase": "training",
        "created": "2026-11-09T00:00:00Z", "finished": null,
        "config": { "teacher": "m_1", "pool": { "kind": "label_job", "job_id": "lb_1" } },
        "generations": [{ "generation": 0, "labels": null, "run_id": null, "model_id": "m_1", "model_name": "t",
                          "entry_id": entry_id, "score": 40.0, "map50_95": 0.3, "ci95": [0.2, 0.4] }],
        "verdict": null, "error": null } });
    std::fs::create_dir_all(dir.join("loops")).unwrap();
    std::fs::write(dir.join("loops").join("bl_1.json"), lp.to_string()).unwrap();
    std::fs::write(dir.join("loops").join("broken.json"), "{").unwrap();

    let f = fixture(&home);
    let e: BenchEntry =
        ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/entries/{entry_id}"))).await;
    assert_eq!(
        (e.status, e.error.as_deref()),
        (BenchEntryStatus::Failed, Some("the server stopped before the entry finished"))
    );
    assert!(e.hint.is_some() && e.finished.is_some());
    let l: BenchLoop = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops/bl_1"))).await;
    assert_eq!(
        (l.status, l.phase, l.error.as_deref()),
        (BenchLoopStatus::Failed, None, Some("the server stopped"))
    );
    assert_eq!(l.generations.len(), 1, "its finished generations are kept");
    let loops: BenchLoopList = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops"))).await;
    assert_eq!(loops.loops.len(), 1, "an unreadable loop file is skipped");
    assert!(dir.join("loops").join("broken.json").is_file(), "and never overwritten");
    // Written back: the next start reads them as failed too.
    let saved: Value =
        serde_json::from_slice(&std::fs::read(dir.join("loops").join("bl_1.json")).unwrap()).unwrap();
    assert_eq!(saved["loop"]["status"], "failed");
}

// ---------------------------------------------------------------------------------------------------------------
// Self-improvement loops
// ---------------------------------------------------------------------------------------------------------------

/// A labeling job whose images are `copies` (bytes) plus the images of `ids` of dataset `ds`, every one labeled
/// by hand with the same wrong box.
async fn pool_job(app: &Router, ds: &Dataset, ids: &[String], copies: &[Vec<u8>]) -> LabelJob {
    let job: LabelJob =
        ok(app, post("/api/label/jobs", json!({ "name": "pool", "classes": ["circle", "square"] }))).await;
    let boundary = "XfoloPool";
    let mut body = Vec::new();
    let mut files: Vec<Vec<u8>> = copies.to_vec();
    for id in ids {
        files.push(bytes(app, &format!("/api/datasets/{}/images/{id}", ds.id)).await.1);
    }
    for (i, b) in files.iter().enumerate() {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"p{i}.jpg\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(b);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    let req = Request::post(format!("/api/label/jobs/{}/images", job.id))
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let added: LabelImagesAdded = ok(app, req).await;
    // A label no teacher would give: if it reached the student's dataset, the pool's own labels were read.
    for im in &added.added {
        let wrong = json!({ "boxes": [{ "class": 1, "x1": 1, "y1": 1, "x2": 7, "y2": 7 }] });
        ok::<LabelImage>(app, put(&format!("/api/label/jobs/{}/images/{}", job.id, im.id), wrong)).await;
    }
    added.job
}

#[tokio::test(flavor = "multi_thread")]
async fn a_loop_teaches_a_student_on_the_teachers_labels_only() {
    let t = TestDir::new("bench-loop");
    let home = t.join("home");
    let f = fixture(&home);
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
    let b = bench(
        app,
        json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": plan }),
    )
    .await;
    let teacher = model(app, &home, "teacher", &["circle", "square"], 7).await;
    let loops = format!("/api/bench/benchmarks/{}/loops", b.id);
    let train: ImagePage = ok(app, get(&format!("/api/datasets/{}/images?split=train&limit=4", ds.id))).await;
    let train_ids: Vec<String> = train.images.iter().map(|i| i.id.clone()).collect();
    let copies = vec![
        bytes(app, &format!("/api/bench/benchmarks/{}/images/1", b.id)).await.1,
        bytes(app, &format!("/api/bench/benchmarks/{}/images/2", b.id)).await.1,
    ];
    let pool = pool_job(app, &ds, &train_ids, &copies).await;
    let student = json!({ "epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu" });

    // Checked up front: the teacher's classes, the pool, the numbers.
    let wrong_order = model(app, &home, "reversed", &["square", "circle"], 8).await;
    let pool_src = json!({ "kind": "label_job", "job_id": pool.id });
    let e = err(
        app,
        post(&loops, json!({ "teacher": wrong_order, "pool": pool_src })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e.message.contains("same names, same order"), "{}", e.message);
    let same_images = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "val" });
    let e = err(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": same_images })),
        StatusCode::BAD_REQUEST,
        "invalid_dataset",
    )
    .await;
    assert!(e.message.contains("every image of the pool"), "{}", e.message);
    err(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "generations": 6 })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(
        app,
        post(&loops, json!({ "teacher": "m_none", "pool": pool_src })),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;

    let started: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 0.001, "student": student, "name": "selfie" })),
    )
    .await;
    assert!(started.id.starts_with("bl_") && started.status == BenchLoopStatus::Running);
    assert_eq!(started.config.generations, Some(1));
    assert_eq!(
        (started.config.student.arch.as_deref(), started.config.student.epochs),
        (Some("folo-n"), Some(1))
    );
    let done = loop_ended(app, &b.id, &started.id).await;
    assert_eq!(done.status, BenchLoopStatus::Done, "{done:?}");
    assert!(done.verdict.is_some() && done.phase.is_none() && done.finished.is_some());
    assert_eq!(done.generations.len(), 2);
    let g0 = &done.generations[0];
    assert_eq!(
        (g0.model_id.as_deref(), g0.labels.clone(), g0.run_id.clone()),
        (Some(teacher.as_str()), None, None)
    );
    assert!(g0.entry_id.is_some() && g0.score.is_some() && g0.map50_95.is_some() && g0.ci95.is_some());
    let g1 = &done.generations[1];
    let labels = g1.labels.clone().unwrap();
    assert_eq!((labels.images, labels.excluded, labels.threshold), (4, 2, 0.001));
    assert!(labels.boxes > 0 && labels.labeled_images > 0);
    assert!(g1.model_id.is_some() && g1.entry_id.is_some() && g1.score.is_some() && g1.ci95.is_some());
    // The student's run is an ordinary run.
    let run: Run = ok(app, get(&format!("/api/runs/{}", g1.run_id.clone().unwrap()))).await;
    assert_eq!(
        (run.status, run.name.as_str(), run.config.dataset_id.as_str()),
        (RunStatus::Completed, "selfie gen 1", labels.dataset_id.as_str())
    );
    assert_eq!(g1.model_id, run.model_id);
    // Its dataset has the teacher's labels and nothing of the pool's own.
    let student_ds: Dataset = ok(app, get(&format!("/api/datasets/{}", labels.dataset_id))).await;
    assert_eq!(
        (student_ds.classes.clone(), student_ds.splits.train + student_ds.splits.val),
        (vec!["circle".to_string(), "square".to_string()], 4)
    );
    let mut lines = Vec::new();
    for e in walk(Path::new(&student_ds.path).join("labels")) {
        lines.extend(std::fs::read_to_string(e).unwrap().lines().map(str::to_string));
    }
    assert_eq!(lines.len() as u64, labels.boxes);
    let wrong = format!("1 {:.6} {:.6} {:.6} {:.6}", 4.0 / 64.0, 4.0 / 64.0, 6.0 / 64.0, 6.0 / 64.0);
    assert!(!lines.contains(&wrong), "the pool's own label reached the student");
    // The teacher's entry was a fresh one, the student's another; both are on the leaderboard.
    let board: BenchEntryList = ok(app, get(&format!("/api/bench/benchmarks/{}/entries", b.id))).await;
    assert_eq!(board.entries.len(), 2);

    // A second loop reuses the teacher's finished entry; stopping it while its student trains stops the run.
    f.store.hold_runs_after_epochs(true);
    let second: BenchLoop = ok(
        app,
        post(&loops, json!({ "teacher": teacher, "pool": pool_src, "min_confidence": 0.001, "student": json!({ "epochs": 3, "imgsz": 64, "batch": 4, "device": "cpu" }) })),
    )
    .await;
    let t0 = std::time::Instant::now();
    let run_id = loop {
        let l: BenchLoop = ok(app, get(&format!("{loops}/{}", second.id))).await;
        if let Some(id) = l.generations.get(1).and_then(|g| g.run_id.clone()) {
            assert_eq!(l.generations[0].entry_id, g0.entry_id, "the teacher's entry is reused");
            break id;
        }
        assert!(t0.elapsed() < TIMEOUT && l.status.is_active(), "{l:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    err(app, delete(&format!("{loops}/{}", second.id)), StatusCode::CONFLICT, "bench_loop_active").await;
    let used = g0.entry_id.clone().unwrap();
    err(
        app,
        delete(&format!("/api/bench/benchmarks/{}/entries/{used}", b.id)),
        StatusCode::CONFLICT,
        "bench_entry_in_loop",
    )
    .await;
    let stopping: BenchLoop = ok(app, post(&format!("{loops}/{}/stop", second.id), json!({}))).await;
    assert_eq!(stopping.status, BenchLoopStatus::Stopping);
    f.store.hold_runs_after_epochs(false);
    let stopped = loop_ended(app, &b.id, &second.id).await;
    assert_eq!((stopped.status, stopped.verdict), (BenchLoopStatus::Stopped, None));
    let run: Run = ok(app, get(&format!("/api/runs/{run_id}"))).await;
    assert_eq!(run.status, RunStatus::Stopped);
    err(
        app,
        post(&format!("{loops}/{}/stop", second.id), json!({})),
        StatusCode::CONFLICT,
        "bench_loop_not_active",
    )
    .await;
    let list: BenchLoopList = ok(app, get(&loops)).await;
    assert_eq!(
        list.loops.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
        [second.id.as_str(), started.id.as_str()]
    );
    let (status, _, _) = send(app, delete(&format!("{loops}/{}", second.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(app, get(&format!("{loops}/{}", second.id)), StatusCode::NOT_FOUND, "bench_loop_not_found").await;
}

/// Every file under `dir`.
fn walk(dir: std::path::PathBuf) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(p));
        } else {
            out.push(p);
        }
    }
    out
}
