//! Agent labeling end to end over HTTP (docs/API.md, Agent labeling), with scripted agents in place of the
//! Claude Code and Codex CLIs: jobs, uploads and folder imports, runs and what they do when calls fail, stopping,
//! hand edits, events, and the dataset a job becomes.

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
use folo_server::{MockStore, Options, RealStore, Server, Store};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;

/// What a scripted agent does on its next call.
enum Step {
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
struct Asked {
    prompt: String,
    images: usize,
    model: Option<String>,
    effort: LabelEffort,
}

/// An agent that follows a script, then labels every image with one box of the first class.
struct Scripted {
    id: LabelAgentId,
    available: bool,
    steps: Mutex<VecDeque<Step>>,
    asked: Mutex<Vec<Asked>>,
    calls: AtomicUsize,
    /// When set, the next probe ([`Agent::info`]) says it has begun (`.0`) and waits to be let go (`.1`).
    slow_probe: Mutex<Option<(Arc<AtomicBool>, Arc<AtomicBool>)>>,
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

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// A reply with one box of `label` per image, at the same relative place in each.
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

struct Fixture {
    app: Router,
    store: Arc<RealStore>,
    agent: Arc<Scripted>,
}

/// A server on `home` whose Claude Code agent follows `steps`.
fn fixture(home: &Path, steps: Vec<Step>) -> Fixture {
    fixture_with(home, steps, |store| store)
}

/// [`fixture`] with the store changed by `tune` before it serves.
fn fixture_with(home: &Path, steps: Vec<Step>, tune: impl FnOnce(RealStore) -> RealStore) -> Fixture {
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

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    (parts.status, body.collect().await.unwrap().to_bytes().to_vec())
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
    let (status, body) = send(app, req).await;
    assert!(status.is_success(), "{uri}: {status} {}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&body)))
}

async fn err(app: &Router, req: Request<Body>, status: StatusCode, code: &str) -> ApiError {
    let uri = format!("{} {}", req.method(), req.uri());
    let (got, body) = send(app, req).await;
    assert_eq!(got, status, "{uri}: {}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, code, "{uri}: {}", e.error.message);
    e.error
}

/// A `width` x `height` PNG whose colour (and so whose bytes) depends on `n`.
fn png(n: u8, width: u32, height: u32) -> Vec<u8> {
    let image = folo_data::Image::new(width, height, [n.wrapping_mul(37), 120, 255 - n]);
    folo_data::encode(&image, folo_data::ImageFormat::Png).unwrap()
}

/// A multipart upload of `files` (name, bytes) as `image` parts.
fn upload(uri: &str, files: &[(&str, Vec<u8>)]) -> Request<Body> {
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
async fn job_with_images(app: &Router, images: u8) -> LabelJob {
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
async fn idle(app: &Router, id: &str) -> LabelJob {
    for _ in 0..2000 {
        let job: LabelJob = ok(app, get(&format!("/api/label/jobs/{id}"))).await;
        if job.status == LabelJobStatus::Idle {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("job {id} did not become idle");
}

async fn images(app: &Router, id: &str, query: &str) -> LabelImagePage {
    ok(app, get(&format!("/api/label/jobs/{id}/images{query}"))).await
}

#[tokio::test]
async fn upload_label_review_and_make_a_dataset() {
    let t = TestDir::new("label-flow");
    let home = t.join("home");
    let f = fixture(&home, vec![]);
    let app = &f.app;

    // The server says it can label, and which agents it found.
    let system: Value = ok(app, get("/api/system")).await;
    assert!(system["features"].as_array().unwrap().contains(&json!("agent_labeling")), "{system}");
    let agents: LabelAgentList = ok(app, get("/api/label/agents")).await;
    assert_eq!(agents.agents.len(), 2);
    assert!(
        agents.agents[0].available && agents.agents[0].version.is_some() && agents.agents[0].hint.is_none()
    );
    assert!(!agents.agents[1].available && agents.agents[1].hint.is_some());

    // A job, with its inputs checked.
    for (bad, field) in [
        (json!({ "name": " ", "classes": ["car"] }), "name"),
        (json!({ "name": "street\npath: D:/other", "classes": ["car"] }), "name"),
        (json!({ "name": "x", "classes": [] }), "classes"),
        (json!({ "name": "x", "classes": ["car", "CAR"] }), "class"),
        (json!({ "name": "x", "classes": ["a,b"] }), "class"),
        (json!({ "name": "x", "classes": ["car"], "instructions": "y".repeat(4001) }), "instructions"),
    ] {
        let e = err(app, post("/api/label/jobs", bad), StatusCode::BAD_REQUEST, "invalid_body").await;
        assert!(e.message.contains(field), "{}", e.message);
    }
    let missing = json!({ "name": "x", "classes": ["car"], "path": t.join("nope").to_string_lossy() });
    err(app, post("/api/label/jobs", missing), StatusCode::NOT_FOUND, "path_not_found").await;
    let none: LabelJobList = ok(app, get("/api/label/jobs")).await;
    assert!(none.jobs.is_empty(), "a refused request creates nothing");

    let job = job_with_images(app, 0).await;
    assert!(job.id.starts_with("lb_"));
    assert_eq!(
        (job.status, job.counts, job.run.clone()),
        (LabelJobStatus::Idle, LabelCounts::default(), None)
    );
    let uri = format!("/api/label/jobs/{}", job.id);

    // Uploads: five images, one of them twice, plus two files that are not images.
    let mut files: Vec<(String, Vec<u8>)> =
        (0..5).map(|n| (format!("shot {n}.png"), png(n, 200, 100))).collect();
    files.push(("copy.png".into(), png(0, 200, 100)));
    files.push(("notes.txt".into(), b"hello".to_vec()));
    files.push(("broken.jpg".into(), b"not a jpeg".to_vec()));
    let files: Vec<(&str, Vec<u8>)> = files.iter().map(|(n, b)| (n.as_str(), b.clone())).collect();
    let added: LabelImagesAdded = ok(app, upload(&format!("{uri}/images"), &files)).await;
    assert_eq!(added.added.len(), 5);
    assert_eq!(added.added[0].file, "shot_0.png");
    assert_eq!((added.added[0].width, added.added[0].height), (200, 100));
    assert_eq!(added.added[0].status, LabelImageStatus::Pending);
    let reasons: Vec<(&str, &str)> =
        added.skipped.iter().map(|s| (s.file.as_str(), s.reason.as_str())).collect();
    assert_eq!(
        reasons,
        [
            ("notes.txt", "not a JPEG, PNG, BMP or WebP file"),
            ("broken.jpg", "not a readable image"),
            ("copy.png", "the job already has this image (shot_0.png)"),
        ]
    );
    assert_eq!(added.job.counts, LabelCounts { images: 5, pending: 5, ..Default::default() });
    // Only the five images are in the job's folder (nothing left of the refused uploads).
    assert_eq!(std::fs::read_dir(home.join("label").join(&job.id).join("images")).unwrap().count(), 5);
    // The same name again gets a suffix.
    let again: LabelImagesAdded =
        ok(app, upload(&format!("{uri}/images"), &[("shot 0.png", png(9, 64, 64))])).await;
    assert_eq!(again.added[0].file, "shot_0-2.png");
    let no_parts = Request::post(format!("{uri}/images"))
        .header(header::CONTENT_TYPE, "multipart/form-data; boundary=X")
        .body(Body::from("--X--\r\n"))
        .unwrap();
    err(app, no_parts, StatusCode::BAD_REQUEST, "invalid_body").await;

    // Images and thumbnails are served.
    let first = &added.added[0].id;
    let (status, bytes) = send(app, get(&format!("{uri}/images/{first}"))).await;
    assert_eq!((status, bytes), (StatusCode::OK, png(0, 200, 100)));
    let (status, thumb) = send(app, get(&format!("{uri}/images/{first}?thumb=64"))).await;
    assert_eq!(status, StatusCode::OK);
    let thumb = folo_data::decode(&thumb).unwrap();
    assert_eq!((thumb.width, thumb.height), (64, 32));
    err(app, get(&format!("{uri}/images/999")), StatusCode::NOT_FOUND, "image_not_found").await;

    // An agent run: 6 images in calls of 2 (the agent's default batch).
    let started: LabelJob = ok(
        app,
        post(&format!("{uri}/start"), json!({ "agent": "claude", "effort": "high", "note": "Tight boxes." })),
    )
    .await;
    assert_eq!(started.status, LabelJobStatus::Running);
    let run = started.run.unwrap();
    assert_eq!((run.agent, run.model.as_deref(), run.total), (LabelAgentId::Claude, Some("small"), 6));
    let done = idle(app, &job.id).await;
    let run = done.run.clone().unwrap();
    assert_eq!((run.outcome, run.done, run.failed, run.error), (Some(LabelOutcome::Completed), 6, 0, None));
    assert!(run.finished.is_some());
    assert_eq!(done.counts, LabelCounts { images: 6, pending: 0, labeled: 6, failed: 0, boxes: 6 });
    let usage = LabelUsage {
        calls: 3,
        input_tokens: 3000,
        cached_input_tokens: 1200,
        output_tokens: 150,
        cost_usd: Some(0.03),
    };
    assert_eq!(run.usage.calls, 3);
    assert!((run.usage.cost_usd.unwrap() - 0.03).abs() < 1e-9);
    assert_eq!(LabelUsage { cost_usd: usage.cost_usd, ..done.usage }, usage);

    // What the agent was asked: the classes, the job's instructions, the run's note, two images a call.
    let asked = f.agent.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 3);
    for a in &asked {
        assert_eq!((a.images, a.model.as_deref(), a.effort), (2, Some("small"), LabelEffort::High));
        assert!(
            a.prompt.starts_with("Classes: car, person\nInstructions: Only parked cars.\n"),
            "{}",
            a.prompt
        );
        assert!(a.prompt.contains("For this request: Tight boxes."), "{}", a.prompt);
    }

    // The labels are in pixels of each image.
    let page = images(app, &job.id, "?status=labeled").await;
    assert_eq!((page.total, page.images.len()), (6, 6));
    assert_eq!(page.images[0].source, Some(LabelSource::Agent));
    assert_eq!(page.images[0].boxes, [BBox { class: 0, x1: 50.0, y1: 10.0, x2: 150.0, y2: 60.0 }]);
    assert_eq!(page.images[5].boxes, [BBox { class: 0, x1: 16.0, y1: 6.4, x2: 48.0, y2: 38.4 }]);
    assert_eq!(images(app, &job.id, "?status=pending").await.total, 0);
    assert_eq!(images(app, &job.id, "?offset=4&limit=5").await.images.len(), 2);
    err(app, get(&format!("{uri}/images?status=done")), StatusCode::BAD_REQUEST, "invalid_query").await;
    // Nothing is left to label, unless everything is labeled again.
    let e = err(
        app,
        post(&format!("{uri}/start"), json!({ "agent": "claude" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e.message.contains("every image is labeled") && e.hint.unwrap().contains("scope"));

    // A hand edit: clipped to the image, the image now labeled by hand; a bad box is refused.
    let second = &page.images[1].id;
    let boxes = json!({ "boxes": [{ "class": 1, "x1": 150, "y1": 20, "x2": 260, "y2": 80 }] });
    let edited: LabelImage = ok(app, put(&format!("{uri}/images/{second}"), boxes)).await;
    assert_eq!(edited.source, Some(LabelSource::Manual));
    assert_eq!(edited.boxes, [BBox { class: 1, x1: 150.0, y1: 20.0, x2: 200.0, y2: 80.0 }]);
    let bad_class = json!({ "boxes": [{ "class": 2, "x1": 1, "y1": 1, "x2": 9, "y2": 9 }] });
    err(app, put(&format!("{uri}/images/{second}"), bad_class), StatusCode::BAD_REQUEST, "invalid_body")
        .await;
    // A background image: labeled, without boxes.
    let third = &page.images[2].id;
    let empty: LabelImage = ok(app, put(&format!("{uri}/images/{third}"), json!({ "boxes": [] }))).await;
    assert_eq!((empty.status, empty.boxes.len()), (LabelImageStatus::Labeled, 0));
    // An image removed (its uploaded file too).
    let last = &page.images[5];
    let (status, _) = send(app, delete(&format!("{uri}/images/{}", last.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!home.join("label").join(&job.id).join("images").join(&last.file).exists());
    err(app, delete(&format!("{uri}/images/{}", last.id)), StatusCode::NOT_FOUND, "image_not_found").await;

    // Classes can be renamed and appended once there are boxes, not removed.
    err(app, put(&uri, json!({ "classes": ["car"] })), StatusCode::BAD_REQUEST, "invalid_body").await;
    let renamed: LabelJob =
        ok(app, put(&uri, json!({ "name": "Street 2", "classes": ["car", "pedestrian", "bus"] }))).await;
    assert_eq!((renamed.name.as_str(), renamed.classes.len()), ("Street 2", 3));
    assert_eq!(renamed.instructions, "Only parked cars.");

    // The dataset: 5 labeled images, 1 of them (20%) in val, registered and readable like any other.
    err(
        app,
        post(&format!("{uri}/dataset"), json!({ "val_fraction": 0.9 })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    let (status, body) = send(app, post(&format!("{uri}/dataset"), json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&body));
    let ds: Dataset = serde_json::from_slice(&body).unwrap();
    assert_eq!((ds.name.as_str(), ds.format), ("Street 2", DatasetFormat::Yolo));
    assert_eq!(ds.classes, ["car", "pedestrian", "bus"]);
    assert_eq!(ds.splits, Splits { train: 4, val: 1, test: 0 });
    assert_eq!(Path::new(&ds.path), home.join("datasets").join("street-2"));
    let stats = ds.stats.unwrap();
    assert_eq!((stats.boxes, stats.empty_images, stats.class_counts), (4, 1, vec![3, 1, 0]));
    let in_dataset: ImagePage = ok(app, get(&format!("/api/datasets/{}/images", ds.id))).await;
    assert_eq!(in_dataset.total, 5);
    let hand = in_dataset.images.iter().find(|i| i.file.ends_with("shot_1.png")).unwrap();
    assert_eq!(hand.boxes.len(), 1);
    let b = hand.boxes[0];
    assert_eq!(b.class, 1);
    assert!(
        (b.x1 - 150.0).abs() < 0.01 && (b.x2 - 200.0).abs() < 0.01 && (b.y2 - 80.0).abs() < 0.01,
        "{b:?}"
    );
    // The job remembers it; a second export is another dataset.
    let after: LabelJob = ok(app, get(&uri)).await;
    assert_eq!(after.dataset_id.as_deref(), Some(ds.id.as_str()));
    let second_ds: Dataset =
        ok(app, post(&format!("{uri}/dataset"), json!({ "name": "Street 2", "seed": 3 }))).await;
    assert_eq!(Path::new(&second_ds.path), home.join("datasets").join("street-2-2"));

    // The log tells the story, and all of it survives a restart.
    let logs: RunLogs = ok(app, get(&format!("{uri}/logs"))).await;
    let text: Vec<&str> = logs.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(
        text[0].starts_with("Claude Code (small, high effort) starts on 6 images: 3 calls of up to 2"),
        "{text:?}"
    );
    assert!(text.iter().any(|m| m.starts_with("call 1: 2 images labeled, 2 boxes · 1000 tokens in, 50 out")));
    assert!(
        text.iter().any(|m| m.starts_with("Claude Code is done: 6 of 6 images labeled, 0 failed · 3 calls"))
    );
    assert!(
        text.last().unwrap().starts_with("dataset 'Street 2' made: 4 train and 1 val images"),
        "{text:?}"
    );
    drop(f);
    let f = fixture(&home, vec![]);
    let jobs: LabelJobList = ok(&f.app, get("/api/label/jobs")).await;
    assert_eq!(jobs.jobs.len(), 1);
    let kept = &jobs.jobs[0];
    assert_eq!((kept.name.as_str(), &kept.classes, kept.counts), ("Street 2", &after.classes, after.counts));
    assert_eq!(kept.dataset_id.as_deref(), Some(second_ds.id.as_str()));
    assert_eq!(kept.run.as_ref().unwrap().outcome, Some(LabelOutcome::Completed));
    assert_eq!(kept.usage.calls, 3);
    assert_eq!(images(&f.app, &job.id, "").await.total, 5);
    let logs_again: RunLogs = ok(&f.app, get(&format!("{uri}/logs?limit=2"))).await;
    assert_eq!(logs_again.lines.len(), 2);

    // Deleting the job removes its folder and thumbnails; the datasets stay.
    let (status, _) = send(&f.app, delete(&uri)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!home.join("label").join(&job.id).exists() && !home.join("cache/thumbs").join(&job.id).exists());
    err(&f.app, get(&uri), StatusCode::NOT_FOUND, "label_job_not_found").await;
    assert!(Path::new(&ds.path).join("data.yaml").is_file());
}

#[tokio::test]
async fn folder_images_are_read_in_place() {
    let t = TestDir::new("label-folder");
    let photos = t.join("photos");
    std::fs::create_dir_all(photos.join("day 2")).unwrap();
    std::fs::write(photos.join("b.png"), png(1, 80, 60)).unwrap();
    std::fs::write(photos.join("a.png"), png(2, 80, 60)).unwrap();
    std::fs::write(photos.join("day 2").join("a.png"), png(3, 80, 60)).unwrap();
    std::fs::write(photos.join("day 2").join("readme.md"), "not an image").unwrap();
    std::fs::write(photos.join("cut.jpg"), b"\xff\xd8 truncated").unwrap();
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;

    let body = json!({ "name": "photos", "classes": ["cat"], "path": photos.to_string_lossy() });
    let job: LabelJob = ok(app, post("/api/label/jobs", body)).await;
    assert_eq!(job.counts.images, 3);
    let page = images(app, &job.id, "").await;
    let files: Vec<&str> = page.images.iter().map(|i| i.file.as_str()).collect();
    // In path order; a second `a.png` gets a suffix so the dataset's label files do not collide.
    assert_eq!(files, ["a.png", "b.png", "a-2.png"]);
    // Nothing was copied into the job's folder.
    assert!(!t.join("home/label").join(&job.id).join("images").exists());
    let (status, bytes) =
        send(app, get(&format!("/api/label/jobs/{}/images/{}", job.id, page.images[2].id))).await;
    assert_eq!((status, bytes), (StatusCode::OK, png(3, 80, 60)));

    // Adding the folder again adds nothing; one new file is picked up.
    std::fs::write(photos.join("c.png"), png(4, 80, 60)).unwrap();
    let uri = format!("/api/label/jobs/{}/images", job.id);
    let again: LabelImagesAdded = ok(app, post(&uri, json!({ "path": photos.to_string_lossy() }))).await;
    assert_eq!(again.added.iter().map(|i| i.file.as_str()).collect::<Vec<_>>(), ["c.png"]);
    let skipped: Vec<&str> = again.skipped.iter().map(|s| s.reason.as_str()).collect();
    assert_eq!(skipped.iter().filter(|r| **r == "the job already has this file").count(), 3);
    assert_eq!(skipped.iter().filter(|r| **r == "not a readable image").count(), 1);
    err(
        app,
        post(&uri, json!({ "path": t.join("nope").to_string_lossy() })),
        StatusCode::NOT_FOUND,
        "path_not_found",
    )
    .await;

    // Labeled and exported: the dataset has copies; deleting an image or the job leaves the folder alone.
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), json!({ "agent": "claude" })))
        .await;
    idle(app, &job.id).await;
    let ds: Dataset =
        ok(app, post(&format!("/api/label/jobs/{}/dataset", job.id), json!({ "val_fraction": 0 }))).await;
    assert_eq!(ds.splits, Splits { train: 4, val: 0, test: 0 });
    assert!(Path::new(&ds.path).join("images/train/a-2.png").is_file());
    assert!(Path::new(&ds.path).join("labels/train/a-2.txt").is_file());
    send(app, delete(&format!("/api/label/jobs/{}/images/{}", job.id, page.images[0].id))).await;
    send(app, delete(&format!("/api/label/jobs/{}", job.id))).await;
    assert!(photos.join("a.png").is_file() && photos.join("day 2").join("a.png").is_file());
}

#[tokio::test]
async fn replies_that_cannot_be_read_fail_their_images_only() {
    let t = TestDir::new("label-bad-replies");
    // Call 1: garbage twice (the retry), so its two images fail. Call 2: one image left out, one box of an unknown
    // class and one good box. Call 3 (the retry of the failed ones, later): fine.
    let partial = json!({ "images": [{ "image": 2, "boxes": [
        { "label": "truck", "box": [0, 0, 500, 500] },
        { "label": "Person", "box": [500, 500, 1000, 1000] },
    ] }] });
    let steps = vec![
        Step::Reply("I'm sorry, I can't see any images.".into()),
        Step::Reply("{\"oops\": true}".into()),
        Step::Reply(format!("```json\n{partial}\n```")),
    ];
    let f = fixture(&t.join("home"), steps);
    let app = &f.app;
    let job = job_with_images(app, 4).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    let body = json!({ "agent": "claude", "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("{uri}/start"), body)).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.outcome, run.done, run.failed), (Some(LabelOutcome::Completed), 1, 3));
    // Every call that reached the model is counted, the retry too.
    assert_eq!((run.usage.calls, f.agent.calls()), (3, 3));
    assert_eq!(done.counts, LabelCounts { images: 4, pending: 0, labeled: 1, failed: 3, boxes: 1 });

    let page = images(app, &job.id, "").await;
    let state: Vec<(LabelImageStatus, usize)> =
        page.images.iter().map(|i| (i.status, i.boxes.len())).collect();
    use LabelImageStatus::{Failed, Labeled};
    assert_eq!(state, [(Failed, 0), (Failed, 0), (Failed, 0), (Labeled, 1)]);
    assert!(page.images[0].error.as_deref().unwrap().starts_with("the reply is not the JSON asked for"));
    assert_eq!(page.images[2].error.as_deref(), Some("the agent's reply left this image out"));
    // The label matched without case; the unknown one was dropped.
    assert_eq!(page.images[3].boxes, [BBox { class: 1, x1: 100.0, y1: 50.0, x2: 200.0, y2: 100.0 }]);
    let logs: RunLogs = ok(app, get(&format!("{uri}/logs"))).await;
    let text: Vec<&str> = logs.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(
        text.iter().any(|m| m.starts_with("call 1: the reply has no JSON object; asking once more")),
        "{text:?}"
    );
    assert!(text.iter().any(|m| m.starts_with("call 2: 1 box dropped")), "{text:?}");

    // The next run takes the failed images (not the labeled one) and clears their errors.
    let again: LabelJob = ok(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    assert_eq!(again.run.unwrap().total, 3);
    let done = idle(app, &job.id).await;
    assert_eq!(done.counts, LabelCounts { images: 4, pending: 0, labeled: 4, failed: 0, boxes: 4 });
    assert!(images(app, &job.id, "").await.images.iter().all(|i| i.error.is_none()));
    // Usage adds up over the runs.
    assert_eq!((done.usage.calls, done.run.unwrap().usage.calls), (5, 2));
}

#[tokio::test]
async fn a_cli_error_ends_the_run_before_more_calls_are_made() {
    let t = TestDir::new("label-cli-error");
    let steps = vec![Step::Reply(boxes_reply(1, "car")), Step::Fail("Invalid API key · Please run /login")];
    let f = fixture(&t.join("home"), steps);
    let app = &f.app;
    let job = job_with_images(app, 6).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("{uri}/start"), body)).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!(run.outcome, Some(LabelOutcome::Failed));
    assert_eq!(run.error.as_deref(), Some("Invalid API key · Please run /login"));
    assert_eq!(run.hint.as_deref(), Some("sign in first"));
    // Two calls were made, not six: the other images wait, untouched, for the next run.
    assert_eq!((f.agent.calls(), run.done, run.failed, run.total), (2, 1, 0, 6));
    assert_eq!(done.counts, LabelCounts { images: 6, pending: 5, labeled: 1, failed: 0, boxes: 1 });
    let logs: RunLogs = ok(app, get(&format!("{uri}/logs"))).await;
    let last = logs.lines.last().unwrap();
    assert_eq!(last.level, LogLevel::Error);
    assert!(last.message.starts_with("Claude Code stopped: Invalid API key"), "{}", last.message);

    // Signed in again, the next run finishes the job.
    let again: LabelJob = ok(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    assert_eq!(again.run.unwrap().total, 5);
    assert_eq!(idle(app, &job.id).await.counts.labeled, 6);
}

#[tokio::test]
async fn an_agent_that_only_returns_garbage_is_given_up_on_after_three_calls() {
    let t = TestDir::new("label-garbage");
    let garbage = || Step::Reply("Sure! Here are the boxes you asked for.".into());
    let f = fixture(&t.join("home"), vec![garbage(), garbage(), garbage(), garbage(), garbage(), garbage()]);
    let app = &f.app;
    let job = job_with_images(app, 6).await;

    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), body)).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    // Three calls, not three batches of two: the first image's call and its retry, then the second image's.
    assert_eq!((run.outcome, f.agent.calls(), run.usage.calls), (Some(LabelOutcome::Failed), 3, 3));
    assert!(
        run.error.unwrap().starts_with("3 calls in a row failed; the last one: the reply has no JSON object")
    );
    assert_eq!(done.counts, LabelCounts { images: 6, pending: 4, labeled: 0, failed: 2, boxes: 0 });
}

#[tokio::test]
async fn a_job_deleted_while_its_start_waits_for_the_agent_does_not_run() {
    let t = TestDir::new("label-start-delete");
    let f = fixture(&t.join("home"), vec![]);
    let job = job_with_images(&f.app, 2).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    // The start request finds the job, then asks the agent whether it is there, which takes a while.
    let (begun, go) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    *f.agent.slow_probe.lock().unwrap() = Some((begun.clone(), go.clone()));
    let (app, start) = (f.app.clone(), format!("{uri}/start"));
    let starting = tokio::spawn(async move { send(&app, post(&start, json!({ "agent": "claude" }))).await });
    for _ in 0..2000 {
        if begun.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(begun.load(Ordering::SeqCst));
    // Meanwhile the job is deleted.
    let (status, _) = send(&f.app, delete(&uri)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    go.store(true, Ordering::SeqCst);

    // The start finds no job, and no call is ever made for it.
    let (status, body) = starting.await.unwrap();
    assert_eq!(status, StatusCode::NOT_FOUND, "{}", String::from_utf8_lossy(&body));
    let e: ErrorBody = serde_json::from_slice(&body).unwrap();
    assert_eq!(e.error.code, "label_job_not_found");
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(f.agent.calls(), 0);
    err(&f.app, post(&format!("{uri}/stop"), json!({})), StatusCode::NOT_FOUND, "label_job_not_found").await;
}

#[tokio::test]
async fn a_run_saves_its_labels_while_later_calls_are_still_out() {
    let t = TestDir::new("label-save-during-run");
    let home = t.join("home");
    let gate = Arc::new(AtomicBool::new(false));
    // A run writes `job.json` at most every 5 s; here every 50 ms.
    let steps = vec![Step::Reply(boxes_reply(1, "car")), Step::Until(gate.clone())];
    let f = fixture_with(&home, steps, |store| store.label_save_every(Duration::from_millis(50)));
    let job = job_with_images(&f.app, 2).await;
    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(&f.app, post(&format!("/api/label/jobs/{}/start", job.id), body)).await;

    // The first image is labeled; the second call does not come back. Soon the job's file has the first image's
    // labels, so a crash during the long call would not lose them.
    let file = home.join("label").join(&job.id).join("job.json");
    let saved = || {
        let on_disk: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        on_disk["images"][0]["status"] == "labeled" && on_disk["run"]["usage"]["calls"] == 1
    };
    let mut waited = Duration::ZERO;
    while !saved() && waited < Duration::from_secs(20) {
        tokio::time::sleep(Duration::from_millis(10)).await;
        waited += Duration::from_millis(10);
    }
    assert!(saved(), "job.json was not written while the run waited on a call");
    assert_eq!(f.agent.calls(), 2);
    gate.store(true, Ordering::SeqCst);
    assert_eq!(idle(&f.app, &job.id).await.counts.labeled, 2);
}

#[tokio::test]
async fn what_a_crash_left_in_a_job_folder_is_swept_on_start() {
    let t = TestDir::new("label-sweep");
    let home = t.join("home");
    let f = fixture(&home, vec![]);
    let job = job_with_images(&f.app, 2).await;
    drop(f);
    // An upload that was being staged, one that was stored but never recorded, and a call's scratch folder.
    let dir = home.join("label").join(&job.id);
    std::fs::write(dir.join("images").join(".upload-1-7"), b"half an image").unwrap();
    std::fs::write(dir.join("images").join("never-recorded.png"), png(9, 8, 8)).unwrap();
    std::fs::create_dir_all(dir.join("tmp").join("call-3")).unwrap();
    std::fs::write(dir.join("tmp").join("call-3").join("1.jpg"), b"x").unwrap();

    let f = fixture(&home, vec![]);
    let mut left: Vec<String> = std::fs::read_dir(dir.join("images"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, ["img0.png", "img1.png"]);
    assert!(!dir.join("tmp").exists());
    // The job's own images are as they were.
    let page = images(&f.app, &job.id, "").await;
    assert_eq!(page.total, 2);
    let (status, bytes) =
        send(&f.app, get(&format!("/api/label/jobs/{}/images/{}", job.id, page.images[1].id))).await;
    assert_eq!((status, bytes), (StatusCode::OK, png(1, 200, 100)));
}

#[tokio::test]
async fn calls_that_time_out_in_a_row_end_the_run() {
    let t = TestDir::new("label-timeouts");
    let f = fixture(&t.join("home"), vec![Step::Timeout, Step::Timeout, Step::Timeout]);
    let app = &f.app;
    let job = job_with_images(app, 5).await;

    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), body)).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.outcome, f.agent.calls()), (Some(LabelOutcome::Failed), 3));
    assert!(
        run.error.unwrap().starts_with("3 calls in a row failed; the last one: Claude Code did not answer")
    );
    assert_eq!(done.counts, LabelCounts { images: 5, pending: 2, labeled: 0, failed: 3, boxes: 0 });
}

#[tokio::test]
async fn stopping_ends_the_calls_and_guards_images_in_the_run() {
    let t = TestDir::new("label-stop");
    let f = fixture(&t.join("home"), vec![Step::Reply(boxes_reply(1, "car")), Step::Hold]);
    let app = &f.app;
    let job = job_with_images(app, 4).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    let all = images(app, &job.id, "").await.images;
    err(app, post(&format!("{uri}/stop"), json!({})), StatusCode::CONFLICT, "label_job_not_active").await;

    // The run labels images 1 to 3; the fourth is not part of it.
    let ids: Vec<&str> = all[..3].iter().map(|i| i.id.as_str()).collect();
    let body = json!({ "agent": "claude", "image_ids": ids, "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("{uri}/start"), body)).await;
    // Wait for the second call, which holds.
    for _ in 0..2000 {
        if f.agent.calls() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let page = images(app, &job.id, "").await.images;
    use LabelImageStatus::{Labeled, Labeling, Pending};
    assert_eq!(page.iter().map(|i| i.status).collect::<Vec<_>>(), [Labeled, Labeling, Pending, Pending]);
    assert_eq!(images(app, &job.id, "?status=labeling").await.total, 1);

    // While it runs: the job and the image in the call are locked. A finished image and one outside the run can
    // be edited; so can one that still waits, which then leaves the run. Images can be added.
    err(
        app,
        post(&format!("{uri}/start"), json!({ "agent": "claude" })),
        StatusCode::CONFLICT,
        "label_job_active",
    )
    .await;
    err(app, put(&uri, json!({ "name": "x" })), StatusCode::CONFLICT, "label_job_active").await;
    err(app, delete(&uri), StatusCode::CONFLICT, "label_job_active").await;
    err(app, post(&format!("{uri}/dataset"), json!({})), StatusCode::CONFLICT, "label_job_active").await;
    let boxes = json!({ "boxes": [] });
    let held = format!("{uri}/images/{}", page[1].id);
    err(app, put(&held, boxes.clone()), StatusCode::CONFLICT, "label_job_active").await;
    err(app, delete(&held), StatusCode::CONFLICT, "label_job_active").await;
    ok::<LabelImage>(app, put(&format!("{uri}/images/{}", page[0].id), boxes.clone())).await;
    ok::<LabelImage>(app, put(&format!("{uri}/images/{}", page[3].id), boxes.clone())).await;
    let waiting: LabelImage = ok(app, put(&format!("{uri}/images/{}", page[2].id), boxes)).await;
    assert_eq!((waiting.status, waiting.source), (Labeled, Some(LabelSource::Manual)));
    let running: LabelJob = ok(app, get(&uri)).await;
    assert_eq!(running.run.unwrap().total, 2, "the image labeled by hand left the run");
    let added: LabelImagesAdded =
        ok(app, upload(&format!("{uri}/images"), &[("late.png", png(77, 50, 50))])).await;
    assert_eq!(added.job.status, LabelJobStatus::Running);

    let stopping: LabelJob = ok(app, post(&format!("{uri}/stop"), json!({}))).await;
    assert_eq!(stopping.status, LabelJobStatus::Stopping);
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.outcome, run.done, run.failed, run.total), (Some(LabelOutcome::Stopped), 1, 0, 2));
    // The held call was ended, not waited for, and no third one was made.
    assert_eq!(f.agent.calls(), 2);
    let page = images(app, &job.id, "").await.images;
    assert_eq!(
        page.iter().map(|i| i.status).collect::<Vec<_>>(),
        [Labeled, Pending, Labeled, Labeled, Pending]
    );
    assert!(!t.join("home/label").join(&job.id).join("tmp").exists(), "scratch folders are removed");
}

#[tokio::test]
async fn images_changed_by_hand_while_they_wait_leave_the_run() {
    let t = TestDir::new("label-withdraw");
    let gate = Arc::new(AtomicBool::new(false));
    let f = fixture(&t.join("home"), vec![Step::Until(gate.clone())]);
    let app = &f.app;
    let job = job_with_images(app, 4).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    let all = images(app, &job.id, "").await.images;

    // One image a call, one call at a time: the first call waits at the gate, three images wait for theirs.
    ok::<LabelJob>(
        app,
        post(&format!("{uri}/start"), json!({ "agent": "claude", "batch": 1, "parallel": 1 })),
    )
    .await;
    for _ in 0..2000 {
        if f.agent.calls() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The second is labeled by hand, the third removed: neither is the agent's any more.
    let boxes = json!({ "boxes": [{ "class": 1, "x1": 10, "y1": 10, "x2": 60, "y2": 60 }] });
    ok::<LabelImage>(app, put(&format!("{uri}/images/{}", all[1].id), boxes)).await;
    let (status, _) = send(app, delete(&format!("{uri}/images/{}", all[2].id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    gate.store(true, Ordering::SeqCst);

    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.outcome, run.total, run.done, run.failed), (Some(LabelOutcome::Completed), 2, 2, 0));
    // Two calls, for the first and the fourth image: the hand-drawn box was not sent anywhere, nor replaced.
    assert_eq!(f.agent.calls(), 2);
    let page = images(app, &job.id, "").await.images;
    let sources: Vec<Option<LabelSource>> = page.iter().map(|i| i.source).collect();
    assert_eq!(sources, [Some(LabelSource::Agent), Some(LabelSource::Manual), Some(LabelSource::Agent)]);
    assert_eq!(page[1].boxes, [BBox { class: 1, x1: 10.0, y1: 10.0, x2: 60.0, y2: 60.0 }]);
}

/// When `job.json` cannot be written the request answers 500, and the job stays as it was: a change that is not
/// on disk would be shown until a restart and then vanish.
#[tokio::test]
async fn a_change_that_cannot_be_saved_is_undone() {
    let t = TestDir::new("label-save-fails");
    let home = t.join("home");
    let f = fixture(&home, Vec::new());
    let app = &f.app;
    let job = job_with_images(app, 1).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    let before = images(app, &job.id, "").await;
    let first = before.images[0].clone();
    // A folder where job.json goes: writing it fails.
    let dir = home.join("label").join(&job.id);
    let file = dir.join("job.json");
    std::fs::remove_file(&file).unwrap();
    std::fs::create_dir_all(file.join("blocked")).unwrap();

    let boxes = json!({ "boxes": [{ "class": 1, "x1": 10, "y1": 10, "x2": 60, "y2": 60 }] });
    err(
        app,
        put(&format!("{uri}/images/{}", first.id), boxes.clone()),
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal",
    )
    .await;
    let files = [("new1.png", png(7, 200, 100)), ("new2.png", png(8, 200, 100))];
    err(app, upload(&format!("{uri}/images"), &files), StatusCode::INTERNAL_SERVER_ERROR, "internal").await;
    let after = images(app, &job.id, "").await;
    assert_eq!(after.total, 1, "the images that were not saved are not listed");
    assert_eq!(after.images[0].status, first.status);
    assert!(after.images[0].boxes.is_empty() && after.images[0].source.is_none(), "{:?}", after.images[0]);
    let counts: LabelJob = ok(app, get(&uri)).await;
    assert_eq!((counts.counts.images, counts.counts.labeled), (1, 0));
    let stored: Vec<_> =
        std::fs::read_dir(dir.join("images")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert_eq!(stored.len(), 1, "the files of the images that were not saved are gone: {stored:?}");

    // Once job.json can be written again, the same requests work, and the ids go on where they were.
    std::fs::remove_dir_all(&file).unwrap();
    let added: LabelImagesAdded = ok(app, upload(&format!("{uri}/images"), &files)).await;
    let ids: Vec<_> = added.added.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids, ["2", "3"]);
    let image: LabelImage = ok(app, put(&format!("{uri}/images/{}", first.id), boxes)).await;
    assert_eq!(image.source, Some(LabelSource::Manual));
}

#[tokio::test]
async fn events_follow_a_run() {
    let t = TestDir::new("label-events");
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;
    let job = job_with_images(app, 2).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    // Over HTTP the stream opens with the job.
    let res = app.clone().oneshot(get(&format!("{uri}/events"))).await.unwrap();
    assert_eq!(res.headers()[header::CONTENT_TYPE], "text/event-stream");
    let mut body = res.into_body();
    let first =
        String::from_utf8(body.frame().await.unwrap().unwrap().into_data().unwrap().to_vec()).unwrap();
    assert!(first.starts_with("event: job\ndata: {"), "{first}");
    err(app, get("/api/label/jobs/lb_nope/events"), StatusCode::NOT_FOUND, "label_job_not_found").await;

    let (snapshot, mut events) = f.store.labeler().unwrap().subscribe(&job.id).unwrap();
    assert_eq!(snapshot.counts.pending, 2);
    ok::<LabelJob>(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    idle(app, &job.id).await;
    let page = images(app, &job.id, "").await.images;
    ok::<LabelImage>(app, put(&format!("{uri}/images/{}", page[0].id), json!({ "boxes": [] }))).await;
    send(app, delete(&format!("{uri}/images/{}", page[1].id))).await;

    let mut seen = Vec::new();
    while let Ok(event) = events.try_recv() {
        seen.push(match &event {
            LabelEvent::Job(j) => format!("job {:?} {}/{}", j.status, j.counts.labeled, j.counts.images),
            LabelEvent::Image(i) => format!("image {} {:?}", i.id, i.status),
            LabelEvent::Removed(r) => format!("removed {}", r.id),
            LabelEvent::Log(_) => "log".to_string(),
            LabelEvent::Reload => "reload".to_string(),
        });
    }
    seen.retain(|e| e != "log");
    assert_eq!(
        seen,
        [
            "job Running 0/2",
            "image 1 Labeling",
            "image 2 Labeling",
            "image 1 Labeled",
            "image 2 Labeled",
            "job Running 2/2",
            "job Idle 2/2",
            "image 1 Labeled",
            "job Idle 2/2",
            "removed 2",
            "job Idle 1/1",
        ]
    );
}

#[tokio::test]
async fn many_images_at_once_send_one_reload() {
    let t = TestDir::new("label-bulk");
    let photos = t.join("photos");
    std::fs::create_dir_all(&photos).unwrap();
    let one = png(1, 16, 16);
    for n in 0..120 {
        std::fs::write(photos.join(format!("{n:03}.png")), &one).unwrap();
    }
    let f = fixture(&t.join("home"), vec![]);
    let job = job_with_images(&f.app, 0).await;
    let (_, mut events) = f.store.labeler().unwrap().subscribe(&job.id).unwrap();
    let uri = format!("/api/label/jobs/{}/images", job.id);
    let added: LabelImagesAdded = ok(&f.app, post(&uri, json!({ "path": photos.to_string_lossy() }))).await;
    assert_eq!(added.added.len(), 120);
    assert_eq!(events.try_recv().unwrap(), LabelEvent::Reload);
    assert!(matches!(events.try_recv().unwrap(), LabelEvent::Job(j) if j.counts.images == 120));
    assert!(events.try_recv().is_err());
}

#[tokio::test]
async fn start_refuses_what_it_cannot_run() {
    let t = TestDir::new("label-start-errors");
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;
    let job = job_with_images(app, 1).await;
    let start = format!("/api/label/jobs/{}/start", job.id);

    err(app, post(&start, json!({ "agent": "gemini" })), StatusCode::UNPROCESSABLE_ENTITY, "invalid_body")
        .await;
    let e = err(app, post(&start, json!({ "agent": "codex" })), StatusCode::NOT_IMPLEMENTED, "not_supported")
        .await;
    assert_eq!(
        (e.message.as_str(), e.hint.as_deref()),
        ("Codex is not installed on the server's machine", Some("install it"))
    );
    for bad in [
        json!({ "agent": "claude", "model": "x; rm -rf /" }),
        json!({ "agent": "claude", "model": "m".repeat(101) }),
        json!({ "agent": "claude", "batch": 0 }),
        json!({ "agent": "claude", "batch": 17 }),
        json!({ "agent": "claude", "parallel": 9 }),
        json!({ "agent": "claude", "image_ids": ["404"] }),
        json!({ "agent": "claude", "note": "n".repeat(2001) }),
    ] {
        err(app, post(&start, bad), StatusCode::BAD_REQUEST, "invalid_body").await;
    }
    err(
        app,
        post("/api/label/jobs/lb_nope/start", json!({ "agent": "claude" })),
        StatusCode::NOT_FOUND,
        "label_job_not_found",
    )
    .await;
    assert_eq!(f.agent.calls(), 0);
    // Nothing labeled yet: no dataset.
    let dataset = format!("/api/label/jobs/{}/dataset", job.id);
    err(app, post(&dataset, json!({})), StatusCode::BAD_REQUEST, "invalid_dataset").await;
    err(app, post(&dataset, json!({ "name": "a\nb" })), StatusCode::BAD_REQUEST, "invalid_body").await;

    // A model named in the request reaches the agent; an empty job has nothing to label.
    ok::<LabelJob>(app, post(&start, json!({ "agent": "claude", "model": "claude-opus-5-5[1m]" }))).await;
    idle(app, &job.id).await;
    // ... and without an effort the agent's default is used.
    let asked = f.agent.asked.lock().unwrap()[0].clone();
    assert_eq!((asked.model.as_deref(), asked.effort), (Some("claude-opus-5-5[1m]"), LabelEffort::Medium));
    let empty = job_with_images(app, 0).await;
    let e = err(
        app,
        post(&format!("/api/label/jobs/{}/start", empty.id), json!({ "agent": "claude" })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e.message.contains("the job has no images"), "{}", e.message);
    // `scope: all` labels the labeled image again.
    let all: LabelJob = ok(app, post(&start, json!({ "agent": "claude", "scope": "all" }))).await;
    assert_eq!(all.run.unwrap().total, 1);
    idle(app, &job.id).await;
}

#[tokio::test]
async fn the_mock_server_has_no_agent_labeling() {
    let app = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let system: Value = ok(&app, get("/api/system")).await;
    assert!(!system["features"].as_array().unwrap().contains(&json!("agent_labeling")));
    for uri in ["/api/label/agents", "/api/label/jobs", "/api/label/jobs/lb_1", "/api/label/jobs/lb_1/events"]
    {
        err(&app, get(uri), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
    }
    let body = json!({ "name": "x", "classes": ["a"] });
    err(&app, post("/api/label/jobs", body), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
}

#[tokio::test]
async fn a_run_cut_short_by_a_restart_is_closed_as_stopped() {
    let t = TestDir::new("label-restart");
    let home = t.join("home");
    let f = fixture(&home, vec![Step::Hold]);
    let job = job_with_images(&f.app, 2).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    ok::<LabelJob>(&f.app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    // The job file as a killed server would leave it: a run without an end.
    let saved = std::fs::read(home.join("label").join(&job.id).join("job.json")).unwrap();
    f.store.shutdown();
    std::fs::write(home.join("label").join(&job.id).join("job.json"), saved).unwrap();
    drop(f);

    let f = fixture(&home, vec![]);
    let job: LabelJob = ok(&f.app, get(&uri)).await;
    let run = job.run.unwrap();
    assert_eq!((job.status, run.outcome), (LabelJobStatus::Idle, Some(LabelOutcome::Stopped)));
    assert!(run.finished.is_some());
    assert_eq!(job.counts.pending, 2);
    let logs: RunLogs = ok(&f.app, get(&format!("{uri}/logs"))).await;
    assert!(logs.lines.last().unwrap().message.starts_with("the server stopped during the agent run"));
}
