//! The labeling tools end to end with a scripted agent: in process on a real store, and through a `folo serve` on
//! loopback (proxy mode), with the same calls; plus `--confine` and argument checks.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use folo_api::{LabelAgent, LabelAgentId, LabelEffort, LabelModel, LabelUsage};
use folo_server::label::{Agent, AgentError, AgentReply, AgentRequest};
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::{Progress, find};
use crate::proxy::HttpStore;
use crate::server::FoloMcp;
use crate::{Options, errors::Error};

/// An agent that labels every image with one `car` box at the same relative place.
struct Scripted {
    calls: AtomicUsize,
}

impl Agent for Scripted {
    fn id(&self) -> LabelAgentId {
        LabelAgentId::Claude
    }

    fn info(&self) -> LabelAgent {
        LabelAgent {
            id: LabelAgentId::Claude,
            name: "Claude Code".into(),
            available: true,
            version: Some("1.0.0".into()),
            models: vec![LabelModel { id: "small".into(), label: "Small".into() }],
            default_model: Some("small".into()),
            default_effort: LabelEffort::Medium,
            hint: None,
        }
    }

    fn default_batch(&self) -> usize {
        2
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let images: Vec<Value> = (1..=req.images.len())
            .map(|i| json!({ "image": i, "boxes": [{ "label": "car", "box": [250, 100, 750, 600] }] }))
            .collect();
        let usage = LabelUsage { calls: 1, input_tokens: 600, output_tokens: 20, ..LabelUsage::default() };
        Ok(AgentReply { text: json!({ "images": images }).to_string(), usage })
    }
}

fn store(home: &Path) -> Arc<RealStore> {
    let agent = Arc::new(Scripted { calls: AtomicUsize::new(0) });
    Arc::new(RealStore::open(home.to_path_buf()).label_agents(vec![agent]))
}

fn options(data_dir: &Path, confine: bool) -> Options {
    Options {
        data_dir: data_dir.to_path_buf(),
        read_only: false,
        confine,
        device: "cpu".into(),
        published_weights: None,
    }
}

/// Five 64 × 48 PNGs in `dir`.
fn images(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    for n in 0..5u8 {
        let img = image::RgbImage::from_pixel(64, 48, image::Rgb([n * 40, 120, 200]));
        img.save(dir.join(format!("{n}.png"))).unwrap();
    }
}

/// A tool call as `tools/call` makes it: the argument checks, the handler, and the structured result with the
/// text and the other content.
async fn call(s: &FoloMcp, tool: &str, args: Value) -> Result<(Value, String, Vec<Value>), Error> {
    let def = find(tool).unwrap();
    let args = args.as_object().unwrap().clone();
    let o = s.inner.call(def, args, &mut Progress::none()).await?;
    let blocks = o.links.iter().map(|b| serde_json::to_value(b).unwrap()).collect();
    Ok((o.value, o.summary, blocks))
}

async fn ok(s: &FoloMcp, tool: &str, args: Value) -> Value {
    match call(s, tool, args).await {
        Ok((v, _, _)) => v,
        Err(e) => panic!("{tool}: {} {}", e.body.code, e.body.message),
    }
}

async fn code(s: &FoloMcp, tool: &str, args: Value) -> String {
    call(s, tool, args).await.err().unwrap_or_else(|| panic!("{tool} should fail")).body.code
}

/// The whole flow on `server`: a job from a folder, a run, the review (a view, a patch, the rest marked right),
/// and the dataset.
async fn flow(s: &FoloMcp, folder: &Path) {
    let agents = ok(s, "list_label_agents", json!({})).await;
    assert_eq!(agents["agents"][0]["id"], "claude");
    assert_eq!(agents["agents"][0]["models"], json!(["small"]));

    let folder = folo_api::json::path(folder);
    let made = ok(
        s,
        "create_label_job",
        json!({ "name": "street", "classes": ["car", "person"], "paths": [folder] }),
    )
    .await;
    let job = made["job_id"].as_str().unwrap().to_string();
    assert_eq!(
        (made["images"].as_u64(), made["added"].as_u64(), made["skipped"].as_u64()),
        (Some(5), Some(5), Some(0))
    );
    // The same folder again adds nothing and says why, briefly.
    let again = ok(s, "add_label_images", json!({ "job_id": job, "paths": [folder] })).await;
    assert_eq!((again["added"].as_u64(), again["skipped"].as_u64()), (Some(0), Some(5)));
    assert_eq!(again["skipped_first"].as_array().unwrap().len(), 5);

    // Ask about two images first, then the rest.
    let started =
        ok(s, "start_labeling", json!({ "job_id": job, "agent": "claude", "image_ids": ["1", "2"] })).await;
    assert_eq!((started["status"].as_str(), started["run"]["total"].as_u64()), (Some("running"), Some(2)));
    wait_idle(s, &job).await;
    ok(s, "start_labeling", json!({ "job_id": job, "agent": "claude", "note": "cars only" })).await;
    let status = wait_idle(s, &job).await;
    assert_eq!(status["counts"]["labeled"], 5);
    assert_eq!(
        (status["reviewed"].as_u64(), status["run"]["outcome"].as_str()),
        (Some(0), Some("completed"))
    );
    assert_eq!(status["usage"]["calls"], 3);
    assert!(status["log"].as_array().unwrap().len() <= 5);

    let page = ok(s, "list_label_images", json!({ "job_id": job, "reviewed": false, "limit": 2 })).await;
    assert_eq!((page["total"].as_u64(), page["items"].as_array().unwrap().len()), (Some(5), 2));
    assert_eq!(page["next_cursor"], "o2");
    assert!(page["items"][0].get("boxes").is_none(), "boxes only when asked");
    let page =
        ok(s, "list_label_images", json!({ "job_id": job, "include_boxes": true, "cursor": "o4" })).await;
    assert_eq!(page["items"][0]["boxes"], json!([[0, 250, 100, 750, 600]]));

    // The next image to review, as a JPEG with its boxes drawn, and the box list as text.
    let (v, text, blocks) = call(s, "view_label_image", json!({ "job_id": job })).await.unwrap();
    assert_eq!((v["image"]["id"].as_str(), v["to_review"].as_u64()), (Some("1"), Some(5)));
    assert_eq!(v["classes"], json!(["car", "person"]));
    assert!(text.contains("1 car 250,100,750,600") && text.contains("Classes: 0 car, 1 person"), "{text}");
    assert_eq!(blocks.len(), 1);
    assert_eq!(
        (blocks[0]["type"].as_str(), blocks[0]["mimeType"].as_str()),
        (Some("image"), Some("image/jpeg"))
    );
    let jpeg = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        blocks[0]["data"].as_str().unwrap(),
    )
    .unwrap();
    // Never larger than the image: 64 × 48.
    let img = image::load_from_memory(&jpeg).unwrap();
    assert_eq!((img.width(), img.height()), (64, 48));

    // Corrections by number keep the other boxes' pixels; the image counts as edited by hand and reviewed.
    let fixed = ok(
        s,
        "set_label_boxes",
        json!({ "job_id": job, "image_id": "1", "relabel": [[1, 1]], "add": [[0, 0, 0, 500, 500]] }),
    )
    .await;
    assert_eq!(fixed["boxes"], json!([[1, 250, 100, 750, 600], [0, 0, 0, 500, 500]]));
    assert_eq!((fixed["source"].as_str(), fixed["reviewed"].as_bool()), (Some("manual"), Some(true)));
    let fixed = ok(s, "set_label_boxes", json!({ "job_id": job, "image_id": "1", "remove": [2] })).await;
    assert_eq!(fixed["boxes"].as_array().unwrap().len(), 1);
    for (args, want) in [
        (json!({ "job_id": job, "image_id": "1" }), "invalid_arguments"),
        (json!({ "job_id": job, "image_id": "1", "boxes": [], "remove": [1] }), "invalid_arguments"),
        (json!({ "job_id": job, "image_id": "1", "remove": [9] }), "invalid_arguments"),
        (json!({ "job_id": job, "image_id": "1", "boxes": [[0, 600, 0, 500, 10]] }), "invalid_arguments"),
        (json!({ "job_id": job, "image_id": "1", "boxes": [[7, 0, 0, 500, 500]] }), "invalid_arguments"),
        (json!({ "job_id": job, "image_id": "99", "boxes": [] }), "image_not_found"),
    ] {
        assert_eq!(code(s, "set_label_boxes", args.clone()).await, want, "{args}");
    }

    // "Looks right" for the rest, in one call; an unknown image is reported, not fatal.
    let r =
        ok(s, "review_label_images", json!({ "job_id": job, "image_ids": ["2", "3", "4", "5", "77"] })).await;
    assert_eq!((r["updated"].as_u64(), r["to_review"].as_u64()), (Some(4), Some(0)));
    assert_eq!(r["failed"][0]["code"], "image_not_found");
    let (v, text, blocks) = call(s, "view_label_image", json!({ "job_id": job })).await.unwrap();
    assert!(v["image"].is_null() && blocks.is_empty() && text.contains("nothing left to review"), "{text}");
    let status = ok(s, "label_status", json!({ "job_id": job, "log_lines": 0 })).await;
    assert_eq!((status["reviewed"].as_u64(), status["log"].as_array().unwrap().len()), (Some(5), 0));

    let ds = ok(s, "export_label_dataset", json!({ "job_id": job, "val_fraction": 0.2 })).await;
    assert_eq!((ds["train"].as_u64(), ds["val"].as_u64()), (Some(4), Some(1)));
    let jobs = ok(s, "list_label_jobs", json!({})).await;
    assert_eq!(jobs["items"][0]["dataset_id"], ds["dataset_id"]);

    assert_eq!(code(s, "label_status", json!({ "job_id": "lb_nope" })).await, "label_job_not_found");
    assert_eq!(code(s, "view_label_image", json!({ "job_id": job, "size": 99 })).await, "invalid_arguments");
}

/// Polls `label_status` until the job is idle.
async fn wait_idle(s: &FoloMcp, job: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let st = ok(s, "label_status", json!({ "job_id": job })).await;
        if st["status"] == "idle" {
            return st;
        }
        assert!(Instant::now() < deadline, "the run never ended: {st}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn labeling_in_process() {
    let home = TestDir::new("mcp-label");
    images(&home.join("photos"));
    let store = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    flow(&s, &home.join("photos")).await;
    store.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn labeling_through_a_server() {
    let home = TestDir::new("mcp-label-proxy");
    images(&home.join("photos"));
    let store = store(&home.join("server"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
    let serving = tokio::spawn(server.serve(listener, async move {
        let _ = stopped.await;
    }));
    let http = Arc::new(HttpStore::new(&url, None));
    let s = FoloMcp::new(http.clone(), Some(http), options(&home.join("client"), false));
    flow(&s, &home.join("photos")).await;

    // A server without the image filters: the same answers, filtered here.
    let old = Arc::new(HttpStore::new(&url, None).assume_features(vec![]));
    let o = FoloMcp::new(old.clone(), Some(old), options(&home.join("client"), false));
    let job = ok(&s, "list_label_jobs", json!({})).await["items"][0]["id"].clone();
    for (tool, args) in [
        ("label_status", json!({ "job_id": job, "log_lines": 0 })),
        ("list_label_images", json!({ "job_id": job, "status": "labeled", "reviewed": false, "limit": 2 })),
        ("list_label_images", json!({ "job_id": job, "reviewed": true, "cursor": "o1" })),
        ("view_label_image", json!({ "job_id": job, "image_id": "2" })),
    ] {
        assert_eq!(ok(&o, tool, args.clone()).await, ok(&s, tool, args).await, "{tool}");
    }
    let view = json!({ "job_id": job, "image_id": "nope" });
    assert_eq!(code(&o, "view_label_image", view).await, "image_not_found");
    let _ = stop.send(());
    serving.await.unwrap().unwrap();
}

/// With `--confine`, nothing outside the data dir is added, and a job made earlier with outside images is neither
/// labeled nor exported (its images would be read).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn confine_keeps_labeling_inside_the_data_dir() {
    let home = TestDir::new("mcp-label-confine");
    let data = home.join("data");
    images(&data.join("photos"));
    images(&home.join("outside"));
    let store = store(&data);
    // A job made before, without --confine.
    let open = FoloMcp::new(store.clone(), None, options(&data, false));
    let outside = folo_api::json::path(home.join("outside"));
    let old =
        ok(&open, "create_label_job", json!({ "name": "old", "classes": ["car"], "paths": [outside] })).await;
    let old = old["job_id"].as_str().unwrap().to_string();

    let s = FoloMcp::new(store.clone(), None, options(&data, true));
    let job = json!({ "name": "x", "classes": ["car"], "paths": [outside] });
    assert_eq!(code(&s, "create_label_job", job).await, "path_outside_data_dir");
    let job = json!({ "name": "x", "classes": ["car"], "paths": ["../outside/0.png"] });
    assert_eq!(code(&s, "create_label_job", job).await, "path_outside_data_dir");
    // Relative to the data dir: inside.
    let made =
        ok(&s, "create_label_job", json!({ "name": "in", "classes": ["car"], "paths": ["photos"] })).await;
    assert_eq!(made["images"], 5);
    let start = json!({ "job_id": old, "agent": "claude" });
    assert_eq!(code(&s, "start_labeling", start).await, "path_outside_data_dir");
    assert_eq!(code(&s, "export_label_dataset", json!({ "job_id": old })).await, "path_outside_data_dir");
    let view = json!({ "job_id": old, "image_id": "1" });
    assert_eq!(code(&s, "view_label_image", view).await, "path_outside_data_dir");
    store.shutdown();
}

/// A path that is not there makes no job; one after a good path names the job that was made.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_path_leaves_no_empty_job() {
    let home = TestDir::new("mcp-label-missing");
    images(&home.join("photos"));
    let store = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let missing = folo_api::json::path(home.join("nope"));
    let job = json!({ "name": "x", "classes": ["car"], "paths": [missing] });
    assert_eq!(code(&s, "create_label_job", job).await, "path_not_found");
    assert_eq!(ok(&s, "list_label_jobs", json!({})).await["total"], 0);
    let photos = folo_api::json::path(home.join("photos"));
    let job = json!({ "name": "x", "classes": ["car"], "paths": [photos, missing] });
    let e = call(&s, "create_label_job", job).await.err().unwrap();
    assert_eq!(e.body.code, "path_not_found");
    assert!(e.body.hint.unwrap().contains("add_label_images"));
    assert_eq!(ok(&s, "list_label_jobs", json!({})).await["items"][0]["counts"]["images"], 5);
    store.shutdown();
}
