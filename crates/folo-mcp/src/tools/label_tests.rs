//! The labeling tools end to end with a scripted agent: in process on a real store, and through a `folo serve` on
//! loopback (proxy mode, and as an older server without the image filters and revisions), with the same calls;
//! plus revisions against a run, `--confine` and argument checks.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
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

/// An agent that labels every image with one `car` box at the same place of the 64 x 48 test images, in pixels as
/// sent (the whole image when the run's note says `wide`). With a gate, each call first waits for a message on it.
struct Scripted {
    calls: AtomicUsize,
    gate: Option<Mutex<mpsc::Receiver<()>>>,
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
            models: vec![LabelModel { id: "small".into(), label: "Small".into(), fast: false }],
            default_model: Some("small".into()),
            default_effort: LabelEffort::Medium,
            fast: false,
            hint: None,
        }
    }

    fn default_batch(&self) -> usize {
        2
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        if let Some(gate) = &self.gate {
            let _ = gate.lock().unwrap().recv_timeout(Duration::from_secs(20));
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        let b = if req.prompt.contains("wide") {
            json!(["car", 0, 0, 64, 48])
        } else {
            json!(["car", 16, 4.8, 48, 28.8])
        };
        let images: Vec<Value> =
            (1..=req.images.len()).map(|i| json!({ "image": i, "boxes": [b] })).collect();
        let usage = LabelUsage { calls: 1, input_tokens: 600, output_tokens: 20, ..LabelUsage::default() };
        Ok(AgentReply { text: json!({ "images": images }).to_string(), usage })
    }
}

fn store_with(home: &Path, gate: Option<mpsc::Receiver<()>>) -> Arc<RealStore> {
    let agent = Arc::new(Scripted { calls: AtomicUsize::new(0), gate: gate.map(Mutex::new) });
    Arc::new(RealStore::open(home.to_path_buf()).label_agents(vec![agent]))
}

fn store(home: &Path) -> Arc<RealStore> {
    store_with(home, None)
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

/// The revision image `image_id` of `job` has now.
async fn revision(s: &FoloMcp, job: &str, image_id: &str) -> String {
    let v = ok(s, "view_label_image", json!({ "job_id": job, "image_id": image_id })).await;
    v["image"]["revision"].as_str().unwrap().to_string()
}

/// The whole flow on `server`: a job from a folder, a run, the review (a view, edits, the rest marked right),
/// and the dataset.
async fn flow(s: &FoloMcp, folder: &Path) {
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
    // The same folder again, to the job, adds nothing and says why, briefly.
    let again = ok(s, "create_label_job", json!({ "job_id": job, "paths": [folder] })).await;
    assert_eq!((again["added"].as_u64(), again["skipped"].as_u64()), (Some(0), Some(5)));
    assert_eq!(again["skipped_first"].as_array().unwrap().len(), 5);
    let both = json!({ "job_id": job, "name": "x", "paths": [folder] });
    assert_eq!(code(s, "create_label_job", both).await, "invalid_arguments");

    // Ask about two images first, then the rest.
    let started =
        ok(s, "start_labeling", json!({ "job_id": job, "agent": "claude", "image_ids": ["1", "2"] })).await;
    assert_eq!(started, json!({ "job_id": job, "total": 2, "model": "small" }));
    wait_idle(s, &job).await;
    ok(s, "start_labeling", json!({ "job_id": job, "agent": "claude", "note": "cars only" })).await;
    let status = wait_idle(s, &job).await;
    assert_eq!(status["counts"]["labeled"], 5);
    assert_eq!(
        (status["reviewed"].as_u64(), status["run"]["outcome"].as_str()),
        (Some(0), Some("completed"))
    );
    // The tokens are the result's own fields; the run carries no error or timestamps.
    assert_eq!((status["calls"].as_u64(), status["input_tokens"].as_u64()), (Some(3), Some(1800)));
    assert!(status["run"].get("error").is_none() && status["run"].get("started").is_none());
    assert!(status["log"].as_array().unwrap().len() <= 5);
    // Nothing left to label: the hint names the tools.
    let e = call(s, "start_labeling", json!({ "job_id": job, "agent": "claude" })).await.err().unwrap();
    assert_eq!(e.body.code, "invalid_arguments");
    assert!(e.body.hint.as_deref().unwrap().contains("scope: \"all\""), "{e:?}");

    let page = ok(s, "list_label_images", json!({ "job_id": job, "reviewed": false, "limit": 2 })).await;
    assert_eq!((page["total"].as_u64(), page["items"].as_array().unwrap().len()), (Some(5), 2));
    assert_eq!(page["next_cursor"], "o2");
    assert!(page["items"][0].get("boxes").is_none(), "boxes only when asked");
    let page =
        ok(s, "list_label_images", json!({ "job_id": job, "include_boxes": true, "cursor": "o4" })).await;
    // 64 × 48 is its own view (never upscaled): the agent's box is 16, 4.8, 48, 28.8 in pixels.
    assert_eq!(page["items"][0]["size"], json!([64, 48]));
    assert_eq!(page["items"][0]["boxes"], json!([[0, 16, 5, 48, 29]]));

    // The next image to review, as a JPEG with its boxes drawn, and the box list as text.
    let (v, text, blocks) = call(s, "view_label_image", json!({ "job_id": job })).await.unwrap();
    assert_eq!((v["image"]["id"].as_str(), v["to_review"].as_u64()), (Some("1"), Some(5)));
    assert_eq!(v["classes"], json!(["car", "person"]));
    assert_eq!((v["image"]["size"].clone(), v["image"]["num_boxes"].as_u64()), (json!([64, 48]), Some(1)));
    let rev = v["image"]["revision"].as_str().unwrap().to_string();
    assert_eq!(page["items"][0]["revision"].as_str().unwrap().split('-').next(), Some("1"));
    for want in ["View 64x48 px", "1 car 16,5,48,29", "Classes: 0 car, 1 person", rev.as_str()] {
        assert!(text.contains(want), "{want}: {text}");
    }
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

    // Corrections by number, on the revision seen, keep the other boxes' pixels; the image counts as edited by
    // hand and reviewed. `size` gives the frame of the coordinates: [0, 0, 16, 12] of a 32 × 24 view.
    let edit = json!({
        "job_id": job, "image_id": "1", "expect": rev, "size": [32, 24],
        "relabel": [[1, 1]], "add": [[0, 0, 0, 16, 12]]
    });
    let fixed = ok(s, "set_label_boxes", edit.clone()).await;
    assert_eq!(fixed["boxes"], json!([[1, 8, 2, 24, 14], [0, 0, 0, 16, 12]]));
    assert_eq!((fixed["source"].as_str(), fixed["reviewed"].as_bool()), (Some("manual"), Some(true)));
    // The same edit again is on an old revision.
    assert_eq!(code(s, "set_label_boxes", edit).await, "label_boxes_changed");
    let rev = fixed["revision"].as_str().unwrap().to_string();
    assert_ne!(rev, revision(s, &job, "2").await);
    // Boxes sent back as the view shows them keep their stored pixels: the revision does not change.
    let shown = ok(s, "view_label_image", json!({ "job_id": job, "image_id": "1" })).await;
    let same = json!({
        "job_id": job, "image_id": "1", "expect": rev, "size": shown["image"]["size"], "boxes": shown["image"]["boxes"]
    });
    assert_eq!(ok(s, "set_label_boxes", same).await["revision"].as_str(), Some(rev.as_str()));
    let fixed =
        ok(s, "set_label_boxes", json!({ "job_id": job, "image_id": "1", "expect": rev, "remove": [2] }))
            .await;
    assert_eq!(fixed["boxes"], json!([[1, 16, 5, 48, 29]]));
    let rev = fixed["revision"].as_str().unwrap().to_string();
    for (args, want) in [
        (json!({ "image_id": "1" }), "invalid_arguments"),
        (json!({ "image_id": "1", "boxes": [], "remove": [1], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "remove": [1] }), "invalid_arguments"),
        (json!({ "image_id": "1", "remove": [9], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "relabel": [[1, 0]], "remove": [1], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "relabel": [[1, 0], [1, 1]], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "relabel": [[1, 5]], "expect": rev }), "invalid_arguments"),
        (
            json!({ "image_id": "1", "boxes": [[0, 60, 0, 50, 10]], "size": [64, 48], "expect": rev }),
            "invalid_arguments",
        ),
        (
            json!({ "image_id": "1", "boxes": [[0, 0, 0, 65, 10]], "size": [64, 48], "expect": rev }),
            "invalid_arguments",
        ),
        (
            json!({ "image_id": "1", "boxes": [[7, 0, 0, 50, 10]], "size": [64, 48], "expect": rev }),
            "invalid_arguments",
        ),
        (json!({ "image_id": "1", "boxes": [], "size": [32, 30], "expect": rev }), "invalid_arguments"),
        // Every edit names the revision it was made on; coordinates name their view.
        (json!({ "image_id": "1", "boxes": [] }), "invalid_arguments"),
        (json!({ "image_id": "1", "boxes": [[0, 0, 0, 9, 9]], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "add": [[0, 0, 0, 9, 9]], "expect": rev }), "invalid_arguments"),
        (json!({ "image_id": "1", "remove": [1], "expect": "1-00000000" }), "label_boxes_changed"),
        (json!({ "image_id": "99", "boxes": [], "expect": rev }), "image_not_found"),
        (json!({ "image_id": "1,2", "boxes": [], "expect": rev }), "invalid_arguments"),
    ] {
        let mut args = args;
        args["job_id"] = json!(job);
        assert_eq!(code(s, "set_label_boxes", args.clone()).await, want, "{args}");
    }
    let e = call(
        s,
        "set_label_boxes",
        json!({ "job_id": job, "image_id": "1", "add": [[0, 0, 0, 9, 9]], "expect": rev, "size": [64, 48] }),
    )
    .await;
    let rev = e.unwrap().0["revision"].as_str().unwrap().to_string();

    // Two boxes that differ but look the same in a 32 x 24 view: a list naming that look once cannot say which
    // to keep, and is refused; naming it twice keeps both as stored.
    let two = json!({
        "job_id": job, "image_id": "1", "expect": rev, "size": [64, 48], "add": [[0, 0, 0, 16, 12], [0, 0, 0, 15, 12]]
    });
    let rev = ok(s, "set_label_boxes", two).await["revision"].as_str().unwrap().to_string();
    let mut boxes = json!([[1, 8, 2, 24, 14], [0, 0, 0, 5, 5], [0, 0, 0, 8, 6]]);
    let one = json!({ "job_id": job, "image_id": "1", "expect": rev, "size": [32, 24], "boxes": boxes });
    let e = call(s, "set_label_boxes", one).await.err().unwrap();
    assert!(e.body.code == "invalid_arguments" && e.body.message.contains("`remove`"), "{e:?}");
    boxes.as_array_mut().unwrap().push(json!([0, 0, 0, 8, 6]));
    let both = json!({ "job_id": job, "image_id": "1", "expect": rev, "size": [32, 24], "boxes": boxes });
    let kept = ok(s, "set_label_boxes", both).await;
    assert_eq!((kept["revision"].as_str(), kept["num_boxes"].as_u64()), (Some(rev.as_str()), Some(4)));

    // "Looks right" for the rest, in one call; an unknown image, or one whose boxes are not the revision seen,
    // is reported, not fatal.
    let review =
        json!({ "job_id": job, "image_ids": ["2", "3", "4", "5", "77"], "expect": { "2": "1-00000000" } });
    let r = ok(s, "review_label_images", review).await;
    assert_eq!(
        (r["updated"].as_u64(), r["failed_total"].as_u64(), r["to_review"].as_u64()),
        (Some(3), Some(2), Some(1))
    );
    assert_eq!(
        (r["failed"][0]["id"].as_str(), r["failed"][0]["code"].as_str()),
        (Some("2"), Some("label_boxes_changed"))
    );
    assert_eq!(r["failed"][1]["code"], "image_not_found");
    let rev2 = revision(s, &job, "2").await;
    // A revision for an image that is not in the list is a mistake, refused before anything is marked.
    let stray = json!({ "job_id": job, "image_ids": ["2"], "expect": { "3": rev2 } });
    assert_eq!(code(s, "review_label_images", stray).await, "invalid_arguments");
    let r =
        ok(s, "review_label_images", json!({ "job_id": job, "image_ids": ["2"], "expect": { "2": rev2 } }))
            .await;
    assert_eq!((r["updated"].as_u64(), r["to_review"].as_u64()), (Some(1), Some(0)));
    assert!(r.get("failed").is_none() && r.get("failed_total").is_none(), "{r}");
    let (v, text, blocks) = call(s, "view_label_image", json!({ "job_id": job })).await.unwrap();
    assert!(v["image"].is_null() && blocks.is_empty() && text.contains("nothing left to review"), "{text}");
    let status = ok(s, "label_status", json!({ "job_id": job, "log_lines": 0 })).await;
    assert_eq!((status["reviewed"].as_u64(), status["log"].as_array().unwrap().len()), (Some(5), 0));

    let ds = ok(s, "export_label_dataset", json!({ "job_id": job, "val_fraction": 0.2 })).await;
    assert_eq!((ds["train"].as_u64(), ds["val"].as_u64()), (Some(4), Some(1)));
    let jobs = ok(s, "list_label_jobs", json!({})).await;
    assert_eq!(jobs["items"][0]["dataset_id"], ds["dataset_id"]);
    let stopped = call(s, "stop_labeling", json!({ "job_id": job })).await.err().unwrap();
    assert_eq!(stopped.body.code, "label_job_not_active");
    assert!(stopped.body.hint.unwrap().contains("label_status"));

    let e = call(s, "label_status", json!({ "job_id": "lb_nope" })).await.err().unwrap();
    assert_eq!(e.body.code, "label_job_not_found");
    assert!(e.body.hint.unwrap().contains("list_label_jobs"));
    assert_eq!(code(s, "view_label_image", json!({ "job_id": job, "size": 99 })).await, "invalid_arguments");
    assert_eq!(code(s, "view_label_image", json!({ "job_id": job, "from": 0 })).await, "invalid_arguments");
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
    // The features come from /api/health, outside the timed flow.
    let warm = http.clone();
    let labels = tokio::task::spawn_blocking(move || warm.has_feature(folo_api::FEATURE_LABEL_REVISIONS));
    assert!(labels.await.unwrap().unwrap());
    let s = FoloMcp::new(http.clone(), Some(http.clone()), options(&home.join("client"), false));
    flow(&s, &home.join("photos")).await;
    // Paths go to the server as given, so they must be its absolute paths.
    let rel = json!({ "name": "x", "classes": ["car"], "paths": ["photos"] });
    assert_eq!(code(&s, "create_label_job", rel).await, "invalid_arguments");

    // A server without the image filters and revisions: the same answers, filtered here (page by page: the
    // fallback reads two images a request in tests).
    let old = Arc::new(HttpStore::new(&url, None).assume_features(vec![]));
    let o = FoloMcp::new(old.clone(), Some(old), options(&home.join("client"), false));
    let job = ok(&s, "list_label_jobs", json!({})).await["items"][0]["id"].as_str().unwrap().to_string();
    for (tool, args) in [
        ("label_status", json!({ "job_id": job, "log_lines": 0 })),
        ("list_label_images", json!({ "job_id": job, "status": "labeled", "reviewed": true, "limit": 2 })),
        ("list_label_images", json!({ "job_id": job, "reviewed": true, "cursor": "o3" })),
        ("view_label_image", json!({ "job_id": job, "image_id": "2" })),
        ("view_label_image", json!({ "job_id": job, "image_id": "5" })),
    ] {
        assert_eq!(ok(&o, tool, args.clone()).await, ok(&s, tool, args).await, "{tool}");
    }
    let view = json!({ "job_id": job, "image_id": "nope" });
    assert_eq!(code(&o, "view_label_image", view).await, "image_not_found");
    // Review there is one request per image, the revisions checked first.
    let rev = revision(&o, &job, "3").await;
    let undo =
        json!({ "job_id": job, "image_ids": ["3", "4", "77"], "reviewed": false, "expect": { "4": "0-0" } });
    let r = ok(&o, "review_label_images", undo).await;
    assert_eq!(
        (r["updated"].as_u64(), r["failed_total"].as_u64(), r["to_review"].as_u64()),
        (Some(1), Some(2), Some(1))
    );
    let redo = json!({ "job_id": job, "image_ids": ["3"], "expect": { "3": rev } });
    assert_eq!(ok(&o, "review_label_images", redo).await["to_review"], 0);
    let edit = json!({ "job_id": job, "image_id": "3", "expect": "0-0", "remove": [1] });
    assert_eq!(code(&o, "set_label_boxes", edit).await, "label_boxes_changed");

    // With --confine through a server, nothing that has the server read image files.
    let c = FoloMcp::new(http.clone(), Some(http), options(&home.join("client"), true));
    let photos = folo_api::json::path(home.join("photos"));
    for (tool, args) in [
        ("create_label_job", json!({ "name": "x", "classes": ["car"], "paths": [photos] })),
        ("create_label_job", json!({ "job_id": job, "paths": [photos] })),
        ("start_labeling", json!({ "job_id": job, "agent": "claude", "scope": "all" })),
        ("view_label_image", json!({ "job_id": job })),
        ("export_label_dataset", json!({ "job_id": job })),
    ] {
        assert_eq!(code(&c, tool, args.clone()).await, "not_supported", "{tool} {args}");
    }
    // A job without images reads nothing, so it can be made; a drive-relative path is no server path.
    let empty = ok(&c, "create_label_job", json!({ "name": "later", "classes": ["car"] })).await;
    assert_eq!(empty["images"], 0);
    let drive = json!({ "name": "x", "classes": ["car"], "paths": [r"\photos"] });
    let e = call(&s, "create_label_job", drive).await.err().unwrap();
    assert!(e.body.code == "invalid_arguments" && e.body.message.contains("drive"), "{e:?}");
    // Fast where the server lists no Fast tier is refused before the start, not run at the standard tier.
    let fast = json!({ "job_id": job, "agent": "claude", "scope": "all", "fast": true });
    let e = call(&s, "start_labeling", fast).await.err().unwrap();
    assert!(e.body.code == "invalid_arguments" && e.body.message.contains("no Fast tier"), "{e:?}");
    assert_eq!(ok(&c, "label_status", json!({ "job_id": job })).await["reviewed"], 5);
    let _ = stop.send(());
    serving.await.unwrap().unwrap();
    store.shutdown();
}

/// An edit made on boxes a run has changed since is refused, and so is one of an image with the agent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edits_on_an_old_revision_are_refused() {
    let home = TestDir::new("mcp-label-race");
    images(&home.join("photos"));
    let (go, gate) = mpsc::channel();
    let store = store_with(home.path(), Some(gate));
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let photos = folo_api::json::path(home.join("photos"));
    let made =
        ok(&s, "create_label_job", json!({ "name": "x", "classes": ["car"], "paths": [photos] })).await;
    let job = made["job_id"].as_str().unwrap().to_string();
    ok(&s, "start_labeling", json!({ "job_id": job, "agent": "claude", "image_ids": ["1"] })).await;
    go.send(()).unwrap();
    wait_idle(&s, &job).await;
    let seen = revision(&s, &job, "1").await;

    // Again with a note that changes the box; the call waits at the gate.
    let again =
        json!({ "job_id": job, "agent": "claude", "scope": "all", "image_ids": ["1"], "note": "wide" });
    ok(&s, "start_labeling", again).await;
    let deadline = Instant::now() + Duration::from_secs(20);
    while ok(&s, "list_label_images", json!({ "job_id": job, "status": "labeling" })).await["total"] != 1 {
        assert!(Instant::now() < deadline, "the image never reached the agent");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let edit = json!({ "job_id": job, "image_id": "1", "expect": seen, "remove": [1] });
    let busy = call(&s, "set_label_boxes", edit.clone()).await.err().unwrap();
    assert_eq!(busy.body.code, "label_job_active");
    assert!(busy.body.hint.unwrap().contains("stop_labeling"));
    go.send(()).unwrap();
    wait_idle(&s, &job).await;

    // The run has drawn other boxes: both the edit and the review on what was seen are refused.
    assert_eq!(code(&s, "set_label_boxes", edit).await, "label_boxes_changed");
    let review = json!({ "job_id": job, "image_ids": ["1"], "expect": { "1": seen } });
    let r = ok(&s, "review_label_images", review).await;
    assert_eq!(
        (r["updated"].as_u64(), r["failed"][0]["code"].as_str()),
        (Some(0), Some("label_boxes_changed"))
    );
    let now = revision(&s, &job, "1").await;
    assert_ne!(now, seen);
    let v = ok(&s, "view_label_image", json!({ "job_id": job, "image_id": "1" })).await;
    assert_eq!(v["image"]["boxes"], json!([[0, 0, 0, 64, 48]]));
    let edit = json!({ "job_id": job, "image_id": "1", "expect": now, "remove": [1] });
    assert_eq!(ok(&s, "set_label_boxes", edit).await["num_boxes"], 0);
    store.shutdown();
}

/// With `--confine`, nothing outside the data dir is added, and a job made earlier with outside images is
/// neither listed nor read nor changed.
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
    let jobs = ok(&s, "list_label_jobs", json!({})).await;
    assert_eq!((jobs["total"].as_u64(), jobs["items"][0]["id"].clone()), (Some(1), made["job_id"].clone()));
    for (tool, args) in [
        ("start_labeling", json!({ "job_id": old, "agent": "claude" })),
        ("export_label_dataset", json!({ "job_id": old })),
        ("view_label_image", json!({ "job_id": old, "image_id": "1" })),
        ("list_label_images", json!({ "job_id": old })),
        ("set_label_boxes", json!({ "job_id": old, "image_id": "1", "boxes": [], "expect": "0-811c9dc5" })),
        ("review_label_images", json!({ "job_id": old, "image_ids": ["1"] })),
    ] {
        assert_eq!(code(&s, tool, args).await, "path_outside_data_dir", "{tool}");
    }
    // The check is redone when the job's files change, even to as many as before.
    let inside = made["job_id"].as_str().unwrap().to_string();
    ok(&s, "list_label_images", json!({ "job_id": inside })).await;
    store.labeler().unwrap().delete_image(&inside, "1").unwrap();
    let swap = json!({ "job_id": inside, "paths": [format!("{outside}/0.png")] });
    assert_eq!(ok(&open, "create_label_job", swap).await["images"], 5);
    let list = json!({ "job_id": inside });
    assert_eq!(code(&s, "list_label_images", list).await, "path_outside_data_dir");
    store.shutdown();
}

/// What a waiter saw end is what it reports: the details read after are left out when they would belong to a
/// later run, or cannot be read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_finished_run_is_reported_as_seen() {
    let home = TestDir::new("mcp-label-seen");
    images(&home.join("photos"));
    let store = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let photos = folo_api::json::path(home.join("photos"));
    let made =
        ok(&s, "create_label_job", json!({ "name": "x", "classes": ["car"], "paths": [photos] })).await;
    let job = made["job_id"].as_str().unwrap().to_string();
    ok(&s, "start_labeling", json!({ "job_id": job, "agent": "claude", "image_ids": ["1"] })).await;
    wait_idle(&s, &job).await;
    let labels = crate::label::Labels::Store(store.clone());
    let seen = labels.job(&job).unwrap();
    let first = seen.run.clone().unwrap();
    // As a record: the same run, so the review count and the log are there.
    let record = |st: crate::types::LabelStatusResult| serde_json::to_value(st).unwrap();
    let same = record(labels.status_of(seen.clone(), 5));
    assert!(same["reviewed"] == 0 && !same["log"].as_array().unwrap().is_empty(), "{same}");
    // Another run since: the review count and the log read now could be the new run's, so neither is given.
    ok(&s, "start_labeling", json!({ "job_id": job, "agent": "claude", "image_ids": ["2"] })).await;
    wait_idle(&s, &job).await;
    for lines in [5, 0] {
        let st = record(labels.status_of(seen.clone(), lines));
        assert!(st.get("log").is_none() && st.get("reviewed").is_none(), "{st}");
        assert_eq!(st["run"]["total"], first.total, "{st}");
    }
    // The job is gone: the run as seen, without the details.
    labels.delete(&job).unwrap();
    let st = record(labels.status_of(seen, 5));
    assert!(st.get("log").is_none() && st.get("reviewed").is_none(), "{st}");
    assert_eq!(st["run"]["outcome"], "completed", "{st}");
    store.shutdown();
}

/// The `size` a view returns is taken back as is, a portrait image's included (rounding and the one-pixel side).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_views_size_is_taken_back() {
    let home = TestDir::new("mcp-label-portrait");
    let dir = home.join("tall");
    std::fs::create_dir_all(&dir).unwrap();
    image::RgbImage::from_pixel(100, 4000, image::Rgb([90, 90, 90])).save(dir.join("tall.png")).unwrap();
    image::RgbImage::from_pixel(5000, 3, image::Rgb([90, 90, 90])).save(dir.join("wide.png")).unwrap();
    let store = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let made = json!({ "name": "x", "classes": ["car"], "paths": [folo_api::json::path(&dir)] });
    let job = ok(&s, "create_label_job", made).await["job_id"].as_str().unwrap().to_string();
    let mut tall = None;
    for image in ["1", "2"] {
        let v = ok(&s, "view_label_image", json!({ "job_id": job, "image_id": image, "size": 256 })).await;
        let v = &v["image"];
        // 100 x 4000 at 256: 6.4 rounds to 6 (a check by aspect ratio wanted 240 for the height); 5000 x 3:
        // 0.15 is raised to one pixel.
        let is_tall = v["file"].as_str().unwrap().contains("tall");
        let [w, h] = if is_tall { [6, 256] } else { [256, 1] };
        assert_eq!(v["size"], json!([w, h]));
        let edit = json!({
            "job_id": job, "image_id": image, "expect": v["revision"], "size": v["size"], "add": [[0, 0, 0, w, h]]
        });
        let saved = ok(&s, "set_label_boxes", edit).await;
        assert_eq!(saved["boxes"], json!([[0, 0, 0, w, h]]));
        if is_tall {
            tall = Some((image, saved["revision"].clone()));
        }
    }
    // A frame of another shape is refused.
    let (image, rev) = tall.unwrap();
    let off = json!({ "job_id": job, "image_id": image, "expect": rev, "size": [7, 256], "remove": [1] });
    assert_eq!(code(&s, "set_label_boxes", off).await, "invalid_arguments");
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
    let hint = e.body.hint.unwrap();
    assert!(hint.contains("create_label_job with job_id") && !hint.contains("  "), "{hint}");
    assert_eq!(ok(&s, "list_label_jobs", json!({})).await["items"][0]["counts"]["images"], 5);
    let no_classes = json!({ "name": "x", "paths": [photos] });
    assert_eq!(code(&s, "create_label_job", no_classes).await, "invalid_arguments");
    store.shutdown();
}

/// Cursors near `u64::MAX` return an empty page with no `next_cursor` and a non-wrapping summary instead of
/// overflowing `offset + 1`, while normal cursors keep reporting the 1-based offset.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_label_images_handles_max_cursor_without_overflow() {
    let home = TestDir::new("mcp-label-cursor");
    images(&home.join("photos"));
    let store = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let photos = folo_api::json::path(home.join("photos"));
    let job = ok(&s, "create_label_job", json!({ "name": "x", "classes": ["car"], "paths": [photos] })).await
        ["job_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (mid, mid_summary, _) =
        call(&s, "list_label_images", json!({ "job_id": job, "cursor": "o2", "limit": 2 })).await.unwrap();
    assert_eq!(mid["items"].as_array().unwrap().len(), 2);
    assert_eq!(mid["total"], 5);
    assert_eq!(mid["next_cursor"], "o4");
    assert_eq!(mid_summary, "2 of 5 images (from 3)");

    let (page, summary, _) =
        call(&s, "list_label_images", json!({ "job_id": job, "cursor": format!("o{}", u64::MAX) }))
            .await
            .unwrap();
    assert_eq!(page["items"], json!([]));
    assert_eq!(page["total"], 5);
    assert!(page["next_cursor"].is_null(), "{page}");
    assert_eq!(summary, "0 of 5 images");
    store.shutdown();
}

/// Scripted Antigravity agent for MCP labeling tests.
struct ScriptedAntigravity;

impl Agent for ScriptedAntigravity {
    fn id(&self) -> LabelAgentId {
        LabelAgentId::Antigravity
    }

    fn info(&self) -> LabelAgent {
        LabelAgent {
            id: LabelAgentId::Antigravity,
            name: "Antigravity".into(),
            available: true,
            version: Some("1.5.13".into()),
            models: vec![LabelModel {
                id: "gemini-4-neon".into(),
                label: "Gemini 4 Neon".into(),
                fast: false,
            }],
            default_model: None,
            default_effort: LabelEffort::Low,
            fast: false,
            hint: None,
        }
    }

    fn default_batch(&self) -> usize {
        8
    }

    fn label(&self, req: &AgentRequest<'_>) -> Result<AgentReply, AgentError> {
        if req.prompt.contains("auth-fail") {
            return Err(AgentError::failed("Authentication required"));
        }
        let images: Vec<Value> = (1..=req.images.len())
            .map(|i| json!({ "image": i, "boxes": [["car", 16, 4.8, 48, 28.8]] }))
            .collect();
        let usage = LabelUsage {
            calls: 1,
            input_tokens: 600,
            cached_input_tokens: 120,
            output_tokens: 40,
            cost_usd: None,
        };
        Ok(AgentReply { text: json!({ "images": images }).to_string(), usage })
    }

    fn hint(&self, message: &str) -> Option<String> {
        message
            .contains("Authentication required")
            .then(|| "run `agy` (or `agy acp --login`) in a terminal on the server's machine to sign in, then start again".into())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn antigravity_agent_labels_and_rejects_fast() {
    let home = TestDir::new("mcp-label-agy");
    images(&home.join("photos"));
    let store =
        Arc::new(RealStore::open(home.to_path_buf()).label_agents(vec![Arc::new(ScriptedAntigravity)]));
    let s = FoloMcp::new(store.clone(), None, options(home.path(), false));
    let photos = folo_api::json::path(home.join("photos"));
    let job = ok(&s, "create_label_job", json!({ "name": "street", "classes": ["car"], "paths": [photos] }))
        .await["job_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Antigravity has no Fast tier: refused before any call.
    assert_eq!(
        code(&s, "start_labeling", json!({ "job_id": job, "agent": "antigravity", "fast": true })).await,
        "invalid_arguments"
    );

    let started = ok(
        &s,
        "start_labeling",
        json!({ "job_id": job, "agent": "antigravity", "model": "gemini-4-neon", "image_ids": ["1"] }),
    )
    .await;
    assert_eq!(started, json!({ "job_id": job, "total": 1, "model": "gemini-4-neon" }));
    let status = wait_idle(&s, &job).await;
    assert_eq!(status["counts"]["labeled"], 1);
    assert_eq!(status["run"]["agent"], "antigravity");
    assert_eq!(status["run"]["model"], "gemini-4-neon");
    assert_eq!(status["run"]["effort"], "low");
    assert_eq!(status["run"]["outcome"], "completed");
    assert!(status["run"].get("error").is_none() && status["run"].get("hint").is_none());
    assert_eq!(
        (status["calls"].as_u64(), status["input_tokens"].as_u64(), status["output_tokens"].as_u64()),
        (Some(1), Some(600), Some(40))
    );

    // Failed run exposes actionable `error` and `hint` directly on `label_status`'s `run` summary.
    let _ = ok(
        &s,
        "start_labeling",
        json!({ "job_id": job, "agent": "antigravity", "image_ids": ["2"], "note": "auth-fail" }),
    )
    .await;
    let failed = wait_idle(&s, &job).await;
    assert_eq!(failed["run"]["outcome"], "failed");
    assert_eq!(failed["run"]["error"], "Authentication required");
    assert!(failed["run"]["hint"].as_str().unwrap().contains("agy acp --login"));

    let prompt = crate::prompts::get(
        &s.inner,
        rmcp::model::GetPromptRequestParams::new("label_folder")
            .with_arguments(json!({ "folder_path": photos, "classes": "car" }).as_object().unwrap().clone()),
    )
    .await
    .unwrap();
    let msg = serde_json::to_string(&prompt).unwrap();
    assert!(msg.contains("`claude`") && msg.contains("`codex`") && msg.contains("`antigravity`"), "{msg}");
    store.shutdown();
}
