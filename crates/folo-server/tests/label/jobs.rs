//! Jobs: upload, label, review and make a dataset; folder images read in place; the image filters.

use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    delete, err, fixture, get, idle, images, job_with_images, ok, png, post, put, send, upload,
};

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

    // The labels are in pixels of each image (the reply's pixels, each image being sent as it is).
    let page = images(app, &job.id, "?status=labeled").await;
    assert_eq!((page.total, page.images.len()), (6, 6));
    assert_eq!(page.images[0].source, Some(LabelSource::Agent));
    assert_eq!(page.images[0].boxes, [BBox { class: 0, x1: 50.0, y1: 10.0, x2: 150.0, y2: 60.0 }]);
    // The 64 x 64 image: the box is clipped to it.
    assert_eq!(page.images[5].boxes, [BBox { class: 0, x1: 50.0, y1: 10.0, x2: 64.0, y2: 60.0 }]);
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
    // A path as Explorer's "Copy as path" gives it, in quotes with spaces around, is the same folder.
    let quoted = format!(" \"{}\" ", photos.to_string_lossy());
    let again: LabelImagesAdded = ok(app, post(&uri, json!({ "path": quoted }))).await;
    assert!(again.added.is_empty());
    assert_eq!(again.skipped.iter().filter(|s| s.reason == "the job already has this file").count(), 4);
    // A folder without images, or a file that is not one, says why nothing was added.
    std::fs::create_dir_all(t.join("notes/sub")).unwrap();
    std::fs::write(t.join("notes/a.md"), "x").unwrap();
    std::fs::write(t.join("notes/sub/b.TIF"), "x").unwrap();
    let none: LabelImagesAdded =
        ok(app, post(&uri, json!({ "path": t.join("notes").to_string_lossy() }))).await;
    assert!(none.added.is_empty());
    let reasons: Vec<&str> = none.skipped.iter().map(|s| s.reason.as_str()).collect();
    assert_eq!(
        reasons,
        ["no JPEG, PNG, BMP or WebP images in this folder or its subfolders (it has .md, .tif files)"]
    );
    let file: LabelImagesAdded =
        ok(app, post(&uri, json!({ "path": t.join("notes/a.md").to_string_lossy() }))).await;
    assert_eq!(
        (file.skipped[0].file.as_str(), file.skipped[0].reason.as_str()),
        ("a.md", "not a JPEG, PNG, BMP or WebP file")
    );
    // A relative path is refused as such, never read from (and named after) the server's working directory.
    let relative = post(&uri, json!({ "path": "\"photos\"" }));
    let e = err(app, relative, StatusCode::BAD_REQUEST, "invalid_body").await;
    assert_eq!(e.message, "path must be a full path: photos is relative");

    // Labeled and exported: the dataset has copies; deleting an image or the job leaves the folder alone.
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), json!({ "agent": "claude" })))
        .await;
    idle(app, &job.id).await;
    // A file moved away since: the export says which, and makes no folder.
    let away = t.join("a.png");
    std::fs::rename(photos.join("day 2").join("a.png"), &away).unwrap();
    let dataset = format!("/api/label/jobs/{}/dataset", job.id);
    let e = err(app, post(&dataset, json!({})), StatusCode::NOT_FOUND, "image_not_found").await;
    let gone = photos.join("day 2").join("a.png");
    // The folder as the server keeps it (made absolute), so only its end is compared.
    let named = e.message.strip_prefix("1 labeled image is no longer on disk, such as ").unwrap();
    assert!(Path::new(named).ends_with(Path::new("photos").join("day 2").join("a.png")), "{}", e.message);
    assert!(
        !t.join("home/datasets").exists()
            || std::fs::read_dir(t.join("home/datasets")).unwrap().next().is_none()
    );
    std::fs::rename(&away, &gone).unwrap();
    let ds: Dataset = ok(app, post(&dataset, json!({ "val_fraction": 0 }))).await;
    assert_eq!(ds.splits, Splits { train: 4, val: 0, test: 0 });
    assert!(Path::new(&ds.path).join("images/train/a-2.png").is_file());
    assert!(Path::new(&ds.path).join("labels/train/a-2.txt").is_file());
    send(app, delete(&format!("/api/label/jobs/{}/images/{}", job.id, page.images[0].id))).await;
    send(app, delete(&format!("/api/label/jobs/{}", job.id))).await;
    assert!(photos.join("a.png").is_file() && photos.join("day 2").join("a.png").is_file());
}

#[tokio::test]
async fn a_person_marks_images_reviewed_and_the_agent_relabeling_one_clears_it() {
    let t = TestDir::new("label-reviewed");
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;
    let job = job_with_images(app, 3).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    let page = images(app, &job.id, "").await;
    let (a, b) = (page.images[0].id.clone(), page.images[1].id.clone());
    let image = |id: &str| format!("{uri}/images/{id}");

    // Without labels there is nothing to mark right; a request asking nothing is refused.
    let none = put(&image(&a), json!({ "reviewed": true }));
    assert!(err(app, none, StatusCode::BAD_REQUEST, "invalid_body").await.message.contains("save its boxes"));
    err(app, put(&image(&a), json!({})), StatusCode::BAD_REQUEST, "invalid_body").await;

    ok::<LabelJob>(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    idle(app, &job.id).await;
    let labeled = images(app, &job.id, "").await;
    assert!(labeled.images.iter().all(|i| i.status == LabelImageStatus::Labeled && !i.reviewed));

    // "Looks right": the agent's boxes stay its own, now reviewed. Saved boxes are reviewed too, and it can be undone.
    let right: LabelImage = ok(app, put(&image(&a), json!({ "reviewed": true }))).await;
    assert_eq!((right.reviewed, right.source), (true, Some(LabelSource::Agent)));
    assert_eq!(right.boxes, labeled.images[0].boxes);
    let saved: LabelImage = ok(app, put(&image(&b), json!({ "boxes": [] }))).await;
    assert!(saved.reviewed);
    let undone: LabelImage = ok(app, put(&image(&b), json!({ "reviewed": false }))).await;
    assert_eq!((undone.reviewed, undone.source), (false, Some(LabelSource::Manual)));

    // The agent labels the reviewed image again: its new boxes are not reviewed.
    ok::<LabelJob>(app, post(&format!("{uri}/start"), json!({ "agent": "claude", "image_ids": [a] }))).await;
    idle(app, &job.id).await;
    let again = images(app, &job.id, "").await;
    assert_eq!((again.images[0].source, again.images[0].reviewed), (Some(LabelSource::Agent), false));
}

#[tokio::test]
async fn images_filter_by_review_state_and_ids() {
    let t = TestDir::new("label-filters");
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;
    let system: Value = ok(app, get("/api/system")).await;
    assert!(system["features"].as_array().unwrap().contains(&json!(FEATURE_LABEL_IMAGE_FILTERS)), "{system}");
    let job = job_with_images(app, 3).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    ok::<LabelJob>(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    idle(app, &job.id).await;
    let all = images(app, &job.id, "").await;
    let ids: Vec<&str> = all.images.iter().map(|i| i.id.as_str()).collect();
    ok::<LabelImage>(app, put(&format!("{uri}/images/{}", ids[1]), json!({ "reviewed": true }))).await;

    let reviewed = images(app, &job.id, "?reviewed=true").await;
    assert_eq!((reviewed.total, reviewed.images[0].id.as_str()), (1, ids[1]));
    let to_review = images(app, &job.id, "?status=labeled&reviewed=false&limit=1").await;
    assert_eq!((to_review.total, to_review.images.len(), to_review.images[0].id.as_str()), (2, 1, ids[0]));
    let picked = images(app, &job.id, &format!("?ids={},{},nope", ids[2], ids[0])).await;
    let got: Vec<&str> = picked.images.iter().map(|i| i.id.as_str()).collect();
    assert_eq!((picked.total, got), (2, vec![ids[0], ids[2]]), "in the order the images were added");
    assert_eq!(images(app, &job.id, "?ids=nope&reviewed=true").await.total, 0);
    err(app, get(&format!("{uri}/images?reviewed=maybe")), StatusCode::BAD_REQUEST, "invalid_query").await;
}

#[tokio::test]
async fn box_edits_and_reviews_check_the_revision() {
    let t = TestDir::new("label-revisions");
    let f = fixture(&t.join("home"), vec![]);
    let app = &f.app;
    let system: Value = ok(app, get("/api/system")).await;
    assert!(system["features"].as_array().unwrap().contains(&json!(FEATURE_LABEL_REVISIONS)), "{system}");
    let job = job_with_images(app, 3).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    let first: LabelJob = ok(app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    idle(app, &job.id).await;
    // Runs started within the same second still have their own ids.
    let again = json!({ "agent": "claude", "scope": "all" });
    let second: LabelJob = ok(app, post(&format!("{uri}/start"), again)).await;
    idle(app, &job.id).await;
    let (first, second) = (first.run.unwrap().id, second.run.unwrap().id);
    assert!(first.starts_with("lr_") && second.starts_with("lr_") && first != second, "{first} {second}");
    let all = images(app, &job.id, "").await;
    let ids: Vec<String> = all.images.iter().map(|i| i.id.clone()).collect();
    let rev = label_box_revision(&all.images[0].boxes);
    assert!(rev.starts_with(&format!("{}-", all.images[0].boxes.len())), "{rev}");

    // An edit on another revision changes nothing.
    let image = format!("{uri}/images/{}", ids[0]);
    let stale = json!({ "boxes": [], "expect": "0-00000000" });
    err(app, put(&image, stale), StatusCode::CONFLICT, "label_boxes_changed").await;
    assert_eq!(images(app, &job.id, "").await.images[0].boxes, all.images[0].boxes);
    let saved: LabelImage = ok(app, put(&image, json!({ "boxes": [], "expect": rev }))).await;
    assert!(saved.boxes.is_empty() && saved.reviewed);
    assert_eq!(label_box_revision(&saved.boxes), "0-811c9dc5");

    // Many marks in one request: the stale one and the unknown one fail, a repeated id counts once.
    let review = format!("{uri}/review");
    let mut body = json!({ "image_ids": [ids[1], ids[2], "nope", ids[1]] });
    body["expect"][&ids[2]] = json!("9-0");
    let r: LabelReviewResult = ok(app, post(&review, body)).await;
    assert_eq!((r.updated, r.failed_total, r.to_review), (1, 2, 1));
    let codes: Vec<(&str, &str)> = r.failed.iter().map(|f| (f.id.as_str(), f.code.as_str())).collect();
    assert_eq!(codes, vec![(ids[2].as_str(), "label_boxes_changed"), ("nope", "image_not_found")]);
    let r: LabelReviewResult =
        ok(app, post(&review, json!({ "image_ids": [ids[1]], "reviewed": false }))).await;
    assert_eq!((r.updated, r.to_review, r.failed.len()), (1, 2, 0));
    let reviewed = images(app, &job.id, "?reviewed=true").await;
    assert_eq!((reviewed.total, reviewed.images[0].id.as_str()), (1, ids[0].as_str()));
    let many: Vec<String> = (0..1001).map(|i| i.to_string()).collect();
    err(app, post(&review, json!({ "image_ids": many })), StatusCode::BAD_REQUEST, "invalid_body").await;
    // A revision for an image not in the list is refused before anything is marked.
    let mut stray = json!({ "image_ids": [ids[1]] });
    stray["expect"][&ids[2]] = json!(label_box_revision(&all.images[2].boxes));
    err(app, post(&review, stray), StatusCode::BAD_REQUEST, "invalid_body").await;
    assert_eq!(images(app, &job.id, "?reviewed=true").await.total, 1);
    // The cheap health check lists the same features as /api/system.
    let health: Value = ok(app, get("/api/health")).await;
    assert_eq!(health["features"], system["features"]);
}

#[tokio::test]
async fn review_false_refuses_unlabeled_images_and_keeps_active_run_waiting() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::harness::Step;

    let t = TestDir::new("label-review-unlabeled");
    let home = t.join("home");
    let gate = Arc::new(AtomicBool::new(false));
    let f = fixture(&home, vec![Step::Until(gate.clone())]);
    let app = &f.app;
    let job = job_with_images(app, 2).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    // Idle job: neither PUT nor POST /review may mark an unlabeled image unreviewed.
    err(
        app,
        put(&format!("{uri}/images/1"), json!({ "reviewed": false })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    let r: LabelReviewResult =
        ok(app, post(&format!("{uri}/review"), json!({ "image_ids": ["1", "2"], "reviewed": false }))).await;
    assert_eq!((r.updated, r.failed_total), (0, 2));
    assert!(r.failed.iter().all(|f| f.code == "invalid_body"), "{:?}", r.failed);

    // Active run: image 1 is held in an agent call (`labeling`) and image 2 is waiting (`pending`).
    // Wait on the scripted agent's call count (matching `runs.rs` and `idle`), with no tighter wall-clock bound.
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
    assert_eq!(f.agent.calls(), 1, "agent call did not start");
    let live = images(app, &job.id, "").await.images;
    assert_eq!(
        live.iter().map(|i| i.status).collect::<Vec<_>>(),
        [LabelImageStatus::Labeling, LabelImageStatus::Pending]
    );

    // Image 1 is inside an agent call: both PUT and POST /review must return `409 label_job_active`
    // (for both `reviewed: false` and `reviewed: true`), not `400 invalid_body`.
    for reviewed in [false, true] {
        err(
            app,
            put(&format!("{uri}/images/1"), json!({ "reviewed": reviewed })),
            StatusCode::CONFLICT,
            "label_job_active",
        )
        .await;
        let r: LabelReviewResult =
            ok(app, post(&format!("{uri}/review"), json!({ "image_ids": ["1", "2"], "reviewed": reviewed })))
                .await;
        let codes: Vec<(&str, &str)> = r.failed.iter().map(|f| (f.id.as_str(), f.code.as_str())).collect();
        assert_eq!((r.updated, r.failed_total), (0, 2), "reviewed={reviewed}");
        assert_eq!(codes, vec![("1", "label_job_active"), ("2", "invalid_body")], "reviewed={reviewed}");
    }
    // Image 2 is waiting (`pending`): refused with `400 invalid_body` and kept in the active run's queue.
    err(
        app,
        put(&format!("{uri}/images/2"), json!({ "reviewed": false })),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    let running: LabelJob = ok(app, get(&uri)).await;
    assert_eq!(running.run.unwrap().total, 2, "waiting image must stay in the run");

    gate.store(true, Ordering::SeqCst);
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.total, run.done, run.failed), (2, 2, 0));
    assert_eq!((done.counts.labeled, done.counts.pending), (2, 0));

    // Once labeled, both `reviewed: true` and `reviewed: false` still work in PUT and POST /review.
    let marked: LabelImage = ok(app, put(&format!("{uri}/images/1"), json!({ "reviewed": true }))).await;
    assert!(marked.reviewed);
    let unmarked: LabelImage = ok(app, put(&format!("{uri}/images/1"), json!({ "reviewed": false }))).await;
    assert!(!unmarked.reviewed);
    let r: LabelReviewResult =
        ok(app, post(&format!("{uri}/review"), json!({ "image_ids": ["1", "2"], "reviewed": true }))).await;
    assert_eq!((r.updated, r.failed_total, r.to_review), (2, 0, 0));
    let r: LabelReviewResult =
        ok(app, post(&format!("{uri}/review"), json!({ "image_ids": ["1", "2"], "reviewed": false }))).await;
    assert_eq!((r.updated, r.failed_total, r.to_review), (2, 0, 2));
}
