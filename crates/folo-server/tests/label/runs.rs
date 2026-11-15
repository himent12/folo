//! Runs: a deleted job, labels saved as calls return, crash sweeps, stopping, hand edits, changes that
//! cannot be saved, events, and restarts.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::http::{StatusCode, header};
use folo_server::Store;
use folo_server::api::*;
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::harness::{
    Step, boxes_reply, delete, err, fixture, fixture_with, get, idle, images, job_with_images, ok, png, post,
    put, send, upload,
};

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

/// A copy of a job folder made while its run was unfinished is skipped before loading: the restart closes the
/// job's run, never the backup's, whose files stay byte for byte (review round 2).
#[tokio::test]
async fn a_copied_job_folder_is_left_as_it_is() {
    let t = TestDir::new("label-copy");
    let home = t.join("home");
    let f = fixture(&home, vec![Step::Hold]);
    let job = job_with_images(&f.app, 2).await;
    let uri = format!("/api/label/jobs/{}", job.id);
    ok::<LabelJob>(&f.app, post(&format!("{uri}/start"), json!({ "agent": "claude" }))).await;
    let dir = home.join("label").join(&job.id);
    let saved = std::fs::read(dir.join("job.json")).unwrap();
    f.store.shutdown();
    drop(f);
    let copy = home.join("label").join(format!("{} - Copy", job.id));
    std::fs::create_dir_all(&copy).unwrap();
    std::fs::write(dir.join("job.json"), &saved).unwrap();
    std::fs::write(copy.join("job.json"), &saved).unwrap();
    let names = |d: &std::path::Path| {
        let mut v: Vec<_> = std::fs::read_dir(d).unwrap().map(|e| e.unwrap().file_name()).collect();
        v.sort();
        v
    };
    let before = names(&copy);

    let f = fixture(&home, vec![]);
    let list: LabelJobList = ok(&f.app, get("/api/label/jobs")).await;
    assert_eq!(list.jobs.len(), 1, "one job of its id");
    let job: LabelJob = ok(&f.app, get(&uri)).await;
    assert_eq!(job.run.unwrap().outcome, Some(LabelOutcome::Stopped), "the job's own run is closed");
    assert_eq!(std::fs::read(copy.join("job.json")).unwrap(), saved, "the backup is not rewritten");
    assert_eq!(names(&copy), before, "and nothing is added to it");
}
