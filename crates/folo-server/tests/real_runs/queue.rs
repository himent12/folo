//! The queue: stop, resume and restart, moves, the resolved config, and shutdown.

use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_server::Store;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{Events, app, dataset, delete, err, get, ok, post, post_json, send, store, tiny};

#[tokio::test(flavor = "multi_thread")]
async fn queue_stop_resume_and_restart() {
    let t = TestDir::new("runs-stop");
    let home = t.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "stoppy", 6).await;
    // Runs wait after each epoch until stopped (or the hold is lifted), so the stops below always land before a
    // run could finish on its own.
    s.hold_runs_after_epochs(true);

    // One run at a time: the second waits in the queue, and stopping it there is immediate.
    let a: Run = ok(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    // The run records its dataset's name and classes, so it still names them once the dataset is removed.
    assert_eq!(a.dataset, Some(RunDataset { name: ds.name.clone(), classes: ds.classes.clone() }));
    let b: Run = ok(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    let mut sse = Events::open(&app, &a.id).await;
    sse.until_epoch().await;
    assert_eq!(ok::<Run>(&app, get(&format!("/api/runs/{}", b.id))).await.status, RunStatus::Queued);
    let stopped_b: Run = ok(&app, post(&format!("/api/runs/{}/stop", b.id))).await;
    assert_eq!(stopped_b.status, RunStatus::Stopped);
    assert!(stopped_b.model_id.is_none() && stopped_b.started.is_none());
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", b.id))).await;
    assert!(logs.lines[0].message.starts_with("queued behind 1"), "{:?}", logs.lines);

    // Deleting a run that waits and never started takes it out of the queue (the one behind moves up) and away.
    let c: Run = ok(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    let d: Run = ok(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    assert_eq!((c.queue_position, d.queue_position), (Some(1), Some(2)));
    assert_eq!(send(&app, delete(&format!("/api/runs/{}", c.id))).await.0, StatusCode::NO_CONTENT);
    err(&app, get(&format!("/api/runs/{}", c.id)), StatusCode::NOT_FOUND, "run_not_found").await;
    assert!(!home.join("runs").join(&c.id).exists());
    assert_eq!(ok::<Run>(&app, get(&format!("/api/runs/{}", d.id))).await.queue_position, Some(1));
    assert_eq!(send(&app, delete(&format!("/api/runs/{}", d.id))).await.0, StatusCode::NO_CONTENT);

    // A running run cannot be deleted; a stop takes effect after the current batch and keeps the checkpoint.
    err(&app, delete(&format!("/api/runs/{}", a.id)), StatusCode::CONFLICT, "run_active").await;
    let stopping: Run = ok(&app, post(&format!("/api/runs/{}/stop", a.id))).await;
    assert!(matches!(stopping.status, RunStatus::Stopping | RunStatus::Stopped), "{:?}", stopping.status);
    // Stopping again while it stops answers at once (with the run, or 409 once it has stopped).
    let again =
        tokio::time::timeout(Duration::from_secs(10), send(&app, post(&format!("/api/runs/{}/stop", a.id))))
            .await
            .expect("a repeated stop answers");
    assert!(matches!(again.0, StatusCode::OK | StatusCode::CONFLICT), "{}", again.0);
    let events = sse.until_status(&["stopped", "completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "stopped", "{events:?}");
    let a1: Run = ok(&app, get(&format!("/api/runs/{}", a.id))).await;
    assert_eq!(a1.status, RunStatus::Stopped);
    assert_eq!(a1.epoch, 1);
    assert!(a1.model_id.is_some(), "stopped after a validated epoch");
    // One log line per finished epoch, in plain terms.
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", a.id))).await;
    assert!(logs.lines.iter().any(|l| l.message.starts_with("epoch 1/2: loss ")), "{:?}", logs.lines);
    err(&app, post(&format!("/api/runs/{}/stop", a.id)), StatusCode::CONFLICT, "run_not_active").await;
    s.hold_runs_after_epochs(false);

    // Resume: same id, the stored config (only `device` may change), on to the last epoch.
    let resume = |fields: Value| post_json("/api/runs", fields);
    let e = err(
        &app,
        resume(json!({ "resume_run_id": a.id, "epochs": 5 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("epochs"), "{}", e.message);
    err(&app, resume(json!({ "resume_run_id": "run_none" })), StatusCode::NOT_FOUND, "run_not_found").await;
    let (status, body) =
        send(&app, resume(json!({ "resume_run_id": a.id, "dataset_id": ds.id, "device": "cpu" }))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let resumed: Run = serde_json::from_slice(&body).unwrap();
    assert_eq!((resumed.id.as_str(), resumed.status), (a.id.as_str(), RunStatus::Queued));
    let events = sse.until_status(&["completed", "failed", "stopped"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let a2: Run = ok(&app, get(&format!("/api/runs/{}", a.id))).await;
    assert_eq!((a2.status, a2.epoch, a2.step), (RunStatus::Completed, 2, 4));
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", a.id))).await;
    assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), (1..=4).collect::<Vec<_>>());
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", a.id))).await;
    assert!(logs.lines.iter().any(|l| l.message == "resuming from weights/last.folo"));
    assert!(logs.lines.iter().any(|l| l.message.starts_with("resumed from")));
    err(&app, resume(json!({ "resume_run_id": a.id })), StatusCode::CONFLICT, "run_not_resumable").await;

    // A run that was never trained resumes from scratch (held after its first epoch, for the shutdown below).
    s.hold_runs_after_epochs(true);
    let (status, body) = send(&app, resume(json!({ "resume_run_id": b.id }))).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let mut sse_b = Events::open(&app, &b.id).await;
    sse_b.until_epoch().await;

    // Shutting the store down stops the active run gracefully.
    drop(sse);
    drop(sse_b);
    drop(app);
    s.shutdown();
    let b1 = s.get_run(&b.id).unwrap();
    assert_eq!(b1.status, RunStatus::Stopped);
    // Interrupted by the server, not stopped by a client: the run says so.
    assert_eq!(b1.error_code.as_deref(), Some("interrupted"));
    drop(s);

    // A run a crashed server left `running` comes back `stopped` (and resumable).
    let path = home.join("runs").join(&b.id).join("run.json");
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["run"]["status"] = json!("running");
    file["run"]["finished"] = Value::Null;
    // ... and that crashed after writing its checkpoint, before the epoch reached metrics.jsonl and run.json.
    let ckpt_epochs = file["epochs_done"].as_u64().unwrap() as u32;
    assert!(ckpt_epochs >= 1);
    file["epochs_done"] = json!(0);
    file["run"]["best"] = Value::Null;
    file["last_metrics"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let metrics = home.join("runs").join(&b.id).join("metrics.jsonl");
    let text = std::fs::read_to_string(&metrics).unwrap();
    let cut = text.rfind("{\"event\":\"epoch\"").unwrap();
    let tail = &text[cut..];
    let rest = tail.find('\n').map_or("", |n| &tail[n + 1..]);
    std::fs::write(&metrics, format!("{}{rest}", &text[..cut])).unwrap();
    let s = store(&home);
    let app = self::app(&s);
    let b2: Run = ok(&app, get(&format!("/api/runs/{}", b.id))).await;
    assert_eq!(b2.status, RunStatus::Stopped);
    assert!(b2.finished.is_some());
    assert_eq!(b2.epoch, ckpt_epochs, "progress comes back from the checkpoint");
    assert!(b2.best.is_some());
    let b_last = format!("{}-last", b2.model_id.clone().unwrap().trim_end_matches("-last"));
    let last: Model = ok(&app, get(&format!("/api/models/{b_last}"))).await;
    assert_eq!(last.metrics, None, "the last epoch's metrics are unknown, not the best epoch's");
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", b.id))).await;
    assert_eq!(b2.error_code.as_deref(), Some("interrupted"));
    let line = &logs.lines.last().unwrap().message;
    assert!(line.contains("server stopped during training") && !line.contains("POST /api"), "{line}");
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(runs.runs.len(), 2);
    let r: Run = ok(&app, resume(json!({ "resume_run_id": b.id }))).await;
    assert_eq!(r.status, RunStatus::Queued);
    let mut sse = Events::open(&app, &b.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let done: Run = ok(&app, get(&format!("/api/runs/{}", b.id))).await;
    assert_eq!(done.epoch, 2);
    let last: Model = ok(&app, get(&format!("/api/models/{b_last}"))).await;
    assert!(last.metrics.is_some(), "known again once an epoch was validated");
}

/// docs/API.md `POST /api/runs/{id}/queue` on the real store: positions, moves, `start` behind the active run,
/// `queue` events, `run_not_queued` for the active run, the worker taking the front run; and `Run.resolved`, from
/// the trainer, from `run.json` after a restart and from `config.json` for runs recorded before the field.
#[tokio::test(flavor = "multi_thread")]
async fn queue_moves_and_resolved_config() {
    let t = TestDir::new("runs-queue");
    let home = t.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "queued", 12).await;
    // Runs wait after each epoch until stopped, so every run below is still active when the test stops it.
    s.hold_runs_after_epochs(true);
    let mv = |id: &str, body: Value| post_json(&format!("/api/runs/{id}/queue"), body);
    let run = |id: &str| {
        let (app, id) = (app.clone(), id.to_string());
        async move { ok::<Run>(&app, get(&format!("/api/runs/{id}"))).await }
    };
    // Until the next `queue` event equal to `want` (earlier ones are skipped).
    async fn until_queue(sse: &mut Events, want: Value) {
        loop {
            let (name, v) = sse.next().await;
            assert_ne!(name, "status", "a status before {want}: {v}");
            if name == "queue" && v == want {
                return;
            }
        }
    }

    // `a` trains (long enough to stay running while the others move).
    let a: Run = ok(&app, post_json("/api/runs", tiny(&ds, 200))).await;
    let mut sse_a = Events::open(&app, &a.id).await;
    sse_a.until_status(&["running"]).await;
    let a_now = run(&a.id).await;
    assert_eq!(a_now.queue_position, None);
    let resolved = a_now.resolved.clone().expect("set once the trainer is built");
    // `auto` rule 4: MuSGD at lr0 0.01 for any run length (no betas).
    assert_eq!(resolved.optimizer, ResolvedOptimizer::Musgd);
    assert_eq!((resolved.lr0, resolved.betas), (0.01, None));
    assert_eq!((resolved.batch, resolved.accumulate, resolved.nbs), (4, 16, 64));
    assert_eq!(resolved.optimizer_steps_per_epoch, 1, "3 batches of 4, accumulated over 16");
    // The trainer writes config.json once it runs, before its first epoch.
    sse_a.until_epoch().await;
    let config: Value =
        serde_json::from_slice(&std::fs::read(home.join("runs").join(&a.id).join("config.json")).unwrap())
            .unwrap();
    assert_eq!(serde_json::from_value::<ResolvedTrain>(config["resolved"].clone()).unwrap(), resolved);
    assert_eq!(config["optimizer"]["lr0"], json!(resolved.lr0));
    assert_eq!(a_now.config.optimizer, Optimizer::Auto, "the requested config is kept as requested");

    let mut ids = Vec::new();
    for i in 0..3 {
        let r: Run = ok(&app, post_json("/api/runs", tiny(&ds, 3))).await;
        assert_eq!(
            (r.status, r.queue_position, r.resolved.is_none()),
            (RunStatus::Queued, Some(i + 1), true)
        );
        ids.push(r.id);
    }
    let (b, c, d) = (ids[0].clone(), ids[1].clone(), ids[2].clone());
    let mut sse_b = Events::open(&app, &b).await;
    assert_eq!(sse_b.next().await.0, "status");

    let r: Run = ok(&app, mv(&d, json!({ "action": "top" }))).await;
    assert_eq!(r.queue_position, Some(1));
    until_queue(&mut sse_b, json!({ "position": 2, "waiting": 3 })).await;
    let list: RunList = ok(&app, get("/api/runs")).await;
    let place = |id: &str| list.runs.iter().find(|r| r.id == id).unwrap().queue_position;
    assert_eq!((place(&d), place(&b), place(&c), place(&a.id)), (Some(1), Some(2), Some(3), None));
    let r: Run = ok(&app, mv(&b, json!({ "position": 3 }))).await;
    assert_eq!(r.queue_position, Some(3));
    until_queue(&mut sse_b, json!({ "position": 3, "waiting": 3 })).await;
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{b}/logs"))).await;
    assert!(logs.lines.last().unwrap().message.contains("position 3 of 3"), "{:?}", logs.lines);

    // The active run is not in the queue; `start` while it trains only moves to the front.
    let e = err(&app, mv(&a.id, json!({ "action": "top" })), StatusCode::CONFLICT, "run_not_queued").await;
    assert!(e.message.contains("running"), "{}", e.message);
    err(&app, mv(&b, json!({ "action": "top", "position": 1 })), StatusCode::BAD_REQUEST, "invalid_body")
        .await;
    let r: Run = ok(&app, mv(&c, json!({ "action": "start" }))).await;
    assert_eq!((r.status, r.queue_position), (RunStatus::Queued, Some(1)));
    assert_eq!(run(&a.id).await.status, RunStatus::Running, "never preempted");
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{c}/logs"))).await;
    assert!(
        logs.lines.last().unwrap().message.contains("starts when the active run ends"),
        "{:?}",
        logs.lines
    );

    // Stopping the active run: the worker takes the front run (c); the others move up.
    let mut sse_c = Events::open(&app, &c).await;
    let _: Run = ok(&app, post(&format!("/api/runs/{}/stop", a.id))).await;
    sse_a.until_status(&["stopped"]).await;
    let seen = sse_c.until_status(&["running"]).await;
    let last_queue = seen.iter().rev().find(|(n, _)| n == "queue");
    assert!(last_queue.is_none_or(|(_, v)| v["position"] == 1), "{seen:?}");
    until_queue(&mut sse_b, json!({ "position": 2, "waiting": 2 })).await;
    err(&app, mv(&c, json!({ "action": "bottom" })), StatusCode::CONFLICT, "run_not_queued").await;
    // Stopping a waiting run moves the ones behind it up.
    let _: Run = ok(&app, post(&format!("/api/runs/{d}/stop"))).await;
    until_queue(&mut sse_b, json!({ "position": 1, "waiting": 1 })).await;
    assert_eq!(run(&b).await.queue_position, Some(1));
    for id in [&c, &b] {
        let _: Run = ok(&app, post(&format!("/api/runs/{id}/stop"))).await;
    }
    drop((sse_a, sse_b, sse_c));
    drop(app);
    s.shutdown();
    drop(s);

    // After a restart `resolved` comes from run.json; a run recorded before the field gets it from config.json
    // (also from one written before config.json had `resolved`).
    let dir = home.join("runs").join(&a.id);
    for (file, key) in [("run.json", "run"), ("config.json", "")] {
        let p = dir.join(file);
        let mut v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        let obj = if key.is_empty() { &mut v } else { &mut v[key] };
        assert!(obj.as_object_mut().unwrap().remove("resolved").is_some(), "{file}");
        std::fs::write(&p, v.to_string()).unwrap();
    }
    let s = store(&home);
    let app = self::app(&s);
    assert_eq!(ok::<Run>(&app, get(&format!("/api/runs/{}", a.id))).await.resolved, Some(resolved));
    let d_run: Run = ok(&app, get(&format!("/api/runs/{d}"))).await;
    assert_eq!((d_run.resolved, d_run.queue_position), (None, None), "stopped before its trainer was built");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_runs_start_after_shutdown() {
    let t = TestDir::new("runs-shutdown");
    let s = store(&t.join("home"));
    let app = app(&s);
    let ds = dataset(&app, "tiny", 4).await;
    // Held after its first epoch of two, so the stop always wins (it cannot complete first).
    s.hold_runs_after_epochs(true);
    let stopped: Run = ok(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    ok::<Run>(&app, post(&format!("/api/runs/{}/stop", stopped.id))).await;
    let s2 = s.clone();
    tokio::task::spawn_blocking(move || s2.shutdown()).await.unwrap();
    let e = err(&app, post_json("/api/runs", tiny(&ds, 1)), StatusCode::SERVICE_UNAVAILABLE, "shutting_down")
        .await;
    assert!(e.hint.is_some());
    let resume = json!({ "resume_run_id": stopped.id });
    err(&app, post_json("/api/runs", resume), StatusCode::SERVICE_UNAVAILABLE, "shutting_down").await;
    assert_eq!(s.list_runs().unwrap().len(), 1, "a refused run is not created");
}

/// A stop after the last batch (the run held after its only epoch: validated, nothing left to train) is accepted as
/// `stopping`, but the run completes with its model, and the log's acknowledgment says that this can happen; no
/// line says it stopped (#333). The run is read by polling it, so no event can be missed.
#[tokio::test(flavor = "multi_thread")]
async fn a_stop_after_the_last_batch_completes_the_run() {
    let t = TestDir::new("runs-stop-last");
    let s = store(&t.join("home"));
    let app = app(&s);
    let ds = dataset(&app, "last", 4).await;
    s.hold_runs_after_epochs(true);
    let run: Run = ok(&app, post_json("/api/runs", tiny(&ds, 1))).await;
    let until = async |done: fn(&Run) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            let now: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
            if done(&now) {
                return now;
            }
            assert!(Instant::now() < deadline, "{:?} at epoch {}", now.status, now.epoch);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    until(|r| r.epoch == 1).await;
    let stopping: Run = ok(&app, post(&format!("/api/runs/{}/stop", run.id))).await;
    assert_eq!(stopping.status, RunStatus::Stopping);
    let done = until(|r| !r.status.is_active()).await;
    assert_eq!((done.status, done.epoch, done.model_id.is_some()), (RunStatus::Completed, 1, true));
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    let messages: Vec<&str> = logs.lines.iter().map(|l| l.message.as_str()).collect();
    assert_eq!(messages.iter().filter(|m| **m == folo_server::STOP_REQUESTED).count(), 1, "{messages:?}");
    assert!(!messages.iter().any(|m| m.starts_with("stopped")), "{messages:?}");
    s.hold_runs_after_epochs(false);
}
