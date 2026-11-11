//! Runs: their lifecycle, SSE events, log history and queue control.

use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::harness::{err, get, mock, ok, post_json, read_events, send};

#[tokio::test]
async fn runs_lifecycle() {
    let (server, store) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let statuses: Vec<_> = list.runs.iter().map(|r| r.status).collect();
    assert!(statuses.contains(&RunStatus::Running));
    assert!(statuses.contains(&RunStatus::Completed));
    assert!(statuses.contains(&RunStatus::Failed));
    assert!(statuses.contains(&RunStatus::Stopped));
    let failed = list.runs.iter().find(|r| r.status == RunStatus::Failed).unwrap();
    assert!(failed.error.as_deref().unwrap().contains("out of memory"));
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap();
    assert!(live.epoch > 10 && live.eta_s.is_some() && live.best.is_some());

    // Finished run: full history.
    let done = list.runs.iter().find(|r| r.status == RunStatus::Completed).unwrap();
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", done.id))).await;
    assert_eq!(m.steps.len() as u64, done.step);
    assert_eq!(m.epochs.len() as u32, done.epochs);
    let first = &m.epochs[0];
    let last = m.epochs.last().unwrap();
    assert!(last.train_loss.total < first.train_loss.total);
    assert!(last.val.map50_95 > first.val.map50_95);
    assert_eq!(done.best.unwrap().map50_95, m.epochs.iter().map(|e| e.val.map50_95).fold(0.0, f32::max));
    let thin: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics?every=100", done.id))).await;
    assert_eq!(thin.steps.last().unwrap().step, done.step);
    assert!(thin.steps.len() as u64 <= done.step / 100 + 1);

    let s: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples?epoch=5", done.id))).await;
    assert_eq!((s.epoch, s.samples.len()), (5, 16));
    assert!(s.samples.iter().all(|x| x.pred.iter().all(|p| p.score >= 0.25)));
    let latest: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples", done.id))).await;
    assert_eq!(latest.epoch, done.epochs);
    err(
        &app,
        get(&format!("/api/runs/{}/samples?epoch=9999", done.id)),
        StatusCode::NOT_FOUND,
        "epoch_not_available",
    )
    .await;

    // Create, stop, delete.
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let cfg = json!({ "dataset_id": ds.datasets[0].id, "arch": "folo-s", "epochs": 5 });
    let (status, _, b) = send(&app, post_json("/api/runs", cfg)).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let run: Run = serde_json::from_slice(&b).unwrap();
    assert_eq!(run.status, RunStatus::Queued);
    assert_eq!(run.name, "shapes-1k-s");
    assert_eq!(run.config.batch, 16);
    let list: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(list.runs[0].id, run.id, "newest first");
    err(
        &app,
        Request::delete(format!("/api/runs/{}", run.id)).body(Body::empty()).unwrap(),
        StatusCode::CONFLICT,
        "run_active",
    )
    .await;
    let stopped: Run =
        ok(&app, Request::post(format!("/api/runs/{}/stop", run.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(stopped.status, RunStatus::Stopping);
    err(
        &app,
        Request::post(format!("/api/runs/{}/stop", done.id)).body(Body::empty()).unwrap(),
        StatusCode::CONFLICT,
        "run_not_active",
    )
    .await;
    // A mock run takes 1.5 s to stop.
    store.pass_time(Duration::from_secs(2));
    let r: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(r.status, RunStatus::Stopped);
    let (status, _, _) =
        send(&app, Request::delete(format!("/api/runs/{}", run.id)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    err(&app, get(&format!("/api/runs/{}", run.id)), StatusCode::NOT_FOUND, "run_not_found").await;

    // Validation.
    let bad = json!({ "dataset_id": ds.datasets[0].id, "imgsz": 100 });
    let e = err(&app, post_json("/api/runs", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert!(e.message.contains("imgsz"));
    // Run names are at most 64 characters, as dataset names are: 64 is fine, 65 is not.
    let long = json!({ "dataset_id": ds.datasets[0].id, "name": "x".repeat(65) });
    let e = err(&app, post_json("/api/runs", long), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert_eq!(e.message, "name: must be at most 64 characters");
    let fits = json!({ "dataset_id": ds.datasets[0].id, "name": "é".repeat(64) });
    let fits: Run = ok(&app, post_json("/api/runs", fits)).await;
    assert_eq!(fits.name.chars().count(), 64);
    // An optimizer this build does not know decodes (Optimizer::Unknown) but is refused, not trained with.
    let bad = json!({ "dataset_id": ds.datasets[0].id, "optimizer": "lion" });
    let e = err(&app, post_json("/api/runs", bad), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert!(e.message.contains("optimizer"), "{}", e.message);
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": "ds_none" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
    err(
        &app,
        post_json("/api/runs", json!({ "epochs": 3 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    let (status, _, _) =
        send(&app, Request::post("/api/runs").body(Body::from(r#"{"dataset_id":"x"}"#)).unwrap()).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn sse_live_run_streams_and_cleans_up() {
    let (server, store) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap().id.clone();

    // Two subscribers at once.
    let uri = format!("/api/runs/{live}/events");
    let res_a = app.clone().oneshot(get(&uri)).await.unwrap();
    let res_b = app.clone().oneshot(get(&uri)).await.unwrap();
    assert_eq!(res_a.status(), StatusCode::OK);
    assert_eq!(res_a.headers()[header::CONTENT_TYPE], "text/event-stream");
    assert_eq!(store.hub_state(&live), (2, true));

    let mut body_a = res_a.into_body();
    let mut body_b = res_b.into_body();
    let a = read_events(&mut body_a, 4, Duration::from_secs(5)).await;
    let b = read_events(&mut body_b, 2, Duration::from_secs(5)).await;
    assert_eq!(a.len(), 4, "{a:?}");
    assert_eq!(a[0].0, "status");
    let st: StatusEvent = serde_json::from_str(&a[0].1).unwrap();
    assert_eq!(st.status, RunStatus::Running);
    // The events carry the run's length, so clients need not re-read the run.
    let run: Run = ok(&app, get(&format!("/api/runs/{live}"))).await;
    assert!(run.steps_per_epoch > 0);
    assert_eq!((st.epochs, st.steps_per_epoch), (run.epochs, run.steps_per_epoch));
    for (name, data) in &a[1..] {
        let ev = RunEvent::from_sse(name, data).expect("known event").expect("valid payload");
        if let RunEvent::Step(s) = ev {
            assert!(s.step > 0 && s.lr > 0.0 && s.loss.total > 0.0);
            assert_eq!((s.epochs, s.steps_per_epoch), (run.epochs, run.steps_per_epoch));
        }
    }
    assert!(a.iter().any(|(n, _)| n == "step"));
    assert_eq!(b[0].0, "status");

    // Disconnect one client: the other keeps streaming, the hub notices.
    drop(body_a);
    let more = read_events(&mut body_b, 2, Duration::from_secs(5)).await;
    assert_eq!(more.len(), 2);
    assert_eq!(store.hub_state(&live).0, 1);

    // Disconnect the last one: the ticker exits on its next tick.
    drop(body_b);
    let t0 = tokio::time::Instant::now();
    while store.hub_state(&live) != (0, false) {
        assert!(
            t0.elapsed() < folo_server::mock::TICK * 20,
            "the ticker still runs: {:?}",
            store.hub_state(&live)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // A new subscriber restarts it; shutting the server down ends the stream.
    let res = app.clone().oneshot(get(&uri)).await.unwrap();
    let mut body = res.into_body();
    assert_eq!(read_events(&mut body, 1, Duration::from_secs(5)).await.len(), 1);
    server.close_streams();
    let end = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match body.frame().await {
                None => break,
                Some(_) => continue,
            }
        }
    })
    .await;
    assert!(end.is_ok(), "stream ended after close_streams");
}

#[tokio::test]
async fn sse_finished_run_and_errors() {
    let (server, store) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let failed = list.runs.iter().find(|r| r.status == RunStatus::Failed).unwrap();
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", failed.id))).await.unwrap();
    let mut body = res.into_body();
    let ev = read_events(&mut body, 1, Duration::from_secs(2)).await;
    let st: StatusEvent = serde_json::from_str(&ev[0].1).unwrap();
    assert_eq!(st.status, RunStatus::Failed);
    assert!(st.error.is_some());
    assert_eq!(store.hub_state(&failed.id), (1, false), "finished runs need no ticker");
    err(&app, get("/api/runs/run_none/events"), StatusCode::NOT_FOUND, "run_not_found").await;
}

#[tokio::test]
async fn sse_first_subscriber_sees_queued_to_running() {
    let (server, store) = mock();
    let app = server.router();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let run: Run =
        ok(&app, post_json("/api/runs", json!({ "dataset_id": ds.datasets[0].id, "epochs": 3 }))).await;
    assert_eq!(run.status, RunStatus::Queued);
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", run.id))).await.unwrap();
    let mut body = res.into_body();
    // A new mock run waits 2 s in the queue.
    store.pass_time(Duration::from_secs(2));
    let mut statuses = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    'outer: loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let evs = read_events(&mut body, 1, left).await;
        assert!(!evs.is_empty(), "no step within 5 s; statuses {statuses:?}");
        for (name, data) in evs {
            match name.as_str() {
                "status" => statuses.push(serde_json::from_str::<StatusEvent>(&data).unwrap().status),
                "step" => break 'outer,
                _ => {}
            }
        }
    }
    assert_eq!(
        statuses,
        [RunStatus::Queued, RunStatus::Running],
        "every transition is reported before steps"
    );
}

#[tokio::test]
async fn run_logs_history() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let by = |s: RunStatus| list.runs.iter().find(|r| r.status == s).unwrap().clone();

    // A run trained here (a cloud run's log starts with its launch and ends with its release).
    let done =
        list.runs.iter().find(|r| r.status == RunStatus::Completed && r.cloud.is_none()).unwrap().clone();
    let all: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=5000", done.id))).await;
    let msgs: Vec<&str> = all.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(msgs[0].starts_with("training "));
    assert!(msgs.last().unwrap().starts_with("training complete"));
    let epochs = msgs.iter().filter(|m| m.starts_with("epoch ")).count();
    assert_eq!(epochs as u32, done.epochs);
    assert!(all.lines.windows(2).all(|w| w[0].t <= w[1].t), "chronological");
    let tail: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=5", done.id))).await;
    assert_eq!(tail.lines.as_slice(), &all.lines[all.lines.len() - 5..]);
    let default: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", done.id))).await;
    assert_eq!(default.lines.len(), all.lines.len().min(500));

    let failed: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", by(RunStatus::Failed).id))).await;
    let last = failed.lines.last().unwrap();
    assert_eq!(last.level, LogLevel::Error);
    assert!(last.message.contains("out of memory"));
    let stopped: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", by(RunStatus::Stopped).id))).await;
    assert!(stopped.lines.last().unwrap().message.starts_with("stopped at step"));

    // The live run: history up to now, in the same format as the SSE `log` events.
    let live = by(RunStatus::Running);
    let hist: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", live.id))).await;
    let n = hist.lines.iter().filter(|l| l.message.starts_with("epoch ")).count() as u32;
    assert!(
        n + 1 >= live.epoch.saturating_sub(1) && n <= live.epoch,
        "{n} epochs logged, run at {}",
        live.epoch
    );

    err(&app, get(&format!("/api/runs/{}/logs?limit=x", done.id)), StatusCode::BAD_REQUEST, "invalid_query")
        .await;
    err(&app, get("/api/runs/run_none/logs"), StatusCode::NOT_FOUND, "run_not_found").await;
}

#[tokio::test]
async fn logs_of_a_run_stopped_while_queued_match_sse() {
    let (server, store) = mock();
    let app = server.router();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let run: Run = ok(&app, post_json("/api/runs", json!({ "dataset_id": ds.datasets[0].id }))).await;
    let res = app.clone().oneshot(get(&format!("/api/runs/{}/events", run.id))).await.unwrap();
    let mut body = res.into_body();
    let stop = Request::post(format!("/api/runs/{}/stop", run.id)).body(Body::empty()).unwrap();
    let r: Run = ok(&app, stop).await;
    assert_eq!(r.status, RunStatus::Stopping);
    // Collect SSE log lines until the final `stopped` status. A mock run takes 1.5 s to stop: they pass once the
    // stream said `stopping`.
    let mut sse_logs = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    'outer: loop {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let evs = read_events(&mut body, 1, left).await;
        assert!(!evs.is_empty(), "no stopped status; logs so far {sse_logs:?}");
        for (name, data) in evs {
            match name.as_str() {
                "log" => sse_logs.push(serde_json::from_str::<LogLine>(&data).unwrap().message),
                "status" => match serde_json::from_str::<StatusEvent>(&data).unwrap().status {
                    RunStatus::Stopping => store.pass_time(Duration::from_secs(2)),
                    RunStatus::Stopped => break 'outer,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    let hist: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    let hist: Vec<String> = hist.lines.into_iter().map(|l| l.message).collect();
    assert_eq!(hist.len(), 2, "{hist:?}");
    assert!(hist[0].starts_with("stop requested") && hist[1].starts_with("stopped at step 0"));
    assert_eq!(hist, sse_logs, "history matches what the stream sent");
}

/// docs/API.md `POST /api/runs/{id}/queue` in mock mode: runs created through the API share one training slot;
/// moves, `start`, the `queue` events, `Run.resolved` once a run starts, and the errors.
#[tokio::test]
async fn mock_queue_control() {
    let (server, store) = mock();
    let app = server.router();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let ds = ds.datasets[0].id.clone();
    let mut ids = Vec::new();
    for (i, name) in ["qa", "qb", "qc"].into_iter().enumerate() {
        let r: Run =
            ok(&app, post_json("/api/runs", json!({ "dataset_id": ds, "name": name, "epochs": 50 }))).await;
        assert_eq!((r.status, r.queue_position), (RunStatus::Queued, Some(i as u32 + 1)), "{name}");
        assert!(r.resolved.is_none(), "not started yet");
        ids.push(r.id);
    }
    let (a, b, c) = (ids[0].clone(), ids[1].clone(), ids[2].clone());
    let pos = |id: &str| {
        let (app, id) = (app.clone(), id.to_string());
        async move { ok::<Run>(&app, get(&format!("/api/runs/{id}"))).await.queue_position }
    };
    let mv = |id: &str, body: Value| post_json(&format!("/api/runs/{id}/queue"), body);
    // The seeded live run is outside the queue; the list carries positions too.
    let list: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(list.runs.iter().find(|r| r.id == "run_3c02").unwrap().queue_position, None);
    assert_eq!(list.runs.iter().find(|r| r.id == b).unwrap().queue_position, Some(2));

    // Follow b's stream from here.
    let res = app.clone().oneshot(get(&format!("/api/runs/{b}/events"))).await.unwrap();
    let mut events = res.into_body();

    let r: Run = ok(&app, mv(&c, json!({ "action": "top" }))).await;
    assert_eq!(r.queue_position, Some(1));
    assert_eq!((pos(&a).await, pos(&b).await), (Some(2), Some(3)));
    let r: Run = ok(&app, mv(&c, json!({ "action": "up" }))).await;
    assert_eq!(r.queue_position, Some(1), "up at the front is a no-op");
    let r: Run = ok(&app, mv(&c, json!({ "action": "down" }))).await;
    assert_eq!(r.queue_position, Some(2));
    let r: Run = ok(&app, mv(&c, json!({ "position": 99 }))).await;
    assert_eq!(r.queue_position, Some(3), "past the end is the end");
    let r: Run = ok(&app, mv(&a, json!({ "action": "bottom" }))).await;
    assert_eq!(r.queue_position, Some(3));
    assert_eq!((pos(&b).await, pos(&c).await), (Some(1), Some(2)));

    // Nothing trains (the first run waits 2 s), so `start` starts c at once; b and a wait behind it.
    let r: Run = ok(&app, mv(&c, json!({ "action": "start" }))).await;
    assert!(r.queue_position.is_none() && r.status == RunStatus::Running, "{r:?}");
    let resolved = r.resolved.expect("resolved once it started");
    assert_ne!(serde_json::to_value(resolved.optimizer).unwrap(), "auto");
    assert!(resolved.lr0 > 0.0 && resolved.accumulate >= 1 && resolved.warmup_iters > 0);
    assert_eq!((pos(&b).await, pos(&a).await), (Some(1), Some(2)));
    store.pass_time(Duration::from_millis(2500));
    let r: Run = ok(&app, get(&format!("/api/runs/{b}"))).await;
    assert_eq!((r.status, r.queue_position), (RunStatus::Queued, Some(1)), "the slot is taken");
    // `start` with a run training only moves to the front.
    let r: Run = ok(&app, mv(&a, json!({ "action": "start" }))).await;
    assert_eq!((r.status, r.queue_position), (RunStatus::Queued, Some(1)));

    // b's stream reported its moves: the last `queue` event says where it is now (2 of 2).
    let mut last = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while last != Some(QueueEvent { position: 2, waiting: 2 }) {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        let evs = read_events(&mut events, 1, left).await;
        assert!(!evs.is_empty(), "no queue event; last {last:?}");
        for (name, data) in evs {
            if name == "queue" {
                last = Some(serde_json::from_str(&data).unwrap());
            }
        }
    }

    // Moves are logged: in `/logs`, and live on the moved run's stream.
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{c}/logs"))).await;
    let msgs: Vec<&str> = logs.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(msgs.contains(&"moved in the queue to position 1 of 3"), "{msgs:?}");
    assert!(msgs.contains(&"start requested: starting now"), "{msgs:?}");
    let r: Run = ok(&app, mv(&b, json!({ "action": "down" }))).await;
    assert_eq!(r.queue_position, Some(2));
    let mut logged = false;
    while !logged {
        let evs = read_events(&mut events, 1, Duration::from_secs(3)).await;
        assert!(!evs.is_empty(), "no log event for the move");
        logged = evs.iter().any(|(n, d)| n == "log" && d.contains("queue position unchanged (2 of 2)"));
    }

    // Errors: a running or finished run, bad bodies, unknown runs.
    let e = err(&app, mv(&c, json!({ "action": "top" })), StatusCode::CONFLICT, "run_not_queued").await;
    assert!(e.message.contains("running"), "{}", e.message);
    err(&app, mv("run_1a01", json!({ "action": "top" })), StatusCode::CONFLICT, "run_not_queued").await;
    err(&app, mv("run_3c02", json!({ "position": 1 })), StatusCode::CONFLICT, "run_not_queued").await;
    for body in [json!({}), json!({ "action": "top", "position": 1 }), json!({ "position": 0 })] {
        err(&app, mv(&b, body), StatusCode::BAD_REQUEST, "invalid_body").await;
    }
    err(&app, mv(&b, json!({ "action": "sideways" })), StatusCode::UNPROCESSABLE_ENTITY, "invalid_body")
        .await;
    err(&app, mv("run_none", json!({ "action": "top" })), StatusCode::NOT_FOUND, "run_not_found").await;

    // Stopping the running run lets the front run start.
    let _: Run = ok(&app, Request::post(format!("/api/runs/{c}/stop")).body(Body::empty()).unwrap()).await;
    store.pass_time(Duration::from_millis(1800));
    let r: Run = ok(&app, get(&format!("/api/runs/{a}"))).await;
    assert_eq!((r.status, r.queue_position), (RunStatus::Running, None));
    assert_eq!(pos(&b).await, Some(1));
    for id in [&a, &b] {
        let _: Run =
            ok(&app, Request::post(format!("/api/runs/{id}/stop")).body(Body::empty()).unwrap()).await;
    }
}
