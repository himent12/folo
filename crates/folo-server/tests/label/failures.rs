//! When calls fail: unreadable replies, CLI errors, garbage, timeouts, what start refuses, and the
//! mock without agent labeling.

use std::sync::Arc;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_server::{MockStore, Options, Server};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{Step, boxes_reply, err, fixture, get, idle, images, job_with_images, ok, post};

#[tokio::test]
async fn replies_that_cannot_be_read_fail_their_images_only() {
    let t = TestDir::new("label-bad-replies");
    // Call 1: garbage twice (the retry), so its two images fail. Call 2: one image left out, one box of an unknown
    // class and one good box. Call 3 (the retry of the failed ones, later): fine.
    let partial = json!({ "images": [{ "image": 2, "boxes": [
        { "label": "truck", "box": [0, 0, 500, 500] },
        { "label": "Person", "box": [100, 50, 200, 100] },
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

/// What Claude Code prints when Anthropic's API stays overloaded through its own retries.
const OVERLOADED: &str =
    r#"API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;

#[tokio::test]
async fn a_passing_fault_is_asked_again_once_and_the_run_goes_on() {
    let t = TestDir::new("label-transient");
    // Call 1 is overloaded, then goes through; call 2 is overloaded twice, so its image fails; call 3 is fine.
    let steps = vec![
        Step::Fail(OVERLOADED),
        Step::Reply(boxes_reply(1, "car")),
        Step::Fail("stream disconnected before completion: error sending request"),
        Step::Fail("exceeded retry limit, last status: 503 Service Unavailable"),
    ];
    let f = fixture(&t.join("home"), steps);
    let app = &f.app;
    let job = job_with_images(app, 3).await;
    let uri = format!("/api/label/jobs/{}", job.id);

    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("{uri}/start"), body)).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    assert_eq!((run.outcome, run.done, run.failed), (Some(LabelOutcome::Completed), 2, 1));
    // Two calls for each of the first two images, one for the third; only replies count as usage.
    assert_eq!((f.agent.calls(), run.usage.calls), (5, 2));
    let page = images(app, &job.id, "").await;
    assert_eq!(
        page.images[1].error.as_deref(),
        Some("exceeded retry limit, last status: 503 Service Unavailable")
    );
    let logs: RunLogs = ok(app, get(&format!("{uri}/logs"))).await;
    let text: Vec<&str> = logs.lines.iter().map(|l| l.message.as_str()).collect();
    assert!(
        text.iter().any(|m| m.starts_with("call 1: API Error: 529") && m.ends_with("asking again in 0 s")),
        "{text:?}"
    );
    assert!(text.iter().any(|m| m.starts_with("call 2: stream disconnected")), "{text:?}");
}

#[tokio::test]
async fn passing_faults_in_a_row_end_the_run_and_a_usage_limit_ends_it_at_once() {
    let t = TestDir::new("label-transient-run");
    let f = fixture(&t.join("home"), (0..6).map(|_| Step::Fail(OVERLOADED)).collect());
    let app = &f.app;
    let job = job_with_images(app, 5).await;
    let body = json!({ "agent": "claude", "batch": 1, "parallel": 1 });
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), body.clone())).await;
    let done = idle(app, &job.id).await;
    let run = done.run.unwrap();
    // Three calls (each asked twice) failed: the run gives up as on three timeouts.
    assert_eq!((run.outcome, f.agent.calls()), (Some(LabelOutcome::Failed), 6));
    assert!(run.error.unwrap().starts_with("3 calls in a row failed; the last one: API Error: 529"));
    assert_eq!(done.counts, LabelCounts { images: 5, pending: 2, labeled: 0, failed: 3, boxes: 0 });

    // A usage limit is not a passing fault: no second try, the run ends on it.
    let t = TestDir::new("label-usage-limit");
    let limit = "You've hit your session limit · resets 3pm (Europe/Berlin)";
    let f = fixture(&t.join("home"), vec![Step::Fail(limit)]);
    let app = &f.app;
    let job = job_with_images(app, 2).await;
    ok::<LabelJob>(app, post(&format!("/api/label/jobs/{}/start", job.id), body)).await;
    let run = idle(app, &job.id).await.run.unwrap();
    assert_eq!((run.outcome, f.agent.calls()), (Some(LabelOutcome::Failed), 1));
    assert_eq!(run.error.as_deref(), Some(limit));
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
        // This agent has no faster tier to ask for.
        json!({ "agent": "claude", "fast": true }),
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
    assert_eq!(
        (asked.model.as_deref(), asked.effort, asked.fast),
        (Some("claude-opus-5-5[1m]"), LabelEffort::Medium, false)
    );
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
