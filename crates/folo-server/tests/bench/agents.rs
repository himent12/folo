//! Agents: agent entries under the labeling policy, and designing a plan with an agent.

use std::sync::atomic::Ordering;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{
    Step, axis, bench, dataset, delete, ended, err, fixture, get, ok, paid, post, put, send, small_plan,
};

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
    assert_eq!(prompts[0], ("Classes: circle, square\nInstructions: Count the shapes.\n2 images follow, in order: image 1 (64 x 64 pixels), image 2 (64 x 64 pixels).".to_string(), 2));
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
    // A client error whose text names a server fault: the status counts, so it ends the entry without the
    // "overloaded" hint the text alone would get.
    f.agent.script(vec![Step::FailStatus("API Error: 503 Service Unavailable", 400)]);
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    let failed = ended(app, &b.id, &e.id).await;
    assert_eq!(
        (failed.status, failed.error.as_deref(), failed.hint.as_deref()),
        (BenchEntryStatus::Failed, Some("API Error: 503 Service Unavailable"), None)
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
async fn a_passing_fault_in_an_agent_entry_is_asked_again_once() {
    let t = TestDir::new("bench-agent-transient");
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

    // One call is overloaded after reaching the model, then goes through: the entry is done, and its usage has
    // the failed attempt too.
    f.agent.script(vec![Step::FailPaid("API Error: 529 Overloaded")]);
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    assert_eq!(f.agent.calls.load(Ordering::SeqCst), 3, "two batches, one of them asked twice");
    let usage = done.usage.unwrap();
    assert_eq!((usage.calls, usage.input_tokens), (3, 2000 + paid().input_tokens));

    // Overloaded twice in a row, in every call: the images get no answer and the entry fails.
    f.agent.script((0..4).map(|_| Step::Fail("API Error: 529 Overloaded")).collect());
    let e: BenchEntry = ok(app, post(&entries, agent)).await;
    let failed = ended(app, &b.id, &e.id).await;
    assert_eq!(failed.status, BenchEntryStatus::Failed, "{failed:?}");
    assert_eq!(f.agent.calls.load(Ordering::SeqCst), 3 + 4);
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
