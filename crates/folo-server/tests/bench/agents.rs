//! Agents: agent entries under the labeling policy, and designing a plan with an agent.

use std::sync::atomic::Ordering;

use axum::http::StatusCode;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{
    Step, axis, bench, dataset, delete, ended, err, fixture, get, ok, page, paid, post, put, send, small_plan,
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

    // Defaults: the first 48 images (here all 4), unchanged only; starts requesting confidences (`scored: true`),
    // and falls back to `scored: false` when the default scripted reply omits confidences.
    let e: BenchEntry = ok(app, post(&entries, agent.clone())).await;
    assert_eq!((e.name.as_str(), e.images, e.conditions, e.scored), ("Claude Code (small)", 4, false, true));
    assert!(
        matches!(&e.contestant, BenchContestant::Agent { agent: LabelAgentId::Claude, model: Some(m), effort: LabelEffort::Medium } if m == "small")
    );
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    assert!(!done.scored, "legacy reply without confidences falls back to unscored");
    let usage = done.usage.unwrap();
    assert_eq!((usage.calls, usage.input_tokens, usage.output_tokens), (2, 2000, 100));
    let score = done.score.unwrap();
    assert!(
        score.partial
            && axis(&score, BenchAxis::Calibration).is_none()
            && axis(&score, BenchAxis::Robustness).is_none()
    );
    // Asked with the benchmark confidence prompt/schema, the classes, description, and two images per call.
    let prompts = f.agent.prompts.lock().unwrap().clone();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0], ("Classes: circle, square\nInstructions: Count the shapes.\n2 images follow, in order: image 1 (64 x 64 pixels), image 2 (64 x 64 pixels).".to_string(), 2));
    assert_eq!(f.agent.systems.lock().unwrap()[0], folo_server::label::prompt::BENCH_SYSTEM);
    assert_eq!(
        f.agent.schemas.lock().unwrap()[0]["properties"]["images"]["items"]["properties"]["boxes"]["items"]["minItems"],
        json!(6)
    );
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

#[tokio::test(flavor = "multi_thread")]
async fn agent_entries_score_confidences_and_calibration_with_fallbacks() {
    let t = TestDir::new("bench-agent-conf");
    let f = fixture(&t.join("home"));
    let app = &f.app;
    let ds = dataset(app, "shapes", 20).await;
    // Empty condition plan so a full-image agent entry with confidences is complete (not partial).
    let b = bench(
        app,
        json!({
            "name": "clean-only",
            "description": "Find circles and squares.",
            "source": { "kind": "dataset", "dataset_id": ds.id },
            "plan": { "conditions": [] }
        }),
    )
    .await;
    let imgs = page(app, &b.id, "").await.images;
    assert_eq!(imgs.len(), 4);
    let class_name = |c: u32| b.classes[c as usize].clone();
    let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
    let run_agent = json!({ "contestant": { "kind": "agent", "agent": "claude" } });

    // Helper: turn one image's ground-truth boxes into scored 6-item boxes or object boxes at `conf`.
    let truth_boxes = |im: &BenchImage, conf: f32| -> Vec<serde_json::Value> {
        im.boxes.iter().map(|bx| json!([class_name(bx.class), bx.x1, bx.y1, bx.x2, bx.y2, conf])).collect()
    };
    let obj_boxes = |im: &BenchImage, conf: f32| -> Vec<serde_json::Value> {
        im.boxes
            .iter()
            .map(|bx| json!({ "label": class_name(bx.class), "box": [bx.x1, bx.y1, bx.x2, bx.y2], "confidence": conf }))
            .collect()
    };

    // 1a. Single-batch (max_images: 2) exact hand-worked ECE & calibration verification (mixing 6-item list
    // boxes on image 1 with object boxes carrying `confidence` on image 2).
    // Expected ECE over (N_right at 0.90 + 1 wrong at 0.40):
    //   bin 9 ([0.9, 1.0]): N_right items, mean conf 0.9, prec 1.0 -> |0.9 - 1.0| = 0.1
    //   bin 4 ([0.4, 0.5)): 1 item, mean conf 0.4, prec 0.0 -> |0.4 - 0.0| = 0.4
    //   ECE = (N_right * 0.1 + 1 * 0.4) / (N_right + 1).
    let mut img1_boxes = truth_boxes(&imgs[0], 0.9);
    // 1x1 box of the opposite class at (0,0)-(2,2) cannot match a shape in this image of that class unless one
    // sits at (0,0)-(2,2); check explicitly so the test is 100% deterministic.
    let wrong_box = (0..62)
        .find_map(|y| {
            let cand = (0.0f32, y as f32, 2.0f32, (y + 2) as f32);
            let overlaps = imgs[0].boxes.iter().any(|bx| {
                bx.class == 0 && cand.0 < bx.x2 && cand.2 > bx.x1 && cand.1 < bx.y2 && cand.3 > bx.y1
            });
            (!overlaps).then_some(cand)
        })
        .expect("free 2x2 patch exists");
    img1_boxes.push(json!([class_name(0), wrong_box.0, wrong_box.1, wrong_box.2, wrong_box.3, 0.4]));
    let n_right = (imgs[0].boxes.len() + imgs[1].boxes.len()) as f64;
    assert!(n_right >= 1.0);
    let reply_scored = json!({
        "images": [
            { "image": 1, "boxes": img1_boxes },
            { "image": 2, "boxes": obj_boxes(&imgs[1], 0.9) }
        ]
    });
    f.agent.script(vec![Step::Reply(reply_scored.to_string())]);
    let e: BenchEntry = ok(
        app,
        post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 })),
    )
    .await;
    assert!(e.scored);
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done, "{done:?}");
    assert!(done.scored);
    let res: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    let cal = res.calibration.as_ref().expect("calibration measured for scored agent");
    let expected_ece = (n_right * 0.1 + 0.4) / (n_right + 1.0);
    assert_eq!(cal.detections as usize, imgs[0].boxes.len() + imgs[1].boxes.len() + 1);
    assert!((cal.ece as f64 - expected_ece).abs() < 1e-5, "ece={} expected={expected_ece}", cal.ece);
    let score = done.score.as_ref().unwrap();
    let cal_axis = axis(score, BenchAxis::Calibration).expect("calibration axis populated");
    assert!((cal_axis as f64 - (1.0 - expected_ece)).abs() < 1e-5, "cal_axis={cal_axis}");
    assert_eq!(res.self_labeling.curve.len(), 101, "scored agent gets full 101-point self-labeling curve");
    assert!((res.accuracy.map50_95 - 1.0).abs() < 1e-5, "perfect truth boxes ranked above 0.4 FP");
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images", e.id))).await;
    assert!(gallery.images[0].detections.iter().all(|d| (d.score - 0.9).abs() < 1e-5));

    // 1b. Complete (all 4 images, empty plan) with real confidences -> NOT partial!
    let uniform_scored = json!({
        "images": [
            { "image": 1, "boxes": [["circle", 4, 4, 24, 24, 0.8]] },
            { "image": 2, "boxes": [["square", 4, 4, 24, 24, 0.6]] }
        ]
    });
    f.agent.script(vec![Step::Reply(uniform_scored.to_string()), Step::Reply(uniform_scored.to_string())]);
    let e: BenchEntry = ok(app, post(&entries, run_agent.clone())).await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(done.scored);
    assert!(!done.score.as_ref().unwrap().partial, "complete agent entry with confidences is not partial");
    assert!(axis(done.score.as_ref().unwrap(), BenchAxis::Calibration).is_some());

    // 2. Mixed confidences (one box with confidence, one legacy 5-item box without) -> falls back to unscored
    // so constant 1.0 fallback scores never fabricate a calibration axis.
    let reply_mixed = json!({
        "images": [
            { "image": 1, "boxes": [["circle", 4, 4, 24, 24, 0.85], ["square", 28, 28, 56, 56]] },
            { "image": 2, "boxes": [["circle", 4, 4, 24, 24, 0.75]] }
        ]
    });
    f.agent.script(vec![Step::Reply(reply_mixed.to_string())]);
    let e: BenchEntry = ok(
        app,
        post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 })),
    )
    .await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(!done.scored, "mixed replies must fall back to unscored");
    let res: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(res.calibration, None);
    assert_eq!(res.self_labeling.curve.len(), 1);
    assert!(done.score.as_ref().unwrap().partial);
    let gallery: BenchEntryImagePage = ok(app, get(&format!("{entries}/{}/images", e.id))).await;
    assert!(gallery.images.iter().flat_map(|i| &i.detections).all(|d| d.score == 1.0));

    // 3. Malformed / out-of-range confidences alongside valid ones: bad boxes are dropped, valid ones kept and calibrated.
    let reply_malformed = json!({
        "images": [
            { "image": 1, "boxes": [["circle", 4, 4, 24, 24, 1.5], ["circle", 4, 4, 24, 24, -0.2], ["circle", 4, 4, 24, 24, "high"], ["circle", 4, 4, 24, 24, 0.7]] },
            { "image": 2, "boxes": [["square", 4, 4, 24, 24, 0.8]] }
        ]
    });
    f.agent.script(vec![Step::Reply(reply_malformed.to_string())]);
    let e: BenchEntry = ok(
        app,
        post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 })),
    )
    .await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(done.scored);
    let res: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(
        res.calibration.as_ref().unwrap().detections,
        2,
        "only the 2 valid detections enter calibration"
    );

    // 4. All detections below CALIBRATION_FLOOR (0.1): entry stays scored, calibration is None, partial is true.
    let reply_below_floor = json!({
        "images": [
            { "image": 1, "boxes": [["circle", 4, 4, 24, 24, 0.05]] },
            { "image": 2, "boxes": [["square", 4, 4, 24, 24, 0.0]] }
        ]
    });
    f.agent.script(vec![Step::Reply(reply_below_floor.to_string())]);
    let e: BenchEntry = ok(
        app,
        post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 })),
    )
    .await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(done.scored);
    assert!(done.score.as_ref().unwrap().partial);
    let res: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(res.calibration, None);
    assert_eq!(res.self_labeling.curve.len(), 101);

    // 5. Zero detections across every image: calibration is None and partial is true (never fabricated).
    let reply_empty = json!({ "images": [{ "image": 1, "boxes": [] }, { "image": 2, "boxes": [] }] });
    f.agent.script(vec![Step::Reply(reply_empty.to_string())]);
    let e: BenchEntry = ok(
        app,
        post(&entries, json!({ "contestant": { "kind": "agent", "agent": "claude" }, "max_images": 2 })),
    )
    .await;
    let done = ended(app, &b.id, &e.id).await;
    assert_eq!(done.status, BenchEntryStatus::Done);
    assert!(done.scored);
    assert!(done.score.as_ref().unwrap().partial);
    let res: BenchResult = ok(app, get(&format!("{entries}/{}/result", e.id))).await;
    assert_eq!(res.calibration, None);
}
