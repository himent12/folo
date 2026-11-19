//! The mock's cloud (docs/API.md, Mock mode → Cloud) over HTTP: the real providers, the demo connection and its
//! offers, connecting (a secret with `bad` is rejected), the seeded cloud run `run_3c01`, a new cloud run through
//! its phases with its instance (terminated by hand, it fails with `cloud_instance_lost` and ended in `training`), and
//! `FOLO_MOCK_CLOUD`'s `none` and `off`. The phase a run ended in at each moment: `mock/runs/tests.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use folo_server::api::*;
use folo_server::mock::{MockCloud, PHASE_S};
use folo_server::{MockStore, Options, Server, Store};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn mock(cloud: MockCloud) -> (Arc<MockStore>, Router) {
    let store = Arc::new(MockStore::with_cloud(cloud));
    let router = Server::new(store.clone(), Options::default()).router();
    (store, router)
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let req = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(b) => req.header("content-type", "application/json").body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    };
    let res = app.clone().oneshot(req.unwrap()).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Waits until run `id` satisfies `done`.
async fn wait_run(app: &Router, id: &str, done: impl Fn(&Value) -> bool) -> Value {
    let t0 = Instant::now();
    loop {
        let (_, run) = send(app, "GET", &format!("/api/runs/{id}"), None).await;
        if done(&run) {
            return run;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "run {id} stuck: {run}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn the_mock_cloud_has_a_connection_offers_and_a_finished_cloud_run() {
    let (_, app) = mock(MockCloud::On);
    let (_, system) = send(&app, "GET", "/api/system", None).await;
    assert!(system["features"].as_array().unwrap().iter().any(|f| f == "cloud"));
    let (_, p) = send(&app, "GET", "/api/cloud/providers", None).await;
    let ids: Vec<&str> =
        p["providers"].as_array().unwrap().iter().map(|p| p["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["runpod", "remote"]);
    let (_, list) = send(&app, "GET", "/api/cloud/connections", None).await;
    let c = &list["connections"][0];
    assert_eq!(
        (c["id"].as_str(), c["account"].as_str(), c["balance_usd"].as_f64()),
        (Some("cc_4d2a"), Some("you@example.com"), Some(42.17))
    );
    let (_, offers) = send(&app, "GET", "/api/cloud/connections/cc_4d2a/offers", None).await;
    let offers = offers["offers"].as_array().unwrap();
    assert_eq!(offers.iter().filter(|o| o["availability"] == "none").count(), 1);
    let prices: Vec<f64> = offers.iter().map(|o| o["price_per_hour"].as_f64().unwrap()).collect();
    assert!(prices.windows(2).all(|w| w[0] <= w[1]), "cheapest first: {prices:?}");

    // Any values connect, except a secret with `bad`; secrets come back as a hint.
    let connect = |key: &str| json!({ "provider": "runpod", "values": { "api_key": key } });
    let (status, e) = send(&app, "POST", "/api/cloud/connections", Some(connect("rpa_bad_key"))).await;
    assert_eq!(
        (status, e["error"]["code"].as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some("cloud_auth_failed"))
    );
    let (status, made) =
        send(&app, "POST", "/api/cloud/connections", Some(connect("rpa_0123456789wxyz"))).await;
    assert_eq!(
        (status, made["name"].as_str(), made["values"]["api_key"].as_str()),
        (StatusCode::CREATED, Some("RunPod 2"), Some("••••wxyz"))
    );
    let (status, _) =
        send(&app, "DELETE", &format!("/api/cloud/connections/{}", made["id"].as_str().unwrap()), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    // run_3c01 trained on RunPod: released, with its cost, and its instance terminated.
    let (_, run) = send(&app, "GET", "/api/runs/run_3c01", None).await;
    assert_eq!(
        (run["status"].as_str(), run["config"]["compute"]["connection_id"].as_str()),
        (Some("completed"), Some("cc_4d2a"))
    );
    assert_eq!(run["cloud"]["phase"], "released");
    assert_eq!(run["cloud"]["ended_in"], "collecting", "it trained and copied its weights");
    assert!(run["cloud"]["cost_usd"].as_f64().unwrap() > 0.0);
    let (_, instances) = send(&app, "GET", "/api/cloud/instances", None).await;
    let i = &instances["instances"][0];
    assert_eq!((i["run_id"].as_str(), i["status"].as_str()), (Some("run_3c01"), Some("terminated")));
    assert_eq!(i["cost_usd"], run["cloud"]["cost_usd"]);

    // The estimate of a cloud run says what it would cost.
    let body = json!({ "dataset_id": "ds_5a1e", "compute": { "connection_id": "cc_4d2a", "offer_id": "NVIDIA GeForce RTX 4090@secure" } });
    let (status, est) = send(&app, "POST", "/api/runs/estimate", Some(body)).await;
    assert_eq!((status, est["device"].as_str()), (StatusCode::OK, Some("cloud")));
    assert_eq!(est["cost"]["max_usd"], 8.28);
}

#[tokio::test]
async fn a_mock_cloud_run_hands_over_and_fails_when_its_instance_is_terminated() {
    let (store, app) = mock(MockCloud::On);
    let compute = json!({ "connection_id": "cc_4d2a", "offer_id": "NVIDIA RTX A4000@community" });
    let (status, run) = send(
        &app,
        "POST",
        "/api/runs",
        Some(json!({ "dataset_id": "ds_5a1e", "epochs": 3, "compute": compute })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{run}");
    let id = run["id"].as_str().unwrap().to_string();
    assert_eq!(
        (run["status"].as_str(), run["cloud"]["phase"].as_str()),
        (Some("queued"), Some("provisioning"))
    );
    assert!(run["queue_position"].is_null(), "outside the training queue");
    // Each phase before the training takes `PHASE_S`: they pass one at a time, and the run's stream reports each
    // (its ticker runs every 100 ms, so an event follows the run's state a little).
    let mut sub = store.subscribe(&id).unwrap();
    let mut phases = Vec::new();
    let mut next_phase = async |want: CloudPhase| {
        store.pass_time(Duration::from_secs_f64(PHASE_S));
        while phases.last() != Some(&want) {
            let ev = tokio::time::timeout(Duration::from_secs(5), sub.events.recv()).await.unwrap().unwrap();
            if let RunEvent::Cloud(c) = ev
                && phases.last() != Some(&c.phase)
            {
                phases.push(c.phase);
            }
        }
        send(&app, "GET", &format!("/api/runs/{id}"), None).await.1
    };
    next_phase(CloudPhase::Booting).await;
    let syncing = next_phase(CloudPhase::Syncing).await;
    assert_eq!((&syncing["status"], &syncing["cloud"]["phase"]), (&json!("queued"), &json!("syncing")));
    assert_eq!(syncing["cloud"]["progress"]["total_bytes"], 48_000_000);
    let training = next_phase(CloudPhase::Training).await;
    assert_eq!((&training["status"], &training["cloud"]["phase"]), (&json!("running"), &json!("training")));
    assert!(
        phases.ends_with(&[CloudPhase::Booting, CloudPhase::Syncing, CloudPhase::Training]),
        "{phases:?}"
    );
    let (_, logs) = send(&app, "GET", &format!("/api/runs/{id}/logs"), None).await;
    assert!(logs.to_string().contains("launching RTX A4000 at RunPod"), "{logs}");

    let (_, instances) = send(&app, "GET", "/api/cloud/instances", None).await;
    let i = instances["instances"][0].clone();
    assert_eq!(
        (i["run_id"].as_str(), i["status"].as_str()),
        (Some(id.as_str()), Some("running")),
        "alive first"
    );
    let (status, e) = send(&app, "DELETE", "/api/cloud/connections/cc_4d2a", None).await;
    assert_eq!((status, e["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("connection_in_use")));
    let (status, ended) =
        send(&app, "DELETE", &format!("/api/cloud/instances/{}", i["id"].as_str().unwrap()), None).await;
    assert_eq!((status, ended["status"].as_str()), (StatusCode::OK, Some("terminated")));
    let lost = wait_run(&app, &id, |r| r["status"] == "failed").await;
    assert_eq!(
        (lost["error_code"].as_str(), lost["cloud"]["phase"].as_str(), lost["cloud"]["ended_in"].as_str()),
        (Some("cloud_instance_lost"), Some("released"), Some("training"))
    );
    let (status, e) = send(&app, "DELETE", "/api/cloud/instances/ci_none", None).await;
    assert_eq!((status, e["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("instance_not_found")));
    let (status, e) = send(
        &app,
        "POST",
        "/api/runs",
        Some(json!({ "dataset_id": "ds_5a1e", "device": "cpu", "compute": compute })),
    )
    .await;
    assert_eq!((status, e["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("invalid_config")));
}

#[tokio::test]
async fn folo_mock_cloud_none_and_off() {
    assert_eq!(MockCloud::parse("NONE"), MockCloud::NoConnection);
    assert_eq!(MockCloud::parse("off"), MockCloud::Off);
    assert_eq!(MockCloud::parse(""), MockCloud::On);
    let (store, app) = mock(MockCloud::NoConnection);
    assert!(store.cloud_connections().unwrap().is_empty());
    let (_, run) = send(&app, "GET", "/api/runs/run_3c01", None).await;
    assert_eq!(run["cloud"]["phase"], "released", "the history stays");

    let (store, app) = mock(MockCloud::Off);
    assert!(!store.features().contains(&"cloud"));
    let (status, e) = send(&app, "GET", "/api/cloud/providers", None).await;
    assert_eq!((status, e["error"]["code"].as_str()), (StatusCode::NOT_IMPLEMENTED, Some("not_implemented")));
    let (_, run) = send(&app, "GET", "/api/runs/run_3c01", None).await;
    assert!(run.get("cloud").is_none() && run["config"].get("compute").is_none(), "{run}");
}
