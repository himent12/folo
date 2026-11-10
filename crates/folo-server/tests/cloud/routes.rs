//! `/api/cloud` over HTTP.

use std::path::{Path, PathBuf};

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::StatusCode;
use folo_cloud::ErrorKind;
use folo_server::{Options, Server, Store};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::harness::{Call, Fault, SECRET, TestCloud, dataset, fast, finished, no_gpu, plane, until};

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

async fn code(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    status: StatusCode,
    want: &str,
) -> Value {
    let (got, v) = send(app, method, uri, body).await;
    assert_eq!((got, v["error"]["code"].as_str()), (status, Some(want)), "{method} {uri}: {v}");
    v
}

/// Every file under `dir` that holds `needle`.
fn files_with(dir: &Path, needle: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in walk(dir) {
        if std::fs::read(&e).is_ok_and(|b| b.windows(needle.len()).any(|w| w == needle.as_bytes())) {
            out.push(e);
        }
    }
    out
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() { out.extend(walk(&p)) } else { out.push(p) }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cloud_routes_check_redact_cache_and_refuse() {
    no_gpu();
    let t = TestDir::new("cloud-routes");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let app = Server::new(store.clone(), Options::default()).router();

    let (_, system) = send(&app, "GET", "/api/system", None).await;
    assert!(system["features"].as_array().unwrap().iter().any(|f| f == "cloud"));
    let (status, p) = send(&app, "GET", "/api/cloud/providers", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(p["providers"][0]["id"], "test");

    // Connecting checks everything before anything is stored.
    let connect = |values: Value| json!({ "provider": "test", "values": values });
    code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(json!({ "provider": "nope", "values": {} })),
        StatusCode::NOT_FOUND,
        "provider_not_found",
    )
    .await;
    let e = code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({}))),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains("api_key"));
    let e = code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({ "api_key": SECRET, "region": "x" }))),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains("region"));
    code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({ "api_key": "bad-key-0123" }))),
        StatusCode::UNPROCESSABLE_ENTITY,
        "cloud_auth_failed",
    )
    .await;
    assert!(!home.join("cloud").join("connections.json").exists(), "nothing stored");
    let (status, c) =
        send(&app, "POST", "/api/cloud/connections", Some(connect(json!({ "api_key": SECRET })))).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!((c["name"].as_str(), c["values"]["api_key"].as_str()), (Some("Test cloud"), Some("••••6789")));
    let id = c["id"].as_str().unwrap().to_string();
    let (_, c2) =
        send(&app, "POST", "/api/cloud/connections", Some(connect(json!({ "api_key": "short" })))).await;
    assert_eq!((c2["name"].as_str(), c2["values"]["api_key"].as_str()), (Some("Test cloud 2"), Some("••••")));

    // A failed check is an answer, not an error.
    cloud.fail(Call::Check, Fault::Fail(ErrorKind::Auth), 1);
    let (status, checked) = send(&app, "POST", &format!("/api/cloud/connections/{id}/check"), None).await;
    assert_eq!((status, checked["status"].as_str()), (StatusCode::OK, Some("error")));
    assert!(checked["error"].as_str().unwrap().contains("refuses Check"));
    let (_, checked) = send(&app, "POST", &format!("/api/cloud/connections/{id}/check"), None).await;
    assert_eq!((checked["status"].as_str(), checked["error"].as_str()), (Some("ok"), None));
    code(
        &app,
        "POST",
        "/api/cloud/connections/cc_none/check",
        None,
        StatusCode::NOT_FOUND,
        "connection_not_found",
    )
    .await;

    // Offers are cached for a minute, unless refreshed.
    let offers = format!("/api/cloud/connections/{id}/offers");
    let (_, first) = send(&app, "GET", &offers, None).await;
    let (_, second) = send(&app, "GET", &offers, None).await;
    assert_eq!(
        (first["offers"][0]["id"].as_str(), &first["fetched"]),
        (Some("cpu@test"), &second["fetched"])
    );
    assert_eq!(cloud.calls(Call::Offers), 1);
    send(&app, "GET", &format!("{offers}?refresh=true"), None).await;
    assert_eq!(cloud.calls(Call::Offers), 2);

    // Cloud runs are checked like any run, plus compute.
    let ds = dataset(&store);
    let run = |compute: Value, extra: Value| {
        let mut body =
            json!({ "dataset_id": ds.id, "epochs": 1, "batch": 4, "imgsz": 64, "compute": compute });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        Some(body)
    };
    let compute = |offer: &str| json!({ "connection_id": id, "offer_id": offer });
    code(
        &app,
        "POST",
        "/api/runs",
        run(json!({ "connection_id": "cc_none", "offer_id": "cpu@test" }), json!({})),
        StatusCode::NOT_FOUND,
        "connection_not_found",
    )
    .await;
    code(
        &app,
        "POST",
        "/api/runs",
        run(compute("gone@test"), json!({})),
        StatusCode::NOT_FOUND,
        "offer_not_found",
    )
    .await;
    code(
        &app,
        "POST",
        "/api/runs",
        run(compute("cpu@test"), json!({ "device": "cpu" })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    let long = json!({ "connection_id": id, "offer_id": "cpu@test", "max_hours": 1000 });
    code(&app, "POST", "/api/runs", run(long, json!({})), StatusCode::BAD_REQUEST, "invalid_config").await;
    let small = json!({ "connection_id": id, "offer_id": "cpu@test", "disk_gb": 5 });
    code(&app, "POST", "/api/runs", run(small, json!({})), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert!(store.list_runs().unwrap().is_empty(), "refused runs leave nothing");

    // The estimate is the reference's, with what it would cost.
    let (status, est) = send(&app, "POST", "/api/runs/estimate", run(compute("cpu@test"), json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{est}");
    assert_eq!(
        (est["device"].as_str(), est["source"].as_str(), est["confidence"].as_str()),
        (Some("cloud"), Some("reference"), Some("low"))
    );
    assert_eq!(est["cost"]["max_usd"], 4.32);
    assert!(est["cost"]["est_usd"].as_f64().unwrap() <= est["cost"]["est_usd_high"].as_f64().unwrap());

    // A connection with an active run cannot go; one launched instance is listed, and terminated by hand.
    cloud.hold(Call::State);
    let (status, r) = send(&app, "POST", "/api/runs", run(compute("cpu@test"), json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{r}");
    let run_id = r["id"].as_str().unwrap().to_string();
    let e = code(
        &app,
        "DELETE",
        &format!("/api/cloud/connections/{id}"),
        None,
        StatusCode::CONFLICT,
        "connection_in_use",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains(&run_id));
    let instance = until("the instance", || {
        store.cloud_instances().unwrap().into_iter().find(|i| i.provider_ref.is_some())
    });
    cloud.release(Call::State);
    let (status, i) = send(&app, "DELETE", &format!("/api/cloud/instances/{}", instance.id), None).await;
    assert_eq!((status, i["status"].as_str()), (StatusCode::OK, Some("terminated")));
    let (_, again) = send(&app, "DELETE", &format!("/api/cloud/instances/{}", instance.id), None).await;
    assert_eq!(again["ended"], i["ended"], "terminating one that is gone answers its record");
    code(&app, "DELETE", "/api/cloud/instances/ci_none", None, StatusCode::NOT_FOUND, "instance_not_found")
        .await;
    let lost = finished(&store, &run_id);
    assert_eq!(lost.error_code.as_deref(), Some("cloud_instance_lost"));
    let (_, listed) = send(&app, "GET", "/api/cloud/instances", None).await;
    assert_eq!(listed["instances"].as_array().unwrap().len(), 1);
    let (status, _) =
        send(&app, "DELETE", &format!("/api/cloud/connections/{}", c2["id"].as_str().unwrap()), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, list) = send(&app, "GET", "/api/cloud/connections", None).await;
    assert_eq!(list["connections"].as_array().unwrap().len(), 1);

    // The secret is nowhere but in connections.json.
    let (_, everything) = send(&app, "GET", &format!("/api/runs/{run_id}/logs?limit=5000"), None).await;
    assert!(!everything.to_string().contains(SECRET) && !list.to_string().contains(SECRET));
    assert_eq!(files_with(&home, SECRET), [home.join("cloud").join("connections.json")]);
    let shown = format!("{:?}", store.cloud_connections().unwrap());
    assert!(!shown.contains(SECRET));
    drop(app);
}
