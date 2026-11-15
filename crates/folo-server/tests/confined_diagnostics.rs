//! The hints of a dataset registration that failed honour the process-wide read guard (`folo mcp --confine`): a
//! folder or YAML it refuses is not looked into to phrase the error (review round 3). A separate test binary,
//! because a guard stays installed for the whole process: here it refuses any `forbidden` folder and any
//! `custom.yaml`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::{Options, RealStore, Server};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

#[tokio::test]
async fn a_failed_registration_does_not_read_what_the_guard_refuses() {
    assert!(folo_data::guard::install(|p: &Path| {
        !p.components().any(|c| c.as_os_str() == "forbidden")
            && p.file_name().is_none_or(|n| n != "custom.yaml")
    }));
    let t = TestDir::new("confined-diagnostics");
    let home = t.join("home");
    let yaml = "path: .\ntrain: images\nnames: [x]\n";
    for (dir, file) in [("a/child", "custom.yaml"), ("b/forbidden", "data.yaml"), ("c/open", "other.yaml")] {
        std::fs::create_dir_all(t.join(dir).join("images")).unwrap();
        std::fs::write(t.join(dir).join(file), yaml).unwrap();
    }
    let store = RealStore::open(home).recheck_every(Duration::ZERO);
    let app = Server::new(Arc::new(store), Options::default()).router();
    let mut said = Vec::new();
    for parent in ["a", "b", "c"] {
        let req = Request::post("/api/datasets")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "path": t.join(parent) }).to_string()))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{parent}");
        let body: Value =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        said.push(body["error"]["message"].as_str().unwrap().replace('\\', "/"));
    }
    let names_inside = |m: &str| m.ends_with("inside it holds a dataset");
    assert!(!names_inside(&said[0]), "a refused YAML is not read: {}", said[0]);
    assert!(!names_inside(&said[1]), "a refused folder is not looked into: {}", said[1]);
    assert!(names_inside(&said[2]) && said[2].contains("c/open"), "an allowed one is: {}", said[2]);
}
