//! System endpoints and the mock's capabilities.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use folo_server::api::*;
use folo_server::{MockStore, Options, Server};
use serde_json::Value;

use super::harness::{err, get, mock, ok};

/// `/api/system`'s `capabilities` in each mock tensor-core state, as front ends see it before submitting a run.
#[tokio::test]
async fn mock_system_capabilities_in_every_state() {
    use folo_server::mock::MockTensorCores as Tc;
    assert_eq!(
        ["", "on", "OFF", " none ", "0", "false", "bogus"].map(Tc::parse),
        [Tc::On, Tc::On, Tc::Off, Tc::None, Tc::None, Tc::None, Tc::On]
    );
    let system = |tc: Tc| async move {
        let app = Server::new(Arc::new(MockStore::new().with_tensor_cores(tc)), Options::default()).router();
        let raw: Value = ok(&app, get("/api/system")).await;
        let s: SystemResponse = serde_json::from_value(raw.clone()).unwrap();
        (raw, s)
    };

    let (raw, s) = system(Tc::On).await;
    let c = s.capabilities.expect("the mock reports capabilities");
    assert_eq!((c.device.as_str(), c.shader_f16), ("gpu:0", true));
    assert!(c.tensor_cores.available && c.tensor_cores.enabled && c.amp.available, "{c:?}");
    assert!(raw["capabilities"]["tensor_cores"]["reason"].is_null(), "{raw}");
    assert!(raw["capabilities"]["amp"]["reason"].is_null(), "{raw}");
    assert_eq!(s.system.gpus[0].backend, GpuBackend::Vulkan);

    let (_, s) = system(Tc::Off).await;
    let c = s.capabilities.unwrap();
    assert!(c.tensor_cores.available && !c.tensor_cores.enabled && !c.amp.available, "{c:?}");
    let why = c.tensor_cores.reason.unwrap();
    assert!(why.starts_with("off (not requested"), "{why}");
    let amp = c.amp.reason.unwrap();
    assert!(amp.starts_with("mixed-precision training (amp) needs") && amp.contains(&why), "{amp}");

    let (_, s) = system(Tc::None).await;
    let c = s.capabilities.unwrap();
    assert!(!c.tensor_cores.available && !c.tensor_cores.enabled && !c.amp.available, "{c:?}");
    assert!(c.tensor_cores.reason.unwrap().contains("no cooperative matrices"));
    assert!(c.amp.reason.unwrap().contains("tensor cores"));
    assert!(c.shader_f16, "f16 inference works without tensor cores");
    assert_eq!(s.system.gpus[0].backend, GpuBackend::Dx12);
}

#[tokio::test]
async fn system_endpoints() {
    let (server, _) = mock();
    let app = server.router();
    let h: Health = ok(&app, get("/api/health")).await;
    assert_eq!((h.status.as_str(), h.mock, h.api.as_str()), ("ok", true, "v1"));
    let s: SystemInfo = ok(&app, get("/api/system")).await;
    assert!(s.cpu.threads >= 1);
    assert!(s.memory.total_bytes > 0);
    assert_eq!(s.gpus.len(), 1);
    err(&app, get("/api/nope"), StatusCode::NOT_FOUND, "route_not_found").await;
    err(
        &app,
        Request::put("/api/health").body(Body::empty()).unwrap(),
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
    )
    .await;
}
