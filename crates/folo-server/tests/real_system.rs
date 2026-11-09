//! The real store's `/api/system` `capabilities` agree with what a run would get: the trainer's own `amp_support`
//! for the device `auto` resolves to, the GPU context's tensor-core status (what `folo doctor` prints) and f16.
//!
//! Its own test binary: it asks for tensor cores (as `folo serve --tensor-cores` does) before the shared GPU
//! context exists, which would change every other test in a shared process. Without a GPU, or on an adapter
//! without tensor cores, it checks the refusal and its reason instead.

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use folo_server::api::*;
use folo_server::{Options, RealStore, Server};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn real_capabilities_match_the_trainer() {
    // Before anything creates the shared context.
    folo_gpu::context::request_tensor_cores();
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let home = TempDir(std::env::temp_dir().join(format!("folo-real-system-{}-{nanos}", std::process::id())));
    let app = Server::new(Arc::new(RealStore::open(home.0.clone())), Options::default()).router();
    let res = app.oneshot(Request::get("/api/system").body(Body::empty()).unwrap()).await.unwrap();
    assert!(res.status().is_success());
    let raw: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let c: Capabilities = serde_json::from_value(raw["capabilities"].clone()).expect("capabilities");
    eprintln!("capabilities: {c:?}");
    for k in ["device", "shader_f16", "tensor_cores", "amp"] {
        assert!(raw["capabilities"].get(k).is_some(), "{k} missing: {raw}");
    }

    // The same device a `device: auto` run trains on, and the trainer's answer for it, word for word.
    let device = folo_vision::train::resolve_device("auto").unwrap();
    assert_eq!(c.device, device.to_string());
    match folo_vision::train::amp_support(device) {
        Ok(()) => assert_eq!(c.amp, AmpSupport { available: true, reason: None }),
        Err(e) => assert_eq!(c.amp, AmpSupport { available: false, reason: Some(e.to_string()) }),
    }
    assert_eq!(c.shader_f16, folo_vision::predict::f16_unavailable(device).is_none());
    assert_eq!(c.tensor_cores.enabled, c.tensor_cores.reason.is_none(), "{c:?}");
    assert!(!c.tensor_cores.enabled || c.tensor_cores.available, "{c:?}");

    match folo_gpu::context_for(device) {
        None => {
            assert_eq!(c.device, "cpu");
            assert!(!c.tensor_cores.available && !c.amp.available && !c.shader_f16, "{c:?}");
            assert!(c.tensor_cores.reason.unwrap().starts_with("off"));
            assert!(c.amp.reason.unwrap().contains("the run is on the CPU"));
        }
        Some(ctx) => {
            assert_eq!(c.tensor_cores.available, ctx.summary().tensor_cores);
            assert_eq!(c.tensor_cores.enabled, ctx.tensor_cores());
            if let Some(why) = &c.tensor_cores.reason {
                assert_eq!(why, &ctx.tensor_cores_status(), "the status `folo doctor` prints");
            }
            // Tensor cores were asked for: an adapter that qualifies has them (a failed self-test is a bug).
            let qualifies = folo_gpu::kernels::coop::adapter_unavailable_reason(ctx.adapter()).is_none();
            let no_f16 = std::env::var("FOLO_GPU_NO_F16").is_ok_and(|v| v != "0" && !v.is_empty());
            if qualifies && !no_f16 {
                assert!(c.tensor_cores.enabled && c.amp.available, "{c:?}");
            } else {
                assert!(
                    !c.amp.available && c.amp.reason.as_deref().unwrap().contains("tensor cores"),
                    "{c:?}"
                );
            }
        }
    }
}
