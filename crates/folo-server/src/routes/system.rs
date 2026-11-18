//! `/api/health` and `/api/system`: health, system info, the throughput table and benchmark probes.

use axum::Json;
use axum::extract::State;
use folo_api::{
    API_VERSION, BenchmarkReport, BenchmarkRequest, Health, SystemInfo, SystemResponse, ThroughputTable,
};

use crate::AppState;
use crate::error::{ApiJson, Error, Result};

use super::blocking;

// ---- system ------------------------------------------------------------------------------------------------------

pub(super) async fn health(State(st): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        api: API_VERSION.into(),
        mock: st.store.is_mock(),
        features: features(&st),
    })
}

/// The features both `/api/health` and `/api/system` report. Cheap: no hardware probe, so a client that only
/// needs to know what the server can do asks `/api/health`.
fn features(st: &AppState) -> Vec<String> {
    let always = [
        folo_api::FEATURE_EXPORT_ONNX_SWITCHES,
        folo_api::FEATURE_QUEUE_CONTROL,
        folo_api::FEATURE_WEIGHTS_ONNX,
        folo_api::FEATURE_THROUGHPUT,
        folo_api::FEATURE_RUN_ESTIMATE,
        folo_api::FEATURE_BENCHMARK,
        folo_api::FEATURE_RUN_CLI,
    ];
    let labeling = st.store.labeler().is_ok();
    let optional = [
        (st.web_wasm, folo_api::FEATURE_WEB_WASM),
        (labeling, folo_api::FEATURE_AGENT_LABELING),
        (labeling, folo_api::FEATURE_LABEL_IMAGE_FILTERS),
        (labeling, folo_api::FEATURE_LABEL_REVISIONS),
        (st.store.benches().is_ok(), folo_api::FEATURE_VISION_BENCH),
    ];
    always
        .iter()
        .chain(st.store.features())
        .copied()
        .chain(optional.into_iter().filter(|(on, _)| *on).map(|(_, f)| f))
        .map(str::to_string)
        .collect()
}

pub(super) async fn system(State(st): State<AppState>) -> Result<Json<SystemResponse>> {
    let probe = st.probe.clone();
    let store = st.store.clone();
    let ((cpu, memory), capabilities, gpu_pool) =
        tokio::task::spawn_blocking(move || (probe.sample(), store.capabilities(), store.gpu_pool()))
            .await
            .map_err(|e| Error::internal(format!("task failed: {e}")))?;
    let data_dir = st.store.data_dir().map(folo_api::json::path);
    let features = features(&st);
    Ok(Json(SystemResponse {
        system: SystemInfo { cpu, memory, gpus: st.store.gpus() },
        data_dir,
        features,
        capabilities,
        gpu_pool,
    }))
}

pub(super) async fn throughput(State(st): State<AppState>) -> Result<Json<ThroughputTable>> {
    Ok(Json(blocking(&st, |s| s.throughput()).await?))
}

pub(super) async fn benchmark(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<BenchmarkRequest>,
) -> Result<Json<BenchmarkReport>> {
    Ok(Json(blocking(&st, move |s| s.benchmark(req)).await?))
}
