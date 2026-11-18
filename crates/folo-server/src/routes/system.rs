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
    st.system_ready.get_or_try_init(|| prepare_system(&st)).await?;
    let probe = st.probe.clone();
    let store = st.store.clone();
    // Quick once prepared: a CPU and memory reading and what the preparation keeps. `gpus` comes after the device
    // (`capabilities`), which it puts first.
    let ((cpu, memory), capabilities, gpu_pool, gpus) = tokio::task::spawn_blocking(move || {
        (probe.sample(), store.capabilities(), store.gpu_pool(), store.gpus())
    })
    .await
    .map_err(|e| Error::internal(format!("task failed: {e}")))?;
    let data_dir = st.store.data_dir().map(folo_api::json::path);
    let features = features(&st);
    Ok(Json(SystemResponse {
        system: SystemInfo { cpu, memory, gpus },
        data_dir,
        features,
        capabilities,
        gpu_pool,
    }))
}

/// The slow parts of the first `/api/system`, side by side, each a blocking task of its own: creating the GPU device
/// (`capabilities`), enumerating the adapters, and the CPU counters' set-up and first interval. Run once per server
/// ([`AppState`]'s `system_ready`); requests that arrive meanwhile wait on the same run without holding a blocking
/// thread. A part that panicked fails this request and leaves the preparation to the next.
async fn prepare_system(st: &AppState) -> Result<()> {
    let (store, adapters_store, probe) = (st.store.clone(), st.store.clone(), st.probe.clone());
    let device = tokio::task::spawn_blocking(move || drop(store.capabilities()));
    let adapters = tokio::task::spawn_blocking(move || adapters_store.prepare_gpus());
    let counters = tokio::task::spawn_blocking(move || probe.warm_up());
    let (device, adapters, counters) = tokio::join!(device, adapters, counters);
    device.and(adapters).and(counters).map_err(|e| Error::internal(format!("task failed: {e}")))
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::*;

    /// A burst of first `/api/system` calls shares one preparation, and nothing waits on a blocking thread for
    /// another blocking job: with a single blocking thread every call still answers. Later calls prepare nothing.
    #[test]
    fn first_calls_share_one_preparation_and_hold_no_blocking_thread_while_waiting() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let st = AppState {
            store: Arc::new(crate::MockStore::new()),
            probe: Arc::new(crate::system::SystemProbe::default()),
            system_ready: Arc::default(),
            shutdown: Arc::new(tokio::sync::watch::channel(false).0),
            web_wasm: false,
        };
        rt.block_on(async {
            let burst = (0..16).map(|_| tokio::spawn(system(State(st.clone()))));
            let answers = tokio::time::timeout(std::time::Duration::from_secs(10), async {
                let mut answers = Vec::new();
                for call in burst {
                    answers.push(call.await.unwrap());
                }
                answers
            })
            .await
            .expect("every call answers");
            for a in answers {
                let Json(r) = a.unwrap_or_else(|e| panic!("{e:?}"));
                assert!(r.system.cpu.threads >= 1 && r.capabilities.is_some());
            }
            assert_eq!(st.probe.warm_ups.load(Ordering::Relaxed), 1);
            for _ in 0..3 {
                assert!(system(State(st.clone())).await.is_ok());
            }
            assert_eq!(st.probe.warm_ups.load(Ordering::Relaxed), 1, "prepared once");
        });
    }
}
